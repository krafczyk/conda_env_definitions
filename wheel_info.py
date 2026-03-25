#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from packaging.version import InvalidVersion, Version
from packaging.utils import parse_wheel_filename

import html.parser


class WheelInfoError(Exception):
    pass


def normalize_project_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


class SimpleIndexHTMLParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.files = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return

        attrs_dict = dict(attrs)
        href = attrs_dict.get("href")
        if not href:
            return

        self.files.append(
            {
                "filename": urllib.parse.unquote(href.split("/")[-1].split("#")[0]),
                "url": href,
                "requires-python": attrs_dict.get("data-requires-python"),
                "yanked": "data-yanked" in attrs_dict,
            }
        )


def _make_simple_project_url(base_url: str, project_name: str) -> str:
    normalized = normalize_project_name(project_name)
    base_url = base_url.rstrip("/") + "/"
    return urllib.parse.urljoin(base_url, urllib.parse.quote(normalized) + "/")


def fetch_simple_index(base_url: str, project_name: str) -> dict:
    """
    Fetch one package index page from a PEP 503/691-style simple index.

    Tries JSON first, then falls back to HTML parsing.
    Returns a dict with at least:
        {
            "files": [...],
            "project": <normalized name>,
            "index_url": <base_url>,
        }
    """
    url = _make_simple_project_url(base_url, project_name)

    # Try JSON simple API first
    json_req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.pypi.simple.v1+json",
            "User-Agent": "wheel_info.py",
        },
    )

    try:
        with urllib.request.urlopen(json_req) as resp:
            data = json.load(resp)
            data["index_url"] = base_url
            return data
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise WheelInfoError(
                f"Project not found on index {base_url!r}: {project_name!r}"
            ) from e
        # Other HTTP errors: fall through to HTML attempt
    except (urllib.error.URLError, json.JSONDecodeError):
        # Fall through to HTML attempt
        pass

    # Fall back to HTML simple index
    html_req = urllib.request.Request(
        url,
        headers={
            "Accept": "text/html,application/xhtml+xml",
            "User-Agent": "wheel_info.py",
        },
    )

    try:
        with urllib.request.urlopen(html_req) as resp:
            html_text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if e.code == (403, 404):
            raise WheelInfoError(
                f"Project not found or inaccessible on index {base_url!r}: {project_name!r}"
            ) from e
        raise WheelInfoError(
            f"Failed to fetch index page for {project_name!r} from {base_url!r}: {e}"
        ) from e
    except urllib.error.URLError as e:
        raise WheelInfoError(
            f"Failed to fetch index page for {project_name!r} from {base_url!r}: {e}"
        ) from e

    parser = SimpleIndexHTMLParser()
    parser.feed(html_text)

    # Resolve relative URLs against the page URL
    for file_info in parser.files:
        file_info["url"] = urllib.parse.urljoin(url, file_info["url"])

    return {
        "project": normalize_project_name(project_name),
        "files": parser.files,
        "index_url": base_url,
    }


def fetch_package_indexes(
    project_name: str,
    extra_index_urls: list[str] | None = None,
    primary_index_url: str = "https://pypi.org/simple",
) -> dict:
    """
    Fetch package listings from PyPI plus any extra indexes and merge them.
    Deduplicates by (filename, url).
    """
    index_urls = [primary_index_url]
    if extra_index_urls:
        index_urls.extend(extra_index_urls)

    merged_files = []
    seen = set()
    errors = []

    for index_url in index_urls:
        try:
            data = fetch_simple_index(index_url, project_name)
        except WheelInfoError as e:
            errors.append(str(e))
            continue

        for file_info in data.get("files", []):
            key = (file_info.get("filename", ""), file_info.get("url", ""))
            if key in seen:
                continue
            seen.add(key)

            item = dict(file_info)
            item["index_url"] = index_url
            merged_files.append(item)

    if not merged_files:
        if errors:
            raise WheelInfoError("; ".join(errors))
        raise WheelInfoError(f"No files found for project {project_name!r}")

    return {
        "project": normalize_project_name(project_name),
        "files": merged_files,
    }


@dataclass(frozen=True)
class ParsedQuery:
    name: str
    specifier: str = ""


_QUERY_RE = re.compile(
    r"^\s*([A-Za-z0-9_.-]+)\s*(.*)\s*$"
)

_WHEEL_RE = re.compile(
    r"^(?P<name>.+)-(?P<version>.+?)-(?P<pyver>.+?)-(?P<abi>.+?)-(?P<plat>.+?)\.whl$"
)


def parse_query(query: str) -> ParsedQuery:
    match = _QUERY_RE.match(query)
    if match is None:
        raise WheelInfoError(f"Could not parse query: {query!r}")

    name = match.group(1)
    specifier = match.group(2).strip()
    if not name:
        raise WheelInfoError(f"Missing package name in query: {query!r}")

    return ParsedQuery(name=name, specifier=specifier)


def normalize_project_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def sort_key(item: dict):
    build = item.get("build")
    # build is None or a build tuple from packaging
    return (item["version_obj"], build is None, build or (), item["filename"])


def version_matches(version: str, specifier: str) -> bool:
    if not specifier:
        return True

    # Prefer packaging if available.
    try:
        from packaging.specifiers import SpecifierSet
    except ImportError:
        # Fallback: support only exact ==
        if specifier.startswith("=="):
            return version == specifier[2:].strip()
        raise WheelInfoError(
            "Non-exact version specifiers require the 'packaging' package.\n"
            "Install it with: python -m pip install packaging"
        )

    return version in SpecifierSet(specifier)


def extract_wheel_info(filename: str) -> tuple[str, object, object]:
    filename = urllib.parse.unquote(filename)
    try:
        name, version, build, tags = parse_wheel_filename(filename)
    except Exception as e:
        raise WheelInfoError(f"Could not parse wheel filename: {filename}") from e
    return name, version, build


def list_matching_wheels(query: str, extra_index_urls: list[str]) -> list[dict]:
    parsed = parse_query(query)

    data = fetch_package_indexes(
        parsed.name,
        extra_index_urls=extra_index_urls,
    )

    files = data.get("files", [])
    result: list[dict] = []

    for file_info in files:
        filename = file_info.get("filename", "")
        if not filename.endswith(".whl"):
            continue

        try:
            _, version, build = extract_wheel_info(filename)
        except WheelInfoError:
            continue

        if not version_matches(version, parsed.specifier):
            continue

        result.append(
            {
                "filename": filename,
                "version": str(version),
                "version_obj": version,
                "build": build,
                "url": file_info.get("url", ""),
                "requires_python": file_info.get("requires-python"),
                "yanked": bool(file_info.get("yanked", False)),
            }
        )

    result.sort(key=sort_key)
    return result


def download_single_wheel(
    package_spec: str,
    dest_dir: Path,
    extra_index_urls: list[str] | None = None,
) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--no-deps",
        "--only-binary=:all:",
        "-d",
        str(dest_dir),
    ]

    if extra_index_urls:
        for url in extra_index_urls:
            cmd.extend(["--extra-index-url", url])

    cmd.append(package_spec)

    subprocess.run(cmd, check=True)

    wheels = sorted(dest_dir.glob("*.whl"))
    if not wheels:
        raise WheelInfoError(f"No wheel downloaded for {package_spec!r}")
    if len(wheels) > 1:
        raise WheelInfoError(
            f"Expected exactly one wheel for {package_spec!r}, found: "
            f"{', '.join(w.name for w in wheels)}"
        )

    return wheels[0]


def read_wheel_metadata(wheel_path: Path) -> str:
    with zipfile.ZipFile(wheel_path) as zf:
        for name in zf.namelist():
            if name.endswith(".dist-info/METADATA"):
                return zf.read(name).decode("utf-8")
    raise WheelInfoError(f"No METADATA file found in wheel {wheel_path.name!r}")


def filter_requires_dist(metadata: str) -> str:
    lines = []
    for line in metadata.splitlines():
        if line.startswith("Requires-Dist:"):
            lines.append(line)
    return "\n".join(lines) + ("\n" if lines else "")


def cmd_metadata(
    package_spec: str,
    extra_index_urls: list[str] | None = None,
    requires_only: bool = False,
) -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        wheel_path = download_single_wheel(
            package_spec,
            Path(tmpdir),
            extra_index_urls=extra_index_urls)
        metadata = read_wheel_metadata(wheel_path)
        if requires_only:
            metadata = filter_requires_dist(metadata)
        print(metadata, end="")
    return 0


def cmd_list(query: str, show_urls: bool, extra_index_urls: list[str]) -> int:
    matches = list_matching_wheels(query, extra_index_urls)
    if not matches:
        print(f"No matching wheels found for {query!r}", file=sys.stderr)
        return 1

    for item in matches:
        line = item["filename"]
        extras: list[str] = []

        if item["requires_python"]:
            extras.append(f"requires_python={item['requires_python']}")
        if item["yanked"]:
            extras.append("yanked=true")
        if show_urls and item["url"]:
            extras.append(item["url"])

        if extras:
            line += "    " + "    ".join(extras)

        print(line)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect PyPI wheels and wheel metadata."
    )
    parser.add_argument(
        "--extra-index-url",
        action="append",
        default=[],
        help="Additional simple package index URL. Can be passed multiple times.",
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    metadata_parser = subparsers.add_parser(
        "metadata",
        help="Download a single wheel for a package spec and print its METADATA.",
    )
    metadata_parser.add_argument(
        "package_spec",
        help="Example: tensorflow==2.19.0",
    )
    metadata_parser.add_argument(
        "--requires-only",
        action="store_true",
        help="Print only Requires-Dist lines from METADATA.",
    )

    list_parser = subparsers.add_parser(
        "list",
        help="List available wheel files on PyPI matching a package query.",
    )
    list_parser.add_argument(
        "query",
        help="Examples: tensorflow, tensorflow==2.19.0, tensorflow>=2.18,<2.21",
    )
    list_parser.add_argument(
        "--show-urls",
        action="store_true",
        help="Also print the wheel download URLs.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.mode == "metadata":
            return cmd_metadata(
                args.package_spec,
                extra_index_urls=args.extra_index_url,
                requires_only=args.requires_only)
        if args.mode == "list":
            return cmd_list(
                args.query,
                args.show_urls,
                args.extra_index_url)
        raise WheelInfoError(f"Unknown mode: {args.mode}")
    except subprocess.CalledProcessError as e:
        print(f"pip download failed with exit code {e.returncode}", file=sys.stderr)
        return 1
    except WheelInfoError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
