# Used to fix pip cuda installations so all libraries can find 
# needed tools.

pushd "$(dirname "$(python -c 'print(__import__("tensorflow").__file__)')")"
ln -svf ../nvidia/*/lib/*.so* .
popd

ptxas_path=$(find "$(python - <<'PY'
import importlib.metadata as md
print(md.distribution("nvidia-cuda-nvcc-cu12").locate_file(""))
PY
)" -name ptxas -type f -print -quit)

ln -sf "$ptxas_path" "$CONDA_PREFIX/bin/ptxas"
which ptxas
ptxas --version
