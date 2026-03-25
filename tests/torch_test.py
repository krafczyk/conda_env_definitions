import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torch import optim


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),  # 28 -> 26
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0), # 26 -> 24
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                           # 24 -> 12
        )

        self.classifier = nn.Linear(32 * 12 * 12, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)


def evaluate(model, loader, loss_func, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_func(logits, labels)

            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train(num_epochs, model, loaders, loss_func, optimizer, device):
    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        running_samples = 0

        for i, (images, labels) in enumerate(loaders["train"]):
            images = images.to(device, memory_format=torch.channels_last)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = loss_func(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_samples += images.size(0)

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Step [{i+1}/{len(loaders['train'])}] "
                    f"Batch Loss: {loss.item():.4f}"
                )

        train_loss = running_loss / running_samples
        test_loss, test_acc = evaluate(model, loaders["test"], loss_func, device)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Test Loss: {test_loss:.4f} "
            f"Test Acc: {test_acc:.4%}"
        )


if __name__ == "__main__":
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_data = datasets.MNIST(
        root="data",
        train=True,
        transform=transform,
        download=True,
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        transform=transform,
        download=True,
    )

    loaders = {
        "train": torch.utils.data.DataLoader(
            train_data,
            batch_size=128,
            shuffle=True,
            num_workers=1,
            pin_memory=(device.type == "cuda"),
        ),
        "test": torch.utils.data.DataLoader(
            test_data,
            batch_size=256,
            shuffle=False,
            num_workers=1,
            pin_memory=(device.type == "cuda"),
        ),
    }

    model = CNN().to(device=device, memory_format=torch.channels_last)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train(5, model, loaders, loss_func, optimizer, device)

    print("Training finished")
