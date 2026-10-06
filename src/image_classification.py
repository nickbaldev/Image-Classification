"""Compare classical ML, fully connected networks, and CNNs on MNIST/CIFAR-10."""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn import ensemble, naive_bayes, neighbors, tree
from sklearn.linear_model import SGDClassifier
from torch.utils.data import DataLoader, TensorDataset

SEED = 1


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_datasets():
    cifar_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)]
    )
    mnist_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    cifar_train = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=cifar_transform
    )
    cifar_test = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=cifar_transform
    )
    mnist_train = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=mnist_transform
    )
    mnist_test = torchvision.datasets.MNIST(
        "./data", train=False, download=True, transform=mnist_transform
    )
    return cifar_train, cifar_test, mnist_train, mnist_test


def flatten_to_numpy(dataset, max_samples: int):
    images, labels = [], []
    for i in range(min(len(dataset), max_samples)):
        image, label = dataset[i]
        images.append(image.numpy().reshape(-1))
        labels.append(label)
    return np.asarray(images), np.asarray(labels)


def classical_models():
    return [
        ("KNN", neighbors.KNeighborsClassifier(n_neighbors=3, metric="euclidean")),
        ("Decision Tree", tree.DecisionTreeClassifier(max_depth=10, random_state=SEED)),
        (
            "Logistic Regression",
            SGDClassifier(
                loss="log_loss",
                max_iter=100,
                shuffle=False,
                tol=None,
                penalty=None,
                learning_rate="constant",
                eta0=0.1,
                random_state=SEED,
            ),
        ),
        (
            "Random Forest",
            ensemble.RandomForestClassifier(
                n_estimators=50, max_depth=10, random_state=SEED
            ),
        ),
        ("Naive Bayes", naive_bayes.GaussianNB()),
    ]


def run_classical(X_train, y_train, X_test, y_test):
    results = {}
    for name, model in classical_models():
        model.fit(X_train, y_train)
        results[name] = 100.0 * model.score(X_test, y_test)
    return results


class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes=(128, 64), activation="relu"):
        super().__init__()
        layers = []
        previous = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous, hidden_size))
            layers.append(nn.ReLU() if activation == "relu" else nn.Sigmoid())
            previous = hidden_size
        layers.append(nn.Linear(previous, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ConvolutionalNetwork(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        spatial = 8 if in_channels == 3 else 7
        self.fc1 = nn.Linear(64 * spatial * spatial, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        return self.fc2(F.relu(self.fc1(x)))


def tensor_loaders(X_train, y_train, X_test, y_test):
    train = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    return (
        DataLoader(train, batch_size=64, shuffle=True),
        DataLoader(test, batch_size=64, shuffle=False),
    )


def train(model, loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for _ in range(epochs):
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct = total = 0
    for inputs, labels in loader:
        predictions = model(inputs).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def run_fc_experiment(X_train, y_train, X_test, y_test, hidden=(128, 64), activation="relu", epochs=10):
    train_loader, test_loader = tensor_loaders(X_train, y_train, X_test, y_test)
    model = FullyConnectedNetwork(X_train.shape[1], hidden, activation)
    train(model, train_loader, epochs)
    return accuracy(model, test_loader)


def run_cnn(dataset_train, dataset_test, channels):
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False)
    model = ConvolutionalNetwork(channels)
    train(model, train_loader)
    return accuracy(model, test_loader)


def print_results(title, results):
    print(f"\n{title}\n" + "-" * len(title))
    for name, value in results.items():
        print(f"{name:24s} {value:6.2f}%")


def main(sample_size=5000):
    set_seed()
    cifar_train, cifar_test, mnist_train, mnist_test = load_datasets()

    Xc_train, yc_train = flatten_to_numpy(cifar_train, sample_size)
    Xc_test, yc_test = flatten_to_numpy(cifar_test, min(1000, len(cifar_test)))
    Xm_train, ym_train = flatten_to_numpy(mnist_train, sample_size)
    Xm_test, ym_test = flatten_to_numpy(mnist_test, min(1000, len(mnist_test)))

    print_results("CIFAR-10 — classical models", run_classical(Xc_train, yc_train, Xc_test, yc_test))
    print_results("MNIST — classical models", run_classical(Xm_train, ym_train, Xm_test, ym_test))

    cifar_fc = run_fc_experiment(Xc_train, yc_train, Xc_test, yc_test)
    mnist_fc = run_fc_experiment(Xm_train, ym_train, Xm_test, ym_test)
    print_results("Fully connected neural network", {"CIFAR-10": cifar_fc, "MNIST": mnist_fc})

    architectures = {
        "1 layer / 64 ReLU": ((64,), "relu"),
        "3 layers / 256-128-64 ReLU": ((256, 128, 64), "relu"),
        "2 layers / 128-64 Sigmoid": ((128, 64), "sigmoid"),
        "2 layers / 64-32 ReLU": ((64, 32), "relu"),
        "2 layers / 512-256 ReLU": ((512, 256), "relu"),
    }
    tuning = {"Baseline 128-64 ReLU": cifar_fc}
    for name, (hidden, activation) in architectures.items():
        tuning[name] = run_fc_experiment(
            Xc_train, yc_train, Xc_test, yc_test, hidden, activation, epochs=5
        )
    print_results("CIFAR-10 — architecture experiments", tuning)

    cifar_cnn = run_cnn(cifar_train, cifar_test, 3)
    mnist_cnn = run_cnn(mnist_train, mnist_test, 1)
    print_results("CNN", {"CIFAR-10": cifar_cnn, "MNIST": mnist_cnn})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Training examples per dataset for classical/FC experiments.",
    )
    args = parser.parse_args()
    main(args.sample_size)
