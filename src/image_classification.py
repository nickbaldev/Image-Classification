import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, tree, ensemble, naive_bayes
from sklearn.linear_model import SGDClassifier

# CIFAR-10 Dataset
cifar_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)
cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform)

# MNIST Dataset
mnist_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)

# Convert to numpy arrays (flatten images)
def flatten_to_numpy(dataset, max_samples): 
    '''helper function to convert the dataset from tensor to numpy array'''
    images, labels = [], []
    for i in range(min(len(dataset), max_samples)):
        img, label = dataset[i]
        images.append(img.numpy().flatten())
        labels.append(label)
    return np.array(images), np.array(labels)

X_cifar_train, y_cifar_train = flatten_to_numpy(cifar_train, 5000) # only use 5k for train and 1k for test to speed up runtime
X_cifar_test, y_cifar_test = flatten_to_numpy(cifar_test, 1000)
X_mnist_train, y_mnist_train = flatten_to_numpy(mnist_train, 5000)
X_mnist_test, y_mnist_test = flatten_to_numpy(mnist_test, 1000)

# Basic ML Models
knn_classifier = neighbors.KNeighborsClassifier(n_neighbors=3, metric='euclidean')
dt_classifier = tree.DecisionTreeClassifier(max_depth=10, random_state=1)
lr_classifier = SGDClassifier(loss='log_loss', max_iter=100, shuffle=False, tol=None, penalty=None, learning_rate='constant', eta0=0.1, random_state=1)
rf_classifier = ensemble.RandomForestClassifier(n_estimators=50, max_depth=10, random_state=1)
nb_classifier = naive_bayes.GaussianNB()

classifiers = [
    ("K-Nearest Neighbors", knn_classifier),
    ("Decision Tree", dt_classifier),
    ("Logistic Regression", lr_classifier),
    ("Random Forest", rf_classifier),
    ("Naive Bayes", nb_classifier)
]

# Test basic classifiers
print("Basic ML Model Performance")
print("\nCIFAR-10\n" + "-"*30)
for name, classifier in classifiers:
    classifier.fit(X_cifar_train, y_cifar_train)
    acc = classifier.score(X_cifar_test, y_cifar_test)
    print(f"{name}: {acc * 100:.2f}%")

print("\nMNIST\n" + "-"*30)
for name, classifier in classifiers:
    classifier.fit(X_mnist_train, y_mnist_train)
    acc = classifier.score(X_mnist_test, y_mnist_test)
    print(f"{name}: {acc * 100:.2f}%")

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

''' Neural Network Implementation'''
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Train the NN
def train_NN(model, train_loader, epochs=10):
    '''function to train the NN'''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# Evaluate NN
def check_accuracy(model, test_loader):
    '''function that computes the accuracy of the model'''
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def convert(X_train, y_train, X_test, y_test):
    '''helper function to convert the data back into tensor format'''
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader

# Train/test on CIFAR-10
print("Neural Network Performance")
print("\nCIFAR-10\n" + "-"*30)
cifar_train_loader, cifar_test_loader = convert(X_cifar_train, y_cifar_train, X_cifar_test, y_cifar_test)
cifar_nn = NeuralNetwork(input_size=3072, num_classes=10)  #32*32*3=3072; 10 classes
train_NN(cifar_nn, cifar_train_loader, epochs=10)
cifar_acc = check_accuracy(cifar_nn, cifar_test_loader)
print(f"NN Accuracy: {cifar_acc:.2f}%")

# Train/test on MNIST
print("\nMNIST\n" + "-"*30)
mnist_train_loader, mnist_test_loader = convert(X_mnist_train, y_mnist_train, X_mnist_test, y_mnist_test)
mnist_nn = NeuralNetwork(input_size=784, num_classes=10)  #28*28*1=784; 10 classes
train_NN(mnist_nn, mnist_train_loader, epochs=10)
mnist_acc = check_accuracy(mnist_nn, mnist_test_loader)
print(f"NN Accuracy: {mnist_acc:.2f}%")

# Try varying aspects of the network such as the number of layers, the activation function, the number of neurons per layer

# CIFAR-10

print("\nTesting different hyperparameters on CIFAR-10\n" + "-"*30)

# Current (2 hidden layers (128, 64) ReLU
print("\n2 hidden layers (128,64) ReLU:")
print(f"Accuracy: {cifar_acc:.2f}%")

# 1. fewer layers (only 1 hidden layer instead of 2)
print("\n1 hidden layer (64) ReLU:")
class NN1(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)  # Only one hidden layer
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

nn1 = NN1(3072, 10)
train_NN(nn1, cifar_train_loader, epochs=5)
nn1_acc = check_accuracy(nn1, cifar_test_loader)
print(f"Accuracy: {nn1_acc:.2f}%")

# 2. more layers (3 hidden layers)
print("\n3 hidden layers (256, 128, 64) ReLU:")
class NN2(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

nn2 = NN2(3072, 10)
train_NN(nn2, cifar_train_loader, epochs=5)
nn2_acc = check_accuracy(nn2, cifar_test_loader)
print(f"Accuracy: {nn2_acc:.2f}%")

# 3. different activation function (sigmoid). 2 hidden layers
print("\n2 hidden layers (128, 64) Sigmoid:")
class NN3(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

nn3 = NN3(3072, 10)
train_NN(nn3, cifar_train_loader, epochs=5)
nn3_acc = check_accuracy(nn3, cifar_test_loader)
print(f"Accuracy: {nn3_acc:.2f}%")

# 4. fewer neurons per layer; (64, 32) instead of (128, 64)
print("\n2 hidden layers (64, 32) ReLU:")
class NN4(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)  
        self.fc2 = nn.Linear(64, 32)           
        self.fc3 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

nn4 = NN4(3072, 10)
train_NN(nn4, cifar_train_loader, epochs=5)
nn4_acc = check_accuracy(nn4, cifar_test_loader)
print(f"Accuracy: {nn4_acc:.2f}%")

# 5. more neurons per layer; (512, 256) instead of (128, 64)
print("\n2 hidden layers (512, 256) ReLU:")
class NN5(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)  
        self.fc2 = nn.Linear(512, 256)       
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

nn5 = NN5(3072, 10)
train_NN(nn5, cifar_train_loader, epochs=5)
nn5_acc = check_accuracy(nn5, cifar_test_loader)
print(f"Accuracy: {nn5_acc:.2f}%")

# Side by side
print("\nComparison: CIFAR-10\n" + "-"*30)
print(f"1. Current (128, 64), ReLU:     {cifar_acc:.2f}%")
print(f"2. Fewer layers                 {nn1_acc:.2f}%")
print(f"3. More layers:                 {nn2_acc:.2f}%")
print(f"4. Sigmoid Activation Function: {nn3_acc:.2f}%")
print(f"5. Less neuroms (64, 32):       {nn4_acc:.2f}%")
print(f"6. More neurons (512, 256):     {nn5_acc:.2f}%")
print()

# MNIST
nn1_mnist = NN1(784, 10)
train_NN(nn1_mnist, mnist_train_loader, epochs=5)
nn1_mnist_acc = check_accuracy(nn1_mnist, mnist_test_loader)

nn2_mnist = NN2(784, 10)  
train_NN(nn2_mnist, mnist_train_loader, epochs=5)
nn2_mnist_acc = check_accuracy(nn2_mnist, mnist_test_loader)

nn3_mnist = NN3(784, 10)  
train_NN(nn3_mnist, mnist_train_loader, epochs=5)
nn3_mnist_acc = check_accuracy(nn3_mnist, mnist_test_loader)

nn4_mnist = NN4(784, 10) 
train_NN(nn4_mnist, mnist_train_loader, epochs=5)
nn4_mnist_acc = check_accuracy(nn4_mnist, mnist_test_loader)

nn5_mnist = NN5(784, 10)
train_NN(nn5_mnist, mnist_train_loader, epochs=5)
nn5_mnist_acc = check_accuracy(nn5_mnist, mnist_test_loader)

print("\nComparison: MNIST\n" + "-"*30)
print(f"1. Current (128, 64), ReLU:     {mnist_acc:.2f}%")
print(f"2. Fewer layers                 {nn1_mnist_acc:.2f}%")
print(f"3. More layers:                 {nn2_mnist_acc:.2f}%")
print(f"4. Sigmoid Activation Function: {nn3_mnist_acc:.2f}%")
print(f"5. Less neuroms (64, 32):       {nn4_mnist_acc:.2f}%")
print(f"6. More neurons (512, 256):     {nn5_mnist_acc:.2f}%")

import torch.nn as nn
import torch.optim as optim

''' Convolutional Neural Network Implementation'''
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8 if in_channels==3 else 64 * 7 * 7, 128)  #CIFAR has 3 in_channels while MNIST has only 1
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Train the CNN
def train_CNN(model, train_loader, epochs=10, dataset="CIFAR-10"):
    '''function to train the CNN'''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model

def check_accuracy_CNN(model, test_loader):
    '''function to compute accuracy of CNN'''
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Test CNN

# CIFAR-10
print("Convolutional Neural Network Performance")
print("\nCIFAR-10\n" + "-"*30)
cifar_train_loader_cnn = DataLoader(cifar_train, batch_size=64, shuffle=True)
cifar_test_loader_cnn = DataLoader(cifar_test, batch_size=64, shuffle=False)
cifar_cnn = ConvolutionalNeuralNetwork(num_classes=10, in_channels=3)
cifar_cnn = train_CNN(cifar_cnn, cifar_train_loader_cnn, epochs=10, dataset="CIFAR-10")
cifar_cnn_acc = check_accuracy_CNN(cifar_cnn, cifar_test_loader_cnn)
print(f"CNN Accuracy: {cifar_cnn_acc:.2f}% +({cifar_cnn_acc-cifar_acc:.2f}%)")

# MNIST
print("\nMNIST\n" + "-"*30)
mnist_train_loader_cnn = DataLoader(mnist_train, batch_size=64, shuffle=True)
mnist_test_loader_cnn = DataLoader(mnist_test, batch_size=64, shuffle=False)
mnist_cnn = ConvolutionalNeuralNetwork(num_classes=10, in_channels=1)
mnist_cnn = train_CNN(mnist_cnn, mnist_train_loader_cnn, epochs=10, dataset="MNIST")
mnist_cnn_acc = check_accuracy_CNN(mnist_cnn, mnist_test_loader_cnn)
print(f"CNN Accuracy: {mnist_cnn_acc:.2f}% +({mnist_cnn_acc-mnist_acc:.2f}%)")

# Figures, graphs, images, tables, etc
# Used AI to help make plots look nice

cifar_img1 = cifar_train[0][0]
cifar_img2 = cifar_train[1][0]
mnist_img1, mnist_label1 = mnist_train[0]
mnist_img2, mnist_label2 = mnist_train[1]

# CIFAR-10 Image Examples
plt.figure(figsize=(4, 4))
plt.imshow(cifar_img1.permute(1, 2, 0).numpy() * 0.5 + 0.5)
plt.title(f'CIFAR-10: Class (Frog)')
plt.axis('off')
plt.savefig('frog.png', dpi=150, bbox_inches='tight')
plt.show()
plt.figure(figsize=(4, 4))
plt.imshow(cifar_img2.permute(1, 2, 0).numpy() * 0.5 + 0.5)
plt.title(f'CIFAR-10: Class (Truck)')
plt.axis('off')
plt.savefig('truck.png', dpi=150, bbox_inches='tight')
plt.show()

# MNIST Image Examples
plt.figure(figsize=(4, 4))
plt.imshow(mnist_img1.squeeze().numpy(), cmap='gray')
plt.title(f'MNIST: Class ({mnist_label1})')
plt.axis('off')
plt.savefig('5.png', dpi=150, bbox_inches='tight')
plt.show()
plt.figure(figsize=(4, 4))
plt.imshow(mnist_img2.squeeze().numpy(), cmap='gray')
plt.title(f'MNIST: Class ({mnist_label2})')
plt.axis('off')
plt.savefig('0.png', dpi=150, bbox_inches='tight')
plt.show()

# Basic ML Performance Bar Graph
plt.figure(figsize=(8, 5))
basic_ml_cifar = [0.26, 0.23, 0.24, 0.36, 0.29]
basic_ml_mnist = [0.91, 0.76, 0.82, 0.91, 0.56]
models = ['KNN', 'DT', 'LR', 'RF', 'NB']
x = np.arange(len(models))
bars1 = plt.bar(x - 0.2, basic_ml_cifar, width=0.4, label='CIFAR-10', color='blue')
bars2 = plt.bar(x + 0.2, basic_ml_mnist, width=0.4, label='MNIST', color='red')
for bar in bars1:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.2f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.2f}', ha='center', va='bottom', fontsize=9)
plt.title('Basic ML Performance')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(x, models)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('bargraph1.png', dpi=150, bbox_inches='tight')
plt.show()

# Hyperparamters Bar Graph
plt.figure(figsize=(8, 5))
nn_arch_names = ['2 Layers', '1 Layer', '3 Layers', 'Sigmoid', 'Fewer Neurons', 'More Neurons']
nn_cifar_acc = [42.60, 40.80, 42.10, 34.10, 39.80, 40.50]
nn_mnist_acc = [89.80, 88.30, 89.30, 87.30, 87.10, 91.20]
x_nn = np.arange(len(nn_arch_names))
bars1 = plt.bar(x_nn - 0.2, nn_cifar_acc, width=0.4, label='CIFAR-10', color='skyblue')
bars2 = plt.bar(x_nn + 0.2, nn_mnist_acc, width=0.4, label='MNIST', color='crimson')
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.3, f'{height:.1f}%', 
             ha='center', va='bottom', fontsize=8)

for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.3, f'{height:.1f}%', 
             ha='center', va='bottom', fontsize=8)

plt.title('Hyperparameter Tuning Performances')
plt.xlabel('Architecture')
plt.ylabel('Accuracy (%)')
plt.xticks(x_nn, nn_arch_names, rotation=15)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('bargraph3.png', dpi=150, bbox_inches='tight')
plt.show()

# NN and CNN Performance Bar Graph
plt.figure(figsize=(8, 5))
model_types = ['NN', 'CNN']
cifar_final = [.43, .71]
mnist_final = [.91, .99]
x_final = np.arange(len(model_types))

bars1 = plt.bar(x_final - 0.2, cifar_final, width=0.4, label='CIFAR-10', color='orange')
bars2 = plt.bar(x_final + 0.2, mnist_final, width=0.4, label='MNIST', color='purple')
for bar in bars1:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.2f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.2f}', ha='center', va='bottom', fontsize=9)
plt.title('Neural Network Performance')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(x_final, model_types)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('bargraph2.png', dpi=150, bbox_inches='tight')
plt.show()
