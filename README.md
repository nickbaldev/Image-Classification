# Image Classification: From Classical Models to CNNs

A comparative study of image classification models using **MNIST** and **CIFAR-10**. The project evaluates classical machine-learning classifiers, fully connected neural networks, and a convolutional neural network (CNN) implemented with scikit-learn and PyTorch.

## What this project asks

- How well do classical classifiers perform when image structure is flattened into feature vectors?
- How much do fully connected neural networks improve on those baselines?
- How do neural-network depth, width, and activation functions affect performance?
- How much does preserving spatial structure with convolutions help?

## Models

**Classical ML**
- K-Nearest Neighbors
- Decision Tree
- Logistic Regression (SGD)
- Random Forest
- Gaussian Naive Bayes

**Neural networks**
- Fully connected baseline: 128 → 64 hidden units with ReLU
- Architecture experiments varying depth, width, and activation
- CNN with two convolution/max-pooling stages followed by a fully connected layer

## Datasets

The experiments use the standard MNIST and CIFAR-10 datasets downloaded automatically through `torchvision`.

The original study used 5,000 training examples and 1,000 test examples for the classical/fully connected comparisons to keep runtime manageable. The CNN experiment used the full torchvision train/test datasets.

## Original experimental results

These values are preserved from the original project and are included here as historical experiment results rather than claims of a newly reproduced benchmark.

| Model | CIFAR-10 | MNIST |
| --- | ---: | ---: |
| KNN | 26% | 91% |
| Decision Tree | 23% | 76–77% |
| Logistic Regression | 24% | 82% |
| Random Forest | 36% | 91% |
| Naive Bayes | 29% | 56% |
| Fully connected NN | 43% | 90% |
| CNN | **71%** | **99%** |

The main result was the large improvement from the fully connected network to the CNN on CIFAR-10, illustrating the value of preserving spatial structure in image data.

## Repository structure

```text
.
├── src/
│   └── image_classification.py
├── requirements.txt
└── README.md
```

## Running the project

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/image_classification.py
```

The script downloads MNIST/CIFAR-10 through torchvision when needed and runs the comparison experiments.

## What I learned

The experiments reinforced an important distinction in image modeling: flattening an image into independent pixel features removes spatial relationships that are useful for recognizing visual patterns. Classical models and fully connected networks can still learn useful representations, especially on MNIST, but the CNN's ability to operate directly on image structure produced the strongest results in the original experiments.

## Background

Originally developed as a comparative machine-learning project at Swarthmore College. The current repository is a cleaned, recruiting-facing version of the original work; the experimental results above are retained from the original project.
