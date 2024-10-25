# Digit Recognition on the MNIST Dataset

## Overview

This project, completed as part of the course **MITx 6.86x: Machine Learning with Python - From Linear Models to Deep Learning**, aims to classify images of handwritten digits using multiple machine learning techniques. The MNIST dataset, widely used for benchmarking image processing systems, provides 60,000 training and 10,000 testing images of digits ranging from 0 to 9, each standardized to a 28 × 28-pixel binary image.

## Project Structure

The project is divided into two main parts, each exploring different classification techniques.

### Part 1: Traditional Machine Learning Models

In this section, various machine learning models are implemented to classify handwritten digit images. The methods include:

- **Linear Regression**: Implemented in `part1/linear_regression.py`
- **Support Vector Machine (SVM)**: Implemented in `part1/svm.py`
- **Multinomial Regression**: Implemented in `part1/softmax.py`
- **Dimensionality Reduction (PCA)**: Implemented in `part1/features.py`
- **Kernel Methods**: Polynomial and Gaussian RBF kernels in `part1/kernel.py`

The main script (`part1/main.py`) brings together these methods for training and testing on the MNIST dataset.

To begin, simply run the main script:

```bash
python main.py
```

This file provides code that reads the MNIST data via the `get_MNIST_data` function in `utils.py`. It returns data in the form of Numpy arrays:

- `train_x`: Training data matrix, where each row represents a flattened 28 × 28-pixel image.
- `train_y`: Labels for each training image (digit between 0-9).
- `test_x`: Testing data matrix, similarly formatted.
- `test_y`: Labels for testing data, used only for accuracy evaluation.

To explore the data, use the `plot_images` function to visualize the first 20 images from the training set.

### Part 2: Neural Network Models

In this section, a neural network model is implemented to classify MNIST digits. The implementations evolve in complexity, starting with a basic neural network from scratch and moving to convolutional neural networks (CNNs) using PyTorch.

- **Neural Network from Scratch**: Implemented in `part2-nn/neural_nets.py`
- **Fully Connected Network (PyTorch)**: Implemented in `part2-mnist/nnet_fc.py`
- **Convolutional Neural Network (PyTorch)**: Implemented in `part2-mnist/nnet_conv.py`
- **Enhanced Models for More Complex Digits**: Implemented in `part2-twodigit/mlp.py` and `part2-twodigit/conv.py` for classifying a more complex version of the MNIST dataset.

## Setup and Installation

### Prerequisites

- Python 3.8 or above
- Libraries:
  - `numpy` (numerical computations)
  - `matplotlib` (visualizations)
  - `scikit-learn` (for traditional ML models in Part 1)
  - `PyTorch` (for neural networks in Part 2)
  - `scipy` (for handling sparse matrices in Part 2)

Install the required libraries with `pip`:

```bash
pip install numpy matplotlib scikit-learn torch scipy
```

### Dataset Setup

1. Download `mnist.tar.gz` and extract it into your working directory:

   ```bash
   tar -xzf mnist.tar.gz
   ```

2. The archive contains a `Dataset` folder with data files, along with necessary Python files for both parts of the project.

## Running the Project

- **Part 1**: To run traditional ML models, navigate to the `part1` folder and execute:

  ```bash
  python main.py
  ```

- **Part 2**: For neural network experiments, navigate to the respective subfolders (e.g., `part2-nn` or `part2-mnist`) and run the relevant scripts, such as:

  ```bash
  python nnet_fc.py
  ```

## Feature Engineering and Exploration

Each part includes room for experimenting with feature extraction, dimensionality reduction, and more complex neural network architectures to improve classification accuracy.

## License

This project is for educational and demonstrative purposes. Please ensure compliance with academic integrity policies if using this project for course assignments or similar evaluations.
