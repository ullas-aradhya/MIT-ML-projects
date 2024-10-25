# Sentiment Analysis Classifier Project

## Overview

The goal of this project is to design a classifier for sentiment analysis of product reviews. The dataset used consists of product reviews written by Amazon customers, specifically focusing on food products. Each review has been labeled as either positive (`+1`) or negative (`-1`), based on an adjusted 5-point scale.

The project explores the implementation and comparison of three types of linear classifiers:

1. **Perceptron Algorithm**
2. **Average Perceptron Algorithm**
3. **Pegasos Algorithm**

### Dataset

The dataset consists of text reviews and corresponding labels. Here are two sample entries:

1. **Review**: "Nasty. No flavor. The candy is just red, No flavor. Just plain and chewy. I would never buy them again."
   - **Label**: `-1` (Negative)

2. **Review**: "YUMMY! You would never guess that they're sugar-free and it's great that you can eat them pretty much guilt-free! I was so impressed that I've ordered some for myself. These are just EXCELLENT!"
   - **Label**: `+1` (Positive)

### Objectives

- **Implement three linear classifiers**: the Perceptron, Average Perceptron, and Pegasos algorithms.
- **Experiment with text features** to enhance classifier performance.
- **Evaluate and compare classifier performance** on the dataset.

## Project Structure

The project consists of the following Python files:

- **`project1.py`**: Contains various functions and templates necessary for implementing the learning algorithms.
- **`main.py`**: A script skeleton where the functions are called and experiments are executed.
- **`utils.py`**: Includes utility functions provided for common operations.
- **`test.py`**: Contains test cases to help debug the implementations. These tests serve as a guide for validating the correctness of your code but may differ from the tests used for grading.

## Setup and Installation

### Prerequisites

- Python 3.8 or above
- Libraries:
  - `numpy` (for numerical computations)
  - `matplotlib` (for plotting)


## Running the Project

1. To run experiments, use the `main.py` script:
   ```bash
   python main.py
   ```
2. To validate your implementation with provided tests, run the `test.py` script:
   ```bash
   python test.py
   ```

Feel free to add additional test cases in `test.py` to further validate your implementation locally.

## Feature Engineering and Exploration

In addition to the basic text features, experiment with adding and modifying features to see how they affect classifier performance. Possible approaches include:

- Using word counts or TF-IDF (Term Frequency-Inverse Document Frequency).
- Exploring n-grams (bigrams or trigrams) to capture word context.
- Applying feature scaling or normalization.

## License

This project is for educational purposes as part of the course Machine Learning with Python-From Linear Models to Deep Learning. 
