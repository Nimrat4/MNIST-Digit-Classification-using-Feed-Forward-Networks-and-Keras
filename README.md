# MNIST-Digit-Classification-using-Feed-Forward-Networks-and-Keras
Project Overview

This project aims to classify handwritten digits from the MNIST dataset using two different approaches:

Feed Forward Network (FFN) using Scikit-learn

Deep Neural Networks using Keras

The MNIST dataset contains 70,000 images of handwritten digits (0-9), each of size 28x28 pixels. The dataset is loaded using fetch_openml("mnist_784").

Part 1: Feed Forward Network (FFN) using Scikit-learn

Steps:

Download the Dataset:

Fetch the dataset from OpenML using fetch_openml("mnist_784").

Print the shape of the input data and target data to verify correct loading.

Display Images:

Reshape the dataset into (70,000, 28, 28) dimensions.

Display the first 10 images using matplotlib.

Feature Scaling:

Normalize the dataset (scale pixel values between 0 and 1).

Train-Test Split:

Split the dataset into training (80%) and testing (20%) using train_test_split.

Train the FFN Model:

Use MLPClassifier() with one hidden layer of 64 neurons and max_iter=10.

Model Evaluation:

Predict on the test set and calculate:

Accuracy (accuracy_score)

Precision, Recall, and F1-score (precision_recall_fscore_support)

Compare Accuracy for Different Splits:

Evaluate test accuracy for 60-40, 75-25, 80-20, and 90-10 splits.

Visualize the accuracy trends using graphs.

Impact of Iterations:

Increase max_iter to 20, 50, 100, 150, and 200.

Observe variations in accuracy.

Part 2: Exploring Different Neural Network Designs using Keras

Experiments Conducted:

Number of Nodes in a Single Hidden Layer:

Train networks with different node sizes (4, 32, 64, 128, 512, 2056) for 10 epochs.

Record training/testing accuracy, model parameters, and training time.

Number of Layers:

Train networks with 5 hidden layers (64 nodes each) for 10 epochs.

Repeat with 4, 6, 8, and 16 layers.

Extend training to 30 epochs to observe changes.

Layer-Node Combinations:

Train models with varying numbers of neurons per layer (e.g., 256, 128, 64, 32).

Identify the best-performing architecture based on accuracy.

Input Size Impact:

Train a network with 4 hidden layers (256 nodes each) using ReLU activation.

Evaluate accuracy changes.

Dataset Splitting Variations:

Experiment with different training/testing splits (e.g., 50k-20k, 55k-15k, etc.).

Analyze accuracy variations.

Activation Functions:

Train models with 4 hidden layers (64 nodes each) using different activation functions:

Sigmoid

Tanh

ReLU

Compare training/testing accuracies over 10 and 30 epochs.

Activation Function Combinations:

Train models with different activation function combinations across three layers:

Layer 1: Sigmoid, Layer 2: ReLU, Layer 3: Tanh

Experiment with other activation orderings.

Identify the best-performing combination.

Conclusion

This project explores how different network architectures, activation functions, and dataset splits impact digit classification performance. The results provide insights into designing efficient neural networks for image classification tasks.

Repository Structure

mnist_ffn.py: Implementation of Feed Forward Network using Scikit-learn.

mnist_keras.py: Implementation of Neural Networks using Keras.

results/: Contains graphs and logs for different experiments.

README.md: Project documentation (this file).

Requirements

Python 3.x

Scikit-learn

TensorFlow/Keras

Matplotlib

NumPy

To install dependencies, run:
