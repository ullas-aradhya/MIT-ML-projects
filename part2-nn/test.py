import numpy as np
import random
import neural_nets as nn
from neural_nets import NeuralNetwork


def approx_equal(x, y, tolerance=1e-9):
    return abs(x - y) < tolerance

def test_relu():
    try:
        #Positive
        x = random.uniform(0,1000)
        assert nn.rectified_linear_unit(x) == x
        #Negative
        x = random.uniform(-1000,0)
        assert nn.rectified_linear_unit(x) == 0
        #Zero
        x = 0
        assert nn.rectified_linear_unit(x) == 0
        print("rectified_linear_unit: PASS")

    except AssertionError:
        print("rectified_linear_unit: FAIL")

def test_relu_derivative():
    try:
        #Positive
        x = random.uniform(0,1000)
        assert nn.rectified_linear_unit_derivative(x) == 1
        #Negative
        x = random.uniform(-1000,0)
        assert nn.rectified_linear_unit_derivative(x) == 0
        #Zero
        x = 0
        try:
            assert nn.rectified_linear_unit_derivative(x) == 0
        except AssertionError:
            print("rectified_linear_unit_derivative: PASS")

    except AssertionError:
        print("rectified_linear_unit_derivative: FAIL")

def test_neural_network_train():
    # Initialize the neural network
    nn = NeuralNetwork()
    
    # Initial weights and biases
    initial_input_to_hidden = nn.input_to_hidden_weights.copy()
    initial_hidden_to_output = nn.hidden_to_output_weights.copy()
    initial_biases = nn.biases.copy()
    
    # Train on a single example
    nn.train(2, 1, 10)
    
    # Check if weights and biases have been updated
    assert not np.array_equal(nn.input_to_hidden_weights, initial_input_to_hidden), "Input to hidden weights were not updated"
    assert not np.array_equal(nn.hidden_to_output_weights, initial_hidden_to_output), "Hidden to output weights were not updated"
    assert not np.array_equal(nn.biases, initial_biases), "Biases were not updated"
    
    # Check specific values (adjust these based on your expected output)
    np.testing.assert_almost_equal(nn.input_to_hidden_weights, np.matrix([[1.002, 1.001], [1.002, 1.001], [1.002, 1.001]]), decimal=3)
    np.testing.assert_almost_equal(nn.hidden_to_output_weights, np.matrix([[1.003, 1.003, 1.003]]), decimal=3)
    np.testing.assert_almost_equal(nn.biases, np.matrix([[0.001], [0.001], [0.001]]), decimal=3)
    
    print("train_neural_network: PASS")

if __name__ == "__main__":
    test_relu()
    test_relu_derivative()
    test_neural_network_train()

