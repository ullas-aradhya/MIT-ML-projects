import numpy as np
import math
import random
import neural_nets as nn

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

if __name__ == "__main__":
    test_relu()
    test_relu_derivative()

