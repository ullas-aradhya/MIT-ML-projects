import numpy as np
import math
import random
import neural_nets as nn

def approx_equal(x, y, tolerance=1e-9):
    return abs(x - y) < tolerance

def test_relu():
    #Positive
    x = random.uniform(0,1000)
    assert nn.rectified_linear_unit(x) == x
    print("Positive test passed")
    #Negative
    x = random.uniform(-1000,0)
    print("x: ", x)
    assert nn.rectified_linear_unit(x) == 0
    #Zero
    x = 0
    assert nn.rectified_linear_unit(x) == 0
    print("Zero test passed")

if __name__ == "__main__":
    test_relu()

