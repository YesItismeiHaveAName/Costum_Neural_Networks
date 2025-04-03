from random import random


class Activation_Function:
    def activate(self, value, other_values):
        pass
class RELU(Activation_Function):
    def __init__(self):
        pass
    def activate(self, value, other_values):
        if value > 0.0:
            return value
        return 0.0

class BINARY_STEP(Activation_Function):
    def __init__(self, threshold):
        self.threshold = threshold

    def activate(self, value, other_values):
        if value > self.threshold:
            return 1.0
        return 0.0

class SOFTMAX(Activation_Function):
    def __init__(self, exponent):
        self.exponent = exponent

    def activate(self, value, other_values):
        div = 0.0
        for num in other_values:
            div += num**self.exponent
        #important: when all values in the layer are 0s, then we return 1 / Number of Neurons of that Layer.
        if div == 0.0:
            return 1 / len(other_values)
        return float(value**self.exponent / div)



def rng_value():
    return float(random()-0.5)*3
