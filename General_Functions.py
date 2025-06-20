from random import random


e = 2.718


class Activation_Function:
    def activate(self, value, other_values):
        pass
    def get_derivative(self, value):
        pass
class RELU(Activation_Function):
    def __init__(self):
        pass
    def activate(self, value, other_values):
        if value > 0.0:
            return value
        return 0.0

    def get_derivative(self, value):
        if value > 0.0:
            return 1.0
        return 0.0

class BINARY_STEP(Activation_Function):
    def __init__(self, threshold):
        self.threshold = threshold

    def activate(self, value, other_values):
        if value > self.threshold:
            return 1.0
        return 0.0

    def get_derivative(self, value):
        if value >= self.threshold:
            return 1.0
        return 0.0

class SOFTMAX(Activation_Function):
    def __init__(self):
        self.value = 0.0
    def activate(self, value, other_values):
        div = 0.0
        for num in other_values:
            div += e ** num
        #important: when all values in the layer are 0s, then we return 1 / Number of Neurons of that Layer.
        if div == 0.0:
            return 1 / len(other_values)
        self.value = float(e ** value / div)
        return float(e ** value / div)

    def get_derivative(self, value):
        return self.value * (1-self.value)

class LINEAR(Activation_Function):
    def __init__(self):
        pass
    def activate(self, value, other_values):
        return value

    def get_derivative(self, value):
        return 1

class SIGMOID(Activation_Function):
    def __init__(self):
        pass
    def activate(self, value, other_values):
        return float(1 / (1 + (e ** (-value))))

    def get_derivative(self, value):
        activated_value = self.activate(value, [])
        return activated_value * (1 - activated_value)

class Loss_Functions:
    def calculate_error(self, prediction, label):
        pass
    def derivative(self, prediction, label):
        pass

class SQUARED_ERROR(Loss_Functions):
    def calculate_error(self, prediction, label):
        result = 0.0
        for i in range(len(prediction)):
            result += (prediction[i] - label[i])**2
        result = result / len(prediction)
        return result

    def derivative(self, prediction, label):
        result = []
        for i in range(len(prediction)):
            result += [(2/len(prediction)) * (prediction[i] - label[i])]
        return result
#add more Loss-Functions

def rng_value():
    return float(random()-0.5)*0.1

def cut_decimals(value, number_of_decimals):
    div = 10**number_of_decimals
    return float(int(value * div) / div)
