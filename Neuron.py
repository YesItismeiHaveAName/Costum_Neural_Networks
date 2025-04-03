class Neuron:
    def __init__(self):
        self.value = 0.0
        self.bias = 0.2

    def get_value(self):
        return self.value
    def set_value(self, value):
        self.value = value

    def get_bias(self):
        return self.bias
    def set_bias(self, value):
        self.bias = value
