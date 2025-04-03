class Neuron:
    def __init__(self):
        self.value = 0.0
        self.usage = False

    def get_usage(self):
        return self.usage
    def set_usage(self, usage):
        self.usage = usage


    def get_value(self):
        return self.value
    def set_value(self, value):
        self.value = value
