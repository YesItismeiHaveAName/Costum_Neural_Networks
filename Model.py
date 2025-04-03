from Layers import *


class Model:
    def __init__(self):
        self.hidden_layers = []
        self.input_layer_dimension = None
        self.output_layer = None
        self.loss_function = None
        pass

    def add_input_layer(self, input_dimension):
        if input_dimension > 0:
            self.input_layer_dimension = input_dimension
        else:
            raise ValueError("Inputlayer-Dimension has to be a natural Number.")

    def add_output_layer(self, output_dimension, act_func):
        if output_dimension < 0:
            raise ValueError("Inputlayer-Dimension has to be a natural Number.")
        if self.input_layer_dimension is None:
            raise ValueError("Missing Input-Layer")
        elif len(self.hidden_layers) == 0:
            self.output_layer = Output_Layer(self.input_layer_dimension, output_dimension, act_func)
        else:
            self.output_layer = Output_Layer(self.hidden_layers[-1].get_output_dimension(), output_dimension, act_func)

    def add_layer(self, input_dimension, output_dimension, act_func):
        if len(self.hidden_layers) == 0:
            if self.input_layer_dimension != input_dimension:
                raise ValueError("Incompatible Dimensions: Inputlayer: " + str(self.input_layer_dimension) + ", Found: "
                                 + str(input_dimension))
            elif self.input_layer_dimension is None:
                raise ValueError("Missing Input-Layer")
            else:
                self.hidden_layers += [Layer(input_dimension, output_dimension, act_func)]
        else:
            if self.hidden_layers[-1].get_output_dimension() != input_dimension:
                raise ValueError("Incompatible Dimensions: Last Layer: " + str(
                    self.hidden_layers[-1].get_output_dimension()) + ", Found: " + str(input_dimension))
            else:
                self.hidden_layers += [Layer(input_dimension, output_dimension, act_func)]


    def print_model_layers(self):
        print(self.input_layer_dimension)
        for layer in self.hidden_layers:
            print(str(layer.get_input_dimension()) + ", " + str(layer.get_output_dimension()))
        print(self.output_layer.get_output_dimension())

    def check_softmax_sum(self):
        result = 0.0
        for value in self.output_layer.get_output_values():
            result += value
        return result

    def get_output(self):
        return self.output_layer.get_output_values()
    def normalize_outputs(self):
        self.output_layer.normalize_neurons()

    def add_labels(self, labels):
        self.output_layer.set_labels(labels)

    def get_prediction(self):
        self.output_layer.print_prediction()
        print(self.output_layer.get_predicted_label())


    def check_model(self):
        if self.input_layer_dimension is None or self.output_layer is None or self.loss_function is None:
            raise RuntimeError("Model hasn't been provided with Input-, Outputlayer or Loss-Function.")

    def activate_model(self, input_values):
        #just Check whether the Input-Data matches the Input-Dimension.
        if len(input_values) != self.input_layer_dimension:
            raise ValueError("Uncompatible Dimensions: Last Layer: " + str(
                self.hidden_layers[-1].get_output_dimension()) + ", Found: " + str(input_values))

        #Starting with the first Hidden Layer, you compute the Values of the Neurons of the Following Layer using
        #the previous Layer's Values.
        values = input_values
        for layer in self.hidden_layers:
            layer.activate_layer(values)
            values = layer.get_output_values()

        self.output_layer.activate_layer(values)


