from General_Functions import rng_value, cut_decimals
from Neuron import Neuron


class Layer:
    def __init__(self, input_dimension, output_dimension, activation_function):
        self.connections = []
        self.neurons = []
        self.activation_function = activation_function
        for i in range(output_dimension):
            column = []
            for j in range(input_dimension):
                column += [rng_value()]
            self.connections += [column]
            self.neurons += [Neuron()]
        #first dimension is the output: if you want 5 neurons to be the output, the output-dimension is 5.
        #if the amount of the previous layer was 10, the INPUT-dimension is 10 as well.

    def calculate_raw_values(self, input_values):
        for y in range(len(self.connections)):
            value = 0.0
            for x in range(len(self.connections[y])):
                value += input_values[x] * self.connections[y][x]
            self.neurons[y].set_raw_value(value + self.neurons[y].get_bias())

    def activate_neurons(self):
        all_neuron_values = self.get_raw_values()
        for neuron in self.neurons:
            neuron.set_value(self.activation_function.activate(neuron.get_raw_value(), all_neuron_values))


    def get_output_values(self):
        values = []
        for neuron in self.neurons:
            values += [neuron.get_value()]
        return values

    def get_raw_values(self):
        values = []
        for neuron in self.neurons:
            values += [neuron.get_raw_value()]
        return values

    def activate_layer(self, input_values):
        self.calculate_raw_values(input_values)
        self.activate_neurons()

    def get_input_dimension(self):
        return len(self.connections[0])
    def get_output_dimension(self):
        return len(self.connections)

    def normalize_neurons(self):
        for neuron in self.neurons:
            new_value = cut_decimals(neuron.get_value(), 2)
            neuron.set_value(new_value)

    def print_layer(self):
        print('Y: ' + str(self.get_output_dimension()) + ', X: ' + str(self.get_input_dimension()))

    def backpropagate(self, previous_values, gradients, learn_rate):
        #gradients hat die Output_dimension.
        #previous_values hat die Input_dimension

        for y in range(self.get_output_dimension()):
            for x in range(self.get_input_dimension()):
                self.connections[y][x] -= (learn_rate * gradients[y] * previous_values[x] * self.activation_function.get_derivative(self.neurons[y].get_raw_value()))
            self.neurons[y].set_bias(self.neurons[y].get_bias() - (learn_rate * gradients[y] * self.activation_function.get_derivative(self.neurons[y].get_raw_value())))

    def calculate_gradients(self, next_layer_gradients):
        raw_values = self.get_raw_values()
        gradients = []
        for i in range(self.get_input_dimension()):
            gradient_i = 0.0
            for j in range(self.get_output_dimension()):
                gradient_i += self.connections[j][i] * next_layer_gradients[j]
            gradients.append(gradient_i * self.activation_function.get_derivative(raw_values[j]))
        return gradients




class Output_Layer(Layer):
    def __init__(self, input_dimension, output_dimension, activation_function):
        super().__init__(input_dimension, output_dimension, activation_function)
        self.labels = None

    def set_labels(self, labels):
        if len(labels) != len(self.neurons):
            raise ValueError("Amount of Labels does not fit the Amount of Neurons of the Output-Layer: Labels: "
                             + str(len(labels))+" Neurons: "+str(len(self.neurons)))
        self.labels = labels

    def get_predicted_label(self):
        index = 0
        for i in range(len(self.labels)):
            if self.neurons[index].get_value() < self.neurons[i].get_value():
                index = i
        return self.labels[index]

    def print_prediction(self):
        for i in range(len(self.labels)):
            print(str(self.labels[i])+": " + str(self.neurons[i].get_value()))