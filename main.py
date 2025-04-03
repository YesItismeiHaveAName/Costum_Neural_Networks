# This is a sample Python script.
from General_Functions import *
from Model import Model





if __name__ == '__main__':
    x = Model()
    input_data = [1,1,1,2,3,4,5]
    labels = ['Car', 'Bird', 'Dog']
    x.add_input_layer(len(input_data))
    x.add_layer(len(input_data), 10, RELU())
    x.add_layer(10, 10, RELU())
    x.add_layer(10, 10, RELU())
    x.add_layer(10, 10, RELU())
    x.add_layer(10, 10, RELU())
    x.add_layer(10, 10, RELU())
    x.add_layer(10, 10, RELU())
    x.add_output_layer(3, SOFTMAX(2))
    x.print_model_layers()
    x.add_labels(labels)

    x.activate_model(input_data)
    print(x.get_output())
    print(x.check_softmax_sum())
    x.normalize_outputs()
    print(x.get_output())
    print(x.check_softmax_sum())
    x.get_prediction()
