# This is a sample Python script.
from General_Functions import *
from Model import Model





if __name__ == '__main__':
    x = Model()
    input_data = [0, 1]
    labels = ['X', 'Y']
    x.add_input_layer(len(input_data))
    x.add_layer(2,2, RELU())
    x.add_output_layer(2, LINEAR())
    x.set_loss_function(SQUARED_ERROR())
    x.add_labels(labels)



    data_set = [[1, 0], [1,0], [0,1],[0,1], [1, 0], [1,0], [0,1],[0,1]]
    labels = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    x.train_model(data_set, labels, 2000, learn_rate=0.001)

    x.activate_model(input_data)
    x.normalize_outputs()
    x.get_prediction()
