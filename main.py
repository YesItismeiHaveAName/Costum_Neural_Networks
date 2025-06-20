# This is a sample Python script.
from General_Functions import *
from Model import Model





if __name__ == '__main__':
    xor_model = Model()
    input_data = [1.0, 0.0]
    labels = ['TRUE', 'FALSE']
    xor_model.add_input_layer(len(input_data))
    xor_model.add_layer(2, 4, RELU())
    xor_model.add_layer(4, 3, SIGMOID())
    xor_model.add_layer(3, 3, RELU())
    xor_model.add_output_layer(2, SOFTMAX())
    xor_model.set_loss_function(SQUARED_ERROR())
    xor_model.add_labels(labels)


    #XOR -> [1 , 0] -> TRUE [0, 1] -> TRUE [0,0] [1,1] -> FALSE

    data_set = [[1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [1.0, 1.0]]


    labels = [[1.0, 0.0],
              [1.0, 0.0],
              [0.0, 1.0],
              [0.0, 1.0]]
    xor_model.train_model(data_set, labels, 20000, learn_rate=0.3)


    print('TEST_1_TRUE')
    xor_model.activate_model(input_data)
    xor_model.normalize_outputs()
    xor_model.get_prediction()
    print('TEST_2_FALSE')
    xor_model.activate_model([1.0, 1.0])
    xor_model.normalize_outputs()
    xor_model.get_prediction()
    print('TEST_3_TRUE')
    xor_model.activate_model([1.0, 0.0])
    xor_model.normalize_outputs()
    xor_model.get_prediction()
    print('TEST_4_TRUE')
    xor_model.activate_model([0.0, 1.0])
    xor_model.normalize_outputs()
    xor_model.get_prediction()
    print('TEST_5_FALSE')
    xor_model.activate_model([1.0, 1.0])
    xor_model.normalize_outputs()
    xor_model.get_prediction()
    print('TEST_6_FALSE')
    xor_model.activate_model([1.0, 1.0])
    xor_model.normalize_outputs()
    xor_model.get_prediction()





