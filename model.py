
import yaml
import numpy as np
from keras.layers import Conv2D, Input, MaxPooling2D
from keras.layers import Bidirectional, LSTM, Dense
from keras.layers import BatchNormalization, Bidirectional, Activation, Dropout, Reshape
from keras.models import Model
from keras.utils import plot_model
from keras import backend as K 

config_path = "./conf/conf.yaml"

class CRNN:

    def __init__(self, input_shapes, class_num=37):
        
        self.config = self.get_Config()
        self.input_shapes = input_shapes
        self.class_num = class_num
        self.CNN_layer = self.make_CNN()
        self.RNN_layer = self.make_RNN()
        self.CRNN_model = self.create_model()

    def get_Config(self):
        with open(config_path, "r") as stream:
            conf = yaml.load(stream, Loader=yaml.FullLoader)
        return conf["model"]

    # create CNN
    def make_CNN(self):
        CNN_CONFIG = self.config["CNN"]
        print(CNN_CONFIG)
        
        layers = ["L1", "L2", "M", "L3", "L4", "C", "L5", "L6", "L7", "C", "L8", "L9", "C", "M", "L10", "L11", "L12", "M"]
        
        input_shapes = self.input_shapes
        inputs = Input(shape=input_shapes, name="CNN_inputs")

        inner = Conv2D(filters=CNN_CONFIG["L1"]["filters"], kernel_size=(CNN_CONFIG["L1"]["kernels"], CNN_CONFIG["L1"]["kernels"]),
                        padding=CNN_CONFIG["L1"]["padding"])(inputs)
        inner = BatchNormalization()(inner)
        inner = Activation(CNN_CONFIG["L1"]["activation"])(inner)

        for _, layer in enumerate(layers[1:], 1):
            if layer == "M":
                inner = MaxPooling2D(pool_size=(2, 2), strides=2)(inner)
            elif layer == "C":
                inner = MaxPooling2D(pool_size=(2, 2), strides=(2, 1))(inner)
            else:
                inner = Conv2D(filters=CNN_CONFIG[layer]["filters"], kernel_size=(CNN_CONFIG[layer]["kernels"], CNN_CONFIG[layer]["kernels"]),
                                padding=CNN_CONFIG[layer]["padding"])(inner)
                inner = BatchNormalization()(inner)
                inner = Activation(CNN_CONFIG[layer]["activation"])(inner)

        model = Model(inputs, inner)
        plot_model(model, to_file='CNN.png', show_shapes=True)
        return model

    def make_RNN(self):
        input_shapes = K.int_shape(self.CNN_layer.layers[-1].output)
        input_shapes = input_shapes[2:]
        print("RNN input shapes = ", input_shapes)

        inputs = Input(shape=input_shapes, name="RNN_input")
        inner = Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.3))(inputs)
        inner = Bidirectional(LSTM(units=64, return_sequences=True, dropout=0.3))(inner)
        inner = Dropout(rate=0.25)(inner)
        outputs = Dense(units=self.class_num)(inner)

        model = Model(inputs, outputs)
        plot_model(model, to_file='RNN.png', show_shapes=True)
        return model

    def create_model(self):
        inputs = Input(shape=self.input_shapes, name='CRNN_input')
        x = inputs
        for i in range(1, len(self.CNN_layer.layers)):
            x = self.CNN_layer.layers[i](x)
        rnn_input_shapes = np.shape(x)[2:]
        x = Reshape((rnn_input_shapes[0], rnn_input_shapes[1]))(x)
        for j in range(1, len(self.RNN_layer.layers)):
            x = self.RNN_layer.layers[j](x)
        outputs = x
        
        crnn_model = Model(inputs, outputs)
        return crnn_model

input_shapes = (64, 128, 3)
model = CRNN(input_shapes).CRNN_model


model.summary()


