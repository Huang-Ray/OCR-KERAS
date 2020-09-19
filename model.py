
import yaml
import numpy as np
from keras.layers import Conv2D, Input, MaxPooling2D
from keras.layers import BatchNormalization, Bidirectional, Activation
from keras.models import Model

from keras.utils import plot_model

config_path = "./conf/conf.yaml"

class CRNN:

    def __init__(self, input_shapes):
        
        self.config = self.get_Config()
        self.input_shapes = input_shapes
        self.CNN_layer = self.make_CNN

    def get_Config(self):
        with open(config_path, "r") as stream:
            conf = yaml.load(stream, Loader=yaml.FullLoader)
        return conf["model"]

    # create CNN
    def make_CNN(self):
        CNN_CONFIG = self.config["CNN"]
        print(CNN_CONFIG)
        
        layers = ["L1", "L2", "M", "L3", "L4", "C", "L5", "L6", "L7", "C", "L8", "L9", "C", "M", "L10", "L11", "L12", "M"]
        
        inputs = Input(shape=self.input_shapes, name="inputs")

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


input_shapes = (64, 128, 3)
model = CRNN(input_shapes).make_CNN()

model.summary()



