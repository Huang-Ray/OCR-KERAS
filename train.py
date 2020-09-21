import numpy as np
from prepare_data import generatorData
from model import CRNN

import tensorflow as tf
from keras.layers import Lambda
from keras.utils import to_categorical
from keras import backend as K 


def ctc_lambda_func(args):


    pass

class model_training:

    def __init__(self):


        pass

gendata = generatorData()
train_datas, train_labels = gendata.gen_Data(batch_size=60)
test_datas, test_labels = gendata.gen_Data(batch_size=40)

train_hot = to_categorical(train_labels)

input_shapes = np.shape(train_datas)[1:]

model = CRNN(input_shapes).CRNN_model


loss = Lambda()



"""
model.compile(loss=ctc_loss,
                optimizer=Adam(lr=0.001),
                metrics=['accuracy', 'categorical_accuracy'])
"""


