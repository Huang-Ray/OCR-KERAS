import numpy as np
from prepare_data import generatorData
from model import CRNN

import tensorflow as tf
from keras.layers import Input, Layer
from keras.layers import Lambda
from keras.utils import to_categorical
from keras.optimizers import RMSprop
from keras import backend as K 

class model_training:

    def __init__(self):


        pass

gendata = generatorData()
train_datas, train_labels = gendata.gen_Data(batch_size=60)
test_datas, test_labels = gendata.gen_Data(batch_size=40)

train_datas_shapes = np.shape(train_datas)

train_labels_s = np.reshape(train_labels, (int(len(train_labels) / 4), 4))

train_hot = to_categorical(train_labels_s)

input_shapes = np.shape(train_datas)[1:]

model = CRNN(input_shapes).CRNN_model

model.summary()





