# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 13:56:16 2020

@author: ab89141
"""

import keras
import tensorflow as tf
import numpy as np

y_true = np.array([[4, 2, 1], [2, 3, 0]])                                    # (2, 3)
y_pred = keras.utils.to_categorical(np.array([[4, 1, 3], [1, 2, 4]]), 5)     # (2, 3, 5)


input_length = np.array([[2], [2]])                                         # (2, 1)
label_length = np.array([[2], [2]])                                         # (2, 1)

cost = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)