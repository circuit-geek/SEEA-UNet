# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 16:57:28 2022

@author: Gaurav Prasanna
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K

def SqueezeAndExcitation(inputs, ratio=8):
    b, _, _, c = inputs.shape
    x = GlobalAveragePooling2D()(inputs)
    x = Dense(c//ratio, activation="relu", use_bias=False)(x)
    x = Dense(c, activation="sigmoid", use_bias=False)(x)
    x = inputs * x
    return x

if __name__ == "__main__":
    inputs = Input(shape=(256,256,32))
    y = SqueezeAndExcitation(inputs)
    print(y.shape)
    



