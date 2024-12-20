# fix for keras v3.0 update
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1' 

import tensorflow as tf
# import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import Sequence
from qkeras import *


# from tensorflow.keras import datasets, layers, models

def var_network(var, hidden=10, output=2):
    var = Flatten()(var)
    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(8, 0, 1)")(var)
    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(8, 0, 1)")(var)
    return QDense(
        output,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
    )(var)

def conv_network(var, n_filters=5, kernel_size=3):
    var = QSeparableConv2D(
        n_filters,kernel_size,
        depthwise_quantizer=quantized_bits(4, 0, 1, alpha=1),
        pointwise_quantizer=quantized_bits(4, 0, 1, alpha=1),
        bias_quantizer=quantized_bits(4, 0, alpha=1),
        depthwise_regularizer=tf.keras.regularizers.L1L2(0.01),
        pointwise_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(4, 0, 1)")(var)
    var = QConv2D(
        n_filters,1,
        kernel_quantizer=quantized_bits(4, 0, alpha=1),
        bias_quantizer=quantized_bits(4, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = QActivation("quantized_tanh(4, 0, 1)")(var)    
    return var

def CreatePredictionModel(shape, n_filters, pool_size, include_y_local):
    x_base = x_in = Input(shape)
    stack = conv_network(x_base, n_filters)
    stack = AveragePooling2D(
        pool_size=(pool_size, pool_size), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(stack)
    stack = QActivation("quantized_bits(8, 0, alpha=1)")(stack)
    if include_y_local:
        stack = Flatten()(stack)
        y_local_in = Input(shape=(1,), name="y_local_input")
        stack = Concatenate()([stack, y_local_in])
    stack = var_network(stack, hidden=16, output=14)
    model = Model(inputs=x_in, outputs=stack)
    return model
"""
def CreatePredictionModelYLocal(shape, n_filters, pool_size):
    x_base = x_in = Input(shape, name="X_input")  # Main input (X)
    y_local_in = Input(shape=(1,), name="y_local_input")  # y_local input
    
    stack = conv_network(x_base, n_filters)
    stack = AveragePooling2D(
        pool_size=(pool_size, pool_size), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(stack)
    stack = QActivation("quantized_bits(8, 0, alpha=1)")(stack)
    stack = Flatten()(stack)
    stack = Concatenate()([stack, y_local_in])
    stack = var_network(stack, hidden=16, output=14)
    model = Model(inputs=[x_in, y_local_in], outputs=stack)
    return model
"""

def CreateClassificationModel(shape, n_filters, pool_size, include_y_local):
    x_in = Input(shape)
    stack = Reshape((13, 21))(x_in)
    stack = Lambda(lambda x: tf.reduce_sum(x, axis=1))(stack) # convert to y_profile
    #stack = QActivation("quantized_tanh(4, 0, 1)")(stack)    
    #stack = Flatten()(stack)
    if include_y_local:
        stack = QDense(
            1,
            kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
            activity_regularizer=tf.keras.regularizers.L2(0.01),
        )(stack)
        y_local_in = Input(shape=(1,), name="y_local_input")
        stack = Concatenate()([stack, y_local_in])
        stack = QDense(
            1,
            kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
            activity_regularizer=tf.keras.regularizers.L2(0.01),
        )(stack)
        stack = Dense(1, activation='sigmoid')(stack)
        model = Model(inputs=[x_in, y_local_in], outputs=stack)
    else:
        stack = QDense(
            1,
            kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
            activity_regularizer=tf.keras.regularizers.L2(0.01),
        )(stack)
        stack = Dense(1, activation='sigmoid')(stack)
        model = Model(inputs=x_in, outputs=stack)
    return model
