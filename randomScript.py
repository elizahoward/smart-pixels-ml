from OptimizedDataGenerator4 import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from qkeras import QDense, QActivation, QDenseBatchnorm
from qkeras.quantizers import quantized_bits, quantized_relu
import hls4ml

noGPU=False
if noGPU:
    tf.config.set_visible_devices([], 'GPU')

print(tf.config.experimental.list_physical_devices())
print(tf.test.is_built_with_cuda())
print(tf.test.is_built_with_gpu_support())
print(tf.test.is_gpu_available())
os.system("echo $PATH")

#######################
batch_size = 16384
validation_dir = f"/local/d1/smartpixML/filtering_models/shuffling_data/all_batches_shuffled_bigData_try2/filtering_records{batch_size}_data_shuffled_single_bigData/tfrecords_validation/"
train_dir = f"/local/d1/smartpixML/filtering_models/shuffling_data/all_batches_shuffled_bigData_try2/filtering_records{batch_size}_data_shuffled_single_bigData/tfrecords_train/"
x_feature_description: list = ['x_size','z_global','y_profile','x_profile','cluster','y_local']
trainODG = OptimizedDataGenerator(tf_records_dir=train_dir,load_records=True, x_feature_description=x_feature_description)
validationODG = OptimizedDataGenerator(tf_records_dir=validation_dir,load_records=True, x_feature_description=x_feature_description)



##########################

def CreateQModel(shape=16, model_file=None):
    # Define inputs
    input1 = tf.keras.layers.Input(shape=(1,), name="z_global")
    input2 = tf.keras.layers.Input(shape=(1,), name="y_local")
    input3 = tf.keras.layers.Input(shape=(1,), name="x_size")
    input4 = tf.keras.layers.Input(shape=(13,), name="y_profile")
    inputList = [input1, input2, input3, input4]
    inputList = [input2,input4]
    
    # Concatenate all inputs
    x =  tf.keras.layers.Concatenate()(inputList)
    # x = x_in = Input(shape, name="input1")
    x = QDenseBatchnorm(58,
      kernel_quantizer=quantized_bits(4,0,alpha=1),
      bias_quantizer=quantized_bits(4,0,alpha=1),
      name="dense1")(x)
    x = QActivation("quantized_relu(8,0)", name="relu1")(x)
    #Fermilab's output layer, not right for our dataset currently
    # x = QDense(3,
    #     kernel_quantizer=quantized_bits(4,0,alpha=1),
    #     bias_quantizer=quantized_bits(4,0,alpha=1),
    #     name="dense2")(x)
    # x = QActivation("linear", name="linear")(x)

    #Eric's Output layer
    x = QDense(
        1,
        kernel_quantizer=quantized_bits(4,0,alpha=1),
        bias_quantizer=quantized_bits(4,0,alpha=1),
        name="output_dense"
    )(x)
    output = QActivation("smooth_sigmoid", name="output")(x)
    model = tf.keras.Model(inputs=inputList, outputs=x)

    model.summary()
    return model
quantizedModel = CreateQModel()


###########################
#Training

quantizedModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
# callbacks=[]
# learningRates = [0.1,0.9,0.6,0.3,0.1,0.03,0.01,0.001,0.0001,0.00001,0.000001]
# callbacks.append(tf.keras.callbacks.LearningRateScheduler(lambda epoch,lr : lr if epoch<5 else lr*np.exp(-0.1)))
callbacks=[]
historyQ = quantizedModel.fit(x=trainODG,validation_data=validationODG, callbacks=callbacks,epochs=5)


######################

config = hls4ml.utils.config_from_keras_model(quantizedModel, granularity='name')
# Convert to an hls model
output_dir = "./hlsTmp"
hls_model = hls4ml.converters.convert_from_keras_model(quantizedModel, hls_config=config, output_dir=output_dir)
hls_model.write()


hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)

###############
hls_model.build(csim=False)


