# fix for keras v3.0 update
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1' 

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from qkeras import *
import OptimizedDataGenerator as ODG
import time

os.environ['TF_USE_LEGACY_KERAS'] = '1' 

# python based
import random
from pathlib import Path
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# custom code
from loss import *
from models import *


class PredictCluster:

    def __init__(self,
            data_directory_path: str = "./",
            labels_directory_path: str = "./",
            is_directory_recursive: bool = False,
            file_type: str = "parquet",
            data_format: str = "3D",
            batch_size: int = 500,
            labels_list: list= ['x-midplane','y-midplane','cotAlpha','cotBeta'],
            to_standardize: bool = False,
            input_shape: tuple = (2,13,21),
            transpose = (0,2,3,1),
            include_y_local: bool = False,
            use_time_stamps = [0,19],
            output_dir: str = "./ouput",
            learning_rate: float = 0.001
            ):
        
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        # create tf records directory
        stamp = '%08x' % random.randrange(16**8)
        tfrecords_dir_train = Path(self.output_dir, f"tfrecords_train_{stamp}").resolve()
        tfrecords_dir_validation = Path(self.output_dir, f"tfrecords_validation_{stamp}").resolve()
        if not os.path.exists(tfrecords_dir_train):
            os.mkdir(tfrecords_dir_train)
            os.mkdir(tfrecords_dir_validation)

        start_time = time.time()

        self.training_generator = ODG.OptimizedDataGenerator(
            data_directory_path = data_directory_path,
            labels_directory_path = labels_directory_path,
            is_directory_recursive = is_directory_recursive,
            file_type = file_type,
            data_format = data_format,
            batch_size = batch_size,
            to_standardize= to_standardize,
            include_y_local= include_y_local,
            labels_list = labels_list,
            input_shape = input_shape,
            transpose = transpose,
            save=True,
            use_time_stamps = use_time_stamps,
            tfrecords_dir = tfrecords_dir_train
        )

        print("--- Training generator %s seconds ---" % (time.time() - start_time))

        start_time = time.time()

        self.validation_generator = ODG.OptimizedDataGenerator(
            data_directory_path = data_directory_path,
            labels_directory_path = labels_directory_path,
            is_directory_recursive = is_directory_recursive,
            file_type = file_type,
            data_format = data_format,
            batch_size = batch_size,
            to_standardize= to_standardize,
            include_y_local= include_y_local,
            labels_list = labels_list,
            input_shape = input_shape,
            transpose = transpose,
            save=True,
            use_time_stamps = use_time_stamps,
            tfrecords_dir = tfrecords_dir_validation
        )

        print("--- Validation generator %s seconds ---" % (time.time() - start_time))

        # compiles model
        self.n_filters = 5 # model number of filters
        self.pool_size = 3 # model pool size

        # Rearrange input shape for the model
        self.shape = (input_shape[1], input_shape[2], input_shape[0])

        self.createModel()

        self.compileModel(learning_rate=learning_rate)

    def createModel(self):
        start_time = time.time()
        self.model=CreatePredictionModel(shape=self.shape, n_filters=self.n_filters, pool_size=self.pool_size)
        self.model.summary()
        print("--- Model create and compile %s seconds ---" % (time.time() - start_time))


    def compileModel(self, learning_rate):
        print(f"Compiling model with learning rate: {learning_rate}")
        self.learning_rate = learning_rate
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss=custom_loss)


    def runTraining(self):
        epochs = 50
        early_stopping_patience = 50

        # launch quick training once gpu is available
        es = EarlyStopping(
        patience=early_stopping_patience,
        restore_best_weights=True
        )
    
        # checkpoint path
        checkpoint_filepath = Path(self.output_dir,"weights", 'weights.{epoch:02d}-t{loss:.2f}-v{val_loss:.2f}.hdf5').resolve()
        mcp = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            save_best_only=True,
        )

        # train
        self.history = self.model.fit(x=self.training_generator,
                        validation_data=self.validation_generator,
                        callbacks=[mcp],
                        epochs=epochs,
                        shuffle=False,
                        verbose=1)

    def checkResiduals(self):
        p_test = self.model.predict(self.validation_generator)

        complete_truth = None
        for _, y in tqdm(self.validation_generator):
            if complete_truth is None:
                complete_truth = y
            else:
                complete_truth = np.concatenate((complete_truth, y), axis=0)
        
        minval=1e-9

        # creates df with all predicted values and matrix elements - 4 predictions, all 10 unique matrix elements
        df = pd.DataFrame(p_test,columns=['x','M11','y','M22','cotA','M33','cotB','M44','M21','M31','M32','M41','M42','M43'])
        df['x'] *= 75
        df['y'] *= 18.75
        df['cotA'] *= 8
        df['cotB'] *= 0.5

        # stores all true values in same matrix as xtrue, ytrue, etc.
        df['xtrue'] = complete_truth[:,0]*75
        df['ytrue'] = complete_truth[:,1]*18.75
        df['cotAtrue'] = complete_truth[:,2]*8
        df['cotBtrue'] = complete_truth[:,3]*0.5
        df['M11'] = minval+tf.math.maximum(df['M11'], 0)
        df['M22'] = minval+tf.math.maximum(df['M22'], 0)
        df['M33'] = minval+tf.math.maximum(df['M33'], 0)
        df['M44'] = minval+tf.math.maximum(df['M44'], 0)

        df['sigmax'] = abs(df['M11'])
        df['sigmay'] = np.sqrt(df['M21']**2 + df['M22']**2)
        df['sigmacotA'] = np.sqrt(df['M31']**2+df['M32']**2+df['M33']**2)
        df['sigmacotB'] = np.sqrt(df['M41']**2+df['M42']**2+df['M43']**2+df['M44']**2)

        # calculates residuals for x, y, cotA, cotB
        residualsx = df['xtrue'] - df['x']
        residualsy = df['ytrue'] - df['y']
        residualsA = df['cotAtrue'] - df['cotA']
        residualsB = df['cotBtrue'] - df['cotB']

        # x
        xmean, xstd = (np.mean(np.abs(residualsx)),np.std(np.abs(residualsx)))
        print(f"Mean and standard deviation of residuals for x-midplane: ({xmean},{xstd})")
        # y
        ymean, ystd = (np.mean(np.abs(residualsy)),np.std(np.abs(residualsy)))
        print(f"Mean and standard deviation of residuals for y-midplane: ({ymean},{ystd})")
        # cotA
        cotAmean, cotAstd = (np.mean(np.abs(residualsA)),np.std(np.abs(residualsA)))
        print(f"Mean and standard deviation of residuals for cot(alpha): ({cotAmean},{cotAstd})")
        # cotB
        cotBmean, cotBstd = (np.mean(np.abs(residualsB)),np.std(np.abs(residualsB)))
        print(f"Mean and standard deviation of residuals for cot(beta): ({cotBmean},{cotBstd})")

        x = sns.regplot(x=df['xtrue'], y=residualsx, x_bins=np.linspace(-75,75,50), fit_reg=None, marker='.')
        plt.xlabel(r'True $x$ [um]')
        plt.ylabel(r'$x-\mu_x$ [um]')
        plt.show()

        plt.hist(residualsx, bins = 20)
        plt.xlabel(r'$x-\mu_x$ [um]')
        plt.show()
