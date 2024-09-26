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
plt.rcParams['figure.dpi'] = 300
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
            units_list: list = ["[\u03BCm]", "[\u03BCm]", "", ""],
            normalization: list = np.array([75., 18.75, 8.0, 0.5]),
            to_standardize: bool = False,
            input_shape: tuple = (2,13,21),
            transpose = (0,2,3,1),
            include_y_local: bool = False,
            use_time_stamps = [0,19],
            output_dir: str = "./ouput",
            learning_rate: float = 0.001
            ):
        if len(labels_list) != 4:
            raise ValueError(f"Invalid list length: {len(labels_list)}. Required length is 4.")
        
        self.labels_list = labels_list
        self.units_list = units_list
        self.output_dir = output_dir
        self.normalization = normalization

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
            normalization=normalization,
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
            normalization=normalization,
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


    def loadWeights(self, weightsFile):
        self.model.load_weights(weightsFile)


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
        self.df = pd.DataFrame(p_test,columns=[self.labels_list[0],'M11',self.labels_list[1],'M22',self.labels_list[2],'M33',self.labels_list[3],'M44','M21','M31','M32','M41','M42','M43'])
        self.df[self.labels_list[0]] *= self.normalization[0]
        self.df[self.labels_list[1]] *= self.normalization[1]
        self.df[self.labels_list[2]] *= self.normalization[2]
        self.df[self.labels_list[3]] *= self.normalization[3]

        # stores all true values in same matrix as xtrue, ytrue, etc.
        self.df[self.labels_list[0]+'true'] = complete_truth[:,0]*self.normalization[0]
        self.df[self.labels_list[1]+'true'] = complete_truth[:,1]*self.normalization[1]
        self.df[self.labels_list[2]+'true'] = complete_truth[:,2]*self.normalization[2]
        self.df[self.labels_list[3]+'true'] = complete_truth[:,3]*self.normalization[3]
        self.df['M11'] = minval+tf.math.maximum(self.df['M11'], 0)
        self.df['M22'] = minval+tf.math.maximum(self.df['M22'], 0)
        self.df['M33'] = minval+tf.math.maximum(self.df['M33'], 0)
        self.df['M44'] = minval+tf.math.maximum(self.df['M44'], 0)

        self.df['sigma'+self.labels_list[0]] = abs(self.df['M11'])
        self.df['sigma'+self.labels_list[1]] = np.sqrt(self.df['M21']**2 + self.df['M22']**2)
        self.df['sigma'+self.labels_list[2]] = np.sqrt(self.df['M31']**2+self.df['M32']**2+self.df['M33']**2)
        self.df['sigma'+self.labels_list[3]] = np.sqrt(self.df['M41']**2+self.df['M42']**2+self.df['M43']**2+self.df['M44']**2)

        # calculates residuals for x, y, cotA, cotB
        self.residuals = np.empty(shape=(4, len(self.df["M11"])))
        self.residuals[0] = self.df[self.labels_list[0]+'true'] - self.df[self.labels_list[0]]
        self.residuals[1] = self.df[self.labels_list[1]+'true'] - self.df[self.labels_list[1]]
        self.residuals[2] = self.df[self.labels_list[2]+'true'] - self.df[self.labels_list[2]]
        self.residuals[3] = self.df[self.labels_list[3]+'true'] - self.df[self.labels_list[3]]

        mean0, std0 = (np.mean(np.abs(self.residuals[0])),np.std(np.abs(self.residuals[0])))
        print(f"Mean and standard deviation of residuals for {self.labels_list[0]}: ({mean0},{std0})")

        mean1, std1 = (np.mean(np.abs(self.residuals[1])),np.std(np.abs(self.residuals[1])))
        print(f"Mean and standard deviation of residuals for {self.labels_list[1]}: ({mean1},{std1})")

        mean2, std2 = (np.mean(np.abs(self.residuals[2])),np.std(np.abs(self.residuals[2])))
        print(f"Mean and standard deviation of residuals for {self.labels_list[2]}: ({mean2},{std2})")

        mean3, std3 = (np.mean(np.abs(self.residuals[3])),np.std(np.abs(self.residuals[3])))
        print(f"Mean and standard deviation of residuals for {self.labels_list[3]}: ({mean3},{std3})")
        

    def plotResiduals(self):
        for i in range(4):
            sns.regplot(x=self.df[f'{self.labels_list[i]}true'], y=self.residuals[i], x_bins=np.linspace(-self.normalization[i],self.normalization[i],50), fit_reg=None, marker='.',color='b')
            plt.xlabel(f'True {self.labels_list[i]} {self.units_list[i]}')
            plt.ylabel(f'{self.labels_list[i]} residuals {self.units_list[i]}')
            plt.title(f"{self.labels_list[i]} residuals")
            plt.show()

            plt.hist(self.residuals[i], bins = 25, align='mid', density=True, histtype='step', color='b')
            plt.xlabel(f'{self.labels_list[i]} residuals {self.units_list[i]}')
            plt.title(f"{self.labels_list[i]} residuals")
            plt.show()