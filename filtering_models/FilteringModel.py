import os

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import roc_curve, auc

from qkeras import *
import OptimizedDataGenerator4 as ODG
import time

# python based
from pathlib import Path
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
plt.rcParams['figure.dpi'] = 300
from tqdm import tqdm
import numpy as np

# custom code
from models import *

def scheduler(epoch, lr):
    if epoch == 100 or epoch == 130 or epoch == 160:
        return lr/2
    else:
        return lr
    
class FilteringModel:

    def __init__(self, 
                 learning_rate: float = 0.001, 
                 x_feature_description: list = ['x_size', 'y_size', 'y_local', 'z_global'],
                 tf_records_dir: str = None,
                 model_number: int = None,
                 ):

        self.history = None


        training_dir = Path(tf_records_dir, 'tfrecords_train')
        validation_dir = Path(tf_records_dir, 'tfrecords_validation')

        self.training_generator = ODG.OptimizedDataGenerator(load_records=True, tf_records_dir=training_dir, x_feature_description=x_feature_description)

        self.validation_generator = ODG.OptimizedDataGenerator(load_records=True, tf_records_dir=validation_dir, x_feature_description=x_feature_description)
        self.validation_generator_with_hit_time = ODG.OptimizedDataGenerator(load_records=True, tf_records_dir=validation_dir, x_feature_description=["adjusted_hit_time", "adjusted_hit_time_30ps_gaussian", "adjusted_hit_time_60ps_gaussian"])

        self.x_features=x_feature_description

        if model_number is None:
            self.createModel()
        else:
            self.loadModel(model_number)

        self.compileModel(learning_rate=learning_rate)

        self.residuals = None


    def createModel(self, layer1=3, layer2=3, numLayers=1):
        start_time = time.time()
        self.model=CreateClassificationModel(self.x_features, layer1, layer2, numLayers)
        self.model.summary()
        print("--- Model create and compile %s seconds ---" % (time.time() - start_time))


    def compileModel(self, learning_rate = 0.001):
        print(f"Compiling model with learning rate: {learning_rate}")
        self.learning_rate = learning_rate
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['binary_accuracy'])
    

    def loadWeights(self, weightsFile):
        self.model.load_weights(weightsFile)


    def loadModel(self, model_number):
        self.modelName=f"Model {model_number}"
        file_path = Path(f'./Model_{model_number}.keras').resolve()
        self.model=tf.keras.models.load_model(file_path, compile=False)


    def saveModel(self, model_number, overwrite=False):
        file_path = Path(f'./Model_{model_number}.keras').resolve()
        if not overwrite and os.path.exists(file_path):
            raise Exception("Model exists. To overwrite existing saved model, set overwrite to True.")
        self.modelName=f"Model {model_number}"
        self.model.save(file_path)


    def loadWeights(self, fileName):
        filePath = Path(f'./weights_{fileName}.hdf5').resolve()
        self.model.load_weights(filePath)


    def saveWeights(self, fileName):
        filePath = Path(f'./weights_{fileName}.hdf5').resolve()
        self.model.save_weights(filePath)


    def runTraining(self, epochs=50, early_stopping=True, save_all_weights=False, schedule_lr=True, save_prev_history=False):
        early_stopping_patience = 10

        # launch quick training once gpu is available
        es = EarlyStopping(
        patience=early_stopping_patience,
        restore_best_weights=True
        )
    
        # checkpoint path
        checkpoint_filepath = Path("./weights", 'epoch.{epoch:02d}-t{loss:.2f}-v{val_loss:.2f}.weights.h5').resolve()
        mcp = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            save_best_only=True,
        )
        
        # learning rate scheduler
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

        callbacks = []
        if early_stopping:
            callbacks.append(es)
        if save_all_weights:
            callbacks.append(mcp)
        if schedule_lr:
            callbacks.append(lr_scheduler)

        # train
        prev_history = self.history

        self.history = self.model.fit(x=self.training_generator,
                        validation_data=self.validation_generator,
                        callbacks=callbacks,
                        epochs=epochs,
                        shuffle=False,
                        verbose=1)

        print(type(self.history))

        if save_prev_history and prev_history is not None:
            self.history += prev_history # check if this works ...

        self.residuals = None

        self.plotTraining() 

        self.labels=None


    def plotTraining(self):
        fig, ax = plt.subplots(ncols=1, nrows=2, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        ax[0].plot(range(1,len(self.history.history['val_loss'])+1),self.history.history['val_loss'], c='royalblue')
        ax[0].set_ylabel("Validation Loss")
        ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))

        
        ax[1].plot(range(1,len(self.history.history['lr'])+1),self.history.history['lr'], c='seagreen')
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Learning Rate")
        ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[1].yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0e}'.format(x)))


        ax[0].grid(True, linestyle='--', linewidth=0.5, color='gray')
        ax[1].grid(True, linestyle='--', linewidth=0.5, color='gray')

        fig.tight_layout()

        plt.show()


    def checkAccuracy(self, threshold=0.5): 
        if self.prediction is None:
            p_test = self.model.predict(self.validation_generator)
            self.prediction=p_test.flatten()

        if self.labels is None:

            complete_truth = None
            hit_times = None

            for (t1, t2, t3), y in tqdm(self.validation_generator_with_hit_time):
                if complete_truth is None:
                    complete_truth = y
                    hit_times = np.array([t1.numpy().flatten(), t2.numpy().flatten(), t3.numpy().flatten()])
                else:
                    complete_truth = np.concatenate((complete_truth, y), axis=0)
                    hit_times = np.hstack([hit_times, np.array([t1.numpy().flatten(), t2.numpy().flatten(), t3.numpy().flatten()])])
            
            self.hit_times = hit_times.reshape((hit_times.shape[1], hit_times.shape[0]))
            self.labels = np.array(complete_truth).flatten()

        # background regection
        backgroundCount = 0
        rejectedBackgrounds = np.array([0,0])
        # signal efficiency
        acceptedSignals = np.array([0,0])
        signalCount = 0

        mu=2.09e-3
        sigma=36.6e-3

        for l, p, t in zip(self.labels, self.prediction, self.hit_times):
            # background
            if l <= threshold:
                backgroundCount += 1 
                if p <= threshold:
                    rejectedBackgrounds[0] += 1
            else:
                signalCount += 1
                if p > threshold:
                    acceptedSignals[0] += 1

            if l <= threshold:
                if t[1]<mu-3*sigma or t[1]>mu+3*sigma or p <= threshold:
                    rejectedBackgrounds[1] += 1
            else:
                if (t[1]>mu-3*sigma and t[1]<mu+3*sigma) and p > threshold:
                    acceptedSignals[1] += 1

        signalEfficiency = acceptedSignals/signalCount*100
        backgroundRejection = rejectedBackgrounds/backgroundCount*100

        accuracy = (acceptedSignals+rejectedBackgrounds)/(signalCount+backgroundCount)*100

        fractionSignal = signalCount/(signalCount+backgroundCount)*100

        print(f"Overall Accuracy of Neural Network: {round(accuracy[0],2)}%\nFraction of Data that are Signal: {round(fractionSignal,2)}%")

        print(f"Neural Network results without cutting based on hit time: \n")

        print(f"\nSignal Efficiency: {round(signalEfficiency[0],2)}%\nBackground Rejection: {round(backgroundRejection[0],2)}%\n\n")

        print(f"Neural Network results using hit time with 30ps gaussian: \n")

        print(f"\nSignal Efficiency: {round(signalEfficiency[1],2)}%\nBackground Rejection: {round(backgroundRejection[1],2)}%\n\n")

        print(f"Total number of clusters: {signalCount+backgroundCount}")

        print(f"\nTotal number of clusters: {signalCount+backgroundCount}")



    def checkAccuracyTrainingData(self, threshold=0.5):
        p_test = self.model.predict(self.training_generator)

        complete_truth = None
        for _, y in tqdm(self.training_generator):
            if complete_truth is None:
                complete_truth = y
            else:
                complete_truth = np.concatenate((complete_truth, y), axis=0)

        prediction=p_test.flatten()
        labels = complete_truth.flatten()

        # background regection
        backgroundCount = 0
        rejectedBackground = 0
        # signal efficiency
        acceptedSignal = 0
        signalCount = 0

        for l, p in zip(labels, prediction):
            # background
            if l <= threshold:
                backgroundCount += 1
                if p <= threshold:
                    rejectedBackground += 1
            else:
                signalCount += 1
                if p > threshold:
                    acceptedSignal += 1

        signalEfficiency = acceptedSignal/signalCount*100
        backgroundRejection = rejectedBackground/backgroundCount*100

        accuracy = (acceptedSignal+rejectedBackground)/(signalCount+backgroundCount)*100

        fractionSignal = signalCount/(signalCount+backgroundCount)*100

        print(f"\nSignal Efficiency: {round(signalEfficiency,2)}%\nBackground Rejection: {round(backgroundRejection,2)}%\n")

        print(f"Overall Accuracy: {round(accuracy,2)}%\nFraction of Data that are Signal: {round(fractionSignal,2)}%")

        print(f"\nTotal number of clusters: {signalCount+backgroundCount}")


    def countClusters(self):
        complete_truth = None
        for _, y in tqdm(self.training_generator):
            if complete_truth is None:
                complete_truth = y
            else:
                complete_truth = np.concatenate((complete_truth, y), axis=0)

        labelst = complete_truth.flatten()

        complete_truth = None
        for _, y in tqdm(self.validation_generator):
            if complete_truth is None:
                complete_truth = y
            else:
                complete_truth = np.concatenate((complete_truth, y), axis=0)

        labelsv = complete_truth.flatten()
        
        print(f"# of training clusters: {len(labelst)}")
        print(f"# of training clusters: {len(labelsv)}")
    

    def plotTraining(self):
        fig, ax = plt.subplots(ncols=1, nrows=2, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        ax[0].plot(range(1,len(self.history.history['val_binary_accuracy'])+1),self.history.history['val_binary_accuracy'],c='royalblue')
        ax[0].set_ylabel("Validation Binary Accuracy")
        ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))

        
        ax[1].plot(range(1,len(self.history.history['val_binary_accuracy'])+1),self.history.history['learning_rate'],c='seagreen')
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Learning Rate")
        ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[1].yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0e}'.format(x)))

        ax[0].grid(True, linestyle='--', linewidth=0.5, color='gray')
        ax[1].grid(True, linestyle='--', linewidth=0.5, color='gray')

        fig.tight_layout()

        plt.show()

    
    def plotROCcurve(self):
        complete_truth = None
        for _, y in tqdm(self.validation_generator):
            if complete_truth is None:
                complete_truth = y
            else:
                complete_truth = np.concatenate((complete_truth, y), axis=0)
        labels = complete_truth.flatten()

        fig, ax = plt.subplots()

        p_test = self.model.predict(self.validation_generator)
            
        prediction=p_test.flatten()

        fpr_keras, tpr_keras, thresholds_keras = roc_curve(labels, prediction)
        auc_keras = auc(fpr_keras, tpr_keras)

        # get index of threshold=0.5
        selected_threshold = 0.5
        closest = thresholds_keras[min(range(len(thresholds_keras)), key = lambda i: abs(thresholds_keras[i]-selected_threshold))]
        selected_index = np.where(thresholds_keras == closest)[0]

        # get index of best threshold
        temp = tpr_keras-fpr_keras
        max_value = max(temp)
        best_index = np.where(temp == max_value)[0]

        print(f"Optimal threshold: {thresholds_keras[best_index][0]}")

        ax.plot(fpr_keras, tpr_keras, label=f'ROC curve (area = {round(auc_keras,3)})')
        ax.scatter(fpr_keras[selected_index], tpr_keras[selected_index], label='selected threshold: 0.5',c='orange')
        ax.scatter(fpr_keras[best_index], tpr_keras[best_index], label=f'optimal threshold: {round(thresholds_keras[best_index][0],2)}', c='green')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.minorticks_on()
        ax.grid(which='both', color='lightgray', linestyle='--', linewidth=0.5)
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.set_title('ROC curve')
        ax.legend(loc='best')
        plt.show()


def CompareModelROCCurves(models):
    fig, ax = plt.subplots()

    for model in models:
        name=model.modelName

        complete_truth = None
        for _, y in tqdm(model.validation_generator):
            if complete_truth is None:
                complete_truth = y
            else:
                complete_truth = np.concatenate((complete_truth, y), axis=0)
        labels = complete_truth.flatten()

        p_test = model.model.predict(model.validation_generator)
            
        prediction=p_test.flatten()

        fpr_keras, tpr_keras, thresholds_keras = roc_curve(labels, prediction)
        auc_keras = auc(fpr_keras, tpr_keras)

        # get index of best threshold
        temp = tpr_keras-fpr_keras
        max_value = max(temp)
        best_index = np.where(temp == max_value)[0]

        print(f"Optimal threshold for {name}: {thresholds_keras[best_index][0]}")

        ax.plot(fpr_keras, tpr_keras, label=f'{name} (area = {round(auc_keras,3)})')

    ax.plot([0, 1], [0, 1], 'k--')
    ax.minorticks_on()
    ax.grid(which='both', color='lightgray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC curve')
    ax.legend(loc='best')
    plt.show()
