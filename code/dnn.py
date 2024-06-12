import random
from datetime import datetime
import time
import sys

import numpy as np
import pandas as pd
import pickle as pc

import tensorflow as tf
import tensorflow.keras as kr
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, AlphaDropout, BatchNormalization
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.layers import LeakyReLU, PReLU
import tensorflow.keras.backend as K


from auxiliary import MAE, sMAPE, RMSE, MAPE, rMAE
from preprocessing import read_data, DataScaler, scaling, setup_directories


########################### TRAIN/VALIDATION/TEST SPLIT ##############################

def update_datasets(Xtrain, Ytrain, Xval, Yval, Xtest, Ytest):
    first_val_X, first_val_Y = Xval[0], Yval[0]
    Xtrain = np.vstack([Xtrain[1:], first_val_X])
    Ytrain = np.vstack([Ytrain[1:], first_val_Y])
    first_test_X, first_test_Y = Xtest[0], Ytest[0]
    Xval = np.vstack([Xval[1:], first_test_X])
    Yval = np.vstack([Yval[1:], first_test_Y])
    Xtest, Ytest = Xtest[1:], Ytest[1:]
    return Xtrain, Ytrain, Xval, Yval, Xtest, Ytest

def build_and_split_XYs(dfTrain, features, shuffle_train, n_exog_panel_inputs,n_exog_single_inputs, dfTest=None, percentage_val=0.25,
                        date_test=None, hyperoptimization=False, data_augmentation=False):
    """Method to buil the X,Y pairs for training/test DNN models using dataframes and a list of
    the selected inputs
    
    Parameters
    ----------
    dfTrain : pandas.DataFrame
        Pandas dataframe containing the training data
    features : dict
        Dictionary that define the selected input features. The dictionary is based on the results
        of a hyperparameter/feature optimization run using the :class:`hyperparameter_optimizer`function
    shuffle_train : bool
        If true, the validation and training datasets are shuffled
    n_exog_panel_inputs : int
        Number of exogenous inputs for the entire panel of 24 hours, i.e. inputs besides historical prices, load
    n_exog_single_inputs : int
        Number of exogenous inputs with a single value per day, i.e. gas
    dfTest : pandas.DataFrame
        Pandas dataframe containing the test data
    percentage_val : TYPE, optional
        Percentage of data to be used for validation
    date_test : None, optional
        If given, then the test dataset is only built for that date
    hyperoptimization : bool, optional
        Description
    data_augmentation : bool, optional
        Description
    
    Returns
    -------
    list
        A list ``[Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, indexTest]`` that contains the X, Y pairs 
        for training, validation, and testing, as well as the date index of the test dataset
    """

    # Checking that the first index in the dataframes corresponds with the hour 00:00 
    if dfTrain.index[0].hour != 1 or dfTest.index[0].hour != 1:
        print('Problem with the index')

        
    # Calculating the number of input features
    n_features = features['In: Day'] + \
        24 * features['In: Price D-1'] + 24 * features['In: Price D-2'] + \
        24 * features['In: Price D-3'] + 24 * features['In: Price D-7']

    for n_ex in range(1, n_exog_panel_inputs + 1):

        n_features += 24 * features['In: 24Exog-' + str(n_ex) + ' D'] + \
                     24 * features['In: 24Exog-' + str(n_ex) + ' D-1'] + \
                     24 * features['In: 24Exog-' + str(n_ex) + ' D-7']
    
    for n_s_ex in range(1+n_exog_panel_inputs, n_exog_panel_inputs+ n_exog_single_inputs + 1):

        n_features += features['In: S Exog-' + str(n_s_ex) + ' D-0'] + \
                     features['In: S Exog-' + str(n_s_ex) + ' D-1'] + \
                    features['In: S Exog-' + str(n_s_ex) + ' D-7']

    # Extracting the predicted dates for testing and training. We leave the first week of data
    # out of the prediction as we the maximum lag can be one week
    # In addition, if we allow training using all possible predictions within a day, we consider
    # a indexTrain per starting hour of prediction
    
    # We define the potential time indexes that have to be forecasted in training
    # and testing
    indexTrain = dfTrain.loc[dfTrain.index[0] + pd.Timedelta(weeks=1):].index

    if date_test is None:
        # indexTest = dfTest.loc[dfTest.index[0] + pd.Timedelta(weeks=1):].index
        indexTest = dfTest.loc[dfTest.index[0]+ pd.Timedelta(weeks=1):].index
        indexTest_temp = dfTest.loc[dfTest.index[0]:].index

    else:
        indexTest = dfTest.loc[date_test:date_test + pd.Timedelta(hours=23)].index
    

    # We extract the prediction dates/days. For the regular case, 
    # it is just the index resample to 24 so we have a date per day.
    # For the multiple datapoints per day, we have as many dates as indexs
    if data_augmentation:
        predDatesTrain = indexTrain.round('1H')
    else:
        predDatesTrain = indexTrain.round('1H')[::24]            
            
    predDatesTest = indexTest_temp.round('1H')[::24] + pd.Timedelta(weeks=1) 

    # We create dataframe where the index is the time where a prediction is made
    # and the columns is the horizons of the prediction

    indexTrain = pd.DataFrame(index=predDatesTrain)
    indexTest = pd.DataFrame(index=predDatesTest)
    for hour in range(24):
        indexTrain['h' + str(hour)] = None
        indexTest['h' + str(hour)] = None
        indexTrain.loc[:, 'h' + str(hour)] = indexTrain.index + pd.Timedelta(hours=hour)
        indexTest.loc[:, 'h' + str(hour)] = indexTest.index + pd.Timedelta(hours=hour)

    # If we consider 24 predictions per day, the last 23 indexs cannot be used as there is not data
    # for that horizon:
    if data_augmentation:
        indexTrain = indexTrain.iloc[:-23]
    
    # Preallocating in memory the X and Y arrays    

    Xtrain = np.zeros([indexTrain.shape[0], n_features])
    Xtest = np.zeros([indexTest.shape[0], n_features])
    Ytrain = np.zeros([indexTrain.shape[0], 24])
    Ytest = np.zeros([indexTest.shape[0], 24])

    # Adding the day of the week as a feature if needed
    indexFeatures = 0
    if features['In: Day']:
        # For training, we assume the day of the week is a continuous variable.
        # So monday at 00 is 1. Monday at 1h is 1.04, Tuesday at 2h is 2.08, etc.
        Xtrain[:, 0] = indexTrain.index.dayofweek + indexTrain.index.hour / 24
        Xtest[:, 0] = indexTest.index.dayofweek            
        indexFeatures += 1
    
    # For each possible horizon
    for hour in range(24):
        # For each possible past day where prices can be included
        for past_day in [1, 2, 3, 7]:

            # We define the corresponding past time indexs 
            pastIndexTrain = pd.to_datetime(indexTrain.loc[:, 'h' + str(hour)].values) - \
                pd.Timedelta(hours=24 * past_day)
            pastIndexTest = pd.to_datetime(indexTest.loc[:, 'h' + str(hour)].values) - \
                pd.Timedelta(hours=24 * past_day) 


            # We include feature if feature selection indicates it
            if features['In: Price D-' + str(past_day)]:
                Xtrain[:, indexFeatures] = dfTrain[dfTrain.index.isin(pastIndexTrain)]['Price']
                # Xtest[:, indexFeatures] = dfTest[dfTest.index.isin(pastIndexTest)]['Price'] 
                price_data = dfTest[dfTest.index.isin(pastIndexTest)]['Price']
                Xtest[:, indexFeatures] = np.concatenate([price_data , np.zeros(Xtest.shape[0] - len(price_data))])
                indexFeatures += 1


    
    # For each possible horizon
    for hour in range(24):
        # For each possible past day where exogeneous can be included
        for past_day in [1, 7]:

            # We define the corresponding past time indexs 
            pastIndexTrain = pd.to_datetime(indexTrain.loc[:, 'h' + str(hour)].values) - \
                pd.Timedelta(hours=24 * past_day)
            pastIndexTest = pd.to_datetime(indexTest.loc[:, 'h' + str(hour)].values) - \
                pd.Timedelta(hours=24 * past_day)
            
            # For each of the exogenous inputs we include feature if feature selection indicates it
            for exog in range(1, n_exog_panel_inputs + 1):
                if features['In: 24Exog-' + str(exog) + ' D-' + str(past_day)]:
                    Xtrain[:, indexFeatures] = dfTrain[dfTrain.index.isin(pastIndexTrain)][ 'Exogenous ' + str(exog)]                    
                    # Xtest[:, indexFeatures] = dfTest[dfTest.index.isin(pastIndexTest)]['Exogenous ' + str(exog)]
                    ex_data = dfTest[dfTest.index.isin(pastIndexTest)]['Exogenous ' + str(exog)]
                    Xtest[:, indexFeatures] = np.concatenate([ex_data ,np.zeros(Xtest.shape[0] - len(ex_data))])
                    indexFeatures += 1

        # For each of the exogenous inputs we include feature if feature selection indicates it
        for exog in range(1, n_exog_panel_inputs + 1):
            # Adding exogenous inputs at day D
            if features['In: 24Exog-' + str(exog) + ' D']:
                futureIndexTrain = pd.to_datetime(indexTrain.loc[:, 'h' + str(hour)].values)
                futureIndexTest = pd.to_datetime(indexTest.loc[:, 'h' + str(hour)].values)
                Xtrain[:, indexFeatures] = dfTrain[dfTrain.index.isin(futureIndexTrain)]['Exogenous ' + str(exog)]        
                # Xtest[:, indexFeatures] = dfTest[dfTest.index.isin(futureIndexTest)]['Exogenous ' + str(exog)] 
                ex_data = dfTest[dfTest.index.isin(futureIndexTest)]['Exogenous ' + str(exog)] 
                Xtest[:, indexFeatures] = np.concatenate([ex_data ,np.zeros(Xtest.shape[0] - len(ex_data))])
                indexFeatures += 1

    for past_day in [0, 1, 7]:
        for s_exog in range(1+n_exog_panel_inputs,n_exog_panel_inputs+ n_exog_single_inputs + 1):
            # Adding single value exogenous inputs at day D
            if features['In: S Exog-' + str(s_exog) + ' D-' + str(past_day)]:
                futureIndexTrain = pd.to_datetime(indexTrain.loc[:, 'h0'].values) - \
                    pd.Timedelta(hours=24 * past_day)
                futureIndexTest = pd.to_datetime(indexTest.loc[:, 'h0'].values) - \
                    pd.Timedelta(hours=24 * past_day)
                Xtrain[:, indexFeatures] = dfTrain[dfTrain.index.isin(futureIndexTrain)]['Exogenous ' + str(s_exog)]    
                # Xtest[:, indexFeatures] = dfTest[dfTest.index.isin(futureIndexTest)]['Exogenous ' + str(s_exog)]
                s_ex_data = dfTest[dfTest.index.isin(futureIndexTest)]['Exogenous ' + str(s_exog)]
                Xtest[:, indexFeatures] = np.concatenate([s_ex_data ,np.zeros(Xtest.shape[0] - len(s_ex_data))])
                indexFeatures += 1


    # Extracting the predicted values Y
    for hour in range(24):
        futureIndexTrain = pd.to_datetime(indexTrain.loc[:, 'h' + str(hour)].values)
        futureIndexTest = pd.to_datetime(indexTest.loc[:, 'h' + str(hour)].values)

        Ytrain[:, hour] = dfTrain[dfTrain.index.isin(futureIndexTrain)]['Price']        
        # Ytest[:, hour] = dfTest[dfTest.index.isin(futureIndexTest)]['Price'] 
        y_test_data = dfTest[dfTest.index.isin(futureIndexTest)]['Price']
        Ytest[:, hour] = np.concatenate([y_test_data ,np.zeros(Ytest.shape[0] - len(y_test_data))])

    # Redefining indexTest to return only the dates at which a prediction is made
    indexTest = indexTest.index


    if shuffle_train:
        nVal = int(percentage_val * Xtrain.shape[0])

        if hyperoptimization:
            # We fixed the random shuffle index so that the validation dataset does not change during the
            # hyperparameter optimization process
            np.random.seed(7)

        # We shuffle the data per week to avoid data contamination
        index = np.arange(Xtrain.shape[0])
        index_week = index[::7]
        np.random.shuffle(index_week)
        index_shuffle = [ind + i for ind in index_week for i in range(7) if ind + i in index]

        Xtrain = Xtrain[index_shuffle]
        Ytrain = Ytrain[index_shuffle]

    else:
        nVal = int(percentage_val * Xtrain.shape[0])
    nTrain = Xtrain.shape[0] - nVal # complements nVal
    
    Xval = Xtrain[nTrain:] # last nVal obs
    Xtrain = Xtrain[:nTrain] # first nTrain obs
    Yval = Ytrain[nTrain:]
    Ytrain = Ytrain[:nTrain]
    # Xtest = Xtest[:-7]
    # Ytest = Ytest[:-7]
    indexTest = indexTest[:-7]

    return Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, indexTest


"""
Classes and functions to implement the DNN model for electricity price forecasting. The module does not
include the hyperparameter optimization functions; these are included in the module
_dnn_hyperopt.py
"""

class DNNModel(object):

    """Basic DNN model based on keras and tensorflow. 
    
    The model can be used standalone to train and predict a DNN using its fit/predict methods.
    However, it is intended to be used within the :class:`hyperparameter_optimizer` method
    and the :class:`DNN` class. The former obtains a set of best hyperparameter using the :class:`DNNModel` class. 
    The latter employes the set of best hyperparameters to recalibrate a :class:`DNNModel` object
    and make predictions.
    
    Parameters
    ----------
    neurons : list
        List containing the number of neurons in each hidden layer. E.g. if ``len(neurons)`` is 2,
        the DNN model has an input layer of size ``n_features``, two hidden layers, and an output 
        layer of size ``outputShape``.
    n_features : int
        Number of input features in the model. This number defines the size of the input layer.
    outputShape : int, optional
        Default number of output neurons. It is 24 as it is the default in most day-ahead markets.
    dropout : float, optional
        Number between [0, 1] that selects the percentage of dropout. A value of 0 indicates
        no dropout.
    batch_normalization : bool, optional
        Boolean that selects whether batch normalization is considered.
    lr : float, optional
        Learning rate for optimizer algorithm. If none provided, the default one is employed
        (see the `keras documentation <https://keras.io/>`_ for the default learning rates of each algorithm).
    verbose : bool, optional
        Boolean that controls the logs. If set to true, a minimum amount of information is 
        displayed.
    epochs_early_stopping : int, optional
        Number of epochs used in early stopping to stop training. When no improvement is observed
        in the validation dataset after ``epochs_early_stopping`` epochs, the training stops.
    scaler : :class:`epftoolbox.data.DataScaler`, optional
        Scaler object to invert-scale the output of the neural network if the neural network
        is trained with scaled outputs.
    loss : str, optional
        Loss to be used when training the neural network. Any of the regression losses defined in 
        keras can be used.
    optimizer : str, optional
        Name of the optimizer when training the DNN. See the `keras documentation <https://keras.io/>`_ 
        for a list of optimizers.
    activation : str, optional
        Name of the activation function in the hidden layers. See the `keras documentation <https://keras.io/>`_ for a list
        of activation function.
    initializer : str, optional
        Name of the initializer function for the weights of the neural network. See the 
        `keras documentation <https://keras.io/>`_ for a list of initializer functions.
    regularization : None, optional
        Name of the regularization technique. It can can have three values ``'l2'`` for l2-norm
        regularization, ``'l1'`` for l1-norm regularization, or ``None`` for no regularization .
    lambda_reg : int, optional
        The weight for regulization if ``regularization`` is ``'l2'`` or ``'l1'``.
    """

    
    def __init__(self, neurons, n_features, outputShape=24, dropout=0, batch_normalization=False, lr=None,
                 verbose=False, epochs_early_stopping=40,scaler = None, scale_mean=None, scale_std =None, loss='mae',
                 optimizer='adam', activation='relu', initializer='glorot_uniform',
                 regularization=None, lambda_reg=0):

        self.neurons = neurons
        self.dropout = dropout

        if self.dropout > 1 or self.dropout < 0:
            raise ValueError('Dropout parameter must be between 0 and 1')

        self.batch_normalization = batch_normalization
        self.verbose = verbose
        self.epochs_early_stopping = epochs_early_stopping
        self.n_features = n_features
        self.scaler = scaler
        self.scale_mean = scale_mean
        self.scale_std = scale_std
        self.outputShape = outputShape
        self.activation = activation
        self.initializer = initializer
        self.regularization = regularization
        self.lambda_reg = lambda_reg

        self.model = self._build_model()

        if lr is None:
            opt = 'adam'
        else:
            if optimizer == 'adam':
                opt = kr.optimizers.Adam(learning_rate=lr, clipvalue=10000)
            if optimizer == 'RMSprop':
                opt = kr.optimizers.RMSprop(learning_rate=lr, clipvalue=10000)
            if optimizer == 'adagrad':
                opt = kr.optimizers.Adagrad(learning_rate=lr, clipvalue=10000)
            if optimizer == 'adadelta':
                opt = kr.optimizers.Adadelta(learning_rate=lr, clipvalue=10000)

        self.model.compile(loss=loss, optimizer=opt)

    def _reg(self, lambda_reg):
        """Internal method to build an l1 or l2 regularizer for the DNN
        
        Parameters
        ----------
        lambda_reg : float
            Weight of the regularization
        
        Returns
        -------
        tensorflow.keras.regularizers.L1L2
            The regularizer object
        """
        if self.regularization == 'l2':
            return l2(lambda_reg)
        if self.regularization == 'l1':
            return l1(lambda_reg)
        else:
            return None

    def _build_model(self):
        """Internal method that defines the structure of the DNN
        
        Returns
        -------
        tensorflow.keras.models.Model
            A neural network model using keras and tensorflow
        """
        inputShape = (None, self.n_features)

        past_data = Input(batch_shape=inputShape)

        past_Dense = past_data
        if self.activation == 'selu':
            self.initializer = 'lecun_normal'

        for k, neurons in enumerate(self.neurons):

            if self.activation == 'LeakyReLU':
                past_Dense = Dense(neurons, activation='linear', batch_input_shape=inputShape,
                                   kernel_initializer=self.initializer,
                                   kernel_regularizer=self._reg(self.lambda_reg))(past_Dense)
                past_Dense = LeakyReLU(alpha=.001)(past_Dense)

            elif self.activation == 'PReLU':
                past_Dense = Dense(neurons, activation='linear', batch_input_shape=inputShape,
                                   kernel_initializer=self.initializer,
                                   kernel_regularizer=self._reg(self.lambda_reg))(past_Dense)
                past_Dense = PReLU()(past_Dense)

            else:
                past_Dense = Dense(neurons, activation=self.activation,
                                   batch_input_shape=inputShape,
                                   kernel_initializer=self.initializer,
                                   kernel_regularizer=self._reg(self.lambda_reg))(past_Dense)

            if self.batch_normalization:
                past_Dense = BatchNormalization()(past_Dense)

            if self.dropout > 0:
                if self.activation == 'selu':
                    past_Dense = AlphaDropout(self.dropout)(past_Dense)
                else:
                    past_Dense = Dropout(self.dropout)(past_Dense)

        output_layer = Dense(self.outputShape, kernel_initializer=self.initializer,
                             kernel_regularizer=self._reg(self.lambda_reg))(past_Dense)
        model = Model(inputs=[past_data], outputs=[output_layer])

        return model

    def _obtain_metrics(self, X, Y):
        """Internal method to update the metrics used to train the network
        
        Parameters
        ----------
        X : numpy.array
            Input array for evaluating the model
        Y : numpy.array
            Output array for evaluating the model
        
        Returns
        -------
        list
            A list containing the metric based on the loss of the neural network and a second metric
            representing the MAE of the DNN
        """
        error = self.model.evaluate(X, Y, verbose=0)
        Ybar = self.model.predict(X, verbose=0)

        if self.scaler is not None:
            if len(Y.shape) == 1:
                Y = Y.reshape(-1, 1)
                Ybar = Ybar.reshape(-1, 1)
            Y = self.scaler.inverse_transform(Y)
            Ybar = self.scaler.inverse_transform(Ybar)
        # if self.scale_mean != None and self.scale_std != None:
        #     if len(Y.shape) == 1:
        #         Y = Y.reshape(-1, 1)
        #         Ybar = Ybar.reshape(-1, 1)
        #     Y = (Y * self.scale_std) + self.scale_mean
        #     Ybar = (Ybar * self.scale_std) + self.scale_mean


        mae = MAE(Y, Ybar)

        return error, np.mean(mae)

    def _display_info_training(self, bestError, bestMAE, countNoImprovement):
        """Internal method that displays useful information during training
        
        Parameters
        ----------
        bestError : float
            Loss of the neural network in the validation dataset
        bestMAE : float
            MAE of the neural network in the validation dataset
        countNoImprovement : int
            Number of epochs in the validation dataset without improvements
        """
        print(" Best error:\t\t{:.1e}".format(bestError))
        print(" Best MAE:\t\t{:.2f}".format(bestMAE))                
        print(" Epochs without improvement:\t{}\n".format(countNoImprovement))


    def fit(self, trainX, trainY, valX, valY):
        """Method to estimate the DNN model.
        
        Parameters
        ----------
        trainX : numpy.array
            Inputs fo the training dataset.
        trainY : numpy.array
            Outputs fo the training dataset.
        valX : numpy.array
            Inputs fo the validation dataset used for early-stopping.
        valY : numpy.array
            Outputs fo the validation dataset used for early-stopping.
        """

        # Variables to control training improvement
        bestError = 1e20
        bestMAE = 1e20

        countNoImprovement = 0

        bestWeights = self.model.get_weights()

        for epoch in range(1000):
            start_time = time.time()

            self.model.fit(trainX, trainY, batch_size=192,
                           epochs=1, verbose=False, shuffle=True)

            # Updating epoch metrics and displaying useful information
            if self.verbose:
                print("\nEpoch {} of {} took {:.3f}s".format(epoch + 1, 1000,
                                                             time.time() - start_time))

            # Calculating relevant metrics to perform early-stopping
            valError, valMAE = self._obtain_metrics(valX, valY)

            # Early-stopping
            # Checking if current validation metrics are better than best so far metrics.
            # If the network does not improve, we stop.
            # If it improves, the optimal weights are saved
            if valError < bestError:
                countNoImprovement = 0
                bestWeights = self.model.get_weights()

                bestError = valError
                bestMAE = valMAE
                if valMAE < bestMAE:
                    bestMAE = valMAE

            elif valMAE < bestMAE:
                countNoImprovement = 0
                bestWeights = self.model.get_weights()
                bestMAE = valMAE
            else:
                countNoImprovement += 1

            if countNoImprovement >= self.epochs_early_stopping:
                if self.verbose:
                    self._display_info_training(bestError, bestMAE, countNoImprovement)
                break

            # Displaying final information
            if self.verbose:
                self._display_info_training(bestError, bestMAE, countNoImprovement)

        # After early-stopping, the best weights are set in the model
        self.model.set_weights(bestWeights)

    def predict(self, X):
        """Method to make a prediction after the DNN is trained.
        
        Parameters
        ----------
        X : numpy.array
            Input to the DNN. It has to be of size *[n, n_features]* where *n* can be any 
            integer, and *n_features* is the attribute of the DNN representing the number of
            input features.
        
        Returns
        -------
        numpy.array
            Output of the DNN after making the prediction.
        """

        Ybar = self.model.predict(X, verbose=0)
        return Ybar

    def clear_session(self):
        """Method to clear the tensorflow session. 

        It is used in the :class:`DNN` class during recalibration to avoid RAM memory leakages.
        In particular, if the DNN is retrained continuosly, at each step tensorflow slightly increases 
        the total RAM usage.

        """

        K.clear_session()


def test_dnn(dataset_file, 
             calibration_windows, 
             years_test, 
             neurons, 
             dropout, 
             batch_normalization, 
             learningrate, 
             reg_lambda, 
             epochs_early_stopping, 
             scaling_type, 
             Xscaling_type, 
             activation_fn, 
             initializer, 
             input_hyperparameters,
             n_exog_single_inputs):
    
    model_name = 'dnn'
    results_file_path = f'forecasts/{model_name}'
    setup_directories(results_file_path = results_file_path)
    for cal_window in calibration_windows:

        seed_value = 42
        random.seed(seed_value)
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)
        loss_arr, val_smape_arr  = [], []
        Ytrue, Yp = [], []
        times_in_seconds = []
        formatted_string = datetime.now().strftime('_%H_%d%m') + f'_{activation_fn}_Epoch_{epochs_early_stopping}_LR_{str(learningrate)[2:]}_REG_{str(reg_lambda)[2:]}_scale_{scaling_type}'


        df_train, df_test = read_data(dataset_path = dataset_file ,window = cal_window, years_test=years_test)

        Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, indexTest = build_and_split_XYs(dfTrain=df_train, dfTest=df_test, 
                                                                                features=input_hyperparameters, 
                                                                                shuffle_train=False, 
                                                                                n_exog_panel_inputs=1,
                                                                                n_exog_single_inputs=n_exog_single_inputs)
        
        for day in range(len(Xtest) - 15):
            Xtrain_og, Ytrain_og, Xval_og, Yval_og, Xtest_og, Ytest_og = Xtrain, Ytrain, Xval, Yval, Xtest, Ytest
            current_Xtest = Xtest[7:8]
            current_Ytest = Ytest[7:8]

            if Xscaling_type in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
                Xdatasets, Xscaler_object = scaling([Xtrain, Xval, current_Xtest], normalize = scaling_type)
                [Xtrain, Xval, current_Xtest] = Xdatasets
            else:
                Xscaler_object = None
            if scaling_type in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
                Ydatasets, scaler_object = scaling([Ytrain, Yval], normalize = scaling_type)
                [Ytrain,Yval] = Ydatasets
            else:
                scaler_object = None

            print("Xtrain", Xtrain)
            print("Ytrain", Ytrain)
            print("Xtest", current_Xtest)

            start_time = time.time()
            forecaster = DNNModel(neurons=neurons,
                                n_features=Xtrain.shape[-1], 
                                dropout=dropout, 
                                batch_normalization=batch_normalization, 
                                lr=learningrate, 
                                verbose=False,
                                optimizer='adam', 
                                activation=activation_fn,
                                epochs_early_stopping=epochs_early_stopping, 
                                initializer=initializer,
                                scaler=scaler_object,
                                # scale_mean = price_mean,
                                # scale_std = price_std,
                                loss='mae',
                                regularization='l1', 
                                lambda_reg=reg_lambda)
            

            # Retrain the model with the updated datasets
            forecaster.fit(Xtrain, Ytrain, valX=Xval, valY=Yval)
            Yp_val = forecaster.predict(Xval).squeeze()

            # Forecast for the next day using the first day in the updated test set
            daily_prediction = forecaster.predict(current_Xtest)  # Use only the first row of Xtest
            end_time = time.time()
            elapsed_time = end_time - start_time
            times_in_seconds.append(elapsed_time)


            if scaling_type in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
                Ytrue_val = Yval_og
                Yp_val = scaler_object.inverse_transform(Yp_val)
            else:
                Ytrue_val = Yval
                Yp_val = Yp_val


            mae_validation = np.mean(MAE(Ytrue_val, Yp_val))
            smape_validation = np.mean(sMAPE(Ytrue_val, Yp_val)) * 100
            loss_arr.append(mae_validation)
            val_smape_arr.append(smape_validation)




            # Output daily prediction if needed
            print("Day", day + 1, "Prediction:", daily_prediction, type(daily_prediction), daily_prediction.shape)


            if scaling_type in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
                daily_prediction = scaler_object.inverse_transform(daily_prediction).squeeze()


            print("Day", day + 1, "Prediction:", daily_prediction, type(daily_prediction), daily_prediction.shape)

            Yp.append(daily_prediction)
            Ytrue.append(current_Ytest)
            
            # Update the datasets after making the day's prediction
            Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = update_datasets(Xtrain_og, Ytrain_og, Xval_og, Yval_og, Xtest_og, Ytest_og)
            forecaster.clear_session()
            Ydatasets,Xdatasets,scaler_object, Xscaler_object,  = None, None, None, None


        Ytrue = np.array(Ytrue).flatten()
        Yp = np.array(Yp).flatten()

        print('Ytrue : ', Ytrue)
        print('Yp: ', Yp)

        return_values = pd.DataFrame({'Ytrue': Ytrue,'Yp': Yp})
        return_values.to_csv(f'{results_file_path}/DNN_2L_CAL_{cal_window}_{formatted_string}.csv', index=False)

        maeTest = np.mean(MAE(Ytrue, Yp)) 
        smape_test = np.mean(sMAPE(Ytrue, Yp)) * 100
        rmse_test = np.mean(RMSE(Ytrue, Yp))
        mape_test = np.mean(MAPE(Ytrue, Yp)) * 100
        return_values_test = {'MAE Test': maeTest, 'sMAPE Test': smape_test, 'RMSE Test': rmse_test, 'MAPE Test': mape_test}
        return_values_val = pd.DataFrame({'Val Loss MAE': loss_arr,'Val SMAPE': val_smape_arr})

        print("Validation Metrics", return_values_val)
        print("Test Metrics", return_values_test)


        # Open a file in write mode
        with open(f'{results_file_path}/metrics_{cal_window}_{formatted_string}.txt', 'w') as file:
            file.write("\nTest Metrics\n")
            for key, value in return_values_test.items():
                file.write(f"{key}: {value}\n")
            file.write("\nTime in Seconds\n")
            file.write(", ".join(str(time) for time in times_in_seconds) + "\n")

                
        print(f"Metrics have been written to metrics_{cal_window}_{formatted_string}.txt")


        K.clear_session()
        print(f"Completed processing for window size {cal_window}")
