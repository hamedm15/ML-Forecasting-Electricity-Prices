import os
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.robust import mad
from sklearn.metrics import mean_squared_error, r2_score


def read_data(dataset_path,window, years_test=1, begin_test_date=None, end_test_date=None):
    """Function to read and import data from day-ahead electricity markets. 
    
    It receives a ``dataset`` name, and the ``path`` of the folder where datasets are saved. 
    It reads the file ``dataset.csv`` in the ``path`` directory and provides a split between training and
    testing dataset based on the test dates provided.

    It also names the columns of the training and testing dataset to match the requirements of the
    prediction models of the library. Namely, assuming that there are `N` exogenous inputs,
    the columns of the resulting training and testing dataframes are named
    ``['Price', 'Exogenous 1', 'Exogenous 2', ...., 'Exogenous N']``.

    Parameters
    ----------
    path : str, optional
        Path where the datasets are stored or, if they do not exist yet, the path where the datasets 
        are to be stored
    nlayers : int, optional
        Number of hidden layers in the neural network
    dataset : str, optional
        Name of the dataset/market under study. If it is one one of the standard markets, 
        i.e. ``"PJM"``, ``"NP"``, ``"BE"``, ``"FR"``, or ``"DE"``, the dataset is automatically downloaded. If the name
        is different, a dataset with a csv format should be place in the ``path``.
    years_test : int, optional
        Number of years (a year is 364 days) in the test dataset. It is only used if 
        the arguments begin_test_date and end_test_date are not provided.
    begin_test_date : datetime/str, optional
        Optional parameter to select the test dataset. Used in combination with the argument
        ``end_test_date``. If either of them is not provided, the test dataset is built using the 
        ``years_test`` argument. ``begin_test_date`` should either be a string with the following 
        format ``"%d/%m/%Y %H:%M"``, or a datetime object.
    end_test_date : datetime/str, optional
        Optional parameter to select the test dataset. Used in combination with the argument
        ``begin_test_date``. If either of them is not provided, the test dataset is built using the 
        ``years_test`` argument. ``end_test_date`` should either be a string with the following 
        format ``"%d/%m/%Y %H:%M"``, or a datetime object.       
    Returns
    -------
    pandas.DataFrame, pandas.DataFrame
        Training dataset, testing dataset
        
    """
    try:
        data = pd.read_csv(dataset_path, index_col=0)
    except IOError as e:
        raise IOError("%s: %s" % (dataset_path, e.strerror))

    data.index = pd.to_datetime(data.index)

    columns = ['Price']
    n_exogeneous_inputs = len(data.columns) - 1

    for n_ex in range(1, n_exogeneous_inputs + 1):
        columns.append('Exogenous ' + str(n_ex))
        
    data.columns = columns

    # The training and test datasets can be defined by providing a number of years for testing
    # or by providing the init and end date of the test period
    if begin_test_date is None and end_test_date is None:
        number_datapoints = len(data.index)
        number_training_datapoints = number_datapoints - (24 * 364 * years_test)

        # We consider that a year is 52 weeks (364 days) instead of the traditional 365
        # df_train = data.loc[:data.index[0] + pd.Timedelta(hours=number_training_datapoints - 1), :]
        df_train = data.loc[:data.index[0] + pd.Timedelta(hours=number_training_datapoints - 1), :].iloc[-window*24:, :]
        df_test = data.loc[data.index[0] + pd.Timedelta(hours=number_training_datapoints - 168):, :]
    
    else:
        try:
            begin_test_date = pd.to_datetime(begin_test_date, dayfirst=True)
            end_test_date = pd.to_datetime(end_test_date, dayfirst=True)
        except ValueError:
            print("Provided values for dates are not valid")

        if begin_test_date.hour != 0:
            raise Exception("Starting date for test dataset should be midnight") 
        if end_test_date.hour != 23:
            if end_test_date.hour == 0:
                end_test_date = end_test_date + pd.Timedelta(hours=23)
            else:
                raise Exception("End date for test dataset should be at 0h or 23h") 

        print('Test datasets: {} - {}'.format(begin_test_date, end_test_date))
        df_train = data.loc[:begin_test_date - pd.Timedelta(hours=1), :]
        df_test = data.loc[begin_test_date:end_test_date, :]

    return df_train, df_test


########################### SCALING AND PREPROCESSING ##############################

class MedianScaler(object):

    def __init__(self):
        self.fitted = False

    def fit(self, data):

        if len(data.shape)!=2:
            raise IndexError('Error: Provide 2-D array. First dimension is datapoints and' + 
                  ' second features')
            return -1

        self.median = np.median(data, axis=0)
        self.mad = mad(data, axis=0)
        self.fitted = True
        
    def fit_transform(self, data):

        self.fit(data)
        return self.transform(data)
    
    def transform(self, data):

        if not self.fitted:
            print('Error: The scaler has not been yet fitted. Called fit or fit_transform')
            return -1
        if len(data.shape)!=2:
            raise IndexError('Error: Provide 2-D array. First dimension is datapoints and' + 
                  ' second features')
        transformed_data = np.zeros(shape=data.shape)
        for i in range(data.shape[1]):
            transformed_data[:, i] = (data[:, i] - self.median[i]) / self.mad[i]

        return transformed_data

    def inverse_transform(self, data):

        if not self.fitted:
            print('Error: The scaler has not been yet fitted. Called fit or fit_transform')
            return -1

        if len(data.shape)!=2:
            raise IndexError('Error: Provide 2-D array. First dimension is datapoints and' + 
                  ' second features')

        transformed_data = np.zeros(shape=data.shape)

        for i in range(data.shape[1]):
            transformed_data[:, i] = data[:, i] * self.mad[i] + self.median[i] 

        return transformed_data

class InvariantScaler(MedianScaler):

    def __init__(self):
        super()

    def fit(self, data):

        super().fit(data)
        
    def fit_transform(self, data):

        self.fit(data)
        return self.transform(data)
    
    def transform(self, data):

        transformed_data = super().transform(data)
        transformed_data = np.arcsinh(transformed_data)

        return transformed_data

    def inverse_transform(self, data):

        transformed_data = np.sinh(data)
        transformed_data = super().inverse_transform(transformed_data)

        return transformed_data

class DataScaler(object):

    """Class to perform data scaling operations

    The scaling technique is defined by the ``normalize`` parameter which takes one of the 
    following values: 

    - ``'Norm'`` for normalizing the data to the interval [0, 1].

    - ``'Norm1'`` for normalizing the data to the interval [-1, 1]. 

    - ``'Std'`` for standarizing the data to follow a normal distribution. 

    - ``'Median'`` for normalizing the data based on the median as defined in as defined in `here <https://doi.org/10.1109/TPWRS.2017.2734563>`_.

    - ``'Invariant'`` for scaling the data based on the asinh transformation (a variance stabilizing transformations) as defined in `here <https://doi.org/10.1109/TPWRS.2017.2734563>`_. 
    
    This class follows the same syntax of the scalers defined in the 
    `sklearn.preprocessing <https://scikit-learn.org/stable/modules/preprocessing.html>`_ module of the 
    scikit-learn library

    Parameters
    ----------
    normalize : str
        Type of scaling to be performed. Possible values are ``'Norm'``, ``'Norm1'``, ``'Std'``, 
        ``'Median'``, or ``'Invariant'``
    """
    
    def __init__(self, normalize):

        if normalize == 'Norm':
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif normalize == 'Norm1':
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        elif normalize == 'Std':
            self.scaler = StandardScaler()
        elif normalize == 'Median':
            self.scaler = MedianScaler()
        elif normalize == 'Invariant':
            self.scaler = InvariantScaler()        

    def fit_transform(self, dataset):
        """Method that estimates an scaler object using the data in ``dataset`` and scales the data in  ``dataset``
        
        Parameters
        ----------
        dataset : numpy.array
            Dataset used to estimate the scaler
        
        Returns
        -------
        numpy.array
            Scaled data
        """

        return self.scaler.fit_transform(dataset)

    def transform(self, dataset):
        """Method that scales the data in ``dataset``
        
        It must be called after calling the :class:`fit_transform` method for estimating the scaler
        Parameters
        ----------
        dataset : numpy.array
            Dataset to be scaled
        
        Returns
        -------
        numpy.array
            Scaled data
        """

        return self.scaler.transform(dataset)

    def inverse_transform(self, dataset):
        """Method that inverse-scale the data in ``dataset``
        
        It must be called after calling the :class:`fit_transform` method for estimating the scaler

        Parameters
        ----------
        dataset : numpy.array
            Dataset to be scaled
        
        Returns
        -------
        numpy.array
            Inverse-scaled data
        """

        return self.scaler.inverse_transform(dataset)

def scaling(datasets, normalize):
    """Function that scales data and returns the scaled data and the :class:`DataScaler` used for scaling.

    It rescales all the datasets contained in the list ``datasets`` using the first dataset as reference. 
    For example, if ``datasets=[X_1, X_2, X_3]``, the function estimates a :class:`DataScaler` object using the array ``X_1``, 
    and transform ``X_1``, ``X_2``, and ``X_3`` using the :class:`DataScaler` object.

    Each dataset must be a numpy.array and it should have the same column-dimensions. For example, if
    ``datasets=[X_1, X_2, X_3]``, ``X_1`` must be a numpy.array of size ``[n_1, m]``,
    ``X_2`` of size ``[n_2, m]``, and ``X_3`` of size ``[n_3, m]``, where ``n_1``, ``n_2``, ``n_3`` can be
    different.

    The scaling technique is defined by the ``normalize`` parameter which takes one of the 
    following values: 

    - ``'Norm'`` for normalizing the data to the interval [0, 1].

    - ``'Norm1'`` for normalizing the data to the interval [-1, 1]. 

    - ``'Std'`` for standarizing the data to follow a normal distribution. 

    - ``'Median'`` for normalizing the data based on the median as defined in as defined in `here <https://doi.org/10.1109/TPWRS.2017.2734563>`_.

    - ``'Invariant'`` for scaling the data based on the asinh transformation (a variance stabilizing transformations) as defined in `here <https://doi.org/10.1109/TPWRS.2017.2734563>`_. 


    The function returns the scaled data together with a :class:`DataScaler` object representing the scaling. 
    This object can be used to scale other dataset using the same rules or to inverse-transform the data.
    
    Parameters
    ----------
    datasets : list
        List of numpy.array objects to be scaled.
    normalize : str
        Type of scaling to be performed. Possible values are ``'Norm'``, ``'Norm1'``, ``'Std'``, 
        ``'Median'``, or ``'Invariant'``
    
    Returns
    -------
    List, :class:`DataScaler`
        List of scaled datasets and the :class:`DataScaler` object used for scaling. Each dataset in the 
        list is a numpy.array.

    """
    scaler = DataScaler(normalize)

    for i, dataset in enumerate(datasets):
        if i == 0:
            dataset = scaler.fit_transform(dataset)
        else:
            dataset = scaler.transform(dataset)

        datasets[i] = dataset

    return datasets, scaler



def setup_directories(results_file_path,hour_data_dir=None):
    if not os.path.exists(results_file_path):
        os.makedirs(results_file_path)
    if hour_data_dir is not None:
        if not os.path.exists(hour_data_dir):
            os.makedirs(hour_data_dir)


def process_data(file_path):
    read_data = pd.read_csv(file_path)
    df = read_data.copy()
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df.fillna(method='ffill', inplace=True)  # Fill missing data
    df['Net Load'] = df['Demand Forecast [MWh]'] - df['Wind Forecast [MW]'] - df['Solar Forecast [MW]']
    for column in ['Day-ahead Price [GBP/MWh]', 'Net Load','Last Closing Gas Price [GBpence/therm]']:
        mean = df[column].mean()
        std = df[column].std()
        df[column + '_standardized'] = (df[column] - mean) / std
    df['Hour,h'] = df['Hour']
    df['Dummy,D'] = df['Dummy']
    return df

def train_test_split(df, window_size, test_index, train_size):
    train_window_data = df[(train_size + test_index - window_size) *24 : (test_index + train_size) * 24]
    test_window_data = df[(train_size + test_index )*24 : (train_size + test_index + 1) * 24]
    return train_window_data, test_window_data

########################### Multivariate ##############################


def prepare_features(df, base_date):
    feature_dict = {}
    base_date = pd.to_datetime(base_date, format='%d/%m/%Y')
    feature_dict['Dummy,D'] = df[df['Date'] == base_date]["Dummy,D"].iloc[0]
    for feature, offsets in [
        ('Last Closing Gas Price [GBpence/therm]_standardized', [0]),
        ('Day-ahead Price [GBP/MWh]_standardized', [1, 2, 3, 7]),
        ('Net Load_standardized', [0, 1, 7]),
    ]:
        for offset in offsets:
            feature_values = df[df['Date'] == (base_date - timedelta(days=offset))][feature]
            for hour in range(24):
                if feature == "Day-ahead Price [GBP/MWh]_standardized" or feature == "Net Load_standardized":
                  feature_dict[f'{feature} d-{offset} h={hour+1}'] = feature_values.iloc[hour] if len(feature_values) == 24 else np.nan
                else:
                  feature_dict[f'{feature}'] = feature_values.iloc[hour] if len(feature_values) == 24 else np.nan

    return feature_dict


def save_input_data(inputdata,global_data_path):
    df = inputdata.copy()
    X, y,dates,features, prices = [], [], [], [], []
    for date in df['Date'].unique():
        day_data = df[df['Date'] == date]
        if len(day_data) == 24:
            features.append(prepare_features(inputdata, date))
            prices.extend(day_data['Day-ahead Price [GBP/MWh]_standardized'].values)
    if features and prices:
        X = pd.DataFrame(features)
        # y.append(prices)
        y = pd.DataFrame(prices, columns=['Target'])
        dates = pd.DataFrame(df['Date'], columns=['Date'])
        X = pd.DataFrame(np.repeat(X.values, 24, axis=0), columns=X.columns)
        X['Hour,h']= np.tile(np.arange(1, 25), int(len(X)/24) + 1)[:len(X)]
        # X['Hour,h'] = X['Hour,h'].apply(lambda x: format(x, '05b'))
    for i in range(1, 25):
        X[f'h{i}'] = 0
    for index, row in X.iterrows():
        hour_col = f'h{int(row["Hour,h"])}'
        X.at[index, hour_col] = 1

    for i in range(1, 8):
        X[f'd{i}'] = 0
    for index, row in X.iterrows():
        dummy_col = f'd{int(row["Dummy,D"])}'
        X.at[index, dummy_col] = 1
    X = X.drop('Hour,h', axis=1)
    X = X.drop('Dummy,D', axis=1)
    combined_data = pd.concat([dates.reset_index(drop=True),y.reset_index(drop=True),X.reset_index(drop=True)], axis=1)
    combined_data.to_csv(global_data_path, index=False)


def load_input_data(global_data_path):
    input_data = pd.read_csv(global_data_path, dtype={'Hour,h': str, 'Dummy,D': str})
    input_data['Date'] = pd.to_datetime(input_data['Date'])
    X = input_data.copy().drop(columns=['Target'])
    y = pd.DataFrame({'Date': input_data['Date'],'Target': input_data['Target']})
    return X,y


########################### Univariate ##############################



def prepare_hourly_features(df, base_date, hour_stamp):
    feature_dict = {}
    base_date = pd.to_datetime(base_date, format='%d/%m/%Y')
    feature_dict['Hour,h'] = format(hour_stamp+1, '05b')
    feature_dict['Dummy,D'] = df[df['Date'] == base_date]["Dummy,D"].iloc[0]
    for feature, offsets in [
        ('Last Closing Gas Price [GBpence/therm]_standardized', [0]),
        ('Day-ahead Price [GBP/MWh]_standardized', [1, 2, 3, 7]),
        ('Net Load_standardized', [0, 1, 7]),
    ]:
        for offset in offsets:
            feature_values = df[df['Date'] == (base_date - timedelta(days=offset))][feature]
            for hour in range(24):
                if feature == "Day-ahead Price [GBP/MWh]_standardized" or feature == "Net Load_standardized":
                  feature_dict[f'{feature} d-{offset} h={hour+1}'] = feature_values.iloc[hour] if len(feature_values) == 24 else np.nan
                else:
                  feature_dict[f'{feature}'] = feature_values.iloc[hour] if len(feature_values) == 24 else np.nan

    return feature_dict


def get_data_per_hour(inputdata):
    df = inputdata.copy()
    X, y, dates = [], [], []
    for hour in range(24):
        features, prices = [], []
        for date in df['Date'].unique():
            day_data = df[df['Date'] == date]
            if len(day_data) == 24:
                features.append(prepare_hourly_features(inputdata, date, hour))
                prices.append(day_data['Day-ahead Price [GBP/MWh]_standardized'].iloc[hour])
        if features and prices:
            X.append(pd.DataFrame(features))
            y.append(prices)
            dates.append(df['Date'].unique())
    return X, y, dates


def save_data_per_hour(dataset,hour_data_dir):
    X, y, dates = get_data_per_hour(dataset)  # Call get_data_per_hour once for all data
    for hour in range(24):
        X_hour = X[hour]
        y_hour = pd.DataFrame(y[hour], columns=['Target'])  
        dates_hour = pd.DataFrame(dates[hour], columns=['Date'])

        for i in range(1, 8):
            X_hour[f'd{i}'] = 0
        for index, row in X_hour.iterrows():
            dummy_col = f'd{int(row["Dummy,D"])}'
            X_hour.at[index, dummy_col] = 1
        X_hour = X_hour.drop('Hour,h', axis=1)
        X_hour = X_hour.drop('Dummy,D', axis=1)
        combined_data = pd.concat([dates_hour.reset_index(drop=True),y_hour.reset_index(drop=True),X_hour.reset_index(drop=True)], axis=1)
        combined_data.to_csv(f'{hour_data_dir}/hour_{hour+1}_data.csv', index=False)


def load_hourly_data(hour_data_dir):
    hourly_data = {}
    for hour in range(24):
        hour_data = pd.read_csv(f'{hour_data_dir}/hour_{hour+1}_data.csv', dtype={'Hour,h': str, 'Dummy,D': str})
        hour_data['Date'] = pd.to_datetime(hour_data['Date'])
        hourly_data[hour] = hour_data
    return hourly_data
