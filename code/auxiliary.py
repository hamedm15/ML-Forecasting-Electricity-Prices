import os
import shutil
import numpy as np
import pandas as pd

def process_inputs_for_metrics(p_real, p_pred):
    """Function that checks that the two standard inputs of the metric functions satisfy some requirements
    

    Parameters
    ----------
    p_real : numpy.ndarray, pandas.DataFrame, pandas.Series
        Array/dataframe containing the real prices
    p_pred : numpy.ndarray, pandas.DataFrame, pandas.Series
        Array/dataframe containing the predicted prices
    
    Returns
    -------
    np.ndarray, np.ndarray
        The p_real and p_pred as numpy.ndarray objects after checking that they satisfy requirements 
    
    """
    
    # Checking that both arrays are of the same type
    if type(p_real) != type(p_pred):
        raise TypeError('p_real and p_pred must be of the same type. p_real is of type {}'.format(type(p_real)) +
            ' and p_pred of type {}'.format(type(p_pred)))

    # Checking that arrays are of the allowed types
    if type(p_real) != pd.DataFrame and \
       type(p_real) != pd.Series and \
       type(p_real) != np.ndarray:
        raise TypeError('p_real and p_pred must be either a pandas.DataFrame, a pandas.Serie, or ' +
        ' a numpy.aray. They are of type {}'.format(type(p_real)))

    # Transforming dataset if it is a pandas.Series to pandas.DataFrame
    if type(p_real) == pd.Series:
        p_real = p_real.to_frame()
        p_pred = p_pred.to_frame()
    
    # Checking that both datasets share the same indices
    if type(p_real) == pd.DataFrame:
        if not (p_real.index == p_pred.index).all():
            raise ValueError('p_real and p_pred must have the same indices')

        # Extracting their values as numpy.ndarrays
        p_real = p_real.values.squeeze()
        p_pred = p_pred.values.squeeze()

    return p_real, p_pred


########################### EVALUATION METRICS ##############################

def MAE(p_real, p_pred):
    # Checking if inputs are compatible
    p_real, p_pred = process_inputs_for_metrics(p_real, p_pred)

    return np.mean(np.abs(p_real - p_pred))

def sMAPE(p_real, p_pred):
    # Checking if inputs are compatible
    p_real, p_pred = process_inputs_for_metrics(p_real, p_pred)

    return np.mean(np.abs(p_real - p_pred) / ((np.abs(p_real) + np.abs(p_pred)) / 2))

def RMSE(p_real, p_pred):
    # Checking if inputs are compatible
    p_real, p_pred = process_inputs_for_metrics(p_real, p_pred)

    return np.sqrt(np.mean((p_real - p_pred)**2))

def MAPE(p_real, p_pred, noNaN=False):
    # Checking if inputs are compatible
    p_real, p_pred = process_inputs_for_metrics(p_real, p_pred)

    # Computing MAPE at every time point
    mape = np.abs(p_real - p_pred) / np.abs(p_real)

    # Eliminating NaN values if requested and averaging
    if noNaN:
        mape = np.mean(mape[np.isfinite(mape)])
    else:
        mape = np.mean(mape)

    return mape

def rMAE(p_real, p_pred, naive_mae):
    mae = MAE(p_real, p_pred)
    return mae / naive_mae



def calculate_error_metrics(df, naive_forecast_path = None):
    # Assuming 'Ytrue' is the true values column and all other columns are predictions
    prediction_columns = [col for col in df.columns if col != 'Actual']
    error_metrics = {'MAE': [], 'sMAPE': [], 'RMSE': [], 'MAPE': [], 'rMAE': []}

    if naive_forecast_path is None:
        naive_MAE = 20.186730  # Default value for the naive forecast
    else:
        naive_df = pd.read_csv(naive_forecast_path)
        naive_MAE = MAE(naive_df['Day-ahead Price [GBP/MWh]'], naive_df['Forecasted Price [GBP/MWh]'])
    
    # Compute metrics for each prediction column
    for col in prediction_columns:
        p_real = df['Actual']
        p_pred = df[col]

        # Calculate each metric
        mae = MAE(p_real, p_pred)
        smape = sMAPE(p_real, p_pred)
        rmse = RMSE(p_real, p_pred)
        mape = MAPE(p_real, p_pred)
        rmae = rMAE(p_real, p_pred, naive_mae= naive_MAE) 

        # Append the calculated metrics to the dictionary
        error_metrics['MAE'].append(mae)
        error_metrics['sMAPE'].append(smape)
        error_metrics['RMSE'].append(rmse)
        error_metrics['MAPE'].append(mape)
        error_metrics['rMAE'].append(rmae)

    # Creating a DataFrame from the metrics dictionary
    metrics_df = pd.DataFrame(error_metrics, index=prediction_columns)
    metrics_df.columns.name = 'Metric'

    return metrics_df.transpose()

########################### NAIVE FORECAST ##############################

def get_reference_date(current_date, current_dummy):
    if current_dummy in [1, 6, 7]:  # If it's Monday, Saturday, or Sunday
        return current_date - pd.Timedelta(days=7)
    else:  # For other days
        return current_date - pd.Timedelta(days=1)
    
def naive_model(data_path,naive_path):
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    training_length = 8904  
    forecast_map = {}

    for i in range(training_length):
        date = df.iloc[i]['Date']
        hour = df.iloc[i]['Hour']
        price = df.iloc[i]['Day-ahead Price [GBP/MWh]']
        forecast_map[(date, hour)] = price

    forecasts = []

    for i in range(training_length, len(df)):
        current_date = df.iloc[i]['Date']
        current_hour = df.iloc[i]['Hour']
        current_dummy = df.iloc[i]['Dummy']
        reference_date = get_reference_date(current_date, current_dummy)
        forecasted_price = forecast_map.get((reference_date, current_hour), None)

        if forecasted_price is None:
            forecasted_price = df.iloc[i - 1]['Day-ahead Price [GBP/MWh]']
        forecast_map[(current_date, current_hour)] = df.iloc[i]['Day-ahead Price [GBP/MWh]']
        
        forecasts.append(forecasted_price)

    df['Forecasted Price [GBP/MWh]'] = [None]*training_length + forecasts
    actual_prices = df['Day-ahead Price [GBP/MWh]'][training_length:]
    forecasted_prices = df['Forecasted Price [GBP/MWh]'][training_length:]
    df_test = df.iloc[training_length:].copy()
    df_test.drop(['Date','Dummy','Hour','Demand Forecast [MWh]','Wind Forecast [MW]','Solar Forecast [MW]','Last Closing Gas Price [GBpence/therm]'],axis=1,inplace=True)
    df_test.to_csv(naive_path, index=False)

    return actual_prices,forecasted_prices

def get_naive_results(data_path, naive_path = 'forecasts/naive_forecast.csv'):
    if not os.path.exists(naive_path):
        print('Creating Naive forecast')
        _,_= naive_model(data_path = data_path  ,naive_path = naive_path)
        print('Naive forecast created')
    else:
        print('File already exists')
    df = pd.read_csv(naive_path)
    predictions = df['Forecasted Price [GBP/MWh]'].values
    start_date = pd.to_datetime('2022-12-06 01:00:00+00:00')
    date_range = pd.date_range(start=start_date, periods=len(df), freq='H')
    clean_df = df.copy().drop(columns=['Datetime','Forecasted Price [GBP/MWh]'])
    clean_df['Datetime'] = date_range
    clean_df = clean_df.set_index('Datetime')
    
    clean_df[f'Naive'] = predictions
        
    return clean_df

########################### FORMATTING ##############################

def copy_csv_files(directory_path):
    for filename in os.listdir(directory_path):
        # Check if the file is a CSV file
        if filename.endswith(".csv") and "CAL_" in filename:
            # Extract the number following "CAL_" from the filename
            start_index = filename.index("CAL_") + 4  # Start after "CAL_"
            end_index = filename.index("_", start_index)  # End at the next "_"
            number = filename[start_index:end_index]  # Extract the number

            # Construct the new filename
            new_filename = f"win_{number}.csv"
            
            # Full paths for source and destination files
            src_path = os.path.join(directory_path, filename)
            dest_path = os.path.join(directory_path, new_filename)

            # Copy the file to the new filename in the same directory
            if not os.path.exists(dest_path):
                shutil.copy(src_path, dest_path)
                print(f"Copied {filename} to {new_filename}")
            else:
                print(f"File {new_filename} already exists")

def get_data_lear(results_path, calibration_windows,extra_cols = True):
    i  = 0
    for window in calibration_windows[:-1]:

        df = pd.read_csv(f'{results_path}win_{window}.csv')
        predictions = df['Predicted'].values

        if i == 0:
            start_date = pd.to_datetime('2022-12-06 01:00:00+00:00')
            date_range = pd.date_range(start=start_date, periods=len(df), freq='H')
            if extra_cols:
                clean_df = df.copy().drop(columns=['Date','Best Alpha','Dual gap','Iterations','Predicted'])
            else:
                clean_df = df.copy().drop(columns=['Date','Predicted'])
            clean_df['Datetime'] = date_range
            clean_df = clean_df.set_index('Datetime')
        
        clean_df[f'Predicted {window}'] = predictions
        i += 1

    predicted_columns = [col for col in clean_df.columns if 'Predicted' in col]
    og_df = clean_df.copy()
    clean_df['Predicted avg'] = clean_df[predicted_columns].mean(axis=1)

    return clean_df,og_df

def get_data_dnn(results_path, calibration_windows):
    i  = 0
    for window in calibration_windows[:-1]:

        df = pd.read_csv(f'{results_path}win_{window}.csv')
        predictions = df['Yp'].values

        if window == 28:
            start_date = pd.to_datetime('2022-12-09 01:00:00+00:00')
            date_range = pd.date_range(start=start_date, periods=len(df), freq='H')
            # clean_df = df.copy().drop(columns=['Date','Best Alpha','Dual gap','Iterations','Yp'])
            clean_df = df.copy().drop(columns=['Yp'])
            clean_df['Datetime'] = date_range
            clean_df = clean_df.set_index('Datetime')
        
        clean_df[f'Predicted {window}'] = predictions


    # Calculate the mean across the specified predicted columns

    predicted_columns = [col for col in clean_df.columns if 'Predicted' in col]
    og_df = clean_df.copy()
    clean_df['Predicted avg'] = clean_df[predicted_columns].mean(axis=1)
        
    return clean_df,og_df

