import os
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
import time


from preprocessing import setup_directories, process_data, train_test_split, save_input_data, load_input_data, save_data_per_hour, load_hourly_data

def fit_model_cross_validation(x, y, alphas=np.logspace(-5, -1, 50), cv=10, max_iter=200000, tol=0.4e-4):
    model = LassoCV(alphas=alphas, cv=cv, max_iter=max_iter, tol=tol)
    model.fit(x, y)
    return model


########################### Multivariate ##############################

def multivariate_rolling_forecast(df, results_file_path, windows, train_size, input_X, output_y, alphas=[0.001], cv=10, max_iter=200000, tol=0.4e-4):
    for window_size in windows:
        start_time = time.time()
        predictions, actuals = [], []
        res_dates, best_alpha_array, dual_gap_array, n_iter_array = [], [], [], []

        for test_index in range(6,int(((len(df)/24) - train_size))):
            print("test_index: ",test_index-5)
            train_window_data, test_window_data = train_test_split(df, window_size, test_index, train_size)
            test_data = input_X[(input_X['Date'] >= test_window_data['Date'].min()) & (input_X['Date'] <= test_window_data['Date'].max())]

            x_train = input_X[(input_X['Date'] >= train_window_data['Date'].min()) & (input_X['Date'] <= train_window_data['Date'].max())].copy().drop('Date', axis=1)
            y_train = output_y[(output_y['Date'] >= train_window_data['Date'].min()) & (output_y['Date'] <= train_window_data['Date'].max())]['Target'].values

            x_test = test_data.copy().drop('Date', axis=1)
            y_test = output_y[(output_y['Date'] >= test_window_data['Date'].min()) & (output_y['Date'] <= test_window_data['Date'].max())]['Target'].values

            # model = fit_model_cross_validation(x_train.dropna(), y_train, alphas=np.logspace(-4, 2, 100), cv=cv, max_iter=max_iter, tol=tol)

            model = LassoCV(alphas=alphas, cv=cv, max_iter=max_iter, tol=tol)
            model.fit(x_train.dropna(), y_train)
            pred = model.predict(x_test.dropna())

            predictions.extend(pred)
            actuals.extend(y_test)
            if test_index<7:
                feature_names = x_train.columns
                coef_df = pd.DataFrame({
                        'Feature': feature_names,
                    })

            coef_df[f'Coefficient - {test_index-5}'] = model.coef_

            best_alpha_array.extend([model.alpha_]* 24)
            dual_gap_array.extend([model.dual_gap_]* 24)
            n_iter_array.extend([model.n_iter_]* 24)
            res_dates.extend(test_data['Date'].values)
            coef_df = coef_df.copy()
        
        mean = df['Day-ahead Price [GBP/MWh]'].mean()
        std = df['Day-ahead Price [GBP/MWh]'].std()
        end_time = time.time()
        print("Multivariate LEAR ",window_size ," took: ", end_time - start_time)
        results_df = pd.DataFrame({
            'Date': res_dates,
            'Actual': actuals,
            'Predicted': predictions,
            'Best Alpha': best_alpha_array,
            'Dual gap': dual_gap_array,
            'Iterations': n_iter_array,
        })
        results_df['Actual'] = (results_df['Actual'] * std) + mean
        results_df['Predicted'] = (results_df['Predicted'] * std) + mean
        results_df.to_csv(f'{results_file_path}/win_{window_size}.csv', index=False)
        mse = mean_squared_error(results_df['Actual'], results_df['Predicted'])
        r2 = r2_score(results_df['Actual'], results_df['Predicted'])
        print(f"Window {window_size}, Test Index {test_index+1}: MSE = {mse}, R2 = {r2}")
        coef_df.to_csv(f'{results_file_path}/lasso_win_{window_size}.csv', index=False)
        coef_df = None
        coef_df = pd.DataFrame({
                'Feature': feature_names,
            })
        


def test_multivariate_LEAR(data_file_path,calibration_windows, train_size, lambdas, cv_folds,  global_data_path = None):
    model_name = 'global_lear'


    today_ddmm = datetime.now().strftime('%M%H_%d%m')
    global_data_path = 'data/processed_dataset.csv'
    results_file_path = f'forecasts/{model_name}/{today_ddmm}'

    if os.path.exists(global_data_path):
        setup_directories(results_file_path = results_file_path)
        data = process_data(data_file_path)
    else:
        setup_directories(results_file_path = results_file_path)
        data = process_data(data_file_path)
        save_input_data(data,global_data_path)

    input_X, output_y = load_input_data(global_data_path)
    multivariate_rolling_forecast(data,results_file_path, calibration_windows, train_size, input_X, output_y, alphas=lambdas, cv=cv_folds)
    print("Multivariate LEAR Forecasting Done")

########################### Univariate ##############################


def univariate_rolling_forecast(df, results_file_path, windows, train_size, hourly_data, alphas=[0.01], cv=10, max_iter=200000, tol=0.4e-4):
    for window_size in windows:
        start_time = time.time()
        predictions, actuals = [], []
        res_dates, best_alpha_array, dual_gap_array, n_iter_array = [], [], [], []
        for test_index in range(6,int(((len(df)/24) - train_size))):
            print("test_index: ",test_index-5)
            train_window_data, test_window_data = train_test_split(df, window_size, test_index, train_size)
            for hour in range(24):
                hour_data = hourly_data[hour]
                X_hour_data = hour_data.copy().drop(columns=['Target'])
                y_hour_data = pd.DataFrame({'Date': hour_data['Date'],'Target': hour_data['Target']})
                x_train = X_hour_data[(X_hour_data['Date'] >= train_window_data['Date'].min()) & (X_hour_data['Date'] <= train_window_data['Date'].max())].copy().drop('Date', axis=1)
                y_train = y_hour_data[(y_hour_data['Date'] >= train_window_data['Date'].min()) & (y_hour_data['Date'] <= train_window_data['Date'].max())]['Target'].values
                x_test = X_hour_data[(X_hour_data['Date'] >= test_window_data['Date'].min()) & (X_hour_data['Date'] <= test_window_data['Date'].max())].copy().drop('Date', axis=1)
                y_test = y_hour_data[(y_hour_data['Date'] >= test_window_data['Date'].min()) & (y_hour_data['Date'] <= test_window_data['Date'].max())]['Target'].values

                # model = fit_model_cross_validation(x_train.dropna(), y_train, alphas=np.logspace(-4, 2, 100), cv=cv, max_iter=max_iter, tol=tol)

                model = LassoCV(alphas=alphas, cv=cv, max_iter=max_iter, tol=tol)
                model.fit(x_train.dropna(), y_train)
                pred = model.predict(x_test.dropna())

                
                predictions.extend(pred)
                actuals.extend(y_test)
                if test_index<7 and hour == 0:
                    feature_names = x_train.columns
                    coef_df = pd.DataFrame({
                            'Feature': feature_names,
                        })

                coef_df[f'H{hour+1} - Coefficient - {test_index-5}'] = model.coef_

                res_dates.extend(X_hour_data[(X_hour_data['Date'] >= test_window_data['Date'].min()) & (X_hour_data['Date'] <= test_window_data['Date'].max())]['Date'])
                best_alpha_array.extend([model.alpha_])
                dual_gap_array.extend([model.dual_gap_])
                n_iter_array.extend([model.n_iter_])

            coef_df = coef_df.copy()

        mean = df['Day-ahead Price [GBP/MWh]'].mean()
        std = df['Day-ahead Price [GBP/MWh]'].std()
        end_time = time.time()
        print("---- UNIvariate LEAR ",window_size ," took: ", end_time - start_time)
        results_df = pd.DataFrame({
            'Date': res_dates,
            'Actual': actuals,
            'Predicted': predictions,
            'Best Alpha': best_alpha_array,
            'Dual gap': dual_gap_array,
            'Iterations': n_iter_array,
        })
        results_df['Actual'] = (results_df['Actual'] * std) + mean
        results_df['Predicted'] = (results_df['Predicted'] * std) + mean

        results_df.to_csv(f'{results_file_path}/win_{window_size}.csv', index=False)
        mse = mean_squared_error(results_df['Actual'], results_df['Predicted'])
        r2 = r2_score(results_df['Actual'], results_df['Predicted'])
        print(f"Window {window_size}, Test Index {test_index+1}: MSE = {mse}, R2 = {r2}")
        coef_df.to_csv(f'{results_file_path}/lasso_win_{window_size}.csv', index=False)
        coef_df = None
        coef_df = pd.DataFrame({
                'Feature': feature_names,
            })
        
def test_univariate_LEAR(data_file_path, calibration_windows, train_size, lambdas, cv_folds, hour_data_dir = None):
    model_name = '24lear'

    today_ddmm = datetime.now().strftime('%M%H_%d%m')
    hour_data_dir = 'data/hourly_dataset'
    results_file_path = f'forecasts/{model_name}/{today_ddmm}'
    if os.path.exists(hour_data_dir):
        setup_directories(results_file_path = results_file_path,hour_data_dir=hour_data_dir)
        data = process_data(data_file_path)
    else:
        setup_directories(results_file_path = results_file_path,hour_data_dir=hour_data_dir)
        data = process_data(data_file_path)
        save_data_per_hour(data,hour_data_dir)

    hourly_data = load_hourly_data(hour_data_dir)
    univariate_rolling_forecast(data, results_file_path, calibration_windows, train_size, hourly_data, alphas=lambdas, cv=cv_folds)
    print("Univariate LEAR Forecasting Done")