from lear import test_univariate_LEAR, test_multivariate_LEAR

data_file_path = 'data/gb_data.csv'

train_size = 365  # One year of data for initial training
calibration_windows = [364,182,91,56,28] 

uni_lambdas = [0.015]
# multi_lambdas = [0.001]
multi_lambdas = [0.0004]
cv_folds = 10


test_multivariate_LEAR(data_file_path,calibration_windows, train_size, uni_lambdas, cv_folds, global_data_path = None)
test_univariate_LEAR(data_file_path,calibration_windows, train_size, multi_lambdas, cv_folds, hour_data_dir = None)
