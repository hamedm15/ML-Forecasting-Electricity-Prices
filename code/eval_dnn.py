from dnn import test_dnn

dataset_file = 'data/clean_gb.csv'

calibration_windows = [364,182,91,56,28] 

years_test = 1
nlayers = 2
neurons = [256, 128]

learningrate = 0.01
reg_lambda = 0.0001
epochs_early_stopping = 20
dropout = 0
batch_normalization = False
# options include: "relu", "softplus", "tanh", 'selu','LeakyReLU', 'PReLU', 'sigmoid'
activation_fn = 'relu'
# scaling options include: 'No', 'Norm', 'Norm1', 'Std', 'Median', 'Invariant'
scaling_type = 'Norm'
Xscaling_type = 'Norm'
# options include: 'Orthogonal', 'lecun_uniform', 'glorot_uniform','glorot_normal', 'he_uniform', 'he_normal'
initializer='glorot_normal'


input_hyperparameters = {
        'In: Day': 1,
        'In: Price D-1': 1, 'In: Price D-2': 1, 'In: Price D-3': 1, 'In: Price D-7': 1, # Price Lags
        'In: 24Exog-1 D': 1, 'In: 24Exog-1 D-1': 1, 'In: 24Exog-1 D-7': 1, # Net Load Forecast + Lags
        'In: S Exog-2 D-0': 1, 'In: S Exog-2 D-1': 0, 'In: S Exog-2 D-7': 0 # Last closing Gas Price yesterday
        }
n_exog_single_inputs = 1

test_dnn(dataset_file, 
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
        n_exog_single_inputs)