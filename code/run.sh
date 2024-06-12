#!/bin/bash

echo "Starting the execution script."

echo "Activating the conda environment 'deepnn'."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate deepnn

# Run the Python script and redirect output to a file
echo "Running the Python script 'eval_dnn'."
python eval_dnn.py > data/outputs/dnn.txt
echo "DNN forecast execution completed."

# Deactivate the environment
echo "Deactivating the conda environment 'deepnn'."
conda deactivate
echo "Ensuring no conda environment is active."
conda info --envs

echo "Activating the virtual environment 'sklearn_env'."
source sklearn_env/bin/activate

python eval_lear.py > data/outputs/lear.txt
echo "LEAR forecasts' execution completed."

echo "Deactivating the virtual environment."
deactivate

echo "END - Execution completed."

