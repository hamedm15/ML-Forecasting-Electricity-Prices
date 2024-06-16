#!/bin/bash

echo "Starting the setup script."

echo "Creating conda environment 'deepnn'."
conda create --name deepnn python=3.7.2 -y

echo "Activating the conda environment 'deepnn'."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate deepnn

# Install required packages
echo "Installing required packages."
conda install numpy=1.21.6 pandas=0.25.2 scikit-learn=1.0.2 matplotlib scipy statsmodels -y
conda install -c conda-forge tensorflow -y

# Deactivate the environment
echo "Deactivating the conda environment 'deepnn'."
conda deactivate
echo "Ensuring no conda environment is active."
conda info --envs

echo "Creating and activating a virtual environment 'sklearn_env'."
python -m venv sklearn_env
source sklearn_env/bin/activate

# Install required packages in the virtual environment
echo "Installing required packages in the virtual environment."
pip install --upgrade pip
pip install pandas numpy scikit-learn python-dateutil pytz

echo "Deactivating the virtual environment."
deactivate

echo "END - Execution completed."

