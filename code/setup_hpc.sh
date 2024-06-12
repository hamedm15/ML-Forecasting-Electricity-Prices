#!/bin/bash
#PBS -l select=1:ncpus=20:mem=25gb
#PBS -l walltime=01:00:00
#PBS -N py372

cd $PBS_O_WORKDIR
echo "Starting the setup script."

module load anaconda3/personal
anaconda-setup
conda init

echo "Creating conda environment 'deepnn'."
conda create --name deepnn python=3.7.2 -y

# Activate the environment
source activate deepnn

# Install required packages
echo "Installing required packages."
conda install numpy=1.21.6 pandas=0.25.2 scikit-learn=1.0.2 matplotlib scipy statsmodels -y
conda install -c conda-forge tensorflow -y

# Deactivate the environment
echo "Deactivating the conda environment 'deepnn'."
conda deactivate
echo "Ensuring no conda environment is active."
conda info --envs

module purge
module load 7zip/22.01
module load tools/prod
module load SciPy-bundle/2022.05-foss-2022a

echo "Creating and activating a virtual environment 'sklearn_env'."
python -m venv sklearn_env
source sklearn_env/bin/activate

# Install required packages in the virtual environment
echo "Installing required packages in the virtual environment."
pip install --upgrade pip
pip install pandas numpy scikit-learn python-dateutil pytz

echo "Deactivating the virtual environment."
deactivate

echo "END - Setup completed."

