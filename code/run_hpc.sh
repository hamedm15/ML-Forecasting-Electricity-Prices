#!/bin/bash
#PBS -l select=1:ncpus=20:mem=25gb
#PBS -l walltime=60:00:00
#PBS -N RunHPC

echo "Starting the execution script."
module load anaconda3/personal
anaconda-setup
conda init

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

module purge
module load 7zip/22.01
module load tools/prod
module load SciPy-bundle/2022.05-foss-2022a

echo "Activating the virtual environment 'sklearn_env'."
source sklearn_env/bin/activate

python eval_lear.py > data/outputs/lear.txt
echo "LEAR forecasts' execution completed."

echo "Deactivating the virtual environment."
deactivate

echo "END - Execution completed."

