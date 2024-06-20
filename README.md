# Masters Thesis: Using Statistical and Machine Learning Methods to Forecast Day-Ahead Electricity Prices

## Project Overview

The application of Neural Networks (NN) in trading and price forecasting in energy markets is increasingly gaining attention. Specifically, the use of machine learning (ML) in predicting energy market behaviors has been a focal point of several studies and industry applications. However, the rapid evolution of power systems presents challenges and opportunities for ML applications in this field. This project delves into these aspects, exploring the effectiveness and adaptability of ML models in the context of the Great Britain (GB) electricity market.


## Repository Structure

- `code/`: Code for data processing, model training, and analysis.
- `docs/`: Documentation including final reports.
- [`logbook/`](docs/logbook/LogBook.md): Notes and progress from meetings, research findings.
- `code/data/`: Contains datasets used for model training and analysis.
- `presentations/`: Slides and materials for the presentation.

## Installation and Usage

### - Getting Started

Clone the repository and navigate to the directory
```bash
$ git clone https://github.com/hamedm15/ML-Forecasting-Electricity-Prices.git
$ cd ML-Forecasting-Electricity-Prices
```
**LOCAL :** Run [`setup.sh`](setup.sh) to create the conda environments and install all the required dependencies to avoid conflicts
```bash
./setup.sh
```
**HPC :** After SSHing into the server and cloning the repository, run [`setup_hpc.sh`](setup_hpc.sh)

```bash
./setup_hpc.sh
```
###  - Running the forecasting models

This repository provides three modelling frameworks for electricity price forecasting:

- [`gLEAR`](code/eval_lear.py) - Global *(fully multivariate)* LASSO Estiamted Autoregressive Model
- [`24LEAR`](code/eval_lear.py) - Separable *(set of 24 univariate)* LASSO Estiamted Autoregressive Models
- [`DNN`](code/eval_dnn.py) - Deep Neural Netowork Model


The code below, runs all three frameworks over all the specified calibration windows

**LOCAL :**
```bash
chmod +x run.sh
./run.sh
```
**HPC :** 
```bash
chmod +x run_hpc.sh
screen
./run_hpc.sh
```

Navigate to the examples folder and check the existing examples to get you started. The examples include several applications of the two state-of-the art forecasting model: a deep neural net and the LEAR model.

### - Code Structure
- Data management : [`preprocessing.py`](code/preprocessing.py.py).
- Forecasting Models : [`LEAR.py`](code/lear.py) and [`DNN.py`](code/dnn.py).
- Python Notebook to display results and plots : [`analysis.ipynb`](code/analysis.ipynb).
- Diebold-Mariano (DM) and Giacomini-White (GW) tests : [`statisticaltests.py`](code/statisticaltests.py).
- Auxiliary Functions : [`auxiliary.py`](code/auxiliary.py).


## Acknowlegements

Code was adapted and taken from the following open-source library
### **Epftoolbox**

- [Github Repository](https://github.com/jeslago/epftoolbox) *(AGPL-3.0 License)*
- [Documentation](https://epftoolbox.readthedocs.io/en/latest/)

**Reference:** Jesus Lago, Grzegorz Marcjasz, Bart De Schutter, Rafał Weron, *"[Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark](https://www.sciencedirect.com/science/article/pii/S0306261921004529?via%3Dihub)"*, Applied Energy 2021.



## Contact

This project is part of a Masters Thesis at **Imperial College London**, under the guidance of **Dr. Elina Spyrou**. For detailed information, refer to the linked studies and the documentation within this repository.
