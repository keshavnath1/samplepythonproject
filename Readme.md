# Temporal Model Environment Setup

This guide outlines the steps to set up a Conda environment for running a PyTorch-based neural network model on temporal data, and how to execute unit tests using PyTest within a Jupyter Notebook.

## Creating the Conda Environment

Create an environment named `temporal_model_env` using the provided `environment.yml` file:

```sh
conda env create -f environment.yml

conda activate temporal_model_env

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

python temporal_model.py


pytest test_temporal_model.py


conda install jupyter
conda install ipykernel
python -m ipykernel install --user --name temporal_model_env --display-name "Python (temporal_model_env)"


jupyter notebook

jupyter lab


jupyter nbconvert --to script [YOUR_NOTEBOOK].ipynb

```
## Running PyTest in Jupyter Notebook
To run pytest within a Jupyter Notebook, execute the following in a cell:
```
!pytest test_temporal_model.py -s
```

