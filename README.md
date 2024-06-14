# Opti_ML_Project
## Environment Setup
To set up the environment, please first install the packages in `requirement.txt`.
For Pytorch, the version in requirement.txt should be the cuda12.1 version.
As the specific version depends on CUDA, install your preferred version from [here](https://pytorch.org/get-started/locally/), using the stable release (2.3.0). 

## Benchmarks
The benchmarks consist of the following problems:
- P1: simple network of 3 fully connected layers with gaussian error activation function for classification on wine quality dataset, evaluated on accuracy
- P2: simple CNN (2c2d) for classification on the Fashion MNIST dataset, evaluated on accuracy
- P3: VAE, same as P2 but with Fashion MNIST
- P4: All-CNN-C for classification on CIFAR-100, evaluated on accuracy
- P5: next word prediction using a 2-layer bidirectional LSTM trained on wikitext-2 and evaluated on accuracy

## Code structure
- Models are all defined in `models.py`, and their state dict is saved in the models folder
- Custom datasets for wikitext (LSTM, for P5) and Wine (P1) are in `custom_datasets.py`
- Function used to process the data are in `utils.py`
- The manager for pytorch optimizer is in `pytorch_optim_training_manager.py`
- `torch_problems_eval.ipynb` and `nevergrad_problem_eval.ipynb` are the notebooks to produce results
- The manager for the Nevergrad optimizer is build in `nevergrad_problem_eval.ipynb`
- Our best results for each problem and for each optimizer are in `results.md`
- Results for each epoch are stored as a file in the results folder
- `generate_graph.ipynb` is the notebook used to produce graphs from the files in results folder

## Reproducing results

To reproduce raw results for a given optimizer you need to do the following steps:

If the optimizer is a first order optimizer from Pytorch, run `torch_problems_eval.ipynb`.
If the optimizer is a zero order optimizer from Nevergrad, run `nevergrad_problem_eval.ipynb`.

Run the notebook making sure the optimizer defined in each problem is the optimizer you want to evaluate
and, if you save the results, make sure that the names correspond to the optimizer.

To evaluate on multiple optimizers, rerun the end of each problem starting at the "restart here" cell
while changing the optimizer (and the file name).

