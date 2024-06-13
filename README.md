# Opti_ML_Project
## Environement Setup
To setup, please first install the packages in `requirement.txt`.
For Pytorch, the version in requirement.txt should be the cuda12.1 version.
As the specific version depends on CUDA, install your prefered version from [here](https://pytorch.org/get-started/locally/), using the stable release (2.3.0). 

## Benchmarks
The benchmarks consist of the following problems:
- P1: simple network of 3 fully connected layers with gaussian error activation function for classification on wine quality dataset, evaluated on accuracy
- P2: simple CNN (2c2d) for classification on the Fashion MNIST dataset, evaluated on accuracy
- P3: VAE, same as P2 but with Fashion MNIST
- P4: All-CNN-C for classification on CIFAR-100, evaluated on accuracy
- P5: next word prediction using a 2-layer bidirectional LSTM trained on wikitext-2 and evaluated on accuracy

## Running the code
