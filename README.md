# Opti_ML_Project
## environement setup
install the packages in requirement.txt

special case: pytorch, as the specific version depends on CUDA (if you have it, or specific version) install your prefered version from [here](https://pytorch.org/get-started/locally/), use stable release (2.3.0)

## benchmark
The benchmark consist of the following problems:
- P1: simple CNN (2c2d) for classification on the Fashion MNIST dataset, evaluated on accuracy
- P2: VAE (2 linear layers for both encoder and decoder) for generation on MNIST, evaluated on the loss
- P3 :