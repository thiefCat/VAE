# VAE
## Introduction
This report presents an extensive investigation into Variational Autoencoders (VAEs), a powerful class of generative models. 
I made an in-depth exploration of the mathematical formulations underpinning VAE and achieved a comprehensive understanding 
of latent models. I implemented VAE and tested it on MNIST and FashionMNIST dataset. A comparative analysis was performed 
between MLP-based and CNN-based encoder-decoder architectures, as well as different latent space dimensions, revealing insights 
into how specific network architectures can influence overall performance.

## Download the dataset
This model uses both MNIST dataset and Fashion MNIST dataset. Please download the data (four .gz files for both datasets), and
put the data to "data/MNIST" or "data/Fashion_MNIST". The file structure is illustrated below:

<img src="file_structure.png" width="200" height="200">

## How to run
To run the model, just type "python train.py", the model will be trained by default parameters. You can look at how to change hyperparameters by running "python train.py -h" and use command line codes to change them. e.g., the data used for training, the latent dimension, the architecture for the model .etc.

## Result
You can visualize the training result by looking at /result folder.
