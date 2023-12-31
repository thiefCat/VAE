# VAE
## Introduction
This report presents my investigation into Variational Autoencoders (VAEs), a powerful class of generative models. I made an in-depth exploration of the mathematical formulations underpinning VAE and achieved a comprehensive understanding of latent models. I implemented VAE from scratch and tested them on MNIST and FashionMNIST dataset. I also implemented two network structures (MLP-based and CNN-based encoder-decoder architectures), and a comparative analysis was performed between them, as well as different latent space dimensions, revealing insights into how specific network architectures can influence overall performance.  
More details can be found in Variational_Autoencoders.pdf

## Download the dataset
This model uses both MNIST dataset and Fashion MNIST dataset. Please download the data (four .gz files for both datasets), and
put the data to "data/MNIST" or "data/Fashion_MNIST". The file structure is illustrated below:

<img src="file_structure.png" width="200" height="200">

## How to run
Here is the code to run by default:

  ```bash
  git clone https://github.com/thiefCat/VAE.git
  cd VAE
  python train.py 
  ```
You can look at how to change hyperparameters by running "python train.py -h" and use command line codes to change them. e.g., the dataset used for training, the latent dimension, the architecture for the model .etc.

## Result
You can visualize the training result by looking at /result folder, including the training curve, the visualized latent space, and the generation result.  
The generation result by MLP structure, MNIST dataset, using c=0.05, num_latents=2, as well as the 2d latent space is as follows:

<img src="results\generated.png" width="200" height="200">
<img src="results\latent_space.png" width="200" height="200">
