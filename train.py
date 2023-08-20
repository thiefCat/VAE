import torch
import torchvision
import torch.nn as nn
import os
from source import dataset
from source import models
from pprint import pprint
import argparse
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from torchsummary import summary


def build_config_from_args(is_jupyter=False):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_latents', type=int, default=2, help="Number of latent dimensions")
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion'])
    parser.add_argument('--c', type=float, default=0.05, help="The hyperparameter controlling the balance between rec loss and kl loss")
    parser.add_argument('--iter_save', type=int, default=50, help="Save running loss every n iterations")
    parser.add_argument('--train', type=int, default=1, help="Flag for training")
    parser.add_argument('--architecture', type=str, default='fc', choices=['fc', 'cnn'])
    parser.add_argument('--out_dir', type=str, default="Vanilla_VAE/results", help="Flag for output logging")
    if is_jupyter:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()
    return args

def load_dataset(type, train_batch_size=256):
    if type == 'mnist':
        train_loader, test_loader = dataset.get_mnist_data(train_batch_size=256, test_batch_size=10)
    else:
        train_loader, test_loader = dataset.get_fashion_mnist_data(train_batch_size=256, test_batch_size=10)
    return train_loader, test_loader


class VaeExperiment:
    def __init__(self, train_data, eval_data, model: nn.Module,
                 lr: float, num_epochs, config, save=True):
        self.config = config
        self.train_data = train_data
        self.eval_data = eval_data
        self.num_epochs = num_epochs
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.save = save
        self.train_loss=[]
        self.test_loss=[]
        self.save = save
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model


    def train(self):
        os.makedirs(self.config.out_dir, exist_ok=True)
        # self.model.train()
        epoch_iterator = trange(self.num_epochs)
        iter_save = self.config.iter_save
        for epoch in epoch_iterator: 
            running_loss = running_rec = running_kl = 0.0
            data_iterator = tqdm(self.train_data)
            for batch_idx, (data, _) in enumerate(data_iterator, 0):
                data = data.to(device)  # (batch, 28, 28)
                self.optimizer.zero_grad()

                loss, rec, kl = self.model.negative_elbo(data)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                running_rec += rec.item()
                running_kl += kl.item()
                
                if batch_idx % iter_save == iter_save - 1:    # recalculate the loss every iter_save batches
                    self.train_loss.append(running_loss / iter_save)
                    data_iterator.set_postfix({'loss': running_loss / iter_save,
                                                'rec': running_rec / iter_save,
                                                'kl': running_kl / iter_save})
                    running_loss = running_rec = running_kl = 0.0
            self.evaluate_lower_bound()

        if self.save:
            torch.save(self.model.state_dict(), self.config.out_dir + '/model.pt')
    
    def evaluate_lower_bound(self):
        eval = next(iter(self.eval_data))[0].to(device)
        metrics = [0, 0, 0]
        repeat = 20
        for _ in range(repeat):
            nelbo, kl, rec = self.model.negative_elbo(eval)
            metrics[0] += nelbo / repeat
            metrics[1] += kl / repeat
            metrics[2] += rec / repeat

        # Run multiple times to get low-var estimate
        nelbo, rec, kl = metrics
        print("Test Loss: nELBO: {}. Rec: {}. KL: {}".format(nelbo, rec, kl))
        return nelbo, rec, kl

    
    def visualize_latent_space(self, size):
        if self.model.z_dim == 2:
            with torch.no_grad():
                z_test = self.model.generate_z_grid(size).cuda()
                x_test = self.model.sample_given_z(z_test)
                x_test = x_test.view(size, size, 1, 28, 28)
                x_test = x_test.permute(1, 0, 2, 3, 4)
                x_test = x_test.reshape(-1, 1, 28, 28)
            torchvision.utils.save_image(
            x_test, self.config.out_dir + '/latent_space.png', nrow=size)

    def plt_loss(self):
        x1 = range(0, len(self.train_loss))
        y1 = self.train_loss
        plt.plot(x1, y1, 'o-')
        plt.title('Train loss vs. batches')
        plt.ylabel('Train loss')
        plt.savefig(self.config.out_dir + '/train_loss.png')  # specify the full path here
        plt.close()

    def inference(self):
        """
        Randomly generate 100 samles, save it in results/generated.png
        """
        with torch.no_grad():
            x_test = self.model.sample(100)
            x_test = x_test.reshape(100, 1, 28, 28)
        torchvision.utils.save_image(
            x_test, self.config.out_dir + '/generated.png', nrow=10)

        

if __name__ == '__main__':
    config = build_config_from_args()
    pprint(vars(config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_loader, test_loader = load_dataset(config.dataset, train_batch_size=256)
    eval_data = dataset.get_eval_data(test_loader)
    # print(type(eval_data))
    model = models.VAE(config.c, z_dim=config.num_latents, v=config.architecture)
    vae_experiment = VaeExperiment(train_loader, eval_data, model, lr=0.0015, num_epochs=15, config=config)
    if config.train:
        vae_experiment.train()
        vae_experiment.plt_loss()
    else:
        vae_experiment.model.load_state_dict(torch.load(config.out_dir + '/model.pt'))
    vae_experiment.inference()
    vae_experiment.visualize_latent_space(size=25)
