import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
from models.vq_vae.vq_vae import VqVae


class TrainVqVae:

    def __init__(self, model, training_data, validation_data, batch_size, num_steps, lr):

        self.model = model
        self.training_data = training_data
        self.data_variance = np.var(training_data.data / 255.0)
        self.validation_data = validation_data
        self.batch_size = batch_size
        self.lr = lr
        self.num_steps = num_steps

    @property
    def training_loader(self):
        return torch.utils.data.DataLoader(self.training_data, batch_size=self.batch_size, shuffle=True)

    @property
    def validation_loader(self):
        return torch.utils.data.DataLoader(self.training_data, batch_size=self.batch_size, shuffle=False)

    @property
    def optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, amsgrad=False)

    def train(self):
        self.model.train()
        train_res_recon_error = []
        train_res_perplexity = []

        for i in range(self.num_steps):
            data, _ = next(iter(self.training_loader))
            data = data.to('cuda')
            self.optimizer.zero_grad()

            vq_loss, data_recon, perplexity = self.model(data)
            recon_error = F.mse_loss(data_recon, data) / self.data_variance
            loss = recon_error + vq_loss
            loss.backward()

            self.optimizer.step()

            train_res_recon_error.append(recon_error.item())
            train_res_perplexity.append(perplexity.item())

            if (i + 1) % 100 == 0:
                print('%d iterations' % (i + 1))
                print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
                print()
        return train_res_recon_error, train_res_perplexity


def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n = sum([np.prod(p.size()) for p in model_parameters])
    return n


def main(model_args, train_args, training_data, validation_data):
    model = VqVae(**model_args).to('cuda')
    print('num of trainable parameters: %d' % get_model_size(model))
    print(model)

    train_object = TrainVqVae(model=model, training_data=training_data, validation_data=validation_data,
                              **train_args)
    train_object.train()


if __name__ == '__main__':
    model_args = {
        'num_hiddens': 128,
        'num_residual_hiddens': 32,
        'num_residual_layers': 2,
        'embedding_dim': 64,
        'num_embeddings': 512,
        'commitment_cost': 0.25,
        'decay': 0.99
    }

    train_args = {
        'batch_size': 256,
        'num_steps': 15000,
        'lr': 1e-3
    }

    training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                     ]))

    validation_data = datasets.CIFAR10(root="data", train=False, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                       ]))
    main(model_args, train_args, training_data, validation_data)
