import torch

""" Declare and initialize the deep learning models. """
class Generator(torch.nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(args.latent_size, args.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_size, args.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_size, args.image_size),
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(torch.nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(args.image_size, args.hidden_size),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(args.hidden_size, args.hidden_size),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(args.hidden_size, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
