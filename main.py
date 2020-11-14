import os
from tqdm import tqdm
import argparse
import torch
import torchvision

from models import Generator, Discriminator
from utils import prepare_dataloaders, AverageMeter
from torchvision import transforms
from torchvision.utils import save_image

def run_epoch(models, mode, data_loader, data_size, args, epoch, criterion, optimizers, device):
    if mode == 'train':
        models[0].train()
        models[1].train()
    else:
        models[0].eval()
        models[1].eval()


    loss = AverageMeter()
    tq = tqdm(data_loader, desc='   - ({})   '.format(mode), leave=False)
    for data, _ in tq:
        real_labels = torch.ones(data.size(0), 1).to(device)
        fake_labels = torch.zeros(data.size(0), 1).to(device)

        real_images = data.reshape(data.size(0), -1).to(device)

        """ Train the discriminator """
        outputs = models[1](real_images) # discriminator
        d_real_loss = criterion(outputs, real_labels)

        z = torch.randn(data.size(0), args.latent_size).to(device)
        fake_images = models[0](z) # generator
        outputs = models[1](fake_images) # discriminator
        d_fake_loss = criterion(outputs, fake_labels)

        d_loss = d_real_loss + d_fake_loss
        if mode == 'train':
            optimizers[1].zero_grad()
            d_real_loss.backward()
            d_fake_loss.backward()
            optimizers[1].step()

        """ Train the generator """
        z = torch.randn(data.size(0), args.latent_size).to(device)
        fake_images = models[0](z)
        outputs = models[1](fake_images)

        g_loss = criterion(outputs, real_labels)
        if mode == 'train':
            optimizers[0].zero_grad()
            g_loss.backward()
            optimizers[0].step()
        loss.update(g_loss.item(), real_images.size(0))
        if epoch % 10 == 1:
            real_images = real_images.reshape(real_images.size(0), 1, 28, 28)
            fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
            save_image(real_images, args.save_path + '/real_images-{}.png'.format(epoch))
            save_image(fake_images, args.save_path + '/fake_images-{}.png'.format(epoch))
        tq.set_description('    - ({})  epoch {}/{} dLoss {:.3f} gLoss {:.3f}'.format(mode, epoch, args.epochs, d_loss, g_loss))
    return loss.avg

def main(args):
    args.latent_size = 64
    args.image_size = 784
    args.hidden_size = 256
    """ Create a directory if not exists """
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.isdir(args.data_path):
        os.mkdir(args.data_path)
    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)

    """ Settings for training """
    torch.manual_seed(args.seed)

    """ Device configuration """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    multi_gpu = torch.cuda.device_count() > 1
    print("[Info] The system consist of {} {}.".format(torch.cuda.device_count() if device == 'cuda' else '1', device))

    """ Prepare the dataloaders """
    train_loader, valid_loader, train_size, valid_size = prepare_dataloaders(args)

    """ Define the generator & discriminator. """
    generator = Generator(args).to(device)
    discriminator = Discriminator(args).to(device)
    models = [generator, discriminator]

    """ Define the loss function & optimizers """
    criterion = torch.nn.BCELoss().to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
    optimizers = [g_optimizer, d_optimizer]

    """ Define the learning rate scheduler """
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    """ Iterate over the data. """
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        # adjust_learning_rate(optimizers, epoch)

        # train for one epochs
        _ = run_epoch(models, 'train', train_loader, train_size, args, epoch, criterion, optimizers, device)
        with torch.no_grad():
            loss = run_epoch(models, 'valid', valid_loader, valid_size, args, epoch, criterion, optimizers, device)
            if best_loss > loss:
                best_loss = loss
                print("[Info] Save the model. : {}".format(epoch))
                torch.save(models[0].state_dict(), './checkpoints/G.ckpt')
                torch.save(models[1].state_dict(), './checkpoints/D.ckpt')
                # save_checkpoint()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch GAN Example.')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (defulat: 128)')
    parser.add_argument('--epochs', type=int, default=512, metavar='N',
                        help='number of epochs to train (defulat: 512)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (defulat: 0.001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--dataset', type=str, default='mnist', metavar='N',
                        help='training dataset for GAN  (mnist, CIFAR, ImageNet)')
    parser.add_argument('--data_path', type=str, default='./data', metavar='N',
                        help='Path of datasets (default: ./data)')
    parser.add_argument('--save_path', type=str, default='./results', metavar='N',
                        help='Path of datasets (default: ./results)')
    parser.add_argument('--workers', type=int, default=4, metavar='N',
                        help='number of workers (default: 4)')

    args = parser.parse_args()
    main(args)
