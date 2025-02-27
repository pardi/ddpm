import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from model.unet import UNet, BasicUNet, GUNet, RUNet
from collections import deque

# Tensorboard
from torch.utils.tensorboard import SummaryWriter

load = False

writer = SummaryWriter()

logging.basicConfig(level=logging.INFO)

def display_4_images(images, labels):

    if images.shape[0] < 4:
        logging.error('Need at least 4 images to display')
        return

    fig = plt.figure()

    for idx in range(4):
        ax = fig.add_subplot(2, 2, idx + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images[idx]), cmap='gray')
        ax.set_title(str(labels[idx]))

    plt.show()    


def q_sample(x, sqrt_alpha_hat, one_minus_alphas_hat):
    noise = torch.randn_like(x, device=x.device)
    return sqrt_alpha_hat.view(-1, 1, 1, 1) * x + one_minus_alphas_hat.view(-1, 1, 1, 1) * noise, noise


def sample(model, max_time_steps, alphas, betas, alphas_hat, device):

    x_t = torch.randn(1, 1, 32, 32).to(device)
    model.eval()
    with torch.no_grad():
        for t in reversed(range(max_time_steps)):
            # t_tensor = torch.ones(1, dtype=torch.long, device=device) * t
            pred_noise = model(x_t, torch.asarray([t], device=device))
            alpha_t = alphas[t]
            beta_t = betas[t]
            noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
            x_t = (1 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alphas_hat[t])) * pred_noise) + torch.sqrt(beta_t) * noise
            
    return x_t


def train(verbose=True):

    return 
    transf = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    # Get mnist dataset
    if os.path.exists('./data/MNIST') == False:
        mnist = torchvision.datasets.MNIST('./data', download=True, transform=transf)
    else:
        logging.info('MNIST dataset already exists')
        mnist = torchvision.datasets.MNIST('./data', download=False, transform=transf)

    batch_size = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True)

    # Define model
    model = RUNet()   

    model = model.to(device)     

    ## ----------------------------
    #  Parameters for training
    ## ----------------------------

    max_time_steps = 1000
    beta_min = 1e-4
    beta_max = 0.02
    betas = torch.linspace(beta_min, beta_max, max_time_steps, device=device)
    sqrt_betas = torch.sqrt(betas)
    alphas = 1 - betas
    alphas_hat = torch.cumprod(alphas, dim=0)
    one_minus_alphas_hat = (1 - alphas_hat)
    sqrt_alphas_hat = torch.sqrt(alphas_hat)
    sqrt_one_minus_alphas_hat = torch.sqrt(one_minus_alphas_hat)

    # sqrt_betas_hat = torch.sqrt((1 - alphas_hat[0:-1] * betas[1:] / (1 - alphas_hat[1:])))
    # sqrt_betas_hat = torch.cat((torch.tensor([0.0], device=device), sqrt_betas_hat), dim=0)
    sqrt_betas_hat = sqrt_betas

    num_epochs = 50

    ## ----------------------------
    #  Setup for training
    ## ----------------------------

    train_loader = torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True)

    # Training 
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss() 


    # Vector of losses
    loss_buffer = deque(maxlen=batch_size)


    best_loss = 10000
    if load == True:
        # load parameter
        model.load_state_dict(torch.load('./best.pth', weights_only=True))
    else:
        for epoch in range(1, num_epochs + 1):
            for idx, (x, _) in  enumerate(tqdm(train_loader)):

                x = x.to(device)
                x = F.pad(x, (2, 2, 2, 2))

                t = torch.randint(1, max_time_steps, (x.shape[0],), device=device)

                noisy_x, noise = q_sample(x, sqrt_alphas_hat[t], sqrt_one_minus_alphas_hat[t])
                
                # Get the model prediction
                noise_pred = model(noisy_x, t)

                loss = criterion(noise_pred, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar('Loss', loss.item(), epoch * len(train_loader) + idx)
                
                loss_buffer.append(loss.item())

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    # save model
                    torch.save(model.state_dict(), './best.pth')


            logging.info('Epoch: {} Loss: {}'.format(epoch, np.mean(loss_buffer)))

        
        torch.save(model.state_dict(), './last.pth')

    ## ----------------------------
    #  Test sampling
    ## ----------------------------

    # Sample from the model
    x_samples = np.zeros((4, 1, 32, 32))

    for idx, t in enumerate(range(4)):
        x_sample = sample(model, max_time_steps, alphas, betas, alphas_hat, device).cpu().detach()
        x_sample = (x_sample.cpu().detach().numpy() * 0.5 + 0.5).clip(0, 1) 
        x_samples[idx] = x_sample

    display_4_images(x_samples, torch.tensor([0, 1, 2, 3]))    


if __name__ == '__main__':    
    train()