import torch
import torch.nn as nn
import torch.functional as F
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import deque

# Tensorboard
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(logging.INFO)

class Trainer:
    def __init__(self, model: nn.Module, batch_size: int, board: bool = False, save: bool = True, device: str = 'cpu'):
        self.model = model
        self._is_board = board
        self._is_save = save
        self.device = device

        self.model = self.model.to(self.device)

        self._board_writer = SummaryWriter()
        self.batch_size = batch_size


    
    def run(self, model_path: str) -> None:
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

    def train(self, epochs: int, train_loader, verbose: bool = True):

        best_loss = 1e10
        losses_buffer = deque(batch_size)


        for epoch in range(1, epochs + 1):
            for idx, (x, _) in  enumerate(tqdm(train_loader)):

                x = x.to(self.device)
                x = F.pad(x, (2, 2, 2, 2))

                t = torch.randint(1, max_time_steps, (x.shape[0],), device=self.device)

                noisy_x, noise = q_sample(x, sqrt_alphas_hat[t], sqrt_one_minus_alphas_hat[t])
                
                # Get the model prediction
                noise_pred = model(noisy_x, t)

                loss = criterion(noise_pred, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar('Loss', loss.item(), epoch * len(train_loader) + idx)
                
                losses_buffer.append(loss.item())

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    
                    if self._is_save:
                        torch.save(self.model.state_dict(), './best.pth')

            if verbose:
                logging.info('Epoch: {} Loss: {}'.format(epoch, np.mean(loss_buffer)))


        if self._is_save:
            torch.save(self.model.state_dict(), './last.pth')


