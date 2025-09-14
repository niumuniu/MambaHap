import os
import numpy as np
from tqdm import tqdm  

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, channels, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ReadAE(nn.Module):
    def __init__(self, nSNP: int, latent_dim: int=None):
        super().__init__()
        self.nSNP = nSNP
        if latent_dim is None:
            latent_dim = int(np.ceil(nSNP/4))

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, (4,5), (4,1), (0, 2)),
            nn.PReLU(),
            CBAM(32),

            nn.Conv2d(32, 64, (1,5), (1,1), padding='same'),
            nn.PReLU(),
            CBAM(64),

            nn.Conv2d(64, 128, (1,3), (1,1), padding='same'),
            nn.PReLU(),
            CBAM(128),

            nn.Flatten(),
        )

        self.fc1 = nn.Linear(128 * nSNP, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 128 * nSNP)
        self.act1 = nn.PReLU()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (1,3), (1,1), (0, 1)),
            nn.PReLU(),
            CBAM(64),

            nn.ConvTranspose2d(64, 32, (1,5), (1,1), (0, 2)),
            nn.PReLU(),
            CBAM(32),

            nn.ConvTranspose2d(32, 1, (4,5), (4,1), (0, 2)),
        )

    def forward(self, x):
        x_code = self.encoder(x)
        x_fc1 = self.fc1(x_code)
        x_flatten = self.act1(self.fc2(x_fc1))
        x_reshape = x_flatten.view(-1, 128, 1, self.nSNP)
        return x_fc1, self.decoder(x_reshape)



def AE_train(dataset: Dataset, num_epoch: int, embed_dim: int = None,savefile: str = None) -> ReadAE:
    batch_size = int(np.ceil(len(dataset)/20))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU....')
    else:
        device = torch.device('cpu')
        print('The code uses CPU....')

    nSNP = dataset[0][0].shape[-1]
    model = ReadAE(nSNP, embed_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_func = nn.MSELoss()

    for epoch in tqdm(range(num_epoch)):
        loss = 0
        for batch_data, _ in data_loader:
            batch_data = batch_data.to(device) 
            optimizer.zero_grad()  
            x_code, recon = model(batch_data) 
            reconstruction_loss = loss_func(recon, batch_data)  
            #sparsity_penalty = model.calculate_sparsity_penalty(x_code)
            train_loss = reconstruction_loss #+sparsity_penalty
            train_loss.backward()  
            optimizer.step()      
            loss += train_loss.item()  
        loss = loss / len(data_loader)  

        with open('AE_training_log.txt', 'a') as log_file:
            log_file.write(f"Epoch: {epoch + 1}/{num_epoch}, Loss: {loss}\n")

    return model  

