import torch
import torch.nn as nn


# ========================
#  Encoder & Decoder
# ========================

class Encoder(nn.Module):
    def __init__(self, input_dim=196, hidden_dim=128, latent_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2 * latent_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):

        h = self.tanh(self.fc1(x))
        h2 = self.fc2(h)
        mu, logvar = torch.chunk(h2, 2, dim=1)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=8, hidden_dim=128, output_dim=196):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):

        h = self.tanh(self.fc1(z))
        y = self.sigmoid(self.fc2(h))
        return y



