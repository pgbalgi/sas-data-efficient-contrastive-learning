import torch
from torch import nn
from util import Random

class LinearCritic(nn.Module):

    def __init__(self, latent_dim, temperature=1., num_negatives = -1):
        super(LinearCritic, self).__init__()
        self.temperature = temperature
        self.projection_dim = 128
        self.w1 = nn.Linear(latent_dim, latent_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(latent_dim)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(latent_dim, self.projection_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(self.projection_dim, affine=False)
        self.cossim = nn.CosineSimilarity(dim=-1)
        self.num_negatives = num_negatives

    def project(self, h):
        return self.bn2(self.w2(self.relu(self.bn1(self.w1(h)))))

    def compute_sim(self, z):
        p = self.project(z)
        p = nn.functional.normalize(p, dim=-1)

        sim = (p @ p.T) / self.temperature
        sim.fill_diagonal_(float('-inf'))
        
        return sim

    def forward(self, z):
        return self.compute_sim(z)