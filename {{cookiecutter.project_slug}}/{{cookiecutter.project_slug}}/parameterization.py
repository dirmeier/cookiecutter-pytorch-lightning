import numpy as np
import torch as th
from pytorch_lightning import LightningModule

class Parameterization:
    def sigma(self, n_samples):
        pass

    def loss_weighting(self, sigma):
        pass

    def c_skip(self, sigma):
        pass

    def c_out(self, sigma):
        pass

    def c_in(self, sigma):
        pass

    def c_noise(self, sigma):
        pass

    def sampling_sigmas(self, n_steps):
        pass


class VEParameterization(Parameterization):
    sigma_min: th.tensor = th.tensor(0.002)
    sigma_max: th.tensor = th.tensor(100.0)

    def sigma(self, n_samples):
        min, max = th.log(self.sigma_min), th.log(self.sigma_max)
        return th.exp(min + th.rand(n_samples) * (max - min))

    def loss_weighting(self, sigma):
        return th.reciprocal(th.square(sigma))

    def c_skip(self, sigma):
        return th.ones_like(sigma)

    def c_out(self, sigma):
        return sigma

    def c_in(self, sigma):
        return th.ones_like(sigma)

    def c_noise(self, sigma):
        return th.log(0.5 * sigma)

    def sampling_sigmas(self, n_steps):
        idxs = th.arange(n_steps, dtype=th.float32) / (n_steps - 1)
        sigmas = th.sqrt(
            th.square(self.sigma_max) *
            th.exp(th.square(self.sigma_min) - th.square(self.sigma_max), idxs)
        )
        return th.concatenate([sigmas, th.zeros_like(sigmas[:1])])
