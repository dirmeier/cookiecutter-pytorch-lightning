import torch as th
from pytorch_lightning import LightningModule
from {{cookiecutter.project_slug}}.parameterization import Parameterization, VEParameterization


def to_output_shape(x, target_dims):
    dims_to_append = target_dims - x.ndim
    return x[(...,) + (None,) * dims_to_append]


class DenoisingDiffusion(LightningModule):
    """Implements a continuous-time denoising diffusion model.

    Uses the vanilla variance exploding SDE from [1] using the parameterization
    from [2] and samples synthetics using a conventional Euler solver.

    References:
        [1] Score-Based Generative Modeling through Stochastic Differential Equations.
          https://arxiv.org/abs/2011.13456
        [2] Elucidating the Design Space of Diffusion-Based Generative Models,
          https://arxiv.org/abs/2206.00364
    """

    def __init__(
        self,
        score_net,
        optimizer_params,
        n_sampling_steps,
        parameterization: Parameterization = VEParameterization(),
        time_eps=1e-3,
    ):
        super().__init__()
        self.score_net = score_net
        self.n_sampling_steps = n_sampling_steps
        self.time_max = 1.0
        self.time_eps = time_eps
        self.parameterization = parameterization
        self.optimizer_params = optimizer_params

    def forward(self, inputs, sigma, context=None):
        # scale the noise
        scaled_noise = self.parameterization.c_noise(sigma)
        # scale the inputs
        scaled_inputs = inputs * to_output_shape(self.parameterization.c_in(sigma), inputs.dim())
        # compute the score
        score = self.score_net(scaled_inputs, scaled_noise, context=context, return_dict=False)
        # scale skips and outputs
        out = score * to_output_shape(self.parameterization.c_out(sigma), inputs.dim())
        skip = inputs * to_output_shape(self.parameterization.c_skip(sigma), inputs.dim())
        return out + skip

    def step(self, batch):
        inputs, context = batch
        # get sigma and noise
        sigma = self.parameterization.sigma(inputs.shape[0])
        noise = th.randn_like(inputs) * to_output_shape(sigma, inputs.dim())
        # compute scores
        score = self(inputs + noise, sigma, context)
        # compute weighted loss
        loss_weight = to_output_shape(self.parameterization.loss_weighting(sigma), score.dim())
        loss = loss_weight * th.square(score - inputs)
        return th.mean(loss)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train/loss", loss.item())
        return loss

    def validation_step(self, batch):
        loss = self.step(batch)
        self.log("val/loss", loss.item())
        return loss

    @th.no_grad()
    def sample(self, shape, context=None):
        """Euler solver."""
        n = shape[0]
        sigmas = self.parameterization.sampling_sigmas(self.n_sampling_steps)
        noise = th.randn(shape, device=self.device) * sigmas[0]

        sample_next = noise
        for i, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            sample_curr = sample_next
            grad = self(
                sample_curr,
                th.repeat(sigma, n),
                context,
            )
            dt = sigma_next - sigma
            d_cur = (sample_curr - grad) / sigma
            sample_next = sample_curr + d_cur * dt
        return sample_next

    def configure_optimizers(self):
        optimizer = th.optim.Adam(
            self.parameters(), lr=self.optimizer_params["learning_rate"]
        )
        lr_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.optimizer_params["max_steps"]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }
