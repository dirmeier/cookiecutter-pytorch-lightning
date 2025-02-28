import torch as th
from pytorch_lightning import LightningModule


class RectifiedFlowMatching(LightningModule):
    """Implements rectified flow matching.

    Uses the parameterization from [1].

    References:
        [1] Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.
          https://arxiv.org/abs/2209.03003
    """
    def __init__(self, vector_field, optimizer_params, n_sampling: int = 10):
        super().__init__()
        self.score_net = vector_field
        self.n_sampling_steps = n_sampling
        self.time_max = 1.0
        self.time_eps = 1e-3
        self.optimizer_params = optimizer_params

    def forward(self, inputs, times, condition):
        ret = self.score_net(
            inputs, times * 999.0, condition, return_dict=False
        )
        return ret[0]

    def step(self, batch):
        inputs, context = batch
        times = th.rand(inputs.shape[0], device=self.device)
        times = times * (self.time_max - self.time_eps) + self.time_eps
        noise = th.randn_like(inputs, device=self.device)
        inputs = (
            times.view(inputs.shape[0], 1, 1, 1) * inputs
            + (1.0 - times.view(noise.shape[0], 1, 1, 1)) * noise
        )
        ret = self(inputs, times, context)
        target = inputs - noise
        loss = th.mean(th.square(target - ret))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train/loss", loss.item())
        return loss

    def validation_step(self, batch):
        loss = self.step(batch)
        self.log("val/loss", loss.item())
        return loss

    @th.no_grad()
    def sample(self, shape, context):
        dt = 1.0 / self.n_sampling_steps
        samples = th.randn(shape, device=self.device)
        for i in range(self.n_sampling_steps):
            times = i / self.n_sampling_steps
            times = times * (self.time_max - self.time_eps) + self.time_eps
            vt = self(
                samples,
                th.tensor(times, device=self.device)
                .to(self.dtype)
                .repeat(len(samples)),
                context,
            )
            samples = samples + vt * dt
        return samples

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
