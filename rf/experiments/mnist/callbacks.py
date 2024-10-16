import os

import wandb
from matplotlib import pyplot as plt
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


def plot_figures(samples):
    samples = samples.cpu().detach().numpy()
    ncols = samples.shape[0] // 8
    nrows = samples.shape[0] // 16
    size = samples.shape[-1]

    def convert_batch_to_image_grid(image_batch):
        reshaped = (
            image_batch.reshape(nrows, ncols, size, size, 1)
            .transpose([0, 2, 1, 3, 4])
            .reshape(nrows * size, ncols * size, 1)
        )
        # undo initial scaling, i.e., map [-1, 1] -> [0, 1]
        return reshaped / 2.0 + 0.5

    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(
        convert_batch_to_image_grid(samples),
        interpolation="nearest",
        cmap="gray",
    )
    plt.axis("off")
    plt.tight_layout()
    return fig


class LogCallback(Callback):
    def __init__(self, output_dir, val_loader, use_wand, check_val_every_n_epoch):
        self.val_loader = val_loader
        self.use_wand = use_wand
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.output_dir = output_dir

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if pl_module.current_epoch == 0:
            return
        if pl_module.current_epoch % self.check_val_every_n_epoch != 0:
            return

        inputs, labels = next(iter(self.val_loader))
        samples = pl_module.sample(inputs.shape, labels)
        fig = plot_figures(samples)

        fl = os.path.join(
            self.output_dir, f"samples-step_{pl_module.global_step}.png"
        )
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        fig.savefig(fl, dpi=200)

        if self.use_wand:
            wandb.log({"images": wandb.Image(fig)}, step=pl_module.global_step)


def get_callbacks(config, model_dir, val_loader, use_wand):
    return [
        LearningRateMonitor(),
        ModelCheckpoint(
            dirpath=os.path.join(model_dir, "checkpoints"),
            **config.checkpointer.to_dict(),
        ),
        LogCallback(
            output_dir=os.path.join(model_dir, "figures"),
            val_loader=val_loader,
            use_wand=use_wand,
            check_val_every_n_epoch=config.sampling.check_val_every_n_epoch
        )
    ]


