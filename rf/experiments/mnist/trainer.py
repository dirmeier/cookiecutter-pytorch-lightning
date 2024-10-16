import pytorch_lightning as pl
import torch


def get_trainer(
    config,
    root_dir,
    callbacks,
    logger
):
    trainer = pl.Trainer(
        **config.to_dict(),
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=callbacks,
        default_root_dir=root_dir,
    )

    return trainer
