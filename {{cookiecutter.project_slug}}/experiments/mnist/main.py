import hashlib
import os
import pathlib

from absl import flags, app
from ml_collections import config_flags
from pytorch_lightning.loggers import WandbLogger, CSVLogger


from {{cookiecutter.project_slug}} import RectifiedFlowMatching, DenoisingDiffusion
from {{cookiecutter.project_slug}}.nn import DiT

from callbacks import get_callbacks
from trainer import get_trainer
from dataloaders import get_data_loaders

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "configuration file")
flags.DEFINE_enum("mode", "train", ["train", "eval"], "execution mpode")
flags.DEFINE_string("workdir", None, "work directory")
flags.DEFINE_bool("usewand", False, "use wandb for logging")
flags.mark_flags_as_required(["workdir", "config"])


def get_model(config, optimizer):
    if config.nn.name == "dit":
        nn = DiT(**config.nn.architecture.to_dict())
    else:
        raise ValueError("nn not recognized")

    if config.name == "rectified_flow":
        model = RectifiedFlowMatching(nn, optimizer.params.to_dict())
    elif config.name == "denoising_diffusion":
        model = DenoisingDiffusion(nn, optimizer.params.to_dict(), n_sampling_steps=50)
    else:
        raise ValueError("model not recognized")
    return model


def hash_value(config):
    h = hashlib.new("sha256")
    h.update(str(config).encode("utf-8"))
    return h.hexdigest()


def main(argv):
    config = FLAGS.config
    model_id = f"{hash_value(config)}"
    workdir = pathlib.Path(FLAGS.workdir)
    if not workdir.exists():
        for dir in ["data", "models", "wandb"]:
            (workdir / dir).mkdir(parents=True, exist_ok=False)

    if FLAGS.usewand:
        logger = WandbLogger(
            project="mnist-experiment",
            name=model_id,
            dir=os.path.join(FLAGS.workdir, "wandb")
        )
    else:
        logger = CSVLogger(
            save_dir=os.path.join(FLAGS.workdir, "wandb"),
            name=model_id,
        )

    if FLAGS.mode == "train":
        train_loader, val_loader = get_data_loaders(
            os.path.join(FLAGS.workdir, "data"),
            config.data
        )
        callbacks = get_callbacks(
            config.call_backs,
            os.path.join(FLAGS.workdir, "models", model_id),
            val_loader,
            FLAGS.usewand
        )
        trainer = get_trainer(
            config.training,
            os.path.join(FLAGS.workdir, "models", model_id),
            callbacks,
            logger
        )
        trainer.fit(
            model=get_model(config.model, config.optimizer),
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
    else:
        raise NotImplementedError("")


if __name__ == "__main__":
    app.run(main)