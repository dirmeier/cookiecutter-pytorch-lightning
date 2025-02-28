import ml_collections


def new_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.seed=1
    image_size=32
    config.model = new_dict(
        name="denoising_diffusion",
            nn=new_dict(
                name="dit",
                architecture=new_dict(
                    num_layers=12,
                    num_attention_heads=6,
                    attention_head_dim=64,
                    sample_size=image_size, # size of the image
                    patch_size=4,
                    num_embeds_ada_norm=384,
                    dropout=0.0,
                    norm_num_groups=32,
                    in_channels=1,
                    out_channels=1,
                )
        )
    )

    config.data = new_dict(
        train_split=0.95,
        image_size=image_size,
        data_loader=new_dict(
            batch_size=128,
            shuffle=False,
            num_workers=1,
            drop_last=True,
            prefetch_factor=2
        )
    )

    config.call_backs=new_dict(
        checkpointer=new_dict(
            monitor="val/loss",
            save_last=True,
            save_top_k=5,
            every_n_epochs=1
        ),
        sampling=new_dict(
            check_val_every_n_epoch=1
        )
    )

    config.training = new_dict(
        max_epochs=1,
        check_val_every_n_epoch=1
    )

    config.optimizer = new_dict(
        params=new_dict(
            learning_rate=1e-4,
            weight_decay=1e-6,
            max_steps=200_000
        ),
    )

    return config