import os
from pathlib import Path

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper



def save_image(img, name):
    left_image, right_image = img["context"]["image"].chunk(2, 0)
    left_image_np = left_image.squeeze().numpy().transpose(1, 2, 0)
    right_image_np = right_image.squeeze().numpy().transpose(1, 2, 0)
    plt.imsave(f'demo_images/left_image{name}.png', left_image_np)
    plt.imsave(f'demo_images/right_image{name}.png', right_image_np)


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"

# TODO load models from pretrained checkpoints
# TODO load example images into batch format
@hydra.main(version_base=None, config_path="../config", config_name="main")
def demo(cfg_dict):
    # Set configs and seed
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict) 
    test_cfg = cfg.test
    torch.manual_seed(cfg_dict.seed)

    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    print(cyan(f"Saving outputs to {output_dir}."))
    latest_run = output_dir.parents[1] / "latest-run"
    os.system(f"rm {latest_run}")
    os.system(f"ln -s {output_dir} {latest_run}")

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model="all",
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()
    print("setting up checkpointing")
    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
        )
    )

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()
    print("init trainer")
    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy=(
            "ddp_find_unused_parameters_true"
            if torch.cuda.device_count() > 1
            else "auto"
        ),
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=True,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        log_every_n_steps=1,
        plugins=[SLURMEnvironment(auto_requeue=False)],
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    # Get the encoder (EncoderEpipolar)
    # This is a large model, presumably contributing the most free parameters
    # Input has 3 channels
    # Outputs the Gaussians
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    # Get the decoder (DecoderSplattingCUDA)
    # The decoder uses:
    #    intrinsics, extrinsics, imgage shape, background color, far, near,
    #    Gaussian means, covariances, harmonics, opacities 
    # The decoder outputs:
    #    With the above the decoder generates colors (and depth if depth_mode=True)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)

    # Get the dataset
    dataset = get_dataset(cfg.dataset, "test", StepTracker(), demo=True)
    
    cropped_ex, ex = next(iter(dataset))
    # return 
    save_image(ex, ""); save_image(cropped_ex, "_cropped")
    cropped_ex["context"]["image"] = cropped_ex["context"]["image"].unsqueeze(0)
    cropped_ex["target"]["image"] = cropped_ex["target"]["image"].unsqueeze(0)
    cropped_ex["context"]["intrinsics"] = cropped_ex["context"]["intrinsics"].unsqueeze(0)
    cropped_ex["target"]["intrinsics"] = cropped_ex["target"]["intrinsics"].unsqueeze(0)
    cropped_ex["context"]["extrinsics"] = cropped_ex["context"]["extrinsics"].unsqueeze(0)
    cropped_ex["target"]["extrinsics"] = cropped_ex["target"]["extrinsics"].unsqueeze(0)

    # A batch contains: Target, Context, Scene
    # Target and Context are contain:
    #    extrinsics, intrinsics, image, near, far, index
    # Scene is a list of strings
    data_shim = get_data_shim(encoder)
    batch: BatchedExample = data_shim(cropped_ex)  
    b, v, _, h, w = batch["target"]["image"].shape

    # Render Gaussians
    gaussians = encoder(batch["context"])

    # Decoder, TODO this seems heavily inefficient (but is used)
    color = []
    for i in range(0, batch["target"]["far"].shape[1], 32):
        output = decoder.forward(
            gaussians,
            batch["target"]["extrinsics"][:1, i : i + 32],
            batch["target"]["intrinsics"][:1, i : i + 32],
            batch["target"]["near"][:1, i : i + 32],
            batch["target"]["far"][:1, i : i + 32],
            (h, w),
        )
        color.append(output.color)
    color = torch.cat(color, dim=1)

    # Save images
    (scene,) = batch["scene"]
    name = get_cfg()["wandb"]["name"]
    path = test_cfg.output_path / name
    for index, color in zip(batch["target"]["index"][0], color[0]):
        save_image(color, path / scene / f"color/{index:0>6}.png")
    for index, color in zip(batch["context"]["index"][0], batch["context"]["image"][0]):
        save_image(color, path / scene / f"context/{index:0>6}.png")


if __name__ == "__main__":
    demo()