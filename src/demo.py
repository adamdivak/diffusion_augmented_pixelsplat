from jaxtyping import install_import_hook
import hydra
import torch
import warnings
import matplotlib
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from torch import Generator
import random
import numpy as np

from .dataset.data_module import get_data_shim, get_data_shim, DatasetCfg, get_dataset
from .dataset.types import BatchedExample
from torch.utils.data import DataLoader

with install_import_hook(("src",), ("beartype", "beartype")):
    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg, get_cfg
    from src.model.decoder import get_decoder
    from src.misc.step_tracker import StepTracker
    from src.model.encoder import get_encoder
    from src.misc.image_io import save_image


def save_image(img, name):
    left_image, right_image = img["context"]["image"].chunk(2, 0)
    left_image_np = left_image.squeeze().numpy().transpose(1, 2, 0)
    right_image_np = right_image.squeeze().numpy().transpose(1, 2, 0)
    plt.imsave(f'demo_images/left_image{name}.png', left_image_np)
    plt.imsave(f'demo_images/right_image{name}.png', right_image_np)


# TODO load models from pretrained checkpoints
# TODO load example images into batch format
@hydra.main(version_base=None, config_path="../config", config_name="main")
def demo(cfg_dict):
    # Set configs and seed
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict) 
    test_cfg = cfg.test
    torch.manual_seed(cfg_dict.seed)

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
    print(dir(cropped_ex))
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
    this_is_input = BatchedExample(
        target=cropped_ex["target"],
        context=cropped_ex["context"],
        scene=cropped_ex["scene"]
    )
    batch: BatchedExample = data_shim(this_is_input)  
    b, v, _, h, w = batch["target"]["image"].shape

    # Render Gaussians
    gaussians = encoder(batch["context"], global_step=1)

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