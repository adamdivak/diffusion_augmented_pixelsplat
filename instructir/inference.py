import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import yaml
import random
from tqdm import tqdm
from utils import *
from models import instructir
from huggingface_hub import hf_hub_download
from datasets import load_dataset

from text.models import LanguageModel, LMHead
from torchvision import transforms
from torchmetrics.image import PeakSignalNoiseRatio
from einops import reduce
from functools import cache

from lpips import LPIPS
from skimage.metrics import structural_similarity
from torch import Tensor


SEED=42
seed_everything(SEED=SEED)
torch.backends.cudnn.deterministic = True


@torch.no_grad()
def compute_psnr(
    ground_truth,
    predicted,
):
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * mse.log10()

@cache
def get_lpips(device: torch.device) -> LPIPS:
    return LPIPS(net="vgg").to(device)


@torch.no_grad()
def compute_lpips(
    ground_truth,
    predicted,
):
    value = get_lpips(predicted.device).forward(ground_truth, predicted, normalize=True)
    return value[:, 0, 0, 0]

@torch.no_grad()
def compute_ssim(
    ground_truth,
    predicted,
):
    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)


def main():
    CONFIG     = "configs/eval5d.yml"
    LM_MODEL   = "models/lm_instructir-7d.pt"
    MODEL_NAME =  hf_hub_download(repo_id="Wouter01/InstructIR_re10k_hard", filename="best_model.pt")
    MODEL_NAME_ORIGINAL = "models/im_instructir-7d.pt" 
    with open(os.path.join(CONFIG), "r") as f: config = yaml.safe_load(f)
    cfg = dict2namespace(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print ("Creating InstructIR")
    model = instructir.create_model(input_channels =cfg.model.in_ch, width=cfg.model.width, enc_blks = cfg.model.enc_blks, 
                                middle_blk_num = cfg.model.middle_blk_num, dec_blks = cfg.model.dec_blks, txtdim=cfg.model.textdim)
                            
    model_original = instructir.create_model(input_channels =cfg.model.in_ch, width=cfg.model.width, enc_blks = cfg.model.enc_blks, 
                                middle_blk_num = cfg.model.middle_blk_num, dec_blks = cfg.model.dec_blks, txtdim=cfg.model.textdim)

    ################### LOAD IMAGE MODEL

    assert MODEL_NAME, "Model weights required for evaluation"

    print ("IMAGE MODEL CKPT:", MODEL_NAME)
    model.load_state_dict(torch.load(MODEL_NAME, map_location=device), strict=True)
    model_original.load_state_dict(torch.load(MODEL_NAME_ORIGINAL, map_location=device), strict=True)
    nparams   = count_params (model)
    print ("Loaded weights!", nparams / 1e6)


    # Initialize the LanguageModel class
    LMODEL = cfg.llm.model
    language_model = LanguageModel(model=LMODEL).to(device)
    language_model.eval()
    for p in language_model.parameters(): p.requires_grad = False

    # Initialize the LMHead class
    lm_head = LMHead(embedding_dim=cfg.llm.model_dim, hidden_dim=cfg.llm.embd_dim, num_classes=cfg.llm.nclasses).to(device)
    lm_head.load_state_dict(torch.load(LM_MODEL, map_location=device), strict=True)
    lm_head = lm_head.to(device)
    lm_head.eval()
    for p in lm_head.parameters(): p.requires_grad = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Move your language model to the same device
    language_model = language_model.to("cpu")

    # Ensure prompt_input is on the same device
    prompt_input = language_model("Please make the image crispier and sharper").to("cpu")
    lm_head = lm_head.to("cpu")
    prompt_output = lm_head(prompt_input)[0]

    # def preprocess_train(examples):
    #     transform = transforms.Compose([
    #         transforms.Resize((256, 256)),  # Resize the image to 256x256
    #         transforms.ToTensor(),           # Convert the image to a PyTorch tensor
    #     ])
    
    #     images = [transform(image.convert("RGB")) for image in examples["ground_truth_image"]]
    #     conditioning_images = [transform(image.convert("RGB")) for image in examples["conditioning_image"]]
    #     prompts = [prompt_output for p in examples["prompt"]]

    #     examples["ground_truth_image"] = images
    #     examples["conditioning_image"] = conditioning_images
    #     examples["prompt"] = prompts

    #     return examples
    def preprocess_train(examples):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize the image to 256x256
            transforms.ToTensor(),           # Convert the image to a PyTorch tensor
        ])

        images, conditioning_images, prompts = list(), list(), list()
        for i1, i2, i3 in zip(examples["ground_truth_image"], examples["conditioning_image"], examples["prompt"]):
            image_tensor = transform(i1.convert("RGB"))
            if not torch.any(image_tensor == 0):
                images.append(image_tensor)
                conditioning_images.append(transform(i2.convert("RGB")))
                prompts.append(prompt_output)

        examples["ground_truth_image"] = images
        examples["conditioning_image"] = conditioning_images
        examples["prompt"] = prompts

        return examples

    dataset_val = load_dataset("Wouter01/re10k_pixelsplat_hard", cache_dir="cachedir")["test"].with_transform(preprocess_train)
    
    def collate_fn(examples):
        if not examples: return None
        conditioning_image = torch.stack([example["conditioning_image"] for example in examples])
        conditioning_image = conditioning_image.to(memory_format=torch.contiguous_format).float()

        ground_truth_image = torch.stack([example["ground_truth_image"] for example in examples])
        ground_truth_image = ground_truth_image.to(memory_format=torch.contiguous_format).float()

        prompt = torch.stack([example["prompt"] for example in examples])
        prompt = prompt.to(memory_format=torch.contiguous_format).float()

        return {
            "conditioning_image": conditioning_image,
            "ground_truth_image": ground_truth_image,
            "prompt": prompt,
        }

    val_loader = torch.utils.data.DataLoader(dataset_val, collate_fn=collate_fn, batch_size=2, shuffle=False, num_workers=4)
    
    criterion = torch.nn.L1Loss()  # replace this with a function that calculates a metric
    psnr = PeakSignalNoiseRatio()
    psnr = psnr.to(device)

    model.eval()
    model = model.to(device)
    model_original.eval()
    model_original = model_original.to(device)
    original_loss = our_loss = pixelsplat_loss = 0
    original_psnr = our_psnr = pixelsplat_psnr = 0
    original_lpips = our_lpips = pixelsplat_lpips = 0
    original_ssim = our_ssim = pixelsplat_ssim = 0
    count = 0
    for batch in tqdm(val_loader):  # this is test set 
        if batch == None: continue
        image = batch["conditioning_image"].to(device)
        target = batch["ground_truth_image"].to(device)
        prompt = batch["prompt"].to(device)
        
        output = model(image, prompt.squeeze())
        ouptut_original = model_original(image, prompt.squeeze())

        loss = criterion(output, target)
        loss_original = criterion(ouptut_original, target)
        our_loss += loss.item()
        original_loss += loss_original.item()
        our_psnr += compute_psnr(output, target).sum().item()
        original_psnr += compute_psnr(ouptut_original, target).sum().item()
        count += image.size(0)
        loss = criterion(image, target)
        pixelsplat_loss += loss.item()
        pixelsplat_psnr += compute_psnr(image, target).sum().item()

        our_lpips += compute_lpips(output, target).sum().item()
        pixelsplat_lpips += compute_lpips(image, target).sum().item()
        original_lpips += compute_lpips(ouptut_original, target).sum().item()

        our_ssim += compute_ssim(output, target).sum().item()
        pixelsplat_ssim += compute_ssim(image, target).sum().item()
        original_ssim += compute_ssim(ouptut_original, target).sum().item()


    print(count)
    print("Our loss: ", our_loss / count)
    print("Original loss: ", original_loss / count)
    print("Pixelsplat loss: ", pixelsplat_loss / count)

    print("Our psnr: ", our_psnr / count)
    print("Original psnr: ", original_psnr / count)
    print("pixelsplat psnr", pixelsplat_psnr / count)

    print("Our lpips: ", our_lpips / count)
    print("Original lpips: ", original_lpips / count)
    print("pixelsplat lpips", pixelsplat_lpips / count)

    print("Our ssim: ", our_ssim / count)
    print("Original ssim: ", original_ssim / count)
    print("pixelsplat ssim", pixelsplat_ssim / count)


if __name__ == "__main__":
    main()