from models import instructir
import yaml
from datasets import load_dataset
from text.models import LanguageModel, LMHead
from torchvision import transforms
import wandb
from huggingface_hub import create_repo, upload_folder, HfApi, hf_hub_download
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from PIL import Image
import os
import argparse
import torch
import lpips


def get_lpips(device):
    return lpips.LPIPS(net="vgg").to(device)

@torch.no_grad()
def compute_lpips(
    ground_truth,
    predicted,
):
    value = get_lpips(predicted.device).forward(ground_truth, predicted, normalize=True)
    return value[:, 0, 0, 0].squeeze().mean()

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    c, w, h = imgs[0].shape
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        # Convert numpy array to PIL Image
        img_pil = Image.fromarray(img.transpose(1, 2, 0))
        grid.paste(img_pil, box=(i % cols * w, i // cols * h))
    return grid

def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n\n"
        for i, log in enumerate(image_logs):
            images = log["input"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            print(images.shape)
            print(validation_image.shape)
            print(log["output"].shape)
            output = log["output"]
            img_str += f"prompt: {validation_prompt}\n"
            images = [Image.fromarray(validation_image.numpy().transpose((1, 2, 0)))] + [Image.fromarray(images.numpy().transpose((1, 2, 0)))] + [Image.fromarray(output.numpy().transpose((1, 2, 0)))]
            img_str += f"![images_{i})](./images_{i}.png)\n"

    model_description = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion",
        "stable-diffusion-diffusers",
        "text-to-image",
        "diffusers",
        "controlnet",
        "diffusers-training",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def main(args):
    os.environ["WANDB_API_KEY"] = ""  # TODO change this: ask wouter
    wandb.init(project="nice_model", entity="CV2-project")

    CONFIG     = "configs/eval5d.yml"
    LM_MODEL   = "models/lm_instructir-7d.pt"
    MODEL_NAME = "models/im_instructir-7d.pt"

    if args.model_path != None:
        MODEL_NAME = hf_hub_download(repo_id="Wouter01/really_nice_model", filename=args.model_path)

    with open(os.path.join(CONFIG), "r") as f: config = yaml.safe_load(f)
    cfg = dict2namespace(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

    # For some reason the collator doesnt work with prompts obtained from gpu
    # This is not a problem as we use the same prompt for all images and thus the only forward pass is here
    language_model = language_model.to("cpu")
    prompt_input = language_model("Please make the image crispier and sharper").to("cpu")
    lm_head = lm_head.to("cpu")
    prompt_output = lm_head(prompt_input)[0]

    def preprocess_train(examples):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize the image to 256x256, if not already
            transforms.ToTensor(),          # Convert the image to a PyTorch tensor
        ])

        images = [transform(image.convert("RGB")) for image in examples["ground_truth_image"]]
        conditioning_images = [transform(image.convert("RGB")) for image in examples["conditioning_image"]]
        prompts = [prompt_output for _ in examples["prompt"]]  # Same prompt for all images

        examples["ground_truth_image"] = images
        examples["conditioning_image"] = conditioning_images
        examples["prompt"] = prompts

        return examples

    # Get the datasets from huggingface and cache them locally
    dataset_train = load_dataset("Wouter01/re10ktrain", cache_dir="cachedir")["train"].with_transform(preprocess_train)
    dataset_val = load_dataset("Wouter01/re10ksmalltest", cache_dir="cachedir")["train"].with_transform(preprocess_train)
    dataset_img = load_dataset("Wouter01/re10ksmalltrainn", cache_dir="cachedir")["train"].with_transform(preprocess_train)
    
    # Merges a list of samples to form a mini-batch of Tensor(s) for batched loading from a map-style dataset
    def collate_fn(examples):
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

    # Get the dataloaders, img_loader is for displaying images on wandb
    train_loader = torch.utils.data.DataLoader(dataset_train, collate_fn=collate_fn, batch_size=2, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(dataset_val, collate_fn=collate_fn, batch_size=2, shuffle=False, num_workers=4)
    img_loader = torch.utils.data.DataLoader(dataset_img, collate_fn=collate_fn, batch_size=1, shuffle=False, num_workers=4)

    # Load the main model
    model = instructir.create_model(input_channels =cfg.model.in_ch, width=cfg.model.width, enc_blks = cfg.model.enc_blks, middle_blk_num = cfg.model.middle_blk_num, dec_blks = cfg.model.dec_blks, txtdim=cfg.model.textdim)
    model.load_state_dict(torch.load(MODEL_NAME, map_location=device), strict=True)
    model = model.to(device)  # just to be sure

    # We push the best model to huggingface
    os.makedirs("model_out", exist_ok=True)
    repo_id = create_repo(
                repo_id="InstructIR_with_inpainting", exist_ok=True, token=""
            ).repo_id
    
    l1_criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    model.eval()
    val_loss =0
    for batch in val_loader:
        image = batch["conditioning_image"].to(device)
        target = batch["ground_truth_image"].to(device)
        prompt = batch["prompt"].to(device)
        output = model(image, prompt.squeeze())

        loss = l1_criterion(output, target) + compute_lpips(output, target)
        val_loss += loss.item()
    best_val_loss =  val_loss / len(val_loader)
    best_model = None

    n_steps = 0  # Used to determine when to log
    
    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        language_model.eval()

        # Iterate over dataset
        for batch in train_loader:
            image = batch["conditioning_image"].to(device)
            target = batch["ground_truth_image"].to(device)
            prompt = batch["prompt"].to(device)

            optimizer.zero_grad()
            output = model(image, prompt.squeeze())
            loss = l1_criterion(output, target) + compute_lpips(output, target)
            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss.item()})

            # validation and printing every n steps
            n_steps += 1
            if n_steps % args.eval_steps == 0:
                n_steps = 0  # reset steps

                # validation
                model.eval()
                val_loss = 0
                for batch in val_loader:
                    image = batch["conditioning_image"].to(device)
                    target = batch["ground_truth_image"].to(device)
                    prompt = batch["prompt"].to(device)
                    
                    output = model(image, prompt.squeeze())

                    loss = l1_criterion(output, target) + compute_lpips(output, target)
                    val_loss += loss.item()

                val_loss /= len(val_loader)
                wandb.log({"val_loss": val_loss})  # Log the validation loss

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model.state_dict()
                    torch.save(best_model, "best_model.pt")

                    # Push the best model so far to the hub, sometimes this doesnt work so 10 retries
                    for _ in range(10):
                        try:
                            api = HfApi()
                            api.upload_file(
                                path_or_fileobj="best_model.pt",
                                path_in_repo="InstructIR_with_inpainting/best_model.pt",
                                repo_id=repo_id,
                            )
                        except:
                            continue
                        break

                # Log some sample images that are not in the training set to wandb
                image_logs = []
                log = []
                for batch in img_loader:
                    image = batch["conditioning_image"].to(device)
                    target = batch["ground_truth_image"].to(device)
                    prompt = batch["prompt"].to(device)
                    output = model(image, prompt.squeeze())

                    image_logs.append(
                        {
                            "input": image.squeeze(),
                            "output": output.squeeze(),
                            "validation_prompt": "Please make the image crispier, sharper",
                            "validation_image": target.squeeze(),
                        }
                    )

                    log += [image.squeeze(), output.squeeze(), target.squeeze()]

                wandb.log({"INPUT OUTPUT TARGET": [wandb.Image(image.detach().cpu().numpy().transpose((1, 2, 0))) for image in log]})

    # Some display of latest results for huggingface TODO if time 
    # save_model_card(
    #     repo_id,
    #     image_logs=image_logs,
    #     repo_folder="InstructIR_with_inpainting",
    # )


def parse_args():
    parser = argparse.ArgumentParser(description="Script for training a model.")
    
    # Learning rate argument
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learning rate for training the model (default: 0.0001)")

    # Number of epochs argument
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of epochs for training (default: 3)")
    
    # Number of steps between evaluation argument
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Number of steps between evaluations (default: 500)")
    
    # Model path argument
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to save the trained model (default: None)")
    
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    main(args)