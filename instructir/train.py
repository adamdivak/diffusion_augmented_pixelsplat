from utils import *
from models import instructir
import yaml
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from text.models import LanguageModel, LMHead
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import save_image
import wandb
from huggingface_hub import create_repo, upload_folder
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from PIL import Image
from huggingface_hub import HfApi

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


def main():
    os.environ["WANDB_API_KEY"] = ""  # TODO change this: ask wouter

    wandb.init(
        project="nice_model",
        entity="CV2-project",
    )

    CONFIG     = "configs/eval5d.yml"
    LM_MODEL   = "models/lm_instructir-7d.pt"
    MODEL_NAME = "models/im_instructir-7d.pt"

    # parse config file
    with open(os.path.join(CONFIG), "r") as f:
        config = yaml.safe_load(f)

    cfg = dict2namespace(config)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Initialize the LanguageModel class
    LMODEL = cfg.llm.model
    language_model = LanguageModel(model=LMODEL)
    for p in language_model.parameters():
        p.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm_head = LMHead(embedding_dim=cfg.llm.model_dim, hidden_dim=cfg.llm.embd_dim, num_classes=cfg.llm.nclasses)
    lm_head = lm_head.to(device)

    
    lm_head.load_state_dict(torch.load(LM_MODEL, map_location=device), strict=True)

    lm_embd = language_model("Please make the image crispier, sharper")
    text_embd, deg_pred = lm_head (lm_embd)

    data = load_dataset("Wouter01/re10ksmalltest")["train"] #["default"] #["test"]
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize the image to 256x256
        transforms.ToTensor(),           # Convert the image to a PyTorch tensor
    ])

    text_embd = text_embd.detach().clone()

    transformed_data = [(transform(sample['conditioning_image']).squeeze(), transform(sample['ground_truth_image']).squeeze(), text_embd) for sample in data]
    dataloader = torch.utils.data.DataLoader(transformed_data[:280], batch_size=32, shuffle=True)


    # validation data
    # val_data = load_dataset("Wouter01/re10ksmalltrain")["train"]
    # t_data = [(transform(sample['conditioning_image']).squeeze(), transform(sample['ground_truth_image']).squeeze(), text_embd) for sample in val_data]
    val_loader = torch.utils.data.DataLoader(transformed_data[-20:], batch_size=32, shuffle=False)

    img_loader = torch.utils.data.DataLoader(transformed_data[-3:], batch_size=1, shuffle=False)

    with open(os.path.join(CONFIG), "r") as f:
        config = yaml.safe_load(f)
    
    cfg = dict2namespace(config)
    model = instructir.create_model(input_channels =cfg.model.in_ch, width=cfg.model.width, enc_blks = cfg.model.enc_blks, 
                                middle_blk_num = cfg.model.middle_blk_num, dec_blks = cfg.model.dec_blks, txtdim=cfg.model.textdim)
    model.load_state_dict(torch.load(MODEL_NAME, map_location=device), strict=True)

    model = model.to(device)

    nparams   = count_params (model)
    print ("Loaded weights!", nparams / 1e6)
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss()

    # logging stuff
    os.makedirs("model_out", exist_ok=True)
    repo_id = create_repo(
                repo_id="really_nice_model", exist_ok=True, token=""
            ).repo_id
    
    # Define optimizer (e.g., Adam)
    # use cosine annealing learning rate decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    model.eval()
    val_loss = 0
    # for image, target, prompt in val_loader:
    #     image = image.to(device)
    #     target = target.to(device)
    #     prompt = prompt.to(device)
    #     output = model(image, prompt.squeeze())

    #     loss = criterion(output, target)
    #     val_loss += loss.item()
    # best_val_loss =  val_loss / len(val_loader)
    best_val_loss = 100000
    best_model = None

    n_steps = 0
    # Training loop
    for epoch in range(30):
        model.train()
        # Iterate over dataset
        for input, target, prompt in dataloader:
            input = input.to(device)
            target = target.to(device)
            prompt = prompt.to(device)

            optimizer.zero_grad()

            output = model(input, prompt.squeeze())

            loss = criterion(output, target)

            wandb.log({"loss": loss.item()})

            loss.backward()
            
            optimizer.step()
            n_steps += 1
    
            # validation and printing
            if n_steps % 1 == 0:
                n_steps = 0
                # validation
                model.eval()
                val_loss = 0
                for image, target, prompt in val_loader:
                    image = image.to(device)
                    target = target.to(device)
                    prompt = prompt.to(device)
                    output = model(image, prompt.squeeze())

                    loss = criterion(output, target)
                    val_loss += loss.item()
                val_loss /= len(val_loader)
                wandb.log({"val_loss": val_loss})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model.state_dict()
                    torch.save(best_model, "best_model.pt")

                    # push the model to the hub
                    
                    for _ in range(10):
                        try:
                            api = HfApi()
                            api.upload_file(
                                path_or_fileobj="best_model.pt",
                                path_in_repo="really_nice_model/best_model.pt",
                                repo_id=repo_id,
                            )
                        except:
                            continue
                        break

                # printing
                image_logs = []
                log = []
                for image, target, prompt in img_loader:
                    image = image.to(device)
                    target = target.to(device)
                    prompt = prompt.to(device)
                    output = model(image, prompt.squeeze())

                    wandb.log({"val_loss": criterion(output, target).item()})
                    image_logs.append(
                        {
                            "input": image.squeeze(),
                            "output": output.squeeze(),
                            "validation_prompt": "Please make the image crispier, sharper",
                            "validation_image": target.squeeze(),
                        }
                    )

                    log += [image.squeeze(), output.squeeze(), target.squeeze()]

                wandb.log({"INPUT OUTPUT TARGET": [wandb.Image(image.detach().numpy().transpose((1, 2, 0))) for image in log]})
            # save_model_card(
            #     repo_id,
            #     image_logs=image_logs,
            #     repo_folder="really_nice_model",
            # )
            # upload_folder(
            #     repo_id=repo_id,
            #     folder_path="really_nice_model",
            #     commit_message="End of training",
            #     ignore_patterns=["step_*", "epoch_*"],
            # )
    
if __name__ == "__main__":
    main()