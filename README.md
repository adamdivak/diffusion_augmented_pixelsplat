# Improving novel view synthesis of 3D Gaussian splats using 2D image enhancement methods
### Wouter Bant, Ádám Divák, Jasper Eppink, Clio Feng, Roos Hutter

*Novel view reconstruction based on only 2 input images is an important but extremely challenging task. pixelSplat is a potential solution that was shown to deliver high-quality results at a competitive speed. We first evaluate pixelSplat on more challenging reconstruction tasks by applying cam- era positions that are further away from each other, and find that its performance is heavily impacted. We then explore 2D image enhancement methods to fix the corrupted novel view images. A diffusion model-based solution proves to be able to restore significantly impacted areas, but fails to stay consistent with the original scene even after long fine- tuning, resulting in flickering videos. An alternative solu- tion based on an image restoration model results in pleasant videos and quantitative improvements in most metrics, but does not address all errors seen in the novel view images. We explore the underlying reasons for these shortcomings, and propose future research directions for fixing them.*

This code builds upon the code from the paper **pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction** by David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. 

Check out their [project website here](https://dcharatan.github.io/pixelsplat).

## Demo

https://github.com/WouterBant/diffusion-augmented-pixelsplat/assets/73896544/790b1a18-75e1-48f8-aef6-f7f9581e1c90

## Important files

- [Training ControlNet](controlnet/train_controlnet.py)
- [Training InstructIR](instructir/train.py): note the authors didn't provide a training script so we made one based on the information from [their paper](https://arxiv.org/abs/2401.16468)
- [Inference](instructir/inference.py)
- [Testing ControlNet on out of domain data](instructir/demo_models.ipynb)
- [Creating demo video](src/video.ipynb)
  
## Export training images
```bash
!python3 -m src.main +experiment=re10k mode=test test.data_loader="train" test.output_path="outputs/re10k_train_data" data_loader.train.batch_size=1 checkpointing.load=checkpoints/re10k.ckpt
```

## Installation

### Installation on Snellius supercomputer
This is not straightforward as we don't have sudo privileges and many default packages are outdated.
Also, new versions of g++ are not compatible.
After cloning the repo, execute the following commands, in order and only after the previous command is finished:

<details>
  <summary>Step by step instructions</summary>
  
  ```bash
  cd installation_jobs
  ``` 
  This takes approximately 30 minutes, all others are much faster.
  ```bash
  sbatch install_env.job
  ```
  
  This will return an error but we will fix this afterwards.
  ```bash
  sbatch install_packages.job
  ```
  
  Debugging jobs:
  ```bash
  sbatch debug.job
  ```
  ```bash
  sbatch debug2.job
  ```
  ```bash
  sbatch debug3.job
  ```
  ```bash
  sbatch debug4.job
  ```
  ```bash
  sbatch debug5.job
  ```
  
  Now this should run without any errors.
  ```bash
  sbatch install_packages.job
  ```
</details>

## Acquiring Datasets

pixelSplat was trained using versions of the RealEstate10k and ACID datasets that were split into ~100 MB chunks for use on server cluster file systems. Small subsets of the Real Estate 10k and ACID datasets in this format can be found [here](https://drive.google.com/drive/folders/1joiezNCyQK2BvWMnfwHJpm2V77c7iYGe?usp=sharing). To use them, simply unzip them into a newly created `datasets` folder in the project root directory.

The datasets that were used to finetune the diffusion model and InstructIR can be found on Huggingface (https://huggingface.co/datasets/Wouter01/re10k_hard)

## Acquiring Pre-trained Checkpoints

You can find pre-trained checkpoints [here](https://drive.google.com/drive/folders/1ZYInQyBHav979dH7arITG8Z-wTSR_Bkm?usp=sharing). You can find the checkpoints for the original codebase (without the improvements from the camera-ready version of the paper) [here](https://drive.google.com/drive/folders/18nGNWIn8RN0aEWLR6MC2mshAkx2uN6fL?usp=sharing).

Also the finetuned diffusion and InstructIR models can be found on Huggingface (https://huggingface.co/Wouter01/diffusion_re10k_hard, https://huggingface.co/Wouter01/InstructIR_re10k_hard)

## Citation

```
@misc{vlasenko2024efficient,
  title={Improving novel view synthesis of 3D Gaussian splats using 2D image enhancement methods},
  author={Wouter Bant and Ádám Divák and Jasper Eppink and Clio Feng and Roos Hutter},
  year={2024},
  url={https://github.com/adamdivak/diffusion_augmented_pixelsplat}
}
```

## Acknowledgements

This code is mainly from https://dcharatan.github.io/pixelsplat
```
@inproceedings{charatan23pixelsplat,
      title={pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction},
      author={David Charatan and Sizhe Li and Andrea Tagliasacchi and Vincent Sitzmann},
      year={2023},
      booktitle={arXiv},
}
```

