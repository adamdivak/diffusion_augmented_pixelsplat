# Diffusion augmented pixelSplat

This code builds upon the code from the paper **pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction** by David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann.

Check out their [project website here](https://dcharatan.github.io/pixelsplat).

## Demo

<img src="demo_images/video.gif" width="1000"/>

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

## Demo - in progress
Running the demo:
```bash
python -m src.demo
```

My first impression: the left and right image are not always very similar, possibly leading to worse results (but still need to test this)

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

If you would like to convert downloaded versions of the Real Estate 10k and ACID datasets to our format, you can use the [scripts here](https://github.com/dcharatan/real_estate_10k_tools). Reach out to us if you want the full versions of our processed datasets, which are about 500 GB and 160 GB for Real Estate 10k and ACID respectively.

## Acquiring Pre-trained Checkpoints

You can find pre-trained checkpoints [here](https://drive.google.com/drive/folders/1ZYInQyBHav979dH7arITG8Z-wTSR_Bkm?usp=sharing). You can find the checkpoints for the original codebase (without the improvements from the camera-ready version of the paper) [here](https://drive.google.com/drive/folders/18nGNWIn8RN0aEWLR6MC2mshAkx2uN6fL?usp=sharing).


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

