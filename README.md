# Diffusion augmented pixelSplat

This code builds upon the code from the paper **pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction** by David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann.

Check out their [project website here](https://dcharatan.github.io/pixelsplat).

## TODO
- Consider which diffusion model to use
- Find a way to process the full data in a meaningful and possible way

## Video (click to watch)
<a href="https://youtu.be/yOB8nqlEnpw">
  <img src="https://img.youtube.com/vi/yOB8nqlEnpw/0.jpg" width="800"/>
</a>
  
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

## Camera-ready Updates

* A configuration for 3-view pixelSplat has been added. In general, it's now possible to run pixelSplat with an arbitrary number of views, although you'll need a lot of GPU memory to do so.

## Installation

### Installation on Snellius supercomputer
This is not straightforward as we don't have sudo privileges and many default packages are outdated.
Also, new versions of g++ are not compatible.
After cloning the repo, execute the following commands, in order and only after the previous command is finished:

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


### Original instructions
To get started, create a virtual environment using Python 3.10+:

```bash
python3.10 -m venv venv
source venv/bin/activate
# Install these first! Also, make sure you have python3.11-dev installed if using Ubuntu.
pip install wheel torch torchvision torchaudio
pip install -r requirements.txt
```

## Acquiring Datasets

pixelSplat was trained using versions of the RealEstate10k and ACID datasets that were split into ~100 MB chunks for use on server cluster file systems. Small subsets of the Real Estate 10k and ACID datasets in this format can be found [here](https://drive.google.com/drive/folders/1joiezNCyQK2BvWMnfwHJpm2V77c7iYGe?usp=sharing). To use them, simply unzip them into a newly created `datasets` folder in the project root directory.

If you would like to convert downloaded versions of the Real Estate 10k and ACID datasets to our format, you can use the [scripts here](https://github.com/dcharatan/real_estate_10k_tools). Reach out to us if you want the full versions of our processed datasets, which are about 500 GB and 160 GB for Real Estate 10k and ACID respectively.

## Acquiring Pre-trained Checkpoints

You can find pre-trained checkpoints [here](https://drive.google.com/drive/folders/1ZYInQyBHav979dH7arITG8Z-wTSR_Bkm?usp=sharing). You can find the checkpoints for the original codebase (without the improvements from the camera-ready version of the paper) [here](https://drive.google.com/drive/folders/18nGNWIn8RN0aEWLR6MC2mshAkx2uN6fL?usp=sharing).

## Running the Code

### Training

The main entry point is `src/main.py`. Call it via:

```bash
python3 -m src.main +experiment=re10k
```

This configuration requires a single GPU with 80 GB of VRAM (A100 or H100). To reduce memory usage, you can change the batch size as follows:

```bash
python3 -m src.main +experiment=re10k data_loader.train.batch_size=1
```

Our code supports multi-GPU training. The above batch size is the per-GPU batch size.

### Evaluation

To render frames from an existing checkpoint, run the following:

```bash
# Real Estate 10k
python3 -m src.main +experiment=re10k mode=test dataset/view_sampler=evaluation dataset.view_sampler.index_path=assets/evaluation_index_re10k.json checkpointing.load=checkpoints/re10k.ckpt

# ACID
python3 -m src.main +experiment=acid mode=test dataset/view_sampler=evaluation dataset.view_sampler.index_path=assets/evaluation_index_acid.json checkpointing.load=checkpoints/acid.ckpt
```

Note that you can also use the evaluation indices that end with `_video` (in `/assets`) to render the videos shown on the website.

### Ablations

You can run the ablations from the paper by using the corresponding experiment configurations. For example, to ablate the epipolar encoder:

```bash
python3 -m src.main +experiment=re10k_ablation_no_epipolar_transformer
```

Our collection of pre-trained [checkpoints](https://drive.google.com/drive/folders/18nGNWIn8RN0aEWLR6MC2mshAkx2uN6fL?usp=sharing) includes checkpoints for the ablations.

### VS Code Launch Configuration

We provide VS Code launch configurations for easy debugging.

## Camera Conventions

Our extrinsics are OpenCV-style camera-to-world matrices. This means that +Z is the camera look vector, +X is the camera right vector, and -Y is the camera up vector. Our intrinsics are normalized, meaning that the first row is divided by image width, and the second row is divided by image height.

## Figure Generation Code

We've included the scripts that generate tables and figures in the paper. Note that since these are one-offs, they might have to be modified to be run.

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

