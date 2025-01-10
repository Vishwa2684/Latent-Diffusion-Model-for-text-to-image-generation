# Latent Diffusion Model Implementation

This repository contains an implementation of the Latent Diffusion Model based on the paper ["High-Resolution Image Synthesis with Latent Diffusion Models"](https://arxiv.org/abs/2112.10752). The model is trained on COCO 2014 dataset and uses HuggingFace's diffusers library for conditional UNet architecture.

- [To download COCO 2014 images click here](http://images.cocodataset.org/zips/train2014.zip)
- [To download COCO 2014 train/val annotations click here](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)
- [To download model weights click here](https://mega.nz/file/Y0wTxJBA#Y38NS9eX54D8s_kJz0wU485DtJlqUWeSi5pluoju4Ug)

## Model Architecture

- **Backbone**: Conditional UNet from HuggingFace diffusers
- **Scheduler**: Linear scheduler implementation from diffusers
- **VAE**: Pretrained Variational Autoencoder for latent space compression
- **Training**: Mixed precision training for 4 epochs
- **Inference**: Classifier-free guidance for generation
- **CLIP**: Contrastive Language-Image Pre-training for text-image alignment

## Training Details

### Dataset
- COCO 2014 dataset
- Images processed through a pretrained VAE for latent space representation

### Training Configuration
- Number of epochs: 4
- Image size: 64
- Batch size: 8
- Training mode: Mixed precision (FP16)
- Scheduler: Linear noise scheduler
- Conditional generation enabled
- CLIP integration for text-image alignment and conditioning

### Training Workflow
1. **Initial Training** (`test.ipynb`)
   - Contains the initial training setup and execution
   - Implements mixed precision training pipeline
   - Configures CLIP for text-image conditioning

2. **Training Resumption** (`continue.ipynb`)
   - Handles checkpoint loading and training continuation
   - Maintains training state and optimizer settings
   - Allows for training interruption and resumption

## Evaluation

### Metrics to Implement
1. **Image Quality Metrics**
   - FID (Fréchet Inception Distance)
   - LPIPS (Learned Perceptual Image Patch Similarity)
   - IS (Inception Score)

2. **Text-Image Alignment**
   - CLIP Score for text-image similarity
   - Caption similarity metrics
   - Semantic consistency evaluation

3. **Model Performance**
   - Training convergence analysis
   - Loss curves and metrics
   - Classifier-free guidance effectiveness

### Key Components
1. **Variational Autoencoder (VAE)**
   - Pretrained model used for encoding images into latent space
   - Enables efficient training by working in compressed latent space

2. **Conditional UNet**
   - Implements the denoising backbone
   - Sourced from HuggingFace's diffusers library

3. **Linear Scheduler**
   - Manages the noise scheduling during training
   - Implemented using diffusers library components

4. **CLIP Integration**
   - Provides text-image alignment capabilities
   - Enables better conditioning through language understanding
   - Used for evaluation of generation quality

## Inference

Inference details can be found in `eval.ipynb`, which includes:
- Classifier-free guidance implementation
- Generation pipeline
- Sample outputs and evaluations

## Areas for Improvement

1. **Mixed Precision Training**
   - Implement iteration skipping when NaN loss occurs during mixed precision training
   - Current implementation could be made more robust by handling these cases

2. **Image Resolution Scaling**
   - Potential to increase image resolution by scaling the number of latents
   - Would require adjustments to model architecture and training pipeline

3. **Evaluation Pipeline**
   - Implement comprehensive evaluation metrics suite
   - Add automated evaluation scripts
   - Create visualization tools for metric analysis

4. **Future Enhancements**
   - Experiment with different VAE architectures
   - Implement additional conditioning methods
   - Explore alternative noise schedulers
   - Fine-tune CLIP integration for better text-image alignment

## Requirements

```
torch
diffusers
transformers
huggingface-hub
huggingface
numpy
pillow
```

## Usage

### Initial Training

Open and run `test.ipynb` for initial training setup and execution.

### Resume Training

Use `continue.ipynb` to resume training from a checkpoint.

### Evaluation

See `eval.ipynb` for detailed inference examples and generation using classifier-free guidance.


## Citation

```bibtex
@article{rombach2022high,
  title={High-Resolution Image Synthesis with Latent Diffusion Models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  journal={arXiv preprint arXiv:2112.10752},
  year={2021}
}
```

