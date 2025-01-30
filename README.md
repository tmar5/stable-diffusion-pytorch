# Stable Diffusion Implementation in PyTorch

Welcome to the Stable Diffusion implementation in PyTorch!

## Papers
- **Stable Diffusion:** [Stable Diffusion Paper](https://arxiv.org/pdf/2006.11239)
- **DDPM Scheduler:** [DDPM Paper](https://arxiv.org/pdf/2006.11239)

## Instructions

1. **Download the checkpoint file** based on SD1.5 and place it in the `data` folder.
   - Example: [v1-5-pruned-emaonly.ckpt](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt)

2. **Update the checkpoint file path** in `test.ipynb`.

3. **Run `test.ipynb` and enjoy!** 🎨

## Img2Img Pipeline
To activate the Img2Img pipeline, provide an `input_image` as input.

## Sources
- [Stable Diffusion (CompVis)](https://github.com/CompVis/stable-diffusion/)
- [Stable Diffusion PyTorch (kjsman)](https://github.com/kjsman/stable-diffusion-pytorch)
- [PyTorch Stable Diffusion (hkproj)](https://github.com/hkproj/pytorch-stable-diffusion)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers/)
