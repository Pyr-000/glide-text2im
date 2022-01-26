# GLIDE

This is a fork of the official codebase for running the small, filtered-data GLIDE model from [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/abs/2112.10741).

For details on the pre-trained models in this repository, see the [Model Card](model-card.md).

# Usage

To install this package, clone this repository and then run:

```
pip install -e .
```

For detailed usage examples, see the [notebooks](notebooks) directory.

 * The [text2im](notebooks/text2im.ipynb) [![][colab]][colab-text2im] notebook shows how to use GLIDE (filtered) with classifier-free guidance to produce images conditioned on text prompts. The local version of this notebook is ``text2im.py``
 * The [inpaint](notebooks/inpaint.ipynb) [![][colab]][colab-inpaint] notebook shows how to use GLIDE (filtered) to fill in a masked region of an image, conditioned on a text prompt. The local version of this notebook is ``inpaint.py``.
 * The [clip_guided](notebooks/clip_guided.ipynb) [![][colab]][colab-guided] notebook shows how to use GLIDE (filtered) + a filtered noise-aware CLIP model to produce images conditioned on text prompts. The local version of this notebook is ``clip_guided.py``.

[colab]: <https://colab.research.google.com/assets/colab-badge.svg>
[colab-text2im]: <https://colab.research.google.com/github/openai/glide-text2im/blob/main/notebooks/text2im.ipynb>
[colab-inpaint]: <https://colab.research.google.com/github/openai/glide-text2im/blob/main/notebooks/inpaint.ipynb>
[colab-guided]: <https://colab.research.google.com/github/openai/glide-text2im/blob/main/notebooks/clip_guided.ipynb>

# Local versions

## Converted Notebooks

The local versions of the notebooks are as close as possible to the original notebooks, which remain unchanged here. Changes to local versions include:

 * No need for "display"
 * Individual images are also saved, as well as the image strip (only upscaled images are saved by default)

## Generation script

Additionally, a more commandline-friendly generation script, ``generate.py``, is available. It can be set to use either classifier-free guidance, or CLIP guidance.

To use the generation script, simply run it with a text prompt as an an additional commandline parameter:
```
python generate.py "Painting of an apple"
```

Parameters for configuring the generation script can be viewed with the `-h` flag:
```
> python generate.py -h
usage: GLIDE Text2Image [-h] [-s S] [-gs GS] [-cf] [-tb TB] [-tu TU] [-ut UT] [-ss] [-ni] prompt

positional arguments:
  prompt

optional arguments:
  -h, --help  show this help message and exit
  -s S        Batch size: Higher values generate more images at once while using more RAM
  -gs GS      Guidance scale parameter during generation (Higher values may improve quality, but reduce diversity)
  -cf         Use classifier-free guidance instead of CLIP guidance. CF guidance may yield 'cleaner' images, while CLIP
              guidance may be better at interpreting more complex prompts.
  -tb TB      Timestep value for base model. For faster generation, lower values (e.g. '100') can be used
  -tu TU      Timestep value for upscaler. For faster generation, use 'fast27'
  -ut UT      Temperature value for the upscaler. '1.0' will result in sharper, but potentially noisier/grainier images
  -ss         Additionally save the small 64x64 images (before the upscaling step)
  -ni         Don't save individual images (after the upscaling step)

Text2Image generation using GLIDE, with classifier-free or CLIP guidance.
```