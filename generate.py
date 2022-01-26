import math
from PIL import Image
import torch as th
from datetime import datetime
import torch.nn as nn
import sys
import os
import argparse

parser = argparse.ArgumentParser(prog="GLIDE Text2Image", epilog="Text2Image generation using GLIDE, with classifier-free or CLIP guidance.")
parser.add_argument("prompt", default="House on a hill")
parser.add_argument("-s", default=12, type=int, help="Batch size: Higher values generate more images at once while using more RAM")
parser.add_argument("-gs", default=3.0, type=float, help="Guidance scale parameter during generation (Higher values may improve quality, but reduce diversity)")
parser.add_argument("-cf", action="store_true", help="Use classifier-free guidance instead of CLIP guidance. CF guidance may yield 'cleaner' images, while CLIP guidance may be better at interpreting more complex prompts.")
parser.add_argument("-tb", default='200', help="Timestep value for base model. For faster generation, lower values (e.g. '100') can be used")
parser.add_argument("-tu", default='100', help="Timestep value for upscaler. For faster generation, use 'fast27'")
parser.add_argument("-ut", default=0.997, type= float, help="Temperature value for the upscaler. '1.0' will result in sharper, but potentially noisier/grainier images")
parser.add_argument("-ss", action="store_true", help="Additionally save the small 64x64 images (before the upscaling step)")
parser.add_argument("-ni", action="store_false", help="Don't save individual images (after the upscaling step)")

args = parser.parse_args()
print(args)
OUTPATHBASE = "outputs/"
OUTPATH_INDIVIDUAL = f"{OUTPATHBASE}individual_images/"
os.makedirs(OUTPATHBASE, exist_ok=True)
os.makedirs(OUTPATH_INDIVIDUAL, exist_ok=True)

from glide_text2im.clip.model_creation import create_clip_model
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)
from glide_text2im.tokenizer.simple_tokenizer import SimpleTokenizer

# Sampling parameters
# Default prompt
prompt = args.prompt
# Use CLIP guidance if enabled. If disabled, classifier-free guidance is used instead
use_clip = not args.cf
# Amount of images in batch. Higher values generate more images at once while using more RAM
batch_size = args.s
# Guidance scale of either CLIP or classifier-free guidance during generation
guidance_scale = args.gs
# Timesteps count for base and upscaler models. For quick generation, use '100', 'fast27'
timestep_base = args.tb
timestep_upscale = args.tu
# Tune this parameter to control the sharpness of 256x256 images. A value of 1.0 is sharper, but sometimes results in grainy artifacts.
upsample_temp = args.ut
# Also save direct model outputs before upscaling (64x64)
save_small_images= args.ss
# Also save the individual images 
save_individual_images = args.ni

# additional image separation (pixels of padding), between grid items.
GRID_IMAGE_SEPARATION = 10

#########################################
# This notebook supports both CPU and GPU.
# On CPU, generating one sample may take on the order of 20 minutes.
# On a GPU, it should be under a minute.

has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')

# Make a filename
# xprompt = prompt.replace(" ", "_")[:] + "-gs_" + str(guidance_scale)
xprompt = "".join([ char if char.isalnum() else "_" for char in prompt ])
# limit filename to something reasonable
while "__" in xprompt:
    xprompt = xprompt.replace("__","_")
xprompt = xprompt[:32]
xprompt += f"-{'CL' if use_clip else 'CF'}gs{guidance_scale:.2f}-ut{upsample_temp:.3f}-{timestep_base}-{timestep_upscale}"

# Create base model.
options = model_and_diffusion_defaults()
options['use_fp16'] = has_cuda
options['timestep_respacing'] = timestep_base
model, diffusion = create_model_and_diffusion(**options)
model.eval()
if has_cuda:
    model.convert_to_fp16()
model.to(device)
model.load_state_dict(load_checkpoint('base', device))
print('total base parameters', sum(x.numel() for x in model.parameters()))

# Create upsampler model.
options_up = model_and_diffusion_defaults_upsampler()
options_up['use_fp16'] = has_cuda
# options_up['timestep_respacing'] = 'fast27' # use 27 diffusion steps for very fast sampling
options_up['timestep_respacing'] = timestep_upscale
model_up, diffusion_up = create_model_and_diffusion(**options_up)
model_up.eval()
if has_cuda:
    model_up.convert_to_fp16()
model_up.to(device)
model_up.load_state_dict(load_checkpoint('upsample', device))
print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))


# function to create one image containing all input images in a grid.
# currently not intended for images of differing sizes.
def image_autogrid(imgs):
    side_len = math.sqrt(len(imgs))
    # round up cols from square root, attempt to round down rows
    # if required to actually fit all images, both cols and rows are rounded up.
    cols = math.ceil(side_len)
    rows = math.floor(side_len)
    if (rows*cols) < len(imgs):
        rows = math.ceil(side_len)
    # get grid item size from first image
    w, h = imgs[0].size
    # add separation to size between images as 'padding'
    w += GRID_IMAGE_SEPARATION
    h += GRID_IMAGE_SEPARATION
    # remove one image separation size from the overall size (no added padding after the final row/col)
    grid = Image.new('RGB', size=(cols*w-GRID_IMAGE_SEPARATION, rows*h-GRID_IMAGE_SEPARATION))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def save_images(batch: th.Tensor, name_suffix=""):
    """ Save images """
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])

    stamp = datetime.today().strftime('%Y_%m_%d_%H%M%S')
    pil_images = []
    # Save individual
    for _ in range(0,batch.shape[0]):
        test_single = scaled.select(0,_)
        test_reshape = test_single.permute(1, 2, 0).reshape([batch.shape[2], -1, 3])
        image_item = Image.fromarray(test_reshape.numpy())
        if save_individual_images:
            image_item.save(f'{OUTPATH_INDIVIDUAL}{stamp}-[{_}].png')
        pil_images.append(image_item)
    image_autogrid(pil_images).save(f'{OUTPATHBASE}{stamp}{name_suffix}-{xprompt}.png')


##############################
# Sample from the base model #
##############################

# Create the text tokens to feed to the model.
tokens = model.tokenizer.encode(prompt)
tokens, mask = model.tokenizer.padded_tokens_and_mask(
    tokens, options['text_ctx']
)

if use_clip:
    # Create CLIP model.
    clip_model = create_clip_model(device=device)
    clip_model.image_encoder.load_state_dict(load_checkpoint('clip/image-enc', device))
    clip_model.text_encoder.load_state_dict(load_checkpoint('clip/text-enc', device))
    model_tokens = th.tensor([tokens] * batch_size, device=device)
    model_mask = th.tensor([mask] * batch_size, dtype=th.bool, device=device)
    # Setup guidance function for CLIP model.
    cond_fn = clip_model.cond_fn([prompt] * batch_size, guidance_scale)
    sample_loop_model = model
    sample_loop_batch_size = batch_size
else:
    # Create a classifier-free guidance sampling function
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return th.cat([eps, rest], dim=1)
    # Create the classifier-free guidance tokens (empty)
    full_batch_size = batch_size * 2
    uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
        [], options['text_ctx']
    )
    model_tokens = th.tensor([tokens] * batch_size + [uncond_tokens] * batch_size, device=device)
    model_mask = th.tensor(
        [mask] * batch_size + [uncond_mask] * batch_size,
        dtype=th.bool,
        device=device,
    )
    cond_fn = None
    sample_loop_model = model_fn
    sample_loop_batch_size = full_batch_size

# Pack the tokens together into model kwargs.
model_kwargs = dict(
    tokens=model_tokens,
    mask=model_mask,
)

# Sample from the base model.
model.del_cache()
samples = diffusion.p_sample_loop(
    sample_loop_model,
    (sample_loop_batch_size, 3, options["image_size"], options["image_size"]),
    device=device,
    clip_denoised=True,
    progress=True,
    model_kwargs=model_kwargs,
    cond_fn=cond_fn,
)
if not use_clip:
    samples = samples[:batch_size]
model.del_cache()

# Tiny output - 64x64 - uncomment to save if you like!
if save_small_images:
	save_images(samples,"_64")

##############################
# Upsample the 64x64 samples #
##############################

tokens = model_up.tokenizer.encode(prompt)
tokens, mask = model_up.tokenizer.padded_tokens_and_mask(
    tokens, options_up['text_ctx']
)

# Create the model conditioning dict.
model_kwargs = dict(
    # Low-res image to upsample.
    low_res=((samples+1)*127.5).round()/127.5 - 1,

    # Text tokens
    tokens=th.tensor(
        [tokens] * batch_size, device=device
    ),
    mask=th.tensor(
        [mask] * batch_size,
        dtype=th.bool,
        device=device,
    ),
)

# Sample from the base model.
model_up.del_cache()
up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
up_samples = diffusion_up.ddim_sample_loop(
    model_up,
    up_shape,
    noise=th.randn(up_shape, device=device) * upsample_temp,
    device=device,
    clip_denoised=True,
    progress=True,
    model_kwargs=model_kwargs,
    cond_fn=None,
)[:batch_size]
model_up.del_cache()

# Save the output
save_images(up_samples)
