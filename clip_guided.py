# pip install pyyaml
from PIL import Image
from datetime import datetime

import torch as th
import torch.nn as nn
import sys
import os


OUTPATHBASE = "outputs/"
os.makedirs(OUTPATHBASE, exist_ok=True)

from glide_text2im.clip.model_creation import create_clip_model
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)
from glide_text2im.tokenizer.simple_tokenizer import SimpleTokenizer

# Sampling parameters
prompt = "A woman's face"
batch_size = 12
guidance_scale = 3.0
timestep_base = '200'
timestep_upscale = '100'

if len(sys.argv) > 1:
	prompt = sys.argv[1]
	print(f"using prompt: '{prompt}'")
if len(sys.argv) > 2:
    try:
        float_val = float(sys.argv[2])
        guidance_scale = float_val
    except Exception as e:
        print(f"Could not derive (float) guidance scale from second parameter [{sys.argv[2]}]: {e}")

# Tune this parameter to control the sharpness of 256x256 images.
# A value of 1.0 is sharper, but sometimes results in grainy artifacts.
upsample_temp = 0.997

#########################################
# This notebook supports both CPU and GPU.
# On CPU, generating one sample may take on the order of 20 minutes.
# On a GPU, it should be under a minute.

has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')

# Make a filename
# xprompt = prompt.replace(" ", "_")[:] + "-gs_" + str(guidance_scale)
xprompt = "".join([ char if char.isalnum() else "_" for char in prompt ]) + f"-CLgs{guidance_scale}-ut{upsample_temp}"
# limit filename to something reasonable
while "__" in xprompt:
    xprompt = xprompt.replace("__","_")
xprompt = xprompt[:32]

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

# Create CLIP model.
clip_model = create_clip_model(device=device)
clip_model.image_encoder.load_state_dict(load_checkpoint('clip/image-enc', device))
clip_model.text_encoder.load_state_dict(load_checkpoint('clip/text-enc', device))

# function to create one image containing all input images in a grid.
# currently not intended for images of differing sizes.
def image_autogrid(imgs):
    # additional image separation (pixels of padding), between grid items.
    GRID_IMAGE_SEPARATION = 10
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

def save_images(batch: th.Tensor):
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
        # image_item.save(f'{OUTPATHBASE}{stamp}-[{_}].png')
        pil_images.append(image_item)
    image_autogrid(pil_images).save(f'{OUTPATHBASE}{stamp}-{xprompt}.png')


##############################
# Sample from the base model #
##############################

# Create the text tokens to feed to the model.
tokens = model.tokenizer.encode(prompt)
tokens, mask = model.tokenizer.padded_tokens_and_mask(
    tokens, options['text_ctx']
)

# Pack the tokens together into model kwargs.
model_kwargs = dict(
    tokens=th.tensor([tokens] * batch_size, device=device),
    mask=th.tensor([mask] * batch_size, dtype=th.bool, device=device),
)

# Setup guidance function for CLIP model.
cond_fn = clip_model.cond_fn([prompt] * batch_size, guidance_scale)

# Sample from the base model.
model.del_cache()
samples = diffusion.p_sample_loop(
    model,
    (batch_size, 3, options["image_size"], options["image_size"]),
    device=device,
    clip_denoised=True,
    progress=True,
    model_kwargs=model_kwargs,
    cond_fn=cond_fn,
)
model.del_cache()

# Tiny output - 64x64 - uncomment to save if you like!
#save_images(samples)

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
