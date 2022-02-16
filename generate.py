import math
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import torch as th
from datetime import datetime
import torch.nn as nn
import sys
import os
import argparse

from glide_text2im.clip.model_creation import create_clip_model
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)
from glide_text2im.tokenizer.simple_tokenizer import SimpleTokenizer

def parse_args():
    parser = argparse.ArgumentParser(prog="GLIDE Text2Image", epilog="Text2Image generation using GLIDE, with classifier-free or CLIP guidance.")
    parser.add_argument("prompt", nargs='?', default=None, help="Prompt for image generation. Cycles through multiple prompts separated by ||")
    parser.add_argument("-s", default=12, type=int, help="Batch size: Higher values generate more images at once while using more RAM")
    parser.add_argument("-gs", default=3.0, type=float, help="Guidance scale parameter during generation (Higher values may improve quality, but reduce diversity)")
    parser.add_argument("-cf", action="store_true", help="Use classifier-free guidance instead of CLIP guidance. CF guidance may yield 'cleaner' images, while CLIP guidance may be better at interpreting more complex prompts.")
    parser.add_argument("-tb", default='200', help="Timestep value for base model. For faster generation, lower values (e.g. '100') can be used")
    parser.add_argument("-tu", default='100', help="Timestep value for upscaler. For faster generation, use 'fast27'")
    parser.add_argument("-ut", default=0.998, type= float, help="Temperature value for the upscaler. '1.0' will result in sharper, but potentially noisier/grainier images")
    parser.add_argument("-ss", action="store_true", help="Additionally save the small 64x64 images (before the upscaling step)")
    parser.add_argument("-ni", action="store_false", help="Don't save individual images (after the upscaling step)")
    parser.add_argument("-v", action="store_true", help="Verbose mode: print additional runtime information")
    parser.add_argument("-rc", default=None, type=int, help="Amount of different random prompts to use when no prompt is given")
    args = parser.parse_args()
    if args.v:
        print(args)
    return args

def main():
    global ARGS
    ARGS = parse_args()

    # Sampling parameters
    # Use CLIP guidance if enabled. If disabled, classifier-free guidance is used instead
    use_clip = not ARGS.cf
    # Amount of images in batch. Higher values generate more images at once while using more RAM
    batch_size = ARGS.s
    # Guidance scale of either CLIP or classifier-free guidance during generation
    global GUIDANCE_SCALE
    GUIDANCE_SCALE = ARGS.gs
    # Timesteps count for base and upscaler models. For quick generation, use '100', 'fast27'
    timestep_base = ARGS.tb
    timestep_upscale = ARGS.tu
    # Tune this parameter to control the sharpness of 256x256 images. A value of 1.0 is sharper, but sometimes results in grainy artifacts.
    upsample_temp = ARGS.ut
    # Also save direct model outputs before upscaling (64x64)
    save_small_images= ARGS.ss

    # Prompt, requires batch size for autogen
    if ARGS.prompt is None:
        amount = batch_size
        if ARGS.rc is not None:
            amount = ARGS.rc
        ARGS.prompt = random_prompt(amount, ARGS.v)
    global PROMPT
    global PROMPTS
    PROMPT = ARGS.prompt
    PROMPTS = [phrase.strip() for phrase in PROMPT.split("||")]

    # additional image separation (pixels of padding), between grid items.
    global GRID_IMAGE_SEPARATION
    global OUTPATHBASE
    global OUTPATH_INDIVIDUAL
    GRID_IMAGE_SEPARATION = 10
    OUTPATHBASE = "outputs/"
    OUTPATH_INDIVIDUAL = f"{OUTPATHBASE}individual_images/"
    os.makedirs(OUTPATHBASE, exist_ok=True)
    os.makedirs(OUTPATH_INDIVIDUAL, exist_ok=True)

    global DEVICE
    global HAS_CUDA
    HAS_CUDA = th.cuda.is_available()
    DEVICE = th.device('cpu' if not HAS_CUDA else 'cuda')

    # Make a filename
    global MAIN_FILENAME
    MAIN_FILENAME = cleanup_str(PROMPT)
    MAIN_FILENAME += f"-{'CL' if use_clip else 'CF'}gs{GUIDANCE_SCALE:.2f}-ut{upsample_temp:.3f}-{timestep_base}-{timestep_upscale}"

    # setup models
    global GLIDE_MODEL
    global GLIDE_OPTIONS
    GLIDE_OPTIONS, GLIDE_MODEL, diffusion, options_up, model_up, diffusion_up = create_models(
        timestep_base,
        timestep_upscale,
    )
    # create the batch of tokens and masks
    batch_tokens, batch_masks, batch_prompts = create_tokens_batch(
        batch_size,
        ARGS.v
    )
    # set up guidance
    cond_fn, sample_loop_model, sample_loop_batch_size, model_kwargs = create_guidance(
        use_clip,
        batch_size,
        batch_tokens,
        batch_masks,
        batch_prompts
    )
    # sample the base model (generate small images)
    samples = generate_samples(
        use_clip,
        batch_size,
        diffusion,
        cond_fn,
        sample_loop_model,
        sample_loop_batch_size,
        model_kwargs
    )
    # save small (64x64) images if configured
    if save_small_images:
        save_images(samples,"_64")
    # create args for upscaling
    model_kwargs = setup_upscale_args(
        batch_size,
        options_up,
        model_up,
        samples
    )
    # run upscaling
    up_samples = run_upscale(
        batch_size,
        upsample_temp,
        options_up,
        model_up,
        diffusion_up,
        model_kwargs
    )
    # Save the output
    save_images(up_samples)

################################
#       Prompts 
################################

# get prompt for a given batch item index.
# for multiple given prompts, cycle through them.
def get_idx_prompt(index:int):
    global PROMPTS
    return PROMPTS[index%len(PROMPTS)]

# cleanup string for use in a filename
def cleanup_str(s:str):
    new_string = "".join([ char if char.isalnum() else "_" for char in s ])
    # limit to something reasonable
    while "__" in new_string:
        new_string = new_string.replace("__","_")
    return new_string[:32]

# create one or more random prompts
def random_prompt(n:int=1, verbose:bool=False):
    stdout = sys.stdout
    try:
        import words
        if not verbose:
            # block prints of word gen if not verbose
            sys.stdout = None
        phrases = words.get_phrases(n)
        sys.stdout = stdout
        return "||".join(phrases)
    except ImportError:
        # restore stdout
        sys.stdout = stdout
        # no local word generator found, carry on.
        print("No local word generator present.")
        pass
    try:
        from essential_generators import DocumentGenerator
    except ImportError:
        print("Random prompt selection failed: unable to import essential_generators")
        return "House on a hill"
    dg = DocumentGenerator()
    random_words = []
    for _ in range(n):
        words.append(dg.gen_word())
    return random_words

################################
#       Image saving 
################################

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
    # reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])

    stamp = datetime.today().strftime('%Y_%m_%d_%H%M%S')
    # embed parameters in image metadata
    metadata = PngInfo()
    for k in ARGS.__dict__:
        metadata.add_text(key=k, value=repr(ARGS.__dict__[k]))
    pil_images = []
    # Save individual
    for _ in range(0,batch.shape[0]):
        test_single = scaled.select(0,_)
        test_reshape = test_single.permute(1, 2, 0).reshape([batch.shape[2], -1, 3])
        image_item = Image.fromarray(test_reshape.numpy())
        if ARGS.ni:
            image_item.save(f'{OUTPATH_INDIVIDUAL}{stamp}-[{_}]_{cleanup_str(get_idx_prompt(_))}.png', pnginfo=metadata)
        pil_images.append(image_item)
    image_path = f'{OUTPATHBASE}{stamp}{name_suffix}-{MAIN_FILENAME}.png'
    image_autogrid(pil_images).save(image_path, pnginfo=metadata)

################################
#       Tokens 
################################

def tokens_string(tokens, tokenizer):
    token_decoder_dict = tokenizer.decoder
    token_strings = [token_decoder_dict[t] for t in tokens]
    # properly display the leading spaces in tokens
    return [t.replace('\u0120',' ') for t in token_strings]

# Create the text tokens and mask of a given prompt
def get_tokens_mask(prompt:str):
    tokens = GLIDE_MODEL.tokenizer.encode(prompt)
    tokens, mask = GLIDE_MODEL.tokenizer.padded_tokens_and_mask(
        tokens, GLIDE_OPTIONS['text_ctx']
    )
    return tokens,mask

# create the token batch for generation
def create_tokens_batch(batch_size, verbose=False):
    batch_tokens = []
    batch_masks = []
    batch_prompts = []
    for i in range(batch_size):
        this_prompt = get_idx_prompt(i)
        tokens, mask = get_tokens_mask(this_prompt)
        batch_tokens.append(tokens)
        batch_masks.append(mask)
        batch_prompts.append(this_prompt)
    # print out tokenization of each prompt. Only once per unique prompt.
        if verbose and i < len(PROMPTS):
            tokenized_prompt = [t for i,t in enumerate(tokens_string(tokens, GLIDE_MODEL.tokenizer)) if mask[i]]
            print(f"Tokenized String ({len(tokenized_prompt)}/{len(tokens)}):\n{tokenized_prompt}")
    return batch_tokens,batch_masks,batch_prompts

################################
#       Model setup 
################################

def create_models(timestep_base, timestep_upscale):
    # Create base model.
    options = model_and_diffusion_defaults()
    options['use_fp16'] = HAS_CUDA
    options['timestep_respacing'] = timestep_base
    model, diffusion = create_model_and_diffusion(**options)
    model.eval()
    if HAS_CUDA:
        model.convert_to_fp16()
    model.to(DEVICE)
    model.load_state_dict(load_checkpoint('base', DEVICE))
    print('total base parameters', sum(x.numel() for x in model.parameters()))
    # Create upsampler model.
    options_up = model_and_diffusion_defaults_upsampler()
    options_up['use_fp16'] = HAS_CUDA
    options_up['timestep_respacing'] = timestep_upscale
    model_up, diffusion_up = create_model_and_diffusion(**options_up)
    model_up.eval()
    if HAS_CUDA:
        model_up.convert_to_fp16()
    model_up.to(DEVICE)
    model_up.load_state_dict(load_checkpoint('upsample', DEVICE))
    print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))
    return options,model,diffusion,options_up,model_up,diffusion_up

# classifier-free guidance function
def cf_guidance_fn(x_t, ts, **kwargs):
    half = x_t[: len(x_t) // 2]
    combined = th.cat([half, half], dim=0)
    model_out = GLIDE_MODEL(combined, ts, **kwargs)
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + GUIDANCE_SCALE * (cond_eps - uncond_eps)
    eps = th.cat([half_eps, half_eps], dim=0)
    return th.cat([eps, rest], dim=1)

# set-up guidance, either CLIP or classifier-free.
def create_guidance(use_clip, batch_size, batch_tokens, batch_masks, batch_prompts):
    if use_clip:
        # Create CLIP model.
        clip_model = create_clip_model(device=DEVICE)
        clip_model.image_encoder.load_state_dict(load_checkpoint('clip/image-enc', DEVICE))
        clip_model.text_encoder.load_state_dict(load_checkpoint('clip/text-enc', DEVICE))
        model_tokens = th.tensor(batch_tokens, device=DEVICE)
        model_mask = th.tensor(batch_masks, dtype=th.bool, device=DEVICE)
        # Setup guidance function for CLIP model.
        cond_fn = clip_model.cond_fn(batch_prompts, GUIDANCE_SCALE)
        sample_loop_model = GLIDE_MODEL
        sample_loop_batch_size = batch_size
    else:
        # Create the classifier-free guidance tokens (empty)
        full_batch_size = batch_size * 2
        uncond_tokens, uncond_mask = GLIDE_MODEL.tokenizer.padded_tokens_and_mask([], GLIDE_OPTIONS['text_ctx'])
        model_tokens = th.tensor(batch_tokens + [uncond_tokens] * batch_size, device=DEVICE)
        model_mask = th.tensor(batch_masks + [uncond_mask] * batch_size, dtype=th.bool, device=DEVICE)
        cond_fn = None
        sample_loop_model = cf_guidance_fn
        sample_loop_batch_size = full_batch_size

    # Pack the tokens together into model kwargs.
    model_kwargs = dict(
        tokens=model_tokens,
        mask=model_mask,
    )
    
    return cond_fn,sample_loop_model,sample_loop_batch_size,model_kwargs

#  configure model args of the upscaling model
def setup_upscale_args(batch_size, options_up, model_up, samples):
    tokens = model_up.tokenizer.encode(PROMPT)
    tokens, mask = model_up.tokenizer.padded_tokens_and_mask(
        tokens, options_up['text_ctx']
    )
    # Create the model conditioning dict.
    model_kwargs = dict(
    # Low-res image to upsample.
    low_res=((samples+1)*127.5).round()/127.5 - 1,
    # Text tokens
    tokens=th.tensor(
        [tokens] * batch_size, device=DEVICE
    ),
    mask=th.tensor(
        [mask] * batch_size,
        dtype=th.bool,
        device=DEVICE,
    ),
)
    return model_kwargs

################################
#       Run models 
################################

# Sample from the base model.
def generate_samples(use_clip, batch_size, diffusion, cond_fn, sample_loop_model, sample_loop_batch_size, model_kwargs):
    GLIDE_MODEL.del_cache()
    samples = diffusion.p_sample_loop(
        sample_loop_model,
        (sample_loop_batch_size, 3, GLIDE_OPTIONS["image_size"], GLIDE_OPTIONS["image_size"]),
        device=DEVICE,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=cond_fn,
    )
    if not use_clip:
        samples = samples[:batch_size]
    GLIDE_MODEL.del_cache()
    return samples

# Sample from the upscale model.
def run_upscale(batch_size, upsample_temp, options_up, model_up, diffusion_up, model_kwargs):
    model_up.del_cache()
    up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
    up_samples = diffusion_up.ddim_sample_loop(
        model_up,
        up_shape,
        noise=th.randn(up_shape, device=DEVICE) * upsample_temp,
        device=DEVICE,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]
    model_up.del_cache()
    return up_samples

if __name__ == "__main__":
    main()