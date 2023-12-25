"""
stable_diffusion_upscaler.py

This module contains tools and functions for stable diffusion upscaling.
"""

import argparse
import time

import k_diffusion as K
import numpy as np
import torch
from pytorch_lightning import seed_everything
from torchvision.transforms import functional as TF

from models import CFGUpscaler, SdLatentGenerator, get_models
from utils import save_image


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Stable Diffusion Image Generation Script"
    )
    parser.add_argument("--seed", type=int, help="Seed for image generation")
    parser.add_argument(
        "--prompt",
        type=str,
        help="Text prompt for image generation",
        default=(
            "the temple of fire by Ross Tran and Gerardo Dottori, oil on canvas"
        ),
    )
    parser.add_argument(
        "-n", "--num-samples", type=int, default=1, help="Number of samples"
    )
    parser.add_argument(
        "-g", "--guidance-scale", type=float, default=7.5, help="Guidance scale"
    )
    parser.add_argument(
        "-a",
        "--noise-aug-level",
        type=float,
        default=0,
        help="Noise augmentation level",
    )
    parser.add_argument(
        "-t",
        "--noise-aug-type",
        type=str,
        default="gaussian",
        choices=["gaussian", "fake"],
        help="Noise augmentation type",
    )
    parser.add_argument(
        "-s",
        "--sampler",
        type=str,
        default="k_dpm_adaptive",
        choices=[
            "k_euler",
            "k_euler_ancestral",
            "k_dpm_2_ancestral",
            "k_dpm_fast",
            "k_dpm_adaptive",
        ],
        help="Sampler settings",
    )
    parser.add_argument(
        "-st", "--steps", type=int, default=50, help="Number of steps"
    )
    parser.add_argument(
        "-ts",
        "--tol-scale",
        type=float,
        default=0.25,
        help="Error tolerance scale",
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        default=1.0,
        help="Amount of noise to add per step",
    )
    parser.add_argument(
        "-q",
        "--SD-Q",
        type=float,
        default=0.18215,
        help="Scaling for latents in first stage models",
    )
    return parser.parse_args()


@torch.no_grad()
def run(args):
    prompt = args.prompt
    steps = args.steps
    guidance_scale = args.guidance_scale
    noise_aug_level = args.noise_aug_level
    noise_aug_type = args.noise_aug_type
    sampler = args.sampler
    num_samples = args.num_samples
    eta = args.eta
    tol_scale = args.tol_scale
    SD_Q = args.SD_Q

    device = torch.device("cuda")
    batch_size = 1

    timestamp = int(time.time())
    if args.seed is None:
        print("No seed was provided, using the current time.")
        seed = timestamp
    else:
        seed = args.seed
    print(f"Generating with seed={seed}")
    seed_everything(seed)

    vae, model_up, tok_up, text_encoder_up = get_models()

    # Generate the text embeddings for the prompt used in upscaling.
    @torch.no_grad()
    def condition_up(prompts):
        return text_encoder_up(tok_up(prompts))

    uc = condition_up(batch_size * [""])
    c = condition_up(batch_size * [prompt])

    # Generate the low-resolution latents.
    sd_latent_generator = SdLatentGenerator()

    low_res_latent = sd_latent_generator.generate_latent(
        prompt, batch_size, steps, guidance_scale
    )

    print("low_res_latent.shape", low_res_latent.shape)

    [_, C, H, W] = low_res_latent.shape

    # Noise levels from stable diffusion.
    sigma_min, sigma_max = 0.029167532920837402, 14.614642143249512

    model_wrap = CFGUpscaler(model_up, uc, cond_scale=guidance_scale)
    low_res_sigma = torch.full([batch_size], noise_aug_level, device=device)
    x_shape = [batch_size, C, 2 * H, 2 * W]

    def do_sample(noise, extra_args):
        # We take log-linear steps in noise-level from sigma_max to sigma_min,
        # using one of the k diffusion samplers.
        sigmas = (
            torch.linspace(np.log(sigma_max), np.log(sigma_min), steps + 1)
            .exp()
            .to(device)
        )
        if sampler == "k_euler":
            return K.sampling.sample_euler(
                model_wrap, noise * sigma_max, sigmas, extra_args=extra_args
            )
        elif sampler == "k_euler_ancestral":
            return K.sampling.sample_euler_ancestral(
                model_wrap,
                noise * sigma_max,
                sigmas,
                extra_args=extra_args,
                eta=eta,
            )
        elif sampler == "k_dpm_2_ancestral":
            return K.sampling.sample_dpm_2_ancestral(
                model_wrap,
                noise * sigma_max,
                sigmas,
                extra_args=extra_args,
                eta=eta,
            )
        elif sampler == "k_dpm_fast":
            return K.sampling.sample_dpm_fast(
                model_wrap,
                noise * sigma_max,
                sigma_min,
                sigma_max,
                steps,
                extra_args=extra_args,
                eta=eta,
            )
        elif sampler == "k_dpm_adaptive":
            sampler_opts = dict(
                s_noise=1.0,
                rtol=tol_scale * 0.05,
                atol=tol_scale / 127.5,
                pcoeff=0.2,
                icoeff=0.4,
                dcoeff=0,
            )
            return K.sampling.sample_dpm_adaptive(
                model_wrap,
                noise * sigma_max,
                sigma_min,
                sigma_max,
                extra_args=extra_args,
                eta=eta,
                **sampler_opts,
            )

    image_id = 0
    for _ in range((num_samples - 1) // batch_size + 1):
        if noise_aug_type == "gaussian":
            latent_noised = low_res_latent + noise_aug_level * torch.randn_like(
                low_res_latent
            )
        elif noise_aug_type == "fake":
            latent_noised = low_res_latent * (noise_aug_level**2 + 1) ** 0.5
        extra_args = {
            "low_res": latent_noised,
            "low_res_sigma": low_res_sigma,
            "c": c,
        }
        noise = torch.randn(x_shape, device=device)
        up_latents = do_sample(noise, extra_args)

        pixels = vae.decode(
            up_latents / SD_Q
        )  # equivalent to sd_model.decode_first_stage(up_latents)

        pixels = pixels.add(1).div(2).clamp(0, 1)

        # Save samples.
        for j in range(pixels.shape[0]):
            img = TF.to_pil_image(pixels[j])
            save_image(
                img,
                timestamp=timestamp,
                index=image_id,
                prompt=prompt,
                seed=seed,
            )
            image_id += 1


def main():
    args = parse_arguments()
    run(args)

    print(f"Generating image for prompt: {args.prompt} with seed: {args.seed}")


if __name__ == "__main__":
    main()
