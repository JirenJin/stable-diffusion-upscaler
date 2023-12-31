"""Load necessary models for the upscaling process."""

import hashlib
import os
import sys

import huggingface_hub
import k_diffusion as K
import requests
import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, UniPCMultistepScheduler
from omegaconf import OmegaConf
from requests.exceptions import HTTPError
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer, logging

sys.path.extend(
    ["./taming-transformers", "./stable-diffusion", "./latent-diffusion"]
)
# pylint: disable=wrong-import-position
from ldm.util import instantiate_from_config


def fetch(url_or_path):
    if url_or_path.startswith("http:") or url_or_path.startswith("https:"):
        _, ext = os.path.splitext(os.path.basename(url_or_path))
        cachekey = hashlib.md5(url_or_path.encode("utf-8")).hexdigest()
        cachename = f"{cachekey}{ext}"
        if not os.path.exists(f"cache/{cachename}"):
            os.makedirs("tmp", exist_ok=True)
            os.makedirs("cache", exist_ok=True)
            response = requests.get(url_or_path, timeout=600)
            with open(f"tmp/{cachename}", "wb") as f:
                f.write(response.content)
            os.rename(f"tmp/{cachename}", f"cache/{cachename}")
        return f"cache/{cachename}"
    return url_or_path


class NoiseLevelAndTextConditionedUpscaler(nn.Module):
    def __init__(self, inner_model, sigma_data=1.0, embed_dim=256):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = sigma_data
        self.low_res_noise_embed = K.layers.FourierFeatures(1, embed_dim, std=2)

    def forward(self, input, sigma, low_res, low_res_sigma, c, **kwargs):
        cross_cond, cross_cond_padding, pooler = c
        c_in = 1 / (low_res_sigma**2 + self.sigma_data**2) ** 0.5
        c_noise = low_res_sigma.log1p()[:, None]
        c_in = K.utils.append_dims(c_in, low_res.ndim)
        low_res_noise_embed = self.low_res_noise_embed(c_noise)
        low_res_in = (
            F.interpolate(low_res, scale_factor=2, mode="nearest") * c_in
        )
        mapping_cond = torch.cat([low_res_noise_embed, pooler], dim=1)
        return self.inner_model(
            input,
            sigma,
            unet_cond=low_res_in,
            mapping_cond=mapping_cond,
            cross_cond=cross_cond,
            cross_cond_padding=cross_cond_padding,
            **kwargs,
        )


def make_upscaler_model(
    config_path, model_path, pooler_dim=768, train=False, device="cpu"
):
    config = K.config.load_config(open(config_path, encoding="utf-8"))
    model = K.config.make_model(config)
    model = NoiseLevelAndTextConditionedUpscaler(
        model,
        sigma_data=config["model"]["sigma_data"],
        embed_dim=config["model"]["mapping_cond_dim"] - pooler_dim,
    )
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["model_ema"])
    model = K.config.make_denoiser_wrapper(config)(model)
    if not train:
        model = model.eval().requires_grad_(False)
    return model.to(device)


def download_from_huggingface(repo, filename):
    while True:
        try:
            return huggingface_hub.hf_hub_download(repo, filename)
        except HTTPError as e:
            if e.response.status_code == 401:
                # Need to log into huggingface api
                huggingface_hub.interpreter_login()
                continue
            elif e.response.status_code == 403:
                # Need to do the click through license thing
                print(
                    "Go here and agree to the click through license on your"
                    f" account: https://huggingface.co/{repo}"
                )
                input("Hit enter when ready:")
                continue
            else:
                raise e


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model = model.to(torch.device("cpu")).eval().requires_grad_(False)
    return model


class CFGUpscaler(nn.Module):
    def __init__(self, model, uc, cond_scale):
        super().__init__()
        self.inner_model = model
        self.uc = uc
        self.cond_scale = cond_scale

    def forward(self, x, sigma, low_res, low_res_sigma, c):
        if self.cond_scale in (0.0, 1.0):
            # Shortcut for when we don't need to run both.
            if self.cond_scale == 0.0:
                c_in = self.uc
            elif self.cond_scale == 1.0:
                c_in = c
            return self.inner_model(
                x, sigma, low_res=low_res, low_res_sigma=low_res_sigma, c=c_in
            )

        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        low_res_in = torch.cat([low_res] * 2)
        low_res_sigma_in = torch.cat([low_res_sigma] * 2)
        c_in = [
            torch.cat([uc_item, c_item]) for uc_item, c_item in zip(self.uc, c)
        ]
        uncond, cond = self.inner_model(
            x_in,
            sigma_in,
            low_res=low_res_in,
            low_res_sigma=low_res_sigma_in,
            c=c_in,
        ).chunk(2)
        return uncond + (cond - uncond) * self.cond_scale


class CLIPTokenizerTransform:
    def __init__(self, version="openai/clip-vit-large-patch14", max_length=77):
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.max_length = max_length

    def __call__(self, text):
        indexer = 0 if isinstance(text, str) else ...
        tok_out = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tok_out["input_ids"][indexer]
        attention_mask = 1 - tok_out["attention_mask"][indexer]
        return input_ids, attention_mask


class CLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda"):
        super().__init__()

        logging.set_verbosity_error()
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.transformer = (
            self.transformer.eval().requires_grad_(False).to(device)
        )

    @property
    def device(self):
        return self.transformer.device

    def forward(self, tok_out):
        input_ids, cross_cond_padding = tok_out
        clip_out = self.transformer(
            input_ids=input_ids.to(self.device), output_hidden_states=True
        )
        return (
            clip_out.hidden_states[-1],
            cross_cond_padding.to(self.device),
            clip_out.pooler_output,
        )


def get_models():
    # Load models on GPU
    device = torch.device("cuda")

    vae_840k_model_path = download_from_huggingface(
        "stabilityai/sd-vae-ft-mse-original",
        "vae-ft-mse-840000-ema-pruned.ckpt",
    )
    vae_model_840k = load_model_from_config(
        "latent-diffusion/models/first_stage_models/kl-f8/config.yaml",
        vae_840k_model_path,
    )
    vae_model = vae_model_840k.to(device)

    model_up = make_upscaler_model(
        fetch(
            "https://models.rivershavewings.workers.dev/config_laion_text_cond_latent_upscaler_2.json"
        ),
        fetch(
            "https://models.rivershavewings.workers.dev/laion_text_cond_latent_upscaler_2_1_00470000_slim.pth"
        ),
    )
    model_up = model_up.to(device)

    tok_up = CLIPTokenizerTransform()
    text_encoder_up = CLIPEmbedder(device=device)

    return vae_model, model_up, tok_up, text_encoder_up


class SdLatentGenerator:
    """Generates the low resolution latent vector from a text prompt."""

    def __init__(self, device=torch.device("cuda")):
        self.unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="unet",
            use_safetensors=True,
        )

        self.scheduler = UniPCMultistepScheduler.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
        )

        self.tokenizer = CLIPTokenizer.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="text_encoder",
            use_safetensors=True,
        )

        self.text_encoder.to(device)
        self.unet.to(device)

    def generate_latent(
        self,
        prompt,
        batch_size,
        steps,
        guidance_scale,
        device=torch.device("cuda"),
    ):
        height = 512  # default height of Stable Diffusion
        width = 512  # default width of Stable Diffusion

        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_input.input_ids.to(device)
            )[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(
            uncond_input.input_ids.to(device)
        )[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, height // 8, width // 8),
            device=device,
        )

        latents = latents * self.scheduler.init_noise_sigma
        print("SD init_noise_sigma", self.scheduler.init_noise_sigma)

        self.scheduler.set_timesteps(steps)

        for t in self.scheduler.timesteps:
            # Expand the latents if we are doing classifier-free guidance
            # to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, timestep=t
            )

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents
