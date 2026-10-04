"""Persist the SAME-S architecture details missing from Diffusers 0.40.0.

The SA3 DiT, sampler, tokenizer and pipeline remain native Diffusers. Only SAME
resampling blocks are extended: Small uses two shifted attention chunks and a
3-tap decoder mapping, while Medium uses the native sliding window. Reference:
https://github.com/Stability-AI/stable-audio-3/blob/main/stable_audio_3/models/autoencoders.py
https://huggingface.co/stabilityai/stable-audio-3-small-music-base/blob/main/model_config.json
No installed package files are edited. Extra architecture fields are saved in
vae/config.json and registered before converting or loading that directory.
"""

from types import MethodType


def same_config_fields(model_config: dict) -> dict:
    """Require the actual reference config instead of guessing Small's layers."""
    ae = model_config["model"]["pretransform"]["config"]
    result = {}
    for side in ("encoder", "decoder"):
        config = ae[side]["config"]
        window = config.get("sliding_window")
        if window is not None and (len(window) != 2 or window[0] != window[1]):
            raise ValueError("Asymmetric SAME sliding windows are not supported")
        chunk_size = int(config.get("chunk_size", 128)) if window is None else 0
        strides = list(config["strides"])
        if not config.get("variable_stride", False):
            raise ValueError("SAME conversion requires variable_stride=true")
        if chunk_size and any(chunk_size % int(stride) for stride in strides):
            raise ValueError("SAME chunk_size must be divisible by each stride")
        result.update({
            f"{side}_chunk_size": chunk_size,
            f"{side}_chunk_midpoint_shift": bool(config.get("chunk_midpoint_shift", False)),
            f"{side}_mapping_kernel": 3 if config.get("conv_mapping", False) else 1,
            f"{side}_mask_noise": float(config.get("mask_noise", 0.0)),
        })
    return result


def _run_chunks(sequence, layers, chunk_tokens: int, midpoint_shift: bool):
    """Run local attention, then repeat the end subsegments for the shifted half."""
    import torch

    batch, length, channels = sequence.shape
    if length % chunk_tokens:
        raise ValueError("SAME token length must be divisible by chunk size")

    def run(value, chosen):
        total = value.shape[1]
        value = value.reshape(batch * (total // chunk_tokens), chunk_tokens, channels)
        for layer in chosen:
            value = layer(value)
        return value.reshape(batch, total, channels)

    if not midpoint_shift:
        return run(sequence, layers)
    midpoint = len(layers) // 2
    sequence = run(sequence, layers[:midpoint])
    shift = chunk_tokens // 2
    padded = torch.cat((sequence[:, :shift], sequence, sequence[:, -shift:]), dim=1)
    return run(padded, layers[midpoint:])[:, shift:-shift]


def _chunked_block_forward(block, value):
    import torch
    from torch.nn import functional

    batch, _, length = value.shape
    stride = block.stride
    is_encoder = block.mode == "encoder"
    padding_multiple = (
        block.sa3_chunk_size if is_encoder else block.sa3_chunk_size // stride
    ) if block.sa3_chunk_size else (stride if is_encoder else 1)
    value = functional.pad(value, (0, (-length) % padding_multiple))
    if is_encoder:
        value = block.mapping(value)
    sequence = value.transpose(1, 2)
    channels = sequence.shape[-1]
    segment_input = stride if is_encoder else 1
    segments = sequence.shape[1] // segment_input
    sequence = sequence.reshape(batch * segments, segment_input, channels)
    segment_output = 1 if is_encoder else stride
    tokens = block.new_tokens.expand(batch * segments, segment_output, channels)
    if block.sa3_mask_noise:
        tokens = tokens + torch.randn_like(tokens) * block.sa3_mask_noise
    sequence = torch.cat((sequence, tokens), dim=1)
    segment_tokens = stride + 1
    sequence = sequence.reshape(batch, segments * segment_tokens, channels)
    if block.sa3_chunk_size:
        chunk_tokens = block.sa3_chunk_size * segment_tokens // stride
        sequence = _run_chunks(
            sequence, block.transformers, chunk_tokens, block.sa3_chunk_midpoint_shift
        )
    else:
        from diffusers.models.autoencoders.autoencoder_same import _band_mask

        mask = _band_mask(
            sequence.shape[1], block.sliding_window * segment_tokens,
            sequence.device, sequence.dtype,
        )
        for layer in block.transformers:
            sequence = layer(sequence, attn_mask=mask)
    sequence = sequence.reshape(batch * segments, segment_tokens, channels)
    sequence = sequence[:, -segment_output:].reshape(batch, segments * segment_output, channels)
    value = sequence.transpose(1, 2)
    return value if is_encoder else block.mapping(value)


def install_same_compat():
    """Extend native SAME config before Diffusers from_pretrained reads weights."""
    from diffusers import AutoencoderSAME
    from diffusers.configuration_utils import register_to_config
    from torch import nn
    from torch.nn.utils import weight_norm

    if getattr(AutoencoderSAME, "_sa3_small_compat", False):
        return
    original_init = AutoencoderSAME.__init__

    @register_to_config
    def compatible_init(
        self, audio_channels=2, patch_size=256, encoder_channels=128,
        encoder_c_mults=(6,), encoder_strides=(16,), encoder_transformer_depths=(6,),
        latent_dim=256, use_differential_attention=True, dim_heads=64, ff_mult=3,
        sliding_window=1, encoder_sinusoidal_blocks=(0,), decoder_sinusoidal_blocks=(0,),
        sampling_rate=44100, encoder_chunk_size=0, decoder_chunk_size=0,
        encoder_chunk_midpoint_shift=False, decoder_chunk_midpoint_shift=False,
        encoder_mapping_kernel=1, decoder_mapping_kernel=1,
        encoder_mask_noise=0.0, decoder_mask_noise=0.0,
    ):
        original_init(
            self, audio_channels=audio_channels, patch_size=patch_size,
            encoder_channels=encoder_channels, encoder_c_mults=encoder_c_mults,
            encoder_strides=encoder_strides, encoder_transformer_depths=encoder_transformer_depths,
            latent_dim=latent_dim, use_differential_attention=use_differential_attention,
            dim_heads=dim_heads, ff_mult=ff_mult, sliding_window=sliding_window,
            encoder_sinusoidal_blocks=encoder_sinusoidal_blocks,
            decoder_sinusoidal_blocks=decoder_sinusoidal_blocks, sampling_rate=sampling_rate,
        )
        settings = (
            (self.encoder, encoder_chunk_size, encoder_chunk_midpoint_shift,
             encoder_mapping_kernel, encoder_mask_noise),
            (self.decoder, decoder_chunk_size, decoder_chunk_midpoint_shift,
             decoder_mapping_kernel, decoder_mask_noise),
        )
        for component, chunk, midpoint, kernel, noise in settings:
            if kernel not in (1, 3) or chunk < 0 or noise < 0:
                raise ValueError("Invalid persisted SAME architecture settings")
            for block in component.blocks:
                if kernel == 3 and block.in_channels != block.out_channels:
                    block.mapping = weight_norm(nn.Conv1d(
                        block.in_channels, block.out_channels, 3, padding=1
                    ))
                if chunk and (chunk % block.stride or (midpoint and (chunk // block.stride) % 2)):
                    raise ValueError("SAME chunk size does not support the requested stride/shift")
                if chunk or noise:
                    block.sa3_chunk_size = chunk
                    block.sa3_chunk_midpoint_shift = midpoint
                    block.sa3_mask_noise = noise
                    block.forward = MethodType(_chunked_block_forward, block)

    AutoencoderSAME.__init__ = compatible_init
    AutoencoderSAME._sa3_small_compat = True


def enable_chunked_decode(vae, chunk_size=128, overlap=32):
    """Bound SAME attention memory by decoding overlapping latent windows.

    Crop half the overlap at internal boundaries, as the official SA3 runtime
    does. Each sample retains at least 16 surrounding latents for the default
    12-layer Medium decoder. Keep even starts for SAME-S shifted chunk alignment.
    """
    if chunk_size <= overlap or overlap < 0 or chunk_size % 2 or overlap % 2:
        raise ValueError("SAME decode requires even chunk size/overlap, chunk_size > overlap")
    if getattr(vae, "_sa3_chunked_decode", False):
        return
    original_decode = vae.decode

    def decode(self, latents, return_dict=True):
        from diffusers.models.autoencoders.autoencoder_same import AutoencoderSAMEDecoderOutput

        length = latents.shape[-1]
        if length <= chunk_size:
            return original_decode(latents, return_dict=return_dict)
        step = chunk_size - overlap
        starts = list(range(0, length - chunk_size + 1, step))
        # Pad the tail to an even start: original SAME-S attention groups 2 latents.
        final_start = ((length - chunk_size + 1) // 2) * 2
        if starts[-1] != final_start:
            starts.append(final_start)
        ratio = self.downsampling_ratio
        result = latents.new_zeros(latents.shape[0], self.config.audio_channels, length * ratio)
        half = overlap // 2
        for index, start in enumerate(starts):
            output = original_decode(latents[..., start:start + chunk_size]).sample
            left = 0 if index == 0 else half
            right = min(length - start, chunk_size) if index == len(starts) - 1 else chunk_size - half
            result[..., (start + left) * ratio:(start + right) * ratio] = output[..., left * ratio:right * ratio]
        return AutoencoderSAMEDecoderOutput(sample=result) if return_dict else (result,)

    vae.decode = MethodType(decode, vae)
    vae._sa3_chunked_decode = True
