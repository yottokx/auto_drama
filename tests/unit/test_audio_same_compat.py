"""Numerical checks use PyTorch only when an audio runtime is available."""

from types import SimpleNamespace

import pytest

from scripts.audio.same_compat import (
    _chunked_block_forward,
    enable_chunked_decode,
    install_same_compat,
)


def torch_runtime():
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers.models.autoencoders.autoencoder_same")
    return torch


@pytest.mark.parametrize("mode,length", [("encoder", 7), ("encoder", 12), ("decoder", 5), ("decoder", 8)])
def test_small_shifted_chunks_match_reference_layout(mode, length):
    torch = torch_runtime()

    class MixTokens:
        def __init__(self, factor):
            self.factor = factor

        def __call__(self, value):
            return value + value.mean(dim=1, keepdim=True) * self.factor

    stride = 2
    layers = [MixTokens(0.1), MixTokens(0.2), MixTokens(0.3), MixTokens(0.4)]
    block = SimpleNamespace(
        mode=mode, stride=stride, sa3_chunk_size=4, sa3_chunk_midpoint_shift=True,
        sa3_mask_noise=0, mapping=torch.nn.Identity(), transformers=layers,
        new_tokens=torch.full((1, 1, 3), 0.1),
    )
    value = torch.arange(2 * 3 * length, dtype=torch.float32).reshape(2, 3, length) / 10
    actual = _chunked_block_forward(block, value)
    # Independently enumerate reference subsegments, folding pairs for the first
    # half and shifted neighbouring pairs for the second half. This catches the
    # tempting but incorrect native sliding-window approximation for SAME-S.
    padded = torch.nn.functional.pad(value, (0, (-length) % (4 if mode == "encoder" else 2)))
    segment_length = stride if mode == "encoder" else 1
    output_length = 1 if mode == "encoder" else stride
    rows = []
    for row in padded:
        items = []
        for start in range(0, padded.shape[-1], segment_length):
            items.append(torch.cat((row[:, start:start + segment_length].T,
                                    block.new_tokens[0].repeat(output_length, 1)), dim=0))
        segments = torch.stack(items)
        first = []
        for pair in segments.split(2):
            pair = pair.reshape(1, -1, 3)
            for layer in layers[:2]:
                pair = layer(pair)
            first.append(pair.reshape(2, stride + 1, 3))
        segments = torch.cat(first)
        shifted = torch.cat((segments[:1], segments, segments[-1:]))
        second = []
        for pair in shifted.split(2):
            pair = pair.reshape(1, -1, 3)
            for layer in layers[2:]:
                pair = layer(pair)
            second.append(pair.reshape(2, stride + 1, 3))
        result = torch.cat(second)[1:-1, -output_length:]
        rows.append(result.reshape(-1, 3).T)
    torch.testing.assert_close(actual, torch.stack(rows))


def test_small_kernel_and_chunk_architecture_survive_save_and_reload(tmp_path):
    torch = torch_runtime()
    from diffusers import AutoencoderSAME

    install_same_compat()
    model = AutoencoderSAME(
        patch_size=2, encoder_channels=4, encoder_c_mults=(2,), encoder_strides=(2,),
        encoder_transformer_depths=(4,), latent_dim=4, dim_heads=4,
        encoder_chunk_size=4, decoder_chunk_size=4,
        encoder_chunk_midpoint_shift=True, decoder_chunk_midpoint_shift=True,
        decoder_mapping_kernel=3,
    ).eval()
    assert model.decoder.blocks[0].mapping.weight_v.shape[-1] == 3
    # Strict loading is essential: the unextended native kernel has width 1 and
    # cannot accept the official Small decoder checkpoint's width-3 tensor.
    model.load_state_dict(model.state_dict(), strict=True)
    model.save_pretrained(tmp_path)
    loaded = AutoencoderSAME.from_pretrained(tmp_path).eval()
    assert loaded.config.decoder_chunk_size == 4
    assert loaded.config.decoder_mapping_kernel == 3
    latents = torch.randn(1, 4, 5)
    with torch.inference_mode():
        torch.testing.assert_close(model.decode(latents).sample, loaded.decode(latents).sample)
        source = torch.randn(1, 2, 17)
        torch.testing.assert_close(model.encode(source).latents, loaded.encode(source).latents)


@pytest.mark.parametrize("length", [129, 130, 191, 192, 193, 255, 256, 321])
def test_chunked_decode_preserves_every_sample_including_odd_tail(length):
    torch = torch_runtime()
    from diffusers.models.autoencoders.autoencoder_same import AutoencoderSAMEDecoderOutput

    class Decoder:
        downsampling_ratio = 3
        config = SimpleNamespace(audio_channels=2)

        def __init__(self):
            self.lengths = []

        def decode(self, latents, return_dict=True):
            self.lengths.append(latents.shape[-1])
            value = latents.repeat_interleave(self.downsampling_ratio, dim=-1)
            return AutoencoderSAMEDecoderOutput(sample=value) if return_dict else (value,)

    decoder = Decoder()
    latents = torch.arange(length, dtype=torch.float32).reshape(1, 1, length).repeat(1, 2, 1)
    expected = decoder.decode(latents).sample
    decoder.lengths = []
    enable_chunked_decode(decoder)
    actual = decoder.decode(latents).sample
    torch.testing.assert_close(actual, expected)
    assert max(decoder.lengths) <= 128
    torch.testing.assert_close(decoder.decode(latents, return_dict=False)[0], expected)
