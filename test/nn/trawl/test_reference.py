"""Numerical parity against frozen original-script outputs, not self-comparison."""

from pathlib import Path

import numpy as np
import torch

from topobench.nn.backbones.combinatorial.trawl import SISABlock
from topobench.nn.backbones.combinatorial.trawl import (
    ContinuousTRAWL,
)


def test_frozen_continuous_encoder():
    with np.load(
        Path(__file__).parent / "fixtures/reference.npz", allow_pickle=False
    ) as fixture:
        net = ContinuousTRAWL(
            hidden_dim=16,
            pe_dim=6,
            embed_dim=4,
            depth=2,
            architecture="mamba",
            dropout=0.0,
            layer_options={"mamba": {"d_state": 4}},
        ).eval()
        state = {
            key.removeprefix("continuous::"): torch.from_numpy(fixture[key])
            for key in fixture.files
            if key.startswith("continuous::")
        }
        input_prefixes = {
            "node_norm",
            "pe_norm",
            "node_proj",
            "edge_proj",
            "pe_proj",
            "input_norm",
            "in_proj",
        }
        mapped = {}
        key_mapping = {}
        for key, value in state.items():
            old_keys = set(mapped)
            if key.split(".")[0] in input_prefixes:
                mapped["continuous_input." + key] = value
            elif key.startswith("layers."):
                mapped[
                    key.replace("layers.", "encoders.0.", 1).replace(
                        ".block.", ".module."
                    )
                ] = value
            elif key.startswith("norm."):
                mapped[key.replace("norm.", "output_norm.", 1)] = value
            elif key.startswith("recon_decoder."):
                mapped[
                    key.replace("recon_decoder.", "reconstruction_decoder.", 1)
                ] = value
            new_keys = set(mapped) - old_keys
            if new_keys:
                key_mapping[key] = new_keys.pop()
        missing, unexpected = net.load_state_dict(mapped, strict=False)
        assert not unexpected
        assert missing == ["sampling_step"]
        signals = torch.from_numpy(fixture["signals"]).flatten(0, 1)
        pe = torch.from_numpy(fixture["pe"]).flatten(0, 1)
        _, actual = net.encode_walks(signals, pe)
        torch.testing.assert_close(
            actual, torch.from_numpy(fixture["pooled"]), atol=2e-6, rtol=2e-6
        )
        actual.square().mean().backward()
        parameters = dict(net.named_parameters())
        for key in fixture.files:
            if key.startswith("continuous_grad::"):
                parameter = parameters[key_mapping[key.split("::", 1)[1]]]
                torch.testing.assert_close(
                    parameter.grad,
                    torch.from_numpy(fixture[key]),
                    atol=2e-6,
                    rtol=2e-5,
                )
        torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-3).step()
        for key in fixture.files:
            if key.startswith("continuous_step::"):
                parameter = parameters[key_mapping[key.split("::", 1)[1]]]
                torch.testing.assert_close(
                    parameter,
                    torch.from_numpy(fixture[key]),
                    atol=2e-6,
                    rtol=2e-5,
                )


def test_frozen_sisa_layer():
    with np.load(
        Path(__file__).parent / "fixtures/reference.npz", allow_pickle=False
    ) as fixture:
        net = SISABlock(16, n_heads=2, d_ssm=4).eval()
        net.load_state_dict(
            {
                key.removeprefix("sisa::"): torch.from_numpy(fixture[key])
                for key in fixture.files
                if key.startswith("sisa::")
            }
        )
        actual = net(torch.from_numpy(fixture["sisa_input"]))
        torch.testing.assert_close(
            actual,
            torch.from_numpy(fixture["sisa_output"]),
            atol=2e-6,
            rtol=2e-6,
        )
        actual.square().mean().backward()
        parameters = dict(net.named_parameters())
        for key in fixture.files:
            if key.startswith("sisa_grad::"):
                torch.testing.assert_close(
                    parameters[key.split("::", 1)[1]].grad,
                    torch.from_numpy(fixture[key]),
                    atol=2e-6,
                    rtol=2e-5,
                )
        torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-3).step()
        for key in fixture.files:
            if key.startswith("sisa_step::"):
                torch.testing.assert_close(
                    parameters[key.split("::", 1)[1]],
                    torch.from_numpy(fixture[key]),
                    atol=2e-6,
                    rtol=2e-5,
                )
