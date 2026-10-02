"""Runtime optimizations must preserve checkpoint and logical-batch contracts."""

from types import SimpleNamespace

import pytest
import torch

from topobench.model.model import TBModel
from topobench.model.trawl_pretraining import historical_accumulation_weight

from .test_trawl import model, prepare


def test_layer_compilation_preserves_state_keys(monkeypatch):
    backbone = model([prepare()])
    wrapped = TBModel(
        backbone,
        SimpleNamespace(task_level="graph"),
        None,
        compile=True,
        compile_scope="layers",
    )
    before = set(wrapped.state_dict())
    calls = []

    def compile_layer(layer):
        calls.append(layer)
        layer._compiled_call_impl = lambda *args: None

    monkeypatch.setattr(torch.nn.Module, "compile", compile_layer)
    wrapped.configure_compilation()
    wrapped.configure_compilation()
    assert len(calls) == 2
    assert set(wrapped.state_dict()) == before
    assert all(not key.startswith("backbone._orig_mod") for key in before)


@pytest.mark.parametrize(
    "index,size,expected", [(0, 2, 1.0), (4, 2, 8 / 3), (5, 1, 4 / 3)]
)
def test_short_logical_batch_gradient_weight(index, size, expected):
    loader = SimpleNamespace(sampler=range(11), batch_size=2, drop_last=False)
    trainer = SimpleNamespace(
        accumulate_grad_batches=4,
        train_dataloader=loader,
        num_training_batches=6,
    )
    module = SimpleNamespace(_trainer=trainer)
    batch = SimpleNamespace(trawl_counts=torch.zeros(size, 2))
    assert historical_accumulation_weight(
        module, batch, index
    ) == pytest.approx(expected)
