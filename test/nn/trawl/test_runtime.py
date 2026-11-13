"""Runtime optimizations must preserve checkpoint contracts."""

from types import SimpleNamespace

import torch

from topobench.model.model import TBModel

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

    def fake_compile(fn, **kwargs):
        calls.append(fn)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    wrapped.configure_compilation()
    wrapped.configure_compilation()
    # TRAWL compiles each layer call twice (checkpointed and plain), once.
    layers = sum(len(stack) for stack in backbone.encoders)
    assert len(calls) == 2 * layers
    assert set(wrapped.state_dict()) == before
    assert all(not key.startswith("backbone._orig_mod") for key in before)
