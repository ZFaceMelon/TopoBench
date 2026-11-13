"""Cell-complex extensions: wide cells, linear pointers, vertex features."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from topobench.dataloader import DataloadDataset
from topobench.dataloader.utils import collate_fn
from topobench.nn.backbones.combinatorial.trawl import TRAWL
from topobench.transforms.data_manipulations.trawl import TRAWLTransform


def quadratic_vertex_ids(cell_vertices, paths, pivots, rng, cap):
    """The original [W, P, P] implementation, kept as a reference."""
    walks, length = paths.shape
    verts = cell_vertices[paths]
    width = verts.shape[2]
    order = np.argsort(rng.random(verts.shape), axis=2)
    verts = np.take_along_axis(verts, order, axis=2)
    flat = verts.reshape(walks, length * width)
    valid = flat >= 0
    same = (flat[:, :, None] == flat[:, None, :]) & valid[:, None, :]
    first = np.argmax(same, axis=2)
    position = np.arange(length * width)
    new = valid & (first == position)
    rank = np.cumsum(new, axis=1) - 1
    ids = np.take_along_axis(rank, first, axis=1)
    ids = np.where(valid, np.minimum(ids, cap - 1), -1)
    ids = ids.reshape(walks, length, width)
    crossed = np.zeros(ids.shape, dtype=bool)
    if pivots is not None:
        face = np.where(
            (pivots >= 0)[..., None], cell_vertices[np.maximum(pivots, 0)], -1
        )
        crossed = (verts[..., :, None] == face[..., None, :]).any(-1) & (
            verts >= 0
        )
    inverse = np.argsort(order, axis=2)
    ids = np.take_along_axis(ids, inverse, axis=2)
    crossed = np.take_along_axis(crossed, inverse, axis=2)
    step = position // width
    earlier = same & (step[None, None, :] < step[None, :, None])
    previous = np.where(earlier, step[None, None, :], -1).max(axis=2)
    previous = np.where(valid, previous, -1).reshape(walks, length, width)
    previous = np.take_along_axis(previous, inverse, axis=2)
    return ids, crossed, previous


def quadratic_previous_visit(paths):
    length = paths.shape[1]
    time = torch.arange(length)
    same = paths.unsqueeze(2) == paths.unsqueeze(1)
    earlier = same & (time.view(1, 1, -1) < time.view(1, -1, 1))
    return torch.where(earlier, time, -1).amax(dim=2)


def random_cells(rng, states, vertices, width):
    table = np.full((states, width), -1, dtype=np.int64)
    for row in range(states):
        size = int(rng.integers(1, width + 1))
        table[row, :size] = np.sort(
            rng.choice(vertices, size=min(size, vertices), replace=False)
        )
    return table


@pytest.mark.parametrize("width", [1, 3, 10])
@pytest.mark.parametrize("trial", range(4))
def test_vertex_scan_matches_quadratic_reference(width, trial):
    rng = np.random.default_rng([width, trial])
    states, vertices = 40, 25 if width > 3 else 8
    table = random_cells(rng, states, vertices, width)
    paths = rng.integers(0, states, size=(7, 16))
    pivots = np.where(
        rng.random(paths.shape) < 0.7, rng.integers(0, states, paths.shape), -1
    )
    pivots[:, 0] = -1
    for cap in (4, 32):
        for piv in (pivots, None):
            new = TRAWL.anonymous_vertex_ids(
                table, paths, piv, np.random.default_rng(trial), cap,
                pointers=True,
            )
            old = quadratic_vertex_ids(
                table, paths, piv, np.random.default_rng(trial), cap
            )
            for a, b in zip(new, old, strict=True):
                assert a.dtype == b.dtype
                np.testing.assert_array_equal(a, b)
            plain = TRAWL.anonymous_vertex_ids(
                table, paths, piv, np.random.default_rng(trial), cap
            )
            assert len(plain) == 2
            np.testing.assert_array_equal(plain[0], old[0])
            np.testing.assert_array_equal(plain[1], old[1])


def test_previous_visit_matches_quadratic_reference():
    generator = torch.Generator().manual_seed(0)
    for states in (1, 3, 50):
        paths = torch.randint(0, states, (9, 32), generator=generator)
        torch.testing.assert_close(
            TRAWL.previous_visit(paths), quadratic_previous_visit(paths)
        )


def fused_rings(pendant=True, fused=True):
    """A 5-ring and a 6-ring sharing edge (3, 4), plus a pendant vertex."""
    ring_a = [0, 1, 2, 3, 4]
    ring_b = [3, 4, 5, 6, 7, 8] if fused else [9, 10, 11, 12, 13, 14]
    edges = []
    for ring in (ring_a, ring_b):
        for i, u in enumerate(ring):
            pair = tuple(sorted((u, ring[(i + 1) % len(ring)])))
            if pair not in edges:
                edges.append(pair)
    nodes = max(max(e) for e in edges) + 1
    if pendant:
        edges.append((0, nodes))
        nodes += 1
    inc1 = torch.zeros(nodes, len(edges))
    for j, (u, v) in enumerate(edges):
        inc1[u, j] = inc1[v, j] = 1
    inc2 = torch.zeros(len(edges), 2)
    for c, ring in enumerate((ring_a, ring_b)):
        for i, u in enumerate(ring):
            pair = tuple(sorted((u, ring[(i + 1) % len(ring)])))
            inc2[edges.index(pair), c] = 1
    x0 = torch.randn(nodes, 4)
    return Data(
        x_0=x0,
        x_1=inc1.T @ x0,
        x_2=inc2.T @ (inc1.T @ x0),
        incidence_1=inc1.to_sparse(),
        incidence_2=inc2.to_sparse(),
        num_nodes=nodes,
        y=torch.tensor([1]),
        train_mask=torch.tensor([True]),
        val_mask=torch.tensor([True]),
        test_mask=torch.tensor([True]),
    )


def prepare(data, **kwargs):
    options = {
        "encodings": {"local": True, "RWSE": {"max_pe_dim": 2}},
        "neighborhoods": ["down_adjacency-2", "down_adjacency-1"],
        "pivots": True,
        "cell_vertices": True,
        "cell_vertex_width": 10,
    }
    options.update(kwargs)
    return TRAWLTransform(**options)(data)


def collate(graphs):
    dataset = DataloadDataset(graphs)
    return collate_fn([dataset[i] for i in range(len(dataset))])


def model(graphs, **kwargs):
    torch.manual_seed(0)
    options = {
        "hidden_dim": 16,
        "depth": 1,
        "architecture": "gru",
        "walks": {"k": 4, "length": 12},
        "walk_scope": "separate",
        "num_neighborhoods": 2,
        "vertex_ids": True,
        "back_pointers": True,
        "max_cell_vertices": 10,
        "fallback_family": 1,
    }
    options.update(kwargs)
    backbone = TRAWL(**options)
    backbone.initialize(graphs)
    return backbone.eval()


def test_cell_vertex_width():
    data = fused_rings()
    with pytest.raises(ValueError, match="cell_vertex_width"):
        prepare(data.clone(), cell_vertex_width=None)
    graph = prepare(data)
    table = graph.trawl_cell_vertices
    assert table.shape[1] == 10
    counts = graph.trawl_counts[0].tolist()
    rings = table[counts[0] + counts[1] :]
    assert sorted((rings >= 0).sum(1).tolist()) == [5, 6]
    # Default width is unchanged for simplices.
    assert TRAWLTransform(max_rank=2).cell_vertex_width == 3


def test_pointer_slots_follow_cell_size():
    default = TRAWL(
        hidden_dim=8,
        architecture="gru",
        depth=1,
        vertex_ids=True,
        back_pointers=True,
    )
    assert default.pointer_slot.num_embeddings == 4
    wide = TRAWL(
        hidden_dim=8,
        architecture="gru",
        depth=1,
        vertex_ids=True,
        back_pointers=True,
        max_cell_vertices=10,
    )
    assert wide.pointer_slot.num_embeddings == 11


def test_defaults_create_no_new_modules():
    torch.manual_seed(0)
    shapes = {"in_channels": [2, 2, 2], "pe_dim": 4}
    a = TRAWL(
        hidden_dim=8,
        architecture="gru",
        depth=1,
        vertex_ids=True,
        back_pointers=True,
        **shapes,
    )
    torch.manual_seed(0)
    b = TRAWL(
        **shapes,
        hidden_dim=8,
        architecture="gru",
        depth=1,
        vertex_ids=True,
        back_pointers=True,
        max_cell_vertices=None,
        vertex_features=False,
        fallback_family=None,
    )
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for key in sa:
        torch.testing.assert_close(sa[key], sb[key])


@pytest.mark.parametrize("vertex_features", [False, True])
def test_wide_cells_fallback_and_batch_isolation(vertex_features):
    graphs = [
        prepare(fused_rings()),
        prepare(fused_rings(fused=False)),  # rings share no edge
    ]
    net = model(graphs, vertex_features=vertex_features)
    batched = net(collate(graphs))
    separate = [net(collate([graph])) for graph in graphs]
    torch.testing.assert_close(
        batched["graph_embedding"],
        torch.cat([out["graph_embedding"] for out in separate]),
    )
    assert torch.isfinite(batched["graph_embedding"]).all()
    # The fused graph walks its 2-cells; the other falls back to edges.
    assert len(separate[1]["walk_embedding"]) > 0
    loss = batched["graph_embedding"].sum()
    loss.backward()
    grad = net.pointer_slot.weight.grad.abs().sum(dim=1)
    assert grad.shape == (11,)
    assert (grad[6:10] == 0).all()  # no cell has more than six vertices
    assert (grad[:5] > 0).all() and grad[10] > 0  # vertex and cell slots


def test_fallback_walks_edges_only_without_adjacent_cells():
    fused, apart = prepare(fused_rings()), prepare(fused_rings(fused=False))
    net = model([fused, apart])
    seen = {}
    for name, graph in (("fused", fused), ("apart", apart)):
        captured = []
        original = net._encode

        def encode(x, paths, ranks, *args, **kwargs):
            captured.append(ranks[paths].unique().tolist())
            return original(x, paths, ranks, *args, **kwargs)

        net._encode = encode
        net(collate([graph]))
        net._encode = original
        seen[name] = captured
    assert seen["fused"] == [[2]]
    assert seen["apart"] == [[1]]
    with pytest.raises(ValueError, match="fallback_family"):
        TRAWL(hidden_dim=8, architecture="gru", fallback_family=1)


def test_vertex_features_carry_vertex_inputs():
    base = fused_rings(pendant=False)
    changed = base.clone()
    changed.x_0 = base.x_0.clone()
    changed.x_0[3] += 5.0  # a vertex of the shared edge; x_2 is unchanged
    graphs = [prepare(base), prepare(changed)]
    for flag, differs in ((False, False), (True, True)):
        net = model(graphs, vertex_features=flag)
        a = net(collate([graphs[0]]))["graph_embedding"]
        b = net(collate([graphs[1]]))["graph_embedding"]
        assert (not torch.allclose(a, b)) == differs
    with pytest.raises(ValueError, match="vertex_features"):
        TRAWL(hidden_dim=8, architecture="gru", vertex_features=True)


def test_compact_wide_paths_match_dense():
    graphs = [prepare(fused_rings()), prepare(fused_rings(fused=False))]
    net = model(graphs, walks={"k": 6, "length": 16})
    generator = torch.Generator().manual_seed(0)
    ids = torch.randint(-1, 8, (5, 16, 10), generator=generator)
    crossed = (torch.rand(ids.shape, generator=generator) < 0.3) & (ids >= 0)
    compact = net._vertex_tokens(ids, crossed)
    tokens = torch.randn(5, 16, 16, generator=generator)
    paths = torch.randint(0, 4, (5, 16), generator=generator)
    pointers = torch.randint(-1, 16, (5, 16, 10), generator=generator)
    pointers = torch.minimum(pointers, torch.arange(16).view(1, -1, 1) - 1)
    compact_pointers = net._follow_pointers(tokens, paths, pointers)
    net.max_rank = 9  # forces the dense paths for width 10
    torch.testing.assert_close(compact, net._vertex_tokens(ids, crossed))
    torch.testing.assert_close(
        compact_pointers, net._follow_pointers(tokens, paths, pointers)
    )


def test_eval_chunks_match_unchunked():
    graphs = [prepare(fused_rings()), prepare(fused_rings(fused=False))]
    whole = model(graphs, eval_views=3)
    chunked = model(graphs, eval_views=3, eval_chunk_walks=5)
    chunked.load_state_dict(whole.state_dict())
    a = whole(collate(graphs))
    b = chunked(collate(graphs))
    torch.testing.assert_close(a["graph_embedding"], b["graph_embedding"])
    torch.testing.assert_close(a["walk_embedding"], b["walk_embedding"])


def test_joined_relation_is_union_and_single_names_unchanged():
    data = fused_rings()
    joined = prepare(
        data.clone(),
        neighborhoods=["up_incidence-0+up_incidence-1", "up_adjacency-0"],
    )
    parts = prepare(
        data.clone(),
        neighborhoods=["up_incidence-0", "up_incidence-1", "up_adjacency-0"],
    )
    pairs = lambda g, r: set(
        map(tuple, g.trawl_edges[g.trawl_edges[:, 2] == r, :2].tolist())
    )
    assert pairs(joined, 0) == pairs(parts, 0) | pairs(parts, 1)
    assert pairs(joined, 1) == pairs(parts, 2)
    assert joined.trawl_relations.item() == 2
    # Unjoined names give exactly the previous edge order.
    again = prepare(
        data.clone(),
        neighborhoods=["up_incidence-0", "up_incidence-1", "up_adjacency-0"],
    )
    assert torch.equal(again.trawl_edges, parts.trawl_edges)


@pytest.mark.parametrize("fusion", ["learned", "gate_mlp"])
def test_family_gates_report_weights(fusion):
    families = [
        "down_adjacency-2",
        "down_adjacency-1",
        "up_adjacency-0",
        "up_incidence-0+up_incidence-1",
    ]
    graphs = [
        prepare(fused_rings(), neighborhoods=families),
        prepare(fused_rings(fused=False), neighborhoods=families),
    ]
    net = model(
        graphs, num_neighborhoods=4, fallback_family=1, fusion=fusion
    )
    out = net(collate(graphs))
    weights = out["family_weights"]
    assert weights.shape == (2, 3)
    torch.testing.assert_close(weights.sum(1), torch.ones(2))
    if fusion == "learned":
        torch.testing.assert_close(weights, torch.full((2, 3), 1 / 3))
    out["graph_embedding"].sum().backward()
    gate = net.fusion.weights if fusion == "learned" else net.fusion.gate[0].weight
    assert gate.grad is not None and gate.grad.abs().sum() > 0


def test_mean_fusion_reports_no_weights_and_is_unchanged():
    graphs = [prepare(fused_rings()), prepare(fused_rings(fused=False))]
    net = model(graphs)
    out = net(collate(graphs))
    assert "family_weights" not in out
    stacked = torch.randn(3, 8)
    available = torch.tensor([True, False, True])
    fusion = net.fusion.__class__(8, 3, "mean")
    torch.testing.assert_close(
        fusion(stacked, available, stacked[0]),
        stacked[[0, 2]].mean(0),
    )


def test_rich_features_off_by_default_and_on_values():
    data = fused_rings()
    plain = prepare(data.clone())
    rich = prepare(data.clone(), rich_features=True)
    assert rich.trawl_pe.shape[1] == plain.trawl_pe.shape[1] + 40
    torch.testing.assert_close(rich.trawl_pe[:, : plain.trawl_pe.shape[1]], plain.trawl_pe)
    assert torch.isfinite(rich.trawl_pe).all()
    counts = rich.trawl_counts[0].tolist()
    table = rich.trawl_cell_vertices
    shared = next(
        i
        for i in range(counts[0], counts[0] + counts[1])
        if sorted(table[i][table[i] >= 0].tolist()) == [3, 4]
    )
    # ring-size flags (3, 4, 5, >=6) are columns 25..28 of the 40.
    flags = rich.trawl_pe[shared, -40 + 25 : -40 + 29]
    torch.testing.assert_close(flags, torch.tensor([0.0, 0.0, 1.0, 1.0]))
    pendant = rich.trawl_pe[counts[0] - 1, -40 + 25 : -40 + 29]
    torch.testing.assert_close(pendant, torch.zeros(4))
    with_attr = prepare(data.clone(), rich_features=True, attr_dim=1)
    assert with_attr.trawl_pe.shape[1] == plain.trawl_pe.shape[1] + 46


def test_feature_encodings_on_graphs_with_empty_relations():
    encodings = {
        "local": True,
        "RWSE": {"max_pe_dim": 2},
        "KHopFE": {"max_hop": 3},
        "HKFE": {"kernel_param_HKFE": [1, 21]},
        "PPRFE": {"alpha_param_PPRFE": [0.1, 10]},
    }
    graphs = [
        prepare(fused_rings(), encodings=encodings),
        prepare(fused_rings(fused=False), encodings=encodings),
    ]
    assert graphs[0].trawl_pe.shape[1] == graphs[1].trawl_pe.shape[1]
    for graph in graphs:
        assert torch.isfinite(graph.trawl_pe).all()
    net = model(graphs)
    assert torch.isfinite(net(collate(graphs))["graph_embedding"]).all()


def test_no_decay_param_groups_default_unchanged():
    from topobench.optimizer import TBOptimizer

    params = {"lr": 0.001, "weight_decay": 0.01}
    assert TBOptimizer("AdamW", params).no_decay is None
    opt = TBOptimizer("AdamW", params, no_decay=["fusion\.weights$"])
    assert opt.no_decay == ["fusion\.weights$"]
    graphs = [prepare(fused_rings())]
    net = model(graphs, fusion="learned")
    names = [n for n, _ in net.named_parameters()]
    import re

    skip = [bool(re.search("fusion\.weights$", n)) for n in names]
    assert sum(skip) == 1


def test_attr_scaling_signed_log1p():
    data = fused_rings()
    data.x_0[:, 0] = torch.linspace(-500, 800, len(data.x_0))
    inc1 = data.incidence_1.to_dense()
    inc2 = data.incidence_2.to_dense()
    data.x_1 = inc1.T @ data.x_0
    data.x_2 = inc2.T @ data.x_1
    plain = prepare(data.clone(), attr_dim=1)
    scaled = prepare(data.clone(), attr_dim=1, attr_scaling="signed_log1p")
    a0 = torch.sign(data.x_0[:, 0]) * torch.log1p(data.x_0[:, 0].abs())
    torch.testing.assert_close(scaled.trawl_signal_0[:, 0], a0)
    torch.testing.assert_close(scaled.trawl_signal_1[:, 0], inc1.T @ a0)
    torch.testing.assert_close(scaled.trawl_signal_2[:, 0], inc2.T @ (inc1.T @ a0))
    torch.testing.assert_close(scaled.trawl_signal_0[:, 1:], plain.trawl_signal_0[:, 1:])
    torch.testing.assert_close(plain.trawl_signal_0[:, 0], data.x_0[:, 0])


def test_gate_entropy_and_family_dropout():
    families = [
        "down_adjacency-2",
        "down_adjacency-1",
        "up_adjacency-0",
        "up_incidence-0+up_incidence-1",
    ]
    graphs = [
        prepare(fused_rings(), neighborhoods=families),
        prepare(fused_rings(fused=False), neighborhoods=families),
    ]
    base = model(graphs, num_neighborhoods=4, fallback_family=1, fusion="gate_mlp")
    assert "aux_loss" not in base.train()(collate(graphs))
    net = model(
        graphs,
        num_neighborhoods=4,
        fallback_family=1,
        fusion="gate_mlp",
        gate_entropy=0.01,
        family_dropout=0.5,
    )
    net.train()
    torch.manual_seed(0)
    out = net(collate(graphs))
    weights = out["family_weights"]
    entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum(1).mean()
    torch.testing.assert_close(out["aux_loss"], -0.01 * entropy)
    assert ((weights > 0).sum(1) >= 1).all()
    out["aux_loss"].backward()
    assert net.fusion.gate[0].weight.grad is not None
    net.eval()
    assert "aux_loss" not in net(collate(graphs))
    with pytest.raises(ValueError, match="gate"):
        TRAWL(hidden_dim=8, architecture="gru", gate_entropy=0.1)


@pytest.mark.parametrize("training", [True, False])
def test_sampling_workers_give_identical_outputs(training):
    graphs = [prepare(fused_rings()), prepare(fused_rings(fused=False))] * 3
    serial = model(graphs, eval_views=2, vertex_features=True)
    parallel = model(graphs, eval_views=2, vertex_features=True, sampling_workers=2)
    parallel.load_state_dict(serial.state_dict())
    serial.train(training)
    parallel.train(training)
    torch.manual_seed(0)
    a = serial(collate(graphs))
    torch.manual_seed(0)
    b = parallel(collate(graphs))
    for key in ("graph_embedding", "walk_embedding", "walk_batch", "walk_view"):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
