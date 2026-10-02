# TRAWL in TopoBench

TRAWL lifts input data to cells, computes structural encodings, samples
topological walks, and processes their sequences with configurable neural
layers. The integration uses TopoBench's loaders, transform cache, collator,
`TBModel`, losses, evaluators, optimizers, Lightning trainer, callbacks and Hydra
sweeps. No backbone selects behavior by dataset name.

## Repository layout

TRAWL follows the existing component directories rather than defining a separate
neural-network framework namespace:

- `nn/backbones/general/trawl.py`: the base backbone. The sibling
  `trawl_continuous.py`, `trawl_categorical.py` and `trawl_blocks.py` contain its
  input variants and neural blocks. The `general` category is for architectures
  that are independent of the input topology domain. This placement is proposed
  for discussion with the project maintainers; it does not classify TRAWL as a
  combinatorial-complex-only model.
- `nn/encoders/trawl.py` and `nn/readouts/trawl.py`: feature adapter and task head.
- `transforms/data_manipulations/trawl.py` and `trawl_historical.py`: native
  preprocessing and the opt-in historical preprocessing profile. The latter
  constructs coupled legacy lifting/features/walk fields, rather than claiming
  to implement the general `Graph2CellLifting` interface.
- `data/utils/trawl/`: CPU topology mathematics, historical feature definitions
  and walk sampling shared by preprocessing and the backbones; no neural layers.
- `loss/model/trawl.py`, `model/trawl_pretraining.py`, `optimizer/schedulers.py`,
  `evaluator/checkpoint.py`, and `utils/trawl_provenance.py`: auxiliary objective,
  optional training stage, schedule, checkpoint evaluation and run provenance.

The base/variant model configs live under `configs/model/general`; the
graph-only specialization lives under `configs/model/graph`. Experiments remain
under `configs/experiment/trawl`. The earlier draft's `model=cell/trawl` and
`model=combinatorial/trawl` commands are now `model=general/trawl`. There is no
`topobench.nn.trawl` package.
Integration tests remain grouped in `test/nn/trawl` because they exercise the
complete model pipeline across these component boundaries.

TRAWL is the complete TopoBench backbone. Mamba, SISA and the other sequence
layers are configurable components inside it. General placement does not make
every existing graph or cell backbone a drop-in sequence layer: custom modules
must satisfy the sequence interface or use an appropriate adapter.

## Start here

Use the repository's normal installation procedure. Pure PyTorch Mamba is the
default and works on CPU; the optional `mamba_ssm` backend must be installed
separately on a supported system. Backend selection is explicit and never falls
back silently. No imports from the parent TRAWL checkout are needed at runtime.

The optional backend was validated on Linux/CUDA with `mamba-ssm==2.2.6.post3`
and `transformers==4.44.2`; Transformers 5 is incompatible with that Mamba
release's generation imports. Set
`model.backbone.layer_options.mamba.backend=mamba_ssm` to select it. The `scan`
option controls only the pure-PyTorch backend and is ignored by official Mamba.

```bash
python -m topobench model=general/trawl dataset=graph/PROTEINS logger=csv
python -m topobench model=general/trawl dataset=graph/NCI1 model.backbone.architecture=sisa logger=csv
python -m topobench model=graph/trawl dataset=graph/cocitation_cora logger=csv
```

`general/trawl` defaults to a cycle lifting followed by TRAWL preprocessing for
graph inputs. Already-lifted/native topological datasets use `trawl_existing`.
`graph/trawl` uses the original graph edges without introducing higher cells.
For another lifting, create a transform composition with that lifting first and
`/transforms/data_manipulations@trawl: trawl` last. Existing `x_r` and
`incidence_r` are consumed without assuming simplices or a particular dataset.
Set `model.backbone.max_rank` to the largest rank you want represented.

## Graphs, neighborhoods and walks

The base configuration uses **separate neighborhood encodings and union
walks**. The default relation set is immediate vertex–edge and edge–face
incidence, bidirectional. Thus the default connectivity is a Hasse graph even
though the configurable graph mode is named `augmented_hasse`.

The graph modes are:

- `hasse`: immediate incidence only. Vertices and higher cells are states.
  A step changes rank. Multi-hop shortcuts and adjacency are rejected.
- `augmented_hasse`: selected incidence and/or same-rank adjacency relations.
  More direct routes can improve reachability at the cost of changing the walk
  distribution and increasing connectivity.
- `cell_overlap`: cells in `overlap_ranks` connect when they share an original
  vertex. This is the historical family often called “dual.” It omits explicit
  incidence traversal and can become dense around high-degree vertices.

`cell_overlap` in the general transform derives membership from incidence.
The historical compatibility transform separately preserves weighted overlap,
bond/ring ordering and its original neighbor order. The two are deliberate,
documented constructions, not numerical aliases.

Neighborhood names follow TopoBench conventions: `up_incidence-0`,
`down_incidence-2`, `up_adjacency-1`, `2-up_adjacency-0`, and so on. Up adjacency
means sharing upper cells; down adjacency means sharing lower cells. Multi-hop
incidences compose consecutive incidence matrices. Signed incidence is converted
to unsigned membership for walk probabilities. Explicit lifting-provided
relations take precedence over derived relations. Hypergraph
`incidence_hyperedges`/`x_hyperedges` are adapted to rank 1.

HOPSE-style adjacency and mixed relation sets are supplied:

```bash
python -m topobench experiment=trawl/adjacency dataset=graph/PROTEINS logger=csv
python -m topobench experiment=trawl/mixed dataset=graph/PROTEINS logger=csv
```

These presets use the relation-set idea; they do not claim to reproduce another
model's implementation or its results. The base two-incidence set is the
incidence preset. Add reverse-direction names only when they represent distinct
experiments: bidirectional expansion already provides both traversal directions.

Walk settings live under `model.backbone.walks`: `k`, `length`, `start_policy`,
`epsilon`, `reverse`, and `guidance` (`gamma`, `diffusion_t`, `dense_limit`).
Default K/L are 32/32, heat guidance gamma is 0.2, and starts use the historical
coverage policy: first uniform, then inverse-visit-count preference while
coverage changes, with a uniform fallback at a coverage plateau. This does
**not** stop sampling early. Nonbacktracking excludes the previous state unless
that leaves no usable transitions, when ordinary weighted transitions are used.

- `walk_scope=union`: K sampled walks per input graph over the relation union.
- `walk_scope=separate`: K walks per available relation. Empty relations receive
  none; their budget is not redistributed.
- `reverse=true`: K original paths plus their reversals, hence 2K encoded
  sequences. Rank-change embeddings are recomputed for reversed paths.
- `walk_refresh=train`: training calls resample; evaluation is deterministic.
  `fixed` holds the sampling seed fixed for controlled cached-walk ablations.

For equal-total-walk comparisons with two relations, compare union K=32 with
separate K=16; equal K=32 per relation is a different compute budget. Multiply
walks × length × reverse factor when comparing token budgets. Missing relations
reduce actual counts under both comparisons.

## Encodings and empty structures

`transforms.trawl.encodings` selects local features, empirical nonbacktracking
RWSE (`rw_steps`, `rw_samples`), heat diagonals (`heat_times`), electrostatic
potentials (`electrostatic_betas`), and Laplacian eigenvectors (`laplacian_dim`).
`encoding_scope=separate` concatenates relation-specific encodings;
`encoding_scope=union` computes one encoding on the union. Encoding scope does
not change walk scope.

Implicit rank and color embeddings are separate from PSEs. To use categorical
colors, provide one nonnegative color ID per state through `color_key`, and set
the backbone's `num_colors` to cover the vocabulary. Continuous features are
never converted into colors automatically. `color_refinement=true` computes
fixed PSE slots for unordered color-pair subgraphs, using transform `num_colors`
(default: number of ranks). It leaves walk connectivity unchanged and can be
combined with implicit color embeddings.

Absent ranks stay empty. Empty relation PSEs are zero. Separate-neighborhood
mean, learned-weight and attention fusion exclude unavailable relations;
concatenation retains a zero slot and availability indicator. If every selected
relation is empty, base TRAWL pools projected input features instead of
fabricating walks. Isolated/unvisited cells use projected input features as
their contextual fallback. Fully empty inputs with zero vertices are rejected.

Exact dense spectral encoding and guidance are bounded by `dense_limit=2048`.
Exceeding the limit raises an actionable error; it does not silently substitute
an approximation. Disable guidance with `model.backbone.walks.guidance=null`,
disable spectral channels, or explicitly raise the limit. A bounded CPU cache
of transition distributions (`transition_cache_size=1024`) avoids repeated
eigendecomposition; sampling still resamples. Cache keys include connectivity
and weights, including topology masks. Spectral eigenbasis choices can differ
between numerical-library versions, especially at repeated eigenvalues.

## Layers, outputs and native components

`architecture` selects `mamba`, `sisa`, `hybrid`, `transformer`, `gru`, or `mlp`.
The five-layer hybrid is M–S–M–S–M. `depth`, `hidden_dim`, `dropout`, and
`layer_options` control shared settings. SISA requires even rotary dimensions
and a hidden width divisible by its head count. Mamba exposes state size,
convolution width, expansion, backend, and parallel/sequential scan.

For arbitrary ordering, use `experiment=trawl/custom_layers` or replace
`model.backbone.layers` with a list of per-layer dictionaries. A layer can also
be a Hydra `_target_` implementing `[walk, time, hidden] -> same shape`.
`kind=graph` adapts a PyG-compatible module accepting `(x, edge_index)` to
disjoint walk path graphs. Graph/cell backbones with additional required inputs
must receive a purpose-built adapter; they are not interchangeable with a
sequence module merely because they exist in TopoBench.

Separate walks support `encoder_sharing=shared` (default) or `independent` and
`fusion=mean|concat|learned|attention`. Pooling over time is `mean`, `max`, or
`mean_max` (max followed by mean). Contextual occurrence aggregation is
`occurrence_pooling=mean|attention`. `graph_readout=cells` pools contextual cells
across ranks; `walks` uses pooled walk embeddings. All rank outputs `x_r` and
`batch_r` are available for native TopoBench readouts.

The included readout supports graph and node tasks. `aggregation=embedding`
applies the head after graph fusion; `walk_logits` applies it per walk and then
averages logits. These operations are not equivalent for nonlinear heads.
`deepset` supplies trainable sum-based walk aggregation for the categorical
profile, applying the head within each evaluation view before averaging logits.
Separate-neighborhood fusion and cell readout require `aggregation=embedding`;
incompatible walk-level heads raise an error rather than discard the fusion.
Classification, one-logit BCE, multilabel classification with missing
labels, and scalar/vector regression use the ordinary loss/evaluator pipeline.

Input widths are inferred from training data before constructing the optimizer.
For a custom native feature encoder, wrap it in `TRAWLFeatureEncoder` and set
`backbone.in_channels` explicitly to its output widths. Raw rank feature widths
must be consistent across populated graphs. Empty ranks are normalized to the
corresponding empty tensor shape before caching/batching.
Base pretraining also trains the adapted feature encoder and reconstructs raw
rank features. The historical input profiles use their own encoders and reject
these alternative input/readout combinations. Custom structural encoders used
with topology reconstruction must not reintroduce held-out connectivity.

## Optional pretraining

```bash
python -m topobench model=general/trawl dataset=graph/PROTEINS pretraining.enabled=true logger=csv
```

The base objectives are masked cell-feature reconstruction, optional color
classification, and optional masked-connectivity prediction. Configure their
weights under `pretraining.objectives`. Only training examples update weights;
validation chooses the checkpoint. Labels are not used. Topology masking removes
both orientations and parallel copies of held-out edges and suppresses PSE
shortcuts. Complete graphs may supply no negative links; that objective then
contributes zero for that graph.

Pretraining uses its own AdamW optimizer and reconstruction scheduler, restores
the best validation encoder, and then starts normal supervised fitting with a
fresh optimizer. `reset_head` optionally applies Xavier initialization. Resuming
a supervised `ckpt_path` skips pretraining. Pretraining saves its best and last
checkpoints and CSV metrics under the run's `pretraining/` directory. Resume an
interrupted stage with `pretraining.ckpt_path=/path/to/pretraining/last.ckpt`;
`pretraining.max_epochs` is the total epoch budget, including completed epochs.
Resuming into a new run directory copies the earlier best checkpoint (from its
recorded path or beside `last.ckpt`) and keeps its score, so a worse epoch after
the resume cannot replace it. `pretraining.min_delta` sets the minimum
validation improvement for both the best checkpoint and early stopping; the
historical presets use 1e-5. The reconstruction scheduler steps only on
validated epochs, and pretraining validation scores one walk view.

For supervised evaluation without fitting, set `train=false` and
`ckpt_path=/path/to/saved.ckpt`. The `best` evaluation strategy loads that file.
For `weight_average` or `logit_ensemble`, point `ckpt_path` to the best checkpoint
named in `checkpoint_index.json` and keep the index and all top-K checkpoint
files together. The runner resolves their filenames relative to the supplied
checkpoint directory, allowing a saved run to move between machines. Retain
the original recipe, seed, split and preprocessing configuration for replay.

The continuous/categorical historical profiles instead use the original
walk-summary reconstruction target (mean state signals, mean PSE, mean absolute
step differences). The continuous profile masks inputs before BatchNorm and
constructs synthetic steps after normalization. This objective is distinct from
the base cell/topology/color objective.

`experiment=trawl/joint` enables optional reconstruction during supervised
fine-tuning through the native module-loss interface. Its contribution starts
at 0.05 and decays to zero over 20 epochs. The default uses smooth-L1 on the
same pooled features as classification, matching the inspected later joint
variant; optional input masking and MSE are explicit loss settings. This
option is independent of the pretraining flag and is off in the base model.

## Historical recipe profiles and evidence

Run configurations, not training-script copies, select historical behavior:

```bash
python -m topobench experiment=trawl/proteins_mamba logger=csv
python -m topobench experiment=trawl/proteins_hybrid logger=csv
python -m topobench experiment=trawl/nci1_hybrid logger=csv
python -m topobench experiment=trawl/nci1_sisa logger=csv
python -m topobench experiment=trawl/nci1_mamba logger=csv
python -m topobench experiment=trawl/proteins_gated logger=csv
```

The classic profile ports the frozen PROTEINS 74.70 snapshot: bond plus 5/6-cycle
basis lifting, weighted overlap, original four state channels and 32 PSE
channels, minibatch-wide BatchNorm, synthetic difference/average transitions,
configurable Mamba/SISA layers, max+mean walk pooling, and one-logit BCE. Rich
structural features add 40 channels (46 when continuous attributes are enabled).
The extended profile uses the later 11-token/18-constituent vocabulary and
cycle18/Adj-3, short-cycle, star, merged, or gated lifting variants. The gated
variant keeps branches disconnected, splits total K between them, computes
branch-local guidance/PSEs and gates branch logits with initial prior 0.7.
`proteins_gated` loads PROTEINS' native continuous node attribute
(`use_node_attr=true`, `attr_dim=1`, 78 PSE channels) and uses top-5 weight
averaging, as its historical launcher did. Branch gating requires the historical
sampling protocol, which allocates walks per branch.

The categorical profile (`experiment=trawl/categorical`) preserves token and
constituent embeddings, sum-for-bonds/mean-for-larger-cells pooling, PSE fusion,
and DeepSet readout. `experiment=trawl/zinc_categorical` selects native ZINC
splits, MAE, Adam and its linear-warmup/cosine schedule, with the historical
script's K=8 walks, patience 40 and three final test views. The reusable
`optimizer=trawl_cosine` exposes warmup duration, start factor and minimum LR.
This is a historical-style profile, not a certified recreation of the ZINC run.
The categorical encoder regularizes with `layer_dropout` and readout dropout,
as the historical model did; `backbone.dropout` only affects the optional
reconstruction decoder.

**These are runnable compatibility profiles, not certified score reproductions.**
The recorded PROTEINS hybrid 75.196% and NCI1 rich hybrid 76.752% results have
incomplete exact code-to-run provenance. A similarly named later snapshot does
not resolve that gap. Classic versus extended token conventions must be chosen
from the actual run artifact, not inferred from the dataset name. The supplied
headline hybrid profiles currently select the classic conventions explicitly.
See `trawl_recipes.json` for evidence and remaining acceptance work.

The strongest available numerical check compares the continuous encoder and
SISA layer to fixed outputs generated from the original snapshots, with mapped
weights. It does not establish full-run equality. In particular:

- Pretraining and supervised module construction consume initialization RNG in
  a different order from the monolithic script. A matching seed alone does not
  imply matching initial tensors; use mapped/imported states for parity work.
- The original code started ARPACK from its internal random state, so its
  eigenvector features were not repeatable even within one environment. The
  port passes a fixed starting vector, making spectral features a pure function
  of each graph; they therefore cannot equal any particular historical draw.
  The classic tiny-graph fallback fixes an original solver limitation; that is
  a documented behavior change.
- Microbatch size affects BatchNorm and seeds. Historical walk seeds and
  epoch-indexed permutations are exposed, but multi-GPU/microbatch launch
  equivalence and old pretraining RNG trajectories are not certified.
- The repository defaults run in ordinary precision. For a matching supported
  GPU, explicitly select the historical AMP precision (BF16 or FP16 as used by
  that run). Historical runs evaluated in full precision; the continuous
  presets set `model.evaluation_autocast=false`, so validation and test (and
  pretraining validation) disable autocast under mixed-precision training.
  CPU/GPU agreement for the pure-PyTorch implementation and FP16
  updates with both backends are tested. Official Mamba is an alternative
  implementation, not a certified numerically identical replacement.
- The historical sampling protocol seeds evaluation walks by each graph's
  position in its evaluation batch, as the original code did. Its validation and
  test metrics are exactly repeatable only with the same evaluation batch size
  and order. The stable protocol hashes seed components and has no such
  dependence.
- No full 300-epoch, five-seed benchmark is claimed by the unit/smoke tests.

## Reproducibility

Walk sampling, splits, epoch order, pretraining masks and initialization are
seeded, and spectral features use a fixed ARPACK start. `deterministic=strict`
(set by the continuous presets) additionally enables PyTorch deterministic
kernels, raising on any operation without one, and passes the setting to every
Lightning trainer; the stock trainer configuration would otherwise switch it
off. On CUDA it needs `CUBLAS_WORKSPACE_CONFIG=:4096:8` before cuBLAS is first
used in the process; the runner sets it, and launchers should export it. SISA's
cumulative sums use a deterministic triangular product under this mode. With
these settings, two PROTEINS hybrid runs with the same seed on separate RTX
A4000 GPUs, compiled and in BF16, produced bit-identical pretraining and final
weights, metric histories and evaluation reports. The measured cost on that job
was about 9% wall time. Exact repeats assume the same software stack, GPU model,
device count and evaluation batch size; `deterministic=true` only warns about
nondeterministic operations.

## Performance

All of the following leave results bit-identical; each was verified against
the reproducible v10 run (identical checkpoints, metric history and
evaluation report).

- **Walk sampling.** Each step reproduces `Generator.choice` exactly (one
  `random()` draw against the normalized cumulative weights, with NumPy's
  summation order) without its per-call validation, and neighbor rows are
  prepared once per graph. With the optional `numba` dependency
  (`pip install topobench[trawl]`) the same arithmetic runs compiled.
- **No host-device synchronization.** `TBModel` stashes CPU copies of the
  backbone's `host_fields` before each transfer and copies batches with
  `non_blocking`; walk bookkeeping stays on the host, and losses are logged
  as tensors. The historical presets pin batch memory and disable
  torchmetrics input checks (`evaluator.validate_args=false`).
- **Caches.** Per-graph sampler inputs are cached by content hash, and
  evaluation walks (whose seeds do not depend on the epoch) are sampled once.
  Split-seeded historical preprocessing is stored under the processed-data
  directory, keyed by settings, inputs, code and library versions.

On the PROTEINS hybrid probe (two pretraining and six training epochs, RTX
A4000, compiled BF16) wall time fell from 648 s to 167 s.

## Splits, evaluation and sweeps

Historical TU profiles use `split_type=stratified`, performing the original two
stratified 50/25/25 split calls. `split_type=imported` plus `split_file` accepts
an NPZ with `train`, `valid`, `test` integer arrays that partition the input
exactly once. Historical RWSE seeds are assigned after splitting, using
split-local indices. Base PSEs remain in TopoBench's normal transform cache.

`evaluation.checkpoint` selects `best`, `weight_average`, or `logit_ensemble`.
The last two use `evaluation.top_k` validation-ranked checkpoints and require
the callback to save enough checkpoints. Floating parameters and buffers are
averaged; integer buffers come from the best checkpoint. When several
checkpoints tie for the K-th place, the most recent one is evicted first, so
earlier epochs win ties as in the historical top-K code. `evaluation.walk_views`
changes final rerun TTA independently of in-training validation views
(`model.backbone.eval_views`). The continuous presets validate with three views
every epoch, as the historical recipes did; that score drives the LR plateau,
early stopping and top-K selection. Validation uses Lightning's validation path,
including validation masks for node tasks. One-logit binary outputs are scored
as two-class probabilities, so AUROC is independent of the evaluation batch size.
Plateau schedulers step only on epochs that validate
(`trainer.check_val_every_n_epoch`).

```bash
python -m topobench -m model=general/trawl dataset=graph/PROTEINS model.backbone.architecture=mamba,sisa,hybrid model.backbone.walks.k=16,32 seed=40,41,42 logger=csv
python -m topobench -m experiment=trawl/proteins_hybrid seed=40,41,42,43,44 logger=csv
python -m topobench model=general/trawl model.backbone.walk_scope=separate model.backbone.fusion=attention dataset=graph/PROTEINS logger=csv
```

Every TRAWL run writes `trawl_manifest.json` beside its output: resolved config,
ordered input/split hashes, implementation hashes, package versions, CUDA
version, and initial supervised-state hash. Keep it with checkpoints and
results. Checkpoint ensembles are evaluation products, not resumable optimizer
states. `trawl_evaluation.json` records the selected checkpoint paths, validation
scores, count, policy, view override and final metrics. Transform schema versions
belong to cache keys; bump them when changing
the meaning of cached fields.

The native `RankedModelCheckpoint` callback retains Lightning's checkpoint
behavior and writes `checkpoints/checkpoint_index.json` at the end of fitting.
This preserves validation-ranked top-K selection when a spawned DDP launcher
returns only the best path to its parent process. The index is checked against
the current best path, monitor and mode before restoration.

## Development checks

```bash
python -m pytest test/nn/trawl -q
python -m ruff check topobench/nn/backbones/general topobench/nn/encoders/trawl.py topobench/nn/readouts/trawl.py topobench/data/utils/trawl
```

The post-reorganization targeted check passed 177 tests across `test/nn/trawl`,
dataset loss, evaluator, optimizer, split utilities, preprocessing, dataloaders,
configuration resolvers and instantiators on CPU with Torch 2.5.1. This includes
the native NCI1 dataloader fixture and synthetic end-to-end runs with and without
pretraining, best-checkpoint selection, top-K weight averaging and logit ensembles.

The September 23 non-notebook run, after the `general` migration and compatibility
repairs, finished with **661 passed, 5 skipped, and no failures**. All **8 tutorial
notebooks passed** in a separate run, giving **669 passed and 5 skipped** across
the two suites. Repairs include:

- Use `Path.name` when composing dataset configs, so the all-datasets loader
  test works on Windows as well as POSIX systems.
- Use `networkx.from_numpy_array` in random-flag lifting and latent-clique tests.
- Verify the cycle-lifting contract: simple connected cycles, correct cycle-space
  dimension and independent columns over GF(2). NetworkX can choose a different
  valid cycle basis across versions; the old test incorrectly required one exact
  choice. The lifting algorithm was not changed. Exact historical reproduction
  still requires matching the original basis and dependency versions.
- Execute tutorial tests with `sys.executable`, a temporary kernel specification,
  and the repository on the kernel import path. No global kernel configuration
  is modified. Install `nbconvert` and `ipykernel` in that same environment.
- Bound the custom-transform tutorial demonstration to ten graphs, with a
  separate subset cache. Production dataset and training configurations still
  use the full dataset.

On Windows, use a fresh workspace-local pytest `--basetemp` when system temporary
file permissions differ. Tutorial notebook execution is validated separately
because it performs dataset downloads and longer training examples.
Do not install `pyg-nightly` alongside stable `torch-geometric`: they overwrite
the same package. Run manifests record both distribution metadata and the actual
loaded PyG version to make such discrepancies visible.

The full validation commands were:

```bash
python -m pytest test --ignore=test/test_tutorials.py -q --basetemp=.test_tmp/general_core_tests --tb=short
python -m pytest test/test_tutorials.py -q --basetemp=.test_tmp/general_notebook_bounded_tests --tb=short
```

Tests cover topology/ranks, union and separate walks, batching isolation,
empty-neighborhood masking, all layer families and gradients, native
classification/regression/multilabel tasks, pretraining without labels, Hydra
composition, a full local synthetic runner, split replay, checkpoint averaging
and ensembling, extended/gated profiles, and frozen numerical references.
Fixtures can be regenerated with
`python scripts/trawl/build_reference_fixture.py --source-root /path/to/trawl`.
The fixture JSON records source hashes; the parent checkout is only needed to
regenerate these development fixtures.

### Extended validation, September 24

After fixing issues exposed by actual GPU execution:

- Windows CPU non-notebook suite: **666 passed, 27 skipped**. Twenty-two skips
  require CUDA, Linux compilation or two GPUs; the other five are existing suite
  skips. All **8 tutorial notebooks passed** again after the shared fixes.
- Linux on mllab05, two NVIDIA RTX A4000 GPUs: the complete TRAWL suite passed
  **101 tests, with no skips**. It covers CPU/GPU forward and gradient agreement,
  FP16 optimizer updates for base/continuous/categorical Mamba, SISA and hybrid
  backbones, official Mamba FP32/FP16 updates, and full-graph compilation of the
  Mamba and SISA sequence layers.
- A strengthened native-runner check passed **8 cases** with Mamba–SISA hybrids
  on one and two GPUs: optional pretraining, best-checkpoint evaluation, top-K
  weight averaging and logit ensembling. The DDP strategy tested is
  `ddp_spawn_find_unused_parameters_true`; this is not a claim about every
  multi-node launcher or a fully compiled topology/sampling pipeline.
- Full-size `proteins_mamba`, `proteins_hybrid`, `nci1_hybrid`, `nci1_sisa`,
  `proteins_gated` and `zinc_categorical` presets passed real-data forward,
  backward and optimizer updates, using two graphs on CPU and eight on GPU.
  Model dimensions, depth, walk budgets and preprocessing settings were retained.
  These are sample-based execution checks, not complete benchmark training.
- Frozen original-source Mamba encoder and SISA fixtures now verify gradients
  and an Adam update in addition to forward outputs. A 10,000-state sparse graph
  passed encoding and walk checks; oversized dense spectral requests correctly
  fail at the configured limit. This does not certify arbitrary graph sizes.

These runs overlap; their counts should not be summed as distinct tests. GPU
validation used Python 3.12, Torch 2.5.1+cu124, PyG 2.6.1, Lightning 2.4.0,
Mamba 2.2.6.post3 and Transformers 4.44.2. Local validation evidence is retained
under `.test_tmp/cluster_evidence`, including logs, resolved recipe configurations
and the cluster dependency freeze. It is intentionally excluded from the PR.

The fixes include full-precision occurrence accumulation under autocast,
importable loss class identities for spawned workers, metric states on the
prediction device for NCCL, distributed loss synchronization, and persisted
checkpoint rankings. The new CPU autocast regression checks the accumulation
forward path; mixed-precision backward is validated on CUDA.

Reproduce the extended checks with:

```bash
python -m pytest test/nn/trawl -q --tb=short
python -m scripts.trawl.validate_recipes --device cuda --graphs 8 --output outputs/recipe_validation.json
```

Under Lightning's `ddp`, `ddp_spawn`, `ddp_fork` or `ddp_notebook` string
strategies, TRAWL runs select the matching `*_find_unused_parameters_true`
strategy, because which parameters receive gradients depends on configuration
and data. Custom strategy objects are left unchanged. In a multi-process group
validation and test loaders shard each split without padding, so every example
is scored exactly once, and the epoch-seeded training order is sharded by rank.
Only rank zero writes `trawl_manifest.json` and `trawl_evaluation.json`.

**Historical benchmark score reproduction remains unverified.** The old cluster
project/environment directory is absent, and the local search recovered source
snapshots, score records and data/walk caches, but no original model checkpoints.
Rebuilding an environment does not establish equality with the lost run state.
The provenance gaps and RNG/initialization differences listed above remain;
no full 300-epoch, five-seed rerun or headline-score equivalence is claimed.
Adaptive walk stopping, pretrained GPSE and proposal-level theoretical claims
remain outside this implementation.

### Benchmark follow-up, September 29

The latest local regression run passed **673 tests, with 29 skips**, and all
**eight tutorials passed**. This includes persistent pretraining checkpoints,
pretraining resume through the next epoch, logical-batch gradient weighting,
and compilation without changing state-dictionary keys. A repeated dataset
loading failure was fixed by declaring the hypergraph loader's actual raw
files, so cached datasets are not downloaded again. Logs are retained locally
in `.test_tmp/regression_final_v8.log` and `.test_tmp/notebooks_final_v2.log`.
The final Linux/CUDA TRAWL suite passed **110 tests with no skips**, including
checkpoint replay, BF16 gating, categorical compilation and walk-head RNG
regressions. Four isolated distributed cases also passed.
An NCCL teardown error in the earlier full-suite run did not recur in either
the isolated repeat or the final full-suite run.

The historical artifact audit recovered W&B configuration, output logs,
dependency records and summaries for all ten headline hybrid runs. Their
listed artifacts contain run history, but no model checkpoints. These records
confirmed microbatches of two graphs, accumulation of four for PROTEINS and
sixteen for NCI1, and hybrid pretraining weight decay of 0.003. The presets now
reflect those settings, including the reconstruction scheduler's 1e-6 floor.
Recovered dependency versions differ from the validated cluster environment;
the recovered records do not establish bitwise historical reproduction.

Two initial seed-40 full training runs finished without runtime errors:
PROTEINS hybrid reached 72.7599% test accuracy and NCI1 hybrid reached 76.3619%.
They used the earlier pretraining weight decay and are diagnostic results,
not accepted reproduction results. All ten corrected hybrid runs completed:
PROTEINS seeds 40–44 averaged **73.835% ± 1.469%**, and NCI1 averaged
**77.121% ± 1.397%** (population standard deviations). Historical recorded means
were 75.196% and 76.752%, respectively. These are measured new-run results,
not exact historical reproduction. NCI1 SISA seed 40 completed at 77.432%, and
The five PROTEINS Mamba seeds completed at **74.337% ± 2.216%**, versus the
historical recorded mean of 74.70%.

**These completed scores are diagnostic and have been superseded for final
acceptance.** A subsequent source audit found an unused graph-head call before
walk-logit prediction. Its dropout consumed extra random draws during training.
The readout now calls the head only for walk predictions (and for graphs without
walks when a fallback is necessary). A regression test verifies both predictions
and RNG state against one direct head call. The affected full seed matrix is
being rerun from a new frozen source snapshot.

The full gated run exposed ARPACK nonconvergence on a larger graph during
spectral preprocessing. Historical spectral encoding now retries that failure
with sparse shift-invert iteration and emits a warning. Successful ordinary
iterations are unchanged, and the retry does not substitute zero features or
construct a dense matrix. A regression test checks the recovered eigenproblem
residual and finite, nonzero features. The gated rerun and final GPU suite
must pass before this validation effort is complete. The final GPU suite has
since passed; subsequent full runs also exposed BF16 dtype mismatches in gate
assignment and weighted readout, now corrected. Categorical encoding had
bypassed the compiled layer call through its residual helper; it now preserves
the same residual-dropout computation through the compiled entry point. ZINC's
full batch exceeded a 16 GiB GPU without activation checkpointing, so its rerun
retains checkpointing and first performs a full-batch memory check.

Backups for sixteen completed diagnostic runs have been downloaded and SHA-256 verified.
All **108 checkpoint files** loaded successfully with finite model tensors and
saved optimizer state. The archives include frozen source, configurations,
metrics and pretraining checkpoints. Local inventory and state-audit records
are `.test_tmp/checkpoint_backup_inventory.json` and
`.test_tmp/checkpoint_state_audit.json`; these large development artifacts are
excluded from the PR. They preserve replacement runs, not the lost originals.

### Audit fixes after the v8 matrix, September 29

The v8 matrix completed eleven PROTEINS runs before a code and configuration
audit; ZINC was stopped at the user's request. PROTEINS hybrid seeds 40–44
averaged **74.69% ± 1.29%** and PROTEINS Mamba **73.19% ± 1.63%** (population
standard deviations). Splits, walk seeds, walk semantics and parameter counts
were checked against the historical code and match for the same seed. The audit
found the following differences and defects, now corrected with regression
tests in `test/nn/trawl/test_regressions.py`:

- Continuous presets validated each epoch with one walk view; the historical
  recipes used three, and that score drives the scheduler, early stopping and
  top-K selection. They now use three.
- One-logit AUROC was computed from padded raw logits, which torchmetrics
  interprets inconsistently between batches. All v8 AUROC values are invalid;
  accuracy, precision and recall were unaffected.
- Top-K ties evicted the earliest checkpoint; historical code evicted the latest.
- Pretraining counted any improvement; historical code required 1e-5.
- Validation ran under BF16 autocast; historical evaluation was full precision.
- `proteins_gated` omitted the native node attribute and used logit ensembling.
- Disconnected historical cell graphs could receive a stationary distribution
  that is zero on whole components (at most 0.3% of PROTEINS/NCI1 graphs); it is
  now solved per component. Connected graphs keep the original ARPACK call.
- Plateau schedulers failed or reused stale metrics when validation ran less
  often than every epoch, including the documented base pretraining command.
- Pretraining resume into a new directory, repeated base-pretraining masks
  within an accumulation window, multi-column bond attributes, attribute-aware
  loader caching, classic-profile `attr_dim`, the stable-protocol branch gate,
  SISA overflow under extreme learned decay, collision-prone stable walk seeds,
  late CSV rerun metrics, the linear-head input dropout and several distributed
  training issues (unused parameters, sampler epochs, padded evaluation,
  concurrent report writes) were also corrected.
- `zinc_categorical` now follows `trawl_zinc.py` for K, patience, test views and
  split-local seeds.

The v8 PROTEINS and NCI1 results are therefore superseded by the v9 rerun from
frozen source `benchmark_source_v9` (archive SHA-256 `8d957029…`), trained in
BF16 with full-precision three-view evaluation. Test accuracy over seeds 40–44
(population standard deviations):

| Recipe | v9 rerun | Historical record |
|---|---|---|
| PROTEINS hybrid | 74.48% ± 1.59% | 75.196% ± 1.328% |
| PROTEINS Mamba | 74.27% ± 1.33% | 74.70% ± 1.03% |
| NCI1 hybrid | 77.61% ± 1.72% | 76.752% ± 1.483% |
| NCI1 SISA (seed 40) | 76.46% | — |
| PROTEINS gated (seed 40) | 73.48% | — |

Each difference from the historical mean is within one standard deviation.
These are measured reruns of the corrected implementation, not certified exact
reproductions. All metric histories, model tensors and optimizer states were
finite. Evaluation-only replay from relocated checkpoints reproduced the recorded
validation and test accuracy exactly for PROTEINS hybrid seed 40 (top-5 weight
average) and PROTEINS Mamba seed 40 (best checkpoint). Before the rerun, the
Windows CPU suite passed 693 tests with 31 hardware skips, all eight tutorials
passed, and the Linux CUDA TRAWL suite on two A4000 GPUs passed 133 tests with no
skips. The CUDA run also exposed, and the rerun source fixes, late CSV metrics
under spawned DDP and parent file-descriptor exhaustion from spawned DDP dataset
sharing (TRAWL spawn runs on Linux now use the file-system sharing strategy).
Per-seed values are recorded in `trawl_recipes.json`.

### Reproducible v10 matrix, October 1

v9 results were valid measurements but not repeatable: ARPACK started from
process state and the stock trainer configuration disabled deterministic CUDA
kernels. The v10 rerun (`benchmark_source_v10`, archive SHA-256 `8aeb7725…`)
uses the reproducibility settings above with otherwise identical recipes:

| Recipe | v10 rerun | Historical record |
|---|---|---|
| PROTEINS hybrid | 73.91% ± 1.78% | 75.196% ± 1.328% |
| PROTEINS Mamba | 74.55% ± 2.08% | 74.70% ± 1.03% |
| NCI1 hybrid | 77.53% ± 0.65% | 76.752% ± 1.483% |
| NCI1 SISA (seed 40) | 78.02% | — |
| PROTEINS gated (seed 40) | 73.48% | — |

PROTEINS hybrid seed 40 was repeated on a different host: every model,
optimizer and scheduler tensor in all eight checkpoints, the metric history and
the evaluation report were bit-identical. v9 and v10 differ only in the
reproducibility changes, yet individual seeds moved by up to about two points;
this is the run-to-run variance at this test-set size (one PROTEINS test graph
is 0.36 points), and every five-seed mean remains within one standard deviation
of the historical record.

Runtime profiling on an RTX A4000 measured the full-size hybrid at batch size
eight: eager FP32 took 1.166 seconds per update; compiled FP32 took 0.324 seconds,
compiled FP16 0.300 seconds, and compiled BF16 0.317 seconds. Compilation warmup
took approximately 78–88 seconds. One, two and four CPU math threads had similar
eager throughput. These measurements support layer compilation and avoiding
CPU oversubscription; they do not prove optimal settings for every recipe or
for the historical microbatch size of two. The profiling script records batch
size in new result files.

The follow-up measurement at microbatch size two confirmed the benefit:
eager FP32 took 0.305 seconds per update and compiled FP32 0.084 seconds;
eager BF16 took 0.302 seconds and compiled BF16 0.080 seconds. Compiled BF16
used approximately 682 MiB peak allocated GPU memory, versus 929 MiB eager.
The BF16 compile warmup reused the host's compiler cache, so its 10-second
warmup should not be treated as a cold-start measurement. These timings include
walk sampling, forward/backward, clipping and an optimizer step on one fixed
two-graph batch; they are not whole-epoch timings or a dataloader-worker study.

The corrected benchmark runners use one job per GPU, BF16, one CPU math thread,
and layer compilation. Continuous presets disable activation checkpointing to
avoid recomputation; the full-batch ZINC preset retains it to fit 16 GiB GPUs.
They preserve the configured logical batch size. Precision and compilation may
change numerical trajectories, so speed measurements and execution success
must be distinguished from exact score reproduction. Enable compilation with
`model.compile=true`; the TRAWL model's `compile_scope: layers` compiles sequence
layers while leaving graph preparation and walk sampling outside compilation.

Use the measured runtime settings on a compatible CUDA host with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
TORCHINDUCTOR_COMPILE_THREADS=2 CUDA_VISIBLE_DEVICES=0 \
python -m topobench experiment=trawl/proteins_hybrid trainer=gpu \
  model.compile=true +trainer.precision=bf16-mixed \
  model.backbone.checkpoint_layers=false logger=csv
```

The base configuration keeps compilation optional for CPU compatibility.
Change the experiment name to select another recipe. For `zinc_categorical`,
omit `model.backbone.checkpoint_layers=false` to retain its preset's memory
saving checkpointing. Keep the recipe's
microbatch and accumulation settings for historical comparisons; increasing
batch size can change BatchNorm statistics and the random-number trajectory.
