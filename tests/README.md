<!--
    SPDX-FileCopyrightText: 2026 SeisSol Group

    SPDX-License-Identifier: MIT
-->

# TensorForge end-to-end test harness

Numerical, on-device tests: generate → compile → run → compare against NumPy.

## Quickstart

```bash
pip install -e '.[test]'              # at the repo root
cd tests
pytest -v                             # auto-discovers GPUs and toolchains
```

What you should see at the top of the output:

```bash
tensorforge targets:
  - cuda-sm_86-dev0  (NVIDIA GeForce RTX 3090)
```

If no targets are found, all GPU-bound tests are skipped (with a clear
reason); the host-only checks still run, so the harness itself stays
covered.

## Golden snapshots

`test_snapshots.py` freezes the generated source for every case on every
codegen backend under `snapshots/`. It needs no GPU and no toolchain, so it
runs everywhere and is the check that makes a codegen refactor reviewable:
the diff *is* the review. A snapshot changing is not a failure, it is a
request to look. When the change is intended:

```bash
pytest --snapshot-update      # rewrites snapshots/
```

and commit the rewritten snapshots in the same change that caused them.

Cases that fail to generate are snapshotted too, as `FAILED: <Exception>`.
Losing or gaining such a failure is as interesting as a shifted line.
`test_no_orphaned_snapshots` catches the other direction: a snapshot with no
case behind it means a case was renamed and quietly stopped being covered.

## What this suite covers

Cases live under `cases/`, grouped by feature:

| Group                 | Path                  | What it exercises                                                                       |
|-----------------------|-----------------------|-----------------------------------------------------------------------------------------|
| Plain GEMM            | `cases/*.py`          | dense GEMM in every dtype, transposes, alpha scaling, accumulation, fused chains        |
| Elementwise           | `cases/elementwise/`  | `ElementwiseDescr`, mostly one op per case (sqrt, exp, log1p, …)                        |
| Slicing               | `cases/slicing/`      | non-trivial `BoundingBox` inside larger `Tensor.shape`, temporaries written in parts    |
| Reductions            | `cases/reduction/`    | `ReductionDescr` for sum/min/max/prod outside the multilinear lowering                  |
| Mixed                 | `cases/mixed/`        | sequences of multilinear, elementwise and reduction descriptors sharing temporaries     |
| Barriers              | `cases/barrier/`      | `GridFenceDescr` / `GridBarrierDescr` between descrs (multi-section, cooperative launch)|
| Flags                 | `cases/flags/`        | the per-element mask: asked for, or attributes without it                               |

Top-level `cases/*.py` also include the *single-feature* coverage
cases — `trans_b`, `add_true`, `addressing_none`,
`addressing_ptr_based`, `sparsity_band` — each of which carries a
host-only smoke test in `test_kernels.py` asserting its distinguishing
property (so a case that gets accidentally rewritten into a plain GEMM
is caught).

### XFAIL strict

A case can be marked `XFAIL=True` with an `XFAIL_REASON` string,
which `conftest.py:pytest_generate_tests` translates into
`pytest.mark.xfail(strict=True, run=True)`. The strict flag matters:
the first time one of these cases passes, it turns into a hard
failure, which is the signal to drop the marker.

No case carries the marker at the moment.

## Selecting subsets

```bash
pytest -k square                       # only the square-GEMM case
pytest -k sm_86                        # only sm_86 targets
pytest -k 'sparse and sm_86'           # both
pytest --co -q                         # list, don't run
```

## When a case fails

The runner deposits artifacts under
`<cache>/_failures/<case>-<target>/`:

| File              | What it is                              |
|-------------------|-----------------------------------------|
| `kernel.cu`       | the generated kernel source             |
| `expected.npy`    | the NumPy reference output              |
| `got.npy`         | what the GPU actually wrote             |
| `compile.log`     | nvcc/hipcc output (if build itself died)|
| `stderr.txt`      | runtime stderr (if the binary died)     |

Default cache root is `~/.cache/tensorforge-tests`; override via
`TF_TEST_CACHE=/path/to/somewhere`.

## Adding a case

A case is one Python file under `cases/` (or a subdirectory of it).
The minimum:

```python
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr

NAME  = "my_case"           # unique; appears in pytest IDs
DTYPE = Datatype.F32
BATCH = 8                   # optional, default 8
TOL   = (1e-5, 1e-5)        # (rtol, atol); optional, sane defaults per dtype

def descr_list():
    a = SubTensor(Tensor([M, K], Addressing.STRIDED,
                         BoundingBox([0, 0], [M, K]),
                         alias="A", datatype=DTYPE))
    # ... b, c ...
    return [GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)]
```

For a single `MultilinearDescr` (the GEMM case), the harness derives
the NumPy reference automatically — you only need to set `alias=` on
each operand. For anything more exotic (chains, fused operations,
custom epilogues), define `reference(inputs, dest_in)` returning the
expected `(batch, *shape)` array.

Optional module-level knobs the harness recognizes:

| Name              | Type               | Effect                                                                                              |
|-------------------|--------------------|-----------------------------------------------------------------------------------------------------|
| `NAME`            | `str`              | required; becomes the pytest parametrize ID                                                         |
| `DTYPE`           | `Datatype`         | required; selects the kernel float type                                                             |
| `BATCH`           | `int`              | batch size (default 8)                                                                              |
| `TOL`             | `(rtol, atol)`     | NumPy `allclose` tolerances                                                                         |
| `descr_list()`    | `-> List[Descr]`   | required; the description sequence the generator consumes                                           |
| `reference(...)`  | `-> np.ndarray`    | optional; per-case NumPy oracle, otherwise auto-derived for single-MultilinearDescr cases           |
| `INPUT_TRANSFORM` | `Dict[str, fn]`    | optional; per-input domain shaping, applied before both kernel and reference see the values         |
| `XFAIL`           | `bool`             | optional; when true, conftest attaches `pytest.mark.xfail(strict=True, run=True)`                   |
| `XFAIL_REASON`    | `str`              | optional but required-with-XFAIL; what's broken upstream and what fix turns the marker into xpass  |
| `ATTRS`           | `dict`             | optional; the kernel attributes a frontend would pass. Omitting it is *not* `{}`: no `ATTRS` means the flag mask is generated with a null check (what a frontend without attributes gets), `{}` means no mask at all, `{"flags": True}` means a mask the launcher requires |

### Elementwise cases

`tensorforge.generators.elementwise` builds an `ElementwiseDescr` per
operation, destination first:

```python
from tensorforge.generators import elementwise as ew

def descr_list():
    a = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="A", datatype=DTYPE))
    b = SubTensor(Tensor([16, 16], Addressing.STRIDED,
                         BoundingBox([0, 0], [16, 16]),
                         alias="B", datatype=DTYPE))
    return [ew.sqrt(b, a)]

def reference(inputs, dest_in):
    return np.sqrt(inputs["A"])
```

For domain-sensitive ops (`sqrt`, `log`, `rcp`), set
`INPUT_TRANSFORM = {"A": lambda x: np.abs(x) + 0.1}` to constrain
the RNG-generated input before it reaches both the kernel and the
reference. The transform writes through the per-element view, which
aliases the underlying flat buffer (see `harness/layout.py`), so the
bytes shipped to the GPU are the post-transform values.

Most cases in this group apply one operation, which keeps each a focused
contract: if it fails, that operation is the suspect.  They cover `abs`,
`exp`, `expm1`, `log`, `log1p`, `rcp`, `sin`, `sqrt` and `tanh`, and the
general power in `pow_int` (an exponent `ew.pow` does not fold into a
cheaper operation).  `slice_after_whole` and `sliced_max` combine
elementwise writes with slices of a shared tensor.

### Slicing cases

When `Tensor.shape` is larger than the active `BoundingBox`, the
generator allocates and addresses the full storage but only reads from
and writes to the bbox region. The harness mirrors that: the input
buffer `inputs["A"]` is full-storage and the reference is responsible
for the bbox math.

The standard pattern:

```python
STORAGE = (32, 32)
SUB = (slice(8, 24), slice(8, 24))      # bbox as numpy slices

def descr_list():
    a = SubTensor(Tensor(list(STORAGE), Addressing.STRIDED,
                         BoundingBox([8, 8], [24, 24]),
                         alias="A", datatype=DTYPE))
    # ... b, c with matching pattern ...

def reference(inputs, dest_in):
    out = np.array(dest_in, copy=True)  # non-bbox cells preserved
    A_sub = inputs["A"][:, *SUB]
    B_sub = inputs["B"][:, *SUB]
    out[:, *SUB] = np.einsum("bik,bkj->bij", A_sub, B_sub)
    return out
```

Two invariants the reference must respect:

* the kernel does **not** touch non-bbox cells — the reference must
  copy `dest_in` and overwrite only the bbox window;
* the host-side input arrays the kernel sees include both bbox and
  non-bbox cells (filled with RNG noise) — the reference must
  contract/operate only over the bbox window, otherwise its result
  silently disagrees with what the kernel actually computes.

Coverage includes `slicing/inner_region` (offset bbox in larger
storage), `offset_a` (only A sliced — exercises the mixed-stride code
path), `slice_offset_a`, and `chain` (a chain over sliced inputs, where a
GEMM in the middle has to keep the offset); the `slicing/temp_*` cases
write temporaries in parts and read them back whole or sliced.

### Barriers

`GridFenceDescr` and `GridBarrierDescr` split the descr list into
multiple sections; the generator emits *one* kernel that contains all
sections, with a section boundary inside it. Two cases under
`cases/barrier/`:

* `fence_two_gemms` — `GridFenceDescr.trueBarrier()` returns `False`;
  two sections inside one kernel, plain `<<<...>>>` launch.  The two
  GEMMs are independent, because a fence does not order its sections;
  the case pins the per-section launcher parameters and the offset
  traversal that still covers every element.
* `barrier_two_gemms` — `GridBarrierDescr.trueBarrier()` returns
  `True`; switches the generator into persistent-threading mode and
  emits `cudaLaunchCooperativeKernel`, which needs
  `tensorforge_aux.h::argsPtrs` (on the driver's include path via
  `toolchain.py`) and `cooperative_groups.h` (included by
  `tensorforge_device/cuda.h`).  Its batch exceeds the grid, which is
  what guards where the grid sync sits.

The docstrings of both cases say what they can and cannot show.

#### Multi-section dispatch in the driver

The launcher signature carries one `numElements{i}` and one
`flags{i}` parameter per section. `driver_emit.py` counts the sections
of the generator and emits that many arguments at the call site; a
fixed single pair would be an arity mismatch for every case with a
fence or a barrier.

### Reduction cases

`ReductionInstruction` emits the *non-lead* fold: every contracted axis
is a sequential axis, so each lane owns one point of the kept iteration
space and folds into a register, with no cross-lane traffic, no shared
memory and no barrier. `sum_axis`, `max_axis`, `min_axis` and
`prod_axis` take that path.

`max_all`, `sum_axis0` and `sum_axis0_40` contract the lead axis, so
their fold crosses lanes.  In `max_all`, for one, each
lane folds what it owns, taking the neutral element where it owns nothing,
then one `tensorforge::reduction` call over the lane partials, then lane 0
stores. The guard around step one is an `if`/`else` yielding a value rather
than a bare bounds check, because a shuffle reached by only part of the warp
is undefined.

The cases reflect the operator space:

* `sum_axis` — `AddOperator` (numerically equivalent to
`cases/trace.py` which uses `MultilinearDescr`);

* `max_axis`, `min_axis` — the cases that genuinely
need `ReductionDescr` (max/min aren't multilinear), and the pair that
pins down the `±inf` neutral elements;

* `max_all` — full reduction to a single-element
sink, and the lead-axis contraction;

* `prod_axis` — `MulOperator` with neutral element
1 (constrained-domain inputs to avoid overflow), the one operator
`Op.ACCUM` cannot express since it lowers to `+=`;

* `sum_axis0`, `sum_axis0_40` — the lead axis contracted and the other
kept, with one lead element per lane and with more lead elements than
lanes.

### F64 variants

Feature axes get an F64 sibling: `add_true_f64`,
`addressing_none_f64`, `slicing/inner_region_f64`,
`elementwise/{exp,sqrt,tanh}_f64`, `sparsity_band_f64`. They exist because
several emit paths split on dtype — `sqrtf` vs `sqrt`, `0.0f` vs `0.0`
in the sparsity unrolled sequences, dtype-dependent literals in the
`NONE`/`PTR_BASED` offset arithmetic — and a regression that only
breaks F64 is invisible to the F32 cases.

A host-only check (`test_f64_variant_cases_use_double_precision`)
asserts that every `*_f64.py` actually carries `Datatype.F64` on its
operand tensors, catching the copy-paste mistake of changing the
module-level `DTYPE` constant but forgetting the per-tensor argument.

### Yateto frontend

`tensorforge.frontend.yateto.YatetoFrontend` is the production entry
point used by SeisSol. The in-repo suite carries one host-only smoke
(`test_yateto_frontend_imports_and_constructs`) that imports
`YatetoFrontend` and constructs an instance — enough to catch
interface drift in `tensorforge.interface` that would break SeisSol's
import path. End-to-end tests that actually
feed yateto-emitted descriptions are out of scope for this suite;
SeisSol's CI is the right place for those.

### Cases that target a single feature

The top-level `cases/*.py` includes some files that look like plain
GEMMs but each tests exactly one feature axis:

| File                       | Feature                                            | Notes                                                                                                                    |
|----------------------------|----------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `trans_a.py`               | `trans_a=True` (transpose first operand)           | green                                                                                                                    |
| `trans_b.py`               | `trans_b=True` (transpose second operand)          | green                                                                                                                    |
| `csa_alpha.py`             | `alpha != 1` synthetic-scalar path                 | green; alpha travels as a scalar tensor with the operands' datatype                                                       |
| `add_true.py`              | `add=True` on bare `MultilinearDescr`              | green; the constructor the yateto frontend calls, rather than `GemmDescr` with `beta=1`                                  |
| `addressing_none.py`       | `Addressing.NONE` (batch-constant operator matrix) | green; SeisSol's static-operator pattern                                                                                  |
| `addressing_ptr_based.py`  | `Addressing.PTR_BASED` (heterogeneous batches)     | green; the driver builds the per-batch pointer array (`T**`)                                                             |
| `sparsity_band.py`         | `Tensor(..., spp=MaskSPP(...))`                    | green; banded `B` operand with cells outside the mask zeroed by `INPUT_TRANSFORM`                                          |
| `f64.py`, `f128.py`        | non-F32 dtypes                                     | green; F128 requires a compiler with `__float128` support                                                                 |

Each of these has a dedicated host-only smoke test in
`test_kernels.py` that asserts the distinguishing property (e.g. that
`trans_b` actually sets `permute=[1, 0]` on the second operand). That
way a case that gets refactored into "just another GEMM" is caught
even if no GPU is available.

## Backends

The harness supports four backends:

| Backend  | Compiler  | Vendor required | Notes                          |
|----------|-----------|-----------------|--------------------------------|
| `cuda`   | `nvcc`    | NVIDIA          | sets `-arch=sm_XX`             |
| `hip`    | `hipcc`   | AMD             | sets `--offload-arch=gfxYYY`   |
| `oneapi` | `icpx`    | Intel           | uses `-fsycl` (JIT)            |
| `acpp`   | `acpp`    | any             | AdaptiveCpp (any vendor)       |

Override the compiler binary via `$NVCC`, `$HIPCC`, `$ICPX`, or
`$ACPP`.

## CI use

A runner without a GPU can still validate generation + compile by
setting `TF_TEST_FAKE_GPUS=sm_86,gfx90a`. The probe still requires a
real toolchain — that's the point: catch a generator change that
produces uncompilable C++ before it reaches a hardware runner.

The host-only tests (`test_layout_roundtrip`,
`test_reference_matches_einsum`, `test_elementwise_descr_constructs`,
`test_slicing_cases_construct_and_generate`,
`test_reduction_descr_constructs`,
`test_barriers_cases_construct_and_generate`,
`test_yateto_frontend_imports_and_constructs`,
`test_f64_variant_cases_use_double_precision`) plus the six per-feature
smokes are completely independent of any GPU or toolchain — they catch
regressions in the case-level descriptors themselves.

## Extending

* New backend: add a probe + compile recipe in `harness/toolchain.py`,
  plus a preamble in `harness/driver_emit.py`. The driver template is
  otherwise shared.
* New addressing mode: handle it in `driver_emit.collect_operands` and
  in `harness/layout.py`. The generator already emits the right device
  code; the harness just needs to mirror the host-side allocation.
* New `OperationDescription` subclass (custom epilogue, fused
  reduction-then-elementwise, etc.): add a branch in `Generator._emit_ir`
  and an `Instruction` to back it; the test side just needs `XFAIL=True`
  until the path lights up.
