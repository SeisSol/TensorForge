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
reason); the host-only `test_layout_roundtrip` and
`test_reference_matches_einsum` still run, so the harness itself stays
covered.

## Selecting subsets

```bash
pytest -k square                       # only the square-GEMM case
pytest -k sm_86                        # only sm_86 targets
pytest -k 'square and sm_86'           # both
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

A case is one Python file under `cases/<category>/`. The minimum:

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
expected ``(batch, *shape)`` array.

## Backends

The harness supports four backends:

| Backend  | Compiler  | Vendor required | Notes                          |
|----------|-----------|-----------------|--------------------------------|
| ``cuda`` | ``nvcc``  | NVIDIA          | sets ``-arch=sm_XX``           |
| ``hip``  | ``hipcc`` | AMD             | sets ``--offload-arch=gfxYYY`` |
| ``oneapi`` | ``icpx``  | Intel         | uses ``-fsycl`` (JIT)          |
| ``acpp`` | ``acpp``  | any             | AdaptiveCpp (any vendor)       |

Override the compiler binary via ``$NVCC``, ``$HIPCC``, ``$ICPX``, or
``$ACPP``.

## CI use

A runner without a GPU can still validate generation + compile by
setting ``TF_TEST_FAKE_GPUS=sm_86,gfx90a``. The probe still requires a
real toolchain — that's the point: catch a generator change that
produces uncompilable C++ before it reaches a hardware runner.

## Known issue this harness is designed to expose

Running the rectangular-GEMM case with `alpha != 1.0` or `beta != 0.0`
currently triggers `assert False` in `tensorforge/backend/symbol.py`
inside `Symbol.get_fptype`: the synthetic `alpha_tensor` made by
`GemmDescr.__init__` has no `datatype` set. The harness reproduces
this deterministically, which makes it the first concrete fix to land.

## Extending

* New backend: add a probe + compile recipe in
  `harness/toolchain.py`, plus a preamble in
  `harness/driver_emit.py`. The driver template is otherwise shared.
* New addressing mode (`PTR_BASED`, scalar): handle it in
  `driver_emit.collect_operands` and in `harness/layout.py`. The
  generator already emits the right device code; the harness just
  needs to mirror the host-side allocation.
