<!--
    SPDX-FileCopyrightText: 2026 SeisSol Group

    SPDX-License-Identifier: MIT
-->

# Host oracle for TensorForge kernels

Checking a generated kernel by reading it stopped working somewhere around the
fourth defect: the code looks right, the numbers are wrong, and the guessing
takes a round each time.  These tools replace the guessing with an oracle that
needs no GPU.

The idea is small.  `tests/kernel_eval.py` already interprets one thread of a
generated CUDA kernel.  Shared memory is where threads meet, so a single thread
sees whatever the others have not written --- but splitting the per-element body
at its barriers and driving all 32 lanes through one phase at a time gives the
same guarantee the hardware does, on one shared `Slot`.  The other half is a
NumPy evaluation of the same descriptor list the frontend handed the backend.
Agreement to machine precision is then a real statement about the kernel.

Validated against the poroelastic order-4 set: 56 of 60 kernels run (the other
four use vectorised loads the interpreter does not model), and on a correct
backend all 56 match with a relative deviation below 1e-15.

## Setup

An editable install, so the tools can find `tests/kernel_eval.py`:

    pip install -e /path/to/tensorforge

Otherwise set `TF_TESTS` to the directory holding `kernel_eval.py`.  NumPy is
the only other requirement.

## The usual sequence

Capture what the frontend produced, from SeisSol's `codegen/`:

    python3 dump_descriptors.py --out descriptors.json -- \
        --equations poroelastic --matricesDir matrices --outputDir gen \
        --host_arch hsw --device_backend cuda --device_arch sm_86 \
        --device_vendor nvidia --order 4 --precision s \
        --numberOfMechanisms 0 --memLayout config/gpu/dense.xml \
        --multipleSimulations 1 --PlasticityMethod nb \
        --gemm_tools tensorforge --device_codegen tensorforge \
        --drQuadRule dunavant

Then check every kernel it generated:

    python3 validate_dump.py gen/gpulike_subroutine.cpp descriptors.json

    kernel_01064e7714        worst rel =          0  OK
    kernel_0bf208a83b        worst rel =      843.2  MISMATCH  [('m2', 843.2)]
    ...

And narrow a failure to a minimal case:

    python3 prefix_bisect.py descriptors.json kernel_0bf208a83b

    prefix  50: worst rel = 225.6
    prefix  25: worst rel = 395.7
    ...
    shortest failing prefix: 4

Four descriptors, rebuilt as live objects from the capture rather than guessed
at --- small enough to read the generated code for, and to turn into a test.

## The tools

| | |
|---|---|
| `dump_descriptors.py` | Capture every kernel's descriptor list from a codegen run. Everything else works off this file. |
| `validate_dump.py` | Run each kernel on the host, all lanes, and compare with NumPy. |
| `prefix_bisect.py` | Rebuild a descriptor list as live objects and bisect to the shortest prefix that is wrong. |
| `check_structure.py` | Structural checks over a dump, no reference needed: results computed and discarded, a register serving as bias twice, a load overtaken by a store to what it reads, a register array indexed outside its declared range. Each of these was a real defect. |
| `read_before_write.py` | Per descriptor, in program order: which regions does a kernel read that nothing wrote first? |
| `lockstep.py` | The lane-parallel runner. |
| `reference.py` | The NumPy evaluation of a descriptor list. |

`check_structure.py` is the cheap one --- it takes a dump and nothing else, and
is worth running on any new generation:

    python3 check_structure.py gen/gpulike_subroutine.cpp
    flagged: 0 of 60

## What it does not cover

`validate_dump.py` seeds a destination that is *only* accumulated onto with a
nonzero value, so a dropped bias shows.  It deliberately does not do that for a
tensor with an assignment among its writers: yateto's contract is that such a
tensor is fully defined by the kernel, and the store zero-fills outside the
eqspp window --- which `reference.py` does not model, so a seeded value would
register as a disagreement that is not one.

`read_before_write.py` reports reads with no preceding write.  Not all of them
are defects: a global output may be filled by the caller.  In the poroelastic
set it flags `spaceTimePredictor` row 0 and the alignment padding, which are
SeisSol's to initialise --- worth confirming on that side, since nothing in the
kernel does it.

One kernel is executed per run, for one batch element, with `flags0 == nullptr`
and no extra offset.  Backends other than CUDA are not interpretable: HIP
kernels use cross-lane primitives the interpreter has no model for.
