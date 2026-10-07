<!--
    SPDX-FileCopyrightText: 2026 SeisSol Group

    SPDX-License-Identifier: MIT
-->

# Tools

Scripts that answer questions about the generator and what it emits. They
read; the exceptions say so below: `mutation_check.py` edits the source and
restores it, the `amd_matrix_*` scripts regenerate reference data under
`tests/data`, and the `seissol_*` scripts write captures.

Run them from the repository root, since they find `tests/cases` and
`tests/snapshots` by relative path. Each puts the `src` directory of the
checkout it belongs to ahead of an installed TensorForge, so a tool reports on
its own tree whatever else is installed.

Two subdirectories have their own READMEs: [`bench/`](bench/README.md) builds
and times kernels on a device, and [`host/`](host/README.md) runs generated
kernels on the host against a NumPy evaluation of their descriptors.

## The generated code

| tool | question |
|---|---|
| `syntax_check.py` | does every snapshot's kernel parse, and does every call in it resolve? `g++` over the host shim, for machines without a vendor compiler |
| `arch_sweep.py` | does every AMD target generate the case corpus, and which helpers does each emit? |
| `undefined_symbols.py` | does any target call an `fmacdpp` variant its runtime does not define? |
| `duplicate_elements.py` | is any output element computed by more than one path? |
| `operand_layouts.py` | does the A operand of `fmacdpp{step}` always arrive in the distribution the instruction requires? |
| `reachability.py` | what in the AMD package can its single entry point, `amd.matmul`, reach, and what is defined twice? |

## Guards for a change

| tool | question |
|---|---|
| `access_equiv.py` | did a change alter *which* memory a kernel touches, or only what things are called? |
| `mutation_check.py` | do the tests catch the defects they exist for? |

`access_equiv.py` answers the question a snapshot diff cannot. Unpinning an
address renumbers every later SSA value and lets single-use addresses fold
into their loads, so thousands of lines move without a single access moving,
and reviewing that by eye is how a real change gets waved through inside it.
The tool expands every name in every subscript down to leaves (loop variables,
thread indices, literals), canonicalizes, and compares the multiset of
`(base, address)` pairs with the snapshots at a git revision:

```bash
python3 tools/access_equiv.py            # against HEAD
python3 tools/access_equiv.py HEAD~3
```

Renumbering, parenthesization and identity terms (`0 + x`, `1 * x`) are
canonicalized away; associativity and distribution deliberately are not,
because on an address `a*(b+c)` and `a*b + c` usually differ for a reason. A
subscript it cannot parse raises rather than falling back to comparing text:
the answer is worth something only because it licenses not reading the diff.
`tests/test_access_equiv.py` pins both directions, a pair it must call
identical next to a pair it must call different for everything it ignores.

`mutation_check.py` plants defects and checks that the tests notice. A
test can pass because its property is trivially true, or because it shares a
mistake with the code it checks, and either reads as coverage. So each guard
has a matching mutation, a defect the code can actually have:

```bash
python3 tools/mutation_check.py            # every group
python3 tools/mutation_check.py layout     # one group
python3 tools/mutation_check.py --dry-run  # do the anchors still match?
```

Source files are edited in place and restored in a `finally`, and the paths
under mutation are also written to a lock file, so a run killed before its
`finally` is restored from git by the next one. Run it on a clean tree. A
mutation that no longer applies is reported as skipped rather than passed:
the code has moved, and that check has stopped testing anything. The tests
run with this tree's `src` first and without bytecode caches, since CPython
compares source mtimes at one-second resolution and a stale `.pyc` would let
the run import the unmutated module.

## What the passes can see

| tool | question |
|---|---|
| `ir_opacity.py` | how much of each kernel is raw text a pass cannot see through, and which function emitted it? |
| `macro_surface.py` | which statements are written straight to the output and never reach the IR at all? |
| `buffer_spans.py` | which names still connect definitions and uses across separate IR bodies? |
| `layout_census.py` | how many distinct register layouts does the generator produce? |
| `wrap_census.py` | which transfers does `enable_wrap_loads` move across the back edge, and why does it leave the rest? |
| `overlap_census.py` | how far apart are a transfer, its wait, and the next transfer? |
| `lane_census.py` | how much of the wave does each descriptor use, given that a kernel's widest descriptor sets the thread count for all? |
| `staging_census.py` | what does each staged shared image accomplish: a broadcast, a relayout, or a round trip that only spills? |

`ir_opacity.py` sorts raw nodes into three kinds. A `rawexpr` still carries
vendor text, but it has an SSA result and a declared memory effect, so a pass
can reorder around it and reuse it; a `rawstmt` with `Effect.UNKNOWN` can do
neither; and a comment is inert. It counts twice: *constructed* is what the
generator hands the builder, and *lowered* is what survives `pir.optimize` and
reaches codegen. The second is the one that says how much of the output is
opaque, and it is smaller, because passes delete raw nodes -- `flatten_scopes`
removes every scope that declares nothing. Both are attributed to the
function that emitted them, lowered nodes by their text, since the passes
rebuild statements and the emitting frame is gone by then.

It runs the whole corpus, recursively, as `conftest.py` discovers it. A case
that stops generating still counts what it emitted before stopping, so the
total does not move whenever an unrelated defect is fixed.

## Cost models

| tool | question |
|---|---|
| `bank_conflicts.py` | what does every shared-memory access cost in bank cycles? |
| `register_usage.py` | does the register-pressure model rank two lane configurations the way the vendor compiler does? |
| `register_tradeoff.py` | would keeping a staged image in registers fit? |
| `calibrate_icache.py` | fits the instruction-cache estimate to what the compilers emit |
| `calibrate_mix.py` | fits the machine instructions per emitted statement, by kind of statement |

## Constant operands and their staging

For SeisSol's globally constant matrices: how sparse they are, and what
staging them in shared memory buys.

| tool | question |
|---|---|
| `spp_metrics.py` | the metrics over one sparsity pattern |
| `spp_layout.py` | the storage layout, in the form the frontend takes |
| `spp_plan.py` | which operands get staged, and in what layout |
| `spp_occupancy.py` | what staging an operand costs in residency |
| `spp_sweep.py` | where the staging decision changes, across the corpus |
| `spp_kernels.py` | the constant operands as the code generator sees them, from a descriptor capture |
| `seissol_corpus.py` | the metrics over SeisSol's `matrices_N.xml` |

## SeisSol captures

| tool | question |
|---|---|
| `seissol_export.py` | what does SeisSol's code generator hand a GPU exporter for one configuration? |
| `seissol_store.py` | packs those exports into the content-addressed store under `tests/fixtures/seissol` |

## Reference data

| tool | question |
|---|---|
| `amd_matrix_table.py` | regenerates `tests/data/amd_matrix_builtins.json` from LLVM |
| `amd_matrix_layouts.py` | regenerates `tests/data/amd_matrix_layouts.json` from AMD's matrix instruction calculator |
