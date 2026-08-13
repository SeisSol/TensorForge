# Diagnostics

Read-only reports on the state of the AMD code generation path. None of them
change anything; they answer questions that came up repeatedly while working
on it, and that are easier to re-run than to re-derive.

Run from the repository root.

| tool | question |
|---|---|
| `reachability.py` | what can `matmul()` actually reach, and what is defined twice? |
| `arch_sweep.py` | does every supported AMD target still generate, and which helpers does it emit? |
| `undefined_symbols.py` | does any target call a `fmacdpp` variant its runtime does not define? |
| `duplicate_elements.py` | does any output element get computed by more than one path? |
| `ir_opacity.py` | how much of the emitted IR is opaque to the passes, versus structured? |
| `layout_census.py` | which register layouts does the generator actually produce? |
| `operand_layouts.py` | do the vendor intrinsics receive their operands in the distribution they require? |

`ir_opacity.py` distinguishes three kinds rather than two. A `rawexpr` still
carries vendor-specific text, but it has an SSA result and a declared memory
effect, so a pass can reorder around it and reuse it; a `rawstmt` with
`Effect.UNKNOWN` can do neither. Counting them together hides the difference
that matters.

## mutation_check.py

Not a report: it puts defects *back* and checks that the guards notice.

Every test added alongside these tools is only worth its runtime if it can
fail, and two ways of failing silently are easy to reach — a property that is
trivially true, or a test that shares a mistake with the code it checks. Both
happened here. So each guard has a matching mutation, taken from the defects
that were actually found rather than invented:

```bash
python3 tools/mutation_check.py            # every group
python3 tools/mutation_check.py layout     # one group
```

Source files are edited in place and restored in a `finally`; run it on a
clean tree. A mutation that no longer applies is reported as skipped rather
than passed — if the code has moved, that check has stopped testing anything
and should be repaired, not trusted.

The harness clears `__pycache__` and runs pytest with `-B`. That is not
hygiene: CPython invalidates a `.pyc` by comparing the source mtime at
one-second resolution, and these files are rewritten several times a second,
so a stale cache would let the subprocess import the unmutated module and
report a guard as working when it was never exercised. That is exactly the
false confidence this harness exists to prevent, and it produced it once
before the caches were cleared.
