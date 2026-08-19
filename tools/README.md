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
| `access_equiv.py` | did a refactor change *which* memory is touched, or only what it is called? |
| `ir_opacity.py` | how much of the emitted IR is opaque to the passes, versus structured -- and which function emitted it? |
| `access_equiv.py` | did a refactor change *which* memory gets touched, or only what it is called? |
| `layout_census.py` | which register layouts does the generator actually produce? |
| `operand_layouts.py` | do the vendor intrinsics receive their operands in the distribution they require? |

`ir_opacity.py` distinguishes three kinds rather than two. A `rawexpr` still
carries vendor-specific text, but it has an SSA result and a declared memory
effect, so a pass can reorder around it and reuse it; a `rawstmt` with
`Effect.UNKNOWN` can do neither. Counting them together hides the difference
that matters.

`access_equiv.py` exists because a textual snapshot diff answers "did the
source change", which during a migration is almost always yes and almost never
the question. Unpinning an address renumbers every SSA value after it and lets
single-use addresses inline into their loads, so thousands of lines move
without a single access moving. It expands every subscript down to leaves and
compares the multiset of `(base, address)` pairs against a git revision.

What it canonicalises away -- renumbering, parenthesisation, `0 + x` -- is
chosen; what it refuses to canonicalise -- associativity, distribution -- is
chosen just as deliberately, because on an address those are usually real. Its
answer licenses not reading the diff, so `tests/test_access_equiv.py` pins both
directions: for each thing it ignores, a pair it must call identical, and next
to it a pair it must call different.

It runs the whole case corpus -- recursively, the way `conftest.py` discovers
cases, because `barrier/`, `elementwise/`, `reduction/` and `slicing/` hold 24
of the 52 between them -- and attributes every raw node to the function that
emitted it. The percentage says how far along
the migration is; the site table says what to change next, which is the
question that actually gets asked. Cases that stop generating still count
whatever they emitted before stopping -- dropping them would move the total
whenever an unrelated defect is fixed, and a baseline that moves for reasons
outside the change under test is not a baseline.

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

`access_equiv.py` answers the question a snapshot diff cannot during a
migration. Unpinning an address renumbers every later SSA value and lets the
emitter fold single-use addresses into their loads, so 55000 lines move and
almost none of it is a change in behaviour. Reviewing that by eye is how a real
change gets waved through in the middle of it.

So it expands every SSA name in every subscript down to leaves, canonicalises,
and compares the multiset of `(base, address)` pairs against a git revision.
Renumbering, parenthesisation and identity terms are canonicalised away;
associativity and distribution deliberately are not, because on an address
`a*(b+c)` and `a*b + c` usually differ for a reason.

A subscript it cannot parse raises rather than falling back to a text
comparison. The quiet alternative ends with the tool reporting "identical" over
accesses it stopped reading, and its entire value is that the answer licenses
not reading the diff.
