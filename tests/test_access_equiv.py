# SPDX-License-Identifier: MIT
"""Can `access_equiv` tell two different addressings apart?

The tool's output is a licence to not read a 55000-line diff, which makes it
exactly the kind of check that is dangerous when it is too permissive. It
canonicalises three things away on purpose --- renumbering, parenthesisation,
identity terms --- and each of those is one step from canonicalising away a
real difference. A checker that says "identical" for everything reads as
verification and is worse than not checking.

So: for each thing it is meant to ignore, a pair it must call identical, and
next to it a pair it must call different.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# `tools/` is a directory of scripts, not a package, and making it one just to
# be importable here would change how every one of them is run.
_spec = importlib.util.spec_from_file_location(
    'access_equiv',
    Path(__file__).resolve().parents[1] / 'tools' / 'access_equiv.py')
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
accesses = _mod.accesses


def eq(a: str, b: str) -> bool:
    return accesses(a) == accesses(b)


# ---------------------------------------------------------------------- #
# what it must ignore
# ---------------------------------------------------------------------- #

def test_renumbering_is_ignored():
    """The pin removal shifts every later SSA number; nothing moved."""
    assert eq('float v9_data = r2[v8_a];',
              'float v11_data = r2[v10_a];')


def test_inlining_is_ignored():
    """`v8_a` folded into its single use is the same access, spelled shorter."""
    assert eq('int32_t v8_a = v6_i0 + v7_i1;\nfloat v9_data = r2[v8_a];',
              'float v9_data = r2[(v6_i0 + v7_i1)];')


def test_redundant_parentheses_are_ignored():
    assert eq('float v1_d = s0[((threadIdx.x % 32) * 56)];',
              'float v1_d = s0[(((threadIdx.x % 32)) * 56)];')


def test_identity_terms_are_ignored():
    """`0 + x` is what an address folds to once it is foldable."""
    assert eq('float v1_d = r2[0 + v5_i1];', 'float v1_d = r2[v5_i1];')
    assert eq('float v1_d = r2[1 * v5_i1];', 'float v1_d = r2[v5_i1];')


def test_definitions_expand_through_a_chain():
    assert eq('int32_t v1_a = v0_i * 4;\n'
              'int32_t v2_b = v1_a + 3;\n'
              'float v3_d = s0[v2_b];',
              'float v3_d = s0[v0_i * 4 + 3];')


# ---------------------------------------------------------------------- #
# what it must not ignore
# ---------------------------------------------------------------------- #

def test_a_different_stride_is_a_difference():
    assert not eq('float v1_d = s0[v0_i * 56];', 'float v1_d = s0[v0_i * 57];')


def test_a_different_base_is_a_difference():
    assert not eq('float v1_d = s0[v0_i];', 'float v1_d = s1[v0_i];')


def test_distribution_is_not_ignored():
    """`a*(b+c)` and `a*b + c` differ, and on an address they usually mean it."""
    assert not eq('float v1_d = s0[v0_a * (v2_b + v3_c)];',
                  'float v1_d = s0[v0_a * v2_b + v3_c];')


def test_a_dropped_access_is_a_difference():
    """The multiset counts: sixteen reads collapsing to one is a change."""
    assert not eq('float a = s0[v0_i];\nfloat b = s0[v0_i];',
                  'float a = s0[v0_i];')


def test_a_loop_variable_is_a_leaf():
    """Expanding it to its initial value would claim every iteration reads
    iteration zero --- which would make a genuine off-by-one invisible.

    It stays a leaf for two independent reasons: a loop header does not start
    with a type, and does not end in `;`. Breaking either alone changes
    nothing, which is why the matching mutation has to break both.
    """
    a = ('for (int32_t v0_i = 0; v0_i < 16; ++v0_i) {\n'
         '  float v1_d = s0[v0_i];\n}')
    b = ('for (int32_t v0_i = 0; v0_i < 16; ++v0_i) {\n'
         '  float v1_d = s0[0];\n}')
    assert not eq(a, b)
    # the same access with the loop bound changed is still the same access
    c = ('for (int32_t v0_i = 0; v0_i < 32; ++v0_i) {\n'
         '  float v1_d = s0[v0_i];\n}')
    assert eq(a, c)


def test_identity_folding_runs_to_fixpoint():
    """One pass leaves `0 + x` behind when it was nested."""
    assert eq('float v1_d = r2[0 + (0 + (0 + v5_i1))];',
              'float v1_d = r2[v5_i1];')


def test_a_typed_literal_is_not_a_parse_failure():
    """`32_i32` is a C++ literal and a Python syntax error; 115 appear in the
    corpus. The suffix carries no address information."""
    assert eq('float a = s0[(v0_i % 32) + 32_i32];',
              'float a = s0[(v0_i % 32) + 32];')


def test_an_unparseable_subscript_raises():
    """The tool's answer licenses not reading the diff, so a construct it
    cannot parse has to be loud. Falling back to text, or dropping the entry,
    would both end in a quiet "identical" over accesses it stopped reading."""
    import pytest
    with pytest.raises(ValueError, match='cannot parse the subscript'):
        accesses('float a = s0[p ? i : j];')


def test_renaming_is_positional_not_wholesale():
    """Two distinct names must not both canonicalise to the same one.

    Renaming by order of first appearance is what makes renumbering
    invisible; done wrong it would also make `s0[i] + s0[j]` and
    `s0[i] + s0[i]` agree.
    """
    assert not eq('float a = s0[v0_i];\nfloat b = s0[v1_j];',
                  'float a = s0[v0_i];\nfloat b = s0[v0_i];')
