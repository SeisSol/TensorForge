// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: MIT
//
// The SYCL shim is only worth anything if its types have the properties the
// generated code relies on.  A shim that compiles but models the wrong thing
// makes `test_syntax.py` green for the wrong reason -- which has happened
// here before, when GCC dropped a `vector_size` attribute off an alias
// template and every MFMA call then type-checked against plain `float`.
//
// So the properties are asserted, not assumed.  Everything here is a
// `static_assert`; a successful compile is the pass.

#include "../shim/tensorforge_sycl.h"

#include <type_traits>

// --------------------------------------------------------------------------
// `sycl::vec` is bit-cast through, never indexed.
//
// The generator emits
//     *(sycl::vec<float, 4>*)&s0[i] = *(sycl::vec<float, 4>*)&glb[j];
// which is a 16-byte copy *if and only if* the type is 16 bytes.  A shim
// whose `vec<float,4>` were, say, a single float would still compile that
// line and would silently be checking a quarter of the access.
// --------------------------------------------------------------------------

static_assert(sizeof(sycl::vec<float, 2>) == 2 * sizeof(float), "vec<float,2>");
static_assert(sizeof(sycl::vec<float, 4>) == 4 * sizeof(float), "vec<float,4>");
static_assert(sizeof(sycl::vec<double, 4>) == 4 * sizeof(double),
              "vec<double,4>");
static_assert(std::is_trivially_copyable<sycl::vec<float, 4>>::value,
              "vec must be bit-castable");

// --------------------------------------------------------------------------
// An accessor subscript is an lvalue.
//
// `float* localShrMem0 = &totalShrMem[272 * item.get_local_id(1)];` needs
// `operator[]` to return something addressable.  Returning by value compiles
// the declaration of `totalShrMem` and fails only at the address-of, which
// is a confusing place to learn it.
// --------------------------------------------------------------------------

static_assert(
    std::is_lvalue_reference<
        decltype(std::declval<sycl::local_accessor<float, 1>>()[0])>::value,
    "local_accessor[] must yield an lvalue");

static_assert(
    std::is_lvalue_reference<
        decltype(std::declval<
                 sycl::accessor<float, 1, sycl::access::mode::read_write,
                                sycl::access::target::local>>()[0])>::value,
    "accessor[] must yield an lvalue");

// --------------------------------------------------------------------------
// ESIMD: `simd<T,N>` holds N elements, and `select` is a writable view.
//
// Both are load-bearing.  `simd` sizing is what makes a register-pressure
// claim mean anything, and `select` has to be assignable because the
// broadcast path in `SyclLexic.broadcast` produces
// `x.select<Block, Sub>(lane)` on the *left* of an assignment.
// --------------------------------------------------------------------------

namespace esimd = tensorforge::intel_esimd;

static_assert(sizeof(esimd::simd<float, 16>) == 16 * sizeof(float),
              "simd<float,16>");
static_assert(sizeof(esimd::simd<float, 128>) == 128 * sizeof(float),
              "simd<float,128>");

static_assert(
    std::is_assignable<
        decltype(std::declval<esimd::simd<float, 16> &>().select<4, 1>(0)),
        float>::value,
    "select<>() must be assignable");

// TF32 is a distinct type, not an alias for float: `dpas` deduces its operand
// precision from the element type, so a typedef would make the TF32 and the
// direct-float paths indistinguishable to overload resolution.
static_assert(!std::is_same<tensorforge::TF32, float>::value,
              "TF32 must not be an alias for float");
static_assert(std::is_convertible<float, tensorforge::TF32>::value,
              "TF32 must be constructible from float");

int main() { return 0; }
