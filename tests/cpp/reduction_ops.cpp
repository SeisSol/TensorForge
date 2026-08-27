// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: MIT
//
// `base.h` is plain C++: no intrinsics, no execution-space keywords, nothing
// that needs a device. Every `ReductionOperation` member is `constexpr`, so
// the whole contract is checkable at compile time by a host compiler that is
// already installed for `test_syntax.py`.
//
// Nothing checked it, and three of the seven specialisations were wrong:
//
//   - `Max::neutral()` was `numeric_limits<T>::min()`. For an integer type
//     that is the lower bound and correct; for floating point it is the
//     smallest positive *normal* value, about 1e-38 for float. A max over
//     data that happened to be entirely negative therefore returned 1e-38.
//   - `And::neutral()` was `numeric_limits<T>::max()`, which on a signed type
//     is `0x7fff...`: the sign bit came back cleared whatever the data held.
//   - the `Or` specialisation tagged itself `Op = Operation::And`, so every
//     `Op::Op == Operation::Or` dispatch in `cuda.h` and `hip.h` was false.
//
// The identity law is the useful thing to assert rather than the literal
// values, because it is what the reduction actually relies on and it holds
// for every type at once.

#include "tensorforge_device/base.h"

#include <cstdint>
#include <limits>
#include <type_traits>

namespace {

using namespace tensorforge;

template <typename T, Operation Op> using RO = ReductionOperation<T, Op>;

// --- the tag matches the specialisation ---------------------------------- //
//
// A mismatch is invisible at the call site: dispatch silently takes another
// operator's branch and returns a number of the right type.

static_assert(RO<int, Operation::Add>::Op == Operation::Add, "Add tag");
static_assert(RO<int, Operation::Mul>::Op == Operation::Mul, "Mul tag");
static_assert(RO<int, Operation::Min>::Op == Operation::Min, "Min tag");
static_assert(RO<int, Operation::Max>::Op == Operation::Max, "Max tag");
static_assert(RO<int, Operation::And>::Op == Operation::And, "And tag");
static_assert(RO<int, Operation::Or>::Op == Operation::Or, "Or tag");
static_assert(RO<int, Operation::Xor>::Op == Operation::Xor, "Xor tag");

// --- op(neutral, x) == x ------------------------------------------------- //

template <typename T, Operation Op> constexpr bool isIdentity(T x) {
  return RO<T, Op>::applyOperation(RO<T, Op>::neutral(), x) == x &&
         RO<T, Op>::applyOperation(x, RO<T, Op>::neutral()) == x;
}

template <typename T> constexpr bool arithmeticIdentities() {
  return isIdentity<T, Operation::Add>(T(0)) &&
         isIdentity<T, Operation::Add>(T(7)) &&
         isIdentity<T, Operation::Mul>(T(1)) &&
         isIdentity<T, Operation::Mul>(T(7)) &&
         isIdentity<T, Operation::Min>(T(7)) &&
         isIdentity<T, Operation::Max>(T(7));
}

static_assert(arithmeticIdentities<float>(), "float identities");
static_assert(arithmeticIdentities<double>(), "double identities");
static_assert(arithmeticIdentities<std::int8_t>(), "int8 identities");
static_assert(arithmeticIdentities<std::int32_t>(), "int32 identities");
static_assert(arithmeticIdentities<std::int64_t>(), "int64 identities");
static_assert(arithmeticIdentities<std::uint32_t>(), "uint32 identities");

template <typename T> constexpr bool bitwiseIdentities() {
  return isIdentity<T, Operation::And>(T(0)) &&
         isIdentity<T, Operation::And>(T(0x5a)) &&
         isIdentity<T, Operation::And>(std::numeric_limits<T>::max()) &&
         isIdentity<T, Operation::And>(std::numeric_limits<T>::lowest()) &&
         isIdentity<T, Operation::Or>(T(0x5a)) &&
         isIdentity<T, Operation::Xor>(T(0x5a));
}

static_assert(bitwiseIdentities<std::int8_t>(), "int8 bitwise identities");
static_assert(bitwiseIdentities<std::int32_t>(), "int32 bitwise identities");
static_assert(bitwiseIdentities<std::int64_t>(), "int64 bitwise identities");
static_assert(bitwiseIdentities<std::uint8_t>(), "uint8 bitwise identities");
static_assert(bitwiseIdentities<std::uint32_t>(), "uint32 bitwise identities");

// The sign bit is the one `numeric_limits<T>::max()` dropped, so it gets its
// own assertion rather than relying on a value that happens to set it.
static_assert(RO<std::int32_t, Operation::And>::neutral() ==
                  static_cast<std::int32_t>(-1),
              "and neutral is all ones on a signed type");
static_assert(RO<std::uint64_t, Operation::And>::neutral() == ~std::uint64_t(0),
              "and neutral is all ones at 64 bits");
static_assert(RO<bool, Operation::And>::neutral(), "and neutral is true");

// --- the neutral element is a bound, not just an identity ---------------- //
//
// `op(neutral, x) == x` alone does not catch the float `min()`/`lowest()` mix
// up: 1e-38 is larger than most of the values a test would think to try. What
// separates them is whether the seed loses to *every* value the type holds.

template <typename T> constexpr bool maxNeutralIsALowerBound() {
  return RO<T, Operation::Max>::neutral() <= std::numeric_limits<T>::lowest();
}

template <typename T> constexpr bool minNeutralIsAnUpperBound() {
  return RO<T, Operation::Min>::neutral() >= std::numeric_limits<T>::max();
}

static_assert(maxNeutralIsALowerBound<float>(), "float max seed");
static_assert(maxNeutralIsALowerBound<double>(), "double max seed");
static_assert(maxNeutralIsALowerBound<std::int8_t>(), "int8 max seed");
static_assert(maxNeutralIsALowerBound<std::int32_t>(), "int32 max seed");
static_assert(maxNeutralIsALowerBound<std::uint32_t>(), "uint32 max seed");

static_assert(minNeutralIsAnUpperBound<float>(), "float min seed");
static_assert(minNeutralIsAnUpperBound<double>(), "double min seed");
static_assert(minNeutralIsAnUpperBound<std::int8_t>(), "int8 min seed");
static_assert(minNeutralIsAnUpperBound<std::int32_t>(), "int32 min seed");
static_assert(minNeutralIsAnUpperBound<std::uint32_t>(), "uint32 min seed");

// The concrete regression: a max-reduction over negative data.
constexpr float foldMax(const float *v, int n) {
  float acc = RO<float, Operation::Max>::neutral();
  for (int i = 0; i < n; ++i) {
    acc = RO<float, Operation::Max>::applyOperation(acc, v[i]);
  }
  return acc;
}

constexpr float AllNegative[] = {-3.0f, -1.0f, -7.0f, -2.5f};
static_assert(foldMax(AllNegative, 4) == -1.0f,
              "max over negative data returns the largest element");

// --- applyOperation agrees with the operator it names -------------------- //

static_assert(RO<int, Operation::Add>::applyOperation(3, 4) == 7, "add");
static_assert(RO<int, Operation::Mul>::applyOperation(3, 4) == 12, "mul");
static_assert(RO<int, Operation::Min>::applyOperation(3, 4) == 3, "min");
static_assert(RO<int, Operation::Max>::applyOperation(3, 4) == 4, "max");
static_assert(RO<int, Operation::And>::applyOperation(6, 3) == 2, "and");
static_assert(RO<int, Operation::Or>::applyOperation(6, 3) == 7, "or");
static_assert(RO<int, Operation::Xor>::applyOperation(6, 3) == 5, "xor");

// NaN is not asserted on: `min`/`max` here are a plain comparison, which
// propagates whichever operand the branch happens to select, and the device
// intrinsics the generated code uses do not agree with it. Pinning a
// behaviour neither side actually guarantees would be worse than saying
// nothing.

} // namespace

int main() { return 0; }
