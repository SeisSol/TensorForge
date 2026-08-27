// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: MIT
#ifndef SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_BASE_H_
#define SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_BASE_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

constexpr std::int8_t operator""_i8(unsigned long long value) {
  return static_cast<std::int8_t>(value);
}

constexpr std::int16_t operator""_i16(unsigned long long value) {
  return static_cast<std::int16_t>(value);
}

constexpr std::int32_t operator""_i32(unsigned long long value) {
  return static_cast<std::int32_t>(value);
}

constexpr std::int64_t operator""_i64(unsigned long long value) {
  return static_cast<std::int64_t>(value);
}

constexpr std::uint8_t operator""_u8(unsigned long long value) {
  return static_cast<std::uint8_t>(value);
}

constexpr std::uint16_t operator""_u16(unsigned long long value) {
  return static_cast<std::uint16_t>(value);
}

constexpr std::uint32_t operator""_u32(unsigned long long value) {
  return static_cast<std::uint32_t>(value);
}

constexpr std::uint64_t operator""_u64(unsigned long long value) {
  return static_cast<std::uint64_t>(value);
}

namespace tensorforge {

enum class Operation { Add, Mul, And, Or, Xor, Min, Max };

template <typename T, Operation OpT> struct ReductionOperation {
  static constexpr Operation Op = OpT;
  static constexpr T applyOperation(const T &a1, const T &a2);
  static constexpr T neutral();
};

template <typename T> struct ReductionOperation<T, Operation::Add> {
  static constexpr Operation Op = Operation::Add;
  static constexpr T applyOperation(const T &a1, const T &a2) {
    return a1 + a2;
  }

  static constexpr T neutral() { return T(0); }
};

template <typename T> struct ReductionOperation<T, Operation::Mul> {
  static constexpr Operation Op = Operation::Mul;
  static constexpr T applyOperation(const T &a1, const T &a2) {
    return a1 * a2;
  }

  static constexpr T neutral() { return T(1); }
};

template <typename T> struct ReductionOperation<T, Operation::Min> {
  static constexpr Operation Op = Operation::Min;
  static constexpr T applyOperation(const T &a1, const T &a2) {
    // Not `std::min`: it is a host function, and reaching it from device code
    // needs `--expt-relaxed-constexpr` under nvcc.  The comparison is the
    // whole of it anyway.
    return a2 < a1 ? a2 : a1;
  }

  // `max()` is the largest finite value for every arithmetic type, and for
  // floating point `infinity()` is a strictly better seed where it exists.
  static constexpr T neutral() {
    return std::numeric_limits<T>::has_infinity
               ? std::numeric_limits<T>::infinity()
               : std::numeric_limits<T>::max();
  }
};

template <typename T> struct ReductionOperation<T, Operation::Max> {
  static constexpr Operation Op = Operation::Max;
  static constexpr T applyOperation(const T &a1, const T &a2) {
    return a1 < a2 ? a2 : a1;
  }

  // `lowest()`, not `min()`.  For an integer type they agree; for floating
  // point `min()` is the smallest positive *normal* value, so a max-reduction
  // over data that happens to be entirely negative returned about 1e-38
  // instead of the largest element -- a plausible-looking number, which is the
  // worst kind.
  static constexpr T neutral() {
    return std::numeric_limits<T>::has_infinity
               ? -std::numeric_limits<T>::infinity()
               : std::numeric_limits<T>::lowest();
  }
};

template <typename T> struct ReductionOperation<T, Operation::And> {
  static constexpr Operation Op = Operation::And;
  static constexpr T applyOperation(const T &a1, const T &a2) {
    return a1 & a2;
  }

  // All ones.  `max()` is `0x7fff...` on a signed type, so the sign bit came
  // back cleared no matter what the data held.  Converting -1 sets every bit
  // at every width and signedness, and gives `true` for `bool`, where `~T(0)`
  // would promote to `int` first and warn.
  static constexpr T neutral() { return static_cast<T>(-1); }
};

template <typename T> struct ReductionOperation<T, Operation::Or> {
  // Was `Operation::And`, copied along with the rest of the specialisation.
  // Every `Op::Op == Operation::Or` test was therefore false, and an Or
  // reduction took whichever branch And had been given.
  static constexpr Operation Op = Operation::Or;
  static constexpr T applyOperation(const T &a1, const T &a2) {
    return a1 | a2;
  }

  static constexpr T neutral() { return T(0); }
};

template <typename T> struct ReductionOperation<T, Operation::Xor> {
  static constexpr Operation Op = Operation::Xor;
  static constexpr T applyOperation(const T &a1, const T &a2) {
    return a1 ^ a2;
  }

  // Xor's identity is 0 at every width: `x ^ 0 == x`.
  static constexpr T neutral() { return T(0); }
};

} // namespace tensorforge
#endif // SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_BASE_H_
