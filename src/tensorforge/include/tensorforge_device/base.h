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

namespace tensorforge {

// The suffixes the generator spells integer constants with.  Inline, so
// code inside `tensorforge` sees them as they are; a generated kernel
// opens with `using namespace tensorforge::literals;`.
inline namespace literals {

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

} // namespace literals

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
  // over data that happens to be entirely negative would return about 1e-38
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

  // All ones.  `max()` is `0x7fff...` on a signed type, so the sign bit would
  // come back cleared no matter what the data held.  Converting -1 sets every
  // bit at every width and signedness, and gives `true` for `bool`, where
  // `~T(0)` would promote to `int` first and warn.
  static constexpr T neutral() { return static_cast<T>(-1); }
};

template <typename T> struct ReductionOperation<T, Operation::Or> {
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

/// The greatest common divisor, for `coreGrid` (host code; `std::gcd` is not
/// constexpr in every library a device compiler brings).
constexpr std::size_t gcdOf(std::size_t a, std::size_t b) {
  while (b != 0) {
    const std::size_t r = a % b;
    a = b;
    b = r;
  }
  return a;
}

/// The work-groups of a grid-stride launch sized for `threads` hardware
/// threads an Xe-core (`Options.sycl_core_threads`): what `cores` Xe-cores
/// hold at that many threads, a work-group `groupThreads` of them, in whole
/// `parts` of the batch (`Options.batch_parts`, 0 or 1 for one) -- at least
/// one a part.  Where a traversal's items come in dependent blocks of
/// `block`, the rows a part strides by (`groupRows` a work-group, the
/// multiplications it holds) are kept coprime to it: a row takes items `r,
/// r + n, r + 2n, ...`, and with `n` sharing a factor with `block` the
/// block's dependent item lands on the same few rows every time (hvp on the
/// Max 1550: blocks of nine at 24 or 48 threads an Xe-core, every ninth
/// quantity on a third of the threads, 1.9x slower).  A block of one leaves
/// the grid as it is.
constexpr std::size_t coreGrid(std::size_t cores, std::size_t threads,
                               std::size_t groupThreads, std::size_t groupRows,
                               std::size_t parts = 1, std::size_t block = 1) {
  const std::size_t p = parts > 0 ? parts : 1;
  const std::size_t g = groupThreads > 0 ? groupThreads : 1;
  std::size_t perPart = cores * threads / g / p;
  if (perPart == 0) {
    perPart = 1;
  }
  while (perPart > 1 && block > 1 && gcdOf(perPart * groupRows, block) != 1) {
    --perPart;
  }
  return perPart * p;
}

} // namespace tensorforge
#endif // SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_BASE_H_
