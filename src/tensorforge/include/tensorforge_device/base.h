// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: MIT
#ifndef SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_BASE_H_
#define SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_BASE_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

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
    return std::min(a1, a2);
  }

  static constexpr T neutral() { return std::numeric_limits<T>::max(); }
};

template <typename T> struct ReductionOperation<T, Operation::Max> {
  static constexpr Operation Op = Operation::Max;
  static constexpr T applyOperation(const T &a1, const T &a2) {
    return std::max(a1, a2);
  }

  static constexpr T neutral() { return std::numeric_limits<T>::min(); }
};

template <typename T> struct ReductionOperation<T, Operation::And> {
  static constexpr Operation Op = Operation::And;
  static constexpr T applyOperation(const T &a1, const T &a2) {
    return a1 & a2;
  }

  static constexpr T neutral() { return std::numeric_limits<T>::max(); }
};

template <typename T> struct ReductionOperation<T, Operation::Or> {
  static constexpr Operation Op = Operation::And;
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

  static constexpr T neutral() {
    return T(0); // ?
  }
};

} // namespace tensorforge
#endif // SEISSOL_TENSORFORGE_INCLUDE_TENSORFORGE_DEVICE_BASE_H_
