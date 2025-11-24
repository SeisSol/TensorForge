#pragma once

#include <cmath>
#include <limits>
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
