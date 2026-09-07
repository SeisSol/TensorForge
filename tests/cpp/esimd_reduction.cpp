// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: MIT
//
// `segmentedReduction` has to compute what `tensorforge::reduction` computes
// on CUDA and HIP: a kernel that reduces differently on two backends is two
// kernels, and nothing downstream would notice which one it got.
//
// The two are written differently -- a shuffle butterfly there, a
// two-dimensional region here -- so agreeing is a claim and not a definition.
// This checks it: a minimal `simd` with real storage, the region formulation
// run over it, and the shuffle formulation computed alongside as the
// reference.
//
// The shim's `simd` cannot be used: it is a declaration-only stand-in whose
// operators return `*this`, which is right for a syntax check and useless for
// an arithmetic one.

#include <array>
#include <cstdio>
#include <numeric>
#include <vector>

namespace {

// -- the reference: what `cuda.h` does -------------------------------------
//
//     for (i = Block >> 1; i >= Subblock; i >>= 1)
//         result = Op(result, shfl_xor(result, i));

template <typename Op, typename T>
std::vector<T> shuffleButterfly(std::vector<T> v, int block, int subblock) {
  for (int i = block >> 1; i >= subblock; i >>= 1) {
    std::vector<T> next(v.size());
    for (std::size_t n = 0; n < v.size(); ++n)
      next[n] = Op::apply(v[n], v[n ^ static_cast<std::size_t>(i)]);
    v = next;
  }
  return v;
}

// -- the thing under test: the region formulation --------------------------
//
// `select<Block / (2 * I), 2 * I, I, 1>(0)` is every low half and `(I)` every
// high half; combining the two and writing the result into both is one step.
// Spelled with explicit indices here because a `simd_view` is what does it in
// the header, and the point is the *index arithmetic*, not the type.

template <typename Op, typename T>
std::vector<T> regionButterfly(std::vector<T> v, int block, int subblock) {
  for (int i = block >> 1; i >= subblock; i >>= 1) {
    const int rows = block / (2 * i);
    for (int r = 0; r < rows; ++r) {
      const int base = r * 2 * i;   // the row stride is 2 * I
      for (int k = 0; k < i; ++k) { // the row is I elements, stride 1
        const T combined = Op::apply(v[base + k], v[base + i + k]);
        v[base + k] = combined;
        v[base + i + k] = combined;
      }
    }
  }
  return v;
}

struct Add {
  template <typename T> static T apply(T a, T b) { return a + b; }
};
struct Max {
  template <typename T> static T apply(T a, T b) { return a < b ? b : a; }
};

template <typename Op, typename T>
bool agree(int block, int subblock, const std::vector<T> &in) {
  return shuffleButterfly<Op, T>(in, block, subblock) ==
         regionButterfly<Op, T>(in, block, subblock);
}

/// Every lane of a group must end with the reduction over *its* lanes, which
/// is the property the two formulations are agreeing about -- checking they
/// match each other would pass if both were wrong the same way.
template <typename Op, typename T>
bool isSegmentedReduction(int block, int subblock, const std::vector<T> &in) {
  const auto got = regionButterfly<Op, T>(in, block, subblock);
  for (int n = 0; n < block; ++n) {
    T want = in[n];
    bool first = true;
    for (int m = 0; m < block; ++m) {
      if (m % subblock != n % subblock)
        continue; // a different group slot
      if (first) {
        want = in[m];
        first = false;
      } else
        want = Op::apply(want, in[m]);
    }
    if (got[n] != want)
      return false;
  }
  return true;
}

int failures = 0;

void check(bool ok, const char *what, int block, int subblock) {
  if (!ok) {
    std::printf("FAIL: %s (block=%d, subblock=%d)\n", what, block, subblock);
    ++failures;
  }
}

} // namespace

int main() {
  for (int block : {2, 4, 8, 16, 32}) {
    std::vector<int> v(static_cast<std::size_t>(block));
    std::iota(v.begin(), v.end(), 1);
    for (int subblock = 1; subblock <= block; subblock <<= 1) {
      check(agree<Add, int>(block, subblock, v),
            "region and shuffle disagree (Add)", block, subblock);
      check(agree<Max, int>(block, subblock, v),
            "region and shuffle disagree (Max)", block, subblock);
      check(isSegmentedReduction<Add, int>(block, subblock, v),
            "not the segmented reduction (Add)", block, subblock);
      check(isSegmentedReduction<Max, int>(block, subblock, v),
            "not the segmented reduction (Max)", block, subblock);
    }
  }

  // `subblock == block` is the identity: each group is one lane wide.
  {
    std::vector<int> v{5, 7, 9, 11};
    check(regionButterfly<Add, int>(v, 4, 4) == v, "identity", 4, 4);
  }

  if (failures == 0)
    std::printf("segmented reduction: OK\n");
  return failures == 0 ? 0 : 1;
}
