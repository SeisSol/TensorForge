// === base name ===
kernel_190be3eced4babcf

// === header ===
#ifndef TENSORFORGE_LAUNCH_TYPES
#define TENSORFORGE_LAUNCH_TYPES
#include <cstddef>
namespace tensorforge {
// Fixed when the kernel is generated: `launch_info_<kernel>`.
struct LaunchInfo {
  unsigned block[3];
  unsigned threadsPerMult;
  unsigned activeThreads;
  unsigned leadWidth;
  unsigned multsPerBlock;
  std::size_t sharedMemBytes;
  bool cooperative;
  bool persistent;
  unsigned sections;
};
// What one launch uses, the grid included: `launch_config_<kernel>`.
struct LaunchConfig {
  std::size_t grid[3];
  std::size_t block[3];
  std::size_t sharedMemBytes;
  bool cooperative;
};
} // namespace tensorforge
#endif
inline constexpr tensorforge::LaunchInfo launch_info_kernel_190be3eced4babcf = {{1, 16, 1}, 16, 12, 1, 16, 19456, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_190be3eced4babcf(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_190be3eced4babcf(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
#ifndef TENSORFORGE_LAUNCH_TYPES
#define TENSORFORGE_LAUNCH_TYPES
#include <cstddef>
namespace tensorforge {
// Fixed when the kernel is generated: `launch_info_<kernel>`.
struct LaunchInfo {
  unsigned block[3];
  unsigned threadsPerMult;
  unsigned activeThreads;
  unsigned leadWidth;
  unsigned multsPerBlock;
  std::size_t sharedMemBytes;
  bool cooperative;
  bool persistent;
  unsigned sections;
};
// What one launch uses, the grid included: `launch_config_<kernel>`.
struct LaunchConfig {
  std::size_t grid[3];
  std::size_t block[3];
  std::size_t sharedMemBytes;
  bool cooperative;
};
} // namespace tensorforge
#endif
tensorforge::LaunchConfig launch_config_kernel_190be3eced4babcf(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 16, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 16 - 1) / 16;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 4864 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_190be3eced4babcf(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_190be3eced4babcf(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_190be3eced4babcf(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_190be3eced4babcf(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<4864 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 19456 B shared, occupancy grid
        // operands:
        //   m0 32×32(12×12) {0..12}×{0..12} strided
        //   m1 32×32(12×12) {0..12}×{0..12} strided
        //   m2 32×32(12×12) {0..12}×{0..12} strided
        //   m3 32×32(4×12) {4..8}×{0..12} strided
        //   m4 32×32(12×12) {0..12}×{0..12} strided
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   t1[i,j] = t0[i,k] × m2[k,j]
        //   t0 12×12(12×12) {0..12}×{0..12} pointer_based({4..8}×{0..12}) = abs(N)
        //   m4[i,j] = t1[i,k] × t0[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":4864}],"shared_bytes":19456,"shared_elements":4864,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B","bbox":[[0,0],[12,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"E","bbox":[[0,0],[12,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"N","bbox":[[4,0],[8,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m4","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[4,0],[8,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[4,0],[8,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[12,12]},{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (304 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (288);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s2 = localShrMem0 + (144);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 144 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 144 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v7_batchId0 * 48 + 0 + m3_extraOffset];
              float *const __restrict__ glb_m4 = &m4[v7_batchId0 * 144 + 0 + m4_extraOffset];
              tensorforge::intel_esimd::simd<float, 192> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v21_i1 = 0; v21_i1 < 12; ++v21_i1) {
                tensorforge::intel_esimd::simd<float, 12> v26_data;
                v26_data.copy_from(glb_m0 + ((v21_i1 * 12)));
                r0.template select<12, 1>((v21_i1 * 16)) = v26_data;
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v29_ld;
              v29_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v29_ld);
              tensorforge::intel_esimd::simd<float, 64> v30_ld;
              v30_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v30_ld);
              tensorforge::intel_esimd::simd<float, 16> v31_ld;
              v31_ld.copy_from(glb_m1 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v31_ld);
              // wait(r0 = load{g>r}(glb_m0););
              // s2 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v32_ld;
              v32_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s2 + (0 + 0 + 4 * 0 + 0), v32_ld);
              tensorforge::intel_esimd::simd<float, 64> v33_ld;
              v33_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s2 + (0 + 0 + 4 * 0 + 64), v33_ld);
              tensorforge::intel_esimd::simd<float, 16> v34_ld;
              v34_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s2 + (0 + 0 + 1 * 0 + 128), v34_ld);
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 192> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 12), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 16> v36_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v37_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v38_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v39_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v40_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v48_acc{};
              tensorforge::intel_esimd::simd<float, 16> v52_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v48_acc += ((static_cast<float>(v52_data[0])) * v36_data);
              v48_acc += ((static_cast<float>(v52_data[1])) * v37_data);
              v48_acc += ((static_cast<float>(v52_data[2])) * v38_data);
              v48_acc += ((static_cast<float>(v52_data[3])) * v39_data);
              v48_acc += ((static_cast<float>(v52_data[4])) * v40_data);
              v48_acc += ((static_cast<float>(v52_data[5])) * v41_data);
              v48_acc += ((static_cast<float>(v52_data[6])) * v42_data);
              v48_acc += ((static_cast<float>(v52_data[7])) * v43_data);
              v48_acc += ((static_cast<float>(v52_data[8])) * v44_data);
              v48_acc += ((static_cast<float>(v52_data[9])) * v45_data);
              v48_acc += ((static_cast<float>(v52_data[10])) * v46_data);
              v48_acc += ((static_cast<float>(v52_data[11])) * v47_data);
              r1.template select<16, 1>(0) = v48_acc;
              tensorforge::intel_esimd::simd<float, 16> v77_acc{};
              tensorforge::intel_esimd::simd<float, 16> v79_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              v77_acc += ((static_cast<float>(v79_data[0])) * v36_data);
              v77_acc += ((static_cast<float>(v79_data[1])) * v37_data);
              v77_acc += ((static_cast<float>(v79_data[2])) * v38_data);
              v77_acc += ((static_cast<float>(v79_data[3])) * v39_data);
              v77_acc += ((static_cast<float>(v79_data[4])) * v40_data);
              v77_acc += ((static_cast<float>(v79_data[5])) * v41_data);
              v77_acc += ((static_cast<float>(v79_data[6])) * v42_data);
              v77_acc += ((static_cast<float>(v79_data[7])) * v43_data);
              v77_acc += ((static_cast<float>(v79_data[8])) * v44_data);
              v77_acc += ((static_cast<float>(v79_data[9])) * v45_data);
              v77_acc += ((static_cast<float>(v79_data[10])) * v46_data);
              v77_acc += ((static_cast<float>(v79_data[11])) * v47_data);
              r1.template select<16, 1>(16) = v77_acc;
              tensorforge::intel_esimd::simd<float, 16> v104_acc{};
              tensorforge::intel_esimd::simd<float, 16> v106_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              v104_acc += ((static_cast<float>(v106_data[0])) * v36_data);
              v104_acc += ((static_cast<float>(v106_data[1])) * v37_data);
              v104_acc += ((static_cast<float>(v106_data[2])) * v38_data);
              v104_acc += ((static_cast<float>(v106_data[3])) * v39_data);
              v104_acc += ((static_cast<float>(v106_data[4])) * v40_data);
              v104_acc += ((static_cast<float>(v106_data[5])) * v41_data);
              v104_acc += ((static_cast<float>(v106_data[6])) * v42_data);
              v104_acc += ((static_cast<float>(v106_data[7])) * v43_data);
              v104_acc += ((static_cast<float>(v106_data[8])) * v44_data);
              v104_acc += ((static_cast<float>(v106_data[9])) * v45_data);
              v104_acc += ((static_cast<float>(v106_data[10])) * v46_data);
              v104_acc += ((static_cast<float>(v106_data[11])) * v47_data);
              r1.template select<16, 1>(32) = v104_acc;
              tensorforge::intel_esimd::simd<float, 16> v131_acc{};
              tensorforge::intel_esimd::simd<float, 16> v133_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              v131_acc += ((static_cast<float>(v133_data[0])) * v36_data);
              v131_acc += ((static_cast<float>(v133_data[1])) * v37_data);
              v131_acc += ((static_cast<float>(v133_data[2])) * v38_data);
              v131_acc += ((static_cast<float>(v133_data[3])) * v39_data);
              v131_acc += ((static_cast<float>(v133_data[4])) * v40_data);
              v131_acc += ((static_cast<float>(v133_data[5])) * v41_data);
              v131_acc += ((static_cast<float>(v133_data[6])) * v42_data);
              v131_acc += ((static_cast<float>(v133_data[7])) * v43_data);
              v131_acc += ((static_cast<float>(v133_data[8])) * v44_data);
              v131_acc += ((static_cast<float>(v133_data[9])) * v45_data);
              v131_acc += ((static_cast<float>(v133_data[10])) * v46_data);
              v131_acc += ((static_cast<float>(v133_data[11])) * v47_data);
              r1.template select<16, 1>(48) = v131_acc;
              tensorforge::intel_esimd::simd<float, 16> v158_acc{};
              tensorforge::intel_esimd::simd<float, 16> v160_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v158_acc += ((static_cast<float>(v160_data[0])) * v36_data);
              v158_acc += ((static_cast<float>(v160_data[1])) * v37_data);
              v158_acc += ((static_cast<float>(v160_data[2])) * v38_data);
              v158_acc += ((static_cast<float>(v160_data[3])) * v39_data);
              v158_acc += ((static_cast<float>(v160_data[4])) * v40_data);
              v158_acc += ((static_cast<float>(v160_data[5])) * v41_data);
              v158_acc += ((static_cast<float>(v160_data[6])) * v42_data);
              v158_acc += ((static_cast<float>(v160_data[7])) * v43_data);
              v158_acc += ((static_cast<float>(v160_data[8])) * v44_data);
              v158_acc += ((static_cast<float>(v160_data[9])) * v45_data);
              v158_acc += ((static_cast<float>(v160_data[10])) * v46_data);
              v158_acc += ((static_cast<float>(v160_data[11])) * v47_data);
              r1.template select<16, 1>(64) = v158_acc;
              tensorforge::intel_esimd::simd<float, 16> v185_acc{};
              tensorforge::intel_esimd::simd<float, 16> v187_data = tensorforge::slmLoad<float, 16>(s0 + (60_i32));
              v185_acc += ((static_cast<float>(v187_data[0])) * v36_data);
              v185_acc += ((static_cast<float>(v187_data[1])) * v37_data);
              v185_acc += ((static_cast<float>(v187_data[2])) * v38_data);
              v185_acc += ((static_cast<float>(v187_data[3])) * v39_data);
              v185_acc += ((static_cast<float>(v187_data[4])) * v40_data);
              v185_acc += ((static_cast<float>(v187_data[5])) * v41_data);
              v185_acc += ((static_cast<float>(v187_data[6])) * v42_data);
              v185_acc += ((static_cast<float>(v187_data[7])) * v43_data);
              v185_acc += ((static_cast<float>(v187_data[8])) * v44_data);
              v185_acc += ((static_cast<float>(v187_data[9])) * v45_data);
              v185_acc += ((static_cast<float>(v187_data[10])) * v46_data);
              v185_acc += ((static_cast<float>(v187_data[11])) * v47_data);
              r1.template select<16, 1>(80) = v185_acc;
              tensorforge::intel_esimd::simd<float, 16> v212_acc{};
              tensorforge::intel_esimd::simd<float, 16> v214_data = tensorforge::slmLoad<float, 16>(s0 + (72_i32));
              v212_acc += ((static_cast<float>(v214_data[0])) * v36_data);
              v212_acc += ((static_cast<float>(v214_data[1])) * v37_data);
              v212_acc += ((static_cast<float>(v214_data[2])) * v38_data);
              v212_acc += ((static_cast<float>(v214_data[3])) * v39_data);
              v212_acc += ((static_cast<float>(v214_data[4])) * v40_data);
              v212_acc += ((static_cast<float>(v214_data[5])) * v41_data);
              v212_acc += ((static_cast<float>(v214_data[6])) * v42_data);
              v212_acc += ((static_cast<float>(v214_data[7])) * v43_data);
              v212_acc += ((static_cast<float>(v214_data[8])) * v44_data);
              v212_acc += ((static_cast<float>(v214_data[9])) * v45_data);
              v212_acc += ((static_cast<float>(v214_data[10])) * v46_data);
              v212_acc += ((static_cast<float>(v214_data[11])) * v47_data);
              r1.template select<16, 1>(96) = v212_acc;
              tensorforge::intel_esimd::simd<float, 16> v239_acc{};
              tensorforge::intel_esimd::simd<float, 16> v241_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v239_acc += ((static_cast<float>(v241_data[0])) * v36_data);
              v239_acc += ((static_cast<float>(v241_data[1])) * v37_data);
              v239_acc += ((static_cast<float>(v241_data[2])) * v38_data);
              v239_acc += ((static_cast<float>(v241_data[3])) * v39_data);
              v239_acc += ((static_cast<float>(v241_data[4])) * v40_data);
              v239_acc += ((static_cast<float>(v241_data[5])) * v41_data);
              v239_acc += ((static_cast<float>(v241_data[6])) * v42_data);
              v239_acc += ((static_cast<float>(v241_data[7])) * v43_data);
              v239_acc += ((static_cast<float>(v241_data[8])) * v44_data);
              v239_acc += ((static_cast<float>(v241_data[9])) * v45_data);
              v239_acc += ((static_cast<float>(v241_data[10])) * v46_data);
              v239_acc += ((static_cast<float>(v241_data[11])) * v47_data);
              r1.template select<16, 1>(112) = v239_acc;
              tensorforge::intel_esimd::simd<float, 16> v266_acc{};
              tensorforge::intel_esimd::simd<float, 16> v268_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              v266_acc += ((static_cast<float>(v268_data[0])) * v36_data);
              v266_acc += ((static_cast<float>(v268_data[1])) * v37_data);
              v266_acc += ((static_cast<float>(v268_data[2])) * v38_data);
              v266_acc += ((static_cast<float>(v268_data[3])) * v39_data);
              v266_acc += ((static_cast<float>(v268_data[4])) * v40_data);
              v266_acc += ((static_cast<float>(v268_data[5])) * v41_data);
              v266_acc += ((static_cast<float>(v268_data[6])) * v42_data);
              v266_acc += ((static_cast<float>(v268_data[7])) * v43_data);
              v266_acc += ((static_cast<float>(v268_data[8])) * v44_data);
              v266_acc += ((static_cast<float>(v268_data[9])) * v45_data);
              v266_acc += ((static_cast<float>(v268_data[10])) * v46_data);
              v266_acc += ((static_cast<float>(v268_data[11])) * v47_data);
              r1.template select<16, 1>(128) = v266_acc;
              tensorforge::intel_esimd::simd<float, 16> v293_acc{};
              tensorforge::intel_esimd::simd<float, 16> v295_data = tensorforge::slmLoad<float, 16>(s0 + (108_i32));
              v293_acc += ((static_cast<float>(v295_data[0])) * v36_data);
              v293_acc += ((static_cast<float>(v295_data[1])) * v37_data);
              v293_acc += ((static_cast<float>(v295_data[2])) * v38_data);
              v293_acc += ((static_cast<float>(v295_data[3])) * v39_data);
              v293_acc += ((static_cast<float>(v295_data[4])) * v40_data);
              v293_acc += ((static_cast<float>(v295_data[5])) * v41_data);
              v293_acc += ((static_cast<float>(v295_data[6])) * v42_data);
              v293_acc += ((static_cast<float>(v295_data[7])) * v43_data);
              v293_acc += ((static_cast<float>(v295_data[8])) * v44_data);
              v293_acc += ((static_cast<float>(v295_data[9])) * v45_data);
              v293_acc += ((static_cast<float>(v295_data[10])) * v46_data);
              v293_acc += ((static_cast<float>(v295_data[11])) * v47_data);
              r1.template select<16, 1>(144) = v293_acc;
              tensorforge::intel_esimd::simd<float, 16> v320_acc{};
              tensorforge::intel_esimd::simd<float, 16> v322_data = tensorforge::slmLoad<float, 16>(s0 + (120_i32));
              v320_acc += ((static_cast<float>(v322_data[0])) * v36_data);
              v320_acc += ((static_cast<float>(v322_data[1])) * v37_data);
              v320_acc += ((static_cast<float>(v322_data[2])) * v38_data);
              v320_acc += ((static_cast<float>(v322_data[3])) * v39_data);
              v320_acc += ((static_cast<float>(v322_data[4])) * v40_data);
              v320_acc += ((static_cast<float>(v322_data[5])) * v41_data);
              v320_acc += ((static_cast<float>(v322_data[6])) * v42_data);
              v320_acc += ((static_cast<float>(v322_data[7])) * v43_data);
              v320_acc += ((static_cast<float>(v322_data[8])) * v44_data);
              v320_acc += ((static_cast<float>(v322_data[9])) * v45_data);
              v320_acc += ((static_cast<float>(v322_data[10])) * v46_data);
              v320_acc += ((static_cast<float>(v322_data[11])) * v47_data);
              r1.template select<16, 1>(160) = v320_acc;
              tensorforge::intel_esimd::simd<float, 16> v347_acc{};
              tensorforge::intel_esimd::simd<float, 16> v349_data = tensorforge::slmLoad<float, 16>(s0 + (132_i32));
              v347_acc += ((static_cast<float>(v349_data[0])) * v36_data);
              v347_acc += ((static_cast<float>(v349_data[1])) * v37_data);
              v347_acc += ((static_cast<float>(v349_data[2])) * v38_data);
              v347_acc += ((static_cast<float>(v349_data[3])) * v39_data);
              v347_acc += ((static_cast<float>(v349_data[4])) * v40_data);
              v347_acc += ((static_cast<float>(v349_data[5])) * v41_data);
              v347_acc += ((static_cast<float>(v349_data[6])) * v42_data);
              v347_acc += ((static_cast<float>(v349_data[7])) * v43_data);
              v347_acc += ((static_cast<float>(v349_data[8])) * v44_data);
              v347_acc += ((static_cast<float>(v349_data[9])) * v45_data);
              v347_acc += ((static_cast<float>(v349_data[10])) * v46_data);
              v347_acc += ((static_cast<float>(v349_data[11])) * v47_data);
              r1.template select<16, 1>(176) = v347_acc;
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v374_i1 = 0; v374_i1 < 12; ++v374_i1) {
                tensorforge::intel_esimd::simd<float, 12> v377_data(r1.template select<12, 1>((v374_i1 * 16)));
                tensorforge::slmStore<float, 12>(s1 + ((v374_i1 * 12)), v377_data);
              }
              // wait(s2 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 192> r2(0.0f);
              // r2 = +(s1 * s2) + None
              // [(0, 12), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 16> v386_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v388_data = tensorforge::slmLoad<float, 16>(s1 + (12_i32));
              tensorforge::intel_esimd::simd<float, 16> v390_data = tensorforge::slmLoad<float, 16>(s1 + (24_i32));
              tensorforge::intel_esimd::simd<float, 16> v392_data = tensorforge::slmLoad<float, 16>(s1 + (36_i32));
              tensorforge::intel_esimd::simd<float, 16> v394_data = tensorforge::slmLoad<float, 16>(s1 + (48_i32));
              tensorforge::intel_esimd::simd<float, 16> v396_data = tensorforge::slmLoad<float, 16>(s1 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v398_data = tensorforge::slmLoad<float, 16>(s1 + (72_i32));
              tensorforge::intel_esimd::simd<float, 16> v400_data = tensorforge::slmLoad<float, 16>(s1 + (84_i32));
              tensorforge::intel_esimd::simd<float, 16> v402_data = tensorforge::slmLoad<float, 16>(s1 + (96_i32));
              tensorforge::intel_esimd::simd<float, 16> v404_data = tensorforge::slmLoad<float, 16>(s1 + (108_i32));
              tensorforge::intel_esimd::simd<float, 16> v406_data = tensorforge::slmLoad<float, 16>(s1 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v408_data = tensorforge::slmLoad<float, 16>(s1 + (132_i32));
              tensorforge::intel_esimd::simd<float, 16> v409_acc{};
              tensorforge::intel_esimd::simd<float, 16> v410_data = tensorforge::slmLoad<float, 16>(s2 + (0_i32));
              v409_acc += ((static_cast<float>(v410_data[0])) * v386_data);
              v409_acc += ((static_cast<float>(v410_data[1])) * v388_data);
              v409_acc += ((static_cast<float>(v410_data[2])) * v390_data);
              v409_acc += ((static_cast<float>(v410_data[3])) * v392_data);
              v409_acc += ((static_cast<float>(v410_data[4])) * v394_data);
              v409_acc += ((static_cast<float>(v410_data[5])) * v396_data);
              v409_acc += ((static_cast<float>(v410_data[6])) * v398_data);
              v409_acc += ((static_cast<float>(v410_data[7])) * v400_data);
              v409_acc += ((static_cast<float>(v410_data[8])) * v402_data);
              v409_acc += ((static_cast<float>(v410_data[9])) * v404_data);
              v409_acc += ((static_cast<float>(v410_data[10])) * v406_data);
              v409_acc += ((static_cast<float>(v410_data[11])) * v408_data);
              r2.template select<16, 1>(0) = v409_acc;
              tensorforge::intel_esimd::simd<float, 16> v435_acc{};
              tensorforge::intel_esimd::simd<float, 16> v436_data = tensorforge::slmLoad<float, 16>(s2 + (12_i32));
              v435_acc += ((static_cast<float>(v436_data[0])) * v386_data);
              v435_acc += ((static_cast<float>(v436_data[1])) * v388_data);
              v435_acc += ((static_cast<float>(v436_data[2])) * v390_data);
              v435_acc += ((static_cast<float>(v436_data[3])) * v392_data);
              v435_acc += ((static_cast<float>(v436_data[4])) * v394_data);
              v435_acc += ((static_cast<float>(v436_data[5])) * v396_data);
              v435_acc += ((static_cast<float>(v436_data[6])) * v398_data);
              v435_acc += ((static_cast<float>(v436_data[7])) * v400_data);
              v435_acc += ((static_cast<float>(v436_data[8])) * v402_data);
              v435_acc += ((static_cast<float>(v436_data[9])) * v404_data);
              v435_acc += ((static_cast<float>(v436_data[10])) * v406_data);
              v435_acc += ((static_cast<float>(v436_data[11])) * v408_data);
              r2.template select<16, 1>(16) = v435_acc;
              tensorforge::intel_esimd::simd<float, 16> v461_acc{};
              tensorforge::intel_esimd::simd<float, 16> v462_data = tensorforge::slmLoad<float, 16>(s2 + (24_i32));
              v461_acc += ((static_cast<float>(v462_data[0])) * v386_data);
              v461_acc += ((static_cast<float>(v462_data[1])) * v388_data);
              v461_acc += ((static_cast<float>(v462_data[2])) * v390_data);
              v461_acc += ((static_cast<float>(v462_data[3])) * v392_data);
              v461_acc += ((static_cast<float>(v462_data[4])) * v394_data);
              v461_acc += ((static_cast<float>(v462_data[5])) * v396_data);
              v461_acc += ((static_cast<float>(v462_data[6])) * v398_data);
              v461_acc += ((static_cast<float>(v462_data[7])) * v400_data);
              v461_acc += ((static_cast<float>(v462_data[8])) * v402_data);
              v461_acc += ((static_cast<float>(v462_data[9])) * v404_data);
              v461_acc += ((static_cast<float>(v462_data[10])) * v406_data);
              v461_acc += ((static_cast<float>(v462_data[11])) * v408_data);
              r2.template select<16, 1>(32) = v461_acc;
              tensorforge::intel_esimd::simd<float, 16> v487_acc{};
              tensorforge::intel_esimd::simd<float, 16> v488_data = tensorforge::slmLoad<float, 16>(s2 + (36_i32));
              v487_acc += ((static_cast<float>(v488_data[0])) * v386_data);
              v487_acc += ((static_cast<float>(v488_data[1])) * v388_data);
              v487_acc += ((static_cast<float>(v488_data[2])) * v390_data);
              v487_acc += ((static_cast<float>(v488_data[3])) * v392_data);
              v487_acc += ((static_cast<float>(v488_data[4])) * v394_data);
              v487_acc += ((static_cast<float>(v488_data[5])) * v396_data);
              v487_acc += ((static_cast<float>(v488_data[6])) * v398_data);
              v487_acc += ((static_cast<float>(v488_data[7])) * v400_data);
              v487_acc += ((static_cast<float>(v488_data[8])) * v402_data);
              v487_acc += ((static_cast<float>(v488_data[9])) * v404_data);
              v487_acc += ((static_cast<float>(v488_data[10])) * v406_data);
              v487_acc += ((static_cast<float>(v488_data[11])) * v408_data);
              r2.template select<16, 1>(48) = v487_acc;
              tensorforge::intel_esimd::simd<float, 16> v513_acc{};
              tensorforge::intel_esimd::simd<float, 16> v514_data = tensorforge::slmLoad<float, 16>(s2 + (48_i32));
              v513_acc += ((static_cast<float>(v514_data[0])) * v386_data);
              v513_acc += ((static_cast<float>(v514_data[1])) * v388_data);
              v513_acc += ((static_cast<float>(v514_data[2])) * v390_data);
              v513_acc += ((static_cast<float>(v514_data[3])) * v392_data);
              v513_acc += ((static_cast<float>(v514_data[4])) * v394_data);
              v513_acc += ((static_cast<float>(v514_data[5])) * v396_data);
              v513_acc += ((static_cast<float>(v514_data[6])) * v398_data);
              v513_acc += ((static_cast<float>(v514_data[7])) * v400_data);
              v513_acc += ((static_cast<float>(v514_data[8])) * v402_data);
              v513_acc += ((static_cast<float>(v514_data[9])) * v404_data);
              v513_acc += ((static_cast<float>(v514_data[10])) * v406_data);
              v513_acc += ((static_cast<float>(v514_data[11])) * v408_data);
              r2.template select<16, 1>(64) = v513_acc;
              tensorforge::intel_esimd::simd<float, 16> v539_acc{};
              tensorforge::intel_esimd::simd<float, 16> v540_data = tensorforge::slmLoad<float, 16>(s2 + (60_i32));
              v539_acc += ((static_cast<float>(v540_data[0])) * v386_data);
              v539_acc += ((static_cast<float>(v540_data[1])) * v388_data);
              v539_acc += ((static_cast<float>(v540_data[2])) * v390_data);
              v539_acc += ((static_cast<float>(v540_data[3])) * v392_data);
              v539_acc += ((static_cast<float>(v540_data[4])) * v394_data);
              v539_acc += ((static_cast<float>(v540_data[5])) * v396_data);
              v539_acc += ((static_cast<float>(v540_data[6])) * v398_data);
              v539_acc += ((static_cast<float>(v540_data[7])) * v400_data);
              v539_acc += ((static_cast<float>(v540_data[8])) * v402_data);
              v539_acc += ((static_cast<float>(v540_data[9])) * v404_data);
              v539_acc += ((static_cast<float>(v540_data[10])) * v406_data);
              v539_acc += ((static_cast<float>(v540_data[11])) * v408_data);
              r2.template select<16, 1>(80) = v539_acc;
              tensorforge::intel_esimd::simd<float, 16> v565_acc{};
              tensorforge::intel_esimd::simd<float, 16> v566_data = tensorforge::slmLoad<float, 16>(s2 + (72_i32));
              v565_acc += ((static_cast<float>(v566_data[0])) * v386_data);
              v565_acc += ((static_cast<float>(v566_data[1])) * v388_data);
              v565_acc += ((static_cast<float>(v566_data[2])) * v390_data);
              v565_acc += ((static_cast<float>(v566_data[3])) * v392_data);
              v565_acc += ((static_cast<float>(v566_data[4])) * v394_data);
              v565_acc += ((static_cast<float>(v566_data[5])) * v396_data);
              v565_acc += ((static_cast<float>(v566_data[6])) * v398_data);
              v565_acc += ((static_cast<float>(v566_data[7])) * v400_data);
              v565_acc += ((static_cast<float>(v566_data[8])) * v402_data);
              v565_acc += ((static_cast<float>(v566_data[9])) * v404_data);
              v565_acc += ((static_cast<float>(v566_data[10])) * v406_data);
              v565_acc += ((static_cast<float>(v566_data[11])) * v408_data);
              r2.template select<16, 1>(96) = v565_acc;
              tensorforge::intel_esimd::simd<float, 16> v591_acc{};
              tensorforge::intel_esimd::simd<float, 16> v592_data = tensorforge::slmLoad<float, 16>(s2 + (84_i32));
              v591_acc += ((static_cast<float>(v592_data[0])) * v386_data);
              v591_acc += ((static_cast<float>(v592_data[1])) * v388_data);
              v591_acc += ((static_cast<float>(v592_data[2])) * v390_data);
              v591_acc += ((static_cast<float>(v592_data[3])) * v392_data);
              v591_acc += ((static_cast<float>(v592_data[4])) * v394_data);
              v591_acc += ((static_cast<float>(v592_data[5])) * v396_data);
              v591_acc += ((static_cast<float>(v592_data[6])) * v398_data);
              v591_acc += ((static_cast<float>(v592_data[7])) * v400_data);
              v591_acc += ((static_cast<float>(v592_data[8])) * v402_data);
              v591_acc += ((static_cast<float>(v592_data[9])) * v404_data);
              v591_acc += ((static_cast<float>(v592_data[10])) * v406_data);
              v591_acc += ((static_cast<float>(v592_data[11])) * v408_data);
              r2.template select<16, 1>(112) = v591_acc;
              tensorforge::intel_esimd::simd<float, 16> v617_acc{};
              tensorforge::intel_esimd::simd<float, 16> v618_data = tensorforge::slmLoad<float, 16>(s2 + (96_i32));
              v617_acc += ((static_cast<float>(v618_data[0])) * v386_data);
              v617_acc += ((static_cast<float>(v618_data[1])) * v388_data);
              v617_acc += ((static_cast<float>(v618_data[2])) * v390_data);
              v617_acc += ((static_cast<float>(v618_data[3])) * v392_data);
              v617_acc += ((static_cast<float>(v618_data[4])) * v394_data);
              v617_acc += ((static_cast<float>(v618_data[5])) * v396_data);
              v617_acc += ((static_cast<float>(v618_data[6])) * v398_data);
              v617_acc += ((static_cast<float>(v618_data[7])) * v400_data);
              v617_acc += ((static_cast<float>(v618_data[8])) * v402_data);
              v617_acc += ((static_cast<float>(v618_data[9])) * v404_data);
              v617_acc += ((static_cast<float>(v618_data[10])) * v406_data);
              v617_acc += ((static_cast<float>(v618_data[11])) * v408_data);
              r2.template select<16, 1>(128) = v617_acc;
              tensorforge::intel_esimd::simd<float, 16> v643_acc{};
              tensorforge::intel_esimd::simd<float, 16> v644_data = tensorforge::slmLoad<float, 16>(s2 + (108_i32));
              v643_acc += ((static_cast<float>(v644_data[0])) * v386_data);
              v643_acc += ((static_cast<float>(v644_data[1])) * v388_data);
              v643_acc += ((static_cast<float>(v644_data[2])) * v390_data);
              v643_acc += ((static_cast<float>(v644_data[3])) * v392_data);
              v643_acc += ((static_cast<float>(v644_data[4])) * v394_data);
              v643_acc += ((static_cast<float>(v644_data[5])) * v396_data);
              v643_acc += ((static_cast<float>(v644_data[6])) * v398_data);
              v643_acc += ((static_cast<float>(v644_data[7])) * v400_data);
              v643_acc += ((static_cast<float>(v644_data[8])) * v402_data);
              v643_acc += ((static_cast<float>(v644_data[9])) * v404_data);
              v643_acc += ((static_cast<float>(v644_data[10])) * v406_data);
              v643_acc += ((static_cast<float>(v644_data[11])) * v408_data);
              r2.template select<16, 1>(144) = v643_acc;
              tensorforge::intel_esimd::simd<float, 16> v669_acc{};
              tensorforge::intel_esimd::simd<float, 16> v670_data = tensorforge::slmLoad<float, 16>(s2 + (120_i32));
              v669_acc += ((static_cast<float>(v670_data[0])) * v386_data);
              v669_acc += ((static_cast<float>(v670_data[1])) * v388_data);
              v669_acc += ((static_cast<float>(v670_data[2])) * v390_data);
              v669_acc += ((static_cast<float>(v670_data[3])) * v392_data);
              v669_acc += ((static_cast<float>(v670_data[4])) * v394_data);
              v669_acc += ((static_cast<float>(v670_data[5])) * v396_data);
              v669_acc += ((static_cast<float>(v670_data[6])) * v398_data);
              v669_acc += ((static_cast<float>(v670_data[7])) * v400_data);
              v669_acc += ((static_cast<float>(v670_data[8])) * v402_data);
              v669_acc += ((static_cast<float>(v670_data[9])) * v404_data);
              v669_acc += ((static_cast<float>(v670_data[10])) * v406_data);
              v669_acc += ((static_cast<float>(v670_data[11])) * v408_data);
              r2.template select<16, 1>(160) = v669_acc;
              tensorforge::intel_esimd::simd<float, 16> v695_acc{};
              tensorforge::intel_esimd::simd<float, 16> v696_data = tensorforge::slmLoad<float, 16>(s2 + (132_i32));
              v695_acc += ((static_cast<float>(v696_data[0])) * v386_data);
              v695_acc += ((static_cast<float>(v696_data[1])) * v388_data);
              v695_acc += ((static_cast<float>(v696_data[2])) * v390_data);
              v695_acc += ((static_cast<float>(v696_data[3])) * v392_data);
              v695_acc += ((static_cast<float>(v696_data[4])) * v394_data);
              v695_acc += ((static_cast<float>(v696_data[5])) * v396_data);
              v695_acc += ((static_cast<float>(v696_data[6])) * v398_data);
              v695_acc += ((static_cast<float>(v696_data[7])) * v400_data);
              v695_acc += ((static_cast<float>(v696_data[8])) * v402_data);
              v695_acc += ((static_cast<float>(v696_data[9])) * v404_data);
              v695_acc += ((static_cast<float>(v696_data[10])) * v406_data);
              v695_acc += ((static_cast<float>(v696_data[11])) * v408_data);
              r2.template select<16, 1>(176) = v695_acc;
              tensorforge::intel_esimd::simd<float, 192> r3(0.0f);
              // r3 = abs(glb_m3)
              #pragma unroll
              for (int32_t v722_k1 = 0; v722_k1 < 12; ++v722_k1) {
                tensorforge::intel_esimd::simd<float, 4> v729_data;
                v729_data.copy_from(glb_m3 + ((v722_k1 * 4)));
                r3.template select<4, 1>((v722_k1 * 16)) = (tensorforge::intel_esimd::abs(v729_data));
              }
              // s1 = store{r>s, clear}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v733_z1 = 0; v733_z1 < 12; ++v733_z1) {
                s1[(v733_z1 * 12)] = 0.0f;
              }
              #pragma unroll
              for (int32_t v739_z1 = 0; v739_z1 < 12; ++v739_z1) {
                s1[(8_i32 + (v739_z1 * 12))] = 0.0f;
              }
              #pragma unroll
              for (int32_t v746_i1 = 0; v746_i1 < 12; ++v746_i1) {
                tensorforge::intel_esimd::simd<float, 4> v749_data(r3.template select<4, 1>((v746_i1 * 16)));
                tensorforge::slmStore<float, 4>(s1 + ((4_i32 + (v746_i1 * 12))), v749_data);
              }
              tensorforge::intel_esimd::simd<float, 192> r4(0.0f);
              // ir4 = +(r2 * s1)
              // [(0, 12), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 192> ir4(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v757_data(r2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v758_data(r2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v759_data(r2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v760_data(r2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v761_data(r2.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v762_data(r2.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v763_data(r2.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v764_data(r2.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v765_data(r2.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v766_data(r2.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v767_data(r2.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v768_data(r2.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v769_acc{};
              tensorforge::intel_esimd::simd<float, 16> v773_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v769_acc += ((static_cast<float>(v773_data[0])) * v757_data);
              v769_acc += ((static_cast<float>(v773_data[1])) * v758_data);
              v769_acc += ((static_cast<float>(v773_data[2])) * v759_data);
              v769_acc += ((static_cast<float>(v773_data[3])) * v760_data);
              v769_acc += ((static_cast<float>(v773_data[4])) * v761_data);
              v769_acc += ((static_cast<float>(v773_data[5])) * v762_data);
              v769_acc += ((static_cast<float>(v773_data[6])) * v763_data);
              v769_acc += ((static_cast<float>(v773_data[7])) * v764_data);
              v769_acc += ((static_cast<float>(v773_data[8])) * v765_data);
              v769_acc += ((static_cast<float>(v773_data[9])) * v766_data);
              v769_acc += ((static_cast<float>(v773_data[10])) * v767_data);
              v769_acc += ((static_cast<float>(v773_data[11])) * v768_data);
              ir4.template select<16, 1>(0) = v769_acc;
              tensorforge::intel_esimd::simd<float, 16> v798_acc{};
              tensorforge::intel_esimd::simd<float, 16> v800_data = tensorforge::slmLoad<float, 16>(s1 + (12_i32));
              v798_acc += ((static_cast<float>(v800_data[0])) * v757_data);
              v798_acc += ((static_cast<float>(v800_data[1])) * v758_data);
              v798_acc += ((static_cast<float>(v800_data[2])) * v759_data);
              v798_acc += ((static_cast<float>(v800_data[3])) * v760_data);
              v798_acc += ((static_cast<float>(v800_data[4])) * v761_data);
              v798_acc += ((static_cast<float>(v800_data[5])) * v762_data);
              v798_acc += ((static_cast<float>(v800_data[6])) * v763_data);
              v798_acc += ((static_cast<float>(v800_data[7])) * v764_data);
              v798_acc += ((static_cast<float>(v800_data[8])) * v765_data);
              v798_acc += ((static_cast<float>(v800_data[9])) * v766_data);
              v798_acc += ((static_cast<float>(v800_data[10])) * v767_data);
              v798_acc += ((static_cast<float>(v800_data[11])) * v768_data);
              ir4.template select<16, 1>(16) = v798_acc;
              tensorforge::intel_esimd::simd<float, 16> v825_acc{};
              tensorforge::intel_esimd::simd<float, 16> v827_data = tensorforge::slmLoad<float, 16>(s1 + (24_i32));
              v825_acc += ((static_cast<float>(v827_data[0])) * v757_data);
              v825_acc += ((static_cast<float>(v827_data[1])) * v758_data);
              v825_acc += ((static_cast<float>(v827_data[2])) * v759_data);
              v825_acc += ((static_cast<float>(v827_data[3])) * v760_data);
              v825_acc += ((static_cast<float>(v827_data[4])) * v761_data);
              v825_acc += ((static_cast<float>(v827_data[5])) * v762_data);
              v825_acc += ((static_cast<float>(v827_data[6])) * v763_data);
              v825_acc += ((static_cast<float>(v827_data[7])) * v764_data);
              v825_acc += ((static_cast<float>(v827_data[8])) * v765_data);
              v825_acc += ((static_cast<float>(v827_data[9])) * v766_data);
              v825_acc += ((static_cast<float>(v827_data[10])) * v767_data);
              v825_acc += ((static_cast<float>(v827_data[11])) * v768_data);
              ir4.template select<16, 1>(32) = v825_acc;
              tensorforge::intel_esimd::simd<float, 16> v852_acc{};
              tensorforge::intel_esimd::simd<float, 16> v854_data = tensorforge::slmLoad<float, 16>(s1 + (36_i32));
              v852_acc += ((static_cast<float>(v854_data[0])) * v757_data);
              v852_acc += ((static_cast<float>(v854_data[1])) * v758_data);
              v852_acc += ((static_cast<float>(v854_data[2])) * v759_data);
              v852_acc += ((static_cast<float>(v854_data[3])) * v760_data);
              v852_acc += ((static_cast<float>(v854_data[4])) * v761_data);
              v852_acc += ((static_cast<float>(v854_data[5])) * v762_data);
              v852_acc += ((static_cast<float>(v854_data[6])) * v763_data);
              v852_acc += ((static_cast<float>(v854_data[7])) * v764_data);
              v852_acc += ((static_cast<float>(v854_data[8])) * v765_data);
              v852_acc += ((static_cast<float>(v854_data[9])) * v766_data);
              v852_acc += ((static_cast<float>(v854_data[10])) * v767_data);
              v852_acc += ((static_cast<float>(v854_data[11])) * v768_data);
              ir4.template select<16, 1>(48) = v852_acc;
              tensorforge::intel_esimd::simd<float, 16> v879_acc{};
              tensorforge::intel_esimd::simd<float, 16> v881_data = tensorforge::slmLoad<float, 16>(s1 + (48_i32));
              v879_acc += ((static_cast<float>(v881_data[0])) * v757_data);
              v879_acc += ((static_cast<float>(v881_data[1])) * v758_data);
              v879_acc += ((static_cast<float>(v881_data[2])) * v759_data);
              v879_acc += ((static_cast<float>(v881_data[3])) * v760_data);
              v879_acc += ((static_cast<float>(v881_data[4])) * v761_data);
              v879_acc += ((static_cast<float>(v881_data[5])) * v762_data);
              v879_acc += ((static_cast<float>(v881_data[6])) * v763_data);
              v879_acc += ((static_cast<float>(v881_data[7])) * v764_data);
              v879_acc += ((static_cast<float>(v881_data[8])) * v765_data);
              v879_acc += ((static_cast<float>(v881_data[9])) * v766_data);
              v879_acc += ((static_cast<float>(v881_data[10])) * v767_data);
              v879_acc += ((static_cast<float>(v881_data[11])) * v768_data);
              ir4.template select<16, 1>(64) = v879_acc;
              tensorforge::intel_esimd::simd<float, 16> v906_acc{};
              tensorforge::intel_esimd::simd<float, 16> v908_data = tensorforge::slmLoad<float, 16>(s1 + (60_i32));
              v906_acc += ((static_cast<float>(v908_data[0])) * v757_data);
              v906_acc += ((static_cast<float>(v908_data[1])) * v758_data);
              v906_acc += ((static_cast<float>(v908_data[2])) * v759_data);
              v906_acc += ((static_cast<float>(v908_data[3])) * v760_data);
              v906_acc += ((static_cast<float>(v908_data[4])) * v761_data);
              v906_acc += ((static_cast<float>(v908_data[5])) * v762_data);
              v906_acc += ((static_cast<float>(v908_data[6])) * v763_data);
              v906_acc += ((static_cast<float>(v908_data[7])) * v764_data);
              v906_acc += ((static_cast<float>(v908_data[8])) * v765_data);
              v906_acc += ((static_cast<float>(v908_data[9])) * v766_data);
              v906_acc += ((static_cast<float>(v908_data[10])) * v767_data);
              v906_acc += ((static_cast<float>(v908_data[11])) * v768_data);
              ir4.template select<16, 1>(80) = v906_acc;
              tensorforge::intel_esimd::simd<float, 16> v933_acc{};
              tensorforge::intel_esimd::simd<float, 16> v935_data = tensorforge::slmLoad<float, 16>(s1 + (72_i32));
              v933_acc += ((static_cast<float>(v935_data[0])) * v757_data);
              v933_acc += ((static_cast<float>(v935_data[1])) * v758_data);
              v933_acc += ((static_cast<float>(v935_data[2])) * v759_data);
              v933_acc += ((static_cast<float>(v935_data[3])) * v760_data);
              v933_acc += ((static_cast<float>(v935_data[4])) * v761_data);
              v933_acc += ((static_cast<float>(v935_data[5])) * v762_data);
              v933_acc += ((static_cast<float>(v935_data[6])) * v763_data);
              v933_acc += ((static_cast<float>(v935_data[7])) * v764_data);
              v933_acc += ((static_cast<float>(v935_data[8])) * v765_data);
              v933_acc += ((static_cast<float>(v935_data[9])) * v766_data);
              v933_acc += ((static_cast<float>(v935_data[10])) * v767_data);
              v933_acc += ((static_cast<float>(v935_data[11])) * v768_data);
              ir4.template select<16, 1>(96) = v933_acc;
              tensorforge::intel_esimd::simd<float, 16> v960_acc{};
              tensorforge::intel_esimd::simd<float, 16> v962_data = tensorforge::slmLoad<float, 16>(s1 + (84_i32));
              v960_acc += ((static_cast<float>(v962_data[0])) * v757_data);
              v960_acc += ((static_cast<float>(v962_data[1])) * v758_data);
              v960_acc += ((static_cast<float>(v962_data[2])) * v759_data);
              v960_acc += ((static_cast<float>(v962_data[3])) * v760_data);
              v960_acc += ((static_cast<float>(v962_data[4])) * v761_data);
              v960_acc += ((static_cast<float>(v962_data[5])) * v762_data);
              v960_acc += ((static_cast<float>(v962_data[6])) * v763_data);
              v960_acc += ((static_cast<float>(v962_data[7])) * v764_data);
              v960_acc += ((static_cast<float>(v962_data[8])) * v765_data);
              v960_acc += ((static_cast<float>(v962_data[9])) * v766_data);
              v960_acc += ((static_cast<float>(v962_data[10])) * v767_data);
              v960_acc += ((static_cast<float>(v962_data[11])) * v768_data);
              ir4.template select<16, 1>(112) = v960_acc;
              tensorforge::intel_esimd::simd<float, 16> v987_acc{};
              tensorforge::intel_esimd::simd<float, 16> v989_data = tensorforge::slmLoad<float, 16>(s1 + (96_i32));
              v987_acc += ((static_cast<float>(v989_data[0])) * v757_data);
              v987_acc += ((static_cast<float>(v989_data[1])) * v758_data);
              v987_acc += ((static_cast<float>(v989_data[2])) * v759_data);
              v987_acc += ((static_cast<float>(v989_data[3])) * v760_data);
              v987_acc += ((static_cast<float>(v989_data[4])) * v761_data);
              v987_acc += ((static_cast<float>(v989_data[5])) * v762_data);
              v987_acc += ((static_cast<float>(v989_data[6])) * v763_data);
              v987_acc += ((static_cast<float>(v989_data[7])) * v764_data);
              v987_acc += ((static_cast<float>(v989_data[8])) * v765_data);
              v987_acc += ((static_cast<float>(v989_data[9])) * v766_data);
              v987_acc += ((static_cast<float>(v989_data[10])) * v767_data);
              v987_acc += ((static_cast<float>(v989_data[11])) * v768_data);
              ir4.template select<16, 1>(128) = v987_acc;
              tensorforge::intel_esimd::simd<float, 16> v1014_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1016_data = tensorforge::slmLoad<float, 16>(s1 + (108_i32));
              v1014_acc += ((static_cast<float>(v1016_data[0])) * v757_data);
              v1014_acc += ((static_cast<float>(v1016_data[1])) * v758_data);
              v1014_acc += ((static_cast<float>(v1016_data[2])) * v759_data);
              v1014_acc += ((static_cast<float>(v1016_data[3])) * v760_data);
              v1014_acc += ((static_cast<float>(v1016_data[4])) * v761_data);
              v1014_acc += ((static_cast<float>(v1016_data[5])) * v762_data);
              v1014_acc += ((static_cast<float>(v1016_data[6])) * v763_data);
              v1014_acc += ((static_cast<float>(v1016_data[7])) * v764_data);
              v1014_acc += ((static_cast<float>(v1016_data[8])) * v765_data);
              v1014_acc += ((static_cast<float>(v1016_data[9])) * v766_data);
              v1014_acc += ((static_cast<float>(v1016_data[10])) * v767_data);
              v1014_acc += ((static_cast<float>(v1016_data[11])) * v768_data);
              ir4.template select<16, 1>(144) = v1014_acc;
              tensorforge::intel_esimd::simd<float, 16> v1041_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1043_data = tensorforge::slmLoad<float, 16>(s1 + (120_i32));
              v1041_acc += ((static_cast<float>(v1043_data[0])) * v757_data);
              v1041_acc += ((static_cast<float>(v1043_data[1])) * v758_data);
              v1041_acc += ((static_cast<float>(v1043_data[2])) * v759_data);
              v1041_acc += ((static_cast<float>(v1043_data[3])) * v760_data);
              v1041_acc += ((static_cast<float>(v1043_data[4])) * v761_data);
              v1041_acc += ((static_cast<float>(v1043_data[5])) * v762_data);
              v1041_acc += ((static_cast<float>(v1043_data[6])) * v763_data);
              v1041_acc += ((static_cast<float>(v1043_data[7])) * v764_data);
              v1041_acc += ((static_cast<float>(v1043_data[8])) * v765_data);
              v1041_acc += ((static_cast<float>(v1043_data[9])) * v766_data);
              v1041_acc += ((static_cast<float>(v1043_data[10])) * v767_data);
              v1041_acc += ((static_cast<float>(v1043_data[11])) * v768_data);
              ir4.template select<16, 1>(160) = v1041_acc;
              tensorforge::intel_esimd::simd<float, 16> v1068_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1070_data = tensorforge::slmLoad<float, 16>(s1 + (132_i32));
              v1068_acc += ((static_cast<float>(v1070_data[0])) * v757_data);
              v1068_acc += ((static_cast<float>(v1070_data[1])) * v758_data);
              v1068_acc += ((static_cast<float>(v1070_data[2])) * v759_data);
              v1068_acc += ((static_cast<float>(v1070_data[3])) * v760_data);
              v1068_acc += ((static_cast<float>(v1070_data[4])) * v761_data);
              v1068_acc += ((static_cast<float>(v1070_data[5])) * v762_data);
              v1068_acc += ((static_cast<float>(v1070_data[6])) * v763_data);
              v1068_acc += ((static_cast<float>(v1070_data[7])) * v764_data);
              v1068_acc += ((static_cast<float>(v1070_data[8])) * v765_data);
              v1068_acc += ((static_cast<float>(v1070_data[9])) * v766_data);
              v1068_acc += ((static_cast<float>(v1070_data[10])) * v767_data);
              v1068_acc += ((static_cast<float>(v1070_data[11])) * v768_data);
              ir4.template select<16, 1>(176) = v1068_acc;
              // r4 = ir4
              #pragma unroll
              for (int32_t v1095_n1 = 0; v1095_n1 < 12; ++v1095_n1) {
                int32_t v1096_a = v1095_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v1098_data(ir4.template select<12, 1>(v1096_a));
                r4.template select<12, 1>(v1096_a) = v1098_data;
              }
              // glb_m4 = store{r>g}(r4);
              #pragma unroll
              for (int32_t v1099_i1 = 0; v1099_i1 < 12; ++v1099_i1) {
                tensorforge::intel_esimd::simd<float, 12> v1102_data(r4.template select<12, 1>((v1099_i1 * 16)));
                v1102_data.copy_to(glb_m4 + ((v1099_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}

