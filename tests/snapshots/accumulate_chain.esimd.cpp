// === base name ===
kernel_3f680083972439e5

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_3f680083972439e5 = {{1, 16, 1}, 16, 12, 1, 16, 7168, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_3f680083972439e5(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_3f680083972439e5(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_3f680083972439e5(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 1792 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_3f680083972439e5(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_3f680083972439e5(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_3f680083972439e5(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, m6, m6_extraOffset, m7, m7_extraOffset, m8, m8_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_3f680083972439e5(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<1792 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 7168 B shared, occupancy grid
        // operands:
        //   m0 12×8(12×8) {0..12}×{0..8} strided
        //   m1 12×12(12×12) {0..12}×{0..12} strided
        //   m2 12×8(12×8) {0..12}×{0..8} strided
        //   m3 12×12(12×12) {0..12}×{0..12} strided
        //   m4 12×8(12×8) {0..12}×{0..8} strided
        //   m5 12×12(12×12) {0..12}×{0..12} strided
        //   m6 12×8(12×8) {0..12}×{0..8} strided
        //   m7 12×12(12×12) {0..12}×{0..12} strided
        //   m8 12×8(12×8) {0..12}×{0..8} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        //   m0[i,j] += m3[i,k] × m4[k,j]
        //   m0[i,j] += m5[i,k] × m6[k,j]
        //   m0[i,j] += m7[i,k] × m8[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1792}],"shared_bytes":7168,"shared_elements":1792,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[12,8]],"name":"m0","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A0","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B0","bbox":[[0,0],[12,8]],"name":"m2","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A1","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[12,8]],"name":"m4","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A2","bbox":[[0,0],[12,12]],"name":"m5","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[12,8]],"name":"m6","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A3","bbox":[[0,0],[12,12]],"name":"m7","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B3","bbox":[[0,0],[12,8]],"name":"m8","ordered":false,"parts":1,"shape":[12,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m7","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m8","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (112 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s2 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s3 = localShrMem0 + (0);
          for (size_t v11_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v11_batchId0 < numElements0; v11_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v12_ahead1 = v11_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v11_batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v11_batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v11_batchId0 * 96 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v11_batchId0 * 144 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v11_batchId0 * 96 + 0 + m4_extraOffset];
              const float *const __restrict__ glb_m5 = &m5[v11_batchId0 * 144 + 0 + m5_extraOffset];
              const float *const __restrict__ glb_m6 = &m6[v11_batchId0 * 96 + 0 + m6_extraOffset];
              const float *const __restrict__ glb_m7 = &m7[v11_batchId0 * 144 + 0 + m7_extraOffset];
              const float *const __restrict__ glb_m8 = &m8[v11_batchId0 * 96 + 0 + m8_extraOffset];
              tensorforge::intel_esimd::simd<float, 192> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v29_i1 = 0; v29_i1 < 12; ++v29_i1) {
                tensorforge::intel_esimd::simd<float, 12> v34_data;
                v34_data.copy_from(glb_m1 + ((v29_i1 * 12)));
                r0.template select<12, 1>((v29_i1 * 16)) = v34_data;
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v37_ld;
              v37_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v37_ld);
              tensorforge::intel_esimd::simd<float, 32> v38_ld;
              v38_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 64));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 2 * 0 + 64), v38_ld);
              tensorforge::intel_esimd::simd<float, 192> r2(0.0f);
              // r2 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v276_i1 = 0; v276_i1 < 12; ++v276_i1) {
                tensorforge::intel_esimd::simd<float, 12> v281_data;
                v281_data.copy_from(glb_m3 + ((v276_i1 * 12)));
                r2.template select<12, 1>((v276_i1 * 16)) = v281_data;
              }
              tensorforge::intel_esimd::simd<float, 128> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 12), (0, 8)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 128> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v48_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v49_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v50_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v51_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v52_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v53_acc{};
              tensorforge::intel_esimd::simd<float, 16> v57_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v53_acc += ((static_cast<float>(v57_data[0])) * v41_data);
              v53_acc += ((static_cast<float>(v57_data[1])) * v42_data);
              v53_acc += ((static_cast<float>(v57_data[2])) * v43_data);
              v53_acc += ((static_cast<float>(v57_data[3])) * v44_data);
              v53_acc += ((static_cast<float>(v57_data[4])) * v45_data);
              v53_acc += ((static_cast<float>(v57_data[5])) * v46_data);
              v53_acc += ((static_cast<float>(v57_data[6])) * v47_data);
              v53_acc += ((static_cast<float>(v57_data[7])) * v48_data);
              v53_acc += ((static_cast<float>(v57_data[8])) * v49_data);
              v53_acc += ((static_cast<float>(v57_data[9])) * v50_data);
              v53_acc += ((static_cast<float>(v57_data[10])) * v51_data);
              v53_acc += ((static_cast<float>(v57_data[11])) * v52_data);
              ir1.template select<16, 1>(0) = v53_acc;
              tensorforge::intel_esimd::simd<float, 16> v82_acc{};
              tensorforge::intel_esimd::simd<float, 16> v84_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              v82_acc += ((static_cast<float>(v84_data[0])) * v41_data);
              v82_acc += ((static_cast<float>(v84_data[1])) * v42_data);
              v82_acc += ((static_cast<float>(v84_data[2])) * v43_data);
              v82_acc += ((static_cast<float>(v84_data[3])) * v44_data);
              v82_acc += ((static_cast<float>(v84_data[4])) * v45_data);
              v82_acc += ((static_cast<float>(v84_data[5])) * v46_data);
              v82_acc += ((static_cast<float>(v84_data[6])) * v47_data);
              v82_acc += ((static_cast<float>(v84_data[7])) * v48_data);
              v82_acc += ((static_cast<float>(v84_data[8])) * v49_data);
              v82_acc += ((static_cast<float>(v84_data[9])) * v50_data);
              v82_acc += ((static_cast<float>(v84_data[10])) * v51_data);
              v82_acc += ((static_cast<float>(v84_data[11])) * v52_data);
              ir1.template select<16, 1>(16) = v82_acc;
              tensorforge::intel_esimd::simd<float, 16> v109_acc{};
              tensorforge::intel_esimd::simd<float, 16> v111_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              v109_acc += ((static_cast<float>(v111_data[0])) * v41_data);
              v109_acc += ((static_cast<float>(v111_data[1])) * v42_data);
              v109_acc += ((static_cast<float>(v111_data[2])) * v43_data);
              v109_acc += ((static_cast<float>(v111_data[3])) * v44_data);
              v109_acc += ((static_cast<float>(v111_data[4])) * v45_data);
              v109_acc += ((static_cast<float>(v111_data[5])) * v46_data);
              v109_acc += ((static_cast<float>(v111_data[6])) * v47_data);
              v109_acc += ((static_cast<float>(v111_data[7])) * v48_data);
              v109_acc += ((static_cast<float>(v111_data[8])) * v49_data);
              v109_acc += ((static_cast<float>(v111_data[9])) * v50_data);
              v109_acc += ((static_cast<float>(v111_data[10])) * v51_data);
              v109_acc += ((static_cast<float>(v111_data[11])) * v52_data);
              ir1.template select<16, 1>(32) = v109_acc;
              tensorforge::intel_esimd::simd<float, 16> v136_acc{};
              tensorforge::intel_esimd::simd<float, 16> v138_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              v136_acc += ((static_cast<float>(v138_data[0])) * v41_data);
              v136_acc += ((static_cast<float>(v138_data[1])) * v42_data);
              v136_acc += ((static_cast<float>(v138_data[2])) * v43_data);
              v136_acc += ((static_cast<float>(v138_data[3])) * v44_data);
              v136_acc += ((static_cast<float>(v138_data[4])) * v45_data);
              v136_acc += ((static_cast<float>(v138_data[5])) * v46_data);
              v136_acc += ((static_cast<float>(v138_data[6])) * v47_data);
              v136_acc += ((static_cast<float>(v138_data[7])) * v48_data);
              v136_acc += ((static_cast<float>(v138_data[8])) * v49_data);
              v136_acc += ((static_cast<float>(v138_data[9])) * v50_data);
              v136_acc += ((static_cast<float>(v138_data[10])) * v51_data);
              v136_acc += ((static_cast<float>(v138_data[11])) * v52_data);
              ir1.template select<16, 1>(48) = v136_acc;
              tensorforge::intel_esimd::simd<float, 16> v163_acc{};
              tensorforge::intel_esimd::simd<float, 16> v165_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v163_acc += ((static_cast<float>(v165_data[0])) * v41_data);
              v163_acc += ((static_cast<float>(v165_data[1])) * v42_data);
              v163_acc += ((static_cast<float>(v165_data[2])) * v43_data);
              v163_acc += ((static_cast<float>(v165_data[3])) * v44_data);
              v163_acc += ((static_cast<float>(v165_data[4])) * v45_data);
              v163_acc += ((static_cast<float>(v165_data[5])) * v46_data);
              v163_acc += ((static_cast<float>(v165_data[6])) * v47_data);
              v163_acc += ((static_cast<float>(v165_data[7])) * v48_data);
              v163_acc += ((static_cast<float>(v165_data[8])) * v49_data);
              v163_acc += ((static_cast<float>(v165_data[9])) * v50_data);
              v163_acc += ((static_cast<float>(v165_data[10])) * v51_data);
              v163_acc += ((static_cast<float>(v165_data[11])) * v52_data);
              ir1.template select<16, 1>(64) = v163_acc;
              tensorforge::intel_esimd::simd<float, 16> v190_acc{};
              tensorforge::intel_esimd::simd<float, 16> v192_data = tensorforge::slmLoad<float, 16>(s0 + (60_i32));
              v190_acc += ((static_cast<float>(v192_data[0])) * v41_data);
              v190_acc += ((static_cast<float>(v192_data[1])) * v42_data);
              v190_acc += ((static_cast<float>(v192_data[2])) * v43_data);
              v190_acc += ((static_cast<float>(v192_data[3])) * v44_data);
              v190_acc += ((static_cast<float>(v192_data[4])) * v45_data);
              v190_acc += ((static_cast<float>(v192_data[5])) * v46_data);
              v190_acc += ((static_cast<float>(v192_data[6])) * v47_data);
              v190_acc += ((static_cast<float>(v192_data[7])) * v48_data);
              v190_acc += ((static_cast<float>(v192_data[8])) * v49_data);
              v190_acc += ((static_cast<float>(v192_data[9])) * v50_data);
              v190_acc += ((static_cast<float>(v192_data[10])) * v51_data);
              v190_acc += ((static_cast<float>(v192_data[11])) * v52_data);
              ir1.template select<16, 1>(80) = v190_acc;
              tensorforge::intel_esimd::simd<float, 16> v217_acc{};
              tensorforge::intel_esimd::simd<float, 16> v219_data = tensorforge::slmLoad<float, 16>(s0 + (72_i32));
              v217_acc += ((static_cast<float>(v219_data[0])) * v41_data);
              v217_acc += ((static_cast<float>(v219_data[1])) * v42_data);
              v217_acc += ((static_cast<float>(v219_data[2])) * v43_data);
              v217_acc += ((static_cast<float>(v219_data[3])) * v44_data);
              v217_acc += ((static_cast<float>(v219_data[4])) * v45_data);
              v217_acc += ((static_cast<float>(v219_data[5])) * v46_data);
              v217_acc += ((static_cast<float>(v219_data[6])) * v47_data);
              v217_acc += ((static_cast<float>(v219_data[7])) * v48_data);
              v217_acc += ((static_cast<float>(v219_data[8])) * v49_data);
              v217_acc += ((static_cast<float>(v219_data[9])) * v50_data);
              v217_acc += ((static_cast<float>(v219_data[10])) * v51_data);
              v217_acc += ((static_cast<float>(v219_data[11])) * v52_data);
              ir1.template select<16, 1>(96) = v217_acc;
              tensorforge::intel_esimd::simd<float, 16> v244_acc{};
              tensorforge::intel_esimd::simd<float, 16> v246_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v244_acc += ((static_cast<float>(v246_data[0])) * v41_data);
              v244_acc += ((static_cast<float>(v246_data[1])) * v42_data);
              v244_acc += ((static_cast<float>(v246_data[2])) * v43_data);
              v244_acc += ((static_cast<float>(v246_data[3])) * v44_data);
              v244_acc += ((static_cast<float>(v246_data[4])) * v45_data);
              v244_acc += ((static_cast<float>(v246_data[5])) * v46_data);
              v244_acc += ((static_cast<float>(v246_data[6])) * v47_data);
              v244_acc += ((static_cast<float>(v246_data[7])) * v48_data);
              v244_acc += ((static_cast<float>(v246_data[8])) * v49_data);
              v244_acc += ((static_cast<float>(v246_data[9])) * v50_data);
              v244_acc += ((static_cast<float>(v246_data[10])) * v51_data);
              v244_acc += ((static_cast<float>(v246_data[11])) * v52_data);
              ir1.template select<16, 1>(112) = v244_acc;
              // r1 = ir1
              #pragma unroll
              for (int32_t v271_n1 = 0; v271_n1 < 8; ++v271_n1) {
                int32_t v272_a = v271_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v274_data(ir1.template select<12, 1>(v272_a));
                r1.template select<12, 1>(v272_a) = v274_data;
              }
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v284_ld;
              v284_ld.copy_from(glb_m4 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 4 * 0 + 0), v284_ld);
              tensorforge::intel_esimd::simd<float, 32> v285_ld;
              v285_ld.copy_from(glb_m4 + (0 + 0 + 2 * 0 + 64));
              tensorforge::slmStore<float, 32>(s1 + (0 + 0 + 2 * 0 + 64), v285_ld);
              tensorforge::intel_esimd::simd<float, 192> r4(0.0f);
              // r4 = load{g>r}(glb_m5);
              #pragma unroll
              for (int32_t v525_i1 = 0; v525_i1 < 12; ++v525_i1) {
                tensorforge::intel_esimd::simd<float, 12> v530_data;
                v530_data.copy_from(glb_m5 + ((v525_i1 * 12)));
                r4.template select<12, 1>((v525_i1 * 16)) = v530_data;
              }
              tensorforge::intel_esimd::simd<float, 128> r3(0.0f);
              // ir3 = +(r2 * s1)
              // [(0, 12), (0, 8)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 128> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v288_data(r2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v289_data(r2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v290_data(r2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v291_data(r2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v292_data(r2.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v293_data(r2.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v294_data(r2.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v295_data(r2.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v296_data(r2.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v297_data(r2.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v298_data(r2.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v299_data(r2.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v300_acc{};
              tensorforge::intel_esimd::simd<float, 16> v304_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v300_acc += ((static_cast<float>(v304_data[0])) * v288_data);
              v300_acc += ((static_cast<float>(v304_data[1])) * v289_data);
              v300_acc += ((static_cast<float>(v304_data[2])) * v290_data);
              v300_acc += ((static_cast<float>(v304_data[3])) * v291_data);
              v300_acc += ((static_cast<float>(v304_data[4])) * v292_data);
              v300_acc += ((static_cast<float>(v304_data[5])) * v293_data);
              v300_acc += ((static_cast<float>(v304_data[6])) * v294_data);
              v300_acc += ((static_cast<float>(v304_data[7])) * v295_data);
              v300_acc += ((static_cast<float>(v304_data[8])) * v296_data);
              v300_acc += ((static_cast<float>(v304_data[9])) * v297_data);
              v300_acc += ((static_cast<float>(v304_data[10])) * v298_data);
              v300_acc += ((static_cast<float>(v304_data[11])) * v299_data);
              ir3.template select<16, 1>(0) = v300_acc;
              tensorforge::intel_esimd::simd<float, 16> v329_acc{};
              tensorforge::intel_esimd::simd<float, 16> v331_data = tensorforge::slmLoad<float, 16>(s1 + (12_i32));
              v329_acc += ((static_cast<float>(v331_data[0])) * v288_data);
              v329_acc += ((static_cast<float>(v331_data[1])) * v289_data);
              v329_acc += ((static_cast<float>(v331_data[2])) * v290_data);
              v329_acc += ((static_cast<float>(v331_data[3])) * v291_data);
              v329_acc += ((static_cast<float>(v331_data[4])) * v292_data);
              v329_acc += ((static_cast<float>(v331_data[5])) * v293_data);
              v329_acc += ((static_cast<float>(v331_data[6])) * v294_data);
              v329_acc += ((static_cast<float>(v331_data[7])) * v295_data);
              v329_acc += ((static_cast<float>(v331_data[8])) * v296_data);
              v329_acc += ((static_cast<float>(v331_data[9])) * v297_data);
              v329_acc += ((static_cast<float>(v331_data[10])) * v298_data);
              v329_acc += ((static_cast<float>(v331_data[11])) * v299_data);
              ir3.template select<16, 1>(16) = v329_acc;
              tensorforge::intel_esimd::simd<float, 16> v356_acc{};
              tensorforge::intel_esimd::simd<float, 16> v358_data = tensorforge::slmLoad<float, 16>(s1 + (24_i32));
              v356_acc += ((static_cast<float>(v358_data[0])) * v288_data);
              v356_acc += ((static_cast<float>(v358_data[1])) * v289_data);
              v356_acc += ((static_cast<float>(v358_data[2])) * v290_data);
              v356_acc += ((static_cast<float>(v358_data[3])) * v291_data);
              v356_acc += ((static_cast<float>(v358_data[4])) * v292_data);
              v356_acc += ((static_cast<float>(v358_data[5])) * v293_data);
              v356_acc += ((static_cast<float>(v358_data[6])) * v294_data);
              v356_acc += ((static_cast<float>(v358_data[7])) * v295_data);
              v356_acc += ((static_cast<float>(v358_data[8])) * v296_data);
              v356_acc += ((static_cast<float>(v358_data[9])) * v297_data);
              v356_acc += ((static_cast<float>(v358_data[10])) * v298_data);
              v356_acc += ((static_cast<float>(v358_data[11])) * v299_data);
              ir3.template select<16, 1>(32) = v356_acc;
              tensorforge::intel_esimd::simd<float, 16> v383_acc{};
              tensorforge::intel_esimd::simd<float, 16> v385_data = tensorforge::slmLoad<float, 16>(s1 + (36_i32));
              v383_acc += ((static_cast<float>(v385_data[0])) * v288_data);
              v383_acc += ((static_cast<float>(v385_data[1])) * v289_data);
              v383_acc += ((static_cast<float>(v385_data[2])) * v290_data);
              v383_acc += ((static_cast<float>(v385_data[3])) * v291_data);
              v383_acc += ((static_cast<float>(v385_data[4])) * v292_data);
              v383_acc += ((static_cast<float>(v385_data[5])) * v293_data);
              v383_acc += ((static_cast<float>(v385_data[6])) * v294_data);
              v383_acc += ((static_cast<float>(v385_data[7])) * v295_data);
              v383_acc += ((static_cast<float>(v385_data[8])) * v296_data);
              v383_acc += ((static_cast<float>(v385_data[9])) * v297_data);
              v383_acc += ((static_cast<float>(v385_data[10])) * v298_data);
              v383_acc += ((static_cast<float>(v385_data[11])) * v299_data);
              ir3.template select<16, 1>(48) = v383_acc;
              tensorforge::intel_esimd::simd<float, 16> v410_acc{};
              tensorforge::intel_esimd::simd<float, 16> v412_data = tensorforge::slmLoad<float, 16>(s1 + (48_i32));
              v410_acc += ((static_cast<float>(v412_data[0])) * v288_data);
              v410_acc += ((static_cast<float>(v412_data[1])) * v289_data);
              v410_acc += ((static_cast<float>(v412_data[2])) * v290_data);
              v410_acc += ((static_cast<float>(v412_data[3])) * v291_data);
              v410_acc += ((static_cast<float>(v412_data[4])) * v292_data);
              v410_acc += ((static_cast<float>(v412_data[5])) * v293_data);
              v410_acc += ((static_cast<float>(v412_data[6])) * v294_data);
              v410_acc += ((static_cast<float>(v412_data[7])) * v295_data);
              v410_acc += ((static_cast<float>(v412_data[8])) * v296_data);
              v410_acc += ((static_cast<float>(v412_data[9])) * v297_data);
              v410_acc += ((static_cast<float>(v412_data[10])) * v298_data);
              v410_acc += ((static_cast<float>(v412_data[11])) * v299_data);
              ir3.template select<16, 1>(64) = v410_acc;
              tensorforge::intel_esimd::simd<float, 16> v437_acc{};
              tensorforge::intel_esimd::simd<float, 16> v439_data = tensorforge::slmLoad<float, 16>(s1 + (60_i32));
              v437_acc += ((static_cast<float>(v439_data[0])) * v288_data);
              v437_acc += ((static_cast<float>(v439_data[1])) * v289_data);
              v437_acc += ((static_cast<float>(v439_data[2])) * v290_data);
              v437_acc += ((static_cast<float>(v439_data[3])) * v291_data);
              v437_acc += ((static_cast<float>(v439_data[4])) * v292_data);
              v437_acc += ((static_cast<float>(v439_data[5])) * v293_data);
              v437_acc += ((static_cast<float>(v439_data[6])) * v294_data);
              v437_acc += ((static_cast<float>(v439_data[7])) * v295_data);
              v437_acc += ((static_cast<float>(v439_data[8])) * v296_data);
              v437_acc += ((static_cast<float>(v439_data[9])) * v297_data);
              v437_acc += ((static_cast<float>(v439_data[10])) * v298_data);
              v437_acc += ((static_cast<float>(v439_data[11])) * v299_data);
              ir3.template select<16, 1>(80) = v437_acc;
              tensorforge::intel_esimd::simd<float, 16> v464_acc{};
              tensorforge::intel_esimd::simd<float, 16> v466_data = tensorforge::slmLoad<float, 16>(s1 + (72_i32));
              v464_acc += ((static_cast<float>(v466_data[0])) * v288_data);
              v464_acc += ((static_cast<float>(v466_data[1])) * v289_data);
              v464_acc += ((static_cast<float>(v466_data[2])) * v290_data);
              v464_acc += ((static_cast<float>(v466_data[3])) * v291_data);
              v464_acc += ((static_cast<float>(v466_data[4])) * v292_data);
              v464_acc += ((static_cast<float>(v466_data[5])) * v293_data);
              v464_acc += ((static_cast<float>(v466_data[6])) * v294_data);
              v464_acc += ((static_cast<float>(v466_data[7])) * v295_data);
              v464_acc += ((static_cast<float>(v466_data[8])) * v296_data);
              v464_acc += ((static_cast<float>(v466_data[9])) * v297_data);
              v464_acc += ((static_cast<float>(v466_data[10])) * v298_data);
              v464_acc += ((static_cast<float>(v466_data[11])) * v299_data);
              ir3.template select<16, 1>(96) = v464_acc;
              tensorforge::intel_esimd::simd<float, 16> v491_acc{};
              tensorforge::intel_esimd::simd<float, 16> v493_data = tensorforge::slmLoad<float, 16>(s1 + (84_i32));
              v491_acc += ((static_cast<float>(v493_data[0])) * v288_data);
              v491_acc += ((static_cast<float>(v493_data[1])) * v289_data);
              v491_acc += ((static_cast<float>(v493_data[2])) * v290_data);
              v491_acc += ((static_cast<float>(v493_data[3])) * v291_data);
              v491_acc += ((static_cast<float>(v493_data[4])) * v292_data);
              v491_acc += ((static_cast<float>(v493_data[5])) * v293_data);
              v491_acc += ((static_cast<float>(v493_data[6])) * v294_data);
              v491_acc += ((static_cast<float>(v493_data[7])) * v295_data);
              v491_acc += ((static_cast<float>(v493_data[8])) * v296_data);
              v491_acc += ((static_cast<float>(v493_data[9])) * v297_data);
              v491_acc += ((static_cast<float>(v493_data[10])) * v298_data);
              v491_acc += ((static_cast<float>(v493_data[11])) * v299_data);
              ir3.template select<16, 1>(112) = v491_acc;
              // r3 = ir3 + r1
              #pragma unroll
              for (int32_t v518_n1 = 0; v518_n1 < 8; ++v518_n1) {
                int32_t v519_a = v518_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v521_data(ir3.template select<12, 1>(v519_a));
                tensorforge::intel_esimd::simd<float, 12> v522_data(r1.template select<12, 1>(v519_a));
                r3.template select<12, 1>(v519_a) = (v522_data + v521_data);
              }
              // s2 = load{g>s}(glb_m6[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v533_ld;
              v533_ld.copy_from(glb_m6 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s2 + (0 + 0 + 4 * 0 + 0), v533_ld);
              tensorforge::intel_esimd::simd<float, 32> v534_ld;
              v534_ld.copy_from(glb_m6 + (0 + 0 + 2 * 0 + 64));
              tensorforge::slmStore<float, 32>(s2 + (0 + 0 + 2 * 0 + 64), v534_ld);
              tensorforge::intel_esimd::simd<float, 192> r6(0.0f);
              // r6 = load{g>r}(glb_m7);
              #pragma unroll
              for (int32_t v774_i1 = 0; v774_i1 < 12; ++v774_i1) {
                tensorforge::intel_esimd::simd<float, 12> v779_data;
                v779_data.copy_from(glb_m7 + ((v774_i1 * 12)));
                r6.template select<12, 1>((v774_i1 * 16)) = v779_data;
              }
              tensorforge::intel_esimd::simd<float, 128> r5(0.0f);
              // ir5 = +(r4 * s2)
              // [(0, 12), (0, 8)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 128> ir5(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v537_data(r4.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v538_data(r4.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v539_data(r4.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v540_data(r4.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v541_data(r4.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v542_data(r4.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v543_data(r4.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v544_data(r4.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v545_data(r4.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v546_data(r4.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v547_data(r4.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v548_data(r4.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v549_acc{};
              tensorforge::intel_esimd::simd<float, 16> v553_data = tensorforge::slmLoad<float, 16>(s2 + (0_i32));
              v549_acc += ((static_cast<float>(v553_data[0])) * v537_data);
              v549_acc += ((static_cast<float>(v553_data[1])) * v538_data);
              v549_acc += ((static_cast<float>(v553_data[2])) * v539_data);
              v549_acc += ((static_cast<float>(v553_data[3])) * v540_data);
              v549_acc += ((static_cast<float>(v553_data[4])) * v541_data);
              v549_acc += ((static_cast<float>(v553_data[5])) * v542_data);
              v549_acc += ((static_cast<float>(v553_data[6])) * v543_data);
              v549_acc += ((static_cast<float>(v553_data[7])) * v544_data);
              v549_acc += ((static_cast<float>(v553_data[8])) * v545_data);
              v549_acc += ((static_cast<float>(v553_data[9])) * v546_data);
              v549_acc += ((static_cast<float>(v553_data[10])) * v547_data);
              v549_acc += ((static_cast<float>(v553_data[11])) * v548_data);
              ir5.template select<16, 1>(0) = v549_acc;
              tensorforge::intel_esimd::simd<float, 16> v578_acc{};
              tensorforge::intel_esimd::simd<float, 16> v580_data = tensorforge::slmLoad<float, 16>(s2 + (12_i32));
              v578_acc += ((static_cast<float>(v580_data[0])) * v537_data);
              v578_acc += ((static_cast<float>(v580_data[1])) * v538_data);
              v578_acc += ((static_cast<float>(v580_data[2])) * v539_data);
              v578_acc += ((static_cast<float>(v580_data[3])) * v540_data);
              v578_acc += ((static_cast<float>(v580_data[4])) * v541_data);
              v578_acc += ((static_cast<float>(v580_data[5])) * v542_data);
              v578_acc += ((static_cast<float>(v580_data[6])) * v543_data);
              v578_acc += ((static_cast<float>(v580_data[7])) * v544_data);
              v578_acc += ((static_cast<float>(v580_data[8])) * v545_data);
              v578_acc += ((static_cast<float>(v580_data[9])) * v546_data);
              v578_acc += ((static_cast<float>(v580_data[10])) * v547_data);
              v578_acc += ((static_cast<float>(v580_data[11])) * v548_data);
              ir5.template select<16, 1>(16) = v578_acc;
              tensorforge::intel_esimd::simd<float, 16> v605_acc{};
              tensorforge::intel_esimd::simd<float, 16> v607_data = tensorforge::slmLoad<float, 16>(s2 + (24_i32));
              v605_acc += ((static_cast<float>(v607_data[0])) * v537_data);
              v605_acc += ((static_cast<float>(v607_data[1])) * v538_data);
              v605_acc += ((static_cast<float>(v607_data[2])) * v539_data);
              v605_acc += ((static_cast<float>(v607_data[3])) * v540_data);
              v605_acc += ((static_cast<float>(v607_data[4])) * v541_data);
              v605_acc += ((static_cast<float>(v607_data[5])) * v542_data);
              v605_acc += ((static_cast<float>(v607_data[6])) * v543_data);
              v605_acc += ((static_cast<float>(v607_data[7])) * v544_data);
              v605_acc += ((static_cast<float>(v607_data[8])) * v545_data);
              v605_acc += ((static_cast<float>(v607_data[9])) * v546_data);
              v605_acc += ((static_cast<float>(v607_data[10])) * v547_data);
              v605_acc += ((static_cast<float>(v607_data[11])) * v548_data);
              ir5.template select<16, 1>(32) = v605_acc;
              tensorforge::intel_esimd::simd<float, 16> v632_acc{};
              tensorforge::intel_esimd::simd<float, 16> v634_data = tensorforge::slmLoad<float, 16>(s2 + (36_i32));
              v632_acc += ((static_cast<float>(v634_data[0])) * v537_data);
              v632_acc += ((static_cast<float>(v634_data[1])) * v538_data);
              v632_acc += ((static_cast<float>(v634_data[2])) * v539_data);
              v632_acc += ((static_cast<float>(v634_data[3])) * v540_data);
              v632_acc += ((static_cast<float>(v634_data[4])) * v541_data);
              v632_acc += ((static_cast<float>(v634_data[5])) * v542_data);
              v632_acc += ((static_cast<float>(v634_data[6])) * v543_data);
              v632_acc += ((static_cast<float>(v634_data[7])) * v544_data);
              v632_acc += ((static_cast<float>(v634_data[8])) * v545_data);
              v632_acc += ((static_cast<float>(v634_data[9])) * v546_data);
              v632_acc += ((static_cast<float>(v634_data[10])) * v547_data);
              v632_acc += ((static_cast<float>(v634_data[11])) * v548_data);
              ir5.template select<16, 1>(48) = v632_acc;
              tensorforge::intel_esimd::simd<float, 16> v659_acc{};
              tensorforge::intel_esimd::simd<float, 16> v661_data = tensorforge::slmLoad<float, 16>(s2 + (48_i32));
              v659_acc += ((static_cast<float>(v661_data[0])) * v537_data);
              v659_acc += ((static_cast<float>(v661_data[1])) * v538_data);
              v659_acc += ((static_cast<float>(v661_data[2])) * v539_data);
              v659_acc += ((static_cast<float>(v661_data[3])) * v540_data);
              v659_acc += ((static_cast<float>(v661_data[4])) * v541_data);
              v659_acc += ((static_cast<float>(v661_data[5])) * v542_data);
              v659_acc += ((static_cast<float>(v661_data[6])) * v543_data);
              v659_acc += ((static_cast<float>(v661_data[7])) * v544_data);
              v659_acc += ((static_cast<float>(v661_data[8])) * v545_data);
              v659_acc += ((static_cast<float>(v661_data[9])) * v546_data);
              v659_acc += ((static_cast<float>(v661_data[10])) * v547_data);
              v659_acc += ((static_cast<float>(v661_data[11])) * v548_data);
              ir5.template select<16, 1>(64) = v659_acc;
              tensorforge::intel_esimd::simd<float, 16> v686_acc{};
              tensorforge::intel_esimd::simd<float, 16> v688_data = tensorforge::slmLoad<float, 16>(s2 + (60_i32));
              v686_acc += ((static_cast<float>(v688_data[0])) * v537_data);
              v686_acc += ((static_cast<float>(v688_data[1])) * v538_data);
              v686_acc += ((static_cast<float>(v688_data[2])) * v539_data);
              v686_acc += ((static_cast<float>(v688_data[3])) * v540_data);
              v686_acc += ((static_cast<float>(v688_data[4])) * v541_data);
              v686_acc += ((static_cast<float>(v688_data[5])) * v542_data);
              v686_acc += ((static_cast<float>(v688_data[6])) * v543_data);
              v686_acc += ((static_cast<float>(v688_data[7])) * v544_data);
              v686_acc += ((static_cast<float>(v688_data[8])) * v545_data);
              v686_acc += ((static_cast<float>(v688_data[9])) * v546_data);
              v686_acc += ((static_cast<float>(v688_data[10])) * v547_data);
              v686_acc += ((static_cast<float>(v688_data[11])) * v548_data);
              ir5.template select<16, 1>(80) = v686_acc;
              tensorforge::intel_esimd::simd<float, 16> v713_acc{};
              tensorforge::intel_esimd::simd<float, 16> v715_data = tensorforge::slmLoad<float, 16>(s2 + (72_i32));
              v713_acc += ((static_cast<float>(v715_data[0])) * v537_data);
              v713_acc += ((static_cast<float>(v715_data[1])) * v538_data);
              v713_acc += ((static_cast<float>(v715_data[2])) * v539_data);
              v713_acc += ((static_cast<float>(v715_data[3])) * v540_data);
              v713_acc += ((static_cast<float>(v715_data[4])) * v541_data);
              v713_acc += ((static_cast<float>(v715_data[5])) * v542_data);
              v713_acc += ((static_cast<float>(v715_data[6])) * v543_data);
              v713_acc += ((static_cast<float>(v715_data[7])) * v544_data);
              v713_acc += ((static_cast<float>(v715_data[8])) * v545_data);
              v713_acc += ((static_cast<float>(v715_data[9])) * v546_data);
              v713_acc += ((static_cast<float>(v715_data[10])) * v547_data);
              v713_acc += ((static_cast<float>(v715_data[11])) * v548_data);
              ir5.template select<16, 1>(96) = v713_acc;
              tensorforge::intel_esimd::simd<float, 16> v740_acc{};
              tensorforge::intel_esimd::simd<float, 16> v742_data = tensorforge::slmLoad<float, 16>(s2 + (84_i32));
              v740_acc += ((static_cast<float>(v742_data[0])) * v537_data);
              v740_acc += ((static_cast<float>(v742_data[1])) * v538_data);
              v740_acc += ((static_cast<float>(v742_data[2])) * v539_data);
              v740_acc += ((static_cast<float>(v742_data[3])) * v540_data);
              v740_acc += ((static_cast<float>(v742_data[4])) * v541_data);
              v740_acc += ((static_cast<float>(v742_data[5])) * v542_data);
              v740_acc += ((static_cast<float>(v742_data[6])) * v543_data);
              v740_acc += ((static_cast<float>(v742_data[7])) * v544_data);
              v740_acc += ((static_cast<float>(v742_data[8])) * v545_data);
              v740_acc += ((static_cast<float>(v742_data[9])) * v546_data);
              v740_acc += ((static_cast<float>(v742_data[10])) * v547_data);
              v740_acc += ((static_cast<float>(v742_data[11])) * v548_data);
              ir5.template select<16, 1>(112) = v740_acc;
              // r5 = ir5 + r3
              #pragma unroll
              for (int32_t v767_n1 = 0; v767_n1 < 8; ++v767_n1) {
                int32_t v768_a = v767_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v770_data(ir5.template select<12, 1>(v768_a));
                tensorforge::intel_esimd::simd<float, 12> v771_data(r3.template select<12, 1>(v768_a));
                r5.template select<12, 1>(v768_a) = (v771_data + v770_data);
              }
              // s3 = load{g>s}(glb_m8[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v782_ld;
              v782_ld.copy_from(glb_m8 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s3 + (0 + 0 + 4 * 0 + 0), v782_ld);
              tensorforge::intel_esimd::simd<float, 32> v783_ld;
              v783_ld.copy_from(glb_m8 + (0 + 0 + 2 * 0 + 64));
              tensorforge::slmStore<float, 32>(s3 + (0 + 0 + 2 * 0 + 64), v783_ld);
              tensorforge::intel_esimd::simd<float, 128> r7(0.0f);
              // ir7 = +(r6 * s3)
              // [(0, 12), (0, 8)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 128> ir7(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v786_data(r6.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v787_data(r6.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v788_data(r6.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v789_data(r6.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v790_data(r6.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v791_data(r6.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v792_data(r6.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v793_data(r6.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v794_data(r6.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v795_data(r6.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v796_data(r6.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v797_data(r6.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v798_acc{};
              tensorforge::intel_esimd::simd<float, 16> v802_data = tensorforge::slmLoad<float, 16>(s3 + (0_i32));
              v798_acc += ((static_cast<float>(v802_data[0])) * v786_data);
              v798_acc += ((static_cast<float>(v802_data[1])) * v787_data);
              v798_acc += ((static_cast<float>(v802_data[2])) * v788_data);
              v798_acc += ((static_cast<float>(v802_data[3])) * v789_data);
              v798_acc += ((static_cast<float>(v802_data[4])) * v790_data);
              v798_acc += ((static_cast<float>(v802_data[5])) * v791_data);
              v798_acc += ((static_cast<float>(v802_data[6])) * v792_data);
              v798_acc += ((static_cast<float>(v802_data[7])) * v793_data);
              v798_acc += ((static_cast<float>(v802_data[8])) * v794_data);
              v798_acc += ((static_cast<float>(v802_data[9])) * v795_data);
              v798_acc += ((static_cast<float>(v802_data[10])) * v796_data);
              v798_acc += ((static_cast<float>(v802_data[11])) * v797_data);
              ir7.template select<16, 1>(0) = v798_acc;
              tensorforge::intel_esimd::simd<float, 16> v827_acc{};
              tensorforge::intel_esimd::simd<float, 16> v829_data = tensorforge::slmLoad<float, 16>(s3 + (12_i32));
              v827_acc += ((static_cast<float>(v829_data[0])) * v786_data);
              v827_acc += ((static_cast<float>(v829_data[1])) * v787_data);
              v827_acc += ((static_cast<float>(v829_data[2])) * v788_data);
              v827_acc += ((static_cast<float>(v829_data[3])) * v789_data);
              v827_acc += ((static_cast<float>(v829_data[4])) * v790_data);
              v827_acc += ((static_cast<float>(v829_data[5])) * v791_data);
              v827_acc += ((static_cast<float>(v829_data[6])) * v792_data);
              v827_acc += ((static_cast<float>(v829_data[7])) * v793_data);
              v827_acc += ((static_cast<float>(v829_data[8])) * v794_data);
              v827_acc += ((static_cast<float>(v829_data[9])) * v795_data);
              v827_acc += ((static_cast<float>(v829_data[10])) * v796_data);
              v827_acc += ((static_cast<float>(v829_data[11])) * v797_data);
              ir7.template select<16, 1>(16) = v827_acc;
              tensorforge::intel_esimd::simd<float, 16> v854_acc{};
              tensorforge::intel_esimd::simd<float, 16> v856_data = tensorforge::slmLoad<float, 16>(s3 + (24_i32));
              v854_acc += ((static_cast<float>(v856_data[0])) * v786_data);
              v854_acc += ((static_cast<float>(v856_data[1])) * v787_data);
              v854_acc += ((static_cast<float>(v856_data[2])) * v788_data);
              v854_acc += ((static_cast<float>(v856_data[3])) * v789_data);
              v854_acc += ((static_cast<float>(v856_data[4])) * v790_data);
              v854_acc += ((static_cast<float>(v856_data[5])) * v791_data);
              v854_acc += ((static_cast<float>(v856_data[6])) * v792_data);
              v854_acc += ((static_cast<float>(v856_data[7])) * v793_data);
              v854_acc += ((static_cast<float>(v856_data[8])) * v794_data);
              v854_acc += ((static_cast<float>(v856_data[9])) * v795_data);
              v854_acc += ((static_cast<float>(v856_data[10])) * v796_data);
              v854_acc += ((static_cast<float>(v856_data[11])) * v797_data);
              ir7.template select<16, 1>(32) = v854_acc;
              tensorforge::intel_esimd::simd<float, 16> v881_acc{};
              tensorforge::intel_esimd::simd<float, 16> v883_data = tensorforge::slmLoad<float, 16>(s3 + (36_i32));
              v881_acc += ((static_cast<float>(v883_data[0])) * v786_data);
              v881_acc += ((static_cast<float>(v883_data[1])) * v787_data);
              v881_acc += ((static_cast<float>(v883_data[2])) * v788_data);
              v881_acc += ((static_cast<float>(v883_data[3])) * v789_data);
              v881_acc += ((static_cast<float>(v883_data[4])) * v790_data);
              v881_acc += ((static_cast<float>(v883_data[5])) * v791_data);
              v881_acc += ((static_cast<float>(v883_data[6])) * v792_data);
              v881_acc += ((static_cast<float>(v883_data[7])) * v793_data);
              v881_acc += ((static_cast<float>(v883_data[8])) * v794_data);
              v881_acc += ((static_cast<float>(v883_data[9])) * v795_data);
              v881_acc += ((static_cast<float>(v883_data[10])) * v796_data);
              v881_acc += ((static_cast<float>(v883_data[11])) * v797_data);
              ir7.template select<16, 1>(48) = v881_acc;
              tensorforge::intel_esimd::simd<float, 16> v908_acc{};
              tensorforge::intel_esimd::simd<float, 16> v910_data = tensorforge::slmLoad<float, 16>(s3 + (48_i32));
              v908_acc += ((static_cast<float>(v910_data[0])) * v786_data);
              v908_acc += ((static_cast<float>(v910_data[1])) * v787_data);
              v908_acc += ((static_cast<float>(v910_data[2])) * v788_data);
              v908_acc += ((static_cast<float>(v910_data[3])) * v789_data);
              v908_acc += ((static_cast<float>(v910_data[4])) * v790_data);
              v908_acc += ((static_cast<float>(v910_data[5])) * v791_data);
              v908_acc += ((static_cast<float>(v910_data[6])) * v792_data);
              v908_acc += ((static_cast<float>(v910_data[7])) * v793_data);
              v908_acc += ((static_cast<float>(v910_data[8])) * v794_data);
              v908_acc += ((static_cast<float>(v910_data[9])) * v795_data);
              v908_acc += ((static_cast<float>(v910_data[10])) * v796_data);
              v908_acc += ((static_cast<float>(v910_data[11])) * v797_data);
              ir7.template select<16, 1>(64) = v908_acc;
              tensorforge::intel_esimd::simd<float, 16> v935_acc{};
              tensorforge::intel_esimd::simd<float, 16> v937_data = tensorforge::slmLoad<float, 16>(s3 + (60_i32));
              v935_acc += ((static_cast<float>(v937_data[0])) * v786_data);
              v935_acc += ((static_cast<float>(v937_data[1])) * v787_data);
              v935_acc += ((static_cast<float>(v937_data[2])) * v788_data);
              v935_acc += ((static_cast<float>(v937_data[3])) * v789_data);
              v935_acc += ((static_cast<float>(v937_data[4])) * v790_data);
              v935_acc += ((static_cast<float>(v937_data[5])) * v791_data);
              v935_acc += ((static_cast<float>(v937_data[6])) * v792_data);
              v935_acc += ((static_cast<float>(v937_data[7])) * v793_data);
              v935_acc += ((static_cast<float>(v937_data[8])) * v794_data);
              v935_acc += ((static_cast<float>(v937_data[9])) * v795_data);
              v935_acc += ((static_cast<float>(v937_data[10])) * v796_data);
              v935_acc += ((static_cast<float>(v937_data[11])) * v797_data);
              ir7.template select<16, 1>(80) = v935_acc;
              tensorforge::intel_esimd::simd<float, 16> v962_acc{};
              tensorforge::intel_esimd::simd<float, 16> v964_data = tensorforge::slmLoad<float, 16>(s3 + (72_i32));
              v962_acc += ((static_cast<float>(v964_data[0])) * v786_data);
              v962_acc += ((static_cast<float>(v964_data[1])) * v787_data);
              v962_acc += ((static_cast<float>(v964_data[2])) * v788_data);
              v962_acc += ((static_cast<float>(v964_data[3])) * v789_data);
              v962_acc += ((static_cast<float>(v964_data[4])) * v790_data);
              v962_acc += ((static_cast<float>(v964_data[5])) * v791_data);
              v962_acc += ((static_cast<float>(v964_data[6])) * v792_data);
              v962_acc += ((static_cast<float>(v964_data[7])) * v793_data);
              v962_acc += ((static_cast<float>(v964_data[8])) * v794_data);
              v962_acc += ((static_cast<float>(v964_data[9])) * v795_data);
              v962_acc += ((static_cast<float>(v964_data[10])) * v796_data);
              v962_acc += ((static_cast<float>(v964_data[11])) * v797_data);
              ir7.template select<16, 1>(96) = v962_acc;
              tensorforge::intel_esimd::simd<float, 16> v989_acc{};
              tensorforge::intel_esimd::simd<float, 16> v991_data = tensorforge::slmLoad<float, 16>(s3 + (84_i32));
              v989_acc += ((static_cast<float>(v991_data[0])) * v786_data);
              v989_acc += ((static_cast<float>(v991_data[1])) * v787_data);
              v989_acc += ((static_cast<float>(v991_data[2])) * v788_data);
              v989_acc += ((static_cast<float>(v991_data[3])) * v789_data);
              v989_acc += ((static_cast<float>(v991_data[4])) * v790_data);
              v989_acc += ((static_cast<float>(v991_data[5])) * v791_data);
              v989_acc += ((static_cast<float>(v991_data[6])) * v792_data);
              v989_acc += ((static_cast<float>(v991_data[7])) * v793_data);
              v989_acc += ((static_cast<float>(v991_data[8])) * v794_data);
              v989_acc += ((static_cast<float>(v991_data[9])) * v795_data);
              v989_acc += ((static_cast<float>(v991_data[10])) * v796_data);
              v989_acc += ((static_cast<float>(v991_data[11])) * v797_data);
              ir7.template select<16, 1>(112) = v989_acc;
              // r7 = ir7 + r5
              #pragma unroll
              for (int32_t v1016_n1 = 0; v1016_n1 < 8; ++v1016_n1) {
                int32_t v1017_a = v1016_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v1019_data(ir7.template select<12, 1>(v1017_a));
                tensorforge::intel_esimd::simd<float, 12> v1020_data(r5.template select<12, 1>(v1017_a));
                r7.template select<12, 1>(v1017_a) = (v1020_data + v1019_data);
              }
              // glb_m0 = store{r>g}(r7);
              #pragma unroll
              for (int32_t v1022_i1 = 0; v1022_i1 < 8; ++v1022_i1) {
                tensorforge::intel_esimd::simd<float, 12> v1025_data(r7.template select<12, 1>((v1022_i1 * 16)));
                v1025_data.copy_to(glb_m0 + ((v1022_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}

