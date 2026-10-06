// === base name ===
kernel_77a8aa689eb4362b

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_77a8aa689eb4362b = {{1, 16, 1}, 16, 12, 1, 16, 7168, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_77a8aa689eb4362b(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_77a8aa689eb4362b(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_77a8aa689eb4362b(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_77a8aa689eb4362b(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_77a8aa689eb4362b(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_77a8aa689eb4362b(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, m6, m6_extraOffset, m7, m7_extraOffset, m8, m8_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_77a8aa689eb4362b(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0) {
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
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (96);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s2 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s3 = localShrMem0 + (0);
          for (size_t v14_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v14_batchId0 < numElements0; v14_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v15_ahead1 = v14_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v17_batchId1 = (v15_ahead1 < numElements0) ? v15_ahead1 : v14_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v14_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v14_batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v14_batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v14_batchId0 * 96 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v14_batchId0 * 144 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v14_batchId0 * 96 + 0 + m4_extraOffset];
              const float *const __restrict__ glb_m5 = &m5[v14_batchId0 * 144 + 0 + m5_extraOffset];
              const float *const __restrict__ glb_m6 = &m6[v14_batchId0 * 96 + 0 + m6_extraOffset];
              const float *const __restrict__ glb_m7 = &m7[v14_batchId0 * 144 + 0 + m7_extraOffset];
              const float *const __restrict__ glb_m8 = &m8[v14_batchId0 * 96 + 0 + m8_extraOffset];
              tensorforge::intel_esimd::simd<float, 192> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v32_i1 = 0; v32_i1 < 12; ++v32_i1) {
                tensorforge::intel_esimd::simd<float, 12> v37_data;
                v37_data.copy_from(glb_m1 + ((v32_i1 * 12)));
                r0.template select<12, 1>((v32_i1 * 16)) = v37_data;
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v40_ld;
              v40_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v40_ld);
              tensorforge::intel_esimd::simd<float, 32> v41_ld;
              v41_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 64));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 2 * 0 + 64), v41_ld);
              // wait(r0 = load{g>r}(glb_m1););
              tensorforge::intel_esimd::simd<float, 192> r2(0.0f);
              // r2 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v43_i1 = 0; v43_i1 < 12; ++v43_i1) {
                tensorforge::intel_esimd::simd<float, 12> v48_data;
                v48_data.copy_from(glb_m3 + ((v43_i1 * 12)));
                r2.template select<12, 1>((v43_i1 * 16)) = v48_data;
              }
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 128> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 12), (0, 8)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 128> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v53_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v54_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v55_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v56_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v57_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v58_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v59_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v60_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v61_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v62_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v63_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v64_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v65_acc{};
              tensorforge::intel_esimd::simd<float, 16> v69_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v65_acc += ((static_cast<float>(v69_data[0])) * v53_data);
              v65_acc += ((static_cast<float>(v69_data[1])) * v54_data);
              v65_acc += ((static_cast<float>(v69_data[2])) * v55_data);
              v65_acc += ((static_cast<float>(v69_data[3])) * v56_data);
              v65_acc += ((static_cast<float>(v69_data[4])) * v57_data);
              v65_acc += ((static_cast<float>(v69_data[5])) * v58_data);
              v65_acc += ((static_cast<float>(v69_data[6])) * v59_data);
              v65_acc += ((static_cast<float>(v69_data[7])) * v60_data);
              v65_acc += ((static_cast<float>(v69_data[8])) * v61_data);
              v65_acc += ((static_cast<float>(v69_data[9])) * v62_data);
              v65_acc += ((static_cast<float>(v69_data[10])) * v63_data);
              v65_acc += ((static_cast<float>(v69_data[11])) * v64_data);
              ir1.template select<16, 1>(0) = v65_acc;
              tensorforge::intel_esimd::simd<float, 16> v94_acc{};
              tensorforge::intel_esimd::simd<float, 16> v96_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              v94_acc += ((static_cast<float>(v96_data[0])) * v53_data);
              v94_acc += ((static_cast<float>(v96_data[1])) * v54_data);
              v94_acc += ((static_cast<float>(v96_data[2])) * v55_data);
              v94_acc += ((static_cast<float>(v96_data[3])) * v56_data);
              v94_acc += ((static_cast<float>(v96_data[4])) * v57_data);
              v94_acc += ((static_cast<float>(v96_data[5])) * v58_data);
              v94_acc += ((static_cast<float>(v96_data[6])) * v59_data);
              v94_acc += ((static_cast<float>(v96_data[7])) * v60_data);
              v94_acc += ((static_cast<float>(v96_data[8])) * v61_data);
              v94_acc += ((static_cast<float>(v96_data[9])) * v62_data);
              v94_acc += ((static_cast<float>(v96_data[10])) * v63_data);
              v94_acc += ((static_cast<float>(v96_data[11])) * v64_data);
              ir1.template select<16, 1>(16) = v94_acc;
              tensorforge::intel_esimd::simd<float, 16> v121_acc{};
              tensorforge::intel_esimd::simd<float, 16> v123_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              v121_acc += ((static_cast<float>(v123_data[0])) * v53_data);
              v121_acc += ((static_cast<float>(v123_data[1])) * v54_data);
              v121_acc += ((static_cast<float>(v123_data[2])) * v55_data);
              v121_acc += ((static_cast<float>(v123_data[3])) * v56_data);
              v121_acc += ((static_cast<float>(v123_data[4])) * v57_data);
              v121_acc += ((static_cast<float>(v123_data[5])) * v58_data);
              v121_acc += ((static_cast<float>(v123_data[6])) * v59_data);
              v121_acc += ((static_cast<float>(v123_data[7])) * v60_data);
              v121_acc += ((static_cast<float>(v123_data[8])) * v61_data);
              v121_acc += ((static_cast<float>(v123_data[9])) * v62_data);
              v121_acc += ((static_cast<float>(v123_data[10])) * v63_data);
              v121_acc += ((static_cast<float>(v123_data[11])) * v64_data);
              ir1.template select<16, 1>(32) = v121_acc;
              tensorforge::intel_esimd::simd<float, 16> v148_acc{};
              tensorforge::intel_esimd::simd<float, 16> v150_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              v148_acc += ((static_cast<float>(v150_data[0])) * v53_data);
              v148_acc += ((static_cast<float>(v150_data[1])) * v54_data);
              v148_acc += ((static_cast<float>(v150_data[2])) * v55_data);
              v148_acc += ((static_cast<float>(v150_data[3])) * v56_data);
              v148_acc += ((static_cast<float>(v150_data[4])) * v57_data);
              v148_acc += ((static_cast<float>(v150_data[5])) * v58_data);
              v148_acc += ((static_cast<float>(v150_data[6])) * v59_data);
              v148_acc += ((static_cast<float>(v150_data[7])) * v60_data);
              v148_acc += ((static_cast<float>(v150_data[8])) * v61_data);
              v148_acc += ((static_cast<float>(v150_data[9])) * v62_data);
              v148_acc += ((static_cast<float>(v150_data[10])) * v63_data);
              v148_acc += ((static_cast<float>(v150_data[11])) * v64_data);
              ir1.template select<16, 1>(48) = v148_acc;
              tensorforge::intel_esimd::simd<float, 16> v175_acc{};
              tensorforge::intel_esimd::simd<float, 16> v177_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v175_acc += ((static_cast<float>(v177_data[0])) * v53_data);
              v175_acc += ((static_cast<float>(v177_data[1])) * v54_data);
              v175_acc += ((static_cast<float>(v177_data[2])) * v55_data);
              v175_acc += ((static_cast<float>(v177_data[3])) * v56_data);
              v175_acc += ((static_cast<float>(v177_data[4])) * v57_data);
              v175_acc += ((static_cast<float>(v177_data[5])) * v58_data);
              v175_acc += ((static_cast<float>(v177_data[6])) * v59_data);
              v175_acc += ((static_cast<float>(v177_data[7])) * v60_data);
              v175_acc += ((static_cast<float>(v177_data[8])) * v61_data);
              v175_acc += ((static_cast<float>(v177_data[9])) * v62_data);
              v175_acc += ((static_cast<float>(v177_data[10])) * v63_data);
              v175_acc += ((static_cast<float>(v177_data[11])) * v64_data);
              ir1.template select<16, 1>(64) = v175_acc;
              tensorforge::intel_esimd::simd<float, 16> v202_acc{};
              tensorforge::intel_esimd::simd<float, 16> v204_data = tensorforge::slmLoad<float, 16>(s0 + (60_i32));
              v202_acc += ((static_cast<float>(v204_data[0])) * v53_data);
              v202_acc += ((static_cast<float>(v204_data[1])) * v54_data);
              v202_acc += ((static_cast<float>(v204_data[2])) * v55_data);
              v202_acc += ((static_cast<float>(v204_data[3])) * v56_data);
              v202_acc += ((static_cast<float>(v204_data[4])) * v57_data);
              v202_acc += ((static_cast<float>(v204_data[5])) * v58_data);
              v202_acc += ((static_cast<float>(v204_data[6])) * v59_data);
              v202_acc += ((static_cast<float>(v204_data[7])) * v60_data);
              v202_acc += ((static_cast<float>(v204_data[8])) * v61_data);
              v202_acc += ((static_cast<float>(v204_data[9])) * v62_data);
              v202_acc += ((static_cast<float>(v204_data[10])) * v63_data);
              v202_acc += ((static_cast<float>(v204_data[11])) * v64_data);
              ir1.template select<16, 1>(80) = v202_acc;
              tensorforge::intel_esimd::simd<float, 16> v229_acc{};
              tensorforge::intel_esimd::simd<float, 16> v231_data = tensorforge::slmLoad<float, 16>(s0 + (72_i32));
              v229_acc += ((static_cast<float>(v231_data[0])) * v53_data);
              v229_acc += ((static_cast<float>(v231_data[1])) * v54_data);
              v229_acc += ((static_cast<float>(v231_data[2])) * v55_data);
              v229_acc += ((static_cast<float>(v231_data[3])) * v56_data);
              v229_acc += ((static_cast<float>(v231_data[4])) * v57_data);
              v229_acc += ((static_cast<float>(v231_data[5])) * v58_data);
              v229_acc += ((static_cast<float>(v231_data[6])) * v59_data);
              v229_acc += ((static_cast<float>(v231_data[7])) * v60_data);
              v229_acc += ((static_cast<float>(v231_data[8])) * v61_data);
              v229_acc += ((static_cast<float>(v231_data[9])) * v62_data);
              v229_acc += ((static_cast<float>(v231_data[10])) * v63_data);
              v229_acc += ((static_cast<float>(v231_data[11])) * v64_data);
              ir1.template select<16, 1>(96) = v229_acc;
              tensorforge::intel_esimd::simd<float, 16> v256_acc{};
              tensorforge::intel_esimd::simd<float, 16> v258_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v256_acc += ((static_cast<float>(v258_data[0])) * v53_data);
              v256_acc += ((static_cast<float>(v258_data[1])) * v54_data);
              v256_acc += ((static_cast<float>(v258_data[2])) * v55_data);
              v256_acc += ((static_cast<float>(v258_data[3])) * v56_data);
              v256_acc += ((static_cast<float>(v258_data[4])) * v57_data);
              v256_acc += ((static_cast<float>(v258_data[5])) * v58_data);
              v256_acc += ((static_cast<float>(v258_data[6])) * v59_data);
              v256_acc += ((static_cast<float>(v258_data[7])) * v60_data);
              v256_acc += ((static_cast<float>(v258_data[8])) * v61_data);
              v256_acc += ((static_cast<float>(v258_data[9])) * v62_data);
              v256_acc += ((static_cast<float>(v258_data[10])) * v63_data);
              v256_acc += ((static_cast<float>(v258_data[11])) * v64_data);
              ir1.template select<16, 1>(112) = v256_acc;
              // r1 = ir1
              #pragma unroll
              for (int32_t v283_n1 = 0; v283_n1 < 8; ++v283_n1) {
                int32_t v284_a = v283_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v286_data(ir1.template select<12, 1>(v284_a));
                r1.template select<12, 1>(v284_a) = v286_data;
              }
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v287_ld;
              v287_ld.copy_from(glb_m4 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 4 * 0 + 0), v287_ld);
              tensorforge::intel_esimd::simd<float, 32> v288_ld;
              v288_ld.copy_from(glb_m4 + (0 + 0 + 2 * 0 + 64));
              tensorforge::slmStore<float, 32>(s1 + (0 + 0 + 2 * 0 + 64), v288_ld);
              // wait(r2 = load{g>r}(glb_m3););
              tensorforge::intel_esimd::simd<float, 192> r4(0.0f);
              // r4 = load{g>r}(glb_m5);
              #pragma unroll
              for (int32_t v290_i1 = 0; v290_i1 < 12; ++v290_i1) {
                tensorforge::intel_esimd::simd<float, 12> v295_data;
                v295_data.copy_from(glb_m5 + ((v290_i1 * 12)));
                r4.template select<12, 1>((v290_i1 * 16)) = v295_data;
              }
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              tensorforge::intel_esimd::simd<float, 128> r3(0.0f);
              // ir3 = +(r2 * s1)
              // [(0, 12), (0, 8)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 128> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v300_data(r2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v301_data(r2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v302_data(r2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v303_data(r2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v304_data(r2.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v305_data(r2.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v306_data(r2.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v307_data(r2.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v308_data(r2.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v309_data(r2.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v310_data(r2.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v311_data(r2.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v312_acc{};
              tensorforge::intel_esimd::simd<float, 16> v316_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v312_acc += ((static_cast<float>(v316_data[0])) * v300_data);
              v312_acc += ((static_cast<float>(v316_data[1])) * v301_data);
              v312_acc += ((static_cast<float>(v316_data[2])) * v302_data);
              v312_acc += ((static_cast<float>(v316_data[3])) * v303_data);
              v312_acc += ((static_cast<float>(v316_data[4])) * v304_data);
              v312_acc += ((static_cast<float>(v316_data[5])) * v305_data);
              v312_acc += ((static_cast<float>(v316_data[6])) * v306_data);
              v312_acc += ((static_cast<float>(v316_data[7])) * v307_data);
              v312_acc += ((static_cast<float>(v316_data[8])) * v308_data);
              v312_acc += ((static_cast<float>(v316_data[9])) * v309_data);
              v312_acc += ((static_cast<float>(v316_data[10])) * v310_data);
              v312_acc += ((static_cast<float>(v316_data[11])) * v311_data);
              ir3.template select<16, 1>(0) = v312_acc;
              tensorforge::intel_esimd::simd<float, 16> v341_acc{};
              tensorforge::intel_esimd::simd<float, 16> v343_data = tensorforge::slmLoad<float, 16>(s1 + (12_i32));
              v341_acc += ((static_cast<float>(v343_data[0])) * v300_data);
              v341_acc += ((static_cast<float>(v343_data[1])) * v301_data);
              v341_acc += ((static_cast<float>(v343_data[2])) * v302_data);
              v341_acc += ((static_cast<float>(v343_data[3])) * v303_data);
              v341_acc += ((static_cast<float>(v343_data[4])) * v304_data);
              v341_acc += ((static_cast<float>(v343_data[5])) * v305_data);
              v341_acc += ((static_cast<float>(v343_data[6])) * v306_data);
              v341_acc += ((static_cast<float>(v343_data[7])) * v307_data);
              v341_acc += ((static_cast<float>(v343_data[8])) * v308_data);
              v341_acc += ((static_cast<float>(v343_data[9])) * v309_data);
              v341_acc += ((static_cast<float>(v343_data[10])) * v310_data);
              v341_acc += ((static_cast<float>(v343_data[11])) * v311_data);
              ir3.template select<16, 1>(16) = v341_acc;
              tensorforge::intel_esimd::simd<float, 16> v368_acc{};
              tensorforge::intel_esimd::simd<float, 16> v370_data = tensorforge::slmLoad<float, 16>(s1 + (24_i32));
              v368_acc += ((static_cast<float>(v370_data[0])) * v300_data);
              v368_acc += ((static_cast<float>(v370_data[1])) * v301_data);
              v368_acc += ((static_cast<float>(v370_data[2])) * v302_data);
              v368_acc += ((static_cast<float>(v370_data[3])) * v303_data);
              v368_acc += ((static_cast<float>(v370_data[4])) * v304_data);
              v368_acc += ((static_cast<float>(v370_data[5])) * v305_data);
              v368_acc += ((static_cast<float>(v370_data[6])) * v306_data);
              v368_acc += ((static_cast<float>(v370_data[7])) * v307_data);
              v368_acc += ((static_cast<float>(v370_data[8])) * v308_data);
              v368_acc += ((static_cast<float>(v370_data[9])) * v309_data);
              v368_acc += ((static_cast<float>(v370_data[10])) * v310_data);
              v368_acc += ((static_cast<float>(v370_data[11])) * v311_data);
              ir3.template select<16, 1>(32) = v368_acc;
              tensorforge::intel_esimd::simd<float, 16> v395_acc{};
              tensorforge::intel_esimd::simd<float, 16> v397_data = tensorforge::slmLoad<float, 16>(s1 + (36_i32));
              v395_acc += ((static_cast<float>(v397_data[0])) * v300_data);
              v395_acc += ((static_cast<float>(v397_data[1])) * v301_data);
              v395_acc += ((static_cast<float>(v397_data[2])) * v302_data);
              v395_acc += ((static_cast<float>(v397_data[3])) * v303_data);
              v395_acc += ((static_cast<float>(v397_data[4])) * v304_data);
              v395_acc += ((static_cast<float>(v397_data[5])) * v305_data);
              v395_acc += ((static_cast<float>(v397_data[6])) * v306_data);
              v395_acc += ((static_cast<float>(v397_data[7])) * v307_data);
              v395_acc += ((static_cast<float>(v397_data[8])) * v308_data);
              v395_acc += ((static_cast<float>(v397_data[9])) * v309_data);
              v395_acc += ((static_cast<float>(v397_data[10])) * v310_data);
              v395_acc += ((static_cast<float>(v397_data[11])) * v311_data);
              ir3.template select<16, 1>(48) = v395_acc;
              tensorforge::intel_esimd::simd<float, 16> v422_acc{};
              tensorforge::intel_esimd::simd<float, 16> v424_data = tensorforge::slmLoad<float, 16>(s1 + (48_i32));
              v422_acc += ((static_cast<float>(v424_data[0])) * v300_data);
              v422_acc += ((static_cast<float>(v424_data[1])) * v301_data);
              v422_acc += ((static_cast<float>(v424_data[2])) * v302_data);
              v422_acc += ((static_cast<float>(v424_data[3])) * v303_data);
              v422_acc += ((static_cast<float>(v424_data[4])) * v304_data);
              v422_acc += ((static_cast<float>(v424_data[5])) * v305_data);
              v422_acc += ((static_cast<float>(v424_data[6])) * v306_data);
              v422_acc += ((static_cast<float>(v424_data[7])) * v307_data);
              v422_acc += ((static_cast<float>(v424_data[8])) * v308_data);
              v422_acc += ((static_cast<float>(v424_data[9])) * v309_data);
              v422_acc += ((static_cast<float>(v424_data[10])) * v310_data);
              v422_acc += ((static_cast<float>(v424_data[11])) * v311_data);
              ir3.template select<16, 1>(64) = v422_acc;
              tensorforge::intel_esimd::simd<float, 16> v449_acc{};
              tensorforge::intel_esimd::simd<float, 16> v451_data = tensorforge::slmLoad<float, 16>(s1 + (60_i32));
              v449_acc += ((static_cast<float>(v451_data[0])) * v300_data);
              v449_acc += ((static_cast<float>(v451_data[1])) * v301_data);
              v449_acc += ((static_cast<float>(v451_data[2])) * v302_data);
              v449_acc += ((static_cast<float>(v451_data[3])) * v303_data);
              v449_acc += ((static_cast<float>(v451_data[4])) * v304_data);
              v449_acc += ((static_cast<float>(v451_data[5])) * v305_data);
              v449_acc += ((static_cast<float>(v451_data[6])) * v306_data);
              v449_acc += ((static_cast<float>(v451_data[7])) * v307_data);
              v449_acc += ((static_cast<float>(v451_data[8])) * v308_data);
              v449_acc += ((static_cast<float>(v451_data[9])) * v309_data);
              v449_acc += ((static_cast<float>(v451_data[10])) * v310_data);
              v449_acc += ((static_cast<float>(v451_data[11])) * v311_data);
              ir3.template select<16, 1>(80) = v449_acc;
              tensorforge::intel_esimd::simd<float, 16> v476_acc{};
              tensorforge::intel_esimd::simd<float, 16> v478_data = tensorforge::slmLoad<float, 16>(s1 + (72_i32));
              v476_acc += ((static_cast<float>(v478_data[0])) * v300_data);
              v476_acc += ((static_cast<float>(v478_data[1])) * v301_data);
              v476_acc += ((static_cast<float>(v478_data[2])) * v302_data);
              v476_acc += ((static_cast<float>(v478_data[3])) * v303_data);
              v476_acc += ((static_cast<float>(v478_data[4])) * v304_data);
              v476_acc += ((static_cast<float>(v478_data[5])) * v305_data);
              v476_acc += ((static_cast<float>(v478_data[6])) * v306_data);
              v476_acc += ((static_cast<float>(v478_data[7])) * v307_data);
              v476_acc += ((static_cast<float>(v478_data[8])) * v308_data);
              v476_acc += ((static_cast<float>(v478_data[9])) * v309_data);
              v476_acc += ((static_cast<float>(v478_data[10])) * v310_data);
              v476_acc += ((static_cast<float>(v478_data[11])) * v311_data);
              ir3.template select<16, 1>(96) = v476_acc;
              tensorforge::intel_esimd::simd<float, 16> v503_acc{};
              tensorforge::intel_esimd::simd<float, 16> v505_data = tensorforge::slmLoad<float, 16>(s1 + (84_i32));
              v503_acc += ((static_cast<float>(v505_data[0])) * v300_data);
              v503_acc += ((static_cast<float>(v505_data[1])) * v301_data);
              v503_acc += ((static_cast<float>(v505_data[2])) * v302_data);
              v503_acc += ((static_cast<float>(v505_data[3])) * v303_data);
              v503_acc += ((static_cast<float>(v505_data[4])) * v304_data);
              v503_acc += ((static_cast<float>(v505_data[5])) * v305_data);
              v503_acc += ((static_cast<float>(v505_data[6])) * v306_data);
              v503_acc += ((static_cast<float>(v505_data[7])) * v307_data);
              v503_acc += ((static_cast<float>(v505_data[8])) * v308_data);
              v503_acc += ((static_cast<float>(v505_data[9])) * v309_data);
              v503_acc += ((static_cast<float>(v505_data[10])) * v310_data);
              v503_acc += ((static_cast<float>(v505_data[11])) * v311_data);
              ir3.template select<16, 1>(112) = v503_acc;
              // r3 = ir3 + r1
              #pragma unroll
              for (int32_t v530_n1 = 0; v530_n1 < 8; ++v530_n1) {
                int32_t v531_a = v530_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v533_data(ir3.template select<12, 1>(v531_a));
                tensorforge::intel_esimd::simd<float, 12> v534_data(r1.template select<12, 1>(v531_a));
                r3.template select<12, 1>(v531_a) = (v534_data + v533_data);
              }
              // s2 = load{g>s}(glb_m6[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v536_ld;
              v536_ld.copy_from(glb_m6 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s2 + (0 + 0 + 4 * 0 + 0), v536_ld);
              tensorforge::intel_esimd::simd<float, 32> v537_ld;
              v537_ld.copy_from(glb_m6 + (0 + 0 + 2 * 0 + 64));
              tensorforge::slmStore<float, 32>(s2 + (0 + 0 + 2 * 0 + 64), v537_ld);
              // wait(r4 = load{g>r}(glb_m5););
              tensorforge::intel_esimd::simd<float, 192> r6(0.0f);
              // r6 = load{g>r}(glb_m7);
              #pragma unroll
              for (int32_t v539_i1 = 0; v539_i1 < 12; ++v539_i1) {
                tensorforge::intel_esimd::simd<float, 12> v544_data;
                v544_data.copy_from(glb_m7 + ((v539_i1 * 12)));
                r6.template select<12, 1>((v539_i1 * 16)) = v544_data;
              }
              // wait(s2 = load{g>s}(glb_m6[0, 1]));
              tensorforge::intel_esimd::simd<float, 128> r5(0.0f);
              // ir5 = +(r4 * s2)
              // [(0, 12), (0, 8)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 128> ir5(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v549_data(r4.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v550_data(r4.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v551_data(r4.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v552_data(r4.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v553_data(r4.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v554_data(r4.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v555_data(r4.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v556_data(r4.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v557_data(r4.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v558_data(r4.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v559_data(r4.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v560_data(r4.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v561_acc{};
              tensorforge::intel_esimd::simd<float, 16> v565_data = tensorforge::slmLoad<float, 16>(s2 + (0_i32));
              v561_acc += ((static_cast<float>(v565_data[0])) * v549_data);
              v561_acc += ((static_cast<float>(v565_data[1])) * v550_data);
              v561_acc += ((static_cast<float>(v565_data[2])) * v551_data);
              v561_acc += ((static_cast<float>(v565_data[3])) * v552_data);
              v561_acc += ((static_cast<float>(v565_data[4])) * v553_data);
              v561_acc += ((static_cast<float>(v565_data[5])) * v554_data);
              v561_acc += ((static_cast<float>(v565_data[6])) * v555_data);
              v561_acc += ((static_cast<float>(v565_data[7])) * v556_data);
              v561_acc += ((static_cast<float>(v565_data[8])) * v557_data);
              v561_acc += ((static_cast<float>(v565_data[9])) * v558_data);
              v561_acc += ((static_cast<float>(v565_data[10])) * v559_data);
              v561_acc += ((static_cast<float>(v565_data[11])) * v560_data);
              ir5.template select<16, 1>(0) = v561_acc;
              tensorforge::intel_esimd::simd<float, 16> v590_acc{};
              tensorforge::intel_esimd::simd<float, 16> v592_data = tensorforge::slmLoad<float, 16>(s2 + (12_i32));
              v590_acc += ((static_cast<float>(v592_data[0])) * v549_data);
              v590_acc += ((static_cast<float>(v592_data[1])) * v550_data);
              v590_acc += ((static_cast<float>(v592_data[2])) * v551_data);
              v590_acc += ((static_cast<float>(v592_data[3])) * v552_data);
              v590_acc += ((static_cast<float>(v592_data[4])) * v553_data);
              v590_acc += ((static_cast<float>(v592_data[5])) * v554_data);
              v590_acc += ((static_cast<float>(v592_data[6])) * v555_data);
              v590_acc += ((static_cast<float>(v592_data[7])) * v556_data);
              v590_acc += ((static_cast<float>(v592_data[8])) * v557_data);
              v590_acc += ((static_cast<float>(v592_data[9])) * v558_data);
              v590_acc += ((static_cast<float>(v592_data[10])) * v559_data);
              v590_acc += ((static_cast<float>(v592_data[11])) * v560_data);
              ir5.template select<16, 1>(16) = v590_acc;
              tensorforge::intel_esimd::simd<float, 16> v617_acc{};
              tensorforge::intel_esimd::simd<float, 16> v619_data = tensorforge::slmLoad<float, 16>(s2 + (24_i32));
              v617_acc += ((static_cast<float>(v619_data[0])) * v549_data);
              v617_acc += ((static_cast<float>(v619_data[1])) * v550_data);
              v617_acc += ((static_cast<float>(v619_data[2])) * v551_data);
              v617_acc += ((static_cast<float>(v619_data[3])) * v552_data);
              v617_acc += ((static_cast<float>(v619_data[4])) * v553_data);
              v617_acc += ((static_cast<float>(v619_data[5])) * v554_data);
              v617_acc += ((static_cast<float>(v619_data[6])) * v555_data);
              v617_acc += ((static_cast<float>(v619_data[7])) * v556_data);
              v617_acc += ((static_cast<float>(v619_data[8])) * v557_data);
              v617_acc += ((static_cast<float>(v619_data[9])) * v558_data);
              v617_acc += ((static_cast<float>(v619_data[10])) * v559_data);
              v617_acc += ((static_cast<float>(v619_data[11])) * v560_data);
              ir5.template select<16, 1>(32) = v617_acc;
              tensorforge::intel_esimd::simd<float, 16> v644_acc{};
              tensorforge::intel_esimd::simd<float, 16> v646_data = tensorforge::slmLoad<float, 16>(s2 + (36_i32));
              v644_acc += ((static_cast<float>(v646_data[0])) * v549_data);
              v644_acc += ((static_cast<float>(v646_data[1])) * v550_data);
              v644_acc += ((static_cast<float>(v646_data[2])) * v551_data);
              v644_acc += ((static_cast<float>(v646_data[3])) * v552_data);
              v644_acc += ((static_cast<float>(v646_data[4])) * v553_data);
              v644_acc += ((static_cast<float>(v646_data[5])) * v554_data);
              v644_acc += ((static_cast<float>(v646_data[6])) * v555_data);
              v644_acc += ((static_cast<float>(v646_data[7])) * v556_data);
              v644_acc += ((static_cast<float>(v646_data[8])) * v557_data);
              v644_acc += ((static_cast<float>(v646_data[9])) * v558_data);
              v644_acc += ((static_cast<float>(v646_data[10])) * v559_data);
              v644_acc += ((static_cast<float>(v646_data[11])) * v560_data);
              ir5.template select<16, 1>(48) = v644_acc;
              tensorforge::intel_esimd::simd<float, 16> v671_acc{};
              tensorforge::intel_esimd::simd<float, 16> v673_data = tensorforge::slmLoad<float, 16>(s2 + (48_i32));
              v671_acc += ((static_cast<float>(v673_data[0])) * v549_data);
              v671_acc += ((static_cast<float>(v673_data[1])) * v550_data);
              v671_acc += ((static_cast<float>(v673_data[2])) * v551_data);
              v671_acc += ((static_cast<float>(v673_data[3])) * v552_data);
              v671_acc += ((static_cast<float>(v673_data[4])) * v553_data);
              v671_acc += ((static_cast<float>(v673_data[5])) * v554_data);
              v671_acc += ((static_cast<float>(v673_data[6])) * v555_data);
              v671_acc += ((static_cast<float>(v673_data[7])) * v556_data);
              v671_acc += ((static_cast<float>(v673_data[8])) * v557_data);
              v671_acc += ((static_cast<float>(v673_data[9])) * v558_data);
              v671_acc += ((static_cast<float>(v673_data[10])) * v559_data);
              v671_acc += ((static_cast<float>(v673_data[11])) * v560_data);
              ir5.template select<16, 1>(64) = v671_acc;
              tensorforge::intel_esimd::simd<float, 16> v698_acc{};
              tensorforge::intel_esimd::simd<float, 16> v700_data = tensorforge::slmLoad<float, 16>(s2 + (60_i32));
              v698_acc += ((static_cast<float>(v700_data[0])) * v549_data);
              v698_acc += ((static_cast<float>(v700_data[1])) * v550_data);
              v698_acc += ((static_cast<float>(v700_data[2])) * v551_data);
              v698_acc += ((static_cast<float>(v700_data[3])) * v552_data);
              v698_acc += ((static_cast<float>(v700_data[4])) * v553_data);
              v698_acc += ((static_cast<float>(v700_data[5])) * v554_data);
              v698_acc += ((static_cast<float>(v700_data[6])) * v555_data);
              v698_acc += ((static_cast<float>(v700_data[7])) * v556_data);
              v698_acc += ((static_cast<float>(v700_data[8])) * v557_data);
              v698_acc += ((static_cast<float>(v700_data[9])) * v558_data);
              v698_acc += ((static_cast<float>(v700_data[10])) * v559_data);
              v698_acc += ((static_cast<float>(v700_data[11])) * v560_data);
              ir5.template select<16, 1>(80) = v698_acc;
              tensorforge::intel_esimd::simd<float, 16> v725_acc{};
              tensorforge::intel_esimd::simd<float, 16> v727_data = tensorforge::slmLoad<float, 16>(s2 + (72_i32));
              v725_acc += ((static_cast<float>(v727_data[0])) * v549_data);
              v725_acc += ((static_cast<float>(v727_data[1])) * v550_data);
              v725_acc += ((static_cast<float>(v727_data[2])) * v551_data);
              v725_acc += ((static_cast<float>(v727_data[3])) * v552_data);
              v725_acc += ((static_cast<float>(v727_data[4])) * v553_data);
              v725_acc += ((static_cast<float>(v727_data[5])) * v554_data);
              v725_acc += ((static_cast<float>(v727_data[6])) * v555_data);
              v725_acc += ((static_cast<float>(v727_data[7])) * v556_data);
              v725_acc += ((static_cast<float>(v727_data[8])) * v557_data);
              v725_acc += ((static_cast<float>(v727_data[9])) * v558_data);
              v725_acc += ((static_cast<float>(v727_data[10])) * v559_data);
              v725_acc += ((static_cast<float>(v727_data[11])) * v560_data);
              ir5.template select<16, 1>(96) = v725_acc;
              tensorforge::intel_esimd::simd<float, 16> v752_acc{};
              tensorforge::intel_esimd::simd<float, 16> v754_data = tensorforge::slmLoad<float, 16>(s2 + (84_i32));
              v752_acc += ((static_cast<float>(v754_data[0])) * v549_data);
              v752_acc += ((static_cast<float>(v754_data[1])) * v550_data);
              v752_acc += ((static_cast<float>(v754_data[2])) * v551_data);
              v752_acc += ((static_cast<float>(v754_data[3])) * v552_data);
              v752_acc += ((static_cast<float>(v754_data[4])) * v553_data);
              v752_acc += ((static_cast<float>(v754_data[5])) * v554_data);
              v752_acc += ((static_cast<float>(v754_data[6])) * v555_data);
              v752_acc += ((static_cast<float>(v754_data[7])) * v556_data);
              v752_acc += ((static_cast<float>(v754_data[8])) * v557_data);
              v752_acc += ((static_cast<float>(v754_data[9])) * v558_data);
              v752_acc += ((static_cast<float>(v754_data[10])) * v559_data);
              v752_acc += ((static_cast<float>(v754_data[11])) * v560_data);
              ir5.template select<16, 1>(112) = v752_acc;
              // r5 = ir5 + r3
              #pragma unroll
              for (int32_t v779_n1 = 0; v779_n1 < 8; ++v779_n1) {
                int32_t v780_a = v779_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v782_data(ir5.template select<12, 1>(v780_a));
                tensorforge::intel_esimd::simd<float, 12> v783_data(r3.template select<12, 1>(v780_a));
                r5.template select<12, 1>(v780_a) = (v783_data + v782_data);
              }
              // s3 = load{g>s}(glb_m8[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v785_ld;
              v785_ld.copy_from(glb_m8 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s3 + (0 + 0 + 4 * 0 + 0), v785_ld);
              tensorforge::intel_esimd::simd<float, 32> v786_ld;
              v786_ld.copy_from(glb_m8 + (0 + 0 + 2 * 0 + 64));
              tensorforge::slmStore<float, 32>(s3 + (0 + 0 + 2 * 0 + 64), v786_ld);
              // wait(r6 = load{g>r}(glb_m7););
              // wait(s3 = load{g>s}(glb_m8[0, 1]));
              tensorforge::intel_esimd::simd<float, 128> r7(0.0f);
              // ir7 = +(r6 * s3)
              // [(0, 12), (0, 8)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 128> ir7(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v789_data(r6.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v790_data(r6.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v791_data(r6.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v792_data(r6.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v793_data(r6.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v794_data(r6.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v795_data(r6.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v796_data(r6.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v797_data(r6.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v798_data(r6.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v799_data(r6.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v800_data(r6.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v801_acc{};
              tensorforge::intel_esimd::simd<float, 16> v805_data = tensorforge::slmLoad<float, 16>(s3 + (0_i32));
              v801_acc += ((static_cast<float>(v805_data[0])) * v789_data);
              v801_acc += ((static_cast<float>(v805_data[1])) * v790_data);
              v801_acc += ((static_cast<float>(v805_data[2])) * v791_data);
              v801_acc += ((static_cast<float>(v805_data[3])) * v792_data);
              v801_acc += ((static_cast<float>(v805_data[4])) * v793_data);
              v801_acc += ((static_cast<float>(v805_data[5])) * v794_data);
              v801_acc += ((static_cast<float>(v805_data[6])) * v795_data);
              v801_acc += ((static_cast<float>(v805_data[7])) * v796_data);
              v801_acc += ((static_cast<float>(v805_data[8])) * v797_data);
              v801_acc += ((static_cast<float>(v805_data[9])) * v798_data);
              v801_acc += ((static_cast<float>(v805_data[10])) * v799_data);
              v801_acc += ((static_cast<float>(v805_data[11])) * v800_data);
              ir7.template select<16, 1>(0) = v801_acc;
              tensorforge::intel_esimd::simd<float, 16> v830_acc{};
              tensorforge::intel_esimd::simd<float, 16> v832_data = tensorforge::slmLoad<float, 16>(s3 + (12_i32));
              v830_acc += ((static_cast<float>(v832_data[0])) * v789_data);
              v830_acc += ((static_cast<float>(v832_data[1])) * v790_data);
              v830_acc += ((static_cast<float>(v832_data[2])) * v791_data);
              v830_acc += ((static_cast<float>(v832_data[3])) * v792_data);
              v830_acc += ((static_cast<float>(v832_data[4])) * v793_data);
              v830_acc += ((static_cast<float>(v832_data[5])) * v794_data);
              v830_acc += ((static_cast<float>(v832_data[6])) * v795_data);
              v830_acc += ((static_cast<float>(v832_data[7])) * v796_data);
              v830_acc += ((static_cast<float>(v832_data[8])) * v797_data);
              v830_acc += ((static_cast<float>(v832_data[9])) * v798_data);
              v830_acc += ((static_cast<float>(v832_data[10])) * v799_data);
              v830_acc += ((static_cast<float>(v832_data[11])) * v800_data);
              ir7.template select<16, 1>(16) = v830_acc;
              tensorforge::intel_esimd::simd<float, 16> v857_acc{};
              tensorforge::intel_esimd::simd<float, 16> v859_data = tensorforge::slmLoad<float, 16>(s3 + (24_i32));
              v857_acc += ((static_cast<float>(v859_data[0])) * v789_data);
              v857_acc += ((static_cast<float>(v859_data[1])) * v790_data);
              v857_acc += ((static_cast<float>(v859_data[2])) * v791_data);
              v857_acc += ((static_cast<float>(v859_data[3])) * v792_data);
              v857_acc += ((static_cast<float>(v859_data[4])) * v793_data);
              v857_acc += ((static_cast<float>(v859_data[5])) * v794_data);
              v857_acc += ((static_cast<float>(v859_data[6])) * v795_data);
              v857_acc += ((static_cast<float>(v859_data[7])) * v796_data);
              v857_acc += ((static_cast<float>(v859_data[8])) * v797_data);
              v857_acc += ((static_cast<float>(v859_data[9])) * v798_data);
              v857_acc += ((static_cast<float>(v859_data[10])) * v799_data);
              v857_acc += ((static_cast<float>(v859_data[11])) * v800_data);
              ir7.template select<16, 1>(32) = v857_acc;
              tensorforge::intel_esimd::simd<float, 16> v884_acc{};
              tensorforge::intel_esimd::simd<float, 16> v886_data = tensorforge::slmLoad<float, 16>(s3 + (36_i32));
              v884_acc += ((static_cast<float>(v886_data[0])) * v789_data);
              v884_acc += ((static_cast<float>(v886_data[1])) * v790_data);
              v884_acc += ((static_cast<float>(v886_data[2])) * v791_data);
              v884_acc += ((static_cast<float>(v886_data[3])) * v792_data);
              v884_acc += ((static_cast<float>(v886_data[4])) * v793_data);
              v884_acc += ((static_cast<float>(v886_data[5])) * v794_data);
              v884_acc += ((static_cast<float>(v886_data[6])) * v795_data);
              v884_acc += ((static_cast<float>(v886_data[7])) * v796_data);
              v884_acc += ((static_cast<float>(v886_data[8])) * v797_data);
              v884_acc += ((static_cast<float>(v886_data[9])) * v798_data);
              v884_acc += ((static_cast<float>(v886_data[10])) * v799_data);
              v884_acc += ((static_cast<float>(v886_data[11])) * v800_data);
              ir7.template select<16, 1>(48) = v884_acc;
              tensorforge::intel_esimd::simd<float, 16> v911_acc{};
              tensorforge::intel_esimd::simd<float, 16> v913_data = tensorforge::slmLoad<float, 16>(s3 + (48_i32));
              v911_acc += ((static_cast<float>(v913_data[0])) * v789_data);
              v911_acc += ((static_cast<float>(v913_data[1])) * v790_data);
              v911_acc += ((static_cast<float>(v913_data[2])) * v791_data);
              v911_acc += ((static_cast<float>(v913_data[3])) * v792_data);
              v911_acc += ((static_cast<float>(v913_data[4])) * v793_data);
              v911_acc += ((static_cast<float>(v913_data[5])) * v794_data);
              v911_acc += ((static_cast<float>(v913_data[6])) * v795_data);
              v911_acc += ((static_cast<float>(v913_data[7])) * v796_data);
              v911_acc += ((static_cast<float>(v913_data[8])) * v797_data);
              v911_acc += ((static_cast<float>(v913_data[9])) * v798_data);
              v911_acc += ((static_cast<float>(v913_data[10])) * v799_data);
              v911_acc += ((static_cast<float>(v913_data[11])) * v800_data);
              ir7.template select<16, 1>(64) = v911_acc;
              tensorforge::intel_esimd::simd<float, 16> v938_acc{};
              tensorforge::intel_esimd::simd<float, 16> v940_data = tensorforge::slmLoad<float, 16>(s3 + (60_i32));
              v938_acc += ((static_cast<float>(v940_data[0])) * v789_data);
              v938_acc += ((static_cast<float>(v940_data[1])) * v790_data);
              v938_acc += ((static_cast<float>(v940_data[2])) * v791_data);
              v938_acc += ((static_cast<float>(v940_data[3])) * v792_data);
              v938_acc += ((static_cast<float>(v940_data[4])) * v793_data);
              v938_acc += ((static_cast<float>(v940_data[5])) * v794_data);
              v938_acc += ((static_cast<float>(v940_data[6])) * v795_data);
              v938_acc += ((static_cast<float>(v940_data[7])) * v796_data);
              v938_acc += ((static_cast<float>(v940_data[8])) * v797_data);
              v938_acc += ((static_cast<float>(v940_data[9])) * v798_data);
              v938_acc += ((static_cast<float>(v940_data[10])) * v799_data);
              v938_acc += ((static_cast<float>(v940_data[11])) * v800_data);
              ir7.template select<16, 1>(80) = v938_acc;
              tensorforge::intel_esimd::simd<float, 16> v965_acc{};
              tensorforge::intel_esimd::simd<float, 16> v967_data = tensorforge::slmLoad<float, 16>(s3 + (72_i32));
              v965_acc += ((static_cast<float>(v967_data[0])) * v789_data);
              v965_acc += ((static_cast<float>(v967_data[1])) * v790_data);
              v965_acc += ((static_cast<float>(v967_data[2])) * v791_data);
              v965_acc += ((static_cast<float>(v967_data[3])) * v792_data);
              v965_acc += ((static_cast<float>(v967_data[4])) * v793_data);
              v965_acc += ((static_cast<float>(v967_data[5])) * v794_data);
              v965_acc += ((static_cast<float>(v967_data[6])) * v795_data);
              v965_acc += ((static_cast<float>(v967_data[7])) * v796_data);
              v965_acc += ((static_cast<float>(v967_data[8])) * v797_data);
              v965_acc += ((static_cast<float>(v967_data[9])) * v798_data);
              v965_acc += ((static_cast<float>(v967_data[10])) * v799_data);
              v965_acc += ((static_cast<float>(v967_data[11])) * v800_data);
              ir7.template select<16, 1>(96) = v965_acc;
              tensorforge::intel_esimd::simd<float, 16> v992_acc{};
              tensorforge::intel_esimd::simd<float, 16> v994_data = tensorforge::slmLoad<float, 16>(s3 + (84_i32));
              v992_acc += ((static_cast<float>(v994_data[0])) * v789_data);
              v992_acc += ((static_cast<float>(v994_data[1])) * v790_data);
              v992_acc += ((static_cast<float>(v994_data[2])) * v791_data);
              v992_acc += ((static_cast<float>(v994_data[3])) * v792_data);
              v992_acc += ((static_cast<float>(v994_data[4])) * v793_data);
              v992_acc += ((static_cast<float>(v994_data[5])) * v794_data);
              v992_acc += ((static_cast<float>(v994_data[6])) * v795_data);
              v992_acc += ((static_cast<float>(v994_data[7])) * v796_data);
              v992_acc += ((static_cast<float>(v994_data[8])) * v797_data);
              v992_acc += ((static_cast<float>(v994_data[9])) * v798_data);
              v992_acc += ((static_cast<float>(v994_data[10])) * v799_data);
              v992_acc += ((static_cast<float>(v994_data[11])) * v800_data);
              ir7.template select<16, 1>(112) = v992_acc;
              // r7 = ir7 + r5
              #pragma unroll
              for (int32_t v1019_n1 = 0; v1019_n1 < 8; ++v1019_n1) {
                int32_t v1020_a = v1019_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v1022_data(ir7.template select<12, 1>(v1020_a));
                tensorforge::intel_esimd::simd<float, 12> v1023_data(r5.template select<12, 1>(v1020_a));
                r7.template select<12, 1>(v1020_a) = (v1023_data + v1022_data);
              }
              // glb_m0 = store{r>g}(r7);
              #pragma unroll
              for (int32_t v1025_i1 = 0; v1025_i1 < 8; ++v1025_i1) {
                tensorforge::intel_esimd::simd<float, 12> v1028_data(r7.template select<12, 1>((v1025_i1 * 16)));
                v1028_data.copy_to(glb_m0 + ((v1025_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}

