// === base name ===
kernel_afa5546598b79e7d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_afa5546598b79e7d = {{1, 16, 1}, 16, 12, 1, 16, 7168, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_afa5546598b79e7d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_afa5546598b79e7d(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_afa5546598b79e7d(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_afa5546598b79e7d(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_afa5546598b79e7d(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_afa5546598b79e7d(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, m6, m6_extraOffset, m7, m7_extraOffset, m8, m8_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_afa5546598b79e7d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0) {
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
              // wait(r0 = load{g>r}(glb_m1););
              tensorforge::intel_esimd::simd<float, 192> r2(0.0f);
              // r2 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v40_i1 = 0; v40_i1 < 12; ++v40_i1) {
                tensorforge::intel_esimd::simd<float, 12> v45_data;
                v45_data.copy_from(glb_m3 + ((v40_i1 * 12)));
                r2.template select<12, 1>((v40_i1 * 16)) = v45_data;
              }
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 128> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 12), (0, 8)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 128> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v50_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v51_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v52_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v53_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v54_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v55_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v56_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v57_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v58_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v59_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v60_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v61_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v62_acc{};
              tensorforge::intel_esimd::simd<float, 16> v66_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v62_acc += ((static_cast<float>(v66_data[0])) * v50_data);
              v62_acc += ((static_cast<float>(v66_data[1])) * v51_data);
              v62_acc += ((static_cast<float>(v66_data[2])) * v52_data);
              v62_acc += ((static_cast<float>(v66_data[3])) * v53_data);
              v62_acc += ((static_cast<float>(v66_data[4])) * v54_data);
              v62_acc += ((static_cast<float>(v66_data[5])) * v55_data);
              v62_acc += ((static_cast<float>(v66_data[6])) * v56_data);
              v62_acc += ((static_cast<float>(v66_data[7])) * v57_data);
              v62_acc += ((static_cast<float>(v66_data[8])) * v58_data);
              v62_acc += ((static_cast<float>(v66_data[9])) * v59_data);
              v62_acc += ((static_cast<float>(v66_data[10])) * v60_data);
              v62_acc += ((static_cast<float>(v66_data[11])) * v61_data);
              ir1.template select<16, 1>(0) = v62_acc;
              tensorforge::intel_esimd::simd<float, 16> v91_acc{};
              tensorforge::intel_esimd::simd<float, 16> v93_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              v91_acc += ((static_cast<float>(v93_data[0])) * v50_data);
              v91_acc += ((static_cast<float>(v93_data[1])) * v51_data);
              v91_acc += ((static_cast<float>(v93_data[2])) * v52_data);
              v91_acc += ((static_cast<float>(v93_data[3])) * v53_data);
              v91_acc += ((static_cast<float>(v93_data[4])) * v54_data);
              v91_acc += ((static_cast<float>(v93_data[5])) * v55_data);
              v91_acc += ((static_cast<float>(v93_data[6])) * v56_data);
              v91_acc += ((static_cast<float>(v93_data[7])) * v57_data);
              v91_acc += ((static_cast<float>(v93_data[8])) * v58_data);
              v91_acc += ((static_cast<float>(v93_data[9])) * v59_data);
              v91_acc += ((static_cast<float>(v93_data[10])) * v60_data);
              v91_acc += ((static_cast<float>(v93_data[11])) * v61_data);
              ir1.template select<16, 1>(16) = v91_acc;
              tensorforge::intel_esimd::simd<float, 16> v118_acc{};
              tensorforge::intel_esimd::simd<float, 16> v120_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              v118_acc += ((static_cast<float>(v120_data[0])) * v50_data);
              v118_acc += ((static_cast<float>(v120_data[1])) * v51_data);
              v118_acc += ((static_cast<float>(v120_data[2])) * v52_data);
              v118_acc += ((static_cast<float>(v120_data[3])) * v53_data);
              v118_acc += ((static_cast<float>(v120_data[4])) * v54_data);
              v118_acc += ((static_cast<float>(v120_data[5])) * v55_data);
              v118_acc += ((static_cast<float>(v120_data[6])) * v56_data);
              v118_acc += ((static_cast<float>(v120_data[7])) * v57_data);
              v118_acc += ((static_cast<float>(v120_data[8])) * v58_data);
              v118_acc += ((static_cast<float>(v120_data[9])) * v59_data);
              v118_acc += ((static_cast<float>(v120_data[10])) * v60_data);
              v118_acc += ((static_cast<float>(v120_data[11])) * v61_data);
              ir1.template select<16, 1>(32) = v118_acc;
              tensorforge::intel_esimd::simd<float, 16> v145_acc{};
              tensorforge::intel_esimd::simd<float, 16> v147_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              v145_acc += ((static_cast<float>(v147_data[0])) * v50_data);
              v145_acc += ((static_cast<float>(v147_data[1])) * v51_data);
              v145_acc += ((static_cast<float>(v147_data[2])) * v52_data);
              v145_acc += ((static_cast<float>(v147_data[3])) * v53_data);
              v145_acc += ((static_cast<float>(v147_data[4])) * v54_data);
              v145_acc += ((static_cast<float>(v147_data[5])) * v55_data);
              v145_acc += ((static_cast<float>(v147_data[6])) * v56_data);
              v145_acc += ((static_cast<float>(v147_data[7])) * v57_data);
              v145_acc += ((static_cast<float>(v147_data[8])) * v58_data);
              v145_acc += ((static_cast<float>(v147_data[9])) * v59_data);
              v145_acc += ((static_cast<float>(v147_data[10])) * v60_data);
              v145_acc += ((static_cast<float>(v147_data[11])) * v61_data);
              ir1.template select<16, 1>(48) = v145_acc;
              tensorforge::intel_esimd::simd<float, 16> v172_acc{};
              tensorforge::intel_esimd::simd<float, 16> v174_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v172_acc += ((static_cast<float>(v174_data[0])) * v50_data);
              v172_acc += ((static_cast<float>(v174_data[1])) * v51_data);
              v172_acc += ((static_cast<float>(v174_data[2])) * v52_data);
              v172_acc += ((static_cast<float>(v174_data[3])) * v53_data);
              v172_acc += ((static_cast<float>(v174_data[4])) * v54_data);
              v172_acc += ((static_cast<float>(v174_data[5])) * v55_data);
              v172_acc += ((static_cast<float>(v174_data[6])) * v56_data);
              v172_acc += ((static_cast<float>(v174_data[7])) * v57_data);
              v172_acc += ((static_cast<float>(v174_data[8])) * v58_data);
              v172_acc += ((static_cast<float>(v174_data[9])) * v59_data);
              v172_acc += ((static_cast<float>(v174_data[10])) * v60_data);
              v172_acc += ((static_cast<float>(v174_data[11])) * v61_data);
              ir1.template select<16, 1>(64) = v172_acc;
              tensorforge::intel_esimd::simd<float, 16> v199_acc{};
              tensorforge::intel_esimd::simd<float, 16> v201_data = tensorforge::slmLoad<float, 16>(s0 + (60_i32));
              v199_acc += ((static_cast<float>(v201_data[0])) * v50_data);
              v199_acc += ((static_cast<float>(v201_data[1])) * v51_data);
              v199_acc += ((static_cast<float>(v201_data[2])) * v52_data);
              v199_acc += ((static_cast<float>(v201_data[3])) * v53_data);
              v199_acc += ((static_cast<float>(v201_data[4])) * v54_data);
              v199_acc += ((static_cast<float>(v201_data[5])) * v55_data);
              v199_acc += ((static_cast<float>(v201_data[6])) * v56_data);
              v199_acc += ((static_cast<float>(v201_data[7])) * v57_data);
              v199_acc += ((static_cast<float>(v201_data[8])) * v58_data);
              v199_acc += ((static_cast<float>(v201_data[9])) * v59_data);
              v199_acc += ((static_cast<float>(v201_data[10])) * v60_data);
              v199_acc += ((static_cast<float>(v201_data[11])) * v61_data);
              ir1.template select<16, 1>(80) = v199_acc;
              tensorforge::intel_esimd::simd<float, 16> v226_acc{};
              tensorforge::intel_esimd::simd<float, 16> v228_data = tensorforge::slmLoad<float, 16>(s0 + (72_i32));
              v226_acc += ((static_cast<float>(v228_data[0])) * v50_data);
              v226_acc += ((static_cast<float>(v228_data[1])) * v51_data);
              v226_acc += ((static_cast<float>(v228_data[2])) * v52_data);
              v226_acc += ((static_cast<float>(v228_data[3])) * v53_data);
              v226_acc += ((static_cast<float>(v228_data[4])) * v54_data);
              v226_acc += ((static_cast<float>(v228_data[5])) * v55_data);
              v226_acc += ((static_cast<float>(v228_data[6])) * v56_data);
              v226_acc += ((static_cast<float>(v228_data[7])) * v57_data);
              v226_acc += ((static_cast<float>(v228_data[8])) * v58_data);
              v226_acc += ((static_cast<float>(v228_data[9])) * v59_data);
              v226_acc += ((static_cast<float>(v228_data[10])) * v60_data);
              v226_acc += ((static_cast<float>(v228_data[11])) * v61_data);
              ir1.template select<16, 1>(96) = v226_acc;
              tensorforge::intel_esimd::simd<float, 16> v253_acc{};
              tensorforge::intel_esimd::simd<float, 16> v255_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v253_acc += ((static_cast<float>(v255_data[0])) * v50_data);
              v253_acc += ((static_cast<float>(v255_data[1])) * v51_data);
              v253_acc += ((static_cast<float>(v255_data[2])) * v52_data);
              v253_acc += ((static_cast<float>(v255_data[3])) * v53_data);
              v253_acc += ((static_cast<float>(v255_data[4])) * v54_data);
              v253_acc += ((static_cast<float>(v255_data[5])) * v55_data);
              v253_acc += ((static_cast<float>(v255_data[6])) * v56_data);
              v253_acc += ((static_cast<float>(v255_data[7])) * v57_data);
              v253_acc += ((static_cast<float>(v255_data[8])) * v58_data);
              v253_acc += ((static_cast<float>(v255_data[9])) * v59_data);
              v253_acc += ((static_cast<float>(v255_data[10])) * v60_data);
              v253_acc += ((static_cast<float>(v255_data[11])) * v61_data);
              ir1.template select<16, 1>(112) = v253_acc;
              // r1 = ir1
              #pragma unroll
              for (int32_t v280_n1 = 0; v280_n1 < 8; ++v280_n1) {
                int32_t v281_a = v280_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v283_data(ir1.template select<12, 1>(v281_a));
                r1.template select<12, 1>(v281_a) = v283_data;
              }
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v284_ld;
              v284_ld.copy_from(glb_m4 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 4 * 0 + 0), v284_ld);
              tensorforge::intel_esimd::simd<float, 32> v285_ld;
              v285_ld.copy_from(glb_m4 + (0 + 0 + 2 * 0 + 64));
              tensorforge::slmStore<float, 32>(s1 + (0 + 0 + 2 * 0 + 64), v285_ld);
              // wait(r2 = load{g>r}(glb_m3););
              tensorforge::intel_esimd::simd<float, 192> r4(0.0f);
              // r4 = load{g>r}(glb_m5);
              #pragma unroll
              for (int32_t v287_i1 = 0; v287_i1 < 12; ++v287_i1) {
                tensorforge::intel_esimd::simd<float, 12> v292_data;
                v292_data.copy_from(glb_m5 + ((v287_i1 * 12)));
                r4.template select<12, 1>((v287_i1 * 16)) = v292_data;
              }
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              tensorforge::intel_esimd::simd<float, 128> r3(0.0f);
              // ir3 = +(r2 * s1)
              // [(0, 12), (0, 8)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 128> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v297_data(r2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v298_data(r2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v299_data(r2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v300_data(r2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v301_data(r2.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v302_data(r2.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v303_data(r2.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v304_data(r2.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v305_data(r2.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v306_data(r2.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v307_data(r2.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v308_data(r2.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v309_acc{};
              tensorforge::intel_esimd::simd<float, 16> v313_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v309_acc += ((static_cast<float>(v313_data[0])) * v297_data);
              v309_acc += ((static_cast<float>(v313_data[1])) * v298_data);
              v309_acc += ((static_cast<float>(v313_data[2])) * v299_data);
              v309_acc += ((static_cast<float>(v313_data[3])) * v300_data);
              v309_acc += ((static_cast<float>(v313_data[4])) * v301_data);
              v309_acc += ((static_cast<float>(v313_data[5])) * v302_data);
              v309_acc += ((static_cast<float>(v313_data[6])) * v303_data);
              v309_acc += ((static_cast<float>(v313_data[7])) * v304_data);
              v309_acc += ((static_cast<float>(v313_data[8])) * v305_data);
              v309_acc += ((static_cast<float>(v313_data[9])) * v306_data);
              v309_acc += ((static_cast<float>(v313_data[10])) * v307_data);
              v309_acc += ((static_cast<float>(v313_data[11])) * v308_data);
              ir3.template select<16, 1>(0) = v309_acc;
              tensorforge::intel_esimd::simd<float, 16> v338_acc{};
              tensorforge::intel_esimd::simd<float, 16> v340_data = tensorforge::slmLoad<float, 16>(s1 + (12_i32));
              v338_acc += ((static_cast<float>(v340_data[0])) * v297_data);
              v338_acc += ((static_cast<float>(v340_data[1])) * v298_data);
              v338_acc += ((static_cast<float>(v340_data[2])) * v299_data);
              v338_acc += ((static_cast<float>(v340_data[3])) * v300_data);
              v338_acc += ((static_cast<float>(v340_data[4])) * v301_data);
              v338_acc += ((static_cast<float>(v340_data[5])) * v302_data);
              v338_acc += ((static_cast<float>(v340_data[6])) * v303_data);
              v338_acc += ((static_cast<float>(v340_data[7])) * v304_data);
              v338_acc += ((static_cast<float>(v340_data[8])) * v305_data);
              v338_acc += ((static_cast<float>(v340_data[9])) * v306_data);
              v338_acc += ((static_cast<float>(v340_data[10])) * v307_data);
              v338_acc += ((static_cast<float>(v340_data[11])) * v308_data);
              ir3.template select<16, 1>(16) = v338_acc;
              tensorforge::intel_esimd::simd<float, 16> v365_acc{};
              tensorforge::intel_esimd::simd<float, 16> v367_data = tensorforge::slmLoad<float, 16>(s1 + (24_i32));
              v365_acc += ((static_cast<float>(v367_data[0])) * v297_data);
              v365_acc += ((static_cast<float>(v367_data[1])) * v298_data);
              v365_acc += ((static_cast<float>(v367_data[2])) * v299_data);
              v365_acc += ((static_cast<float>(v367_data[3])) * v300_data);
              v365_acc += ((static_cast<float>(v367_data[4])) * v301_data);
              v365_acc += ((static_cast<float>(v367_data[5])) * v302_data);
              v365_acc += ((static_cast<float>(v367_data[6])) * v303_data);
              v365_acc += ((static_cast<float>(v367_data[7])) * v304_data);
              v365_acc += ((static_cast<float>(v367_data[8])) * v305_data);
              v365_acc += ((static_cast<float>(v367_data[9])) * v306_data);
              v365_acc += ((static_cast<float>(v367_data[10])) * v307_data);
              v365_acc += ((static_cast<float>(v367_data[11])) * v308_data);
              ir3.template select<16, 1>(32) = v365_acc;
              tensorforge::intel_esimd::simd<float, 16> v392_acc{};
              tensorforge::intel_esimd::simd<float, 16> v394_data = tensorforge::slmLoad<float, 16>(s1 + (36_i32));
              v392_acc += ((static_cast<float>(v394_data[0])) * v297_data);
              v392_acc += ((static_cast<float>(v394_data[1])) * v298_data);
              v392_acc += ((static_cast<float>(v394_data[2])) * v299_data);
              v392_acc += ((static_cast<float>(v394_data[3])) * v300_data);
              v392_acc += ((static_cast<float>(v394_data[4])) * v301_data);
              v392_acc += ((static_cast<float>(v394_data[5])) * v302_data);
              v392_acc += ((static_cast<float>(v394_data[6])) * v303_data);
              v392_acc += ((static_cast<float>(v394_data[7])) * v304_data);
              v392_acc += ((static_cast<float>(v394_data[8])) * v305_data);
              v392_acc += ((static_cast<float>(v394_data[9])) * v306_data);
              v392_acc += ((static_cast<float>(v394_data[10])) * v307_data);
              v392_acc += ((static_cast<float>(v394_data[11])) * v308_data);
              ir3.template select<16, 1>(48) = v392_acc;
              tensorforge::intel_esimd::simd<float, 16> v419_acc{};
              tensorforge::intel_esimd::simd<float, 16> v421_data = tensorforge::slmLoad<float, 16>(s1 + (48_i32));
              v419_acc += ((static_cast<float>(v421_data[0])) * v297_data);
              v419_acc += ((static_cast<float>(v421_data[1])) * v298_data);
              v419_acc += ((static_cast<float>(v421_data[2])) * v299_data);
              v419_acc += ((static_cast<float>(v421_data[3])) * v300_data);
              v419_acc += ((static_cast<float>(v421_data[4])) * v301_data);
              v419_acc += ((static_cast<float>(v421_data[5])) * v302_data);
              v419_acc += ((static_cast<float>(v421_data[6])) * v303_data);
              v419_acc += ((static_cast<float>(v421_data[7])) * v304_data);
              v419_acc += ((static_cast<float>(v421_data[8])) * v305_data);
              v419_acc += ((static_cast<float>(v421_data[9])) * v306_data);
              v419_acc += ((static_cast<float>(v421_data[10])) * v307_data);
              v419_acc += ((static_cast<float>(v421_data[11])) * v308_data);
              ir3.template select<16, 1>(64) = v419_acc;
              tensorforge::intel_esimd::simd<float, 16> v446_acc{};
              tensorforge::intel_esimd::simd<float, 16> v448_data = tensorforge::slmLoad<float, 16>(s1 + (60_i32));
              v446_acc += ((static_cast<float>(v448_data[0])) * v297_data);
              v446_acc += ((static_cast<float>(v448_data[1])) * v298_data);
              v446_acc += ((static_cast<float>(v448_data[2])) * v299_data);
              v446_acc += ((static_cast<float>(v448_data[3])) * v300_data);
              v446_acc += ((static_cast<float>(v448_data[4])) * v301_data);
              v446_acc += ((static_cast<float>(v448_data[5])) * v302_data);
              v446_acc += ((static_cast<float>(v448_data[6])) * v303_data);
              v446_acc += ((static_cast<float>(v448_data[7])) * v304_data);
              v446_acc += ((static_cast<float>(v448_data[8])) * v305_data);
              v446_acc += ((static_cast<float>(v448_data[9])) * v306_data);
              v446_acc += ((static_cast<float>(v448_data[10])) * v307_data);
              v446_acc += ((static_cast<float>(v448_data[11])) * v308_data);
              ir3.template select<16, 1>(80) = v446_acc;
              tensorforge::intel_esimd::simd<float, 16> v473_acc{};
              tensorforge::intel_esimd::simd<float, 16> v475_data = tensorforge::slmLoad<float, 16>(s1 + (72_i32));
              v473_acc += ((static_cast<float>(v475_data[0])) * v297_data);
              v473_acc += ((static_cast<float>(v475_data[1])) * v298_data);
              v473_acc += ((static_cast<float>(v475_data[2])) * v299_data);
              v473_acc += ((static_cast<float>(v475_data[3])) * v300_data);
              v473_acc += ((static_cast<float>(v475_data[4])) * v301_data);
              v473_acc += ((static_cast<float>(v475_data[5])) * v302_data);
              v473_acc += ((static_cast<float>(v475_data[6])) * v303_data);
              v473_acc += ((static_cast<float>(v475_data[7])) * v304_data);
              v473_acc += ((static_cast<float>(v475_data[8])) * v305_data);
              v473_acc += ((static_cast<float>(v475_data[9])) * v306_data);
              v473_acc += ((static_cast<float>(v475_data[10])) * v307_data);
              v473_acc += ((static_cast<float>(v475_data[11])) * v308_data);
              ir3.template select<16, 1>(96) = v473_acc;
              tensorforge::intel_esimd::simd<float, 16> v500_acc{};
              tensorforge::intel_esimd::simd<float, 16> v502_data = tensorforge::slmLoad<float, 16>(s1 + (84_i32));
              v500_acc += ((static_cast<float>(v502_data[0])) * v297_data);
              v500_acc += ((static_cast<float>(v502_data[1])) * v298_data);
              v500_acc += ((static_cast<float>(v502_data[2])) * v299_data);
              v500_acc += ((static_cast<float>(v502_data[3])) * v300_data);
              v500_acc += ((static_cast<float>(v502_data[4])) * v301_data);
              v500_acc += ((static_cast<float>(v502_data[5])) * v302_data);
              v500_acc += ((static_cast<float>(v502_data[6])) * v303_data);
              v500_acc += ((static_cast<float>(v502_data[7])) * v304_data);
              v500_acc += ((static_cast<float>(v502_data[8])) * v305_data);
              v500_acc += ((static_cast<float>(v502_data[9])) * v306_data);
              v500_acc += ((static_cast<float>(v502_data[10])) * v307_data);
              v500_acc += ((static_cast<float>(v502_data[11])) * v308_data);
              ir3.template select<16, 1>(112) = v500_acc;
              // r3 = ir3 + r1
              #pragma unroll
              for (int32_t v527_n1 = 0; v527_n1 < 8; ++v527_n1) {
                int32_t v528_a = v527_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v530_data(ir3.template select<12, 1>(v528_a));
                tensorforge::intel_esimd::simd<float, 12> v531_data(r1.template select<12, 1>(v528_a));
                r3.template select<12, 1>(v528_a) = (v531_data + v530_data);
              }
              // s2 = load{g>s}(glb_m6[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v533_ld;
              v533_ld.copy_from(glb_m6 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s2 + (0 + 0 + 4 * 0 + 0), v533_ld);
              tensorforge::intel_esimd::simd<float, 32> v534_ld;
              v534_ld.copy_from(glb_m6 + (0 + 0 + 2 * 0 + 64));
              tensorforge::slmStore<float, 32>(s2 + (0 + 0 + 2 * 0 + 64), v534_ld);
              // wait(r4 = load{g>r}(glb_m5););
              tensorforge::intel_esimd::simd<float, 192> r6(0.0f);
              // r6 = load{g>r}(glb_m7);
              #pragma unroll
              for (int32_t v536_i1 = 0; v536_i1 < 12; ++v536_i1) {
                tensorforge::intel_esimd::simd<float, 12> v541_data;
                v541_data.copy_from(glb_m7 + ((v536_i1 * 12)));
                r6.template select<12, 1>((v536_i1 * 16)) = v541_data;
              }
              // wait(s2 = load{g>s}(glb_m6[0, 1]));
              tensorforge::intel_esimd::simd<float, 128> r5(0.0f);
              // ir5 = +(r4 * s2)
              // [(0, 12), (0, 8)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 128> ir5(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v546_data(r4.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v547_data(r4.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v548_data(r4.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v549_data(r4.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v550_data(r4.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v551_data(r4.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v552_data(r4.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v553_data(r4.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v554_data(r4.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v555_data(r4.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v556_data(r4.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v557_data(r4.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v558_acc{};
              tensorforge::intel_esimd::simd<float, 16> v562_data = tensorforge::slmLoad<float, 16>(s2 + (0_i32));
              v558_acc += ((static_cast<float>(v562_data[0])) * v546_data);
              v558_acc += ((static_cast<float>(v562_data[1])) * v547_data);
              v558_acc += ((static_cast<float>(v562_data[2])) * v548_data);
              v558_acc += ((static_cast<float>(v562_data[3])) * v549_data);
              v558_acc += ((static_cast<float>(v562_data[4])) * v550_data);
              v558_acc += ((static_cast<float>(v562_data[5])) * v551_data);
              v558_acc += ((static_cast<float>(v562_data[6])) * v552_data);
              v558_acc += ((static_cast<float>(v562_data[7])) * v553_data);
              v558_acc += ((static_cast<float>(v562_data[8])) * v554_data);
              v558_acc += ((static_cast<float>(v562_data[9])) * v555_data);
              v558_acc += ((static_cast<float>(v562_data[10])) * v556_data);
              v558_acc += ((static_cast<float>(v562_data[11])) * v557_data);
              ir5.template select<16, 1>(0) = v558_acc;
              tensorforge::intel_esimd::simd<float, 16> v587_acc{};
              tensorforge::intel_esimd::simd<float, 16> v589_data = tensorforge::slmLoad<float, 16>(s2 + (12_i32));
              v587_acc += ((static_cast<float>(v589_data[0])) * v546_data);
              v587_acc += ((static_cast<float>(v589_data[1])) * v547_data);
              v587_acc += ((static_cast<float>(v589_data[2])) * v548_data);
              v587_acc += ((static_cast<float>(v589_data[3])) * v549_data);
              v587_acc += ((static_cast<float>(v589_data[4])) * v550_data);
              v587_acc += ((static_cast<float>(v589_data[5])) * v551_data);
              v587_acc += ((static_cast<float>(v589_data[6])) * v552_data);
              v587_acc += ((static_cast<float>(v589_data[7])) * v553_data);
              v587_acc += ((static_cast<float>(v589_data[8])) * v554_data);
              v587_acc += ((static_cast<float>(v589_data[9])) * v555_data);
              v587_acc += ((static_cast<float>(v589_data[10])) * v556_data);
              v587_acc += ((static_cast<float>(v589_data[11])) * v557_data);
              ir5.template select<16, 1>(16) = v587_acc;
              tensorforge::intel_esimd::simd<float, 16> v614_acc{};
              tensorforge::intel_esimd::simd<float, 16> v616_data = tensorforge::slmLoad<float, 16>(s2 + (24_i32));
              v614_acc += ((static_cast<float>(v616_data[0])) * v546_data);
              v614_acc += ((static_cast<float>(v616_data[1])) * v547_data);
              v614_acc += ((static_cast<float>(v616_data[2])) * v548_data);
              v614_acc += ((static_cast<float>(v616_data[3])) * v549_data);
              v614_acc += ((static_cast<float>(v616_data[4])) * v550_data);
              v614_acc += ((static_cast<float>(v616_data[5])) * v551_data);
              v614_acc += ((static_cast<float>(v616_data[6])) * v552_data);
              v614_acc += ((static_cast<float>(v616_data[7])) * v553_data);
              v614_acc += ((static_cast<float>(v616_data[8])) * v554_data);
              v614_acc += ((static_cast<float>(v616_data[9])) * v555_data);
              v614_acc += ((static_cast<float>(v616_data[10])) * v556_data);
              v614_acc += ((static_cast<float>(v616_data[11])) * v557_data);
              ir5.template select<16, 1>(32) = v614_acc;
              tensorforge::intel_esimd::simd<float, 16> v641_acc{};
              tensorforge::intel_esimd::simd<float, 16> v643_data = tensorforge::slmLoad<float, 16>(s2 + (36_i32));
              v641_acc += ((static_cast<float>(v643_data[0])) * v546_data);
              v641_acc += ((static_cast<float>(v643_data[1])) * v547_data);
              v641_acc += ((static_cast<float>(v643_data[2])) * v548_data);
              v641_acc += ((static_cast<float>(v643_data[3])) * v549_data);
              v641_acc += ((static_cast<float>(v643_data[4])) * v550_data);
              v641_acc += ((static_cast<float>(v643_data[5])) * v551_data);
              v641_acc += ((static_cast<float>(v643_data[6])) * v552_data);
              v641_acc += ((static_cast<float>(v643_data[7])) * v553_data);
              v641_acc += ((static_cast<float>(v643_data[8])) * v554_data);
              v641_acc += ((static_cast<float>(v643_data[9])) * v555_data);
              v641_acc += ((static_cast<float>(v643_data[10])) * v556_data);
              v641_acc += ((static_cast<float>(v643_data[11])) * v557_data);
              ir5.template select<16, 1>(48) = v641_acc;
              tensorforge::intel_esimd::simd<float, 16> v668_acc{};
              tensorforge::intel_esimd::simd<float, 16> v670_data = tensorforge::slmLoad<float, 16>(s2 + (48_i32));
              v668_acc += ((static_cast<float>(v670_data[0])) * v546_data);
              v668_acc += ((static_cast<float>(v670_data[1])) * v547_data);
              v668_acc += ((static_cast<float>(v670_data[2])) * v548_data);
              v668_acc += ((static_cast<float>(v670_data[3])) * v549_data);
              v668_acc += ((static_cast<float>(v670_data[4])) * v550_data);
              v668_acc += ((static_cast<float>(v670_data[5])) * v551_data);
              v668_acc += ((static_cast<float>(v670_data[6])) * v552_data);
              v668_acc += ((static_cast<float>(v670_data[7])) * v553_data);
              v668_acc += ((static_cast<float>(v670_data[8])) * v554_data);
              v668_acc += ((static_cast<float>(v670_data[9])) * v555_data);
              v668_acc += ((static_cast<float>(v670_data[10])) * v556_data);
              v668_acc += ((static_cast<float>(v670_data[11])) * v557_data);
              ir5.template select<16, 1>(64) = v668_acc;
              tensorforge::intel_esimd::simd<float, 16> v695_acc{};
              tensorforge::intel_esimd::simd<float, 16> v697_data = tensorforge::slmLoad<float, 16>(s2 + (60_i32));
              v695_acc += ((static_cast<float>(v697_data[0])) * v546_data);
              v695_acc += ((static_cast<float>(v697_data[1])) * v547_data);
              v695_acc += ((static_cast<float>(v697_data[2])) * v548_data);
              v695_acc += ((static_cast<float>(v697_data[3])) * v549_data);
              v695_acc += ((static_cast<float>(v697_data[4])) * v550_data);
              v695_acc += ((static_cast<float>(v697_data[5])) * v551_data);
              v695_acc += ((static_cast<float>(v697_data[6])) * v552_data);
              v695_acc += ((static_cast<float>(v697_data[7])) * v553_data);
              v695_acc += ((static_cast<float>(v697_data[8])) * v554_data);
              v695_acc += ((static_cast<float>(v697_data[9])) * v555_data);
              v695_acc += ((static_cast<float>(v697_data[10])) * v556_data);
              v695_acc += ((static_cast<float>(v697_data[11])) * v557_data);
              ir5.template select<16, 1>(80) = v695_acc;
              tensorforge::intel_esimd::simd<float, 16> v722_acc{};
              tensorforge::intel_esimd::simd<float, 16> v724_data = tensorforge::slmLoad<float, 16>(s2 + (72_i32));
              v722_acc += ((static_cast<float>(v724_data[0])) * v546_data);
              v722_acc += ((static_cast<float>(v724_data[1])) * v547_data);
              v722_acc += ((static_cast<float>(v724_data[2])) * v548_data);
              v722_acc += ((static_cast<float>(v724_data[3])) * v549_data);
              v722_acc += ((static_cast<float>(v724_data[4])) * v550_data);
              v722_acc += ((static_cast<float>(v724_data[5])) * v551_data);
              v722_acc += ((static_cast<float>(v724_data[6])) * v552_data);
              v722_acc += ((static_cast<float>(v724_data[7])) * v553_data);
              v722_acc += ((static_cast<float>(v724_data[8])) * v554_data);
              v722_acc += ((static_cast<float>(v724_data[9])) * v555_data);
              v722_acc += ((static_cast<float>(v724_data[10])) * v556_data);
              v722_acc += ((static_cast<float>(v724_data[11])) * v557_data);
              ir5.template select<16, 1>(96) = v722_acc;
              tensorforge::intel_esimd::simd<float, 16> v749_acc{};
              tensorforge::intel_esimd::simd<float, 16> v751_data = tensorforge::slmLoad<float, 16>(s2 + (84_i32));
              v749_acc += ((static_cast<float>(v751_data[0])) * v546_data);
              v749_acc += ((static_cast<float>(v751_data[1])) * v547_data);
              v749_acc += ((static_cast<float>(v751_data[2])) * v548_data);
              v749_acc += ((static_cast<float>(v751_data[3])) * v549_data);
              v749_acc += ((static_cast<float>(v751_data[4])) * v550_data);
              v749_acc += ((static_cast<float>(v751_data[5])) * v551_data);
              v749_acc += ((static_cast<float>(v751_data[6])) * v552_data);
              v749_acc += ((static_cast<float>(v751_data[7])) * v553_data);
              v749_acc += ((static_cast<float>(v751_data[8])) * v554_data);
              v749_acc += ((static_cast<float>(v751_data[9])) * v555_data);
              v749_acc += ((static_cast<float>(v751_data[10])) * v556_data);
              v749_acc += ((static_cast<float>(v751_data[11])) * v557_data);
              ir5.template select<16, 1>(112) = v749_acc;
              // r5 = ir5 + r3
              #pragma unroll
              for (int32_t v776_n1 = 0; v776_n1 < 8; ++v776_n1) {
                int32_t v777_a = v776_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v779_data(ir5.template select<12, 1>(v777_a));
                tensorforge::intel_esimd::simd<float, 12> v780_data(r3.template select<12, 1>(v777_a));
                r5.template select<12, 1>(v777_a) = (v780_data + v779_data);
              }
              // s3 = load{g>s}(glb_m8[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v782_ld;
              v782_ld.copy_from(glb_m8 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s3 + (0 + 0 + 4 * 0 + 0), v782_ld);
              tensorforge::intel_esimd::simd<float, 32> v783_ld;
              v783_ld.copy_from(glb_m8 + (0 + 0 + 2 * 0 + 64));
              tensorforge::slmStore<float, 32>(s3 + (0 + 0 + 2 * 0 + 64), v783_ld);
              // wait(r6 = load{g>r}(glb_m7););
              // wait(s3 = load{g>s}(glb_m8[0, 1]));
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

