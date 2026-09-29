// === base name ===
kernel_dcf78d877d16d6e0

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_dcf78d877d16d6e0 = {{1, 8, 1}, 32, 20, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_dcf78d877d16d6e0(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_dcf78d877d16d6e0(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_dcf78d877d16d6e0(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 8, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 8 - 1) / 8;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 0 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_dcf78d877d16d6e0(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_dcf78d877d16d6e0(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_dcf78d877d16d6e0(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_dcf78d877d16d6e0(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      // generated with TensorForge. Version: 0.0.1
      // options: default
      // launch: 32 lanes (20 active) x 8 per block = block 1x8x1, 0 B shared, occupancy grid
      // operands:
      //   m0 20×9(20×9) {0..20}×{0..9} strided
      //   m1 20×9(20×9) {0..20}×{0..9} strided
      //   m2 10×9(10×9) {0..10}×{0..9} strided
      //   m3 4×9(4×9) {0..4}×{0..9} strided
      //   m4 1×9(1×9) {0..1}×{0..9} strided
      // operations:
      //   m0[i,j] = m1[i,j]
      //   m0[i,j] += m2[i,j]
      //   m0[i,j] += m3[i,j]
      //   m0[i,j] += m4[i,j]
      // tensorforge-meta: {"fp":"float","launch":{"active_threads":20,"block":[1,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[20,9]],"name":"m0","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"strided","alias":"Q","bbox":[[0,0],[20,9]],"name":"m1","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"strided","alias":"F0","bbox":[[0,0],[10,9]],"name":"m2","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"strided","alias":"F1","bbox":[[0,0],[4,9]],"name":"m3","ordered":false,"parts":1,"shape":[4,9],"variant":false},{"addressing":"strided","alias":"F2","bbox":[[0,0],[1,9]],"name":"m4","ordered":false,"parts":1,"shape":[1,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[20,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[10,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[4,9]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[4,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[1,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[1,9]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1\n"}
      {
        const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
        const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
        const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
        for (size_t v1_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v1_batchId0 < numElements0; v1_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
          size_t v2_ahead1 = v1_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
          const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
          if (allowed) {
            float *const __restrict__ glb_m0 = &m0[v1_batchId0 * 180 + 0 + m0_extraOffset];
            const float *const __restrict__ glb_m1 = &m1[v1_batchId0 * 180 + 0 + m1_extraOffset];
            const float *const __restrict__ glb_m2 = &m2[v1_batchId0 * 90 + 0 + m2_extraOffset];
            const float *const __restrict__ glb_m3 = &m3[v1_batchId0 * 36 + 0 + m3_extraOffset];
            const float *const __restrict__ glb_m4 = &m4[v1_batchId0 * 9 + 0 + m4_extraOffset];
            tensorforge::intel_esimd::simd<float, 288> r0(0.0f);
            // r0 = load{g>r}(glb_m1);
            #pragma unroll
            for (int32_t v15_i1 = 0; v15_i1 < 9; ++v15_i1) {
              tensorforge::intel_esimd::simd<float, 20> v20_data;
              v20_data.copy_from(glb_m1 + ((v15_i1 * 20)));
              r0.template select<20, 1>((v15_i1 * 32)) = v20_data;
            }
            tensorforge::intel_esimd::simd<float, 288> r2(0.0f);
            // r2 = load{g>r}(glb_m2);
            #pragma unroll
            for (int32_t v24_i1 = 0; v24_i1 < 9; ++v24_i1) {
              tensorforge::intel_esimd::simd<float, 10> v29_data;
              v29_data.copy_from(glb_m2 + ((v24_i1 * 10)));
              r2.template select<10, 1>((v24_i1 * 32)) = v29_data;
            }
            // wait(r0 = load{g>r}(glb_m1););
            tensorforge::intel_esimd::simd<float, 288> r1(0.0f);
            // ir1 = +(r0)
            // [(0, 20), (0, 9)] []
            tensorforge::intel_esimd::simd<float, 288> ir1(0.0f);
            tensorforge::intel_esimd::simd<float, 32> v34_data(r0.template select<32, 1>(0));
            tensorforge::intel_esimd::simd<float, 32> v35_data(ir1.template select<32, 1>(0));
            ir1.template select<32, 1>(0) = (v35_data + v34_data);
            tensorforge::intel_esimd::simd<float, 32> v37_data(r0.template select<32, 1>(32));
            tensorforge::intel_esimd::simd<float, 32> v38_data(ir1.template select<32, 1>(32));
            ir1.template select<32, 1>(32) = (v38_data + v37_data);
            tensorforge::intel_esimd::simd<float, 32> v40_data(r0.template select<32, 1>(64));
            tensorforge::intel_esimd::simd<float, 32> v41_data(ir1.template select<32, 1>(64));
            ir1.template select<32, 1>(64) = (v41_data + v40_data);
            tensorforge::intel_esimd::simd<float, 32> v43_data(r0.template select<32, 1>(96));
            tensorforge::intel_esimd::simd<float, 32> v44_data(ir1.template select<32, 1>(96));
            ir1.template select<32, 1>(96) = (v44_data + v43_data);
            tensorforge::intel_esimd::simd<float, 32> v46_data(r0.template select<32, 1>(128));
            tensorforge::intel_esimd::simd<float, 32> v47_data(ir1.template select<32, 1>(128));
            ir1.template select<32, 1>(128) = (v47_data + v46_data);
            tensorforge::intel_esimd::simd<float, 32> v49_data(r0.template select<32, 1>(160));
            tensorforge::intel_esimd::simd<float, 32> v50_data(ir1.template select<32, 1>(160));
            ir1.template select<32, 1>(160) = (v50_data + v49_data);
            tensorforge::intel_esimd::simd<float, 32> v52_data(r0.template select<32, 1>(192));
            tensorforge::intel_esimd::simd<float, 32> v53_data(ir1.template select<32, 1>(192));
            ir1.template select<32, 1>(192) = (v53_data + v52_data);
            tensorforge::intel_esimd::simd<float, 32> v55_data(r0.template select<32, 1>(224));
            tensorforge::intel_esimd::simd<float, 32> v56_data(ir1.template select<32, 1>(224));
            ir1.template select<32, 1>(224) = (v56_data + v55_data);
            tensorforge::intel_esimd::simd<float, 32> v58_data(r0.template select<32, 1>(256));
            tensorforge::intel_esimd::simd<float, 32> v59_data(ir1.template select<32, 1>(256));
            ir1.template select<32, 1>(256) = (v59_data + v58_data);
            // r1 = ir1
            #pragma unroll
            for (int32_t v61_n1 = 0; v61_n1 < 9; ++v61_n1) {
              int32_t v62_a = v61_n1 * 32;
              tensorforge::intel_esimd::simd<float, 20> v64_data(ir1.template select<20, 1>(v62_a));
              r1.template select<20, 1>(v62_a) = v64_data;
            }
            tensorforge::intel_esimd::simd<float, 288> r4(0.0f);
            // r4 = load{g>r}(glb_m3);
            #pragma unroll
            for (int32_t v66_i1 = 0; v66_i1 < 9; ++v66_i1) {
              tensorforge::intel_esimd::simd<float, 4> v71_data;
              v71_data.copy_from(glb_m3 + ((v66_i1 * 4)));
              r4.template select<4, 1>((v66_i1 * 32)) = v71_data;
            }
            // wait(r2 = load{g>r}(glb_m2););
            tensorforge::intel_esimd::simd<float, 288> r3(0.0f);
            // ir3 = +(r2)
            // [(0, 10), (0, 9)] []
            tensorforge::intel_esimd::simd<float, 288> ir3(0.0f);
            tensorforge::intel_esimd::simd<float, 32> v76_data(r2.template select<32, 1>(0));
            tensorforge::intel_esimd::simd<float, 32> v77_data(ir3.template select<32, 1>(0));
            ir3.template select<32, 1>(0) = (v77_data + v76_data);
            tensorforge::intel_esimd::simd<float, 32> v79_data(r2.template select<32, 1>(32));
            tensorforge::intel_esimd::simd<float, 32> v80_data(ir3.template select<32, 1>(32));
            ir3.template select<32, 1>(32) = (v80_data + v79_data);
            tensorforge::intel_esimd::simd<float, 32> v82_data(r2.template select<32, 1>(64));
            tensorforge::intel_esimd::simd<float, 32> v83_data(ir3.template select<32, 1>(64));
            ir3.template select<32, 1>(64) = (v83_data + v82_data);
            tensorforge::intel_esimd::simd<float, 32> v85_data(r2.template select<32, 1>(96));
            tensorforge::intel_esimd::simd<float, 32> v86_data(ir3.template select<32, 1>(96));
            ir3.template select<32, 1>(96) = (v86_data + v85_data);
            tensorforge::intel_esimd::simd<float, 32> v88_data(r2.template select<32, 1>(128));
            tensorforge::intel_esimd::simd<float, 32> v89_data(ir3.template select<32, 1>(128));
            ir3.template select<32, 1>(128) = (v89_data + v88_data);
            tensorforge::intel_esimd::simd<float, 32> v91_data(r2.template select<32, 1>(160));
            tensorforge::intel_esimd::simd<float, 32> v92_data(ir3.template select<32, 1>(160));
            ir3.template select<32, 1>(160) = (v92_data + v91_data);
            tensorforge::intel_esimd::simd<float, 32> v94_data(r2.template select<32, 1>(192));
            tensorforge::intel_esimd::simd<float, 32> v95_data(ir3.template select<32, 1>(192));
            ir3.template select<32, 1>(192) = (v95_data + v94_data);
            tensorforge::intel_esimd::simd<float, 32> v97_data(r2.template select<32, 1>(224));
            tensorforge::intel_esimd::simd<float, 32> v98_data(ir3.template select<32, 1>(224));
            ir3.template select<32, 1>(224) = (v98_data + v97_data);
            tensorforge::intel_esimd::simd<float, 32> v100_data(r2.template select<32, 1>(256));
            tensorforge::intel_esimd::simd<float, 32> v101_data(ir3.template select<32, 1>(256));
            ir3.template select<32, 1>(256) = (v101_data + v100_data);
            // r3 = ir3 + r1
            #pragma unroll
            for (int32_t v103_n1 = 0; v103_n1 < 9; ++v103_n1) {
              int32_t v104_a = v103_n1 * 32;
              tensorforge::intel_esimd::simd<float, 20> v106_data(ir3.template select<20, 1>(v104_a));
              tensorforge::intel_esimd::simd<float, 20> v107_data(r1.template select<20, 1>(v104_a));
              r3.template select<20, 1>(v104_a) = (v107_data + v106_data);
            }
            tensorforge::intel_esimd::simd<float, 288> r6(0.0f);
            // r6 = load{g>r}(glb_m4);
            #pragma unroll
            for (int32_t v110_i1 = 0; v110_i1 < 9; ++v110_i1) {
              float v112_data = glb_m4[v110_i1];
              r6[(v110_i1 * 32)] = v112_data;
            }
            // wait(r4 = load{g>r}(glb_m3););
            tensorforge::intel_esimd::simd<float, 288> r5(0.0f);
            // ir5 = +(r4)
            // [(0, 4), (0, 9)] []
            tensorforge::intel_esimd::simd<float, 288> ir5(0.0f);
            tensorforge::intel_esimd::simd<float, 32> v117_data(r4.template select<32, 1>(0));
            tensorforge::intel_esimd::simd<float, 32> v118_data(ir5.template select<32, 1>(0));
            ir5.template select<32, 1>(0) = (v118_data + v117_data);
            tensorforge::intel_esimd::simd<float, 32> v120_data(r4.template select<32, 1>(32));
            tensorforge::intel_esimd::simd<float, 32> v121_data(ir5.template select<32, 1>(32));
            ir5.template select<32, 1>(32) = (v121_data + v120_data);
            tensorforge::intel_esimd::simd<float, 32> v123_data(r4.template select<32, 1>(64));
            tensorforge::intel_esimd::simd<float, 32> v124_data(ir5.template select<32, 1>(64));
            ir5.template select<32, 1>(64) = (v124_data + v123_data);
            tensorforge::intel_esimd::simd<float, 32> v126_data(r4.template select<32, 1>(96));
            tensorforge::intel_esimd::simd<float, 32> v127_data(ir5.template select<32, 1>(96));
            ir5.template select<32, 1>(96) = (v127_data + v126_data);
            tensorforge::intel_esimd::simd<float, 32> v129_data(r4.template select<32, 1>(128));
            tensorforge::intel_esimd::simd<float, 32> v130_data(ir5.template select<32, 1>(128));
            ir5.template select<32, 1>(128) = (v130_data + v129_data);
            tensorforge::intel_esimd::simd<float, 32> v132_data(r4.template select<32, 1>(160));
            tensorforge::intel_esimd::simd<float, 32> v133_data(ir5.template select<32, 1>(160));
            ir5.template select<32, 1>(160) = (v133_data + v132_data);
            tensorforge::intel_esimd::simd<float, 32> v135_data(r4.template select<32, 1>(192));
            tensorforge::intel_esimd::simd<float, 32> v136_data(ir5.template select<32, 1>(192));
            ir5.template select<32, 1>(192) = (v136_data + v135_data);
            tensorforge::intel_esimd::simd<float, 32> v138_data(r4.template select<32, 1>(224));
            tensorforge::intel_esimd::simd<float, 32> v139_data(ir5.template select<32, 1>(224));
            ir5.template select<32, 1>(224) = (v139_data + v138_data);
            tensorforge::intel_esimd::simd<float, 32> v141_data(r4.template select<32, 1>(256));
            tensorforge::intel_esimd::simd<float, 32> v142_data(ir5.template select<32, 1>(256));
            ir5.template select<32, 1>(256) = (v142_data + v141_data);
            // r5 = ir5 + r3
            #pragma unroll
            for (int32_t v144_n1 = 0; v144_n1 < 9; ++v144_n1) {
              int32_t v145_a = v144_n1 * 32;
              tensorforge::intel_esimd::simd<float, 20> v147_data(ir5.template select<20, 1>(v145_a));
              tensorforge::intel_esimd::simd<float, 20> v148_data(r3.template select<20, 1>(v145_a));
              r5.template select<20, 1>(v145_a) = (v148_data + v147_data);
            }
            // wait(r6 = load{g>r}(glb_m4););
            tensorforge::intel_esimd::simd<float, 288> r7(0.0f);
            // ir7 = +(r6)
            // [(0, 1), (0, 9)] []
            tensorforge::intel_esimd::simd<float, 288> ir7(0.0f);
            tensorforge::intel_esimd::simd<float, 32> v152_data(r6.template select<32, 1>(0));
            tensorforge::intel_esimd::simd<float, 32> v153_data(ir7.template select<32, 1>(0));
            ir7.template select<32, 1>(0) = (v153_data + v152_data);
            tensorforge::intel_esimd::simd<float, 32> v155_data(r6.template select<32, 1>(32));
            tensorforge::intel_esimd::simd<float, 32> v156_data(ir7.template select<32, 1>(32));
            ir7.template select<32, 1>(32) = (v156_data + v155_data);
            tensorforge::intel_esimd::simd<float, 32> v158_data(r6.template select<32, 1>(64));
            tensorforge::intel_esimd::simd<float, 32> v159_data(ir7.template select<32, 1>(64));
            ir7.template select<32, 1>(64) = (v159_data + v158_data);
            tensorforge::intel_esimd::simd<float, 32> v161_data(r6.template select<32, 1>(96));
            tensorforge::intel_esimd::simd<float, 32> v162_data(ir7.template select<32, 1>(96));
            ir7.template select<32, 1>(96) = (v162_data + v161_data);
            tensorforge::intel_esimd::simd<float, 32> v164_data(r6.template select<32, 1>(128));
            tensorforge::intel_esimd::simd<float, 32> v165_data(ir7.template select<32, 1>(128));
            ir7.template select<32, 1>(128) = (v165_data + v164_data);
            tensorforge::intel_esimd::simd<float, 32> v167_data(r6.template select<32, 1>(160));
            tensorforge::intel_esimd::simd<float, 32> v168_data(ir7.template select<32, 1>(160));
            ir7.template select<32, 1>(160) = (v168_data + v167_data);
            tensorforge::intel_esimd::simd<float, 32> v170_data(r6.template select<32, 1>(192));
            tensorforge::intel_esimd::simd<float, 32> v171_data(ir7.template select<32, 1>(192));
            ir7.template select<32, 1>(192) = (v171_data + v170_data);
            tensorforge::intel_esimd::simd<float, 32> v173_data(r6.template select<32, 1>(224));
            tensorforge::intel_esimd::simd<float, 32> v174_data(ir7.template select<32, 1>(224));
            ir7.template select<32, 1>(224) = (v174_data + v173_data);
            tensorforge::intel_esimd::simd<float, 32> v176_data(r6.template select<32, 1>(256));
            tensorforge::intel_esimd::simd<float, 32> v177_data(ir7.template select<32, 1>(256));
            ir7.template select<32, 1>(256) = (v177_data + v176_data);
            // r7 = ir7 + r5
            #pragma unroll
            for (int32_t v179_n1 = 0; v179_n1 < 9; ++v179_n1) {
              int32_t v180_a = v179_n1 * 32;
              tensorforge::intel_esimd::simd<float, 20> v182_data(ir7.template select<20, 1>(v180_a));
              tensorforge::intel_esimd::simd<float, 20> v183_data(r5.template select<20, 1>(v180_a));
              r7.template select<20, 1>(v180_a) = (v183_data + v182_data);
            }
            // glb_m0 = store{r>g}(r7);
            #pragma unroll
            for (int32_t v185_i1 = 0; v185_i1 < 9; ++v185_i1) {
              tensorforge::intel_esimd::simd<float, 20> v188_data(r7.template select<20, 1>((v185_i1 * 32)));
              v188_data.copy_to(glb_m0 + ((v185_i1 * 20)));
            }
          }
        }
      }
    });
  });
}

