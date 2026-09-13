// === base name ===
kernel_47e60894ef69baf9

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_47e60894ef69baf9 = {{1, 8, 1}, 32, 64, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_47e60894ef69baf9(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_47e60894ef69baf9(const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_47e60894ef69baf9(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 8, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 0 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_47e60894ef69baf9(const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_47e60894ef69baf9(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_47e60894ef69baf9(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_47e60894ef69baf9(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      // generated with TensorForge. Version: 0.0.1
      // options: default
      // launch: 32 lanes (64 active) x 8 per block = block 1x8x1, 0 B shared, occupancy grid
      // operands:
      //   m0 64×13(64×13) {0..64}×{0..13} pointer_based
      //   m1 6(6) {0..6} none
      //   m2 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} pointer_based
      // operations:
      //   t0[i,j,l] = m0[i,j] × m1[l]
      //   m2[i,j,l]@{20..35}×{12..13}×{0..6} += t0[i,j,l]@{20..35}×{12..13}×{0..6}
      // tensorforge-meta: {"fp":"float","launch":{"active_threads":64,"block":[1,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"pointer_based","alias":"A","bbox":[[0,0],[64,13]],"name":"m0","ordered":false,"parts":1,"shape":[64,13],"variant":false},{"addressing":"none","alias":"v","bbox":[[0],[6]],"name":"m1","ordered":false,"parts":1,"shape":[6],"variant":false},{"addressing":"pointer_based","alias":"D","bbox":[[0,0,0],[64,13,6]],"name":"m2","ordered":false,"parts":1,"shape":[64,13,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0,0],[64,13,6]],"is_tmp":true,"name":"t0","offset":[0,0,0],"shape":[64,13,6]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[64,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[64,13]},{"addressing":"none","bbox":[[0],[6]],"is_tmp":false,"name":"m1","offset":[0],"shape":[6]}],"permute":[[0,1],[0]],"target":[[0,1],[2]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0,0],[15,1,6]],"is_tmp":false,"name":"m2","offset":[20,12,0],"shape":[64,13,6]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0,0],[15,1,6]],"is_tmp":true,"name":"t0","offset":[20,12,0],"shape":[64,13,6]}],"permute":[[0,1,2]],"target":[[0,1,2]]}],"version":"0.0.1\n"}
      {
        const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
        const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
        const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
        const float *const __restrict__ glb_m1 = &m1[0];
        for (size_t v2_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v2_batchId0 < numElements0; v2_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
          size_t v3_ahead1 = v2_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          size_t v5_batchId1 = (v3_ahead1 < numElements0) ? v3_ahead1 : v2_batchId0;
          const float *const __restrict__ pf_glb_m0 = &m0[v5_batchId1][0 + m0_extraOffset];
          float *const __restrict__ pf_glb_m2 = &m2[v5_batchId1][0 + m2_extraOffset];
          const bool allowed_next = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId1]);
          const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v2_batchId0]);
          if (allowed) {
            const float *const __restrict__ glb_m0 = &m0[v2_batchId0][0 + m0_extraOffset];
            float *const __restrict__ glb_m2 = &m2[v2_batchId0][0 + m2_extraOffset];
            float r0[832]{};
            // r0 = load{g>r}(glb_m0);
            #pragma unroll
            for (int32_t v16_i0 = 0; v16_i0 < 2; ++v16_i0) {
              int32_t v18_lead = v16_i0 * 32;
              #pragma unroll
              for (int32_t v17_i1 = 0; v17_i1 < 13; ++v17_i1) {
                int32_t v21_a = v18_lead + (v17_i1 * 64);
                tensorforge::intel_esimd::simd<float, 32> v22_data;
                v22_data.copy_from(glb_m0 + (v21_a));
                v22_data.copy_to(r0 + (v21_a));
              }
            }
            float r2[384]{};
            // r2 = load{g>r}(glb_m2);
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 1; ++v25_i1) {
              int32_t v33_a = 20_i32 + ((v25_i1 + 12) * 64);
              int32_t v38_a = 20 + (v25_i1 * 64);
              #pragma unroll
              for (int32_t v26_i2 = 0; v26_i2 < 6; ++v26_i2) {
                tensorforge::intel_esimd::simd<float, 12> v35_data;
                v35_data.copy_from(glb_m2 + ((v33_a + (v26_i2 * 832))));
                v35_data.copy_to(r2 + ((v38_a + (v26_i2 * 64))));
              }
            }
            #pragma unroll
            for (int32_t v40_i1 = 0; v40_i1 < 1; ++v40_i1) {
              int32_t v48_a = 32_i32 + ((v40_i1 + 12) * 64);
              int32_t v53_a = 32 + (v40_i1 * 64);
              #pragma unroll
              for (int32_t v41_i2 = 0; v41_i2 < 6; ++v41_i2) {
                tensorforge::intel_esimd::simd<float, 3> v50_data;
                v50_data.copy_from(glb_m2 + ((v48_a + (v41_i2 * 832))));
                v50_data.copy_to(r2 + ((v53_a + (v41_i2 * 64))));
              }
            }
            // wait(r0 = load{g>r}(glb_m0););
            float r1[4992]{};
            // r1 = +(r0 * glb_m1) + None
            // [(0, 64), (0, 13), (0, 6)] []
            tensorforge::intel_esimd::simd<float, 32> v56_data;
            v56_data.copy_from(r0 + (0));
            float v57_data = glb_m1[0];
            tensorforge::intel_esimd::simd<float, 32> v59_data;
            v59_data.copy_from(r1 + (0));
            (v59_data + (v56_data * v57_data)).copy_to(r1 + (0));
            float v62_data = glb_m1[1];
            tensorforge::intel_esimd::simd<float, 32> v64_data;
            v64_data.copy_from(r1 + (832));
            (v64_data + (v56_data * v62_data)).copy_to(r1 + (832));
            float v67_data = glb_m1[2];
            tensorforge::intel_esimd::simd<float, 32> v69_data;
            v69_data.copy_from(r1 + (1664));
            (v69_data + (v56_data * v67_data)).copy_to(r1 + (1664));
            float v72_data = glb_m1[3];
            tensorforge::intel_esimd::simd<float, 32> v74_data;
            v74_data.copy_from(r1 + (2496));
            (v74_data + (v56_data * v72_data)).copy_to(r1 + (2496));
            float v77_data = glb_m1[4];
            tensorforge::intel_esimd::simd<float, 32> v79_data;
            v79_data.copy_from(r1 + (3328));
            (v79_data + (v56_data * v77_data)).copy_to(r1 + (3328));
            float v82_data = glb_m1[5];
            tensorforge::intel_esimd::simd<float, 32> v84_data;
            v84_data.copy_from(r1 + (4160));
            (v84_data + (v56_data * v82_data)).copy_to(r1 + (4160));
            tensorforge::intel_esimd::simd<float, 32> v86_data;
            v86_data.copy_from(r0 + (64));
            tensorforge::intel_esimd::simd<float, 32> v89_data;
            v89_data.copy_from(r1 + (64));
            (v89_data + (v86_data * v57_data)).copy_to(r1 + (64));
            tensorforge::intel_esimd::simd<float, 32> v94_data;
            v94_data.copy_from(r1 + (896));
            (v94_data + (v86_data * v62_data)).copy_to(r1 + (896));
            tensorforge::intel_esimd::simd<float, 32> v99_data;
            v99_data.copy_from(r1 + (1728));
            (v99_data + (v86_data * v67_data)).copy_to(r1 + (1728));
            tensorforge::intel_esimd::simd<float, 32> v104_data;
            v104_data.copy_from(r1 + (2560));
            (v104_data + (v86_data * v72_data)).copy_to(r1 + (2560));
            tensorforge::intel_esimd::simd<float, 32> v109_data;
            v109_data.copy_from(r1 + (3392));
            (v109_data + (v86_data * v77_data)).copy_to(r1 + (3392));
            tensorforge::intel_esimd::simd<float, 32> v114_data;
            v114_data.copy_from(r1 + (4224));
            (v114_data + (v86_data * v82_data)).copy_to(r1 + (4224));
            tensorforge::intel_esimd::simd<float, 32> v116_data;
            v116_data.copy_from(r0 + (128));
            tensorforge::intel_esimd::simd<float, 32> v119_data;
            v119_data.copy_from(r1 + (128));
            (v119_data + (v116_data * v57_data)).copy_to(r1 + (128));
            tensorforge::intel_esimd::simd<float, 32> v124_data;
            v124_data.copy_from(r1 + (960));
            (v124_data + (v116_data * v62_data)).copy_to(r1 + (960));
            tensorforge::intel_esimd::simd<float, 32> v129_data;
            v129_data.copy_from(r1 + (1792));
            (v129_data + (v116_data * v67_data)).copy_to(r1 + (1792));
            tensorforge::intel_esimd::simd<float, 32> v134_data;
            v134_data.copy_from(r1 + (2624));
            (v134_data + (v116_data * v72_data)).copy_to(r1 + (2624));
            tensorforge::intel_esimd::simd<float, 32> v139_data;
            v139_data.copy_from(r1 + (3456));
            (v139_data + (v116_data * v77_data)).copy_to(r1 + (3456));
            tensorforge::intel_esimd::simd<float, 32> v144_data;
            v144_data.copy_from(r1 + (4288));
            (v144_data + (v116_data * v82_data)).copy_to(r1 + (4288));
            tensorforge::intel_esimd::simd<float, 32> v146_data;
            v146_data.copy_from(r0 + (192));
            tensorforge::intel_esimd::simd<float, 32> v149_data;
            v149_data.copy_from(r1 + (192));
            (v149_data + (v146_data * v57_data)).copy_to(r1 + (192));
            tensorforge::intel_esimd::simd<float, 32> v154_data;
            v154_data.copy_from(r1 + (1024));
            (v154_data + (v146_data * v62_data)).copy_to(r1 + (1024));
            tensorforge::intel_esimd::simd<float, 32> v159_data;
            v159_data.copy_from(r1 + (1856));
            (v159_data + (v146_data * v67_data)).copy_to(r1 + (1856));
            tensorforge::intel_esimd::simd<float, 32> v164_data;
            v164_data.copy_from(r1 + (2688));
            (v164_data + (v146_data * v72_data)).copy_to(r1 + (2688));
            tensorforge::intel_esimd::simd<float, 32> v169_data;
            v169_data.copy_from(r1 + (3520));
            (v169_data + (v146_data * v77_data)).copy_to(r1 + (3520));
            tensorforge::intel_esimd::simd<float, 32> v174_data;
            v174_data.copy_from(r1 + (4352));
            (v174_data + (v146_data * v82_data)).copy_to(r1 + (4352));
            tensorforge::intel_esimd::simd<float, 32> v176_data;
            v176_data.copy_from(r0 + (256));
            tensorforge::intel_esimd::simd<float, 32> v179_data;
            v179_data.copy_from(r1 + (256));
            (v179_data + (v176_data * v57_data)).copy_to(r1 + (256));
            tensorforge::intel_esimd::simd<float, 32> v184_data;
            v184_data.copy_from(r1 + (1088));
            (v184_data + (v176_data * v62_data)).copy_to(r1 + (1088));
            tensorforge::intel_esimd::simd<float, 32> v189_data;
            v189_data.copy_from(r1 + (1920));
            (v189_data + (v176_data * v67_data)).copy_to(r1 + (1920));
            tensorforge::intel_esimd::simd<float, 32> v194_data;
            v194_data.copy_from(r1 + (2752));
            (v194_data + (v176_data * v72_data)).copy_to(r1 + (2752));
            tensorforge::intel_esimd::simd<float, 32> v199_data;
            v199_data.copy_from(r1 + (3584));
            (v199_data + (v176_data * v77_data)).copy_to(r1 + (3584));
            tensorforge::intel_esimd::simd<float, 32> v204_data;
            v204_data.copy_from(r1 + (4416));
            (v204_data + (v176_data * v82_data)).copy_to(r1 + (4416));
            tensorforge::intel_esimd::simd<float, 32> v206_data;
            v206_data.copy_from(r0 + (320));
            tensorforge::intel_esimd::simd<float, 32> v209_data;
            v209_data.copy_from(r1 + (320));
            (v209_data + (v206_data * v57_data)).copy_to(r1 + (320));
            tensorforge::intel_esimd::simd<float, 32> v214_data;
            v214_data.copy_from(r1 + (1152));
            (v214_data + (v206_data * v62_data)).copy_to(r1 + (1152));
            tensorforge::intel_esimd::simd<float, 32> v219_data;
            v219_data.copy_from(r1 + (1984));
            (v219_data + (v206_data * v67_data)).copy_to(r1 + (1984));
            tensorforge::intel_esimd::simd<float, 32> v224_data;
            v224_data.copy_from(r1 + (2816));
            (v224_data + (v206_data * v72_data)).copy_to(r1 + (2816));
            tensorforge::intel_esimd::simd<float, 32> v229_data;
            v229_data.copy_from(r1 + (3648));
            (v229_data + (v206_data * v77_data)).copy_to(r1 + (3648));
            tensorforge::intel_esimd::simd<float, 32> v234_data;
            v234_data.copy_from(r1 + (4480));
            (v234_data + (v206_data * v82_data)).copy_to(r1 + (4480));
            tensorforge::intel_esimd::simd<float, 32> v236_data;
            v236_data.copy_from(r0 + (384));
            tensorforge::intel_esimd::simd<float, 32> v239_data;
            v239_data.copy_from(r1 + (384));
            (v239_data + (v236_data * v57_data)).copy_to(r1 + (384));
            tensorforge::intel_esimd::simd<float, 32> v244_data;
            v244_data.copy_from(r1 + (1216));
            (v244_data + (v236_data * v62_data)).copy_to(r1 + (1216));
            tensorforge::intel_esimd::simd<float, 32> v249_data;
            v249_data.copy_from(r1 + (2048));
            (v249_data + (v236_data * v67_data)).copy_to(r1 + (2048));
            tensorforge::intel_esimd::simd<float, 32> v254_data;
            v254_data.copy_from(r1 + (2880));
            (v254_data + (v236_data * v72_data)).copy_to(r1 + (2880));
            tensorforge::intel_esimd::simd<float, 32> v259_data;
            v259_data.copy_from(r1 + (3712));
            (v259_data + (v236_data * v77_data)).copy_to(r1 + (3712));
            tensorforge::intel_esimd::simd<float, 32> v264_data;
            v264_data.copy_from(r1 + (4544));
            (v264_data + (v236_data * v82_data)).copy_to(r1 + (4544));
            tensorforge::intel_esimd::simd<float, 32> v266_data;
            v266_data.copy_from(r0 + (448));
            tensorforge::intel_esimd::simd<float, 32> v269_data;
            v269_data.copy_from(r1 + (448));
            (v269_data + (v266_data * v57_data)).copy_to(r1 + (448));
            tensorforge::intel_esimd::simd<float, 32> v274_data;
            v274_data.copy_from(r1 + (1280));
            (v274_data + (v266_data * v62_data)).copy_to(r1 + (1280));
            tensorforge::intel_esimd::simd<float, 32> v279_data;
            v279_data.copy_from(r1 + (2112));
            (v279_data + (v266_data * v67_data)).copy_to(r1 + (2112));
            tensorforge::intel_esimd::simd<float, 32> v284_data;
            v284_data.copy_from(r1 + (2944));
            (v284_data + (v266_data * v72_data)).copy_to(r1 + (2944));
            tensorforge::intel_esimd::simd<float, 32> v289_data;
            v289_data.copy_from(r1 + (3776));
            (v289_data + (v266_data * v77_data)).copy_to(r1 + (3776));
            tensorforge::intel_esimd::simd<float, 32> v294_data;
            v294_data.copy_from(r1 + (4608));
            (v294_data + (v266_data * v82_data)).copy_to(r1 + (4608));
            tensorforge::intel_esimd::simd<float, 32> v296_data;
            v296_data.copy_from(r0 + (512));
            tensorforge::intel_esimd::simd<float, 32> v299_data;
            v299_data.copy_from(r1 + (512));
            (v299_data + (v296_data * v57_data)).copy_to(r1 + (512));
            tensorforge::intel_esimd::simd<float, 32> v304_data;
            v304_data.copy_from(r1 + (1344));
            (v304_data + (v296_data * v62_data)).copy_to(r1 + (1344));
            tensorforge::intel_esimd::simd<float, 32> v309_data;
            v309_data.copy_from(r1 + (2176));
            (v309_data + (v296_data * v67_data)).copy_to(r1 + (2176));
            tensorforge::intel_esimd::simd<float, 32> v314_data;
            v314_data.copy_from(r1 + (3008));
            (v314_data + (v296_data * v72_data)).copy_to(r1 + (3008));
            tensorforge::intel_esimd::simd<float, 32> v319_data;
            v319_data.copy_from(r1 + (3840));
            (v319_data + (v296_data * v77_data)).copy_to(r1 + (3840));
            tensorforge::intel_esimd::simd<float, 32> v324_data;
            v324_data.copy_from(r1 + (4672));
            (v324_data + (v296_data * v82_data)).copy_to(r1 + (4672));
            tensorforge::intel_esimd::simd<float, 32> v326_data;
            v326_data.copy_from(r0 + (576));
            tensorforge::intel_esimd::simd<float, 32> v329_data;
            v329_data.copy_from(r1 + (576));
            (v329_data + (v326_data * v57_data)).copy_to(r1 + (576));
            tensorforge::intel_esimd::simd<float, 32> v334_data;
            v334_data.copy_from(r1 + (1408));
            (v334_data + (v326_data * v62_data)).copy_to(r1 + (1408));
            tensorforge::intel_esimd::simd<float, 32> v339_data;
            v339_data.copy_from(r1 + (2240));
            (v339_data + (v326_data * v67_data)).copy_to(r1 + (2240));
            tensorforge::intel_esimd::simd<float, 32> v344_data;
            v344_data.copy_from(r1 + (3072));
            (v344_data + (v326_data * v72_data)).copy_to(r1 + (3072));
            tensorforge::intel_esimd::simd<float, 32> v349_data;
            v349_data.copy_from(r1 + (3904));
            (v349_data + (v326_data * v77_data)).copy_to(r1 + (3904));
            tensorforge::intel_esimd::simd<float, 32> v354_data;
            v354_data.copy_from(r1 + (4736));
            (v354_data + (v326_data * v82_data)).copy_to(r1 + (4736));
            tensorforge::intel_esimd::simd<float, 32> v356_data;
            v356_data.copy_from(r0 + (640));
            tensorforge::intel_esimd::simd<float, 32> v359_data;
            v359_data.copy_from(r1 + (640));
            (v359_data + (v356_data * v57_data)).copy_to(r1 + (640));
            tensorforge::intel_esimd::simd<float, 32> v364_data;
            v364_data.copy_from(r1 + (1472));
            (v364_data + (v356_data * v62_data)).copy_to(r1 + (1472));
            tensorforge::intel_esimd::simd<float, 32> v369_data;
            v369_data.copy_from(r1 + (2304));
            (v369_data + (v356_data * v67_data)).copy_to(r1 + (2304));
            tensorforge::intel_esimd::simd<float, 32> v374_data;
            v374_data.copy_from(r1 + (3136));
            (v374_data + (v356_data * v72_data)).copy_to(r1 + (3136));
            tensorforge::intel_esimd::simd<float, 32> v379_data;
            v379_data.copy_from(r1 + (3968));
            (v379_data + (v356_data * v77_data)).copy_to(r1 + (3968));
            tensorforge::intel_esimd::simd<float, 32> v384_data;
            v384_data.copy_from(r1 + (4800));
            (v384_data + (v356_data * v82_data)).copy_to(r1 + (4800));
            tensorforge::intel_esimd::simd<float, 32> v386_data;
            v386_data.copy_from(r0 + (704));
            tensorforge::intel_esimd::simd<float, 32> v389_data;
            v389_data.copy_from(r1 + (704));
            (v389_data + (v386_data * v57_data)).copy_to(r1 + (704));
            tensorforge::intel_esimd::simd<float, 32> v394_data;
            v394_data.copy_from(r1 + (1536));
            (v394_data + (v386_data * v62_data)).copy_to(r1 + (1536));
            tensorforge::intel_esimd::simd<float, 32> v399_data;
            v399_data.copy_from(r1 + (2368));
            (v399_data + (v386_data * v67_data)).copy_to(r1 + (2368));
            tensorforge::intel_esimd::simd<float, 32> v404_data;
            v404_data.copy_from(r1 + (3200));
            (v404_data + (v386_data * v72_data)).copy_to(r1 + (3200));
            tensorforge::intel_esimd::simd<float, 32> v409_data;
            v409_data.copy_from(r1 + (4032));
            (v409_data + (v386_data * v77_data)).copy_to(r1 + (4032));
            tensorforge::intel_esimd::simd<float, 32> v414_data;
            v414_data.copy_from(r1 + (4864));
            (v414_data + (v386_data * v82_data)).copy_to(r1 + (4864));
            tensorforge::intel_esimd::simd<float, 32> v416_data;
            v416_data.copy_from(r0 + (768));
            tensorforge::intel_esimd::simd<float, 32> v419_data;
            v419_data.copy_from(r1 + (768));
            (v419_data + (v416_data * v57_data)).copy_to(r1 + (768));
            tensorforge::intel_esimd::simd<float, 32> v424_data;
            v424_data.copy_from(r1 + (1600));
            (v424_data + (v416_data * v62_data)).copy_to(r1 + (1600));
            tensorforge::intel_esimd::simd<float, 32> v429_data;
            v429_data.copy_from(r1 + (2432));
            (v429_data + (v416_data * v67_data)).copy_to(r1 + (2432));
            tensorforge::intel_esimd::simd<float, 32> v434_data;
            v434_data.copy_from(r1 + (3264));
            (v434_data + (v416_data * v72_data)).copy_to(r1 + (3264));
            tensorforge::intel_esimd::simd<float, 32> v439_data;
            v439_data.copy_from(r1 + (4096));
            (v439_data + (v416_data * v77_data)).copy_to(r1 + (4096));
            tensorforge::intel_esimd::simd<float, 32> v444_data;
            v444_data.copy_from(r1 + (4928));
            (v444_data + (v416_data * v82_data)).copy_to(r1 + (4928));
            tensorforge::intel_esimd::simd<float, 32> v446_data;
            v446_data.copy_from(r0 + (32));
            tensorforge::intel_esimd::simd<float, 32> v449_data;
            v449_data.copy_from(r1 + (32));
            (v449_data + (v446_data * v57_data)).copy_to(r1 + (32));
            tensorforge::intel_esimd::simd<float, 32> v454_data;
            v454_data.copy_from(r1 + (864));
            (v454_data + (v446_data * v62_data)).copy_to(r1 + (864));
            tensorforge::intel_esimd::simd<float, 32> v459_data;
            v459_data.copy_from(r1 + (1696));
            (v459_data + (v446_data * v67_data)).copy_to(r1 + (1696));
            tensorforge::intel_esimd::simd<float, 32> v464_data;
            v464_data.copy_from(r1 + (2528));
            (v464_data + (v446_data * v72_data)).copy_to(r1 + (2528));
            tensorforge::intel_esimd::simd<float, 32> v469_data;
            v469_data.copy_from(r1 + (3360));
            (v469_data + (v446_data * v77_data)).copy_to(r1 + (3360));
            tensorforge::intel_esimd::simd<float, 32> v474_data;
            v474_data.copy_from(r1 + (4192));
            (v474_data + (v446_data * v82_data)).copy_to(r1 + (4192));
            tensorforge::intel_esimd::simd<float, 32> v476_data;
            v476_data.copy_from(r0 + (96));
            tensorforge::intel_esimd::simd<float, 32> v479_data;
            v479_data.copy_from(r1 + (96));
            (v479_data + (v476_data * v57_data)).copy_to(r1 + (96));
            tensorforge::intel_esimd::simd<float, 32> v484_data;
            v484_data.copy_from(r1 + (928));
            (v484_data + (v476_data * v62_data)).copy_to(r1 + (928));
            tensorforge::intel_esimd::simd<float, 32> v489_data;
            v489_data.copy_from(r1 + (1760));
            (v489_data + (v476_data * v67_data)).copy_to(r1 + (1760));
            tensorforge::intel_esimd::simd<float, 32> v494_data;
            v494_data.copy_from(r1 + (2592));
            (v494_data + (v476_data * v72_data)).copy_to(r1 + (2592));
            tensorforge::intel_esimd::simd<float, 32> v499_data;
            v499_data.copy_from(r1 + (3424));
            (v499_data + (v476_data * v77_data)).copy_to(r1 + (3424));
            tensorforge::intel_esimd::simd<float, 32> v504_data;
            v504_data.copy_from(r1 + (4256));
            (v504_data + (v476_data * v82_data)).copy_to(r1 + (4256));
            tensorforge::intel_esimd::simd<float, 32> v506_data;
            v506_data.copy_from(r0 + (160));
            tensorforge::intel_esimd::simd<float, 32> v509_data;
            v509_data.copy_from(r1 + (160));
            (v509_data + (v506_data * v57_data)).copy_to(r1 + (160));
            tensorforge::intel_esimd::simd<float, 32> v514_data;
            v514_data.copy_from(r1 + (992));
            (v514_data + (v506_data * v62_data)).copy_to(r1 + (992));
            tensorforge::intel_esimd::simd<float, 32> v519_data;
            v519_data.copy_from(r1 + (1824));
            (v519_data + (v506_data * v67_data)).copy_to(r1 + (1824));
            tensorforge::intel_esimd::simd<float, 32> v524_data;
            v524_data.copy_from(r1 + (2656));
            (v524_data + (v506_data * v72_data)).copy_to(r1 + (2656));
            tensorforge::intel_esimd::simd<float, 32> v529_data;
            v529_data.copy_from(r1 + (3488));
            (v529_data + (v506_data * v77_data)).copy_to(r1 + (3488));
            tensorforge::intel_esimd::simd<float, 32> v534_data;
            v534_data.copy_from(r1 + (4320));
            (v534_data + (v506_data * v82_data)).copy_to(r1 + (4320));
            tensorforge::intel_esimd::simd<float, 32> v536_data;
            v536_data.copy_from(r0 + (224));
            tensorforge::intel_esimd::simd<float, 32> v539_data;
            v539_data.copy_from(r1 + (224));
            (v539_data + (v536_data * v57_data)).copy_to(r1 + (224));
            tensorforge::intel_esimd::simd<float, 32> v544_data;
            v544_data.copy_from(r1 + (1056));
            (v544_data + (v536_data * v62_data)).copy_to(r1 + (1056));
            tensorforge::intel_esimd::simd<float, 32> v549_data;
            v549_data.copy_from(r1 + (1888));
            (v549_data + (v536_data * v67_data)).copy_to(r1 + (1888));
            tensorforge::intel_esimd::simd<float, 32> v554_data;
            v554_data.copy_from(r1 + (2720));
            (v554_data + (v536_data * v72_data)).copy_to(r1 + (2720));
            tensorforge::intel_esimd::simd<float, 32> v559_data;
            v559_data.copy_from(r1 + (3552));
            (v559_data + (v536_data * v77_data)).copy_to(r1 + (3552));
            tensorforge::intel_esimd::simd<float, 32> v564_data;
            v564_data.copy_from(r1 + (4384));
            (v564_data + (v536_data * v82_data)).copy_to(r1 + (4384));
            tensorforge::intel_esimd::simd<float, 32> v566_data;
            v566_data.copy_from(r0 + (288));
            tensorforge::intel_esimd::simd<float, 32> v569_data;
            v569_data.copy_from(r1 + (288));
            (v569_data + (v566_data * v57_data)).copy_to(r1 + (288));
            tensorforge::intel_esimd::simd<float, 32> v574_data;
            v574_data.copy_from(r1 + (1120));
            (v574_data + (v566_data * v62_data)).copy_to(r1 + (1120));
            tensorforge::intel_esimd::simd<float, 32> v579_data;
            v579_data.copy_from(r1 + (1952));
            (v579_data + (v566_data * v67_data)).copy_to(r1 + (1952));
            tensorforge::intel_esimd::simd<float, 32> v584_data;
            v584_data.copy_from(r1 + (2784));
            (v584_data + (v566_data * v72_data)).copy_to(r1 + (2784));
            tensorforge::intel_esimd::simd<float, 32> v589_data;
            v589_data.copy_from(r1 + (3616));
            (v589_data + (v566_data * v77_data)).copy_to(r1 + (3616));
            tensorforge::intel_esimd::simd<float, 32> v594_data;
            v594_data.copy_from(r1 + (4448));
            (v594_data + (v566_data * v82_data)).copy_to(r1 + (4448));
            tensorforge::intel_esimd::simd<float, 32> v596_data;
            v596_data.copy_from(r0 + (352));
            tensorforge::intel_esimd::simd<float, 32> v599_data;
            v599_data.copy_from(r1 + (352));
            (v599_data + (v596_data * v57_data)).copy_to(r1 + (352));
            tensorforge::intel_esimd::simd<float, 32> v604_data;
            v604_data.copy_from(r1 + (1184));
            (v604_data + (v596_data * v62_data)).copy_to(r1 + (1184));
            tensorforge::intel_esimd::simd<float, 32> v609_data;
            v609_data.copy_from(r1 + (2016));
            (v609_data + (v596_data * v67_data)).copy_to(r1 + (2016));
            tensorforge::intel_esimd::simd<float, 32> v614_data;
            v614_data.copy_from(r1 + (2848));
            (v614_data + (v596_data * v72_data)).copy_to(r1 + (2848));
            tensorforge::intel_esimd::simd<float, 32> v619_data;
            v619_data.copy_from(r1 + (3680));
            (v619_data + (v596_data * v77_data)).copy_to(r1 + (3680));
            tensorforge::intel_esimd::simd<float, 32> v624_data;
            v624_data.copy_from(r1 + (4512));
            (v624_data + (v596_data * v82_data)).copy_to(r1 + (4512));
            tensorforge::intel_esimd::simd<float, 32> v626_data;
            v626_data.copy_from(r0 + (416));
            tensorforge::intel_esimd::simd<float, 32> v629_data;
            v629_data.copy_from(r1 + (416));
            (v629_data + (v626_data * v57_data)).copy_to(r1 + (416));
            tensorforge::intel_esimd::simd<float, 32> v634_data;
            v634_data.copy_from(r1 + (1248));
            (v634_data + (v626_data * v62_data)).copy_to(r1 + (1248));
            tensorforge::intel_esimd::simd<float, 32> v639_data;
            v639_data.copy_from(r1 + (2080));
            (v639_data + (v626_data * v67_data)).copy_to(r1 + (2080));
            tensorforge::intel_esimd::simd<float, 32> v644_data;
            v644_data.copy_from(r1 + (2912));
            (v644_data + (v626_data * v72_data)).copy_to(r1 + (2912));
            tensorforge::intel_esimd::simd<float, 32> v649_data;
            v649_data.copy_from(r1 + (3744));
            (v649_data + (v626_data * v77_data)).copy_to(r1 + (3744));
            tensorforge::intel_esimd::simd<float, 32> v654_data;
            v654_data.copy_from(r1 + (4576));
            (v654_data + (v626_data * v82_data)).copy_to(r1 + (4576));
            tensorforge::intel_esimd::simd<float, 32> v656_data;
            v656_data.copy_from(r0 + (480));
            tensorforge::intel_esimd::simd<float, 32> v659_data;
            v659_data.copy_from(r1 + (480));
            (v659_data + (v656_data * v57_data)).copy_to(r1 + (480));
            tensorforge::intel_esimd::simd<float, 32> v664_data;
            v664_data.copy_from(r1 + (1312));
            (v664_data + (v656_data * v62_data)).copy_to(r1 + (1312));
            tensorforge::intel_esimd::simd<float, 32> v669_data;
            v669_data.copy_from(r1 + (2144));
            (v669_data + (v656_data * v67_data)).copy_to(r1 + (2144));
            tensorforge::intel_esimd::simd<float, 32> v674_data;
            v674_data.copy_from(r1 + (2976));
            (v674_data + (v656_data * v72_data)).copy_to(r1 + (2976));
            tensorforge::intel_esimd::simd<float, 32> v679_data;
            v679_data.copy_from(r1 + (3808));
            (v679_data + (v656_data * v77_data)).copy_to(r1 + (3808));
            tensorforge::intel_esimd::simd<float, 32> v684_data;
            v684_data.copy_from(r1 + (4640));
            (v684_data + (v656_data * v82_data)).copy_to(r1 + (4640));
            tensorforge::intel_esimd::simd<float, 32> v686_data;
            v686_data.copy_from(r0 + (544));
            tensorforge::intel_esimd::simd<float, 32> v689_data;
            v689_data.copy_from(r1 + (544));
            (v689_data + (v686_data * v57_data)).copy_to(r1 + (544));
            tensorforge::intel_esimd::simd<float, 32> v694_data;
            v694_data.copy_from(r1 + (1376));
            (v694_data + (v686_data * v62_data)).copy_to(r1 + (1376));
            tensorforge::intel_esimd::simd<float, 32> v699_data;
            v699_data.copy_from(r1 + (2208));
            (v699_data + (v686_data * v67_data)).copy_to(r1 + (2208));
            tensorforge::intel_esimd::simd<float, 32> v704_data;
            v704_data.copy_from(r1 + (3040));
            (v704_data + (v686_data * v72_data)).copy_to(r1 + (3040));
            tensorforge::intel_esimd::simd<float, 32> v709_data;
            v709_data.copy_from(r1 + (3872));
            (v709_data + (v686_data * v77_data)).copy_to(r1 + (3872));
            tensorforge::intel_esimd::simd<float, 32> v714_data;
            v714_data.copy_from(r1 + (4704));
            (v714_data + (v686_data * v82_data)).copy_to(r1 + (4704));
            tensorforge::intel_esimd::simd<float, 32> v716_data;
            v716_data.copy_from(r0 + (608));
            tensorforge::intel_esimd::simd<float, 32> v719_data;
            v719_data.copy_from(r1 + (608));
            (v719_data + (v716_data * v57_data)).copy_to(r1 + (608));
            tensorforge::intel_esimd::simd<float, 32> v724_data;
            v724_data.copy_from(r1 + (1440));
            (v724_data + (v716_data * v62_data)).copy_to(r1 + (1440));
            tensorforge::intel_esimd::simd<float, 32> v729_data;
            v729_data.copy_from(r1 + (2272));
            (v729_data + (v716_data * v67_data)).copy_to(r1 + (2272));
            tensorforge::intel_esimd::simd<float, 32> v734_data;
            v734_data.copy_from(r1 + (3104));
            (v734_data + (v716_data * v72_data)).copy_to(r1 + (3104));
            tensorforge::intel_esimd::simd<float, 32> v739_data;
            v739_data.copy_from(r1 + (3936));
            (v739_data + (v716_data * v77_data)).copy_to(r1 + (3936));
            tensorforge::intel_esimd::simd<float, 32> v744_data;
            v744_data.copy_from(r1 + (4768));
            (v744_data + (v716_data * v82_data)).copy_to(r1 + (4768));
            tensorforge::intel_esimd::simd<float, 32> v746_data;
            v746_data.copy_from(r0 + (672));
            tensorforge::intel_esimd::simd<float, 32> v749_data;
            v749_data.copy_from(r1 + (672));
            (v749_data + (v746_data * v57_data)).copy_to(r1 + (672));
            tensorforge::intel_esimd::simd<float, 32> v754_data;
            v754_data.copy_from(r1 + (1504));
            (v754_data + (v746_data * v62_data)).copy_to(r1 + (1504));
            tensorforge::intel_esimd::simd<float, 32> v759_data;
            v759_data.copy_from(r1 + (2336));
            (v759_data + (v746_data * v67_data)).copy_to(r1 + (2336));
            tensorforge::intel_esimd::simd<float, 32> v764_data;
            v764_data.copy_from(r1 + (3168));
            (v764_data + (v746_data * v72_data)).copy_to(r1 + (3168));
            tensorforge::intel_esimd::simd<float, 32> v769_data;
            v769_data.copy_from(r1 + (4000));
            (v769_data + (v746_data * v77_data)).copy_to(r1 + (4000));
            tensorforge::intel_esimd::simd<float, 32> v774_data;
            v774_data.copy_from(r1 + (4832));
            (v774_data + (v746_data * v82_data)).copy_to(r1 + (4832));
            tensorforge::intel_esimd::simd<float, 32> v776_data;
            v776_data.copy_from(r0 + (736));
            tensorforge::intel_esimd::simd<float, 32> v779_data;
            v779_data.copy_from(r1 + (736));
            (v779_data + (v776_data * v57_data)).copy_to(r1 + (736));
            tensorforge::intel_esimd::simd<float, 32> v784_data;
            v784_data.copy_from(r1 + (1568));
            (v784_data + (v776_data * v62_data)).copy_to(r1 + (1568));
            tensorforge::intel_esimd::simd<float, 32> v789_data;
            v789_data.copy_from(r1 + (2400));
            (v789_data + (v776_data * v67_data)).copy_to(r1 + (2400));
            tensorforge::intel_esimd::simd<float, 32> v794_data;
            v794_data.copy_from(r1 + (3232));
            (v794_data + (v776_data * v72_data)).copy_to(r1 + (3232));
            tensorforge::intel_esimd::simd<float, 32> v799_data;
            v799_data.copy_from(r1 + (4064));
            (v799_data + (v776_data * v77_data)).copy_to(r1 + (4064));
            tensorforge::intel_esimd::simd<float, 32> v804_data;
            v804_data.copy_from(r1 + (4896));
            (v804_data + (v776_data * v82_data)).copy_to(r1 + (4896));
            tensorforge::intel_esimd::simd<float, 32> v806_data;
            v806_data.copy_from(r0 + (800));
            tensorforge::intel_esimd::simd<float, 32> v809_data;
            v809_data.copy_from(r1 + (800));
            (v809_data + (v806_data * v57_data)).copy_to(r1 + (800));
            tensorforge::intel_esimd::simd<float, 32> v814_data;
            v814_data.copy_from(r1 + (1632));
            (v814_data + (v806_data * v62_data)).copy_to(r1 + (1632));
            tensorforge::intel_esimd::simd<float, 32> v819_data;
            v819_data.copy_from(r1 + (2464));
            (v819_data + (v806_data * v67_data)).copy_to(r1 + (2464));
            tensorforge::intel_esimd::simd<float, 32> v824_data;
            v824_data.copy_from(r1 + (3296));
            (v824_data + (v806_data * v72_data)).copy_to(r1 + (3296));
            tensorforge::intel_esimd::simd<float, 32> v829_data;
            v829_data.copy_from(r1 + (4128));
            (v829_data + (v806_data * v77_data)).copy_to(r1 + (4128));
            tensorforge::intel_esimd::simd<float, 32> v834_data;
            v834_data.copy_from(r1 + (4960));
            (v834_data + (v806_data * v82_data)).copy_to(r1 + (4960));
            // wait(r2 = load{g>r}(glb_m2););
            float r3[384]{};
            // r3 = +(r1) + name: r2, type: SymbolType.Register, lead: [0]
            // [(20, 35), (0, 1), (0, 6)] []
            float ir3[384]{};
            tensorforge::intel_esimd::simd<float, 12> v838_data;
            v838_data.copy_from(r1 + (788));
            tensorforge::intel_esimd::simd<float, 12> v839_data;
            v839_data.copy_from(ir3 + (20));
            (v839_data + v838_data).copy_to(ir3 + (20));
            tensorforge::intel_esimd::simd<float, 12> v841_data;
            v841_data.copy_from(r1 + (1620));
            tensorforge::intel_esimd::simd<float, 12> v842_data;
            v842_data.copy_from(ir3 + (84));
            (v842_data + v841_data).copy_to(ir3 + (84));
            tensorforge::intel_esimd::simd<float, 12> v844_data;
            v844_data.copy_from(r1 + (2452));
            tensorforge::intel_esimd::simd<float, 12> v845_data;
            v845_data.copy_from(ir3 + (148));
            (v845_data + v844_data).copy_to(ir3 + (148));
            tensorforge::intel_esimd::simd<float, 12> v847_data;
            v847_data.copy_from(r1 + (3284));
            tensorforge::intel_esimd::simd<float, 12> v848_data;
            v848_data.copy_from(ir3 + (212));
            (v848_data + v847_data).copy_to(ir3 + (212));
            tensorforge::intel_esimd::simd<float, 12> v850_data;
            v850_data.copy_from(r1 + (4116));
            tensorforge::intel_esimd::simd<float, 12> v851_data;
            v851_data.copy_from(ir3 + (276));
            (v851_data + v850_data).copy_to(ir3 + (276));
            tensorforge::intel_esimd::simd<float, 12> v853_data;
            v853_data.copy_from(r1 + (4948));
            tensorforge::intel_esimd::simd<float, 12> v854_data;
            v854_data.copy_from(ir3 + (340));
            (v854_data + v853_data).copy_to(ir3 + (340));
            tensorforge::intel_esimd::simd<float, 3> v856_data;
            v856_data.copy_from(r1 + (800));
            tensorforge::intel_esimd::simd<float, 3> v857_data;
            v857_data.copy_from(ir3 + (32));
            (v857_data + v856_data).copy_to(ir3 + (32));
            tensorforge::intel_esimd::simd<float, 3> v859_data;
            v859_data.copy_from(r1 + (1632));
            tensorforge::intel_esimd::simd<float, 3> v860_data;
            v860_data.copy_from(ir3 + (96));
            (v860_data + v859_data).copy_to(ir3 + (96));
            tensorforge::intel_esimd::simd<float, 3> v862_data;
            v862_data.copy_from(r1 + (2464));
            tensorforge::intel_esimd::simd<float, 3> v863_data;
            v863_data.copy_from(ir3 + (160));
            (v863_data + v862_data).copy_to(ir3 + (160));
            tensorforge::intel_esimd::simd<float, 3> v865_data;
            v865_data.copy_from(r1 + (3296));
            tensorforge::intel_esimd::simd<float, 3> v866_data;
            v866_data.copy_from(ir3 + (224));
            (v866_data + v865_data).copy_to(ir3 + (224));
            tensorforge::intel_esimd::simd<float, 3> v868_data;
            v868_data.copy_from(r1 + (4128));
            tensorforge::intel_esimd::simd<float, 3> v869_data;
            v869_data.copy_from(ir3 + (288));
            (v869_data + v868_data).copy_to(ir3 + (288));
            tensorforge::intel_esimd::simd<float, 3> v871_data;
            v871_data.copy_from(r1 + (4960));
            tensorforge::intel_esimd::simd<float, 3> v872_data;
            v872_data.copy_from(ir3 + (352));
            (v872_data + v871_data).copy_to(ir3 + (352));
            #pragma unroll
            for (int32_t v874_n1 = 0; v874_n1 < 1; ++v874_n1) {
              int32_t v878_a = 20 + (v874_n1 * 64);
              #pragma unroll
              for (int32_t v875_n2 = 0; v875_n2 < 6; ++v875_n2) {
                int32_t v879_a = v878_a + (v875_n2 * 64);
                tensorforge::intel_esimd::simd<float, 12> v880_data;
                v880_data.copy_from(ir3 + (v879_a));
                tensorforge::intel_esimd::simd<float, 12> v881_data;
                v881_data.copy_from(r2 + (v879_a));
                (v881_data + v880_data).copy_to(r3 + (v879_a));
              }
            }
            #pragma unroll
            for (int32_t v883_n1 = 0; v883_n1 < 1; ++v883_n1) {
              int32_t v887_a = 32 + (v883_n1 * 64);
              #pragma unroll
              for (int32_t v884_n2 = 0; v884_n2 < 6; ++v884_n2) {
                int32_t v888_a = v887_a + (v884_n2 * 64);
                tensorforge::intel_esimd::simd<float, 3> v889_data;
                v889_data.copy_from(ir3 + (v888_a));
                tensorforge::intel_esimd::simd<float, 3> v890_data;
                v890_data.copy_from(r2 + (v888_a));
                (v890_data + v889_data).copy_to(r3 + (v888_a));
              }
            }
            // glb_m2 = store{r>g}(r3);
            #pragma unroll
            for (int32_t v892_i1 = 0; v892_i1 < 1; ++v892_i1) {
              int32_t v896_a = 20 + (v892_i1 * 64);
              int32_t v905_a = 20_i32 + ((v892_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v893_i2 = 0; v893_i2 < 6; ++v893_i2) {
                tensorforge::intel_esimd::simd<float, 12> v898_data;
                v898_data.copy_from(r3 + ((v896_a + (v893_i2 * 64))));
                v898_data.copy_to(glb_m2 + ((v905_a + (v893_i2 * 832))));
              }
            }
            #pragma unroll
            for (int32_t v907_i1 = 0; v907_i1 < 1; ++v907_i1) {
              int32_t v911_a = 32 + (v907_i1 * 64);
              int32_t v920_a = 32_i32 + ((v907_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v908_i2 = 0; v908_i2 < 6; ++v908_i2) {
                tensorforge::intel_esimd::simd<float, 3> v913_data;
                v913_data.copy_from(r3 + ((v911_a + (v908_i2 * 64))));
                v913_data.copy_to(glb_m2 + ((v920_a + (v908_i2 * 832))));
              }
            }
          }
          if (allowed_next) {
            tensorforge::prefetchL2<496>(&pf_glb_m0[0]);
          }
          if (allowed_next) {
            tensorforge::prefetchL2<336>(&pf_glb_m0[496]);
          }
          if (allowed_next) {
            tensorforge::prefetchL2<496>(&pf_glb_m2[0]);
          }
          if (allowed_next) {
            tensorforge::prefetchL2<496>(&pf_glb_m2[496]);
          }
          if (allowed_next) {
            tensorforge::prefetchL2<496>(&pf_glb_m2[992]);
          }
          if (allowed_next) {
            tensorforge::prefetchL2<496>(&pf_glb_m2[1488]);
          }
          if (allowed_next) {
            tensorforge::prefetchL2<496>(&pf_glb_m2[1984]);
          }
          if (allowed_next) {
            tensorforge::prefetchL2<496>(&pf_glb_m2[2480]);
          }
          if (allowed_next) {
            tensorforge::prefetchL2<496>(&pf_glb_m2[2976]);
          }
          if (allowed_next) {
            tensorforge::prefetchL2<496>(&pf_glb_m2[3472]);
          }
          if (allowed_next) {
            tensorforge::prefetchL2<496>(&pf_glb_m2[3968]);
          }
          if (allowed_next) {
            tensorforge::prefetchL2<496>(&pf_glb_m2[4464]);
          }
          if (allowed_next) {
            tensorforge::prefetchL2<32>(&pf_glb_m2[4960]);
          }
        }
      }
    });
  });
}

