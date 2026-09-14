// === base name ===
kernel_baff1e485e95b5ba

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_baff1e485e95b5ba = {{1, 8, 1}, 32, 64, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_baff1e485e95b5ba(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_baff1e485e95b5ba(const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_baff1e485e95b5ba(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_baff1e485e95b5ba(const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_baff1e485e95b5ba(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_baff1e485e95b5ba(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_baff1e485e95b5ba(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
          const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v2_batchId0]);
          if (allowed) {
            const float *const __restrict__ glb_m0 = &m0[v2_batchId0][0 + m0_extraOffset];
            float *const __restrict__ glb_m2 = &m2[v2_batchId0][0 + m2_extraOffset];
            float r0[832]{};
            // r0 = load{g>r}(glb_m0);
            #pragma unroll
            for (int32_t v13_i0 = 0; v13_i0 < 2; ++v13_i0) {
              int32_t v15_lead = v13_i0 * 32;
              #pragma unroll
              for (int32_t v14_i1 = 0; v14_i1 < 13; ++v14_i1) {
                int32_t v18_a = v15_lead + (v14_i1 * 64);
                tensorforge::intel_esimd::simd<float, 32> v19_data;
                v19_data.copy_from(glb_m0 + (v18_a));
                v19_data.copy_to(r0 + (v18_a));
              }
            }
            float r2[384]{};
            // r2 = load{g>r}(glb_m2);
            #pragma unroll
            for (int32_t v22_i1 = 0; v22_i1 < 1; ++v22_i1) {
              int32_t v30_a = 20_i32 + ((v22_i1 + 12) * 64);
              int32_t v35_a = 20 + (v22_i1 * 64);
              #pragma unroll
              for (int32_t v23_i2 = 0; v23_i2 < 6; ++v23_i2) {
                tensorforge::intel_esimd::simd<float, 12> v32_data;
                v32_data.copy_from(glb_m2 + ((v30_a + (v23_i2 * 832))));
                v32_data.copy_to(r2 + ((v35_a + (v23_i2 * 64))));
              }
            }
            #pragma unroll
            for (int32_t v37_i1 = 0; v37_i1 < 1; ++v37_i1) {
              int32_t v45_a = 32_i32 + ((v37_i1 + 12) * 64);
              int32_t v50_a = 32 + (v37_i1 * 64);
              #pragma unroll
              for (int32_t v38_i2 = 0; v38_i2 < 6; ++v38_i2) {
                tensorforge::intel_esimd::simd<float, 3> v47_data;
                v47_data.copy_from(glb_m2 + ((v45_a + (v38_i2 * 832))));
                v47_data.copy_to(r2 + ((v50_a + (v38_i2 * 64))));
              }
            }
            // wait(r0 = load{g>r}(glb_m0););
            float r1[4992]{};
            // r1 = +(r0 * glb_m1) + None
            // [(0, 64), (0, 13), (0, 6)] []
            tensorforge::intel_esimd::simd<float, 32> v53_data;
            v53_data.copy_from(r0 + (0));
            float v54_data = glb_m1[0];
            tensorforge::intel_esimd::simd<float, 32> v56_data;
            v56_data.copy_from(r1 + (0));
            (v56_data + (v53_data * v54_data)).copy_to(r1 + (0));
            float v59_data = glb_m1[1];
            tensorforge::intel_esimd::simd<float, 32> v61_data;
            v61_data.copy_from(r1 + (832));
            (v61_data + (v53_data * v59_data)).copy_to(r1 + (832));
            float v64_data = glb_m1[2];
            tensorforge::intel_esimd::simd<float, 32> v66_data;
            v66_data.copy_from(r1 + (1664));
            (v66_data + (v53_data * v64_data)).copy_to(r1 + (1664));
            float v69_data = glb_m1[3];
            tensorforge::intel_esimd::simd<float, 32> v71_data;
            v71_data.copy_from(r1 + (2496));
            (v71_data + (v53_data * v69_data)).copy_to(r1 + (2496));
            float v74_data = glb_m1[4];
            tensorforge::intel_esimd::simd<float, 32> v76_data;
            v76_data.copy_from(r1 + (3328));
            (v76_data + (v53_data * v74_data)).copy_to(r1 + (3328));
            float v79_data = glb_m1[5];
            tensorforge::intel_esimd::simd<float, 32> v81_data;
            v81_data.copy_from(r1 + (4160));
            (v81_data + (v53_data * v79_data)).copy_to(r1 + (4160));
            tensorforge::intel_esimd::simd<float, 32> v83_data;
            v83_data.copy_from(r0 + (64));
            tensorforge::intel_esimd::simd<float, 32> v86_data;
            v86_data.copy_from(r1 + (64));
            (v86_data + (v83_data * v54_data)).copy_to(r1 + (64));
            tensorforge::intel_esimd::simd<float, 32> v91_data;
            v91_data.copy_from(r1 + (896));
            (v91_data + (v83_data * v59_data)).copy_to(r1 + (896));
            tensorforge::intel_esimd::simd<float, 32> v96_data;
            v96_data.copy_from(r1 + (1728));
            (v96_data + (v83_data * v64_data)).copy_to(r1 + (1728));
            tensorforge::intel_esimd::simd<float, 32> v101_data;
            v101_data.copy_from(r1 + (2560));
            (v101_data + (v83_data * v69_data)).copy_to(r1 + (2560));
            tensorforge::intel_esimd::simd<float, 32> v106_data;
            v106_data.copy_from(r1 + (3392));
            (v106_data + (v83_data * v74_data)).copy_to(r1 + (3392));
            tensorforge::intel_esimd::simd<float, 32> v111_data;
            v111_data.copy_from(r1 + (4224));
            (v111_data + (v83_data * v79_data)).copy_to(r1 + (4224));
            tensorforge::intel_esimd::simd<float, 32> v113_data;
            v113_data.copy_from(r0 + (128));
            tensorforge::intel_esimd::simd<float, 32> v116_data;
            v116_data.copy_from(r1 + (128));
            (v116_data + (v113_data * v54_data)).copy_to(r1 + (128));
            tensorforge::intel_esimd::simd<float, 32> v121_data;
            v121_data.copy_from(r1 + (960));
            (v121_data + (v113_data * v59_data)).copy_to(r1 + (960));
            tensorforge::intel_esimd::simd<float, 32> v126_data;
            v126_data.copy_from(r1 + (1792));
            (v126_data + (v113_data * v64_data)).copy_to(r1 + (1792));
            tensorforge::intel_esimd::simd<float, 32> v131_data;
            v131_data.copy_from(r1 + (2624));
            (v131_data + (v113_data * v69_data)).copy_to(r1 + (2624));
            tensorforge::intel_esimd::simd<float, 32> v136_data;
            v136_data.copy_from(r1 + (3456));
            (v136_data + (v113_data * v74_data)).copy_to(r1 + (3456));
            tensorforge::intel_esimd::simd<float, 32> v141_data;
            v141_data.copy_from(r1 + (4288));
            (v141_data + (v113_data * v79_data)).copy_to(r1 + (4288));
            tensorforge::intel_esimd::simd<float, 32> v143_data;
            v143_data.copy_from(r0 + (192));
            tensorforge::intel_esimd::simd<float, 32> v146_data;
            v146_data.copy_from(r1 + (192));
            (v146_data + (v143_data * v54_data)).copy_to(r1 + (192));
            tensorforge::intel_esimd::simd<float, 32> v151_data;
            v151_data.copy_from(r1 + (1024));
            (v151_data + (v143_data * v59_data)).copy_to(r1 + (1024));
            tensorforge::intel_esimd::simd<float, 32> v156_data;
            v156_data.copy_from(r1 + (1856));
            (v156_data + (v143_data * v64_data)).copy_to(r1 + (1856));
            tensorforge::intel_esimd::simd<float, 32> v161_data;
            v161_data.copy_from(r1 + (2688));
            (v161_data + (v143_data * v69_data)).copy_to(r1 + (2688));
            tensorforge::intel_esimd::simd<float, 32> v166_data;
            v166_data.copy_from(r1 + (3520));
            (v166_data + (v143_data * v74_data)).copy_to(r1 + (3520));
            tensorforge::intel_esimd::simd<float, 32> v171_data;
            v171_data.copy_from(r1 + (4352));
            (v171_data + (v143_data * v79_data)).copy_to(r1 + (4352));
            tensorforge::intel_esimd::simd<float, 32> v173_data;
            v173_data.copy_from(r0 + (256));
            tensorforge::intel_esimd::simd<float, 32> v176_data;
            v176_data.copy_from(r1 + (256));
            (v176_data + (v173_data * v54_data)).copy_to(r1 + (256));
            tensorforge::intel_esimd::simd<float, 32> v181_data;
            v181_data.copy_from(r1 + (1088));
            (v181_data + (v173_data * v59_data)).copy_to(r1 + (1088));
            tensorforge::intel_esimd::simd<float, 32> v186_data;
            v186_data.copy_from(r1 + (1920));
            (v186_data + (v173_data * v64_data)).copy_to(r1 + (1920));
            tensorforge::intel_esimd::simd<float, 32> v191_data;
            v191_data.copy_from(r1 + (2752));
            (v191_data + (v173_data * v69_data)).copy_to(r1 + (2752));
            tensorforge::intel_esimd::simd<float, 32> v196_data;
            v196_data.copy_from(r1 + (3584));
            (v196_data + (v173_data * v74_data)).copy_to(r1 + (3584));
            tensorforge::intel_esimd::simd<float, 32> v201_data;
            v201_data.copy_from(r1 + (4416));
            (v201_data + (v173_data * v79_data)).copy_to(r1 + (4416));
            tensorforge::intel_esimd::simd<float, 32> v203_data;
            v203_data.copy_from(r0 + (320));
            tensorforge::intel_esimd::simd<float, 32> v206_data;
            v206_data.copy_from(r1 + (320));
            (v206_data + (v203_data * v54_data)).copy_to(r1 + (320));
            tensorforge::intel_esimd::simd<float, 32> v211_data;
            v211_data.copy_from(r1 + (1152));
            (v211_data + (v203_data * v59_data)).copy_to(r1 + (1152));
            tensorforge::intel_esimd::simd<float, 32> v216_data;
            v216_data.copy_from(r1 + (1984));
            (v216_data + (v203_data * v64_data)).copy_to(r1 + (1984));
            tensorforge::intel_esimd::simd<float, 32> v221_data;
            v221_data.copy_from(r1 + (2816));
            (v221_data + (v203_data * v69_data)).copy_to(r1 + (2816));
            tensorforge::intel_esimd::simd<float, 32> v226_data;
            v226_data.copy_from(r1 + (3648));
            (v226_data + (v203_data * v74_data)).copy_to(r1 + (3648));
            tensorforge::intel_esimd::simd<float, 32> v231_data;
            v231_data.copy_from(r1 + (4480));
            (v231_data + (v203_data * v79_data)).copy_to(r1 + (4480));
            tensorforge::intel_esimd::simd<float, 32> v233_data;
            v233_data.copy_from(r0 + (384));
            tensorforge::intel_esimd::simd<float, 32> v236_data;
            v236_data.copy_from(r1 + (384));
            (v236_data + (v233_data * v54_data)).copy_to(r1 + (384));
            tensorforge::intel_esimd::simd<float, 32> v241_data;
            v241_data.copy_from(r1 + (1216));
            (v241_data + (v233_data * v59_data)).copy_to(r1 + (1216));
            tensorforge::intel_esimd::simd<float, 32> v246_data;
            v246_data.copy_from(r1 + (2048));
            (v246_data + (v233_data * v64_data)).copy_to(r1 + (2048));
            tensorforge::intel_esimd::simd<float, 32> v251_data;
            v251_data.copy_from(r1 + (2880));
            (v251_data + (v233_data * v69_data)).copy_to(r1 + (2880));
            tensorforge::intel_esimd::simd<float, 32> v256_data;
            v256_data.copy_from(r1 + (3712));
            (v256_data + (v233_data * v74_data)).copy_to(r1 + (3712));
            tensorforge::intel_esimd::simd<float, 32> v261_data;
            v261_data.copy_from(r1 + (4544));
            (v261_data + (v233_data * v79_data)).copy_to(r1 + (4544));
            tensorforge::intel_esimd::simd<float, 32> v263_data;
            v263_data.copy_from(r0 + (448));
            tensorforge::intel_esimd::simd<float, 32> v266_data;
            v266_data.copy_from(r1 + (448));
            (v266_data + (v263_data * v54_data)).copy_to(r1 + (448));
            tensorforge::intel_esimd::simd<float, 32> v271_data;
            v271_data.copy_from(r1 + (1280));
            (v271_data + (v263_data * v59_data)).copy_to(r1 + (1280));
            tensorforge::intel_esimd::simd<float, 32> v276_data;
            v276_data.copy_from(r1 + (2112));
            (v276_data + (v263_data * v64_data)).copy_to(r1 + (2112));
            tensorforge::intel_esimd::simd<float, 32> v281_data;
            v281_data.copy_from(r1 + (2944));
            (v281_data + (v263_data * v69_data)).copy_to(r1 + (2944));
            tensorforge::intel_esimd::simd<float, 32> v286_data;
            v286_data.copy_from(r1 + (3776));
            (v286_data + (v263_data * v74_data)).copy_to(r1 + (3776));
            tensorforge::intel_esimd::simd<float, 32> v291_data;
            v291_data.copy_from(r1 + (4608));
            (v291_data + (v263_data * v79_data)).copy_to(r1 + (4608));
            tensorforge::intel_esimd::simd<float, 32> v293_data;
            v293_data.copy_from(r0 + (512));
            tensorforge::intel_esimd::simd<float, 32> v296_data;
            v296_data.copy_from(r1 + (512));
            (v296_data + (v293_data * v54_data)).copy_to(r1 + (512));
            tensorforge::intel_esimd::simd<float, 32> v301_data;
            v301_data.copy_from(r1 + (1344));
            (v301_data + (v293_data * v59_data)).copy_to(r1 + (1344));
            tensorforge::intel_esimd::simd<float, 32> v306_data;
            v306_data.copy_from(r1 + (2176));
            (v306_data + (v293_data * v64_data)).copy_to(r1 + (2176));
            tensorforge::intel_esimd::simd<float, 32> v311_data;
            v311_data.copy_from(r1 + (3008));
            (v311_data + (v293_data * v69_data)).copy_to(r1 + (3008));
            tensorforge::intel_esimd::simd<float, 32> v316_data;
            v316_data.copy_from(r1 + (3840));
            (v316_data + (v293_data * v74_data)).copy_to(r1 + (3840));
            tensorforge::intel_esimd::simd<float, 32> v321_data;
            v321_data.copy_from(r1 + (4672));
            (v321_data + (v293_data * v79_data)).copy_to(r1 + (4672));
            tensorforge::intel_esimd::simd<float, 32> v323_data;
            v323_data.copy_from(r0 + (576));
            tensorforge::intel_esimd::simd<float, 32> v326_data;
            v326_data.copy_from(r1 + (576));
            (v326_data + (v323_data * v54_data)).copy_to(r1 + (576));
            tensorforge::intel_esimd::simd<float, 32> v331_data;
            v331_data.copy_from(r1 + (1408));
            (v331_data + (v323_data * v59_data)).copy_to(r1 + (1408));
            tensorforge::intel_esimd::simd<float, 32> v336_data;
            v336_data.copy_from(r1 + (2240));
            (v336_data + (v323_data * v64_data)).copy_to(r1 + (2240));
            tensorforge::intel_esimd::simd<float, 32> v341_data;
            v341_data.copy_from(r1 + (3072));
            (v341_data + (v323_data * v69_data)).copy_to(r1 + (3072));
            tensorforge::intel_esimd::simd<float, 32> v346_data;
            v346_data.copy_from(r1 + (3904));
            (v346_data + (v323_data * v74_data)).copy_to(r1 + (3904));
            tensorforge::intel_esimd::simd<float, 32> v351_data;
            v351_data.copy_from(r1 + (4736));
            (v351_data + (v323_data * v79_data)).copy_to(r1 + (4736));
            tensorforge::intel_esimd::simd<float, 32> v353_data;
            v353_data.copy_from(r0 + (640));
            tensorforge::intel_esimd::simd<float, 32> v356_data;
            v356_data.copy_from(r1 + (640));
            (v356_data + (v353_data * v54_data)).copy_to(r1 + (640));
            tensorforge::intel_esimd::simd<float, 32> v361_data;
            v361_data.copy_from(r1 + (1472));
            (v361_data + (v353_data * v59_data)).copy_to(r1 + (1472));
            tensorforge::intel_esimd::simd<float, 32> v366_data;
            v366_data.copy_from(r1 + (2304));
            (v366_data + (v353_data * v64_data)).copy_to(r1 + (2304));
            tensorforge::intel_esimd::simd<float, 32> v371_data;
            v371_data.copy_from(r1 + (3136));
            (v371_data + (v353_data * v69_data)).copy_to(r1 + (3136));
            tensorforge::intel_esimd::simd<float, 32> v376_data;
            v376_data.copy_from(r1 + (3968));
            (v376_data + (v353_data * v74_data)).copy_to(r1 + (3968));
            tensorforge::intel_esimd::simd<float, 32> v381_data;
            v381_data.copy_from(r1 + (4800));
            (v381_data + (v353_data * v79_data)).copy_to(r1 + (4800));
            tensorforge::intel_esimd::simd<float, 32> v383_data;
            v383_data.copy_from(r0 + (704));
            tensorforge::intel_esimd::simd<float, 32> v386_data;
            v386_data.copy_from(r1 + (704));
            (v386_data + (v383_data * v54_data)).copy_to(r1 + (704));
            tensorforge::intel_esimd::simd<float, 32> v391_data;
            v391_data.copy_from(r1 + (1536));
            (v391_data + (v383_data * v59_data)).copy_to(r1 + (1536));
            tensorforge::intel_esimd::simd<float, 32> v396_data;
            v396_data.copy_from(r1 + (2368));
            (v396_data + (v383_data * v64_data)).copy_to(r1 + (2368));
            tensorforge::intel_esimd::simd<float, 32> v401_data;
            v401_data.copy_from(r1 + (3200));
            (v401_data + (v383_data * v69_data)).copy_to(r1 + (3200));
            tensorforge::intel_esimd::simd<float, 32> v406_data;
            v406_data.copy_from(r1 + (4032));
            (v406_data + (v383_data * v74_data)).copy_to(r1 + (4032));
            tensorforge::intel_esimd::simd<float, 32> v411_data;
            v411_data.copy_from(r1 + (4864));
            (v411_data + (v383_data * v79_data)).copy_to(r1 + (4864));
            tensorforge::intel_esimd::simd<float, 32> v413_data;
            v413_data.copy_from(r0 + (768));
            tensorforge::intel_esimd::simd<float, 32> v416_data;
            v416_data.copy_from(r1 + (768));
            (v416_data + (v413_data * v54_data)).copy_to(r1 + (768));
            tensorforge::intel_esimd::simd<float, 32> v421_data;
            v421_data.copy_from(r1 + (1600));
            (v421_data + (v413_data * v59_data)).copy_to(r1 + (1600));
            tensorforge::intel_esimd::simd<float, 32> v426_data;
            v426_data.copy_from(r1 + (2432));
            (v426_data + (v413_data * v64_data)).copy_to(r1 + (2432));
            tensorforge::intel_esimd::simd<float, 32> v431_data;
            v431_data.copy_from(r1 + (3264));
            (v431_data + (v413_data * v69_data)).copy_to(r1 + (3264));
            tensorforge::intel_esimd::simd<float, 32> v436_data;
            v436_data.copy_from(r1 + (4096));
            (v436_data + (v413_data * v74_data)).copy_to(r1 + (4096));
            tensorforge::intel_esimd::simd<float, 32> v441_data;
            v441_data.copy_from(r1 + (4928));
            (v441_data + (v413_data * v79_data)).copy_to(r1 + (4928));
            tensorforge::intel_esimd::simd<float, 32> v443_data;
            v443_data.copy_from(r0 + (32));
            tensorforge::intel_esimd::simd<float, 32> v446_data;
            v446_data.copy_from(r1 + (32));
            (v446_data + (v443_data * v54_data)).copy_to(r1 + (32));
            tensorforge::intel_esimd::simd<float, 32> v451_data;
            v451_data.copy_from(r1 + (864));
            (v451_data + (v443_data * v59_data)).copy_to(r1 + (864));
            tensorforge::intel_esimd::simd<float, 32> v456_data;
            v456_data.copy_from(r1 + (1696));
            (v456_data + (v443_data * v64_data)).copy_to(r1 + (1696));
            tensorforge::intel_esimd::simd<float, 32> v461_data;
            v461_data.copy_from(r1 + (2528));
            (v461_data + (v443_data * v69_data)).copy_to(r1 + (2528));
            tensorforge::intel_esimd::simd<float, 32> v466_data;
            v466_data.copy_from(r1 + (3360));
            (v466_data + (v443_data * v74_data)).copy_to(r1 + (3360));
            tensorforge::intel_esimd::simd<float, 32> v471_data;
            v471_data.copy_from(r1 + (4192));
            (v471_data + (v443_data * v79_data)).copy_to(r1 + (4192));
            tensorforge::intel_esimd::simd<float, 32> v473_data;
            v473_data.copy_from(r0 + (96));
            tensorforge::intel_esimd::simd<float, 32> v476_data;
            v476_data.copy_from(r1 + (96));
            (v476_data + (v473_data * v54_data)).copy_to(r1 + (96));
            tensorforge::intel_esimd::simd<float, 32> v481_data;
            v481_data.copy_from(r1 + (928));
            (v481_data + (v473_data * v59_data)).copy_to(r1 + (928));
            tensorforge::intel_esimd::simd<float, 32> v486_data;
            v486_data.copy_from(r1 + (1760));
            (v486_data + (v473_data * v64_data)).copy_to(r1 + (1760));
            tensorforge::intel_esimd::simd<float, 32> v491_data;
            v491_data.copy_from(r1 + (2592));
            (v491_data + (v473_data * v69_data)).copy_to(r1 + (2592));
            tensorforge::intel_esimd::simd<float, 32> v496_data;
            v496_data.copy_from(r1 + (3424));
            (v496_data + (v473_data * v74_data)).copy_to(r1 + (3424));
            tensorforge::intel_esimd::simd<float, 32> v501_data;
            v501_data.copy_from(r1 + (4256));
            (v501_data + (v473_data * v79_data)).copy_to(r1 + (4256));
            tensorforge::intel_esimd::simd<float, 32> v503_data;
            v503_data.copy_from(r0 + (160));
            tensorforge::intel_esimd::simd<float, 32> v506_data;
            v506_data.copy_from(r1 + (160));
            (v506_data + (v503_data * v54_data)).copy_to(r1 + (160));
            tensorforge::intel_esimd::simd<float, 32> v511_data;
            v511_data.copy_from(r1 + (992));
            (v511_data + (v503_data * v59_data)).copy_to(r1 + (992));
            tensorforge::intel_esimd::simd<float, 32> v516_data;
            v516_data.copy_from(r1 + (1824));
            (v516_data + (v503_data * v64_data)).copy_to(r1 + (1824));
            tensorforge::intel_esimd::simd<float, 32> v521_data;
            v521_data.copy_from(r1 + (2656));
            (v521_data + (v503_data * v69_data)).copy_to(r1 + (2656));
            tensorforge::intel_esimd::simd<float, 32> v526_data;
            v526_data.copy_from(r1 + (3488));
            (v526_data + (v503_data * v74_data)).copy_to(r1 + (3488));
            tensorforge::intel_esimd::simd<float, 32> v531_data;
            v531_data.copy_from(r1 + (4320));
            (v531_data + (v503_data * v79_data)).copy_to(r1 + (4320));
            tensorforge::intel_esimd::simd<float, 32> v533_data;
            v533_data.copy_from(r0 + (224));
            tensorforge::intel_esimd::simd<float, 32> v536_data;
            v536_data.copy_from(r1 + (224));
            (v536_data + (v533_data * v54_data)).copy_to(r1 + (224));
            tensorforge::intel_esimd::simd<float, 32> v541_data;
            v541_data.copy_from(r1 + (1056));
            (v541_data + (v533_data * v59_data)).copy_to(r1 + (1056));
            tensorforge::intel_esimd::simd<float, 32> v546_data;
            v546_data.copy_from(r1 + (1888));
            (v546_data + (v533_data * v64_data)).copy_to(r1 + (1888));
            tensorforge::intel_esimd::simd<float, 32> v551_data;
            v551_data.copy_from(r1 + (2720));
            (v551_data + (v533_data * v69_data)).copy_to(r1 + (2720));
            tensorforge::intel_esimd::simd<float, 32> v556_data;
            v556_data.copy_from(r1 + (3552));
            (v556_data + (v533_data * v74_data)).copy_to(r1 + (3552));
            tensorforge::intel_esimd::simd<float, 32> v561_data;
            v561_data.copy_from(r1 + (4384));
            (v561_data + (v533_data * v79_data)).copy_to(r1 + (4384));
            tensorforge::intel_esimd::simd<float, 32> v563_data;
            v563_data.copy_from(r0 + (288));
            tensorforge::intel_esimd::simd<float, 32> v566_data;
            v566_data.copy_from(r1 + (288));
            (v566_data + (v563_data * v54_data)).copy_to(r1 + (288));
            tensorforge::intel_esimd::simd<float, 32> v571_data;
            v571_data.copy_from(r1 + (1120));
            (v571_data + (v563_data * v59_data)).copy_to(r1 + (1120));
            tensorforge::intel_esimd::simd<float, 32> v576_data;
            v576_data.copy_from(r1 + (1952));
            (v576_data + (v563_data * v64_data)).copy_to(r1 + (1952));
            tensorforge::intel_esimd::simd<float, 32> v581_data;
            v581_data.copy_from(r1 + (2784));
            (v581_data + (v563_data * v69_data)).copy_to(r1 + (2784));
            tensorforge::intel_esimd::simd<float, 32> v586_data;
            v586_data.copy_from(r1 + (3616));
            (v586_data + (v563_data * v74_data)).copy_to(r1 + (3616));
            tensorforge::intel_esimd::simd<float, 32> v591_data;
            v591_data.copy_from(r1 + (4448));
            (v591_data + (v563_data * v79_data)).copy_to(r1 + (4448));
            tensorforge::intel_esimd::simd<float, 32> v593_data;
            v593_data.copy_from(r0 + (352));
            tensorforge::intel_esimd::simd<float, 32> v596_data;
            v596_data.copy_from(r1 + (352));
            (v596_data + (v593_data * v54_data)).copy_to(r1 + (352));
            tensorforge::intel_esimd::simd<float, 32> v601_data;
            v601_data.copy_from(r1 + (1184));
            (v601_data + (v593_data * v59_data)).copy_to(r1 + (1184));
            tensorforge::intel_esimd::simd<float, 32> v606_data;
            v606_data.copy_from(r1 + (2016));
            (v606_data + (v593_data * v64_data)).copy_to(r1 + (2016));
            tensorforge::intel_esimd::simd<float, 32> v611_data;
            v611_data.copy_from(r1 + (2848));
            (v611_data + (v593_data * v69_data)).copy_to(r1 + (2848));
            tensorforge::intel_esimd::simd<float, 32> v616_data;
            v616_data.copy_from(r1 + (3680));
            (v616_data + (v593_data * v74_data)).copy_to(r1 + (3680));
            tensorforge::intel_esimd::simd<float, 32> v621_data;
            v621_data.copy_from(r1 + (4512));
            (v621_data + (v593_data * v79_data)).copy_to(r1 + (4512));
            tensorforge::intel_esimd::simd<float, 32> v623_data;
            v623_data.copy_from(r0 + (416));
            tensorforge::intel_esimd::simd<float, 32> v626_data;
            v626_data.copy_from(r1 + (416));
            (v626_data + (v623_data * v54_data)).copy_to(r1 + (416));
            tensorforge::intel_esimd::simd<float, 32> v631_data;
            v631_data.copy_from(r1 + (1248));
            (v631_data + (v623_data * v59_data)).copy_to(r1 + (1248));
            tensorforge::intel_esimd::simd<float, 32> v636_data;
            v636_data.copy_from(r1 + (2080));
            (v636_data + (v623_data * v64_data)).copy_to(r1 + (2080));
            tensorforge::intel_esimd::simd<float, 32> v641_data;
            v641_data.copy_from(r1 + (2912));
            (v641_data + (v623_data * v69_data)).copy_to(r1 + (2912));
            tensorforge::intel_esimd::simd<float, 32> v646_data;
            v646_data.copy_from(r1 + (3744));
            (v646_data + (v623_data * v74_data)).copy_to(r1 + (3744));
            tensorforge::intel_esimd::simd<float, 32> v651_data;
            v651_data.copy_from(r1 + (4576));
            (v651_data + (v623_data * v79_data)).copy_to(r1 + (4576));
            tensorforge::intel_esimd::simd<float, 32> v653_data;
            v653_data.copy_from(r0 + (480));
            tensorforge::intel_esimd::simd<float, 32> v656_data;
            v656_data.copy_from(r1 + (480));
            (v656_data + (v653_data * v54_data)).copy_to(r1 + (480));
            tensorforge::intel_esimd::simd<float, 32> v661_data;
            v661_data.copy_from(r1 + (1312));
            (v661_data + (v653_data * v59_data)).copy_to(r1 + (1312));
            tensorforge::intel_esimd::simd<float, 32> v666_data;
            v666_data.copy_from(r1 + (2144));
            (v666_data + (v653_data * v64_data)).copy_to(r1 + (2144));
            tensorforge::intel_esimd::simd<float, 32> v671_data;
            v671_data.copy_from(r1 + (2976));
            (v671_data + (v653_data * v69_data)).copy_to(r1 + (2976));
            tensorforge::intel_esimd::simd<float, 32> v676_data;
            v676_data.copy_from(r1 + (3808));
            (v676_data + (v653_data * v74_data)).copy_to(r1 + (3808));
            tensorforge::intel_esimd::simd<float, 32> v681_data;
            v681_data.copy_from(r1 + (4640));
            (v681_data + (v653_data * v79_data)).copy_to(r1 + (4640));
            tensorforge::intel_esimd::simd<float, 32> v683_data;
            v683_data.copy_from(r0 + (544));
            tensorforge::intel_esimd::simd<float, 32> v686_data;
            v686_data.copy_from(r1 + (544));
            (v686_data + (v683_data * v54_data)).copy_to(r1 + (544));
            tensorforge::intel_esimd::simd<float, 32> v691_data;
            v691_data.copy_from(r1 + (1376));
            (v691_data + (v683_data * v59_data)).copy_to(r1 + (1376));
            tensorforge::intel_esimd::simd<float, 32> v696_data;
            v696_data.copy_from(r1 + (2208));
            (v696_data + (v683_data * v64_data)).copy_to(r1 + (2208));
            tensorforge::intel_esimd::simd<float, 32> v701_data;
            v701_data.copy_from(r1 + (3040));
            (v701_data + (v683_data * v69_data)).copy_to(r1 + (3040));
            tensorforge::intel_esimd::simd<float, 32> v706_data;
            v706_data.copy_from(r1 + (3872));
            (v706_data + (v683_data * v74_data)).copy_to(r1 + (3872));
            tensorforge::intel_esimd::simd<float, 32> v711_data;
            v711_data.copy_from(r1 + (4704));
            (v711_data + (v683_data * v79_data)).copy_to(r1 + (4704));
            tensorforge::intel_esimd::simd<float, 32> v713_data;
            v713_data.copy_from(r0 + (608));
            tensorforge::intel_esimd::simd<float, 32> v716_data;
            v716_data.copy_from(r1 + (608));
            (v716_data + (v713_data * v54_data)).copy_to(r1 + (608));
            tensorforge::intel_esimd::simd<float, 32> v721_data;
            v721_data.copy_from(r1 + (1440));
            (v721_data + (v713_data * v59_data)).copy_to(r1 + (1440));
            tensorforge::intel_esimd::simd<float, 32> v726_data;
            v726_data.copy_from(r1 + (2272));
            (v726_data + (v713_data * v64_data)).copy_to(r1 + (2272));
            tensorforge::intel_esimd::simd<float, 32> v731_data;
            v731_data.copy_from(r1 + (3104));
            (v731_data + (v713_data * v69_data)).copy_to(r1 + (3104));
            tensorforge::intel_esimd::simd<float, 32> v736_data;
            v736_data.copy_from(r1 + (3936));
            (v736_data + (v713_data * v74_data)).copy_to(r1 + (3936));
            tensorforge::intel_esimd::simd<float, 32> v741_data;
            v741_data.copy_from(r1 + (4768));
            (v741_data + (v713_data * v79_data)).copy_to(r1 + (4768));
            tensorforge::intel_esimd::simd<float, 32> v743_data;
            v743_data.copy_from(r0 + (672));
            tensorforge::intel_esimd::simd<float, 32> v746_data;
            v746_data.copy_from(r1 + (672));
            (v746_data + (v743_data * v54_data)).copy_to(r1 + (672));
            tensorforge::intel_esimd::simd<float, 32> v751_data;
            v751_data.copy_from(r1 + (1504));
            (v751_data + (v743_data * v59_data)).copy_to(r1 + (1504));
            tensorforge::intel_esimd::simd<float, 32> v756_data;
            v756_data.copy_from(r1 + (2336));
            (v756_data + (v743_data * v64_data)).copy_to(r1 + (2336));
            tensorforge::intel_esimd::simd<float, 32> v761_data;
            v761_data.copy_from(r1 + (3168));
            (v761_data + (v743_data * v69_data)).copy_to(r1 + (3168));
            tensorforge::intel_esimd::simd<float, 32> v766_data;
            v766_data.copy_from(r1 + (4000));
            (v766_data + (v743_data * v74_data)).copy_to(r1 + (4000));
            tensorforge::intel_esimd::simd<float, 32> v771_data;
            v771_data.copy_from(r1 + (4832));
            (v771_data + (v743_data * v79_data)).copy_to(r1 + (4832));
            tensorforge::intel_esimd::simd<float, 32> v773_data;
            v773_data.copy_from(r0 + (736));
            tensorforge::intel_esimd::simd<float, 32> v776_data;
            v776_data.copy_from(r1 + (736));
            (v776_data + (v773_data * v54_data)).copy_to(r1 + (736));
            tensorforge::intel_esimd::simd<float, 32> v781_data;
            v781_data.copy_from(r1 + (1568));
            (v781_data + (v773_data * v59_data)).copy_to(r1 + (1568));
            tensorforge::intel_esimd::simd<float, 32> v786_data;
            v786_data.copy_from(r1 + (2400));
            (v786_data + (v773_data * v64_data)).copy_to(r1 + (2400));
            tensorforge::intel_esimd::simd<float, 32> v791_data;
            v791_data.copy_from(r1 + (3232));
            (v791_data + (v773_data * v69_data)).copy_to(r1 + (3232));
            tensorforge::intel_esimd::simd<float, 32> v796_data;
            v796_data.copy_from(r1 + (4064));
            (v796_data + (v773_data * v74_data)).copy_to(r1 + (4064));
            tensorforge::intel_esimd::simd<float, 32> v801_data;
            v801_data.copy_from(r1 + (4896));
            (v801_data + (v773_data * v79_data)).copy_to(r1 + (4896));
            tensorforge::intel_esimd::simd<float, 32> v803_data;
            v803_data.copy_from(r0 + (800));
            tensorforge::intel_esimd::simd<float, 32> v806_data;
            v806_data.copy_from(r1 + (800));
            (v806_data + (v803_data * v54_data)).copy_to(r1 + (800));
            tensorforge::intel_esimd::simd<float, 32> v811_data;
            v811_data.copy_from(r1 + (1632));
            (v811_data + (v803_data * v59_data)).copy_to(r1 + (1632));
            tensorforge::intel_esimd::simd<float, 32> v816_data;
            v816_data.copy_from(r1 + (2464));
            (v816_data + (v803_data * v64_data)).copy_to(r1 + (2464));
            tensorforge::intel_esimd::simd<float, 32> v821_data;
            v821_data.copy_from(r1 + (3296));
            (v821_data + (v803_data * v69_data)).copy_to(r1 + (3296));
            tensorforge::intel_esimd::simd<float, 32> v826_data;
            v826_data.copy_from(r1 + (4128));
            (v826_data + (v803_data * v74_data)).copy_to(r1 + (4128));
            tensorforge::intel_esimd::simd<float, 32> v831_data;
            v831_data.copy_from(r1 + (4960));
            (v831_data + (v803_data * v79_data)).copy_to(r1 + (4960));
            // wait(r2 = load{g>r}(glb_m2););
            float r3[384]{};
            // ir3 = +(r1)
            // [(20, 35), (0, 1), (0, 6)] []
            float ir3[384]{};
            tensorforge::intel_esimd::simd<float, 12> v835_data;
            v835_data.copy_from(r1 + (788));
            tensorforge::intel_esimd::simd<float, 12> v836_data;
            v836_data.copy_from(ir3 + (20));
            (v836_data + v835_data).copy_to(ir3 + (20));
            tensorforge::intel_esimd::simd<float, 12> v838_data;
            v838_data.copy_from(r1 + (1620));
            tensorforge::intel_esimd::simd<float, 12> v839_data;
            v839_data.copy_from(ir3 + (84));
            (v839_data + v838_data).copy_to(ir3 + (84));
            tensorforge::intel_esimd::simd<float, 12> v841_data;
            v841_data.copy_from(r1 + (2452));
            tensorforge::intel_esimd::simd<float, 12> v842_data;
            v842_data.copy_from(ir3 + (148));
            (v842_data + v841_data).copy_to(ir3 + (148));
            tensorforge::intel_esimd::simd<float, 12> v844_data;
            v844_data.copy_from(r1 + (3284));
            tensorforge::intel_esimd::simd<float, 12> v845_data;
            v845_data.copy_from(ir3 + (212));
            (v845_data + v844_data).copy_to(ir3 + (212));
            tensorforge::intel_esimd::simd<float, 12> v847_data;
            v847_data.copy_from(r1 + (4116));
            tensorforge::intel_esimd::simd<float, 12> v848_data;
            v848_data.copy_from(ir3 + (276));
            (v848_data + v847_data).copy_to(ir3 + (276));
            tensorforge::intel_esimd::simd<float, 12> v850_data;
            v850_data.copy_from(r1 + (4948));
            tensorforge::intel_esimd::simd<float, 12> v851_data;
            v851_data.copy_from(ir3 + (340));
            (v851_data + v850_data).copy_to(ir3 + (340));
            tensorforge::intel_esimd::simd<float, 3> v853_data;
            v853_data.copy_from(r1 + (800));
            tensorforge::intel_esimd::simd<float, 3> v854_data;
            v854_data.copy_from(ir3 + (32));
            (v854_data + v853_data).copy_to(ir3 + (32));
            tensorforge::intel_esimd::simd<float, 3> v856_data;
            v856_data.copy_from(r1 + (1632));
            tensorforge::intel_esimd::simd<float, 3> v857_data;
            v857_data.copy_from(ir3 + (96));
            (v857_data + v856_data).copy_to(ir3 + (96));
            tensorforge::intel_esimd::simd<float, 3> v859_data;
            v859_data.copy_from(r1 + (2464));
            tensorforge::intel_esimd::simd<float, 3> v860_data;
            v860_data.copy_from(ir3 + (160));
            (v860_data + v859_data).copy_to(ir3 + (160));
            tensorforge::intel_esimd::simd<float, 3> v862_data;
            v862_data.copy_from(r1 + (3296));
            tensorforge::intel_esimd::simd<float, 3> v863_data;
            v863_data.copy_from(ir3 + (224));
            (v863_data + v862_data).copy_to(ir3 + (224));
            tensorforge::intel_esimd::simd<float, 3> v865_data;
            v865_data.copy_from(r1 + (4128));
            tensorforge::intel_esimd::simd<float, 3> v866_data;
            v866_data.copy_from(ir3 + (288));
            (v866_data + v865_data).copy_to(ir3 + (288));
            tensorforge::intel_esimd::simd<float, 3> v868_data;
            v868_data.copy_from(r1 + (4960));
            tensorforge::intel_esimd::simd<float, 3> v869_data;
            v869_data.copy_from(ir3 + (352));
            (v869_data + v868_data).copy_to(ir3 + (352));
            // r3 = ir3 + r2
            #pragma unroll
            for (int32_t v871_n1 = 0; v871_n1 < 1; ++v871_n1) {
              int32_t v875_a = 20 + (v871_n1 * 64);
              #pragma unroll
              for (int32_t v872_n2 = 0; v872_n2 < 6; ++v872_n2) {
                int32_t v876_a = v875_a + (v872_n2 * 64);
                tensorforge::intel_esimd::simd<float, 12> v877_data;
                v877_data.copy_from(ir3 + (v876_a));
                tensorforge::intel_esimd::simd<float, 12> v878_data;
                v878_data.copy_from(r2 + (v876_a));
                (v878_data + v877_data).copy_to(r3 + (v876_a));
              }
            }
            #pragma unroll
            for (int32_t v880_n1 = 0; v880_n1 < 1; ++v880_n1) {
              int32_t v884_a = 32 + (v880_n1 * 64);
              #pragma unroll
              for (int32_t v881_n2 = 0; v881_n2 < 6; ++v881_n2) {
                int32_t v885_a = v884_a + (v881_n2 * 64);
                tensorforge::intel_esimd::simd<float, 3> v886_data;
                v886_data.copy_from(ir3 + (v885_a));
                tensorforge::intel_esimd::simd<float, 3> v887_data;
                v887_data.copy_from(r2 + (v885_a));
                (v887_data + v886_data).copy_to(r3 + (v885_a));
              }
            }
            // glb_m2 = store{r>g}(r3);
            #pragma unroll
            for (int32_t v889_i1 = 0; v889_i1 < 1; ++v889_i1) {
              int32_t v893_a = 20 + (v889_i1 * 64);
              int32_t v902_a = 20_i32 + ((v889_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v890_i2 = 0; v890_i2 < 6; ++v890_i2) {
                tensorforge::intel_esimd::simd<float, 12> v895_data;
                v895_data.copy_from(r3 + ((v893_a + (v890_i2 * 64))));
                v895_data.copy_to(glb_m2 + ((v902_a + (v890_i2 * 832))));
              }
            }
            #pragma unroll
            for (int32_t v904_i1 = 0; v904_i1 < 1; ++v904_i1) {
              int32_t v908_a = 32 + (v904_i1 * 64);
              int32_t v917_a = 32_i32 + ((v904_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v905_i2 = 0; v905_i2 < 6; ++v905_i2) {
                tensorforge::intel_esimd::simd<float, 3> v910_data;
                v910_data.copy_from(r3 + ((v908_a + (v905_i2 * 64))));
                v910_data.copy_to(glb_m2 + ((v917_a + (v905_i2 * 832))));
              }
            }
          }
        }
      }
    });
  });
}

