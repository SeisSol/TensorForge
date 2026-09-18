// === base name ===
kernel_29cbad88bb22b183

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_29cbad88bb22b183 = {{1, 8, 1}, 32, 64, 1, 8, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_29cbad88bb22b183(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_29cbad88bb22b183(const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_29cbad88bb22b183(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_29cbad88bb22b183(const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_29cbad88bb22b183(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_29cbad88bb22b183(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_29cbad88bb22b183(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
            tensorforge::intel_esimd::simd<float, 832> r0(0.0f);
            // r0 = load{g>r}(glb_m0);
            #pragma unroll
            for (int32_t v13_i0 = 0; v13_i0 < 2; ++v13_i0) {
              int32_t v15_lead = v13_i0 * 32;
              #pragma unroll
              for (int32_t v14_i1 = 0; v14_i1 < 13; ++v14_i1) {
                int32_t v18_a = v15_lead + (v14_i1 * 64);
                tensorforge::intel_esimd::simd<float, 32> v19_data;
                v19_data.copy_from(glb_m0 + (v18_a));
                r0.template select<32, 1>(v18_a) = v19_data;
              }
            }
            tensorforge::intel_esimd::simd<float, 384> r2(0.0f);
            // r2 = load{g>r}(glb_m2);
            #pragma unroll
            for (int32_t v22_i1 = 0; v22_i1 < 1; ++v22_i1) {
              int32_t v30_a = 20_i32 + ((v22_i1 + 12) * 64);
              int32_t v35_a = 20 + (v22_i1 * 64);
              #pragma unroll
              for (int32_t v23_i2 = 0; v23_i2 < 6; ++v23_i2) {
                tensorforge::intel_esimd::simd<float, 12> v32_data;
                v32_data.copy_from(glb_m2 + ((v30_a + (v23_i2 * 832))));
                r2.template select<12, 1>((v35_a + (v23_i2 * 64))) = v32_data;
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
                r2.template select<3, 1>((v50_a + (v38_i2 * 64))) = v47_data;
              }
            }
            // wait(r0 = load{g>r}(glb_m0););
            tensorforge::intel_esimd::simd<float, 4992> r1(0.0f);
            // r1 = +(r0 * glb_m1) + None
            // [(0, 64), (0, 13), (0, 6)] []
            tensorforge::intel_esimd::simd<float, 32> v53_data(r0.template select<32, 1>(0));
            float v54_data = glb_m1[0];
            tensorforge::intel_esimd::simd<float, 32> v56_data(r1.template select<32, 1>(0));
            r1.template select<32, 1>(0) = (v56_data + (v53_data * v54_data));
            float v59_data = glb_m1[1];
            tensorforge::intel_esimd::simd<float, 32> v61_data(r1.template select<32, 1>(832));
            r1.template select<32, 1>(832) = (v61_data + (v53_data * v59_data));
            float v64_data = glb_m1[2];
            tensorforge::intel_esimd::simd<float, 32> v66_data(r1.template select<32, 1>(1664));
            r1.template select<32, 1>(1664) = (v66_data + (v53_data * v64_data));
            float v69_data = glb_m1[3];
            tensorforge::intel_esimd::simd<float, 32> v71_data(r1.template select<32, 1>(2496));
            r1.template select<32, 1>(2496) = (v71_data + (v53_data * v69_data));
            float v74_data = glb_m1[4];
            tensorforge::intel_esimd::simd<float, 32> v76_data(r1.template select<32, 1>(3328));
            r1.template select<32, 1>(3328) = (v76_data + (v53_data * v74_data));
            float v79_data = glb_m1[5];
            tensorforge::intel_esimd::simd<float, 32> v81_data(r1.template select<32, 1>(4160));
            r1.template select<32, 1>(4160) = (v81_data + (v53_data * v79_data));
            tensorforge::intel_esimd::simd<float, 32> v83_data(r0.template select<32, 1>(64));
            tensorforge::intel_esimd::simd<float, 32> v86_data(r1.template select<32, 1>(64));
            r1.template select<32, 1>(64) = (v86_data + (v83_data * v54_data));
            tensorforge::intel_esimd::simd<float, 32> v91_data(r1.template select<32, 1>(896));
            r1.template select<32, 1>(896) = (v91_data + (v83_data * v59_data));
            tensorforge::intel_esimd::simd<float, 32> v96_data(r1.template select<32, 1>(1728));
            r1.template select<32, 1>(1728) = (v96_data + (v83_data * v64_data));
            tensorforge::intel_esimd::simd<float, 32> v101_data(r1.template select<32, 1>(2560));
            r1.template select<32, 1>(2560) = (v101_data + (v83_data * v69_data));
            tensorforge::intel_esimd::simd<float, 32> v106_data(r1.template select<32, 1>(3392));
            r1.template select<32, 1>(3392) = (v106_data + (v83_data * v74_data));
            tensorforge::intel_esimd::simd<float, 32> v111_data(r1.template select<32, 1>(4224));
            r1.template select<32, 1>(4224) = (v111_data + (v83_data * v79_data));
            tensorforge::intel_esimd::simd<float, 32> v113_data(r0.template select<32, 1>(128));
            tensorforge::intel_esimd::simd<float, 32> v116_data(r1.template select<32, 1>(128));
            r1.template select<32, 1>(128) = (v116_data + (v113_data * v54_data));
            tensorforge::intel_esimd::simd<float, 32> v121_data(r1.template select<32, 1>(960));
            r1.template select<32, 1>(960) = (v121_data + (v113_data * v59_data));
            tensorforge::intel_esimd::simd<float, 32> v126_data(r1.template select<32, 1>(1792));
            r1.template select<32, 1>(1792) = (v126_data + (v113_data * v64_data));
            tensorforge::intel_esimd::simd<float, 32> v131_data(r1.template select<32, 1>(2624));
            r1.template select<32, 1>(2624) = (v131_data + (v113_data * v69_data));
            tensorforge::intel_esimd::simd<float, 32> v136_data(r1.template select<32, 1>(3456));
            r1.template select<32, 1>(3456) = (v136_data + (v113_data * v74_data));
            tensorforge::intel_esimd::simd<float, 32> v141_data(r1.template select<32, 1>(4288));
            r1.template select<32, 1>(4288) = (v141_data + (v113_data * v79_data));
            tensorforge::intel_esimd::simd<float, 32> v143_data(r0.template select<32, 1>(192));
            tensorforge::intel_esimd::simd<float, 32> v146_data(r1.template select<32, 1>(192));
            r1.template select<32, 1>(192) = (v146_data + (v143_data * v54_data));
            tensorforge::intel_esimd::simd<float, 32> v151_data(r1.template select<32, 1>(1024));
            r1.template select<32, 1>(1024) = (v151_data + (v143_data * v59_data));
            tensorforge::intel_esimd::simd<float, 32> v156_data(r1.template select<32, 1>(1856));
            r1.template select<32, 1>(1856) = (v156_data + (v143_data * v64_data));
            tensorforge::intel_esimd::simd<float, 32> v161_data(r1.template select<32, 1>(2688));
            r1.template select<32, 1>(2688) = (v161_data + (v143_data * v69_data));
            tensorforge::intel_esimd::simd<float, 32> v166_data(r1.template select<32, 1>(3520));
            r1.template select<32, 1>(3520) = (v166_data + (v143_data * v74_data));
            tensorforge::intel_esimd::simd<float, 32> v171_data(r1.template select<32, 1>(4352));
            r1.template select<32, 1>(4352) = (v171_data + (v143_data * v79_data));
            tensorforge::intel_esimd::simd<float, 32> v173_data(r0.template select<32, 1>(256));
            tensorforge::intel_esimd::simd<float, 32> v176_data(r1.template select<32, 1>(256));
            r1.template select<32, 1>(256) = (v176_data + (v173_data * v54_data));
            tensorforge::intel_esimd::simd<float, 32> v181_data(r1.template select<32, 1>(1088));
            r1.template select<32, 1>(1088) = (v181_data + (v173_data * v59_data));
            tensorforge::intel_esimd::simd<float, 32> v186_data(r1.template select<32, 1>(1920));
            r1.template select<32, 1>(1920) = (v186_data + (v173_data * v64_data));
            tensorforge::intel_esimd::simd<float, 32> v191_data(r1.template select<32, 1>(2752));
            r1.template select<32, 1>(2752) = (v191_data + (v173_data * v69_data));
            tensorforge::intel_esimd::simd<float, 32> v196_data(r1.template select<32, 1>(3584));
            r1.template select<32, 1>(3584) = (v196_data + (v173_data * v74_data));
            tensorforge::intel_esimd::simd<float, 32> v201_data(r1.template select<32, 1>(4416));
            r1.template select<32, 1>(4416) = (v201_data + (v173_data * v79_data));
            tensorforge::intel_esimd::simd<float, 32> v203_data(r0.template select<32, 1>(320));
            tensorforge::intel_esimd::simd<float, 32> v206_data(r1.template select<32, 1>(320));
            r1.template select<32, 1>(320) = (v206_data + (v203_data * v54_data));
            tensorforge::intel_esimd::simd<float, 32> v211_data(r1.template select<32, 1>(1152));
            r1.template select<32, 1>(1152) = (v211_data + (v203_data * v59_data));
            tensorforge::intel_esimd::simd<float, 32> v216_data(r1.template select<32, 1>(1984));
            r1.template select<32, 1>(1984) = (v216_data + (v203_data * v64_data));
            tensorforge::intel_esimd::simd<float, 32> v221_data(r1.template select<32, 1>(2816));
            r1.template select<32, 1>(2816) = (v221_data + (v203_data * v69_data));
            tensorforge::intel_esimd::simd<float, 32> v226_data(r1.template select<32, 1>(3648));
            r1.template select<32, 1>(3648) = (v226_data + (v203_data * v74_data));
            tensorforge::intel_esimd::simd<float, 32> v231_data(r1.template select<32, 1>(4480));
            r1.template select<32, 1>(4480) = (v231_data + (v203_data * v79_data));
            tensorforge::intel_esimd::simd<float, 32> v233_data(r0.template select<32, 1>(384));
            tensorforge::intel_esimd::simd<float, 32> v236_data(r1.template select<32, 1>(384));
            r1.template select<32, 1>(384) = (v236_data + (v233_data * v54_data));
            tensorforge::intel_esimd::simd<float, 32> v241_data(r1.template select<32, 1>(1216));
            r1.template select<32, 1>(1216) = (v241_data + (v233_data * v59_data));
            tensorforge::intel_esimd::simd<float, 32> v246_data(r1.template select<32, 1>(2048));
            r1.template select<32, 1>(2048) = (v246_data + (v233_data * v64_data));
            tensorforge::intel_esimd::simd<float, 32> v251_data(r1.template select<32, 1>(2880));
            r1.template select<32, 1>(2880) = (v251_data + (v233_data * v69_data));
            tensorforge::intel_esimd::simd<float, 32> v256_data(r1.template select<32, 1>(3712));
            r1.template select<32, 1>(3712) = (v256_data + (v233_data * v74_data));
            tensorforge::intel_esimd::simd<float, 32> v261_data(r1.template select<32, 1>(4544));
            r1.template select<32, 1>(4544) = (v261_data + (v233_data * v79_data));
            tensorforge::intel_esimd::simd<float, 32> v263_data(r0.template select<32, 1>(448));
            tensorforge::intel_esimd::simd<float, 32> v266_data(r1.template select<32, 1>(448));
            r1.template select<32, 1>(448) = (v266_data + (v263_data * v54_data));
            tensorforge::intel_esimd::simd<float, 32> v271_data(r1.template select<32, 1>(1280));
            r1.template select<32, 1>(1280) = (v271_data + (v263_data * v59_data));
            tensorforge::intel_esimd::simd<float, 32> v276_data(r1.template select<32, 1>(2112));
            r1.template select<32, 1>(2112) = (v276_data + (v263_data * v64_data));
            tensorforge::intel_esimd::simd<float, 32> v281_data(r1.template select<32, 1>(2944));
            r1.template select<32, 1>(2944) = (v281_data + (v263_data * v69_data));
            tensorforge::intel_esimd::simd<float, 32> v286_data(r1.template select<32, 1>(3776));
            r1.template select<32, 1>(3776) = (v286_data + (v263_data * v74_data));
            tensorforge::intel_esimd::simd<float, 32> v291_data(r1.template select<32, 1>(4608));
            r1.template select<32, 1>(4608) = (v291_data + (v263_data * v79_data));
            tensorforge::intel_esimd::simd<float, 32> v293_data(r0.template select<32, 1>(512));
            tensorforge::intel_esimd::simd<float, 32> v296_data(r1.template select<32, 1>(512));
            r1.template select<32, 1>(512) = (v296_data + (v293_data * v54_data));
            tensorforge::intel_esimd::simd<float, 32> v301_data(r1.template select<32, 1>(1344));
            r1.template select<32, 1>(1344) = (v301_data + (v293_data * v59_data));
            tensorforge::intel_esimd::simd<float, 32> v306_data(r1.template select<32, 1>(2176));
            r1.template select<32, 1>(2176) = (v306_data + (v293_data * v64_data));
            tensorforge::intel_esimd::simd<float, 32> v311_data(r1.template select<32, 1>(3008));
            r1.template select<32, 1>(3008) = (v311_data + (v293_data * v69_data));
            tensorforge::intel_esimd::simd<float, 32> v316_data(r1.template select<32, 1>(3840));
            r1.template select<32, 1>(3840) = (v316_data + (v293_data * v74_data));
            tensorforge::intel_esimd::simd<float, 32> v321_data(r1.template select<32, 1>(4672));
            r1.template select<32, 1>(4672) = (v321_data + (v293_data * v79_data));
            tensorforge::intel_esimd::simd<float, 32> v323_data(r0.template select<32, 1>(576));
            tensorforge::intel_esimd::simd<float, 32> v326_data(r1.template select<32, 1>(576));
            r1.template select<32, 1>(576) = (v326_data + (v323_data * v54_data));
            tensorforge::intel_esimd::simd<float, 32> v331_data(r1.template select<32, 1>(1408));
            r1.template select<32, 1>(1408) = (v331_data + (v323_data * v59_data));
            tensorforge::intel_esimd::simd<float, 32> v336_data(r1.template select<32, 1>(2240));
            r1.template select<32, 1>(2240) = (v336_data + (v323_data * v64_data));
            tensorforge::intel_esimd::simd<float, 32> v341_data(r1.template select<32, 1>(3072));
            r1.template select<32, 1>(3072) = (v341_data + (v323_data * v69_data));
            tensorforge::intel_esimd::simd<float, 32> v346_data(r1.template select<32, 1>(3904));
            r1.template select<32, 1>(3904) = (v346_data + (v323_data * v74_data));
            tensorforge::intel_esimd::simd<float, 32> v351_data(r1.template select<32, 1>(4736));
            r1.template select<32, 1>(4736) = (v351_data + (v323_data * v79_data));
            tensorforge::intel_esimd::simd<float, 32> v353_data(r0.template select<32, 1>(640));
            tensorforge::intel_esimd::simd<float, 32> v356_data(r1.template select<32, 1>(640));
            r1.template select<32, 1>(640) = (v356_data + (v353_data * v54_data));
            tensorforge::intel_esimd::simd<float, 32> v361_data(r1.template select<32, 1>(1472));
            r1.template select<32, 1>(1472) = (v361_data + (v353_data * v59_data));
            tensorforge::intel_esimd::simd<float, 32> v366_data(r1.template select<32, 1>(2304));
            r1.template select<32, 1>(2304) = (v366_data + (v353_data * v64_data));
            tensorforge::intel_esimd::simd<float, 32> v371_data(r1.template select<32, 1>(3136));
            r1.template select<32, 1>(3136) = (v371_data + (v353_data * v69_data));
            tensorforge::intel_esimd::simd<float, 32> v376_data(r1.template select<32, 1>(3968));
            r1.template select<32, 1>(3968) = (v376_data + (v353_data * v74_data));
            tensorforge::intel_esimd::simd<float, 32> v381_data(r1.template select<32, 1>(4800));
            r1.template select<32, 1>(4800) = (v381_data + (v353_data * v79_data));
            tensorforge::intel_esimd::simd<float, 32> v383_data(r0.template select<32, 1>(704));
            tensorforge::intel_esimd::simd<float, 32> v386_data(r1.template select<32, 1>(704));
            r1.template select<32, 1>(704) = (v386_data + (v383_data * v54_data));
            tensorforge::intel_esimd::simd<float, 32> v391_data(r1.template select<32, 1>(1536));
            r1.template select<32, 1>(1536) = (v391_data + (v383_data * v59_data));
            tensorforge::intel_esimd::simd<float, 32> v396_data(r1.template select<32, 1>(2368));
            r1.template select<32, 1>(2368) = (v396_data + (v383_data * v64_data));
            tensorforge::intel_esimd::simd<float, 32> v401_data(r1.template select<32, 1>(3200));
            r1.template select<32, 1>(3200) = (v401_data + (v383_data * v69_data));
            tensorforge::intel_esimd::simd<float, 32> v406_data(r1.template select<32, 1>(4032));
            r1.template select<32, 1>(4032) = (v406_data + (v383_data * v74_data));
            tensorforge::intel_esimd::simd<float, 32> v411_data(r1.template select<32, 1>(4864));
            r1.template select<32, 1>(4864) = (v411_data + (v383_data * v79_data));
            tensorforge::intel_esimd::simd<float, 32> v413_data(r0.template select<32, 1>(768));
            tensorforge::intel_esimd::simd<float, 32> v416_data(r1.template select<32, 1>(768));
            r1.template select<32, 1>(768) = (v416_data + (v413_data * v54_data));
            tensorforge::intel_esimd::simd<float, 32> v421_data(r1.template select<32, 1>(1600));
            r1.template select<32, 1>(1600) = (v421_data + (v413_data * v59_data));
            tensorforge::intel_esimd::simd<float, 32> v426_data(r1.template select<32, 1>(2432));
            r1.template select<32, 1>(2432) = (v426_data + (v413_data * v64_data));
            tensorforge::intel_esimd::simd<float, 32> v431_data(r1.template select<32, 1>(3264));
            r1.template select<32, 1>(3264) = (v431_data + (v413_data * v69_data));
            tensorforge::intel_esimd::simd<float, 32> v436_data(r1.template select<32, 1>(4096));
            r1.template select<32, 1>(4096) = (v436_data + (v413_data * v74_data));
            tensorforge::intel_esimd::simd<float, 32> v441_data(r1.template select<32, 1>(4928));
            r1.template select<32, 1>(4928) = (v441_data + (v413_data * v79_data));
            tensorforge::intel_esimd::simd<float, 32> v443_data(r0.template select<32, 1>(32));
            tensorforge::intel_esimd::simd<float, 32> v446_data(r1.template select<32, 1>(32));
            r1.template select<32, 1>(32) = (v446_data + (v443_data * v54_data));
            tensorforge::intel_esimd::simd<float, 32> v451_data(r1.template select<32, 1>(864));
            r1.template select<32, 1>(864) = (v451_data + (v443_data * v59_data));
            tensorforge::intel_esimd::simd<float, 32> v456_data(r1.template select<32, 1>(1696));
            r1.template select<32, 1>(1696) = (v456_data + (v443_data * v64_data));
            tensorforge::intel_esimd::simd<float, 32> v461_data(r1.template select<32, 1>(2528));
            r1.template select<32, 1>(2528) = (v461_data + (v443_data * v69_data));
            tensorforge::intel_esimd::simd<float, 32> v466_data(r1.template select<32, 1>(3360));
            r1.template select<32, 1>(3360) = (v466_data + (v443_data * v74_data));
            tensorforge::intel_esimd::simd<float, 32> v471_data(r1.template select<32, 1>(4192));
            r1.template select<32, 1>(4192) = (v471_data + (v443_data * v79_data));
            tensorforge::intel_esimd::simd<float, 32> v473_data(r0.template select<32, 1>(96));
            tensorforge::intel_esimd::simd<float, 32> v476_data(r1.template select<32, 1>(96));
            r1.template select<32, 1>(96) = (v476_data + (v473_data * v54_data));
            tensorforge::intel_esimd::simd<float, 32> v481_data(r1.template select<32, 1>(928));
            r1.template select<32, 1>(928) = (v481_data + (v473_data * v59_data));
            tensorforge::intel_esimd::simd<float, 32> v486_data(r1.template select<32, 1>(1760));
            r1.template select<32, 1>(1760) = (v486_data + (v473_data * v64_data));
            tensorforge::intel_esimd::simd<float, 32> v491_data(r1.template select<32, 1>(2592));
            r1.template select<32, 1>(2592) = (v491_data + (v473_data * v69_data));
            tensorforge::intel_esimd::simd<float, 32> v496_data(r1.template select<32, 1>(3424));
            r1.template select<32, 1>(3424) = (v496_data + (v473_data * v74_data));
            tensorforge::intel_esimd::simd<float, 32> v501_data(r1.template select<32, 1>(4256));
            r1.template select<32, 1>(4256) = (v501_data + (v473_data * v79_data));
            tensorforge::intel_esimd::simd<float, 32> v503_data(r0.template select<32, 1>(160));
            tensorforge::intel_esimd::simd<float, 32> v506_data(r1.template select<32, 1>(160));
            r1.template select<32, 1>(160) = (v506_data + (v503_data * v54_data));
            tensorforge::intel_esimd::simd<float, 32> v511_data(r1.template select<32, 1>(992));
            r1.template select<32, 1>(992) = (v511_data + (v503_data * v59_data));
            tensorforge::intel_esimd::simd<float, 32> v516_data(r1.template select<32, 1>(1824));
            r1.template select<32, 1>(1824) = (v516_data + (v503_data * v64_data));
            tensorforge::intel_esimd::simd<float, 32> v521_data(r1.template select<32, 1>(2656));
            r1.template select<32, 1>(2656) = (v521_data + (v503_data * v69_data));
            tensorforge::intel_esimd::simd<float, 32> v526_data(r1.template select<32, 1>(3488));
            r1.template select<32, 1>(3488) = (v526_data + (v503_data * v74_data));
            tensorforge::intel_esimd::simd<float, 32> v531_data(r1.template select<32, 1>(4320));
            r1.template select<32, 1>(4320) = (v531_data + (v503_data * v79_data));
            tensorforge::intel_esimd::simd<float, 32> v533_data(r0.template select<32, 1>(224));
            tensorforge::intel_esimd::simd<float, 32> v536_data(r1.template select<32, 1>(224));
            r1.template select<32, 1>(224) = (v536_data + (v533_data * v54_data));
            tensorforge::intel_esimd::simd<float, 32> v541_data(r1.template select<32, 1>(1056));
            r1.template select<32, 1>(1056) = (v541_data + (v533_data * v59_data));
            tensorforge::intel_esimd::simd<float, 32> v546_data(r1.template select<32, 1>(1888));
            r1.template select<32, 1>(1888) = (v546_data + (v533_data * v64_data));
            tensorforge::intel_esimd::simd<float, 32> v551_data(r1.template select<32, 1>(2720));
            r1.template select<32, 1>(2720) = (v551_data + (v533_data * v69_data));
            tensorforge::intel_esimd::simd<float, 32> v556_data(r1.template select<32, 1>(3552));
            r1.template select<32, 1>(3552) = (v556_data + (v533_data * v74_data));
            tensorforge::intel_esimd::simd<float, 32> v561_data(r1.template select<32, 1>(4384));
            r1.template select<32, 1>(4384) = (v561_data + (v533_data * v79_data));
            tensorforge::intel_esimd::simd<float, 32> v563_data(r0.template select<32, 1>(288));
            tensorforge::intel_esimd::simd<float, 32> v566_data(r1.template select<32, 1>(288));
            r1.template select<32, 1>(288) = (v566_data + (v563_data * v54_data));
            tensorforge::intel_esimd::simd<float, 32> v571_data(r1.template select<32, 1>(1120));
            r1.template select<32, 1>(1120) = (v571_data + (v563_data * v59_data));
            tensorforge::intel_esimd::simd<float, 32> v576_data(r1.template select<32, 1>(1952));
            r1.template select<32, 1>(1952) = (v576_data + (v563_data * v64_data));
            tensorforge::intel_esimd::simd<float, 32> v581_data(r1.template select<32, 1>(2784));
            r1.template select<32, 1>(2784) = (v581_data + (v563_data * v69_data));
            tensorforge::intel_esimd::simd<float, 32> v586_data(r1.template select<32, 1>(3616));
            r1.template select<32, 1>(3616) = (v586_data + (v563_data * v74_data));
            tensorforge::intel_esimd::simd<float, 32> v591_data(r1.template select<32, 1>(4448));
            r1.template select<32, 1>(4448) = (v591_data + (v563_data * v79_data));
            tensorforge::intel_esimd::simd<float, 32> v593_data(r0.template select<32, 1>(352));
            tensorforge::intel_esimd::simd<float, 32> v596_data(r1.template select<32, 1>(352));
            r1.template select<32, 1>(352) = (v596_data + (v593_data * v54_data));
            tensorforge::intel_esimd::simd<float, 32> v601_data(r1.template select<32, 1>(1184));
            r1.template select<32, 1>(1184) = (v601_data + (v593_data * v59_data));
            tensorforge::intel_esimd::simd<float, 32> v606_data(r1.template select<32, 1>(2016));
            r1.template select<32, 1>(2016) = (v606_data + (v593_data * v64_data));
            tensorforge::intel_esimd::simd<float, 32> v611_data(r1.template select<32, 1>(2848));
            r1.template select<32, 1>(2848) = (v611_data + (v593_data * v69_data));
            tensorforge::intel_esimd::simd<float, 32> v616_data(r1.template select<32, 1>(3680));
            r1.template select<32, 1>(3680) = (v616_data + (v593_data * v74_data));
            tensorforge::intel_esimd::simd<float, 32> v621_data(r1.template select<32, 1>(4512));
            r1.template select<32, 1>(4512) = (v621_data + (v593_data * v79_data));
            tensorforge::intel_esimd::simd<float, 32> v623_data(r0.template select<32, 1>(416));
            tensorforge::intel_esimd::simd<float, 32> v626_data(r1.template select<32, 1>(416));
            r1.template select<32, 1>(416) = (v626_data + (v623_data * v54_data));
            tensorforge::intel_esimd::simd<float, 32> v631_data(r1.template select<32, 1>(1248));
            r1.template select<32, 1>(1248) = (v631_data + (v623_data * v59_data));
            tensorforge::intel_esimd::simd<float, 32> v636_data(r1.template select<32, 1>(2080));
            r1.template select<32, 1>(2080) = (v636_data + (v623_data * v64_data));
            tensorforge::intel_esimd::simd<float, 32> v641_data(r1.template select<32, 1>(2912));
            r1.template select<32, 1>(2912) = (v641_data + (v623_data * v69_data));
            tensorforge::intel_esimd::simd<float, 32> v646_data(r1.template select<32, 1>(3744));
            r1.template select<32, 1>(3744) = (v646_data + (v623_data * v74_data));
            tensorforge::intel_esimd::simd<float, 32> v651_data(r1.template select<32, 1>(4576));
            r1.template select<32, 1>(4576) = (v651_data + (v623_data * v79_data));
            tensorforge::intel_esimd::simd<float, 32> v653_data(r0.template select<32, 1>(480));
            tensorforge::intel_esimd::simd<float, 32> v656_data(r1.template select<32, 1>(480));
            r1.template select<32, 1>(480) = (v656_data + (v653_data * v54_data));
            tensorforge::intel_esimd::simd<float, 32> v661_data(r1.template select<32, 1>(1312));
            r1.template select<32, 1>(1312) = (v661_data + (v653_data * v59_data));
            tensorforge::intel_esimd::simd<float, 32> v666_data(r1.template select<32, 1>(2144));
            r1.template select<32, 1>(2144) = (v666_data + (v653_data * v64_data));
            tensorforge::intel_esimd::simd<float, 32> v671_data(r1.template select<32, 1>(2976));
            r1.template select<32, 1>(2976) = (v671_data + (v653_data * v69_data));
            tensorforge::intel_esimd::simd<float, 32> v676_data(r1.template select<32, 1>(3808));
            r1.template select<32, 1>(3808) = (v676_data + (v653_data * v74_data));
            tensorforge::intel_esimd::simd<float, 32> v681_data(r1.template select<32, 1>(4640));
            r1.template select<32, 1>(4640) = (v681_data + (v653_data * v79_data));
            tensorforge::intel_esimd::simd<float, 32> v683_data(r0.template select<32, 1>(544));
            tensorforge::intel_esimd::simd<float, 32> v686_data(r1.template select<32, 1>(544));
            r1.template select<32, 1>(544) = (v686_data + (v683_data * v54_data));
            tensorforge::intel_esimd::simd<float, 32> v691_data(r1.template select<32, 1>(1376));
            r1.template select<32, 1>(1376) = (v691_data + (v683_data * v59_data));
            tensorforge::intel_esimd::simd<float, 32> v696_data(r1.template select<32, 1>(2208));
            r1.template select<32, 1>(2208) = (v696_data + (v683_data * v64_data));
            tensorforge::intel_esimd::simd<float, 32> v701_data(r1.template select<32, 1>(3040));
            r1.template select<32, 1>(3040) = (v701_data + (v683_data * v69_data));
            tensorforge::intel_esimd::simd<float, 32> v706_data(r1.template select<32, 1>(3872));
            r1.template select<32, 1>(3872) = (v706_data + (v683_data * v74_data));
            tensorforge::intel_esimd::simd<float, 32> v711_data(r1.template select<32, 1>(4704));
            r1.template select<32, 1>(4704) = (v711_data + (v683_data * v79_data));
            tensorforge::intel_esimd::simd<float, 32> v713_data(r0.template select<32, 1>(608));
            tensorforge::intel_esimd::simd<float, 32> v716_data(r1.template select<32, 1>(608));
            r1.template select<32, 1>(608) = (v716_data + (v713_data * v54_data));
            tensorforge::intel_esimd::simd<float, 32> v721_data(r1.template select<32, 1>(1440));
            r1.template select<32, 1>(1440) = (v721_data + (v713_data * v59_data));
            tensorforge::intel_esimd::simd<float, 32> v726_data(r1.template select<32, 1>(2272));
            r1.template select<32, 1>(2272) = (v726_data + (v713_data * v64_data));
            tensorforge::intel_esimd::simd<float, 32> v731_data(r1.template select<32, 1>(3104));
            r1.template select<32, 1>(3104) = (v731_data + (v713_data * v69_data));
            tensorforge::intel_esimd::simd<float, 32> v736_data(r1.template select<32, 1>(3936));
            r1.template select<32, 1>(3936) = (v736_data + (v713_data * v74_data));
            tensorforge::intel_esimd::simd<float, 32> v741_data(r1.template select<32, 1>(4768));
            r1.template select<32, 1>(4768) = (v741_data + (v713_data * v79_data));
            tensorforge::intel_esimd::simd<float, 32> v743_data(r0.template select<32, 1>(672));
            tensorforge::intel_esimd::simd<float, 32> v746_data(r1.template select<32, 1>(672));
            r1.template select<32, 1>(672) = (v746_data + (v743_data * v54_data));
            tensorforge::intel_esimd::simd<float, 32> v751_data(r1.template select<32, 1>(1504));
            r1.template select<32, 1>(1504) = (v751_data + (v743_data * v59_data));
            tensorforge::intel_esimd::simd<float, 32> v756_data(r1.template select<32, 1>(2336));
            r1.template select<32, 1>(2336) = (v756_data + (v743_data * v64_data));
            tensorforge::intel_esimd::simd<float, 32> v761_data(r1.template select<32, 1>(3168));
            r1.template select<32, 1>(3168) = (v761_data + (v743_data * v69_data));
            tensorforge::intel_esimd::simd<float, 32> v766_data(r1.template select<32, 1>(4000));
            r1.template select<32, 1>(4000) = (v766_data + (v743_data * v74_data));
            tensorforge::intel_esimd::simd<float, 32> v771_data(r1.template select<32, 1>(4832));
            r1.template select<32, 1>(4832) = (v771_data + (v743_data * v79_data));
            tensorforge::intel_esimd::simd<float, 32> v773_data(r0.template select<32, 1>(736));
            tensorforge::intel_esimd::simd<float, 32> v776_data(r1.template select<32, 1>(736));
            r1.template select<32, 1>(736) = (v776_data + (v773_data * v54_data));
            tensorforge::intel_esimd::simd<float, 32> v781_data(r1.template select<32, 1>(1568));
            r1.template select<32, 1>(1568) = (v781_data + (v773_data * v59_data));
            tensorforge::intel_esimd::simd<float, 32> v786_data(r1.template select<32, 1>(2400));
            r1.template select<32, 1>(2400) = (v786_data + (v773_data * v64_data));
            tensorforge::intel_esimd::simd<float, 32> v791_data(r1.template select<32, 1>(3232));
            r1.template select<32, 1>(3232) = (v791_data + (v773_data * v69_data));
            tensorforge::intel_esimd::simd<float, 32> v796_data(r1.template select<32, 1>(4064));
            r1.template select<32, 1>(4064) = (v796_data + (v773_data * v74_data));
            tensorforge::intel_esimd::simd<float, 32> v801_data(r1.template select<32, 1>(4896));
            r1.template select<32, 1>(4896) = (v801_data + (v773_data * v79_data));
            tensorforge::intel_esimd::simd<float, 32> v803_data(r0.template select<32, 1>(800));
            tensorforge::intel_esimd::simd<float, 32> v806_data(r1.template select<32, 1>(800));
            r1.template select<32, 1>(800) = (v806_data + (v803_data * v54_data));
            tensorforge::intel_esimd::simd<float, 32> v811_data(r1.template select<32, 1>(1632));
            r1.template select<32, 1>(1632) = (v811_data + (v803_data * v59_data));
            tensorforge::intel_esimd::simd<float, 32> v816_data(r1.template select<32, 1>(2464));
            r1.template select<32, 1>(2464) = (v816_data + (v803_data * v64_data));
            tensorforge::intel_esimd::simd<float, 32> v821_data(r1.template select<32, 1>(3296));
            r1.template select<32, 1>(3296) = (v821_data + (v803_data * v69_data));
            tensorforge::intel_esimd::simd<float, 32> v826_data(r1.template select<32, 1>(4128));
            r1.template select<32, 1>(4128) = (v826_data + (v803_data * v74_data));
            tensorforge::intel_esimd::simd<float, 32> v831_data(r1.template select<32, 1>(4960));
            r1.template select<32, 1>(4960) = (v831_data + (v803_data * v79_data));
            // wait(r2 = load{g>r}(glb_m2););
            tensorforge::intel_esimd::simd<float, 384> r3(0.0f);
            // ir3 = +(r1)
            // [(20, 35), (0, 1), (0, 6)] []
            tensorforge::intel_esimd::simd<float, 384> ir3(0.0f);
            tensorforge::intel_esimd::simd<float, 12> v835_data(r1.template select<12, 1>(788));
            tensorforge::intel_esimd::simd<float, 12> v836_data(ir3.template select<12, 1>(20));
            ir3.template select<12, 1>(20) = (v836_data + v835_data);
            tensorforge::intel_esimd::simd<float, 12> v838_data(r1.template select<12, 1>(1620));
            tensorforge::intel_esimd::simd<float, 12> v839_data(ir3.template select<12, 1>(84));
            ir3.template select<12, 1>(84) = (v839_data + v838_data);
            tensorforge::intel_esimd::simd<float, 12> v841_data(r1.template select<12, 1>(2452));
            tensorforge::intel_esimd::simd<float, 12> v842_data(ir3.template select<12, 1>(148));
            ir3.template select<12, 1>(148) = (v842_data + v841_data);
            tensorforge::intel_esimd::simd<float, 12> v844_data(r1.template select<12, 1>(3284));
            tensorforge::intel_esimd::simd<float, 12> v845_data(ir3.template select<12, 1>(212));
            ir3.template select<12, 1>(212) = (v845_data + v844_data);
            tensorforge::intel_esimd::simd<float, 12> v847_data(r1.template select<12, 1>(4116));
            tensorforge::intel_esimd::simd<float, 12> v848_data(ir3.template select<12, 1>(276));
            ir3.template select<12, 1>(276) = (v848_data + v847_data);
            tensorforge::intel_esimd::simd<float, 12> v850_data(r1.template select<12, 1>(4948));
            tensorforge::intel_esimd::simd<float, 12> v851_data(ir3.template select<12, 1>(340));
            ir3.template select<12, 1>(340) = (v851_data + v850_data);
            tensorforge::intel_esimd::simd<float, 3> v853_data(r1.template select<3, 1>(800));
            tensorforge::intel_esimd::simd<float, 3> v854_data(ir3.template select<3, 1>(32));
            ir3.template select<3, 1>(32) = (v854_data + v853_data);
            tensorforge::intel_esimd::simd<float, 3> v856_data(r1.template select<3, 1>(1632));
            tensorforge::intel_esimd::simd<float, 3> v857_data(ir3.template select<3, 1>(96));
            ir3.template select<3, 1>(96) = (v857_data + v856_data);
            tensorforge::intel_esimd::simd<float, 3> v859_data(r1.template select<3, 1>(2464));
            tensorforge::intel_esimd::simd<float, 3> v860_data(ir3.template select<3, 1>(160));
            ir3.template select<3, 1>(160) = (v860_data + v859_data);
            tensorforge::intel_esimd::simd<float, 3> v862_data(r1.template select<3, 1>(3296));
            tensorforge::intel_esimd::simd<float, 3> v863_data(ir3.template select<3, 1>(224));
            ir3.template select<3, 1>(224) = (v863_data + v862_data);
            tensorforge::intel_esimd::simd<float, 3> v865_data(r1.template select<3, 1>(4128));
            tensorforge::intel_esimd::simd<float, 3> v866_data(ir3.template select<3, 1>(288));
            ir3.template select<3, 1>(288) = (v866_data + v865_data);
            tensorforge::intel_esimd::simd<float, 3> v868_data(r1.template select<3, 1>(4960));
            tensorforge::intel_esimd::simd<float, 3> v869_data(ir3.template select<3, 1>(352));
            ir3.template select<3, 1>(352) = (v869_data + v868_data);
            // r3 = ir3 + r2
            #pragma unroll
            for (int32_t v871_n1 = 0; v871_n1 < 1; ++v871_n1) {
              int32_t v875_a = 20 + (v871_n1 * 64);
              #pragma unroll
              for (int32_t v872_n2 = 0; v872_n2 < 6; ++v872_n2) {
                int32_t v876_a = v875_a + (v872_n2 * 64);
                tensorforge::intel_esimd::simd<float, 12> v877_data(ir3.template select<12, 1>(v876_a));
                tensorforge::intel_esimd::simd<float, 12> v878_data(r2.template select<12, 1>(v876_a));
                r3.template select<12, 1>(v876_a) = (v878_data + v877_data);
              }
            }
            #pragma unroll
            for (int32_t v880_n1 = 0; v880_n1 < 1; ++v880_n1) {
              int32_t v884_a = 32 + (v880_n1 * 64);
              #pragma unroll
              for (int32_t v881_n2 = 0; v881_n2 < 6; ++v881_n2) {
                int32_t v885_a = v884_a + (v881_n2 * 64);
                tensorforge::intel_esimd::simd<float, 3> v886_data(ir3.template select<3, 1>(v885_a));
                tensorforge::intel_esimd::simd<float, 3> v887_data(r2.template select<3, 1>(v885_a));
                r3.template select<3, 1>(v885_a) = (v887_data + v886_data);
              }
            }
            // glb_m2 = store{r>g}(r3);
            #pragma unroll
            for (int32_t v889_i1 = 0; v889_i1 < 1; ++v889_i1) {
              int32_t v893_a = 20 + (v889_i1 * 64);
              int32_t v902_a = 20_i32 + ((v889_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v890_i2 = 0; v890_i2 < 6; ++v890_i2) {
                tensorforge::intel_esimd::simd<float, 12> v895_data(r3.template select<12, 1>((v893_a + (v890_i2 * 64))));
                v895_data.copy_to(glb_m2 + ((v902_a + (v890_i2 * 832))));
              }
            }
            #pragma unroll
            for (int32_t v904_i1 = 0; v904_i1 < 1; ++v904_i1) {
              int32_t v908_a = 32 + (v904_i1 * 64);
              int32_t v917_a = 32_i32 + ((v904_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v905_i2 = 0; v905_i2 < 6; ++v905_i2) {
                tensorforge::intel_esimd::simd<float, 3> v910_data(r3.template select<3, 1>((v908_a + (v905_i2 * 64))));
                v910_data.copy_to(glb_m2 + ((v917_a + (v905_i2 * 832))));
              }
            }
          }
        }
      }
    });
  });
}

