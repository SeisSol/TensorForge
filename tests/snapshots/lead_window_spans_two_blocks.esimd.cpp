// === base name ===
kernel_792d9edb0e1aaa43

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_792d9edb0e1aaa43 = {{1, 32, 1}, 32, 64, 1, 32, 64, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_792d9edb0e1aaa43(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_792d9edb0e1aaa43(const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_792d9edb0e1aaa43(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 32, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 32 - 1) / 32;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 32;
  config.block[2] = 1;
  config.sharedMemBytes = 16 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_792d9edb0e1aaa43(const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_792d9edb0e1aaa43(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_792d9edb0e1aaa43(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_792d9edb0e1aaa43(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<16 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes (64 active) x 32 per block = block 1x32x1, 64 B shared, occupancy grid
        // operands:
        //   m0 64×13(64×13) {0..64}×{0..13} pointer_based
        //   m1 6(6) {0..6} none
        //   m2 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} pointer_based
        // operations:
        //   t0[i,j,l] = m0[i,j] × m1[l]
        //   m2[i,j,l]@{20..35}×{12..13}×{0..6} += t0[i,j,l]@{20..35}×{12..13}×{0..6}
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":64,"block":[1,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":16}],"shared_bytes":64,"shared_elements":16,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"pointer_based","alias":"A","bbox":[[0,0],[64,13]],"name":"m0","ordered":false,"parts":1,"shape":[64,13],"variant":false},{"addressing":"none","alias":"v","bbox":[[0],[6]],"name":"m1","ordered":false,"parts":1,"shape":[6],"variant":false},{"addressing":"pointer_based","alias":"D","bbox":[[0,0,0],[64,13,6]],"name":"m2","ordered":false,"parts":1,"shape":[64,13,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0,0],[64,13,6]],"is_tmp":true,"name":"t0","offset":[0,0,0],"shape":[64,13,6]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[64,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[64,13]},{"addressing":"none","bbox":[[0],[6]],"is_tmp":false,"name":"m1","offset":[0],"shape":[6]}],"permute":[[0,1],[0]],"target":[[0,1],[2]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0,0],[15,1,6]],"is_tmp":false,"name":"m2","offset":[20,12,0],"shape":[64,13,6]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0,0],[15,1,6]],"is_tmp":true,"name":"t0","offset":[20,12,0],"shape":[64,13,6]}],"permute":[[0,1,2]],"target":[[0,1,2]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (0 * item.get_local_id(1) + 16);
          const float *const __restrict__ ptr_glb_m1 = &m1[0];
          tensorforge::SlmPtr<float> glb_m1 = totalShrMem + (0);
          // glb_m1 = load{g>s}(ptr_glb_m1[0])
          if (item.get_local_id(1) == 0) {
            tensorforge::intel_esimd::simd<float, 6> v9_ld;
            v9_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 6>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v9_ld);
          }
          item.barrier();
          for (size_t v10_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v10_batchId0 < numElements0; v10_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v11_ahead1 = v10_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v13_batchId1 = (v11_ahead1 < numElements0) ? v11_ahead1 : v10_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v10_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v10_batchId0][0 + m0_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v10_batchId0][0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 832> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v21_i0 = 0; v21_i0 < 2; ++v21_i0) {
                int32_t v23_lead = v21_i0 * 32;
                #pragma unroll
                for (int32_t v22_i1 = 0; v22_i1 < 13; ++v22_i1) {
                  int32_t v26_a = v23_lead + (v22_i1 * 64);
                  tensorforge::intel_esimd::simd<float, 32> v27_data;
                  v27_data.copy_from(glb_m0 + (v26_a));
                  r0.template select<32, 1>(v26_a) = v27_data;
                }
              }
              tensorforge::intel_esimd::simd<float, 384> r2(0.0f);
              // r2 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v811_i1 = 0; v811_i1 < 1; ++v811_i1) {
                int32_t v819_a = 20_i32 + ((v811_i1 + 12) * 64);
                int32_t v824_a = 20 + (v811_i1 * 64);
                #pragma unroll
                for (int32_t v812_i2 = 0; v812_i2 < 6; ++v812_i2) {
                  tensorforge::intel_esimd::simd<float, 12> v821_data;
                  v821_data.copy_from(glb_m2 + ((v819_a + (v812_i2 * 832))));
                  r2.template select<12, 1>((v824_a + (v812_i2 * 64))) = v821_data;
                }
              }
              #pragma unroll
              for (int32_t v826_i1 = 0; v826_i1 < 1; ++v826_i1) {
                int32_t v834_a = 32_i32 + ((v826_i1 + 12) * 64);
                int32_t v839_a = 32 + (v826_i1 * 64);
                #pragma unroll
                for (int32_t v827_i2 = 0; v827_i2 < 6; ++v827_i2) {
                  tensorforge::intel_esimd::simd<float, 3> v836_data;
                  v836_data.copy_from(glb_m2 + ((v834_a + (v827_i2 * 832))));
                  r2.template select<3, 1>((v839_a + (v827_i2 * 64))) = v836_data;
                }
              }
              tensorforge::intel_esimd::simd<float, 4992> r1(0.0f);
              // r1 = +(r0 * glb_m1) + None
              // [(0, 64), (0, 13), (0, 6)] []
              tensorforge::intel_esimd::simd<float, 32> v30_data(r0.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> glb_m1_w0 = tensorforge::slmLoad<float, 16>(glb_m1 + 0);
              float v31_data = glb_m1_w0[0];
              tensorforge::intel_esimd::simd<float, 32> v33_data(r1.template select<32, 1>(0));
              r1.template select<32, 1>(0) = (v33_data + (v30_data * v31_data));
              float v36_data = glb_m1_w0[1];
              tensorforge::intel_esimd::simd<float, 32> v38_data(r1.template select<32, 1>(832));
              r1.template select<32, 1>(832) = (v38_data + (v30_data * v36_data));
              float v41_data = glb_m1_w0[2];
              tensorforge::intel_esimd::simd<float, 32> v43_data(r1.template select<32, 1>(1664));
              r1.template select<32, 1>(1664) = (v43_data + (v30_data * v41_data));
              float v46_data = glb_m1_w0[3];
              tensorforge::intel_esimd::simd<float, 32> v48_data(r1.template select<32, 1>(2496));
              r1.template select<32, 1>(2496) = (v48_data + (v30_data * v46_data));
              float v51_data = glb_m1_w0[4];
              tensorforge::intel_esimd::simd<float, 32> v53_data(r1.template select<32, 1>(3328));
              r1.template select<32, 1>(3328) = (v53_data + (v30_data * v51_data));
              float v56_data = glb_m1_w0[5];
              tensorforge::intel_esimd::simd<float, 32> v58_data(r1.template select<32, 1>(4160));
              r1.template select<32, 1>(4160) = (v58_data + (v30_data * v56_data));
              tensorforge::intel_esimd::simd<float, 32> v60_data(r0.template select<32, 1>(64));
              tensorforge::intel_esimd::simd<float, 32> v63_data(r1.template select<32, 1>(64));
              r1.template select<32, 1>(64) = (v63_data + (v60_data * v31_data));
              tensorforge::intel_esimd::simd<float, 32> v68_data(r1.template select<32, 1>(896));
              r1.template select<32, 1>(896) = (v68_data + (v60_data * v36_data));
              tensorforge::intel_esimd::simd<float, 32> v73_data(r1.template select<32, 1>(1728));
              r1.template select<32, 1>(1728) = (v73_data + (v60_data * v41_data));
              tensorforge::intel_esimd::simd<float, 32> v78_data(r1.template select<32, 1>(2560));
              r1.template select<32, 1>(2560) = (v78_data + (v60_data * v46_data));
              tensorforge::intel_esimd::simd<float, 32> v83_data(r1.template select<32, 1>(3392));
              r1.template select<32, 1>(3392) = (v83_data + (v60_data * v51_data));
              tensorforge::intel_esimd::simd<float, 32> v88_data(r1.template select<32, 1>(4224));
              r1.template select<32, 1>(4224) = (v88_data + (v60_data * v56_data));
              tensorforge::intel_esimd::simd<float, 32> v90_data(r0.template select<32, 1>(128));
              tensorforge::intel_esimd::simd<float, 32> v93_data(r1.template select<32, 1>(128));
              r1.template select<32, 1>(128) = (v93_data + (v90_data * v31_data));
              tensorforge::intel_esimd::simd<float, 32> v98_data(r1.template select<32, 1>(960));
              r1.template select<32, 1>(960) = (v98_data + (v90_data * v36_data));
              tensorforge::intel_esimd::simd<float, 32> v103_data(r1.template select<32, 1>(1792));
              r1.template select<32, 1>(1792) = (v103_data + (v90_data * v41_data));
              tensorforge::intel_esimd::simd<float, 32> v108_data(r1.template select<32, 1>(2624));
              r1.template select<32, 1>(2624) = (v108_data + (v90_data * v46_data));
              tensorforge::intel_esimd::simd<float, 32> v113_data(r1.template select<32, 1>(3456));
              r1.template select<32, 1>(3456) = (v113_data + (v90_data * v51_data));
              tensorforge::intel_esimd::simd<float, 32> v118_data(r1.template select<32, 1>(4288));
              r1.template select<32, 1>(4288) = (v118_data + (v90_data * v56_data));
              tensorforge::intel_esimd::simd<float, 32> v120_data(r0.template select<32, 1>(192));
              tensorforge::intel_esimd::simd<float, 32> v123_data(r1.template select<32, 1>(192));
              r1.template select<32, 1>(192) = (v123_data + (v120_data * v31_data));
              tensorforge::intel_esimd::simd<float, 32> v128_data(r1.template select<32, 1>(1024));
              r1.template select<32, 1>(1024) = (v128_data + (v120_data * v36_data));
              tensorforge::intel_esimd::simd<float, 32> v133_data(r1.template select<32, 1>(1856));
              r1.template select<32, 1>(1856) = (v133_data + (v120_data * v41_data));
              tensorforge::intel_esimd::simd<float, 32> v138_data(r1.template select<32, 1>(2688));
              r1.template select<32, 1>(2688) = (v138_data + (v120_data * v46_data));
              tensorforge::intel_esimd::simd<float, 32> v143_data(r1.template select<32, 1>(3520));
              r1.template select<32, 1>(3520) = (v143_data + (v120_data * v51_data));
              tensorforge::intel_esimd::simd<float, 32> v148_data(r1.template select<32, 1>(4352));
              r1.template select<32, 1>(4352) = (v148_data + (v120_data * v56_data));
              tensorforge::intel_esimd::simd<float, 32> v150_data(r0.template select<32, 1>(256));
              tensorforge::intel_esimd::simd<float, 32> v153_data(r1.template select<32, 1>(256));
              r1.template select<32, 1>(256) = (v153_data + (v150_data * v31_data));
              tensorforge::intel_esimd::simd<float, 32> v158_data(r1.template select<32, 1>(1088));
              r1.template select<32, 1>(1088) = (v158_data + (v150_data * v36_data));
              tensorforge::intel_esimd::simd<float, 32> v163_data(r1.template select<32, 1>(1920));
              r1.template select<32, 1>(1920) = (v163_data + (v150_data * v41_data));
              tensorforge::intel_esimd::simd<float, 32> v168_data(r1.template select<32, 1>(2752));
              r1.template select<32, 1>(2752) = (v168_data + (v150_data * v46_data));
              tensorforge::intel_esimd::simd<float, 32> v173_data(r1.template select<32, 1>(3584));
              r1.template select<32, 1>(3584) = (v173_data + (v150_data * v51_data));
              tensorforge::intel_esimd::simd<float, 32> v178_data(r1.template select<32, 1>(4416));
              r1.template select<32, 1>(4416) = (v178_data + (v150_data * v56_data));
              tensorforge::intel_esimd::simd<float, 32> v180_data(r0.template select<32, 1>(320));
              tensorforge::intel_esimd::simd<float, 32> v183_data(r1.template select<32, 1>(320));
              r1.template select<32, 1>(320) = (v183_data + (v180_data * v31_data));
              tensorforge::intel_esimd::simd<float, 32> v188_data(r1.template select<32, 1>(1152));
              r1.template select<32, 1>(1152) = (v188_data + (v180_data * v36_data));
              tensorforge::intel_esimd::simd<float, 32> v193_data(r1.template select<32, 1>(1984));
              r1.template select<32, 1>(1984) = (v193_data + (v180_data * v41_data));
              tensorforge::intel_esimd::simd<float, 32> v198_data(r1.template select<32, 1>(2816));
              r1.template select<32, 1>(2816) = (v198_data + (v180_data * v46_data));
              tensorforge::intel_esimd::simd<float, 32> v203_data(r1.template select<32, 1>(3648));
              r1.template select<32, 1>(3648) = (v203_data + (v180_data * v51_data));
              tensorforge::intel_esimd::simd<float, 32> v208_data(r1.template select<32, 1>(4480));
              r1.template select<32, 1>(4480) = (v208_data + (v180_data * v56_data));
              tensorforge::intel_esimd::simd<float, 32> v210_data(r0.template select<32, 1>(384));
              tensorforge::intel_esimd::simd<float, 32> v213_data(r1.template select<32, 1>(384));
              r1.template select<32, 1>(384) = (v213_data + (v210_data * v31_data));
              tensorforge::intel_esimd::simd<float, 32> v218_data(r1.template select<32, 1>(1216));
              r1.template select<32, 1>(1216) = (v218_data + (v210_data * v36_data));
              tensorforge::intel_esimd::simd<float, 32> v223_data(r1.template select<32, 1>(2048));
              r1.template select<32, 1>(2048) = (v223_data + (v210_data * v41_data));
              tensorforge::intel_esimd::simd<float, 32> v228_data(r1.template select<32, 1>(2880));
              r1.template select<32, 1>(2880) = (v228_data + (v210_data * v46_data));
              tensorforge::intel_esimd::simd<float, 32> v233_data(r1.template select<32, 1>(3712));
              r1.template select<32, 1>(3712) = (v233_data + (v210_data * v51_data));
              tensorforge::intel_esimd::simd<float, 32> v238_data(r1.template select<32, 1>(4544));
              r1.template select<32, 1>(4544) = (v238_data + (v210_data * v56_data));
              tensorforge::intel_esimd::simd<float, 32> v240_data(r0.template select<32, 1>(448));
              tensorforge::intel_esimd::simd<float, 32> v243_data(r1.template select<32, 1>(448));
              r1.template select<32, 1>(448) = (v243_data + (v240_data * v31_data));
              tensorforge::intel_esimd::simd<float, 32> v248_data(r1.template select<32, 1>(1280));
              r1.template select<32, 1>(1280) = (v248_data + (v240_data * v36_data));
              tensorforge::intel_esimd::simd<float, 32> v253_data(r1.template select<32, 1>(2112));
              r1.template select<32, 1>(2112) = (v253_data + (v240_data * v41_data));
              tensorforge::intel_esimd::simd<float, 32> v258_data(r1.template select<32, 1>(2944));
              r1.template select<32, 1>(2944) = (v258_data + (v240_data * v46_data));
              tensorforge::intel_esimd::simd<float, 32> v263_data(r1.template select<32, 1>(3776));
              r1.template select<32, 1>(3776) = (v263_data + (v240_data * v51_data));
              tensorforge::intel_esimd::simd<float, 32> v268_data(r1.template select<32, 1>(4608));
              r1.template select<32, 1>(4608) = (v268_data + (v240_data * v56_data));
              tensorforge::intel_esimd::simd<float, 32> v270_data(r0.template select<32, 1>(512));
              tensorforge::intel_esimd::simd<float, 32> v273_data(r1.template select<32, 1>(512));
              r1.template select<32, 1>(512) = (v273_data + (v270_data * v31_data));
              tensorforge::intel_esimd::simd<float, 32> v278_data(r1.template select<32, 1>(1344));
              r1.template select<32, 1>(1344) = (v278_data + (v270_data * v36_data));
              tensorforge::intel_esimd::simd<float, 32> v283_data(r1.template select<32, 1>(2176));
              r1.template select<32, 1>(2176) = (v283_data + (v270_data * v41_data));
              tensorforge::intel_esimd::simd<float, 32> v288_data(r1.template select<32, 1>(3008));
              r1.template select<32, 1>(3008) = (v288_data + (v270_data * v46_data));
              tensorforge::intel_esimd::simd<float, 32> v293_data(r1.template select<32, 1>(3840));
              r1.template select<32, 1>(3840) = (v293_data + (v270_data * v51_data));
              tensorforge::intel_esimd::simd<float, 32> v298_data(r1.template select<32, 1>(4672));
              r1.template select<32, 1>(4672) = (v298_data + (v270_data * v56_data));
              tensorforge::intel_esimd::simd<float, 32> v300_data(r0.template select<32, 1>(576));
              tensorforge::intel_esimd::simd<float, 32> v303_data(r1.template select<32, 1>(576));
              r1.template select<32, 1>(576) = (v303_data + (v300_data * v31_data));
              tensorforge::intel_esimd::simd<float, 32> v308_data(r1.template select<32, 1>(1408));
              r1.template select<32, 1>(1408) = (v308_data + (v300_data * v36_data));
              tensorforge::intel_esimd::simd<float, 32> v313_data(r1.template select<32, 1>(2240));
              r1.template select<32, 1>(2240) = (v313_data + (v300_data * v41_data));
              tensorforge::intel_esimd::simd<float, 32> v318_data(r1.template select<32, 1>(3072));
              r1.template select<32, 1>(3072) = (v318_data + (v300_data * v46_data));
              tensorforge::intel_esimd::simd<float, 32> v323_data(r1.template select<32, 1>(3904));
              r1.template select<32, 1>(3904) = (v323_data + (v300_data * v51_data));
              tensorforge::intel_esimd::simd<float, 32> v328_data(r1.template select<32, 1>(4736));
              r1.template select<32, 1>(4736) = (v328_data + (v300_data * v56_data));
              tensorforge::intel_esimd::simd<float, 32> v330_data(r0.template select<32, 1>(640));
              tensorforge::intel_esimd::simd<float, 32> v333_data(r1.template select<32, 1>(640));
              r1.template select<32, 1>(640) = (v333_data + (v330_data * v31_data));
              tensorforge::intel_esimd::simd<float, 32> v338_data(r1.template select<32, 1>(1472));
              r1.template select<32, 1>(1472) = (v338_data + (v330_data * v36_data));
              tensorforge::intel_esimd::simd<float, 32> v343_data(r1.template select<32, 1>(2304));
              r1.template select<32, 1>(2304) = (v343_data + (v330_data * v41_data));
              tensorforge::intel_esimd::simd<float, 32> v348_data(r1.template select<32, 1>(3136));
              r1.template select<32, 1>(3136) = (v348_data + (v330_data * v46_data));
              tensorforge::intel_esimd::simd<float, 32> v353_data(r1.template select<32, 1>(3968));
              r1.template select<32, 1>(3968) = (v353_data + (v330_data * v51_data));
              tensorforge::intel_esimd::simd<float, 32> v358_data(r1.template select<32, 1>(4800));
              r1.template select<32, 1>(4800) = (v358_data + (v330_data * v56_data));
              tensorforge::intel_esimd::simd<float, 32> v360_data(r0.template select<32, 1>(704));
              tensorforge::intel_esimd::simd<float, 32> v363_data(r1.template select<32, 1>(704));
              r1.template select<32, 1>(704) = (v363_data + (v360_data * v31_data));
              tensorforge::intel_esimd::simd<float, 32> v368_data(r1.template select<32, 1>(1536));
              r1.template select<32, 1>(1536) = (v368_data + (v360_data * v36_data));
              tensorforge::intel_esimd::simd<float, 32> v373_data(r1.template select<32, 1>(2368));
              r1.template select<32, 1>(2368) = (v373_data + (v360_data * v41_data));
              tensorforge::intel_esimd::simd<float, 32> v378_data(r1.template select<32, 1>(3200));
              r1.template select<32, 1>(3200) = (v378_data + (v360_data * v46_data));
              tensorforge::intel_esimd::simd<float, 32> v383_data(r1.template select<32, 1>(4032));
              r1.template select<32, 1>(4032) = (v383_data + (v360_data * v51_data));
              tensorforge::intel_esimd::simd<float, 32> v388_data(r1.template select<32, 1>(4864));
              r1.template select<32, 1>(4864) = (v388_data + (v360_data * v56_data));
              tensorforge::intel_esimd::simd<float, 32> v390_data(r0.template select<32, 1>(768));
              tensorforge::intel_esimd::simd<float, 32> v393_data(r1.template select<32, 1>(768));
              r1.template select<32, 1>(768) = (v393_data + (v390_data * v31_data));
              tensorforge::intel_esimd::simd<float, 32> v398_data(r1.template select<32, 1>(1600));
              r1.template select<32, 1>(1600) = (v398_data + (v390_data * v36_data));
              tensorforge::intel_esimd::simd<float, 32> v403_data(r1.template select<32, 1>(2432));
              r1.template select<32, 1>(2432) = (v403_data + (v390_data * v41_data));
              tensorforge::intel_esimd::simd<float, 32> v408_data(r1.template select<32, 1>(3264));
              r1.template select<32, 1>(3264) = (v408_data + (v390_data * v46_data));
              tensorforge::intel_esimd::simd<float, 32> v413_data(r1.template select<32, 1>(4096));
              r1.template select<32, 1>(4096) = (v413_data + (v390_data * v51_data));
              tensorforge::intel_esimd::simd<float, 32> v418_data(r1.template select<32, 1>(4928));
              r1.template select<32, 1>(4928) = (v418_data + (v390_data * v56_data));
              tensorforge::intel_esimd::simd<float, 32> v420_data(r0.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v423_data(r1.template select<32, 1>(32));
              r1.template select<32, 1>(32) = (v423_data + (v420_data * v31_data));
              tensorforge::intel_esimd::simd<float, 32> v428_data(r1.template select<32, 1>(864));
              r1.template select<32, 1>(864) = (v428_data + (v420_data * v36_data));
              tensorforge::intel_esimd::simd<float, 32> v433_data(r1.template select<32, 1>(1696));
              r1.template select<32, 1>(1696) = (v433_data + (v420_data * v41_data));
              tensorforge::intel_esimd::simd<float, 32> v438_data(r1.template select<32, 1>(2528));
              r1.template select<32, 1>(2528) = (v438_data + (v420_data * v46_data));
              tensorforge::intel_esimd::simd<float, 32> v443_data(r1.template select<32, 1>(3360));
              r1.template select<32, 1>(3360) = (v443_data + (v420_data * v51_data));
              tensorforge::intel_esimd::simd<float, 32> v448_data(r1.template select<32, 1>(4192));
              r1.template select<32, 1>(4192) = (v448_data + (v420_data * v56_data));
              tensorforge::intel_esimd::simd<float, 32> v450_data(r0.template select<32, 1>(96));
              tensorforge::intel_esimd::simd<float, 32> v453_data(r1.template select<32, 1>(96));
              r1.template select<32, 1>(96) = (v453_data + (v450_data * v31_data));
              tensorforge::intel_esimd::simd<float, 32> v458_data(r1.template select<32, 1>(928));
              r1.template select<32, 1>(928) = (v458_data + (v450_data * v36_data));
              tensorforge::intel_esimd::simd<float, 32> v463_data(r1.template select<32, 1>(1760));
              r1.template select<32, 1>(1760) = (v463_data + (v450_data * v41_data));
              tensorforge::intel_esimd::simd<float, 32> v468_data(r1.template select<32, 1>(2592));
              r1.template select<32, 1>(2592) = (v468_data + (v450_data * v46_data));
              tensorforge::intel_esimd::simd<float, 32> v473_data(r1.template select<32, 1>(3424));
              r1.template select<32, 1>(3424) = (v473_data + (v450_data * v51_data));
              tensorforge::intel_esimd::simd<float, 32> v478_data(r1.template select<32, 1>(4256));
              r1.template select<32, 1>(4256) = (v478_data + (v450_data * v56_data));
              tensorforge::intel_esimd::simd<float, 32> v480_data(r0.template select<32, 1>(160));
              tensorforge::intel_esimd::simd<float, 32> v483_data(r1.template select<32, 1>(160));
              r1.template select<32, 1>(160) = (v483_data + (v480_data * v31_data));
              tensorforge::intel_esimd::simd<float, 32> v488_data(r1.template select<32, 1>(992));
              r1.template select<32, 1>(992) = (v488_data + (v480_data * v36_data));
              tensorforge::intel_esimd::simd<float, 32> v493_data(r1.template select<32, 1>(1824));
              r1.template select<32, 1>(1824) = (v493_data + (v480_data * v41_data));
              tensorforge::intel_esimd::simd<float, 32> v498_data(r1.template select<32, 1>(2656));
              r1.template select<32, 1>(2656) = (v498_data + (v480_data * v46_data));
              tensorforge::intel_esimd::simd<float, 32> v503_data(r1.template select<32, 1>(3488));
              r1.template select<32, 1>(3488) = (v503_data + (v480_data * v51_data));
              tensorforge::intel_esimd::simd<float, 32> v508_data(r1.template select<32, 1>(4320));
              r1.template select<32, 1>(4320) = (v508_data + (v480_data * v56_data));
              tensorforge::intel_esimd::simd<float, 32> v510_data(r0.template select<32, 1>(224));
              tensorforge::intel_esimd::simd<float, 32> v513_data(r1.template select<32, 1>(224));
              r1.template select<32, 1>(224) = (v513_data + (v510_data * v31_data));
              tensorforge::intel_esimd::simd<float, 32> v518_data(r1.template select<32, 1>(1056));
              r1.template select<32, 1>(1056) = (v518_data + (v510_data * v36_data));
              tensorforge::intel_esimd::simd<float, 32> v523_data(r1.template select<32, 1>(1888));
              r1.template select<32, 1>(1888) = (v523_data + (v510_data * v41_data));
              tensorforge::intel_esimd::simd<float, 32> v528_data(r1.template select<32, 1>(2720));
              r1.template select<32, 1>(2720) = (v528_data + (v510_data * v46_data));
              tensorforge::intel_esimd::simd<float, 32> v533_data(r1.template select<32, 1>(3552));
              r1.template select<32, 1>(3552) = (v533_data + (v510_data * v51_data));
              tensorforge::intel_esimd::simd<float, 32> v538_data(r1.template select<32, 1>(4384));
              r1.template select<32, 1>(4384) = (v538_data + (v510_data * v56_data));
              tensorforge::intel_esimd::simd<float, 32> v540_data(r0.template select<32, 1>(288));
              tensorforge::intel_esimd::simd<float, 32> v543_data(r1.template select<32, 1>(288));
              r1.template select<32, 1>(288) = (v543_data + (v540_data * v31_data));
              tensorforge::intel_esimd::simd<float, 32> v548_data(r1.template select<32, 1>(1120));
              r1.template select<32, 1>(1120) = (v548_data + (v540_data * v36_data));
              tensorforge::intel_esimd::simd<float, 32> v553_data(r1.template select<32, 1>(1952));
              r1.template select<32, 1>(1952) = (v553_data + (v540_data * v41_data));
              tensorforge::intel_esimd::simd<float, 32> v558_data(r1.template select<32, 1>(2784));
              r1.template select<32, 1>(2784) = (v558_data + (v540_data * v46_data));
              tensorforge::intel_esimd::simd<float, 32> v563_data(r1.template select<32, 1>(3616));
              r1.template select<32, 1>(3616) = (v563_data + (v540_data * v51_data));
              tensorforge::intel_esimd::simd<float, 32> v568_data(r1.template select<32, 1>(4448));
              r1.template select<32, 1>(4448) = (v568_data + (v540_data * v56_data));
              tensorforge::intel_esimd::simd<float, 32> v570_data(r0.template select<32, 1>(352));
              tensorforge::intel_esimd::simd<float, 32> v573_data(r1.template select<32, 1>(352));
              r1.template select<32, 1>(352) = (v573_data + (v570_data * v31_data));
              tensorforge::intel_esimd::simd<float, 32> v578_data(r1.template select<32, 1>(1184));
              r1.template select<32, 1>(1184) = (v578_data + (v570_data * v36_data));
              tensorforge::intel_esimd::simd<float, 32> v583_data(r1.template select<32, 1>(2016));
              r1.template select<32, 1>(2016) = (v583_data + (v570_data * v41_data));
              tensorforge::intel_esimd::simd<float, 32> v588_data(r1.template select<32, 1>(2848));
              r1.template select<32, 1>(2848) = (v588_data + (v570_data * v46_data));
              tensorforge::intel_esimd::simd<float, 32> v593_data(r1.template select<32, 1>(3680));
              r1.template select<32, 1>(3680) = (v593_data + (v570_data * v51_data));
              tensorforge::intel_esimd::simd<float, 32> v598_data(r1.template select<32, 1>(4512));
              r1.template select<32, 1>(4512) = (v598_data + (v570_data * v56_data));
              tensorforge::intel_esimd::simd<float, 32> v600_data(r0.template select<32, 1>(416));
              tensorforge::intel_esimd::simd<float, 32> v603_data(r1.template select<32, 1>(416));
              r1.template select<32, 1>(416) = (v603_data + (v600_data * v31_data));
              tensorforge::intel_esimd::simd<float, 32> v608_data(r1.template select<32, 1>(1248));
              r1.template select<32, 1>(1248) = (v608_data + (v600_data * v36_data));
              tensorforge::intel_esimd::simd<float, 32> v613_data(r1.template select<32, 1>(2080));
              r1.template select<32, 1>(2080) = (v613_data + (v600_data * v41_data));
              tensorforge::intel_esimd::simd<float, 32> v618_data(r1.template select<32, 1>(2912));
              r1.template select<32, 1>(2912) = (v618_data + (v600_data * v46_data));
              tensorforge::intel_esimd::simd<float, 32> v623_data(r1.template select<32, 1>(3744));
              r1.template select<32, 1>(3744) = (v623_data + (v600_data * v51_data));
              tensorforge::intel_esimd::simd<float, 32> v628_data(r1.template select<32, 1>(4576));
              r1.template select<32, 1>(4576) = (v628_data + (v600_data * v56_data));
              tensorforge::intel_esimd::simd<float, 32> v630_data(r0.template select<32, 1>(480));
              tensorforge::intel_esimd::simd<float, 32> v633_data(r1.template select<32, 1>(480));
              r1.template select<32, 1>(480) = (v633_data + (v630_data * v31_data));
              tensorforge::intel_esimd::simd<float, 32> v638_data(r1.template select<32, 1>(1312));
              r1.template select<32, 1>(1312) = (v638_data + (v630_data * v36_data));
              tensorforge::intel_esimd::simd<float, 32> v643_data(r1.template select<32, 1>(2144));
              r1.template select<32, 1>(2144) = (v643_data + (v630_data * v41_data));
              tensorforge::intel_esimd::simd<float, 32> v648_data(r1.template select<32, 1>(2976));
              r1.template select<32, 1>(2976) = (v648_data + (v630_data * v46_data));
              tensorforge::intel_esimd::simd<float, 32> v653_data(r1.template select<32, 1>(3808));
              r1.template select<32, 1>(3808) = (v653_data + (v630_data * v51_data));
              tensorforge::intel_esimd::simd<float, 32> v658_data(r1.template select<32, 1>(4640));
              r1.template select<32, 1>(4640) = (v658_data + (v630_data * v56_data));
              tensorforge::intel_esimd::simd<float, 32> v660_data(r0.template select<32, 1>(544));
              tensorforge::intel_esimd::simd<float, 32> v663_data(r1.template select<32, 1>(544));
              r1.template select<32, 1>(544) = (v663_data + (v660_data * v31_data));
              tensorforge::intel_esimd::simd<float, 32> v668_data(r1.template select<32, 1>(1376));
              r1.template select<32, 1>(1376) = (v668_data + (v660_data * v36_data));
              tensorforge::intel_esimd::simd<float, 32> v673_data(r1.template select<32, 1>(2208));
              r1.template select<32, 1>(2208) = (v673_data + (v660_data * v41_data));
              tensorforge::intel_esimd::simd<float, 32> v678_data(r1.template select<32, 1>(3040));
              r1.template select<32, 1>(3040) = (v678_data + (v660_data * v46_data));
              tensorforge::intel_esimd::simd<float, 32> v683_data(r1.template select<32, 1>(3872));
              r1.template select<32, 1>(3872) = (v683_data + (v660_data * v51_data));
              tensorforge::intel_esimd::simd<float, 32> v688_data(r1.template select<32, 1>(4704));
              r1.template select<32, 1>(4704) = (v688_data + (v660_data * v56_data));
              tensorforge::intel_esimd::simd<float, 32> v690_data(r0.template select<32, 1>(608));
              tensorforge::intel_esimd::simd<float, 32> v693_data(r1.template select<32, 1>(608));
              r1.template select<32, 1>(608) = (v693_data + (v690_data * v31_data));
              tensorforge::intel_esimd::simd<float, 32> v698_data(r1.template select<32, 1>(1440));
              r1.template select<32, 1>(1440) = (v698_data + (v690_data * v36_data));
              tensorforge::intel_esimd::simd<float, 32> v703_data(r1.template select<32, 1>(2272));
              r1.template select<32, 1>(2272) = (v703_data + (v690_data * v41_data));
              tensorforge::intel_esimd::simd<float, 32> v708_data(r1.template select<32, 1>(3104));
              r1.template select<32, 1>(3104) = (v708_data + (v690_data * v46_data));
              tensorforge::intel_esimd::simd<float, 32> v713_data(r1.template select<32, 1>(3936));
              r1.template select<32, 1>(3936) = (v713_data + (v690_data * v51_data));
              tensorforge::intel_esimd::simd<float, 32> v718_data(r1.template select<32, 1>(4768));
              r1.template select<32, 1>(4768) = (v718_data + (v690_data * v56_data));
              tensorforge::intel_esimd::simd<float, 32> v720_data(r0.template select<32, 1>(672));
              tensorforge::intel_esimd::simd<float, 32> v723_data(r1.template select<32, 1>(672));
              r1.template select<32, 1>(672) = (v723_data + (v720_data * v31_data));
              tensorforge::intel_esimd::simd<float, 32> v728_data(r1.template select<32, 1>(1504));
              r1.template select<32, 1>(1504) = (v728_data + (v720_data * v36_data));
              tensorforge::intel_esimd::simd<float, 32> v733_data(r1.template select<32, 1>(2336));
              r1.template select<32, 1>(2336) = (v733_data + (v720_data * v41_data));
              tensorforge::intel_esimd::simd<float, 32> v738_data(r1.template select<32, 1>(3168));
              r1.template select<32, 1>(3168) = (v738_data + (v720_data * v46_data));
              tensorforge::intel_esimd::simd<float, 32> v743_data(r1.template select<32, 1>(4000));
              r1.template select<32, 1>(4000) = (v743_data + (v720_data * v51_data));
              tensorforge::intel_esimd::simd<float, 32> v748_data(r1.template select<32, 1>(4832));
              r1.template select<32, 1>(4832) = (v748_data + (v720_data * v56_data));
              tensorforge::intel_esimd::simd<float, 32> v750_data(r0.template select<32, 1>(736));
              tensorforge::intel_esimd::simd<float, 32> v753_data(r1.template select<32, 1>(736));
              r1.template select<32, 1>(736) = (v753_data + (v750_data * v31_data));
              tensorforge::intel_esimd::simd<float, 32> v758_data(r1.template select<32, 1>(1568));
              r1.template select<32, 1>(1568) = (v758_data + (v750_data * v36_data));
              tensorforge::intel_esimd::simd<float, 32> v763_data(r1.template select<32, 1>(2400));
              r1.template select<32, 1>(2400) = (v763_data + (v750_data * v41_data));
              tensorforge::intel_esimd::simd<float, 32> v768_data(r1.template select<32, 1>(3232));
              r1.template select<32, 1>(3232) = (v768_data + (v750_data * v46_data));
              tensorforge::intel_esimd::simd<float, 32> v773_data(r1.template select<32, 1>(4064));
              r1.template select<32, 1>(4064) = (v773_data + (v750_data * v51_data));
              tensorforge::intel_esimd::simd<float, 32> v778_data(r1.template select<32, 1>(4896));
              r1.template select<32, 1>(4896) = (v778_data + (v750_data * v56_data));
              tensorforge::intel_esimd::simd<float, 32> v780_data(r0.template select<32, 1>(800));
              tensorforge::intel_esimd::simd<float, 32> v783_data(r1.template select<32, 1>(800));
              r1.template select<32, 1>(800) = (v783_data + (v780_data * v31_data));
              tensorforge::intel_esimd::simd<float, 32> v788_data(r1.template select<32, 1>(1632));
              r1.template select<32, 1>(1632) = (v788_data + (v780_data * v36_data));
              tensorforge::intel_esimd::simd<float, 32> v793_data(r1.template select<32, 1>(2464));
              r1.template select<32, 1>(2464) = (v793_data + (v780_data * v41_data));
              tensorforge::intel_esimd::simd<float, 32> v798_data(r1.template select<32, 1>(3296));
              r1.template select<32, 1>(3296) = (v798_data + (v780_data * v46_data));
              tensorforge::intel_esimd::simd<float, 32> v803_data(r1.template select<32, 1>(4128));
              r1.template select<32, 1>(4128) = (v803_data + (v780_data * v51_data));
              tensorforge::intel_esimd::simd<float, 32> v808_data(r1.template select<32, 1>(4960));
              r1.template select<32, 1>(4960) = (v808_data + (v780_data * v56_data));
              tensorforge::intel_esimd::simd<float, 384> r3(0.0f);
              // ir3 = +(r1)
              // [(20, 35), (0, 1), (0, 6)] []
              tensorforge::intel_esimd::simd<float, 384> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 12> v843_data(r1.template select<12, 1>(788));
              tensorforge::intel_esimd::simd<float, 12> v844_data(ir3.template select<12, 1>(20));
              ir3.template select<12, 1>(20) = (v844_data + v843_data);
              tensorforge::intel_esimd::simd<float, 12> v846_data(r1.template select<12, 1>(1620));
              tensorforge::intel_esimd::simd<float, 12> v847_data(ir3.template select<12, 1>(84));
              ir3.template select<12, 1>(84) = (v847_data + v846_data);
              tensorforge::intel_esimd::simd<float, 12> v849_data(r1.template select<12, 1>(2452));
              tensorforge::intel_esimd::simd<float, 12> v850_data(ir3.template select<12, 1>(148));
              ir3.template select<12, 1>(148) = (v850_data + v849_data);
              tensorforge::intel_esimd::simd<float, 12> v852_data(r1.template select<12, 1>(3284));
              tensorforge::intel_esimd::simd<float, 12> v853_data(ir3.template select<12, 1>(212));
              ir3.template select<12, 1>(212) = (v853_data + v852_data);
              tensorforge::intel_esimd::simd<float, 12> v855_data(r1.template select<12, 1>(4116));
              tensorforge::intel_esimd::simd<float, 12> v856_data(ir3.template select<12, 1>(276));
              ir3.template select<12, 1>(276) = (v856_data + v855_data);
              tensorforge::intel_esimd::simd<float, 12> v858_data(r1.template select<12, 1>(4948));
              tensorforge::intel_esimd::simd<float, 12> v859_data(ir3.template select<12, 1>(340));
              ir3.template select<12, 1>(340) = (v859_data + v858_data);
              tensorforge::intel_esimd::simd<float, 3> v861_data(r1.template select<3, 1>(800));
              tensorforge::intel_esimd::simd<float, 3> v862_data(ir3.template select<3, 1>(32));
              ir3.template select<3, 1>(32) = (v862_data + v861_data);
              tensorforge::intel_esimd::simd<float, 3> v864_data(r1.template select<3, 1>(1632));
              tensorforge::intel_esimd::simd<float, 3> v865_data(ir3.template select<3, 1>(96));
              ir3.template select<3, 1>(96) = (v865_data + v864_data);
              tensorforge::intel_esimd::simd<float, 3> v867_data(r1.template select<3, 1>(2464));
              tensorforge::intel_esimd::simd<float, 3> v868_data(ir3.template select<3, 1>(160));
              ir3.template select<3, 1>(160) = (v868_data + v867_data);
              tensorforge::intel_esimd::simd<float, 3> v870_data(r1.template select<3, 1>(3296));
              tensorforge::intel_esimd::simd<float, 3> v871_data(ir3.template select<3, 1>(224));
              ir3.template select<3, 1>(224) = (v871_data + v870_data);
              tensorforge::intel_esimd::simd<float, 3> v873_data(r1.template select<3, 1>(4128));
              tensorforge::intel_esimd::simd<float, 3> v874_data(ir3.template select<3, 1>(288));
              ir3.template select<3, 1>(288) = (v874_data + v873_data);
              tensorforge::intel_esimd::simd<float, 3> v876_data(r1.template select<3, 1>(4960));
              tensorforge::intel_esimd::simd<float, 3> v877_data(ir3.template select<3, 1>(352));
              ir3.template select<3, 1>(352) = (v877_data + v876_data);
              // r3 = ir3 + r2
              #pragma unroll
              for (int32_t v879_n1 = 0; v879_n1 < 1; ++v879_n1) {
                int32_t v883_a = 20 + (v879_n1 * 64);
                #pragma unroll
                for (int32_t v880_n2 = 0; v880_n2 < 6; ++v880_n2) {
                  int32_t v884_a = v883_a + (v880_n2 * 64);
                  tensorforge::intel_esimd::simd<float, 12> v885_data(ir3.template select<12, 1>(v884_a));
                  tensorforge::intel_esimd::simd<float, 12> v886_data(r2.template select<12, 1>(v884_a));
                  r3.template select<12, 1>(v884_a) = (v886_data + v885_data);
                }
              }
              #pragma unroll
              for (int32_t v888_n1 = 0; v888_n1 < 1; ++v888_n1) {
                int32_t v892_a = 32 + (v888_n1 * 64);
                #pragma unroll
                for (int32_t v889_n2 = 0; v889_n2 < 6; ++v889_n2) {
                  int32_t v893_a = v892_a + (v889_n2 * 64);
                  tensorforge::intel_esimd::simd<float, 3> v894_data(ir3.template select<3, 1>(v893_a));
                  tensorforge::intel_esimd::simd<float, 3> v895_data(r2.template select<3, 1>(v893_a));
                  r3.template select<3, 1>(v893_a) = (v895_data + v894_data);
                }
              }
              // glb_m2 = store{r>g}(r3);
              #pragma unroll
              for (int32_t v897_i1 = 0; v897_i1 < 1; ++v897_i1) {
                int32_t v901_a = 20 + (v897_i1 * 64);
                int32_t v910_a = 20_i32 + ((v897_i1 + 12) * 64);
                #pragma unroll
                for (int32_t v898_i2 = 0; v898_i2 < 6; ++v898_i2) {
                  tensorforge::intel_esimd::simd<float, 12> v903_data(r3.template select<12, 1>((v901_a + (v898_i2 * 64))));
                  v903_data.copy_to(glb_m2 + ((v910_a + (v898_i2 * 832))));
                }
              }
              #pragma unroll
              for (int32_t v912_i1 = 0; v912_i1 < 1; ++v912_i1) {
                int32_t v916_a = 32 + (v912_i1 * 64);
                int32_t v925_a = 32_i32 + ((v912_i1 + 12) * 64);
                #pragma unroll
                for (int32_t v913_i2 = 0; v913_i2 < 6; ++v913_i2) {
                  tensorforge::intel_esimd::simd<float, 3> v918_data(r3.template select<3, 1>((v916_a + (v913_i2 * 64))));
                  v918_data.copy_to(glb_m2 + ((v925_a + (v913_i2 * 832))));
                }
              }
            }
          }
        }
      }
    });
  });
}

