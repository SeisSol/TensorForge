// === base name ===
kernel_1589805576e88e1e

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_1589805576e88e1e = {{1, 32, 1}, 32, 64, 1, 32, 64, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_1589805576e88e1e(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_1589805576e88e1e(const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_1589805576e88e1e(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_1589805576e88e1e(const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_1589805576e88e1e(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_1589805576e88e1e(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_1589805576e88e1e(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
          // wait(glb_m1 = load{g>s}(ptr_glb_m1[0]));
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
              for (int32_t v30_i1 = 0; v30_i1 < 1; ++v30_i1) {
                int32_t v38_a = 20_i32 + ((v30_i1 + 12) * 64);
                int32_t v43_a = 20 + (v30_i1 * 64);
                #pragma unroll
                for (int32_t v31_i2 = 0; v31_i2 < 6; ++v31_i2) {
                  tensorforge::intel_esimd::simd<float, 12> v40_data;
                  v40_data.copy_from(glb_m2 + ((v38_a + (v31_i2 * 832))));
                  r2.template select<12, 1>((v43_a + (v31_i2 * 64))) = v40_data;
                }
              }
              #pragma unroll
              for (int32_t v45_i1 = 0; v45_i1 < 1; ++v45_i1) {
                int32_t v53_a = 32_i32 + ((v45_i1 + 12) * 64);
                int32_t v58_a = 32 + (v45_i1 * 64);
                #pragma unroll
                for (int32_t v46_i2 = 0; v46_i2 < 6; ++v46_i2) {
                  tensorforge::intel_esimd::simd<float, 3> v55_data;
                  v55_data.copy_from(glb_m2 + ((v53_a + (v46_i2 * 832))));
                  r2.template select<3, 1>((v58_a + (v46_i2 * 64))) = v55_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<float, 4992> r1(0.0f);
              // r1 = +(r0 * glb_m1) + None
              // [(0, 64), (0, 13), (0, 6)] []
              tensorforge::intel_esimd::simd<float, 32> v61_data(r0.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> glb_m1_w0 = tensorforge::slmLoad<float, 16>(glb_m1 + 0);
              float v62_data = glb_m1_w0[0];
              tensorforge::intel_esimd::simd<float, 32> v64_data(r1.template select<32, 1>(0));
              r1.template select<32, 1>(0) = (v64_data + (v61_data * v62_data));
              float v67_data = glb_m1_w0[1];
              tensorforge::intel_esimd::simd<float, 32> v69_data(r1.template select<32, 1>(832));
              r1.template select<32, 1>(832) = (v69_data + (v61_data * v67_data));
              float v72_data = glb_m1_w0[2];
              tensorforge::intel_esimd::simd<float, 32> v74_data(r1.template select<32, 1>(1664));
              r1.template select<32, 1>(1664) = (v74_data + (v61_data * v72_data));
              float v77_data = glb_m1_w0[3];
              tensorforge::intel_esimd::simd<float, 32> v79_data(r1.template select<32, 1>(2496));
              r1.template select<32, 1>(2496) = (v79_data + (v61_data * v77_data));
              float v82_data = glb_m1_w0[4];
              tensorforge::intel_esimd::simd<float, 32> v84_data(r1.template select<32, 1>(3328));
              r1.template select<32, 1>(3328) = (v84_data + (v61_data * v82_data));
              float v87_data = glb_m1_w0[5];
              tensorforge::intel_esimd::simd<float, 32> v89_data(r1.template select<32, 1>(4160));
              r1.template select<32, 1>(4160) = (v89_data + (v61_data * v87_data));
              tensorforge::intel_esimd::simd<float, 32> v91_data(r0.template select<32, 1>(64));
              tensorforge::intel_esimd::simd<float, 32> v94_data(r1.template select<32, 1>(64));
              r1.template select<32, 1>(64) = (v94_data + (v91_data * v62_data));
              tensorforge::intel_esimd::simd<float, 32> v99_data(r1.template select<32, 1>(896));
              r1.template select<32, 1>(896) = (v99_data + (v91_data * v67_data));
              tensorforge::intel_esimd::simd<float, 32> v104_data(r1.template select<32, 1>(1728));
              r1.template select<32, 1>(1728) = (v104_data + (v91_data * v72_data));
              tensorforge::intel_esimd::simd<float, 32> v109_data(r1.template select<32, 1>(2560));
              r1.template select<32, 1>(2560) = (v109_data + (v91_data * v77_data));
              tensorforge::intel_esimd::simd<float, 32> v114_data(r1.template select<32, 1>(3392));
              r1.template select<32, 1>(3392) = (v114_data + (v91_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v119_data(r1.template select<32, 1>(4224));
              r1.template select<32, 1>(4224) = (v119_data + (v91_data * v87_data));
              tensorforge::intel_esimd::simd<float, 32> v121_data(r0.template select<32, 1>(128));
              tensorforge::intel_esimd::simd<float, 32> v124_data(r1.template select<32, 1>(128));
              r1.template select<32, 1>(128) = (v124_data + (v121_data * v62_data));
              tensorforge::intel_esimd::simd<float, 32> v129_data(r1.template select<32, 1>(960));
              r1.template select<32, 1>(960) = (v129_data + (v121_data * v67_data));
              tensorforge::intel_esimd::simd<float, 32> v134_data(r1.template select<32, 1>(1792));
              r1.template select<32, 1>(1792) = (v134_data + (v121_data * v72_data));
              tensorforge::intel_esimd::simd<float, 32> v139_data(r1.template select<32, 1>(2624));
              r1.template select<32, 1>(2624) = (v139_data + (v121_data * v77_data));
              tensorforge::intel_esimd::simd<float, 32> v144_data(r1.template select<32, 1>(3456));
              r1.template select<32, 1>(3456) = (v144_data + (v121_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v149_data(r1.template select<32, 1>(4288));
              r1.template select<32, 1>(4288) = (v149_data + (v121_data * v87_data));
              tensorforge::intel_esimd::simd<float, 32> v151_data(r0.template select<32, 1>(192));
              tensorforge::intel_esimd::simd<float, 32> v154_data(r1.template select<32, 1>(192));
              r1.template select<32, 1>(192) = (v154_data + (v151_data * v62_data));
              tensorforge::intel_esimd::simd<float, 32> v159_data(r1.template select<32, 1>(1024));
              r1.template select<32, 1>(1024) = (v159_data + (v151_data * v67_data));
              tensorforge::intel_esimd::simd<float, 32> v164_data(r1.template select<32, 1>(1856));
              r1.template select<32, 1>(1856) = (v164_data + (v151_data * v72_data));
              tensorforge::intel_esimd::simd<float, 32> v169_data(r1.template select<32, 1>(2688));
              r1.template select<32, 1>(2688) = (v169_data + (v151_data * v77_data));
              tensorforge::intel_esimd::simd<float, 32> v174_data(r1.template select<32, 1>(3520));
              r1.template select<32, 1>(3520) = (v174_data + (v151_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v179_data(r1.template select<32, 1>(4352));
              r1.template select<32, 1>(4352) = (v179_data + (v151_data * v87_data));
              tensorforge::intel_esimd::simd<float, 32> v181_data(r0.template select<32, 1>(256));
              tensorforge::intel_esimd::simd<float, 32> v184_data(r1.template select<32, 1>(256));
              r1.template select<32, 1>(256) = (v184_data + (v181_data * v62_data));
              tensorforge::intel_esimd::simd<float, 32> v189_data(r1.template select<32, 1>(1088));
              r1.template select<32, 1>(1088) = (v189_data + (v181_data * v67_data));
              tensorforge::intel_esimd::simd<float, 32> v194_data(r1.template select<32, 1>(1920));
              r1.template select<32, 1>(1920) = (v194_data + (v181_data * v72_data));
              tensorforge::intel_esimd::simd<float, 32> v199_data(r1.template select<32, 1>(2752));
              r1.template select<32, 1>(2752) = (v199_data + (v181_data * v77_data));
              tensorforge::intel_esimd::simd<float, 32> v204_data(r1.template select<32, 1>(3584));
              r1.template select<32, 1>(3584) = (v204_data + (v181_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v209_data(r1.template select<32, 1>(4416));
              r1.template select<32, 1>(4416) = (v209_data + (v181_data * v87_data));
              tensorforge::intel_esimd::simd<float, 32> v211_data(r0.template select<32, 1>(320));
              tensorforge::intel_esimd::simd<float, 32> v214_data(r1.template select<32, 1>(320));
              r1.template select<32, 1>(320) = (v214_data + (v211_data * v62_data));
              tensorforge::intel_esimd::simd<float, 32> v219_data(r1.template select<32, 1>(1152));
              r1.template select<32, 1>(1152) = (v219_data + (v211_data * v67_data));
              tensorforge::intel_esimd::simd<float, 32> v224_data(r1.template select<32, 1>(1984));
              r1.template select<32, 1>(1984) = (v224_data + (v211_data * v72_data));
              tensorforge::intel_esimd::simd<float, 32> v229_data(r1.template select<32, 1>(2816));
              r1.template select<32, 1>(2816) = (v229_data + (v211_data * v77_data));
              tensorforge::intel_esimd::simd<float, 32> v234_data(r1.template select<32, 1>(3648));
              r1.template select<32, 1>(3648) = (v234_data + (v211_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v239_data(r1.template select<32, 1>(4480));
              r1.template select<32, 1>(4480) = (v239_data + (v211_data * v87_data));
              tensorforge::intel_esimd::simd<float, 32> v241_data(r0.template select<32, 1>(384));
              tensorforge::intel_esimd::simd<float, 32> v244_data(r1.template select<32, 1>(384));
              r1.template select<32, 1>(384) = (v244_data + (v241_data * v62_data));
              tensorforge::intel_esimd::simd<float, 32> v249_data(r1.template select<32, 1>(1216));
              r1.template select<32, 1>(1216) = (v249_data + (v241_data * v67_data));
              tensorforge::intel_esimd::simd<float, 32> v254_data(r1.template select<32, 1>(2048));
              r1.template select<32, 1>(2048) = (v254_data + (v241_data * v72_data));
              tensorforge::intel_esimd::simd<float, 32> v259_data(r1.template select<32, 1>(2880));
              r1.template select<32, 1>(2880) = (v259_data + (v241_data * v77_data));
              tensorforge::intel_esimd::simd<float, 32> v264_data(r1.template select<32, 1>(3712));
              r1.template select<32, 1>(3712) = (v264_data + (v241_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v269_data(r1.template select<32, 1>(4544));
              r1.template select<32, 1>(4544) = (v269_data + (v241_data * v87_data));
              tensorforge::intel_esimd::simd<float, 32> v271_data(r0.template select<32, 1>(448));
              tensorforge::intel_esimd::simd<float, 32> v274_data(r1.template select<32, 1>(448));
              r1.template select<32, 1>(448) = (v274_data + (v271_data * v62_data));
              tensorforge::intel_esimd::simd<float, 32> v279_data(r1.template select<32, 1>(1280));
              r1.template select<32, 1>(1280) = (v279_data + (v271_data * v67_data));
              tensorforge::intel_esimd::simd<float, 32> v284_data(r1.template select<32, 1>(2112));
              r1.template select<32, 1>(2112) = (v284_data + (v271_data * v72_data));
              tensorforge::intel_esimd::simd<float, 32> v289_data(r1.template select<32, 1>(2944));
              r1.template select<32, 1>(2944) = (v289_data + (v271_data * v77_data));
              tensorforge::intel_esimd::simd<float, 32> v294_data(r1.template select<32, 1>(3776));
              r1.template select<32, 1>(3776) = (v294_data + (v271_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v299_data(r1.template select<32, 1>(4608));
              r1.template select<32, 1>(4608) = (v299_data + (v271_data * v87_data));
              tensorforge::intel_esimd::simd<float, 32> v301_data(r0.template select<32, 1>(512));
              tensorforge::intel_esimd::simd<float, 32> v304_data(r1.template select<32, 1>(512));
              r1.template select<32, 1>(512) = (v304_data + (v301_data * v62_data));
              tensorforge::intel_esimd::simd<float, 32> v309_data(r1.template select<32, 1>(1344));
              r1.template select<32, 1>(1344) = (v309_data + (v301_data * v67_data));
              tensorforge::intel_esimd::simd<float, 32> v314_data(r1.template select<32, 1>(2176));
              r1.template select<32, 1>(2176) = (v314_data + (v301_data * v72_data));
              tensorforge::intel_esimd::simd<float, 32> v319_data(r1.template select<32, 1>(3008));
              r1.template select<32, 1>(3008) = (v319_data + (v301_data * v77_data));
              tensorforge::intel_esimd::simd<float, 32> v324_data(r1.template select<32, 1>(3840));
              r1.template select<32, 1>(3840) = (v324_data + (v301_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v329_data(r1.template select<32, 1>(4672));
              r1.template select<32, 1>(4672) = (v329_data + (v301_data * v87_data));
              tensorforge::intel_esimd::simd<float, 32> v331_data(r0.template select<32, 1>(576));
              tensorforge::intel_esimd::simd<float, 32> v334_data(r1.template select<32, 1>(576));
              r1.template select<32, 1>(576) = (v334_data + (v331_data * v62_data));
              tensorforge::intel_esimd::simd<float, 32> v339_data(r1.template select<32, 1>(1408));
              r1.template select<32, 1>(1408) = (v339_data + (v331_data * v67_data));
              tensorforge::intel_esimd::simd<float, 32> v344_data(r1.template select<32, 1>(2240));
              r1.template select<32, 1>(2240) = (v344_data + (v331_data * v72_data));
              tensorforge::intel_esimd::simd<float, 32> v349_data(r1.template select<32, 1>(3072));
              r1.template select<32, 1>(3072) = (v349_data + (v331_data * v77_data));
              tensorforge::intel_esimd::simd<float, 32> v354_data(r1.template select<32, 1>(3904));
              r1.template select<32, 1>(3904) = (v354_data + (v331_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v359_data(r1.template select<32, 1>(4736));
              r1.template select<32, 1>(4736) = (v359_data + (v331_data * v87_data));
              tensorforge::intel_esimd::simd<float, 32> v361_data(r0.template select<32, 1>(640));
              tensorforge::intel_esimd::simd<float, 32> v364_data(r1.template select<32, 1>(640));
              r1.template select<32, 1>(640) = (v364_data + (v361_data * v62_data));
              tensorforge::intel_esimd::simd<float, 32> v369_data(r1.template select<32, 1>(1472));
              r1.template select<32, 1>(1472) = (v369_data + (v361_data * v67_data));
              tensorforge::intel_esimd::simd<float, 32> v374_data(r1.template select<32, 1>(2304));
              r1.template select<32, 1>(2304) = (v374_data + (v361_data * v72_data));
              tensorforge::intel_esimd::simd<float, 32> v379_data(r1.template select<32, 1>(3136));
              r1.template select<32, 1>(3136) = (v379_data + (v361_data * v77_data));
              tensorforge::intel_esimd::simd<float, 32> v384_data(r1.template select<32, 1>(3968));
              r1.template select<32, 1>(3968) = (v384_data + (v361_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v389_data(r1.template select<32, 1>(4800));
              r1.template select<32, 1>(4800) = (v389_data + (v361_data * v87_data));
              tensorforge::intel_esimd::simd<float, 32> v391_data(r0.template select<32, 1>(704));
              tensorforge::intel_esimd::simd<float, 32> v394_data(r1.template select<32, 1>(704));
              r1.template select<32, 1>(704) = (v394_data + (v391_data * v62_data));
              tensorforge::intel_esimd::simd<float, 32> v399_data(r1.template select<32, 1>(1536));
              r1.template select<32, 1>(1536) = (v399_data + (v391_data * v67_data));
              tensorforge::intel_esimd::simd<float, 32> v404_data(r1.template select<32, 1>(2368));
              r1.template select<32, 1>(2368) = (v404_data + (v391_data * v72_data));
              tensorforge::intel_esimd::simd<float, 32> v409_data(r1.template select<32, 1>(3200));
              r1.template select<32, 1>(3200) = (v409_data + (v391_data * v77_data));
              tensorforge::intel_esimd::simd<float, 32> v414_data(r1.template select<32, 1>(4032));
              r1.template select<32, 1>(4032) = (v414_data + (v391_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v419_data(r1.template select<32, 1>(4864));
              r1.template select<32, 1>(4864) = (v419_data + (v391_data * v87_data));
              tensorforge::intel_esimd::simd<float, 32> v421_data(r0.template select<32, 1>(768));
              tensorforge::intel_esimd::simd<float, 32> v424_data(r1.template select<32, 1>(768));
              r1.template select<32, 1>(768) = (v424_data + (v421_data * v62_data));
              tensorforge::intel_esimd::simd<float, 32> v429_data(r1.template select<32, 1>(1600));
              r1.template select<32, 1>(1600) = (v429_data + (v421_data * v67_data));
              tensorforge::intel_esimd::simd<float, 32> v434_data(r1.template select<32, 1>(2432));
              r1.template select<32, 1>(2432) = (v434_data + (v421_data * v72_data));
              tensorforge::intel_esimd::simd<float, 32> v439_data(r1.template select<32, 1>(3264));
              r1.template select<32, 1>(3264) = (v439_data + (v421_data * v77_data));
              tensorforge::intel_esimd::simd<float, 32> v444_data(r1.template select<32, 1>(4096));
              r1.template select<32, 1>(4096) = (v444_data + (v421_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v449_data(r1.template select<32, 1>(4928));
              r1.template select<32, 1>(4928) = (v449_data + (v421_data * v87_data));
              tensorforge::intel_esimd::simd<float, 32> v451_data(r0.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v454_data(r1.template select<32, 1>(32));
              r1.template select<32, 1>(32) = (v454_data + (v451_data * v62_data));
              tensorforge::intel_esimd::simd<float, 32> v459_data(r1.template select<32, 1>(864));
              r1.template select<32, 1>(864) = (v459_data + (v451_data * v67_data));
              tensorforge::intel_esimd::simd<float, 32> v464_data(r1.template select<32, 1>(1696));
              r1.template select<32, 1>(1696) = (v464_data + (v451_data * v72_data));
              tensorforge::intel_esimd::simd<float, 32> v469_data(r1.template select<32, 1>(2528));
              r1.template select<32, 1>(2528) = (v469_data + (v451_data * v77_data));
              tensorforge::intel_esimd::simd<float, 32> v474_data(r1.template select<32, 1>(3360));
              r1.template select<32, 1>(3360) = (v474_data + (v451_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v479_data(r1.template select<32, 1>(4192));
              r1.template select<32, 1>(4192) = (v479_data + (v451_data * v87_data));
              tensorforge::intel_esimd::simd<float, 32> v481_data(r0.template select<32, 1>(96));
              tensorforge::intel_esimd::simd<float, 32> v484_data(r1.template select<32, 1>(96));
              r1.template select<32, 1>(96) = (v484_data + (v481_data * v62_data));
              tensorforge::intel_esimd::simd<float, 32> v489_data(r1.template select<32, 1>(928));
              r1.template select<32, 1>(928) = (v489_data + (v481_data * v67_data));
              tensorforge::intel_esimd::simd<float, 32> v494_data(r1.template select<32, 1>(1760));
              r1.template select<32, 1>(1760) = (v494_data + (v481_data * v72_data));
              tensorforge::intel_esimd::simd<float, 32> v499_data(r1.template select<32, 1>(2592));
              r1.template select<32, 1>(2592) = (v499_data + (v481_data * v77_data));
              tensorforge::intel_esimd::simd<float, 32> v504_data(r1.template select<32, 1>(3424));
              r1.template select<32, 1>(3424) = (v504_data + (v481_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v509_data(r1.template select<32, 1>(4256));
              r1.template select<32, 1>(4256) = (v509_data + (v481_data * v87_data));
              tensorforge::intel_esimd::simd<float, 32> v511_data(r0.template select<32, 1>(160));
              tensorforge::intel_esimd::simd<float, 32> v514_data(r1.template select<32, 1>(160));
              r1.template select<32, 1>(160) = (v514_data + (v511_data * v62_data));
              tensorforge::intel_esimd::simd<float, 32> v519_data(r1.template select<32, 1>(992));
              r1.template select<32, 1>(992) = (v519_data + (v511_data * v67_data));
              tensorforge::intel_esimd::simd<float, 32> v524_data(r1.template select<32, 1>(1824));
              r1.template select<32, 1>(1824) = (v524_data + (v511_data * v72_data));
              tensorforge::intel_esimd::simd<float, 32> v529_data(r1.template select<32, 1>(2656));
              r1.template select<32, 1>(2656) = (v529_data + (v511_data * v77_data));
              tensorforge::intel_esimd::simd<float, 32> v534_data(r1.template select<32, 1>(3488));
              r1.template select<32, 1>(3488) = (v534_data + (v511_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v539_data(r1.template select<32, 1>(4320));
              r1.template select<32, 1>(4320) = (v539_data + (v511_data * v87_data));
              tensorforge::intel_esimd::simd<float, 32> v541_data(r0.template select<32, 1>(224));
              tensorforge::intel_esimd::simd<float, 32> v544_data(r1.template select<32, 1>(224));
              r1.template select<32, 1>(224) = (v544_data + (v541_data * v62_data));
              tensorforge::intel_esimd::simd<float, 32> v549_data(r1.template select<32, 1>(1056));
              r1.template select<32, 1>(1056) = (v549_data + (v541_data * v67_data));
              tensorforge::intel_esimd::simd<float, 32> v554_data(r1.template select<32, 1>(1888));
              r1.template select<32, 1>(1888) = (v554_data + (v541_data * v72_data));
              tensorforge::intel_esimd::simd<float, 32> v559_data(r1.template select<32, 1>(2720));
              r1.template select<32, 1>(2720) = (v559_data + (v541_data * v77_data));
              tensorforge::intel_esimd::simd<float, 32> v564_data(r1.template select<32, 1>(3552));
              r1.template select<32, 1>(3552) = (v564_data + (v541_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v569_data(r1.template select<32, 1>(4384));
              r1.template select<32, 1>(4384) = (v569_data + (v541_data * v87_data));
              tensorforge::intel_esimd::simd<float, 32> v571_data(r0.template select<32, 1>(288));
              tensorforge::intel_esimd::simd<float, 32> v574_data(r1.template select<32, 1>(288));
              r1.template select<32, 1>(288) = (v574_data + (v571_data * v62_data));
              tensorforge::intel_esimd::simd<float, 32> v579_data(r1.template select<32, 1>(1120));
              r1.template select<32, 1>(1120) = (v579_data + (v571_data * v67_data));
              tensorforge::intel_esimd::simd<float, 32> v584_data(r1.template select<32, 1>(1952));
              r1.template select<32, 1>(1952) = (v584_data + (v571_data * v72_data));
              tensorforge::intel_esimd::simd<float, 32> v589_data(r1.template select<32, 1>(2784));
              r1.template select<32, 1>(2784) = (v589_data + (v571_data * v77_data));
              tensorforge::intel_esimd::simd<float, 32> v594_data(r1.template select<32, 1>(3616));
              r1.template select<32, 1>(3616) = (v594_data + (v571_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v599_data(r1.template select<32, 1>(4448));
              r1.template select<32, 1>(4448) = (v599_data + (v571_data * v87_data));
              tensorforge::intel_esimd::simd<float, 32> v601_data(r0.template select<32, 1>(352));
              tensorforge::intel_esimd::simd<float, 32> v604_data(r1.template select<32, 1>(352));
              r1.template select<32, 1>(352) = (v604_data + (v601_data * v62_data));
              tensorforge::intel_esimd::simd<float, 32> v609_data(r1.template select<32, 1>(1184));
              r1.template select<32, 1>(1184) = (v609_data + (v601_data * v67_data));
              tensorforge::intel_esimd::simd<float, 32> v614_data(r1.template select<32, 1>(2016));
              r1.template select<32, 1>(2016) = (v614_data + (v601_data * v72_data));
              tensorforge::intel_esimd::simd<float, 32> v619_data(r1.template select<32, 1>(2848));
              r1.template select<32, 1>(2848) = (v619_data + (v601_data * v77_data));
              tensorforge::intel_esimd::simd<float, 32> v624_data(r1.template select<32, 1>(3680));
              r1.template select<32, 1>(3680) = (v624_data + (v601_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v629_data(r1.template select<32, 1>(4512));
              r1.template select<32, 1>(4512) = (v629_data + (v601_data * v87_data));
              tensorforge::intel_esimd::simd<float, 32> v631_data(r0.template select<32, 1>(416));
              tensorforge::intel_esimd::simd<float, 32> v634_data(r1.template select<32, 1>(416));
              r1.template select<32, 1>(416) = (v634_data + (v631_data * v62_data));
              tensorforge::intel_esimd::simd<float, 32> v639_data(r1.template select<32, 1>(1248));
              r1.template select<32, 1>(1248) = (v639_data + (v631_data * v67_data));
              tensorforge::intel_esimd::simd<float, 32> v644_data(r1.template select<32, 1>(2080));
              r1.template select<32, 1>(2080) = (v644_data + (v631_data * v72_data));
              tensorforge::intel_esimd::simd<float, 32> v649_data(r1.template select<32, 1>(2912));
              r1.template select<32, 1>(2912) = (v649_data + (v631_data * v77_data));
              tensorforge::intel_esimd::simd<float, 32> v654_data(r1.template select<32, 1>(3744));
              r1.template select<32, 1>(3744) = (v654_data + (v631_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v659_data(r1.template select<32, 1>(4576));
              r1.template select<32, 1>(4576) = (v659_data + (v631_data * v87_data));
              tensorforge::intel_esimd::simd<float, 32> v661_data(r0.template select<32, 1>(480));
              tensorforge::intel_esimd::simd<float, 32> v664_data(r1.template select<32, 1>(480));
              r1.template select<32, 1>(480) = (v664_data + (v661_data * v62_data));
              tensorforge::intel_esimd::simd<float, 32> v669_data(r1.template select<32, 1>(1312));
              r1.template select<32, 1>(1312) = (v669_data + (v661_data * v67_data));
              tensorforge::intel_esimd::simd<float, 32> v674_data(r1.template select<32, 1>(2144));
              r1.template select<32, 1>(2144) = (v674_data + (v661_data * v72_data));
              tensorforge::intel_esimd::simd<float, 32> v679_data(r1.template select<32, 1>(2976));
              r1.template select<32, 1>(2976) = (v679_data + (v661_data * v77_data));
              tensorforge::intel_esimd::simd<float, 32> v684_data(r1.template select<32, 1>(3808));
              r1.template select<32, 1>(3808) = (v684_data + (v661_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v689_data(r1.template select<32, 1>(4640));
              r1.template select<32, 1>(4640) = (v689_data + (v661_data * v87_data));
              tensorforge::intel_esimd::simd<float, 32> v691_data(r0.template select<32, 1>(544));
              tensorforge::intel_esimd::simd<float, 32> v694_data(r1.template select<32, 1>(544));
              r1.template select<32, 1>(544) = (v694_data + (v691_data * v62_data));
              tensorforge::intel_esimd::simd<float, 32> v699_data(r1.template select<32, 1>(1376));
              r1.template select<32, 1>(1376) = (v699_data + (v691_data * v67_data));
              tensorforge::intel_esimd::simd<float, 32> v704_data(r1.template select<32, 1>(2208));
              r1.template select<32, 1>(2208) = (v704_data + (v691_data * v72_data));
              tensorforge::intel_esimd::simd<float, 32> v709_data(r1.template select<32, 1>(3040));
              r1.template select<32, 1>(3040) = (v709_data + (v691_data * v77_data));
              tensorforge::intel_esimd::simd<float, 32> v714_data(r1.template select<32, 1>(3872));
              r1.template select<32, 1>(3872) = (v714_data + (v691_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v719_data(r1.template select<32, 1>(4704));
              r1.template select<32, 1>(4704) = (v719_data + (v691_data * v87_data));
              tensorforge::intel_esimd::simd<float, 32> v721_data(r0.template select<32, 1>(608));
              tensorforge::intel_esimd::simd<float, 32> v724_data(r1.template select<32, 1>(608));
              r1.template select<32, 1>(608) = (v724_data + (v721_data * v62_data));
              tensorforge::intel_esimd::simd<float, 32> v729_data(r1.template select<32, 1>(1440));
              r1.template select<32, 1>(1440) = (v729_data + (v721_data * v67_data));
              tensorforge::intel_esimd::simd<float, 32> v734_data(r1.template select<32, 1>(2272));
              r1.template select<32, 1>(2272) = (v734_data + (v721_data * v72_data));
              tensorforge::intel_esimd::simd<float, 32> v739_data(r1.template select<32, 1>(3104));
              r1.template select<32, 1>(3104) = (v739_data + (v721_data * v77_data));
              tensorforge::intel_esimd::simd<float, 32> v744_data(r1.template select<32, 1>(3936));
              r1.template select<32, 1>(3936) = (v744_data + (v721_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v749_data(r1.template select<32, 1>(4768));
              r1.template select<32, 1>(4768) = (v749_data + (v721_data * v87_data));
              tensorforge::intel_esimd::simd<float, 32> v751_data(r0.template select<32, 1>(672));
              tensorforge::intel_esimd::simd<float, 32> v754_data(r1.template select<32, 1>(672));
              r1.template select<32, 1>(672) = (v754_data + (v751_data * v62_data));
              tensorforge::intel_esimd::simd<float, 32> v759_data(r1.template select<32, 1>(1504));
              r1.template select<32, 1>(1504) = (v759_data + (v751_data * v67_data));
              tensorforge::intel_esimd::simd<float, 32> v764_data(r1.template select<32, 1>(2336));
              r1.template select<32, 1>(2336) = (v764_data + (v751_data * v72_data));
              tensorforge::intel_esimd::simd<float, 32> v769_data(r1.template select<32, 1>(3168));
              r1.template select<32, 1>(3168) = (v769_data + (v751_data * v77_data));
              tensorforge::intel_esimd::simd<float, 32> v774_data(r1.template select<32, 1>(4000));
              r1.template select<32, 1>(4000) = (v774_data + (v751_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v779_data(r1.template select<32, 1>(4832));
              r1.template select<32, 1>(4832) = (v779_data + (v751_data * v87_data));
              tensorforge::intel_esimd::simd<float, 32> v781_data(r0.template select<32, 1>(736));
              tensorforge::intel_esimd::simd<float, 32> v784_data(r1.template select<32, 1>(736));
              r1.template select<32, 1>(736) = (v784_data + (v781_data * v62_data));
              tensorforge::intel_esimd::simd<float, 32> v789_data(r1.template select<32, 1>(1568));
              r1.template select<32, 1>(1568) = (v789_data + (v781_data * v67_data));
              tensorforge::intel_esimd::simd<float, 32> v794_data(r1.template select<32, 1>(2400));
              r1.template select<32, 1>(2400) = (v794_data + (v781_data * v72_data));
              tensorforge::intel_esimd::simd<float, 32> v799_data(r1.template select<32, 1>(3232));
              r1.template select<32, 1>(3232) = (v799_data + (v781_data * v77_data));
              tensorforge::intel_esimd::simd<float, 32> v804_data(r1.template select<32, 1>(4064));
              r1.template select<32, 1>(4064) = (v804_data + (v781_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v809_data(r1.template select<32, 1>(4896));
              r1.template select<32, 1>(4896) = (v809_data + (v781_data * v87_data));
              tensorforge::intel_esimd::simd<float, 32> v811_data(r0.template select<32, 1>(800));
              tensorforge::intel_esimd::simd<float, 32> v814_data(r1.template select<32, 1>(800));
              r1.template select<32, 1>(800) = (v814_data + (v811_data * v62_data));
              tensorforge::intel_esimd::simd<float, 32> v819_data(r1.template select<32, 1>(1632));
              r1.template select<32, 1>(1632) = (v819_data + (v811_data * v67_data));
              tensorforge::intel_esimd::simd<float, 32> v824_data(r1.template select<32, 1>(2464));
              r1.template select<32, 1>(2464) = (v824_data + (v811_data * v72_data));
              tensorforge::intel_esimd::simd<float, 32> v829_data(r1.template select<32, 1>(3296));
              r1.template select<32, 1>(3296) = (v829_data + (v811_data * v77_data));
              tensorforge::intel_esimd::simd<float, 32> v834_data(r1.template select<32, 1>(4128));
              r1.template select<32, 1>(4128) = (v834_data + (v811_data * v82_data));
              tensorforge::intel_esimd::simd<float, 32> v839_data(r1.template select<32, 1>(4960));
              r1.template select<32, 1>(4960) = (v839_data + (v811_data * v87_data));
              // wait(r2 = load{g>r}(glb_m2););
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

