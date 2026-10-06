// === base name ===
kernel_17b7452ae136e3d3

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_17b7452ae136e3d3 = {{1, 32, 1}, 32, 64, 1, 32, 64, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_17b7452ae136e3d3(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_17b7452ae136e3d3(const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_17b7452ae136e3d3(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_17b7452ae136e3d3(const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_17b7452ae136e3d3(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_17b7452ae136e3d3(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_17b7452ae136e3d3(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (0);
          const float *const __restrict__ ptr_glb_m1 = &m1[0];
          tensorforge::SlmPtr<float> glb_m1 = totalShrMem + (0);
          // glb_m1 = load{g>s}(ptr_glb_m1[0])
          if (item.get_local_id(1) == 0) {
            tensorforge::intel_esimd::simd<float, 6> v12_ld;
            v12_ld.copy_from(ptr_glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0));
            tensorforge::slmStore<float, 6>(glb_m1 + (0 + 0 + 1 * (item.get_local_id(1) * 32) + 0), v12_ld);
          }
          // wait(glb_m1 = load{g>s}(ptr_glb_m1[0]));
          item.barrier();
          for (size_t v13_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v13_batchId0 < numElements0; v13_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v14_ahead1 = v13_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v16_batchId1 = (v14_ahead1 < numElements0) ? v14_ahead1 : v13_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v13_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v13_batchId0][0 + m0_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v13_batchId0][0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 832> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v24_i0 = 0; v24_i0 < 2; ++v24_i0) {
                int32_t v26_lead = v24_i0 * 32;
                #pragma unroll
                for (int32_t v25_i1 = 0; v25_i1 < 13; ++v25_i1) {
                  int32_t v29_a = v26_lead + (v25_i1 * 64);
                  tensorforge::intel_esimd::simd<float, 32> v30_data;
                  v30_data.copy_from(glb_m0 + (v29_a));
                  r0.template select<32, 1>(v29_a) = v30_data;
                }
              }
              tensorforge::intel_esimd::simd<float, 384> r2(0.0f);
              // r2 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v33_i1 = 0; v33_i1 < 1; ++v33_i1) {
                int32_t v41_a = 20_i32 + ((v33_i1 + 12) * 64);
                int32_t v46_a = 20 + (v33_i1 * 64);
                #pragma unroll
                for (int32_t v34_i2 = 0; v34_i2 < 6; ++v34_i2) {
                  tensorforge::intel_esimd::simd<float, 12> v43_data;
                  v43_data.copy_from(glb_m2 + ((v41_a + (v34_i2 * 832))));
                  r2.template select<12, 1>((v46_a + (v34_i2 * 64))) = v43_data;
                }
              }
              #pragma unroll
              for (int32_t v48_i1 = 0; v48_i1 < 1; ++v48_i1) {
                int32_t v56_a = 32_i32 + ((v48_i1 + 12) * 64);
                int32_t v61_a = 32 + (v48_i1 * 64);
                #pragma unroll
                for (int32_t v49_i2 = 0; v49_i2 < 6; ++v49_i2) {
                  tensorforge::intel_esimd::simd<float, 3> v58_data;
                  v58_data.copy_from(glb_m2 + ((v56_a + (v49_i2 * 832))));
                  r2.template select<3, 1>((v61_a + (v49_i2 * 64))) = v58_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<float, 4992> r1(0.0f);
              // r1 = +(r0 * glb_m1) + None
              // [(0, 64), (0, 13), (0, 6)] []
              tensorforge::intel_esimd::simd<float, 32> v64_data(r0.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> glb_m1_w0 = tensorforge::slmLoad<float, 16>(glb_m1 + 0);
              float v65_data = glb_m1_w0[0];
              tensorforge::intel_esimd::simd<float, 32> v67_data(r1.template select<32, 1>(0));
              r1.template select<32, 1>(0) = (v67_data + (v64_data * v65_data));
              float v70_data = glb_m1_w0[1];
              tensorforge::intel_esimd::simd<float, 32> v72_data(r1.template select<32, 1>(832));
              r1.template select<32, 1>(832) = (v72_data + (v64_data * v70_data));
              float v75_data = glb_m1_w0[2];
              tensorforge::intel_esimd::simd<float, 32> v77_data(r1.template select<32, 1>(1664));
              r1.template select<32, 1>(1664) = (v77_data + (v64_data * v75_data));
              float v80_data = glb_m1_w0[3];
              tensorforge::intel_esimd::simd<float, 32> v82_data(r1.template select<32, 1>(2496));
              r1.template select<32, 1>(2496) = (v82_data + (v64_data * v80_data));
              float v85_data = glb_m1_w0[4];
              tensorforge::intel_esimd::simd<float, 32> v87_data(r1.template select<32, 1>(3328));
              r1.template select<32, 1>(3328) = (v87_data + (v64_data * v85_data));
              float v90_data = glb_m1_w0[5];
              tensorforge::intel_esimd::simd<float, 32> v92_data(r1.template select<32, 1>(4160));
              r1.template select<32, 1>(4160) = (v92_data + (v64_data * v90_data));
              tensorforge::intel_esimd::simd<float, 32> v94_data(r0.template select<32, 1>(64));
              tensorforge::intel_esimd::simd<float, 32> v97_data(r1.template select<32, 1>(64));
              r1.template select<32, 1>(64) = (v97_data + (v94_data * v65_data));
              tensorforge::intel_esimd::simd<float, 32> v102_data(r1.template select<32, 1>(896));
              r1.template select<32, 1>(896) = (v102_data + (v94_data * v70_data));
              tensorforge::intel_esimd::simd<float, 32> v107_data(r1.template select<32, 1>(1728));
              r1.template select<32, 1>(1728) = (v107_data + (v94_data * v75_data));
              tensorforge::intel_esimd::simd<float, 32> v112_data(r1.template select<32, 1>(2560));
              r1.template select<32, 1>(2560) = (v112_data + (v94_data * v80_data));
              tensorforge::intel_esimd::simd<float, 32> v117_data(r1.template select<32, 1>(3392));
              r1.template select<32, 1>(3392) = (v117_data + (v94_data * v85_data));
              tensorforge::intel_esimd::simd<float, 32> v122_data(r1.template select<32, 1>(4224));
              r1.template select<32, 1>(4224) = (v122_data + (v94_data * v90_data));
              tensorforge::intel_esimd::simd<float, 32> v124_data(r0.template select<32, 1>(128));
              tensorforge::intel_esimd::simd<float, 32> v127_data(r1.template select<32, 1>(128));
              r1.template select<32, 1>(128) = (v127_data + (v124_data * v65_data));
              tensorforge::intel_esimd::simd<float, 32> v132_data(r1.template select<32, 1>(960));
              r1.template select<32, 1>(960) = (v132_data + (v124_data * v70_data));
              tensorforge::intel_esimd::simd<float, 32> v137_data(r1.template select<32, 1>(1792));
              r1.template select<32, 1>(1792) = (v137_data + (v124_data * v75_data));
              tensorforge::intel_esimd::simd<float, 32> v142_data(r1.template select<32, 1>(2624));
              r1.template select<32, 1>(2624) = (v142_data + (v124_data * v80_data));
              tensorforge::intel_esimd::simd<float, 32> v147_data(r1.template select<32, 1>(3456));
              r1.template select<32, 1>(3456) = (v147_data + (v124_data * v85_data));
              tensorforge::intel_esimd::simd<float, 32> v152_data(r1.template select<32, 1>(4288));
              r1.template select<32, 1>(4288) = (v152_data + (v124_data * v90_data));
              tensorforge::intel_esimd::simd<float, 32> v154_data(r0.template select<32, 1>(192));
              tensorforge::intel_esimd::simd<float, 32> v157_data(r1.template select<32, 1>(192));
              r1.template select<32, 1>(192) = (v157_data + (v154_data * v65_data));
              tensorforge::intel_esimd::simd<float, 32> v162_data(r1.template select<32, 1>(1024));
              r1.template select<32, 1>(1024) = (v162_data + (v154_data * v70_data));
              tensorforge::intel_esimd::simd<float, 32> v167_data(r1.template select<32, 1>(1856));
              r1.template select<32, 1>(1856) = (v167_data + (v154_data * v75_data));
              tensorforge::intel_esimd::simd<float, 32> v172_data(r1.template select<32, 1>(2688));
              r1.template select<32, 1>(2688) = (v172_data + (v154_data * v80_data));
              tensorforge::intel_esimd::simd<float, 32> v177_data(r1.template select<32, 1>(3520));
              r1.template select<32, 1>(3520) = (v177_data + (v154_data * v85_data));
              tensorforge::intel_esimd::simd<float, 32> v182_data(r1.template select<32, 1>(4352));
              r1.template select<32, 1>(4352) = (v182_data + (v154_data * v90_data));
              tensorforge::intel_esimd::simd<float, 32> v184_data(r0.template select<32, 1>(256));
              tensorforge::intel_esimd::simd<float, 32> v187_data(r1.template select<32, 1>(256));
              r1.template select<32, 1>(256) = (v187_data + (v184_data * v65_data));
              tensorforge::intel_esimd::simd<float, 32> v192_data(r1.template select<32, 1>(1088));
              r1.template select<32, 1>(1088) = (v192_data + (v184_data * v70_data));
              tensorforge::intel_esimd::simd<float, 32> v197_data(r1.template select<32, 1>(1920));
              r1.template select<32, 1>(1920) = (v197_data + (v184_data * v75_data));
              tensorforge::intel_esimd::simd<float, 32> v202_data(r1.template select<32, 1>(2752));
              r1.template select<32, 1>(2752) = (v202_data + (v184_data * v80_data));
              tensorforge::intel_esimd::simd<float, 32> v207_data(r1.template select<32, 1>(3584));
              r1.template select<32, 1>(3584) = (v207_data + (v184_data * v85_data));
              tensorforge::intel_esimd::simd<float, 32> v212_data(r1.template select<32, 1>(4416));
              r1.template select<32, 1>(4416) = (v212_data + (v184_data * v90_data));
              tensorforge::intel_esimd::simd<float, 32> v214_data(r0.template select<32, 1>(320));
              tensorforge::intel_esimd::simd<float, 32> v217_data(r1.template select<32, 1>(320));
              r1.template select<32, 1>(320) = (v217_data + (v214_data * v65_data));
              tensorforge::intel_esimd::simd<float, 32> v222_data(r1.template select<32, 1>(1152));
              r1.template select<32, 1>(1152) = (v222_data + (v214_data * v70_data));
              tensorforge::intel_esimd::simd<float, 32> v227_data(r1.template select<32, 1>(1984));
              r1.template select<32, 1>(1984) = (v227_data + (v214_data * v75_data));
              tensorforge::intel_esimd::simd<float, 32> v232_data(r1.template select<32, 1>(2816));
              r1.template select<32, 1>(2816) = (v232_data + (v214_data * v80_data));
              tensorforge::intel_esimd::simd<float, 32> v237_data(r1.template select<32, 1>(3648));
              r1.template select<32, 1>(3648) = (v237_data + (v214_data * v85_data));
              tensorforge::intel_esimd::simd<float, 32> v242_data(r1.template select<32, 1>(4480));
              r1.template select<32, 1>(4480) = (v242_data + (v214_data * v90_data));
              tensorforge::intel_esimd::simd<float, 32> v244_data(r0.template select<32, 1>(384));
              tensorforge::intel_esimd::simd<float, 32> v247_data(r1.template select<32, 1>(384));
              r1.template select<32, 1>(384) = (v247_data + (v244_data * v65_data));
              tensorforge::intel_esimd::simd<float, 32> v252_data(r1.template select<32, 1>(1216));
              r1.template select<32, 1>(1216) = (v252_data + (v244_data * v70_data));
              tensorforge::intel_esimd::simd<float, 32> v257_data(r1.template select<32, 1>(2048));
              r1.template select<32, 1>(2048) = (v257_data + (v244_data * v75_data));
              tensorforge::intel_esimd::simd<float, 32> v262_data(r1.template select<32, 1>(2880));
              r1.template select<32, 1>(2880) = (v262_data + (v244_data * v80_data));
              tensorforge::intel_esimd::simd<float, 32> v267_data(r1.template select<32, 1>(3712));
              r1.template select<32, 1>(3712) = (v267_data + (v244_data * v85_data));
              tensorforge::intel_esimd::simd<float, 32> v272_data(r1.template select<32, 1>(4544));
              r1.template select<32, 1>(4544) = (v272_data + (v244_data * v90_data));
              tensorforge::intel_esimd::simd<float, 32> v274_data(r0.template select<32, 1>(448));
              tensorforge::intel_esimd::simd<float, 32> v277_data(r1.template select<32, 1>(448));
              r1.template select<32, 1>(448) = (v277_data + (v274_data * v65_data));
              tensorforge::intel_esimd::simd<float, 32> v282_data(r1.template select<32, 1>(1280));
              r1.template select<32, 1>(1280) = (v282_data + (v274_data * v70_data));
              tensorforge::intel_esimd::simd<float, 32> v287_data(r1.template select<32, 1>(2112));
              r1.template select<32, 1>(2112) = (v287_data + (v274_data * v75_data));
              tensorforge::intel_esimd::simd<float, 32> v292_data(r1.template select<32, 1>(2944));
              r1.template select<32, 1>(2944) = (v292_data + (v274_data * v80_data));
              tensorforge::intel_esimd::simd<float, 32> v297_data(r1.template select<32, 1>(3776));
              r1.template select<32, 1>(3776) = (v297_data + (v274_data * v85_data));
              tensorforge::intel_esimd::simd<float, 32> v302_data(r1.template select<32, 1>(4608));
              r1.template select<32, 1>(4608) = (v302_data + (v274_data * v90_data));
              tensorforge::intel_esimd::simd<float, 32> v304_data(r0.template select<32, 1>(512));
              tensorforge::intel_esimd::simd<float, 32> v307_data(r1.template select<32, 1>(512));
              r1.template select<32, 1>(512) = (v307_data + (v304_data * v65_data));
              tensorforge::intel_esimd::simd<float, 32> v312_data(r1.template select<32, 1>(1344));
              r1.template select<32, 1>(1344) = (v312_data + (v304_data * v70_data));
              tensorforge::intel_esimd::simd<float, 32> v317_data(r1.template select<32, 1>(2176));
              r1.template select<32, 1>(2176) = (v317_data + (v304_data * v75_data));
              tensorforge::intel_esimd::simd<float, 32> v322_data(r1.template select<32, 1>(3008));
              r1.template select<32, 1>(3008) = (v322_data + (v304_data * v80_data));
              tensorforge::intel_esimd::simd<float, 32> v327_data(r1.template select<32, 1>(3840));
              r1.template select<32, 1>(3840) = (v327_data + (v304_data * v85_data));
              tensorforge::intel_esimd::simd<float, 32> v332_data(r1.template select<32, 1>(4672));
              r1.template select<32, 1>(4672) = (v332_data + (v304_data * v90_data));
              tensorforge::intel_esimd::simd<float, 32> v334_data(r0.template select<32, 1>(576));
              tensorforge::intel_esimd::simd<float, 32> v337_data(r1.template select<32, 1>(576));
              r1.template select<32, 1>(576) = (v337_data + (v334_data * v65_data));
              tensorforge::intel_esimd::simd<float, 32> v342_data(r1.template select<32, 1>(1408));
              r1.template select<32, 1>(1408) = (v342_data + (v334_data * v70_data));
              tensorforge::intel_esimd::simd<float, 32> v347_data(r1.template select<32, 1>(2240));
              r1.template select<32, 1>(2240) = (v347_data + (v334_data * v75_data));
              tensorforge::intel_esimd::simd<float, 32> v352_data(r1.template select<32, 1>(3072));
              r1.template select<32, 1>(3072) = (v352_data + (v334_data * v80_data));
              tensorforge::intel_esimd::simd<float, 32> v357_data(r1.template select<32, 1>(3904));
              r1.template select<32, 1>(3904) = (v357_data + (v334_data * v85_data));
              tensorforge::intel_esimd::simd<float, 32> v362_data(r1.template select<32, 1>(4736));
              r1.template select<32, 1>(4736) = (v362_data + (v334_data * v90_data));
              tensorforge::intel_esimd::simd<float, 32> v364_data(r0.template select<32, 1>(640));
              tensorforge::intel_esimd::simd<float, 32> v367_data(r1.template select<32, 1>(640));
              r1.template select<32, 1>(640) = (v367_data + (v364_data * v65_data));
              tensorforge::intel_esimd::simd<float, 32> v372_data(r1.template select<32, 1>(1472));
              r1.template select<32, 1>(1472) = (v372_data + (v364_data * v70_data));
              tensorforge::intel_esimd::simd<float, 32> v377_data(r1.template select<32, 1>(2304));
              r1.template select<32, 1>(2304) = (v377_data + (v364_data * v75_data));
              tensorforge::intel_esimd::simd<float, 32> v382_data(r1.template select<32, 1>(3136));
              r1.template select<32, 1>(3136) = (v382_data + (v364_data * v80_data));
              tensorforge::intel_esimd::simd<float, 32> v387_data(r1.template select<32, 1>(3968));
              r1.template select<32, 1>(3968) = (v387_data + (v364_data * v85_data));
              tensorforge::intel_esimd::simd<float, 32> v392_data(r1.template select<32, 1>(4800));
              r1.template select<32, 1>(4800) = (v392_data + (v364_data * v90_data));
              tensorforge::intel_esimd::simd<float, 32> v394_data(r0.template select<32, 1>(704));
              tensorforge::intel_esimd::simd<float, 32> v397_data(r1.template select<32, 1>(704));
              r1.template select<32, 1>(704) = (v397_data + (v394_data * v65_data));
              tensorforge::intel_esimd::simd<float, 32> v402_data(r1.template select<32, 1>(1536));
              r1.template select<32, 1>(1536) = (v402_data + (v394_data * v70_data));
              tensorforge::intel_esimd::simd<float, 32> v407_data(r1.template select<32, 1>(2368));
              r1.template select<32, 1>(2368) = (v407_data + (v394_data * v75_data));
              tensorforge::intel_esimd::simd<float, 32> v412_data(r1.template select<32, 1>(3200));
              r1.template select<32, 1>(3200) = (v412_data + (v394_data * v80_data));
              tensorforge::intel_esimd::simd<float, 32> v417_data(r1.template select<32, 1>(4032));
              r1.template select<32, 1>(4032) = (v417_data + (v394_data * v85_data));
              tensorforge::intel_esimd::simd<float, 32> v422_data(r1.template select<32, 1>(4864));
              r1.template select<32, 1>(4864) = (v422_data + (v394_data * v90_data));
              tensorforge::intel_esimd::simd<float, 32> v424_data(r0.template select<32, 1>(768));
              tensorforge::intel_esimd::simd<float, 32> v427_data(r1.template select<32, 1>(768));
              r1.template select<32, 1>(768) = (v427_data + (v424_data * v65_data));
              tensorforge::intel_esimd::simd<float, 32> v432_data(r1.template select<32, 1>(1600));
              r1.template select<32, 1>(1600) = (v432_data + (v424_data * v70_data));
              tensorforge::intel_esimd::simd<float, 32> v437_data(r1.template select<32, 1>(2432));
              r1.template select<32, 1>(2432) = (v437_data + (v424_data * v75_data));
              tensorforge::intel_esimd::simd<float, 32> v442_data(r1.template select<32, 1>(3264));
              r1.template select<32, 1>(3264) = (v442_data + (v424_data * v80_data));
              tensorforge::intel_esimd::simd<float, 32> v447_data(r1.template select<32, 1>(4096));
              r1.template select<32, 1>(4096) = (v447_data + (v424_data * v85_data));
              tensorforge::intel_esimd::simd<float, 32> v452_data(r1.template select<32, 1>(4928));
              r1.template select<32, 1>(4928) = (v452_data + (v424_data * v90_data));
              tensorforge::intel_esimd::simd<float, 32> v454_data(r0.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v457_data(r1.template select<32, 1>(32));
              r1.template select<32, 1>(32) = (v457_data + (v454_data * v65_data));
              tensorforge::intel_esimd::simd<float, 32> v462_data(r1.template select<32, 1>(864));
              r1.template select<32, 1>(864) = (v462_data + (v454_data * v70_data));
              tensorforge::intel_esimd::simd<float, 32> v467_data(r1.template select<32, 1>(1696));
              r1.template select<32, 1>(1696) = (v467_data + (v454_data * v75_data));
              tensorforge::intel_esimd::simd<float, 32> v472_data(r1.template select<32, 1>(2528));
              r1.template select<32, 1>(2528) = (v472_data + (v454_data * v80_data));
              tensorforge::intel_esimd::simd<float, 32> v477_data(r1.template select<32, 1>(3360));
              r1.template select<32, 1>(3360) = (v477_data + (v454_data * v85_data));
              tensorforge::intel_esimd::simd<float, 32> v482_data(r1.template select<32, 1>(4192));
              r1.template select<32, 1>(4192) = (v482_data + (v454_data * v90_data));
              tensorforge::intel_esimd::simd<float, 32> v484_data(r0.template select<32, 1>(96));
              tensorforge::intel_esimd::simd<float, 32> v487_data(r1.template select<32, 1>(96));
              r1.template select<32, 1>(96) = (v487_data + (v484_data * v65_data));
              tensorforge::intel_esimd::simd<float, 32> v492_data(r1.template select<32, 1>(928));
              r1.template select<32, 1>(928) = (v492_data + (v484_data * v70_data));
              tensorforge::intel_esimd::simd<float, 32> v497_data(r1.template select<32, 1>(1760));
              r1.template select<32, 1>(1760) = (v497_data + (v484_data * v75_data));
              tensorforge::intel_esimd::simd<float, 32> v502_data(r1.template select<32, 1>(2592));
              r1.template select<32, 1>(2592) = (v502_data + (v484_data * v80_data));
              tensorforge::intel_esimd::simd<float, 32> v507_data(r1.template select<32, 1>(3424));
              r1.template select<32, 1>(3424) = (v507_data + (v484_data * v85_data));
              tensorforge::intel_esimd::simd<float, 32> v512_data(r1.template select<32, 1>(4256));
              r1.template select<32, 1>(4256) = (v512_data + (v484_data * v90_data));
              tensorforge::intel_esimd::simd<float, 32> v514_data(r0.template select<32, 1>(160));
              tensorforge::intel_esimd::simd<float, 32> v517_data(r1.template select<32, 1>(160));
              r1.template select<32, 1>(160) = (v517_data + (v514_data * v65_data));
              tensorforge::intel_esimd::simd<float, 32> v522_data(r1.template select<32, 1>(992));
              r1.template select<32, 1>(992) = (v522_data + (v514_data * v70_data));
              tensorforge::intel_esimd::simd<float, 32> v527_data(r1.template select<32, 1>(1824));
              r1.template select<32, 1>(1824) = (v527_data + (v514_data * v75_data));
              tensorforge::intel_esimd::simd<float, 32> v532_data(r1.template select<32, 1>(2656));
              r1.template select<32, 1>(2656) = (v532_data + (v514_data * v80_data));
              tensorforge::intel_esimd::simd<float, 32> v537_data(r1.template select<32, 1>(3488));
              r1.template select<32, 1>(3488) = (v537_data + (v514_data * v85_data));
              tensorforge::intel_esimd::simd<float, 32> v542_data(r1.template select<32, 1>(4320));
              r1.template select<32, 1>(4320) = (v542_data + (v514_data * v90_data));
              tensorforge::intel_esimd::simd<float, 32> v544_data(r0.template select<32, 1>(224));
              tensorforge::intel_esimd::simd<float, 32> v547_data(r1.template select<32, 1>(224));
              r1.template select<32, 1>(224) = (v547_data + (v544_data * v65_data));
              tensorforge::intel_esimd::simd<float, 32> v552_data(r1.template select<32, 1>(1056));
              r1.template select<32, 1>(1056) = (v552_data + (v544_data * v70_data));
              tensorforge::intel_esimd::simd<float, 32> v557_data(r1.template select<32, 1>(1888));
              r1.template select<32, 1>(1888) = (v557_data + (v544_data * v75_data));
              tensorforge::intel_esimd::simd<float, 32> v562_data(r1.template select<32, 1>(2720));
              r1.template select<32, 1>(2720) = (v562_data + (v544_data * v80_data));
              tensorforge::intel_esimd::simd<float, 32> v567_data(r1.template select<32, 1>(3552));
              r1.template select<32, 1>(3552) = (v567_data + (v544_data * v85_data));
              tensorforge::intel_esimd::simd<float, 32> v572_data(r1.template select<32, 1>(4384));
              r1.template select<32, 1>(4384) = (v572_data + (v544_data * v90_data));
              tensorforge::intel_esimd::simd<float, 32> v574_data(r0.template select<32, 1>(288));
              tensorforge::intel_esimd::simd<float, 32> v577_data(r1.template select<32, 1>(288));
              r1.template select<32, 1>(288) = (v577_data + (v574_data * v65_data));
              tensorforge::intel_esimd::simd<float, 32> v582_data(r1.template select<32, 1>(1120));
              r1.template select<32, 1>(1120) = (v582_data + (v574_data * v70_data));
              tensorforge::intel_esimd::simd<float, 32> v587_data(r1.template select<32, 1>(1952));
              r1.template select<32, 1>(1952) = (v587_data + (v574_data * v75_data));
              tensorforge::intel_esimd::simd<float, 32> v592_data(r1.template select<32, 1>(2784));
              r1.template select<32, 1>(2784) = (v592_data + (v574_data * v80_data));
              tensorforge::intel_esimd::simd<float, 32> v597_data(r1.template select<32, 1>(3616));
              r1.template select<32, 1>(3616) = (v597_data + (v574_data * v85_data));
              tensorforge::intel_esimd::simd<float, 32> v602_data(r1.template select<32, 1>(4448));
              r1.template select<32, 1>(4448) = (v602_data + (v574_data * v90_data));
              tensorforge::intel_esimd::simd<float, 32> v604_data(r0.template select<32, 1>(352));
              tensorforge::intel_esimd::simd<float, 32> v607_data(r1.template select<32, 1>(352));
              r1.template select<32, 1>(352) = (v607_data + (v604_data * v65_data));
              tensorforge::intel_esimd::simd<float, 32> v612_data(r1.template select<32, 1>(1184));
              r1.template select<32, 1>(1184) = (v612_data + (v604_data * v70_data));
              tensorforge::intel_esimd::simd<float, 32> v617_data(r1.template select<32, 1>(2016));
              r1.template select<32, 1>(2016) = (v617_data + (v604_data * v75_data));
              tensorforge::intel_esimd::simd<float, 32> v622_data(r1.template select<32, 1>(2848));
              r1.template select<32, 1>(2848) = (v622_data + (v604_data * v80_data));
              tensorforge::intel_esimd::simd<float, 32> v627_data(r1.template select<32, 1>(3680));
              r1.template select<32, 1>(3680) = (v627_data + (v604_data * v85_data));
              tensorforge::intel_esimd::simd<float, 32> v632_data(r1.template select<32, 1>(4512));
              r1.template select<32, 1>(4512) = (v632_data + (v604_data * v90_data));
              tensorforge::intel_esimd::simd<float, 32> v634_data(r0.template select<32, 1>(416));
              tensorforge::intel_esimd::simd<float, 32> v637_data(r1.template select<32, 1>(416));
              r1.template select<32, 1>(416) = (v637_data + (v634_data * v65_data));
              tensorforge::intel_esimd::simd<float, 32> v642_data(r1.template select<32, 1>(1248));
              r1.template select<32, 1>(1248) = (v642_data + (v634_data * v70_data));
              tensorforge::intel_esimd::simd<float, 32> v647_data(r1.template select<32, 1>(2080));
              r1.template select<32, 1>(2080) = (v647_data + (v634_data * v75_data));
              tensorforge::intel_esimd::simd<float, 32> v652_data(r1.template select<32, 1>(2912));
              r1.template select<32, 1>(2912) = (v652_data + (v634_data * v80_data));
              tensorforge::intel_esimd::simd<float, 32> v657_data(r1.template select<32, 1>(3744));
              r1.template select<32, 1>(3744) = (v657_data + (v634_data * v85_data));
              tensorforge::intel_esimd::simd<float, 32> v662_data(r1.template select<32, 1>(4576));
              r1.template select<32, 1>(4576) = (v662_data + (v634_data * v90_data));
              tensorforge::intel_esimd::simd<float, 32> v664_data(r0.template select<32, 1>(480));
              tensorforge::intel_esimd::simd<float, 32> v667_data(r1.template select<32, 1>(480));
              r1.template select<32, 1>(480) = (v667_data + (v664_data * v65_data));
              tensorforge::intel_esimd::simd<float, 32> v672_data(r1.template select<32, 1>(1312));
              r1.template select<32, 1>(1312) = (v672_data + (v664_data * v70_data));
              tensorforge::intel_esimd::simd<float, 32> v677_data(r1.template select<32, 1>(2144));
              r1.template select<32, 1>(2144) = (v677_data + (v664_data * v75_data));
              tensorforge::intel_esimd::simd<float, 32> v682_data(r1.template select<32, 1>(2976));
              r1.template select<32, 1>(2976) = (v682_data + (v664_data * v80_data));
              tensorforge::intel_esimd::simd<float, 32> v687_data(r1.template select<32, 1>(3808));
              r1.template select<32, 1>(3808) = (v687_data + (v664_data * v85_data));
              tensorforge::intel_esimd::simd<float, 32> v692_data(r1.template select<32, 1>(4640));
              r1.template select<32, 1>(4640) = (v692_data + (v664_data * v90_data));
              tensorforge::intel_esimd::simd<float, 32> v694_data(r0.template select<32, 1>(544));
              tensorforge::intel_esimd::simd<float, 32> v697_data(r1.template select<32, 1>(544));
              r1.template select<32, 1>(544) = (v697_data + (v694_data * v65_data));
              tensorforge::intel_esimd::simd<float, 32> v702_data(r1.template select<32, 1>(1376));
              r1.template select<32, 1>(1376) = (v702_data + (v694_data * v70_data));
              tensorforge::intel_esimd::simd<float, 32> v707_data(r1.template select<32, 1>(2208));
              r1.template select<32, 1>(2208) = (v707_data + (v694_data * v75_data));
              tensorforge::intel_esimd::simd<float, 32> v712_data(r1.template select<32, 1>(3040));
              r1.template select<32, 1>(3040) = (v712_data + (v694_data * v80_data));
              tensorforge::intel_esimd::simd<float, 32> v717_data(r1.template select<32, 1>(3872));
              r1.template select<32, 1>(3872) = (v717_data + (v694_data * v85_data));
              tensorforge::intel_esimd::simd<float, 32> v722_data(r1.template select<32, 1>(4704));
              r1.template select<32, 1>(4704) = (v722_data + (v694_data * v90_data));
              tensorforge::intel_esimd::simd<float, 32> v724_data(r0.template select<32, 1>(608));
              tensorforge::intel_esimd::simd<float, 32> v727_data(r1.template select<32, 1>(608));
              r1.template select<32, 1>(608) = (v727_data + (v724_data * v65_data));
              tensorforge::intel_esimd::simd<float, 32> v732_data(r1.template select<32, 1>(1440));
              r1.template select<32, 1>(1440) = (v732_data + (v724_data * v70_data));
              tensorforge::intel_esimd::simd<float, 32> v737_data(r1.template select<32, 1>(2272));
              r1.template select<32, 1>(2272) = (v737_data + (v724_data * v75_data));
              tensorforge::intel_esimd::simd<float, 32> v742_data(r1.template select<32, 1>(3104));
              r1.template select<32, 1>(3104) = (v742_data + (v724_data * v80_data));
              tensorforge::intel_esimd::simd<float, 32> v747_data(r1.template select<32, 1>(3936));
              r1.template select<32, 1>(3936) = (v747_data + (v724_data * v85_data));
              tensorforge::intel_esimd::simd<float, 32> v752_data(r1.template select<32, 1>(4768));
              r1.template select<32, 1>(4768) = (v752_data + (v724_data * v90_data));
              tensorforge::intel_esimd::simd<float, 32> v754_data(r0.template select<32, 1>(672));
              tensorforge::intel_esimd::simd<float, 32> v757_data(r1.template select<32, 1>(672));
              r1.template select<32, 1>(672) = (v757_data + (v754_data * v65_data));
              tensorforge::intel_esimd::simd<float, 32> v762_data(r1.template select<32, 1>(1504));
              r1.template select<32, 1>(1504) = (v762_data + (v754_data * v70_data));
              tensorforge::intel_esimd::simd<float, 32> v767_data(r1.template select<32, 1>(2336));
              r1.template select<32, 1>(2336) = (v767_data + (v754_data * v75_data));
              tensorforge::intel_esimd::simd<float, 32> v772_data(r1.template select<32, 1>(3168));
              r1.template select<32, 1>(3168) = (v772_data + (v754_data * v80_data));
              tensorforge::intel_esimd::simd<float, 32> v777_data(r1.template select<32, 1>(4000));
              r1.template select<32, 1>(4000) = (v777_data + (v754_data * v85_data));
              tensorforge::intel_esimd::simd<float, 32> v782_data(r1.template select<32, 1>(4832));
              r1.template select<32, 1>(4832) = (v782_data + (v754_data * v90_data));
              tensorforge::intel_esimd::simd<float, 32> v784_data(r0.template select<32, 1>(736));
              tensorforge::intel_esimd::simd<float, 32> v787_data(r1.template select<32, 1>(736));
              r1.template select<32, 1>(736) = (v787_data + (v784_data * v65_data));
              tensorforge::intel_esimd::simd<float, 32> v792_data(r1.template select<32, 1>(1568));
              r1.template select<32, 1>(1568) = (v792_data + (v784_data * v70_data));
              tensorforge::intel_esimd::simd<float, 32> v797_data(r1.template select<32, 1>(2400));
              r1.template select<32, 1>(2400) = (v797_data + (v784_data * v75_data));
              tensorforge::intel_esimd::simd<float, 32> v802_data(r1.template select<32, 1>(3232));
              r1.template select<32, 1>(3232) = (v802_data + (v784_data * v80_data));
              tensorforge::intel_esimd::simd<float, 32> v807_data(r1.template select<32, 1>(4064));
              r1.template select<32, 1>(4064) = (v807_data + (v784_data * v85_data));
              tensorforge::intel_esimd::simd<float, 32> v812_data(r1.template select<32, 1>(4896));
              r1.template select<32, 1>(4896) = (v812_data + (v784_data * v90_data));
              tensorforge::intel_esimd::simd<float, 32> v814_data(r0.template select<32, 1>(800));
              tensorforge::intel_esimd::simd<float, 32> v817_data(r1.template select<32, 1>(800));
              r1.template select<32, 1>(800) = (v817_data + (v814_data * v65_data));
              tensorforge::intel_esimd::simd<float, 32> v822_data(r1.template select<32, 1>(1632));
              r1.template select<32, 1>(1632) = (v822_data + (v814_data * v70_data));
              tensorforge::intel_esimd::simd<float, 32> v827_data(r1.template select<32, 1>(2464));
              r1.template select<32, 1>(2464) = (v827_data + (v814_data * v75_data));
              tensorforge::intel_esimd::simd<float, 32> v832_data(r1.template select<32, 1>(3296));
              r1.template select<32, 1>(3296) = (v832_data + (v814_data * v80_data));
              tensorforge::intel_esimd::simd<float, 32> v837_data(r1.template select<32, 1>(4128));
              r1.template select<32, 1>(4128) = (v837_data + (v814_data * v85_data));
              tensorforge::intel_esimd::simd<float, 32> v842_data(r1.template select<32, 1>(4960));
              r1.template select<32, 1>(4960) = (v842_data + (v814_data * v90_data));
              // wait(r2 = load{g>r}(glb_m2););
              tensorforge::intel_esimd::simd<float, 384> r3(0.0f);
              // ir3 = +(r1)
              // [(20, 35), (0, 1), (0, 6)] []
              tensorforge::intel_esimd::simd<float, 384> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 12> v846_data(r1.template select<12, 1>(788));
              tensorforge::intel_esimd::simd<float, 12> v847_data(ir3.template select<12, 1>(20));
              ir3.template select<12, 1>(20) = (v847_data + v846_data);
              tensorforge::intel_esimd::simd<float, 12> v849_data(r1.template select<12, 1>(1620));
              tensorforge::intel_esimd::simd<float, 12> v850_data(ir3.template select<12, 1>(84));
              ir3.template select<12, 1>(84) = (v850_data + v849_data);
              tensorforge::intel_esimd::simd<float, 12> v852_data(r1.template select<12, 1>(2452));
              tensorforge::intel_esimd::simd<float, 12> v853_data(ir3.template select<12, 1>(148));
              ir3.template select<12, 1>(148) = (v853_data + v852_data);
              tensorforge::intel_esimd::simd<float, 12> v855_data(r1.template select<12, 1>(3284));
              tensorforge::intel_esimd::simd<float, 12> v856_data(ir3.template select<12, 1>(212));
              ir3.template select<12, 1>(212) = (v856_data + v855_data);
              tensorforge::intel_esimd::simd<float, 12> v858_data(r1.template select<12, 1>(4116));
              tensorforge::intel_esimd::simd<float, 12> v859_data(ir3.template select<12, 1>(276));
              ir3.template select<12, 1>(276) = (v859_data + v858_data);
              tensorforge::intel_esimd::simd<float, 12> v861_data(r1.template select<12, 1>(4948));
              tensorforge::intel_esimd::simd<float, 12> v862_data(ir3.template select<12, 1>(340));
              ir3.template select<12, 1>(340) = (v862_data + v861_data);
              tensorforge::intel_esimd::simd<float, 3> v864_data(r1.template select<3, 1>(800));
              tensorforge::intel_esimd::simd<float, 3> v865_data(ir3.template select<3, 1>(32));
              ir3.template select<3, 1>(32) = (v865_data + v864_data);
              tensorforge::intel_esimd::simd<float, 3> v867_data(r1.template select<3, 1>(1632));
              tensorforge::intel_esimd::simd<float, 3> v868_data(ir3.template select<3, 1>(96));
              ir3.template select<3, 1>(96) = (v868_data + v867_data);
              tensorforge::intel_esimd::simd<float, 3> v870_data(r1.template select<3, 1>(2464));
              tensorforge::intel_esimd::simd<float, 3> v871_data(ir3.template select<3, 1>(160));
              ir3.template select<3, 1>(160) = (v871_data + v870_data);
              tensorforge::intel_esimd::simd<float, 3> v873_data(r1.template select<3, 1>(3296));
              tensorforge::intel_esimd::simd<float, 3> v874_data(ir3.template select<3, 1>(224));
              ir3.template select<3, 1>(224) = (v874_data + v873_data);
              tensorforge::intel_esimd::simd<float, 3> v876_data(r1.template select<3, 1>(4128));
              tensorforge::intel_esimd::simd<float, 3> v877_data(ir3.template select<3, 1>(288));
              ir3.template select<3, 1>(288) = (v877_data + v876_data);
              tensorforge::intel_esimd::simd<float, 3> v879_data(r1.template select<3, 1>(4960));
              tensorforge::intel_esimd::simd<float, 3> v880_data(ir3.template select<3, 1>(352));
              ir3.template select<3, 1>(352) = (v880_data + v879_data);
              // r3 = ir3 + r2
              #pragma unroll
              for (int32_t v882_n1 = 0; v882_n1 < 1; ++v882_n1) {
                int32_t v886_a = 20 + (v882_n1 * 64);
                #pragma unroll
                for (int32_t v883_n2 = 0; v883_n2 < 6; ++v883_n2) {
                  int32_t v887_a = v886_a + (v883_n2 * 64);
                  tensorforge::intel_esimd::simd<float, 12> v888_data(ir3.template select<12, 1>(v887_a));
                  tensorforge::intel_esimd::simd<float, 12> v889_data(r2.template select<12, 1>(v887_a));
                  r3.template select<12, 1>(v887_a) = (v889_data + v888_data);
                }
              }
              #pragma unroll
              for (int32_t v891_n1 = 0; v891_n1 < 1; ++v891_n1) {
                int32_t v895_a = 32 + (v891_n1 * 64);
                #pragma unroll
                for (int32_t v892_n2 = 0; v892_n2 < 6; ++v892_n2) {
                  int32_t v896_a = v895_a + (v892_n2 * 64);
                  tensorforge::intel_esimd::simd<float, 3> v897_data(ir3.template select<3, 1>(v896_a));
                  tensorforge::intel_esimd::simd<float, 3> v898_data(r2.template select<3, 1>(v896_a));
                  r3.template select<3, 1>(v896_a) = (v898_data + v897_data);
                }
              }
              // glb_m2 = store{r>g}(r3);
              #pragma unroll
              for (int32_t v900_i1 = 0; v900_i1 < 1; ++v900_i1) {
                int32_t v904_a = 20 + (v900_i1 * 64);
                int32_t v913_a = 20_i32 + ((v900_i1 + 12) * 64);
                #pragma unroll
                for (int32_t v901_i2 = 0; v901_i2 < 6; ++v901_i2) {
                  tensorforge::intel_esimd::simd<float, 12> v906_data(r3.template select<12, 1>((v904_a + (v901_i2 * 64))));
                  v906_data.copy_to(glb_m2 + ((v913_a + (v901_i2 * 832))));
                }
              }
              #pragma unroll
              for (int32_t v915_i1 = 0; v915_i1 < 1; ++v915_i1) {
                int32_t v919_a = 32 + (v915_i1 * 64);
                int32_t v928_a = 32_i32 + ((v915_i1 + 12) * 64);
                #pragma unroll
                for (int32_t v916_i2 = 0; v916_i2 < 6; ++v916_i2) {
                  tensorforge::intel_esimd::simd<float, 3> v921_data(r3.template select<3, 1>((v919_a + (v916_i2 * 64))));
                  v921_data.copy_to(glb_m2 + ((v928_a + (v916_i2 * 832))));
                }
              }
            }
          }
        }
      }
    });
  });
}

