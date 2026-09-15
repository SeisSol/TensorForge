// === base name ===
kernel_755ae40f71941066

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_755ae40f71941066 = {{1, 16, 1}, 16, 10, 1, 16, 22528, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_755ae40f71941066(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_755ae40f71941066(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_755ae40f71941066(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 16, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 5632 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_755ae40f71941066(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_755ae40f71941066(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_755ae40f71941066(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, m3, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_755ae40f71941066(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<5632 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (10 active) x 16 per block = block 1x16x1, 22528 B shared, occupancy grid
        // operands:
        //   m0 10×9(10×9) {0..10}×{0..9} strided
        //   m1 16×20(10×17) {0..10}×{1..18} none
        //   m2 20×9(17×9) {1..18}×{0..9} strided
        //   m3 16×20(10×18) {0..10}×{1..19} none
        //   m4 20×9(18×9) {1..19}×{0..9} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        //   m0[i,j] += m3[i,k] × m4[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":10,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":5632}],"shared_bytes":22528,"shared_elements":5632,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[10,9]],"name":"m0","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"none","alias":"A1","bbox":[[0,1],[10,18]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[1,0],[18,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"none","alias":"A2","bbox":[[0,1],[10,19]],"name":"m3","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[1,0],[19,9]],"name":"m4","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[10,18]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[18,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[10,19]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[19,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (352 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (336);
          const float *const __restrict__ glb_m1 = &m1[0];
          const float *const __restrict__ glb_m3 = &m3[0];
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (160);
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const float *const __restrict__ pf_glb_m2 = &m2[v11_batchId1 * 153 + 0 + m2_extraOffset];
            const float *const __restrict__ pf_glb_m4 = &m4[v11_batchId1 * 162 + 0 + m4_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 90 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 153 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v8_batchId0 * 162 + 0 + m4_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v21_ld;
              v21_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v21_ld);
              tensorforge::intel_esimd::simd<float, 64> v22_ld;
              v22_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v22_ld);
              tensorforge::intel_esimd::simd<float, 16> v23_ld;
              v23_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v23_ld);
              tensorforge::intel_esimd::simd<float, 9> v24_ld;
              v24_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 144));
              tensorforge::slmStore<float, 9>(s0 + (0 + 0 + 1 * 0 + 144), v24_ld);
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v25_ld;
              v25_ld.copy_from(glb_m4 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 4 * 0 + 0), v25_ld);
              tensorforge::intel_esimd::simd<float, 64> v26_ld;
              v26_ld.copy_from(glb_m4 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 4 * 0 + 64), v26_ld);
              tensorforge::intel_esimd::simd<float, 32> v27_ld;
              v27_ld.copy_from(glb_m4 + (0 + 0 + 2 * 0 + 128));
              tensorforge::slmStore<float, 32>(s1 + (0 + 0 + 2 * 0 + 128), v27_ld);
              tensorforge::intel_esimd::simd<float, 2> v28_ld;
              v28_ld.copy_from(glb_m4 + (0 + 0 + 1 * 0 + 160));
              tensorforge::slmStore<float, 2>(s1 + (0 + 0 + 1 * 0 + 160), v28_ld);
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 10), (0, 9)] [(1, 18)]
              tensorforge::intel_esimd::simd<float, 144> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v36_data;
              v36_data.copy_from(glb_m1 + (10_i32));
              tensorforge::intel_esimd::simd<float, 16> v38_data;
              v38_data.copy_from(glb_m1 + (20_i32));
              tensorforge::intel_esimd::simd<float, 16> v40_data;
              v40_data.copy_from(glb_m1 + (30_i32));
              tensorforge::intel_esimd::simd<float, 16> v42_data;
              v42_data.copy_from(glb_m1 + (40_i32));
              tensorforge::intel_esimd::simd<float, 16> v44_data;
              v44_data.copy_from(glb_m1 + (50_i32));
              tensorforge::intel_esimd::simd<float, 16> v46_data;
              v46_data.copy_from(glb_m1 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v48_data;
              v48_data.copy_from(glb_m1 + (70_i32));
              tensorforge::intel_esimd::simd<float, 16> v50_data;
              v50_data.copy_from(glb_m1 + (80_i32));
              tensorforge::intel_esimd::simd<float, 16> v52_data;
              v52_data.copy_from(glb_m1 + (90_i32));
              tensorforge::intel_esimd::simd<float, 16> v54_data;
              v54_data.copy_from(glb_m1 + (100_i32));
              tensorforge::intel_esimd::simd<float, 16> v56_data;
              v56_data.copy_from(glb_m1 + (110_i32));
              tensorforge::intel_esimd::simd<float, 16> v58_data;
              v58_data.copy_from(glb_m1 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v60_data;
              v60_data.copy_from(glb_m1 + (130_i32));
              tensorforge::intel_esimd::simd<float, 16> v62_data;
              v62_data.copy_from(glb_m1 + (140_i32));
              tensorforge::intel_esimd::simd<float, 16> v64_data;
              v64_data.copy_from(glb_m1 + (150_i32));
              tensorforge::intel_esimd::simd<float, 16> v66_data;
              v66_data.copy_from(glb_m1 + (160_i32));
              tensorforge::intel_esimd::simd<float, 16> v68_acc{};
              tensorforge::intel_esimd::simd<float, 16> v71_data(0.0f);
              v71_data.template select<15, 1>(1) = tensorforge::slmLoad<float, 15>((s0 + (-1_i32)) + 1);
              v68_acc += ((static_cast<float>(v71_data[0])) * v34_data);
              v68_acc += ((static_cast<float>(v71_data[1])) * v36_data);
              v68_acc += ((static_cast<float>(v71_data[2])) * v38_data);
              v68_acc += ((static_cast<float>(v71_data[3])) * v40_data);
              v68_acc += ((static_cast<float>(v71_data[4])) * v42_data);
              v68_acc += ((static_cast<float>(v71_data[5])) * v44_data);
              v68_acc += ((static_cast<float>(v71_data[6])) * v46_data);
              v68_acc += ((static_cast<float>(v71_data[7])) * v48_data);
              v68_acc += ((static_cast<float>(v71_data[8])) * v50_data);
              v68_acc += ((static_cast<float>(v71_data[9])) * v52_data);
              v68_acc += ((static_cast<float>(v71_data[10])) * v54_data);
              v68_acc += ((static_cast<float>(v71_data[11])) * v56_data);
              v68_acc += ((static_cast<float>(v71_data[12])) * v58_data);
              v68_acc += ((static_cast<float>(v71_data[13])) * v60_data);
              v68_acc += ((static_cast<float>(v71_data[14])) * v62_data);
              v68_acc += ((static_cast<float>(v71_data[15])) * v64_data);
              tensorforge::intel_esimd::simd<float, 16> v108_data = tensorforge::slmLoad<float, 16>(s0 + (15_i32));
              v68_acc += ((static_cast<float>(v108_data[0])) * v66_data);
              v68_acc += ((static_cast<float>(v108_data[1])) * v34_data);
              ir0.template select<16, 1>(0) = v68_acc;
              tensorforge::intel_esimd::simd<float, 16> v113_acc{};
              tensorforge::intel_esimd::simd<float, 16> v115_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v113_acc += ((static_cast<float>(v115_data[0])) * v34_data);
              v113_acc += ((static_cast<float>(v115_data[1])) * v36_data);
              v113_acc += ((static_cast<float>(v115_data[2])) * v38_data);
              v113_acc += ((static_cast<float>(v115_data[3])) * v40_data);
              v113_acc += ((static_cast<float>(v115_data[4])) * v42_data);
              v113_acc += ((static_cast<float>(v115_data[5])) * v44_data);
              v113_acc += ((static_cast<float>(v115_data[6])) * v46_data);
              v113_acc += ((static_cast<float>(v115_data[7])) * v48_data);
              v113_acc += ((static_cast<float>(v115_data[8])) * v50_data);
              v113_acc += ((static_cast<float>(v115_data[9])) * v52_data);
              v113_acc += ((static_cast<float>(v115_data[10])) * v54_data);
              v113_acc += ((static_cast<float>(v115_data[11])) * v56_data);
              v113_acc += ((static_cast<float>(v115_data[12])) * v58_data);
              v113_acc += ((static_cast<float>(v115_data[13])) * v60_data);
              v113_acc += ((static_cast<float>(v115_data[14])) * v62_data);
              v113_acc += ((static_cast<float>(v115_data[15])) * v64_data);
              tensorforge::intel_esimd::simd<float, 16> v149_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v113_acc += ((static_cast<float>(v149_data[0])) * v66_data);
              v113_acc += ((static_cast<float>(v149_data[1])) * v34_data);
              ir0.template select<16, 1>(16) = v113_acc;
              tensorforge::intel_esimd::simd<float, 16> v154_acc{};
              tensorforge::intel_esimd::simd<float, 16> v156_data = tensorforge::slmLoad<float, 16>(s0 + (33_i32));
              v154_acc += ((static_cast<float>(v156_data[0])) * v34_data);
              v154_acc += ((static_cast<float>(v156_data[1])) * v36_data);
              v154_acc += ((static_cast<float>(v156_data[2])) * v38_data);
              v154_acc += ((static_cast<float>(v156_data[3])) * v40_data);
              v154_acc += ((static_cast<float>(v156_data[4])) * v42_data);
              v154_acc += ((static_cast<float>(v156_data[5])) * v44_data);
              v154_acc += ((static_cast<float>(v156_data[6])) * v46_data);
              v154_acc += ((static_cast<float>(v156_data[7])) * v48_data);
              v154_acc += ((static_cast<float>(v156_data[8])) * v50_data);
              v154_acc += ((static_cast<float>(v156_data[9])) * v52_data);
              v154_acc += ((static_cast<float>(v156_data[10])) * v54_data);
              v154_acc += ((static_cast<float>(v156_data[11])) * v56_data);
              v154_acc += ((static_cast<float>(v156_data[12])) * v58_data);
              v154_acc += ((static_cast<float>(v156_data[13])) * v60_data);
              v154_acc += ((static_cast<float>(v156_data[14])) * v62_data);
              v154_acc += ((static_cast<float>(v156_data[15])) * v64_data);
              tensorforge::intel_esimd::simd<float, 16> v190_data = tensorforge::slmLoad<float, 16>(s0 + (49_i32));
              v154_acc += ((static_cast<float>(v190_data[0])) * v66_data);
              v154_acc += ((static_cast<float>(v190_data[1])) * v34_data);
              ir0.template select<16, 1>(32) = v154_acc;
              tensorforge::intel_esimd::simd<float, 16> v195_acc{};
              tensorforge::intel_esimd::simd<float, 16> v197_data = tensorforge::slmLoad<float, 16>(s0 + (50_i32));
              v195_acc += ((static_cast<float>(v197_data[0])) * v34_data);
              v195_acc += ((static_cast<float>(v197_data[1])) * v36_data);
              v195_acc += ((static_cast<float>(v197_data[2])) * v38_data);
              v195_acc += ((static_cast<float>(v197_data[3])) * v40_data);
              v195_acc += ((static_cast<float>(v197_data[4])) * v42_data);
              v195_acc += ((static_cast<float>(v197_data[5])) * v44_data);
              v195_acc += ((static_cast<float>(v197_data[6])) * v46_data);
              v195_acc += ((static_cast<float>(v197_data[7])) * v48_data);
              v195_acc += ((static_cast<float>(v197_data[8])) * v50_data);
              v195_acc += ((static_cast<float>(v197_data[9])) * v52_data);
              v195_acc += ((static_cast<float>(v197_data[10])) * v54_data);
              v195_acc += ((static_cast<float>(v197_data[11])) * v56_data);
              v195_acc += ((static_cast<float>(v197_data[12])) * v58_data);
              v195_acc += ((static_cast<float>(v197_data[13])) * v60_data);
              v195_acc += ((static_cast<float>(v197_data[14])) * v62_data);
              v195_acc += ((static_cast<float>(v197_data[15])) * v64_data);
              tensorforge::intel_esimd::simd<float, 16> v231_data = tensorforge::slmLoad<float, 16>(s0 + (66_i32));
              v195_acc += ((static_cast<float>(v231_data[0])) * v66_data);
              v195_acc += ((static_cast<float>(v231_data[1])) * v34_data);
              ir0.template select<16, 1>(48) = v195_acc;
              tensorforge::intel_esimd::simd<float, 16> v236_acc{};
              tensorforge::intel_esimd::simd<float, 16> v238_data = tensorforge::slmLoad<float, 16>(s0 + (67_i32));
              v236_acc += ((static_cast<float>(v238_data[0])) * v34_data);
              v236_acc += ((static_cast<float>(v238_data[1])) * v36_data);
              v236_acc += ((static_cast<float>(v238_data[2])) * v38_data);
              v236_acc += ((static_cast<float>(v238_data[3])) * v40_data);
              v236_acc += ((static_cast<float>(v238_data[4])) * v42_data);
              v236_acc += ((static_cast<float>(v238_data[5])) * v44_data);
              v236_acc += ((static_cast<float>(v238_data[6])) * v46_data);
              v236_acc += ((static_cast<float>(v238_data[7])) * v48_data);
              v236_acc += ((static_cast<float>(v238_data[8])) * v50_data);
              v236_acc += ((static_cast<float>(v238_data[9])) * v52_data);
              v236_acc += ((static_cast<float>(v238_data[10])) * v54_data);
              v236_acc += ((static_cast<float>(v238_data[11])) * v56_data);
              v236_acc += ((static_cast<float>(v238_data[12])) * v58_data);
              v236_acc += ((static_cast<float>(v238_data[13])) * v60_data);
              v236_acc += ((static_cast<float>(v238_data[14])) * v62_data);
              v236_acc += ((static_cast<float>(v238_data[15])) * v64_data);
              tensorforge::intel_esimd::simd<float, 16> v272_data = tensorforge::slmLoad<float, 16>(s0 + (83_i32));
              v236_acc += ((static_cast<float>(v272_data[0])) * v66_data);
              v236_acc += ((static_cast<float>(v272_data[1])) * v34_data);
              ir0.template select<16, 1>(64) = v236_acc;
              tensorforge::intel_esimd::simd<float, 16> v277_acc{};
              tensorforge::intel_esimd::simd<float, 16> v279_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v277_acc += ((static_cast<float>(v279_data[0])) * v34_data);
              v277_acc += ((static_cast<float>(v279_data[1])) * v36_data);
              v277_acc += ((static_cast<float>(v279_data[2])) * v38_data);
              v277_acc += ((static_cast<float>(v279_data[3])) * v40_data);
              v277_acc += ((static_cast<float>(v279_data[4])) * v42_data);
              v277_acc += ((static_cast<float>(v279_data[5])) * v44_data);
              v277_acc += ((static_cast<float>(v279_data[6])) * v46_data);
              v277_acc += ((static_cast<float>(v279_data[7])) * v48_data);
              v277_acc += ((static_cast<float>(v279_data[8])) * v50_data);
              v277_acc += ((static_cast<float>(v279_data[9])) * v52_data);
              v277_acc += ((static_cast<float>(v279_data[10])) * v54_data);
              v277_acc += ((static_cast<float>(v279_data[11])) * v56_data);
              v277_acc += ((static_cast<float>(v279_data[12])) * v58_data);
              v277_acc += ((static_cast<float>(v279_data[13])) * v60_data);
              v277_acc += ((static_cast<float>(v279_data[14])) * v62_data);
              v277_acc += ((static_cast<float>(v279_data[15])) * v64_data);
              tensorforge::intel_esimd::simd<float, 16> v313_data = tensorforge::slmLoad<float, 16>(s0 + (100_i32));
              v277_acc += ((static_cast<float>(v313_data[0])) * v66_data);
              v277_acc += ((static_cast<float>(v313_data[1])) * v34_data);
              ir0.template select<16, 1>(80) = v277_acc;
              tensorforge::intel_esimd::simd<float, 16> v318_acc{};
              tensorforge::intel_esimd::simd<float, 16> v320_data = tensorforge::slmLoad<float, 16>(s0 + (101_i32));
              v318_acc += ((static_cast<float>(v320_data[0])) * v34_data);
              v318_acc += ((static_cast<float>(v320_data[1])) * v36_data);
              v318_acc += ((static_cast<float>(v320_data[2])) * v38_data);
              v318_acc += ((static_cast<float>(v320_data[3])) * v40_data);
              v318_acc += ((static_cast<float>(v320_data[4])) * v42_data);
              v318_acc += ((static_cast<float>(v320_data[5])) * v44_data);
              v318_acc += ((static_cast<float>(v320_data[6])) * v46_data);
              v318_acc += ((static_cast<float>(v320_data[7])) * v48_data);
              v318_acc += ((static_cast<float>(v320_data[8])) * v50_data);
              v318_acc += ((static_cast<float>(v320_data[9])) * v52_data);
              v318_acc += ((static_cast<float>(v320_data[10])) * v54_data);
              v318_acc += ((static_cast<float>(v320_data[11])) * v56_data);
              v318_acc += ((static_cast<float>(v320_data[12])) * v58_data);
              v318_acc += ((static_cast<float>(v320_data[13])) * v60_data);
              v318_acc += ((static_cast<float>(v320_data[14])) * v62_data);
              v318_acc += ((static_cast<float>(v320_data[15])) * v64_data);
              tensorforge::intel_esimd::simd<float, 16> v354_data = tensorforge::slmLoad<float, 16>(s0 + (117_i32));
              v318_acc += ((static_cast<float>(v354_data[0])) * v66_data);
              v318_acc += ((static_cast<float>(v354_data[1])) * v34_data);
              ir0.template select<16, 1>(96) = v318_acc;
              tensorforge::intel_esimd::simd<float, 16> v359_acc{};
              tensorforge::intel_esimd::simd<float, 16> v361_data = tensorforge::slmLoad<float, 16>(s0 + (118_i32));
              v359_acc += ((static_cast<float>(v361_data[0])) * v34_data);
              v359_acc += ((static_cast<float>(v361_data[1])) * v36_data);
              v359_acc += ((static_cast<float>(v361_data[2])) * v38_data);
              v359_acc += ((static_cast<float>(v361_data[3])) * v40_data);
              v359_acc += ((static_cast<float>(v361_data[4])) * v42_data);
              v359_acc += ((static_cast<float>(v361_data[5])) * v44_data);
              v359_acc += ((static_cast<float>(v361_data[6])) * v46_data);
              v359_acc += ((static_cast<float>(v361_data[7])) * v48_data);
              v359_acc += ((static_cast<float>(v361_data[8])) * v50_data);
              v359_acc += ((static_cast<float>(v361_data[9])) * v52_data);
              v359_acc += ((static_cast<float>(v361_data[10])) * v54_data);
              v359_acc += ((static_cast<float>(v361_data[11])) * v56_data);
              v359_acc += ((static_cast<float>(v361_data[12])) * v58_data);
              v359_acc += ((static_cast<float>(v361_data[13])) * v60_data);
              v359_acc += ((static_cast<float>(v361_data[14])) * v62_data);
              v359_acc += ((static_cast<float>(v361_data[15])) * v64_data);
              tensorforge::intel_esimd::simd<float, 16> v395_data = tensorforge::slmLoad<float, 16>(s0 + (134_i32));
              v359_acc += ((static_cast<float>(v395_data[0])) * v66_data);
              v359_acc += ((static_cast<float>(v395_data[1])) * v34_data);
              ir0.template select<16, 1>(112) = v359_acc;
              tensorforge::intel_esimd::simd<float, 16> v400_acc{};
              tensorforge::intel_esimd::simd<float, 16> v402_data = tensorforge::slmLoad<float, 16>(s0 + (135_i32));
              v400_acc += ((static_cast<float>(v402_data[0])) * v34_data);
              v400_acc += ((static_cast<float>(v402_data[1])) * v36_data);
              v400_acc += ((static_cast<float>(v402_data[2])) * v38_data);
              v400_acc += ((static_cast<float>(v402_data[3])) * v40_data);
              v400_acc += ((static_cast<float>(v402_data[4])) * v42_data);
              v400_acc += ((static_cast<float>(v402_data[5])) * v44_data);
              v400_acc += ((static_cast<float>(v402_data[6])) * v46_data);
              v400_acc += ((static_cast<float>(v402_data[7])) * v48_data);
              v400_acc += ((static_cast<float>(v402_data[8])) * v50_data);
              v400_acc += ((static_cast<float>(v402_data[9])) * v52_data);
              v400_acc += ((static_cast<float>(v402_data[10])) * v54_data);
              v400_acc += ((static_cast<float>(v402_data[11])) * v56_data);
              v400_acc += ((static_cast<float>(v402_data[12])) * v58_data);
              v400_acc += ((static_cast<float>(v402_data[13])) * v60_data);
              v400_acc += ((static_cast<float>(v402_data[14])) * v62_data);
              v400_acc += ((static_cast<float>(v402_data[15])) * v64_data);
              tensorforge::intel_esimd::simd<float, 16> v436_data = tensorforge::slmLoad<float, 16>(s0 + (151_i32));
              v400_acc += ((static_cast<float>(v436_data[0])) * v66_data);
              v400_acc += ((static_cast<float>(v436_data[1])) * v34_data);
              ir0.template select<16, 1>(128) = v400_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v441_n1 = 0; v441_n1 < 9; ++v441_n1) {
                int32_t v442_a = v441_n1 * 16;
                tensorforge::intel_esimd::simd<float, 10> v444_data(ir0.template select<10, 1>(v442_a));
                r0.template select<10, 1>(v442_a) = v444_data;
              }
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r1(0.0f);
              // ir1 = +(glb_m3 * s1)
              // [(0, 10), (0, 9)] [(1, 19)]
              tensorforge::intel_esimd::simd<float, 144> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v450_data;
              v450_data.copy_from(glb_m3 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v452_data;
              v452_data.copy_from(glb_m3 + (10_i32));
              tensorforge::intel_esimd::simd<float, 16> v454_data;
              v454_data.copy_from(glb_m3 + (20_i32));
              tensorforge::intel_esimd::simd<float, 16> v456_data;
              v456_data.copy_from(glb_m3 + (30_i32));
              tensorforge::intel_esimd::simd<float, 16> v458_data;
              v458_data.copy_from(glb_m3 + (40_i32));
              tensorforge::intel_esimd::simd<float, 16> v460_data;
              v460_data.copy_from(glb_m3 + (50_i32));
              tensorforge::intel_esimd::simd<float, 16> v462_data;
              v462_data.copy_from(glb_m3 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v464_data;
              v464_data.copy_from(glb_m3 + (70_i32));
              tensorforge::intel_esimd::simd<float, 16> v466_data;
              v466_data.copy_from(glb_m3 + (80_i32));
              tensorforge::intel_esimd::simd<float, 16> v468_data;
              v468_data.copy_from(glb_m3 + (90_i32));
              tensorforge::intel_esimd::simd<float, 16> v470_data;
              v470_data.copy_from(glb_m3 + (100_i32));
              tensorforge::intel_esimd::simd<float, 16> v472_data;
              v472_data.copy_from(glb_m3 + (110_i32));
              tensorforge::intel_esimd::simd<float, 16> v474_data;
              v474_data.copy_from(glb_m3 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v476_data;
              v476_data.copy_from(glb_m3 + (130_i32));
              tensorforge::intel_esimd::simd<float, 16> v478_data;
              v478_data.copy_from(glb_m3 + (140_i32));
              tensorforge::intel_esimd::simd<float, 16> v480_data;
              v480_data.copy_from(glb_m3 + (150_i32));
              tensorforge::intel_esimd::simd<float, 16> v482_data;
              v482_data.copy_from(glb_m3 + (160_i32));
              tensorforge::intel_esimd::simd<float, 16> v484_data;
              v484_data.copy_from(glb_m3 + (170_i32));
              tensorforge::intel_esimd::simd<float, 16> v486_acc{};
              tensorforge::intel_esimd::simd<float, 16> v489_data(0.0f);
              v489_data.template select<15, 1>(1) = tensorforge::slmLoad<float, 15>((s1 + (-1_i32)) + 1);
              v486_acc += ((static_cast<float>(v489_data[0])) * v450_data);
              v486_acc += ((static_cast<float>(v489_data[1])) * v452_data);
              v486_acc += ((static_cast<float>(v489_data[2])) * v454_data);
              v486_acc += ((static_cast<float>(v489_data[3])) * v456_data);
              v486_acc += ((static_cast<float>(v489_data[4])) * v458_data);
              v486_acc += ((static_cast<float>(v489_data[5])) * v460_data);
              v486_acc += ((static_cast<float>(v489_data[6])) * v462_data);
              v486_acc += ((static_cast<float>(v489_data[7])) * v464_data);
              v486_acc += ((static_cast<float>(v489_data[8])) * v466_data);
              v486_acc += ((static_cast<float>(v489_data[9])) * v468_data);
              v486_acc += ((static_cast<float>(v489_data[10])) * v470_data);
              v486_acc += ((static_cast<float>(v489_data[11])) * v472_data);
              v486_acc += ((static_cast<float>(v489_data[12])) * v474_data);
              v486_acc += ((static_cast<float>(v489_data[13])) * v476_data);
              v486_acc += ((static_cast<float>(v489_data[14])) * v478_data);
              v486_acc += ((static_cast<float>(v489_data[15])) * v480_data);
              tensorforge::intel_esimd::simd<float, 16> v526_data = tensorforge::slmLoad<float, 16>(s1 + (15_i32));
              v486_acc += ((static_cast<float>(v526_data[0])) * v482_data);
              v486_acc += ((static_cast<float>(v526_data[1])) * v484_data);
              v486_acc += ((static_cast<float>(v526_data[2])) * v450_data);
              ir1.template select<16, 1>(0) = v486_acc;
              tensorforge::intel_esimd::simd<float, 16> v533_acc{};
              tensorforge::intel_esimd::simd<float, 16> v535_data = tensorforge::slmLoad<float, 16>(s1 + (17_i32));
              v533_acc += ((static_cast<float>(v535_data[0])) * v450_data);
              v533_acc += ((static_cast<float>(v535_data[1])) * v452_data);
              v533_acc += ((static_cast<float>(v535_data[2])) * v454_data);
              v533_acc += ((static_cast<float>(v535_data[3])) * v456_data);
              v533_acc += ((static_cast<float>(v535_data[4])) * v458_data);
              v533_acc += ((static_cast<float>(v535_data[5])) * v460_data);
              v533_acc += ((static_cast<float>(v535_data[6])) * v462_data);
              v533_acc += ((static_cast<float>(v535_data[7])) * v464_data);
              v533_acc += ((static_cast<float>(v535_data[8])) * v466_data);
              v533_acc += ((static_cast<float>(v535_data[9])) * v468_data);
              v533_acc += ((static_cast<float>(v535_data[10])) * v470_data);
              v533_acc += ((static_cast<float>(v535_data[11])) * v472_data);
              v533_acc += ((static_cast<float>(v535_data[12])) * v474_data);
              v533_acc += ((static_cast<float>(v535_data[13])) * v476_data);
              v533_acc += ((static_cast<float>(v535_data[14])) * v478_data);
              v533_acc += ((static_cast<float>(v535_data[15])) * v480_data);
              tensorforge::intel_esimd::simd<float, 16> v569_data = tensorforge::slmLoad<float, 16>(s1 + (33_i32));
              v533_acc += ((static_cast<float>(v569_data[0])) * v482_data);
              v533_acc += ((static_cast<float>(v569_data[1])) * v484_data);
              v533_acc += ((static_cast<float>(v569_data[2])) * v450_data);
              ir1.template select<16, 1>(16) = v533_acc;
              tensorforge::intel_esimd::simd<float, 16> v576_acc{};
              tensorforge::intel_esimd::simd<float, 16> v578_data = tensorforge::slmLoad<float, 16>(s1 + (35_i32));
              v576_acc += ((static_cast<float>(v578_data[0])) * v450_data);
              v576_acc += ((static_cast<float>(v578_data[1])) * v452_data);
              v576_acc += ((static_cast<float>(v578_data[2])) * v454_data);
              v576_acc += ((static_cast<float>(v578_data[3])) * v456_data);
              v576_acc += ((static_cast<float>(v578_data[4])) * v458_data);
              v576_acc += ((static_cast<float>(v578_data[5])) * v460_data);
              v576_acc += ((static_cast<float>(v578_data[6])) * v462_data);
              v576_acc += ((static_cast<float>(v578_data[7])) * v464_data);
              v576_acc += ((static_cast<float>(v578_data[8])) * v466_data);
              v576_acc += ((static_cast<float>(v578_data[9])) * v468_data);
              v576_acc += ((static_cast<float>(v578_data[10])) * v470_data);
              v576_acc += ((static_cast<float>(v578_data[11])) * v472_data);
              v576_acc += ((static_cast<float>(v578_data[12])) * v474_data);
              v576_acc += ((static_cast<float>(v578_data[13])) * v476_data);
              v576_acc += ((static_cast<float>(v578_data[14])) * v478_data);
              v576_acc += ((static_cast<float>(v578_data[15])) * v480_data);
              tensorforge::intel_esimd::simd<float, 16> v612_data = tensorforge::slmLoad<float, 16>(s1 + (51_i32));
              v576_acc += ((static_cast<float>(v612_data[0])) * v482_data);
              v576_acc += ((static_cast<float>(v612_data[1])) * v484_data);
              v576_acc += ((static_cast<float>(v612_data[2])) * v450_data);
              ir1.template select<16, 1>(32) = v576_acc;
              tensorforge::intel_esimd::simd<float, 16> v619_acc{};
              tensorforge::intel_esimd::simd<float, 16> v621_data = tensorforge::slmLoad<float, 16>(s1 + (53_i32));
              v619_acc += ((static_cast<float>(v621_data[0])) * v450_data);
              v619_acc += ((static_cast<float>(v621_data[1])) * v452_data);
              v619_acc += ((static_cast<float>(v621_data[2])) * v454_data);
              v619_acc += ((static_cast<float>(v621_data[3])) * v456_data);
              v619_acc += ((static_cast<float>(v621_data[4])) * v458_data);
              v619_acc += ((static_cast<float>(v621_data[5])) * v460_data);
              v619_acc += ((static_cast<float>(v621_data[6])) * v462_data);
              v619_acc += ((static_cast<float>(v621_data[7])) * v464_data);
              v619_acc += ((static_cast<float>(v621_data[8])) * v466_data);
              v619_acc += ((static_cast<float>(v621_data[9])) * v468_data);
              v619_acc += ((static_cast<float>(v621_data[10])) * v470_data);
              v619_acc += ((static_cast<float>(v621_data[11])) * v472_data);
              v619_acc += ((static_cast<float>(v621_data[12])) * v474_data);
              v619_acc += ((static_cast<float>(v621_data[13])) * v476_data);
              v619_acc += ((static_cast<float>(v621_data[14])) * v478_data);
              v619_acc += ((static_cast<float>(v621_data[15])) * v480_data);
              tensorforge::intel_esimd::simd<float, 16> v655_data = tensorforge::slmLoad<float, 16>(s1 + (69_i32));
              v619_acc += ((static_cast<float>(v655_data[0])) * v482_data);
              v619_acc += ((static_cast<float>(v655_data[1])) * v484_data);
              v619_acc += ((static_cast<float>(v655_data[2])) * v450_data);
              ir1.template select<16, 1>(48) = v619_acc;
              tensorforge::intel_esimd::simd<float, 16> v662_acc{};
              tensorforge::intel_esimd::simd<float, 16> v664_data = tensorforge::slmLoad<float, 16>(s1 + (71_i32));
              v662_acc += ((static_cast<float>(v664_data[0])) * v450_data);
              v662_acc += ((static_cast<float>(v664_data[1])) * v452_data);
              v662_acc += ((static_cast<float>(v664_data[2])) * v454_data);
              v662_acc += ((static_cast<float>(v664_data[3])) * v456_data);
              v662_acc += ((static_cast<float>(v664_data[4])) * v458_data);
              v662_acc += ((static_cast<float>(v664_data[5])) * v460_data);
              v662_acc += ((static_cast<float>(v664_data[6])) * v462_data);
              v662_acc += ((static_cast<float>(v664_data[7])) * v464_data);
              v662_acc += ((static_cast<float>(v664_data[8])) * v466_data);
              v662_acc += ((static_cast<float>(v664_data[9])) * v468_data);
              v662_acc += ((static_cast<float>(v664_data[10])) * v470_data);
              v662_acc += ((static_cast<float>(v664_data[11])) * v472_data);
              v662_acc += ((static_cast<float>(v664_data[12])) * v474_data);
              v662_acc += ((static_cast<float>(v664_data[13])) * v476_data);
              v662_acc += ((static_cast<float>(v664_data[14])) * v478_data);
              v662_acc += ((static_cast<float>(v664_data[15])) * v480_data);
              tensorforge::intel_esimd::simd<float, 16> v698_data = tensorforge::slmLoad<float, 16>(s1 + (87_i32));
              v662_acc += ((static_cast<float>(v698_data[0])) * v482_data);
              v662_acc += ((static_cast<float>(v698_data[1])) * v484_data);
              v662_acc += ((static_cast<float>(v698_data[2])) * v450_data);
              ir1.template select<16, 1>(64) = v662_acc;
              tensorforge::intel_esimd::simd<float, 16> v705_acc{};
              tensorforge::intel_esimd::simd<float, 16> v707_data = tensorforge::slmLoad<float, 16>(s1 + (89_i32));
              v705_acc += ((static_cast<float>(v707_data[0])) * v450_data);
              v705_acc += ((static_cast<float>(v707_data[1])) * v452_data);
              v705_acc += ((static_cast<float>(v707_data[2])) * v454_data);
              v705_acc += ((static_cast<float>(v707_data[3])) * v456_data);
              v705_acc += ((static_cast<float>(v707_data[4])) * v458_data);
              v705_acc += ((static_cast<float>(v707_data[5])) * v460_data);
              v705_acc += ((static_cast<float>(v707_data[6])) * v462_data);
              v705_acc += ((static_cast<float>(v707_data[7])) * v464_data);
              v705_acc += ((static_cast<float>(v707_data[8])) * v466_data);
              v705_acc += ((static_cast<float>(v707_data[9])) * v468_data);
              v705_acc += ((static_cast<float>(v707_data[10])) * v470_data);
              v705_acc += ((static_cast<float>(v707_data[11])) * v472_data);
              v705_acc += ((static_cast<float>(v707_data[12])) * v474_data);
              v705_acc += ((static_cast<float>(v707_data[13])) * v476_data);
              v705_acc += ((static_cast<float>(v707_data[14])) * v478_data);
              v705_acc += ((static_cast<float>(v707_data[15])) * v480_data);
              tensorforge::intel_esimd::simd<float, 16> v741_data = tensorforge::slmLoad<float, 16>(s1 + (105_i32));
              v705_acc += ((static_cast<float>(v741_data[0])) * v482_data);
              v705_acc += ((static_cast<float>(v741_data[1])) * v484_data);
              v705_acc += ((static_cast<float>(v741_data[2])) * v450_data);
              ir1.template select<16, 1>(80) = v705_acc;
              tensorforge::intel_esimd::simd<float, 16> v748_acc{};
              tensorforge::intel_esimd::simd<float, 16> v750_data = tensorforge::slmLoad<float, 16>(s1 + (107_i32));
              v748_acc += ((static_cast<float>(v750_data[0])) * v450_data);
              v748_acc += ((static_cast<float>(v750_data[1])) * v452_data);
              v748_acc += ((static_cast<float>(v750_data[2])) * v454_data);
              v748_acc += ((static_cast<float>(v750_data[3])) * v456_data);
              v748_acc += ((static_cast<float>(v750_data[4])) * v458_data);
              v748_acc += ((static_cast<float>(v750_data[5])) * v460_data);
              v748_acc += ((static_cast<float>(v750_data[6])) * v462_data);
              v748_acc += ((static_cast<float>(v750_data[7])) * v464_data);
              v748_acc += ((static_cast<float>(v750_data[8])) * v466_data);
              v748_acc += ((static_cast<float>(v750_data[9])) * v468_data);
              v748_acc += ((static_cast<float>(v750_data[10])) * v470_data);
              v748_acc += ((static_cast<float>(v750_data[11])) * v472_data);
              v748_acc += ((static_cast<float>(v750_data[12])) * v474_data);
              v748_acc += ((static_cast<float>(v750_data[13])) * v476_data);
              v748_acc += ((static_cast<float>(v750_data[14])) * v478_data);
              v748_acc += ((static_cast<float>(v750_data[15])) * v480_data);
              tensorforge::intel_esimd::simd<float, 16> v784_data = tensorforge::slmLoad<float, 16>(s1 + (123_i32));
              v748_acc += ((static_cast<float>(v784_data[0])) * v482_data);
              v748_acc += ((static_cast<float>(v784_data[1])) * v484_data);
              v748_acc += ((static_cast<float>(v784_data[2])) * v450_data);
              ir1.template select<16, 1>(96) = v748_acc;
              tensorforge::intel_esimd::simd<float, 16> v791_acc{};
              tensorforge::intel_esimd::simd<float, 16> v793_data = tensorforge::slmLoad<float, 16>(s1 + (125_i32));
              v791_acc += ((static_cast<float>(v793_data[0])) * v450_data);
              v791_acc += ((static_cast<float>(v793_data[1])) * v452_data);
              v791_acc += ((static_cast<float>(v793_data[2])) * v454_data);
              v791_acc += ((static_cast<float>(v793_data[3])) * v456_data);
              v791_acc += ((static_cast<float>(v793_data[4])) * v458_data);
              v791_acc += ((static_cast<float>(v793_data[5])) * v460_data);
              v791_acc += ((static_cast<float>(v793_data[6])) * v462_data);
              v791_acc += ((static_cast<float>(v793_data[7])) * v464_data);
              v791_acc += ((static_cast<float>(v793_data[8])) * v466_data);
              v791_acc += ((static_cast<float>(v793_data[9])) * v468_data);
              v791_acc += ((static_cast<float>(v793_data[10])) * v470_data);
              v791_acc += ((static_cast<float>(v793_data[11])) * v472_data);
              v791_acc += ((static_cast<float>(v793_data[12])) * v474_data);
              v791_acc += ((static_cast<float>(v793_data[13])) * v476_data);
              v791_acc += ((static_cast<float>(v793_data[14])) * v478_data);
              v791_acc += ((static_cast<float>(v793_data[15])) * v480_data);
              tensorforge::intel_esimd::simd<float, 16> v827_data = tensorforge::slmLoad<float, 16>(s1 + (141_i32));
              v791_acc += ((static_cast<float>(v827_data[0])) * v482_data);
              v791_acc += ((static_cast<float>(v827_data[1])) * v484_data);
              v791_acc += ((static_cast<float>(v827_data[2])) * v450_data);
              ir1.template select<16, 1>(112) = v791_acc;
              tensorforge::intel_esimd::simd<float, 16> v834_acc{};
              tensorforge::intel_esimd::simd<float, 16> v836_data = tensorforge::slmLoad<float, 16>(s1 + (143_i32));
              v834_acc += ((static_cast<float>(v836_data[0])) * v450_data);
              v834_acc += ((static_cast<float>(v836_data[1])) * v452_data);
              v834_acc += ((static_cast<float>(v836_data[2])) * v454_data);
              v834_acc += ((static_cast<float>(v836_data[3])) * v456_data);
              v834_acc += ((static_cast<float>(v836_data[4])) * v458_data);
              v834_acc += ((static_cast<float>(v836_data[5])) * v460_data);
              v834_acc += ((static_cast<float>(v836_data[6])) * v462_data);
              v834_acc += ((static_cast<float>(v836_data[7])) * v464_data);
              v834_acc += ((static_cast<float>(v836_data[8])) * v466_data);
              v834_acc += ((static_cast<float>(v836_data[9])) * v468_data);
              v834_acc += ((static_cast<float>(v836_data[10])) * v470_data);
              v834_acc += ((static_cast<float>(v836_data[11])) * v472_data);
              v834_acc += ((static_cast<float>(v836_data[12])) * v474_data);
              v834_acc += ((static_cast<float>(v836_data[13])) * v476_data);
              v834_acc += ((static_cast<float>(v836_data[14])) * v478_data);
              v834_acc += ((static_cast<float>(v836_data[15])) * v480_data);
              tensorforge::intel_esimd::simd<float, 16> v870_data = tensorforge::slmLoad<float, 16>(s1 + (159_i32));
              v834_acc += ((static_cast<float>(v870_data[0])) * v482_data);
              v834_acc += ((static_cast<float>(v870_data[1])) * v484_data);
              v834_acc += ((static_cast<float>(v870_data[2])) * v450_data);
              ir1.template select<16, 1>(128) = v834_acc;
              // r1 = ir1 + r0
              #pragma unroll
              for (int32_t v877_n1 = 0; v877_n1 < 9; ++v877_n1) {
                int32_t v878_a = v877_n1 * 16;
                tensorforge::intel_esimd::simd<float, 10> v880_data(ir1.template select<10, 1>(v878_a));
                tensorforge::intel_esimd::simd<float, 10> v881_data(r0.template select<10, 1>(v878_a));
                r1.template select<10, 1>(v878_a) = (v881_data + v880_data);
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v883_i1 = 0; v883_i1 < 9; ++v883_i1) {
                tensorforge::intel_esimd::simd<float, 10> v886_data(r1.template select<10, 1>((v883_i1 * 16)));
                v886_data.copy_to(glb_m0 + ((v883_i1 * 10)));
              }
            }
            tensorforge::prefetchRunsL2<612, 648>(&pf_glb_m2[0], &pf_glb_m4[0]);
          }
        }
      }
    });
  });
}

