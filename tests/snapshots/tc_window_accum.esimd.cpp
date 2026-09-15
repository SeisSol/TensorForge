// === base name ===
kernel_e7f67371f743dae7

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_e7f67371f743dae7 = {{1, 16, 1}, 16, 10, 1, 16, 22528, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_e7f67371f743dae7(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_e7f67371f743dae7(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_e7f67371f743dae7(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_e7f67371f743dae7(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_e7f67371f743dae7(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_e7f67371f743dae7(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, m3, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_e7f67371f743dae7(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
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
              tensorforge::intel_esimd::simd<float, 16> v67_acc{};
              tensorforge::intel_esimd::simd<float, 16> v70_data(0.0f);
              v70_data.template select<15, 1>(1) = tensorforge::slmLoad<float, 15>((s0 + (-1_i32)) + 1);
              v67_acc += ((static_cast<float>(v70_data[1])) * v34_data);
              v67_acc += ((static_cast<float>(v70_data[2])) * v36_data);
              v67_acc += ((static_cast<float>(v70_data[3])) * v38_data);
              v67_acc += ((static_cast<float>(v70_data[4])) * v40_data);
              v67_acc += ((static_cast<float>(v70_data[5])) * v42_data);
              v67_acc += ((static_cast<float>(v70_data[6])) * v44_data);
              v67_acc += ((static_cast<float>(v70_data[7])) * v46_data);
              v67_acc += ((static_cast<float>(v70_data[8])) * v48_data);
              v67_acc += ((static_cast<float>(v70_data[9])) * v50_data);
              v67_acc += ((static_cast<float>(v70_data[10])) * v52_data);
              v67_acc += ((static_cast<float>(v70_data[11])) * v54_data);
              v67_acc += ((static_cast<float>(v70_data[12])) * v56_data);
              v67_acc += ((static_cast<float>(v70_data[13])) * v58_data);
              v67_acc += ((static_cast<float>(v70_data[14])) * v60_data);
              v67_acc += ((static_cast<float>(v70_data[15])) * v62_data);
              tensorforge::intel_esimd::simd<float, 16> v106_data = tensorforge::slmLoad<float, 16>(s0 + (15_i32));
              v67_acc += ((static_cast<float>(v106_data[0])) * v64_data);
              v67_acc += ((static_cast<float>(v106_data[1])) * v66_data);
              ir0.template select<16, 1>(0) = v67_acc;
              tensorforge::intel_esimd::simd<float, 16> v111_acc{};
              tensorforge::intel_esimd::simd<float, 16> v113_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v111_acc += ((static_cast<float>(v113_data[1])) * v34_data);
              v111_acc += ((static_cast<float>(v113_data[2])) * v36_data);
              v111_acc += ((static_cast<float>(v113_data[3])) * v38_data);
              v111_acc += ((static_cast<float>(v113_data[4])) * v40_data);
              v111_acc += ((static_cast<float>(v113_data[5])) * v42_data);
              v111_acc += ((static_cast<float>(v113_data[6])) * v44_data);
              v111_acc += ((static_cast<float>(v113_data[7])) * v46_data);
              v111_acc += ((static_cast<float>(v113_data[8])) * v48_data);
              v111_acc += ((static_cast<float>(v113_data[9])) * v50_data);
              v111_acc += ((static_cast<float>(v113_data[10])) * v52_data);
              v111_acc += ((static_cast<float>(v113_data[11])) * v54_data);
              v111_acc += ((static_cast<float>(v113_data[12])) * v56_data);
              v111_acc += ((static_cast<float>(v113_data[13])) * v58_data);
              v111_acc += ((static_cast<float>(v113_data[14])) * v60_data);
              v111_acc += ((static_cast<float>(v113_data[15])) * v62_data);
              tensorforge::intel_esimd::simd<float, 16> v146_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v111_acc += ((static_cast<float>(v146_data[0])) * v64_data);
              v111_acc += ((static_cast<float>(v146_data[1])) * v66_data);
              ir0.template select<16, 1>(16) = v111_acc;
              tensorforge::intel_esimd::simd<float, 16> v151_acc{};
              tensorforge::intel_esimd::simd<float, 16> v153_data = tensorforge::slmLoad<float, 16>(s0 + (33_i32));
              v151_acc += ((static_cast<float>(v153_data[1])) * v34_data);
              v151_acc += ((static_cast<float>(v153_data[2])) * v36_data);
              v151_acc += ((static_cast<float>(v153_data[3])) * v38_data);
              v151_acc += ((static_cast<float>(v153_data[4])) * v40_data);
              v151_acc += ((static_cast<float>(v153_data[5])) * v42_data);
              v151_acc += ((static_cast<float>(v153_data[6])) * v44_data);
              v151_acc += ((static_cast<float>(v153_data[7])) * v46_data);
              v151_acc += ((static_cast<float>(v153_data[8])) * v48_data);
              v151_acc += ((static_cast<float>(v153_data[9])) * v50_data);
              v151_acc += ((static_cast<float>(v153_data[10])) * v52_data);
              v151_acc += ((static_cast<float>(v153_data[11])) * v54_data);
              v151_acc += ((static_cast<float>(v153_data[12])) * v56_data);
              v151_acc += ((static_cast<float>(v153_data[13])) * v58_data);
              v151_acc += ((static_cast<float>(v153_data[14])) * v60_data);
              v151_acc += ((static_cast<float>(v153_data[15])) * v62_data);
              tensorforge::intel_esimd::simd<float, 16> v186_data = tensorforge::slmLoad<float, 16>(s0 + (49_i32));
              v151_acc += ((static_cast<float>(v186_data[0])) * v64_data);
              v151_acc += ((static_cast<float>(v186_data[1])) * v66_data);
              ir0.template select<16, 1>(32) = v151_acc;
              tensorforge::intel_esimd::simd<float, 16> v191_acc{};
              tensorforge::intel_esimd::simd<float, 16> v193_data = tensorforge::slmLoad<float, 16>(s0 + (50_i32));
              v191_acc += ((static_cast<float>(v193_data[1])) * v34_data);
              v191_acc += ((static_cast<float>(v193_data[2])) * v36_data);
              v191_acc += ((static_cast<float>(v193_data[3])) * v38_data);
              v191_acc += ((static_cast<float>(v193_data[4])) * v40_data);
              v191_acc += ((static_cast<float>(v193_data[5])) * v42_data);
              v191_acc += ((static_cast<float>(v193_data[6])) * v44_data);
              v191_acc += ((static_cast<float>(v193_data[7])) * v46_data);
              v191_acc += ((static_cast<float>(v193_data[8])) * v48_data);
              v191_acc += ((static_cast<float>(v193_data[9])) * v50_data);
              v191_acc += ((static_cast<float>(v193_data[10])) * v52_data);
              v191_acc += ((static_cast<float>(v193_data[11])) * v54_data);
              v191_acc += ((static_cast<float>(v193_data[12])) * v56_data);
              v191_acc += ((static_cast<float>(v193_data[13])) * v58_data);
              v191_acc += ((static_cast<float>(v193_data[14])) * v60_data);
              v191_acc += ((static_cast<float>(v193_data[15])) * v62_data);
              tensorforge::intel_esimd::simd<float, 16> v226_data = tensorforge::slmLoad<float, 16>(s0 + (66_i32));
              v191_acc += ((static_cast<float>(v226_data[0])) * v64_data);
              v191_acc += ((static_cast<float>(v226_data[1])) * v66_data);
              ir0.template select<16, 1>(48) = v191_acc;
              tensorforge::intel_esimd::simd<float, 16> v231_acc{};
              tensorforge::intel_esimd::simd<float, 16> v233_data = tensorforge::slmLoad<float, 16>(s0 + (67_i32));
              v231_acc += ((static_cast<float>(v233_data[1])) * v34_data);
              v231_acc += ((static_cast<float>(v233_data[2])) * v36_data);
              v231_acc += ((static_cast<float>(v233_data[3])) * v38_data);
              v231_acc += ((static_cast<float>(v233_data[4])) * v40_data);
              v231_acc += ((static_cast<float>(v233_data[5])) * v42_data);
              v231_acc += ((static_cast<float>(v233_data[6])) * v44_data);
              v231_acc += ((static_cast<float>(v233_data[7])) * v46_data);
              v231_acc += ((static_cast<float>(v233_data[8])) * v48_data);
              v231_acc += ((static_cast<float>(v233_data[9])) * v50_data);
              v231_acc += ((static_cast<float>(v233_data[10])) * v52_data);
              v231_acc += ((static_cast<float>(v233_data[11])) * v54_data);
              v231_acc += ((static_cast<float>(v233_data[12])) * v56_data);
              v231_acc += ((static_cast<float>(v233_data[13])) * v58_data);
              v231_acc += ((static_cast<float>(v233_data[14])) * v60_data);
              v231_acc += ((static_cast<float>(v233_data[15])) * v62_data);
              tensorforge::intel_esimd::simd<float, 16> v266_data = tensorforge::slmLoad<float, 16>(s0 + (83_i32));
              v231_acc += ((static_cast<float>(v266_data[0])) * v64_data);
              v231_acc += ((static_cast<float>(v266_data[1])) * v66_data);
              ir0.template select<16, 1>(64) = v231_acc;
              tensorforge::intel_esimd::simd<float, 16> v271_acc{};
              tensorforge::intel_esimd::simd<float, 16> v273_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v271_acc += ((static_cast<float>(v273_data[1])) * v34_data);
              v271_acc += ((static_cast<float>(v273_data[2])) * v36_data);
              v271_acc += ((static_cast<float>(v273_data[3])) * v38_data);
              v271_acc += ((static_cast<float>(v273_data[4])) * v40_data);
              v271_acc += ((static_cast<float>(v273_data[5])) * v42_data);
              v271_acc += ((static_cast<float>(v273_data[6])) * v44_data);
              v271_acc += ((static_cast<float>(v273_data[7])) * v46_data);
              v271_acc += ((static_cast<float>(v273_data[8])) * v48_data);
              v271_acc += ((static_cast<float>(v273_data[9])) * v50_data);
              v271_acc += ((static_cast<float>(v273_data[10])) * v52_data);
              v271_acc += ((static_cast<float>(v273_data[11])) * v54_data);
              v271_acc += ((static_cast<float>(v273_data[12])) * v56_data);
              v271_acc += ((static_cast<float>(v273_data[13])) * v58_data);
              v271_acc += ((static_cast<float>(v273_data[14])) * v60_data);
              v271_acc += ((static_cast<float>(v273_data[15])) * v62_data);
              tensorforge::intel_esimd::simd<float, 16> v306_data = tensorforge::slmLoad<float, 16>(s0 + (100_i32));
              v271_acc += ((static_cast<float>(v306_data[0])) * v64_data);
              v271_acc += ((static_cast<float>(v306_data[1])) * v66_data);
              ir0.template select<16, 1>(80) = v271_acc;
              tensorforge::intel_esimd::simd<float, 16> v311_acc{};
              tensorforge::intel_esimd::simd<float, 16> v313_data = tensorforge::slmLoad<float, 16>(s0 + (101_i32));
              v311_acc += ((static_cast<float>(v313_data[1])) * v34_data);
              v311_acc += ((static_cast<float>(v313_data[2])) * v36_data);
              v311_acc += ((static_cast<float>(v313_data[3])) * v38_data);
              v311_acc += ((static_cast<float>(v313_data[4])) * v40_data);
              v311_acc += ((static_cast<float>(v313_data[5])) * v42_data);
              v311_acc += ((static_cast<float>(v313_data[6])) * v44_data);
              v311_acc += ((static_cast<float>(v313_data[7])) * v46_data);
              v311_acc += ((static_cast<float>(v313_data[8])) * v48_data);
              v311_acc += ((static_cast<float>(v313_data[9])) * v50_data);
              v311_acc += ((static_cast<float>(v313_data[10])) * v52_data);
              v311_acc += ((static_cast<float>(v313_data[11])) * v54_data);
              v311_acc += ((static_cast<float>(v313_data[12])) * v56_data);
              v311_acc += ((static_cast<float>(v313_data[13])) * v58_data);
              v311_acc += ((static_cast<float>(v313_data[14])) * v60_data);
              v311_acc += ((static_cast<float>(v313_data[15])) * v62_data);
              tensorforge::intel_esimd::simd<float, 16> v346_data = tensorforge::slmLoad<float, 16>(s0 + (117_i32));
              v311_acc += ((static_cast<float>(v346_data[0])) * v64_data);
              v311_acc += ((static_cast<float>(v346_data[1])) * v66_data);
              ir0.template select<16, 1>(96) = v311_acc;
              tensorforge::intel_esimd::simd<float, 16> v351_acc{};
              tensorforge::intel_esimd::simd<float, 16> v353_data = tensorforge::slmLoad<float, 16>(s0 + (118_i32));
              v351_acc += ((static_cast<float>(v353_data[1])) * v34_data);
              v351_acc += ((static_cast<float>(v353_data[2])) * v36_data);
              v351_acc += ((static_cast<float>(v353_data[3])) * v38_data);
              v351_acc += ((static_cast<float>(v353_data[4])) * v40_data);
              v351_acc += ((static_cast<float>(v353_data[5])) * v42_data);
              v351_acc += ((static_cast<float>(v353_data[6])) * v44_data);
              v351_acc += ((static_cast<float>(v353_data[7])) * v46_data);
              v351_acc += ((static_cast<float>(v353_data[8])) * v48_data);
              v351_acc += ((static_cast<float>(v353_data[9])) * v50_data);
              v351_acc += ((static_cast<float>(v353_data[10])) * v52_data);
              v351_acc += ((static_cast<float>(v353_data[11])) * v54_data);
              v351_acc += ((static_cast<float>(v353_data[12])) * v56_data);
              v351_acc += ((static_cast<float>(v353_data[13])) * v58_data);
              v351_acc += ((static_cast<float>(v353_data[14])) * v60_data);
              v351_acc += ((static_cast<float>(v353_data[15])) * v62_data);
              tensorforge::intel_esimd::simd<float, 16> v386_data = tensorforge::slmLoad<float, 16>(s0 + (134_i32));
              v351_acc += ((static_cast<float>(v386_data[0])) * v64_data);
              v351_acc += ((static_cast<float>(v386_data[1])) * v66_data);
              ir0.template select<16, 1>(112) = v351_acc;
              tensorforge::intel_esimd::simd<float, 16> v391_acc{};
              tensorforge::intel_esimd::simd<float, 16> v393_data = tensorforge::slmLoad<float, 16>(s0 + (135_i32));
              v391_acc += ((static_cast<float>(v393_data[1])) * v34_data);
              v391_acc += ((static_cast<float>(v393_data[2])) * v36_data);
              v391_acc += ((static_cast<float>(v393_data[3])) * v38_data);
              v391_acc += ((static_cast<float>(v393_data[4])) * v40_data);
              v391_acc += ((static_cast<float>(v393_data[5])) * v42_data);
              v391_acc += ((static_cast<float>(v393_data[6])) * v44_data);
              v391_acc += ((static_cast<float>(v393_data[7])) * v46_data);
              v391_acc += ((static_cast<float>(v393_data[8])) * v48_data);
              v391_acc += ((static_cast<float>(v393_data[9])) * v50_data);
              v391_acc += ((static_cast<float>(v393_data[10])) * v52_data);
              v391_acc += ((static_cast<float>(v393_data[11])) * v54_data);
              v391_acc += ((static_cast<float>(v393_data[12])) * v56_data);
              v391_acc += ((static_cast<float>(v393_data[13])) * v58_data);
              v391_acc += ((static_cast<float>(v393_data[14])) * v60_data);
              v391_acc += ((static_cast<float>(v393_data[15])) * v62_data);
              tensorforge::intel_esimd::simd<float, 16> v426_data = tensorforge::slmLoad<float, 16>(s0 + (151_i32));
              v391_acc += ((static_cast<float>(v426_data[0])) * v64_data);
              v391_acc += ((static_cast<float>(v426_data[1])) * v66_data);
              ir0.template select<16, 1>(128) = v391_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v431_n1 = 0; v431_n1 < 9; ++v431_n1) {
                int32_t v432_a = v431_n1 * 16;
                tensorforge::intel_esimd::simd<float, 10> v434_data(ir0.template select<10, 1>(v432_a));
                r0.template select<10, 1>(v432_a) = v434_data;
              }
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r1(0.0f);
              // ir1 = +(glb_m3 * s1)
              // [(0, 10), (0, 9)] [(1, 19)]
              tensorforge::intel_esimd::simd<float, 144> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v440_data;
              v440_data.copy_from(glb_m3 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v442_data;
              v442_data.copy_from(glb_m3 + (10_i32));
              tensorforge::intel_esimd::simd<float, 16> v444_data;
              v444_data.copy_from(glb_m3 + (20_i32));
              tensorforge::intel_esimd::simd<float, 16> v446_data;
              v446_data.copy_from(glb_m3 + (30_i32));
              tensorforge::intel_esimd::simd<float, 16> v448_data;
              v448_data.copy_from(glb_m3 + (40_i32));
              tensorforge::intel_esimd::simd<float, 16> v450_data;
              v450_data.copy_from(glb_m3 + (50_i32));
              tensorforge::intel_esimd::simd<float, 16> v452_data;
              v452_data.copy_from(glb_m3 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v454_data;
              v454_data.copy_from(glb_m3 + (70_i32));
              tensorforge::intel_esimd::simd<float, 16> v456_data;
              v456_data.copy_from(glb_m3 + (80_i32));
              tensorforge::intel_esimd::simd<float, 16> v458_data;
              v458_data.copy_from(glb_m3 + (90_i32));
              tensorforge::intel_esimd::simd<float, 16> v460_data;
              v460_data.copy_from(glb_m3 + (100_i32));
              tensorforge::intel_esimd::simd<float, 16> v462_data;
              v462_data.copy_from(glb_m3 + (110_i32));
              tensorforge::intel_esimd::simd<float, 16> v464_data;
              v464_data.copy_from(glb_m3 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v466_data;
              v466_data.copy_from(glb_m3 + (130_i32));
              tensorforge::intel_esimd::simd<float, 16> v468_data;
              v468_data.copy_from(glb_m3 + (140_i32));
              tensorforge::intel_esimd::simd<float, 16> v470_data;
              v470_data.copy_from(glb_m3 + (150_i32));
              tensorforge::intel_esimd::simd<float, 16> v472_data;
              v472_data.copy_from(glb_m3 + (160_i32));
              tensorforge::intel_esimd::simd<float, 16> v474_data;
              v474_data.copy_from(glb_m3 + (170_i32));
              tensorforge::intel_esimd::simd<float, 16> v475_acc{};
              tensorforge::intel_esimd::simd<float, 16> v478_data(0.0f);
              v478_data.template select<15, 1>(1) = tensorforge::slmLoad<float, 15>((s1 + (-1_i32)) + 1);
              v475_acc += ((static_cast<float>(v478_data[1])) * v440_data);
              v475_acc += ((static_cast<float>(v478_data[2])) * v442_data);
              v475_acc += ((static_cast<float>(v478_data[3])) * v444_data);
              v475_acc += ((static_cast<float>(v478_data[4])) * v446_data);
              v475_acc += ((static_cast<float>(v478_data[5])) * v448_data);
              v475_acc += ((static_cast<float>(v478_data[6])) * v450_data);
              v475_acc += ((static_cast<float>(v478_data[7])) * v452_data);
              v475_acc += ((static_cast<float>(v478_data[8])) * v454_data);
              v475_acc += ((static_cast<float>(v478_data[9])) * v456_data);
              v475_acc += ((static_cast<float>(v478_data[10])) * v458_data);
              v475_acc += ((static_cast<float>(v478_data[11])) * v460_data);
              v475_acc += ((static_cast<float>(v478_data[12])) * v462_data);
              v475_acc += ((static_cast<float>(v478_data[13])) * v464_data);
              v475_acc += ((static_cast<float>(v478_data[14])) * v466_data);
              v475_acc += ((static_cast<float>(v478_data[15])) * v468_data);
              tensorforge::intel_esimd::simd<float, 16> v514_data = tensorforge::slmLoad<float, 16>(s1 + (15_i32));
              v475_acc += ((static_cast<float>(v514_data[0])) * v470_data);
              v475_acc += ((static_cast<float>(v514_data[1])) * v472_data);
              v475_acc += ((static_cast<float>(v514_data[2])) * v474_data);
              ir1.template select<16, 1>(0) = v475_acc;
              tensorforge::intel_esimd::simd<float, 16> v521_acc{};
              tensorforge::intel_esimd::simd<float, 16> v523_data = tensorforge::slmLoad<float, 16>(s1 + (17_i32));
              v521_acc += ((static_cast<float>(v523_data[1])) * v440_data);
              v521_acc += ((static_cast<float>(v523_data[2])) * v442_data);
              v521_acc += ((static_cast<float>(v523_data[3])) * v444_data);
              v521_acc += ((static_cast<float>(v523_data[4])) * v446_data);
              v521_acc += ((static_cast<float>(v523_data[5])) * v448_data);
              v521_acc += ((static_cast<float>(v523_data[6])) * v450_data);
              v521_acc += ((static_cast<float>(v523_data[7])) * v452_data);
              v521_acc += ((static_cast<float>(v523_data[8])) * v454_data);
              v521_acc += ((static_cast<float>(v523_data[9])) * v456_data);
              v521_acc += ((static_cast<float>(v523_data[10])) * v458_data);
              v521_acc += ((static_cast<float>(v523_data[11])) * v460_data);
              v521_acc += ((static_cast<float>(v523_data[12])) * v462_data);
              v521_acc += ((static_cast<float>(v523_data[13])) * v464_data);
              v521_acc += ((static_cast<float>(v523_data[14])) * v466_data);
              v521_acc += ((static_cast<float>(v523_data[15])) * v468_data);
              tensorforge::intel_esimd::simd<float, 16> v556_data = tensorforge::slmLoad<float, 16>(s1 + (33_i32));
              v521_acc += ((static_cast<float>(v556_data[0])) * v470_data);
              v521_acc += ((static_cast<float>(v556_data[1])) * v472_data);
              v521_acc += ((static_cast<float>(v556_data[2])) * v474_data);
              ir1.template select<16, 1>(16) = v521_acc;
              tensorforge::intel_esimd::simd<float, 16> v563_acc{};
              tensorforge::intel_esimd::simd<float, 16> v565_data = tensorforge::slmLoad<float, 16>(s1 + (35_i32));
              v563_acc += ((static_cast<float>(v565_data[1])) * v440_data);
              v563_acc += ((static_cast<float>(v565_data[2])) * v442_data);
              v563_acc += ((static_cast<float>(v565_data[3])) * v444_data);
              v563_acc += ((static_cast<float>(v565_data[4])) * v446_data);
              v563_acc += ((static_cast<float>(v565_data[5])) * v448_data);
              v563_acc += ((static_cast<float>(v565_data[6])) * v450_data);
              v563_acc += ((static_cast<float>(v565_data[7])) * v452_data);
              v563_acc += ((static_cast<float>(v565_data[8])) * v454_data);
              v563_acc += ((static_cast<float>(v565_data[9])) * v456_data);
              v563_acc += ((static_cast<float>(v565_data[10])) * v458_data);
              v563_acc += ((static_cast<float>(v565_data[11])) * v460_data);
              v563_acc += ((static_cast<float>(v565_data[12])) * v462_data);
              v563_acc += ((static_cast<float>(v565_data[13])) * v464_data);
              v563_acc += ((static_cast<float>(v565_data[14])) * v466_data);
              v563_acc += ((static_cast<float>(v565_data[15])) * v468_data);
              tensorforge::intel_esimd::simd<float, 16> v598_data = tensorforge::slmLoad<float, 16>(s1 + (51_i32));
              v563_acc += ((static_cast<float>(v598_data[0])) * v470_data);
              v563_acc += ((static_cast<float>(v598_data[1])) * v472_data);
              v563_acc += ((static_cast<float>(v598_data[2])) * v474_data);
              ir1.template select<16, 1>(32) = v563_acc;
              tensorforge::intel_esimd::simd<float, 16> v605_acc{};
              tensorforge::intel_esimd::simd<float, 16> v607_data = tensorforge::slmLoad<float, 16>(s1 + (53_i32));
              v605_acc += ((static_cast<float>(v607_data[1])) * v440_data);
              v605_acc += ((static_cast<float>(v607_data[2])) * v442_data);
              v605_acc += ((static_cast<float>(v607_data[3])) * v444_data);
              v605_acc += ((static_cast<float>(v607_data[4])) * v446_data);
              v605_acc += ((static_cast<float>(v607_data[5])) * v448_data);
              v605_acc += ((static_cast<float>(v607_data[6])) * v450_data);
              v605_acc += ((static_cast<float>(v607_data[7])) * v452_data);
              v605_acc += ((static_cast<float>(v607_data[8])) * v454_data);
              v605_acc += ((static_cast<float>(v607_data[9])) * v456_data);
              v605_acc += ((static_cast<float>(v607_data[10])) * v458_data);
              v605_acc += ((static_cast<float>(v607_data[11])) * v460_data);
              v605_acc += ((static_cast<float>(v607_data[12])) * v462_data);
              v605_acc += ((static_cast<float>(v607_data[13])) * v464_data);
              v605_acc += ((static_cast<float>(v607_data[14])) * v466_data);
              v605_acc += ((static_cast<float>(v607_data[15])) * v468_data);
              tensorforge::intel_esimd::simd<float, 16> v640_data = tensorforge::slmLoad<float, 16>(s1 + (69_i32));
              v605_acc += ((static_cast<float>(v640_data[0])) * v470_data);
              v605_acc += ((static_cast<float>(v640_data[1])) * v472_data);
              v605_acc += ((static_cast<float>(v640_data[2])) * v474_data);
              ir1.template select<16, 1>(48) = v605_acc;
              tensorforge::intel_esimd::simd<float, 16> v647_acc{};
              tensorforge::intel_esimd::simd<float, 16> v649_data = tensorforge::slmLoad<float, 16>(s1 + (71_i32));
              v647_acc += ((static_cast<float>(v649_data[1])) * v440_data);
              v647_acc += ((static_cast<float>(v649_data[2])) * v442_data);
              v647_acc += ((static_cast<float>(v649_data[3])) * v444_data);
              v647_acc += ((static_cast<float>(v649_data[4])) * v446_data);
              v647_acc += ((static_cast<float>(v649_data[5])) * v448_data);
              v647_acc += ((static_cast<float>(v649_data[6])) * v450_data);
              v647_acc += ((static_cast<float>(v649_data[7])) * v452_data);
              v647_acc += ((static_cast<float>(v649_data[8])) * v454_data);
              v647_acc += ((static_cast<float>(v649_data[9])) * v456_data);
              v647_acc += ((static_cast<float>(v649_data[10])) * v458_data);
              v647_acc += ((static_cast<float>(v649_data[11])) * v460_data);
              v647_acc += ((static_cast<float>(v649_data[12])) * v462_data);
              v647_acc += ((static_cast<float>(v649_data[13])) * v464_data);
              v647_acc += ((static_cast<float>(v649_data[14])) * v466_data);
              v647_acc += ((static_cast<float>(v649_data[15])) * v468_data);
              tensorforge::intel_esimd::simd<float, 16> v682_data = tensorforge::slmLoad<float, 16>(s1 + (87_i32));
              v647_acc += ((static_cast<float>(v682_data[0])) * v470_data);
              v647_acc += ((static_cast<float>(v682_data[1])) * v472_data);
              v647_acc += ((static_cast<float>(v682_data[2])) * v474_data);
              ir1.template select<16, 1>(64) = v647_acc;
              tensorforge::intel_esimd::simd<float, 16> v689_acc{};
              tensorforge::intel_esimd::simd<float, 16> v691_data = tensorforge::slmLoad<float, 16>(s1 + (89_i32));
              v689_acc += ((static_cast<float>(v691_data[1])) * v440_data);
              v689_acc += ((static_cast<float>(v691_data[2])) * v442_data);
              v689_acc += ((static_cast<float>(v691_data[3])) * v444_data);
              v689_acc += ((static_cast<float>(v691_data[4])) * v446_data);
              v689_acc += ((static_cast<float>(v691_data[5])) * v448_data);
              v689_acc += ((static_cast<float>(v691_data[6])) * v450_data);
              v689_acc += ((static_cast<float>(v691_data[7])) * v452_data);
              v689_acc += ((static_cast<float>(v691_data[8])) * v454_data);
              v689_acc += ((static_cast<float>(v691_data[9])) * v456_data);
              v689_acc += ((static_cast<float>(v691_data[10])) * v458_data);
              v689_acc += ((static_cast<float>(v691_data[11])) * v460_data);
              v689_acc += ((static_cast<float>(v691_data[12])) * v462_data);
              v689_acc += ((static_cast<float>(v691_data[13])) * v464_data);
              v689_acc += ((static_cast<float>(v691_data[14])) * v466_data);
              v689_acc += ((static_cast<float>(v691_data[15])) * v468_data);
              tensorforge::intel_esimd::simd<float, 16> v724_data = tensorforge::slmLoad<float, 16>(s1 + (105_i32));
              v689_acc += ((static_cast<float>(v724_data[0])) * v470_data);
              v689_acc += ((static_cast<float>(v724_data[1])) * v472_data);
              v689_acc += ((static_cast<float>(v724_data[2])) * v474_data);
              ir1.template select<16, 1>(80) = v689_acc;
              tensorforge::intel_esimd::simd<float, 16> v731_acc{};
              tensorforge::intel_esimd::simd<float, 16> v733_data = tensorforge::slmLoad<float, 16>(s1 + (107_i32));
              v731_acc += ((static_cast<float>(v733_data[1])) * v440_data);
              v731_acc += ((static_cast<float>(v733_data[2])) * v442_data);
              v731_acc += ((static_cast<float>(v733_data[3])) * v444_data);
              v731_acc += ((static_cast<float>(v733_data[4])) * v446_data);
              v731_acc += ((static_cast<float>(v733_data[5])) * v448_data);
              v731_acc += ((static_cast<float>(v733_data[6])) * v450_data);
              v731_acc += ((static_cast<float>(v733_data[7])) * v452_data);
              v731_acc += ((static_cast<float>(v733_data[8])) * v454_data);
              v731_acc += ((static_cast<float>(v733_data[9])) * v456_data);
              v731_acc += ((static_cast<float>(v733_data[10])) * v458_data);
              v731_acc += ((static_cast<float>(v733_data[11])) * v460_data);
              v731_acc += ((static_cast<float>(v733_data[12])) * v462_data);
              v731_acc += ((static_cast<float>(v733_data[13])) * v464_data);
              v731_acc += ((static_cast<float>(v733_data[14])) * v466_data);
              v731_acc += ((static_cast<float>(v733_data[15])) * v468_data);
              tensorforge::intel_esimd::simd<float, 16> v766_data = tensorforge::slmLoad<float, 16>(s1 + (123_i32));
              v731_acc += ((static_cast<float>(v766_data[0])) * v470_data);
              v731_acc += ((static_cast<float>(v766_data[1])) * v472_data);
              v731_acc += ((static_cast<float>(v766_data[2])) * v474_data);
              ir1.template select<16, 1>(96) = v731_acc;
              tensorforge::intel_esimd::simd<float, 16> v773_acc{};
              tensorforge::intel_esimd::simd<float, 16> v775_data = tensorforge::slmLoad<float, 16>(s1 + (125_i32));
              v773_acc += ((static_cast<float>(v775_data[1])) * v440_data);
              v773_acc += ((static_cast<float>(v775_data[2])) * v442_data);
              v773_acc += ((static_cast<float>(v775_data[3])) * v444_data);
              v773_acc += ((static_cast<float>(v775_data[4])) * v446_data);
              v773_acc += ((static_cast<float>(v775_data[5])) * v448_data);
              v773_acc += ((static_cast<float>(v775_data[6])) * v450_data);
              v773_acc += ((static_cast<float>(v775_data[7])) * v452_data);
              v773_acc += ((static_cast<float>(v775_data[8])) * v454_data);
              v773_acc += ((static_cast<float>(v775_data[9])) * v456_data);
              v773_acc += ((static_cast<float>(v775_data[10])) * v458_data);
              v773_acc += ((static_cast<float>(v775_data[11])) * v460_data);
              v773_acc += ((static_cast<float>(v775_data[12])) * v462_data);
              v773_acc += ((static_cast<float>(v775_data[13])) * v464_data);
              v773_acc += ((static_cast<float>(v775_data[14])) * v466_data);
              v773_acc += ((static_cast<float>(v775_data[15])) * v468_data);
              tensorforge::intel_esimd::simd<float, 16> v808_data = tensorforge::slmLoad<float, 16>(s1 + (141_i32));
              v773_acc += ((static_cast<float>(v808_data[0])) * v470_data);
              v773_acc += ((static_cast<float>(v808_data[1])) * v472_data);
              v773_acc += ((static_cast<float>(v808_data[2])) * v474_data);
              ir1.template select<16, 1>(112) = v773_acc;
              tensorforge::intel_esimd::simd<float, 16> v815_acc{};
              tensorforge::intel_esimd::simd<float, 16> v817_data = tensorforge::slmLoad<float, 16>(s1 + (143_i32));
              v815_acc += ((static_cast<float>(v817_data[1])) * v440_data);
              v815_acc += ((static_cast<float>(v817_data[2])) * v442_data);
              v815_acc += ((static_cast<float>(v817_data[3])) * v444_data);
              v815_acc += ((static_cast<float>(v817_data[4])) * v446_data);
              v815_acc += ((static_cast<float>(v817_data[5])) * v448_data);
              v815_acc += ((static_cast<float>(v817_data[6])) * v450_data);
              v815_acc += ((static_cast<float>(v817_data[7])) * v452_data);
              v815_acc += ((static_cast<float>(v817_data[8])) * v454_data);
              v815_acc += ((static_cast<float>(v817_data[9])) * v456_data);
              v815_acc += ((static_cast<float>(v817_data[10])) * v458_data);
              v815_acc += ((static_cast<float>(v817_data[11])) * v460_data);
              v815_acc += ((static_cast<float>(v817_data[12])) * v462_data);
              v815_acc += ((static_cast<float>(v817_data[13])) * v464_data);
              v815_acc += ((static_cast<float>(v817_data[14])) * v466_data);
              v815_acc += ((static_cast<float>(v817_data[15])) * v468_data);
              tensorforge::intel_esimd::simd<float, 16> v850_data = tensorforge::slmLoad<float, 16>(s1 + (159_i32));
              v815_acc += ((static_cast<float>(v850_data[0])) * v470_data);
              v815_acc += ((static_cast<float>(v850_data[1])) * v472_data);
              v815_acc += ((static_cast<float>(v850_data[2])) * v474_data);
              ir1.template select<16, 1>(128) = v815_acc;
              // r1 = ir1 + r0
              #pragma unroll
              for (int32_t v857_n1 = 0; v857_n1 < 9; ++v857_n1) {
                int32_t v858_a = v857_n1 * 16;
                tensorforge::intel_esimd::simd<float, 10> v860_data(ir1.template select<10, 1>(v858_a));
                tensorforge::intel_esimd::simd<float, 10> v861_data(r0.template select<10, 1>(v858_a));
                r1.template select<10, 1>(v858_a) = (v861_data + v860_data);
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v863_i1 = 0; v863_i1 < 9; ++v863_i1) {
                tensorforge::intel_esimd::simd<float, 10> v866_data(r1.template select<10, 1>((v863_i1 * 16)));
                v866_data.copy_to(glb_m0 + ((v863_i1 * 10)));
              }
            }
            tensorforge::prefetchRunsL2<612, 648>(&pf_glb_m2[0], &pf_glb_m4[0]);
          }
        }
      }
    });
  });
}

