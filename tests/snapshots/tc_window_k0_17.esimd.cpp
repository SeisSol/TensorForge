// === base name ===
kernel_7814bc39dce69da3

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_7814bc39dce69da3 = {{1, 16, 1}, 16, 16, 1, 16, 11264, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_7814bc39dce69da3(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_7814bc39dce69da3(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_7814bc39dce69da3(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 2816 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_7814bc39dce69da3(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_7814bc39dce69da3(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_7814bc39dce69da3(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_7814bc39dce69da3(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2816 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 11264 B shared, occupancy grid
        // operands:
        //   m0 16×9(16×9) {0..16}×{0..9} strided
        //   m1 16×20(16×17) {0..16}×{0..17} none
        //   m2 20×9(17×9) {0..17}×{0..9} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2816}],"shared_bytes":11264,"shared_elements":2816,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,9]],"name":"m0","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[16,17]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[17,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[16,17]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[0,0],[17,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (176 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (160);
          const float *const __restrict__ glb_m1 = &m1[0];
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v6_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v6_batchId0 < numElements0; v6_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v7_ahead1 = v6_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
            const float *const __restrict__ pf_glb_m2 = &m2[v9_batchId1 * 153 + 0 + m2_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 144 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v6_batchId0 * 153 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v17_ld;
              v17_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v17_ld);
              tensorforge::intel_esimd::simd<float, 64> v18_ld;
              v18_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v18_ld);
              tensorforge::intel_esimd::simd<float, 16> v19_ld;
              v19_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v19_ld);
              tensorforge::intel_esimd::simd<float, 9> v20_ld;
              v20_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 144));
              tensorforge::slmStore<float, 9>(s0 + (0 + 0 + 1 * 0 + 144), v20_ld);
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 16), (0, 9)] [(0, 17)]
              tensorforge::intel_esimd::simd<float, 144> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run0;
              glb_m1_run0.copy_from(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v26_data(glb_m1_run0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v28_data(glb_m1_run0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v30_data(glb_m1_run0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v32_data(glb_m1_run0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run1;
              glb_m1_run1.copy_from(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v34_data(glb_m1_run1.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v36_data(glb_m1_run1.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v38_data(glb_m1_run1.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v40_data(glb_m1_run1.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run2;
              glb_m1_run2.copy_from(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v42_data(glb_m1_run2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v44_data(glb_m1_run2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v46_data(glb_m1_run2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v48_data(glb_m1_run2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run3;
              glb_m1_run3.copy_from(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v50_data(glb_m1_run3.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v52_data(glb_m1_run3.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v54_data(glb_m1_run3.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v56_data(glb_m1_run3.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v58_data;
              v58_data.copy_from(glb_m1 + (256_i32));
              tensorforge::intel_esimd::simd<float, 16> v59_acc{};
              tensorforge::intel_esimd::simd<float, 16> v60_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v59_acc += ((static_cast<float>(v60_data[0])) * v26_data);
              v59_acc += ((static_cast<float>(v60_data[1])) * v28_data);
              v59_acc += ((static_cast<float>(v60_data[2])) * v30_data);
              v59_acc += ((static_cast<float>(v60_data[3])) * v32_data);
              v59_acc += ((static_cast<float>(v60_data[4])) * v34_data);
              v59_acc += ((static_cast<float>(v60_data[5])) * v36_data);
              v59_acc += ((static_cast<float>(v60_data[6])) * v38_data);
              v59_acc += ((static_cast<float>(v60_data[7])) * v40_data);
              v59_acc += ((static_cast<float>(v60_data[8])) * v42_data);
              v59_acc += ((static_cast<float>(v60_data[9])) * v44_data);
              v59_acc += ((static_cast<float>(v60_data[10])) * v46_data);
              v59_acc += ((static_cast<float>(v60_data[11])) * v48_data);
              v59_acc += ((static_cast<float>(v60_data[12])) * v50_data);
              v59_acc += ((static_cast<float>(v60_data[13])) * v52_data);
              v59_acc += ((static_cast<float>(v60_data[14])) * v54_data);
              v59_acc += ((static_cast<float>(v60_data[15])) * v56_data);
              tensorforge::intel_esimd::simd<float, 16> v96_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v59_acc += ((static_cast<float>(v96_data[0])) * v58_data);
              ir0.template select<16, 1>(0) = v59_acc;
              tensorforge::intel_esimd::simd<float, 16> v99_acc{};
              tensorforge::intel_esimd::simd<float, 16> v101_data = tensorforge::slmLoad<float, 16>(s0 + (17_i32));
              v99_acc += ((static_cast<float>(v101_data[0])) * v26_data);
              v99_acc += ((static_cast<float>(v101_data[1])) * v28_data);
              v99_acc += ((static_cast<float>(v101_data[2])) * v30_data);
              v99_acc += ((static_cast<float>(v101_data[3])) * v32_data);
              v99_acc += ((static_cast<float>(v101_data[4])) * v34_data);
              v99_acc += ((static_cast<float>(v101_data[5])) * v36_data);
              v99_acc += ((static_cast<float>(v101_data[6])) * v38_data);
              v99_acc += ((static_cast<float>(v101_data[7])) * v40_data);
              v99_acc += ((static_cast<float>(v101_data[8])) * v42_data);
              v99_acc += ((static_cast<float>(v101_data[9])) * v44_data);
              v99_acc += ((static_cast<float>(v101_data[10])) * v46_data);
              v99_acc += ((static_cast<float>(v101_data[11])) * v48_data);
              v99_acc += ((static_cast<float>(v101_data[12])) * v50_data);
              v99_acc += ((static_cast<float>(v101_data[13])) * v52_data);
              v99_acc += ((static_cast<float>(v101_data[14])) * v54_data);
              v99_acc += ((static_cast<float>(v101_data[15])) * v56_data);
              tensorforge::intel_esimd::simd<float, 16> v135_data = tensorforge::slmLoad<float, 16>(s0 + (33_i32));
              v99_acc += ((static_cast<float>(v135_data[0])) * v58_data);
              ir0.template select<16, 1>(16) = v99_acc;
              tensorforge::intel_esimd::simd<float, 16> v138_acc{};
              tensorforge::intel_esimd::simd<float, 16> v140_data = tensorforge::slmLoad<float, 16>(s0 + (34_i32));
              v138_acc += ((static_cast<float>(v140_data[0])) * v26_data);
              v138_acc += ((static_cast<float>(v140_data[1])) * v28_data);
              v138_acc += ((static_cast<float>(v140_data[2])) * v30_data);
              v138_acc += ((static_cast<float>(v140_data[3])) * v32_data);
              v138_acc += ((static_cast<float>(v140_data[4])) * v34_data);
              v138_acc += ((static_cast<float>(v140_data[5])) * v36_data);
              v138_acc += ((static_cast<float>(v140_data[6])) * v38_data);
              v138_acc += ((static_cast<float>(v140_data[7])) * v40_data);
              v138_acc += ((static_cast<float>(v140_data[8])) * v42_data);
              v138_acc += ((static_cast<float>(v140_data[9])) * v44_data);
              v138_acc += ((static_cast<float>(v140_data[10])) * v46_data);
              v138_acc += ((static_cast<float>(v140_data[11])) * v48_data);
              v138_acc += ((static_cast<float>(v140_data[12])) * v50_data);
              v138_acc += ((static_cast<float>(v140_data[13])) * v52_data);
              v138_acc += ((static_cast<float>(v140_data[14])) * v54_data);
              v138_acc += ((static_cast<float>(v140_data[15])) * v56_data);
              tensorforge::intel_esimd::simd<float, 16> v174_data = tensorforge::slmLoad<float, 16>(s0 + (50_i32));
              v138_acc += ((static_cast<float>(v174_data[0])) * v58_data);
              ir0.template select<16, 1>(32) = v138_acc;
              tensorforge::intel_esimd::simd<float, 16> v177_acc{};
              tensorforge::intel_esimd::simd<float, 16> v179_data = tensorforge::slmLoad<float, 16>(s0 + (51_i32));
              v177_acc += ((static_cast<float>(v179_data[0])) * v26_data);
              v177_acc += ((static_cast<float>(v179_data[1])) * v28_data);
              v177_acc += ((static_cast<float>(v179_data[2])) * v30_data);
              v177_acc += ((static_cast<float>(v179_data[3])) * v32_data);
              v177_acc += ((static_cast<float>(v179_data[4])) * v34_data);
              v177_acc += ((static_cast<float>(v179_data[5])) * v36_data);
              v177_acc += ((static_cast<float>(v179_data[6])) * v38_data);
              v177_acc += ((static_cast<float>(v179_data[7])) * v40_data);
              v177_acc += ((static_cast<float>(v179_data[8])) * v42_data);
              v177_acc += ((static_cast<float>(v179_data[9])) * v44_data);
              v177_acc += ((static_cast<float>(v179_data[10])) * v46_data);
              v177_acc += ((static_cast<float>(v179_data[11])) * v48_data);
              v177_acc += ((static_cast<float>(v179_data[12])) * v50_data);
              v177_acc += ((static_cast<float>(v179_data[13])) * v52_data);
              v177_acc += ((static_cast<float>(v179_data[14])) * v54_data);
              v177_acc += ((static_cast<float>(v179_data[15])) * v56_data);
              tensorforge::intel_esimd::simd<float, 16> v213_data = tensorforge::slmLoad<float, 16>(s0 + (67_i32));
              v177_acc += ((static_cast<float>(v213_data[0])) * v58_data);
              ir0.template select<16, 1>(48) = v177_acc;
              tensorforge::intel_esimd::simd<float, 16> v216_acc{};
              tensorforge::intel_esimd::simd<float, 16> v218_data = tensorforge::slmLoad<float, 16>(s0 + (68_i32));
              v216_acc += ((static_cast<float>(v218_data[0])) * v26_data);
              v216_acc += ((static_cast<float>(v218_data[1])) * v28_data);
              v216_acc += ((static_cast<float>(v218_data[2])) * v30_data);
              v216_acc += ((static_cast<float>(v218_data[3])) * v32_data);
              v216_acc += ((static_cast<float>(v218_data[4])) * v34_data);
              v216_acc += ((static_cast<float>(v218_data[5])) * v36_data);
              v216_acc += ((static_cast<float>(v218_data[6])) * v38_data);
              v216_acc += ((static_cast<float>(v218_data[7])) * v40_data);
              v216_acc += ((static_cast<float>(v218_data[8])) * v42_data);
              v216_acc += ((static_cast<float>(v218_data[9])) * v44_data);
              v216_acc += ((static_cast<float>(v218_data[10])) * v46_data);
              v216_acc += ((static_cast<float>(v218_data[11])) * v48_data);
              v216_acc += ((static_cast<float>(v218_data[12])) * v50_data);
              v216_acc += ((static_cast<float>(v218_data[13])) * v52_data);
              v216_acc += ((static_cast<float>(v218_data[14])) * v54_data);
              v216_acc += ((static_cast<float>(v218_data[15])) * v56_data);
              tensorforge::intel_esimd::simd<float, 16> v252_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v216_acc += ((static_cast<float>(v252_data[0])) * v58_data);
              ir0.template select<16, 1>(64) = v216_acc;
              tensorforge::intel_esimd::simd<float, 16> v255_acc{};
              tensorforge::intel_esimd::simd<float, 16> v257_data = tensorforge::slmLoad<float, 16>(s0 + (85_i32));
              v255_acc += ((static_cast<float>(v257_data[0])) * v26_data);
              v255_acc += ((static_cast<float>(v257_data[1])) * v28_data);
              v255_acc += ((static_cast<float>(v257_data[2])) * v30_data);
              v255_acc += ((static_cast<float>(v257_data[3])) * v32_data);
              v255_acc += ((static_cast<float>(v257_data[4])) * v34_data);
              v255_acc += ((static_cast<float>(v257_data[5])) * v36_data);
              v255_acc += ((static_cast<float>(v257_data[6])) * v38_data);
              v255_acc += ((static_cast<float>(v257_data[7])) * v40_data);
              v255_acc += ((static_cast<float>(v257_data[8])) * v42_data);
              v255_acc += ((static_cast<float>(v257_data[9])) * v44_data);
              v255_acc += ((static_cast<float>(v257_data[10])) * v46_data);
              v255_acc += ((static_cast<float>(v257_data[11])) * v48_data);
              v255_acc += ((static_cast<float>(v257_data[12])) * v50_data);
              v255_acc += ((static_cast<float>(v257_data[13])) * v52_data);
              v255_acc += ((static_cast<float>(v257_data[14])) * v54_data);
              v255_acc += ((static_cast<float>(v257_data[15])) * v56_data);
              tensorforge::intel_esimd::simd<float, 16> v291_data = tensorforge::slmLoad<float, 16>(s0 + (101_i32));
              v255_acc += ((static_cast<float>(v291_data[0])) * v58_data);
              ir0.template select<16, 1>(80) = v255_acc;
              tensorforge::intel_esimd::simd<float, 16> v294_acc{};
              tensorforge::intel_esimd::simd<float, 16> v296_data = tensorforge::slmLoad<float, 16>(s0 + (102_i32));
              v294_acc += ((static_cast<float>(v296_data[0])) * v26_data);
              v294_acc += ((static_cast<float>(v296_data[1])) * v28_data);
              v294_acc += ((static_cast<float>(v296_data[2])) * v30_data);
              v294_acc += ((static_cast<float>(v296_data[3])) * v32_data);
              v294_acc += ((static_cast<float>(v296_data[4])) * v34_data);
              v294_acc += ((static_cast<float>(v296_data[5])) * v36_data);
              v294_acc += ((static_cast<float>(v296_data[6])) * v38_data);
              v294_acc += ((static_cast<float>(v296_data[7])) * v40_data);
              v294_acc += ((static_cast<float>(v296_data[8])) * v42_data);
              v294_acc += ((static_cast<float>(v296_data[9])) * v44_data);
              v294_acc += ((static_cast<float>(v296_data[10])) * v46_data);
              v294_acc += ((static_cast<float>(v296_data[11])) * v48_data);
              v294_acc += ((static_cast<float>(v296_data[12])) * v50_data);
              v294_acc += ((static_cast<float>(v296_data[13])) * v52_data);
              v294_acc += ((static_cast<float>(v296_data[14])) * v54_data);
              v294_acc += ((static_cast<float>(v296_data[15])) * v56_data);
              tensorforge::intel_esimd::simd<float, 16> v330_data = tensorforge::slmLoad<float, 16>(s0 + (118_i32));
              v294_acc += ((static_cast<float>(v330_data[0])) * v58_data);
              ir0.template select<16, 1>(96) = v294_acc;
              tensorforge::intel_esimd::simd<float, 16> v333_acc{};
              tensorforge::intel_esimd::simd<float, 16> v335_data = tensorforge::slmLoad<float, 16>(s0 + (119_i32));
              v333_acc += ((static_cast<float>(v335_data[0])) * v26_data);
              v333_acc += ((static_cast<float>(v335_data[1])) * v28_data);
              v333_acc += ((static_cast<float>(v335_data[2])) * v30_data);
              v333_acc += ((static_cast<float>(v335_data[3])) * v32_data);
              v333_acc += ((static_cast<float>(v335_data[4])) * v34_data);
              v333_acc += ((static_cast<float>(v335_data[5])) * v36_data);
              v333_acc += ((static_cast<float>(v335_data[6])) * v38_data);
              v333_acc += ((static_cast<float>(v335_data[7])) * v40_data);
              v333_acc += ((static_cast<float>(v335_data[8])) * v42_data);
              v333_acc += ((static_cast<float>(v335_data[9])) * v44_data);
              v333_acc += ((static_cast<float>(v335_data[10])) * v46_data);
              v333_acc += ((static_cast<float>(v335_data[11])) * v48_data);
              v333_acc += ((static_cast<float>(v335_data[12])) * v50_data);
              v333_acc += ((static_cast<float>(v335_data[13])) * v52_data);
              v333_acc += ((static_cast<float>(v335_data[14])) * v54_data);
              v333_acc += ((static_cast<float>(v335_data[15])) * v56_data);
              tensorforge::intel_esimd::simd<float, 16> v369_data = tensorforge::slmLoad<float, 16>(s0 + (135_i32));
              v333_acc += ((static_cast<float>(v369_data[0])) * v58_data);
              ir0.template select<16, 1>(112) = v333_acc;
              tensorforge::intel_esimd::simd<float, 16> v372_acc{};
              tensorforge::intel_esimd::simd<float, 16> v374_data = tensorforge::slmLoad<float, 16>(s0 + (136_i32));
              v372_acc += ((static_cast<float>(v374_data[0])) * v26_data);
              v372_acc += ((static_cast<float>(v374_data[1])) * v28_data);
              v372_acc += ((static_cast<float>(v374_data[2])) * v30_data);
              v372_acc += ((static_cast<float>(v374_data[3])) * v32_data);
              v372_acc += ((static_cast<float>(v374_data[4])) * v34_data);
              v372_acc += ((static_cast<float>(v374_data[5])) * v36_data);
              v372_acc += ((static_cast<float>(v374_data[6])) * v38_data);
              v372_acc += ((static_cast<float>(v374_data[7])) * v40_data);
              v372_acc += ((static_cast<float>(v374_data[8])) * v42_data);
              v372_acc += ((static_cast<float>(v374_data[9])) * v44_data);
              v372_acc += ((static_cast<float>(v374_data[10])) * v46_data);
              v372_acc += ((static_cast<float>(v374_data[11])) * v48_data);
              v372_acc += ((static_cast<float>(v374_data[12])) * v50_data);
              v372_acc += ((static_cast<float>(v374_data[13])) * v52_data);
              v372_acc += ((static_cast<float>(v374_data[14])) * v54_data);
              v372_acc += ((static_cast<float>(v374_data[15])) * v56_data);
              tensorforge::intel_esimd::simd<float, 16> v408_data = tensorforge::slmLoad<float, 16>(s0 + (152_i32));
              v372_acc += ((static_cast<float>(v408_data[0])) * v58_data);
              ir0.template select<16, 1>(128) = v372_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v411_n0 = 0; v411_n0 < 1; ++v411_n0) {
                int32_t v413_a = v411_n0 * 16;
                #pragma unroll
                for (int32_t v412_n1 = 0; v412_n1 < 9; ++v412_n1) {
                  int32_t v415_a = v413_a + (v412_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v416_data(ir0.template select<16, 1>(v415_a));
                  r0.template select<16, 1>(v415_a) = v416_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v417_i0 = 0; v417_i0 < 1; ++v417_i0) {
                int32_t v419_a = v417_i0 * 16;
                #pragma unroll
                for (int32_t v418_i1 = 0; v418_i1 < 9; ++v418_i1) {
                  int32_t v421_a = v419_a + (v418_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v422_data(r0.template select<16, 1>(v421_a));
                  v422_data.copy_to(glb_m0 + (v421_a));
                }
              }
            }
            tensorforge::prefetchL2<153>(&pf_glb_m2[0]);
          }
        }
      }
    });
  });
}

