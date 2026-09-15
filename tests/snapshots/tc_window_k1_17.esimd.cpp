// === base name ===
kernel_2106a1a22d6f1231

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_2106a1a22d6f1231 = {{1, 16, 1}, 16, 16, 1, 16, 11264, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_2106a1a22d6f1231(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_2106a1a22d6f1231(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_2106a1a22d6f1231(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_2106a1a22d6f1231(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_2106a1a22d6f1231(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_2106a1a22d6f1231(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_2106a1a22d6f1231(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2816 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 11264 B shared, occupancy grid
        // operands:
        //   m0 16×9(16×9) {0..16}×{0..9} strided
        //   m1 16×20(16×17) {0..16}×{1..18} none
        //   m2 20×9(17×9) {1..18}×{0..9} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2816}],"shared_bytes":11264,"shared_elements":2816,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,9]],"name":"m0","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,1],[16,18]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[1,0],[18,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[16,18]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[18,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
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
              // [(0, 16), (0, 9)] [(1, 18)]
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
              tensorforge::intel_esimd::simd<float, 16> v62_data(0.0f);
              v62_data.template select<15, 1>(1) = tensorforge::slmLoad<float, 15>((s0 + (-1_i32)) + 1);
              v59_acc += ((static_cast<float>(v62_data[1])) * v26_data);
              v59_acc += ((static_cast<float>(v62_data[2])) * v28_data);
              v59_acc += ((static_cast<float>(v62_data[3])) * v30_data);
              v59_acc += ((static_cast<float>(v62_data[4])) * v32_data);
              v59_acc += ((static_cast<float>(v62_data[5])) * v34_data);
              v59_acc += ((static_cast<float>(v62_data[6])) * v36_data);
              v59_acc += ((static_cast<float>(v62_data[7])) * v38_data);
              v59_acc += ((static_cast<float>(v62_data[8])) * v40_data);
              v59_acc += ((static_cast<float>(v62_data[9])) * v42_data);
              v59_acc += ((static_cast<float>(v62_data[10])) * v44_data);
              v59_acc += ((static_cast<float>(v62_data[11])) * v46_data);
              v59_acc += ((static_cast<float>(v62_data[12])) * v48_data);
              v59_acc += ((static_cast<float>(v62_data[13])) * v50_data);
              v59_acc += ((static_cast<float>(v62_data[14])) * v52_data);
              v59_acc += ((static_cast<float>(v62_data[15])) * v54_data);
              tensorforge::intel_esimd::simd<float, 16> v98_data = tensorforge::slmLoad<float, 16>(s0 + (15_i32));
              v59_acc += ((static_cast<float>(v98_data[0])) * v56_data);
              v59_acc += ((static_cast<float>(v98_data[1])) * v58_data);
              ir0.template select<16, 1>(0) = v59_acc;
              tensorforge::intel_esimd::simd<float, 16> v103_acc{};
              tensorforge::intel_esimd::simd<float, 16> v105_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v103_acc += ((static_cast<float>(v105_data[1])) * v26_data);
              v103_acc += ((static_cast<float>(v105_data[2])) * v28_data);
              v103_acc += ((static_cast<float>(v105_data[3])) * v30_data);
              v103_acc += ((static_cast<float>(v105_data[4])) * v32_data);
              v103_acc += ((static_cast<float>(v105_data[5])) * v34_data);
              v103_acc += ((static_cast<float>(v105_data[6])) * v36_data);
              v103_acc += ((static_cast<float>(v105_data[7])) * v38_data);
              v103_acc += ((static_cast<float>(v105_data[8])) * v40_data);
              v103_acc += ((static_cast<float>(v105_data[9])) * v42_data);
              v103_acc += ((static_cast<float>(v105_data[10])) * v44_data);
              v103_acc += ((static_cast<float>(v105_data[11])) * v46_data);
              v103_acc += ((static_cast<float>(v105_data[12])) * v48_data);
              v103_acc += ((static_cast<float>(v105_data[13])) * v50_data);
              v103_acc += ((static_cast<float>(v105_data[14])) * v52_data);
              v103_acc += ((static_cast<float>(v105_data[15])) * v54_data);
              tensorforge::intel_esimd::simd<float, 16> v138_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v103_acc += ((static_cast<float>(v138_data[0])) * v56_data);
              v103_acc += ((static_cast<float>(v138_data[1])) * v58_data);
              ir0.template select<16, 1>(16) = v103_acc;
              tensorforge::intel_esimd::simd<float, 16> v143_acc{};
              tensorforge::intel_esimd::simd<float, 16> v145_data = tensorforge::slmLoad<float, 16>(s0 + (33_i32));
              v143_acc += ((static_cast<float>(v145_data[1])) * v26_data);
              v143_acc += ((static_cast<float>(v145_data[2])) * v28_data);
              v143_acc += ((static_cast<float>(v145_data[3])) * v30_data);
              v143_acc += ((static_cast<float>(v145_data[4])) * v32_data);
              v143_acc += ((static_cast<float>(v145_data[5])) * v34_data);
              v143_acc += ((static_cast<float>(v145_data[6])) * v36_data);
              v143_acc += ((static_cast<float>(v145_data[7])) * v38_data);
              v143_acc += ((static_cast<float>(v145_data[8])) * v40_data);
              v143_acc += ((static_cast<float>(v145_data[9])) * v42_data);
              v143_acc += ((static_cast<float>(v145_data[10])) * v44_data);
              v143_acc += ((static_cast<float>(v145_data[11])) * v46_data);
              v143_acc += ((static_cast<float>(v145_data[12])) * v48_data);
              v143_acc += ((static_cast<float>(v145_data[13])) * v50_data);
              v143_acc += ((static_cast<float>(v145_data[14])) * v52_data);
              v143_acc += ((static_cast<float>(v145_data[15])) * v54_data);
              tensorforge::intel_esimd::simd<float, 16> v178_data = tensorforge::slmLoad<float, 16>(s0 + (49_i32));
              v143_acc += ((static_cast<float>(v178_data[0])) * v56_data);
              v143_acc += ((static_cast<float>(v178_data[1])) * v58_data);
              ir0.template select<16, 1>(32) = v143_acc;
              tensorforge::intel_esimd::simd<float, 16> v183_acc{};
              tensorforge::intel_esimd::simd<float, 16> v185_data = tensorforge::slmLoad<float, 16>(s0 + (50_i32));
              v183_acc += ((static_cast<float>(v185_data[1])) * v26_data);
              v183_acc += ((static_cast<float>(v185_data[2])) * v28_data);
              v183_acc += ((static_cast<float>(v185_data[3])) * v30_data);
              v183_acc += ((static_cast<float>(v185_data[4])) * v32_data);
              v183_acc += ((static_cast<float>(v185_data[5])) * v34_data);
              v183_acc += ((static_cast<float>(v185_data[6])) * v36_data);
              v183_acc += ((static_cast<float>(v185_data[7])) * v38_data);
              v183_acc += ((static_cast<float>(v185_data[8])) * v40_data);
              v183_acc += ((static_cast<float>(v185_data[9])) * v42_data);
              v183_acc += ((static_cast<float>(v185_data[10])) * v44_data);
              v183_acc += ((static_cast<float>(v185_data[11])) * v46_data);
              v183_acc += ((static_cast<float>(v185_data[12])) * v48_data);
              v183_acc += ((static_cast<float>(v185_data[13])) * v50_data);
              v183_acc += ((static_cast<float>(v185_data[14])) * v52_data);
              v183_acc += ((static_cast<float>(v185_data[15])) * v54_data);
              tensorforge::intel_esimd::simd<float, 16> v218_data = tensorforge::slmLoad<float, 16>(s0 + (66_i32));
              v183_acc += ((static_cast<float>(v218_data[0])) * v56_data);
              v183_acc += ((static_cast<float>(v218_data[1])) * v58_data);
              ir0.template select<16, 1>(48) = v183_acc;
              tensorforge::intel_esimd::simd<float, 16> v223_acc{};
              tensorforge::intel_esimd::simd<float, 16> v225_data = tensorforge::slmLoad<float, 16>(s0 + (67_i32));
              v223_acc += ((static_cast<float>(v225_data[1])) * v26_data);
              v223_acc += ((static_cast<float>(v225_data[2])) * v28_data);
              v223_acc += ((static_cast<float>(v225_data[3])) * v30_data);
              v223_acc += ((static_cast<float>(v225_data[4])) * v32_data);
              v223_acc += ((static_cast<float>(v225_data[5])) * v34_data);
              v223_acc += ((static_cast<float>(v225_data[6])) * v36_data);
              v223_acc += ((static_cast<float>(v225_data[7])) * v38_data);
              v223_acc += ((static_cast<float>(v225_data[8])) * v40_data);
              v223_acc += ((static_cast<float>(v225_data[9])) * v42_data);
              v223_acc += ((static_cast<float>(v225_data[10])) * v44_data);
              v223_acc += ((static_cast<float>(v225_data[11])) * v46_data);
              v223_acc += ((static_cast<float>(v225_data[12])) * v48_data);
              v223_acc += ((static_cast<float>(v225_data[13])) * v50_data);
              v223_acc += ((static_cast<float>(v225_data[14])) * v52_data);
              v223_acc += ((static_cast<float>(v225_data[15])) * v54_data);
              tensorforge::intel_esimd::simd<float, 16> v258_data = tensorforge::slmLoad<float, 16>(s0 + (83_i32));
              v223_acc += ((static_cast<float>(v258_data[0])) * v56_data);
              v223_acc += ((static_cast<float>(v258_data[1])) * v58_data);
              ir0.template select<16, 1>(64) = v223_acc;
              tensorforge::intel_esimd::simd<float, 16> v263_acc{};
              tensorforge::intel_esimd::simd<float, 16> v265_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v263_acc += ((static_cast<float>(v265_data[1])) * v26_data);
              v263_acc += ((static_cast<float>(v265_data[2])) * v28_data);
              v263_acc += ((static_cast<float>(v265_data[3])) * v30_data);
              v263_acc += ((static_cast<float>(v265_data[4])) * v32_data);
              v263_acc += ((static_cast<float>(v265_data[5])) * v34_data);
              v263_acc += ((static_cast<float>(v265_data[6])) * v36_data);
              v263_acc += ((static_cast<float>(v265_data[7])) * v38_data);
              v263_acc += ((static_cast<float>(v265_data[8])) * v40_data);
              v263_acc += ((static_cast<float>(v265_data[9])) * v42_data);
              v263_acc += ((static_cast<float>(v265_data[10])) * v44_data);
              v263_acc += ((static_cast<float>(v265_data[11])) * v46_data);
              v263_acc += ((static_cast<float>(v265_data[12])) * v48_data);
              v263_acc += ((static_cast<float>(v265_data[13])) * v50_data);
              v263_acc += ((static_cast<float>(v265_data[14])) * v52_data);
              v263_acc += ((static_cast<float>(v265_data[15])) * v54_data);
              tensorforge::intel_esimd::simd<float, 16> v298_data = tensorforge::slmLoad<float, 16>(s0 + (100_i32));
              v263_acc += ((static_cast<float>(v298_data[0])) * v56_data);
              v263_acc += ((static_cast<float>(v298_data[1])) * v58_data);
              ir0.template select<16, 1>(80) = v263_acc;
              tensorforge::intel_esimd::simd<float, 16> v303_acc{};
              tensorforge::intel_esimd::simd<float, 16> v305_data = tensorforge::slmLoad<float, 16>(s0 + (101_i32));
              v303_acc += ((static_cast<float>(v305_data[1])) * v26_data);
              v303_acc += ((static_cast<float>(v305_data[2])) * v28_data);
              v303_acc += ((static_cast<float>(v305_data[3])) * v30_data);
              v303_acc += ((static_cast<float>(v305_data[4])) * v32_data);
              v303_acc += ((static_cast<float>(v305_data[5])) * v34_data);
              v303_acc += ((static_cast<float>(v305_data[6])) * v36_data);
              v303_acc += ((static_cast<float>(v305_data[7])) * v38_data);
              v303_acc += ((static_cast<float>(v305_data[8])) * v40_data);
              v303_acc += ((static_cast<float>(v305_data[9])) * v42_data);
              v303_acc += ((static_cast<float>(v305_data[10])) * v44_data);
              v303_acc += ((static_cast<float>(v305_data[11])) * v46_data);
              v303_acc += ((static_cast<float>(v305_data[12])) * v48_data);
              v303_acc += ((static_cast<float>(v305_data[13])) * v50_data);
              v303_acc += ((static_cast<float>(v305_data[14])) * v52_data);
              v303_acc += ((static_cast<float>(v305_data[15])) * v54_data);
              tensorforge::intel_esimd::simd<float, 16> v338_data = tensorforge::slmLoad<float, 16>(s0 + (117_i32));
              v303_acc += ((static_cast<float>(v338_data[0])) * v56_data);
              v303_acc += ((static_cast<float>(v338_data[1])) * v58_data);
              ir0.template select<16, 1>(96) = v303_acc;
              tensorforge::intel_esimd::simd<float, 16> v343_acc{};
              tensorforge::intel_esimd::simd<float, 16> v345_data = tensorforge::slmLoad<float, 16>(s0 + (118_i32));
              v343_acc += ((static_cast<float>(v345_data[1])) * v26_data);
              v343_acc += ((static_cast<float>(v345_data[2])) * v28_data);
              v343_acc += ((static_cast<float>(v345_data[3])) * v30_data);
              v343_acc += ((static_cast<float>(v345_data[4])) * v32_data);
              v343_acc += ((static_cast<float>(v345_data[5])) * v34_data);
              v343_acc += ((static_cast<float>(v345_data[6])) * v36_data);
              v343_acc += ((static_cast<float>(v345_data[7])) * v38_data);
              v343_acc += ((static_cast<float>(v345_data[8])) * v40_data);
              v343_acc += ((static_cast<float>(v345_data[9])) * v42_data);
              v343_acc += ((static_cast<float>(v345_data[10])) * v44_data);
              v343_acc += ((static_cast<float>(v345_data[11])) * v46_data);
              v343_acc += ((static_cast<float>(v345_data[12])) * v48_data);
              v343_acc += ((static_cast<float>(v345_data[13])) * v50_data);
              v343_acc += ((static_cast<float>(v345_data[14])) * v52_data);
              v343_acc += ((static_cast<float>(v345_data[15])) * v54_data);
              tensorforge::intel_esimd::simd<float, 16> v378_data = tensorforge::slmLoad<float, 16>(s0 + (134_i32));
              v343_acc += ((static_cast<float>(v378_data[0])) * v56_data);
              v343_acc += ((static_cast<float>(v378_data[1])) * v58_data);
              ir0.template select<16, 1>(112) = v343_acc;
              tensorforge::intel_esimd::simd<float, 16> v383_acc{};
              tensorforge::intel_esimd::simd<float, 16> v385_data = tensorforge::slmLoad<float, 16>(s0 + (135_i32));
              v383_acc += ((static_cast<float>(v385_data[1])) * v26_data);
              v383_acc += ((static_cast<float>(v385_data[2])) * v28_data);
              v383_acc += ((static_cast<float>(v385_data[3])) * v30_data);
              v383_acc += ((static_cast<float>(v385_data[4])) * v32_data);
              v383_acc += ((static_cast<float>(v385_data[5])) * v34_data);
              v383_acc += ((static_cast<float>(v385_data[6])) * v36_data);
              v383_acc += ((static_cast<float>(v385_data[7])) * v38_data);
              v383_acc += ((static_cast<float>(v385_data[8])) * v40_data);
              v383_acc += ((static_cast<float>(v385_data[9])) * v42_data);
              v383_acc += ((static_cast<float>(v385_data[10])) * v44_data);
              v383_acc += ((static_cast<float>(v385_data[11])) * v46_data);
              v383_acc += ((static_cast<float>(v385_data[12])) * v48_data);
              v383_acc += ((static_cast<float>(v385_data[13])) * v50_data);
              v383_acc += ((static_cast<float>(v385_data[14])) * v52_data);
              v383_acc += ((static_cast<float>(v385_data[15])) * v54_data);
              tensorforge::intel_esimd::simd<float, 16> v418_data = tensorforge::slmLoad<float, 16>(s0 + (151_i32));
              v383_acc += ((static_cast<float>(v418_data[0])) * v56_data);
              v383_acc += ((static_cast<float>(v418_data[1])) * v58_data);
              ir0.template select<16, 1>(128) = v383_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v423_n0 = 0; v423_n0 < 1; ++v423_n0) {
                int32_t v425_a = v423_n0 * 16;
                #pragma unroll
                for (int32_t v424_n1 = 0; v424_n1 < 9; ++v424_n1) {
                  int32_t v427_a = v425_a + (v424_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v428_data(ir0.template select<16, 1>(v427_a));
                  r0.template select<16, 1>(v427_a) = v428_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v429_i0 = 0; v429_i0 < 1; ++v429_i0) {
                int32_t v431_a = v429_i0 * 16;
                #pragma unroll
                for (int32_t v430_i1 = 0; v430_i1 < 9; ++v430_i1) {
                  int32_t v433_a = v431_a + (v430_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v434_data(r0.template select<16, 1>(v433_a));
                  v434_data.copy_to(glb_m0 + (v433_a));
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

