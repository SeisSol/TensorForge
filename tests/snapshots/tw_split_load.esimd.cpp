// === base name ===
kernel_98caab1357f7a771

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_98caab1357f7a771 = {{1, 16, 1}, 16, 10, 1, 16, 21504, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_98caab1357f7a771(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_98caab1357f7a771(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_98caab1357f7a771(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 16, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 16 - 1) / 16;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 5376 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_98caab1357f7a771(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_98caab1357f7a771(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_98caab1357f7a771(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, m3, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_98caab1357f7a771(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<5376 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (10 active) x 16 per block = block 1x16x1, 21504 B shared, occupancy grid
        // operands:
        //   m0 10×9(10×9) {0..10}×{0..9} strided
        //   m1 10×17(10×17) {0..10}×{0..17} none
        //   m2 17×9(17×9) {0..17}×{0..9} strided
        //   m3 10×17(10×17) {0..10}×{0..17} none
        //   m4 17×9(17×9) {0..17}×{0..9} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        //   m0[i,j] += m3[i,k] × m4[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":10,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":5376}],"shared_bytes":21504,"shared_elements":5376,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[10,9]],"name":"m0","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"none","alias":"A1","bbox":[[0,0],[10,17]],"name":"m1","ordered":false,"parts":1,"shape":[10,17],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[17,9]],"name":"m2","ordered":false,"parts":1,"shape":[17,9],"variant":false},{"addressing":"none","alias":"A2","bbox":[[0,0],[10,17]],"name":"m3","ordered":false,"parts":1,"shape":[10,17],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[17,9]],"name":"m4","ordered":false,"parts":1,"shape":[17,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[10,17]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[10,17]},{"addressing":"strided","bbox":[[0,0],[17,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[17,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[10,17]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[10,17]},{"addressing":"strided","bbox":[[0,0],[17,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[17,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (336 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (320);
          const float *const __restrict__ glb_m1 = &m1[0];
          const float *const __restrict__ glb_m3 = &m3[0];
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (160);
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 90 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 153 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v8_batchId0 * 153 + 0 + m4_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v19_ld;
              v19_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v19_ld);
              tensorforge::intel_esimd::simd<float, 64> v20_ld;
              v20_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v20_ld);
              tensorforge::intel_esimd::simd<float, 16> v21_ld;
              v21_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v21_ld);
              tensorforge::intel_esimd::simd<float, 9> v22_ld;
              v22_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 144));
              tensorforge::slmStore<float, 9>(s0 + (0 + 0 + 1 * 0 + 144), v22_ld);
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v23_ld;
              v23_ld.copy_from(glb_m4 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 4 * 0 + 0), v23_ld);
              tensorforge::intel_esimd::simd<float, 64> v24_ld;
              v24_ld.copy_from(glb_m4 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 4 * 0 + 64), v24_ld);
              tensorforge::intel_esimd::simd<float, 16> v25_ld;
              v25_ld.copy_from(glb_m4 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s1 + (0 + 0 + 1 * 0 + 128), v25_ld);
              tensorforge::intel_esimd::simd<float, 9> v26_ld;
              v26_ld.copy_from(glb_m4 + (0 + 0 + 1 * 0 + 144));
              tensorforge::slmStore<float, 9>(s1 + (0 + 0 + 1 * 0 + 144), v26_ld);
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 10), (0, 9)] [(0, 17)]
              tensorforge::intel_esimd::simd<float, 144> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v32_data;
              v32_data.copy_from(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(glb_m1 + (10_i32));
              tensorforge::intel_esimd::simd<float, 16> v36_data;
              v36_data.copy_from(glb_m1 + (20_i32));
              tensorforge::intel_esimd::simd<float, 16> v38_data;
              v38_data.copy_from(glb_m1 + (30_i32));
              tensorforge::intel_esimd::simd<float, 16> v40_data;
              v40_data.copy_from(glb_m1 + (40_i32));
              tensorforge::intel_esimd::simd<float, 16> v42_data;
              v42_data.copy_from(glb_m1 + (50_i32));
              tensorforge::intel_esimd::simd<float, 16> v44_data;
              v44_data.copy_from(glb_m1 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v46_data;
              v46_data.copy_from(glb_m1 + (70_i32));
              tensorforge::intel_esimd::simd<float, 16> v48_data;
              v48_data.copy_from(glb_m1 + (80_i32));
              tensorforge::intel_esimd::simd<float, 16> v50_data;
              v50_data.copy_from(glb_m1 + (90_i32));
              tensorforge::intel_esimd::simd<float, 16> v52_data;
              v52_data.copy_from(glb_m1 + (100_i32));
              tensorforge::intel_esimd::simd<float, 16> v54_data;
              v54_data.copy_from(glb_m1 + (110_i32));
              tensorforge::intel_esimd::simd<float, 16> v56_data;
              v56_data.copy_from(glb_m1 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v58_data;
              v58_data.copy_from(glb_m1 + (130_i32));
              tensorforge::intel_esimd::simd<float, 16> v60_data;
              v60_data.copy_from(glb_m1 + (140_i32));
              tensorforge::intel_esimd::simd<float, 16> v62_data;
              v62_data.copy_from(glb_m1 + (150_i32));
              tensorforge::intel_esimd::simd<float, 16> v64_data;
              v64_data.copy_from(glb_m1 + (160_i32));
              tensorforge::intel_esimd::simd<float, 16> v65_acc{};
              tensorforge::intel_esimd::simd<float, 16> v66_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v65_acc += ((static_cast<float>(v66_data[0])) * v32_data);
              v65_acc += ((static_cast<float>(v66_data[1])) * v34_data);
              v65_acc += ((static_cast<float>(v66_data[2])) * v36_data);
              v65_acc += ((static_cast<float>(v66_data[3])) * v38_data);
              v65_acc += ((static_cast<float>(v66_data[4])) * v40_data);
              v65_acc += ((static_cast<float>(v66_data[5])) * v42_data);
              v65_acc += ((static_cast<float>(v66_data[6])) * v44_data);
              v65_acc += ((static_cast<float>(v66_data[7])) * v46_data);
              v65_acc += ((static_cast<float>(v66_data[8])) * v48_data);
              v65_acc += ((static_cast<float>(v66_data[9])) * v50_data);
              v65_acc += ((static_cast<float>(v66_data[10])) * v52_data);
              v65_acc += ((static_cast<float>(v66_data[11])) * v54_data);
              v65_acc += ((static_cast<float>(v66_data[12])) * v56_data);
              v65_acc += ((static_cast<float>(v66_data[13])) * v58_data);
              v65_acc += ((static_cast<float>(v66_data[14])) * v60_data);
              v65_acc += ((static_cast<float>(v66_data[15])) * v62_data);
              tensorforge::intel_esimd::simd<float, 16> v102_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v65_acc += ((static_cast<float>(v102_data[0])) * v64_data);
              ir0.template select<16, 1>(0) = v65_acc;
              tensorforge::intel_esimd::simd<float, 16> v105_acc{};
              tensorforge::intel_esimd::simd<float, 16> v107_data = tensorforge::slmLoad<float, 16>(s0 + (17_i32));
              v105_acc += ((static_cast<float>(v107_data[0])) * v32_data);
              v105_acc += ((static_cast<float>(v107_data[1])) * v34_data);
              v105_acc += ((static_cast<float>(v107_data[2])) * v36_data);
              v105_acc += ((static_cast<float>(v107_data[3])) * v38_data);
              v105_acc += ((static_cast<float>(v107_data[4])) * v40_data);
              v105_acc += ((static_cast<float>(v107_data[5])) * v42_data);
              v105_acc += ((static_cast<float>(v107_data[6])) * v44_data);
              v105_acc += ((static_cast<float>(v107_data[7])) * v46_data);
              v105_acc += ((static_cast<float>(v107_data[8])) * v48_data);
              v105_acc += ((static_cast<float>(v107_data[9])) * v50_data);
              v105_acc += ((static_cast<float>(v107_data[10])) * v52_data);
              v105_acc += ((static_cast<float>(v107_data[11])) * v54_data);
              v105_acc += ((static_cast<float>(v107_data[12])) * v56_data);
              v105_acc += ((static_cast<float>(v107_data[13])) * v58_data);
              v105_acc += ((static_cast<float>(v107_data[14])) * v60_data);
              v105_acc += ((static_cast<float>(v107_data[15])) * v62_data);
              tensorforge::intel_esimd::simd<float, 16> v141_data = tensorforge::slmLoad<float, 16>(s0 + (33_i32));
              v105_acc += ((static_cast<float>(v141_data[0])) * v64_data);
              ir0.template select<16, 1>(16) = v105_acc;
              tensorforge::intel_esimd::simd<float, 16> v144_acc{};
              tensorforge::intel_esimd::simd<float, 16> v146_data = tensorforge::slmLoad<float, 16>(s0 + (34_i32));
              v144_acc += ((static_cast<float>(v146_data[0])) * v32_data);
              v144_acc += ((static_cast<float>(v146_data[1])) * v34_data);
              v144_acc += ((static_cast<float>(v146_data[2])) * v36_data);
              v144_acc += ((static_cast<float>(v146_data[3])) * v38_data);
              v144_acc += ((static_cast<float>(v146_data[4])) * v40_data);
              v144_acc += ((static_cast<float>(v146_data[5])) * v42_data);
              v144_acc += ((static_cast<float>(v146_data[6])) * v44_data);
              v144_acc += ((static_cast<float>(v146_data[7])) * v46_data);
              v144_acc += ((static_cast<float>(v146_data[8])) * v48_data);
              v144_acc += ((static_cast<float>(v146_data[9])) * v50_data);
              v144_acc += ((static_cast<float>(v146_data[10])) * v52_data);
              v144_acc += ((static_cast<float>(v146_data[11])) * v54_data);
              v144_acc += ((static_cast<float>(v146_data[12])) * v56_data);
              v144_acc += ((static_cast<float>(v146_data[13])) * v58_data);
              v144_acc += ((static_cast<float>(v146_data[14])) * v60_data);
              v144_acc += ((static_cast<float>(v146_data[15])) * v62_data);
              tensorforge::intel_esimd::simd<float, 16> v180_data = tensorforge::slmLoad<float, 16>(s0 + (50_i32));
              v144_acc += ((static_cast<float>(v180_data[0])) * v64_data);
              ir0.template select<16, 1>(32) = v144_acc;
              tensorforge::intel_esimd::simd<float, 16> v183_acc{};
              tensorforge::intel_esimd::simd<float, 16> v185_data = tensorforge::slmLoad<float, 16>(s0 + (51_i32));
              v183_acc += ((static_cast<float>(v185_data[0])) * v32_data);
              v183_acc += ((static_cast<float>(v185_data[1])) * v34_data);
              v183_acc += ((static_cast<float>(v185_data[2])) * v36_data);
              v183_acc += ((static_cast<float>(v185_data[3])) * v38_data);
              v183_acc += ((static_cast<float>(v185_data[4])) * v40_data);
              v183_acc += ((static_cast<float>(v185_data[5])) * v42_data);
              v183_acc += ((static_cast<float>(v185_data[6])) * v44_data);
              v183_acc += ((static_cast<float>(v185_data[7])) * v46_data);
              v183_acc += ((static_cast<float>(v185_data[8])) * v48_data);
              v183_acc += ((static_cast<float>(v185_data[9])) * v50_data);
              v183_acc += ((static_cast<float>(v185_data[10])) * v52_data);
              v183_acc += ((static_cast<float>(v185_data[11])) * v54_data);
              v183_acc += ((static_cast<float>(v185_data[12])) * v56_data);
              v183_acc += ((static_cast<float>(v185_data[13])) * v58_data);
              v183_acc += ((static_cast<float>(v185_data[14])) * v60_data);
              v183_acc += ((static_cast<float>(v185_data[15])) * v62_data);
              tensorforge::intel_esimd::simd<float, 16> v219_data = tensorforge::slmLoad<float, 16>(s0 + (67_i32));
              v183_acc += ((static_cast<float>(v219_data[0])) * v64_data);
              ir0.template select<16, 1>(48) = v183_acc;
              tensorforge::intel_esimd::simd<float, 16> v222_acc{};
              tensorforge::intel_esimd::simd<float, 16> v224_data = tensorforge::slmLoad<float, 16>(s0 + (68_i32));
              v222_acc += ((static_cast<float>(v224_data[0])) * v32_data);
              v222_acc += ((static_cast<float>(v224_data[1])) * v34_data);
              v222_acc += ((static_cast<float>(v224_data[2])) * v36_data);
              v222_acc += ((static_cast<float>(v224_data[3])) * v38_data);
              v222_acc += ((static_cast<float>(v224_data[4])) * v40_data);
              v222_acc += ((static_cast<float>(v224_data[5])) * v42_data);
              v222_acc += ((static_cast<float>(v224_data[6])) * v44_data);
              v222_acc += ((static_cast<float>(v224_data[7])) * v46_data);
              v222_acc += ((static_cast<float>(v224_data[8])) * v48_data);
              v222_acc += ((static_cast<float>(v224_data[9])) * v50_data);
              v222_acc += ((static_cast<float>(v224_data[10])) * v52_data);
              v222_acc += ((static_cast<float>(v224_data[11])) * v54_data);
              v222_acc += ((static_cast<float>(v224_data[12])) * v56_data);
              v222_acc += ((static_cast<float>(v224_data[13])) * v58_data);
              v222_acc += ((static_cast<float>(v224_data[14])) * v60_data);
              v222_acc += ((static_cast<float>(v224_data[15])) * v62_data);
              tensorforge::intel_esimd::simd<float, 16> v258_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v222_acc += ((static_cast<float>(v258_data[0])) * v64_data);
              ir0.template select<16, 1>(64) = v222_acc;
              tensorforge::intel_esimd::simd<float, 16> v261_acc{};
              tensorforge::intel_esimd::simd<float, 16> v263_data = tensorforge::slmLoad<float, 16>(s0 + (85_i32));
              v261_acc += ((static_cast<float>(v263_data[0])) * v32_data);
              v261_acc += ((static_cast<float>(v263_data[1])) * v34_data);
              v261_acc += ((static_cast<float>(v263_data[2])) * v36_data);
              v261_acc += ((static_cast<float>(v263_data[3])) * v38_data);
              v261_acc += ((static_cast<float>(v263_data[4])) * v40_data);
              v261_acc += ((static_cast<float>(v263_data[5])) * v42_data);
              v261_acc += ((static_cast<float>(v263_data[6])) * v44_data);
              v261_acc += ((static_cast<float>(v263_data[7])) * v46_data);
              v261_acc += ((static_cast<float>(v263_data[8])) * v48_data);
              v261_acc += ((static_cast<float>(v263_data[9])) * v50_data);
              v261_acc += ((static_cast<float>(v263_data[10])) * v52_data);
              v261_acc += ((static_cast<float>(v263_data[11])) * v54_data);
              v261_acc += ((static_cast<float>(v263_data[12])) * v56_data);
              v261_acc += ((static_cast<float>(v263_data[13])) * v58_data);
              v261_acc += ((static_cast<float>(v263_data[14])) * v60_data);
              v261_acc += ((static_cast<float>(v263_data[15])) * v62_data);
              tensorforge::intel_esimd::simd<float, 16> v297_data = tensorforge::slmLoad<float, 16>(s0 + (101_i32));
              v261_acc += ((static_cast<float>(v297_data[0])) * v64_data);
              ir0.template select<16, 1>(80) = v261_acc;
              tensorforge::intel_esimd::simd<float, 16> v300_acc{};
              tensorforge::intel_esimd::simd<float, 16> v302_data = tensorforge::slmLoad<float, 16>(s0 + (102_i32));
              v300_acc += ((static_cast<float>(v302_data[0])) * v32_data);
              v300_acc += ((static_cast<float>(v302_data[1])) * v34_data);
              v300_acc += ((static_cast<float>(v302_data[2])) * v36_data);
              v300_acc += ((static_cast<float>(v302_data[3])) * v38_data);
              v300_acc += ((static_cast<float>(v302_data[4])) * v40_data);
              v300_acc += ((static_cast<float>(v302_data[5])) * v42_data);
              v300_acc += ((static_cast<float>(v302_data[6])) * v44_data);
              v300_acc += ((static_cast<float>(v302_data[7])) * v46_data);
              v300_acc += ((static_cast<float>(v302_data[8])) * v48_data);
              v300_acc += ((static_cast<float>(v302_data[9])) * v50_data);
              v300_acc += ((static_cast<float>(v302_data[10])) * v52_data);
              v300_acc += ((static_cast<float>(v302_data[11])) * v54_data);
              v300_acc += ((static_cast<float>(v302_data[12])) * v56_data);
              v300_acc += ((static_cast<float>(v302_data[13])) * v58_data);
              v300_acc += ((static_cast<float>(v302_data[14])) * v60_data);
              v300_acc += ((static_cast<float>(v302_data[15])) * v62_data);
              tensorforge::intel_esimd::simd<float, 16> v336_data = tensorforge::slmLoad<float, 16>(s0 + (118_i32));
              v300_acc += ((static_cast<float>(v336_data[0])) * v64_data);
              ir0.template select<16, 1>(96) = v300_acc;
              tensorforge::intel_esimd::simd<float, 16> v339_acc{};
              tensorforge::intel_esimd::simd<float, 16> v341_data = tensorforge::slmLoad<float, 16>(s0 + (119_i32));
              v339_acc += ((static_cast<float>(v341_data[0])) * v32_data);
              v339_acc += ((static_cast<float>(v341_data[1])) * v34_data);
              v339_acc += ((static_cast<float>(v341_data[2])) * v36_data);
              v339_acc += ((static_cast<float>(v341_data[3])) * v38_data);
              v339_acc += ((static_cast<float>(v341_data[4])) * v40_data);
              v339_acc += ((static_cast<float>(v341_data[5])) * v42_data);
              v339_acc += ((static_cast<float>(v341_data[6])) * v44_data);
              v339_acc += ((static_cast<float>(v341_data[7])) * v46_data);
              v339_acc += ((static_cast<float>(v341_data[8])) * v48_data);
              v339_acc += ((static_cast<float>(v341_data[9])) * v50_data);
              v339_acc += ((static_cast<float>(v341_data[10])) * v52_data);
              v339_acc += ((static_cast<float>(v341_data[11])) * v54_data);
              v339_acc += ((static_cast<float>(v341_data[12])) * v56_data);
              v339_acc += ((static_cast<float>(v341_data[13])) * v58_data);
              v339_acc += ((static_cast<float>(v341_data[14])) * v60_data);
              v339_acc += ((static_cast<float>(v341_data[15])) * v62_data);
              tensorforge::intel_esimd::simd<float, 16> v375_data = tensorforge::slmLoad<float, 16>(s0 + (135_i32));
              v339_acc += ((static_cast<float>(v375_data[0])) * v64_data);
              ir0.template select<16, 1>(112) = v339_acc;
              tensorforge::intel_esimd::simd<float, 16> v378_acc{};
              tensorforge::intel_esimd::simd<float, 16> v380_data = tensorforge::slmLoad<float, 16>(s0 + (136_i32));
              v378_acc += ((static_cast<float>(v380_data[0])) * v32_data);
              v378_acc += ((static_cast<float>(v380_data[1])) * v34_data);
              v378_acc += ((static_cast<float>(v380_data[2])) * v36_data);
              v378_acc += ((static_cast<float>(v380_data[3])) * v38_data);
              v378_acc += ((static_cast<float>(v380_data[4])) * v40_data);
              v378_acc += ((static_cast<float>(v380_data[5])) * v42_data);
              v378_acc += ((static_cast<float>(v380_data[6])) * v44_data);
              v378_acc += ((static_cast<float>(v380_data[7])) * v46_data);
              v378_acc += ((static_cast<float>(v380_data[8])) * v48_data);
              v378_acc += ((static_cast<float>(v380_data[9])) * v50_data);
              v378_acc += ((static_cast<float>(v380_data[10])) * v52_data);
              v378_acc += ((static_cast<float>(v380_data[11])) * v54_data);
              v378_acc += ((static_cast<float>(v380_data[12])) * v56_data);
              v378_acc += ((static_cast<float>(v380_data[13])) * v58_data);
              v378_acc += ((static_cast<float>(v380_data[14])) * v60_data);
              v378_acc += ((static_cast<float>(v380_data[15])) * v62_data);
              tensorforge::intel_esimd::simd<float, 16> v414_data = tensorforge::slmLoad<float, 16>(s0 + (152_i32));
              v378_acc += ((static_cast<float>(v414_data[0])) * v64_data);
              ir0.template select<16, 1>(128) = v378_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v417_n1 = 0; v417_n1 < 9; ++v417_n1) {
                int32_t v418_a = v417_n1 * 16;
                tensorforge::intel_esimd::simd<float, 10> v420_data(ir0.template select<10, 1>(v418_a));
                r0.template select<10, 1>(v418_a) = v420_data;
              }
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r1(0.0f);
              // ir1 = +(glb_m3 * s1)
              // [(0, 10), (0, 9)] [(0, 17)]
              tensorforge::intel_esimd::simd<float, 144> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v426_data;
              v426_data.copy_from(glb_m3 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v428_data;
              v428_data.copy_from(glb_m3 + (10_i32));
              tensorforge::intel_esimd::simd<float, 16> v430_data;
              v430_data.copy_from(glb_m3 + (20_i32));
              tensorforge::intel_esimd::simd<float, 16> v432_data;
              v432_data.copy_from(glb_m3 + (30_i32));
              tensorforge::intel_esimd::simd<float, 16> v434_data;
              v434_data.copy_from(glb_m3 + (40_i32));
              tensorforge::intel_esimd::simd<float, 16> v436_data;
              v436_data.copy_from(glb_m3 + (50_i32));
              tensorforge::intel_esimd::simd<float, 16> v438_data;
              v438_data.copy_from(glb_m3 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v440_data;
              v440_data.copy_from(glb_m3 + (70_i32));
              tensorforge::intel_esimd::simd<float, 16> v442_data;
              v442_data.copy_from(glb_m3 + (80_i32));
              tensorforge::intel_esimd::simd<float, 16> v444_data;
              v444_data.copy_from(glb_m3 + (90_i32));
              tensorforge::intel_esimd::simd<float, 16> v446_data;
              v446_data.copy_from(glb_m3 + (100_i32));
              tensorforge::intel_esimd::simd<float, 16> v448_data;
              v448_data.copy_from(glb_m3 + (110_i32));
              tensorforge::intel_esimd::simd<float, 16> v450_data;
              v450_data.copy_from(glb_m3 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v452_data;
              v452_data.copy_from(glb_m3 + (130_i32));
              tensorforge::intel_esimd::simd<float, 16> v454_data;
              v454_data.copy_from(glb_m3 + (140_i32));
              tensorforge::intel_esimd::simd<float, 16> v456_data;
              v456_data.copy_from(glb_m3 + (150_i32));
              tensorforge::intel_esimd::simd<float, 16> v458_data;
              v458_data.copy_from(glb_m3 + (160_i32));
              tensorforge::intel_esimd::simd<float, 16> v459_acc{};
              tensorforge::intel_esimd::simd<float, 16> v460_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v459_acc += ((static_cast<float>(v460_data[0])) * v426_data);
              v459_acc += ((static_cast<float>(v460_data[1])) * v428_data);
              v459_acc += ((static_cast<float>(v460_data[2])) * v430_data);
              v459_acc += ((static_cast<float>(v460_data[3])) * v432_data);
              v459_acc += ((static_cast<float>(v460_data[4])) * v434_data);
              v459_acc += ((static_cast<float>(v460_data[5])) * v436_data);
              v459_acc += ((static_cast<float>(v460_data[6])) * v438_data);
              v459_acc += ((static_cast<float>(v460_data[7])) * v440_data);
              v459_acc += ((static_cast<float>(v460_data[8])) * v442_data);
              v459_acc += ((static_cast<float>(v460_data[9])) * v444_data);
              v459_acc += ((static_cast<float>(v460_data[10])) * v446_data);
              v459_acc += ((static_cast<float>(v460_data[11])) * v448_data);
              v459_acc += ((static_cast<float>(v460_data[12])) * v450_data);
              v459_acc += ((static_cast<float>(v460_data[13])) * v452_data);
              v459_acc += ((static_cast<float>(v460_data[14])) * v454_data);
              v459_acc += ((static_cast<float>(v460_data[15])) * v456_data);
              tensorforge::intel_esimd::simd<float, 16> v496_data = tensorforge::slmLoad<float, 16>(s1 + (16_i32));
              v459_acc += ((static_cast<float>(v496_data[0])) * v458_data);
              ir1.template select<16, 1>(0) = v459_acc;
              tensorforge::intel_esimd::simd<float, 16> v499_acc{};
              tensorforge::intel_esimd::simd<float, 16> v501_data = tensorforge::slmLoad<float, 16>(s1 + (17_i32));
              v499_acc += ((static_cast<float>(v501_data[0])) * v426_data);
              v499_acc += ((static_cast<float>(v501_data[1])) * v428_data);
              v499_acc += ((static_cast<float>(v501_data[2])) * v430_data);
              v499_acc += ((static_cast<float>(v501_data[3])) * v432_data);
              v499_acc += ((static_cast<float>(v501_data[4])) * v434_data);
              v499_acc += ((static_cast<float>(v501_data[5])) * v436_data);
              v499_acc += ((static_cast<float>(v501_data[6])) * v438_data);
              v499_acc += ((static_cast<float>(v501_data[7])) * v440_data);
              v499_acc += ((static_cast<float>(v501_data[8])) * v442_data);
              v499_acc += ((static_cast<float>(v501_data[9])) * v444_data);
              v499_acc += ((static_cast<float>(v501_data[10])) * v446_data);
              v499_acc += ((static_cast<float>(v501_data[11])) * v448_data);
              v499_acc += ((static_cast<float>(v501_data[12])) * v450_data);
              v499_acc += ((static_cast<float>(v501_data[13])) * v452_data);
              v499_acc += ((static_cast<float>(v501_data[14])) * v454_data);
              v499_acc += ((static_cast<float>(v501_data[15])) * v456_data);
              tensorforge::intel_esimd::simd<float, 16> v535_data = tensorforge::slmLoad<float, 16>(s1 + (33_i32));
              v499_acc += ((static_cast<float>(v535_data[0])) * v458_data);
              ir1.template select<16, 1>(16) = v499_acc;
              tensorforge::intel_esimd::simd<float, 16> v538_acc{};
              tensorforge::intel_esimd::simd<float, 16> v540_data = tensorforge::slmLoad<float, 16>(s1 + (34_i32));
              v538_acc += ((static_cast<float>(v540_data[0])) * v426_data);
              v538_acc += ((static_cast<float>(v540_data[1])) * v428_data);
              v538_acc += ((static_cast<float>(v540_data[2])) * v430_data);
              v538_acc += ((static_cast<float>(v540_data[3])) * v432_data);
              v538_acc += ((static_cast<float>(v540_data[4])) * v434_data);
              v538_acc += ((static_cast<float>(v540_data[5])) * v436_data);
              v538_acc += ((static_cast<float>(v540_data[6])) * v438_data);
              v538_acc += ((static_cast<float>(v540_data[7])) * v440_data);
              v538_acc += ((static_cast<float>(v540_data[8])) * v442_data);
              v538_acc += ((static_cast<float>(v540_data[9])) * v444_data);
              v538_acc += ((static_cast<float>(v540_data[10])) * v446_data);
              v538_acc += ((static_cast<float>(v540_data[11])) * v448_data);
              v538_acc += ((static_cast<float>(v540_data[12])) * v450_data);
              v538_acc += ((static_cast<float>(v540_data[13])) * v452_data);
              v538_acc += ((static_cast<float>(v540_data[14])) * v454_data);
              v538_acc += ((static_cast<float>(v540_data[15])) * v456_data);
              tensorforge::intel_esimd::simd<float, 16> v574_data = tensorforge::slmLoad<float, 16>(s1 + (50_i32));
              v538_acc += ((static_cast<float>(v574_data[0])) * v458_data);
              ir1.template select<16, 1>(32) = v538_acc;
              tensorforge::intel_esimd::simd<float, 16> v577_acc{};
              tensorforge::intel_esimd::simd<float, 16> v579_data = tensorforge::slmLoad<float, 16>(s1 + (51_i32));
              v577_acc += ((static_cast<float>(v579_data[0])) * v426_data);
              v577_acc += ((static_cast<float>(v579_data[1])) * v428_data);
              v577_acc += ((static_cast<float>(v579_data[2])) * v430_data);
              v577_acc += ((static_cast<float>(v579_data[3])) * v432_data);
              v577_acc += ((static_cast<float>(v579_data[4])) * v434_data);
              v577_acc += ((static_cast<float>(v579_data[5])) * v436_data);
              v577_acc += ((static_cast<float>(v579_data[6])) * v438_data);
              v577_acc += ((static_cast<float>(v579_data[7])) * v440_data);
              v577_acc += ((static_cast<float>(v579_data[8])) * v442_data);
              v577_acc += ((static_cast<float>(v579_data[9])) * v444_data);
              v577_acc += ((static_cast<float>(v579_data[10])) * v446_data);
              v577_acc += ((static_cast<float>(v579_data[11])) * v448_data);
              v577_acc += ((static_cast<float>(v579_data[12])) * v450_data);
              v577_acc += ((static_cast<float>(v579_data[13])) * v452_data);
              v577_acc += ((static_cast<float>(v579_data[14])) * v454_data);
              v577_acc += ((static_cast<float>(v579_data[15])) * v456_data);
              tensorforge::intel_esimd::simd<float, 16> v613_data = tensorforge::slmLoad<float, 16>(s1 + (67_i32));
              v577_acc += ((static_cast<float>(v613_data[0])) * v458_data);
              ir1.template select<16, 1>(48) = v577_acc;
              tensorforge::intel_esimd::simd<float, 16> v616_acc{};
              tensorforge::intel_esimd::simd<float, 16> v618_data = tensorforge::slmLoad<float, 16>(s1 + (68_i32));
              v616_acc += ((static_cast<float>(v618_data[0])) * v426_data);
              v616_acc += ((static_cast<float>(v618_data[1])) * v428_data);
              v616_acc += ((static_cast<float>(v618_data[2])) * v430_data);
              v616_acc += ((static_cast<float>(v618_data[3])) * v432_data);
              v616_acc += ((static_cast<float>(v618_data[4])) * v434_data);
              v616_acc += ((static_cast<float>(v618_data[5])) * v436_data);
              v616_acc += ((static_cast<float>(v618_data[6])) * v438_data);
              v616_acc += ((static_cast<float>(v618_data[7])) * v440_data);
              v616_acc += ((static_cast<float>(v618_data[8])) * v442_data);
              v616_acc += ((static_cast<float>(v618_data[9])) * v444_data);
              v616_acc += ((static_cast<float>(v618_data[10])) * v446_data);
              v616_acc += ((static_cast<float>(v618_data[11])) * v448_data);
              v616_acc += ((static_cast<float>(v618_data[12])) * v450_data);
              v616_acc += ((static_cast<float>(v618_data[13])) * v452_data);
              v616_acc += ((static_cast<float>(v618_data[14])) * v454_data);
              v616_acc += ((static_cast<float>(v618_data[15])) * v456_data);
              tensorforge::intel_esimd::simd<float, 16> v652_data = tensorforge::slmLoad<float, 16>(s1 + (84_i32));
              v616_acc += ((static_cast<float>(v652_data[0])) * v458_data);
              ir1.template select<16, 1>(64) = v616_acc;
              tensorforge::intel_esimd::simd<float, 16> v655_acc{};
              tensorforge::intel_esimd::simd<float, 16> v657_data = tensorforge::slmLoad<float, 16>(s1 + (85_i32));
              v655_acc += ((static_cast<float>(v657_data[0])) * v426_data);
              v655_acc += ((static_cast<float>(v657_data[1])) * v428_data);
              v655_acc += ((static_cast<float>(v657_data[2])) * v430_data);
              v655_acc += ((static_cast<float>(v657_data[3])) * v432_data);
              v655_acc += ((static_cast<float>(v657_data[4])) * v434_data);
              v655_acc += ((static_cast<float>(v657_data[5])) * v436_data);
              v655_acc += ((static_cast<float>(v657_data[6])) * v438_data);
              v655_acc += ((static_cast<float>(v657_data[7])) * v440_data);
              v655_acc += ((static_cast<float>(v657_data[8])) * v442_data);
              v655_acc += ((static_cast<float>(v657_data[9])) * v444_data);
              v655_acc += ((static_cast<float>(v657_data[10])) * v446_data);
              v655_acc += ((static_cast<float>(v657_data[11])) * v448_data);
              v655_acc += ((static_cast<float>(v657_data[12])) * v450_data);
              v655_acc += ((static_cast<float>(v657_data[13])) * v452_data);
              v655_acc += ((static_cast<float>(v657_data[14])) * v454_data);
              v655_acc += ((static_cast<float>(v657_data[15])) * v456_data);
              tensorforge::intel_esimd::simd<float, 16> v691_data = tensorforge::slmLoad<float, 16>(s1 + (101_i32));
              v655_acc += ((static_cast<float>(v691_data[0])) * v458_data);
              ir1.template select<16, 1>(80) = v655_acc;
              tensorforge::intel_esimd::simd<float, 16> v694_acc{};
              tensorforge::intel_esimd::simd<float, 16> v696_data = tensorforge::slmLoad<float, 16>(s1 + (102_i32));
              v694_acc += ((static_cast<float>(v696_data[0])) * v426_data);
              v694_acc += ((static_cast<float>(v696_data[1])) * v428_data);
              v694_acc += ((static_cast<float>(v696_data[2])) * v430_data);
              v694_acc += ((static_cast<float>(v696_data[3])) * v432_data);
              v694_acc += ((static_cast<float>(v696_data[4])) * v434_data);
              v694_acc += ((static_cast<float>(v696_data[5])) * v436_data);
              v694_acc += ((static_cast<float>(v696_data[6])) * v438_data);
              v694_acc += ((static_cast<float>(v696_data[7])) * v440_data);
              v694_acc += ((static_cast<float>(v696_data[8])) * v442_data);
              v694_acc += ((static_cast<float>(v696_data[9])) * v444_data);
              v694_acc += ((static_cast<float>(v696_data[10])) * v446_data);
              v694_acc += ((static_cast<float>(v696_data[11])) * v448_data);
              v694_acc += ((static_cast<float>(v696_data[12])) * v450_data);
              v694_acc += ((static_cast<float>(v696_data[13])) * v452_data);
              v694_acc += ((static_cast<float>(v696_data[14])) * v454_data);
              v694_acc += ((static_cast<float>(v696_data[15])) * v456_data);
              tensorforge::intel_esimd::simd<float, 16> v730_data = tensorforge::slmLoad<float, 16>(s1 + (118_i32));
              v694_acc += ((static_cast<float>(v730_data[0])) * v458_data);
              ir1.template select<16, 1>(96) = v694_acc;
              tensorforge::intel_esimd::simd<float, 16> v733_acc{};
              tensorforge::intel_esimd::simd<float, 16> v735_data = tensorforge::slmLoad<float, 16>(s1 + (119_i32));
              v733_acc += ((static_cast<float>(v735_data[0])) * v426_data);
              v733_acc += ((static_cast<float>(v735_data[1])) * v428_data);
              v733_acc += ((static_cast<float>(v735_data[2])) * v430_data);
              v733_acc += ((static_cast<float>(v735_data[3])) * v432_data);
              v733_acc += ((static_cast<float>(v735_data[4])) * v434_data);
              v733_acc += ((static_cast<float>(v735_data[5])) * v436_data);
              v733_acc += ((static_cast<float>(v735_data[6])) * v438_data);
              v733_acc += ((static_cast<float>(v735_data[7])) * v440_data);
              v733_acc += ((static_cast<float>(v735_data[8])) * v442_data);
              v733_acc += ((static_cast<float>(v735_data[9])) * v444_data);
              v733_acc += ((static_cast<float>(v735_data[10])) * v446_data);
              v733_acc += ((static_cast<float>(v735_data[11])) * v448_data);
              v733_acc += ((static_cast<float>(v735_data[12])) * v450_data);
              v733_acc += ((static_cast<float>(v735_data[13])) * v452_data);
              v733_acc += ((static_cast<float>(v735_data[14])) * v454_data);
              v733_acc += ((static_cast<float>(v735_data[15])) * v456_data);
              tensorforge::intel_esimd::simd<float, 16> v769_data = tensorforge::slmLoad<float, 16>(s1 + (135_i32));
              v733_acc += ((static_cast<float>(v769_data[0])) * v458_data);
              ir1.template select<16, 1>(112) = v733_acc;
              tensorforge::intel_esimd::simd<float, 16> v772_acc{};
              tensorforge::intel_esimd::simd<float, 16> v774_data = tensorforge::slmLoad<float, 16>(s1 + (136_i32));
              v772_acc += ((static_cast<float>(v774_data[0])) * v426_data);
              v772_acc += ((static_cast<float>(v774_data[1])) * v428_data);
              v772_acc += ((static_cast<float>(v774_data[2])) * v430_data);
              v772_acc += ((static_cast<float>(v774_data[3])) * v432_data);
              v772_acc += ((static_cast<float>(v774_data[4])) * v434_data);
              v772_acc += ((static_cast<float>(v774_data[5])) * v436_data);
              v772_acc += ((static_cast<float>(v774_data[6])) * v438_data);
              v772_acc += ((static_cast<float>(v774_data[7])) * v440_data);
              v772_acc += ((static_cast<float>(v774_data[8])) * v442_data);
              v772_acc += ((static_cast<float>(v774_data[9])) * v444_data);
              v772_acc += ((static_cast<float>(v774_data[10])) * v446_data);
              v772_acc += ((static_cast<float>(v774_data[11])) * v448_data);
              v772_acc += ((static_cast<float>(v774_data[12])) * v450_data);
              v772_acc += ((static_cast<float>(v774_data[13])) * v452_data);
              v772_acc += ((static_cast<float>(v774_data[14])) * v454_data);
              v772_acc += ((static_cast<float>(v774_data[15])) * v456_data);
              tensorforge::intel_esimd::simd<float, 16> v808_data = tensorforge::slmLoad<float, 16>(s1 + (152_i32));
              v772_acc += ((static_cast<float>(v808_data[0])) * v458_data);
              ir1.template select<16, 1>(128) = v772_acc;
              // r1 = ir1 + r0
              #pragma unroll
              for (int32_t v811_n1 = 0; v811_n1 < 9; ++v811_n1) {
                int32_t v812_a = v811_n1 * 16;
                tensorforge::intel_esimd::simd<float, 10> v814_data(ir1.template select<10, 1>(v812_a));
                tensorforge::intel_esimd::simd<float, 10> v815_data(r0.template select<10, 1>(v812_a));
                r1.template select<10, 1>(v812_a) = (v815_data + v814_data);
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v817_i1 = 0; v817_i1 < 9; ++v817_i1) {
                tensorforge::intel_esimd::simd<float, 10> v820_data(r1.template select<10, 1>((v817_i1 * 16)));
                v820_data.copy_to(glb_m0 + ((v817_i1 * 10)));
              }
            }
          }
        }
      }
    });
  });
}

