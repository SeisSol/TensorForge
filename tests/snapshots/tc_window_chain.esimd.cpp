// === base name ===
kernel_8dfeac4e36f2d521

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_8dfeac4e36f2d521 = {{1, 16, 1}, 16, 16, 1, 16, 11264, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_8dfeac4e36f2d521(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_8dfeac4e36f2d521(const float * m0, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_8dfeac4e36f2d521(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_8dfeac4e36f2d521(const float * m0, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_8dfeac4e36f2d521(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_8dfeac4e36f2d521(stream, grid, block, m0, m1, m1_extraOffset, m2, m2_extraOffset, m3, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_8dfeac4e36f2d521(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2816 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 11264 B shared, occupancy grid
        // operands:
        //   m0 16×20(16×17) {0..16}×{1..18} none
        //   m1 20×9(17×9) {1..18}×{0..9} strided
        //   m2 16×9(16×9) {0..16}×{0..9} strided
        //   m3 16×20(16×15) {0..16}×{1..16} none
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   m2[i,j] = m3[i,k] × t0[k,j]@{1..16}×{0..9}
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2816}],"shared_bytes":11264,"shared_elements":2816,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"none","alias":"A1","bbox":[[0,1],[16,18]],"name":"m0","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[1,0],[18,9]],"name":"m1","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[16,9]],"name":"m2","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"none","alias":"A2","bbox":[[0,1],[16,16]],"name":"m3","ordered":false,"parts":1,"shape":[16,20],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[16,18]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[18,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[16,16]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,20]},{"addressing":"pointer_based","bbox":[[1,0],[16,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[16,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (176 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (160);
          const float *const __restrict__ glb_m0 = &m0[0];
          const float *const __restrict__ glb_m3 = &m3[0];
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const float *const __restrict__ pf_glb_m1 = &m1[v11_batchId1 * 153 + 0 + m1_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 153 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 144 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v19_ld;
              v19_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v19_ld);
              tensorforge::intel_esimd::simd<float, 64> v20_ld;
              v20_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v20_ld);
              tensorforge::intel_esimd::simd<float, 16> v21_ld;
              v21_ld.copy_from(glb_m1 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v21_ld);
              tensorforge::intel_esimd::simd<float, 9> v22_ld;
              v22_ld.copy_from(glb_m1 + (0 + 0 + 1 * 0 + 144));
              tensorforge::slmStore<float, 9>(s0 + (0 + 0 + 1 * 0 + 144), v22_ld);
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r0(0.0f);
              // r0 = +(glb_m0 * s0) + None
              // [(0, 16), (0, 9)] [(1, 18)]
              tensorforge::intel_esimd::simd<float, 64> glb_m0_run0;
              glb_m0_run0.copy_from(glb_m0 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v27_data(glb_m0_run0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v29_data(glb_m0_run0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v31_data(glb_m0_run0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v33_data(glb_m0_run0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m0_run1;
              glb_m0_run1.copy_from(glb_m0 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v35_data(glb_m0_run1.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v37_data(glb_m0_run1.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v39_data(glb_m0_run1.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v41_data(glb_m0_run1.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m0_run2;
              glb_m0_run2.copy_from(glb_m0 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v43_data(glb_m0_run2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v45_data(glb_m0_run2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v47_data(glb_m0_run2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v49_data(glb_m0_run2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m0_run3;
              glb_m0_run3.copy_from(glb_m0 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v51_data(glb_m0_run3.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v53_data(glb_m0_run3.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v55_data(glb_m0_run3.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v57_data(glb_m0_run3.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v59_data;
              v59_data.copy_from(glb_m0 + (256_i32));
              tensorforge::intel_esimd::simd<float, 16> v61_acc{};
              tensorforge::intel_esimd::simd<float, 16> v64_data(0.0f);
              v64_data.template select<15, 1>(1) = tensorforge::slmLoad<float, 15>((s0 + (-1_i32)) + 1);
              v61_acc += ((static_cast<float>(v64_data[0])) * v27_data);
              v61_acc += ((static_cast<float>(v64_data[1])) * v29_data);
              v61_acc += ((static_cast<float>(v64_data[2])) * v31_data);
              v61_acc += ((static_cast<float>(v64_data[3])) * v33_data);
              v61_acc += ((static_cast<float>(v64_data[4])) * v35_data);
              v61_acc += ((static_cast<float>(v64_data[5])) * v37_data);
              v61_acc += ((static_cast<float>(v64_data[6])) * v39_data);
              v61_acc += ((static_cast<float>(v64_data[7])) * v41_data);
              v61_acc += ((static_cast<float>(v64_data[8])) * v43_data);
              v61_acc += ((static_cast<float>(v64_data[9])) * v45_data);
              v61_acc += ((static_cast<float>(v64_data[10])) * v47_data);
              v61_acc += ((static_cast<float>(v64_data[11])) * v49_data);
              v61_acc += ((static_cast<float>(v64_data[12])) * v51_data);
              v61_acc += ((static_cast<float>(v64_data[13])) * v53_data);
              v61_acc += ((static_cast<float>(v64_data[14])) * v55_data);
              v61_acc += ((static_cast<float>(v64_data[15])) * v57_data);
              tensorforge::intel_esimd::simd<float, 16> v101_data = tensorforge::slmLoad<float, 16>(s0 + (15_i32));
              v61_acc += ((static_cast<float>(v101_data[0])) * v59_data);
              v61_acc += ((static_cast<float>(v101_data[1])) * v27_data);
              r0.template select<16, 1>(0) = v61_acc;
              tensorforge::intel_esimd::simd<float, 16> v106_acc{};
              tensorforge::intel_esimd::simd<float, 16> v108_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v106_acc += ((static_cast<float>(v108_data[0])) * v27_data);
              v106_acc += ((static_cast<float>(v108_data[1])) * v29_data);
              v106_acc += ((static_cast<float>(v108_data[2])) * v31_data);
              v106_acc += ((static_cast<float>(v108_data[3])) * v33_data);
              v106_acc += ((static_cast<float>(v108_data[4])) * v35_data);
              v106_acc += ((static_cast<float>(v108_data[5])) * v37_data);
              v106_acc += ((static_cast<float>(v108_data[6])) * v39_data);
              v106_acc += ((static_cast<float>(v108_data[7])) * v41_data);
              v106_acc += ((static_cast<float>(v108_data[8])) * v43_data);
              v106_acc += ((static_cast<float>(v108_data[9])) * v45_data);
              v106_acc += ((static_cast<float>(v108_data[10])) * v47_data);
              v106_acc += ((static_cast<float>(v108_data[11])) * v49_data);
              v106_acc += ((static_cast<float>(v108_data[12])) * v51_data);
              v106_acc += ((static_cast<float>(v108_data[13])) * v53_data);
              v106_acc += ((static_cast<float>(v108_data[14])) * v55_data);
              v106_acc += ((static_cast<float>(v108_data[15])) * v57_data);
              tensorforge::intel_esimd::simd<float, 16> v142_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v106_acc += ((static_cast<float>(v142_data[0])) * v59_data);
              v106_acc += ((static_cast<float>(v142_data[1])) * v27_data);
              r0.template select<16, 1>(16) = v106_acc;
              tensorforge::intel_esimd::simd<float, 16> v147_acc{};
              tensorforge::intel_esimd::simd<float, 16> v149_data = tensorforge::slmLoad<float, 16>(s0 + (33_i32));
              v147_acc += ((static_cast<float>(v149_data[0])) * v27_data);
              v147_acc += ((static_cast<float>(v149_data[1])) * v29_data);
              v147_acc += ((static_cast<float>(v149_data[2])) * v31_data);
              v147_acc += ((static_cast<float>(v149_data[3])) * v33_data);
              v147_acc += ((static_cast<float>(v149_data[4])) * v35_data);
              v147_acc += ((static_cast<float>(v149_data[5])) * v37_data);
              v147_acc += ((static_cast<float>(v149_data[6])) * v39_data);
              v147_acc += ((static_cast<float>(v149_data[7])) * v41_data);
              v147_acc += ((static_cast<float>(v149_data[8])) * v43_data);
              v147_acc += ((static_cast<float>(v149_data[9])) * v45_data);
              v147_acc += ((static_cast<float>(v149_data[10])) * v47_data);
              v147_acc += ((static_cast<float>(v149_data[11])) * v49_data);
              v147_acc += ((static_cast<float>(v149_data[12])) * v51_data);
              v147_acc += ((static_cast<float>(v149_data[13])) * v53_data);
              v147_acc += ((static_cast<float>(v149_data[14])) * v55_data);
              v147_acc += ((static_cast<float>(v149_data[15])) * v57_data);
              tensorforge::intel_esimd::simd<float, 16> v183_data = tensorforge::slmLoad<float, 16>(s0 + (49_i32));
              v147_acc += ((static_cast<float>(v183_data[0])) * v59_data);
              v147_acc += ((static_cast<float>(v183_data[1])) * v27_data);
              r0.template select<16, 1>(32) = v147_acc;
              tensorforge::intel_esimd::simd<float, 16> v188_acc{};
              tensorforge::intel_esimd::simd<float, 16> v190_data = tensorforge::slmLoad<float, 16>(s0 + (50_i32));
              v188_acc += ((static_cast<float>(v190_data[0])) * v27_data);
              v188_acc += ((static_cast<float>(v190_data[1])) * v29_data);
              v188_acc += ((static_cast<float>(v190_data[2])) * v31_data);
              v188_acc += ((static_cast<float>(v190_data[3])) * v33_data);
              v188_acc += ((static_cast<float>(v190_data[4])) * v35_data);
              v188_acc += ((static_cast<float>(v190_data[5])) * v37_data);
              v188_acc += ((static_cast<float>(v190_data[6])) * v39_data);
              v188_acc += ((static_cast<float>(v190_data[7])) * v41_data);
              v188_acc += ((static_cast<float>(v190_data[8])) * v43_data);
              v188_acc += ((static_cast<float>(v190_data[9])) * v45_data);
              v188_acc += ((static_cast<float>(v190_data[10])) * v47_data);
              v188_acc += ((static_cast<float>(v190_data[11])) * v49_data);
              v188_acc += ((static_cast<float>(v190_data[12])) * v51_data);
              v188_acc += ((static_cast<float>(v190_data[13])) * v53_data);
              v188_acc += ((static_cast<float>(v190_data[14])) * v55_data);
              v188_acc += ((static_cast<float>(v190_data[15])) * v57_data);
              tensorforge::intel_esimd::simd<float, 16> v224_data = tensorforge::slmLoad<float, 16>(s0 + (66_i32));
              v188_acc += ((static_cast<float>(v224_data[0])) * v59_data);
              v188_acc += ((static_cast<float>(v224_data[1])) * v27_data);
              r0.template select<16, 1>(48) = v188_acc;
              tensorforge::intel_esimd::simd<float, 16> v229_acc{};
              tensorforge::intel_esimd::simd<float, 16> v231_data = tensorforge::slmLoad<float, 16>(s0 + (67_i32));
              v229_acc += ((static_cast<float>(v231_data[0])) * v27_data);
              v229_acc += ((static_cast<float>(v231_data[1])) * v29_data);
              v229_acc += ((static_cast<float>(v231_data[2])) * v31_data);
              v229_acc += ((static_cast<float>(v231_data[3])) * v33_data);
              v229_acc += ((static_cast<float>(v231_data[4])) * v35_data);
              v229_acc += ((static_cast<float>(v231_data[5])) * v37_data);
              v229_acc += ((static_cast<float>(v231_data[6])) * v39_data);
              v229_acc += ((static_cast<float>(v231_data[7])) * v41_data);
              v229_acc += ((static_cast<float>(v231_data[8])) * v43_data);
              v229_acc += ((static_cast<float>(v231_data[9])) * v45_data);
              v229_acc += ((static_cast<float>(v231_data[10])) * v47_data);
              v229_acc += ((static_cast<float>(v231_data[11])) * v49_data);
              v229_acc += ((static_cast<float>(v231_data[12])) * v51_data);
              v229_acc += ((static_cast<float>(v231_data[13])) * v53_data);
              v229_acc += ((static_cast<float>(v231_data[14])) * v55_data);
              v229_acc += ((static_cast<float>(v231_data[15])) * v57_data);
              tensorforge::intel_esimd::simd<float, 16> v265_data = tensorforge::slmLoad<float, 16>(s0 + (83_i32));
              v229_acc += ((static_cast<float>(v265_data[0])) * v59_data);
              v229_acc += ((static_cast<float>(v265_data[1])) * v27_data);
              r0.template select<16, 1>(64) = v229_acc;
              tensorforge::intel_esimd::simd<float, 16> v270_acc{};
              tensorforge::intel_esimd::simd<float, 16> v272_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v270_acc += ((static_cast<float>(v272_data[0])) * v27_data);
              v270_acc += ((static_cast<float>(v272_data[1])) * v29_data);
              v270_acc += ((static_cast<float>(v272_data[2])) * v31_data);
              v270_acc += ((static_cast<float>(v272_data[3])) * v33_data);
              v270_acc += ((static_cast<float>(v272_data[4])) * v35_data);
              v270_acc += ((static_cast<float>(v272_data[5])) * v37_data);
              v270_acc += ((static_cast<float>(v272_data[6])) * v39_data);
              v270_acc += ((static_cast<float>(v272_data[7])) * v41_data);
              v270_acc += ((static_cast<float>(v272_data[8])) * v43_data);
              v270_acc += ((static_cast<float>(v272_data[9])) * v45_data);
              v270_acc += ((static_cast<float>(v272_data[10])) * v47_data);
              v270_acc += ((static_cast<float>(v272_data[11])) * v49_data);
              v270_acc += ((static_cast<float>(v272_data[12])) * v51_data);
              v270_acc += ((static_cast<float>(v272_data[13])) * v53_data);
              v270_acc += ((static_cast<float>(v272_data[14])) * v55_data);
              v270_acc += ((static_cast<float>(v272_data[15])) * v57_data);
              tensorforge::intel_esimd::simd<float, 16> v306_data = tensorforge::slmLoad<float, 16>(s0 + (100_i32));
              v270_acc += ((static_cast<float>(v306_data[0])) * v59_data);
              v270_acc += ((static_cast<float>(v306_data[1])) * v27_data);
              r0.template select<16, 1>(80) = v270_acc;
              tensorforge::intel_esimd::simd<float, 16> v311_acc{};
              tensorforge::intel_esimd::simd<float, 16> v313_data = tensorforge::slmLoad<float, 16>(s0 + (101_i32));
              v311_acc += ((static_cast<float>(v313_data[0])) * v27_data);
              v311_acc += ((static_cast<float>(v313_data[1])) * v29_data);
              v311_acc += ((static_cast<float>(v313_data[2])) * v31_data);
              v311_acc += ((static_cast<float>(v313_data[3])) * v33_data);
              v311_acc += ((static_cast<float>(v313_data[4])) * v35_data);
              v311_acc += ((static_cast<float>(v313_data[5])) * v37_data);
              v311_acc += ((static_cast<float>(v313_data[6])) * v39_data);
              v311_acc += ((static_cast<float>(v313_data[7])) * v41_data);
              v311_acc += ((static_cast<float>(v313_data[8])) * v43_data);
              v311_acc += ((static_cast<float>(v313_data[9])) * v45_data);
              v311_acc += ((static_cast<float>(v313_data[10])) * v47_data);
              v311_acc += ((static_cast<float>(v313_data[11])) * v49_data);
              v311_acc += ((static_cast<float>(v313_data[12])) * v51_data);
              v311_acc += ((static_cast<float>(v313_data[13])) * v53_data);
              v311_acc += ((static_cast<float>(v313_data[14])) * v55_data);
              v311_acc += ((static_cast<float>(v313_data[15])) * v57_data);
              tensorforge::intel_esimd::simd<float, 16> v347_data = tensorforge::slmLoad<float, 16>(s0 + (117_i32));
              v311_acc += ((static_cast<float>(v347_data[0])) * v59_data);
              v311_acc += ((static_cast<float>(v347_data[1])) * v27_data);
              r0.template select<16, 1>(96) = v311_acc;
              tensorforge::intel_esimd::simd<float, 16> v352_acc{};
              tensorforge::intel_esimd::simd<float, 16> v354_data = tensorforge::slmLoad<float, 16>(s0 + (118_i32));
              v352_acc += ((static_cast<float>(v354_data[0])) * v27_data);
              v352_acc += ((static_cast<float>(v354_data[1])) * v29_data);
              v352_acc += ((static_cast<float>(v354_data[2])) * v31_data);
              v352_acc += ((static_cast<float>(v354_data[3])) * v33_data);
              v352_acc += ((static_cast<float>(v354_data[4])) * v35_data);
              v352_acc += ((static_cast<float>(v354_data[5])) * v37_data);
              v352_acc += ((static_cast<float>(v354_data[6])) * v39_data);
              v352_acc += ((static_cast<float>(v354_data[7])) * v41_data);
              v352_acc += ((static_cast<float>(v354_data[8])) * v43_data);
              v352_acc += ((static_cast<float>(v354_data[9])) * v45_data);
              v352_acc += ((static_cast<float>(v354_data[10])) * v47_data);
              v352_acc += ((static_cast<float>(v354_data[11])) * v49_data);
              v352_acc += ((static_cast<float>(v354_data[12])) * v51_data);
              v352_acc += ((static_cast<float>(v354_data[13])) * v53_data);
              v352_acc += ((static_cast<float>(v354_data[14])) * v55_data);
              v352_acc += ((static_cast<float>(v354_data[15])) * v57_data);
              tensorforge::intel_esimd::simd<float, 16> v388_data = tensorforge::slmLoad<float, 16>(s0 + (134_i32));
              v352_acc += ((static_cast<float>(v388_data[0])) * v59_data);
              v352_acc += ((static_cast<float>(v388_data[1])) * v27_data);
              r0.template select<16, 1>(112) = v352_acc;
              tensorforge::intel_esimd::simd<float, 16> v393_acc{};
              tensorforge::intel_esimd::simd<float, 16> v395_data = tensorforge::slmLoad<float, 16>(s0 + (135_i32));
              v393_acc += ((static_cast<float>(v395_data[0])) * v27_data);
              v393_acc += ((static_cast<float>(v395_data[1])) * v29_data);
              v393_acc += ((static_cast<float>(v395_data[2])) * v31_data);
              v393_acc += ((static_cast<float>(v395_data[3])) * v33_data);
              v393_acc += ((static_cast<float>(v395_data[4])) * v35_data);
              v393_acc += ((static_cast<float>(v395_data[5])) * v37_data);
              v393_acc += ((static_cast<float>(v395_data[6])) * v39_data);
              v393_acc += ((static_cast<float>(v395_data[7])) * v41_data);
              v393_acc += ((static_cast<float>(v395_data[8])) * v43_data);
              v393_acc += ((static_cast<float>(v395_data[9])) * v45_data);
              v393_acc += ((static_cast<float>(v395_data[10])) * v47_data);
              v393_acc += ((static_cast<float>(v395_data[11])) * v49_data);
              v393_acc += ((static_cast<float>(v395_data[12])) * v51_data);
              v393_acc += ((static_cast<float>(v395_data[13])) * v53_data);
              v393_acc += ((static_cast<float>(v395_data[14])) * v55_data);
              v393_acc += ((static_cast<float>(v395_data[15])) * v57_data);
              tensorforge::intel_esimd::simd<float, 16> v429_data = tensorforge::slmLoad<float, 16>(s0 + (151_i32));
              v393_acc += ((static_cast<float>(v429_data[0])) * v59_data);
              v393_acc += ((static_cast<float>(v429_data[1])) * v27_data);
              r0.template select<16, 1>(128) = v393_acc;
              // s1 = store{r>s}(localShrMem0, r0);
              #pragma unroll
              for (int32_t v434_i0 = 0; v434_i0 < 1; ++v434_i0) {
                int32_t v436_a = v434_i0 * 16;
                #pragma unroll
                for (int32_t v435_i1 = 0; v435_i1 < 9; ++v435_i1) {
                  int32_t v438_a = v436_a + (v435_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v439_data(r0.template select<16, 1>(v438_a));
                  tensorforge::slmStore<float, 16>(s1 + (v438_a), v439_data);
                }
              }
              tensorforge::intel_esimd::simd<float, 144> r1(0.0f);
              // ir1 = +(glb_m3 * s1)
              // [(0, 16), (0, 9)] [(1, 16)]
              tensorforge::intel_esimd::simd<float, 144> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 64> glb_m3_run4;
              glb_m3_run4.copy_from(glb_m3 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v447_data(glb_m3_run4.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v449_data(glb_m3_run4.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v451_data(glb_m3_run4.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v453_data(glb_m3_run4.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m3_run5;
              glb_m3_run5.copy_from(glb_m3 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v455_data(glb_m3_run5.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v457_data(glb_m3_run5.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v459_data(glb_m3_run5.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v461_data(glb_m3_run5.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m3_run6;
              glb_m3_run6.copy_from(glb_m3 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v463_data(glb_m3_run6.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v465_data(glb_m3_run6.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v467_data(glb_m3_run6.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v469_data(glb_m3_run6.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 48> glb_m3_run7;
              glb_m3_run7.copy_from(glb_m3 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v471_data(glb_m3_run7.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v473_data(glb_m3_run7.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v475_data(glb_m3_run7.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v477_acc{};
              tensorforge::intel_esimd::simd<float, 16> v478_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v477_acc += ((static_cast<float>(v478_data[0])) * v447_data);
              v477_acc += ((static_cast<float>(v478_data[1])) * v449_data);
              v477_acc += ((static_cast<float>(v478_data[2])) * v451_data);
              v477_acc += ((static_cast<float>(v478_data[3])) * v453_data);
              v477_acc += ((static_cast<float>(v478_data[4])) * v455_data);
              v477_acc += ((static_cast<float>(v478_data[5])) * v457_data);
              v477_acc += ((static_cast<float>(v478_data[6])) * v459_data);
              v477_acc += ((static_cast<float>(v478_data[7])) * v461_data);
              v477_acc += ((static_cast<float>(v478_data[8])) * v463_data);
              v477_acc += ((static_cast<float>(v478_data[9])) * v465_data);
              v477_acc += ((static_cast<float>(v478_data[10])) * v467_data);
              v477_acc += ((static_cast<float>(v478_data[11])) * v469_data);
              v477_acc += ((static_cast<float>(v478_data[12])) * v471_data);
              v477_acc += ((static_cast<float>(v478_data[13])) * v473_data);
              v477_acc += ((static_cast<float>(v478_data[14])) * v475_data);
              v477_acc += ((static_cast<float>(v478_data[15])) * v447_data);
              ir1.template select<16, 1>(0) = v477_acc;
              tensorforge::intel_esimd::simd<float, 16> v511_acc{};
              tensorforge::intel_esimd::simd<float, 16> v512_data = tensorforge::slmLoad<float, 16>(s1 + (16_i32));
              v511_acc += ((static_cast<float>(v512_data[0])) * v447_data);
              v511_acc += ((static_cast<float>(v512_data[1])) * v449_data);
              v511_acc += ((static_cast<float>(v512_data[2])) * v451_data);
              v511_acc += ((static_cast<float>(v512_data[3])) * v453_data);
              v511_acc += ((static_cast<float>(v512_data[4])) * v455_data);
              v511_acc += ((static_cast<float>(v512_data[5])) * v457_data);
              v511_acc += ((static_cast<float>(v512_data[6])) * v459_data);
              v511_acc += ((static_cast<float>(v512_data[7])) * v461_data);
              v511_acc += ((static_cast<float>(v512_data[8])) * v463_data);
              v511_acc += ((static_cast<float>(v512_data[9])) * v465_data);
              v511_acc += ((static_cast<float>(v512_data[10])) * v467_data);
              v511_acc += ((static_cast<float>(v512_data[11])) * v469_data);
              v511_acc += ((static_cast<float>(v512_data[12])) * v471_data);
              v511_acc += ((static_cast<float>(v512_data[13])) * v473_data);
              v511_acc += ((static_cast<float>(v512_data[14])) * v475_data);
              v511_acc += ((static_cast<float>(v512_data[15])) * v447_data);
              ir1.template select<16, 1>(16) = v511_acc;
              tensorforge::intel_esimd::simd<float, 16> v545_acc{};
              tensorforge::intel_esimd::simd<float, 16> v546_data = tensorforge::slmLoad<float, 16>(s1 + (32_i32));
              v545_acc += ((static_cast<float>(v546_data[0])) * v447_data);
              v545_acc += ((static_cast<float>(v546_data[1])) * v449_data);
              v545_acc += ((static_cast<float>(v546_data[2])) * v451_data);
              v545_acc += ((static_cast<float>(v546_data[3])) * v453_data);
              v545_acc += ((static_cast<float>(v546_data[4])) * v455_data);
              v545_acc += ((static_cast<float>(v546_data[5])) * v457_data);
              v545_acc += ((static_cast<float>(v546_data[6])) * v459_data);
              v545_acc += ((static_cast<float>(v546_data[7])) * v461_data);
              v545_acc += ((static_cast<float>(v546_data[8])) * v463_data);
              v545_acc += ((static_cast<float>(v546_data[9])) * v465_data);
              v545_acc += ((static_cast<float>(v546_data[10])) * v467_data);
              v545_acc += ((static_cast<float>(v546_data[11])) * v469_data);
              v545_acc += ((static_cast<float>(v546_data[12])) * v471_data);
              v545_acc += ((static_cast<float>(v546_data[13])) * v473_data);
              v545_acc += ((static_cast<float>(v546_data[14])) * v475_data);
              v545_acc += ((static_cast<float>(v546_data[15])) * v447_data);
              ir1.template select<16, 1>(32) = v545_acc;
              tensorforge::intel_esimd::simd<float, 16> v579_acc{};
              tensorforge::intel_esimd::simd<float, 16> v580_data = tensorforge::slmLoad<float, 16>(s1 + (48_i32));
              v579_acc += ((static_cast<float>(v580_data[0])) * v447_data);
              v579_acc += ((static_cast<float>(v580_data[1])) * v449_data);
              v579_acc += ((static_cast<float>(v580_data[2])) * v451_data);
              v579_acc += ((static_cast<float>(v580_data[3])) * v453_data);
              v579_acc += ((static_cast<float>(v580_data[4])) * v455_data);
              v579_acc += ((static_cast<float>(v580_data[5])) * v457_data);
              v579_acc += ((static_cast<float>(v580_data[6])) * v459_data);
              v579_acc += ((static_cast<float>(v580_data[7])) * v461_data);
              v579_acc += ((static_cast<float>(v580_data[8])) * v463_data);
              v579_acc += ((static_cast<float>(v580_data[9])) * v465_data);
              v579_acc += ((static_cast<float>(v580_data[10])) * v467_data);
              v579_acc += ((static_cast<float>(v580_data[11])) * v469_data);
              v579_acc += ((static_cast<float>(v580_data[12])) * v471_data);
              v579_acc += ((static_cast<float>(v580_data[13])) * v473_data);
              v579_acc += ((static_cast<float>(v580_data[14])) * v475_data);
              v579_acc += ((static_cast<float>(v580_data[15])) * v447_data);
              ir1.template select<16, 1>(48) = v579_acc;
              tensorforge::intel_esimd::simd<float, 16> v613_acc{};
              tensorforge::intel_esimd::simd<float, 16> v614_data = tensorforge::slmLoad<float, 16>(s1 + (64_i32));
              v613_acc += ((static_cast<float>(v614_data[0])) * v447_data);
              v613_acc += ((static_cast<float>(v614_data[1])) * v449_data);
              v613_acc += ((static_cast<float>(v614_data[2])) * v451_data);
              v613_acc += ((static_cast<float>(v614_data[3])) * v453_data);
              v613_acc += ((static_cast<float>(v614_data[4])) * v455_data);
              v613_acc += ((static_cast<float>(v614_data[5])) * v457_data);
              v613_acc += ((static_cast<float>(v614_data[6])) * v459_data);
              v613_acc += ((static_cast<float>(v614_data[7])) * v461_data);
              v613_acc += ((static_cast<float>(v614_data[8])) * v463_data);
              v613_acc += ((static_cast<float>(v614_data[9])) * v465_data);
              v613_acc += ((static_cast<float>(v614_data[10])) * v467_data);
              v613_acc += ((static_cast<float>(v614_data[11])) * v469_data);
              v613_acc += ((static_cast<float>(v614_data[12])) * v471_data);
              v613_acc += ((static_cast<float>(v614_data[13])) * v473_data);
              v613_acc += ((static_cast<float>(v614_data[14])) * v475_data);
              v613_acc += ((static_cast<float>(v614_data[15])) * v447_data);
              ir1.template select<16, 1>(64) = v613_acc;
              tensorforge::intel_esimd::simd<float, 16> v647_acc{};
              tensorforge::intel_esimd::simd<float, 16> v648_data = tensorforge::slmLoad<float, 16>(s1 + (80_i32));
              v647_acc += ((static_cast<float>(v648_data[0])) * v447_data);
              v647_acc += ((static_cast<float>(v648_data[1])) * v449_data);
              v647_acc += ((static_cast<float>(v648_data[2])) * v451_data);
              v647_acc += ((static_cast<float>(v648_data[3])) * v453_data);
              v647_acc += ((static_cast<float>(v648_data[4])) * v455_data);
              v647_acc += ((static_cast<float>(v648_data[5])) * v457_data);
              v647_acc += ((static_cast<float>(v648_data[6])) * v459_data);
              v647_acc += ((static_cast<float>(v648_data[7])) * v461_data);
              v647_acc += ((static_cast<float>(v648_data[8])) * v463_data);
              v647_acc += ((static_cast<float>(v648_data[9])) * v465_data);
              v647_acc += ((static_cast<float>(v648_data[10])) * v467_data);
              v647_acc += ((static_cast<float>(v648_data[11])) * v469_data);
              v647_acc += ((static_cast<float>(v648_data[12])) * v471_data);
              v647_acc += ((static_cast<float>(v648_data[13])) * v473_data);
              v647_acc += ((static_cast<float>(v648_data[14])) * v475_data);
              v647_acc += ((static_cast<float>(v648_data[15])) * v447_data);
              ir1.template select<16, 1>(80) = v647_acc;
              tensorforge::intel_esimd::simd<float, 16> v681_acc{};
              tensorforge::intel_esimd::simd<float, 16> v682_data = tensorforge::slmLoad<float, 16>(s1 + (96_i32));
              v681_acc += ((static_cast<float>(v682_data[0])) * v447_data);
              v681_acc += ((static_cast<float>(v682_data[1])) * v449_data);
              v681_acc += ((static_cast<float>(v682_data[2])) * v451_data);
              v681_acc += ((static_cast<float>(v682_data[3])) * v453_data);
              v681_acc += ((static_cast<float>(v682_data[4])) * v455_data);
              v681_acc += ((static_cast<float>(v682_data[5])) * v457_data);
              v681_acc += ((static_cast<float>(v682_data[6])) * v459_data);
              v681_acc += ((static_cast<float>(v682_data[7])) * v461_data);
              v681_acc += ((static_cast<float>(v682_data[8])) * v463_data);
              v681_acc += ((static_cast<float>(v682_data[9])) * v465_data);
              v681_acc += ((static_cast<float>(v682_data[10])) * v467_data);
              v681_acc += ((static_cast<float>(v682_data[11])) * v469_data);
              v681_acc += ((static_cast<float>(v682_data[12])) * v471_data);
              v681_acc += ((static_cast<float>(v682_data[13])) * v473_data);
              v681_acc += ((static_cast<float>(v682_data[14])) * v475_data);
              v681_acc += ((static_cast<float>(v682_data[15])) * v447_data);
              ir1.template select<16, 1>(96) = v681_acc;
              tensorforge::intel_esimd::simd<float, 16> v715_acc{};
              tensorforge::intel_esimd::simd<float, 16> v716_data = tensorforge::slmLoad<float, 16>(s1 + (112_i32));
              v715_acc += ((static_cast<float>(v716_data[0])) * v447_data);
              v715_acc += ((static_cast<float>(v716_data[1])) * v449_data);
              v715_acc += ((static_cast<float>(v716_data[2])) * v451_data);
              v715_acc += ((static_cast<float>(v716_data[3])) * v453_data);
              v715_acc += ((static_cast<float>(v716_data[4])) * v455_data);
              v715_acc += ((static_cast<float>(v716_data[5])) * v457_data);
              v715_acc += ((static_cast<float>(v716_data[6])) * v459_data);
              v715_acc += ((static_cast<float>(v716_data[7])) * v461_data);
              v715_acc += ((static_cast<float>(v716_data[8])) * v463_data);
              v715_acc += ((static_cast<float>(v716_data[9])) * v465_data);
              v715_acc += ((static_cast<float>(v716_data[10])) * v467_data);
              v715_acc += ((static_cast<float>(v716_data[11])) * v469_data);
              v715_acc += ((static_cast<float>(v716_data[12])) * v471_data);
              v715_acc += ((static_cast<float>(v716_data[13])) * v473_data);
              v715_acc += ((static_cast<float>(v716_data[14])) * v475_data);
              v715_acc += ((static_cast<float>(v716_data[15])) * v447_data);
              ir1.template select<16, 1>(112) = v715_acc;
              tensorforge::intel_esimd::simd<float, 16> v749_acc{};
              tensorforge::intel_esimd::simd<float, 16> v750_data = tensorforge::slmLoad<float, 16>(s1 + (128_i32));
              v749_acc += ((static_cast<float>(v750_data[0])) * v447_data);
              v749_acc += ((static_cast<float>(v750_data[1])) * v449_data);
              v749_acc += ((static_cast<float>(v750_data[2])) * v451_data);
              v749_acc += ((static_cast<float>(v750_data[3])) * v453_data);
              v749_acc += ((static_cast<float>(v750_data[4])) * v455_data);
              v749_acc += ((static_cast<float>(v750_data[5])) * v457_data);
              v749_acc += ((static_cast<float>(v750_data[6])) * v459_data);
              v749_acc += ((static_cast<float>(v750_data[7])) * v461_data);
              v749_acc += ((static_cast<float>(v750_data[8])) * v463_data);
              v749_acc += ((static_cast<float>(v750_data[9])) * v465_data);
              v749_acc += ((static_cast<float>(v750_data[10])) * v467_data);
              v749_acc += ((static_cast<float>(v750_data[11])) * v469_data);
              v749_acc += ((static_cast<float>(v750_data[12])) * v471_data);
              v749_acc += ((static_cast<float>(v750_data[13])) * v473_data);
              v749_acc += ((static_cast<float>(v750_data[14])) * v475_data);
              v749_acc += ((static_cast<float>(v750_data[15])) * v447_data);
              ir1.template select<16, 1>(128) = v749_acc;
              // r1 = ir1
              #pragma unroll
              for (int32_t v783_n0 = 0; v783_n0 < 1; ++v783_n0) {
                int32_t v785_a = v783_n0 * 16;
                #pragma unroll
                for (int32_t v784_n1 = 0; v784_n1 < 9; ++v784_n1) {
                  int32_t v787_a = v785_a + (v784_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v788_data(ir1.template select<16, 1>(v787_a));
                  r1.template select<16, 1>(v787_a) = v788_data;
                }
              }
              // glb_m2 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v789_i0 = 0; v789_i0 < 1; ++v789_i0) {
                int32_t v791_a = v789_i0 * 16;
                #pragma unroll
                for (int32_t v790_i1 = 0; v790_i1 < 9; ++v790_i1) {
                  int32_t v793_a = v791_a + (v790_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v794_data(r1.template select<16, 1>(v793_a));
                  v794_data.copy_to(glb_m2 + (v793_a));
                }
              }
            }
            tensorforge::prefetchL2<153>(&pf_glb_m1[0]);
          }
        }
      }
    });
  });
}

