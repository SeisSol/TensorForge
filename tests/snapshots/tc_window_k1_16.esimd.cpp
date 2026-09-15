// === base name ===
kernel_23d69f63317e39c6

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_23d69f63317e39c6 = {{1, 16, 1}, 16, 16, 1, 16, 10240, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_23d69f63317e39c6(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_23d69f63317e39c6(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_23d69f63317e39c6(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 2560 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_23d69f63317e39c6(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_23d69f63317e39c6(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_23d69f63317e39c6(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_23d69f63317e39c6(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2560 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 10240 B shared, occupancy grid
        // operands:
        //   m0 16×9(16×9) {0..16}×{0..9} strided
        //   m1 16×20(16×16) {0..16}×{1..17} none
        //   m2 20×9(16×9) {1..17}×{0..9} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2560}],"shared_bytes":10240,"shared_elements":2560,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,9]],"name":"m0","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,1],[16,17]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[1,0],[17,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[16,17]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[17,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (160 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (144);
          const float *const __restrict__ glb_m1 = &m1[0];
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v6_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v6_batchId0 < numElements0; v6_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v7_ahead1 = v6_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
            const float *const __restrict__ pf_glb_m2 = &m2[v9_batchId1 * 144 + 0 + m2_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 144 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v6_batchId0 * 144 + 0 + m2_extraOffset];
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
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 16), (0, 9)] [(1, 17)]
              tensorforge::intel_esimd::simd<float, 144> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run0;
              glb_m1_run0.copy_from(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v25_data(glb_m1_run0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v27_data(glb_m1_run0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v29_data(glb_m1_run0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v31_data(glb_m1_run0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run1;
              glb_m1_run1.copy_from(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v33_data(glb_m1_run1.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v35_data(glb_m1_run1.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v37_data(glb_m1_run1.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v39_data(glb_m1_run1.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run2;
              glb_m1_run2.copy_from(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v41_data(glb_m1_run2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v43_data(glb_m1_run2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v45_data(glb_m1_run2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v47_data(glb_m1_run2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run3;
              glb_m1_run3.copy_from(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v49_data(glb_m1_run3.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v51_data(glb_m1_run3.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v53_data(glb_m1_run3.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v55_data(glb_m1_run3.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v56_acc{};
              tensorforge::intel_esimd::simd<float, 16> v59_data(0.0f);
              v59_data.template select<15, 1>(1) = tensorforge::slmLoad<float, 15>((s0 + (-1_i32)) + 1);
              v56_acc += ((static_cast<float>(v59_data[1])) * v25_data);
              v56_acc += ((static_cast<float>(v59_data[2])) * v27_data);
              v56_acc += ((static_cast<float>(v59_data[3])) * v29_data);
              v56_acc += ((static_cast<float>(v59_data[4])) * v31_data);
              v56_acc += ((static_cast<float>(v59_data[5])) * v33_data);
              v56_acc += ((static_cast<float>(v59_data[6])) * v35_data);
              v56_acc += ((static_cast<float>(v59_data[7])) * v37_data);
              v56_acc += ((static_cast<float>(v59_data[8])) * v39_data);
              v56_acc += ((static_cast<float>(v59_data[9])) * v41_data);
              v56_acc += ((static_cast<float>(v59_data[10])) * v43_data);
              v56_acc += ((static_cast<float>(v59_data[11])) * v45_data);
              v56_acc += ((static_cast<float>(v59_data[12])) * v47_data);
              v56_acc += ((static_cast<float>(v59_data[13])) * v49_data);
              v56_acc += ((static_cast<float>(v59_data[14])) * v51_data);
              v56_acc += ((static_cast<float>(v59_data[15])) * v53_data);
              tensorforge::intel_esimd::simd<float, 16> v95_data = tensorforge::slmLoad<float, 16>(s0 + (15_i32));
              v56_acc += ((static_cast<float>(v95_data[0])) * v55_data);
              ir0.template select<16, 1>(0) = v56_acc;
              tensorforge::intel_esimd::simd<float, 16> v98_acc{};
              v98_acc += ((static_cast<float>(v95_data[1])) * v25_data);
              v98_acc += ((static_cast<float>(v95_data[2])) * v27_data);
              v98_acc += ((static_cast<float>(v95_data[3])) * v29_data);
              v98_acc += ((static_cast<float>(v95_data[4])) * v31_data);
              v98_acc += ((static_cast<float>(v95_data[5])) * v33_data);
              v98_acc += ((static_cast<float>(v95_data[6])) * v35_data);
              v98_acc += ((static_cast<float>(v95_data[7])) * v37_data);
              v98_acc += ((static_cast<float>(v95_data[8])) * v39_data);
              v98_acc += ((static_cast<float>(v95_data[9])) * v41_data);
              v98_acc += ((static_cast<float>(v95_data[10])) * v43_data);
              v98_acc += ((static_cast<float>(v95_data[11])) * v45_data);
              v98_acc += ((static_cast<float>(v95_data[12])) * v47_data);
              v98_acc += ((static_cast<float>(v95_data[13])) * v49_data);
              v98_acc += ((static_cast<float>(v95_data[14])) * v51_data);
              v98_acc += ((static_cast<float>(v95_data[15])) * v53_data);
              tensorforge::intel_esimd::simd<float, 16> v133_data = tensorforge::slmLoad<float, 16>(s0 + (31_i32));
              v98_acc += ((static_cast<float>(v133_data[0])) * v55_data);
              ir0.template select<16, 1>(16) = v98_acc;
              tensorforge::intel_esimd::simd<float, 16> v136_acc{};
              v136_acc += ((static_cast<float>(v133_data[1])) * v25_data);
              v136_acc += ((static_cast<float>(v133_data[2])) * v27_data);
              v136_acc += ((static_cast<float>(v133_data[3])) * v29_data);
              v136_acc += ((static_cast<float>(v133_data[4])) * v31_data);
              v136_acc += ((static_cast<float>(v133_data[5])) * v33_data);
              v136_acc += ((static_cast<float>(v133_data[6])) * v35_data);
              v136_acc += ((static_cast<float>(v133_data[7])) * v37_data);
              v136_acc += ((static_cast<float>(v133_data[8])) * v39_data);
              v136_acc += ((static_cast<float>(v133_data[9])) * v41_data);
              v136_acc += ((static_cast<float>(v133_data[10])) * v43_data);
              v136_acc += ((static_cast<float>(v133_data[11])) * v45_data);
              v136_acc += ((static_cast<float>(v133_data[12])) * v47_data);
              v136_acc += ((static_cast<float>(v133_data[13])) * v49_data);
              v136_acc += ((static_cast<float>(v133_data[14])) * v51_data);
              v136_acc += ((static_cast<float>(v133_data[15])) * v53_data);
              tensorforge::intel_esimd::simd<float, 16> v171_data = tensorforge::slmLoad<float, 16>(s0 + (47_i32));
              v136_acc += ((static_cast<float>(v171_data[0])) * v55_data);
              ir0.template select<16, 1>(32) = v136_acc;
              tensorforge::intel_esimd::simd<float, 16> v174_acc{};
              v174_acc += ((static_cast<float>(v171_data[1])) * v25_data);
              v174_acc += ((static_cast<float>(v171_data[2])) * v27_data);
              v174_acc += ((static_cast<float>(v171_data[3])) * v29_data);
              v174_acc += ((static_cast<float>(v171_data[4])) * v31_data);
              v174_acc += ((static_cast<float>(v171_data[5])) * v33_data);
              v174_acc += ((static_cast<float>(v171_data[6])) * v35_data);
              v174_acc += ((static_cast<float>(v171_data[7])) * v37_data);
              v174_acc += ((static_cast<float>(v171_data[8])) * v39_data);
              v174_acc += ((static_cast<float>(v171_data[9])) * v41_data);
              v174_acc += ((static_cast<float>(v171_data[10])) * v43_data);
              v174_acc += ((static_cast<float>(v171_data[11])) * v45_data);
              v174_acc += ((static_cast<float>(v171_data[12])) * v47_data);
              v174_acc += ((static_cast<float>(v171_data[13])) * v49_data);
              v174_acc += ((static_cast<float>(v171_data[14])) * v51_data);
              v174_acc += ((static_cast<float>(v171_data[15])) * v53_data);
              tensorforge::intel_esimd::simd<float, 16> v209_data = tensorforge::slmLoad<float, 16>(s0 + (63_i32));
              v174_acc += ((static_cast<float>(v209_data[0])) * v55_data);
              ir0.template select<16, 1>(48) = v174_acc;
              tensorforge::intel_esimd::simd<float, 16> v212_acc{};
              v212_acc += ((static_cast<float>(v209_data[1])) * v25_data);
              v212_acc += ((static_cast<float>(v209_data[2])) * v27_data);
              v212_acc += ((static_cast<float>(v209_data[3])) * v29_data);
              v212_acc += ((static_cast<float>(v209_data[4])) * v31_data);
              v212_acc += ((static_cast<float>(v209_data[5])) * v33_data);
              v212_acc += ((static_cast<float>(v209_data[6])) * v35_data);
              v212_acc += ((static_cast<float>(v209_data[7])) * v37_data);
              v212_acc += ((static_cast<float>(v209_data[8])) * v39_data);
              v212_acc += ((static_cast<float>(v209_data[9])) * v41_data);
              v212_acc += ((static_cast<float>(v209_data[10])) * v43_data);
              v212_acc += ((static_cast<float>(v209_data[11])) * v45_data);
              v212_acc += ((static_cast<float>(v209_data[12])) * v47_data);
              v212_acc += ((static_cast<float>(v209_data[13])) * v49_data);
              v212_acc += ((static_cast<float>(v209_data[14])) * v51_data);
              v212_acc += ((static_cast<float>(v209_data[15])) * v53_data);
              tensorforge::intel_esimd::simd<float, 16> v247_data = tensorforge::slmLoad<float, 16>(s0 + (79_i32));
              v212_acc += ((static_cast<float>(v247_data[0])) * v55_data);
              ir0.template select<16, 1>(64) = v212_acc;
              tensorforge::intel_esimd::simd<float, 16> v250_acc{};
              v250_acc += ((static_cast<float>(v247_data[1])) * v25_data);
              v250_acc += ((static_cast<float>(v247_data[2])) * v27_data);
              v250_acc += ((static_cast<float>(v247_data[3])) * v29_data);
              v250_acc += ((static_cast<float>(v247_data[4])) * v31_data);
              v250_acc += ((static_cast<float>(v247_data[5])) * v33_data);
              v250_acc += ((static_cast<float>(v247_data[6])) * v35_data);
              v250_acc += ((static_cast<float>(v247_data[7])) * v37_data);
              v250_acc += ((static_cast<float>(v247_data[8])) * v39_data);
              v250_acc += ((static_cast<float>(v247_data[9])) * v41_data);
              v250_acc += ((static_cast<float>(v247_data[10])) * v43_data);
              v250_acc += ((static_cast<float>(v247_data[11])) * v45_data);
              v250_acc += ((static_cast<float>(v247_data[12])) * v47_data);
              v250_acc += ((static_cast<float>(v247_data[13])) * v49_data);
              v250_acc += ((static_cast<float>(v247_data[14])) * v51_data);
              v250_acc += ((static_cast<float>(v247_data[15])) * v53_data);
              tensorforge::intel_esimd::simd<float, 16> v285_data = tensorforge::slmLoad<float, 16>(s0 + (95_i32));
              v250_acc += ((static_cast<float>(v285_data[0])) * v55_data);
              ir0.template select<16, 1>(80) = v250_acc;
              tensorforge::intel_esimd::simd<float, 16> v288_acc{};
              v288_acc += ((static_cast<float>(v285_data[1])) * v25_data);
              v288_acc += ((static_cast<float>(v285_data[2])) * v27_data);
              v288_acc += ((static_cast<float>(v285_data[3])) * v29_data);
              v288_acc += ((static_cast<float>(v285_data[4])) * v31_data);
              v288_acc += ((static_cast<float>(v285_data[5])) * v33_data);
              v288_acc += ((static_cast<float>(v285_data[6])) * v35_data);
              v288_acc += ((static_cast<float>(v285_data[7])) * v37_data);
              v288_acc += ((static_cast<float>(v285_data[8])) * v39_data);
              v288_acc += ((static_cast<float>(v285_data[9])) * v41_data);
              v288_acc += ((static_cast<float>(v285_data[10])) * v43_data);
              v288_acc += ((static_cast<float>(v285_data[11])) * v45_data);
              v288_acc += ((static_cast<float>(v285_data[12])) * v47_data);
              v288_acc += ((static_cast<float>(v285_data[13])) * v49_data);
              v288_acc += ((static_cast<float>(v285_data[14])) * v51_data);
              v288_acc += ((static_cast<float>(v285_data[15])) * v53_data);
              tensorforge::intel_esimd::simd<float, 16> v323_data = tensorforge::slmLoad<float, 16>(s0 + (111_i32));
              v288_acc += ((static_cast<float>(v323_data[0])) * v55_data);
              ir0.template select<16, 1>(96) = v288_acc;
              tensorforge::intel_esimd::simd<float, 16> v326_acc{};
              v326_acc += ((static_cast<float>(v323_data[1])) * v25_data);
              v326_acc += ((static_cast<float>(v323_data[2])) * v27_data);
              v326_acc += ((static_cast<float>(v323_data[3])) * v29_data);
              v326_acc += ((static_cast<float>(v323_data[4])) * v31_data);
              v326_acc += ((static_cast<float>(v323_data[5])) * v33_data);
              v326_acc += ((static_cast<float>(v323_data[6])) * v35_data);
              v326_acc += ((static_cast<float>(v323_data[7])) * v37_data);
              v326_acc += ((static_cast<float>(v323_data[8])) * v39_data);
              v326_acc += ((static_cast<float>(v323_data[9])) * v41_data);
              v326_acc += ((static_cast<float>(v323_data[10])) * v43_data);
              v326_acc += ((static_cast<float>(v323_data[11])) * v45_data);
              v326_acc += ((static_cast<float>(v323_data[12])) * v47_data);
              v326_acc += ((static_cast<float>(v323_data[13])) * v49_data);
              v326_acc += ((static_cast<float>(v323_data[14])) * v51_data);
              v326_acc += ((static_cast<float>(v323_data[15])) * v53_data);
              tensorforge::intel_esimd::simd<float, 16> v361_data = tensorforge::slmLoad<float, 16>(s0 + (127_i32));
              v326_acc += ((static_cast<float>(v361_data[0])) * v55_data);
              ir0.template select<16, 1>(112) = v326_acc;
              tensorforge::intel_esimd::simd<float, 16> v364_acc{};
              v364_acc += ((static_cast<float>(v361_data[1])) * v25_data);
              v364_acc += ((static_cast<float>(v361_data[2])) * v27_data);
              v364_acc += ((static_cast<float>(v361_data[3])) * v29_data);
              v364_acc += ((static_cast<float>(v361_data[4])) * v31_data);
              v364_acc += ((static_cast<float>(v361_data[5])) * v33_data);
              v364_acc += ((static_cast<float>(v361_data[6])) * v35_data);
              v364_acc += ((static_cast<float>(v361_data[7])) * v37_data);
              v364_acc += ((static_cast<float>(v361_data[8])) * v39_data);
              v364_acc += ((static_cast<float>(v361_data[9])) * v41_data);
              v364_acc += ((static_cast<float>(v361_data[10])) * v43_data);
              v364_acc += ((static_cast<float>(v361_data[11])) * v45_data);
              v364_acc += ((static_cast<float>(v361_data[12])) * v47_data);
              v364_acc += ((static_cast<float>(v361_data[13])) * v49_data);
              v364_acc += ((static_cast<float>(v361_data[14])) * v51_data);
              v364_acc += ((static_cast<float>(v361_data[15])) * v53_data);
              tensorforge::intel_esimd::simd<float, 16> v399_data = tensorforge::slmLoad<float, 16>(s0 + (143_i32));
              v364_acc += ((static_cast<float>(v399_data[0])) * v55_data);
              ir0.template select<16, 1>(128) = v364_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v402_n0 = 0; v402_n0 < 1; ++v402_n0) {
                int32_t v404_a = v402_n0 * 16;
                #pragma unroll
                for (int32_t v403_n1 = 0; v403_n1 < 9; ++v403_n1) {
                  int32_t v406_a = v404_a + (v403_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v407_data(ir0.template select<16, 1>(v406_a));
                  r0.template select<16, 1>(v406_a) = v407_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v408_i0 = 0; v408_i0 < 1; ++v408_i0) {
                int32_t v410_a = v408_i0 * 16;
                #pragma unroll
                for (int32_t v409_i1 = 0; v409_i1 < 9; ++v409_i1) {
                  int32_t v412_a = v410_a + (v409_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v413_data(r0.template select<16, 1>(v412_a));
                  v413_data.copy_to(glb_m0 + (v412_a));
                }
              }
            }
            tensorforge::prefetchL2<144>(&pf_glb_m2[0]);
          }
        }
      }
    });
  });
}

