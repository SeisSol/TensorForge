// === base name ===
kernel_8d175c17d6481a85

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_8d175c17d6481a85 = {{1, 16, 1}, 16, 16, 1, 16, 17408, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_8d175c17d6481a85(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_8d175c17d6481a85(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_8d175c17d6481a85(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 4352 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_8d175c17d6481a85(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_8d175c17d6481a85(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_8d175c17d6481a85(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_8d175c17d6481a85(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<4352 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 17408 B shared, occupancy grid
        // operands:
        //   m0 16×16(16×16) {0..16}×{0..16} strided
        //   m1 16×16(16×16) {0..16}×{0..16} none
        //   m2 16×16(16×16) {0..16}×{0..16} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":4352}],"shared_bytes":17408,"shared_elements":4352,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (272 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (256);
          const float *const __restrict__ glb_m1 = &m1[0];
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v6_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v6_batchId0 < numElements0; v6_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v7_ahead1 = v6_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 256 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v6_batchId0 * 256 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v16_ld;
              v16_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v16_ld);
              tensorforge::intel_esimd::simd<float, 64> v17_ld;
              v17_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v17_ld);
              tensorforge::intel_esimd::simd<float, 64> v18_ld;
              v18_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 128));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 128), v18_ld);
              tensorforge::intel_esimd::simd<float, 64> v19_ld;
              v19_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 192));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 192), v19_ld);
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 16), (0, 16)] [(0, 16)]
              tensorforge::intel_esimd::simd<float, 256> ir0(0.0f);
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
              tensorforge::intel_esimd::simd<float, 16> v57_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v56_acc += ((static_cast<float>(v57_data[0])) * v25_data);
              v56_acc += ((static_cast<float>(v57_data[1])) * v27_data);
              v56_acc += ((static_cast<float>(v57_data[2])) * v29_data);
              v56_acc += ((static_cast<float>(v57_data[3])) * v31_data);
              v56_acc += ((static_cast<float>(v57_data[4])) * v33_data);
              v56_acc += ((static_cast<float>(v57_data[5])) * v35_data);
              v56_acc += ((static_cast<float>(v57_data[6])) * v37_data);
              v56_acc += ((static_cast<float>(v57_data[7])) * v39_data);
              v56_acc += ((static_cast<float>(v57_data[8])) * v41_data);
              v56_acc += ((static_cast<float>(v57_data[9])) * v43_data);
              v56_acc += ((static_cast<float>(v57_data[10])) * v45_data);
              v56_acc += ((static_cast<float>(v57_data[11])) * v47_data);
              v56_acc += ((static_cast<float>(v57_data[12])) * v49_data);
              v56_acc += ((static_cast<float>(v57_data[13])) * v51_data);
              v56_acc += ((static_cast<float>(v57_data[14])) * v53_data);
              v56_acc += ((static_cast<float>(v57_data[15])) * v55_data);
              ir0.template select<16, 1>(0) = v56_acc;
              tensorforge::intel_esimd::simd<float, 16> v90_acc{};
              tensorforge::intel_esimd::simd<float, 16> v91_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v90_acc += ((static_cast<float>(v91_data[0])) * v25_data);
              v90_acc += ((static_cast<float>(v91_data[1])) * v27_data);
              v90_acc += ((static_cast<float>(v91_data[2])) * v29_data);
              v90_acc += ((static_cast<float>(v91_data[3])) * v31_data);
              v90_acc += ((static_cast<float>(v91_data[4])) * v33_data);
              v90_acc += ((static_cast<float>(v91_data[5])) * v35_data);
              v90_acc += ((static_cast<float>(v91_data[6])) * v37_data);
              v90_acc += ((static_cast<float>(v91_data[7])) * v39_data);
              v90_acc += ((static_cast<float>(v91_data[8])) * v41_data);
              v90_acc += ((static_cast<float>(v91_data[9])) * v43_data);
              v90_acc += ((static_cast<float>(v91_data[10])) * v45_data);
              v90_acc += ((static_cast<float>(v91_data[11])) * v47_data);
              v90_acc += ((static_cast<float>(v91_data[12])) * v49_data);
              v90_acc += ((static_cast<float>(v91_data[13])) * v51_data);
              v90_acc += ((static_cast<float>(v91_data[14])) * v53_data);
              v90_acc += ((static_cast<float>(v91_data[15])) * v55_data);
              ir0.template select<16, 1>(16) = v90_acc;
              tensorforge::intel_esimd::simd<float, 16> v124_acc{};
              tensorforge::intel_esimd::simd<float, 16> v125_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v124_acc += ((static_cast<float>(v125_data[0])) * v25_data);
              v124_acc += ((static_cast<float>(v125_data[1])) * v27_data);
              v124_acc += ((static_cast<float>(v125_data[2])) * v29_data);
              v124_acc += ((static_cast<float>(v125_data[3])) * v31_data);
              v124_acc += ((static_cast<float>(v125_data[4])) * v33_data);
              v124_acc += ((static_cast<float>(v125_data[5])) * v35_data);
              v124_acc += ((static_cast<float>(v125_data[6])) * v37_data);
              v124_acc += ((static_cast<float>(v125_data[7])) * v39_data);
              v124_acc += ((static_cast<float>(v125_data[8])) * v41_data);
              v124_acc += ((static_cast<float>(v125_data[9])) * v43_data);
              v124_acc += ((static_cast<float>(v125_data[10])) * v45_data);
              v124_acc += ((static_cast<float>(v125_data[11])) * v47_data);
              v124_acc += ((static_cast<float>(v125_data[12])) * v49_data);
              v124_acc += ((static_cast<float>(v125_data[13])) * v51_data);
              v124_acc += ((static_cast<float>(v125_data[14])) * v53_data);
              v124_acc += ((static_cast<float>(v125_data[15])) * v55_data);
              ir0.template select<16, 1>(32) = v124_acc;
              tensorforge::intel_esimd::simd<float, 16> v158_acc{};
              tensorforge::intel_esimd::simd<float, 16> v159_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v158_acc += ((static_cast<float>(v159_data[0])) * v25_data);
              v158_acc += ((static_cast<float>(v159_data[1])) * v27_data);
              v158_acc += ((static_cast<float>(v159_data[2])) * v29_data);
              v158_acc += ((static_cast<float>(v159_data[3])) * v31_data);
              v158_acc += ((static_cast<float>(v159_data[4])) * v33_data);
              v158_acc += ((static_cast<float>(v159_data[5])) * v35_data);
              v158_acc += ((static_cast<float>(v159_data[6])) * v37_data);
              v158_acc += ((static_cast<float>(v159_data[7])) * v39_data);
              v158_acc += ((static_cast<float>(v159_data[8])) * v41_data);
              v158_acc += ((static_cast<float>(v159_data[9])) * v43_data);
              v158_acc += ((static_cast<float>(v159_data[10])) * v45_data);
              v158_acc += ((static_cast<float>(v159_data[11])) * v47_data);
              v158_acc += ((static_cast<float>(v159_data[12])) * v49_data);
              v158_acc += ((static_cast<float>(v159_data[13])) * v51_data);
              v158_acc += ((static_cast<float>(v159_data[14])) * v53_data);
              v158_acc += ((static_cast<float>(v159_data[15])) * v55_data);
              ir0.template select<16, 1>(48) = v158_acc;
              tensorforge::intel_esimd::simd<float, 16> v192_acc{};
              tensorforge::intel_esimd::simd<float, 16> v193_data = tensorforge::slmLoad<float, 16>(s0 + (64_i32));
              v192_acc += ((static_cast<float>(v193_data[0])) * v25_data);
              v192_acc += ((static_cast<float>(v193_data[1])) * v27_data);
              v192_acc += ((static_cast<float>(v193_data[2])) * v29_data);
              v192_acc += ((static_cast<float>(v193_data[3])) * v31_data);
              v192_acc += ((static_cast<float>(v193_data[4])) * v33_data);
              v192_acc += ((static_cast<float>(v193_data[5])) * v35_data);
              v192_acc += ((static_cast<float>(v193_data[6])) * v37_data);
              v192_acc += ((static_cast<float>(v193_data[7])) * v39_data);
              v192_acc += ((static_cast<float>(v193_data[8])) * v41_data);
              v192_acc += ((static_cast<float>(v193_data[9])) * v43_data);
              v192_acc += ((static_cast<float>(v193_data[10])) * v45_data);
              v192_acc += ((static_cast<float>(v193_data[11])) * v47_data);
              v192_acc += ((static_cast<float>(v193_data[12])) * v49_data);
              v192_acc += ((static_cast<float>(v193_data[13])) * v51_data);
              v192_acc += ((static_cast<float>(v193_data[14])) * v53_data);
              v192_acc += ((static_cast<float>(v193_data[15])) * v55_data);
              ir0.template select<16, 1>(64) = v192_acc;
              tensorforge::intel_esimd::simd<float, 16> v226_acc{};
              tensorforge::intel_esimd::simd<float, 16> v227_data = tensorforge::slmLoad<float, 16>(s0 + (80_i32));
              v226_acc += ((static_cast<float>(v227_data[0])) * v25_data);
              v226_acc += ((static_cast<float>(v227_data[1])) * v27_data);
              v226_acc += ((static_cast<float>(v227_data[2])) * v29_data);
              v226_acc += ((static_cast<float>(v227_data[3])) * v31_data);
              v226_acc += ((static_cast<float>(v227_data[4])) * v33_data);
              v226_acc += ((static_cast<float>(v227_data[5])) * v35_data);
              v226_acc += ((static_cast<float>(v227_data[6])) * v37_data);
              v226_acc += ((static_cast<float>(v227_data[7])) * v39_data);
              v226_acc += ((static_cast<float>(v227_data[8])) * v41_data);
              v226_acc += ((static_cast<float>(v227_data[9])) * v43_data);
              v226_acc += ((static_cast<float>(v227_data[10])) * v45_data);
              v226_acc += ((static_cast<float>(v227_data[11])) * v47_data);
              v226_acc += ((static_cast<float>(v227_data[12])) * v49_data);
              v226_acc += ((static_cast<float>(v227_data[13])) * v51_data);
              v226_acc += ((static_cast<float>(v227_data[14])) * v53_data);
              v226_acc += ((static_cast<float>(v227_data[15])) * v55_data);
              ir0.template select<16, 1>(80) = v226_acc;
              tensorforge::intel_esimd::simd<float, 16> v260_acc{};
              tensorforge::intel_esimd::simd<float, 16> v261_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              v260_acc += ((static_cast<float>(v261_data[0])) * v25_data);
              v260_acc += ((static_cast<float>(v261_data[1])) * v27_data);
              v260_acc += ((static_cast<float>(v261_data[2])) * v29_data);
              v260_acc += ((static_cast<float>(v261_data[3])) * v31_data);
              v260_acc += ((static_cast<float>(v261_data[4])) * v33_data);
              v260_acc += ((static_cast<float>(v261_data[5])) * v35_data);
              v260_acc += ((static_cast<float>(v261_data[6])) * v37_data);
              v260_acc += ((static_cast<float>(v261_data[7])) * v39_data);
              v260_acc += ((static_cast<float>(v261_data[8])) * v41_data);
              v260_acc += ((static_cast<float>(v261_data[9])) * v43_data);
              v260_acc += ((static_cast<float>(v261_data[10])) * v45_data);
              v260_acc += ((static_cast<float>(v261_data[11])) * v47_data);
              v260_acc += ((static_cast<float>(v261_data[12])) * v49_data);
              v260_acc += ((static_cast<float>(v261_data[13])) * v51_data);
              v260_acc += ((static_cast<float>(v261_data[14])) * v53_data);
              v260_acc += ((static_cast<float>(v261_data[15])) * v55_data);
              ir0.template select<16, 1>(96) = v260_acc;
              tensorforge::intel_esimd::simd<float, 16> v294_acc{};
              tensorforge::intel_esimd::simd<float, 16> v295_data = tensorforge::slmLoad<float, 16>(s0 + (112_i32));
              v294_acc += ((static_cast<float>(v295_data[0])) * v25_data);
              v294_acc += ((static_cast<float>(v295_data[1])) * v27_data);
              v294_acc += ((static_cast<float>(v295_data[2])) * v29_data);
              v294_acc += ((static_cast<float>(v295_data[3])) * v31_data);
              v294_acc += ((static_cast<float>(v295_data[4])) * v33_data);
              v294_acc += ((static_cast<float>(v295_data[5])) * v35_data);
              v294_acc += ((static_cast<float>(v295_data[6])) * v37_data);
              v294_acc += ((static_cast<float>(v295_data[7])) * v39_data);
              v294_acc += ((static_cast<float>(v295_data[8])) * v41_data);
              v294_acc += ((static_cast<float>(v295_data[9])) * v43_data);
              v294_acc += ((static_cast<float>(v295_data[10])) * v45_data);
              v294_acc += ((static_cast<float>(v295_data[11])) * v47_data);
              v294_acc += ((static_cast<float>(v295_data[12])) * v49_data);
              v294_acc += ((static_cast<float>(v295_data[13])) * v51_data);
              v294_acc += ((static_cast<float>(v295_data[14])) * v53_data);
              v294_acc += ((static_cast<float>(v295_data[15])) * v55_data);
              ir0.template select<16, 1>(112) = v294_acc;
              tensorforge::intel_esimd::simd<float, 16> v328_acc{};
              tensorforge::intel_esimd::simd<float, 16> v329_data = tensorforge::slmLoad<float, 16>(s0 + (128_i32));
              v328_acc += ((static_cast<float>(v329_data[0])) * v25_data);
              v328_acc += ((static_cast<float>(v329_data[1])) * v27_data);
              v328_acc += ((static_cast<float>(v329_data[2])) * v29_data);
              v328_acc += ((static_cast<float>(v329_data[3])) * v31_data);
              v328_acc += ((static_cast<float>(v329_data[4])) * v33_data);
              v328_acc += ((static_cast<float>(v329_data[5])) * v35_data);
              v328_acc += ((static_cast<float>(v329_data[6])) * v37_data);
              v328_acc += ((static_cast<float>(v329_data[7])) * v39_data);
              v328_acc += ((static_cast<float>(v329_data[8])) * v41_data);
              v328_acc += ((static_cast<float>(v329_data[9])) * v43_data);
              v328_acc += ((static_cast<float>(v329_data[10])) * v45_data);
              v328_acc += ((static_cast<float>(v329_data[11])) * v47_data);
              v328_acc += ((static_cast<float>(v329_data[12])) * v49_data);
              v328_acc += ((static_cast<float>(v329_data[13])) * v51_data);
              v328_acc += ((static_cast<float>(v329_data[14])) * v53_data);
              v328_acc += ((static_cast<float>(v329_data[15])) * v55_data);
              ir0.template select<16, 1>(128) = v328_acc;
              tensorforge::intel_esimd::simd<float, 16> v362_acc{};
              tensorforge::intel_esimd::simd<float, 16> v363_data = tensorforge::slmLoad<float, 16>(s0 + (144_i32));
              v362_acc += ((static_cast<float>(v363_data[0])) * v25_data);
              v362_acc += ((static_cast<float>(v363_data[1])) * v27_data);
              v362_acc += ((static_cast<float>(v363_data[2])) * v29_data);
              v362_acc += ((static_cast<float>(v363_data[3])) * v31_data);
              v362_acc += ((static_cast<float>(v363_data[4])) * v33_data);
              v362_acc += ((static_cast<float>(v363_data[5])) * v35_data);
              v362_acc += ((static_cast<float>(v363_data[6])) * v37_data);
              v362_acc += ((static_cast<float>(v363_data[7])) * v39_data);
              v362_acc += ((static_cast<float>(v363_data[8])) * v41_data);
              v362_acc += ((static_cast<float>(v363_data[9])) * v43_data);
              v362_acc += ((static_cast<float>(v363_data[10])) * v45_data);
              v362_acc += ((static_cast<float>(v363_data[11])) * v47_data);
              v362_acc += ((static_cast<float>(v363_data[12])) * v49_data);
              v362_acc += ((static_cast<float>(v363_data[13])) * v51_data);
              v362_acc += ((static_cast<float>(v363_data[14])) * v53_data);
              v362_acc += ((static_cast<float>(v363_data[15])) * v55_data);
              ir0.template select<16, 1>(144) = v362_acc;
              tensorforge::intel_esimd::simd<float, 16> v396_acc{};
              tensorforge::intel_esimd::simd<float, 16> v397_data = tensorforge::slmLoad<float, 16>(s0 + (160_i32));
              v396_acc += ((static_cast<float>(v397_data[0])) * v25_data);
              v396_acc += ((static_cast<float>(v397_data[1])) * v27_data);
              v396_acc += ((static_cast<float>(v397_data[2])) * v29_data);
              v396_acc += ((static_cast<float>(v397_data[3])) * v31_data);
              v396_acc += ((static_cast<float>(v397_data[4])) * v33_data);
              v396_acc += ((static_cast<float>(v397_data[5])) * v35_data);
              v396_acc += ((static_cast<float>(v397_data[6])) * v37_data);
              v396_acc += ((static_cast<float>(v397_data[7])) * v39_data);
              v396_acc += ((static_cast<float>(v397_data[8])) * v41_data);
              v396_acc += ((static_cast<float>(v397_data[9])) * v43_data);
              v396_acc += ((static_cast<float>(v397_data[10])) * v45_data);
              v396_acc += ((static_cast<float>(v397_data[11])) * v47_data);
              v396_acc += ((static_cast<float>(v397_data[12])) * v49_data);
              v396_acc += ((static_cast<float>(v397_data[13])) * v51_data);
              v396_acc += ((static_cast<float>(v397_data[14])) * v53_data);
              v396_acc += ((static_cast<float>(v397_data[15])) * v55_data);
              ir0.template select<16, 1>(160) = v396_acc;
              tensorforge::intel_esimd::simd<float, 16> v430_acc{};
              tensorforge::intel_esimd::simd<float, 16> v431_data = tensorforge::slmLoad<float, 16>(s0 + (176_i32));
              v430_acc += ((static_cast<float>(v431_data[0])) * v25_data);
              v430_acc += ((static_cast<float>(v431_data[1])) * v27_data);
              v430_acc += ((static_cast<float>(v431_data[2])) * v29_data);
              v430_acc += ((static_cast<float>(v431_data[3])) * v31_data);
              v430_acc += ((static_cast<float>(v431_data[4])) * v33_data);
              v430_acc += ((static_cast<float>(v431_data[5])) * v35_data);
              v430_acc += ((static_cast<float>(v431_data[6])) * v37_data);
              v430_acc += ((static_cast<float>(v431_data[7])) * v39_data);
              v430_acc += ((static_cast<float>(v431_data[8])) * v41_data);
              v430_acc += ((static_cast<float>(v431_data[9])) * v43_data);
              v430_acc += ((static_cast<float>(v431_data[10])) * v45_data);
              v430_acc += ((static_cast<float>(v431_data[11])) * v47_data);
              v430_acc += ((static_cast<float>(v431_data[12])) * v49_data);
              v430_acc += ((static_cast<float>(v431_data[13])) * v51_data);
              v430_acc += ((static_cast<float>(v431_data[14])) * v53_data);
              v430_acc += ((static_cast<float>(v431_data[15])) * v55_data);
              ir0.template select<16, 1>(176) = v430_acc;
              tensorforge::intel_esimd::simd<float, 16> v464_acc{};
              tensorforge::intel_esimd::simd<float, 16> v465_data = tensorforge::slmLoad<float, 16>(s0 + (192_i32));
              v464_acc += ((static_cast<float>(v465_data[0])) * v25_data);
              v464_acc += ((static_cast<float>(v465_data[1])) * v27_data);
              v464_acc += ((static_cast<float>(v465_data[2])) * v29_data);
              v464_acc += ((static_cast<float>(v465_data[3])) * v31_data);
              v464_acc += ((static_cast<float>(v465_data[4])) * v33_data);
              v464_acc += ((static_cast<float>(v465_data[5])) * v35_data);
              v464_acc += ((static_cast<float>(v465_data[6])) * v37_data);
              v464_acc += ((static_cast<float>(v465_data[7])) * v39_data);
              v464_acc += ((static_cast<float>(v465_data[8])) * v41_data);
              v464_acc += ((static_cast<float>(v465_data[9])) * v43_data);
              v464_acc += ((static_cast<float>(v465_data[10])) * v45_data);
              v464_acc += ((static_cast<float>(v465_data[11])) * v47_data);
              v464_acc += ((static_cast<float>(v465_data[12])) * v49_data);
              v464_acc += ((static_cast<float>(v465_data[13])) * v51_data);
              v464_acc += ((static_cast<float>(v465_data[14])) * v53_data);
              v464_acc += ((static_cast<float>(v465_data[15])) * v55_data);
              ir0.template select<16, 1>(192) = v464_acc;
              tensorforge::intel_esimd::simd<float, 16> v498_acc{};
              tensorforge::intel_esimd::simd<float, 16> v499_data = tensorforge::slmLoad<float, 16>(s0 + (208_i32));
              v498_acc += ((static_cast<float>(v499_data[0])) * v25_data);
              v498_acc += ((static_cast<float>(v499_data[1])) * v27_data);
              v498_acc += ((static_cast<float>(v499_data[2])) * v29_data);
              v498_acc += ((static_cast<float>(v499_data[3])) * v31_data);
              v498_acc += ((static_cast<float>(v499_data[4])) * v33_data);
              v498_acc += ((static_cast<float>(v499_data[5])) * v35_data);
              v498_acc += ((static_cast<float>(v499_data[6])) * v37_data);
              v498_acc += ((static_cast<float>(v499_data[7])) * v39_data);
              v498_acc += ((static_cast<float>(v499_data[8])) * v41_data);
              v498_acc += ((static_cast<float>(v499_data[9])) * v43_data);
              v498_acc += ((static_cast<float>(v499_data[10])) * v45_data);
              v498_acc += ((static_cast<float>(v499_data[11])) * v47_data);
              v498_acc += ((static_cast<float>(v499_data[12])) * v49_data);
              v498_acc += ((static_cast<float>(v499_data[13])) * v51_data);
              v498_acc += ((static_cast<float>(v499_data[14])) * v53_data);
              v498_acc += ((static_cast<float>(v499_data[15])) * v55_data);
              ir0.template select<16, 1>(208) = v498_acc;
              tensorforge::intel_esimd::simd<float, 16> v532_acc{};
              tensorforge::intel_esimd::simd<float, 16> v533_data = tensorforge::slmLoad<float, 16>(s0 + (224_i32));
              v532_acc += ((static_cast<float>(v533_data[0])) * v25_data);
              v532_acc += ((static_cast<float>(v533_data[1])) * v27_data);
              v532_acc += ((static_cast<float>(v533_data[2])) * v29_data);
              v532_acc += ((static_cast<float>(v533_data[3])) * v31_data);
              v532_acc += ((static_cast<float>(v533_data[4])) * v33_data);
              v532_acc += ((static_cast<float>(v533_data[5])) * v35_data);
              v532_acc += ((static_cast<float>(v533_data[6])) * v37_data);
              v532_acc += ((static_cast<float>(v533_data[7])) * v39_data);
              v532_acc += ((static_cast<float>(v533_data[8])) * v41_data);
              v532_acc += ((static_cast<float>(v533_data[9])) * v43_data);
              v532_acc += ((static_cast<float>(v533_data[10])) * v45_data);
              v532_acc += ((static_cast<float>(v533_data[11])) * v47_data);
              v532_acc += ((static_cast<float>(v533_data[12])) * v49_data);
              v532_acc += ((static_cast<float>(v533_data[13])) * v51_data);
              v532_acc += ((static_cast<float>(v533_data[14])) * v53_data);
              v532_acc += ((static_cast<float>(v533_data[15])) * v55_data);
              ir0.template select<16, 1>(224) = v532_acc;
              tensorforge::intel_esimd::simd<float, 16> v566_acc{};
              tensorforge::intel_esimd::simd<float, 16> v567_data = tensorforge::slmLoad<float, 16>(s0 + (240_i32));
              v566_acc += ((static_cast<float>(v567_data[0])) * v25_data);
              v566_acc += ((static_cast<float>(v567_data[1])) * v27_data);
              v566_acc += ((static_cast<float>(v567_data[2])) * v29_data);
              v566_acc += ((static_cast<float>(v567_data[3])) * v31_data);
              v566_acc += ((static_cast<float>(v567_data[4])) * v33_data);
              v566_acc += ((static_cast<float>(v567_data[5])) * v35_data);
              v566_acc += ((static_cast<float>(v567_data[6])) * v37_data);
              v566_acc += ((static_cast<float>(v567_data[7])) * v39_data);
              v566_acc += ((static_cast<float>(v567_data[8])) * v41_data);
              v566_acc += ((static_cast<float>(v567_data[9])) * v43_data);
              v566_acc += ((static_cast<float>(v567_data[10])) * v45_data);
              v566_acc += ((static_cast<float>(v567_data[11])) * v47_data);
              v566_acc += ((static_cast<float>(v567_data[12])) * v49_data);
              v566_acc += ((static_cast<float>(v567_data[13])) * v51_data);
              v566_acc += ((static_cast<float>(v567_data[14])) * v53_data);
              v566_acc += ((static_cast<float>(v567_data[15])) * v55_data);
              ir0.template select<16, 1>(240) = v566_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v600_n0 = 0; v600_n0 < 1; ++v600_n0) {
                int32_t v602_a = v600_n0 * 16;
                #pragma unroll
                for (int32_t v601_n1 = 0; v601_n1 < 16; ++v601_n1) {
                  int32_t v604_a = v602_a + (v601_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v605_data(ir0.template select<16, 1>(v604_a));
                  r0.template select<16, 1>(v604_a) = v605_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v606_i0 = 0; v606_i0 < 1; ++v606_i0) {
                int32_t v608_a = v606_i0 * 16;
                #pragma unroll
                for (int32_t v607_i1 = 0; v607_i1 < 16; ++v607_i1) {
                  int32_t v610_a = v608_a + (v607_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v611_data(r0.template select<16, 1>(v610_a));
                  v611_data.copy_to(glb_m0 + (v610_a));
                }
              }
            }
          }
        }
      }
    });
  });
}

