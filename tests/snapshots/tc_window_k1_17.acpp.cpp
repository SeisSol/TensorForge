// === base name ===
kernel_da6e0163e33df625

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_da6e0163e33df625 = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_da6e0163e33df625(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_da6e0163e33df625(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_da6e0163e33df625(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (16, 16, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 16;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_da6e0163e33df625(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_da6e0163e33df625(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_da6e0163e33df625(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_da6e0163e33df625(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 16×9(16×9) {0..16}×{0..9} strided
        //   m1 16×20(16×17) {0..16}×{1..18} none
        //   m2 20×9(17×9) {1..18}×{0..9} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,9]],"name":"m0","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,1],[16,18]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[1,0],[18,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[16,18]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[18,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          const float *const __restrict__ glb_m1 = &m1[0];
          for (size_t v4_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v4_batchId0 < numElements0; v4_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v5_ahead1 = v4_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v4_batchId0 * 144 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v4_batchId0 * 153 + 0 + m2_extraOffset];
              float r0[18]{};
              // r0 = load{g>r}(glb_m2);
              int32_t v17_lead = item.get_local_id(2) % 16;
              if (v17_lead >= 1) {
                int32_t v22_a = v17_lead - 1;
                #pragma unroll
                for (int32_t v19_i1 = 0; v19_i1 < 9; ++v19_i1) {
                  float v25_data = glb_m2[(v22_a + (v19_i1 * 17))];
                  r0[(v19_i1 * 2)] = v25_data;
                }
              }
              if (v17_lead < 2) {
                int32_t v32_a = (v17_lead + 16_i32) - 1;
                #pragma unroll
                for (int32_t v29_i1 = 0; v29_i1 < 9; ++v29_i1) {
                  float v35_data = glb_m2[(v32_a + (v29_i1 * 17))];
                  r0[(1 + (v29_i1 * 2))] = v35_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m2););
              float r1[9]{};
              // ir1 = +(glb_m1 * r0)
              // [(0, 16), (0, 9)] [(1, 18)]
              float ir1[9]{};
              float v43_data = glb_m1[v17_lead];
              float v44_data = r0[0];
              float v47_data = ir1[0];
              ir1[0] = (v47_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v50_data = r0[2];
              float v53_data = ir1[1];
              ir1[1] = (v53_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v56_data = r0[4];
              float v59_data = ir1[2];
              ir1[2] = (v59_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v62_data = r0[6];
              float v65_data = ir1[3];
              ir1[3] = (v65_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v68_data = r0[8];
              float v71_data = ir1[4];
              ir1[4] = (v71_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v74_data = r0[10];
              float v77_data = ir1[5];
              ir1[5] = (v77_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v80_data = r0[12];
              float v83_data = ir1[6];
              ir1[6] = (v83_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v86_data = r0[14];
              float v89_data = ir1[7];
              ir1[7] = (v89_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v92_data = r0[16];
              float v95_data = ir1[8];
              ir1[8] = (v95_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v98_data = glb_m1[(v17_lead + 16)];
              float v102_data = ir1[0];
              ir1[0] = (v102_data + (v98_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v108_data = ir1[1];
              ir1[1] = (v108_data + (v98_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v114_data = ir1[2];
              ir1[2] = (v114_data + (v98_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v120_data = ir1[3];
              ir1[3] = (v120_data + (v98_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v126_data = ir1[4];
              ir1[4] = (v126_data + (v98_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v132_data = ir1[5];
              ir1[5] = (v132_data + (v98_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v138_data = ir1[6];
              ir1[6] = (v138_data + (v98_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v144_data = ir1[7];
              ir1[7] = (v144_data + (v98_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v150_data = ir1[8];
              ir1[8] = (v150_data + (v98_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v153_data = glb_m1[(v17_lead + 32)];
              float v157_data = ir1[0];
              ir1[0] = (v157_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v163_data = ir1[1];
              ir1[1] = (v163_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v169_data = ir1[2];
              ir1[2] = (v169_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v175_data = ir1[3];
              ir1[3] = (v175_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v181_data = ir1[4];
              ir1[4] = (v181_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v187_data = ir1[5];
              ir1[5] = (v187_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v193_data = ir1[6];
              ir1[6] = (v193_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v199_data = ir1[7];
              ir1[7] = (v199_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v205_data = ir1[8];
              ir1[8] = (v205_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v208_data = glb_m1[(v17_lead + 48)];
              float v212_data = ir1[0];
              ir1[0] = (v212_data + (v208_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v218_data = ir1[1];
              ir1[1] = (v218_data + (v208_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v224_data = ir1[2];
              ir1[2] = (v224_data + (v208_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v230_data = ir1[3];
              ir1[3] = (v230_data + (v208_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v236_data = ir1[4];
              ir1[4] = (v236_data + (v208_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v242_data = ir1[5];
              ir1[5] = (v242_data + (v208_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v248_data = ir1[6];
              ir1[6] = (v248_data + (v208_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v254_data = ir1[7];
              ir1[7] = (v254_data + (v208_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v260_data = ir1[8];
              ir1[8] = (v260_data + (v208_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v263_data = glb_m1[(v17_lead + 64)];
              float v267_data = ir1[0];
              ir1[0] = (v267_data + (v263_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v273_data = ir1[1];
              ir1[1] = (v273_data + (v263_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v279_data = ir1[2];
              ir1[2] = (v279_data + (v263_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v285_data = ir1[3];
              ir1[3] = (v285_data + (v263_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v291_data = ir1[4];
              ir1[4] = (v291_data + (v263_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v297_data = ir1[5];
              ir1[5] = (v297_data + (v263_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v303_data = ir1[6];
              ir1[6] = (v303_data + (v263_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v309_data = ir1[7];
              ir1[7] = (v309_data + (v263_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v315_data = ir1[8];
              ir1[8] = (v315_data + (v263_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v318_data = glb_m1[(v17_lead + 80)];
              float v322_data = ir1[0];
              ir1[0] = (v322_data + (v318_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v328_data = ir1[1];
              ir1[1] = (v328_data + (v318_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v334_data = ir1[2];
              ir1[2] = (v334_data + (v318_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v340_data = ir1[3];
              ir1[3] = (v340_data + (v318_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v346_data = ir1[4];
              ir1[4] = (v346_data + (v318_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v352_data = ir1[5];
              ir1[5] = (v352_data + (v318_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v358_data = ir1[6];
              ir1[6] = (v358_data + (v318_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v364_data = ir1[7];
              ir1[7] = (v364_data + (v318_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v370_data = ir1[8];
              ir1[8] = (v370_data + (v318_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v373_data = glb_m1[(v17_lead + 96)];
              float v377_data = ir1[0];
              ir1[0] = (v377_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v383_data = ir1[1];
              ir1[1] = (v383_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v389_data = ir1[2];
              ir1[2] = (v389_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v395_data = ir1[3];
              ir1[3] = (v395_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v401_data = ir1[4];
              ir1[4] = (v401_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v407_data = ir1[5];
              ir1[5] = (v407_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v413_data = ir1[6];
              ir1[6] = (v413_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v419_data = ir1[7];
              ir1[7] = (v419_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v425_data = ir1[8];
              ir1[8] = (v425_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v428_data = glb_m1[(v17_lead + 112)];
              float v432_data = ir1[0];
              ir1[0] = (v432_data + (v428_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v438_data = ir1[1];
              ir1[1] = (v438_data + (v428_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v444_data = ir1[2];
              ir1[2] = (v444_data + (v428_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v450_data = ir1[3];
              ir1[3] = (v450_data + (v428_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v456_data = ir1[4];
              ir1[4] = (v456_data + (v428_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v462_data = ir1[5];
              ir1[5] = (v462_data + (v428_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v468_data = ir1[6];
              ir1[6] = (v468_data + (v428_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v474_data = ir1[7];
              ir1[7] = (v474_data + (v428_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v480_data = ir1[8];
              ir1[8] = (v480_data + (v428_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v483_data = glb_m1[(v17_lead + 128)];
              float v487_data = ir1[0];
              ir1[0] = (v487_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v493_data = ir1[1];
              ir1[1] = (v493_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v499_data = ir1[2];
              ir1[2] = (v499_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v505_data = ir1[3];
              ir1[3] = (v505_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v511_data = ir1[4];
              ir1[4] = (v511_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v517_data = ir1[5];
              ir1[5] = (v517_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v523_data = ir1[6];
              ir1[6] = (v523_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v529_data = ir1[7];
              ir1[7] = (v529_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v535_data = ir1[8];
              ir1[8] = (v535_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v538_data = glb_m1[(v17_lead + 144)];
              float v542_data = ir1[0];
              ir1[0] = (v542_data + (v538_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v548_data = ir1[1];
              ir1[1] = (v548_data + (v538_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v554_data = ir1[2];
              ir1[2] = (v554_data + (v538_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v560_data = ir1[3];
              ir1[3] = (v560_data + (v538_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v566_data = ir1[4];
              ir1[4] = (v566_data + (v538_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v572_data = ir1[5];
              ir1[5] = (v572_data + (v538_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v578_data = ir1[6];
              ir1[6] = (v578_data + (v538_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v584_data = ir1[7];
              ir1[7] = (v584_data + (v538_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v590_data = ir1[8];
              ir1[8] = (v590_data + (v538_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v593_data = glb_m1[(v17_lead + 160)];
              float v597_data = ir1[0];
              ir1[0] = (v597_data + (v593_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v603_data = ir1[1];
              ir1[1] = (v603_data + (v593_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v609_data = ir1[2];
              ir1[2] = (v609_data + (v593_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v615_data = ir1[3];
              ir1[3] = (v615_data + (v593_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v621_data = ir1[4];
              ir1[4] = (v621_data + (v593_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v627_data = ir1[5];
              ir1[5] = (v627_data + (v593_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v633_data = ir1[6];
              ir1[6] = (v633_data + (v593_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v639_data = ir1[7];
              ir1[7] = (v639_data + (v593_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v645_data = ir1[8];
              ir1[8] = (v645_data + (v593_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v648_data = glb_m1[(v17_lead + 176)];
              float v652_data = ir1[0];
              ir1[0] = (v652_data + (v648_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v658_data = ir1[1];
              ir1[1] = (v658_data + (v648_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v664_data = ir1[2];
              ir1[2] = (v664_data + (v648_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v670_data = ir1[3];
              ir1[3] = (v670_data + (v648_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v676_data = ir1[4];
              ir1[4] = (v676_data + (v648_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v682_data = ir1[5];
              ir1[5] = (v682_data + (v648_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v688_data = ir1[6];
              ir1[6] = (v688_data + (v648_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v694_data = ir1[7];
              ir1[7] = (v694_data + (v648_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v700_data = ir1[8];
              ir1[8] = (v700_data + (v648_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v703_data = glb_m1[(v17_lead + 192)];
              float v707_data = ir1[0];
              ir1[0] = (v707_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v713_data = ir1[1];
              ir1[1] = (v713_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v719_data = ir1[2];
              ir1[2] = (v719_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v725_data = ir1[3];
              ir1[3] = (v725_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v731_data = ir1[4];
              ir1[4] = (v731_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v737_data = ir1[5];
              ir1[5] = (v737_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v743_data = ir1[6];
              ir1[6] = (v743_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v749_data = ir1[7];
              ir1[7] = (v749_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v755_data = ir1[8];
              ir1[8] = (v755_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v758_data = glb_m1[(v17_lead + 208)];
              float v762_data = ir1[0];
              ir1[0] = (v762_data + (v758_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v768_data = ir1[1];
              ir1[1] = (v768_data + (v758_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v774_data = ir1[2];
              ir1[2] = (v774_data + (v758_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v780_data = ir1[3];
              ir1[3] = (v780_data + (v758_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v786_data = ir1[4];
              ir1[4] = (v786_data + (v758_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v792_data = ir1[5];
              ir1[5] = (v792_data + (v758_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v798_data = ir1[6];
              ir1[6] = (v798_data + (v758_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v804_data = ir1[7];
              ir1[7] = (v804_data + (v758_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v810_data = ir1[8];
              ir1[8] = (v810_data + (v758_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v813_data = glb_m1[(v17_lead + 224)];
              float v817_data = ir1[0];
              ir1[0] = (v817_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v823_data = ir1[1];
              ir1[1] = (v823_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v829_data = ir1[2];
              ir1[2] = (v829_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v835_data = ir1[3];
              ir1[3] = (v835_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v841_data = ir1[4];
              ir1[4] = (v841_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v847_data = ir1[5];
              ir1[5] = (v847_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v853_data = ir1[6];
              ir1[6] = (v853_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v859_data = ir1[7];
              ir1[7] = (v859_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v865_data = ir1[8];
              ir1[8] = (v865_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v868_data = glb_m1[(v17_lead + 240)];
              float v869_data = r0[1];
              float v872_data = ir1[0];
              ir1[0] = (v872_data + (v868_data * (sycl::select_from_group(item.get_sub_group(), v869_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v875_data = r0[3];
              float v878_data = ir1[1];
              ir1[1] = (v878_data + (v868_data * (sycl::select_from_group(item.get_sub_group(), v875_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v881_data = r0[5];
              float v884_data = ir1[2];
              ir1[2] = (v884_data + (v868_data * (sycl::select_from_group(item.get_sub_group(), v881_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v887_data = r0[7];
              float v890_data = ir1[3];
              ir1[3] = (v890_data + (v868_data * (sycl::select_from_group(item.get_sub_group(), v887_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v893_data = r0[9];
              float v896_data = ir1[4];
              ir1[4] = (v896_data + (v868_data * (sycl::select_from_group(item.get_sub_group(), v893_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v899_data = r0[11];
              float v902_data = ir1[5];
              ir1[5] = (v902_data + (v868_data * (sycl::select_from_group(item.get_sub_group(), v899_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v905_data = r0[13];
              float v908_data = ir1[6];
              ir1[6] = (v908_data + (v868_data * (sycl::select_from_group(item.get_sub_group(), v905_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v911_data = r0[15];
              float v914_data = ir1[7];
              ir1[7] = (v914_data + (v868_data * (sycl::select_from_group(item.get_sub_group(), v911_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v917_data = r0[17];
              float v920_data = ir1[8];
              ir1[8] = (v920_data + (v868_data * (sycl::select_from_group(item.get_sub_group(), v917_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v923_data = glb_m1[(v17_lead + 256)];
              float v927_data = ir1[0];
              ir1[0] = (v927_data + (v923_data * (sycl::select_from_group(item.get_sub_group(), v869_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v933_data = ir1[1];
              ir1[1] = (v933_data + (v923_data * (sycl::select_from_group(item.get_sub_group(), v875_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v939_data = ir1[2];
              ir1[2] = (v939_data + (v923_data * (sycl::select_from_group(item.get_sub_group(), v881_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v945_data = ir1[3];
              ir1[3] = (v945_data + (v923_data * (sycl::select_from_group(item.get_sub_group(), v887_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v951_data = ir1[4];
              ir1[4] = (v951_data + (v923_data * (sycl::select_from_group(item.get_sub_group(), v893_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v957_data = ir1[5];
              ir1[5] = (v957_data + (v923_data * (sycl::select_from_group(item.get_sub_group(), v899_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v963_data = ir1[6];
              ir1[6] = (v963_data + (v923_data * (sycl::select_from_group(item.get_sub_group(), v905_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v969_data = ir1[7];
              ir1[7] = (v969_data + (v923_data * (sycl::select_from_group(item.get_sub_group(), v911_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v975_data = ir1[8];
              ir1[8] = (v975_data + (v923_data * (sycl::select_from_group(item.get_sub_group(), v917_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              // r1 = ir1
              #pragma unroll
              for (int32_t v977_n0 = 0; v977_n0 < 1; ++v977_n0) {
                #pragma unroll
                for (int32_t v978_n1 = 0; v978_n1 < 9; ++v978_n1) {
                  int32_t v979_a = v977_n0 + v978_n1;
                  float v980_data = ir1[v979_a];
                  r1[v979_a] = v980_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v981_i0 = 0; v981_i0 < 1; ++v981_i0) {
                int32_t v986_lead = v17_lead + (v981_i0 * 16);
                #pragma unroll
                for (int32_t v982_i1 = 0; v982_i1 < 9; ++v982_i1) {
                  float v984_data = r1[(v981_i0 + v982_i1)];
                  glb_m0[(v986_lead + (v982_i1 * 16))] = v984_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
            }
          }
        }
      });
    }
  });
}

