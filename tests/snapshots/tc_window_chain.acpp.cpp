// === base name ===
kernel_3c42764e78270781

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_3c42764e78270781 = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_3c42764e78270781(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_3c42764e78270781(const float * m0, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_3c42764e78270781(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_3c42764e78270781(const float * m0, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_3c42764e78270781(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_3c42764e78270781(stream, grid, block, m0, m1, m1_extraOffset, m2, m2_extraOffset, m3, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_3c42764e78270781(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 16×20(16×17) {0..16}×{1..18} none
        //   m1 20×9(17×9) {1..18}×{0..9} strided
        //   m2 16×9(16×9) {0..16}×{0..9} strided
        //   m3 16×20(16×15) {0..16}×{1..16} none
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   m2[i,j] = m3[i,k] × t0[k,j]@{1..16}×{0..9}
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"none","alias":"A1","bbox":[[0,1],[16,18]],"name":"m0","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[1,0],[18,9]],"name":"m1","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[16,9]],"name":"m2","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"none","alias":"A2","bbox":[[0,1],[16,16]],"name":"m3","ordered":false,"parts":1,"shape":[16,20],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[16,18]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[18,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[16,16]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,20]},{"addressing":"pointer_based","bbox":[[1,0],[16,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[16,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          const float *const __restrict__ glb_m0 = &m0[0];
          const float *const __restrict__ glb_m3 = &m3[0];
          for (size_t v5_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v5_batchId0 < numElements0; v5_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v6_ahead1 = v5_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 153 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 144 + 0 + m2_extraOffset];
              float r0[18]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v18_lead = item.get_local_id(2) % 16;
              if (v18_lead >= 1) {
                int32_t v23_a = v18_lead - 1;
                #pragma unroll
                for (int32_t v20_i1 = 0; v20_i1 < 9; ++v20_i1) {
                  float v26_data = glb_m1[(v23_a + (v20_i1 * 17))];
                  r0[(v20_i1 * 2)] = v26_data;
                }
              }
              if (v18_lead < 2) {
                int32_t v33_a = (v18_lead + 16_i32) - 1;
                #pragma unroll
                for (int32_t v30_i1 = 0; v30_i1 < 9; ++v30_i1) {
                  float v36_data = glb_m1[(v33_a + (v30_i1 * 17))];
                  r0[(1 + (v30_i1 * 2))] = v36_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              float r1[9]{};
              // r1 = +(glb_m0 * r0) + None
              // [(0, 16), (0, 9)] [(1, 18)]
              float v43_data = glb_m0[v18_lead];
              float v44_data = r0[0];
              float v47_data = r1[0];
              r1[0] = (v47_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v50_data = r0[2];
              float v53_data = r1[1];
              r1[1] = (v53_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v56_data = r0[4];
              float v59_data = r1[2];
              r1[2] = (v59_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v62_data = r0[6];
              float v65_data = r1[3];
              r1[3] = (v65_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v68_data = r0[8];
              float v71_data = r1[4];
              r1[4] = (v71_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v74_data = r0[10];
              float v77_data = r1[5];
              r1[5] = (v77_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v80_data = r0[12];
              float v83_data = r1[6];
              r1[6] = (v83_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v86_data = r0[14];
              float v89_data = r1[7];
              r1[7] = (v89_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v92_data = r0[16];
              float v95_data = r1[8];
              r1[8] = (v95_data + (v43_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              int32_t v97_a = v18_lead + 16;
              float v98_data = glb_m0[v97_a];
              float v102_data = r1[0];
              r1[0] = (v102_data + (v98_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v108_data = r1[1];
              r1[1] = (v108_data + (v98_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v114_data = r1[2];
              r1[2] = (v114_data + (v98_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v120_data = r1[3];
              r1[3] = (v120_data + (v98_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v126_data = r1[4];
              r1[4] = (v126_data + (v98_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v132_data = r1[5];
              r1[5] = (v132_data + (v98_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v138_data = r1[6];
              r1[6] = (v138_data + (v98_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v144_data = r1[7];
              r1[7] = (v144_data + (v98_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v150_data = r1[8];
              r1[8] = (v150_data + (v98_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              int32_t v152_a = v18_lead + 32;
              float v153_data = glb_m0[v152_a];
              float v157_data = r1[0];
              r1[0] = (v157_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v163_data = r1[1];
              r1[1] = (v163_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v169_data = r1[2];
              r1[2] = (v169_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v175_data = r1[3];
              r1[3] = (v175_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v181_data = r1[4];
              r1[4] = (v181_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v187_data = r1[5];
              r1[5] = (v187_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v193_data = r1[6];
              r1[6] = (v193_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v199_data = r1[7];
              r1[7] = (v199_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v205_data = r1[8];
              r1[8] = (v205_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              int32_t v207_a = v18_lead + 48;
              float v208_data = glb_m0[v207_a];
              float v212_data = r1[0];
              r1[0] = (v212_data + (v208_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v218_data = r1[1];
              r1[1] = (v218_data + (v208_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v224_data = r1[2];
              r1[2] = (v224_data + (v208_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v230_data = r1[3];
              r1[3] = (v230_data + (v208_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v236_data = r1[4];
              r1[4] = (v236_data + (v208_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v242_data = r1[5];
              r1[5] = (v242_data + (v208_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v248_data = r1[6];
              r1[6] = (v248_data + (v208_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v254_data = r1[7];
              r1[7] = (v254_data + (v208_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v260_data = r1[8];
              r1[8] = (v260_data + (v208_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              int32_t v262_a = v18_lead + 64;
              float v263_data = glb_m0[v262_a];
              float v267_data = r1[0];
              r1[0] = (v267_data + (v263_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v273_data = r1[1];
              r1[1] = (v273_data + (v263_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v279_data = r1[2];
              r1[2] = (v279_data + (v263_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v285_data = r1[3];
              r1[3] = (v285_data + (v263_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v291_data = r1[4];
              r1[4] = (v291_data + (v263_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v297_data = r1[5];
              r1[5] = (v297_data + (v263_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v303_data = r1[6];
              r1[6] = (v303_data + (v263_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v309_data = r1[7];
              r1[7] = (v309_data + (v263_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v315_data = r1[8];
              r1[8] = (v315_data + (v263_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              int32_t v317_a = v18_lead + 80;
              float v318_data = glb_m0[v317_a];
              float v322_data = r1[0];
              r1[0] = (v322_data + (v318_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v328_data = r1[1];
              r1[1] = (v328_data + (v318_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v334_data = r1[2];
              r1[2] = (v334_data + (v318_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v340_data = r1[3];
              r1[3] = (v340_data + (v318_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v346_data = r1[4];
              r1[4] = (v346_data + (v318_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v352_data = r1[5];
              r1[5] = (v352_data + (v318_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v358_data = r1[6];
              r1[6] = (v358_data + (v318_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v364_data = r1[7];
              r1[7] = (v364_data + (v318_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v370_data = r1[8];
              r1[8] = (v370_data + (v318_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              int32_t v372_a = v18_lead + 96;
              float v373_data = glb_m0[v372_a];
              float v377_data = r1[0];
              r1[0] = (v377_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v383_data = r1[1];
              r1[1] = (v383_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v389_data = r1[2];
              r1[2] = (v389_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v395_data = r1[3];
              r1[3] = (v395_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v401_data = r1[4];
              r1[4] = (v401_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v407_data = r1[5];
              r1[5] = (v407_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v413_data = r1[6];
              r1[6] = (v413_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v419_data = r1[7];
              r1[7] = (v419_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v425_data = r1[8];
              r1[8] = (v425_data + (v373_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              int32_t v427_a = v18_lead + 112;
              float v428_data = glb_m0[v427_a];
              float v432_data = r1[0];
              r1[0] = (v432_data + (v428_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v438_data = r1[1];
              r1[1] = (v438_data + (v428_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v444_data = r1[2];
              r1[2] = (v444_data + (v428_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v450_data = r1[3];
              r1[3] = (v450_data + (v428_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v456_data = r1[4];
              r1[4] = (v456_data + (v428_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v462_data = r1[5];
              r1[5] = (v462_data + (v428_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v468_data = r1[6];
              r1[6] = (v468_data + (v428_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v474_data = r1[7];
              r1[7] = (v474_data + (v428_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v480_data = r1[8];
              r1[8] = (v480_data + (v428_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              int32_t v482_a = v18_lead + 128;
              float v483_data = glb_m0[v482_a];
              float v487_data = r1[0];
              r1[0] = (v487_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v493_data = r1[1];
              r1[1] = (v493_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v499_data = r1[2];
              r1[2] = (v499_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v505_data = r1[3];
              r1[3] = (v505_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v511_data = r1[4];
              r1[4] = (v511_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v517_data = r1[5];
              r1[5] = (v517_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v523_data = r1[6];
              r1[6] = (v523_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v529_data = r1[7];
              r1[7] = (v529_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v535_data = r1[8];
              r1[8] = (v535_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              int32_t v537_a = v18_lead + 144;
              float v538_data = glb_m0[v537_a];
              float v542_data = r1[0];
              r1[0] = (v542_data + (v538_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v548_data = r1[1];
              r1[1] = (v548_data + (v538_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v554_data = r1[2];
              r1[2] = (v554_data + (v538_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v560_data = r1[3];
              r1[3] = (v560_data + (v538_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v566_data = r1[4];
              r1[4] = (v566_data + (v538_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v572_data = r1[5];
              r1[5] = (v572_data + (v538_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v578_data = r1[6];
              r1[6] = (v578_data + (v538_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v584_data = r1[7];
              r1[7] = (v584_data + (v538_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v590_data = r1[8];
              r1[8] = (v590_data + (v538_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              int32_t v592_a = v18_lead + 160;
              float v593_data = glb_m0[v592_a];
              float v597_data = r1[0];
              r1[0] = (v597_data + (v593_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v603_data = r1[1];
              r1[1] = (v603_data + (v593_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v609_data = r1[2];
              r1[2] = (v609_data + (v593_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v615_data = r1[3];
              r1[3] = (v615_data + (v593_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v621_data = r1[4];
              r1[4] = (v621_data + (v593_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v627_data = r1[5];
              r1[5] = (v627_data + (v593_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v633_data = r1[6];
              r1[6] = (v633_data + (v593_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v639_data = r1[7];
              r1[7] = (v639_data + (v593_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v645_data = r1[8];
              r1[8] = (v645_data + (v593_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              int32_t v647_a = v18_lead + 176;
              float v648_data = glb_m0[v647_a];
              float v652_data = r1[0];
              r1[0] = (v652_data + (v648_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v658_data = r1[1];
              r1[1] = (v658_data + (v648_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v664_data = r1[2];
              r1[2] = (v664_data + (v648_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v670_data = r1[3];
              r1[3] = (v670_data + (v648_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v676_data = r1[4];
              r1[4] = (v676_data + (v648_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v682_data = r1[5];
              r1[5] = (v682_data + (v648_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v688_data = r1[6];
              r1[6] = (v688_data + (v648_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v694_data = r1[7];
              r1[7] = (v694_data + (v648_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v700_data = r1[8];
              r1[8] = (v700_data + (v648_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              int32_t v702_a = v18_lead + 192;
              float v703_data = glb_m0[v702_a];
              float v707_data = r1[0];
              r1[0] = (v707_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v713_data = r1[1];
              r1[1] = (v713_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v719_data = r1[2];
              r1[2] = (v719_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v725_data = r1[3];
              r1[3] = (v725_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v731_data = r1[4];
              r1[4] = (v731_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v737_data = r1[5];
              r1[5] = (v737_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v743_data = r1[6];
              r1[6] = (v743_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v749_data = r1[7];
              r1[7] = (v749_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v755_data = r1[8];
              r1[8] = (v755_data + (v703_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              int32_t v757_a = v18_lead + 208;
              float v758_data = glb_m0[v757_a];
              float v762_data = r1[0];
              r1[0] = (v762_data + (v758_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v768_data = r1[1];
              r1[1] = (v768_data + (v758_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v774_data = r1[2];
              r1[2] = (v774_data + (v758_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v780_data = r1[3];
              r1[3] = (v780_data + (v758_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v786_data = r1[4];
              r1[4] = (v786_data + (v758_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v792_data = r1[5];
              r1[5] = (v792_data + (v758_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v798_data = r1[6];
              r1[6] = (v798_data + (v758_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v804_data = r1[7];
              r1[7] = (v804_data + (v758_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v810_data = r1[8];
              r1[8] = (v810_data + (v758_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              int32_t v812_a = v18_lead + 224;
              float v813_data = glb_m0[v812_a];
              float v817_data = r1[0];
              r1[0] = (v817_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v823_data = r1[1];
              r1[1] = (v823_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v829_data = r1[2];
              r1[2] = (v829_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v835_data = r1[3];
              r1[3] = (v835_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v841_data = r1[4];
              r1[4] = (v841_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v847_data = r1[5];
              r1[5] = (v847_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v853_data = r1[6];
              r1[6] = (v853_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v859_data = r1[7];
              r1[7] = (v859_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v865_data = r1[8];
              r1[8] = (v865_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v868_data = glb_m0[(v18_lead + 240)];
              float v869_data = r0[1];
              float v872_data = r1[0];
              r1[0] = (v872_data + (v868_data * (sycl::select_from_group(item.get_sub_group(), v869_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v875_data = r0[3];
              float v878_data = r1[1];
              r1[1] = (v878_data + (v868_data * (sycl::select_from_group(item.get_sub_group(), v875_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v881_data = r0[5];
              float v884_data = r1[2];
              r1[2] = (v884_data + (v868_data * (sycl::select_from_group(item.get_sub_group(), v881_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v887_data = r0[7];
              float v890_data = r1[3];
              r1[3] = (v890_data + (v868_data * (sycl::select_from_group(item.get_sub_group(), v887_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v893_data = r0[9];
              float v896_data = r1[4];
              r1[4] = (v896_data + (v868_data * (sycl::select_from_group(item.get_sub_group(), v893_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v899_data = r0[11];
              float v902_data = r1[5];
              r1[5] = (v902_data + (v868_data * (sycl::select_from_group(item.get_sub_group(), v899_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v905_data = r0[13];
              float v908_data = r1[6];
              r1[6] = (v908_data + (v868_data * (sycl::select_from_group(item.get_sub_group(), v905_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v911_data = r0[15];
              float v914_data = r1[7];
              r1[7] = (v914_data + (v868_data * (sycl::select_from_group(item.get_sub_group(), v911_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v917_data = r0[17];
              float v920_data = r1[8];
              r1[8] = (v920_data + (v868_data * (sycl::select_from_group(item.get_sub_group(), v917_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v923_data = glb_m0[(v18_lead + 256)];
              float v927_data = r1[0];
              r1[0] = (v927_data + (v923_data * (sycl::select_from_group(item.get_sub_group(), v869_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v933_data = r1[1];
              r1[1] = (v933_data + (v923_data * (sycl::select_from_group(item.get_sub_group(), v875_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v939_data = r1[2];
              r1[2] = (v939_data + (v923_data * (sycl::select_from_group(item.get_sub_group(), v881_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v945_data = r1[3];
              r1[3] = (v945_data + (v923_data * (sycl::select_from_group(item.get_sub_group(), v887_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v951_data = r1[4];
              r1[4] = (v951_data + (v923_data * (sycl::select_from_group(item.get_sub_group(), v893_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v957_data = r1[5];
              r1[5] = (v957_data + (v923_data * (sycl::select_from_group(item.get_sub_group(), v899_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v963_data = r1[6];
              r1[6] = (v963_data + (v923_data * (sycl::select_from_group(item.get_sub_group(), v905_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v969_data = r1[7];
              r1[7] = (v969_data + (v923_data * (sycl::select_from_group(item.get_sub_group(), v911_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v975_data = r1[8];
              r1[8] = (v975_data + (v923_data * (sycl::select_from_group(item.get_sub_group(), v917_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float r2[9]{};
              // ir2 = +(glb_m3 * r1)
              // [(0, 16), (0, 9)] [(1, 16)]
              float ir2[9]{};
              float v982_data = glb_m3[v18_lead];
              float v983_data = r1[0];
              float v986_data = ir2[0];
              ir2[0] = (v986_data + (v982_data * (sycl::select_from_group(item.get_sub_group(), v983_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v989_data = r1[1];
              float v992_data = ir2[1];
              ir2[1] = (v992_data + (v982_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v995_data = r1[2];
              float v998_data = ir2[2];
              ir2[2] = (v998_data + (v982_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1001_data = r1[3];
              float v1004_data = ir2[3];
              ir2[3] = (v1004_data + (v982_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1007_data = r1[4];
              float v1010_data = ir2[4];
              ir2[4] = (v1010_data + (v982_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1013_data = r1[5];
              float v1016_data = ir2[5];
              ir2[5] = (v1016_data + (v982_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1019_data = r1[6];
              float v1022_data = ir2[6];
              ir2[6] = (v1022_data + (v982_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1025_data = r1[7];
              float v1028_data = ir2[7];
              ir2[7] = (v1028_data + (v982_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1031_data = r1[8];
              float v1034_data = ir2[8];
              ir2[8] = (v1034_data + (v982_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1037_data = glb_m3[v97_a];
              float v1041_data = ir2[0];
              ir2[0] = (v1041_data + (v1037_data * (sycl::select_from_group(item.get_sub_group(), v983_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1047_data = ir2[1];
              ir2[1] = (v1047_data + (v1037_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1053_data = ir2[2];
              ir2[2] = (v1053_data + (v1037_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1059_data = ir2[3];
              ir2[3] = (v1059_data + (v1037_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1065_data = ir2[4];
              ir2[4] = (v1065_data + (v1037_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1071_data = ir2[5];
              ir2[5] = (v1071_data + (v1037_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1077_data = ir2[6];
              ir2[6] = (v1077_data + (v1037_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1083_data = ir2[7];
              ir2[7] = (v1083_data + (v1037_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1089_data = ir2[8];
              ir2[8] = (v1089_data + (v1037_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1092_data = glb_m3[v152_a];
              float v1096_data = ir2[0];
              ir2[0] = (v1096_data + (v1092_data * (sycl::select_from_group(item.get_sub_group(), v983_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1102_data = ir2[1];
              ir2[1] = (v1102_data + (v1092_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1108_data = ir2[2];
              ir2[2] = (v1108_data + (v1092_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1114_data = ir2[3];
              ir2[3] = (v1114_data + (v1092_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1120_data = ir2[4];
              ir2[4] = (v1120_data + (v1092_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1126_data = ir2[5];
              ir2[5] = (v1126_data + (v1092_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1132_data = ir2[6];
              ir2[6] = (v1132_data + (v1092_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1138_data = ir2[7];
              ir2[7] = (v1138_data + (v1092_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1144_data = ir2[8];
              ir2[8] = (v1144_data + (v1092_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1147_data = glb_m3[v207_a];
              float v1151_data = ir2[0];
              ir2[0] = (v1151_data + (v1147_data * (sycl::select_from_group(item.get_sub_group(), v983_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1157_data = ir2[1];
              ir2[1] = (v1157_data + (v1147_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1163_data = ir2[2];
              ir2[2] = (v1163_data + (v1147_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1169_data = ir2[3];
              ir2[3] = (v1169_data + (v1147_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1175_data = ir2[4];
              ir2[4] = (v1175_data + (v1147_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1181_data = ir2[5];
              ir2[5] = (v1181_data + (v1147_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1187_data = ir2[6];
              ir2[6] = (v1187_data + (v1147_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1193_data = ir2[7];
              ir2[7] = (v1193_data + (v1147_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1199_data = ir2[8];
              ir2[8] = (v1199_data + (v1147_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1202_data = glb_m3[v262_a];
              float v1206_data = ir2[0];
              ir2[0] = (v1206_data + (v1202_data * (sycl::select_from_group(item.get_sub_group(), v983_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1212_data = ir2[1];
              ir2[1] = (v1212_data + (v1202_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1218_data = ir2[2];
              ir2[2] = (v1218_data + (v1202_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1224_data = ir2[3];
              ir2[3] = (v1224_data + (v1202_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1230_data = ir2[4];
              ir2[4] = (v1230_data + (v1202_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1236_data = ir2[5];
              ir2[5] = (v1236_data + (v1202_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1242_data = ir2[6];
              ir2[6] = (v1242_data + (v1202_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1248_data = ir2[7];
              ir2[7] = (v1248_data + (v1202_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1254_data = ir2[8];
              ir2[8] = (v1254_data + (v1202_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1257_data = glb_m3[v317_a];
              float v1261_data = ir2[0];
              ir2[0] = (v1261_data + (v1257_data * (sycl::select_from_group(item.get_sub_group(), v983_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1267_data = ir2[1];
              ir2[1] = (v1267_data + (v1257_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1273_data = ir2[2];
              ir2[2] = (v1273_data + (v1257_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1279_data = ir2[3];
              ir2[3] = (v1279_data + (v1257_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1285_data = ir2[4];
              ir2[4] = (v1285_data + (v1257_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1291_data = ir2[5];
              ir2[5] = (v1291_data + (v1257_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1297_data = ir2[6];
              ir2[6] = (v1297_data + (v1257_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1303_data = ir2[7];
              ir2[7] = (v1303_data + (v1257_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1309_data = ir2[8];
              ir2[8] = (v1309_data + (v1257_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1312_data = glb_m3[v372_a];
              float v1316_data = ir2[0];
              ir2[0] = (v1316_data + (v1312_data * (sycl::select_from_group(item.get_sub_group(), v983_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1322_data = ir2[1];
              ir2[1] = (v1322_data + (v1312_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1328_data = ir2[2];
              ir2[2] = (v1328_data + (v1312_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1334_data = ir2[3];
              ir2[3] = (v1334_data + (v1312_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1340_data = ir2[4];
              ir2[4] = (v1340_data + (v1312_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1346_data = ir2[5];
              ir2[5] = (v1346_data + (v1312_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1352_data = ir2[6];
              ir2[6] = (v1352_data + (v1312_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1358_data = ir2[7];
              ir2[7] = (v1358_data + (v1312_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1364_data = ir2[8];
              ir2[8] = (v1364_data + (v1312_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1367_data = glb_m3[v427_a];
              float v1371_data = ir2[0];
              ir2[0] = (v1371_data + (v1367_data * (sycl::select_from_group(item.get_sub_group(), v983_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1377_data = ir2[1];
              ir2[1] = (v1377_data + (v1367_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1383_data = ir2[2];
              ir2[2] = (v1383_data + (v1367_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1389_data = ir2[3];
              ir2[3] = (v1389_data + (v1367_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1395_data = ir2[4];
              ir2[4] = (v1395_data + (v1367_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1401_data = ir2[5];
              ir2[5] = (v1401_data + (v1367_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1407_data = ir2[6];
              ir2[6] = (v1407_data + (v1367_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1413_data = ir2[7];
              ir2[7] = (v1413_data + (v1367_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1419_data = ir2[8];
              ir2[8] = (v1419_data + (v1367_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1422_data = glb_m3[v482_a];
              float v1426_data = ir2[0];
              ir2[0] = (v1426_data + (v1422_data * (sycl::select_from_group(item.get_sub_group(), v983_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1432_data = ir2[1];
              ir2[1] = (v1432_data + (v1422_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1438_data = ir2[2];
              ir2[2] = (v1438_data + (v1422_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1444_data = ir2[3];
              ir2[3] = (v1444_data + (v1422_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1450_data = ir2[4];
              ir2[4] = (v1450_data + (v1422_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1456_data = ir2[5];
              ir2[5] = (v1456_data + (v1422_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1462_data = ir2[6];
              ir2[6] = (v1462_data + (v1422_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1468_data = ir2[7];
              ir2[7] = (v1468_data + (v1422_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1474_data = ir2[8];
              ir2[8] = (v1474_data + (v1422_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1477_data = glb_m3[v537_a];
              float v1481_data = ir2[0];
              ir2[0] = (v1481_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v983_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1487_data = ir2[1];
              ir2[1] = (v1487_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1493_data = ir2[2];
              ir2[2] = (v1493_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1499_data = ir2[3];
              ir2[3] = (v1499_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1505_data = ir2[4];
              ir2[4] = (v1505_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1511_data = ir2[5];
              ir2[5] = (v1511_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1517_data = ir2[6];
              ir2[6] = (v1517_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1523_data = ir2[7];
              ir2[7] = (v1523_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1529_data = ir2[8];
              ir2[8] = (v1529_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1532_data = glb_m3[v592_a];
              float v1536_data = ir2[0];
              ir2[0] = (v1536_data + (v1532_data * (sycl::select_from_group(item.get_sub_group(), v983_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1542_data = ir2[1];
              ir2[1] = (v1542_data + (v1532_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1548_data = ir2[2];
              ir2[2] = (v1548_data + (v1532_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1554_data = ir2[3];
              ir2[3] = (v1554_data + (v1532_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1560_data = ir2[4];
              ir2[4] = (v1560_data + (v1532_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1566_data = ir2[5];
              ir2[5] = (v1566_data + (v1532_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1572_data = ir2[6];
              ir2[6] = (v1572_data + (v1532_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1578_data = ir2[7];
              ir2[7] = (v1578_data + (v1532_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1584_data = ir2[8];
              ir2[8] = (v1584_data + (v1532_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1587_data = glb_m3[v647_a];
              float v1591_data = ir2[0];
              ir2[0] = (v1591_data + (v1587_data * (sycl::select_from_group(item.get_sub_group(), v983_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1597_data = ir2[1];
              ir2[1] = (v1597_data + (v1587_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1603_data = ir2[2];
              ir2[2] = (v1603_data + (v1587_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1609_data = ir2[3];
              ir2[3] = (v1609_data + (v1587_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1615_data = ir2[4];
              ir2[4] = (v1615_data + (v1587_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1621_data = ir2[5];
              ir2[5] = (v1621_data + (v1587_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1627_data = ir2[6];
              ir2[6] = (v1627_data + (v1587_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1633_data = ir2[7];
              ir2[7] = (v1633_data + (v1587_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1639_data = ir2[8];
              ir2[8] = (v1639_data + (v1587_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1642_data = glb_m3[v702_a];
              float v1646_data = ir2[0];
              ir2[0] = (v1646_data + (v1642_data * (sycl::select_from_group(item.get_sub_group(), v983_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1652_data = ir2[1];
              ir2[1] = (v1652_data + (v1642_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1658_data = ir2[2];
              ir2[2] = (v1658_data + (v1642_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1664_data = ir2[3];
              ir2[3] = (v1664_data + (v1642_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1670_data = ir2[4];
              ir2[4] = (v1670_data + (v1642_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1676_data = ir2[5];
              ir2[5] = (v1676_data + (v1642_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1682_data = ir2[6];
              ir2[6] = (v1682_data + (v1642_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1688_data = ir2[7];
              ir2[7] = (v1688_data + (v1642_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1694_data = ir2[8];
              ir2[8] = (v1694_data + (v1642_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1697_data = glb_m3[v757_a];
              float v1701_data = ir2[0];
              ir2[0] = (v1701_data + (v1697_data * (sycl::select_from_group(item.get_sub_group(), v983_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1707_data = ir2[1];
              ir2[1] = (v1707_data + (v1697_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1713_data = ir2[2];
              ir2[2] = (v1713_data + (v1697_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1719_data = ir2[3];
              ir2[3] = (v1719_data + (v1697_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1725_data = ir2[4];
              ir2[4] = (v1725_data + (v1697_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1731_data = ir2[5];
              ir2[5] = (v1731_data + (v1697_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1737_data = ir2[6];
              ir2[6] = (v1737_data + (v1697_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1743_data = ir2[7];
              ir2[7] = (v1743_data + (v1697_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1749_data = ir2[8];
              ir2[8] = (v1749_data + (v1697_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1752_data = glb_m3[v812_a];
              float v1756_data = ir2[0];
              ir2[0] = (v1756_data + (v1752_data * (sycl::select_from_group(item.get_sub_group(), v983_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1762_data = ir2[1];
              ir2[1] = (v1762_data + (v1752_data * (sycl::select_from_group(item.get_sub_group(), v989_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1768_data = ir2[2];
              ir2[2] = (v1768_data + (v1752_data * (sycl::select_from_group(item.get_sub_group(), v995_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1774_data = ir2[3];
              ir2[3] = (v1774_data + (v1752_data * (sycl::select_from_group(item.get_sub_group(), v1001_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1780_data = ir2[4];
              ir2[4] = (v1780_data + (v1752_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1786_data = ir2[5];
              ir2[5] = (v1786_data + (v1752_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1792_data = ir2[6];
              ir2[6] = (v1792_data + (v1752_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1798_data = ir2[7];
              ir2[7] = (v1798_data + (v1752_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1804_data = ir2[8];
              ir2[8] = (v1804_data + (v1752_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              // r2 = ir2
              #pragma unroll
              for (int32_t v1806_n0 = 0; v1806_n0 < 1; ++v1806_n0) {
                #pragma unroll
                for (int32_t v1807_n1 = 0; v1807_n1 < 9; ++v1807_n1) {
                  int32_t v1808_a = v1806_n0 + v1807_n1;
                  float v1809_data = ir2[v1808_a];
                  r2[v1808_a] = v1809_data;
                }
              }
              // glb_m2 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v1810_i0 = 0; v1810_i0 < 1; ++v1810_i0) {
                int32_t v1815_lead = v18_lead + (v1810_i0 * 16);
                #pragma unroll
                for (int32_t v1811_i1 = 0; v1811_i1 < 9; ++v1811_i1) {
                  float v1813_data = r2[(v1810_i0 + v1811_i1)];
                  glb_m2[(v1815_lead + (v1811_i1 * 16))] = v1813_data;
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

