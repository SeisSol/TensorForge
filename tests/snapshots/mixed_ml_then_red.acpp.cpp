// === base name ===
kernel_0dd6c1a895d3cf2e

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_0dd6c1a895d3cf2e = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_0dd6c1a895d3cf2e(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_0dd6c1a895d3cf2e(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_0dd6c1a895d3cf2e(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (16, 16, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 16 - 1) / 16;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 16;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_0dd6c1a895d3cf2e(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_0dd6c1a895d3cf2e(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_0dd6c1a895d3cf2e(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_0dd6c1a895d3cf2e(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 8×8(8×8) {0..8}×{0..8} strided
        //   m1 8×8(8×8) {0..8}×{0..8} strided
        //   m2 8(8) {0..8} strided
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   OUT = +(TMP, dims=[1])
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"OUT","bbox":[[0],[8]],"name":"m2","ordered":false,"parts":1,"shape":[8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0],[8]],"is_tmp":false,"name":"m2","offset":[0],"shape":[8]},"kind":"reduction","op":"+","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"target":[[0,-1]]}],"version":"0.0.1"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t v3_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v3_batchId0 < numElements0; v3_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v4_ahead1 = v3_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v3_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v3_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 64 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v3_batchId0 * 8 + 0 + m2_extraOffset];
              float r0[8]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v17_lead = item.get_local_id(2) % 16;
              bool v18_g = v17_lead < 8;
              if (v18_g) {
                #pragma unroll
                for (int32_t v19_i1 = 0; v19_i1 < 8; ++v19_i1) {
                  float v24_data = glb_m0[(v17_lead + (v19_i1 * 8))];
                  r0[v19_i1] = v24_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m1);
              if (v18_g) {
                #pragma unroll
                for (int32_t v27_i1 = 0; v27_i1 < 8; ++v27_i1) {
                  float v32_data = glb_m1[(v17_lead + (v27_i1 * 8))];
                  r1[v27_i1] = v32_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              // wait(r1 = load{g>r}(glb_m1););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              float v35_data = r0[0];
              float v36_data = r1[0];
              float v39_data = r2[0];
              r2[0] = (v39_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v42_data = r1[1];
              float v45_data = r2[1];
              r2[1] = (v45_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v48_data = r1[2];
              float v51_data = r2[2];
              r2[2] = (v51_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v54_data = r1[3];
              float v57_data = r2[3];
              r2[3] = (v57_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v60_data = r1[4];
              float v63_data = r2[4];
              r2[4] = (v63_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v66_data = r1[5];
              float v69_data = r2[5];
              r2[5] = (v69_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v72_data = r1[6];
              float v75_data = r2[6];
              r2[6] = (v75_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v78_data = r1[7];
              float v81_data = r2[7];
              r2[7] = (v81_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v83_data = r0[1];
              float v87_data = r2[0];
              r2[0] = (v87_data + (v83_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v93_data = r2[1];
              r2[1] = (v93_data + (v83_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v99_data = r2[2];
              r2[2] = (v99_data + (v83_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v105_data = r2[3];
              r2[3] = (v105_data + (v83_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v111_data = r2[4];
              r2[4] = (v111_data + (v83_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v117_data = r2[5];
              r2[5] = (v117_data + (v83_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v123_data = r2[6];
              r2[6] = (v123_data + (v83_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v129_data = r2[7];
              r2[7] = (v129_data + (v83_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v131_data = r0[2];
              float v135_data = r2[0];
              r2[0] = (v135_data + (v131_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v141_data = r2[1];
              r2[1] = (v141_data + (v131_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v147_data = r2[2];
              r2[2] = (v147_data + (v131_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v153_data = r2[3];
              r2[3] = (v153_data + (v131_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v159_data = r2[4];
              r2[4] = (v159_data + (v131_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v165_data = r2[5];
              r2[5] = (v165_data + (v131_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v171_data = r2[6];
              r2[6] = (v171_data + (v131_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v177_data = r2[7];
              r2[7] = (v177_data + (v131_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v179_data = r0[3];
              float v183_data = r2[0];
              r2[0] = (v183_data + (v179_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v189_data = r2[1];
              r2[1] = (v189_data + (v179_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v195_data = r2[2];
              r2[2] = (v195_data + (v179_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v201_data = r2[3];
              r2[3] = (v201_data + (v179_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v207_data = r2[4];
              r2[4] = (v207_data + (v179_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v213_data = r2[5];
              r2[5] = (v213_data + (v179_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v219_data = r2[6];
              r2[6] = (v219_data + (v179_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v225_data = r2[7];
              r2[7] = (v225_data + (v179_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v227_data = r0[4];
              float v231_data = r2[0];
              r2[0] = (v231_data + (v227_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v237_data = r2[1];
              r2[1] = (v237_data + (v227_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v243_data = r2[2];
              r2[2] = (v243_data + (v227_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v249_data = r2[3];
              r2[3] = (v249_data + (v227_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v255_data = r2[4];
              r2[4] = (v255_data + (v227_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v261_data = r2[5];
              r2[5] = (v261_data + (v227_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v267_data = r2[6];
              r2[6] = (v267_data + (v227_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v273_data = r2[7];
              r2[7] = (v273_data + (v227_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v275_data = r0[5];
              float v279_data = r2[0];
              r2[0] = (v279_data + (v275_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v285_data = r2[1];
              r2[1] = (v285_data + (v275_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v291_data = r2[2];
              r2[2] = (v291_data + (v275_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v297_data = r2[3];
              r2[3] = (v297_data + (v275_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v303_data = r2[4];
              r2[4] = (v303_data + (v275_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v309_data = r2[5];
              r2[5] = (v309_data + (v275_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v315_data = r2[6];
              r2[6] = (v315_data + (v275_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v321_data = r2[7];
              r2[7] = (v321_data + (v275_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v323_data = r0[6];
              float v327_data = r2[0];
              r2[0] = (v327_data + (v323_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v333_data = r2[1];
              r2[1] = (v333_data + (v323_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v339_data = r2[2];
              r2[2] = (v339_data + (v323_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v345_data = r2[3];
              r2[3] = (v345_data + (v323_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v351_data = r2[4];
              r2[4] = (v351_data + (v323_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v357_data = r2[5];
              r2[5] = (v357_data + (v323_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v363_data = r2[6];
              r2[6] = (v363_data + (v323_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v369_data = r2[7];
              r2[7] = (v369_data + (v323_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v371_data = r0[7];
              float v375_data = r2[0];
              r2[0] = (v375_data + (v371_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v381_data = r2[1];
              r2[1] = (v381_data + (v371_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v387_data = r2[2];
              r2[2] = (v387_data + (v371_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v393_data = r2[3];
              r2[3] = (v393_data + (v371_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v399_data = r2[4];
              r2[4] = (v399_data + (v371_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v405_data = r2[5];
              r2[5] = (v405_data + (v371_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v411_data = r2[6];
              r2[6] = (v411_data + (v371_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v417_data = r2[7];
              r2[7] = (v417_data + (v371_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              // glb_m2 = +(r2, dims=[1])
              if (v18_g) {
                float v420_acc0 = 0.0f;
                #pragma unroll
                for (int32_t v419_r1 = 0; v419_r1 < 8; ++v419_r1) {
                  float v422_data = r2[v419_r1];
                  v420_acc0 = (v420_acc0 + v422_data);
                }
                glb_m2[v17_lead] = v420_acc0;
              }
              sycl::group_barrier(item.get_sub_group());
            }
          }
        }
      });
    }
  });
}

