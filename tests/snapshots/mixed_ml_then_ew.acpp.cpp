// === base name ===
kernel_df15c9e90d7ae598

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_df15c9e90d7ae598 = {{8, 2, 1}, 8, 8, 1, 2, 576, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_df15c9e90d7ae598(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_df15c9e90d7ae598(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_df15c9e90d7ae598(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (8, 2, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 2 - 1) / 2;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 8;
  config.block[1] = 2;
  config.block[2] = 1;
  config.sharedMemBytes = 144 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_df15c9e90d7ae598(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_df15c9e90d7ae598(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_df15c9e90d7ae598(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_df15c9e90d7ae598(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (144, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 8 lanes x 2 per block = block 8x2x1, 576 B shared, occupancy grid
        // operands:
        //   m0 8×8(8×8) {0..8}×{0..8} strided
        //   m1 8×8(8×8) {0..8}×{0..8} strided
        //   m2 8×8(8×8) {0..8}×{0..8} strided
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   C = abs(TMP)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,2,1],"cooperative":false,"lead_width":1,"mults_per_block":2,"persistent":true,"sections":[{"barrier":false,"mults_per_block":2,"shared_elements":144}],"shared_bytes":576,"shared_elements":144,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[72 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[64];
          float * __restrict__ s0 = &localShrMem0[0];
          size_t v4_batchIdLane0 = item.get_local_id(1) % 2;
          int32_t v22_lead = item.get_local_id(2) % 8;
          for (size_t v5_batchIdGroup0 = (item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)); v5_batchIdGroup0 < numElements0; v5_batchIdGroup0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v6_row = v5_batchIdGroup0 + v4_batchIdLane0;
            const bool batchIdActive0 = v6_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v6_row]));
            size_t v8_batchId0 = batchIdActive0 ? v6_row : v5_batchIdGroup0;
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 64 + 0 + m0_extraOffset];
            const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 64 + 0 + m1_extraOffset];
            float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 64 + 0 + m2_extraOffset];
            float r0[8]{};
            // r0 = load{g>r}(glb_m0);
            #pragma unroll
            for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
              int32_t v26_lead = v22_lead + (v23_i0 * 8);
              #pragma unroll
              for (int32_t v24_i1 = 0; v24_i1 < 8; ++v24_i1) {
                float v29_data = glb_m0[(v26_lead + (v24_i1 * 8))];
                r0[(v23_i0 + v24_i1)] = v29_data;
              }
            }
            float r1[8]{};
            // r1 = load{g>r}(glb_m1);
            #pragma unroll
            for (int32_t v32_i0 = 0; v32_i0 < 1; ++v32_i0) {
              int32_t v35_lead = v22_lead + (v32_i0 * 8);
              #pragma unroll
              for (int32_t v33_i1 = 0; v33_i1 < 8; ++v33_i1) {
                float v38_data = glb_m1[(v35_lead + (v33_i1 * 8))];
                r1[(v32_i0 + v33_i1)] = v38_data;
              }
            }
            // wait(r0 = load{g>r}(glb_m0););
            // wait(r1 = load{g>r}(glb_m1););
            float r2[8]{};
            // r2 = +(r0 * r1) + None
            // [(0, 8), (0, 8)] [(0, 8)]
            float v41_data = r0[0];
            float v42_data = r1[0];
            float v45_data = r2[0];
            r2[0] = (v45_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v48_data = r1[1];
            float v51_data = r2[1];
            r2[1] = (v51_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v54_data = r1[2];
            float v57_data = r2[2];
            r2[2] = (v57_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v60_data = r1[3];
            float v63_data = r2[3];
            r2[3] = (v63_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v66_data = r1[4];
            float v69_data = r2[4];
            r2[4] = (v69_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v72_data = r1[5];
            float v75_data = r2[5];
            r2[5] = (v75_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v78_data = r1[6];
            float v81_data = r2[6];
            r2[6] = (v81_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v84_data = r1[7];
            float v87_data = r2[7];
            r2[7] = (v87_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v89_data = r0[1];
            float v93_data = r2[0];
            r2[0] = (v93_data + (v89_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v99_data = r2[1];
            r2[1] = (v99_data + (v89_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v105_data = r2[2];
            r2[2] = (v105_data + (v89_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v111_data = r2[3];
            r2[3] = (v111_data + (v89_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v117_data = r2[4];
            r2[4] = (v117_data + (v89_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v123_data = r2[5];
            r2[5] = (v123_data + (v89_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v129_data = r2[6];
            r2[6] = (v129_data + (v89_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v135_data = r2[7];
            r2[7] = (v135_data + (v89_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v137_data = r0[2];
            float v141_data = r2[0];
            r2[0] = (v141_data + (v137_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v147_data = r2[1];
            r2[1] = (v147_data + (v137_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v153_data = r2[2];
            r2[2] = (v153_data + (v137_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v159_data = r2[3];
            r2[3] = (v159_data + (v137_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v165_data = r2[4];
            r2[4] = (v165_data + (v137_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v171_data = r2[5];
            r2[5] = (v171_data + (v137_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v177_data = r2[6];
            r2[6] = (v177_data + (v137_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v183_data = r2[7];
            r2[7] = (v183_data + (v137_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v185_data = r0[3];
            float v189_data = r2[0];
            r2[0] = (v189_data + (v185_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v195_data = r2[1];
            r2[1] = (v195_data + (v185_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v201_data = r2[2];
            r2[2] = (v201_data + (v185_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v207_data = r2[3];
            r2[3] = (v207_data + (v185_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v213_data = r2[4];
            r2[4] = (v213_data + (v185_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v219_data = r2[5];
            r2[5] = (v219_data + (v185_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v225_data = r2[6];
            r2[6] = (v225_data + (v185_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v231_data = r2[7];
            r2[7] = (v231_data + (v185_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v233_data = r0[4];
            float v237_data = r2[0];
            r2[0] = (v237_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v243_data = r2[1];
            r2[1] = (v243_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v249_data = r2[2];
            r2[2] = (v249_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v255_data = r2[3];
            r2[3] = (v255_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v261_data = r2[4];
            r2[4] = (v261_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v267_data = r2[5];
            r2[5] = (v267_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v273_data = r2[6];
            r2[6] = (v273_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v279_data = r2[7];
            r2[7] = (v279_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v281_data = r0[5];
            float v285_data = r2[0];
            r2[0] = (v285_data + (v281_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v291_data = r2[1];
            r2[1] = (v291_data + (v281_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v297_data = r2[2];
            r2[2] = (v297_data + (v281_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v303_data = r2[3];
            r2[3] = (v303_data + (v281_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v309_data = r2[4];
            r2[4] = (v309_data + (v281_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v315_data = r2[5];
            r2[5] = (v315_data + (v281_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v321_data = r2[6];
            r2[6] = (v321_data + (v281_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v327_data = r2[7];
            r2[7] = (v327_data + (v281_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v329_data = r0[6];
            float v333_data = r2[0];
            r2[0] = (v333_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v339_data = r2[1];
            r2[1] = (v339_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v345_data = r2[2];
            r2[2] = (v345_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v351_data = r2[3];
            r2[3] = (v351_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v357_data = r2[4];
            r2[4] = (v357_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v363_data = r2[5];
            r2[5] = (v363_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v369_data = r2[6];
            r2[6] = (v369_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v375_data = r2[7];
            r2[7] = (v375_data + (v329_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v377_data = r0[7];
            float v381_data = r2[0];
            r2[0] = (v381_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v387_data = r2[1];
            r2[1] = (v387_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v393_data = r2[2];
            r2[2] = (v393_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v399_data = r2[3];
            r2[3] = (v399_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v405_data = r2[4];
            r2[4] = (v405_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v411_data = r2[5];
            r2[5] = (v411_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v417_data = r2[6];
            r2[6] = (v417_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v423_data = r2[7];
            r2[7] = (v423_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            // s0 = store{r>s}(localShrMem0, r2);
            #pragma unroll
            for (int32_t v425_i0 = 0; v425_i0 < 1; ++v425_i0) {
              int32_t v430_lead = v22_lead + (v425_i0 * 8);
              #pragma unroll
              for (int32_t v426_i1 = 0; v426_i1 < 8; ++v426_i1) {
                float v428_data = r2[(v425_i0 + v426_i1)];
                int32_t v432_a = v430_lead + (v426_i1 * 8);
                s0[(v432_a ^ ((v432_a >> 5) & 31))] = v428_data;
              }
            }
            item.barrier();
            // glb_m2 = abs(s0)
            #pragma unroll
            for (int32_t v436_k0 = 0; v436_k0 < 1; ++v436_k0) {
              int32_t v439_lead = v22_lead + (v436_k0 * 8);
              #pragma unroll
              for (int32_t v437_k1 = 0; v437_k1 < 8; ++v437_k1) {
                int32_t v441_a = v439_lead + (v437_k1 * 8);
                float v445_data = s0[(v441_a ^ ((v441_a >> 5) & 31))];
                float v446_e = sycl::fabs(v445_data);
                if (batchIdActive0) {
                  glb_m2[v441_a] = v446_e;
                }
              }
            }
            item.barrier();
          }
        }
      });
    }
  });
}

