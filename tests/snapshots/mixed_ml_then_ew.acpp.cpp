// === base name ===
kernel_9a47ba8b8dea71e3

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_9a47ba8b8dea71e3 = {{8, 2, 1}, 8, 8, 1, 2, 64, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_9a47ba8b8dea71e3(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_9a47ba8b8dea71e3(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_9a47ba8b8dea71e3(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 16 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_9a47ba8b8dea71e3(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_9a47ba8b8dea71e3(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_9a47ba8b8dea71e3(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_9a47ba8b8dea71e3(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (16, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 8 lanes x 2 per block = block 8x2x1, 64 B shared, occupancy grid
        // operands:
        //   m0 8×8(8×8) {0..8}×{0..8} strided
        //   m1 8×8(8×8) {0..8}×{0..8} strided
        //   m2 8×8(8×8) {0..8}×{0..8} strided
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   C = abs(TMP)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,2,1],"cooperative":false,"lead_width":1,"mults_per_block":2,"persistent":true,"sections":[{"barrier":false,"mults_per_block":2,"shared_elements":16}],"shared_bytes":64,"shared_elements":16,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[8 * item.get_local_id(1) + 0];
          size_t v8_batchIdLane0 = item.get_local_id(1) % 2;
          size_t v11_batchIdGroup0First = ((item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)) < numElements0) ? (item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)) : 0u;
          int32_t v28_lead = item.get_local_id(2) % 8;
          float r0[8]{};
          size_t v443_row = v11_batchIdGroup0First + v8_batchIdLane0;
          const bool peel_batchIdActive0 = v443_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v443_row]));
          const float *const __restrict__ peel_glb_m0 = &m0[(peel_batchIdActive0 ? v443_row : v11_batchIdGroup0First) * 64 + 0 + m0_extraOffset];
          // r0 = load{g>r}(peel_glb_m0);
          #pragma unroll
          for (int32_t v451_i0 = 0; v451_i0 < 1; ++v451_i0) {
            int32_t v453_lead = v28_lead + (v451_i0 * 8);
            #pragma unroll
            for (int32_t v454_i1 = 0; v454_i1 < 8; ++v454_i1) {
              float v457_data = peel_glb_m0[(v453_lead + (v454_i1 * 8))];
              r0[(v451_i0 + v454_i1)] = v457_data;
            }
          }
          for (size_t v12_batchIdGroup0 = (item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)); v12_batchIdGroup0 < numElements0; v12_batchIdGroup0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v440_groupNextAhead = v12_batchIdGroup0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v442_groupNext = (v440_groupNextAhead < numElements0) ? v440_groupNextAhead : v12_batchIdGroup0;
            size_t v447_row = v442_groupNext + v8_batchIdLane0;
            const bool wrap_batchIdActive0 = v447_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v447_row]));
            const float *const __restrict__ wrap_glb_m0 = &m0[(wrap_batchIdActive0 ? v447_row : v442_groupNext) * 64 + 0 + m0_extraOffset];
            size_t v13_row = v12_batchIdGroup0 + v8_batchIdLane0;
            const bool batchIdActive0 = v13_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v13_row]));
            size_t v15_batchId0 = batchIdActive0 ? v13_row : v12_batchIdGroup0;
            size_t v16_ahead1 = v15_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v18_batchId1 = (v16_ahead1 < numElements0) ? v16_ahead1 : v15_batchId0;
            const float *const __restrict__ glb_m1 = &m1[v15_batchId0 * 64 + 0 + m1_extraOffset];
            float *const __restrict__ glb_m2 = &m2[v15_batchId0 * 64 + 0 + m2_extraOffset];
            float r1[8]{};
            // r1 = load{g>r}(glb_m1);
            #pragma unroll
            for (int32_t v38_i0 = 0; v38_i0 < 1; ++v38_i0) {
              int32_t v41_lead = v28_lead + (v38_i0 * 8);
              #pragma unroll
              for (int32_t v39_i1 = 0; v39_i1 < 8; ++v39_i1) {
                float v44_data = glb_m1[(v41_lead + (v39_i1 * 8))];
                r1[(v38_i0 + v39_i1)] = v44_data;
              }
            }
            float r2[8]{};
            // r2 = +(r0 * r1) + None
            // [(0, 8), (0, 8)] [(0, 8)]
            float v47_data = r0[0];
            float v48_data = r1[0];
            float v51_data = r2[0];
            r2[0] = (v51_data + (v47_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v54_data = r1[1];
            float v57_data = r2[1];
            r2[1] = (v57_data + (v47_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v60_data = r1[2];
            float v63_data = r2[2];
            r2[2] = (v63_data + (v47_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v66_data = r1[3];
            float v69_data = r2[3];
            r2[3] = (v69_data + (v47_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v72_data = r1[4];
            float v75_data = r2[4];
            r2[4] = (v75_data + (v47_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v78_data = r1[5];
            float v81_data = r2[5];
            r2[5] = (v81_data + (v47_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v84_data = r1[6];
            float v87_data = r2[6];
            r2[6] = (v87_data + (v47_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v90_data = r1[7];
            float v93_data = r2[7];
            r2[7] = (v93_data + (v47_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v95_data = r0[1];
            float v99_data = r2[0];
            r2[0] = (v99_data + (v95_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v105_data = r2[1];
            r2[1] = (v105_data + (v95_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v111_data = r2[2];
            r2[2] = (v111_data + (v95_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v117_data = r2[3];
            r2[3] = (v117_data + (v95_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v123_data = r2[4];
            r2[4] = (v123_data + (v95_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v129_data = r2[5];
            r2[5] = (v129_data + (v95_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v135_data = r2[6];
            r2[6] = (v135_data + (v95_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v141_data = r2[7];
            r2[7] = (v141_data + (v95_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v143_data = r0[2];
            float v147_data = r2[0];
            r2[0] = (v147_data + (v143_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v153_data = r2[1];
            r2[1] = (v153_data + (v143_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v159_data = r2[2];
            r2[2] = (v159_data + (v143_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v165_data = r2[3];
            r2[3] = (v165_data + (v143_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v171_data = r2[4];
            r2[4] = (v171_data + (v143_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v177_data = r2[5];
            r2[5] = (v177_data + (v143_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v183_data = r2[6];
            r2[6] = (v183_data + (v143_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v189_data = r2[7];
            r2[7] = (v189_data + (v143_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v191_data = r0[3];
            float v195_data = r2[0];
            r2[0] = (v195_data + (v191_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v201_data = r2[1];
            r2[1] = (v201_data + (v191_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v207_data = r2[2];
            r2[2] = (v207_data + (v191_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v213_data = r2[3];
            r2[3] = (v213_data + (v191_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v219_data = r2[4];
            r2[4] = (v219_data + (v191_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v225_data = r2[5];
            r2[5] = (v225_data + (v191_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v231_data = r2[6];
            r2[6] = (v231_data + (v191_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v237_data = r2[7];
            r2[7] = (v237_data + (v191_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v239_data = r0[4];
            float v243_data = r2[0];
            r2[0] = (v243_data + (v239_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v249_data = r2[1];
            r2[1] = (v249_data + (v239_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v255_data = r2[2];
            r2[2] = (v255_data + (v239_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v261_data = r2[3];
            r2[3] = (v261_data + (v239_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v267_data = r2[4];
            r2[4] = (v267_data + (v239_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v273_data = r2[5];
            r2[5] = (v273_data + (v239_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v279_data = r2[6];
            r2[6] = (v279_data + (v239_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v285_data = r2[7];
            r2[7] = (v285_data + (v239_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v287_data = r0[5];
            float v291_data = r2[0];
            r2[0] = (v291_data + (v287_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v297_data = r2[1];
            r2[1] = (v297_data + (v287_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v303_data = r2[2];
            r2[2] = (v303_data + (v287_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v309_data = r2[3];
            r2[3] = (v309_data + (v287_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v315_data = r2[4];
            r2[4] = (v315_data + (v287_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v321_data = r2[5];
            r2[5] = (v321_data + (v287_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v327_data = r2[6];
            r2[6] = (v327_data + (v287_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v333_data = r2[7];
            r2[7] = (v333_data + (v287_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v335_data = r0[6];
            float v339_data = r2[0];
            r2[0] = (v339_data + (v335_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v345_data = r2[1];
            r2[1] = (v345_data + (v335_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v351_data = r2[2];
            r2[2] = (v351_data + (v335_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v357_data = r2[3];
            r2[3] = (v357_data + (v335_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v363_data = r2[4];
            r2[4] = (v363_data + (v335_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v369_data = r2[5];
            r2[5] = (v369_data + (v335_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v375_data = r2[6];
            r2[6] = (v375_data + (v335_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v381_data = r2[7];
            r2[7] = (v381_data + (v335_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v383_data = r0[7];
            // r0 = load{g>r}(wrap_glb_m0);
            #pragma unroll
            for (int32_t v459_i0 = 0; v459_i0 < 1; ++v459_i0) {
              int32_t v461_lead = v28_lead + (v459_i0 * 8);
              #pragma unroll
              for (int32_t v462_i1 = 0; v462_i1 < 8; ++v462_i1) {
                float v465_data = wrap_glb_m0[(v461_lead + (v462_i1 * 8))];
                r0[(v459_i0 + v462_i1)] = v465_data;
              }
            }
            float v387_data = r2[0];
            r2[0] = (v387_data + (v383_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v393_data = r2[1];
            r2[1] = (v393_data + (v383_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v399_data = r2[2];
            r2[2] = (v399_data + (v383_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v405_data = r2[3];
            r2[3] = (v405_data + (v383_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v411_data = r2[4];
            r2[4] = (v411_data + (v383_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v417_data = r2[5];
            r2[5] = (v417_data + (v383_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v423_data = r2[6];
            r2[6] = (v423_data + (v383_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v429_data = r2[7];
            r2[7] = (v429_data + (v383_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            // glb_m2 = abs(r2)
            #pragma unroll
            for (int32_t v431_k0 = 0; v431_k0 < 1; ++v431_k0) {
              #pragma unroll
              for (int32_t v432_k1 = 0; v432_k1 < 8; ++v432_k1) {
                float v434_data = r2[(v431_k0 + v432_k1)];
                float v435_e = sycl::fabs(v434_data);
                if (batchIdActive0) {
                  glb_m2[((v28_lead + (v431_k0 * 8)) + (v432_k1 * 8))] = v435_e;
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

