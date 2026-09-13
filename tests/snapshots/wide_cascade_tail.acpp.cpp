// === base name ===
kernel_97d65c141d387c19

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_97d65c141d387c19 = {{32, 1, 1}, 32, 24, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_97d65c141d387c19(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_97d65c141d387c19(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_97d65c141d387c19(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (32, 1, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 32;
  config.block[1] = 1;
  config.block[2] = 1;
  config.sharedMemBytes = 0 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_97d65c141d387c19(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_97d65c141d387c19(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_97d65c141d387c19(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_97d65c141d387c19(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes (24 active) x 1 per block = block 32x1x1, 0 B shared, occupancy grid
        // operands:
        //   m0 24×9(24×9) {0..24}×{0..9} strided
        //   m1 24×24(24×24) {0..24}×{0..24} strided
        //   m2 24×9(24×9) {0..24}×{0..9} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":24,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[24,9]],"name":"m0","ordered":false,"parts":1,"shape":[24,9],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[24,24]],"name":"m1","ordered":false,"parts":1,"shape":[24,24],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[24,9]],"name":"m2","ordered":false,"parts":1,"shape":[24,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[24,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[24,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[24,24]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[24,24]},{"addressing":"strided","bbox":[[0,0],[24,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[24,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          for (size_t v1_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v1_batchId0 < numElements0; v1_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v2_ahead1 = v1_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v1_batchId0 * 216 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v1_batchId0 * 576 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v1_batchId0 * 216 + 0 + m2_extraOffset];
              float r0[24]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v15_lead = item.get_local_id(2) % 32;
              bool v16_g = v15_lead < 24;
              if (v16_g) {
                #pragma unroll
                for (int32_t v17_i1 = 0; v17_i1 < 24; ++v17_i1) {
                  float v22_data = glb_m1[(v15_lead + (v17_i1 * 24))];
                  r0[v17_i1] = v22_data;
                }
              }
              float r1[9]{};
              // r1 = load{g>r}(glb_m2);
              if (v16_g) {
                #pragma unroll
                for (int32_t v25_i1 = 0; v25_i1 < 9; ++v25_i1) {
                  float v30_data = glb_m2[(v15_lead + (v25_i1 * 24))];
                  r1[v25_i1] = v30_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[9]{};
              // r2 = +(r0 * r1) + None
              // [(0, 24), (0, 9)] [(0, 24)]
              float ir2[9]{};
              float v34_data = r0[0];
              float v35_data = r1[0];
              float v38_data = ir2[0];
              ir2[0] = (v38_data + (v34_data * (sycl::select_from_group(item.get_sub_group(), v35_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v41_data = r1[1];
              float v44_data = ir2[1];
              ir2[1] = (v44_data + (v34_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v47_data = r1[2];
              float v50_data = ir2[2];
              ir2[2] = (v50_data + (v34_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v53_data = r1[3];
              float v56_data = ir2[3];
              ir2[3] = (v56_data + (v34_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v59_data = r1[4];
              float v62_data = ir2[4];
              ir2[4] = (v62_data + (v34_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v65_data = r1[5];
              float v68_data = ir2[5];
              ir2[5] = (v68_data + (v34_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v71_data = r1[6];
              float v74_data = ir2[6];
              ir2[6] = (v74_data + (v34_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v77_data = r1[7];
              float v80_data = ir2[7];
              ir2[7] = (v80_data + (v34_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v83_data = r1[8];
              float v86_data = ir2[8];
              ir2[8] = (v86_data + (v34_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v88_data = r0[1];
              float v92_data = ir2[0];
              ir2[0] = (v92_data + (v88_data * (sycl::select_from_group(item.get_sub_group(), v35_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v98_data = ir2[1];
              ir2[1] = (v98_data + (v88_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v104_data = ir2[2];
              ir2[2] = (v104_data + (v88_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v110_data = ir2[3];
              ir2[3] = (v110_data + (v88_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v116_data = ir2[4];
              ir2[4] = (v116_data + (v88_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v122_data = ir2[5];
              ir2[5] = (v122_data + (v88_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v128_data = ir2[6];
              ir2[6] = (v128_data + (v88_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v134_data = ir2[7];
              ir2[7] = (v134_data + (v88_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v140_data = ir2[8];
              ir2[8] = (v140_data + (v88_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v142_data = r0[2];
              float v146_data = ir2[0];
              ir2[0] = (v146_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v35_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v152_data = ir2[1];
              ir2[1] = (v152_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v158_data = ir2[2];
              ir2[2] = (v158_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v164_data = ir2[3];
              ir2[3] = (v164_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v170_data = ir2[4];
              ir2[4] = (v170_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v176_data = ir2[5];
              ir2[5] = (v176_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v182_data = ir2[6];
              ir2[6] = (v182_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v188_data = ir2[7];
              ir2[7] = (v188_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v194_data = ir2[8];
              ir2[8] = (v194_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v196_data = r0[3];
              float v200_data = ir2[0];
              ir2[0] = (v200_data + (v196_data * (sycl::select_from_group(item.get_sub_group(), v35_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v206_data = ir2[1];
              ir2[1] = (v206_data + (v196_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v212_data = ir2[2];
              ir2[2] = (v212_data + (v196_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v218_data = ir2[3];
              ir2[3] = (v218_data + (v196_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v224_data = ir2[4];
              ir2[4] = (v224_data + (v196_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v230_data = ir2[5];
              ir2[5] = (v230_data + (v196_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v236_data = ir2[6];
              ir2[6] = (v236_data + (v196_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v242_data = ir2[7];
              ir2[7] = (v242_data + (v196_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v248_data = ir2[8];
              ir2[8] = (v248_data + (v196_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v250_data = r0[4];
              float v254_data = ir2[0];
              ir2[0] = (v254_data + (v250_data * (sycl::select_from_group(item.get_sub_group(), v35_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v260_data = ir2[1];
              ir2[1] = (v260_data + (v250_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v266_data = ir2[2];
              ir2[2] = (v266_data + (v250_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v272_data = ir2[3];
              ir2[3] = (v272_data + (v250_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v278_data = ir2[4];
              ir2[4] = (v278_data + (v250_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v284_data = ir2[5];
              ir2[5] = (v284_data + (v250_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v290_data = ir2[6];
              ir2[6] = (v290_data + (v250_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v296_data = ir2[7];
              ir2[7] = (v296_data + (v250_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v302_data = ir2[8];
              ir2[8] = (v302_data + (v250_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v304_data = r0[5];
              float v308_data = ir2[0];
              ir2[0] = (v308_data + (v304_data * (sycl::select_from_group(item.get_sub_group(), v35_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v314_data = ir2[1];
              ir2[1] = (v314_data + (v304_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v320_data = ir2[2];
              ir2[2] = (v320_data + (v304_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v326_data = ir2[3];
              ir2[3] = (v326_data + (v304_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v332_data = ir2[4];
              ir2[4] = (v332_data + (v304_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v338_data = ir2[5];
              ir2[5] = (v338_data + (v304_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v344_data = ir2[6];
              ir2[6] = (v344_data + (v304_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v350_data = ir2[7];
              ir2[7] = (v350_data + (v304_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v356_data = ir2[8];
              ir2[8] = (v356_data + (v304_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v358_data = r0[6];
              float v362_data = ir2[0];
              ir2[0] = (v362_data + (v358_data * (sycl::select_from_group(item.get_sub_group(), v35_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v368_data = ir2[1];
              ir2[1] = (v368_data + (v358_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v374_data = ir2[2];
              ir2[2] = (v374_data + (v358_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v380_data = ir2[3];
              ir2[3] = (v380_data + (v358_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v386_data = ir2[4];
              ir2[4] = (v386_data + (v358_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v392_data = ir2[5];
              ir2[5] = (v392_data + (v358_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v398_data = ir2[6];
              ir2[6] = (v398_data + (v358_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v404_data = ir2[7];
              ir2[7] = (v404_data + (v358_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v410_data = ir2[8];
              ir2[8] = (v410_data + (v358_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v412_data = r0[7];
              float v416_data = ir2[0];
              ir2[0] = (v416_data + (v412_data * (sycl::select_from_group(item.get_sub_group(), v35_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v422_data = ir2[1];
              ir2[1] = (v422_data + (v412_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v428_data = ir2[2];
              ir2[2] = (v428_data + (v412_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v434_data = ir2[3];
              ir2[3] = (v434_data + (v412_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v440_data = ir2[4];
              ir2[4] = (v440_data + (v412_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v446_data = ir2[5];
              ir2[5] = (v446_data + (v412_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v452_data = ir2[6];
              ir2[6] = (v452_data + (v412_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v458_data = ir2[7];
              ir2[7] = (v458_data + (v412_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v464_data = ir2[8];
              ir2[8] = (v464_data + (v412_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v466_data = r0[8];
              float v470_data = ir2[0];
              ir2[0] = (v470_data + (v466_data * (sycl::select_from_group(item.get_sub_group(), v35_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v476_data = ir2[1];
              ir2[1] = (v476_data + (v466_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v482_data = ir2[2];
              ir2[2] = (v482_data + (v466_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v488_data = ir2[3];
              ir2[3] = (v488_data + (v466_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v494_data = ir2[4];
              ir2[4] = (v494_data + (v466_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v500_data = ir2[5];
              ir2[5] = (v500_data + (v466_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v506_data = ir2[6];
              ir2[6] = (v506_data + (v466_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v512_data = ir2[7];
              ir2[7] = (v512_data + (v466_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v518_data = ir2[8];
              ir2[8] = (v518_data + (v466_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v520_data = r0[9];
              float v524_data = ir2[0];
              ir2[0] = (v524_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v35_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v530_data = ir2[1];
              ir2[1] = (v530_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v536_data = ir2[2];
              ir2[2] = (v536_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v542_data = ir2[3];
              ir2[3] = (v542_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v548_data = ir2[4];
              ir2[4] = (v548_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v554_data = ir2[5];
              ir2[5] = (v554_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v560_data = ir2[6];
              ir2[6] = (v560_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v566_data = ir2[7];
              ir2[7] = (v566_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v572_data = ir2[8];
              ir2[8] = (v572_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v574_data = r0[10];
              float v578_data = ir2[0];
              ir2[0] = (v578_data + (v574_data * (sycl::select_from_group(item.get_sub_group(), v35_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v584_data = ir2[1];
              ir2[1] = (v584_data + (v574_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v590_data = ir2[2];
              ir2[2] = (v590_data + (v574_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v596_data = ir2[3];
              ir2[3] = (v596_data + (v574_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v602_data = ir2[4];
              ir2[4] = (v602_data + (v574_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v608_data = ir2[5];
              ir2[5] = (v608_data + (v574_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v614_data = ir2[6];
              ir2[6] = (v614_data + (v574_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v620_data = ir2[7];
              ir2[7] = (v620_data + (v574_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v626_data = ir2[8];
              ir2[8] = (v626_data + (v574_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v628_data = r0[11];
              float v632_data = ir2[0];
              ir2[0] = (v632_data + (v628_data * (sycl::select_from_group(item.get_sub_group(), v35_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v638_data = ir2[1];
              ir2[1] = (v638_data + (v628_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v644_data = ir2[2];
              ir2[2] = (v644_data + (v628_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v650_data = ir2[3];
              ir2[3] = (v650_data + (v628_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v656_data = ir2[4];
              ir2[4] = (v656_data + (v628_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v662_data = ir2[5];
              ir2[5] = (v662_data + (v628_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v668_data = ir2[6];
              ir2[6] = (v668_data + (v628_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v674_data = ir2[7];
              ir2[7] = (v674_data + (v628_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v680_data = ir2[8];
              ir2[8] = (v680_data + (v628_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v682_data = r0[12];
              float v686_data = ir2[0];
              ir2[0] = (v686_data + (v682_data * (sycl::select_from_group(item.get_sub_group(), v35_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v692_data = ir2[1];
              ir2[1] = (v692_data + (v682_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v698_data = ir2[2];
              ir2[2] = (v698_data + (v682_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v704_data = ir2[3];
              ir2[3] = (v704_data + (v682_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v710_data = ir2[4];
              ir2[4] = (v710_data + (v682_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v716_data = ir2[5];
              ir2[5] = (v716_data + (v682_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v722_data = ir2[6];
              ir2[6] = (v722_data + (v682_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v728_data = ir2[7];
              ir2[7] = (v728_data + (v682_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v734_data = ir2[8];
              ir2[8] = (v734_data + (v682_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v736_data = r0[13];
              float v740_data = ir2[0];
              ir2[0] = (v740_data + (v736_data * (sycl::select_from_group(item.get_sub_group(), v35_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13)))));
              float v746_data = ir2[1];
              ir2[1] = (v746_data + (v736_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13)))));
              float v752_data = ir2[2];
              ir2[2] = (v752_data + (v736_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13)))));
              float v758_data = ir2[3];
              ir2[3] = (v758_data + (v736_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13)))));
              float v764_data = ir2[4];
              ir2[4] = (v764_data + (v736_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13)))));
              float v770_data = ir2[5];
              ir2[5] = (v770_data + (v736_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13)))));
              float v776_data = ir2[6];
              ir2[6] = (v776_data + (v736_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13)))));
              float v782_data = ir2[7];
              ir2[7] = (v782_data + (v736_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13)))));
              float v788_data = ir2[8];
              ir2[8] = (v788_data + (v736_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (13)))));
              float v790_data = r0[14];
              float v794_data = ir2[0];
              ir2[0] = (v794_data + (v790_data * (sycl::select_from_group(item.get_sub_group(), v35_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14)))));
              float v800_data = ir2[1];
              ir2[1] = (v800_data + (v790_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14)))));
              float v806_data = ir2[2];
              ir2[2] = (v806_data + (v790_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14)))));
              float v812_data = ir2[3];
              ir2[3] = (v812_data + (v790_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14)))));
              float v818_data = ir2[4];
              ir2[4] = (v818_data + (v790_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14)))));
              float v824_data = ir2[5];
              ir2[5] = (v824_data + (v790_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14)))));
              float v830_data = ir2[6];
              ir2[6] = (v830_data + (v790_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14)))));
              float v836_data = ir2[7];
              ir2[7] = (v836_data + (v790_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14)))));
              float v842_data = ir2[8];
              ir2[8] = (v842_data + (v790_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (14)))));
              float v844_data = r0[15];
              float v848_data = ir2[0];
              ir2[0] = (v848_data + (v844_data * (sycl::select_from_group(item.get_sub_group(), v35_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15)))));
              float v854_data = ir2[1];
              ir2[1] = (v854_data + (v844_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15)))));
              float v860_data = ir2[2];
              ir2[2] = (v860_data + (v844_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15)))));
              float v866_data = ir2[3];
              ir2[3] = (v866_data + (v844_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15)))));
              float v872_data = ir2[4];
              ir2[4] = (v872_data + (v844_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15)))));
              float v878_data = ir2[5];
              ir2[5] = (v878_data + (v844_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15)))));
              float v884_data = ir2[6];
              ir2[6] = (v884_data + (v844_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15)))));
              float v890_data = ir2[7];
              ir2[7] = (v890_data + (v844_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15)))));
              float v896_data = ir2[8];
              ir2[8] = (v896_data + (v844_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (15)))));
              float v898_data = r0[16];
              float v902_data = ir2[0];
              ir2[0] = (v902_data + (v898_data * (sycl::select_from_group(item.get_sub_group(), v35_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (16)))));
              float v908_data = ir2[1];
              ir2[1] = (v908_data + (v898_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (16)))));
              float v914_data = ir2[2];
              ir2[2] = (v914_data + (v898_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (16)))));
              float v920_data = ir2[3];
              ir2[3] = (v920_data + (v898_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (16)))));
              float v926_data = ir2[4];
              ir2[4] = (v926_data + (v898_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (16)))));
              float v932_data = ir2[5];
              ir2[5] = (v932_data + (v898_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (16)))));
              float v938_data = ir2[6];
              ir2[6] = (v938_data + (v898_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (16)))));
              float v944_data = ir2[7];
              ir2[7] = (v944_data + (v898_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (16)))));
              float v950_data = ir2[8];
              ir2[8] = (v950_data + (v898_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (16)))));
              float v952_data = r0[17];
              float v956_data = ir2[0];
              ir2[0] = (v956_data + (v952_data * (sycl::select_from_group(item.get_sub_group(), v35_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (17)))));
              float v962_data = ir2[1];
              ir2[1] = (v962_data + (v952_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (17)))));
              float v968_data = ir2[2];
              ir2[2] = (v968_data + (v952_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (17)))));
              float v974_data = ir2[3];
              ir2[3] = (v974_data + (v952_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (17)))));
              float v980_data = ir2[4];
              ir2[4] = (v980_data + (v952_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (17)))));
              float v986_data = ir2[5];
              ir2[5] = (v986_data + (v952_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (17)))));
              float v992_data = ir2[6];
              ir2[6] = (v992_data + (v952_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (17)))));
              float v998_data = ir2[7];
              ir2[7] = (v998_data + (v952_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (17)))));
              float v1004_data = ir2[8];
              ir2[8] = (v1004_data + (v952_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (17)))));
              float v1006_data = r0[18];
              float v1010_data = ir2[0];
              ir2[0] = (v1010_data + (v1006_data * (sycl::select_from_group(item.get_sub_group(), v35_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (18)))));
              float v1016_data = ir2[1];
              ir2[1] = (v1016_data + (v1006_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (18)))));
              float v1022_data = ir2[2];
              ir2[2] = (v1022_data + (v1006_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (18)))));
              float v1028_data = ir2[3];
              ir2[3] = (v1028_data + (v1006_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (18)))));
              float v1034_data = ir2[4];
              ir2[4] = (v1034_data + (v1006_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (18)))));
              float v1040_data = ir2[5];
              ir2[5] = (v1040_data + (v1006_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (18)))));
              float v1046_data = ir2[6];
              ir2[6] = (v1046_data + (v1006_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (18)))));
              float v1052_data = ir2[7];
              ir2[7] = (v1052_data + (v1006_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (18)))));
              float v1058_data = ir2[8];
              ir2[8] = (v1058_data + (v1006_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (18)))));
              float v1060_data = r0[19];
              float v1064_data = ir2[0];
              ir2[0] = (v1064_data + (v1060_data * (sycl::select_from_group(item.get_sub_group(), v35_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (19)))));
              float v1070_data = ir2[1];
              ir2[1] = (v1070_data + (v1060_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (19)))));
              float v1076_data = ir2[2];
              ir2[2] = (v1076_data + (v1060_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (19)))));
              float v1082_data = ir2[3];
              ir2[3] = (v1082_data + (v1060_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (19)))));
              float v1088_data = ir2[4];
              ir2[4] = (v1088_data + (v1060_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (19)))));
              float v1094_data = ir2[5];
              ir2[5] = (v1094_data + (v1060_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (19)))));
              float v1100_data = ir2[6];
              ir2[6] = (v1100_data + (v1060_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (19)))));
              float v1106_data = ir2[7];
              ir2[7] = (v1106_data + (v1060_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (19)))));
              float v1112_data = ir2[8];
              ir2[8] = (v1112_data + (v1060_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (19)))));
              float v1114_data = r0[20];
              float v1118_data = ir2[0];
              ir2[0] = (v1118_data + (v1114_data * (sycl::select_from_group(item.get_sub_group(), v35_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (20)))));
              float v1124_data = ir2[1];
              ir2[1] = (v1124_data + (v1114_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (20)))));
              float v1130_data = ir2[2];
              ir2[2] = (v1130_data + (v1114_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (20)))));
              float v1136_data = ir2[3];
              ir2[3] = (v1136_data + (v1114_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (20)))));
              float v1142_data = ir2[4];
              ir2[4] = (v1142_data + (v1114_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (20)))));
              float v1148_data = ir2[5];
              ir2[5] = (v1148_data + (v1114_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (20)))));
              float v1154_data = ir2[6];
              ir2[6] = (v1154_data + (v1114_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (20)))));
              float v1160_data = ir2[7];
              ir2[7] = (v1160_data + (v1114_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (20)))));
              float v1166_data = ir2[8];
              ir2[8] = (v1166_data + (v1114_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (20)))));
              float v1168_data = r0[21];
              float v1172_data = ir2[0];
              ir2[0] = (v1172_data + (v1168_data * (sycl::select_from_group(item.get_sub_group(), v35_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (21)))));
              float v1178_data = ir2[1];
              ir2[1] = (v1178_data + (v1168_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (21)))));
              float v1184_data = ir2[2];
              ir2[2] = (v1184_data + (v1168_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (21)))));
              float v1190_data = ir2[3];
              ir2[3] = (v1190_data + (v1168_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (21)))));
              float v1196_data = ir2[4];
              ir2[4] = (v1196_data + (v1168_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (21)))));
              float v1202_data = ir2[5];
              ir2[5] = (v1202_data + (v1168_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (21)))));
              float v1208_data = ir2[6];
              ir2[6] = (v1208_data + (v1168_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (21)))));
              float v1214_data = ir2[7];
              ir2[7] = (v1214_data + (v1168_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (21)))));
              float v1220_data = ir2[8];
              ir2[8] = (v1220_data + (v1168_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (21)))));
              float v1222_data = r0[22];
              float v1226_data = ir2[0];
              ir2[0] = (v1226_data + (v1222_data * (sycl::select_from_group(item.get_sub_group(), v35_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (22)))));
              float v1232_data = ir2[1];
              ir2[1] = (v1232_data + (v1222_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (22)))));
              float v1238_data = ir2[2];
              ir2[2] = (v1238_data + (v1222_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (22)))));
              float v1244_data = ir2[3];
              ir2[3] = (v1244_data + (v1222_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (22)))));
              float v1250_data = ir2[4];
              ir2[4] = (v1250_data + (v1222_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (22)))));
              float v1256_data = ir2[5];
              ir2[5] = (v1256_data + (v1222_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (22)))));
              float v1262_data = ir2[6];
              ir2[6] = (v1262_data + (v1222_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (22)))));
              float v1268_data = ir2[7];
              ir2[7] = (v1268_data + (v1222_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (22)))));
              float v1274_data = ir2[8];
              ir2[8] = (v1274_data + (v1222_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (22)))));
              float v1276_data = r0[23];
              float v1280_data = ir2[0];
              ir2[0] = (v1280_data + (v1276_data * (sycl::select_from_group(item.get_sub_group(), v35_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (23)))));
              float v1286_data = ir2[1];
              ir2[1] = (v1286_data + (v1276_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (23)))));
              float v1292_data = ir2[2];
              ir2[2] = (v1292_data + (v1276_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (23)))));
              float v1298_data = ir2[3];
              ir2[3] = (v1298_data + (v1276_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (23)))));
              float v1304_data = ir2[4];
              ir2[4] = (v1304_data + (v1276_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (23)))));
              float v1310_data = ir2[5];
              ir2[5] = (v1310_data + (v1276_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (23)))));
              float v1316_data = ir2[6];
              ir2[6] = (v1316_data + (v1276_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (23)))));
              float v1322_data = ir2[7];
              ir2[7] = (v1322_data + (v1276_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (23)))));
              float v1328_data = ir2[8];
              ir2[8] = (v1328_data + (v1276_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (23)))));
              if (v16_g) {
                #pragma unroll
                for (int32_t v1330_n1 = 0; v1330_n1 < 9; ++v1330_n1) {
                  float v1332_data = ir2[v1330_n1];
                  r2[v1330_n1] = v1332_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v16_g) {
                #pragma unroll
                for (int32_t v1333_i1 = 0; v1333_i1 < 9; ++v1333_i1) {
                  float v1335_data = r2[v1333_i1];
                  glb_m0[(v15_lead + (v1333_i1 * 24))] = v1335_data;
                }
              }
              item.barrier();
            }
          }
        }
      });
    }
  });
}

