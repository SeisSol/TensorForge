// === base name ===
kernel_a510064bf49a6633

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_a510064bf49a6633 = {{16, 16, 1}, 16, 16, 1, 16, 5120, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_a510064bf49a6633(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_a510064bf49a6633(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_a510064bf49a6633(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 1280 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_a510064bf49a6633(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_a510064bf49a6633(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_a510064bf49a6633(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_a510064bf49a6633(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 5120 B shared, occupancy grid
        // operands:
        //   m0 8×8(8×8) {0..8}×{0..8} strided
        //   m1 8×8(8×8) {0..8}×{0..8} strided
        //   m2 8×8(8×8) {0..8}×{0..8} strided
        //   m3 8×8(8×8) {0..8}×{0..8} strided
        //   m4 8×8(8×8) {0..8}×{0..8} strided
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   t0[i,j] += m2[i,k] × m3[k,j]
        //   C = abs(TMP)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1280}],"shared_bytes":5120,"shared_elements":1280,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"A1","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"A2","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m4","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[80 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[64];
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v4_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v4_batchId0 < numElements0; v4_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v5_ahead1 = v4_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v4_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v4_batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v4_batchId0 * 64 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v4_batchId0 * 64 + 0 + m3_extraOffset];
              float *const __restrict__ glb_m4 = &m4[v4_batchId0 * 64 + 0 + m4_extraOffset];
              float r0[8]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v20_lead = item.get_local_id(2) % 16;
              bool v21_g = v20_lead < 8;
              if (v21_g) {
                #pragma unroll
                for (int32_t v22_i1 = 0; v22_i1 < 8; ++v22_i1) {
                  float v27_data = glb_m0[(v20_lead + (v22_i1 * 8))];
                  r0[v22_i1] = v27_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m1);
              if (v21_g) {
                #pragma unroll
                for (int32_t v30_i1 = 0; v30_i1 < 8; ++v30_i1) {
                  float v35_data = glb_m1[(v20_lead + (v30_i1 * 8))];
                  r1[v30_i1] = v35_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r3[8]{};
              // r3 = load{g>r}(glb_m2);
              if (v21_g) {
                #pragma unroll
                for (int32_t v38_i1 = 0; v38_i1 < 8; ++v38_i1) {
                  float v43_data = glb_m2[(v20_lead + (v38_i1 * 8))];
                  r3[v38_i1] = v43_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m1););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              float v46_data = r0[0];
              float v47_data = r1[0];
              float v50_data = r2[0];
              r2[0] = (v50_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v53_data = r1[1];
              float v56_data = r2[1];
              r2[1] = (v56_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v59_data = r1[2];
              float v62_data = r2[2];
              r2[2] = (v62_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v65_data = r1[3];
              float v68_data = r2[3];
              r2[3] = (v68_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v71_data = r1[4];
              float v74_data = r2[4];
              r2[4] = (v74_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v77_data = r1[5];
              float v80_data = r2[5];
              r2[5] = (v80_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v83_data = r1[6];
              float v86_data = r2[6];
              r2[6] = (v86_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v89_data = r1[7];
              float v92_data = r2[7];
              r2[7] = (v92_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v94_data = r0[1];
              float v98_data = r2[0];
              r2[0] = (v98_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v104_data = r2[1];
              r2[1] = (v104_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v110_data = r2[2];
              r2[2] = (v110_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v116_data = r2[3];
              r2[3] = (v116_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v122_data = r2[4];
              r2[4] = (v122_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v128_data = r2[5];
              r2[5] = (v128_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v134_data = r2[6];
              r2[6] = (v134_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v140_data = r2[7];
              r2[7] = (v140_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v142_data = r0[2];
              float v146_data = r2[0];
              r2[0] = (v146_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v152_data = r2[1];
              r2[1] = (v152_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v158_data = r2[2];
              r2[2] = (v158_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v164_data = r2[3];
              r2[3] = (v164_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v170_data = r2[4];
              r2[4] = (v170_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v176_data = r2[5];
              r2[5] = (v176_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v182_data = r2[6];
              r2[6] = (v182_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v188_data = r2[7];
              r2[7] = (v188_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v190_data = r0[3];
              float v194_data = r2[0];
              r2[0] = (v194_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v200_data = r2[1];
              r2[1] = (v200_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v206_data = r2[2];
              r2[2] = (v206_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v212_data = r2[3];
              r2[3] = (v212_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v218_data = r2[4];
              r2[4] = (v218_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v224_data = r2[5];
              r2[5] = (v224_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v230_data = r2[6];
              r2[6] = (v230_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v236_data = r2[7];
              r2[7] = (v236_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v238_data = r0[4];
              float v242_data = r2[0];
              r2[0] = (v242_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v248_data = r2[1];
              r2[1] = (v248_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v254_data = r2[2];
              r2[2] = (v254_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v260_data = r2[3];
              r2[3] = (v260_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v266_data = r2[4];
              r2[4] = (v266_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v272_data = r2[5];
              r2[5] = (v272_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v278_data = r2[6];
              r2[6] = (v278_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v284_data = r2[7];
              r2[7] = (v284_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v286_data = r0[5];
              float v290_data = r2[0];
              r2[0] = (v290_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v296_data = r2[1];
              r2[1] = (v296_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v302_data = r2[2];
              r2[2] = (v302_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v308_data = r2[3];
              r2[3] = (v308_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v314_data = r2[4];
              r2[4] = (v314_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v320_data = r2[5];
              r2[5] = (v320_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v326_data = r2[6];
              r2[6] = (v326_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v332_data = r2[7];
              r2[7] = (v332_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v334_data = r0[6];
              float v338_data = r2[0];
              r2[0] = (v338_data + (v334_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v344_data = r2[1];
              r2[1] = (v344_data + (v334_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v350_data = r2[2];
              r2[2] = (v350_data + (v334_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v356_data = r2[3];
              r2[3] = (v356_data + (v334_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v362_data = r2[4];
              r2[4] = (v362_data + (v334_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v368_data = r2[5];
              r2[5] = (v368_data + (v334_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v374_data = r2[6];
              r2[6] = (v374_data + (v334_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v380_data = r2[7];
              r2[7] = (v380_data + (v334_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v382_data = r0[7];
              float v386_data = r2[0];
              r2[0] = (v386_data + (v382_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v392_data = r2[1];
              r2[1] = (v392_data + (v382_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v398_data = r2[2];
              r2[2] = (v398_data + (v382_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v404_data = r2[3];
              r2[3] = (v404_data + (v382_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v410_data = r2[4];
              r2[4] = (v410_data + (v382_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v416_data = r2[5];
              r2[5] = (v416_data + (v382_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v422_data = r2[6];
              r2[6] = (v422_data + (v382_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v428_data = r2[7];
              r2[7] = (v428_data + (v382_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float r4[8]{};
              // r4 = load{g>r}(glb_m3);
              if (v21_g) {
                #pragma unroll
                for (int32_t v431_i1 = 0; v431_i1 < 8; ++v431_i1) {
                  float v436_data = glb_m3[(v20_lead + (v431_i1 * 8))];
                  r4[v431_i1] = v436_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m2););
              // wait(r4 = load{g>r}(glb_m3););
              float r5[8]{};
              // r5 = +(r3 * r4) + name: r2, type: SymbolType.Register, lead: [0]
              // [(0, 8), (0, 8)] [(0, 8)]
              float ir5[8]{};
              float v440_data = r3[0];
              float v441_data = r4[0];
              float v444_data = ir5[0];
              ir5[0] = (v444_data + (v440_data * (sycl::select_from_group(item.get_sub_group(), v441_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v447_data = r4[1];
              float v450_data = ir5[1];
              ir5[1] = (v450_data + (v440_data * (sycl::select_from_group(item.get_sub_group(), v447_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v453_data = r4[2];
              float v456_data = ir5[2];
              ir5[2] = (v456_data + (v440_data * (sycl::select_from_group(item.get_sub_group(), v453_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v459_data = r4[3];
              float v462_data = ir5[3];
              ir5[3] = (v462_data + (v440_data * (sycl::select_from_group(item.get_sub_group(), v459_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v465_data = r4[4];
              float v468_data = ir5[4];
              ir5[4] = (v468_data + (v440_data * (sycl::select_from_group(item.get_sub_group(), v465_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v471_data = r4[5];
              float v474_data = ir5[5];
              ir5[5] = (v474_data + (v440_data * (sycl::select_from_group(item.get_sub_group(), v471_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v477_data = r4[6];
              float v480_data = ir5[6];
              ir5[6] = (v480_data + (v440_data * (sycl::select_from_group(item.get_sub_group(), v477_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v483_data = r4[7];
              float v486_data = ir5[7];
              ir5[7] = (v486_data + (v440_data * (sycl::select_from_group(item.get_sub_group(), v483_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v488_data = r3[1];
              float v492_data = ir5[0];
              ir5[0] = (v492_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v441_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v498_data = ir5[1];
              ir5[1] = (v498_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v447_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v504_data = ir5[2];
              ir5[2] = (v504_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v453_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v510_data = ir5[3];
              ir5[3] = (v510_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v459_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v516_data = ir5[4];
              ir5[4] = (v516_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v465_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v522_data = ir5[5];
              ir5[5] = (v522_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v471_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v528_data = ir5[6];
              ir5[6] = (v528_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v477_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v534_data = ir5[7];
              ir5[7] = (v534_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v483_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v536_data = r3[2];
              float v540_data = ir5[0];
              ir5[0] = (v540_data + (v536_data * (sycl::select_from_group(item.get_sub_group(), v441_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v546_data = ir5[1];
              ir5[1] = (v546_data + (v536_data * (sycl::select_from_group(item.get_sub_group(), v447_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v552_data = ir5[2];
              ir5[2] = (v552_data + (v536_data * (sycl::select_from_group(item.get_sub_group(), v453_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v558_data = ir5[3];
              ir5[3] = (v558_data + (v536_data * (sycl::select_from_group(item.get_sub_group(), v459_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v564_data = ir5[4];
              ir5[4] = (v564_data + (v536_data * (sycl::select_from_group(item.get_sub_group(), v465_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v570_data = ir5[5];
              ir5[5] = (v570_data + (v536_data * (sycl::select_from_group(item.get_sub_group(), v471_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v576_data = ir5[6];
              ir5[6] = (v576_data + (v536_data * (sycl::select_from_group(item.get_sub_group(), v477_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v582_data = ir5[7];
              ir5[7] = (v582_data + (v536_data * (sycl::select_from_group(item.get_sub_group(), v483_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v584_data = r3[3];
              float v588_data = ir5[0];
              ir5[0] = (v588_data + (v584_data * (sycl::select_from_group(item.get_sub_group(), v441_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v594_data = ir5[1];
              ir5[1] = (v594_data + (v584_data * (sycl::select_from_group(item.get_sub_group(), v447_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v600_data = ir5[2];
              ir5[2] = (v600_data + (v584_data * (sycl::select_from_group(item.get_sub_group(), v453_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v606_data = ir5[3];
              ir5[3] = (v606_data + (v584_data * (sycl::select_from_group(item.get_sub_group(), v459_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v612_data = ir5[4];
              ir5[4] = (v612_data + (v584_data * (sycl::select_from_group(item.get_sub_group(), v465_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v618_data = ir5[5];
              ir5[5] = (v618_data + (v584_data * (sycl::select_from_group(item.get_sub_group(), v471_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v624_data = ir5[6];
              ir5[6] = (v624_data + (v584_data * (sycl::select_from_group(item.get_sub_group(), v477_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v630_data = ir5[7];
              ir5[7] = (v630_data + (v584_data * (sycl::select_from_group(item.get_sub_group(), v483_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v632_data = r3[4];
              float v636_data = ir5[0];
              ir5[0] = (v636_data + (v632_data * (sycl::select_from_group(item.get_sub_group(), v441_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v642_data = ir5[1];
              ir5[1] = (v642_data + (v632_data * (sycl::select_from_group(item.get_sub_group(), v447_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v648_data = ir5[2];
              ir5[2] = (v648_data + (v632_data * (sycl::select_from_group(item.get_sub_group(), v453_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v654_data = ir5[3];
              ir5[3] = (v654_data + (v632_data * (sycl::select_from_group(item.get_sub_group(), v459_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v660_data = ir5[4];
              ir5[4] = (v660_data + (v632_data * (sycl::select_from_group(item.get_sub_group(), v465_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v666_data = ir5[5];
              ir5[5] = (v666_data + (v632_data * (sycl::select_from_group(item.get_sub_group(), v471_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v672_data = ir5[6];
              ir5[6] = (v672_data + (v632_data * (sycl::select_from_group(item.get_sub_group(), v477_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v678_data = ir5[7];
              ir5[7] = (v678_data + (v632_data * (sycl::select_from_group(item.get_sub_group(), v483_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v680_data = r3[5];
              float v684_data = ir5[0];
              ir5[0] = (v684_data + (v680_data * (sycl::select_from_group(item.get_sub_group(), v441_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v690_data = ir5[1];
              ir5[1] = (v690_data + (v680_data * (sycl::select_from_group(item.get_sub_group(), v447_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v696_data = ir5[2];
              ir5[2] = (v696_data + (v680_data * (sycl::select_from_group(item.get_sub_group(), v453_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v702_data = ir5[3];
              ir5[3] = (v702_data + (v680_data * (sycl::select_from_group(item.get_sub_group(), v459_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v708_data = ir5[4];
              ir5[4] = (v708_data + (v680_data * (sycl::select_from_group(item.get_sub_group(), v465_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v714_data = ir5[5];
              ir5[5] = (v714_data + (v680_data * (sycl::select_from_group(item.get_sub_group(), v471_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v720_data = ir5[6];
              ir5[6] = (v720_data + (v680_data * (sycl::select_from_group(item.get_sub_group(), v477_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v726_data = ir5[7];
              ir5[7] = (v726_data + (v680_data * (sycl::select_from_group(item.get_sub_group(), v483_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v728_data = r3[6];
              float v732_data = ir5[0];
              ir5[0] = (v732_data + (v728_data * (sycl::select_from_group(item.get_sub_group(), v441_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v738_data = ir5[1];
              ir5[1] = (v738_data + (v728_data * (sycl::select_from_group(item.get_sub_group(), v447_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v744_data = ir5[2];
              ir5[2] = (v744_data + (v728_data * (sycl::select_from_group(item.get_sub_group(), v453_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v750_data = ir5[3];
              ir5[3] = (v750_data + (v728_data * (sycl::select_from_group(item.get_sub_group(), v459_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v756_data = ir5[4];
              ir5[4] = (v756_data + (v728_data * (sycl::select_from_group(item.get_sub_group(), v465_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v762_data = ir5[5];
              ir5[5] = (v762_data + (v728_data * (sycl::select_from_group(item.get_sub_group(), v471_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v768_data = ir5[6];
              ir5[6] = (v768_data + (v728_data * (sycl::select_from_group(item.get_sub_group(), v477_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v774_data = ir5[7];
              ir5[7] = (v774_data + (v728_data * (sycl::select_from_group(item.get_sub_group(), v483_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v776_data = r3[7];
              float v780_data = ir5[0];
              ir5[0] = (v780_data + (v776_data * (sycl::select_from_group(item.get_sub_group(), v441_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v786_data = ir5[1];
              ir5[1] = (v786_data + (v776_data * (sycl::select_from_group(item.get_sub_group(), v447_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v792_data = ir5[2];
              ir5[2] = (v792_data + (v776_data * (sycl::select_from_group(item.get_sub_group(), v453_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v798_data = ir5[3];
              ir5[3] = (v798_data + (v776_data * (sycl::select_from_group(item.get_sub_group(), v459_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v804_data = ir5[4];
              ir5[4] = (v804_data + (v776_data * (sycl::select_from_group(item.get_sub_group(), v465_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v810_data = ir5[5];
              ir5[5] = (v810_data + (v776_data * (sycl::select_from_group(item.get_sub_group(), v471_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v816_data = ir5[6];
              ir5[6] = (v816_data + (v776_data * (sycl::select_from_group(item.get_sub_group(), v477_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v822_data = ir5[7];
              ir5[7] = (v822_data + (v776_data * (sycl::select_from_group(item.get_sub_group(), v483_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              if (v21_g) {
                #pragma unroll
                for (int32_t v824_n1 = 0; v824_n1 < 8; ++v824_n1) {
                  float v826_data = ir5[v824_n1];
                  float v827_data = r2[v824_n1];
                  r5[v824_n1] = (v827_data + v826_data);
                }
              }
              // s0 = store{r>s}(localShrMem0, r5);
              if (v21_g) {
                #pragma unroll
                for (int32_t v829_i1 = 0; v829_i1 < 8; ++v829_i1) {
                  float v831_data = r5[v829_i1];
                  int32_t v835_a = v20_lead + (v829_i1 * 8);
                  s0[(v835_a ^ ((v835_a >> 5) & 31))] = v831_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
              // glb_m4 = abs(s0)
              if (v21_g) {
                #pragma unroll
                for (int32_t v839_k1 = 0; v839_k1 < 8; ++v839_k1) {
                  int32_t v843_a = v20_lead + (v839_k1 * 8);
                  float v847_data = s0[(v843_a ^ ((v843_a >> 5) & 31))];
                  glb_m4[v843_a] = (sycl::fabs(v847_data));
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

