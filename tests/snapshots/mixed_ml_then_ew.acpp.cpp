// === base name ===
kernel_f0381199d2fa9a9c

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_f0381199d2fa9a9c = {{8, 2, 1}, 8, 8, 1, 2, 64, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_f0381199d2fa9a9c(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_f0381199d2fa9a9c(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_f0381199d2fa9a9c(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_f0381199d2fa9a9c(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_f0381199d2fa9a9c(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_f0381199d2fa9a9c(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f0381199d2fa9a9c(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (16, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
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
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,2,1],"cooperative":false,"lead_width":1,"mults_per_block":2,"persistent":true,"sections":[{"barrier":false,"mults_per_block":2,"shared_elements":16}],"shared_bytes":64,"shared_elements":16,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[8 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          size_t v3_batchIdLane0 = item.get_local_id(1) % 2;
          int32_t v21_lead = item.get_local_id(2) % 8;
          for (size_t v4_batchIdGroup0 = (item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)); v4_batchIdGroup0 < numElements0; v4_batchIdGroup0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v5_row = v4_batchIdGroup0 + v3_batchIdLane0;
            const bool batchIdActive0 = v5_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v5_row]));
            size_t v7_batchId0 = batchIdActive0 ? v5_row : v4_batchIdGroup0;
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 64 + 0 + m0_extraOffset];
            const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 64 + 0 + m1_extraOffset];
            float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 64 + 0 + m2_extraOffset];
            float r0[8]{};
            // r0 = load{g>r}(glb_m0);
            #pragma unroll
            for (int32_t v22_i0 = 0; v22_i0 < 1; ++v22_i0) {
              int32_t v25_lead = v21_lead + (v22_i0 * 8);
              #pragma unroll
              for (int32_t v23_i1 = 0; v23_i1 < 8; ++v23_i1) {
                float v28_data = glb_m0[(v25_lead + (v23_i1 * 8))];
                r0[(v22_i0 + v23_i1)] = v28_data;
              }
            }
            float r1[8]{};
            // r1 = load{g>r}(glb_m1);
            #pragma unroll
            for (int32_t v31_i0 = 0; v31_i0 < 1; ++v31_i0) {
              int32_t v34_lead = v21_lead + (v31_i0 * 8);
              #pragma unroll
              for (int32_t v32_i1 = 0; v32_i1 < 8; ++v32_i1) {
                float v37_data = glb_m1[(v34_lead + (v32_i1 * 8))];
                r1[(v31_i0 + v32_i1)] = v37_data;
              }
            }
            // wait(r0 = load{g>r}(glb_m0););
            // wait(r1 = load{g>r}(glb_m1););
            float r2[8]{};
            // r2 = +(r0 * r1) + None
            // [(0, 8), (0, 8)] [(0, 8)]
            float v40_data = r0[0];
            float v41_data = r1[0];
            float v44_data = r2[0];
            r2[0] = (v44_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v47_data = r1[1];
            float v50_data = r2[1];
            r2[1] = (v50_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v53_data = r1[2];
            float v56_data = r2[2];
            r2[2] = (v56_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v59_data = r1[3];
            float v62_data = r2[3];
            r2[3] = (v62_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v65_data = r1[4];
            float v68_data = r2[4];
            r2[4] = (v68_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v71_data = r1[5];
            float v74_data = r2[5];
            r2[5] = (v74_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v77_data = r1[6];
            float v80_data = r2[6];
            r2[6] = (v80_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v83_data = r1[7];
            float v86_data = r2[7];
            r2[7] = (v86_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v88_data = r0[1];
            float v92_data = r2[0];
            r2[0] = (v92_data + (v88_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v98_data = r2[1];
            r2[1] = (v98_data + (v88_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v104_data = r2[2];
            r2[2] = (v104_data + (v88_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v110_data = r2[3];
            r2[3] = (v110_data + (v88_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v116_data = r2[4];
            r2[4] = (v116_data + (v88_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v122_data = r2[5];
            r2[5] = (v122_data + (v88_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v128_data = r2[6];
            r2[6] = (v128_data + (v88_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v134_data = r2[7];
            r2[7] = (v134_data + (v88_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v136_data = r0[2];
            float v140_data = r2[0];
            r2[0] = (v140_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v146_data = r2[1];
            r2[1] = (v146_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v152_data = r2[2];
            r2[2] = (v152_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v158_data = r2[3];
            r2[3] = (v158_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v164_data = r2[4];
            r2[4] = (v164_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v170_data = r2[5];
            r2[5] = (v170_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v176_data = r2[6];
            r2[6] = (v176_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v182_data = r2[7];
            r2[7] = (v182_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v184_data = r0[3];
            float v188_data = r2[0];
            r2[0] = (v188_data + (v184_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v194_data = r2[1];
            r2[1] = (v194_data + (v184_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v200_data = r2[2];
            r2[2] = (v200_data + (v184_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v206_data = r2[3];
            r2[3] = (v206_data + (v184_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v212_data = r2[4];
            r2[4] = (v212_data + (v184_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v218_data = r2[5];
            r2[5] = (v218_data + (v184_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v224_data = r2[6];
            r2[6] = (v224_data + (v184_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v230_data = r2[7];
            r2[7] = (v230_data + (v184_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v232_data = r0[4];
            float v236_data = r2[0];
            r2[0] = (v236_data + (v232_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v242_data = r2[1];
            r2[1] = (v242_data + (v232_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v248_data = r2[2];
            r2[2] = (v248_data + (v232_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v254_data = r2[3];
            r2[3] = (v254_data + (v232_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v260_data = r2[4];
            r2[4] = (v260_data + (v232_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v266_data = r2[5];
            r2[5] = (v266_data + (v232_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v272_data = r2[6];
            r2[6] = (v272_data + (v232_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v278_data = r2[7];
            r2[7] = (v278_data + (v232_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v280_data = r0[5];
            float v284_data = r2[0];
            r2[0] = (v284_data + (v280_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v290_data = r2[1];
            r2[1] = (v290_data + (v280_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v296_data = r2[2];
            r2[2] = (v296_data + (v280_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v302_data = r2[3];
            r2[3] = (v302_data + (v280_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v308_data = r2[4];
            r2[4] = (v308_data + (v280_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v314_data = r2[5];
            r2[5] = (v314_data + (v280_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v320_data = r2[6];
            r2[6] = (v320_data + (v280_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v326_data = r2[7];
            r2[7] = (v326_data + (v280_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v328_data = r0[6];
            float v332_data = r2[0];
            r2[0] = (v332_data + (v328_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v338_data = r2[1];
            r2[1] = (v338_data + (v328_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v344_data = r2[2];
            r2[2] = (v344_data + (v328_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v350_data = r2[3];
            r2[3] = (v350_data + (v328_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v356_data = r2[4];
            r2[4] = (v356_data + (v328_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v362_data = r2[5];
            r2[5] = (v362_data + (v328_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v368_data = r2[6];
            r2[6] = (v368_data + (v328_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v374_data = r2[7];
            r2[7] = (v374_data + (v328_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v376_data = r0[7];
            float v380_data = r2[0];
            r2[0] = (v380_data + (v376_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v386_data = r2[1];
            r2[1] = (v386_data + (v376_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v392_data = r2[2];
            r2[2] = (v392_data + (v376_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v398_data = r2[3];
            r2[3] = (v398_data + (v376_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v404_data = r2[4];
            r2[4] = (v404_data + (v376_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v410_data = r2[5];
            r2[5] = (v410_data + (v376_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v416_data = r2[6];
            r2[6] = (v416_data + (v376_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v422_data = r2[7];
            r2[7] = (v422_data + (v376_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            // glb_m2 = abs(r2)
            #pragma unroll
            for (int32_t v424_k0 = 0; v424_k0 < 1; ++v424_k0) {
              #pragma unroll
              for (int32_t v425_k1 = 0; v425_k1 < 8; ++v425_k1) {
                float v427_data = r2[(v424_k0 + v425_k1)];
                float v428_e = sycl::fabs(v427_data);
                if (batchIdActive0) {
                  glb_m2[((v21_lead + (v424_k0 * 8)) + (v425_k1 * 8))] = v428_e;
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

