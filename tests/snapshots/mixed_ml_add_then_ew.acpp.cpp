// === base name ===
kernel_cfc4246fe69554b5

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_cfc4246fe69554b5 = {{8, 2, 1}, 8, 8, 1, 2, 576, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_cfc4246fe69554b5(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_cfc4246fe69554b5(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_cfc4246fe69554b5(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_cfc4246fe69554b5(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_cfc4246fe69554b5(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_cfc4246fe69554b5(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_cfc4246fe69554b5(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
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
        //   m3 8×8(8×8) {0..8}×{0..8} strided
        //   m4 8×8(8×8) {0..8}×{0..8} strided
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   t0[i,j] += m2[i,k] × m3[k,j]
        //   C = abs(TMP)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,2,1],"cooperative":false,"lead_width":1,"mults_per_block":2,"persistent":true,"sections":[{"barrier":false,"mults_per_block":2,"shared_elements":144}],"shared_bytes":576,"shared_elements":144,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A1","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"A2","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m4","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[72 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[64];
          float * __restrict__ s0 = &localShrMem0[0];
          size_t v4_batchIdLane0 = item.get_local_id(1) % 2;
          int32_t v24_lead = item.get_local_id(2) % 8;
          for (size_t v5_batchIdGroup0 = (item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)); v5_batchIdGroup0 < numElements0; v5_batchIdGroup0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v6_row = v5_batchIdGroup0 + v4_batchIdLane0;
            const bool batchIdActive0 = v6_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v6_row]));
            size_t v8_batchId0 = batchIdActive0 ? v6_row : v5_batchIdGroup0;
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 64 + 0 + m0_extraOffset];
            const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 64 + 0 + m1_extraOffset];
            const float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 64 + 0 + m2_extraOffset];
            const float *const __restrict__ glb_m3 = &m3[v8_batchId0 * 64 + 0 + m3_extraOffset];
            float *const __restrict__ glb_m4 = &m4[v8_batchId0 * 64 + 0 + m4_extraOffset];
            float r0[8]{};
            // r0 = load{g>r}(glb_m0);
            #pragma unroll
            for (int32_t v25_i0 = 0; v25_i0 < 1; ++v25_i0) {
              int32_t v28_lead = v24_lead + (v25_i0 * 8);
              #pragma unroll
              for (int32_t v26_i1 = 0; v26_i1 < 8; ++v26_i1) {
                float v31_data = glb_m0[(v28_lead + (v26_i1 * 8))];
                r0[(v25_i0 + v26_i1)] = v31_data;
              }
            }
            float r1[8]{};
            // r1 = load{g>r}(glb_m1);
            #pragma unroll
            for (int32_t v34_i0 = 0; v34_i0 < 1; ++v34_i0) {
              int32_t v37_lead = v24_lead + (v34_i0 * 8);
              #pragma unroll
              for (int32_t v35_i1 = 0; v35_i1 < 8; ++v35_i1) {
                float v40_data = glb_m1[(v37_lead + (v35_i1 * 8))];
                r1[(v34_i0 + v35_i1)] = v40_data;
              }
            }
            // wait(r0 = load{g>r}(glb_m0););
            float r3[8]{};
            // r3 = load{g>r}(glb_m2);
            #pragma unroll
            for (int32_t v43_i0 = 0; v43_i0 < 1; ++v43_i0) {
              int32_t v46_lead = v24_lead + (v43_i0 * 8);
              #pragma unroll
              for (int32_t v44_i1 = 0; v44_i1 < 8; ++v44_i1) {
                float v49_data = glb_m2[(v46_lead + (v44_i1 * 8))];
                r3[(v43_i0 + v44_i1)] = v49_data;
              }
            }
            // wait(r1 = load{g>r}(glb_m1););
            float r2[8]{};
            // r2 = +(r0 * r1) + None
            // [(0, 8), (0, 8)] [(0, 8)]
            float v52_data = r0[0];
            float v53_data = r1[0];
            float v56_data = r2[0];
            r2[0] = (v56_data + (v52_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v59_data = r1[1];
            float v62_data = r2[1];
            r2[1] = (v62_data + (v52_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v65_data = r1[2];
            float v68_data = r2[2];
            r2[2] = (v68_data + (v52_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v71_data = r1[3];
            float v74_data = r2[3];
            r2[3] = (v74_data + (v52_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v77_data = r1[4];
            float v80_data = r2[4];
            r2[4] = (v80_data + (v52_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v83_data = r1[5];
            float v86_data = r2[5];
            r2[5] = (v86_data + (v52_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v89_data = r1[6];
            float v92_data = r2[6];
            r2[6] = (v92_data + (v52_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v95_data = r1[7];
            float v98_data = r2[7];
            r2[7] = (v98_data + (v52_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v100_data = r0[1];
            float v104_data = r2[0];
            r2[0] = (v104_data + (v100_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v110_data = r2[1];
            r2[1] = (v110_data + (v100_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v116_data = r2[2];
            r2[2] = (v116_data + (v100_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v122_data = r2[3];
            r2[3] = (v122_data + (v100_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v128_data = r2[4];
            r2[4] = (v128_data + (v100_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v134_data = r2[5];
            r2[5] = (v134_data + (v100_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v140_data = r2[6];
            r2[6] = (v140_data + (v100_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v146_data = r2[7];
            r2[7] = (v146_data + (v100_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v148_data = r0[2];
            float v152_data = r2[0];
            r2[0] = (v152_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v158_data = r2[1];
            r2[1] = (v158_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v164_data = r2[2];
            r2[2] = (v164_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v170_data = r2[3];
            r2[3] = (v170_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v176_data = r2[4];
            r2[4] = (v176_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v182_data = r2[5];
            r2[5] = (v182_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v188_data = r2[6];
            r2[6] = (v188_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v194_data = r2[7];
            r2[7] = (v194_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v196_data = r0[3];
            float v200_data = r2[0];
            r2[0] = (v200_data + (v196_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v206_data = r2[1];
            r2[1] = (v206_data + (v196_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v212_data = r2[2];
            r2[2] = (v212_data + (v196_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v218_data = r2[3];
            r2[3] = (v218_data + (v196_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v224_data = r2[4];
            r2[4] = (v224_data + (v196_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v230_data = r2[5];
            r2[5] = (v230_data + (v196_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v236_data = r2[6];
            r2[6] = (v236_data + (v196_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v242_data = r2[7];
            r2[7] = (v242_data + (v196_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v244_data = r0[4];
            float v248_data = r2[0];
            r2[0] = (v248_data + (v244_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v254_data = r2[1];
            r2[1] = (v254_data + (v244_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v260_data = r2[2];
            r2[2] = (v260_data + (v244_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v266_data = r2[3];
            r2[3] = (v266_data + (v244_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v272_data = r2[4];
            r2[4] = (v272_data + (v244_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v278_data = r2[5];
            r2[5] = (v278_data + (v244_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v284_data = r2[6];
            r2[6] = (v284_data + (v244_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v290_data = r2[7];
            r2[7] = (v290_data + (v244_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v292_data = r0[5];
            float v296_data = r2[0];
            r2[0] = (v296_data + (v292_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v302_data = r2[1];
            r2[1] = (v302_data + (v292_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v308_data = r2[2];
            r2[2] = (v308_data + (v292_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v314_data = r2[3];
            r2[3] = (v314_data + (v292_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v320_data = r2[4];
            r2[4] = (v320_data + (v292_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v326_data = r2[5];
            r2[5] = (v326_data + (v292_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v332_data = r2[6];
            r2[6] = (v332_data + (v292_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v338_data = r2[7];
            r2[7] = (v338_data + (v292_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v340_data = r0[6];
            float v344_data = r2[0];
            r2[0] = (v344_data + (v340_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v350_data = r2[1];
            r2[1] = (v350_data + (v340_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v356_data = r2[2];
            r2[2] = (v356_data + (v340_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v362_data = r2[3];
            r2[3] = (v362_data + (v340_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v368_data = r2[4];
            r2[4] = (v368_data + (v340_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v374_data = r2[5];
            r2[5] = (v374_data + (v340_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v380_data = r2[6];
            r2[6] = (v380_data + (v340_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v386_data = r2[7];
            r2[7] = (v386_data + (v340_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v388_data = r0[7];
            float v392_data = r2[0];
            r2[0] = (v392_data + (v388_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v398_data = r2[1];
            r2[1] = (v398_data + (v388_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v404_data = r2[2];
            r2[2] = (v404_data + (v388_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v410_data = r2[3];
            r2[3] = (v410_data + (v388_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v416_data = r2[4];
            r2[4] = (v416_data + (v388_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v422_data = r2[5];
            r2[5] = (v422_data + (v388_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v428_data = r2[6];
            r2[6] = (v428_data + (v388_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v434_data = r2[7];
            r2[7] = (v434_data + (v388_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float r4[8]{};
            // r4 = load{g>r}(glb_m3);
            #pragma unroll
            for (int32_t v437_i0 = 0; v437_i0 < 1; ++v437_i0) {
              int32_t v440_lead = v24_lead + (v437_i0 * 8);
              #pragma unroll
              for (int32_t v438_i1 = 0; v438_i1 < 8; ++v438_i1) {
                float v443_data = glb_m3[(v440_lead + (v438_i1 * 8))];
                r4[(v437_i0 + v438_i1)] = v443_data;
              }
            }
            // wait(r3 = load{g>r}(glb_m2););
            // wait(r4 = load{g>r}(glb_m3););
            float r5[8]{};
            // ir5 = +(r3 * r4)
            // [(0, 8), (0, 8)] [(0, 8)]
            float ir5[8]{};
            float v447_data = r3[0];
            float v448_data = r4[0];
            float v451_data = ir5[0];
            ir5[0] = (v451_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v448_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v454_data = r4[1];
            float v457_data = ir5[1];
            ir5[1] = (v457_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v454_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v460_data = r4[2];
            float v463_data = ir5[2];
            ir5[2] = (v463_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v460_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v466_data = r4[3];
            float v469_data = ir5[3];
            ir5[3] = (v469_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v466_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v472_data = r4[4];
            float v475_data = ir5[4];
            ir5[4] = (v475_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v472_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v478_data = r4[5];
            float v481_data = ir5[5];
            ir5[5] = (v481_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v478_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v484_data = r4[6];
            float v487_data = ir5[6];
            ir5[6] = (v487_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v484_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v490_data = r4[7];
            float v493_data = ir5[7];
            ir5[7] = (v493_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v490_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v495_data = r3[1];
            float v499_data = ir5[0];
            ir5[0] = (v499_data + (v495_data * (sycl::select_from_group(item.get_sub_group(), v448_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v505_data = ir5[1];
            ir5[1] = (v505_data + (v495_data * (sycl::select_from_group(item.get_sub_group(), v454_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v511_data = ir5[2];
            ir5[2] = (v511_data + (v495_data * (sycl::select_from_group(item.get_sub_group(), v460_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v517_data = ir5[3];
            ir5[3] = (v517_data + (v495_data * (sycl::select_from_group(item.get_sub_group(), v466_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v523_data = ir5[4];
            ir5[4] = (v523_data + (v495_data * (sycl::select_from_group(item.get_sub_group(), v472_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v529_data = ir5[5];
            ir5[5] = (v529_data + (v495_data * (sycl::select_from_group(item.get_sub_group(), v478_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v535_data = ir5[6];
            ir5[6] = (v535_data + (v495_data * (sycl::select_from_group(item.get_sub_group(), v484_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v541_data = ir5[7];
            ir5[7] = (v541_data + (v495_data * (sycl::select_from_group(item.get_sub_group(), v490_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v543_data = r3[2];
            float v547_data = ir5[0];
            ir5[0] = (v547_data + (v543_data * (sycl::select_from_group(item.get_sub_group(), v448_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v553_data = ir5[1];
            ir5[1] = (v553_data + (v543_data * (sycl::select_from_group(item.get_sub_group(), v454_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v559_data = ir5[2];
            ir5[2] = (v559_data + (v543_data * (sycl::select_from_group(item.get_sub_group(), v460_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v565_data = ir5[3];
            ir5[3] = (v565_data + (v543_data * (sycl::select_from_group(item.get_sub_group(), v466_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v571_data = ir5[4];
            ir5[4] = (v571_data + (v543_data * (sycl::select_from_group(item.get_sub_group(), v472_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v577_data = ir5[5];
            ir5[5] = (v577_data + (v543_data * (sycl::select_from_group(item.get_sub_group(), v478_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v583_data = ir5[6];
            ir5[6] = (v583_data + (v543_data * (sycl::select_from_group(item.get_sub_group(), v484_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v589_data = ir5[7];
            ir5[7] = (v589_data + (v543_data * (sycl::select_from_group(item.get_sub_group(), v490_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v591_data = r3[3];
            float v595_data = ir5[0];
            ir5[0] = (v595_data + (v591_data * (sycl::select_from_group(item.get_sub_group(), v448_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v601_data = ir5[1];
            ir5[1] = (v601_data + (v591_data * (sycl::select_from_group(item.get_sub_group(), v454_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v607_data = ir5[2];
            ir5[2] = (v607_data + (v591_data * (sycl::select_from_group(item.get_sub_group(), v460_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v613_data = ir5[3];
            ir5[3] = (v613_data + (v591_data * (sycl::select_from_group(item.get_sub_group(), v466_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v619_data = ir5[4];
            ir5[4] = (v619_data + (v591_data * (sycl::select_from_group(item.get_sub_group(), v472_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v625_data = ir5[5];
            ir5[5] = (v625_data + (v591_data * (sycl::select_from_group(item.get_sub_group(), v478_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v631_data = ir5[6];
            ir5[6] = (v631_data + (v591_data * (sycl::select_from_group(item.get_sub_group(), v484_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v637_data = ir5[7];
            ir5[7] = (v637_data + (v591_data * (sycl::select_from_group(item.get_sub_group(), v490_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v639_data = r3[4];
            float v643_data = ir5[0];
            ir5[0] = (v643_data + (v639_data * (sycl::select_from_group(item.get_sub_group(), v448_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v649_data = ir5[1];
            ir5[1] = (v649_data + (v639_data * (sycl::select_from_group(item.get_sub_group(), v454_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v655_data = ir5[2];
            ir5[2] = (v655_data + (v639_data * (sycl::select_from_group(item.get_sub_group(), v460_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v661_data = ir5[3];
            ir5[3] = (v661_data + (v639_data * (sycl::select_from_group(item.get_sub_group(), v466_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v667_data = ir5[4];
            ir5[4] = (v667_data + (v639_data * (sycl::select_from_group(item.get_sub_group(), v472_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v673_data = ir5[5];
            ir5[5] = (v673_data + (v639_data * (sycl::select_from_group(item.get_sub_group(), v478_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v679_data = ir5[6];
            ir5[6] = (v679_data + (v639_data * (sycl::select_from_group(item.get_sub_group(), v484_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v685_data = ir5[7];
            ir5[7] = (v685_data + (v639_data * (sycl::select_from_group(item.get_sub_group(), v490_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v687_data = r3[5];
            float v691_data = ir5[0];
            ir5[0] = (v691_data + (v687_data * (sycl::select_from_group(item.get_sub_group(), v448_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v697_data = ir5[1];
            ir5[1] = (v697_data + (v687_data * (sycl::select_from_group(item.get_sub_group(), v454_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v703_data = ir5[2];
            ir5[2] = (v703_data + (v687_data * (sycl::select_from_group(item.get_sub_group(), v460_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v709_data = ir5[3];
            ir5[3] = (v709_data + (v687_data * (sycl::select_from_group(item.get_sub_group(), v466_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v715_data = ir5[4];
            ir5[4] = (v715_data + (v687_data * (sycl::select_from_group(item.get_sub_group(), v472_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v721_data = ir5[5];
            ir5[5] = (v721_data + (v687_data * (sycl::select_from_group(item.get_sub_group(), v478_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v727_data = ir5[6];
            ir5[6] = (v727_data + (v687_data * (sycl::select_from_group(item.get_sub_group(), v484_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v733_data = ir5[7];
            ir5[7] = (v733_data + (v687_data * (sycl::select_from_group(item.get_sub_group(), v490_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v735_data = r3[6];
            float v739_data = ir5[0];
            ir5[0] = (v739_data + (v735_data * (sycl::select_from_group(item.get_sub_group(), v448_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v745_data = ir5[1];
            ir5[1] = (v745_data + (v735_data * (sycl::select_from_group(item.get_sub_group(), v454_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v751_data = ir5[2];
            ir5[2] = (v751_data + (v735_data * (sycl::select_from_group(item.get_sub_group(), v460_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v757_data = ir5[3];
            ir5[3] = (v757_data + (v735_data * (sycl::select_from_group(item.get_sub_group(), v466_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v763_data = ir5[4];
            ir5[4] = (v763_data + (v735_data * (sycl::select_from_group(item.get_sub_group(), v472_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v769_data = ir5[5];
            ir5[5] = (v769_data + (v735_data * (sycl::select_from_group(item.get_sub_group(), v478_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v775_data = ir5[6];
            ir5[6] = (v775_data + (v735_data * (sycl::select_from_group(item.get_sub_group(), v484_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v781_data = ir5[7];
            ir5[7] = (v781_data + (v735_data * (sycl::select_from_group(item.get_sub_group(), v490_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v783_data = r3[7];
            float v787_data = ir5[0];
            ir5[0] = (v787_data + (v783_data * (sycl::select_from_group(item.get_sub_group(), v448_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v793_data = ir5[1];
            ir5[1] = (v793_data + (v783_data * (sycl::select_from_group(item.get_sub_group(), v454_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v799_data = ir5[2];
            ir5[2] = (v799_data + (v783_data * (sycl::select_from_group(item.get_sub_group(), v460_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v805_data = ir5[3];
            ir5[3] = (v805_data + (v783_data * (sycl::select_from_group(item.get_sub_group(), v466_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v811_data = ir5[4];
            ir5[4] = (v811_data + (v783_data * (sycl::select_from_group(item.get_sub_group(), v472_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v817_data = ir5[5];
            ir5[5] = (v817_data + (v783_data * (sycl::select_from_group(item.get_sub_group(), v478_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v823_data = ir5[6];
            ir5[6] = (v823_data + (v783_data * (sycl::select_from_group(item.get_sub_group(), v484_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v829_data = ir5[7];
            ir5[7] = (v829_data + (v783_data * (sycl::select_from_group(item.get_sub_group(), v490_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            // r5 = ir5 + r2
            #pragma unroll
            for (int32_t v831_n0 = 0; v831_n0 < 1; ++v831_n0) {
              #pragma unroll
              for (int32_t v832_n1 = 0; v832_n1 < 8; ++v832_n1) {
                int32_t v833_a = v831_n0 + v832_n1;
                float v834_data = ir5[v833_a];
                float v835_data = r2[v833_a];
                r5[v833_a] = (v835_data + v834_data);
              }
            }
            // s0 = store{r>s}(localShrMem0, r5);
            #pragma unroll
            for (int32_t v837_i0 = 0; v837_i0 < 1; ++v837_i0) {
              int32_t v842_lead = v24_lead + (v837_i0 * 8);
              #pragma unroll
              for (int32_t v838_i1 = 0; v838_i1 < 8; ++v838_i1) {
                float v840_data = r5[(v837_i0 + v838_i1)];
                int32_t v844_a = v842_lead + (v838_i1 * 8);
                s0[(v844_a ^ ((v844_a >> 5) & 31))] = v840_data;
              }
            }
            item.barrier();
            // glb_m4 = abs(s0)
            #pragma unroll
            for (int32_t v848_k0 = 0; v848_k0 < 1; ++v848_k0) {
              int32_t v851_lead = v24_lead + (v848_k0 * 8);
              #pragma unroll
              for (int32_t v849_k1 = 0; v849_k1 < 8; ++v849_k1) {
                int32_t v853_a = v851_lead + (v849_k1 * 8);
                float v857_data = s0[(v853_a ^ ((v853_a >> 5) & 31))];
                float v858_e = sycl::fabs(v857_data);
                if (batchIdActive0) {
                  glb_m4[v853_a] = v858_e;
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

