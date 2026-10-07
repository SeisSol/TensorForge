// === base name ===
kernel_66fd1fc6d510d587

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_66fd1fc6d510d587 = {{8, 2, 1}, 8, 8, 1, 2, 64, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_66fd1fc6d510d587(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_66fd1fc6d510d587(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_66fd1fc6d510d587(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_66fd1fc6d510d587(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_66fd1fc6d510d587(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_66fd1fc6d510d587(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_66fd1fc6d510d587(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
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
        //   m3 8×8(8×8) {0..8}×{0..8} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        //   C = abs(M)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,2,1],"cooperative":false,"lead_width":1,"mults_per_block":2,"persistent":true,"sections":[{"barrier":false,"mults_per_block":2,"shared_elements":16}],"shared_bytes":64,"shared_elements":16,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"M","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[8 * item.get_local_id(1) + 0];
          size_t v8_batchIdLane0 = item.get_local_id(1) % 2;
          int32_t v26_lead = item.get_local_id(2) % 8;
          for (size_t v9_batchIdGroup0 = (item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)); v9_batchIdGroup0 < numElements0; v9_batchIdGroup0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_row = v9_batchIdGroup0 + v8_batchIdLane0;
            const bool batchIdActive0 = v10_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v10_row]));
            size_t v12_batchId0 = batchIdActive0 ? v10_row : v9_batchIdGroup0;
            size_t v13_ahead1 = v12_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v15_batchId1 = (v13_ahead1 < numElements0) ? v13_ahead1 : v12_batchId0;
            float *const __restrict__ glb_m0 = &m0[v12_batchId0 * 64 + 0 + m0_extraOffset];
            const float *const __restrict__ glb_m1 = &m1[v12_batchId0 * 64 + 0 + m1_extraOffset];
            const float *const __restrict__ glb_m2 = &m2[v12_batchId0 * 64 + 0 + m2_extraOffset];
            float *const __restrict__ glb_m3 = &m3[v12_batchId0 * 64 + 0 + m3_extraOffset];
            float r0[8]{};
            // r0 = load{g>r}(glb_m1);
            #pragma unroll
            for (int32_t v27_i0 = 0; v27_i0 < 1; ++v27_i0) {
              int32_t v30_lead = v26_lead + (v27_i0 * 8);
              #pragma unroll
              for (int32_t v28_i1 = 0; v28_i1 < 8; ++v28_i1) {
                float v33_data = glb_m1[(v30_lead + (v28_i1 * 8))];
                r0[(v27_i0 + v28_i1)] = v33_data;
              }
            }
            float r1[8]{};
            // r1 = load{g>r}(glb_m2);
            #pragma unroll
            for (int32_t v36_i0 = 0; v36_i0 < 1; ++v36_i0) {
              int32_t v39_lead = v26_lead + (v36_i0 * 8);
              #pragma unroll
              for (int32_t v37_i1 = 0; v37_i1 < 8; ++v37_i1) {
                float v42_data = glb_m2[(v39_lead + (v37_i1 * 8))];
                r1[(v36_i0 + v37_i1)] = v42_data;
              }
            }
            float r2[8]{};
            // ir2 = +(r0 * r1)
            // [(0, 8), (0, 8)] [(0, 8)]
            float ir2[8]{};
            float v46_data = r0[0];
            float v47_data = r1[0];
            float v50_data = ir2[0];
            ir2[0] = (v50_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v53_data = r1[1];
            float v56_data = ir2[1];
            ir2[1] = (v56_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v59_data = r1[2];
            float v62_data = ir2[2];
            ir2[2] = (v62_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v65_data = r1[3];
            float v68_data = ir2[3];
            ir2[3] = (v68_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v71_data = r1[4];
            float v74_data = ir2[4];
            ir2[4] = (v74_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v77_data = r1[5];
            float v80_data = ir2[5];
            ir2[5] = (v80_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v83_data = r1[6];
            float v86_data = ir2[6];
            ir2[6] = (v86_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v89_data = r1[7];
            float v92_data = ir2[7];
            ir2[7] = (v92_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v94_data = r0[1];
            float v98_data = ir2[0];
            ir2[0] = (v98_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v104_data = ir2[1];
            ir2[1] = (v104_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v110_data = ir2[2];
            ir2[2] = (v110_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v116_data = ir2[3];
            ir2[3] = (v116_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v122_data = ir2[4];
            ir2[4] = (v122_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v128_data = ir2[5];
            ir2[5] = (v128_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v134_data = ir2[6];
            ir2[6] = (v134_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v140_data = ir2[7];
            ir2[7] = (v140_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v142_data = r0[2];
            float v146_data = ir2[0];
            ir2[0] = (v146_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v152_data = ir2[1];
            ir2[1] = (v152_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v158_data = ir2[2];
            ir2[2] = (v158_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v164_data = ir2[3];
            ir2[3] = (v164_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v170_data = ir2[4];
            ir2[4] = (v170_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v176_data = ir2[5];
            ir2[5] = (v176_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v182_data = ir2[6];
            ir2[6] = (v182_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v188_data = ir2[7];
            ir2[7] = (v188_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v190_data = r0[3];
            float v194_data = ir2[0];
            ir2[0] = (v194_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v200_data = ir2[1];
            ir2[1] = (v200_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v206_data = ir2[2];
            ir2[2] = (v206_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v212_data = ir2[3];
            ir2[3] = (v212_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v218_data = ir2[4];
            ir2[4] = (v218_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v224_data = ir2[5];
            ir2[5] = (v224_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v230_data = ir2[6];
            ir2[6] = (v230_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v236_data = ir2[7];
            ir2[7] = (v236_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v238_data = r0[4];
            float v242_data = ir2[0];
            ir2[0] = (v242_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v248_data = ir2[1];
            ir2[1] = (v248_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v254_data = ir2[2];
            ir2[2] = (v254_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v260_data = ir2[3];
            ir2[3] = (v260_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v266_data = ir2[4];
            ir2[4] = (v266_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v272_data = ir2[5];
            ir2[5] = (v272_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v278_data = ir2[6];
            ir2[6] = (v278_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v284_data = ir2[7];
            ir2[7] = (v284_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v286_data = r0[5];
            float v290_data = ir2[0];
            ir2[0] = (v290_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v296_data = ir2[1];
            ir2[1] = (v296_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v302_data = ir2[2];
            ir2[2] = (v302_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v308_data = ir2[3];
            ir2[3] = (v308_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v314_data = ir2[4];
            ir2[4] = (v314_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v320_data = ir2[5];
            ir2[5] = (v320_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v326_data = ir2[6];
            ir2[6] = (v326_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v332_data = ir2[7];
            ir2[7] = (v332_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v334_data = r0[6];
            float v338_data = ir2[0];
            ir2[0] = (v338_data + (v334_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v344_data = ir2[1];
            ir2[1] = (v344_data + (v334_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v350_data = ir2[2];
            ir2[2] = (v350_data + (v334_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v356_data = ir2[3];
            ir2[3] = (v356_data + (v334_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v362_data = ir2[4];
            ir2[4] = (v362_data + (v334_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v368_data = ir2[5];
            ir2[5] = (v368_data + (v334_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v374_data = ir2[6];
            ir2[6] = (v374_data + (v334_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v380_data = ir2[7];
            ir2[7] = (v380_data + (v334_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v382_data = r0[7];
            float v386_data = ir2[0];
            ir2[0] = (v386_data + (v382_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v392_data = ir2[1];
            ir2[1] = (v392_data + (v382_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v398_data = ir2[2];
            ir2[2] = (v398_data + (v382_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v404_data = ir2[3];
            ir2[3] = (v404_data + (v382_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v410_data = ir2[4];
            ir2[4] = (v410_data + (v382_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v416_data = ir2[5];
            ir2[5] = (v416_data + (v382_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v422_data = ir2[6];
            ir2[6] = (v422_data + (v382_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v428_data = ir2[7];
            ir2[7] = (v428_data + (v382_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            // r2 = ir2
            #pragma unroll
            for (int32_t v430_n0 = 0; v430_n0 < 1; ++v430_n0) {
              #pragma unroll
              for (int32_t v431_n1 = 0; v431_n1 < 8; ++v431_n1) {
                int32_t v432_a = v430_n0 + v431_n1;
                float v433_data = ir2[v432_a];
                r2[v432_a] = v433_data;
              }
            }
            // glb_m0 = store{r>g}(r2);
            #pragma unroll
            for (int32_t v434_i0 = 0; v434_i0 < 1; ++v434_i0) {
              #pragma unroll
              for (int32_t v435_i1 = 0; v435_i1 < 8; ++v435_i1) {
                float v437_data = r2[(v434_i0 + v435_i1)];
                if (batchIdActive0) {
                  glb_m0[((v26_lead + (v434_i0 * 8)) + (v435_i1 * 8))] = v437_data;
                }
              }
            }
            // glb_m3 = abs(glb_m0)
            #pragma unroll
            for (int32_t v442_k0 = 0; v442_k0 < 1; ++v442_k0) {
              int32_t v445_lead = v26_lead + (v442_k0 * 8);
              #pragma unroll
              for (int32_t v443_k1 = 0; v443_k1 < 8; ++v443_k1) {
                int32_t v447_a = v445_lead + (v443_k1 * 8);
                float v448_data = glb_m0[v447_a];
                float v449_e = sycl::fabs(v448_data);
                if (batchIdActive0) {
                  glb_m3[v447_a] = v449_e;
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

