// === base name ===
kernel_84f094baec016dd0

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_84f094baec016dd0 = {{16, 16, 1}, 16, 12, 1, 16, 10240, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_84f094baec016dd0(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_84f094baec016dd0(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_84f094baec016dd0(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 2560 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_84f094baec016dd0(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_84f094baec016dd0(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_84f094baec016dd0(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_84f094baec016dd0(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (2560, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 10240 B shared, occupancy grid
        // operands:
        //   m0 6×12(6×12) {0..6}×{0..12} strided
        //   m1 12×12(12×12) {0..12}×{0..12} strided
        //   m2 6×12(6×12) {0..6}×{0..12} strided
        //   m3 12×12(12×12) {0..12}×{0..12} strided
        //   m4 2×12(2×12) {0..2}×{0..12} strided
        //   m5 12×12(12×12) {0..12}×{0..12} strided
        // operations:
        //   t0[i,j]@{0..6}×{0..12} = m0[i,k] × m1[k,j]
        //   t0[i,j]@{6..12}×{0..12} = m2[i,k] × m1[k,j]
        //   m3[i,j] = t0[i,j]
        //   t0[i,j]@{6..12}×{0..12} = m4[i,k] × m1[k,j]
        //   m5[i,j] = t0[i,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2560}],"shared_bytes":10240,"shared_elements":2560,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B1","bbox":[[0,0],[6,12]],"name":"m0","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"X","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"N2","bbox":[[0,0],[2,12]],"name":"m4","ordered":false,"parts":1,"shape":[2,12],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m5","ordered":false,"parts":1,"shape":[12,12],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[2,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[2,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[160 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[144];
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v4_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v4_batchId0 < numElements0; v4_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v5_ahead1 = v4_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v4_batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v4_batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v4_batchId0 * 72 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v4_batchId0 * 144 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v4_batchId0 * 24 + 0 + m4_extraOffset];
              float *const __restrict__ glb_m5 = &m5[v4_batchId0 * 144 + 0 + m5_extraOffset];
              float r0[12]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v21_lead = item.get_local_id(2) % 16;
              bool v22_g = v21_lead < 6;
              if (v22_g) {
                #pragma unroll
                for (int32_t v23_i1 = 0; v23_i1 < 12; ++v23_i1) {
                  float v28_data = glb_m0[(v21_lead + (v23_i1 * 6))];
                  r0[v23_i1] = v28_data;
                }
              }
              float r1[12]{};
              // r1 = load{g>r}(glb_m1);
              bool v31_g = v21_lead < 12;
              if (v31_g) {
                #pragma unroll
                for (int32_t v32_i1 = 0; v32_i1 < 12; ++v32_i1) {
                  float v37_data = glb_m1[(v21_lead + (v32_i1 * 12))];
                  r1[v32_i1] = v37_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r3[12]{};
              // r3 = load{g>r}(glb_m2);
              if (v22_g) {
                #pragma unroll
                for (int32_t v40_i1 = 0; v40_i1 < 12; ++v40_i1) {
                  float v45_data = glb_m2[(v21_lead + (v40_i1 * 6))];
                  r3[v40_i1] = v45_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m1););
              float r2[12]{};
              // r2 = +(r0 * r1) + None
              // [(0, 6), (0, 12)] [(0, 12)]
              float v48_data = r0[0];
              float v49_data = r1[0];
              float v50_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v52_data = r2[0];
              r2[0] = (v52_data + (v48_data * v50_bc));
              float v55_data = r1[1];
              float v56_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v58_data = r2[1];
              r2[1] = (v58_data + (v48_data * v56_bc));
              float v61_data = r1[2];
              float v62_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v64_data = r2[2];
              r2[2] = (v64_data + (v48_data * v62_bc));
              float v67_data = r1[3];
              float v68_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v70_data = r2[3];
              r2[3] = (v70_data + (v48_data * v68_bc));
              float v73_data = r1[4];
              float v74_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v76_data = r2[4];
              r2[4] = (v76_data + (v48_data * v74_bc));
              float v79_data = r1[5];
              float v80_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v82_data = r2[5];
              r2[5] = (v82_data + (v48_data * v80_bc));
              float v85_data = r1[6];
              float v86_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v88_data = r2[6];
              r2[6] = (v88_data + (v48_data * v86_bc));
              float v91_data = r1[7];
              float v92_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v94_data = r2[7];
              r2[7] = (v94_data + (v48_data * v92_bc));
              float v97_data = r1[8];
              float v98_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v100_data = r2[8];
              r2[8] = (v100_data + (v48_data * v98_bc));
              float v103_data = r1[9];
              float v104_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v106_data = r2[9];
              r2[9] = (v106_data + (v48_data * v104_bc));
              float v109_data = r1[10];
              float v110_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v112_data = r2[10];
              r2[10] = (v112_data + (v48_data * v110_bc));
              float v115_data = r1[11];
              float v116_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v118_data = r2[11];
              r2[11] = (v118_data + (v48_data * v116_bc));
              float v120_data = r0[1];
              float v122_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v124_data = r2[0];
              r2[0] = (v124_data + (v120_data * v122_bc));
              float v128_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v130_data = r2[1];
              r2[1] = (v130_data + (v120_data * v128_bc));
              float v134_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v136_data = r2[2];
              r2[2] = (v136_data + (v120_data * v134_bc));
              float v140_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v142_data = r2[3];
              r2[3] = (v142_data + (v120_data * v140_bc));
              float v146_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v148_data = r2[4];
              r2[4] = (v148_data + (v120_data * v146_bc));
              float v152_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v154_data = r2[5];
              r2[5] = (v154_data + (v120_data * v152_bc));
              float v158_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v160_data = r2[6];
              r2[6] = (v160_data + (v120_data * v158_bc));
              float v164_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v166_data = r2[7];
              r2[7] = (v166_data + (v120_data * v164_bc));
              float v170_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v172_data = r2[8];
              r2[8] = (v172_data + (v120_data * v170_bc));
              float v176_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v178_data = r2[9];
              r2[9] = (v178_data + (v120_data * v176_bc));
              float v182_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v184_data = r2[10];
              r2[10] = (v184_data + (v120_data * v182_bc));
              float v188_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v190_data = r2[11];
              r2[11] = (v190_data + (v120_data * v188_bc));
              float v192_data = r0[2];
              float v194_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v196_data = r2[0];
              r2[0] = (v196_data + (v192_data * v194_bc));
              float v200_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v202_data = r2[1];
              r2[1] = (v202_data + (v192_data * v200_bc));
              float v206_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v208_data = r2[2];
              r2[2] = (v208_data + (v192_data * v206_bc));
              float v212_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v214_data = r2[3];
              r2[3] = (v214_data + (v192_data * v212_bc));
              float v218_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v220_data = r2[4];
              r2[4] = (v220_data + (v192_data * v218_bc));
              float v224_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v226_data = r2[5];
              r2[5] = (v226_data + (v192_data * v224_bc));
              float v230_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v232_data = r2[6];
              r2[6] = (v232_data + (v192_data * v230_bc));
              float v236_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v238_data = r2[7];
              r2[7] = (v238_data + (v192_data * v236_bc));
              float v242_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v244_data = r2[8];
              r2[8] = (v244_data + (v192_data * v242_bc));
              float v248_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v250_data = r2[9];
              r2[9] = (v250_data + (v192_data * v248_bc));
              float v254_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v256_data = r2[10];
              r2[10] = (v256_data + (v192_data * v254_bc));
              float v260_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v262_data = r2[11];
              r2[11] = (v262_data + (v192_data * v260_bc));
              float v264_data = r0[3];
              float v266_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v268_data = r2[0];
              r2[0] = (v268_data + (v264_data * v266_bc));
              float v272_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v274_data = r2[1];
              r2[1] = (v274_data + (v264_data * v272_bc));
              float v278_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v280_data = r2[2];
              r2[2] = (v280_data + (v264_data * v278_bc));
              float v284_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v286_data = r2[3];
              r2[3] = (v286_data + (v264_data * v284_bc));
              float v290_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v292_data = r2[4];
              r2[4] = (v292_data + (v264_data * v290_bc));
              float v296_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v298_data = r2[5];
              r2[5] = (v298_data + (v264_data * v296_bc));
              float v302_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v304_data = r2[6];
              r2[6] = (v304_data + (v264_data * v302_bc));
              float v308_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v310_data = r2[7];
              r2[7] = (v310_data + (v264_data * v308_bc));
              float v314_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v316_data = r2[8];
              r2[8] = (v316_data + (v264_data * v314_bc));
              float v320_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v322_data = r2[9];
              r2[9] = (v322_data + (v264_data * v320_bc));
              float v326_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v328_data = r2[10];
              r2[10] = (v328_data + (v264_data * v326_bc));
              float v332_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v334_data = r2[11];
              r2[11] = (v334_data + (v264_data * v332_bc));
              float v336_data = r0[4];
              float v338_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v340_data = r2[0];
              r2[0] = (v340_data + (v336_data * v338_bc));
              float v344_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v346_data = r2[1];
              r2[1] = (v346_data + (v336_data * v344_bc));
              float v350_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v352_data = r2[2];
              r2[2] = (v352_data + (v336_data * v350_bc));
              float v356_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v358_data = r2[3];
              r2[3] = (v358_data + (v336_data * v356_bc));
              float v362_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v364_data = r2[4];
              r2[4] = (v364_data + (v336_data * v362_bc));
              float v368_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v370_data = r2[5];
              r2[5] = (v370_data + (v336_data * v368_bc));
              float v374_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v376_data = r2[6];
              r2[6] = (v376_data + (v336_data * v374_bc));
              float v380_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v382_data = r2[7];
              r2[7] = (v382_data + (v336_data * v380_bc));
              float v386_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v388_data = r2[8];
              r2[8] = (v388_data + (v336_data * v386_bc));
              float v392_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v394_data = r2[9];
              r2[9] = (v394_data + (v336_data * v392_bc));
              float v398_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v400_data = r2[10];
              r2[10] = (v400_data + (v336_data * v398_bc));
              float v404_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v406_data = r2[11];
              r2[11] = (v406_data + (v336_data * v404_bc));
              float v408_data = r0[5];
              float v410_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v412_data = r2[0];
              r2[0] = (v412_data + (v408_data * v410_bc));
              float v416_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v418_data = r2[1];
              r2[1] = (v418_data + (v408_data * v416_bc));
              float v422_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v424_data = r2[2];
              r2[2] = (v424_data + (v408_data * v422_bc));
              float v428_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v430_data = r2[3];
              r2[3] = (v430_data + (v408_data * v428_bc));
              float v434_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v436_data = r2[4];
              r2[4] = (v436_data + (v408_data * v434_bc));
              float v440_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v442_data = r2[5];
              r2[5] = (v442_data + (v408_data * v440_bc));
              float v446_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v448_data = r2[6];
              r2[6] = (v448_data + (v408_data * v446_bc));
              float v452_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v454_data = r2[7];
              r2[7] = (v454_data + (v408_data * v452_bc));
              float v458_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v460_data = r2[8];
              r2[8] = (v460_data + (v408_data * v458_bc));
              float v464_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v466_data = r2[9];
              r2[9] = (v466_data + (v408_data * v464_bc));
              float v470_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v472_data = r2[10];
              r2[10] = (v472_data + (v408_data * v470_bc));
              float v476_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v478_data = r2[11];
              r2[11] = (v478_data + (v408_data * v476_bc));
              float v480_data = r0[6];
              float v482_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v484_data = r2[0];
              r2[0] = (v484_data + (v480_data * v482_bc));
              float v488_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v490_data = r2[1];
              r2[1] = (v490_data + (v480_data * v488_bc));
              float v494_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v496_data = r2[2];
              r2[2] = (v496_data + (v480_data * v494_bc));
              float v500_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v502_data = r2[3];
              r2[3] = (v502_data + (v480_data * v500_bc));
              float v506_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v508_data = r2[4];
              r2[4] = (v508_data + (v480_data * v506_bc));
              float v512_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v514_data = r2[5];
              r2[5] = (v514_data + (v480_data * v512_bc));
              float v518_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v520_data = r2[6];
              r2[6] = (v520_data + (v480_data * v518_bc));
              float v524_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v526_data = r2[7];
              r2[7] = (v526_data + (v480_data * v524_bc));
              float v530_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v532_data = r2[8];
              r2[8] = (v532_data + (v480_data * v530_bc));
              float v536_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v538_data = r2[9];
              r2[9] = (v538_data + (v480_data * v536_bc));
              float v542_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v544_data = r2[10];
              r2[10] = (v544_data + (v480_data * v542_bc));
              float v548_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v550_data = r2[11];
              r2[11] = (v550_data + (v480_data * v548_bc));
              float v552_data = r0[7];
              float v554_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v556_data = r2[0];
              r2[0] = (v556_data + (v552_data * v554_bc));
              float v560_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v562_data = r2[1];
              r2[1] = (v562_data + (v552_data * v560_bc));
              float v566_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v568_data = r2[2];
              r2[2] = (v568_data + (v552_data * v566_bc));
              float v572_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v574_data = r2[3];
              r2[3] = (v574_data + (v552_data * v572_bc));
              float v578_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v580_data = r2[4];
              r2[4] = (v580_data + (v552_data * v578_bc));
              float v584_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v586_data = r2[5];
              r2[5] = (v586_data + (v552_data * v584_bc));
              float v590_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v592_data = r2[6];
              r2[6] = (v592_data + (v552_data * v590_bc));
              float v596_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v598_data = r2[7];
              r2[7] = (v598_data + (v552_data * v596_bc));
              float v602_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v604_data = r2[8];
              r2[8] = (v604_data + (v552_data * v602_bc));
              float v608_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v610_data = r2[9];
              r2[9] = (v610_data + (v552_data * v608_bc));
              float v614_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v616_data = r2[10];
              r2[10] = (v616_data + (v552_data * v614_bc));
              float v620_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v622_data = r2[11];
              r2[11] = (v622_data + (v552_data * v620_bc));
              float v624_data = r0[8];
              float v626_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v628_data = r2[0];
              r2[0] = (v628_data + (v624_data * v626_bc));
              float v632_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v634_data = r2[1];
              r2[1] = (v634_data + (v624_data * v632_bc));
              float v638_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v640_data = r2[2];
              r2[2] = (v640_data + (v624_data * v638_bc));
              float v644_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v646_data = r2[3];
              r2[3] = (v646_data + (v624_data * v644_bc));
              float v650_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v652_data = r2[4];
              r2[4] = (v652_data + (v624_data * v650_bc));
              float v656_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v658_data = r2[5];
              r2[5] = (v658_data + (v624_data * v656_bc));
              float v662_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v664_data = r2[6];
              r2[6] = (v664_data + (v624_data * v662_bc));
              float v668_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v670_data = r2[7];
              r2[7] = (v670_data + (v624_data * v668_bc));
              float v674_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v676_data = r2[8];
              r2[8] = (v676_data + (v624_data * v674_bc));
              float v680_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v682_data = r2[9];
              r2[9] = (v682_data + (v624_data * v680_bc));
              float v686_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v688_data = r2[10];
              r2[10] = (v688_data + (v624_data * v686_bc));
              float v692_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v694_data = r2[11];
              r2[11] = (v694_data + (v624_data * v692_bc));
              float v696_data = r0[9];
              float v698_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v700_data = r2[0];
              r2[0] = (v700_data + (v696_data * v698_bc));
              float v704_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v706_data = r2[1];
              r2[1] = (v706_data + (v696_data * v704_bc));
              float v710_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v712_data = r2[2];
              r2[2] = (v712_data + (v696_data * v710_bc));
              float v716_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v718_data = r2[3];
              r2[3] = (v718_data + (v696_data * v716_bc));
              float v722_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v724_data = r2[4];
              r2[4] = (v724_data + (v696_data * v722_bc));
              float v728_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v730_data = r2[5];
              r2[5] = (v730_data + (v696_data * v728_bc));
              float v734_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v736_data = r2[6];
              r2[6] = (v736_data + (v696_data * v734_bc));
              float v740_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v742_data = r2[7];
              r2[7] = (v742_data + (v696_data * v740_bc));
              float v746_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v748_data = r2[8];
              r2[8] = (v748_data + (v696_data * v746_bc));
              float v752_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v754_data = r2[9];
              r2[9] = (v754_data + (v696_data * v752_bc));
              float v758_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v760_data = r2[10];
              r2[10] = (v760_data + (v696_data * v758_bc));
              float v764_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v766_data = r2[11];
              r2[11] = (v766_data + (v696_data * v764_bc));
              float v768_data = r0[10];
              float v770_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v772_data = r2[0];
              r2[0] = (v772_data + (v768_data * v770_bc));
              float v776_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v778_data = r2[1];
              r2[1] = (v778_data + (v768_data * v776_bc));
              float v782_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v784_data = r2[2];
              r2[2] = (v784_data + (v768_data * v782_bc));
              float v788_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v790_data = r2[3];
              r2[3] = (v790_data + (v768_data * v788_bc));
              float v794_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v796_data = r2[4];
              r2[4] = (v796_data + (v768_data * v794_bc));
              float v800_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v802_data = r2[5];
              r2[5] = (v802_data + (v768_data * v800_bc));
              float v806_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v808_data = r2[6];
              r2[6] = (v808_data + (v768_data * v806_bc));
              float v812_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v814_data = r2[7];
              r2[7] = (v814_data + (v768_data * v812_bc));
              float v818_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v820_data = r2[8];
              r2[8] = (v820_data + (v768_data * v818_bc));
              float v824_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v826_data = r2[9];
              r2[9] = (v826_data + (v768_data * v824_bc));
              float v830_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v832_data = r2[10];
              r2[10] = (v832_data + (v768_data * v830_bc));
              float v836_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v838_data = r2[11];
              r2[11] = (v838_data + (v768_data * v836_bc));
              float v840_data = r0[11];
              float v842_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v844_data = r2[0];
              r2[0] = (v844_data + (v840_data * v842_bc));
              float v848_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v850_data = r2[1];
              r2[1] = (v850_data + (v840_data * v848_bc));
              float v854_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v856_data = r2[2];
              r2[2] = (v856_data + (v840_data * v854_bc));
              float v860_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v862_data = r2[3];
              r2[3] = (v862_data + (v840_data * v860_bc));
              float v866_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v868_data = r2[4];
              r2[4] = (v868_data + (v840_data * v866_bc));
              float v872_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v874_data = r2[5];
              r2[5] = (v874_data + (v840_data * v872_bc));
              float v878_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v880_data = r2[6];
              r2[6] = (v880_data + (v840_data * v878_bc));
              float v884_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v886_data = r2[7];
              r2[7] = (v886_data + (v840_data * v884_bc));
              float v890_bc = sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v892_data = r2[8];
              r2[8] = (v892_data + (v840_data * v890_bc));
              float v896_bc = sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v898_data = r2[9];
              r2[9] = (v898_data + (v840_data * v896_bc));
              float v902_bc = sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v904_data = r2[10];
              r2[10] = (v904_data + (v840_data * v902_bc));
              float v908_bc = sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v910_data = r2[11];
              r2[11] = (v910_data + (v840_data * v908_bc));
              // s0 = store{r>s}(localShrMem0, r2);
              if (v22_g) {
                #pragma unroll
                for (int32_t v912_i1 = 0; v912_i1 < 12; ++v912_i1) {
                  float v914_data = r2[v912_i1];
                  int32_t v918_a = v21_lead + (v912_i1 * 12);
                  s0[(v918_a ^ ((v918_a >> 4) & 15))] = v914_data;
                }
              }
              float r6[12]{};
              // r6 = load{g>r}(glb_m4);
              bool v923_g = v21_lead < 2;
              if (v923_g) {
                #pragma unroll
                for (int32_t v924_i1 = 0; v924_i1 < 12; ++v924_i1) {
                  float v929_data = glb_m4[(v21_lead + (v924_i1 * 2))];
                  r6[v924_i1] = v929_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m2););
              float r4[12]{};
              // ir4 = +(r3 * r1)
              // [(0, 6), (0, 12)] [(0, 12)]
              float ir4[12]{};
              float v933_data = r3[0];
              float v937_data = ir4[0];
              ir4[0] = (v937_data + (v933_data * v50_bc));
              float v943_data = ir4[1];
              ir4[1] = (v943_data + (v933_data * v56_bc));
              float v949_data = ir4[2];
              ir4[2] = (v949_data + (v933_data * v62_bc));
              float v955_data = ir4[3];
              ir4[3] = (v955_data + (v933_data * v68_bc));
              float v961_data = ir4[4];
              ir4[4] = (v961_data + (v933_data * v74_bc));
              float v967_data = ir4[5];
              ir4[5] = (v967_data + (v933_data * v80_bc));
              float v973_data = ir4[6];
              ir4[6] = (v973_data + (v933_data * v86_bc));
              float v979_data = ir4[7];
              ir4[7] = (v979_data + (v933_data * v92_bc));
              float v985_data = ir4[8];
              ir4[8] = (v985_data + (v933_data * v98_bc));
              float v991_data = ir4[9];
              ir4[9] = (v991_data + (v933_data * v104_bc));
              float v997_data = ir4[10];
              ir4[10] = (v997_data + (v933_data * v110_bc));
              float v1003_data = ir4[11];
              ir4[11] = (v1003_data + (v933_data * v116_bc));
              float v1005_data = r3[1];
              float v1009_data = ir4[0];
              ir4[0] = (v1009_data + (v1005_data * v122_bc));
              float v1015_data = ir4[1];
              ir4[1] = (v1015_data + (v1005_data * v128_bc));
              float v1021_data = ir4[2];
              ir4[2] = (v1021_data + (v1005_data * v134_bc));
              float v1027_data = ir4[3];
              ir4[3] = (v1027_data + (v1005_data * v140_bc));
              float v1033_data = ir4[4];
              ir4[4] = (v1033_data + (v1005_data * v146_bc));
              float v1039_data = ir4[5];
              ir4[5] = (v1039_data + (v1005_data * v152_bc));
              float v1045_data = ir4[6];
              ir4[6] = (v1045_data + (v1005_data * v158_bc));
              float v1051_data = ir4[7];
              ir4[7] = (v1051_data + (v1005_data * v164_bc));
              float v1057_data = ir4[8];
              ir4[8] = (v1057_data + (v1005_data * v170_bc));
              float v1063_data = ir4[9];
              ir4[9] = (v1063_data + (v1005_data * v176_bc));
              float v1069_data = ir4[10];
              ir4[10] = (v1069_data + (v1005_data * v182_bc));
              float v1075_data = ir4[11];
              ir4[11] = (v1075_data + (v1005_data * v188_bc));
              float v1077_data = r3[2];
              float v1081_data = ir4[0];
              ir4[0] = (v1081_data + (v1077_data * v194_bc));
              float v1087_data = ir4[1];
              ir4[1] = (v1087_data + (v1077_data * v200_bc));
              float v1093_data = ir4[2];
              ir4[2] = (v1093_data + (v1077_data * v206_bc));
              float v1099_data = ir4[3];
              ir4[3] = (v1099_data + (v1077_data * v212_bc));
              float v1105_data = ir4[4];
              ir4[4] = (v1105_data + (v1077_data * v218_bc));
              float v1111_data = ir4[5];
              ir4[5] = (v1111_data + (v1077_data * v224_bc));
              float v1117_data = ir4[6];
              ir4[6] = (v1117_data + (v1077_data * v230_bc));
              float v1123_data = ir4[7];
              ir4[7] = (v1123_data + (v1077_data * v236_bc));
              float v1129_data = ir4[8];
              ir4[8] = (v1129_data + (v1077_data * v242_bc));
              float v1135_data = ir4[9];
              ir4[9] = (v1135_data + (v1077_data * v248_bc));
              float v1141_data = ir4[10];
              ir4[10] = (v1141_data + (v1077_data * v254_bc));
              float v1147_data = ir4[11];
              ir4[11] = (v1147_data + (v1077_data * v260_bc));
              float v1149_data = r3[3];
              float v1153_data = ir4[0];
              ir4[0] = (v1153_data + (v1149_data * v266_bc));
              float v1159_data = ir4[1];
              ir4[1] = (v1159_data + (v1149_data * v272_bc));
              float v1165_data = ir4[2];
              ir4[2] = (v1165_data + (v1149_data * v278_bc));
              float v1171_data = ir4[3];
              ir4[3] = (v1171_data + (v1149_data * v284_bc));
              float v1177_data = ir4[4];
              ir4[4] = (v1177_data + (v1149_data * v290_bc));
              float v1183_data = ir4[5];
              ir4[5] = (v1183_data + (v1149_data * v296_bc));
              float v1189_data = ir4[6];
              ir4[6] = (v1189_data + (v1149_data * v302_bc));
              float v1195_data = ir4[7];
              ir4[7] = (v1195_data + (v1149_data * v308_bc));
              float v1201_data = ir4[8];
              ir4[8] = (v1201_data + (v1149_data * v314_bc));
              float v1207_data = ir4[9];
              ir4[9] = (v1207_data + (v1149_data * v320_bc));
              float v1213_data = ir4[10];
              ir4[10] = (v1213_data + (v1149_data * v326_bc));
              float v1219_data = ir4[11];
              ir4[11] = (v1219_data + (v1149_data * v332_bc));
              float v1221_data = r3[4];
              float v1225_data = ir4[0];
              ir4[0] = (v1225_data + (v1221_data * v338_bc));
              float v1231_data = ir4[1];
              ir4[1] = (v1231_data + (v1221_data * v344_bc));
              float v1237_data = ir4[2];
              ir4[2] = (v1237_data + (v1221_data * v350_bc));
              float v1243_data = ir4[3];
              ir4[3] = (v1243_data + (v1221_data * v356_bc));
              float v1249_data = ir4[4];
              ir4[4] = (v1249_data + (v1221_data * v362_bc));
              float v1255_data = ir4[5];
              ir4[5] = (v1255_data + (v1221_data * v368_bc));
              float v1261_data = ir4[6];
              ir4[6] = (v1261_data + (v1221_data * v374_bc));
              float v1267_data = ir4[7];
              ir4[7] = (v1267_data + (v1221_data * v380_bc));
              float v1273_data = ir4[8];
              ir4[8] = (v1273_data + (v1221_data * v386_bc));
              float v1279_data = ir4[9];
              ir4[9] = (v1279_data + (v1221_data * v392_bc));
              float v1285_data = ir4[10];
              ir4[10] = (v1285_data + (v1221_data * v398_bc));
              float v1291_data = ir4[11];
              ir4[11] = (v1291_data + (v1221_data * v404_bc));
              float v1293_data = r3[5];
              float v1297_data = ir4[0];
              ir4[0] = (v1297_data + (v1293_data * v410_bc));
              float v1303_data = ir4[1];
              ir4[1] = (v1303_data + (v1293_data * v416_bc));
              float v1309_data = ir4[2];
              ir4[2] = (v1309_data + (v1293_data * v422_bc));
              float v1315_data = ir4[3];
              ir4[3] = (v1315_data + (v1293_data * v428_bc));
              float v1321_data = ir4[4];
              ir4[4] = (v1321_data + (v1293_data * v434_bc));
              float v1327_data = ir4[5];
              ir4[5] = (v1327_data + (v1293_data * v440_bc));
              float v1333_data = ir4[6];
              ir4[6] = (v1333_data + (v1293_data * v446_bc));
              float v1339_data = ir4[7];
              ir4[7] = (v1339_data + (v1293_data * v452_bc));
              float v1345_data = ir4[8];
              ir4[8] = (v1345_data + (v1293_data * v458_bc));
              float v1351_data = ir4[9];
              ir4[9] = (v1351_data + (v1293_data * v464_bc));
              float v1357_data = ir4[10];
              ir4[10] = (v1357_data + (v1293_data * v470_bc));
              float v1363_data = ir4[11];
              ir4[11] = (v1363_data + (v1293_data * v476_bc));
              float v1365_data = r3[6];
              float v1369_data = ir4[0];
              ir4[0] = (v1369_data + (v1365_data * v482_bc));
              float v1375_data = ir4[1];
              ir4[1] = (v1375_data + (v1365_data * v488_bc));
              float v1381_data = ir4[2];
              ir4[2] = (v1381_data + (v1365_data * v494_bc));
              float v1387_data = ir4[3];
              ir4[3] = (v1387_data + (v1365_data * v500_bc));
              float v1393_data = ir4[4];
              ir4[4] = (v1393_data + (v1365_data * v506_bc));
              float v1399_data = ir4[5];
              ir4[5] = (v1399_data + (v1365_data * v512_bc));
              float v1405_data = ir4[6];
              ir4[6] = (v1405_data + (v1365_data * v518_bc));
              float v1411_data = ir4[7];
              ir4[7] = (v1411_data + (v1365_data * v524_bc));
              float v1417_data = ir4[8];
              ir4[8] = (v1417_data + (v1365_data * v530_bc));
              float v1423_data = ir4[9];
              ir4[9] = (v1423_data + (v1365_data * v536_bc));
              float v1429_data = ir4[10];
              ir4[10] = (v1429_data + (v1365_data * v542_bc));
              float v1435_data = ir4[11];
              ir4[11] = (v1435_data + (v1365_data * v548_bc));
              float v1437_data = r3[7];
              float v1441_data = ir4[0];
              ir4[0] = (v1441_data + (v1437_data * v554_bc));
              float v1447_data = ir4[1];
              ir4[1] = (v1447_data + (v1437_data * v560_bc));
              float v1453_data = ir4[2];
              ir4[2] = (v1453_data + (v1437_data * v566_bc));
              float v1459_data = ir4[3];
              ir4[3] = (v1459_data + (v1437_data * v572_bc));
              float v1465_data = ir4[4];
              ir4[4] = (v1465_data + (v1437_data * v578_bc));
              float v1471_data = ir4[5];
              ir4[5] = (v1471_data + (v1437_data * v584_bc));
              float v1477_data = ir4[6];
              ir4[6] = (v1477_data + (v1437_data * v590_bc));
              float v1483_data = ir4[7];
              ir4[7] = (v1483_data + (v1437_data * v596_bc));
              float v1489_data = ir4[8];
              ir4[8] = (v1489_data + (v1437_data * v602_bc));
              float v1495_data = ir4[9];
              ir4[9] = (v1495_data + (v1437_data * v608_bc));
              float v1501_data = ir4[10];
              ir4[10] = (v1501_data + (v1437_data * v614_bc));
              float v1507_data = ir4[11];
              ir4[11] = (v1507_data + (v1437_data * v620_bc));
              float v1509_data = r3[8];
              float v1513_data = ir4[0];
              ir4[0] = (v1513_data + (v1509_data * v626_bc));
              float v1519_data = ir4[1];
              ir4[1] = (v1519_data + (v1509_data * v632_bc));
              float v1525_data = ir4[2];
              ir4[2] = (v1525_data + (v1509_data * v638_bc));
              float v1531_data = ir4[3];
              ir4[3] = (v1531_data + (v1509_data * v644_bc));
              float v1537_data = ir4[4];
              ir4[4] = (v1537_data + (v1509_data * v650_bc));
              float v1543_data = ir4[5];
              ir4[5] = (v1543_data + (v1509_data * v656_bc));
              float v1549_data = ir4[6];
              ir4[6] = (v1549_data + (v1509_data * v662_bc));
              float v1555_data = ir4[7];
              ir4[7] = (v1555_data + (v1509_data * v668_bc));
              float v1561_data = ir4[8];
              ir4[8] = (v1561_data + (v1509_data * v674_bc));
              float v1567_data = ir4[9];
              ir4[9] = (v1567_data + (v1509_data * v680_bc));
              float v1573_data = ir4[10];
              ir4[10] = (v1573_data + (v1509_data * v686_bc));
              float v1579_data = ir4[11];
              ir4[11] = (v1579_data + (v1509_data * v692_bc));
              float v1581_data = r3[9];
              float v1585_data = ir4[0];
              ir4[0] = (v1585_data + (v1581_data * v698_bc));
              float v1591_data = ir4[1];
              ir4[1] = (v1591_data + (v1581_data * v704_bc));
              float v1597_data = ir4[2];
              ir4[2] = (v1597_data + (v1581_data * v710_bc));
              float v1603_data = ir4[3];
              ir4[3] = (v1603_data + (v1581_data * v716_bc));
              float v1609_data = ir4[4];
              ir4[4] = (v1609_data + (v1581_data * v722_bc));
              float v1615_data = ir4[5];
              ir4[5] = (v1615_data + (v1581_data * v728_bc));
              float v1621_data = ir4[6];
              ir4[6] = (v1621_data + (v1581_data * v734_bc));
              float v1627_data = ir4[7];
              ir4[7] = (v1627_data + (v1581_data * v740_bc));
              float v1633_data = ir4[8];
              ir4[8] = (v1633_data + (v1581_data * v746_bc));
              float v1639_data = ir4[9];
              ir4[9] = (v1639_data + (v1581_data * v752_bc));
              float v1645_data = ir4[10];
              ir4[10] = (v1645_data + (v1581_data * v758_bc));
              float v1651_data = ir4[11];
              ir4[11] = (v1651_data + (v1581_data * v764_bc));
              float v1653_data = r3[10];
              float v1657_data = ir4[0];
              ir4[0] = (v1657_data + (v1653_data * v770_bc));
              float v1663_data = ir4[1];
              ir4[1] = (v1663_data + (v1653_data * v776_bc));
              float v1669_data = ir4[2];
              ir4[2] = (v1669_data + (v1653_data * v782_bc));
              float v1675_data = ir4[3];
              ir4[3] = (v1675_data + (v1653_data * v788_bc));
              float v1681_data = ir4[4];
              ir4[4] = (v1681_data + (v1653_data * v794_bc));
              float v1687_data = ir4[5];
              ir4[5] = (v1687_data + (v1653_data * v800_bc));
              float v1693_data = ir4[6];
              ir4[6] = (v1693_data + (v1653_data * v806_bc));
              float v1699_data = ir4[7];
              ir4[7] = (v1699_data + (v1653_data * v812_bc));
              float v1705_data = ir4[8];
              ir4[8] = (v1705_data + (v1653_data * v818_bc));
              float v1711_data = ir4[9];
              ir4[9] = (v1711_data + (v1653_data * v824_bc));
              float v1717_data = ir4[10];
              ir4[10] = (v1717_data + (v1653_data * v830_bc));
              float v1723_data = ir4[11];
              ir4[11] = (v1723_data + (v1653_data * v836_bc));
              float v1725_data = r3[11];
              float v1729_data = ir4[0];
              ir4[0] = (v1729_data + (v1725_data * v842_bc));
              float v1735_data = ir4[1];
              ir4[1] = (v1735_data + (v1725_data * v848_bc));
              float v1741_data = ir4[2];
              ir4[2] = (v1741_data + (v1725_data * v854_bc));
              float v1747_data = ir4[3];
              ir4[3] = (v1747_data + (v1725_data * v860_bc));
              float v1753_data = ir4[4];
              ir4[4] = (v1753_data + (v1725_data * v866_bc));
              float v1759_data = ir4[5];
              ir4[5] = (v1759_data + (v1725_data * v872_bc));
              float v1765_data = ir4[6];
              ir4[6] = (v1765_data + (v1725_data * v878_bc));
              float v1771_data = ir4[7];
              ir4[7] = (v1771_data + (v1725_data * v884_bc));
              float v1777_data = ir4[8];
              ir4[8] = (v1777_data + (v1725_data * v890_bc));
              float v1783_data = ir4[9];
              ir4[9] = (v1783_data + (v1725_data * v896_bc));
              float v1789_data = ir4[10];
              ir4[10] = (v1789_data + (v1725_data * v902_bc));
              float v1795_data = ir4[11];
              ir4[11] = (v1795_data + (v1725_data * v908_bc));
              // r4 = ir4
              if (v22_g) {
                #pragma unroll
                for (int32_t v1797_n1 = 0; v1797_n1 < 12; ++v1797_n1) {
                  float v1799_data = ir4[v1797_n1];
                  r4[v1797_n1] = v1799_data;
                }
              }
              // s0 = store{r>s}(localShrMem0, r4);
              if (v22_g) {
                int32_t v1805_off = v21_lead + 6;
                #pragma unroll
                for (int32_t v1800_i1 = 0; v1800_i1 < 12; ++v1800_i1) {
                  float v1802_data = r4[v1800_i1];
                  int32_t v1807_a = v1805_off + (v1800_i1 * 12);
                  s0[(v1807_a ^ ((v1807_a >> 4) & 15))] = v1802_data;
                }
              }
              float r5[12]{};
              sycl::group_barrier(item.get_sub_group());
              // ir5 = +(s0)
              // [(0, 12), (0, 12)] []
              float ir5[12]{};
              int32_t v1817_sw = (v21_lead >> 4) & 15;
              float v1819_data = v31_g ? (s0[(v21_lead ^ v1817_sw)]) : (0.0f);
              float v1820_data = ir5[0];
              ir5[0] = (v1820_data + v1819_data);
              int32_t v1822_a = v21_lead + 12;
              int32_t v1823_sw = v1822_a >> 4;
              float v1826_data = v31_g ? (s0[(v1822_a ^ (v1823_sw & 15))]) : (0.0f);
              float v1827_data = ir5[1];
              ir5[1] = (v1827_data + v1826_data);
              int32_t v1829_a = v21_lead + 24;
              int32_t v1830_sw = v1829_a >> 4;
              float v1833_data = v31_g ? (s0[(v1829_a ^ (v1830_sw & 15))]) : (0.0f);
              float v1834_data = ir5[2];
              ir5[2] = (v1834_data + v1833_data);
              int32_t v1836_a = v21_lead + 36;
              int32_t v1837_sw = v1836_a >> 4;
              float v1840_data = v31_g ? (s0[(v1836_a ^ (v1837_sw & 15))]) : (0.0f);
              float v1841_data = ir5[3];
              ir5[3] = (v1841_data + v1840_data);
              int32_t v1843_a = v21_lead + 48;
              int32_t v1844_sw = v1843_a >> 4;
              float v1847_data = v31_g ? (s0[(v1843_a ^ (v1844_sw & 15))]) : (0.0f);
              float v1848_data = ir5[4];
              ir5[4] = (v1848_data + v1847_data);
              int32_t v1850_a = v21_lead + 60;
              int32_t v1851_sw = v1850_a >> 4;
              float v1854_data = v31_g ? (s0[(v1850_a ^ (v1851_sw & 15))]) : (0.0f);
              float v1855_data = ir5[5];
              ir5[5] = (v1855_data + v1854_data);
              int32_t v1857_a = v21_lead + 72;
              int32_t v1858_sw = v1857_a >> 4;
              float v1861_data = v31_g ? (s0[(v1857_a ^ (v1858_sw & 15))]) : (0.0f);
              float v1862_data = ir5[6];
              ir5[6] = (v1862_data + v1861_data);
              int32_t v1864_a = v21_lead + 84;
              int32_t v1865_sw = v1864_a >> 4;
              float v1868_data = v31_g ? (s0[(v1864_a ^ (v1865_sw & 15))]) : (0.0f);
              float v1869_data = ir5[7];
              ir5[7] = (v1869_data + v1868_data);
              int32_t v1871_a = v21_lead + 96;
              int32_t v1872_sw = v1871_a >> 4;
              float v1875_data = v31_g ? (s0[(v1871_a ^ (v1872_sw & 15))]) : (0.0f);
              float v1876_data = ir5[8];
              ir5[8] = (v1876_data + v1875_data);
              int32_t v1878_a = v21_lead + 108;
              int32_t v1879_sw = v1878_a >> 4;
              float v1882_data = v31_g ? (s0[(v1878_a ^ (v1879_sw & 15))]) : (0.0f);
              float v1883_data = ir5[9];
              ir5[9] = (v1883_data + v1882_data);
              int32_t v1885_a = v21_lead + 120;
              int32_t v1886_sw = v1885_a >> 4;
              float v1889_data = v31_g ? (s0[(v1885_a ^ (v1886_sw & 15))]) : (0.0f);
              float v1890_data = ir5[10];
              ir5[10] = (v1890_data + v1889_data);
              int32_t v1892_a = v21_lead + 132;
              int32_t v1893_sw = v1892_a >> 4;
              float v1896_data = v31_g ? (s0[(v1892_a ^ (v1893_sw & 15))]) : (0.0f);
              float v1897_data = ir5[11];
              ir5[11] = (v1897_data + v1896_data);
              // r5 = ir5
              if (v31_g) {
                #pragma unroll
                for (int32_t v1899_n1 = 0; v1899_n1 < 12; ++v1899_n1) {
                  float v1901_data = ir5[v1899_n1];
                  r5[v1899_n1] = v1901_data;
                }
              }
              // glb_m3 = store{r>g}(r5);
              if (v31_g) {
                #pragma unroll
                for (int32_t v1902_i1 = 0; v1902_i1 < 12; ++v1902_i1) {
                  float v1904_data = r5[v1902_i1];
                  glb_m3[(v21_lead + (v1902_i1 * 12))] = v1904_data;
                }
              }
              // wait(r6 = load{g>r}(glb_m4););
              float r7[12]{};
              // ir7 = +(r6 * r1)
              // [(0, 2), (0, 12)] [(0, 12)]
              float ir7[12]{};
              float v1911_data = r6[0];
              float v1915_data = ir7[0];
              ir7[0] = (v1915_data + (v1911_data * v50_bc));
              float v1921_data = ir7[1];
              ir7[1] = (v1921_data + (v1911_data * v56_bc));
              float v1927_data = ir7[2];
              ir7[2] = (v1927_data + (v1911_data * v62_bc));
              float v1933_data = ir7[3];
              ir7[3] = (v1933_data + (v1911_data * v68_bc));
              float v1939_data = ir7[4];
              ir7[4] = (v1939_data + (v1911_data * v74_bc));
              float v1945_data = ir7[5];
              ir7[5] = (v1945_data + (v1911_data * v80_bc));
              float v1951_data = ir7[6];
              ir7[6] = (v1951_data + (v1911_data * v86_bc));
              float v1957_data = ir7[7];
              ir7[7] = (v1957_data + (v1911_data * v92_bc));
              float v1963_data = ir7[8];
              ir7[8] = (v1963_data + (v1911_data * v98_bc));
              float v1969_data = ir7[9];
              ir7[9] = (v1969_data + (v1911_data * v104_bc));
              float v1975_data = ir7[10];
              ir7[10] = (v1975_data + (v1911_data * v110_bc));
              float v1981_data = ir7[11];
              ir7[11] = (v1981_data + (v1911_data * v116_bc));
              float v1983_data = r6[1];
              float v1987_data = ir7[0];
              ir7[0] = (v1987_data + (v1983_data * v122_bc));
              float v1993_data = ir7[1];
              ir7[1] = (v1993_data + (v1983_data * v128_bc));
              float v1999_data = ir7[2];
              ir7[2] = (v1999_data + (v1983_data * v134_bc));
              float v2005_data = ir7[3];
              ir7[3] = (v2005_data + (v1983_data * v140_bc));
              float v2011_data = ir7[4];
              ir7[4] = (v2011_data + (v1983_data * v146_bc));
              float v2017_data = ir7[5];
              ir7[5] = (v2017_data + (v1983_data * v152_bc));
              float v2023_data = ir7[6];
              ir7[6] = (v2023_data + (v1983_data * v158_bc));
              float v2029_data = ir7[7];
              ir7[7] = (v2029_data + (v1983_data * v164_bc));
              float v2035_data = ir7[8];
              ir7[8] = (v2035_data + (v1983_data * v170_bc));
              float v2041_data = ir7[9];
              ir7[9] = (v2041_data + (v1983_data * v176_bc));
              float v2047_data = ir7[10];
              ir7[10] = (v2047_data + (v1983_data * v182_bc));
              float v2053_data = ir7[11];
              ir7[11] = (v2053_data + (v1983_data * v188_bc));
              float v2055_data = r6[2];
              float v2059_data = ir7[0];
              ir7[0] = (v2059_data + (v2055_data * v194_bc));
              float v2065_data = ir7[1];
              ir7[1] = (v2065_data + (v2055_data * v200_bc));
              float v2071_data = ir7[2];
              ir7[2] = (v2071_data + (v2055_data * v206_bc));
              float v2077_data = ir7[3];
              ir7[3] = (v2077_data + (v2055_data * v212_bc));
              float v2083_data = ir7[4];
              ir7[4] = (v2083_data + (v2055_data * v218_bc));
              float v2089_data = ir7[5];
              ir7[5] = (v2089_data + (v2055_data * v224_bc));
              float v2095_data = ir7[6];
              ir7[6] = (v2095_data + (v2055_data * v230_bc));
              float v2101_data = ir7[7];
              ir7[7] = (v2101_data + (v2055_data * v236_bc));
              float v2107_data = ir7[8];
              ir7[8] = (v2107_data + (v2055_data * v242_bc));
              float v2113_data = ir7[9];
              ir7[9] = (v2113_data + (v2055_data * v248_bc));
              float v2119_data = ir7[10];
              ir7[10] = (v2119_data + (v2055_data * v254_bc));
              float v2125_data = ir7[11];
              ir7[11] = (v2125_data + (v2055_data * v260_bc));
              float v2127_data = r6[3];
              float v2131_data = ir7[0];
              ir7[0] = (v2131_data + (v2127_data * v266_bc));
              float v2137_data = ir7[1];
              ir7[1] = (v2137_data + (v2127_data * v272_bc));
              float v2143_data = ir7[2];
              ir7[2] = (v2143_data + (v2127_data * v278_bc));
              float v2149_data = ir7[3];
              ir7[3] = (v2149_data + (v2127_data * v284_bc));
              float v2155_data = ir7[4];
              ir7[4] = (v2155_data + (v2127_data * v290_bc));
              float v2161_data = ir7[5];
              ir7[5] = (v2161_data + (v2127_data * v296_bc));
              float v2167_data = ir7[6];
              ir7[6] = (v2167_data + (v2127_data * v302_bc));
              float v2173_data = ir7[7];
              ir7[7] = (v2173_data + (v2127_data * v308_bc));
              float v2179_data = ir7[8];
              ir7[8] = (v2179_data + (v2127_data * v314_bc));
              float v2185_data = ir7[9];
              ir7[9] = (v2185_data + (v2127_data * v320_bc));
              float v2191_data = ir7[10];
              ir7[10] = (v2191_data + (v2127_data * v326_bc));
              float v2197_data = ir7[11];
              ir7[11] = (v2197_data + (v2127_data * v332_bc));
              float v2199_data = r6[4];
              float v2203_data = ir7[0];
              ir7[0] = (v2203_data + (v2199_data * v338_bc));
              float v2209_data = ir7[1];
              ir7[1] = (v2209_data + (v2199_data * v344_bc));
              float v2215_data = ir7[2];
              ir7[2] = (v2215_data + (v2199_data * v350_bc));
              float v2221_data = ir7[3];
              ir7[3] = (v2221_data + (v2199_data * v356_bc));
              float v2227_data = ir7[4];
              ir7[4] = (v2227_data + (v2199_data * v362_bc));
              float v2233_data = ir7[5];
              ir7[5] = (v2233_data + (v2199_data * v368_bc));
              float v2239_data = ir7[6];
              ir7[6] = (v2239_data + (v2199_data * v374_bc));
              float v2245_data = ir7[7];
              ir7[7] = (v2245_data + (v2199_data * v380_bc));
              float v2251_data = ir7[8];
              ir7[8] = (v2251_data + (v2199_data * v386_bc));
              float v2257_data = ir7[9];
              ir7[9] = (v2257_data + (v2199_data * v392_bc));
              float v2263_data = ir7[10];
              ir7[10] = (v2263_data + (v2199_data * v398_bc));
              float v2269_data = ir7[11];
              ir7[11] = (v2269_data + (v2199_data * v404_bc));
              float v2271_data = r6[5];
              float v2275_data = ir7[0];
              ir7[0] = (v2275_data + (v2271_data * v410_bc));
              float v2281_data = ir7[1];
              ir7[1] = (v2281_data + (v2271_data * v416_bc));
              float v2287_data = ir7[2];
              ir7[2] = (v2287_data + (v2271_data * v422_bc));
              float v2293_data = ir7[3];
              ir7[3] = (v2293_data + (v2271_data * v428_bc));
              float v2299_data = ir7[4];
              ir7[4] = (v2299_data + (v2271_data * v434_bc));
              float v2305_data = ir7[5];
              ir7[5] = (v2305_data + (v2271_data * v440_bc));
              float v2311_data = ir7[6];
              ir7[6] = (v2311_data + (v2271_data * v446_bc));
              float v2317_data = ir7[7];
              ir7[7] = (v2317_data + (v2271_data * v452_bc));
              float v2323_data = ir7[8];
              ir7[8] = (v2323_data + (v2271_data * v458_bc));
              float v2329_data = ir7[9];
              ir7[9] = (v2329_data + (v2271_data * v464_bc));
              float v2335_data = ir7[10];
              ir7[10] = (v2335_data + (v2271_data * v470_bc));
              float v2341_data = ir7[11];
              ir7[11] = (v2341_data + (v2271_data * v476_bc));
              float v2343_data = r6[6];
              float v2347_data = ir7[0];
              ir7[0] = (v2347_data + (v2343_data * v482_bc));
              float v2353_data = ir7[1];
              ir7[1] = (v2353_data + (v2343_data * v488_bc));
              float v2359_data = ir7[2];
              ir7[2] = (v2359_data + (v2343_data * v494_bc));
              float v2365_data = ir7[3];
              ir7[3] = (v2365_data + (v2343_data * v500_bc));
              float v2371_data = ir7[4];
              ir7[4] = (v2371_data + (v2343_data * v506_bc));
              float v2377_data = ir7[5];
              ir7[5] = (v2377_data + (v2343_data * v512_bc));
              float v2383_data = ir7[6];
              ir7[6] = (v2383_data + (v2343_data * v518_bc));
              float v2389_data = ir7[7];
              ir7[7] = (v2389_data + (v2343_data * v524_bc));
              float v2395_data = ir7[8];
              ir7[8] = (v2395_data + (v2343_data * v530_bc));
              float v2401_data = ir7[9];
              ir7[9] = (v2401_data + (v2343_data * v536_bc));
              float v2407_data = ir7[10];
              ir7[10] = (v2407_data + (v2343_data * v542_bc));
              float v2413_data = ir7[11];
              ir7[11] = (v2413_data + (v2343_data * v548_bc));
              float v2415_data = r6[7];
              float v2419_data = ir7[0];
              ir7[0] = (v2419_data + (v2415_data * v554_bc));
              float v2425_data = ir7[1];
              ir7[1] = (v2425_data + (v2415_data * v560_bc));
              float v2431_data = ir7[2];
              ir7[2] = (v2431_data + (v2415_data * v566_bc));
              float v2437_data = ir7[3];
              ir7[3] = (v2437_data + (v2415_data * v572_bc));
              float v2443_data = ir7[4];
              ir7[4] = (v2443_data + (v2415_data * v578_bc));
              float v2449_data = ir7[5];
              ir7[5] = (v2449_data + (v2415_data * v584_bc));
              float v2455_data = ir7[6];
              ir7[6] = (v2455_data + (v2415_data * v590_bc));
              float v2461_data = ir7[7];
              ir7[7] = (v2461_data + (v2415_data * v596_bc));
              float v2467_data = ir7[8];
              ir7[8] = (v2467_data + (v2415_data * v602_bc));
              float v2473_data = ir7[9];
              ir7[9] = (v2473_data + (v2415_data * v608_bc));
              float v2479_data = ir7[10];
              ir7[10] = (v2479_data + (v2415_data * v614_bc));
              float v2485_data = ir7[11];
              ir7[11] = (v2485_data + (v2415_data * v620_bc));
              float v2487_data = r6[8];
              float v2491_data = ir7[0];
              ir7[0] = (v2491_data + (v2487_data * v626_bc));
              float v2497_data = ir7[1];
              ir7[1] = (v2497_data + (v2487_data * v632_bc));
              float v2503_data = ir7[2];
              ir7[2] = (v2503_data + (v2487_data * v638_bc));
              float v2509_data = ir7[3];
              ir7[3] = (v2509_data + (v2487_data * v644_bc));
              float v2515_data = ir7[4];
              ir7[4] = (v2515_data + (v2487_data * v650_bc));
              float v2521_data = ir7[5];
              ir7[5] = (v2521_data + (v2487_data * v656_bc));
              float v2527_data = ir7[6];
              ir7[6] = (v2527_data + (v2487_data * v662_bc));
              float v2533_data = ir7[7];
              ir7[7] = (v2533_data + (v2487_data * v668_bc));
              float v2539_data = ir7[8];
              ir7[8] = (v2539_data + (v2487_data * v674_bc));
              float v2545_data = ir7[9];
              ir7[9] = (v2545_data + (v2487_data * v680_bc));
              float v2551_data = ir7[10];
              ir7[10] = (v2551_data + (v2487_data * v686_bc));
              float v2557_data = ir7[11];
              ir7[11] = (v2557_data + (v2487_data * v692_bc));
              float v2559_data = r6[9];
              float v2563_data = ir7[0];
              ir7[0] = (v2563_data + (v2559_data * v698_bc));
              float v2569_data = ir7[1];
              ir7[1] = (v2569_data + (v2559_data * v704_bc));
              float v2575_data = ir7[2];
              ir7[2] = (v2575_data + (v2559_data * v710_bc));
              float v2581_data = ir7[3];
              ir7[3] = (v2581_data + (v2559_data * v716_bc));
              float v2587_data = ir7[4];
              ir7[4] = (v2587_data + (v2559_data * v722_bc));
              float v2593_data = ir7[5];
              ir7[5] = (v2593_data + (v2559_data * v728_bc));
              float v2599_data = ir7[6];
              ir7[6] = (v2599_data + (v2559_data * v734_bc));
              float v2605_data = ir7[7];
              ir7[7] = (v2605_data + (v2559_data * v740_bc));
              float v2611_data = ir7[8];
              ir7[8] = (v2611_data + (v2559_data * v746_bc));
              float v2617_data = ir7[9];
              ir7[9] = (v2617_data + (v2559_data * v752_bc));
              float v2623_data = ir7[10];
              ir7[10] = (v2623_data + (v2559_data * v758_bc));
              float v2629_data = ir7[11];
              ir7[11] = (v2629_data + (v2559_data * v764_bc));
              float v2631_data = r6[10];
              float v2635_data = ir7[0];
              ir7[0] = (v2635_data + (v2631_data * v770_bc));
              float v2641_data = ir7[1];
              ir7[1] = (v2641_data + (v2631_data * v776_bc));
              float v2647_data = ir7[2];
              ir7[2] = (v2647_data + (v2631_data * v782_bc));
              float v2653_data = ir7[3];
              ir7[3] = (v2653_data + (v2631_data * v788_bc));
              float v2659_data = ir7[4];
              ir7[4] = (v2659_data + (v2631_data * v794_bc));
              float v2665_data = ir7[5];
              ir7[5] = (v2665_data + (v2631_data * v800_bc));
              float v2671_data = ir7[6];
              ir7[6] = (v2671_data + (v2631_data * v806_bc));
              float v2677_data = ir7[7];
              ir7[7] = (v2677_data + (v2631_data * v812_bc));
              float v2683_data = ir7[8];
              ir7[8] = (v2683_data + (v2631_data * v818_bc));
              float v2689_data = ir7[9];
              ir7[9] = (v2689_data + (v2631_data * v824_bc));
              float v2695_data = ir7[10];
              ir7[10] = (v2695_data + (v2631_data * v830_bc));
              float v2701_data = ir7[11];
              ir7[11] = (v2701_data + (v2631_data * v836_bc));
              float v2703_data = r6[11];
              float v2707_data = ir7[0];
              ir7[0] = (v2707_data + (v2703_data * v842_bc));
              float v2713_data = ir7[1];
              ir7[1] = (v2713_data + (v2703_data * v848_bc));
              float v2719_data = ir7[2];
              ir7[2] = (v2719_data + (v2703_data * v854_bc));
              float v2725_data = ir7[3];
              ir7[3] = (v2725_data + (v2703_data * v860_bc));
              float v2731_data = ir7[4];
              ir7[4] = (v2731_data + (v2703_data * v866_bc));
              float v2737_data = ir7[5];
              ir7[5] = (v2737_data + (v2703_data * v872_bc));
              float v2743_data = ir7[6];
              ir7[6] = (v2743_data + (v2703_data * v878_bc));
              float v2749_data = ir7[7];
              ir7[7] = (v2749_data + (v2703_data * v884_bc));
              float v2755_data = ir7[8];
              ir7[8] = (v2755_data + (v2703_data * v890_bc));
              float v2761_data = ir7[9];
              ir7[9] = (v2761_data + (v2703_data * v896_bc));
              float v2767_data = ir7[10];
              ir7[10] = (v2767_data + (v2703_data * v902_bc));
              float v2773_data = ir7[11];
              ir7[11] = (v2773_data + (v2703_data * v908_bc));
              // r7 = ir7
              if (v923_g) {
                #pragma unroll
                for (int32_t v2775_n1 = 0; v2775_n1 < 12; ++v2775_n1) {
                  float v2777_data = ir7[v2775_n1];
                  r7[v2775_n1] = v2777_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
              // s0 = store{r>s, clear}(localShrMem0, r7);
              if ((v21_lead >= 8) && v31_g) {
                #pragma unroll
                for (int32_t v2780_z1 = 0; v2780_z1 < 12; ++v2780_z1) {
                  int32_t v2785_a = v21_lead + (v2780_z1 * 12);
                  s0[(v2785_a ^ ((v2785_a >> 4) & 15))] = 0.0f;
                }
              }
              if (v923_g) {
                int32_t v2794_off = v21_lead + 6;
                #pragma unroll
                for (int32_t v2789_i1 = 0; v2789_i1 < 12; ++v2789_i1) {
                  float v2791_data = r7[v2789_i1];
                  int32_t v2796_a = v2794_off + (v2789_i1 * 12);
                  s0[(v2796_a ^ ((v2796_a >> 4) & 15))] = v2791_data;
                }
              }
              float r8[12]{};
              sycl::group_barrier(item.get_sub_group());
              // ir8 = +(s0)
              // [(0, 12), (0, 12)] []
              float ir8[12]{};
              float v2808_data = v31_g ? (s0[(v21_lead ^ v1817_sw)]) : (0.0f);
              float v2809_data = ir8[0];
              ir8[0] = (v2809_data + v2808_data);
              float v2815_data = v31_g ? (s0[(v1822_a ^ (v1823_sw & 15))]) : (0.0f);
              float v2816_data = ir8[1];
              ir8[1] = (v2816_data + v2815_data);
              float v2822_data = v31_g ? (s0[(v1829_a ^ (v1830_sw & 15))]) : (0.0f);
              float v2823_data = ir8[2];
              ir8[2] = (v2823_data + v2822_data);
              float v2829_data = v31_g ? (s0[(v1836_a ^ (v1837_sw & 15))]) : (0.0f);
              float v2830_data = ir8[3];
              ir8[3] = (v2830_data + v2829_data);
              float v2836_data = v31_g ? (s0[(v1843_a ^ (v1844_sw & 15))]) : (0.0f);
              float v2837_data = ir8[4];
              ir8[4] = (v2837_data + v2836_data);
              float v2843_data = v31_g ? (s0[(v1850_a ^ (v1851_sw & 15))]) : (0.0f);
              float v2844_data = ir8[5];
              ir8[5] = (v2844_data + v2843_data);
              float v2850_data = v31_g ? (s0[(v1857_a ^ (v1858_sw & 15))]) : (0.0f);
              float v2851_data = ir8[6];
              ir8[6] = (v2851_data + v2850_data);
              float v2857_data = v31_g ? (s0[(v1864_a ^ (v1865_sw & 15))]) : (0.0f);
              float v2858_data = ir8[7];
              ir8[7] = (v2858_data + v2857_data);
              float v2864_data = v31_g ? (s0[(v1871_a ^ (v1872_sw & 15))]) : (0.0f);
              float v2865_data = ir8[8];
              ir8[8] = (v2865_data + v2864_data);
              float v2871_data = v31_g ? (s0[(v1878_a ^ (v1879_sw & 15))]) : (0.0f);
              float v2872_data = ir8[9];
              ir8[9] = (v2872_data + v2871_data);
              float v2878_data = v31_g ? (s0[(v1885_a ^ (v1886_sw & 15))]) : (0.0f);
              float v2879_data = ir8[10];
              ir8[10] = (v2879_data + v2878_data);
              float v2885_data = v31_g ? (s0[(v1892_a ^ (v1893_sw & 15))]) : (0.0f);
              float v2886_data = ir8[11];
              ir8[11] = (v2886_data + v2885_data);
              // r8 = ir8
              if (v31_g) {
                #pragma unroll
                for (int32_t v2888_n1 = 0; v2888_n1 < 12; ++v2888_n1) {
                  float v2890_data = ir8[v2888_n1];
                  r8[v2888_n1] = v2890_data;
                }
              }
              // glb_m5 = store{r>g}(r8);
              if (v31_g) {
                #pragma unroll
                for (int32_t v2891_i1 = 0; v2891_i1 < 12; ++v2891_i1) {
                  float v2893_data = r8[v2891_i1];
                  glb_m5[(v21_lead + (v2891_i1 * 12))] = v2893_data;
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

