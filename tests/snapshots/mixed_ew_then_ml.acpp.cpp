// === base name ===
kernel_1e36780e3cb0fc77

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_1e36780e3cb0fc77 = {{8, 2, 1}, 8, 8, 1, 2, 64, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_1e36780e3cb0fc77(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_1e36780e3cb0fc77(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_1e36780e3cb0fc77(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (8, 2, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 8;
  config.block[1] = 2;
  config.block[2] = 1;
  config.sharedMemBytes = 16 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_1e36780e3cb0fc77(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_1e36780e3cb0fc77(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_1e36780e3cb0fc77(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_1e36780e3cb0fc77(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
        //   TMP = abs(A)
        //   m1[i,j] = t0[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,2,1],"cooperative":false,"lead_width":1,"mults_per_block":2,"persistent":true,"sections":[{"barrier":false,"mults_per_block":2,"shared_elements":16}],"shared_bytes":64,"shared_elements":16,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
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
            float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 64 + 0 + m1_extraOffset];
            const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 64 + 0 + m2_extraOffset];
            float r1[8]{};
            // r1 = load{g>r}(glb_m2);
            #pragma unroll
            for (int32_t v22_i0 = 0; v22_i0 < 1; ++v22_i0) {
              int32_t v25_lead = v21_lead + (v22_i0 * 8);
              #pragma unroll
              for (int32_t v23_i1 = 0; v23_i1 < 8; ++v23_i1) {
                float v28_data = glb_m2[(v25_lead + (v23_i1 * 8))];
                r1[(v22_i0 + v23_i1)] = v28_data;
              }
            }
            float r0[8]{};
            // r0 = abs(glb_m0)
            #pragma unroll
            for (int32_t v31_k0 = 0; v31_k0 < 1; ++v31_k0) {
              int32_t v34_lead = v21_lead + (v31_k0 * 8);
              #pragma unroll
              for (int32_t v32_k1 = 0; v32_k1 < 8; ++v32_k1) {
                float v37_data = glb_m0[(v34_lead + (v32_k1 * 8))];
                r0[(v31_k0 + v32_k1)] = (sycl::fabs(v37_data));
              }
            }
            // wait(r1 = load{g>r}(glb_m2););
            float r2[8]{};
            // ir2 = +(r0 * r1)
            // [(0, 8), (0, 8)] [(0, 8)]
            float ir2[8]{};
            float v42_data = r0[0];
            float v43_data = r1[0];
            float v46_data = ir2[0];
            ir2[0] = (v46_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v49_data = r1[1];
            float v52_data = ir2[1];
            ir2[1] = (v52_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v55_data = r1[2];
            float v58_data = ir2[2];
            ir2[2] = (v58_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v61_data = r1[3];
            float v64_data = ir2[3];
            ir2[3] = (v64_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v67_data = r1[4];
            float v70_data = ir2[4];
            ir2[4] = (v70_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v73_data = r1[5];
            float v76_data = ir2[5];
            ir2[5] = (v76_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v79_data = r1[6];
            float v82_data = ir2[6];
            ir2[6] = (v82_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v85_data = r1[7];
            float v88_data = ir2[7];
            ir2[7] = (v88_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v90_data = r0[1];
            float v94_data = ir2[0];
            ir2[0] = (v94_data + (v90_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v100_data = ir2[1];
            ir2[1] = (v100_data + (v90_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v106_data = ir2[2];
            ir2[2] = (v106_data + (v90_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v112_data = ir2[3];
            ir2[3] = (v112_data + (v90_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v118_data = ir2[4];
            ir2[4] = (v118_data + (v90_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v124_data = ir2[5];
            ir2[5] = (v124_data + (v90_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v130_data = ir2[6];
            ir2[6] = (v130_data + (v90_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v136_data = ir2[7];
            ir2[7] = (v136_data + (v90_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v138_data = r0[2];
            float v142_data = ir2[0];
            ir2[0] = (v142_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v148_data = ir2[1];
            ir2[1] = (v148_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v154_data = ir2[2];
            ir2[2] = (v154_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v160_data = ir2[3];
            ir2[3] = (v160_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v166_data = ir2[4];
            ir2[4] = (v166_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v172_data = ir2[5];
            ir2[5] = (v172_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v178_data = ir2[6];
            ir2[6] = (v178_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v184_data = ir2[7];
            ir2[7] = (v184_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v186_data = r0[3];
            float v190_data = ir2[0];
            ir2[0] = (v190_data + (v186_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v196_data = ir2[1];
            ir2[1] = (v196_data + (v186_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v202_data = ir2[2];
            ir2[2] = (v202_data + (v186_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v208_data = ir2[3];
            ir2[3] = (v208_data + (v186_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v214_data = ir2[4];
            ir2[4] = (v214_data + (v186_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v220_data = ir2[5];
            ir2[5] = (v220_data + (v186_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v226_data = ir2[6];
            ir2[6] = (v226_data + (v186_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v232_data = ir2[7];
            ir2[7] = (v232_data + (v186_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v234_data = r0[4];
            float v238_data = ir2[0];
            ir2[0] = (v238_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v244_data = ir2[1];
            ir2[1] = (v244_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v250_data = ir2[2];
            ir2[2] = (v250_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v256_data = ir2[3];
            ir2[3] = (v256_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v262_data = ir2[4];
            ir2[4] = (v262_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v268_data = ir2[5];
            ir2[5] = (v268_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v274_data = ir2[6];
            ir2[6] = (v274_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v280_data = ir2[7];
            ir2[7] = (v280_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v282_data = r0[5];
            float v286_data = ir2[0];
            ir2[0] = (v286_data + (v282_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v292_data = ir2[1];
            ir2[1] = (v292_data + (v282_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v298_data = ir2[2];
            ir2[2] = (v298_data + (v282_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v304_data = ir2[3];
            ir2[3] = (v304_data + (v282_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v310_data = ir2[4];
            ir2[4] = (v310_data + (v282_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v316_data = ir2[5];
            ir2[5] = (v316_data + (v282_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v322_data = ir2[6];
            ir2[6] = (v322_data + (v282_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v328_data = ir2[7];
            ir2[7] = (v328_data + (v282_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v330_data = r0[6];
            float v334_data = ir2[0];
            ir2[0] = (v334_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v340_data = ir2[1];
            ir2[1] = (v340_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v346_data = ir2[2];
            ir2[2] = (v346_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v352_data = ir2[3];
            ir2[3] = (v352_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v358_data = ir2[4];
            ir2[4] = (v358_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v364_data = ir2[5];
            ir2[5] = (v364_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v370_data = ir2[6];
            ir2[6] = (v370_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v376_data = ir2[7];
            ir2[7] = (v376_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v378_data = r0[7];
            float v382_data = ir2[0];
            ir2[0] = (v382_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v388_data = ir2[1];
            ir2[1] = (v388_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v394_data = ir2[2];
            ir2[2] = (v394_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v400_data = ir2[3];
            ir2[3] = (v400_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v406_data = ir2[4];
            ir2[4] = (v406_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v412_data = ir2[5];
            ir2[5] = (v412_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v418_data = ir2[6];
            ir2[6] = (v418_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v424_data = ir2[7];
            ir2[7] = (v424_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            // r2 = ir2
            #pragma unroll
            for (int32_t v426_n0 = 0; v426_n0 < 1; ++v426_n0) {
              #pragma unroll
              for (int32_t v427_n1 = 0; v427_n1 < 8; ++v427_n1) {
                int32_t v428_a = v426_n0 + v427_n1;
                float v429_data = ir2[v428_a];
                r2[v428_a] = v429_data;
              }
            }
            // glb_m1 = store{r>g}(r2);
            #pragma unroll
            for (int32_t v430_i0 = 0; v430_i0 < 1; ++v430_i0) {
              #pragma unroll
              for (int32_t v431_i1 = 0; v431_i1 < 8; ++v431_i1) {
                float v433_data = r2[(v430_i0 + v431_i1)];
                if (batchIdActive0) {
                  glb_m1[((v21_lead + (v430_i0 * 8)) + (v431_i1 * 8))] = v433_data;
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

