// === base name ===
kernel_785a8ede7e026073

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_785a8ede7e026073 = {{8, 2, 1}, 8, 8, 1, 2, 64, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_785a8ede7e026073(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_785a8ede7e026073(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_785a8ede7e026073(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_785a8ede7e026073(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_785a8ede7e026073(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_785a8ede7e026073(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_785a8ede7e026073(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
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
        //   m3 8×8(8×8) {0..8}×{0..8} strided
        //   m4 8×8(8×8) {0..8}×{0..8} strided
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   t0[i,j] += m2[i,k] × m3[k,j]
        //   C = abs(TMP)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,2,1],"cooperative":false,"lead_width":1,"mults_per_block":2,"persistent":true,"sections":[{"barrier":false,"mults_per_block":2,"shared_elements":16}],"shared_bytes":64,"shared_elements":16,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A1","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"A2","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m4","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[8 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          size_t v3_batchIdLane0 = item.get_local_id(1) % 2;
          int32_t v23_lead = item.get_local_id(2) % 8;
          for (size_t v4_batchIdGroup0 = (item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)); v4_batchIdGroup0 < numElements0; v4_batchIdGroup0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v5_row = v4_batchIdGroup0 + v3_batchIdLane0;
            const bool batchIdActive0 = v5_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v5_row]));
            size_t v7_batchId0 = batchIdActive0 ? v5_row : v4_batchIdGroup0;
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 64 + 0 + m0_extraOffset];
            const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 64 + 0 + m1_extraOffset];
            const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 64 + 0 + m2_extraOffset];
            const float *const __restrict__ glb_m3 = &m3[v7_batchId0 * 64 + 0 + m3_extraOffset];
            float *const __restrict__ glb_m4 = &m4[v7_batchId0 * 64 + 0 + m4_extraOffset];
            float r0[8]{};
            // r0 = load{g>r}(glb_m0);
            #pragma unroll
            for (int32_t v24_i0 = 0; v24_i0 < 1; ++v24_i0) {
              int32_t v27_lead = v23_lead + (v24_i0 * 8);
              #pragma unroll
              for (int32_t v25_i1 = 0; v25_i1 < 8; ++v25_i1) {
                float v30_data = glb_m0[(v27_lead + (v25_i1 * 8))];
                r0[(v24_i0 + v25_i1)] = v30_data;
              }
            }
            float r1[8]{};
            // r1 = load{g>r}(glb_m1);
            #pragma unroll
            for (int32_t v33_i0 = 0; v33_i0 < 1; ++v33_i0) {
              int32_t v36_lead = v23_lead + (v33_i0 * 8);
              #pragma unroll
              for (int32_t v34_i1 = 0; v34_i1 < 8; ++v34_i1) {
                float v39_data = glb_m1[(v36_lead + (v34_i1 * 8))];
                r1[(v33_i0 + v34_i1)] = v39_data;
              }
            }
            // wait(r0 = load{g>r}(glb_m0););
            float r3[8]{};
            // r3 = load{g>r}(glb_m2);
            #pragma unroll
            for (int32_t v42_i0 = 0; v42_i0 < 1; ++v42_i0) {
              int32_t v45_lead = v23_lead + (v42_i0 * 8);
              #pragma unroll
              for (int32_t v43_i1 = 0; v43_i1 < 8; ++v43_i1) {
                float v48_data = glb_m2[(v45_lead + (v43_i1 * 8))];
                r3[(v42_i0 + v43_i1)] = v48_data;
              }
            }
            // wait(r1 = load{g>r}(glb_m1););
            float r2[8]{};
            // r2 = +(r0 * r1) + None
            // [(0, 8), (0, 8)] [(0, 8)]
            float v51_data = r0[0];
            float v52_data = r1[0];
            float v55_data = r2[0];
            r2[0] = (v55_data + (v51_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v58_data = r1[1];
            float v61_data = r2[1];
            r2[1] = (v61_data + (v51_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v64_data = r1[2];
            float v67_data = r2[2];
            r2[2] = (v67_data + (v51_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v70_data = r1[3];
            float v73_data = r2[3];
            r2[3] = (v73_data + (v51_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v76_data = r1[4];
            float v79_data = r2[4];
            r2[4] = (v79_data + (v51_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v82_data = r1[5];
            float v85_data = r2[5];
            r2[5] = (v85_data + (v51_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v88_data = r1[6];
            float v91_data = r2[6];
            r2[6] = (v91_data + (v51_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v94_data = r1[7];
            float v97_data = r2[7];
            r2[7] = (v97_data + (v51_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v99_data = r0[1];
            float v103_data = r2[0];
            r2[0] = (v103_data + (v99_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v109_data = r2[1];
            r2[1] = (v109_data + (v99_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v115_data = r2[2];
            r2[2] = (v115_data + (v99_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v121_data = r2[3];
            r2[3] = (v121_data + (v99_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v127_data = r2[4];
            r2[4] = (v127_data + (v99_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v133_data = r2[5];
            r2[5] = (v133_data + (v99_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v139_data = r2[6];
            r2[6] = (v139_data + (v99_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v145_data = r2[7];
            r2[7] = (v145_data + (v99_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v147_data = r0[2];
            float v151_data = r2[0];
            r2[0] = (v151_data + (v147_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v157_data = r2[1];
            r2[1] = (v157_data + (v147_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v163_data = r2[2];
            r2[2] = (v163_data + (v147_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v169_data = r2[3];
            r2[3] = (v169_data + (v147_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v175_data = r2[4];
            r2[4] = (v175_data + (v147_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v181_data = r2[5];
            r2[5] = (v181_data + (v147_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v187_data = r2[6];
            r2[6] = (v187_data + (v147_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v193_data = r2[7];
            r2[7] = (v193_data + (v147_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v195_data = r0[3];
            float v199_data = r2[0];
            r2[0] = (v199_data + (v195_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v205_data = r2[1];
            r2[1] = (v205_data + (v195_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v211_data = r2[2];
            r2[2] = (v211_data + (v195_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v217_data = r2[3];
            r2[3] = (v217_data + (v195_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v223_data = r2[4];
            r2[4] = (v223_data + (v195_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v229_data = r2[5];
            r2[5] = (v229_data + (v195_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v235_data = r2[6];
            r2[6] = (v235_data + (v195_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v241_data = r2[7];
            r2[7] = (v241_data + (v195_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v243_data = r0[4];
            float v247_data = r2[0];
            r2[0] = (v247_data + (v243_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v253_data = r2[1];
            r2[1] = (v253_data + (v243_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v259_data = r2[2];
            r2[2] = (v259_data + (v243_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v265_data = r2[3];
            r2[3] = (v265_data + (v243_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v271_data = r2[4];
            r2[4] = (v271_data + (v243_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v277_data = r2[5];
            r2[5] = (v277_data + (v243_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v283_data = r2[6];
            r2[6] = (v283_data + (v243_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v289_data = r2[7];
            r2[7] = (v289_data + (v243_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v291_data = r0[5];
            float v295_data = r2[0];
            r2[0] = (v295_data + (v291_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v301_data = r2[1];
            r2[1] = (v301_data + (v291_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v307_data = r2[2];
            r2[2] = (v307_data + (v291_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v313_data = r2[3];
            r2[3] = (v313_data + (v291_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v319_data = r2[4];
            r2[4] = (v319_data + (v291_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v325_data = r2[5];
            r2[5] = (v325_data + (v291_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v331_data = r2[6];
            r2[6] = (v331_data + (v291_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v337_data = r2[7];
            r2[7] = (v337_data + (v291_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v339_data = r0[6];
            float v343_data = r2[0];
            r2[0] = (v343_data + (v339_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v349_data = r2[1];
            r2[1] = (v349_data + (v339_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v355_data = r2[2];
            r2[2] = (v355_data + (v339_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v361_data = r2[3];
            r2[3] = (v361_data + (v339_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v367_data = r2[4];
            r2[4] = (v367_data + (v339_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v373_data = r2[5];
            r2[5] = (v373_data + (v339_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v379_data = r2[6];
            r2[6] = (v379_data + (v339_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v385_data = r2[7];
            r2[7] = (v385_data + (v339_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v387_data = r0[7];
            float v391_data = r2[0];
            r2[0] = (v391_data + (v387_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v397_data = r2[1];
            r2[1] = (v397_data + (v387_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v403_data = r2[2];
            r2[2] = (v403_data + (v387_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v409_data = r2[3];
            r2[3] = (v409_data + (v387_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v415_data = r2[4];
            r2[4] = (v415_data + (v387_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v421_data = r2[5];
            r2[5] = (v421_data + (v387_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v427_data = r2[6];
            r2[6] = (v427_data + (v387_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v433_data = r2[7];
            r2[7] = (v433_data + (v387_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float r4[8]{};
            // r4 = load{g>r}(glb_m3);
            #pragma unroll
            for (int32_t v436_i0 = 0; v436_i0 < 1; ++v436_i0) {
              int32_t v439_lead = v23_lead + (v436_i0 * 8);
              #pragma unroll
              for (int32_t v437_i1 = 0; v437_i1 < 8; ++v437_i1) {
                float v442_data = glb_m3[(v439_lead + (v437_i1 * 8))];
                r4[(v436_i0 + v437_i1)] = v442_data;
              }
            }
            // wait(r3 = load{g>r}(glb_m2););
            // wait(r4 = load{g>r}(glb_m3););
            float r5[8]{};
            // ir5 = +(r3 * r4)
            // [(0, 8), (0, 8)] [(0, 8)]
            float ir5[8]{};
            float v446_data = r3[0];
            float v447_data = r4[0];
            float v450_data = ir5[0];
            ir5[0] = (v450_data + (v446_data * (sycl::select_from_group(item.get_sub_group(), v447_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v453_data = r4[1];
            float v456_data = ir5[1];
            ir5[1] = (v456_data + (v446_data * (sycl::select_from_group(item.get_sub_group(), v453_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v459_data = r4[2];
            float v462_data = ir5[2];
            ir5[2] = (v462_data + (v446_data * (sycl::select_from_group(item.get_sub_group(), v459_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v465_data = r4[3];
            float v468_data = ir5[3];
            ir5[3] = (v468_data + (v446_data * (sycl::select_from_group(item.get_sub_group(), v465_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v471_data = r4[4];
            float v474_data = ir5[4];
            ir5[4] = (v474_data + (v446_data * (sycl::select_from_group(item.get_sub_group(), v471_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v477_data = r4[5];
            float v480_data = ir5[5];
            ir5[5] = (v480_data + (v446_data * (sycl::select_from_group(item.get_sub_group(), v477_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v483_data = r4[6];
            float v486_data = ir5[6];
            ir5[6] = (v486_data + (v446_data * (sycl::select_from_group(item.get_sub_group(), v483_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v489_data = r4[7];
            float v492_data = ir5[7];
            ir5[7] = (v492_data + (v446_data * (sycl::select_from_group(item.get_sub_group(), v489_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v494_data = r3[1];
            float v498_data = ir5[0];
            ir5[0] = (v498_data + (v494_data * (sycl::select_from_group(item.get_sub_group(), v447_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v504_data = ir5[1];
            ir5[1] = (v504_data + (v494_data * (sycl::select_from_group(item.get_sub_group(), v453_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v510_data = ir5[2];
            ir5[2] = (v510_data + (v494_data * (sycl::select_from_group(item.get_sub_group(), v459_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v516_data = ir5[3];
            ir5[3] = (v516_data + (v494_data * (sycl::select_from_group(item.get_sub_group(), v465_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v522_data = ir5[4];
            ir5[4] = (v522_data + (v494_data * (sycl::select_from_group(item.get_sub_group(), v471_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v528_data = ir5[5];
            ir5[5] = (v528_data + (v494_data * (sycl::select_from_group(item.get_sub_group(), v477_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v534_data = ir5[6];
            ir5[6] = (v534_data + (v494_data * (sycl::select_from_group(item.get_sub_group(), v483_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v540_data = ir5[7];
            ir5[7] = (v540_data + (v494_data * (sycl::select_from_group(item.get_sub_group(), v489_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v542_data = r3[2];
            float v546_data = ir5[0];
            ir5[0] = (v546_data + (v542_data * (sycl::select_from_group(item.get_sub_group(), v447_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v552_data = ir5[1];
            ir5[1] = (v552_data + (v542_data * (sycl::select_from_group(item.get_sub_group(), v453_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v558_data = ir5[2];
            ir5[2] = (v558_data + (v542_data * (sycl::select_from_group(item.get_sub_group(), v459_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v564_data = ir5[3];
            ir5[3] = (v564_data + (v542_data * (sycl::select_from_group(item.get_sub_group(), v465_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v570_data = ir5[4];
            ir5[4] = (v570_data + (v542_data * (sycl::select_from_group(item.get_sub_group(), v471_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v576_data = ir5[5];
            ir5[5] = (v576_data + (v542_data * (sycl::select_from_group(item.get_sub_group(), v477_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v582_data = ir5[6];
            ir5[6] = (v582_data + (v542_data * (sycl::select_from_group(item.get_sub_group(), v483_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v588_data = ir5[7];
            ir5[7] = (v588_data + (v542_data * (sycl::select_from_group(item.get_sub_group(), v489_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v590_data = r3[3];
            float v594_data = ir5[0];
            ir5[0] = (v594_data + (v590_data * (sycl::select_from_group(item.get_sub_group(), v447_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v600_data = ir5[1];
            ir5[1] = (v600_data + (v590_data * (sycl::select_from_group(item.get_sub_group(), v453_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v606_data = ir5[2];
            ir5[2] = (v606_data + (v590_data * (sycl::select_from_group(item.get_sub_group(), v459_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v612_data = ir5[3];
            ir5[3] = (v612_data + (v590_data * (sycl::select_from_group(item.get_sub_group(), v465_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v618_data = ir5[4];
            ir5[4] = (v618_data + (v590_data * (sycl::select_from_group(item.get_sub_group(), v471_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v624_data = ir5[5];
            ir5[5] = (v624_data + (v590_data * (sycl::select_from_group(item.get_sub_group(), v477_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v630_data = ir5[6];
            ir5[6] = (v630_data + (v590_data * (sycl::select_from_group(item.get_sub_group(), v483_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v636_data = ir5[7];
            ir5[7] = (v636_data + (v590_data * (sycl::select_from_group(item.get_sub_group(), v489_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v638_data = r3[4];
            float v642_data = ir5[0];
            ir5[0] = (v642_data + (v638_data * (sycl::select_from_group(item.get_sub_group(), v447_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v648_data = ir5[1];
            ir5[1] = (v648_data + (v638_data * (sycl::select_from_group(item.get_sub_group(), v453_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v654_data = ir5[2];
            ir5[2] = (v654_data + (v638_data * (sycl::select_from_group(item.get_sub_group(), v459_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v660_data = ir5[3];
            ir5[3] = (v660_data + (v638_data * (sycl::select_from_group(item.get_sub_group(), v465_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v666_data = ir5[4];
            ir5[4] = (v666_data + (v638_data * (sycl::select_from_group(item.get_sub_group(), v471_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v672_data = ir5[5];
            ir5[5] = (v672_data + (v638_data * (sycl::select_from_group(item.get_sub_group(), v477_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v678_data = ir5[6];
            ir5[6] = (v678_data + (v638_data * (sycl::select_from_group(item.get_sub_group(), v483_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v684_data = ir5[7];
            ir5[7] = (v684_data + (v638_data * (sycl::select_from_group(item.get_sub_group(), v489_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v686_data = r3[5];
            float v690_data = ir5[0];
            ir5[0] = (v690_data + (v686_data * (sycl::select_from_group(item.get_sub_group(), v447_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v696_data = ir5[1];
            ir5[1] = (v696_data + (v686_data * (sycl::select_from_group(item.get_sub_group(), v453_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v702_data = ir5[2];
            ir5[2] = (v702_data + (v686_data * (sycl::select_from_group(item.get_sub_group(), v459_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v708_data = ir5[3];
            ir5[3] = (v708_data + (v686_data * (sycl::select_from_group(item.get_sub_group(), v465_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v714_data = ir5[4];
            ir5[4] = (v714_data + (v686_data * (sycl::select_from_group(item.get_sub_group(), v471_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v720_data = ir5[5];
            ir5[5] = (v720_data + (v686_data * (sycl::select_from_group(item.get_sub_group(), v477_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v726_data = ir5[6];
            ir5[6] = (v726_data + (v686_data * (sycl::select_from_group(item.get_sub_group(), v483_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v732_data = ir5[7];
            ir5[7] = (v732_data + (v686_data * (sycl::select_from_group(item.get_sub_group(), v489_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v734_data = r3[6];
            float v738_data = ir5[0];
            ir5[0] = (v738_data + (v734_data * (sycl::select_from_group(item.get_sub_group(), v447_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v744_data = ir5[1];
            ir5[1] = (v744_data + (v734_data * (sycl::select_from_group(item.get_sub_group(), v453_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v750_data = ir5[2];
            ir5[2] = (v750_data + (v734_data * (sycl::select_from_group(item.get_sub_group(), v459_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v756_data = ir5[3];
            ir5[3] = (v756_data + (v734_data * (sycl::select_from_group(item.get_sub_group(), v465_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v762_data = ir5[4];
            ir5[4] = (v762_data + (v734_data * (sycl::select_from_group(item.get_sub_group(), v471_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v768_data = ir5[5];
            ir5[5] = (v768_data + (v734_data * (sycl::select_from_group(item.get_sub_group(), v477_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v774_data = ir5[6];
            ir5[6] = (v774_data + (v734_data * (sycl::select_from_group(item.get_sub_group(), v483_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v780_data = ir5[7];
            ir5[7] = (v780_data + (v734_data * (sycl::select_from_group(item.get_sub_group(), v489_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v782_data = r3[7];
            float v786_data = ir5[0];
            ir5[0] = (v786_data + (v782_data * (sycl::select_from_group(item.get_sub_group(), v447_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v792_data = ir5[1];
            ir5[1] = (v792_data + (v782_data * (sycl::select_from_group(item.get_sub_group(), v453_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v798_data = ir5[2];
            ir5[2] = (v798_data + (v782_data * (sycl::select_from_group(item.get_sub_group(), v459_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v804_data = ir5[3];
            ir5[3] = (v804_data + (v782_data * (sycl::select_from_group(item.get_sub_group(), v465_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v810_data = ir5[4];
            ir5[4] = (v810_data + (v782_data * (sycl::select_from_group(item.get_sub_group(), v471_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v816_data = ir5[5];
            ir5[5] = (v816_data + (v782_data * (sycl::select_from_group(item.get_sub_group(), v477_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v822_data = ir5[6];
            ir5[6] = (v822_data + (v782_data * (sycl::select_from_group(item.get_sub_group(), v483_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v828_data = ir5[7];
            ir5[7] = (v828_data + (v782_data * (sycl::select_from_group(item.get_sub_group(), v489_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            // r5 = ir5 + r2
            #pragma unroll
            for (int32_t v830_n0 = 0; v830_n0 < 1; ++v830_n0) {
              #pragma unroll
              for (int32_t v831_n1 = 0; v831_n1 < 8; ++v831_n1) {
                int32_t v832_a = v830_n0 + v831_n1;
                float v833_data = ir5[v832_a];
                float v834_data = r2[v832_a];
                r5[v832_a] = (v834_data + v833_data);
              }
            }
            // glb_m4 = abs(r5)
            #pragma unroll
            for (int32_t v836_k0 = 0; v836_k0 < 1; ++v836_k0) {
              #pragma unroll
              for (int32_t v837_k1 = 0; v837_k1 < 8; ++v837_k1) {
                float v839_data = r5[(v836_k0 + v837_k1)];
                float v840_e = sycl::fabs(v839_data);
                if (batchIdActive0) {
                  glb_m4[((v23_lead + (v836_k0 * 8)) + (v837_k1 * 8))] = v840_e;
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

