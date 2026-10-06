// === base name ===
kernel_bba054128b0cb3e5

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_bba054128b0cb3e5 = {{8, 2, 1}, 8, 8, 1, 2, 64, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_bba054128b0cb3e5(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_bba054128b0cb3e5(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_bba054128b0cb3e5(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_bba054128b0cb3e5(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_bba054128b0cb3e5(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_bba054128b0cb3e5(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_bba054128b0cb3e5(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
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
        //   m4 8×8(8×8) {0..8}×{0..8} strided
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   t0[i,j] += m2[i,k] × m3[k,j]
        //   C = abs(TMP)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,2,1],"cooperative":false,"lead_width":1,"mults_per_block":2,"persistent":true,"sections":[{"barrier":false,"mults_per_block":2,"shared_elements":16}],"shared_bytes":64,"shared_elements":16,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A1","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"A2","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m4","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[8 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          size_t v10_batchIdLane0 = item.get_local_id(1) % 2;
          int32_t v29_lead = item.get_local_id(2) % 8;
          for (size_t v11_batchIdGroup0 = (item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)); v11_batchIdGroup0 < numElements0; v11_batchIdGroup0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v12_row = v11_batchIdGroup0 + v10_batchIdLane0;
            const bool batchIdActive0 = v12_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v12_row]));
            size_t v14_batchId0 = batchIdActive0 ? v12_row : v11_batchIdGroup0;
            size_t v15_ahead1 = v14_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v17_batchId1 = (v15_ahead1 < numElements0) ? v15_ahead1 : v14_batchId0;
            const float *const __restrict__ glb_m0 = &m0[v14_batchId0 * 64 + 0 + m0_extraOffset];
            const float *const __restrict__ glb_m1 = &m1[v14_batchId0 * 64 + 0 + m1_extraOffset];
            const float *const __restrict__ glb_m2 = &m2[v14_batchId0 * 64 + 0 + m2_extraOffset];
            const float *const __restrict__ glb_m3 = &m3[v14_batchId0 * 64 + 0 + m3_extraOffset];
            float *const __restrict__ glb_m4 = &m4[v14_batchId0 * 64 + 0 + m4_extraOffset];
            float r0[8]{};
            // r0 = load{g>r}(glb_m0);
            #pragma unroll
            for (int32_t v30_i0 = 0; v30_i0 < 1; ++v30_i0) {
              int32_t v33_lead = v29_lead + (v30_i0 * 8);
              #pragma unroll
              for (int32_t v31_i1 = 0; v31_i1 < 8; ++v31_i1) {
                float v36_data = glb_m0[(v33_lead + (v31_i1 * 8))];
                r0[(v30_i0 + v31_i1)] = v36_data;
              }
            }
            float r1[8]{};
            // r1 = load{g>r}(glb_m1);
            #pragma unroll
            for (int32_t v39_i0 = 0; v39_i0 < 1; ++v39_i0) {
              int32_t v42_lead = v29_lead + (v39_i0 * 8);
              #pragma unroll
              for (int32_t v40_i1 = 0; v40_i1 < 8; ++v40_i1) {
                float v45_data = glb_m1[(v42_lead + (v40_i1 * 8))];
                r1[(v39_i0 + v40_i1)] = v45_data;
              }
            }
            // wait(r0 = load{g>r}(glb_m0););
            float r3[8]{};
            // r3 = load{g>r}(glb_m2);
            #pragma unroll
            for (int32_t v48_i0 = 0; v48_i0 < 1; ++v48_i0) {
              int32_t v51_lead = v29_lead + (v48_i0 * 8);
              #pragma unroll
              for (int32_t v49_i1 = 0; v49_i1 < 8; ++v49_i1) {
                float v54_data = glb_m2[(v51_lead + (v49_i1 * 8))];
                r3[(v48_i0 + v49_i1)] = v54_data;
              }
            }
            // wait(r1 = load{g>r}(glb_m1););
            float r2[8]{};
            // r2 = +(r0 * r1) + None
            // [(0, 8), (0, 8)] [(0, 8)]
            float v57_data = r0[0];
            float v58_data = r1[0];
            float v61_data = r2[0];
            r2[0] = (v61_data + (v57_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v64_data = r1[1];
            float v67_data = r2[1];
            r2[1] = (v67_data + (v57_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v70_data = r1[2];
            float v73_data = r2[2];
            r2[2] = (v73_data + (v57_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v76_data = r1[3];
            float v79_data = r2[3];
            r2[3] = (v79_data + (v57_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v82_data = r1[4];
            float v85_data = r2[4];
            r2[4] = (v85_data + (v57_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v88_data = r1[5];
            float v91_data = r2[5];
            r2[5] = (v91_data + (v57_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v94_data = r1[6];
            float v97_data = r2[6];
            r2[6] = (v97_data + (v57_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v100_data = r1[7];
            float v103_data = r2[7];
            r2[7] = (v103_data + (v57_data * (sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v105_data = r0[1];
            float v109_data = r2[0];
            r2[0] = (v109_data + (v105_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v115_data = r2[1];
            r2[1] = (v115_data + (v105_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v121_data = r2[2];
            r2[2] = (v121_data + (v105_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v127_data = r2[3];
            r2[3] = (v127_data + (v105_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v133_data = r2[4];
            r2[4] = (v133_data + (v105_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v139_data = r2[5];
            r2[5] = (v139_data + (v105_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v145_data = r2[6];
            r2[6] = (v145_data + (v105_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v151_data = r2[7];
            r2[7] = (v151_data + (v105_data * (sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v153_data = r0[2];
            float v157_data = r2[0];
            r2[0] = (v157_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v163_data = r2[1];
            r2[1] = (v163_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v169_data = r2[2];
            r2[2] = (v169_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v175_data = r2[3];
            r2[3] = (v175_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v181_data = r2[4];
            r2[4] = (v181_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v187_data = r2[5];
            r2[5] = (v187_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v193_data = r2[6];
            r2[6] = (v193_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v199_data = r2[7];
            r2[7] = (v199_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v201_data = r0[3];
            float v205_data = r2[0];
            r2[0] = (v205_data + (v201_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v211_data = r2[1];
            r2[1] = (v211_data + (v201_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v217_data = r2[2];
            r2[2] = (v217_data + (v201_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v223_data = r2[3];
            r2[3] = (v223_data + (v201_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v229_data = r2[4];
            r2[4] = (v229_data + (v201_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v235_data = r2[5];
            r2[5] = (v235_data + (v201_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v241_data = r2[6];
            r2[6] = (v241_data + (v201_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v247_data = r2[7];
            r2[7] = (v247_data + (v201_data * (sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v249_data = r0[4];
            float v253_data = r2[0];
            r2[0] = (v253_data + (v249_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v259_data = r2[1];
            r2[1] = (v259_data + (v249_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v265_data = r2[2];
            r2[2] = (v265_data + (v249_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v271_data = r2[3];
            r2[3] = (v271_data + (v249_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v277_data = r2[4];
            r2[4] = (v277_data + (v249_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v283_data = r2[5];
            r2[5] = (v283_data + (v249_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v289_data = r2[6];
            r2[6] = (v289_data + (v249_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v295_data = r2[7];
            r2[7] = (v295_data + (v249_data * (sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v297_data = r0[5];
            float v301_data = r2[0];
            r2[0] = (v301_data + (v297_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v307_data = r2[1];
            r2[1] = (v307_data + (v297_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v313_data = r2[2];
            r2[2] = (v313_data + (v297_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v319_data = r2[3];
            r2[3] = (v319_data + (v297_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v325_data = r2[4];
            r2[4] = (v325_data + (v297_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v331_data = r2[5];
            r2[5] = (v331_data + (v297_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v337_data = r2[6];
            r2[6] = (v337_data + (v297_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v343_data = r2[7];
            r2[7] = (v343_data + (v297_data * (sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v345_data = r0[6];
            float v349_data = r2[0];
            r2[0] = (v349_data + (v345_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v355_data = r2[1];
            r2[1] = (v355_data + (v345_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v361_data = r2[2];
            r2[2] = (v361_data + (v345_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v367_data = r2[3];
            r2[3] = (v367_data + (v345_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v373_data = r2[4];
            r2[4] = (v373_data + (v345_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v379_data = r2[5];
            r2[5] = (v379_data + (v345_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v385_data = r2[6];
            r2[6] = (v385_data + (v345_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v391_data = r2[7];
            r2[7] = (v391_data + (v345_data * (sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v393_data = r0[7];
            float v397_data = r2[0];
            r2[0] = (v397_data + (v393_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v403_data = r2[1];
            r2[1] = (v403_data + (v393_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v409_data = r2[2];
            r2[2] = (v409_data + (v393_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v415_data = r2[3];
            r2[3] = (v415_data + (v393_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v421_data = r2[4];
            r2[4] = (v421_data + (v393_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v427_data = r2[5];
            r2[5] = (v427_data + (v393_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v433_data = r2[6];
            r2[6] = (v433_data + (v393_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v439_data = r2[7];
            r2[7] = (v439_data + (v393_data * (sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float r4[8]{};
            // r4 = load{g>r}(glb_m3);
            #pragma unroll
            for (int32_t v442_i0 = 0; v442_i0 < 1; ++v442_i0) {
              int32_t v445_lead = v29_lead + (v442_i0 * 8);
              #pragma unroll
              for (int32_t v443_i1 = 0; v443_i1 < 8; ++v443_i1) {
                float v448_data = glb_m3[(v445_lead + (v443_i1 * 8))];
                r4[(v442_i0 + v443_i1)] = v448_data;
              }
            }
            // wait(r3 = load{g>r}(glb_m2););
            // wait(r4 = load{g>r}(glb_m3););
            float r5[8]{};
            // ir5 = +(r3 * r4)
            // [(0, 8), (0, 8)] [(0, 8)]
            float ir5[8]{};
            float v452_data = r3[0];
            float v453_data = r4[0];
            float v456_data = ir5[0];
            ir5[0] = (v456_data + (v452_data * (sycl::select_from_group(item.get_sub_group(), v453_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v459_data = r4[1];
            float v462_data = ir5[1];
            ir5[1] = (v462_data + (v452_data * (sycl::select_from_group(item.get_sub_group(), v459_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v465_data = r4[2];
            float v468_data = ir5[2];
            ir5[2] = (v468_data + (v452_data * (sycl::select_from_group(item.get_sub_group(), v465_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v471_data = r4[3];
            float v474_data = ir5[3];
            ir5[3] = (v474_data + (v452_data * (sycl::select_from_group(item.get_sub_group(), v471_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v477_data = r4[4];
            float v480_data = ir5[4];
            ir5[4] = (v480_data + (v452_data * (sycl::select_from_group(item.get_sub_group(), v477_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v483_data = r4[5];
            float v486_data = ir5[5];
            ir5[5] = (v486_data + (v452_data * (sycl::select_from_group(item.get_sub_group(), v483_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v489_data = r4[6];
            float v492_data = ir5[6];
            ir5[6] = (v492_data + (v452_data * (sycl::select_from_group(item.get_sub_group(), v489_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v495_data = r4[7];
            float v498_data = ir5[7];
            ir5[7] = (v498_data + (v452_data * (sycl::select_from_group(item.get_sub_group(), v495_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v500_data = r3[1];
            float v504_data = ir5[0];
            ir5[0] = (v504_data + (v500_data * (sycl::select_from_group(item.get_sub_group(), v453_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v510_data = ir5[1];
            ir5[1] = (v510_data + (v500_data * (sycl::select_from_group(item.get_sub_group(), v459_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v516_data = ir5[2];
            ir5[2] = (v516_data + (v500_data * (sycl::select_from_group(item.get_sub_group(), v465_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v522_data = ir5[3];
            ir5[3] = (v522_data + (v500_data * (sycl::select_from_group(item.get_sub_group(), v471_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v528_data = ir5[4];
            ir5[4] = (v528_data + (v500_data * (sycl::select_from_group(item.get_sub_group(), v477_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v534_data = ir5[5];
            ir5[5] = (v534_data + (v500_data * (sycl::select_from_group(item.get_sub_group(), v483_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v540_data = ir5[6];
            ir5[6] = (v540_data + (v500_data * (sycl::select_from_group(item.get_sub_group(), v489_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v546_data = ir5[7];
            ir5[7] = (v546_data + (v500_data * (sycl::select_from_group(item.get_sub_group(), v495_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v548_data = r3[2];
            float v552_data = ir5[0];
            ir5[0] = (v552_data + (v548_data * (sycl::select_from_group(item.get_sub_group(), v453_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v558_data = ir5[1];
            ir5[1] = (v558_data + (v548_data * (sycl::select_from_group(item.get_sub_group(), v459_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v564_data = ir5[2];
            ir5[2] = (v564_data + (v548_data * (sycl::select_from_group(item.get_sub_group(), v465_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v570_data = ir5[3];
            ir5[3] = (v570_data + (v548_data * (sycl::select_from_group(item.get_sub_group(), v471_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v576_data = ir5[4];
            ir5[4] = (v576_data + (v548_data * (sycl::select_from_group(item.get_sub_group(), v477_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v582_data = ir5[5];
            ir5[5] = (v582_data + (v548_data * (sycl::select_from_group(item.get_sub_group(), v483_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v588_data = ir5[6];
            ir5[6] = (v588_data + (v548_data * (sycl::select_from_group(item.get_sub_group(), v489_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v594_data = ir5[7];
            ir5[7] = (v594_data + (v548_data * (sycl::select_from_group(item.get_sub_group(), v495_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v596_data = r3[3];
            float v600_data = ir5[0];
            ir5[0] = (v600_data + (v596_data * (sycl::select_from_group(item.get_sub_group(), v453_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v606_data = ir5[1];
            ir5[1] = (v606_data + (v596_data * (sycl::select_from_group(item.get_sub_group(), v459_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v612_data = ir5[2];
            ir5[2] = (v612_data + (v596_data * (sycl::select_from_group(item.get_sub_group(), v465_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v618_data = ir5[3];
            ir5[3] = (v618_data + (v596_data * (sycl::select_from_group(item.get_sub_group(), v471_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v624_data = ir5[4];
            ir5[4] = (v624_data + (v596_data * (sycl::select_from_group(item.get_sub_group(), v477_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v630_data = ir5[5];
            ir5[5] = (v630_data + (v596_data * (sycl::select_from_group(item.get_sub_group(), v483_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v636_data = ir5[6];
            ir5[6] = (v636_data + (v596_data * (sycl::select_from_group(item.get_sub_group(), v489_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v642_data = ir5[7];
            ir5[7] = (v642_data + (v596_data * (sycl::select_from_group(item.get_sub_group(), v495_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v644_data = r3[4];
            float v648_data = ir5[0];
            ir5[0] = (v648_data + (v644_data * (sycl::select_from_group(item.get_sub_group(), v453_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v654_data = ir5[1];
            ir5[1] = (v654_data + (v644_data * (sycl::select_from_group(item.get_sub_group(), v459_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v660_data = ir5[2];
            ir5[2] = (v660_data + (v644_data * (sycl::select_from_group(item.get_sub_group(), v465_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v666_data = ir5[3];
            ir5[3] = (v666_data + (v644_data * (sycl::select_from_group(item.get_sub_group(), v471_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v672_data = ir5[4];
            ir5[4] = (v672_data + (v644_data * (sycl::select_from_group(item.get_sub_group(), v477_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v678_data = ir5[5];
            ir5[5] = (v678_data + (v644_data * (sycl::select_from_group(item.get_sub_group(), v483_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v684_data = ir5[6];
            ir5[6] = (v684_data + (v644_data * (sycl::select_from_group(item.get_sub_group(), v489_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v690_data = ir5[7];
            ir5[7] = (v690_data + (v644_data * (sycl::select_from_group(item.get_sub_group(), v495_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v692_data = r3[5];
            float v696_data = ir5[0];
            ir5[0] = (v696_data + (v692_data * (sycl::select_from_group(item.get_sub_group(), v453_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v702_data = ir5[1];
            ir5[1] = (v702_data + (v692_data * (sycl::select_from_group(item.get_sub_group(), v459_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v708_data = ir5[2];
            ir5[2] = (v708_data + (v692_data * (sycl::select_from_group(item.get_sub_group(), v465_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v714_data = ir5[3];
            ir5[3] = (v714_data + (v692_data * (sycl::select_from_group(item.get_sub_group(), v471_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v720_data = ir5[4];
            ir5[4] = (v720_data + (v692_data * (sycl::select_from_group(item.get_sub_group(), v477_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v726_data = ir5[5];
            ir5[5] = (v726_data + (v692_data * (sycl::select_from_group(item.get_sub_group(), v483_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v732_data = ir5[6];
            ir5[6] = (v732_data + (v692_data * (sycl::select_from_group(item.get_sub_group(), v489_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v738_data = ir5[7];
            ir5[7] = (v738_data + (v692_data * (sycl::select_from_group(item.get_sub_group(), v495_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v740_data = r3[6];
            float v744_data = ir5[0];
            ir5[0] = (v744_data + (v740_data * (sycl::select_from_group(item.get_sub_group(), v453_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v750_data = ir5[1];
            ir5[1] = (v750_data + (v740_data * (sycl::select_from_group(item.get_sub_group(), v459_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v756_data = ir5[2];
            ir5[2] = (v756_data + (v740_data * (sycl::select_from_group(item.get_sub_group(), v465_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v762_data = ir5[3];
            ir5[3] = (v762_data + (v740_data * (sycl::select_from_group(item.get_sub_group(), v471_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v768_data = ir5[4];
            ir5[4] = (v768_data + (v740_data * (sycl::select_from_group(item.get_sub_group(), v477_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v774_data = ir5[5];
            ir5[5] = (v774_data + (v740_data * (sycl::select_from_group(item.get_sub_group(), v483_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v780_data = ir5[6];
            ir5[6] = (v780_data + (v740_data * (sycl::select_from_group(item.get_sub_group(), v489_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v786_data = ir5[7];
            ir5[7] = (v786_data + (v740_data * (sycl::select_from_group(item.get_sub_group(), v495_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v788_data = r3[7];
            float v792_data = ir5[0];
            ir5[0] = (v792_data + (v788_data * (sycl::select_from_group(item.get_sub_group(), v453_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v798_data = ir5[1];
            ir5[1] = (v798_data + (v788_data * (sycl::select_from_group(item.get_sub_group(), v459_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v804_data = ir5[2];
            ir5[2] = (v804_data + (v788_data * (sycl::select_from_group(item.get_sub_group(), v465_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v810_data = ir5[3];
            ir5[3] = (v810_data + (v788_data * (sycl::select_from_group(item.get_sub_group(), v471_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v816_data = ir5[4];
            ir5[4] = (v816_data + (v788_data * (sycl::select_from_group(item.get_sub_group(), v477_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v822_data = ir5[5];
            ir5[5] = (v822_data + (v788_data * (sycl::select_from_group(item.get_sub_group(), v483_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v828_data = ir5[6];
            ir5[6] = (v828_data + (v788_data * (sycl::select_from_group(item.get_sub_group(), v489_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v834_data = ir5[7];
            ir5[7] = (v834_data + (v788_data * (sycl::select_from_group(item.get_sub_group(), v495_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            // r5 = ir5 + r2
            #pragma unroll
            for (int32_t v836_n0 = 0; v836_n0 < 1; ++v836_n0) {
              #pragma unroll
              for (int32_t v837_n1 = 0; v837_n1 < 8; ++v837_n1) {
                int32_t v838_a = v836_n0 + v837_n1;
                float v839_data = ir5[v838_a];
                float v840_data = r2[v838_a];
                r5[v838_a] = (v840_data + v839_data);
              }
            }
            // glb_m4 = abs(r5)
            #pragma unroll
            for (int32_t v842_k0 = 0; v842_k0 < 1; ++v842_k0) {
              #pragma unroll
              for (int32_t v843_k1 = 0; v843_k1 < 8; ++v843_k1) {
                float v845_data = r5[(v842_k0 + v843_k1)];
                float v846_e = sycl::fabs(v845_data);
                if (batchIdActive0) {
                  glb_m4[((v29_lead + (v842_k0 * 8)) + (v843_k1 * 8))] = v846_e;
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

