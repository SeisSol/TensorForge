// === base name ===
kernel_10fe4c1bd7d8e118

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_10fe4c1bd7d8e118 = {{16, 16, 1}, 16, 12, 1, 16, 10240, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_10fe4c1bd7d8e118(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_10fe4c1bd7d8e118(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_10fe4c1bd7d8e118(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_10fe4c1bd7d8e118(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_10fe4c1bd7d8e118(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_10fe4c1bd7d8e118(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_10fe4c1bd7d8e118(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (2560, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 10240 B shared, occupancy grid
        // operands:
        //   m0 32×32(12×12) {0..12}×{0..12} strided
        //   m1 32×32(12×12) {0..12}×{0..12} strided
        //   m2 32×32(12×12) {0..12}×{0..12} strided
        //   m3 32×32(12×12) {0..12}×{0..12} strided
        // operations:
        //   t0[i,j]@{0..12}×{0..6} = m0[i,k] × m1[k,j]
        //   m2[i,j] = m3[i,k] × t0[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2560}],"shared_bytes":10240,"shared_elements":2560,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B","bbox":[[0,0],[12,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,6]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]},{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
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
              const float *const __restrict__ glb_m0 = &m0[v4_batchId0 * 144 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v4_batchId0 * 144 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v4_batchId0 * 144 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v4_batchId0 * 144 + 0 + m3_extraOffset];
              float r0[12]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v19_lead = item.get_local_id(2) % 16;
              bool v20_g = v19_lead < 12;
              if (v20_g) {
                #pragma unroll
                for (int32_t v21_i1 = 0; v21_i1 < 12; ++v21_i1) {
                  float v26_data = glb_m0[(v19_lead + (v21_i1 * 12))];
                  r0[v21_i1] = v26_data;
                }
              }
              float r1[12]{};
              // r1 = load{g>r}(glb_m1);
              if (v20_g) {
                #pragma unroll
                for (int32_t v29_i1 = 0; v29_i1 < 12; ++v29_i1) {
                  float v34_data = glb_m1[(v19_lead + (v29_i1 * 12))];
                  r1[v29_i1] = v34_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r3[12]{};
              // r3 = load{g>r}(glb_m3);
              if (v20_g) {
                #pragma unroll
                for (int32_t v37_i1 = 0; v37_i1 < 12; ++v37_i1) {
                  float v42_data = glb_m3[(v19_lead + (v37_i1 * 12))];
                  r3[v37_i1] = v42_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m1););
              float r2[6]{};
              // r2 = +(r0 * r1) + None
              // [(0, 12), (0, 6)] [(0, 12)]
              float v45_data = r0[0];
              float v46_data = r1[0];
              float v49_data = r2[0];
              r2[0] = (v49_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v52_data = r1[1];
              float v55_data = r2[1];
              r2[1] = (v55_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v58_data = r1[2];
              float v61_data = r2[2];
              r2[2] = (v61_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v64_data = r1[3];
              float v67_data = r2[3];
              r2[3] = (v67_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v70_data = r1[4];
              float v73_data = r2[4];
              r2[4] = (v73_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v76_data = r1[5];
              float v79_data = r2[5];
              r2[5] = (v79_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v81_data = r0[1];
              float v85_data = r2[0];
              r2[0] = (v85_data + (v81_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v91_data = r2[1];
              r2[1] = (v91_data + (v81_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v97_data = r2[2];
              r2[2] = (v97_data + (v81_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v103_data = r2[3];
              r2[3] = (v103_data + (v81_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v109_data = r2[4];
              r2[4] = (v109_data + (v81_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v115_data = r2[5];
              r2[5] = (v115_data + (v81_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v117_data = r0[2];
              float v121_data = r2[0];
              r2[0] = (v121_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v127_data = r2[1];
              r2[1] = (v127_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v133_data = r2[2];
              r2[2] = (v133_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v139_data = r2[3];
              r2[3] = (v139_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v145_data = r2[4];
              r2[4] = (v145_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v151_data = r2[5];
              r2[5] = (v151_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v153_data = r0[3];
              float v157_data = r2[0];
              r2[0] = (v157_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v163_data = r2[1];
              r2[1] = (v163_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v169_data = r2[2];
              r2[2] = (v169_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v175_data = r2[3];
              r2[3] = (v175_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v181_data = r2[4];
              r2[4] = (v181_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v187_data = r2[5];
              r2[5] = (v187_data + (v153_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v189_data = r0[4];
              float v193_data = r2[0];
              r2[0] = (v193_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v199_data = r2[1];
              r2[1] = (v199_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v205_data = r2[2];
              r2[2] = (v205_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v211_data = r2[3];
              r2[3] = (v211_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v217_data = r2[4];
              r2[4] = (v217_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v223_data = r2[5];
              r2[5] = (v223_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v225_data = r0[5];
              float v229_data = r2[0];
              r2[0] = (v229_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v235_data = r2[1];
              r2[1] = (v235_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v241_data = r2[2];
              r2[2] = (v241_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v247_data = r2[3];
              r2[3] = (v247_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v253_data = r2[4];
              r2[4] = (v253_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v259_data = r2[5];
              r2[5] = (v259_data + (v225_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v261_data = r0[6];
              float v265_data = r2[0];
              r2[0] = (v265_data + (v261_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v271_data = r2[1];
              r2[1] = (v271_data + (v261_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v277_data = r2[2];
              r2[2] = (v277_data + (v261_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v283_data = r2[3];
              r2[3] = (v283_data + (v261_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v289_data = r2[4];
              r2[4] = (v289_data + (v261_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v295_data = r2[5];
              r2[5] = (v295_data + (v261_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v297_data = r0[7];
              float v301_data = r2[0];
              r2[0] = (v301_data + (v297_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v307_data = r2[1];
              r2[1] = (v307_data + (v297_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v313_data = r2[2];
              r2[2] = (v313_data + (v297_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v319_data = r2[3];
              r2[3] = (v319_data + (v297_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v325_data = r2[4];
              r2[4] = (v325_data + (v297_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v331_data = r2[5];
              r2[5] = (v331_data + (v297_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v333_data = r0[8];
              float v337_data = r2[0];
              r2[0] = (v337_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v343_data = r2[1];
              r2[1] = (v343_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v349_data = r2[2];
              r2[2] = (v349_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v355_data = r2[3];
              r2[3] = (v355_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v361_data = r2[4];
              r2[4] = (v361_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v367_data = r2[5];
              r2[5] = (v367_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v369_data = r0[9];
              float v373_data = r2[0];
              r2[0] = (v373_data + (v369_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v379_data = r2[1];
              r2[1] = (v379_data + (v369_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v385_data = r2[2];
              r2[2] = (v385_data + (v369_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v391_data = r2[3];
              r2[3] = (v391_data + (v369_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v397_data = r2[4];
              r2[4] = (v397_data + (v369_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v403_data = r2[5];
              r2[5] = (v403_data + (v369_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v405_data = r0[10];
              float v409_data = r2[0];
              r2[0] = (v409_data + (v405_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v415_data = r2[1];
              r2[1] = (v415_data + (v405_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v421_data = r2[2];
              r2[2] = (v421_data + (v405_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v427_data = r2[3];
              r2[3] = (v427_data + (v405_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v433_data = r2[4];
              r2[4] = (v433_data + (v405_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v439_data = r2[5];
              r2[5] = (v439_data + (v405_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v441_data = r0[11];
              float v445_data = r2[0];
              r2[0] = (v445_data + (v441_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v451_data = r2[1];
              r2[1] = (v451_data + (v441_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v457_data = r2[2];
              r2[2] = (v457_data + (v441_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v463_data = r2[3];
              r2[3] = (v463_data + (v441_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v469_data = r2[4];
              r2[4] = (v469_data + (v441_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v475_data = r2[5];
              r2[5] = (v475_data + (v441_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              // s0 = store{r>s, clear}(localShrMem0, r2);
              if (v20_g) {
                #pragma unroll
                for (int32_t v477_z1 = 6; v477_z1 < 12; ++v477_z1) {
                  int32_t v482_a = v19_lead + (v477_z1 * 12);
                  s0[(v482_a ^ ((v482_a >> 4) & 15))] = 0.0f;
                }
              }
              if (v20_g) {
                #pragma unroll
                for (int32_t v486_i1 = 0; v486_i1 < 6; ++v486_i1) {
                  float v488_data = r2[v486_i1];
                  int32_t v492_a = v19_lead + (v486_i1 * 12);
                  s0[(v492_a ^ ((v492_a >> 4) & 15))] = v488_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m3););
              float r4[12]{};
              sycl::group_barrier(item.get_sub_group());
              // ir4 = +(r3 * s0)
              // [(0, 12), (0, 12)] [(0, 12)]
              float ir4[12]{};
              float v498_data = r3[0];
              float v499_data = s0[0];
              float v501_data = ir4[0];
              ir4[0] = (v501_data + (v498_data * v499_data));
              float v504_data = s0[12];
              float v506_data = ir4[1];
              ir4[1] = (v506_data + (v498_data * v504_data));
              float v509_data = s0[25];
              float v511_data = ir4[2];
              ir4[2] = (v511_data + (v498_data * v509_data));
              float v514_data = s0[38];
              float v516_data = ir4[3];
              ir4[3] = (v516_data + (v498_data * v514_data));
              float v519_data = s0[51];
              float v521_data = ir4[4];
              ir4[4] = (v521_data + (v498_data * v519_data));
              float v524_data = s0[63];
              float v526_data = ir4[5];
              ir4[5] = (v526_data + (v498_data * v524_data));
              float v529_data = s0[76];
              float v531_data = ir4[6];
              ir4[6] = (v531_data + (v498_data * v529_data));
              float v534_data = s0[81];
              float v536_data = ir4[7];
              ir4[7] = (v536_data + (v498_data * v534_data));
              float v539_data = s0[102];
              float v541_data = ir4[8];
              ir4[8] = (v541_data + (v498_data * v539_data));
              float v544_data = s0[106];
              float v546_data = ir4[9];
              ir4[9] = (v546_data + (v498_data * v544_data));
              float v549_data = s0[127];
              float v551_data = ir4[10];
              ir4[10] = (v551_data + (v498_data * v549_data));
              float v554_data = s0[140];
              float v556_data = ir4[11];
              ir4[11] = (v556_data + (v498_data * v554_data));
              float v558_data = r3[1];
              float v559_data = s0[1];
              float v561_data = ir4[0];
              ir4[0] = (v561_data + (v558_data * v559_data));
              float v564_data = s0[13];
              float v566_data = ir4[1];
              ir4[1] = (v566_data + (v558_data * v564_data));
              float v569_data = s0[24];
              float v571_data = ir4[2];
              ir4[2] = (v571_data + (v558_data * v569_data));
              float v574_data = s0[39];
              float v576_data = ir4[3];
              ir4[3] = (v576_data + (v558_data * v574_data));
              float v579_data = s0[50];
              float v581_data = ir4[4];
              ir4[4] = (v581_data + (v558_data * v579_data));
              float v584_data = s0[62];
              float v586_data = ir4[5];
              ir4[5] = (v586_data + (v558_data * v584_data));
              float v589_data = s0[77];
              float v591_data = ir4[6];
              ir4[6] = (v591_data + (v558_data * v589_data));
              float v594_data = s0[80];
              float v596_data = ir4[7];
              ir4[7] = (v596_data + (v558_data * v594_data));
              float v599_data = s0[103];
              float v601_data = ir4[8];
              ir4[8] = (v601_data + (v558_data * v599_data));
              float v604_data = s0[107];
              float v606_data = ir4[9];
              ir4[9] = (v606_data + (v558_data * v604_data));
              float v609_data = s0[126];
              float v611_data = ir4[10];
              ir4[10] = (v611_data + (v558_data * v609_data));
              float v614_data = s0[141];
              float v616_data = ir4[11];
              ir4[11] = (v616_data + (v558_data * v614_data));
              float v618_data = r3[2];
              float v619_data = s0[2];
              float v621_data = ir4[0];
              ir4[0] = (v621_data + (v618_data * v619_data));
              float v624_data = s0[14];
              float v626_data = ir4[1];
              ir4[1] = (v626_data + (v618_data * v624_data));
              float v629_data = s0[27];
              float v631_data = ir4[2];
              ir4[2] = (v631_data + (v618_data * v629_data));
              float v634_data = s0[36];
              float v636_data = ir4[3];
              ir4[3] = (v636_data + (v618_data * v634_data));
              float v639_data = s0[49];
              float v641_data = ir4[4];
              ir4[4] = (v641_data + (v618_data * v639_data));
              float v644_data = s0[61];
              float v646_data = ir4[5];
              ir4[5] = (v646_data + (v618_data * v644_data));
              float v649_data = s0[78];
              float v651_data = ir4[6];
              ir4[6] = (v651_data + (v618_data * v649_data));
              float v654_data = s0[83];
              float v656_data = ir4[7];
              ir4[7] = (v656_data + (v618_data * v654_data));
              float v659_data = s0[100];
              float v661_data = ir4[8];
              ir4[8] = (v661_data + (v618_data * v659_data));
              float v664_data = s0[104];
              float v666_data = ir4[9];
              ir4[9] = (v666_data + (v618_data * v664_data));
              float v669_data = s0[125];
              float v671_data = ir4[10];
              ir4[10] = (v671_data + (v618_data * v669_data));
              float v674_data = s0[142];
              float v676_data = ir4[11];
              ir4[11] = (v676_data + (v618_data * v674_data));
              float v678_data = r3[3];
              float v679_data = s0[3];
              float v681_data = ir4[0];
              ir4[0] = (v681_data + (v678_data * v679_data));
              float v684_data = s0[15];
              float v686_data = ir4[1];
              ir4[1] = (v686_data + (v678_data * v684_data));
              float v689_data = s0[26];
              float v691_data = ir4[2];
              ir4[2] = (v691_data + (v678_data * v689_data));
              float v694_data = s0[37];
              float v696_data = ir4[3];
              ir4[3] = (v696_data + (v678_data * v694_data));
              float v699_data = s0[48];
              float v701_data = ir4[4];
              ir4[4] = (v701_data + (v678_data * v699_data));
              float v704_data = s0[60];
              float v706_data = ir4[5];
              ir4[5] = (v706_data + (v678_data * v704_data));
              float v709_data = s0[79];
              float v711_data = ir4[6];
              ir4[6] = (v711_data + (v678_data * v709_data));
              float v714_data = s0[82];
              float v716_data = ir4[7];
              ir4[7] = (v716_data + (v678_data * v714_data));
              float v719_data = s0[101];
              float v721_data = ir4[8];
              ir4[8] = (v721_data + (v678_data * v719_data));
              float v724_data = s0[105];
              float v726_data = ir4[9];
              ir4[9] = (v726_data + (v678_data * v724_data));
              float v729_data = s0[124];
              float v731_data = ir4[10];
              ir4[10] = (v731_data + (v678_data * v729_data));
              float v734_data = s0[143];
              float v736_data = ir4[11];
              ir4[11] = (v736_data + (v678_data * v734_data));
              float v738_data = r3[4];
              float v739_data = s0[4];
              float v741_data = ir4[0];
              ir4[0] = (v741_data + (v738_data * v739_data));
              float v744_data = s0[17];
              float v746_data = ir4[1];
              ir4[1] = (v746_data + (v738_data * v744_data));
              float v749_data = s0[29];
              float v751_data = ir4[2];
              ir4[2] = (v751_data + (v738_data * v749_data));
              float v754_data = s0[42];
              float v756_data = ir4[3];
              ir4[3] = (v756_data + (v738_data * v754_data));
              float v759_data = s0[55];
              float v761_data = ir4[4];
              ir4[4] = (v761_data + (v738_data * v759_data));
              float v764_data = s0[68];
              float v766_data = ir4[5];
              ir4[5] = (v766_data + (v738_data * v764_data));
              float v769_data = s0[72];
              float v771_data = ir4[6];
              ir4[6] = (v771_data + (v738_data * v769_data));
              float v774_data = s0[93];
              float v776_data = ir4[7];
              ir4[7] = (v776_data + (v738_data * v774_data));
              float v779_data = s0[98];
              float v781_data = ir4[8];
              ir4[8] = (v781_data + (v738_data * v779_data));
              float v784_data = s0[119];
              float v786_data = ir4[9];
              ir4[9] = (v786_data + (v738_data * v784_data));
              float v789_data = s0[123];
              float v791_data = ir4[10];
              ir4[10] = (v791_data + (v738_data * v789_data));
              float v794_data = s0[128];
              float v796_data = ir4[11];
              ir4[11] = (v796_data + (v738_data * v794_data));
              float v798_data = r3[5];
              float v799_data = s0[5];
              float v801_data = ir4[0];
              ir4[0] = (v801_data + (v798_data * v799_data));
              float v804_data = s0[16];
              float v806_data = ir4[1];
              ir4[1] = (v806_data + (v798_data * v804_data));
              float v809_data = s0[28];
              float v811_data = ir4[2];
              ir4[2] = (v811_data + (v798_data * v809_data));
              float v814_data = s0[43];
              float v816_data = ir4[3];
              ir4[3] = (v816_data + (v798_data * v814_data));
              float v819_data = s0[54];
              float v821_data = ir4[4];
              ir4[4] = (v821_data + (v798_data * v819_data));
              float v824_data = s0[69];
              float v826_data = ir4[5];
              ir4[5] = (v826_data + (v798_data * v824_data));
              float v829_data = s0[73];
              float v831_data = ir4[6];
              ir4[6] = (v831_data + (v798_data * v829_data));
              float v834_data = s0[92];
              float v836_data = ir4[7];
              ir4[7] = (v836_data + (v798_data * v834_data));
              float v839_data = s0[99];
              float v841_data = ir4[8];
              ir4[8] = (v841_data + (v798_data * v839_data));
              float v844_data = s0[118];
              float v846_data = ir4[9];
              ir4[9] = (v846_data + (v798_data * v844_data));
              float v849_data = s0[122];
              float v851_data = ir4[10];
              ir4[10] = (v851_data + (v798_data * v849_data));
              float v854_data = s0[129];
              float v856_data = ir4[11];
              ir4[11] = (v856_data + (v798_data * v854_data));
              float v858_data = r3[6];
              float v859_data = s0[6];
              float v861_data = ir4[0];
              ir4[0] = (v861_data + (v858_data * v859_data));
              float v864_data = s0[19];
              float v866_data = ir4[1];
              ir4[1] = (v866_data + (v858_data * v864_data));
              float v869_data = s0[31];
              float v871_data = ir4[2];
              ir4[2] = (v871_data + (v858_data * v869_data));
              float v874_data = s0[40];
              float v876_data = ir4[3];
              ir4[3] = (v876_data + (v858_data * v874_data));
              float v879_data = s0[53];
              float v881_data = ir4[4];
              ir4[4] = (v881_data + (v858_data * v879_data));
              float v884_data = s0[70];
              float v886_data = ir4[5];
              ir4[5] = (v886_data + (v858_data * v884_data));
              float v889_data = s0[74];
              float v891_data = ir4[6];
              ir4[6] = (v891_data + (v858_data * v889_data));
              float v894_data = s0[95];
              float v896_data = ir4[7];
              ir4[7] = (v896_data + (v858_data * v894_data));
              float v899_data = s0[96];
              float v901_data = ir4[8];
              ir4[8] = (v901_data + (v858_data * v899_data));
              float v904_data = s0[117];
              float v906_data = ir4[9];
              ir4[9] = (v906_data + (v858_data * v904_data));
              float v909_data = s0[121];
              float v911_data = ir4[10];
              ir4[10] = (v911_data + (v858_data * v909_data));
              float v914_data = s0[130];
              float v916_data = ir4[11];
              ir4[11] = (v916_data + (v858_data * v914_data));
              float v918_data = r3[7];
              float v919_data = s0[7];
              float v921_data = ir4[0];
              ir4[0] = (v921_data + (v918_data * v919_data));
              float v924_data = s0[18];
              float v926_data = ir4[1];
              ir4[1] = (v926_data + (v918_data * v924_data));
              float v929_data = s0[30];
              float v931_data = ir4[2];
              ir4[2] = (v931_data + (v918_data * v929_data));
              float v934_data = s0[41];
              float v936_data = ir4[3];
              ir4[3] = (v936_data + (v918_data * v934_data));
              float v939_data = s0[52];
              float v941_data = ir4[4];
              ir4[4] = (v941_data + (v918_data * v939_data));
              float v944_data = s0[71];
              float v946_data = ir4[5];
              ir4[5] = (v946_data + (v918_data * v944_data));
              float v949_data = s0[75];
              float v951_data = ir4[6];
              ir4[6] = (v951_data + (v918_data * v949_data));
              float v954_data = s0[94];
              float v956_data = ir4[7];
              ir4[7] = (v956_data + (v918_data * v954_data));
              float v959_data = s0[97];
              float v961_data = ir4[8];
              ir4[8] = (v961_data + (v918_data * v959_data));
              float v964_data = s0[116];
              float v966_data = ir4[9];
              ir4[9] = (v966_data + (v918_data * v964_data));
              float v969_data = s0[120];
              float v971_data = ir4[10];
              ir4[10] = (v971_data + (v918_data * v969_data));
              float v974_data = s0[131];
              float v976_data = ir4[11];
              ir4[11] = (v976_data + (v918_data * v974_data));
              float v978_data = r3[8];
              float v979_data = s0[8];
              float v981_data = ir4[0];
              ir4[0] = (v981_data + (v978_data * v979_data));
              float v984_data = s0[21];
              float v986_data = ir4[1];
              ir4[1] = (v986_data + (v978_data * v984_data));
              float v989_data = s0[34];
              float v991_data = ir4[2];
              ir4[2] = (v991_data + (v978_data * v989_data));
              float v994_data = s0[46];
              float v996_data = ir4[3];
              ir4[3] = (v996_data + (v978_data * v994_data));
              float v999_data = s0[59];
              float v1001_data = ir4[4];
              ir4[4] = (v1001_data + (v978_data * v999_data));
              float v1004_data = s0[64];
              float v1006_data = ir4[5];
              ir4[5] = (v1006_data + (v978_data * v1004_data));
              float v1009_data = s0[85];
              float v1011_data = ir4[6];
              ir4[6] = (v1011_data + (v978_data * v1009_data));
              float v1014_data = s0[89];
              float v1016_data = ir4[7];
              ir4[7] = (v1016_data + (v978_data * v1014_data));
              float v1019_data = s0[110];
              float v1021_data = ir4[8];
              ir4[8] = (v1021_data + (v978_data * v1019_data));
              float v1024_data = s0[115];
              float v1026_data = ir4[9];
              ir4[9] = (v1026_data + (v978_data * v1024_data));
              float v1029_data = s0[136];
              float v1031_data = ir4[10];
              ir4[10] = (v1031_data + (v978_data * v1029_data));
              float v1034_data = s0[132];
              float v1036_data = ir4[11];
              ir4[11] = (v1036_data + (v978_data * v1034_data));
              float v1038_data = r3[9];
              float v1039_data = s0[9];
              float v1041_data = ir4[0];
              ir4[0] = (v1041_data + (v1038_data * v1039_data));
              float v1044_data = s0[20];
              float v1046_data = ir4[1];
              ir4[1] = (v1046_data + (v1038_data * v1044_data));
              float v1049_data = s0[35];
              float v1051_data = ir4[2];
              ir4[2] = (v1051_data + (v1038_data * v1049_data));
              float v1054_data = s0[47];
              float v1056_data = ir4[3];
              ir4[3] = (v1056_data + (v1038_data * v1054_data));
              float v1059_data = s0[58];
              float v1061_data = ir4[4];
              ir4[4] = (v1061_data + (v1038_data * v1059_data));
              float v1064_data = s0[65];
              float v1066_data = ir4[5];
              ir4[5] = (v1066_data + (v1038_data * v1064_data));
              float v1069_data = s0[84];
              float v1071_data = ir4[6];
              ir4[6] = (v1071_data + (v1038_data * v1069_data));
              float v1074_data = s0[88];
              float v1076_data = ir4[7];
              ir4[7] = (v1076_data + (v1038_data * v1074_data));
              float v1079_data = s0[111];
              float v1081_data = ir4[8];
              ir4[8] = (v1081_data + (v1038_data * v1079_data));
              float v1084_data = s0[114];
              float v1086_data = ir4[9];
              ir4[9] = (v1086_data + (v1038_data * v1084_data));
              float v1089_data = s0[137];
              float v1091_data = ir4[10];
              ir4[10] = (v1091_data + (v1038_data * v1089_data));
              float v1094_data = s0[133];
              float v1096_data = ir4[11];
              ir4[11] = (v1096_data + (v1038_data * v1094_data));
              float v1098_data = r3[10];
              float v1099_data = s0[10];
              float v1101_data = ir4[0];
              ir4[0] = (v1101_data + (v1098_data * v1099_data));
              float v1104_data = s0[23];
              float v1106_data = ir4[1];
              ir4[1] = (v1106_data + (v1098_data * v1104_data));
              float v1109_data = s0[32];
              float v1111_data = ir4[2];
              ir4[2] = (v1111_data + (v1098_data * v1109_data));
              float v1114_data = s0[44];
              float v1116_data = ir4[3];
              ir4[3] = (v1116_data + (v1098_data * v1114_data));
              float v1119_data = s0[57];
              float v1121_data = ir4[4];
              ir4[4] = (v1121_data + (v1098_data * v1119_data));
              float v1124_data = s0[66];
              float v1126_data = ir4[5];
              ir4[5] = (v1126_data + (v1098_data * v1124_data));
              float v1129_data = s0[87];
              float v1131_data = ir4[6];
              ir4[6] = (v1131_data + (v1098_data * v1129_data));
              float v1134_data = s0[91];
              float v1136_data = ir4[7];
              ir4[7] = (v1136_data + (v1098_data * v1134_data));
              float v1139_data = s0[108];
              float v1141_data = ir4[8];
              ir4[8] = (v1141_data + (v1098_data * v1139_data));
              float v1144_data = s0[113];
              float v1146_data = ir4[9];
              ir4[9] = (v1146_data + (v1098_data * v1144_data));
              float v1149_data = s0[138];
              float v1151_data = ir4[10];
              ir4[10] = (v1151_data + (v1098_data * v1149_data));
              float v1154_data = s0[134];
              float v1156_data = ir4[11];
              ir4[11] = (v1156_data + (v1098_data * v1154_data));
              float v1158_data = r3[11];
              float v1159_data = s0[11];
              float v1161_data = ir4[0];
              ir4[0] = (v1161_data + (v1158_data * v1159_data));
              float v1164_data = s0[22];
              float v1166_data = ir4[1];
              ir4[1] = (v1166_data + (v1158_data * v1164_data));
              float v1169_data = s0[33];
              float v1171_data = ir4[2];
              ir4[2] = (v1171_data + (v1158_data * v1169_data));
              float v1174_data = s0[45];
              float v1176_data = ir4[3];
              ir4[3] = (v1176_data + (v1158_data * v1174_data));
              float v1179_data = s0[56];
              float v1181_data = ir4[4];
              ir4[4] = (v1181_data + (v1158_data * v1179_data));
              float v1184_data = s0[67];
              float v1186_data = ir4[5];
              ir4[5] = (v1186_data + (v1158_data * v1184_data));
              float v1189_data = s0[86];
              float v1191_data = ir4[6];
              ir4[6] = (v1191_data + (v1158_data * v1189_data));
              float v1194_data = s0[90];
              float v1196_data = ir4[7];
              ir4[7] = (v1196_data + (v1158_data * v1194_data));
              float v1199_data = s0[109];
              float v1201_data = ir4[8];
              ir4[8] = (v1201_data + (v1158_data * v1199_data));
              float v1204_data = s0[112];
              float v1206_data = ir4[9];
              ir4[9] = (v1206_data + (v1158_data * v1204_data));
              float v1209_data = s0[139];
              float v1211_data = ir4[10];
              ir4[10] = (v1211_data + (v1158_data * v1209_data));
              float v1214_data = s0[135];
              float v1216_data = ir4[11];
              ir4[11] = (v1216_data + (v1158_data * v1214_data));
              // r4 = ir4
              if (v20_g) {
                #pragma unroll
                for (int32_t v1218_n1 = 0; v1218_n1 < 12; ++v1218_n1) {
                  float v1220_data = ir4[v1218_n1];
                  r4[v1218_n1] = v1220_data;
                }
              }
              // glb_m2 = store{r>g}(r4);
              if (v20_g) {
                #pragma unroll
                for (int32_t v1221_i1 = 0; v1221_i1 < 12; ++v1221_i1) {
                  float v1223_data = r4[v1221_i1];
                  glb_m2[(v19_lead + (v1221_i1 * 12))] = v1223_data;
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

