// === base name ===
kernel_3d9d1aa407ae31d1

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_3d9d1aa407ae31d1 = {{32, 1, 1}, 32, 32, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_3d9d1aa407ae31d1(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_3d9d1aa407ae31d1(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_3d9d1aa407ae31d1(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_3d9d1aa407ae31d1(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_3d9d1aa407ae31d1(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_3d9d1aa407ae31d1(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_3d9d1aa407ae31d1(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes x 1 per block = block 32x1x1, 0 B shared, occupancy grid
        // operands:
        //   m0 32×13(32×13) {0..32}×{0..13} strided
        //   m1 32×13(32×13) {0..32}×{0..13} strided
        //   m2 13×13(13×13) {0..13}×{0..13} strided
        // operations:
        //   m0[i,j]@{0..32}×{6..13} = m1[i,k]@{0..32}×{10..13} × m2[k,j]@{10..13}×{6..13}
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[32,13]],"name":"m0","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"Q","bbox":[[0,0],[32,13]],"name":"m1","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"S","bbox":[[0,0],[13,13]],"name":"m2","ordered":false,"parts":1,"shape":[13,13],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,6],[32,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,10],[32,13]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[10,6],[13,13]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          for (size_t v1_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v1_batchId0 < numElements0; v1_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v2_ahead1 = v1_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v1_batchId0 * 416 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v1_batchId0 * 416 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v1_batchId0 * 169 + 0 + m2_extraOffset];
              float r0[3]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v15_lead = item.get_local_id(2) % 32;
              #pragma unroll
              for (int32_t v16_i0 = 0; v16_i0 < 1; ++v16_i0) {
                int32_t v19_lead = v15_lead + (v16_i0 * 32);
                #pragma unroll
                for (int32_t v17_i1 = 10; v17_i1 < 13; ++v17_i1) {
                  float v22_data = glb_m1[(v19_lead + (v17_i1 * 32))];
                  r0[(v16_i0 + (v17_i1 - 10))] = v22_data;
                }
              }
              float r1[7]{};
              // r1 = load{g>r}(glb_m2);
              if ((v15_lead >= 10) && (v15_lead < 13)) {
                #pragma unroll
                for (int32_t v29_i1 = 6; v29_i1 < 13; ++v29_i1) {
                  float v34_data = glb_m2[(v15_lead + (v29_i1 * 13))];
                  r1[(v29_i1 - 6)] = v34_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[7]{};
              // r2 = +(r0 * r1) + None
              // [(0, 32), (6, 13)] [(10, 13)]
              float ir2[7]{};
              float v39_data = r0[0];
              float v40_data = r1[0];
              float v43_data = ir2[0];
              ir2[0] = (v43_data + (v39_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v46_data = r1[1];
              float v49_data = ir2[1];
              ir2[1] = (v49_data + (v39_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v52_data = r1[2];
              float v55_data = ir2[2];
              ir2[2] = (v55_data + (v39_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v58_data = r1[3];
              float v61_data = ir2[3];
              ir2[3] = (v61_data + (v39_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v64_data = r1[4];
              float v67_data = ir2[4];
              ir2[4] = (v67_data + (v39_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v70_data = r1[5];
              float v73_data = ir2[5];
              ir2[5] = (v73_data + (v39_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v76_data = r1[6];
              float v79_data = ir2[6];
              ir2[6] = (v79_data + (v39_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v81_data = r0[1];
              float v85_data = ir2[0];
              ir2[0] = (v85_data + (v81_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v91_data = ir2[1];
              ir2[1] = (v91_data + (v81_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v97_data = ir2[2];
              ir2[2] = (v97_data + (v81_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v103_data = ir2[3];
              ir2[3] = (v103_data + (v81_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v109_data = ir2[4];
              ir2[4] = (v109_data + (v81_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v115_data = ir2[5];
              ir2[5] = (v115_data + (v81_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v121_data = ir2[6];
              ir2[6] = (v121_data + (v81_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v123_data = r0[2];
              float v127_data = ir2[0];
              ir2[0] = (v127_data + (v123_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v133_data = ir2[1];
              ir2[1] = (v133_data + (v123_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v139_data = ir2[2];
              ir2[2] = (v139_data + (v123_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v145_data = ir2[3];
              ir2[3] = (v145_data + (v123_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v151_data = ir2[4];
              ir2[4] = (v151_data + (v123_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v157_data = ir2[5];
              ir2[5] = (v157_data + (v123_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v163_data = ir2[6];
              ir2[6] = (v163_data + (v123_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              #pragma unroll
              for (int32_t v165_n0 = 0; v165_n0 < 1; ++v165_n0) {
                #pragma unroll
                for (int32_t v166_n1 = 6; v166_n1 < 13; ++v166_n1) {
                  int32_t v168_a = v165_n0 + (v166_n1 - 6);
                  float v169_data = ir2[v168_a];
                  r2[v168_a] = v169_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v170_i0 = 0; v170_i0 < 1; ++v170_i0) {
                int32_t v173_lead = v15_lead + (v170_i0 * 32);
                glb_m0[v173_lead] = 0.0f;
                glb_m0[(v173_lead + 32)] = 0.0f;
                glb_m0[(v173_lead + 64)] = 0.0f;
                glb_m0[(v173_lead + 96)] = 0.0f;
                glb_m0[(v173_lead + 128)] = 0.0f;
                glb_m0[(v173_lead + 160)] = 0.0f;
                float v181_data = r2[v170_i0];
                glb_m0[(v173_lead + 192)] = v181_data;
                float v184_data = r2[(v170_i0 + 1)];
                glb_m0[(v173_lead + 224)] = v184_data;
                float v187_data = r2[(v170_i0 + 2)];
                glb_m0[(v173_lead + 256)] = v187_data;
                float v190_data = r2[(v170_i0 + 3)];
                glb_m0[(v173_lead + 288)] = v190_data;
                float v193_data = r2[(v170_i0 + 4)];
                glb_m0[(v173_lead + 320)] = v193_data;
                float v196_data = r2[(v170_i0 + 5)];
                glb_m0[(v173_lead + 352)] = v196_data;
                float v199_data = r2[(v170_i0 + 6)];
                glb_m0[(v173_lead + 384)] = v199_data;
              }
              item.barrier();
            }
          }
        }
      });
    }
  });
}

