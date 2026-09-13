// === base name ===
kernel_41a75482f565163f

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_41a75482f565163f = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_41a75482f565163f(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_41a75482f565163f(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_41a75482f565163f(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_41a75482f565163f(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_41a75482f565163f(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_41a75482f565163f(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_41a75482f565163f(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 16(16) {0..16} strided
        //   m1 16×16(16×16) {0..16}×{0..16} strided
        // operations:
        //   m0[i] = m1[i,k]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"OUT","bbox":[[0],[16]],"name":"m0","ordered":false,"parts":1,"shape":[16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0],[16]],"is_tmp":false,"name":"m0","offset":[0],"shape":[16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]}],"permute":[[0,1]],"target":[[0,-1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t v3_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v3_batchId0 < numElements0; v3_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v4_ahead1 = v3_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v3_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v3_batchId0 * 16 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 256 + 0 + m1_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v16_lead = item.get_local_id(2) % 16;
              #pragma unroll
              for (int32_t v17_i0 = 0; v17_i0 < 1; ++v17_i0) {
                int32_t v20_lead = v16_lead + (v17_i0 * 16);
                #pragma unroll
                for (int32_t v18_i1 = 0; v18_i1 < 16; ++v18_i1) {
                  float v23_data = glb_m1[(v20_lead + (v18_i1 * 16))];
                  r0[(v17_i0 + v18_i1)] = v23_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              float r1[1]{};
              // r1 = +(r0) + None
              // [(0, 16)] [(0, 16)]
              float ir1[1]{};
              float v27_data = r0[0];
              float v28_data = ir1[0];
              ir1[0] = (v28_data + v27_data);
              float v30_data = r0[1];
              float v31_data = ir1[0];
              ir1[0] = (v31_data + v30_data);
              float v33_data = r0[2];
              float v34_data = ir1[0];
              ir1[0] = (v34_data + v33_data);
              float v36_data = r0[3];
              float v37_data = ir1[0];
              ir1[0] = (v37_data + v36_data);
              float v39_data = r0[4];
              float v40_data = ir1[0];
              ir1[0] = (v40_data + v39_data);
              float v42_data = r0[5];
              float v43_data = ir1[0];
              ir1[0] = (v43_data + v42_data);
              float v45_data = r0[6];
              float v46_data = ir1[0];
              ir1[0] = (v46_data + v45_data);
              float v48_data = r0[7];
              float v49_data = ir1[0];
              ir1[0] = (v49_data + v48_data);
              float v51_data = r0[8];
              float v52_data = ir1[0];
              ir1[0] = (v52_data + v51_data);
              float v54_data = r0[9];
              float v55_data = ir1[0];
              ir1[0] = (v55_data + v54_data);
              float v57_data = r0[10];
              float v58_data = ir1[0];
              ir1[0] = (v58_data + v57_data);
              float v60_data = r0[11];
              float v61_data = ir1[0];
              ir1[0] = (v61_data + v60_data);
              float v63_data = r0[12];
              float v64_data = ir1[0];
              ir1[0] = (v64_data + v63_data);
              float v66_data = r0[13];
              float v67_data = ir1[0];
              ir1[0] = (v67_data + v66_data);
              float v69_data = r0[14];
              float v70_data = ir1[0];
              ir1[0] = (v70_data + v69_data);
              float v72_data = r0[15];
              float v73_data = ir1[0];
              ir1[0] = (v73_data + v72_data);
              #pragma unroll
              for (int32_t v75_n0 = 0; v75_n0 < 1; ++v75_n0) {
                float v76_data = ir1[v75_n0];
                r1[v75_n0] = v76_data;
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v77_i0 = 0; v77_i0 < 1; ++v77_i0) {
                float v78_data = r1[v77_i0];
                glb_m0[(v16_lead + (v77_i0 * 16))] = v78_data;
              }
              sycl::group_barrier(item.get_sub_group());
            }
          }
        }
      });
    }
  });
}

