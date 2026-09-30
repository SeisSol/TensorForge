// === base name ===
kernel_5a20beff4cf1338f

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_5a20beff4cf1338f = {{32, 1, 1}, 32, 32, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_5a20beff4cf1338f(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_5a20beff4cf1338f(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_5a20beff4cf1338f(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (32, 1, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 1 - 1) / 1;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 32;
  config.block[1] = 1;
  config.block[2] = 1;
  config.sharedMemBytes = 0 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_5a20beff4cf1338f(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_5a20beff4cf1338f(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_5a20beff4cf1338f(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_5a20beff4cf1338f(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes x 1 per block = block 32x1x1, 0 B shared, occupancy grid
        // operands:
        //   m0 32(32) {0..32} strided
        //   m1 32(32) {0..32} strided
        //   m2 ()  scalar
        //   m3 ()  scalar
        // operations:
        //   m0[i] = m1[i]
        //   m0[i] += m2[] × m3[]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"O","bbox":[[0],[32]],"name":"m0","ordered":false,"parts":1,"shape":[32],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0],[32]],"name":"m1","ordered":false,"parts":1,"shape":[32],"variant":false},{"addressing":"scalar","alias":null,"bbox":[[],[]],"name":"m2","ordered":false,"parts":1,"shape":[],"variant":false},{"addressing":"scalar","alias":null,"bbox":[[],[]],"name":"m3","ordered":false,"parts":1,"shape":[],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0],[32]],"is_tmp":false,"name":"m0","offset":[0],"shape":[32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0],[32]],"is_tmp":false,"name":"m1","offset":[0],"shape":[32]}],"permute":[[0]],"target":[[0]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0],[32]],"is_tmp":false,"name":"m0","offset":[0],"shape":[32]},"kind":"multilinear","ops":[{"addressing":"scalar","bbox":[[],[]],"is_tmp":false,"name":"m2","offset":[],"shape":[]},{"addressing":"scalar","bbox":[[],[]],"is_tmp":false,"name":"m3","offset":[],"shape":[]}],"permute":[[],[]],"target":[[],[]]}],"version":"0.0.1"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          for (size_t v1_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v1_batchId0 < numElements0; v1_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v2_ahead1 = v1_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v1_batchId0 * 32 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v1_batchId0 * 32 + 0 + m1_extraOffset];
              float r0[1]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v14_lead = item.get_local_id(2) % 32;
              #pragma unroll
              for (int32_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
                float v18_data = glb_m1[(v14_lead + (v15_i0 * 32))];
                r0[v15_i0] = v18_data;
              }
              // wait(r0 = load{g>r}(glb_m1););
              float r1[1]{};
              // ir1 = +(r0)
              // [(0, 32)] []
              float ir1[1]{};
              float v21_data = r0[0];
              float v22_data = ir1[0];
              ir1[0] = (v22_data + v21_data);
              // r1 = ir1
              #pragma unroll
              for (int32_t v24_n0 = 0; v24_n0 < 1; ++v24_n0) {
                float v25_data = ir1[v24_n0];
                r1[v24_n0] = v25_data;
              }
              float r2[1]{};
              // ir2 = +()
              // [(0, 32)] []
              float ir2[1]{};
              // r2 = ir2 * glb_m2 * glb_m3 + r1
              #pragma unroll
              for (int32_t v31_n0 = 0; v31_n0 < 1; ++v31_n0) {
                float v34_data = r1[v31_n0];
                r2[v31_n0] = (v34_data + 6.0f);
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v36_i0 = 0; v36_i0 < 1; ++v36_i0) {
                float v37_data = r2[v36_i0];
                glb_m0[(v14_lead + (v36_i0 * 32))] = v37_data;
              }
              item.barrier();
            }
          }
        }
      });
    }
  });
}

