// === base name ===
kernel_5b695339825af194

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_5b695339825af194 = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_5b695339825af194(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_5b695339825af194(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_5b695339825af194(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_5b695339825af194(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_5b695339825af194(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_5b695339825af194(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_5b695339825af194(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 40×3(40×3) {0..40}×{0..3} strided
        //   m1 3(3) {0..3} strided
        // operations:
        //   OUT = +(A, dims=[0])
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[40,3]],"name":"m0","ordered":false,"parts":1,"shape":[40,3],"variant":false},{"addressing":"strided","alias":"OUT","bbox":[[0],[3]],"name":"m1","ordered":false,"parts":1,"shape":[3],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0],[3]],"is_tmp":false,"name":"m1","offset":[0],"shape":[3]},"kind":"reduction","op":"+","ops":[{"addressing":"strided","bbox":[[0,0],[40,3]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[40,3]}],"permute":[[0,1]],"target":[[-1,0]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 120 + 0 + m0_extraOffset];
              float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 3 + 0 + m1_extraOffset];
              // glb_m1 = +(glb_m0, dims=[0])
              int32_t v20_lead = item.get_local_id(2) % 16;
              int32_t v27_lead = v20_lead + 16_i32;
              bool v31_own = v20_lead < 8;
              bool v47_w = v20_lead == 0;
              #pragma unroll
              for (int32_t v17_k1 = 0; v17_k1 < 3; ++v17_k1) {
                int32_t v23_a = v17_k1 * 40;
                float v25_data = glb_m0[(v20_lead + v23_a)];
                float v29_data = glb_m0[(v27_lead + v23_a)];
                float v30_r = v25_data + v29_data;
                float v37_sel0;
                if (v31_own) {
                  float v35_data = glb_m0[((v20_lead + 32_i32) + v23_a)];
                  v37_sel0 = v35_data;
                }
                else {
                  v37_sel0 = 0.0f;
                }
                float v38_r = v30_r + v37_sel0;
                float v40_r = v38_r + (sycl::permute_group_by_xor(item.get_sub_group(), v38_r, 1));
                float v42_r = v40_r + (sycl::permute_group_by_xor(item.get_sub_group(), v40_r, 2));
                float v44_r = v42_r + (sycl::permute_group_by_xor(item.get_sub_group(), v42_r, 4));
                float v46_r = v44_r + (sycl::permute_group_by_xor(item.get_sub_group(), v44_r, 8));
                if (v47_w) {
                  glb_m1[v17_k1] = v46_r;
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

