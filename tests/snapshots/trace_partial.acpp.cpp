// === base name ===
kernel_f275445d84710d1a

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_f275445d84710d1a = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_f275445d84710d1a(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_f275445d84710d1a(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_f275445d84710d1a(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_f275445d84710d1a(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_f275445d84710d1a(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_f275445d84710d1a(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f275445d84710d1a(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 16(16) {0..16} strided
        //   m1 16×16(16×16) {0..16}×{0..16} strided
        // operations:
        //   m0[i] = m1[i,k]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"OUT","bbox":[[0],[16]],"name":"m0","ordered":false,"parts":1,"shape":[16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0],[16]],"is_tmp":false,"name":"m0","offset":[0],"shape":[16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]}],"permute":[[0,1]],"target":[[0,-1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 16 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 256 + 0 + m1_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v20_lead = item.get_local_id(2) % 16;
              #pragma unroll
              for (int32_t v21_i0 = 0; v21_i0 < 1; ++v21_i0) {
                int32_t v24_lead = v20_lead + (v21_i0 * 16);
                #pragma unroll
                for (int32_t v22_i1 = 0; v22_i1 < 16; ++v22_i1) {
                  float v27_data = glb_m1[(v24_lead + (v22_i1 * 16))];
                  r0[(v21_i0 + v22_i1)] = v27_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              float r1[1]{};
              // ir1 = +(r0)
              // [(0, 16)] [(0, 16)]
              float ir1[1]{};
              float v31_data = r0[0];
              float v32_data = ir1[0];
              ir1[0] = (v32_data + v31_data);
              float v34_data = r0[1];
              float v35_data = ir1[0];
              ir1[0] = (v35_data + v34_data);
              float v37_data = r0[2];
              float v38_data = ir1[0];
              ir1[0] = (v38_data + v37_data);
              float v40_data = r0[3];
              float v41_data = ir1[0];
              ir1[0] = (v41_data + v40_data);
              float v43_data = r0[4];
              float v44_data = ir1[0];
              ir1[0] = (v44_data + v43_data);
              float v46_data = r0[5];
              float v47_data = ir1[0];
              ir1[0] = (v47_data + v46_data);
              float v49_data = r0[6];
              float v50_data = ir1[0];
              ir1[0] = (v50_data + v49_data);
              float v52_data = r0[7];
              float v53_data = ir1[0];
              ir1[0] = (v53_data + v52_data);
              float v55_data = r0[8];
              float v56_data = ir1[0];
              ir1[0] = (v56_data + v55_data);
              float v58_data = r0[9];
              float v59_data = ir1[0];
              ir1[0] = (v59_data + v58_data);
              float v61_data = r0[10];
              float v62_data = ir1[0];
              ir1[0] = (v62_data + v61_data);
              float v64_data = r0[11];
              float v65_data = ir1[0];
              ir1[0] = (v65_data + v64_data);
              float v67_data = r0[12];
              float v68_data = ir1[0];
              ir1[0] = (v68_data + v67_data);
              float v70_data = r0[13];
              float v71_data = ir1[0];
              ir1[0] = (v71_data + v70_data);
              float v73_data = r0[14];
              float v74_data = ir1[0];
              ir1[0] = (v74_data + v73_data);
              float v76_data = r0[15];
              float v77_data = ir1[0];
              ir1[0] = (v77_data + v76_data);
              // r1 = ir1
              #pragma unroll
              for (int32_t v79_n0 = 0; v79_n0 < 1; ++v79_n0) {
                float v80_data = ir1[v79_n0];
                r1[v79_n0] = v80_data;
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v81_i0 = 0; v81_i0 < 1; ++v81_i0) {
                float v82_data = r1[v81_i0];
                glb_m0[(v20_lead + (v81_i0 * 16))] = v82_data;
              }
              sycl::group_barrier(item.get_sub_group());
            }
          }
        }
      });
    }
  });
}

