// === base name ===
kernel_aae0490eac9c85a0

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_aae0490eac9c85a0 = {{32, 1, 1}, 32, 20, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_aae0490eac9c85a0(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_aae0490eac9c85a0(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_aae0490eac9c85a0(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_aae0490eac9c85a0(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_aae0490eac9c85a0(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_aae0490eac9c85a0(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_aae0490eac9c85a0(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes (20 active) x 1 per block = block 32x1x1, 0 B shared, occupancy grid
        // operands:
        //   m0 20×9(20×9) {0..20}×{0..9} strided
        //   m1 1×20(1×20) {0..1}×{0..20} strided
        //   m2 1×9(1×9) {0..1}×{0..9} strided
        // operations:
        //   m0[i,j] = m1[k,i] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":20,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[20,9]],"name":"m0","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[1,20]],"name":"m1","ordered":false,"parts":1,"shape":[1,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[1,9]],"name":"m2","ordered":false,"parts":1,"shape":[1,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[1,20]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[1,20]},{"addressing":"strided","bbox":[[0,0],[1,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[1,9]}],"permute":[[0,1],[0,1]],"target":[[-1,0],[-1,1]]}],"version":"0.0.1"}
        {
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 180 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 20 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 9 + 0 + m2_extraOffset];
              float r0[1]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v22_lead = item.get_local_id(2) % 32;
              bool v23_g = v22_lead < 20;
              #pragma unroll
              for (int32_t v19_i0 = 0; v19_i0 < 1; ++v19_i0) {
                if (v23_g) {
                  float v27_data = glb_m1[(v19_i0 + v22_lead)];
                  r0[v19_i0] = v27_data;
                }
              }
              float r1[9]{};
              // r1 = load{g>r}(glb_m2);
              int32_t v32_lead = item.get_local_id(2) % 32;
              if (v32_lead < 1) {
                #pragma unroll
                for (int32_t v34_i1 = 0; v34_i1 < 9; ++v34_i1) {
                  float v38_data = glb_m2[(v32_lead + v34_i1)];
                  r1[v34_i1] = v38_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[9]{};
              // ir2 = +(r0 * r1)
              // [(0, 20), (0, 9)] [(0, 1)]
              float ir2[9]{};
              float v42_data = r0[0];
              float v43_data = r1[0];
              float v46_data = ir2[0];
              ir2[0] = (v46_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v49_data = r1[1];
              float v52_data = ir2[1];
              ir2[1] = (v52_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v55_data = r1[2];
              float v58_data = ir2[2];
              ir2[2] = (v58_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v61_data = r1[3];
              float v64_data = ir2[3];
              ir2[3] = (v64_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v67_data = r1[4];
              float v70_data = ir2[4];
              ir2[4] = (v70_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v73_data = r1[5];
              float v76_data = ir2[5];
              ir2[5] = (v76_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v79_data = r1[6];
              float v82_data = ir2[6];
              ir2[6] = (v82_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v85_data = r1[7];
              float v88_data = ir2[7];
              ir2[7] = (v88_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v91_data = r1[8];
              float v94_data = ir2[8];
              ir2[8] = (v94_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              // r2 = ir2
              bool v96_g = v32_lead < 20;
              if (v96_g) {
                #pragma unroll
                for (int32_t v97_n1 = 0; v97_n1 < 9; ++v97_n1) {
                  float v99_data = ir2[v97_n1];
                  r2[v97_n1] = v99_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v96_g) {
                #pragma unroll
                for (int32_t v101_i1 = 0; v101_i1 < 9; ++v101_i1) {
                  float v103_data = r2[v101_i1];
                  glb_m0[(v32_lead + (v101_i1 * 20))] = v103_data;
                }
              }
              item.barrier();
            }
          }
        }
      });
    }
  });
}

