// === base name ===
kernel_1f8641f0fbadc90c

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_1f8641f0fbadc90c = {{2, 8, 1}, 2, 2, 1, 8, 256, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_1f8641f0fbadc90c(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_1f8641f0fbadc90c(__float128 * m0, size_t m0_extraOffset, const __float128 * m1, size_t m1_extraOffset, const __float128 * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_1f8641f0fbadc90c(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (2, 8, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 8 - 1) / 8;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 2;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 16 * sizeof(__float128);
  config.cooperative = false;
  return config;
}
void launcher_kernel_1f8641f0fbadc90c(__float128 * m0, size_t m0_extraOffset, const __float128 * m1, size_t m1_extraOffset, const __float128 * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_1f8641f0fbadc90c(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_1f8641f0fbadc90c(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_1f8641f0fbadc90c(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, __float128 * m0, size_t m0_extraOffset, const __float128 * m1, size_t m1_extraOffset, const __float128 * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<__float128, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (16, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 2 lanes x 8 per block = block 2x8x1, 256 B shared, occupancy grid
        // operands:
        //   m0 2×2(2×2) {0..2}×{0..2} strided
        //   m1 2×2(2×2) {0..2}×{0..2} strided
        //   m2 2×2(2×2) {0..2}×{0..2} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"__float128","launch":{"active_threads":2,"block":[2,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":16}],"shared_bytes":256,"shared_elements":16,"threads_per_mult":2},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[2,2]],"name":"m0","ordered":false,"parts":1,"shape":[2,2],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[2,2]],"name":"m1","ordered":false,"parts":1,"shape":[2,2],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[2,2]],"name":"m2","ordered":false,"parts":1,"shape":[2,2],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[2,2]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[2,2]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[2,2]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[2,2]},{"addressing":"strided","bbox":[[0,0],[2,2]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[2,2]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          __float128* localShrMem0 = &totalShrMem[2 * item.get_local_id(1) + 0];
          size_t v8_batchIdLane0 = item.get_local_id(1) % 8;
          int32_t v25_lead = item.get_local_id(2) % 2;
          for (size_t v9_batchIdGroup0 = (item.get_local_id(1) - item.get_local_id(1) % 8) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)); v9_batchIdGroup0 < numElements0; v9_batchIdGroup0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_row = v9_batchIdGroup0 + v8_batchIdLane0;
            const bool batchIdActive0 = v10_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v10_row]));
            size_t v12_batchId0 = batchIdActive0 ? v10_row : v9_batchIdGroup0;
            size_t v13_ahead1 = v12_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v15_batchId1 = (v13_ahead1 < numElements0) ? v13_ahead1 : v12_batchId0;
            __float128 *const __restrict__ glb_m0 = &m0[v12_batchId0 * 4 + 0 + m0_extraOffset];
            const __float128 *const __restrict__ glb_m1 = &m1[v12_batchId0 * 4 + 0 + m1_extraOffset];
            const __float128 *const __restrict__ glb_m2 = &m2[v12_batchId0 * 4 + 0 + m2_extraOffset];
            __float128 r0[2]{};
            // r0 = load{g>r}(glb_m1);
            #pragma unroll
            for (int32_t v26_i0 = 0; v26_i0 < 1; ++v26_i0) {
              int32_t v29_lead = v25_lead + (v26_i0 * 2);
              #pragma unroll
              for (int32_t v27_i1 = 0; v27_i1 < 2; ++v27_i1) {
                __float128 v32_data = glb_m1[(v29_lead + (v27_i1 * 2))];
                r0[(v26_i0 + v27_i1)] = v32_data;
              }
            }
            __float128 r1[2]{};
            // r1 = load{g>r}(glb_m2);
            #pragma unroll
            for (int32_t v35_i0 = 0; v35_i0 < 1; ++v35_i0) {
              int32_t v38_lead = v25_lead + (v35_i0 * 2);
              #pragma unroll
              for (int32_t v36_i1 = 0; v36_i1 < 2; ++v36_i1) {
                __float128 v41_data = glb_m2[(v38_lead + (v36_i1 * 2))];
                r1[(v35_i0 + v36_i1)] = v41_data;
              }
            }
            __float128 r2[2]{};
            // ir2 = +(r0 * r1)
            // [(0, 2), (0, 2)] [(0, 2)]
            __float128 ir2[2]{};
            __float128 v45_data = r0[0];
            __float128 v46_data = r1[0];
            __float128 v49_data = ir2[0];
            ir2[0] = (v49_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 2) * 2 + (0)))));
            __float128 v52_data = r1[1];
            __float128 v55_data = ir2[1];
            ir2[1] = (v55_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 2) * 2 + (0)))));
            __float128 v57_data = r0[1];
            __float128 v61_data = ir2[0];
            ir2[0] = (v61_data + (v57_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 2) * 2 + (1)))));
            __float128 v67_data = ir2[1];
            ir2[1] = (v67_data + (v57_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 2) * 2 + (1)))));
            // r2 = ir2
            #pragma unroll
            for (int32_t v69_n0 = 0; v69_n0 < 1; ++v69_n0) {
              #pragma unroll
              for (int32_t v70_n1 = 0; v70_n1 < 2; ++v70_n1) {
                int32_t v71_a = v69_n0 + v70_n1;
                __float128 v72_data = ir2[v71_a];
                r2[v71_a] = v72_data;
              }
            }
            // glb_m0 = store{r>g}(r2);
            #pragma unroll
            for (int32_t v73_i0 = 0; v73_i0 < 1; ++v73_i0) {
              #pragma unroll
              for (int32_t v74_i1 = 0; v74_i1 < 2; ++v74_i1) {
                __float128 v76_data = r2[(v73_i0 + v74_i1)];
                if (batchIdActive0) {
                  glb_m0[((v25_lead + (v73_i0 * 2)) + (v74_i1 * 2))] = v76_data;
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

