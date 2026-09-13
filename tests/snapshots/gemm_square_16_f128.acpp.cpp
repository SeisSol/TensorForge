// === base name ===
kernel_7b2df644bc629cc6

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_7b2df644bc629cc6 = {{2, 8, 1}, 2, 2, 1, 8, 256, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_7b2df644bc629cc6(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_7b2df644bc629cc6(__float128 * m0, size_t m0_extraOffset, const __float128 * m1, size_t m1_extraOffset, const __float128 * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_7b2df644bc629cc6(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (2, 8, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 2;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 16 * sizeof(__float128);
  config.cooperative = false;
  return config;
}
void launcher_kernel_7b2df644bc629cc6(__float128 * m0, size_t m0_extraOffset, const __float128 * m1, size_t m1_extraOffset, const __float128 * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_7b2df644bc629cc6(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_7b2df644bc629cc6(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_7b2df644bc629cc6(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, __float128 * m0, size_t m0_extraOffset, const __float128 * m1, size_t m1_extraOffset, const __float128 * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<__float128, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (16, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 2 lanes x 8 per block = block 2x8x1, 256 B shared, occupancy grid
        // operands:
        //   m0 2×2(2×2) {0..2}×{0..2} strided
        //   m1 2×2(2×2) {0..2}×{0..2} strided
        //   m2 2×2(2×2) {0..2}×{0..2} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"__float128","launch":{"active_threads":2,"block":[2,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":16}],"shared_bytes":256,"shared_elements":16,"threads_per_mult":2},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[2,2]],"name":"m0","ordered":false,"parts":1,"shape":[2,2],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[2,2]],"name":"m1","ordered":false,"parts":1,"shape":[2,2],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[2,2]],"name":"m2","ordered":false,"parts":1,"shape":[2,2],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[2,2]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[2,2]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[2,2]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[2,2]},{"addressing":"strided","bbox":[[0,0],[2,2]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[2,2]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          __float128* localShrMem0 = &totalShrMem[2 * item.get_local_id(1) + 0];
          __float128* tempShrMem = &localShrMem0[0];
          size_t v3_batchIdLane0 = item.get_local_id(1) % 8;
          int32_t v21_lead = item.get_local_id(2) % 2;
          for (size_t v4_batchIdGroup0 = (item.get_local_id(1) - item.get_local_id(1) % 8) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)); v4_batchIdGroup0 < numElements0; v4_batchIdGroup0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v5_row = v4_batchIdGroup0 + v3_batchIdLane0;
            const bool batchIdActive0 = v5_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v5_row]));
            size_t v7_batchId0 = batchIdActive0 ? v5_row : v4_batchIdGroup0;
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            __float128 *const __restrict__ glb_m0 = &m0[v7_batchId0 * 4 + 0 + m0_extraOffset];
            const __float128 *const __restrict__ glb_m1 = &m1[v7_batchId0 * 4 + 0 + m1_extraOffset];
            const __float128 *const __restrict__ glb_m2 = &m2[v7_batchId0 * 4 + 0 + m2_extraOffset];
            __float128 r0[2]{};
            // r0 = load{g>r}(glb_m1);
            #pragma unroll
            for (int32_t v22_i0 = 0; v22_i0 < 1; ++v22_i0) {
              int32_t v25_lead = v21_lead + (v22_i0 * 2);
              #pragma unroll
              for (int32_t v23_i1 = 0; v23_i1 < 2; ++v23_i1) {
                __float128 v28_data = glb_m1[(v25_lead + (v23_i1 * 2))];
                r0[(v22_i0 + v23_i1)] = v28_data;
              }
            }
            __float128 r1[2]{};
            // r1 = load{g>r}(glb_m2);
            #pragma unroll
            for (int32_t v31_i0 = 0; v31_i0 < 1; ++v31_i0) {
              int32_t v34_lead = v21_lead + (v31_i0 * 2);
              #pragma unroll
              for (int32_t v32_i1 = 0; v32_i1 < 2; ++v32_i1) {
                __float128 v37_data = glb_m2[(v34_lead + (v32_i1 * 2))];
                r1[(v31_i0 + v32_i1)] = v37_data;
              }
            }
            // wait(r0 = load{g>r}(glb_m1););
            // wait(r1 = load{g>r}(glb_m2););
            __float128 r2[2]{};
            // ir2 = +(r0 * r1)
            // [(0, 2), (0, 2)] [(0, 2)]
            __float128 ir2[2]{};
            __float128 v41_data = r0[0];
            __float128 v42_data = r1[0];
            __float128 v45_data = ir2[0];
            ir2[0] = (v45_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 2) * 2 + (0)))));
            __float128 v48_data = r1[1];
            __float128 v51_data = ir2[1];
            ir2[1] = (v51_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 2) * 2 + (0)))));
            __float128 v53_data = r0[1];
            __float128 v57_data = ir2[0];
            ir2[0] = (v57_data + (v53_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 2) * 2 + (1)))));
            __float128 v63_data = ir2[1];
            ir2[1] = (v63_data + (v53_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 2) * 2 + (1)))));
            // r2 = ir2
            #pragma unroll
            for (int32_t v65_n0 = 0; v65_n0 < 1; ++v65_n0) {
              #pragma unroll
              for (int32_t v66_n1 = 0; v66_n1 < 2; ++v66_n1) {
                int32_t v67_a = v65_n0 + v66_n1;
                __float128 v68_data = ir2[v67_a];
                r2[v67_a] = v68_data;
              }
            }
            // glb_m0 = store{r>g}(r2);
            #pragma unroll
            for (int32_t v69_i0 = 0; v69_i0 < 1; ++v69_i0) {
              #pragma unroll
              for (int32_t v70_i1 = 0; v70_i1 < 2; ++v70_i1) {
                __float128 v72_data = r2[(v69_i0 + v70_i1)];
                if (batchIdActive0) {
                  glb_m0[((v21_lead + (v69_i0 * 2)) + (v70_i1 * 2))] = v72_data;
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

