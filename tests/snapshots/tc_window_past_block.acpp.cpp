// === base name ===
kernel_28b2e8137b383edd

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_28b2e8137b383edd = {{8, 2, 1}, 8, 8, 1, 2, 64, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_28b2e8137b383edd(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_28b2e8137b383edd(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_28b2e8137b383edd(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_28b2e8137b383edd(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_28b2e8137b383edd(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_28b2e8137b383edd(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_28b2e8137b383edd(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (16, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 8 lanes x 2 per block = block 8x2x1, 64 B shared, occupancy grid
        // operands:
        //   m0 8×9(8×9) {0..8}×{0..9} strided
        //   m1 8×16(8×4) {0..8}×{9..13} none
        //   m2 16×9(4×9) {9..13}×{0..9} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,2,1],"cooperative":false,"lead_width":1,"mults_per_block":2,"persistent":true,"sections":[{"barrier":false,"mults_per_block":2,"shared_elements":16}],"shared_bytes":64,"shared_elements":16,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[8,9]],"name":"m0","ordered":false,"parts":1,"shape":[8,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,9],[8,13]],"name":"m1","ordered":false,"parts":1,"shape":[8,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[9,0],[13,9]],"name":"m2","ordered":false,"parts":1,"shape":[16,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,9],[8,13]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,16]},{"addressing":"strided","bbox":[[9,0],[13,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[8 * item.get_local_id(1) + 0];
          const float *const __restrict__ glb_m1 = &m1[0];
          size_t v9_batchIdLane0 = item.get_local_id(1) % 2;
          size_t v12_batchIdGroup0First = ((item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)) < numElements0) ? (item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)) : 0u;
          int32_t v28_lead = item.get_local_id(2) % 8;
          bool v31_g = (v28_lead >= 1) && (v28_lead < 5);
          int32_t v99_a = v28_lead + 8;
          int32_t v154_a = v28_lead + 16;
          int32_t v209_a = v28_lead + 24;
          float r0[9]{};
          size_t v279_row = v12_batchIdGroup0First + v9_batchIdLane0;
          const bool peel_batchIdActive0 = v279_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v279_row]));
          const float *const __restrict__ peel_glb_m2 = &m2[(peel_batchIdActive0 ? v279_row : v12_batchIdGroup0First) * 36 + 0 + m2_extraOffset];
          // r0 = load{g>r}(peel_glb_m2);
          if (v31_g) {
            int32_t v289_a = (v28_lead + 8_i32) - 9;
            #pragma unroll
            for (int32_t v290_i1 = 0; v290_i1 < 9; ++v290_i1) {
              float v293_data = peel_glb_m2[(v289_a + (v290_i1 * 4))];
              r0[v290_i1] = v293_data;
            }
          }
          for (size_t v13_batchIdGroup0 = (item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)); v13_batchIdGroup0 < numElements0; v13_batchIdGroup0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v276_groupNextAhead = v13_batchIdGroup0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v278_groupNext = (v276_groupNextAhead < numElements0) ? v276_groupNextAhead : v13_batchIdGroup0;
            size_t v283_row = v278_groupNext + v9_batchIdLane0;
            const bool wrap_batchIdActive0 = v283_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v283_row]));
            const float *const __restrict__ wrap_glb_m2 = &m2[(wrap_batchIdActive0 ? v283_row : v278_groupNext) * 36 + 0 + m2_extraOffset];
            size_t v14_row = v13_batchIdGroup0 + v9_batchIdLane0;
            const bool batchIdActive0 = v14_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v14_row]));
            size_t v16_batchId0 = batchIdActive0 ? v14_row : v13_batchIdGroup0;
            size_t v17_ahead1 = v16_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v19_batchId1 = (v17_ahead1 < numElements0) ? v17_ahead1 : v16_batchId0;
            float *const __restrict__ glb_m0 = &m0[v16_batchId0 * 72 + 0 + m0_extraOffset];
            float r1[9]{};
            // ir1 = +(glb_m1 * r0)
            // [(0, 8), (0, 9)] [(9, 13)]
            float ir1[9]{};
            float v45_data = glb_m1[v28_lead];
            float v46_data = r0[0];
            float v48_p = v45_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)));
            float v52_data = r0[1];
            float v54_p = v45_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)));
            float v58_data = r0[2];
            float v60_p = v45_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)));
            float v64_data = r0[3];
            float v66_p = v45_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)));
            float v70_data = r0[4];
            float v72_p = v45_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)));
            float v76_data = r0[5];
            float v78_p = v45_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)));
            float v82_data = r0[6];
            float v84_p = v45_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)));
            float v88_data = r0[7];
            float v90_p = v45_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)));
            float v94_data = r0[8];
            // r0 = load{g>r}(wrap_glb_m2);
            if (v31_g) {
              int32_t v296_a = (v28_lead + 8_i32) - 9;
              #pragma unroll
              for (int32_t v297_i1 = 0; v297_i1 < 9; ++v297_i1) {
                float v300_data = wrap_glb_m2[(v296_a + (v297_i1 * 4))];
                r0[v297_i1] = v300_data;
              }
            }
            float v100_data = glb_m1[v99_a];
            float v155_data = glb_m1[v154_a];
            float v166_p = (v54_p + (v100_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2))))) + (v155_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3))));
            float v172_p = (v60_p + (v100_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2))))) + (v155_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3))));
            float v178_p = (v66_p + (v100_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2))))) + (v155_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3))));
            float v184_p = (v72_p + (v100_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2))))) + (v155_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3))));
            float v190_p = (v78_p + (v100_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2))))) + (v155_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3))));
            float v196_p = (v84_p + (v100_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2))))) + (v155_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3))));
            float v202_p = (v90_p + (v100_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2))))) + (v155_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3))));
            float v208_p = ((v45_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))) + (v100_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2))))) + (v155_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3))));
            float v210_data = glb_m1[v209_a];
            ir1[0] = (((v48_p + (v100_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2))))) + (v155_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3))))) + (v210_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            ir1[1] = (v166_p + (v210_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            ir1[2] = (v172_p + (v210_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            ir1[3] = (v178_p + (v210_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            ir1[4] = (v184_p + (v210_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            ir1[5] = (v190_p + (v210_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            ir1[6] = (v196_p + (v210_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            ir1[7] = (v202_p + (v210_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            ir1[8] = (v208_p + (v210_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            // r1 = ir1
            // glb_m0 = store{r>g}(r1);
            #pragma unroll
            for (int32_t v268_i0 = 0; v268_i0 < 1; ++v268_i0) {
              #pragma unroll
              for (int32_t v269_i1 = 0; v269_i1 < 9; ++v269_i1) {
                float v271_data = ir1[v269_i1];
                if (batchIdActive0) {
                  glb_m0[((v28_lead + (v268_i0 * 8)) + (v269_i1 * 8))] = v271_data;
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

