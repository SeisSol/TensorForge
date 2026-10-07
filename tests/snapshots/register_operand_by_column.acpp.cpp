// === base name ===
kernel_ba9c07ae3c62f35d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_ba9c07ae3c62f35d = {{32, 1, 1}, 32, 21, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_ba9c07ae3c62f35d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_ba9c07ae3c62f35d(double ** m0, size_t m0_extraOffset, const double ** m1, size_t m1_extraOffset, const double ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_ba9c07ae3c62f35d(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 0 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_ba9c07ae3c62f35d(double ** m0, size_t m0_extraOffset, const double ** m1, size_t m1_extraOffset, const double ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_ba9c07ae3c62f35d(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_ba9c07ae3c62f35d(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_ba9c07ae3c62f35d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double ** m0, size_t m0_extraOffset, const double ** m1, size_t m1_extraOffset, const double ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes (21 active) x 1 per block = block 32x1x1, 0 B shared, occupancy grid
        // operands:
        //   m0 32×3(32×3) {0..32}×{0..3} pointer_based
        //   m1 32×3(32×3) {0..32}×{0..3} pointer_based
        //   m2 9×9(9×9) {0..9}×{0..9} pointer_based
        // operations:
        //   m0[i,j]@{0..21}×{0..3} = m1[i,k]@{0..21}×{0..3} × m2[j,k]@{6..9}×{6..9}
        // tensorforge-meta: {"fp":"double","launch":{"active_threads":21,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"pointer_based","alias":"Q","bbox":[[0,0],[32,3]],"name":"m0","ordered":false,"parts":1,"shape":[32,3],"variant":false},{"addressing":"pointer_based","alias":"M0","bbox":[[0,0],[32,3]],"name":"m1","ordered":false,"parts":1,"shape":[32,3],"variant":false},{"addressing":"pointer_based","alias":"T","bbox":[[0,0],[9,9]],"name":"m2","ordered":false,"parts":1,"shape":[9,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[21,3]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,3]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[21,3]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,3]},{"addressing":"pointer_based","bbox":[[0,0],[3,3]],"is_tmp":false,"name":"m2","offset":[6,6],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[1,-1]]}],"version":"0.0.1"}
        {
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[v7_batchId0][0 + m0_extraOffset];
              const double *const __restrict__ glb_m1 = &m1[v7_batchId0][0 + m1_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[v7_batchId0][0 + m2_extraOffset];
              double r0[3]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v21_lead = item.get_local_id(2) % 32;
              bool v22_g = v21_lead < 21;
              if (v22_g) {
                #pragma unroll
                for (int32_t v23_i1 = 0; v23_i1 < 3; ++v23_i1) {
                  double v28_data = glb_m1[(v21_lead + (v23_i1 * 32))];
                  r0[v23_i1] = v28_data;
                }
              }
              double r1[3]{};
              // r1 = load{g>r}(glb_m2);
              bool v34_g = (v21_lead >= 6) && (v21_lead < 9);
              #pragma unroll
              for (int32_t v31_i0 = 6; v31_i0 < 9; ++v31_i0) {
                if (v34_g) {
                  double v39_data = glb_m2[(v31_i0 + (v21_lead * 9))];
                  r1[(v31_i0 - 6)] = v39_data;
                }
              }
              double r2[3]{};
              // ir2 = +(r0 * r1)
              // [(0, 21), (0, 3)] [(0, 3)]
              double ir2[3]{};
              double v44_data = r0[0];
              double v45_data = r1[0];
              double v48_data = ir2[0];
              ir2[0] = (v48_data + (v44_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              double v51_data = r1[1];
              double v54_data = ir2[1];
              ir2[1] = (v54_data + (v44_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              double v57_data = r1[2];
              double v60_data = ir2[2];
              ir2[2] = (v60_data + (v44_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              double v62_data = r0[1];
              double v66_data = ir2[0];
              ir2[0] = (v66_data + (v62_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              double v72_data = ir2[1];
              ir2[1] = (v72_data + (v62_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              double v78_data = ir2[2];
              ir2[2] = (v78_data + (v62_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              double v80_data = r0[2];
              double v84_data = ir2[0];
              ir2[0] = (v84_data + (v80_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              double v90_data = ir2[1];
              ir2[1] = (v90_data + (v80_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              double v96_data = ir2[2];
              ir2[2] = (v96_data + (v80_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              // r2 = ir2
              if (v22_g) {
                #pragma unroll
                for (int32_t v98_n1 = 0; v98_n1 < 3; ++v98_n1) {
                  double v100_data = ir2[v98_n1];
                  r2[v98_n1] = v100_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v101_i1 = 0; v101_i1 < 3; ++v101_i1) {
                double v103_data = r2[v101_i1];
                glb_m0[(v21_lead + (v101_i1 * 32))] = (v22_g ? v103_data : 0.0);
              }
              item.barrier();
            }
          }
        }
      });
    }
  });
}

