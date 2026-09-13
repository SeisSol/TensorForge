// === base name ===
kernel_da62b7fbbc038e37

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_da62b7fbbc038e37 = {{32, 1, 1}, 32, 32, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_da62b7fbbc038e37(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_da62b7fbbc038e37(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_da62b7fbbc038e37(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_da62b7fbbc038e37(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_da62b7fbbc038e37(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_da62b7fbbc038e37(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_da62b7fbbc038e37(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes x 1 per block = block 32x1x1, 0 B shared, occupancy grid
        // operands:
        //   m0 32(32) {0..32} pointer_based
        //   m1 32×3(32×3) {0..32}×{0..3} pointer_based
        //   m2 32×3(32×3) {0..32}×{0..3} pointer_based
        // operations:
        //   t0[i] = m0[i]
        //   t1[i,j] = m1[i,j]
        //   t2[i,j] = t0[i]
        //   t2[i,j] += t1[i,j]
        //   m2[i,j] = t2[i,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"pointer_based","alias":"A","bbox":[[0],[32]],"name":"m0","ordered":false,"parts":1,"shape":[32],"variant":false},{"addressing":"pointer_based","alias":"B","bbox":[[0,0],[32,3]],"name":"m1","ordered":false,"parts":1,"shape":[32,3],"variant":false},{"addressing":"pointer_based","alias":"O","bbox":[[0,0],[32,3]],"name":"m2","ordered":false,"parts":1,"shape":[32,3],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0],[32]],"is_tmp":true,"name":"t0","offset":[0],"shape":[32]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0],[32]],"is_tmp":false,"name":"m0","offset":[0],"shape":[32]}],"permute":[[0]],"target":[[0]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,3]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[32,3]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[32,3]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,3]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,3]],"is_tmp":true,"name":"t2","offset":[0,0],"shape":[32,3]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0],[32]],"is_tmp":true,"name":"t0","offset":[0],"shape":[32]}],"permute":[[0]],"target":[[0]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,3]],"is_tmp":true,"name":"t2","offset":[0,0],"shape":[32,3]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,3]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[32,3]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[32,3]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,3]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,3]],"is_tmp":true,"name":"t2","offset":[0,0],"shape":[32,3]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          for (size_t v1_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v1_batchId0 < numElements0; v1_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v2_ahead1 = v1_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v1_batchId0][0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v1_batchId0][0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v1_batchId0][0 + m2_extraOffset];
              float r0[1]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v15_lead = item.get_local_id(2) % 32;
              #pragma unroll
              for (int32_t v16_i0 = 0; v16_i0 < 1; ++v16_i0) {
                float v19_data = glb_m0[(v15_lead + (v16_i0 * 32))];
                r0[v16_i0] = v19_data;
              }
              float r2[3]{};
              // r2 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v21_i0 = 0; v21_i0 < 1; ++v21_i0) {
                int32_t v24_lead = v15_lead + (v21_i0 * 32);
                #pragma unroll
                for (int32_t v22_i1 = 0; v22_i1 < 3; ++v22_i1) {
                  float v27_data = glb_m1[(v24_lead + (v22_i1 * 32))];
                  r2[(v21_i0 + v22_i1)] = v27_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r1[1]{};
              // r1 = +(r0) + None
              // [(0, 32)] []
              float v30_data = r0[0];
              float v31_data = r1[0];
              r1[0] = (v31_data + v30_data);
              // wait(r2 = load{g>r}(glb_m1););
              float r3[3]{};
              // r3 = +(r2) + None
              // [(0, 32), (0, 3)] []
              float v34_data = r2[0];
              float v35_data = r3[0];
              r3[0] = (v35_data + v34_data);
              float v37_data = r2[1];
              float v38_data = r3[1];
              r3[1] = (v38_data + v37_data);
              float v40_data = r2[2];
              float v41_data = r3[2];
              r3[2] = (v41_data + v40_data);
              float r4[3]{};
              // r4 = +(r1) + None
              // [(0, 32), (0, 3)] []
              float v44_data = r1[0];
              float v45_data = r4[0];
              r4[0] = (v45_data + v44_data);
              float v48_data = r4[1];
              r4[1] = (v48_data + v44_data);
              float v51_data = r4[2];
              r4[2] = (v51_data + v44_data);
              float r5[3]{};
              // r5 = +(r3) + name: r4, type: SymbolType.Register, lead: [0]
              // [(0, 32), (0, 3)] []
              float ir5[3]{};
              float v55_data = r3[0];
              float v56_data = ir5[0];
              ir5[0] = (v56_data + v55_data);
              float v58_data = r3[1];
              float v59_data = ir5[1];
              ir5[1] = (v59_data + v58_data);
              float v61_data = r3[2];
              float v62_data = ir5[2];
              ir5[2] = (v62_data + v61_data);
              #pragma unroll
              for (int32_t v64_n0 = 0; v64_n0 < 1; ++v64_n0) {
                #pragma unroll
                for (int32_t v65_n1 = 0; v65_n1 < 3; ++v65_n1) {
                  int32_t v66_a = v64_n0 + v65_n1;
                  float v67_data = ir5[v66_a];
                  float v68_data = r4[v66_a];
                  r5[v66_a] = (v68_data + v67_data);
                }
              }
              float r6[3]{};
              // r6 = +(r5) + None
              // [(0, 32), (0, 3)] []
              float ir6[3]{};
              float v72_data = r5[0];
              float v73_data = ir6[0];
              ir6[0] = (v73_data + v72_data);
              float v75_data = r5[1];
              float v76_data = ir6[1];
              ir6[1] = (v76_data + v75_data);
              float v78_data = r5[2];
              float v79_data = ir6[2];
              ir6[2] = (v79_data + v78_data);
              #pragma unroll
              for (int32_t v81_n0 = 0; v81_n0 < 1; ++v81_n0) {
                #pragma unroll
                for (int32_t v82_n1 = 0; v82_n1 < 3; ++v82_n1) {
                  int32_t v83_a = v81_n0 + v82_n1;
                  float v84_data = ir6[v83_a];
                  r6[v83_a] = v84_data;
                }
              }
              // glb_m2 = store{r>g}(r6);
              #pragma unroll
              for (int32_t v85_i0 = 0; v85_i0 < 1; ++v85_i0) {
                int32_t v90_lead = v15_lead + (v85_i0 * 32);
                #pragma unroll
                for (int32_t v86_i1 = 0; v86_i1 < 3; ++v86_i1) {
                  float v88_data = r6[(v85_i0 + v86_i1)];
                  glb_m2[(v90_lead + (v86_i1 * 32))] = v88_data;
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

