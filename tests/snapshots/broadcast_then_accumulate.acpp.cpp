// === base name ===
kernel_2106141424ef7399

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_2106141424ef7399 = {{32, 1, 1}, 32, 32, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_2106141424ef7399(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_2106141424ef7399(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_2106141424ef7399(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_2106141424ef7399(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_2106141424ef7399(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_2106141424ef7399(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_2106141424ef7399(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
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
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"pointer_based","alias":"A","bbox":[[0],[32]],"name":"m0","ordered":false,"parts":1,"shape":[32],"variant":false},{"addressing":"pointer_based","alias":"B","bbox":[[0,0],[32,3]],"name":"m1","ordered":false,"parts":1,"shape":[32,3],"variant":false},{"addressing":"pointer_based","alias":"O","bbox":[[0,0],[32,3]],"name":"m2","ordered":false,"parts":1,"shape":[32,3],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0],[32]],"is_tmp":true,"name":"t0","offset":[0],"shape":[32]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0],[32]],"is_tmp":false,"name":"m0","offset":[0],"shape":[32]}],"permute":[[0]],"target":[[0]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,3]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[32,3]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[32,3]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,3]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,3]],"is_tmp":true,"name":"t2","offset":[0,0],"shape":[32,3]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0],[32]],"is_tmp":true,"name":"t0","offset":[0],"shape":[32]}],"permute":[[0]],"target":[[0]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,3]],"is_tmp":true,"name":"t2","offset":[0,0],"shape":[32,3]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,3]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[32,3]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[32,3]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,3]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,3]],"is_tmp":true,"name":"t2","offset":[0,0],"shape":[32,3]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1"}
        {
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v7_batchId0][0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v7_batchId0][0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v7_batchId0][0 + m2_extraOffset];
              float r0[1]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v21_lead = item.get_local_id(2) % 32;
              #pragma unroll
              for (int32_t v22_i0 = 0; v22_i0 < 1; ++v22_i0) {
                float v25_data = glb_m0[(v21_lead + (v22_i0 * 32))];
                r0[v22_i0] = v25_data;
              }
              float r2[3]{};
              // r2 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v31_i0 = 0; v31_i0 < 1; ++v31_i0) {
                int32_t v34_lead = v21_lead + (v31_i0 * 32);
                #pragma unroll
                for (int32_t v32_i1 = 0; v32_i1 < 3; ++v32_i1) {
                  float v37_data = glb_m1[(v34_lead + (v32_i1 * 32))];
                  r2[(v31_i0 + v32_i1)] = v37_data;
                }
              }
              float r1[1]{};
              // r1 = +(r0) + None
              // [(0, 32)] []
              float v27_data = r0[0];
              float v28_data = r1[0];
              r1[0] = (v28_data + v27_data);
              float r3[3]{};
              // r3 = +(r2) + None
              // [(0, 32), (0, 3)] []
              float v40_data = r2[0];
              float v41_data = r3[0];
              r3[0] = (v41_data + v40_data);
              float v43_data = r2[1];
              float v44_data = r3[1];
              r3[1] = (v44_data + v43_data);
              float v46_data = r2[2];
              float v47_data = r3[2];
              r3[2] = (v47_data + v46_data);
              float r4[3]{};
              // r4 = +(r1) + None
              // [(0, 32), (0, 3)] []
              float v50_data = r1[0];
              float v51_data = r4[0];
              r4[0] = (v51_data + v50_data);
              float v54_data = r4[1];
              r4[1] = (v54_data + v50_data);
              float v57_data = r4[2];
              r4[2] = (v57_data + v50_data);
              float r5[3]{};
              // ir5 = +(r3)
              // [(0, 32), (0, 3)] []
              float ir5[3]{};
              float v61_data = r3[0];
              float v62_data = ir5[0];
              ir5[0] = (v62_data + v61_data);
              float v64_data = r3[1];
              float v65_data = ir5[1];
              ir5[1] = (v65_data + v64_data);
              float v67_data = r3[2];
              float v68_data = ir5[2];
              ir5[2] = (v68_data + v67_data);
              // r5 = ir5 + r4
              #pragma unroll
              for (int32_t v70_n0 = 0; v70_n0 < 1; ++v70_n0) {
                #pragma unroll
                for (int32_t v71_n1 = 0; v71_n1 < 3; ++v71_n1) {
                  int32_t v72_a = v70_n0 + v71_n1;
                  float v73_data = ir5[v72_a];
                  float v74_data = r4[v72_a];
                  r5[v72_a] = (v74_data + v73_data);
                }
              }
              float r6[3]{};
              // ir6 = +(r5)
              // [(0, 32), (0, 3)] []
              float ir6[3]{};
              float v78_data = r5[0];
              float v79_data = ir6[0];
              ir6[0] = (v79_data + v78_data);
              float v81_data = r5[1];
              float v82_data = ir6[1];
              ir6[1] = (v82_data + v81_data);
              float v84_data = r5[2];
              float v85_data = ir6[2];
              ir6[2] = (v85_data + v84_data);
              // r6 = ir6
              #pragma unroll
              for (int32_t v87_n0 = 0; v87_n0 < 1; ++v87_n0) {
                #pragma unroll
                for (int32_t v88_n1 = 0; v88_n1 < 3; ++v88_n1) {
                  int32_t v89_a = v87_n0 + v88_n1;
                  float v90_data = ir6[v89_a];
                  r6[v89_a] = v90_data;
                }
              }
              // glb_m2 = store{r>g}(r6);
              #pragma unroll
              for (int32_t v91_i0 = 0; v91_i0 < 1; ++v91_i0) {
                int32_t v96_lead = v21_lead + (v91_i0 * 32);
                #pragma unroll
                for (int32_t v92_i1 = 0; v92_i1 < 3; ++v92_i1) {
                  float v94_data = r6[(v91_i0 + v92_i1)];
                  glb_m2[(v96_lead + (v92_i1 * 32))] = v94_data;
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

