// === base name ===
kernel_bd47484559f15ba5

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_bd47484559f15ba5 = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_bd47484559f15ba5(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_bd47484559f15ba5(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_bd47484559f15ba5(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_bd47484559f15ba5(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_bd47484559f15ba5(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_bd47484559f15ba5(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_bd47484559f15ba5(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 8×8(8×8) {0..8}×{0..8} strided
        //   m1 8×8(8×8) {0..8}×{0..8} strided
        //   m2 8×8(8×8) {0..8}×{0..8} strided
        // operations:
        //   TMP = +(A, dims=[1])
        //   m1[i,j] = t0[i] × m2[i,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0],[8]],"is_tmp":true,"name":"t0","offset":[0],"shape":[8]},"kind":"reduction","op":"+","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"target":[[0,-1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0],[8]],"is_tmp":true,"name":"t0","offset":[0],"shape":[8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]}],"permute":[[0],[0,1]],"target":[[0],[0,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 64 + 0 + m0_extraOffset];
              float *const __restrict__ glb_m1 = &m1[v9_batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v9_batchId0 * 64 + 0 + m2_extraOffset];
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              int32_t v23_lead = item.get_local_id(2) % 16;
              bool v24_g = v23_lead < 8;
              if (v24_g) {
                #pragma unroll
                for (int32_t v25_i1 = 0; v25_i1 < 8; ++v25_i1) {
                  float v30_data = glb_m2[(v23_lead + (v25_i1 * 8))];
                  r1[v25_i1] = v30_data;
                }
              }
              float r0[1]{};
              // r0 = +(glb_m0, dims=[1])
              if (v24_g) {
                float v34_acc0 = 0.0f;
                #pragma unroll
                for (int32_t v33_r1 = 0; v33_r1 < 8; ++v33_r1) {
                  float v39_data = glb_m0[(v23_lead + (v33_r1 * 8))];
                  v34_acc0 = (v34_acc0 + v39_data);
                }
                r0[0] = v34_acc0;
              }
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // ir2 = +(r0 * r1)
              // [(0, 8), (0, 8)] []
              float ir2[8]{};
              float v44_data = r0[0];
              float v45_data = r1[0];
              float v47_data = ir2[0];
              ir2[0] = (v47_data + (v44_data * v45_data));
              float v50_data = r1[1];
              float v52_data = ir2[1];
              ir2[1] = (v52_data + (v44_data * v50_data));
              float v55_data = r1[2];
              float v57_data = ir2[2];
              ir2[2] = (v57_data + (v44_data * v55_data));
              float v60_data = r1[3];
              float v62_data = ir2[3];
              ir2[3] = (v62_data + (v44_data * v60_data));
              float v65_data = r1[4];
              float v67_data = ir2[4];
              ir2[4] = (v67_data + (v44_data * v65_data));
              float v70_data = r1[5];
              float v72_data = ir2[5];
              ir2[5] = (v72_data + (v44_data * v70_data));
              float v75_data = r1[6];
              float v77_data = ir2[6];
              ir2[6] = (v77_data + (v44_data * v75_data));
              float v80_data = r1[7];
              float v82_data = ir2[7];
              ir2[7] = (v82_data + (v44_data * v80_data));
              // r2 = ir2
              if (v24_g) {
                #pragma unroll
                for (int32_t v84_n1 = 0; v84_n1 < 8; ++v84_n1) {
                  float v86_data = ir2[v84_n1];
                  r2[v84_n1] = v86_data;
                }
              }
              // glb_m1 = store{r>g}(r2);
              if (v24_g) {
                #pragma unroll
                for (int32_t v87_i1 = 0; v87_i1 < 8; ++v87_i1) {
                  float v89_data = r2[v87_i1];
                  glb_m1[(v23_lead + (v87_i1 * 8))] = v89_data;
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

