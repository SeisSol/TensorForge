// === base name ===
kernel_0a9b6436839ac4ce

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_0a9b6436839ac4ce = {{16, 16, 1}, 16, 16, 1, 16, 4096, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_0a9b6436839ac4ce(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_0a9b6436839ac4ce(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_0a9b6436839ac4ce(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (16, 16, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 16;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 1024 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_0a9b6436839ac4ce(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_0a9b6436839ac4ce(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_0a9b6436839ac4ce(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_0a9b6436839ac4ce(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (1024, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 4096 B shared, occupancy grid
        // operands:
        //   m0 40(40) {0..40} strided
        //   m1 40(40) {0..40} strided
        // operations:
        //   P = abs(A)
        //   M = max(P, dims=[0])
        //   C = mul(A, M)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1024}],"shared_bytes":4096,"shared_elements":1024,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0],[40]],"name":"m0","ordered":false,"parts":1,"shape":[40],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0],[40]],"name":"m1","ordered":false,"parts":1,"shape":[40],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0],[40]],"is_tmp":true,"name":"t0","offset":[0],"shape":[40]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[0],[40]],"is_tmp":false,"name":"m0","offset":[0],"shape":[40]}],"permute":[[0]],"scalars":[],"target":[[0]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[],[]],"is_tmp":true,"name":"t1","offset":[],"shape":[]},"kind":"reduction","op":"max","ops":[{"addressing":"pointer_based","bbox":[[0],[40]],"is_tmp":true,"name":"t0","offset":[0],"shape":[40]}],"permute":[[0]],"target":[[-1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0],[40]],"is_tmp":false,"name":"m1","offset":[0],"shape":[40]},"kind":"elementwise","op":"MUL","ops":[{"addressing":"strided","bbox":[[0],[40]],"is_tmp":false,"name":"m0","offset":[0],"shape":[40]},{"addressing":"pointer_based","bbox":[[],[]],"is_tmp":true,"name":"t1","offset":[],"shape":[]}],"permute":[[0],[]],"scalars":[],"target":[[0],[]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[64 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[48];
          float * __restrict__ s0 = &localShrMem0[0];
          float * __restrict__ s1 = &localShrMem0[0];
          for (size_t v5_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v5_batchId0 < numElements0; v5_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v6_ahead1 = v5_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 40 + 0 + m0_extraOffset];
              float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 40 + 0 + m1_extraOffset];
              float r0[3]{};
              // r0 = abs(glb_m0)
              int32_t v18_lead = item.get_local_id(2) % 16;
              #pragma unroll
              for (int32_t v19_k0 = 0; v19_k0 < 2; ++v19_k0) {
                float v22_data = glb_m0[(v18_lead + (v19_k0 * 16))];
                r0[v19_k0] = (sycl::fabs(v22_data));
              }
              if (v18_lead < 8) {
                float v27_data = glb_m0[(v18_lead + 32_i32)];
                r0[2] = (sycl::fabs(v27_data));
              }
              // s0 = store{r>s}(localShrMem0, r0);
              #pragma unroll
              for (int32_t v32_i0 = 0; v32_i0 < 2; ++v32_i0) {
                float v33_data = r0[v32_i0];
                s0[(v18_lead + (v32_i0 * 16))] = v33_data;
              }
              bool v36_g = v18_lead < 8;
              if (v36_g) {
                float v37_data = r0[2];
                s0[(v18_lead + 32_i32)] = v37_data;
              }
              float r1[1]{};
              sycl::group_barrier(item.get_sub_group());
              // r1 = max(s0, dims=[0])
              float v43_data = s0[v18_lead];
              float v46_data = s0[(v18_lead + 16_i32)];
              float v47_r = sycl::max(float(v43_data), float(v46_data));
              float v52_sel0;
              if (v36_g) {
                float v50_data = s0[(v18_lead + 32_i32)];
                v52_sel0 = v50_data;
              }
              else {
                v52_sel0 = -INFINITY;
              }
              float v53_r = sycl::max(float(v47_r), float(v52_sel0));
              float v55_r = sycl::max(float(v53_r), float((sycl::permute_group_by_xor(item.get_sub_group(), v53_r, 1))));
              float v57_r = sycl::max(float(v55_r), float((sycl::permute_group_by_xor(item.get_sub_group(), v55_r, 2))));
              float v59_r = sycl::max(float(v57_r), float((sycl::permute_group_by_xor(item.get_sub_group(), v57_r, 4))));
              r1[0] = (sycl::max(float(v59_r), float((sycl::permute_group_by_xor(item.get_sub_group(), v59_r, 8)))));
              sycl::group_barrier(item.get_sub_group());
              // s1 = store{r>s}(localShrMem0, r1);
              float v62_data = r1[0];
              s1[0] = v62_data;
              sycl::group_barrier(item.get_sub_group());
              // glb_m1 = mul(glb_m0, s1)
              float v67_data = s1[0];
              #pragma unroll
              for (int32_t v63_k0 = 0; v63_k0 < 2; ++v63_k0) {
                int32_t v65_lead = v18_lead + (v63_k0 * 16);
                float v66_data = glb_m0[v65_lead];
                glb_m1[v65_lead] = ((v66_data * v67_data));
              }
              if (v36_g) {
                int32_t v70_lead = v18_lead + 32_i32;
                float v71_data = glb_m0[v70_lead];
                float v72_data = s1[0];
                glb_m1[v70_lead] = ((v71_data * v72_data));
              }
              sycl::group_barrier(item.get_sub_group());
            }
          }
        }
      });
    }
  });
}

