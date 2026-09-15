// === base name ===
kernel_0ca72d399185c380

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_0ca72d399185c380 = {{16, 16, 1}, 16, 16, 1, 16, 4096, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_0ca72d399185c380(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_0ca72d399185c380(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_0ca72d399185c380(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_0ca72d399185c380(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_0ca72d399185c380(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_0ca72d399185c380(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_0ca72d399185c380(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0) {
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
          for (size_t v4_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v4_batchId0 < numElements0; v4_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v5_ahead1 = v4_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v4_batchId0 * 40 + 0 + m0_extraOffset];
              float *const __restrict__ glb_m1 = &m1[v4_batchId0 * 40 + 0 + m1_extraOffset];
              float r0[3]{};
              // r0 = abs(glb_m0)
              int32_t v17_lead = item.get_local_id(2) % 16;
              #pragma unroll
              for (int32_t v18_k0 = 0; v18_k0 < 2; ++v18_k0) {
                float v21_data = glb_m0[(v17_lead + (v18_k0 * 16))];
                r0[v18_k0] = (sycl::fabs(v21_data));
              }
              if (v17_lead < 8) {
                float v26_data = glb_m0[(v17_lead + 32_i32)];
                r0[2] = (sycl::fabs(v26_data));
              }
              // s0 = store{r>s}(localShrMem0, r0);
              #pragma unroll
              for (int32_t v31_i0 = 0; v31_i0 < 2; ++v31_i0) {
                float v32_data = r0[v31_i0];
                s0[(v17_lead + (v31_i0 * 16))] = v32_data;
              }
              bool v35_g = v17_lead < 8;
              if (v35_g) {
                float v36_data = r0[2];
                s0[(v17_lead + 32_i32)] = v36_data;
              }
              float r1[1]{};
              sycl::group_barrier(item.get_sub_group());
              // r1 = max(s0, dims=[0])
              float v42_data = s0[v17_lead];
              float v45_data = s0[(v17_lead + 16_i32)];
              float v46_r = sycl::max(float(v42_data), float(v45_data));
              float v51_sel0;
              if (v35_g) {
                float v49_data = s0[(v17_lead + 32_i32)];
                v51_sel0 = v49_data;
              }
              else {
                v51_sel0 = -INFINITY;
              }
              float v52_r = sycl::max(float(v46_r), float(v51_sel0));
              float v54_r = sycl::max(float(v52_r), float((sycl::permute_group_by_xor(item.get_sub_group(), v52_r, 1))));
              float v56_r = sycl::max(float(v54_r), float((sycl::permute_group_by_xor(item.get_sub_group(), v54_r, 2))));
              float v58_r = sycl::max(float(v56_r), float((sycl::permute_group_by_xor(item.get_sub_group(), v56_r, 4))));
              r1[0] = (sycl::max(float(v58_r), float((sycl::permute_group_by_xor(item.get_sub_group(), v58_r, 8)))));
              // glb_m1 = mul(glb_m0, r1)
              float v65_data = r1[0];
              #pragma unroll
              for (int32_t v61_k0 = 0; v61_k0 < 2; ++v61_k0) {
                int32_t v63_lead = v17_lead + (v61_k0 * 16);
                float v64_data = glb_m0[v63_lead];
                glb_m1[v63_lead] = ((v64_data * v65_data));
              }
              if (v35_g) {
                int32_t v68_lead = v17_lead + 32_i32;
                float v69_data = glb_m0[v68_lead];
                float v70_data = r1[0];
                glb_m1[v68_lead] = ((v69_data * v70_data));
              }
              sycl::group_barrier(item.get_sub_group());
            }
          }
        }
      });
    }
  });
}

