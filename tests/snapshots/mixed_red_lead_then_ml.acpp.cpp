// === base name ===
kernel_52c7c09f1c922251

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_52c7c09f1c922251 = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_52c7c09f1c922251(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_52c7c09f1c922251(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_52c7c09f1c922251(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_52c7c09f1c922251(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_52c7c09f1c922251(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_52c7c09f1c922251(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_52c7c09f1c922251(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 24×6(24×6) {0..24}×{0..6} strided
        //   m1 6(6) {0..6} strided
        // operations:
        //   TMP = +(A, dims=[0])
        //   m1[i] = t0[i]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[24,6]],"name":"m0","ordered":false,"parts":1,"shape":[24,6],"variant":false},{"addressing":"strided","alias":"OUT","bbox":[[0],[6]],"name":"m1","ordered":false,"parts":1,"shape":[6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0],[6]],"is_tmp":true,"name":"t0","offset":[0],"shape":[6]},"kind":"reduction","op":"+","ops":[{"addressing":"strided","bbox":[[0,0],[24,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[24,6]}],"permute":[[0,1]],"target":[[-1,0]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0],[6]],"is_tmp":false,"name":"m1","offset":[0],"shape":[6]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0],[6]],"is_tmp":true,"name":"t0","offset":[0],"shape":[6]}],"permute":[[0]],"target":[[0]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 144 + 0 + m0_extraOffset];
              float *const __restrict__ glb_m1 = &m1[v9_batchId0 * 6 + 0 + m1_extraOffset];
              float r0[1]{};
              // r0 = +(glb_m0, dims=[0])
              int32_t v22_lead = item.get_local_id(2) % 16;
              float v26_data = glb_m0[v22_lead];
              bool v27_own = v22_lead < 8;
              float v33_sel0;
              if (v27_own) {
                float v31_data = glb_m0[(v22_lead + 16_i32)];
                v33_sel0 = v31_data;
              }
              else {
                v33_sel0 = 0.0f;
              }
              float v34_r = v26_data + v33_sel0;
              float v36_r = v34_r + (sycl::permute_group_by_xor(item.get_sub_group(), v34_r, 1));
              float v38_r = v36_r + (sycl::permute_group_by_xor(item.get_sub_group(), v36_r, 2));
              float v40_r = v38_r + (sycl::permute_group_by_xor(item.get_sub_group(), v38_r, 4));
              float v42_r = v40_r + (sycl::permute_group_by_xor(item.get_sub_group(), v40_r, 8));
              if (item.get_local_id(2) == 0) {
                r0[0] = v42_r;
              }
              float v44_data = glb_m0[(v22_lead + 24)];
              float v50_sel0;
              if (v27_own) {
                float v48_data = glb_m0[((v22_lead + 16_i32) + 24)];
                v50_sel0 = v48_data;
              }
              else {
                v50_sel0 = 0.0f;
              }
              float v51_r = v44_data + v50_sel0;
              float v53_r = v51_r + (sycl::permute_group_by_xor(item.get_sub_group(), v51_r, 1));
              float v55_r = v53_r + (sycl::permute_group_by_xor(item.get_sub_group(), v53_r, 2));
              float v57_r = v55_r + (sycl::permute_group_by_xor(item.get_sub_group(), v55_r, 4));
              float v59_r = v57_r + (sycl::permute_group_by_xor(item.get_sub_group(), v57_r, 8));
              if (item.get_local_id(2) == 1) {
                r0[0] = v59_r;
              }
              float v61_data = glb_m0[(v22_lead + 48)];
              float v67_sel0;
              if (v27_own) {
                float v65_data = glb_m0[((v22_lead + 16_i32) + 48)];
                v67_sel0 = v65_data;
              }
              else {
                v67_sel0 = 0.0f;
              }
              float v68_r = v61_data + v67_sel0;
              float v70_r = v68_r + (sycl::permute_group_by_xor(item.get_sub_group(), v68_r, 1));
              float v72_r = v70_r + (sycl::permute_group_by_xor(item.get_sub_group(), v70_r, 2));
              float v74_r = v72_r + (sycl::permute_group_by_xor(item.get_sub_group(), v72_r, 4));
              float v76_r = v74_r + (sycl::permute_group_by_xor(item.get_sub_group(), v74_r, 8));
              if (item.get_local_id(2) == 2) {
                r0[0] = v76_r;
              }
              float v78_data = glb_m0[(v22_lead + 72)];
              float v84_sel0;
              if (v27_own) {
                float v82_data = glb_m0[((v22_lead + 16_i32) + 72)];
                v84_sel0 = v82_data;
              }
              else {
                v84_sel0 = 0.0f;
              }
              float v85_r = v78_data + v84_sel0;
              float v87_r = v85_r + (sycl::permute_group_by_xor(item.get_sub_group(), v85_r, 1));
              float v89_r = v87_r + (sycl::permute_group_by_xor(item.get_sub_group(), v87_r, 2));
              float v91_r = v89_r + (sycl::permute_group_by_xor(item.get_sub_group(), v89_r, 4));
              float v93_r = v91_r + (sycl::permute_group_by_xor(item.get_sub_group(), v91_r, 8));
              if (item.get_local_id(2) == 3) {
                r0[0] = v93_r;
              }
              float v95_data = glb_m0[(v22_lead + 96)];
              float v101_sel0;
              if (v27_own) {
                float v99_data = glb_m0[((v22_lead + 16_i32) + 96)];
                v101_sel0 = v99_data;
              }
              else {
                v101_sel0 = 0.0f;
              }
              float v102_r = v95_data + v101_sel0;
              float v104_r = v102_r + (sycl::permute_group_by_xor(item.get_sub_group(), v102_r, 1));
              float v106_r = v104_r + (sycl::permute_group_by_xor(item.get_sub_group(), v104_r, 2));
              float v108_r = v106_r + (sycl::permute_group_by_xor(item.get_sub_group(), v106_r, 4));
              float v110_r = v108_r + (sycl::permute_group_by_xor(item.get_sub_group(), v108_r, 8));
              if (item.get_local_id(2) == 4) {
                r0[0] = v110_r;
              }
              float v112_data = glb_m0[(v22_lead + 120)];
              float v118_sel0;
              if (v27_own) {
                float v116_data = glb_m0[((v22_lead + 16_i32) + 120)];
                v118_sel0 = v116_data;
              }
              else {
                v118_sel0 = 0.0f;
              }
              float v119_r = v112_data + v118_sel0;
              float v121_r = v119_r + (sycl::permute_group_by_xor(item.get_sub_group(), v119_r, 1));
              float v123_r = v121_r + (sycl::permute_group_by_xor(item.get_sub_group(), v121_r, 2));
              float v125_r = v123_r + (sycl::permute_group_by_xor(item.get_sub_group(), v123_r, 4));
              float v127_r = v125_r + (sycl::permute_group_by_xor(item.get_sub_group(), v125_r, 8));
              if (item.get_local_id(2) == 5) {
                r0[0] = v127_r;
              }
              float r1[1]{};
              // ir1 = +(r0)
              // [(0, 6)] []
              float ir1[1]{};
              float v133_data = r0[0];
              float v134_data = ir1[0];
              ir1[0] = (v134_data + v133_data);
              // r1 = ir1
              if (v22_lead < 6) {
                float v140_data = ir1[0];
                r1[0] = v140_data;
              }
              // glb_m1 = store{r>g}(r1);
              if (v22_lead < 6) {
                float v145_data = r1[0];
                glb_m1[v22_lead] = v145_data;
              }
              sycl::group_barrier(item.get_sub_group());
            }
          }
        }
      });
    }
  });
}

