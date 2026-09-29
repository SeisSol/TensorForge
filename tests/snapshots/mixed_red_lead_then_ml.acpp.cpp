// === base name ===
kernel_f96c88af7f0709d9

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_f96c88af7f0709d9 = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_f96c88af7f0709d9(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_f96c88af7f0709d9(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_f96c88af7f0709d9(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_f96c88af7f0709d9(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_f96c88af7f0709d9(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_f96c88af7f0709d9(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f96c88af7f0709d9(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 24×6(24×6) {0..24}×{0..6} strided
        //   m1 6(6) {0..6} strided
        // operations:
        //   TMP = +(A, dims=[0])
        //   m1[i] = t0[i]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[24,6]],"name":"m0","ordered":false,"parts":1,"shape":[24,6],"variant":false},{"addressing":"strided","alias":"OUT","bbox":[[0],[6]],"name":"m1","ordered":false,"parts":1,"shape":[6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0],[6]],"is_tmp":true,"name":"t0","offset":[0],"shape":[6]},"kind":"reduction","op":"+","ops":[{"addressing":"strided","bbox":[[0,0],[24,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[24,6]}],"permute":[[0,1]],"target":[[-1,0]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0],[6]],"is_tmp":false,"name":"m1","offset":[0],"shape":[6]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0],[6]],"is_tmp":true,"name":"t0","offset":[0],"shape":[6]}],"permute":[[0]],"target":[[0]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t v3_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v3_batchId0 < numElements0; v3_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v4_ahead1 = v3_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v3_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v3_batchId0 * 144 + 0 + m0_extraOffset];
              float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 6 + 0 + m1_extraOffset];
              float r0[1]{};
              // r0 = +(glb_m0, dims=[0])
              int32_t v16_lead = item.get_local_id(2) % 16;
              float v20_data = glb_m0[v16_lead];
              bool v21_own = v16_lead < 8;
              float v27_sel0;
              if (v21_own) {
                float v25_data = glb_m0[(v16_lead + 16_i32)];
                v27_sel0 = v25_data;
              }
              else {
                v27_sel0 = 0.0f;
              }
              float v28_r = v20_data + v27_sel0;
              float v30_r = v28_r + (sycl::permute_group_by_xor(item.get_sub_group(), v28_r, 1));
              float v32_r = v30_r + (sycl::permute_group_by_xor(item.get_sub_group(), v30_r, 2));
              float v34_r = v32_r + (sycl::permute_group_by_xor(item.get_sub_group(), v32_r, 4));
              float v36_r = v34_r + (sycl::permute_group_by_xor(item.get_sub_group(), v34_r, 8));
              if (item.get_local_id(2) == 0) {
                r0[0] = v36_r;
              }
              float v38_data = glb_m0[(v16_lead + 24)];
              float v44_sel0;
              if (v21_own) {
                float v42_data = glb_m0[((v16_lead + 16_i32) + 24)];
                v44_sel0 = v42_data;
              }
              else {
                v44_sel0 = 0.0f;
              }
              float v45_r = v38_data + v44_sel0;
              float v47_r = v45_r + (sycl::permute_group_by_xor(item.get_sub_group(), v45_r, 1));
              float v49_r = v47_r + (sycl::permute_group_by_xor(item.get_sub_group(), v47_r, 2));
              float v51_r = v49_r + (sycl::permute_group_by_xor(item.get_sub_group(), v49_r, 4));
              float v53_r = v51_r + (sycl::permute_group_by_xor(item.get_sub_group(), v51_r, 8));
              if (item.get_local_id(2) == 1) {
                r0[0] = v53_r;
              }
              float v55_data = glb_m0[(v16_lead + 48)];
              float v61_sel0;
              if (v21_own) {
                float v59_data = glb_m0[((v16_lead + 16_i32) + 48)];
                v61_sel0 = v59_data;
              }
              else {
                v61_sel0 = 0.0f;
              }
              float v62_r = v55_data + v61_sel0;
              float v64_r = v62_r + (sycl::permute_group_by_xor(item.get_sub_group(), v62_r, 1));
              float v66_r = v64_r + (sycl::permute_group_by_xor(item.get_sub_group(), v64_r, 2));
              float v68_r = v66_r + (sycl::permute_group_by_xor(item.get_sub_group(), v66_r, 4));
              float v70_r = v68_r + (sycl::permute_group_by_xor(item.get_sub_group(), v68_r, 8));
              if (item.get_local_id(2) == 2) {
                r0[0] = v70_r;
              }
              float v72_data = glb_m0[(v16_lead + 72)];
              float v78_sel0;
              if (v21_own) {
                float v76_data = glb_m0[((v16_lead + 16_i32) + 72)];
                v78_sel0 = v76_data;
              }
              else {
                v78_sel0 = 0.0f;
              }
              float v79_r = v72_data + v78_sel0;
              float v81_r = v79_r + (sycl::permute_group_by_xor(item.get_sub_group(), v79_r, 1));
              float v83_r = v81_r + (sycl::permute_group_by_xor(item.get_sub_group(), v81_r, 2));
              float v85_r = v83_r + (sycl::permute_group_by_xor(item.get_sub_group(), v83_r, 4));
              float v87_r = v85_r + (sycl::permute_group_by_xor(item.get_sub_group(), v85_r, 8));
              if (item.get_local_id(2) == 3) {
                r0[0] = v87_r;
              }
              float v89_data = glb_m0[(v16_lead + 96)];
              float v95_sel0;
              if (v21_own) {
                float v93_data = glb_m0[((v16_lead + 16_i32) + 96)];
                v95_sel0 = v93_data;
              }
              else {
                v95_sel0 = 0.0f;
              }
              float v96_r = v89_data + v95_sel0;
              float v98_r = v96_r + (sycl::permute_group_by_xor(item.get_sub_group(), v96_r, 1));
              float v100_r = v98_r + (sycl::permute_group_by_xor(item.get_sub_group(), v98_r, 2));
              float v102_r = v100_r + (sycl::permute_group_by_xor(item.get_sub_group(), v100_r, 4));
              float v104_r = v102_r + (sycl::permute_group_by_xor(item.get_sub_group(), v102_r, 8));
              if (item.get_local_id(2) == 4) {
                r0[0] = v104_r;
              }
              float v106_data = glb_m0[(v16_lead + 120)];
              float v112_sel0;
              if (v21_own) {
                float v110_data = glb_m0[((v16_lead + 16_i32) + 120)];
                v112_sel0 = v110_data;
              }
              else {
                v112_sel0 = 0.0f;
              }
              float v113_r = v106_data + v112_sel0;
              float v115_r = v113_r + (sycl::permute_group_by_xor(item.get_sub_group(), v113_r, 1));
              float v117_r = v115_r + (sycl::permute_group_by_xor(item.get_sub_group(), v115_r, 2));
              float v119_r = v117_r + (sycl::permute_group_by_xor(item.get_sub_group(), v117_r, 4));
              float v121_r = v119_r + (sycl::permute_group_by_xor(item.get_sub_group(), v119_r, 8));
              if (item.get_local_id(2) == 5) {
                r0[0] = v121_r;
              }
              float r1[1]{};
              // ir1 = +(r0)
              // [(0, 6)] []
              float ir1[1]{};
              float v127_data = r0[0];
              float v128_data = ir1[0];
              ir1[0] = (v128_data + v127_data);
              // r1 = ir1
              if (v16_lead < 6) {
                float v134_data = ir1[0];
                r1[0] = v134_data;
              }
              // glb_m1 = store{r>g}(r1);
              if (v16_lead < 6) {
                float v139_data = r1[0];
                glb_m1[v16_lead] = v139_data;
              }
              sycl::group_barrier(item.get_sub_group());
            }
          }
        }
      });
    }
  });
}

