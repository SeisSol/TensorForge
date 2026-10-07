// === base name ===
kernel_25fc471b42091529

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_25fc471b42091529 = {{16, 16, 1}, 16, 16, 1, 16, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_25fc471b42091529(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_25fc471b42091529(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_25fc471b42091529(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 512 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_25fc471b42091529(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_25fc471b42091529(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_25fc471b42091529(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_25fc471b42091529(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (512, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 2048 B shared, occupancy grid
        // operands:
        //   m0 16(16) {0..16} strided
        //   m1 24×16(24×6) {0..24}×{0..6} strided
        //   m2 16(16) {0..16} strided
        // operations:
        //   t0[i] = m0[i]
        //   TMP = +(A, dims=[0])
        //   m2[i] = t0[i]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":512}],"shared_bytes":2048,"shared_elements":512,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"X","bbox":[[0],[16]],"name":"m0","ordered":false,"parts":1,"shape":[16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[24,6]],"name":"m1","ordered":false,"parts":1,"shape":[24,16],"variant":false},{"addressing":"strided","alias":"OUT","bbox":[[0],[16]],"name":"m2","ordered":false,"parts":1,"shape":[16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0],[16]],"is_tmp":true,"name":"t0","offset":[0],"shape":[16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0],[16]],"is_tmp":false,"name":"m0","offset":[0],"shape":[16]}],"permute":[[0]],"target":[[0]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0],[6]],"is_tmp":true,"name":"t0","offset":[0],"shape":[16]},"kind":"reduction","op":"+","ops":[{"addressing":"strided","bbox":[[0,0],[24,6]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[24,16]}],"permute":[[0,1]],"target":[[-1,0]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0],[16]],"is_tmp":false,"name":"m2","offset":[0],"shape":[16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0],[16]],"is_tmp":true,"name":"t0","offset":[0],"shape":[16]}],"permute":[[0]],"target":[[0]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[32 * item.get_local_id(1) + 0];
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 16 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 144 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 16 + 0 + m2_extraOffset];
              float r0[1]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v22_lead = item.get_local_id(2) % 16;
              #pragma unroll
              for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
                float v26_data = glb_m0[(v22_lead + (v23_i0 * 16))];
                r0[v23_i0] = v26_data;
              }
              float r1[1]{};
              // r1 = +(r0) + None
              // [(0, 16)] []
              float v28_data = r0[0];
              float v29_data = r1[0];
              r1[0] = (v29_data + v28_data);
              // s0 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v31_i0 = 0; v31_i0 < 1; ++v31_i0) {
                float v32_data = r1[v31_i0];
                s0[(v22_lead + (v31_i0 * 16))] = v32_data;
              }
              float r2[1]{};
              // r2 = +(glb_m1, dims=[0])
              float v39_data = glb_m1[v22_lead];
              bool v40_own = v22_lead < 8;
              float v46_sel0;
              if (v40_own) {
                float v44_data = glb_m1[(v22_lead + 16_i32)];
                v46_sel0 = v44_data;
              }
              else {
                v46_sel0 = 0.0f;
              }
              float v47_r = v39_data + v46_sel0;
              float v49_r = v47_r + (sycl::permute_group_by_xor(item.get_sub_group(), v47_r, 1));
              float v51_r = v49_r + (sycl::permute_group_by_xor(item.get_sub_group(), v49_r, 2));
              float v53_r = v51_r + (sycl::permute_group_by_xor(item.get_sub_group(), v51_r, 4));
              float v55_r = v53_r + (sycl::permute_group_by_xor(item.get_sub_group(), v53_r, 8));
              if (item.get_local_id(2) == 0) {
                r2[0] = v55_r;
              }
              float v57_data = glb_m1[(v22_lead + 24)];
              float v63_sel0;
              if (v40_own) {
                float v61_data = glb_m1[((v22_lead + 16_i32) + 24)];
                v63_sel0 = v61_data;
              }
              else {
                v63_sel0 = 0.0f;
              }
              float v64_r = v57_data + v63_sel0;
              float v66_r = v64_r + (sycl::permute_group_by_xor(item.get_sub_group(), v64_r, 1));
              float v68_r = v66_r + (sycl::permute_group_by_xor(item.get_sub_group(), v66_r, 2));
              float v70_r = v68_r + (sycl::permute_group_by_xor(item.get_sub_group(), v68_r, 4));
              float v72_r = v70_r + (sycl::permute_group_by_xor(item.get_sub_group(), v70_r, 8));
              if (item.get_local_id(2) == 1) {
                r2[0] = v72_r;
              }
              float v74_data = glb_m1[(v22_lead + 48)];
              float v80_sel0;
              if (v40_own) {
                float v78_data = glb_m1[((v22_lead + 16_i32) + 48)];
                v80_sel0 = v78_data;
              }
              else {
                v80_sel0 = 0.0f;
              }
              float v81_r = v74_data + v80_sel0;
              float v83_r = v81_r + (sycl::permute_group_by_xor(item.get_sub_group(), v81_r, 1));
              float v85_r = v83_r + (sycl::permute_group_by_xor(item.get_sub_group(), v83_r, 2));
              float v87_r = v85_r + (sycl::permute_group_by_xor(item.get_sub_group(), v85_r, 4));
              float v89_r = v87_r + (sycl::permute_group_by_xor(item.get_sub_group(), v87_r, 8));
              if (item.get_local_id(2) == 2) {
                r2[0] = v89_r;
              }
              float v91_data = glb_m1[(v22_lead + 72)];
              float v97_sel0;
              if (v40_own) {
                float v95_data = glb_m1[((v22_lead + 16_i32) + 72)];
                v97_sel0 = v95_data;
              }
              else {
                v97_sel0 = 0.0f;
              }
              float v98_r = v91_data + v97_sel0;
              float v100_r = v98_r + (sycl::permute_group_by_xor(item.get_sub_group(), v98_r, 1));
              float v102_r = v100_r + (sycl::permute_group_by_xor(item.get_sub_group(), v100_r, 2));
              float v104_r = v102_r + (sycl::permute_group_by_xor(item.get_sub_group(), v102_r, 4));
              float v106_r = v104_r + (sycl::permute_group_by_xor(item.get_sub_group(), v104_r, 8));
              if (item.get_local_id(2) == 3) {
                r2[0] = v106_r;
              }
              float v108_data = glb_m1[(v22_lead + 96)];
              float v114_sel0;
              if (v40_own) {
                float v112_data = glb_m1[((v22_lead + 16_i32) + 96)];
                v114_sel0 = v112_data;
              }
              else {
                v114_sel0 = 0.0f;
              }
              float v115_r = v108_data + v114_sel0;
              float v117_r = v115_r + (sycl::permute_group_by_xor(item.get_sub_group(), v115_r, 1));
              float v119_r = v117_r + (sycl::permute_group_by_xor(item.get_sub_group(), v117_r, 2));
              float v121_r = v119_r + (sycl::permute_group_by_xor(item.get_sub_group(), v119_r, 4));
              float v123_r = v121_r + (sycl::permute_group_by_xor(item.get_sub_group(), v121_r, 8));
              if (item.get_local_id(2) == 4) {
                r2[0] = v123_r;
              }
              float v125_data = glb_m1[(v22_lead + 120)];
              float v131_sel0;
              if (v40_own) {
                float v129_data = glb_m1[((v22_lead + 16_i32) + 120)];
                v131_sel0 = v129_data;
              }
              else {
                v131_sel0 = 0.0f;
              }
              float v132_r = v125_data + v131_sel0;
              float v134_r = v132_r + (sycl::permute_group_by_xor(item.get_sub_group(), v132_r, 1));
              float v136_r = v134_r + (sycl::permute_group_by_xor(item.get_sub_group(), v134_r, 2));
              float v138_r = v136_r + (sycl::permute_group_by_xor(item.get_sub_group(), v136_r, 4));
              float v140_r = v138_r + (sycl::permute_group_by_xor(item.get_sub_group(), v138_r, 8));
              if (item.get_local_id(2) == 5) {
                r2[0] = v140_r;
              }
              // s0 = store{r>s, clear}(localShrMem0, r2);
              sycl::group_barrier(item.get_sub_group());
              if (v22_lead >= 6) {
                s0[v22_lead] = 0.0f;
              }
              if (v22_lead < 6) {
                float v146_data = r2[0];
                s0[v22_lead] = v146_data;
              }
              float r3[1]{};
              // ir3 = +(s0)
              // [(0, 16)] []
              float ir3[1]{};
              sycl::group_barrier(item.get_sub_group());
              float v153_data = s0[v22_lead];
              float v154_data = ir3[0];
              ir3[0] = (v154_data + v153_data);
              // r3 = ir3
              #pragma unroll
              for (int32_t v156_n0 = 0; v156_n0 < 1; ++v156_n0) {
                float v157_data = ir3[v156_n0];
                r3[v156_n0] = v157_data;
              }
              // glb_m2 = store{r>g}(r3);
              #pragma unroll
              for (int32_t v158_i0 = 0; v158_i0 < 1; ++v158_i0) {
                float v159_data = r3[v158_i0];
                glb_m2[(v22_lead + (v158_i0 * 16))] = v159_data;
              }
              sycl::group_barrier(item.get_sub_group());
            }
          }
        }
      });
    }
  });
}

