// === base name ===
kernel_18cc65ed20e77d84

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_18cc65ed20e77d84 = {{16, 16, 1}, 16, 16, 1, 16, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_18cc65ed20e77d84(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_18cc65ed20e77d84(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_18cc65ed20e77d84(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_18cc65ed20e77d84(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_18cc65ed20e77d84(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_18cc65ed20e77d84(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_18cc65ed20e77d84(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (512, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
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
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":512}],"shared_bytes":2048,"shared_elements":512,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"X","bbox":[[0],[16]],"name":"m0","ordered":false,"parts":1,"shape":[16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[24,6]],"name":"m1","ordered":false,"parts":1,"shape":[24,16],"variant":false},{"addressing":"strided","alias":"OUT","bbox":[[0],[16]],"name":"m2","ordered":false,"parts":1,"shape":[16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0],[16]],"is_tmp":true,"name":"t0","offset":[0],"shape":[16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0],[16]],"is_tmp":false,"name":"m0","offset":[0],"shape":[16]}],"permute":[[0]],"target":[[0]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0],[6]],"is_tmp":true,"name":"t0","offset":[0],"shape":[16]},"kind":"reduction","op":"+","ops":[{"addressing":"strided","bbox":[[0,0],[24,6]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[24,16]}],"permute":[[0,1]],"target":[[-1,0]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0],[16]],"is_tmp":false,"name":"m2","offset":[0],"shape":[16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0],[16]],"is_tmp":true,"name":"t0","offset":[0],"shape":[16]}],"permute":[[0]],"target":[[0]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[32 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[16];
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v4_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v4_batchId0 < numElements0; v4_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v5_ahead1 = v4_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v4_batchId0 * 16 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v4_batchId0 * 144 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v4_batchId0 * 16 + 0 + m2_extraOffset];
              float r0[1]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v18_lead = item.get_local_id(2) % 16;
              #pragma unroll
              for (int32_t v19_i0 = 0; v19_i0 < 1; ++v19_i0) {
                float v22_data = glb_m0[(v18_lead + (v19_i0 * 16))];
                r0[v19_i0] = v22_data;
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r1[1]{};
              // r1 = +(r0) + None
              // [(0, 16)] []
              float v24_data = r0[0];
              float v25_data = r1[0];
              r1[0] = (v25_data + v24_data);
              // s0 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v27_i0 = 0; v27_i0 < 1; ++v27_i0) {
                float v28_data = r1[v27_i0];
                s0[(v18_lead + (v27_i0 * 16))] = v28_data;
              }
              float r2[1]{};
              // r2 = +(glb_m1, dims=[0])
              float v35_data = glb_m1[v18_lead];
              bool v36_own = v18_lead < 8;
              float v42_sel0;
              if (v36_own) {
                float v40_data = glb_m1[(v18_lead + 16_i32)];
                v42_sel0 = v40_data;
              }
              else {
                v42_sel0 = 0.0f;
              }
              float v43_r = v35_data + v42_sel0;
              float v45_r = v43_r + (sycl::permute_group_by_xor(item.get_sub_group(), v43_r, 1));
              float v47_r = v45_r + (sycl::permute_group_by_xor(item.get_sub_group(), v45_r, 2));
              float v49_r = v47_r + (sycl::permute_group_by_xor(item.get_sub_group(), v47_r, 4));
              float v51_r = v49_r + (sycl::permute_group_by_xor(item.get_sub_group(), v49_r, 8));
              if (item.get_local_id(2) == 0) {
                r2[0] = v51_r;
              }
              float v53_data = glb_m1[(v18_lead + 24)];
              float v59_sel0;
              if (v36_own) {
                float v57_data = glb_m1[((v18_lead + 16_i32) + 24)];
                v59_sel0 = v57_data;
              }
              else {
                v59_sel0 = 0.0f;
              }
              float v60_r = v53_data + v59_sel0;
              float v62_r = v60_r + (sycl::permute_group_by_xor(item.get_sub_group(), v60_r, 1));
              float v64_r = v62_r + (sycl::permute_group_by_xor(item.get_sub_group(), v62_r, 2));
              float v66_r = v64_r + (sycl::permute_group_by_xor(item.get_sub_group(), v64_r, 4));
              float v68_r = v66_r + (sycl::permute_group_by_xor(item.get_sub_group(), v66_r, 8));
              if (item.get_local_id(2) == 1) {
                r2[0] = v68_r;
              }
              float v70_data = glb_m1[(v18_lead + 48)];
              float v76_sel0;
              if (v36_own) {
                float v74_data = glb_m1[((v18_lead + 16_i32) + 48)];
                v76_sel0 = v74_data;
              }
              else {
                v76_sel0 = 0.0f;
              }
              float v77_r = v70_data + v76_sel0;
              float v79_r = v77_r + (sycl::permute_group_by_xor(item.get_sub_group(), v77_r, 1));
              float v81_r = v79_r + (sycl::permute_group_by_xor(item.get_sub_group(), v79_r, 2));
              float v83_r = v81_r + (sycl::permute_group_by_xor(item.get_sub_group(), v81_r, 4));
              float v85_r = v83_r + (sycl::permute_group_by_xor(item.get_sub_group(), v83_r, 8));
              if (item.get_local_id(2) == 2) {
                r2[0] = v85_r;
              }
              float v87_data = glb_m1[(v18_lead + 72)];
              float v93_sel0;
              if (v36_own) {
                float v91_data = glb_m1[((v18_lead + 16_i32) + 72)];
                v93_sel0 = v91_data;
              }
              else {
                v93_sel0 = 0.0f;
              }
              float v94_r = v87_data + v93_sel0;
              float v96_r = v94_r + (sycl::permute_group_by_xor(item.get_sub_group(), v94_r, 1));
              float v98_r = v96_r + (sycl::permute_group_by_xor(item.get_sub_group(), v96_r, 2));
              float v100_r = v98_r + (sycl::permute_group_by_xor(item.get_sub_group(), v98_r, 4));
              float v102_r = v100_r + (sycl::permute_group_by_xor(item.get_sub_group(), v100_r, 8));
              if (item.get_local_id(2) == 3) {
                r2[0] = v102_r;
              }
              float v104_data = glb_m1[(v18_lead + 96)];
              float v110_sel0;
              if (v36_own) {
                float v108_data = glb_m1[((v18_lead + 16_i32) + 96)];
                v110_sel0 = v108_data;
              }
              else {
                v110_sel0 = 0.0f;
              }
              float v111_r = v104_data + v110_sel0;
              float v113_r = v111_r + (sycl::permute_group_by_xor(item.get_sub_group(), v111_r, 1));
              float v115_r = v113_r + (sycl::permute_group_by_xor(item.get_sub_group(), v113_r, 2));
              float v117_r = v115_r + (sycl::permute_group_by_xor(item.get_sub_group(), v115_r, 4));
              float v119_r = v117_r + (sycl::permute_group_by_xor(item.get_sub_group(), v117_r, 8));
              if (item.get_local_id(2) == 4) {
                r2[0] = v119_r;
              }
              float v121_data = glb_m1[(v18_lead + 120)];
              float v127_sel0;
              if (v36_own) {
                float v125_data = glb_m1[((v18_lead + 16_i32) + 120)];
                v127_sel0 = v125_data;
              }
              else {
                v127_sel0 = 0.0f;
              }
              float v128_r = v121_data + v127_sel0;
              float v130_r = v128_r + (sycl::permute_group_by_xor(item.get_sub_group(), v128_r, 1));
              float v132_r = v130_r + (sycl::permute_group_by_xor(item.get_sub_group(), v130_r, 2));
              float v134_r = v132_r + (sycl::permute_group_by_xor(item.get_sub_group(), v132_r, 4));
              float v136_r = v134_r + (sycl::permute_group_by_xor(item.get_sub_group(), v134_r, 8));
              if (item.get_local_id(2) == 5) {
                r2[0] = v136_r;
              }
              // s0 = store{r>s, clear}(localShrMem0, r2);
              if (v18_lead >= 6) {
                s0[v18_lead] = 0.0f;
              }
              if (v18_lead < 6) {
                float v142_data = r2[0];
                s0[v18_lead] = v142_data;
              }
              float r3[1]{};
              sycl::group_barrier(item.get_sub_group());
              // ir3 = +(s0)
              // [(0, 16)] []
              float ir3[1]{};
              float v149_data = s0[v18_lead];
              float v150_data = ir3[0];
              ir3[0] = (v150_data + v149_data);
              // r3 = ir3
              #pragma unroll
              for (int32_t v152_n0 = 0; v152_n0 < 1; ++v152_n0) {
                float v153_data = ir3[v152_n0];
                r3[v152_n0] = v153_data;
              }
              // glb_m2 = store{r>g}(r3);
              #pragma unroll
              for (int32_t v154_i0 = 0; v154_i0 < 1; ++v154_i0) {
                float v155_data = r3[v154_i0];
                glb_m2[(v18_lead + (v154_i0 * 16))] = v155_data;
              }
              sycl::group_barrier(item.get_sub_group());
            }
          }
        }
      });
    }
  });
}

