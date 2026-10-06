// === base name ===
kernel_eaee8cafe6f34bcd

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_eaee8cafe6f34bcd = {{16, 16, 1}, 16, 16, 1, 16, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_eaee8cafe6f34bcd(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_eaee8cafe6f34bcd(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_eaee8cafe6f34bcd(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_eaee8cafe6f34bcd(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_eaee8cafe6f34bcd(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_eaee8cafe6f34bcd(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_eaee8cafe6f34bcd(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
          float* tempShrMem = &localShrMem0[16];
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v10_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v10_batchId0 < numElements0; v10_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v11_ahead1 = v10_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v13_batchId1 = (v11_ahead1 < numElements0) ? v11_ahead1 : v10_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v10_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v10_batchId0 * 16 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v10_batchId0 * 144 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v10_batchId0 * 16 + 0 + m2_extraOffset];
              float r0[1]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v24_lead = item.get_local_id(2) % 16;
              #pragma unroll
              for (int32_t v25_i0 = 0; v25_i0 < 1; ++v25_i0) {
                float v28_data = glb_m0[(v24_lead + (v25_i0 * 16))];
                r0[v25_i0] = v28_data;
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r1[1]{};
              // r1 = +(r0) + None
              // [(0, 16)] []
              float v30_data = r0[0];
              float v31_data = r1[0];
              r1[0] = (v31_data + v30_data);
              // s0 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v33_i0 = 0; v33_i0 < 1; ++v33_i0) {
                float v34_data = r1[v33_i0];
                s0[(v24_lead + (v33_i0 * 16))] = v34_data;
              }
              float r2[1]{};
              // r2 = +(glb_m1, dims=[0])
              float v41_data = glb_m1[v24_lead];
              bool v42_own = v24_lead < 8;
              float v48_sel0;
              if (v42_own) {
                float v46_data = glb_m1[(v24_lead + 16_i32)];
                v48_sel0 = v46_data;
              }
              else {
                v48_sel0 = 0.0f;
              }
              float v49_r = v41_data + v48_sel0;
              float v51_r = v49_r + (sycl::permute_group_by_xor(item.get_sub_group(), v49_r, 1));
              float v53_r = v51_r + (sycl::permute_group_by_xor(item.get_sub_group(), v51_r, 2));
              float v55_r = v53_r + (sycl::permute_group_by_xor(item.get_sub_group(), v53_r, 4));
              float v57_r = v55_r + (sycl::permute_group_by_xor(item.get_sub_group(), v55_r, 8));
              if (item.get_local_id(2) == 0) {
                r2[0] = v57_r;
              }
              float v59_data = glb_m1[(v24_lead + 24)];
              float v65_sel0;
              if (v42_own) {
                float v63_data = glb_m1[((v24_lead + 16_i32) + 24)];
                v65_sel0 = v63_data;
              }
              else {
                v65_sel0 = 0.0f;
              }
              float v66_r = v59_data + v65_sel0;
              float v68_r = v66_r + (sycl::permute_group_by_xor(item.get_sub_group(), v66_r, 1));
              float v70_r = v68_r + (sycl::permute_group_by_xor(item.get_sub_group(), v68_r, 2));
              float v72_r = v70_r + (sycl::permute_group_by_xor(item.get_sub_group(), v70_r, 4));
              float v74_r = v72_r + (sycl::permute_group_by_xor(item.get_sub_group(), v72_r, 8));
              if (item.get_local_id(2) == 1) {
                r2[0] = v74_r;
              }
              float v76_data = glb_m1[(v24_lead + 48)];
              float v82_sel0;
              if (v42_own) {
                float v80_data = glb_m1[((v24_lead + 16_i32) + 48)];
                v82_sel0 = v80_data;
              }
              else {
                v82_sel0 = 0.0f;
              }
              float v83_r = v76_data + v82_sel0;
              float v85_r = v83_r + (sycl::permute_group_by_xor(item.get_sub_group(), v83_r, 1));
              float v87_r = v85_r + (sycl::permute_group_by_xor(item.get_sub_group(), v85_r, 2));
              float v89_r = v87_r + (sycl::permute_group_by_xor(item.get_sub_group(), v87_r, 4));
              float v91_r = v89_r + (sycl::permute_group_by_xor(item.get_sub_group(), v89_r, 8));
              if (item.get_local_id(2) == 2) {
                r2[0] = v91_r;
              }
              float v93_data = glb_m1[(v24_lead + 72)];
              float v99_sel0;
              if (v42_own) {
                float v97_data = glb_m1[((v24_lead + 16_i32) + 72)];
                v99_sel0 = v97_data;
              }
              else {
                v99_sel0 = 0.0f;
              }
              float v100_r = v93_data + v99_sel0;
              float v102_r = v100_r + (sycl::permute_group_by_xor(item.get_sub_group(), v100_r, 1));
              float v104_r = v102_r + (sycl::permute_group_by_xor(item.get_sub_group(), v102_r, 2));
              float v106_r = v104_r + (sycl::permute_group_by_xor(item.get_sub_group(), v104_r, 4));
              float v108_r = v106_r + (sycl::permute_group_by_xor(item.get_sub_group(), v106_r, 8));
              if (item.get_local_id(2) == 3) {
                r2[0] = v108_r;
              }
              float v110_data = glb_m1[(v24_lead + 96)];
              float v116_sel0;
              if (v42_own) {
                float v114_data = glb_m1[((v24_lead + 16_i32) + 96)];
                v116_sel0 = v114_data;
              }
              else {
                v116_sel0 = 0.0f;
              }
              float v117_r = v110_data + v116_sel0;
              float v119_r = v117_r + (sycl::permute_group_by_xor(item.get_sub_group(), v117_r, 1));
              float v121_r = v119_r + (sycl::permute_group_by_xor(item.get_sub_group(), v119_r, 2));
              float v123_r = v121_r + (sycl::permute_group_by_xor(item.get_sub_group(), v121_r, 4));
              float v125_r = v123_r + (sycl::permute_group_by_xor(item.get_sub_group(), v123_r, 8));
              if (item.get_local_id(2) == 4) {
                r2[0] = v125_r;
              }
              float v127_data = glb_m1[(v24_lead + 120)];
              float v133_sel0;
              if (v42_own) {
                float v131_data = glb_m1[((v24_lead + 16_i32) + 120)];
                v133_sel0 = v131_data;
              }
              else {
                v133_sel0 = 0.0f;
              }
              float v134_r = v127_data + v133_sel0;
              float v136_r = v134_r + (sycl::permute_group_by_xor(item.get_sub_group(), v134_r, 1));
              float v138_r = v136_r + (sycl::permute_group_by_xor(item.get_sub_group(), v136_r, 2));
              float v140_r = v138_r + (sycl::permute_group_by_xor(item.get_sub_group(), v138_r, 4));
              float v142_r = v140_r + (sycl::permute_group_by_xor(item.get_sub_group(), v140_r, 8));
              if (item.get_local_id(2) == 5) {
                r2[0] = v142_r;
              }
              // s0 = store{r>s, clear}(localShrMem0, r2);
              if (v24_lead >= 6) {
                s0[v24_lead] = 0.0f;
              }
              if (v24_lead < 6) {
                float v148_data = r2[0];
                s0[v24_lead] = v148_data;
              }
              float r3[1]{};
              sycl::group_barrier(item.get_sub_group());
              // ir3 = +(s0)
              // [(0, 16)] []
              float ir3[1]{};
              float v155_data = s0[v24_lead];
              float v156_data = ir3[0];
              ir3[0] = (v156_data + v155_data);
              // r3 = ir3
              #pragma unroll
              for (int32_t v158_n0 = 0; v158_n0 < 1; ++v158_n0) {
                float v159_data = ir3[v158_n0];
                r3[v158_n0] = v159_data;
              }
              // glb_m2 = store{r>g}(r3);
              #pragma unroll
              for (int32_t v160_i0 = 0; v160_i0 < 1; ++v160_i0) {
                float v161_data = r3[v160_i0];
                glb_m2[(v24_lead + (v160_i0 * 16))] = v161_data;
              }
              sycl::group_barrier(item.get_sub_group());
            }
          }
        }
      });
    }
  });
}

