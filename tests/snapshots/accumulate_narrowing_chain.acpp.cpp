// === base name ===
kernel_0536996b29ebe54d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_0536996b29ebe54d = {{32, 1, 1}, 32, 20, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_0536996b29ebe54d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_0536996b29ebe54d(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_0536996b29ebe54d(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_0536996b29ebe54d(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_0536996b29ebe54d(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_0536996b29ebe54d(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_0536996b29ebe54d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes (20 active) x 1 per block = block 32x1x1, 0 B shared, occupancy grid
        // operands:
        //   m0 20×9(20×9) {0..20}×{0..9} strided
        //   m1 20×9(20×9) {0..20}×{0..9} strided
        //   m2 10×9(10×9) {0..10}×{0..9} strided
        //   m3 4×9(4×9) {0..4}×{0..9} strided
        //   m4 1×9(1×9) {0..1}×{0..9} strided
        // operations:
        //   m0[i,j] = m1[i,j]
        //   m0[i,j] += m2[i,j]
        //   m0[i,j] += m3[i,j]
        //   m0[i,j] += m4[i,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":20,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[20,9]],"name":"m0","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"strided","alias":"Q","bbox":[[0,0],[20,9]],"name":"m1","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"strided","alias":"F0","bbox":[[0,0],[10,9]],"name":"m2","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"strided","alias":"F1","bbox":[[0,0],[4,9]],"name":"m3","ordered":false,"parts":1,"shape":[4,9],"variant":false},{"addressing":"strided","alias":"F2","bbox":[[0,0],[1,9]],"name":"m4","ordered":false,"parts":1,"shape":[1,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[20,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[10,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[4,9]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[4,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[1,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[1,9]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1"}
        {
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 180 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 180 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 90 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v7_batchId0 * 36 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v7_batchId0 * 9 + 0 + m4_extraOffset];
              float r0[9]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v23_lead = item.get_local_id(2) % 32;
              bool v24_g = v23_lead < 20;
              if (v24_g) {
                #pragma unroll
                for (int32_t v25_i1 = 0; v25_i1 < 9; ++v25_i1) {
                  float v30_data = glb_m1[(v23_lead + (v25_i1 * 20))];
                  r0[v25_i1] = v30_data;
                }
              }
              float r2[9]{};
              // r2 = load{g>r}(glb_m2);
              if (v23_lead < 10) {
                #pragma unroll
                for (int32_t v66_i1 = 0; v66_i1 < 9; ++v66_i1) {
                  float v71_data = glb_m2[(v23_lead + (v66_i1 * 10))];
                  r2[v66_i1] = v71_data;
                }
              }
              float r1[9]{};
              // ir1 = +(r0)
              // [(0, 20), (0, 9)] []
              float ir1[9]{};
              float v34_data = r0[0];
              float v35_data = ir1[0];
              ir1[0] = (v35_data + v34_data);
              float v37_data = r0[1];
              float v38_data = ir1[1];
              ir1[1] = (v38_data + v37_data);
              float v40_data = r0[2];
              float v41_data = ir1[2];
              ir1[2] = (v41_data + v40_data);
              float v43_data = r0[3];
              float v44_data = ir1[3];
              ir1[3] = (v44_data + v43_data);
              float v46_data = r0[4];
              float v47_data = ir1[4];
              ir1[4] = (v47_data + v46_data);
              float v49_data = r0[5];
              float v50_data = ir1[5];
              ir1[5] = (v50_data + v49_data);
              float v52_data = r0[6];
              float v53_data = ir1[6];
              ir1[6] = (v53_data + v52_data);
              float v55_data = r0[7];
              float v56_data = ir1[7];
              ir1[7] = (v56_data + v55_data);
              float v58_data = r0[8];
              float v59_data = ir1[8];
              ir1[8] = (v59_data + v58_data);
              // r1 = ir1
              if (v24_g) {
                #pragma unroll
                for (int32_t v61_n1 = 0; v61_n1 < 9; ++v61_n1) {
                  float v63_data = ir1[v61_n1];
                  r1[v61_n1] = v63_data;
                }
              }
              float r4[9]{};
              // r4 = load{g>r}(glb_m3);
              if (v23_lead < 4) {
                #pragma unroll
                for (int32_t v109_i1 = 0; v109_i1 < 9; ++v109_i1) {
                  float v114_data = glb_m3[(v23_lead + (v109_i1 * 4))];
                  r4[v109_i1] = v114_data;
                }
              }
              float r3[9]{};
              // ir3 = +(r2)
              // [(0, 10), (0, 9)] []
              float ir3[9]{};
              float v75_data = r2[0];
              float v76_data = ir3[0];
              ir3[0] = (v76_data + v75_data);
              float v78_data = r2[1];
              float v79_data = ir3[1];
              ir3[1] = (v79_data + v78_data);
              float v81_data = r2[2];
              float v82_data = ir3[2];
              ir3[2] = (v82_data + v81_data);
              float v84_data = r2[3];
              float v85_data = ir3[3];
              ir3[3] = (v85_data + v84_data);
              float v87_data = r2[4];
              float v88_data = ir3[4];
              ir3[4] = (v88_data + v87_data);
              float v90_data = r2[5];
              float v91_data = ir3[5];
              ir3[5] = (v91_data + v90_data);
              float v93_data = r2[6];
              float v94_data = ir3[6];
              ir3[6] = (v94_data + v93_data);
              float v96_data = r2[7];
              float v97_data = ir3[7];
              ir3[7] = (v97_data + v96_data);
              float v99_data = r2[8];
              float v100_data = ir3[8];
              ir3[8] = (v100_data + v99_data);
              // r3 = ir3 + r1
              if (v24_g) {
                #pragma unroll
                for (int32_t v102_n1 = 0; v102_n1 < 9; ++v102_n1) {
                  float v104_data = ir3[v102_n1];
                  float v105_data = r1[v102_n1];
                  r3[v102_n1] = (v105_data + v104_data);
                }
              }
              float r6[9]{};
              // r6 = load{g>r}(glb_m4);
              if (v23_lead < 1) {
                #pragma unroll
                for (int32_t v152_i1 = 0; v152_i1 < 9; ++v152_i1) {
                  float v156_data = glb_m4[(v23_lead + v152_i1)];
                  r6[v152_i1] = v156_data;
                }
              }
              float r5[9]{};
              // ir5 = +(r4)
              // [(0, 4), (0, 9)] []
              float ir5[9]{};
              float v118_data = r4[0];
              float v119_data = ir5[0];
              ir5[0] = (v119_data + v118_data);
              float v121_data = r4[1];
              float v122_data = ir5[1];
              ir5[1] = (v122_data + v121_data);
              float v124_data = r4[2];
              float v125_data = ir5[2];
              ir5[2] = (v125_data + v124_data);
              float v127_data = r4[3];
              float v128_data = ir5[3];
              ir5[3] = (v128_data + v127_data);
              float v130_data = r4[4];
              float v131_data = ir5[4];
              ir5[4] = (v131_data + v130_data);
              float v133_data = r4[5];
              float v134_data = ir5[5];
              ir5[5] = (v134_data + v133_data);
              float v136_data = r4[6];
              float v137_data = ir5[6];
              ir5[6] = (v137_data + v136_data);
              float v139_data = r4[7];
              float v140_data = ir5[7];
              ir5[7] = (v140_data + v139_data);
              float v142_data = r4[8];
              float v143_data = ir5[8];
              ir5[8] = (v143_data + v142_data);
              // r5 = ir5 + r3
              if (v24_g) {
                #pragma unroll
                for (int32_t v145_n1 = 0; v145_n1 < 9; ++v145_n1) {
                  float v147_data = ir5[v145_n1];
                  float v148_data = r3[v145_n1];
                  r5[v145_n1] = (v148_data + v147_data);
                }
              }
              float r7[9]{};
              // ir7 = +(r6)
              // [(0, 1), (0, 9)] []
              float ir7[9]{};
              float v160_data = r6[0];
              float v161_data = ir7[0];
              ir7[0] = (v161_data + v160_data);
              float v163_data = r6[1];
              float v164_data = ir7[1];
              ir7[1] = (v164_data + v163_data);
              float v166_data = r6[2];
              float v167_data = ir7[2];
              ir7[2] = (v167_data + v166_data);
              float v169_data = r6[3];
              float v170_data = ir7[3];
              ir7[3] = (v170_data + v169_data);
              float v172_data = r6[4];
              float v173_data = ir7[4];
              ir7[4] = (v173_data + v172_data);
              float v175_data = r6[5];
              float v176_data = ir7[5];
              ir7[5] = (v176_data + v175_data);
              float v178_data = r6[6];
              float v179_data = ir7[6];
              ir7[6] = (v179_data + v178_data);
              float v181_data = r6[7];
              float v182_data = ir7[7];
              ir7[7] = (v182_data + v181_data);
              float v184_data = r6[8];
              float v185_data = ir7[8];
              ir7[8] = (v185_data + v184_data);
              // r7 = ir7 + r5
              if (v24_g) {
                #pragma unroll
                for (int32_t v187_n1 = 0; v187_n1 < 9; ++v187_n1) {
                  float v189_data = ir7[v187_n1];
                  float v190_data = r5[v187_n1];
                  r7[v187_n1] = (v190_data + v189_data);
                }
              }
              // glb_m0 = store{r>g}(r7);
              if (v24_g) {
                #pragma unroll
                for (int32_t v192_i1 = 0; v192_i1 < 9; ++v192_i1) {
                  float v194_data = r7[v192_i1];
                  glb_m0[(v23_lead + (v192_i1 * 20))] = v194_data;
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

