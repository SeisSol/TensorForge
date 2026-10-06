// === base name ===
kernel_e058bfa68350e6ad

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_e058bfa68350e6ad = {{32, 1, 1}, 32, 64, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_e058bfa68350e6ad(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_e058bfa68350e6ad(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_e058bfa68350e6ad(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_e058bfa68350e6ad(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_e058bfa68350e6ad(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_e058bfa68350e6ad(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_e058bfa68350e6ad(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes (64 active) x 1 per block = block 32x1x1, 0 B shared, occupancy grid
        // operands:
        //   m0 64×19(64×19) {0..64}×{0..19} strided
        //   m1 64×19(64×19) {0..64}×{0..19} strided
        // operations:
        //   m0[i,j]@{0..64}×{0..17} += m1[i,j]@{0..64}×{0..17}
        //   t = max(IA, I)
        //   m0[i,j]@{0..64}×{17..19} = t0[i,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":64,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"IA","bbox":[[0,0],[64,19]],"name":"m0","ordered":false,"parts":1,"shape":[64,19],"variant":false},{"addressing":"strided","alias":"I","bbox":[[0,0],[64,19]],"name":"m1","ordered":false,"parts":1,"shape":[64,19],"variant":false}],"operations":[{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[64,17]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[64,19]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[64,17]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[64,19]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[64,2]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[64,2]},"kind":"elementwise","op":"MAX","ops":[{"addressing":"strided","bbox":[[0,0],[64,2]],"is_tmp":false,"name":"m0","offset":[0,17],"shape":[64,19]},{"addressing":"strided","bbox":[[0,0],[64,2]],"is_tmp":false,"name":"m1","offset":[0,17],"shape":[64,19]}],"permute":[[0,1],[0,1]],"scalars":[],"target":[[0,1],[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[64,2]],"is_tmp":false,"name":"m0","offset":[0,17],"shape":[64,19]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[64,2]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[64,2]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1"}
        {
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 1216 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 1216 + 0 + m1_extraOffset];
              float r0[38]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v20_lead = item.get_local_id(2) % 32;
              #pragma unroll
              for (int32_t v21_i0 = 0; v21_i0 < 2; ++v21_i0) {
                int32_t v24_lead = v20_lead + (v21_i0 * 32);
                #pragma unroll
                for (int32_t v22_i1 = 0; v22_i1 < 19; ++v22_i1) {
                  float v27_data = glb_m1[(v24_lead + (v22_i1 * 64))];
                  r0[(v21_i0 + (v22_i1 * 2))] = v27_data;
                }
              }
              float r1[34]{};
              // r1 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v31_i0 = 0; v31_i0 < 2; ++v31_i0) {
                int32_t v34_lead = v20_lead + (v31_i0 * 32);
                #pragma unroll
                for (int32_t v32_i1 = 0; v32_i1 < 17; ++v32_i1) {
                  float v37_data = glb_m0[(v34_lead + (v32_i1 * 64))];
                  r1[(v31_i0 + (v32_i1 * 2))] = v37_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m0););
              float r2[34]{};
              // ir2 = +(r0)
              // [(0, 64), (0, 17)] []
              float ir2[34]{};
              float v42_data = r0[0];
              float v43_data = ir2[0];
              ir2[0] = (v43_data + v42_data);
              float v45_data = r0[2];
              float v46_data = ir2[2];
              ir2[2] = (v46_data + v45_data);
              float v48_data = r0[4];
              float v49_data = ir2[4];
              ir2[4] = (v49_data + v48_data);
              float v51_data = r0[6];
              float v52_data = ir2[6];
              ir2[6] = (v52_data + v51_data);
              float v54_data = r0[8];
              float v55_data = ir2[8];
              ir2[8] = (v55_data + v54_data);
              float v57_data = r0[10];
              float v58_data = ir2[10];
              ir2[10] = (v58_data + v57_data);
              float v60_data = r0[12];
              float v61_data = ir2[12];
              ir2[12] = (v61_data + v60_data);
              float v63_data = r0[14];
              float v64_data = ir2[14];
              ir2[14] = (v64_data + v63_data);
              float v66_data = r0[16];
              float v67_data = ir2[16];
              ir2[16] = (v67_data + v66_data);
              float v69_data = r0[18];
              float v70_data = ir2[18];
              ir2[18] = (v70_data + v69_data);
              float v72_data = r0[20];
              float v73_data = ir2[20];
              ir2[20] = (v73_data + v72_data);
              float v75_data = r0[22];
              float v76_data = ir2[22];
              ir2[22] = (v76_data + v75_data);
              float v78_data = r0[24];
              float v79_data = ir2[24];
              ir2[24] = (v79_data + v78_data);
              float v81_data = r0[26];
              float v82_data = ir2[26];
              ir2[26] = (v82_data + v81_data);
              float v84_data = r0[28];
              float v85_data = ir2[28];
              ir2[28] = (v85_data + v84_data);
              float v87_data = r0[30];
              float v88_data = ir2[30];
              ir2[30] = (v88_data + v87_data);
              float v90_data = r0[32];
              float v91_data = ir2[32];
              ir2[32] = (v91_data + v90_data);
              float v93_data = r0[1];
              float v94_data = ir2[1];
              ir2[1] = (v94_data + v93_data);
              float v96_data = r0[3];
              float v97_data = ir2[3];
              ir2[3] = (v97_data + v96_data);
              float v99_data = r0[5];
              float v100_data = ir2[5];
              ir2[5] = (v100_data + v99_data);
              float v102_data = r0[7];
              float v103_data = ir2[7];
              ir2[7] = (v103_data + v102_data);
              float v105_data = r0[9];
              float v106_data = ir2[9];
              ir2[9] = (v106_data + v105_data);
              float v108_data = r0[11];
              float v109_data = ir2[11];
              ir2[11] = (v109_data + v108_data);
              float v111_data = r0[13];
              float v112_data = ir2[13];
              ir2[13] = (v112_data + v111_data);
              float v114_data = r0[15];
              float v115_data = ir2[15];
              ir2[15] = (v115_data + v114_data);
              float v117_data = r0[17];
              float v118_data = ir2[17];
              ir2[17] = (v118_data + v117_data);
              float v120_data = r0[19];
              float v121_data = ir2[19];
              ir2[19] = (v121_data + v120_data);
              float v123_data = r0[21];
              float v124_data = ir2[21];
              ir2[21] = (v124_data + v123_data);
              float v126_data = r0[23];
              float v127_data = ir2[23];
              ir2[23] = (v127_data + v126_data);
              float v129_data = r0[25];
              float v130_data = ir2[25];
              ir2[25] = (v130_data + v129_data);
              float v132_data = r0[27];
              float v133_data = ir2[27];
              ir2[27] = (v133_data + v132_data);
              float v135_data = r0[29];
              float v136_data = ir2[29];
              ir2[29] = (v136_data + v135_data);
              float v138_data = r0[31];
              float v139_data = ir2[31];
              ir2[31] = (v139_data + v138_data);
              float v141_data = r0[33];
              float v142_data = ir2[33];
              ir2[33] = (v142_data + v141_data);
              // r2 = ir2 + r1
              #pragma unroll
              for (int32_t v144_n0 = 0; v144_n0 < 2; ++v144_n0) {
                #pragma unroll
                for (int32_t v145_n1 = 0; v145_n1 < 17; ++v145_n1) {
                  int32_t v147_a = v144_n0 + (v145_n1 * 2);
                  float v148_data = ir2[v147_a];
                  float v149_data = r1[v147_a];
                  r2[v147_a] = (v149_data + v148_data);
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v151_i0 = 0; v151_i0 < 2; ++v151_i0) {
                int32_t v157_lead = v20_lead + (v151_i0 * 32);
                #pragma unroll
                for (int32_t v152_i1 = 0; v152_i1 < 17; ++v152_i1) {
                  float v155_data = r2[(v151_i0 + (v152_i1 * 2))];
                  glb_m0[(v157_lead + (v152_i1 * 64))] = v155_data;
                }
              }
              float r3[4]{};
              // r3 = max(glb_m0, glb_m1)
              #pragma unroll
              for (int32_t v161_k0 = 0; v161_k0 < 2; ++v161_k0) {
                int32_t v164_lead = v20_lead + (v161_k0 * 32);
                #pragma unroll
                for (int32_t v162_k1 = 0; v162_k1 < 2; ++v162_k1) {
                  int32_t v167_a = v164_lead + ((v162_k1 + 17) * 64);
                  float v168_data = glb_m0[v167_a];
                  float v169_data = glb_m1[v167_a];
                  r3[(v161_k0 + (v162_k1 * 2))] = (sycl::max(float(v168_data), float(v169_data)));
                }
              }
              float r4[4]{};
              // ir4 = +(r3)
              // [(0, 64), (0, 2)] []
              float ir4[4]{};
              float v175_data = r3[0];
              float v176_data = ir4[0];
              ir4[0] = (v176_data + v175_data);
              float v178_data = r3[2];
              float v179_data = ir4[2];
              ir4[2] = (v179_data + v178_data);
              float v181_data = r3[1];
              float v182_data = ir4[1];
              ir4[1] = (v182_data + v181_data);
              float v184_data = r3[3];
              float v185_data = ir4[3];
              ir4[3] = (v185_data + v184_data);
              // r4 = ir4
              #pragma unroll
              for (int32_t v187_n0 = 0; v187_n0 < 2; ++v187_n0) {
                #pragma unroll
                for (int32_t v188_n1 = 0; v188_n1 < 2; ++v188_n1) {
                  int32_t v190_a = v187_n0 + (v188_n1 * 2);
                  float v191_data = ir4[v190_a];
                  r4[v190_a] = v191_data;
                }
              }
              // glb_m0 = store{r>g}(r4);
              #pragma unroll
              for (int32_t v192_i0 = 0; v192_i0 < 2; ++v192_i0) {
                int32_t v198_lead = v20_lead + (v192_i0 * 32);
                #pragma unroll
                for (int32_t v193_i1 = 0; v193_i1 < 2; ++v193_i1) {
                  float v196_data = r4[(v192_i0 + (v193_i1 * 2))];
                  glb_m0[(v198_lead + ((v193_i1 + 17) * 64))] = v196_data;
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

