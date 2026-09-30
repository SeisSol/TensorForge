// === base name ===
kernel_5afaddd829fda1dc

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_5afaddd829fda1dc = {{32, 1, 1}, 32, 64, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_5afaddd829fda1dc(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_5afaddd829fda1dc(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_5afaddd829fda1dc(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_5afaddd829fda1dc(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_5afaddd829fda1dc(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_5afaddd829fda1dc(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_5afaddd829fda1dc(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t numElements0, unsigned * flags0) {
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
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          for (size_t v1_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v1_batchId0 < numElements0; v1_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v2_ahead1 = v1_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v1_batchId0 * 1216 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v1_batchId0 * 1216 + 0 + m1_extraOffset];
              float r0[38]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v14_lead = item.get_local_id(2) % 32;
              #pragma unroll
              for (int32_t v15_i0 = 0; v15_i0 < 2; ++v15_i0) {
                int32_t v18_lead = v14_lead + (v15_i0 * 32);
                #pragma unroll
                for (int32_t v16_i1 = 0; v16_i1 < 19; ++v16_i1) {
                  float v21_data = glb_m1[(v18_lead + (v16_i1 * 64))];
                  r0[(v15_i0 + (v16_i1 * 2))] = v21_data;
                }
              }
              float r1[34]{};
              // r1 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v25_i0 = 0; v25_i0 < 2; ++v25_i0) {
                int32_t v28_lead = v14_lead + (v25_i0 * 32);
                #pragma unroll
                for (int32_t v26_i1 = 0; v26_i1 < 17; ++v26_i1) {
                  float v31_data = glb_m0[(v28_lead + (v26_i1 * 64))];
                  r1[(v25_i0 + (v26_i1 * 2))] = v31_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m0););
              float r2[34]{};
              // ir2 = +(r0)
              // [(0, 64), (0, 17)] []
              float ir2[34]{};
              float v36_data = r0[0];
              float v37_data = ir2[0];
              ir2[0] = (v37_data + v36_data);
              float v39_data = r0[2];
              float v40_data = ir2[2];
              ir2[2] = (v40_data + v39_data);
              float v42_data = r0[4];
              float v43_data = ir2[4];
              ir2[4] = (v43_data + v42_data);
              float v45_data = r0[6];
              float v46_data = ir2[6];
              ir2[6] = (v46_data + v45_data);
              float v48_data = r0[8];
              float v49_data = ir2[8];
              ir2[8] = (v49_data + v48_data);
              float v51_data = r0[10];
              float v52_data = ir2[10];
              ir2[10] = (v52_data + v51_data);
              float v54_data = r0[12];
              float v55_data = ir2[12];
              ir2[12] = (v55_data + v54_data);
              float v57_data = r0[14];
              float v58_data = ir2[14];
              ir2[14] = (v58_data + v57_data);
              float v60_data = r0[16];
              float v61_data = ir2[16];
              ir2[16] = (v61_data + v60_data);
              float v63_data = r0[18];
              float v64_data = ir2[18];
              ir2[18] = (v64_data + v63_data);
              float v66_data = r0[20];
              float v67_data = ir2[20];
              ir2[20] = (v67_data + v66_data);
              float v69_data = r0[22];
              float v70_data = ir2[22];
              ir2[22] = (v70_data + v69_data);
              float v72_data = r0[24];
              float v73_data = ir2[24];
              ir2[24] = (v73_data + v72_data);
              float v75_data = r0[26];
              float v76_data = ir2[26];
              ir2[26] = (v76_data + v75_data);
              float v78_data = r0[28];
              float v79_data = ir2[28];
              ir2[28] = (v79_data + v78_data);
              float v81_data = r0[30];
              float v82_data = ir2[30];
              ir2[30] = (v82_data + v81_data);
              float v84_data = r0[32];
              float v85_data = ir2[32];
              ir2[32] = (v85_data + v84_data);
              float v87_data = r0[1];
              float v88_data = ir2[1];
              ir2[1] = (v88_data + v87_data);
              float v90_data = r0[3];
              float v91_data = ir2[3];
              ir2[3] = (v91_data + v90_data);
              float v93_data = r0[5];
              float v94_data = ir2[5];
              ir2[5] = (v94_data + v93_data);
              float v96_data = r0[7];
              float v97_data = ir2[7];
              ir2[7] = (v97_data + v96_data);
              float v99_data = r0[9];
              float v100_data = ir2[9];
              ir2[9] = (v100_data + v99_data);
              float v102_data = r0[11];
              float v103_data = ir2[11];
              ir2[11] = (v103_data + v102_data);
              float v105_data = r0[13];
              float v106_data = ir2[13];
              ir2[13] = (v106_data + v105_data);
              float v108_data = r0[15];
              float v109_data = ir2[15];
              ir2[15] = (v109_data + v108_data);
              float v111_data = r0[17];
              float v112_data = ir2[17];
              ir2[17] = (v112_data + v111_data);
              float v114_data = r0[19];
              float v115_data = ir2[19];
              ir2[19] = (v115_data + v114_data);
              float v117_data = r0[21];
              float v118_data = ir2[21];
              ir2[21] = (v118_data + v117_data);
              float v120_data = r0[23];
              float v121_data = ir2[23];
              ir2[23] = (v121_data + v120_data);
              float v123_data = r0[25];
              float v124_data = ir2[25];
              ir2[25] = (v124_data + v123_data);
              float v126_data = r0[27];
              float v127_data = ir2[27];
              ir2[27] = (v127_data + v126_data);
              float v129_data = r0[29];
              float v130_data = ir2[29];
              ir2[29] = (v130_data + v129_data);
              float v132_data = r0[31];
              float v133_data = ir2[31];
              ir2[31] = (v133_data + v132_data);
              float v135_data = r0[33];
              float v136_data = ir2[33];
              ir2[33] = (v136_data + v135_data);
              // r2 = ir2 + r1
              #pragma unroll
              for (int32_t v138_n0 = 0; v138_n0 < 2; ++v138_n0) {
                #pragma unroll
                for (int32_t v139_n1 = 0; v139_n1 < 17; ++v139_n1) {
                  int32_t v141_a = v138_n0 + (v139_n1 * 2);
                  float v142_data = ir2[v141_a];
                  float v143_data = r1[v141_a];
                  r2[v141_a] = (v143_data + v142_data);
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v145_i0 = 0; v145_i0 < 2; ++v145_i0) {
                int32_t v151_lead = v14_lead + (v145_i0 * 32);
                #pragma unroll
                for (int32_t v146_i1 = 0; v146_i1 < 17; ++v146_i1) {
                  float v149_data = r2[(v145_i0 + (v146_i1 * 2))];
                  glb_m0[(v151_lead + (v146_i1 * 64))] = v149_data;
                }
              }
              float r3[4]{};
              // r3 = max(glb_m0, glb_m1)
              #pragma unroll
              for (int32_t v155_k0 = 0; v155_k0 < 2; ++v155_k0) {
                int32_t v158_lead = v14_lead + (v155_k0 * 32);
                #pragma unroll
                for (int32_t v156_k1 = 0; v156_k1 < 2; ++v156_k1) {
                  int32_t v161_a = v158_lead + ((v156_k1 + 17) * 64);
                  float v162_data = glb_m0[v161_a];
                  float v163_data = glb_m1[v161_a];
                  r3[(v155_k0 + (v156_k1 * 2))] = (sycl::max(float(v162_data), float(v163_data)));
                }
              }
              float r4[4]{};
              // ir4 = +(r3)
              // [(0, 64), (0, 2)] []
              float ir4[4]{};
              float v169_data = r3[0];
              float v170_data = ir4[0];
              ir4[0] = (v170_data + v169_data);
              float v172_data = r3[2];
              float v173_data = ir4[2];
              ir4[2] = (v173_data + v172_data);
              float v175_data = r3[1];
              float v176_data = ir4[1];
              ir4[1] = (v176_data + v175_data);
              float v178_data = r3[3];
              float v179_data = ir4[3];
              ir4[3] = (v179_data + v178_data);
              // r4 = ir4
              #pragma unroll
              for (int32_t v181_n0 = 0; v181_n0 < 2; ++v181_n0) {
                #pragma unroll
                for (int32_t v182_n1 = 0; v182_n1 < 2; ++v182_n1) {
                  int32_t v184_a = v181_n0 + (v182_n1 * 2);
                  float v185_data = ir4[v184_a];
                  r4[v184_a] = v185_data;
                }
              }
              // glb_m0 = store{r>g}(r4);
              #pragma unroll
              for (int32_t v186_i0 = 0; v186_i0 < 2; ++v186_i0) {
                int32_t v192_lead = v14_lead + (v186_i0 * 32);
                #pragma unroll
                for (int32_t v187_i1 = 0; v187_i1 < 2; ++v187_i1) {
                  float v190_data = r4[(v186_i0 + (v187_i1 * 2))];
                  glb_m0[(v192_lead + ((v187_i1 + 17) * 64))] = v190_data;
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

