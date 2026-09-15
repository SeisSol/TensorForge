// === base name ===
kernel_9d5d9fa60dcc4098

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_9d5d9fa60dcc4098 = {{32, 4, 1}, 32, 20, 1, 4, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_9d5d9fa60dcc4098(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_9d5d9fa60dcc4098(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_9d5d9fa60dcc4098(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 4, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_9d5d9fa60dcc4098, block.x * block.y * block.z, 0 * sizeof(float));
        CHECK_ERR;
        if (blocksPerSM > 0) {
          gridsize = smCount * blocksPerSM;
        }
        else {
          gridsize = smCount;
        }
      }
      
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 32;
  config.block[1] = 4;
  config.block[2] = 1;
  config.sharedMemBytes = 0 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_9d5d9fa60dcc4098(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_9d5d9fa60dcc4098(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_9d5d9fa60dcc4098, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_9d5d9fa60dcc4098<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_9d5d9fa60dcc4098(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (20 active) x 4 per block = block 32x4x1, 0 B shared, occupancy grid
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
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":20,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[20,9]],"name":"m0","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"strided","alias":"Q","bbox":[[0,0],[20,9]],"name":"m1","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"strided","alias":"F0","bbox":[[0,0],[10,9]],"name":"m2","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"strided","alias":"F1","bbox":[[0,0],[4,9]],"name":"m3","ordered":false,"parts":1,"shape":[4,9],"variant":false},{"addressing":"strided","alias":"F2","bbox":[[0,0],[1,9]],"name":"m4","ordered":false,"parts":1,"shape":[1,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[20,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[10,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[4,9]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[4,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[1,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[1,9]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      for (size_t v1_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v1_batchId0 < numElements0; v1_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v2_ahead1 = v1_batchId0 + (gridDim.x * blockDim.y);
        size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v1_batchId0 * 180 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v1_batchId0 * 180 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v1_batchId0 * 90 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[v1_batchId0 * 36 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[v1_batchId0 * 9 + 0 + m4_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v17_lead = threadIdx.x % 32;
          bool v18_g = v17_lead < 20;
          if (v18_g) {
            #pragma unroll
            for (int32_t v19_i1 = 0; v19_i1 < 9; ++v19_i1) {
              float v24_data = __ldcg(&glb_m1[(v17_lead + (v19_i1 * 20))]);
              r0[v19_i1] = v24_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m2);
          if (v17_lead < 10) {
            #pragma unroll
            for (int32_t v28_i1 = 0; v28_i1 < 9; ++v28_i1) {
              float v33_data = __ldcg(&glb_m2[(v17_lead + (v28_i1 * 10))]);
              r2[v28_i1] = v33_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          float r1[9]{};
          // ir1 = +(r0)
          // [(0, 20), (0, 9)] []
          float ir1[9]{};
          float v37_data = r0[0];
          float v38_data = ir1[0];
          ir1[0] = (v38_data + v37_data);
          float v40_data = r0[1];
          float v41_data = ir1[1];
          ir1[1] = (v41_data + v40_data);
          float v43_data = r0[2];
          float v44_data = ir1[2];
          ir1[2] = (v44_data + v43_data);
          float v46_data = r0[3];
          float v47_data = ir1[3];
          ir1[3] = (v47_data + v46_data);
          float v49_data = r0[4];
          float v50_data = ir1[4];
          ir1[4] = (v50_data + v49_data);
          float v52_data = r0[5];
          float v53_data = ir1[5];
          ir1[5] = (v53_data + v52_data);
          float v55_data = r0[6];
          float v56_data = ir1[6];
          ir1[6] = (v56_data + v55_data);
          float v58_data = r0[7];
          float v59_data = ir1[7];
          ir1[7] = (v59_data + v58_data);
          float v61_data = r0[8];
          float v62_data = ir1[8];
          ir1[8] = (v62_data + v61_data);
          // r1 = ir1
          if (v18_g) {
            #pragma unroll
            for (int32_t v64_n1 = 0; v64_n1 < 9; ++v64_n1) {
              float v66_data = ir1[v64_n1];
              r1[v64_n1] = v66_data;
            }
          }
          float r4[9]{};
          // r4 = load{g>r}(glb_m3);
          if (v17_lead < 4) {
            #pragma unroll
            for (int32_t v69_i1 = 0; v69_i1 < 9; ++v69_i1) {
              float v74_data = __ldcg(&glb_m3[(v17_lead + (v69_i1 * 4))]);
              r4[v69_i1] = v74_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m2););
          float r3[9]{};
          // ir3 = +(r2)
          // [(0, 10), (0, 9)] []
          float ir3[9]{};
          float v78_data = r2[0];
          float v79_data = ir3[0];
          ir3[0] = (v79_data + v78_data);
          float v81_data = r2[1];
          float v82_data = ir3[1];
          ir3[1] = (v82_data + v81_data);
          float v84_data = r2[2];
          float v85_data = ir3[2];
          ir3[2] = (v85_data + v84_data);
          float v87_data = r2[3];
          float v88_data = ir3[3];
          ir3[3] = (v88_data + v87_data);
          float v90_data = r2[4];
          float v91_data = ir3[4];
          ir3[4] = (v91_data + v90_data);
          float v93_data = r2[5];
          float v94_data = ir3[5];
          ir3[5] = (v94_data + v93_data);
          float v96_data = r2[6];
          float v97_data = ir3[6];
          ir3[6] = (v97_data + v96_data);
          float v99_data = r2[7];
          float v100_data = ir3[7];
          ir3[7] = (v100_data + v99_data);
          float v102_data = r2[8];
          float v103_data = ir3[8];
          ir3[8] = (v103_data + v102_data);
          // r3 = ir3 + r1
          if (v18_g) {
            #pragma unroll
            for (int32_t v105_n1 = 0; v105_n1 < 9; ++v105_n1) {
              float v107_data = ir3[v105_n1];
              float v108_data = r1[v105_n1];
              r3[v105_n1] = (v108_data + v107_data);
            }
          }
          float r6[9]{};
          // r6 = load{g>r}(glb_m4);
          if (v17_lead < 1) {
            #pragma unroll
            for (int32_t v112_i1 = 0; v112_i1 < 9; ++v112_i1) {
              float v116_data = __ldcg(&glb_m4[(v17_lead + v112_i1)]);
              r6[v112_i1] = v116_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m3););
          float r5[9]{};
          // ir5 = +(r4)
          // [(0, 4), (0, 9)] []
          float ir5[9]{};
          float v120_data = r4[0];
          float v121_data = ir5[0];
          ir5[0] = (v121_data + v120_data);
          float v123_data = r4[1];
          float v124_data = ir5[1];
          ir5[1] = (v124_data + v123_data);
          float v126_data = r4[2];
          float v127_data = ir5[2];
          ir5[2] = (v127_data + v126_data);
          float v129_data = r4[3];
          float v130_data = ir5[3];
          ir5[3] = (v130_data + v129_data);
          float v132_data = r4[4];
          float v133_data = ir5[4];
          ir5[4] = (v133_data + v132_data);
          float v135_data = r4[5];
          float v136_data = ir5[5];
          ir5[5] = (v136_data + v135_data);
          float v138_data = r4[6];
          float v139_data = ir5[6];
          ir5[6] = (v139_data + v138_data);
          float v141_data = r4[7];
          float v142_data = ir5[7];
          ir5[7] = (v142_data + v141_data);
          float v144_data = r4[8];
          float v145_data = ir5[8];
          ir5[8] = (v145_data + v144_data);
          // r5 = ir5 + r3
          if (v18_g) {
            #pragma unroll
            for (int32_t v147_n1 = 0; v147_n1 < 9; ++v147_n1) {
              float v149_data = ir5[v147_n1];
              float v150_data = r3[v147_n1];
              r5[v147_n1] = (v150_data + v149_data);
            }
          }
          // wait(r6 = load{g>r}(glb_m4););
          float r7[9]{};
          // ir7 = +(r6)
          // [(0, 1), (0, 9)] []
          float ir7[9]{};
          float v154_data = r6[0];
          float v155_data = ir7[0];
          ir7[0] = (v155_data + v154_data);
          float v157_data = r6[1];
          float v158_data = ir7[1];
          ir7[1] = (v158_data + v157_data);
          float v160_data = r6[2];
          float v161_data = ir7[2];
          ir7[2] = (v161_data + v160_data);
          float v163_data = r6[3];
          float v164_data = ir7[3];
          ir7[3] = (v164_data + v163_data);
          float v166_data = r6[4];
          float v167_data = ir7[4];
          ir7[4] = (v167_data + v166_data);
          float v169_data = r6[5];
          float v170_data = ir7[5];
          ir7[5] = (v170_data + v169_data);
          float v172_data = r6[6];
          float v173_data = ir7[6];
          ir7[6] = (v173_data + v172_data);
          float v175_data = r6[7];
          float v176_data = ir7[7];
          ir7[7] = (v176_data + v175_data);
          float v178_data = r6[8];
          float v179_data = ir7[8];
          ir7[8] = (v179_data + v178_data);
          // r7 = ir7 + r5
          if (v18_g) {
            #pragma unroll
            for (int32_t v181_n1 = 0; v181_n1 < 9; ++v181_n1) {
              float v183_data = ir7[v181_n1];
              float v184_data = r5[v181_n1];
              r7[v181_n1] = (v184_data + v183_data);
            }
          }
          // glb_m0 = store{r>g}(r7);
          if (v18_g) {
            #pragma unroll
            for (int32_t v186_i1 = 0; v186_i1 < 9; ++v186_i1) {
              float v188_data = r7[v186_i1];
              glb_m0[(v17_lead + (v186_i1 * 20))] = v188_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

