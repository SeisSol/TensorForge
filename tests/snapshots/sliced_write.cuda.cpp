// === base name ===
kernel_1cad8c11019286d6

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_1cad8c11019286d6 = {{32, 4, 1}, 32, 32, 1, 4, 3072, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_1cad8c11019286d6(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_1cad8c11019286d6(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_1cad8c11019286d6(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_1cad8c11019286d6, block.x * block.y * block.z, 768 * sizeof(float));
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
  config.sharedMemBytes = 768 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_1cad8c11019286d6(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_1cad8c11019286d6(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_1cad8c11019286d6, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_1cad8c11019286d6<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_1cad8c11019286d6(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 4 per block = block 32x4x1, 3072 B shared, occupancy grid
    // operands:
    //   m0 32×13(32×13) {0..32}×{0..13} strided
    //   m1 32×13(32×13) {0..32}×{0..13} strided
    //   m2 13×13(13×13) {0..13}×{0..13} strided
    // operations:
    //   m0[i,j]@{0..32}×{6..13} = m1[i,k]@{0..32}×{10..13} × m2[k,j]@{10..13}×{6..13}
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":768}],"shared_bytes":3072,"shared_elements":768,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[32,13]],"name":"m0","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"Q","bbox":[[0,0],[32,13]],"name":"m1","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"S","bbox":[[0,0],[13,13]],"name":"m2","ordered":false,"parts":1,"shape":[13,13],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,6],[32,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,10],[32,13]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[10,6],[13,13]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[192 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[192];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v11_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v11_batchId0 < numElements0; v11_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v12_ahead1 = v11_batchId0 + (gridDim.x * blockDim.y);
        size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v11_batchId0 * 416 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v11_batchId0 * 416 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v11_batchId0 * 169 + 0 + m2_extraOffset];
          float r0[3]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v25_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v26_i0 = 0; v26_i0 < 1; ++v26_i0) {
            int32_t v29_lead = v25_lead + (v26_i0 * 32);
            #pragma unroll
            for (int32_t v27_i1 = 10; v27_i1 < 13; ++v27_i1) {
              float v32_data = __ldcg(&glb_m1[(v29_lead + (v27_i1 * 32))]);
              r0[(v26_i0 + (v27_i1 - 10))] = v32_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 5; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 32], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 32], 4);
          }
          if (threadIdx.x < 9) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 160], &glb_m2[0 + 0 + 1 * threadIdx.x + 160], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[7]{};
          __syncwarp();
          // ir1 = +(r0 * s0)
          // [(0, 32), (6, 13)] [(10, 13)]
          float ir1[7]{};
          float v39_data = r0[0];
          float v40_data = s0[88];
          float v42_data = ir1[0];
          ir1[0] = (v42_data + (v39_data * v40_data));
          float v45_data = s0[101];
          float v47_data = ir1[1];
          ir1[1] = (v47_data + (v39_data * v45_data));
          float v50_data = s0[114];
          float v52_data = ir1[2];
          ir1[2] = (v52_data + (v39_data * v50_data));
          float v55_data = s0[127];
          float v57_data = ir1[3];
          ir1[3] = (v57_data + (v39_data * v55_data));
          float v60_data = s0[140];
          float v62_data = ir1[4];
          ir1[4] = (v62_data + (v39_data * v60_data));
          float v65_data = s0[153];
          float v67_data = ir1[5];
          ir1[5] = (v67_data + (v39_data * v65_data));
          float v70_data = s0[166];
          float v72_data = ir1[6];
          ir1[6] = (v72_data + (v39_data * v70_data));
          float v74_data = r0[1];
          float v75_data = s0[89];
          float v77_data = ir1[0];
          ir1[0] = (v77_data + (v74_data * v75_data));
          float v80_data = s0[102];
          float v82_data = ir1[1];
          ir1[1] = (v82_data + (v74_data * v80_data));
          float v85_data = s0[115];
          float v87_data = ir1[2];
          ir1[2] = (v87_data + (v74_data * v85_data));
          float v90_data = s0[128];
          float v92_data = ir1[3];
          ir1[3] = (v92_data + (v74_data * v90_data));
          float v95_data = s0[141];
          float v97_data = ir1[4];
          ir1[4] = (v97_data + (v74_data * v95_data));
          float v100_data = s0[154];
          float v102_data = ir1[5];
          ir1[5] = (v102_data + (v74_data * v100_data));
          float v105_data = s0[167];
          float v107_data = ir1[6];
          ir1[6] = (v107_data + (v74_data * v105_data));
          float v109_data = r0[2];
          float v110_data = s0[90];
          float v112_data = ir1[0];
          ir1[0] = (v112_data + (v109_data * v110_data));
          float v115_data = s0[103];
          float v117_data = ir1[1];
          ir1[1] = (v117_data + (v109_data * v115_data));
          float v120_data = s0[116];
          float v122_data = ir1[2];
          ir1[2] = (v122_data + (v109_data * v120_data));
          float v125_data = s0[129];
          float v127_data = ir1[3];
          ir1[3] = (v127_data + (v109_data * v125_data));
          float v130_data = s0[142];
          float v132_data = ir1[4];
          ir1[4] = (v132_data + (v109_data * v130_data));
          float v135_data = s0[155];
          float v137_data = ir1[5];
          ir1[5] = (v137_data + (v109_data * v135_data));
          float v140_data = s0[168];
          float v142_data = ir1[6];
          ir1[6] = (v142_data + (v109_data * v140_data));
          // r1 = ir1
          #pragma unroll
          for (int32_t v144_n0 = 0; v144_n0 < 1; ++v144_n0) {
            #pragma unroll
            for (int32_t v145_n1 = 6; v145_n1 < 13; ++v145_n1) {
              int32_t v147_a = v144_n0 + (v145_n1 - 6);
              float v148_data = ir1[v147_a];
              r1[v147_a] = v148_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v149_i0 = 0; v149_i0 < 1; ++v149_i0) {
            int32_t v152_lead = v25_lead + (v149_i0 * 32);
            glb_m0[v152_lead] = 0.0f;
            glb_m0[(v152_lead + 32)] = 0.0f;
            glb_m0[(v152_lead + 64)] = 0.0f;
            glb_m0[(v152_lead + 96)] = 0.0f;
            glb_m0[(v152_lead + 128)] = 0.0f;
            glb_m0[(v152_lead + 160)] = 0.0f;
            float v160_data = r1[v149_i0];
            glb_m0[(v152_lead + 192)] = v160_data;
            float v163_data = r1[(v149_i0 + 1)];
            glb_m0[(v152_lead + 224)] = v163_data;
            float v166_data = r1[(v149_i0 + 2)];
            glb_m0[(v152_lead + 256)] = v166_data;
            float v169_data = r1[(v149_i0 + 3)];
            glb_m0[(v152_lead + 288)] = v169_data;
            float v172_data = r1[(v149_i0 + 4)];
            glb_m0[(v152_lead + 320)] = v172_data;
            float v175_data = r1[(v149_i0 + 5)];
            glb_m0[(v152_lead + 352)] = v175_data;
            float v178_data = r1[(v149_i0 + 6)];
            glb_m0[(v152_lead + 384)] = v178_data;
          }
          __syncwarp();
        }
      }
    }
  }
}

