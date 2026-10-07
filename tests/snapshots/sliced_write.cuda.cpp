// === base name ===
kernel_89d3aad4185b5b0d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_89d3aad4185b5b0d = {{32, 4, 1}, 32, 32, 1, 4, 3072, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_89d3aad4185b5b0d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_89d3aad4185b5b0d(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_89d3aad4185b5b0d(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_89d3aad4185b5b0d, block.x * block.y * block.z, 768 * sizeof(float));
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
void launcher_kernel_89d3aad4185b5b0d(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_89d3aad4185b5b0d(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_89d3aad4185b5b0d, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_89d3aad4185b5b0d<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_89d3aad4185b5b0d(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 416 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 416 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 169 + 0 + m2_extraOffset];
          float r0[3]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v22_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
            int32_t v26_lead = v22_lead + (v23_i0 * 32);
            #pragma unroll
            for (int32_t v24_i1 = 10; v24_i1 < 13; ++v24_i1) {
              float v29_data = __ldcg(&glb_m1[(v26_lead + (v24_i1 * 32))]);
              r0[(v23_i0 + (v24_i1 - 10))] = v29_data;
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
          // ir1 = +(r0 * s0)
          // [(0, 32), (6, 13)] [(10, 13)]
          float ir1[7]{};
          float v36_data = r0[0];
          __syncwarp();
          float v37_data = s0[88];
          float v39_data = ir1[0];
          ir1[0] = (v39_data + (v36_data * v37_data));
          float v42_data = s0[101];
          float v44_data = ir1[1];
          ir1[1] = (v44_data + (v36_data * v42_data));
          float v47_data = s0[114];
          float v49_data = ir1[2];
          ir1[2] = (v49_data + (v36_data * v47_data));
          float v52_data = s0[127];
          float v54_data = ir1[3];
          ir1[3] = (v54_data + (v36_data * v52_data));
          float v57_data = s0[140];
          float v59_data = ir1[4];
          ir1[4] = (v59_data + (v36_data * v57_data));
          float v62_data = s0[153];
          float v64_data = ir1[5];
          ir1[5] = (v64_data + (v36_data * v62_data));
          float v67_data = s0[166];
          float v69_data = ir1[6];
          ir1[6] = (v69_data + (v36_data * v67_data));
          float v71_data = r0[1];
          float v72_data = s0[89];
          float v74_data = ir1[0];
          ir1[0] = (v74_data + (v71_data * v72_data));
          float v77_data = s0[102];
          float v79_data = ir1[1];
          ir1[1] = (v79_data + (v71_data * v77_data));
          float v82_data = s0[115];
          float v84_data = ir1[2];
          ir1[2] = (v84_data + (v71_data * v82_data));
          float v87_data = s0[128];
          float v89_data = ir1[3];
          ir1[3] = (v89_data + (v71_data * v87_data));
          float v92_data = s0[141];
          float v94_data = ir1[4];
          ir1[4] = (v94_data + (v71_data * v92_data));
          float v97_data = s0[154];
          float v99_data = ir1[5];
          ir1[5] = (v99_data + (v71_data * v97_data));
          float v102_data = s0[167];
          float v104_data = ir1[6];
          ir1[6] = (v104_data + (v71_data * v102_data));
          float v106_data = r0[2];
          float v107_data = s0[90];
          float v109_data = ir1[0];
          ir1[0] = (v109_data + (v106_data * v107_data));
          float v112_data = s0[103];
          float v114_data = ir1[1];
          ir1[1] = (v114_data + (v106_data * v112_data));
          float v117_data = s0[116];
          float v119_data = ir1[2];
          ir1[2] = (v119_data + (v106_data * v117_data));
          float v122_data = s0[129];
          float v124_data = ir1[3];
          ir1[3] = (v124_data + (v106_data * v122_data));
          float v127_data = s0[142];
          float v129_data = ir1[4];
          ir1[4] = (v129_data + (v106_data * v127_data));
          float v132_data = s0[155];
          float v134_data = ir1[5];
          ir1[5] = (v134_data + (v106_data * v132_data));
          float v137_data = s0[168];
          float v139_data = ir1[6];
          ir1[6] = (v139_data + (v106_data * v137_data));
          // r1 = ir1
          #pragma unroll
          for (int32_t v141_n0 = 0; v141_n0 < 1; ++v141_n0) {
            #pragma unroll
            for (int32_t v142_n1 = 6; v142_n1 < 13; ++v142_n1) {
              int32_t v144_a = v141_n0 + (v142_n1 - 6);
              float v145_data = ir1[v144_a];
              r1[v144_a] = v145_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v146_i0 = 0; v146_i0 < 1; ++v146_i0) {
            int32_t v149_lead = v22_lead + (v146_i0 * 32);
            glb_m0[v149_lead] = 0.0f;
            glb_m0[(v149_lead + 32)] = 0.0f;
            glb_m0[(v149_lead + 64)] = 0.0f;
            glb_m0[(v149_lead + 96)] = 0.0f;
            glb_m0[(v149_lead + 128)] = 0.0f;
            glb_m0[(v149_lead + 160)] = 0.0f;
            float v157_data = r1[v146_i0];
            glb_m0[(v149_lead + 192)] = v157_data;
            float v160_data = r1[(v146_i0 + 1)];
            glb_m0[(v149_lead + 224)] = v160_data;
            float v163_data = r1[(v146_i0 + 2)];
            glb_m0[(v149_lead + 256)] = v163_data;
            float v166_data = r1[(v146_i0 + 3)];
            glb_m0[(v149_lead + 288)] = v166_data;
            float v169_data = r1[(v146_i0 + 4)];
            glb_m0[(v149_lead + 320)] = v169_data;
            float v172_data = r1[(v146_i0 + 5)];
            glb_m0[(v149_lead + 352)] = v172_data;
            float v175_data = r1[(v146_i0 + 6)];
            glb_m0[(v149_lead + 384)] = v175_data;
          }
          __syncwarp();
        }
      }
    }
  }
}

