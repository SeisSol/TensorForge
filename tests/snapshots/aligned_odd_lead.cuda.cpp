// === base name ===
kernel_3067d2ef8f25f6e3

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_3067d2ef8f25f6e3 = {{32, 4, 1}, 32, 35, 1, 4, 512, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_3067d2ef8f25f6e3(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_3067d2ef8f25f6e3(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_3067d2ef8f25f6e3(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_3067d2ef8f25f6e3, block.x * block.y * block.z, 128 * sizeof(float));
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
  config.sharedMemBytes = 128 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_3067d2ef8f25f6e3(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_3067d2ef8f25f6e3(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_3067d2ef8f25f6e3, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_3067d2ef8f25f6e3<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_3067d2ef8f25f6e3(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (35 active) x 4 per block = block 32x4x1, 512 B shared, occupancy grid
    // operands:
    //   m0 35×4(35×4) {0..35}×{0..4} strided
    //   m1 35×8(35×8) {0..35}×{0..8} strided
    //   m2 8×4(8×4) {0..8}×{0..4} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":35,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":128}],"shared_bytes":512,"shared_elements":128,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[35,4]],"name":"m0","ordered":false,"parts":1,"shape":[35,4],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[35,8]],"name":"m1","ordered":false,"parts":1,"shape":[35,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,4]],"name":"m2","ordered":false,"parts":1,"shape":[8,4],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[35,4]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[35,4]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[35,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[35,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[32 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[32];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v5_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v5_batchId0 < numElements0; v5_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v6_ahead1 = v5_batchId0 + (gridDim.x * blockDim.y);
        size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 140 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 280 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 32 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v19_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v20_i0 = 0; v20_i0 < 1; ++v20_i0) {
            int32_t v23_lead = v19_lead + (v20_i0 * 32);
            #pragma unroll
            for (int32_t v21_i1 = 0; v21_i1 < 8; ++v21_i1) {
              float v26_data = __ldcg(&glb_m1[(v23_lead + (v21_i1 * 35))]);
              r0[(v20_i0 + (v21_i1 * 2))] = v26_data;
            }
          }
          bool v29_g = v19_lead < 3;
          if (v29_g) {
            int32_t v32_lead = v19_lead + 32_i32;
            #pragma unroll
            for (int32_t v30_i1 = 0; v30_i1 < 8; ++v30_i1) {
              float v35_data = __ldcg(&glb_m1[(v32_lead + (v30_i1 * 35))]);
              r0[(1 + (v30_i1 * 2))] = v35_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 35), (0, 4)] [(0, 8)]
          float ir1[8]{};
          float v41_data = r0[0];
          float v42_data = s0[0];
          float v44_data = ir1[0];
          ir1[0] = (v44_data + (v41_data * v42_data));
          float v47_data = s0[8];
          float v49_data = ir1[2];
          ir1[2] = (v49_data + (v41_data * v47_data));
          float v52_data = s0[16];
          float v54_data = ir1[4];
          ir1[4] = (v54_data + (v41_data * v52_data));
          float v57_data = s0[24];
          float v59_data = ir1[6];
          ir1[6] = (v59_data + (v41_data * v57_data));
          float v61_data = r0[1];
          float v64_data = ir1[1];
          ir1[1] = (v64_data + (v61_data * v42_data));
          float v69_data = ir1[3];
          ir1[3] = (v69_data + (v61_data * v47_data));
          float v74_data = ir1[5];
          ir1[5] = (v74_data + (v61_data * v52_data));
          float v79_data = ir1[7];
          ir1[7] = (v79_data + (v61_data * v57_data));
          float v81_data = r0[2];
          float v82_data = s0[1];
          float v84_data = ir1[0];
          ir1[0] = (v84_data + (v81_data * v82_data));
          float v87_data = s0[9];
          float v89_data = ir1[2];
          ir1[2] = (v89_data + (v81_data * v87_data));
          float v92_data = s0[17];
          float v94_data = ir1[4];
          ir1[4] = (v94_data + (v81_data * v92_data));
          float v97_data = s0[25];
          float v99_data = ir1[6];
          ir1[6] = (v99_data + (v81_data * v97_data));
          float v101_data = r0[3];
          float v104_data = ir1[1];
          ir1[1] = (v104_data + (v101_data * v82_data));
          float v109_data = ir1[3];
          ir1[3] = (v109_data + (v101_data * v87_data));
          float v114_data = ir1[5];
          ir1[5] = (v114_data + (v101_data * v92_data));
          float v119_data = ir1[7];
          ir1[7] = (v119_data + (v101_data * v97_data));
          float v121_data = r0[4];
          float v122_data = s0[2];
          float v124_data = ir1[0];
          ir1[0] = (v124_data + (v121_data * v122_data));
          float v127_data = s0[10];
          float v129_data = ir1[2];
          ir1[2] = (v129_data + (v121_data * v127_data));
          float v132_data = s0[18];
          float v134_data = ir1[4];
          ir1[4] = (v134_data + (v121_data * v132_data));
          float v137_data = s0[26];
          float v139_data = ir1[6];
          ir1[6] = (v139_data + (v121_data * v137_data));
          float v141_data = r0[5];
          float v144_data = ir1[1];
          ir1[1] = (v144_data + (v141_data * v122_data));
          float v149_data = ir1[3];
          ir1[3] = (v149_data + (v141_data * v127_data));
          float v154_data = ir1[5];
          ir1[5] = (v154_data + (v141_data * v132_data));
          float v159_data = ir1[7];
          ir1[7] = (v159_data + (v141_data * v137_data));
          float v161_data = r0[6];
          float v162_data = s0[3];
          float v164_data = ir1[0];
          ir1[0] = (v164_data + (v161_data * v162_data));
          float v167_data = s0[11];
          float v169_data = ir1[2];
          ir1[2] = (v169_data + (v161_data * v167_data));
          float v172_data = s0[19];
          float v174_data = ir1[4];
          ir1[4] = (v174_data + (v161_data * v172_data));
          float v177_data = s0[27];
          float v179_data = ir1[6];
          ir1[6] = (v179_data + (v161_data * v177_data));
          float v181_data = r0[7];
          float v184_data = ir1[1];
          ir1[1] = (v184_data + (v181_data * v162_data));
          float v189_data = ir1[3];
          ir1[3] = (v189_data + (v181_data * v167_data));
          float v194_data = ir1[5];
          ir1[5] = (v194_data + (v181_data * v172_data));
          float v199_data = ir1[7];
          ir1[7] = (v199_data + (v181_data * v177_data));
          float v201_data = r0[8];
          float v202_data = s0[4];
          float v204_data = ir1[0];
          ir1[0] = (v204_data + (v201_data * v202_data));
          float v207_data = s0[12];
          float v209_data = ir1[2];
          ir1[2] = (v209_data + (v201_data * v207_data));
          float v212_data = s0[20];
          float v214_data = ir1[4];
          ir1[4] = (v214_data + (v201_data * v212_data));
          float v217_data = s0[28];
          float v219_data = ir1[6];
          ir1[6] = (v219_data + (v201_data * v217_data));
          float v221_data = r0[9];
          float v224_data = ir1[1];
          ir1[1] = (v224_data + (v221_data * v202_data));
          float v229_data = ir1[3];
          ir1[3] = (v229_data + (v221_data * v207_data));
          float v234_data = ir1[5];
          ir1[5] = (v234_data + (v221_data * v212_data));
          float v239_data = ir1[7];
          ir1[7] = (v239_data + (v221_data * v217_data));
          float v241_data = r0[10];
          float v242_data = s0[5];
          float v244_data = ir1[0];
          ir1[0] = (v244_data + (v241_data * v242_data));
          float v247_data = s0[13];
          float v249_data = ir1[2];
          ir1[2] = (v249_data + (v241_data * v247_data));
          float v252_data = s0[21];
          float v254_data = ir1[4];
          ir1[4] = (v254_data + (v241_data * v252_data));
          float v257_data = s0[29];
          float v259_data = ir1[6];
          ir1[6] = (v259_data + (v241_data * v257_data));
          float v261_data = r0[11];
          float v264_data = ir1[1];
          ir1[1] = (v264_data + (v261_data * v242_data));
          float v269_data = ir1[3];
          ir1[3] = (v269_data + (v261_data * v247_data));
          float v274_data = ir1[5];
          ir1[5] = (v274_data + (v261_data * v252_data));
          float v279_data = ir1[7];
          ir1[7] = (v279_data + (v261_data * v257_data));
          float v281_data = r0[12];
          float v282_data = s0[6];
          float v284_data = ir1[0];
          ir1[0] = (v284_data + (v281_data * v282_data));
          float v287_data = s0[14];
          float v289_data = ir1[2];
          ir1[2] = (v289_data + (v281_data * v287_data));
          float v292_data = s0[22];
          float v294_data = ir1[4];
          ir1[4] = (v294_data + (v281_data * v292_data));
          float v297_data = s0[30];
          float v299_data = ir1[6];
          ir1[6] = (v299_data + (v281_data * v297_data));
          float v301_data = r0[13];
          float v304_data = ir1[1];
          ir1[1] = (v304_data + (v301_data * v282_data));
          float v309_data = ir1[3];
          ir1[3] = (v309_data + (v301_data * v287_data));
          float v314_data = ir1[5];
          ir1[5] = (v314_data + (v301_data * v292_data));
          float v319_data = ir1[7];
          ir1[7] = (v319_data + (v301_data * v297_data));
          float v321_data = r0[14];
          float v322_data = s0[7];
          float v324_data = ir1[0];
          ir1[0] = (v324_data + (v321_data * v322_data));
          float v327_data = s0[15];
          float v329_data = ir1[2];
          ir1[2] = (v329_data + (v321_data * v327_data));
          float v332_data = s0[23];
          float v334_data = ir1[4];
          ir1[4] = (v334_data + (v321_data * v332_data));
          float v337_data = s0[31];
          float v339_data = ir1[6];
          ir1[6] = (v339_data + (v321_data * v337_data));
          float v341_data = r0[15];
          float v344_data = ir1[1];
          ir1[1] = (v344_data + (v341_data * v322_data));
          float v349_data = ir1[3];
          ir1[3] = (v349_data + (v341_data * v327_data));
          float v354_data = ir1[5];
          ir1[5] = (v354_data + (v341_data * v332_data));
          float v359_data = ir1[7];
          ir1[7] = (v359_data + (v341_data * v337_data));
          #pragma unroll
          for (int32_t v361_n0 = 0; v361_n0 < 1; ++v361_n0) {
            #pragma unroll
            for (int32_t v362_n1 = 0; v362_n1 < 4; ++v362_n1) {
              int32_t v364_a = v361_n0 + (v362_n1 * 2);
              float v365_data = ir1[v364_a];
              r1[v364_a] = v365_data;
            }
          }
          if (v29_g) {
            #pragma unroll
            for (int32_t v366_n1 = 0; v366_n1 < 4; ++v366_n1) {
              int32_t v368_a = 1 + (v366_n1 * 2);
              float v369_data = ir1[v368_a];
              r1[v368_a] = v369_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v370_i0 = 0; v370_i0 < 1; ++v370_i0) {
            int32_t v376_lead = v19_lead + (v370_i0 * 32);
            #pragma unroll
            for (int32_t v371_i1 = 0; v371_i1 < 4; ++v371_i1) {
              float v374_data = r1[(v370_i0 + (v371_i1 * 2))];
              glb_m0[(v376_lead + (v371_i1 * 35))] = v374_data;
            }
          }
          if (v29_g) {
            int32_t v384_lead = v19_lead + 32_i32;
            #pragma unroll
            for (int32_t v379_i1 = 0; v379_i1 < 4; ++v379_i1) {
              float v382_data = r1[(1 + (v379_i1 * 2))];
              glb_m0[(v384_lead + (v379_i1 * 35))] = v382_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

