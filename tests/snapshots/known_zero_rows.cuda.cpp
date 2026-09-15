// === base name ===
kernel_0f3e67036ce4597e

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_0f3e67036ce4597e = {{32, 4, 1}, 32, 40, 1, 4, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_0f3e67036ce4597e(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_0f3e67036ce4597e(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_0f3e67036ce4597e(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_0f3e67036ce4597e, block.x * block.y * block.z, 256 * sizeof(float));
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
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_0f3e67036ce4597e(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_0f3e67036ce4597e(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_0f3e67036ce4597e, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_0f3e67036ce4597e<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_0f3e67036ce4597e(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (40 active) x 4 per block = block 32x4x1, 1024 B shared, occupancy grid
    // operands:
    //   m0 40×6(40×6) {0..40}×{0..6} strided
    //   m1 40×8(40×8) {0..40}×{0..8} none
    //   m2 8×6(8×6) {0..8}×{0..6} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":40,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[40,6]],"name":"m0","ordered":false,"parts":1,"shape":[40,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[40,8]],"name":"m1","ordered":false,"parts":1,"shape":[40,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,6]],"name":"m2","ordered":false,"parts":1,"shape":[8,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[40,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[40,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[40,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[40,8]},{"addressing":"strided","bbox":[[0,0],[8,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[64 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      const float *const __restrict__ glb_m1 = &m1[0];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v6_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v6_batchId0 < numElements0; v6_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v7_ahead1 = v6_batchId0 + (gridDim.x * blockDim.y);
        size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 240 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v6_batchId0 * 48 + 0 + m2_extraOffset];
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
          if (threadIdx.x < 16) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
          }
          __pipeline_commit();
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r0[12]{};
          __syncwarp();
          // ir0 = +(glb_m1 * s0)
          // [(0, 40), (0, 6)] [(0, 8)]
          float ir0[12]{};
          int32_t v22_lead = threadIdx.x % 32;
          int32_t v24_lead = v22_lead + 32_i32;
          bool v26_g = v22_lead < 8;
          float v27_data = v26_g ? (glb_m1[v24_lead]) : (0.0f);
          float v28_data = s0[0];
          float v30_data = ir0[1];
          ir0[1] = (v30_data + (v27_data * v28_data));
          float v33_data = s0[8];
          float v35_data = ir0[3];
          ir0[3] = (v35_data + (v27_data * v33_data));
          float v38_data = s0[16];
          float v40_data = ir0[5];
          ir0[5] = (v40_data + (v27_data * v38_data));
          float v43_data = s0[24];
          float v45_data = ir0[7];
          ir0[7] = (v45_data + (v27_data * v43_data));
          float v48_data = s0[32];
          float v50_data = ir0[9];
          ir0[9] = (v50_data + (v27_data * v48_data));
          float v53_data = s0[40];
          float v55_data = ir0[11];
          ir0[11] = (v55_data + (v27_data * v53_data));
          float v60_data = glb_m1[(v22_lead + 40)];
          float v61_data = s0[1];
          float v63_data = ir0[0];
          ir0[0] = (v63_data + (v60_data * v61_data));
          float v66_data = s0[9];
          float v68_data = ir0[2];
          ir0[2] = (v68_data + (v60_data * v66_data));
          float v71_data = s0[17];
          float v73_data = ir0[4];
          ir0[4] = (v73_data + (v60_data * v71_data));
          float v76_data = s0[25];
          float v78_data = ir0[6];
          ir0[6] = (v78_data + (v60_data * v76_data));
          float v81_data = s0[33];
          float v83_data = ir0[8];
          ir0[8] = (v83_data + (v60_data * v81_data));
          float v86_data = s0[41];
          float v88_data = ir0[10];
          ir0[10] = (v88_data + (v60_data * v86_data));
          float v91_data = v26_g ? (glb_m1[(v24_lead + 40)]) : (0.0f);
          float v94_data = ir0[1];
          ir0[1] = (v94_data + (v91_data * v61_data));
          float v99_data = ir0[3];
          ir0[3] = (v99_data + (v91_data * v66_data));
          float v104_data = ir0[5];
          ir0[5] = (v104_data + (v91_data * v71_data));
          float v109_data = ir0[7];
          ir0[7] = (v109_data + (v91_data * v76_data));
          float v114_data = ir0[9];
          ir0[9] = (v114_data + (v91_data * v81_data));
          float v119_data = ir0[11];
          ir0[11] = (v119_data + (v91_data * v86_data));
          float v122_data = v26_g ? (glb_m1[(v24_lead + 80)]) : (0.0f);
          float v123_data = s0[2];
          float v125_data = ir0[1];
          ir0[1] = (v125_data + (v122_data * v123_data));
          float v128_data = s0[10];
          float v130_data = ir0[3];
          ir0[3] = (v130_data + (v122_data * v128_data));
          float v133_data = s0[18];
          float v135_data = ir0[5];
          ir0[5] = (v135_data + (v122_data * v133_data));
          float v138_data = s0[26];
          float v140_data = ir0[7];
          ir0[7] = (v140_data + (v122_data * v138_data));
          float v143_data = s0[34];
          float v145_data = ir0[9];
          ir0[9] = (v145_data + (v122_data * v143_data));
          float v148_data = s0[42];
          float v150_data = ir0[11];
          ir0[11] = (v150_data + (v122_data * v148_data));
          float v153_data = glb_m1[(v22_lead + 120)];
          float v154_data = s0[3];
          float v156_data = ir0[0];
          ir0[0] = (v156_data + (v153_data * v154_data));
          float v159_data = s0[11];
          float v161_data = ir0[2];
          ir0[2] = (v161_data + (v153_data * v159_data));
          float v164_data = s0[19];
          float v166_data = ir0[4];
          ir0[4] = (v166_data + (v153_data * v164_data));
          float v169_data = s0[27];
          float v171_data = ir0[6];
          ir0[6] = (v171_data + (v153_data * v169_data));
          float v174_data = s0[35];
          float v176_data = ir0[8];
          ir0[8] = (v176_data + (v153_data * v174_data));
          float v179_data = s0[43];
          float v181_data = ir0[10];
          ir0[10] = (v181_data + (v153_data * v179_data));
          float v184_data = v26_g ? (glb_m1[(v24_lead + 120)]) : (0.0f);
          float v187_data = ir0[1];
          ir0[1] = (v187_data + (v184_data * v154_data));
          float v192_data = ir0[3];
          ir0[3] = (v192_data + (v184_data * v159_data));
          float v197_data = ir0[5];
          ir0[5] = (v197_data + (v184_data * v164_data));
          float v202_data = ir0[7];
          ir0[7] = (v202_data + (v184_data * v169_data));
          float v207_data = ir0[9];
          ir0[9] = (v207_data + (v184_data * v174_data));
          float v212_data = ir0[11];
          ir0[11] = (v212_data + (v184_data * v179_data));
          float v215_data = glb_m1[(v22_lead + 160)];
          float v216_data = s0[4];
          float v218_data = ir0[0];
          ir0[0] = (v218_data + (v215_data * v216_data));
          float v221_data = s0[12];
          float v223_data = ir0[2];
          ir0[2] = (v223_data + (v215_data * v221_data));
          float v226_data = s0[20];
          float v228_data = ir0[4];
          ir0[4] = (v228_data + (v215_data * v226_data));
          float v231_data = s0[28];
          float v233_data = ir0[6];
          ir0[6] = (v233_data + (v215_data * v231_data));
          float v236_data = s0[36];
          float v238_data = ir0[8];
          ir0[8] = (v238_data + (v215_data * v236_data));
          float v241_data = s0[44];
          float v243_data = ir0[10];
          ir0[10] = (v243_data + (v215_data * v241_data));
          float v246_data = v26_g ? (glb_m1[(v24_lead + 160)]) : (0.0f);
          float v249_data = ir0[1];
          ir0[1] = (v249_data + (v246_data * v216_data));
          float v254_data = ir0[3];
          ir0[3] = (v254_data + (v246_data * v221_data));
          float v259_data = ir0[5];
          ir0[5] = (v259_data + (v246_data * v226_data));
          float v264_data = ir0[7];
          ir0[7] = (v264_data + (v246_data * v231_data));
          float v269_data = ir0[9];
          ir0[9] = (v269_data + (v246_data * v236_data));
          float v274_data = ir0[11];
          ir0[11] = (v274_data + (v246_data * v241_data));
          float v277_data = v26_g ? (glb_m1[(v24_lead + 200)]) : (0.0f);
          float v278_data = s0[5];
          float v280_data = ir0[1];
          ir0[1] = (v280_data + (v277_data * v278_data));
          float v283_data = s0[13];
          float v285_data = ir0[3];
          ir0[3] = (v285_data + (v277_data * v283_data));
          float v288_data = s0[21];
          float v290_data = ir0[5];
          ir0[5] = (v290_data + (v277_data * v288_data));
          float v293_data = s0[29];
          float v295_data = ir0[7];
          ir0[7] = (v295_data + (v277_data * v293_data));
          float v298_data = s0[37];
          float v300_data = ir0[9];
          ir0[9] = (v300_data + (v277_data * v298_data));
          float v303_data = s0[45];
          float v305_data = ir0[11];
          ir0[11] = (v305_data + (v277_data * v303_data));
          float v308_data = glb_m1[(v22_lead + 240)];
          float v309_data = s0[6];
          float v311_data = ir0[0];
          ir0[0] = (v311_data + (v308_data * v309_data));
          float v314_data = s0[14];
          float v316_data = ir0[2];
          ir0[2] = (v316_data + (v308_data * v314_data));
          float v319_data = s0[22];
          float v321_data = ir0[4];
          ir0[4] = (v321_data + (v308_data * v319_data));
          float v324_data = s0[30];
          float v326_data = ir0[6];
          ir0[6] = (v326_data + (v308_data * v324_data));
          float v329_data = s0[38];
          float v331_data = ir0[8];
          ir0[8] = (v331_data + (v308_data * v329_data));
          float v334_data = s0[46];
          float v336_data = ir0[10];
          ir0[10] = (v336_data + (v308_data * v334_data));
          float v339_data = v26_g ? (glb_m1[(v24_lead + 240)]) : (0.0f);
          float v342_data = ir0[1];
          ir0[1] = (v342_data + (v339_data * v309_data));
          float v347_data = ir0[3];
          ir0[3] = (v347_data + (v339_data * v314_data));
          float v352_data = ir0[5];
          ir0[5] = (v352_data + (v339_data * v319_data));
          float v357_data = ir0[7];
          ir0[7] = (v357_data + (v339_data * v324_data));
          float v362_data = ir0[9];
          ir0[9] = (v362_data + (v339_data * v329_data));
          float v367_data = ir0[11];
          ir0[11] = (v367_data + (v339_data * v334_data));
          float v370_data = glb_m1[(v22_lead + 280)];
          float v371_data = s0[7];
          float v373_data = ir0[0];
          ir0[0] = (v373_data + (v370_data * v371_data));
          float v376_data = s0[15];
          float v378_data = ir0[2];
          ir0[2] = (v378_data + (v370_data * v376_data));
          float v381_data = s0[23];
          float v383_data = ir0[4];
          ir0[4] = (v383_data + (v370_data * v381_data));
          float v386_data = s0[31];
          float v388_data = ir0[6];
          ir0[6] = (v388_data + (v370_data * v386_data));
          float v391_data = s0[39];
          float v393_data = ir0[8];
          ir0[8] = (v393_data + (v370_data * v391_data));
          float v396_data = s0[47];
          float v398_data = ir0[10];
          ir0[10] = (v398_data + (v370_data * v396_data));
          float v401_data = v26_g ? (glb_m1[(v24_lead + 280)]) : (0.0f);
          float v404_data = ir0[1];
          ir0[1] = (v404_data + (v401_data * v371_data));
          float v409_data = ir0[3];
          ir0[3] = (v409_data + (v401_data * v376_data));
          float v414_data = ir0[5];
          ir0[5] = (v414_data + (v401_data * v381_data));
          float v419_data = ir0[7];
          ir0[7] = (v419_data + (v401_data * v386_data));
          float v424_data = ir0[9];
          ir0[9] = (v424_data + (v401_data * v391_data));
          float v429_data = ir0[11];
          ir0[11] = (v429_data + (v401_data * v396_data));
          // r0 = ir0
          #pragma unroll
          for (int32_t v434_n0 = 0; v434_n0 < 1; ++v434_n0) {
            #pragma unroll
            for (int32_t v435_n1 = 0; v435_n1 < 6; ++v435_n1) {
              int32_t v437_a = v434_n0 + (v435_n1 * 2);
              float v438_data = ir0[v437_a];
              r0[v437_a] = v438_data;
            }
          }
          if (v22_lead < 8) {
            #pragma unroll
            for (int32_t v440_n1 = 0; v440_n1 < 6; ++v440_n1) {
              int32_t v442_a = 1 + (v440_n1 * 2);
              float v443_data = ir0[v442_a];
              r0[v442_a] = v443_data;
            }
          }
          // glb_m0 = store{r>g}(r0);
          #pragma unroll
          for (int32_t v447_i0 = 0; v447_i0 < 1; ++v447_i0) {
            int32_t v453_lead = v22_lead + (v447_i0 * 32);
            #pragma unroll
            for (int32_t v448_i1 = 0; v448_i1 < 6; ++v448_i1) {
              float v451_data = r0[(v447_i0 + (v448_i1 * 2))];
              glb_m0[(v453_lead + (v448_i1 * 40))] = v451_data;
            }
          }
          if (v22_lead < 8) {
            int32_t v462_lead = v22_lead + 32_i32;
            #pragma unroll
            for (int32_t v457_i1 = 0; v457_i1 < 6; ++v457_i1) {
              float v460_data = r0[(1 + (v457_i1 * 2))];
              glb_m0[(v462_lead + (v457_i1 * 40))] = v460_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

