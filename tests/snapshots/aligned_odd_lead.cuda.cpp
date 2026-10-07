// === base name ===
kernel_e4cf4960b36e99de

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_e4cf4960b36e99de = {{32, 4, 1}, 32, 35, 1, 4, 512, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_e4cf4960b36e99de(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_e4cf4960b36e99de(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_e4cf4960b36e99de(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_e4cf4960b36e99de, block.x * block.y * block.z, 128 * sizeof(float));
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
void launcher_kernel_e4cf4960b36e99de(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_e4cf4960b36e99de(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_e4cf4960b36e99de, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_e4cf4960b36e99de<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_e4cf4960b36e99de(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (35 active) x 4 per block = block 32x4x1, 512 B shared, occupancy grid
    // operands:
    //   m0 35×4(35×4) {0..35}×{0..4} strided
    //   m1 35×8(35×8) {0..35}×{0..8} strided
    //   m2 8×4(8×4) {0..8}×{0..4} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":35,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":128}],"shared_bytes":512,"shared_elements":128,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[35,4]],"name":"m0","ordered":false,"parts":1,"shape":[35,4],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[35,8]],"name":"m1","ordered":false,"parts":1,"shape":[35,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,4]],"name":"m2","ordered":false,"parts":1,"shape":[8,4],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[35,4]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[35,4]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[35,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[35,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[32 * threadIdx.y + 0];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 140 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 280 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 32 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v22_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
            int32_t v26_lead = v22_lead + (v23_i0 * 32);
            #pragma unroll
            for (int32_t v24_i1 = 0; v24_i1 < 8; ++v24_i1) {
              float v29_data = __ldcg(&glb_m1[(v26_lead + (v24_i1 * 35))]);
              r0[(v23_i0 + (v24_i1 * 2))] = v29_data;
            }
          }
          bool v32_g = v22_lead < 3;
          if (v32_g) {
            int32_t v35_lead = v22_lead + 32_i32;
            #pragma unroll
            for (int32_t v33_i1 = 0; v33_i1 < 8; ++v33_i1) {
              float v38_data = __ldcg(&glb_m1[(v35_lead + (v33_i1 * 35))]);
              r0[(1 + (v33_i1 * 2))] = v38_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_commit();
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          // ir1 = +(r0 * s0)
          // [(0, 35), (0, 4)] [(0, 8)]
          float ir1[8]{};
          float v44_data = r0[0];
          __syncwarp();
          float v45_data = s0[0];
          float v47_data = ir1[0];
          ir1[0] = (v47_data + (v44_data * v45_data));
          float v50_data = s0[8];
          float v52_data = ir1[2];
          ir1[2] = (v52_data + (v44_data * v50_data));
          float v55_data = s0[16];
          float v57_data = ir1[4];
          ir1[4] = (v57_data + (v44_data * v55_data));
          float v60_data = s0[24];
          float v62_data = ir1[6];
          ir1[6] = (v62_data + (v44_data * v60_data));
          float v64_data = r0[1];
          float v67_data = ir1[1];
          ir1[1] = (v67_data + (v64_data * v45_data));
          float v72_data = ir1[3];
          ir1[3] = (v72_data + (v64_data * v50_data));
          float v77_data = ir1[5];
          ir1[5] = (v77_data + (v64_data * v55_data));
          float v82_data = ir1[7];
          ir1[7] = (v82_data + (v64_data * v60_data));
          float v84_data = r0[2];
          float v85_data = s0[1];
          float v87_data = ir1[0];
          ir1[0] = (v87_data + (v84_data * v85_data));
          float v90_data = s0[9];
          float v92_data = ir1[2];
          ir1[2] = (v92_data + (v84_data * v90_data));
          float v95_data = s0[17];
          float v97_data = ir1[4];
          ir1[4] = (v97_data + (v84_data * v95_data));
          float v100_data = s0[25];
          float v102_data = ir1[6];
          ir1[6] = (v102_data + (v84_data * v100_data));
          float v104_data = r0[3];
          float v107_data = ir1[1];
          ir1[1] = (v107_data + (v104_data * v85_data));
          float v112_data = ir1[3];
          ir1[3] = (v112_data + (v104_data * v90_data));
          float v117_data = ir1[5];
          ir1[5] = (v117_data + (v104_data * v95_data));
          float v122_data = ir1[7];
          ir1[7] = (v122_data + (v104_data * v100_data));
          float v124_data = r0[4];
          float v125_data = s0[2];
          float v127_data = ir1[0];
          ir1[0] = (v127_data + (v124_data * v125_data));
          float v130_data = s0[10];
          float v132_data = ir1[2];
          ir1[2] = (v132_data + (v124_data * v130_data));
          float v135_data = s0[18];
          float v137_data = ir1[4];
          ir1[4] = (v137_data + (v124_data * v135_data));
          float v140_data = s0[26];
          float v142_data = ir1[6];
          ir1[6] = (v142_data + (v124_data * v140_data));
          float v144_data = r0[5];
          float v147_data = ir1[1];
          ir1[1] = (v147_data + (v144_data * v125_data));
          float v152_data = ir1[3];
          ir1[3] = (v152_data + (v144_data * v130_data));
          float v157_data = ir1[5];
          ir1[5] = (v157_data + (v144_data * v135_data));
          float v162_data = ir1[7];
          ir1[7] = (v162_data + (v144_data * v140_data));
          float v164_data = r0[6];
          float v165_data = s0[3];
          float v167_data = ir1[0];
          ir1[0] = (v167_data + (v164_data * v165_data));
          float v170_data = s0[11];
          float v172_data = ir1[2];
          ir1[2] = (v172_data + (v164_data * v170_data));
          float v175_data = s0[19];
          float v177_data = ir1[4];
          ir1[4] = (v177_data + (v164_data * v175_data));
          float v180_data = s0[27];
          float v182_data = ir1[6];
          ir1[6] = (v182_data + (v164_data * v180_data));
          float v184_data = r0[7];
          float v187_data = ir1[1];
          ir1[1] = (v187_data + (v184_data * v165_data));
          float v192_data = ir1[3];
          ir1[3] = (v192_data + (v184_data * v170_data));
          float v197_data = ir1[5];
          ir1[5] = (v197_data + (v184_data * v175_data));
          float v202_data = ir1[7];
          ir1[7] = (v202_data + (v184_data * v180_data));
          float v204_data = r0[8];
          float v205_data = s0[4];
          float v207_data = ir1[0];
          ir1[0] = (v207_data + (v204_data * v205_data));
          float v210_data = s0[12];
          float v212_data = ir1[2];
          ir1[2] = (v212_data + (v204_data * v210_data));
          float v215_data = s0[20];
          float v217_data = ir1[4];
          ir1[4] = (v217_data + (v204_data * v215_data));
          float v220_data = s0[28];
          float v222_data = ir1[6];
          ir1[6] = (v222_data + (v204_data * v220_data));
          float v224_data = r0[9];
          float v227_data = ir1[1];
          ir1[1] = (v227_data + (v224_data * v205_data));
          float v232_data = ir1[3];
          ir1[3] = (v232_data + (v224_data * v210_data));
          float v237_data = ir1[5];
          ir1[5] = (v237_data + (v224_data * v215_data));
          float v242_data = ir1[7];
          ir1[7] = (v242_data + (v224_data * v220_data));
          float v244_data = r0[10];
          float v245_data = s0[5];
          float v247_data = ir1[0];
          ir1[0] = (v247_data + (v244_data * v245_data));
          float v250_data = s0[13];
          float v252_data = ir1[2];
          ir1[2] = (v252_data + (v244_data * v250_data));
          float v255_data = s0[21];
          float v257_data = ir1[4];
          ir1[4] = (v257_data + (v244_data * v255_data));
          float v260_data = s0[29];
          float v262_data = ir1[6];
          ir1[6] = (v262_data + (v244_data * v260_data));
          float v264_data = r0[11];
          float v267_data = ir1[1];
          ir1[1] = (v267_data + (v264_data * v245_data));
          float v272_data = ir1[3];
          ir1[3] = (v272_data + (v264_data * v250_data));
          float v277_data = ir1[5];
          ir1[5] = (v277_data + (v264_data * v255_data));
          float v282_data = ir1[7];
          ir1[7] = (v282_data + (v264_data * v260_data));
          float v284_data = r0[12];
          float v285_data = s0[6];
          float v287_data = ir1[0];
          ir1[0] = (v287_data + (v284_data * v285_data));
          float v290_data = s0[14];
          float v292_data = ir1[2];
          ir1[2] = (v292_data + (v284_data * v290_data));
          float v295_data = s0[22];
          float v297_data = ir1[4];
          ir1[4] = (v297_data + (v284_data * v295_data));
          float v300_data = s0[30];
          float v302_data = ir1[6];
          ir1[6] = (v302_data + (v284_data * v300_data));
          float v304_data = r0[13];
          float v307_data = ir1[1];
          ir1[1] = (v307_data + (v304_data * v285_data));
          float v312_data = ir1[3];
          ir1[3] = (v312_data + (v304_data * v290_data));
          float v317_data = ir1[5];
          ir1[5] = (v317_data + (v304_data * v295_data));
          float v322_data = ir1[7];
          ir1[7] = (v322_data + (v304_data * v300_data));
          float v324_data = r0[14];
          float v325_data = s0[7];
          float v327_data = ir1[0];
          ir1[0] = (v327_data + (v324_data * v325_data));
          float v330_data = s0[15];
          float v332_data = ir1[2];
          ir1[2] = (v332_data + (v324_data * v330_data));
          float v335_data = s0[23];
          float v337_data = ir1[4];
          ir1[4] = (v337_data + (v324_data * v335_data));
          float v340_data = s0[31];
          float v342_data = ir1[6];
          ir1[6] = (v342_data + (v324_data * v340_data));
          float v344_data = r0[15];
          float v347_data = ir1[1];
          ir1[1] = (v347_data + (v344_data * v325_data));
          float v352_data = ir1[3];
          ir1[3] = (v352_data + (v344_data * v330_data));
          float v357_data = ir1[5];
          ir1[5] = (v357_data + (v344_data * v335_data));
          float v362_data = ir1[7];
          ir1[7] = (v362_data + (v344_data * v340_data));
          // r1 = ir1
          #pragma unroll
          for (int32_t v364_n0 = 0; v364_n0 < 1; ++v364_n0) {
            #pragma unroll
            for (int32_t v365_n1 = 0; v365_n1 < 4; ++v365_n1) {
              int32_t v367_a = v364_n0 + (v365_n1 * 2);
              float v368_data = ir1[v367_a];
              r1[v367_a] = v368_data;
            }
          }
          if (v32_g) {
            #pragma unroll
            for (int32_t v369_n1 = 0; v369_n1 < 4; ++v369_n1) {
              int32_t v371_a = 1 + (v369_n1 * 2);
              float v372_data = ir1[v371_a];
              r1[v371_a] = v372_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v373_i0 = 0; v373_i0 < 1; ++v373_i0) {
            int32_t v379_lead = v22_lead + (v373_i0 * 32);
            #pragma unroll
            for (int32_t v374_i1 = 0; v374_i1 < 4; ++v374_i1) {
              float v377_data = r1[(v373_i0 + (v374_i1 * 2))];
              glb_m0[(v379_lead + (v374_i1 * 35))] = v377_data;
            }
          }
          if (v32_g) {
            int32_t v387_lead = v22_lead + 32_i32;
            #pragma unroll
            for (int32_t v382_i1 = 0; v382_i1 < 4; ++v382_i1) {
              float v385_data = r1[(1 + (v382_i1 * 2))];
              glb_m0[(v387_lead + (v382_i1 * 35))] = v385_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

