// === base name ===
kernel_4eb9f39d05617db4

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_4eb9f39d05617db4 = {{32, 4, 1}, 32, 40, 1, 4, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_4eb9f39d05617db4(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_4eb9f39d05617db4(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_4eb9f39d05617db4(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_4eb9f39d05617db4, block.x * block.y * block.z, 256 * sizeof(float));
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
void launcher_kernel_4eb9f39d05617db4(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_4eb9f39d05617db4(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_4eb9f39d05617db4, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_4eb9f39d05617db4<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_4eb9f39d05617db4(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (40 active) x 4 per block = block 32x4x1, 1024 B shared, occupancy grid
    // operands:
    //   m0 40×6(40×6) {0..40}×{0..6} strided
    //   m1 40×8(40×8) {0..40}×{0..8} none
    //   m2 8×6(8×6) {0..8}×{0..6} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":40,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[40,6]],"name":"m0","ordered":false,"parts":1,"shape":[40,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[40,8]],"name":"m1","ordered":false,"parts":1,"shape":[40,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,6]],"name":"m2","ordered":false,"parts":1,"shape":[8,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[40,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[40,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[40,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[40,8]},{"addressing":"strided","bbox":[[0,0],[8,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[64 * threadIdx.y + 0];
      const float *const __restrict__ glb_m1 = &m1[0];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < numElements0; v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 240 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v9_batchId0 * 48 + 0 + m2_extraOffset];
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
          if (threadIdx.x < 16) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
          }
          __pipeline_commit();
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r0[12]{};
          // ir0 = +(glb_m1 * s0)
          // [(0, 40), (0, 6)] [(0, 8)]
          float ir0[12]{};
          int32_t v25_lead = threadIdx.x % 32;
          int32_t v27_lead = v25_lead + 32_i32;
          bool v29_g = v25_lead < 8;
          float v30_data = v29_g ? (glb_m1[v27_lead]) : (0.0f);
          __syncwarp();
          float v31_data = s0[0];
          float v33_data = ir0[1];
          ir0[1] = (v33_data + (v30_data * v31_data));
          float v36_data = s0[8];
          float v38_data = ir0[3];
          ir0[3] = (v38_data + (v30_data * v36_data));
          float v41_data = s0[16];
          float v43_data = ir0[5];
          ir0[5] = (v43_data + (v30_data * v41_data));
          float v46_data = s0[24];
          float v48_data = ir0[7];
          ir0[7] = (v48_data + (v30_data * v46_data));
          float v51_data = s0[32];
          float v53_data = ir0[9];
          ir0[9] = (v53_data + (v30_data * v51_data));
          float v56_data = s0[40];
          float v58_data = ir0[11];
          ir0[11] = (v58_data + (v30_data * v56_data));
          float v63_data = glb_m1[(v25_lead + 40)];
          float v64_data = s0[1];
          float v66_data = ir0[0];
          ir0[0] = (v66_data + (v63_data * v64_data));
          float v69_data = s0[9];
          float v71_data = ir0[2];
          ir0[2] = (v71_data + (v63_data * v69_data));
          float v74_data = s0[17];
          float v76_data = ir0[4];
          ir0[4] = (v76_data + (v63_data * v74_data));
          float v79_data = s0[25];
          float v81_data = ir0[6];
          ir0[6] = (v81_data + (v63_data * v79_data));
          float v84_data = s0[33];
          float v86_data = ir0[8];
          ir0[8] = (v86_data + (v63_data * v84_data));
          float v89_data = s0[41];
          float v91_data = ir0[10];
          ir0[10] = (v91_data + (v63_data * v89_data));
          float v94_data = v29_g ? (glb_m1[(v27_lead + 40)]) : (0.0f);
          float v97_data = ir0[1];
          ir0[1] = (v97_data + (v94_data * v64_data));
          float v102_data = ir0[3];
          ir0[3] = (v102_data + (v94_data * v69_data));
          float v107_data = ir0[5];
          ir0[5] = (v107_data + (v94_data * v74_data));
          float v112_data = ir0[7];
          ir0[7] = (v112_data + (v94_data * v79_data));
          float v117_data = ir0[9];
          ir0[9] = (v117_data + (v94_data * v84_data));
          float v122_data = ir0[11];
          ir0[11] = (v122_data + (v94_data * v89_data));
          float v125_data = v29_g ? (glb_m1[(v27_lead + 80)]) : (0.0f);
          float v126_data = s0[2];
          float v128_data = ir0[1];
          ir0[1] = (v128_data + (v125_data * v126_data));
          float v131_data = s0[10];
          float v133_data = ir0[3];
          ir0[3] = (v133_data + (v125_data * v131_data));
          float v136_data = s0[18];
          float v138_data = ir0[5];
          ir0[5] = (v138_data + (v125_data * v136_data));
          float v141_data = s0[26];
          float v143_data = ir0[7];
          ir0[7] = (v143_data + (v125_data * v141_data));
          float v146_data = s0[34];
          float v148_data = ir0[9];
          ir0[9] = (v148_data + (v125_data * v146_data));
          float v151_data = s0[42];
          float v153_data = ir0[11];
          ir0[11] = (v153_data + (v125_data * v151_data));
          float v156_data = glb_m1[(v25_lead + 120)];
          float v157_data = s0[3];
          float v159_data = ir0[0];
          ir0[0] = (v159_data + (v156_data * v157_data));
          float v162_data = s0[11];
          float v164_data = ir0[2];
          ir0[2] = (v164_data + (v156_data * v162_data));
          float v167_data = s0[19];
          float v169_data = ir0[4];
          ir0[4] = (v169_data + (v156_data * v167_data));
          float v172_data = s0[27];
          float v174_data = ir0[6];
          ir0[6] = (v174_data + (v156_data * v172_data));
          float v177_data = s0[35];
          float v179_data = ir0[8];
          ir0[8] = (v179_data + (v156_data * v177_data));
          float v182_data = s0[43];
          float v184_data = ir0[10];
          ir0[10] = (v184_data + (v156_data * v182_data));
          float v187_data = v29_g ? (glb_m1[(v27_lead + 120)]) : (0.0f);
          float v190_data = ir0[1];
          ir0[1] = (v190_data + (v187_data * v157_data));
          float v195_data = ir0[3];
          ir0[3] = (v195_data + (v187_data * v162_data));
          float v200_data = ir0[5];
          ir0[5] = (v200_data + (v187_data * v167_data));
          float v205_data = ir0[7];
          ir0[7] = (v205_data + (v187_data * v172_data));
          float v210_data = ir0[9];
          ir0[9] = (v210_data + (v187_data * v177_data));
          float v215_data = ir0[11];
          ir0[11] = (v215_data + (v187_data * v182_data));
          float v218_data = glb_m1[(v25_lead + 160)];
          float v219_data = s0[4];
          float v221_data = ir0[0];
          ir0[0] = (v221_data + (v218_data * v219_data));
          float v224_data = s0[12];
          float v226_data = ir0[2];
          ir0[2] = (v226_data + (v218_data * v224_data));
          float v229_data = s0[20];
          float v231_data = ir0[4];
          ir0[4] = (v231_data + (v218_data * v229_data));
          float v234_data = s0[28];
          float v236_data = ir0[6];
          ir0[6] = (v236_data + (v218_data * v234_data));
          float v239_data = s0[36];
          float v241_data = ir0[8];
          ir0[8] = (v241_data + (v218_data * v239_data));
          float v244_data = s0[44];
          float v246_data = ir0[10];
          ir0[10] = (v246_data + (v218_data * v244_data));
          float v249_data = v29_g ? (glb_m1[(v27_lead + 160)]) : (0.0f);
          float v252_data = ir0[1];
          ir0[1] = (v252_data + (v249_data * v219_data));
          float v257_data = ir0[3];
          ir0[3] = (v257_data + (v249_data * v224_data));
          float v262_data = ir0[5];
          ir0[5] = (v262_data + (v249_data * v229_data));
          float v267_data = ir0[7];
          ir0[7] = (v267_data + (v249_data * v234_data));
          float v272_data = ir0[9];
          ir0[9] = (v272_data + (v249_data * v239_data));
          float v277_data = ir0[11];
          ir0[11] = (v277_data + (v249_data * v244_data));
          float v280_data = v29_g ? (glb_m1[(v27_lead + 200)]) : (0.0f);
          float v281_data = s0[5];
          float v283_data = ir0[1];
          ir0[1] = (v283_data + (v280_data * v281_data));
          float v286_data = s0[13];
          float v288_data = ir0[3];
          ir0[3] = (v288_data + (v280_data * v286_data));
          float v291_data = s0[21];
          float v293_data = ir0[5];
          ir0[5] = (v293_data + (v280_data * v291_data));
          float v296_data = s0[29];
          float v298_data = ir0[7];
          ir0[7] = (v298_data + (v280_data * v296_data));
          float v301_data = s0[37];
          float v303_data = ir0[9];
          ir0[9] = (v303_data + (v280_data * v301_data));
          float v306_data = s0[45];
          float v308_data = ir0[11];
          ir0[11] = (v308_data + (v280_data * v306_data));
          float v311_data = glb_m1[(v25_lead + 240)];
          float v312_data = s0[6];
          float v314_data = ir0[0];
          ir0[0] = (v314_data + (v311_data * v312_data));
          float v317_data = s0[14];
          float v319_data = ir0[2];
          ir0[2] = (v319_data + (v311_data * v317_data));
          float v322_data = s0[22];
          float v324_data = ir0[4];
          ir0[4] = (v324_data + (v311_data * v322_data));
          float v327_data = s0[30];
          float v329_data = ir0[6];
          ir0[6] = (v329_data + (v311_data * v327_data));
          float v332_data = s0[38];
          float v334_data = ir0[8];
          ir0[8] = (v334_data + (v311_data * v332_data));
          float v337_data = s0[46];
          float v339_data = ir0[10];
          ir0[10] = (v339_data + (v311_data * v337_data));
          float v342_data = v29_g ? (glb_m1[(v27_lead + 240)]) : (0.0f);
          float v345_data = ir0[1];
          ir0[1] = (v345_data + (v342_data * v312_data));
          float v350_data = ir0[3];
          ir0[3] = (v350_data + (v342_data * v317_data));
          float v355_data = ir0[5];
          ir0[5] = (v355_data + (v342_data * v322_data));
          float v360_data = ir0[7];
          ir0[7] = (v360_data + (v342_data * v327_data));
          float v365_data = ir0[9];
          ir0[9] = (v365_data + (v342_data * v332_data));
          float v370_data = ir0[11];
          ir0[11] = (v370_data + (v342_data * v337_data));
          float v373_data = glb_m1[(v25_lead + 280)];
          float v374_data = s0[7];
          float v376_data = ir0[0];
          ir0[0] = (v376_data + (v373_data * v374_data));
          float v379_data = s0[15];
          float v381_data = ir0[2];
          ir0[2] = (v381_data + (v373_data * v379_data));
          float v384_data = s0[23];
          float v386_data = ir0[4];
          ir0[4] = (v386_data + (v373_data * v384_data));
          float v389_data = s0[31];
          float v391_data = ir0[6];
          ir0[6] = (v391_data + (v373_data * v389_data));
          float v394_data = s0[39];
          float v396_data = ir0[8];
          ir0[8] = (v396_data + (v373_data * v394_data));
          float v399_data = s0[47];
          float v401_data = ir0[10];
          ir0[10] = (v401_data + (v373_data * v399_data));
          float v404_data = v29_g ? (glb_m1[(v27_lead + 280)]) : (0.0f);
          float v407_data = ir0[1];
          ir0[1] = (v407_data + (v404_data * v374_data));
          float v412_data = ir0[3];
          ir0[3] = (v412_data + (v404_data * v379_data));
          float v417_data = ir0[5];
          ir0[5] = (v417_data + (v404_data * v384_data));
          float v422_data = ir0[7];
          ir0[7] = (v422_data + (v404_data * v389_data));
          float v427_data = ir0[9];
          ir0[9] = (v427_data + (v404_data * v394_data));
          float v432_data = ir0[11];
          ir0[11] = (v432_data + (v404_data * v399_data));
          // r0 = ir0
          #pragma unroll
          for (int32_t v437_n0 = 0; v437_n0 < 1; ++v437_n0) {
            #pragma unroll
            for (int32_t v438_n1 = 0; v438_n1 < 6; ++v438_n1) {
              int32_t v440_a = v437_n0 + (v438_n1 * 2);
              float v441_data = ir0[v440_a];
              r0[v440_a] = v441_data;
            }
          }
          if (v25_lead < 8) {
            #pragma unroll
            for (int32_t v443_n1 = 0; v443_n1 < 6; ++v443_n1) {
              int32_t v445_a = 1 + (v443_n1 * 2);
              float v446_data = ir0[v445_a];
              r0[v445_a] = v446_data;
            }
          }
          // glb_m0 = store{r>g}(r0);
          #pragma unroll
          for (int32_t v450_i0 = 0; v450_i0 < 1; ++v450_i0) {
            int32_t v456_lead = v25_lead + (v450_i0 * 32);
            #pragma unroll
            for (int32_t v451_i1 = 0; v451_i1 < 6; ++v451_i1) {
              float v454_data = r0[(v450_i0 + (v451_i1 * 2))];
              glb_m0[(v456_lead + (v451_i1 * 40))] = v454_data;
            }
          }
          if (v25_lead < 8) {
            int32_t v465_lead = v25_lead + 32_i32;
            #pragma unroll
            for (int32_t v460_i1 = 0; v460_i1 < 6; ++v460_i1) {
              float v463_data = r0[(1 + (v460_i1 * 2))];
              glb_m0[(v465_lead + (v460_i1 * 40))] = v463_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

