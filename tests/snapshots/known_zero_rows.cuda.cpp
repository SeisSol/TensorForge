// === base name ===
kernel_74fa7aa669fbd09a

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_74fa7aa669fbd09a = {{32, 4, 1}, 32, 40, 1, 4, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_74fa7aa669fbd09a(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_74fa7aa669fbd09a(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_74fa7aa669fbd09a(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_74fa7aa669fbd09a, block.x * block.y * block.z, 256 * sizeof(float));
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
void launcher_kernel_74fa7aa669fbd09a(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_74fa7aa669fbd09a(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_74fa7aa669fbd09a, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_74fa7aa669fbd09a<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_74fa7aa669fbd09a(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
      float* tempShrMem = &localShrMem0[64];
      const float *const __restrict__ glb_m1 = &m1[0];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v12_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v12_batchId0 < numElements0; v12_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v13_ahead1 = v12_batchId0 + (gridDim.x * blockDim.y);
        size_t v15_batchId1 = (v13_ahead1 < numElements0) ? v13_ahead1 : v12_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v12_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v12_batchId0 * 240 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v12_batchId0 * 48 + 0 + m2_extraOffset];
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
          int32_t v28_lead = threadIdx.x % 32;
          int32_t v30_lead = v28_lead + 32_i32;
          bool v32_g = v28_lead < 8;
          float v33_data = v32_g ? (glb_m1[v30_lead]) : (0.0f);
          float v34_data = s0[0];
          float v36_data = ir0[1];
          ir0[1] = (v36_data + (v33_data * v34_data));
          float v39_data = s0[8];
          float v41_data = ir0[3];
          ir0[3] = (v41_data + (v33_data * v39_data));
          float v44_data = s0[16];
          float v46_data = ir0[5];
          ir0[5] = (v46_data + (v33_data * v44_data));
          float v49_data = s0[24];
          float v51_data = ir0[7];
          ir0[7] = (v51_data + (v33_data * v49_data));
          float v54_data = s0[32];
          float v56_data = ir0[9];
          ir0[9] = (v56_data + (v33_data * v54_data));
          float v59_data = s0[40];
          float v61_data = ir0[11];
          ir0[11] = (v61_data + (v33_data * v59_data));
          float v66_data = glb_m1[(v28_lead + 40)];
          float v67_data = s0[1];
          float v69_data = ir0[0];
          ir0[0] = (v69_data + (v66_data * v67_data));
          float v72_data = s0[9];
          float v74_data = ir0[2];
          ir0[2] = (v74_data + (v66_data * v72_data));
          float v77_data = s0[17];
          float v79_data = ir0[4];
          ir0[4] = (v79_data + (v66_data * v77_data));
          float v82_data = s0[25];
          float v84_data = ir0[6];
          ir0[6] = (v84_data + (v66_data * v82_data));
          float v87_data = s0[33];
          float v89_data = ir0[8];
          ir0[8] = (v89_data + (v66_data * v87_data));
          float v92_data = s0[41];
          float v94_data = ir0[10];
          ir0[10] = (v94_data + (v66_data * v92_data));
          float v97_data = v32_g ? (glb_m1[(v30_lead + 40)]) : (0.0f);
          float v100_data = ir0[1];
          ir0[1] = (v100_data + (v97_data * v67_data));
          float v105_data = ir0[3];
          ir0[3] = (v105_data + (v97_data * v72_data));
          float v110_data = ir0[5];
          ir0[5] = (v110_data + (v97_data * v77_data));
          float v115_data = ir0[7];
          ir0[7] = (v115_data + (v97_data * v82_data));
          float v120_data = ir0[9];
          ir0[9] = (v120_data + (v97_data * v87_data));
          float v125_data = ir0[11];
          ir0[11] = (v125_data + (v97_data * v92_data));
          float v128_data = v32_g ? (glb_m1[(v30_lead + 80)]) : (0.0f);
          float v129_data = s0[2];
          float v131_data = ir0[1];
          ir0[1] = (v131_data + (v128_data * v129_data));
          float v134_data = s0[10];
          float v136_data = ir0[3];
          ir0[3] = (v136_data + (v128_data * v134_data));
          float v139_data = s0[18];
          float v141_data = ir0[5];
          ir0[5] = (v141_data + (v128_data * v139_data));
          float v144_data = s0[26];
          float v146_data = ir0[7];
          ir0[7] = (v146_data + (v128_data * v144_data));
          float v149_data = s0[34];
          float v151_data = ir0[9];
          ir0[9] = (v151_data + (v128_data * v149_data));
          float v154_data = s0[42];
          float v156_data = ir0[11];
          ir0[11] = (v156_data + (v128_data * v154_data));
          float v159_data = glb_m1[(v28_lead + 120)];
          float v160_data = s0[3];
          float v162_data = ir0[0];
          ir0[0] = (v162_data + (v159_data * v160_data));
          float v165_data = s0[11];
          float v167_data = ir0[2];
          ir0[2] = (v167_data + (v159_data * v165_data));
          float v170_data = s0[19];
          float v172_data = ir0[4];
          ir0[4] = (v172_data + (v159_data * v170_data));
          float v175_data = s0[27];
          float v177_data = ir0[6];
          ir0[6] = (v177_data + (v159_data * v175_data));
          float v180_data = s0[35];
          float v182_data = ir0[8];
          ir0[8] = (v182_data + (v159_data * v180_data));
          float v185_data = s0[43];
          float v187_data = ir0[10];
          ir0[10] = (v187_data + (v159_data * v185_data));
          float v190_data = v32_g ? (glb_m1[(v30_lead + 120)]) : (0.0f);
          float v193_data = ir0[1];
          ir0[1] = (v193_data + (v190_data * v160_data));
          float v198_data = ir0[3];
          ir0[3] = (v198_data + (v190_data * v165_data));
          float v203_data = ir0[5];
          ir0[5] = (v203_data + (v190_data * v170_data));
          float v208_data = ir0[7];
          ir0[7] = (v208_data + (v190_data * v175_data));
          float v213_data = ir0[9];
          ir0[9] = (v213_data + (v190_data * v180_data));
          float v218_data = ir0[11];
          ir0[11] = (v218_data + (v190_data * v185_data));
          float v221_data = glb_m1[(v28_lead + 160)];
          float v222_data = s0[4];
          float v224_data = ir0[0];
          ir0[0] = (v224_data + (v221_data * v222_data));
          float v227_data = s0[12];
          float v229_data = ir0[2];
          ir0[2] = (v229_data + (v221_data * v227_data));
          float v232_data = s0[20];
          float v234_data = ir0[4];
          ir0[4] = (v234_data + (v221_data * v232_data));
          float v237_data = s0[28];
          float v239_data = ir0[6];
          ir0[6] = (v239_data + (v221_data * v237_data));
          float v242_data = s0[36];
          float v244_data = ir0[8];
          ir0[8] = (v244_data + (v221_data * v242_data));
          float v247_data = s0[44];
          float v249_data = ir0[10];
          ir0[10] = (v249_data + (v221_data * v247_data));
          float v252_data = v32_g ? (glb_m1[(v30_lead + 160)]) : (0.0f);
          float v255_data = ir0[1];
          ir0[1] = (v255_data + (v252_data * v222_data));
          float v260_data = ir0[3];
          ir0[3] = (v260_data + (v252_data * v227_data));
          float v265_data = ir0[5];
          ir0[5] = (v265_data + (v252_data * v232_data));
          float v270_data = ir0[7];
          ir0[7] = (v270_data + (v252_data * v237_data));
          float v275_data = ir0[9];
          ir0[9] = (v275_data + (v252_data * v242_data));
          float v280_data = ir0[11];
          ir0[11] = (v280_data + (v252_data * v247_data));
          float v283_data = v32_g ? (glb_m1[(v30_lead + 200)]) : (0.0f);
          float v284_data = s0[5];
          float v286_data = ir0[1];
          ir0[1] = (v286_data + (v283_data * v284_data));
          float v289_data = s0[13];
          float v291_data = ir0[3];
          ir0[3] = (v291_data + (v283_data * v289_data));
          float v294_data = s0[21];
          float v296_data = ir0[5];
          ir0[5] = (v296_data + (v283_data * v294_data));
          float v299_data = s0[29];
          float v301_data = ir0[7];
          ir0[7] = (v301_data + (v283_data * v299_data));
          float v304_data = s0[37];
          float v306_data = ir0[9];
          ir0[9] = (v306_data + (v283_data * v304_data));
          float v309_data = s0[45];
          float v311_data = ir0[11];
          ir0[11] = (v311_data + (v283_data * v309_data));
          float v314_data = glb_m1[(v28_lead + 240)];
          float v315_data = s0[6];
          float v317_data = ir0[0];
          ir0[0] = (v317_data + (v314_data * v315_data));
          float v320_data = s0[14];
          float v322_data = ir0[2];
          ir0[2] = (v322_data + (v314_data * v320_data));
          float v325_data = s0[22];
          float v327_data = ir0[4];
          ir0[4] = (v327_data + (v314_data * v325_data));
          float v330_data = s0[30];
          float v332_data = ir0[6];
          ir0[6] = (v332_data + (v314_data * v330_data));
          float v335_data = s0[38];
          float v337_data = ir0[8];
          ir0[8] = (v337_data + (v314_data * v335_data));
          float v340_data = s0[46];
          float v342_data = ir0[10];
          ir0[10] = (v342_data + (v314_data * v340_data));
          float v345_data = v32_g ? (glb_m1[(v30_lead + 240)]) : (0.0f);
          float v348_data = ir0[1];
          ir0[1] = (v348_data + (v345_data * v315_data));
          float v353_data = ir0[3];
          ir0[3] = (v353_data + (v345_data * v320_data));
          float v358_data = ir0[5];
          ir0[5] = (v358_data + (v345_data * v325_data));
          float v363_data = ir0[7];
          ir0[7] = (v363_data + (v345_data * v330_data));
          float v368_data = ir0[9];
          ir0[9] = (v368_data + (v345_data * v335_data));
          float v373_data = ir0[11];
          ir0[11] = (v373_data + (v345_data * v340_data));
          float v376_data = glb_m1[(v28_lead + 280)];
          float v377_data = s0[7];
          float v379_data = ir0[0];
          ir0[0] = (v379_data + (v376_data * v377_data));
          float v382_data = s0[15];
          float v384_data = ir0[2];
          ir0[2] = (v384_data + (v376_data * v382_data));
          float v387_data = s0[23];
          float v389_data = ir0[4];
          ir0[4] = (v389_data + (v376_data * v387_data));
          float v392_data = s0[31];
          float v394_data = ir0[6];
          ir0[6] = (v394_data + (v376_data * v392_data));
          float v397_data = s0[39];
          float v399_data = ir0[8];
          ir0[8] = (v399_data + (v376_data * v397_data));
          float v402_data = s0[47];
          float v404_data = ir0[10];
          ir0[10] = (v404_data + (v376_data * v402_data));
          float v407_data = v32_g ? (glb_m1[(v30_lead + 280)]) : (0.0f);
          float v410_data = ir0[1];
          ir0[1] = (v410_data + (v407_data * v377_data));
          float v415_data = ir0[3];
          ir0[3] = (v415_data + (v407_data * v382_data));
          float v420_data = ir0[5];
          ir0[5] = (v420_data + (v407_data * v387_data));
          float v425_data = ir0[7];
          ir0[7] = (v425_data + (v407_data * v392_data));
          float v430_data = ir0[9];
          ir0[9] = (v430_data + (v407_data * v397_data));
          float v435_data = ir0[11];
          ir0[11] = (v435_data + (v407_data * v402_data));
          // r0 = ir0
          #pragma unroll
          for (int32_t v440_n0 = 0; v440_n0 < 1; ++v440_n0) {
            #pragma unroll
            for (int32_t v441_n1 = 0; v441_n1 < 6; ++v441_n1) {
              int32_t v443_a = v440_n0 + (v441_n1 * 2);
              float v444_data = ir0[v443_a];
              r0[v443_a] = v444_data;
            }
          }
          if (v28_lead < 8) {
            #pragma unroll
            for (int32_t v446_n1 = 0; v446_n1 < 6; ++v446_n1) {
              int32_t v448_a = 1 + (v446_n1 * 2);
              float v449_data = ir0[v448_a];
              r0[v448_a] = v449_data;
            }
          }
          // glb_m0 = store{r>g}(r0);
          #pragma unroll
          for (int32_t v453_i0 = 0; v453_i0 < 1; ++v453_i0) {
            int32_t v459_lead = v28_lead + (v453_i0 * 32);
            #pragma unroll
            for (int32_t v454_i1 = 0; v454_i1 < 6; ++v454_i1) {
              float v457_data = r0[(v453_i0 + (v454_i1 * 2))];
              glb_m0[(v459_lead + (v454_i1 * 40))] = v457_data;
            }
          }
          if (v28_lead < 8) {
            int32_t v468_lead = v28_lead + 32_i32;
            #pragma unroll
            for (int32_t v463_i1 = 0; v463_i1 < 6; ++v463_i1) {
              float v466_data = r0[(1 + (v463_i1 * 2))];
              glb_m0[(v468_lead + (v463_i1 * 40))] = v466_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

