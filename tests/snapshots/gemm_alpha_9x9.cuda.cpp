// === base name ===
kernel_b4e495a49670db83

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b4e495a49670db83 = {{16, 8, 1}, 16, 9, 1, 8, 3584, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b4e495a49670db83(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b4e495a49670db83(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b4e495a49670db83(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_b4e495a49670db83, block.x * block.y * block.z, 896 * sizeof(float));
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
  config.block[0] = 16;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 896 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_b4e495a49670db83(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b4e495a49670db83(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_b4e495a49670db83, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_b4e495a49670db83<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_b4e495a49670db83(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (9 active) x 8 per block = block 16x8x1, 3584 B shared, occupancy grid
    // operands:
    //   m0 9×9(9×9) {0..9}×{0..9} strided
    //   m1 9×9(9×9) {0..9}×{0..9} strided
    //   m2 9×9(9×9) {0..9}×{0..9} strided
    //   m3 ()  scalar
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j] × m3[]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":9,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":896}],"shared_bytes":3584,"shared_elements":896,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[9,9]],"name":"m0","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[9,9]],"name":"m1","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[9,9]],"name":"m2","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"scalar","alias":null,"bbox":[[],[]],"name":"m3","ordered":false,"parts":1,"shape":[],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[9,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[9,9]},{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[9,9]},{"addressing":"scalar","bbox":[[],[]],"is_tmp":false,"name":"m3","offset":[],"shape":[]}],"permute":[[0,1],[0,1],[]],"target":[[0,-1],[-1,1],[]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[112 * threadIdx.y + 0];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 81 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 81 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 81 + 0 + m2_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v22_lead = threadIdx.x % 16;
          bool v23_g = v22_lead < 9;
          if (v23_g) {
            #pragma unroll
            for (int32_t v24_i1 = 0; v24_i1 < 9; ++v24_i1) {
              float v29_data = __ldcg(&glb_m1[(v22_lead + (v24_i1 * 9))]);
              r0[v24_i1] = v29_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 5; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
          if (threadIdx.x < 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 80], &glb_m2[0 + 0 + 1 * threadIdx.x + 80], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[9]{};
          // ir1 = +(r0 * s0)
          // [(0, 9), (0, 9)] [(0, 9)]
          float ir1[9]{};
          float v35_data = r0[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v36_data = s0[0];
          float v38_data = ir1[0];
          ir1[0] = (v38_data + (v35_data * v36_data));
          float v41_data = s0[9];
          float v43_data = ir1[1];
          ir1[1] = (v43_data + (v35_data * v41_data));
          float v46_data = s0[18];
          float v48_data = ir1[2];
          ir1[2] = (v48_data + (v35_data * v46_data));
          float v51_data = s0[27];
          float v53_data = ir1[3];
          ir1[3] = (v53_data + (v35_data * v51_data));
          float v56_data = s0[36];
          float v58_data = ir1[4];
          ir1[4] = (v58_data + (v35_data * v56_data));
          float v61_data = s0[45];
          float v63_data = ir1[5];
          ir1[5] = (v63_data + (v35_data * v61_data));
          float v66_data = s0[54];
          float v68_data = ir1[6];
          ir1[6] = (v68_data + (v35_data * v66_data));
          float v71_data = s0[63];
          float v73_data = ir1[7];
          ir1[7] = (v73_data + (v35_data * v71_data));
          float v76_data = s0[72];
          float v78_data = ir1[8];
          ir1[8] = (v78_data + (v35_data * v76_data));
          float v80_data = r0[1];
          float v81_data = s0[1];
          float v83_data = ir1[0];
          ir1[0] = (v83_data + (v80_data * v81_data));
          float v86_data = s0[10];
          float v88_data = ir1[1];
          ir1[1] = (v88_data + (v80_data * v86_data));
          float v91_data = s0[19];
          float v93_data = ir1[2];
          ir1[2] = (v93_data + (v80_data * v91_data));
          float v96_data = s0[28];
          float v98_data = ir1[3];
          ir1[3] = (v98_data + (v80_data * v96_data));
          float v101_data = s0[37];
          float v103_data = ir1[4];
          ir1[4] = (v103_data + (v80_data * v101_data));
          float v106_data = s0[46];
          float v108_data = ir1[5];
          ir1[5] = (v108_data + (v80_data * v106_data));
          float v111_data = s0[55];
          float v113_data = ir1[6];
          ir1[6] = (v113_data + (v80_data * v111_data));
          float v116_data = s0[64];
          float v118_data = ir1[7];
          ir1[7] = (v118_data + (v80_data * v116_data));
          float v121_data = s0[73];
          float v123_data = ir1[8];
          ir1[8] = (v123_data + (v80_data * v121_data));
          float v125_data = r0[2];
          float v126_data = s0[2];
          float v128_data = ir1[0];
          ir1[0] = (v128_data + (v125_data * v126_data));
          float v131_data = s0[11];
          float v133_data = ir1[1];
          ir1[1] = (v133_data + (v125_data * v131_data));
          float v136_data = s0[20];
          float v138_data = ir1[2];
          ir1[2] = (v138_data + (v125_data * v136_data));
          float v141_data = s0[29];
          float v143_data = ir1[3];
          ir1[3] = (v143_data + (v125_data * v141_data));
          float v146_data = s0[38];
          float v148_data = ir1[4];
          ir1[4] = (v148_data + (v125_data * v146_data));
          float v151_data = s0[47];
          float v153_data = ir1[5];
          ir1[5] = (v153_data + (v125_data * v151_data));
          float v156_data = s0[56];
          float v158_data = ir1[6];
          ir1[6] = (v158_data + (v125_data * v156_data));
          float v161_data = s0[65];
          float v163_data = ir1[7];
          ir1[7] = (v163_data + (v125_data * v161_data));
          float v166_data = s0[74];
          float v168_data = ir1[8];
          ir1[8] = (v168_data + (v125_data * v166_data));
          float v170_data = r0[3];
          float v171_data = s0[3];
          float v173_data = ir1[0];
          ir1[0] = (v173_data + (v170_data * v171_data));
          float v176_data = s0[12];
          float v178_data = ir1[1];
          ir1[1] = (v178_data + (v170_data * v176_data));
          float v181_data = s0[21];
          float v183_data = ir1[2];
          ir1[2] = (v183_data + (v170_data * v181_data));
          float v186_data = s0[30];
          float v188_data = ir1[3];
          ir1[3] = (v188_data + (v170_data * v186_data));
          float v191_data = s0[39];
          float v193_data = ir1[4];
          ir1[4] = (v193_data + (v170_data * v191_data));
          float v196_data = s0[48];
          float v198_data = ir1[5];
          ir1[5] = (v198_data + (v170_data * v196_data));
          float v201_data = s0[57];
          float v203_data = ir1[6];
          ir1[6] = (v203_data + (v170_data * v201_data));
          float v206_data = s0[66];
          float v208_data = ir1[7];
          ir1[7] = (v208_data + (v170_data * v206_data));
          float v211_data = s0[75];
          float v213_data = ir1[8];
          ir1[8] = (v213_data + (v170_data * v211_data));
          float v215_data = r0[4];
          float v216_data = s0[4];
          float v218_data = ir1[0];
          ir1[0] = (v218_data + (v215_data * v216_data));
          float v221_data = s0[13];
          float v223_data = ir1[1];
          ir1[1] = (v223_data + (v215_data * v221_data));
          float v226_data = s0[22];
          float v228_data = ir1[2];
          ir1[2] = (v228_data + (v215_data * v226_data));
          float v231_data = s0[31];
          float v233_data = ir1[3];
          ir1[3] = (v233_data + (v215_data * v231_data));
          float v236_data = s0[40];
          float v238_data = ir1[4];
          ir1[4] = (v238_data + (v215_data * v236_data));
          float v241_data = s0[49];
          float v243_data = ir1[5];
          ir1[5] = (v243_data + (v215_data * v241_data));
          float v246_data = s0[58];
          float v248_data = ir1[6];
          ir1[6] = (v248_data + (v215_data * v246_data));
          float v251_data = s0[67];
          float v253_data = ir1[7];
          ir1[7] = (v253_data + (v215_data * v251_data));
          float v256_data = s0[76];
          float v258_data = ir1[8];
          ir1[8] = (v258_data + (v215_data * v256_data));
          float v260_data = r0[5];
          float v261_data = s0[5];
          float v263_data = ir1[0];
          ir1[0] = (v263_data + (v260_data * v261_data));
          float v266_data = s0[14];
          float v268_data = ir1[1];
          ir1[1] = (v268_data + (v260_data * v266_data));
          float v271_data = s0[23];
          float v273_data = ir1[2];
          ir1[2] = (v273_data + (v260_data * v271_data));
          float v276_data = s0[32];
          float v278_data = ir1[3];
          ir1[3] = (v278_data + (v260_data * v276_data));
          float v281_data = s0[41];
          float v283_data = ir1[4];
          ir1[4] = (v283_data + (v260_data * v281_data));
          float v286_data = s0[50];
          float v288_data = ir1[5];
          ir1[5] = (v288_data + (v260_data * v286_data));
          float v291_data = s0[59];
          float v293_data = ir1[6];
          ir1[6] = (v293_data + (v260_data * v291_data));
          float v296_data = s0[68];
          float v298_data = ir1[7];
          ir1[7] = (v298_data + (v260_data * v296_data));
          float v301_data = s0[77];
          float v303_data = ir1[8];
          ir1[8] = (v303_data + (v260_data * v301_data));
          float v305_data = r0[6];
          float v306_data = s0[6];
          float v308_data = ir1[0];
          ir1[0] = (v308_data + (v305_data * v306_data));
          float v311_data = s0[15];
          float v313_data = ir1[1];
          ir1[1] = (v313_data + (v305_data * v311_data));
          float v316_data = s0[24];
          float v318_data = ir1[2];
          ir1[2] = (v318_data + (v305_data * v316_data));
          float v321_data = s0[33];
          float v323_data = ir1[3];
          ir1[3] = (v323_data + (v305_data * v321_data));
          float v326_data = s0[42];
          float v328_data = ir1[4];
          ir1[4] = (v328_data + (v305_data * v326_data));
          float v331_data = s0[51];
          float v333_data = ir1[5];
          ir1[5] = (v333_data + (v305_data * v331_data));
          float v336_data = s0[60];
          float v338_data = ir1[6];
          ir1[6] = (v338_data + (v305_data * v336_data));
          float v341_data = s0[69];
          float v343_data = ir1[7];
          ir1[7] = (v343_data + (v305_data * v341_data));
          float v346_data = s0[78];
          float v348_data = ir1[8];
          ir1[8] = (v348_data + (v305_data * v346_data));
          float v350_data = r0[7];
          float v351_data = s0[7];
          float v353_data = ir1[0];
          ir1[0] = (v353_data + (v350_data * v351_data));
          float v356_data = s0[16];
          float v358_data = ir1[1];
          ir1[1] = (v358_data + (v350_data * v356_data));
          float v361_data = s0[25];
          float v363_data = ir1[2];
          ir1[2] = (v363_data + (v350_data * v361_data));
          float v366_data = s0[34];
          float v368_data = ir1[3];
          ir1[3] = (v368_data + (v350_data * v366_data));
          float v371_data = s0[43];
          float v373_data = ir1[4];
          ir1[4] = (v373_data + (v350_data * v371_data));
          float v376_data = s0[52];
          float v378_data = ir1[5];
          ir1[5] = (v378_data + (v350_data * v376_data));
          float v381_data = s0[61];
          float v383_data = ir1[6];
          ir1[6] = (v383_data + (v350_data * v381_data));
          float v386_data = s0[70];
          float v388_data = ir1[7];
          ir1[7] = (v388_data + (v350_data * v386_data));
          float v391_data = s0[79];
          float v393_data = ir1[8];
          ir1[8] = (v393_data + (v350_data * v391_data));
          float v395_data = r0[8];
          float v396_data = s0[8];
          float v398_data = ir1[0];
          ir1[0] = (v398_data + (v395_data * v396_data));
          float v401_data = s0[17];
          float v403_data = ir1[1];
          ir1[1] = (v403_data + (v395_data * v401_data));
          float v406_data = s0[26];
          float v408_data = ir1[2];
          ir1[2] = (v408_data + (v395_data * v406_data));
          float v411_data = s0[35];
          float v413_data = ir1[3];
          ir1[3] = (v413_data + (v395_data * v411_data));
          float v416_data = s0[44];
          float v418_data = ir1[4];
          ir1[4] = (v418_data + (v395_data * v416_data));
          float v421_data = s0[53];
          float v423_data = ir1[5];
          ir1[5] = (v423_data + (v395_data * v421_data));
          float v426_data = s0[62];
          float v428_data = ir1[6];
          ir1[6] = (v428_data + (v395_data * v426_data));
          float v431_data = s0[71];
          float v433_data = ir1[7];
          ir1[7] = (v433_data + (v395_data * v431_data));
          float v436_data = s0[80];
          float v438_data = ir1[8];
          ir1[8] = (v438_data + (v395_data * v436_data));
          // r1 = ir1 * glb_m3
          if (v23_g) {
            #pragma unroll
            for (int32_t v441_n1 = 0; v441_n1 < 9; ++v441_n1) {
              float v443_data = ir1[v441_n1];
              r1[v441_n1] = (v443_data * 13.0f);
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v23_g) {
            #pragma unroll
            for (int32_t v445_i1 = 0; v445_i1 < 9; ++v445_i1) {
              float v447_data = r1[v445_i1];
              glb_m0[(v22_lead + (v445_i1 * 9))] = v447_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

