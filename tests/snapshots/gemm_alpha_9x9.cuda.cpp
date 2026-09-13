// === base name ===
kernel_d195194a30c68155

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_d195194a30c68155 = {{16, 8, 1}, 16, 9, 1, 8, 3584, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_d195194a30c68155(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_d195194a30c68155(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_d195194a30c68155(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_d195194a30c68155, block.x * block.y * block.z, 896 * sizeof(float));
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
void launcher_kernel_d195194a30c68155(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_d195194a30c68155(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_d195194a30c68155, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_d195194a30c68155<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_d195194a30c68155(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
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
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":9,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":896}],"shared_bytes":3584,"shared_elements":896,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[9,9]],"name":"m0","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[9,9]],"name":"m1","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[9,9]],"name":"m2","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"scalar","alias":null,"bbox":[[],[]],"name":"m3","ordered":false,"parts":1,"shape":[],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[9,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[9,9]},{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[9,9]},{"addressing":"scalar","bbox":[[],[]],"is_tmp":false,"name":"m3","offset":[],"shape":[]}],"permute":[[0,1],[0,1],[]],"target":[[0,-1],[-1,1],[]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[112 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[96];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v5_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v5_batchId0 < numElements0; v5_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v6_ahead1 = v5_batchId0 + (gridDim.x * blockDim.y);
        size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 81 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 81 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 81 + 0 + m2_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v19_lead = threadIdx.x % 16;
          bool v20_g = v19_lead < 9;
          if (v20_g) {
            #pragma unroll
            for (int32_t v21_i1 = 0; v21_i1 < 9; ++v21_i1) {
              float v26_data = __ldcg(&glb_m1[(v19_lead + (v21_i1 * 9))]);
              r0[v21_i1] = v26_data;
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
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // r1 = +(r0 * s0) + None
          // [(0, 9), (0, 9)] [(0, 9)]
          float ir1[9]{};
          float v32_data = r0[0];
          float v33_data = s0[0];
          float v35_data = ir1[0];
          ir1[0] = (v35_data + (v32_data * v33_data));
          float v38_data = s0[9];
          float v40_data = ir1[1];
          ir1[1] = (v40_data + (v32_data * v38_data));
          float v43_data = s0[18];
          float v45_data = ir1[2];
          ir1[2] = (v45_data + (v32_data * v43_data));
          float v48_data = s0[27];
          float v50_data = ir1[3];
          ir1[3] = (v50_data + (v32_data * v48_data));
          float v53_data = s0[36];
          float v55_data = ir1[4];
          ir1[4] = (v55_data + (v32_data * v53_data));
          float v58_data = s0[45];
          float v60_data = ir1[5];
          ir1[5] = (v60_data + (v32_data * v58_data));
          float v63_data = s0[54];
          float v65_data = ir1[6];
          ir1[6] = (v65_data + (v32_data * v63_data));
          float v68_data = s0[63];
          float v70_data = ir1[7];
          ir1[7] = (v70_data + (v32_data * v68_data));
          float v73_data = s0[72];
          float v75_data = ir1[8];
          ir1[8] = (v75_data + (v32_data * v73_data));
          float v77_data = r0[1];
          float v78_data = s0[1];
          float v80_data = ir1[0];
          ir1[0] = (v80_data + (v77_data * v78_data));
          float v83_data = s0[10];
          float v85_data = ir1[1];
          ir1[1] = (v85_data + (v77_data * v83_data));
          float v88_data = s0[19];
          float v90_data = ir1[2];
          ir1[2] = (v90_data + (v77_data * v88_data));
          float v93_data = s0[28];
          float v95_data = ir1[3];
          ir1[3] = (v95_data + (v77_data * v93_data));
          float v98_data = s0[37];
          float v100_data = ir1[4];
          ir1[4] = (v100_data + (v77_data * v98_data));
          float v103_data = s0[46];
          float v105_data = ir1[5];
          ir1[5] = (v105_data + (v77_data * v103_data));
          float v108_data = s0[55];
          float v110_data = ir1[6];
          ir1[6] = (v110_data + (v77_data * v108_data));
          float v113_data = s0[64];
          float v115_data = ir1[7];
          ir1[7] = (v115_data + (v77_data * v113_data));
          float v118_data = s0[73];
          float v120_data = ir1[8];
          ir1[8] = (v120_data + (v77_data * v118_data));
          float v122_data = r0[2];
          float v123_data = s0[2];
          float v125_data = ir1[0];
          ir1[0] = (v125_data + (v122_data * v123_data));
          float v128_data = s0[11];
          float v130_data = ir1[1];
          ir1[1] = (v130_data + (v122_data * v128_data));
          float v133_data = s0[20];
          float v135_data = ir1[2];
          ir1[2] = (v135_data + (v122_data * v133_data));
          float v138_data = s0[29];
          float v140_data = ir1[3];
          ir1[3] = (v140_data + (v122_data * v138_data));
          float v143_data = s0[38];
          float v145_data = ir1[4];
          ir1[4] = (v145_data + (v122_data * v143_data));
          float v148_data = s0[47];
          float v150_data = ir1[5];
          ir1[5] = (v150_data + (v122_data * v148_data));
          float v153_data = s0[56];
          float v155_data = ir1[6];
          ir1[6] = (v155_data + (v122_data * v153_data));
          float v158_data = s0[65];
          float v160_data = ir1[7];
          ir1[7] = (v160_data + (v122_data * v158_data));
          float v163_data = s0[74];
          float v165_data = ir1[8];
          ir1[8] = (v165_data + (v122_data * v163_data));
          float v167_data = r0[3];
          float v168_data = s0[3];
          float v170_data = ir1[0];
          ir1[0] = (v170_data + (v167_data * v168_data));
          float v173_data = s0[12];
          float v175_data = ir1[1];
          ir1[1] = (v175_data + (v167_data * v173_data));
          float v178_data = s0[21];
          float v180_data = ir1[2];
          ir1[2] = (v180_data + (v167_data * v178_data));
          float v183_data = s0[30];
          float v185_data = ir1[3];
          ir1[3] = (v185_data + (v167_data * v183_data));
          float v188_data = s0[39];
          float v190_data = ir1[4];
          ir1[4] = (v190_data + (v167_data * v188_data));
          float v193_data = s0[48];
          float v195_data = ir1[5];
          ir1[5] = (v195_data + (v167_data * v193_data));
          float v198_data = s0[57];
          float v200_data = ir1[6];
          ir1[6] = (v200_data + (v167_data * v198_data));
          float v203_data = s0[66];
          float v205_data = ir1[7];
          ir1[7] = (v205_data + (v167_data * v203_data));
          float v208_data = s0[75];
          float v210_data = ir1[8];
          ir1[8] = (v210_data + (v167_data * v208_data));
          float v212_data = r0[4];
          float v213_data = s0[4];
          float v215_data = ir1[0];
          ir1[0] = (v215_data + (v212_data * v213_data));
          float v218_data = s0[13];
          float v220_data = ir1[1];
          ir1[1] = (v220_data + (v212_data * v218_data));
          float v223_data = s0[22];
          float v225_data = ir1[2];
          ir1[2] = (v225_data + (v212_data * v223_data));
          float v228_data = s0[31];
          float v230_data = ir1[3];
          ir1[3] = (v230_data + (v212_data * v228_data));
          float v233_data = s0[40];
          float v235_data = ir1[4];
          ir1[4] = (v235_data + (v212_data * v233_data));
          float v238_data = s0[49];
          float v240_data = ir1[5];
          ir1[5] = (v240_data + (v212_data * v238_data));
          float v243_data = s0[58];
          float v245_data = ir1[6];
          ir1[6] = (v245_data + (v212_data * v243_data));
          float v248_data = s0[67];
          float v250_data = ir1[7];
          ir1[7] = (v250_data + (v212_data * v248_data));
          float v253_data = s0[76];
          float v255_data = ir1[8];
          ir1[8] = (v255_data + (v212_data * v253_data));
          float v257_data = r0[5];
          float v258_data = s0[5];
          float v260_data = ir1[0];
          ir1[0] = (v260_data + (v257_data * v258_data));
          float v263_data = s0[14];
          float v265_data = ir1[1];
          ir1[1] = (v265_data + (v257_data * v263_data));
          float v268_data = s0[23];
          float v270_data = ir1[2];
          ir1[2] = (v270_data + (v257_data * v268_data));
          float v273_data = s0[32];
          float v275_data = ir1[3];
          ir1[3] = (v275_data + (v257_data * v273_data));
          float v278_data = s0[41];
          float v280_data = ir1[4];
          ir1[4] = (v280_data + (v257_data * v278_data));
          float v283_data = s0[50];
          float v285_data = ir1[5];
          ir1[5] = (v285_data + (v257_data * v283_data));
          float v288_data = s0[59];
          float v290_data = ir1[6];
          ir1[6] = (v290_data + (v257_data * v288_data));
          float v293_data = s0[68];
          float v295_data = ir1[7];
          ir1[7] = (v295_data + (v257_data * v293_data));
          float v298_data = s0[77];
          float v300_data = ir1[8];
          ir1[8] = (v300_data + (v257_data * v298_data));
          float v302_data = r0[6];
          float v303_data = s0[6];
          float v305_data = ir1[0];
          ir1[0] = (v305_data + (v302_data * v303_data));
          float v308_data = s0[15];
          float v310_data = ir1[1];
          ir1[1] = (v310_data + (v302_data * v308_data));
          float v313_data = s0[24];
          float v315_data = ir1[2];
          ir1[2] = (v315_data + (v302_data * v313_data));
          float v318_data = s0[33];
          float v320_data = ir1[3];
          ir1[3] = (v320_data + (v302_data * v318_data));
          float v323_data = s0[42];
          float v325_data = ir1[4];
          ir1[4] = (v325_data + (v302_data * v323_data));
          float v328_data = s0[51];
          float v330_data = ir1[5];
          ir1[5] = (v330_data + (v302_data * v328_data));
          float v333_data = s0[60];
          float v335_data = ir1[6];
          ir1[6] = (v335_data + (v302_data * v333_data));
          float v338_data = s0[69];
          float v340_data = ir1[7];
          ir1[7] = (v340_data + (v302_data * v338_data));
          float v343_data = s0[78];
          float v345_data = ir1[8];
          ir1[8] = (v345_data + (v302_data * v343_data));
          float v347_data = r0[7];
          float v348_data = s0[7];
          float v350_data = ir1[0];
          ir1[0] = (v350_data + (v347_data * v348_data));
          float v353_data = s0[16];
          float v355_data = ir1[1];
          ir1[1] = (v355_data + (v347_data * v353_data));
          float v358_data = s0[25];
          float v360_data = ir1[2];
          ir1[2] = (v360_data + (v347_data * v358_data));
          float v363_data = s0[34];
          float v365_data = ir1[3];
          ir1[3] = (v365_data + (v347_data * v363_data));
          float v368_data = s0[43];
          float v370_data = ir1[4];
          ir1[4] = (v370_data + (v347_data * v368_data));
          float v373_data = s0[52];
          float v375_data = ir1[5];
          ir1[5] = (v375_data + (v347_data * v373_data));
          float v378_data = s0[61];
          float v380_data = ir1[6];
          ir1[6] = (v380_data + (v347_data * v378_data));
          float v383_data = s0[70];
          float v385_data = ir1[7];
          ir1[7] = (v385_data + (v347_data * v383_data));
          float v388_data = s0[79];
          float v390_data = ir1[8];
          ir1[8] = (v390_data + (v347_data * v388_data));
          float v392_data = r0[8];
          float v393_data = s0[8];
          float v395_data = ir1[0];
          ir1[0] = (v395_data + (v392_data * v393_data));
          float v398_data = s0[17];
          float v400_data = ir1[1];
          ir1[1] = (v400_data + (v392_data * v398_data));
          float v403_data = s0[26];
          float v405_data = ir1[2];
          ir1[2] = (v405_data + (v392_data * v403_data));
          float v408_data = s0[35];
          float v410_data = ir1[3];
          ir1[3] = (v410_data + (v392_data * v408_data));
          float v413_data = s0[44];
          float v415_data = ir1[4];
          ir1[4] = (v415_data + (v392_data * v413_data));
          float v418_data = s0[53];
          float v420_data = ir1[5];
          ir1[5] = (v420_data + (v392_data * v418_data));
          float v423_data = s0[62];
          float v425_data = ir1[6];
          ir1[6] = (v425_data + (v392_data * v423_data));
          float v428_data = s0[71];
          float v430_data = ir1[7];
          ir1[7] = (v430_data + (v392_data * v428_data));
          float v433_data = s0[80];
          float v435_data = ir1[8];
          ir1[8] = (v435_data + (v392_data * v433_data));
          if (v20_g) {
            #pragma unroll
            for (int32_t v438_n1 = 0; v438_n1 < 9; ++v438_n1) {
              float v440_data = ir1[v438_n1];
              r1[v438_n1] = (v440_data * 13.0f);
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v20_g) {
            #pragma unroll
            for (int32_t v442_i1 = 0; v442_i1 < 9; ++v442_i1) {
              float v444_data = r1[v442_i1];
              glb_m0[(v19_lead + (v442_i1 * 9))] = v444_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

