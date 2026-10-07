// === base name ===
kernel_16d4d7d443ab73c9

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_16d4d7d443ab73c9 = {{16, 8, 1}, 16, 12, 1, 8, 5632, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_16d4d7d443ab73c9(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_16d4d7d443ab73c9(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_16d4d7d443ab73c9(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_16d4d7d443ab73c9, block.x * block.y * block.z, 1408 * sizeof(float));
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
  config.sharedMemBytes = 1408 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_16d4d7d443ab73c9(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_16d4d7d443ab73c9(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_16d4d7d443ab73c9, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_16d4d7d443ab73c9<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_16d4d7d443ab73c9(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 8 per block = block 16x8x1, 5632 B shared, occupancy grid
    // operands:
    //   m0 6×12(6×12) {0..6}×{0..12} strided
    //   m1 12×12(12×12) {0..12}×{0..12} strided
    //   m2 6×12(6×12) {0..6}×{0..12} strided
    //   m3 6×12(6×12) {0..6}×{0..12} strided
    //   m4 12×12(12×12) {0..12}×{0..12} strided
    // operations:
    //   t0[i,j]@{6..12}×{0..12} = m0[i,k] × m1[k,j]
    //   t0[i,j] = m2[i,k] × m1[k,j]
    //   t0[i,j]@{6..12}×{0..12} = m3[i,j]
    //   m4[i,j] = t0[i,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1408}],"shared_bytes":5632,"shared_elements":1408,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B2","bbox":[[0,0],[6,12]],"name":"m0","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"N","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"Y","bbox":[[0,0],[6,12]],"name":"m3","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m4","ordered":false,"parts":1,"shape":[12,12],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[6,12]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[176 * threadIdx.y + 0];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < numElements0; v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 72 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v9_batchId0 * 144 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v9_batchId0 * 72 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[v9_batchId0 * 72 + 0 + m3_extraOffset];
          float *const __restrict__ glb_m4 = &m4[v9_batchId0 * 144 + 0 + m4_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v25_lead = threadIdx.x % 16;
          bool v26_g = v25_lead < 6;
          if (v26_g) {
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 12; ++v27_i1) {
              float v32_data = __ldcg(&glb_m0[(v25_lead + (v27_i1 * 6))]);
              r0[v27_i1] = v32_data;
            }
          }
          // s0 = load{g>s}(glb_m1[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 9; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m1[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m0););
          float r2[12]{};
          // r2 = load{g>r}(glb_m2);
          if (v26_g) {
            #pragma unroll
            for (int32_t v36_i1 = 0; v36_i1 < 12; ++v36_i1) {
              float v41_data = __ldcg(&glb_m2[(v25_lead + (v36_i1 * 6))]);
              r2[v36_i1] = v41_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[12]{};
          // r1 = +(r0 * s0) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v44_data = r0[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v45_data = s0[0];
          float v47_data = r1[0];
          r1[0] = (v47_data + (v44_data * v45_data));
          float v50_data = s0[12];
          float v52_data = r1[1];
          r1[1] = (v52_data + (v44_data * v50_data));
          float v55_data = s0[24];
          float v57_data = r1[2];
          r1[2] = (v57_data + (v44_data * v55_data));
          float v60_data = s0[36];
          float v62_data = r1[3];
          r1[3] = (v62_data + (v44_data * v60_data));
          float v65_data = s0[48];
          float v67_data = r1[4];
          r1[4] = (v67_data + (v44_data * v65_data));
          float v70_data = s0[60];
          float v72_data = r1[5];
          r1[5] = (v72_data + (v44_data * v70_data));
          float v75_data = s0[72];
          float v77_data = r1[6];
          r1[6] = (v77_data + (v44_data * v75_data));
          float v80_data = s0[84];
          float v82_data = r1[7];
          r1[7] = (v82_data + (v44_data * v80_data));
          float v85_data = s0[96];
          float v87_data = r1[8];
          r1[8] = (v87_data + (v44_data * v85_data));
          float v90_data = s0[108];
          float v92_data = r1[9];
          r1[9] = (v92_data + (v44_data * v90_data));
          float v95_data = s0[120];
          float v97_data = r1[10];
          r1[10] = (v97_data + (v44_data * v95_data));
          float v100_data = s0[132];
          float v102_data = r1[11];
          r1[11] = (v102_data + (v44_data * v100_data));
          float v104_data = r0[1];
          float v105_data = s0[1];
          float v107_data = r1[0];
          r1[0] = (v107_data + (v104_data * v105_data));
          float v110_data = s0[13];
          float v112_data = r1[1];
          r1[1] = (v112_data + (v104_data * v110_data));
          float v115_data = s0[25];
          float v117_data = r1[2];
          r1[2] = (v117_data + (v104_data * v115_data));
          float v120_data = s0[37];
          float v122_data = r1[3];
          r1[3] = (v122_data + (v104_data * v120_data));
          float v125_data = s0[49];
          float v127_data = r1[4];
          r1[4] = (v127_data + (v104_data * v125_data));
          float v130_data = s0[61];
          float v132_data = r1[5];
          r1[5] = (v132_data + (v104_data * v130_data));
          float v135_data = s0[73];
          float v137_data = r1[6];
          r1[6] = (v137_data + (v104_data * v135_data));
          float v140_data = s0[85];
          float v142_data = r1[7];
          r1[7] = (v142_data + (v104_data * v140_data));
          float v145_data = s0[97];
          float v147_data = r1[8];
          r1[8] = (v147_data + (v104_data * v145_data));
          float v150_data = s0[109];
          float v152_data = r1[9];
          r1[9] = (v152_data + (v104_data * v150_data));
          float v155_data = s0[121];
          float v157_data = r1[10];
          r1[10] = (v157_data + (v104_data * v155_data));
          float v160_data = s0[133];
          float v162_data = r1[11];
          r1[11] = (v162_data + (v104_data * v160_data));
          float v164_data = r0[2];
          float v165_data = s0[2];
          float v167_data = r1[0];
          r1[0] = (v167_data + (v164_data * v165_data));
          float v170_data = s0[14];
          float v172_data = r1[1];
          r1[1] = (v172_data + (v164_data * v170_data));
          float v175_data = s0[26];
          float v177_data = r1[2];
          r1[2] = (v177_data + (v164_data * v175_data));
          float v180_data = s0[38];
          float v182_data = r1[3];
          r1[3] = (v182_data + (v164_data * v180_data));
          float v185_data = s0[50];
          float v187_data = r1[4];
          r1[4] = (v187_data + (v164_data * v185_data));
          float v190_data = s0[62];
          float v192_data = r1[5];
          r1[5] = (v192_data + (v164_data * v190_data));
          float v195_data = s0[74];
          float v197_data = r1[6];
          r1[6] = (v197_data + (v164_data * v195_data));
          float v200_data = s0[86];
          float v202_data = r1[7];
          r1[7] = (v202_data + (v164_data * v200_data));
          float v205_data = s0[98];
          float v207_data = r1[8];
          r1[8] = (v207_data + (v164_data * v205_data));
          float v210_data = s0[110];
          float v212_data = r1[9];
          r1[9] = (v212_data + (v164_data * v210_data));
          float v215_data = s0[122];
          float v217_data = r1[10];
          r1[10] = (v217_data + (v164_data * v215_data));
          float v220_data = s0[134];
          float v222_data = r1[11];
          r1[11] = (v222_data + (v164_data * v220_data));
          float v224_data = r0[3];
          float v225_data = s0[3];
          float v227_data = r1[0];
          r1[0] = (v227_data + (v224_data * v225_data));
          float v230_data = s0[15];
          float v232_data = r1[1];
          r1[1] = (v232_data + (v224_data * v230_data));
          float v235_data = s0[27];
          float v237_data = r1[2];
          r1[2] = (v237_data + (v224_data * v235_data));
          float v240_data = s0[39];
          float v242_data = r1[3];
          r1[3] = (v242_data + (v224_data * v240_data));
          float v245_data = s0[51];
          float v247_data = r1[4];
          r1[4] = (v247_data + (v224_data * v245_data));
          float v250_data = s0[63];
          float v252_data = r1[5];
          r1[5] = (v252_data + (v224_data * v250_data));
          float v255_data = s0[75];
          float v257_data = r1[6];
          r1[6] = (v257_data + (v224_data * v255_data));
          float v260_data = s0[87];
          float v262_data = r1[7];
          r1[7] = (v262_data + (v224_data * v260_data));
          float v265_data = s0[99];
          float v267_data = r1[8];
          r1[8] = (v267_data + (v224_data * v265_data));
          float v270_data = s0[111];
          float v272_data = r1[9];
          r1[9] = (v272_data + (v224_data * v270_data));
          float v275_data = s0[123];
          float v277_data = r1[10];
          r1[10] = (v277_data + (v224_data * v275_data));
          float v280_data = s0[135];
          float v282_data = r1[11];
          r1[11] = (v282_data + (v224_data * v280_data));
          float v284_data = r0[4];
          float v285_data = s0[4];
          float v287_data = r1[0];
          r1[0] = (v287_data + (v284_data * v285_data));
          float v290_data = s0[16];
          float v292_data = r1[1];
          r1[1] = (v292_data + (v284_data * v290_data));
          float v295_data = s0[28];
          float v297_data = r1[2];
          r1[2] = (v297_data + (v284_data * v295_data));
          float v300_data = s0[40];
          float v302_data = r1[3];
          r1[3] = (v302_data + (v284_data * v300_data));
          float v305_data = s0[52];
          float v307_data = r1[4];
          r1[4] = (v307_data + (v284_data * v305_data));
          float v310_data = s0[64];
          float v312_data = r1[5];
          r1[5] = (v312_data + (v284_data * v310_data));
          float v315_data = s0[76];
          float v317_data = r1[6];
          r1[6] = (v317_data + (v284_data * v315_data));
          float v320_data = s0[88];
          float v322_data = r1[7];
          r1[7] = (v322_data + (v284_data * v320_data));
          float v325_data = s0[100];
          float v327_data = r1[8];
          r1[8] = (v327_data + (v284_data * v325_data));
          float v330_data = s0[112];
          float v332_data = r1[9];
          r1[9] = (v332_data + (v284_data * v330_data));
          float v335_data = s0[124];
          float v337_data = r1[10];
          r1[10] = (v337_data + (v284_data * v335_data));
          float v340_data = s0[136];
          float v342_data = r1[11];
          r1[11] = (v342_data + (v284_data * v340_data));
          float v344_data = r0[5];
          float v345_data = s0[5];
          float v347_data = r1[0];
          r1[0] = (v347_data + (v344_data * v345_data));
          float v350_data = s0[17];
          float v352_data = r1[1];
          r1[1] = (v352_data + (v344_data * v350_data));
          float v355_data = s0[29];
          float v357_data = r1[2];
          r1[2] = (v357_data + (v344_data * v355_data));
          float v360_data = s0[41];
          float v362_data = r1[3];
          r1[3] = (v362_data + (v344_data * v360_data));
          float v365_data = s0[53];
          float v367_data = r1[4];
          r1[4] = (v367_data + (v344_data * v365_data));
          float v370_data = s0[65];
          float v372_data = r1[5];
          r1[5] = (v372_data + (v344_data * v370_data));
          float v375_data = s0[77];
          float v377_data = r1[6];
          r1[6] = (v377_data + (v344_data * v375_data));
          float v380_data = s0[89];
          float v382_data = r1[7];
          r1[7] = (v382_data + (v344_data * v380_data));
          float v385_data = s0[101];
          float v387_data = r1[8];
          r1[8] = (v387_data + (v344_data * v385_data));
          float v390_data = s0[113];
          float v392_data = r1[9];
          r1[9] = (v392_data + (v344_data * v390_data));
          float v395_data = s0[125];
          float v397_data = r1[10];
          r1[10] = (v397_data + (v344_data * v395_data));
          float v400_data = s0[137];
          float v402_data = r1[11];
          r1[11] = (v402_data + (v344_data * v400_data));
          float v404_data = r0[6];
          float v405_data = s0[6];
          float v407_data = r1[0];
          r1[0] = (v407_data + (v404_data * v405_data));
          float v410_data = s0[18];
          float v412_data = r1[1];
          r1[1] = (v412_data + (v404_data * v410_data));
          float v415_data = s0[30];
          float v417_data = r1[2];
          r1[2] = (v417_data + (v404_data * v415_data));
          float v420_data = s0[42];
          float v422_data = r1[3];
          r1[3] = (v422_data + (v404_data * v420_data));
          float v425_data = s0[54];
          float v427_data = r1[4];
          r1[4] = (v427_data + (v404_data * v425_data));
          float v430_data = s0[66];
          float v432_data = r1[5];
          r1[5] = (v432_data + (v404_data * v430_data));
          float v435_data = s0[78];
          float v437_data = r1[6];
          r1[6] = (v437_data + (v404_data * v435_data));
          float v440_data = s0[90];
          float v442_data = r1[7];
          r1[7] = (v442_data + (v404_data * v440_data));
          float v445_data = s0[102];
          float v447_data = r1[8];
          r1[8] = (v447_data + (v404_data * v445_data));
          float v450_data = s0[114];
          float v452_data = r1[9];
          r1[9] = (v452_data + (v404_data * v450_data));
          float v455_data = s0[126];
          float v457_data = r1[10];
          r1[10] = (v457_data + (v404_data * v455_data));
          float v460_data = s0[138];
          float v462_data = r1[11];
          r1[11] = (v462_data + (v404_data * v460_data));
          float v464_data = r0[7];
          float v465_data = s0[7];
          float v467_data = r1[0];
          r1[0] = (v467_data + (v464_data * v465_data));
          float v470_data = s0[19];
          float v472_data = r1[1];
          r1[1] = (v472_data + (v464_data * v470_data));
          float v475_data = s0[31];
          float v477_data = r1[2];
          r1[2] = (v477_data + (v464_data * v475_data));
          float v480_data = s0[43];
          float v482_data = r1[3];
          r1[3] = (v482_data + (v464_data * v480_data));
          float v485_data = s0[55];
          float v487_data = r1[4];
          r1[4] = (v487_data + (v464_data * v485_data));
          float v490_data = s0[67];
          float v492_data = r1[5];
          r1[5] = (v492_data + (v464_data * v490_data));
          float v495_data = s0[79];
          float v497_data = r1[6];
          r1[6] = (v497_data + (v464_data * v495_data));
          float v500_data = s0[91];
          float v502_data = r1[7];
          r1[7] = (v502_data + (v464_data * v500_data));
          float v505_data = s0[103];
          float v507_data = r1[8];
          r1[8] = (v507_data + (v464_data * v505_data));
          float v510_data = s0[115];
          float v512_data = r1[9];
          r1[9] = (v512_data + (v464_data * v510_data));
          float v515_data = s0[127];
          float v517_data = r1[10];
          r1[10] = (v517_data + (v464_data * v515_data));
          float v520_data = s0[139];
          float v522_data = r1[11];
          r1[11] = (v522_data + (v464_data * v520_data));
          float v524_data = r0[8];
          float v525_data = s0[8];
          float v527_data = r1[0];
          r1[0] = (v527_data + (v524_data * v525_data));
          float v530_data = s0[20];
          float v532_data = r1[1];
          r1[1] = (v532_data + (v524_data * v530_data));
          float v535_data = s0[32];
          float v537_data = r1[2];
          r1[2] = (v537_data + (v524_data * v535_data));
          float v540_data = s0[44];
          float v542_data = r1[3];
          r1[3] = (v542_data + (v524_data * v540_data));
          float v545_data = s0[56];
          float v547_data = r1[4];
          r1[4] = (v547_data + (v524_data * v545_data));
          float v550_data = s0[68];
          float v552_data = r1[5];
          r1[5] = (v552_data + (v524_data * v550_data));
          float v555_data = s0[80];
          float v557_data = r1[6];
          r1[6] = (v557_data + (v524_data * v555_data));
          float v560_data = s0[92];
          float v562_data = r1[7];
          r1[7] = (v562_data + (v524_data * v560_data));
          float v565_data = s0[104];
          float v567_data = r1[8];
          r1[8] = (v567_data + (v524_data * v565_data));
          float v570_data = s0[116];
          float v572_data = r1[9];
          r1[9] = (v572_data + (v524_data * v570_data));
          float v575_data = s0[128];
          float v577_data = r1[10];
          r1[10] = (v577_data + (v524_data * v575_data));
          float v580_data = s0[140];
          float v582_data = r1[11];
          r1[11] = (v582_data + (v524_data * v580_data));
          float v584_data = r0[9];
          float v585_data = s0[9];
          float v587_data = r1[0];
          r1[0] = (v587_data + (v584_data * v585_data));
          float v590_data = s0[21];
          float v592_data = r1[1];
          r1[1] = (v592_data + (v584_data * v590_data));
          float v595_data = s0[33];
          float v597_data = r1[2];
          r1[2] = (v597_data + (v584_data * v595_data));
          float v600_data = s0[45];
          float v602_data = r1[3];
          r1[3] = (v602_data + (v584_data * v600_data));
          float v605_data = s0[57];
          float v607_data = r1[4];
          r1[4] = (v607_data + (v584_data * v605_data));
          float v610_data = s0[69];
          float v612_data = r1[5];
          r1[5] = (v612_data + (v584_data * v610_data));
          float v615_data = s0[81];
          float v617_data = r1[6];
          r1[6] = (v617_data + (v584_data * v615_data));
          float v620_data = s0[93];
          float v622_data = r1[7];
          r1[7] = (v622_data + (v584_data * v620_data));
          float v625_data = s0[105];
          float v627_data = r1[8];
          r1[8] = (v627_data + (v584_data * v625_data));
          float v630_data = s0[117];
          float v632_data = r1[9];
          r1[9] = (v632_data + (v584_data * v630_data));
          float v635_data = s0[129];
          float v637_data = r1[10];
          r1[10] = (v637_data + (v584_data * v635_data));
          float v640_data = s0[141];
          float v642_data = r1[11];
          r1[11] = (v642_data + (v584_data * v640_data));
          float v644_data = r0[10];
          float v645_data = s0[10];
          float v647_data = r1[0];
          r1[0] = (v647_data + (v644_data * v645_data));
          float v650_data = s0[22];
          float v652_data = r1[1];
          r1[1] = (v652_data + (v644_data * v650_data));
          float v655_data = s0[34];
          float v657_data = r1[2];
          r1[2] = (v657_data + (v644_data * v655_data));
          float v660_data = s0[46];
          float v662_data = r1[3];
          r1[3] = (v662_data + (v644_data * v660_data));
          float v665_data = s0[58];
          float v667_data = r1[4];
          r1[4] = (v667_data + (v644_data * v665_data));
          float v670_data = s0[70];
          float v672_data = r1[5];
          r1[5] = (v672_data + (v644_data * v670_data));
          float v675_data = s0[82];
          float v677_data = r1[6];
          r1[6] = (v677_data + (v644_data * v675_data));
          float v680_data = s0[94];
          float v682_data = r1[7];
          r1[7] = (v682_data + (v644_data * v680_data));
          float v685_data = s0[106];
          float v687_data = r1[8];
          r1[8] = (v687_data + (v644_data * v685_data));
          float v690_data = s0[118];
          float v692_data = r1[9];
          r1[9] = (v692_data + (v644_data * v690_data));
          float v695_data = s0[130];
          float v697_data = r1[10];
          r1[10] = (v697_data + (v644_data * v695_data));
          float v700_data = s0[142];
          float v702_data = r1[11];
          r1[11] = (v702_data + (v644_data * v700_data));
          float v704_data = r0[11];
          float v705_data = s0[11];
          float v707_data = r1[0];
          r1[0] = (v707_data + (v704_data * v705_data));
          float v710_data = s0[23];
          float v712_data = r1[1];
          r1[1] = (v712_data + (v704_data * v710_data));
          float v715_data = s0[35];
          float v717_data = r1[2];
          r1[2] = (v717_data + (v704_data * v715_data));
          float v720_data = s0[47];
          float v722_data = r1[3];
          r1[3] = (v722_data + (v704_data * v720_data));
          float v725_data = s0[59];
          float v727_data = r1[4];
          r1[4] = (v727_data + (v704_data * v725_data));
          float v730_data = s0[71];
          float v732_data = r1[5];
          r1[5] = (v732_data + (v704_data * v730_data));
          float v735_data = s0[83];
          float v737_data = r1[6];
          r1[6] = (v737_data + (v704_data * v735_data));
          float v740_data = s0[95];
          float v742_data = r1[7];
          r1[7] = (v742_data + (v704_data * v740_data));
          float v745_data = s0[107];
          float v747_data = r1[8];
          r1[8] = (v747_data + (v704_data * v745_data));
          float v750_data = s0[119];
          float v752_data = r1[9];
          r1[9] = (v752_data + (v704_data * v750_data));
          float v755_data = s0[131];
          float v757_data = r1[10];
          r1[10] = (v757_data + (v704_data * v755_data));
          float v760_data = s0[143];
          float v762_data = r1[11];
          r1[11] = (v762_data + (v704_data * v760_data));
          // s1 = store{r>s}(localShrMem0, r1);
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          if (v26_g) {
            int32_t v769_off = v25_lead + 6;
            #pragma unroll
            for (int32_t v764_i1 = 0; v764_i1 < 12; ++v764_i1) {
              float v766_data = r1[v764_i1];
              int32_t v771_a = v769_off + (v764_i1 * 12);
              s1[(v771_a ^ ((v771_a >> 4) & 15))] = v766_data;
            }
          }
          float r4[12]{};
          // r4 = load{g>r}(glb_m3);
          if (v26_g) {
            #pragma unroll
            for (int32_t v776_i1 = 0; v776_i1 < 12; ++v776_i1) {
              float v781_data = __ldcg(&glb_m3[(v25_lead + (v776_i1 * 6))]);
              r4[v776_i1] = v781_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m2););
          float r3[12]{};
          // ir3 = +(r2 * s0)
          // [(0, 6), (0, 12)] [(0, 12)]
          float ir3[12]{};
          float v785_data = r2[0];
          float v788_data = ir3[0];
          ir3[0] = (v788_data + (v785_data * v45_data));
          float v793_data = ir3[1];
          ir3[1] = (v793_data + (v785_data * v50_data));
          float v798_data = ir3[2];
          ir3[2] = (v798_data + (v785_data * v55_data));
          float v803_data = ir3[3];
          ir3[3] = (v803_data + (v785_data * v60_data));
          float v808_data = ir3[4];
          ir3[4] = (v808_data + (v785_data * v65_data));
          float v813_data = ir3[5];
          ir3[5] = (v813_data + (v785_data * v70_data));
          float v818_data = ir3[6];
          ir3[6] = (v818_data + (v785_data * v75_data));
          float v823_data = ir3[7];
          ir3[7] = (v823_data + (v785_data * v80_data));
          float v828_data = ir3[8];
          ir3[8] = (v828_data + (v785_data * v85_data));
          float v833_data = ir3[9];
          ir3[9] = (v833_data + (v785_data * v90_data));
          float v838_data = ir3[10];
          ir3[10] = (v838_data + (v785_data * v95_data));
          float v843_data = ir3[11];
          ir3[11] = (v843_data + (v785_data * v100_data));
          float v845_data = r2[1];
          float v848_data = ir3[0];
          ir3[0] = (v848_data + (v845_data * v105_data));
          float v853_data = ir3[1];
          ir3[1] = (v853_data + (v845_data * v110_data));
          float v858_data = ir3[2];
          ir3[2] = (v858_data + (v845_data * v115_data));
          float v863_data = ir3[3];
          ir3[3] = (v863_data + (v845_data * v120_data));
          float v868_data = ir3[4];
          ir3[4] = (v868_data + (v845_data * v125_data));
          float v873_data = ir3[5];
          ir3[5] = (v873_data + (v845_data * v130_data));
          float v878_data = ir3[6];
          ir3[6] = (v878_data + (v845_data * v135_data));
          float v883_data = ir3[7];
          ir3[7] = (v883_data + (v845_data * v140_data));
          float v888_data = ir3[8];
          ir3[8] = (v888_data + (v845_data * v145_data));
          float v893_data = ir3[9];
          ir3[9] = (v893_data + (v845_data * v150_data));
          float v898_data = ir3[10];
          ir3[10] = (v898_data + (v845_data * v155_data));
          float v903_data = ir3[11];
          ir3[11] = (v903_data + (v845_data * v160_data));
          float v905_data = r2[2];
          float v908_data = ir3[0];
          ir3[0] = (v908_data + (v905_data * v165_data));
          float v913_data = ir3[1];
          ir3[1] = (v913_data + (v905_data * v170_data));
          float v918_data = ir3[2];
          ir3[2] = (v918_data + (v905_data * v175_data));
          float v923_data = ir3[3];
          ir3[3] = (v923_data + (v905_data * v180_data));
          float v928_data = ir3[4];
          ir3[4] = (v928_data + (v905_data * v185_data));
          float v933_data = ir3[5];
          ir3[5] = (v933_data + (v905_data * v190_data));
          float v938_data = ir3[6];
          ir3[6] = (v938_data + (v905_data * v195_data));
          float v943_data = ir3[7];
          ir3[7] = (v943_data + (v905_data * v200_data));
          float v948_data = ir3[8];
          ir3[8] = (v948_data + (v905_data * v205_data));
          float v953_data = ir3[9];
          ir3[9] = (v953_data + (v905_data * v210_data));
          float v958_data = ir3[10];
          ir3[10] = (v958_data + (v905_data * v215_data));
          float v963_data = ir3[11];
          ir3[11] = (v963_data + (v905_data * v220_data));
          float v965_data = r2[3];
          float v968_data = ir3[0];
          ir3[0] = (v968_data + (v965_data * v225_data));
          float v973_data = ir3[1];
          ir3[1] = (v973_data + (v965_data * v230_data));
          float v978_data = ir3[2];
          ir3[2] = (v978_data + (v965_data * v235_data));
          float v983_data = ir3[3];
          ir3[3] = (v983_data + (v965_data * v240_data));
          float v988_data = ir3[4];
          ir3[4] = (v988_data + (v965_data * v245_data));
          float v993_data = ir3[5];
          ir3[5] = (v993_data + (v965_data * v250_data));
          float v998_data = ir3[6];
          ir3[6] = (v998_data + (v965_data * v255_data));
          float v1003_data = ir3[7];
          ir3[7] = (v1003_data + (v965_data * v260_data));
          float v1008_data = ir3[8];
          ir3[8] = (v1008_data + (v965_data * v265_data));
          float v1013_data = ir3[9];
          ir3[9] = (v1013_data + (v965_data * v270_data));
          float v1018_data = ir3[10];
          ir3[10] = (v1018_data + (v965_data * v275_data));
          float v1023_data = ir3[11];
          ir3[11] = (v1023_data + (v965_data * v280_data));
          float v1025_data = r2[4];
          float v1028_data = ir3[0];
          ir3[0] = (v1028_data + (v1025_data * v285_data));
          float v1033_data = ir3[1];
          ir3[1] = (v1033_data + (v1025_data * v290_data));
          float v1038_data = ir3[2];
          ir3[2] = (v1038_data + (v1025_data * v295_data));
          float v1043_data = ir3[3];
          ir3[3] = (v1043_data + (v1025_data * v300_data));
          float v1048_data = ir3[4];
          ir3[4] = (v1048_data + (v1025_data * v305_data));
          float v1053_data = ir3[5];
          ir3[5] = (v1053_data + (v1025_data * v310_data));
          float v1058_data = ir3[6];
          ir3[6] = (v1058_data + (v1025_data * v315_data));
          float v1063_data = ir3[7];
          ir3[7] = (v1063_data + (v1025_data * v320_data));
          float v1068_data = ir3[8];
          ir3[8] = (v1068_data + (v1025_data * v325_data));
          float v1073_data = ir3[9];
          ir3[9] = (v1073_data + (v1025_data * v330_data));
          float v1078_data = ir3[10];
          ir3[10] = (v1078_data + (v1025_data * v335_data));
          float v1083_data = ir3[11];
          ir3[11] = (v1083_data + (v1025_data * v340_data));
          float v1085_data = r2[5];
          float v1088_data = ir3[0];
          ir3[0] = (v1088_data + (v1085_data * v345_data));
          float v1093_data = ir3[1];
          ir3[1] = (v1093_data + (v1085_data * v350_data));
          float v1098_data = ir3[2];
          ir3[2] = (v1098_data + (v1085_data * v355_data));
          float v1103_data = ir3[3];
          ir3[3] = (v1103_data + (v1085_data * v360_data));
          float v1108_data = ir3[4];
          ir3[4] = (v1108_data + (v1085_data * v365_data));
          float v1113_data = ir3[5];
          ir3[5] = (v1113_data + (v1085_data * v370_data));
          float v1118_data = ir3[6];
          ir3[6] = (v1118_data + (v1085_data * v375_data));
          float v1123_data = ir3[7];
          ir3[7] = (v1123_data + (v1085_data * v380_data));
          float v1128_data = ir3[8];
          ir3[8] = (v1128_data + (v1085_data * v385_data));
          float v1133_data = ir3[9];
          ir3[9] = (v1133_data + (v1085_data * v390_data));
          float v1138_data = ir3[10];
          ir3[10] = (v1138_data + (v1085_data * v395_data));
          float v1143_data = ir3[11];
          ir3[11] = (v1143_data + (v1085_data * v400_data));
          float v1145_data = r2[6];
          float v1148_data = ir3[0];
          ir3[0] = (v1148_data + (v1145_data * v405_data));
          float v1153_data = ir3[1];
          ir3[1] = (v1153_data + (v1145_data * v410_data));
          float v1158_data = ir3[2];
          ir3[2] = (v1158_data + (v1145_data * v415_data));
          float v1163_data = ir3[3];
          ir3[3] = (v1163_data + (v1145_data * v420_data));
          float v1168_data = ir3[4];
          ir3[4] = (v1168_data + (v1145_data * v425_data));
          float v1173_data = ir3[5];
          ir3[5] = (v1173_data + (v1145_data * v430_data));
          float v1178_data = ir3[6];
          ir3[6] = (v1178_data + (v1145_data * v435_data));
          float v1183_data = ir3[7];
          ir3[7] = (v1183_data + (v1145_data * v440_data));
          float v1188_data = ir3[8];
          ir3[8] = (v1188_data + (v1145_data * v445_data));
          float v1193_data = ir3[9];
          ir3[9] = (v1193_data + (v1145_data * v450_data));
          float v1198_data = ir3[10];
          ir3[10] = (v1198_data + (v1145_data * v455_data));
          float v1203_data = ir3[11];
          ir3[11] = (v1203_data + (v1145_data * v460_data));
          float v1205_data = r2[7];
          float v1208_data = ir3[0];
          ir3[0] = (v1208_data + (v1205_data * v465_data));
          float v1213_data = ir3[1];
          ir3[1] = (v1213_data + (v1205_data * v470_data));
          float v1218_data = ir3[2];
          ir3[2] = (v1218_data + (v1205_data * v475_data));
          float v1223_data = ir3[3];
          ir3[3] = (v1223_data + (v1205_data * v480_data));
          float v1228_data = ir3[4];
          ir3[4] = (v1228_data + (v1205_data * v485_data));
          float v1233_data = ir3[5];
          ir3[5] = (v1233_data + (v1205_data * v490_data));
          float v1238_data = ir3[6];
          ir3[6] = (v1238_data + (v1205_data * v495_data));
          float v1243_data = ir3[7];
          ir3[7] = (v1243_data + (v1205_data * v500_data));
          float v1248_data = ir3[8];
          ir3[8] = (v1248_data + (v1205_data * v505_data));
          float v1253_data = ir3[9];
          ir3[9] = (v1253_data + (v1205_data * v510_data));
          float v1258_data = ir3[10];
          ir3[10] = (v1258_data + (v1205_data * v515_data));
          float v1263_data = ir3[11];
          ir3[11] = (v1263_data + (v1205_data * v520_data));
          float v1265_data = r2[8];
          float v1268_data = ir3[0];
          ir3[0] = (v1268_data + (v1265_data * v525_data));
          float v1273_data = ir3[1];
          ir3[1] = (v1273_data + (v1265_data * v530_data));
          float v1278_data = ir3[2];
          ir3[2] = (v1278_data + (v1265_data * v535_data));
          float v1283_data = ir3[3];
          ir3[3] = (v1283_data + (v1265_data * v540_data));
          float v1288_data = ir3[4];
          ir3[4] = (v1288_data + (v1265_data * v545_data));
          float v1293_data = ir3[5];
          ir3[5] = (v1293_data + (v1265_data * v550_data));
          float v1298_data = ir3[6];
          ir3[6] = (v1298_data + (v1265_data * v555_data));
          float v1303_data = ir3[7];
          ir3[7] = (v1303_data + (v1265_data * v560_data));
          float v1308_data = ir3[8];
          ir3[8] = (v1308_data + (v1265_data * v565_data));
          float v1313_data = ir3[9];
          ir3[9] = (v1313_data + (v1265_data * v570_data));
          float v1318_data = ir3[10];
          ir3[10] = (v1318_data + (v1265_data * v575_data));
          float v1323_data = ir3[11];
          ir3[11] = (v1323_data + (v1265_data * v580_data));
          float v1325_data = r2[9];
          float v1328_data = ir3[0];
          ir3[0] = (v1328_data + (v1325_data * v585_data));
          float v1333_data = ir3[1];
          ir3[1] = (v1333_data + (v1325_data * v590_data));
          float v1338_data = ir3[2];
          ir3[2] = (v1338_data + (v1325_data * v595_data));
          float v1343_data = ir3[3];
          ir3[3] = (v1343_data + (v1325_data * v600_data));
          float v1348_data = ir3[4];
          ir3[4] = (v1348_data + (v1325_data * v605_data));
          float v1353_data = ir3[5];
          ir3[5] = (v1353_data + (v1325_data * v610_data));
          float v1358_data = ir3[6];
          ir3[6] = (v1358_data + (v1325_data * v615_data));
          float v1363_data = ir3[7];
          ir3[7] = (v1363_data + (v1325_data * v620_data));
          float v1368_data = ir3[8];
          ir3[8] = (v1368_data + (v1325_data * v625_data));
          float v1373_data = ir3[9];
          ir3[9] = (v1373_data + (v1325_data * v630_data));
          float v1378_data = ir3[10];
          ir3[10] = (v1378_data + (v1325_data * v635_data));
          float v1383_data = ir3[11];
          ir3[11] = (v1383_data + (v1325_data * v640_data));
          float v1385_data = r2[10];
          float v1388_data = ir3[0];
          ir3[0] = (v1388_data + (v1385_data * v645_data));
          float v1393_data = ir3[1];
          ir3[1] = (v1393_data + (v1385_data * v650_data));
          float v1398_data = ir3[2];
          ir3[2] = (v1398_data + (v1385_data * v655_data));
          float v1403_data = ir3[3];
          ir3[3] = (v1403_data + (v1385_data * v660_data));
          float v1408_data = ir3[4];
          ir3[4] = (v1408_data + (v1385_data * v665_data));
          float v1413_data = ir3[5];
          ir3[5] = (v1413_data + (v1385_data * v670_data));
          float v1418_data = ir3[6];
          ir3[6] = (v1418_data + (v1385_data * v675_data));
          float v1423_data = ir3[7];
          ir3[7] = (v1423_data + (v1385_data * v680_data));
          float v1428_data = ir3[8];
          ir3[8] = (v1428_data + (v1385_data * v685_data));
          float v1433_data = ir3[9];
          ir3[9] = (v1433_data + (v1385_data * v690_data));
          float v1438_data = ir3[10];
          ir3[10] = (v1438_data + (v1385_data * v695_data));
          float v1443_data = ir3[11];
          ir3[11] = (v1443_data + (v1385_data * v700_data));
          float v1445_data = r2[11];
          float v1448_data = ir3[0];
          ir3[0] = (v1448_data + (v1445_data * v705_data));
          float v1453_data = ir3[1];
          ir3[1] = (v1453_data + (v1445_data * v710_data));
          float v1458_data = ir3[2];
          ir3[2] = (v1458_data + (v1445_data * v715_data));
          float v1463_data = ir3[3];
          ir3[3] = (v1463_data + (v1445_data * v720_data));
          float v1468_data = ir3[4];
          ir3[4] = (v1468_data + (v1445_data * v725_data));
          float v1473_data = ir3[5];
          ir3[5] = (v1473_data + (v1445_data * v730_data));
          float v1478_data = ir3[6];
          ir3[6] = (v1478_data + (v1445_data * v735_data));
          float v1483_data = ir3[7];
          ir3[7] = (v1483_data + (v1445_data * v740_data));
          float v1488_data = ir3[8];
          ir3[8] = (v1488_data + (v1445_data * v745_data));
          float v1493_data = ir3[9];
          ir3[9] = (v1493_data + (v1445_data * v750_data));
          float v1498_data = ir3[10];
          ir3[10] = (v1498_data + (v1445_data * v755_data));
          float v1503_data = ir3[11];
          ir3[11] = (v1503_data + (v1445_data * v760_data));
          // r3 = ir3
          if (v26_g) {
            #pragma unroll
            for (int32_t v1505_n1 = 0; v1505_n1 < 12; ++v1505_n1) {
              float v1507_data = ir3[v1505_n1];
              r3[v1505_n1] = v1507_data;
            }
          }
          // s1 = store{r>s, clear}(localShrMem0, r3);
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          bool v1509_g = v25_lead < 12;
          if ((v25_lead >= 6) && v1509_g) {
            #pragma unroll
            for (int32_t v1511_z1 = 0; v1511_z1 < 12; ++v1511_z1) {
              int32_t v1516_a = v25_lead + (v1511_z1 * 12);
              s1[(v1516_a ^ ((v1516_a >> 4) & 15))] = 0.0f;
            }
          }
          if (v26_g) {
            #pragma unroll
            for (int32_t v1520_i1 = 0; v1520_i1 < 12; ++v1520_i1) {
              float v1522_data = r3[v1520_i1];
              int32_t v1526_a = v25_lead + (v1520_i1 * 12);
              s1[(v1526_a ^ ((v1526_a >> 4) & 15))] = v1522_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m3););
          float r5[12]{};
          // ir5 = +(r4)
          // [(0, 6), (0, 12)] []
          float ir5[12]{};
          float v1532_data = r4[0];
          float v1533_data = ir5[0];
          ir5[0] = (v1533_data + v1532_data);
          float v1535_data = r4[1];
          float v1536_data = ir5[1];
          ir5[1] = (v1536_data + v1535_data);
          float v1538_data = r4[2];
          float v1539_data = ir5[2];
          ir5[2] = (v1539_data + v1538_data);
          float v1541_data = r4[3];
          float v1542_data = ir5[3];
          ir5[3] = (v1542_data + v1541_data);
          float v1544_data = r4[4];
          float v1545_data = ir5[4];
          ir5[4] = (v1545_data + v1544_data);
          float v1547_data = r4[5];
          float v1548_data = ir5[5];
          ir5[5] = (v1548_data + v1547_data);
          float v1550_data = r4[6];
          float v1551_data = ir5[6];
          ir5[6] = (v1551_data + v1550_data);
          float v1553_data = r4[7];
          float v1554_data = ir5[7];
          ir5[7] = (v1554_data + v1553_data);
          float v1556_data = r4[8];
          float v1557_data = ir5[8];
          ir5[8] = (v1557_data + v1556_data);
          float v1559_data = r4[9];
          float v1560_data = ir5[9];
          ir5[9] = (v1560_data + v1559_data);
          float v1562_data = r4[10];
          float v1563_data = ir5[10];
          ir5[10] = (v1563_data + v1562_data);
          float v1565_data = r4[11];
          float v1566_data = ir5[11];
          ir5[11] = (v1566_data + v1565_data);
          // r5 = ir5
          if (v26_g) {
            #pragma unroll
            for (int32_t v1568_n1 = 0; v1568_n1 < 12; ++v1568_n1) {
              float v1570_data = ir5[v1568_n1];
              r5[v1568_n1] = v1570_data;
            }
          }
          // s1 = store{r>s}(localShrMem0, r5);
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          if (v26_g) {
            int32_t v1576_off = v25_lead + 6;
            #pragma unroll
            for (int32_t v1571_i1 = 0; v1571_i1 < 12; ++v1571_i1) {
              float v1573_data = r5[v1571_i1];
              int32_t v1578_a = v1576_off + (v1571_i1 * 12);
              s1[(v1578_a ^ ((v1578_a >> 4) & 15))] = v1573_data;
            }
          }
          float r6[12]{};
          // ir6 = +(s1)
          // [(0, 12), (0, 12)] []
          float ir6[12]{};
          int32_t v1589_sw = v25_lead ^ ((v25_lead >> 4) & 15);
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v1590_data = v1509_g ? (s1[v1589_sw]) : (0.0f);
          float v1591_data = ir6[0];
          ir6[0] = (v1591_data + v1590_data);
          int32_t v1593_a = v25_lead + 12;
          float v1597_data = v1509_g ? (s1[(v1593_a ^ ((v1593_a >> 4) & 15))]) : (0.0f);
          float v1598_data = ir6[1];
          ir6[1] = (v1598_data + v1597_data);
          int32_t v1600_a = v25_lead + 24;
          float v1604_data = v1509_g ? (s1[(v1600_a ^ ((v1600_a >> 4) & 15))]) : (0.0f);
          float v1605_data = ir6[2];
          ir6[2] = (v1605_data + v1604_data);
          int32_t v1607_a = v25_lead + 36;
          float v1611_data = v1509_g ? (s1[(v1607_a ^ ((v1607_a >> 4) & 15))]) : (0.0f);
          float v1612_data = ir6[3];
          ir6[3] = (v1612_data + v1611_data);
          int32_t v1614_a = v25_lead + 48;
          float v1618_data = v1509_g ? (s1[(v1614_a ^ ((v1614_a >> 4) & 15))]) : (0.0f);
          float v1619_data = ir6[4];
          ir6[4] = (v1619_data + v1618_data);
          int32_t v1621_a = v25_lead + 60;
          float v1625_data = v1509_g ? (s1[(v1621_a ^ ((v1621_a >> 4) & 15))]) : (0.0f);
          float v1626_data = ir6[5];
          ir6[5] = (v1626_data + v1625_data);
          int32_t v1628_a = v25_lead + 72;
          float v1632_data = v1509_g ? (s1[(v1628_a ^ ((v1628_a >> 4) & 15))]) : (0.0f);
          float v1633_data = ir6[6];
          ir6[6] = (v1633_data + v1632_data);
          int32_t v1635_a = v25_lead + 84;
          float v1639_data = v1509_g ? (s1[(v1635_a ^ ((v1635_a >> 4) & 15))]) : (0.0f);
          float v1640_data = ir6[7];
          ir6[7] = (v1640_data + v1639_data);
          int32_t v1642_a = v25_lead + 96;
          float v1646_data = v1509_g ? (s1[(v1642_a ^ ((v1642_a >> 4) & 15))]) : (0.0f);
          float v1647_data = ir6[8];
          ir6[8] = (v1647_data + v1646_data);
          int32_t v1649_a = v25_lead + 108;
          float v1653_data = v1509_g ? (s1[(v1649_a ^ ((v1649_a >> 4) & 15))]) : (0.0f);
          float v1654_data = ir6[9];
          ir6[9] = (v1654_data + v1653_data);
          int32_t v1656_a = v25_lead + 120;
          float v1660_data = v1509_g ? (s1[(v1656_a ^ ((v1656_a >> 4) & 15))]) : (0.0f);
          float v1661_data = ir6[10];
          ir6[10] = (v1661_data + v1660_data);
          int32_t v1663_a = v25_lead + 132;
          float v1667_data = v1509_g ? (s1[(v1663_a ^ ((v1663_a >> 4) & 15))]) : (0.0f);
          float v1668_data = ir6[11];
          ir6[11] = (v1668_data + v1667_data);
          // r6 = ir6
          if (v1509_g) {
            #pragma unroll
            for (int32_t v1670_n1 = 0; v1670_n1 < 12; ++v1670_n1) {
              float v1672_data = ir6[v1670_n1];
              r6[v1670_n1] = v1672_data;
            }
          }
          // glb_m4 = store{r>g}(r6);
          if (v1509_g) {
            #pragma unroll
            for (int32_t v1673_i1 = 0; v1673_i1 < 12; ++v1673_i1) {
              float v1675_data = r6[v1673_i1];
              glb_m4[(v25_lead + (v1673_i1 * 12))] = v1675_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

