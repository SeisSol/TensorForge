// === base name ===
kernel_3ea9cc9a83711564

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_3ea9cc9a83711564 = {{16, 8, 1}, 16, 12, 1, 8, 5632, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_3ea9cc9a83711564(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_3ea9cc9a83711564(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_3ea9cc9a83711564(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_3ea9cc9a83711564, block.x * block.y * block.z, 1408 * sizeof(float));
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
void launcher_kernel_3ea9cc9a83711564(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_3ea9cc9a83711564(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_3ea9cc9a83711564, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_3ea9cc9a83711564<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_3ea9cc9a83711564(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 8 per block = block 16x8x1, 5632 B shared, occupancy grid
    // operands:
    //   m0 32×32(12×12) {0..12}×{0..12} strided
    //   m1 32×32(12×12) {0..12}×{0..12} strided
    //   m2 32×32(12×12) {0..12}×{0..12} strided
    //   m3 32×32(4×12) {4..8}×{0..12} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    //   D = abs(N)
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1408}],"shared_bytes":5632,"shared_elements":1408,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[12,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"N","bbox":[[4,0],[8,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[4,0],[8,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[4,0],[8,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[176 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[160];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v5_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v5_batchId0 < numElements0; v5_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v6_ahead1 = v5_batchId0 + (gridDim.x * blockDim.y);
        size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 144 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 144 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 144 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[v5_batchId0 * 48 + 0 + m3_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v20_lead = threadIdx.x % 16;
          bool v21_g = v20_lead < 12;
          if (v21_g) {
            #pragma unroll
            for (int32_t v22_i1 = 0; v22_i1 < 12; ++v22_i1) {
              float v27_data = __ldcg(&glb_m1[(v20_lead + (v22_i1 * 12))]);
              r0[v22_i1] = v27_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 9; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[12]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // ir1 = +(r0 * s0)
          // [(0, 12), (0, 12)] [(0, 12)]
          float ir1[12]{};
          float v32_data = r0[0];
          float v33_data = s0[0];
          float v35_data = ir1[0];
          ir1[0] = (v35_data + (v32_data * v33_data));
          float v38_data = s0[12];
          float v40_data = ir1[1];
          ir1[1] = (v40_data + (v32_data * v38_data));
          float v43_data = s0[24];
          float v45_data = ir1[2];
          ir1[2] = (v45_data + (v32_data * v43_data));
          float v48_data = s0[36];
          float v50_data = ir1[3];
          ir1[3] = (v50_data + (v32_data * v48_data));
          float v53_data = s0[48];
          float v55_data = ir1[4];
          ir1[4] = (v55_data + (v32_data * v53_data));
          float v58_data = s0[60];
          float v60_data = ir1[5];
          ir1[5] = (v60_data + (v32_data * v58_data));
          float v63_data = s0[72];
          float v65_data = ir1[6];
          ir1[6] = (v65_data + (v32_data * v63_data));
          float v68_data = s0[84];
          float v70_data = ir1[7];
          ir1[7] = (v70_data + (v32_data * v68_data));
          float v73_data = s0[96];
          float v75_data = ir1[8];
          ir1[8] = (v75_data + (v32_data * v73_data));
          float v78_data = s0[108];
          float v80_data = ir1[9];
          ir1[9] = (v80_data + (v32_data * v78_data));
          float v83_data = s0[120];
          float v85_data = ir1[10];
          ir1[10] = (v85_data + (v32_data * v83_data));
          float v88_data = s0[132];
          float v90_data = ir1[11];
          ir1[11] = (v90_data + (v32_data * v88_data));
          float v92_data = r0[1];
          float v93_data = s0[1];
          float v95_data = ir1[0];
          ir1[0] = (v95_data + (v92_data * v93_data));
          float v98_data = s0[13];
          float v100_data = ir1[1];
          ir1[1] = (v100_data + (v92_data * v98_data));
          float v103_data = s0[25];
          float v105_data = ir1[2];
          ir1[2] = (v105_data + (v92_data * v103_data));
          float v108_data = s0[37];
          float v110_data = ir1[3];
          ir1[3] = (v110_data + (v92_data * v108_data));
          float v113_data = s0[49];
          float v115_data = ir1[4];
          ir1[4] = (v115_data + (v92_data * v113_data));
          float v118_data = s0[61];
          float v120_data = ir1[5];
          ir1[5] = (v120_data + (v92_data * v118_data));
          float v123_data = s0[73];
          float v125_data = ir1[6];
          ir1[6] = (v125_data + (v92_data * v123_data));
          float v128_data = s0[85];
          float v130_data = ir1[7];
          ir1[7] = (v130_data + (v92_data * v128_data));
          float v133_data = s0[97];
          float v135_data = ir1[8];
          ir1[8] = (v135_data + (v92_data * v133_data));
          float v138_data = s0[109];
          float v140_data = ir1[9];
          ir1[9] = (v140_data + (v92_data * v138_data));
          float v143_data = s0[121];
          float v145_data = ir1[10];
          ir1[10] = (v145_data + (v92_data * v143_data));
          float v148_data = s0[133];
          float v150_data = ir1[11];
          ir1[11] = (v150_data + (v92_data * v148_data));
          float v152_data = r0[2];
          float v153_data = s0[2];
          float v155_data = ir1[0];
          ir1[0] = (v155_data + (v152_data * v153_data));
          float v158_data = s0[14];
          float v160_data = ir1[1];
          ir1[1] = (v160_data + (v152_data * v158_data));
          float v163_data = s0[26];
          float v165_data = ir1[2];
          ir1[2] = (v165_data + (v152_data * v163_data));
          float v168_data = s0[38];
          float v170_data = ir1[3];
          ir1[3] = (v170_data + (v152_data * v168_data));
          float v173_data = s0[50];
          float v175_data = ir1[4];
          ir1[4] = (v175_data + (v152_data * v173_data));
          float v178_data = s0[62];
          float v180_data = ir1[5];
          ir1[5] = (v180_data + (v152_data * v178_data));
          float v183_data = s0[74];
          float v185_data = ir1[6];
          ir1[6] = (v185_data + (v152_data * v183_data));
          float v188_data = s0[86];
          float v190_data = ir1[7];
          ir1[7] = (v190_data + (v152_data * v188_data));
          float v193_data = s0[98];
          float v195_data = ir1[8];
          ir1[8] = (v195_data + (v152_data * v193_data));
          float v198_data = s0[110];
          float v200_data = ir1[9];
          ir1[9] = (v200_data + (v152_data * v198_data));
          float v203_data = s0[122];
          float v205_data = ir1[10];
          ir1[10] = (v205_data + (v152_data * v203_data));
          float v208_data = s0[134];
          float v210_data = ir1[11];
          ir1[11] = (v210_data + (v152_data * v208_data));
          float v212_data = r0[3];
          float v213_data = s0[3];
          float v215_data = ir1[0];
          ir1[0] = (v215_data + (v212_data * v213_data));
          float v218_data = s0[15];
          float v220_data = ir1[1];
          ir1[1] = (v220_data + (v212_data * v218_data));
          float v223_data = s0[27];
          float v225_data = ir1[2];
          ir1[2] = (v225_data + (v212_data * v223_data));
          float v228_data = s0[39];
          float v230_data = ir1[3];
          ir1[3] = (v230_data + (v212_data * v228_data));
          float v233_data = s0[51];
          float v235_data = ir1[4];
          ir1[4] = (v235_data + (v212_data * v233_data));
          float v238_data = s0[63];
          float v240_data = ir1[5];
          ir1[5] = (v240_data + (v212_data * v238_data));
          float v243_data = s0[75];
          float v245_data = ir1[6];
          ir1[6] = (v245_data + (v212_data * v243_data));
          float v248_data = s0[87];
          float v250_data = ir1[7];
          ir1[7] = (v250_data + (v212_data * v248_data));
          float v253_data = s0[99];
          float v255_data = ir1[8];
          ir1[8] = (v255_data + (v212_data * v253_data));
          float v258_data = s0[111];
          float v260_data = ir1[9];
          ir1[9] = (v260_data + (v212_data * v258_data));
          float v263_data = s0[123];
          float v265_data = ir1[10];
          ir1[10] = (v265_data + (v212_data * v263_data));
          float v268_data = s0[135];
          float v270_data = ir1[11];
          ir1[11] = (v270_data + (v212_data * v268_data));
          float v272_data = r0[4];
          float v273_data = s0[4];
          float v275_data = ir1[0];
          ir1[0] = (v275_data + (v272_data * v273_data));
          float v278_data = s0[16];
          float v280_data = ir1[1];
          ir1[1] = (v280_data + (v272_data * v278_data));
          float v283_data = s0[28];
          float v285_data = ir1[2];
          ir1[2] = (v285_data + (v272_data * v283_data));
          float v288_data = s0[40];
          float v290_data = ir1[3];
          ir1[3] = (v290_data + (v272_data * v288_data));
          float v293_data = s0[52];
          float v295_data = ir1[4];
          ir1[4] = (v295_data + (v272_data * v293_data));
          float v298_data = s0[64];
          float v300_data = ir1[5];
          ir1[5] = (v300_data + (v272_data * v298_data));
          float v303_data = s0[76];
          float v305_data = ir1[6];
          ir1[6] = (v305_data + (v272_data * v303_data));
          float v308_data = s0[88];
          float v310_data = ir1[7];
          ir1[7] = (v310_data + (v272_data * v308_data));
          float v313_data = s0[100];
          float v315_data = ir1[8];
          ir1[8] = (v315_data + (v272_data * v313_data));
          float v318_data = s0[112];
          float v320_data = ir1[9];
          ir1[9] = (v320_data + (v272_data * v318_data));
          float v323_data = s0[124];
          float v325_data = ir1[10];
          ir1[10] = (v325_data + (v272_data * v323_data));
          float v328_data = s0[136];
          float v330_data = ir1[11];
          ir1[11] = (v330_data + (v272_data * v328_data));
          float v332_data = r0[5];
          float v333_data = s0[5];
          float v335_data = ir1[0];
          ir1[0] = (v335_data + (v332_data * v333_data));
          float v338_data = s0[17];
          float v340_data = ir1[1];
          ir1[1] = (v340_data + (v332_data * v338_data));
          float v343_data = s0[29];
          float v345_data = ir1[2];
          ir1[2] = (v345_data + (v332_data * v343_data));
          float v348_data = s0[41];
          float v350_data = ir1[3];
          ir1[3] = (v350_data + (v332_data * v348_data));
          float v353_data = s0[53];
          float v355_data = ir1[4];
          ir1[4] = (v355_data + (v332_data * v353_data));
          float v358_data = s0[65];
          float v360_data = ir1[5];
          ir1[5] = (v360_data + (v332_data * v358_data));
          float v363_data = s0[77];
          float v365_data = ir1[6];
          ir1[6] = (v365_data + (v332_data * v363_data));
          float v368_data = s0[89];
          float v370_data = ir1[7];
          ir1[7] = (v370_data + (v332_data * v368_data));
          float v373_data = s0[101];
          float v375_data = ir1[8];
          ir1[8] = (v375_data + (v332_data * v373_data));
          float v378_data = s0[113];
          float v380_data = ir1[9];
          ir1[9] = (v380_data + (v332_data * v378_data));
          float v383_data = s0[125];
          float v385_data = ir1[10];
          ir1[10] = (v385_data + (v332_data * v383_data));
          float v388_data = s0[137];
          float v390_data = ir1[11];
          ir1[11] = (v390_data + (v332_data * v388_data));
          float v392_data = r0[6];
          float v393_data = s0[6];
          float v395_data = ir1[0];
          ir1[0] = (v395_data + (v392_data * v393_data));
          float v398_data = s0[18];
          float v400_data = ir1[1];
          ir1[1] = (v400_data + (v392_data * v398_data));
          float v403_data = s0[30];
          float v405_data = ir1[2];
          ir1[2] = (v405_data + (v392_data * v403_data));
          float v408_data = s0[42];
          float v410_data = ir1[3];
          ir1[3] = (v410_data + (v392_data * v408_data));
          float v413_data = s0[54];
          float v415_data = ir1[4];
          ir1[4] = (v415_data + (v392_data * v413_data));
          float v418_data = s0[66];
          float v420_data = ir1[5];
          ir1[5] = (v420_data + (v392_data * v418_data));
          float v423_data = s0[78];
          float v425_data = ir1[6];
          ir1[6] = (v425_data + (v392_data * v423_data));
          float v428_data = s0[90];
          float v430_data = ir1[7];
          ir1[7] = (v430_data + (v392_data * v428_data));
          float v433_data = s0[102];
          float v435_data = ir1[8];
          ir1[8] = (v435_data + (v392_data * v433_data));
          float v438_data = s0[114];
          float v440_data = ir1[9];
          ir1[9] = (v440_data + (v392_data * v438_data));
          float v443_data = s0[126];
          float v445_data = ir1[10];
          ir1[10] = (v445_data + (v392_data * v443_data));
          float v448_data = s0[138];
          float v450_data = ir1[11];
          ir1[11] = (v450_data + (v392_data * v448_data));
          float v452_data = r0[7];
          float v453_data = s0[7];
          float v455_data = ir1[0];
          ir1[0] = (v455_data + (v452_data * v453_data));
          float v458_data = s0[19];
          float v460_data = ir1[1];
          ir1[1] = (v460_data + (v452_data * v458_data));
          float v463_data = s0[31];
          float v465_data = ir1[2];
          ir1[2] = (v465_data + (v452_data * v463_data));
          float v468_data = s0[43];
          float v470_data = ir1[3];
          ir1[3] = (v470_data + (v452_data * v468_data));
          float v473_data = s0[55];
          float v475_data = ir1[4];
          ir1[4] = (v475_data + (v452_data * v473_data));
          float v478_data = s0[67];
          float v480_data = ir1[5];
          ir1[5] = (v480_data + (v452_data * v478_data));
          float v483_data = s0[79];
          float v485_data = ir1[6];
          ir1[6] = (v485_data + (v452_data * v483_data));
          float v488_data = s0[91];
          float v490_data = ir1[7];
          ir1[7] = (v490_data + (v452_data * v488_data));
          float v493_data = s0[103];
          float v495_data = ir1[8];
          ir1[8] = (v495_data + (v452_data * v493_data));
          float v498_data = s0[115];
          float v500_data = ir1[9];
          ir1[9] = (v500_data + (v452_data * v498_data));
          float v503_data = s0[127];
          float v505_data = ir1[10];
          ir1[10] = (v505_data + (v452_data * v503_data));
          float v508_data = s0[139];
          float v510_data = ir1[11];
          ir1[11] = (v510_data + (v452_data * v508_data));
          float v512_data = r0[8];
          float v513_data = s0[8];
          float v515_data = ir1[0];
          ir1[0] = (v515_data + (v512_data * v513_data));
          float v518_data = s0[20];
          float v520_data = ir1[1];
          ir1[1] = (v520_data + (v512_data * v518_data));
          float v523_data = s0[32];
          float v525_data = ir1[2];
          ir1[2] = (v525_data + (v512_data * v523_data));
          float v528_data = s0[44];
          float v530_data = ir1[3];
          ir1[3] = (v530_data + (v512_data * v528_data));
          float v533_data = s0[56];
          float v535_data = ir1[4];
          ir1[4] = (v535_data + (v512_data * v533_data));
          float v538_data = s0[68];
          float v540_data = ir1[5];
          ir1[5] = (v540_data + (v512_data * v538_data));
          float v543_data = s0[80];
          float v545_data = ir1[6];
          ir1[6] = (v545_data + (v512_data * v543_data));
          float v548_data = s0[92];
          float v550_data = ir1[7];
          ir1[7] = (v550_data + (v512_data * v548_data));
          float v553_data = s0[104];
          float v555_data = ir1[8];
          ir1[8] = (v555_data + (v512_data * v553_data));
          float v558_data = s0[116];
          float v560_data = ir1[9];
          ir1[9] = (v560_data + (v512_data * v558_data));
          float v563_data = s0[128];
          float v565_data = ir1[10];
          ir1[10] = (v565_data + (v512_data * v563_data));
          float v568_data = s0[140];
          float v570_data = ir1[11];
          ir1[11] = (v570_data + (v512_data * v568_data));
          float v572_data = r0[9];
          float v573_data = s0[9];
          float v575_data = ir1[0];
          ir1[0] = (v575_data + (v572_data * v573_data));
          float v578_data = s0[21];
          float v580_data = ir1[1];
          ir1[1] = (v580_data + (v572_data * v578_data));
          float v583_data = s0[33];
          float v585_data = ir1[2];
          ir1[2] = (v585_data + (v572_data * v583_data));
          float v588_data = s0[45];
          float v590_data = ir1[3];
          ir1[3] = (v590_data + (v572_data * v588_data));
          float v593_data = s0[57];
          float v595_data = ir1[4];
          ir1[4] = (v595_data + (v572_data * v593_data));
          float v598_data = s0[69];
          float v600_data = ir1[5];
          ir1[5] = (v600_data + (v572_data * v598_data));
          float v603_data = s0[81];
          float v605_data = ir1[6];
          ir1[6] = (v605_data + (v572_data * v603_data));
          float v608_data = s0[93];
          float v610_data = ir1[7];
          ir1[7] = (v610_data + (v572_data * v608_data));
          float v613_data = s0[105];
          float v615_data = ir1[8];
          ir1[8] = (v615_data + (v572_data * v613_data));
          float v618_data = s0[117];
          float v620_data = ir1[9];
          ir1[9] = (v620_data + (v572_data * v618_data));
          float v623_data = s0[129];
          float v625_data = ir1[10];
          ir1[10] = (v625_data + (v572_data * v623_data));
          float v628_data = s0[141];
          float v630_data = ir1[11];
          ir1[11] = (v630_data + (v572_data * v628_data));
          float v632_data = r0[10];
          float v633_data = s0[10];
          float v635_data = ir1[0];
          ir1[0] = (v635_data + (v632_data * v633_data));
          float v638_data = s0[22];
          float v640_data = ir1[1];
          ir1[1] = (v640_data + (v632_data * v638_data));
          float v643_data = s0[34];
          float v645_data = ir1[2];
          ir1[2] = (v645_data + (v632_data * v643_data));
          float v648_data = s0[46];
          float v650_data = ir1[3];
          ir1[3] = (v650_data + (v632_data * v648_data));
          float v653_data = s0[58];
          float v655_data = ir1[4];
          ir1[4] = (v655_data + (v632_data * v653_data));
          float v658_data = s0[70];
          float v660_data = ir1[5];
          ir1[5] = (v660_data + (v632_data * v658_data));
          float v663_data = s0[82];
          float v665_data = ir1[6];
          ir1[6] = (v665_data + (v632_data * v663_data));
          float v668_data = s0[94];
          float v670_data = ir1[7];
          ir1[7] = (v670_data + (v632_data * v668_data));
          float v673_data = s0[106];
          float v675_data = ir1[8];
          ir1[8] = (v675_data + (v632_data * v673_data));
          float v678_data = s0[118];
          float v680_data = ir1[9];
          ir1[9] = (v680_data + (v632_data * v678_data));
          float v683_data = s0[130];
          float v685_data = ir1[10];
          ir1[10] = (v685_data + (v632_data * v683_data));
          float v688_data = s0[142];
          float v690_data = ir1[11];
          ir1[11] = (v690_data + (v632_data * v688_data));
          float v692_data = r0[11];
          float v693_data = s0[11];
          float v695_data = ir1[0];
          ir1[0] = (v695_data + (v692_data * v693_data));
          float v698_data = s0[23];
          float v700_data = ir1[1];
          ir1[1] = (v700_data + (v692_data * v698_data));
          float v703_data = s0[35];
          float v705_data = ir1[2];
          ir1[2] = (v705_data + (v692_data * v703_data));
          float v708_data = s0[47];
          float v710_data = ir1[3];
          ir1[3] = (v710_data + (v692_data * v708_data));
          float v713_data = s0[59];
          float v715_data = ir1[4];
          ir1[4] = (v715_data + (v692_data * v713_data));
          float v718_data = s0[71];
          float v720_data = ir1[5];
          ir1[5] = (v720_data + (v692_data * v718_data));
          float v723_data = s0[83];
          float v725_data = ir1[6];
          ir1[6] = (v725_data + (v692_data * v723_data));
          float v728_data = s0[95];
          float v730_data = ir1[7];
          ir1[7] = (v730_data + (v692_data * v728_data));
          float v733_data = s0[107];
          float v735_data = ir1[8];
          ir1[8] = (v735_data + (v692_data * v733_data));
          float v738_data = s0[119];
          float v740_data = ir1[9];
          ir1[9] = (v740_data + (v692_data * v738_data));
          float v743_data = s0[131];
          float v745_data = ir1[10];
          ir1[10] = (v745_data + (v692_data * v743_data));
          float v748_data = s0[143];
          float v750_data = ir1[11];
          ir1[11] = (v750_data + (v692_data * v748_data));
          // r1 = ir1
          if (v21_g) {
            #pragma unroll
            for (int32_t v752_n1 = 0; v752_n1 < 12; ++v752_n1) {
              float v754_data = ir1[v752_n1];
              r1[v752_n1] = v754_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v21_g) {
            #pragma unroll
            for (int32_t v755_i1 = 0; v755_i1 < 12; ++v755_i1) {
              float v757_data = r1[v755_i1];
              glb_m0[(v20_lead + (v755_i1 * 12))] = v757_data;
            }
          }
          float r2[12]{};
          // r2 = abs(glb_m3)
          bool v763_g = v20_lead < 4;
          if (v763_g) {
            int32_t v768_a = (v20_lead + 4) - 4;
            #pragma unroll
            for (int32_t v764_k1 = 0; v764_k1 < 12; ++v764_k1) {
              float v771_data = glb_m3[(v768_a + (v764_k1 * 4))];
              r2[v764_k1] = (fabsf(v771_data));
            }
          }
          // glb_m0 = store{r>g}(r2);
          if (v763_g) {
            int32_t v780_off = v20_lead + 4;
            #pragma unroll
            for (int32_t v775_i1 = 0; v775_i1 < 12; ++v775_i1) {
              float v777_data = r2[v775_i1];
              glb_m0[(v780_off + (v775_i1 * 12))] = v777_data;
            }
          }
          if (v20_lead >= 12) {
            int32_t v788_off = (v20_lead + -16_i32) + 4;
            #pragma unroll
            for (int32_t v784_z1 = 0; v784_z1 < 12; ++v784_z1) {
              glb_m0[(v788_off + (v784_z1 * 12))] = 0.0f;
            }
          }
          if ((v20_lead >= 4) && (v20_lead < 8)) {
            int32_t v798_off = v20_lead + 4;
            #pragma unroll
            for (int32_t v794_z1 = 0; v794_z1 < 12; ++v794_z1) {
              glb_m0[(v798_off + (v794_z1 * 12))] = 0.0f;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

