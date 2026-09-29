// === base name ===
kernel_f66fb02f25bfd5f0

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_f66fb02f25bfd5f0 = {{32, 4, 1}, 32, 24, 1, 4, 3584, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_f66fb02f25bfd5f0(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_f66fb02f25bfd5f0(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_f66fb02f25bfd5f0(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_f66fb02f25bfd5f0, block.x * block.y * block.z, 896 * sizeof(float));
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
  config.sharedMemBytes = 896 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_f66fb02f25bfd5f0(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_f66fb02f25bfd5f0(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_f66fb02f25bfd5f0, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_f66fb02f25bfd5f0<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_f66fb02f25bfd5f0(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (24 active) x 4 per block = block 32x4x1, 3584 B shared, occupancy grid
    // operands:
    //   m0 24×9(24×9) {0..24}×{0..9} strided
    //   m1 24×24(24×24) {0..24}×{0..24} strided
    //   m2 24×9(24×9) {0..24}×{0..9} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":24,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":896}],"shared_bytes":3584,"shared_elements":896,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[24,9]],"name":"m0","ordered":false,"parts":1,"shape":[24,9],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[24,24]],"name":"m1","ordered":false,"parts":1,"shape":[24,24],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[24,9]],"name":"m2","ordered":false,"parts":1,"shape":[24,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[24,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[24,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[24,24]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[24,24]},{"addressing":"strided","bbox":[[0,0],[24,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[24,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[224 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[224];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v5_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v5_batchId0 < numElements0; v5_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v6_ahead1 = v5_batchId0 + (gridDim.x * blockDim.y);
        size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 216 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 576 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 216 + 0 + m2_extraOffset];
          float r0[24]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v19_lead = threadIdx.x % 32;
          bool v20_g = v19_lead < 24;
          if (v20_g) {
            #pragma unroll
            for (int32_t v21_i1 = 0; v21_i1 < 24; ++v21_i1) {
              float v26_data = __ldcg(&glb_m1[(v19_lead + (v21_i1 * 24))]);
              r0[v21_i1] = v26_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 4 * threadIdx.x + 0], &glb_m2[0 + 0 + 4 * threadIdx.x + 0], 16);
          if (threadIdx.x < 22) {
            __pipeline_memcpy_async(&s0[0 + 0 + 4 * threadIdx.x + 128], &glb_m2[0 + 0 + 4 * threadIdx.x + 128], 16);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[9]{};
          __syncwarp();
          // ir1 = +(r0 * s0)
          // [(0, 24), (0, 9)] [(0, 24)]
          float ir1[9]{};
          float v32_data = r0[0];
          float v33_data = s0[0];
          float v35_data = ir1[0];
          ir1[0] = (v35_data + (v32_data * v33_data));
          float v38_data = s0[24];
          float v40_data = ir1[1];
          ir1[1] = (v40_data + (v32_data * v38_data));
          float v43_data = s0[48];
          float v45_data = ir1[2];
          ir1[2] = (v45_data + (v32_data * v43_data));
          float v48_data = s0[72];
          float v50_data = ir1[3];
          ir1[3] = (v50_data + (v32_data * v48_data));
          float v53_data = s0[96];
          float v55_data = ir1[4];
          ir1[4] = (v55_data + (v32_data * v53_data));
          float v58_data = s0[120];
          float v60_data = ir1[5];
          ir1[5] = (v60_data + (v32_data * v58_data));
          float v63_data = s0[144];
          float v65_data = ir1[6];
          ir1[6] = (v65_data + (v32_data * v63_data));
          float v68_data = s0[168];
          float v70_data = ir1[7];
          ir1[7] = (v70_data + (v32_data * v68_data));
          float v73_data = s0[192];
          float v75_data = ir1[8];
          ir1[8] = (v75_data + (v32_data * v73_data));
          float v77_data = r0[1];
          float v78_data = s0[1];
          float v80_data = ir1[0];
          ir1[0] = (v80_data + (v77_data * v78_data));
          float v83_data = s0[25];
          float v85_data = ir1[1];
          ir1[1] = (v85_data + (v77_data * v83_data));
          float v88_data = s0[49];
          float v90_data = ir1[2];
          ir1[2] = (v90_data + (v77_data * v88_data));
          float v93_data = s0[73];
          float v95_data = ir1[3];
          ir1[3] = (v95_data + (v77_data * v93_data));
          float v98_data = s0[97];
          float v100_data = ir1[4];
          ir1[4] = (v100_data + (v77_data * v98_data));
          float v103_data = s0[121];
          float v105_data = ir1[5];
          ir1[5] = (v105_data + (v77_data * v103_data));
          float v108_data = s0[145];
          float v110_data = ir1[6];
          ir1[6] = (v110_data + (v77_data * v108_data));
          float v113_data = s0[169];
          float v115_data = ir1[7];
          ir1[7] = (v115_data + (v77_data * v113_data));
          float v118_data = s0[193];
          float v120_data = ir1[8];
          ir1[8] = (v120_data + (v77_data * v118_data));
          float v122_data = r0[2];
          float v123_data = s0[2];
          float v125_data = ir1[0];
          ir1[0] = (v125_data + (v122_data * v123_data));
          float v128_data = s0[26];
          float v130_data = ir1[1];
          ir1[1] = (v130_data + (v122_data * v128_data));
          float v133_data = s0[50];
          float v135_data = ir1[2];
          ir1[2] = (v135_data + (v122_data * v133_data));
          float v138_data = s0[74];
          float v140_data = ir1[3];
          ir1[3] = (v140_data + (v122_data * v138_data));
          float v143_data = s0[98];
          float v145_data = ir1[4];
          ir1[4] = (v145_data + (v122_data * v143_data));
          float v148_data = s0[122];
          float v150_data = ir1[5];
          ir1[5] = (v150_data + (v122_data * v148_data));
          float v153_data = s0[146];
          float v155_data = ir1[6];
          ir1[6] = (v155_data + (v122_data * v153_data));
          float v158_data = s0[170];
          float v160_data = ir1[7];
          ir1[7] = (v160_data + (v122_data * v158_data));
          float v163_data = s0[194];
          float v165_data = ir1[8];
          ir1[8] = (v165_data + (v122_data * v163_data));
          float v167_data = r0[3];
          float v168_data = s0[3];
          float v170_data = ir1[0];
          ir1[0] = (v170_data + (v167_data * v168_data));
          float v173_data = s0[27];
          float v175_data = ir1[1];
          ir1[1] = (v175_data + (v167_data * v173_data));
          float v178_data = s0[51];
          float v180_data = ir1[2];
          ir1[2] = (v180_data + (v167_data * v178_data));
          float v183_data = s0[75];
          float v185_data = ir1[3];
          ir1[3] = (v185_data + (v167_data * v183_data));
          float v188_data = s0[99];
          float v190_data = ir1[4];
          ir1[4] = (v190_data + (v167_data * v188_data));
          float v193_data = s0[123];
          float v195_data = ir1[5];
          ir1[5] = (v195_data + (v167_data * v193_data));
          float v198_data = s0[147];
          float v200_data = ir1[6];
          ir1[6] = (v200_data + (v167_data * v198_data));
          float v203_data = s0[171];
          float v205_data = ir1[7];
          ir1[7] = (v205_data + (v167_data * v203_data));
          float v208_data = s0[195];
          float v210_data = ir1[8];
          ir1[8] = (v210_data + (v167_data * v208_data));
          float v212_data = r0[4];
          float v213_data = s0[4];
          float v215_data = ir1[0];
          ir1[0] = (v215_data + (v212_data * v213_data));
          float v218_data = s0[28];
          float v220_data = ir1[1];
          ir1[1] = (v220_data + (v212_data * v218_data));
          float v223_data = s0[52];
          float v225_data = ir1[2];
          ir1[2] = (v225_data + (v212_data * v223_data));
          float v228_data = s0[76];
          float v230_data = ir1[3];
          ir1[3] = (v230_data + (v212_data * v228_data));
          float v233_data = s0[100];
          float v235_data = ir1[4];
          ir1[4] = (v235_data + (v212_data * v233_data));
          float v238_data = s0[124];
          float v240_data = ir1[5];
          ir1[5] = (v240_data + (v212_data * v238_data));
          float v243_data = s0[148];
          float v245_data = ir1[6];
          ir1[6] = (v245_data + (v212_data * v243_data));
          float v248_data = s0[172];
          float v250_data = ir1[7];
          ir1[7] = (v250_data + (v212_data * v248_data));
          float v253_data = s0[196];
          float v255_data = ir1[8];
          ir1[8] = (v255_data + (v212_data * v253_data));
          float v257_data = r0[5];
          float v258_data = s0[5];
          float v260_data = ir1[0];
          ir1[0] = (v260_data + (v257_data * v258_data));
          float v263_data = s0[29];
          float v265_data = ir1[1];
          ir1[1] = (v265_data + (v257_data * v263_data));
          float v268_data = s0[53];
          float v270_data = ir1[2];
          ir1[2] = (v270_data + (v257_data * v268_data));
          float v273_data = s0[77];
          float v275_data = ir1[3];
          ir1[3] = (v275_data + (v257_data * v273_data));
          float v278_data = s0[101];
          float v280_data = ir1[4];
          ir1[4] = (v280_data + (v257_data * v278_data));
          float v283_data = s0[125];
          float v285_data = ir1[5];
          ir1[5] = (v285_data + (v257_data * v283_data));
          float v288_data = s0[149];
          float v290_data = ir1[6];
          ir1[6] = (v290_data + (v257_data * v288_data));
          float v293_data = s0[173];
          float v295_data = ir1[7];
          ir1[7] = (v295_data + (v257_data * v293_data));
          float v298_data = s0[197];
          float v300_data = ir1[8];
          ir1[8] = (v300_data + (v257_data * v298_data));
          float v302_data = r0[6];
          float v303_data = s0[6];
          float v305_data = ir1[0];
          ir1[0] = (v305_data + (v302_data * v303_data));
          float v308_data = s0[30];
          float v310_data = ir1[1];
          ir1[1] = (v310_data + (v302_data * v308_data));
          float v313_data = s0[54];
          float v315_data = ir1[2];
          ir1[2] = (v315_data + (v302_data * v313_data));
          float v318_data = s0[78];
          float v320_data = ir1[3];
          ir1[3] = (v320_data + (v302_data * v318_data));
          float v323_data = s0[102];
          float v325_data = ir1[4];
          ir1[4] = (v325_data + (v302_data * v323_data));
          float v328_data = s0[126];
          float v330_data = ir1[5];
          ir1[5] = (v330_data + (v302_data * v328_data));
          float v333_data = s0[150];
          float v335_data = ir1[6];
          ir1[6] = (v335_data + (v302_data * v333_data));
          float v338_data = s0[174];
          float v340_data = ir1[7];
          ir1[7] = (v340_data + (v302_data * v338_data));
          float v343_data = s0[198];
          float v345_data = ir1[8];
          ir1[8] = (v345_data + (v302_data * v343_data));
          float v347_data = r0[7];
          float v348_data = s0[7];
          float v350_data = ir1[0];
          ir1[0] = (v350_data + (v347_data * v348_data));
          float v353_data = s0[31];
          float v355_data = ir1[1];
          ir1[1] = (v355_data + (v347_data * v353_data));
          float v358_data = s0[55];
          float v360_data = ir1[2];
          ir1[2] = (v360_data + (v347_data * v358_data));
          float v363_data = s0[79];
          float v365_data = ir1[3];
          ir1[3] = (v365_data + (v347_data * v363_data));
          float v368_data = s0[103];
          float v370_data = ir1[4];
          ir1[4] = (v370_data + (v347_data * v368_data));
          float v373_data = s0[127];
          float v375_data = ir1[5];
          ir1[5] = (v375_data + (v347_data * v373_data));
          float v378_data = s0[151];
          float v380_data = ir1[6];
          ir1[6] = (v380_data + (v347_data * v378_data));
          float v383_data = s0[175];
          float v385_data = ir1[7];
          ir1[7] = (v385_data + (v347_data * v383_data));
          float v388_data = s0[199];
          float v390_data = ir1[8];
          ir1[8] = (v390_data + (v347_data * v388_data));
          float v392_data = r0[8];
          float v393_data = s0[8];
          float v395_data = ir1[0];
          ir1[0] = (v395_data + (v392_data * v393_data));
          float v398_data = s0[32];
          float v400_data = ir1[1];
          ir1[1] = (v400_data + (v392_data * v398_data));
          float v403_data = s0[56];
          float v405_data = ir1[2];
          ir1[2] = (v405_data + (v392_data * v403_data));
          float v408_data = s0[80];
          float v410_data = ir1[3];
          ir1[3] = (v410_data + (v392_data * v408_data));
          float v413_data = s0[104];
          float v415_data = ir1[4];
          ir1[4] = (v415_data + (v392_data * v413_data));
          float v418_data = s0[128];
          float v420_data = ir1[5];
          ir1[5] = (v420_data + (v392_data * v418_data));
          float v423_data = s0[152];
          float v425_data = ir1[6];
          ir1[6] = (v425_data + (v392_data * v423_data));
          float v428_data = s0[176];
          float v430_data = ir1[7];
          ir1[7] = (v430_data + (v392_data * v428_data));
          float v433_data = s0[200];
          float v435_data = ir1[8];
          ir1[8] = (v435_data + (v392_data * v433_data));
          float v437_data = r0[9];
          float v438_data = s0[9];
          float v440_data = ir1[0];
          ir1[0] = (v440_data + (v437_data * v438_data));
          float v443_data = s0[33];
          float v445_data = ir1[1];
          ir1[1] = (v445_data + (v437_data * v443_data));
          float v448_data = s0[57];
          float v450_data = ir1[2];
          ir1[2] = (v450_data + (v437_data * v448_data));
          float v453_data = s0[81];
          float v455_data = ir1[3];
          ir1[3] = (v455_data + (v437_data * v453_data));
          float v458_data = s0[105];
          float v460_data = ir1[4];
          ir1[4] = (v460_data + (v437_data * v458_data));
          float v463_data = s0[129];
          float v465_data = ir1[5];
          ir1[5] = (v465_data + (v437_data * v463_data));
          float v468_data = s0[153];
          float v470_data = ir1[6];
          ir1[6] = (v470_data + (v437_data * v468_data));
          float v473_data = s0[177];
          float v475_data = ir1[7];
          ir1[7] = (v475_data + (v437_data * v473_data));
          float v478_data = s0[201];
          float v480_data = ir1[8];
          ir1[8] = (v480_data + (v437_data * v478_data));
          float v482_data = r0[10];
          float v483_data = s0[10];
          float v485_data = ir1[0];
          ir1[0] = (v485_data + (v482_data * v483_data));
          float v488_data = s0[34];
          float v490_data = ir1[1];
          ir1[1] = (v490_data + (v482_data * v488_data));
          float v493_data = s0[58];
          float v495_data = ir1[2];
          ir1[2] = (v495_data + (v482_data * v493_data));
          float v498_data = s0[82];
          float v500_data = ir1[3];
          ir1[3] = (v500_data + (v482_data * v498_data));
          float v503_data = s0[106];
          float v505_data = ir1[4];
          ir1[4] = (v505_data + (v482_data * v503_data));
          float v508_data = s0[130];
          float v510_data = ir1[5];
          ir1[5] = (v510_data + (v482_data * v508_data));
          float v513_data = s0[154];
          float v515_data = ir1[6];
          ir1[6] = (v515_data + (v482_data * v513_data));
          float v518_data = s0[178];
          float v520_data = ir1[7];
          ir1[7] = (v520_data + (v482_data * v518_data));
          float v523_data = s0[202];
          float v525_data = ir1[8];
          ir1[8] = (v525_data + (v482_data * v523_data));
          float v527_data = r0[11];
          float v528_data = s0[11];
          float v530_data = ir1[0];
          ir1[0] = (v530_data + (v527_data * v528_data));
          float v533_data = s0[35];
          float v535_data = ir1[1];
          ir1[1] = (v535_data + (v527_data * v533_data));
          float v538_data = s0[59];
          float v540_data = ir1[2];
          ir1[2] = (v540_data + (v527_data * v538_data));
          float v543_data = s0[83];
          float v545_data = ir1[3];
          ir1[3] = (v545_data + (v527_data * v543_data));
          float v548_data = s0[107];
          float v550_data = ir1[4];
          ir1[4] = (v550_data + (v527_data * v548_data));
          float v553_data = s0[131];
          float v555_data = ir1[5];
          ir1[5] = (v555_data + (v527_data * v553_data));
          float v558_data = s0[155];
          float v560_data = ir1[6];
          ir1[6] = (v560_data + (v527_data * v558_data));
          float v563_data = s0[179];
          float v565_data = ir1[7];
          ir1[7] = (v565_data + (v527_data * v563_data));
          float v568_data = s0[203];
          float v570_data = ir1[8];
          ir1[8] = (v570_data + (v527_data * v568_data));
          float v572_data = r0[12];
          float v573_data = s0[12];
          float v575_data = ir1[0];
          ir1[0] = (v575_data + (v572_data * v573_data));
          float v578_data = s0[36];
          float v580_data = ir1[1];
          ir1[1] = (v580_data + (v572_data * v578_data));
          float v583_data = s0[60];
          float v585_data = ir1[2];
          ir1[2] = (v585_data + (v572_data * v583_data));
          float v588_data = s0[84];
          float v590_data = ir1[3];
          ir1[3] = (v590_data + (v572_data * v588_data));
          float v593_data = s0[108];
          float v595_data = ir1[4];
          ir1[4] = (v595_data + (v572_data * v593_data));
          float v598_data = s0[132];
          float v600_data = ir1[5];
          ir1[5] = (v600_data + (v572_data * v598_data));
          float v603_data = s0[156];
          float v605_data = ir1[6];
          ir1[6] = (v605_data + (v572_data * v603_data));
          float v608_data = s0[180];
          float v610_data = ir1[7];
          ir1[7] = (v610_data + (v572_data * v608_data));
          float v613_data = s0[204];
          float v615_data = ir1[8];
          ir1[8] = (v615_data + (v572_data * v613_data));
          float v617_data = r0[13];
          float v618_data = s0[13];
          float v620_data = ir1[0];
          ir1[0] = (v620_data + (v617_data * v618_data));
          float v623_data = s0[37];
          float v625_data = ir1[1];
          ir1[1] = (v625_data + (v617_data * v623_data));
          float v628_data = s0[61];
          float v630_data = ir1[2];
          ir1[2] = (v630_data + (v617_data * v628_data));
          float v633_data = s0[85];
          float v635_data = ir1[3];
          ir1[3] = (v635_data + (v617_data * v633_data));
          float v638_data = s0[109];
          float v640_data = ir1[4];
          ir1[4] = (v640_data + (v617_data * v638_data));
          float v643_data = s0[133];
          float v645_data = ir1[5];
          ir1[5] = (v645_data + (v617_data * v643_data));
          float v648_data = s0[157];
          float v650_data = ir1[6];
          ir1[6] = (v650_data + (v617_data * v648_data));
          float v653_data = s0[181];
          float v655_data = ir1[7];
          ir1[7] = (v655_data + (v617_data * v653_data));
          float v658_data = s0[205];
          float v660_data = ir1[8];
          ir1[8] = (v660_data + (v617_data * v658_data));
          float v662_data = r0[14];
          float v663_data = s0[14];
          float v665_data = ir1[0];
          ir1[0] = (v665_data + (v662_data * v663_data));
          float v668_data = s0[38];
          float v670_data = ir1[1];
          ir1[1] = (v670_data + (v662_data * v668_data));
          float v673_data = s0[62];
          float v675_data = ir1[2];
          ir1[2] = (v675_data + (v662_data * v673_data));
          float v678_data = s0[86];
          float v680_data = ir1[3];
          ir1[3] = (v680_data + (v662_data * v678_data));
          float v683_data = s0[110];
          float v685_data = ir1[4];
          ir1[4] = (v685_data + (v662_data * v683_data));
          float v688_data = s0[134];
          float v690_data = ir1[5];
          ir1[5] = (v690_data + (v662_data * v688_data));
          float v693_data = s0[158];
          float v695_data = ir1[6];
          ir1[6] = (v695_data + (v662_data * v693_data));
          float v698_data = s0[182];
          float v700_data = ir1[7];
          ir1[7] = (v700_data + (v662_data * v698_data));
          float v703_data = s0[206];
          float v705_data = ir1[8];
          ir1[8] = (v705_data + (v662_data * v703_data));
          float v707_data = r0[15];
          float v708_data = s0[15];
          float v710_data = ir1[0];
          ir1[0] = (v710_data + (v707_data * v708_data));
          float v713_data = s0[39];
          float v715_data = ir1[1];
          ir1[1] = (v715_data + (v707_data * v713_data));
          float v718_data = s0[63];
          float v720_data = ir1[2];
          ir1[2] = (v720_data + (v707_data * v718_data));
          float v723_data = s0[87];
          float v725_data = ir1[3];
          ir1[3] = (v725_data + (v707_data * v723_data));
          float v728_data = s0[111];
          float v730_data = ir1[4];
          ir1[4] = (v730_data + (v707_data * v728_data));
          float v733_data = s0[135];
          float v735_data = ir1[5];
          ir1[5] = (v735_data + (v707_data * v733_data));
          float v738_data = s0[159];
          float v740_data = ir1[6];
          ir1[6] = (v740_data + (v707_data * v738_data));
          float v743_data = s0[183];
          float v745_data = ir1[7];
          ir1[7] = (v745_data + (v707_data * v743_data));
          float v748_data = s0[207];
          float v750_data = ir1[8];
          ir1[8] = (v750_data + (v707_data * v748_data));
          float v752_data = r0[16];
          float v753_data = s0[16];
          float v755_data = ir1[0];
          ir1[0] = (v755_data + (v752_data * v753_data));
          float v758_data = s0[40];
          float v760_data = ir1[1];
          ir1[1] = (v760_data + (v752_data * v758_data));
          float v763_data = s0[64];
          float v765_data = ir1[2];
          ir1[2] = (v765_data + (v752_data * v763_data));
          float v768_data = s0[88];
          float v770_data = ir1[3];
          ir1[3] = (v770_data + (v752_data * v768_data));
          float v773_data = s0[112];
          float v775_data = ir1[4];
          ir1[4] = (v775_data + (v752_data * v773_data));
          float v778_data = s0[136];
          float v780_data = ir1[5];
          ir1[5] = (v780_data + (v752_data * v778_data));
          float v783_data = s0[160];
          float v785_data = ir1[6];
          ir1[6] = (v785_data + (v752_data * v783_data));
          float v788_data = s0[184];
          float v790_data = ir1[7];
          ir1[7] = (v790_data + (v752_data * v788_data));
          float v793_data = s0[208];
          float v795_data = ir1[8];
          ir1[8] = (v795_data + (v752_data * v793_data));
          float v797_data = r0[17];
          float v798_data = s0[17];
          float v800_data = ir1[0];
          ir1[0] = (v800_data + (v797_data * v798_data));
          float v803_data = s0[41];
          float v805_data = ir1[1];
          ir1[1] = (v805_data + (v797_data * v803_data));
          float v808_data = s0[65];
          float v810_data = ir1[2];
          ir1[2] = (v810_data + (v797_data * v808_data));
          float v813_data = s0[89];
          float v815_data = ir1[3];
          ir1[3] = (v815_data + (v797_data * v813_data));
          float v818_data = s0[113];
          float v820_data = ir1[4];
          ir1[4] = (v820_data + (v797_data * v818_data));
          float v823_data = s0[137];
          float v825_data = ir1[5];
          ir1[5] = (v825_data + (v797_data * v823_data));
          float v828_data = s0[161];
          float v830_data = ir1[6];
          ir1[6] = (v830_data + (v797_data * v828_data));
          float v833_data = s0[185];
          float v835_data = ir1[7];
          ir1[7] = (v835_data + (v797_data * v833_data));
          float v838_data = s0[209];
          float v840_data = ir1[8];
          ir1[8] = (v840_data + (v797_data * v838_data));
          float v842_data = r0[18];
          float v843_data = s0[18];
          float v845_data = ir1[0];
          ir1[0] = (v845_data + (v842_data * v843_data));
          float v848_data = s0[42];
          float v850_data = ir1[1];
          ir1[1] = (v850_data + (v842_data * v848_data));
          float v853_data = s0[66];
          float v855_data = ir1[2];
          ir1[2] = (v855_data + (v842_data * v853_data));
          float v858_data = s0[90];
          float v860_data = ir1[3];
          ir1[3] = (v860_data + (v842_data * v858_data));
          float v863_data = s0[114];
          float v865_data = ir1[4];
          ir1[4] = (v865_data + (v842_data * v863_data));
          float v868_data = s0[138];
          float v870_data = ir1[5];
          ir1[5] = (v870_data + (v842_data * v868_data));
          float v873_data = s0[162];
          float v875_data = ir1[6];
          ir1[6] = (v875_data + (v842_data * v873_data));
          float v878_data = s0[186];
          float v880_data = ir1[7];
          ir1[7] = (v880_data + (v842_data * v878_data));
          float v883_data = s0[210];
          float v885_data = ir1[8];
          ir1[8] = (v885_data + (v842_data * v883_data));
          float v887_data = r0[19];
          float v888_data = s0[19];
          float v890_data = ir1[0];
          ir1[0] = (v890_data + (v887_data * v888_data));
          float v893_data = s0[43];
          float v895_data = ir1[1];
          ir1[1] = (v895_data + (v887_data * v893_data));
          float v898_data = s0[67];
          float v900_data = ir1[2];
          ir1[2] = (v900_data + (v887_data * v898_data));
          float v903_data = s0[91];
          float v905_data = ir1[3];
          ir1[3] = (v905_data + (v887_data * v903_data));
          float v908_data = s0[115];
          float v910_data = ir1[4];
          ir1[4] = (v910_data + (v887_data * v908_data));
          float v913_data = s0[139];
          float v915_data = ir1[5];
          ir1[5] = (v915_data + (v887_data * v913_data));
          float v918_data = s0[163];
          float v920_data = ir1[6];
          ir1[6] = (v920_data + (v887_data * v918_data));
          float v923_data = s0[187];
          float v925_data = ir1[7];
          ir1[7] = (v925_data + (v887_data * v923_data));
          float v928_data = s0[211];
          float v930_data = ir1[8];
          ir1[8] = (v930_data + (v887_data * v928_data));
          float v932_data = r0[20];
          float v933_data = s0[20];
          float v935_data = ir1[0];
          ir1[0] = (v935_data + (v932_data * v933_data));
          float v938_data = s0[44];
          float v940_data = ir1[1];
          ir1[1] = (v940_data + (v932_data * v938_data));
          float v943_data = s0[68];
          float v945_data = ir1[2];
          ir1[2] = (v945_data + (v932_data * v943_data));
          float v948_data = s0[92];
          float v950_data = ir1[3];
          ir1[3] = (v950_data + (v932_data * v948_data));
          float v953_data = s0[116];
          float v955_data = ir1[4];
          ir1[4] = (v955_data + (v932_data * v953_data));
          float v958_data = s0[140];
          float v960_data = ir1[5];
          ir1[5] = (v960_data + (v932_data * v958_data));
          float v963_data = s0[164];
          float v965_data = ir1[6];
          ir1[6] = (v965_data + (v932_data * v963_data));
          float v968_data = s0[188];
          float v970_data = ir1[7];
          ir1[7] = (v970_data + (v932_data * v968_data));
          float v973_data = s0[212];
          float v975_data = ir1[8];
          ir1[8] = (v975_data + (v932_data * v973_data));
          float v977_data = r0[21];
          float v978_data = s0[21];
          float v980_data = ir1[0];
          ir1[0] = (v980_data + (v977_data * v978_data));
          float v983_data = s0[45];
          float v985_data = ir1[1];
          ir1[1] = (v985_data + (v977_data * v983_data));
          float v988_data = s0[69];
          float v990_data = ir1[2];
          ir1[2] = (v990_data + (v977_data * v988_data));
          float v993_data = s0[93];
          float v995_data = ir1[3];
          ir1[3] = (v995_data + (v977_data * v993_data));
          float v998_data = s0[117];
          float v1000_data = ir1[4];
          ir1[4] = (v1000_data + (v977_data * v998_data));
          float v1003_data = s0[141];
          float v1005_data = ir1[5];
          ir1[5] = (v1005_data + (v977_data * v1003_data));
          float v1008_data = s0[165];
          float v1010_data = ir1[6];
          ir1[6] = (v1010_data + (v977_data * v1008_data));
          float v1013_data = s0[189];
          float v1015_data = ir1[7];
          ir1[7] = (v1015_data + (v977_data * v1013_data));
          float v1018_data = s0[213];
          float v1020_data = ir1[8];
          ir1[8] = (v1020_data + (v977_data * v1018_data));
          float v1022_data = r0[22];
          float v1023_data = s0[22];
          float v1025_data = ir1[0];
          ir1[0] = (v1025_data + (v1022_data * v1023_data));
          float v1028_data = s0[46];
          float v1030_data = ir1[1];
          ir1[1] = (v1030_data + (v1022_data * v1028_data));
          float v1033_data = s0[70];
          float v1035_data = ir1[2];
          ir1[2] = (v1035_data + (v1022_data * v1033_data));
          float v1038_data = s0[94];
          float v1040_data = ir1[3];
          ir1[3] = (v1040_data + (v1022_data * v1038_data));
          float v1043_data = s0[118];
          float v1045_data = ir1[4];
          ir1[4] = (v1045_data + (v1022_data * v1043_data));
          float v1048_data = s0[142];
          float v1050_data = ir1[5];
          ir1[5] = (v1050_data + (v1022_data * v1048_data));
          float v1053_data = s0[166];
          float v1055_data = ir1[6];
          ir1[6] = (v1055_data + (v1022_data * v1053_data));
          float v1058_data = s0[190];
          float v1060_data = ir1[7];
          ir1[7] = (v1060_data + (v1022_data * v1058_data));
          float v1063_data = s0[214];
          float v1065_data = ir1[8];
          ir1[8] = (v1065_data + (v1022_data * v1063_data));
          float v1067_data = r0[23];
          float v1068_data = s0[23];
          float v1070_data = ir1[0];
          ir1[0] = (v1070_data + (v1067_data * v1068_data));
          float v1073_data = s0[47];
          float v1075_data = ir1[1];
          ir1[1] = (v1075_data + (v1067_data * v1073_data));
          float v1078_data = s0[71];
          float v1080_data = ir1[2];
          ir1[2] = (v1080_data + (v1067_data * v1078_data));
          float v1083_data = s0[95];
          float v1085_data = ir1[3];
          ir1[3] = (v1085_data + (v1067_data * v1083_data));
          float v1088_data = s0[119];
          float v1090_data = ir1[4];
          ir1[4] = (v1090_data + (v1067_data * v1088_data));
          float v1093_data = s0[143];
          float v1095_data = ir1[5];
          ir1[5] = (v1095_data + (v1067_data * v1093_data));
          float v1098_data = s0[167];
          float v1100_data = ir1[6];
          ir1[6] = (v1100_data + (v1067_data * v1098_data));
          float v1103_data = s0[191];
          float v1105_data = ir1[7];
          ir1[7] = (v1105_data + (v1067_data * v1103_data));
          float v1108_data = s0[215];
          float v1110_data = ir1[8];
          ir1[8] = (v1110_data + (v1067_data * v1108_data));
          // r1 = ir1
          if (v20_g) {
            #pragma unroll
            for (int32_t v1112_n1 = 0; v1112_n1 < 9; ++v1112_n1) {
              float v1114_data = ir1[v1112_n1];
              r1[v1112_n1] = v1114_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v20_g) {
            #pragma unroll
            for (int32_t v1115_i1 = 0; v1115_i1 < 9; ++v1115_i1) {
              float v1117_data = r1[v1115_i1];
              glb_m0[(v19_lead + (v1115_i1 * 24))] = v1117_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

