// === base name ===
kernel_b5ee1a7a5c2a4c25

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b5ee1a7a5c2a4c25 = {{16, 8, 1}, 16, 16, 1, 8, 5632, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b5ee1a7a5c2a4c25(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b5ee1a7a5c2a4c25(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b5ee1a7a5c2a4c25(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_b5ee1a7a5c2a4c25, block.x * block.y * block.z, 1408 * sizeof(float));
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
void launcher_kernel_b5ee1a7a5c2a4c25(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b5ee1a7a5c2a4c25(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_b5ee1a7a5c2a4c25, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_b5ee1a7a5c2a4c25<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_b5ee1a7a5c2a4c25(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 8 per block = block 16x8x1, 5632 B shared, occupancy grid
    // operands:
    //   m0 16×9(16×9) {0..16}×{0..9} strided
    //   m1 16×20(16×17) {0..16}×{0..17} none
    //   m2 20×9(17×9) {0..17}×{0..9} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1408}],"shared_bytes":5632,"shared_elements":1408,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,9]],"name":"m0","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[16,17]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[17,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[16,17]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[0,0],[17,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[176 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[160];
      const float *const __restrict__ glb_m1 = &m1[0];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v6_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v6_batchId0 < numElements0; v6_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v7_ahead1 = v6_batchId0 + (gridDim.x * blockDim.y);
        size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 144 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v6_batchId0 * 153 + 0 + m2_extraOffset];
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 9; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
          if (threadIdx.x < 9) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 144], &glb_m2[0 + 0 + 1 * threadIdx.x + 144], 4);
          }
          __pipeline_commit();
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r0[9]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // ir0 = +(glb_m1 * s0)
          // [(0, 16), (0, 9)] [(0, 17)]
          float ir0[9]{};
          int32_t v22_lead = threadIdx.x % 16;
          float v26_data = glb_m1[v22_lead];
          float v27_data = s0[0];
          float v29_data = ir0[0];
          ir0[0] = (v29_data + (v26_data * v27_data));
          float v32_data = s0[17];
          float v34_data = ir0[1];
          ir0[1] = (v34_data + (v26_data * v32_data));
          float v37_data = s0[34];
          float v39_data = ir0[2];
          ir0[2] = (v39_data + (v26_data * v37_data));
          float v42_data = s0[51];
          float v44_data = ir0[3];
          ir0[3] = (v44_data + (v26_data * v42_data));
          float v47_data = s0[68];
          float v49_data = ir0[4];
          ir0[4] = (v49_data + (v26_data * v47_data));
          float v52_data = s0[85];
          float v54_data = ir0[5];
          ir0[5] = (v54_data + (v26_data * v52_data));
          float v57_data = s0[102];
          float v59_data = ir0[6];
          ir0[6] = (v59_data + (v26_data * v57_data));
          float v62_data = s0[119];
          float v64_data = ir0[7];
          ir0[7] = (v64_data + (v26_data * v62_data));
          float v67_data = s0[136];
          float v69_data = ir0[8];
          ir0[8] = (v69_data + (v26_data * v67_data));
          float v72_data = glb_m1[(v22_lead + 16)];
          float v73_data = s0[1];
          float v75_data = ir0[0];
          ir0[0] = (v75_data + (v72_data * v73_data));
          float v78_data = s0[18];
          float v80_data = ir0[1];
          ir0[1] = (v80_data + (v72_data * v78_data));
          float v83_data = s0[35];
          float v85_data = ir0[2];
          ir0[2] = (v85_data + (v72_data * v83_data));
          float v88_data = s0[52];
          float v90_data = ir0[3];
          ir0[3] = (v90_data + (v72_data * v88_data));
          float v93_data = s0[69];
          float v95_data = ir0[4];
          ir0[4] = (v95_data + (v72_data * v93_data));
          float v98_data = s0[86];
          float v100_data = ir0[5];
          ir0[5] = (v100_data + (v72_data * v98_data));
          float v103_data = s0[103];
          float v105_data = ir0[6];
          ir0[6] = (v105_data + (v72_data * v103_data));
          float v108_data = s0[120];
          float v110_data = ir0[7];
          ir0[7] = (v110_data + (v72_data * v108_data));
          float v113_data = s0[137];
          float v115_data = ir0[8];
          ir0[8] = (v115_data + (v72_data * v113_data));
          float v118_data = glb_m1[(v22_lead + 32)];
          float v119_data = s0[2];
          float v121_data = ir0[0];
          ir0[0] = (v121_data + (v118_data * v119_data));
          float v124_data = s0[19];
          float v126_data = ir0[1];
          ir0[1] = (v126_data + (v118_data * v124_data));
          float v129_data = s0[36];
          float v131_data = ir0[2];
          ir0[2] = (v131_data + (v118_data * v129_data));
          float v134_data = s0[53];
          float v136_data = ir0[3];
          ir0[3] = (v136_data + (v118_data * v134_data));
          float v139_data = s0[70];
          float v141_data = ir0[4];
          ir0[4] = (v141_data + (v118_data * v139_data));
          float v144_data = s0[87];
          float v146_data = ir0[5];
          ir0[5] = (v146_data + (v118_data * v144_data));
          float v149_data = s0[104];
          float v151_data = ir0[6];
          ir0[6] = (v151_data + (v118_data * v149_data));
          float v154_data = s0[121];
          float v156_data = ir0[7];
          ir0[7] = (v156_data + (v118_data * v154_data));
          float v159_data = s0[138];
          float v161_data = ir0[8];
          ir0[8] = (v161_data + (v118_data * v159_data));
          float v164_data = glb_m1[(v22_lead + 48)];
          float v165_data = s0[3];
          float v167_data = ir0[0];
          ir0[0] = (v167_data + (v164_data * v165_data));
          float v170_data = s0[20];
          float v172_data = ir0[1];
          ir0[1] = (v172_data + (v164_data * v170_data));
          float v175_data = s0[37];
          float v177_data = ir0[2];
          ir0[2] = (v177_data + (v164_data * v175_data));
          float v180_data = s0[54];
          float v182_data = ir0[3];
          ir0[3] = (v182_data + (v164_data * v180_data));
          float v185_data = s0[71];
          float v187_data = ir0[4];
          ir0[4] = (v187_data + (v164_data * v185_data));
          float v190_data = s0[88];
          float v192_data = ir0[5];
          ir0[5] = (v192_data + (v164_data * v190_data));
          float v195_data = s0[105];
          float v197_data = ir0[6];
          ir0[6] = (v197_data + (v164_data * v195_data));
          float v200_data = s0[122];
          float v202_data = ir0[7];
          ir0[7] = (v202_data + (v164_data * v200_data));
          float v205_data = s0[139];
          float v207_data = ir0[8];
          ir0[8] = (v207_data + (v164_data * v205_data));
          float v210_data = glb_m1[(v22_lead + 64)];
          float v211_data = s0[4];
          float v213_data = ir0[0];
          ir0[0] = (v213_data + (v210_data * v211_data));
          float v216_data = s0[21];
          float v218_data = ir0[1];
          ir0[1] = (v218_data + (v210_data * v216_data));
          float v221_data = s0[38];
          float v223_data = ir0[2];
          ir0[2] = (v223_data + (v210_data * v221_data));
          float v226_data = s0[55];
          float v228_data = ir0[3];
          ir0[3] = (v228_data + (v210_data * v226_data));
          float v231_data = s0[72];
          float v233_data = ir0[4];
          ir0[4] = (v233_data + (v210_data * v231_data));
          float v236_data = s0[89];
          float v238_data = ir0[5];
          ir0[5] = (v238_data + (v210_data * v236_data));
          float v241_data = s0[106];
          float v243_data = ir0[6];
          ir0[6] = (v243_data + (v210_data * v241_data));
          float v246_data = s0[123];
          float v248_data = ir0[7];
          ir0[7] = (v248_data + (v210_data * v246_data));
          float v251_data = s0[140];
          float v253_data = ir0[8];
          ir0[8] = (v253_data + (v210_data * v251_data));
          float v256_data = glb_m1[(v22_lead + 80)];
          float v257_data = s0[5];
          float v259_data = ir0[0];
          ir0[0] = (v259_data + (v256_data * v257_data));
          float v262_data = s0[22];
          float v264_data = ir0[1];
          ir0[1] = (v264_data + (v256_data * v262_data));
          float v267_data = s0[39];
          float v269_data = ir0[2];
          ir0[2] = (v269_data + (v256_data * v267_data));
          float v272_data = s0[56];
          float v274_data = ir0[3];
          ir0[3] = (v274_data + (v256_data * v272_data));
          float v277_data = s0[73];
          float v279_data = ir0[4];
          ir0[4] = (v279_data + (v256_data * v277_data));
          float v282_data = s0[90];
          float v284_data = ir0[5];
          ir0[5] = (v284_data + (v256_data * v282_data));
          float v287_data = s0[107];
          float v289_data = ir0[6];
          ir0[6] = (v289_data + (v256_data * v287_data));
          float v292_data = s0[124];
          float v294_data = ir0[7];
          ir0[7] = (v294_data + (v256_data * v292_data));
          float v297_data = s0[141];
          float v299_data = ir0[8];
          ir0[8] = (v299_data + (v256_data * v297_data));
          float v302_data = glb_m1[(v22_lead + 96)];
          float v303_data = s0[6];
          float v305_data = ir0[0];
          ir0[0] = (v305_data + (v302_data * v303_data));
          float v308_data = s0[23];
          float v310_data = ir0[1];
          ir0[1] = (v310_data + (v302_data * v308_data));
          float v313_data = s0[40];
          float v315_data = ir0[2];
          ir0[2] = (v315_data + (v302_data * v313_data));
          float v318_data = s0[57];
          float v320_data = ir0[3];
          ir0[3] = (v320_data + (v302_data * v318_data));
          float v323_data = s0[74];
          float v325_data = ir0[4];
          ir0[4] = (v325_data + (v302_data * v323_data));
          float v328_data = s0[91];
          float v330_data = ir0[5];
          ir0[5] = (v330_data + (v302_data * v328_data));
          float v333_data = s0[108];
          float v335_data = ir0[6];
          ir0[6] = (v335_data + (v302_data * v333_data));
          float v338_data = s0[125];
          float v340_data = ir0[7];
          ir0[7] = (v340_data + (v302_data * v338_data));
          float v343_data = s0[142];
          float v345_data = ir0[8];
          ir0[8] = (v345_data + (v302_data * v343_data));
          float v348_data = glb_m1[(v22_lead + 112)];
          float v349_data = s0[7];
          float v351_data = ir0[0];
          ir0[0] = (v351_data + (v348_data * v349_data));
          float v354_data = s0[24];
          float v356_data = ir0[1];
          ir0[1] = (v356_data + (v348_data * v354_data));
          float v359_data = s0[41];
          float v361_data = ir0[2];
          ir0[2] = (v361_data + (v348_data * v359_data));
          float v364_data = s0[58];
          float v366_data = ir0[3];
          ir0[3] = (v366_data + (v348_data * v364_data));
          float v369_data = s0[75];
          float v371_data = ir0[4];
          ir0[4] = (v371_data + (v348_data * v369_data));
          float v374_data = s0[92];
          float v376_data = ir0[5];
          ir0[5] = (v376_data + (v348_data * v374_data));
          float v379_data = s0[109];
          float v381_data = ir0[6];
          ir0[6] = (v381_data + (v348_data * v379_data));
          float v384_data = s0[126];
          float v386_data = ir0[7];
          ir0[7] = (v386_data + (v348_data * v384_data));
          float v389_data = s0[143];
          float v391_data = ir0[8];
          ir0[8] = (v391_data + (v348_data * v389_data));
          float v394_data = glb_m1[(v22_lead + 128)];
          float v395_data = s0[8];
          float v397_data = ir0[0];
          ir0[0] = (v397_data + (v394_data * v395_data));
          float v400_data = s0[25];
          float v402_data = ir0[1];
          ir0[1] = (v402_data + (v394_data * v400_data));
          float v405_data = s0[42];
          float v407_data = ir0[2];
          ir0[2] = (v407_data + (v394_data * v405_data));
          float v410_data = s0[59];
          float v412_data = ir0[3];
          ir0[3] = (v412_data + (v394_data * v410_data));
          float v415_data = s0[76];
          float v417_data = ir0[4];
          ir0[4] = (v417_data + (v394_data * v415_data));
          float v420_data = s0[93];
          float v422_data = ir0[5];
          ir0[5] = (v422_data + (v394_data * v420_data));
          float v425_data = s0[110];
          float v427_data = ir0[6];
          ir0[6] = (v427_data + (v394_data * v425_data));
          float v430_data = s0[127];
          float v432_data = ir0[7];
          ir0[7] = (v432_data + (v394_data * v430_data));
          float v435_data = s0[144];
          float v437_data = ir0[8];
          ir0[8] = (v437_data + (v394_data * v435_data));
          float v440_data = glb_m1[(v22_lead + 144)];
          float v441_data = s0[9];
          float v443_data = ir0[0];
          ir0[0] = (v443_data + (v440_data * v441_data));
          float v446_data = s0[26];
          float v448_data = ir0[1];
          ir0[1] = (v448_data + (v440_data * v446_data));
          float v451_data = s0[43];
          float v453_data = ir0[2];
          ir0[2] = (v453_data + (v440_data * v451_data));
          float v456_data = s0[60];
          float v458_data = ir0[3];
          ir0[3] = (v458_data + (v440_data * v456_data));
          float v461_data = s0[77];
          float v463_data = ir0[4];
          ir0[4] = (v463_data + (v440_data * v461_data));
          float v466_data = s0[94];
          float v468_data = ir0[5];
          ir0[5] = (v468_data + (v440_data * v466_data));
          float v471_data = s0[111];
          float v473_data = ir0[6];
          ir0[6] = (v473_data + (v440_data * v471_data));
          float v476_data = s0[128];
          float v478_data = ir0[7];
          ir0[7] = (v478_data + (v440_data * v476_data));
          float v481_data = s0[145];
          float v483_data = ir0[8];
          ir0[8] = (v483_data + (v440_data * v481_data));
          float v486_data = glb_m1[(v22_lead + 160)];
          float v487_data = s0[10];
          float v489_data = ir0[0];
          ir0[0] = (v489_data + (v486_data * v487_data));
          float v492_data = s0[27];
          float v494_data = ir0[1];
          ir0[1] = (v494_data + (v486_data * v492_data));
          float v497_data = s0[44];
          float v499_data = ir0[2];
          ir0[2] = (v499_data + (v486_data * v497_data));
          float v502_data = s0[61];
          float v504_data = ir0[3];
          ir0[3] = (v504_data + (v486_data * v502_data));
          float v507_data = s0[78];
          float v509_data = ir0[4];
          ir0[4] = (v509_data + (v486_data * v507_data));
          float v512_data = s0[95];
          float v514_data = ir0[5];
          ir0[5] = (v514_data + (v486_data * v512_data));
          float v517_data = s0[112];
          float v519_data = ir0[6];
          ir0[6] = (v519_data + (v486_data * v517_data));
          float v522_data = s0[129];
          float v524_data = ir0[7];
          ir0[7] = (v524_data + (v486_data * v522_data));
          float v527_data = s0[146];
          float v529_data = ir0[8];
          ir0[8] = (v529_data + (v486_data * v527_data));
          float v532_data = glb_m1[(v22_lead + 176)];
          float v533_data = s0[11];
          float v535_data = ir0[0];
          ir0[0] = (v535_data + (v532_data * v533_data));
          float v538_data = s0[28];
          float v540_data = ir0[1];
          ir0[1] = (v540_data + (v532_data * v538_data));
          float v543_data = s0[45];
          float v545_data = ir0[2];
          ir0[2] = (v545_data + (v532_data * v543_data));
          float v548_data = s0[62];
          float v550_data = ir0[3];
          ir0[3] = (v550_data + (v532_data * v548_data));
          float v553_data = s0[79];
          float v555_data = ir0[4];
          ir0[4] = (v555_data + (v532_data * v553_data));
          float v558_data = s0[96];
          float v560_data = ir0[5];
          ir0[5] = (v560_data + (v532_data * v558_data));
          float v563_data = s0[113];
          float v565_data = ir0[6];
          ir0[6] = (v565_data + (v532_data * v563_data));
          float v568_data = s0[130];
          float v570_data = ir0[7];
          ir0[7] = (v570_data + (v532_data * v568_data));
          float v573_data = s0[147];
          float v575_data = ir0[8];
          ir0[8] = (v575_data + (v532_data * v573_data));
          float v578_data = glb_m1[(v22_lead + 192)];
          float v579_data = s0[12];
          float v581_data = ir0[0];
          ir0[0] = (v581_data + (v578_data * v579_data));
          float v584_data = s0[29];
          float v586_data = ir0[1];
          ir0[1] = (v586_data + (v578_data * v584_data));
          float v589_data = s0[46];
          float v591_data = ir0[2];
          ir0[2] = (v591_data + (v578_data * v589_data));
          float v594_data = s0[63];
          float v596_data = ir0[3];
          ir0[3] = (v596_data + (v578_data * v594_data));
          float v599_data = s0[80];
          float v601_data = ir0[4];
          ir0[4] = (v601_data + (v578_data * v599_data));
          float v604_data = s0[97];
          float v606_data = ir0[5];
          ir0[5] = (v606_data + (v578_data * v604_data));
          float v609_data = s0[114];
          float v611_data = ir0[6];
          ir0[6] = (v611_data + (v578_data * v609_data));
          float v614_data = s0[131];
          float v616_data = ir0[7];
          ir0[7] = (v616_data + (v578_data * v614_data));
          float v619_data = s0[148];
          float v621_data = ir0[8];
          ir0[8] = (v621_data + (v578_data * v619_data));
          float v624_data = glb_m1[(v22_lead + 208)];
          float v625_data = s0[13];
          float v627_data = ir0[0];
          ir0[0] = (v627_data + (v624_data * v625_data));
          float v630_data = s0[30];
          float v632_data = ir0[1];
          ir0[1] = (v632_data + (v624_data * v630_data));
          float v635_data = s0[47];
          float v637_data = ir0[2];
          ir0[2] = (v637_data + (v624_data * v635_data));
          float v640_data = s0[64];
          float v642_data = ir0[3];
          ir0[3] = (v642_data + (v624_data * v640_data));
          float v645_data = s0[81];
          float v647_data = ir0[4];
          ir0[4] = (v647_data + (v624_data * v645_data));
          float v650_data = s0[98];
          float v652_data = ir0[5];
          ir0[5] = (v652_data + (v624_data * v650_data));
          float v655_data = s0[115];
          float v657_data = ir0[6];
          ir0[6] = (v657_data + (v624_data * v655_data));
          float v660_data = s0[132];
          float v662_data = ir0[7];
          ir0[7] = (v662_data + (v624_data * v660_data));
          float v665_data = s0[149];
          float v667_data = ir0[8];
          ir0[8] = (v667_data + (v624_data * v665_data));
          float v670_data = glb_m1[(v22_lead + 224)];
          float v671_data = s0[14];
          float v673_data = ir0[0];
          ir0[0] = (v673_data + (v670_data * v671_data));
          float v676_data = s0[31];
          float v678_data = ir0[1];
          ir0[1] = (v678_data + (v670_data * v676_data));
          float v681_data = s0[48];
          float v683_data = ir0[2];
          ir0[2] = (v683_data + (v670_data * v681_data));
          float v686_data = s0[65];
          float v688_data = ir0[3];
          ir0[3] = (v688_data + (v670_data * v686_data));
          float v691_data = s0[82];
          float v693_data = ir0[4];
          ir0[4] = (v693_data + (v670_data * v691_data));
          float v696_data = s0[99];
          float v698_data = ir0[5];
          ir0[5] = (v698_data + (v670_data * v696_data));
          float v701_data = s0[116];
          float v703_data = ir0[6];
          ir0[6] = (v703_data + (v670_data * v701_data));
          float v706_data = s0[133];
          float v708_data = ir0[7];
          ir0[7] = (v708_data + (v670_data * v706_data));
          float v711_data = s0[150];
          float v713_data = ir0[8];
          ir0[8] = (v713_data + (v670_data * v711_data));
          float v716_data = glb_m1[(v22_lead + 240)];
          float v717_data = s0[15];
          float v719_data = ir0[0];
          ir0[0] = (v719_data + (v716_data * v717_data));
          float v722_data = s0[32];
          float v724_data = ir0[1];
          ir0[1] = (v724_data + (v716_data * v722_data));
          float v727_data = s0[49];
          float v729_data = ir0[2];
          ir0[2] = (v729_data + (v716_data * v727_data));
          float v732_data = s0[66];
          float v734_data = ir0[3];
          ir0[3] = (v734_data + (v716_data * v732_data));
          float v737_data = s0[83];
          float v739_data = ir0[4];
          ir0[4] = (v739_data + (v716_data * v737_data));
          float v742_data = s0[100];
          float v744_data = ir0[5];
          ir0[5] = (v744_data + (v716_data * v742_data));
          float v747_data = s0[117];
          float v749_data = ir0[6];
          ir0[6] = (v749_data + (v716_data * v747_data));
          float v752_data = s0[134];
          float v754_data = ir0[7];
          ir0[7] = (v754_data + (v716_data * v752_data));
          float v757_data = s0[151];
          float v759_data = ir0[8];
          ir0[8] = (v759_data + (v716_data * v757_data));
          float v762_data = glb_m1[(v22_lead + 256)];
          float v763_data = s0[16];
          float v765_data = ir0[0];
          ir0[0] = (v765_data + (v762_data * v763_data));
          float v768_data = s0[33];
          float v770_data = ir0[1];
          ir0[1] = (v770_data + (v762_data * v768_data));
          float v773_data = s0[50];
          float v775_data = ir0[2];
          ir0[2] = (v775_data + (v762_data * v773_data));
          float v778_data = s0[67];
          float v780_data = ir0[3];
          ir0[3] = (v780_data + (v762_data * v778_data));
          float v783_data = s0[84];
          float v785_data = ir0[4];
          ir0[4] = (v785_data + (v762_data * v783_data));
          float v788_data = s0[101];
          float v790_data = ir0[5];
          ir0[5] = (v790_data + (v762_data * v788_data));
          float v793_data = s0[118];
          float v795_data = ir0[6];
          ir0[6] = (v795_data + (v762_data * v793_data));
          float v798_data = s0[135];
          float v800_data = ir0[7];
          ir0[7] = (v800_data + (v762_data * v798_data));
          float v803_data = s0[152];
          float v805_data = ir0[8];
          ir0[8] = (v805_data + (v762_data * v803_data));
          // r0 = ir0
          #pragma unroll
          for (int32_t v810_n0 = 0; v810_n0 < 1; ++v810_n0) {
            #pragma unroll
            for (int32_t v811_n1 = 0; v811_n1 < 9; ++v811_n1) {
              int32_t v812_a = v810_n0 + v811_n1;
              float v813_data = ir0[v812_a];
              r0[v812_a] = v813_data;
            }
          }
          // glb_m0 = store{r>g}(r0);
          #pragma unroll
          for (int32_t v817_i0 = 0; v817_i0 < 1; ++v817_i0) {
            int32_t v822_lead = v22_lead + (v817_i0 * 16);
            #pragma unroll
            for (int32_t v818_i1 = 0; v818_i1 < 9; ++v818_i1) {
              float v820_data = r0[(v817_i0 + v818_i1)];
              glb_m0[(v822_lead + (v818_i1 * 16))] = v820_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

