// === base name ===
kernel_3384bdf8bc6ea781

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_3384bdf8bc6ea781 = {{16, 8, 1}, 16, 16, 1, 8, 5632, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_3384bdf8bc6ea781(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_3384bdf8bc6ea781(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_3384bdf8bc6ea781(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_3384bdf8bc6ea781, block.x * block.y * block.z, 1408 * sizeof(float));
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
void launcher_kernel_3384bdf8bc6ea781(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_3384bdf8bc6ea781(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_3384bdf8bc6ea781, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_3384bdf8bc6ea781<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_3384bdf8bc6ea781(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 8 per block = block 16x8x1, 5632 B shared, occupancy grid
    // operands:
    //   m0 16×9(16×9) {0..16}×{0..9} strided
    //   m1 16×20(16×17) {0..16}×{1..18} none
    //   m2 20×9(17×9) {1..18}×{0..9} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1408}],"shared_bytes":5632,"shared_elements":1408,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,9]],"name":"m0","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,1],[16,18]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[1,0],[18,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[16,18]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[18,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[176 * threadIdx.y + 0];
      const float *const __restrict__ glb_m1 = &m1[0];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < numElements0; v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 144 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v9_batchId0 * 153 + 0 + m2_extraOffset];
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
          // ir0 = +(glb_m1 * s0)
          // [(0, 16), (0, 9)] [(1, 18)]
          float ir0[9]{};
          int32_t v25_lead = threadIdx.x % 16;
          float v29_data = glb_m1[v25_lead];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v30_data = s0[0];
          float v32_data = ir0[0];
          ir0[0] = (v32_data + (v29_data * v30_data));
          float v35_data = s0[17];
          float v37_data = ir0[1];
          ir0[1] = (v37_data + (v29_data * v35_data));
          float v40_data = s0[34];
          float v42_data = ir0[2];
          ir0[2] = (v42_data + (v29_data * v40_data));
          float v45_data = s0[51];
          float v47_data = ir0[3];
          ir0[3] = (v47_data + (v29_data * v45_data));
          float v50_data = s0[68];
          float v52_data = ir0[4];
          ir0[4] = (v52_data + (v29_data * v50_data));
          float v55_data = s0[85];
          float v57_data = ir0[5];
          ir0[5] = (v57_data + (v29_data * v55_data));
          float v60_data = s0[102];
          float v62_data = ir0[6];
          ir0[6] = (v62_data + (v29_data * v60_data));
          float v65_data = s0[119];
          float v67_data = ir0[7];
          ir0[7] = (v67_data + (v29_data * v65_data));
          float v70_data = s0[136];
          float v72_data = ir0[8];
          ir0[8] = (v72_data + (v29_data * v70_data));
          float v75_data = glb_m1[(v25_lead + 16)];
          float v76_data = s0[1];
          float v78_data = ir0[0];
          ir0[0] = (v78_data + (v75_data * v76_data));
          float v81_data = s0[18];
          float v83_data = ir0[1];
          ir0[1] = (v83_data + (v75_data * v81_data));
          float v86_data = s0[35];
          float v88_data = ir0[2];
          ir0[2] = (v88_data + (v75_data * v86_data));
          float v91_data = s0[52];
          float v93_data = ir0[3];
          ir0[3] = (v93_data + (v75_data * v91_data));
          float v96_data = s0[69];
          float v98_data = ir0[4];
          ir0[4] = (v98_data + (v75_data * v96_data));
          float v101_data = s0[86];
          float v103_data = ir0[5];
          ir0[5] = (v103_data + (v75_data * v101_data));
          float v106_data = s0[103];
          float v108_data = ir0[6];
          ir0[6] = (v108_data + (v75_data * v106_data));
          float v111_data = s0[120];
          float v113_data = ir0[7];
          ir0[7] = (v113_data + (v75_data * v111_data));
          float v116_data = s0[137];
          float v118_data = ir0[8];
          ir0[8] = (v118_data + (v75_data * v116_data));
          float v121_data = glb_m1[(v25_lead + 32)];
          float v122_data = s0[2];
          float v124_data = ir0[0];
          ir0[0] = (v124_data + (v121_data * v122_data));
          float v127_data = s0[19];
          float v129_data = ir0[1];
          ir0[1] = (v129_data + (v121_data * v127_data));
          float v132_data = s0[36];
          float v134_data = ir0[2];
          ir0[2] = (v134_data + (v121_data * v132_data));
          float v137_data = s0[53];
          float v139_data = ir0[3];
          ir0[3] = (v139_data + (v121_data * v137_data));
          float v142_data = s0[70];
          float v144_data = ir0[4];
          ir0[4] = (v144_data + (v121_data * v142_data));
          float v147_data = s0[87];
          float v149_data = ir0[5];
          ir0[5] = (v149_data + (v121_data * v147_data));
          float v152_data = s0[104];
          float v154_data = ir0[6];
          ir0[6] = (v154_data + (v121_data * v152_data));
          float v157_data = s0[121];
          float v159_data = ir0[7];
          ir0[7] = (v159_data + (v121_data * v157_data));
          float v162_data = s0[138];
          float v164_data = ir0[8];
          ir0[8] = (v164_data + (v121_data * v162_data));
          float v167_data = glb_m1[(v25_lead + 48)];
          float v168_data = s0[3];
          float v170_data = ir0[0];
          ir0[0] = (v170_data + (v167_data * v168_data));
          float v173_data = s0[20];
          float v175_data = ir0[1];
          ir0[1] = (v175_data + (v167_data * v173_data));
          float v178_data = s0[37];
          float v180_data = ir0[2];
          ir0[2] = (v180_data + (v167_data * v178_data));
          float v183_data = s0[54];
          float v185_data = ir0[3];
          ir0[3] = (v185_data + (v167_data * v183_data));
          float v188_data = s0[71];
          float v190_data = ir0[4];
          ir0[4] = (v190_data + (v167_data * v188_data));
          float v193_data = s0[88];
          float v195_data = ir0[5];
          ir0[5] = (v195_data + (v167_data * v193_data));
          float v198_data = s0[105];
          float v200_data = ir0[6];
          ir0[6] = (v200_data + (v167_data * v198_data));
          float v203_data = s0[122];
          float v205_data = ir0[7];
          ir0[7] = (v205_data + (v167_data * v203_data));
          float v208_data = s0[139];
          float v210_data = ir0[8];
          ir0[8] = (v210_data + (v167_data * v208_data));
          float v213_data = glb_m1[(v25_lead + 64)];
          float v214_data = s0[4];
          float v216_data = ir0[0];
          ir0[0] = (v216_data + (v213_data * v214_data));
          float v219_data = s0[21];
          float v221_data = ir0[1];
          ir0[1] = (v221_data + (v213_data * v219_data));
          float v224_data = s0[38];
          float v226_data = ir0[2];
          ir0[2] = (v226_data + (v213_data * v224_data));
          float v229_data = s0[55];
          float v231_data = ir0[3];
          ir0[3] = (v231_data + (v213_data * v229_data));
          float v234_data = s0[72];
          float v236_data = ir0[4];
          ir0[4] = (v236_data + (v213_data * v234_data));
          float v239_data = s0[89];
          float v241_data = ir0[5];
          ir0[5] = (v241_data + (v213_data * v239_data));
          float v244_data = s0[106];
          float v246_data = ir0[6];
          ir0[6] = (v246_data + (v213_data * v244_data));
          float v249_data = s0[123];
          float v251_data = ir0[7];
          ir0[7] = (v251_data + (v213_data * v249_data));
          float v254_data = s0[140];
          float v256_data = ir0[8];
          ir0[8] = (v256_data + (v213_data * v254_data));
          float v259_data = glb_m1[(v25_lead + 80)];
          float v260_data = s0[5];
          float v262_data = ir0[0];
          ir0[0] = (v262_data + (v259_data * v260_data));
          float v265_data = s0[22];
          float v267_data = ir0[1];
          ir0[1] = (v267_data + (v259_data * v265_data));
          float v270_data = s0[39];
          float v272_data = ir0[2];
          ir0[2] = (v272_data + (v259_data * v270_data));
          float v275_data = s0[56];
          float v277_data = ir0[3];
          ir0[3] = (v277_data + (v259_data * v275_data));
          float v280_data = s0[73];
          float v282_data = ir0[4];
          ir0[4] = (v282_data + (v259_data * v280_data));
          float v285_data = s0[90];
          float v287_data = ir0[5];
          ir0[5] = (v287_data + (v259_data * v285_data));
          float v290_data = s0[107];
          float v292_data = ir0[6];
          ir0[6] = (v292_data + (v259_data * v290_data));
          float v295_data = s0[124];
          float v297_data = ir0[7];
          ir0[7] = (v297_data + (v259_data * v295_data));
          float v300_data = s0[141];
          float v302_data = ir0[8];
          ir0[8] = (v302_data + (v259_data * v300_data));
          float v305_data = glb_m1[(v25_lead + 96)];
          float v306_data = s0[6];
          float v308_data = ir0[0];
          ir0[0] = (v308_data + (v305_data * v306_data));
          float v311_data = s0[23];
          float v313_data = ir0[1];
          ir0[1] = (v313_data + (v305_data * v311_data));
          float v316_data = s0[40];
          float v318_data = ir0[2];
          ir0[2] = (v318_data + (v305_data * v316_data));
          float v321_data = s0[57];
          float v323_data = ir0[3];
          ir0[3] = (v323_data + (v305_data * v321_data));
          float v326_data = s0[74];
          float v328_data = ir0[4];
          ir0[4] = (v328_data + (v305_data * v326_data));
          float v331_data = s0[91];
          float v333_data = ir0[5];
          ir0[5] = (v333_data + (v305_data * v331_data));
          float v336_data = s0[108];
          float v338_data = ir0[6];
          ir0[6] = (v338_data + (v305_data * v336_data));
          float v341_data = s0[125];
          float v343_data = ir0[7];
          ir0[7] = (v343_data + (v305_data * v341_data));
          float v346_data = s0[142];
          float v348_data = ir0[8];
          ir0[8] = (v348_data + (v305_data * v346_data));
          float v351_data = glb_m1[(v25_lead + 112)];
          float v352_data = s0[7];
          float v354_data = ir0[0];
          ir0[0] = (v354_data + (v351_data * v352_data));
          float v357_data = s0[24];
          float v359_data = ir0[1];
          ir0[1] = (v359_data + (v351_data * v357_data));
          float v362_data = s0[41];
          float v364_data = ir0[2];
          ir0[2] = (v364_data + (v351_data * v362_data));
          float v367_data = s0[58];
          float v369_data = ir0[3];
          ir0[3] = (v369_data + (v351_data * v367_data));
          float v372_data = s0[75];
          float v374_data = ir0[4];
          ir0[4] = (v374_data + (v351_data * v372_data));
          float v377_data = s0[92];
          float v379_data = ir0[5];
          ir0[5] = (v379_data + (v351_data * v377_data));
          float v382_data = s0[109];
          float v384_data = ir0[6];
          ir0[6] = (v384_data + (v351_data * v382_data));
          float v387_data = s0[126];
          float v389_data = ir0[7];
          ir0[7] = (v389_data + (v351_data * v387_data));
          float v392_data = s0[143];
          float v394_data = ir0[8];
          ir0[8] = (v394_data + (v351_data * v392_data));
          float v397_data = glb_m1[(v25_lead + 128)];
          float v398_data = s0[8];
          float v400_data = ir0[0];
          ir0[0] = (v400_data + (v397_data * v398_data));
          float v403_data = s0[25];
          float v405_data = ir0[1];
          ir0[1] = (v405_data + (v397_data * v403_data));
          float v408_data = s0[42];
          float v410_data = ir0[2];
          ir0[2] = (v410_data + (v397_data * v408_data));
          float v413_data = s0[59];
          float v415_data = ir0[3];
          ir0[3] = (v415_data + (v397_data * v413_data));
          float v418_data = s0[76];
          float v420_data = ir0[4];
          ir0[4] = (v420_data + (v397_data * v418_data));
          float v423_data = s0[93];
          float v425_data = ir0[5];
          ir0[5] = (v425_data + (v397_data * v423_data));
          float v428_data = s0[110];
          float v430_data = ir0[6];
          ir0[6] = (v430_data + (v397_data * v428_data));
          float v433_data = s0[127];
          float v435_data = ir0[7];
          ir0[7] = (v435_data + (v397_data * v433_data));
          float v438_data = s0[144];
          float v440_data = ir0[8];
          ir0[8] = (v440_data + (v397_data * v438_data));
          float v443_data = glb_m1[(v25_lead + 144)];
          float v444_data = s0[9];
          float v446_data = ir0[0];
          ir0[0] = (v446_data + (v443_data * v444_data));
          float v449_data = s0[26];
          float v451_data = ir0[1];
          ir0[1] = (v451_data + (v443_data * v449_data));
          float v454_data = s0[43];
          float v456_data = ir0[2];
          ir0[2] = (v456_data + (v443_data * v454_data));
          float v459_data = s0[60];
          float v461_data = ir0[3];
          ir0[3] = (v461_data + (v443_data * v459_data));
          float v464_data = s0[77];
          float v466_data = ir0[4];
          ir0[4] = (v466_data + (v443_data * v464_data));
          float v469_data = s0[94];
          float v471_data = ir0[5];
          ir0[5] = (v471_data + (v443_data * v469_data));
          float v474_data = s0[111];
          float v476_data = ir0[6];
          ir0[6] = (v476_data + (v443_data * v474_data));
          float v479_data = s0[128];
          float v481_data = ir0[7];
          ir0[7] = (v481_data + (v443_data * v479_data));
          float v484_data = s0[145];
          float v486_data = ir0[8];
          ir0[8] = (v486_data + (v443_data * v484_data));
          float v489_data = glb_m1[(v25_lead + 160)];
          float v490_data = s0[10];
          float v492_data = ir0[0];
          ir0[0] = (v492_data + (v489_data * v490_data));
          float v495_data = s0[27];
          float v497_data = ir0[1];
          ir0[1] = (v497_data + (v489_data * v495_data));
          float v500_data = s0[44];
          float v502_data = ir0[2];
          ir0[2] = (v502_data + (v489_data * v500_data));
          float v505_data = s0[61];
          float v507_data = ir0[3];
          ir0[3] = (v507_data + (v489_data * v505_data));
          float v510_data = s0[78];
          float v512_data = ir0[4];
          ir0[4] = (v512_data + (v489_data * v510_data));
          float v515_data = s0[95];
          float v517_data = ir0[5];
          ir0[5] = (v517_data + (v489_data * v515_data));
          float v520_data = s0[112];
          float v522_data = ir0[6];
          ir0[6] = (v522_data + (v489_data * v520_data));
          float v525_data = s0[129];
          float v527_data = ir0[7];
          ir0[7] = (v527_data + (v489_data * v525_data));
          float v530_data = s0[146];
          float v532_data = ir0[8];
          ir0[8] = (v532_data + (v489_data * v530_data));
          float v535_data = glb_m1[(v25_lead + 176)];
          float v536_data = s0[11];
          float v538_data = ir0[0];
          ir0[0] = (v538_data + (v535_data * v536_data));
          float v541_data = s0[28];
          float v543_data = ir0[1];
          ir0[1] = (v543_data + (v535_data * v541_data));
          float v546_data = s0[45];
          float v548_data = ir0[2];
          ir0[2] = (v548_data + (v535_data * v546_data));
          float v551_data = s0[62];
          float v553_data = ir0[3];
          ir0[3] = (v553_data + (v535_data * v551_data));
          float v556_data = s0[79];
          float v558_data = ir0[4];
          ir0[4] = (v558_data + (v535_data * v556_data));
          float v561_data = s0[96];
          float v563_data = ir0[5];
          ir0[5] = (v563_data + (v535_data * v561_data));
          float v566_data = s0[113];
          float v568_data = ir0[6];
          ir0[6] = (v568_data + (v535_data * v566_data));
          float v571_data = s0[130];
          float v573_data = ir0[7];
          ir0[7] = (v573_data + (v535_data * v571_data));
          float v576_data = s0[147];
          float v578_data = ir0[8];
          ir0[8] = (v578_data + (v535_data * v576_data));
          float v581_data = glb_m1[(v25_lead + 192)];
          float v582_data = s0[12];
          float v584_data = ir0[0];
          ir0[0] = (v584_data + (v581_data * v582_data));
          float v587_data = s0[29];
          float v589_data = ir0[1];
          ir0[1] = (v589_data + (v581_data * v587_data));
          float v592_data = s0[46];
          float v594_data = ir0[2];
          ir0[2] = (v594_data + (v581_data * v592_data));
          float v597_data = s0[63];
          float v599_data = ir0[3];
          ir0[3] = (v599_data + (v581_data * v597_data));
          float v602_data = s0[80];
          float v604_data = ir0[4];
          ir0[4] = (v604_data + (v581_data * v602_data));
          float v607_data = s0[97];
          float v609_data = ir0[5];
          ir0[5] = (v609_data + (v581_data * v607_data));
          float v612_data = s0[114];
          float v614_data = ir0[6];
          ir0[6] = (v614_data + (v581_data * v612_data));
          float v617_data = s0[131];
          float v619_data = ir0[7];
          ir0[7] = (v619_data + (v581_data * v617_data));
          float v622_data = s0[148];
          float v624_data = ir0[8];
          ir0[8] = (v624_data + (v581_data * v622_data));
          float v627_data = glb_m1[(v25_lead + 208)];
          float v628_data = s0[13];
          float v630_data = ir0[0];
          ir0[0] = (v630_data + (v627_data * v628_data));
          float v633_data = s0[30];
          float v635_data = ir0[1];
          ir0[1] = (v635_data + (v627_data * v633_data));
          float v638_data = s0[47];
          float v640_data = ir0[2];
          ir0[2] = (v640_data + (v627_data * v638_data));
          float v643_data = s0[64];
          float v645_data = ir0[3];
          ir0[3] = (v645_data + (v627_data * v643_data));
          float v648_data = s0[81];
          float v650_data = ir0[4];
          ir0[4] = (v650_data + (v627_data * v648_data));
          float v653_data = s0[98];
          float v655_data = ir0[5];
          ir0[5] = (v655_data + (v627_data * v653_data));
          float v658_data = s0[115];
          float v660_data = ir0[6];
          ir0[6] = (v660_data + (v627_data * v658_data));
          float v663_data = s0[132];
          float v665_data = ir0[7];
          ir0[7] = (v665_data + (v627_data * v663_data));
          float v668_data = s0[149];
          float v670_data = ir0[8];
          ir0[8] = (v670_data + (v627_data * v668_data));
          float v673_data = glb_m1[(v25_lead + 224)];
          float v674_data = s0[14];
          float v676_data = ir0[0];
          ir0[0] = (v676_data + (v673_data * v674_data));
          float v679_data = s0[31];
          float v681_data = ir0[1];
          ir0[1] = (v681_data + (v673_data * v679_data));
          float v684_data = s0[48];
          float v686_data = ir0[2];
          ir0[2] = (v686_data + (v673_data * v684_data));
          float v689_data = s0[65];
          float v691_data = ir0[3];
          ir0[3] = (v691_data + (v673_data * v689_data));
          float v694_data = s0[82];
          float v696_data = ir0[4];
          ir0[4] = (v696_data + (v673_data * v694_data));
          float v699_data = s0[99];
          float v701_data = ir0[5];
          ir0[5] = (v701_data + (v673_data * v699_data));
          float v704_data = s0[116];
          float v706_data = ir0[6];
          ir0[6] = (v706_data + (v673_data * v704_data));
          float v709_data = s0[133];
          float v711_data = ir0[7];
          ir0[7] = (v711_data + (v673_data * v709_data));
          float v714_data = s0[150];
          float v716_data = ir0[8];
          ir0[8] = (v716_data + (v673_data * v714_data));
          float v719_data = glb_m1[(v25_lead + 240)];
          float v720_data = s0[15];
          float v722_data = ir0[0];
          ir0[0] = (v722_data + (v719_data * v720_data));
          float v725_data = s0[32];
          float v727_data = ir0[1];
          ir0[1] = (v727_data + (v719_data * v725_data));
          float v730_data = s0[49];
          float v732_data = ir0[2];
          ir0[2] = (v732_data + (v719_data * v730_data));
          float v735_data = s0[66];
          float v737_data = ir0[3];
          ir0[3] = (v737_data + (v719_data * v735_data));
          float v740_data = s0[83];
          float v742_data = ir0[4];
          ir0[4] = (v742_data + (v719_data * v740_data));
          float v745_data = s0[100];
          float v747_data = ir0[5];
          ir0[5] = (v747_data + (v719_data * v745_data));
          float v750_data = s0[117];
          float v752_data = ir0[6];
          ir0[6] = (v752_data + (v719_data * v750_data));
          float v755_data = s0[134];
          float v757_data = ir0[7];
          ir0[7] = (v757_data + (v719_data * v755_data));
          float v760_data = s0[151];
          float v762_data = ir0[8];
          ir0[8] = (v762_data + (v719_data * v760_data));
          float v765_data = glb_m1[(v25_lead + 256)];
          float v766_data = s0[16];
          float v768_data = ir0[0];
          ir0[0] = (v768_data + (v765_data * v766_data));
          float v771_data = s0[33];
          float v773_data = ir0[1];
          ir0[1] = (v773_data + (v765_data * v771_data));
          float v776_data = s0[50];
          float v778_data = ir0[2];
          ir0[2] = (v778_data + (v765_data * v776_data));
          float v781_data = s0[67];
          float v783_data = ir0[3];
          ir0[3] = (v783_data + (v765_data * v781_data));
          float v786_data = s0[84];
          float v788_data = ir0[4];
          ir0[4] = (v788_data + (v765_data * v786_data));
          float v791_data = s0[101];
          float v793_data = ir0[5];
          ir0[5] = (v793_data + (v765_data * v791_data));
          float v796_data = s0[118];
          float v798_data = ir0[6];
          ir0[6] = (v798_data + (v765_data * v796_data));
          float v801_data = s0[135];
          float v803_data = ir0[7];
          ir0[7] = (v803_data + (v765_data * v801_data));
          float v806_data = s0[152];
          float v808_data = ir0[8];
          ir0[8] = (v808_data + (v765_data * v806_data));
          // r0 = ir0
          #pragma unroll
          for (int32_t v813_n0 = 0; v813_n0 < 1; ++v813_n0) {
            #pragma unroll
            for (int32_t v814_n1 = 0; v814_n1 < 9; ++v814_n1) {
              int32_t v815_a = v813_n0 + v814_n1;
              float v816_data = ir0[v815_a];
              r0[v815_a] = v816_data;
            }
          }
          // glb_m0 = store{r>g}(r0);
          #pragma unroll
          for (int32_t v820_i0 = 0; v820_i0 < 1; ++v820_i0) {
            int32_t v825_lead = v25_lead + (v820_i0 * 16);
            #pragma unroll
            for (int32_t v821_i1 = 0; v821_i1 < 9; ++v821_i1) {
              float v823_data = r0[(v820_i0 + v821_i1)];
              glb_m0[(v825_lead + (v821_i1 * 16))] = v823_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

