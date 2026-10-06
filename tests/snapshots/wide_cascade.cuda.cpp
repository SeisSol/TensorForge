// === base name ===
kernel_b835fc999ab3a514

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b835fc999ab3a514 = {{16, 8, 1}, 16, 16, 1, 8, 6656, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b835fc999ab3a514(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b835fc999ab3a514(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b835fc999ab3a514(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_b835fc999ab3a514, block.x * block.y * block.z, 1664 * sizeof(float));
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
  config.sharedMemBytes = 1664 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_b835fc999ab3a514(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b835fc999ab3a514(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_b835fc999ab3a514, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_b835fc999ab3a514<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_b835fc999ab3a514(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 8 per block = block 16x8x1, 6656 B shared, occupancy grid
    // operands:
    //   m0 16×11(16×11) {0..16}×{0..11} strided
    //   m1 16×16(16×16) {0..16}×{0..16} strided
    //   m2 16×11(16×11) {0..16}×{0..11} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1664}],"shared_bytes":6656,"shared_elements":1664,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[16,11]],"name":"m0","ordered":false,"parts":1,"shape":[16,11],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,11]],"name":"m2","ordered":false,"parts":1,"shape":[16,11],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,11]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,11]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,11]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,11]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[208 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[192];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v11_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v11_batchId0 < numElements0; v11_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v12_ahead1 = v11_batchId0 + (gridDim.x * blockDim.y);
        size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v11_batchId0 * 176 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v11_batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v11_batchId0 * 176 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v25_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v26_i0 = 0; v26_i0 < 1; ++v26_i0) {
            int32_t v29_lead = v25_lead + (v26_i0 * 16);
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 16; ++v27_i1) {
              float v32_data = __ldcg(&glb_m1[(v29_lead + (v27_i1 * 16))]);
              r0[(v26_i0 + v27_i1)] = v32_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 4 * threadIdx.x + 0], &glb_m2[0 + 0 + 4 * threadIdx.x + 0], 16);
          __pipeline_memcpy_async(&s0[0 + 0 + 4 * threadIdx.x + 64], &glb_m2[0 + 0 + 4 * threadIdx.x + 64], 16);
          if (threadIdx.x < 12) {
            __pipeline_memcpy_async(&s0[0 + 0 + 4 * threadIdx.x + 128], &glb_m2[0 + 0 + 4 * threadIdx.x + 128], 16);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[11]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // ir1 = +(r0 * s0)
          // [(0, 16), (0, 11)] [(0, 16)]
          float ir1[11]{};
          float v39_data = r0[0];
          float v40_data = s0[0];
          float v42_data = ir1[0];
          ir1[0] = (v42_data + (v39_data * v40_data));
          float v45_data = s0[16];
          float v47_data = ir1[1];
          ir1[1] = (v47_data + (v39_data * v45_data));
          float v50_data = s0[32];
          float v52_data = ir1[2];
          ir1[2] = (v52_data + (v39_data * v50_data));
          float v55_data = s0[48];
          float v57_data = ir1[3];
          ir1[3] = (v57_data + (v39_data * v55_data));
          float v60_data = s0[64];
          float v62_data = ir1[4];
          ir1[4] = (v62_data + (v39_data * v60_data));
          float v65_data = s0[80];
          float v67_data = ir1[5];
          ir1[5] = (v67_data + (v39_data * v65_data));
          float v70_data = s0[96];
          float v72_data = ir1[6];
          ir1[6] = (v72_data + (v39_data * v70_data));
          float v75_data = s0[112];
          float v77_data = ir1[7];
          ir1[7] = (v77_data + (v39_data * v75_data));
          float v80_data = s0[128];
          float v82_data = ir1[8];
          ir1[8] = (v82_data + (v39_data * v80_data));
          float v85_data = s0[144];
          float v87_data = ir1[9];
          ir1[9] = (v87_data + (v39_data * v85_data));
          float v90_data = s0[160];
          float v92_data = ir1[10];
          ir1[10] = (v92_data + (v39_data * v90_data));
          float v94_data = r0[1];
          float v95_data = s0[1];
          float v97_data = ir1[0];
          ir1[0] = (v97_data + (v94_data * v95_data));
          float v100_data = s0[17];
          float v102_data = ir1[1];
          ir1[1] = (v102_data + (v94_data * v100_data));
          float v105_data = s0[33];
          float v107_data = ir1[2];
          ir1[2] = (v107_data + (v94_data * v105_data));
          float v110_data = s0[49];
          float v112_data = ir1[3];
          ir1[3] = (v112_data + (v94_data * v110_data));
          float v115_data = s0[65];
          float v117_data = ir1[4];
          ir1[4] = (v117_data + (v94_data * v115_data));
          float v120_data = s0[81];
          float v122_data = ir1[5];
          ir1[5] = (v122_data + (v94_data * v120_data));
          float v125_data = s0[97];
          float v127_data = ir1[6];
          ir1[6] = (v127_data + (v94_data * v125_data));
          float v130_data = s0[113];
          float v132_data = ir1[7];
          ir1[7] = (v132_data + (v94_data * v130_data));
          float v135_data = s0[129];
          float v137_data = ir1[8];
          ir1[8] = (v137_data + (v94_data * v135_data));
          float v140_data = s0[145];
          float v142_data = ir1[9];
          ir1[9] = (v142_data + (v94_data * v140_data));
          float v145_data = s0[161];
          float v147_data = ir1[10];
          ir1[10] = (v147_data + (v94_data * v145_data));
          float v149_data = r0[2];
          float v150_data = s0[2];
          float v152_data = ir1[0];
          ir1[0] = (v152_data + (v149_data * v150_data));
          float v155_data = s0[18];
          float v157_data = ir1[1];
          ir1[1] = (v157_data + (v149_data * v155_data));
          float v160_data = s0[34];
          float v162_data = ir1[2];
          ir1[2] = (v162_data + (v149_data * v160_data));
          float v165_data = s0[50];
          float v167_data = ir1[3];
          ir1[3] = (v167_data + (v149_data * v165_data));
          float v170_data = s0[66];
          float v172_data = ir1[4];
          ir1[4] = (v172_data + (v149_data * v170_data));
          float v175_data = s0[82];
          float v177_data = ir1[5];
          ir1[5] = (v177_data + (v149_data * v175_data));
          float v180_data = s0[98];
          float v182_data = ir1[6];
          ir1[6] = (v182_data + (v149_data * v180_data));
          float v185_data = s0[114];
          float v187_data = ir1[7];
          ir1[7] = (v187_data + (v149_data * v185_data));
          float v190_data = s0[130];
          float v192_data = ir1[8];
          ir1[8] = (v192_data + (v149_data * v190_data));
          float v195_data = s0[146];
          float v197_data = ir1[9];
          ir1[9] = (v197_data + (v149_data * v195_data));
          float v200_data = s0[162];
          float v202_data = ir1[10];
          ir1[10] = (v202_data + (v149_data * v200_data));
          float v204_data = r0[3];
          float v205_data = s0[3];
          float v207_data = ir1[0];
          ir1[0] = (v207_data + (v204_data * v205_data));
          float v210_data = s0[19];
          float v212_data = ir1[1];
          ir1[1] = (v212_data + (v204_data * v210_data));
          float v215_data = s0[35];
          float v217_data = ir1[2];
          ir1[2] = (v217_data + (v204_data * v215_data));
          float v220_data = s0[51];
          float v222_data = ir1[3];
          ir1[3] = (v222_data + (v204_data * v220_data));
          float v225_data = s0[67];
          float v227_data = ir1[4];
          ir1[4] = (v227_data + (v204_data * v225_data));
          float v230_data = s0[83];
          float v232_data = ir1[5];
          ir1[5] = (v232_data + (v204_data * v230_data));
          float v235_data = s0[99];
          float v237_data = ir1[6];
          ir1[6] = (v237_data + (v204_data * v235_data));
          float v240_data = s0[115];
          float v242_data = ir1[7];
          ir1[7] = (v242_data + (v204_data * v240_data));
          float v245_data = s0[131];
          float v247_data = ir1[8];
          ir1[8] = (v247_data + (v204_data * v245_data));
          float v250_data = s0[147];
          float v252_data = ir1[9];
          ir1[9] = (v252_data + (v204_data * v250_data));
          float v255_data = s0[163];
          float v257_data = ir1[10];
          ir1[10] = (v257_data + (v204_data * v255_data));
          float v259_data = r0[4];
          float v260_data = s0[4];
          float v262_data = ir1[0];
          ir1[0] = (v262_data + (v259_data * v260_data));
          float v265_data = s0[20];
          float v267_data = ir1[1];
          ir1[1] = (v267_data + (v259_data * v265_data));
          float v270_data = s0[36];
          float v272_data = ir1[2];
          ir1[2] = (v272_data + (v259_data * v270_data));
          float v275_data = s0[52];
          float v277_data = ir1[3];
          ir1[3] = (v277_data + (v259_data * v275_data));
          float v280_data = s0[68];
          float v282_data = ir1[4];
          ir1[4] = (v282_data + (v259_data * v280_data));
          float v285_data = s0[84];
          float v287_data = ir1[5];
          ir1[5] = (v287_data + (v259_data * v285_data));
          float v290_data = s0[100];
          float v292_data = ir1[6];
          ir1[6] = (v292_data + (v259_data * v290_data));
          float v295_data = s0[116];
          float v297_data = ir1[7];
          ir1[7] = (v297_data + (v259_data * v295_data));
          float v300_data = s0[132];
          float v302_data = ir1[8];
          ir1[8] = (v302_data + (v259_data * v300_data));
          float v305_data = s0[148];
          float v307_data = ir1[9];
          ir1[9] = (v307_data + (v259_data * v305_data));
          float v310_data = s0[164];
          float v312_data = ir1[10];
          ir1[10] = (v312_data + (v259_data * v310_data));
          float v314_data = r0[5];
          float v315_data = s0[5];
          float v317_data = ir1[0];
          ir1[0] = (v317_data + (v314_data * v315_data));
          float v320_data = s0[21];
          float v322_data = ir1[1];
          ir1[1] = (v322_data + (v314_data * v320_data));
          float v325_data = s0[37];
          float v327_data = ir1[2];
          ir1[2] = (v327_data + (v314_data * v325_data));
          float v330_data = s0[53];
          float v332_data = ir1[3];
          ir1[3] = (v332_data + (v314_data * v330_data));
          float v335_data = s0[69];
          float v337_data = ir1[4];
          ir1[4] = (v337_data + (v314_data * v335_data));
          float v340_data = s0[85];
          float v342_data = ir1[5];
          ir1[5] = (v342_data + (v314_data * v340_data));
          float v345_data = s0[101];
          float v347_data = ir1[6];
          ir1[6] = (v347_data + (v314_data * v345_data));
          float v350_data = s0[117];
          float v352_data = ir1[7];
          ir1[7] = (v352_data + (v314_data * v350_data));
          float v355_data = s0[133];
          float v357_data = ir1[8];
          ir1[8] = (v357_data + (v314_data * v355_data));
          float v360_data = s0[149];
          float v362_data = ir1[9];
          ir1[9] = (v362_data + (v314_data * v360_data));
          float v365_data = s0[165];
          float v367_data = ir1[10];
          ir1[10] = (v367_data + (v314_data * v365_data));
          float v369_data = r0[6];
          float v370_data = s0[6];
          float v372_data = ir1[0];
          ir1[0] = (v372_data + (v369_data * v370_data));
          float v375_data = s0[22];
          float v377_data = ir1[1];
          ir1[1] = (v377_data + (v369_data * v375_data));
          float v380_data = s0[38];
          float v382_data = ir1[2];
          ir1[2] = (v382_data + (v369_data * v380_data));
          float v385_data = s0[54];
          float v387_data = ir1[3];
          ir1[3] = (v387_data + (v369_data * v385_data));
          float v390_data = s0[70];
          float v392_data = ir1[4];
          ir1[4] = (v392_data + (v369_data * v390_data));
          float v395_data = s0[86];
          float v397_data = ir1[5];
          ir1[5] = (v397_data + (v369_data * v395_data));
          float v400_data = s0[102];
          float v402_data = ir1[6];
          ir1[6] = (v402_data + (v369_data * v400_data));
          float v405_data = s0[118];
          float v407_data = ir1[7];
          ir1[7] = (v407_data + (v369_data * v405_data));
          float v410_data = s0[134];
          float v412_data = ir1[8];
          ir1[8] = (v412_data + (v369_data * v410_data));
          float v415_data = s0[150];
          float v417_data = ir1[9];
          ir1[9] = (v417_data + (v369_data * v415_data));
          float v420_data = s0[166];
          float v422_data = ir1[10];
          ir1[10] = (v422_data + (v369_data * v420_data));
          float v424_data = r0[7];
          float v425_data = s0[7];
          float v427_data = ir1[0];
          ir1[0] = (v427_data + (v424_data * v425_data));
          float v430_data = s0[23];
          float v432_data = ir1[1];
          ir1[1] = (v432_data + (v424_data * v430_data));
          float v435_data = s0[39];
          float v437_data = ir1[2];
          ir1[2] = (v437_data + (v424_data * v435_data));
          float v440_data = s0[55];
          float v442_data = ir1[3];
          ir1[3] = (v442_data + (v424_data * v440_data));
          float v445_data = s0[71];
          float v447_data = ir1[4];
          ir1[4] = (v447_data + (v424_data * v445_data));
          float v450_data = s0[87];
          float v452_data = ir1[5];
          ir1[5] = (v452_data + (v424_data * v450_data));
          float v455_data = s0[103];
          float v457_data = ir1[6];
          ir1[6] = (v457_data + (v424_data * v455_data));
          float v460_data = s0[119];
          float v462_data = ir1[7];
          ir1[7] = (v462_data + (v424_data * v460_data));
          float v465_data = s0[135];
          float v467_data = ir1[8];
          ir1[8] = (v467_data + (v424_data * v465_data));
          float v470_data = s0[151];
          float v472_data = ir1[9];
          ir1[9] = (v472_data + (v424_data * v470_data));
          float v475_data = s0[167];
          float v477_data = ir1[10];
          ir1[10] = (v477_data + (v424_data * v475_data));
          float v479_data = r0[8];
          float v480_data = s0[8];
          float v482_data = ir1[0];
          ir1[0] = (v482_data + (v479_data * v480_data));
          float v485_data = s0[24];
          float v487_data = ir1[1];
          ir1[1] = (v487_data + (v479_data * v485_data));
          float v490_data = s0[40];
          float v492_data = ir1[2];
          ir1[2] = (v492_data + (v479_data * v490_data));
          float v495_data = s0[56];
          float v497_data = ir1[3];
          ir1[3] = (v497_data + (v479_data * v495_data));
          float v500_data = s0[72];
          float v502_data = ir1[4];
          ir1[4] = (v502_data + (v479_data * v500_data));
          float v505_data = s0[88];
          float v507_data = ir1[5];
          ir1[5] = (v507_data + (v479_data * v505_data));
          float v510_data = s0[104];
          float v512_data = ir1[6];
          ir1[6] = (v512_data + (v479_data * v510_data));
          float v515_data = s0[120];
          float v517_data = ir1[7];
          ir1[7] = (v517_data + (v479_data * v515_data));
          float v520_data = s0[136];
          float v522_data = ir1[8];
          ir1[8] = (v522_data + (v479_data * v520_data));
          float v525_data = s0[152];
          float v527_data = ir1[9];
          ir1[9] = (v527_data + (v479_data * v525_data));
          float v530_data = s0[168];
          float v532_data = ir1[10];
          ir1[10] = (v532_data + (v479_data * v530_data));
          float v534_data = r0[9];
          float v535_data = s0[9];
          float v537_data = ir1[0];
          ir1[0] = (v537_data + (v534_data * v535_data));
          float v540_data = s0[25];
          float v542_data = ir1[1];
          ir1[1] = (v542_data + (v534_data * v540_data));
          float v545_data = s0[41];
          float v547_data = ir1[2];
          ir1[2] = (v547_data + (v534_data * v545_data));
          float v550_data = s0[57];
          float v552_data = ir1[3];
          ir1[3] = (v552_data + (v534_data * v550_data));
          float v555_data = s0[73];
          float v557_data = ir1[4];
          ir1[4] = (v557_data + (v534_data * v555_data));
          float v560_data = s0[89];
          float v562_data = ir1[5];
          ir1[5] = (v562_data + (v534_data * v560_data));
          float v565_data = s0[105];
          float v567_data = ir1[6];
          ir1[6] = (v567_data + (v534_data * v565_data));
          float v570_data = s0[121];
          float v572_data = ir1[7];
          ir1[7] = (v572_data + (v534_data * v570_data));
          float v575_data = s0[137];
          float v577_data = ir1[8];
          ir1[8] = (v577_data + (v534_data * v575_data));
          float v580_data = s0[153];
          float v582_data = ir1[9];
          ir1[9] = (v582_data + (v534_data * v580_data));
          float v585_data = s0[169];
          float v587_data = ir1[10];
          ir1[10] = (v587_data + (v534_data * v585_data));
          float v589_data = r0[10];
          float v590_data = s0[10];
          float v592_data = ir1[0];
          ir1[0] = (v592_data + (v589_data * v590_data));
          float v595_data = s0[26];
          float v597_data = ir1[1];
          ir1[1] = (v597_data + (v589_data * v595_data));
          float v600_data = s0[42];
          float v602_data = ir1[2];
          ir1[2] = (v602_data + (v589_data * v600_data));
          float v605_data = s0[58];
          float v607_data = ir1[3];
          ir1[3] = (v607_data + (v589_data * v605_data));
          float v610_data = s0[74];
          float v612_data = ir1[4];
          ir1[4] = (v612_data + (v589_data * v610_data));
          float v615_data = s0[90];
          float v617_data = ir1[5];
          ir1[5] = (v617_data + (v589_data * v615_data));
          float v620_data = s0[106];
          float v622_data = ir1[6];
          ir1[6] = (v622_data + (v589_data * v620_data));
          float v625_data = s0[122];
          float v627_data = ir1[7];
          ir1[7] = (v627_data + (v589_data * v625_data));
          float v630_data = s0[138];
          float v632_data = ir1[8];
          ir1[8] = (v632_data + (v589_data * v630_data));
          float v635_data = s0[154];
          float v637_data = ir1[9];
          ir1[9] = (v637_data + (v589_data * v635_data));
          float v640_data = s0[170];
          float v642_data = ir1[10];
          ir1[10] = (v642_data + (v589_data * v640_data));
          float v644_data = r0[11];
          float v645_data = s0[11];
          float v647_data = ir1[0];
          ir1[0] = (v647_data + (v644_data * v645_data));
          float v650_data = s0[27];
          float v652_data = ir1[1];
          ir1[1] = (v652_data + (v644_data * v650_data));
          float v655_data = s0[43];
          float v657_data = ir1[2];
          ir1[2] = (v657_data + (v644_data * v655_data));
          float v660_data = s0[59];
          float v662_data = ir1[3];
          ir1[3] = (v662_data + (v644_data * v660_data));
          float v665_data = s0[75];
          float v667_data = ir1[4];
          ir1[4] = (v667_data + (v644_data * v665_data));
          float v670_data = s0[91];
          float v672_data = ir1[5];
          ir1[5] = (v672_data + (v644_data * v670_data));
          float v675_data = s0[107];
          float v677_data = ir1[6];
          ir1[6] = (v677_data + (v644_data * v675_data));
          float v680_data = s0[123];
          float v682_data = ir1[7];
          ir1[7] = (v682_data + (v644_data * v680_data));
          float v685_data = s0[139];
          float v687_data = ir1[8];
          ir1[8] = (v687_data + (v644_data * v685_data));
          float v690_data = s0[155];
          float v692_data = ir1[9];
          ir1[9] = (v692_data + (v644_data * v690_data));
          float v695_data = s0[171];
          float v697_data = ir1[10];
          ir1[10] = (v697_data + (v644_data * v695_data));
          float v699_data = r0[12];
          float v700_data = s0[12];
          float v702_data = ir1[0];
          ir1[0] = (v702_data + (v699_data * v700_data));
          float v705_data = s0[28];
          float v707_data = ir1[1];
          ir1[1] = (v707_data + (v699_data * v705_data));
          float v710_data = s0[44];
          float v712_data = ir1[2];
          ir1[2] = (v712_data + (v699_data * v710_data));
          float v715_data = s0[60];
          float v717_data = ir1[3];
          ir1[3] = (v717_data + (v699_data * v715_data));
          float v720_data = s0[76];
          float v722_data = ir1[4];
          ir1[4] = (v722_data + (v699_data * v720_data));
          float v725_data = s0[92];
          float v727_data = ir1[5];
          ir1[5] = (v727_data + (v699_data * v725_data));
          float v730_data = s0[108];
          float v732_data = ir1[6];
          ir1[6] = (v732_data + (v699_data * v730_data));
          float v735_data = s0[124];
          float v737_data = ir1[7];
          ir1[7] = (v737_data + (v699_data * v735_data));
          float v740_data = s0[140];
          float v742_data = ir1[8];
          ir1[8] = (v742_data + (v699_data * v740_data));
          float v745_data = s0[156];
          float v747_data = ir1[9];
          ir1[9] = (v747_data + (v699_data * v745_data));
          float v750_data = s0[172];
          float v752_data = ir1[10];
          ir1[10] = (v752_data + (v699_data * v750_data));
          float v754_data = r0[13];
          float v755_data = s0[13];
          float v757_data = ir1[0];
          ir1[0] = (v757_data + (v754_data * v755_data));
          float v760_data = s0[29];
          float v762_data = ir1[1];
          ir1[1] = (v762_data + (v754_data * v760_data));
          float v765_data = s0[45];
          float v767_data = ir1[2];
          ir1[2] = (v767_data + (v754_data * v765_data));
          float v770_data = s0[61];
          float v772_data = ir1[3];
          ir1[3] = (v772_data + (v754_data * v770_data));
          float v775_data = s0[77];
          float v777_data = ir1[4];
          ir1[4] = (v777_data + (v754_data * v775_data));
          float v780_data = s0[93];
          float v782_data = ir1[5];
          ir1[5] = (v782_data + (v754_data * v780_data));
          float v785_data = s0[109];
          float v787_data = ir1[6];
          ir1[6] = (v787_data + (v754_data * v785_data));
          float v790_data = s0[125];
          float v792_data = ir1[7];
          ir1[7] = (v792_data + (v754_data * v790_data));
          float v795_data = s0[141];
          float v797_data = ir1[8];
          ir1[8] = (v797_data + (v754_data * v795_data));
          float v800_data = s0[157];
          float v802_data = ir1[9];
          ir1[9] = (v802_data + (v754_data * v800_data));
          float v805_data = s0[173];
          float v807_data = ir1[10];
          ir1[10] = (v807_data + (v754_data * v805_data));
          float v809_data = r0[14];
          float v810_data = s0[14];
          float v812_data = ir1[0];
          ir1[0] = (v812_data + (v809_data * v810_data));
          float v815_data = s0[30];
          float v817_data = ir1[1];
          ir1[1] = (v817_data + (v809_data * v815_data));
          float v820_data = s0[46];
          float v822_data = ir1[2];
          ir1[2] = (v822_data + (v809_data * v820_data));
          float v825_data = s0[62];
          float v827_data = ir1[3];
          ir1[3] = (v827_data + (v809_data * v825_data));
          float v830_data = s0[78];
          float v832_data = ir1[4];
          ir1[4] = (v832_data + (v809_data * v830_data));
          float v835_data = s0[94];
          float v837_data = ir1[5];
          ir1[5] = (v837_data + (v809_data * v835_data));
          float v840_data = s0[110];
          float v842_data = ir1[6];
          ir1[6] = (v842_data + (v809_data * v840_data));
          float v845_data = s0[126];
          float v847_data = ir1[7];
          ir1[7] = (v847_data + (v809_data * v845_data));
          float v850_data = s0[142];
          float v852_data = ir1[8];
          ir1[8] = (v852_data + (v809_data * v850_data));
          float v855_data = s0[158];
          float v857_data = ir1[9];
          ir1[9] = (v857_data + (v809_data * v855_data));
          float v860_data = s0[174];
          float v862_data = ir1[10];
          ir1[10] = (v862_data + (v809_data * v860_data));
          float v864_data = r0[15];
          float v865_data = s0[15];
          float v867_data = ir1[0];
          ir1[0] = (v867_data + (v864_data * v865_data));
          float v870_data = s0[31];
          float v872_data = ir1[1];
          ir1[1] = (v872_data + (v864_data * v870_data));
          float v875_data = s0[47];
          float v877_data = ir1[2];
          ir1[2] = (v877_data + (v864_data * v875_data));
          float v880_data = s0[63];
          float v882_data = ir1[3];
          ir1[3] = (v882_data + (v864_data * v880_data));
          float v885_data = s0[79];
          float v887_data = ir1[4];
          ir1[4] = (v887_data + (v864_data * v885_data));
          float v890_data = s0[95];
          float v892_data = ir1[5];
          ir1[5] = (v892_data + (v864_data * v890_data));
          float v895_data = s0[111];
          float v897_data = ir1[6];
          ir1[6] = (v897_data + (v864_data * v895_data));
          float v900_data = s0[127];
          float v902_data = ir1[7];
          ir1[7] = (v902_data + (v864_data * v900_data));
          float v905_data = s0[143];
          float v907_data = ir1[8];
          ir1[8] = (v907_data + (v864_data * v905_data));
          float v910_data = s0[159];
          float v912_data = ir1[9];
          ir1[9] = (v912_data + (v864_data * v910_data));
          float v915_data = s0[175];
          float v917_data = ir1[10];
          ir1[10] = (v917_data + (v864_data * v915_data));
          // r1 = ir1
          #pragma unroll
          for (int32_t v919_n0 = 0; v919_n0 < 1; ++v919_n0) {
            #pragma unroll
            for (int32_t v920_n1 = 0; v920_n1 < 11; ++v920_n1) {
              int32_t v921_a = v919_n0 + v920_n1;
              float v922_data = ir1[v921_a];
              r1[v921_a] = v922_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v923_i0 = 0; v923_i0 < 1; ++v923_i0) {
            int32_t v928_lead = v25_lead + (v923_i0 * 16);
            #pragma unroll
            for (int32_t v924_i1 = 0; v924_i1 < 11; ++v924_i1) {
              float v926_data = r1[(v923_i0 + v924_i1)];
              glb_m0[(v928_lead + (v924_i1 * 16))] = v926_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

