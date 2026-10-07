// === base name ===
kernel_99a58baf1c1309bf

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_99a58baf1c1309bf = {{16, 8, 1}, 16, 12, 1, 8, 4608, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_99a58baf1c1309bf(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_99a58baf1c1309bf(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_99a58baf1c1309bf(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_99a58baf1c1309bf, block.x * block.y * block.z, 1152 * sizeof(float));
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
  config.sharedMemBytes = 1152 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_99a58baf1c1309bf(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_99a58baf1c1309bf(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_99a58baf1c1309bf, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_99a58baf1c1309bf<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_99a58baf1c1309bf(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 8 per block = block 16x8x1, 4608 B shared, occupancy grid
    // operands:
    //   m0 12×8(12×8) {0..12}×{0..8} strided
    //   m1 12×16(12×16) {0..12}×{0..16} strided
    //   m2 16×8(16×8) {0..16}×{0..8} strided
    // operations:
    //   m0[i,j] += m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1152}],"shared_bytes":4608,"shared_elements":1152,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,8]],"name":"m0","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,16]],"name":"m1","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,16]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[144 * threadIdx.y + 0];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 96 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 192 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 128 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v22_lead = threadIdx.x % 16;
          bool v23_g = v22_lead < 12;
          if (v23_g) {
            #pragma unroll
            for (int32_t v24_i1 = 0; v24_i1 < 16; ++v24_i1) {
              float v29_data = __ldcg(&glb_m1[(v22_lead + (v24_i1 * 12))]);
              r0[v24_i1] = v29_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 8; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
          __pipeline_commit();
          float r1[8]{};
          // r1 = load{g>r}(glb_m0);
          if (v23_g) {
            #pragma unroll
            for (int32_t v33_i1 = 0; v33_i1 < 8; ++v33_i1) {
              float v38_data = glb_m0[(v22_lead + (v33_i1 * 12))];
              r1[v33_i1] = v38_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r2[8]{};
          // ir2 = +(r0 * s0)
          // [(0, 12), (0, 8)] [(0, 16)]
          float ir2[8]{};
          float v42_data = r0[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v43_data = s0[0];
          float v45_data = ir2[0];
          ir2[0] = (v45_data + (v42_data * v43_data));
          float v48_data = s0[16];
          float v50_data = ir2[1];
          ir2[1] = (v50_data + (v42_data * v48_data));
          float v53_data = s0[32];
          float v55_data = ir2[2];
          ir2[2] = (v55_data + (v42_data * v53_data));
          float v58_data = s0[48];
          float v60_data = ir2[3];
          ir2[3] = (v60_data + (v42_data * v58_data));
          float v63_data = s0[64];
          float v65_data = ir2[4];
          ir2[4] = (v65_data + (v42_data * v63_data));
          float v68_data = s0[80];
          float v70_data = ir2[5];
          ir2[5] = (v70_data + (v42_data * v68_data));
          float v73_data = s0[96];
          float v75_data = ir2[6];
          ir2[6] = (v75_data + (v42_data * v73_data));
          float v78_data = s0[112];
          float v80_data = ir2[7];
          ir2[7] = (v80_data + (v42_data * v78_data));
          float v82_data = r0[1];
          float v83_data = s0[1];
          float v85_data = ir2[0];
          ir2[0] = (v85_data + (v82_data * v83_data));
          float v88_data = s0[17];
          float v90_data = ir2[1];
          ir2[1] = (v90_data + (v82_data * v88_data));
          float v93_data = s0[33];
          float v95_data = ir2[2];
          ir2[2] = (v95_data + (v82_data * v93_data));
          float v98_data = s0[49];
          float v100_data = ir2[3];
          ir2[3] = (v100_data + (v82_data * v98_data));
          float v103_data = s0[65];
          float v105_data = ir2[4];
          ir2[4] = (v105_data + (v82_data * v103_data));
          float v108_data = s0[81];
          float v110_data = ir2[5];
          ir2[5] = (v110_data + (v82_data * v108_data));
          float v113_data = s0[97];
          float v115_data = ir2[6];
          ir2[6] = (v115_data + (v82_data * v113_data));
          float v118_data = s0[113];
          float v120_data = ir2[7];
          ir2[7] = (v120_data + (v82_data * v118_data));
          float v122_data = r0[2];
          float v123_data = s0[2];
          float v125_data = ir2[0];
          ir2[0] = (v125_data + (v122_data * v123_data));
          float v128_data = s0[18];
          float v130_data = ir2[1];
          ir2[1] = (v130_data + (v122_data * v128_data));
          float v133_data = s0[34];
          float v135_data = ir2[2];
          ir2[2] = (v135_data + (v122_data * v133_data));
          float v138_data = s0[50];
          float v140_data = ir2[3];
          ir2[3] = (v140_data + (v122_data * v138_data));
          float v143_data = s0[66];
          float v145_data = ir2[4];
          ir2[4] = (v145_data + (v122_data * v143_data));
          float v148_data = s0[82];
          float v150_data = ir2[5];
          ir2[5] = (v150_data + (v122_data * v148_data));
          float v153_data = s0[98];
          float v155_data = ir2[6];
          ir2[6] = (v155_data + (v122_data * v153_data));
          float v158_data = s0[114];
          float v160_data = ir2[7];
          ir2[7] = (v160_data + (v122_data * v158_data));
          float v162_data = r0[3];
          float v163_data = s0[3];
          float v165_data = ir2[0];
          ir2[0] = (v165_data + (v162_data * v163_data));
          float v168_data = s0[19];
          float v170_data = ir2[1];
          ir2[1] = (v170_data + (v162_data * v168_data));
          float v173_data = s0[35];
          float v175_data = ir2[2];
          ir2[2] = (v175_data + (v162_data * v173_data));
          float v178_data = s0[51];
          float v180_data = ir2[3];
          ir2[3] = (v180_data + (v162_data * v178_data));
          float v183_data = s0[67];
          float v185_data = ir2[4];
          ir2[4] = (v185_data + (v162_data * v183_data));
          float v188_data = s0[83];
          float v190_data = ir2[5];
          ir2[5] = (v190_data + (v162_data * v188_data));
          float v193_data = s0[99];
          float v195_data = ir2[6];
          ir2[6] = (v195_data + (v162_data * v193_data));
          float v198_data = s0[115];
          float v200_data = ir2[7];
          ir2[7] = (v200_data + (v162_data * v198_data));
          float v202_data = r0[4];
          float v203_data = s0[4];
          float v205_data = ir2[0];
          ir2[0] = (v205_data + (v202_data * v203_data));
          float v208_data = s0[20];
          float v210_data = ir2[1];
          ir2[1] = (v210_data + (v202_data * v208_data));
          float v213_data = s0[36];
          float v215_data = ir2[2];
          ir2[2] = (v215_data + (v202_data * v213_data));
          float v218_data = s0[52];
          float v220_data = ir2[3];
          ir2[3] = (v220_data + (v202_data * v218_data));
          float v223_data = s0[68];
          float v225_data = ir2[4];
          ir2[4] = (v225_data + (v202_data * v223_data));
          float v228_data = s0[84];
          float v230_data = ir2[5];
          ir2[5] = (v230_data + (v202_data * v228_data));
          float v233_data = s0[100];
          float v235_data = ir2[6];
          ir2[6] = (v235_data + (v202_data * v233_data));
          float v238_data = s0[116];
          float v240_data = ir2[7];
          ir2[7] = (v240_data + (v202_data * v238_data));
          float v242_data = r0[5];
          float v243_data = s0[5];
          float v245_data = ir2[0];
          ir2[0] = (v245_data + (v242_data * v243_data));
          float v248_data = s0[21];
          float v250_data = ir2[1];
          ir2[1] = (v250_data + (v242_data * v248_data));
          float v253_data = s0[37];
          float v255_data = ir2[2];
          ir2[2] = (v255_data + (v242_data * v253_data));
          float v258_data = s0[53];
          float v260_data = ir2[3];
          ir2[3] = (v260_data + (v242_data * v258_data));
          float v263_data = s0[69];
          float v265_data = ir2[4];
          ir2[4] = (v265_data + (v242_data * v263_data));
          float v268_data = s0[85];
          float v270_data = ir2[5];
          ir2[5] = (v270_data + (v242_data * v268_data));
          float v273_data = s0[101];
          float v275_data = ir2[6];
          ir2[6] = (v275_data + (v242_data * v273_data));
          float v278_data = s0[117];
          float v280_data = ir2[7];
          ir2[7] = (v280_data + (v242_data * v278_data));
          float v282_data = r0[6];
          float v283_data = s0[6];
          float v285_data = ir2[0];
          ir2[0] = (v285_data + (v282_data * v283_data));
          float v288_data = s0[22];
          float v290_data = ir2[1];
          ir2[1] = (v290_data + (v282_data * v288_data));
          float v293_data = s0[38];
          float v295_data = ir2[2];
          ir2[2] = (v295_data + (v282_data * v293_data));
          float v298_data = s0[54];
          float v300_data = ir2[3];
          ir2[3] = (v300_data + (v282_data * v298_data));
          float v303_data = s0[70];
          float v305_data = ir2[4];
          ir2[4] = (v305_data + (v282_data * v303_data));
          float v308_data = s0[86];
          float v310_data = ir2[5];
          ir2[5] = (v310_data + (v282_data * v308_data));
          float v313_data = s0[102];
          float v315_data = ir2[6];
          ir2[6] = (v315_data + (v282_data * v313_data));
          float v318_data = s0[118];
          float v320_data = ir2[7];
          ir2[7] = (v320_data + (v282_data * v318_data));
          float v322_data = r0[7];
          float v323_data = s0[7];
          float v325_data = ir2[0];
          ir2[0] = (v325_data + (v322_data * v323_data));
          float v328_data = s0[23];
          float v330_data = ir2[1];
          ir2[1] = (v330_data + (v322_data * v328_data));
          float v333_data = s0[39];
          float v335_data = ir2[2];
          ir2[2] = (v335_data + (v322_data * v333_data));
          float v338_data = s0[55];
          float v340_data = ir2[3];
          ir2[3] = (v340_data + (v322_data * v338_data));
          float v343_data = s0[71];
          float v345_data = ir2[4];
          ir2[4] = (v345_data + (v322_data * v343_data));
          float v348_data = s0[87];
          float v350_data = ir2[5];
          ir2[5] = (v350_data + (v322_data * v348_data));
          float v353_data = s0[103];
          float v355_data = ir2[6];
          ir2[6] = (v355_data + (v322_data * v353_data));
          float v358_data = s0[119];
          float v360_data = ir2[7];
          ir2[7] = (v360_data + (v322_data * v358_data));
          float v362_data = r0[8];
          float v363_data = s0[8];
          float v365_data = ir2[0];
          ir2[0] = (v365_data + (v362_data * v363_data));
          float v368_data = s0[24];
          float v370_data = ir2[1];
          ir2[1] = (v370_data + (v362_data * v368_data));
          float v373_data = s0[40];
          float v375_data = ir2[2];
          ir2[2] = (v375_data + (v362_data * v373_data));
          float v378_data = s0[56];
          float v380_data = ir2[3];
          ir2[3] = (v380_data + (v362_data * v378_data));
          float v383_data = s0[72];
          float v385_data = ir2[4];
          ir2[4] = (v385_data + (v362_data * v383_data));
          float v388_data = s0[88];
          float v390_data = ir2[5];
          ir2[5] = (v390_data + (v362_data * v388_data));
          float v393_data = s0[104];
          float v395_data = ir2[6];
          ir2[6] = (v395_data + (v362_data * v393_data));
          float v398_data = s0[120];
          float v400_data = ir2[7];
          ir2[7] = (v400_data + (v362_data * v398_data));
          float v402_data = r0[9];
          float v403_data = s0[9];
          float v405_data = ir2[0];
          ir2[0] = (v405_data + (v402_data * v403_data));
          float v408_data = s0[25];
          float v410_data = ir2[1];
          ir2[1] = (v410_data + (v402_data * v408_data));
          float v413_data = s0[41];
          float v415_data = ir2[2];
          ir2[2] = (v415_data + (v402_data * v413_data));
          float v418_data = s0[57];
          float v420_data = ir2[3];
          ir2[3] = (v420_data + (v402_data * v418_data));
          float v423_data = s0[73];
          float v425_data = ir2[4];
          ir2[4] = (v425_data + (v402_data * v423_data));
          float v428_data = s0[89];
          float v430_data = ir2[5];
          ir2[5] = (v430_data + (v402_data * v428_data));
          float v433_data = s0[105];
          float v435_data = ir2[6];
          ir2[6] = (v435_data + (v402_data * v433_data));
          float v438_data = s0[121];
          float v440_data = ir2[7];
          ir2[7] = (v440_data + (v402_data * v438_data));
          float v442_data = r0[10];
          float v443_data = s0[10];
          float v445_data = ir2[0];
          ir2[0] = (v445_data + (v442_data * v443_data));
          float v448_data = s0[26];
          float v450_data = ir2[1];
          ir2[1] = (v450_data + (v442_data * v448_data));
          float v453_data = s0[42];
          float v455_data = ir2[2];
          ir2[2] = (v455_data + (v442_data * v453_data));
          float v458_data = s0[58];
          float v460_data = ir2[3];
          ir2[3] = (v460_data + (v442_data * v458_data));
          float v463_data = s0[74];
          float v465_data = ir2[4];
          ir2[4] = (v465_data + (v442_data * v463_data));
          float v468_data = s0[90];
          float v470_data = ir2[5];
          ir2[5] = (v470_data + (v442_data * v468_data));
          float v473_data = s0[106];
          float v475_data = ir2[6];
          ir2[6] = (v475_data + (v442_data * v473_data));
          float v478_data = s0[122];
          float v480_data = ir2[7];
          ir2[7] = (v480_data + (v442_data * v478_data));
          float v482_data = r0[11];
          float v483_data = s0[11];
          float v485_data = ir2[0];
          ir2[0] = (v485_data + (v482_data * v483_data));
          float v488_data = s0[27];
          float v490_data = ir2[1];
          ir2[1] = (v490_data + (v482_data * v488_data));
          float v493_data = s0[43];
          float v495_data = ir2[2];
          ir2[2] = (v495_data + (v482_data * v493_data));
          float v498_data = s0[59];
          float v500_data = ir2[3];
          ir2[3] = (v500_data + (v482_data * v498_data));
          float v503_data = s0[75];
          float v505_data = ir2[4];
          ir2[4] = (v505_data + (v482_data * v503_data));
          float v508_data = s0[91];
          float v510_data = ir2[5];
          ir2[5] = (v510_data + (v482_data * v508_data));
          float v513_data = s0[107];
          float v515_data = ir2[6];
          ir2[6] = (v515_data + (v482_data * v513_data));
          float v518_data = s0[123];
          float v520_data = ir2[7];
          ir2[7] = (v520_data + (v482_data * v518_data));
          float v522_data = r0[12];
          float v523_data = s0[12];
          float v525_data = ir2[0];
          ir2[0] = (v525_data + (v522_data * v523_data));
          float v528_data = s0[28];
          float v530_data = ir2[1];
          ir2[1] = (v530_data + (v522_data * v528_data));
          float v533_data = s0[44];
          float v535_data = ir2[2];
          ir2[2] = (v535_data + (v522_data * v533_data));
          float v538_data = s0[60];
          float v540_data = ir2[3];
          ir2[3] = (v540_data + (v522_data * v538_data));
          float v543_data = s0[76];
          float v545_data = ir2[4];
          ir2[4] = (v545_data + (v522_data * v543_data));
          float v548_data = s0[92];
          float v550_data = ir2[5];
          ir2[5] = (v550_data + (v522_data * v548_data));
          float v553_data = s0[108];
          float v555_data = ir2[6];
          ir2[6] = (v555_data + (v522_data * v553_data));
          float v558_data = s0[124];
          float v560_data = ir2[7];
          ir2[7] = (v560_data + (v522_data * v558_data));
          float v562_data = r0[13];
          float v563_data = s0[13];
          float v565_data = ir2[0];
          ir2[0] = (v565_data + (v562_data * v563_data));
          float v568_data = s0[29];
          float v570_data = ir2[1];
          ir2[1] = (v570_data + (v562_data * v568_data));
          float v573_data = s0[45];
          float v575_data = ir2[2];
          ir2[2] = (v575_data + (v562_data * v573_data));
          float v578_data = s0[61];
          float v580_data = ir2[3];
          ir2[3] = (v580_data + (v562_data * v578_data));
          float v583_data = s0[77];
          float v585_data = ir2[4];
          ir2[4] = (v585_data + (v562_data * v583_data));
          float v588_data = s0[93];
          float v590_data = ir2[5];
          ir2[5] = (v590_data + (v562_data * v588_data));
          float v593_data = s0[109];
          float v595_data = ir2[6];
          ir2[6] = (v595_data + (v562_data * v593_data));
          float v598_data = s0[125];
          float v600_data = ir2[7];
          ir2[7] = (v600_data + (v562_data * v598_data));
          float v602_data = r0[14];
          float v603_data = s0[14];
          float v605_data = ir2[0];
          ir2[0] = (v605_data + (v602_data * v603_data));
          float v608_data = s0[30];
          float v610_data = ir2[1];
          ir2[1] = (v610_data + (v602_data * v608_data));
          float v613_data = s0[46];
          float v615_data = ir2[2];
          ir2[2] = (v615_data + (v602_data * v613_data));
          float v618_data = s0[62];
          float v620_data = ir2[3];
          ir2[3] = (v620_data + (v602_data * v618_data));
          float v623_data = s0[78];
          float v625_data = ir2[4];
          ir2[4] = (v625_data + (v602_data * v623_data));
          float v628_data = s0[94];
          float v630_data = ir2[5];
          ir2[5] = (v630_data + (v602_data * v628_data));
          float v633_data = s0[110];
          float v635_data = ir2[6];
          ir2[6] = (v635_data + (v602_data * v633_data));
          float v638_data = s0[126];
          float v640_data = ir2[7];
          ir2[7] = (v640_data + (v602_data * v638_data));
          float v642_data = r0[15];
          float v643_data = s0[15];
          float v645_data = ir2[0];
          ir2[0] = (v645_data + (v642_data * v643_data));
          float v648_data = s0[31];
          float v650_data = ir2[1];
          ir2[1] = (v650_data + (v642_data * v648_data));
          float v653_data = s0[47];
          float v655_data = ir2[2];
          ir2[2] = (v655_data + (v642_data * v653_data));
          float v658_data = s0[63];
          float v660_data = ir2[3];
          ir2[3] = (v660_data + (v642_data * v658_data));
          float v663_data = s0[79];
          float v665_data = ir2[4];
          ir2[4] = (v665_data + (v642_data * v663_data));
          float v668_data = s0[95];
          float v670_data = ir2[5];
          ir2[5] = (v670_data + (v642_data * v668_data));
          float v673_data = s0[111];
          float v675_data = ir2[6];
          ir2[6] = (v675_data + (v642_data * v673_data));
          float v678_data = s0[127];
          float v680_data = ir2[7];
          ir2[7] = (v680_data + (v642_data * v678_data));
          // r2 = ir2 + r1
          if (v23_g) {
            #pragma unroll
            for (int32_t v682_n1 = 0; v682_n1 < 8; ++v682_n1) {
              float v684_data = ir2[v682_n1];
              float v685_data = r1[v682_n1];
              r2[v682_n1] = (v685_data + v684_data);
            }
          }
          // glb_m0 = store{r>g}(r2);
          if (v23_g) {
            #pragma unroll
            for (int32_t v687_i1 = 0; v687_i1 < 8; ++v687_i1) {
              float v689_data = r2[v687_i1];
              glb_m0[(v22_lead + (v687_i1 * 12))] = v689_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

