// === base name ===
kernel_66c4456f0b9d3821

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_66c4456f0b9d3821 = {{16, 8, 1}, 16, 12, 1, 8, 9216, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_66c4456f0b9d3821(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_66c4456f0b9d3821(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_66c4456f0b9d3821(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_66c4456f0b9d3821, block.x * block.y * block.z, 1152 * sizeof(double));
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
  config.sharedMemBytes = 1152 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_66c4456f0b9d3821(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_66c4456f0b9d3821(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_66c4456f0b9d3821, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_66c4456f0b9d3821<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_66c4456f0b9d3821(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 8 per block = block 16x8x1, 9216 B shared, occupancy grid
    // operands:
    //   m0 12×8(12×8) {0..12}×{0..8} strided
    //   m1 12×16(12×16) {0..12}×{0..16} strided
    //   m2 16×8(16×8) {0..16}×{0..8} strided
    // operations:
    //   m0[i,j] += m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"double","launch":{"active_threads":12,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1152}],"shared_bytes":9216,"shared_elements":1152,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,8]],"name":"m0","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,16]],"name":"m1","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,16]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[144 * threadIdx.y + 0];
      double * __restrict__ s0 = &localShrMem0[0];
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[v8_batchId0 * 96 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m1 = &m1[v8_batchId0 * 192 + 0 + m1_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[v8_batchId0 * 128 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v22_lead = threadIdx.x % 16;
          bool v23_g = v22_lead < 12;
          if (v23_g) {
            #pragma unroll
            for (int32_t v24_i1 = 0; v24_i1 < 16; ++v24_i1) {
              double v29_data = __ldcg(&glb_m1[(v22_lead + (v24_i1 * 12))]);
              r0[v24_i1] = v29_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 8; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 8);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          double r1[8]{};
          // r1 = load{g>r}(glb_m0);
          if (v23_g) {
            #pragma unroll
            for (int32_t v33_i1 = 0; v33_i1 < 8; ++v33_i1) {
              double v38_data = glb_m0[(v22_lead + (v33_i1 * 12))];
              r1[v33_i1] = v38_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          // wait(r1 = load{g>r}(glb_m0););
          double r2[8]{};
          // ir2 = +(r0 * s0)
          // [(0, 12), (0, 8)] [(0, 16)]
          double ir2[8]{};
          double v42_data = r0[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          double v43_data = s0[0];
          double v45_data = ir2[0];
          ir2[0] = (v45_data + (v42_data * v43_data));
          double v48_data = s0[16];
          double v50_data = ir2[1];
          ir2[1] = (v50_data + (v42_data * v48_data));
          double v53_data = s0[32];
          double v55_data = ir2[2];
          ir2[2] = (v55_data + (v42_data * v53_data));
          double v58_data = s0[48];
          double v60_data = ir2[3];
          ir2[3] = (v60_data + (v42_data * v58_data));
          double v63_data = s0[64];
          double v65_data = ir2[4];
          ir2[4] = (v65_data + (v42_data * v63_data));
          double v68_data = s0[80];
          double v70_data = ir2[5];
          ir2[5] = (v70_data + (v42_data * v68_data));
          double v73_data = s0[96];
          double v75_data = ir2[6];
          ir2[6] = (v75_data + (v42_data * v73_data));
          double v78_data = s0[112];
          double v80_data = ir2[7];
          ir2[7] = (v80_data + (v42_data * v78_data));
          double v82_data = r0[1];
          double v83_data = s0[1];
          double v85_data = ir2[0];
          ir2[0] = (v85_data + (v82_data * v83_data));
          double v88_data = s0[17];
          double v90_data = ir2[1];
          ir2[1] = (v90_data + (v82_data * v88_data));
          double v93_data = s0[33];
          double v95_data = ir2[2];
          ir2[2] = (v95_data + (v82_data * v93_data));
          double v98_data = s0[49];
          double v100_data = ir2[3];
          ir2[3] = (v100_data + (v82_data * v98_data));
          double v103_data = s0[65];
          double v105_data = ir2[4];
          ir2[4] = (v105_data + (v82_data * v103_data));
          double v108_data = s0[81];
          double v110_data = ir2[5];
          ir2[5] = (v110_data + (v82_data * v108_data));
          double v113_data = s0[97];
          double v115_data = ir2[6];
          ir2[6] = (v115_data + (v82_data * v113_data));
          double v118_data = s0[113];
          double v120_data = ir2[7];
          ir2[7] = (v120_data + (v82_data * v118_data));
          double v122_data = r0[2];
          double v123_data = s0[2];
          double v125_data = ir2[0];
          ir2[0] = (v125_data + (v122_data * v123_data));
          double v128_data = s0[18];
          double v130_data = ir2[1];
          ir2[1] = (v130_data + (v122_data * v128_data));
          double v133_data = s0[34];
          double v135_data = ir2[2];
          ir2[2] = (v135_data + (v122_data * v133_data));
          double v138_data = s0[50];
          double v140_data = ir2[3];
          ir2[3] = (v140_data + (v122_data * v138_data));
          double v143_data = s0[66];
          double v145_data = ir2[4];
          ir2[4] = (v145_data + (v122_data * v143_data));
          double v148_data = s0[82];
          double v150_data = ir2[5];
          ir2[5] = (v150_data + (v122_data * v148_data));
          double v153_data = s0[98];
          double v155_data = ir2[6];
          ir2[6] = (v155_data + (v122_data * v153_data));
          double v158_data = s0[114];
          double v160_data = ir2[7];
          ir2[7] = (v160_data + (v122_data * v158_data));
          double v162_data = r0[3];
          double v163_data = s0[3];
          double v165_data = ir2[0];
          ir2[0] = (v165_data + (v162_data * v163_data));
          double v168_data = s0[19];
          double v170_data = ir2[1];
          ir2[1] = (v170_data + (v162_data * v168_data));
          double v173_data = s0[35];
          double v175_data = ir2[2];
          ir2[2] = (v175_data + (v162_data * v173_data));
          double v178_data = s0[51];
          double v180_data = ir2[3];
          ir2[3] = (v180_data + (v162_data * v178_data));
          double v183_data = s0[67];
          double v185_data = ir2[4];
          ir2[4] = (v185_data + (v162_data * v183_data));
          double v188_data = s0[83];
          double v190_data = ir2[5];
          ir2[5] = (v190_data + (v162_data * v188_data));
          double v193_data = s0[99];
          double v195_data = ir2[6];
          ir2[6] = (v195_data + (v162_data * v193_data));
          double v198_data = s0[115];
          double v200_data = ir2[7];
          ir2[7] = (v200_data + (v162_data * v198_data));
          double v202_data = r0[4];
          double v203_data = s0[4];
          double v205_data = ir2[0];
          ir2[0] = (v205_data + (v202_data * v203_data));
          double v208_data = s0[20];
          double v210_data = ir2[1];
          ir2[1] = (v210_data + (v202_data * v208_data));
          double v213_data = s0[36];
          double v215_data = ir2[2];
          ir2[2] = (v215_data + (v202_data * v213_data));
          double v218_data = s0[52];
          double v220_data = ir2[3];
          ir2[3] = (v220_data + (v202_data * v218_data));
          double v223_data = s0[68];
          double v225_data = ir2[4];
          ir2[4] = (v225_data + (v202_data * v223_data));
          double v228_data = s0[84];
          double v230_data = ir2[5];
          ir2[5] = (v230_data + (v202_data * v228_data));
          double v233_data = s0[100];
          double v235_data = ir2[6];
          ir2[6] = (v235_data + (v202_data * v233_data));
          double v238_data = s0[116];
          double v240_data = ir2[7];
          ir2[7] = (v240_data + (v202_data * v238_data));
          double v242_data = r0[5];
          double v243_data = s0[5];
          double v245_data = ir2[0];
          ir2[0] = (v245_data + (v242_data * v243_data));
          double v248_data = s0[21];
          double v250_data = ir2[1];
          ir2[1] = (v250_data + (v242_data * v248_data));
          double v253_data = s0[37];
          double v255_data = ir2[2];
          ir2[2] = (v255_data + (v242_data * v253_data));
          double v258_data = s0[53];
          double v260_data = ir2[3];
          ir2[3] = (v260_data + (v242_data * v258_data));
          double v263_data = s0[69];
          double v265_data = ir2[4];
          ir2[4] = (v265_data + (v242_data * v263_data));
          double v268_data = s0[85];
          double v270_data = ir2[5];
          ir2[5] = (v270_data + (v242_data * v268_data));
          double v273_data = s0[101];
          double v275_data = ir2[6];
          ir2[6] = (v275_data + (v242_data * v273_data));
          double v278_data = s0[117];
          double v280_data = ir2[7];
          ir2[7] = (v280_data + (v242_data * v278_data));
          double v282_data = r0[6];
          double v283_data = s0[6];
          double v285_data = ir2[0];
          ir2[0] = (v285_data + (v282_data * v283_data));
          double v288_data = s0[22];
          double v290_data = ir2[1];
          ir2[1] = (v290_data + (v282_data * v288_data));
          double v293_data = s0[38];
          double v295_data = ir2[2];
          ir2[2] = (v295_data + (v282_data * v293_data));
          double v298_data = s0[54];
          double v300_data = ir2[3];
          ir2[3] = (v300_data + (v282_data * v298_data));
          double v303_data = s0[70];
          double v305_data = ir2[4];
          ir2[4] = (v305_data + (v282_data * v303_data));
          double v308_data = s0[86];
          double v310_data = ir2[5];
          ir2[5] = (v310_data + (v282_data * v308_data));
          double v313_data = s0[102];
          double v315_data = ir2[6];
          ir2[6] = (v315_data + (v282_data * v313_data));
          double v318_data = s0[118];
          double v320_data = ir2[7];
          ir2[7] = (v320_data + (v282_data * v318_data));
          double v322_data = r0[7];
          double v323_data = s0[7];
          double v325_data = ir2[0];
          ir2[0] = (v325_data + (v322_data * v323_data));
          double v328_data = s0[23];
          double v330_data = ir2[1];
          ir2[1] = (v330_data + (v322_data * v328_data));
          double v333_data = s0[39];
          double v335_data = ir2[2];
          ir2[2] = (v335_data + (v322_data * v333_data));
          double v338_data = s0[55];
          double v340_data = ir2[3];
          ir2[3] = (v340_data + (v322_data * v338_data));
          double v343_data = s0[71];
          double v345_data = ir2[4];
          ir2[4] = (v345_data + (v322_data * v343_data));
          double v348_data = s0[87];
          double v350_data = ir2[5];
          ir2[5] = (v350_data + (v322_data * v348_data));
          double v353_data = s0[103];
          double v355_data = ir2[6];
          ir2[6] = (v355_data + (v322_data * v353_data));
          double v358_data = s0[119];
          double v360_data = ir2[7];
          ir2[7] = (v360_data + (v322_data * v358_data));
          double v362_data = r0[8];
          double v363_data = s0[8];
          double v365_data = ir2[0];
          ir2[0] = (v365_data + (v362_data * v363_data));
          double v368_data = s0[24];
          double v370_data = ir2[1];
          ir2[1] = (v370_data + (v362_data * v368_data));
          double v373_data = s0[40];
          double v375_data = ir2[2];
          ir2[2] = (v375_data + (v362_data * v373_data));
          double v378_data = s0[56];
          double v380_data = ir2[3];
          ir2[3] = (v380_data + (v362_data * v378_data));
          double v383_data = s0[72];
          double v385_data = ir2[4];
          ir2[4] = (v385_data + (v362_data * v383_data));
          double v388_data = s0[88];
          double v390_data = ir2[5];
          ir2[5] = (v390_data + (v362_data * v388_data));
          double v393_data = s0[104];
          double v395_data = ir2[6];
          ir2[6] = (v395_data + (v362_data * v393_data));
          double v398_data = s0[120];
          double v400_data = ir2[7];
          ir2[7] = (v400_data + (v362_data * v398_data));
          double v402_data = r0[9];
          double v403_data = s0[9];
          double v405_data = ir2[0];
          ir2[0] = (v405_data + (v402_data * v403_data));
          double v408_data = s0[25];
          double v410_data = ir2[1];
          ir2[1] = (v410_data + (v402_data * v408_data));
          double v413_data = s0[41];
          double v415_data = ir2[2];
          ir2[2] = (v415_data + (v402_data * v413_data));
          double v418_data = s0[57];
          double v420_data = ir2[3];
          ir2[3] = (v420_data + (v402_data * v418_data));
          double v423_data = s0[73];
          double v425_data = ir2[4];
          ir2[4] = (v425_data + (v402_data * v423_data));
          double v428_data = s0[89];
          double v430_data = ir2[5];
          ir2[5] = (v430_data + (v402_data * v428_data));
          double v433_data = s0[105];
          double v435_data = ir2[6];
          ir2[6] = (v435_data + (v402_data * v433_data));
          double v438_data = s0[121];
          double v440_data = ir2[7];
          ir2[7] = (v440_data + (v402_data * v438_data));
          double v442_data = r0[10];
          double v443_data = s0[10];
          double v445_data = ir2[0];
          ir2[0] = (v445_data + (v442_data * v443_data));
          double v448_data = s0[26];
          double v450_data = ir2[1];
          ir2[1] = (v450_data + (v442_data * v448_data));
          double v453_data = s0[42];
          double v455_data = ir2[2];
          ir2[2] = (v455_data + (v442_data * v453_data));
          double v458_data = s0[58];
          double v460_data = ir2[3];
          ir2[3] = (v460_data + (v442_data * v458_data));
          double v463_data = s0[74];
          double v465_data = ir2[4];
          ir2[4] = (v465_data + (v442_data * v463_data));
          double v468_data = s0[90];
          double v470_data = ir2[5];
          ir2[5] = (v470_data + (v442_data * v468_data));
          double v473_data = s0[106];
          double v475_data = ir2[6];
          ir2[6] = (v475_data + (v442_data * v473_data));
          double v478_data = s0[122];
          double v480_data = ir2[7];
          ir2[7] = (v480_data + (v442_data * v478_data));
          double v482_data = r0[11];
          double v483_data = s0[11];
          double v485_data = ir2[0];
          ir2[0] = (v485_data + (v482_data * v483_data));
          double v488_data = s0[27];
          double v490_data = ir2[1];
          ir2[1] = (v490_data + (v482_data * v488_data));
          double v493_data = s0[43];
          double v495_data = ir2[2];
          ir2[2] = (v495_data + (v482_data * v493_data));
          double v498_data = s0[59];
          double v500_data = ir2[3];
          ir2[3] = (v500_data + (v482_data * v498_data));
          double v503_data = s0[75];
          double v505_data = ir2[4];
          ir2[4] = (v505_data + (v482_data * v503_data));
          double v508_data = s0[91];
          double v510_data = ir2[5];
          ir2[5] = (v510_data + (v482_data * v508_data));
          double v513_data = s0[107];
          double v515_data = ir2[6];
          ir2[6] = (v515_data + (v482_data * v513_data));
          double v518_data = s0[123];
          double v520_data = ir2[7];
          ir2[7] = (v520_data + (v482_data * v518_data));
          double v522_data = r0[12];
          double v523_data = s0[12];
          double v525_data = ir2[0];
          ir2[0] = (v525_data + (v522_data * v523_data));
          double v528_data = s0[28];
          double v530_data = ir2[1];
          ir2[1] = (v530_data + (v522_data * v528_data));
          double v533_data = s0[44];
          double v535_data = ir2[2];
          ir2[2] = (v535_data + (v522_data * v533_data));
          double v538_data = s0[60];
          double v540_data = ir2[3];
          ir2[3] = (v540_data + (v522_data * v538_data));
          double v543_data = s0[76];
          double v545_data = ir2[4];
          ir2[4] = (v545_data + (v522_data * v543_data));
          double v548_data = s0[92];
          double v550_data = ir2[5];
          ir2[5] = (v550_data + (v522_data * v548_data));
          double v553_data = s0[108];
          double v555_data = ir2[6];
          ir2[6] = (v555_data + (v522_data * v553_data));
          double v558_data = s0[124];
          double v560_data = ir2[7];
          ir2[7] = (v560_data + (v522_data * v558_data));
          double v562_data = r0[13];
          double v563_data = s0[13];
          double v565_data = ir2[0];
          ir2[0] = (v565_data + (v562_data * v563_data));
          double v568_data = s0[29];
          double v570_data = ir2[1];
          ir2[1] = (v570_data + (v562_data * v568_data));
          double v573_data = s0[45];
          double v575_data = ir2[2];
          ir2[2] = (v575_data + (v562_data * v573_data));
          double v578_data = s0[61];
          double v580_data = ir2[3];
          ir2[3] = (v580_data + (v562_data * v578_data));
          double v583_data = s0[77];
          double v585_data = ir2[4];
          ir2[4] = (v585_data + (v562_data * v583_data));
          double v588_data = s0[93];
          double v590_data = ir2[5];
          ir2[5] = (v590_data + (v562_data * v588_data));
          double v593_data = s0[109];
          double v595_data = ir2[6];
          ir2[6] = (v595_data + (v562_data * v593_data));
          double v598_data = s0[125];
          double v600_data = ir2[7];
          ir2[7] = (v600_data + (v562_data * v598_data));
          double v602_data = r0[14];
          double v603_data = s0[14];
          double v605_data = ir2[0];
          ir2[0] = (v605_data + (v602_data * v603_data));
          double v608_data = s0[30];
          double v610_data = ir2[1];
          ir2[1] = (v610_data + (v602_data * v608_data));
          double v613_data = s0[46];
          double v615_data = ir2[2];
          ir2[2] = (v615_data + (v602_data * v613_data));
          double v618_data = s0[62];
          double v620_data = ir2[3];
          ir2[3] = (v620_data + (v602_data * v618_data));
          double v623_data = s0[78];
          double v625_data = ir2[4];
          ir2[4] = (v625_data + (v602_data * v623_data));
          double v628_data = s0[94];
          double v630_data = ir2[5];
          ir2[5] = (v630_data + (v602_data * v628_data));
          double v633_data = s0[110];
          double v635_data = ir2[6];
          ir2[6] = (v635_data + (v602_data * v633_data));
          double v638_data = s0[126];
          double v640_data = ir2[7];
          ir2[7] = (v640_data + (v602_data * v638_data));
          double v642_data = r0[15];
          double v643_data = s0[15];
          double v645_data = ir2[0];
          ir2[0] = (v645_data + (v642_data * v643_data));
          double v648_data = s0[31];
          double v650_data = ir2[1];
          ir2[1] = (v650_data + (v642_data * v648_data));
          double v653_data = s0[47];
          double v655_data = ir2[2];
          ir2[2] = (v655_data + (v642_data * v653_data));
          double v658_data = s0[63];
          double v660_data = ir2[3];
          ir2[3] = (v660_data + (v642_data * v658_data));
          double v663_data = s0[79];
          double v665_data = ir2[4];
          ir2[4] = (v665_data + (v642_data * v663_data));
          double v668_data = s0[95];
          double v670_data = ir2[5];
          ir2[5] = (v670_data + (v642_data * v668_data));
          double v673_data = s0[111];
          double v675_data = ir2[6];
          ir2[6] = (v675_data + (v642_data * v673_data));
          double v678_data = s0[127];
          double v680_data = ir2[7];
          ir2[7] = (v680_data + (v642_data * v678_data));
          // r2 = ir2 + r1
          if (v23_g) {
            #pragma unroll
            for (int32_t v682_n1 = 0; v682_n1 < 8; ++v682_n1) {
              double v684_data = ir2[v682_n1];
              double v685_data = r1[v682_n1];
              r2[v682_n1] = (v685_data + v684_data);
            }
          }
          // glb_m0 = store{r>g}(r2);
          if (v23_g) {
            #pragma unroll
            for (int32_t v687_i1 = 0; v687_i1 < 8; ++v687_i1) {
              double v689_data = r2[v687_i1];
              glb_m0[(v22_lead + (v687_i1 * 12))] = v689_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

