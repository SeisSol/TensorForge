// === base name ===
kernel_74b76b72de381f4c

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_74b76b72de381f4c = {{16, 8, 1}, 16, 12, 1, 8, 9216, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_74b76b72de381f4c(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_74b76b72de381f4c(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_74b76b72de381f4c(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_74b76b72de381f4c, block.x * block.y * block.z, 1152 * sizeof(double));
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
void launcher_kernel_74b76b72de381f4c(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_74b76b72de381f4c(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_74b76b72de381f4c, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_74b76b72de381f4c<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_74b76b72de381f4c(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 8 per block = block 16x8x1, 9216 B shared, occupancy grid
    // operands:
    //   m0 12×8(12×8) {0..12}×{0..8} strided
    //   m1 12×16(12×16) {0..12}×{0..16} strided
    //   m2 16×8(16×8) {0..16}×{0..8} strided
    // operations:
    //   m0[i,j] += m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"double","launch":{"active_threads":12,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1152}],"shared_bytes":9216,"shared_elements":1152,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,8]],"name":"m0","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,16]],"name":"m1","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,16]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[144 * threadIdx.y + 0];
      double* tempShrMem = &localShrMem0[128];
      double * __restrict__ s0 = &localShrMem0[0];
      for (size_t v5_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v5_batchId0 < numElements0; v5_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v6_ahead1 = v5_batchId0 + (gridDim.x * blockDim.y);
        size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[v5_batchId0 * 96 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m1 = &m1[v5_batchId0 * 192 + 0 + m1_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[v5_batchId0 * 128 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v19_lead = threadIdx.x % 16;
          bool v20_g = v19_lead < 12;
          if (v20_g) {
            #pragma unroll
            for (int32_t v21_i1 = 0; v21_i1 < 16; ++v21_i1) {
              double v26_data = __ldcg(&glb_m1[(v19_lead + (v21_i1 * 12))]);
              r0[v21_i1] = v26_data;
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
          if (v20_g) {
            #pragma unroll
            for (int32_t v30_i1 = 0; v30_i1 < 8; ++v30_i1) {
              double v35_data = glb_m0[(v19_lead + (v30_i1 * 12))];
              r1[v30_i1] = v35_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          // wait(r1 = load{g>r}(glb_m0););
          double r2[8]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // r2 = +(r0 * s0) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 12), (0, 8)] [(0, 16)]
          double ir2[8]{};
          double v39_data = r0[0];
          double v40_data = s0[0];
          double v42_data = ir2[0];
          ir2[0] = (v42_data + (v39_data * v40_data));
          double v45_data = s0[16];
          double v47_data = ir2[1];
          ir2[1] = (v47_data + (v39_data * v45_data));
          double v50_data = s0[32];
          double v52_data = ir2[2];
          ir2[2] = (v52_data + (v39_data * v50_data));
          double v55_data = s0[48];
          double v57_data = ir2[3];
          ir2[3] = (v57_data + (v39_data * v55_data));
          double v60_data = s0[64];
          double v62_data = ir2[4];
          ir2[4] = (v62_data + (v39_data * v60_data));
          double v65_data = s0[80];
          double v67_data = ir2[5];
          ir2[5] = (v67_data + (v39_data * v65_data));
          double v70_data = s0[96];
          double v72_data = ir2[6];
          ir2[6] = (v72_data + (v39_data * v70_data));
          double v75_data = s0[112];
          double v77_data = ir2[7];
          ir2[7] = (v77_data + (v39_data * v75_data));
          double v79_data = r0[1];
          double v80_data = s0[1];
          double v82_data = ir2[0];
          ir2[0] = (v82_data + (v79_data * v80_data));
          double v85_data = s0[17];
          double v87_data = ir2[1];
          ir2[1] = (v87_data + (v79_data * v85_data));
          double v90_data = s0[33];
          double v92_data = ir2[2];
          ir2[2] = (v92_data + (v79_data * v90_data));
          double v95_data = s0[49];
          double v97_data = ir2[3];
          ir2[3] = (v97_data + (v79_data * v95_data));
          double v100_data = s0[65];
          double v102_data = ir2[4];
          ir2[4] = (v102_data + (v79_data * v100_data));
          double v105_data = s0[81];
          double v107_data = ir2[5];
          ir2[5] = (v107_data + (v79_data * v105_data));
          double v110_data = s0[97];
          double v112_data = ir2[6];
          ir2[6] = (v112_data + (v79_data * v110_data));
          double v115_data = s0[113];
          double v117_data = ir2[7];
          ir2[7] = (v117_data + (v79_data * v115_data));
          double v119_data = r0[2];
          double v120_data = s0[2];
          double v122_data = ir2[0];
          ir2[0] = (v122_data + (v119_data * v120_data));
          double v125_data = s0[18];
          double v127_data = ir2[1];
          ir2[1] = (v127_data + (v119_data * v125_data));
          double v130_data = s0[34];
          double v132_data = ir2[2];
          ir2[2] = (v132_data + (v119_data * v130_data));
          double v135_data = s0[50];
          double v137_data = ir2[3];
          ir2[3] = (v137_data + (v119_data * v135_data));
          double v140_data = s0[66];
          double v142_data = ir2[4];
          ir2[4] = (v142_data + (v119_data * v140_data));
          double v145_data = s0[82];
          double v147_data = ir2[5];
          ir2[5] = (v147_data + (v119_data * v145_data));
          double v150_data = s0[98];
          double v152_data = ir2[6];
          ir2[6] = (v152_data + (v119_data * v150_data));
          double v155_data = s0[114];
          double v157_data = ir2[7];
          ir2[7] = (v157_data + (v119_data * v155_data));
          double v159_data = r0[3];
          double v160_data = s0[3];
          double v162_data = ir2[0];
          ir2[0] = (v162_data + (v159_data * v160_data));
          double v165_data = s0[19];
          double v167_data = ir2[1];
          ir2[1] = (v167_data + (v159_data * v165_data));
          double v170_data = s0[35];
          double v172_data = ir2[2];
          ir2[2] = (v172_data + (v159_data * v170_data));
          double v175_data = s0[51];
          double v177_data = ir2[3];
          ir2[3] = (v177_data + (v159_data * v175_data));
          double v180_data = s0[67];
          double v182_data = ir2[4];
          ir2[4] = (v182_data + (v159_data * v180_data));
          double v185_data = s0[83];
          double v187_data = ir2[5];
          ir2[5] = (v187_data + (v159_data * v185_data));
          double v190_data = s0[99];
          double v192_data = ir2[6];
          ir2[6] = (v192_data + (v159_data * v190_data));
          double v195_data = s0[115];
          double v197_data = ir2[7];
          ir2[7] = (v197_data + (v159_data * v195_data));
          double v199_data = r0[4];
          double v200_data = s0[4];
          double v202_data = ir2[0];
          ir2[0] = (v202_data + (v199_data * v200_data));
          double v205_data = s0[20];
          double v207_data = ir2[1];
          ir2[1] = (v207_data + (v199_data * v205_data));
          double v210_data = s0[36];
          double v212_data = ir2[2];
          ir2[2] = (v212_data + (v199_data * v210_data));
          double v215_data = s0[52];
          double v217_data = ir2[3];
          ir2[3] = (v217_data + (v199_data * v215_data));
          double v220_data = s0[68];
          double v222_data = ir2[4];
          ir2[4] = (v222_data + (v199_data * v220_data));
          double v225_data = s0[84];
          double v227_data = ir2[5];
          ir2[5] = (v227_data + (v199_data * v225_data));
          double v230_data = s0[100];
          double v232_data = ir2[6];
          ir2[6] = (v232_data + (v199_data * v230_data));
          double v235_data = s0[116];
          double v237_data = ir2[7];
          ir2[7] = (v237_data + (v199_data * v235_data));
          double v239_data = r0[5];
          double v240_data = s0[5];
          double v242_data = ir2[0];
          ir2[0] = (v242_data + (v239_data * v240_data));
          double v245_data = s0[21];
          double v247_data = ir2[1];
          ir2[1] = (v247_data + (v239_data * v245_data));
          double v250_data = s0[37];
          double v252_data = ir2[2];
          ir2[2] = (v252_data + (v239_data * v250_data));
          double v255_data = s0[53];
          double v257_data = ir2[3];
          ir2[3] = (v257_data + (v239_data * v255_data));
          double v260_data = s0[69];
          double v262_data = ir2[4];
          ir2[4] = (v262_data + (v239_data * v260_data));
          double v265_data = s0[85];
          double v267_data = ir2[5];
          ir2[5] = (v267_data + (v239_data * v265_data));
          double v270_data = s0[101];
          double v272_data = ir2[6];
          ir2[6] = (v272_data + (v239_data * v270_data));
          double v275_data = s0[117];
          double v277_data = ir2[7];
          ir2[7] = (v277_data + (v239_data * v275_data));
          double v279_data = r0[6];
          double v280_data = s0[6];
          double v282_data = ir2[0];
          ir2[0] = (v282_data + (v279_data * v280_data));
          double v285_data = s0[22];
          double v287_data = ir2[1];
          ir2[1] = (v287_data + (v279_data * v285_data));
          double v290_data = s0[38];
          double v292_data = ir2[2];
          ir2[2] = (v292_data + (v279_data * v290_data));
          double v295_data = s0[54];
          double v297_data = ir2[3];
          ir2[3] = (v297_data + (v279_data * v295_data));
          double v300_data = s0[70];
          double v302_data = ir2[4];
          ir2[4] = (v302_data + (v279_data * v300_data));
          double v305_data = s0[86];
          double v307_data = ir2[5];
          ir2[5] = (v307_data + (v279_data * v305_data));
          double v310_data = s0[102];
          double v312_data = ir2[6];
          ir2[6] = (v312_data + (v279_data * v310_data));
          double v315_data = s0[118];
          double v317_data = ir2[7];
          ir2[7] = (v317_data + (v279_data * v315_data));
          double v319_data = r0[7];
          double v320_data = s0[7];
          double v322_data = ir2[0];
          ir2[0] = (v322_data + (v319_data * v320_data));
          double v325_data = s0[23];
          double v327_data = ir2[1];
          ir2[1] = (v327_data + (v319_data * v325_data));
          double v330_data = s0[39];
          double v332_data = ir2[2];
          ir2[2] = (v332_data + (v319_data * v330_data));
          double v335_data = s0[55];
          double v337_data = ir2[3];
          ir2[3] = (v337_data + (v319_data * v335_data));
          double v340_data = s0[71];
          double v342_data = ir2[4];
          ir2[4] = (v342_data + (v319_data * v340_data));
          double v345_data = s0[87];
          double v347_data = ir2[5];
          ir2[5] = (v347_data + (v319_data * v345_data));
          double v350_data = s0[103];
          double v352_data = ir2[6];
          ir2[6] = (v352_data + (v319_data * v350_data));
          double v355_data = s0[119];
          double v357_data = ir2[7];
          ir2[7] = (v357_data + (v319_data * v355_data));
          double v359_data = r0[8];
          double v360_data = s0[8];
          double v362_data = ir2[0];
          ir2[0] = (v362_data + (v359_data * v360_data));
          double v365_data = s0[24];
          double v367_data = ir2[1];
          ir2[1] = (v367_data + (v359_data * v365_data));
          double v370_data = s0[40];
          double v372_data = ir2[2];
          ir2[2] = (v372_data + (v359_data * v370_data));
          double v375_data = s0[56];
          double v377_data = ir2[3];
          ir2[3] = (v377_data + (v359_data * v375_data));
          double v380_data = s0[72];
          double v382_data = ir2[4];
          ir2[4] = (v382_data + (v359_data * v380_data));
          double v385_data = s0[88];
          double v387_data = ir2[5];
          ir2[5] = (v387_data + (v359_data * v385_data));
          double v390_data = s0[104];
          double v392_data = ir2[6];
          ir2[6] = (v392_data + (v359_data * v390_data));
          double v395_data = s0[120];
          double v397_data = ir2[7];
          ir2[7] = (v397_data + (v359_data * v395_data));
          double v399_data = r0[9];
          double v400_data = s0[9];
          double v402_data = ir2[0];
          ir2[0] = (v402_data + (v399_data * v400_data));
          double v405_data = s0[25];
          double v407_data = ir2[1];
          ir2[1] = (v407_data + (v399_data * v405_data));
          double v410_data = s0[41];
          double v412_data = ir2[2];
          ir2[2] = (v412_data + (v399_data * v410_data));
          double v415_data = s0[57];
          double v417_data = ir2[3];
          ir2[3] = (v417_data + (v399_data * v415_data));
          double v420_data = s0[73];
          double v422_data = ir2[4];
          ir2[4] = (v422_data + (v399_data * v420_data));
          double v425_data = s0[89];
          double v427_data = ir2[5];
          ir2[5] = (v427_data + (v399_data * v425_data));
          double v430_data = s0[105];
          double v432_data = ir2[6];
          ir2[6] = (v432_data + (v399_data * v430_data));
          double v435_data = s0[121];
          double v437_data = ir2[7];
          ir2[7] = (v437_data + (v399_data * v435_data));
          double v439_data = r0[10];
          double v440_data = s0[10];
          double v442_data = ir2[0];
          ir2[0] = (v442_data + (v439_data * v440_data));
          double v445_data = s0[26];
          double v447_data = ir2[1];
          ir2[1] = (v447_data + (v439_data * v445_data));
          double v450_data = s0[42];
          double v452_data = ir2[2];
          ir2[2] = (v452_data + (v439_data * v450_data));
          double v455_data = s0[58];
          double v457_data = ir2[3];
          ir2[3] = (v457_data + (v439_data * v455_data));
          double v460_data = s0[74];
          double v462_data = ir2[4];
          ir2[4] = (v462_data + (v439_data * v460_data));
          double v465_data = s0[90];
          double v467_data = ir2[5];
          ir2[5] = (v467_data + (v439_data * v465_data));
          double v470_data = s0[106];
          double v472_data = ir2[6];
          ir2[6] = (v472_data + (v439_data * v470_data));
          double v475_data = s0[122];
          double v477_data = ir2[7];
          ir2[7] = (v477_data + (v439_data * v475_data));
          double v479_data = r0[11];
          double v480_data = s0[11];
          double v482_data = ir2[0];
          ir2[0] = (v482_data + (v479_data * v480_data));
          double v485_data = s0[27];
          double v487_data = ir2[1];
          ir2[1] = (v487_data + (v479_data * v485_data));
          double v490_data = s0[43];
          double v492_data = ir2[2];
          ir2[2] = (v492_data + (v479_data * v490_data));
          double v495_data = s0[59];
          double v497_data = ir2[3];
          ir2[3] = (v497_data + (v479_data * v495_data));
          double v500_data = s0[75];
          double v502_data = ir2[4];
          ir2[4] = (v502_data + (v479_data * v500_data));
          double v505_data = s0[91];
          double v507_data = ir2[5];
          ir2[5] = (v507_data + (v479_data * v505_data));
          double v510_data = s0[107];
          double v512_data = ir2[6];
          ir2[6] = (v512_data + (v479_data * v510_data));
          double v515_data = s0[123];
          double v517_data = ir2[7];
          ir2[7] = (v517_data + (v479_data * v515_data));
          double v519_data = r0[12];
          double v520_data = s0[12];
          double v522_data = ir2[0];
          ir2[0] = (v522_data + (v519_data * v520_data));
          double v525_data = s0[28];
          double v527_data = ir2[1];
          ir2[1] = (v527_data + (v519_data * v525_data));
          double v530_data = s0[44];
          double v532_data = ir2[2];
          ir2[2] = (v532_data + (v519_data * v530_data));
          double v535_data = s0[60];
          double v537_data = ir2[3];
          ir2[3] = (v537_data + (v519_data * v535_data));
          double v540_data = s0[76];
          double v542_data = ir2[4];
          ir2[4] = (v542_data + (v519_data * v540_data));
          double v545_data = s0[92];
          double v547_data = ir2[5];
          ir2[5] = (v547_data + (v519_data * v545_data));
          double v550_data = s0[108];
          double v552_data = ir2[6];
          ir2[6] = (v552_data + (v519_data * v550_data));
          double v555_data = s0[124];
          double v557_data = ir2[7];
          ir2[7] = (v557_data + (v519_data * v555_data));
          double v559_data = r0[13];
          double v560_data = s0[13];
          double v562_data = ir2[0];
          ir2[0] = (v562_data + (v559_data * v560_data));
          double v565_data = s0[29];
          double v567_data = ir2[1];
          ir2[1] = (v567_data + (v559_data * v565_data));
          double v570_data = s0[45];
          double v572_data = ir2[2];
          ir2[2] = (v572_data + (v559_data * v570_data));
          double v575_data = s0[61];
          double v577_data = ir2[3];
          ir2[3] = (v577_data + (v559_data * v575_data));
          double v580_data = s0[77];
          double v582_data = ir2[4];
          ir2[4] = (v582_data + (v559_data * v580_data));
          double v585_data = s0[93];
          double v587_data = ir2[5];
          ir2[5] = (v587_data + (v559_data * v585_data));
          double v590_data = s0[109];
          double v592_data = ir2[6];
          ir2[6] = (v592_data + (v559_data * v590_data));
          double v595_data = s0[125];
          double v597_data = ir2[7];
          ir2[7] = (v597_data + (v559_data * v595_data));
          double v599_data = r0[14];
          double v600_data = s0[14];
          double v602_data = ir2[0];
          ir2[0] = (v602_data + (v599_data * v600_data));
          double v605_data = s0[30];
          double v607_data = ir2[1];
          ir2[1] = (v607_data + (v599_data * v605_data));
          double v610_data = s0[46];
          double v612_data = ir2[2];
          ir2[2] = (v612_data + (v599_data * v610_data));
          double v615_data = s0[62];
          double v617_data = ir2[3];
          ir2[3] = (v617_data + (v599_data * v615_data));
          double v620_data = s0[78];
          double v622_data = ir2[4];
          ir2[4] = (v622_data + (v599_data * v620_data));
          double v625_data = s0[94];
          double v627_data = ir2[5];
          ir2[5] = (v627_data + (v599_data * v625_data));
          double v630_data = s0[110];
          double v632_data = ir2[6];
          ir2[6] = (v632_data + (v599_data * v630_data));
          double v635_data = s0[126];
          double v637_data = ir2[7];
          ir2[7] = (v637_data + (v599_data * v635_data));
          double v639_data = r0[15];
          double v640_data = s0[15];
          double v642_data = ir2[0];
          ir2[0] = (v642_data + (v639_data * v640_data));
          double v645_data = s0[31];
          double v647_data = ir2[1];
          ir2[1] = (v647_data + (v639_data * v645_data));
          double v650_data = s0[47];
          double v652_data = ir2[2];
          ir2[2] = (v652_data + (v639_data * v650_data));
          double v655_data = s0[63];
          double v657_data = ir2[3];
          ir2[3] = (v657_data + (v639_data * v655_data));
          double v660_data = s0[79];
          double v662_data = ir2[4];
          ir2[4] = (v662_data + (v639_data * v660_data));
          double v665_data = s0[95];
          double v667_data = ir2[5];
          ir2[5] = (v667_data + (v639_data * v665_data));
          double v670_data = s0[111];
          double v672_data = ir2[6];
          ir2[6] = (v672_data + (v639_data * v670_data));
          double v675_data = s0[127];
          double v677_data = ir2[7];
          ir2[7] = (v677_data + (v639_data * v675_data));
          if (v20_g) {
            #pragma unroll
            for (int32_t v679_n1 = 0; v679_n1 < 8; ++v679_n1) {
              double v681_data = ir2[v679_n1];
              double v682_data = r1[v679_n1];
              r2[v679_n1] = (v682_data + v681_data);
            }
          }
          // glb_m0 = store{r>g}(r2);
          if (v20_g) {
            #pragma unroll
            for (int32_t v684_i1 = 0; v684_i1 < 8; ++v684_i1) {
              double v686_data = r2[v684_i1];
              glb_m0[(v19_lead + (v684_i1 * 12))] = v686_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

