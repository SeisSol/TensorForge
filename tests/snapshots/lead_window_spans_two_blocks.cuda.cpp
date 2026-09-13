// === base name ===
kernel_f69df16dce9ee9ad

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_f69df16dce9ee9ad = {{32, 4, 1}, 32, 64, 1, 4, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_f69df16dce9ee9ad(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_f69df16dce9ee9ad(const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_f69df16dce9ee9ad(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_f69df16dce9ee9ad, block.x * block.y * block.z, 0 * sizeof(float));
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
  config.sharedMemBytes = 0 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_f69df16dce9ee9ad(const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_f69df16dce9ee9ad(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_f69df16dce9ee9ad, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_f69df16dce9ee9ad<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_f69df16dce9ee9ad(const float ** m0, size_t m0_extraOffset, const float * m1, float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (64 active) x 4 per block = block 32x4x1, 0 B shared, occupancy grid
    // operands:
    //   m0 64×13(64×13) {0..64}×{0..13} pointer_based
    //   m1 6(6) {0..6} none
    //   m2 64×13×6(64×13×6) {0..64}×{0..13}×{0..6} pointer_based
    // operations:
    //   t0[i,j,l] = m0[i,j] × m1[l]
    //   m2[i,j,l]@{20..35}×{12..13}×{0..6} += t0[i,j,l]@{20..35}×{12..13}×{0..6}
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":64,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"pointer_based","alias":"A","bbox":[[0,0],[64,13]],"name":"m0","ordered":false,"parts":1,"shape":[64,13],"variant":false},{"addressing":"none","alias":"v","bbox":[[0],[6]],"name":"m1","ordered":false,"parts":1,"shape":[6],"variant":false},{"addressing":"pointer_based","alias":"D","bbox":[[0,0,0],[64,13,6]],"name":"m2","ordered":false,"parts":1,"shape":[64,13,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0,0],[64,13,6]],"is_tmp":true,"name":"t0","offset":[0,0,0],"shape":[64,13,6]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[64,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[64,13]},{"addressing":"none","bbox":[[0],[6]],"is_tmp":false,"name":"m1","offset":[0],"shape":[6]}],"permute":[[0,1],[0]],"target":[[0,1],[2]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0,0],[15,1,6]],"is_tmp":false,"name":"m2","offset":[20,12,0],"shape":[64,13,6]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0,0],[15,1,6]],"is_tmp":true,"name":"t0","offset":[20,12,0],"shape":[64,13,6]}],"permute":[[0,1,2]],"target":[[0,1,2]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      const float *const __restrict__ glb_m1 = &m1[0];
      for (size_t v2_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v2_batchId0 < numElements0; v2_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v3_ahead1 = v2_batchId0 + (gridDim.x * blockDim.y);
        size_t v5_batchId1 = (v3_ahead1 < numElements0) ? v3_ahead1 : v2_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v2_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v2_batchId0][0 + m0_extraOffset];
          float *const __restrict__ glb_m2 = &m2[v2_batchId0][0 + m2_extraOffset];
          float r0[26]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v15_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v16_i0 = 0; v16_i0 < 2; ++v16_i0) {
            int32_t v19_lead = v15_lead + (v16_i0 * 32);
            #pragma unroll
            for (int32_t v17_i1 = 0; v17_i1 < 13; ++v17_i1) {
              float v22_data = __ldcg(&glb_m0[(v19_lead + (v17_i1 * 64))]);
              r0[(v16_i0 + (v17_i1 * 2))] = v22_data;
            }
          }
          float r2[12]{};
          // r2 = load{g>r}(glb_m2);
          bool v26_g = v15_lead >= 20;
          if (v26_g) {
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 1; ++v27_i1) {
              int32_t v34_a = v15_lead + ((v27_i1 + 12) * 64);
              int32_t v37_a = v27_i1 * 2;
              #pragma unroll
              for (int32_t v28_i2 = 0; v28_i2 < 6; ++v28_i2) {
                float v36_data = glb_m2[(v34_a + (v28_i2 * 832))];
                r2[(v37_a + (v28_i2 * 2))] = v36_data;
              }
            }
          }
          bool v41_g = v15_lead < 3;
          if (v41_g) {
            int32_t v45_lead = v15_lead + 32_i32;
            #pragma unroll
            for (int32_t v42_i1 = 0; v42_i1 < 1; ++v42_i1) {
              int32_t v49_a = v45_lead + ((v42_i1 + 12) * 64);
              int32_t v54_a = 1 + (v42_i1 * 2);
              #pragma unroll
              for (int32_t v43_i2 = 0; v43_i2 < 6; ++v43_i2) {
                float v51_data = glb_m2[(v49_a + (v43_i2 * 832))];
                r2[(v54_a + (v43_i2 * 2))] = v51_data;
              }
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[156]{};
          // r1 = +(r0 * glb_m1) + None
          // [(0, 64), (0, 13), (0, 6)] []
          float v57_data = r0[0];
          float v58_data = glb_m1[0];
          float v60_data = r1[0];
          r1[0] = (v60_data + (v57_data * v58_data));
          float v63_data = glb_m1[1];
          float v65_data = r1[26];
          r1[26] = (v65_data + (v57_data * v63_data));
          float v68_data = glb_m1[2];
          float v70_data = r1[52];
          r1[52] = (v70_data + (v57_data * v68_data));
          float v73_data = glb_m1[3];
          float v75_data = r1[78];
          r1[78] = (v75_data + (v57_data * v73_data));
          float v78_data = glb_m1[4];
          float v80_data = r1[104];
          r1[104] = (v80_data + (v57_data * v78_data));
          float v83_data = glb_m1[5];
          float v85_data = r1[130];
          r1[130] = (v85_data + (v57_data * v83_data));
          float v87_data = r0[2];
          float v90_data = r1[2];
          r1[2] = (v90_data + (v87_data * v58_data));
          float v95_data = r1[28];
          r1[28] = (v95_data + (v87_data * v63_data));
          float v100_data = r1[54];
          r1[54] = (v100_data + (v87_data * v68_data));
          float v105_data = r1[80];
          r1[80] = (v105_data + (v87_data * v73_data));
          float v110_data = r1[106];
          r1[106] = (v110_data + (v87_data * v78_data));
          float v115_data = r1[132];
          r1[132] = (v115_data + (v87_data * v83_data));
          float v117_data = r0[4];
          float v120_data = r1[4];
          r1[4] = (v120_data + (v117_data * v58_data));
          float v125_data = r1[30];
          r1[30] = (v125_data + (v117_data * v63_data));
          float v130_data = r1[56];
          r1[56] = (v130_data + (v117_data * v68_data));
          float v135_data = r1[82];
          r1[82] = (v135_data + (v117_data * v73_data));
          float v140_data = r1[108];
          r1[108] = (v140_data + (v117_data * v78_data));
          float v145_data = r1[134];
          r1[134] = (v145_data + (v117_data * v83_data));
          float v147_data = r0[6];
          float v150_data = r1[6];
          r1[6] = (v150_data + (v147_data * v58_data));
          float v155_data = r1[32];
          r1[32] = (v155_data + (v147_data * v63_data));
          float v160_data = r1[58];
          r1[58] = (v160_data + (v147_data * v68_data));
          float v165_data = r1[84];
          r1[84] = (v165_data + (v147_data * v73_data));
          float v170_data = r1[110];
          r1[110] = (v170_data + (v147_data * v78_data));
          float v175_data = r1[136];
          r1[136] = (v175_data + (v147_data * v83_data));
          float v177_data = r0[8];
          float v180_data = r1[8];
          r1[8] = (v180_data + (v177_data * v58_data));
          float v185_data = r1[34];
          r1[34] = (v185_data + (v177_data * v63_data));
          float v190_data = r1[60];
          r1[60] = (v190_data + (v177_data * v68_data));
          float v195_data = r1[86];
          r1[86] = (v195_data + (v177_data * v73_data));
          float v200_data = r1[112];
          r1[112] = (v200_data + (v177_data * v78_data));
          float v205_data = r1[138];
          r1[138] = (v205_data + (v177_data * v83_data));
          float v207_data = r0[10];
          float v210_data = r1[10];
          r1[10] = (v210_data + (v207_data * v58_data));
          float v215_data = r1[36];
          r1[36] = (v215_data + (v207_data * v63_data));
          float v220_data = r1[62];
          r1[62] = (v220_data + (v207_data * v68_data));
          float v225_data = r1[88];
          r1[88] = (v225_data + (v207_data * v73_data));
          float v230_data = r1[114];
          r1[114] = (v230_data + (v207_data * v78_data));
          float v235_data = r1[140];
          r1[140] = (v235_data + (v207_data * v83_data));
          float v237_data = r0[12];
          float v240_data = r1[12];
          r1[12] = (v240_data + (v237_data * v58_data));
          float v245_data = r1[38];
          r1[38] = (v245_data + (v237_data * v63_data));
          float v250_data = r1[64];
          r1[64] = (v250_data + (v237_data * v68_data));
          float v255_data = r1[90];
          r1[90] = (v255_data + (v237_data * v73_data));
          float v260_data = r1[116];
          r1[116] = (v260_data + (v237_data * v78_data));
          float v265_data = r1[142];
          r1[142] = (v265_data + (v237_data * v83_data));
          float v267_data = r0[14];
          float v270_data = r1[14];
          r1[14] = (v270_data + (v267_data * v58_data));
          float v275_data = r1[40];
          r1[40] = (v275_data + (v267_data * v63_data));
          float v280_data = r1[66];
          r1[66] = (v280_data + (v267_data * v68_data));
          float v285_data = r1[92];
          r1[92] = (v285_data + (v267_data * v73_data));
          float v290_data = r1[118];
          r1[118] = (v290_data + (v267_data * v78_data));
          float v295_data = r1[144];
          r1[144] = (v295_data + (v267_data * v83_data));
          float v297_data = r0[16];
          float v300_data = r1[16];
          r1[16] = (v300_data + (v297_data * v58_data));
          float v305_data = r1[42];
          r1[42] = (v305_data + (v297_data * v63_data));
          float v310_data = r1[68];
          r1[68] = (v310_data + (v297_data * v68_data));
          float v315_data = r1[94];
          r1[94] = (v315_data + (v297_data * v73_data));
          float v320_data = r1[120];
          r1[120] = (v320_data + (v297_data * v78_data));
          float v325_data = r1[146];
          r1[146] = (v325_data + (v297_data * v83_data));
          float v327_data = r0[18];
          float v330_data = r1[18];
          r1[18] = (v330_data + (v327_data * v58_data));
          float v335_data = r1[44];
          r1[44] = (v335_data + (v327_data * v63_data));
          float v340_data = r1[70];
          r1[70] = (v340_data + (v327_data * v68_data));
          float v345_data = r1[96];
          r1[96] = (v345_data + (v327_data * v73_data));
          float v350_data = r1[122];
          r1[122] = (v350_data + (v327_data * v78_data));
          float v355_data = r1[148];
          r1[148] = (v355_data + (v327_data * v83_data));
          float v357_data = r0[20];
          float v360_data = r1[20];
          r1[20] = (v360_data + (v357_data * v58_data));
          float v365_data = r1[46];
          r1[46] = (v365_data + (v357_data * v63_data));
          float v370_data = r1[72];
          r1[72] = (v370_data + (v357_data * v68_data));
          float v375_data = r1[98];
          r1[98] = (v375_data + (v357_data * v73_data));
          float v380_data = r1[124];
          r1[124] = (v380_data + (v357_data * v78_data));
          float v385_data = r1[150];
          r1[150] = (v385_data + (v357_data * v83_data));
          float v387_data = r0[22];
          float v390_data = r1[22];
          r1[22] = (v390_data + (v387_data * v58_data));
          float v395_data = r1[48];
          r1[48] = (v395_data + (v387_data * v63_data));
          float v400_data = r1[74];
          r1[74] = (v400_data + (v387_data * v68_data));
          float v405_data = r1[100];
          r1[100] = (v405_data + (v387_data * v73_data));
          float v410_data = r1[126];
          r1[126] = (v410_data + (v387_data * v78_data));
          float v415_data = r1[152];
          r1[152] = (v415_data + (v387_data * v83_data));
          float v417_data = r0[24];
          float v420_data = r1[24];
          r1[24] = (v420_data + (v417_data * v58_data));
          float v425_data = r1[50];
          r1[50] = (v425_data + (v417_data * v63_data));
          float v430_data = r1[76];
          r1[76] = (v430_data + (v417_data * v68_data));
          float v435_data = r1[102];
          r1[102] = (v435_data + (v417_data * v73_data));
          float v440_data = r1[128];
          r1[128] = (v440_data + (v417_data * v78_data));
          float v445_data = r1[154];
          r1[154] = (v445_data + (v417_data * v83_data));
          float v447_data = r0[1];
          float v450_data = r1[1];
          r1[1] = (v450_data + (v447_data * v58_data));
          float v455_data = r1[27];
          r1[27] = (v455_data + (v447_data * v63_data));
          float v460_data = r1[53];
          r1[53] = (v460_data + (v447_data * v68_data));
          float v465_data = r1[79];
          r1[79] = (v465_data + (v447_data * v73_data));
          float v470_data = r1[105];
          r1[105] = (v470_data + (v447_data * v78_data));
          float v475_data = r1[131];
          r1[131] = (v475_data + (v447_data * v83_data));
          float v477_data = r0[3];
          float v480_data = r1[3];
          r1[3] = (v480_data + (v477_data * v58_data));
          float v485_data = r1[29];
          r1[29] = (v485_data + (v477_data * v63_data));
          float v490_data = r1[55];
          r1[55] = (v490_data + (v477_data * v68_data));
          float v495_data = r1[81];
          r1[81] = (v495_data + (v477_data * v73_data));
          float v500_data = r1[107];
          r1[107] = (v500_data + (v477_data * v78_data));
          float v505_data = r1[133];
          r1[133] = (v505_data + (v477_data * v83_data));
          float v507_data = r0[5];
          float v510_data = r1[5];
          r1[5] = (v510_data + (v507_data * v58_data));
          float v515_data = r1[31];
          r1[31] = (v515_data + (v507_data * v63_data));
          float v520_data = r1[57];
          r1[57] = (v520_data + (v507_data * v68_data));
          float v525_data = r1[83];
          r1[83] = (v525_data + (v507_data * v73_data));
          float v530_data = r1[109];
          r1[109] = (v530_data + (v507_data * v78_data));
          float v535_data = r1[135];
          r1[135] = (v535_data + (v507_data * v83_data));
          float v537_data = r0[7];
          float v540_data = r1[7];
          r1[7] = (v540_data + (v537_data * v58_data));
          float v545_data = r1[33];
          r1[33] = (v545_data + (v537_data * v63_data));
          float v550_data = r1[59];
          r1[59] = (v550_data + (v537_data * v68_data));
          float v555_data = r1[85];
          r1[85] = (v555_data + (v537_data * v73_data));
          float v560_data = r1[111];
          r1[111] = (v560_data + (v537_data * v78_data));
          float v565_data = r1[137];
          r1[137] = (v565_data + (v537_data * v83_data));
          float v567_data = r0[9];
          float v570_data = r1[9];
          r1[9] = (v570_data + (v567_data * v58_data));
          float v575_data = r1[35];
          r1[35] = (v575_data + (v567_data * v63_data));
          float v580_data = r1[61];
          r1[61] = (v580_data + (v567_data * v68_data));
          float v585_data = r1[87];
          r1[87] = (v585_data + (v567_data * v73_data));
          float v590_data = r1[113];
          r1[113] = (v590_data + (v567_data * v78_data));
          float v595_data = r1[139];
          r1[139] = (v595_data + (v567_data * v83_data));
          float v597_data = r0[11];
          float v600_data = r1[11];
          r1[11] = (v600_data + (v597_data * v58_data));
          float v605_data = r1[37];
          r1[37] = (v605_data + (v597_data * v63_data));
          float v610_data = r1[63];
          r1[63] = (v610_data + (v597_data * v68_data));
          float v615_data = r1[89];
          r1[89] = (v615_data + (v597_data * v73_data));
          float v620_data = r1[115];
          r1[115] = (v620_data + (v597_data * v78_data));
          float v625_data = r1[141];
          r1[141] = (v625_data + (v597_data * v83_data));
          float v627_data = r0[13];
          float v630_data = r1[13];
          r1[13] = (v630_data + (v627_data * v58_data));
          float v635_data = r1[39];
          r1[39] = (v635_data + (v627_data * v63_data));
          float v640_data = r1[65];
          r1[65] = (v640_data + (v627_data * v68_data));
          float v645_data = r1[91];
          r1[91] = (v645_data + (v627_data * v73_data));
          float v650_data = r1[117];
          r1[117] = (v650_data + (v627_data * v78_data));
          float v655_data = r1[143];
          r1[143] = (v655_data + (v627_data * v83_data));
          float v657_data = r0[15];
          float v660_data = r1[15];
          r1[15] = (v660_data + (v657_data * v58_data));
          float v665_data = r1[41];
          r1[41] = (v665_data + (v657_data * v63_data));
          float v670_data = r1[67];
          r1[67] = (v670_data + (v657_data * v68_data));
          float v675_data = r1[93];
          r1[93] = (v675_data + (v657_data * v73_data));
          float v680_data = r1[119];
          r1[119] = (v680_data + (v657_data * v78_data));
          float v685_data = r1[145];
          r1[145] = (v685_data + (v657_data * v83_data));
          float v687_data = r0[17];
          float v690_data = r1[17];
          r1[17] = (v690_data + (v687_data * v58_data));
          float v695_data = r1[43];
          r1[43] = (v695_data + (v687_data * v63_data));
          float v700_data = r1[69];
          r1[69] = (v700_data + (v687_data * v68_data));
          float v705_data = r1[95];
          r1[95] = (v705_data + (v687_data * v73_data));
          float v710_data = r1[121];
          r1[121] = (v710_data + (v687_data * v78_data));
          float v715_data = r1[147];
          r1[147] = (v715_data + (v687_data * v83_data));
          float v717_data = r0[19];
          float v720_data = r1[19];
          r1[19] = (v720_data + (v717_data * v58_data));
          float v725_data = r1[45];
          r1[45] = (v725_data + (v717_data * v63_data));
          float v730_data = r1[71];
          r1[71] = (v730_data + (v717_data * v68_data));
          float v735_data = r1[97];
          r1[97] = (v735_data + (v717_data * v73_data));
          float v740_data = r1[123];
          r1[123] = (v740_data + (v717_data * v78_data));
          float v745_data = r1[149];
          r1[149] = (v745_data + (v717_data * v83_data));
          float v747_data = r0[21];
          float v750_data = r1[21];
          r1[21] = (v750_data + (v747_data * v58_data));
          float v755_data = r1[47];
          r1[47] = (v755_data + (v747_data * v63_data));
          float v760_data = r1[73];
          r1[73] = (v760_data + (v747_data * v68_data));
          float v765_data = r1[99];
          r1[99] = (v765_data + (v747_data * v73_data));
          float v770_data = r1[125];
          r1[125] = (v770_data + (v747_data * v78_data));
          float v775_data = r1[151];
          r1[151] = (v775_data + (v747_data * v83_data));
          float v777_data = r0[23];
          float v780_data = r1[23];
          r1[23] = (v780_data + (v777_data * v58_data));
          float v785_data = r1[49];
          r1[49] = (v785_data + (v777_data * v63_data));
          float v790_data = r1[75];
          r1[75] = (v790_data + (v777_data * v68_data));
          float v795_data = r1[101];
          r1[101] = (v795_data + (v777_data * v73_data));
          float v800_data = r1[127];
          r1[127] = (v800_data + (v777_data * v78_data));
          float v805_data = r1[153];
          r1[153] = (v805_data + (v777_data * v83_data));
          float v807_data = r0[25];
          float v810_data = r1[25];
          r1[25] = (v810_data + (v807_data * v58_data));
          float v815_data = r1[51];
          r1[51] = (v815_data + (v807_data * v63_data));
          float v820_data = r1[77];
          r1[77] = (v820_data + (v807_data * v68_data));
          float v825_data = r1[103];
          r1[103] = (v825_data + (v807_data * v73_data));
          float v830_data = r1[129];
          r1[129] = (v830_data + (v807_data * v78_data));
          float v835_data = r1[155];
          r1[155] = (v835_data + (v807_data * v83_data));
          // wait(r2 = load{g>r}(glb_m2););
          float r3[12]{};
          // r3 = +(r1) + name: r2, type: SymbolType.Register, lead: [0]
          // [(20, 35), (0, 1), (0, 6)] []
          float ir3[12]{};
          if (v26_g) {
            float v839_data = r1[24];
            float v840_data = ir3[0];
            ir3[0] = (v840_data + v839_data);
            float v842_data = r1[50];
            float v843_data = ir3[2];
            ir3[2] = (v843_data + v842_data);
            float v845_data = r1[76];
            float v846_data = ir3[4];
            ir3[4] = (v846_data + v845_data);
            float v848_data = r1[102];
            float v849_data = ir3[6];
            ir3[6] = (v849_data + v848_data);
            float v851_data = r1[128];
            float v852_data = ir3[8];
            ir3[8] = (v852_data + v851_data);
            float v854_data = r1[154];
            float v855_data = ir3[10];
            ir3[10] = (v855_data + v854_data);
          }
          if (v41_g) {
            float v857_data = r1[25];
            float v858_data = ir3[1];
            ir3[1] = (v858_data + v857_data);
            float v860_data = r1[51];
            float v861_data = ir3[3];
            ir3[3] = (v861_data + v860_data);
            float v863_data = r1[77];
            float v864_data = ir3[5];
            ir3[5] = (v864_data + v863_data);
            float v866_data = r1[103];
            float v867_data = ir3[7];
            ir3[7] = (v867_data + v866_data);
            float v869_data = r1[129];
            float v870_data = ir3[9];
            ir3[9] = (v870_data + v869_data);
            float v872_data = r1[155];
            float v873_data = ir3[11];
            ir3[11] = (v873_data + v872_data);
          }
          if (v26_g) {
            #pragma unroll
            for (int32_t v875_n1 = 0; v875_n1 < 1; ++v875_n1) {
              int32_t v877_a = v875_n1 * 2;
              #pragma unroll
              for (int32_t v876_n2 = 0; v876_n2 < 6; ++v876_n2) {
                int32_t v880_a = v877_a + (v876_n2 * 2);
                float v881_data = ir3[v880_a];
                float v882_data = r2[v880_a];
                r3[v880_a] = (v882_data + v881_data);
              }
            }
          }
          if (v41_g) {
            #pragma unroll
            for (int32_t v884_n1 = 0; v884_n1 < 1; ++v884_n1) {
              int32_t v888_a = 1 + (v884_n1 * 2);
              #pragma unroll
              for (int32_t v885_n2 = 0; v885_n2 < 6; ++v885_n2) {
                int32_t v889_a = v888_a + (v885_n2 * 2);
                float v890_data = ir3[v889_a];
                float v891_data = r2[v889_a];
                r3[v889_a] = (v891_data + v890_data);
              }
            }
          }
          // glb_m2 = store{r>g}(r3);
          if (v26_g) {
            #pragma unroll
            for (int32_t v893_i1 = 0; v893_i1 < 1; ++v893_i1) {
              int32_t v895_a = v893_i1 * 2;
              int32_t v905_a = v15_lead + ((v893_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v894_i2 = 0; v894_i2 < 6; ++v894_i2) {
                float v899_data = r3[(v895_a + (v894_i2 * 2))];
                glb_m2[(v905_a + (v894_i2 * 832))] = v899_data;
              }
            }
          }
          if (v41_g) {
            int32_t v915_lead = v15_lead + 32_i32;
            #pragma unroll
            for (int32_t v907_i1 = 0; v907_i1 < 1; ++v907_i1) {
              int32_t v911_a = 1 + (v907_i1 * 2);
              int32_t v919_a = v915_lead + ((v907_i1 + 12) * 64);
              #pragma unroll
              for (int32_t v908_i2 = 0; v908_i2 < 6; ++v908_i2) {
                float v913_data = r3[(v911_a + (v908_i2 * 2))];
                glb_m2[(v919_a + (v908_i2 * 832))] = v913_data;
              }
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

