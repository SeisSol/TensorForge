// === base name ===
kernel_8d3a4b67dfb55bdc

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_8d3a4b67dfb55bdc = {{16, 8, 1}, 16, 12, 1, 8, 5632, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_8d3a4b67dfb55bdc(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_8d3a4b67dfb55bdc(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_8d3a4b67dfb55bdc(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_8d3a4b67dfb55bdc, block.x * block.y * block.z, 1408 * sizeof(float));
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
void launcher_kernel_8d3a4b67dfb55bdc(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_8d3a4b67dfb55bdc(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_8d3a4b67dfb55bdc, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  const unsigned * elementIndicesArg = (const unsigned *)elementIndices;
  const unsigned * flags0Arg = (const unsigned *)flags0;
  kernel_kernel_8d3a4b67dfb55bdc<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m1_stride, m2, m2_extraOffset, numElements0, elementIndicesArg, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_8d3a4b67dfb55bdc(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * __restrict__ elementIndices, const unsigned * __restrict__ flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 8 per block = block 16x8x1, 5632 B shared, occupancy grid
    // operands:
    //   m0 12×7(12×7) {0..12}×{0..7} strided
    //   m1 12×20(12×20) {0..12}×{0..20} indexed(stride=runtime)
    //   m2 20×7(20×7) {0..20}×{0..7} indexed
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1408}],"shared_bytes":5632,"shared_elements":1408,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,7]],"name":"m0","ordered":false,"parts":1,"shape":[12,7],"variant":false},{"addressing":"indexed(stride=runtime)","alias":"A","bbox":[[0,0],[12,20]],"name":"m1","ordered":false,"parts":1,"shape":[12,20],"variant":false},{"addressing":"indexed","alias":"B","bbox":[[0,0],[20,7]],"name":"m2","ordered":false,"parts":1,"shape":[20,7],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,7]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,7]},"kind":"multilinear","ops":[{"addressing":"indexed(stride=runtime)","bbox":[[0,0],[12,20]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,20]},{"addressing":"indexed","bbox":[[0,0],[20,7]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,7]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[176 * threadIdx.y + 0];
      size_t v3_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < numElements0) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      float * __restrict__ s0 = &localShrMem0[0];
      float r0[20]{};
      const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
      size_t v752_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v752_flagSecondAhead < numElements0) ? v752_flagSecondAhead : v3_batchId1)];
      const bool allowed_peel = static_cast<bool>(flagWordFirst);
      const float *const __restrict__ peel_glb_m1 = &m1[static_cast<size_t>(elementIndices[v3_batchId1]) * m1_stride + 0 + m1_extraOffset];
      int32_t v762_lead = threadIdx.x % 16;
      bool v763_g = v762_lead < 12;
      bool v780_g = allowed_peel && v763_g;
      // r0 = load{g>r}(peel_glb_m1);
      #pragma unroll
      for (int32_t v772_i1 = 0; v772_i1 < 20; ++v772_i1) {
        float v775_data = v780_g ? (__ldcg(&peel_glb_m1[(v762_lead + (v772_i1 * 12))])) : (0.0f);
        r0[v772_i1] = v775_data;
      }
      const float *const __restrict__ peel_glb_m2 = &m2[static_cast<size_t>(elementIndices[v3_batchId1]) * 140 + 0 + m2_extraOffset];
      bool v784_g = threadIdx.x < 12;
      // s0 = load{g>s}(peel_glb_m2[0, 1])
      #pragma unroll
      for (int32_t v787_i = 0; v787_i < 8; ++v787_i) {
        int32_t v789_a = threadIdx.x + (v787_i * 16);
        if (allowed_peel) {
          __pipeline_memcpy_async(&s0[v789_a], &peel_glb_m2[v789_a], 4);
        }
      }
      if (v784_g) {
        if (allowed_peel) {
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 128], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 128], 4);
        }
      }
      __pipeline_commit();
      uint32_t v756_flagWord = flagWordFirst;
      uint32_t v757_flagWordNext = flagWordSecond;
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const float *const __restrict__ wrap_glb_m1 = &m1[static_cast<size_t>(elementIndices[v11_batchId1]) * m1_stride + 0 + m1_extraOffset];
        const float *const __restrict__ wrap_glb_m2 = &m2[static_cast<size_t>(elementIndices[v11_batchId1]) * 140 + 0 + m2_extraOffset];
        size_t v803_flagAheadAhead = v11_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v803_flagAheadAhead < numElements0) ? v803_flagAheadAhead : v11_batchId1)];
        const bool allowed = static_cast<bool>(v756_flagWord);
        const bool allowed_next = static_cast<bool>(v757_flagWordNext);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 84 + 0 + m0_extraOffset];
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[7]{};
          // ir1 = +(r0 * s0)
          // [(0, 12), (0, 7)] [(0, 20)]
          float ir1[7]{};
          float v39_data = r0[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v40_data = s0[0];
          float v45_data = s0[20];
          float v50_data = s0[40];
          float v55_data = s0[60];
          float v60_data = s0[80];
          float v65_data = s0[100];
          float v70_data = s0[120];
          float v74_data = r0[1];
          float v75_data = s0[1];
          float v80_data = s0[21];
          float v85_data = s0[41];
          float v90_data = s0[61];
          float v95_data = s0[81];
          float v100_data = s0[101];
          float v105_data = s0[121];
          float v109_data = r0[2];
          float v110_data = s0[2];
          float v115_data = s0[22];
          float v120_data = s0[42];
          float v125_data = s0[62];
          float v130_data = s0[82];
          float v135_data = s0[102];
          float v140_data = s0[122];
          float v144_data = r0[3];
          float v145_data = s0[3];
          float v150_data = s0[23];
          float v155_data = s0[43];
          float v160_data = s0[63];
          float v165_data = s0[83];
          float v170_data = s0[103];
          float v175_data = s0[123];
          float v179_data = r0[4];
          float v180_data = s0[4];
          float v185_data = s0[24];
          float v190_data = s0[44];
          float v195_data = s0[64];
          float v200_data = s0[84];
          float v205_data = s0[104];
          float v210_data = s0[124];
          float v214_data = r0[5];
          float v215_data = s0[5];
          float v220_data = s0[25];
          float v225_data = s0[45];
          float v230_data = s0[65];
          float v235_data = s0[85];
          float v240_data = s0[105];
          float v245_data = s0[125];
          float v249_data = r0[6];
          float v250_data = s0[6];
          float v255_data = s0[26];
          float v260_data = s0[46];
          float v265_data = s0[66];
          float v270_data = s0[86];
          float v275_data = s0[106];
          float v280_data = s0[126];
          float v284_data = r0[7];
          float v285_data = s0[7];
          float v290_data = s0[27];
          float v295_data = s0[47];
          float v300_data = s0[67];
          float v305_data = s0[87];
          float v310_data = s0[107];
          float v315_data = s0[127];
          float v319_data = r0[8];
          float v320_data = s0[8];
          float v325_data = s0[28];
          float v330_data = s0[48];
          float v335_data = s0[68];
          float v340_data = s0[88];
          float v345_data = s0[108];
          float v350_data = s0[128];
          float v354_data = r0[9];
          float v355_data = s0[9];
          float v360_data = s0[29];
          float v365_data = s0[49];
          float v370_data = s0[69];
          float v375_data = s0[89];
          float v380_data = s0[109];
          float v385_data = s0[129];
          float v389_data = r0[10];
          float v390_data = s0[10];
          float v395_data = s0[30];
          float v400_data = s0[50];
          float v405_data = s0[70];
          float v410_data = s0[90];
          float v415_data = s0[110];
          float v420_data = s0[130];
          float v424_data = r0[11];
          float v425_data = s0[11];
          float v430_data = s0[31];
          float v435_data = s0[51];
          float v440_data = s0[71];
          float v445_data = s0[91];
          float v450_data = s0[111];
          float v455_data = s0[131];
          float v459_data = r0[12];
          float v460_data = s0[12];
          float v465_data = s0[32];
          float v470_data = s0[52];
          float v475_data = s0[72];
          float v480_data = s0[92];
          float v485_data = s0[112];
          float v490_data = s0[132];
          float v494_data = r0[13];
          float v495_data = s0[13];
          float v500_data = s0[33];
          float v505_data = s0[53];
          float v510_data = s0[73];
          float v515_data = s0[93];
          float v520_data = s0[113];
          float v525_data = s0[133];
          float v529_data = r0[14];
          float v530_data = s0[14];
          float v535_data = s0[34];
          float v540_data = s0[54];
          float v545_data = s0[74];
          float v550_data = s0[94];
          float v555_data = s0[114];
          float v560_data = s0[134];
          float v564_data = r0[15];
          float v565_data = s0[15];
          float v570_data = s0[35];
          float v575_data = s0[55];
          float v580_data = s0[75];
          float v585_data = s0[95];
          float v590_data = s0[115];
          float v595_data = s0[135];
          float v599_data = r0[16];
          float v600_data = s0[16];
          float v605_data = s0[36];
          float v610_data = s0[56];
          float v615_data = s0[76];
          float v620_data = s0[96];
          float v625_data = s0[116];
          float v630_data = s0[136];
          float v634_data = r0[17];
          float v635_data = s0[17];
          float v640_data = s0[37];
          float v645_data = s0[57];
          float v650_data = s0[77];
          float v655_data = s0[97];
          float v660_data = s0[117];
          float v665_data = s0[137];
          float v669_data = r0[18];
          float v670_data = s0[18];
          float v673_p = ((((((((((((((((((v39_data * v40_data) + (v74_data * v75_data)) + (v109_data * v110_data)) + (v144_data * v145_data)) + (v179_data * v180_data)) + (v214_data * v215_data)) + (v249_data * v250_data)) + (v284_data * v285_data)) + (v319_data * v320_data)) + (v354_data * v355_data)) + (v389_data * v390_data)) + (v424_data * v425_data)) + (v459_data * v460_data)) + (v494_data * v495_data)) + (v529_data * v530_data)) + (v564_data * v565_data)) + (v599_data * v600_data)) + (v634_data * v635_data)) + (v669_data * v670_data);
          float v675_data = s0[38];
          float v678_p = ((((((((((((((((((v39_data * v45_data) + (v74_data * v80_data)) + (v109_data * v115_data)) + (v144_data * v150_data)) + (v179_data * v185_data)) + (v214_data * v220_data)) + (v249_data * v255_data)) + (v284_data * v290_data)) + (v319_data * v325_data)) + (v354_data * v360_data)) + (v389_data * v395_data)) + (v424_data * v430_data)) + (v459_data * v465_data)) + (v494_data * v500_data)) + (v529_data * v535_data)) + (v564_data * v570_data)) + (v599_data * v605_data)) + (v634_data * v640_data)) + (v669_data * v675_data);
          float v680_data = s0[58];
          float v683_p = ((((((((((((((((((v39_data * v50_data) + (v74_data * v85_data)) + (v109_data * v120_data)) + (v144_data * v155_data)) + (v179_data * v190_data)) + (v214_data * v225_data)) + (v249_data * v260_data)) + (v284_data * v295_data)) + (v319_data * v330_data)) + (v354_data * v365_data)) + (v389_data * v400_data)) + (v424_data * v435_data)) + (v459_data * v470_data)) + (v494_data * v505_data)) + (v529_data * v540_data)) + (v564_data * v575_data)) + (v599_data * v610_data)) + (v634_data * v645_data)) + (v669_data * v680_data);
          float v685_data = s0[78];
          float v688_p = ((((((((((((((((((v39_data * v55_data) + (v74_data * v90_data)) + (v109_data * v125_data)) + (v144_data * v160_data)) + (v179_data * v195_data)) + (v214_data * v230_data)) + (v249_data * v265_data)) + (v284_data * v300_data)) + (v319_data * v335_data)) + (v354_data * v370_data)) + (v389_data * v405_data)) + (v424_data * v440_data)) + (v459_data * v475_data)) + (v494_data * v510_data)) + (v529_data * v545_data)) + (v564_data * v580_data)) + (v599_data * v615_data)) + (v634_data * v650_data)) + (v669_data * v685_data);
          float v690_data = s0[98];
          float v693_p = ((((((((((((((((((v39_data * v60_data) + (v74_data * v95_data)) + (v109_data * v130_data)) + (v144_data * v165_data)) + (v179_data * v200_data)) + (v214_data * v235_data)) + (v249_data * v270_data)) + (v284_data * v305_data)) + (v319_data * v340_data)) + (v354_data * v375_data)) + (v389_data * v410_data)) + (v424_data * v445_data)) + (v459_data * v480_data)) + (v494_data * v515_data)) + (v529_data * v550_data)) + (v564_data * v585_data)) + (v599_data * v620_data)) + (v634_data * v655_data)) + (v669_data * v690_data);
          float v695_data = s0[118];
          float v698_p = ((((((((((((((((((v39_data * v65_data) + (v74_data * v100_data)) + (v109_data * v135_data)) + (v144_data * v170_data)) + (v179_data * v205_data)) + (v214_data * v240_data)) + (v249_data * v275_data)) + (v284_data * v310_data)) + (v319_data * v345_data)) + (v354_data * v380_data)) + (v389_data * v415_data)) + (v424_data * v450_data)) + (v459_data * v485_data)) + (v494_data * v520_data)) + (v529_data * v555_data)) + (v564_data * v590_data)) + (v599_data * v625_data)) + (v634_data * v660_data)) + (v669_data * v695_data);
          float v700_data = s0[138];
          float v703_p = ((((((((((((((((((v39_data * v70_data) + (v74_data * v105_data)) + (v109_data * v140_data)) + (v144_data * v175_data)) + (v179_data * v210_data)) + (v214_data * v245_data)) + (v249_data * v280_data)) + (v284_data * v315_data)) + (v319_data * v350_data)) + (v354_data * v385_data)) + (v389_data * v420_data)) + (v424_data * v455_data)) + (v459_data * v490_data)) + (v494_data * v525_data)) + (v529_data * v560_data)) + (v564_data * v595_data)) + (v599_data * v630_data)) + (v634_data * v665_data)) + (v669_data * v700_data);
          float v704_data = r0[19];
          bool v781_g = allowed_next && v763_g;
          // r0 = load{g>r}(wrap_glb_m1);
          #pragma unroll
          for (int32_t v776_i1 = 0; v776_i1 < 20; ++v776_i1) {
            float v779_data = v781_g ? (__ldcg(&wrap_glb_m1[(v762_lead + (v776_i1 * 12))])) : (0.0f);
            r0[v776_i1] = v779_data;
          }
          float v705_data = s0[19];
          ir1[0] = (v673_p + (v704_data * v705_data));
          float v710_data = s0[39];
          ir1[1] = (v678_p + (v704_data * v710_data));
          float v715_data = s0[59];
          ir1[2] = (v683_p + (v704_data * v715_data));
          float v720_data = s0[79];
          ir1[3] = (v688_p + (v704_data * v720_data));
          float v725_data = s0[99];
          ir1[4] = (v693_p + (v704_data * v725_data));
          float v730_data = s0[119];
          ir1[5] = (v698_p + (v704_data * v730_data));
          float v735_data = s0[139];
          ir1[6] = (v703_p + (v704_data * v735_data));
          // r1 = ir1
          // glb_m0 = store{r>g}(r1);
          if (v763_g) {
            #pragma unroll
            for (int32_t v744_i1 = 0; v744_i1 < 7; ++v744_i1) {
              float v746_data = ir1[v744_i1];
              glb_m0[(v762_lead + (v744_i1 * 12))] = v746_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
        else {
          __pipeline_wait_prior(0);
          bool v782_g = allowed_next && v763_g;
          // r0 = load{g>r}(wrap_glb_m1);
          #pragma unroll
          for (int32_t v768_i1 = 0; v768_i1 < 20; ++v768_i1) {
            float v771_data = v782_g ? (__ldcg(&wrap_glb_m1[(v762_lead + (v768_i1 * 12))])) : (0.0f);
            r0[v768_i1] = v771_data;
          }
        }
        // s0 = load{g>s}(wrap_glb_m2[0, 1])
        #pragma unroll
        for (int32_t v792_i = 0; v792_i < 8; ++v792_i) {
          int32_t v794_a = threadIdx.x + (v792_i * 16);
          if (allowed_next) {
            __pipeline_memcpy_async(&s0[v794_a], &wrap_glb_m2[v794_a], 4);
          }
        }
        if (v784_g) {
          if (allowed_next) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 128], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 128], 4);
          }
        }
        __pipeline_commit();
        v756_flagWord = v757_flagWordNext;
        v757_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
  }
}

