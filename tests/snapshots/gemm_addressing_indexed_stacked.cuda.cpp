// === base name ===
kernel_15a192d53010b9ff

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_15a192d53010b9ff = {{16, 8, 1}, 16, 16, 1, 8, 8704, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_15a192d53010b9ff(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_15a192d53010b9ff(float * m0, size_t m0_extraOffset, const float * m1, const float ** m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_15a192d53010b9ff(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_15a192d53010b9ff, block.x * block.y * block.z, 2176 * sizeof(float));
        CHECK_ERR;
        if (blocksPerSM > 0) {
          gridsize = smCount * blocksPerSM;
        }
        else {
          gridsize = smCount;
        }
      }
      
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, static_cast<size_t>((numElements0 + 1) / 2));
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 16;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 2176 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_15a192d53010b9ff(float * m0, size_t m0_extraOffset, const float * m1, const float ** m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_15a192d53010b9ff(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_15a192d53010b9ff, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  const float *const * m2Arg = (const float *const *)m2;
  const unsigned * elementIndicesArg = (const unsigned *)elementIndices;
  const unsigned * flags0Arg = (const unsigned *)flags0;
  kernel_kernel_15a192d53010b9ff<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m2Arg, m2_extraOffset, numElements0, elementIndicesArg, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_15a192d53010b9ff(float * m0, size_t m0_extraOffset, const float * m1, const float *const * __restrict__ m2, size_t m2_extraOffset, size_t numElements0, const unsigned * __restrict__ elementIndices, const unsigned * __restrict__ flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: stack_elements=2
    // launch: 16 lanes x 8 per block = block 16x8x1, 8704 B shared, occupancy grid
    // operands:
    //   m0 16×8×2(16×8×2) {0..16}×{0..8}×{0..2} indexed(stride=136, offset=4)
    //   m1 16×16(16×16) {0..16}×{0..16} none
    //   m2 16×8×2(16×8×2) {0..16}×{0..8}×{0..2} indexed_pointer(slots=3, slot=2)
    // operations:
    //   m0[i,j,l] = m1[i,k] × m2[k,j,l]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":2176}],"shared_bytes":8704,"shared_elements":2176,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"indexed(stride=136, offset=4)","alias":"C","bbox":[[0,0,0],[16,8,2]],"name":"m0","ordered":false,"parts":1,"shape":[16,8,2],"stack":2,"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"indexed_pointer(slots=3, slot=2)","alias":"B","bbox":[[0,0,0],[16,8,2]],"name":"m2","ordered":false,"parts":1,"shape":[16,8,2],"stack":2,"variant":false}],"operations":[{"add":false,"dest":{"addressing":"indexed(stride=136, offset=4)","bbox":[[0,0,0],[16,8,2]],"is_tmp":false,"name":"m0","offset":[0,0,0],"shape":[16,8,2]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"indexed_pointer(slots=3, slot=2)","bbox":[[0,0,0],[16,8,2]],"is_tmp":false,"name":"m2","offset":[0,0,0],"shape":[16,8,2]}],"permute":[[0,1],[0,1,2]],"target":[[0,-1],[-1,1,2]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[272 * threadIdx.y + 0];
      size_t v3_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < ((numElements0 + 1) / 2)) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      const float *const __restrict__ glb_m1 = &m1[0];
      float * __restrict__ s0 = &localShrMem0[0];
      const uint32_t flagWordFirst = flags0 == nullptr ? 1 : (flags0[2 * v3_batchId1] || (2 * v3_batchId1 + 1 < numElements0 && flags0[2 * v3_batchId1 + 1]));
      size_t v1354_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      size_t v1356_flagSecond = (v1354_flagSecondAhead < ((numElements0 + 1) / 2)) ? v1354_flagSecondAhead : v3_batchId1;
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : (flags0[2 * v1356_flagSecond] || (2 * v1356_flagSecond + 1 < numElements0 && flags0[2 * v1356_flagSecond + 1]));
      const bool allowed_peel = static_cast<bool>(flagWordFirst);
      const bool peel_batchIdStack0_0 = 2 * v3_batchId1 + 0 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v3_batchId1 + 0]));
      const bool peel_batchIdStack0_1 = 2 * v3_batchId1 + 1 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v3_batchId1 + 1]));
      const size_t peel_elementIndex_s0 = static_cast<size_t>(elementIndices[(peel_batchIdStack0_0 ? 2 * v3_batchId1 + 0 : (peel_batchIdStack0_0 ? 2 * v3_batchId1 + 0 : peel_batchIdStack0_1 ? 2 * v3_batchId1 + 1 : 2 * v3_batchId1))]);
      const size_t peel_elementIndex_s1 = static_cast<size_t>(elementIndices[(peel_batchIdStack0_1 ? 2 * v3_batchId1 + 1 : (peel_batchIdStack0_0 ? 2 * v3_batchId1 + 0 : peel_batchIdStack0_1 ? 2 * v3_batchId1 + 1 : 2 * v3_batchId1))]);
      const float *const __restrict__ peel_glb_m2_s0 = &m2[peel_elementIndex_s0 * 3 + 2][0 + m2_extraOffset];
      const float *const __restrict__ peel_glb_m2_s1 = &m2[peel_elementIndex_s1 * 3 + 2][0 + m2_extraOffset];
      // s0 = load{g>s}(glb_m2[0, 1, 2])
      #pragma unroll
      for (int32_t i = 0; i < 8; i += 1) {
        if (allowed_peel) {
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &peel_glb_m2_s0[0 + 0 + 1 * threadIdx.x + i * 16], 4);
        }
      }
      #pragma unroll
      for (int32_t i = 0; i < 8; i += 1) {
        if (allowed_peel) {
          __pipeline_memcpy_async(&s0[0 + 128 + 1 * threadIdx.x + i * 16], &peel_glb_m2_s1[0 + 0 + 1 * threadIdx.x + i * 16], 4);
        }
      }
      __pipeline_commit();
      uint32_t v1358_flagWord = flagWordFirst;
      uint32_t v1359_flagWordNext = flagWordSecond;
      size_t v1389_elementIndex_s0 = peel_elementIndex_s0;
      size_t v1391_elementIndex_s1 = peel_elementIndex_s1;
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < ((numElements0 + 1) / 2); v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < ((numElements0 + 1) / 2)) ? v10_ahead1 : v9_batchId0;
        const bool wrap_batchIdStack0_0 = 2 * v12_batchId1 + 0 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v12_batchId1 + 0]));
        const bool wrap_batchIdStack0_1 = 2 * v12_batchId1 + 1 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v12_batchId1 + 1]));
        const size_t wrap_elementIndex_s0 = static_cast<size_t>(elementIndices[(wrap_batchIdStack0_0 ? 2 * v12_batchId1 + 0 : (wrap_batchIdStack0_0 ? 2 * v12_batchId1 + 0 : wrap_batchIdStack0_1 ? 2 * v12_batchId1 + 1 : 2 * v12_batchId1))]);
        const size_t wrap_elementIndex_s1 = static_cast<size_t>(elementIndices[(wrap_batchIdStack0_1 ? 2 * v12_batchId1 + 1 : (wrap_batchIdStack0_0 ? 2 * v12_batchId1 + 0 : wrap_batchIdStack0_1 ? 2 * v12_batchId1 + 1 : 2 * v12_batchId1))]);
        const float *const __restrict__ wrap_glb_m2_s0 = &m2[wrap_elementIndex_s0 * 3 + 2][0 + m2_extraOffset];
        const float *const __restrict__ wrap_glb_m2_s1 = &m2[wrap_elementIndex_s1 * 3 + 2][0 + m2_extraOffset];
        size_t v1384_flagAheadAhead = v12_batchId1 + (gridDim.x * blockDim.y);
        size_t v1386_flagAhead = (v1384_flagAheadAhead < ((numElements0 + 1) / 2)) ? v1384_flagAheadAhead : v12_batchId1;
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : (flags0[2 * v1386_flagAhead] || (2 * v1386_flagAhead + 1 < numElements0 && flags0[2 * v1386_flagAhead + 1]));
        const bool batchIdStack0_0 = 2 * v9_batchId0 + 0 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v9_batchId0 + 0]));
        const bool batchIdStack0_1 = 2 * v9_batchId0 + 1 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v9_batchId0 + 1]));
        const bool allowed = static_cast<bool>(v1358_flagWord);
        if (allowed) {
          float *const glb_m0_s0 = &m0[v1389_elementIndex_s0 * 136 + 0 + 4 + m0_extraOffset];
          float *const glb_m0_s1 = &m0[v1391_elementIndex_s1 * 136 + 0 + 4 + m0_extraOffset];
          // wait(s0 = load{g>s}(glb_m2[0, 1, 2]));
          __pipeline_wait_prior(0);
          float r0[16]{};
          // ir0 = +(glb_m1 * s0)
          // [(0, 16), (0, 8), (0, 2)] [(0, 16)]
          float ir0[16]{};
          int32_t v31_lead = threadIdx.x % 16;
          float v35_data = glb_m1[v31_lead];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v36_data = s0[0];
          float v41_data = s0[128];
          float v46_data = s0[16];
          float v51_data = s0[144];
          float v56_data = s0[32];
          float v61_data = s0[160];
          float v66_data = s0[48];
          float v71_data = s0[176];
          float v76_data = s0[64];
          float v81_data = s0[192];
          float v86_data = s0[80];
          float v91_data = s0[208];
          float v96_data = s0[96];
          float v101_data = s0[224];
          float v106_data = s0[112];
          float v111_data = s0[240];
          float v116_data = glb_m1[(v31_lead + 16)];
          float v117_data = s0[1];
          float v122_data = s0[129];
          float v127_data = s0[17];
          float v132_data = s0[145];
          float v137_data = s0[33];
          float v142_data = s0[161];
          float v147_data = s0[49];
          float v152_data = s0[177];
          float v157_data = s0[65];
          float v162_data = s0[193];
          float v167_data = s0[81];
          float v172_data = s0[209];
          float v177_data = s0[97];
          float v182_data = s0[225];
          float v187_data = s0[113];
          float v192_data = s0[241];
          float v197_data = glb_m1[(v31_lead + 32)];
          float v198_data = s0[2];
          float v203_data = s0[130];
          float v208_data = s0[18];
          float v213_data = s0[146];
          float v218_data = s0[34];
          float v223_data = s0[162];
          float v228_data = s0[50];
          float v233_data = s0[178];
          float v238_data = s0[66];
          float v243_data = s0[194];
          float v248_data = s0[82];
          float v253_data = s0[210];
          float v258_data = s0[98];
          float v263_data = s0[226];
          float v268_data = s0[114];
          float v273_data = s0[242];
          float v278_data = glb_m1[(v31_lead + 48)];
          float v279_data = s0[3];
          float v284_data = s0[131];
          float v289_data = s0[19];
          float v294_data = s0[147];
          float v299_data = s0[35];
          float v304_data = s0[163];
          float v309_data = s0[51];
          float v314_data = s0[179];
          float v319_data = s0[67];
          float v324_data = s0[195];
          float v329_data = s0[83];
          float v334_data = s0[211];
          float v339_data = s0[99];
          float v344_data = s0[227];
          float v349_data = s0[115];
          float v354_data = s0[243];
          float v359_data = glb_m1[(v31_lead + 64)];
          float v360_data = s0[4];
          float v365_data = s0[132];
          float v370_data = s0[20];
          float v375_data = s0[148];
          float v380_data = s0[36];
          float v385_data = s0[164];
          float v390_data = s0[52];
          float v395_data = s0[180];
          float v400_data = s0[68];
          float v405_data = s0[196];
          float v410_data = s0[84];
          float v415_data = s0[212];
          float v420_data = s0[100];
          float v425_data = s0[228];
          float v430_data = s0[116];
          float v435_data = s0[244];
          float v440_data = glb_m1[(v31_lead + 80)];
          float v441_data = s0[5];
          float v446_data = s0[133];
          float v451_data = s0[21];
          float v456_data = s0[149];
          float v461_data = s0[37];
          float v466_data = s0[165];
          float v471_data = s0[53];
          float v476_data = s0[181];
          float v481_data = s0[69];
          float v486_data = s0[197];
          float v491_data = s0[85];
          float v496_data = s0[213];
          float v501_data = s0[101];
          float v506_data = s0[229];
          float v511_data = s0[117];
          float v516_data = s0[245];
          float v521_data = glb_m1[(v31_lead + 96)];
          float v522_data = s0[6];
          float v527_data = s0[134];
          float v532_data = s0[22];
          float v537_data = s0[150];
          float v542_data = s0[38];
          float v547_data = s0[166];
          float v552_data = s0[54];
          float v557_data = s0[182];
          float v562_data = s0[70];
          float v567_data = s0[198];
          float v572_data = s0[86];
          float v577_data = s0[214];
          float v582_data = s0[102];
          float v587_data = s0[230];
          float v592_data = s0[118];
          float v597_data = s0[246];
          float v602_data = glb_m1[(v31_lead + 112)];
          float v603_data = s0[7];
          float v608_data = s0[135];
          float v613_data = s0[23];
          float v618_data = s0[151];
          float v623_data = s0[39];
          float v628_data = s0[167];
          float v633_data = s0[55];
          float v638_data = s0[183];
          float v643_data = s0[71];
          float v648_data = s0[199];
          float v653_data = s0[87];
          float v658_data = s0[215];
          float v663_data = s0[103];
          float v668_data = s0[231];
          float v673_data = s0[119];
          float v678_data = s0[247];
          float v683_data = glb_m1[(v31_lead + 128)];
          float v684_data = s0[8];
          float v689_data = s0[136];
          float v694_data = s0[24];
          float v699_data = s0[152];
          float v704_data = s0[40];
          float v709_data = s0[168];
          float v714_data = s0[56];
          float v719_data = s0[184];
          float v724_data = s0[72];
          float v729_data = s0[200];
          float v734_data = s0[88];
          float v739_data = s0[216];
          float v744_data = s0[104];
          float v749_data = s0[232];
          float v754_data = s0[120];
          float v759_data = s0[248];
          float v764_data = glb_m1[(v31_lead + 144)];
          float v765_data = s0[9];
          float v770_data = s0[137];
          float v775_data = s0[25];
          float v780_data = s0[153];
          float v785_data = s0[41];
          float v790_data = s0[169];
          float v795_data = s0[57];
          float v800_data = s0[185];
          float v805_data = s0[73];
          float v810_data = s0[201];
          float v815_data = s0[89];
          float v820_data = s0[217];
          float v825_data = s0[105];
          float v830_data = s0[233];
          float v835_data = s0[121];
          float v840_data = s0[249];
          float v845_data = glb_m1[(v31_lead + 160)];
          float v846_data = s0[10];
          float v851_data = s0[138];
          float v856_data = s0[26];
          float v861_data = s0[154];
          float v866_data = s0[42];
          float v871_data = s0[170];
          float v876_data = s0[58];
          float v881_data = s0[186];
          float v886_data = s0[74];
          float v891_data = s0[202];
          float v896_data = s0[90];
          float v901_data = s0[218];
          float v906_data = s0[106];
          float v911_data = s0[234];
          float v916_data = s0[122];
          float v921_data = s0[250];
          float v926_data = glb_m1[(v31_lead + 176)];
          float v927_data = s0[11];
          float v932_data = s0[139];
          float v937_data = s0[27];
          float v942_data = s0[155];
          float v947_data = s0[43];
          float v952_data = s0[171];
          float v957_data = s0[59];
          float v962_data = s0[187];
          float v967_data = s0[75];
          float v972_data = s0[203];
          float v977_data = s0[91];
          float v982_data = s0[219];
          float v987_data = s0[107];
          float v992_data = s0[235];
          float v997_data = s0[123];
          float v1002_data = s0[251];
          float v1007_data = glb_m1[(v31_lead + 192)];
          float v1008_data = s0[12];
          float v1013_data = s0[140];
          float v1018_data = s0[28];
          float v1023_data = s0[156];
          float v1028_data = s0[44];
          float v1033_data = s0[172];
          float v1038_data = s0[60];
          float v1043_data = s0[188];
          float v1048_data = s0[76];
          float v1053_data = s0[204];
          float v1058_data = s0[92];
          float v1063_data = s0[220];
          float v1068_data = s0[108];
          float v1073_data = s0[236];
          float v1078_data = s0[124];
          float v1083_data = s0[252];
          float v1088_data = glb_m1[(v31_lead + 208)];
          float v1089_data = s0[13];
          float v1094_data = s0[141];
          float v1099_data = s0[29];
          float v1104_data = s0[157];
          float v1109_data = s0[45];
          float v1114_data = s0[173];
          float v1119_data = s0[61];
          float v1124_data = s0[189];
          float v1129_data = s0[77];
          float v1134_data = s0[205];
          float v1139_data = s0[93];
          float v1144_data = s0[221];
          float v1149_data = s0[109];
          float v1154_data = s0[237];
          float v1159_data = s0[125];
          float v1164_data = s0[253];
          float v1169_data = glb_m1[(v31_lead + 224)];
          float v1170_data = s0[14];
          float v1175_data = s0[142];
          float v1178_p = ((((((((((((((v35_data * v41_data) + (v116_data * v122_data)) + (v197_data * v203_data)) + (v278_data * v284_data)) + (v359_data * v365_data)) + (v440_data * v446_data)) + (v521_data * v527_data)) + (v602_data * v608_data)) + (v683_data * v689_data)) + (v764_data * v770_data)) + (v845_data * v851_data)) + (v926_data * v932_data)) + (v1007_data * v1013_data)) + (v1088_data * v1094_data)) + (v1169_data * v1175_data);
          float v1180_data = s0[30];
          float v1183_p = ((((((((((((((v35_data * v46_data) + (v116_data * v127_data)) + (v197_data * v208_data)) + (v278_data * v289_data)) + (v359_data * v370_data)) + (v440_data * v451_data)) + (v521_data * v532_data)) + (v602_data * v613_data)) + (v683_data * v694_data)) + (v764_data * v775_data)) + (v845_data * v856_data)) + (v926_data * v937_data)) + (v1007_data * v1018_data)) + (v1088_data * v1099_data)) + (v1169_data * v1180_data);
          float v1185_data = s0[158];
          float v1188_p = ((((((((((((((v35_data * v51_data) + (v116_data * v132_data)) + (v197_data * v213_data)) + (v278_data * v294_data)) + (v359_data * v375_data)) + (v440_data * v456_data)) + (v521_data * v537_data)) + (v602_data * v618_data)) + (v683_data * v699_data)) + (v764_data * v780_data)) + (v845_data * v861_data)) + (v926_data * v942_data)) + (v1007_data * v1023_data)) + (v1088_data * v1104_data)) + (v1169_data * v1185_data);
          float v1190_data = s0[46];
          float v1193_p = ((((((((((((((v35_data * v56_data) + (v116_data * v137_data)) + (v197_data * v218_data)) + (v278_data * v299_data)) + (v359_data * v380_data)) + (v440_data * v461_data)) + (v521_data * v542_data)) + (v602_data * v623_data)) + (v683_data * v704_data)) + (v764_data * v785_data)) + (v845_data * v866_data)) + (v926_data * v947_data)) + (v1007_data * v1028_data)) + (v1088_data * v1109_data)) + (v1169_data * v1190_data);
          float v1195_data = s0[174];
          float v1198_p = ((((((((((((((v35_data * v61_data) + (v116_data * v142_data)) + (v197_data * v223_data)) + (v278_data * v304_data)) + (v359_data * v385_data)) + (v440_data * v466_data)) + (v521_data * v547_data)) + (v602_data * v628_data)) + (v683_data * v709_data)) + (v764_data * v790_data)) + (v845_data * v871_data)) + (v926_data * v952_data)) + (v1007_data * v1033_data)) + (v1088_data * v1114_data)) + (v1169_data * v1195_data);
          float v1200_data = s0[62];
          float v1203_p = ((((((((((((((v35_data * v66_data) + (v116_data * v147_data)) + (v197_data * v228_data)) + (v278_data * v309_data)) + (v359_data * v390_data)) + (v440_data * v471_data)) + (v521_data * v552_data)) + (v602_data * v633_data)) + (v683_data * v714_data)) + (v764_data * v795_data)) + (v845_data * v876_data)) + (v926_data * v957_data)) + (v1007_data * v1038_data)) + (v1088_data * v1119_data)) + (v1169_data * v1200_data);
          float v1205_data = s0[190];
          float v1208_p = ((((((((((((((v35_data * v71_data) + (v116_data * v152_data)) + (v197_data * v233_data)) + (v278_data * v314_data)) + (v359_data * v395_data)) + (v440_data * v476_data)) + (v521_data * v557_data)) + (v602_data * v638_data)) + (v683_data * v719_data)) + (v764_data * v800_data)) + (v845_data * v881_data)) + (v926_data * v962_data)) + (v1007_data * v1043_data)) + (v1088_data * v1124_data)) + (v1169_data * v1205_data);
          float v1210_data = s0[78];
          float v1213_p = ((((((((((((((v35_data * v76_data) + (v116_data * v157_data)) + (v197_data * v238_data)) + (v278_data * v319_data)) + (v359_data * v400_data)) + (v440_data * v481_data)) + (v521_data * v562_data)) + (v602_data * v643_data)) + (v683_data * v724_data)) + (v764_data * v805_data)) + (v845_data * v886_data)) + (v926_data * v967_data)) + (v1007_data * v1048_data)) + (v1088_data * v1129_data)) + (v1169_data * v1210_data);
          float v1215_data = s0[206];
          float v1218_p = ((((((((((((((v35_data * v81_data) + (v116_data * v162_data)) + (v197_data * v243_data)) + (v278_data * v324_data)) + (v359_data * v405_data)) + (v440_data * v486_data)) + (v521_data * v567_data)) + (v602_data * v648_data)) + (v683_data * v729_data)) + (v764_data * v810_data)) + (v845_data * v891_data)) + (v926_data * v972_data)) + (v1007_data * v1053_data)) + (v1088_data * v1134_data)) + (v1169_data * v1215_data);
          float v1220_data = s0[94];
          float v1223_p = ((((((((((((((v35_data * v86_data) + (v116_data * v167_data)) + (v197_data * v248_data)) + (v278_data * v329_data)) + (v359_data * v410_data)) + (v440_data * v491_data)) + (v521_data * v572_data)) + (v602_data * v653_data)) + (v683_data * v734_data)) + (v764_data * v815_data)) + (v845_data * v896_data)) + (v926_data * v977_data)) + (v1007_data * v1058_data)) + (v1088_data * v1139_data)) + (v1169_data * v1220_data);
          float v1225_data = s0[222];
          float v1228_p = ((((((((((((((v35_data * v91_data) + (v116_data * v172_data)) + (v197_data * v253_data)) + (v278_data * v334_data)) + (v359_data * v415_data)) + (v440_data * v496_data)) + (v521_data * v577_data)) + (v602_data * v658_data)) + (v683_data * v739_data)) + (v764_data * v820_data)) + (v845_data * v901_data)) + (v926_data * v982_data)) + (v1007_data * v1063_data)) + (v1088_data * v1144_data)) + (v1169_data * v1225_data);
          float v1230_data = s0[110];
          float v1233_p = ((((((((((((((v35_data * v96_data) + (v116_data * v177_data)) + (v197_data * v258_data)) + (v278_data * v339_data)) + (v359_data * v420_data)) + (v440_data * v501_data)) + (v521_data * v582_data)) + (v602_data * v663_data)) + (v683_data * v744_data)) + (v764_data * v825_data)) + (v845_data * v906_data)) + (v926_data * v987_data)) + (v1007_data * v1068_data)) + (v1088_data * v1149_data)) + (v1169_data * v1230_data);
          float v1235_data = s0[238];
          float v1238_p = ((((((((((((((v35_data * v101_data) + (v116_data * v182_data)) + (v197_data * v263_data)) + (v278_data * v344_data)) + (v359_data * v425_data)) + (v440_data * v506_data)) + (v521_data * v587_data)) + (v602_data * v668_data)) + (v683_data * v749_data)) + (v764_data * v830_data)) + (v845_data * v911_data)) + (v926_data * v992_data)) + (v1007_data * v1073_data)) + (v1088_data * v1154_data)) + (v1169_data * v1235_data);
          float v1240_data = s0[126];
          float v1243_p = ((((((((((((((v35_data * v106_data) + (v116_data * v187_data)) + (v197_data * v268_data)) + (v278_data * v349_data)) + (v359_data * v430_data)) + (v440_data * v511_data)) + (v521_data * v592_data)) + (v602_data * v673_data)) + (v683_data * v754_data)) + (v764_data * v835_data)) + (v845_data * v916_data)) + (v926_data * v997_data)) + (v1007_data * v1078_data)) + (v1088_data * v1159_data)) + (v1169_data * v1240_data);
          float v1245_data = s0[254];
          float v1248_p = ((((((((((((((v35_data * v111_data) + (v116_data * v192_data)) + (v197_data * v273_data)) + (v278_data * v354_data)) + (v359_data * v435_data)) + (v440_data * v516_data)) + (v521_data * v597_data)) + (v602_data * v678_data)) + (v683_data * v759_data)) + (v764_data * v840_data)) + (v845_data * v921_data)) + (v926_data * v1002_data)) + (v1007_data * v1083_data)) + (v1088_data * v1164_data)) + (v1169_data * v1245_data);
          float v1250_data = glb_m1[(v31_lead + 240)];
          float v1251_data = s0[15];
          ir0[0] = ((((((((((((((((v35_data * v36_data) + (v116_data * v117_data)) + (v197_data * v198_data)) + (v278_data * v279_data)) + (v359_data * v360_data)) + (v440_data * v441_data)) + (v521_data * v522_data)) + (v602_data * v603_data)) + (v683_data * v684_data)) + (v764_data * v765_data)) + (v845_data * v846_data)) + (v926_data * v927_data)) + (v1007_data * v1008_data)) + (v1088_data * v1089_data)) + (v1169_data * v1170_data)) + (v1250_data * v1251_data));
          float v1256_data = s0[143];
          ir0[8] = (v1178_p + (v1250_data * v1256_data));
          float v1261_data = s0[31];
          ir0[1] = (v1183_p + (v1250_data * v1261_data));
          float v1266_data = s0[159];
          ir0[9] = (v1188_p + (v1250_data * v1266_data));
          float v1271_data = s0[47];
          ir0[2] = (v1193_p + (v1250_data * v1271_data));
          float v1276_data = s0[175];
          ir0[10] = (v1198_p + (v1250_data * v1276_data));
          float v1281_data = s0[63];
          ir0[3] = (v1203_p + (v1250_data * v1281_data));
          float v1286_data = s0[191];
          ir0[11] = (v1208_p + (v1250_data * v1286_data));
          float v1291_data = s0[79];
          ir0[4] = (v1213_p + (v1250_data * v1291_data));
          float v1296_data = s0[207];
          ir0[12] = (v1218_p + (v1250_data * v1296_data));
          float v1301_data = s0[95];
          ir0[5] = (v1223_p + (v1250_data * v1301_data));
          float v1306_data = s0[223];
          ir0[13] = (v1228_p + (v1250_data * v1306_data));
          float v1311_data = s0[111];
          ir0[6] = (v1233_p + (v1250_data * v1311_data));
          float v1316_data = s0[239];
          ir0[14] = (v1238_p + (v1250_data * v1316_data));
          float v1321_data = s0[127];
          ir0[7] = (v1243_p + (v1250_data * v1321_data));
          float v1326_data = s0[255];
          ir0[15] = (v1248_p + (v1250_data * v1326_data));
          // r0 = ir0
          // glb_m0 = store{r>g}(r0);
          #pragma unroll
          for (int32_t v1341_i0 = 0; v1341_i0 < 1; ++v1341_i0) {
            #pragma unroll
            for (int32_t v1342_i1 = 0; v1342_i1 < 8; ++v1342_i1) {
              #pragma unroll
              for (int32_t v1343_i2 = 0; v1343_i2 < 2; ++v1343_i2) {
                float v1347_data = ir0[(v1342_i1 + (v1343_i2 * 8))];
                if ((v1343_i2 == 0 ? batchIdStack0_0 : batchIdStack0_1)) {
                  float *const glb_m0_se0 = v1343_i2 == 0 ? glb_m0_s0 : glb_m0_s1;
                  glb_m0_se0[((v31_lead + (v1341_i0 * 16)) + (v1342_i1 * 16))] = v1347_data;
                }
              }
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
        else {
          __pipeline_wait_prior(0);
        }
        const bool allowed_next = static_cast<bool>(v1359_flagWordNext);
        // s0 = load{g>s}(glb_m2[0, 1, 2])
        #pragma unroll
        for (int32_t i = 0; i < 8; i += 1) {
          if (allowed_next) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &wrap_glb_m2_s0[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
        }
        #pragma unroll
        for (int32_t i = 0; i < 8; i += 1) {
          if (allowed_next) {
            __pipeline_memcpy_async(&s0[0 + 128 + 1 * threadIdx.x + i * 16], &wrap_glb_m2_s1[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
        }
        __pipeline_commit();
        v1358_flagWord = v1359_flagWordNext;
        v1359_flagWordNext = flagWordAhead;
        v1389_elementIndex_s0 = wrap_elementIndex_s0;
        v1391_elementIndex_s1 = wrap_elementIndex_s1;
      }
      __pipeline_wait_prior(0);
    }
  }
}

