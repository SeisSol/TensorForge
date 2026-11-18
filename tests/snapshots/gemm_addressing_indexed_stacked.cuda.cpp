// === base name ===
kernel_aa41d2f907f53c0c

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_aa41d2f907f53c0c = {{16, 8, 1}, 16, 16, 1, 8, 8704, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_aa41d2f907f53c0c(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_aa41d2f907f53c0c(float * m0, size_t m0_extraOffset, const float * m1, const float ** m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_aa41d2f907f53c0c(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_aa41d2f907f53c0c, block.x * block.y * block.z, 2176 * sizeof(float));
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
void launcher_kernel_aa41d2f907f53c0c(float * m0, size_t m0_extraOffset, const float * m1, const float ** m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_aa41d2f907f53c0c(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_aa41d2f907f53c0c, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  const float *const * m2Arg = (const float *const *)m2;
  const unsigned * elementIndicesArg = (const unsigned *)elementIndices;
  const unsigned * flags0Arg = (const unsigned *)flags0;
  kernel_kernel_aa41d2f907f53c0c<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m2Arg, m2_extraOffset, numElements0, elementIndicesArg, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_aa41d2f907f53c0c(float * m0, size_t m0_extraOffset, const float * m1, const float *const * __restrict__ m2, size_t m2_extraOffset, size_t numElements0, const unsigned * __restrict__ elementIndices, const unsigned * __restrict__ flags0) {
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
      size_t v1352_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      size_t v1354_flagSecond = (v1352_flagSecondAhead < ((numElements0 + 1) / 2)) ? v1352_flagSecondAhead : v3_batchId1;
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : (flags0[2 * v1354_flagSecond] || (2 * v1354_flagSecond + 1 < numElements0 && flags0[2 * v1354_flagSecond + 1]));
      const bool allowed_peel = static_cast<bool>(flagWordFirst);
      const bool peel_batchIdStack0_0 = 2 * v3_batchId1 + 0 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v3_batchId1 + 0]));
      const bool peel_batchIdStack0_1 = 2 * v3_batchId1 + 1 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v3_batchId1 + 1]));
      const float *const __restrict__ peel_glb_m2_s0 = &m2[static_cast<size_t>(elementIndices[(peel_batchIdStack0_0 ? 2 * v3_batchId1 + 0 : (peel_batchIdStack0_0 ? 2 * v3_batchId1 + 0 : peel_batchIdStack0_1 ? 2 * v3_batchId1 + 1 : 2 * v3_batchId1))]) * 3 + 2][0 + m2_extraOffset];
      const float *const __restrict__ peel_glb_m2_s1 = &m2[static_cast<size_t>(elementIndices[(peel_batchIdStack0_1 ? 2 * v3_batchId1 + 1 : (peel_batchIdStack0_0 ? 2 * v3_batchId1 + 0 : peel_batchIdStack0_1 ? 2 * v3_batchId1 + 1 : 2 * v3_batchId1))]) * 3 + 2][0 + m2_extraOffset];
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
      uint32_t v1356_flagWord = flagWordFirst;
      uint32_t v1357_flagWordNext = flagWordSecond;
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < ((numElements0 + 1) / 2); v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < ((numElements0 + 1) / 2)) ? v10_ahead1 : v9_batchId0;
        const bool wrap_batchIdStack0_0 = 2 * v12_batchId1 + 0 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v12_batchId1 + 0]));
        const bool wrap_batchIdStack0_1 = 2 * v12_batchId1 + 1 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v12_batchId1 + 1]));
        const float *const __restrict__ wrap_glb_m2_s0 = &m2[static_cast<size_t>(elementIndices[(wrap_batchIdStack0_0 ? 2 * v12_batchId1 + 0 : (wrap_batchIdStack0_0 ? 2 * v12_batchId1 + 0 : wrap_batchIdStack0_1 ? 2 * v12_batchId1 + 1 : 2 * v12_batchId1))]) * 3 + 2][0 + m2_extraOffset];
        const float *const __restrict__ wrap_glb_m2_s1 = &m2[static_cast<size_t>(elementIndices[(wrap_batchIdStack0_1 ? 2 * v12_batchId1 + 1 : (wrap_batchIdStack0_0 ? 2 * v12_batchId1 + 0 : wrap_batchIdStack0_1 ? 2 * v12_batchId1 + 1 : 2 * v12_batchId1))]) * 3 + 2][0 + m2_extraOffset];
        size_t v1378_flagAheadAhead = v12_batchId1 + (gridDim.x * blockDim.y);
        size_t v1380_flagAhead = (v1378_flagAheadAhead < ((numElements0 + 1) / 2)) ? v1378_flagAheadAhead : v12_batchId1;
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : (flags0[2 * v1380_flagAhead] || (2 * v1380_flagAhead + 1 < numElements0 && flags0[2 * v1380_flagAhead + 1]));
        const bool batchIdStack0_0 = 2 * v9_batchId0 + 0 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v9_batchId0 + 0]));
        const bool batchIdStack0_1 = 2 * v9_batchId0 + 1 < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[2 * v9_batchId0 + 1]));
        const bool allowed = static_cast<bool>(v1356_flagWord);
        if (allowed) {
          float *const glb_m0_s0 = &m0[static_cast<size_t>(elementIndices[(batchIdStack0_0 ? 2 * v9_batchId0 + 0 : (batchIdStack0_0 ? 2 * v9_batchId0 + 0 : batchIdStack0_1 ? 2 * v9_batchId0 + 1 : 2 * v9_batchId0))]) * 136 + 0 + 4 + m0_extraOffset];
          float *const glb_m0_s1 = &m0[static_cast<size_t>(elementIndices[(batchIdStack0_1 ? 2 * v9_batchId0 + 1 : (batchIdStack0_0 ? 2 * v9_batchId0 + 0 : batchIdStack0_1 ? 2 * v9_batchId0 + 1 : 2 * v9_batchId0))]) * 136 + 0 + 4 + m0_extraOffset];
          // wait(s0 = load{g>s}(glb_m2[0, 1, 2]));
          __pipeline_wait_prior(0);
          float r0[16]{};
          // ir0 = +(glb_m1 * s0)
          // [(0, 16), (0, 8), (0, 2)] [(0, 16)]
          float ir0[16]{};
          int32_t v29_lead = threadIdx.x % 16;
          float v33_data = glb_m1[v29_lead];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v34_data = s0[0];
          float v39_data = s0[128];
          float v44_data = s0[16];
          float v49_data = s0[144];
          float v54_data = s0[32];
          float v59_data = s0[160];
          float v64_data = s0[48];
          float v69_data = s0[176];
          float v74_data = s0[64];
          float v79_data = s0[192];
          float v84_data = s0[80];
          float v89_data = s0[208];
          float v94_data = s0[96];
          float v99_data = s0[224];
          float v104_data = s0[112];
          float v109_data = s0[240];
          float v114_data = glb_m1[(v29_lead + 16)];
          float v115_data = s0[1];
          float v120_data = s0[129];
          float v125_data = s0[17];
          float v130_data = s0[145];
          float v135_data = s0[33];
          float v140_data = s0[161];
          float v145_data = s0[49];
          float v150_data = s0[177];
          float v155_data = s0[65];
          float v160_data = s0[193];
          float v165_data = s0[81];
          float v170_data = s0[209];
          float v175_data = s0[97];
          float v180_data = s0[225];
          float v185_data = s0[113];
          float v190_data = s0[241];
          float v195_data = glb_m1[(v29_lead + 32)];
          float v196_data = s0[2];
          float v201_data = s0[130];
          float v206_data = s0[18];
          float v211_data = s0[146];
          float v216_data = s0[34];
          float v221_data = s0[162];
          float v226_data = s0[50];
          float v231_data = s0[178];
          float v236_data = s0[66];
          float v241_data = s0[194];
          float v246_data = s0[82];
          float v251_data = s0[210];
          float v256_data = s0[98];
          float v261_data = s0[226];
          float v266_data = s0[114];
          float v271_data = s0[242];
          float v276_data = glb_m1[(v29_lead + 48)];
          float v277_data = s0[3];
          float v282_data = s0[131];
          float v287_data = s0[19];
          float v292_data = s0[147];
          float v297_data = s0[35];
          float v302_data = s0[163];
          float v307_data = s0[51];
          float v312_data = s0[179];
          float v317_data = s0[67];
          float v322_data = s0[195];
          float v327_data = s0[83];
          float v332_data = s0[211];
          float v337_data = s0[99];
          float v342_data = s0[227];
          float v347_data = s0[115];
          float v352_data = s0[243];
          float v357_data = glb_m1[(v29_lead + 64)];
          float v358_data = s0[4];
          float v363_data = s0[132];
          float v368_data = s0[20];
          float v373_data = s0[148];
          float v378_data = s0[36];
          float v383_data = s0[164];
          float v388_data = s0[52];
          float v393_data = s0[180];
          float v398_data = s0[68];
          float v403_data = s0[196];
          float v408_data = s0[84];
          float v413_data = s0[212];
          float v418_data = s0[100];
          float v423_data = s0[228];
          float v428_data = s0[116];
          float v433_data = s0[244];
          float v438_data = glb_m1[(v29_lead + 80)];
          float v439_data = s0[5];
          float v444_data = s0[133];
          float v449_data = s0[21];
          float v454_data = s0[149];
          float v459_data = s0[37];
          float v464_data = s0[165];
          float v469_data = s0[53];
          float v474_data = s0[181];
          float v479_data = s0[69];
          float v484_data = s0[197];
          float v489_data = s0[85];
          float v494_data = s0[213];
          float v499_data = s0[101];
          float v504_data = s0[229];
          float v509_data = s0[117];
          float v514_data = s0[245];
          float v519_data = glb_m1[(v29_lead + 96)];
          float v520_data = s0[6];
          float v525_data = s0[134];
          float v530_data = s0[22];
          float v535_data = s0[150];
          float v540_data = s0[38];
          float v545_data = s0[166];
          float v550_data = s0[54];
          float v555_data = s0[182];
          float v560_data = s0[70];
          float v565_data = s0[198];
          float v570_data = s0[86];
          float v575_data = s0[214];
          float v580_data = s0[102];
          float v585_data = s0[230];
          float v590_data = s0[118];
          float v595_data = s0[246];
          float v600_data = glb_m1[(v29_lead + 112)];
          float v601_data = s0[7];
          float v606_data = s0[135];
          float v611_data = s0[23];
          float v616_data = s0[151];
          float v621_data = s0[39];
          float v626_data = s0[167];
          float v631_data = s0[55];
          float v636_data = s0[183];
          float v641_data = s0[71];
          float v646_data = s0[199];
          float v651_data = s0[87];
          float v656_data = s0[215];
          float v661_data = s0[103];
          float v666_data = s0[231];
          float v671_data = s0[119];
          float v676_data = s0[247];
          float v681_data = glb_m1[(v29_lead + 128)];
          float v682_data = s0[8];
          float v687_data = s0[136];
          float v692_data = s0[24];
          float v697_data = s0[152];
          float v702_data = s0[40];
          float v707_data = s0[168];
          float v712_data = s0[56];
          float v717_data = s0[184];
          float v722_data = s0[72];
          float v727_data = s0[200];
          float v732_data = s0[88];
          float v737_data = s0[216];
          float v742_data = s0[104];
          float v747_data = s0[232];
          float v752_data = s0[120];
          float v757_data = s0[248];
          float v762_data = glb_m1[(v29_lead + 144)];
          float v763_data = s0[9];
          float v768_data = s0[137];
          float v773_data = s0[25];
          float v778_data = s0[153];
          float v783_data = s0[41];
          float v788_data = s0[169];
          float v793_data = s0[57];
          float v798_data = s0[185];
          float v803_data = s0[73];
          float v808_data = s0[201];
          float v813_data = s0[89];
          float v818_data = s0[217];
          float v823_data = s0[105];
          float v828_data = s0[233];
          float v833_data = s0[121];
          float v838_data = s0[249];
          float v843_data = glb_m1[(v29_lead + 160)];
          float v844_data = s0[10];
          float v849_data = s0[138];
          float v854_data = s0[26];
          float v859_data = s0[154];
          float v864_data = s0[42];
          float v869_data = s0[170];
          float v874_data = s0[58];
          float v879_data = s0[186];
          float v884_data = s0[74];
          float v889_data = s0[202];
          float v894_data = s0[90];
          float v899_data = s0[218];
          float v904_data = s0[106];
          float v909_data = s0[234];
          float v914_data = s0[122];
          float v919_data = s0[250];
          float v924_data = glb_m1[(v29_lead + 176)];
          float v925_data = s0[11];
          float v930_data = s0[139];
          float v935_data = s0[27];
          float v940_data = s0[155];
          float v945_data = s0[43];
          float v950_data = s0[171];
          float v955_data = s0[59];
          float v960_data = s0[187];
          float v965_data = s0[75];
          float v970_data = s0[203];
          float v975_data = s0[91];
          float v980_data = s0[219];
          float v985_data = s0[107];
          float v990_data = s0[235];
          float v995_data = s0[123];
          float v1000_data = s0[251];
          float v1005_data = glb_m1[(v29_lead + 192)];
          float v1006_data = s0[12];
          float v1011_data = s0[140];
          float v1016_data = s0[28];
          float v1021_data = s0[156];
          float v1026_data = s0[44];
          float v1031_data = s0[172];
          float v1036_data = s0[60];
          float v1041_data = s0[188];
          float v1046_data = s0[76];
          float v1051_data = s0[204];
          float v1056_data = s0[92];
          float v1061_data = s0[220];
          float v1066_data = s0[108];
          float v1071_data = s0[236];
          float v1076_data = s0[124];
          float v1081_data = s0[252];
          float v1086_data = glb_m1[(v29_lead + 208)];
          float v1087_data = s0[13];
          float v1092_data = s0[141];
          float v1097_data = s0[29];
          float v1102_data = s0[157];
          float v1107_data = s0[45];
          float v1112_data = s0[173];
          float v1117_data = s0[61];
          float v1122_data = s0[189];
          float v1127_data = s0[77];
          float v1132_data = s0[205];
          float v1137_data = s0[93];
          float v1142_data = s0[221];
          float v1147_data = s0[109];
          float v1152_data = s0[237];
          float v1157_data = s0[125];
          float v1162_data = s0[253];
          float v1167_data = glb_m1[(v29_lead + 224)];
          float v1168_data = s0[14];
          float v1173_data = s0[142];
          float v1176_p = ((((((((((((((v33_data * v39_data) + (v114_data * v120_data)) + (v195_data * v201_data)) + (v276_data * v282_data)) + (v357_data * v363_data)) + (v438_data * v444_data)) + (v519_data * v525_data)) + (v600_data * v606_data)) + (v681_data * v687_data)) + (v762_data * v768_data)) + (v843_data * v849_data)) + (v924_data * v930_data)) + (v1005_data * v1011_data)) + (v1086_data * v1092_data)) + (v1167_data * v1173_data);
          float v1178_data = s0[30];
          float v1181_p = ((((((((((((((v33_data * v44_data) + (v114_data * v125_data)) + (v195_data * v206_data)) + (v276_data * v287_data)) + (v357_data * v368_data)) + (v438_data * v449_data)) + (v519_data * v530_data)) + (v600_data * v611_data)) + (v681_data * v692_data)) + (v762_data * v773_data)) + (v843_data * v854_data)) + (v924_data * v935_data)) + (v1005_data * v1016_data)) + (v1086_data * v1097_data)) + (v1167_data * v1178_data);
          float v1183_data = s0[158];
          float v1186_p = ((((((((((((((v33_data * v49_data) + (v114_data * v130_data)) + (v195_data * v211_data)) + (v276_data * v292_data)) + (v357_data * v373_data)) + (v438_data * v454_data)) + (v519_data * v535_data)) + (v600_data * v616_data)) + (v681_data * v697_data)) + (v762_data * v778_data)) + (v843_data * v859_data)) + (v924_data * v940_data)) + (v1005_data * v1021_data)) + (v1086_data * v1102_data)) + (v1167_data * v1183_data);
          float v1188_data = s0[46];
          float v1191_p = ((((((((((((((v33_data * v54_data) + (v114_data * v135_data)) + (v195_data * v216_data)) + (v276_data * v297_data)) + (v357_data * v378_data)) + (v438_data * v459_data)) + (v519_data * v540_data)) + (v600_data * v621_data)) + (v681_data * v702_data)) + (v762_data * v783_data)) + (v843_data * v864_data)) + (v924_data * v945_data)) + (v1005_data * v1026_data)) + (v1086_data * v1107_data)) + (v1167_data * v1188_data);
          float v1193_data = s0[174];
          float v1196_p = ((((((((((((((v33_data * v59_data) + (v114_data * v140_data)) + (v195_data * v221_data)) + (v276_data * v302_data)) + (v357_data * v383_data)) + (v438_data * v464_data)) + (v519_data * v545_data)) + (v600_data * v626_data)) + (v681_data * v707_data)) + (v762_data * v788_data)) + (v843_data * v869_data)) + (v924_data * v950_data)) + (v1005_data * v1031_data)) + (v1086_data * v1112_data)) + (v1167_data * v1193_data);
          float v1198_data = s0[62];
          float v1201_p = ((((((((((((((v33_data * v64_data) + (v114_data * v145_data)) + (v195_data * v226_data)) + (v276_data * v307_data)) + (v357_data * v388_data)) + (v438_data * v469_data)) + (v519_data * v550_data)) + (v600_data * v631_data)) + (v681_data * v712_data)) + (v762_data * v793_data)) + (v843_data * v874_data)) + (v924_data * v955_data)) + (v1005_data * v1036_data)) + (v1086_data * v1117_data)) + (v1167_data * v1198_data);
          float v1203_data = s0[190];
          float v1206_p = ((((((((((((((v33_data * v69_data) + (v114_data * v150_data)) + (v195_data * v231_data)) + (v276_data * v312_data)) + (v357_data * v393_data)) + (v438_data * v474_data)) + (v519_data * v555_data)) + (v600_data * v636_data)) + (v681_data * v717_data)) + (v762_data * v798_data)) + (v843_data * v879_data)) + (v924_data * v960_data)) + (v1005_data * v1041_data)) + (v1086_data * v1122_data)) + (v1167_data * v1203_data);
          float v1208_data = s0[78];
          float v1211_p = ((((((((((((((v33_data * v74_data) + (v114_data * v155_data)) + (v195_data * v236_data)) + (v276_data * v317_data)) + (v357_data * v398_data)) + (v438_data * v479_data)) + (v519_data * v560_data)) + (v600_data * v641_data)) + (v681_data * v722_data)) + (v762_data * v803_data)) + (v843_data * v884_data)) + (v924_data * v965_data)) + (v1005_data * v1046_data)) + (v1086_data * v1127_data)) + (v1167_data * v1208_data);
          float v1213_data = s0[206];
          float v1216_p = ((((((((((((((v33_data * v79_data) + (v114_data * v160_data)) + (v195_data * v241_data)) + (v276_data * v322_data)) + (v357_data * v403_data)) + (v438_data * v484_data)) + (v519_data * v565_data)) + (v600_data * v646_data)) + (v681_data * v727_data)) + (v762_data * v808_data)) + (v843_data * v889_data)) + (v924_data * v970_data)) + (v1005_data * v1051_data)) + (v1086_data * v1132_data)) + (v1167_data * v1213_data);
          float v1218_data = s0[94];
          float v1221_p = ((((((((((((((v33_data * v84_data) + (v114_data * v165_data)) + (v195_data * v246_data)) + (v276_data * v327_data)) + (v357_data * v408_data)) + (v438_data * v489_data)) + (v519_data * v570_data)) + (v600_data * v651_data)) + (v681_data * v732_data)) + (v762_data * v813_data)) + (v843_data * v894_data)) + (v924_data * v975_data)) + (v1005_data * v1056_data)) + (v1086_data * v1137_data)) + (v1167_data * v1218_data);
          float v1223_data = s0[222];
          float v1226_p = ((((((((((((((v33_data * v89_data) + (v114_data * v170_data)) + (v195_data * v251_data)) + (v276_data * v332_data)) + (v357_data * v413_data)) + (v438_data * v494_data)) + (v519_data * v575_data)) + (v600_data * v656_data)) + (v681_data * v737_data)) + (v762_data * v818_data)) + (v843_data * v899_data)) + (v924_data * v980_data)) + (v1005_data * v1061_data)) + (v1086_data * v1142_data)) + (v1167_data * v1223_data);
          float v1228_data = s0[110];
          float v1231_p = ((((((((((((((v33_data * v94_data) + (v114_data * v175_data)) + (v195_data * v256_data)) + (v276_data * v337_data)) + (v357_data * v418_data)) + (v438_data * v499_data)) + (v519_data * v580_data)) + (v600_data * v661_data)) + (v681_data * v742_data)) + (v762_data * v823_data)) + (v843_data * v904_data)) + (v924_data * v985_data)) + (v1005_data * v1066_data)) + (v1086_data * v1147_data)) + (v1167_data * v1228_data);
          float v1233_data = s0[238];
          float v1236_p = ((((((((((((((v33_data * v99_data) + (v114_data * v180_data)) + (v195_data * v261_data)) + (v276_data * v342_data)) + (v357_data * v423_data)) + (v438_data * v504_data)) + (v519_data * v585_data)) + (v600_data * v666_data)) + (v681_data * v747_data)) + (v762_data * v828_data)) + (v843_data * v909_data)) + (v924_data * v990_data)) + (v1005_data * v1071_data)) + (v1086_data * v1152_data)) + (v1167_data * v1233_data);
          float v1238_data = s0[126];
          float v1241_p = ((((((((((((((v33_data * v104_data) + (v114_data * v185_data)) + (v195_data * v266_data)) + (v276_data * v347_data)) + (v357_data * v428_data)) + (v438_data * v509_data)) + (v519_data * v590_data)) + (v600_data * v671_data)) + (v681_data * v752_data)) + (v762_data * v833_data)) + (v843_data * v914_data)) + (v924_data * v995_data)) + (v1005_data * v1076_data)) + (v1086_data * v1157_data)) + (v1167_data * v1238_data);
          float v1243_data = s0[254];
          float v1246_p = ((((((((((((((v33_data * v109_data) + (v114_data * v190_data)) + (v195_data * v271_data)) + (v276_data * v352_data)) + (v357_data * v433_data)) + (v438_data * v514_data)) + (v519_data * v595_data)) + (v600_data * v676_data)) + (v681_data * v757_data)) + (v762_data * v838_data)) + (v843_data * v919_data)) + (v924_data * v1000_data)) + (v1005_data * v1081_data)) + (v1086_data * v1162_data)) + (v1167_data * v1243_data);
          float v1248_data = glb_m1[(v29_lead + 240)];
          float v1249_data = s0[15];
          ir0[0] = ((((((((((((((((v33_data * v34_data) + (v114_data * v115_data)) + (v195_data * v196_data)) + (v276_data * v277_data)) + (v357_data * v358_data)) + (v438_data * v439_data)) + (v519_data * v520_data)) + (v600_data * v601_data)) + (v681_data * v682_data)) + (v762_data * v763_data)) + (v843_data * v844_data)) + (v924_data * v925_data)) + (v1005_data * v1006_data)) + (v1086_data * v1087_data)) + (v1167_data * v1168_data)) + (v1248_data * v1249_data));
          float v1254_data = s0[143];
          ir0[8] = (v1176_p + (v1248_data * v1254_data));
          float v1259_data = s0[31];
          ir0[1] = (v1181_p + (v1248_data * v1259_data));
          float v1264_data = s0[159];
          ir0[9] = (v1186_p + (v1248_data * v1264_data));
          float v1269_data = s0[47];
          ir0[2] = (v1191_p + (v1248_data * v1269_data));
          float v1274_data = s0[175];
          ir0[10] = (v1196_p + (v1248_data * v1274_data));
          float v1279_data = s0[63];
          ir0[3] = (v1201_p + (v1248_data * v1279_data));
          float v1284_data = s0[191];
          ir0[11] = (v1206_p + (v1248_data * v1284_data));
          float v1289_data = s0[79];
          ir0[4] = (v1211_p + (v1248_data * v1289_data));
          float v1294_data = s0[207];
          ir0[12] = (v1216_p + (v1248_data * v1294_data));
          float v1299_data = s0[95];
          ir0[5] = (v1221_p + (v1248_data * v1299_data));
          float v1304_data = s0[223];
          ir0[13] = (v1226_p + (v1248_data * v1304_data));
          float v1309_data = s0[111];
          ir0[6] = (v1231_p + (v1248_data * v1309_data));
          float v1314_data = s0[239];
          ir0[14] = (v1236_p + (v1248_data * v1314_data));
          float v1319_data = s0[127];
          ir0[7] = (v1241_p + (v1248_data * v1319_data));
          float v1324_data = s0[255];
          ir0[15] = (v1246_p + (v1248_data * v1324_data));
          // r0 = ir0
          // glb_m0 = store{r>g}(r0);
          #pragma unroll
          for (int32_t v1339_i0 = 0; v1339_i0 < 1; ++v1339_i0) {
            #pragma unroll
            for (int32_t v1340_i1 = 0; v1340_i1 < 8; ++v1340_i1) {
              #pragma unroll
              for (int32_t v1341_i2 = 0; v1341_i2 < 2; ++v1341_i2) {
                float v1345_data = ir0[(v1340_i1 + (v1341_i2 * 8))];
                if ((v1341_i2 == 0 ? batchIdStack0_0 : batchIdStack0_1)) {
                  float *const glb_m0_se0 = v1341_i2 == 0 ? glb_m0_s0 : glb_m0_s1;
                  glb_m0_se0[((v29_lead + (v1339_i0 * 16)) + (v1340_i1 * 16))] = v1345_data;
                }
              }
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
        else {
          __pipeline_wait_prior(0);
        }
        const bool allowed_next = static_cast<bool>(v1357_flagWordNext);
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
        v1356_flagWord = v1357_flagWordNext;
        v1357_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
  }
}

