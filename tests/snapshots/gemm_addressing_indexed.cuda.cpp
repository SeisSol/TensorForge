// === base name ===
kernel_1829d2f8f1662b6b

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_1829d2f8f1662b6b = {{16, 8, 1}, 16, 16, 1, 8, 8704, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_1829d2f8f1662b6b(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_1829d2f8f1662b6b(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float ** m2, size_t m2_extraOffset, size_t m2_slot, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_1829d2f8f1662b6b(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_1829d2f8f1662b6b, block.x * block.y * block.z, 2176 * sizeof(float));
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
  config.sharedMemBytes = 2176 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_1829d2f8f1662b6b(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float ** m2, size_t m2_extraOffset, size_t m2_slot, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_1829d2f8f1662b6b(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_1829d2f8f1662b6b, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  const float *const * m2Arg = (const float *const *)m2;
  const unsigned * elementIndicesArg = (const unsigned *)elementIndices;
  const unsigned * flags0Arg = (const unsigned *)flags0;
  kernel_kernel_1829d2f8f1662b6b<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m1_stride, m2Arg, m2_extraOffset, m2_slot, numElements0, elementIndicesArg, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_1829d2f8f1662b6b(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float *const * __restrict__ m2, size_t m2_extraOffset, size_t m2_slot, size_t numElements0, const unsigned * __restrict__ elementIndices, const unsigned * __restrict__ flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 8 per block = block 16x8x1, 8704 B shared, occupancy grid
    // operands:
    //   m0 16×16(16×16) {0..16}×{0..16} strided(stride=300, offset=8)
    //   m1 16×16(16×16) {0..16}×{0..16} indexed(stride=runtime)
    //   m2 16×16(16×16) {0..16}×{0..16} indexed_pointer(offset=4, slots=4, slot=runtime)
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":2176}],"shared_bytes":8704,"shared_elements":2176,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided(stride=300, offset=8)","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"indexed(stride=runtime)","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"indexed_pointer(offset=4, slots=4, slot=runtime)","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided(stride=300, offset=8)","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"indexed(stride=runtime)","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"indexed_pointer(offset=4, slots=4, slot=runtime)","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[272 * threadIdx.y + 0];
      size_t v3_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < numElements0) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      float * __restrict__ s0 = &localShrMem0[0];
      float r0[16]{};
      const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
      size_t v1330_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v1330_flagSecondAhead < numElements0) ? v1330_flagSecondAhead : v3_batchId1)];
      const bool allowed_peel = static_cast<bool>(flagWordFirst);
      const float *const __restrict__ peel_glb_m1 = &m1[static_cast<size_t>(elementIndices[v3_batchId1]) * m1_stride + 0 + m1_extraOffset];
      int32_t v1340_lead = threadIdx.x % 16;
      // r0 = load{g>r}(peel_glb_m1);
      #pragma unroll
      for (int32_t v1352_i0 = 0; v1352_i0 < 1; ++v1352_i0) {
        int32_t v1354_lead = v1340_lead + (v1352_i0 * 16);
        #pragma unroll
        for (int32_t v1355_i1 = 0; v1355_i1 < 16; ++v1355_i1) {
          float v1358_data = allowed_peel ? (__ldcg(&peel_glb_m1[(v1354_lead + (v1355_i1 * 16))])) : (0.0f);
          r0[(v1352_i0 + v1355_i1)] = v1358_data;
        }
      }
      const float *const __restrict__ peel_glb_m2 = &m2[static_cast<size_t>(elementIndices[v3_batchId1]) * 4 + m2_slot][0 + 4 + m2_extraOffset];
      // s0 = load{g>s}(peel_glb_m2[0, 1])
      #pragma unroll
      for (int32_t v1370_i = 0; v1370_i < 16; ++v1370_i) {
        int32_t v1372_a = threadIdx.x + (v1370_i * 16);
        if (allowed_peel) {
          __pipeline_memcpy_async(&s0[v1372_a], &peel_glb_m2[v1372_a], 4);
        }
      }
      __pipeline_commit();
      uint32_t v1334_flagWord = flagWordFirst;
      uint32_t v1335_flagWordNext = flagWordSecond;
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const float *const __restrict__ wrap_glb_m1 = &m1[static_cast<size_t>(elementIndices[v11_batchId1]) * m1_stride + 0 + m1_extraOffset];
        const float *const __restrict__ wrap_glb_m2 = &m2[static_cast<size_t>(elementIndices[v11_batchId1]) * 4 + m2_slot][0 + 4 + m2_extraOffset];
        size_t v1382_flagAheadAhead = v11_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v1382_flagAheadAhead < numElements0) ? v1382_flagAheadAhead : v11_batchId1)];
        const bool allowed = static_cast<bool>(v1334_flagWord);
        const bool allowed_next = static_cast<bool>(v1335_flagWordNext);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 300 + 0 + 8 + m0_extraOffset];
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[16]{};
          // ir1 = +(r0 * s0)
          // [(0, 16), (0, 16)] [(0, 16)]
          float ir1[16]{};
          float v37_data = r0[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v38_data = s0[0];
          float v43_data = s0[16];
          float v48_data = s0[32];
          float v53_data = s0[48];
          float v58_data = s0[64];
          float v63_data = s0[80];
          float v68_data = s0[96];
          float v73_data = s0[112];
          float v78_data = s0[128];
          float v83_data = s0[144];
          float v88_data = s0[160];
          float v93_data = s0[176];
          float v98_data = s0[192];
          float v103_data = s0[208];
          float v108_data = s0[224];
          float v113_data = s0[240];
          float v117_data = r0[1];
          float v118_data = s0[1];
          float v123_data = s0[17];
          float v128_data = s0[33];
          float v133_data = s0[49];
          float v138_data = s0[65];
          float v143_data = s0[81];
          float v148_data = s0[97];
          float v153_data = s0[113];
          float v158_data = s0[129];
          float v163_data = s0[145];
          float v168_data = s0[161];
          float v173_data = s0[177];
          float v178_data = s0[193];
          float v183_data = s0[209];
          float v188_data = s0[225];
          float v193_data = s0[241];
          float v197_data = r0[2];
          float v198_data = s0[2];
          float v203_data = s0[18];
          float v208_data = s0[34];
          float v213_data = s0[50];
          float v218_data = s0[66];
          float v223_data = s0[82];
          float v228_data = s0[98];
          float v233_data = s0[114];
          float v238_data = s0[130];
          float v243_data = s0[146];
          float v248_data = s0[162];
          float v253_data = s0[178];
          float v258_data = s0[194];
          float v263_data = s0[210];
          float v268_data = s0[226];
          float v273_data = s0[242];
          float v277_data = r0[3];
          float v278_data = s0[3];
          float v283_data = s0[19];
          float v288_data = s0[35];
          float v293_data = s0[51];
          float v298_data = s0[67];
          float v303_data = s0[83];
          float v308_data = s0[99];
          float v313_data = s0[115];
          float v318_data = s0[131];
          float v323_data = s0[147];
          float v328_data = s0[163];
          float v333_data = s0[179];
          float v338_data = s0[195];
          float v343_data = s0[211];
          float v348_data = s0[227];
          float v353_data = s0[243];
          float v357_data = r0[4];
          float v358_data = s0[4];
          float v363_data = s0[20];
          float v368_data = s0[36];
          float v373_data = s0[52];
          float v378_data = s0[68];
          float v383_data = s0[84];
          float v388_data = s0[100];
          float v393_data = s0[116];
          float v398_data = s0[132];
          float v403_data = s0[148];
          float v408_data = s0[164];
          float v413_data = s0[180];
          float v418_data = s0[196];
          float v423_data = s0[212];
          float v428_data = s0[228];
          float v433_data = s0[244];
          float v437_data = r0[5];
          float v438_data = s0[5];
          float v443_data = s0[21];
          float v448_data = s0[37];
          float v453_data = s0[53];
          float v458_data = s0[69];
          float v463_data = s0[85];
          float v468_data = s0[101];
          float v473_data = s0[117];
          float v478_data = s0[133];
          float v483_data = s0[149];
          float v488_data = s0[165];
          float v493_data = s0[181];
          float v498_data = s0[197];
          float v503_data = s0[213];
          float v508_data = s0[229];
          float v513_data = s0[245];
          float v517_data = r0[6];
          float v518_data = s0[6];
          float v523_data = s0[22];
          float v528_data = s0[38];
          float v533_data = s0[54];
          float v538_data = s0[70];
          float v543_data = s0[86];
          float v548_data = s0[102];
          float v553_data = s0[118];
          float v558_data = s0[134];
          float v563_data = s0[150];
          float v568_data = s0[166];
          float v573_data = s0[182];
          float v578_data = s0[198];
          float v583_data = s0[214];
          float v588_data = s0[230];
          float v593_data = s0[246];
          float v597_data = r0[7];
          float v598_data = s0[7];
          float v603_data = s0[23];
          float v608_data = s0[39];
          float v613_data = s0[55];
          float v618_data = s0[71];
          float v623_data = s0[87];
          float v628_data = s0[103];
          float v633_data = s0[119];
          float v638_data = s0[135];
          float v643_data = s0[151];
          float v648_data = s0[167];
          float v653_data = s0[183];
          float v658_data = s0[199];
          float v663_data = s0[215];
          float v668_data = s0[231];
          float v673_data = s0[247];
          float v677_data = r0[8];
          float v678_data = s0[8];
          float v683_data = s0[24];
          float v688_data = s0[40];
          float v693_data = s0[56];
          float v698_data = s0[72];
          float v703_data = s0[88];
          float v708_data = s0[104];
          float v713_data = s0[120];
          float v718_data = s0[136];
          float v723_data = s0[152];
          float v728_data = s0[168];
          float v733_data = s0[184];
          float v738_data = s0[200];
          float v743_data = s0[216];
          float v748_data = s0[232];
          float v753_data = s0[248];
          float v757_data = r0[9];
          float v758_data = s0[9];
          float v763_data = s0[25];
          float v768_data = s0[41];
          float v773_data = s0[57];
          float v778_data = s0[73];
          float v783_data = s0[89];
          float v788_data = s0[105];
          float v793_data = s0[121];
          float v798_data = s0[137];
          float v803_data = s0[153];
          float v808_data = s0[169];
          float v813_data = s0[185];
          float v818_data = s0[201];
          float v823_data = s0[217];
          float v828_data = s0[233];
          float v833_data = s0[249];
          float v837_data = r0[10];
          float v838_data = s0[10];
          float v843_data = s0[26];
          float v848_data = s0[42];
          float v853_data = s0[58];
          float v858_data = s0[74];
          float v863_data = s0[90];
          float v868_data = s0[106];
          float v873_data = s0[122];
          float v878_data = s0[138];
          float v883_data = s0[154];
          float v888_data = s0[170];
          float v893_data = s0[186];
          float v898_data = s0[202];
          float v903_data = s0[218];
          float v908_data = s0[234];
          float v913_data = s0[250];
          float v917_data = r0[11];
          float v918_data = s0[11];
          float v923_data = s0[27];
          float v928_data = s0[43];
          float v933_data = s0[59];
          float v938_data = s0[75];
          float v943_data = s0[91];
          float v948_data = s0[107];
          float v953_data = s0[123];
          float v958_data = s0[139];
          float v963_data = s0[155];
          float v968_data = s0[171];
          float v973_data = s0[187];
          float v978_data = s0[203];
          float v983_data = s0[219];
          float v988_data = s0[235];
          float v993_data = s0[251];
          float v997_data = r0[12];
          float v998_data = s0[12];
          float v1003_data = s0[28];
          float v1008_data = s0[44];
          float v1013_data = s0[60];
          float v1018_data = s0[76];
          float v1023_data = s0[92];
          float v1028_data = s0[108];
          float v1033_data = s0[124];
          float v1038_data = s0[140];
          float v1043_data = s0[156];
          float v1048_data = s0[172];
          float v1053_data = s0[188];
          float v1058_data = s0[204];
          float v1063_data = s0[220];
          float v1068_data = s0[236];
          float v1073_data = s0[252];
          float v1077_data = r0[13];
          float v1078_data = s0[13];
          float v1083_data = s0[29];
          float v1088_data = s0[45];
          float v1093_data = s0[61];
          float v1098_data = s0[77];
          float v1103_data = s0[93];
          float v1108_data = s0[109];
          float v1113_data = s0[125];
          float v1118_data = s0[141];
          float v1123_data = s0[157];
          float v1128_data = s0[173];
          float v1133_data = s0[189];
          float v1138_data = s0[205];
          float v1143_data = s0[221];
          float v1148_data = s0[237];
          float v1153_data = s0[253];
          float v1157_data = r0[14];
          float v1158_data = s0[14];
          float v1161_p = ((((((((((((((v37_data * v38_data) + (v117_data * v118_data)) + (v197_data * v198_data)) + (v277_data * v278_data)) + (v357_data * v358_data)) + (v437_data * v438_data)) + (v517_data * v518_data)) + (v597_data * v598_data)) + (v677_data * v678_data)) + (v757_data * v758_data)) + (v837_data * v838_data)) + (v917_data * v918_data)) + (v997_data * v998_data)) + (v1077_data * v1078_data)) + (v1157_data * v1158_data);
          float v1163_data = s0[30];
          float v1166_p = ((((((((((((((v37_data * v43_data) + (v117_data * v123_data)) + (v197_data * v203_data)) + (v277_data * v283_data)) + (v357_data * v363_data)) + (v437_data * v443_data)) + (v517_data * v523_data)) + (v597_data * v603_data)) + (v677_data * v683_data)) + (v757_data * v763_data)) + (v837_data * v843_data)) + (v917_data * v923_data)) + (v997_data * v1003_data)) + (v1077_data * v1083_data)) + (v1157_data * v1163_data);
          float v1168_data = s0[46];
          float v1171_p = ((((((((((((((v37_data * v48_data) + (v117_data * v128_data)) + (v197_data * v208_data)) + (v277_data * v288_data)) + (v357_data * v368_data)) + (v437_data * v448_data)) + (v517_data * v528_data)) + (v597_data * v608_data)) + (v677_data * v688_data)) + (v757_data * v768_data)) + (v837_data * v848_data)) + (v917_data * v928_data)) + (v997_data * v1008_data)) + (v1077_data * v1088_data)) + (v1157_data * v1168_data);
          float v1173_data = s0[62];
          float v1176_p = ((((((((((((((v37_data * v53_data) + (v117_data * v133_data)) + (v197_data * v213_data)) + (v277_data * v293_data)) + (v357_data * v373_data)) + (v437_data * v453_data)) + (v517_data * v533_data)) + (v597_data * v613_data)) + (v677_data * v693_data)) + (v757_data * v773_data)) + (v837_data * v853_data)) + (v917_data * v933_data)) + (v997_data * v1013_data)) + (v1077_data * v1093_data)) + (v1157_data * v1173_data);
          float v1178_data = s0[78];
          float v1181_p = ((((((((((((((v37_data * v58_data) + (v117_data * v138_data)) + (v197_data * v218_data)) + (v277_data * v298_data)) + (v357_data * v378_data)) + (v437_data * v458_data)) + (v517_data * v538_data)) + (v597_data * v618_data)) + (v677_data * v698_data)) + (v757_data * v778_data)) + (v837_data * v858_data)) + (v917_data * v938_data)) + (v997_data * v1018_data)) + (v1077_data * v1098_data)) + (v1157_data * v1178_data);
          float v1183_data = s0[94];
          float v1186_p = ((((((((((((((v37_data * v63_data) + (v117_data * v143_data)) + (v197_data * v223_data)) + (v277_data * v303_data)) + (v357_data * v383_data)) + (v437_data * v463_data)) + (v517_data * v543_data)) + (v597_data * v623_data)) + (v677_data * v703_data)) + (v757_data * v783_data)) + (v837_data * v863_data)) + (v917_data * v943_data)) + (v997_data * v1023_data)) + (v1077_data * v1103_data)) + (v1157_data * v1183_data);
          float v1188_data = s0[110];
          float v1191_p = ((((((((((((((v37_data * v68_data) + (v117_data * v148_data)) + (v197_data * v228_data)) + (v277_data * v308_data)) + (v357_data * v388_data)) + (v437_data * v468_data)) + (v517_data * v548_data)) + (v597_data * v628_data)) + (v677_data * v708_data)) + (v757_data * v788_data)) + (v837_data * v868_data)) + (v917_data * v948_data)) + (v997_data * v1028_data)) + (v1077_data * v1108_data)) + (v1157_data * v1188_data);
          float v1193_data = s0[126];
          float v1196_p = ((((((((((((((v37_data * v73_data) + (v117_data * v153_data)) + (v197_data * v233_data)) + (v277_data * v313_data)) + (v357_data * v393_data)) + (v437_data * v473_data)) + (v517_data * v553_data)) + (v597_data * v633_data)) + (v677_data * v713_data)) + (v757_data * v793_data)) + (v837_data * v873_data)) + (v917_data * v953_data)) + (v997_data * v1033_data)) + (v1077_data * v1113_data)) + (v1157_data * v1193_data);
          float v1198_data = s0[142];
          float v1201_p = ((((((((((((((v37_data * v78_data) + (v117_data * v158_data)) + (v197_data * v238_data)) + (v277_data * v318_data)) + (v357_data * v398_data)) + (v437_data * v478_data)) + (v517_data * v558_data)) + (v597_data * v638_data)) + (v677_data * v718_data)) + (v757_data * v798_data)) + (v837_data * v878_data)) + (v917_data * v958_data)) + (v997_data * v1038_data)) + (v1077_data * v1118_data)) + (v1157_data * v1198_data);
          float v1203_data = s0[158];
          float v1206_p = ((((((((((((((v37_data * v83_data) + (v117_data * v163_data)) + (v197_data * v243_data)) + (v277_data * v323_data)) + (v357_data * v403_data)) + (v437_data * v483_data)) + (v517_data * v563_data)) + (v597_data * v643_data)) + (v677_data * v723_data)) + (v757_data * v803_data)) + (v837_data * v883_data)) + (v917_data * v963_data)) + (v997_data * v1043_data)) + (v1077_data * v1123_data)) + (v1157_data * v1203_data);
          float v1208_data = s0[174];
          float v1211_p = ((((((((((((((v37_data * v88_data) + (v117_data * v168_data)) + (v197_data * v248_data)) + (v277_data * v328_data)) + (v357_data * v408_data)) + (v437_data * v488_data)) + (v517_data * v568_data)) + (v597_data * v648_data)) + (v677_data * v728_data)) + (v757_data * v808_data)) + (v837_data * v888_data)) + (v917_data * v968_data)) + (v997_data * v1048_data)) + (v1077_data * v1128_data)) + (v1157_data * v1208_data);
          float v1213_data = s0[190];
          float v1216_p = ((((((((((((((v37_data * v93_data) + (v117_data * v173_data)) + (v197_data * v253_data)) + (v277_data * v333_data)) + (v357_data * v413_data)) + (v437_data * v493_data)) + (v517_data * v573_data)) + (v597_data * v653_data)) + (v677_data * v733_data)) + (v757_data * v813_data)) + (v837_data * v893_data)) + (v917_data * v973_data)) + (v997_data * v1053_data)) + (v1077_data * v1133_data)) + (v1157_data * v1213_data);
          float v1218_data = s0[206];
          float v1221_p = ((((((((((((((v37_data * v98_data) + (v117_data * v178_data)) + (v197_data * v258_data)) + (v277_data * v338_data)) + (v357_data * v418_data)) + (v437_data * v498_data)) + (v517_data * v578_data)) + (v597_data * v658_data)) + (v677_data * v738_data)) + (v757_data * v818_data)) + (v837_data * v898_data)) + (v917_data * v978_data)) + (v997_data * v1058_data)) + (v1077_data * v1138_data)) + (v1157_data * v1218_data);
          float v1223_data = s0[222];
          float v1226_p = ((((((((((((((v37_data * v103_data) + (v117_data * v183_data)) + (v197_data * v263_data)) + (v277_data * v343_data)) + (v357_data * v423_data)) + (v437_data * v503_data)) + (v517_data * v583_data)) + (v597_data * v663_data)) + (v677_data * v743_data)) + (v757_data * v823_data)) + (v837_data * v903_data)) + (v917_data * v983_data)) + (v997_data * v1063_data)) + (v1077_data * v1143_data)) + (v1157_data * v1223_data);
          float v1228_data = s0[238];
          float v1231_p = ((((((((((((((v37_data * v108_data) + (v117_data * v188_data)) + (v197_data * v268_data)) + (v277_data * v348_data)) + (v357_data * v428_data)) + (v437_data * v508_data)) + (v517_data * v588_data)) + (v597_data * v668_data)) + (v677_data * v748_data)) + (v757_data * v828_data)) + (v837_data * v908_data)) + (v917_data * v988_data)) + (v997_data * v1068_data)) + (v1077_data * v1148_data)) + (v1157_data * v1228_data);
          float v1233_data = s0[254];
          float v1236_p = ((((((((((((((v37_data * v113_data) + (v117_data * v193_data)) + (v197_data * v273_data)) + (v277_data * v353_data)) + (v357_data * v433_data)) + (v437_data * v513_data)) + (v517_data * v593_data)) + (v597_data * v673_data)) + (v677_data * v753_data)) + (v757_data * v833_data)) + (v837_data * v913_data)) + (v917_data * v993_data)) + (v997_data * v1073_data)) + (v1077_data * v1153_data)) + (v1157_data * v1233_data);
          float v1237_data = r0[15];
          // r0 = load{g>r}(wrap_glb_m1);
          #pragma unroll
          for (int32_t v1360_i0 = 0; v1360_i0 < 1; ++v1360_i0) {
            int32_t v1362_lead = v1340_lead + (v1360_i0 * 16);
            #pragma unroll
            for (int32_t v1363_i1 = 0; v1363_i1 < 16; ++v1363_i1) {
              float v1366_data = allowed_next ? (__ldcg(&wrap_glb_m1[(v1362_lead + (v1363_i1 * 16))])) : (0.0f);
              r0[(v1360_i0 + v1363_i1)] = v1366_data;
            }
          }
          float v1238_data = s0[15];
          ir1[0] = (v1161_p + (v1237_data * v1238_data));
          float v1243_data = s0[31];
          ir1[1] = (v1166_p + (v1237_data * v1243_data));
          float v1248_data = s0[47];
          ir1[2] = (v1171_p + (v1237_data * v1248_data));
          float v1253_data = s0[63];
          ir1[3] = (v1176_p + (v1237_data * v1253_data));
          float v1258_data = s0[79];
          ir1[4] = (v1181_p + (v1237_data * v1258_data));
          float v1263_data = s0[95];
          ir1[5] = (v1186_p + (v1237_data * v1263_data));
          float v1268_data = s0[111];
          ir1[6] = (v1191_p + (v1237_data * v1268_data));
          float v1273_data = s0[127];
          ir1[7] = (v1196_p + (v1237_data * v1273_data));
          float v1278_data = s0[143];
          ir1[8] = (v1201_p + (v1237_data * v1278_data));
          float v1283_data = s0[159];
          ir1[9] = (v1206_p + (v1237_data * v1283_data));
          float v1288_data = s0[175];
          ir1[10] = (v1211_p + (v1237_data * v1288_data));
          float v1293_data = s0[191];
          ir1[11] = (v1216_p + (v1237_data * v1293_data));
          float v1298_data = s0[207];
          ir1[12] = (v1221_p + (v1237_data * v1298_data));
          float v1303_data = s0[223];
          ir1[13] = (v1226_p + (v1237_data * v1303_data));
          float v1308_data = s0[239];
          ir1[14] = (v1231_p + (v1237_data * v1308_data));
          float v1313_data = s0[255];
          ir1[15] = (v1236_p + (v1237_data * v1313_data));
          // r1 = ir1
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v1321_i0 = 0; v1321_i0 < 1; ++v1321_i0) {
            int32_t v1326_lead = v1340_lead + (v1321_i0 * 16);
            #pragma unroll
            for (int32_t v1322_i1 = 0; v1322_i1 < 16; ++v1322_i1) {
              float v1324_data = ir1[v1322_i1];
              glb_m0[(v1326_lead + (v1322_i1 * 16))] = v1324_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
        else {
          __pipeline_wait_prior(0);
          // r0 = load{g>r}(wrap_glb_m1);
          #pragma unroll
          for (int32_t v1344_i0 = 0; v1344_i0 < 1; ++v1344_i0) {
            int32_t v1346_lead = v1340_lead + (v1344_i0 * 16);
            #pragma unroll
            for (int32_t v1347_i1 = 0; v1347_i1 < 16; ++v1347_i1) {
              float v1350_data = allowed_next ? (__ldcg(&wrap_glb_m1[(v1346_lead + (v1347_i1 * 16))])) : (0.0f);
              r0[(v1344_i0 + v1347_i1)] = v1350_data;
            }
          }
        }
        // s0 = load{g>s}(wrap_glb_m2[0, 1])
        #pragma unroll
        for (int32_t v1374_i = 0; v1374_i < 16; ++v1374_i) {
          int32_t v1376_a = threadIdx.x + (v1374_i * 16);
          if (allowed_next) {
            __pipeline_memcpy_async(&s0[v1376_a], &wrap_glb_m2[v1376_a], 4);
          }
        }
        __pipeline_commit();
        v1334_flagWord = v1335_flagWordNext;
        v1335_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
  }
}

