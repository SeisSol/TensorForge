// === base name ===
kernel_3724e0aea9a330e3

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_3724e0aea9a330e3 = {{32, 4, 1}, 32, 32, 1, 4, 4096, false, true, 2};
tensorforge::LaunchConfig launch_config_kernel_3724e0aea9a330e3(size_t numElements0, size_t numElements1, void* streamPtr = nullptr);
void launcher_kernel_3724e0aea9a330e3(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, size_t numElements0, size_t numElements1, unsigned * flags0 = nullptr, unsigned * flags1 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_3724e0aea9a330e3(size_t numElements0, size_t numElements1, void* streamPtr) {
  (void)numElements0;
  (void)numElements1;
  (void)streamPtr;
  dim3 block (32, 4, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_3724e0aea9a330e3, block.x * block.y * block.z, 1024 * sizeof(float));
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
  config.sharedMemBytes = 1024 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_3724e0aea9a330e3(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, size_t numElements0, size_t numElements1, unsigned * flags0, unsigned * flags1, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_3724e0aea9a330e3(numElements0, numElements1, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_3724e0aea9a330e3, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  const unsigned * flags0Arg = (const unsigned *)flags0;
  const unsigned * flags1Arg = (const unsigned *)flags1;
  kernel_kernel_3724e0aea9a330e3<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, numElements0, numElements1, flags0Arg, flags1Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_3724e0aea9a330e3(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, size_t numElements0, size_t numElements1, const unsigned * __restrict__ flags0, const unsigned * __restrict__ flags1) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 4 per block = block 32x4x1, 4096 B shared, occupancy grid
    // operands:
    //   m0 16×16(16×16) {0..16}×{0..16} strided
    //   m1 16×16(16×16) {0..16}×{0..16} strided
    //   m2 16×16(16×16) {0..16}×{0..16} strided
    //   m3 16×16(16×16) {0..16}×{0..16} strided
    //   m4 16×16(16×16) {0..16}×{0..16} strided
    //   m5 16×16(16×16) {0..16}×{0..16} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    //   siblings
    //   m3[i,j] = m4[i,k] × m5[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":1024},{"barrier":false,"mults_per_block":4,"shared_elements":1024}],"shared_bytes":4096,"shared_elements":1024,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"E","bbox":[[0,0],[16,16]],"name":"m3","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m4","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"F","bbox":[[0,0],[16,16]],"name":"m5","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"kind":"siblings"},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[256 * threadIdx.y + 0];
      size_t v3_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < numElements0) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      float * __restrict__ s0 = &localShrMem0[0];
      float r0[16]{};
      const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
      size_t v1333_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v1333_flagSecondAhead < numElements0) ? v1333_flagSecondAhead : v3_batchId1)];
      const float *const __restrict__ peel_glb_m1 = &m1[v3_batchId1 * 256 + 0 + m1_extraOffset];
      int32_t v1341_lead = threadIdx.x % 32;
      bool v1342_g = v1341_lead < 16;
      // r0 = load{g>r}(peel_glb_m1);
      #pragma unroll
      for (int32_t v1351_i1 = 0; v1351_i1 < 16; ++v1351_i1) {
        float v1354_data = v1342_g ? (__ldcg(&peel_glb_m1[(v1341_lead + (v1351_i1 * 16))])) : (0.0f);
        r0[v1351_i1] = v1354_data;
      }
      const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 256 + 0 + m2_extraOffset];
      // s0 = load{g>s}(peel_glb_m2[0, 1])
      #pragma unroll
      for (int32_t v1361_i = 0; v1361_i < 8; ++v1361_i) {
        int32_t v1363_a = threadIdx.x + (v1361_i * 32);
        __pipeline_memcpy_async(&s0[v1363_a], &peel_glb_m2[v1363_a], 4);
      }
      __pipeline_commit();
      uint32_t v1337_flagWord = flagWordFirst;
      uint32_t v1338_flagWordNext = flagWordSecond;
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const float *const __restrict__ wrap_glb_m1 = &m1[v11_batchId1 * 256 + 0 + m1_extraOffset];
        const float *const __restrict__ wrap_glb_m2 = &m2[v11_batchId1 * 256 + 0 + m2_extraOffset];
        size_t v1373_flagAheadAhead = v11_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v1373_flagAheadAhead < numElements0) ? v1373_flagAheadAhead : v11_batchId1)];
        const bool allowed = static_cast<bool>(v1337_flagWord);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 256 + 0 + m0_extraOffset];
          float *const __restrict__ glb_m3 = &m3[v8_batchId0 * 256 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[v8_batchId0 * 256 + 0 + m4_extraOffset];
          const float *const __restrict__ glb_m5 = &m5[v8_batchId0 * 256 + 0 + m5_extraOffset];
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[16]{};
          // ir1 = +(r0 * s0)
          // [(0, 16), (0, 16)] [(0, 16)]
          float ir1[16]{};
          float v40_data = r0[0];
          __syncwarp();
          float v41_data = s0[0];
          float v46_data = s0[16];
          float v51_data = s0[32];
          float v56_data = s0[48];
          float v61_data = s0[64];
          float v66_data = s0[80];
          float v71_data = s0[96];
          float v76_data = s0[112];
          float v81_data = s0[128];
          float v86_data = s0[144];
          float v91_data = s0[160];
          float v96_data = s0[176];
          float v101_data = s0[192];
          float v106_data = s0[208];
          float v111_data = s0[224];
          float v116_data = s0[240];
          float v120_data = r0[1];
          float v121_data = s0[1];
          float v126_data = s0[17];
          float v131_data = s0[33];
          float v136_data = s0[49];
          float v141_data = s0[65];
          float v146_data = s0[81];
          float v151_data = s0[97];
          float v156_data = s0[113];
          float v161_data = s0[129];
          float v166_data = s0[145];
          float v171_data = s0[161];
          float v176_data = s0[177];
          float v181_data = s0[193];
          float v186_data = s0[209];
          float v191_data = s0[225];
          float v196_data = s0[241];
          float v200_data = r0[2];
          float v201_data = s0[2];
          float v206_data = s0[18];
          float v211_data = s0[34];
          float v216_data = s0[50];
          float v221_data = s0[66];
          float v226_data = s0[82];
          float v231_data = s0[98];
          float v236_data = s0[114];
          float v241_data = s0[130];
          float v246_data = s0[146];
          float v251_data = s0[162];
          float v256_data = s0[178];
          float v261_data = s0[194];
          float v266_data = s0[210];
          float v271_data = s0[226];
          float v276_data = s0[242];
          float v280_data = r0[3];
          float v281_data = s0[3];
          float v286_data = s0[19];
          float v291_data = s0[35];
          float v296_data = s0[51];
          float v301_data = s0[67];
          float v306_data = s0[83];
          float v311_data = s0[99];
          float v316_data = s0[115];
          float v321_data = s0[131];
          float v326_data = s0[147];
          float v331_data = s0[163];
          float v336_data = s0[179];
          float v341_data = s0[195];
          float v346_data = s0[211];
          float v351_data = s0[227];
          float v356_data = s0[243];
          float v360_data = r0[4];
          float v361_data = s0[4];
          float v366_data = s0[20];
          float v371_data = s0[36];
          float v376_data = s0[52];
          float v381_data = s0[68];
          float v386_data = s0[84];
          float v391_data = s0[100];
          float v396_data = s0[116];
          float v401_data = s0[132];
          float v406_data = s0[148];
          float v411_data = s0[164];
          float v416_data = s0[180];
          float v421_data = s0[196];
          float v426_data = s0[212];
          float v431_data = s0[228];
          float v436_data = s0[244];
          float v440_data = r0[5];
          float v441_data = s0[5];
          float v446_data = s0[21];
          float v451_data = s0[37];
          float v456_data = s0[53];
          float v461_data = s0[69];
          float v466_data = s0[85];
          float v471_data = s0[101];
          float v476_data = s0[117];
          float v481_data = s0[133];
          float v486_data = s0[149];
          float v491_data = s0[165];
          float v496_data = s0[181];
          float v501_data = s0[197];
          float v506_data = s0[213];
          float v511_data = s0[229];
          float v516_data = s0[245];
          float v520_data = r0[6];
          float v521_data = s0[6];
          float v526_data = s0[22];
          float v531_data = s0[38];
          float v536_data = s0[54];
          float v541_data = s0[70];
          float v546_data = s0[86];
          float v551_data = s0[102];
          float v556_data = s0[118];
          float v561_data = s0[134];
          float v566_data = s0[150];
          float v571_data = s0[166];
          float v576_data = s0[182];
          float v581_data = s0[198];
          float v586_data = s0[214];
          float v591_data = s0[230];
          float v596_data = s0[246];
          float v600_data = r0[7];
          float v601_data = s0[7];
          float v606_data = s0[23];
          float v611_data = s0[39];
          float v616_data = s0[55];
          float v621_data = s0[71];
          float v626_data = s0[87];
          float v631_data = s0[103];
          float v636_data = s0[119];
          float v641_data = s0[135];
          float v646_data = s0[151];
          float v651_data = s0[167];
          float v656_data = s0[183];
          float v661_data = s0[199];
          float v666_data = s0[215];
          float v671_data = s0[231];
          float v676_data = s0[247];
          float v680_data = r0[8];
          float v681_data = s0[8];
          float v686_data = s0[24];
          float v691_data = s0[40];
          float v696_data = s0[56];
          float v701_data = s0[72];
          float v706_data = s0[88];
          float v711_data = s0[104];
          float v716_data = s0[120];
          float v721_data = s0[136];
          float v726_data = s0[152];
          float v731_data = s0[168];
          float v736_data = s0[184];
          float v741_data = s0[200];
          float v746_data = s0[216];
          float v751_data = s0[232];
          float v756_data = s0[248];
          float v760_data = r0[9];
          float v761_data = s0[9];
          float v766_data = s0[25];
          float v771_data = s0[41];
          float v776_data = s0[57];
          float v781_data = s0[73];
          float v786_data = s0[89];
          float v791_data = s0[105];
          float v796_data = s0[121];
          float v801_data = s0[137];
          float v806_data = s0[153];
          float v811_data = s0[169];
          float v816_data = s0[185];
          float v821_data = s0[201];
          float v826_data = s0[217];
          float v831_data = s0[233];
          float v836_data = s0[249];
          float v840_data = r0[10];
          float v841_data = s0[10];
          float v846_data = s0[26];
          float v851_data = s0[42];
          float v856_data = s0[58];
          float v861_data = s0[74];
          float v866_data = s0[90];
          float v871_data = s0[106];
          float v876_data = s0[122];
          float v881_data = s0[138];
          float v886_data = s0[154];
          float v891_data = s0[170];
          float v896_data = s0[186];
          float v901_data = s0[202];
          float v906_data = s0[218];
          float v911_data = s0[234];
          float v916_data = s0[250];
          float v920_data = r0[11];
          float v921_data = s0[11];
          float v926_data = s0[27];
          float v931_data = s0[43];
          float v936_data = s0[59];
          float v941_data = s0[75];
          float v946_data = s0[91];
          float v951_data = s0[107];
          float v956_data = s0[123];
          float v961_data = s0[139];
          float v966_data = s0[155];
          float v971_data = s0[171];
          float v976_data = s0[187];
          float v981_data = s0[203];
          float v986_data = s0[219];
          float v991_data = s0[235];
          float v996_data = s0[251];
          float v1000_data = r0[12];
          float v1001_data = s0[12];
          float v1006_data = s0[28];
          float v1011_data = s0[44];
          float v1016_data = s0[60];
          float v1021_data = s0[76];
          float v1026_data = s0[92];
          float v1031_data = s0[108];
          float v1036_data = s0[124];
          float v1041_data = s0[140];
          float v1046_data = s0[156];
          float v1051_data = s0[172];
          float v1056_data = s0[188];
          float v1061_data = s0[204];
          float v1066_data = s0[220];
          float v1071_data = s0[236];
          float v1076_data = s0[252];
          float v1080_data = r0[13];
          float v1081_data = s0[13];
          float v1086_data = s0[29];
          float v1091_data = s0[45];
          float v1096_data = s0[61];
          float v1101_data = s0[77];
          float v1106_data = s0[93];
          float v1111_data = s0[109];
          float v1116_data = s0[125];
          float v1121_data = s0[141];
          float v1126_data = s0[157];
          float v1131_data = s0[173];
          float v1136_data = s0[189];
          float v1141_data = s0[205];
          float v1146_data = s0[221];
          float v1151_data = s0[237];
          float v1156_data = s0[253];
          float v1160_data = r0[14];
          float v1161_data = s0[14];
          float v1164_p = ((((((((((((((v40_data * v41_data) + (v120_data * v121_data)) + (v200_data * v201_data)) + (v280_data * v281_data)) + (v360_data * v361_data)) + (v440_data * v441_data)) + (v520_data * v521_data)) + (v600_data * v601_data)) + (v680_data * v681_data)) + (v760_data * v761_data)) + (v840_data * v841_data)) + (v920_data * v921_data)) + (v1000_data * v1001_data)) + (v1080_data * v1081_data)) + (v1160_data * v1161_data);
          float v1166_data = s0[30];
          float v1169_p = ((((((((((((((v40_data * v46_data) + (v120_data * v126_data)) + (v200_data * v206_data)) + (v280_data * v286_data)) + (v360_data * v366_data)) + (v440_data * v446_data)) + (v520_data * v526_data)) + (v600_data * v606_data)) + (v680_data * v686_data)) + (v760_data * v766_data)) + (v840_data * v846_data)) + (v920_data * v926_data)) + (v1000_data * v1006_data)) + (v1080_data * v1086_data)) + (v1160_data * v1166_data);
          float v1171_data = s0[46];
          float v1174_p = ((((((((((((((v40_data * v51_data) + (v120_data * v131_data)) + (v200_data * v211_data)) + (v280_data * v291_data)) + (v360_data * v371_data)) + (v440_data * v451_data)) + (v520_data * v531_data)) + (v600_data * v611_data)) + (v680_data * v691_data)) + (v760_data * v771_data)) + (v840_data * v851_data)) + (v920_data * v931_data)) + (v1000_data * v1011_data)) + (v1080_data * v1091_data)) + (v1160_data * v1171_data);
          float v1176_data = s0[62];
          float v1179_p = ((((((((((((((v40_data * v56_data) + (v120_data * v136_data)) + (v200_data * v216_data)) + (v280_data * v296_data)) + (v360_data * v376_data)) + (v440_data * v456_data)) + (v520_data * v536_data)) + (v600_data * v616_data)) + (v680_data * v696_data)) + (v760_data * v776_data)) + (v840_data * v856_data)) + (v920_data * v936_data)) + (v1000_data * v1016_data)) + (v1080_data * v1096_data)) + (v1160_data * v1176_data);
          float v1181_data = s0[78];
          float v1184_p = ((((((((((((((v40_data * v61_data) + (v120_data * v141_data)) + (v200_data * v221_data)) + (v280_data * v301_data)) + (v360_data * v381_data)) + (v440_data * v461_data)) + (v520_data * v541_data)) + (v600_data * v621_data)) + (v680_data * v701_data)) + (v760_data * v781_data)) + (v840_data * v861_data)) + (v920_data * v941_data)) + (v1000_data * v1021_data)) + (v1080_data * v1101_data)) + (v1160_data * v1181_data);
          float v1186_data = s0[94];
          float v1189_p = ((((((((((((((v40_data * v66_data) + (v120_data * v146_data)) + (v200_data * v226_data)) + (v280_data * v306_data)) + (v360_data * v386_data)) + (v440_data * v466_data)) + (v520_data * v546_data)) + (v600_data * v626_data)) + (v680_data * v706_data)) + (v760_data * v786_data)) + (v840_data * v866_data)) + (v920_data * v946_data)) + (v1000_data * v1026_data)) + (v1080_data * v1106_data)) + (v1160_data * v1186_data);
          float v1191_data = s0[110];
          float v1194_p = ((((((((((((((v40_data * v71_data) + (v120_data * v151_data)) + (v200_data * v231_data)) + (v280_data * v311_data)) + (v360_data * v391_data)) + (v440_data * v471_data)) + (v520_data * v551_data)) + (v600_data * v631_data)) + (v680_data * v711_data)) + (v760_data * v791_data)) + (v840_data * v871_data)) + (v920_data * v951_data)) + (v1000_data * v1031_data)) + (v1080_data * v1111_data)) + (v1160_data * v1191_data);
          float v1196_data = s0[126];
          float v1199_p = ((((((((((((((v40_data * v76_data) + (v120_data * v156_data)) + (v200_data * v236_data)) + (v280_data * v316_data)) + (v360_data * v396_data)) + (v440_data * v476_data)) + (v520_data * v556_data)) + (v600_data * v636_data)) + (v680_data * v716_data)) + (v760_data * v796_data)) + (v840_data * v876_data)) + (v920_data * v956_data)) + (v1000_data * v1036_data)) + (v1080_data * v1116_data)) + (v1160_data * v1196_data);
          float v1201_data = s0[142];
          float v1204_p = ((((((((((((((v40_data * v81_data) + (v120_data * v161_data)) + (v200_data * v241_data)) + (v280_data * v321_data)) + (v360_data * v401_data)) + (v440_data * v481_data)) + (v520_data * v561_data)) + (v600_data * v641_data)) + (v680_data * v721_data)) + (v760_data * v801_data)) + (v840_data * v881_data)) + (v920_data * v961_data)) + (v1000_data * v1041_data)) + (v1080_data * v1121_data)) + (v1160_data * v1201_data);
          float v1206_data = s0[158];
          float v1209_p = ((((((((((((((v40_data * v86_data) + (v120_data * v166_data)) + (v200_data * v246_data)) + (v280_data * v326_data)) + (v360_data * v406_data)) + (v440_data * v486_data)) + (v520_data * v566_data)) + (v600_data * v646_data)) + (v680_data * v726_data)) + (v760_data * v806_data)) + (v840_data * v886_data)) + (v920_data * v966_data)) + (v1000_data * v1046_data)) + (v1080_data * v1126_data)) + (v1160_data * v1206_data);
          float v1211_data = s0[174];
          float v1214_p = ((((((((((((((v40_data * v91_data) + (v120_data * v171_data)) + (v200_data * v251_data)) + (v280_data * v331_data)) + (v360_data * v411_data)) + (v440_data * v491_data)) + (v520_data * v571_data)) + (v600_data * v651_data)) + (v680_data * v731_data)) + (v760_data * v811_data)) + (v840_data * v891_data)) + (v920_data * v971_data)) + (v1000_data * v1051_data)) + (v1080_data * v1131_data)) + (v1160_data * v1211_data);
          float v1216_data = s0[190];
          float v1219_p = ((((((((((((((v40_data * v96_data) + (v120_data * v176_data)) + (v200_data * v256_data)) + (v280_data * v336_data)) + (v360_data * v416_data)) + (v440_data * v496_data)) + (v520_data * v576_data)) + (v600_data * v656_data)) + (v680_data * v736_data)) + (v760_data * v816_data)) + (v840_data * v896_data)) + (v920_data * v976_data)) + (v1000_data * v1056_data)) + (v1080_data * v1136_data)) + (v1160_data * v1216_data);
          float v1221_data = s0[206];
          float v1224_p = ((((((((((((((v40_data * v101_data) + (v120_data * v181_data)) + (v200_data * v261_data)) + (v280_data * v341_data)) + (v360_data * v421_data)) + (v440_data * v501_data)) + (v520_data * v581_data)) + (v600_data * v661_data)) + (v680_data * v741_data)) + (v760_data * v821_data)) + (v840_data * v901_data)) + (v920_data * v981_data)) + (v1000_data * v1061_data)) + (v1080_data * v1141_data)) + (v1160_data * v1221_data);
          float v1226_data = s0[222];
          float v1229_p = ((((((((((((((v40_data * v106_data) + (v120_data * v186_data)) + (v200_data * v266_data)) + (v280_data * v346_data)) + (v360_data * v426_data)) + (v440_data * v506_data)) + (v520_data * v586_data)) + (v600_data * v666_data)) + (v680_data * v746_data)) + (v760_data * v826_data)) + (v840_data * v906_data)) + (v920_data * v986_data)) + (v1000_data * v1066_data)) + (v1080_data * v1146_data)) + (v1160_data * v1226_data);
          float v1231_data = s0[238];
          float v1234_p = ((((((((((((((v40_data * v111_data) + (v120_data * v191_data)) + (v200_data * v271_data)) + (v280_data * v351_data)) + (v360_data * v431_data)) + (v440_data * v511_data)) + (v520_data * v591_data)) + (v600_data * v671_data)) + (v680_data * v751_data)) + (v760_data * v831_data)) + (v840_data * v911_data)) + (v920_data * v991_data)) + (v1000_data * v1071_data)) + (v1080_data * v1151_data)) + (v1160_data * v1231_data);
          float v1236_data = s0[254];
          float v1239_p = ((((((((((((((v40_data * v116_data) + (v120_data * v196_data)) + (v200_data * v276_data)) + (v280_data * v356_data)) + (v360_data * v436_data)) + (v440_data * v516_data)) + (v520_data * v596_data)) + (v600_data * v676_data)) + (v680_data * v756_data)) + (v760_data * v836_data)) + (v840_data * v916_data)) + (v920_data * v996_data)) + (v1000_data * v1076_data)) + (v1080_data * v1156_data)) + (v1160_data * v1236_data);
          float v1240_data = r0[15];
          // r0 = load{g>r}(wrap_glb_m1);
          #pragma unroll
          for (int32_t v1355_i1 = 0; v1355_i1 < 16; ++v1355_i1) {
            float v1358_data = v1342_g ? (__ldcg(&wrap_glb_m1[(v1341_lead + (v1355_i1 * 16))])) : (0.0f);
            r0[v1355_i1] = v1358_data;
          }
          float v1241_data = s0[15];
          ir1[0] = (v1164_p + (v1240_data * v1241_data));
          float v1246_data = s0[31];
          ir1[1] = (v1169_p + (v1240_data * v1246_data));
          float v1251_data = s0[47];
          ir1[2] = (v1174_p + (v1240_data * v1251_data));
          float v1256_data = s0[63];
          ir1[3] = (v1179_p + (v1240_data * v1256_data));
          float v1261_data = s0[79];
          ir1[4] = (v1184_p + (v1240_data * v1261_data));
          float v1266_data = s0[95];
          ir1[5] = (v1189_p + (v1240_data * v1266_data));
          float v1271_data = s0[111];
          ir1[6] = (v1194_p + (v1240_data * v1271_data));
          float v1276_data = s0[127];
          ir1[7] = (v1199_p + (v1240_data * v1276_data));
          float v1281_data = s0[143];
          ir1[8] = (v1204_p + (v1240_data * v1281_data));
          float v1286_data = s0[159];
          ir1[9] = (v1209_p + (v1240_data * v1286_data));
          float v1291_data = s0[175];
          ir1[10] = (v1214_p + (v1240_data * v1291_data));
          float v1296_data = s0[191];
          ir1[11] = (v1219_p + (v1240_data * v1296_data));
          float v1301_data = s0[207];
          ir1[12] = (v1224_p + (v1240_data * v1301_data));
          float v1306_data = s0[223];
          ir1[13] = (v1229_p + (v1240_data * v1306_data));
          float v1311_data = s0[239];
          ir1[14] = (v1234_p + (v1240_data * v1311_data));
          float v1316_data = s0[255];
          ir1[15] = (v1239_p + (v1240_data * v1316_data));
          // r1 = ir1
          // glb_m0 = store{r>g}(r1);
          if (v1342_g) {
            #pragma unroll
            for (int32_t v1325_i1 = 0; v1325_i1 < 16; ++v1325_i1) {
              float v1327_data = ir1[v1325_i1];
              glb_m0[(v1341_lead + (v1325_i1 * 16))] = v1327_data;
            }
          }
          __syncwarp();
        }
        else {
          __pipeline_wait_prior(0);
          // r0 = load{g>r}(wrap_glb_m1);
          #pragma unroll
          for (int32_t v1347_i1 = 0; v1347_i1 < 16; ++v1347_i1) {
            float v1350_data = v1342_g ? (__ldcg(&wrap_glb_m1[(v1341_lead + (v1347_i1 * 16))])) : (0.0f);
            r0[v1347_i1] = v1350_data;
          }
        }
        // s0 = load{g>s}(wrap_glb_m2[0, 1])
        #pragma unroll
        for (int32_t v1365_i = 0; v1365_i < 8; ++v1365_i) {
          int32_t v1367_a = threadIdx.x + (v1365_i * 32);
          __pipeline_memcpy_async(&s0[v1367_a], &wrap_glb_m2[v1367_a], 4);
        }
        __pipeline_commit();
        v1337_flagWord = v1338_flagWordNext;
        v1338_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[256 * threadIdx.y + 0];
      size_t v1383_batchId1 = (((threadIdx.y + blockDim.y * (blockIdx.x)) + numElements0) % (gridDim.x * blockDim.y) < numElements1) ? ((threadIdx.y + blockDim.y * (blockIdx.x)) + numElements0) % (gridDim.x * blockDim.y) : 0u;
      __syncthreads();
      float * __restrict__ s0 = &localShrMem0[0];
      float r0[16]{};
      const uint32_t flagWordFirst = flags1 == nullptr ? 1 : flags1[v1383_batchId1];
      size_t v2713_flagSecondAhead = v1383_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags1 == nullptr ? 1 : flags1[((v2713_flagSecondAhead < numElements1) ? v2713_flagSecondAhead : v1383_batchId1)];
      const float *const __restrict__ peel_glb_m4 = &m4[v1383_batchId1 * 256 + 0 + m4_extraOffset];
      int32_t v2721_lead = threadIdx.x % 32;
      bool v2722_g = v2721_lead < 16;
      // r0 = load{g>r}(peel_glb_m4);
      #pragma unroll
      for (int32_t v2731_i1 = 0; v2731_i1 < 16; ++v2731_i1) {
        float v2734_data = v2722_g ? (__ldcg(&peel_glb_m4[(v2721_lead + (v2731_i1 * 16))])) : (0.0f);
        r0[v2731_i1] = v2734_data;
      }
      const float *const __restrict__ peel_glb_m5 = &m5[v1383_batchId1 * 256 + 0 + m5_extraOffset];
      // s0 = load{g>s}(peel_glb_m5[0, 1])
      #pragma unroll
      for (int32_t v2741_i = 0; v2741_i < 8; ++v2741_i) {
        int32_t v2743_a = threadIdx.x + (v2741_i * 32);
        __pipeline_memcpy_async(&s0[v2743_a], &peel_glb_m5[v2743_a], 4);
      }
      __pipeline_commit();
      uint32_t v2717_flagWord = flagWordFirst;
      uint32_t v2718_flagWordNext = flagWordSecond;
      for (size_t v1388_batchId0 = ((threadIdx.y + blockDim.y * (blockIdx.x)) + numElements0) % (gridDim.x * blockDim.y); v1388_batchId0 < numElements1; v1388_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v1389_ahead1 = v1388_batchId0 + (gridDim.x * blockDim.y);
        size_t v1391_batchId1 = (v1389_ahead1 < numElements1) ? v1389_ahead1 : v1388_batchId0;
        const float *const __restrict__ wrap_glb_m4 = &m4[v1391_batchId1 * 256 + 0 + m4_extraOffset];
        const float *const __restrict__ wrap_glb_m5 = &m5[v1391_batchId1 * 256 + 0 + m5_extraOffset];
        size_t v2753_flagAheadAhead = v1391_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags1 == nullptr ? 1 : flags1[((v2753_flagAheadAhead < numElements1) ? v2753_flagAheadAhead : v1391_batchId1)];
        const bool allowed = static_cast<bool>(v2717_flagWord);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v1388_batchId0 * 256 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v1388_batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v1388_batchId0 * 256 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[v1388_batchId0 * 256 + 0 + m3_extraOffset];
          // wait(s0 = load{g>s}(glb_m5[0, 1]));
          __pipeline_wait_prior(0);
          float r1[16]{};
          // ir1 = +(r0 * s0)
          // [(0, 16), (0, 16)] [(0, 16)]
          float ir1[16]{};
          float v1420_data = r0[0];
          __syncwarp();
          float v1421_data = s0[0];
          float v1426_data = s0[16];
          float v1431_data = s0[32];
          float v1436_data = s0[48];
          float v1441_data = s0[64];
          float v1446_data = s0[80];
          float v1451_data = s0[96];
          float v1456_data = s0[112];
          float v1461_data = s0[128];
          float v1466_data = s0[144];
          float v1471_data = s0[160];
          float v1476_data = s0[176];
          float v1481_data = s0[192];
          float v1486_data = s0[208];
          float v1491_data = s0[224];
          float v1496_data = s0[240];
          float v1500_data = r0[1];
          float v1501_data = s0[1];
          float v1506_data = s0[17];
          float v1511_data = s0[33];
          float v1516_data = s0[49];
          float v1521_data = s0[65];
          float v1526_data = s0[81];
          float v1531_data = s0[97];
          float v1536_data = s0[113];
          float v1541_data = s0[129];
          float v1546_data = s0[145];
          float v1551_data = s0[161];
          float v1556_data = s0[177];
          float v1561_data = s0[193];
          float v1566_data = s0[209];
          float v1571_data = s0[225];
          float v1576_data = s0[241];
          float v1580_data = r0[2];
          float v1581_data = s0[2];
          float v1586_data = s0[18];
          float v1591_data = s0[34];
          float v1596_data = s0[50];
          float v1601_data = s0[66];
          float v1606_data = s0[82];
          float v1611_data = s0[98];
          float v1616_data = s0[114];
          float v1621_data = s0[130];
          float v1626_data = s0[146];
          float v1631_data = s0[162];
          float v1636_data = s0[178];
          float v1641_data = s0[194];
          float v1646_data = s0[210];
          float v1651_data = s0[226];
          float v1656_data = s0[242];
          float v1660_data = r0[3];
          float v1661_data = s0[3];
          float v1666_data = s0[19];
          float v1671_data = s0[35];
          float v1676_data = s0[51];
          float v1681_data = s0[67];
          float v1686_data = s0[83];
          float v1691_data = s0[99];
          float v1696_data = s0[115];
          float v1701_data = s0[131];
          float v1706_data = s0[147];
          float v1711_data = s0[163];
          float v1716_data = s0[179];
          float v1721_data = s0[195];
          float v1726_data = s0[211];
          float v1731_data = s0[227];
          float v1736_data = s0[243];
          float v1740_data = r0[4];
          float v1741_data = s0[4];
          float v1746_data = s0[20];
          float v1751_data = s0[36];
          float v1756_data = s0[52];
          float v1761_data = s0[68];
          float v1766_data = s0[84];
          float v1771_data = s0[100];
          float v1776_data = s0[116];
          float v1781_data = s0[132];
          float v1786_data = s0[148];
          float v1791_data = s0[164];
          float v1796_data = s0[180];
          float v1801_data = s0[196];
          float v1806_data = s0[212];
          float v1811_data = s0[228];
          float v1816_data = s0[244];
          float v1820_data = r0[5];
          float v1821_data = s0[5];
          float v1826_data = s0[21];
          float v1831_data = s0[37];
          float v1836_data = s0[53];
          float v1841_data = s0[69];
          float v1846_data = s0[85];
          float v1851_data = s0[101];
          float v1856_data = s0[117];
          float v1861_data = s0[133];
          float v1866_data = s0[149];
          float v1871_data = s0[165];
          float v1876_data = s0[181];
          float v1881_data = s0[197];
          float v1886_data = s0[213];
          float v1891_data = s0[229];
          float v1896_data = s0[245];
          float v1900_data = r0[6];
          float v1901_data = s0[6];
          float v1906_data = s0[22];
          float v1911_data = s0[38];
          float v1916_data = s0[54];
          float v1921_data = s0[70];
          float v1926_data = s0[86];
          float v1931_data = s0[102];
          float v1936_data = s0[118];
          float v1941_data = s0[134];
          float v1946_data = s0[150];
          float v1951_data = s0[166];
          float v1956_data = s0[182];
          float v1961_data = s0[198];
          float v1966_data = s0[214];
          float v1971_data = s0[230];
          float v1976_data = s0[246];
          float v1980_data = r0[7];
          float v1981_data = s0[7];
          float v1986_data = s0[23];
          float v1991_data = s0[39];
          float v1996_data = s0[55];
          float v2001_data = s0[71];
          float v2006_data = s0[87];
          float v2011_data = s0[103];
          float v2016_data = s0[119];
          float v2021_data = s0[135];
          float v2026_data = s0[151];
          float v2031_data = s0[167];
          float v2036_data = s0[183];
          float v2041_data = s0[199];
          float v2046_data = s0[215];
          float v2051_data = s0[231];
          float v2056_data = s0[247];
          float v2060_data = r0[8];
          float v2061_data = s0[8];
          float v2066_data = s0[24];
          float v2071_data = s0[40];
          float v2076_data = s0[56];
          float v2081_data = s0[72];
          float v2086_data = s0[88];
          float v2091_data = s0[104];
          float v2096_data = s0[120];
          float v2101_data = s0[136];
          float v2106_data = s0[152];
          float v2111_data = s0[168];
          float v2116_data = s0[184];
          float v2121_data = s0[200];
          float v2126_data = s0[216];
          float v2131_data = s0[232];
          float v2136_data = s0[248];
          float v2140_data = r0[9];
          float v2141_data = s0[9];
          float v2146_data = s0[25];
          float v2151_data = s0[41];
          float v2156_data = s0[57];
          float v2161_data = s0[73];
          float v2166_data = s0[89];
          float v2171_data = s0[105];
          float v2176_data = s0[121];
          float v2181_data = s0[137];
          float v2186_data = s0[153];
          float v2191_data = s0[169];
          float v2196_data = s0[185];
          float v2201_data = s0[201];
          float v2206_data = s0[217];
          float v2211_data = s0[233];
          float v2216_data = s0[249];
          float v2220_data = r0[10];
          float v2221_data = s0[10];
          float v2226_data = s0[26];
          float v2231_data = s0[42];
          float v2236_data = s0[58];
          float v2241_data = s0[74];
          float v2246_data = s0[90];
          float v2251_data = s0[106];
          float v2256_data = s0[122];
          float v2261_data = s0[138];
          float v2266_data = s0[154];
          float v2271_data = s0[170];
          float v2276_data = s0[186];
          float v2281_data = s0[202];
          float v2286_data = s0[218];
          float v2291_data = s0[234];
          float v2296_data = s0[250];
          float v2300_data = r0[11];
          float v2301_data = s0[11];
          float v2306_data = s0[27];
          float v2311_data = s0[43];
          float v2316_data = s0[59];
          float v2321_data = s0[75];
          float v2326_data = s0[91];
          float v2331_data = s0[107];
          float v2336_data = s0[123];
          float v2341_data = s0[139];
          float v2346_data = s0[155];
          float v2351_data = s0[171];
          float v2356_data = s0[187];
          float v2361_data = s0[203];
          float v2366_data = s0[219];
          float v2371_data = s0[235];
          float v2376_data = s0[251];
          float v2380_data = r0[12];
          float v2381_data = s0[12];
          float v2386_data = s0[28];
          float v2391_data = s0[44];
          float v2396_data = s0[60];
          float v2401_data = s0[76];
          float v2406_data = s0[92];
          float v2411_data = s0[108];
          float v2416_data = s0[124];
          float v2421_data = s0[140];
          float v2426_data = s0[156];
          float v2431_data = s0[172];
          float v2436_data = s0[188];
          float v2441_data = s0[204];
          float v2446_data = s0[220];
          float v2451_data = s0[236];
          float v2456_data = s0[252];
          float v2460_data = r0[13];
          float v2461_data = s0[13];
          float v2466_data = s0[29];
          float v2471_data = s0[45];
          float v2476_data = s0[61];
          float v2481_data = s0[77];
          float v2486_data = s0[93];
          float v2491_data = s0[109];
          float v2496_data = s0[125];
          float v2501_data = s0[141];
          float v2506_data = s0[157];
          float v2511_data = s0[173];
          float v2516_data = s0[189];
          float v2521_data = s0[205];
          float v2526_data = s0[221];
          float v2531_data = s0[237];
          float v2536_data = s0[253];
          float v2540_data = r0[14];
          float v2541_data = s0[14];
          float v2544_p = ((((((((((((((v1420_data * v1421_data) + (v1500_data * v1501_data)) + (v1580_data * v1581_data)) + (v1660_data * v1661_data)) + (v1740_data * v1741_data)) + (v1820_data * v1821_data)) + (v1900_data * v1901_data)) + (v1980_data * v1981_data)) + (v2060_data * v2061_data)) + (v2140_data * v2141_data)) + (v2220_data * v2221_data)) + (v2300_data * v2301_data)) + (v2380_data * v2381_data)) + (v2460_data * v2461_data)) + (v2540_data * v2541_data);
          float v2546_data = s0[30];
          float v2549_p = ((((((((((((((v1420_data * v1426_data) + (v1500_data * v1506_data)) + (v1580_data * v1586_data)) + (v1660_data * v1666_data)) + (v1740_data * v1746_data)) + (v1820_data * v1826_data)) + (v1900_data * v1906_data)) + (v1980_data * v1986_data)) + (v2060_data * v2066_data)) + (v2140_data * v2146_data)) + (v2220_data * v2226_data)) + (v2300_data * v2306_data)) + (v2380_data * v2386_data)) + (v2460_data * v2466_data)) + (v2540_data * v2546_data);
          float v2551_data = s0[46];
          float v2554_p = ((((((((((((((v1420_data * v1431_data) + (v1500_data * v1511_data)) + (v1580_data * v1591_data)) + (v1660_data * v1671_data)) + (v1740_data * v1751_data)) + (v1820_data * v1831_data)) + (v1900_data * v1911_data)) + (v1980_data * v1991_data)) + (v2060_data * v2071_data)) + (v2140_data * v2151_data)) + (v2220_data * v2231_data)) + (v2300_data * v2311_data)) + (v2380_data * v2391_data)) + (v2460_data * v2471_data)) + (v2540_data * v2551_data);
          float v2556_data = s0[62];
          float v2559_p = ((((((((((((((v1420_data * v1436_data) + (v1500_data * v1516_data)) + (v1580_data * v1596_data)) + (v1660_data * v1676_data)) + (v1740_data * v1756_data)) + (v1820_data * v1836_data)) + (v1900_data * v1916_data)) + (v1980_data * v1996_data)) + (v2060_data * v2076_data)) + (v2140_data * v2156_data)) + (v2220_data * v2236_data)) + (v2300_data * v2316_data)) + (v2380_data * v2396_data)) + (v2460_data * v2476_data)) + (v2540_data * v2556_data);
          float v2561_data = s0[78];
          float v2564_p = ((((((((((((((v1420_data * v1441_data) + (v1500_data * v1521_data)) + (v1580_data * v1601_data)) + (v1660_data * v1681_data)) + (v1740_data * v1761_data)) + (v1820_data * v1841_data)) + (v1900_data * v1921_data)) + (v1980_data * v2001_data)) + (v2060_data * v2081_data)) + (v2140_data * v2161_data)) + (v2220_data * v2241_data)) + (v2300_data * v2321_data)) + (v2380_data * v2401_data)) + (v2460_data * v2481_data)) + (v2540_data * v2561_data);
          float v2566_data = s0[94];
          float v2569_p = ((((((((((((((v1420_data * v1446_data) + (v1500_data * v1526_data)) + (v1580_data * v1606_data)) + (v1660_data * v1686_data)) + (v1740_data * v1766_data)) + (v1820_data * v1846_data)) + (v1900_data * v1926_data)) + (v1980_data * v2006_data)) + (v2060_data * v2086_data)) + (v2140_data * v2166_data)) + (v2220_data * v2246_data)) + (v2300_data * v2326_data)) + (v2380_data * v2406_data)) + (v2460_data * v2486_data)) + (v2540_data * v2566_data);
          float v2571_data = s0[110];
          float v2574_p = ((((((((((((((v1420_data * v1451_data) + (v1500_data * v1531_data)) + (v1580_data * v1611_data)) + (v1660_data * v1691_data)) + (v1740_data * v1771_data)) + (v1820_data * v1851_data)) + (v1900_data * v1931_data)) + (v1980_data * v2011_data)) + (v2060_data * v2091_data)) + (v2140_data * v2171_data)) + (v2220_data * v2251_data)) + (v2300_data * v2331_data)) + (v2380_data * v2411_data)) + (v2460_data * v2491_data)) + (v2540_data * v2571_data);
          float v2576_data = s0[126];
          float v2579_p = ((((((((((((((v1420_data * v1456_data) + (v1500_data * v1536_data)) + (v1580_data * v1616_data)) + (v1660_data * v1696_data)) + (v1740_data * v1776_data)) + (v1820_data * v1856_data)) + (v1900_data * v1936_data)) + (v1980_data * v2016_data)) + (v2060_data * v2096_data)) + (v2140_data * v2176_data)) + (v2220_data * v2256_data)) + (v2300_data * v2336_data)) + (v2380_data * v2416_data)) + (v2460_data * v2496_data)) + (v2540_data * v2576_data);
          float v2581_data = s0[142];
          float v2584_p = ((((((((((((((v1420_data * v1461_data) + (v1500_data * v1541_data)) + (v1580_data * v1621_data)) + (v1660_data * v1701_data)) + (v1740_data * v1781_data)) + (v1820_data * v1861_data)) + (v1900_data * v1941_data)) + (v1980_data * v2021_data)) + (v2060_data * v2101_data)) + (v2140_data * v2181_data)) + (v2220_data * v2261_data)) + (v2300_data * v2341_data)) + (v2380_data * v2421_data)) + (v2460_data * v2501_data)) + (v2540_data * v2581_data);
          float v2586_data = s0[158];
          float v2589_p = ((((((((((((((v1420_data * v1466_data) + (v1500_data * v1546_data)) + (v1580_data * v1626_data)) + (v1660_data * v1706_data)) + (v1740_data * v1786_data)) + (v1820_data * v1866_data)) + (v1900_data * v1946_data)) + (v1980_data * v2026_data)) + (v2060_data * v2106_data)) + (v2140_data * v2186_data)) + (v2220_data * v2266_data)) + (v2300_data * v2346_data)) + (v2380_data * v2426_data)) + (v2460_data * v2506_data)) + (v2540_data * v2586_data);
          float v2591_data = s0[174];
          float v2594_p = ((((((((((((((v1420_data * v1471_data) + (v1500_data * v1551_data)) + (v1580_data * v1631_data)) + (v1660_data * v1711_data)) + (v1740_data * v1791_data)) + (v1820_data * v1871_data)) + (v1900_data * v1951_data)) + (v1980_data * v2031_data)) + (v2060_data * v2111_data)) + (v2140_data * v2191_data)) + (v2220_data * v2271_data)) + (v2300_data * v2351_data)) + (v2380_data * v2431_data)) + (v2460_data * v2511_data)) + (v2540_data * v2591_data);
          float v2596_data = s0[190];
          float v2599_p = ((((((((((((((v1420_data * v1476_data) + (v1500_data * v1556_data)) + (v1580_data * v1636_data)) + (v1660_data * v1716_data)) + (v1740_data * v1796_data)) + (v1820_data * v1876_data)) + (v1900_data * v1956_data)) + (v1980_data * v2036_data)) + (v2060_data * v2116_data)) + (v2140_data * v2196_data)) + (v2220_data * v2276_data)) + (v2300_data * v2356_data)) + (v2380_data * v2436_data)) + (v2460_data * v2516_data)) + (v2540_data * v2596_data);
          float v2601_data = s0[206];
          float v2604_p = ((((((((((((((v1420_data * v1481_data) + (v1500_data * v1561_data)) + (v1580_data * v1641_data)) + (v1660_data * v1721_data)) + (v1740_data * v1801_data)) + (v1820_data * v1881_data)) + (v1900_data * v1961_data)) + (v1980_data * v2041_data)) + (v2060_data * v2121_data)) + (v2140_data * v2201_data)) + (v2220_data * v2281_data)) + (v2300_data * v2361_data)) + (v2380_data * v2441_data)) + (v2460_data * v2521_data)) + (v2540_data * v2601_data);
          float v2606_data = s0[222];
          float v2609_p = ((((((((((((((v1420_data * v1486_data) + (v1500_data * v1566_data)) + (v1580_data * v1646_data)) + (v1660_data * v1726_data)) + (v1740_data * v1806_data)) + (v1820_data * v1886_data)) + (v1900_data * v1966_data)) + (v1980_data * v2046_data)) + (v2060_data * v2126_data)) + (v2140_data * v2206_data)) + (v2220_data * v2286_data)) + (v2300_data * v2366_data)) + (v2380_data * v2446_data)) + (v2460_data * v2526_data)) + (v2540_data * v2606_data);
          float v2611_data = s0[238];
          float v2614_p = ((((((((((((((v1420_data * v1491_data) + (v1500_data * v1571_data)) + (v1580_data * v1651_data)) + (v1660_data * v1731_data)) + (v1740_data * v1811_data)) + (v1820_data * v1891_data)) + (v1900_data * v1971_data)) + (v1980_data * v2051_data)) + (v2060_data * v2131_data)) + (v2140_data * v2211_data)) + (v2220_data * v2291_data)) + (v2300_data * v2371_data)) + (v2380_data * v2451_data)) + (v2460_data * v2531_data)) + (v2540_data * v2611_data);
          float v2616_data = s0[254];
          float v2619_p = ((((((((((((((v1420_data * v1496_data) + (v1500_data * v1576_data)) + (v1580_data * v1656_data)) + (v1660_data * v1736_data)) + (v1740_data * v1816_data)) + (v1820_data * v1896_data)) + (v1900_data * v1976_data)) + (v1980_data * v2056_data)) + (v2060_data * v2136_data)) + (v2140_data * v2216_data)) + (v2220_data * v2296_data)) + (v2300_data * v2376_data)) + (v2380_data * v2456_data)) + (v2460_data * v2536_data)) + (v2540_data * v2616_data);
          float v2620_data = r0[15];
          // r0 = load{g>r}(wrap_glb_m4);
          #pragma unroll
          for (int32_t v2735_i1 = 0; v2735_i1 < 16; ++v2735_i1) {
            float v2738_data = v2722_g ? (__ldcg(&wrap_glb_m4[(v2721_lead + (v2735_i1 * 16))])) : (0.0f);
            r0[v2735_i1] = v2738_data;
          }
          float v2621_data = s0[15];
          ir1[0] = (v2544_p + (v2620_data * v2621_data));
          float v2626_data = s0[31];
          ir1[1] = (v2549_p + (v2620_data * v2626_data));
          float v2631_data = s0[47];
          ir1[2] = (v2554_p + (v2620_data * v2631_data));
          float v2636_data = s0[63];
          ir1[3] = (v2559_p + (v2620_data * v2636_data));
          float v2641_data = s0[79];
          ir1[4] = (v2564_p + (v2620_data * v2641_data));
          float v2646_data = s0[95];
          ir1[5] = (v2569_p + (v2620_data * v2646_data));
          float v2651_data = s0[111];
          ir1[6] = (v2574_p + (v2620_data * v2651_data));
          float v2656_data = s0[127];
          ir1[7] = (v2579_p + (v2620_data * v2656_data));
          float v2661_data = s0[143];
          ir1[8] = (v2584_p + (v2620_data * v2661_data));
          float v2666_data = s0[159];
          ir1[9] = (v2589_p + (v2620_data * v2666_data));
          float v2671_data = s0[175];
          ir1[10] = (v2594_p + (v2620_data * v2671_data));
          float v2676_data = s0[191];
          ir1[11] = (v2599_p + (v2620_data * v2676_data));
          float v2681_data = s0[207];
          ir1[12] = (v2604_p + (v2620_data * v2681_data));
          float v2686_data = s0[223];
          ir1[13] = (v2609_p + (v2620_data * v2686_data));
          float v2691_data = s0[239];
          ir1[14] = (v2614_p + (v2620_data * v2691_data));
          float v2696_data = s0[255];
          ir1[15] = (v2619_p + (v2620_data * v2696_data));
          // r1 = ir1
          // glb_m3 = store{r>g}(r1);
          if (v2722_g) {
            #pragma unroll
            for (int32_t v2705_i1 = 0; v2705_i1 < 16; ++v2705_i1) {
              float v2707_data = ir1[v2705_i1];
              glb_m3[(v2721_lead + (v2705_i1 * 16))] = v2707_data;
            }
          }
          __syncwarp();
        }
        else {
          __pipeline_wait_prior(0);
          // r0 = load{g>r}(wrap_glb_m4);
          #pragma unroll
          for (int32_t v2727_i1 = 0; v2727_i1 < 16; ++v2727_i1) {
            float v2730_data = v2722_g ? (__ldcg(&wrap_glb_m4[(v2721_lead + (v2727_i1 * 16))])) : (0.0f);
            r0[v2727_i1] = v2730_data;
          }
        }
        // s0 = load{g>s}(wrap_glb_m5[0, 1])
        #pragma unroll
        for (int32_t v2745_i = 0; v2745_i < 8; ++v2745_i) {
          int32_t v2747_a = threadIdx.x + (v2745_i * 32);
          __pipeline_memcpy_async(&s0[v2747_a], &wrap_glb_m5[v2747_a], 4);
        }
        __pipeline_commit();
        v2717_flagWord = v2718_flagWordNext;
        v2718_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
  }
}

