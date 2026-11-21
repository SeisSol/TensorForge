// === base name ===
kernel_166f559bf2f46e88

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_166f559bf2f46e88 = {{32, 4, 1}, 32, 32, 1, 4, 4096, true, true, 2};
tensorforge::LaunchConfig launch_config_kernel_166f559bf2f46e88(size_t numElements0, size_t numElements1, void* streamPtr = nullptr);
void launcher_kernel_166f559bf2f46e88(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, size_t numElements1, unsigned * flags0 = nullptr, unsigned * flags1 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_166f559bf2f46e88(size_t numElements0, size_t numElements1, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_166f559bf2f46e88, block.x * block.y * block.z, 1024 * sizeof(float));
        CHECK_ERR;
        if (blocksPerSM > 0) {
          gridsize = smCount * blocksPerSM;
        }
        else {
          gridsize = smCount;
        }
      }
      
  tensorforge::LaunchConfig config{};
  config.grid[0] = gridsize;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 32;
  config.block[1] = 4;
  config.block[2] = 1;
  config.sharedMemBytes = 1024 * sizeof(float);
  config.cooperative = true;
  return config;
}
void launcher_kernel_166f559bf2f46e88(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, size_t numElements1, unsigned * flags0, unsigned * flags1, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_166f559bf2f46e88(numElements0, numElements1, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_166f559bf2f46e88, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  const unsigned * flags0Arg = (const unsigned *)flags0;
  const unsigned * flags1Arg = (const unsigned *)flags1;
  
    auto args = tensorforge::argsPtrs(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, numElements1, flags0Arg, flags1Arg);
    cudaLaunchCooperativeKernel(kernel_kernel_166f559bf2f46e88, grid, block, args.data(), config.sharedMemBytes, stream);
  ;
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_166f559bf2f46e88(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, size_t numElements1, const unsigned * __restrict__ flags0, const unsigned * __restrict__ flags1) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: cooperative_grid=1
    // launch: 32 lanes x 4 per block = block 32x4x1, 4096 B shared, occupancy grid, cooperative
    // operands:
    //   m0 16×16(16×16) {0..16}×{0..16} strided
    //   m1 16×16(16×16) {0..16}×{0..16} strided
    //   m2 16×16(16×16) {0..16}×{0..16} strided
    //   m3 16×16(16×16) {0..16}×{0..16} strided
    //   m4 16×16(16×16) {0..16}×{0..16} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    //   barrier
    //   m3[i,j] = m0[i,k] × m4[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,4,1],"cooperative":true,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":true,"mults_per_block":4,"shared_elements":1024},{"barrier":false,"mults_per_block":4,"shared_elements":1024}],"shared_bytes":4096,"shared_elements":1024,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"E","bbox":[[0,0],[16,16]],"name":"m3","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m4","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"kind":"barrier"},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[256 * threadIdx.y + 0];
      size_t v3_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < numElements0) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      float * __restrict__ s0 = &localShrMem0[0];
      float r0[16]{};
      const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
      size_t v1332_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v1332_flagSecondAhead < numElements0) ? v1332_flagSecondAhead : v3_batchId1)];
      const float *const __restrict__ peel_glb_m1 = &m1[v3_batchId1 * 256 + 0 + m1_extraOffset];
      int32_t v1340_lead = threadIdx.x % 32;
      bool v1341_g = v1340_lead < 16;
      // r0 = load{g>r}(peel_glb_m1);
      #pragma unroll
      for (int32_t v1350_i1 = 0; v1350_i1 < 16; ++v1350_i1) {
        float v1353_data = v1341_g ? (__ldcg(&peel_glb_m1[(v1340_lead + (v1350_i1 * 16))])) : (0.0f);
        r0[v1350_i1] = v1353_data;
      }
      const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 256 + 0 + m2_extraOffset];
      // s0 = load{g>s}(peel_glb_m2[0, 1])
      #pragma unroll
      for (int32_t v1360_i = 0; v1360_i < 8; ++v1360_i) {
        int32_t v1362_a = threadIdx.x + (v1360_i * 32);
        __pipeline_memcpy_async(&s0[v1362_a], &peel_glb_m2[v1362_a], 4);
      }
      __pipeline_commit();
      uint32_t v1336_flagWord = flagWordFirst;
      uint32_t v1337_flagWordNext = flagWordSecond;
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const float *const __restrict__ wrap_glb_m1 = &m1[v11_batchId1 * 256 + 0 + m1_extraOffset];
        const float *const __restrict__ wrap_glb_m2 = &m2[v11_batchId1 * 256 + 0 + m2_extraOffset];
        size_t v1372_flagAheadAhead = v11_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v1372_flagAheadAhead < numElements0) ? v1372_flagAheadAhead : v11_batchId1)];
        const bool allowed = static_cast<bool>(v1336_flagWord);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 256 + 0 + m0_extraOffset];
          float *const __restrict__ glb_m3 = &m3[v8_batchId0 * 256 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[v8_batchId0 * 256 + 0 + m4_extraOffset];
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[16]{};
          // ir1 = +(r0 * s0)
          // [(0, 16), (0, 16)] [(0, 16)]
          float ir1[16]{};
          float v39_data = r0[0];
          __syncwarp();
          float v40_data = s0[0];
          float v45_data = s0[16];
          float v50_data = s0[32];
          float v55_data = s0[48];
          float v60_data = s0[64];
          float v65_data = s0[80];
          float v70_data = s0[96];
          float v75_data = s0[112];
          float v80_data = s0[128];
          float v85_data = s0[144];
          float v90_data = s0[160];
          float v95_data = s0[176];
          float v100_data = s0[192];
          float v105_data = s0[208];
          float v110_data = s0[224];
          float v115_data = s0[240];
          float v119_data = r0[1];
          float v120_data = s0[1];
          float v125_data = s0[17];
          float v130_data = s0[33];
          float v135_data = s0[49];
          float v140_data = s0[65];
          float v145_data = s0[81];
          float v150_data = s0[97];
          float v155_data = s0[113];
          float v160_data = s0[129];
          float v165_data = s0[145];
          float v170_data = s0[161];
          float v175_data = s0[177];
          float v180_data = s0[193];
          float v185_data = s0[209];
          float v190_data = s0[225];
          float v195_data = s0[241];
          float v199_data = r0[2];
          float v200_data = s0[2];
          float v205_data = s0[18];
          float v210_data = s0[34];
          float v215_data = s0[50];
          float v220_data = s0[66];
          float v225_data = s0[82];
          float v230_data = s0[98];
          float v235_data = s0[114];
          float v240_data = s0[130];
          float v245_data = s0[146];
          float v250_data = s0[162];
          float v255_data = s0[178];
          float v260_data = s0[194];
          float v265_data = s0[210];
          float v270_data = s0[226];
          float v275_data = s0[242];
          float v279_data = r0[3];
          float v280_data = s0[3];
          float v285_data = s0[19];
          float v290_data = s0[35];
          float v295_data = s0[51];
          float v300_data = s0[67];
          float v305_data = s0[83];
          float v310_data = s0[99];
          float v315_data = s0[115];
          float v320_data = s0[131];
          float v325_data = s0[147];
          float v330_data = s0[163];
          float v335_data = s0[179];
          float v340_data = s0[195];
          float v345_data = s0[211];
          float v350_data = s0[227];
          float v355_data = s0[243];
          float v359_data = r0[4];
          float v360_data = s0[4];
          float v365_data = s0[20];
          float v370_data = s0[36];
          float v375_data = s0[52];
          float v380_data = s0[68];
          float v385_data = s0[84];
          float v390_data = s0[100];
          float v395_data = s0[116];
          float v400_data = s0[132];
          float v405_data = s0[148];
          float v410_data = s0[164];
          float v415_data = s0[180];
          float v420_data = s0[196];
          float v425_data = s0[212];
          float v430_data = s0[228];
          float v435_data = s0[244];
          float v439_data = r0[5];
          float v440_data = s0[5];
          float v445_data = s0[21];
          float v450_data = s0[37];
          float v455_data = s0[53];
          float v460_data = s0[69];
          float v465_data = s0[85];
          float v470_data = s0[101];
          float v475_data = s0[117];
          float v480_data = s0[133];
          float v485_data = s0[149];
          float v490_data = s0[165];
          float v495_data = s0[181];
          float v500_data = s0[197];
          float v505_data = s0[213];
          float v510_data = s0[229];
          float v515_data = s0[245];
          float v519_data = r0[6];
          float v520_data = s0[6];
          float v525_data = s0[22];
          float v530_data = s0[38];
          float v535_data = s0[54];
          float v540_data = s0[70];
          float v545_data = s0[86];
          float v550_data = s0[102];
          float v555_data = s0[118];
          float v560_data = s0[134];
          float v565_data = s0[150];
          float v570_data = s0[166];
          float v575_data = s0[182];
          float v580_data = s0[198];
          float v585_data = s0[214];
          float v590_data = s0[230];
          float v595_data = s0[246];
          float v599_data = r0[7];
          float v600_data = s0[7];
          float v605_data = s0[23];
          float v610_data = s0[39];
          float v615_data = s0[55];
          float v620_data = s0[71];
          float v625_data = s0[87];
          float v630_data = s0[103];
          float v635_data = s0[119];
          float v640_data = s0[135];
          float v645_data = s0[151];
          float v650_data = s0[167];
          float v655_data = s0[183];
          float v660_data = s0[199];
          float v665_data = s0[215];
          float v670_data = s0[231];
          float v675_data = s0[247];
          float v679_data = r0[8];
          float v680_data = s0[8];
          float v685_data = s0[24];
          float v690_data = s0[40];
          float v695_data = s0[56];
          float v700_data = s0[72];
          float v705_data = s0[88];
          float v710_data = s0[104];
          float v715_data = s0[120];
          float v720_data = s0[136];
          float v725_data = s0[152];
          float v730_data = s0[168];
          float v735_data = s0[184];
          float v740_data = s0[200];
          float v745_data = s0[216];
          float v750_data = s0[232];
          float v755_data = s0[248];
          float v759_data = r0[9];
          float v760_data = s0[9];
          float v765_data = s0[25];
          float v770_data = s0[41];
          float v775_data = s0[57];
          float v780_data = s0[73];
          float v785_data = s0[89];
          float v790_data = s0[105];
          float v795_data = s0[121];
          float v800_data = s0[137];
          float v805_data = s0[153];
          float v810_data = s0[169];
          float v815_data = s0[185];
          float v820_data = s0[201];
          float v825_data = s0[217];
          float v830_data = s0[233];
          float v835_data = s0[249];
          float v839_data = r0[10];
          float v840_data = s0[10];
          float v845_data = s0[26];
          float v850_data = s0[42];
          float v855_data = s0[58];
          float v860_data = s0[74];
          float v865_data = s0[90];
          float v870_data = s0[106];
          float v875_data = s0[122];
          float v880_data = s0[138];
          float v885_data = s0[154];
          float v890_data = s0[170];
          float v895_data = s0[186];
          float v900_data = s0[202];
          float v905_data = s0[218];
          float v910_data = s0[234];
          float v915_data = s0[250];
          float v919_data = r0[11];
          float v920_data = s0[11];
          float v925_data = s0[27];
          float v930_data = s0[43];
          float v935_data = s0[59];
          float v940_data = s0[75];
          float v945_data = s0[91];
          float v950_data = s0[107];
          float v955_data = s0[123];
          float v960_data = s0[139];
          float v965_data = s0[155];
          float v970_data = s0[171];
          float v975_data = s0[187];
          float v980_data = s0[203];
          float v985_data = s0[219];
          float v990_data = s0[235];
          float v995_data = s0[251];
          float v999_data = r0[12];
          float v1000_data = s0[12];
          float v1005_data = s0[28];
          float v1010_data = s0[44];
          float v1015_data = s0[60];
          float v1020_data = s0[76];
          float v1025_data = s0[92];
          float v1030_data = s0[108];
          float v1035_data = s0[124];
          float v1040_data = s0[140];
          float v1045_data = s0[156];
          float v1050_data = s0[172];
          float v1055_data = s0[188];
          float v1060_data = s0[204];
          float v1065_data = s0[220];
          float v1070_data = s0[236];
          float v1075_data = s0[252];
          float v1079_data = r0[13];
          float v1080_data = s0[13];
          float v1085_data = s0[29];
          float v1090_data = s0[45];
          float v1095_data = s0[61];
          float v1100_data = s0[77];
          float v1105_data = s0[93];
          float v1110_data = s0[109];
          float v1115_data = s0[125];
          float v1120_data = s0[141];
          float v1125_data = s0[157];
          float v1130_data = s0[173];
          float v1135_data = s0[189];
          float v1140_data = s0[205];
          float v1145_data = s0[221];
          float v1150_data = s0[237];
          float v1155_data = s0[253];
          float v1159_data = r0[14];
          float v1160_data = s0[14];
          float v1163_p = ((((((((((((((v39_data * v40_data) + (v119_data * v120_data)) + (v199_data * v200_data)) + (v279_data * v280_data)) + (v359_data * v360_data)) + (v439_data * v440_data)) + (v519_data * v520_data)) + (v599_data * v600_data)) + (v679_data * v680_data)) + (v759_data * v760_data)) + (v839_data * v840_data)) + (v919_data * v920_data)) + (v999_data * v1000_data)) + (v1079_data * v1080_data)) + (v1159_data * v1160_data);
          float v1165_data = s0[30];
          float v1168_p = ((((((((((((((v39_data * v45_data) + (v119_data * v125_data)) + (v199_data * v205_data)) + (v279_data * v285_data)) + (v359_data * v365_data)) + (v439_data * v445_data)) + (v519_data * v525_data)) + (v599_data * v605_data)) + (v679_data * v685_data)) + (v759_data * v765_data)) + (v839_data * v845_data)) + (v919_data * v925_data)) + (v999_data * v1005_data)) + (v1079_data * v1085_data)) + (v1159_data * v1165_data);
          float v1170_data = s0[46];
          float v1173_p = ((((((((((((((v39_data * v50_data) + (v119_data * v130_data)) + (v199_data * v210_data)) + (v279_data * v290_data)) + (v359_data * v370_data)) + (v439_data * v450_data)) + (v519_data * v530_data)) + (v599_data * v610_data)) + (v679_data * v690_data)) + (v759_data * v770_data)) + (v839_data * v850_data)) + (v919_data * v930_data)) + (v999_data * v1010_data)) + (v1079_data * v1090_data)) + (v1159_data * v1170_data);
          float v1175_data = s0[62];
          float v1178_p = ((((((((((((((v39_data * v55_data) + (v119_data * v135_data)) + (v199_data * v215_data)) + (v279_data * v295_data)) + (v359_data * v375_data)) + (v439_data * v455_data)) + (v519_data * v535_data)) + (v599_data * v615_data)) + (v679_data * v695_data)) + (v759_data * v775_data)) + (v839_data * v855_data)) + (v919_data * v935_data)) + (v999_data * v1015_data)) + (v1079_data * v1095_data)) + (v1159_data * v1175_data);
          float v1180_data = s0[78];
          float v1183_p = ((((((((((((((v39_data * v60_data) + (v119_data * v140_data)) + (v199_data * v220_data)) + (v279_data * v300_data)) + (v359_data * v380_data)) + (v439_data * v460_data)) + (v519_data * v540_data)) + (v599_data * v620_data)) + (v679_data * v700_data)) + (v759_data * v780_data)) + (v839_data * v860_data)) + (v919_data * v940_data)) + (v999_data * v1020_data)) + (v1079_data * v1100_data)) + (v1159_data * v1180_data);
          float v1185_data = s0[94];
          float v1188_p = ((((((((((((((v39_data * v65_data) + (v119_data * v145_data)) + (v199_data * v225_data)) + (v279_data * v305_data)) + (v359_data * v385_data)) + (v439_data * v465_data)) + (v519_data * v545_data)) + (v599_data * v625_data)) + (v679_data * v705_data)) + (v759_data * v785_data)) + (v839_data * v865_data)) + (v919_data * v945_data)) + (v999_data * v1025_data)) + (v1079_data * v1105_data)) + (v1159_data * v1185_data);
          float v1190_data = s0[110];
          float v1193_p = ((((((((((((((v39_data * v70_data) + (v119_data * v150_data)) + (v199_data * v230_data)) + (v279_data * v310_data)) + (v359_data * v390_data)) + (v439_data * v470_data)) + (v519_data * v550_data)) + (v599_data * v630_data)) + (v679_data * v710_data)) + (v759_data * v790_data)) + (v839_data * v870_data)) + (v919_data * v950_data)) + (v999_data * v1030_data)) + (v1079_data * v1110_data)) + (v1159_data * v1190_data);
          float v1195_data = s0[126];
          float v1198_p = ((((((((((((((v39_data * v75_data) + (v119_data * v155_data)) + (v199_data * v235_data)) + (v279_data * v315_data)) + (v359_data * v395_data)) + (v439_data * v475_data)) + (v519_data * v555_data)) + (v599_data * v635_data)) + (v679_data * v715_data)) + (v759_data * v795_data)) + (v839_data * v875_data)) + (v919_data * v955_data)) + (v999_data * v1035_data)) + (v1079_data * v1115_data)) + (v1159_data * v1195_data);
          float v1200_data = s0[142];
          float v1203_p = ((((((((((((((v39_data * v80_data) + (v119_data * v160_data)) + (v199_data * v240_data)) + (v279_data * v320_data)) + (v359_data * v400_data)) + (v439_data * v480_data)) + (v519_data * v560_data)) + (v599_data * v640_data)) + (v679_data * v720_data)) + (v759_data * v800_data)) + (v839_data * v880_data)) + (v919_data * v960_data)) + (v999_data * v1040_data)) + (v1079_data * v1120_data)) + (v1159_data * v1200_data);
          float v1205_data = s0[158];
          float v1208_p = ((((((((((((((v39_data * v85_data) + (v119_data * v165_data)) + (v199_data * v245_data)) + (v279_data * v325_data)) + (v359_data * v405_data)) + (v439_data * v485_data)) + (v519_data * v565_data)) + (v599_data * v645_data)) + (v679_data * v725_data)) + (v759_data * v805_data)) + (v839_data * v885_data)) + (v919_data * v965_data)) + (v999_data * v1045_data)) + (v1079_data * v1125_data)) + (v1159_data * v1205_data);
          float v1210_data = s0[174];
          float v1213_p = ((((((((((((((v39_data * v90_data) + (v119_data * v170_data)) + (v199_data * v250_data)) + (v279_data * v330_data)) + (v359_data * v410_data)) + (v439_data * v490_data)) + (v519_data * v570_data)) + (v599_data * v650_data)) + (v679_data * v730_data)) + (v759_data * v810_data)) + (v839_data * v890_data)) + (v919_data * v970_data)) + (v999_data * v1050_data)) + (v1079_data * v1130_data)) + (v1159_data * v1210_data);
          float v1215_data = s0[190];
          float v1218_p = ((((((((((((((v39_data * v95_data) + (v119_data * v175_data)) + (v199_data * v255_data)) + (v279_data * v335_data)) + (v359_data * v415_data)) + (v439_data * v495_data)) + (v519_data * v575_data)) + (v599_data * v655_data)) + (v679_data * v735_data)) + (v759_data * v815_data)) + (v839_data * v895_data)) + (v919_data * v975_data)) + (v999_data * v1055_data)) + (v1079_data * v1135_data)) + (v1159_data * v1215_data);
          float v1220_data = s0[206];
          float v1223_p = ((((((((((((((v39_data * v100_data) + (v119_data * v180_data)) + (v199_data * v260_data)) + (v279_data * v340_data)) + (v359_data * v420_data)) + (v439_data * v500_data)) + (v519_data * v580_data)) + (v599_data * v660_data)) + (v679_data * v740_data)) + (v759_data * v820_data)) + (v839_data * v900_data)) + (v919_data * v980_data)) + (v999_data * v1060_data)) + (v1079_data * v1140_data)) + (v1159_data * v1220_data);
          float v1225_data = s0[222];
          float v1228_p = ((((((((((((((v39_data * v105_data) + (v119_data * v185_data)) + (v199_data * v265_data)) + (v279_data * v345_data)) + (v359_data * v425_data)) + (v439_data * v505_data)) + (v519_data * v585_data)) + (v599_data * v665_data)) + (v679_data * v745_data)) + (v759_data * v825_data)) + (v839_data * v905_data)) + (v919_data * v985_data)) + (v999_data * v1065_data)) + (v1079_data * v1145_data)) + (v1159_data * v1225_data);
          float v1230_data = s0[238];
          float v1233_p = ((((((((((((((v39_data * v110_data) + (v119_data * v190_data)) + (v199_data * v270_data)) + (v279_data * v350_data)) + (v359_data * v430_data)) + (v439_data * v510_data)) + (v519_data * v590_data)) + (v599_data * v670_data)) + (v679_data * v750_data)) + (v759_data * v830_data)) + (v839_data * v910_data)) + (v919_data * v990_data)) + (v999_data * v1070_data)) + (v1079_data * v1150_data)) + (v1159_data * v1230_data);
          float v1235_data = s0[254];
          float v1238_p = ((((((((((((((v39_data * v115_data) + (v119_data * v195_data)) + (v199_data * v275_data)) + (v279_data * v355_data)) + (v359_data * v435_data)) + (v439_data * v515_data)) + (v519_data * v595_data)) + (v599_data * v675_data)) + (v679_data * v755_data)) + (v759_data * v835_data)) + (v839_data * v915_data)) + (v919_data * v995_data)) + (v999_data * v1075_data)) + (v1079_data * v1155_data)) + (v1159_data * v1235_data);
          float v1239_data = r0[15];
          // r0 = load{g>r}(wrap_glb_m1);
          #pragma unroll
          for (int32_t v1354_i1 = 0; v1354_i1 < 16; ++v1354_i1) {
            float v1357_data = v1341_g ? (__ldcg(&wrap_glb_m1[(v1340_lead + (v1354_i1 * 16))])) : (0.0f);
            r0[v1354_i1] = v1357_data;
          }
          float v1240_data = s0[15];
          ir1[0] = (v1163_p + (v1239_data * v1240_data));
          float v1245_data = s0[31];
          ir1[1] = (v1168_p + (v1239_data * v1245_data));
          float v1250_data = s0[47];
          ir1[2] = (v1173_p + (v1239_data * v1250_data));
          float v1255_data = s0[63];
          ir1[3] = (v1178_p + (v1239_data * v1255_data));
          float v1260_data = s0[79];
          ir1[4] = (v1183_p + (v1239_data * v1260_data));
          float v1265_data = s0[95];
          ir1[5] = (v1188_p + (v1239_data * v1265_data));
          float v1270_data = s0[111];
          ir1[6] = (v1193_p + (v1239_data * v1270_data));
          float v1275_data = s0[127];
          ir1[7] = (v1198_p + (v1239_data * v1275_data));
          float v1280_data = s0[143];
          ir1[8] = (v1203_p + (v1239_data * v1280_data));
          float v1285_data = s0[159];
          ir1[9] = (v1208_p + (v1239_data * v1285_data));
          float v1290_data = s0[175];
          ir1[10] = (v1213_p + (v1239_data * v1290_data));
          float v1295_data = s0[191];
          ir1[11] = (v1218_p + (v1239_data * v1295_data));
          float v1300_data = s0[207];
          ir1[12] = (v1223_p + (v1239_data * v1300_data));
          float v1305_data = s0[223];
          ir1[13] = (v1228_p + (v1239_data * v1305_data));
          float v1310_data = s0[239];
          ir1[14] = (v1233_p + (v1239_data * v1310_data));
          float v1315_data = s0[255];
          ir1[15] = (v1238_p + (v1239_data * v1315_data));
          // r1 = ir1
          // glb_m0 = store{r>g}(r1);
          if (v1341_g) {
            #pragma unroll
            for (int32_t v1324_i1 = 0; v1324_i1 < 16; ++v1324_i1) {
              float v1326_data = ir1[v1324_i1];
              glb_m0[(v1340_lead + (v1324_i1 * 16))] = v1326_data;
            }
          }
          __syncwarp();
        }
        else {
          __pipeline_wait_prior(0);
          // r0 = load{g>r}(wrap_glb_m1);
          #pragma unroll
          for (int32_t v1346_i1 = 0; v1346_i1 < 16; ++v1346_i1) {
            float v1349_data = v1341_g ? (__ldcg(&wrap_glb_m1[(v1340_lead + (v1346_i1 * 16))])) : (0.0f);
            r0[v1346_i1] = v1349_data;
          }
        }
        // s0 = load{g>s}(wrap_glb_m2[0, 1])
        #pragma unroll
        for (int32_t v1364_i = 0; v1364_i < 8; ++v1364_i) {
          int32_t v1366_a = threadIdx.x + (v1364_i * 32);
          __pipeline_memcpy_async(&s0[v1366_a], &wrap_glb_m2[v1366_a], 4);
        }
        __pipeline_commit();
        v1336_flagWord = v1337_flagWordNext;
        v1337_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[256 * threadIdx.y + 0];
      size_t v1382_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < numElements1) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      cooperative_groups::this_grid().sync();
      float * __restrict__ s0 = &localShrMem0[0];
      float r0[16]{};
      const uint32_t flagWordFirst = flags1 == nullptr ? 1 : flags1[v1382_batchId1];
      size_t v2711_flagSecondAhead = v1382_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags1 == nullptr ? 1 : flags1[((v2711_flagSecondAhead < numElements1) ? v2711_flagSecondAhead : v1382_batchId1)];
      float *const __restrict__ peel_glb_m0 = &m0[v1382_batchId1 * 256 + 0 + m0_extraOffset];
      int32_t v2719_lead = threadIdx.x % 32;
      bool v2720_g = v2719_lead < 16;
      // r0 = load{g>r}(peel_glb_m0);
      #pragma unroll
      for (int32_t v2729_i1 = 0; v2729_i1 < 16; ++v2729_i1) {
        float v2732_data = v2720_g ? (__ldcg(&peel_glb_m0[(v2719_lead + (v2729_i1 * 16))])) : (0.0f);
        r0[v2729_i1] = v2732_data;
      }
      const float *const __restrict__ peel_glb_m4 = &m4[v1382_batchId1 * 256 + 0 + m4_extraOffset];
      // s0 = load{g>s}(peel_glb_m4[0, 1])
      #pragma unroll
      for (int32_t v2739_i = 0; v2739_i < 8; ++v2739_i) {
        int32_t v2741_a = threadIdx.x + (v2739_i * 32);
        __pipeline_memcpy_async(&s0[v2741_a], &peel_glb_m4[v2741_a], 4);
      }
      __pipeline_commit();
      uint32_t v2715_flagWord = flagWordFirst;
      uint32_t v2716_flagWordNext = flagWordSecond;
      for (size_t v1387_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v1387_batchId0 < numElements1; v1387_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v1388_ahead1 = v1387_batchId0 + (gridDim.x * blockDim.y);
        size_t v1390_batchId1 = (v1388_ahead1 < numElements1) ? v1388_ahead1 : v1387_batchId0;
        float *const __restrict__ wrap_glb_m0 = &m0[v1390_batchId1 * 256 + 0 + m0_extraOffset];
        const float *const __restrict__ wrap_glb_m4 = &m4[v1390_batchId1 * 256 + 0 + m4_extraOffset];
        size_t v2751_flagAheadAhead = v1390_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags1 == nullptr ? 1 : flags1[((v2751_flagAheadAhead < numElements1) ? v2751_flagAheadAhead : v1390_batchId1)];
        const bool allowed = static_cast<bool>(v2715_flagWord);
        if (allowed) {
          const float *const __restrict__ glb_m1 = &m1[v1387_batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v1387_batchId0 * 256 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[v1387_batchId0 * 256 + 0 + m3_extraOffset];
          // wait(s0 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          float r1[16]{};
          // ir1 = +(r0 * s0)
          // [(0, 16), (0, 16)] [(0, 16)]
          float ir1[16]{};
          float v1418_data = r0[0];
          __syncwarp();
          float v1419_data = s0[0];
          float v1424_data = s0[16];
          float v1429_data = s0[32];
          float v1434_data = s0[48];
          float v1439_data = s0[64];
          float v1444_data = s0[80];
          float v1449_data = s0[96];
          float v1454_data = s0[112];
          float v1459_data = s0[128];
          float v1464_data = s0[144];
          float v1469_data = s0[160];
          float v1474_data = s0[176];
          float v1479_data = s0[192];
          float v1484_data = s0[208];
          float v1489_data = s0[224];
          float v1494_data = s0[240];
          float v1498_data = r0[1];
          float v1499_data = s0[1];
          float v1504_data = s0[17];
          float v1509_data = s0[33];
          float v1514_data = s0[49];
          float v1519_data = s0[65];
          float v1524_data = s0[81];
          float v1529_data = s0[97];
          float v1534_data = s0[113];
          float v1539_data = s0[129];
          float v1544_data = s0[145];
          float v1549_data = s0[161];
          float v1554_data = s0[177];
          float v1559_data = s0[193];
          float v1564_data = s0[209];
          float v1569_data = s0[225];
          float v1574_data = s0[241];
          float v1578_data = r0[2];
          float v1579_data = s0[2];
          float v1584_data = s0[18];
          float v1589_data = s0[34];
          float v1594_data = s0[50];
          float v1599_data = s0[66];
          float v1604_data = s0[82];
          float v1609_data = s0[98];
          float v1614_data = s0[114];
          float v1619_data = s0[130];
          float v1624_data = s0[146];
          float v1629_data = s0[162];
          float v1634_data = s0[178];
          float v1639_data = s0[194];
          float v1644_data = s0[210];
          float v1649_data = s0[226];
          float v1654_data = s0[242];
          float v1658_data = r0[3];
          float v1659_data = s0[3];
          float v1664_data = s0[19];
          float v1669_data = s0[35];
          float v1674_data = s0[51];
          float v1679_data = s0[67];
          float v1684_data = s0[83];
          float v1689_data = s0[99];
          float v1694_data = s0[115];
          float v1699_data = s0[131];
          float v1704_data = s0[147];
          float v1709_data = s0[163];
          float v1714_data = s0[179];
          float v1719_data = s0[195];
          float v1724_data = s0[211];
          float v1729_data = s0[227];
          float v1734_data = s0[243];
          float v1738_data = r0[4];
          float v1739_data = s0[4];
          float v1744_data = s0[20];
          float v1749_data = s0[36];
          float v1754_data = s0[52];
          float v1759_data = s0[68];
          float v1764_data = s0[84];
          float v1769_data = s0[100];
          float v1774_data = s0[116];
          float v1779_data = s0[132];
          float v1784_data = s0[148];
          float v1789_data = s0[164];
          float v1794_data = s0[180];
          float v1799_data = s0[196];
          float v1804_data = s0[212];
          float v1809_data = s0[228];
          float v1814_data = s0[244];
          float v1818_data = r0[5];
          float v1819_data = s0[5];
          float v1824_data = s0[21];
          float v1829_data = s0[37];
          float v1834_data = s0[53];
          float v1839_data = s0[69];
          float v1844_data = s0[85];
          float v1849_data = s0[101];
          float v1854_data = s0[117];
          float v1859_data = s0[133];
          float v1864_data = s0[149];
          float v1869_data = s0[165];
          float v1874_data = s0[181];
          float v1879_data = s0[197];
          float v1884_data = s0[213];
          float v1889_data = s0[229];
          float v1894_data = s0[245];
          float v1898_data = r0[6];
          float v1899_data = s0[6];
          float v1904_data = s0[22];
          float v1909_data = s0[38];
          float v1914_data = s0[54];
          float v1919_data = s0[70];
          float v1924_data = s0[86];
          float v1929_data = s0[102];
          float v1934_data = s0[118];
          float v1939_data = s0[134];
          float v1944_data = s0[150];
          float v1949_data = s0[166];
          float v1954_data = s0[182];
          float v1959_data = s0[198];
          float v1964_data = s0[214];
          float v1969_data = s0[230];
          float v1974_data = s0[246];
          float v1978_data = r0[7];
          float v1979_data = s0[7];
          float v1984_data = s0[23];
          float v1989_data = s0[39];
          float v1994_data = s0[55];
          float v1999_data = s0[71];
          float v2004_data = s0[87];
          float v2009_data = s0[103];
          float v2014_data = s0[119];
          float v2019_data = s0[135];
          float v2024_data = s0[151];
          float v2029_data = s0[167];
          float v2034_data = s0[183];
          float v2039_data = s0[199];
          float v2044_data = s0[215];
          float v2049_data = s0[231];
          float v2054_data = s0[247];
          float v2058_data = r0[8];
          float v2059_data = s0[8];
          float v2064_data = s0[24];
          float v2069_data = s0[40];
          float v2074_data = s0[56];
          float v2079_data = s0[72];
          float v2084_data = s0[88];
          float v2089_data = s0[104];
          float v2094_data = s0[120];
          float v2099_data = s0[136];
          float v2104_data = s0[152];
          float v2109_data = s0[168];
          float v2114_data = s0[184];
          float v2119_data = s0[200];
          float v2124_data = s0[216];
          float v2129_data = s0[232];
          float v2134_data = s0[248];
          float v2138_data = r0[9];
          float v2139_data = s0[9];
          float v2144_data = s0[25];
          float v2149_data = s0[41];
          float v2154_data = s0[57];
          float v2159_data = s0[73];
          float v2164_data = s0[89];
          float v2169_data = s0[105];
          float v2174_data = s0[121];
          float v2179_data = s0[137];
          float v2184_data = s0[153];
          float v2189_data = s0[169];
          float v2194_data = s0[185];
          float v2199_data = s0[201];
          float v2204_data = s0[217];
          float v2209_data = s0[233];
          float v2214_data = s0[249];
          float v2218_data = r0[10];
          float v2219_data = s0[10];
          float v2224_data = s0[26];
          float v2229_data = s0[42];
          float v2234_data = s0[58];
          float v2239_data = s0[74];
          float v2244_data = s0[90];
          float v2249_data = s0[106];
          float v2254_data = s0[122];
          float v2259_data = s0[138];
          float v2264_data = s0[154];
          float v2269_data = s0[170];
          float v2274_data = s0[186];
          float v2279_data = s0[202];
          float v2284_data = s0[218];
          float v2289_data = s0[234];
          float v2294_data = s0[250];
          float v2298_data = r0[11];
          float v2299_data = s0[11];
          float v2304_data = s0[27];
          float v2309_data = s0[43];
          float v2314_data = s0[59];
          float v2319_data = s0[75];
          float v2324_data = s0[91];
          float v2329_data = s0[107];
          float v2334_data = s0[123];
          float v2339_data = s0[139];
          float v2344_data = s0[155];
          float v2349_data = s0[171];
          float v2354_data = s0[187];
          float v2359_data = s0[203];
          float v2364_data = s0[219];
          float v2369_data = s0[235];
          float v2374_data = s0[251];
          float v2378_data = r0[12];
          float v2379_data = s0[12];
          float v2384_data = s0[28];
          float v2389_data = s0[44];
          float v2394_data = s0[60];
          float v2399_data = s0[76];
          float v2404_data = s0[92];
          float v2409_data = s0[108];
          float v2414_data = s0[124];
          float v2419_data = s0[140];
          float v2424_data = s0[156];
          float v2429_data = s0[172];
          float v2434_data = s0[188];
          float v2439_data = s0[204];
          float v2444_data = s0[220];
          float v2449_data = s0[236];
          float v2454_data = s0[252];
          float v2458_data = r0[13];
          float v2459_data = s0[13];
          float v2464_data = s0[29];
          float v2469_data = s0[45];
          float v2474_data = s0[61];
          float v2479_data = s0[77];
          float v2484_data = s0[93];
          float v2489_data = s0[109];
          float v2494_data = s0[125];
          float v2499_data = s0[141];
          float v2504_data = s0[157];
          float v2509_data = s0[173];
          float v2514_data = s0[189];
          float v2519_data = s0[205];
          float v2524_data = s0[221];
          float v2529_data = s0[237];
          float v2534_data = s0[253];
          float v2538_data = r0[14];
          float v2539_data = s0[14];
          float v2542_p = ((((((((((((((v1418_data * v1419_data) + (v1498_data * v1499_data)) + (v1578_data * v1579_data)) + (v1658_data * v1659_data)) + (v1738_data * v1739_data)) + (v1818_data * v1819_data)) + (v1898_data * v1899_data)) + (v1978_data * v1979_data)) + (v2058_data * v2059_data)) + (v2138_data * v2139_data)) + (v2218_data * v2219_data)) + (v2298_data * v2299_data)) + (v2378_data * v2379_data)) + (v2458_data * v2459_data)) + (v2538_data * v2539_data);
          float v2544_data = s0[30];
          float v2547_p = ((((((((((((((v1418_data * v1424_data) + (v1498_data * v1504_data)) + (v1578_data * v1584_data)) + (v1658_data * v1664_data)) + (v1738_data * v1744_data)) + (v1818_data * v1824_data)) + (v1898_data * v1904_data)) + (v1978_data * v1984_data)) + (v2058_data * v2064_data)) + (v2138_data * v2144_data)) + (v2218_data * v2224_data)) + (v2298_data * v2304_data)) + (v2378_data * v2384_data)) + (v2458_data * v2464_data)) + (v2538_data * v2544_data);
          float v2549_data = s0[46];
          float v2552_p = ((((((((((((((v1418_data * v1429_data) + (v1498_data * v1509_data)) + (v1578_data * v1589_data)) + (v1658_data * v1669_data)) + (v1738_data * v1749_data)) + (v1818_data * v1829_data)) + (v1898_data * v1909_data)) + (v1978_data * v1989_data)) + (v2058_data * v2069_data)) + (v2138_data * v2149_data)) + (v2218_data * v2229_data)) + (v2298_data * v2309_data)) + (v2378_data * v2389_data)) + (v2458_data * v2469_data)) + (v2538_data * v2549_data);
          float v2554_data = s0[62];
          float v2557_p = ((((((((((((((v1418_data * v1434_data) + (v1498_data * v1514_data)) + (v1578_data * v1594_data)) + (v1658_data * v1674_data)) + (v1738_data * v1754_data)) + (v1818_data * v1834_data)) + (v1898_data * v1914_data)) + (v1978_data * v1994_data)) + (v2058_data * v2074_data)) + (v2138_data * v2154_data)) + (v2218_data * v2234_data)) + (v2298_data * v2314_data)) + (v2378_data * v2394_data)) + (v2458_data * v2474_data)) + (v2538_data * v2554_data);
          float v2559_data = s0[78];
          float v2562_p = ((((((((((((((v1418_data * v1439_data) + (v1498_data * v1519_data)) + (v1578_data * v1599_data)) + (v1658_data * v1679_data)) + (v1738_data * v1759_data)) + (v1818_data * v1839_data)) + (v1898_data * v1919_data)) + (v1978_data * v1999_data)) + (v2058_data * v2079_data)) + (v2138_data * v2159_data)) + (v2218_data * v2239_data)) + (v2298_data * v2319_data)) + (v2378_data * v2399_data)) + (v2458_data * v2479_data)) + (v2538_data * v2559_data);
          float v2564_data = s0[94];
          float v2567_p = ((((((((((((((v1418_data * v1444_data) + (v1498_data * v1524_data)) + (v1578_data * v1604_data)) + (v1658_data * v1684_data)) + (v1738_data * v1764_data)) + (v1818_data * v1844_data)) + (v1898_data * v1924_data)) + (v1978_data * v2004_data)) + (v2058_data * v2084_data)) + (v2138_data * v2164_data)) + (v2218_data * v2244_data)) + (v2298_data * v2324_data)) + (v2378_data * v2404_data)) + (v2458_data * v2484_data)) + (v2538_data * v2564_data);
          float v2569_data = s0[110];
          float v2572_p = ((((((((((((((v1418_data * v1449_data) + (v1498_data * v1529_data)) + (v1578_data * v1609_data)) + (v1658_data * v1689_data)) + (v1738_data * v1769_data)) + (v1818_data * v1849_data)) + (v1898_data * v1929_data)) + (v1978_data * v2009_data)) + (v2058_data * v2089_data)) + (v2138_data * v2169_data)) + (v2218_data * v2249_data)) + (v2298_data * v2329_data)) + (v2378_data * v2409_data)) + (v2458_data * v2489_data)) + (v2538_data * v2569_data);
          float v2574_data = s0[126];
          float v2577_p = ((((((((((((((v1418_data * v1454_data) + (v1498_data * v1534_data)) + (v1578_data * v1614_data)) + (v1658_data * v1694_data)) + (v1738_data * v1774_data)) + (v1818_data * v1854_data)) + (v1898_data * v1934_data)) + (v1978_data * v2014_data)) + (v2058_data * v2094_data)) + (v2138_data * v2174_data)) + (v2218_data * v2254_data)) + (v2298_data * v2334_data)) + (v2378_data * v2414_data)) + (v2458_data * v2494_data)) + (v2538_data * v2574_data);
          float v2579_data = s0[142];
          float v2582_p = ((((((((((((((v1418_data * v1459_data) + (v1498_data * v1539_data)) + (v1578_data * v1619_data)) + (v1658_data * v1699_data)) + (v1738_data * v1779_data)) + (v1818_data * v1859_data)) + (v1898_data * v1939_data)) + (v1978_data * v2019_data)) + (v2058_data * v2099_data)) + (v2138_data * v2179_data)) + (v2218_data * v2259_data)) + (v2298_data * v2339_data)) + (v2378_data * v2419_data)) + (v2458_data * v2499_data)) + (v2538_data * v2579_data);
          float v2584_data = s0[158];
          float v2587_p = ((((((((((((((v1418_data * v1464_data) + (v1498_data * v1544_data)) + (v1578_data * v1624_data)) + (v1658_data * v1704_data)) + (v1738_data * v1784_data)) + (v1818_data * v1864_data)) + (v1898_data * v1944_data)) + (v1978_data * v2024_data)) + (v2058_data * v2104_data)) + (v2138_data * v2184_data)) + (v2218_data * v2264_data)) + (v2298_data * v2344_data)) + (v2378_data * v2424_data)) + (v2458_data * v2504_data)) + (v2538_data * v2584_data);
          float v2589_data = s0[174];
          float v2592_p = ((((((((((((((v1418_data * v1469_data) + (v1498_data * v1549_data)) + (v1578_data * v1629_data)) + (v1658_data * v1709_data)) + (v1738_data * v1789_data)) + (v1818_data * v1869_data)) + (v1898_data * v1949_data)) + (v1978_data * v2029_data)) + (v2058_data * v2109_data)) + (v2138_data * v2189_data)) + (v2218_data * v2269_data)) + (v2298_data * v2349_data)) + (v2378_data * v2429_data)) + (v2458_data * v2509_data)) + (v2538_data * v2589_data);
          float v2594_data = s0[190];
          float v2597_p = ((((((((((((((v1418_data * v1474_data) + (v1498_data * v1554_data)) + (v1578_data * v1634_data)) + (v1658_data * v1714_data)) + (v1738_data * v1794_data)) + (v1818_data * v1874_data)) + (v1898_data * v1954_data)) + (v1978_data * v2034_data)) + (v2058_data * v2114_data)) + (v2138_data * v2194_data)) + (v2218_data * v2274_data)) + (v2298_data * v2354_data)) + (v2378_data * v2434_data)) + (v2458_data * v2514_data)) + (v2538_data * v2594_data);
          float v2599_data = s0[206];
          float v2602_p = ((((((((((((((v1418_data * v1479_data) + (v1498_data * v1559_data)) + (v1578_data * v1639_data)) + (v1658_data * v1719_data)) + (v1738_data * v1799_data)) + (v1818_data * v1879_data)) + (v1898_data * v1959_data)) + (v1978_data * v2039_data)) + (v2058_data * v2119_data)) + (v2138_data * v2199_data)) + (v2218_data * v2279_data)) + (v2298_data * v2359_data)) + (v2378_data * v2439_data)) + (v2458_data * v2519_data)) + (v2538_data * v2599_data);
          float v2604_data = s0[222];
          float v2607_p = ((((((((((((((v1418_data * v1484_data) + (v1498_data * v1564_data)) + (v1578_data * v1644_data)) + (v1658_data * v1724_data)) + (v1738_data * v1804_data)) + (v1818_data * v1884_data)) + (v1898_data * v1964_data)) + (v1978_data * v2044_data)) + (v2058_data * v2124_data)) + (v2138_data * v2204_data)) + (v2218_data * v2284_data)) + (v2298_data * v2364_data)) + (v2378_data * v2444_data)) + (v2458_data * v2524_data)) + (v2538_data * v2604_data);
          float v2609_data = s0[238];
          float v2612_p = ((((((((((((((v1418_data * v1489_data) + (v1498_data * v1569_data)) + (v1578_data * v1649_data)) + (v1658_data * v1729_data)) + (v1738_data * v1809_data)) + (v1818_data * v1889_data)) + (v1898_data * v1969_data)) + (v1978_data * v2049_data)) + (v2058_data * v2129_data)) + (v2138_data * v2209_data)) + (v2218_data * v2289_data)) + (v2298_data * v2369_data)) + (v2378_data * v2449_data)) + (v2458_data * v2529_data)) + (v2538_data * v2609_data);
          float v2614_data = s0[254];
          float v2617_p = ((((((((((((((v1418_data * v1494_data) + (v1498_data * v1574_data)) + (v1578_data * v1654_data)) + (v1658_data * v1734_data)) + (v1738_data * v1814_data)) + (v1818_data * v1894_data)) + (v1898_data * v1974_data)) + (v1978_data * v2054_data)) + (v2058_data * v2134_data)) + (v2138_data * v2214_data)) + (v2218_data * v2294_data)) + (v2298_data * v2374_data)) + (v2378_data * v2454_data)) + (v2458_data * v2534_data)) + (v2538_data * v2614_data);
          float v2618_data = r0[15];
          // r0 = load{g>r}(wrap_glb_m0);
          #pragma unroll
          for (int32_t v2733_i1 = 0; v2733_i1 < 16; ++v2733_i1) {
            float v2736_data = v2720_g ? (__ldcg(&wrap_glb_m0[(v2719_lead + (v2733_i1 * 16))])) : (0.0f);
            r0[v2733_i1] = v2736_data;
          }
          float v2619_data = s0[15];
          ir1[0] = (v2542_p + (v2618_data * v2619_data));
          float v2624_data = s0[31];
          ir1[1] = (v2547_p + (v2618_data * v2624_data));
          float v2629_data = s0[47];
          ir1[2] = (v2552_p + (v2618_data * v2629_data));
          float v2634_data = s0[63];
          ir1[3] = (v2557_p + (v2618_data * v2634_data));
          float v2639_data = s0[79];
          ir1[4] = (v2562_p + (v2618_data * v2639_data));
          float v2644_data = s0[95];
          ir1[5] = (v2567_p + (v2618_data * v2644_data));
          float v2649_data = s0[111];
          ir1[6] = (v2572_p + (v2618_data * v2649_data));
          float v2654_data = s0[127];
          ir1[7] = (v2577_p + (v2618_data * v2654_data));
          float v2659_data = s0[143];
          ir1[8] = (v2582_p + (v2618_data * v2659_data));
          float v2664_data = s0[159];
          ir1[9] = (v2587_p + (v2618_data * v2664_data));
          float v2669_data = s0[175];
          ir1[10] = (v2592_p + (v2618_data * v2669_data));
          float v2674_data = s0[191];
          ir1[11] = (v2597_p + (v2618_data * v2674_data));
          float v2679_data = s0[207];
          ir1[12] = (v2602_p + (v2618_data * v2679_data));
          float v2684_data = s0[223];
          ir1[13] = (v2607_p + (v2618_data * v2684_data));
          float v2689_data = s0[239];
          ir1[14] = (v2612_p + (v2618_data * v2689_data));
          float v2694_data = s0[255];
          ir1[15] = (v2617_p + (v2618_data * v2694_data));
          // r1 = ir1
          // glb_m3 = store{r>g}(r1);
          if (v2720_g) {
            #pragma unroll
            for (int32_t v2703_i1 = 0; v2703_i1 < 16; ++v2703_i1) {
              float v2705_data = ir1[v2703_i1];
              glb_m3[(v2719_lead + (v2703_i1 * 16))] = v2705_data;
            }
          }
          __syncwarp();
        }
        else {
          __pipeline_wait_prior(0);
          // r0 = load{g>r}(wrap_glb_m0);
          #pragma unroll
          for (int32_t v2725_i1 = 0; v2725_i1 < 16; ++v2725_i1) {
            float v2728_data = v2720_g ? (__ldcg(&wrap_glb_m0[(v2719_lead + (v2725_i1 * 16))])) : (0.0f);
            r0[v2725_i1] = v2728_data;
          }
        }
        // s0 = load{g>s}(wrap_glb_m4[0, 1])
        #pragma unroll
        for (int32_t v2743_i = 0; v2743_i < 8; ++v2743_i) {
          int32_t v2745_a = threadIdx.x + (v2743_i * 32);
          __pipeline_memcpy_async(&s0[v2745_a], &wrap_glb_m4[v2745_a], 4);
        }
        __pipeline_commit();
        v2715_flagWord = v2716_flagWordNext;
        v2716_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
  }
}

