// === base name ===
kernel_696db451bb16dd92

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_696db451bb16dd92 = {{16, 8, 1}, 16, 16, 1, 8, 8704, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_696db451bb16dd92(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_696db451bb16dd92(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float ** m2, size_t m2_extraOffset, size_t m2_slot, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_696db451bb16dd92(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_696db451bb16dd92, block.x * block.y * block.z, 2176 * sizeof(float));
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
void launcher_kernel_696db451bb16dd92(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float ** m2, size_t m2_extraOffset, size_t m2_slot, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_696db451bb16dd92(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_696db451bb16dd92, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  const float *const * m2Arg = (const float *const *)m2;
  const unsigned * elementIndicesArg = (const unsigned *)elementIndices;
  const unsigned * flags0Arg = (const unsigned *)flags0;
  kernel_kernel_696db451bb16dd92<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m1_stride, m2Arg, m2_extraOffset, m2_slot, numElements0, elementIndicesArg, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_696db451bb16dd92(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float *const * __restrict__ m2, size_t m2_extraOffset, size_t m2_slot, size_t numElements0, const unsigned * __restrict__ elementIndices, const unsigned * __restrict__ flags0) {
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
      size_t v1331_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v1331_flagSecondAhead < numElements0) ? v1331_flagSecondAhead : v3_batchId1)];
      const bool allowed_peel = static_cast<bool>(flagWordFirst);
      const size_t peel_elementIndex = static_cast<size_t>(elementIndices[v3_batchId1]);
      const float *const __restrict__ peel_glb_m1 = &m1[peel_elementIndex * m1_stride + 0 + m1_extraOffset];
      int32_t v1342_lead = threadIdx.x % 16;
      // r0 = load{g>r}(peel_glb_m1);
      #pragma unroll
      for (int32_t v1355_i0 = 0; v1355_i0 < 1; ++v1355_i0) {
        int32_t v1357_lead = v1342_lead + (v1355_i0 * 16);
        #pragma unroll
        for (int32_t v1358_i1 = 0; v1358_i1 < 16; ++v1358_i1) {
          float v1361_data = allowed_peel ? (__ldcg(&peel_glb_m1[(v1357_lead + (v1358_i1 * 16))])) : (0.0f);
          r0[(v1355_i0 + v1358_i1)] = v1361_data;
        }
      }
      const float *const __restrict__ peel_glb_m2 = &m2[peel_elementIndex * 4 + m2_slot][0 + 4 + m2_extraOffset];
      // s0 = load{g>s}(peel_glb_m2[0, 1])
      #pragma unroll
      for (int32_t v1373_i = 0; v1373_i < 16; ++v1373_i) {
        int32_t v1375_a = threadIdx.x + (v1373_i * 16);
        if (allowed_peel) {
          __pipeline_memcpy_async(&s0[v1375_a], &peel_glb_m2[v1375_a], 4);
        }
      }
      __pipeline_commit();
      uint32_t v1335_flagWord = flagWordFirst;
      uint32_t v1336_flagWordNext = flagWordSecond;
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const size_t wrap_elementIndex = static_cast<size_t>(elementIndices[v11_batchId1]);
        const float *const __restrict__ wrap_glb_m1 = &m1[wrap_elementIndex * m1_stride + 0 + m1_extraOffset];
        const float *const __restrict__ wrap_glb_m2 = &m2[wrap_elementIndex * 4 + m2_slot][0 + 4 + m2_extraOffset];
        size_t v1385_flagAheadAhead = v11_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v1385_flagAheadAhead < numElements0) ? v1385_flagAheadAhead : v11_batchId1)];
        const bool allowed = static_cast<bool>(v1335_flagWord);
        const bool allowed_next = static_cast<bool>(v1336_flagWordNext);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 300 + 0 + 8 + m0_extraOffset];
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[16]{};
          // ir1 = +(r0 * s0)
          // [(0, 16), (0, 16)] [(0, 16)]
          float ir1[16]{};
          float v38_data = r0[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v39_data = s0[0];
          float v44_data = s0[16];
          float v49_data = s0[32];
          float v54_data = s0[48];
          float v59_data = s0[64];
          float v64_data = s0[80];
          float v69_data = s0[96];
          float v74_data = s0[112];
          float v79_data = s0[128];
          float v84_data = s0[144];
          float v89_data = s0[160];
          float v94_data = s0[176];
          float v99_data = s0[192];
          float v104_data = s0[208];
          float v109_data = s0[224];
          float v114_data = s0[240];
          float v118_data = r0[1];
          float v119_data = s0[1];
          float v124_data = s0[17];
          float v129_data = s0[33];
          float v134_data = s0[49];
          float v139_data = s0[65];
          float v144_data = s0[81];
          float v149_data = s0[97];
          float v154_data = s0[113];
          float v159_data = s0[129];
          float v164_data = s0[145];
          float v169_data = s0[161];
          float v174_data = s0[177];
          float v179_data = s0[193];
          float v184_data = s0[209];
          float v189_data = s0[225];
          float v194_data = s0[241];
          float v198_data = r0[2];
          float v199_data = s0[2];
          float v204_data = s0[18];
          float v209_data = s0[34];
          float v214_data = s0[50];
          float v219_data = s0[66];
          float v224_data = s0[82];
          float v229_data = s0[98];
          float v234_data = s0[114];
          float v239_data = s0[130];
          float v244_data = s0[146];
          float v249_data = s0[162];
          float v254_data = s0[178];
          float v259_data = s0[194];
          float v264_data = s0[210];
          float v269_data = s0[226];
          float v274_data = s0[242];
          float v278_data = r0[3];
          float v279_data = s0[3];
          float v284_data = s0[19];
          float v289_data = s0[35];
          float v294_data = s0[51];
          float v299_data = s0[67];
          float v304_data = s0[83];
          float v309_data = s0[99];
          float v314_data = s0[115];
          float v319_data = s0[131];
          float v324_data = s0[147];
          float v329_data = s0[163];
          float v334_data = s0[179];
          float v339_data = s0[195];
          float v344_data = s0[211];
          float v349_data = s0[227];
          float v354_data = s0[243];
          float v358_data = r0[4];
          float v359_data = s0[4];
          float v364_data = s0[20];
          float v369_data = s0[36];
          float v374_data = s0[52];
          float v379_data = s0[68];
          float v384_data = s0[84];
          float v389_data = s0[100];
          float v394_data = s0[116];
          float v399_data = s0[132];
          float v404_data = s0[148];
          float v409_data = s0[164];
          float v414_data = s0[180];
          float v419_data = s0[196];
          float v424_data = s0[212];
          float v429_data = s0[228];
          float v434_data = s0[244];
          float v438_data = r0[5];
          float v439_data = s0[5];
          float v444_data = s0[21];
          float v449_data = s0[37];
          float v454_data = s0[53];
          float v459_data = s0[69];
          float v464_data = s0[85];
          float v469_data = s0[101];
          float v474_data = s0[117];
          float v479_data = s0[133];
          float v484_data = s0[149];
          float v489_data = s0[165];
          float v494_data = s0[181];
          float v499_data = s0[197];
          float v504_data = s0[213];
          float v509_data = s0[229];
          float v514_data = s0[245];
          float v518_data = r0[6];
          float v519_data = s0[6];
          float v524_data = s0[22];
          float v529_data = s0[38];
          float v534_data = s0[54];
          float v539_data = s0[70];
          float v544_data = s0[86];
          float v549_data = s0[102];
          float v554_data = s0[118];
          float v559_data = s0[134];
          float v564_data = s0[150];
          float v569_data = s0[166];
          float v574_data = s0[182];
          float v579_data = s0[198];
          float v584_data = s0[214];
          float v589_data = s0[230];
          float v594_data = s0[246];
          float v598_data = r0[7];
          float v599_data = s0[7];
          float v604_data = s0[23];
          float v609_data = s0[39];
          float v614_data = s0[55];
          float v619_data = s0[71];
          float v624_data = s0[87];
          float v629_data = s0[103];
          float v634_data = s0[119];
          float v639_data = s0[135];
          float v644_data = s0[151];
          float v649_data = s0[167];
          float v654_data = s0[183];
          float v659_data = s0[199];
          float v664_data = s0[215];
          float v669_data = s0[231];
          float v674_data = s0[247];
          float v678_data = r0[8];
          float v679_data = s0[8];
          float v684_data = s0[24];
          float v689_data = s0[40];
          float v694_data = s0[56];
          float v699_data = s0[72];
          float v704_data = s0[88];
          float v709_data = s0[104];
          float v714_data = s0[120];
          float v719_data = s0[136];
          float v724_data = s0[152];
          float v729_data = s0[168];
          float v734_data = s0[184];
          float v739_data = s0[200];
          float v744_data = s0[216];
          float v749_data = s0[232];
          float v754_data = s0[248];
          float v758_data = r0[9];
          float v759_data = s0[9];
          float v764_data = s0[25];
          float v769_data = s0[41];
          float v774_data = s0[57];
          float v779_data = s0[73];
          float v784_data = s0[89];
          float v789_data = s0[105];
          float v794_data = s0[121];
          float v799_data = s0[137];
          float v804_data = s0[153];
          float v809_data = s0[169];
          float v814_data = s0[185];
          float v819_data = s0[201];
          float v824_data = s0[217];
          float v829_data = s0[233];
          float v834_data = s0[249];
          float v838_data = r0[10];
          float v839_data = s0[10];
          float v844_data = s0[26];
          float v849_data = s0[42];
          float v854_data = s0[58];
          float v859_data = s0[74];
          float v864_data = s0[90];
          float v869_data = s0[106];
          float v874_data = s0[122];
          float v879_data = s0[138];
          float v884_data = s0[154];
          float v889_data = s0[170];
          float v894_data = s0[186];
          float v899_data = s0[202];
          float v904_data = s0[218];
          float v909_data = s0[234];
          float v914_data = s0[250];
          float v918_data = r0[11];
          float v919_data = s0[11];
          float v924_data = s0[27];
          float v929_data = s0[43];
          float v934_data = s0[59];
          float v939_data = s0[75];
          float v944_data = s0[91];
          float v949_data = s0[107];
          float v954_data = s0[123];
          float v959_data = s0[139];
          float v964_data = s0[155];
          float v969_data = s0[171];
          float v974_data = s0[187];
          float v979_data = s0[203];
          float v984_data = s0[219];
          float v989_data = s0[235];
          float v994_data = s0[251];
          float v998_data = r0[12];
          float v999_data = s0[12];
          float v1004_data = s0[28];
          float v1009_data = s0[44];
          float v1014_data = s0[60];
          float v1019_data = s0[76];
          float v1024_data = s0[92];
          float v1029_data = s0[108];
          float v1034_data = s0[124];
          float v1039_data = s0[140];
          float v1044_data = s0[156];
          float v1049_data = s0[172];
          float v1054_data = s0[188];
          float v1059_data = s0[204];
          float v1064_data = s0[220];
          float v1069_data = s0[236];
          float v1074_data = s0[252];
          float v1078_data = r0[13];
          float v1079_data = s0[13];
          float v1084_data = s0[29];
          float v1089_data = s0[45];
          float v1094_data = s0[61];
          float v1099_data = s0[77];
          float v1104_data = s0[93];
          float v1109_data = s0[109];
          float v1114_data = s0[125];
          float v1119_data = s0[141];
          float v1124_data = s0[157];
          float v1129_data = s0[173];
          float v1134_data = s0[189];
          float v1139_data = s0[205];
          float v1144_data = s0[221];
          float v1149_data = s0[237];
          float v1154_data = s0[253];
          float v1158_data = r0[14];
          float v1159_data = s0[14];
          float v1162_p = ((((((((((((((v38_data * v39_data) + (v118_data * v119_data)) + (v198_data * v199_data)) + (v278_data * v279_data)) + (v358_data * v359_data)) + (v438_data * v439_data)) + (v518_data * v519_data)) + (v598_data * v599_data)) + (v678_data * v679_data)) + (v758_data * v759_data)) + (v838_data * v839_data)) + (v918_data * v919_data)) + (v998_data * v999_data)) + (v1078_data * v1079_data)) + (v1158_data * v1159_data);
          float v1164_data = s0[30];
          float v1167_p = ((((((((((((((v38_data * v44_data) + (v118_data * v124_data)) + (v198_data * v204_data)) + (v278_data * v284_data)) + (v358_data * v364_data)) + (v438_data * v444_data)) + (v518_data * v524_data)) + (v598_data * v604_data)) + (v678_data * v684_data)) + (v758_data * v764_data)) + (v838_data * v844_data)) + (v918_data * v924_data)) + (v998_data * v1004_data)) + (v1078_data * v1084_data)) + (v1158_data * v1164_data);
          float v1169_data = s0[46];
          float v1172_p = ((((((((((((((v38_data * v49_data) + (v118_data * v129_data)) + (v198_data * v209_data)) + (v278_data * v289_data)) + (v358_data * v369_data)) + (v438_data * v449_data)) + (v518_data * v529_data)) + (v598_data * v609_data)) + (v678_data * v689_data)) + (v758_data * v769_data)) + (v838_data * v849_data)) + (v918_data * v929_data)) + (v998_data * v1009_data)) + (v1078_data * v1089_data)) + (v1158_data * v1169_data);
          float v1174_data = s0[62];
          float v1177_p = ((((((((((((((v38_data * v54_data) + (v118_data * v134_data)) + (v198_data * v214_data)) + (v278_data * v294_data)) + (v358_data * v374_data)) + (v438_data * v454_data)) + (v518_data * v534_data)) + (v598_data * v614_data)) + (v678_data * v694_data)) + (v758_data * v774_data)) + (v838_data * v854_data)) + (v918_data * v934_data)) + (v998_data * v1014_data)) + (v1078_data * v1094_data)) + (v1158_data * v1174_data);
          float v1179_data = s0[78];
          float v1182_p = ((((((((((((((v38_data * v59_data) + (v118_data * v139_data)) + (v198_data * v219_data)) + (v278_data * v299_data)) + (v358_data * v379_data)) + (v438_data * v459_data)) + (v518_data * v539_data)) + (v598_data * v619_data)) + (v678_data * v699_data)) + (v758_data * v779_data)) + (v838_data * v859_data)) + (v918_data * v939_data)) + (v998_data * v1019_data)) + (v1078_data * v1099_data)) + (v1158_data * v1179_data);
          float v1184_data = s0[94];
          float v1187_p = ((((((((((((((v38_data * v64_data) + (v118_data * v144_data)) + (v198_data * v224_data)) + (v278_data * v304_data)) + (v358_data * v384_data)) + (v438_data * v464_data)) + (v518_data * v544_data)) + (v598_data * v624_data)) + (v678_data * v704_data)) + (v758_data * v784_data)) + (v838_data * v864_data)) + (v918_data * v944_data)) + (v998_data * v1024_data)) + (v1078_data * v1104_data)) + (v1158_data * v1184_data);
          float v1189_data = s0[110];
          float v1192_p = ((((((((((((((v38_data * v69_data) + (v118_data * v149_data)) + (v198_data * v229_data)) + (v278_data * v309_data)) + (v358_data * v389_data)) + (v438_data * v469_data)) + (v518_data * v549_data)) + (v598_data * v629_data)) + (v678_data * v709_data)) + (v758_data * v789_data)) + (v838_data * v869_data)) + (v918_data * v949_data)) + (v998_data * v1029_data)) + (v1078_data * v1109_data)) + (v1158_data * v1189_data);
          float v1194_data = s0[126];
          float v1197_p = ((((((((((((((v38_data * v74_data) + (v118_data * v154_data)) + (v198_data * v234_data)) + (v278_data * v314_data)) + (v358_data * v394_data)) + (v438_data * v474_data)) + (v518_data * v554_data)) + (v598_data * v634_data)) + (v678_data * v714_data)) + (v758_data * v794_data)) + (v838_data * v874_data)) + (v918_data * v954_data)) + (v998_data * v1034_data)) + (v1078_data * v1114_data)) + (v1158_data * v1194_data);
          float v1199_data = s0[142];
          float v1202_p = ((((((((((((((v38_data * v79_data) + (v118_data * v159_data)) + (v198_data * v239_data)) + (v278_data * v319_data)) + (v358_data * v399_data)) + (v438_data * v479_data)) + (v518_data * v559_data)) + (v598_data * v639_data)) + (v678_data * v719_data)) + (v758_data * v799_data)) + (v838_data * v879_data)) + (v918_data * v959_data)) + (v998_data * v1039_data)) + (v1078_data * v1119_data)) + (v1158_data * v1199_data);
          float v1204_data = s0[158];
          float v1207_p = ((((((((((((((v38_data * v84_data) + (v118_data * v164_data)) + (v198_data * v244_data)) + (v278_data * v324_data)) + (v358_data * v404_data)) + (v438_data * v484_data)) + (v518_data * v564_data)) + (v598_data * v644_data)) + (v678_data * v724_data)) + (v758_data * v804_data)) + (v838_data * v884_data)) + (v918_data * v964_data)) + (v998_data * v1044_data)) + (v1078_data * v1124_data)) + (v1158_data * v1204_data);
          float v1209_data = s0[174];
          float v1212_p = ((((((((((((((v38_data * v89_data) + (v118_data * v169_data)) + (v198_data * v249_data)) + (v278_data * v329_data)) + (v358_data * v409_data)) + (v438_data * v489_data)) + (v518_data * v569_data)) + (v598_data * v649_data)) + (v678_data * v729_data)) + (v758_data * v809_data)) + (v838_data * v889_data)) + (v918_data * v969_data)) + (v998_data * v1049_data)) + (v1078_data * v1129_data)) + (v1158_data * v1209_data);
          float v1214_data = s0[190];
          float v1217_p = ((((((((((((((v38_data * v94_data) + (v118_data * v174_data)) + (v198_data * v254_data)) + (v278_data * v334_data)) + (v358_data * v414_data)) + (v438_data * v494_data)) + (v518_data * v574_data)) + (v598_data * v654_data)) + (v678_data * v734_data)) + (v758_data * v814_data)) + (v838_data * v894_data)) + (v918_data * v974_data)) + (v998_data * v1054_data)) + (v1078_data * v1134_data)) + (v1158_data * v1214_data);
          float v1219_data = s0[206];
          float v1222_p = ((((((((((((((v38_data * v99_data) + (v118_data * v179_data)) + (v198_data * v259_data)) + (v278_data * v339_data)) + (v358_data * v419_data)) + (v438_data * v499_data)) + (v518_data * v579_data)) + (v598_data * v659_data)) + (v678_data * v739_data)) + (v758_data * v819_data)) + (v838_data * v899_data)) + (v918_data * v979_data)) + (v998_data * v1059_data)) + (v1078_data * v1139_data)) + (v1158_data * v1219_data);
          float v1224_data = s0[222];
          float v1227_p = ((((((((((((((v38_data * v104_data) + (v118_data * v184_data)) + (v198_data * v264_data)) + (v278_data * v344_data)) + (v358_data * v424_data)) + (v438_data * v504_data)) + (v518_data * v584_data)) + (v598_data * v664_data)) + (v678_data * v744_data)) + (v758_data * v824_data)) + (v838_data * v904_data)) + (v918_data * v984_data)) + (v998_data * v1064_data)) + (v1078_data * v1144_data)) + (v1158_data * v1224_data);
          float v1229_data = s0[238];
          float v1232_p = ((((((((((((((v38_data * v109_data) + (v118_data * v189_data)) + (v198_data * v269_data)) + (v278_data * v349_data)) + (v358_data * v429_data)) + (v438_data * v509_data)) + (v518_data * v589_data)) + (v598_data * v669_data)) + (v678_data * v749_data)) + (v758_data * v829_data)) + (v838_data * v909_data)) + (v918_data * v989_data)) + (v998_data * v1069_data)) + (v1078_data * v1149_data)) + (v1158_data * v1229_data);
          float v1234_data = s0[254];
          float v1237_p = ((((((((((((((v38_data * v114_data) + (v118_data * v194_data)) + (v198_data * v274_data)) + (v278_data * v354_data)) + (v358_data * v434_data)) + (v438_data * v514_data)) + (v518_data * v594_data)) + (v598_data * v674_data)) + (v678_data * v754_data)) + (v758_data * v834_data)) + (v838_data * v914_data)) + (v918_data * v994_data)) + (v998_data * v1074_data)) + (v1078_data * v1154_data)) + (v1158_data * v1234_data);
          float v1238_data = r0[15];
          // r0 = load{g>r}(wrap_glb_m1);
          #pragma unroll
          for (int32_t v1363_i0 = 0; v1363_i0 < 1; ++v1363_i0) {
            int32_t v1365_lead = v1342_lead + (v1363_i0 * 16);
            #pragma unroll
            for (int32_t v1366_i1 = 0; v1366_i1 < 16; ++v1366_i1) {
              float v1369_data = allowed_next ? (__ldcg(&wrap_glb_m1[(v1365_lead + (v1366_i1 * 16))])) : (0.0f);
              r0[(v1363_i0 + v1366_i1)] = v1369_data;
            }
          }
          float v1239_data = s0[15];
          ir1[0] = (v1162_p + (v1238_data * v1239_data));
          float v1244_data = s0[31];
          ir1[1] = (v1167_p + (v1238_data * v1244_data));
          float v1249_data = s0[47];
          ir1[2] = (v1172_p + (v1238_data * v1249_data));
          float v1254_data = s0[63];
          ir1[3] = (v1177_p + (v1238_data * v1254_data));
          float v1259_data = s0[79];
          ir1[4] = (v1182_p + (v1238_data * v1259_data));
          float v1264_data = s0[95];
          ir1[5] = (v1187_p + (v1238_data * v1264_data));
          float v1269_data = s0[111];
          ir1[6] = (v1192_p + (v1238_data * v1269_data));
          float v1274_data = s0[127];
          ir1[7] = (v1197_p + (v1238_data * v1274_data));
          float v1279_data = s0[143];
          ir1[8] = (v1202_p + (v1238_data * v1279_data));
          float v1284_data = s0[159];
          ir1[9] = (v1207_p + (v1238_data * v1284_data));
          float v1289_data = s0[175];
          ir1[10] = (v1212_p + (v1238_data * v1289_data));
          float v1294_data = s0[191];
          ir1[11] = (v1217_p + (v1238_data * v1294_data));
          float v1299_data = s0[207];
          ir1[12] = (v1222_p + (v1238_data * v1299_data));
          float v1304_data = s0[223];
          ir1[13] = (v1227_p + (v1238_data * v1304_data));
          float v1309_data = s0[239];
          ir1[14] = (v1232_p + (v1238_data * v1309_data));
          float v1314_data = s0[255];
          ir1[15] = (v1237_p + (v1238_data * v1314_data));
          // r1 = ir1
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v1322_i0 = 0; v1322_i0 < 1; ++v1322_i0) {
            int32_t v1327_lead = v1342_lead + (v1322_i0 * 16);
            #pragma unroll
            for (int32_t v1323_i1 = 0; v1323_i1 < 16; ++v1323_i1) {
              float v1325_data = ir1[v1323_i1];
              glb_m0[(v1327_lead + (v1323_i1 * 16))] = v1325_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
        else {
          __pipeline_wait_prior(0);
          // r0 = load{g>r}(wrap_glb_m1);
          #pragma unroll
          for (int32_t v1347_i0 = 0; v1347_i0 < 1; ++v1347_i0) {
            int32_t v1349_lead = v1342_lead + (v1347_i0 * 16);
            #pragma unroll
            for (int32_t v1350_i1 = 0; v1350_i1 < 16; ++v1350_i1) {
              float v1353_data = allowed_next ? (__ldcg(&wrap_glb_m1[(v1349_lead + (v1350_i1 * 16))])) : (0.0f);
              r0[(v1347_i0 + v1350_i1)] = v1353_data;
            }
          }
        }
        // s0 = load{g>s}(wrap_glb_m2[0, 1])
        #pragma unroll
        for (int32_t v1377_i = 0; v1377_i < 16; ++v1377_i) {
          int32_t v1379_a = threadIdx.x + (v1377_i * 16);
          if (allowed_next) {
            __pipeline_memcpy_async(&s0[v1379_a], &wrap_glb_m2[v1379_a], 4);
          }
        }
        __pipeline_commit();
        v1335_flagWord = v1336_flagWordNext;
        v1336_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
  }
}

