// === base name ===
kernel_6aff329e159d5e42

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_6aff329e159d5e42 = {{16, 8, 1}, 16, 10, 1, 8, 5632, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_6aff329e159d5e42(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_6aff329e159d5e42(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_6aff329e159d5e42(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_6aff329e159d5e42, block.x * block.y * block.z, 1408 * sizeof(float));
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
void launcher_kernel_6aff329e159d5e42(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_6aff329e159d5e42(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_6aff329e159d5e42, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_6aff329e159d5e42<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_6aff329e159d5e42(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (10 active) x 8 per block = block 16x8x1, 5632 B shared, occupancy grid
    // operands:
    //   m0 10×9(10×9) {0..10}×{0..9} strided
    //   m1 16×20(10×17) {0..10}×{1..18} none
    //   m2 20×9(17×9) {1..18}×{0..9} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":10,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1408}],"shared_bytes":5632,"shared_elements":1408,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[10,9]],"name":"m0","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,1],[10,18]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[1,0],[18,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[10,18]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[18,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[176 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[160];
      const float *const __restrict__ glb_m1 = &m1[0];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v6_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v6_batchId0 < numElements0; v6_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v7_ahead1 = v6_batchId0 + (gridDim.x * blockDim.y);
        size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 90 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v6_batchId0 * 153 + 0 + m2_extraOffset];
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 9; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
          if (threadIdx.x < 9) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 144], &glb_m2[0 + 0 + 1 * threadIdx.x + 144], 4);
          }
          __pipeline_commit();
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r0[9]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // ir0 = +(glb_m1 * s0)
          // [(0, 10), (0, 9)] [(1, 18)]
          float ir0[9]{};
          int32_t v22_lead = threadIdx.x % 16;
          bool v26_g = v22_lead < 10;
          float v27_data = v26_g ? (glb_m1[v22_lead]) : (0.0f);
          float v28_data = s0[0];
          float v30_data = ir0[0];
          ir0[0] = (v30_data + (v27_data * v28_data));
          float v33_data = s0[17];
          float v35_data = ir0[1];
          ir0[1] = (v35_data + (v27_data * v33_data));
          float v38_data = s0[34];
          float v40_data = ir0[2];
          ir0[2] = (v40_data + (v27_data * v38_data));
          float v43_data = s0[51];
          float v45_data = ir0[3];
          ir0[3] = (v45_data + (v27_data * v43_data));
          float v48_data = s0[68];
          float v50_data = ir0[4];
          ir0[4] = (v50_data + (v27_data * v48_data));
          float v53_data = s0[85];
          float v55_data = ir0[5];
          ir0[5] = (v55_data + (v27_data * v53_data));
          float v58_data = s0[102];
          float v60_data = ir0[6];
          ir0[6] = (v60_data + (v27_data * v58_data));
          float v63_data = s0[119];
          float v65_data = ir0[7];
          ir0[7] = (v65_data + (v27_data * v63_data));
          float v68_data = s0[136];
          float v70_data = ir0[8];
          ir0[8] = (v70_data + (v27_data * v68_data));
          float v73_data = v26_g ? (glb_m1[(v22_lead + 10)]) : (0.0f);
          float v74_data = s0[1];
          float v76_data = ir0[0];
          ir0[0] = (v76_data + (v73_data * v74_data));
          float v79_data = s0[18];
          float v81_data = ir0[1];
          ir0[1] = (v81_data + (v73_data * v79_data));
          float v84_data = s0[35];
          float v86_data = ir0[2];
          ir0[2] = (v86_data + (v73_data * v84_data));
          float v89_data = s0[52];
          float v91_data = ir0[3];
          ir0[3] = (v91_data + (v73_data * v89_data));
          float v94_data = s0[69];
          float v96_data = ir0[4];
          ir0[4] = (v96_data + (v73_data * v94_data));
          float v99_data = s0[86];
          float v101_data = ir0[5];
          ir0[5] = (v101_data + (v73_data * v99_data));
          float v104_data = s0[103];
          float v106_data = ir0[6];
          ir0[6] = (v106_data + (v73_data * v104_data));
          float v109_data = s0[120];
          float v111_data = ir0[7];
          ir0[7] = (v111_data + (v73_data * v109_data));
          float v114_data = s0[137];
          float v116_data = ir0[8];
          ir0[8] = (v116_data + (v73_data * v114_data));
          float v119_data = v26_g ? (glb_m1[(v22_lead + 20)]) : (0.0f);
          float v120_data = s0[2];
          float v122_data = ir0[0];
          ir0[0] = (v122_data + (v119_data * v120_data));
          float v125_data = s0[19];
          float v127_data = ir0[1];
          ir0[1] = (v127_data + (v119_data * v125_data));
          float v130_data = s0[36];
          float v132_data = ir0[2];
          ir0[2] = (v132_data + (v119_data * v130_data));
          float v135_data = s0[53];
          float v137_data = ir0[3];
          ir0[3] = (v137_data + (v119_data * v135_data));
          float v140_data = s0[70];
          float v142_data = ir0[4];
          ir0[4] = (v142_data + (v119_data * v140_data));
          float v145_data = s0[87];
          float v147_data = ir0[5];
          ir0[5] = (v147_data + (v119_data * v145_data));
          float v150_data = s0[104];
          float v152_data = ir0[6];
          ir0[6] = (v152_data + (v119_data * v150_data));
          float v155_data = s0[121];
          float v157_data = ir0[7];
          ir0[7] = (v157_data + (v119_data * v155_data));
          float v160_data = s0[138];
          float v162_data = ir0[8];
          ir0[8] = (v162_data + (v119_data * v160_data));
          float v165_data = v26_g ? (glb_m1[(v22_lead + 30)]) : (0.0f);
          float v166_data = s0[3];
          float v168_data = ir0[0];
          ir0[0] = (v168_data + (v165_data * v166_data));
          float v171_data = s0[20];
          float v173_data = ir0[1];
          ir0[1] = (v173_data + (v165_data * v171_data));
          float v176_data = s0[37];
          float v178_data = ir0[2];
          ir0[2] = (v178_data + (v165_data * v176_data));
          float v181_data = s0[54];
          float v183_data = ir0[3];
          ir0[3] = (v183_data + (v165_data * v181_data));
          float v186_data = s0[71];
          float v188_data = ir0[4];
          ir0[4] = (v188_data + (v165_data * v186_data));
          float v191_data = s0[88];
          float v193_data = ir0[5];
          ir0[5] = (v193_data + (v165_data * v191_data));
          float v196_data = s0[105];
          float v198_data = ir0[6];
          ir0[6] = (v198_data + (v165_data * v196_data));
          float v201_data = s0[122];
          float v203_data = ir0[7];
          ir0[7] = (v203_data + (v165_data * v201_data));
          float v206_data = s0[139];
          float v208_data = ir0[8];
          ir0[8] = (v208_data + (v165_data * v206_data));
          float v211_data = v26_g ? (glb_m1[(v22_lead + 40)]) : (0.0f);
          float v212_data = s0[4];
          float v214_data = ir0[0];
          ir0[0] = (v214_data + (v211_data * v212_data));
          float v217_data = s0[21];
          float v219_data = ir0[1];
          ir0[1] = (v219_data + (v211_data * v217_data));
          float v222_data = s0[38];
          float v224_data = ir0[2];
          ir0[2] = (v224_data + (v211_data * v222_data));
          float v227_data = s0[55];
          float v229_data = ir0[3];
          ir0[3] = (v229_data + (v211_data * v227_data));
          float v232_data = s0[72];
          float v234_data = ir0[4];
          ir0[4] = (v234_data + (v211_data * v232_data));
          float v237_data = s0[89];
          float v239_data = ir0[5];
          ir0[5] = (v239_data + (v211_data * v237_data));
          float v242_data = s0[106];
          float v244_data = ir0[6];
          ir0[6] = (v244_data + (v211_data * v242_data));
          float v247_data = s0[123];
          float v249_data = ir0[7];
          ir0[7] = (v249_data + (v211_data * v247_data));
          float v252_data = s0[140];
          float v254_data = ir0[8];
          ir0[8] = (v254_data + (v211_data * v252_data));
          float v257_data = v26_g ? (glb_m1[(v22_lead + 50)]) : (0.0f);
          float v258_data = s0[5];
          float v260_data = ir0[0];
          ir0[0] = (v260_data + (v257_data * v258_data));
          float v263_data = s0[22];
          float v265_data = ir0[1];
          ir0[1] = (v265_data + (v257_data * v263_data));
          float v268_data = s0[39];
          float v270_data = ir0[2];
          ir0[2] = (v270_data + (v257_data * v268_data));
          float v273_data = s0[56];
          float v275_data = ir0[3];
          ir0[3] = (v275_data + (v257_data * v273_data));
          float v278_data = s0[73];
          float v280_data = ir0[4];
          ir0[4] = (v280_data + (v257_data * v278_data));
          float v283_data = s0[90];
          float v285_data = ir0[5];
          ir0[5] = (v285_data + (v257_data * v283_data));
          float v288_data = s0[107];
          float v290_data = ir0[6];
          ir0[6] = (v290_data + (v257_data * v288_data));
          float v293_data = s0[124];
          float v295_data = ir0[7];
          ir0[7] = (v295_data + (v257_data * v293_data));
          float v298_data = s0[141];
          float v300_data = ir0[8];
          ir0[8] = (v300_data + (v257_data * v298_data));
          float v303_data = v26_g ? (glb_m1[(v22_lead + 60)]) : (0.0f);
          float v304_data = s0[6];
          float v306_data = ir0[0];
          ir0[0] = (v306_data + (v303_data * v304_data));
          float v309_data = s0[23];
          float v311_data = ir0[1];
          ir0[1] = (v311_data + (v303_data * v309_data));
          float v314_data = s0[40];
          float v316_data = ir0[2];
          ir0[2] = (v316_data + (v303_data * v314_data));
          float v319_data = s0[57];
          float v321_data = ir0[3];
          ir0[3] = (v321_data + (v303_data * v319_data));
          float v324_data = s0[74];
          float v326_data = ir0[4];
          ir0[4] = (v326_data + (v303_data * v324_data));
          float v329_data = s0[91];
          float v331_data = ir0[5];
          ir0[5] = (v331_data + (v303_data * v329_data));
          float v334_data = s0[108];
          float v336_data = ir0[6];
          ir0[6] = (v336_data + (v303_data * v334_data));
          float v339_data = s0[125];
          float v341_data = ir0[7];
          ir0[7] = (v341_data + (v303_data * v339_data));
          float v344_data = s0[142];
          float v346_data = ir0[8];
          ir0[8] = (v346_data + (v303_data * v344_data));
          float v349_data = v26_g ? (glb_m1[(v22_lead + 70)]) : (0.0f);
          float v350_data = s0[7];
          float v352_data = ir0[0];
          ir0[0] = (v352_data + (v349_data * v350_data));
          float v355_data = s0[24];
          float v357_data = ir0[1];
          ir0[1] = (v357_data + (v349_data * v355_data));
          float v360_data = s0[41];
          float v362_data = ir0[2];
          ir0[2] = (v362_data + (v349_data * v360_data));
          float v365_data = s0[58];
          float v367_data = ir0[3];
          ir0[3] = (v367_data + (v349_data * v365_data));
          float v370_data = s0[75];
          float v372_data = ir0[4];
          ir0[4] = (v372_data + (v349_data * v370_data));
          float v375_data = s0[92];
          float v377_data = ir0[5];
          ir0[5] = (v377_data + (v349_data * v375_data));
          float v380_data = s0[109];
          float v382_data = ir0[6];
          ir0[6] = (v382_data + (v349_data * v380_data));
          float v385_data = s0[126];
          float v387_data = ir0[7];
          ir0[7] = (v387_data + (v349_data * v385_data));
          float v390_data = s0[143];
          float v392_data = ir0[8];
          ir0[8] = (v392_data + (v349_data * v390_data));
          float v395_data = v26_g ? (glb_m1[(v22_lead + 80)]) : (0.0f);
          float v396_data = s0[8];
          float v398_data = ir0[0];
          ir0[0] = (v398_data + (v395_data * v396_data));
          float v401_data = s0[25];
          float v403_data = ir0[1];
          ir0[1] = (v403_data + (v395_data * v401_data));
          float v406_data = s0[42];
          float v408_data = ir0[2];
          ir0[2] = (v408_data + (v395_data * v406_data));
          float v411_data = s0[59];
          float v413_data = ir0[3];
          ir0[3] = (v413_data + (v395_data * v411_data));
          float v416_data = s0[76];
          float v418_data = ir0[4];
          ir0[4] = (v418_data + (v395_data * v416_data));
          float v421_data = s0[93];
          float v423_data = ir0[5];
          ir0[5] = (v423_data + (v395_data * v421_data));
          float v426_data = s0[110];
          float v428_data = ir0[6];
          ir0[6] = (v428_data + (v395_data * v426_data));
          float v431_data = s0[127];
          float v433_data = ir0[7];
          ir0[7] = (v433_data + (v395_data * v431_data));
          float v436_data = s0[144];
          float v438_data = ir0[8];
          ir0[8] = (v438_data + (v395_data * v436_data));
          float v441_data = v26_g ? (glb_m1[(v22_lead + 90)]) : (0.0f);
          float v442_data = s0[9];
          float v444_data = ir0[0];
          ir0[0] = (v444_data + (v441_data * v442_data));
          float v447_data = s0[26];
          float v449_data = ir0[1];
          ir0[1] = (v449_data + (v441_data * v447_data));
          float v452_data = s0[43];
          float v454_data = ir0[2];
          ir0[2] = (v454_data + (v441_data * v452_data));
          float v457_data = s0[60];
          float v459_data = ir0[3];
          ir0[3] = (v459_data + (v441_data * v457_data));
          float v462_data = s0[77];
          float v464_data = ir0[4];
          ir0[4] = (v464_data + (v441_data * v462_data));
          float v467_data = s0[94];
          float v469_data = ir0[5];
          ir0[5] = (v469_data + (v441_data * v467_data));
          float v472_data = s0[111];
          float v474_data = ir0[6];
          ir0[6] = (v474_data + (v441_data * v472_data));
          float v477_data = s0[128];
          float v479_data = ir0[7];
          ir0[7] = (v479_data + (v441_data * v477_data));
          float v482_data = s0[145];
          float v484_data = ir0[8];
          ir0[8] = (v484_data + (v441_data * v482_data));
          float v487_data = v26_g ? (glb_m1[(v22_lead + 100)]) : (0.0f);
          float v488_data = s0[10];
          float v490_data = ir0[0];
          ir0[0] = (v490_data + (v487_data * v488_data));
          float v493_data = s0[27];
          float v495_data = ir0[1];
          ir0[1] = (v495_data + (v487_data * v493_data));
          float v498_data = s0[44];
          float v500_data = ir0[2];
          ir0[2] = (v500_data + (v487_data * v498_data));
          float v503_data = s0[61];
          float v505_data = ir0[3];
          ir0[3] = (v505_data + (v487_data * v503_data));
          float v508_data = s0[78];
          float v510_data = ir0[4];
          ir0[4] = (v510_data + (v487_data * v508_data));
          float v513_data = s0[95];
          float v515_data = ir0[5];
          ir0[5] = (v515_data + (v487_data * v513_data));
          float v518_data = s0[112];
          float v520_data = ir0[6];
          ir0[6] = (v520_data + (v487_data * v518_data));
          float v523_data = s0[129];
          float v525_data = ir0[7];
          ir0[7] = (v525_data + (v487_data * v523_data));
          float v528_data = s0[146];
          float v530_data = ir0[8];
          ir0[8] = (v530_data + (v487_data * v528_data));
          float v533_data = v26_g ? (glb_m1[(v22_lead + 110)]) : (0.0f);
          float v534_data = s0[11];
          float v536_data = ir0[0];
          ir0[0] = (v536_data + (v533_data * v534_data));
          float v539_data = s0[28];
          float v541_data = ir0[1];
          ir0[1] = (v541_data + (v533_data * v539_data));
          float v544_data = s0[45];
          float v546_data = ir0[2];
          ir0[2] = (v546_data + (v533_data * v544_data));
          float v549_data = s0[62];
          float v551_data = ir0[3];
          ir0[3] = (v551_data + (v533_data * v549_data));
          float v554_data = s0[79];
          float v556_data = ir0[4];
          ir0[4] = (v556_data + (v533_data * v554_data));
          float v559_data = s0[96];
          float v561_data = ir0[5];
          ir0[5] = (v561_data + (v533_data * v559_data));
          float v564_data = s0[113];
          float v566_data = ir0[6];
          ir0[6] = (v566_data + (v533_data * v564_data));
          float v569_data = s0[130];
          float v571_data = ir0[7];
          ir0[7] = (v571_data + (v533_data * v569_data));
          float v574_data = s0[147];
          float v576_data = ir0[8];
          ir0[8] = (v576_data + (v533_data * v574_data));
          float v579_data = v26_g ? (glb_m1[(v22_lead + 120)]) : (0.0f);
          float v580_data = s0[12];
          float v582_data = ir0[0];
          ir0[0] = (v582_data + (v579_data * v580_data));
          float v585_data = s0[29];
          float v587_data = ir0[1];
          ir0[1] = (v587_data + (v579_data * v585_data));
          float v590_data = s0[46];
          float v592_data = ir0[2];
          ir0[2] = (v592_data + (v579_data * v590_data));
          float v595_data = s0[63];
          float v597_data = ir0[3];
          ir0[3] = (v597_data + (v579_data * v595_data));
          float v600_data = s0[80];
          float v602_data = ir0[4];
          ir0[4] = (v602_data + (v579_data * v600_data));
          float v605_data = s0[97];
          float v607_data = ir0[5];
          ir0[5] = (v607_data + (v579_data * v605_data));
          float v610_data = s0[114];
          float v612_data = ir0[6];
          ir0[6] = (v612_data + (v579_data * v610_data));
          float v615_data = s0[131];
          float v617_data = ir0[7];
          ir0[7] = (v617_data + (v579_data * v615_data));
          float v620_data = s0[148];
          float v622_data = ir0[8];
          ir0[8] = (v622_data + (v579_data * v620_data));
          float v625_data = v26_g ? (glb_m1[(v22_lead + 130)]) : (0.0f);
          float v626_data = s0[13];
          float v628_data = ir0[0];
          ir0[0] = (v628_data + (v625_data * v626_data));
          float v631_data = s0[30];
          float v633_data = ir0[1];
          ir0[1] = (v633_data + (v625_data * v631_data));
          float v636_data = s0[47];
          float v638_data = ir0[2];
          ir0[2] = (v638_data + (v625_data * v636_data));
          float v641_data = s0[64];
          float v643_data = ir0[3];
          ir0[3] = (v643_data + (v625_data * v641_data));
          float v646_data = s0[81];
          float v648_data = ir0[4];
          ir0[4] = (v648_data + (v625_data * v646_data));
          float v651_data = s0[98];
          float v653_data = ir0[5];
          ir0[5] = (v653_data + (v625_data * v651_data));
          float v656_data = s0[115];
          float v658_data = ir0[6];
          ir0[6] = (v658_data + (v625_data * v656_data));
          float v661_data = s0[132];
          float v663_data = ir0[7];
          ir0[7] = (v663_data + (v625_data * v661_data));
          float v666_data = s0[149];
          float v668_data = ir0[8];
          ir0[8] = (v668_data + (v625_data * v666_data));
          float v671_data = v26_g ? (glb_m1[(v22_lead + 140)]) : (0.0f);
          float v672_data = s0[14];
          float v674_data = ir0[0];
          ir0[0] = (v674_data + (v671_data * v672_data));
          float v677_data = s0[31];
          float v679_data = ir0[1];
          ir0[1] = (v679_data + (v671_data * v677_data));
          float v682_data = s0[48];
          float v684_data = ir0[2];
          ir0[2] = (v684_data + (v671_data * v682_data));
          float v687_data = s0[65];
          float v689_data = ir0[3];
          ir0[3] = (v689_data + (v671_data * v687_data));
          float v692_data = s0[82];
          float v694_data = ir0[4];
          ir0[4] = (v694_data + (v671_data * v692_data));
          float v697_data = s0[99];
          float v699_data = ir0[5];
          ir0[5] = (v699_data + (v671_data * v697_data));
          float v702_data = s0[116];
          float v704_data = ir0[6];
          ir0[6] = (v704_data + (v671_data * v702_data));
          float v707_data = s0[133];
          float v709_data = ir0[7];
          ir0[7] = (v709_data + (v671_data * v707_data));
          float v712_data = s0[150];
          float v714_data = ir0[8];
          ir0[8] = (v714_data + (v671_data * v712_data));
          float v717_data = v26_g ? (glb_m1[(v22_lead + 150)]) : (0.0f);
          float v718_data = s0[15];
          float v720_data = ir0[0];
          ir0[0] = (v720_data + (v717_data * v718_data));
          float v723_data = s0[32];
          float v725_data = ir0[1];
          ir0[1] = (v725_data + (v717_data * v723_data));
          float v728_data = s0[49];
          float v730_data = ir0[2];
          ir0[2] = (v730_data + (v717_data * v728_data));
          float v733_data = s0[66];
          float v735_data = ir0[3];
          ir0[3] = (v735_data + (v717_data * v733_data));
          float v738_data = s0[83];
          float v740_data = ir0[4];
          ir0[4] = (v740_data + (v717_data * v738_data));
          float v743_data = s0[100];
          float v745_data = ir0[5];
          ir0[5] = (v745_data + (v717_data * v743_data));
          float v748_data = s0[117];
          float v750_data = ir0[6];
          ir0[6] = (v750_data + (v717_data * v748_data));
          float v753_data = s0[134];
          float v755_data = ir0[7];
          ir0[7] = (v755_data + (v717_data * v753_data));
          float v758_data = s0[151];
          float v760_data = ir0[8];
          ir0[8] = (v760_data + (v717_data * v758_data));
          float v763_data = v26_g ? (glb_m1[(v22_lead + 160)]) : (0.0f);
          float v764_data = s0[16];
          float v766_data = ir0[0];
          ir0[0] = (v766_data + (v763_data * v764_data));
          float v769_data = s0[33];
          float v771_data = ir0[1];
          ir0[1] = (v771_data + (v763_data * v769_data));
          float v774_data = s0[50];
          float v776_data = ir0[2];
          ir0[2] = (v776_data + (v763_data * v774_data));
          float v779_data = s0[67];
          float v781_data = ir0[3];
          ir0[3] = (v781_data + (v763_data * v779_data));
          float v784_data = s0[84];
          float v786_data = ir0[4];
          ir0[4] = (v786_data + (v763_data * v784_data));
          float v789_data = s0[101];
          float v791_data = ir0[5];
          ir0[5] = (v791_data + (v763_data * v789_data));
          float v794_data = s0[118];
          float v796_data = ir0[6];
          ir0[6] = (v796_data + (v763_data * v794_data));
          float v799_data = s0[135];
          float v801_data = ir0[7];
          ir0[7] = (v801_data + (v763_data * v799_data));
          float v804_data = s0[152];
          float v806_data = ir0[8];
          ir0[8] = (v806_data + (v763_data * v804_data));
          // r0 = ir0
          if (v22_lead < 10) {
            #pragma unroll
            for (int32_t v812_n1 = 0; v812_n1 < 9; ++v812_n1) {
              float v814_data = ir0[v812_n1];
              r0[v812_n1] = v814_data;
            }
          }
          // glb_m0 = store{r>g}(r0);
          if (v22_lead < 10) {
            #pragma unroll
            for (int32_t v819_i1 = 0; v819_i1 < 9; ++v819_i1) {
              float v821_data = r0[v819_i1];
              glb_m0[(v22_lead + (v819_i1 * 10))] = v821_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

