// === base name ===
kernel_e4b86df4a9e1fac8

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_e4b86df4a9e1fac8 = {{16, 8, 1}, 16, 10, 1, 8, 5632, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_e4b86df4a9e1fac8(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_e4b86df4a9e1fac8(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_e4b86df4a9e1fac8(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_e4b86df4a9e1fac8, block.x * block.y * block.z, 1408 * sizeof(float));
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
void launcher_kernel_e4b86df4a9e1fac8(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_e4b86df4a9e1fac8(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_e4b86df4a9e1fac8, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_e4b86df4a9e1fac8<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_e4b86df4a9e1fac8(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (10 active) x 8 per block = block 16x8x1, 5632 B shared, occupancy grid
    // operands:
    //   m0 10×9(10×9) {0..10}×{0..9} strided
    //   m1 16×20(10×17) {0..10}×{1..18} none
    //   m2 20×9(17×9) {1..18}×{0..9} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":10,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1408}],"shared_bytes":5632,"shared_elements":1408,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[10,9]],"name":"m0","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,1],[10,18]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[1,0],[18,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[10,18]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[18,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[176 * threadIdx.y + 0];
      const float *const __restrict__ glb_m1 = &m1[0];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < numElements0; v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 90 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v9_batchId0 * 153 + 0 + m2_extraOffset];
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
          // ir0 = +(glb_m1 * s0)
          // [(0, 10), (0, 9)] [(1, 18)]
          float ir0[9]{};
          int32_t v25_lead = threadIdx.x % 16;
          bool v29_g = v25_lead < 10;
          float v30_data = v29_g ? (glb_m1[v25_lead]) : (0.0f);
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v31_data = s0[0];
          float v33_data = ir0[0];
          ir0[0] = (v33_data + (v30_data * v31_data));
          float v36_data = s0[17];
          float v38_data = ir0[1];
          ir0[1] = (v38_data + (v30_data * v36_data));
          float v41_data = s0[34];
          float v43_data = ir0[2];
          ir0[2] = (v43_data + (v30_data * v41_data));
          float v46_data = s0[51];
          float v48_data = ir0[3];
          ir0[3] = (v48_data + (v30_data * v46_data));
          float v51_data = s0[68];
          float v53_data = ir0[4];
          ir0[4] = (v53_data + (v30_data * v51_data));
          float v56_data = s0[85];
          float v58_data = ir0[5];
          ir0[5] = (v58_data + (v30_data * v56_data));
          float v61_data = s0[102];
          float v63_data = ir0[6];
          ir0[6] = (v63_data + (v30_data * v61_data));
          float v66_data = s0[119];
          float v68_data = ir0[7];
          ir0[7] = (v68_data + (v30_data * v66_data));
          float v71_data = s0[136];
          float v73_data = ir0[8];
          ir0[8] = (v73_data + (v30_data * v71_data));
          float v76_data = v29_g ? (glb_m1[(v25_lead + 10)]) : (0.0f);
          float v77_data = s0[1];
          float v79_data = ir0[0];
          ir0[0] = (v79_data + (v76_data * v77_data));
          float v82_data = s0[18];
          float v84_data = ir0[1];
          ir0[1] = (v84_data + (v76_data * v82_data));
          float v87_data = s0[35];
          float v89_data = ir0[2];
          ir0[2] = (v89_data + (v76_data * v87_data));
          float v92_data = s0[52];
          float v94_data = ir0[3];
          ir0[3] = (v94_data + (v76_data * v92_data));
          float v97_data = s0[69];
          float v99_data = ir0[4];
          ir0[4] = (v99_data + (v76_data * v97_data));
          float v102_data = s0[86];
          float v104_data = ir0[5];
          ir0[5] = (v104_data + (v76_data * v102_data));
          float v107_data = s0[103];
          float v109_data = ir0[6];
          ir0[6] = (v109_data + (v76_data * v107_data));
          float v112_data = s0[120];
          float v114_data = ir0[7];
          ir0[7] = (v114_data + (v76_data * v112_data));
          float v117_data = s0[137];
          float v119_data = ir0[8];
          ir0[8] = (v119_data + (v76_data * v117_data));
          float v122_data = v29_g ? (glb_m1[(v25_lead + 20)]) : (0.0f);
          float v123_data = s0[2];
          float v125_data = ir0[0];
          ir0[0] = (v125_data + (v122_data * v123_data));
          float v128_data = s0[19];
          float v130_data = ir0[1];
          ir0[1] = (v130_data + (v122_data * v128_data));
          float v133_data = s0[36];
          float v135_data = ir0[2];
          ir0[2] = (v135_data + (v122_data * v133_data));
          float v138_data = s0[53];
          float v140_data = ir0[3];
          ir0[3] = (v140_data + (v122_data * v138_data));
          float v143_data = s0[70];
          float v145_data = ir0[4];
          ir0[4] = (v145_data + (v122_data * v143_data));
          float v148_data = s0[87];
          float v150_data = ir0[5];
          ir0[5] = (v150_data + (v122_data * v148_data));
          float v153_data = s0[104];
          float v155_data = ir0[6];
          ir0[6] = (v155_data + (v122_data * v153_data));
          float v158_data = s0[121];
          float v160_data = ir0[7];
          ir0[7] = (v160_data + (v122_data * v158_data));
          float v163_data = s0[138];
          float v165_data = ir0[8];
          ir0[8] = (v165_data + (v122_data * v163_data));
          float v168_data = v29_g ? (glb_m1[(v25_lead + 30)]) : (0.0f);
          float v169_data = s0[3];
          float v171_data = ir0[0];
          ir0[0] = (v171_data + (v168_data * v169_data));
          float v174_data = s0[20];
          float v176_data = ir0[1];
          ir0[1] = (v176_data + (v168_data * v174_data));
          float v179_data = s0[37];
          float v181_data = ir0[2];
          ir0[2] = (v181_data + (v168_data * v179_data));
          float v184_data = s0[54];
          float v186_data = ir0[3];
          ir0[3] = (v186_data + (v168_data * v184_data));
          float v189_data = s0[71];
          float v191_data = ir0[4];
          ir0[4] = (v191_data + (v168_data * v189_data));
          float v194_data = s0[88];
          float v196_data = ir0[5];
          ir0[5] = (v196_data + (v168_data * v194_data));
          float v199_data = s0[105];
          float v201_data = ir0[6];
          ir0[6] = (v201_data + (v168_data * v199_data));
          float v204_data = s0[122];
          float v206_data = ir0[7];
          ir0[7] = (v206_data + (v168_data * v204_data));
          float v209_data = s0[139];
          float v211_data = ir0[8];
          ir0[8] = (v211_data + (v168_data * v209_data));
          float v214_data = v29_g ? (glb_m1[(v25_lead + 40)]) : (0.0f);
          float v215_data = s0[4];
          float v217_data = ir0[0];
          ir0[0] = (v217_data + (v214_data * v215_data));
          float v220_data = s0[21];
          float v222_data = ir0[1];
          ir0[1] = (v222_data + (v214_data * v220_data));
          float v225_data = s0[38];
          float v227_data = ir0[2];
          ir0[2] = (v227_data + (v214_data * v225_data));
          float v230_data = s0[55];
          float v232_data = ir0[3];
          ir0[3] = (v232_data + (v214_data * v230_data));
          float v235_data = s0[72];
          float v237_data = ir0[4];
          ir0[4] = (v237_data + (v214_data * v235_data));
          float v240_data = s0[89];
          float v242_data = ir0[5];
          ir0[5] = (v242_data + (v214_data * v240_data));
          float v245_data = s0[106];
          float v247_data = ir0[6];
          ir0[6] = (v247_data + (v214_data * v245_data));
          float v250_data = s0[123];
          float v252_data = ir0[7];
          ir0[7] = (v252_data + (v214_data * v250_data));
          float v255_data = s0[140];
          float v257_data = ir0[8];
          ir0[8] = (v257_data + (v214_data * v255_data));
          float v260_data = v29_g ? (glb_m1[(v25_lead + 50)]) : (0.0f);
          float v261_data = s0[5];
          float v263_data = ir0[0];
          ir0[0] = (v263_data + (v260_data * v261_data));
          float v266_data = s0[22];
          float v268_data = ir0[1];
          ir0[1] = (v268_data + (v260_data * v266_data));
          float v271_data = s0[39];
          float v273_data = ir0[2];
          ir0[2] = (v273_data + (v260_data * v271_data));
          float v276_data = s0[56];
          float v278_data = ir0[3];
          ir0[3] = (v278_data + (v260_data * v276_data));
          float v281_data = s0[73];
          float v283_data = ir0[4];
          ir0[4] = (v283_data + (v260_data * v281_data));
          float v286_data = s0[90];
          float v288_data = ir0[5];
          ir0[5] = (v288_data + (v260_data * v286_data));
          float v291_data = s0[107];
          float v293_data = ir0[6];
          ir0[6] = (v293_data + (v260_data * v291_data));
          float v296_data = s0[124];
          float v298_data = ir0[7];
          ir0[7] = (v298_data + (v260_data * v296_data));
          float v301_data = s0[141];
          float v303_data = ir0[8];
          ir0[8] = (v303_data + (v260_data * v301_data));
          float v306_data = v29_g ? (glb_m1[(v25_lead + 60)]) : (0.0f);
          float v307_data = s0[6];
          float v309_data = ir0[0];
          ir0[0] = (v309_data + (v306_data * v307_data));
          float v312_data = s0[23];
          float v314_data = ir0[1];
          ir0[1] = (v314_data + (v306_data * v312_data));
          float v317_data = s0[40];
          float v319_data = ir0[2];
          ir0[2] = (v319_data + (v306_data * v317_data));
          float v322_data = s0[57];
          float v324_data = ir0[3];
          ir0[3] = (v324_data + (v306_data * v322_data));
          float v327_data = s0[74];
          float v329_data = ir0[4];
          ir0[4] = (v329_data + (v306_data * v327_data));
          float v332_data = s0[91];
          float v334_data = ir0[5];
          ir0[5] = (v334_data + (v306_data * v332_data));
          float v337_data = s0[108];
          float v339_data = ir0[6];
          ir0[6] = (v339_data + (v306_data * v337_data));
          float v342_data = s0[125];
          float v344_data = ir0[7];
          ir0[7] = (v344_data + (v306_data * v342_data));
          float v347_data = s0[142];
          float v349_data = ir0[8];
          ir0[8] = (v349_data + (v306_data * v347_data));
          float v352_data = v29_g ? (glb_m1[(v25_lead + 70)]) : (0.0f);
          float v353_data = s0[7];
          float v355_data = ir0[0];
          ir0[0] = (v355_data + (v352_data * v353_data));
          float v358_data = s0[24];
          float v360_data = ir0[1];
          ir0[1] = (v360_data + (v352_data * v358_data));
          float v363_data = s0[41];
          float v365_data = ir0[2];
          ir0[2] = (v365_data + (v352_data * v363_data));
          float v368_data = s0[58];
          float v370_data = ir0[3];
          ir0[3] = (v370_data + (v352_data * v368_data));
          float v373_data = s0[75];
          float v375_data = ir0[4];
          ir0[4] = (v375_data + (v352_data * v373_data));
          float v378_data = s0[92];
          float v380_data = ir0[5];
          ir0[5] = (v380_data + (v352_data * v378_data));
          float v383_data = s0[109];
          float v385_data = ir0[6];
          ir0[6] = (v385_data + (v352_data * v383_data));
          float v388_data = s0[126];
          float v390_data = ir0[7];
          ir0[7] = (v390_data + (v352_data * v388_data));
          float v393_data = s0[143];
          float v395_data = ir0[8];
          ir0[8] = (v395_data + (v352_data * v393_data));
          float v398_data = v29_g ? (glb_m1[(v25_lead + 80)]) : (0.0f);
          float v399_data = s0[8];
          float v401_data = ir0[0];
          ir0[0] = (v401_data + (v398_data * v399_data));
          float v404_data = s0[25];
          float v406_data = ir0[1];
          ir0[1] = (v406_data + (v398_data * v404_data));
          float v409_data = s0[42];
          float v411_data = ir0[2];
          ir0[2] = (v411_data + (v398_data * v409_data));
          float v414_data = s0[59];
          float v416_data = ir0[3];
          ir0[3] = (v416_data + (v398_data * v414_data));
          float v419_data = s0[76];
          float v421_data = ir0[4];
          ir0[4] = (v421_data + (v398_data * v419_data));
          float v424_data = s0[93];
          float v426_data = ir0[5];
          ir0[5] = (v426_data + (v398_data * v424_data));
          float v429_data = s0[110];
          float v431_data = ir0[6];
          ir0[6] = (v431_data + (v398_data * v429_data));
          float v434_data = s0[127];
          float v436_data = ir0[7];
          ir0[7] = (v436_data + (v398_data * v434_data));
          float v439_data = s0[144];
          float v441_data = ir0[8];
          ir0[8] = (v441_data + (v398_data * v439_data));
          float v444_data = v29_g ? (glb_m1[(v25_lead + 90)]) : (0.0f);
          float v445_data = s0[9];
          float v447_data = ir0[0];
          ir0[0] = (v447_data + (v444_data * v445_data));
          float v450_data = s0[26];
          float v452_data = ir0[1];
          ir0[1] = (v452_data + (v444_data * v450_data));
          float v455_data = s0[43];
          float v457_data = ir0[2];
          ir0[2] = (v457_data + (v444_data * v455_data));
          float v460_data = s0[60];
          float v462_data = ir0[3];
          ir0[3] = (v462_data + (v444_data * v460_data));
          float v465_data = s0[77];
          float v467_data = ir0[4];
          ir0[4] = (v467_data + (v444_data * v465_data));
          float v470_data = s0[94];
          float v472_data = ir0[5];
          ir0[5] = (v472_data + (v444_data * v470_data));
          float v475_data = s0[111];
          float v477_data = ir0[6];
          ir0[6] = (v477_data + (v444_data * v475_data));
          float v480_data = s0[128];
          float v482_data = ir0[7];
          ir0[7] = (v482_data + (v444_data * v480_data));
          float v485_data = s0[145];
          float v487_data = ir0[8];
          ir0[8] = (v487_data + (v444_data * v485_data));
          float v490_data = v29_g ? (glb_m1[(v25_lead + 100)]) : (0.0f);
          float v491_data = s0[10];
          float v493_data = ir0[0];
          ir0[0] = (v493_data + (v490_data * v491_data));
          float v496_data = s0[27];
          float v498_data = ir0[1];
          ir0[1] = (v498_data + (v490_data * v496_data));
          float v501_data = s0[44];
          float v503_data = ir0[2];
          ir0[2] = (v503_data + (v490_data * v501_data));
          float v506_data = s0[61];
          float v508_data = ir0[3];
          ir0[3] = (v508_data + (v490_data * v506_data));
          float v511_data = s0[78];
          float v513_data = ir0[4];
          ir0[4] = (v513_data + (v490_data * v511_data));
          float v516_data = s0[95];
          float v518_data = ir0[5];
          ir0[5] = (v518_data + (v490_data * v516_data));
          float v521_data = s0[112];
          float v523_data = ir0[6];
          ir0[6] = (v523_data + (v490_data * v521_data));
          float v526_data = s0[129];
          float v528_data = ir0[7];
          ir0[7] = (v528_data + (v490_data * v526_data));
          float v531_data = s0[146];
          float v533_data = ir0[8];
          ir0[8] = (v533_data + (v490_data * v531_data));
          float v536_data = v29_g ? (glb_m1[(v25_lead + 110)]) : (0.0f);
          float v537_data = s0[11];
          float v539_data = ir0[0];
          ir0[0] = (v539_data + (v536_data * v537_data));
          float v542_data = s0[28];
          float v544_data = ir0[1];
          ir0[1] = (v544_data + (v536_data * v542_data));
          float v547_data = s0[45];
          float v549_data = ir0[2];
          ir0[2] = (v549_data + (v536_data * v547_data));
          float v552_data = s0[62];
          float v554_data = ir0[3];
          ir0[3] = (v554_data + (v536_data * v552_data));
          float v557_data = s0[79];
          float v559_data = ir0[4];
          ir0[4] = (v559_data + (v536_data * v557_data));
          float v562_data = s0[96];
          float v564_data = ir0[5];
          ir0[5] = (v564_data + (v536_data * v562_data));
          float v567_data = s0[113];
          float v569_data = ir0[6];
          ir0[6] = (v569_data + (v536_data * v567_data));
          float v572_data = s0[130];
          float v574_data = ir0[7];
          ir0[7] = (v574_data + (v536_data * v572_data));
          float v577_data = s0[147];
          float v579_data = ir0[8];
          ir0[8] = (v579_data + (v536_data * v577_data));
          float v582_data = v29_g ? (glb_m1[(v25_lead + 120)]) : (0.0f);
          float v583_data = s0[12];
          float v585_data = ir0[0];
          ir0[0] = (v585_data + (v582_data * v583_data));
          float v588_data = s0[29];
          float v590_data = ir0[1];
          ir0[1] = (v590_data + (v582_data * v588_data));
          float v593_data = s0[46];
          float v595_data = ir0[2];
          ir0[2] = (v595_data + (v582_data * v593_data));
          float v598_data = s0[63];
          float v600_data = ir0[3];
          ir0[3] = (v600_data + (v582_data * v598_data));
          float v603_data = s0[80];
          float v605_data = ir0[4];
          ir0[4] = (v605_data + (v582_data * v603_data));
          float v608_data = s0[97];
          float v610_data = ir0[5];
          ir0[5] = (v610_data + (v582_data * v608_data));
          float v613_data = s0[114];
          float v615_data = ir0[6];
          ir0[6] = (v615_data + (v582_data * v613_data));
          float v618_data = s0[131];
          float v620_data = ir0[7];
          ir0[7] = (v620_data + (v582_data * v618_data));
          float v623_data = s0[148];
          float v625_data = ir0[8];
          ir0[8] = (v625_data + (v582_data * v623_data));
          float v628_data = v29_g ? (glb_m1[(v25_lead + 130)]) : (0.0f);
          float v629_data = s0[13];
          float v631_data = ir0[0];
          ir0[0] = (v631_data + (v628_data * v629_data));
          float v634_data = s0[30];
          float v636_data = ir0[1];
          ir0[1] = (v636_data + (v628_data * v634_data));
          float v639_data = s0[47];
          float v641_data = ir0[2];
          ir0[2] = (v641_data + (v628_data * v639_data));
          float v644_data = s0[64];
          float v646_data = ir0[3];
          ir0[3] = (v646_data + (v628_data * v644_data));
          float v649_data = s0[81];
          float v651_data = ir0[4];
          ir0[4] = (v651_data + (v628_data * v649_data));
          float v654_data = s0[98];
          float v656_data = ir0[5];
          ir0[5] = (v656_data + (v628_data * v654_data));
          float v659_data = s0[115];
          float v661_data = ir0[6];
          ir0[6] = (v661_data + (v628_data * v659_data));
          float v664_data = s0[132];
          float v666_data = ir0[7];
          ir0[7] = (v666_data + (v628_data * v664_data));
          float v669_data = s0[149];
          float v671_data = ir0[8];
          ir0[8] = (v671_data + (v628_data * v669_data));
          float v674_data = v29_g ? (glb_m1[(v25_lead + 140)]) : (0.0f);
          float v675_data = s0[14];
          float v677_data = ir0[0];
          ir0[0] = (v677_data + (v674_data * v675_data));
          float v680_data = s0[31];
          float v682_data = ir0[1];
          ir0[1] = (v682_data + (v674_data * v680_data));
          float v685_data = s0[48];
          float v687_data = ir0[2];
          ir0[2] = (v687_data + (v674_data * v685_data));
          float v690_data = s0[65];
          float v692_data = ir0[3];
          ir0[3] = (v692_data + (v674_data * v690_data));
          float v695_data = s0[82];
          float v697_data = ir0[4];
          ir0[4] = (v697_data + (v674_data * v695_data));
          float v700_data = s0[99];
          float v702_data = ir0[5];
          ir0[5] = (v702_data + (v674_data * v700_data));
          float v705_data = s0[116];
          float v707_data = ir0[6];
          ir0[6] = (v707_data + (v674_data * v705_data));
          float v710_data = s0[133];
          float v712_data = ir0[7];
          ir0[7] = (v712_data + (v674_data * v710_data));
          float v715_data = s0[150];
          float v717_data = ir0[8];
          ir0[8] = (v717_data + (v674_data * v715_data));
          float v720_data = v29_g ? (glb_m1[(v25_lead + 150)]) : (0.0f);
          float v721_data = s0[15];
          float v723_data = ir0[0];
          ir0[0] = (v723_data + (v720_data * v721_data));
          float v726_data = s0[32];
          float v728_data = ir0[1];
          ir0[1] = (v728_data + (v720_data * v726_data));
          float v731_data = s0[49];
          float v733_data = ir0[2];
          ir0[2] = (v733_data + (v720_data * v731_data));
          float v736_data = s0[66];
          float v738_data = ir0[3];
          ir0[3] = (v738_data + (v720_data * v736_data));
          float v741_data = s0[83];
          float v743_data = ir0[4];
          ir0[4] = (v743_data + (v720_data * v741_data));
          float v746_data = s0[100];
          float v748_data = ir0[5];
          ir0[5] = (v748_data + (v720_data * v746_data));
          float v751_data = s0[117];
          float v753_data = ir0[6];
          ir0[6] = (v753_data + (v720_data * v751_data));
          float v756_data = s0[134];
          float v758_data = ir0[7];
          ir0[7] = (v758_data + (v720_data * v756_data));
          float v761_data = s0[151];
          float v763_data = ir0[8];
          ir0[8] = (v763_data + (v720_data * v761_data));
          float v766_data = v29_g ? (glb_m1[(v25_lead + 160)]) : (0.0f);
          float v767_data = s0[16];
          float v769_data = ir0[0];
          ir0[0] = (v769_data + (v766_data * v767_data));
          float v772_data = s0[33];
          float v774_data = ir0[1];
          ir0[1] = (v774_data + (v766_data * v772_data));
          float v777_data = s0[50];
          float v779_data = ir0[2];
          ir0[2] = (v779_data + (v766_data * v777_data));
          float v782_data = s0[67];
          float v784_data = ir0[3];
          ir0[3] = (v784_data + (v766_data * v782_data));
          float v787_data = s0[84];
          float v789_data = ir0[4];
          ir0[4] = (v789_data + (v766_data * v787_data));
          float v792_data = s0[101];
          float v794_data = ir0[5];
          ir0[5] = (v794_data + (v766_data * v792_data));
          float v797_data = s0[118];
          float v799_data = ir0[6];
          ir0[6] = (v799_data + (v766_data * v797_data));
          float v802_data = s0[135];
          float v804_data = ir0[7];
          ir0[7] = (v804_data + (v766_data * v802_data));
          float v807_data = s0[152];
          float v809_data = ir0[8];
          ir0[8] = (v809_data + (v766_data * v807_data));
          // r0 = ir0
          if (v25_lead < 10) {
            #pragma unroll
            for (int32_t v815_n1 = 0; v815_n1 < 9; ++v815_n1) {
              float v817_data = ir0[v815_n1];
              r0[v815_n1] = v817_data;
            }
          }
          // glb_m0 = store{r>g}(r0);
          if (v25_lead < 10) {
            #pragma unroll
            for (int32_t v822_i1 = 0; v822_i1 < 9; ++v822_i1) {
              float v824_data = r0[v822_i1];
              glb_m0[(v25_lead + (v822_i1 * 10))] = v824_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

