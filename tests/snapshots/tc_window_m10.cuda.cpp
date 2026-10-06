// === base name ===
kernel_ffb3242fd90bf27b

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_ffb3242fd90bf27b = {{16, 8, 1}, 16, 10, 1, 8, 5632, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_ffb3242fd90bf27b(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_ffb3242fd90bf27b(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_ffb3242fd90bf27b(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_ffb3242fd90bf27b, block.x * block.y * block.z, 1408 * sizeof(float));
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
void launcher_kernel_ffb3242fd90bf27b(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_ffb3242fd90bf27b(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_ffb3242fd90bf27b, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_ffb3242fd90bf27b<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_ffb3242fd90bf27b(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
      float* tempShrMem = &localShrMem0[160];
      const float *const __restrict__ glb_m1 = &m1[0];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v12_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v12_batchId0 < numElements0; v12_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v13_ahead1 = v12_batchId0 + (gridDim.x * blockDim.y);
        size_t v15_batchId1 = (v13_ahead1 < numElements0) ? v13_ahead1 : v12_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v12_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v12_batchId0 * 90 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v12_batchId0 * 153 + 0 + m2_extraOffset];
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
          int32_t v28_lead = threadIdx.x % 16;
          bool v32_g = v28_lead < 10;
          float v33_data = v32_g ? (glb_m1[v28_lead]) : (0.0f);
          float v34_data = s0[0];
          float v36_data = ir0[0];
          ir0[0] = (v36_data + (v33_data * v34_data));
          float v39_data = s0[17];
          float v41_data = ir0[1];
          ir0[1] = (v41_data + (v33_data * v39_data));
          float v44_data = s0[34];
          float v46_data = ir0[2];
          ir0[2] = (v46_data + (v33_data * v44_data));
          float v49_data = s0[51];
          float v51_data = ir0[3];
          ir0[3] = (v51_data + (v33_data * v49_data));
          float v54_data = s0[68];
          float v56_data = ir0[4];
          ir0[4] = (v56_data + (v33_data * v54_data));
          float v59_data = s0[85];
          float v61_data = ir0[5];
          ir0[5] = (v61_data + (v33_data * v59_data));
          float v64_data = s0[102];
          float v66_data = ir0[6];
          ir0[6] = (v66_data + (v33_data * v64_data));
          float v69_data = s0[119];
          float v71_data = ir0[7];
          ir0[7] = (v71_data + (v33_data * v69_data));
          float v74_data = s0[136];
          float v76_data = ir0[8];
          ir0[8] = (v76_data + (v33_data * v74_data));
          float v79_data = v32_g ? (glb_m1[(v28_lead + 10)]) : (0.0f);
          float v80_data = s0[1];
          float v82_data = ir0[0];
          ir0[0] = (v82_data + (v79_data * v80_data));
          float v85_data = s0[18];
          float v87_data = ir0[1];
          ir0[1] = (v87_data + (v79_data * v85_data));
          float v90_data = s0[35];
          float v92_data = ir0[2];
          ir0[2] = (v92_data + (v79_data * v90_data));
          float v95_data = s0[52];
          float v97_data = ir0[3];
          ir0[3] = (v97_data + (v79_data * v95_data));
          float v100_data = s0[69];
          float v102_data = ir0[4];
          ir0[4] = (v102_data + (v79_data * v100_data));
          float v105_data = s0[86];
          float v107_data = ir0[5];
          ir0[5] = (v107_data + (v79_data * v105_data));
          float v110_data = s0[103];
          float v112_data = ir0[6];
          ir0[6] = (v112_data + (v79_data * v110_data));
          float v115_data = s0[120];
          float v117_data = ir0[7];
          ir0[7] = (v117_data + (v79_data * v115_data));
          float v120_data = s0[137];
          float v122_data = ir0[8];
          ir0[8] = (v122_data + (v79_data * v120_data));
          float v125_data = v32_g ? (glb_m1[(v28_lead + 20)]) : (0.0f);
          float v126_data = s0[2];
          float v128_data = ir0[0];
          ir0[0] = (v128_data + (v125_data * v126_data));
          float v131_data = s0[19];
          float v133_data = ir0[1];
          ir0[1] = (v133_data + (v125_data * v131_data));
          float v136_data = s0[36];
          float v138_data = ir0[2];
          ir0[2] = (v138_data + (v125_data * v136_data));
          float v141_data = s0[53];
          float v143_data = ir0[3];
          ir0[3] = (v143_data + (v125_data * v141_data));
          float v146_data = s0[70];
          float v148_data = ir0[4];
          ir0[4] = (v148_data + (v125_data * v146_data));
          float v151_data = s0[87];
          float v153_data = ir0[5];
          ir0[5] = (v153_data + (v125_data * v151_data));
          float v156_data = s0[104];
          float v158_data = ir0[6];
          ir0[6] = (v158_data + (v125_data * v156_data));
          float v161_data = s0[121];
          float v163_data = ir0[7];
          ir0[7] = (v163_data + (v125_data * v161_data));
          float v166_data = s0[138];
          float v168_data = ir0[8];
          ir0[8] = (v168_data + (v125_data * v166_data));
          float v171_data = v32_g ? (glb_m1[(v28_lead + 30)]) : (0.0f);
          float v172_data = s0[3];
          float v174_data = ir0[0];
          ir0[0] = (v174_data + (v171_data * v172_data));
          float v177_data = s0[20];
          float v179_data = ir0[1];
          ir0[1] = (v179_data + (v171_data * v177_data));
          float v182_data = s0[37];
          float v184_data = ir0[2];
          ir0[2] = (v184_data + (v171_data * v182_data));
          float v187_data = s0[54];
          float v189_data = ir0[3];
          ir0[3] = (v189_data + (v171_data * v187_data));
          float v192_data = s0[71];
          float v194_data = ir0[4];
          ir0[4] = (v194_data + (v171_data * v192_data));
          float v197_data = s0[88];
          float v199_data = ir0[5];
          ir0[5] = (v199_data + (v171_data * v197_data));
          float v202_data = s0[105];
          float v204_data = ir0[6];
          ir0[6] = (v204_data + (v171_data * v202_data));
          float v207_data = s0[122];
          float v209_data = ir0[7];
          ir0[7] = (v209_data + (v171_data * v207_data));
          float v212_data = s0[139];
          float v214_data = ir0[8];
          ir0[8] = (v214_data + (v171_data * v212_data));
          float v217_data = v32_g ? (glb_m1[(v28_lead + 40)]) : (0.0f);
          float v218_data = s0[4];
          float v220_data = ir0[0];
          ir0[0] = (v220_data + (v217_data * v218_data));
          float v223_data = s0[21];
          float v225_data = ir0[1];
          ir0[1] = (v225_data + (v217_data * v223_data));
          float v228_data = s0[38];
          float v230_data = ir0[2];
          ir0[2] = (v230_data + (v217_data * v228_data));
          float v233_data = s0[55];
          float v235_data = ir0[3];
          ir0[3] = (v235_data + (v217_data * v233_data));
          float v238_data = s0[72];
          float v240_data = ir0[4];
          ir0[4] = (v240_data + (v217_data * v238_data));
          float v243_data = s0[89];
          float v245_data = ir0[5];
          ir0[5] = (v245_data + (v217_data * v243_data));
          float v248_data = s0[106];
          float v250_data = ir0[6];
          ir0[6] = (v250_data + (v217_data * v248_data));
          float v253_data = s0[123];
          float v255_data = ir0[7];
          ir0[7] = (v255_data + (v217_data * v253_data));
          float v258_data = s0[140];
          float v260_data = ir0[8];
          ir0[8] = (v260_data + (v217_data * v258_data));
          float v263_data = v32_g ? (glb_m1[(v28_lead + 50)]) : (0.0f);
          float v264_data = s0[5];
          float v266_data = ir0[0];
          ir0[0] = (v266_data + (v263_data * v264_data));
          float v269_data = s0[22];
          float v271_data = ir0[1];
          ir0[1] = (v271_data + (v263_data * v269_data));
          float v274_data = s0[39];
          float v276_data = ir0[2];
          ir0[2] = (v276_data + (v263_data * v274_data));
          float v279_data = s0[56];
          float v281_data = ir0[3];
          ir0[3] = (v281_data + (v263_data * v279_data));
          float v284_data = s0[73];
          float v286_data = ir0[4];
          ir0[4] = (v286_data + (v263_data * v284_data));
          float v289_data = s0[90];
          float v291_data = ir0[5];
          ir0[5] = (v291_data + (v263_data * v289_data));
          float v294_data = s0[107];
          float v296_data = ir0[6];
          ir0[6] = (v296_data + (v263_data * v294_data));
          float v299_data = s0[124];
          float v301_data = ir0[7];
          ir0[7] = (v301_data + (v263_data * v299_data));
          float v304_data = s0[141];
          float v306_data = ir0[8];
          ir0[8] = (v306_data + (v263_data * v304_data));
          float v309_data = v32_g ? (glb_m1[(v28_lead + 60)]) : (0.0f);
          float v310_data = s0[6];
          float v312_data = ir0[0];
          ir0[0] = (v312_data + (v309_data * v310_data));
          float v315_data = s0[23];
          float v317_data = ir0[1];
          ir0[1] = (v317_data + (v309_data * v315_data));
          float v320_data = s0[40];
          float v322_data = ir0[2];
          ir0[2] = (v322_data + (v309_data * v320_data));
          float v325_data = s0[57];
          float v327_data = ir0[3];
          ir0[3] = (v327_data + (v309_data * v325_data));
          float v330_data = s0[74];
          float v332_data = ir0[4];
          ir0[4] = (v332_data + (v309_data * v330_data));
          float v335_data = s0[91];
          float v337_data = ir0[5];
          ir0[5] = (v337_data + (v309_data * v335_data));
          float v340_data = s0[108];
          float v342_data = ir0[6];
          ir0[6] = (v342_data + (v309_data * v340_data));
          float v345_data = s0[125];
          float v347_data = ir0[7];
          ir0[7] = (v347_data + (v309_data * v345_data));
          float v350_data = s0[142];
          float v352_data = ir0[8];
          ir0[8] = (v352_data + (v309_data * v350_data));
          float v355_data = v32_g ? (glb_m1[(v28_lead + 70)]) : (0.0f);
          float v356_data = s0[7];
          float v358_data = ir0[0];
          ir0[0] = (v358_data + (v355_data * v356_data));
          float v361_data = s0[24];
          float v363_data = ir0[1];
          ir0[1] = (v363_data + (v355_data * v361_data));
          float v366_data = s0[41];
          float v368_data = ir0[2];
          ir0[2] = (v368_data + (v355_data * v366_data));
          float v371_data = s0[58];
          float v373_data = ir0[3];
          ir0[3] = (v373_data + (v355_data * v371_data));
          float v376_data = s0[75];
          float v378_data = ir0[4];
          ir0[4] = (v378_data + (v355_data * v376_data));
          float v381_data = s0[92];
          float v383_data = ir0[5];
          ir0[5] = (v383_data + (v355_data * v381_data));
          float v386_data = s0[109];
          float v388_data = ir0[6];
          ir0[6] = (v388_data + (v355_data * v386_data));
          float v391_data = s0[126];
          float v393_data = ir0[7];
          ir0[7] = (v393_data + (v355_data * v391_data));
          float v396_data = s0[143];
          float v398_data = ir0[8];
          ir0[8] = (v398_data + (v355_data * v396_data));
          float v401_data = v32_g ? (glb_m1[(v28_lead + 80)]) : (0.0f);
          float v402_data = s0[8];
          float v404_data = ir0[0];
          ir0[0] = (v404_data + (v401_data * v402_data));
          float v407_data = s0[25];
          float v409_data = ir0[1];
          ir0[1] = (v409_data + (v401_data * v407_data));
          float v412_data = s0[42];
          float v414_data = ir0[2];
          ir0[2] = (v414_data + (v401_data * v412_data));
          float v417_data = s0[59];
          float v419_data = ir0[3];
          ir0[3] = (v419_data + (v401_data * v417_data));
          float v422_data = s0[76];
          float v424_data = ir0[4];
          ir0[4] = (v424_data + (v401_data * v422_data));
          float v427_data = s0[93];
          float v429_data = ir0[5];
          ir0[5] = (v429_data + (v401_data * v427_data));
          float v432_data = s0[110];
          float v434_data = ir0[6];
          ir0[6] = (v434_data + (v401_data * v432_data));
          float v437_data = s0[127];
          float v439_data = ir0[7];
          ir0[7] = (v439_data + (v401_data * v437_data));
          float v442_data = s0[144];
          float v444_data = ir0[8];
          ir0[8] = (v444_data + (v401_data * v442_data));
          float v447_data = v32_g ? (glb_m1[(v28_lead + 90)]) : (0.0f);
          float v448_data = s0[9];
          float v450_data = ir0[0];
          ir0[0] = (v450_data + (v447_data * v448_data));
          float v453_data = s0[26];
          float v455_data = ir0[1];
          ir0[1] = (v455_data + (v447_data * v453_data));
          float v458_data = s0[43];
          float v460_data = ir0[2];
          ir0[2] = (v460_data + (v447_data * v458_data));
          float v463_data = s0[60];
          float v465_data = ir0[3];
          ir0[3] = (v465_data + (v447_data * v463_data));
          float v468_data = s0[77];
          float v470_data = ir0[4];
          ir0[4] = (v470_data + (v447_data * v468_data));
          float v473_data = s0[94];
          float v475_data = ir0[5];
          ir0[5] = (v475_data + (v447_data * v473_data));
          float v478_data = s0[111];
          float v480_data = ir0[6];
          ir0[6] = (v480_data + (v447_data * v478_data));
          float v483_data = s0[128];
          float v485_data = ir0[7];
          ir0[7] = (v485_data + (v447_data * v483_data));
          float v488_data = s0[145];
          float v490_data = ir0[8];
          ir0[8] = (v490_data + (v447_data * v488_data));
          float v493_data = v32_g ? (glb_m1[(v28_lead + 100)]) : (0.0f);
          float v494_data = s0[10];
          float v496_data = ir0[0];
          ir0[0] = (v496_data + (v493_data * v494_data));
          float v499_data = s0[27];
          float v501_data = ir0[1];
          ir0[1] = (v501_data + (v493_data * v499_data));
          float v504_data = s0[44];
          float v506_data = ir0[2];
          ir0[2] = (v506_data + (v493_data * v504_data));
          float v509_data = s0[61];
          float v511_data = ir0[3];
          ir0[3] = (v511_data + (v493_data * v509_data));
          float v514_data = s0[78];
          float v516_data = ir0[4];
          ir0[4] = (v516_data + (v493_data * v514_data));
          float v519_data = s0[95];
          float v521_data = ir0[5];
          ir0[5] = (v521_data + (v493_data * v519_data));
          float v524_data = s0[112];
          float v526_data = ir0[6];
          ir0[6] = (v526_data + (v493_data * v524_data));
          float v529_data = s0[129];
          float v531_data = ir0[7];
          ir0[7] = (v531_data + (v493_data * v529_data));
          float v534_data = s0[146];
          float v536_data = ir0[8];
          ir0[8] = (v536_data + (v493_data * v534_data));
          float v539_data = v32_g ? (glb_m1[(v28_lead + 110)]) : (0.0f);
          float v540_data = s0[11];
          float v542_data = ir0[0];
          ir0[0] = (v542_data + (v539_data * v540_data));
          float v545_data = s0[28];
          float v547_data = ir0[1];
          ir0[1] = (v547_data + (v539_data * v545_data));
          float v550_data = s0[45];
          float v552_data = ir0[2];
          ir0[2] = (v552_data + (v539_data * v550_data));
          float v555_data = s0[62];
          float v557_data = ir0[3];
          ir0[3] = (v557_data + (v539_data * v555_data));
          float v560_data = s0[79];
          float v562_data = ir0[4];
          ir0[4] = (v562_data + (v539_data * v560_data));
          float v565_data = s0[96];
          float v567_data = ir0[5];
          ir0[5] = (v567_data + (v539_data * v565_data));
          float v570_data = s0[113];
          float v572_data = ir0[6];
          ir0[6] = (v572_data + (v539_data * v570_data));
          float v575_data = s0[130];
          float v577_data = ir0[7];
          ir0[7] = (v577_data + (v539_data * v575_data));
          float v580_data = s0[147];
          float v582_data = ir0[8];
          ir0[8] = (v582_data + (v539_data * v580_data));
          float v585_data = v32_g ? (glb_m1[(v28_lead + 120)]) : (0.0f);
          float v586_data = s0[12];
          float v588_data = ir0[0];
          ir0[0] = (v588_data + (v585_data * v586_data));
          float v591_data = s0[29];
          float v593_data = ir0[1];
          ir0[1] = (v593_data + (v585_data * v591_data));
          float v596_data = s0[46];
          float v598_data = ir0[2];
          ir0[2] = (v598_data + (v585_data * v596_data));
          float v601_data = s0[63];
          float v603_data = ir0[3];
          ir0[3] = (v603_data + (v585_data * v601_data));
          float v606_data = s0[80];
          float v608_data = ir0[4];
          ir0[4] = (v608_data + (v585_data * v606_data));
          float v611_data = s0[97];
          float v613_data = ir0[5];
          ir0[5] = (v613_data + (v585_data * v611_data));
          float v616_data = s0[114];
          float v618_data = ir0[6];
          ir0[6] = (v618_data + (v585_data * v616_data));
          float v621_data = s0[131];
          float v623_data = ir0[7];
          ir0[7] = (v623_data + (v585_data * v621_data));
          float v626_data = s0[148];
          float v628_data = ir0[8];
          ir0[8] = (v628_data + (v585_data * v626_data));
          float v631_data = v32_g ? (glb_m1[(v28_lead + 130)]) : (0.0f);
          float v632_data = s0[13];
          float v634_data = ir0[0];
          ir0[0] = (v634_data + (v631_data * v632_data));
          float v637_data = s0[30];
          float v639_data = ir0[1];
          ir0[1] = (v639_data + (v631_data * v637_data));
          float v642_data = s0[47];
          float v644_data = ir0[2];
          ir0[2] = (v644_data + (v631_data * v642_data));
          float v647_data = s0[64];
          float v649_data = ir0[3];
          ir0[3] = (v649_data + (v631_data * v647_data));
          float v652_data = s0[81];
          float v654_data = ir0[4];
          ir0[4] = (v654_data + (v631_data * v652_data));
          float v657_data = s0[98];
          float v659_data = ir0[5];
          ir0[5] = (v659_data + (v631_data * v657_data));
          float v662_data = s0[115];
          float v664_data = ir0[6];
          ir0[6] = (v664_data + (v631_data * v662_data));
          float v667_data = s0[132];
          float v669_data = ir0[7];
          ir0[7] = (v669_data + (v631_data * v667_data));
          float v672_data = s0[149];
          float v674_data = ir0[8];
          ir0[8] = (v674_data + (v631_data * v672_data));
          float v677_data = v32_g ? (glb_m1[(v28_lead + 140)]) : (0.0f);
          float v678_data = s0[14];
          float v680_data = ir0[0];
          ir0[0] = (v680_data + (v677_data * v678_data));
          float v683_data = s0[31];
          float v685_data = ir0[1];
          ir0[1] = (v685_data + (v677_data * v683_data));
          float v688_data = s0[48];
          float v690_data = ir0[2];
          ir0[2] = (v690_data + (v677_data * v688_data));
          float v693_data = s0[65];
          float v695_data = ir0[3];
          ir0[3] = (v695_data + (v677_data * v693_data));
          float v698_data = s0[82];
          float v700_data = ir0[4];
          ir0[4] = (v700_data + (v677_data * v698_data));
          float v703_data = s0[99];
          float v705_data = ir0[5];
          ir0[5] = (v705_data + (v677_data * v703_data));
          float v708_data = s0[116];
          float v710_data = ir0[6];
          ir0[6] = (v710_data + (v677_data * v708_data));
          float v713_data = s0[133];
          float v715_data = ir0[7];
          ir0[7] = (v715_data + (v677_data * v713_data));
          float v718_data = s0[150];
          float v720_data = ir0[8];
          ir0[8] = (v720_data + (v677_data * v718_data));
          float v723_data = v32_g ? (glb_m1[(v28_lead + 150)]) : (0.0f);
          float v724_data = s0[15];
          float v726_data = ir0[0];
          ir0[0] = (v726_data + (v723_data * v724_data));
          float v729_data = s0[32];
          float v731_data = ir0[1];
          ir0[1] = (v731_data + (v723_data * v729_data));
          float v734_data = s0[49];
          float v736_data = ir0[2];
          ir0[2] = (v736_data + (v723_data * v734_data));
          float v739_data = s0[66];
          float v741_data = ir0[3];
          ir0[3] = (v741_data + (v723_data * v739_data));
          float v744_data = s0[83];
          float v746_data = ir0[4];
          ir0[4] = (v746_data + (v723_data * v744_data));
          float v749_data = s0[100];
          float v751_data = ir0[5];
          ir0[5] = (v751_data + (v723_data * v749_data));
          float v754_data = s0[117];
          float v756_data = ir0[6];
          ir0[6] = (v756_data + (v723_data * v754_data));
          float v759_data = s0[134];
          float v761_data = ir0[7];
          ir0[7] = (v761_data + (v723_data * v759_data));
          float v764_data = s0[151];
          float v766_data = ir0[8];
          ir0[8] = (v766_data + (v723_data * v764_data));
          float v769_data = v32_g ? (glb_m1[(v28_lead + 160)]) : (0.0f);
          float v770_data = s0[16];
          float v772_data = ir0[0];
          ir0[0] = (v772_data + (v769_data * v770_data));
          float v775_data = s0[33];
          float v777_data = ir0[1];
          ir0[1] = (v777_data + (v769_data * v775_data));
          float v780_data = s0[50];
          float v782_data = ir0[2];
          ir0[2] = (v782_data + (v769_data * v780_data));
          float v785_data = s0[67];
          float v787_data = ir0[3];
          ir0[3] = (v787_data + (v769_data * v785_data));
          float v790_data = s0[84];
          float v792_data = ir0[4];
          ir0[4] = (v792_data + (v769_data * v790_data));
          float v795_data = s0[101];
          float v797_data = ir0[5];
          ir0[5] = (v797_data + (v769_data * v795_data));
          float v800_data = s0[118];
          float v802_data = ir0[6];
          ir0[6] = (v802_data + (v769_data * v800_data));
          float v805_data = s0[135];
          float v807_data = ir0[7];
          ir0[7] = (v807_data + (v769_data * v805_data));
          float v810_data = s0[152];
          float v812_data = ir0[8];
          ir0[8] = (v812_data + (v769_data * v810_data));
          // r0 = ir0
          if (v28_lead < 10) {
            #pragma unroll
            for (int32_t v818_n1 = 0; v818_n1 < 9; ++v818_n1) {
              float v820_data = ir0[v818_n1];
              r0[v818_n1] = v820_data;
            }
          }
          // glb_m0 = store{r>g}(r0);
          if (v28_lead < 10) {
            #pragma unroll
            for (int32_t v825_i1 = 0; v825_i1 < 9; ++v825_i1) {
              float v827_data = r0[v825_i1];
              glb_m0[(v28_lead + (v825_i1 * 10))] = v827_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

