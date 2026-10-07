// === base name ===
kernel_ff1cf1686dfa3285

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_ff1cf1686dfa3285 = {{16, 8, 1}, 16, 16, 1, 8, 5632, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_ff1cf1686dfa3285(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_ff1cf1686dfa3285(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_ff1cf1686dfa3285(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_ff1cf1686dfa3285, block.x * block.y * block.z, 1408 * sizeof(float));
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
void launcher_kernel_ff1cf1686dfa3285(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_ff1cf1686dfa3285(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_ff1cf1686dfa3285, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_ff1cf1686dfa3285<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_ff1cf1686dfa3285(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 8 per block = block 16x8x1, 5632 B shared, occupancy grid
    // operands:
    //   m0 16×9(16×9) {0..16}×{0..9} strided
    //   m1 16×20(16×16) {0..16}×{1..17} none
    //   m2 20×9(16×9) {1..17}×{0..9} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1408}],"shared_bytes":5632,"shared_elements":1408,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,9]],"name":"m0","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,1],[16,17]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[1,0],[17,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[16,17]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[17,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
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
          float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 144 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v9_batchId0 * 144 + 0 + m2_extraOffset];
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 9; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
          __pipeline_commit();
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r0[9]{};
          // ir0 = +(glb_m1 * s0)
          // [(0, 16), (0, 9)] [(1, 17)]
          float ir0[9]{};
          int32_t v24_lead = threadIdx.x % 16;
          float v28_data = glb_m1[v24_lead];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v29_data = s0[0];
          float v31_data = ir0[0];
          ir0[0] = (v31_data + (v28_data * v29_data));
          float v34_data = s0[16];
          float v36_data = ir0[1];
          ir0[1] = (v36_data + (v28_data * v34_data));
          float v39_data = s0[32];
          float v41_data = ir0[2];
          ir0[2] = (v41_data + (v28_data * v39_data));
          float v44_data = s0[48];
          float v46_data = ir0[3];
          ir0[3] = (v46_data + (v28_data * v44_data));
          float v49_data = s0[64];
          float v51_data = ir0[4];
          ir0[4] = (v51_data + (v28_data * v49_data));
          float v54_data = s0[80];
          float v56_data = ir0[5];
          ir0[5] = (v56_data + (v28_data * v54_data));
          float v59_data = s0[96];
          float v61_data = ir0[6];
          ir0[6] = (v61_data + (v28_data * v59_data));
          float v64_data = s0[112];
          float v66_data = ir0[7];
          ir0[7] = (v66_data + (v28_data * v64_data));
          float v69_data = s0[128];
          float v71_data = ir0[8];
          ir0[8] = (v71_data + (v28_data * v69_data));
          float v74_data = glb_m1[(v24_lead + 16)];
          float v75_data = s0[1];
          float v77_data = ir0[0];
          ir0[0] = (v77_data + (v74_data * v75_data));
          float v80_data = s0[17];
          float v82_data = ir0[1];
          ir0[1] = (v82_data + (v74_data * v80_data));
          float v85_data = s0[33];
          float v87_data = ir0[2];
          ir0[2] = (v87_data + (v74_data * v85_data));
          float v90_data = s0[49];
          float v92_data = ir0[3];
          ir0[3] = (v92_data + (v74_data * v90_data));
          float v95_data = s0[65];
          float v97_data = ir0[4];
          ir0[4] = (v97_data + (v74_data * v95_data));
          float v100_data = s0[81];
          float v102_data = ir0[5];
          ir0[5] = (v102_data + (v74_data * v100_data));
          float v105_data = s0[97];
          float v107_data = ir0[6];
          ir0[6] = (v107_data + (v74_data * v105_data));
          float v110_data = s0[113];
          float v112_data = ir0[7];
          ir0[7] = (v112_data + (v74_data * v110_data));
          float v115_data = s0[129];
          float v117_data = ir0[8];
          ir0[8] = (v117_data + (v74_data * v115_data));
          float v120_data = glb_m1[(v24_lead + 32)];
          float v121_data = s0[2];
          float v123_data = ir0[0];
          ir0[0] = (v123_data + (v120_data * v121_data));
          float v126_data = s0[18];
          float v128_data = ir0[1];
          ir0[1] = (v128_data + (v120_data * v126_data));
          float v131_data = s0[34];
          float v133_data = ir0[2];
          ir0[2] = (v133_data + (v120_data * v131_data));
          float v136_data = s0[50];
          float v138_data = ir0[3];
          ir0[3] = (v138_data + (v120_data * v136_data));
          float v141_data = s0[66];
          float v143_data = ir0[4];
          ir0[4] = (v143_data + (v120_data * v141_data));
          float v146_data = s0[82];
          float v148_data = ir0[5];
          ir0[5] = (v148_data + (v120_data * v146_data));
          float v151_data = s0[98];
          float v153_data = ir0[6];
          ir0[6] = (v153_data + (v120_data * v151_data));
          float v156_data = s0[114];
          float v158_data = ir0[7];
          ir0[7] = (v158_data + (v120_data * v156_data));
          float v161_data = s0[130];
          float v163_data = ir0[8];
          ir0[8] = (v163_data + (v120_data * v161_data));
          float v166_data = glb_m1[(v24_lead + 48)];
          float v167_data = s0[3];
          float v169_data = ir0[0];
          ir0[0] = (v169_data + (v166_data * v167_data));
          float v172_data = s0[19];
          float v174_data = ir0[1];
          ir0[1] = (v174_data + (v166_data * v172_data));
          float v177_data = s0[35];
          float v179_data = ir0[2];
          ir0[2] = (v179_data + (v166_data * v177_data));
          float v182_data = s0[51];
          float v184_data = ir0[3];
          ir0[3] = (v184_data + (v166_data * v182_data));
          float v187_data = s0[67];
          float v189_data = ir0[4];
          ir0[4] = (v189_data + (v166_data * v187_data));
          float v192_data = s0[83];
          float v194_data = ir0[5];
          ir0[5] = (v194_data + (v166_data * v192_data));
          float v197_data = s0[99];
          float v199_data = ir0[6];
          ir0[6] = (v199_data + (v166_data * v197_data));
          float v202_data = s0[115];
          float v204_data = ir0[7];
          ir0[7] = (v204_data + (v166_data * v202_data));
          float v207_data = s0[131];
          float v209_data = ir0[8];
          ir0[8] = (v209_data + (v166_data * v207_data));
          float v212_data = glb_m1[(v24_lead + 64)];
          float v213_data = s0[4];
          float v215_data = ir0[0];
          ir0[0] = (v215_data + (v212_data * v213_data));
          float v218_data = s0[20];
          float v220_data = ir0[1];
          ir0[1] = (v220_data + (v212_data * v218_data));
          float v223_data = s0[36];
          float v225_data = ir0[2];
          ir0[2] = (v225_data + (v212_data * v223_data));
          float v228_data = s0[52];
          float v230_data = ir0[3];
          ir0[3] = (v230_data + (v212_data * v228_data));
          float v233_data = s0[68];
          float v235_data = ir0[4];
          ir0[4] = (v235_data + (v212_data * v233_data));
          float v238_data = s0[84];
          float v240_data = ir0[5];
          ir0[5] = (v240_data + (v212_data * v238_data));
          float v243_data = s0[100];
          float v245_data = ir0[6];
          ir0[6] = (v245_data + (v212_data * v243_data));
          float v248_data = s0[116];
          float v250_data = ir0[7];
          ir0[7] = (v250_data + (v212_data * v248_data));
          float v253_data = s0[132];
          float v255_data = ir0[8];
          ir0[8] = (v255_data + (v212_data * v253_data));
          float v258_data = glb_m1[(v24_lead + 80)];
          float v259_data = s0[5];
          float v261_data = ir0[0];
          ir0[0] = (v261_data + (v258_data * v259_data));
          float v264_data = s0[21];
          float v266_data = ir0[1];
          ir0[1] = (v266_data + (v258_data * v264_data));
          float v269_data = s0[37];
          float v271_data = ir0[2];
          ir0[2] = (v271_data + (v258_data * v269_data));
          float v274_data = s0[53];
          float v276_data = ir0[3];
          ir0[3] = (v276_data + (v258_data * v274_data));
          float v279_data = s0[69];
          float v281_data = ir0[4];
          ir0[4] = (v281_data + (v258_data * v279_data));
          float v284_data = s0[85];
          float v286_data = ir0[5];
          ir0[5] = (v286_data + (v258_data * v284_data));
          float v289_data = s0[101];
          float v291_data = ir0[6];
          ir0[6] = (v291_data + (v258_data * v289_data));
          float v294_data = s0[117];
          float v296_data = ir0[7];
          ir0[7] = (v296_data + (v258_data * v294_data));
          float v299_data = s0[133];
          float v301_data = ir0[8];
          ir0[8] = (v301_data + (v258_data * v299_data));
          float v304_data = glb_m1[(v24_lead + 96)];
          float v305_data = s0[6];
          float v307_data = ir0[0];
          ir0[0] = (v307_data + (v304_data * v305_data));
          float v310_data = s0[22];
          float v312_data = ir0[1];
          ir0[1] = (v312_data + (v304_data * v310_data));
          float v315_data = s0[38];
          float v317_data = ir0[2];
          ir0[2] = (v317_data + (v304_data * v315_data));
          float v320_data = s0[54];
          float v322_data = ir0[3];
          ir0[3] = (v322_data + (v304_data * v320_data));
          float v325_data = s0[70];
          float v327_data = ir0[4];
          ir0[4] = (v327_data + (v304_data * v325_data));
          float v330_data = s0[86];
          float v332_data = ir0[5];
          ir0[5] = (v332_data + (v304_data * v330_data));
          float v335_data = s0[102];
          float v337_data = ir0[6];
          ir0[6] = (v337_data + (v304_data * v335_data));
          float v340_data = s0[118];
          float v342_data = ir0[7];
          ir0[7] = (v342_data + (v304_data * v340_data));
          float v345_data = s0[134];
          float v347_data = ir0[8];
          ir0[8] = (v347_data + (v304_data * v345_data));
          float v350_data = glb_m1[(v24_lead + 112)];
          float v351_data = s0[7];
          float v353_data = ir0[0];
          ir0[0] = (v353_data + (v350_data * v351_data));
          float v356_data = s0[23];
          float v358_data = ir0[1];
          ir0[1] = (v358_data + (v350_data * v356_data));
          float v361_data = s0[39];
          float v363_data = ir0[2];
          ir0[2] = (v363_data + (v350_data * v361_data));
          float v366_data = s0[55];
          float v368_data = ir0[3];
          ir0[3] = (v368_data + (v350_data * v366_data));
          float v371_data = s0[71];
          float v373_data = ir0[4];
          ir0[4] = (v373_data + (v350_data * v371_data));
          float v376_data = s0[87];
          float v378_data = ir0[5];
          ir0[5] = (v378_data + (v350_data * v376_data));
          float v381_data = s0[103];
          float v383_data = ir0[6];
          ir0[6] = (v383_data + (v350_data * v381_data));
          float v386_data = s0[119];
          float v388_data = ir0[7];
          ir0[7] = (v388_data + (v350_data * v386_data));
          float v391_data = s0[135];
          float v393_data = ir0[8];
          ir0[8] = (v393_data + (v350_data * v391_data));
          float v396_data = glb_m1[(v24_lead + 128)];
          float v397_data = s0[8];
          float v399_data = ir0[0];
          ir0[0] = (v399_data + (v396_data * v397_data));
          float v402_data = s0[24];
          float v404_data = ir0[1];
          ir0[1] = (v404_data + (v396_data * v402_data));
          float v407_data = s0[40];
          float v409_data = ir0[2];
          ir0[2] = (v409_data + (v396_data * v407_data));
          float v412_data = s0[56];
          float v414_data = ir0[3];
          ir0[3] = (v414_data + (v396_data * v412_data));
          float v417_data = s0[72];
          float v419_data = ir0[4];
          ir0[4] = (v419_data + (v396_data * v417_data));
          float v422_data = s0[88];
          float v424_data = ir0[5];
          ir0[5] = (v424_data + (v396_data * v422_data));
          float v427_data = s0[104];
          float v429_data = ir0[6];
          ir0[6] = (v429_data + (v396_data * v427_data));
          float v432_data = s0[120];
          float v434_data = ir0[7];
          ir0[7] = (v434_data + (v396_data * v432_data));
          float v437_data = s0[136];
          float v439_data = ir0[8];
          ir0[8] = (v439_data + (v396_data * v437_data));
          float v442_data = glb_m1[(v24_lead + 144)];
          float v443_data = s0[9];
          float v445_data = ir0[0];
          ir0[0] = (v445_data + (v442_data * v443_data));
          float v448_data = s0[25];
          float v450_data = ir0[1];
          ir0[1] = (v450_data + (v442_data * v448_data));
          float v453_data = s0[41];
          float v455_data = ir0[2];
          ir0[2] = (v455_data + (v442_data * v453_data));
          float v458_data = s0[57];
          float v460_data = ir0[3];
          ir0[3] = (v460_data + (v442_data * v458_data));
          float v463_data = s0[73];
          float v465_data = ir0[4];
          ir0[4] = (v465_data + (v442_data * v463_data));
          float v468_data = s0[89];
          float v470_data = ir0[5];
          ir0[5] = (v470_data + (v442_data * v468_data));
          float v473_data = s0[105];
          float v475_data = ir0[6];
          ir0[6] = (v475_data + (v442_data * v473_data));
          float v478_data = s0[121];
          float v480_data = ir0[7];
          ir0[7] = (v480_data + (v442_data * v478_data));
          float v483_data = s0[137];
          float v485_data = ir0[8];
          ir0[8] = (v485_data + (v442_data * v483_data));
          float v488_data = glb_m1[(v24_lead + 160)];
          float v489_data = s0[10];
          float v491_data = ir0[0];
          ir0[0] = (v491_data + (v488_data * v489_data));
          float v494_data = s0[26];
          float v496_data = ir0[1];
          ir0[1] = (v496_data + (v488_data * v494_data));
          float v499_data = s0[42];
          float v501_data = ir0[2];
          ir0[2] = (v501_data + (v488_data * v499_data));
          float v504_data = s0[58];
          float v506_data = ir0[3];
          ir0[3] = (v506_data + (v488_data * v504_data));
          float v509_data = s0[74];
          float v511_data = ir0[4];
          ir0[4] = (v511_data + (v488_data * v509_data));
          float v514_data = s0[90];
          float v516_data = ir0[5];
          ir0[5] = (v516_data + (v488_data * v514_data));
          float v519_data = s0[106];
          float v521_data = ir0[6];
          ir0[6] = (v521_data + (v488_data * v519_data));
          float v524_data = s0[122];
          float v526_data = ir0[7];
          ir0[7] = (v526_data + (v488_data * v524_data));
          float v529_data = s0[138];
          float v531_data = ir0[8];
          ir0[8] = (v531_data + (v488_data * v529_data));
          float v534_data = glb_m1[(v24_lead + 176)];
          float v535_data = s0[11];
          float v537_data = ir0[0];
          ir0[0] = (v537_data + (v534_data * v535_data));
          float v540_data = s0[27];
          float v542_data = ir0[1];
          ir0[1] = (v542_data + (v534_data * v540_data));
          float v545_data = s0[43];
          float v547_data = ir0[2];
          ir0[2] = (v547_data + (v534_data * v545_data));
          float v550_data = s0[59];
          float v552_data = ir0[3];
          ir0[3] = (v552_data + (v534_data * v550_data));
          float v555_data = s0[75];
          float v557_data = ir0[4];
          ir0[4] = (v557_data + (v534_data * v555_data));
          float v560_data = s0[91];
          float v562_data = ir0[5];
          ir0[5] = (v562_data + (v534_data * v560_data));
          float v565_data = s0[107];
          float v567_data = ir0[6];
          ir0[6] = (v567_data + (v534_data * v565_data));
          float v570_data = s0[123];
          float v572_data = ir0[7];
          ir0[7] = (v572_data + (v534_data * v570_data));
          float v575_data = s0[139];
          float v577_data = ir0[8];
          ir0[8] = (v577_data + (v534_data * v575_data));
          float v580_data = glb_m1[(v24_lead + 192)];
          float v581_data = s0[12];
          float v583_data = ir0[0];
          ir0[0] = (v583_data + (v580_data * v581_data));
          float v586_data = s0[28];
          float v588_data = ir0[1];
          ir0[1] = (v588_data + (v580_data * v586_data));
          float v591_data = s0[44];
          float v593_data = ir0[2];
          ir0[2] = (v593_data + (v580_data * v591_data));
          float v596_data = s0[60];
          float v598_data = ir0[3];
          ir0[3] = (v598_data + (v580_data * v596_data));
          float v601_data = s0[76];
          float v603_data = ir0[4];
          ir0[4] = (v603_data + (v580_data * v601_data));
          float v606_data = s0[92];
          float v608_data = ir0[5];
          ir0[5] = (v608_data + (v580_data * v606_data));
          float v611_data = s0[108];
          float v613_data = ir0[6];
          ir0[6] = (v613_data + (v580_data * v611_data));
          float v616_data = s0[124];
          float v618_data = ir0[7];
          ir0[7] = (v618_data + (v580_data * v616_data));
          float v621_data = s0[140];
          float v623_data = ir0[8];
          ir0[8] = (v623_data + (v580_data * v621_data));
          float v626_data = glb_m1[(v24_lead + 208)];
          float v627_data = s0[13];
          float v629_data = ir0[0];
          ir0[0] = (v629_data + (v626_data * v627_data));
          float v632_data = s0[29];
          float v634_data = ir0[1];
          ir0[1] = (v634_data + (v626_data * v632_data));
          float v637_data = s0[45];
          float v639_data = ir0[2];
          ir0[2] = (v639_data + (v626_data * v637_data));
          float v642_data = s0[61];
          float v644_data = ir0[3];
          ir0[3] = (v644_data + (v626_data * v642_data));
          float v647_data = s0[77];
          float v649_data = ir0[4];
          ir0[4] = (v649_data + (v626_data * v647_data));
          float v652_data = s0[93];
          float v654_data = ir0[5];
          ir0[5] = (v654_data + (v626_data * v652_data));
          float v657_data = s0[109];
          float v659_data = ir0[6];
          ir0[6] = (v659_data + (v626_data * v657_data));
          float v662_data = s0[125];
          float v664_data = ir0[7];
          ir0[7] = (v664_data + (v626_data * v662_data));
          float v667_data = s0[141];
          float v669_data = ir0[8];
          ir0[8] = (v669_data + (v626_data * v667_data));
          float v672_data = glb_m1[(v24_lead + 224)];
          float v673_data = s0[14];
          float v675_data = ir0[0];
          ir0[0] = (v675_data + (v672_data * v673_data));
          float v678_data = s0[30];
          float v680_data = ir0[1];
          ir0[1] = (v680_data + (v672_data * v678_data));
          float v683_data = s0[46];
          float v685_data = ir0[2];
          ir0[2] = (v685_data + (v672_data * v683_data));
          float v688_data = s0[62];
          float v690_data = ir0[3];
          ir0[3] = (v690_data + (v672_data * v688_data));
          float v693_data = s0[78];
          float v695_data = ir0[4];
          ir0[4] = (v695_data + (v672_data * v693_data));
          float v698_data = s0[94];
          float v700_data = ir0[5];
          ir0[5] = (v700_data + (v672_data * v698_data));
          float v703_data = s0[110];
          float v705_data = ir0[6];
          ir0[6] = (v705_data + (v672_data * v703_data));
          float v708_data = s0[126];
          float v710_data = ir0[7];
          ir0[7] = (v710_data + (v672_data * v708_data));
          float v713_data = s0[142];
          float v715_data = ir0[8];
          ir0[8] = (v715_data + (v672_data * v713_data));
          float v718_data = glb_m1[(v24_lead + 240)];
          float v719_data = s0[15];
          float v721_data = ir0[0];
          ir0[0] = (v721_data + (v718_data * v719_data));
          float v724_data = s0[31];
          float v726_data = ir0[1];
          ir0[1] = (v726_data + (v718_data * v724_data));
          float v729_data = s0[47];
          float v731_data = ir0[2];
          ir0[2] = (v731_data + (v718_data * v729_data));
          float v734_data = s0[63];
          float v736_data = ir0[3];
          ir0[3] = (v736_data + (v718_data * v734_data));
          float v739_data = s0[79];
          float v741_data = ir0[4];
          ir0[4] = (v741_data + (v718_data * v739_data));
          float v744_data = s0[95];
          float v746_data = ir0[5];
          ir0[5] = (v746_data + (v718_data * v744_data));
          float v749_data = s0[111];
          float v751_data = ir0[6];
          ir0[6] = (v751_data + (v718_data * v749_data));
          float v754_data = s0[127];
          float v756_data = ir0[7];
          ir0[7] = (v756_data + (v718_data * v754_data));
          float v759_data = s0[143];
          float v761_data = ir0[8];
          ir0[8] = (v761_data + (v718_data * v759_data));
          // r0 = ir0
          #pragma unroll
          for (int32_t v766_n0 = 0; v766_n0 < 1; ++v766_n0) {
            #pragma unroll
            for (int32_t v767_n1 = 0; v767_n1 < 9; ++v767_n1) {
              int32_t v768_a = v766_n0 + v767_n1;
              float v769_data = ir0[v768_a];
              r0[v768_a] = v769_data;
            }
          }
          // glb_m0 = store{r>g}(r0);
          #pragma unroll
          for (int32_t v773_i0 = 0; v773_i0 < 1; ++v773_i0) {
            int32_t v778_lead = v24_lead + (v773_i0 * 16);
            #pragma unroll
            for (int32_t v774_i1 = 0; v774_i1 < 9; ++v774_i1) {
              float v776_data = r0[(v773_i0 + v774_i1)];
              glb_m0[(v778_lead + (v774_i1 * 16))] = v776_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

