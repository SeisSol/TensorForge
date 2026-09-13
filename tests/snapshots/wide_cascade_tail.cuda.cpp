// === base name ===
kernel_86e96d4803874eb1

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_86e96d4803874eb1 = {{32, 4, 1}, 32, 24, 1, 4, 3584, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_86e96d4803874eb1(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_86e96d4803874eb1(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_86e96d4803874eb1(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 4, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_86e96d4803874eb1, block.x * block.y * block.z, 896 * sizeof(float));
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
  config.sharedMemBytes = 896 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_86e96d4803874eb1(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_86e96d4803874eb1(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_86e96d4803874eb1, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_86e96d4803874eb1<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_86e96d4803874eb1(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (24 active) x 4 per block = block 32x4x1, 3584 B shared, occupancy grid
    // operands:
    //   m0 24×9(24×9) {0..24}×{0..9} strided
    //   m1 24×24(24×24) {0..24}×{0..24} strided
    //   m2 24×9(24×9) {0..24}×{0..9} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":24,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":896}],"shared_bytes":3584,"shared_elements":896,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[24,9]],"name":"m0","ordered":false,"parts":1,"shape":[24,9],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[24,24]],"name":"m1","ordered":false,"parts":1,"shape":[24,24],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[24,9]],"name":"m2","ordered":false,"parts":1,"shape":[24,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[24,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[24,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[24,24]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[24,24]},{"addressing":"strided","bbox":[[0,0],[24,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[24,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[224 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[224];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v5_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v5_batchId0 < numElements0; v5_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v6_ahead1 = v5_batchId0 + (gridDim.x * blockDim.y);
        size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 216 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 576 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 216 + 0 + m2_extraOffset];
          float r0[24]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v19_lead = threadIdx.x % 32;
          bool v20_g = v19_lead < 24;
          if (v20_g) {
            #pragma unroll
            for (int32_t v21_i1 = 0; v21_i1 < 24; ++v21_i1) {
              float v26_data = __ldcg(&glb_m1[(v19_lead + (v21_i1 * 24))]);
              r0[v21_i1] = v26_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 4 * threadIdx.x + 0], &glb_m2[0 + 0 + 4 * threadIdx.x + 0], 16);
          __pipeline_memcpy_async(&s0[0 + 0 + 2 * threadIdx.x + 128], &glb_m2[0 + 0 + 2 * threadIdx.x + 128], 8);
          if (threadIdx.x < 24) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 192], &glb_m2[0 + 0 + 1 * threadIdx.x + 192], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[9]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 24), (0, 9)] [(0, 24)]
          float ir1[9]{};
          float v33_data = r0[0];
          float v34_data = s0[0];
          float v36_data = ir1[0];
          ir1[0] = (v36_data + (v33_data * v34_data));
          float v39_data = s0[24];
          float v41_data = ir1[1];
          ir1[1] = (v41_data + (v33_data * v39_data));
          float v44_data = s0[48];
          float v46_data = ir1[2];
          ir1[2] = (v46_data + (v33_data * v44_data));
          float v49_data = s0[72];
          float v51_data = ir1[3];
          ir1[3] = (v51_data + (v33_data * v49_data));
          float v54_data = s0[96];
          float v56_data = ir1[4];
          ir1[4] = (v56_data + (v33_data * v54_data));
          float v59_data = s0[120];
          float v61_data = ir1[5];
          ir1[5] = (v61_data + (v33_data * v59_data));
          float v64_data = s0[144];
          float v66_data = ir1[6];
          ir1[6] = (v66_data + (v33_data * v64_data));
          float v69_data = s0[168];
          float v71_data = ir1[7];
          ir1[7] = (v71_data + (v33_data * v69_data));
          float v74_data = s0[192];
          float v76_data = ir1[8];
          ir1[8] = (v76_data + (v33_data * v74_data));
          float v78_data = r0[1];
          float v79_data = s0[1];
          float v81_data = ir1[0];
          ir1[0] = (v81_data + (v78_data * v79_data));
          float v84_data = s0[25];
          float v86_data = ir1[1];
          ir1[1] = (v86_data + (v78_data * v84_data));
          float v89_data = s0[49];
          float v91_data = ir1[2];
          ir1[2] = (v91_data + (v78_data * v89_data));
          float v94_data = s0[73];
          float v96_data = ir1[3];
          ir1[3] = (v96_data + (v78_data * v94_data));
          float v99_data = s0[97];
          float v101_data = ir1[4];
          ir1[4] = (v101_data + (v78_data * v99_data));
          float v104_data = s0[121];
          float v106_data = ir1[5];
          ir1[5] = (v106_data + (v78_data * v104_data));
          float v109_data = s0[145];
          float v111_data = ir1[6];
          ir1[6] = (v111_data + (v78_data * v109_data));
          float v114_data = s0[169];
          float v116_data = ir1[7];
          ir1[7] = (v116_data + (v78_data * v114_data));
          float v119_data = s0[193];
          float v121_data = ir1[8];
          ir1[8] = (v121_data + (v78_data * v119_data));
          float v123_data = r0[2];
          float v124_data = s0[2];
          float v126_data = ir1[0];
          ir1[0] = (v126_data + (v123_data * v124_data));
          float v129_data = s0[26];
          float v131_data = ir1[1];
          ir1[1] = (v131_data + (v123_data * v129_data));
          float v134_data = s0[50];
          float v136_data = ir1[2];
          ir1[2] = (v136_data + (v123_data * v134_data));
          float v139_data = s0[74];
          float v141_data = ir1[3];
          ir1[3] = (v141_data + (v123_data * v139_data));
          float v144_data = s0[98];
          float v146_data = ir1[4];
          ir1[4] = (v146_data + (v123_data * v144_data));
          float v149_data = s0[122];
          float v151_data = ir1[5];
          ir1[5] = (v151_data + (v123_data * v149_data));
          float v154_data = s0[146];
          float v156_data = ir1[6];
          ir1[6] = (v156_data + (v123_data * v154_data));
          float v159_data = s0[170];
          float v161_data = ir1[7];
          ir1[7] = (v161_data + (v123_data * v159_data));
          float v164_data = s0[194];
          float v166_data = ir1[8];
          ir1[8] = (v166_data + (v123_data * v164_data));
          float v168_data = r0[3];
          float v169_data = s0[3];
          float v171_data = ir1[0];
          ir1[0] = (v171_data + (v168_data * v169_data));
          float v174_data = s0[27];
          float v176_data = ir1[1];
          ir1[1] = (v176_data + (v168_data * v174_data));
          float v179_data = s0[51];
          float v181_data = ir1[2];
          ir1[2] = (v181_data + (v168_data * v179_data));
          float v184_data = s0[75];
          float v186_data = ir1[3];
          ir1[3] = (v186_data + (v168_data * v184_data));
          float v189_data = s0[99];
          float v191_data = ir1[4];
          ir1[4] = (v191_data + (v168_data * v189_data));
          float v194_data = s0[123];
          float v196_data = ir1[5];
          ir1[5] = (v196_data + (v168_data * v194_data));
          float v199_data = s0[147];
          float v201_data = ir1[6];
          ir1[6] = (v201_data + (v168_data * v199_data));
          float v204_data = s0[171];
          float v206_data = ir1[7];
          ir1[7] = (v206_data + (v168_data * v204_data));
          float v209_data = s0[195];
          float v211_data = ir1[8];
          ir1[8] = (v211_data + (v168_data * v209_data));
          float v213_data = r0[4];
          float v214_data = s0[4];
          float v216_data = ir1[0];
          ir1[0] = (v216_data + (v213_data * v214_data));
          float v219_data = s0[28];
          float v221_data = ir1[1];
          ir1[1] = (v221_data + (v213_data * v219_data));
          float v224_data = s0[52];
          float v226_data = ir1[2];
          ir1[2] = (v226_data + (v213_data * v224_data));
          float v229_data = s0[76];
          float v231_data = ir1[3];
          ir1[3] = (v231_data + (v213_data * v229_data));
          float v234_data = s0[100];
          float v236_data = ir1[4];
          ir1[4] = (v236_data + (v213_data * v234_data));
          float v239_data = s0[124];
          float v241_data = ir1[5];
          ir1[5] = (v241_data + (v213_data * v239_data));
          float v244_data = s0[148];
          float v246_data = ir1[6];
          ir1[6] = (v246_data + (v213_data * v244_data));
          float v249_data = s0[172];
          float v251_data = ir1[7];
          ir1[7] = (v251_data + (v213_data * v249_data));
          float v254_data = s0[196];
          float v256_data = ir1[8];
          ir1[8] = (v256_data + (v213_data * v254_data));
          float v258_data = r0[5];
          float v259_data = s0[5];
          float v261_data = ir1[0];
          ir1[0] = (v261_data + (v258_data * v259_data));
          float v264_data = s0[29];
          float v266_data = ir1[1];
          ir1[1] = (v266_data + (v258_data * v264_data));
          float v269_data = s0[53];
          float v271_data = ir1[2];
          ir1[2] = (v271_data + (v258_data * v269_data));
          float v274_data = s0[77];
          float v276_data = ir1[3];
          ir1[3] = (v276_data + (v258_data * v274_data));
          float v279_data = s0[101];
          float v281_data = ir1[4];
          ir1[4] = (v281_data + (v258_data * v279_data));
          float v284_data = s0[125];
          float v286_data = ir1[5];
          ir1[5] = (v286_data + (v258_data * v284_data));
          float v289_data = s0[149];
          float v291_data = ir1[6];
          ir1[6] = (v291_data + (v258_data * v289_data));
          float v294_data = s0[173];
          float v296_data = ir1[7];
          ir1[7] = (v296_data + (v258_data * v294_data));
          float v299_data = s0[197];
          float v301_data = ir1[8];
          ir1[8] = (v301_data + (v258_data * v299_data));
          float v303_data = r0[6];
          float v304_data = s0[6];
          float v306_data = ir1[0];
          ir1[0] = (v306_data + (v303_data * v304_data));
          float v309_data = s0[30];
          float v311_data = ir1[1];
          ir1[1] = (v311_data + (v303_data * v309_data));
          float v314_data = s0[54];
          float v316_data = ir1[2];
          ir1[2] = (v316_data + (v303_data * v314_data));
          float v319_data = s0[78];
          float v321_data = ir1[3];
          ir1[3] = (v321_data + (v303_data * v319_data));
          float v324_data = s0[102];
          float v326_data = ir1[4];
          ir1[4] = (v326_data + (v303_data * v324_data));
          float v329_data = s0[126];
          float v331_data = ir1[5];
          ir1[5] = (v331_data + (v303_data * v329_data));
          float v334_data = s0[150];
          float v336_data = ir1[6];
          ir1[6] = (v336_data + (v303_data * v334_data));
          float v339_data = s0[174];
          float v341_data = ir1[7];
          ir1[7] = (v341_data + (v303_data * v339_data));
          float v344_data = s0[198];
          float v346_data = ir1[8];
          ir1[8] = (v346_data + (v303_data * v344_data));
          float v348_data = r0[7];
          float v349_data = s0[7];
          float v351_data = ir1[0];
          ir1[0] = (v351_data + (v348_data * v349_data));
          float v354_data = s0[31];
          float v356_data = ir1[1];
          ir1[1] = (v356_data + (v348_data * v354_data));
          float v359_data = s0[55];
          float v361_data = ir1[2];
          ir1[2] = (v361_data + (v348_data * v359_data));
          float v364_data = s0[79];
          float v366_data = ir1[3];
          ir1[3] = (v366_data + (v348_data * v364_data));
          float v369_data = s0[103];
          float v371_data = ir1[4];
          ir1[4] = (v371_data + (v348_data * v369_data));
          float v374_data = s0[127];
          float v376_data = ir1[5];
          ir1[5] = (v376_data + (v348_data * v374_data));
          float v379_data = s0[151];
          float v381_data = ir1[6];
          ir1[6] = (v381_data + (v348_data * v379_data));
          float v384_data = s0[175];
          float v386_data = ir1[7];
          ir1[7] = (v386_data + (v348_data * v384_data));
          float v389_data = s0[199];
          float v391_data = ir1[8];
          ir1[8] = (v391_data + (v348_data * v389_data));
          float v393_data = r0[8];
          float v394_data = s0[8];
          float v396_data = ir1[0];
          ir1[0] = (v396_data + (v393_data * v394_data));
          float v399_data = s0[32];
          float v401_data = ir1[1];
          ir1[1] = (v401_data + (v393_data * v399_data));
          float v404_data = s0[56];
          float v406_data = ir1[2];
          ir1[2] = (v406_data + (v393_data * v404_data));
          float v409_data = s0[80];
          float v411_data = ir1[3];
          ir1[3] = (v411_data + (v393_data * v409_data));
          float v414_data = s0[104];
          float v416_data = ir1[4];
          ir1[4] = (v416_data + (v393_data * v414_data));
          float v419_data = s0[128];
          float v421_data = ir1[5];
          ir1[5] = (v421_data + (v393_data * v419_data));
          float v424_data = s0[152];
          float v426_data = ir1[6];
          ir1[6] = (v426_data + (v393_data * v424_data));
          float v429_data = s0[176];
          float v431_data = ir1[7];
          ir1[7] = (v431_data + (v393_data * v429_data));
          float v434_data = s0[200];
          float v436_data = ir1[8];
          ir1[8] = (v436_data + (v393_data * v434_data));
          float v438_data = r0[9];
          float v439_data = s0[9];
          float v441_data = ir1[0];
          ir1[0] = (v441_data + (v438_data * v439_data));
          float v444_data = s0[33];
          float v446_data = ir1[1];
          ir1[1] = (v446_data + (v438_data * v444_data));
          float v449_data = s0[57];
          float v451_data = ir1[2];
          ir1[2] = (v451_data + (v438_data * v449_data));
          float v454_data = s0[81];
          float v456_data = ir1[3];
          ir1[3] = (v456_data + (v438_data * v454_data));
          float v459_data = s0[105];
          float v461_data = ir1[4];
          ir1[4] = (v461_data + (v438_data * v459_data));
          float v464_data = s0[129];
          float v466_data = ir1[5];
          ir1[5] = (v466_data + (v438_data * v464_data));
          float v469_data = s0[153];
          float v471_data = ir1[6];
          ir1[6] = (v471_data + (v438_data * v469_data));
          float v474_data = s0[177];
          float v476_data = ir1[7];
          ir1[7] = (v476_data + (v438_data * v474_data));
          float v479_data = s0[201];
          float v481_data = ir1[8];
          ir1[8] = (v481_data + (v438_data * v479_data));
          float v483_data = r0[10];
          float v484_data = s0[10];
          float v486_data = ir1[0];
          ir1[0] = (v486_data + (v483_data * v484_data));
          float v489_data = s0[34];
          float v491_data = ir1[1];
          ir1[1] = (v491_data + (v483_data * v489_data));
          float v494_data = s0[58];
          float v496_data = ir1[2];
          ir1[2] = (v496_data + (v483_data * v494_data));
          float v499_data = s0[82];
          float v501_data = ir1[3];
          ir1[3] = (v501_data + (v483_data * v499_data));
          float v504_data = s0[106];
          float v506_data = ir1[4];
          ir1[4] = (v506_data + (v483_data * v504_data));
          float v509_data = s0[130];
          float v511_data = ir1[5];
          ir1[5] = (v511_data + (v483_data * v509_data));
          float v514_data = s0[154];
          float v516_data = ir1[6];
          ir1[6] = (v516_data + (v483_data * v514_data));
          float v519_data = s0[178];
          float v521_data = ir1[7];
          ir1[7] = (v521_data + (v483_data * v519_data));
          float v524_data = s0[202];
          float v526_data = ir1[8];
          ir1[8] = (v526_data + (v483_data * v524_data));
          float v528_data = r0[11];
          float v529_data = s0[11];
          float v531_data = ir1[0];
          ir1[0] = (v531_data + (v528_data * v529_data));
          float v534_data = s0[35];
          float v536_data = ir1[1];
          ir1[1] = (v536_data + (v528_data * v534_data));
          float v539_data = s0[59];
          float v541_data = ir1[2];
          ir1[2] = (v541_data + (v528_data * v539_data));
          float v544_data = s0[83];
          float v546_data = ir1[3];
          ir1[3] = (v546_data + (v528_data * v544_data));
          float v549_data = s0[107];
          float v551_data = ir1[4];
          ir1[4] = (v551_data + (v528_data * v549_data));
          float v554_data = s0[131];
          float v556_data = ir1[5];
          ir1[5] = (v556_data + (v528_data * v554_data));
          float v559_data = s0[155];
          float v561_data = ir1[6];
          ir1[6] = (v561_data + (v528_data * v559_data));
          float v564_data = s0[179];
          float v566_data = ir1[7];
          ir1[7] = (v566_data + (v528_data * v564_data));
          float v569_data = s0[203];
          float v571_data = ir1[8];
          ir1[8] = (v571_data + (v528_data * v569_data));
          float v573_data = r0[12];
          float v574_data = s0[12];
          float v576_data = ir1[0];
          ir1[0] = (v576_data + (v573_data * v574_data));
          float v579_data = s0[36];
          float v581_data = ir1[1];
          ir1[1] = (v581_data + (v573_data * v579_data));
          float v584_data = s0[60];
          float v586_data = ir1[2];
          ir1[2] = (v586_data + (v573_data * v584_data));
          float v589_data = s0[84];
          float v591_data = ir1[3];
          ir1[3] = (v591_data + (v573_data * v589_data));
          float v594_data = s0[108];
          float v596_data = ir1[4];
          ir1[4] = (v596_data + (v573_data * v594_data));
          float v599_data = s0[132];
          float v601_data = ir1[5];
          ir1[5] = (v601_data + (v573_data * v599_data));
          float v604_data = s0[156];
          float v606_data = ir1[6];
          ir1[6] = (v606_data + (v573_data * v604_data));
          float v609_data = s0[180];
          float v611_data = ir1[7];
          ir1[7] = (v611_data + (v573_data * v609_data));
          float v614_data = s0[204];
          float v616_data = ir1[8];
          ir1[8] = (v616_data + (v573_data * v614_data));
          float v618_data = r0[13];
          float v619_data = s0[13];
          float v621_data = ir1[0];
          ir1[0] = (v621_data + (v618_data * v619_data));
          float v624_data = s0[37];
          float v626_data = ir1[1];
          ir1[1] = (v626_data + (v618_data * v624_data));
          float v629_data = s0[61];
          float v631_data = ir1[2];
          ir1[2] = (v631_data + (v618_data * v629_data));
          float v634_data = s0[85];
          float v636_data = ir1[3];
          ir1[3] = (v636_data + (v618_data * v634_data));
          float v639_data = s0[109];
          float v641_data = ir1[4];
          ir1[4] = (v641_data + (v618_data * v639_data));
          float v644_data = s0[133];
          float v646_data = ir1[5];
          ir1[5] = (v646_data + (v618_data * v644_data));
          float v649_data = s0[157];
          float v651_data = ir1[6];
          ir1[6] = (v651_data + (v618_data * v649_data));
          float v654_data = s0[181];
          float v656_data = ir1[7];
          ir1[7] = (v656_data + (v618_data * v654_data));
          float v659_data = s0[205];
          float v661_data = ir1[8];
          ir1[8] = (v661_data + (v618_data * v659_data));
          float v663_data = r0[14];
          float v664_data = s0[14];
          float v666_data = ir1[0];
          ir1[0] = (v666_data + (v663_data * v664_data));
          float v669_data = s0[38];
          float v671_data = ir1[1];
          ir1[1] = (v671_data + (v663_data * v669_data));
          float v674_data = s0[62];
          float v676_data = ir1[2];
          ir1[2] = (v676_data + (v663_data * v674_data));
          float v679_data = s0[86];
          float v681_data = ir1[3];
          ir1[3] = (v681_data + (v663_data * v679_data));
          float v684_data = s0[110];
          float v686_data = ir1[4];
          ir1[4] = (v686_data + (v663_data * v684_data));
          float v689_data = s0[134];
          float v691_data = ir1[5];
          ir1[5] = (v691_data + (v663_data * v689_data));
          float v694_data = s0[158];
          float v696_data = ir1[6];
          ir1[6] = (v696_data + (v663_data * v694_data));
          float v699_data = s0[182];
          float v701_data = ir1[7];
          ir1[7] = (v701_data + (v663_data * v699_data));
          float v704_data = s0[206];
          float v706_data = ir1[8];
          ir1[8] = (v706_data + (v663_data * v704_data));
          float v708_data = r0[15];
          float v709_data = s0[15];
          float v711_data = ir1[0];
          ir1[0] = (v711_data + (v708_data * v709_data));
          float v714_data = s0[39];
          float v716_data = ir1[1];
          ir1[1] = (v716_data + (v708_data * v714_data));
          float v719_data = s0[63];
          float v721_data = ir1[2];
          ir1[2] = (v721_data + (v708_data * v719_data));
          float v724_data = s0[87];
          float v726_data = ir1[3];
          ir1[3] = (v726_data + (v708_data * v724_data));
          float v729_data = s0[111];
          float v731_data = ir1[4];
          ir1[4] = (v731_data + (v708_data * v729_data));
          float v734_data = s0[135];
          float v736_data = ir1[5];
          ir1[5] = (v736_data + (v708_data * v734_data));
          float v739_data = s0[159];
          float v741_data = ir1[6];
          ir1[6] = (v741_data + (v708_data * v739_data));
          float v744_data = s0[183];
          float v746_data = ir1[7];
          ir1[7] = (v746_data + (v708_data * v744_data));
          float v749_data = s0[207];
          float v751_data = ir1[8];
          ir1[8] = (v751_data + (v708_data * v749_data));
          float v753_data = r0[16];
          float v754_data = s0[16];
          float v756_data = ir1[0];
          ir1[0] = (v756_data + (v753_data * v754_data));
          float v759_data = s0[40];
          float v761_data = ir1[1];
          ir1[1] = (v761_data + (v753_data * v759_data));
          float v764_data = s0[64];
          float v766_data = ir1[2];
          ir1[2] = (v766_data + (v753_data * v764_data));
          float v769_data = s0[88];
          float v771_data = ir1[3];
          ir1[3] = (v771_data + (v753_data * v769_data));
          float v774_data = s0[112];
          float v776_data = ir1[4];
          ir1[4] = (v776_data + (v753_data * v774_data));
          float v779_data = s0[136];
          float v781_data = ir1[5];
          ir1[5] = (v781_data + (v753_data * v779_data));
          float v784_data = s0[160];
          float v786_data = ir1[6];
          ir1[6] = (v786_data + (v753_data * v784_data));
          float v789_data = s0[184];
          float v791_data = ir1[7];
          ir1[7] = (v791_data + (v753_data * v789_data));
          float v794_data = s0[208];
          float v796_data = ir1[8];
          ir1[8] = (v796_data + (v753_data * v794_data));
          float v798_data = r0[17];
          float v799_data = s0[17];
          float v801_data = ir1[0];
          ir1[0] = (v801_data + (v798_data * v799_data));
          float v804_data = s0[41];
          float v806_data = ir1[1];
          ir1[1] = (v806_data + (v798_data * v804_data));
          float v809_data = s0[65];
          float v811_data = ir1[2];
          ir1[2] = (v811_data + (v798_data * v809_data));
          float v814_data = s0[89];
          float v816_data = ir1[3];
          ir1[3] = (v816_data + (v798_data * v814_data));
          float v819_data = s0[113];
          float v821_data = ir1[4];
          ir1[4] = (v821_data + (v798_data * v819_data));
          float v824_data = s0[137];
          float v826_data = ir1[5];
          ir1[5] = (v826_data + (v798_data * v824_data));
          float v829_data = s0[161];
          float v831_data = ir1[6];
          ir1[6] = (v831_data + (v798_data * v829_data));
          float v834_data = s0[185];
          float v836_data = ir1[7];
          ir1[7] = (v836_data + (v798_data * v834_data));
          float v839_data = s0[209];
          float v841_data = ir1[8];
          ir1[8] = (v841_data + (v798_data * v839_data));
          float v843_data = r0[18];
          float v844_data = s0[18];
          float v846_data = ir1[0];
          ir1[0] = (v846_data + (v843_data * v844_data));
          float v849_data = s0[42];
          float v851_data = ir1[1];
          ir1[1] = (v851_data + (v843_data * v849_data));
          float v854_data = s0[66];
          float v856_data = ir1[2];
          ir1[2] = (v856_data + (v843_data * v854_data));
          float v859_data = s0[90];
          float v861_data = ir1[3];
          ir1[3] = (v861_data + (v843_data * v859_data));
          float v864_data = s0[114];
          float v866_data = ir1[4];
          ir1[4] = (v866_data + (v843_data * v864_data));
          float v869_data = s0[138];
          float v871_data = ir1[5];
          ir1[5] = (v871_data + (v843_data * v869_data));
          float v874_data = s0[162];
          float v876_data = ir1[6];
          ir1[6] = (v876_data + (v843_data * v874_data));
          float v879_data = s0[186];
          float v881_data = ir1[7];
          ir1[7] = (v881_data + (v843_data * v879_data));
          float v884_data = s0[210];
          float v886_data = ir1[8];
          ir1[8] = (v886_data + (v843_data * v884_data));
          float v888_data = r0[19];
          float v889_data = s0[19];
          float v891_data = ir1[0];
          ir1[0] = (v891_data + (v888_data * v889_data));
          float v894_data = s0[43];
          float v896_data = ir1[1];
          ir1[1] = (v896_data + (v888_data * v894_data));
          float v899_data = s0[67];
          float v901_data = ir1[2];
          ir1[2] = (v901_data + (v888_data * v899_data));
          float v904_data = s0[91];
          float v906_data = ir1[3];
          ir1[3] = (v906_data + (v888_data * v904_data));
          float v909_data = s0[115];
          float v911_data = ir1[4];
          ir1[4] = (v911_data + (v888_data * v909_data));
          float v914_data = s0[139];
          float v916_data = ir1[5];
          ir1[5] = (v916_data + (v888_data * v914_data));
          float v919_data = s0[163];
          float v921_data = ir1[6];
          ir1[6] = (v921_data + (v888_data * v919_data));
          float v924_data = s0[187];
          float v926_data = ir1[7];
          ir1[7] = (v926_data + (v888_data * v924_data));
          float v929_data = s0[211];
          float v931_data = ir1[8];
          ir1[8] = (v931_data + (v888_data * v929_data));
          float v933_data = r0[20];
          float v934_data = s0[20];
          float v936_data = ir1[0];
          ir1[0] = (v936_data + (v933_data * v934_data));
          float v939_data = s0[44];
          float v941_data = ir1[1];
          ir1[1] = (v941_data + (v933_data * v939_data));
          float v944_data = s0[68];
          float v946_data = ir1[2];
          ir1[2] = (v946_data + (v933_data * v944_data));
          float v949_data = s0[92];
          float v951_data = ir1[3];
          ir1[3] = (v951_data + (v933_data * v949_data));
          float v954_data = s0[116];
          float v956_data = ir1[4];
          ir1[4] = (v956_data + (v933_data * v954_data));
          float v959_data = s0[140];
          float v961_data = ir1[5];
          ir1[5] = (v961_data + (v933_data * v959_data));
          float v964_data = s0[164];
          float v966_data = ir1[6];
          ir1[6] = (v966_data + (v933_data * v964_data));
          float v969_data = s0[188];
          float v971_data = ir1[7];
          ir1[7] = (v971_data + (v933_data * v969_data));
          float v974_data = s0[212];
          float v976_data = ir1[8];
          ir1[8] = (v976_data + (v933_data * v974_data));
          float v978_data = r0[21];
          float v979_data = s0[21];
          float v981_data = ir1[0];
          ir1[0] = (v981_data + (v978_data * v979_data));
          float v984_data = s0[45];
          float v986_data = ir1[1];
          ir1[1] = (v986_data + (v978_data * v984_data));
          float v989_data = s0[69];
          float v991_data = ir1[2];
          ir1[2] = (v991_data + (v978_data * v989_data));
          float v994_data = s0[93];
          float v996_data = ir1[3];
          ir1[3] = (v996_data + (v978_data * v994_data));
          float v999_data = s0[117];
          float v1001_data = ir1[4];
          ir1[4] = (v1001_data + (v978_data * v999_data));
          float v1004_data = s0[141];
          float v1006_data = ir1[5];
          ir1[5] = (v1006_data + (v978_data * v1004_data));
          float v1009_data = s0[165];
          float v1011_data = ir1[6];
          ir1[6] = (v1011_data + (v978_data * v1009_data));
          float v1014_data = s0[189];
          float v1016_data = ir1[7];
          ir1[7] = (v1016_data + (v978_data * v1014_data));
          float v1019_data = s0[213];
          float v1021_data = ir1[8];
          ir1[8] = (v1021_data + (v978_data * v1019_data));
          float v1023_data = r0[22];
          float v1024_data = s0[22];
          float v1026_data = ir1[0];
          ir1[0] = (v1026_data + (v1023_data * v1024_data));
          float v1029_data = s0[46];
          float v1031_data = ir1[1];
          ir1[1] = (v1031_data + (v1023_data * v1029_data));
          float v1034_data = s0[70];
          float v1036_data = ir1[2];
          ir1[2] = (v1036_data + (v1023_data * v1034_data));
          float v1039_data = s0[94];
          float v1041_data = ir1[3];
          ir1[3] = (v1041_data + (v1023_data * v1039_data));
          float v1044_data = s0[118];
          float v1046_data = ir1[4];
          ir1[4] = (v1046_data + (v1023_data * v1044_data));
          float v1049_data = s0[142];
          float v1051_data = ir1[5];
          ir1[5] = (v1051_data + (v1023_data * v1049_data));
          float v1054_data = s0[166];
          float v1056_data = ir1[6];
          ir1[6] = (v1056_data + (v1023_data * v1054_data));
          float v1059_data = s0[190];
          float v1061_data = ir1[7];
          ir1[7] = (v1061_data + (v1023_data * v1059_data));
          float v1064_data = s0[214];
          float v1066_data = ir1[8];
          ir1[8] = (v1066_data + (v1023_data * v1064_data));
          float v1068_data = r0[23];
          float v1069_data = s0[23];
          float v1071_data = ir1[0];
          ir1[0] = (v1071_data + (v1068_data * v1069_data));
          float v1074_data = s0[47];
          float v1076_data = ir1[1];
          ir1[1] = (v1076_data + (v1068_data * v1074_data));
          float v1079_data = s0[71];
          float v1081_data = ir1[2];
          ir1[2] = (v1081_data + (v1068_data * v1079_data));
          float v1084_data = s0[95];
          float v1086_data = ir1[3];
          ir1[3] = (v1086_data + (v1068_data * v1084_data));
          float v1089_data = s0[119];
          float v1091_data = ir1[4];
          ir1[4] = (v1091_data + (v1068_data * v1089_data));
          float v1094_data = s0[143];
          float v1096_data = ir1[5];
          ir1[5] = (v1096_data + (v1068_data * v1094_data));
          float v1099_data = s0[167];
          float v1101_data = ir1[6];
          ir1[6] = (v1101_data + (v1068_data * v1099_data));
          float v1104_data = s0[191];
          float v1106_data = ir1[7];
          ir1[7] = (v1106_data + (v1068_data * v1104_data));
          float v1109_data = s0[215];
          float v1111_data = ir1[8];
          ir1[8] = (v1111_data + (v1068_data * v1109_data));
          if (v20_g) {
            #pragma unroll
            for (int32_t v1113_n1 = 0; v1113_n1 < 9; ++v1113_n1) {
              float v1115_data = ir1[v1113_n1];
              r1[v1113_n1] = v1115_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v20_g) {
            #pragma unroll
            for (int32_t v1116_i1 = 0; v1116_i1 < 9; ++v1116_i1) {
              float v1118_data = r1[v1116_i1];
              glb_m0[(v19_lead + (v1116_i1 * 24))] = v1118_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

