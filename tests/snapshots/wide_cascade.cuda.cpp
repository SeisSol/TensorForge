// === base name ===
kernel_35ec5bc5f44398a1

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_35ec5bc5f44398a1 = {{16, 8, 1}, 16, 16, 1, 8, 6656, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_35ec5bc5f44398a1(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_35ec5bc5f44398a1(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_35ec5bc5f44398a1(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_35ec5bc5f44398a1, block.x * block.y * block.z, 1664 * sizeof(float));
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
  config.sharedMemBytes = 1664 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_35ec5bc5f44398a1(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_35ec5bc5f44398a1(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_35ec5bc5f44398a1, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_35ec5bc5f44398a1<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_35ec5bc5f44398a1(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 8 per block = block 16x8x1, 6656 B shared, occupancy grid
    // operands:
    //   m0 16×11(16×11) {0..16}×{0..11} strided
    //   m1 16×16(16×16) {0..16}×{0..16} strided
    //   m2 16×11(16×11) {0..16}×{0..11} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1664}],"shared_bytes":6656,"shared_elements":1664,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[16,11]],"name":"m0","ordered":false,"parts":1,"shape":[16,11],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,11]],"name":"m2","ordered":false,"parts":1,"shape":[16,11],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,11]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,11]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,11]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,11]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[208 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[192];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v5_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v5_batchId0 < numElements0; v5_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v6_ahead1 = v5_batchId0 + (gridDim.x * blockDim.y);
        size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 176 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 176 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v19_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v20_i0 = 0; v20_i0 < 1; ++v20_i0) {
            int32_t v23_lead = v19_lead + (v20_i0 * 16);
            #pragma unroll
            for (int32_t v21_i1 = 0; v21_i1 < 16; ++v21_i1) {
              float v26_data = __ldcg(&glb_m1[(v23_lead + (v21_i1 * 16))]);
              r0[(v20_i0 + v21_i1)] = v26_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 4 * threadIdx.x + 0], &glb_m2[0 + 0 + 4 * threadIdx.x + 0], 16);
          __pipeline_memcpy_async(&s0[0 + 0 + 4 * threadIdx.x + 64], &glb_m2[0 + 0 + 4 * threadIdx.x + 64], 16);
          if (threadIdx.x < 12) {
            __pipeline_memcpy_async(&s0[0 + 0 + 4 * threadIdx.x + 128], &glb_m2[0 + 0 + 4 * threadIdx.x + 128], 16);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[11]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // ir1 = +(r0 * s0)
          // [(0, 16), (0, 11)] [(0, 16)]
          float ir1[11]{};
          float v33_data = r0[0];
          float v34_data = s0[0];
          float v36_data = ir1[0];
          ir1[0] = (v36_data + (v33_data * v34_data));
          float v39_data = s0[16];
          float v41_data = ir1[1];
          ir1[1] = (v41_data + (v33_data * v39_data));
          float v44_data = s0[32];
          float v46_data = ir1[2];
          ir1[2] = (v46_data + (v33_data * v44_data));
          float v49_data = s0[48];
          float v51_data = ir1[3];
          ir1[3] = (v51_data + (v33_data * v49_data));
          float v54_data = s0[64];
          float v56_data = ir1[4];
          ir1[4] = (v56_data + (v33_data * v54_data));
          float v59_data = s0[80];
          float v61_data = ir1[5];
          ir1[5] = (v61_data + (v33_data * v59_data));
          float v64_data = s0[96];
          float v66_data = ir1[6];
          ir1[6] = (v66_data + (v33_data * v64_data));
          float v69_data = s0[112];
          float v71_data = ir1[7];
          ir1[7] = (v71_data + (v33_data * v69_data));
          float v74_data = s0[128];
          float v76_data = ir1[8];
          ir1[8] = (v76_data + (v33_data * v74_data));
          float v79_data = s0[144];
          float v81_data = ir1[9];
          ir1[9] = (v81_data + (v33_data * v79_data));
          float v84_data = s0[160];
          float v86_data = ir1[10];
          ir1[10] = (v86_data + (v33_data * v84_data));
          float v88_data = r0[1];
          float v89_data = s0[1];
          float v91_data = ir1[0];
          ir1[0] = (v91_data + (v88_data * v89_data));
          float v94_data = s0[17];
          float v96_data = ir1[1];
          ir1[1] = (v96_data + (v88_data * v94_data));
          float v99_data = s0[33];
          float v101_data = ir1[2];
          ir1[2] = (v101_data + (v88_data * v99_data));
          float v104_data = s0[49];
          float v106_data = ir1[3];
          ir1[3] = (v106_data + (v88_data * v104_data));
          float v109_data = s0[65];
          float v111_data = ir1[4];
          ir1[4] = (v111_data + (v88_data * v109_data));
          float v114_data = s0[81];
          float v116_data = ir1[5];
          ir1[5] = (v116_data + (v88_data * v114_data));
          float v119_data = s0[97];
          float v121_data = ir1[6];
          ir1[6] = (v121_data + (v88_data * v119_data));
          float v124_data = s0[113];
          float v126_data = ir1[7];
          ir1[7] = (v126_data + (v88_data * v124_data));
          float v129_data = s0[129];
          float v131_data = ir1[8];
          ir1[8] = (v131_data + (v88_data * v129_data));
          float v134_data = s0[145];
          float v136_data = ir1[9];
          ir1[9] = (v136_data + (v88_data * v134_data));
          float v139_data = s0[161];
          float v141_data = ir1[10];
          ir1[10] = (v141_data + (v88_data * v139_data));
          float v143_data = r0[2];
          float v144_data = s0[2];
          float v146_data = ir1[0];
          ir1[0] = (v146_data + (v143_data * v144_data));
          float v149_data = s0[18];
          float v151_data = ir1[1];
          ir1[1] = (v151_data + (v143_data * v149_data));
          float v154_data = s0[34];
          float v156_data = ir1[2];
          ir1[2] = (v156_data + (v143_data * v154_data));
          float v159_data = s0[50];
          float v161_data = ir1[3];
          ir1[3] = (v161_data + (v143_data * v159_data));
          float v164_data = s0[66];
          float v166_data = ir1[4];
          ir1[4] = (v166_data + (v143_data * v164_data));
          float v169_data = s0[82];
          float v171_data = ir1[5];
          ir1[5] = (v171_data + (v143_data * v169_data));
          float v174_data = s0[98];
          float v176_data = ir1[6];
          ir1[6] = (v176_data + (v143_data * v174_data));
          float v179_data = s0[114];
          float v181_data = ir1[7];
          ir1[7] = (v181_data + (v143_data * v179_data));
          float v184_data = s0[130];
          float v186_data = ir1[8];
          ir1[8] = (v186_data + (v143_data * v184_data));
          float v189_data = s0[146];
          float v191_data = ir1[9];
          ir1[9] = (v191_data + (v143_data * v189_data));
          float v194_data = s0[162];
          float v196_data = ir1[10];
          ir1[10] = (v196_data + (v143_data * v194_data));
          float v198_data = r0[3];
          float v199_data = s0[3];
          float v201_data = ir1[0];
          ir1[0] = (v201_data + (v198_data * v199_data));
          float v204_data = s0[19];
          float v206_data = ir1[1];
          ir1[1] = (v206_data + (v198_data * v204_data));
          float v209_data = s0[35];
          float v211_data = ir1[2];
          ir1[2] = (v211_data + (v198_data * v209_data));
          float v214_data = s0[51];
          float v216_data = ir1[3];
          ir1[3] = (v216_data + (v198_data * v214_data));
          float v219_data = s0[67];
          float v221_data = ir1[4];
          ir1[4] = (v221_data + (v198_data * v219_data));
          float v224_data = s0[83];
          float v226_data = ir1[5];
          ir1[5] = (v226_data + (v198_data * v224_data));
          float v229_data = s0[99];
          float v231_data = ir1[6];
          ir1[6] = (v231_data + (v198_data * v229_data));
          float v234_data = s0[115];
          float v236_data = ir1[7];
          ir1[7] = (v236_data + (v198_data * v234_data));
          float v239_data = s0[131];
          float v241_data = ir1[8];
          ir1[8] = (v241_data + (v198_data * v239_data));
          float v244_data = s0[147];
          float v246_data = ir1[9];
          ir1[9] = (v246_data + (v198_data * v244_data));
          float v249_data = s0[163];
          float v251_data = ir1[10];
          ir1[10] = (v251_data + (v198_data * v249_data));
          float v253_data = r0[4];
          float v254_data = s0[4];
          float v256_data = ir1[0];
          ir1[0] = (v256_data + (v253_data * v254_data));
          float v259_data = s0[20];
          float v261_data = ir1[1];
          ir1[1] = (v261_data + (v253_data * v259_data));
          float v264_data = s0[36];
          float v266_data = ir1[2];
          ir1[2] = (v266_data + (v253_data * v264_data));
          float v269_data = s0[52];
          float v271_data = ir1[3];
          ir1[3] = (v271_data + (v253_data * v269_data));
          float v274_data = s0[68];
          float v276_data = ir1[4];
          ir1[4] = (v276_data + (v253_data * v274_data));
          float v279_data = s0[84];
          float v281_data = ir1[5];
          ir1[5] = (v281_data + (v253_data * v279_data));
          float v284_data = s0[100];
          float v286_data = ir1[6];
          ir1[6] = (v286_data + (v253_data * v284_data));
          float v289_data = s0[116];
          float v291_data = ir1[7];
          ir1[7] = (v291_data + (v253_data * v289_data));
          float v294_data = s0[132];
          float v296_data = ir1[8];
          ir1[8] = (v296_data + (v253_data * v294_data));
          float v299_data = s0[148];
          float v301_data = ir1[9];
          ir1[9] = (v301_data + (v253_data * v299_data));
          float v304_data = s0[164];
          float v306_data = ir1[10];
          ir1[10] = (v306_data + (v253_data * v304_data));
          float v308_data = r0[5];
          float v309_data = s0[5];
          float v311_data = ir1[0];
          ir1[0] = (v311_data + (v308_data * v309_data));
          float v314_data = s0[21];
          float v316_data = ir1[1];
          ir1[1] = (v316_data + (v308_data * v314_data));
          float v319_data = s0[37];
          float v321_data = ir1[2];
          ir1[2] = (v321_data + (v308_data * v319_data));
          float v324_data = s0[53];
          float v326_data = ir1[3];
          ir1[3] = (v326_data + (v308_data * v324_data));
          float v329_data = s0[69];
          float v331_data = ir1[4];
          ir1[4] = (v331_data + (v308_data * v329_data));
          float v334_data = s0[85];
          float v336_data = ir1[5];
          ir1[5] = (v336_data + (v308_data * v334_data));
          float v339_data = s0[101];
          float v341_data = ir1[6];
          ir1[6] = (v341_data + (v308_data * v339_data));
          float v344_data = s0[117];
          float v346_data = ir1[7];
          ir1[7] = (v346_data + (v308_data * v344_data));
          float v349_data = s0[133];
          float v351_data = ir1[8];
          ir1[8] = (v351_data + (v308_data * v349_data));
          float v354_data = s0[149];
          float v356_data = ir1[9];
          ir1[9] = (v356_data + (v308_data * v354_data));
          float v359_data = s0[165];
          float v361_data = ir1[10];
          ir1[10] = (v361_data + (v308_data * v359_data));
          float v363_data = r0[6];
          float v364_data = s0[6];
          float v366_data = ir1[0];
          ir1[0] = (v366_data + (v363_data * v364_data));
          float v369_data = s0[22];
          float v371_data = ir1[1];
          ir1[1] = (v371_data + (v363_data * v369_data));
          float v374_data = s0[38];
          float v376_data = ir1[2];
          ir1[2] = (v376_data + (v363_data * v374_data));
          float v379_data = s0[54];
          float v381_data = ir1[3];
          ir1[3] = (v381_data + (v363_data * v379_data));
          float v384_data = s0[70];
          float v386_data = ir1[4];
          ir1[4] = (v386_data + (v363_data * v384_data));
          float v389_data = s0[86];
          float v391_data = ir1[5];
          ir1[5] = (v391_data + (v363_data * v389_data));
          float v394_data = s0[102];
          float v396_data = ir1[6];
          ir1[6] = (v396_data + (v363_data * v394_data));
          float v399_data = s0[118];
          float v401_data = ir1[7];
          ir1[7] = (v401_data + (v363_data * v399_data));
          float v404_data = s0[134];
          float v406_data = ir1[8];
          ir1[8] = (v406_data + (v363_data * v404_data));
          float v409_data = s0[150];
          float v411_data = ir1[9];
          ir1[9] = (v411_data + (v363_data * v409_data));
          float v414_data = s0[166];
          float v416_data = ir1[10];
          ir1[10] = (v416_data + (v363_data * v414_data));
          float v418_data = r0[7];
          float v419_data = s0[7];
          float v421_data = ir1[0];
          ir1[0] = (v421_data + (v418_data * v419_data));
          float v424_data = s0[23];
          float v426_data = ir1[1];
          ir1[1] = (v426_data + (v418_data * v424_data));
          float v429_data = s0[39];
          float v431_data = ir1[2];
          ir1[2] = (v431_data + (v418_data * v429_data));
          float v434_data = s0[55];
          float v436_data = ir1[3];
          ir1[3] = (v436_data + (v418_data * v434_data));
          float v439_data = s0[71];
          float v441_data = ir1[4];
          ir1[4] = (v441_data + (v418_data * v439_data));
          float v444_data = s0[87];
          float v446_data = ir1[5];
          ir1[5] = (v446_data + (v418_data * v444_data));
          float v449_data = s0[103];
          float v451_data = ir1[6];
          ir1[6] = (v451_data + (v418_data * v449_data));
          float v454_data = s0[119];
          float v456_data = ir1[7];
          ir1[7] = (v456_data + (v418_data * v454_data));
          float v459_data = s0[135];
          float v461_data = ir1[8];
          ir1[8] = (v461_data + (v418_data * v459_data));
          float v464_data = s0[151];
          float v466_data = ir1[9];
          ir1[9] = (v466_data + (v418_data * v464_data));
          float v469_data = s0[167];
          float v471_data = ir1[10];
          ir1[10] = (v471_data + (v418_data * v469_data));
          float v473_data = r0[8];
          float v474_data = s0[8];
          float v476_data = ir1[0];
          ir1[0] = (v476_data + (v473_data * v474_data));
          float v479_data = s0[24];
          float v481_data = ir1[1];
          ir1[1] = (v481_data + (v473_data * v479_data));
          float v484_data = s0[40];
          float v486_data = ir1[2];
          ir1[2] = (v486_data + (v473_data * v484_data));
          float v489_data = s0[56];
          float v491_data = ir1[3];
          ir1[3] = (v491_data + (v473_data * v489_data));
          float v494_data = s0[72];
          float v496_data = ir1[4];
          ir1[4] = (v496_data + (v473_data * v494_data));
          float v499_data = s0[88];
          float v501_data = ir1[5];
          ir1[5] = (v501_data + (v473_data * v499_data));
          float v504_data = s0[104];
          float v506_data = ir1[6];
          ir1[6] = (v506_data + (v473_data * v504_data));
          float v509_data = s0[120];
          float v511_data = ir1[7];
          ir1[7] = (v511_data + (v473_data * v509_data));
          float v514_data = s0[136];
          float v516_data = ir1[8];
          ir1[8] = (v516_data + (v473_data * v514_data));
          float v519_data = s0[152];
          float v521_data = ir1[9];
          ir1[9] = (v521_data + (v473_data * v519_data));
          float v524_data = s0[168];
          float v526_data = ir1[10];
          ir1[10] = (v526_data + (v473_data * v524_data));
          float v528_data = r0[9];
          float v529_data = s0[9];
          float v531_data = ir1[0];
          ir1[0] = (v531_data + (v528_data * v529_data));
          float v534_data = s0[25];
          float v536_data = ir1[1];
          ir1[1] = (v536_data + (v528_data * v534_data));
          float v539_data = s0[41];
          float v541_data = ir1[2];
          ir1[2] = (v541_data + (v528_data * v539_data));
          float v544_data = s0[57];
          float v546_data = ir1[3];
          ir1[3] = (v546_data + (v528_data * v544_data));
          float v549_data = s0[73];
          float v551_data = ir1[4];
          ir1[4] = (v551_data + (v528_data * v549_data));
          float v554_data = s0[89];
          float v556_data = ir1[5];
          ir1[5] = (v556_data + (v528_data * v554_data));
          float v559_data = s0[105];
          float v561_data = ir1[6];
          ir1[6] = (v561_data + (v528_data * v559_data));
          float v564_data = s0[121];
          float v566_data = ir1[7];
          ir1[7] = (v566_data + (v528_data * v564_data));
          float v569_data = s0[137];
          float v571_data = ir1[8];
          ir1[8] = (v571_data + (v528_data * v569_data));
          float v574_data = s0[153];
          float v576_data = ir1[9];
          ir1[9] = (v576_data + (v528_data * v574_data));
          float v579_data = s0[169];
          float v581_data = ir1[10];
          ir1[10] = (v581_data + (v528_data * v579_data));
          float v583_data = r0[10];
          float v584_data = s0[10];
          float v586_data = ir1[0];
          ir1[0] = (v586_data + (v583_data * v584_data));
          float v589_data = s0[26];
          float v591_data = ir1[1];
          ir1[1] = (v591_data + (v583_data * v589_data));
          float v594_data = s0[42];
          float v596_data = ir1[2];
          ir1[2] = (v596_data + (v583_data * v594_data));
          float v599_data = s0[58];
          float v601_data = ir1[3];
          ir1[3] = (v601_data + (v583_data * v599_data));
          float v604_data = s0[74];
          float v606_data = ir1[4];
          ir1[4] = (v606_data + (v583_data * v604_data));
          float v609_data = s0[90];
          float v611_data = ir1[5];
          ir1[5] = (v611_data + (v583_data * v609_data));
          float v614_data = s0[106];
          float v616_data = ir1[6];
          ir1[6] = (v616_data + (v583_data * v614_data));
          float v619_data = s0[122];
          float v621_data = ir1[7];
          ir1[7] = (v621_data + (v583_data * v619_data));
          float v624_data = s0[138];
          float v626_data = ir1[8];
          ir1[8] = (v626_data + (v583_data * v624_data));
          float v629_data = s0[154];
          float v631_data = ir1[9];
          ir1[9] = (v631_data + (v583_data * v629_data));
          float v634_data = s0[170];
          float v636_data = ir1[10];
          ir1[10] = (v636_data + (v583_data * v634_data));
          float v638_data = r0[11];
          float v639_data = s0[11];
          float v641_data = ir1[0];
          ir1[0] = (v641_data + (v638_data * v639_data));
          float v644_data = s0[27];
          float v646_data = ir1[1];
          ir1[1] = (v646_data + (v638_data * v644_data));
          float v649_data = s0[43];
          float v651_data = ir1[2];
          ir1[2] = (v651_data + (v638_data * v649_data));
          float v654_data = s0[59];
          float v656_data = ir1[3];
          ir1[3] = (v656_data + (v638_data * v654_data));
          float v659_data = s0[75];
          float v661_data = ir1[4];
          ir1[4] = (v661_data + (v638_data * v659_data));
          float v664_data = s0[91];
          float v666_data = ir1[5];
          ir1[5] = (v666_data + (v638_data * v664_data));
          float v669_data = s0[107];
          float v671_data = ir1[6];
          ir1[6] = (v671_data + (v638_data * v669_data));
          float v674_data = s0[123];
          float v676_data = ir1[7];
          ir1[7] = (v676_data + (v638_data * v674_data));
          float v679_data = s0[139];
          float v681_data = ir1[8];
          ir1[8] = (v681_data + (v638_data * v679_data));
          float v684_data = s0[155];
          float v686_data = ir1[9];
          ir1[9] = (v686_data + (v638_data * v684_data));
          float v689_data = s0[171];
          float v691_data = ir1[10];
          ir1[10] = (v691_data + (v638_data * v689_data));
          float v693_data = r0[12];
          float v694_data = s0[12];
          float v696_data = ir1[0];
          ir1[0] = (v696_data + (v693_data * v694_data));
          float v699_data = s0[28];
          float v701_data = ir1[1];
          ir1[1] = (v701_data + (v693_data * v699_data));
          float v704_data = s0[44];
          float v706_data = ir1[2];
          ir1[2] = (v706_data + (v693_data * v704_data));
          float v709_data = s0[60];
          float v711_data = ir1[3];
          ir1[3] = (v711_data + (v693_data * v709_data));
          float v714_data = s0[76];
          float v716_data = ir1[4];
          ir1[4] = (v716_data + (v693_data * v714_data));
          float v719_data = s0[92];
          float v721_data = ir1[5];
          ir1[5] = (v721_data + (v693_data * v719_data));
          float v724_data = s0[108];
          float v726_data = ir1[6];
          ir1[6] = (v726_data + (v693_data * v724_data));
          float v729_data = s0[124];
          float v731_data = ir1[7];
          ir1[7] = (v731_data + (v693_data * v729_data));
          float v734_data = s0[140];
          float v736_data = ir1[8];
          ir1[8] = (v736_data + (v693_data * v734_data));
          float v739_data = s0[156];
          float v741_data = ir1[9];
          ir1[9] = (v741_data + (v693_data * v739_data));
          float v744_data = s0[172];
          float v746_data = ir1[10];
          ir1[10] = (v746_data + (v693_data * v744_data));
          float v748_data = r0[13];
          float v749_data = s0[13];
          float v751_data = ir1[0];
          ir1[0] = (v751_data + (v748_data * v749_data));
          float v754_data = s0[29];
          float v756_data = ir1[1];
          ir1[1] = (v756_data + (v748_data * v754_data));
          float v759_data = s0[45];
          float v761_data = ir1[2];
          ir1[2] = (v761_data + (v748_data * v759_data));
          float v764_data = s0[61];
          float v766_data = ir1[3];
          ir1[3] = (v766_data + (v748_data * v764_data));
          float v769_data = s0[77];
          float v771_data = ir1[4];
          ir1[4] = (v771_data + (v748_data * v769_data));
          float v774_data = s0[93];
          float v776_data = ir1[5];
          ir1[5] = (v776_data + (v748_data * v774_data));
          float v779_data = s0[109];
          float v781_data = ir1[6];
          ir1[6] = (v781_data + (v748_data * v779_data));
          float v784_data = s0[125];
          float v786_data = ir1[7];
          ir1[7] = (v786_data + (v748_data * v784_data));
          float v789_data = s0[141];
          float v791_data = ir1[8];
          ir1[8] = (v791_data + (v748_data * v789_data));
          float v794_data = s0[157];
          float v796_data = ir1[9];
          ir1[9] = (v796_data + (v748_data * v794_data));
          float v799_data = s0[173];
          float v801_data = ir1[10];
          ir1[10] = (v801_data + (v748_data * v799_data));
          float v803_data = r0[14];
          float v804_data = s0[14];
          float v806_data = ir1[0];
          ir1[0] = (v806_data + (v803_data * v804_data));
          float v809_data = s0[30];
          float v811_data = ir1[1];
          ir1[1] = (v811_data + (v803_data * v809_data));
          float v814_data = s0[46];
          float v816_data = ir1[2];
          ir1[2] = (v816_data + (v803_data * v814_data));
          float v819_data = s0[62];
          float v821_data = ir1[3];
          ir1[3] = (v821_data + (v803_data * v819_data));
          float v824_data = s0[78];
          float v826_data = ir1[4];
          ir1[4] = (v826_data + (v803_data * v824_data));
          float v829_data = s0[94];
          float v831_data = ir1[5];
          ir1[5] = (v831_data + (v803_data * v829_data));
          float v834_data = s0[110];
          float v836_data = ir1[6];
          ir1[6] = (v836_data + (v803_data * v834_data));
          float v839_data = s0[126];
          float v841_data = ir1[7];
          ir1[7] = (v841_data + (v803_data * v839_data));
          float v844_data = s0[142];
          float v846_data = ir1[8];
          ir1[8] = (v846_data + (v803_data * v844_data));
          float v849_data = s0[158];
          float v851_data = ir1[9];
          ir1[9] = (v851_data + (v803_data * v849_data));
          float v854_data = s0[174];
          float v856_data = ir1[10];
          ir1[10] = (v856_data + (v803_data * v854_data));
          float v858_data = r0[15];
          float v859_data = s0[15];
          float v861_data = ir1[0];
          ir1[0] = (v861_data + (v858_data * v859_data));
          float v864_data = s0[31];
          float v866_data = ir1[1];
          ir1[1] = (v866_data + (v858_data * v864_data));
          float v869_data = s0[47];
          float v871_data = ir1[2];
          ir1[2] = (v871_data + (v858_data * v869_data));
          float v874_data = s0[63];
          float v876_data = ir1[3];
          ir1[3] = (v876_data + (v858_data * v874_data));
          float v879_data = s0[79];
          float v881_data = ir1[4];
          ir1[4] = (v881_data + (v858_data * v879_data));
          float v884_data = s0[95];
          float v886_data = ir1[5];
          ir1[5] = (v886_data + (v858_data * v884_data));
          float v889_data = s0[111];
          float v891_data = ir1[6];
          ir1[6] = (v891_data + (v858_data * v889_data));
          float v894_data = s0[127];
          float v896_data = ir1[7];
          ir1[7] = (v896_data + (v858_data * v894_data));
          float v899_data = s0[143];
          float v901_data = ir1[8];
          ir1[8] = (v901_data + (v858_data * v899_data));
          float v904_data = s0[159];
          float v906_data = ir1[9];
          ir1[9] = (v906_data + (v858_data * v904_data));
          float v909_data = s0[175];
          float v911_data = ir1[10];
          ir1[10] = (v911_data + (v858_data * v909_data));
          // r1 = ir1
          #pragma unroll
          for (int32_t v913_n0 = 0; v913_n0 < 1; ++v913_n0) {
            #pragma unroll
            for (int32_t v914_n1 = 0; v914_n1 < 11; ++v914_n1) {
              int32_t v915_a = v913_n0 + v914_n1;
              float v916_data = ir1[v915_a];
              r1[v915_a] = v916_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v917_i0 = 0; v917_i0 < 1; ++v917_i0) {
            int32_t v922_lead = v19_lead + (v917_i0 * 16);
            #pragma unroll
            for (int32_t v918_i1 = 0; v918_i1 < 11; ++v918_i1) {
              float v920_data = r1[(v917_i0 + v918_i1)];
              glb_m0[(v922_lead + (v918_i1 * 16))] = v920_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

