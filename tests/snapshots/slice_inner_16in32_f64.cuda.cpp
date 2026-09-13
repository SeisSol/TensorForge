// === base name ===
kernel_3a043fc54a8f0dc0

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_3a043fc54a8f0dc0 = {{16, 8, 1}, 16, 16, 1, 8, 9216, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_3a043fc54a8f0dc0(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_3a043fc54a8f0dc0(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_3a043fc54a8f0dc0(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_3a043fc54a8f0dc0, block.x * block.y * block.z, 1152 * sizeof(double));
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
  config.sharedMemBytes = 1152 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_3a043fc54a8f0dc0(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_3a043fc54a8f0dc0(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_3a043fc54a8f0dc0, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_3a043fc54a8f0dc0<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_3a043fc54a8f0dc0(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 8 per block = block 16x8x1, 9216 B shared, occupancy grid
    // operands:
    //   m0 16×8(16×8) {0..16}×{0..8} strided
    //   m1 32×32(32×32) {0..32}×{0..32} strided
    //   m2 16×8(16×8) {0..16}×{0..8} strided
    // operations:
    //   m0[i,j] = m1[i,k]@{8..24}×{8..24} × m2[k,j]
    // tensorforge-meta: {"fp":"double","launch":{"active_threads":16,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1152}],"shared_bytes":9216,"shared_elements":1152,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,8]],"name":"m0","ordered":false,"parts":1,"shape":[16,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[32,32]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[8,8],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[144 * threadIdx.y + 0];
      double* tempShrMem = &localShrMem0[128];
      double * __restrict__ s0 = &localShrMem0[0];
      for (size_t v5_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v5_batchId0 < numElements0; v5_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v6_ahead1 = v5_batchId0 + (gridDim.x * blockDim.y);
        size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[v5_batchId0 * 128 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m1 = &m1[v5_batchId0 * 1024 + 0 + m1_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[v5_batchId0 * 128 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v19_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v20_i0 = 0; v20_i0 < 1; ++v20_i0) {
            int32_t v24_off = (v19_lead + (v20_i0 * 16)) + 8;
            #pragma unroll
            for (int32_t v21_i1 = 8; v21_i1 < 24; ++v21_i1) {
              double v27_data = __ldcg(&glb_m1[(v24_off + (v21_i1 * 32))]);
              r0[(v20_i0 + (v21_i1 - 8))] = v27_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 8; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 8);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          double r1[8]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // r1 = +(r0 * s0) + None
          // [(0, 16), (0, 8)] [(0, 16)]
          double ir1[8]{};
          double v33_data = r0[0];
          double v34_data = s0[0];
          double v36_data = ir1[0];
          ir1[0] = (v36_data + (v33_data * v34_data));
          double v39_data = s0[16];
          double v41_data = ir1[1];
          ir1[1] = (v41_data + (v33_data * v39_data));
          double v44_data = s0[32];
          double v46_data = ir1[2];
          ir1[2] = (v46_data + (v33_data * v44_data));
          double v49_data = s0[48];
          double v51_data = ir1[3];
          ir1[3] = (v51_data + (v33_data * v49_data));
          double v54_data = s0[64];
          double v56_data = ir1[4];
          ir1[4] = (v56_data + (v33_data * v54_data));
          double v59_data = s0[80];
          double v61_data = ir1[5];
          ir1[5] = (v61_data + (v33_data * v59_data));
          double v64_data = s0[96];
          double v66_data = ir1[6];
          ir1[6] = (v66_data + (v33_data * v64_data));
          double v69_data = s0[112];
          double v71_data = ir1[7];
          ir1[7] = (v71_data + (v33_data * v69_data));
          double v73_data = r0[1];
          double v74_data = s0[1];
          double v76_data = ir1[0];
          ir1[0] = (v76_data + (v73_data * v74_data));
          double v79_data = s0[17];
          double v81_data = ir1[1];
          ir1[1] = (v81_data + (v73_data * v79_data));
          double v84_data = s0[33];
          double v86_data = ir1[2];
          ir1[2] = (v86_data + (v73_data * v84_data));
          double v89_data = s0[49];
          double v91_data = ir1[3];
          ir1[3] = (v91_data + (v73_data * v89_data));
          double v94_data = s0[65];
          double v96_data = ir1[4];
          ir1[4] = (v96_data + (v73_data * v94_data));
          double v99_data = s0[81];
          double v101_data = ir1[5];
          ir1[5] = (v101_data + (v73_data * v99_data));
          double v104_data = s0[97];
          double v106_data = ir1[6];
          ir1[6] = (v106_data + (v73_data * v104_data));
          double v109_data = s0[113];
          double v111_data = ir1[7];
          ir1[7] = (v111_data + (v73_data * v109_data));
          double v113_data = r0[2];
          double v114_data = s0[2];
          double v116_data = ir1[0];
          ir1[0] = (v116_data + (v113_data * v114_data));
          double v119_data = s0[18];
          double v121_data = ir1[1];
          ir1[1] = (v121_data + (v113_data * v119_data));
          double v124_data = s0[34];
          double v126_data = ir1[2];
          ir1[2] = (v126_data + (v113_data * v124_data));
          double v129_data = s0[50];
          double v131_data = ir1[3];
          ir1[3] = (v131_data + (v113_data * v129_data));
          double v134_data = s0[66];
          double v136_data = ir1[4];
          ir1[4] = (v136_data + (v113_data * v134_data));
          double v139_data = s0[82];
          double v141_data = ir1[5];
          ir1[5] = (v141_data + (v113_data * v139_data));
          double v144_data = s0[98];
          double v146_data = ir1[6];
          ir1[6] = (v146_data + (v113_data * v144_data));
          double v149_data = s0[114];
          double v151_data = ir1[7];
          ir1[7] = (v151_data + (v113_data * v149_data));
          double v153_data = r0[3];
          double v154_data = s0[3];
          double v156_data = ir1[0];
          ir1[0] = (v156_data + (v153_data * v154_data));
          double v159_data = s0[19];
          double v161_data = ir1[1];
          ir1[1] = (v161_data + (v153_data * v159_data));
          double v164_data = s0[35];
          double v166_data = ir1[2];
          ir1[2] = (v166_data + (v153_data * v164_data));
          double v169_data = s0[51];
          double v171_data = ir1[3];
          ir1[3] = (v171_data + (v153_data * v169_data));
          double v174_data = s0[67];
          double v176_data = ir1[4];
          ir1[4] = (v176_data + (v153_data * v174_data));
          double v179_data = s0[83];
          double v181_data = ir1[5];
          ir1[5] = (v181_data + (v153_data * v179_data));
          double v184_data = s0[99];
          double v186_data = ir1[6];
          ir1[6] = (v186_data + (v153_data * v184_data));
          double v189_data = s0[115];
          double v191_data = ir1[7];
          ir1[7] = (v191_data + (v153_data * v189_data));
          double v193_data = r0[4];
          double v194_data = s0[4];
          double v196_data = ir1[0];
          ir1[0] = (v196_data + (v193_data * v194_data));
          double v199_data = s0[20];
          double v201_data = ir1[1];
          ir1[1] = (v201_data + (v193_data * v199_data));
          double v204_data = s0[36];
          double v206_data = ir1[2];
          ir1[2] = (v206_data + (v193_data * v204_data));
          double v209_data = s0[52];
          double v211_data = ir1[3];
          ir1[3] = (v211_data + (v193_data * v209_data));
          double v214_data = s0[68];
          double v216_data = ir1[4];
          ir1[4] = (v216_data + (v193_data * v214_data));
          double v219_data = s0[84];
          double v221_data = ir1[5];
          ir1[5] = (v221_data + (v193_data * v219_data));
          double v224_data = s0[100];
          double v226_data = ir1[6];
          ir1[6] = (v226_data + (v193_data * v224_data));
          double v229_data = s0[116];
          double v231_data = ir1[7];
          ir1[7] = (v231_data + (v193_data * v229_data));
          double v233_data = r0[5];
          double v234_data = s0[5];
          double v236_data = ir1[0];
          ir1[0] = (v236_data + (v233_data * v234_data));
          double v239_data = s0[21];
          double v241_data = ir1[1];
          ir1[1] = (v241_data + (v233_data * v239_data));
          double v244_data = s0[37];
          double v246_data = ir1[2];
          ir1[2] = (v246_data + (v233_data * v244_data));
          double v249_data = s0[53];
          double v251_data = ir1[3];
          ir1[3] = (v251_data + (v233_data * v249_data));
          double v254_data = s0[69];
          double v256_data = ir1[4];
          ir1[4] = (v256_data + (v233_data * v254_data));
          double v259_data = s0[85];
          double v261_data = ir1[5];
          ir1[5] = (v261_data + (v233_data * v259_data));
          double v264_data = s0[101];
          double v266_data = ir1[6];
          ir1[6] = (v266_data + (v233_data * v264_data));
          double v269_data = s0[117];
          double v271_data = ir1[7];
          ir1[7] = (v271_data + (v233_data * v269_data));
          double v273_data = r0[6];
          double v274_data = s0[6];
          double v276_data = ir1[0];
          ir1[0] = (v276_data + (v273_data * v274_data));
          double v279_data = s0[22];
          double v281_data = ir1[1];
          ir1[1] = (v281_data + (v273_data * v279_data));
          double v284_data = s0[38];
          double v286_data = ir1[2];
          ir1[2] = (v286_data + (v273_data * v284_data));
          double v289_data = s0[54];
          double v291_data = ir1[3];
          ir1[3] = (v291_data + (v273_data * v289_data));
          double v294_data = s0[70];
          double v296_data = ir1[4];
          ir1[4] = (v296_data + (v273_data * v294_data));
          double v299_data = s0[86];
          double v301_data = ir1[5];
          ir1[5] = (v301_data + (v273_data * v299_data));
          double v304_data = s0[102];
          double v306_data = ir1[6];
          ir1[6] = (v306_data + (v273_data * v304_data));
          double v309_data = s0[118];
          double v311_data = ir1[7];
          ir1[7] = (v311_data + (v273_data * v309_data));
          double v313_data = r0[7];
          double v314_data = s0[7];
          double v316_data = ir1[0];
          ir1[0] = (v316_data + (v313_data * v314_data));
          double v319_data = s0[23];
          double v321_data = ir1[1];
          ir1[1] = (v321_data + (v313_data * v319_data));
          double v324_data = s0[39];
          double v326_data = ir1[2];
          ir1[2] = (v326_data + (v313_data * v324_data));
          double v329_data = s0[55];
          double v331_data = ir1[3];
          ir1[3] = (v331_data + (v313_data * v329_data));
          double v334_data = s0[71];
          double v336_data = ir1[4];
          ir1[4] = (v336_data + (v313_data * v334_data));
          double v339_data = s0[87];
          double v341_data = ir1[5];
          ir1[5] = (v341_data + (v313_data * v339_data));
          double v344_data = s0[103];
          double v346_data = ir1[6];
          ir1[6] = (v346_data + (v313_data * v344_data));
          double v349_data = s0[119];
          double v351_data = ir1[7];
          ir1[7] = (v351_data + (v313_data * v349_data));
          double v353_data = r0[8];
          double v354_data = s0[8];
          double v356_data = ir1[0];
          ir1[0] = (v356_data + (v353_data * v354_data));
          double v359_data = s0[24];
          double v361_data = ir1[1];
          ir1[1] = (v361_data + (v353_data * v359_data));
          double v364_data = s0[40];
          double v366_data = ir1[2];
          ir1[2] = (v366_data + (v353_data * v364_data));
          double v369_data = s0[56];
          double v371_data = ir1[3];
          ir1[3] = (v371_data + (v353_data * v369_data));
          double v374_data = s0[72];
          double v376_data = ir1[4];
          ir1[4] = (v376_data + (v353_data * v374_data));
          double v379_data = s0[88];
          double v381_data = ir1[5];
          ir1[5] = (v381_data + (v353_data * v379_data));
          double v384_data = s0[104];
          double v386_data = ir1[6];
          ir1[6] = (v386_data + (v353_data * v384_data));
          double v389_data = s0[120];
          double v391_data = ir1[7];
          ir1[7] = (v391_data + (v353_data * v389_data));
          double v393_data = r0[9];
          double v394_data = s0[9];
          double v396_data = ir1[0];
          ir1[0] = (v396_data + (v393_data * v394_data));
          double v399_data = s0[25];
          double v401_data = ir1[1];
          ir1[1] = (v401_data + (v393_data * v399_data));
          double v404_data = s0[41];
          double v406_data = ir1[2];
          ir1[2] = (v406_data + (v393_data * v404_data));
          double v409_data = s0[57];
          double v411_data = ir1[3];
          ir1[3] = (v411_data + (v393_data * v409_data));
          double v414_data = s0[73];
          double v416_data = ir1[4];
          ir1[4] = (v416_data + (v393_data * v414_data));
          double v419_data = s0[89];
          double v421_data = ir1[5];
          ir1[5] = (v421_data + (v393_data * v419_data));
          double v424_data = s0[105];
          double v426_data = ir1[6];
          ir1[6] = (v426_data + (v393_data * v424_data));
          double v429_data = s0[121];
          double v431_data = ir1[7];
          ir1[7] = (v431_data + (v393_data * v429_data));
          double v433_data = r0[10];
          double v434_data = s0[10];
          double v436_data = ir1[0];
          ir1[0] = (v436_data + (v433_data * v434_data));
          double v439_data = s0[26];
          double v441_data = ir1[1];
          ir1[1] = (v441_data + (v433_data * v439_data));
          double v444_data = s0[42];
          double v446_data = ir1[2];
          ir1[2] = (v446_data + (v433_data * v444_data));
          double v449_data = s0[58];
          double v451_data = ir1[3];
          ir1[3] = (v451_data + (v433_data * v449_data));
          double v454_data = s0[74];
          double v456_data = ir1[4];
          ir1[4] = (v456_data + (v433_data * v454_data));
          double v459_data = s0[90];
          double v461_data = ir1[5];
          ir1[5] = (v461_data + (v433_data * v459_data));
          double v464_data = s0[106];
          double v466_data = ir1[6];
          ir1[6] = (v466_data + (v433_data * v464_data));
          double v469_data = s0[122];
          double v471_data = ir1[7];
          ir1[7] = (v471_data + (v433_data * v469_data));
          double v473_data = r0[11];
          double v474_data = s0[11];
          double v476_data = ir1[0];
          ir1[0] = (v476_data + (v473_data * v474_data));
          double v479_data = s0[27];
          double v481_data = ir1[1];
          ir1[1] = (v481_data + (v473_data * v479_data));
          double v484_data = s0[43];
          double v486_data = ir1[2];
          ir1[2] = (v486_data + (v473_data * v484_data));
          double v489_data = s0[59];
          double v491_data = ir1[3];
          ir1[3] = (v491_data + (v473_data * v489_data));
          double v494_data = s0[75];
          double v496_data = ir1[4];
          ir1[4] = (v496_data + (v473_data * v494_data));
          double v499_data = s0[91];
          double v501_data = ir1[5];
          ir1[5] = (v501_data + (v473_data * v499_data));
          double v504_data = s0[107];
          double v506_data = ir1[6];
          ir1[6] = (v506_data + (v473_data * v504_data));
          double v509_data = s0[123];
          double v511_data = ir1[7];
          ir1[7] = (v511_data + (v473_data * v509_data));
          double v513_data = r0[12];
          double v514_data = s0[12];
          double v516_data = ir1[0];
          ir1[0] = (v516_data + (v513_data * v514_data));
          double v519_data = s0[28];
          double v521_data = ir1[1];
          ir1[1] = (v521_data + (v513_data * v519_data));
          double v524_data = s0[44];
          double v526_data = ir1[2];
          ir1[2] = (v526_data + (v513_data * v524_data));
          double v529_data = s0[60];
          double v531_data = ir1[3];
          ir1[3] = (v531_data + (v513_data * v529_data));
          double v534_data = s0[76];
          double v536_data = ir1[4];
          ir1[4] = (v536_data + (v513_data * v534_data));
          double v539_data = s0[92];
          double v541_data = ir1[5];
          ir1[5] = (v541_data + (v513_data * v539_data));
          double v544_data = s0[108];
          double v546_data = ir1[6];
          ir1[6] = (v546_data + (v513_data * v544_data));
          double v549_data = s0[124];
          double v551_data = ir1[7];
          ir1[7] = (v551_data + (v513_data * v549_data));
          double v553_data = r0[13];
          double v554_data = s0[13];
          double v556_data = ir1[0];
          ir1[0] = (v556_data + (v553_data * v554_data));
          double v559_data = s0[29];
          double v561_data = ir1[1];
          ir1[1] = (v561_data + (v553_data * v559_data));
          double v564_data = s0[45];
          double v566_data = ir1[2];
          ir1[2] = (v566_data + (v553_data * v564_data));
          double v569_data = s0[61];
          double v571_data = ir1[3];
          ir1[3] = (v571_data + (v553_data * v569_data));
          double v574_data = s0[77];
          double v576_data = ir1[4];
          ir1[4] = (v576_data + (v553_data * v574_data));
          double v579_data = s0[93];
          double v581_data = ir1[5];
          ir1[5] = (v581_data + (v553_data * v579_data));
          double v584_data = s0[109];
          double v586_data = ir1[6];
          ir1[6] = (v586_data + (v553_data * v584_data));
          double v589_data = s0[125];
          double v591_data = ir1[7];
          ir1[7] = (v591_data + (v553_data * v589_data));
          double v593_data = r0[14];
          double v594_data = s0[14];
          double v596_data = ir1[0];
          ir1[0] = (v596_data + (v593_data * v594_data));
          double v599_data = s0[30];
          double v601_data = ir1[1];
          ir1[1] = (v601_data + (v593_data * v599_data));
          double v604_data = s0[46];
          double v606_data = ir1[2];
          ir1[2] = (v606_data + (v593_data * v604_data));
          double v609_data = s0[62];
          double v611_data = ir1[3];
          ir1[3] = (v611_data + (v593_data * v609_data));
          double v614_data = s0[78];
          double v616_data = ir1[4];
          ir1[4] = (v616_data + (v593_data * v614_data));
          double v619_data = s0[94];
          double v621_data = ir1[5];
          ir1[5] = (v621_data + (v593_data * v619_data));
          double v624_data = s0[110];
          double v626_data = ir1[6];
          ir1[6] = (v626_data + (v593_data * v624_data));
          double v629_data = s0[126];
          double v631_data = ir1[7];
          ir1[7] = (v631_data + (v593_data * v629_data));
          double v633_data = r0[15];
          double v634_data = s0[15];
          double v636_data = ir1[0];
          ir1[0] = (v636_data + (v633_data * v634_data));
          double v639_data = s0[31];
          double v641_data = ir1[1];
          ir1[1] = (v641_data + (v633_data * v639_data));
          double v644_data = s0[47];
          double v646_data = ir1[2];
          ir1[2] = (v646_data + (v633_data * v644_data));
          double v649_data = s0[63];
          double v651_data = ir1[3];
          ir1[3] = (v651_data + (v633_data * v649_data));
          double v654_data = s0[79];
          double v656_data = ir1[4];
          ir1[4] = (v656_data + (v633_data * v654_data));
          double v659_data = s0[95];
          double v661_data = ir1[5];
          ir1[5] = (v661_data + (v633_data * v659_data));
          double v664_data = s0[111];
          double v666_data = ir1[6];
          ir1[6] = (v666_data + (v633_data * v664_data));
          double v669_data = s0[127];
          double v671_data = ir1[7];
          ir1[7] = (v671_data + (v633_data * v669_data));
          #pragma unroll
          for (int32_t v673_n0 = 0; v673_n0 < 1; ++v673_n0) {
            #pragma unroll
            for (int32_t v674_n1 = 0; v674_n1 < 8; ++v674_n1) {
              int32_t v675_a = v673_n0 + v674_n1;
              double v676_data = ir1[v675_a];
              r1[v675_a] = v676_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v677_i0 = 0; v677_i0 < 1; ++v677_i0) {
            int32_t v682_lead = v19_lead + (v677_i0 * 16);
            #pragma unroll
            for (int32_t v678_i1 = 0; v678_i1 < 8; ++v678_i1) {
              double v680_data = r1[(v677_i0 + v678_i1)];
              glb_m0[(v682_lead + (v678_i1 * 16))] = v680_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

