// === base name ===
kernel_9f626a2322d4b6d4

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_9f626a2322d4b6d4 = {{16, 8, 1}, 16, 16, 1, 8, 6656, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_9f626a2322d4b6d4(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_9f626a2322d4b6d4(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_9f626a2322d4b6d4(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_9f626a2322d4b6d4, block.x * block.y * block.z, 1664 * sizeof(float));
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
void launcher_kernel_9f626a2322d4b6d4(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_9f626a2322d4b6d4(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_9f626a2322d4b6d4, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_9f626a2322d4b6d4<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_9f626a2322d4b6d4(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 8 per block = block 16x8x1, 6656 B shared, occupancy grid
    // operands:
    //   m0 16×11(16×11) {0..16}×{0..11} strided
    //   m1 16×16(16×16) {0..16}×{0..16} strided
    //   m2 16×11(16×11) {0..16}×{0..11} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1664}],"shared_bytes":6656,"shared_elements":1664,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[16,11]],"name":"m0","ordered":false,"parts":1,"shape":[16,11],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,11]],"name":"m2","ordered":false,"parts":1,"shape":[16,11],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,11]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,11]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,11]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,11]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[208 * threadIdx.y + 0];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 176 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 176 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v22_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
            int32_t v26_lead = v22_lead + (v23_i0 * 16);
            #pragma unroll
            for (int32_t v24_i1 = 0; v24_i1 < 16; ++v24_i1) {
              float v29_data = __ldcg(&glb_m1[(v26_lead + (v24_i1 * 16))]);
              r0[(v23_i0 + v24_i1)] = v29_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 4 * threadIdx.x + 0], &glb_m2[0 + 0 + 4 * threadIdx.x + 0], 16);
          __pipeline_memcpy_async(&s0[0 + 0 + 4 * threadIdx.x + 64], &glb_m2[0 + 0 + 4 * threadIdx.x + 64], 16);
          if (threadIdx.x < 12) {
            __pipeline_memcpy_async(&s0[0 + 0 + 4 * threadIdx.x + 128], &glb_m2[0 + 0 + 4 * threadIdx.x + 128], 16);
          }
          __pipeline_commit();
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[11]{};
          // ir1 = +(r0 * s0)
          // [(0, 16), (0, 11)] [(0, 16)]
          float ir1[11]{};
          float v36_data = r0[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v37_data = s0[0];
          float v39_data = ir1[0];
          ir1[0] = (v39_data + (v36_data * v37_data));
          float v42_data = s0[16];
          float v44_data = ir1[1];
          ir1[1] = (v44_data + (v36_data * v42_data));
          float v47_data = s0[32];
          float v49_data = ir1[2];
          ir1[2] = (v49_data + (v36_data * v47_data));
          float v52_data = s0[48];
          float v54_data = ir1[3];
          ir1[3] = (v54_data + (v36_data * v52_data));
          float v57_data = s0[64];
          float v59_data = ir1[4];
          ir1[4] = (v59_data + (v36_data * v57_data));
          float v62_data = s0[80];
          float v64_data = ir1[5];
          ir1[5] = (v64_data + (v36_data * v62_data));
          float v67_data = s0[96];
          float v69_data = ir1[6];
          ir1[6] = (v69_data + (v36_data * v67_data));
          float v72_data = s0[112];
          float v74_data = ir1[7];
          ir1[7] = (v74_data + (v36_data * v72_data));
          float v77_data = s0[128];
          float v79_data = ir1[8];
          ir1[8] = (v79_data + (v36_data * v77_data));
          float v82_data = s0[144];
          float v84_data = ir1[9];
          ir1[9] = (v84_data + (v36_data * v82_data));
          float v87_data = s0[160];
          float v89_data = ir1[10];
          ir1[10] = (v89_data + (v36_data * v87_data));
          float v91_data = r0[1];
          float v92_data = s0[1];
          float v94_data = ir1[0];
          ir1[0] = (v94_data + (v91_data * v92_data));
          float v97_data = s0[17];
          float v99_data = ir1[1];
          ir1[1] = (v99_data + (v91_data * v97_data));
          float v102_data = s0[33];
          float v104_data = ir1[2];
          ir1[2] = (v104_data + (v91_data * v102_data));
          float v107_data = s0[49];
          float v109_data = ir1[3];
          ir1[3] = (v109_data + (v91_data * v107_data));
          float v112_data = s0[65];
          float v114_data = ir1[4];
          ir1[4] = (v114_data + (v91_data * v112_data));
          float v117_data = s0[81];
          float v119_data = ir1[5];
          ir1[5] = (v119_data + (v91_data * v117_data));
          float v122_data = s0[97];
          float v124_data = ir1[6];
          ir1[6] = (v124_data + (v91_data * v122_data));
          float v127_data = s0[113];
          float v129_data = ir1[7];
          ir1[7] = (v129_data + (v91_data * v127_data));
          float v132_data = s0[129];
          float v134_data = ir1[8];
          ir1[8] = (v134_data + (v91_data * v132_data));
          float v137_data = s0[145];
          float v139_data = ir1[9];
          ir1[9] = (v139_data + (v91_data * v137_data));
          float v142_data = s0[161];
          float v144_data = ir1[10];
          ir1[10] = (v144_data + (v91_data * v142_data));
          float v146_data = r0[2];
          float v147_data = s0[2];
          float v149_data = ir1[0];
          ir1[0] = (v149_data + (v146_data * v147_data));
          float v152_data = s0[18];
          float v154_data = ir1[1];
          ir1[1] = (v154_data + (v146_data * v152_data));
          float v157_data = s0[34];
          float v159_data = ir1[2];
          ir1[2] = (v159_data + (v146_data * v157_data));
          float v162_data = s0[50];
          float v164_data = ir1[3];
          ir1[3] = (v164_data + (v146_data * v162_data));
          float v167_data = s0[66];
          float v169_data = ir1[4];
          ir1[4] = (v169_data + (v146_data * v167_data));
          float v172_data = s0[82];
          float v174_data = ir1[5];
          ir1[5] = (v174_data + (v146_data * v172_data));
          float v177_data = s0[98];
          float v179_data = ir1[6];
          ir1[6] = (v179_data + (v146_data * v177_data));
          float v182_data = s0[114];
          float v184_data = ir1[7];
          ir1[7] = (v184_data + (v146_data * v182_data));
          float v187_data = s0[130];
          float v189_data = ir1[8];
          ir1[8] = (v189_data + (v146_data * v187_data));
          float v192_data = s0[146];
          float v194_data = ir1[9];
          ir1[9] = (v194_data + (v146_data * v192_data));
          float v197_data = s0[162];
          float v199_data = ir1[10];
          ir1[10] = (v199_data + (v146_data * v197_data));
          float v201_data = r0[3];
          float v202_data = s0[3];
          float v204_data = ir1[0];
          ir1[0] = (v204_data + (v201_data * v202_data));
          float v207_data = s0[19];
          float v209_data = ir1[1];
          ir1[1] = (v209_data + (v201_data * v207_data));
          float v212_data = s0[35];
          float v214_data = ir1[2];
          ir1[2] = (v214_data + (v201_data * v212_data));
          float v217_data = s0[51];
          float v219_data = ir1[3];
          ir1[3] = (v219_data + (v201_data * v217_data));
          float v222_data = s0[67];
          float v224_data = ir1[4];
          ir1[4] = (v224_data + (v201_data * v222_data));
          float v227_data = s0[83];
          float v229_data = ir1[5];
          ir1[5] = (v229_data + (v201_data * v227_data));
          float v232_data = s0[99];
          float v234_data = ir1[6];
          ir1[6] = (v234_data + (v201_data * v232_data));
          float v237_data = s0[115];
          float v239_data = ir1[7];
          ir1[7] = (v239_data + (v201_data * v237_data));
          float v242_data = s0[131];
          float v244_data = ir1[8];
          ir1[8] = (v244_data + (v201_data * v242_data));
          float v247_data = s0[147];
          float v249_data = ir1[9];
          ir1[9] = (v249_data + (v201_data * v247_data));
          float v252_data = s0[163];
          float v254_data = ir1[10];
          ir1[10] = (v254_data + (v201_data * v252_data));
          float v256_data = r0[4];
          float v257_data = s0[4];
          float v259_data = ir1[0];
          ir1[0] = (v259_data + (v256_data * v257_data));
          float v262_data = s0[20];
          float v264_data = ir1[1];
          ir1[1] = (v264_data + (v256_data * v262_data));
          float v267_data = s0[36];
          float v269_data = ir1[2];
          ir1[2] = (v269_data + (v256_data * v267_data));
          float v272_data = s0[52];
          float v274_data = ir1[3];
          ir1[3] = (v274_data + (v256_data * v272_data));
          float v277_data = s0[68];
          float v279_data = ir1[4];
          ir1[4] = (v279_data + (v256_data * v277_data));
          float v282_data = s0[84];
          float v284_data = ir1[5];
          ir1[5] = (v284_data + (v256_data * v282_data));
          float v287_data = s0[100];
          float v289_data = ir1[6];
          ir1[6] = (v289_data + (v256_data * v287_data));
          float v292_data = s0[116];
          float v294_data = ir1[7];
          ir1[7] = (v294_data + (v256_data * v292_data));
          float v297_data = s0[132];
          float v299_data = ir1[8];
          ir1[8] = (v299_data + (v256_data * v297_data));
          float v302_data = s0[148];
          float v304_data = ir1[9];
          ir1[9] = (v304_data + (v256_data * v302_data));
          float v307_data = s0[164];
          float v309_data = ir1[10];
          ir1[10] = (v309_data + (v256_data * v307_data));
          float v311_data = r0[5];
          float v312_data = s0[5];
          float v314_data = ir1[0];
          ir1[0] = (v314_data + (v311_data * v312_data));
          float v317_data = s0[21];
          float v319_data = ir1[1];
          ir1[1] = (v319_data + (v311_data * v317_data));
          float v322_data = s0[37];
          float v324_data = ir1[2];
          ir1[2] = (v324_data + (v311_data * v322_data));
          float v327_data = s0[53];
          float v329_data = ir1[3];
          ir1[3] = (v329_data + (v311_data * v327_data));
          float v332_data = s0[69];
          float v334_data = ir1[4];
          ir1[4] = (v334_data + (v311_data * v332_data));
          float v337_data = s0[85];
          float v339_data = ir1[5];
          ir1[5] = (v339_data + (v311_data * v337_data));
          float v342_data = s0[101];
          float v344_data = ir1[6];
          ir1[6] = (v344_data + (v311_data * v342_data));
          float v347_data = s0[117];
          float v349_data = ir1[7];
          ir1[7] = (v349_data + (v311_data * v347_data));
          float v352_data = s0[133];
          float v354_data = ir1[8];
          ir1[8] = (v354_data + (v311_data * v352_data));
          float v357_data = s0[149];
          float v359_data = ir1[9];
          ir1[9] = (v359_data + (v311_data * v357_data));
          float v362_data = s0[165];
          float v364_data = ir1[10];
          ir1[10] = (v364_data + (v311_data * v362_data));
          float v366_data = r0[6];
          float v367_data = s0[6];
          float v369_data = ir1[0];
          ir1[0] = (v369_data + (v366_data * v367_data));
          float v372_data = s0[22];
          float v374_data = ir1[1];
          ir1[1] = (v374_data + (v366_data * v372_data));
          float v377_data = s0[38];
          float v379_data = ir1[2];
          ir1[2] = (v379_data + (v366_data * v377_data));
          float v382_data = s0[54];
          float v384_data = ir1[3];
          ir1[3] = (v384_data + (v366_data * v382_data));
          float v387_data = s0[70];
          float v389_data = ir1[4];
          ir1[4] = (v389_data + (v366_data * v387_data));
          float v392_data = s0[86];
          float v394_data = ir1[5];
          ir1[5] = (v394_data + (v366_data * v392_data));
          float v397_data = s0[102];
          float v399_data = ir1[6];
          ir1[6] = (v399_data + (v366_data * v397_data));
          float v402_data = s0[118];
          float v404_data = ir1[7];
          ir1[7] = (v404_data + (v366_data * v402_data));
          float v407_data = s0[134];
          float v409_data = ir1[8];
          ir1[8] = (v409_data + (v366_data * v407_data));
          float v412_data = s0[150];
          float v414_data = ir1[9];
          ir1[9] = (v414_data + (v366_data * v412_data));
          float v417_data = s0[166];
          float v419_data = ir1[10];
          ir1[10] = (v419_data + (v366_data * v417_data));
          float v421_data = r0[7];
          float v422_data = s0[7];
          float v424_data = ir1[0];
          ir1[0] = (v424_data + (v421_data * v422_data));
          float v427_data = s0[23];
          float v429_data = ir1[1];
          ir1[1] = (v429_data + (v421_data * v427_data));
          float v432_data = s0[39];
          float v434_data = ir1[2];
          ir1[2] = (v434_data + (v421_data * v432_data));
          float v437_data = s0[55];
          float v439_data = ir1[3];
          ir1[3] = (v439_data + (v421_data * v437_data));
          float v442_data = s0[71];
          float v444_data = ir1[4];
          ir1[4] = (v444_data + (v421_data * v442_data));
          float v447_data = s0[87];
          float v449_data = ir1[5];
          ir1[5] = (v449_data + (v421_data * v447_data));
          float v452_data = s0[103];
          float v454_data = ir1[6];
          ir1[6] = (v454_data + (v421_data * v452_data));
          float v457_data = s0[119];
          float v459_data = ir1[7];
          ir1[7] = (v459_data + (v421_data * v457_data));
          float v462_data = s0[135];
          float v464_data = ir1[8];
          ir1[8] = (v464_data + (v421_data * v462_data));
          float v467_data = s0[151];
          float v469_data = ir1[9];
          ir1[9] = (v469_data + (v421_data * v467_data));
          float v472_data = s0[167];
          float v474_data = ir1[10];
          ir1[10] = (v474_data + (v421_data * v472_data));
          float v476_data = r0[8];
          float v477_data = s0[8];
          float v479_data = ir1[0];
          ir1[0] = (v479_data + (v476_data * v477_data));
          float v482_data = s0[24];
          float v484_data = ir1[1];
          ir1[1] = (v484_data + (v476_data * v482_data));
          float v487_data = s0[40];
          float v489_data = ir1[2];
          ir1[2] = (v489_data + (v476_data * v487_data));
          float v492_data = s0[56];
          float v494_data = ir1[3];
          ir1[3] = (v494_data + (v476_data * v492_data));
          float v497_data = s0[72];
          float v499_data = ir1[4];
          ir1[4] = (v499_data + (v476_data * v497_data));
          float v502_data = s0[88];
          float v504_data = ir1[5];
          ir1[5] = (v504_data + (v476_data * v502_data));
          float v507_data = s0[104];
          float v509_data = ir1[6];
          ir1[6] = (v509_data + (v476_data * v507_data));
          float v512_data = s0[120];
          float v514_data = ir1[7];
          ir1[7] = (v514_data + (v476_data * v512_data));
          float v517_data = s0[136];
          float v519_data = ir1[8];
          ir1[8] = (v519_data + (v476_data * v517_data));
          float v522_data = s0[152];
          float v524_data = ir1[9];
          ir1[9] = (v524_data + (v476_data * v522_data));
          float v527_data = s0[168];
          float v529_data = ir1[10];
          ir1[10] = (v529_data + (v476_data * v527_data));
          float v531_data = r0[9];
          float v532_data = s0[9];
          float v534_data = ir1[0];
          ir1[0] = (v534_data + (v531_data * v532_data));
          float v537_data = s0[25];
          float v539_data = ir1[1];
          ir1[1] = (v539_data + (v531_data * v537_data));
          float v542_data = s0[41];
          float v544_data = ir1[2];
          ir1[2] = (v544_data + (v531_data * v542_data));
          float v547_data = s0[57];
          float v549_data = ir1[3];
          ir1[3] = (v549_data + (v531_data * v547_data));
          float v552_data = s0[73];
          float v554_data = ir1[4];
          ir1[4] = (v554_data + (v531_data * v552_data));
          float v557_data = s0[89];
          float v559_data = ir1[5];
          ir1[5] = (v559_data + (v531_data * v557_data));
          float v562_data = s0[105];
          float v564_data = ir1[6];
          ir1[6] = (v564_data + (v531_data * v562_data));
          float v567_data = s0[121];
          float v569_data = ir1[7];
          ir1[7] = (v569_data + (v531_data * v567_data));
          float v572_data = s0[137];
          float v574_data = ir1[8];
          ir1[8] = (v574_data + (v531_data * v572_data));
          float v577_data = s0[153];
          float v579_data = ir1[9];
          ir1[9] = (v579_data + (v531_data * v577_data));
          float v582_data = s0[169];
          float v584_data = ir1[10];
          ir1[10] = (v584_data + (v531_data * v582_data));
          float v586_data = r0[10];
          float v587_data = s0[10];
          float v589_data = ir1[0];
          ir1[0] = (v589_data + (v586_data * v587_data));
          float v592_data = s0[26];
          float v594_data = ir1[1];
          ir1[1] = (v594_data + (v586_data * v592_data));
          float v597_data = s0[42];
          float v599_data = ir1[2];
          ir1[2] = (v599_data + (v586_data * v597_data));
          float v602_data = s0[58];
          float v604_data = ir1[3];
          ir1[3] = (v604_data + (v586_data * v602_data));
          float v607_data = s0[74];
          float v609_data = ir1[4];
          ir1[4] = (v609_data + (v586_data * v607_data));
          float v612_data = s0[90];
          float v614_data = ir1[5];
          ir1[5] = (v614_data + (v586_data * v612_data));
          float v617_data = s0[106];
          float v619_data = ir1[6];
          ir1[6] = (v619_data + (v586_data * v617_data));
          float v622_data = s0[122];
          float v624_data = ir1[7];
          ir1[7] = (v624_data + (v586_data * v622_data));
          float v627_data = s0[138];
          float v629_data = ir1[8];
          ir1[8] = (v629_data + (v586_data * v627_data));
          float v632_data = s0[154];
          float v634_data = ir1[9];
          ir1[9] = (v634_data + (v586_data * v632_data));
          float v637_data = s0[170];
          float v639_data = ir1[10];
          ir1[10] = (v639_data + (v586_data * v637_data));
          float v641_data = r0[11];
          float v642_data = s0[11];
          float v644_data = ir1[0];
          ir1[0] = (v644_data + (v641_data * v642_data));
          float v647_data = s0[27];
          float v649_data = ir1[1];
          ir1[1] = (v649_data + (v641_data * v647_data));
          float v652_data = s0[43];
          float v654_data = ir1[2];
          ir1[2] = (v654_data + (v641_data * v652_data));
          float v657_data = s0[59];
          float v659_data = ir1[3];
          ir1[3] = (v659_data + (v641_data * v657_data));
          float v662_data = s0[75];
          float v664_data = ir1[4];
          ir1[4] = (v664_data + (v641_data * v662_data));
          float v667_data = s0[91];
          float v669_data = ir1[5];
          ir1[5] = (v669_data + (v641_data * v667_data));
          float v672_data = s0[107];
          float v674_data = ir1[6];
          ir1[6] = (v674_data + (v641_data * v672_data));
          float v677_data = s0[123];
          float v679_data = ir1[7];
          ir1[7] = (v679_data + (v641_data * v677_data));
          float v682_data = s0[139];
          float v684_data = ir1[8];
          ir1[8] = (v684_data + (v641_data * v682_data));
          float v687_data = s0[155];
          float v689_data = ir1[9];
          ir1[9] = (v689_data + (v641_data * v687_data));
          float v692_data = s0[171];
          float v694_data = ir1[10];
          ir1[10] = (v694_data + (v641_data * v692_data));
          float v696_data = r0[12];
          float v697_data = s0[12];
          float v699_data = ir1[0];
          ir1[0] = (v699_data + (v696_data * v697_data));
          float v702_data = s0[28];
          float v704_data = ir1[1];
          ir1[1] = (v704_data + (v696_data * v702_data));
          float v707_data = s0[44];
          float v709_data = ir1[2];
          ir1[2] = (v709_data + (v696_data * v707_data));
          float v712_data = s0[60];
          float v714_data = ir1[3];
          ir1[3] = (v714_data + (v696_data * v712_data));
          float v717_data = s0[76];
          float v719_data = ir1[4];
          ir1[4] = (v719_data + (v696_data * v717_data));
          float v722_data = s0[92];
          float v724_data = ir1[5];
          ir1[5] = (v724_data + (v696_data * v722_data));
          float v727_data = s0[108];
          float v729_data = ir1[6];
          ir1[6] = (v729_data + (v696_data * v727_data));
          float v732_data = s0[124];
          float v734_data = ir1[7];
          ir1[7] = (v734_data + (v696_data * v732_data));
          float v737_data = s0[140];
          float v739_data = ir1[8];
          ir1[8] = (v739_data + (v696_data * v737_data));
          float v742_data = s0[156];
          float v744_data = ir1[9];
          ir1[9] = (v744_data + (v696_data * v742_data));
          float v747_data = s0[172];
          float v749_data = ir1[10];
          ir1[10] = (v749_data + (v696_data * v747_data));
          float v751_data = r0[13];
          float v752_data = s0[13];
          float v754_data = ir1[0];
          ir1[0] = (v754_data + (v751_data * v752_data));
          float v757_data = s0[29];
          float v759_data = ir1[1];
          ir1[1] = (v759_data + (v751_data * v757_data));
          float v762_data = s0[45];
          float v764_data = ir1[2];
          ir1[2] = (v764_data + (v751_data * v762_data));
          float v767_data = s0[61];
          float v769_data = ir1[3];
          ir1[3] = (v769_data + (v751_data * v767_data));
          float v772_data = s0[77];
          float v774_data = ir1[4];
          ir1[4] = (v774_data + (v751_data * v772_data));
          float v777_data = s0[93];
          float v779_data = ir1[5];
          ir1[5] = (v779_data + (v751_data * v777_data));
          float v782_data = s0[109];
          float v784_data = ir1[6];
          ir1[6] = (v784_data + (v751_data * v782_data));
          float v787_data = s0[125];
          float v789_data = ir1[7];
          ir1[7] = (v789_data + (v751_data * v787_data));
          float v792_data = s0[141];
          float v794_data = ir1[8];
          ir1[8] = (v794_data + (v751_data * v792_data));
          float v797_data = s0[157];
          float v799_data = ir1[9];
          ir1[9] = (v799_data + (v751_data * v797_data));
          float v802_data = s0[173];
          float v804_data = ir1[10];
          ir1[10] = (v804_data + (v751_data * v802_data));
          float v806_data = r0[14];
          float v807_data = s0[14];
          float v809_data = ir1[0];
          ir1[0] = (v809_data + (v806_data * v807_data));
          float v812_data = s0[30];
          float v814_data = ir1[1];
          ir1[1] = (v814_data + (v806_data * v812_data));
          float v817_data = s0[46];
          float v819_data = ir1[2];
          ir1[2] = (v819_data + (v806_data * v817_data));
          float v822_data = s0[62];
          float v824_data = ir1[3];
          ir1[3] = (v824_data + (v806_data * v822_data));
          float v827_data = s0[78];
          float v829_data = ir1[4];
          ir1[4] = (v829_data + (v806_data * v827_data));
          float v832_data = s0[94];
          float v834_data = ir1[5];
          ir1[5] = (v834_data + (v806_data * v832_data));
          float v837_data = s0[110];
          float v839_data = ir1[6];
          ir1[6] = (v839_data + (v806_data * v837_data));
          float v842_data = s0[126];
          float v844_data = ir1[7];
          ir1[7] = (v844_data + (v806_data * v842_data));
          float v847_data = s0[142];
          float v849_data = ir1[8];
          ir1[8] = (v849_data + (v806_data * v847_data));
          float v852_data = s0[158];
          float v854_data = ir1[9];
          ir1[9] = (v854_data + (v806_data * v852_data));
          float v857_data = s0[174];
          float v859_data = ir1[10];
          ir1[10] = (v859_data + (v806_data * v857_data));
          float v861_data = r0[15];
          float v862_data = s0[15];
          float v864_data = ir1[0];
          ir1[0] = (v864_data + (v861_data * v862_data));
          float v867_data = s0[31];
          float v869_data = ir1[1];
          ir1[1] = (v869_data + (v861_data * v867_data));
          float v872_data = s0[47];
          float v874_data = ir1[2];
          ir1[2] = (v874_data + (v861_data * v872_data));
          float v877_data = s0[63];
          float v879_data = ir1[3];
          ir1[3] = (v879_data + (v861_data * v877_data));
          float v882_data = s0[79];
          float v884_data = ir1[4];
          ir1[4] = (v884_data + (v861_data * v882_data));
          float v887_data = s0[95];
          float v889_data = ir1[5];
          ir1[5] = (v889_data + (v861_data * v887_data));
          float v892_data = s0[111];
          float v894_data = ir1[6];
          ir1[6] = (v894_data + (v861_data * v892_data));
          float v897_data = s0[127];
          float v899_data = ir1[7];
          ir1[7] = (v899_data + (v861_data * v897_data));
          float v902_data = s0[143];
          float v904_data = ir1[8];
          ir1[8] = (v904_data + (v861_data * v902_data));
          float v907_data = s0[159];
          float v909_data = ir1[9];
          ir1[9] = (v909_data + (v861_data * v907_data));
          float v912_data = s0[175];
          float v914_data = ir1[10];
          ir1[10] = (v914_data + (v861_data * v912_data));
          // r1 = ir1
          #pragma unroll
          for (int32_t v916_n0 = 0; v916_n0 < 1; ++v916_n0) {
            #pragma unroll
            for (int32_t v917_n1 = 0; v917_n1 < 11; ++v917_n1) {
              int32_t v918_a = v916_n0 + v917_n1;
              float v919_data = ir1[v918_a];
              r1[v918_a] = v919_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v920_i0 = 0; v920_i0 < 1; ++v920_i0) {
            int32_t v925_lead = v22_lead + (v920_i0 * 16);
            #pragma unroll
            for (int32_t v921_i1 = 0; v921_i1 < 11; ++v921_i1) {
              float v923_data = r1[(v920_i0 + v921_i1)];
              glb_m0[(v925_lead + (v921_i1 * 16))] = v923_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

