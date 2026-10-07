// === base name ===
kernel_018a6d45968d7a2d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_018a6d45968d7a2d = {{16, 8, 1}, 16, 12, 1, 8, 4608, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_018a6d45968d7a2d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_018a6d45968d7a2d(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_018a6d45968d7a2d(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_018a6d45968d7a2d, block.x * block.y * block.z, 1152 * sizeof(float));
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
  config.sharedMemBytes = 1152 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_018a6d45968d7a2d(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_018a6d45968d7a2d(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_018a6d45968d7a2d, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_018a6d45968d7a2d<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_018a6d45968d7a2d(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 8 per block = block 16x8x1, 4608 B shared, occupancy grid
    // operands:
    //   m0 12×8(12×8) {0..12}×{0..8} strided
    //   m1 32×16(12×16) {4..16}×{0..16} strided
    //   m2 16×8(16×8) {0..16}×{0..8} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1152}],"shared_bytes":4608,"shared_elements":1152,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,8]],"name":"m0","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[4,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[32,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m1","offset":[4,0],"shape":[32,16]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[144 * threadIdx.y + 0];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 96 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 192 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 128 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v22_lead = threadIdx.x % 16;
          bool v23_g = v22_lead < 12;
          if (v23_g) {
            int32_t v28_a = (v22_lead + 4) - 4;
            #pragma unroll
            for (int32_t v24_i1 = 0; v24_i1 < 16; ++v24_i1) {
              float v31_data = __ldcg(&glb_m1[(v28_a + (v24_i1 * 12))]);
              r0[v24_i1] = v31_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 8; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          // ir1 = +(r0 * s0)
          // [(0, 12), (0, 8)] [(0, 16)]
          float ir1[8]{};
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
          float v76_data = r0[1];
          float v77_data = s0[1];
          float v79_data = ir1[0];
          ir1[0] = (v79_data + (v76_data * v77_data));
          float v82_data = s0[17];
          float v84_data = ir1[1];
          ir1[1] = (v84_data + (v76_data * v82_data));
          float v87_data = s0[33];
          float v89_data = ir1[2];
          ir1[2] = (v89_data + (v76_data * v87_data));
          float v92_data = s0[49];
          float v94_data = ir1[3];
          ir1[3] = (v94_data + (v76_data * v92_data));
          float v97_data = s0[65];
          float v99_data = ir1[4];
          ir1[4] = (v99_data + (v76_data * v97_data));
          float v102_data = s0[81];
          float v104_data = ir1[5];
          ir1[5] = (v104_data + (v76_data * v102_data));
          float v107_data = s0[97];
          float v109_data = ir1[6];
          ir1[6] = (v109_data + (v76_data * v107_data));
          float v112_data = s0[113];
          float v114_data = ir1[7];
          ir1[7] = (v114_data + (v76_data * v112_data));
          float v116_data = r0[2];
          float v117_data = s0[2];
          float v119_data = ir1[0];
          ir1[0] = (v119_data + (v116_data * v117_data));
          float v122_data = s0[18];
          float v124_data = ir1[1];
          ir1[1] = (v124_data + (v116_data * v122_data));
          float v127_data = s0[34];
          float v129_data = ir1[2];
          ir1[2] = (v129_data + (v116_data * v127_data));
          float v132_data = s0[50];
          float v134_data = ir1[3];
          ir1[3] = (v134_data + (v116_data * v132_data));
          float v137_data = s0[66];
          float v139_data = ir1[4];
          ir1[4] = (v139_data + (v116_data * v137_data));
          float v142_data = s0[82];
          float v144_data = ir1[5];
          ir1[5] = (v144_data + (v116_data * v142_data));
          float v147_data = s0[98];
          float v149_data = ir1[6];
          ir1[6] = (v149_data + (v116_data * v147_data));
          float v152_data = s0[114];
          float v154_data = ir1[7];
          ir1[7] = (v154_data + (v116_data * v152_data));
          float v156_data = r0[3];
          float v157_data = s0[3];
          float v159_data = ir1[0];
          ir1[0] = (v159_data + (v156_data * v157_data));
          float v162_data = s0[19];
          float v164_data = ir1[1];
          ir1[1] = (v164_data + (v156_data * v162_data));
          float v167_data = s0[35];
          float v169_data = ir1[2];
          ir1[2] = (v169_data + (v156_data * v167_data));
          float v172_data = s0[51];
          float v174_data = ir1[3];
          ir1[3] = (v174_data + (v156_data * v172_data));
          float v177_data = s0[67];
          float v179_data = ir1[4];
          ir1[4] = (v179_data + (v156_data * v177_data));
          float v182_data = s0[83];
          float v184_data = ir1[5];
          ir1[5] = (v184_data + (v156_data * v182_data));
          float v187_data = s0[99];
          float v189_data = ir1[6];
          ir1[6] = (v189_data + (v156_data * v187_data));
          float v192_data = s0[115];
          float v194_data = ir1[7];
          ir1[7] = (v194_data + (v156_data * v192_data));
          float v196_data = r0[4];
          float v197_data = s0[4];
          float v199_data = ir1[0];
          ir1[0] = (v199_data + (v196_data * v197_data));
          float v202_data = s0[20];
          float v204_data = ir1[1];
          ir1[1] = (v204_data + (v196_data * v202_data));
          float v207_data = s0[36];
          float v209_data = ir1[2];
          ir1[2] = (v209_data + (v196_data * v207_data));
          float v212_data = s0[52];
          float v214_data = ir1[3];
          ir1[3] = (v214_data + (v196_data * v212_data));
          float v217_data = s0[68];
          float v219_data = ir1[4];
          ir1[4] = (v219_data + (v196_data * v217_data));
          float v222_data = s0[84];
          float v224_data = ir1[5];
          ir1[5] = (v224_data + (v196_data * v222_data));
          float v227_data = s0[100];
          float v229_data = ir1[6];
          ir1[6] = (v229_data + (v196_data * v227_data));
          float v232_data = s0[116];
          float v234_data = ir1[7];
          ir1[7] = (v234_data + (v196_data * v232_data));
          float v236_data = r0[5];
          float v237_data = s0[5];
          float v239_data = ir1[0];
          ir1[0] = (v239_data + (v236_data * v237_data));
          float v242_data = s0[21];
          float v244_data = ir1[1];
          ir1[1] = (v244_data + (v236_data * v242_data));
          float v247_data = s0[37];
          float v249_data = ir1[2];
          ir1[2] = (v249_data + (v236_data * v247_data));
          float v252_data = s0[53];
          float v254_data = ir1[3];
          ir1[3] = (v254_data + (v236_data * v252_data));
          float v257_data = s0[69];
          float v259_data = ir1[4];
          ir1[4] = (v259_data + (v236_data * v257_data));
          float v262_data = s0[85];
          float v264_data = ir1[5];
          ir1[5] = (v264_data + (v236_data * v262_data));
          float v267_data = s0[101];
          float v269_data = ir1[6];
          ir1[6] = (v269_data + (v236_data * v267_data));
          float v272_data = s0[117];
          float v274_data = ir1[7];
          ir1[7] = (v274_data + (v236_data * v272_data));
          float v276_data = r0[6];
          float v277_data = s0[6];
          float v279_data = ir1[0];
          ir1[0] = (v279_data + (v276_data * v277_data));
          float v282_data = s0[22];
          float v284_data = ir1[1];
          ir1[1] = (v284_data + (v276_data * v282_data));
          float v287_data = s0[38];
          float v289_data = ir1[2];
          ir1[2] = (v289_data + (v276_data * v287_data));
          float v292_data = s0[54];
          float v294_data = ir1[3];
          ir1[3] = (v294_data + (v276_data * v292_data));
          float v297_data = s0[70];
          float v299_data = ir1[4];
          ir1[4] = (v299_data + (v276_data * v297_data));
          float v302_data = s0[86];
          float v304_data = ir1[5];
          ir1[5] = (v304_data + (v276_data * v302_data));
          float v307_data = s0[102];
          float v309_data = ir1[6];
          ir1[6] = (v309_data + (v276_data * v307_data));
          float v312_data = s0[118];
          float v314_data = ir1[7];
          ir1[7] = (v314_data + (v276_data * v312_data));
          float v316_data = r0[7];
          float v317_data = s0[7];
          float v319_data = ir1[0];
          ir1[0] = (v319_data + (v316_data * v317_data));
          float v322_data = s0[23];
          float v324_data = ir1[1];
          ir1[1] = (v324_data + (v316_data * v322_data));
          float v327_data = s0[39];
          float v329_data = ir1[2];
          ir1[2] = (v329_data + (v316_data * v327_data));
          float v332_data = s0[55];
          float v334_data = ir1[3];
          ir1[3] = (v334_data + (v316_data * v332_data));
          float v337_data = s0[71];
          float v339_data = ir1[4];
          ir1[4] = (v339_data + (v316_data * v337_data));
          float v342_data = s0[87];
          float v344_data = ir1[5];
          ir1[5] = (v344_data + (v316_data * v342_data));
          float v347_data = s0[103];
          float v349_data = ir1[6];
          ir1[6] = (v349_data + (v316_data * v347_data));
          float v352_data = s0[119];
          float v354_data = ir1[7];
          ir1[7] = (v354_data + (v316_data * v352_data));
          float v356_data = r0[8];
          float v357_data = s0[8];
          float v359_data = ir1[0];
          ir1[0] = (v359_data + (v356_data * v357_data));
          float v362_data = s0[24];
          float v364_data = ir1[1];
          ir1[1] = (v364_data + (v356_data * v362_data));
          float v367_data = s0[40];
          float v369_data = ir1[2];
          ir1[2] = (v369_data + (v356_data * v367_data));
          float v372_data = s0[56];
          float v374_data = ir1[3];
          ir1[3] = (v374_data + (v356_data * v372_data));
          float v377_data = s0[72];
          float v379_data = ir1[4];
          ir1[4] = (v379_data + (v356_data * v377_data));
          float v382_data = s0[88];
          float v384_data = ir1[5];
          ir1[5] = (v384_data + (v356_data * v382_data));
          float v387_data = s0[104];
          float v389_data = ir1[6];
          ir1[6] = (v389_data + (v356_data * v387_data));
          float v392_data = s0[120];
          float v394_data = ir1[7];
          ir1[7] = (v394_data + (v356_data * v392_data));
          float v396_data = r0[9];
          float v397_data = s0[9];
          float v399_data = ir1[0];
          ir1[0] = (v399_data + (v396_data * v397_data));
          float v402_data = s0[25];
          float v404_data = ir1[1];
          ir1[1] = (v404_data + (v396_data * v402_data));
          float v407_data = s0[41];
          float v409_data = ir1[2];
          ir1[2] = (v409_data + (v396_data * v407_data));
          float v412_data = s0[57];
          float v414_data = ir1[3];
          ir1[3] = (v414_data + (v396_data * v412_data));
          float v417_data = s0[73];
          float v419_data = ir1[4];
          ir1[4] = (v419_data + (v396_data * v417_data));
          float v422_data = s0[89];
          float v424_data = ir1[5];
          ir1[5] = (v424_data + (v396_data * v422_data));
          float v427_data = s0[105];
          float v429_data = ir1[6];
          ir1[6] = (v429_data + (v396_data * v427_data));
          float v432_data = s0[121];
          float v434_data = ir1[7];
          ir1[7] = (v434_data + (v396_data * v432_data));
          float v436_data = r0[10];
          float v437_data = s0[10];
          float v439_data = ir1[0];
          ir1[0] = (v439_data + (v436_data * v437_data));
          float v442_data = s0[26];
          float v444_data = ir1[1];
          ir1[1] = (v444_data + (v436_data * v442_data));
          float v447_data = s0[42];
          float v449_data = ir1[2];
          ir1[2] = (v449_data + (v436_data * v447_data));
          float v452_data = s0[58];
          float v454_data = ir1[3];
          ir1[3] = (v454_data + (v436_data * v452_data));
          float v457_data = s0[74];
          float v459_data = ir1[4];
          ir1[4] = (v459_data + (v436_data * v457_data));
          float v462_data = s0[90];
          float v464_data = ir1[5];
          ir1[5] = (v464_data + (v436_data * v462_data));
          float v467_data = s0[106];
          float v469_data = ir1[6];
          ir1[6] = (v469_data + (v436_data * v467_data));
          float v472_data = s0[122];
          float v474_data = ir1[7];
          ir1[7] = (v474_data + (v436_data * v472_data));
          float v476_data = r0[11];
          float v477_data = s0[11];
          float v479_data = ir1[0];
          ir1[0] = (v479_data + (v476_data * v477_data));
          float v482_data = s0[27];
          float v484_data = ir1[1];
          ir1[1] = (v484_data + (v476_data * v482_data));
          float v487_data = s0[43];
          float v489_data = ir1[2];
          ir1[2] = (v489_data + (v476_data * v487_data));
          float v492_data = s0[59];
          float v494_data = ir1[3];
          ir1[3] = (v494_data + (v476_data * v492_data));
          float v497_data = s0[75];
          float v499_data = ir1[4];
          ir1[4] = (v499_data + (v476_data * v497_data));
          float v502_data = s0[91];
          float v504_data = ir1[5];
          ir1[5] = (v504_data + (v476_data * v502_data));
          float v507_data = s0[107];
          float v509_data = ir1[6];
          ir1[6] = (v509_data + (v476_data * v507_data));
          float v512_data = s0[123];
          float v514_data = ir1[7];
          ir1[7] = (v514_data + (v476_data * v512_data));
          float v516_data = r0[12];
          float v517_data = s0[12];
          float v519_data = ir1[0];
          ir1[0] = (v519_data + (v516_data * v517_data));
          float v522_data = s0[28];
          float v524_data = ir1[1];
          ir1[1] = (v524_data + (v516_data * v522_data));
          float v527_data = s0[44];
          float v529_data = ir1[2];
          ir1[2] = (v529_data + (v516_data * v527_data));
          float v532_data = s0[60];
          float v534_data = ir1[3];
          ir1[3] = (v534_data + (v516_data * v532_data));
          float v537_data = s0[76];
          float v539_data = ir1[4];
          ir1[4] = (v539_data + (v516_data * v537_data));
          float v542_data = s0[92];
          float v544_data = ir1[5];
          ir1[5] = (v544_data + (v516_data * v542_data));
          float v547_data = s0[108];
          float v549_data = ir1[6];
          ir1[6] = (v549_data + (v516_data * v547_data));
          float v552_data = s0[124];
          float v554_data = ir1[7];
          ir1[7] = (v554_data + (v516_data * v552_data));
          float v556_data = r0[13];
          float v557_data = s0[13];
          float v559_data = ir1[0];
          ir1[0] = (v559_data + (v556_data * v557_data));
          float v562_data = s0[29];
          float v564_data = ir1[1];
          ir1[1] = (v564_data + (v556_data * v562_data));
          float v567_data = s0[45];
          float v569_data = ir1[2];
          ir1[2] = (v569_data + (v556_data * v567_data));
          float v572_data = s0[61];
          float v574_data = ir1[3];
          ir1[3] = (v574_data + (v556_data * v572_data));
          float v577_data = s0[77];
          float v579_data = ir1[4];
          ir1[4] = (v579_data + (v556_data * v577_data));
          float v582_data = s0[93];
          float v584_data = ir1[5];
          ir1[5] = (v584_data + (v556_data * v582_data));
          float v587_data = s0[109];
          float v589_data = ir1[6];
          ir1[6] = (v589_data + (v556_data * v587_data));
          float v592_data = s0[125];
          float v594_data = ir1[7];
          ir1[7] = (v594_data + (v556_data * v592_data));
          float v596_data = r0[14];
          float v597_data = s0[14];
          float v599_data = ir1[0];
          ir1[0] = (v599_data + (v596_data * v597_data));
          float v602_data = s0[30];
          float v604_data = ir1[1];
          ir1[1] = (v604_data + (v596_data * v602_data));
          float v607_data = s0[46];
          float v609_data = ir1[2];
          ir1[2] = (v609_data + (v596_data * v607_data));
          float v612_data = s0[62];
          float v614_data = ir1[3];
          ir1[3] = (v614_data + (v596_data * v612_data));
          float v617_data = s0[78];
          float v619_data = ir1[4];
          ir1[4] = (v619_data + (v596_data * v617_data));
          float v622_data = s0[94];
          float v624_data = ir1[5];
          ir1[5] = (v624_data + (v596_data * v622_data));
          float v627_data = s0[110];
          float v629_data = ir1[6];
          ir1[6] = (v629_data + (v596_data * v627_data));
          float v632_data = s0[126];
          float v634_data = ir1[7];
          ir1[7] = (v634_data + (v596_data * v632_data));
          float v636_data = r0[15];
          float v637_data = s0[15];
          float v639_data = ir1[0];
          ir1[0] = (v639_data + (v636_data * v637_data));
          float v642_data = s0[31];
          float v644_data = ir1[1];
          ir1[1] = (v644_data + (v636_data * v642_data));
          float v647_data = s0[47];
          float v649_data = ir1[2];
          ir1[2] = (v649_data + (v636_data * v647_data));
          float v652_data = s0[63];
          float v654_data = ir1[3];
          ir1[3] = (v654_data + (v636_data * v652_data));
          float v657_data = s0[79];
          float v659_data = ir1[4];
          ir1[4] = (v659_data + (v636_data * v657_data));
          float v662_data = s0[95];
          float v664_data = ir1[5];
          ir1[5] = (v664_data + (v636_data * v662_data));
          float v667_data = s0[111];
          float v669_data = ir1[6];
          ir1[6] = (v669_data + (v636_data * v667_data));
          float v672_data = s0[127];
          float v674_data = ir1[7];
          ir1[7] = (v674_data + (v636_data * v672_data));
          // r1 = ir1
          if (v23_g) {
            #pragma unroll
            for (int32_t v676_n1 = 0; v676_n1 < 8; ++v676_n1) {
              float v678_data = ir1[v676_n1];
              r1[v676_n1] = v678_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v23_g) {
            #pragma unroll
            for (int32_t v679_i1 = 0; v679_i1 < 8; ++v679_i1) {
              float v681_data = r1[v679_i1];
              glb_m0[(v22_lead + (v679_i1 * 12))] = v681_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

