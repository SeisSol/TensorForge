// === base name ===
kernel_dfd0eb6ee917dfa5

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_dfd0eb6ee917dfa5 = {{16, 8, 1}, 16, 12, 1, 8, 5632, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_dfd0eb6ee917dfa5(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_dfd0eb6ee917dfa5(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_dfd0eb6ee917dfa5(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_dfd0eb6ee917dfa5, block.x * block.y * block.z, 1408 * sizeof(float));
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
void launcher_kernel_dfd0eb6ee917dfa5(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_dfd0eb6ee917dfa5(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_dfd0eb6ee917dfa5, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_dfd0eb6ee917dfa5<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_dfd0eb6ee917dfa5(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 8 per block = block 16x8x1, 5632 B shared, occupancy grid
    // operands:
    //   m0 32×32(12×12) {0..12}×{0..12} strided
    //   m1 32×32(12×12) {0..12}×{0..12} strided
    //   m2 32×32(12×12) {0..12}×{0..12} strided
    //   m3 32×32(12×12) {0..12}×{0..12} strided
    // operations:
    //   t0[i,j]@{0..12}×{0..6} = m0[i,k] × m1[k,j]
    //   m2[i,j] = m3[i,k] × t0[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1408}],"shared_bytes":5632,"shared_elements":1408,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B","bbox":[[0,0],[12,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,6]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]},{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[176 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[160];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v12_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v12_batchId0 < numElements0; v12_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v13_ahead1 = v12_batchId0 + (gridDim.x * blockDim.y);
        size_t v15_batchId1 = (v13_ahead1 < numElements0) ? v13_ahead1 : v12_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v12_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v12_batchId0 * 144 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v12_batchId0 * 144 + 0 + m1_extraOffset];
          float *const __restrict__ glb_m2 = &m2[v12_batchId0 * 144 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[v12_batchId0 * 144 + 0 + m3_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v27_lead = threadIdx.x % 16;
          bool v28_g = v27_lead < 12;
          if (v28_g) {
            #pragma unroll
            for (int32_t v29_i1 = 0; v29_i1 < 12; ++v29_i1) {
              float v34_data = __ldcg(&glb_m0[(v27_lead + (v29_i1 * 12))]);
              r0[v29_i1] = v34_data;
            }
          }
          // s0 = load{g>s}(glb_m1[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 9; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m1[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m0););
          float r2[12]{};
          // r2 = load{g>r}(glb_m3);
          if (v28_g) {
            #pragma unroll
            for (int32_t v38_i1 = 0; v38_i1 < 12; ++v38_i1) {
              float v43_data = __ldcg(&glb_m3[(v27_lead + (v38_i1 * 12))]);
              r2[v38_i1] = v43_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[6]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // r1 = +(r0 * s0) + None
          // [(0, 12), (0, 6)] [(0, 12)]
          float v46_data = r0[0];
          float v47_data = s0[0];
          float v49_data = r1[0];
          r1[0] = (v49_data + (v46_data * v47_data));
          float v52_data = s0[12];
          float v54_data = r1[1];
          r1[1] = (v54_data + (v46_data * v52_data));
          float v57_data = s0[24];
          float v59_data = r1[2];
          r1[2] = (v59_data + (v46_data * v57_data));
          float v62_data = s0[36];
          float v64_data = r1[3];
          r1[3] = (v64_data + (v46_data * v62_data));
          float v67_data = s0[48];
          float v69_data = r1[4];
          r1[4] = (v69_data + (v46_data * v67_data));
          float v72_data = s0[60];
          float v74_data = r1[5];
          r1[5] = (v74_data + (v46_data * v72_data));
          float v76_data = r0[1];
          float v77_data = s0[1];
          float v79_data = r1[0];
          r1[0] = (v79_data + (v76_data * v77_data));
          float v82_data = s0[13];
          float v84_data = r1[1];
          r1[1] = (v84_data + (v76_data * v82_data));
          float v87_data = s0[25];
          float v89_data = r1[2];
          r1[2] = (v89_data + (v76_data * v87_data));
          float v92_data = s0[37];
          float v94_data = r1[3];
          r1[3] = (v94_data + (v76_data * v92_data));
          float v97_data = s0[49];
          float v99_data = r1[4];
          r1[4] = (v99_data + (v76_data * v97_data));
          float v102_data = s0[61];
          float v104_data = r1[5];
          r1[5] = (v104_data + (v76_data * v102_data));
          float v106_data = r0[2];
          float v107_data = s0[2];
          float v109_data = r1[0];
          r1[0] = (v109_data + (v106_data * v107_data));
          float v112_data = s0[14];
          float v114_data = r1[1];
          r1[1] = (v114_data + (v106_data * v112_data));
          float v117_data = s0[26];
          float v119_data = r1[2];
          r1[2] = (v119_data + (v106_data * v117_data));
          float v122_data = s0[38];
          float v124_data = r1[3];
          r1[3] = (v124_data + (v106_data * v122_data));
          float v127_data = s0[50];
          float v129_data = r1[4];
          r1[4] = (v129_data + (v106_data * v127_data));
          float v132_data = s0[62];
          float v134_data = r1[5];
          r1[5] = (v134_data + (v106_data * v132_data));
          float v136_data = r0[3];
          float v137_data = s0[3];
          float v139_data = r1[0];
          r1[0] = (v139_data + (v136_data * v137_data));
          float v142_data = s0[15];
          float v144_data = r1[1];
          r1[1] = (v144_data + (v136_data * v142_data));
          float v147_data = s0[27];
          float v149_data = r1[2];
          r1[2] = (v149_data + (v136_data * v147_data));
          float v152_data = s0[39];
          float v154_data = r1[3];
          r1[3] = (v154_data + (v136_data * v152_data));
          float v157_data = s0[51];
          float v159_data = r1[4];
          r1[4] = (v159_data + (v136_data * v157_data));
          float v162_data = s0[63];
          float v164_data = r1[5];
          r1[5] = (v164_data + (v136_data * v162_data));
          float v166_data = r0[4];
          float v167_data = s0[4];
          float v169_data = r1[0];
          r1[0] = (v169_data + (v166_data * v167_data));
          float v172_data = s0[16];
          float v174_data = r1[1];
          r1[1] = (v174_data + (v166_data * v172_data));
          float v177_data = s0[28];
          float v179_data = r1[2];
          r1[2] = (v179_data + (v166_data * v177_data));
          float v182_data = s0[40];
          float v184_data = r1[3];
          r1[3] = (v184_data + (v166_data * v182_data));
          float v187_data = s0[52];
          float v189_data = r1[4];
          r1[4] = (v189_data + (v166_data * v187_data));
          float v192_data = s0[64];
          float v194_data = r1[5];
          r1[5] = (v194_data + (v166_data * v192_data));
          float v196_data = r0[5];
          float v197_data = s0[5];
          float v199_data = r1[0];
          r1[0] = (v199_data + (v196_data * v197_data));
          float v202_data = s0[17];
          float v204_data = r1[1];
          r1[1] = (v204_data + (v196_data * v202_data));
          float v207_data = s0[29];
          float v209_data = r1[2];
          r1[2] = (v209_data + (v196_data * v207_data));
          float v212_data = s0[41];
          float v214_data = r1[3];
          r1[3] = (v214_data + (v196_data * v212_data));
          float v217_data = s0[53];
          float v219_data = r1[4];
          r1[4] = (v219_data + (v196_data * v217_data));
          float v222_data = s0[65];
          float v224_data = r1[5];
          r1[5] = (v224_data + (v196_data * v222_data));
          float v226_data = r0[6];
          float v227_data = s0[6];
          float v229_data = r1[0];
          r1[0] = (v229_data + (v226_data * v227_data));
          float v232_data = s0[18];
          float v234_data = r1[1];
          r1[1] = (v234_data + (v226_data * v232_data));
          float v237_data = s0[30];
          float v239_data = r1[2];
          r1[2] = (v239_data + (v226_data * v237_data));
          float v242_data = s0[42];
          float v244_data = r1[3];
          r1[3] = (v244_data + (v226_data * v242_data));
          float v247_data = s0[54];
          float v249_data = r1[4];
          r1[4] = (v249_data + (v226_data * v247_data));
          float v252_data = s0[66];
          float v254_data = r1[5];
          r1[5] = (v254_data + (v226_data * v252_data));
          float v256_data = r0[7];
          float v257_data = s0[7];
          float v259_data = r1[0];
          r1[0] = (v259_data + (v256_data * v257_data));
          float v262_data = s0[19];
          float v264_data = r1[1];
          r1[1] = (v264_data + (v256_data * v262_data));
          float v267_data = s0[31];
          float v269_data = r1[2];
          r1[2] = (v269_data + (v256_data * v267_data));
          float v272_data = s0[43];
          float v274_data = r1[3];
          r1[3] = (v274_data + (v256_data * v272_data));
          float v277_data = s0[55];
          float v279_data = r1[4];
          r1[4] = (v279_data + (v256_data * v277_data));
          float v282_data = s0[67];
          float v284_data = r1[5];
          r1[5] = (v284_data + (v256_data * v282_data));
          float v286_data = r0[8];
          float v287_data = s0[8];
          float v289_data = r1[0];
          r1[0] = (v289_data + (v286_data * v287_data));
          float v292_data = s0[20];
          float v294_data = r1[1];
          r1[1] = (v294_data + (v286_data * v292_data));
          float v297_data = s0[32];
          float v299_data = r1[2];
          r1[2] = (v299_data + (v286_data * v297_data));
          float v302_data = s0[44];
          float v304_data = r1[3];
          r1[3] = (v304_data + (v286_data * v302_data));
          float v307_data = s0[56];
          float v309_data = r1[4];
          r1[4] = (v309_data + (v286_data * v307_data));
          float v312_data = s0[68];
          float v314_data = r1[5];
          r1[5] = (v314_data + (v286_data * v312_data));
          float v316_data = r0[9];
          float v317_data = s0[9];
          float v319_data = r1[0];
          r1[0] = (v319_data + (v316_data * v317_data));
          float v322_data = s0[21];
          float v324_data = r1[1];
          r1[1] = (v324_data + (v316_data * v322_data));
          float v327_data = s0[33];
          float v329_data = r1[2];
          r1[2] = (v329_data + (v316_data * v327_data));
          float v332_data = s0[45];
          float v334_data = r1[3];
          r1[3] = (v334_data + (v316_data * v332_data));
          float v337_data = s0[57];
          float v339_data = r1[4];
          r1[4] = (v339_data + (v316_data * v337_data));
          float v342_data = s0[69];
          float v344_data = r1[5];
          r1[5] = (v344_data + (v316_data * v342_data));
          float v346_data = r0[10];
          float v347_data = s0[10];
          float v349_data = r1[0];
          r1[0] = (v349_data + (v346_data * v347_data));
          float v352_data = s0[22];
          float v354_data = r1[1];
          r1[1] = (v354_data + (v346_data * v352_data));
          float v357_data = s0[34];
          float v359_data = r1[2];
          r1[2] = (v359_data + (v346_data * v357_data));
          float v362_data = s0[46];
          float v364_data = r1[3];
          r1[3] = (v364_data + (v346_data * v362_data));
          float v367_data = s0[58];
          float v369_data = r1[4];
          r1[4] = (v369_data + (v346_data * v367_data));
          float v372_data = s0[70];
          float v374_data = r1[5];
          r1[5] = (v374_data + (v346_data * v372_data));
          float v376_data = r0[11];
          float v377_data = s0[11];
          float v379_data = r1[0];
          r1[0] = (v379_data + (v376_data * v377_data));
          float v382_data = s0[23];
          float v384_data = r1[1];
          r1[1] = (v384_data + (v376_data * v382_data));
          float v387_data = s0[35];
          float v389_data = r1[2];
          r1[2] = (v389_data + (v376_data * v387_data));
          float v392_data = s0[47];
          float v394_data = r1[3];
          r1[3] = (v394_data + (v376_data * v392_data));
          float v397_data = s0[59];
          float v399_data = r1[4];
          r1[4] = (v399_data + (v376_data * v397_data));
          float v402_data = s0[71];
          float v404_data = r1[5];
          r1[5] = (v404_data + (v376_data * v402_data));
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // s1 = store{r>s, clear}(localShrMem0, r1);
          if (v28_g) {
            #pragma unroll
            for (int32_t v406_z1 = 6; v406_z1 < 12; ++v406_z1) {
              int32_t v411_a = v27_lead + (v406_z1 * 12);
              s1[(v411_a ^ ((v411_a >> 4) & 15))] = 0.0f;
            }
          }
          if (v28_g) {
            #pragma unroll
            for (int32_t v415_i1 = 0; v415_i1 < 6; ++v415_i1) {
              float v417_data = r1[v415_i1];
              int32_t v421_a = v27_lead + (v415_i1 * 12);
              s1[(v421_a ^ ((v421_a >> 4) & 15))] = v417_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m3););
          float r3[12]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // ir3 = +(r2 * s1)
          // [(0, 12), (0, 12)] [(0, 12)]
          float ir3[12]{};
          float v427_data = r2[0];
          float v428_data = s1[0];
          float v430_data = ir3[0];
          ir3[0] = (v430_data + (v427_data * v428_data));
          float v433_data = s1[12];
          float v435_data = ir3[1];
          ir3[1] = (v435_data + (v427_data * v433_data));
          float v438_data = s1[25];
          float v440_data = ir3[2];
          ir3[2] = (v440_data + (v427_data * v438_data));
          float v443_data = s1[38];
          float v445_data = ir3[3];
          ir3[3] = (v445_data + (v427_data * v443_data));
          float v448_data = s1[51];
          float v450_data = ir3[4];
          ir3[4] = (v450_data + (v427_data * v448_data));
          float v453_data = s1[63];
          float v455_data = ir3[5];
          ir3[5] = (v455_data + (v427_data * v453_data));
          float v458_data = s1[76];
          float v460_data = ir3[6];
          ir3[6] = (v460_data + (v427_data * v458_data));
          float v463_data = s1[81];
          float v465_data = ir3[7];
          ir3[7] = (v465_data + (v427_data * v463_data));
          float v468_data = s1[102];
          float v470_data = ir3[8];
          ir3[8] = (v470_data + (v427_data * v468_data));
          float v473_data = s1[106];
          float v475_data = ir3[9];
          ir3[9] = (v475_data + (v427_data * v473_data));
          float v478_data = s1[127];
          float v480_data = ir3[10];
          ir3[10] = (v480_data + (v427_data * v478_data));
          float v483_data = s1[140];
          float v485_data = ir3[11];
          ir3[11] = (v485_data + (v427_data * v483_data));
          float v487_data = r2[1];
          float v488_data = s1[1];
          float v490_data = ir3[0];
          ir3[0] = (v490_data + (v487_data * v488_data));
          float v493_data = s1[13];
          float v495_data = ir3[1];
          ir3[1] = (v495_data + (v487_data * v493_data));
          float v498_data = s1[24];
          float v500_data = ir3[2];
          ir3[2] = (v500_data + (v487_data * v498_data));
          float v503_data = s1[39];
          float v505_data = ir3[3];
          ir3[3] = (v505_data + (v487_data * v503_data));
          float v508_data = s1[50];
          float v510_data = ir3[4];
          ir3[4] = (v510_data + (v487_data * v508_data));
          float v513_data = s1[62];
          float v515_data = ir3[5];
          ir3[5] = (v515_data + (v487_data * v513_data));
          float v518_data = s1[77];
          float v520_data = ir3[6];
          ir3[6] = (v520_data + (v487_data * v518_data));
          float v523_data = s1[80];
          float v525_data = ir3[7];
          ir3[7] = (v525_data + (v487_data * v523_data));
          float v528_data = s1[103];
          float v530_data = ir3[8];
          ir3[8] = (v530_data + (v487_data * v528_data));
          float v533_data = s1[107];
          float v535_data = ir3[9];
          ir3[9] = (v535_data + (v487_data * v533_data));
          float v538_data = s1[126];
          float v540_data = ir3[10];
          ir3[10] = (v540_data + (v487_data * v538_data));
          float v543_data = s1[141];
          float v545_data = ir3[11];
          ir3[11] = (v545_data + (v487_data * v543_data));
          float v547_data = r2[2];
          float v548_data = s1[2];
          float v550_data = ir3[0];
          ir3[0] = (v550_data + (v547_data * v548_data));
          float v553_data = s1[14];
          float v555_data = ir3[1];
          ir3[1] = (v555_data + (v547_data * v553_data));
          float v558_data = s1[27];
          float v560_data = ir3[2];
          ir3[2] = (v560_data + (v547_data * v558_data));
          float v563_data = s1[36];
          float v565_data = ir3[3];
          ir3[3] = (v565_data + (v547_data * v563_data));
          float v568_data = s1[49];
          float v570_data = ir3[4];
          ir3[4] = (v570_data + (v547_data * v568_data));
          float v573_data = s1[61];
          float v575_data = ir3[5];
          ir3[5] = (v575_data + (v547_data * v573_data));
          float v578_data = s1[78];
          float v580_data = ir3[6];
          ir3[6] = (v580_data + (v547_data * v578_data));
          float v583_data = s1[83];
          float v585_data = ir3[7];
          ir3[7] = (v585_data + (v547_data * v583_data));
          float v588_data = s1[100];
          float v590_data = ir3[8];
          ir3[8] = (v590_data + (v547_data * v588_data));
          float v593_data = s1[104];
          float v595_data = ir3[9];
          ir3[9] = (v595_data + (v547_data * v593_data));
          float v598_data = s1[125];
          float v600_data = ir3[10];
          ir3[10] = (v600_data + (v547_data * v598_data));
          float v603_data = s1[142];
          float v605_data = ir3[11];
          ir3[11] = (v605_data + (v547_data * v603_data));
          float v607_data = r2[3];
          float v608_data = s1[3];
          float v610_data = ir3[0];
          ir3[0] = (v610_data + (v607_data * v608_data));
          float v613_data = s1[15];
          float v615_data = ir3[1];
          ir3[1] = (v615_data + (v607_data * v613_data));
          float v618_data = s1[26];
          float v620_data = ir3[2];
          ir3[2] = (v620_data + (v607_data * v618_data));
          float v623_data = s1[37];
          float v625_data = ir3[3];
          ir3[3] = (v625_data + (v607_data * v623_data));
          float v628_data = s1[48];
          float v630_data = ir3[4];
          ir3[4] = (v630_data + (v607_data * v628_data));
          float v633_data = s1[60];
          float v635_data = ir3[5];
          ir3[5] = (v635_data + (v607_data * v633_data));
          float v638_data = s1[79];
          float v640_data = ir3[6];
          ir3[6] = (v640_data + (v607_data * v638_data));
          float v643_data = s1[82];
          float v645_data = ir3[7];
          ir3[7] = (v645_data + (v607_data * v643_data));
          float v648_data = s1[101];
          float v650_data = ir3[8];
          ir3[8] = (v650_data + (v607_data * v648_data));
          float v653_data = s1[105];
          float v655_data = ir3[9];
          ir3[9] = (v655_data + (v607_data * v653_data));
          float v658_data = s1[124];
          float v660_data = ir3[10];
          ir3[10] = (v660_data + (v607_data * v658_data));
          float v663_data = s1[143];
          float v665_data = ir3[11];
          ir3[11] = (v665_data + (v607_data * v663_data));
          float v667_data = r2[4];
          float v668_data = s1[4];
          float v670_data = ir3[0];
          ir3[0] = (v670_data + (v667_data * v668_data));
          float v673_data = s1[17];
          float v675_data = ir3[1];
          ir3[1] = (v675_data + (v667_data * v673_data));
          float v678_data = s1[29];
          float v680_data = ir3[2];
          ir3[2] = (v680_data + (v667_data * v678_data));
          float v683_data = s1[42];
          float v685_data = ir3[3];
          ir3[3] = (v685_data + (v667_data * v683_data));
          float v688_data = s1[55];
          float v690_data = ir3[4];
          ir3[4] = (v690_data + (v667_data * v688_data));
          float v693_data = s1[68];
          float v695_data = ir3[5];
          ir3[5] = (v695_data + (v667_data * v693_data));
          float v698_data = s1[72];
          float v700_data = ir3[6];
          ir3[6] = (v700_data + (v667_data * v698_data));
          float v703_data = s1[93];
          float v705_data = ir3[7];
          ir3[7] = (v705_data + (v667_data * v703_data));
          float v708_data = s1[98];
          float v710_data = ir3[8];
          ir3[8] = (v710_data + (v667_data * v708_data));
          float v713_data = s1[119];
          float v715_data = ir3[9];
          ir3[9] = (v715_data + (v667_data * v713_data));
          float v718_data = s1[123];
          float v720_data = ir3[10];
          ir3[10] = (v720_data + (v667_data * v718_data));
          float v723_data = s1[128];
          float v725_data = ir3[11];
          ir3[11] = (v725_data + (v667_data * v723_data));
          float v727_data = r2[5];
          float v728_data = s1[5];
          float v730_data = ir3[0];
          ir3[0] = (v730_data + (v727_data * v728_data));
          float v733_data = s1[16];
          float v735_data = ir3[1];
          ir3[1] = (v735_data + (v727_data * v733_data));
          float v738_data = s1[28];
          float v740_data = ir3[2];
          ir3[2] = (v740_data + (v727_data * v738_data));
          float v743_data = s1[43];
          float v745_data = ir3[3];
          ir3[3] = (v745_data + (v727_data * v743_data));
          float v748_data = s1[54];
          float v750_data = ir3[4];
          ir3[4] = (v750_data + (v727_data * v748_data));
          float v753_data = s1[69];
          float v755_data = ir3[5];
          ir3[5] = (v755_data + (v727_data * v753_data));
          float v758_data = s1[73];
          float v760_data = ir3[6];
          ir3[6] = (v760_data + (v727_data * v758_data));
          float v763_data = s1[92];
          float v765_data = ir3[7];
          ir3[7] = (v765_data + (v727_data * v763_data));
          float v768_data = s1[99];
          float v770_data = ir3[8];
          ir3[8] = (v770_data + (v727_data * v768_data));
          float v773_data = s1[118];
          float v775_data = ir3[9];
          ir3[9] = (v775_data + (v727_data * v773_data));
          float v778_data = s1[122];
          float v780_data = ir3[10];
          ir3[10] = (v780_data + (v727_data * v778_data));
          float v783_data = s1[129];
          float v785_data = ir3[11];
          ir3[11] = (v785_data + (v727_data * v783_data));
          float v787_data = r2[6];
          float v788_data = s1[6];
          float v790_data = ir3[0];
          ir3[0] = (v790_data + (v787_data * v788_data));
          float v793_data = s1[19];
          float v795_data = ir3[1];
          ir3[1] = (v795_data + (v787_data * v793_data));
          float v798_data = s1[31];
          float v800_data = ir3[2];
          ir3[2] = (v800_data + (v787_data * v798_data));
          float v803_data = s1[40];
          float v805_data = ir3[3];
          ir3[3] = (v805_data + (v787_data * v803_data));
          float v808_data = s1[53];
          float v810_data = ir3[4];
          ir3[4] = (v810_data + (v787_data * v808_data));
          float v813_data = s1[70];
          float v815_data = ir3[5];
          ir3[5] = (v815_data + (v787_data * v813_data));
          float v818_data = s1[74];
          float v820_data = ir3[6];
          ir3[6] = (v820_data + (v787_data * v818_data));
          float v823_data = s1[95];
          float v825_data = ir3[7];
          ir3[7] = (v825_data + (v787_data * v823_data));
          float v828_data = s1[96];
          float v830_data = ir3[8];
          ir3[8] = (v830_data + (v787_data * v828_data));
          float v833_data = s1[117];
          float v835_data = ir3[9];
          ir3[9] = (v835_data + (v787_data * v833_data));
          float v838_data = s1[121];
          float v840_data = ir3[10];
          ir3[10] = (v840_data + (v787_data * v838_data));
          float v843_data = s1[130];
          float v845_data = ir3[11];
          ir3[11] = (v845_data + (v787_data * v843_data));
          float v847_data = r2[7];
          float v848_data = s1[7];
          float v850_data = ir3[0];
          ir3[0] = (v850_data + (v847_data * v848_data));
          float v853_data = s1[18];
          float v855_data = ir3[1];
          ir3[1] = (v855_data + (v847_data * v853_data));
          float v858_data = s1[30];
          float v860_data = ir3[2];
          ir3[2] = (v860_data + (v847_data * v858_data));
          float v863_data = s1[41];
          float v865_data = ir3[3];
          ir3[3] = (v865_data + (v847_data * v863_data));
          float v868_data = s1[52];
          float v870_data = ir3[4];
          ir3[4] = (v870_data + (v847_data * v868_data));
          float v873_data = s1[71];
          float v875_data = ir3[5];
          ir3[5] = (v875_data + (v847_data * v873_data));
          float v878_data = s1[75];
          float v880_data = ir3[6];
          ir3[6] = (v880_data + (v847_data * v878_data));
          float v883_data = s1[94];
          float v885_data = ir3[7];
          ir3[7] = (v885_data + (v847_data * v883_data));
          float v888_data = s1[97];
          float v890_data = ir3[8];
          ir3[8] = (v890_data + (v847_data * v888_data));
          float v893_data = s1[116];
          float v895_data = ir3[9];
          ir3[9] = (v895_data + (v847_data * v893_data));
          float v898_data = s1[120];
          float v900_data = ir3[10];
          ir3[10] = (v900_data + (v847_data * v898_data));
          float v903_data = s1[131];
          float v905_data = ir3[11];
          ir3[11] = (v905_data + (v847_data * v903_data));
          float v907_data = r2[8];
          float v908_data = s1[8];
          float v910_data = ir3[0];
          ir3[0] = (v910_data + (v907_data * v908_data));
          float v913_data = s1[21];
          float v915_data = ir3[1];
          ir3[1] = (v915_data + (v907_data * v913_data));
          float v918_data = s1[34];
          float v920_data = ir3[2];
          ir3[2] = (v920_data + (v907_data * v918_data));
          float v923_data = s1[46];
          float v925_data = ir3[3];
          ir3[3] = (v925_data + (v907_data * v923_data));
          float v928_data = s1[59];
          float v930_data = ir3[4];
          ir3[4] = (v930_data + (v907_data * v928_data));
          float v933_data = s1[64];
          float v935_data = ir3[5];
          ir3[5] = (v935_data + (v907_data * v933_data));
          float v938_data = s1[85];
          float v940_data = ir3[6];
          ir3[6] = (v940_data + (v907_data * v938_data));
          float v943_data = s1[89];
          float v945_data = ir3[7];
          ir3[7] = (v945_data + (v907_data * v943_data));
          float v948_data = s1[110];
          float v950_data = ir3[8];
          ir3[8] = (v950_data + (v907_data * v948_data));
          float v953_data = s1[115];
          float v955_data = ir3[9];
          ir3[9] = (v955_data + (v907_data * v953_data));
          float v958_data = s1[136];
          float v960_data = ir3[10];
          ir3[10] = (v960_data + (v907_data * v958_data));
          float v963_data = s1[132];
          float v965_data = ir3[11];
          ir3[11] = (v965_data + (v907_data * v963_data));
          float v967_data = r2[9];
          float v968_data = s1[9];
          float v970_data = ir3[0];
          ir3[0] = (v970_data + (v967_data * v968_data));
          float v973_data = s1[20];
          float v975_data = ir3[1];
          ir3[1] = (v975_data + (v967_data * v973_data));
          float v978_data = s1[35];
          float v980_data = ir3[2];
          ir3[2] = (v980_data + (v967_data * v978_data));
          float v983_data = s1[47];
          float v985_data = ir3[3];
          ir3[3] = (v985_data + (v967_data * v983_data));
          float v988_data = s1[58];
          float v990_data = ir3[4];
          ir3[4] = (v990_data + (v967_data * v988_data));
          float v993_data = s1[65];
          float v995_data = ir3[5];
          ir3[5] = (v995_data + (v967_data * v993_data));
          float v998_data = s1[84];
          float v1000_data = ir3[6];
          ir3[6] = (v1000_data + (v967_data * v998_data));
          float v1003_data = s1[88];
          float v1005_data = ir3[7];
          ir3[7] = (v1005_data + (v967_data * v1003_data));
          float v1008_data = s1[111];
          float v1010_data = ir3[8];
          ir3[8] = (v1010_data + (v967_data * v1008_data));
          float v1013_data = s1[114];
          float v1015_data = ir3[9];
          ir3[9] = (v1015_data + (v967_data * v1013_data));
          float v1018_data = s1[137];
          float v1020_data = ir3[10];
          ir3[10] = (v1020_data + (v967_data * v1018_data));
          float v1023_data = s1[133];
          float v1025_data = ir3[11];
          ir3[11] = (v1025_data + (v967_data * v1023_data));
          float v1027_data = r2[10];
          float v1028_data = s1[10];
          float v1030_data = ir3[0];
          ir3[0] = (v1030_data + (v1027_data * v1028_data));
          float v1033_data = s1[23];
          float v1035_data = ir3[1];
          ir3[1] = (v1035_data + (v1027_data * v1033_data));
          float v1038_data = s1[32];
          float v1040_data = ir3[2];
          ir3[2] = (v1040_data + (v1027_data * v1038_data));
          float v1043_data = s1[44];
          float v1045_data = ir3[3];
          ir3[3] = (v1045_data + (v1027_data * v1043_data));
          float v1048_data = s1[57];
          float v1050_data = ir3[4];
          ir3[4] = (v1050_data + (v1027_data * v1048_data));
          float v1053_data = s1[66];
          float v1055_data = ir3[5];
          ir3[5] = (v1055_data + (v1027_data * v1053_data));
          float v1058_data = s1[87];
          float v1060_data = ir3[6];
          ir3[6] = (v1060_data + (v1027_data * v1058_data));
          float v1063_data = s1[91];
          float v1065_data = ir3[7];
          ir3[7] = (v1065_data + (v1027_data * v1063_data));
          float v1068_data = s1[108];
          float v1070_data = ir3[8];
          ir3[8] = (v1070_data + (v1027_data * v1068_data));
          float v1073_data = s1[113];
          float v1075_data = ir3[9];
          ir3[9] = (v1075_data + (v1027_data * v1073_data));
          float v1078_data = s1[138];
          float v1080_data = ir3[10];
          ir3[10] = (v1080_data + (v1027_data * v1078_data));
          float v1083_data = s1[134];
          float v1085_data = ir3[11];
          ir3[11] = (v1085_data + (v1027_data * v1083_data));
          float v1087_data = r2[11];
          float v1088_data = s1[11];
          float v1090_data = ir3[0];
          ir3[0] = (v1090_data + (v1087_data * v1088_data));
          float v1093_data = s1[22];
          float v1095_data = ir3[1];
          ir3[1] = (v1095_data + (v1087_data * v1093_data));
          float v1098_data = s1[33];
          float v1100_data = ir3[2];
          ir3[2] = (v1100_data + (v1087_data * v1098_data));
          float v1103_data = s1[45];
          float v1105_data = ir3[3];
          ir3[3] = (v1105_data + (v1087_data * v1103_data));
          float v1108_data = s1[56];
          float v1110_data = ir3[4];
          ir3[4] = (v1110_data + (v1087_data * v1108_data));
          float v1113_data = s1[67];
          float v1115_data = ir3[5];
          ir3[5] = (v1115_data + (v1087_data * v1113_data));
          float v1118_data = s1[86];
          float v1120_data = ir3[6];
          ir3[6] = (v1120_data + (v1087_data * v1118_data));
          float v1123_data = s1[90];
          float v1125_data = ir3[7];
          ir3[7] = (v1125_data + (v1087_data * v1123_data));
          float v1128_data = s1[109];
          float v1130_data = ir3[8];
          ir3[8] = (v1130_data + (v1087_data * v1128_data));
          float v1133_data = s1[112];
          float v1135_data = ir3[9];
          ir3[9] = (v1135_data + (v1087_data * v1133_data));
          float v1138_data = s1[139];
          float v1140_data = ir3[10];
          ir3[10] = (v1140_data + (v1087_data * v1138_data));
          float v1143_data = s1[135];
          float v1145_data = ir3[11];
          ir3[11] = (v1145_data + (v1087_data * v1143_data));
          // r3 = ir3
          if (v28_g) {
            #pragma unroll
            for (int32_t v1147_n1 = 0; v1147_n1 < 12; ++v1147_n1) {
              float v1149_data = ir3[v1147_n1];
              r3[v1147_n1] = v1149_data;
            }
          }
          // glb_m2 = store{r>g}(r3);
          if (v28_g) {
            #pragma unroll
            for (int32_t v1150_i1 = 0; v1150_i1 < 12; ++v1150_i1) {
              float v1152_data = r3[v1150_i1];
              glb_m2[(v27_lead + (v1150_i1 * 12))] = v1152_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

