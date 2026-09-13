// === base name ===
kernel_1018efc17d7a566d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_1018efc17d7a566d = {{8, 16, 1}, 8, 8, 1, 16, 4608, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_1018efc17d7a566d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_1018efc17d7a566d(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_1018efc17d7a566d(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (8, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_1018efc17d7a566d, block.x * block.y * block.z, 1152 * sizeof(float));
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
  config.block[0] = 8;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 1152 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_1018efc17d7a566d(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_1018efc17d7a566d(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_1018efc17d7a566d, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_1018efc17d7a566d<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_1018efc17d7a566d(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 8 lanes x 16 per block = block 8x16x1, 4608 B shared, occupancy grid
    // operands:
    //   m0 8×8(8×8) {0..8}×{0..8} strided
    //   m1 8×8(8×8) {0..8}×{0..8} strided
    //   m2 8×8(8×8) {0..8}×{0..8} strided
    //   m3 8×8(8×8) {0..8}×{0..8} strided
    //   m4 8×8(8×8) {0..8}×{0..8} strided
    // operations:
    //   t0[i,j] = m0[i,k] × m1[k,j]
    //   t0[i,j] += m2[i,k] × m3[k,j]
    //   C = abs(TMP)
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1152}],"shared_bytes":4608,"shared_elements":1152,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A1","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"A2","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m4","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[72 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s2 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 64 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 64 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 64 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[v7_batchId0 * 64 + 0 + m3_extraOffset];
          float *const __restrict__ glb_m4 = &m4[v7_batchId0 * 64 + 0 + m4_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v23_lead = threadIdx.x % 8;
          #pragma unroll
          for (int32_t v24_i0 = 0; v24_i0 < 1; ++v24_i0) {
            int32_t v27_lead = v23_lead + (v24_i0 * 8);
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 8; ++v25_i1) {
              float v30_data = __ldcg(&glb_m0[(v27_lead + (v25_i1 * 8))]);
              r0[(v24_i0 + v25_i1)] = v30_data;
            }
          }
          // s0 = load{g>s}(glb_m1[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 8; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 8], &glb_m1[0 + 0 + 1 * threadIdx.x + i * 8], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m0););
          float r2[8]{};
          // r2 = load{g>r}(glb_m2);
          #pragma unroll
          for (int32_t v34_i0 = 0; v34_i0 < 1; ++v34_i0) {
            int32_t v37_lead = v23_lead + (v34_i0 * 8);
            #pragma unroll
            for (int32_t v35_i1 = 0; v35_i1 < 8; ++v35_i1) {
              float v40_data = __ldcg(&glb_m2[(v37_lead + (v35_i1 * 8))]);
              r2[(v34_i0 + v35_i1)] = v40_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          // r1 = +(r0 * s0) + None
          // [(0, 8), (0, 8)] [(0, 8)]
          float v43_data = r0[0];
          float v44_data = s0[0];
          float v46_data = r1[0];
          r1[0] = (v46_data + (v43_data * v44_data));
          float v49_data = s0[8];
          float v51_data = r1[1];
          r1[1] = (v51_data + (v43_data * v49_data));
          float v54_data = s0[16];
          float v56_data = r1[2];
          r1[2] = (v56_data + (v43_data * v54_data));
          float v59_data = s0[24];
          float v61_data = r1[3];
          r1[3] = (v61_data + (v43_data * v59_data));
          float v64_data = s0[32];
          float v66_data = r1[4];
          r1[4] = (v66_data + (v43_data * v64_data));
          float v69_data = s0[40];
          float v71_data = r1[5];
          r1[5] = (v71_data + (v43_data * v69_data));
          float v74_data = s0[48];
          float v76_data = r1[6];
          r1[6] = (v76_data + (v43_data * v74_data));
          float v79_data = s0[56];
          float v81_data = r1[7];
          r1[7] = (v81_data + (v43_data * v79_data));
          float v83_data = r0[1];
          float v84_data = s0[1];
          float v86_data = r1[0];
          r1[0] = (v86_data + (v83_data * v84_data));
          float v89_data = s0[9];
          float v91_data = r1[1];
          r1[1] = (v91_data + (v83_data * v89_data));
          float v94_data = s0[17];
          float v96_data = r1[2];
          r1[2] = (v96_data + (v83_data * v94_data));
          float v99_data = s0[25];
          float v101_data = r1[3];
          r1[3] = (v101_data + (v83_data * v99_data));
          float v104_data = s0[33];
          float v106_data = r1[4];
          r1[4] = (v106_data + (v83_data * v104_data));
          float v109_data = s0[41];
          float v111_data = r1[5];
          r1[5] = (v111_data + (v83_data * v109_data));
          float v114_data = s0[49];
          float v116_data = r1[6];
          r1[6] = (v116_data + (v83_data * v114_data));
          float v119_data = s0[57];
          float v121_data = r1[7];
          r1[7] = (v121_data + (v83_data * v119_data));
          float v123_data = r0[2];
          float v124_data = s0[2];
          float v126_data = r1[0];
          r1[0] = (v126_data + (v123_data * v124_data));
          float v129_data = s0[10];
          float v131_data = r1[1];
          r1[1] = (v131_data + (v123_data * v129_data));
          float v134_data = s0[18];
          float v136_data = r1[2];
          r1[2] = (v136_data + (v123_data * v134_data));
          float v139_data = s0[26];
          float v141_data = r1[3];
          r1[3] = (v141_data + (v123_data * v139_data));
          float v144_data = s0[34];
          float v146_data = r1[4];
          r1[4] = (v146_data + (v123_data * v144_data));
          float v149_data = s0[42];
          float v151_data = r1[5];
          r1[5] = (v151_data + (v123_data * v149_data));
          float v154_data = s0[50];
          float v156_data = r1[6];
          r1[6] = (v156_data + (v123_data * v154_data));
          float v159_data = s0[58];
          float v161_data = r1[7];
          r1[7] = (v161_data + (v123_data * v159_data));
          float v163_data = r0[3];
          float v164_data = s0[3];
          float v166_data = r1[0];
          r1[0] = (v166_data + (v163_data * v164_data));
          float v169_data = s0[11];
          float v171_data = r1[1];
          r1[1] = (v171_data + (v163_data * v169_data));
          float v174_data = s0[19];
          float v176_data = r1[2];
          r1[2] = (v176_data + (v163_data * v174_data));
          float v179_data = s0[27];
          float v181_data = r1[3];
          r1[3] = (v181_data + (v163_data * v179_data));
          float v184_data = s0[35];
          float v186_data = r1[4];
          r1[4] = (v186_data + (v163_data * v184_data));
          float v189_data = s0[43];
          float v191_data = r1[5];
          r1[5] = (v191_data + (v163_data * v189_data));
          float v194_data = s0[51];
          float v196_data = r1[6];
          r1[6] = (v196_data + (v163_data * v194_data));
          float v199_data = s0[59];
          float v201_data = r1[7];
          r1[7] = (v201_data + (v163_data * v199_data));
          float v203_data = r0[4];
          float v204_data = s0[4];
          float v206_data = r1[0];
          r1[0] = (v206_data + (v203_data * v204_data));
          float v209_data = s0[12];
          float v211_data = r1[1];
          r1[1] = (v211_data + (v203_data * v209_data));
          float v214_data = s0[20];
          float v216_data = r1[2];
          r1[2] = (v216_data + (v203_data * v214_data));
          float v219_data = s0[28];
          float v221_data = r1[3];
          r1[3] = (v221_data + (v203_data * v219_data));
          float v224_data = s0[36];
          float v226_data = r1[4];
          r1[4] = (v226_data + (v203_data * v224_data));
          float v229_data = s0[44];
          float v231_data = r1[5];
          r1[5] = (v231_data + (v203_data * v229_data));
          float v234_data = s0[52];
          float v236_data = r1[6];
          r1[6] = (v236_data + (v203_data * v234_data));
          float v239_data = s0[60];
          float v241_data = r1[7];
          r1[7] = (v241_data + (v203_data * v239_data));
          float v243_data = r0[5];
          float v244_data = s0[5];
          float v246_data = r1[0];
          r1[0] = (v246_data + (v243_data * v244_data));
          float v249_data = s0[13];
          float v251_data = r1[1];
          r1[1] = (v251_data + (v243_data * v249_data));
          float v254_data = s0[21];
          float v256_data = r1[2];
          r1[2] = (v256_data + (v243_data * v254_data));
          float v259_data = s0[29];
          float v261_data = r1[3];
          r1[3] = (v261_data + (v243_data * v259_data));
          float v264_data = s0[37];
          float v266_data = r1[4];
          r1[4] = (v266_data + (v243_data * v264_data));
          float v269_data = s0[45];
          float v271_data = r1[5];
          r1[5] = (v271_data + (v243_data * v269_data));
          float v274_data = s0[53];
          float v276_data = r1[6];
          r1[6] = (v276_data + (v243_data * v274_data));
          float v279_data = s0[61];
          float v281_data = r1[7];
          r1[7] = (v281_data + (v243_data * v279_data));
          float v283_data = r0[6];
          float v284_data = s0[6];
          float v286_data = r1[0];
          r1[0] = (v286_data + (v283_data * v284_data));
          float v289_data = s0[14];
          float v291_data = r1[1];
          r1[1] = (v291_data + (v283_data * v289_data));
          float v294_data = s0[22];
          float v296_data = r1[2];
          r1[2] = (v296_data + (v283_data * v294_data));
          float v299_data = s0[30];
          float v301_data = r1[3];
          r1[3] = (v301_data + (v283_data * v299_data));
          float v304_data = s0[38];
          float v306_data = r1[4];
          r1[4] = (v306_data + (v283_data * v304_data));
          float v309_data = s0[46];
          float v311_data = r1[5];
          r1[5] = (v311_data + (v283_data * v309_data));
          float v314_data = s0[54];
          float v316_data = r1[6];
          r1[6] = (v316_data + (v283_data * v314_data));
          float v319_data = s0[62];
          float v321_data = r1[7];
          r1[7] = (v321_data + (v283_data * v319_data));
          float v323_data = r0[7];
          float v324_data = s0[7];
          float v326_data = r1[0];
          r1[0] = (v326_data + (v323_data * v324_data));
          float v329_data = s0[15];
          float v331_data = r1[1];
          r1[1] = (v331_data + (v323_data * v329_data));
          float v334_data = s0[23];
          float v336_data = r1[2];
          r1[2] = (v336_data + (v323_data * v334_data));
          float v339_data = s0[31];
          float v341_data = r1[3];
          r1[3] = (v341_data + (v323_data * v339_data));
          float v344_data = s0[39];
          float v346_data = r1[4];
          r1[4] = (v346_data + (v323_data * v344_data));
          float v349_data = s0[47];
          float v351_data = r1[5];
          r1[5] = (v351_data + (v323_data * v349_data));
          float v354_data = s0[55];
          float v356_data = r1[6];
          r1[6] = (v356_data + (v323_data * v354_data));
          float v359_data = s0[63];
          float v361_data = r1[7];
          r1[7] = (v361_data + (v323_data * v359_data));
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          // s2 = load{g>s}(glb_m3[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 8; i += 1) {
            __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + i * 8], &glb_m3[0 + 0 + 1 * threadIdx.x + i * 8], 4);
          }
          __pipeline_commit();
          // wait(r2 = load{g>r}(glb_m2););
          // wait(s2 = load{g>s}(glb_m3[0, 1]));
          __pipeline_wait_prior(0);
          float r3[8]{};
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          // ir3 = +(r2 * s2)
          // [(0, 8), (0, 8)] [(0, 8)]
          float ir3[8]{};
          float v366_data = r2[0];
          float v367_data = s2[0];
          float v369_data = ir3[0];
          ir3[0] = (v369_data + (v366_data * v367_data));
          float v372_data = s2[8];
          float v374_data = ir3[1];
          ir3[1] = (v374_data + (v366_data * v372_data));
          float v377_data = s2[16];
          float v379_data = ir3[2];
          ir3[2] = (v379_data + (v366_data * v377_data));
          float v382_data = s2[24];
          float v384_data = ir3[3];
          ir3[3] = (v384_data + (v366_data * v382_data));
          float v387_data = s2[32];
          float v389_data = ir3[4];
          ir3[4] = (v389_data + (v366_data * v387_data));
          float v392_data = s2[40];
          float v394_data = ir3[5];
          ir3[5] = (v394_data + (v366_data * v392_data));
          float v397_data = s2[48];
          float v399_data = ir3[6];
          ir3[6] = (v399_data + (v366_data * v397_data));
          float v402_data = s2[56];
          float v404_data = ir3[7];
          ir3[7] = (v404_data + (v366_data * v402_data));
          float v406_data = r2[1];
          float v407_data = s2[1];
          float v409_data = ir3[0];
          ir3[0] = (v409_data + (v406_data * v407_data));
          float v412_data = s2[9];
          float v414_data = ir3[1];
          ir3[1] = (v414_data + (v406_data * v412_data));
          float v417_data = s2[17];
          float v419_data = ir3[2];
          ir3[2] = (v419_data + (v406_data * v417_data));
          float v422_data = s2[25];
          float v424_data = ir3[3];
          ir3[3] = (v424_data + (v406_data * v422_data));
          float v427_data = s2[33];
          float v429_data = ir3[4];
          ir3[4] = (v429_data + (v406_data * v427_data));
          float v432_data = s2[41];
          float v434_data = ir3[5];
          ir3[5] = (v434_data + (v406_data * v432_data));
          float v437_data = s2[49];
          float v439_data = ir3[6];
          ir3[6] = (v439_data + (v406_data * v437_data));
          float v442_data = s2[57];
          float v444_data = ir3[7];
          ir3[7] = (v444_data + (v406_data * v442_data));
          float v446_data = r2[2];
          float v447_data = s2[2];
          float v449_data = ir3[0];
          ir3[0] = (v449_data + (v446_data * v447_data));
          float v452_data = s2[10];
          float v454_data = ir3[1];
          ir3[1] = (v454_data + (v446_data * v452_data));
          float v457_data = s2[18];
          float v459_data = ir3[2];
          ir3[2] = (v459_data + (v446_data * v457_data));
          float v462_data = s2[26];
          float v464_data = ir3[3];
          ir3[3] = (v464_data + (v446_data * v462_data));
          float v467_data = s2[34];
          float v469_data = ir3[4];
          ir3[4] = (v469_data + (v446_data * v467_data));
          float v472_data = s2[42];
          float v474_data = ir3[5];
          ir3[5] = (v474_data + (v446_data * v472_data));
          float v477_data = s2[50];
          float v479_data = ir3[6];
          ir3[6] = (v479_data + (v446_data * v477_data));
          float v482_data = s2[58];
          float v484_data = ir3[7];
          ir3[7] = (v484_data + (v446_data * v482_data));
          float v486_data = r2[3];
          float v487_data = s2[3];
          float v489_data = ir3[0];
          ir3[0] = (v489_data + (v486_data * v487_data));
          float v492_data = s2[11];
          float v494_data = ir3[1];
          ir3[1] = (v494_data + (v486_data * v492_data));
          float v497_data = s2[19];
          float v499_data = ir3[2];
          ir3[2] = (v499_data + (v486_data * v497_data));
          float v502_data = s2[27];
          float v504_data = ir3[3];
          ir3[3] = (v504_data + (v486_data * v502_data));
          float v507_data = s2[35];
          float v509_data = ir3[4];
          ir3[4] = (v509_data + (v486_data * v507_data));
          float v512_data = s2[43];
          float v514_data = ir3[5];
          ir3[5] = (v514_data + (v486_data * v512_data));
          float v517_data = s2[51];
          float v519_data = ir3[6];
          ir3[6] = (v519_data + (v486_data * v517_data));
          float v522_data = s2[59];
          float v524_data = ir3[7];
          ir3[7] = (v524_data + (v486_data * v522_data));
          float v526_data = r2[4];
          float v527_data = s2[4];
          float v529_data = ir3[0];
          ir3[0] = (v529_data + (v526_data * v527_data));
          float v532_data = s2[12];
          float v534_data = ir3[1];
          ir3[1] = (v534_data + (v526_data * v532_data));
          float v537_data = s2[20];
          float v539_data = ir3[2];
          ir3[2] = (v539_data + (v526_data * v537_data));
          float v542_data = s2[28];
          float v544_data = ir3[3];
          ir3[3] = (v544_data + (v526_data * v542_data));
          float v547_data = s2[36];
          float v549_data = ir3[4];
          ir3[4] = (v549_data + (v526_data * v547_data));
          float v552_data = s2[44];
          float v554_data = ir3[5];
          ir3[5] = (v554_data + (v526_data * v552_data));
          float v557_data = s2[52];
          float v559_data = ir3[6];
          ir3[6] = (v559_data + (v526_data * v557_data));
          float v562_data = s2[60];
          float v564_data = ir3[7];
          ir3[7] = (v564_data + (v526_data * v562_data));
          float v566_data = r2[5];
          float v567_data = s2[5];
          float v569_data = ir3[0];
          ir3[0] = (v569_data + (v566_data * v567_data));
          float v572_data = s2[13];
          float v574_data = ir3[1];
          ir3[1] = (v574_data + (v566_data * v572_data));
          float v577_data = s2[21];
          float v579_data = ir3[2];
          ir3[2] = (v579_data + (v566_data * v577_data));
          float v582_data = s2[29];
          float v584_data = ir3[3];
          ir3[3] = (v584_data + (v566_data * v582_data));
          float v587_data = s2[37];
          float v589_data = ir3[4];
          ir3[4] = (v589_data + (v566_data * v587_data));
          float v592_data = s2[45];
          float v594_data = ir3[5];
          ir3[5] = (v594_data + (v566_data * v592_data));
          float v597_data = s2[53];
          float v599_data = ir3[6];
          ir3[6] = (v599_data + (v566_data * v597_data));
          float v602_data = s2[61];
          float v604_data = ir3[7];
          ir3[7] = (v604_data + (v566_data * v602_data));
          float v606_data = r2[6];
          float v607_data = s2[6];
          float v609_data = ir3[0];
          ir3[0] = (v609_data + (v606_data * v607_data));
          float v612_data = s2[14];
          float v614_data = ir3[1];
          ir3[1] = (v614_data + (v606_data * v612_data));
          float v617_data = s2[22];
          float v619_data = ir3[2];
          ir3[2] = (v619_data + (v606_data * v617_data));
          float v622_data = s2[30];
          float v624_data = ir3[3];
          ir3[3] = (v624_data + (v606_data * v622_data));
          float v627_data = s2[38];
          float v629_data = ir3[4];
          ir3[4] = (v629_data + (v606_data * v627_data));
          float v632_data = s2[46];
          float v634_data = ir3[5];
          ir3[5] = (v634_data + (v606_data * v632_data));
          float v637_data = s2[54];
          float v639_data = ir3[6];
          ir3[6] = (v639_data + (v606_data * v637_data));
          float v642_data = s2[62];
          float v644_data = ir3[7];
          ir3[7] = (v644_data + (v606_data * v642_data));
          float v646_data = r2[7];
          float v647_data = s2[7];
          float v649_data = ir3[0];
          ir3[0] = (v649_data + (v646_data * v647_data));
          float v652_data = s2[15];
          float v654_data = ir3[1];
          ir3[1] = (v654_data + (v646_data * v652_data));
          float v657_data = s2[23];
          float v659_data = ir3[2];
          ir3[2] = (v659_data + (v646_data * v657_data));
          float v662_data = s2[31];
          float v664_data = ir3[3];
          ir3[3] = (v664_data + (v646_data * v662_data));
          float v667_data = s2[39];
          float v669_data = ir3[4];
          ir3[4] = (v669_data + (v646_data * v667_data));
          float v672_data = s2[47];
          float v674_data = ir3[5];
          ir3[5] = (v674_data + (v646_data * v672_data));
          float v677_data = s2[55];
          float v679_data = ir3[6];
          ir3[6] = (v679_data + (v646_data * v677_data));
          float v682_data = s2[63];
          float v684_data = ir3[7];
          ir3[7] = (v684_data + (v646_data * v682_data));
          // r3 = ir3 + r1
          #pragma unroll
          for (int32_t v686_n0 = 0; v686_n0 < 1; ++v686_n0) {
            #pragma unroll
            for (int32_t v687_n1 = 0; v687_n1 < 8; ++v687_n1) {
              int32_t v688_a = v686_n0 + v687_n1;
              float v689_data = ir3[v688_a];
              float v690_data = r1[v688_a];
              r3[v688_a] = (v690_data + v689_data);
            }
          }
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          // s1 = store{r>s}(localShrMem0, r3);
          #pragma unroll
          for (int32_t v692_i0 = 0; v692_i0 < 1; ++v692_i0) {
            int32_t v697_lead = v23_lead + (v692_i0 * 8);
            #pragma unroll
            for (int32_t v693_i1 = 0; v693_i1 < 8; ++v693_i1) {
              float v695_data = r3[(v692_i0 + v693_i1)];
              int32_t v699_a = v697_lead + (v693_i1 * 8);
              s1[(v699_a ^ ((v699_a >> 5) & 31))] = v695_data;
            }
          }
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          // glb_m4 = abs(s1)
          #pragma unroll
          for (int32_t v703_k0 = 0; v703_k0 < 1; ++v703_k0) {
            int32_t v706_lead = v23_lead + (v703_k0 * 8);
            #pragma unroll
            for (int32_t v704_k1 = 0; v704_k1 < 8; ++v704_k1) {
              int32_t v708_a = v706_lead + (v704_k1 * 8);
              float v712_data = s1[(v708_a ^ ((v708_a >> 5) & 31))];
              glb_m4[v708_a] = (fabsf(v712_data));
            }
          }
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
        }
      }
    }
  }
}

