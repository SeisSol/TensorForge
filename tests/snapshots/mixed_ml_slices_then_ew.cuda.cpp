// === base name ===
kernel_1050140e180923e5

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_1050140e180923e5 = {{8, 16, 1}, 8, 8, 1, 16, 6656, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_1050140e180923e5(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_1050140e180923e5(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_1050140e180923e5(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_1050140e180923e5, block.x * block.y * block.z, 1664 * sizeof(float));
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
  config.sharedMemBytes = 1664 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_1050140e180923e5(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_1050140e180923e5(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_1050140e180923e5, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_1050140e180923e5<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_1050140e180923e5(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 8 lanes x 16 per block = block 8x16x1, 6656 B shared, occupancy grid
    // operands:
    //   m0 8×8(8×8) {0..8}×{0..8} strided
    //   m1 8×4(8×4) {0..8}×{0..4} strided
    //   m2 8×4(8×4) {0..8}×{0..4} strided
    //   m3 8×8(8×8) {0..8}×{0..8} strided
    // operations:
    //   t0[i,j]@{0..8}×{0..4} = m0[i,k] × m1[k,j]
    //   t0[i,j]@{0..8}×{4..8} = m0[i,k] × m2[k,j]
    //   C = abs(TMP)
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1664}],"shared_bytes":6656,"shared_elements":1664,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[8,4]],"name":"m1","ordered":false,"parts":1,"shape":[8,4],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[8,4]],"name":"m2","ordered":false,"parts":1,"shape":[8,4],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,4]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,4]],"is_tmp":true,"name":"t0","offset":[0,4],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[104 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[96];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s2 = &localShrMem0[64];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 64 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 32 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 32 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[v7_batchId0 * 64 + 0 + m3_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v22_lead = threadIdx.x % 8;
          #pragma unroll
          for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
            int32_t v26_lead = v22_lead + (v23_i0 * 8);
            #pragma unroll
            for (int32_t v24_i1 = 0; v24_i1 < 8; ++v24_i1) {
              float v29_data = __ldcg(&glb_m0[(v26_lead + (v24_i1 * 8))]);
              r0[(v23_i0 + v24_i1)] = v29_data;
            }
          }
          // s0 = load{g>s}(glb_m1[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m1[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 8], &glb_m1[0 + 0 + 1 * threadIdx.x + 8], 4);
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 16], &glb_m1[0 + 0 + 1 * threadIdx.x + 16], 4);
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 24], &glb_m1[0 + 0 + 1 * threadIdx.x + 24], 4);
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m0););
          // s2 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 8], &glb_m2[0 + 0 + 1 * threadIdx.x + 8], 4);
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 16], &glb_m2[0 + 0 + 1 * threadIdx.x + 16], 4);
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 24], &glb_m2[0 + 0 + 1 * threadIdx.x + 24], 4);
          __pipeline_commit();
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(1);
          float r1[4]{};
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          // r1 = +(r0 * s0) + None
          // [(0, 8), (0, 4)] [(0, 8)]
          float v40_data = r0[0];
          float v41_data = s0[0];
          float v43_data = r1[0];
          r1[0] = (v43_data + (v40_data * v41_data));
          float v46_data = s0[8];
          float v48_data = r1[1];
          r1[1] = (v48_data + (v40_data * v46_data));
          float v51_data = s0[16];
          float v53_data = r1[2];
          r1[2] = (v53_data + (v40_data * v51_data));
          float v56_data = s0[24];
          float v58_data = r1[3];
          r1[3] = (v58_data + (v40_data * v56_data));
          float v60_data = r0[1];
          float v61_data = s0[1];
          float v63_data = r1[0];
          r1[0] = (v63_data + (v60_data * v61_data));
          float v66_data = s0[9];
          float v68_data = r1[1];
          r1[1] = (v68_data + (v60_data * v66_data));
          float v71_data = s0[17];
          float v73_data = r1[2];
          r1[2] = (v73_data + (v60_data * v71_data));
          float v76_data = s0[25];
          float v78_data = r1[3];
          r1[3] = (v78_data + (v60_data * v76_data));
          float v80_data = r0[2];
          float v81_data = s0[2];
          float v83_data = r1[0];
          r1[0] = (v83_data + (v80_data * v81_data));
          float v86_data = s0[10];
          float v88_data = r1[1];
          r1[1] = (v88_data + (v80_data * v86_data));
          float v91_data = s0[18];
          float v93_data = r1[2];
          r1[2] = (v93_data + (v80_data * v91_data));
          float v96_data = s0[26];
          float v98_data = r1[3];
          r1[3] = (v98_data + (v80_data * v96_data));
          float v100_data = r0[3];
          float v101_data = s0[3];
          float v103_data = r1[0];
          r1[0] = (v103_data + (v100_data * v101_data));
          float v106_data = s0[11];
          float v108_data = r1[1];
          r1[1] = (v108_data + (v100_data * v106_data));
          float v111_data = s0[19];
          float v113_data = r1[2];
          r1[2] = (v113_data + (v100_data * v111_data));
          float v116_data = s0[27];
          float v118_data = r1[3];
          r1[3] = (v118_data + (v100_data * v116_data));
          float v120_data = r0[4];
          float v121_data = s0[4];
          float v123_data = r1[0];
          r1[0] = (v123_data + (v120_data * v121_data));
          float v126_data = s0[12];
          float v128_data = r1[1];
          r1[1] = (v128_data + (v120_data * v126_data));
          float v131_data = s0[20];
          float v133_data = r1[2];
          r1[2] = (v133_data + (v120_data * v131_data));
          float v136_data = s0[28];
          float v138_data = r1[3];
          r1[3] = (v138_data + (v120_data * v136_data));
          float v140_data = r0[5];
          float v141_data = s0[5];
          float v143_data = r1[0];
          r1[0] = (v143_data + (v140_data * v141_data));
          float v146_data = s0[13];
          float v148_data = r1[1];
          r1[1] = (v148_data + (v140_data * v146_data));
          float v151_data = s0[21];
          float v153_data = r1[2];
          r1[2] = (v153_data + (v140_data * v151_data));
          float v156_data = s0[29];
          float v158_data = r1[3];
          r1[3] = (v158_data + (v140_data * v156_data));
          float v160_data = r0[6];
          float v161_data = s0[6];
          float v163_data = r1[0];
          r1[0] = (v163_data + (v160_data * v161_data));
          float v166_data = s0[14];
          float v168_data = r1[1];
          r1[1] = (v168_data + (v160_data * v166_data));
          float v171_data = s0[22];
          float v173_data = r1[2];
          r1[2] = (v173_data + (v160_data * v171_data));
          float v176_data = s0[30];
          float v178_data = r1[3];
          r1[3] = (v178_data + (v160_data * v176_data));
          float v180_data = r0[7];
          float v181_data = s0[7];
          float v183_data = r1[0];
          r1[0] = (v183_data + (v180_data * v181_data));
          float v186_data = s0[15];
          float v188_data = r1[1];
          r1[1] = (v188_data + (v180_data * v186_data));
          float v191_data = s0[23];
          float v193_data = r1[2];
          r1[2] = (v193_data + (v180_data * v191_data));
          float v196_data = s0[31];
          float v198_data = r1[3];
          r1[3] = (v198_data + (v180_data * v196_data));
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          // s1 = store{r>s}(localShrMem0, r1);
          #pragma unroll
          for (int32_t v200_i0 = 0; v200_i0 < 1; ++v200_i0) {
            int32_t v205_lead = v22_lead + (v200_i0 * 8);
            #pragma unroll
            for (int32_t v201_i1 = 0; v201_i1 < 4; ++v201_i1) {
              float v203_data = r1[(v200_i0 + v201_i1)];
              int32_t v207_a = v205_lead + (v201_i1 * 8);
              s1[(v207_a ^ ((v207_a >> 5) & 31))] = v203_data;
            }
          }
          // wait(s2 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r2[4]{};
          // ir2 = +(r0 * s2)
          // [(0, 8), (0, 4)] [(0, 8)]
          float ir2[4]{};
          float v214_data = s2[0];
          float v216_data = ir2[0];
          ir2[0] = (v216_data + (v40_data * v214_data));
          float v219_data = s2[8];
          float v221_data = ir2[1];
          ir2[1] = (v221_data + (v40_data * v219_data));
          float v224_data = s2[16];
          float v226_data = ir2[2];
          ir2[2] = (v226_data + (v40_data * v224_data));
          float v229_data = s2[24];
          float v231_data = ir2[3];
          ir2[3] = (v231_data + (v40_data * v229_data));
          float v234_data = s2[1];
          float v236_data = ir2[0];
          ir2[0] = (v236_data + (v60_data * v234_data));
          float v239_data = s2[9];
          float v241_data = ir2[1];
          ir2[1] = (v241_data + (v60_data * v239_data));
          float v244_data = s2[17];
          float v246_data = ir2[2];
          ir2[2] = (v246_data + (v60_data * v244_data));
          float v249_data = s2[25];
          float v251_data = ir2[3];
          ir2[3] = (v251_data + (v60_data * v249_data));
          float v254_data = s2[2];
          float v256_data = ir2[0];
          ir2[0] = (v256_data + (v80_data * v254_data));
          float v259_data = s2[10];
          float v261_data = ir2[1];
          ir2[1] = (v261_data + (v80_data * v259_data));
          float v264_data = s2[18];
          float v266_data = ir2[2];
          ir2[2] = (v266_data + (v80_data * v264_data));
          float v269_data = s2[26];
          float v271_data = ir2[3];
          ir2[3] = (v271_data + (v80_data * v269_data));
          float v274_data = s2[3];
          float v276_data = ir2[0];
          ir2[0] = (v276_data + (v100_data * v274_data));
          float v279_data = s2[11];
          float v281_data = ir2[1];
          ir2[1] = (v281_data + (v100_data * v279_data));
          float v284_data = s2[19];
          float v286_data = ir2[2];
          ir2[2] = (v286_data + (v100_data * v284_data));
          float v289_data = s2[27];
          float v291_data = ir2[3];
          ir2[3] = (v291_data + (v100_data * v289_data));
          float v294_data = s2[4];
          float v296_data = ir2[0];
          ir2[0] = (v296_data + (v120_data * v294_data));
          float v299_data = s2[12];
          float v301_data = ir2[1];
          ir2[1] = (v301_data + (v120_data * v299_data));
          float v304_data = s2[20];
          float v306_data = ir2[2];
          ir2[2] = (v306_data + (v120_data * v304_data));
          float v309_data = s2[28];
          float v311_data = ir2[3];
          ir2[3] = (v311_data + (v120_data * v309_data));
          float v314_data = s2[5];
          float v316_data = ir2[0];
          ir2[0] = (v316_data + (v140_data * v314_data));
          float v319_data = s2[13];
          float v321_data = ir2[1];
          ir2[1] = (v321_data + (v140_data * v319_data));
          float v324_data = s2[21];
          float v326_data = ir2[2];
          ir2[2] = (v326_data + (v140_data * v324_data));
          float v329_data = s2[29];
          float v331_data = ir2[3];
          ir2[3] = (v331_data + (v140_data * v329_data));
          float v334_data = s2[6];
          float v336_data = ir2[0];
          ir2[0] = (v336_data + (v160_data * v334_data));
          float v339_data = s2[14];
          float v341_data = ir2[1];
          ir2[1] = (v341_data + (v160_data * v339_data));
          float v344_data = s2[22];
          float v346_data = ir2[2];
          ir2[2] = (v346_data + (v160_data * v344_data));
          float v349_data = s2[30];
          float v351_data = ir2[3];
          ir2[3] = (v351_data + (v160_data * v349_data));
          float v354_data = s2[7];
          float v356_data = ir2[0];
          ir2[0] = (v356_data + (v180_data * v354_data));
          float v359_data = s2[15];
          float v361_data = ir2[1];
          ir2[1] = (v361_data + (v180_data * v359_data));
          float v364_data = s2[23];
          float v366_data = ir2[2];
          ir2[2] = (v366_data + (v180_data * v364_data));
          float v369_data = s2[31];
          float v371_data = ir2[3];
          ir2[3] = (v371_data + (v180_data * v369_data));
          // r2 = ir2
          #pragma unroll
          for (int32_t v373_n0 = 0; v373_n0 < 1; ++v373_n0) {
            #pragma unroll
            for (int32_t v374_n1 = 0; v374_n1 < 4; ++v374_n1) {
              int32_t v375_a = v373_n0 + v374_n1;
              float v376_data = ir2[v375_a];
              r2[v375_a] = v376_data;
            }
          }
          // s1 = store{r>s}(localShrMem0, r2);
          #pragma unroll
          for (int32_t v377_i0 = 0; v377_i0 < 1; ++v377_i0) {
            int32_t v382_lead = v22_lead + (v377_i0 * 8);
            #pragma unroll
            for (int32_t v378_i1 = 0; v378_i1 < 4; ++v378_i1) {
              float v380_data = r2[(v377_i0 + v378_i1)];
              int32_t v385_a = v382_lead + ((v378_i1 + 4) * 8);
              s1[(v385_a ^ ((v385_a >> 5) & 31))] = v380_data;
            }
          }
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          // glb_m3 = abs(s1)
          #pragma unroll
          for (int32_t v389_k0 = 0; v389_k0 < 1; ++v389_k0) {
            int32_t v392_lead = v22_lead + (v389_k0 * 8);
            #pragma unroll
            for (int32_t v390_k1 = 0; v390_k1 < 8; ++v390_k1) {
              int32_t v394_a = v392_lead + (v390_k1 * 8);
              float v398_data = s1[(v394_a ^ ((v394_a >> 5) & 31))];
              glb_m3[v394_a] = (fabsf(v398_data));
            }
          }
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
        }
      }
    }
  }
}

