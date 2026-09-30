// === base name ===
kernel_599682bdf328e72c

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_599682bdf328e72c = {{32, 4, 1}, 32, 32, 1, 4, 1536, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_599682bdf328e72c(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_599682bdf328e72c(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_599682bdf328e72c(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_599682bdf328e72c, block.x * block.y * block.z, 384 * sizeof(float));
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
  config.sharedMemBytes = 384 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_599682bdf328e72c(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_599682bdf328e72c(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_599682bdf328e72c, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_599682bdf328e72c<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_599682bdf328e72c(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 4 per block = block 32x4x1, 1536 B shared, occupancy grid
    // operands:
    //   m0 32×9(32×9) {0..32}×{0..9} pointer_based
    //   m1 16×9(16×9) {0..16}×{0..9} pointer_based
    //   m2 16×9(16×9) {0..16}×{0..9} pointer_based
    //   m3 32×9(32×9) {0..32}×{0..9} pointer_based
    //   m4 9×9(9×9) {0..9}×{0..9} pointer_based
    // operations:
    //   t0[i,j] = m0[i,j]
    //   t0[i,j] += m1[i,j]
    //   t0[i,j] += m2[i,j]
    //   m3[i,j] = t0[i,k] × m4[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":384}],"shared_bytes":1536,"shared_elements":384,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"pointer_based","alias":"Q","bbox":[[0,0],[32,9]],"name":"m0","ordered":false,"parts":1,"shape":[32,9],"variant":false},{"addressing":"pointer_based","alias":"F0","bbox":[[0,0],[16,9]],"name":"m1","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"pointer_based","alias":"F1","bbox":[[0,0],[16,9]],"name":"m2","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"pointer_based","alias":"O","bbox":[[0,0],[32,9]],"name":"m3","ordered":false,"parts":1,"shape":[32,9],"variant":false},{"addressing":"pointer_based","alias":"M","bbox":[[0,0],[9,9]],"name":"m4","ordered":false,"parts":1,"shape":[9,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[32,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[32,9]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[96 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[96];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v5_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v5_batchId0 < numElements0; v5_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v6_ahead1 = v5_batchId0 + (gridDim.x * blockDim.y);
        size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v5_batchId0][0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v5_batchId0][0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v5_batchId0][0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[v5_batchId0][0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[v5_batchId0][0 + m4_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v21_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v22_i0 = 0; v22_i0 < 1; ++v22_i0) {
            int32_t v25_lead = v21_lead + (v22_i0 * 32);
            #pragma unroll
            for (int32_t v23_i1 = 0; v23_i1 < 9; ++v23_i1) {
              float v28_data = __ldcg(&glb_m0[(v25_lead + (v23_i1 * 32))]);
              r0[(v22_i0 + v23_i1)] = v28_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m1);
          bool v31_g = v21_lead < 16;
          if (v31_g) {
            #pragma unroll
            for (int32_t v32_i1 = 0; v32_i1 < 9; ++v32_i1) {
              float v37_data = __ldcg(&glb_m1[(v21_lead + (v32_i1 * 16))]);
              r2[v32_i1] = v37_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 9)] []
          float v40_data = r0[0];
          float v41_data = r1[0];
          r1[0] = (v41_data + v40_data);
          float v43_data = r0[1];
          float v44_data = r1[1];
          r1[1] = (v44_data + v43_data);
          float v46_data = r0[2];
          float v47_data = r1[2];
          r1[2] = (v47_data + v46_data);
          float v49_data = r0[3];
          float v50_data = r1[3];
          r1[3] = (v50_data + v49_data);
          float v52_data = r0[4];
          float v53_data = r1[4];
          r1[4] = (v53_data + v52_data);
          float v55_data = r0[5];
          float v56_data = r1[5];
          r1[5] = (v56_data + v55_data);
          float v58_data = r0[6];
          float v59_data = r1[6];
          r1[6] = (v59_data + v58_data);
          float v61_data = r0[7];
          float v62_data = r1[7];
          r1[7] = (v62_data + v61_data);
          float v64_data = r0[8];
          float v65_data = r1[8];
          r1[8] = (v65_data + v64_data);
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          if (v31_g) {
            #pragma unroll
            for (int32_t v68_i1 = 0; v68_i1 < 9; ++v68_i1) {
              float v73_data = __ldcg(&glb_m2[(v21_lead + (v68_i1 * 16))]);
              r4[v68_i1] = v73_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          // ir3 = +(r2)
          // [(0, 16), (0, 9)] []
          float ir3[9]{};
          float v77_data = r2[0];
          float v78_data = ir3[0];
          ir3[0] = (v78_data + v77_data);
          float v80_data = r2[1];
          float v81_data = ir3[1];
          ir3[1] = (v81_data + v80_data);
          float v83_data = r2[2];
          float v84_data = ir3[2];
          ir3[2] = (v84_data + v83_data);
          float v86_data = r2[3];
          float v87_data = ir3[3];
          ir3[3] = (v87_data + v86_data);
          float v89_data = r2[4];
          float v90_data = ir3[4];
          ir3[4] = (v90_data + v89_data);
          float v92_data = r2[5];
          float v93_data = ir3[5];
          ir3[5] = (v93_data + v92_data);
          float v95_data = r2[6];
          float v96_data = ir3[6];
          ir3[6] = (v96_data + v95_data);
          float v98_data = r2[7];
          float v99_data = ir3[7];
          ir3[7] = (v99_data + v98_data);
          float v101_data = r2[8];
          float v102_data = ir3[8];
          ir3[8] = (v102_data + v101_data);
          // r3 = ir3 + r1
          #pragma unroll
          for (int32_t v104_n1 = 0; v104_n1 < 9; ++v104_n1) {
            float v106_data = ir3[v104_n1];
            float v107_data = r1[v104_n1];
            r3[v104_n1] = (v107_data + v106_data);
          }
          // s1 = load{g>s}(glb_m4[0, 1])
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 0], &glb_m4[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 32], &glb_m4[0 + 0 + 1 * threadIdx.x + 32], 4);
          if (threadIdx.x < 17) {
            __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 64], &glb_m4[0 + 0 + 1 * threadIdx.x + 64], 4);
          }
          __pipeline_commit();
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          // ir5 = +(r4)
          // [(0, 16), (0, 9)] []
          float ir5[9]{};
          float v114_data = r4[0];
          float v115_data = ir5[0];
          ir5[0] = (v115_data + v114_data);
          float v117_data = r4[1];
          float v118_data = ir5[1];
          ir5[1] = (v118_data + v117_data);
          float v120_data = r4[2];
          float v121_data = ir5[2];
          ir5[2] = (v121_data + v120_data);
          float v123_data = r4[3];
          float v124_data = ir5[3];
          ir5[3] = (v124_data + v123_data);
          float v126_data = r4[4];
          float v127_data = ir5[4];
          ir5[4] = (v127_data + v126_data);
          float v129_data = r4[5];
          float v130_data = ir5[5];
          ir5[5] = (v130_data + v129_data);
          float v132_data = r4[6];
          float v133_data = ir5[6];
          ir5[6] = (v133_data + v132_data);
          float v135_data = r4[7];
          float v136_data = ir5[7];
          ir5[7] = (v136_data + v135_data);
          float v138_data = r4[8];
          float v139_data = ir5[8];
          ir5[8] = (v139_data + v138_data);
          // r5 = ir5 + r3
          #pragma unroll
          for (int32_t v141_n1 = 0; v141_n1 < 9; ++v141_n1) {
            float v143_data = ir5[v141_n1];
            float v144_data = r3[v141_n1];
            r5[v141_n1] = (v144_data + v143_data);
          }
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          float r6[9]{};
          __syncwarp();
          // ir6 = +(r5 * s1)
          // [(0, 32), (0, 9)] [(0, 9)]
          float ir6[9]{};
          float v148_data = r5[0];
          float v149_data = s1[0];
          float v151_data = ir6[0];
          ir6[0] = (v151_data + (v148_data * v149_data));
          float v154_data = s1[9];
          float v156_data = ir6[1];
          ir6[1] = (v156_data + (v148_data * v154_data));
          float v159_data = s1[18];
          float v161_data = ir6[2];
          ir6[2] = (v161_data + (v148_data * v159_data));
          float v164_data = s1[27];
          float v166_data = ir6[3];
          ir6[3] = (v166_data + (v148_data * v164_data));
          float v169_data = s1[36];
          float v171_data = ir6[4];
          ir6[4] = (v171_data + (v148_data * v169_data));
          float v174_data = s1[45];
          float v176_data = ir6[5];
          ir6[5] = (v176_data + (v148_data * v174_data));
          float v179_data = s1[54];
          float v181_data = ir6[6];
          ir6[6] = (v181_data + (v148_data * v179_data));
          float v184_data = s1[63];
          float v186_data = ir6[7];
          ir6[7] = (v186_data + (v148_data * v184_data));
          float v189_data = s1[72];
          float v191_data = ir6[8];
          ir6[8] = (v191_data + (v148_data * v189_data));
          float v193_data = r5[1];
          float v194_data = s1[1];
          float v196_data = ir6[0];
          ir6[0] = (v196_data + (v193_data * v194_data));
          float v199_data = s1[10];
          float v201_data = ir6[1];
          ir6[1] = (v201_data + (v193_data * v199_data));
          float v204_data = s1[19];
          float v206_data = ir6[2];
          ir6[2] = (v206_data + (v193_data * v204_data));
          float v209_data = s1[28];
          float v211_data = ir6[3];
          ir6[3] = (v211_data + (v193_data * v209_data));
          float v214_data = s1[37];
          float v216_data = ir6[4];
          ir6[4] = (v216_data + (v193_data * v214_data));
          float v219_data = s1[46];
          float v221_data = ir6[5];
          ir6[5] = (v221_data + (v193_data * v219_data));
          float v224_data = s1[55];
          float v226_data = ir6[6];
          ir6[6] = (v226_data + (v193_data * v224_data));
          float v229_data = s1[64];
          float v231_data = ir6[7];
          ir6[7] = (v231_data + (v193_data * v229_data));
          float v234_data = s1[73];
          float v236_data = ir6[8];
          ir6[8] = (v236_data + (v193_data * v234_data));
          float v238_data = r5[2];
          float v239_data = s1[2];
          float v241_data = ir6[0];
          ir6[0] = (v241_data + (v238_data * v239_data));
          float v244_data = s1[11];
          float v246_data = ir6[1];
          ir6[1] = (v246_data + (v238_data * v244_data));
          float v249_data = s1[20];
          float v251_data = ir6[2];
          ir6[2] = (v251_data + (v238_data * v249_data));
          float v254_data = s1[29];
          float v256_data = ir6[3];
          ir6[3] = (v256_data + (v238_data * v254_data));
          float v259_data = s1[38];
          float v261_data = ir6[4];
          ir6[4] = (v261_data + (v238_data * v259_data));
          float v264_data = s1[47];
          float v266_data = ir6[5];
          ir6[5] = (v266_data + (v238_data * v264_data));
          float v269_data = s1[56];
          float v271_data = ir6[6];
          ir6[6] = (v271_data + (v238_data * v269_data));
          float v274_data = s1[65];
          float v276_data = ir6[7];
          ir6[7] = (v276_data + (v238_data * v274_data));
          float v279_data = s1[74];
          float v281_data = ir6[8];
          ir6[8] = (v281_data + (v238_data * v279_data));
          float v283_data = r5[3];
          float v284_data = s1[3];
          float v286_data = ir6[0];
          ir6[0] = (v286_data + (v283_data * v284_data));
          float v289_data = s1[12];
          float v291_data = ir6[1];
          ir6[1] = (v291_data + (v283_data * v289_data));
          float v294_data = s1[21];
          float v296_data = ir6[2];
          ir6[2] = (v296_data + (v283_data * v294_data));
          float v299_data = s1[30];
          float v301_data = ir6[3];
          ir6[3] = (v301_data + (v283_data * v299_data));
          float v304_data = s1[39];
          float v306_data = ir6[4];
          ir6[4] = (v306_data + (v283_data * v304_data));
          float v309_data = s1[48];
          float v311_data = ir6[5];
          ir6[5] = (v311_data + (v283_data * v309_data));
          float v314_data = s1[57];
          float v316_data = ir6[6];
          ir6[6] = (v316_data + (v283_data * v314_data));
          float v319_data = s1[66];
          float v321_data = ir6[7];
          ir6[7] = (v321_data + (v283_data * v319_data));
          float v324_data = s1[75];
          float v326_data = ir6[8];
          ir6[8] = (v326_data + (v283_data * v324_data));
          float v328_data = r5[4];
          float v329_data = s1[4];
          float v331_data = ir6[0];
          ir6[0] = (v331_data + (v328_data * v329_data));
          float v334_data = s1[13];
          float v336_data = ir6[1];
          ir6[1] = (v336_data + (v328_data * v334_data));
          float v339_data = s1[22];
          float v341_data = ir6[2];
          ir6[2] = (v341_data + (v328_data * v339_data));
          float v344_data = s1[31];
          float v346_data = ir6[3];
          ir6[3] = (v346_data + (v328_data * v344_data));
          float v349_data = s1[40];
          float v351_data = ir6[4];
          ir6[4] = (v351_data + (v328_data * v349_data));
          float v354_data = s1[49];
          float v356_data = ir6[5];
          ir6[5] = (v356_data + (v328_data * v354_data));
          float v359_data = s1[58];
          float v361_data = ir6[6];
          ir6[6] = (v361_data + (v328_data * v359_data));
          float v364_data = s1[67];
          float v366_data = ir6[7];
          ir6[7] = (v366_data + (v328_data * v364_data));
          float v369_data = s1[76];
          float v371_data = ir6[8];
          ir6[8] = (v371_data + (v328_data * v369_data));
          float v373_data = r5[5];
          float v374_data = s1[5];
          float v376_data = ir6[0];
          ir6[0] = (v376_data + (v373_data * v374_data));
          float v379_data = s1[14];
          float v381_data = ir6[1];
          ir6[1] = (v381_data + (v373_data * v379_data));
          float v384_data = s1[23];
          float v386_data = ir6[2];
          ir6[2] = (v386_data + (v373_data * v384_data));
          float v389_data = s1[32];
          float v391_data = ir6[3];
          ir6[3] = (v391_data + (v373_data * v389_data));
          float v394_data = s1[41];
          float v396_data = ir6[4];
          ir6[4] = (v396_data + (v373_data * v394_data));
          float v399_data = s1[50];
          float v401_data = ir6[5];
          ir6[5] = (v401_data + (v373_data * v399_data));
          float v404_data = s1[59];
          float v406_data = ir6[6];
          ir6[6] = (v406_data + (v373_data * v404_data));
          float v409_data = s1[68];
          float v411_data = ir6[7];
          ir6[7] = (v411_data + (v373_data * v409_data));
          float v414_data = s1[77];
          float v416_data = ir6[8];
          ir6[8] = (v416_data + (v373_data * v414_data));
          float v418_data = r5[6];
          float v419_data = s1[6];
          float v421_data = ir6[0];
          ir6[0] = (v421_data + (v418_data * v419_data));
          float v424_data = s1[15];
          float v426_data = ir6[1];
          ir6[1] = (v426_data + (v418_data * v424_data));
          float v429_data = s1[24];
          float v431_data = ir6[2];
          ir6[2] = (v431_data + (v418_data * v429_data));
          float v434_data = s1[33];
          float v436_data = ir6[3];
          ir6[3] = (v436_data + (v418_data * v434_data));
          float v439_data = s1[42];
          float v441_data = ir6[4];
          ir6[4] = (v441_data + (v418_data * v439_data));
          float v444_data = s1[51];
          float v446_data = ir6[5];
          ir6[5] = (v446_data + (v418_data * v444_data));
          float v449_data = s1[60];
          float v451_data = ir6[6];
          ir6[6] = (v451_data + (v418_data * v449_data));
          float v454_data = s1[69];
          float v456_data = ir6[7];
          ir6[7] = (v456_data + (v418_data * v454_data));
          float v459_data = s1[78];
          float v461_data = ir6[8];
          ir6[8] = (v461_data + (v418_data * v459_data));
          float v463_data = r5[7];
          float v464_data = s1[7];
          float v466_data = ir6[0];
          ir6[0] = (v466_data + (v463_data * v464_data));
          float v469_data = s1[16];
          float v471_data = ir6[1];
          ir6[1] = (v471_data + (v463_data * v469_data));
          float v474_data = s1[25];
          float v476_data = ir6[2];
          ir6[2] = (v476_data + (v463_data * v474_data));
          float v479_data = s1[34];
          float v481_data = ir6[3];
          ir6[3] = (v481_data + (v463_data * v479_data));
          float v484_data = s1[43];
          float v486_data = ir6[4];
          ir6[4] = (v486_data + (v463_data * v484_data));
          float v489_data = s1[52];
          float v491_data = ir6[5];
          ir6[5] = (v491_data + (v463_data * v489_data));
          float v494_data = s1[61];
          float v496_data = ir6[6];
          ir6[6] = (v496_data + (v463_data * v494_data));
          float v499_data = s1[70];
          float v501_data = ir6[7];
          ir6[7] = (v501_data + (v463_data * v499_data));
          float v504_data = s1[79];
          float v506_data = ir6[8];
          ir6[8] = (v506_data + (v463_data * v504_data));
          float v508_data = r5[8];
          float v509_data = s1[8];
          float v511_data = ir6[0];
          ir6[0] = (v511_data + (v508_data * v509_data));
          float v514_data = s1[17];
          float v516_data = ir6[1];
          ir6[1] = (v516_data + (v508_data * v514_data));
          float v519_data = s1[26];
          float v521_data = ir6[2];
          ir6[2] = (v521_data + (v508_data * v519_data));
          float v524_data = s1[35];
          float v526_data = ir6[3];
          ir6[3] = (v526_data + (v508_data * v524_data));
          float v529_data = s1[44];
          float v531_data = ir6[4];
          ir6[4] = (v531_data + (v508_data * v529_data));
          float v534_data = s1[53];
          float v536_data = ir6[5];
          ir6[5] = (v536_data + (v508_data * v534_data));
          float v539_data = s1[62];
          float v541_data = ir6[6];
          ir6[6] = (v541_data + (v508_data * v539_data));
          float v544_data = s1[71];
          float v546_data = ir6[7];
          ir6[7] = (v546_data + (v508_data * v544_data));
          float v549_data = s1[80];
          float v551_data = ir6[8];
          ir6[8] = (v551_data + (v508_data * v549_data));
          // r6 = ir6
          #pragma unroll
          for (int32_t v553_n0 = 0; v553_n0 < 1; ++v553_n0) {
            #pragma unroll
            for (int32_t v554_n1 = 0; v554_n1 < 9; ++v554_n1) {
              int32_t v555_a = v553_n0 + v554_n1;
              float v556_data = ir6[v555_a];
              r6[v555_a] = v556_data;
            }
          }
          // glb_m3 = store{r>g}(r6);
          #pragma unroll
          for (int32_t v557_i0 = 0; v557_i0 < 1; ++v557_i0) {
            int32_t v562_lead = v21_lead + (v557_i0 * 32);
            #pragma unroll
            for (int32_t v558_i1 = 0; v558_i1 < 9; ++v558_i1) {
              float v560_data = r6[(v557_i0 + v558_i1)];
              glb_m3[(v562_lead + (v558_i1 * 32))] = v560_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

