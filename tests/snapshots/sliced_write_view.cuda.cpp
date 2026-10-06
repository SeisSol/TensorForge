// === base name ===
kernel_da600bc2e434f40b

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_da600bc2e434f40b = {{32, 4, 1}, 32, 32, 1, 4, 3072, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_da600bc2e434f40b(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_da600bc2e434f40b(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_da600bc2e434f40b(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_da600bc2e434f40b, block.x * block.y * block.z, 768 * sizeof(float));
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
  config.sharedMemBytes = 768 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_da600bc2e434f40b(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_da600bc2e434f40b(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_da600bc2e434f40b, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_da600bc2e434f40b<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_da600bc2e434f40b(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 4 per block = block 32x4x1, 3072 B shared, occupancy grid
    // operands:
    //   m0 32×13(32×13) {0..32}×{0..13} strided
    //   m1 32×13(32×13) {0..32}×{0..13} strided
    //   m2 13×13(13×13) {0..13}×{0..13} strided
    //   m3 32×13(32×13) {0..32}×{0..13} strided
    //   m4 13×13(13×13) {0..13}×{0..13} strided
    // operations:
    //   m0[i,j]@{0..32}×{8..9} = m1[i,k]@{0..32}×{10..13} × m2[k,j]@{10..13}×{8..9}
    //   m3[i,j] = m0[i,k] × m4[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":768}],"shared_bytes":3072,"shared_elements":768,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[32,13]],"name":"m0","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"Q","bbox":[[0,0],[32,13]],"name":"m1","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"S","bbox":[[0,0],[13,13]],"name":"m2","ordered":false,"parts":1,"shape":[13,13],"variant":false},{"addressing":"strided","alias":"O","bbox":[[0,0],[32,13]],"name":"m3","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[13,13]],"name":"m4","ordered":false,"parts":1,"shape":[13,13],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,1]],"is_tmp":false,"name":"m0","offset":[0,8],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,10],[32,13]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[10,0],[13,1]],"is_tmp":false,"name":"m2","offset":[0,8],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[0,0],[13,13]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[192 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[192];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v12_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v12_batchId0 < numElements0; v12_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v13_ahead1 = v12_batchId0 + (gridDim.x * blockDim.y);
        size_t v15_batchId1 = (v13_ahead1 < numElements0) ? v13_ahead1 : v12_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v12_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v12_batchId0 * 416 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v12_batchId0 * 416 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v12_batchId0 * 169 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[v12_batchId0 * 416 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[v12_batchId0 * 169 + 0 + m4_extraOffset];
          float r0[3]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v28_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v29_i0 = 0; v29_i0 < 1; ++v29_i0) {
            int32_t v32_lead = v28_lead + (v29_i0 * 32);
            #pragma unroll
            for (int32_t v30_i1 = 10; v30_i1 < 13; ++v30_i1) {
              float v35_data = __ldcg(&glb_m1[(v32_lead + (v30_i1 * 32))]);
              r0[(v29_i0 + (v30_i1 - 10))] = v35_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 5; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 32], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 32], 4);
          }
          if (threadIdx.x < 9) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 160], &glb_m2[0 + 0 + 1 * threadIdx.x + 160], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[1]{};
          __syncwarp();
          // ir1 = +(r0 * s0)
          // [(0, 32), (0, 1)] [(10, 13)]
          float ir1[1]{};
          float v42_data = r0[0];
          float v43_data = s0[114];
          float v45_data = ir1[0];
          ir1[0] = (v45_data + (v42_data * v43_data));
          float v47_data = r0[1];
          float v48_data = s0[115];
          float v50_data = ir1[0];
          ir1[0] = (v50_data + (v47_data * v48_data));
          float v52_data = r0[2];
          float v53_data = s0[116];
          float v55_data = ir1[0];
          ir1[0] = (v55_data + (v52_data * v53_data));
          // r1 = ir1
          #pragma unroll
          for (int32_t v57_n0 = 0; v57_n0 < 1; ++v57_n0) {
            #pragma unroll
            for (int32_t v58_n1 = 0; v58_n1 < 1; ++v58_n1) {
              int32_t v59_a = v57_n0 + v58_n1;
              float v60_data = ir1[v59_a];
              r1[v59_a] = v60_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v61_i0 = 0; v61_i0 < 1; ++v61_i0) {
            int32_t v66_lead = v28_lead + (v61_i0 * 32);
            #pragma unroll
            for (int32_t v62_i1 = 0; v62_i1 < 1; ++v62_i1) {
              float v64_data = r1[(v61_i0 + v62_i1)];
              glb_m0[(v66_lead + ((v62_i1 + 8) * 32))] = v64_data;
            }
          }
          float r2[13]{};
          // r2 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v71_i0 = 0; v71_i0 < 1; ++v71_i0) {
            int32_t v74_lead = v28_lead + (v71_i0 * 32);
            #pragma unroll
            for (int32_t v72_i1 = 0; v72_i1 < 13; ++v72_i1) {
              float v77_data = glb_m0[(v74_lead + (v72_i1 * 32))];
              r2[(v71_i0 + v72_i1)] = v77_data;
            }
          }
          __syncwarp();
          // s1 = load{g>s}(glb_m4[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 5; i += 1) {
            __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + i * 32], &glb_m4[0 + 0 + 1 * threadIdx.x + i * 32], 4);
          }
          if (threadIdx.x < 9) {
            __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 160], &glb_m4[0 + 0 + 1 * threadIdx.x + 160], 4);
          }
          __pipeline_commit();
          // wait(r2 = load{g>r}(glb_m0););
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          float r3[13]{};
          __syncwarp();
          // ir3 = +(r2 * s1)
          // [(0, 32), (0, 13)] [(0, 13)]
          float ir3[13]{};
          float v83_data = r2[0];
          float v84_data = s1[0];
          float v86_data = ir3[0];
          ir3[0] = (v86_data + (v83_data * v84_data));
          float v89_data = s1[13];
          float v91_data = ir3[1];
          ir3[1] = (v91_data + (v83_data * v89_data));
          float v94_data = s1[26];
          float v96_data = ir3[2];
          ir3[2] = (v96_data + (v83_data * v94_data));
          float v99_data = s1[39];
          float v101_data = ir3[3];
          ir3[3] = (v101_data + (v83_data * v99_data));
          float v104_data = s1[52];
          float v106_data = ir3[4];
          ir3[4] = (v106_data + (v83_data * v104_data));
          float v109_data = s1[65];
          float v111_data = ir3[5];
          ir3[5] = (v111_data + (v83_data * v109_data));
          float v114_data = s1[78];
          float v116_data = ir3[6];
          ir3[6] = (v116_data + (v83_data * v114_data));
          float v119_data = s1[91];
          float v121_data = ir3[7];
          ir3[7] = (v121_data + (v83_data * v119_data));
          float v124_data = s1[104];
          float v126_data = ir3[8];
          ir3[8] = (v126_data + (v83_data * v124_data));
          float v129_data = s1[117];
          float v131_data = ir3[9];
          ir3[9] = (v131_data + (v83_data * v129_data));
          float v134_data = s1[130];
          float v136_data = ir3[10];
          ir3[10] = (v136_data + (v83_data * v134_data));
          float v139_data = s1[143];
          float v141_data = ir3[11];
          ir3[11] = (v141_data + (v83_data * v139_data));
          float v144_data = s1[156];
          float v146_data = ir3[12];
          ir3[12] = (v146_data + (v83_data * v144_data));
          float v148_data = r2[1];
          float v149_data = s1[1];
          float v151_data = ir3[0];
          ir3[0] = (v151_data + (v148_data * v149_data));
          float v154_data = s1[14];
          float v156_data = ir3[1];
          ir3[1] = (v156_data + (v148_data * v154_data));
          float v159_data = s1[27];
          float v161_data = ir3[2];
          ir3[2] = (v161_data + (v148_data * v159_data));
          float v164_data = s1[40];
          float v166_data = ir3[3];
          ir3[3] = (v166_data + (v148_data * v164_data));
          float v169_data = s1[53];
          float v171_data = ir3[4];
          ir3[4] = (v171_data + (v148_data * v169_data));
          float v174_data = s1[66];
          float v176_data = ir3[5];
          ir3[5] = (v176_data + (v148_data * v174_data));
          float v179_data = s1[79];
          float v181_data = ir3[6];
          ir3[6] = (v181_data + (v148_data * v179_data));
          float v184_data = s1[92];
          float v186_data = ir3[7];
          ir3[7] = (v186_data + (v148_data * v184_data));
          float v189_data = s1[105];
          float v191_data = ir3[8];
          ir3[8] = (v191_data + (v148_data * v189_data));
          float v194_data = s1[118];
          float v196_data = ir3[9];
          ir3[9] = (v196_data + (v148_data * v194_data));
          float v199_data = s1[131];
          float v201_data = ir3[10];
          ir3[10] = (v201_data + (v148_data * v199_data));
          float v204_data = s1[144];
          float v206_data = ir3[11];
          ir3[11] = (v206_data + (v148_data * v204_data));
          float v209_data = s1[157];
          float v211_data = ir3[12];
          ir3[12] = (v211_data + (v148_data * v209_data));
          float v213_data = r2[2];
          float v214_data = s1[2];
          float v216_data = ir3[0];
          ir3[0] = (v216_data + (v213_data * v214_data));
          float v219_data = s1[15];
          float v221_data = ir3[1];
          ir3[1] = (v221_data + (v213_data * v219_data));
          float v224_data = s1[28];
          float v226_data = ir3[2];
          ir3[2] = (v226_data + (v213_data * v224_data));
          float v229_data = s1[41];
          float v231_data = ir3[3];
          ir3[3] = (v231_data + (v213_data * v229_data));
          float v234_data = s1[54];
          float v236_data = ir3[4];
          ir3[4] = (v236_data + (v213_data * v234_data));
          float v239_data = s1[67];
          float v241_data = ir3[5];
          ir3[5] = (v241_data + (v213_data * v239_data));
          float v244_data = s1[80];
          float v246_data = ir3[6];
          ir3[6] = (v246_data + (v213_data * v244_data));
          float v249_data = s1[93];
          float v251_data = ir3[7];
          ir3[7] = (v251_data + (v213_data * v249_data));
          float v254_data = s1[106];
          float v256_data = ir3[8];
          ir3[8] = (v256_data + (v213_data * v254_data));
          float v259_data = s1[119];
          float v261_data = ir3[9];
          ir3[9] = (v261_data + (v213_data * v259_data));
          float v264_data = s1[132];
          float v266_data = ir3[10];
          ir3[10] = (v266_data + (v213_data * v264_data));
          float v269_data = s1[145];
          float v271_data = ir3[11];
          ir3[11] = (v271_data + (v213_data * v269_data));
          float v274_data = s1[158];
          float v276_data = ir3[12];
          ir3[12] = (v276_data + (v213_data * v274_data));
          float v278_data = r2[3];
          float v279_data = s1[3];
          float v281_data = ir3[0];
          ir3[0] = (v281_data + (v278_data * v279_data));
          float v284_data = s1[16];
          float v286_data = ir3[1];
          ir3[1] = (v286_data + (v278_data * v284_data));
          float v289_data = s1[29];
          float v291_data = ir3[2];
          ir3[2] = (v291_data + (v278_data * v289_data));
          float v294_data = s1[42];
          float v296_data = ir3[3];
          ir3[3] = (v296_data + (v278_data * v294_data));
          float v299_data = s1[55];
          float v301_data = ir3[4];
          ir3[4] = (v301_data + (v278_data * v299_data));
          float v304_data = s1[68];
          float v306_data = ir3[5];
          ir3[5] = (v306_data + (v278_data * v304_data));
          float v309_data = s1[81];
          float v311_data = ir3[6];
          ir3[6] = (v311_data + (v278_data * v309_data));
          float v314_data = s1[94];
          float v316_data = ir3[7];
          ir3[7] = (v316_data + (v278_data * v314_data));
          float v319_data = s1[107];
          float v321_data = ir3[8];
          ir3[8] = (v321_data + (v278_data * v319_data));
          float v324_data = s1[120];
          float v326_data = ir3[9];
          ir3[9] = (v326_data + (v278_data * v324_data));
          float v329_data = s1[133];
          float v331_data = ir3[10];
          ir3[10] = (v331_data + (v278_data * v329_data));
          float v334_data = s1[146];
          float v336_data = ir3[11];
          ir3[11] = (v336_data + (v278_data * v334_data));
          float v339_data = s1[159];
          float v341_data = ir3[12];
          ir3[12] = (v341_data + (v278_data * v339_data));
          float v343_data = r2[4];
          float v344_data = s1[4];
          float v346_data = ir3[0];
          ir3[0] = (v346_data + (v343_data * v344_data));
          float v349_data = s1[17];
          float v351_data = ir3[1];
          ir3[1] = (v351_data + (v343_data * v349_data));
          float v354_data = s1[30];
          float v356_data = ir3[2];
          ir3[2] = (v356_data + (v343_data * v354_data));
          float v359_data = s1[43];
          float v361_data = ir3[3];
          ir3[3] = (v361_data + (v343_data * v359_data));
          float v364_data = s1[56];
          float v366_data = ir3[4];
          ir3[4] = (v366_data + (v343_data * v364_data));
          float v369_data = s1[69];
          float v371_data = ir3[5];
          ir3[5] = (v371_data + (v343_data * v369_data));
          float v374_data = s1[82];
          float v376_data = ir3[6];
          ir3[6] = (v376_data + (v343_data * v374_data));
          float v379_data = s1[95];
          float v381_data = ir3[7];
          ir3[7] = (v381_data + (v343_data * v379_data));
          float v384_data = s1[108];
          float v386_data = ir3[8];
          ir3[8] = (v386_data + (v343_data * v384_data));
          float v389_data = s1[121];
          float v391_data = ir3[9];
          ir3[9] = (v391_data + (v343_data * v389_data));
          float v394_data = s1[134];
          float v396_data = ir3[10];
          ir3[10] = (v396_data + (v343_data * v394_data));
          float v399_data = s1[147];
          float v401_data = ir3[11];
          ir3[11] = (v401_data + (v343_data * v399_data));
          float v404_data = s1[160];
          float v406_data = ir3[12];
          ir3[12] = (v406_data + (v343_data * v404_data));
          float v408_data = r2[5];
          float v409_data = s1[5];
          float v411_data = ir3[0];
          ir3[0] = (v411_data + (v408_data * v409_data));
          float v414_data = s1[18];
          float v416_data = ir3[1];
          ir3[1] = (v416_data + (v408_data * v414_data));
          float v419_data = s1[31];
          float v421_data = ir3[2];
          ir3[2] = (v421_data + (v408_data * v419_data));
          float v424_data = s1[44];
          float v426_data = ir3[3];
          ir3[3] = (v426_data + (v408_data * v424_data));
          float v429_data = s1[57];
          float v431_data = ir3[4];
          ir3[4] = (v431_data + (v408_data * v429_data));
          float v434_data = s1[70];
          float v436_data = ir3[5];
          ir3[5] = (v436_data + (v408_data * v434_data));
          float v439_data = s1[83];
          float v441_data = ir3[6];
          ir3[6] = (v441_data + (v408_data * v439_data));
          float v444_data = s1[96];
          float v446_data = ir3[7];
          ir3[7] = (v446_data + (v408_data * v444_data));
          float v449_data = s1[109];
          float v451_data = ir3[8];
          ir3[8] = (v451_data + (v408_data * v449_data));
          float v454_data = s1[122];
          float v456_data = ir3[9];
          ir3[9] = (v456_data + (v408_data * v454_data));
          float v459_data = s1[135];
          float v461_data = ir3[10];
          ir3[10] = (v461_data + (v408_data * v459_data));
          float v464_data = s1[148];
          float v466_data = ir3[11];
          ir3[11] = (v466_data + (v408_data * v464_data));
          float v469_data = s1[161];
          float v471_data = ir3[12];
          ir3[12] = (v471_data + (v408_data * v469_data));
          float v473_data = r2[6];
          float v474_data = s1[6];
          float v476_data = ir3[0];
          ir3[0] = (v476_data + (v473_data * v474_data));
          float v479_data = s1[19];
          float v481_data = ir3[1];
          ir3[1] = (v481_data + (v473_data * v479_data));
          float v484_data = s1[32];
          float v486_data = ir3[2];
          ir3[2] = (v486_data + (v473_data * v484_data));
          float v489_data = s1[45];
          float v491_data = ir3[3];
          ir3[3] = (v491_data + (v473_data * v489_data));
          float v494_data = s1[58];
          float v496_data = ir3[4];
          ir3[4] = (v496_data + (v473_data * v494_data));
          float v499_data = s1[71];
          float v501_data = ir3[5];
          ir3[5] = (v501_data + (v473_data * v499_data));
          float v504_data = s1[84];
          float v506_data = ir3[6];
          ir3[6] = (v506_data + (v473_data * v504_data));
          float v509_data = s1[97];
          float v511_data = ir3[7];
          ir3[7] = (v511_data + (v473_data * v509_data));
          float v514_data = s1[110];
          float v516_data = ir3[8];
          ir3[8] = (v516_data + (v473_data * v514_data));
          float v519_data = s1[123];
          float v521_data = ir3[9];
          ir3[9] = (v521_data + (v473_data * v519_data));
          float v524_data = s1[136];
          float v526_data = ir3[10];
          ir3[10] = (v526_data + (v473_data * v524_data));
          float v529_data = s1[149];
          float v531_data = ir3[11];
          ir3[11] = (v531_data + (v473_data * v529_data));
          float v534_data = s1[162];
          float v536_data = ir3[12];
          ir3[12] = (v536_data + (v473_data * v534_data));
          float v538_data = r2[7];
          float v539_data = s1[7];
          float v541_data = ir3[0];
          ir3[0] = (v541_data + (v538_data * v539_data));
          float v544_data = s1[20];
          float v546_data = ir3[1];
          ir3[1] = (v546_data + (v538_data * v544_data));
          float v549_data = s1[33];
          float v551_data = ir3[2];
          ir3[2] = (v551_data + (v538_data * v549_data));
          float v554_data = s1[46];
          float v556_data = ir3[3];
          ir3[3] = (v556_data + (v538_data * v554_data));
          float v559_data = s1[59];
          float v561_data = ir3[4];
          ir3[4] = (v561_data + (v538_data * v559_data));
          float v564_data = s1[72];
          float v566_data = ir3[5];
          ir3[5] = (v566_data + (v538_data * v564_data));
          float v569_data = s1[85];
          float v571_data = ir3[6];
          ir3[6] = (v571_data + (v538_data * v569_data));
          float v574_data = s1[98];
          float v576_data = ir3[7];
          ir3[7] = (v576_data + (v538_data * v574_data));
          float v579_data = s1[111];
          float v581_data = ir3[8];
          ir3[8] = (v581_data + (v538_data * v579_data));
          float v584_data = s1[124];
          float v586_data = ir3[9];
          ir3[9] = (v586_data + (v538_data * v584_data));
          float v589_data = s1[137];
          float v591_data = ir3[10];
          ir3[10] = (v591_data + (v538_data * v589_data));
          float v594_data = s1[150];
          float v596_data = ir3[11];
          ir3[11] = (v596_data + (v538_data * v594_data));
          float v599_data = s1[163];
          float v601_data = ir3[12];
          ir3[12] = (v601_data + (v538_data * v599_data));
          float v603_data = r2[8];
          float v604_data = s1[8];
          float v606_data = ir3[0];
          ir3[0] = (v606_data + (v603_data * v604_data));
          float v609_data = s1[21];
          float v611_data = ir3[1];
          ir3[1] = (v611_data + (v603_data * v609_data));
          float v614_data = s1[34];
          float v616_data = ir3[2];
          ir3[2] = (v616_data + (v603_data * v614_data));
          float v619_data = s1[47];
          float v621_data = ir3[3];
          ir3[3] = (v621_data + (v603_data * v619_data));
          float v624_data = s1[60];
          float v626_data = ir3[4];
          ir3[4] = (v626_data + (v603_data * v624_data));
          float v629_data = s1[73];
          float v631_data = ir3[5];
          ir3[5] = (v631_data + (v603_data * v629_data));
          float v634_data = s1[86];
          float v636_data = ir3[6];
          ir3[6] = (v636_data + (v603_data * v634_data));
          float v639_data = s1[99];
          float v641_data = ir3[7];
          ir3[7] = (v641_data + (v603_data * v639_data));
          float v644_data = s1[112];
          float v646_data = ir3[8];
          ir3[8] = (v646_data + (v603_data * v644_data));
          float v649_data = s1[125];
          float v651_data = ir3[9];
          ir3[9] = (v651_data + (v603_data * v649_data));
          float v654_data = s1[138];
          float v656_data = ir3[10];
          ir3[10] = (v656_data + (v603_data * v654_data));
          float v659_data = s1[151];
          float v661_data = ir3[11];
          ir3[11] = (v661_data + (v603_data * v659_data));
          float v664_data = s1[164];
          float v666_data = ir3[12];
          ir3[12] = (v666_data + (v603_data * v664_data));
          float v668_data = r2[9];
          float v669_data = s1[9];
          float v671_data = ir3[0];
          ir3[0] = (v671_data + (v668_data * v669_data));
          float v674_data = s1[22];
          float v676_data = ir3[1];
          ir3[1] = (v676_data + (v668_data * v674_data));
          float v679_data = s1[35];
          float v681_data = ir3[2];
          ir3[2] = (v681_data + (v668_data * v679_data));
          float v684_data = s1[48];
          float v686_data = ir3[3];
          ir3[3] = (v686_data + (v668_data * v684_data));
          float v689_data = s1[61];
          float v691_data = ir3[4];
          ir3[4] = (v691_data + (v668_data * v689_data));
          float v694_data = s1[74];
          float v696_data = ir3[5];
          ir3[5] = (v696_data + (v668_data * v694_data));
          float v699_data = s1[87];
          float v701_data = ir3[6];
          ir3[6] = (v701_data + (v668_data * v699_data));
          float v704_data = s1[100];
          float v706_data = ir3[7];
          ir3[7] = (v706_data + (v668_data * v704_data));
          float v709_data = s1[113];
          float v711_data = ir3[8];
          ir3[8] = (v711_data + (v668_data * v709_data));
          float v714_data = s1[126];
          float v716_data = ir3[9];
          ir3[9] = (v716_data + (v668_data * v714_data));
          float v719_data = s1[139];
          float v721_data = ir3[10];
          ir3[10] = (v721_data + (v668_data * v719_data));
          float v724_data = s1[152];
          float v726_data = ir3[11];
          ir3[11] = (v726_data + (v668_data * v724_data));
          float v729_data = s1[165];
          float v731_data = ir3[12];
          ir3[12] = (v731_data + (v668_data * v729_data));
          float v733_data = r2[10];
          float v734_data = s1[10];
          float v736_data = ir3[0];
          ir3[0] = (v736_data + (v733_data * v734_data));
          float v739_data = s1[23];
          float v741_data = ir3[1];
          ir3[1] = (v741_data + (v733_data * v739_data));
          float v744_data = s1[36];
          float v746_data = ir3[2];
          ir3[2] = (v746_data + (v733_data * v744_data));
          float v749_data = s1[49];
          float v751_data = ir3[3];
          ir3[3] = (v751_data + (v733_data * v749_data));
          float v754_data = s1[62];
          float v756_data = ir3[4];
          ir3[4] = (v756_data + (v733_data * v754_data));
          float v759_data = s1[75];
          float v761_data = ir3[5];
          ir3[5] = (v761_data + (v733_data * v759_data));
          float v764_data = s1[88];
          float v766_data = ir3[6];
          ir3[6] = (v766_data + (v733_data * v764_data));
          float v769_data = s1[101];
          float v771_data = ir3[7];
          ir3[7] = (v771_data + (v733_data * v769_data));
          float v774_data = s1[114];
          float v776_data = ir3[8];
          ir3[8] = (v776_data + (v733_data * v774_data));
          float v779_data = s1[127];
          float v781_data = ir3[9];
          ir3[9] = (v781_data + (v733_data * v779_data));
          float v784_data = s1[140];
          float v786_data = ir3[10];
          ir3[10] = (v786_data + (v733_data * v784_data));
          float v789_data = s1[153];
          float v791_data = ir3[11];
          ir3[11] = (v791_data + (v733_data * v789_data));
          float v794_data = s1[166];
          float v796_data = ir3[12];
          ir3[12] = (v796_data + (v733_data * v794_data));
          float v798_data = r2[11];
          float v799_data = s1[11];
          float v801_data = ir3[0];
          ir3[0] = (v801_data + (v798_data * v799_data));
          float v804_data = s1[24];
          float v806_data = ir3[1];
          ir3[1] = (v806_data + (v798_data * v804_data));
          float v809_data = s1[37];
          float v811_data = ir3[2];
          ir3[2] = (v811_data + (v798_data * v809_data));
          float v814_data = s1[50];
          float v816_data = ir3[3];
          ir3[3] = (v816_data + (v798_data * v814_data));
          float v819_data = s1[63];
          float v821_data = ir3[4];
          ir3[4] = (v821_data + (v798_data * v819_data));
          float v824_data = s1[76];
          float v826_data = ir3[5];
          ir3[5] = (v826_data + (v798_data * v824_data));
          float v829_data = s1[89];
          float v831_data = ir3[6];
          ir3[6] = (v831_data + (v798_data * v829_data));
          float v834_data = s1[102];
          float v836_data = ir3[7];
          ir3[7] = (v836_data + (v798_data * v834_data));
          float v839_data = s1[115];
          float v841_data = ir3[8];
          ir3[8] = (v841_data + (v798_data * v839_data));
          float v844_data = s1[128];
          float v846_data = ir3[9];
          ir3[9] = (v846_data + (v798_data * v844_data));
          float v849_data = s1[141];
          float v851_data = ir3[10];
          ir3[10] = (v851_data + (v798_data * v849_data));
          float v854_data = s1[154];
          float v856_data = ir3[11];
          ir3[11] = (v856_data + (v798_data * v854_data));
          float v859_data = s1[167];
          float v861_data = ir3[12];
          ir3[12] = (v861_data + (v798_data * v859_data));
          float v863_data = r2[12];
          float v864_data = s1[12];
          float v866_data = ir3[0];
          ir3[0] = (v866_data + (v863_data * v864_data));
          float v869_data = s1[25];
          float v871_data = ir3[1];
          ir3[1] = (v871_data + (v863_data * v869_data));
          float v874_data = s1[38];
          float v876_data = ir3[2];
          ir3[2] = (v876_data + (v863_data * v874_data));
          float v879_data = s1[51];
          float v881_data = ir3[3];
          ir3[3] = (v881_data + (v863_data * v879_data));
          float v884_data = s1[64];
          float v886_data = ir3[4];
          ir3[4] = (v886_data + (v863_data * v884_data));
          float v889_data = s1[77];
          float v891_data = ir3[5];
          ir3[5] = (v891_data + (v863_data * v889_data));
          float v894_data = s1[90];
          float v896_data = ir3[6];
          ir3[6] = (v896_data + (v863_data * v894_data));
          float v899_data = s1[103];
          float v901_data = ir3[7];
          ir3[7] = (v901_data + (v863_data * v899_data));
          float v904_data = s1[116];
          float v906_data = ir3[8];
          ir3[8] = (v906_data + (v863_data * v904_data));
          float v909_data = s1[129];
          float v911_data = ir3[9];
          ir3[9] = (v911_data + (v863_data * v909_data));
          float v914_data = s1[142];
          float v916_data = ir3[10];
          ir3[10] = (v916_data + (v863_data * v914_data));
          float v919_data = s1[155];
          float v921_data = ir3[11];
          ir3[11] = (v921_data + (v863_data * v919_data));
          float v924_data = s1[168];
          float v926_data = ir3[12];
          ir3[12] = (v926_data + (v863_data * v924_data));
          // r3 = ir3
          #pragma unroll
          for (int32_t v928_n0 = 0; v928_n0 < 1; ++v928_n0) {
            #pragma unroll
            for (int32_t v929_n1 = 0; v929_n1 < 13; ++v929_n1) {
              int32_t v930_a = v928_n0 + v929_n1;
              float v931_data = ir3[v930_a];
              r3[v930_a] = v931_data;
            }
          }
          // glb_m3 = store{r>g}(r3);
          #pragma unroll
          for (int32_t v932_i0 = 0; v932_i0 < 1; ++v932_i0) {
            int32_t v937_lead = v28_lead + (v932_i0 * 32);
            #pragma unroll
            for (int32_t v933_i1 = 0; v933_i1 < 13; ++v933_i1) {
              float v935_data = r3[(v932_i0 + v933_i1)];
              glb_m3[(v937_lead + (v933_i1 * 32))] = v935_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

