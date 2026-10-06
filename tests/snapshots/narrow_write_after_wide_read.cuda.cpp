// === base name ===
kernel_490f05cfed86abfc

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_490f05cfed86abfc = {{32, 4, 1}, 32, 32, 1, 4, 3072, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_490f05cfed86abfc(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_490f05cfed86abfc(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_490f05cfed86abfc(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_490f05cfed86abfc, block.x * block.y * block.z, 768 * sizeof(float));
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
void launcher_kernel_490f05cfed86abfc(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_490f05cfed86abfc(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_490f05cfed86abfc, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_490f05cfed86abfc<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_490f05cfed86abfc(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 4 per block = block 32x4x1, 3072 B shared, occupancy grid
    // operands:
    //   m0 32×13(32×13) {0..32}×{0..13} strided
    //   m1 32×12(32×12) {0..32}×{0..12} strided
    //   m2 12×13(12×13) {0..12}×{0..13} strided
    //   m3 32×13(32×13) {0..32}×{0..13} strided
    //   m4 13×13(13×13) {0..13}×{0..13} strided
    // operations:
    //   t0[i,j] = m0[i,j]
    //   t0[i,j] += m1[i,k] × m2[k,j]
    //   m0[i,j]@{0..32}×{4..5} = t0[i,j]@{0..32}×{4..5}
    //   m3[i,j] = m0[i,k] × m4[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":768}],"shared_bytes":3072,"shared_elements":768,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[32,13]],"name":"m0","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[32,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,12],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[12,13]],"name":"m2","ordered":false,"parts":1,"shape":[12,13],"variant":false},{"addressing":"strided","alias":"O","bbox":[[0,0],[32,13]],"name":"m3","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[13,13]],"name":"m4","ordered":false,"parts":1,"shape":[13,13],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,12]},{"addressing":"strided","bbox":[[0,0],[12,13]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[12,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,1]],"is_tmp":false,"name":"m0","offset":[0,4],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,1]],"is_tmp":true,"name":"t0","offset":[0,4],"shape":[32,13]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[0,0],[13,13]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[192 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[192];
      float * __restrict__ s1 = &localShrMem0[0];
      float * __restrict__ s2 = &localShrMem0[0];
      for (size_t v12_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v12_batchId0 < numElements0; v12_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v13_ahead1 = v12_batchId0 + (gridDim.x * blockDim.y);
        size_t v15_batchId1 = (v13_ahead1 < numElements0) ? v13_ahead1 : v12_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v12_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v12_batchId0 * 416 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v12_batchId0 * 384 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v12_batchId0 * 156 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[v12_batchId0 * 416 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[v12_batchId0 * 169 + 0 + m4_extraOffset];
          float r0[13]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v28_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v29_i0 = 0; v29_i0 < 1; ++v29_i0) {
            int32_t v32_lead = v28_lead + (v29_i0 * 32);
            #pragma unroll
            for (int32_t v30_i1 = 0; v30_i1 < 13; ++v30_i1) {
              float v35_data = glb_m0[(v32_lead + (v30_i1 * 32))];
              r0[(v29_i0 + v30_i1)] = v35_data;
            }
          }
          float r2[12]{};
          // r2 = load{g>r}(glb_m1);
          #pragma unroll
          for (int32_t v38_i0 = 0; v38_i0 < 1; ++v38_i0) {
            int32_t v41_lead = v28_lead + (v38_i0 * 32);
            #pragma unroll
            for (int32_t v39_i1 = 0; v39_i1 < 12; ++v39_i1) {
              float v44_data = __ldcg(&glb_m1[(v41_lead + (v39_i1 * 32))]);
              r2[(v38_i0 + v39_i1)] = v44_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[13]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 13)] []
          float v47_data = r0[0];
          float v48_data = r1[0];
          r1[0] = (v48_data + v47_data);
          float v50_data = r0[1];
          float v51_data = r1[1];
          r1[1] = (v51_data + v50_data);
          float v53_data = r0[2];
          float v54_data = r1[2];
          r1[2] = (v54_data + v53_data);
          float v56_data = r0[3];
          float v57_data = r1[3];
          r1[3] = (v57_data + v56_data);
          float v59_data = r0[4];
          float v60_data = r1[4];
          r1[4] = (v60_data + v59_data);
          float v62_data = r0[5];
          float v63_data = r1[5];
          r1[5] = (v63_data + v62_data);
          float v65_data = r0[6];
          float v66_data = r1[6];
          r1[6] = (v66_data + v65_data);
          float v68_data = r0[7];
          float v69_data = r1[7];
          r1[7] = (v69_data + v68_data);
          float v71_data = r0[8];
          float v72_data = r1[8];
          r1[8] = (v72_data + v71_data);
          float v74_data = r0[9];
          float v75_data = r1[9];
          r1[9] = (v75_data + v74_data);
          float v77_data = r0[10];
          float v78_data = r1[10];
          r1[10] = (v78_data + v77_data);
          float v80_data = r0[11];
          float v81_data = r1[11];
          r1[11] = (v81_data + v80_data);
          float v83_data = r0[12];
          float v84_data = r1[12];
          r1[12] = (v84_data + v83_data);
          // s1 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 32], &glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 64], &glb_m2[0 + 0 + 1 * threadIdx.x + 64], 4);
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 96], &glb_m2[0 + 0 + 1 * threadIdx.x + 96], 4);
          if (threadIdx.x < 28) {
            __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 128], &glb_m2[0 + 0 + 1 * threadIdx.x + 128], 4);
          }
          __pipeline_commit();
          // wait(r2 = load{g>r}(glb_m1););
          // wait(s1 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r3[13]{};
          __syncwarp();
          // ir3 = +(r2 * s1)
          // [(0, 32), (0, 13)] [(0, 12)]
          float ir3[13]{};
          float v93_data = r2[0];
          float v94_data = s1[0];
          float v96_data = ir3[0];
          ir3[0] = (v96_data + (v93_data * v94_data));
          float v99_data = s1[12];
          float v101_data = ir3[1];
          ir3[1] = (v101_data + (v93_data * v99_data));
          float v104_data = s1[24];
          float v106_data = ir3[2];
          ir3[2] = (v106_data + (v93_data * v104_data));
          float v109_data = s1[36];
          float v111_data = ir3[3];
          ir3[3] = (v111_data + (v93_data * v109_data));
          float v114_data = s1[48];
          float v116_data = ir3[4];
          ir3[4] = (v116_data + (v93_data * v114_data));
          float v119_data = s1[60];
          float v121_data = ir3[5];
          ir3[5] = (v121_data + (v93_data * v119_data));
          float v124_data = s1[72];
          float v126_data = ir3[6];
          ir3[6] = (v126_data + (v93_data * v124_data));
          float v129_data = s1[84];
          float v131_data = ir3[7];
          ir3[7] = (v131_data + (v93_data * v129_data));
          float v134_data = s1[96];
          float v136_data = ir3[8];
          ir3[8] = (v136_data + (v93_data * v134_data));
          float v139_data = s1[108];
          float v141_data = ir3[9];
          ir3[9] = (v141_data + (v93_data * v139_data));
          float v144_data = s1[120];
          float v146_data = ir3[10];
          ir3[10] = (v146_data + (v93_data * v144_data));
          float v149_data = s1[132];
          float v151_data = ir3[11];
          ir3[11] = (v151_data + (v93_data * v149_data));
          float v154_data = s1[144];
          float v156_data = ir3[12];
          ir3[12] = (v156_data + (v93_data * v154_data));
          float v158_data = r2[1];
          float v159_data = s1[1];
          float v161_data = ir3[0];
          ir3[0] = (v161_data + (v158_data * v159_data));
          float v164_data = s1[13];
          float v166_data = ir3[1];
          ir3[1] = (v166_data + (v158_data * v164_data));
          float v169_data = s1[25];
          float v171_data = ir3[2];
          ir3[2] = (v171_data + (v158_data * v169_data));
          float v174_data = s1[37];
          float v176_data = ir3[3];
          ir3[3] = (v176_data + (v158_data * v174_data));
          float v179_data = s1[49];
          float v181_data = ir3[4];
          ir3[4] = (v181_data + (v158_data * v179_data));
          float v184_data = s1[61];
          float v186_data = ir3[5];
          ir3[5] = (v186_data + (v158_data * v184_data));
          float v189_data = s1[73];
          float v191_data = ir3[6];
          ir3[6] = (v191_data + (v158_data * v189_data));
          float v194_data = s1[85];
          float v196_data = ir3[7];
          ir3[7] = (v196_data + (v158_data * v194_data));
          float v199_data = s1[97];
          float v201_data = ir3[8];
          ir3[8] = (v201_data + (v158_data * v199_data));
          float v204_data = s1[109];
          float v206_data = ir3[9];
          ir3[9] = (v206_data + (v158_data * v204_data));
          float v209_data = s1[121];
          float v211_data = ir3[10];
          ir3[10] = (v211_data + (v158_data * v209_data));
          float v214_data = s1[133];
          float v216_data = ir3[11];
          ir3[11] = (v216_data + (v158_data * v214_data));
          float v219_data = s1[145];
          float v221_data = ir3[12];
          ir3[12] = (v221_data + (v158_data * v219_data));
          float v223_data = r2[2];
          float v224_data = s1[2];
          float v226_data = ir3[0];
          ir3[0] = (v226_data + (v223_data * v224_data));
          float v229_data = s1[14];
          float v231_data = ir3[1];
          ir3[1] = (v231_data + (v223_data * v229_data));
          float v234_data = s1[26];
          float v236_data = ir3[2];
          ir3[2] = (v236_data + (v223_data * v234_data));
          float v239_data = s1[38];
          float v241_data = ir3[3];
          ir3[3] = (v241_data + (v223_data * v239_data));
          float v244_data = s1[50];
          float v246_data = ir3[4];
          ir3[4] = (v246_data + (v223_data * v244_data));
          float v249_data = s1[62];
          float v251_data = ir3[5];
          ir3[5] = (v251_data + (v223_data * v249_data));
          float v254_data = s1[74];
          float v256_data = ir3[6];
          ir3[6] = (v256_data + (v223_data * v254_data));
          float v259_data = s1[86];
          float v261_data = ir3[7];
          ir3[7] = (v261_data + (v223_data * v259_data));
          float v264_data = s1[98];
          float v266_data = ir3[8];
          ir3[8] = (v266_data + (v223_data * v264_data));
          float v269_data = s1[110];
          float v271_data = ir3[9];
          ir3[9] = (v271_data + (v223_data * v269_data));
          float v274_data = s1[122];
          float v276_data = ir3[10];
          ir3[10] = (v276_data + (v223_data * v274_data));
          float v279_data = s1[134];
          float v281_data = ir3[11];
          ir3[11] = (v281_data + (v223_data * v279_data));
          float v284_data = s1[146];
          float v286_data = ir3[12];
          ir3[12] = (v286_data + (v223_data * v284_data));
          float v288_data = r2[3];
          float v289_data = s1[3];
          float v291_data = ir3[0];
          ir3[0] = (v291_data + (v288_data * v289_data));
          float v294_data = s1[15];
          float v296_data = ir3[1];
          ir3[1] = (v296_data + (v288_data * v294_data));
          float v299_data = s1[27];
          float v301_data = ir3[2];
          ir3[2] = (v301_data + (v288_data * v299_data));
          float v304_data = s1[39];
          float v306_data = ir3[3];
          ir3[3] = (v306_data + (v288_data * v304_data));
          float v309_data = s1[51];
          float v311_data = ir3[4];
          ir3[4] = (v311_data + (v288_data * v309_data));
          float v314_data = s1[63];
          float v316_data = ir3[5];
          ir3[5] = (v316_data + (v288_data * v314_data));
          float v319_data = s1[75];
          float v321_data = ir3[6];
          ir3[6] = (v321_data + (v288_data * v319_data));
          float v324_data = s1[87];
          float v326_data = ir3[7];
          ir3[7] = (v326_data + (v288_data * v324_data));
          float v329_data = s1[99];
          float v331_data = ir3[8];
          ir3[8] = (v331_data + (v288_data * v329_data));
          float v334_data = s1[111];
          float v336_data = ir3[9];
          ir3[9] = (v336_data + (v288_data * v334_data));
          float v339_data = s1[123];
          float v341_data = ir3[10];
          ir3[10] = (v341_data + (v288_data * v339_data));
          float v344_data = s1[135];
          float v346_data = ir3[11];
          ir3[11] = (v346_data + (v288_data * v344_data));
          float v349_data = s1[147];
          float v351_data = ir3[12];
          ir3[12] = (v351_data + (v288_data * v349_data));
          float v353_data = r2[4];
          float v354_data = s1[4];
          float v356_data = ir3[0];
          ir3[0] = (v356_data + (v353_data * v354_data));
          float v359_data = s1[16];
          float v361_data = ir3[1];
          ir3[1] = (v361_data + (v353_data * v359_data));
          float v364_data = s1[28];
          float v366_data = ir3[2];
          ir3[2] = (v366_data + (v353_data * v364_data));
          float v369_data = s1[40];
          float v371_data = ir3[3];
          ir3[3] = (v371_data + (v353_data * v369_data));
          float v374_data = s1[52];
          float v376_data = ir3[4];
          ir3[4] = (v376_data + (v353_data * v374_data));
          float v379_data = s1[64];
          float v381_data = ir3[5];
          ir3[5] = (v381_data + (v353_data * v379_data));
          float v384_data = s1[76];
          float v386_data = ir3[6];
          ir3[6] = (v386_data + (v353_data * v384_data));
          float v389_data = s1[88];
          float v391_data = ir3[7];
          ir3[7] = (v391_data + (v353_data * v389_data));
          float v394_data = s1[100];
          float v396_data = ir3[8];
          ir3[8] = (v396_data + (v353_data * v394_data));
          float v399_data = s1[112];
          float v401_data = ir3[9];
          ir3[9] = (v401_data + (v353_data * v399_data));
          float v404_data = s1[124];
          float v406_data = ir3[10];
          ir3[10] = (v406_data + (v353_data * v404_data));
          float v409_data = s1[136];
          float v411_data = ir3[11];
          ir3[11] = (v411_data + (v353_data * v409_data));
          float v414_data = s1[148];
          float v416_data = ir3[12];
          ir3[12] = (v416_data + (v353_data * v414_data));
          float v418_data = r2[5];
          float v419_data = s1[5];
          float v421_data = ir3[0];
          ir3[0] = (v421_data + (v418_data * v419_data));
          float v424_data = s1[17];
          float v426_data = ir3[1];
          ir3[1] = (v426_data + (v418_data * v424_data));
          float v429_data = s1[29];
          float v431_data = ir3[2];
          ir3[2] = (v431_data + (v418_data * v429_data));
          float v434_data = s1[41];
          float v436_data = ir3[3];
          ir3[3] = (v436_data + (v418_data * v434_data));
          float v439_data = s1[53];
          float v441_data = ir3[4];
          ir3[4] = (v441_data + (v418_data * v439_data));
          float v444_data = s1[65];
          float v446_data = ir3[5];
          ir3[5] = (v446_data + (v418_data * v444_data));
          float v449_data = s1[77];
          float v451_data = ir3[6];
          ir3[6] = (v451_data + (v418_data * v449_data));
          float v454_data = s1[89];
          float v456_data = ir3[7];
          ir3[7] = (v456_data + (v418_data * v454_data));
          float v459_data = s1[101];
          float v461_data = ir3[8];
          ir3[8] = (v461_data + (v418_data * v459_data));
          float v464_data = s1[113];
          float v466_data = ir3[9];
          ir3[9] = (v466_data + (v418_data * v464_data));
          float v469_data = s1[125];
          float v471_data = ir3[10];
          ir3[10] = (v471_data + (v418_data * v469_data));
          float v474_data = s1[137];
          float v476_data = ir3[11];
          ir3[11] = (v476_data + (v418_data * v474_data));
          float v479_data = s1[149];
          float v481_data = ir3[12];
          ir3[12] = (v481_data + (v418_data * v479_data));
          float v483_data = r2[6];
          float v484_data = s1[6];
          float v486_data = ir3[0];
          ir3[0] = (v486_data + (v483_data * v484_data));
          float v489_data = s1[18];
          float v491_data = ir3[1];
          ir3[1] = (v491_data + (v483_data * v489_data));
          float v494_data = s1[30];
          float v496_data = ir3[2];
          ir3[2] = (v496_data + (v483_data * v494_data));
          float v499_data = s1[42];
          float v501_data = ir3[3];
          ir3[3] = (v501_data + (v483_data * v499_data));
          float v504_data = s1[54];
          float v506_data = ir3[4];
          ir3[4] = (v506_data + (v483_data * v504_data));
          float v509_data = s1[66];
          float v511_data = ir3[5];
          ir3[5] = (v511_data + (v483_data * v509_data));
          float v514_data = s1[78];
          float v516_data = ir3[6];
          ir3[6] = (v516_data + (v483_data * v514_data));
          float v519_data = s1[90];
          float v521_data = ir3[7];
          ir3[7] = (v521_data + (v483_data * v519_data));
          float v524_data = s1[102];
          float v526_data = ir3[8];
          ir3[8] = (v526_data + (v483_data * v524_data));
          float v529_data = s1[114];
          float v531_data = ir3[9];
          ir3[9] = (v531_data + (v483_data * v529_data));
          float v534_data = s1[126];
          float v536_data = ir3[10];
          ir3[10] = (v536_data + (v483_data * v534_data));
          float v539_data = s1[138];
          float v541_data = ir3[11];
          ir3[11] = (v541_data + (v483_data * v539_data));
          float v544_data = s1[150];
          float v546_data = ir3[12];
          ir3[12] = (v546_data + (v483_data * v544_data));
          float v548_data = r2[7];
          float v549_data = s1[7];
          float v551_data = ir3[0];
          ir3[0] = (v551_data + (v548_data * v549_data));
          float v554_data = s1[19];
          float v556_data = ir3[1];
          ir3[1] = (v556_data + (v548_data * v554_data));
          float v559_data = s1[31];
          float v561_data = ir3[2];
          ir3[2] = (v561_data + (v548_data * v559_data));
          float v564_data = s1[43];
          float v566_data = ir3[3];
          ir3[3] = (v566_data + (v548_data * v564_data));
          float v569_data = s1[55];
          float v571_data = ir3[4];
          ir3[4] = (v571_data + (v548_data * v569_data));
          float v574_data = s1[67];
          float v576_data = ir3[5];
          ir3[5] = (v576_data + (v548_data * v574_data));
          float v579_data = s1[79];
          float v581_data = ir3[6];
          ir3[6] = (v581_data + (v548_data * v579_data));
          float v584_data = s1[91];
          float v586_data = ir3[7];
          ir3[7] = (v586_data + (v548_data * v584_data));
          float v589_data = s1[103];
          float v591_data = ir3[8];
          ir3[8] = (v591_data + (v548_data * v589_data));
          float v594_data = s1[115];
          float v596_data = ir3[9];
          ir3[9] = (v596_data + (v548_data * v594_data));
          float v599_data = s1[127];
          float v601_data = ir3[10];
          ir3[10] = (v601_data + (v548_data * v599_data));
          float v604_data = s1[139];
          float v606_data = ir3[11];
          ir3[11] = (v606_data + (v548_data * v604_data));
          float v609_data = s1[151];
          float v611_data = ir3[12];
          ir3[12] = (v611_data + (v548_data * v609_data));
          float v613_data = r2[8];
          float v614_data = s1[8];
          float v616_data = ir3[0];
          ir3[0] = (v616_data + (v613_data * v614_data));
          float v619_data = s1[20];
          float v621_data = ir3[1];
          ir3[1] = (v621_data + (v613_data * v619_data));
          float v624_data = s1[32];
          float v626_data = ir3[2];
          ir3[2] = (v626_data + (v613_data * v624_data));
          float v629_data = s1[44];
          float v631_data = ir3[3];
          ir3[3] = (v631_data + (v613_data * v629_data));
          float v634_data = s1[56];
          float v636_data = ir3[4];
          ir3[4] = (v636_data + (v613_data * v634_data));
          float v639_data = s1[68];
          float v641_data = ir3[5];
          ir3[5] = (v641_data + (v613_data * v639_data));
          float v644_data = s1[80];
          float v646_data = ir3[6];
          ir3[6] = (v646_data + (v613_data * v644_data));
          float v649_data = s1[92];
          float v651_data = ir3[7];
          ir3[7] = (v651_data + (v613_data * v649_data));
          float v654_data = s1[104];
          float v656_data = ir3[8];
          ir3[8] = (v656_data + (v613_data * v654_data));
          float v659_data = s1[116];
          float v661_data = ir3[9];
          ir3[9] = (v661_data + (v613_data * v659_data));
          float v664_data = s1[128];
          float v666_data = ir3[10];
          ir3[10] = (v666_data + (v613_data * v664_data));
          float v669_data = s1[140];
          float v671_data = ir3[11];
          ir3[11] = (v671_data + (v613_data * v669_data));
          float v674_data = s1[152];
          float v676_data = ir3[12];
          ir3[12] = (v676_data + (v613_data * v674_data));
          float v678_data = r2[9];
          float v679_data = s1[9];
          float v681_data = ir3[0];
          ir3[0] = (v681_data + (v678_data * v679_data));
          float v684_data = s1[21];
          float v686_data = ir3[1];
          ir3[1] = (v686_data + (v678_data * v684_data));
          float v689_data = s1[33];
          float v691_data = ir3[2];
          ir3[2] = (v691_data + (v678_data * v689_data));
          float v694_data = s1[45];
          float v696_data = ir3[3];
          ir3[3] = (v696_data + (v678_data * v694_data));
          float v699_data = s1[57];
          float v701_data = ir3[4];
          ir3[4] = (v701_data + (v678_data * v699_data));
          float v704_data = s1[69];
          float v706_data = ir3[5];
          ir3[5] = (v706_data + (v678_data * v704_data));
          float v709_data = s1[81];
          float v711_data = ir3[6];
          ir3[6] = (v711_data + (v678_data * v709_data));
          float v714_data = s1[93];
          float v716_data = ir3[7];
          ir3[7] = (v716_data + (v678_data * v714_data));
          float v719_data = s1[105];
          float v721_data = ir3[8];
          ir3[8] = (v721_data + (v678_data * v719_data));
          float v724_data = s1[117];
          float v726_data = ir3[9];
          ir3[9] = (v726_data + (v678_data * v724_data));
          float v729_data = s1[129];
          float v731_data = ir3[10];
          ir3[10] = (v731_data + (v678_data * v729_data));
          float v734_data = s1[141];
          float v736_data = ir3[11];
          ir3[11] = (v736_data + (v678_data * v734_data));
          float v739_data = s1[153];
          float v741_data = ir3[12];
          ir3[12] = (v741_data + (v678_data * v739_data));
          float v743_data = r2[10];
          float v744_data = s1[10];
          float v746_data = ir3[0];
          ir3[0] = (v746_data + (v743_data * v744_data));
          float v749_data = s1[22];
          float v751_data = ir3[1];
          ir3[1] = (v751_data + (v743_data * v749_data));
          float v754_data = s1[34];
          float v756_data = ir3[2];
          ir3[2] = (v756_data + (v743_data * v754_data));
          float v759_data = s1[46];
          float v761_data = ir3[3];
          ir3[3] = (v761_data + (v743_data * v759_data));
          float v764_data = s1[58];
          float v766_data = ir3[4];
          ir3[4] = (v766_data + (v743_data * v764_data));
          float v769_data = s1[70];
          float v771_data = ir3[5];
          ir3[5] = (v771_data + (v743_data * v769_data));
          float v774_data = s1[82];
          float v776_data = ir3[6];
          ir3[6] = (v776_data + (v743_data * v774_data));
          float v779_data = s1[94];
          float v781_data = ir3[7];
          ir3[7] = (v781_data + (v743_data * v779_data));
          float v784_data = s1[106];
          float v786_data = ir3[8];
          ir3[8] = (v786_data + (v743_data * v784_data));
          float v789_data = s1[118];
          float v791_data = ir3[9];
          ir3[9] = (v791_data + (v743_data * v789_data));
          float v794_data = s1[130];
          float v796_data = ir3[10];
          ir3[10] = (v796_data + (v743_data * v794_data));
          float v799_data = s1[142];
          float v801_data = ir3[11];
          ir3[11] = (v801_data + (v743_data * v799_data));
          float v804_data = s1[154];
          float v806_data = ir3[12];
          ir3[12] = (v806_data + (v743_data * v804_data));
          float v808_data = r2[11];
          float v809_data = s1[11];
          float v811_data = ir3[0];
          ir3[0] = (v811_data + (v808_data * v809_data));
          float v814_data = s1[23];
          float v816_data = ir3[1];
          ir3[1] = (v816_data + (v808_data * v814_data));
          float v819_data = s1[35];
          float v821_data = ir3[2];
          ir3[2] = (v821_data + (v808_data * v819_data));
          float v824_data = s1[47];
          float v826_data = ir3[3];
          ir3[3] = (v826_data + (v808_data * v824_data));
          float v829_data = s1[59];
          float v831_data = ir3[4];
          ir3[4] = (v831_data + (v808_data * v829_data));
          float v834_data = s1[71];
          float v836_data = ir3[5];
          ir3[5] = (v836_data + (v808_data * v834_data));
          float v839_data = s1[83];
          float v841_data = ir3[6];
          ir3[6] = (v841_data + (v808_data * v839_data));
          float v844_data = s1[95];
          float v846_data = ir3[7];
          ir3[7] = (v846_data + (v808_data * v844_data));
          float v849_data = s1[107];
          float v851_data = ir3[8];
          ir3[8] = (v851_data + (v808_data * v849_data));
          float v854_data = s1[119];
          float v856_data = ir3[9];
          ir3[9] = (v856_data + (v808_data * v854_data));
          float v859_data = s1[131];
          float v861_data = ir3[10];
          ir3[10] = (v861_data + (v808_data * v859_data));
          float v864_data = s1[143];
          float v866_data = ir3[11];
          ir3[11] = (v866_data + (v808_data * v864_data));
          float v869_data = s1[155];
          float v871_data = ir3[12];
          ir3[12] = (v871_data + (v808_data * v869_data));
          // r3 = ir3 + r1
          #pragma unroll
          for (int32_t v873_n0 = 0; v873_n0 < 1; ++v873_n0) {
            #pragma unroll
            for (int32_t v874_n1 = 0; v874_n1 < 13; ++v874_n1) {
              int32_t v875_a = v873_n0 + v874_n1;
              float v876_data = ir3[v875_a];
              float v877_data = r1[v875_a];
              r3[v875_a] = (v877_data + v876_data);
            }
          }
          float r4[1]{};
          // ir4 = +(r3)
          // [(0, 32), (0, 1)] []
          float ir4[1]{};
          float v881_data = r3[4];
          float v882_data = ir4[0];
          ir4[0] = (v882_data + v881_data);
          // r4 = ir4
          #pragma unroll
          for (int32_t v884_n0 = 0; v884_n0 < 1; ++v884_n0) {
            #pragma unroll
            for (int32_t v885_n1 = 0; v885_n1 < 1; ++v885_n1) {
              int32_t v886_a = v884_n0 + v885_n1;
              float v887_data = ir4[v886_a];
              r4[v886_a] = v887_data;
            }
          }
          // glb_m0 = store{r>g}(r4);
          #pragma unroll
          for (int32_t v888_i0 = 0; v888_i0 < 1; ++v888_i0) {
            int32_t v893_lead = v28_lead + (v888_i0 * 32);
            #pragma unroll
            for (int32_t v889_i1 = 0; v889_i1 < 1; ++v889_i1) {
              float v891_data = r4[(v888_i0 + v889_i1)];
              glb_m0[(v893_lead + ((v889_i1 + 4) * 32))] = v891_data;
            }
          }
          float r5[13]{};
          // r5 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v898_i0 = 0; v898_i0 < 1; ++v898_i0) {
            int32_t v901_lead = v28_lead + (v898_i0 * 32);
            #pragma unroll
            for (int32_t v899_i1 = 0; v899_i1 < 13; ++v899_i1) {
              float v904_data = glb_m0[(v901_lead + (v899_i1 * 32))];
              r5[(v898_i0 + v899_i1)] = v904_data;
            }
          }
          __syncwarp();
          // s2 = load{g>s}(glb_m4[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 5; i += 1) {
            __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + i * 32], &glb_m4[0 + 0 + 1 * threadIdx.x + i * 32], 4);
          }
          if (threadIdx.x < 9) {
            __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 160], &glb_m4[0 + 0 + 1 * threadIdx.x + 160], 4);
          }
          __pipeline_commit();
          // wait(r5 = load{g>r}(glb_m0););
          // wait(s2 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          float r6[13]{};
          __syncwarp();
          // ir6 = +(r5 * s2)
          // [(0, 32), (0, 13)] [(0, 13)]
          float ir6[13]{};
          float v910_data = r5[0];
          float v911_data = s2[0];
          float v913_data = ir6[0];
          ir6[0] = (v913_data + (v910_data * v911_data));
          float v916_data = s2[13];
          float v918_data = ir6[1];
          ir6[1] = (v918_data + (v910_data * v916_data));
          float v921_data = s2[26];
          float v923_data = ir6[2];
          ir6[2] = (v923_data + (v910_data * v921_data));
          float v926_data = s2[39];
          float v928_data = ir6[3];
          ir6[3] = (v928_data + (v910_data * v926_data));
          float v931_data = s2[52];
          float v933_data = ir6[4];
          ir6[4] = (v933_data + (v910_data * v931_data));
          float v936_data = s2[65];
          float v938_data = ir6[5];
          ir6[5] = (v938_data + (v910_data * v936_data));
          float v941_data = s2[78];
          float v943_data = ir6[6];
          ir6[6] = (v943_data + (v910_data * v941_data));
          float v946_data = s2[91];
          float v948_data = ir6[7];
          ir6[7] = (v948_data + (v910_data * v946_data));
          float v951_data = s2[104];
          float v953_data = ir6[8];
          ir6[8] = (v953_data + (v910_data * v951_data));
          float v956_data = s2[117];
          float v958_data = ir6[9];
          ir6[9] = (v958_data + (v910_data * v956_data));
          float v961_data = s2[130];
          float v963_data = ir6[10];
          ir6[10] = (v963_data + (v910_data * v961_data));
          float v966_data = s2[143];
          float v968_data = ir6[11];
          ir6[11] = (v968_data + (v910_data * v966_data));
          float v971_data = s2[156];
          float v973_data = ir6[12];
          ir6[12] = (v973_data + (v910_data * v971_data));
          float v975_data = r5[1];
          float v976_data = s2[1];
          float v978_data = ir6[0];
          ir6[0] = (v978_data + (v975_data * v976_data));
          float v981_data = s2[14];
          float v983_data = ir6[1];
          ir6[1] = (v983_data + (v975_data * v981_data));
          float v986_data = s2[27];
          float v988_data = ir6[2];
          ir6[2] = (v988_data + (v975_data * v986_data));
          float v991_data = s2[40];
          float v993_data = ir6[3];
          ir6[3] = (v993_data + (v975_data * v991_data));
          float v996_data = s2[53];
          float v998_data = ir6[4];
          ir6[4] = (v998_data + (v975_data * v996_data));
          float v1001_data = s2[66];
          float v1003_data = ir6[5];
          ir6[5] = (v1003_data + (v975_data * v1001_data));
          float v1006_data = s2[79];
          float v1008_data = ir6[6];
          ir6[6] = (v1008_data + (v975_data * v1006_data));
          float v1011_data = s2[92];
          float v1013_data = ir6[7];
          ir6[7] = (v1013_data + (v975_data * v1011_data));
          float v1016_data = s2[105];
          float v1018_data = ir6[8];
          ir6[8] = (v1018_data + (v975_data * v1016_data));
          float v1021_data = s2[118];
          float v1023_data = ir6[9];
          ir6[9] = (v1023_data + (v975_data * v1021_data));
          float v1026_data = s2[131];
          float v1028_data = ir6[10];
          ir6[10] = (v1028_data + (v975_data * v1026_data));
          float v1031_data = s2[144];
          float v1033_data = ir6[11];
          ir6[11] = (v1033_data + (v975_data * v1031_data));
          float v1036_data = s2[157];
          float v1038_data = ir6[12];
          ir6[12] = (v1038_data + (v975_data * v1036_data));
          float v1040_data = r5[2];
          float v1041_data = s2[2];
          float v1043_data = ir6[0];
          ir6[0] = (v1043_data + (v1040_data * v1041_data));
          float v1046_data = s2[15];
          float v1048_data = ir6[1];
          ir6[1] = (v1048_data + (v1040_data * v1046_data));
          float v1051_data = s2[28];
          float v1053_data = ir6[2];
          ir6[2] = (v1053_data + (v1040_data * v1051_data));
          float v1056_data = s2[41];
          float v1058_data = ir6[3];
          ir6[3] = (v1058_data + (v1040_data * v1056_data));
          float v1061_data = s2[54];
          float v1063_data = ir6[4];
          ir6[4] = (v1063_data + (v1040_data * v1061_data));
          float v1066_data = s2[67];
          float v1068_data = ir6[5];
          ir6[5] = (v1068_data + (v1040_data * v1066_data));
          float v1071_data = s2[80];
          float v1073_data = ir6[6];
          ir6[6] = (v1073_data + (v1040_data * v1071_data));
          float v1076_data = s2[93];
          float v1078_data = ir6[7];
          ir6[7] = (v1078_data + (v1040_data * v1076_data));
          float v1081_data = s2[106];
          float v1083_data = ir6[8];
          ir6[8] = (v1083_data + (v1040_data * v1081_data));
          float v1086_data = s2[119];
          float v1088_data = ir6[9];
          ir6[9] = (v1088_data + (v1040_data * v1086_data));
          float v1091_data = s2[132];
          float v1093_data = ir6[10];
          ir6[10] = (v1093_data + (v1040_data * v1091_data));
          float v1096_data = s2[145];
          float v1098_data = ir6[11];
          ir6[11] = (v1098_data + (v1040_data * v1096_data));
          float v1101_data = s2[158];
          float v1103_data = ir6[12];
          ir6[12] = (v1103_data + (v1040_data * v1101_data));
          float v1105_data = r5[3];
          float v1106_data = s2[3];
          float v1108_data = ir6[0];
          ir6[0] = (v1108_data + (v1105_data * v1106_data));
          float v1111_data = s2[16];
          float v1113_data = ir6[1];
          ir6[1] = (v1113_data + (v1105_data * v1111_data));
          float v1116_data = s2[29];
          float v1118_data = ir6[2];
          ir6[2] = (v1118_data + (v1105_data * v1116_data));
          float v1121_data = s2[42];
          float v1123_data = ir6[3];
          ir6[3] = (v1123_data + (v1105_data * v1121_data));
          float v1126_data = s2[55];
          float v1128_data = ir6[4];
          ir6[4] = (v1128_data + (v1105_data * v1126_data));
          float v1131_data = s2[68];
          float v1133_data = ir6[5];
          ir6[5] = (v1133_data + (v1105_data * v1131_data));
          float v1136_data = s2[81];
          float v1138_data = ir6[6];
          ir6[6] = (v1138_data + (v1105_data * v1136_data));
          float v1141_data = s2[94];
          float v1143_data = ir6[7];
          ir6[7] = (v1143_data + (v1105_data * v1141_data));
          float v1146_data = s2[107];
          float v1148_data = ir6[8];
          ir6[8] = (v1148_data + (v1105_data * v1146_data));
          float v1151_data = s2[120];
          float v1153_data = ir6[9];
          ir6[9] = (v1153_data + (v1105_data * v1151_data));
          float v1156_data = s2[133];
          float v1158_data = ir6[10];
          ir6[10] = (v1158_data + (v1105_data * v1156_data));
          float v1161_data = s2[146];
          float v1163_data = ir6[11];
          ir6[11] = (v1163_data + (v1105_data * v1161_data));
          float v1166_data = s2[159];
          float v1168_data = ir6[12];
          ir6[12] = (v1168_data + (v1105_data * v1166_data));
          float v1170_data = r5[4];
          float v1171_data = s2[4];
          float v1173_data = ir6[0];
          ir6[0] = (v1173_data + (v1170_data * v1171_data));
          float v1176_data = s2[17];
          float v1178_data = ir6[1];
          ir6[1] = (v1178_data + (v1170_data * v1176_data));
          float v1181_data = s2[30];
          float v1183_data = ir6[2];
          ir6[2] = (v1183_data + (v1170_data * v1181_data));
          float v1186_data = s2[43];
          float v1188_data = ir6[3];
          ir6[3] = (v1188_data + (v1170_data * v1186_data));
          float v1191_data = s2[56];
          float v1193_data = ir6[4];
          ir6[4] = (v1193_data + (v1170_data * v1191_data));
          float v1196_data = s2[69];
          float v1198_data = ir6[5];
          ir6[5] = (v1198_data + (v1170_data * v1196_data));
          float v1201_data = s2[82];
          float v1203_data = ir6[6];
          ir6[6] = (v1203_data + (v1170_data * v1201_data));
          float v1206_data = s2[95];
          float v1208_data = ir6[7];
          ir6[7] = (v1208_data + (v1170_data * v1206_data));
          float v1211_data = s2[108];
          float v1213_data = ir6[8];
          ir6[8] = (v1213_data + (v1170_data * v1211_data));
          float v1216_data = s2[121];
          float v1218_data = ir6[9];
          ir6[9] = (v1218_data + (v1170_data * v1216_data));
          float v1221_data = s2[134];
          float v1223_data = ir6[10];
          ir6[10] = (v1223_data + (v1170_data * v1221_data));
          float v1226_data = s2[147];
          float v1228_data = ir6[11];
          ir6[11] = (v1228_data + (v1170_data * v1226_data));
          float v1231_data = s2[160];
          float v1233_data = ir6[12];
          ir6[12] = (v1233_data + (v1170_data * v1231_data));
          float v1235_data = r5[5];
          float v1236_data = s2[5];
          float v1238_data = ir6[0];
          ir6[0] = (v1238_data + (v1235_data * v1236_data));
          float v1241_data = s2[18];
          float v1243_data = ir6[1];
          ir6[1] = (v1243_data + (v1235_data * v1241_data));
          float v1246_data = s2[31];
          float v1248_data = ir6[2];
          ir6[2] = (v1248_data + (v1235_data * v1246_data));
          float v1251_data = s2[44];
          float v1253_data = ir6[3];
          ir6[3] = (v1253_data + (v1235_data * v1251_data));
          float v1256_data = s2[57];
          float v1258_data = ir6[4];
          ir6[4] = (v1258_data + (v1235_data * v1256_data));
          float v1261_data = s2[70];
          float v1263_data = ir6[5];
          ir6[5] = (v1263_data + (v1235_data * v1261_data));
          float v1266_data = s2[83];
          float v1268_data = ir6[6];
          ir6[6] = (v1268_data + (v1235_data * v1266_data));
          float v1271_data = s2[96];
          float v1273_data = ir6[7];
          ir6[7] = (v1273_data + (v1235_data * v1271_data));
          float v1276_data = s2[109];
          float v1278_data = ir6[8];
          ir6[8] = (v1278_data + (v1235_data * v1276_data));
          float v1281_data = s2[122];
          float v1283_data = ir6[9];
          ir6[9] = (v1283_data + (v1235_data * v1281_data));
          float v1286_data = s2[135];
          float v1288_data = ir6[10];
          ir6[10] = (v1288_data + (v1235_data * v1286_data));
          float v1291_data = s2[148];
          float v1293_data = ir6[11];
          ir6[11] = (v1293_data + (v1235_data * v1291_data));
          float v1296_data = s2[161];
          float v1298_data = ir6[12];
          ir6[12] = (v1298_data + (v1235_data * v1296_data));
          float v1300_data = r5[6];
          float v1301_data = s2[6];
          float v1303_data = ir6[0];
          ir6[0] = (v1303_data + (v1300_data * v1301_data));
          float v1306_data = s2[19];
          float v1308_data = ir6[1];
          ir6[1] = (v1308_data + (v1300_data * v1306_data));
          float v1311_data = s2[32];
          float v1313_data = ir6[2];
          ir6[2] = (v1313_data + (v1300_data * v1311_data));
          float v1316_data = s2[45];
          float v1318_data = ir6[3];
          ir6[3] = (v1318_data + (v1300_data * v1316_data));
          float v1321_data = s2[58];
          float v1323_data = ir6[4];
          ir6[4] = (v1323_data + (v1300_data * v1321_data));
          float v1326_data = s2[71];
          float v1328_data = ir6[5];
          ir6[5] = (v1328_data + (v1300_data * v1326_data));
          float v1331_data = s2[84];
          float v1333_data = ir6[6];
          ir6[6] = (v1333_data + (v1300_data * v1331_data));
          float v1336_data = s2[97];
          float v1338_data = ir6[7];
          ir6[7] = (v1338_data + (v1300_data * v1336_data));
          float v1341_data = s2[110];
          float v1343_data = ir6[8];
          ir6[8] = (v1343_data + (v1300_data * v1341_data));
          float v1346_data = s2[123];
          float v1348_data = ir6[9];
          ir6[9] = (v1348_data + (v1300_data * v1346_data));
          float v1351_data = s2[136];
          float v1353_data = ir6[10];
          ir6[10] = (v1353_data + (v1300_data * v1351_data));
          float v1356_data = s2[149];
          float v1358_data = ir6[11];
          ir6[11] = (v1358_data + (v1300_data * v1356_data));
          float v1361_data = s2[162];
          float v1363_data = ir6[12];
          ir6[12] = (v1363_data + (v1300_data * v1361_data));
          float v1365_data = r5[7];
          float v1366_data = s2[7];
          float v1368_data = ir6[0];
          ir6[0] = (v1368_data + (v1365_data * v1366_data));
          float v1371_data = s2[20];
          float v1373_data = ir6[1];
          ir6[1] = (v1373_data + (v1365_data * v1371_data));
          float v1376_data = s2[33];
          float v1378_data = ir6[2];
          ir6[2] = (v1378_data + (v1365_data * v1376_data));
          float v1381_data = s2[46];
          float v1383_data = ir6[3];
          ir6[3] = (v1383_data + (v1365_data * v1381_data));
          float v1386_data = s2[59];
          float v1388_data = ir6[4];
          ir6[4] = (v1388_data + (v1365_data * v1386_data));
          float v1391_data = s2[72];
          float v1393_data = ir6[5];
          ir6[5] = (v1393_data + (v1365_data * v1391_data));
          float v1396_data = s2[85];
          float v1398_data = ir6[6];
          ir6[6] = (v1398_data + (v1365_data * v1396_data));
          float v1401_data = s2[98];
          float v1403_data = ir6[7];
          ir6[7] = (v1403_data + (v1365_data * v1401_data));
          float v1406_data = s2[111];
          float v1408_data = ir6[8];
          ir6[8] = (v1408_data + (v1365_data * v1406_data));
          float v1411_data = s2[124];
          float v1413_data = ir6[9];
          ir6[9] = (v1413_data + (v1365_data * v1411_data));
          float v1416_data = s2[137];
          float v1418_data = ir6[10];
          ir6[10] = (v1418_data + (v1365_data * v1416_data));
          float v1421_data = s2[150];
          float v1423_data = ir6[11];
          ir6[11] = (v1423_data + (v1365_data * v1421_data));
          float v1426_data = s2[163];
          float v1428_data = ir6[12];
          ir6[12] = (v1428_data + (v1365_data * v1426_data));
          float v1430_data = r5[8];
          float v1431_data = s2[8];
          float v1433_data = ir6[0];
          ir6[0] = (v1433_data + (v1430_data * v1431_data));
          float v1436_data = s2[21];
          float v1438_data = ir6[1];
          ir6[1] = (v1438_data + (v1430_data * v1436_data));
          float v1441_data = s2[34];
          float v1443_data = ir6[2];
          ir6[2] = (v1443_data + (v1430_data * v1441_data));
          float v1446_data = s2[47];
          float v1448_data = ir6[3];
          ir6[3] = (v1448_data + (v1430_data * v1446_data));
          float v1451_data = s2[60];
          float v1453_data = ir6[4];
          ir6[4] = (v1453_data + (v1430_data * v1451_data));
          float v1456_data = s2[73];
          float v1458_data = ir6[5];
          ir6[5] = (v1458_data + (v1430_data * v1456_data));
          float v1461_data = s2[86];
          float v1463_data = ir6[6];
          ir6[6] = (v1463_data + (v1430_data * v1461_data));
          float v1466_data = s2[99];
          float v1468_data = ir6[7];
          ir6[7] = (v1468_data + (v1430_data * v1466_data));
          float v1471_data = s2[112];
          float v1473_data = ir6[8];
          ir6[8] = (v1473_data + (v1430_data * v1471_data));
          float v1476_data = s2[125];
          float v1478_data = ir6[9];
          ir6[9] = (v1478_data + (v1430_data * v1476_data));
          float v1481_data = s2[138];
          float v1483_data = ir6[10];
          ir6[10] = (v1483_data + (v1430_data * v1481_data));
          float v1486_data = s2[151];
          float v1488_data = ir6[11];
          ir6[11] = (v1488_data + (v1430_data * v1486_data));
          float v1491_data = s2[164];
          float v1493_data = ir6[12];
          ir6[12] = (v1493_data + (v1430_data * v1491_data));
          float v1495_data = r5[9];
          float v1496_data = s2[9];
          float v1498_data = ir6[0];
          ir6[0] = (v1498_data + (v1495_data * v1496_data));
          float v1501_data = s2[22];
          float v1503_data = ir6[1];
          ir6[1] = (v1503_data + (v1495_data * v1501_data));
          float v1506_data = s2[35];
          float v1508_data = ir6[2];
          ir6[2] = (v1508_data + (v1495_data * v1506_data));
          float v1511_data = s2[48];
          float v1513_data = ir6[3];
          ir6[3] = (v1513_data + (v1495_data * v1511_data));
          float v1516_data = s2[61];
          float v1518_data = ir6[4];
          ir6[4] = (v1518_data + (v1495_data * v1516_data));
          float v1521_data = s2[74];
          float v1523_data = ir6[5];
          ir6[5] = (v1523_data + (v1495_data * v1521_data));
          float v1526_data = s2[87];
          float v1528_data = ir6[6];
          ir6[6] = (v1528_data + (v1495_data * v1526_data));
          float v1531_data = s2[100];
          float v1533_data = ir6[7];
          ir6[7] = (v1533_data + (v1495_data * v1531_data));
          float v1536_data = s2[113];
          float v1538_data = ir6[8];
          ir6[8] = (v1538_data + (v1495_data * v1536_data));
          float v1541_data = s2[126];
          float v1543_data = ir6[9];
          ir6[9] = (v1543_data + (v1495_data * v1541_data));
          float v1546_data = s2[139];
          float v1548_data = ir6[10];
          ir6[10] = (v1548_data + (v1495_data * v1546_data));
          float v1551_data = s2[152];
          float v1553_data = ir6[11];
          ir6[11] = (v1553_data + (v1495_data * v1551_data));
          float v1556_data = s2[165];
          float v1558_data = ir6[12];
          ir6[12] = (v1558_data + (v1495_data * v1556_data));
          float v1560_data = r5[10];
          float v1561_data = s2[10];
          float v1563_data = ir6[0];
          ir6[0] = (v1563_data + (v1560_data * v1561_data));
          float v1566_data = s2[23];
          float v1568_data = ir6[1];
          ir6[1] = (v1568_data + (v1560_data * v1566_data));
          float v1571_data = s2[36];
          float v1573_data = ir6[2];
          ir6[2] = (v1573_data + (v1560_data * v1571_data));
          float v1576_data = s2[49];
          float v1578_data = ir6[3];
          ir6[3] = (v1578_data + (v1560_data * v1576_data));
          float v1581_data = s2[62];
          float v1583_data = ir6[4];
          ir6[4] = (v1583_data + (v1560_data * v1581_data));
          float v1586_data = s2[75];
          float v1588_data = ir6[5];
          ir6[5] = (v1588_data + (v1560_data * v1586_data));
          float v1591_data = s2[88];
          float v1593_data = ir6[6];
          ir6[6] = (v1593_data + (v1560_data * v1591_data));
          float v1596_data = s2[101];
          float v1598_data = ir6[7];
          ir6[7] = (v1598_data + (v1560_data * v1596_data));
          float v1601_data = s2[114];
          float v1603_data = ir6[8];
          ir6[8] = (v1603_data + (v1560_data * v1601_data));
          float v1606_data = s2[127];
          float v1608_data = ir6[9];
          ir6[9] = (v1608_data + (v1560_data * v1606_data));
          float v1611_data = s2[140];
          float v1613_data = ir6[10];
          ir6[10] = (v1613_data + (v1560_data * v1611_data));
          float v1616_data = s2[153];
          float v1618_data = ir6[11];
          ir6[11] = (v1618_data + (v1560_data * v1616_data));
          float v1621_data = s2[166];
          float v1623_data = ir6[12];
          ir6[12] = (v1623_data + (v1560_data * v1621_data));
          float v1625_data = r5[11];
          float v1626_data = s2[11];
          float v1628_data = ir6[0];
          ir6[0] = (v1628_data + (v1625_data * v1626_data));
          float v1631_data = s2[24];
          float v1633_data = ir6[1];
          ir6[1] = (v1633_data + (v1625_data * v1631_data));
          float v1636_data = s2[37];
          float v1638_data = ir6[2];
          ir6[2] = (v1638_data + (v1625_data * v1636_data));
          float v1641_data = s2[50];
          float v1643_data = ir6[3];
          ir6[3] = (v1643_data + (v1625_data * v1641_data));
          float v1646_data = s2[63];
          float v1648_data = ir6[4];
          ir6[4] = (v1648_data + (v1625_data * v1646_data));
          float v1651_data = s2[76];
          float v1653_data = ir6[5];
          ir6[5] = (v1653_data + (v1625_data * v1651_data));
          float v1656_data = s2[89];
          float v1658_data = ir6[6];
          ir6[6] = (v1658_data + (v1625_data * v1656_data));
          float v1661_data = s2[102];
          float v1663_data = ir6[7];
          ir6[7] = (v1663_data + (v1625_data * v1661_data));
          float v1666_data = s2[115];
          float v1668_data = ir6[8];
          ir6[8] = (v1668_data + (v1625_data * v1666_data));
          float v1671_data = s2[128];
          float v1673_data = ir6[9];
          ir6[9] = (v1673_data + (v1625_data * v1671_data));
          float v1676_data = s2[141];
          float v1678_data = ir6[10];
          ir6[10] = (v1678_data + (v1625_data * v1676_data));
          float v1681_data = s2[154];
          float v1683_data = ir6[11];
          ir6[11] = (v1683_data + (v1625_data * v1681_data));
          float v1686_data = s2[167];
          float v1688_data = ir6[12];
          ir6[12] = (v1688_data + (v1625_data * v1686_data));
          float v1690_data = r5[12];
          float v1691_data = s2[12];
          float v1693_data = ir6[0];
          ir6[0] = (v1693_data + (v1690_data * v1691_data));
          float v1696_data = s2[25];
          float v1698_data = ir6[1];
          ir6[1] = (v1698_data + (v1690_data * v1696_data));
          float v1701_data = s2[38];
          float v1703_data = ir6[2];
          ir6[2] = (v1703_data + (v1690_data * v1701_data));
          float v1706_data = s2[51];
          float v1708_data = ir6[3];
          ir6[3] = (v1708_data + (v1690_data * v1706_data));
          float v1711_data = s2[64];
          float v1713_data = ir6[4];
          ir6[4] = (v1713_data + (v1690_data * v1711_data));
          float v1716_data = s2[77];
          float v1718_data = ir6[5];
          ir6[5] = (v1718_data + (v1690_data * v1716_data));
          float v1721_data = s2[90];
          float v1723_data = ir6[6];
          ir6[6] = (v1723_data + (v1690_data * v1721_data));
          float v1726_data = s2[103];
          float v1728_data = ir6[7];
          ir6[7] = (v1728_data + (v1690_data * v1726_data));
          float v1731_data = s2[116];
          float v1733_data = ir6[8];
          ir6[8] = (v1733_data + (v1690_data * v1731_data));
          float v1736_data = s2[129];
          float v1738_data = ir6[9];
          ir6[9] = (v1738_data + (v1690_data * v1736_data));
          float v1741_data = s2[142];
          float v1743_data = ir6[10];
          ir6[10] = (v1743_data + (v1690_data * v1741_data));
          float v1746_data = s2[155];
          float v1748_data = ir6[11];
          ir6[11] = (v1748_data + (v1690_data * v1746_data));
          float v1751_data = s2[168];
          float v1753_data = ir6[12];
          ir6[12] = (v1753_data + (v1690_data * v1751_data));
          // r6 = ir6
          #pragma unroll
          for (int32_t v1755_n0 = 0; v1755_n0 < 1; ++v1755_n0) {
            #pragma unroll
            for (int32_t v1756_n1 = 0; v1756_n1 < 13; ++v1756_n1) {
              int32_t v1757_a = v1755_n0 + v1756_n1;
              float v1758_data = ir6[v1757_a];
              r6[v1757_a] = v1758_data;
            }
          }
          // glb_m3 = store{r>g}(r6);
          #pragma unroll
          for (int32_t v1759_i0 = 0; v1759_i0 < 1; ++v1759_i0) {
            int32_t v1764_lead = v28_lead + (v1759_i0 * 32);
            #pragma unroll
            for (int32_t v1760_i1 = 0; v1760_i1 < 13; ++v1760_i1) {
              float v1762_data = r6[(v1759_i0 + v1760_i1)];
              glb_m3[(v1764_lead + (v1760_i1 * 32))] = v1762_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

