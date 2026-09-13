// === base name ===
kernel_d9bb8c0e74ba26d4

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_d9bb8c0e74ba26d4 = {{16, 8, 1}, 16, 12, 1, 8, 5632, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_d9bb8c0e74ba26d4(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_d9bb8c0e74ba26d4(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_d9bb8c0e74ba26d4(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_d9bb8c0e74ba26d4, block.x * block.y * block.z, 1408 * sizeof(float));
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
void launcher_kernel_d9bb8c0e74ba26d4(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_d9bb8c0e74ba26d4(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_d9bb8c0e74ba26d4, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_d9bb8c0e74ba26d4<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_d9bb8c0e74ba26d4(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
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
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1408}],"shared_bytes":5632,"shared_elements":1408,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B","bbox":[[0,0],[12,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,6]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]},{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[176 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[160];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v6_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v6_batchId0 < numElements0; v6_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v7_ahead1 = v6_batchId0 + (gridDim.x * blockDim.y);
        size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 144 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v6_batchId0 * 144 + 0 + m1_extraOffset];
          float *const __restrict__ glb_m2 = &m2[v6_batchId0 * 144 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[v6_batchId0 * 144 + 0 + m3_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v21_lead = threadIdx.x % 16;
          bool v22_g = v21_lead < 12;
          if (v22_g) {
            #pragma unroll
            for (int32_t v23_i1 = 0; v23_i1 < 12; ++v23_i1) {
              float v28_data = __ldcg(&glb_m0[(v21_lead + (v23_i1 * 12))]);
              r0[v23_i1] = v28_data;
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
          if (v22_g) {
            #pragma unroll
            for (int32_t v32_i1 = 0; v32_i1 < 12; ++v32_i1) {
              float v37_data = __ldcg(&glb_m3[(v21_lead + (v32_i1 * 12))]);
              r2[v32_i1] = v37_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[6]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // r1 = +(r0 * s0) + None
          // [(0, 12), (0, 6)] [(0, 12)]
          float v40_data = r0[0];
          float v41_data = s0[0];
          float v43_data = r1[0];
          r1[0] = (v43_data + (v40_data * v41_data));
          float v46_data = s0[12];
          float v48_data = r1[1];
          r1[1] = (v48_data + (v40_data * v46_data));
          float v51_data = s0[24];
          float v53_data = r1[2];
          r1[2] = (v53_data + (v40_data * v51_data));
          float v56_data = s0[36];
          float v58_data = r1[3];
          r1[3] = (v58_data + (v40_data * v56_data));
          float v61_data = s0[48];
          float v63_data = r1[4];
          r1[4] = (v63_data + (v40_data * v61_data));
          float v66_data = s0[60];
          float v68_data = r1[5];
          r1[5] = (v68_data + (v40_data * v66_data));
          float v70_data = r0[1];
          float v71_data = s0[1];
          float v73_data = r1[0];
          r1[0] = (v73_data + (v70_data * v71_data));
          float v76_data = s0[13];
          float v78_data = r1[1];
          r1[1] = (v78_data + (v70_data * v76_data));
          float v81_data = s0[25];
          float v83_data = r1[2];
          r1[2] = (v83_data + (v70_data * v81_data));
          float v86_data = s0[37];
          float v88_data = r1[3];
          r1[3] = (v88_data + (v70_data * v86_data));
          float v91_data = s0[49];
          float v93_data = r1[4];
          r1[4] = (v93_data + (v70_data * v91_data));
          float v96_data = s0[61];
          float v98_data = r1[5];
          r1[5] = (v98_data + (v70_data * v96_data));
          float v100_data = r0[2];
          float v101_data = s0[2];
          float v103_data = r1[0];
          r1[0] = (v103_data + (v100_data * v101_data));
          float v106_data = s0[14];
          float v108_data = r1[1];
          r1[1] = (v108_data + (v100_data * v106_data));
          float v111_data = s0[26];
          float v113_data = r1[2];
          r1[2] = (v113_data + (v100_data * v111_data));
          float v116_data = s0[38];
          float v118_data = r1[3];
          r1[3] = (v118_data + (v100_data * v116_data));
          float v121_data = s0[50];
          float v123_data = r1[4];
          r1[4] = (v123_data + (v100_data * v121_data));
          float v126_data = s0[62];
          float v128_data = r1[5];
          r1[5] = (v128_data + (v100_data * v126_data));
          float v130_data = r0[3];
          float v131_data = s0[3];
          float v133_data = r1[0];
          r1[0] = (v133_data + (v130_data * v131_data));
          float v136_data = s0[15];
          float v138_data = r1[1];
          r1[1] = (v138_data + (v130_data * v136_data));
          float v141_data = s0[27];
          float v143_data = r1[2];
          r1[2] = (v143_data + (v130_data * v141_data));
          float v146_data = s0[39];
          float v148_data = r1[3];
          r1[3] = (v148_data + (v130_data * v146_data));
          float v151_data = s0[51];
          float v153_data = r1[4];
          r1[4] = (v153_data + (v130_data * v151_data));
          float v156_data = s0[63];
          float v158_data = r1[5];
          r1[5] = (v158_data + (v130_data * v156_data));
          float v160_data = r0[4];
          float v161_data = s0[4];
          float v163_data = r1[0];
          r1[0] = (v163_data + (v160_data * v161_data));
          float v166_data = s0[16];
          float v168_data = r1[1];
          r1[1] = (v168_data + (v160_data * v166_data));
          float v171_data = s0[28];
          float v173_data = r1[2];
          r1[2] = (v173_data + (v160_data * v171_data));
          float v176_data = s0[40];
          float v178_data = r1[3];
          r1[3] = (v178_data + (v160_data * v176_data));
          float v181_data = s0[52];
          float v183_data = r1[4];
          r1[4] = (v183_data + (v160_data * v181_data));
          float v186_data = s0[64];
          float v188_data = r1[5];
          r1[5] = (v188_data + (v160_data * v186_data));
          float v190_data = r0[5];
          float v191_data = s0[5];
          float v193_data = r1[0];
          r1[0] = (v193_data + (v190_data * v191_data));
          float v196_data = s0[17];
          float v198_data = r1[1];
          r1[1] = (v198_data + (v190_data * v196_data));
          float v201_data = s0[29];
          float v203_data = r1[2];
          r1[2] = (v203_data + (v190_data * v201_data));
          float v206_data = s0[41];
          float v208_data = r1[3];
          r1[3] = (v208_data + (v190_data * v206_data));
          float v211_data = s0[53];
          float v213_data = r1[4];
          r1[4] = (v213_data + (v190_data * v211_data));
          float v216_data = s0[65];
          float v218_data = r1[5];
          r1[5] = (v218_data + (v190_data * v216_data));
          float v220_data = r0[6];
          float v221_data = s0[6];
          float v223_data = r1[0];
          r1[0] = (v223_data + (v220_data * v221_data));
          float v226_data = s0[18];
          float v228_data = r1[1];
          r1[1] = (v228_data + (v220_data * v226_data));
          float v231_data = s0[30];
          float v233_data = r1[2];
          r1[2] = (v233_data + (v220_data * v231_data));
          float v236_data = s0[42];
          float v238_data = r1[3];
          r1[3] = (v238_data + (v220_data * v236_data));
          float v241_data = s0[54];
          float v243_data = r1[4];
          r1[4] = (v243_data + (v220_data * v241_data));
          float v246_data = s0[66];
          float v248_data = r1[5];
          r1[5] = (v248_data + (v220_data * v246_data));
          float v250_data = r0[7];
          float v251_data = s0[7];
          float v253_data = r1[0];
          r1[0] = (v253_data + (v250_data * v251_data));
          float v256_data = s0[19];
          float v258_data = r1[1];
          r1[1] = (v258_data + (v250_data * v256_data));
          float v261_data = s0[31];
          float v263_data = r1[2];
          r1[2] = (v263_data + (v250_data * v261_data));
          float v266_data = s0[43];
          float v268_data = r1[3];
          r1[3] = (v268_data + (v250_data * v266_data));
          float v271_data = s0[55];
          float v273_data = r1[4];
          r1[4] = (v273_data + (v250_data * v271_data));
          float v276_data = s0[67];
          float v278_data = r1[5];
          r1[5] = (v278_data + (v250_data * v276_data));
          float v280_data = r0[8];
          float v281_data = s0[8];
          float v283_data = r1[0];
          r1[0] = (v283_data + (v280_data * v281_data));
          float v286_data = s0[20];
          float v288_data = r1[1];
          r1[1] = (v288_data + (v280_data * v286_data));
          float v291_data = s0[32];
          float v293_data = r1[2];
          r1[2] = (v293_data + (v280_data * v291_data));
          float v296_data = s0[44];
          float v298_data = r1[3];
          r1[3] = (v298_data + (v280_data * v296_data));
          float v301_data = s0[56];
          float v303_data = r1[4];
          r1[4] = (v303_data + (v280_data * v301_data));
          float v306_data = s0[68];
          float v308_data = r1[5];
          r1[5] = (v308_data + (v280_data * v306_data));
          float v310_data = r0[9];
          float v311_data = s0[9];
          float v313_data = r1[0];
          r1[0] = (v313_data + (v310_data * v311_data));
          float v316_data = s0[21];
          float v318_data = r1[1];
          r1[1] = (v318_data + (v310_data * v316_data));
          float v321_data = s0[33];
          float v323_data = r1[2];
          r1[2] = (v323_data + (v310_data * v321_data));
          float v326_data = s0[45];
          float v328_data = r1[3];
          r1[3] = (v328_data + (v310_data * v326_data));
          float v331_data = s0[57];
          float v333_data = r1[4];
          r1[4] = (v333_data + (v310_data * v331_data));
          float v336_data = s0[69];
          float v338_data = r1[5];
          r1[5] = (v338_data + (v310_data * v336_data));
          float v340_data = r0[10];
          float v341_data = s0[10];
          float v343_data = r1[0];
          r1[0] = (v343_data + (v340_data * v341_data));
          float v346_data = s0[22];
          float v348_data = r1[1];
          r1[1] = (v348_data + (v340_data * v346_data));
          float v351_data = s0[34];
          float v353_data = r1[2];
          r1[2] = (v353_data + (v340_data * v351_data));
          float v356_data = s0[46];
          float v358_data = r1[3];
          r1[3] = (v358_data + (v340_data * v356_data));
          float v361_data = s0[58];
          float v363_data = r1[4];
          r1[4] = (v363_data + (v340_data * v361_data));
          float v366_data = s0[70];
          float v368_data = r1[5];
          r1[5] = (v368_data + (v340_data * v366_data));
          float v370_data = r0[11];
          float v371_data = s0[11];
          float v373_data = r1[0];
          r1[0] = (v373_data + (v370_data * v371_data));
          float v376_data = s0[23];
          float v378_data = r1[1];
          r1[1] = (v378_data + (v370_data * v376_data));
          float v381_data = s0[35];
          float v383_data = r1[2];
          r1[2] = (v383_data + (v370_data * v381_data));
          float v386_data = s0[47];
          float v388_data = r1[3];
          r1[3] = (v388_data + (v370_data * v386_data));
          float v391_data = s0[59];
          float v393_data = r1[4];
          r1[4] = (v393_data + (v370_data * v391_data));
          float v396_data = s0[71];
          float v398_data = r1[5];
          r1[5] = (v398_data + (v370_data * v396_data));
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // s1 = store{r>s, clear}(localShrMem0, r1);
          if (v22_g) {
            #pragma unroll
            for (int32_t v400_z1 = 6; v400_z1 < 12; ++v400_z1) {
              int32_t v405_a = v21_lead + (v400_z1 * 12);
              s1[(v405_a ^ ((v405_a >> 4) & 15))] = 0.0f;
            }
          }
          if (v22_g) {
            #pragma unroll
            for (int32_t v409_i1 = 0; v409_i1 < 6; ++v409_i1) {
              float v411_data = r1[v409_i1];
              int32_t v415_a = v21_lead + (v409_i1 * 12);
              s1[(v415_a ^ ((v415_a >> 4) & 15))] = v411_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m3););
          float r3[12]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // ir3 = +(r2 * s1)
          // [(0, 12), (0, 12)] [(0, 12)]
          float ir3[12]{};
          float v421_data = r2[0];
          float v422_data = s1[0];
          float v424_data = ir3[0];
          ir3[0] = (v424_data + (v421_data * v422_data));
          float v427_data = s1[12];
          float v429_data = ir3[1];
          ir3[1] = (v429_data + (v421_data * v427_data));
          float v432_data = s1[25];
          float v434_data = ir3[2];
          ir3[2] = (v434_data + (v421_data * v432_data));
          float v437_data = s1[38];
          float v439_data = ir3[3];
          ir3[3] = (v439_data + (v421_data * v437_data));
          float v442_data = s1[51];
          float v444_data = ir3[4];
          ir3[4] = (v444_data + (v421_data * v442_data));
          float v447_data = s1[63];
          float v449_data = ir3[5];
          ir3[5] = (v449_data + (v421_data * v447_data));
          float v452_data = s1[76];
          float v454_data = ir3[6];
          ir3[6] = (v454_data + (v421_data * v452_data));
          float v457_data = s1[81];
          float v459_data = ir3[7];
          ir3[7] = (v459_data + (v421_data * v457_data));
          float v462_data = s1[102];
          float v464_data = ir3[8];
          ir3[8] = (v464_data + (v421_data * v462_data));
          float v467_data = s1[106];
          float v469_data = ir3[9];
          ir3[9] = (v469_data + (v421_data * v467_data));
          float v472_data = s1[127];
          float v474_data = ir3[10];
          ir3[10] = (v474_data + (v421_data * v472_data));
          float v477_data = s1[140];
          float v479_data = ir3[11];
          ir3[11] = (v479_data + (v421_data * v477_data));
          float v481_data = r2[1];
          float v482_data = s1[1];
          float v484_data = ir3[0];
          ir3[0] = (v484_data + (v481_data * v482_data));
          float v487_data = s1[13];
          float v489_data = ir3[1];
          ir3[1] = (v489_data + (v481_data * v487_data));
          float v492_data = s1[24];
          float v494_data = ir3[2];
          ir3[2] = (v494_data + (v481_data * v492_data));
          float v497_data = s1[39];
          float v499_data = ir3[3];
          ir3[3] = (v499_data + (v481_data * v497_data));
          float v502_data = s1[50];
          float v504_data = ir3[4];
          ir3[4] = (v504_data + (v481_data * v502_data));
          float v507_data = s1[62];
          float v509_data = ir3[5];
          ir3[5] = (v509_data + (v481_data * v507_data));
          float v512_data = s1[77];
          float v514_data = ir3[6];
          ir3[6] = (v514_data + (v481_data * v512_data));
          float v517_data = s1[80];
          float v519_data = ir3[7];
          ir3[7] = (v519_data + (v481_data * v517_data));
          float v522_data = s1[103];
          float v524_data = ir3[8];
          ir3[8] = (v524_data + (v481_data * v522_data));
          float v527_data = s1[107];
          float v529_data = ir3[9];
          ir3[9] = (v529_data + (v481_data * v527_data));
          float v532_data = s1[126];
          float v534_data = ir3[10];
          ir3[10] = (v534_data + (v481_data * v532_data));
          float v537_data = s1[141];
          float v539_data = ir3[11];
          ir3[11] = (v539_data + (v481_data * v537_data));
          float v541_data = r2[2];
          float v542_data = s1[2];
          float v544_data = ir3[0];
          ir3[0] = (v544_data + (v541_data * v542_data));
          float v547_data = s1[14];
          float v549_data = ir3[1];
          ir3[1] = (v549_data + (v541_data * v547_data));
          float v552_data = s1[27];
          float v554_data = ir3[2];
          ir3[2] = (v554_data + (v541_data * v552_data));
          float v557_data = s1[36];
          float v559_data = ir3[3];
          ir3[3] = (v559_data + (v541_data * v557_data));
          float v562_data = s1[49];
          float v564_data = ir3[4];
          ir3[4] = (v564_data + (v541_data * v562_data));
          float v567_data = s1[61];
          float v569_data = ir3[5];
          ir3[5] = (v569_data + (v541_data * v567_data));
          float v572_data = s1[78];
          float v574_data = ir3[6];
          ir3[6] = (v574_data + (v541_data * v572_data));
          float v577_data = s1[83];
          float v579_data = ir3[7];
          ir3[7] = (v579_data + (v541_data * v577_data));
          float v582_data = s1[100];
          float v584_data = ir3[8];
          ir3[8] = (v584_data + (v541_data * v582_data));
          float v587_data = s1[104];
          float v589_data = ir3[9];
          ir3[9] = (v589_data + (v541_data * v587_data));
          float v592_data = s1[125];
          float v594_data = ir3[10];
          ir3[10] = (v594_data + (v541_data * v592_data));
          float v597_data = s1[142];
          float v599_data = ir3[11];
          ir3[11] = (v599_data + (v541_data * v597_data));
          float v601_data = r2[3];
          float v602_data = s1[3];
          float v604_data = ir3[0];
          ir3[0] = (v604_data + (v601_data * v602_data));
          float v607_data = s1[15];
          float v609_data = ir3[1];
          ir3[1] = (v609_data + (v601_data * v607_data));
          float v612_data = s1[26];
          float v614_data = ir3[2];
          ir3[2] = (v614_data + (v601_data * v612_data));
          float v617_data = s1[37];
          float v619_data = ir3[3];
          ir3[3] = (v619_data + (v601_data * v617_data));
          float v622_data = s1[48];
          float v624_data = ir3[4];
          ir3[4] = (v624_data + (v601_data * v622_data));
          float v627_data = s1[60];
          float v629_data = ir3[5];
          ir3[5] = (v629_data + (v601_data * v627_data));
          float v632_data = s1[79];
          float v634_data = ir3[6];
          ir3[6] = (v634_data + (v601_data * v632_data));
          float v637_data = s1[82];
          float v639_data = ir3[7];
          ir3[7] = (v639_data + (v601_data * v637_data));
          float v642_data = s1[101];
          float v644_data = ir3[8];
          ir3[8] = (v644_data + (v601_data * v642_data));
          float v647_data = s1[105];
          float v649_data = ir3[9];
          ir3[9] = (v649_data + (v601_data * v647_data));
          float v652_data = s1[124];
          float v654_data = ir3[10];
          ir3[10] = (v654_data + (v601_data * v652_data));
          float v657_data = s1[143];
          float v659_data = ir3[11];
          ir3[11] = (v659_data + (v601_data * v657_data));
          float v661_data = r2[4];
          float v662_data = s1[4];
          float v664_data = ir3[0];
          ir3[0] = (v664_data + (v661_data * v662_data));
          float v667_data = s1[17];
          float v669_data = ir3[1];
          ir3[1] = (v669_data + (v661_data * v667_data));
          float v672_data = s1[29];
          float v674_data = ir3[2];
          ir3[2] = (v674_data + (v661_data * v672_data));
          float v677_data = s1[42];
          float v679_data = ir3[3];
          ir3[3] = (v679_data + (v661_data * v677_data));
          float v682_data = s1[55];
          float v684_data = ir3[4];
          ir3[4] = (v684_data + (v661_data * v682_data));
          float v687_data = s1[68];
          float v689_data = ir3[5];
          ir3[5] = (v689_data + (v661_data * v687_data));
          float v692_data = s1[72];
          float v694_data = ir3[6];
          ir3[6] = (v694_data + (v661_data * v692_data));
          float v697_data = s1[93];
          float v699_data = ir3[7];
          ir3[7] = (v699_data + (v661_data * v697_data));
          float v702_data = s1[98];
          float v704_data = ir3[8];
          ir3[8] = (v704_data + (v661_data * v702_data));
          float v707_data = s1[119];
          float v709_data = ir3[9];
          ir3[9] = (v709_data + (v661_data * v707_data));
          float v712_data = s1[123];
          float v714_data = ir3[10];
          ir3[10] = (v714_data + (v661_data * v712_data));
          float v717_data = s1[128];
          float v719_data = ir3[11];
          ir3[11] = (v719_data + (v661_data * v717_data));
          float v721_data = r2[5];
          float v722_data = s1[5];
          float v724_data = ir3[0];
          ir3[0] = (v724_data + (v721_data * v722_data));
          float v727_data = s1[16];
          float v729_data = ir3[1];
          ir3[1] = (v729_data + (v721_data * v727_data));
          float v732_data = s1[28];
          float v734_data = ir3[2];
          ir3[2] = (v734_data + (v721_data * v732_data));
          float v737_data = s1[43];
          float v739_data = ir3[3];
          ir3[3] = (v739_data + (v721_data * v737_data));
          float v742_data = s1[54];
          float v744_data = ir3[4];
          ir3[4] = (v744_data + (v721_data * v742_data));
          float v747_data = s1[69];
          float v749_data = ir3[5];
          ir3[5] = (v749_data + (v721_data * v747_data));
          float v752_data = s1[73];
          float v754_data = ir3[6];
          ir3[6] = (v754_data + (v721_data * v752_data));
          float v757_data = s1[92];
          float v759_data = ir3[7];
          ir3[7] = (v759_data + (v721_data * v757_data));
          float v762_data = s1[99];
          float v764_data = ir3[8];
          ir3[8] = (v764_data + (v721_data * v762_data));
          float v767_data = s1[118];
          float v769_data = ir3[9];
          ir3[9] = (v769_data + (v721_data * v767_data));
          float v772_data = s1[122];
          float v774_data = ir3[10];
          ir3[10] = (v774_data + (v721_data * v772_data));
          float v777_data = s1[129];
          float v779_data = ir3[11];
          ir3[11] = (v779_data + (v721_data * v777_data));
          float v781_data = r2[6];
          float v782_data = s1[6];
          float v784_data = ir3[0];
          ir3[0] = (v784_data + (v781_data * v782_data));
          float v787_data = s1[19];
          float v789_data = ir3[1];
          ir3[1] = (v789_data + (v781_data * v787_data));
          float v792_data = s1[31];
          float v794_data = ir3[2];
          ir3[2] = (v794_data + (v781_data * v792_data));
          float v797_data = s1[40];
          float v799_data = ir3[3];
          ir3[3] = (v799_data + (v781_data * v797_data));
          float v802_data = s1[53];
          float v804_data = ir3[4];
          ir3[4] = (v804_data + (v781_data * v802_data));
          float v807_data = s1[70];
          float v809_data = ir3[5];
          ir3[5] = (v809_data + (v781_data * v807_data));
          float v812_data = s1[74];
          float v814_data = ir3[6];
          ir3[6] = (v814_data + (v781_data * v812_data));
          float v817_data = s1[95];
          float v819_data = ir3[7];
          ir3[7] = (v819_data + (v781_data * v817_data));
          float v822_data = s1[96];
          float v824_data = ir3[8];
          ir3[8] = (v824_data + (v781_data * v822_data));
          float v827_data = s1[117];
          float v829_data = ir3[9];
          ir3[9] = (v829_data + (v781_data * v827_data));
          float v832_data = s1[121];
          float v834_data = ir3[10];
          ir3[10] = (v834_data + (v781_data * v832_data));
          float v837_data = s1[130];
          float v839_data = ir3[11];
          ir3[11] = (v839_data + (v781_data * v837_data));
          float v841_data = r2[7];
          float v842_data = s1[7];
          float v844_data = ir3[0];
          ir3[0] = (v844_data + (v841_data * v842_data));
          float v847_data = s1[18];
          float v849_data = ir3[1];
          ir3[1] = (v849_data + (v841_data * v847_data));
          float v852_data = s1[30];
          float v854_data = ir3[2];
          ir3[2] = (v854_data + (v841_data * v852_data));
          float v857_data = s1[41];
          float v859_data = ir3[3];
          ir3[3] = (v859_data + (v841_data * v857_data));
          float v862_data = s1[52];
          float v864_data = ir3[4];
          ir3[4] = (v864_data + (v841_data * v862_data));
          float v867_data = s1[71];
          float v869_data = ir3[5];
          ir3[5] = (v869_data + (v841_data * v867_data));
          float v872_data = s1[75];
          float v874_data = ir3[6];
          ir3[6] = (v874_data + (v841_data * v872_data));
          float v877_data = s1[94];
          float v879_data = ir3[7];
          ir3[7] = (v879_data + (v841_data * v877_data));
          float v882_data = s1[97];
          float v884_data = ir3[8];
          ir3[8] = (v884_data + (v841_data * v882_data));
          float v887_data = s1[116];
          float v889_data = ir3[9];
          ir3[9] = (v889_data + (v841_data * v887_data));
          float v892_data = s1[120];
          float v894_data = ir3[10];
          ir3[10] = (v894_data + (v841_data * v892_data));
          float v897_data = s1[131];
          float v899_data = ir3[11];
          ir3[11] = (v899_data + (v841_data * v897_data));
          float v901_data = r2[8];
          float v902_data = s1[8];
          float v904_data = ir3[0];
          ir3[0] = (v904_data + (v901_data * v902_data));
          float v907_data = s1[21];
          float v909_data = ir3[1];
          ir3[1] = (v909_data + (v901_data * v907_data));
          float v912_data = s1[34];
          float v914_data = ir3[2];
          ir3[2] = (v914_data + (v901_data * v912_data));
          float v917_data = s1[46];
          float v919_data = ir3[3];
          ir3[3] = (v919_data + (v901_data * v917_data));
          float v922_data = s1[59];
          float v924_data = ir3[4];
          ir3[4] = (v924_data + (v901_data * v922_data));
          float v927_data = s1[64];
          float v929_data = ir3[5];
          ir3[5] = (v929_data + (v901_data * v927_data));
          float v932_data = s1[85];
          float v934_data = ir3[6];
          ir3[6] = (v934_data + (v901_data * v932_data));
          float v937_data = s1[89];
          float v939_data = ir3[7];
          ir3[7] = (v939_data + (v901_data * v937_data));
          float v942_data = s1[110];
          float v944_data = ir3[8];
          ir3[8] = (v944_data + (v901_data * v942_data));
          float v947_data = s1[115];
          float v949_data = ir3[9];
          ir3[9] = (v949_data + (v901_data * v947_data));
          float v952_data = s1[136];
          float v954_data = ir3[10];
          ir3[10] = (v954_data + (v901_data * v952_data));
          float v957_data = s1[132];
          float v959_data = ir3[11];
          ir3[11] = (v959_data + (v901_data * v957_data));
          float v961_data = r2[9];
          float v962_data = s1[9];
          float v964_data = ir3[0];
          ir3[0] = (v964_data + (v961_data * v962_data));
          float v967_data = s1[20];
          float v969_data = ir3[1];
          ir3[1] = (v969_data + (v961_data * v967_data));
          float v972_data = s1[35];
          float v974_data = ir3[2];
          ir3[2] = (v974_data + (v961_data * v972_data));
          float v977_data = s1[47];
          float v979_data = ir3[3];
          ir3[3] = (v979_data + (v961_data * v977_data));
          float v982_data = s1[58];
          float v984_data = ir3[4];
          ir3[4] = (v984_data + (v961_data * v982_data));
          float v987_data = s1[65];
          float v989_data = ir3[5];
          ir3[5] = (v989_data + (v961_data * v987_data));
          float v992_data = s1[84];
          float v994_data = ir3[6];
          ir3[6] = (v994_data + (v961_data * v992_data));
          float v997_data = s1[88];
          float v999_data = ir3[7];
          ir3[7] = (v999_data + (v961_data * v997_data));
          float v1002_data = s1[111];
          float v1004_data = ir3[8];
          ir3[8] = (v1004_data + (v961_data * v1002_data));
          float v1007_data = s1[114];
          float v1009_data = ir3[9];
          ir3[9] = (v1009_data + (v961_data * v1007_data));
          float v1012_data = s1[137];
          float v1014_data = ir3[10];
          ir3[10] = (v1014_data + (v961_data * v1012_data));
          float v1017_data = s1[133];
          float v1019_data = ir3[11];
          ir3[11] = (v1019_data + (v961_data * v1017_data));
          float v1021_data = r2[10];
          float v1022_data = s1[10];
          float v1024_data = ir3[0];
          ir3[0] = (v1024_data + (v1021_data * v1022_data));
          float v1027_data = s1[23];
          float v1029_data = ir3[1];
          ir3[1] = (v1029_data + (v1021_data * v1027_data));
          float v1032_data = s1[32];
          float v1034_data = ir3[2];
          ir3[2] = (v1034_data + (v1021_data * v1032_data));
          float v1037_data = s1[44];
          float v1039_data = ir3[3];
          ir3[3] = (v1039_data + (v1021_data * v1037_data));
          float v1042_data = s1[57];
          float v1044_data = ir3[4];
          ir3[4] = (v1044_data + (v1021_data * v1042_data));
          float v1047_data = s1[66];
          float v1049_data = ir3[5];
          ir3[5] = (v1049_data + (v1021_data * v1047_data));
          float v1052_data = s1[87];
          float v1054_data = ir3[6];
          ir3[6] = (v1054_data + (v1021_data * v1052_data));
          float v1057_data = s1[91];
          float v1059_data = ir3[7];
          ir3[7] = (v1059_data + (v1021_data * v1057_data));
          float v1062_data = s1[108];
          float v1064_data = ir3[8];
          ir3[8] = (v1064_data + (v1021_data * v1062_data));
          float v1067_data = s1[113];
          float v1069_data = ir3[9];
          ir3[9] = (v1069_data + (v1021_data * v1067_data));
          float v1072_data = s1[138];
          float v1074_data = ir3[10];
          ir3[10] = (v1074_data + (v1021_data * v1072_data));
          float v1077_data = s1[134];
          float v1079_data = ir3[11];
          ir3[11] = (v1079_data + (v1021_data * v1077_data));
          float v1081_data = r2[11];
          float v1082_data = s1[11];
          float v1084_data = ir3[0];
          ir3[0] = (v1084_data + (v1081_data * v1082_data));
          float v1087_data = s1[22];
          float v1089_data = ir3[1];
          ir3[1] = (v1089_data + (v1081_data * v1087_data));
          float v1092_data = s1[33];
          float v1094_data = ir3[2];
          ir3[2] = (v1094_data + (v1081_data * v1092_data));
          float v1097_data = s1[45];
          float v1099_data = ir3[3];
          ir3[3] = (v1099_data + (v1081_data * v1097_data));
          float v1102_data = s1[56];
          float v1104_data = ir3[4];
          ir3[4] = (v1104_data + (v1081_data * v1102_data));
          float v1107_data = s1[67];
          float v1109_data = ir3[5];
          ir3[5] = (v1109_data + (v1081_data * v1107_data));
          float v1112_data = s1[86];
          float v1114_data = ir3[6];
          ir3[6] = (v1114_data + (v1081_data * v1112_data));
          float v1117_data = s1[90];
          float v1119_data = ir3[7];
          ir3[7] = (v1119_data + (v1081_data * v1117_data));
          float v1122_data = s1[109];
          float v1124_data = ir3[8];
          ir3[8] = (v1124_data + (v1081_data * v1122_data));
          float v1127_data = s1[112];
          float v1129_data = ir3[9];
          ir3[9] = (v1129_data + (v1081_data * v1127_data));
          float v1132_data = s1[139];
          float v1134_data = ir3[10];
          ir3[10] = (v1134_data + (v1081_data * v1132_data));
          float v1137_data = s1[135];
          float v1139_data = ir3[11];
          ir3[11] = (v1139_data + (v1081_data * v1137_data));
          // r3 = ir3
          if (v22_g) {
            #pragma unroll
            for (int32_t v1141_n1 = 0; v1141_n1 < 12; ++v1141_n1) {
              float v1143_data = ir3[v1141_n1];
              r3[v1141_n1] = v1143_data;
            }
          }
          // glb_m2 = store{r>g}(r3);
          if (v22_g) {
            #pragma unroll
            for (int32_t v1144_i1 = 0; v1144_i1 < 12; ++v1144_i1) {
              float v1146_data = r3[v1144_i1];
              glb_m2[(v21_lead + (v1144_i1 * 12))] = v1146_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

