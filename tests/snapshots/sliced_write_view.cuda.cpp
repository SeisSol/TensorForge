// === base name ===
kernel_9d70efc396b5b531

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_9d70efc396b5b531 = {{32, 4, 1}, 32, 32, 1, 4, 3072, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_9d70efc396b5b531(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_9d70efc396b5b531(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_9d70efc396b5b531(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_9d70efc396b5b531, block.x * block.y * block.z, 768 * sizeof(float));
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
void launcher_kernel_9d70efc396b5b531(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_9d70efc396b5b531(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_9d70efc396b5b531, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_9d70efc396b5b531<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_9d70efc396b5b531(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
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
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < numElements0; v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 416 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v9_batchId0 * 416 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v9_batchId0 * 169 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[v9_batchId0 * 416 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[v9_batchId0 * 169 + 0 + m4_extraOffset];
          float r0[3]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v25_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v26_i0 = 0; v26_i0 < 1; ++v26_i0) {
            int32_t v29_lead = v25_lead + (v26_i0 * 32);
            #pragma unroll
            for (int32_t v27_i1 = 10; v27_i1 < 13; ++v27_i1) {
              float v32_data = __ldcg(&glb_m1[(v29_lead + (v27_i1 * 32))]);
              r0[(v26_i0 + (v27_i1 - 10))] = v32_data;
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
          // ir1 = +(r0 * s0)
          // [(0, 32), (0, 1)] [(10, 13)]
          float ir1[1]{};
          float v39_data = r0[0];
          __syncwarp();
          float v40_data = s0[114];
          float v42_data = ir1[0];
          ir1[0] = (v42_data + (v39_data * v40_data));
          float v44_data = r0[1];
          float v45_data = s0[115];
          float v47_data = ir1[0];
          ir1[0] = (v47_data + (v44_data * v45_data));
          float v49_data = r0[2];
          float v50_data = s0[116];
          float v52_data = ir1[0];
          ir1[0] = (v52_data + (v49_data * v50_data));
          // r1 = ir1
          #pragma unroll
          for (int32_t v54_n0 = 0; v54_n0 < 1; ++v54_n0) {
            #pragma unroll
            for (int32_t v55_n1 = 0; v55_n1 < 1; ++v55_n1) {
              int32_t v56_a = v54_n0 + v55_n1;
              float v57_data = ir1[v56_a];
              r1[v56_a] = v57_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v58_i0 = 0; v58_i0 < 1; ++v58_i0) {
            int32_t v63_lead = v25_lead + (v58_i0 * 32);
            #pragma unroll
            for (int32_t v59_i1 = 0; v59_i1 < 1; ++v59_i1) {
              float v61_data = r1[(v58_i0 + v59_i1)];
              glb_m0[(v63_lead + ((v59_i1 + 8) * 32))] = v61_data;
            }
          }
          float r2[13]{};
          // r2 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v68_i0 = 0; v68_i0 < 1; ++v68_i0) {
            int32_t v71_lead = v25_lead + (v68_i0 * 32);
            #pragma unroll
            for (int32_t v69_i1 = 0; v69_i1 < 13; ++v69_i1) {
              float v74_data = glb_m0[(v71_lead + (v69_i1 * 32))];
              r2[(v68_i0 + v69_i1)] = v74_data;
            }
          }
          // s1 = load{g>s}(glb_m4[0, 1])
          __syncwarp();
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
          // ir3 = +(r2 * s1)
          // [(0, 32), (0, 13)] [(0, 13)]
          float ir3[13]{};
          float v80_data = r2[0];
          __syncwarp();
          float v81_data = s1[0];
          float v83_data = ir3[0];
          ir3[0] = (v83_data + (v80_data * v81_data));
          float v86_data = s1[13];
          float v88_data = ir3[1];
          ir3[1] = (v88_data + (v80_data * v86_data));
          float v91_data = s1[26];
          float v93_data = ir3[2];
          ir3[2] = (v93_data + (v80_data * v91_data));
          float v96_data = s1[39];
          float v98_data = ir3[3];
          ir3[3] = (v98_data + (v80_data * v96_data));
          float v101_data = s1[52];
          float v103_data = ir3[4];
          ir3[4] = (v103_data + (v80_data * v101_data));
          float v106_data = s1[65];
          float v108_data = ir3[5];
          ir3[5] = (v108_data + (v80_data * v106_data));
          float v111_data = s1[78];
          float v113_data = ir3[6];
          ir3[6] = (v113_data + (v80_data * v111_data));
          float v116_data = s1[91];
          float v118_data = ir3[7];
          ir3[7] = (v118_data + (v80_data * v116_data));
          float v121_data = s1[104];
          float v123_data = ir3[8];
          ir3[8] = (v123_data + (v80_data * v121_data));
          float v126_data = s1[117];
          float v128_data = ir3[9];
          ir3[9] = (v128_data + (v80_data * v126_data));
          float v131_data = s1[130];
          float v133_data = ir3[10];
          ir3[10] = (v133_data + (v80_data * v131_data));
          float v136_data = s1[143];
          float v138_data = ir3[11];
          ir3[11] = (v138_data + (v80_data * v136_data));
          float v141_data = s1[156];
          float v143_data = ir3[12];
          ir3[12] = (v143_data + (v80_data * v141_data));
          float v145_data = r2[1];
          float v146_data = s1[1];
          float v148_data = ir3[0];
          ir3[0] = (v148_data + (v145_data * v146_data));
          float v151_data = s1[14];
          float v153_data = ir3[1];
          ir3[1] = (v153_data + (v145_data * v151_data));
          float v156_data = s1[27];
          float v158_data = ir3[2];
          ir3[2] = (v158_data + (v145_data * v156_data));
          float v161_data = s1[40];
          float v163_data = ir3[3];
          ir3[3] = (v163_data + (v145_data * v161_data));
          float v166_data = s1[53];
          float v168_data = ir3[4];
          ir3[4] = (v168_data + (v145_data * v166_data));
          float v171_data = s1[66];
          float v173_data = ir3[5];
          ir3[5] = (v173_data + (v145_data * v171_data));
          float v176_data = s1[79];
          float v178_data = ir3[6];
          ir3[6] = (v178_data + (v145_data * v176_data));
          float v181_data = s1[92];
          float v183_data = ir3[7];
          ir3[7] = (v183_data + (v145_data * v181_data));
          float v186_data = s1[105];
          float v188_data = ir3[8];
          ir3[8] = (v188_data + (v145_data * v186_data));
          float v191_data = s1[118];
          float v193_data = ir3[9];
          ir3[9] = (v193_data + (v145_data * v191_data));
          float v196_data = s1[131];
          float v198_data = ir3[10];
          ir3[10] = (v198_data + (v145_data * v196_data));
          float v201_data = s1[144];
          float v203_data = ir3[11];
          ir3[11] = (v203_data + (v145_data * v201_data));
          float v206_data = s1[157];
          float v208_data = ir3[12];
          ir3[12] = (v208_data + (v145_data * v206_data));
          float v210_data = r2[2];
          float v211_data = s1[2];
          float v213_data = ir3[0];
          ir3[0] = (v213_data + (v210_data * v211_data));
          float v216_data = s1[15];
          float v218_data = ir3[1];
          ir3[1] = (v218_data + (v210_data * v216_data));
          float v221_data = s1[28];
          float v223_data = ir3[2];
          ir3[2] = (v223_data + (v210_data * v221_data));
          float v226_data = s1[41];
          float v228_data = ir3[3];
          ir3[3] = (v228_data + (v210_data * v226_data));
          float v231_data = s1[54];
          float v233_data = ir3[4];
          ir3[4] = (v233_data + (v210_data * v231_data));
          float v236_data = s1[67];
          float v238_data = ir3[5];
          ir3[5] = (v238_data + (v210_data * v236_data));
          float v241_data = s1[80];
          float v243_data = ir3[6];
          ir3[6] = (v243_data + (v210_data * v241_data));
          float v246_data = s1[93];
          float v248_data = ir3[7];
          ir3[7] = (v248_data + (v210_data * v246_data));
          float v251_data = s1[106];
          float v253_data = ir3[8];
          ir3[8] = (v253_data + (v210_data * v251_data));
          float v256_data = s1[119];
          float v258_data = ir3[9];
          ir3[9] = (v258_data + (v210_data * v256_data));
          float v261_data = s1[132];
          float v263_data = ir3[10];
          ir3[10] = (v263_data + (v210_data * v261_data));
          float v266_data = s1[145];
          float v268_data = ir3[11];
          ir3[11] = (v268_data + (v210_data * v266_data));
          float v271_data = s1[158];
          float v273_data = ir3[12];
          ir3[12] = (v273_data + (v210_data * v271_data));
          float v275_data = r2[3];
          float v276_data = s1[3];
          float v278_data = ir3[0];
          ir3[0] = (v278_data + (v275_data * v276_data));
          float v281_data = s1[16];
          float v283_data = ir3[1];
          ir3[1] = (v283_data + (v275_data * v281_data));
          float v286_data = s1[29];
          float v288_data = ir3[2];
          ir3[2] = (v288_data + (v275_data * v286_data));
          float v291_data = s1[42];
          float v293_data = ir3[3];
          ir3[3] = (v293_data + (v275_data * v291_data));
          float v296_data = s1[55];
          float v298_data = ir3[4];
          ir3[4] = (v298_data + (v275_data * v296_data));
          float v301_data = s1[68];
          float v303_data = ir3[5];
          ir3[5] = (v303_data + (v275_data * v301_data));
          float v306_data = s1[81];
          float v308_data = ir3[6];
          ir3[6] = (v308_data + (v275_data * v306_data));
          float v311_data = s1[94];
          float v313_data = ir3[7];
          ir3[7] = (v313_data + (v275_data * v311_data));
          float v316_data = s1[107];
          float v318_data = ir3[8];
          ir3[8] = (v318_data + (v275_data * v316_data));
          float v321_data = s1[120];
          float v323_data = ir3[9];
          ir3[9] = (v323_data + (v275_data * v321_data));
          float v326_data = s1[133];
          float v328_data = ir3[10];
          ir3[10] = (v328_data + (v275_data * v326_data));
          float v331_data = s1[146];
          float v333_data = ir3[11];
          ir3[11] = (v333_data + (v275_data * v331_data));
          float v336_data = s1[159];
          float v338_data = ir3[12];
          ir3[12] = (v338_data + (v275_data * v336_data));
          float v340_data = r2[4];
          float v341_data = s1[4];
          float v343_data = ir3[0];
          ir3[0] = (v343_data + (v340_data * v341_data));
          float v346_data = s1[17];
          float v348_data = ir3[1];
          ir3[1] = (v348_data + (v340_data * v346_data));
          float v351_data = s1[30];
          float v353_data = ir3[2];
          ir3[2] = (v353_data + (v340_data * v351_data));
          float v356_data = s1[43];
          float v358_data = ir3[3];
          ir3[3] = (v358_data + (v340_data * v356_data));
          float v361_data = s1[56];
          float v363_data = ir3[4];
          ir3[4] = (v363_data + (v340_data * v361_data));
          float v366_data = s1[69];
          float v368_data = ir3[5];
          ir3[5] = (v368_data + (v340_data * v366_data));
          float v371_data = s1[82];
          float v373_data = ir3[6];
          ir3[6] = (v373_data + (v340_data * v371_data));
          float v376_data = s1[95];
          float v378_data = ir3[7];
          ir3[7] = (v378_data + (v340_data * v376_data));
          float v381_data = s1[108];
          float v383_data = ir3[8];
          ir3[8] = (v383_data + (v340_data * v381_data));
          float v386_data = s1[121];
          float v388_data = ir3[9];
          ir3[9] = (v388_data + (v340_data * v386_data));
          float v391_data = s1[134];
          float v393_data = ir3[10];
          ir3[10] = (v393_data + (v340_data * v391_data));
          float v396_data = s1[147];
          float v398_data = ir3[11];
          ir3[11] = (v398_data + (v340_data * v396_data));
          float v401_data = s1[160];
          float v403_data = ir3[12];
          ir3[12] = (v403_data + (v340_data * v401_data));
          float v405_data = r2[5];
          float v406_data = s1[5];
          float v408_data = ir3[0];
          ir3[0] = (v408_data + (v405_data * v406_data));
          float v411_data = s1[18];
          float v413_data = ir3[1];
          ir3[1] = (v413_data + (v405_data * v411_data));
          float v416_data = s1[31];
          float v418_data = ir3[2];
          ir3[2] = (v418_data + (v405_data * v416_data));
          float v421_data = s1[44];
          float v423_data = ir3[3];
          ir3[3] = (v423_data + (v405_data * v421_data));
          float v426_data = s1[57];
          float v428_data = ir3[4];
          ir3[4] = (v428_data + (v405_data * v426_data));
          float v431_data = s1[70];
          float v433_data = ir3[5];
          ir3[5] = (v433_data + (v405_data * v431_data));
          float v436_data = s1[83];
          float v438_data = ir3[6];
          ir3[6] = (v438_data + (v405_data * v436_data));
          float v441_data = s1[96];
          float v443_data = ir3[7];
          ir3[7] = (v443_data + (v405_data * v441_data));
          float v446_data = s1[109];
          float v448_data = ir3[8];
          ir3[8] = (v448_data + (v405_data * v446_data));
          float v451_data = s1[122];
          float v453_data = ir3[9];
          ir3[9] = (v453_data + (v405_data * v451_data));
          float v456_data = s1[135];
          float v458_data = ir3[10];
          ir3[10] = (v458_data + (v405_data * v456_data));
          float v461_data = s1[148];
          float v463_data = ir3[11];
          ir3[11] = (v463_data + (v405_data * v461_data));
          float v466_data = s1[161];
          float v468_data = ir3[12];
          ir3[12] = (v468_data + (v405_data * v466_data));
          float v470_data = r2[6];
          float v471_data = s1[6];
          float v473_data = ir3[0];
          ir3[0] = (v473_data + (v470_data * v471_data));
          float v476_data = s1[19];
          float v478_data = ir3[1];
          ir3[1] = (v478_data + (v470_data * v476_data));
          float v481_data = s1[32];
          float v483_data = ir3[2];
          ir3[2] = (v483_data + (v470_data * v481_data));
          float v486_data = s1[45];
          float v488_data = ir3[3];
          ir3[3] = (v488_data + (v470_data * v486_data));
          float v491_data = s1[58];
          float v493_data = ir3[4];
          ir3[4] = (v493_data + (v470_data * v491_data));
          float v496_data = s1[71];
          float v498_data = ir3[5];
          ir3[5] = (v498_data + (v470_data * v496_data));
          float v501_data = s1[84];
          float v503_data = ir3[6];
          ir3[6] = (v503_data + (v470_data * v501_data));
          float v506_data = s1[97];
          float v508_data = ir3[7];
          ir3[7] = (v508_data + (v470_data * v506_data));
          float v511_data = s1[110];
          float v513_data = ir3[8];
          ir3[8] = (v513_data + (v470_data * v511_data));
          float v516_data = s1[123];
          float v518_data = ir3[9];
          ir3[9] = (v518_data + (v470_data * v516_data));
          float v521_data = s1[136];
          float v523_data = ir3[10];
          ir3[10] = (v523_data + (v470_data * v521_data));
          float v526_data = s1[149];
          float v528_data = ir3[11];
          ir3[11] = (v528_data + (v470_data * v526_data));
          float v531_data = s1[162];
          float v533_data = ir3[12];
          ir3[12] = (v533_data + (v470_data * v531_data));
          float v535_data = r2[7];
          float v536_data = s1[7];
          float v538_data = ir3[0];
          ir3[0] = (v538_data + (v535_data * v536_data));
          float v541_data = s1[20];
          float v543_data = ir3[1];
          ir3[1] = (v543_data + (v535_data * v541_data));
          float v546_data = s1[33];
          float v548_data = ir3[2];
          ir3[2] = (v548_data + (v535_data * v546_data));
          float v551_data = s1[46];
          float v553_data = ir3[3];
          ir3[3] = (v553_data + (v535_data * v551_data));
          float v556_data = s1[59];
          float v558_data = ir3[4];
          ir3[4] = (v558_data + (v535_data * v556_data));
          float v561_data = s1[72];
          float v563_data = ir3[5];
          ir3[5] = (v563_data + (v535_data * v561_data));
          float v566_data = s1[85];
          float v568_data = ir3[6];
          ir3[6] = (v568_data + (v535_data * v566_data));
          float v571_data = s1[98];
          float v573_data = ir3[7];
          ir3[7] = (v573_data + (v535_data * v571_data));
          float v576_data = s1[111];
          float v578_data = ir3[8];
          ir3[8] = (v578_data + (v535_data * v576_data));
          float v581_data = s1[124];
          float v583_data = ir3[9];
          ir3[9] = (v583_data + (v535_data * v581_data));
          float v586_data = s1[137];
          float v588_data = ir3[10];
          ir3[10] = (v588_data + (v535_data * v586_data));
          float v591_data = s1[150];
          float v593_data = ir3[11];
          ir3[11] = (v593_data + (v535_data * v591_data));
          float v596_data = s1[163];
          float v598_data = ir3[12];
          ir3[12] = (v598_data + (v535_data * v596_data));
          float v600_data = r2[8];
          float v601_data = s1[8];
          float v603_data = ir3[0];
          ir3[0] = (v603_data + (v600_data * v601_data));
          float v606_data = s1[21];
          float v608_data = ir3[1];
          ir3[1] = (v608_data + (v600_data * v606_data));
          float v611_data = s1[34];
          float v613_data = ir3[2];
          ir3[2] = (v613_data + (v600_data * v611_data));
          float v616_data = s1[47];
          float v618_data = ir3[3];
          ir3[3] = (v618_data + (v600_data * v616_data));
          float v621_data = s1[60];
          float v623_data = ir3[4];
          ir3[4] = (v623_data + (v600_data * v621_data));
          float v626_data = s1[73];
          float v628_data = ir3[5];
          ir3[5] = (v628_data + (v600_data * v626_data));
          float v631_data = s1[86];
          float v633_data = ir3[6];
          ir3[6] = (v633_data + (v600_data * v631_data));
          float v636_data = s1[99];
          float v638_data = ir3[7];
          ir3[7] = (v638_data + (v600_data * v636_data));
          float v641_data = s1[112];
          float v643_data = ir3[8];
          ir3[8] = (v643_data + (v600_data * v641_data));
          float v646_data = s1[125];
          float v648_data = ir3[9];
          ir3[9] = (v648_data + (v600_data * v646_data));
          float v651_data = s1[138];
          float v653_data = ir3[10];
          ir3[10] = (v653_data + (v600_data * v651_data));
          float v656_data = s1[151];
          float v658_data = ir3[11];
          ir3[11] = (v658_data + (v600_data * v656_data));
          float v661_data = s1[164];
          float v663_data = ir3[12];
          ir3[12] = (v663_data + (v600_data * v661_data));
          float v665_data = r2[9];
          float v666_data = s1[9];
          float v668_data = ir3[0];
          ir3[0] = (v668_data + (v665_data * v666_data));
          float v671_data = s1[22];
          float v673_data = ir3[1];
          ir3[1] = (v673_data + (v665_data * v671_data));
          float v676_data = s1[35];
          float v678_data = ir3[2];
          ir3[2] = (v678_data + (v665_data * v676_data));
          float v681_data = s1[48];
          float v683_data = ir3[3];
          ir3[3] = (v683_data + (v665_data * v681_data));
          float v686_data = s1[61];
          float v688_data = ir3[4];
          ir3[4] = (v688_data + (v665_data * v686_data));
          float v691_data = s1[74];
          float v693_data = ir3[5];
          ir3[5] = (v693_data + (v665_data * v691_data));
          float v696_data = s1[87];
          float v698_data = ir3[6];
          ir3[6] = (v698_data + (v665_data * v696_data));
          float v701_data = s1[100];
          float v703_data = ir3[7];
          ir3[7] = (v703_data + (v665_data * v701_data));
          float v706_data = s1[113];
          float v708_data = ir3[8];
          ir3[8] = (v708_data + (v665_data * v706_data));
          float v711_data = s1[126];
          float v713_data = ir3[9];
          ir3[9] = (v713_data + (v665_data * v711_data));
          float v716_data = s1[139];
          float v718_data = ir3[10];
          ir3[10] = (v718_data + (v665_data * v716_data));
          float v721_data = s1[152];
          float v723_data = ir3[11];
          ir3[11] = (v723_data + (v665_data * v721_data));
          float v726_data = s1[165];
          float v728_data = ir3[12];
          ir3[12] = (v728_data + (v665_data * v726_data));
          float v730_data = r2[10];
          float v731_data = s1[10];
          float v733_data = ir3[0];
          ir3[0] = (v733_data + (v730_data * v731_data));
          float v736_data = s1[23];
          float v738_data = ir3[1];
          ir3[1] = (v738_data + (v730_data * v736_data));
          float v741_data = s1[36];
          float v743_data = ir3[2];
          ir3[2] = (v743_data + (v730_data * v741_data));
          float v746_data = s1[49];
          float v748_data = ir3[3];
          ir3[3] = (v748_data + (v730_data * v746_data));
          float v751_data = s1[62];
          float v753_data = ir3[4];
          ir3[4] = (v753_data + (v730_data * v751_data));
          float v756_data = s1[75];
          float v758_data = ir3[5];
          ir3[5] = (v758_data + (v730_data * v756_data));
          float v761_data = s1[88];
          float v763_data = ir3[6];
          ir3[6] = (v763_data + (v730_data * v761_data));
          float v766_data = s1[101];
          float v768_data = ir3[7];
          ir3[7] = (v768_data + (v730_data * v766_data));
          float v771_data = s1[114];
          float v773_data = ir3[8];
          ir3[8] = (v773_data + (v730_data * v771_data));
          float v776_data = s1[127];
          float v778_data = ir3[9];
          ir3[9] = (v778_data + (v730_data * v776_data));
          float v781_data = s1[140];
          float v783_data = ir3[10];
          ir3[10] = (v783_data + (v730_data * v781_data));
          float v786_data = s1[153];
          float v788_data = ir3[11];
          ir3[11] = (v788_data + (v730_data * v786_data));
          float v791_data = s1[166];
          float v793_data = ir3[12];
          ir3[12] = (v793_data + (v730_data * v791_data));
          float v795_data = r2[11];
          float v796_data = s1[11];
          float v798_data = ir3[0];
          ir3[0] = (v798_data + (v795_data * v796_data));
          float v801_data = s1[24];
          float v803_data = ir3[1];
          ir3[1] = (v803_data + (v795_data * v801_data));
          float v806_data = s1[37];
          float v808_data = ir3[2];
          ir3[2] = (v808_data + (v795_data * v806_data));
          float v811_data = s1[50];
          float v813_data = ir3[3];
          ir3[3] = (v813_data + (v795_data * v811_data));
          float v816_data = s1[63];
          float v818_data = ir3[4];
          ir3[4] = (v818_data + (v795_data * v816_data));
          float v821_data = s1[76];
          float v823_data = ir3[5];
          ir3[5] = (v823_data + (v795_data * v821_data));
          float v826_data = s1[89];
          float v828_data = ir3[6];
          ir3[6] = (v828_data + (v795_data * v826_data));
          float v831_data = s1[102];
          float v833_data = ir3[7];
          ir3[7] = (v833_data + (v795_data * v831_data));
          float v836_data = s1[115];
          float v838_data = ir3[8];
          ir3[8] = (v838_data + (v795_data * v836_data));
          float v841_data = s1[128];
          float v843_data = ir3[9];
          ir3[9] = (v843_data + (v795_data * v841_data));
          float v846_data = s1[141];
          float v848_data = ir3[10];
          ir3[10] = (v848_data + (v795_data * v846_data));
          float v851_data = s1[154];
          float v853_data = ir3[11];
          ir3[11] = (v853_data + (v795_data * v851_data));
          float v856_data = s1[167];
          float v858_data = ir3[12];
          ir3[12] = (v858_data + (v795_data * v856_data));
          float v860_data = r2[12];
          float v861_data = s1[12];
          float v863_data = ir3[0];
          ir3[0] = (v863_data + (v860_data * v861_data));
          float v866_data = s1[25];
          float v868_data = ir3[1];
          ir3[1] = (v868_data + (v860_data * v866_data));
          float v871_data = s1[38];
          float v873_data = ir3[2];
          ir3[2] = (v873_data + (v860_data * v871_data));
          float v876_data = s1[51];
          float v878_data = ir3[3];
          ir3[3] = (v878_data + (v860_data * v876_data));
          float v881_data = s1[64];
          float v883_data = ir3[4];
          ir3[4] = (v883_data + (v860_data * v881_data));
          float v886_data = s1[77];
          float v888_data = ir3[5];
          ir3[5] = (v888_data + (v860_data * v886_data));
          float v891_data = s1[90];
          float v893_data = ir3[6];
          ir3[6] = (v893_data + (v860_data * v891_data));
          float v896_data = s1[103];
          float v898_data = ir3[7];
          ir3[7] = (v898_data + (v860_data * v896_data));
          float v901_data = s1[116];
          float v903_data = ir3[8];
          ir3[8] = (v903_data + (v860_data * v901_data));
          float v906_data = s1[129];
          float v908_data = ir3[9];
          ir3[9] = (v908_data + (v860_data * v906_data));
          float v911_data = s1[142];
          float v913_data = ir3[10];
          ir3[10] = (v913_data + (v860_data * v911_data));
          float v916_data = s1[155];
          float v918_data = ir3[11];
          ir3[11] = (v918_data + (v860_data * v916_data));
          float v921_data = s1[168];
          float v923_data = ir3[12];
          ir3[12] = (v923_data + (v860_data * v921_data));
          // r3 = ir3
          #pragma unroll
          for (int32_t v925_n0 = 0; v925_n0 < 1; ++v925_n0) {
            #pragma unroll
            for (int32_t v926_n1 = 0; v926_n1 < 13; ++v926_n1) {
              int32_t v927_a = v925_n0 + v926_n1;
              float v928_data = ir3[v927_a];
              r3[v927_a] = v928_data;
            }
          }
          // glb_m3 = store{r>g}(r3);
          #pragma unroll
          for (int32_t v929_i0 = 0; v929_i0 < 1; ++v929_i0) {
            int32_t v934_lead = v25_lead + (v929_i0 * 32);
            #pragma unroll
            for (int32_t v930_i1 = 0; v930_i1 < 13; ++v930_i1) {
              float v932_data = r3[(v929_i0 + v930_i1)];
              glb_m3[(v934_lead + (v930_i1 * 32))] = v932_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

