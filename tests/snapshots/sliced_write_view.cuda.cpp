// === base name ===
kernel_8fbe5a020d2ae0ac

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_8fbe5a020d2ae0ac = {{32, 4, 1}, 32, 32, 1, 4, 3072, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_8fbe5a020d2ae0ac(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_8fbe5a020d2ae0ac(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_8fbe5a020d2ae0ac(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_8fbe5a020d2ae0ac, block.x * block.y * block.z, 768 * sizeof(float));
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
void launcher_kernel_8fbe5a020d2ae0ac(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_8fbe5a020d2ae0ac(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_8fbe5a020d2ae0ac, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_8fbe5a020d2ae0ac<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_8fbe5a020d2ae0ac(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
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
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":768}],"shared_bytes":3072,"shared_elements":768,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[32,13]],"name":"m0","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"Q","bbox":[[0,0],[32,13]],"name":"m1","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"S","bbox":[[0,0],[13,13]],"name":"m2","ordered":false,"parts":1,"shape":[13,13],"variant":false},{"addressing":"strided","alias":"O","bbox":[[0,0],[32,13]],"name":"m3","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[13,13]],"name":"m4","ordered":false,"parts":1,"shape":[13,13],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,1]],"is_tmp":false,"name":"m0","offset":[0,8],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,10],[32,13]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[10,0],[13,1]],"is_tmp":false,"name":"m2","offset":[0,8],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[0,0],[13,13]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[192 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[192];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v6_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v6_batchId0 < numElements0; v6_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v7_ahead1 = v6_batchId0 + (gridDim.x * blockDim.y);
        size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 416 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v6_batchId0 * 416 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v6_batchId0 * 169 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[v6_batchId0 * 416 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[v6_batchId0 * 169 + 0 + m4_extraOffset];
          float r0[3]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v22_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
            int32_t v26_lead = v22_lead + (v23_i0 * 32);
            #pragma unroll
            for (int32_t v24_i1 = 10; v24_i1 < 13; ++v24_i1) {
              float v29_data = __ldcg(&glb_m1[(v26_lead + (v24_i1 * 32))]);
              r0[(v23_i0 + (v24_i1 - 10))] = v29_data;
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
          // r1 = +(r0 * s0) + None
          // [(0, 32), (0, 1)] [(10, 13)]
          float ir1[1]{};
          float v36_data = r0[0];
          float v37_data = s0[114];
          float v39_data = ir1[0];
          ir1[0] = (v39_data + (v36_data * v37_data));
          float v41_data = r0[1];
          float v42_data = s0[115];
          float v44_data = ir1[0];
          ir1[0] = (v44_data + (v41_data * v42_data));
          float v46_data = r0[2];
          float v47_data = s0[116];
          float v49_data = ir1[0];
          ir1[0] = (v49_data + (v46_data * v47_data));
          #pragma unroll
          for (int32_t v51_n0 = 0; v51_n0 < 1; ++v51_n0) {
            #pragma unroll
            for (int32_t v52_n1 = 0; v52_n1 < 1; ++v52_n1) {
              int32_t v53_a = v51_n0 + v52_n1;
              float v54_data = ir1[v53_a];
              r1[v53_a] = v54_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v55_i0 = 0; v55_i0 < 1; ++v55_i0) {
            int32_t v60_lead = v22_lead + (v55_i0 * 32);
            #pragma unroll
            for (int32_t v56_i1 = 0; v56_i1 < 1; ++v56_i1) {
              float v58_data = r1[(v55_i0 + v56_i1)];
              glb_m0[(v60_lead + ((v56_i1 + 8) * 32))] = v58_data;
            }
          }
          float r2[13]{};
          // r2 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v65_i0 = 0; v65_i0 < 1; ++v65_i0) {
            int32_t v68_lead = v22_lead + (v65_i0 * 32);
            #pragma unroll
            for (int32_t v66_i1 = 0; v66_i1 < 13; ++v66_i1) {
              float v71_data = glb_m0[(v68_lead + (v66_i1 * 32))];
              r2[(v65_i0 + v66_i1)] = v71_data;
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
          // r3 = +(r2 * s1) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float ir3[13]{};
          float v77_data = r2[0];
          float v78_data = s1[0];
          float v80_data = ir3[0];
          ir3[0] = (v80_data + (v77_data * v78_data));
          float v83_data = s1[13];
          float v85_data = ir3[1];
          ir3[1] = (v85_data + (v77_data * v83_data));
          float v88_data = s1[26];
          float v90_data = ir3[2];
          ir3[2] = (v90_data + (v77_data * v88_data));
          float v93_data = s1[39];
          float v95_data = ir3[3];
          ir3[3] = (v95_data + (v77_data * v93_data));
          float v98_data = s1[52];
          float v100_data = ir3[4];
          ir3[4] = (v100_data + (v77_data * v98_data));
          float v103_data = s1[65];
          float v105_data = ir3[5];
          ir3[5] = (v105_data + (v77_data * v103_data));
          float v108_data = s1[78];
          float v110_data = ir3[6];
          ir3[6] = (v110_data + (v77_data * v108_data));
          float v113_data = s1[91];
          float v115_data = ir3[7];
          ir3[7] = (v115_data + (v77_data * v113_data));
          float v118_data = s1[104];
          float v120_data = ir3[8];
          ir3[8] = (v120_data + (v77_data * v118_data));
          float v123_data = s1[117];
          float v125_data = ir3[9];
          ir3[9] = (v125_data + (v77_data * v123_data));
          float v128_data = s1[130];
          float v130_data = ir3[10];
          ir3[10] = (v130_data + (v77_data * v128_data));
          float v133_data = s1[143];
          float v135_data = ir3[11];
          ir3[11] = (v135_data + (v77_data * v133_data));
          float v138_data = s1[156];
          float v140_data = ir3[12];
          ir3[12] = (v140_data + (v77_data * v138_data));
          float v142_data = r2[1];
          float v143_data = s1[1];
          float v145_data = ir3[0];
          ir3[0] = (v145_data + (v142_data * v143_data));
          float v148_data = s1[14];
          float v150_data = ir3[1];
          ir3[1] = (v150_data + (v142_data * v148_data));
          float v153_data = s1[27];
          float v155_data = ir3[2];
          ir3[2] = (v155_data + (v142_data * v153_data));
          float v158_data = s1[40];
          float v160_data = ir3[3];
          ir3[3] = (v160_data + (v142_data * v158_data));
          float v163_data = s1[53];
          float v165_data = ir3[4];
          ir3[4] = (v165_data + (v142_data * v163_data));
          float v168_data = s1[66];
          float v170_data = ir3[5];
          ir3[5] = (v170_data + (v142_data * v168_data));
          float v173_data = s1[79];
          float v175_data = ir3[6];
          ir3[6] = (v175_data + (v142_data * v173_data));
          float v178_data = s1[92];
          float v180_data = ir3[7];
          ir3[7] = (v180_data + (v142_data * v178_data));
          float v183_data = s1[105];
          float v185_data = ir3[8];
          ir3[8] = (v185_data + (v142_data * v183_data));
          float v188_data = s1[118];
          float v190_data = ir3[9];
          ir3[9] = (v190_data + (v142_data * v188_data));
          float v193_data = s1[131];
          float v195_data = ir3[10];
          ir3[10] = (v195_data + (v142_data * v193_data));
          float v198_data = s1[144];
          float v200_data = ir3[11];
          ir3[11] = (v200_data + (v142_data * v198_data));
          float v203_data = s1[157];
          float v205_data = ir3[12];
          ir3[12] = (v205_data + (v142_data * v203_data));
          float v207_data = r2[2];
          float v208_data = s1[2];
          float v210_data = ir3[0];
          ir3[0] = (v210_data + (v207_data * v208_data));
          float v213_data = s1[15];
          float v215_data = ir3[1];
          ir3[1] = (v215_data + (v207_data * v213_data));
          float v218_data = s1[28];
          float v220_data = ir3[2];
          ir3[2] = (v220_data + (v207_data * v218_data));
          float v223_data = s1[41];
          float v225_data = ir3[3];
          ir3[3] = (v225_data + (v207_data * v223_data));
          float v228_data = s1[54];
          float v230_data = ir3[4];
          ir3[4] = (v230_data + (v207_data * v228_data));
          float v233_data = s1[67];
          float v235_data = ir3[5];
          ir3[5] = (v235_data + (v207_data * v233_data));
          float v238_data = s1[80];
          float v240_data = ir3[6];
          ir3[6] = (v240_data + (v207_data * v238_data));
          float v243_data = s1[93];
          float v245_data = ir3[7];
          ir3[7] = (v245_data + (v207_data * v243_data));
          float v248_data = s1[106];
          float v250_data = ir3[8];
          ir3[8] = (v250_data + (v207_data * v248_data));
          float v253_data = s1[119];
          float v255_data = ir3[9];
          ir3[9] = (v255_data + (v207_data * v253_data));
          float v258_data = s1[132];
          float v260_data = ir3[10];
          ir3[10] = (v260_data + (v207_data * v258_data));
          float v263_data = s1[145];
          float v265_data = ir3[11];
          ir3[11] = (v265_data + (v207_data * v263_data));
          float v268_data = s1[158];
          float v270_data = ir3[12];
          ir3[12] = (v270_data + (v207_data * v268_data));
          float v272_data = r2[3];
          float v273_data = s1[3];
          float v275_data = ir3[0];
          ir3[0] = (v275_data + (v272_data * v273_data));
          float v278_data = s1[16];
          float v280_data = ir3[1];
          ir3[1] = (v280_data + (v272_data * v278_data));
          float v283_data = s1[29];
          float v285_data = ir3[2];
          ir3[2] = (v285_data + (v272_data * v283_data));
          float v288_data = s1[42];
          float v290_data = ir3[3];
          ir3[3] = (v290_data + (v272_data * v288_data));
          float v293_data = s1[55];
          float v295_data = ir3[4];
          ir3[4] = (v295_data + (v272_data * v293_data));
          float v298_data = s1[68];
          float v300_data = ir3[5];
          ir3[5] = (v300_data + (v272_data * v298_data));
          float v303_data = s1[81];
          float v305_data = ir3[6];
          ir3[6] = (v305_data + (v272_data * v303_data));
          float v308_data = s1[94];
          float v310_data = ir3[7];
          ir3[7] = (v310_data + (v272_data * v308_data));
          float v313_data = s1[107];
          float v315_data = ir3[8];
          ir3[8] = (v315_data + (v272_data * v313_data));
          float v318_data = s1[120];
          float v320_data = ir3[9];
          ir3[9] = (v320_data + (v272_data * v318_data));
          float v323_data = s1[133];
          float v325_data = ir3[10];
          ir3[10] = (v325_data + (v272_data * v323_data));
          float v328_data = s1[146];
          float v330_data = ir3[11];
          ir3[11] = (v330_data + (v272_data * v328_data));
          float v333_data = s1[159];
          float v335_data = ir3[12];
          ir3[12] = (v335_data + (v272_data * v333_data));
          float v337_data = r2[4];
          float v338_data = s1[4];
          float v340_data = ir3[0];
          ir3[0] = (v340_data + (v337_data * v338_data));
          float v343_data = s1[17];
          float v345_data = ir3[1];
          ir3[1] = (v345_data + (v337_data * v343_data));
          float v348_data = s1[30];
          float v350_data = ir3[2];
          ir3[2] = (v350_data + (v337_data * v348_data));
          float v353_data = s1[43];
          float v355_data = ir3[3];
          ir3[3] = (v355_data + (v337_data * v353_data));
          float v358_data = s1[56];
          float v360_data = ir3[4];
          ir3[4] = (v360_data + (v337_data * v358_data));
          float v363_data = s1[69];
          float v365_data = ir3[5];
          ir3[5] = (v365_data + (v337_data * v363_data));
          float v368_data = s1[82];
          float v370_data = ir3[6];
          ir3[6] = (v370_data + (v337_data * v368_data));
          float v373_data = s1[95];
          float v375_data = ir3[7];
          ir3[7] = (v375_data + (v337_data * v373_data));
          float v378_data = s1[108];
          float v380_data = ir3[8];
          ir3[8] = (v380_data + (v337_data * v378_data));
          float v383_data = s1[121];
          float v385_data = ir3[9];
          ir3[9] = (v385_data + (v337_data * v383_data));
          float v388_data = s1[134];
          float v390_data = ir3[10];
          ir3[10] = (v390_data + (v337_data * v388_data));
          float v393_data = s1[147];
          float v395_data = ir3[11];
          ir3[11] = (v395_data + (v337_data * v393_data));
          float v398_data = s1[160];
          float v400_data = ir3[12];
          ir3[12] = (v400_data + (v337_data * v398_data));
          float v402_data = r2[5];
          float v403_data = s1[5];
          float v405_data = ir3[0];
          ir3[0] = (v405_data + (v402_data * v403_data));
          float v408_data = s1[18];
          float v410_data = ir3[1];
          ir3[1] = (v410_data + (v402_data * v408_data));
          float v413_data = s1[31];
          float v415_data = ir3[2];
          ir3[2] = (v415_data + (v402_data * v413_data));
          float v418_data = s1[44];
          float v420_data = ir3[3];
          ir3[3] = (v420_data + (v402_data * v418_data));
          float v423_data = s1[57];
          float v425_data = ir3[4];
          ir3[4] = (v425_data + (v402_data * v423_data));
          float v428_data = s1[70];
          float v430_data = ir3[5];
          ir3[5] = (v430_data + (v402_data * v428_data));
          float v433_data = s1[83];
          float v435_data = ir3[6];
          ir3[6] = (v435_data + (v402_data * v433_data));
          float v438_data = s1[96];
          float v440_data = ir3[7];
          ir3[7] = (v440_data + (v402_data * v438_data));
          float v443_data = s1[109];
          float v445_data = ir3[8];
          ir3[8] = (v445_data + (v402_data * v443_data));
          float v448_data = s1[122];
          float v450_data = ir3[9];
          ir3[9] = (v450_data + (v402_data * v448_data));
          float v453_data = s1[135];
          float v455_data = ir3[10];
          ir3[10] = (v455_data + (v402_data * v453_data));
          float v458_data = s1[148];
          float v460_data = ir3[11];
          ir3[11] = (v460_data + (v402_data * v458_data));
          float v463_data = s1[161];
          float v465_data = ir3[12];
          ir3[12] = (v465_data + (v402_data * v463_data));
          float v467_data = r2[6];
          float v468_data = s1[6];
          float v470_data = ir3[0];
          ir3[0] = (v470_data + (v467_data * v468_data));
          float v473_data = s1[19];
          float v475_data = ir3[1];
          ir3[1] = (v475_data + (v467_data * v473_data));
          float v478_data = s1[32];
          float v480_data = ir3[2];
          ir3[2] = (v480_data + (v467_data * v478_data));
          float v483_data = s1[45];
          float v485_data = ir3[3];
          ir3[3] = (v485_data + (v467_data * v483_data));
          float v488_data = s1[58];
          float v490_data = ir3[4];
          ir3[4] = (v490_data + (v467_data * v488_data));
          float v493_data = s1[71];
          float v495_data = ir3[5];
          ir3[5] = (v495_data + (v467_data * v493_data));
          float v498_data = s1[84];
          float v500_data = ir3[6];
          ir3[6] = (v500_data + (v467_data * v498_data));
          float v503_data = s1[97];
          float v505_data = ir3[7];
          ir3[7] = (v505_data + (v467_data * v503_data));
          float v508_data = s1[110];
          float v510_data = ir3[8];
          ir3[8] = (v510_data + (v467_data * v508_data));
          float v513_data = s1[123];
          float v515_data = ir3[9];
          ir3[9] = (v515_data + (v467_data * v513_data));
          float v518_data = s1[136];
          float v520_data = ir3[10];
          ir3[10] = (v520_data + (v467_data * v518_data));
          float v523_data = s1[149];
          float v525_data = ir3[11];
          ir3[11] = (v525_data + (v467_data * v523_data));
          float v528_data = s1[162];
          float v530_data = ir3[12];
          ir3[12] = (v530_data + (v467_data * v528_data));
          float v532_data = r2[7];
          float v533_data = s1[7];
          float v535_data = ir3[0];
          ir3[0] = (v535_data + (v532_data * v533_data));
          float v538_data = s1[20];
          float v540_data = ir3[1];
          ir3[1] = (v540_data + (v532_data * v538_data));
          float v543_data = s1[33];
          float v545_data = ir3[2];
          ir3[2] = (v545_data + (v532_data * v543_data));
          float v548_data = s1[46];
          float v550_data = ir3[3];
          ir3[3] = (v550_data + (v532_data * v548_data));
          float v553_data = s1[59];
          float v555_data = ir3[4];
          ir3[4] = (v555_data + (v532_data * v553_data));
          float v558_data = s1[72];
          float v560_data = ir3[5];
          ir3[5] = (v560_data + (v532_data * v558_data));
          float v563_data = s1[85];
          float v565_data = ir3[6];
          ir3[6] = (v565_data + (v532_data * v563_data));
          float v568_data = s1[98];
          float v570_data = ir3[7];
          ir3[7] = (v570_data + (v532_data * v568_data));
          float v573_data = s1[111];
          float v575_data = ir3[8];
          ir3[8] = (v575_data + (v532_data * v573_data));
          float v578_data = s1[124];
          float v580_data = ir3[9];
          ir3[9] = (v580_data + (v532_data * v578_data));
          float v583_data = s1[137];
          float v585_data = ir3[10];
          ir3[10] = (v585_data + (v532_data * v583_data));
          float v588_data = s1[150];
          float v590_data = ir3[11];
          ir3[11] = (v590_data + (v532_data * v588_data));
          float v593_data = s1[163];
          float v595_data = ir3[12];
          ir3[12] = (v595_data + (v532_data * v593_data));
          float v597_data = r2[8];
          float v598_data = s1[8];
          float v600_data = ir3[0];
          ir3[0] = (v600_data + (v597_data * v598_data));
          float v603_data = s1[21];
          float v605_data = ir3[1];
          ir3[1] = (v605_data + (v597_data * v603_data));
          float v608_data = s1[34];
          float v610_data = ir3[2];
          ir3[2] = (v610_data + (v597_data * v608_data));
          float v613_data = s1[47];
          float v615_data = ir3[3];
          ir3[3] = (v615_data + (v597_data * v613_data));
          float v618_data = s1[60];
          float v620_data = ir3[4];
          ir3[4] = (v620_data + (v597_data * v618_data));
          float v623_data = s1[73];
          float v625_data = ir3[5];
          ir3[5] = (v625_data + (v597_data * v623_data));
          float v628_data = s1[86];
          float v630_data = ir3[6];
          ir3[6] = (v630_data + (v597_data * v628_data));
          float v633_data = s1[99];
          float v635_data = ir3[7];
          ir3[7] = (v635_data + (v597_data * v633_data));
          float v638_data = s1[112];
          float v640_data = ir3[8];
          ir3[8] = (v640_data + (v597_data * v638_data));
          float v643_data = s1[125];
          float v645_data = ir3[9];
          ir3[9] = (v645_data + (v597_data * v643_data));
          float v648_data = s1[138];
          float v650_data = ir3[10];
          ir3[10] = (v650_data + (v597_data * v648_data));
          float v653_data = s1[151];
          float v655_data = ir3[11];
          ir3[11] = (v655_data + (v597_data * v653_data));
          float v658_data = s1[164];
          float v660_data = ir3[12];
          ir3[12] = (v660_data + (v597_data * v658_data));
          float v662_data = r2[9];
          float v663_data = s1[9];
          float v665_data = ir3[0];
          ir3[0] = (v665_data + (v662_data * v663_data));
          float v668_data = s1[22];
          float v670_data = ir3[1];
          ir3[1] = (v670_data + (v662_data * v668_data));
          float v673_data = s1[35];
          float v675_data = ir3[2];
          ir3[2] = (v675_data + (v662_data * v673_data));
          float v678_data = s1[48];
          float v680_data = ir3[3];
          ir3[3] = (v680_data + (v662_data * v678_data));
          float v683_data = s1[61];
          float v685_data = ir3[4];
          ir3[4] = (v685_data + (v662_data * v683_data));
          float v688_data = s1[74];
          float v690_data = ir3[5];
          ir3[5] = (v690_data + (v662_data * v688_data));
          float v693_data = s1[87];
          float v695_data = ir3[6];
          ir3[6] = (v695_data + (v662_data * v693_data));
          float v698_data = s1[100];
          float v700_data = ir3[7];
          ir3[7] = (v700_data + (v662_data * v698_data));
          float v703_data = s1[113];
          float v705_data = ir3[8];
          ir3[8] = (v705_data + (v662_data * v703_data));
          float v708_data = s1[126];
          float v710_data = ir3[9];
          ir3[9] = (v710_data + (v662_data * v708_data));
          float v713_data = s1[139];
          float v715_data = ir3[10];
          ir3[10] = (v715_data + (v662_data * v713_data));
          float v718_data = s1[152];
          float v720_data = ir3[11];
          ir3[11] = (v720_data + (v662_data * v718_data));
          float v723_data = s1[165];
          float v725_data = ir3[12];
          ir3[12] = (v725_data + (v662_data * v723_data));
          float v727_data = r2[10];
          float v728_data = s1[10];
          float v730_data = ir3[0];
          ir3[0] = (v730_data + (v727_data * v728_data));
          float v733_data = s1[23];
          float v735_data = ir3[1];
          ir3[1] = (v735_data + (v727_data * v733_data));
          float v738_data = s1[36];
          float v740_data = ir3[2];
          ir3[2] = (v740_data + (v727_data * v738_data));
          float v743_data = s1[49];
          float v745_data = ir3[3];
          ir3[3] = (v745_data + (v727_data * v743_data));
          float v748_data = s1[62];
          float v750_data = ir3[4];
          ir3[4] = (v750_data + (v727_data * v748_data));
          float v753_data = s1[75];
          float v755_data = ir3[5];
          ir3[5] = (v755_data + (v727_data * v753_data));
          float v758_data = s1[88];
          float v760_data = ir3[6];
          ir3[6] = (v760_data + (v727_data * v758_data));
          float v763_data = s1[101];
          float v765_data = ir3[7];
          ir3[7] = (v765_data + (v727_data * v763_data));
          float v768_data = s1[114];
          float v770_data = ir3[8];
          ir3[8] = (v770_data + (v727_data * v768_data));
          float v773_data = s1[127];
          float v775_data = ir3[9];
          ir3[9] = (v775_data + (v727_data * v773_data));
          float v778_data = s1[140];
          float v780_data = ir3[10];
          ir3[10] = (v780_data + (v727_data * v778_data));
          float v783_data = s1[153];
          float v785_data = ir3[11];
          ir3[11] = (v785_data + (v727_data * v783_data));
          float v788_data = s1[166];
          float v790_data = ir3[12];
          ir3[12] = (v790_data + (v727_data * v788_data));
          float v792_data = r2[11];
          float v793_data = s1[11];
          float v795_data = ir3[0];
          ir3[0] = (v795_data + (v792_data * v793_data));
          float v798_data = s1[24];
          float v800_data = ir3[1];
          ir3[1] = (v800_data + (v792_data * v798_data));
          float v803_data = s1[37];
          float v805_data = ir3[2];
          ir3[2] = (v805_data + (v792_data * v803_data));
          float v808_data = s1[50];
          float v810_data = ir3[3];
          ir3[3] = (v810_data + (v792_data * v808_data));
          float v813_data = s1[63];
          float v815_data = ir3[4];
          ir3[4] = (v815_data + (v792_data * v813_data));
          float v818_data = s1[76];
          float v820_data = ir3[5];
          ir3[5] = (v820_data + (v792_data * v818_data));
          float v823_data = s1[89];
          float v825_data = ir3[6];
          ir3[6] = (v825_data + (v792_data * v823_data));
          float v828_data = s1[102];
          float v830_data = ir3[7];
          ir3[7] = (v830_data + (v792_data * v828_data));
          float v833_data = s1[115];
          float v835_data = ir3[8];
          ir3[8] = (v835_data + (v792_data * v833_data));
          float v838_data = s1[128];
          float v840_data = ir3[9];
          ir3[9] = (v840_data + (v792_data * v838_data));
          float v843_data = s1[141];
          float v845_data = ir3[10];
          ir3[10] = (v845_data + (v792_data * v843_data));
          float v848_data = s1[154];
          float v850_data = ir3[11];
          ir3[11] = (v850_data + (v792_data * v848_data));
          float v853_data = s1[167];
          float v855_data = ir3[12];
          ir3[12] = (v855_data + (v792_data * v853_data));
          float v857_data = r2[12];
          float v858_data = s1[12];
          float v860_data = ir3[0];
          ir3[0] = (v860_data + (v857_data * v858_data));
          float v863_data = s1[25];
          float v865_data = ir3[1];
          ir3[1] = (v865_data + (v857_data * v863_data));
          float v868_data = s1[38];
          float v870_data = ir3[2];
          ir3[2] = (v870_data + (v857_data * v868_data));
          float v873_data = s1[51];
          float v875_data = ir3[3];
          ir3[3] = (v875_data + (v857_data * v873_data));
          float v878_data = s1[64];
          float v880_data = ir3[4];
          ir3[4] = (v880_data + (v857_data * v878_data));
          float v883_data = s1[77];
          float v885_data = ir3[5];
          ir3[5] = (v885_data + (v857_data * v883_data));
          float v888_data = s1[90];
          float v890_data = ir3[6];
          ir3[6] = (v890_data + (v857_data * v888_data));
          float v893_data = s1[103];
          float v895_data = ir3[7];
          ir3[7] = (v895_data + (v857_data * v893_data));
          float v898_data = s1[116];
          float v900_data = ir3[8];
          ir3[8] = (v900_data + (v857_data * v898_data));
          float v903_data = s1[129];
          float v905_data = ir3[9];
          ir3[9] = (v905_data + (v857_data * v903_data));
          float v908_data = s1[142];
          float v910_data = ir3[10];
          ir3[10] = (v910_data + (v857_data * v908_data));
          float v913_data = s1[155];
          float v915_data = ir3[11];
          ir3[11] = (v915_data + (v857_data * v913_data));
          float v918_data = s1[168];
          float v920_data = ir3[12];
          ir3[12] = (v920_data + (v857_data * v918_data));
          #pragma unroll
          for (int32_t v922_n0 = 0; v922_n0 < 1; ++v922_n0) {
            #pragma unroll
            for (int32_t v923_n1 = 0; v923_n1 < 13; ++v923_n1) {
              int32_t v924_a = v922_n0 + v923_n1;
              float v925_data = ir3[v924_a];
              r3[v924_a] = v925_data;
            }
          }
          // glb_m3 = store{r>g}(r3);
          #pragma unroll
          for (int32_t v926_i0 = 0; v926_i0 < 1; ++v926_i0) {
            int32_t v931_lead = v22_lead + (v926_i0 * 32);
            #pragma unroll
            for (int32_t v927_i1 = 0; v927_i1 < 13; ++v927_i1) {
              float v929_data = r3[(v926_i0 + v927_i1)];
              glb_m3[(v931_lead + (v927_i1 * 32))] = v929_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

