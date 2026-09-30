// === base name ===
kernel_34307272418c1d12

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_34307272418c1d12 = {{8, 16, 1}, 8, 8, 1, 16, 4608, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_34307272418c1d12(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_34307272418c1d12(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_34307272418c1d12(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_34307272418c1d12, block.x * block.y * block.z, 1152 * sizeof(float));
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
void launcher_kernel_34307272418c1d12(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_34307272418c1d12(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_34307272418c1d12, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_34307272418c1d12<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_34307272418c1d12(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
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
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1152}],"shared_bytes":4608,"shared_elements":1152,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A1","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"A2","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m4","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[72 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s2 = &localShrMem0[0];
      for (size_t v6_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v6_batchId0 < numElements0; v6_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v7_ahead1 = v6_batchId0 + (gridDim.x * blockDim.y);
        size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 64 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v6_batchId0 * 64 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v6_batchId0 * 64 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[v6_batchId0 * 64 + 0 + m3_extraOffset];
          float *const __restrict__ glb_m4 = &m4[v6_batchId0 * 64 + 0 + m4_extraOffset];
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
          #pragma unroll
          for (int32_t i = 0; i < 8; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 8], &glb_m1[0 + 0 + 1 * threadIdx.x + i * 8], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m0););
          float r2[8]{};
          // r2 = load{g>r}(glb_m2);
          #pragma unroll
          for (int32_t v33_i0 = 0; v33_i0 < 1; ++v33_i0) {
            int32_t v36_lead = v22_lead + (v33_i0 * 8);
            #pragma unroll
            for (int32_t v34_i1 = 0; v34_i1 < 8; ++v34_i1) {
              float v39_data = __ldcg(&glb_m2[(v36_lead + (v34_i1 * 8))]);
              r2[(v33_i0 + v34_i1)] = v39_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          // r1 = +(r0 * s0) + None
          // [(0, 8), (0, 8)] [(0, 8)]
          float v42_data = r0[0];
          float v43_data = s0[0];
          float v45_data = r1[0];
          r1[0] = (v45_data + (v42_data * v43_data));
          float v48_data = s0[8];
          float v50_data = r1[1];
          r1[1] = (v50_data + (v42_data * v48_data));
          float v53_data = s0[16];
          float v55_data = r1[2];
          r1[2] = (v55_data + (v42_data * v53_data));
          float v58_data = s0[24];
          float v60_data = r1[3];
          r1[3] = (v60_data + (v42_data * v58_data));
          float v63_data = s0[32];
          float v65_data = r1[4];
          r1[4] = (v65_data + (v42_data * v63_data));
          float v68_data = s0[40];
          float v70_data = r1[5];
          r1[5] = (v70_data + (v42_data * v68_data));
          float v73_data = s0[48];
          float v75_data = r1[6];
          r1[6] = (v75_data + (v42_data * v73_data));
          float v78_data = s0[56];
          float v80_data = r1[7];
          r1[7] = (v80_data + (v42_data * v78_data));
          float v82_data = r0[1];
          float v83_data = s0[1];
          float v85_data = r1[0];
          r1[0] = (v85_data + (v82_data * v83_data));
          float v88_data = s0[9];
          float v90_data = r1[1];
          r1[1] = (v90_data + (v82_data * v88_data));
          float v93_data = s0[17];
          float v95_data = r1[2];
          r1[2] = (v95_data + (v82_data * v93_data));
          float v98_data = s0[25];
          float v100_data = r1[3];
          r1[3] = (v100_data + (v82_data * v98_data));
          float v103_data = s0[33];
          float v105_data = r1[4];
          r1[4] = (v105_data + (v82_data * v103_data));
          float v108_data = s0[41];
          float v110_data = r1[5];
          r1[5] = (v110_data + (v82_data * v108_data));
          float v113_data = s0[49];
          float v115_data = r1[6];
          r1[6] = (v115_data + (v82_data * v113_data));
          float v118_data = s0[57];
          float v120_data = r1[7];
          r1[7] = (v120_data + (v82_data * v118_data));
          float v122_data = r0[2];
          float v123_data = s0[2];
          float v125_data = r1[0];
          r1[0] = (v125_data + (v122_data * v123_data));
          float v128_data = s0[10];
          float v130_data = r1[1];
          r1[1] = (v130_data + (v122_data * v128_data));
          float v133_data = s0[18];
          float v135_data = r1[2];
          r1[2] = (v135_data + (v122_data * v133_data));
          float v138_data = s0[26];
          float v140_data = r1[3];
          r1[3] = (v140_data + (v122_data * v138_data));
          float v143_data = s0[34];
          float v145_data = r1[4];
          r1[4] = (v145_data + (v122_data * v143_data));
          float v148_data = s0[42];
          float v150_data = r1[5];
          r1[5] = (v150_data + (v122_data * v148_data));
          float v153_data = s0[50];
          float v155_data = r1[6];
          r1[6] = (v155_data + (v122_data * v153_data));
          float v158_data = s0[58];
          float v160_data = r1[7];
          r1[7] = (v160_data + (v122_data * v158_data));
          float v162_data = r0[3];
          float v163_data = s0[3];
          float v165_data = r1[0];
          r1[0] = (v165_data + (v162_data * v163_data));
          float v168_data = s0[11];
          float v170_data = r1[1];
          r1[1] = (v170_data + (v162_data * v168_data));
          float v173_data = s0[19];
          float v175_data = r1[2];
          r1[2] = (v175_data + (v162_data * v173_data));
          float v178_data = s0[27];
          float v180_data = r1[3];
          r1[3] = (v180_data + (v162_data * v178_data));
          float v183_data = s0[35];
          float v185_data = r1[4];
          r1[4] = (v185_data + (v162_data * v183_data));
          float v188_data = s0[43];
          float v190_data = r1[5];
          r1[5] = (v190_data + (v162_data * v188_data));
          float v193_data = s0[51];
          float v195_data = r1[6];
          r1[6] = (v195_data + (v162_data * v193_data));
          float v198_data = s0[59];
          float v200_data = r1[7];
          r1[7] = (v200_data + (v162_data * v198_data));
          float v202_data = r0[4];
          float v203_data = s0[4];
          float v205_data = r1[0];
          r1[0] = (v205_data + (v202_data * v203_data));
          float v208_data = s0[12];
          float v210_data = r1[1];
          r1[1] = (v210_data + (v202_data * v208_data));
          float v213_data = s0[20];
          float v215_data = r1[2];
          r1[2] = (v215_data + (v202_data * v213_data));
          float v218_data = s0[28];
          float v220_data = r1[3];
          r1[3] = (v220_data + (v202_data * v218_data));
          float v223_data = s0[36];
          float v225_data = r1[4];
          r1[4] = (v225_data + (v202_data * v223_data));
          float v228_data = s0[44];
          float v230_data = r1[5];
          r1[5] = (v230_data + (v202_data * v228_data));
          float v233_data = s0[52];
          float v235_data = r1[6];
          r1[6] = (v235_data + (v202_data * v233_data));
          float v238_data = s0[60];
          float v240_data = r1[7];
          r1[7] = (v240_data + (v202_data * v238_data));
          float v242_data = r0[5];
          float v243_data = s0[5];
          float v245_data = r1[0];
          r1[0] = (v245_data + (v242_data * v243_data));
          float v248_data = s0[13];
          float v250_data = r1[1];
          r1[1] = (v250_data + (v242_data * v248_data));
          float v253_data = s0[21];
          float v255_data = r1[2];
          r1[2] = (v255_data + (v242_data * v253_data));
          float v258_data = s0[29];
          float v260_data = r1[3];
          r1[3] = (v260_data + (v242_data * v258_data));
          float v263_data = s0[37];
          float v265_data = r1[4];
          r1[4] = (v265_data + (v242_data * v263_data));
          float v268_data = s0[45];
          float v270_data = r1[5];
          r1[5] = (v270_data + (v242_data * v268_data));
          float v273_data = s0[53];
          float v275_data = r1[6];
          r1[6] = (v275_data + (v242_data * v273_data));
          float v278_data = s0[61];
          float v280_data = r1[7];
          r1[7] = (v280_data + (v242_data * v278_data));
          float v282_data = r0[6];
          float v283_data = s0[6];
          float v285_data = r1[0];
          r1[0] = (v285_data + (v282_data * v283_data));
          float v288_data = s0[14];
          float v290_data = r1[1];
          r1[1] = (v290_data + (v282_data * v288_data));
          float v293_data = s0[22];
          float v295_data = r1[2];
          r1[2] = (v295_data + (v282_data * v293_data));
          float v298_data = s0[30];
          float v300_data = r1[3];
          r1[3] = (v300_data + (v282_data * v298_data));
          float v303_data = s0[38];
          float v305_data = r1[4];
          r1[4] = (v305_data + (v282_data * v303_data));
          float v308_data = s0[46];
          float v310_data = r1[5];
          r1[5] = (v310_data + (v282_data * v308_data));
          float v313_data = s0[54];
          float v315_data = r1[6];
          r1[6] = (v315_data + (v282_data * v313_data));
          float v318_data = s0[62];
          float v320_data = r1[7];
          r1[7] = (v320_data + (v282_data * v318_data));
          float v322_data = r0[7];
          float v323_data = s0[7];
          float v325_data = r1[0];
          r1[0] = (v325_data + (v322_data * v323_data));
          float v328_data = s0[15];
          float v330_data = r1[1];
          r1[1] = (v330_data + (v322_data * v328_data));
          float v333_data = s0[23];
          float v335_data = r1[2];
          r1[2] = (v335_data + (v322_data * v333_data));
          float v338_data = s0[31];
          float v340_data = r1[3];
          r1[3] = (v340_data + (v322_data * v338_data));
          float v343_data = s0[39];
          float v345_data = r1[4];
          r1[4] = (v345_data + (v322_data * v343_data));
          float v348_data = s0[47];
          float v350_data = r1[5];
          r1[5] = (v350_data + (v322_data * v348_data));
          float v353_data = s0[55];
          float v355_data = r1[6];
          r1[6] = (v355_data + (v322_data * v353_data));
          float v358_data = s0[63];
          float v360_data = r1[7];
          r1[7] = (v360_data + (v322_data * v358_data));
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
          float v365_data = r2[0];
          float v366_data = s2[0];
          float v368_data = ir3[0];
          ir3[0] = (v368_data + (v365_data * v366_data));
          float v371_data = s2[8];
          float v373_data = ir3[1];
          ir3[1] = (v373_data + (v365_data * v371_data));
          float v376_data = s2[16];
          float v378_data = ir3[2];
          ir3[2] = (v378_data + (v365_data * v376_data));
          float v381_data = s2[24];
          float v383_data = ir3[3];
          ir3[3] = (v383_data + (v365_data * v381_data));
          float v386_data = s2[32];
          float v388_data = ir3[4];
          ir3[4] = (v388_data + (v365_data * v386_data));
          float v391_data = s2[40];
          float v393_data = ir3[5];
          ir3[5] = (v393_data + (v365_data * v391_data));
          float v396_data = s2[48];
          float v398_data = ir3[6];
          ir3[6] = (v398_data + (v365_data * v396_data));
          float v401_data = s2[56];
          float v403_data = ir3[7];
          ir3[7] = (v403_data + (v365_data * v401_data));
          float v405_data = r2[1];
          float v406_data = s2[1];
          float v408_data = ir3[0];
          ir3[0] = (v408_data + (v405_data * v406_data));
          float v411_data = s2[9];
          float v413_data = ir3[1];
          ir3[1] = (v413_data + (v405_data * v411_data));
          float v416_data = s2[17];
          float v418_data = ir3[2];
          ir3[2] = (v418_data + (v405_data * v416_data));
          float v421_data = s2[25];
          float v423_data = ir3[3];
          ir3[3] = (v423_data + (v405_data * v421_data));
          float v426_data = s2[33];
          float v428_data = ir3[4];
          ir3[4] = (v428_data + (v405_data * v426_data));
          float v431_data = s2[41];
          float v433_data = ir3[5];
          ir3[5] = (v433_data + (v405_data * v431_data));
          float v436_data = s2[49];
          float v438_data = ir3[6];
          ir3[6] = (v438_data + (v405_data * v436_data));
          float v441_data = s2[57];
          float v443_data = ir3[7];
          ir3[7] = (v443_data + (v405_data * v441_data));
          float v445_data = r2[2];
          float v446_data = s2[2];
          float v448_data = ir3[0];
          ir3[0] = (v448_data + (v445_data * v446_data));
          float v451_data = s2[10];
          float v453_data = ir3[1];
          ir3[1] = (v453_data + (v445_data * v451_data));
          float v456_data = s2[18];
          float v458_data = ir3[2];
          ir3[2] = (v458_data + (v445_data * v456_data));
          float v461_data = s2[26];
          float v463_data = ir3[3];
          ir3[3] = (v463_data + (v445_data * v461_data));
          float v466_data = s2[34];
          float v468_data = ir3[4];
          ir3[4] = (v468_data + (v445_data * v466_data));
          float v471_data = s2[42];
          float v473_data = ir3[5];
          ir3[5] = (v473_data + (v445_data * v471_data));
          float v476_data = s2[50];
          float v478_data = ir3[6];
          ir3[6] = (v478_data + (v445_data * v476_data));
          float v481_data = s2[58];
          float v483_data = ir3[7];
          ir3[7] = (v483_data + (v445_data * v481_data));
          float v485_data = r2[3];
          float v486_data = s2[3];
          float v488_data = ir3[0];
          ir3[0] = (v488_data + (v485_data * v486_data));
          float v491_data = s2[11];
          float v493_data = ir3[1];
          ir3[1] = (v493_data + (v485_data * v491_data));
          float v496_data = s2[19];
          float v498_data = ir3[2];
          ir3[2] = (v498_data + (v485_data * v496_data));
          float v501_data = s2[27];
          float v503_data = ir3[3];
          ir3[3] = (v503_data + (v485_data * v501_data));
          float v506_data = s2[35];
          float v508_data = ir3[4];
          ir3[4] = (v508_data + (v485_data * v506_data));
          float v511_data = s2[43];
          float v513_data = ir3[5];
          ir3[5] = (v513_data + (v485_data * v511_data));
          float v516_data = s2[51];
          float v518_data = ir3[6];
          ir3[6] = (v518_data + (v485_data * v516_data));
          float v521_data = s2[59];
          float v523_data = ir3[7];
          ir3[7] = (v523_data + (v485_data * v521_data));
          float v525_data = r2[4];
          float v526_data = s2[4];
          float v528_data = ir3[0];
          ir3[0] = (v528_data + (v525_data * v526_data));
          float v531_data = s2[12];
          float v533_data = ir3[1];
          ir3[1] = (v533_data + (v525_data * v531_data));
          float v536_data = s2[20];
          float v538_data = ir3[2];
          ir3[2] = (v538_data + (v525_data * v536_data));
          float v541_data = s2[28];
          float v543_data = ir3[3];
          ir3[3] = (v543_data + (v525_data * v541_data));
          float v546_data = s2[36];
          float v548_data = ir3[4];
          ir3[4] = (v548_data + (v525_data * v546_data));
          float v551_data = s2[44];
          float v553_data = ir3[5];
          ir3[5] = (v553_data + (v525_data * v551_data));
          float v556_data = s2[52];
          float v558_data = ir3[6];
          ir3[6] = (v558_data + (v525_data * v556_data));
          float v561_data = s2[60];
          float v563_data = ir3[7];
          ir3[7] = (v563_data + (v525_data * v561_data));
          float v565_data = r2[5];
          float v566_data = s2[5];
          float v568_data = ir3[0];
          ir3[0] = (v568_data + (v565_data * v566_data));
          float v571_data = s2[13];
          float v573_data = ir3[1];
          ir3[1] = (v573_data + (v565_data * v571_data));
          float v576_data = s2[21];
          float v578_data = ir3[2];
          ir3[2] = (v578_data + (v565_data * v576_data));
          float v581_data = s2[29];
          float v583_data = ir3[3];
          ir3[3] = (v583_data + (v565_data * v581_data));
          float v586_data = s2[37];
          float v588_data = ir3[4];
          ir3[4] = (v588_data + (v565_data * v586_data));
          float v591_data = s2[45];
          float v593_data = ir3[5];
          ir3[5] = (v593_data + (v565_data * v591_data));
          float v596_data = s2[53];
          float v598_data = ir3[6];
          ir3[6] = (v598_data + (v565_data * v596_data));
          float v601_data = s2[61];
          float v603_data = ir3[7];
          ir3[7] = (v603_data + (v565_data * v601_data));
          float v605_data = r2[6];
          float v606_data = s2[6];
          float v608_data = ir3[0];
          ir3[0] = (v608_data + (v605_data * v606_data));
          float v611_data = s2[14];
          float v613_data = ir3[1];
          ir3[1] = (v613_data + (v605_data * v611_data));
          float v616_data = s2[22];
          float v618_data = ir3[2];
          ir3[2] = (v618_data + (v605_data * v616_data));
          float v621_data = s2[30];
          float v623_data = ir3[3];
          ir3[3] = (v623_data + (v605_data * v621_data));
          float v626_data = s2[38];
          float v628_data = ir3[4];
          ir3[4] = (v628_data + (v605_data * v626_data));
          float v631_data = s2[46];
          float v633_data = ir3[5];
          ir3[5] = (v633_data + (v605_data * v631_data));
          float v636_data = s2[54];
          float v638_data = ir3[6];
          ir3[6] = (v638_data + (v605_data * v636_data));
          float v641_data = s2[62];
          float v643_data = ir3[7];
          ir3[7] = (v643_data + (v605_data * v641_data));
          float v645_data = r2[7];
          float v646_data = s2[7];
          float v648_data = ir3[0];
          ir3[0] = (v648_data + (v645_data * v646_data));
          float v651_data = s2[15];
          float v653_data = ir3[1];
          ir3[1] = (v653_data + (v645_data * v651_data));
          float v656_data = s2[23];
          float v658_data = ir3[2];
          ir3[2] = (v658_data + (v645_data * v656_data));
          float v661_data = s2[31];
          float v663_data = ir3[3];
          ir3[3] = (v663_data + (v645_data * v661_data));
          float v666_data = s2[39];
          float v668_data = ir3[4];
          ir3[4] = (v668_data + (v645_data * v666_data));
          float v671_data = s2[47];
          float v673_data = ir3[5];
          ir3[5] = (v673_data + (v645_data * v671_data));
          float v676_data = s2[55];
          float v678_data = ir3[6];
          ir3[6] = (v678_data + (v645_data * v676_data));
          float v681_data = s2[63];
          float v683_data = ir3[7];
          ir3[7] = (v683_data + (v645_data * v681_data));
          // r3 = ir3 + r1
          #pragma unroll
          for (int32_t v685_n0 = 0; v685_n0 < 1; ++v685_n0) {
            #pragma unroll
            for (int32_t v686_n1 = 0; v686_n1 < 8; ++v686_n1) {
              int32_t v687_a = v685_n0 + v686_n1;
              float v688_data = ir3[v687_a];
              float v689_data = r1[v687_a];
              r3[v687_a] = (v689_data + v688_data);
            }
          }
          // glb_m4 = abs(r3)
          #pragma unroll
          for (int32_t v691_k0 = 0; v691_k0 < 1; ++v691_k0) {
            int32_t v697_lead = v22_lead + (v691_k0 * 8);
            #pragma unroll
            for (int32_t v692_k1 = 0; v692_k1 < 8; ++v692_k1) {
              float v694_data = r3[(v691_k0 + v692_k1)];
              glb_m4[(v697_lead + (v692_k1 * 8))] = (fabsf(v694_data));
            }
          }
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
        }
      }
    }
  }
}

