// === base name ===
kernel_c7a09675828f06e9

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_c7a09675828f06e9 = {{16, 8, 1}, 16, 12, 1, 8, 3584, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_c7a09675828f06e9(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_c7a09675828f06e9(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_c7a09675828f06e9(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_c7a09675828f06e9, block.x * block.y * block.z, 896 * sizeof(float));
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
  config.sharedMemBytes = 896 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_c7a09675828f06e9(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_c7a09675828f06e9(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_c7a09675828f06e9, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_c7a09675828f06e9<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_c7a09675828f06e9(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 8 per block = block 16x8x1, 3584 B shared, occupancy grid
    // operands:
    //   m0 32×32(12×6) {0..12}×{0..6} strided
    //   m1 32×32(6×6) {0..6}×{0..6} strided
    //   m2 32×32(12×6) {0..12}×{0..6} strided
    //   m3 32×32(12×12) {0..12}×{0..12} strided
    // operations:
    //   t0[i,j] = m0[i,k] × m1[k,j]
    //   m2[i,j] = m3[i,k] × t0[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":896}],"shared_bytes":3584,"shared_elements":896,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B","bbox":[[0,0],[12,6]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[6,6]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,6]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,6]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,6]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[6,6]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]},{"addressing":"pointer_based","bbox":[[0,0],[12,6]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[112 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[96];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v6_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v6_batchId0 < numElements0; v6_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v7_ahead1 = v6_batchId0 + (gridDim.x * blockDim.y);
        size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 72 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v6_batchId0 * 36 + 0 + m1_extraOffset];
          float *const __restrict__ glb_m2 = &m2[v6_batchId0 * 72 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[v6_batchId0 * 144 + 0 + m3_extraOffset];
          float r0[6]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v21_lead = threadIdx.x % 16;
          bool v22_g = v21_lead < 12;
          if (v22_g) {
            #pragma unroll
            for (int32_t v23_i1 = 0; v23_i1 < 6; ++v23_i1) {
              float v28_data = __ldcg(&glb_m0[(v21_lead + (v23_i1 * 12))]);
              r0[v23_i1] = v28_data;
            }
          }
          // s0 = load{g>s}(glb_m1[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m1[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 16], &glb_m1[0 + 0 + 1 * threadIdx.x + 16], 4);
          if (threadIdx.x < 4) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m1[0 + 0 + 1 * threadIdx.x + 32], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m0););
          float r2[12]{};
          // r2 = load{g>r}(glb_m3);
          if (v22_g) {
            #pragma unroll
            for (int32_t v34_i1 = 0; v34_i1 < 12; ++v34_i1) {
              float v39_data = __ldcg(&glb_m3[(v21_lead + (v34_i1 * 12))]);
              r2[v34_i1] = v39_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[6]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // r1 = +(r0 * s0) + None
          // [(0, 12), (0, 6)] [(0, 6)]
          float v42_data = r0[0];
          float v43_data = s0[0];
          float v45_data = r1[0];
          r1[0] = (v45_data + (v42_data * v43_data));
          float v48_data = s0[6];
          float v50_data = r1[1];
          r1[1] = (v50_data + (v42_data * v48_data));
          float v53_data = s0[12];
          float v55_data = r1[2];
          r1[2] = (v55_data + (v42_data * v53_data));
          float v58_data = s0[18];
          float v60_data = r1[3];
          r1[3] = (v60_data + (v42_data * v58_data));
          float v63_data = s0[24];
          float v65_data = r1[4];
          r1[4] = (v65_data + (v42_data * v63_data));
          float v68_data = s0[30];
          float v70_data = r1[5];
          r1[5] = (v70_data + (v42_data * v68_data));
          float v72_data = r0[1];
          float v73_data = s0[1];
          float v75_data = r1[0];
          r1[0] = (v75_data + (v72_data * v73_data));
          float v78_data = s0[7];
          float v80_data = r1[1];
          r1[1] = (v80_data + (v72_data * v78_data));
          float v83_data = s0[13];
          float v85_data = r1[2];
          r1[2] = (v85_data + (v72_data * v83_data));
          float v88_data = s0[19];
          float v90_data = r1[3];
          r1[3] = (v90_data + (v72_data * v88_data));
          float v93_data = s0[25];
          float v95_data = r1[4];
          r1[4] = (v95_data + (v72_data * v93_data));
          float v98_data = s0[31];
          float v100_data = r1[5];
          r1[5] = (v100_data + (v72_data * v98_data));
          float v102_data = r0[2];
          float v103_data = s0[2];
          float v105_data = r1[0];
          r1[0] = (v105_data + (v102_data * v103_data));
          float v108_data = s0[8];
          float v110_data = r1[1];
          r1[1] = (v110_data + (v102_data * v108_data));
          float v113_data = s0[14];
          float v115_data = r1[2];
          r1[2] = (v115_data + (v102_data * v113_data));
          float v118_data = s0[20];
          float v120_data = r1[3];
          r1[3] = (v120_data + (v102_data * v118_data));
          float v123_data = s0[26];
          float v125_data = r1[4];
          r1[4] = (v125_data + (v102_data * v123_data));
          float v128_data = s0[32];
          float v130_data = r1[5];
          r1[5] = (v130_data + (v102_data * v128_data));
          float v132_data = r0[3];
          float v133_data = s0[3];
          float v135_data = r1[0];
          r1[0] = (v135_data + (v132_data * v133_data));
          float v138_data = s0[9];
          float v140_data = r1[1];
          r1[1] = (v140_data + (v132_data * v138_data));
          float v143_data = s0[15];
          float v145_data = r1[2];
          r1[2] = (v145_data + (v132_data * v143_data));
          float v148_data = s0[21];
          float v150_data = r1[3];
          r1[3] = (v150_data + (v132_data * v148_data));
          float v153_data = s0[27];
          float v155_data = r1[4];
          r1[4] = (v155_data + (v132_data * v153_data));
          float v158_data = s0[33];
          float v160_data = r1[5];
          r1[5] = (v160_data + (v132_data * v158_data));
          float v162_data = r0[4];
          float v163_data = s0[4];
          float v165_data = r1[0];
          r1[0] = (v165_data + (v162_data * v163_data));
          float v168_data = s0[10];
          float v170_data = r1[1];
          r1[1] = (v170_data + (v162_data * v168_data));
          float v173_data = s0[16];
          float v175_data = r1[2];
          r1[2] = (v175_data + (v162_data * v173_data));
          float v178_data = s0[22];
          float v180_data = r1[3];
          r1[3] = (v180_data + (v162_data * v178_data));
          float v183_data = s0[28];
          float v185_data = r1[4];
          r1[4] = (v185_data + (v162_data * v183_data));
          float v188_data = s0[34];
          float v190_data = r1[5];
          r1[5] = (v190_data + (v162_data * v188_data));
          float v192_data = r0[5];
          float v193_data = s0[5];
          float v195_data = r1[0];
          r1[0] = (v195_data + (v192_data * v193_data));
          float v198_data = s0[11];
          float v200_data = r1[1];
          r1[1] = (v200_data + (v192_data * v198_data));
          float v203_data = s0[17];
          float v205_data = r1[2];
          r1[2] = (v205_data + (v192_data * v203_data));
          float v208_data = s0[23];
          float v210_data = r1[3];
          r1[3] = (v210_data + (v192_data * v208_data));
          float v213_data = s0[29];
          float v215_data = r1[4];
          r1[4] = (v215_data + (v192_data * v213_data));
          float v218_data = s0[35];
          float v220_data = r1[5];
          r1[5] = (v220_data + (v192_data * v218_data));
          // wait(r2 = load{g>r}(glb_m3););
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // s1 = store{r>s}(localShrMem0, r1);
          if (v22_g) {
            #pragma unroll
            for (int32_t v222_i1 = 0; v222_i1 < 6; ++v222_i1) {
              float v224_data = r1[v222_i1];
              int32_t v228_a = v21_lead + (v222_i1 * 12);
              s1[(v228_a ^ ((v228_a >> 3) & 7))] = v224_data;
            }
          }
          float r3[6]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // r3 = +(r2 * s1) + None
          // [(0, 12), (0, 6)] [(0, 12)]
          float ir3[6]{};
          float v234_data = r2[0];
          float v235_data = s1[0];
          float v237_data = ir3[0];
          ir3[0] = (v237_data + (v234_data * v235_data));
          float v240_data = s1[13];
          float v242_data = ir3[1];
          ir3[1] = (v242_data + (v234_data * v240_data));
          float v245_data = s1[27];
          float v247_data = ir3[2];
          ir3[2] = (v247_data + (v234_data * v245_data));
          float v250_data = s1[32];
          float v252_data = ir3[3];
          ir3[3] = (v252_data + (v234_data * v250_data));
          float v255_data = s1[54];
          float v257_data = ir3[4];
          ir3[4] = (v257_data + (v234_data * v255_data));
          float v260_data = s1[59];
          float v262_data = ir3[5];
          ir3[5] = (v262_data + (v234_data * v260_data));
          float v264_data = r2[1];
          float v265_data = s1[1];
          float v267_data = ir3[0];
          ir3[0] = (v267_data + (v264_data * v265_data));
          float v270_data = s1[12];
          float v272_data = ir3[1];
          ir3[1] = (v272_data + (v264_data * v270_data));
          float v275_data = s1[26];
          float v277_data = ir3[2];
          ir3[2] = (v277_data + (v264_data * v275_data));
          float v280_data = s1[33];
          float v282_data = ir3[3];
          ir3[3] = (v282_data + (v264_data * v280_data));
          float v285_data = s1[55];
          float v287_data = ir3[4];
          ir3[4] = (v287_data + (v264_data * v285_data));
          float v290_data = s1[58];
          float v292_data = ir3[5];
          ir3[5] = (v292_data + (v264_data * v290_data));
          float v294_data = r2[2];
          float v295_data = s1[2];
          float v297_data = ir3[0];
          ir3[0] = (v297_data + (v294_data * v295_data));
          float v300_data = s1[15];
          float v302_data = ir3[1];
          ir3[1] = (v302_data + (v294_data * v300_data));
          float v305_data = s1[25];
          float v307_data = ir3[2];
          ir3[2] = (v307_data + (v294_data * v305_data));
          float v310_data = s1[34];
          float v312_data = ir3[3];
          ir3[3] = (v312_data + (v294_data * v310_data));
          float v315_data = s1[52];
          float v317_data = ir3[4];
          ir3[4] = (v317_data + (v294_data * v315_data));
          float v320_data = s1[57];
          float v322_data = ir3[5];
          ir3[5] = (v322_data + (v294_data * v320_data));
          float v324_data = r2[3];
          float v325_data = s1[3];
          float v327_data = ir3[0];
          ir3[0] = (v327_data + (v324_data * v325_data));
          float v330_data = s1[14];
          float v332_data = ir3[1];
          ir3[1] = (v332_data + (v324_data * v330_data));
          float v335_data = s1[24];
          float v337_data = ir3[2];
          ir3[2] = (v337_data + (v324_data * v335_data));
          float v340_data = s1[35];
          float v342_data = ir3[3];
          ir3[3] = (v342_data + (v324_data * v340_data));
          float v345_data = s1[53];
          float v347_data = ir3[4];
          ir3[4] = (v347_data + (v324_data * v345_data));
          float v350_data = s1[56];
          float v352_data = ir3[5];
          ir3[5] = (v352_data + (v324_data * v350_data));
          float v354_data = r2[4];
          float v355_data = s1[4];
          float v357_data = ir3[0];
          ir3[0] = (v357_data + (v354_data * v355_data));
          float v360_data = s1[18];
          float v362_data = ir3[1];
          ir3[1] = (v362_data + (v354_data * v360_data));
          float v365_data = s1[31];
          float v367_data = ir3[2];
          ir3[2] = (v367_data + (v354_data * v365_data));
          float v370_data = s1[45];
          float v372_data = ir3[3];
          ir3[3] = (v372_data + (v354_data * v370_data));
          float v375_data = s1[50];
          float v377_data = ir3[4];
          ir3[4] = (v377_data + (v354_data * v375_data));
          float v380_data = s1[64];
          float v382_data = ir3[5];
          ir3[5] = (v382_data + (v354_data * v380_data));
          float v384_data = r2[5];
          float v385_data = s1[5];
          float v387_data = ir3[0];
          ir3[0] = (v387_data + (v384_data * v385_data));
          float v390_data = s1[19];
          float v392_data = ir3[1];
          ir3[1] = (v392_data + (v384_data * v390_data));
          float v395_data = s1[30];
          float v397_data = ir3[2];
          ir3[2] = (v397_data + (v384_data * v395_data));
          float v400_data = s1[44];
          float v402_data = ir3[3];
          ir3[3] = (v402_data + (v384_data * v400_data));
          float v405_data = s1[51];
          float v407_data = ir3[4];
          ir3[4] = (v407_data + (v384_data * v405_data));
          float v410_data = s1[65];
          float v412_data = ir3[5];
          ir3[5] = (v412_data + (v384_data * v410_data));
          float v414_data = r2[6];
          float v415_data = s1[6];
          float v417_data = ir3[0];
          ir3[0] = (v417_data + (v414_data * v415_data));
          float v420_data = s1[16];
          float v422_data = ir3[1];
          ir3[1] = (v422_data + (v414_data * v420_data));
          float v425_data = s1[29];
          float v427_data = ir3[2];
          ir3[2] = (v427_data + (v414_data * v425_data));
          float v430_data = s1[47];
          float v432_data = ir3[3];
          ir3[3] = (v432_data + (v414_data * v430_data));
          float v435_data = s1[48];
          float v437_data = ir3[4];
          ir3[4] = (v437_data + (v414_data * v435_data));
          float v440_data = s1[66];
          float v442_data = ir3[5];
          ir3[5] = (v442_data + (v414_data * v440_data));
          float v444_data = r2[7];
          float v445_data = s1[7];
          float v447_data = ir3[0];
          ir3[0] = (v447_data + (v444_data * v445_data));
          float v450_data = s1[17];
          float v452_data = ir3[1];
          ir3[1] = (v452_data + (v444_data * v450_data));
          float v455_data = s1[28];
          float v457_data = ir3[2];
          ir3[2] = (v457_data + (v444_data * v455_data));
          float v460_data = s1[46];
          float v462_data = ir3[3];
          ir3[3] = (v462_data + (v444_data * v460_data));
          float v465_data = s1[49];
          float v467_data = ir3[4];
          ir3[4] = (v467_data + (v444_data * v465_data));
          float v470_data = s1[67];
          float v472_data = ir3[5];
          ir3[5] = (v472_data + (v444_data * v470_data));
          float v474_data = r2[8];
          float v475_data = s1[9];
          float v477_data = ir3[0];
          ir3[0] = (v477_data + (v474_data * v475_data));
          float v480_data = s1[22];
          float v482_data = ir3[1];
          ir3[1] = (v482_data + (v474_data * v480_data));
          float v485_data = s1[36];
          float v487_data = ir3[2];
          ir3[2] = (v487_data + (v474_data * v485_data));
          float v490_data = s1[41];
          float v492_data = ir3[3];
          ir3[3] = (v492_data + (v474_data * v490_data));
          float v495_data = s1[63];
          float v497_data = ir3[4];
          ir3[4] = (v497_data + (v474_data * v495_data));
          float v500_data = s1[68];
          float v502_data = ir3[5];
          ir3[5] = (v502_data + (v474_data * v500_data));
          float v504_data = r2[9];
          float v505_data = s1[8];
          float v507_data = ir3[0];
          ir3[0] = (v507_data + (v504_data * v505_data));
          float v510_data = s1[23];
          float v512_data = ir3[1];
          ir3[1] = (v512_data + (v504_data * v510_data));
          float v515_data = s1[37];
          float v517_data = ir3[2];
          ir3[2] = (v517_data + (v504_data * v515_data));
          float v520_data = s1[40];
          float v522_data = ir3[3];
          ir3[3] = (v522_data + (v504_data * v520_data));
          float v525_data = s1[62];
          float v527_data = ir3[4];
          ir3[4] = (v527_data + (v504_data * v525_data));
          float v530_data = s1[69];
          float v532_data = ir3[5];
          ir3[5] = (v532_data + (v504_data * v530_data));
          float v534_data = r2[10];
          float v535_data = s1[11];
          float v537_data = ir3[0];
          ir3[0] = (v537_data + (v534_data * v535_data));
          float v540_data = s1[20];
          float v542_data = ir3[1];
          ir3[1] = (v542_data + (v534_data * v540_data));
          float v545_data = s1[38];
          float v547_data = ir3[2];
          ir3[2] = (v547_data + (v534_data * v545_data));
          float v550_data = s1[43];
          float v552_data = ir3[3];
          ir3[3] = (v552_data + (v534_data * v550_data));
          float v555_data = s1[61];
          float v557_data = ir3[4];
          ir3[4] = (v557_data + (v534_data * v555_data));
          float v560_data = s1[70];
          float v562_data = ir3[5];
          ir3[5] = (v562_data + (v534_data * v560_data));
          float v564_data = r2[11];
          float v565_data = s1[10];
          float v567_data = ir3[0];
          ir3[0] = (v567_data + (v564_data * v565_data));
          float v570_data = s1[21];
          float v572_data = ir3[1];
          ir3[1] = (v572_data + (v564_data * v570_data));
          float v575_data = s1[39];
          float v577_data = ir3[2];
          ir3[2] = (v577_data + (v564_data * v575_data));
          float v580_data = s1[42];
          float v582_data = ir3[3];
          ir3[3] = (v582_data + (v564_data * v580_data));
          float v585_data = s1[60];
          float v587_data = ir3[4];
          ir3[4] = (v587_data + (v564_data * v585_data));
          float v590_data = s1[71];
          float v592_data = ir3[5];
          ir3[5] = (v592_data + (v564_data * v590_data));
          if (v22_g) {
            #pragma unroll
            for (int32_t v594_n1 = 0; v594_n1 < 6; ++v594_n1) {
              float v596_data = ir3[v594_n1];
              r3[v594_n1] = v596_data;
            }
          }
          // glb_m2 = store{r>g}(r3);
          if (v22_g) {
            #pragma unroll
            for (int32_t v597_i1 = 0; v597_i1 < 6; ++v597_i1) {
              float v599_data = r3[v597_i1];
              glb_m2[(v21_lead + (v597_i1 * 12))] = v599_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

