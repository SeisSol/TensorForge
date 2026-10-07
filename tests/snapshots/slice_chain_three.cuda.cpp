// === base name ===
kernel_95c6a00227c7c004

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_95c6a00227c7c004 = {{16, 8, 1}, 16, 12, 1, 8, 3584, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_95c6a00227c7c004(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_95c6a00227c7c004(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_95c6a00227c7c004(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_95c6a00227c7c004, block.x * block.y * block.z, 896 * sizeof(float));
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
void launcher_kernel_95c6a00227c7c004(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_95c6a00227c7c004(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_95c6a00227c7c004, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_95c6a00227c7c004<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_95c6a00227c7c004(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
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
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":896}],"shared_bytes":3584,"shared_elements":896,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B","bbox":[[0,0],[12,6]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[6,6]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,6]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,6]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,6]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[6,6]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]},{"addressing":"pointer_based","bbox":[[0,0],[12,6]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[112 * threadIdx.y + 0];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < numElements0; v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 72 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v9_batchId0 * 36 + 0 + m1_extraOffset];
          float *const __restrict__ glb_m2 = &m2[v9_batchId0 * 72 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[v9_batchId0 * 144 + 0 + m3_extraOffset];
          float r0[6]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v24_lead = threadIdx.x % 16;
          bool v25_g = v24_lead < 12;
          if (v25_g) {
            #pragma unroll
            for (int32_t v26_i1 = 0; v26_i1 < 6; ++v26_i1) {
              float v31_data = __ldcg(&glb_m0[(v24_lead + (v26_i1 * 12))]);
              r0[v26_i1] = v31_data;
            }
          }
          // s0 = load{g>s}(glb_m1[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m1[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 16], &glb_m1[0 + 0 + 1 * threadIdx.x + 16], 4);
          if (threadIdx.x < 4) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m1[0 + 0 + 1 * threadIdx.x + 32], 4);
          }
          __pipeline_commit();
          float r2[12]{};
          // r2 = load{g>r}(glb_m3);
          if (v25_g) {
            #pragma unroll
            for (int32_t v218_i1 = 0; v218_i1 < 12; ++v218_i1) {
              float v223_data = __ldcg(&glb_m3[(v24_lead + (v218_i1 * 12))]);
              r2[v218_i1] = v223_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[6]{};
          // r1 = +(r0 * s0) + None
          // [(0, 12), (0, 6)] [(0, 6)]
          float v37_data = r0[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v38_data = s0[0];
          float v40_data = r1[0];
          r1[0] = (v40_data + (v37_data * v38_data));
          float v43_data = s0[6];
          float v45_data = r1[1];
          r1[1] = (v45_data + (v37_data * v43_data));
          float v48_data = s0[12];
          float v50_data = r1[2];
          r1[2] = (v50_data + (v37_data * v48_data));
          float v53_data = s0[18];
          float v55_data = r1[3];
          r1[3] = (v55_data + (v37_data * v53_data));
          float v58_data = s0[24];
          float v60_data = r1[4];
          r1[4] = (v60_data + (v37_data * v58_data));
          float v63_data = s0[30];
          float v65_data = r1[5];
          r1[5] = (v65_data + (v37_data * v63_data));
          float v67_data = r0[1];
          float v68_data = s0[1];
          float v70_data = r1[0];
          r1[0] = (v70_data + (v67_data * v68_data));
          float v73_data = s0[7];
          float v75_data = r1[1];
          r1[1] = (v75_data + (v67_data * v73_data));
          float v78_data = s0[13];
          float v80_data = r1[2];
          r1[2] = (v80_data + (v67_data * v78_data));
          float v83_data = s0[19];
          float v85_data = r1[3];
          r1[3] = (v85_data + (v67_data * v83_data));
          float v88_data = s0[25];
          float v90_data = r1[4];
          r1[4] = (v90_data + (v67_data * v88_data));
          float v93_data = s0[31];
          float v95_data = r1[5];
          r1[5] = (v95_data + (v67_data * v93_data));
          float v97_data = r0[2];
          float v98_data = s0[2];
          float v100_data = r1[0];
          r1[0] = (v100_data + (v97_data * v98_data));
          float v103_data = s0[8];
          float v105_data = r1[1];
          r1[1] = (v105_data + (v97_data * v103_data));
          float v108_data = s0[14];
          float v110_data = r1[2];
          r1[2] = (v110_data + (v97_data * v108_data));
          float v113_data = s0[20];
          float v115_data = r1[3];
          r1[3] = (v115_data + (v97_data * v113_data));
          float v118_data = s0[26];
          float v120_data = r1[4];
          r1[4] = (v120_data + (v97_data * v118_data));
          float v123_data = s0[32];
          float v125_data = r1[5];
          r1[5] = (v125_data + (v97_data * v123_data));
          float v127_data = r0[3];
          float v128_data = s0[3];
          float v130_data = r1[0];
          r1[0] = (v130_data + (v127_data * v128_data));
          float v133_data = s0[9];
          float v135_data = r1[1];
          r1[1] = (v135_data + (v127_data * v133_data));
          float v138_data = s0[15];
          float v140_data = r1[2];
          r1[2] = (v140_data + (v127_data * v138_data));
          float v143_data = s0[21];
          float v145_data = r1[3];
          r1[3] = (v145_data + (v127_data * v143_data));
          float v148_data = s0[27];
          float v150_data = r1[4];
          r1[4] = (v150_data + (v127_data * v148_data));
          float v153_data = s0[33];
          float v155_data = r1[5];
          r1[5] = (v155_data + (v127_data * v153_data));
          float v157_data = r0[4];
          float v158_data = s0[4];
          float v160_data = r1[0];
          r1[0] = (v160_data + (v157_data * v158_data));
          float v163_data = s0[10];
          float v165_data = r1[1];
          r1[1] = (v165_data + (v157_data * v163_data));
          float v168_data = s0[16];
          float v170_data = r1[2];
          r1[2] = (v170_data + (v157_data * v168_data));
          float v173_data = s0[22];
          float v175_data = r1[3];
          r1[3] = (v175_data + (v157_data * v173_data));
          float v178_data = s0[28];
          float v180_data = r1[4];
          r1[4] = (v180_data + (v157_data * v178_data));
          float v183_data = s0[34];
          float v185_data = r1[5];
          r1[5] = (v185_data + (v157_data * v183_data));
          float v187_data = r0[5];
          float v188_data = s0[5];
          float v190_data = r1[0];
          r1[0] = (v190_data + (v187_data * v188_data));
          float v193_data = s0[11];
          float v195_data = r1[1];
          r1[1] = (v195_data + (v187_data * v193_data));
          float v198_data = s0[17];
          float v200_data = r1[2];
          r1[2] = (v200_data + (v187_data * v198_data));
          float v203_data = s0[23];
          float v205_data = r1[3];
          r1[3] = (v205_data + (v187_data * v203_data));
          float v208_data = s0[29];
          float v210_data = r1[4];
          r1[4] = (v210_data + (v187_data * v208_data));
          float v213_data = s0[35];
          float v215_data = r1[5];
          r1[5] = (v215_data + (v187_data * v213_data));
          // s1 = store{r>s}(localShrMem0, r1);
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          if (v25_g) {
            #pragma unroll
            for (int32_t v225_i1 = 0; v225_i1 < 6; ++v225_i1) {
              float v227_data = r1[v225_i1];
              int32_t v231_a = v24_lead + (v225_i1 * 12);
              s1[(v231_a ^ ((v231_a >> 3) & 7))] = v227_data;
            }
          }
          float r3[6]{};
          // ir3 = +(r2 * s1)
          // [(0, 12), (0, 6)] [(0, 12)]
          float ir3[6]{};
          float v237_data = r2[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v238_data = s1[0];
          float v240_data = ir3[0];
          ir3[0] = (v240_data + (v237_data * v238_data));
          float v243_data = s1[13];
          float v245_data = ir3[1];
          ir3[1] = (v245_data + (v237_data * v243_data));
          float v248_data = s1[27];
          float v250_data = ir3[2];
          ir3[2] = (v250_data + (v237_data * v248_data));
          float v253_data = s1[32];
          float v255_data = ir3[3];
          ir3[3] = (v255_data + (v237_data * v253_data));
          float v258_data = s1[54];
          float v260_data = ir3[4];
          ir3[4] = (v260_data + (v237_data * v258_data));
          float v263_data = s1[59];
          float v265_data = ir3[5];
          ir3[5] = (v265_data + (v237_data * v263_data));
          float v267_data = r2[1];
          float v268_data = s1[1];
          float v270_data = ir3[0];
          ir3[0] = (v270_data + (v267_data * v268_data));
          float v273_data = s1[12];
          float v275_data = ir3[1];
          ir3[1] = (v275_data + (v267_data * v273_data));
          float v278_data = s1[26];
          float v280_data = ir3[2];
          ir3[2] = (v280_data + (v267_data * v278_data));
          float v283_data = s1[33];
          float v285_data = ir3[3];
          ir3[3] = (v285_data + (v267_data * v283_data));
          float v288_data = s1[55];
          float v290_data = ir3[4];
          ir3[4] = (v290_data + (v267_data * v288_data));
          float v293_data = s1[58];
          float v295_data = ir3[5];
          ir3[5] = (v295_data + (v267_data * v293_data));
          float v297_data = r2[2];
          float v298_data = s1[2];
          float v300_data = ir3[0];
          ir3[0] = (v300_data + (v297_data * v298_data));
          float v303_data = s1[15];
          float v305_data = ir3[1];
          ir3[1] = (v305_data + (v297_data * v303_data));
          float v308_data = s1[25];
          float v310_data = ir3[2];
          ir3[2] = (v310_data + (v297_data * v308_data));
          float v313_data = s1[34];
          float v315_data = ir3[3];
          ir3[3] = (v315_data + (v297_data * v313_data));
          float v318_data = s1[52];
          float v320_data = ir3[4];
          ir3[4] = (v320_data + (v297_data * v318_data));
          float v323_data = s1[57];
          float v325_data = ir3[5];
          ir3[5] = (v325_data + (v297_data * v323_data));
          float v327_data = r2[3];
          float v328_data = s1[3];
          float v330_data = ir3[0];
          ir3[0] = (v330_data + (v327_data * v328_data));
          float v333_data = s1[14];
          float v335_data = ir3[1];
          ir3[1] = (v335_data + (v327_data * v333_data));
          float v338_data = s1[24];
          float v340_data = ir3[2];
          ir3[2] = (v340_data + (v327_data * v338_data));
          float v343_data = s1[35];
          float v345_data = ir3[3];
          ir3[3] = (v345_data + (v327_data * v343_data));
          float v348_data = s1[53];
          float v350_data = ir3[4];
          ir3[4] = (v350_data + (v327_data * v348_data));
          float v353_data = s1[56];
          float v355_data = ir3[5];
          ir3[5] = (v355_data + (v327_data * v353_data));
          float v357_data = r2[4];
          float v358_data = s1[4];
          float v360_data = ir3[0];
          ir3[0] = (v360_data + (v357_data * v358_data));
          float v363_data = s1[18];
          float v365_data = ir3[1];
          ir3[1] = (v365_data + (v357_data * v363_data));
          float v368_data = s1[31];
          float v370_data = ir3[2];
          ir3[2] = (v370_data + (v357_data * v368_data));
          float v373_data = s1[45];
          float v375_data = ir3[3];
          ir3[3] = (v375_data + (v357_data * v373_data));
          float v378_data = s1[50];
          float v380_data = ir3[4];
          ir3[4] = (v380_data + (v357_data * v378_data));
          float v383_data = s1[64];
          float v385_data = ir3[5];
          ir3[5] = (v385_data + (v357_data * v383_data));
          float v387_data = r2[5];
          float v388_data = s1[5];
          float v390_data = ir3[0];
          ir3[0] = (v390_data + (v387_data * v388_data));
          float v393_data = s1[19];
          float v395_data = ir3[1];
          ir3[1] = (v395_data + (v387_data * v393_data));
          float v398_data = s1[30];
          float v400_data = ir3[2];
          ir3[2] = (v400_data + (v387_data * v398_data));
          float v403_data = s1[44];
          float v405_data = ir3[3];
          ir3[3] = (v405_data + (v387_data * v403_data));
          float v408_data = s1[51];
          float v410_data = ir3[4];
          ir3[4] = (v410_data + (v387_data * v408_data));
          float v413_data = s1[65];
          float v415_data = ir3[5];
          ir3[5] = (v415_data + (v387_data * v413_data));
          float v417_data = r2[6];
          float v418_data = s1[6];
          float v420_data = ir3[0];
          ir3[0] = (v420_data + (v417_data * v418_data));
          float v423_data = s1[16];
          float v425_data = ir3[1];
          ir3[1] = (v425_data + (v417_data * v423_data));
          float v428_data = s1[29];
          float v430_data = ir3[2];
          ir3[2] = (v430_data + (v417_data * v428_data));
          float v433_data = s1[47];
          float v435_data = ir3[3];
          ir3[3] = (v435_data + (v417_data * v433_data));
          float v438_data = s1[48];
          float v440_data = ir3[4];
          ir3[4] = (v440_data + (v417_data * v438_data));
          float v443_data = s1[66];
          float v445_data = ir3[5];
          ir3[5] = (v445_data + (v417_data * v443_data));
          float v447_data = r2[7];
          float v448_data = s1[7];
          float v450_data = ir3[0];
          ir3[0] = (v450_data + (v447_data * v448_data));
          float v453_data = s1[17];
          float v455_data = ir3[1];
          ir3[1] = (v455_data + (v447_data * v453_data));
          float v458_data = s1[28];
          float v460_data = ir3[2];
          ir3[2] = (v460_data + (v447_data * v458_data));
          float v463_data = s1[46];
          float v465_data = ir3[3];
          ir3[3] = (v465_data + (v447_data * v463_data));
          float v468_data = s1[49];
          float v470_data = ir3[4];
          ir3[4] = (v470_data + (v447_data * v468_data));
          float v473_data = s1[67];
          float v475_data = ir3[5];
          ir3[5] = (v475_data + (v447_data * v473_data));
          float v477_data = r2[8];
          float v478_data = s1[9];
          float v480_data = ir3[0];
          ir3[0] = (v480_data + (v477_data * v478_data));
          float v483_data = s1[22];
          float v485_data = ir3[1];
          ir3[1] = (v485_data + (v477_data * v483_data));
          float v488_data = s1[36];
          float v490_data = ir3[2];
          ir3[2] = (v490_data + (v477_data * v488_data));
          float v493_data = s1[41];
          float v495_data = ir3[3];
          ir3[3] = (v495_data + (v477_data * v493_data));
          float v498_data = s1[63];
          float v500_data = ir3[4];
          ir3[4] = (v500_data + (v477_data * v498_data));
          float v503_data = s1[68];
          float v505_data = ir3[5];
          ir3[5] = (v505_data + (v477_data * v503_data));
          float v507_data = r2[9];
          float v508_data = s1[8];
          float v510_data = ir3[0];
          ir3[0] = (v510_data + (v507_data * v508_data));
          float v513_data = s1[23];
          float v515_data = ir3[1];
          ir3[1] = (v515_data + (v507_data * v513_data));
          float v518_data = s1[37];
          float v520_data = ir3[2];
          ir3[2] = (v520_data + (v507_data * v518_data));
          float v523_data = s1[40];
          float v525_data = ir3[3];
          ir3[3] = (v525_data + (v507_data * v523_data));
          float v528_data = s1[62];
          float v530_data = ir3[4];
          ir3[4] = (v530_data + (v507_data * v528_data));
          float v533_data = s1[69];
          float v535_data = ir3[5];
          ir3[5] = (v535_data + (v507_data * v533_data));
          float v537_data = r2[10];
          float v538_data = s1[11];
          float v540_data = ir3[0];
          ir3[0] = (v540_data + (v537_data * v538_data));
          float v543_data = s1[20];
          float v545_data = ir3[1];
          ir3[1] = (v545_data + (v537_data * v543_data));
          float v548_data = s1[38];
          float v550_data = ir3[2];
          ir3[2] = (v550_data + (v537_data * v548_data));
          float v553_data = s1[43];
          float v555_data = ir3[3];
          ir3[3] = (v555_data + (v537_data * v553_data));
          float v558_data = s1[61];
          float v560_data = ir3[4];
          ir3[4] = (v560_data + (v537_data * v558_data));
          float v563_data = s1[70];
          float v565_data = ir3[5];
          ir3[5] = (v565_data + (v537_data * v563_data));
          float v567_data = r2[11];
          float v568_data = s1[10];
          float v570_data = ir3[0];
          ir3[0] = (v570_data + (v567_data * v568_data));
          float v573_data = s1[21];
          float v575_data = ir3[1];
          ir3[1] = (v575_data + (v567_data * v573_data));
          float v578_data = s1[39];
          float v580_data = ir3[2];
          ir3[2] = (v580_data + (v567_data * v578_data));
          float v583_data = s1[42];
          float v585_data = ir3[3];
          ir3[3] = (v585_data + (v567_data * v583_data));
          float v588_data = s1[60];
          float v590_data = ir3[4];
          ir3[4] = (v590_data + (v567_data * v588_data));
          float v593_data = s1[71];
          float v595_data = ir3[5];
          ir3[5] = (v595_data + (v567_data * v593_data));
          // r3 = ir3
          if (v25_g) {
            #pragma unroll
            for (int32_t v597_n1 = 0; v597_n1 < 6; ++v597_n1) {
              float v599_data = ir3[v597_n1];
              r3[v597_n1] = v599_data;
            }
          }
          // glb_m2 = store{r>g}(r3);
          if (v25_g) {
            #pragma unroll
            for (int32_t v600_i1 = 0; v600_i1 < 6; ++v600_i1) {
              float v602_data = r3[v600_i1];
              glb_m2[(v24_lead + (v600_i1 * 12))] = v602_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

