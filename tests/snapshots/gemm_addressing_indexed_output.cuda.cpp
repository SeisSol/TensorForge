// === base name ===
kernel_b36657c8672fb863

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b36657c8672fb863 = {{16, 8, 1}, 16, 9, 1, 8, 7168, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b36657c8672fb863(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b36657c8672fb863(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b36657c8672fb863(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_b36657c8672fb863, block.x * block.y * block.z, 896 * sizeof(double));
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
  config.sharedMemBytes = 896 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_b36657c8672fb863(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b36657c8672fb863(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_b36657c8672fb863, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  const unsigned * elementIndicesArg = (const unsigned *)elementIndices;
  const unsigned * flags0Arg = (const unsigned *)flags0;
  kernel_kernel_b36657c8672fb863<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, elementIndicesArg, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_b36657c8672fb863(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * __restrict__ elementIndices, const unsigned * __restrict__ flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (9 active) x 8 per block = block 16x8x1, 7168 B shared, occupancy grid
    // operands:
    //   m0 9×9(9×9) {0..9}×{0..9} indexed
    //   m1 9×9(9×9) {0..9}×{0..9} indexed(offset=8)
    //   m2 9×9(9×9) {0..9}×{0..9} indexed(stride=96)
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"double","launch":{"active_threads":9,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":896}],"shared_bytes":7168,"shared_elements":896,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"indexed","alias":"C","bbox":[[0,0],[9,9]],"name":"m0","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"indexed(offset=8)","alias":"A","bbox":[[0,0],[9,9]],"name":"m1","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"indexed(stride=96)","alias":"B","bbox":[[0,0],[9,9]],"name":"m2","ordered":false,"parts":1,"shape":[9,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"indexed","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[9,9]},"kind":"multilinear","ops":[{"addressing":"indexed(offset=8)","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[9,9]},{"addressing":"indexed(stride=96)","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[112 * threadIdx.y + 0];
      size_t v3_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < numElements0) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      double * __restrict__ s0 = &localShrMem0[0];
      double r0[9]{};
      const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
      size_t v457_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v457_flagSecondAhead < numElements0) ? v457_flagSecondAhead : v3_batchId1)];
      const bool allowed_peel = static_cast<bool>(flagWordFirst);
      const double *const __restrict__ peel_glb_m1 = &m1[static_cast<size_t>(elementIndices[v3_batchId1]) * 81 + 0 + 8 + m1_extraOffset];
      int32_t v467_lead = threadIdx.x % 16;
      bool v468_g = v467_lead < 9;
      bool v485_g = allowed_peel && v468_g;
      // r0 = load{g>r}(peel_glb_m1);
      #pragma unroll
      for (int32_t v477_i1 = 0; v477_i1 < 9; ++v477_i1) {
        double v480_data = v485_g ? (__ldcg(&peel_glb_m1[(v467_lead + (v477_i1 * 9))])) : (0.0);
        r0[v477_i1] = v480_data;
      }
      const double *const __restrict__ peel_glb_m2 = &m2[static_cast<size_t>(elementIndices[v3_batchId1]) * 96 + 0 + m2_extraOffset];
      bool v489_g = threadIdx.x < 1;
      // s0 = load{g>s}(peel_glb_m2[0, 1])
      #pragma unroll
      for (int32_t v492_i = 0; v492_i < 5; ++v492_i) {
        int32_t v494_a = threadIdx.x + (v492_i * 16);
        if (allowed_peel) {
          __pipeline_memcpy_async(&s0[v494_a], &peel_glb_m2[v494_a], 8);
        }
      }
      if (v489_g) {
        if (allowed_peel) {
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 80], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 80], 8);
        }
      }
      __pipeline_commit();
      uint32_t v461_flagWord = flagWordFirst;
      uint32_t v462_flagWordNext = flagWordSecond;
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const double *const __restrict__ wrap_glb_m1 = &m1[static_cast<size_t>(elementIndices[v11_batchId1]) * 81 + 0 + 8 + m1_extraOffset];
        const double *const __restrict__ wrap_glb_m2 = &m2[static_cast<size_t>(elementIndices[v11_batchId1]) * 96 + 0 + m2_extraOffset];
        size_t v508_flagAheadAhead = v11_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v508_flagAheadAhead < numElements0) ? v508_flagAheadAhead : v11_batchId1)];
        const bool allowed = static_cast<bool>(v461_flagWord);
        const bool allowed_next = static_cast<bool>(v462_flagWordNext);
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[static_cast<size_t>(elementIndices[v8_batchId0]) * 81 + 0 + m0_extraOffset];
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          double r1[9]{};
          // ir1 = +(r0 * s0)
          // [(0, 9), (0, 9)] [(0, 9)]
          double ir1[9]{};
          double v39_data = r0[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          double v40_data = s0[0];
          double v45_data = s0[9];
          double v50_data = s0[18];
          double v55_data = s0[27];
          double v60_data = s0[36];
          double v65_data = s0[45];
          double v70_data = s0[54];
          double v75_data = s0[63];
          double v80_data = s0[72];
          double v84_data = r0[1];
          double v85_data = s0[1];
          double v90_data = s0[10];
          double v95_data = s0[19];
          double v100_data = s0[28];
          double v105_data = s0[37];
          double v110_data = s0[46];
          double v115_data = s0[55];
          double v120_data = s0[64];
          double v125_data = s0[73];
          double v129_data = r0[2];
          double v130_data = s0[2];
          double v135_data = s0[11];
          double v140_data = s0[20];
          double v145_data = s0[29];
          double v150_data = s0[38];
          double v155_data = s0[47];
          double v160_data = s0[56];
          double v165_data = s0[65];
          double v170_data = s0[74];
          double v174_data = r0[3];
          double v175_data = s0[3];
          double v180_data = s0[12];
          double v185_data = s0[21];
          double v190_data = s0[30];
          double v195_data = s0[39];
          double v200_data = s0[48];
          double v205_data = s0[57];
          double v210_data = s0[66];
          double v215_data = s0[75];
          double v219_data = r0[4];
          double v220_data = s0[4];
          double v225_data = s0[13];
          double v230_data = s0[22];
          double v235_data = s0[31];
          double v240_data = s0[40];
          double v245_data = s0[49];
          double v250_data = s0[58];
          double v255_data = s0[67];
          double v260_data = s0[76];
          double v264_data = r0[5];
          double v265_data = s0[5];
          double v270_data = s0[14];
          double v275_data = s0[23];
          double v280_data = s0[32];
          double v285_data = s0[41];
          double v290_data = s0[50];
          double v295_data = s0[59];
          double v300_data = s0[68];
          double v305_data = s0[77];
          double v309_data = r0[6];
          double v310_data = s0[6];
          double v315_data = s0[15];
          double v320_data = s0[24];
          double v325_data = s0[33];
          double v330_data = s0[42];
          double v335_data = s0[51];
          double v340_data = s0[60];
          double v345_data = s0[69];
          double v350_data = s0[78];
          double v354_data = r0[7];
          double v355_data = s0[7];
          double v358_p = (((((((v39_data * v40_data) + (v84_data * v85_data)) + (v129_data * v130_data)) + (v174_data * v175_data)) + (v219_data * v220_data)) + (v264_data * v265_data)) + (v309_data * v310_data)) + (v354_data * v355_data);
          double v360_data = s0[16];
          double v363_p = (((((((v39_data * v45_data) + (v84_data * v90_data)) + (v129_data * v135_data)) + (v174_data * v180_data)) + (v219_data * v225_data)) + (v264_data * v270_data)) + (v309_data * v315_data)) + (v354_data * v360_data);
          double v365_data = s0[25];
          double v368_p = (((((((v39_data * v50_data) + (v84_data * v95_data)) + (v129_data * v140_data)) + (v174_data * v185_data)) + (v219_data * v230_data)) + (v264_data * v275_data)) + (v309_data * v320_data)) + (v354_data * v365_data);
          double v370_data = s0[34];
          double v373_p = (((((((v39_data * v55_data) + (v84_data * v100_data)) + (v129_data * v145_data)) + (v174_data * v190_data)) + (v219_data * v235_data)) + (v264_data * v280_data)) + (v309_data * v325_data)) + (v354_data * v370_data);
          double v375_data = s0[43];
          double v378_p = (((((((v39_data * v60_data) + (v84_data * v105_data)) + (v129_data * v150_data)) + (v174_data * v195_data)) + (v219_data * v240_data)) + (v264_data * v285_data)) + (v309_data * v330_data)) + (v354_data * v375_data);
          double v380_data = s0[52];
          double v383_p = (((((((v39_data * v65_data) + (v84_data * v110_data)) + (v129_data * v155_data)) + (v174_data * v200_data)) + (v219_data * v245_data)) + (v264_data * v290_data)) + (v309_data * v335_data)) + (v354_data * v380_data);
          double v385_data = s0[61];
          double v388_p = (((((((v39_data * v70_data) + (v84_data * v115_data)) + (v129_data * v160_data)) + (v174_data * v205_data)) + (v219_data * v250_data)) + (v264_data * v295_data)) + (v309_data * v340_data)) + (v354_data * v385_data);
          double v390_data = s0[70];
          double v393_p = (((((((v39_data * v75_data) + (v84_data * v120_data)) + (v129_data * v165_data)) + (v174_data * v210_data)) + (v219_data * v255_data)) + (v264_data * v300_data)) + (v309_data * v345_data)) + (v354_data * v390_data);
          double v395_data = s0[79];
          double v398_p = (((((((v39_data * v80_data) + (v84_data * v125_data)) + (v129_data * v170_data)) + (v174_data * v215_data)) + (v219_data * v260_data)) + (v264_data * v305_data)) + (v309_data * v350_data)) + (v354_data * v395_data);
          double v399_data = r0[8];
          bool v486_g = allowed_next && v468_g;
          // r0 = load{g>r}(wrap_glb_m1);
          #pragma unroll
          for (int32_t v481_i1 = 0; v481_i1 < 9; ++v481_i1) {
            double v484_data = v486_g ? (__ldcg(&wrap_glb_m1[(v467_lead + (v481_i1 * 9))])) : (0.0);
            r0[v481_i1] = v484_data;
          }
          double v400_data = s0[8];
          ir1[0] = (v358_p + (v399_data * v400_data));
          double v405_data = s0[17];
          ir1[1] = (v363_p + (v399_data * v405_data));
          double v410_data = s0[26];
          ir1[2] = (v368_p + (v399_data * v410_data));
          double v415_data = s0[35];
          ir1[3] = (v373_p + (v399_data * v415_data));
          double v420_data = s0[44];
          ir1[4] = (v378_p + (v399_data * v420_data));
          double v425_data = s0[53];
          ir1[5] = (v383_p + (v399_data * v425_data));
          double v430_data = s0[62];
          ir1[6] = (v388_p + (v399_data * v430_data));
          double v435_data = s0[71];
          ir1[7] = (v393_p + (v399_data * v435_data));
          double v440_data = s0[80];
          ir1[8] = (v398_p + (v399_data * v440_data));
          // r1 = ir1
          // glb_m0 = store{r>g}(r1);
          if (v468_g) {
            #pragma unroll
            for (int32_t v449_i1 = 0; v449_i1 < 9; ++v449_i1) {
              double v451_data = ir1[v449_i1];
              glb_m0[(v467_lead + (v449_i1 * 9))] = v451_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
        else {
          __pipeline_wait_prior(0);
          bool v487_g = allowed_next && v468_g;
          // r0 = load{g>r}(wrap_glb_m1);
          #pragma unroll
          for (int32_t v473_i1 = 0; v473_i1 < 9; ++v473_i1) {
            double v476_data = v487_g ? (__ldcg(&wrap_glb_m1[(v467_lead + (v473_i1 * 9))])) : (0.0);
            r0[v473_i1] = v476_data;
          }
        }
        // s0 = load{g>s}(wrap_glb_m2[0, 1])
        #pragma unroll
        for (int32_t v497_i = 0; v497_i < 5; ++v497_i) {
          int32_t v499_a = threadIdx.x + (v497_i * 16);
          if (allowed_next) {
            __pipeline_memcpy_async(&s0[v499_a], &wrap_glb_m2[v499_a], 8);
          }
        }
        if (v489_g) {
          if (allowed_next) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 80], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 80], 8);
          }
        }
        __pipeline_commit();
        v461_flagWord = v462_flagWordNext;
        v462_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
  }
}

