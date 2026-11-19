// === base name ===
kernel_d09cdd7ec5660d3d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_d09cdd7ec5660d3d = {{16, 8, 1}, 16, 9, 1, 8, 7168, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_d09cdd7ec5660d3d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_d09cdd7ec5660d3d(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_d09cdd7ec5660d3d(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_d09cdd7ec5660d3d, block.x * block.y * block.z, 896 * sizeof(double));
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
void launcher_kernel_d09cdd7ec5660d3d(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_d09cdd7ec5660d3d(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_d09cdd7ec5660d3d, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  const unsigned * elementIndicesArg = (const unsigned *)elementIndices;
  const unsigned * flags0Arg = (const unsigned *)flags0;
  kernel_kernel_d09cdd7ec5660d3d<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, elementIndicesArg, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_d09cdd7ec5660d3d(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * __restrict__ elementIndices, const unsigned * __restrict__ flags0) {
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
      size_t v458_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v458_flagSecondAhead < numElements0) ? v458_flagSecondAhead : v3_batchId1)];
      const bool allowed_peel = static_cast<bool>(flagWordFirst);
      const size_t peel_elementIndex = static_cast<size_t>(elementIndices[v3_batchId1]);
      const double *const __restrict__ peel_glb_m1 = &m1[peel_elementIndex * 81 + 0 + 8 + m1_extraOffset];
      int32_t v469_lead = threadIdx.x % 16;
      bool v470_g = v469_lead < 9;
      bool v488_g = allowed_peel && v470_g;
      // r0 = load{g>r}(peel_glb_m1);
      #pragma unroll
      for (int32_t v480_i1 = 0; v480_i1 < 9; ++v480_i1) {
        double v483_data = v488_g ? (__ldcg(&peel_glb_m1[(v469_lead + (v480_i1 * 9))])) : (0.0);
        r0[v480_i1] = v483_data;
      }
      const double *const __restrict__ peel_glb_m2 = &m2[peel_elementIndex * 96 + 0 + m2_extraOffset];
      bool v492_g = threadIdx.x < 1;
      // s0 = load{g>s}(peel_glb_m2[0, 1])
      #pragma unroll
      for (int32_t v495_i = 0; v495_i < 5; ++v495_i) {
        int32_t v497_a = threadIdx.x + (v495_i * 16);
        if (allowed_peel) {
          __pipeline_memcpy_async(&s0[v497_a], &peel_glb_m2[v497_a], 8);
        }
      }
      if (v492_g) {
        if (allowed_peel) {
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 80], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 80], 8);
        }
      }
      __pipeline_commit();
      uint32_t v462_flagWord = flagWordFirst;
      uint32_t v463_flagWordNext = flagWordSecond;
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const size_t wrap_elementIndex = static_cast<size_t>(elementIndices[v11_batchId1]);
        const double *const __restrict__ wrap_glb_m1 = &m1[wrap_elementIndex * 81 + 0 + 8 + m1_extraOffset];
        const double *const __restrict__ wrap_glb_m2 = &m2[wrap_elementIndex * 96 + 0 + m2_extraOffset];
        size_t v511_flagAheadAhead = v11_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v511_flagAheadAhead < numElements0) ? v511_flagAheadAhead : v11_batchId1)];
        const bool allowed = static_cast<bool>(v462_flagWord);
        const bool allowed_next = static_cast<bool>(v463_flagWordNext);
        if (allowed) {
          const size_t elementIndex = static_cast<size_t>(elementIndices[v8_batchId0]);
          double *const __restrict__ glb_m0 = &m0[elementIndex * 81 + 0 + m0_extraOffset];
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          double r1[9]{};
          // ir1 = +(r0 * s0)
          // [(0, 9), (0, 9)] [(0, 9)]
          double ir1[9]{};
          double v40_data = r0[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          double v41_data = s0[0];
          double v46_data = s0[9];
          double v51_data = s0[18];
          double v56_data = s0[27];
          double v61_data = s0[36];
          double v66_data = s0[45];
          double v71_data = s0[54];
          double v76_data = s0[63];
          double v81_data = s0[72];
          double v85_data = r0[1];
          double v86_data = s0[1];
          double v91_data = s0[10];
          double v96_data = s0[19];
          double v101_data = s0[28];
          double v106_data = s0[37];
          double v111_data = s0[46];
          double v116_data = s0[55];
          double v121_data = s0[64];
          double v126_data = s0[73];
          double v130_data = r0[2];
          double v131_data = s0[2];
          double v136_data = s0[11];
          double v141_data = s0[20];
          double v146_data = s0[29];
          double v151_data = s0[38];
          double v156_data = s0[47];
          double v161_data = s0[56];
          double v166_data = s0[65];
          double v171_data = s0[74];
          double v175_data = r0[3];
          double v176_data = s0[3];
          double v181_data = s0[12];
          double v186_data = s0[21];
          double v191_data = s0[30];
          double v196_data = s0[39];
          double v201_data = s0[48];
          double v206_data = s0[57];
          double v211_data = s0[66];
          double v216_data = s0[75];
          double v220_data = r0[4];
          double v221_data = s0[4];
          double v226_data = s0[13];
          double v231_data = s0[22];
          double v236_data = s0[31];
          double v241_data = s0[40];
          double v246_data = s0[49];
          double v251_data = s0[58];
          double v256_data = s0[67];
          double v261_data = s0[76];
          double v265_data = r0[5];
          double v266_data = s0[5];
          double v271_data = s0[14];
          double v276_data = s0[23];
          double v281_data = s0[32];
          double v286_data = s0[41];
          double v291_data = s0[50];
          double v296_data = s0[59];
          double v301_data = s0[68];
          double v306_data = s0[77];
          double v310_data = r0[6];
          double v311_data = s0[6];
          double v316_data = s0[15];
          double v321_data = s0[24];
          double v326_data = s0[33];
          double v331_data = s0[42];
          double v336_data = s0[51];
          double v341_data = s0[60];
          double v346_data = s0[69];
          double v351_data = s0[78];
          double v355_data = r0[7];
          double v356_data = s0[7];
          double v359_p = (((((((v40_data * v41_data) + (v85_data * v86_data)) + (v130_data * v131_data)) + (v175_data * v176_data)) + (v220_data * v221_data)) + (v265_data * v266_data)) + (v310_data * v311_data)) + (v355_data * v356_data);
          double v361_data = s0[16];
          double v364_p = (((((((v40_data * v46_data) + (v85_data * v91_data)) + (v130_data * v136_data)) + (v175_data * v181_data)) + (v220_data * v226_data)) + (v265_data * v271_data)) + (v310_data * v316_data)) + (v355_data * v361_data);
          double v366_data = s0[25];
          double v369_p = (((((((v40_data * v51_data) + (v85_data * v96_data)) + (v130_data * v141_data)) + (v175_data * v186_data)) + (v220_data * v231_data)) + (v265_data * v276_data)) + (v310_data * v321_data)) + (v355_data * v366_data);
          double v371_data = s0[34];
          double v374_p = (((((((v40_data * v56_data) + (v85_data * v101_data)) + (v130_data * v146_data)) + (v175_data * v191_data)) + (v220_data * v236_data)) + (v265_data * v281_data)) + (v310_data * v326_data)) + (v355_data * v371_data);
          double v376_data = s0[43];
          double v379_p = (((((((v40_data * v61_data) + (v85_data * v106_data)) + (v130_data * v151_data)) + (v175_data * v196_data)) + (v220_data * v241_data)) + (v265_data * v286_data)) + (v310_data * v331_data)) + (v355_data * v376_data);
          double v381_data = s0[52];
          double v384_p = (((((((v40_data * v66_data) + (v85_data * v111_data)) + (v130_data * v156_data)) + (v175_data * v201_data)) + (v220_data * v246_data)) + (v265_data * v291_data)) + (v310_data * v336_data)) + (v355_data * v381_data);
          double v386_data = s0[61];
          double v389_p = (((((((v40_data * v71_data) + (v85_data * v116_data)) + (v130_data * v161_data)) + (v175_data * v206_data)) + (v220_data * v251_data)) + (v265_data * v296_data)) + (v310_data * v341_data)) + (v355_data * v386_data);
          double v391_data = s0[70];
          double v394_p = (((((((v40_data * v76_data) + (v85_data * v121_data)) + (v130_data * v166_data)) + (v175_data * v211_data)) + (v220_data * v256_data)) + (v265_data * v301_data)) + (v310_data * v346_data)) + (v355_data * v391_data);
          double v396_data = s0[79];
          double v399_p = (((((((v40_data * v81_data) + (v85_data * v126_data)) + (v130_data * v171_data)) + (v175_data * v216_data)) + (v220_data * v261_data)) + (v265_data * v306_data)) + (v310_data * v351_data)) + (v355_data * v396_data);
          double v400_data = r0[8];
          bool v489_g = allowed_next && v470_g;
          // r0 = load{g>r}(wrap_glb_m1);
          #pragma unroll
          for (int32_t v484_i1 = 0; v484_i1 < 9; ++v484_i1) {
            double v487_data = v489_g ? (__ldcg(&wrap_glb_m1[(v469_lead + (v484_i1 * 9))])) : (0.0);
            r0[v484_i1] = v487_data;
          }
          double v401_data = s0[8];
          ir1[0] = (v359_p + (v400_data * v401_data));
          double v406_data = s0[17];
          ir1[1] = (v364_p + (v400_data * v406_data));
          double v411_data = s0[26];
          ir1[2] = (v369_p + (v400_data * v411_data));
          double v416_data = s0[35];
          ir1[3] = (v374_p + (v400_data * v416_data));
          double v421_data = s0[44];
          ir1[4] = (v379_p + (v400_data * v421_data));
          double v426_data = s0[53];
          ir1[5] = (v384_p + (v400_data * v426_data));
          double v431_data = s0[62];
          ir1[6] = (v389_p + (v400_data * v431_data));
          double v436_data = s0[71];
          ir1[7] = (v394_p + (v400_data * v436_data));
          double v441_data = s0[80];
          ir1[8] = (v399_p + (v400_data * v441_data));
          // r1 = ir1
          // glb_m0 = store{r>g}(r1);
          if (v470_g) {
            #pragma unroll
            for (int32_t v450_i1 = 0; v450_i1 < 9; ++v450_i1) {
              double v452_data = ir1[v450_i1];
              glb_m0[(v469_lead + (v450_i1 * 9))] = v452_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
        else {
          __pipeline_wait_prior(0);
          bool v490_g = allowed_next && v470_g;
          // r0 = load{g>r}(wrap_glb_m1);
          #pragma unroll
          for (int32_t v476_i1 = 0; v476_i1 < 9; ++v476_i1) {
            double v479_data = v490_g ? (__ldcg(&wrap_glb_m1[(v469_lead + (v476_i1 * 9))])) : (0.0);
            r0[v476_i1] = v479_data;
          }
        }
        // s0 = load{g>s}(wrap_glb_m2[0, 1])
        #pragma unroll
        for (int32_t v500_i = 0; v500_i < 5; ++v500_i) {
          int32_t v502_a = threadIdx.x + (v500_i * 16);
          if (allowed_next) {
            __pipeline_memcpy_async(&s0[v502_a], &wrap_glb_m2[v502_a], 8);
          }
        }
        if (v492_g) {
          if (allowed_next) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 80], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 80], 8);
          }
        }
        __pipeline_commit();
        v462_flagWord = v463_flagWordNext;
        v463_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
  }
}

