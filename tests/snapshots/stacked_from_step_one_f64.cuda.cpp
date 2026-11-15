// === base name ===
kernel_969f83c917c21616

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_969f83c917c21616 = {{16, 8, 1}, 16, 40, 1, 8, 18432, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_969f83c917c21616(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_969f83c917c21616(const double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, double * m2, size_t m2_extraOffset, const double * m3, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_969f83c917c21616(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_969f83c917c21616, block.x * block.y * block.z, 2304 * sizeof(double));
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
  config.sharedMemBytes = 2304 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_969f83c917c21616(const double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, double * m2, size_t m2_extraOffset, const double * m3, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_969f83c917c21616(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_969f83c917c21616, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  const unsigned * flags0Arg = (const unsigned *)flags0;
  kernel_kernel_969f83c917c21616<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_969f83c917c21616(const double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, double * m2, size_t m2_extraOffset, const double * m3, size_t numElements0, const unsigned * __restrict__ flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: amd_matrix_k4=1,lanes_per_mult=16
    // launch: 16 lanes (40 active) x 8 per block = block 16x8x1, 18432 B shared, occupancy grid
    // operands:
    //   m0 30×9(30×9) {0..30}×{0..9} strided
    //   m1 9×9(9×9) {0..9}×{0..9} strided
    //   m2 40×9(40×9) {0..40}×{0..9} strided
    //   m3 40×30(40×30) {0..40}×{0..30} none ordered runs-packed
    // operations:
    //   t0[i,j] = m0[i,k] × m1[k,j]
    //   m2[i,j] = m3[i,k]@{0..40}×{1..30} × t0[k,j]@{1..30}×{0..9}
    // tensorforge-meta: {"fp":"double","launch":{"active_threads":40,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":2304}],"shared_bytes":18432,"shared_elements":2304,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"X","bbox":[[0,0],[30,9]],"name":"m0","ordered":false,"parts":1,"shape":[30,9],"variant":false},{"addressing":"strided","alias":"S","bbox":[[0,0],[9,9]],"name":"m1","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[40,9]],"name":"m2","ordered":false,"parts":1,"shape":[40,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[40,30]],"name":"m3","ordered":true,"parts":1,"shape":[40,30],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[30,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[30,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[30,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[30,9]},{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[40,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[40,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[40,30]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[40,30]},{"addressing":"strided","bbox":[[1,0],[30,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[30,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[288 * threadIdx.y + 0];
      size_t v3_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < numElements0) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      const double *const __restrict__ glb_m3 = &m3[0];
      double * __restrict__ s0 = &localShrMem0[0];
      double * __restrict__ s1 = &localShrMem0[0];
      double r0[18]{};
      const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
      size_t v4533_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v4533_flagSecondAhead < numElements0) ? v4533_flagSecondAhead : v3_batchId1)];
      const double *const __restrict__ peel_glb_m0 = &m0[v3_batchId1 * 270 + 0 + m0_extraOffset];
      int32_t v4541_lead = threadIdx.x % 16;
      int32_t v4543_lead = v4541_lead + 16_i32;
      bool v4544_g = v4541_lead < 14;
      // r0 = load{g>r}(peel_glb_m0);
      #pragma unroll
      for (int32_t v4566_i0 = 0; v4566_i0 < 1; ++v4566_i0) {
        int32_t v4568_lead = v4541_lead + (v4566_i0 * 16);
        #pragma unroll
        for (int32_t v4569_i1 = 0; v4569_i1 < 9; ++v4569_i1) {
          double v4572_data = __ldcg(&peel_glb_m0[(v4568_lead + (v4569_i1 * 30))]);
          r0[(v4566_i0 + (v4569_i1 * 2))] = v4572_data;
        }
      }
      #pragma unroll
      for (int32_t v4575_i1 = 0; v4575_i1 < 9; ++v4575_i1) {
        double v4578_data = v4544_g ? (__ldcg(&peel_glb_m0[(v4543_lead + (v4575_i1 * 30))])) : (0.0);
        r0[(1 + (v4575_i1 * 2))] = v4578_data;
      }
      const double *const __restrict__ peel_glb_m1 = &m1[v3_batchId1 * 81 + 0 + m1_extraOffset];
      bool v4597_g = threadIdx.x < 1;
      // s0 = load{g>s}(peel_glb_m1[0, 1])
      #pragma unroll
      for (int32_t v4600_i = 0; v4600_i < 5; ++v4600_i) {
        int32_t v4602_a = threadIdx.x + (v4600_i * 16);
        __pipeline_memcpy_async(&s0[v4602_a], &peel_glb_m1[v4602_a], 8);
      }
      if (v4597_g) {
        __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 80], &peel_glb_m1[0 + 0 + 1 * threadIdx.x + 80], 8);
      }
      __pipeline_commit();
      uint32_t v4537_flagWord = flagWordFirst;
      uint32_t v4538_flagWordNext = flagWordSecond;
      for (size_t v10_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v10_batchId0 < numElements0; v10_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v11_ahead1 = v10_batchId0 + (gridDim.x * blockDim.y);
        size_t v13_batchId1 = (v11_ahead1 < numElements0) ? v11_ahead1 : v10_batchId0;
        const double *const __restrict__ wrap_glb_m0 = &m0[v13_batchId1 * 270 + 0 + m0_extraOffset];
        const double *const __restrict__ wrap_glb_m1 = &m1[v13_batchId1 * 81 + 0 + m1_extraOffset];
        size_t v4616_flagAheadAhead = v13_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v4616_flagAheadAhead < numElements0) ? v4616_flagAheadAhead : v13_batchId1)];
        const bool allowed = static_cast<bool>(v4537_flagWord);
        if (allowed) {
          double *const __restrict__ glb_m2 = &m2[v10_batchId0 * 360 + 0 + m2_extraOffset];
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          double r1[18]{};
          // r1 = +(r0 * s0) + None
          // [(0, 30), (0, 9)] [(0, 9)]
          double v50_data = r0[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          double v51_data = s0[0];
          double v56_data = s0[9];
          double v61_data = s0[18];
          double v66_data = s0[27];
          double v71_data = s0[36];
          double v76_data = s0[45];
          double v81_data = s0[54];
          double v86_data = s0[63];
          double v91_data = s0[72];
          double v95_data = r0[1];
          double v140_data = r0[2];
          double v141_data = s0[1];
          double v146_data = s0[10];
          double v151_data = s0[19];
          double v156_data = s0[28];
          double v161_data = s0[37];
          double v166_data = s0[46];
          double v171_data = s0[55];
          double v176_data = s0[64];
          double v181_data = s0[73];
          double v185_data = r0[3];
          double v230_data = r0[4];
          double v231_data = s0[2];
          double v236_data = s0[11];
          double v241_data = s0[20];
          double v246_data = s0[29];
          double v251_data = s0[38];
          double v256_data = s0[47];
          double v261_data = s0[56];
          double v266_data = s0[65];
          double v271_data = s0[74];
          double v275_data = r0[5];
          double v320_data = r0[6];
          double v321_data = s0[3];
          double v326_data = s0[12];
          double v331_data = s0[21];
          double v336_data = s0[30];
          double v341_data = s0[39];
          double v346_data = s0[48];
          double v351_data = s0[57];
          double v356_data = s0[66];
          double v361_data = s0[75];
          double v365_data = r0[7];
          double v410_data = r0[8];
          double v411_data = s0[4];
          double v416_data = s0[13];
          double v421_data = s0[22];
          double v426_data = s0[31];
          double v431_data = s0[40];
          double v436_data = s0[49];
          double v441_data = s0[58];
          double v446_data = s0[67];
          double v451_data = s0[76];
          double v455_data = r0[9];
          double v500_data = r0[10];
          double v501_data = s0[5];
          double v506_data = s0[14];
          double v511_data = s0[23];
          double v516_data = s0[32];
          double v521_data = s0[41];
          double v526_data = s0[50];
          double v531_data = s0[59];
          double v536_data = s0[68];
          double v541_data = s0[77];
          double v545_data = r0[11];
          double v590_data = r0[12];
          double v591_data = s0[6];
          double v596_data = s0[15];
          double v601_data = s0[24];
          double v606_data = s0[33];
          double v611_data = s0[42];
          double v616_data = s0[51];
          double v621_data = s0[60];
          double v626_data = s0[69];
          double v631_data = s0[78];
          double v635_data = r0[13];
          double v680_data = r0[14];
          double v681_data = s0[7];
          double v686_data = s0[16];
          double v689_p = (((((((v50_data * v56_data) + (v140_data * v146_data)) + (v230_data * v236_data)) + (v320_data * v326_data)) + (v410_data * v416_data)) + (v500_data * v506_data)) + (v590_data * v596_data)) + (v680_data * v686_data);
          double v691_data = s0[25];
          double v694_p = (((((((v50_data * v61_data) + (v140_data * v151_data)) + (v230_data * v241_data)) + (v320_data * v331_data)) + (v410_data * v421_data)) + (v500_data * v511_data)) + (v590_data * v601_data)) + (v680_data * v691_data);
          double v696_data = s0[34];
          double v699_p = (((((((v50_data * v66_data) + (v140_data * v156_data)) + (v230_data * v246_data)) + (v320_data * v336_data)) + (v410_data * v426_data)) + (v500_data * v516_data)) + (v590_data * v606_data)) + (v680_data * v696_data);
          double v701_data = s0[43];
          double v704_p = (((((((v50_data * v71_data) + (v140_data * v161_data)) + (v230_data * v251_data)) + (v320_data * v341_data)) + (v410_data * v431_data)) + (v500_data * v521_data)) + (v590_data * v611_data)) + (v680_data * v701_data);
          double v706_data = s0[52];
          double v709_p = (((((((v50_data * v76_data) + (v140_data * v166_data)) + (v230_data * v256_data)) + (v320_data * v346_data)) + (v410_data * v436_data)) + (v500_data * v526_data)) + (v590_data * v616_data)) + (v680_data * v706_data);
          double v711_data = s0[61];
          double v714_p = (((((((v50_data * v81_data) + (v140_data * v171_data)) + (v230_data * v261_data)) + (v320_data * v351_data)) + (v410_data * v441_data)) + (v500_data * v531_data)) + (v590_data * v621_data)) + (v680_data * v711_data);
          double v716_data = s0[70];
          double v719_p = (((((((v50_data * v86_data) + (v140_data * v176_data)) + (v230_data * v266_data)) + (v320_data * v356_data)) + (v410_data * v446_data)) + (v500_data * v536_data)) + (v590_data * v626_data)) + (v680_data * v716_data);
          double v721_data = s0[79];
          double v724_p = (((((((v50_data * v91_data) + (v140_data * v181_data)) + (v230_data * v271_data)) + (v320_data * v361_data)) + (v410_data * v451_data)) + (v500_data * v541_data)) + (v590_data * v631_data)) + (v680_data * v721_data);
          double v725_data = r0[15];
          double v729_p = (((((((v95_data * v51_data) + (v185_data * v141_data)) + (v275_data * v231_data)) + (v365_data * v321_data)) + (v455_data * v411_data)) + (v545_data * v501_data)) + (v635_data * v591_data)) + (v725_data * v681_data);
          double v734_p = (((((((v95_data * v56_data) + (v185_data * v146_data)) + (v275_data * v236_data)) + (v365_data * v326_data)) + (v455_data * v416_data)) + (v545_data * v506_data)) + (v635_data * v596_data)) + (v725_data * v686_data);
          double v739_p = (((((((v95_data * v61_data) + (v185_data * v151_data)) + (v275_data * v241_data)) + (v365_data * v331_data)) + (v455_data * v421_data)) + (v545_data * v511_data)) + (v635_data * v601_data)) + (v725_data * v691_data);
          double v744_p = (((((((v95_data * v66_data) + (v185_data * v156_data)) + (v275_data * v246_data)) + (v365_data * v336_data)) + (v455_data * v426_data)) + (v545_data * v516_data)) + (v635_data * v606_data)) + (v725_data * v696_data);
          double v749_p = (((((((v95_data * v71_data) + (v185_data * v161_data)) + (v275_data * v251_data)) + (v365_data * v341_data)) + (v455_data * v431_data)) + (v545_data * v521_data)) + (v635_data * v611_data)) + (v725_data * v701_data);
          double v754_p = (((((((v95_data * v76_data) + (v185_data * v166_data)) + (v275_data * v256_data)) + (v365_data * v346_data)) + (v455_data * v436_data)) + (v545_data * v526_data)) + (v635_data * v616_data)) + (v725_data * v706_data);
          double v759_p = (((((((v95_data * v81_data) + (v185_data * v171_data)) + (v275_data * v261_data)) + (v365_data * v351_data)) + (v455_data * v441_data)) + (v545_data * v531_data)) + (v635_data * v621_data)) + (v725_data * v711_data);
          double v764_p = (((((((v95_data * v86_data) + (v185_data * v176_data)) + (v275_data * v266_data)) + (v365_data * v356_data)) + (v455_data * v446_data)) + (v545_data * v536_data)) + (v635_data * v626_data)) + (v725_data * v716_data);
          double v769_p = (((((((v95_data * v91_data) + (v185_data * v181_data)) + (v275_data * v271_data)) + (v365_data * v361_data)) + (v455_data * v451_data)) + (v545_data * v541_data)) + (v635_data * v631_data)) + (v725_data * v721_data);
          double v770_data = r0[16];
          double v771_data = s0[8];
          r1[0] = (((((((((v50_data * v51_data) + (v140_data * v141_data)) + (v230_data * v231_data)) + (v320_data * v321_data)) + (v410_data * v411_data)) + (v500_data * v501_data)) + (v590_data * v591_data)) + (v680_data * v681_data)) + (v770_data * v771_data));
          double v776_data = s0[17];
          r1[2] = (v689_p + (v770_data * v776_data));
          double v781_data = s0[26];
          r1[4] = (v694_p + (v770_data * v781_data));
          double v786_data = s0[35];
          r1[6] = (v699_p + (v770_data * v786_data));
          double v791_data = s0[44];
          r1[8] = (v704_p + (v770_data * v791_data));
          double v796_data = s0[53];
          r1[10] = (v709_p + (v770_data * v796_data));
          double v801_data = s0[62];
          r1[12] = (v714_p + (v770_data * v801_data));
          double v806_data = s0[71];
          r1[14] = (v719_p + (v770_data * v806_data));
          double v811_data = s0[80];
          r1[16] = (v724_p + (v770_data * v811_data));
          double v815_data = r0[17];
          r1[1] = (v729_p + (v815_data * v771_data));
          r1[3] = (v734_p + (v815_data * v776_data));
          r1[5] = (v739_p + (v815_data * v781_data));
          r1[7] = (v744_p + (v815_data * v786_data));
          r1[9] = (v749_p + (v815_data * v791_data));
          r1[11] = (v754_p + (v815_data * v796_data));
          r1[13] = (v759_p + (v815_data * v801_data));
          r1[15] = (v764_p + (v815_data * v806_data));
          r1[17] = (v769_p + (v815_data * v811_data));
          // r1 = 0 past (0, 30)
          // s1 = store{r>s}(localShrMem0, r1);
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          #pragma unroll
          for (int32_t v861_i0 = 0; v861_i0 < 1; ++v861_i0) {
            int32_t v867_lead = v4541_lead + (v861_i0 * 16);
            #pragma unroll
            for (int32_t v862_i1 = 0; v862_i1 < 9; ++v862_i1) {
              double v865_data = r1[(v861_i0 + (v862_i1 * 2))];
              s1[(v867_lead + (v862_i1 * 30))] = v865_data;
            }
          }
          if (v4544_g) {
            #pragma unroll
            for (int32_t v871_i1 = 0; v871_i1 < 9; ++v871_i1) {
              double v874_data = r1[(1 + (v871_i1 * 2))];
              s1[(v4543_lead + (v871_i1 * 30))] = v874_data;
            }
          }
          double r2[27]{};
          // ir2 = +(glb_m3 * s1)
          // [(0, 40), (0, 9)] [(1, 30)]
          double ir2[27]{};
          double v884_data = glb_m3[v4541_lead];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          double v885_data = s1[1];
          double v890_data = s1[31];
          double v895_data = s1[61];
          double v900_data = s1[91];
          double v905_data = s1[121];
          double v910_data = s1[151];
          double v915_data = s1[181];
          double v920_data = s1[211];
          double v925_data = s1[241];
          double v932_data = glb_m3[v4543_lead];
          int32_t v978_lead = v4541_lead + 32_i32;
          bool v980_g = v4541_lead < 8;
          double v981_data = v980_g ? (glb_m3[v978_lead]) : (0.0);
          double v1028_data = glb_m3[(v4541_lead + 40)];
          double v1029_data = s1[2];
          double v1034_data = s1[32];
          double v1039_data = s1[62];
          double v1044_data = s1[92];
          double v1049_data = s1[122];
          double v1054_data = s1[152];
          double v1059_data = s1[182];
          double v1064_data = s1[212];
          double v1069_data = s1[242];
          double v1075_data = glb_m3[(v4543_lead + 40)];
          double v1122_data = v980_g ? (glb_m3[(v978_lead + 40)]) : (0.0);
          double v1169_data = glb_m3[(v4541_lead + 80)];
          double v1170_data = s1[3];
          double v1175_data = s1[33];
          double v1180_data = s1[63];
          double v1185_data = s1[93];
          double v1190_data = s1[123];
          double v1195_data = s1[153];
          double v1200_data = s1[183];
          double v1205_data = s1[213];
          double v1210_data = s1[243];
          double v1216_data = glb_m3[(v4543_lead + 80)];
          double v1263_data = v980_g ? (glb_m3[(v978_lead + 80)]) : (0.0);
          double v1310_data = glb_m3[(v4541_lead + 120)];
          double v1311_data = s1[4];
          double v1316_data = s1[34];
          double v1321_data = s1[64];
          double v1326_data = s1[94];
          double v1331_data = s1[124];
          double v1336_data = s1[154];
          double v1341_data = s1[184];
          double v1346_data = s1[214];
          double v1351_data = s1[244];
          double v1357_data = glb_m3[(v4543_lead + 120)];
          double v1404_data = v980_g ? (glb_m3[(v978_lead + 120)]) : (0.0);
          double v1451_data = glb_m3[(v4541_lead + 160)];
          double v1452_data = s1[5];
          double v1457_data = s1[35];
          double v1462_data = s1[65];
          double v1467_data = s1[95];
          double v1472_data = s1[125];
          double v1477_data = s1[155];
          double v1482_data = s1[185];
          double v1487_data = s1[215];
          double v1492_data = s1[245];
          double v1498_data = glb_m3[(v4543_lead + 160)];
          double v1545_data = v980_g ? (glb_m3[(v978_lead + 160)]) : (0.0);
          double v1592_data = glb_m3[(v4541_lead + 200)];
          double v1593_data = s1[6];
          double v1598_data = s1[36];
          double v1603_data = s1[66];
          double v1608_data = s1[96];
          double v1613_data = s1[126];
          double v1618_data = s1[156];
          double v1623_data = s1[186];
          double v1628_data = s1[216];
          double v1633_data = s1[246];
          double v1639_data = glb_m3[(v4543_lead + 200)];
          double v1686_data = v980_g ? (glb_m3[(v978_lead + 200)]) : (0.0);
          double v1733_data = glb_m3[(v4541_lead + 240)];
          double v1734_data = s1[7];
          double v1739_data = s1[37];
          double v1744_data = s1[67];
          double v1749_data = s1[97];
          double v1754_data = s1[127];
          double v1759_data = s1[157];
          double v1764_data = s1[187];
          double v1769_data = s1[217];
          double v1774_data = s1[247];
          double v1780_data = glb_m3[(v4543_lead + 240)];
          double v1827_data = v980_g ? (glb_m3[(v978_lead + 240)]) : (0.0);
          double v1874_data = glb_m3[(v4541_lead + 280)];
          double v1875_data = s1[8];
          double v1880_data = s1[38];
          double v1885_data = s1[68];
          double v1890_data = s1[98];
          double v1895_data = s1[128];
          double v1900_data = s1[158];
          double v1905_data = s1[188];
          double v1910_data = s1[218];
          double v1915_data = s1[248];
          double v1921_data = glb_m3[(v4543_lead + 280)];
          double v1968_data = v980_g ? (glb_m3[(v978_lead + 280)]) : (0.0);
          double v2015_data = glb_m3[(v4541_lead + 320)];
          double v2016_data = s1[9];
          double v2021_data = s1[39];
          double v2026_data = s1[69];
          double v2031_data = s1[99];
          double v2036_data = s1[129];
          double v2041_data = s1[159];
          double v2046_data = s1[189];
          double v2051_data = s1[219];
          double v2056_data = s1[249];
          double v2062_data = glb_m3[(v4543_lead + 320)];
          double v2109_data = v980_g ? (glb_m3[(v978_lead + 320)]) : (0.0);
          double v2156_data = glb_m3[(v4541_lead + 360)];
          double v2157_data = s1[10];
          double v2162_data = s1[40];
          double v2167_data = s1[70];
          double v2172_data = s1[100];
          double v2177_data = s1[130];
          double v2182_data = s1[160];
          double v2187_data = s1[190];
          double v2192_data = s1[220];
          double v2197_data = s1[250];
          double v2203_data = glb_m3[(v4543_lead + 360)];
          double v2250_data = v980_g ? (glb_m3[(v978_lead + 360)]) : (0.0);
          double v2297_data = glb_m3[(v4541_lead + 400)];
          double v2298_data = s1[11];
          double v2303_data = s1[41];
          double v2308_data = s1[71];
          double v2313_data = s1[101];
          double v2318_data = s1[131];
          double v2323_data = s1[161];
          double v2328_data = s1[191];
          double v2333_data = s1[221];
          double v2338_data = s1[251];
          double v2344_data = glb_m3[(v4543_lead + 400)];
          double v2391_data = v980_g ? (glb_m3[(v978_lead + 400)]) : (0.0);
          double v2438_data = glb_m3[(v4541_lead + 440)];
          double v2439_data = s1[12];
          double v2444_data = s1[42];
          double v2449_data = s1[72];
          double v2454_data = s1[102];
          double v2459_data = s1[132];
          double v2464_data = s1[162];
          double v2469_data = s1[192];
          double v2474_data = s1[222];
          double v2479_data = s1[252];
          double v2485_data = glb_m3[(v4543_lead + 440)];
          double v2532_data = v980_g ? (glb_m3[(v978_lead + 440)]) : (0.0);
          double v2579_data = glb_m3[(v4541_lead + 480)];
          double v2580_data = s1[13];
          double v2585_data = s1[43];
          double v2590_data = s1[73];
          double v2595_data = s1[103];
          double v2600_data = s1[133];
          double v2605_data = s1[163];
          double v2610_data = s1[193];
          double v2615_data = s1[223];
          double v2620_data = s1[253];
          double v2626_data = glb_m3[(v4543_lead + 480)];
          double v2673_data = v980_g ? (glb_m3[(v978_lead + 480)]) : (0.0);
          double v2720_data = glb_m3[(v4541_lead + 520)];
          double v2721_data = s1[14];
          double v2726_data = s1[44];
          double v2731_data = s1[74];
          double v2736_data = s1[104];
          double v2741_data = s1[134];
          double v2746_data = s1[164];
          double v2751_data = s1[194];
          double v2756_data = s1[224];
          double v2761_data = s1[254];
          double v2767_data = glb_m3[(v4543_lead + 520)];
          double v2814_data = v980_g ? (glb_m3[(v978_lead + 520)]) : (0.0);
          double v2861_data = glb_m3[(v4541_lead + 560)];
          double v2862_data = s1[15];
          double v2867_data = s1[45];
          double v2872_data = s1[75];
          double v2877_data = s1[105];
          double v2882_data = s1[135];
          double v2887_data = s1[165];
          double v2892_data = s1[195];
          double v2897_data = s1[225];
          double v2902_data = s1[255];
          double v2908_data = glb_m3[(v4543_lead + 560)];
          double v2955_data = v980_g ? (glb_m3[(v978_lead + 560)]) : (0.0);
          double v3002_data = glb_m3[(v4541_lead + 600)];
          double v3003_data = s1[16];
          double v3008_data = s1[46];
          double v3013_data = s1[76];
          double v3018_data = s1[106];
          double v3023_data = s1[136];
          double v3028_data = s1[166];
          double v3033_data = s1[196];
          double v3038_data = s1[226];
          double v3043_data = s1[256];
          double v3049_data = glb_m3[(v4543_lead + 600)];
          double v3096_data = v980_g ? (glb_m3[(v978_lead + 600)]) : (0.0);
          double v3143_data = glb_m3[(v4541_lead + 640)];
          double v3144_data = s1[17];
          double v3149_data = s1[47];
          double v3154_data = s1[77];
          double v3159_data = s1[107];
          double v3164_data = s1[137];
          double v3169_data = s1[167];
          double v3174_data = s1[197];
          double v3179_data = s1[227];
          double v3184_data = s1[257];
          double v3190_data = glb_m3[(v4543_lead + 640)];
          double v3237_data = v980_g ? (glb_m3[(v978_lead + 640)]) : (0.0);
          double v3284_data = glb_m3[(v4541_lead + 680)];
          double v3285_data = s1[18];
          double v3290_data = s1[48];
          double v3293_p = (((((((((((((((((v884_data * v890_data) + (v1028_data * v1034_data)) + (v1169_data * v1175_data)) + (v1310_data * v1316_data)) + (v1451_data * v1457_data)) + (v1592_data * v1598_data)) + (v1733_data * v1739_data)) + (v1874_data * v1880_data)) + (v2015_data * v2021_data)) + (v2156_data * v2162_data)) + (v2297_data * v2303_data)) + (v2438_data * v2444_data)) + (v2579_data * v2585_data)) + (v2720_data * v2726_data)) + (v2861_data * v2867_data)) + (v3002_data * v3008_data)) + (v3143_data * v3149_data)) + (v3284_data * v3290_data);
          double v3295_data = s1[78];
          double v3298_p = (((((((((((((((((v884_data * v895_data) + (v1028_data * v1039_data)) + (v1169_data * v1180_data)) + (v1310_data * v1321_data)) + (v1451_data * v1462_data)) + (v1592_data * v1603_data)) + (v1733_data * v1744_data)) + (v1874_data * v1885_data)) + (v2015_data * v2026_data)) + (v2156_data * v2167_data)) + (v2297_data * v2308_data)) + (v2438_data * v2449_data)) + (v2579_data * v2590_data)) + (v2720_data * v2731_data)) + (v2861_data * v2872_data)) + (v3002_data * v3013_data)) + (v3143_data * v3154_data)) + (v3284_data * v3295_data);
          double v3300_data = s1[108];
          double v3303_p = (((((((((((((((((v884_data * v900_data) + (v1028_data * v1044_data)) + (v1169_data * v1185_data)) + (v1310_data * v1326_data)) + (v1451_data * v1467_data)) + (v1592_data * v1608_data)) + (v1733_data * v1749_data)) + (v1874_data * v1890_data)) + (v2015_data * v2031_data)) + (v2156_data * v2172_data)) + (v2297_data * v2313_data)) + (v2438_data * v2454_data)) + (v2579_data * v2595_data)) + (v2720_data * v2736_data)) + (v2861_data * v2877_data)) + (v3002_data * v3018_data)) + (v3143_data * v3159_data)) + (v3284_data * v3300_data);
          double v3305_data = s1[138];
          double v3308_p = (((((((((((((((((v884_data * v905_data) + (v1028_data * v1049_data)) + (v1169_data * v1190_data)) + (v1310_data * v1331_data)) + (v1451_data * v1472_data)) + (v1592_data * v1613_data)) + (v1733_data * v1754_data)) + (v1874_data * v1895_data)) + (v2015_data * v2036_data)) + (v2156_data * v2177_data)) + (v2297_data * v2318_data)) + (v2438_data * v2459_data)) + (v2579_data * v2600_data)) + (v2720_data * v2741_data)) + (v2861_data * v2882_data)) + (v3002_data * v3023_data)) + (v3143_data * v3164_data)) + (v3284_data * v3305_data);
          double v3310_data = s1[168];
          double v3313_p = (((((((((((((((((v884_data * v910_data) + (v1028_data * v1054_data)) + (v1169_data * v1195_data)) + (v1310_data * v1336_data)) + (v1451_data * v1477_data)) + (v1592_data * v1618_data)) + (v1733_data * v1759_data)) + (v1874_data * v1900_data)) + (v2015_data * v2041_data)) + (v2156_data * v2182_data)) + (v2297_data * v2323_data)) + (v2438_data * v2464_data)) + (v2579_data * v2605_data)) + (v2720_data * v2746_data)) + (v2861_data * v2887_data)) + (v3002_data * v3028_data)) + (v3143_data * v3169_data)) + (v3284_data * v3310_data);
          double v3315_data = s1[198];
          double v3318_p = (((((((((((((((((v884_data * v915_data) + (v1028_data * v1059_data)) + (v1169_data * v1200_data)) + (v1310_data * v1341_data)) + (v1451_data * v1482_data)) + (v1592_data * v1623_data)) + (v1733_data * v1764_data)) + (v1874_data * v1905_data)) + (v2015_data * v2046_data)) + (v2156_data * v2187_data)) + (v2297_data * v2328_data)) + (v2438_data * v2469_data)) + (v2579_data * v2610_data)) + (v2720_data * v2751_data)) + (v2861_data * v2892_data)) + (v3002_data * v3033_data)) + (v3143_data * v3174_data)) + (v3284_data * v3315_data);
          double v3320_data = s1[228];
          double v3323_p = (((((((((((((((((v884_data * v920_data) + (v1028_data * v1064_data)) + (v1169_data * v1205_data)) + (v1310_data * v1346_data)) + (v1451_data * v1487_data)) + (v1592_data * v1628_data)) + (v1733_data * v1769_data)) + (v1874_data * v1910_data)) + (v2015_data * v2051_data)) + (v2156_data * v2192_data)) + (v2297_data * v2333_data)) + (v2438_data * v2474_data)) + (v2579_data * v2615_data)) + (v2720_data * v2756_data)) + (v2861_data * v2897_data)) + (v3002_data * v3038_data)) + (v3143_data * v3179_data)) + (v3284_data * v3320_data);
          double v3325_data = s1[258];
          double v3328_p = (((((((((((((((((v884_data * v925_data) + (v1028_data * v1069_data)) + (v1169_data * v1210_data)) + (v1310_data * v1351_data)) + (v1451_data * v1492_data)) + (v1592_data * v1633_data)) + (v1733_data * v1774_data)) + (v1874_data * v1915_data)) + (v2015_data * v2056_data)) + (v2156_data * v2197_data)) + (v2297_data * v2338_data)) + (v2438_data * v2479_data)) + (v2579_data * v2620_data)) + (v2720_data * v2761_data)) + (v2861_data * v2902_data)) + (v3002_data * v3043_data)) + (v3143_data * v3184_data)) + (v3284_data * v3325_data);
          double v3331_data = glb_m3[(v4543_lead + 680)];
          double v3335_p = (((((((((((((((((v932_data * v885_data) + (v1075_data * v1029_data)) + (v1216_data * v1170_data)) + (v1357_data * v1311_data)) + (v1498_data * v1452_data)) + (v1639_data * v1593_data)) + (v1780_data * v1734_data)) + (v1921_data * v1875_data)) + (v2062_data * v2016_data)) + (v2203_data * v2157_data)) + (v2344_data * v2298_data)) + (v2485_data * v2439_data)) + (v2626_data * v2580_data)) + (v2767_data * v2721_data)) + (v2908_data * v2862_data)) + (v3049_data * v3003_data)) + (v3190_data * v3144_data)) + (v3331_data * v3285_data);
          double v3340_p = (((((((((((((((((v932_data * v890_data) + (v1075_data * v1034_data)) + (v1216_data * v1175_data)) + (v1357_data * v1316_data)) + (v1498_data * v1457_data)) + (v1639_data * v1598_data)) + (v1780_data * v1739_data)) + (v1921_data * v1880_data)) + (v2062_data * v2021_data)) + (v2203_data * v2162_data)) + (v2344_data * v2303_data)) + (v2485_data * v2444_data)) + (v2626_data * v2585_data)) + (v2767_data * v2726_data)) + (v2908_data * v2867_data)) + (v3049_data * v3008_data)) + (v3190_data * v3149_data)) + (v3331_data * v3290_data);
          double v3345_p = (((((((((((((((((v932_data * v895_data) + (v1075_data * v1039_data)) + (v1216_data * v1180_data)) + (v1357_data * v1321_data)) + (v1498_data * v1462_data)) + (v1639_data * v1603_data)) + (v1780_data * v1744_data)) + (v1921_data * v1885_data)) + (v2062_data * v2026_data)) + (v2203_data * v2167_data)) + (v2344_data * v2308_data)) + (v2485_data * v2449_data)) + (v2626_data * v2590_data)) + (v2767_data * v2731_data)) + (v2908_data * v2872_data)) + (v3049_data * v3013_data)) + (v3190_data * v3154_data)) + (v3331_data * v3295_data);
          double v3350_p = (((((((((((((((((v932_data * v900_data) + (v1075_data * v1044_data)) + (v1216_data * v1185_data)) + (v1357_data * v1326_data)) + (v1498_data * v1467_data)) + (v1639_data * v1608_data)) + (v1780_data * v1749_data)) + (v1921_data * v1890_data)) + (v2062_data * v2031_data)) + (v2203_data * v2172_data)) + (v2344_data * v2313_data)) + (v2485_data * v2454_data)) + (v2626_data * v2595_data)) + (v2767_data * v2736_data)) + (v2908_data * v2877_data)) + (v3049_data * v3018_data)) + (v3190_data * v3159_data)) + (v3331_data * v3300_data);
          double v3355_p = (((((((((((((((((v932_data * v905_data) + (v1075_data * v1049_data)) + (v1216_data * v1190_data)) + (v1357_data * v1331_data)) + (v1498_data * v1472_data)) + (v1639_data * v1613_data)) + (v1780_data * v1754_data)) + (v1921_data * v1895_data)) + (v2062_data * v2036_data)) + (v2203_data * v2177_data)) + (v2344_data * v2318_data)) + (v2485_data * v2459_data)) + (v2626_data * v2600_data)) + (v2767_data * v2741_data)) + (v2908_data * v2882_data)) + (v3049_data * v3023_data)) + (v3190_data * v3164_data)) + (v3331_data * v3305_data);
          double v3360_p = (((((((((((((((((v932_data * v910_data) + (v1075_data * v1054_data)) + (v1216_data * v1195_data)) + (v1357_data * v1336_data)) + (v1498_data * v1477_data)) + (v1639_data * v1618_data)) + (v1780_data * v1759_data)) + (v1921_data * v1900_data)) + (v2062_data * v2041_data)) + (v2203_data * v2182_data)) + (v2344_data * v2323_data)) + (v2485_data * v2464_data)) + (v2626_data * v2605_data)) + (v2767_data * v2746_data)) + (v2908_data * v2887_data)) + (v3049_data * v3028_data)) + (v3190_data * v3169_data)) + (v3331_data * v3310_data);
          double v3365_p = (((((((((((((((((v932_data * v915_data) + (v1075_data * v1059_data)) + (v1216_data * v1200_data)) + (v1357_data * v1341_data)) + (v1498_data * v1482_data)) + (v1639_data * v1623_data)) + (v1780_data * v1764_data)) + (v1921_data * v1905_data)) + (v2062_data * v2046_data)) + (v2203_data * v2187_data)) + (v2344_data * v2328_data)) + (v2485_data * v2469_data)) + (v2626_data * v2610_data)) + (v2767_data * v2751_data)) + (v2908_data * v2892_data)) + (v3049_data * v3033_data)) + (v3190_data * v3174_data)) + (v3331_data * v3315_data);
          double v3370_p = (((((((((((((((((v932_data * v920_data) + (v1075_data * v1064_data)) + (v1216_data * v1205_data)) + (v1357_data * v1346_data)) + (v1498_data * v1487_data)) + (v1639_data * v1628_data)) + (v1780_data * v1769_data)) + (v1921_data * v1910_data)) + (v2062_data * v2051_data)) + (v2203_data * v2192_data)) + (v2344_data * v2333_data)) + (v2485_data * v2474_data)) + (v2626_data * v2615_data)) + (v2767_data * v2756_data)) + (v2908_data * v2897_data)) + (v3049_data * v3038_data)) + (v3190_data * v3179_data)) + (v3331_data * v3320_data);
          double v3375_p = (((((((((((((((((v932_data * v925_data) + (v1075_data * v1069_data)) + (v1216_data * v1210_data)) + (v1357_data * v1351_data)) + (v1498_data * v1492_data)) + (v1639_data * v1633_data)) + (v1780_data * v1774_data)) + (v1921_data * v1915_data)) + (v2062_data * v2056_data)) + (v2203_data * v2197_data)) + (v2344_data * v2338_data)) + (v2485_data * v2479_data)) + (v2626_data * v2620_data)) + (v2767_data * v2761_data)) + (v2908_data * v2902_data)) + (v3049_data * v3043_data)) + (v3190_data * v3184_data)) + (v3331_data * v3325_data);
          double v3378_data = v980_g ? (glb_m3[(v978_lead + 680)]) : (0.0);
          double v3382_p = (((((((((((((((((v981_data * v885_data) + (v1122_data * v1029_data)) + (v1263_data * v1170_data)) + (v1404_data * v1311_data)) + (v1545_data * v1452_data)) + (v1686_data * v1593_data)) + (v1827_data * v1734_data)) + (v1968_data * v1875_data)) + (v2109_data * v2016_data)) + (v2250_data * v2157_data)) + (v2391_data * v2298_data)) + (v2532_data * v2439_data)) + (v2673_data * v2580_data)) + (v2814_data * v2721_data)) + (v2955_data * v2862_data)) + (v3096_data * v3003_data)) + (v3237_data * v3144_data)) + (v3378_data * v3285_data);
          double v3387_p = (((((((((((((((((v981_data * v890_data) + (v1122_data * v1034_data)) + (v1263_data * v1175_data)) + (v1404_data * v1316_data)) + (v1545_data * v1457_data)) + (v1686_data * v1598_data)) + (v1827_data * v1739_data)) + (v1968_data * v1880_data)) + (v2109_data * v2021_data)) + (v2250_data * v2162_data)) + (v2391_data * v2303_data)) + (v2532_data * v2444_data)) + (v2673_data * v2585_data)) + (v2814_data * v2726_data)) + (v2955_data * v2867_data)) + (v3096_data * v3008_data)) + (v3237_data * v3149_data)) + (v3378_data * v3290_data);
          double v3392_p = (((((((((((((((((v981_data * v895_data) + (v1122_data * v1039_data)) + (v1263_data * v1180_data)) + (v1404_data * v1321_data)) + (v1545_data * v1462_data)) + (v1686_data * v1603_data)) + (v1827_data * v1744_data)) + (v1968_data * v1885_data)) + (v2109_data * v2026_data)) + (v2250_data * v2167_data)) + (v2391_data * v2308_data)) + (v2532_data * v2449_data)) + (v2673_data * v2590_data)) + (v2814_data * v2731_data)) + (v2955_data * v2872_data)) + (v3096_data * v3013_data)) + (v3237_data * v3154_data)) + (v3378_data * v3295_data);
          double v3397_p = (((((((((((((((((v981_data * v900_data) + (v1122_data * v1044_data)) + (v1263_data * v1185_data)) + (v1404_data * v1326_data)) + (v1545_data * v1467_data)) + (v1686_data * v1608_data)) + (v1827_data * v1749_data)) + (v1968_data * v1890_data)) + (v2109_data * v2031_data)) + (v2250_data * v2172_data)) + (v2391_data * v2313_data)) + (v2532_data * v2454_data)) + (v2673_data * v2595_data)) + (v2814_data * v2736_data)) + (v2955_data * v2877_data)) + (v3096_data * v3018_data)) + (v3237_data * v3159_data)) + (v3378_data * v3300_data);
          double v3402_p = (((((((((((((((((v981_data * v905_data) + (v1122_data * v1049_data)) + (v1263_data * v1190_data)) + (v1404_data * v1331_data)) + (v1545_data * v1472_data)) + (v1686_data * v1613_data)) + (v1827_data * v1754_data)) + (v1968_data * v1895_data)) + (v2109_data * v2036_data)) + (v2250_data * v2177_data)) + (v2391_data * v2318_data)) + (v2532_data * v2459_data)) + (v2673_data * v2600_data)) + (v2814_data * v2741_data)) + (v2955_data * v2882_data)) + (v3096_data * v3023_data)) + (v3237_data * v3164_data)) + (v3378_data * v3305_data);
          double v3407_p = (((((((((((((((((v981_data * v910_data) + (v1122_data * v1054_data)) + (v1263_data * v1195_data)) + (v1404_data * v1336_data)) + (v1545_data * v1477_data)) + (v1686_data * v1618_data)) + (v1827_data * v1759_data)) + (v1968_data * v1900_data)) + (v2109_data * v2041_data)) + (v2250_data * v2182_data)) + (v2391_data * v2323_data)) + (v2532_data * v2464_data)) + (v2673_data * v2605_data)) + (v2814_data * v2746_data)) + (v2955_data * v2887_data)) + (v3096_data * v3028_data)) + (v3237_data * v3169_data)) + (v3378_data * v3310_data);
          double v3412_p = (((((((((((((((((v981_data * v915_data) + (v1122_data * v1059_data)) + (v1263_data * v1200_data)) + (v1404_data * v1341_data)) + (v1545_data * v1482_data)) + (v1686_data * v1623_data)) + (v1827_data * v1764_data)) + (v1968_data * v1905_data)) + (v2109_data * v2046_data)) + (v2250_data * v2187_data)) + (v2391_data * v2328_data)) + (v2532_data * v2469_data)) + (v2673_data * v2610_data)) + (v2814_data * v2751_data)) + (v2955_data * v2892_data)) + (v3096_data * v3033_data)) + (v3237_data * v3174_data)) + (v3378_data * v3315_data);
          double v3417_p = (((((((((((((((((v981_data * v920_data) + (v1122_data * v1064_data)) + (v1263_data * v1205_data)) + (v1404_data * v1346_data)) + (v1545_data * v1487_data)) + (v1686_data * v1628_data)) + (v1827_data * v1769_data)) + (v1968_data * v1910_data)) + (v2109_data * v2051_data)) + (v2250_data * v2192_data)) + (v2391_data * v2333_data)) + (v2532_data * v2474_data)) + (v2673_data * v2615_data)) + (v2814_data * v2756_data)) + (v2955_data * v2897_data)) + (v3096_data * v3038_data)) + (v3237_data * v3179_data)) + (v3378_data * v3320_data);
          double v3422_p = (((((((((((((((((v981_data * v925_data) + (v1122_data * v1069_data)) + (v1263_data * v1210_data)) + (v1404_data * v1351_data)) + (v1545_data * v1492_data)) + (v1686_data * v1633_data)) + (v1827_data * v1774_data)) + (v1968_data * v1915_data)) + (v2109_data * v2056_data)) + (v2250_data * v2197_data)) + (v2391_data * v2338_data)) + (v2532_data * v2479_data)) + (v2673_data * v2620_data)) + (v2814_data * v2761_data)) + (v2955_data * v2902_data)) + (v3096_data * v3043_data)) + (v3237_data * v3184_data)) + (v3378_data * v3325_data);
          double v3425_data = glb_m3[(v4541_lead + 720)];
          double v3426_data = s1[19];
          ir2[0] = (((((((((((((((((((v884_data * v885_data) + (v1028_data * v1029_data)) + (v1169_data * v1170_data)) + (v1310_data * v1311_data)) + (v1451_data * v1452_data)) + (v1592_data * v1593_data)) + (v1733_data * v1734_data)) + (v1874_data * v1875_data)) + (v2015_data * v2016_data)) + (v2156_data * v2157_data)) + (v2297_data * v2298_data)) + (v2438_data * v2439_data)) + (v2579_data * v2580_data)) + (v2720_data * v2721_data)) + (v2861_data * v2862_data)) + (v3002_data * v3003_data)) + (v3143_data * v3144_data)) + (v3284_data * v3285_data)) + (v3425_data * v3426_data));
          double v3431_data = s1[49];
          ir2[3] = (v3293_p + (v3425_data * v3431_data));
          double v3436_data = s1[79];
          ir2[6] = (v3298_p + (v3425_data * v3436_data));
          double v3441_data = s1[109];
          ir2[9] = (v3303_p + (v3425_data * v3441_data));
          double v3446_data = s1[139];
          ir2[12] = (v3308_p + (v3425_data * v3446_data));
          double v3451_data = s1[169];
          ir2[15] = (v3313_p + (v3425_data * v3451_data));
          double v3456_data = s1[199];
          ir2[18] = (v3318_p + (v3425_data * v3456_data));
          double v3461_data = s1[229];
          ir2[21] = (v3323_p + (v3425_data * v3461_data));
          double v3466_data = s1[259];
          ir2[24] = (v3328_p + (v3425_data * v3466_data));
          double v3472_data = glb_m3[(v4543_lead + 720)];
          double v3519_data = v980_g ? (glb_m3[(v978_lead + 720)]) : (0.0);
          double v3566_data = glb_m3[(v4543_lead + 744)];
          double v3567_data = s1[20];
          double v3572_data = s1[50];
          double v3577_data = s1[80];
          double v3582_data = s1[110];
          double v3587_data = s1[140];
          double v3592_data = s1[170];
          double v3597_data = s1[200];
          double v3602_data = s1[230];
          double v3607_data = s1[260];
          double v3613_data = v980_g ? (glb_m3[(v978_lead + 744)]) : (0.0);
          double v3660_data = glb_m3[(v4543_lead + 768)];
          double v3661_data = s1[21];
          double v3666_data = s1[51];
          double v3671_data = s1[81];
          double v3676_data = s1[111];
          double v3681_data = s1[141];
          double v3686_data = s1[171];
          double v3691_data = s1[201];
          double v3696_data = s1[231];
          double v3701_data = s1[261];
          double v3707_data = v980_g ? (glb_m3[(v978_lead + 768)]) : (0.0);
          double v3754_data = glb_m3[(v4543_lead + 792)];
          double v3755_data = s1[22];
          double v3760_data = s1[52];
          double v3765_data = s1[82];
          double v3770_data = s1[112];
          double v3775_data = s1[142];
          double v3780_data = s1[172];
          double v3785_data = s1[202];
          double v3790_data = s1[232];
          double v3795_data = s1[262];
          double v3801_data = v980_g ? (glb_m3[(v978_lead + 792)]) : (0.0);
          double v3848_data = glb_m3[(v4543_lead + 816)];
          double v3849_data = s1[23];
          double v3854_data = s1[53];
          double v3859_data = s1[83];
          double v3864_data = s1[113];
          double v3869_data = s1[143];
          double v3874_data = s1[173];
          double v3879_data = s1[203];
          double v3884_data = s1[233];
          double v3889_data = s1[263];
          double v3895_data = v980_g ? (glb_m3[(v978_lead + 816)]) : (0.0);
          double v3942_data = glb_m3[(v4543_lead + 840)];
          double v3943_data = s1[24];
          double v3948_data = s1[54];
          double v3953_data = s1[84];
          double v3958_data = s1[114];
          double v3963_data = s1[144];
          double v3968_data = s1[174];
          double v3973_data = s1[204];
          double v3978_data = s1[234];
          double v3983_data = s1[264];
          double v3989_data = v980_g ? (glb_m3[(v978_lead + 840)]) : (0.0);
          double v4036_data = glb_m3[(v4543_lead + 864)];
          double v4037_data = s1[25];
          double v4042_data = s1[55];
          double v4047_data = s1[85];
          double v4052_data = s1[115];
          double v4057_data = s1[145];
          double v4062_data = s1[175];
          double v4067_data = s1[205];
          double v4072_data = s1[235];
          double v4077_data = s1[265];
          double v4083_data = v980_g ? (glb_m3[(v978_lead + 864)]) : (0.0);
          double v4130_data = glb_m3[(v4543_lead + 888)];
          double v4131_data = s1[26];
          double v4136_data = s1[56];
          double v4141_data = s1[86];
          double v4146_data = s1[116];
          double v4151_data = s1[146];
          double v4156_data = s1[176];
          double v4161_data = s1[206];
          double v4166_data = s1[236];
          double v4171_data = s1[266];
          double v4177_data = v980_g ? (glb_m3[(v978_lead + 888)]) : (0.0);
          double v4186_p = (((((((v3387_p + (v3519_data * v3431_data)) + (v3613_data * v3572_data)) + (v3707_data * v3666_data)) + (v3801_data * v3760_data)) + (v3895_data * v3854_data)) + (v3989_data * v3948_data)) + (v4083_data * v4042_data)) + (v4177_data * v4136_data);
          double v4191_p = (((((((v3392_p + (v3519_data * v3436_data)) + (v3613_data * v3577_data)) + (v3707_data * v3671_data)) + (v3801_data * v3765_data)) + (v3895_data * v3859_data)) + (v3989_data * v3953_data)) + (v4083_data * v4047_data)) + (v4177_data * v4141_data);
          double v4196_p = (((((((v3397_p + (v3519_data * v3441_data)) + (v3613_data * v3582_data)) + (v3707_data * v3676_data)) + (v3801_data * v3770_data)) + (v3895_data * v3864_data)) + (v3989_data * v3958_data)) + (v4083_data * v4052_data)) + (v4177_data * v4146_data);
          double v4201_p = (((((((v3402_p + (v3519_data * v3446_data)) + (v3613_data * v3587_data)) + (v3707_data * v3681_data)) + (v3801_data * v3775_data)) + (v3895_data * v3869_data)) + (v3989_data * v3963_data)) + (v4083_data * v4057_data)) + (v4177_data * v4151_data);
          double v4206_p = (((((((v3407_p + (v3519_data * v3451_data)) + (v3613_data * v3592_data)) + (v3707_data * v3686_data)) + (v3801_data * v3780_data)) + (v3895_data * v3874_data)) + (v3989_data * v3968_data)) + (v4083_data * v4062_data)) + (v4177_data * v4156_data);
          double v4211_p = (((((((v3412_p + (v3519_data * v3456_data)) + (v3613_data * v3597_data)) + (v3707_data * v3691_data)) + (v3801_data * v3785_data)) + (v3895_data * v3879_data)) + (v3989_data * v3973_data)) + (v4083_data * v4067_data)) + (v4177_data * v4161_data);
          double v4216_p = (((((((v3417_p + (v3519_data * v3461_data)) + (v3613_data * v3602_data)) + (v3707_data * v3696_data)) + (v3801_data * v3790_data)) + (v3895_data * v3884_data)) + (v3989_data * v3978_data)) + (v4083_data * v4072_data)) + (v4177_data * v4166_data);
          double v4221_p = (((((((v3422_p + (v3519_data * v3466_data)) + (v3613_data * v3607_data)) + (v3707_data * v3701_data)) + (v3801_data * v3795_data)) + (v3895_data * v3889_data)) + (v3989_data * v3983_data)) + (v4083_data * v4077_data)) + (v4177_data * v4171_data);
          double v4224_data = glb_m3[(v4543_lead + 912)];
          double v4225_data = s1[27];
          double v4228_p = ((((((((v3335_p + (v3472_data * v3426_data)) + (v3566_data * v3567_data)) + (v3660_data * v3661_data)) + (v3754_data * v3755_data)) + (v3848_data * v3849_data)) + (v3942_data * v3943_data)) + (v4036_data * v4037_data)) + (v4130_data * v4131_data)) + (v4224_data * v4225_data);
          double v4230_data = s1[57];
          double v4233_p = ((((((((v3340_p + (v3472_data * v3431_data)) + (v3566_data * v3572_data)) + (v3660_data * v3666_data)) + (v3754_data * v3760_data)) + (v3848_data * v3854_data)) + (v3942_data * v3948_data)) + (v4036_data * v4042_data)) + (v4130_data * v4136_data)) + (v4224_data * v4230_data);
          double v4235_data = s1[87];
          double v4238_p = ((((((((v3345_p + (v3472_data * v3436_data)) + (v3566_data * v3577_data)) + (v3660_data * v3671_data)) + (v3754_data * v3765_data)) + (v3848_data * v3859_data)) + (v3942_data * v3953_data)) + (v4036_data * v4047_data)) + (v4130_data * v4141_data)) + (v4224_data * v4235_data);
          double v4240_data = s1[117];
          double v4243_p = ((((((((v3350_p + (v3472_data * v3441_data)) + (v3566_data * v3582_data)) + (v3660_data * v3676_data)) + (v3754_data * v3770_data)) + (v3848_data * v3864_data)) + (v3942_data * v3958_data)) + (v4036_data * v4052_data)) + (v4130_data * v4146_data)) + (v4224_data * v4240_data);
          double v4245_data = s1[147];
          double v4248_p = ((((((((v3355_p + (v3472_data * v3446_data)) + (v3566_data * v3587_data)) + (v3660_data * v3681_data)) + (v3754_data * v3775_data)) + (v3848_data * v3869_data)) + (v3942_data * v3963_data)) + (v4036_data * v4057_data)) + (v4130_data * v4151_data)) + (v4224_data * v4245_data);
          double v4250_data = s1[177];
          double v4253_p = ((((((((v3360_p + (v3472_data * v3451_data)) + (v3566_data * v3592_data)) + (v3660_data * v3686_data)) + (v3754_data * v3780_data)) + (v3848_data * v3874_data)) + (v3942_data * v3968_data)) + (v4036_data * v4062_data)) + (v4130_data * v4156_data)) + (v4224_data * v4250_data);
          double v4255_data = s1[207];
          double v4258_p = ((((((((v3365_p + (v3472_data * v3456_data)) + (v3566_data * v3597_data)) + (v3660_data * v3691_data)) + (v3754_data * v3785_data)) + (v3848_data * v3879_data)) + (v3942_data * v3973_data)) + (v4036_data * v4067_data)) + (v4130_data * v4161_data)) + (v4224_data * v4255_data);
          double v4260_data = s1[237];
          double v4263_p = ((((((((v3370_p + (v3472_data * v3461_data)) + (v3566_data * v3602_data)) + (v3660_data * v3696_data)) + (v3754_data * v3790_data)) + (v3848_data * v3884_data)) + (v3942_data * v3978_data)) + (v4036_data * v4072_data)) + (v4130_data * v4166_data)) + (v4224_data * v4260_data);
          double v4265_data = s1[267];
          double v4268_p = ((((((((v3375_p + (v3472_data * v3466_data)) + (v3566_data * v3607_data)) + (v3660_data * v3701_data)) + (v3754_data * v3795_data)) + (v3848_data * v3889_data)) + (v3942_data * v3983_data)) + (v4036_data * v4077_data)) + (v4130_data * v4171_data)) + (v4224_data * v4265_data);
          double v4271_data = v980_g ? (glb_m3[(v978_lead + 912)]) : (0.0);
          double v4275_p = ((((((((v3382_p + (v3519_data * v3426_data)) + (v3613_data * v3567_data)) + (v3707_data * v3661_data)) + (v3801_data * v3755_data)) + (v3895_data * v3849_data)) + (v3989_data * v3943_data)) + (v4083_data * v4037_data)) + (v4177_data * v4131_data)) + (v4271_data * v4225_data);
          // r0 = load{g>r}(wrap_glb_m0);
          #pragma unroll
          for (int32_t v4581_i0 = 0; v4581_i0 < 1; ++v4581_i0) {
            int32_t v4583_lead = v4541_lead + (v4581_i0 * 16);
            #pragma unroll
            for (int32_t v4584_i1 = 0; v4584_i1 < 9; ++v4584_i1) {
              double v4587_data = __ldcg(&wrap_glb_m0[(v4583_lead + (v4584_i1 * 30))]);
              r0[(v4581_i0 + (v4584_i1 * 2))] = v4587_data;
            }
          }
          #pragma unroll
          for (int32_t v4590_i1 = 0; v4590_i1 < 9; ++v4590_i1) {
            double v4593_data = v4544_g ? (__ldcg(&wrap_glb_m0[(v4543_lead + (v4590_i1 * 30))])) : (0.0);
            r0[(1 + (v4590_i1 * 2))] = v4593_data;
          }
          double v4318_data = glb_m3[(v4543_lead + 936)];
          double v4319_data = s1[28];
          double v4324_data = s1[58];
          double v4327_p = v4233_p + (v4318_data * v4324_data);
          double v4329_data = s1[88];
          double v4332_p = v4238_p + (v4318_data * v4329_data);
          double v4334_data = s1[118];
          double v4337_p = v4243_p + (v4318_data * v4334_data);
          double v4339_data = s1[148];
          double v4342_p = v4248_p + (v4318_data * v4339_data);
          double v4344_data = s1[178];
          double v4347_p = v4253_p + (v4318_data * v4344_data);
          double v4349_data = s1[208];
          double v4352_p = v4258_p + (v4318_data * v4349_data);
          double v4354_data = s1[238];
          double v4357_p = v4263_p + (v4318_data * v4354_data);
          double v4359_data = s1[268];
          double v4362_p = v4268_p + (v4318_data * v4359_data);
          double v4365_data = v980_g ? (glb_m3[(v978_lead + 936)]) : (0.0);
          double v4369_p = v4275_p + (v4365_data * v4319_data);
          double v4374_p = (v4186_p + (v4271_data * v4230_data)) + (v4365_data * v4324_data);
          double v4379_p = (v4191_p + (v4271_data * v4235_data)) + (v4365_data * v4329_data);
          double v4384_p = (v4196_p + (v4271_data * v4240_data)) + (v4365_data * v4334_data);
          double v4389_p = (v4201_p + (v4271_data * v4245_data)) + (v4365_data * v4339_data);
          double v4394_p = (v4206_p + (v4271_data * v4250_data)) + (v4365_data * v4344_data);
          double v4399_p = (v4211_p + (v4271_data * v4255_data)) + (v4365_data * v4349_data);
          double v4404_p = (v4216_p + (v4271_data * v4260_data)) + (v4365_data * v4354_data);
          double v4409_p = (v4221_p + (v4271_data * v4265_data)) + (v4365_data * v4359_data);
          double v4412_data = glb_m3[(v4543_lead + 960)];
          double v4413_data = s1[29];
          ir2[1] = ((v4228_p + (v4318_data * v4319_data)) + (v4412_data * v4413_data));
          double v4418_data = s1[59];
          ir2[4] = (v4327_p + (v4412_data * v4418_data));
          double v4423_data = s1[89];
          ir2[7] = (v4332_p + (v4412_data * v4423_data));
          double v4428_data = s1[119];
          ir2[10] = (v4337_p + (v4412_data * v4428_data));
          double v4433_data = s1[149];
          ir2[13] = (v4342_p + (v4412_data * v4433_data));
          double v4438_data = s1[179];
          ir2[16] = (v4347_p + (v4412_data * v4438_data));
          double v4443_data = s1[209];
          ir2[19] = (v4352_p + (v4412_data * v4443_data));
          double v4448_data = s1[239];
          ir2[22] = (v4357_p + (v4412_data * v4448_data));
          double v4453_data = s1[269];
          ir2[25] = (v4362_p + (v4412_data * v4453_data));
          double v4459_data = v980_g ? (glb_m3[(v978_lead + 960)]) : (0.0);
          ir2[2] = (v4369_p + (v4459_data * v4413_data));
          ir2[5] = (v4374_p + (v4459_data * v4418_data));
          ir2[8] = (v4379_p + (v4459_data * v4423_data));
          ir2[11] = (v4384_p + (v4459_data * v4428_data));
          ir2[14] = (v4389_p + (v4459_data * v4433_data));
          ir2[17] = (v4394_p + (v4459_data * v4438_data));
          ir2[20] = (v4399_p + (v4459_data * v4443_data));
          ir2[23] = (v4404_p + (v4459_data * v4448_data));
          ir2[26] = (v4409_p + (v4459_data * v4453_data));
          // r2 = ir2
          // glb_m2 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v4514_i0 = 0; v4514_i0 < 2; ++v4514_i0) {
            int32_t v4520_lead = v4541_lead + (v4514_i0 * 16);
            #pragma unroll
            for (int32_t v4515_i1 = 0; v4515_i1 < 9; ++v4515_i1) {
              double v4518_data = ir2[(v4514_i0 + (v4515_i1 * 3))];
              glb_m2[(v4520_lead + (v4515_i1 * 40))] = v4518_data;
            }
          }
          if (v980_g) {
            #pragma unroll
            for (int32_t v4524_i1 = 0; v4524_i1 < 9; ++v4524_i1) {
              double v4527_data = ir2[((v4524_i1 * 3) + 2)];
              glb_m2[(v978_lead + (v4524_i1 * 40))] = v4527_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
        else {
          __pipeline_wait_prior(0);
          // r0 = load{g>r}(wrap_glb_m0);
          #pragma unroll
          for (int32_t v4551_i0 = 0; v4551_i0 < 1; ++v4551_i0) {
            int32_t v4553_lead = v4541_lead + (v4551_i0 * 16);
            #pragma unroll
            for (int32_t v4554_i1 = 0; v4554_i1 < 9; ++v4554_i1) {
              double v4557_data = __ldcg(&wrap_glb_m0[(v4553_lead + (v4554_i1 * 30))]);
              r0[(v4551_i0 + (v4554_i1 * 2))] = v4557_data;
            }
          }
          #pragma unroll
          for (int32_t v4560_i1 = 0; v4560_i1 < 9; ++v4560_i1) {
            double v4563_data = v4544_g ? (__ldcg(&wrap_glb_m0[(v4543_lead + (v4560_i1 * 30))])) : (0.0);
            r0[(1 + (v4560_i1 * 2))] = v4563_data;
          }
        }
        // s0 = load{g>s}(wrap_glb_m1[0, 1])
        #pragma unroll
        for (int32_t v4605_i = 0; v4605_i < 5; ++v4605_i) {
          int32_t v4607_a = threadIdx.x + (v4605_i * 16);
          __pipeline_memcpy_async(&s0[v4607_a], &wrap_glb_m1[v4607_a], 8);
        }
        if (v4597_g) {
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 80], &wrap_glb_m1[0 + 0 + 1 * threadIdx.x + 80], 8);
        }
        __pipeline_commit();
        v4537_flagWord = v4538_flagWordNext;
        v4538_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
  }
}

