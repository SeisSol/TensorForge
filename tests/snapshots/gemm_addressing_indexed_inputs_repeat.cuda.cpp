// === base name ===
kernel_5c6c0654605a0737

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_5c6c0654605a0737 = {{16, 8, 1}, 16, 12, 1, 8, 5632, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_5c6c0654605a0737(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_5c6c0654605a0737(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_5c6c0654605a0737(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_5c6c0654605a0737, block.x * block.y * block.z, 1408 * sizeof(float));
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
void launcher_kernel_5c6c0654605a0737(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_5c6c0654605a0737(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_5c6c0654605a0737, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  const unsigned * elementIndicesArg = (const unsigned *)elementIndices;
  const unsigned * flags0Arg = (const unsigned *)flags0;
  kernel_kernel_5c6c0654605a0737<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m1_stride, m2, m2_extraOffset, numElements0, elementIndicesArg, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_5c6c0654605a0737(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * __restrict__ elementIndices, const unsigned * __restrict__ flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 8 per block = block 16x8x1, 5632 B shared, occupancy grid
    // operands:
    //   m0 12×7(12×7) {0..12}×{0..7} strided
    //   m1 12×20(12×20) {0..12}×{0..20} indexed(stride=runtime)
    //   m2 20×7(20×7) {0..20}×{0..7} indexed
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1408}],"shared_bytes":5632,"shared_elements":1408,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,7]],"name":"m0","ordered":false,"parts":1,"shape":[12,7],"variant":false},{"addressing":"indexed(stride=runtime)","alias":"A","bbox":[[0,0],[12,20]],"name":"m1","ordered":false,"parts":1,"shape":[12,20],"variant":false},{"addressing":"indexed","alias":"B","bbox":[[0,0],[20,7]],"name":"m2","ordered":false,"parts":1,"shape":[20,7],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,7]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,7]},"kind":"multilinear","ops":[{"addressing":"indexed(stride=runtime)","bbox":[[0,0],[12,20]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,20]},{"addressing":"indexed","bbox":[[0,0],[20,7]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,7]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[176 * threadIdx.y + 0];
      size_t v3_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < numElements0) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      float * __restrict__ s0 = &localShrMem0[0];
      float r0[20]{};
      const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
      size_t v753_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v753_flagSecondAhead < numElements0) ? v753_flagSecondAhead : v3_batchId1)];
      const bool allowed_peel = static_cast<bool>(flagWordFirst);
      const size_t peel_elementIndex = static_cast<size_t>(elementIndices[v3_batchId1]);
      const float *const __restrict__ peel_glb_m1 = &m1[peel_elementIndex * m1_stride + 0 + m1_extraOffset];
      int32_t v764_lead = threadIdx.x % 16;
      bool v765_g = v764_lead < 12;
      bool v783_g = allowed_peel && v765_g;
      // r0 = load{g>r}(peel_glb_m1);
      #pragma unroll
      for (int32_t v775_i1 = 0; v775_i1 < 20; ++v775_i1) {
        float v778_data = v783_g ? (__ldcg(&peel_glb_m1[(v764_lead + (v775_i1 * 12))])) : (0.0f);
        r0[v775_i1] = v778_data;
      }
      const float *const __restrict__ peel_glb_m2 = &m2[peel_elementIndex * 140 + 0 + m2_extraOffset];
      bool v787_g = threadIdx.x < 12;
      // s0 = load{g>s}(peel_glb_m2[0, 1])
      #pragma unroll
      for (int32_t v790_i = 0; v790_i < 8; ++v790_i) {
        int32_t v792_a = threadIdx.x + (v790_i * 16);
        if (allowed_peel) {
          __pipeline_memcpy_async(&s0[v792_a], &peel_glb_m2[v792_a], 4);
        }
      }
      if (v787_g) {
        if (allowed_peel) {
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 128], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 128], 4);
        }
      }
      __pipeline_commit();
      uint32_t v757_flagWord = flagWordFirst;
      uint32_t v758_flagWordNext = flagWordSecond;
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const size_t wrap_elementIndex = static_cast<size_t>(elementIndices[v11_batchId1]);
        const float *const __restrict__ wrap_glb_m1 = &m1[wrap_elementIndex * m1_stride + 0 + m1_extraOffset];
        const float *const __restrict__ wrap_glb_m2 = &m2[wrap_elementIndex * 140 + 0 + m2_extraOffset];
        size_t v806_flagAheadAhead = v11_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v806_flagAheadAhead < numElements0) ? v806_flagAheadAhead : v11_batchId1)];
        const bool allowed = static_cast<bool>(v757_flagWord);
        const bool allowed_next = static_cast<bool>(v758_flagWordNext);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 84 + 0 + m0_extraOffset];
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[7]{};
          // ir1 = +(r0 * s0)
          // [(0, 12), (0, 7)] [(0, 20)]
          float ir1[7]{};
          float v40_data = r0[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v41_data = s0[0];
          float v46_data = s0[20];
          float v51_data = s0[40];
          float v56_data = s0[60];
          float v61_data = s0[80];
          float v66_data = s0[100];
          float v71_data = s0[120];
          float v75_data = r0[1];
          float v76_data = s0[1];
          float v81_data = s0[21];
          float v86_data = s0[41];
          float v91_data = s0[61];
          float v96_data = s0[81];
          float v101_data = s0[101];
          float v106_data = s0[121];
          float v110_data = r0[2];
          float v111_data = s0[2];
          float v116_data = s0[22];
          float v121_data = s0[42];
          float v126_data = s0[62];
          float v131_data = s0[82];
          float v136_data = s0[102];
          float v141_data = s0[122];
          float v145_data = r0[3];
          float v146_data = s0[3];
          float v151_data = s0[23];
          float v156_data = s0[43];
          float v161_data = s0[63];
          float v166_data = s0[83];
          float v171_data = s0[103];
          float v176_data = s0[123];
          float v180_data = r0[4];
          float v181_data = s0[4];
          float v186_data = s0[24];
          float v191_data = s0[44];
          float v196_data = s0[64];
          float v201_data = s0[84];
          float v206_data = s0[104];
          float v211_data = s0[124];
          float v215_data = r0[5];
          float v216_data = s0[5];
          float v221_data = s0[25];
          float v226_data = s0[45];
          float v231_data = s0[65];
          float v236_data = s0[85];
          float v241_data = s0[105];
          float v246_data = s0[125];
          float v250_data = r0[6];
          float v251_data = s0[6];
          float v256_data = s0[26];
          float v261_data = s0[46];
          float v266_data = s0[66];
          float v271_data = s0[86];
          float v276_data = s0[106];
          float v281_data = s0[126];
          float v285_data = r0[7];
          float v286_data = s0[7];
          float v291_data = s0[27];
          float v296_data = s0[47];
          float v301_data = s0[67];
          float v306_data = s0[87];
          float v311_data = s0[107];
          float v316_data = s0[127];
          float v320_data = r0[8];
          float v321_data = s0[8];
          float v326_data = s0[28];
          float v331_data = s0[48];
          float v336_data = s0[68];
          float v341_data = s0[88];
          float v346_data = s0[108];
          float v351_data = s0[128];
          float v355_data = r0[9];
          float v356_data = s0[9];
          float v361_data = s0[29];
          float v366_data = s0[49];
          float v371_data = s0[69];
          float v376_data = s0[89];
          float v381_data = s0[109];
          float v386_data = s0[129];
          float v390_data = r0[10];
          float v391_data = s0[10];
          float v396_data = s0[30];
          float v401_data = s0[50];
          float v406_data = s0[70];
          float v411_data = s0[90];
          float v416_data = s0[110];
          float v421_data = s0[130];
          float v425_data = r0[11];
          float v426_data = s0[11];
          float v431_data = s0[31];
          float v436_data = s0[51];
          float v441_data = s0[71];
          float v446_data = s0[91];
          float v451_data = s0[111];
          float v456_data = s0[131];
          float v460_data = r0[12];
          float v461_data = s0[12];
          float v466_data = s0[32];
          float v471_data = s0[52];
          float v476_data = s0[72];
          float v481_data = s0[92];
          float v486_data = s0[112];
          float v491_data = s0[132];
          float v495_data = r0[13];
          float v496_data = s0[13];
          float v501_data = s0[33];
          float v506_data = s0[53];
          float v511_data = s0[73];
          float v516_data = s0[93];
          float v521_data = s0[113];
          float v526_data = s0[133];
          float v530_data = r0[14];
          float v531_data = s0[14];
          float v536_data = s0[34];
          float v541_data = s0[54];
          float v546_data = s0[74];
          float v551_data = s0[94];
          float v556_data = s0[114];
          float v561_data = s0[134];
          float v565_data = r0[15];
          float v566_data = s0[15];
          float v571_data = s0[35];
          float v576_data = s0[55];
          float v581_data = s0[75];
          float v586_data = s0[95];
          float v591_data = s0[115];
          float v596_data = s0[135];
          float v600_data = r0[16];
          float v601_data = s0[16];
          float v606_data = s0[36];
          float v611_data = s0[56];
          float v616_data = s0[76];
          float v621_data = s0[96];
          float v626_data = s0[116];
          float v631_data = s0[136];
          float v635_data = r0[17];
          float v636_data = s0[17];
          float v641_data = s0[37];
          float v646_data = s0[57];
          float v651_data = s0[77];
          float v656_data = s0[97];
          float v661_data = s0[117];
          float v666_data = s0[137];
          float v670_data = r0[18];
          float v671_data = s0[18];
          float v674_p = ((((((((((((((((((v40_data * v41_data) + (v75_data * v76_data)) + (v110_data * v111_data)) + (v145_data * v146_data)) + (v180_data * v181_data)) + (v215_data * v216_data)) + (v250_data * v251_data)) + (v285_data * v286_data)) + (v320_data * v321_data)) + (v355_data * v356_data)) + (v390_data * v391_data)) + (v425_data * v426_data)) + (v460_data * v461_data)) + (v495_data * v496_data)) + (v530_data * v531_data)) + (v565_data * v566_data)) + (v600_data * v601_data)) + (v635_data * v636_data)) + (v670_data * v671_data);
          float v676_data = s0[38];
          float v679_p = ((((((((((((((((((v40_data * v46_data) + (v75_data * v81_data)) + (v110_data * v116_data)) + (v145_data * v151_data)) + (v180_data * v186_data)) + (v215_data * v221_data)) + (v250_data * v256_data)) + (v285_data * v291_data)) + (v320_data * v326_data)) + (v355_data * v361_data)) + (v390_data * v396_data)) + (v425_data * v431_data)) + (v460_data * v466_data)) + (v495_data * v501_data)) + (v530_data * v536_data)) + (v565_data * v571_data)) + (v600_data * v606_data)) + (v635_data * v641_data)) + (v670_data * v676_data);
          float v681_data = s0[58];
          float v684_p = ((((((((((((((((((v40_data * v51_data) + (v75_data * v86_data)) + (v110_data * v121_data)) + (v145_data * v156_data)) + (v180_data * v191_data)) + (v215_data * v226_data)) + (v250_data * v261_data)) + (v285_data * v296_data)) + (v320_data * v331_data)) + (v355_data * v366_data)) + (v390_data * v401_data)) + (v425_data * v436_data)) + (v460_data * v471_data)) + (v495_data * v506_data)) + (v530_data * v541_data)) + (v565_data * v576_data)) + (v600_data * v611_data)) + (v635_data * v646_data)) + (v670_data * v681_data);
          float v686_data = s0[78];
          float v689_p = ((((((((((((((((((v40_data * v56_data) + (v75_data * v91_data)) + (v110_data * v126_data)) + (v145_data * v161_data)) + (v180_data * v196_data)) + (v215_data * v231_data)) + (v250_data * v266_data)) + (v285_data * v301_data)) + (v320_data * v336_data)) + (v355_data * v371_data)) + (v390_data * v406_data)) + (v425_data * v441_data)) + (v460_data * v476_data)) + (v495_data * v511_data)) + (v530_data * v546_data)) + (v565_data * v581_data)) + (v600_data * v616_data)) + (v635_data * v651_data)) + (v670_data * v686_data);
          float v691_data = s0[98];
          float v694_p = ((((((((((((((((((v40_data * v61_data) + (v75_data * v96_data)) + (v110_data * v131_data)) + (v145_data * v166_data)) + (v180_data * v201_data)) + (v215_data * v236_data)) + (v250_data * v271_data)) + (v285_data * v306_data)) + (v320_data * v341_data)) + (v355_data * v376_data)) + (v390_data * v411_data)) + (v425_data * v446_data)) + (v460_data * v481_data)) + (v495_data * v516_data)) + (v530_data * v551_data)) + (v565_data * v586_data)) + (v600_data * v621_data)) + (v635_data * v656_data)) + (v670_data * v691_data);
          float v696_data = s0[118];
          float v699_p = ((((((((((((((((((v40_data * v66_data) + (v75_data * v101_data)) + (v110_data * v136_data)) + (v145_data * v171_data)) + (v180_data * v206_data)) + (v215_data * v241_data)) + (v250_data * v276_data)) + (v285_data * v311_data)) + (v320_data * v346_data)) + (v355_data * v381_data)) + (v390_data * v416_data)) + (v425_data * v451_data)) + (v460_data * v486_data)) + (v495_data * v521_data)) + (v530_data * v556_data)) + (v565_data * v591_data)) + (v600_data * v626_data)) + (v635_data * v661_data)) + (v670_data * v696_data);
          float v701_data = s0[138];
          float v704_p = ((((((((((((((((((v40_data * v71_data) + (v75_data * v106_data)) + (v110_data * v141_data)) + (v145_data * v176_data)) + (v180_data * v211_data)) + (v215_data * v246_data)) + (v250_data * v281_data)) + (v285_data * v316_data)) + (v320_data * v351_data)) + (v355_data * v386_data)) + (v390_data * v421_data)) + (v425_data * v456_data)) + (v460_data * v491_data)) + (v495_data * v526_data)) + (v530_data * v561_data)) + (v565_data * v596_data)) + (v600_data * v631_data)) + (v635_data * v666_data)) + (v670_data * v701_data);
          float v705_data = r0[19];
          bool v784_g = allowed_next && v765_g;
          // r0 = load{g>r}(wrap_glb_m1);
          #pragma unroll
          for (int32_t v779_i1 = 0; v779_i1 < 20; ++v779_i1) {
            float v782_data = v784_g ? (__ldcg(&wrap_glb_m1[(v764_lead + (v779_i1 * 12))])) : (0.0f);
            r0[v779_i1] = v782_data;
          }
          float v706_data = s0[19];
          ir1[0] = (v674_p + (v705_data * v706_data));
          float v711_data = s0[39];
          ir1[1] = (v679_p + (v705_data * v711_data));
          float v716_data = s0[59];
          ir1[2] = (v684_p + (v705_data * v716_data));
          float v721_data = s0[79];
          ir1[3] = (v689_p + (v705_data * v721_data));
          float v726_data = s0[99];
          ir1[4] = (v694_p + (v705_data * v726_data));
          float v731_data = s0[119];
          ir1[5] = (v699_p + (v705_data * v731_data));
          float v736_data = s0[139];
          ir1[6] = (v704_p + (v705_data * v736_data));
          // r1 = ir1
          // glb_m0 = store{r>g}(r1);
          if (v765_g) {
            #pragma unroll
            for (int32_t v745_i1 = 0; v745_i1 < 7; ++v745_i1) {
              float v747_data = ir1[v745_i1];
              glb_m0[(v764_lead + (v745_i1 * 12))] = v747_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
        else {
          __pipeline_wait_prior(0);
          bool v785_g = allowed_next && v765_g;
          // r0 = load{g>r}(wrap_glb_m1);
          #pragma unroll
          for (int32_t v771_i1 = 0; v771_i1 < 20; ++v771_i1) {
            float v774_data = v785_g ? (__ldcg(&wrap_glb_m1[(v764_lead + (v771_i1 * 12))])) : (0.0f);
            r0[v771_i1] = v774_data;
          }
        }
        // s0 = load{g>s}(wrap_glb_m2[0, 1])
        #pragma unroll
        for (int32_t v795_i = 0; v795_i < 8; ++v795_i) {
          int32_t v797_a = threadIdx.x + (v795_i * 16);
          if (allowed_next) {
            __pipeline_memcpy_async(&s0[v797_a], &wrap_glb_m2[v797_a], 4);
          }
        }
        if (v787_g) {
          if (allowed_next) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 128], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 128], 4);
          }
        }
        __pipeline_commit();
        v757_flagWord = v758_flagWordNext;
        v758_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
  }
}

