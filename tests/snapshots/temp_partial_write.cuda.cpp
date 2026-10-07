// === base name ===
kernel_1c8ba79120f984be

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_1c8ba79120f984be = {{16, 8, 1}, 16, 12, 1, 8, 5632, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_1c8ba79120f984be(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_1c8ba79120f984be(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_1c8ba79120f984be(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_1c8ba79120f984be, block.x * block.y * block.z, 1408 * sizeof(float));
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
void launcher_kernel_1c8ba79120f984be(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_1c8ba79120f984be(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_1c8ba79120f984be, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_1c8ba79120f984be<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_1c8ba79120f984be(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
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
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1408}],"shared_bytes":5632,"shared_elements":1408,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B","bbox":[[0,0],[12,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,6]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]},{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[176 * threadIdx.y + 0];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < numElements0; v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 144 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v9_batchId0 * 144 + 0 + m1_extraOffset];
          float *const __restrict__ glb_m2 = &m2[v9_batchId0 * 144 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[v9_batchId0 * 144 + 0 + m3_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v24_lead = threadIdx.x % 16;
          bool v25_g = v24_lead < 12;
          if (v25_g) {
            #pragma unroll
            for (int32_t v26_i1 = 0; v26_i1 < 12; ++v26_i1) {
              float v31_data = __ldcg(&glb_m0[(v24_lead + (v26_i1 * 12))]);
              r0[v26_i1] = v31_data;
            }
          }
          // s0 = load{g>s}(glb_m1[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 9; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m1[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
          __pipeline_commit();
          float r2[12]{};
          // r2 = load{g>r}(glb_m3);
          if (v25_g) {
            #pragma unroll
            for (int32_t v415_i1 = 0; v415_i1 < 12; ++v415_i1) {
              float v420_data = __ldcg(&glb_m3[(v24_lead + (v415_i1 * 12))]);
              r2[v415_i1] = v420_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[6]{};
          // r1 = +(r0 * s0) + None
          // [(0, 12), (0, 6)] [(0, 12)]
          float v35_data = r0[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v36_data = s0[0];
          float v38_data = r1[0];
          r1[0] = (v38_data + (v35_data * v36_data));
          float v41_data = s0[12];
          float v43_data = r1[1];
          r1[1] = (v43_data + (v35_data * v41_data));
          float v46_data = s0[24];
          float v48_data = r1[2];
          r1[2] = (v48_data + (v35_data * v46_data));
          float v51_data = s0[36];
          float v53_data = r1[3];
          r1[3] = (v53_data + (v35_data * v51_data));
          float v56_data = s0[48];
          float v58_data = r1[4];
          r1[4] = (v58_data + (v35_data * v56_data));
          float v61_data = s0[60];
          float v63_data = r1[5];
          r1[5] = (v63_data + (v35_data * v61_data));
          float v65_data = r0[1];
          float v66_data = s0[1];
          float v68_data = r1[0];
          r1[0] = (v68_data + (v65_data * v66_data));
          float v71_data = s0[13];
          float v73_data = r1[1];
          r1[1] = (v73_data + (v65_data * v71_data));
          float v76_data = s0[25];
          float v78_data = r1[2];
          r1[2] = (v78_data + (v65_data * v76_data));
          float v81_data = s0[37];
          float v83_data = r1[3];
          r1[3] = (v83_data + (v65_data * v81_data));
          float v86_data = s0[49];
          float v88_data = r1[4];
          r1[4] = (v88_data + (v65_data * v86_data));
          float v91_data = s0[61];
          float v93_data = r1[5];
          r1[5] = (v93_data + (v65_data * v91_data));
          float v95_data = r0[2];
          float v96_data = s0[2];
          float v98_data = r1[0];
          r1[0] = (v98_data + (v95_data * v96_data));
          float v101_data = s0[14];
          float v103_data = r1[1];
          r1[1] = (v103_data + (v95_data * v101_data));
          float v106_data = s0[26];
          float v108_data = r1[2];
          r1[2] = (v108_data + (v95_data * v106_data));
          float v111_data = s0[38];
          float v113_data = r1[3];
          r1[3] = (v113_data + (v95_data * v111_data));
          float v116_data = s0[50];
          float v118_data = r1[4];
          r1[4] = (v118_data + (v95_data * v116_data));
          float v121_data = s0[62];
          float v123_data = r1[5];
          r1[5] = (v123_data + (v95_data * v121_data));
          float v125_data = r0[3];
          float v126_data = s0[3];
          float v128_data = r1[0];
          r1[0] = (v128_data + (v125_data * v126_data));
          float v131_data = s0[15];
          float v133_data = r1[1];
          r1[1] = (v133_data + (v125_data * v131_data));
          float v136_data = s0[27];
          float v138_data = r1[2];
          r1[2] = (v138_data + (v125_data * v136_data));
          float v141_data = s0[39];
          float v143_data = r1[3];
          r1[3] = (v143_data + (v125_data * v141_data));
          float v146_data = s0[51];
          float v148_data = r1[4];
          r1[4] = (v148_data + (v125_data * v146_data));
          float v151_data = s0[63];
          float v153_data = r1[5];
          r1[5] = (v153_data + (v125_data * v151_data));
          float v155_data = r0[4];
          float v156_data = s0[4];
          float v158_data = r1[0];
          r1[0] = (v158_data + (v155_data * v156_data));
          float v161_data = s0[16];
          float v163_data = r1[1];
          r1[1] = (v163_data + (v155_data * v161_data));
          float v166_data = s0[28];
          float v168_data = r1[2];
          r1[2] = (v168_data + (v155_data * v166_data));
          float v171_data = s0[40];
          float v173_data = r1[3];
          r1[3] = (v173_data + (v155_data * v171_data));
          float v176_data = s0[52];
          float v178_data = r1[4];
          r1[4] = (v178_data + (v155_data * v176_data));
          float v181_data = s0[64];
          float v183_data = r1[5];
          r1[5] = (v183_data + (v155_data * v181_data));
          float v185_data = r0[5];
          float v186_data = s0[5];
          float v188_data = r1[0];
          r1[0] = (v188_data + (v185_data * v186_data));
          float v191_data = s0[17];
          float v193_data = r1[1];
          r1[1] = (v193_data + (v185_data * v191_data));
          float v196_data = s0[29];
          float v198_data = r1[2];
          r1[2] = (v198_data + (v185_data * v196_data));
          float v201_data = s0[41];
          float v203_data = r1[3];
          r1[3] = (v203_data + (v185_data * v201_data));
          float v206_data = s0[53];
          float v208_data = r1[4];
          r1[4] = (v208_data + (v185_data * v206_data));
          float v211_data = s0[65];
          float v213_data = r1[5];
          r1[5] = (v213_data + (v185_data * v211_data));
          float v215_data = r0[6];
          float v216_data = s0[6];
          float v218_data = r1[0];
          r1[0] = (v218_data + (v215_data * v216_data));
          float v221_data = s0[18];
          float v223_data = r1[1];
          r1[1] = (v223_data + (v215_data * v221_data));
          float v226_data = s0[30];
          float v228_data = r1[2];
          r1[2] = (v228_data + (v215_data * v226_data));
          float v231_data = s0[42];
          float v233_data = r1[3];
          r1[3] = (v233_data + (v215_data * v231_data));
          float v236_data = s0[54];
          float v238_data = r1[4];
          r1[4] = (v238_data + (v215_data * v236_data));
          float v241_data = s0[66];
          float v243_data = r1[5];
          r1[5] = (v243_data + (v215_data * v241_data));
          float v245_data = r0[7];
          float v246_data = s0[7];
          float v248_data = r1[0];
          r1[0] = (v248_data + (v245_data * v246_data));
          float v251_data = s0[19];
          float v253_data = r1[1];
          r1[1] = (v253_data + (v245_data * v251_data));
          float v256_data = s0[31];
          float v258_data = r1[2];
          r1[2] = (v258_data + (v245_data * v256_data));
          float v261_data = s0[43];
          float v263_data = r1[3];
          r1[3] = (v263_data + (v245_data * v261_data));
          float v266_data = s0[55];
          float v268_data = r1[4];
          r1[4] = (v268_data + (v245_data * v266_data));
          float v271_data = s0[67];
          float v273_data = r1[5];
          r1[5] = (v273_data + (v245_data * v271_data));
          float v275_data = r0[8];
          float v276_data = s0[8];
          float v278_data = r1[0];
          r1[0] = (v278_data + (v275_data * v276_data));
          float v281_data = s0[20];
          float v283_data = r1[1];
          r1[1] = (v283_data + (v275_data * v281_data));
          float v286_data = s0[32];
          float v288_data = r1[2];
          r1[2] = (v288_data + (v275_data * v286_data));
          float v291_data = s0[44];
          float v293_data = r1[3];
          r1[3] = (v293_data + (v275_data * v291_data));
          float v296_data = s0[56];
          float v298_data = r1[4];
          r1[4] = (v298_data + (v275_data * v296_data));
          float v301_data = s0[68];
          float v303_data = r1[5];
          r1[5] = (v303_data + (v275_data * v301_data));
          float v305_data = r0[9];
          float v306_data = s0[9];
          float v308_data = r1[0];
          r1[0] = (v308_data + (v305_data * v306_data));
          float v311_data = s0[21];
          float v313_data = r1[1];
          r1[1] = (v313_data + (v305_data * v311_data));
          float v316_data = s0[33];
          float v318_data = r1[2];
          r1[2] = (v318_data + (v305_data * v316_data));
          float v321_data = s0[45];
          float v323_data = r1[3];
          r1[3] = (v323_data + (v305_data * v321_data));
          float v326_data = s0[57];
          float v328_data = r1[4];
          r1[4] = (v328_data + (v305_data * v326_data));
          float v331_data = s0[69];
          float v333_data = r1[5];
          r1[5] = (v333_data + (v305_data * v331_data));
          float v335_data = r0[10];
          float v336_data = s0[10];
          float v338_data = r1[0];
          r1[0] = (v338_data + (v335_data * v336_data));
          float v341_data = s0[22];
          float v343_data = r1[1];
          r1[1] = (v343_data + (v335_data * v341_data));
          float v346_data = s0[34];
          float v348_data = r1[2];
          r1[2] = (v348_data + (v335_data * v346_data));
          float v351_data = s0[46];
          float v353_data = r1[3];
          r1[3] = (v353_data + (v335_data * v351_data));
          float v356_data = s0[58];
          float v358_data = r1[4];
          r1[4] = (v358_data + (v335_data * v356_data));
          float v361_data = s0[70];
          float v363_data = r1[5];
          r1[5] = (v363_data + (v335_data * v361_data));
          float v365_data = r0[11];
          float v366_data = s0[11];
          float v368_data = r1[0];
          r1[0] = (v368_data + (v365_data * v366_data));
          float v371_data = s0[23];
          float v373_data = r1[1];
          r1[1] = (v373_data + (v365_data * v371_data));
          float v376_data = s0[35];
          float v378_data = r1[2];
          r1[2] = (v378_data + (v365_data * v376_data));
          float v381_data = s0[47];
          float v383_data = r1[3];
          r1[3] = (v383_data + (v365_data * v381_data));
          float v386_data = s0[59];
          float v388_data = r1[4];
          r1[4] = (v388_data + (v365_data * v386_data));
          float v391_data = s0[71];
          float v393_data = r1[5];
          r1[5] = (v393_data + (v365_data * v391_data));
          // s1 = store{r>s, clear}(localShrMem0, r1);
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          if (v25_g) {
            #pragma unroll
            for (int32_t v395_z1 = 6; v395_z1 < 12; ++v395_z1) {
              int32_t v400_a = v24_lead + (v395_z1 * 12);
              s1[(v400_a ^ ((v400_a >> 4) & 15))] = 0.0f;
            }
          }
          if (v25_g) {
            #pragma unroll
            for (int32_t v404_i1 = 0; v404_i1 < 6; ++v404_i1) {
              float v406_data = r1[v404_i1];
              int32_t v410_a = v24_lead + (v404_i1 * 12);
              s1[(v410_a ^ ((v410_a >> 4) & 15))] = v406_data;
            }
          }
          float r3[12]{};
          // ir3 = +(r2 * s1)
          // [(0, 12), (0, 12)] [(0, 12)]
          float ir3[12]{};
          float v424_data = r2[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v425_data = s1[0];
          float v427_data = ir3[0];
          ir3[0] = (v427_data + (v424_data * v425_data));
          float v430_data = s1[12];
          float v432_data = ir3[1];
          ir3[1] = (v432_data + (v424_data * v430_data));
          float v435_data = s1[25];
          float v437_data = ir3[2];
          ir3[2] = (v437_data + (v424_data * v435_data));
          float v440_data = s1[38];
          float v442_data = ir3[3];
          ir3[3] = (v442_data + (v424_data * v440_data));
          float v445_data = s1[51];
          float v447_data = ir3[4];
          ir3[4] = (v447_data + (v424_data * v445_data));
          float v450_data = s1[63];
          float v452_data = ir3[5];
          ir3[5] = (v452_data + (v424_data * v450_data));
          float v455_data = s1[76];
          float v457_data = ir3[6];
          ir3[6] = (v457_data + (v424_data * v455_data));
          float v460_data = s1[81];
          float v462_data = ir3[7];
          ir3[7] = (v462_data + (v424_data * v460_data));
          float v465_data = s1[102];
          float v467_data = ir3[8];
          ir3[8] = (v467_data + (v424_data * v465_data));
          float v470_data = s1[106];
          float v472_data = ir3[9];
          ir3[9] = (v472_data + (v424_data * v470_data));
          float v475_data = s1[127];
          float v477_data = ir3[10];
          ir3[10] = (v477_data + (v424_data * v475_data));
          float v480_data = s1[140];
          float v482_data = ir3[11];
          ir3[11] = (v482_data + (v424_data * v480_data));
          float v484_data = r2[1];
          float v485_data = s1[1];
          float v487_data = ir3[0];
          ir3[0] = (v487_data + (v484_data * v485_data));
          float v490_data = s1[13];
          float v492_data = ir3[1];
          ir3[1] = (v492_data + (v484_data * v490_data));
          float v495_data = s1[24];
          float v497_data = ir3[2];
          ir3[2] = (v497_data + (v484_data * v495_data));
          float v500_data = s1[39];
          float v502_data = ir3[3];
          ir3[3] = (v502_data + (v484_data * v500_data));
          float v505_data = s1[50];
          float v507_data = ir3[4];
          ir3[4] = (v507_data + (v484_data * v505_data));
          float v510_data = s1[62];
          float v512_data = ir3[5];
          ir3[5] = (v512_data + (v484_data * v510_data));
          float v515_data = s1[77];
          float v517_data = ir3[6];
          ir3[6] = (v517_data + (v484_data * v515_data));
          float v520_data = s1[80];
          float v522_data = ir3[7];
          ir3[7] = (v522_data + (v484_data * v520_data));
          float v525_data = s1[103];
          float v527_data = ir3[8];
          ir3[8] = (v527_data + (v484_data * v525_data));
          float v530_data = s1[107];
          float v532_data = ir3[9];
          ir3[9] = (v532_data + (v484_data * v530_data));
          float v535_data = s1[126];
          float v537_data = ir3[10];
          ir3[10] = (v537_data + (v484_data * v535_data));
          float v540_data = s1[141];
          float v542_data = ir3[11];
          ir3[11] = (v542_data + (v484_data * v540_data));
          float v544_data = r2[2];
          float v545_data = s1[2];
          float v547_data = ir3[0];
          ir3[0] = (v547_data + (v544_data * v545_data));
          float v550_data = s1[14];
          float v552_data = ir3[1];
          ir3[1] = (v552_data + (v544_data * v550_data));
          float v555_data = s1[27];
          float v557_data = ir3[2];
          ir3[2] = (v557_data + (v544_data * v555_data));
          float v560_data = s1[36];
          float v562_data = ir3[3];
          ir3[3] = (v562_data + (v544_data * v560_data));
          float v565_data = s1[49];
          float v567_data = ir3[4];
          ir3[4] = (v567_data + (v544_data * v565_data));
          float v570_data = s1[61];
          float v572_data = ir3[5];
          ir3[5] = (v572_data + (v544_data * v570_data));
          float v575_data = s1[78];
          float v577_data = ir3[6];
          ir3[6] = (v577_data + (v544_data * v575_data));
          float v580_data = s1[83];
          float v582_data = ir3[7];
          ir3[7] = (v582_data + (v544_data * v580_data));
          float v585_data = s1[100];
          float v587_data = ir3[8];
          ir3[8] = (v587_data + (v544_data * v585_data));
          float v590_data = s1[104];
          float v592_data = ir3[9];
          ir3[9] = (v592_data + (v544_data * v590_data));
          float v595_data = s1[125];
          float v597_data = ir3[10];
          ir3[10] = (v597_data + (v544_data * v595_data));
          float v600_data = s1[142];
          float v602_data = ir3[11];
          ir3[11] = (v602_data + (v544_data * v600_data));
          float v604_data = r2[3];
          float v605_data = s1[3];
          float v607_data = ir3[0];
          ir3[0] = (v607_data + (v604_data * v605_data));
          float v610_data = s1[15];
          float v612_data = ir3[1];
          ir3[1] = (v612_data + (v604_data * v610_data));
          float v615_data = s1[26];
          float v617_data = ir3[2];
          ir3[2] = (v617_data + (v604_data * v615_data));
          float v620_data = s1[37];
          float v622_data = ir3[3];
          ir3[3] = (v622_data + (v604_data * v620_data));
          float v625_data = s1[48];
          float v627_data = ir3[4];
          ir3[4] = (v627_data + (v604_data * v625_data));
          float v630_data = s1[60];
          float v632_data = ir3[5];
          ir3[5] = (v632_data + (v604_data * v630_data));
          float v635_data = s1[79];
          float v637_data = ir3[6];
          ir3[6] = (v637_data + (v604_data * v635_data));
          float v640_data = s1[82];
          float v642_data = ir3[7];
          ir3[7] = (v642_data + (v604_data * v640_data));
          float v645_data = s1[101];
          float v647_data = ir3[8];
          ir3[8] = (v647_data + (v604_data * v645_data));
          float v650_data = s1[105];
          float v652_data = ir3[9];
          ir3[9] = (v652_data + (v604_data * v650_data));
          float v655_data = s1[124];
          float v657_data = ir3[10];
          ir3[10] = (v657_data + (v604_data * v655_data));
          float v660_data = s1[143];
          float v662_data = ir3[11];
          ir3[11] = (v662_data + (v604_data * v660_data));
          float v664_data = r2[4];
          float v665_data = s1[4];
          float v667_data = ir3[0];
          ir3[0] = (v667_data + (v664_data * v665_data));
          float v670_data = s1[17];
          float v672_data = ir3[1];
          ir3[1] = (v672_data + (v664_data * v670_data));
          float v675_data = s1[29];
          float v677_data = ir3[2];
          ir3[2] = (v677_data + (v664_data * v675_data));
          float v680_data = s1[42];
          float v682_data = ir3[3];
          ir3[3] = (v682_data + (v664_data * v680_data));
          float v685_data = s1[55];
          float v687_data = ir3[4];
          ir3[4] = (v687_data + (v664_data * v685_data));
          float v690_data = s1[68];
          float v692_data = ir3[5];
          ir3[5] = (v692_data + (v664_data * v690_data));
          float v695_data = s1[72];
          float v697_data = ir3[6];
          ir3[6] = (v697_data + (v664_data * v695_data));
          float v700_data = s1[93];
          float v702_data = ir3[7];
          ir3[7] = (v702_data + (v664_data * v700_data));
          float v705_data = s1[98];
          float v707_data = ir3[8];
          ir3[8] = (v707_data + (v664_data * v705_data));
          float v710_data = s1[119];
          float v712_data = ir3[9];
          ir3[9] = (v712_data + (v664_data * v710_data));
          float v715_data = s1[123];
          float v717_data = ir3[10];
          ir3[10] = (v717_data + (v664_data * v715_data));
          float v720_data = s1[128];
          float v722_data = ir3[11];
          ir3[11] = (v722_data + (v664_data * v720_data));
          float v724_data = r2[5];
          float v725_data = s1[5];
          float v727_data = ir3[0];
          ir3[0] = (v727_data + (v724_data * v725_data));
          float v730_data = s1[16];
          float v732_data = ir3[1];
          ir3[1] = (v732_data + (v724_data * v730_data));
          float v735_data = s1[28];
          float v737_data = ir3[2];
          ir3[2] = (v737_data + (v724_data * v735_data));
          float v740_data = s1[43];
          float v742_data = ir3[3];
          ir3[3] = (v742_data + (v724_data * v740_data));
          float v745_data = s1[54];
          float v747_data = ir3[4];
          ir3[4] = (v747_data + (v724_data * v745_data));
          float v750_data = s1[69];
          float v752_data = ir3[5];
          ir3[5] = (v752_data + (v724_data * v750_data));
          float v755_data = s1[73];
          float v757_data = ir3[6];
          ir3[6] = (v757_data + (v724_data * v755_data));
          float v760_data = s1[92];
          float v762_data = ir3[7];
          ir3[7] = (v762_data + (v724_data * v760_data));
          float v765_data = s1[99];
          float v767_data = ir3[8];
          ir3[8] = (v767_data + (v724_data * v765_data));
          float v770_data = s1[118];
          float v772_data = ir3[9];
          ir3[9] = (v772_data + (v724_data * v770_data));
          float v775_data = s1[122];
          float v777_data = ir3[10];
          ir3[10] = (v777_data + (v724_data * v775_data));
          float v780_data = s1[129];
          float v782_data = ir3[11];
          ir3[11] = (v782_data + (v724_data * v780_data));
          float v784_data = r2[6];
          float v785_data = s1[6];
          float v787_data = ir3[0];
          ir3[0] = (v787_data + (v784_data * v785_data));
          float v790_data = s1[19];
          float v792_data = ir3[1];
          ir3[1] = (v792_data + (v784_data * v790_data));
          float v795_data = s1[31];
          float v797_data = ir3[2];
          ir3[2] = (v797_data + (v784_data * v795_data));
          float v800_data = s1[40];
          float v802_data = ir3[3];
          ir3[3] = (v802_data + (v784_data * v800_data));
          float v805_data = s1[53];
          float v807_data = ir3[4];
          ir3[4] = (v807_data + (v784_data * v805_data));
          float v810_data = s1[70];
          float v812_data = ir3[5];
          ir3[5] = (v812_data + (v784_data * v810_data));
          float v815_data = s1[74];
          float v817_data = ir3[6];
          ir3[6] = (v817_data + (v784_data * v815_data));
          float v820_data = s1[95];
          float v822_data = ir3[7];
          ir3[7] = (v822_data + (v784_data * v820_data));
          float v825_data = s1[96];
          float v827_data = ir3[8];
          ir3[8] = (v827_data + (v784_data * v825_data));
          float v830_data = s1[117];
          float v832_data = ir3[9];
          ir3[9] = (v832_data + (v784_data * v830_data));
          float v835_data = s1[121];
          float v837_data = ir3[10];
          ir3[10] = (v837_data + (v784_data * v835_data));
          float v840_data = s1[130];
          float v842_data = ir3[11];
          ir3[11] = (v842_data + (v784_data * v840_data));
          float v844_data = r2[7];
          float v845_data = s1[7];
          float v847_data = ir3[0];
          ir3[0] = (v847_data + (v844_data * v845_data));
          float v850_data = s1[18];
          float v852_data = ir3[1];
          ir3[1] = (v852_data + (v844_data * v850_data));
          float v855_data = s1[30];
          float v857_data = ir3[2];
          ir3[2] = (v857_data + (v844_data * v855_data));
          float v860_data = s1[41];
          float v862_data = ir3[3];
          ir3[3] = (v862_data + (v844_data * v860_data));
          float v865_data = s1[52];
          float v867_data = ir3[4];
          ir3[4] = (v867_data + (v844_data * v865_data));
          float v870_data = s1[71];
          float v872_data = ir3[5];
          ir3[5] = (v872_data + (v844_data * v870_data));
          float v875_data = s1[75];
          float v877_data = ir3[6];
          ir3[6] = (v877_data + (v844_data * v875_data));
          float v880_data = s1[94];
          float v882_data = ir3[7];
          ir3[7] = (v882_data + (v844_data * v880_data));
          float v885_data = s1[97];
          float v887_data = ir3[8];
          ir3[8] = (v887_data + (v844_data * v885_data));
          float v890_data = s1[116];
          float v892_data = ir3[9];
          ir3[9] = (v892_data + (v844_data * v890_data));
          float v895_data = s1[120];
          float v897_data = ir3[10];
          ir3[10] = (v897_data + (v844_data * v895_data));
          float v900_data = s1[131];
          float v902_data = ir3[11];
          ir3[11] = (v902_data + (v844_data * v900_data));
          float v904_data = r2[8];
          float v905_data = s1[8];
          float v907_data = ir3[0];
          ir3[0] = (v907_data + (v904_data * v905_data));
          float v910_data = s1[21];
          float v912_data = ir3[1];
          ir3[1] = (v912_data + (v904_data * v910_data));
          float v915_data = s1[34];
          float v917_data = ir3[2];
          ir3[2] = (v917_data + (v904_data * v915_data));
          float v920_data = s1[46];
          float v922_data = ir3[3];
          ir3[3] = (v922_data + (v904_data * v920_data));
          float v925_data = s1[59];
          float v927_data = ir3[4];
          ir3[4] = (v927_data + (v904_data * v925_data));
          float v930_data = s1[64];
          float v932_data = ir3[5];
          ir3[5] = (v932_data + (v904_data * v930_data));
          float v935_data = s1[85];
          float v937_data = ir3[6];
          ir3[6] = (v937_data + (v904_data * v935_data));
          float v940_data = s1[89];
          float v942_data = ir3[7];
          ir3[7] = (v942_data + (v904_data * v940_data));
          float v945_data = s1[110];
          float v947_data = ir3[8];
          ir3[8] = (v947_data + (v904_data * v945_data));
          float v950_data = s1[115];
          float v952_data = ir3[9];
          ir3[9] = (v952_data + (v904_data * v950_data));
          float v955_data = s1[136];
          float v957_data = ir3[10];
          ir3[10] = (v957_data + (v904_data * v955_data));
          float v960_data = s1[132];
          float v962_data = ir3[11];
          ir3[11] = (v962_data + (v904_data * v960_data));
          float v964_data = r2[9];
          float v965_data = s1[9];
          float v967_data = ir3[0];
          ir3[0] = (v967_data + (v964_data * v965_data));
          float v970_data = s1[20];
          float v972_data = ir3[1];
          ir3[1] = (v972_data + (v964_data * v970_data));
          float v975_data = s1[35];
          float v977_data = ir3[2];
          ir3[2] = (v977_data + (v964_data * v975_data));
          float v980_data = s1[47];
          float v982_data = ir3[3];
          ir3[3] = (v982_data + (v964_data * v980_data));
          float v985_data = s1[58];
          float v987_data = ir3[4];
          ir3[4] = (v987_data + (v964_data * v985_data));
          float v990_data = s1[65];
          float v992_data = ir3[5];
          ir3[5] = (v992_data + (v964_data * v990_data));
          float v995_data = s1[84];
          float v997_data = ir3[6];
          ir3[6] = (v997_data + (v964_data * v995_data));
          float v1000_data = s1[88];
          float v1002_data = ir3[7];
          ir3[7] = (v1002_data + (v964_data * v1000_data));
          float v1005_data = s1[111];
          float v1007_data = ir3[8];
          ir3[8] = (v1007_data + (v964_data * v1005_data));
          float v1010_data = s1[114];
          float v1012_data = ir3[9];
          ir3[9] = (v1012_data + (v964_data * v1010_data));
          float v1015_data = s1[137];
          float v1017_data = ir3[10];
          ir3[10] = (v1017_data + (v964_data * v1015_data));
          float v1020_data = s1[133];
          float v1022_data = ir3[11];
          ir3[11] = (v1022_data + (v964_data * v1020_data));
          float v1024_data = r2[10];
          float v1025_data = s1[10];
          float v1027_data = ir3[0];
          ir3[0] = (v1027_data + (v1024_data * v1025_data));
          float v1030_data = s1[23];
          float v1032_data = ir3[1];
          ir3[1] = (v1032_data + (v1024_data * v1030_data));
          float v1035_data = s1[32];
          float v1037_data = ir3[2];
          ir3[2] = (v1037_data + (v1024_data * v1035_data));
          float v1040_data = s1[44];
          float v1042_data = ir3[3];
          ir3[3] = (v1042_data + (v1024_data * v1040_data));
          float v1045_data = s1[57];
          float v1047_data = ir3[4];
          ir3[4] = (v1047_data + (v1024_data * v1045_data));
          float v1050_data = s1[66];
          float v1052_data = ir3[5];
          ir3[5] = (v1052_data + (v1024_data * v1050_data));
          float v1055_data = s1[87];
          float v1057_data = ir3[6];
          ir3[6] = (v1057_data + (v1024_data * v1055_data));
          float v1060_data = s1[91];
          float v1062_data = ir3[7];
          ir3[7] = (v1062_data + (v1024_data * v1060_data));
          float v1065_data = s1[108];
          float v1067_data = ir3[8];
          ir3[8] = (v1067_data + (v1024_data * v1065_data));
          float v1070_data = s1[113];
          float v1072_data = ir3[9];
          ir3[9] = (v1072_data + (v1024_data * v1070_data));
          float v1075_data = s1[138];
          float v1077_data = ir3[10];
          ir3[10] = (v1077_data + (v1024_data * v1075_data));
          float v1080_data = s1[134];
          float v1082_data = ir3[11];
          ir3[11] = (v1082_data + (v1024_data * v1080_data));
          float v1084_data = r2[11];
          float v1085_data = s1[11];
          float v1087_data = ir3[0];
          ir3[0] = (v1087_data + (v1084_data * v1085_data));
          float v1090_data = s1[22];
          float v1092_data = ir3[1];
          ir3[1] = (v1092_data + (v1084_data * v1090_data));
          float v1095_data = s1[33];
          float v1097_data = ir3[2];
          ir3[2] = (v1097_data + (v1084_data * v1095_data));
          float v1100_data = s1[45];
          float v1102_data = ir3[3];
          ir3[3] = (v1102_data + (v1084_data * v1100_data));
          float v1105_data = s1[56];
          float v1107_data = ir3[4];
          ir3[4] = (v1107_data + (v1084_data * v1105_data));
          float v1110_data = s1[67];
          float v1112_data = ir3[5];
          ir3[5] = (v1112_data + (v1084_data * v1110_data));
          float v1115_data = s1[86];
          float v1117_data = ir3[6];
          ir3[6] = (v1117_data + (v1084_data * v1115_data));
          float v1120_data = s1[90];
          float v1122_data = ir3[7];
          ir3[7] = (v1122_data + (v1084_data * v1120_data));
          float v1125_data = s1[109];
          float v1127_data = ir3[8];
          ir3[8] = (v1127_data + (v1084_data * v1125_data));
          float v1130_data = s1[112];
          float v1132_data = ir3[9];
          ir3[9] = (v1132_data + (v1084_data * v1130_data));
          float v1135_data = s1[139];
          float v1137_data = ir3[10];
          ir3[10] = (v1137_data + (v1084_data * v1135_data));
          float v1140_data = s1[135];
          float v1142_data = ir3[11];
          ir3[11] = (v1142_data + (v1084_data * v1140_data));
          // r3 = ir3
          if (v25_g) {
            #pragma unroll
            for (int32_t v1144_n1 = 0; v1144_n1 < 12; ++v1144_n1) {
              float v1146_data = ir3[v1144_n1];
              r3[v1144_n1] = v1146_data;
            }
          }
          // glb_m2 = store{r>g}(r3);
          if (v25_g) {
            #pragma unroll
            for (int32_t v1147_i1 = 0; v1147_i1 < 12; ++v1147_i1) {
              float v1149_data = r3[v1147_i1];
              glb_m2[(v24_lead + (v1147_i1 * 12))] = v1149_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

