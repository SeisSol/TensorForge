// === base name ===
kernel_70aa67ace7fa624d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_70aa67ace7fa624d = {{16, 8, 1}, 16, 12, 1, 8, 3584, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_70aa67ace7fa624d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_70aa67ace7fa624d(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_70aa67ace7fa624d(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_70aa67ace7fa624d, block.x * block.y * block.z, 896 * sizeof(float));
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
void launcher_kernel_70aa67ace7fa624d(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_70aa67ace7fa624d(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_70aa67ace7fa624d, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_70aa67ace7fa624d<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_70aa67ace7fa624d(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
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
      float* tempShrMem = &localShrMem0[96];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v12_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v12_batchId0 < numElements0; v12_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v13_ahead1 = v12_batchId0 + (gridDim.x * blockDim.y);
        size_t v15_batchId1 = (v13_ahead1 < numElements0) ? v13_ahead1 : v12_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v12_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v12_batchId0 * 72 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v12_batchId0 * 36 + 0 + m1_extraOffset];
          float *const __restrict__ glb_m2 = &m2[v12_batchId0 * 72 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[v12_batchId0 * 144 + 0 + m3_extraOffset];
          float r0[6]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v27_lead = threadIdx.x % 16;
          bool v28_g = v27_lead < 12;
          if (v28_g) {
            #pragma unroll
            for (int32_t v29_i1 = 0; v29_i1 < 6; ++v29_i1) {
              float v34_data = __ldcg(&glb_m0[(v27_lead + (v29_i1 * 12))]);
              r0[v29_i1] = v34_data;
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
          if (v28_g) {
            #pragma unroll
            for (int32_t v40_i1 = 0; v40_i1 < 12; ++v40_i1) {
              float v45_data = __ldcg(&glb_m3[(v27_lead + (v40_i1 * 12))]);
              r2[v40_i1] = v45_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[6]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // r1 = +(r0 * s0) + None
          // [(0, 12), (0, 6)] [(0, 6)]
          float v48_data = r0[0];
          float v49_data = s0[0];
          float v51_data = r1[0];
          r1[0] = (v51_data + (v48_data * v49_data));
          float v54_data = s0[6];
          float v56_data = r1[1];
          r1[1] = (v56_data + (v48_data * v54_data));
          float v59_data = s0[12];
          float v61_data = r1[2];
          r1[2] = (v61_data + (v48_data * v59_data));
          float v64_data = s0[18];
          float v66_data = r1[3];
          r1[3] = (v66_data + (v48_data * v64_data));
          float v69_data = s0[24];
          float v71_data = r1[4];
          r1[4] = (v71_data + (v48_data * v69_data));
          float v74_data = s0[30];
          float v76_data = r1[5];
          r1[5] = (v76_data + (v48_data * v74_data));
          float v78_data = r0[1];
          float v79_data = s0[1];
          float v81_data = r1[0];
          r1[0] = (v81_data + (v78_data * v79_data));
          float v84_data = s0[7];
          float v86_data = r1[1];
          r1[1] = (v86_data + (v78_data * v84_data));
          float v89_data = s0[13];
          float v91_data = r1[2];
          r1[2] = (v91_data + (v78_data * v89_data));
          float v94_data = s0[19];
          float v96_data = r1[3];
          r1[3] = (v96_data + (v78_data * v94_data));
          float v99_data = s0[25];
          float v101_data = r1[4];
          r1[4] = (v101_data + (v78_data * v99_data));
          float v104_data = s0[31];
          float v106_data = r1[5];
          r1[5] = (v106_data + (v78_data * v104_data));
          float v108_data = r0[2];
          float v109_data = s0[2];
          float v111_data = r1[0];
          r1[0] = (v111_data + (v108_data * v109_data));
          float v114_data = s0[8];
          float v116_data = r1[1];
          r1[1] = (v116_data + (v108_data * v114_data));
          float v119_data = s0[14];
          float v121_data = r1[2];
          r1[2] = (v121_data + (v108_data * v119_data));
          float v124_data = s0[20];
          float v126_data = r1[3];
          r1[3] = (v126_data + (v108_data * v124_data));
          float v129_data = s0[26];
          float v131_data = r1[4];
          r1[4] = (v131_data + (v108_data * v129_data));
          float v134_data = s0[32];
          float v136_data = r1[5];
          r1[5] = (v136_data + (v108_data * v134_data));
          float v138_data = r0[3];
          float v139_data = s0[3];
          float v141_data = r1[0];
          r1[0] = (v141_data + (v138_data * v139_data));
          float v144_data = s0[9];
          float v146_data = r1[1];
          r1[1] = (v146_data + (v138_data * v144_data));
          float v149_data = s0[15];
          float v151_data = r1[2];
          r1[2] = (v151_data + (v138_data * v149_data));
          float v154_data = s0[21];
          float v156_data = r1[3];
          r1[3] = (v156_data + (v138_data * v154_data));
          float v159_data = s0[27];
          float v161_data = r1[4];
          r1[4] = (v161_data + (v138_data * v159_data));
          float v164_data = s0[33];
          float v166_data = r1[5];
          r1[5] = (v166_data + (v138_data * v164_data));
          float v168_data = r0[4];
          float v169_data = s0[4];
          float v171_data = r1[0];
          r1[0] = (v171_data + (v168_data * v169_data));
          float v174_data = s0[10];
          float v176_data = r1[1];
          r1[1] = (v176_data + (v168_data * v174_data));
          float v179_data = s0[16];
          float v181_data = r1[2];
          r1[2] = (v181_data + (v168_data * v179_data));
          float v184_data = s0[22];
          float v186_data = r1[3];
          r1[3] = (v186_data + (v168_data * v184_data));
          float v189_data = s0[28];
          float v191_data = r1[4];
          r1[4] = (v191_data + (v168_data * v189_data));
          float v194_data = s0[34];
          float v196_data = r1[5];
          r1[5] = (v196_data + (v168_data * v194_data));
          float v198_data = r0[5];
          float v199_data = s0[5];
          float v201_data = r1[0];
          r1[0] = (v201_data + (v198_data * v199_data));
          float v204_data = s0[11];
          float v206_data = r1[1];
          r1[1] = (v206_data + (v198_data * v204_data));
          float v209_data = s0[17];
          float v211_data = r1[2];
          r1[2] = (v211_data + (v198_data * v209_data));
          float v214_data = s0[23];
          float v216_data = r1[3];
          r1[3] = (v216_data + (v198_data * v214_data));
          float v219_data = s0[29];
          float v221_data = r1[4];
          r1[4] = (v221_data + (v198_data * v219_data));
          float v224_data = s0[35];
          float v226_data = r1[5];
          r1[5] = (v226_data + (v198_data * v224_data));
          // wait(r2 = load{g>r}(glb_m3););
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // s1 = store{r>s}(localShrMem0, r1);
          if (v28_g) {
            #pragma unroll
            for (int32_t v228_i1 = 0; v228_i1 < 6; ++v228_i1) {
              float v230_data = r1[v228_i1];
              int32_t v234_a = v27_lead + (v228_i1 * 12);
              s1[(v234_a ^ ((v234_a >> 3) & 7))] = v230_data;
            }
          }
          float r3[6]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // ir3 = +(r2 * s1)
          // [(0, 12), (0, 6)] [(0, 12)]
          float ir3[6]{};
          float v240_data = r2[0];
          float v241_data = s1[0];
          float v243_data = ir3[0];
          ir3[0] = (v243_data + (v240_data * v241_data));
          float v246_data = s1[13];
          float v248_data = ir3[1];
          ir3[1] = (v248_data + (v240_data * v246_data));
          float v251_data = s1[27];
          float v253_data = ir3[2];
          ir3[2] = (v253_data + (v240_data * v251_data));
          float v256_data = s1[32];
          float v258_data = ir3[3];
          ir3[3] = (v258_data + (v240_data * v256_data));
          float v261_data = s1[54];
          float v263_data = ir3[4];
          ir3[4] = (v263_data + (v240_data * v261_data));
          float v266_data = s1[59];
          float v268_data = ir3[5];
          ir3[5] = (v268_data + (v240_data * v266_data));
          float v270_data = r2[1];
          float v271_data = s1[1];
          float v273_data = ir3[0];
          ir3[0] = (v273_data + (v270_data * v271_data));
          float v276_data = s1[12];
          float v278_data = ir3[1];
          ir3[1] = (v278_data + (v270_data * v276_data));
          float v281_data = s1[26];
          float v283_data = ir3[2];
          ir3[2] = (v283_data + (v270_data * v281_data));
          float v286_data = s1[33];
          float v288_data = ir3[3];
          ir3[3] = (v288_data + (v270_data * v286_data));
          float v291_data = s1[55];
          float v293_data = ir3[4];
          ir3[4] = (v293_data + (v270_data * v291_data));
          float v296_data = s1[58];
          float v298_data = ir3[5];
          ir3[5] = (v298_data + (v270_data * v296_data));
          float v300_data = r2[2];
          float v301_data = s1[2];
          float v303_data = ir3[0];
          ir3[0] = (v303_data + (v300_data * v301_data));
          float v306_data = s1[15];
          float v308_data = ir3[1];
          ir3[1] = (v308_data + (v300_data * v306_data));
          float v311_data = s1[25];
          float v313_data = ir3[2];
          ir3[2] = (v313_data + (v300_data * v311_data));
          float v316_data = s1[34];
          float v318_data = ir3[3];
          ir3[3] = (v318_data + (v300_data * v316_data));
          float v321_data = s1[52];
          float v323_data = ir3[4];
          ir3[4] = (v323_data + (v300_data * v321_data));
          float v326_data = s1[57];
          float v328_data = ir3[5];
          ir3[5] = (v328_data + (v300_data * v326_data));
          float v330_data = r2[3];
          float v331_data = s1[3];
          float v333_data = ir3[0];
          ir3[0] = (v333_data + (v330_data * v331_data));
          float v336_data = s1[14];
          float v338_data = ir3[1];
          ir3[1] = (v338_data + (v330_data * v336_data));
          float v341_data = s1[24];
          float v343_data = ir3[2];
          ir3[2] = (v343_data + (v330_data * v341_data));
          float v346_data = s1[35];
          float v348_data = ir3[3];
          ir3[3] = (v348_data + (v330_data * v346_data));
          float v351_data = s1[53];
          float v353_data = ir3[4];
          ir3[4] = (v353_data + (v330_data * v351_data));
          float v356_data = s1[56];
          float v358_data = ir3[5];
          ir3[5] = (v358_data + (v330_data * v356_data));
          float v360_data = r2[4];
          float v361_data = s1[4];
          float v363_data = ir3[0];
          ir3[0] = (v363_data + (v360_data * v361_data));
          float v366_data = s1[18];
          float v368_data = ir3[1];
          ir3[1] = (v368_data + (v360_data * v366_data));
          float v371_data = s1[31];
          float v373_data = ir3[2];
          ir3[2] = (v373_data + (v360_data * v371_data));
          float v376_data = s1[45];
          float v378_data = ir3[3];
          ir3[3] = (v378_data + (v360_data * v376_data));
          float v381_data = s1[50];
          float v383_data = ir3[4];
          ir3[4] = (v383_data + (v360_data * v381_data));
          float v386_data = s1[64];
          float v388_data = ir3[5];
          ir3[5] = (v388_data + (v360_data * v386_data));
          float v390_data = r2[5];
          float v391_data = s1[5];
          float v393_data = ir3[0];
          ir3[0] = (v393_data + (v390_data * v391_data));
          float v396_data = s1[19];
          float v398_data = ir3[1];
          ir3[1] = (v398_data + (v390_data * v396_data));
          float v401_data = s1[30];
          float v403_data = ir3[2];
          ir3[2] = (v403_data + (v390_data * v401_data));
          float v406_data = s1[44];
          float v408_data = ir3[3];
          ir3[3] = (v408_data + (v390_data * v406_data));
          float v411_data = s1[51];
          float v413_data = ir3[4];
          ir3[4] = (v413_data + (v390_data * v411_data));
          float v416_data = s1[65];
          float v418_data = ir3[5];
          ir3[5] = (v418_data + (v390_data * v416_data));
          float v420_data = r2[6];
          float v421_data = s1[6];
          float v423_data = ir3[0];
          ir3[0] = (v423_data + (v420_data * v421_data));
          float v426_data = s1[16];
          float v428_data = ir3[1];
          ir3[1] = (v428_data + (v420_data * v426_data));
          float v431_data = s1[29];
          float v433_data = ir3[2];
          ir3[2] = (v433_data + (v420_data * v431_data));
          float v436_data = s1[47];
          float v438_data = ir3[3];
          ir3[3] = (v438_data + (v420_data * v436_data));
          float v441_data = s1[48];
          float v443_data = ir3[4];
          ir3[4] = (v443_data + (v420_data * v441_data));
          float v446_data = s1[66];
          float v448_data = ir3[5];
          ir3[5] = (v448_data + (v420_data * v446_data));
          float v450_data = r2[7];
          float v451_data = s1[7];
          float v453_data = ir3[0];
          ir3[0] = (v453_data + (v450_data * v451_data));
          float v456_data = s1[17];
          float v458_data = ir3[1];
          ir3[1] = (v458_data + (v450_data * v456_data));
          float v461_data = s1[28];
          float v463_data = ir3[2];
          ir3[2] = (v463_data + (v450_data * v461_data));
          float v466_data = s1[46];
          float v468_data = ir3[3];
          ir3[3] = (v468_data + (v450_data * v466_data));
          float v471_data = s1[49];
          float v473_data = ir3[4];
          ir3[4] = (v473_data + (v450_data * v471_data));
          float v476_data = s1[67];
          float v478_data = ir3[5];
          ir3[5] = (v478_data + (v450_data * v476_data));
          float v480_data = r2[8];
          float v481_data = s1[9];
          float v483_data = ir3[0];
          ir3[0] = (v483_data + (v480_data * v481_data));
          float v486_data = s1[22];
          float v488_data = ir3[1];
          ir3[1] = (v488_data + (v480_data * v486_data));
          float v491_data = s1[36];
          float v493_data = ir3[2];
          ir3[2] = (v493_data + (v480_data * v491_data));
          float v496_data = s1[41];
          float v498_data = ir3[3];
          ir3[3] = (v498_data + (v480_data * v496_data));
          float v501_data = s1[63];
          float v503_data = ir3[4];
          ir3[4] = (v503_data + (v480_data * v501_data));
          float v506_data = s1[68];
          float v508_data = ir3[5];
          ir3[5] = (v508_data + (v480_data * v506_data));
          float v510_data = r2[9];
          float v511_data = s1[8];
          float v513_data = ir3[0];
          ir3[0] = (v513_data + (v510_data * v511_data));
          float v516_data = s1[23];
          float v518_data = ir3[1];
          ir3[1] = (v518_data + (v510_data * v516_data));
          float v521_data = s1[37];
          float v523_data = ir3[2];
          ir3[2] = (v523_data + (v510_data * v521_data));
          float v526_data = s1[40];
          float v528_data = ir3[3];
          ir3[3] = (v528_data + (v510_data * v526_data));
          float v531_data = s1[62];
          float v533_data = ir3[4];
          ir3[4] = (v533_data + (v510_data * v531_data));
          float v536_data = s1[69];
          float v538_data = ir3[5];
          ir3[5] = (v538_data + (v510_data * v536_data));
          float v540_data = r2[10];
          float v541_data = s1[11];
          float v543_data = ir3[0];
          ir3[0] = (v543_data + (v540_data * v541_data));
          float v546_data = s1[20];
          float v548_data = ir3[1];
          ir3[1] = (v548_data + (v540_data * v546_data));
          float v551_data = s1[38];
          float v553_data = ir3[2];
          ir3[2] = (v553_data + (v540_data * v551_data));
          float v556_data = s1[43];
          float v558_data = ir3[3];
          ir3[3] = (v558_data + (v540_data * v556_data));
          float v561_data = s1[61];
          float v563_data = ir3[4];
          ir3[4] = (v563_data + (v540_data * v561_data));
          float v566_data = s1[70];
          float v568_data = ir3[5];
          ir3[5] = (v568_data + (v540_data * v566_data));
          float v570_data = r2[11];
          float v571_data = s1[10];
          float v573_data = ir3[0];
          ir3[0] = (v573_data + (v570_data * v571_data));
          float v576_data = s1[21];
          float v578_data = ir3[1];
          ir3[1] = (v578_data + (v570_data * v576_data));
          float v581_data = s1[39];
          float v583_data = ir3[2];
          ir3[2] = (v583_data + (v570_data * v581_data));
          float v586_data = s1[42];
          float v588_data = ir3[3];
          ir3[3] = (v588_data + (v570_data * v586_data));
          float v591_data = s1[60];
          float v593_data = ir3[4];
          ir3[4] = (v593_data + (v570_data * v591_data));
          float v596_data = s1[71];
          float v598_data = ir3[5];
          ir3[5] = (v598_data + (v570_data * v596_data));
          // r3 = ir3
          if (v28_g) {
            #pragma unroll
            for (int32_t v600_n1 = 0; v600_n1 < 6; ++v600_n1) {
              float v602_data = ir3[v600_n1];
              r3[v600_n1] = v602_data;
            }
          }
          // glb_m2 = store{r>g}(r3);
          if (v28_g) {
            #pragma unroll
            for (int32_t v603_i1 = 0; v603_i1 < 6; ++v603_i1) {
              float v605_data = r3[v603_i1];
              glb_m2[(v27_lead + (v603_i1 * 12))] = v605_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

