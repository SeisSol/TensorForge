// === base name ===
kernel_dae51b27816bce39

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_dae51b27816bce39 = {{32, 4, 1}, 32, 40, 1, 4, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_dae51b27816bce39(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_dae51b27816bce39(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_dae51b27816bce39(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_dae51b27816bce39, block.x * block.y * block.z, 256 * sizeof(float));
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
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_dae51b27816bce39(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_dae51b27816bce39(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_dae51b27816bce39, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_dae51b27816bce39<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_dae51b27816bce39(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (40 active) x 4 per block = block 32x4x1, 1024 B shared, occupancy grid
    // operands:
    //   m0 40×6(40×6) {0..40}×{0..6} strided
    //   m1 40×8(40×8) {0..40}×{0..8} none ordered
    //   m2 8×6(8×6) {0..8}×{0..6} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":40,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[40,6]],"name":"m0","ordered":false,"parts":1,"shape":[40,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[40,8]],"name":"m1","ordered":true,"parts":1,"shape":[40,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,6]],"name":"m2","ordered":false,"parts":1,"shape":[8,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[40,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[40,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[40,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[40,8]},{"addressing":"strided","bbox":[[0,0],[8,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[64 * threadIdx.y + 0];
      size_t v3_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < numElements0) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      const float *const __restrict__ glb_m1 = &m1[0];
      float * __restrict__ s0 = &localShrMem0[0];
      const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
      size_t v645_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v645_flagSecondAhead < numElements0) ? v645_flagSecondAhead : v3_batchId1)];
      const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 48 + 0 + m2_extraOffset];
      bool v653_g = threadIdx.x < 16;
      // s0 = load{g>s}(peel_glb_m2[0, 1])
      __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
      if (v653_g) {
        __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
      }
      __pipeline_commit();
      uint32_t v649_flagWord = flagWordFirst;
      uint32_t v650_flagWordNext = flagWordSecond;
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < numElements0; v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
        const float *const __restrict__ wrap_glb_m2 = &m2[v12_batchId1 * 48 + 0 + m2_extraOffset];
        size_t v667_flagAheadAhead = v12_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v667_flagAheadAhead < numElements0) ? v667_flagAheadAhead : v12_batchId1)];
        const bool allowed = static_cast<bool>(v649_flagWord);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 240 + 0 + m0_extraOffset];
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r0[12]{};
          // ir0 = +(glb_m1 * s0)
          // [(0, 40), (0, 6)] [(0, 8)]
          float ir0[12]{};
          int32_t v26_lead = threadIdx.x % 32;
          int32_t v27_a = v26_lead * 2;
          tensorforge::VectorT<float, 2> v29_data = *(tensorforge::VectorT<float, 2>*)&glb_m1[v27_a];
          float v30_p = v29_data[0];
          __syncwarp();
          float v31_data = s0[0];
          ir0[0] = (v30_p * v31_data);
          float v37_data = s0[8];
          ir0[2] = (v30_p * v37_data);
          float v43_data = s0[16];
          ir0[4] = (v30_p * v43_data);
          float v49_data = s0[24];
          ir0[6] = (v30_p * v49_data);
          float v55_data = s0[32];
          ir0[8] = (v30_p * v55_data);
          float v61_data = s0[40];
          ir0[10] = (v30_p * v61_data);
          float v66_p = v29_data[1];
          ir0[1] = (v66_p * v31_data);
          ir0[3] = (v66_p * v37_data);
          ir0[5] = (v66_p * v43_data);
          ir0[7] = (v66_p * v49_data);
          ir0[9] = (v66_p * v55_data);
          ir0[11] = (v66_p * v61_data);
          tensorforge::VectorT<float, 2> v102_data = *(tensorforge::VectorT<float, 2>*)&glb_m1[(64 + v27_a)];
          float v103_p = v102_data[0];
          float v104_data = s0[1];
          float v106_data = ir0[0];
          ir0[0] = (v106_data + (v103_p * v104_data));
          float v110_data = s0[9];
          float v112_data = ir0[2];
          ir0[2] = (v112_data + (v103_p * v110_data));
          float v116_data = s0[17];
          float v118_data = ir0[4];
          ir0[4] = (v118_data + (v103_p * v116_data));
          float v122_data = s0[25];
          float v124_data = ir0[6];
          ir0[6] = (v124_data + (v103_p * v122_data));
          float v128_data = s0[33];
          float v130_data = ir0[8];
          ir0[8] = (v130_data + (v103_p * v128_data));
          float v134_data = s0[41];
          float v136_data = ir0[10];
          ir0[10] = (v136_data + (v103_p * v134_data));
          float v139_p = v102_data[1];
          float v142_data = ir0[1];
          ir0[1] = (v142_data + (v139_p * v104_data));
          float v148_data = ir0[3];
          ir0[3] = (v148_data + (v139_p * v110_data));
          float v154_data = ir0[5];
          ir0[5] = (v154_data + (v139_p * v116_data));
          float v160_data = ir0[7];
          ir0[7] = (v160_data + (v139_p * v122_data));
          float v166_data = ir0[9];
          ir0[9] = (v166_data + (v139_p * v128_data));
          float v172_data = ir0[11];
          ir0[11] = (v172_data + (v139_p * v134_data));
          tensorforge::VectorT<float, 2> v175_data = *(tensorforge::VectorT<float, 2>*)&glb_m1[(128 + v27_a)];
          float v176_p = v175_data[0];
          float v177_data = s0[2];
          float v179_data = ir0[0];
          ir0[0] = (v179_data + (v176_p * v177_data));
          float v183_data = s0[10];
          float v185_data = ir0[2];
          ir0[2] = (v185_data + (v176_p * v183_data));
          float v189_data = s0[18];
          float v191_data = ir0[4];
          ir0[4] = (v191_data + (v176_p * v189_data));
          float v195_data = s0[26];
          float v197_data = ir0[6];
          ir0[6] = (v197_data + (v176_p * v195_data));
          float v201_data = s0[34];
          float v203_data = ir0[8];
          ir0[8] = (v203_data + (v176_p * v201_data));
          float v207_data = s0[42];
          float v209_data = ir0[10];
          ir0[10] = (v209_data + (v176_p * v207_data));
          float v212_p = v175_data[1];
          float v215_data = ir0[1];
          ir0[1] = (v215_data + (v212_p * v177_data));
          float v221_data = ir0[3];
          ir0[3] = (v221_data + (v212_p * v183_data));
          float v227_data = ir0[5];
          ir0[5] = (v227_data + (v212_p * v189_data));
          float v233_data = ir0[7];
          ir0[7] = (v233_data + (v212_p * v195_data));
          float v239_data = ir0[9];
          ir0[9] = (v239_data + (v212_p * v201_data));
          float v245_data = ir0[11];
          ir0[11] = (v245_data + (v212_p * v207_data));
          tensorforge::VectorT<float, 2> v248_data = *(tensorforge::VectorT<float, 2>*)&glb_m1[(192 + v27_a)];
          float v249_p = v248_data[0];
          float v250_data = s0[3];
          float v252_data = ir0[0];
          ir0[0] = (v252_data + (v249_p * v250_data));
          float v256_data = s0[11];
          float v258_data = ir0[2];
          ir0[2] = (v258_data + (v249_p * v256_data));
          float v262_data = s0[19];
          float v264_data = ir0[4];
          ir0[4] = (v264_data + (v249_p * v262_data));
          float v268_data = s0[27];
          float v270_data = ir0[6];
          ir0[6] = (v270_data + (v249_p * v268_data));
          float v274_data = s0[35];
          float v276_data = ir0[8];
          ir0[8] = (v276_data + (v249_p * v274_data));
          float v280_data = s0[43];
          float v282_data = ir0[10];
          ir0[10] = (v282_data + (v249_p * v280_data));
          float v285_p = v248_data[1];
          float v288_data = ir0[1];
          ir0[1] = (v288_data + (v285_p * v250_data));
          float v294_data = ir0[3];
          ir0[3] = (v294_data + (v285_p * v256_data));
          float v300_data = ir0[5];
          ir0[5] = (v300_data + (v285_p * v262_data));
          float v306_data = ir0[7];
          ir0[7] = (v306_data + (v285_p * v268_data));
          float v312_data = ir0[9];
          ir0[9] = (v312_data + (v285_p * v274_data));
          float v318_data = ir0[11];
          ir0[11] = (v318_data + (v285_p * v280_data));
          tensorforge::VectorT<float, 2> v321_data = *(tensorforge::VectorT<float, 2>*)&glb_m1[(256 + v27_a)];
          float v322_p = v321_data[0];
          float v323_data = s0[4];
          float v325_data = ir0[0];
          ir0[0] = (v325_data + (v322_p * v323_data));
          float v329_data = s0[12];
          float v331_data = ir0[2];
          ir0[2] = (v331_data + (v322_p * v329_data));
          float v335_data = s0[20];
          float v337_data = ir0[4];
          ir0[4] = (v337_data + (v322_p * v335_data));
          float v341_data = s0[28];
          float v343_data = ir0[6];
          ir0[6] = (v343_data + (v322_p * v341_data));
          float v347_data = s0[36];
          float v349_data = ir0[8];
          ir0[8] = (v349_data + (v322_p * v347_data));
          float v353_data = s0[44];
          float v355_data = ir0[10];
          ir0[10] = (v355_data + (v322_p * v353_data));
          float v358_p = v321_data[1];
          float v361_data = ir0[1];
          ir0[1] = (v361_data + (v358_p * v323_data));
          float v367_data = ir0[3];
          ir0[3] = (v367_data + (v358_p * v329_data));
          float v373_data = ir0[5];
          ir0[5] = (v373_data + (v358_p * v335_data));
          float v379_data = ir0[7];
          ir0[7] = (v379_data + (v358_p * v341_data));
          float v385_data = ir0[9];
          ir0[9] = (v385_data + (v358_p * v347_data));
          float v391_data = ir0[11];
          ir0[11] = (v391_data + (v358_p * v353_data));
          tensorforge::VectorT<float, 2> v394_data = *(tensorforge::VectorT<float, 2>*)&glb_m1[(320 + v27_a)];
          float v395_p = v394_data[0];
          float v396_data = s0[5];
          float v398_data = ir0[0];
          ir0[0] = (v398_data + (v395_p * v396_data));
          float v402_data = s0[13];
          float v404_data = ir0[2];
          ir0[2] = (v404_data + (v395_p * v402_data));
          float v408_data = s0[21];
          float v410_data = ir0[4];
          ir0[4] = (v410_data + (v395_p * v408_data));
          float v414_data = s0[29];
          float v416_data = ir0[6];
          ir0[6] = (v416_data + (v395_p * v414_data));
          float v420_data = s0[37];
          float v422_data = ir0[8];
          ir0[8] = (v422_data + (v395_p * v420_data));
          float v426_data = s0[45];
          float v428_data = ir0[10];
          ir0[10] = (v428_data + (v395_p * v426_data));
          float v431_p = v394_data[1];
          float v434_data = ir0[1];
          ir0[1] = (v434_data + (v431_p * v396_data));
          float v440_data = ir0[3];
          ir0[3] = (v440_data + (v431_p * v402_data));
          float v446_data = ir0[5];
          ir0[5] = (v446_data + (v431_p * v408_data));
          float v452_data = ir0[7];
          ir0[7] = (v452_data + (v431_p * v414_data));
          float v458_data = ir0[9];
          ir0[9] = (v458_data + (v431_p * v420_data));
          float v464_data = ir0[11];
          ir0[11] = (v464_data + (v431_p * v426_data));
          tensorforge::VectorT<float, 2> v467_data = *(tensorforge::VectorT<float, 2>*)&glb_m1[(384 + v27_a)];
          float v468_p = v467_data[0];
          float v469_data = s0[6];
          float v471_data = ir0[0];
          ir0[0] = (v471_data + (v468_p * v469_data));
          float v475_data = s0[14];
          float v477_data = ir0[2];
          ir0[2] = (v477_data + (v468_p * v475_data));
          float v481_data = s0[22];
          float v483_data = ir0[4];
          ir0[4] = (v483_data + (v468_p * v481_data));
          float v487_data = s0[30];
          float v489_data = ir0[6];
          ir0[6] = (v489_data + (v468_p * v487_data));
          float v493_data = s0[38];
          float v495_data = ir0[8];
          ir0[8] = (v495_data + (v468_p * v493_data));
          float v499_data = s0[46];
          float v501_data = ir0[10];
          ir0[10] = (v501_data + (v468_p * v499_data));
          float v504_p = v467_data[1];
          float v507_data = ir0[1];
          ir0[1] = (v507_data + (v504_p * v469_data));
          float v513_data = ir0[3];
          ir0[3] = (v513_data + (v504_p * v475_data));
          float v519_data = ir0[5];
          ir0[5] = (v519_data + (v504_p * v481_data));
          float v525_data = ir0[7];
          ir0[7] = (v525_data + (v504_p * v487_data));
          float v531_data = ir0[9];
          ir0[9] = (v531_data + (v504_p * v493_data));
          float v537_data = ir0[11];
          ir0[11] = (v537_data + (v504_p * v499_data));
          tensorforge::VectorT<float, 2> v540_data = *(tensorforge::VectorT<float, 2>*)&glb_m1[(448 + v27_a)];
          float v541_p = v540_data[0];
          float v542_data = s0[7];
          float v544_data = ir0[0];
          ir0[0] = (v544_data + (v541_p * v542_data));
          float v548_data = s0[15];
          float v550_data = ir0[2];
          ir0[2] = (v550_data + (v541_p * v548_data));
          float v554_data = s0[23];
          float v556_data = ir0[4];
          ir0[4] = (v556_data + (v541_p * v554_data));
          float v560_data = s0[31];
          float v562_data = ir0[6];
          ir0[6] = (v562_data + (v541_p * v560_data));
          float v566_data = s0[39];
          float v568_data = ir0[8];
          ir0[8] = (v568_data + (v541_p * v566_data));
          float v572_data = s0[47];
          float v574_data = ir0[10];
          ir0[10] = (v574_data + (v541_p * v572_data));
          float v577_p = v540_data[1];
          float v580_data = ir0[1];
          ir0[1] = (v580_data + (v577_p * v542_data));
          float v586_data = ir0[3];
          ir0[3] = (v586_data + (v577_p * v548_data));
          float v592_data = ir0[5];
          ir0[5] = (v592_data + (v577_p * v554_data));
          float v598_data = ir0[7];
          ir0[7] = (v598_data + (v577_p * v560_data));
          float v604_data = ir0[9];
          ir0[9] = (v604_data + (v577_p * v566_data));
          float v610_data = ir0[11];
          ir0[11] = (v610_data + (v577_p * v572_data));
          // r0 = ir0
          #pragma unroll
          for (int32_t v614_n0 = 0; v614_n0 < 1; ++v614_n0) {
            #pragma unroll
            for (int32_t v615_n1 = 0; v615_n1 < 6; ++v615_n1) {
              int32_t v617_a = v614_n0 + (v615_n1 * 2);
              float v618_data = ir0[v617_a];
              r0[v617_a] = v618_data;
            }
          }
          bool v619_g = v26_lead < 8;
          if (v619_g) {
            #pragma unroll
            for (int32_t v620_n1 = 0; v620_n1 < 6; ++v620_n1) {
              int32_t v622_a = 1 + (v620_n1 * 2);
              float v623_data = ir0[v622_a];
              r0[v622_a] = v623_data;
            }
          }
          // glb_m0 = store{r>g}(r0);
          #pragma unroll
          for (int32_t v626_i0 = 0; v626_i0 < 1; ++v626_i0) {
            int32_t v632_lead = v26_lead + (v626_i0 * 32);
            #pragma unroll
            for (int32_t v627_i1 = 0; v627_i1 < 6; ++v627_i1) {
              float v630_data = r0[(v626_i0 + (v627_i1 * 2))];
              glb_m0[(v632_lead + (v627_i1 * 40))] = v630_data;
            }
          }
          if (v619_g) {
            int32_t v641_lead = v26_lead + 32_i32;
            #pragma unroll
            for (int32_t v636_i1 = 0; v636_i1 < 6; ++v636_i1) {
              float v639_data = r0[(1 + (v636_i1 * 2))];
              glb_m0[(v641_lead + (v636_i1 * 40))] = v639_data;
            }
          }
          __syncwarp();
        }
        else {
          __pipeline_wait_prior(0);
        }
        // s0 = load{g>s}(wrap_glb_m2[0, 1])
        __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
        if (v653_g) {
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
        }
        __pipeline_commit();
        v649_flagWord = v650_flagWordNext;
        v650_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
  }
}

