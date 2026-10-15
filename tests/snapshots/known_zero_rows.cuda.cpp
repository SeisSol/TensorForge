// === base name ===
kernel_16c0ceba21eabcf1

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_16c0ceba21eabcf1 = {{32, 4, 1}, 32, 40, 1, 4, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_16c0ceba21eabcf1(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_16c0ceba21eabcf1(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_16c0ceba21eabcf1(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_16c0ceba21eabcf1, block.x * block.y * block.z, 256 * sizeof(float));
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
void launcher_kernel_16c0ceba21eabcf1(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_16c0ceba21eabcf1(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_16c0ceba21eabcf1, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_16c0ceba21eabcf1<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_16c0ceba21eabcf1(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
      size_t v537_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v537_flagSecondAhead < numElements0) ? v537_flagSecondAhead : v3_batchId1)];
      const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 48 + 0 + m2_extraOffset];
      bool v545_g = threadIdx.x < 16;
      // s0 = load{g>s}(peel_glb_m2[0, 1])
      __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
      if (v545_g) {
        __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
      }
      __pipeline_commit();
      uint32_t v541_flagWord = flagWordFirst;
      uint32_t v542_flagWordNext = flagWordSecond;
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < numElements0; v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
        const float *const __restrict__ wrap_glb_m2 = &m2[v12_batchId1 * 48 + 0 + m2_extraOffset];
        size_t v559_flagAheadAhead = v12_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v559_flagAheadAhead < numElements0) ? v559_flagAheadAhead : v12_batchId1)];
        const bool allowed = static_cast<bool>(v541_flagWord);
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
          float v30_p = v29_data[1];
          __syncwarp();
          float v31_data = s0[0];
          ir0[1] = (v30_p * v31_data);
          float v37_data = s0[8];
          ir0[3] = (v30_p * v37_data);
          float v43_data = s0[16];
          ir0[5] = (v30_p * v43_data);
          float v49_data = s0[24];
          ir0[7] = (v30_p * v49_data);
          float v55_data = s0[32];
          ir0[9] = (v30_p * v55_data);
          float v61_data = s0[40];
          ir0[11] = (v30_p * v61_data);
          tensorforge::VectorT<float, 2> v66_data = *(tensorforge::VectorT<float, 2>*)&glb_m1[(64 + v27_a)];
          float v67_p = v66_data[0];
          float v68_data = s0[1];
          ir0[0] = (v67_p * v68_data);
          float v74_data = s0[9];
          ir0[2] = (v67_p * v74_data);
          float v80_data = s0[17];
          ir0[4] = (v67_p * v80_data);
          float v86_data = s0[25];
          ir0[6] = (v67_p * v86_data);
          float v92_data = s0[33];
          ir0[8] = (v67_p * v92_data);
          float v98_data = s0[41];
          ir0[10] = (v67_p * v98_data);
          float v103_p = v66_data[1];
          float v106_data = ir0[1];
          ir0[1] = (v106_data + (v103_p * v68_data));
          float v112_data = ir0[3];
          ir0[3] = (v112_data + (v103_p * v74_data));
          float v118_data = ir0[5];
          ir0[5] = (v118_data + (v103_p * v80_data));
          float v124_data = ir0[7];
          ir0[7] = (v124_data + (v103_p * v86_data));
          float v130_data = ir0[9];
          ir0[9] = (v130_data + (v103_p * v92_data));
          float v136_data = ir0[11];
          ir0[11] = (v136_data + (v103_p * v98_data));
          tensorforge::VectorT<float, 2> v139_data = *(tensorforge::VectorT<float, 2>*)&glb_m1[(128 + v27_a)];
          float v140_p = v139_data[1];
          float v141_data = s0[2];
          float v143_data = ir0[1];
          ir0[1] = (v143_data + (v140_p * v141_data));
          float v147_data = s0[10];
          float v149_data = ir0[3];
          ir0[3] = (v149_data + (v140_p * v147_data));
          float v153_data = s0[18];
          float v155_data = ir0[5];
          ir0[5] = (v155_data + (v140_p * v153_data));
          float v159_data = s0[26];
          float v161_data = ir0[7];
          ir0[7] = (v161_data + (v140_p * v159_data));
          float v165_data = s0[34];
          float v167_data = ir0[9];
          ir0[9] = (v167_data + (v140_p * v165_data));
          float v171_data = s0[42];
          float v173_data = ir0[11];
          ir0[11] = (v173_data + (v140_p * v171_data));
          tensorforge::VectorT<float, 2> v176_data = *(tensorforge::VectorT<float, 2>*)&glb_m1[(192 + v27_a)];
          float v177_p = v176_data[0];
          float v178_data = s0[3];
          float v180_data = ir0[0];
          ir0[0] = (v180_data + (v177_p * v178_data));
          float v184_data = s0[11];
          float v186_data = ir0[2];
          ir0[2] = (v186_data + (v177_p * v184_data));
          float v190_data = s0[19];
          float v192_data = ir0[4];
          ir0[4] = (v192_data + (v177_p * v190_data));
          float v196_data = s0[27];
          float v198_data = ir0[6];
          ir0[6] = (v198_data + (v177_p * v196_data));
          float v202_data = s0[35];
          float v204_data = ir0[8];
          ir0[8] = (v204_data + (v177_p * v202_data));
          float v208_data = s0[43];
          float v210_data = ir0[10];
          ir0[10] = (v210_data + (v177_p * v208_data));
          float v213_p = v176_data[1];
          float v216_data = ir0[1];
          ir0[1] = (v216_data + (v213_p * v178_data));
          float v222_data = ir0[3];
          ir0[3] = (v222_data + (v213_p * v184_data));
          float v228_data = ir0[5];
          ir0[5] = (v228_data + (v213_p * v190_data));
          float v234_data = ir0[7];
          ir0[7] = (v234_data + (v213_p * v196_data));
          float v240_data = ir0[9];
          ir0[9] = (v240_data + (v213_p * v202_data));
          float v246_data = ir0[11];
          ir0[11] = (v246_data + (v213_p * v208_data));
          tensorforge::VectorT<float, 2> v249_data = *(tensorforge::VectorT<float, 2>*)&glb_m1[(256 + v27_a)];
          float v250_p = v249_data[0];
          float v251_data = s0[4];
          float v253_data = ir0[0];
          ir0[0] = (v253_data + (v250_p * v251_data));
          float v257_data = s0[12];
          float v259_data = ir0[2];
          ir0[2] = (v259_data + (v250_p * v257_data));
          float v263_data = s0[20];
          float v265_data = ir0[4];
          ir0[4] = (v265_data + (v250_p * v263_data));
          float v269_data = s0[28];
          float v271_data = ir0[6];
          ir0[6] = (v271_data + (v250_p * v269_data));
          float v275_data = s0[36];
          float v277_data = ir0[8];
          ir0[8] = (v277_data + (v250_p * v275_data));
          float v281_data = s0[44];
          float v283_data = ir0[10];
          ir0[10] = (v283_data + (v250_p * v281_data));
          float v286_p = v249_data[1];
          float v289_data = ir0[1];
          ir0[1] = (v289_data + (v286_p * v251_data));
          float v295_data = ir0[3];
          ir0[3] = (v295_data + (v286_p * v257_data));
          float v301_data = ir0[5];
          ir0[5] = (v301_data + (v286_p * v263_data));
          float v307_data = ir0[7];
          ir0[7] = (v307_data + (v286_p * v269_data));
          float v313_data = ir0[9];
          ir0[9] = (v313_data + (v286_p * v275_data));
          float v319_data = ir0[11];
          ir0[11] = (v319_data + (v286_p * v281_data));
          tensorforge::VectorT<float, 2> v322_data = *(tensorforge::VectorT<float, 2>*)&glb_m1[(320 + v27_a)];
          float v323_p = v322_data[1];
          float v324_data = s0[5];
          float v326_data = ir0[1];
          ir0[1] = (v326_data + (v323_p * v324_data));
          float v330_data = s0[13];
          float v332_data = ir0[3];
          ir0[3] = (v332_data + (v323_p * v330_data));
          float v336_data = s0[21];
          float v338_data = ir0[5];
          ir0[5] = (v338_data + (v323_p * v336_data));
          float v342_data = s0[29];
          float v344_data = ir0[7];
          ir0[7] = (v344_data + (v323_p * v342_data));
          float v348_data = s0[37];
          float v350_data = ir0[9];
          ir0[9] = (v350_data + (v323_p * v348_data));
          float v354_data = s0[45];
          float v356_data = ir0[11];
          ir0[11] = (v356_data + (v323_p * v354_data));
          tensorforge::VectorT<float, 2> v359_data = *(tensorforge::VectorT<float, 2>*)&glb_m1[(384 + v27_a)];
          float v360_p = v359_data[0];
          float v361_data = s0[6];
          float v363_data = ir0[0];
          ir0[0] = (v363_data + (v360_p * v361_data));
          float v367_data = s0[14];
          float v369_data = ir0[2];
          ir0[2] = (v369_data + (v360_p * v367_data));
          float v373_data = s0[22];
          float v375_data = ir0[4];
          ir0[4] = (v375_data + (v360_p * v373_data));
          float v379_data = s0[30];
          float v381_data = ir0[6];
          ir0[6] = (v381_data + (v360_p * v379_data));
          float v385_data = s0[38];
          float v387_data = ir0[8];
          ir0[8] = (v387_data + (v360_p * v385_data));
          float v391_data = s0[46];
          float v393_data = ir0[10];
          ir0[10] = (v393_data + (v360_p * v391_data));
          float v396_p = v359_data[1];
          float v399_data = ir0[1];
          ir0[1] = (v399_data + (v396_p * v361_data));
          float v405_data = ir0[3];
          ir0[3] = (v405_data + (v396_p * v367_data));
          float v411_data = ir0[5];
          ir0[5] = (v411_data + (v396_p * v373_data));
          float v417_data = ir0[7];
          ir0[7] = (v417_data + (v396_p * v379_data));
          float v423_data = ir0[9];
          ir0[9] = (v423_data + (v396_p * v385_data));
          float v429_data = ir0[11];
          ir0[11] = (v429_data + (v396_p * v391_data));
          tensorforge::VectorT<float, 2> v432_data = *(tensorforge::VectorT<float, 2>*)&glb_m1[(448 + v27_a)];
          float v433_p = v432_data[0];
          float v434_data = s0[7];
          float v436_data = ir0[0];
          ir0[0] = (v436_data + (v433_p * v434_data));
          float v440_data = s0[15];
          float v442_data = ir0[2];
          ir0[2] = (v442_data + (v433_p * v440_data));
          float v446_data = s0[23];
          float v448_data = ir0[4];
          ir0[4] = (v448_data + (v433_p * v446_data));
          float v452_data = s0[31];
          float v454_data = ir0[6];
          ir0[6] = (v454_data + (v433_p * v452_data));
          float v458_data = s0[39];
          float v460_data = ir0[8];
          ir0[8] = (v460_data + (v433_p * v458_data));
          float v464_data = s0[47];
          float v466_data = ir0[10];
          ir0[10] = (v466_data + (v433_p * v464_data));
          float v469_p = v432_data[1];
          float v472_data = ir0[1];
          ir0[1] = (v472_data + (v469_p * v434_data));
          float v478_data = ir0[3];
          ir0[3] = (v478_data + (v469_p * v440_data));
          float v484_data = ir0[5];
          ir0[5] = (v484_data + (v469_p * v446_data));
          float v490_data = ir0[7];
          ir0[7] = (v490_data + (v469_p * v452_data));
          float v496_data = ir0[9];
          ir0[9] = (v496_data + (v469_p * v458_data));
          float v502_data = ir0[11];
          ir0[11] = (v502_data + (v469_p * v464_data));
          // r0 = ir0
          #pragma unroll
          for (int32_t v506_n0 = 0; v506_n0 < 1; ++v506_n0) {
            #pragma unroll
            for (int32_t v507_n1 = 0; v507_n1 < 6; ++v507_n1) {
              int32_t v509_a = v506_n0 + (v507_n1 * 2);
              float v510_data = ir0[v509_a];
              r0[v509_a] = v510_data;
            }
          }
          bool v511_g = v26_lead < 8;
          if (v511_g) {
            #pragma unroll
            for (int32_t v512_n1 = 0; v512_n1 < 6; ++v512_n1) {
              int32_t v514_a = 1 + (v512_n1 * 2);
              float v515_data = ir0[v514_a];
              r0[v514_a] = v515_data;
            }
          }
          // glb_m0 = store{r>g}(r0);
          #pragma unroll
          for (int32_t v518_i0 = 0; v518_i0 < 1; ++v518_i0) {
            int32_t v524_lead = v26_lead + (v518_i0 * 32);
            #pragma unroll
            for (int32_t v519_i1 = 0; v519_i1 < 6; ++v519_i1) {
              float v522_data = r0[(v518_i0 + (v519_i1 * 2))];
              glb_m0[(v524_lead + (v519_i1 * 40))] = v522_data;
            }
          }
          if (v511_g) {
            int32_t v533_lead = v26_lead + 32_i32;
            #pragma unroll
            for (int32_t v528_i1 = 0; v528_i1 < 6; ++v528_i1) {
              float v531_data = r0[(1 + (v528_i1 * 2))];
              glb_m0[(v533_lead + (v528_i1 * 40))] = v531_data;
            }
          }
          __syncwarp();
        }
        else {
          __pipeline_wait_prior(0);
        }
        // s0 = load{g>s}(wrap_glb_m2[0, 1])
        __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
        if (v545_g) {
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
        }
        __pipeline_commit();
        v541_flagWord = v542_flagWordNext;
        v542_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
  }
}

