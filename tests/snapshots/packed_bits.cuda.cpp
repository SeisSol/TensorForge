// === base name ===
kernel_e972c15451b81afd

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_e972c15451b81afd = {{32, 8, 1}, 32, 24, 1, 8, 2304, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_e972c15451b81afd(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_e972c15451b81afd(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_e972c15451b81afd(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_e972c15451b81afd, block.x * block.y * block.z, 576 * sizeof(float));
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
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 576 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_e972c15451b81afd(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_e972c15451b81afd(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_e972c15451b81afd, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_e972c15451b81afd<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256, 1)
 kernel_kernel_e972c15451b81afd(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: pack_operands=bits,preload_globals=1
    // launch: 32 lanes (24 active) x 8 per block = block 32x8x1, 2304 B shared, occupancy grid
    // operands:
    //   m0 24×6(24×6) {0..24}×{0..6} strided
    //   m1 24×7(24×7) {0..24}×{0..7} none ordered bits-packed
    //   m2 7×6(7×6) {0..7}×{0..6} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":24,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":576}],"shared_bytes":2304,"shared_elements":576,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[24,6]],"name":"m0","ordered":false,"parts":1,"shape":[24,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[24,7]],"name":"m1","ordered":true,"parts":1,"shape":[24,7],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[7,6]],"name":"m2","ordered":false,"parts":1,"shape":[7,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[24,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[24,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[24,7]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[24,7]},{"addressing":"strided","bbox":[[0,0],[7,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[7,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[64 * threadIdx.y + 64];
      size_t v3_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < numElements0) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      const float *const __restrict__ ptr_glb_m1 = &m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      if ((threadIdx.x + (threadIdx.y * 32)) < 47) {
        __pipeline_memcpy_async(&glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0], &ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0], 4);
      }
      __pipeline_commit();
      // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
      __pipeline_wait_prior(0);
      __syncthreads();
      float * __restrict__ s0 = &localShrMem0[0];
      const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
      size_t v319_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v319_flagSecondAhead < numElements0) ? v319_flagSecondAhead : v3_batchId1)];
      const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 42 + 0 + m2_extraOffset];
      bool v327_g = threadIdx.x < 10;
      // s0 = load{g>s}(peel_glb_m2[0, 1])
      __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
      if (v327_g) {
        __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
      }
      __pipeline_commit();
      uint32_t v323_flagWord = flagWordFirst;
      uint32_t v324_flagWordNext = flagWordSecond;
      for (size_t v16_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v16_batchId0 < numElements0; v16_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v17_ahead1 = v16_batchId0 + (gridDim.x * blockDim.y);
        size_t v19_batchId1 = (v17_ahead1 < numElements0) ? v17_ahead1 : v16_batchId0;
        const float *const __restrict__ wrap_glb_m2 = &m2[v19_batchId1 * 42 + 0 + m2_extraOffset];
        size_t v341_flagAheadAhead = v19_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v341_flagAheadAhead < numElements0) ? v341_flagAheadAhead : v19_batchId1)];
        const bool allowed = static_cast<bool>(v323_flagWord);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v16_batchId0 * 144 + 0 + m0_extraOffset];
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r0[6]{};
          // ir0 = +(glb_m1 * s0)
          // [(0, 24), (0, 6)] [(0, 7)]
          float ir0[6]{};
          int32_t v33_lead = threadIdx.x % 32;
          int32_t v35_bit = (1 << v33_lead) - 1;
          bool v42_g = v33_lead < 24;
          float v44_data = (v42_g && (((8390151 >> v33_lead) & 1) != 0)) ? (glb_m1[(__popc((8390151 & v35_bit)))]) : (0.0f);
          __syncwarp();
          float v45_data = s0[0];
          ir0[0] = (v44_data * v45_data);
          float v50_data = s0[7];
          ir0[1] = (v44_data * v50_data);
          float v55_data = s0[14];
          ir0[2] = (v44_data * v55_data);
          float v60_data = s0[21];
          ir0[3] = (v44_data * v60_data);
          float v65_data = s0[28];
          ir0[4] = (v44_data * v65_data);
          float v70_data = s0[35];
          ir0[5] = (v44_data * v70_data);
          float v82_data = (v42_g && (((16 >> v33_lead) & 1) != 0)) ? (glb_m1[((__popc((16 & v35_bit))) + 6)]) : (0.0f);
          float v83_data = s0[1];
          float v85_data = ir0[0];
          ir0[0] = (v85_data + (v82_data * v83_data));
          float v88_data = s0[8];
          float v90_data = ir0[1];
          ir0[1] = (v90_data + (v82_data * v88_data));
          float v93_data = s0[15];
          float v95_data = ir0[2];
          ir0[2] = (v95_data + (v82_data * v93_data));
          float v98_data = s0[22];
          float v100_data = ir0[3];
          ir0[3] = (v100_data + (v82_data * v98_data));
          float v103_data = s0[29];
          float v105_data = ir0[4];
          ir0[4] = (v105_data + (v82_data * v103_data));
          float v108_data = s0[36];
          float v110_data = ir0[5];
          ir0[5] = (v110_data + (v82_data * v108_data));
          float v120_data = (v42_g && (((16777215 >> v33_lead) & 1) != 0)) ? (glb_m1[((__popc((16777215 & v35_bit))) + 7)]) : (0.0f);
          float v121_data = s0[2];
          float v123_data = ir0[0];
          ir0[0] = (v123_data + (v120_data * v121_data));
          float v126_data = s0[9];
          float v128_data = ir0[1];
          ir0[1] = (v128_data + (v120_data * v126_data));
          float v131_data = s0[16];
          float v133_data = ir0[2];
          ir0[2] = (v133_data + (v120_data * v131_data));
          float v136_data = s0[23];
          float v138_data = ir0[3];
          ir0[3] = (v138_data + (v120_data * v136_data));
          float v141_data = s0[30];
          float v143_data = ir0[4];
          ir0[4] = (v143_data + (v120_data * v141_data));
          float v146_data = s0[37];
          float v148_data = ir0[5];
          ir0[5] = (v148_data + (v120_data * v146_data));
          float v158_data = (v42_g && (((8388609 >> v33_lead) & 1) != 0)) ? (glb_m1[((__popc((8388609 & v35_bit))) + 31)]) : (0.0f);
          float v159_data = s0[3];
          float v161_data = ir0[0];
          ir0[0] = (v161_data + (v158_data * v159_data));
          float v164_data = s0[10];
          float v166_data = ir0[1];
          ir0[1] = (v166_data + (v158_data * v164_data));
          float v169_data = s0[17];
          float v171_data = ir0[2];
          ir0[2] = (v171_data + (v158_data * v169_data));
          float v174_data = s0[24];
          float v176_data = ir0[3];
          ir0[3] = (v176_data + (v158_data * v174_data));
          float v179_data = s0[31];
          float v181_data = ir0[4];
          ir0[4] = (v181_data + (v158_data * v179_data));
          float v184_data = s0[38];
          float v186_data = ir0[5];
          ir0[5] = (v186_data + (v158_data * v184_data));
          float v196_data = (v42_g && (((665768 >> v33_lead) & 1) != 0)) ? (glb_m1[((__popc((665768 & v35_bit))) + 33)]) : (0.0f);
          float v197_data = s0[4];
          float v199_data = ir0[0];
          ir0[0] = (v199_data + (v196_data * v197_data));
          float v202_data = s0[11];
          float v204_data = ir0[1];
          ir0[1] = (v204_data + (v196_data * v202_data));
          float v207_data = s0[18];
          float v209_data = ir0[2];
          ir0[2] = (v209_data + (v196_data * v207_data));
          float v212_data = s0[25];
          float v214_data = ir0[3];
          ir0[3] = (v214_data + (v196_data * v212_data));
          float v217_data = s0[32];
          float v219_data = ir0[4];
          ir0[4] = (v219_data + (v196_data * v217_data));
          float v222_data = s0[39];
          float v224_data = ir0[5];
          ir0[5] = (v224_data + (v196_data * v222_data));
          float v234_data = (v42_g && (((126976 >> v33_lead) & 1) != 0)) ? (glb_m1[((__popc((126976 & v35_bit))) + 40)]) : (0.0f);
          float v235_data = s0[5];
          float v237_data = ir0[0];
          ir0[0] = (v237_data + (v234_data * v235_data));
          float v240_data = s0[12];
          float v242_data = ir0[1];
          ir0[1] = (v242_data + (v234_data * v240_data));
          float v245_data = s0[19];
          float v247_data = ir0[2];
          ir0[2] = (v247_data + (v234_data * v245_data));
          float v250_data = s0[26];
          float v252_data = ir0[3];
          ir0[3] = (v252_data + (v234_data * v250_data));
          float v255_data = s0[33];
          float v257_data = ir0[4];
          ir0[4] = (v257_data + (v234_data * v255_data));
          float v260_data = s0[40];
          float v262_data = ir0[5];
          ir0[5] = (v262_data + (v234_data * v260_data));
          float v272_data = (v42_g && (((4194306 >> v33_lead) & 1) != 0)) ? (glb_m1[((__popc((4194306 & v35_bit))) + 45)]) : (0.0f);
          float v273_data = s0[6];
          float v275_data = ir0[0];
          ir0[0] = (v275_data + (v272_data * v273_data));
          float v278_data = s0[13];
          float v280_data = ir0[1];
          ir0[1] = (v280_data + (v272_data * v278_data));
          float v283_data = s0[20];
          float v285_data = ir0[2];
          ir0[2] = (v285_data + (v272_data * v283_data));
          float v288_data = s0[27];
          float v290_data = ir0[3];
          ir0[3] = (v290_data + (v272_data * v288_data));
          float v293_data = s0[34];
          float v295_data = ir0[4];
          ir0[4] = (v295_data + (v272_data * v293_data));
          float v298_data = s0[41];
          float v300_data = ir0[5];
          ir0[5] = (v300_data + (v272_data * v298_data));
          // r0 = ir0
          if (v42_g) {
            #pragma unroll
            for (int32_t v305_n1 = 0; v305_n1 < 6; ++v305_n1) {
              float v307_data = ir0[v305_n1];
              r0[v305_n1] = v307_data;
            }
          }
          // glb_m0 = store{r>g}(r0);
          if (v42_g) {
            #pragma unroll
            for (int32_t v311_i1 = 0; v311_i1 < 6; ++v311_i1) {
              float v313_data = r0[v311_i1];
              glb_m0[(v33_lead + (v311_i1 * 24))] = v313_data;
            }
          }
          __syncwarp();
        }
        else {
          __pipeline_wait_prior(0);
        }
        // s0 = load{g>s}(wrap_glb_m2[0, 1])
        __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
        if (v327_g) {
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
        }
        __pipeline_commit();
        v323_flagWord = v324_flagWordNext;
        v324_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
  }
}

