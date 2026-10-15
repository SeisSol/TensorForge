// === base name ===
kernel_9d7d68a45c59e7db

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_9d7d68a45c59e7db = {{32, 4, 1}, 32, 40, 1, 4, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_9d7d68a45c59e7db(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_9d7d68a45c59e7db(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_9d7d68a45c59e7db(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_9d7d68a45c59e7db, block.x * block.y * block.z, 256 * sizeof(float));
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
void launcher_kernel_9d7d68a45c59e7db(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_9d7d68a45c59e7db(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_9d7d68a45c59e7db, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_9d7d68a45c59e7db<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_9d7d68a45c59e7db(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: pack_operands=rows
    // launch: 32 lanes (40 active) x 4 per block = block 32x4x1, 1024 B shared, occupancy grid
    // operands:
    //   m0 40×6(40×6) {0..40}×{0..6} strided
    //   m1 40×8(40×8) {0..40}×{0..8} none ordered rows-packed
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
      size_t v321_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v321_flagSecondAhead < numElements0) ? v321_flagSecondAhead : v3_batchId1)];
      const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 48 + 0 + m2_extraOffset];
      bool v329_g = threadIdx.x < 16;
      // s0 = load{g>s}(peel_glb_m2[0, 1])
      __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
      if (v329_g) {
        __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
      }
      __pipeline_commit();
      uint32_t v325_flagWord = flagWordFirst;
      uint32_t v326_flagWordNext = flagWordSecond;
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < numElements0; v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
        const float *const __restrict__ wrap_glb_m2 = &m2[v12_batchId1 * 48 + 0 + m2_extraOffset];
        size_t v343_flagAheadAhead = v12_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v343_flagAheadAhead < numElements0) ? v343_flagAheadAhead : v12_batchId1)];
        const bool allowed = static_cast<bool>(v325_flagWord);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 240 + 0 + m0_extraOffset];
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r0[12]{};
          // ir0 = +(glb_m1 * s0)
          // [(0, 40), (0, 6)] [(0, 8)]
          float ir0[12]{};
          int32_t v26_lead = threadIdx.x % 32;
          bool v33_g = (v26_lead >= 5) && (v26_lead < 29);
          float v34_data = v33_g ? (glb_m1[(v26_lead + -5)]) : (0.0f);
          __syncwarp();
          float v35_data = s0[0];
          ir0[0] = (v34_data * v35_data);
          float v40_data = s0[8];
          ir0[2] = (v34_data * v40_data);
          float v45_data = s0[16];
          ir0[4] = (v34_data * v45_data);
          float v50_data = s0[24];
          ir0[6] = (v34_data * v50_data);
          float v55_data = s0[32];
          ir0[8] = (v34_data * v55_data);
          float v60_data = s0[40];
          ir0[10] = (v34_data * v60_data);
          float v66_data = v33_g ? (glb_m1[(v26_lead + 19)]) : (0.0f);
          float v67_data = s0[1];
          float v69_data = ir0[0];
          ir0[0] = (v69_data + (v66_data * v67_data));
          float v72_data = s0[9];
          float v74_data = ir0[2];
          ir0[2] = (v74_data + (v66_data * v72_data));
          float v77_data = s0[17];
          float v79_data = ir0[4];
          ir0[4] = (v79_data + (v66_data * v77_data));
          float v82_data = s0[25];
          float v84_data = ir0[6];
          ir0[6] = (v84_data + (v66_data * v82_data));
          float v87_data = s0[33];
          float v89_data = ir0[8];
          ir0[8] = (v89_data + (v66_data * v87_data));
          float v92_data = s0[41];
          float v94_data = ir0[10];
          ir0[10] = (v94_data + (v66_data * v92_data));
          float v98_data = v33_g ? (glb_m1[(v26_lead + 43)]) : (0.0f);
          float v99_data = s0[2];
          float v101_data = ir0[0];
          ir0[0] = (v101_data + (v98_data * v99_data));
          float v104_data = s0[10];
          float v106_data = ir0[2];
          ir0[2] = (v106_data + (v98_data * v104_data));
          float v109_data = s0[18];
          float v111_data = ir0[4];
          ir0[4] = (v111_data + (v98_data * v109_data));
          float v114_data = s0[26];
          float v116_data = ir0[6];
          ir0[6] = (v116_data + (v98_data * v114_data));
          float v119_data = s0[34];
          float v121_data = ir0[8];
          ir0[8] = (v121_data + (v98_data * v119_data));
          float v124_data = s0[42];
          float v126_data = ir0[10];
          ir0[10] = (v126_data + (v98_data * v124_data));
          float v130_data = v33_g ? (glb_m1[(v26_lead + 67)]) : (0.0f);
          float v131_data = s0[3];
          float v133_data = ir0[0];
          ir0[0] = (v133_data + (v130_data * v131_data));
          float v136_data = s0[11];
          float v138_data = ir0[2];
          ir0[2] = (v138_data + (v130_data * v136_data));
          float v141_data = s0[19];
          float v143_data = ir0[4];
          ir0[4] = (v143_data + (v130_data * v141_data));
          float v146_data = s0[27];
          float v148_data = ir0[6];
          ir0[6] = (v148_data + (v130_data * v146_data));
          float v151_data = s0[35];
          float v153_data = ir0[8];
          ir0[8] = (v153_data + (v130_data * v151_data));
          float v156_data = s0[43];
          float v158_data = ir0[10];
          ir0[10] = (v158_data + (v130_data * v156_data));
          float v162_data = v33_g ? (glb_m1[(v26_lead + 91)]) : (0.0f);
          float v163_data = s0[4];
          float v165_data = ir0[0];
          ir0[0] = (v165_data + (v162_data * v163_data));
          float v168_data = s0[12];
          float v170_data = ir0[2];
          ir0[2] = (v170_data + (v162_data * v168_data));
          float v173_data = s0[20];
          float v175_data = ir0[4];
          ir0[4] = (v175_data + (v162_data * v173_data));
          float v178_data = s0[28];
          float v180_data = ir0[6];
          ir0[6] = (v180_data + (v162_data * v178_data));
          float v183_data = s0[36];
          float v185_data = ir0[8];
          ir0[8] = (v185_data + (v162_data * v183_data));
          float v188_data = s0[44];
          float v190_data = ir0[10];
          ir0[10] = (v190_data + (v162_data * v188_data));
          float v194_data = v33_g ? (glb_m1[(v26_lead + 115)]) : (0.0f);
          float v195_data = s0[5];
          float v197_data = ir0[0];
          ir0[0] = (v197_data + (v194_data * v195_data));
          float v200_data = s0[13];
          float v202_data = ir0[2];
          ir0[2] = (v202_data + (v194_data * v200_data));
          float v205_data = s0[21];
          float v207_data = ir0[4];
          ir0[4] = (v207_data + (v194_data * v205_data));
          float v210_data = s0[29];
          float v212_data = ir0[6];
          ir0[6] = (v212_data + (v194_data * v210_data));
          float v215_data = s0[37];
          float v217_data = ir0[8];
          ir0[8] = (v217_data + (v194_data * v215_data));
          float v220_data = s0[45];
          float v222_data = ir0[10];
          ir0[10] = (v222_data + (v194_data * v220_data));
          float v226_data = v33_g ? (glb_m1[(v26_lead + 139)]) : (0.0f);
          float v227_data = s0[6];
          float v229_data = ir0[0];
          ir0[0] = (v229_data + (v226_data * v227_data));
          float v232_data = s0[14];
          float v234_data = ir0[2];
          ir0[2] = (v234_data + (v226_data * v232_data));
          float v237_data = s0[22];
          float v239_data = ir0[4];
          ir0[4] = (v239_data + (v226_data * v237_data));
          float v242_data = s0[30];
          float v244_data = ir0[6];
          ir0[6] = (v244_data + (v226_data * v242_data));
          float v247_data = s0[38];
          float v249_data = ir0[8];
          ir0[8] = (v249_data + (v226_data * v247_data));
          float v252_data = s0[46];
          float v254_data = ir0[10];
          ir0[10] = (v254_data + (v226_data * v252_data));
          float v258_data = v33_g ? (glb_m1[(v26_lead + 163)]) : (0.0f);
          float v259_data = s0[7];
          float v261_data = ir0[0];
          ir0[0] = (v261_data + (v258_data * v259_data));
          float v264_data = s0[15];
          float v266_data = ir0[2];
          ir0[2] = (v266_data + (v258_data * v264_data));
          float v269_data = s0[23];
          float v271_data = ir0[4];
          ir0[4] = (v271_data + (v258_data * v269_data));
          float v274_data = s0[31];
          float v276_data = ir0[6];
          ir0[6] = (v276_data + (v258_data * v274_data));
          float v279_data = s0[39];
          float v281_data = ir0[8];
          ir0[8] = (v281_data + (v258_data * v279_data));
          float v284_data = s0[47];
          float v286_data = ir0[10];
          ir0[10] = (v286_data + (v258_data * v284_data));
          // r0 = ir0
          #pragma unroll
          for (int32_t v290_n0 = 0; v290_n0 < 1; ++v290_n0) {
            #pragma unroll
            for (int32_t v291_n1 = 0; v291_n1 < 6; ++v291_n1) {
              int32_t v293_a = v290_n0 + (v291_n1 * 2);
              float v294_data = ir0[v293_a];
              r0[v293_a] = v294_data;
            }
          }
          bool v295_g = v26_lead < 8;
          if (v295_g) {
            #pragma unroll
            for (int32_t v296_n1 = 0; v296_n1 < 6; ++v296_n1) {
              int32_t v298_a = 1 + (v296_n1 * 2);
              float v299_data = ir0[v298_a];
              r0[v298_a] = v299_data;
            }
          }
          // glb_m0 = store{r>g}(r0);
          #pragma unroll
          for (int32_t v302_i0 = 0; v302_i0 < 1; ++v302_i0) {
            int32_t v308_lead = v26_lead + (v302_i0 * 32);
            #pragma unroll
            for (int32_t v303_i1 = 0; v303_i1 < 6; ++v303_i1) {
              float v306_data = r0[(v302_i0 + (v303_i1 * 2))];
              glb_m0[(v308_lead + (v303_i1 * 40))] = v306_data;
            }
          }
          if (v295_g) {
            int32_t v317_lead = v26_lead + 32_i32;
            #pragma unroll
            for (int32_t v312_i1 = 0; v312_i1 < 6; ++v312_i1) {
              float v315_data = r0[(1 + (v312_i1 * 2))];
              glb_m0[(v317_lead + (v312_i1 * 40))] = v315_data;
            }
          }
          __syncwarp();
        }
        else {
          __pipeline_wait_prior(0);
        }
        // s0 = load{g>s}(wrap_glb_m2[0, 1])
        __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
        if (v329_g) {
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
        }
        __pipeline_commit();
        v325_flagWord = v326_flagWordNext;
        v326_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
  }
}

