// === base name ===
kernel_f1e392cd3e98dca9

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_f1e392cd3e98dca9 = {{16, 8, 1}, 16, 16, 1, 8, 2560, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_f1e392cd3e98dca9(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_f1e392cd3e98dca9(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_f1e392cd3e98dca9(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_f1e392cd3e98dca9, block.x * block.y * block.z, 640 * sizeof(float));
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
  config.sharedMemBytes = 640 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_f1e392cd3e98dca9(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_f1e392cd3e98dca9(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_f1e392cd3e98dca9, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_f1e392cd3e98dca9<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_f1e392cd3e98dca9(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 8 per block = block 16x8x1, 2560 B shared, occupancy grid
    // operands:
    //   m0 16×16(16×16) {0..16}×{0..16} strided
    //   m1 16×16(16×16) {0..16}×{0..16} strided
    //   m2 16×16(16×16) {0..16}×{0..16} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":640}],"shared_bytes":2560,"shared_elements":640,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[80 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v5_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v5_batchId0 < numElements0; v5_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v6_ahead1 = v5_batchId0 + (gridDim.x * blockDim.y);
        size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 256 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 46 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v19_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v20_i0 = 0; v20_i0 < 1; ++v20_i0) {
            int32_t v23_lead = v19_lead + (v20_i0 * 16);
            #pragma unroll
            for (int32_t v21_i1 = 0; v21_i1 < 16; ++v21_i1) {
              float v26_data = __ldcg(&glb_m1[(v23_lead + (v21_i1 * 16))]);
              r0[(v20_i0 + v21_i1)] = v26_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 16], &glb_m2[0 + 0 + 1 * threadIdx.x + 16], 4);
          if (threadIdx.x < 14) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[16]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // r1 = +(r0 * s0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float ir1[16]{};
          float v33_data = r0[0];
          float v34_data = s0[0];
          float v36_data = ir1[0];
          ir1[0] = (v36_data + (v33_data * v34_data));
          float v39_data = s0[2];
          float v41_data = ir1[1];
          ir1[1] = (v41_data + (v33_data * v39_data));
          float v57_data = r0[1];
          float v58_data = s0[1];
          float v60_data = ir1[0];
          ir1[0] = (v60_data + (v57_data * v58_data));
          float v63_data = s0[3];
          float v65_data = ir1[1];
          ir1[1] = (v65_data + (v57_data * v63_data));
          float v68_data = s0[5];
          float v70_data = ir1[2];
          ir1[2] = (v70_data + (v57_data * v68_data));
          float v85_data = r0[2];
          float v87_data = s0[4];
          float v89_data = ir1[1];
          ir1[1] = (v89_data + (v85_data * v87_data));
          float v92_data = s0[6];
          float v94_data = ir1[2];
          ir1[2] = (v94_data + (v85_data * v92_data));
          float v97_data = s0[8];
          float v99_data = ir1[3];
          ir1[3] = (v99_data + (v85_data * v97_data));
          float v113_data = r0[3];
          float v116_data = s0[7];
          float v118_data = ir1[2];
          ir1[2] = (v118_data + (v113_data * v116_data));
          float v121_data = s0[9];
          float v123_data = ir1[3];
          ir1[3] = (v123_data + (v113_data * v121_data));
          float v126_data = s0[11];
          float v128_data = ir1[4];
          ir1[4] = (v128_data + (v113_data * v126_data));
          float v141_data = r0[4];
          float v145_data = s0[10];
          float v147_data = ir1[3];
          ir1[3] = (v147_data + (v141_data * v145_data));
          float v150_data = s0[12];
          float v152_data = ir1[4];
          ir1[4] = (v152_data + (v141_data * v150_data));
          float v155_data = s0[14];
          float v157_data = ir1[5];
          ir1[5] = (v157_data + (v141_data * v155_data));
          float v169_data = r0[5];
          float v174_data = s0[13];
          float v176_data = ir1[4];
          ir1[4] = (v176_data + (v169_data * v174_data));
          float v179_data = s0[15];
          float v181_data = ir1[5];
          ir1[5] = (v181_data + (v169_data * v179_data));
          float v184_data = s0[17];
          float v186_data = ir1[6];
          ir1[6] = (v186_data + (v169_data * v184_data));
          float v197_data = r0[6];
          float v203_data = s0[16];
          float v205_data = ir1[5];
          ir1[5] = (v205_data + (v197_data * v203_data));
          float v208_data = s0[18];
          float v210_data = ir1[6];
          ir1[6] = (v210_data + (v197_data * v208_data));
          float v213_data = s0[20];
          float v215_data = ir1[7];
          ir1[7] = (v215_data + (v197_data * v213_data));
          float v225_data = r0[7];
          float v232_data = s0[19];
          float v234_data = ir1[6];
          ir1[6] = (v234_data + (v225_data * v232_data));
          float v237_data = s0[21];
          float v239_data = ir1[7];
          ir1[7] = (v239_data + (v225_data * v237_data));
          float v242_data = s0[23];
          float v244_data = ir1[8];
          ir1[8] = (v244_data + (v225_data * v242_data));
          float v253_data = r0[8];
          float v261_data = s0[22];
          float v263_data = ir1[7];
          ir1[7] = (v263_data + (v253_data * v261_data));
          float v266_data = s0[24];
          float v268_data = ir1[8];
          ir1[8] = (v268_data + (v253_data * v266_data));
          float v271_data = s0[26];
          float v273_data = ir1[9];
          ir1[9] = (v273_data + (v253_data * v271_data));
          float v281_data = r0[9];
          float v290_data = s0[25];
          float v292_data = ir1[8];
          ir1[8] = (v292_data + (v281_data * v290_data));
          float v295_data = s0[27];
          float v297_data = ir1[9];
          ir1[9] = (v297_data + (v281_data * v295_data));
          float v300_data = s0[29];
          float v302_data = ir1[10];
          ir1[10] = (v302_data + (v281_data * v300_data));
          float v309_data = r0[10];
          float v319_data = s0[28];
          float v321_data = ir1[9];
          ir1[9] = (v321_data + (v309_data * v319_data));
          float v324_data = s0[30];
          float v326_data = ir1[10];
          ir1[10] = (v326_data + (v309_data * v324_data));
          float v329_data = s0[32];
          float v331_data = ir1[11];
          ir1[11] = (v331_data + (v309_data * v329_data));
          float v337_data = r0[11];
          float v348_data = s0[31];
          float v350_data = ir1[10];
          ir1[10] = (v350_data + (v337_data * v348_data));
          float v353_data = s0[33];
          float v355_data = ir1[11];
          ir1[11] = (v355_data + (v337_data * v353_data));
          float v358_data = s0[35];
          float v360_data = ir1[12];
          ir1[12] = (v360_data + (v337_data * v358_data));
          float v365_data = r0[12];
          float v377_data = s0[34];
          float v379_data = ir1[11];
          ir1[11] = (v379_data + (v365_data * v377_data));
          float v382_data = s0[36];
          float v384_data = ir1[12];
          ir1[12] = (v384_data + (v365_data * v382_data));
          float v387_data = s0[38];
          float v389_data = ir1[13];
          ir1[13] = (v389_data + (v365_data * v387_data));
          float v393_data = r0[13];
          float v406_data = s0[37];
          float v408_data = ir1[12];
          ir1[12] = (v408_data + (v393_data * v406_data));
          float v411_data = s0[39];
          float v413_data = ir1[13];
          ir1[13] = (v413_data + (v393_data * v411_data));
          float v416_data = s0[41];
          float v418_data = ir1[14];
          ir1[14] = (v418_data + (v393_data * v416_data));
          float v421_data = r0[14];
          float v435_data = s0[40];
          float v437_data = ir1[13];
          ir1[13] = (v437_data + (v421_data * v435_data));
          float v440_data = s0[42];
          float v442_data = ir1[14];
          ir1[14] = (v442_data + (v421_data * v440_data));
          float v445_data = s0[44];
          float v447_data = ir1[15];
          ir1[15] = (v447_data + (v421_data * v445_data));
          float v449_data = r0[15];
          float v464_data = s0[43];
          float v466_data = ir1[14];
          ir1[14] = (v466_data + (v449_data * v464_data));
          float v469_data = s0[45];
          float v471_data = ir1[15];
          ir1[15] = (v471_data + (v449_data * v469_data));
          #pragma unroll
          for (int32_t v473_n0 = 0; v473_n0 < 1; ++v473_n0) {
            #pragma unroll
            for (int32_t v474_n1 = 0; v474_n1 < 16; ++v474_n1) {
              int32_t v475_a = v473_n0 + v474_n1;
              float v476_data = ir1[v475_a];
              r1[v475_a] = v476_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v477_i0 = 0; v477_i0 < 1; ++v477_i0) {
            int32_t v482_lead = v19_lead + (v477_i0 * 16);
            #pragma unroll
            for (int32_t v478_i1 = 0; v478_i1 < 16; ++v478_i1) {
              float v480_data = r1[(v477_i0 + v478_i1)];
              glb_m0[(v482_lead + (v478_i1 * 16))] = v480_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

