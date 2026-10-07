// === base name ===
kernel_a8d575448c972fba

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_a8d575448c972fba = {{16, 8, 1}, 16, 16, 1, 8, 2560, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_a8d575448c972fba(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_a8d575448c972fba(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_a8d575448c972fba(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_a8d575448c972fba, block.x * block.y * block.z, 640 * sizeof(float));
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
void launcher_kernel_a8d575448c972fba(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_a8d575448c972fba(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_a8d575448c972fba, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_a8d575448c972fba<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_a8d575448c972fba(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 8 per block = block 16x8x1, 2560 B shared, occupancy grid
    // operands:
    //   m0 16×16(16×16) {0..16}×{0..16} strided
    //   m1 16×16(16×16) {0..16}×{0..16} strided
    //   m2 16×16(16×16) {0..16}×{0..16} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":640}],"shared_bytes":2560,"shared_elements":640,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[80 * threadIdx.y + 0];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 256 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 46 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v22_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
            int32_t v26_lead = v22_lead + (v23_i0 * 16);
            #pragma unroll
            for (int32_t v24_i1 = 0; v24_i1 < 16; ++v24_i1) {
              float v29_data = __ldcg(&glb_m1[(v26_lead + (v24_i1 * 16))]);
              r0[(v23_i0 + v24_i1)] = v29_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 16], &glb_m2[0 + 0 + 1 * threadIdx.x + 16], 4);
          if (threadIdx.x < 14) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
          }
          __pipeline_commit();
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[16]{};
          // ir1 = +(r0 * s0)
          // [(0, 16), (0, 16)] [(0, 16)]
          float ir1[16]{};
          float v36_data = r0[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v37_data = s0[0];
          float v39_data = ir1[0];
          ir1[0] = (v39_data + (v36_data * v37_data));
          float v42_data = s0[2];
          float v44_data = ir1[1];
          ir1[1] = (v44_data + (v36_data * v42_data));
          float v60_data = r0[1];
          float v61_data = s0[1];
          float v63_data = ir1[0];
          ir1[0] = (v63_data + (v60_data * v61_data));
          float v66_data = s0[3];
          float v68_data = ir1[1];
          ir1[1] = (v68_data + (v60_data * v66_data));
          float v71_data = s0[5];
          float v73_data = ir1[2];
          ir1[2] = (v73_data + (v60_data * v71_data));
          float v88_data = r0[2];
          float v90_data = s0[4];
          float v92_data = ir1[1];
          ir1[1] = (v92_data + (v88_data * v90_data));
          float v95_data = s0[6];
          float v97_data = ir1[2];
          ir1[2] = (v97_data + (v88_data * v95_data));
          float v100_data = s0[8];
          float v102_data = ir1[3];
          ir1[3] = (v102_data + (v88_data * v100_data));
          float v116_data = r0[3];
          float v119_data = s0[7];
          float v121_data = ir1[2];
          ir1[2] = (v121_data + (v116_data * v119_data));
          float v124_data = s0[9];
          float v126_data = ir1[3];
          ir1[3] = (v126_data + (v116_data * v124_data));
          float v129_data = s0[11];
          float v131_data = ir1[4];
          ir1[4] = (v131_data + (v116_data * v129_data));
          float v144_data = r0[4];
          float v148_data = s0[10];
          float v150_data = ir1[3];
          ir1[3] = (v150_data + (v144_data * v148_data));
          float v153_data = s0[12];
          float v155_data = ir1[4];
          ir1[4] = (v155_data + (v144_data * v153_data));
          float v158_data = s0[14];
          float v160_data = ir1[5];
          ir1[5] = (v160_data + (v144_data * v158_data));
          float v172_data = r0[5];
          float v177_data = s0[13];
          float v179_data = ir1[4];
          ir1[4] = (v179_data + (v172_data * v177_data));
          float v182_data = s0[15];
          float v184_data = ir1[5];
          ir1[5] = (v184_data + (v172_data * v182_data));
          float v187_data = s0[17];
          float v189_data = ir1[6];
          ir1[6] = (v189_data + (v172_data * v187_data));
          float v200_data = r0[6];
          float v206_data = s0[16];
          float v208_data = ir1[5];
          ir1[5] = (v208_data + (v200_data * v206_data));
          float v211_data = s0[18];
          float v213_data = ir1[6];
          ir1[6] = (v213_data + (v200_data * v211_data));
          float v216_data = s0[20];
          float v218_data = ir1[7];
          ir1[7] = (v218_data + (v200_data * v216_data));
          float v228_data = r0[7];
          float v235_data = s0[19];
          float v237_data = ir1[6];
          ir1[6] = (v237_data + (v228_data * v235_data));
          float v240_data = s0[21];
          float v242_data = ir1[7];
          ir1[7] = (v242_data + (v228_data * v240_data));
          float v245_data = s0[23];
          float v247_data = ir1[8];
          ir1[8] = (v247_data + (v228_data * v245_data));
          float v256_data = r0[8];
          float v264_data = s0[22];
          float v266_data = ir1[7];
          ir1[7] = (v266_data + (v256_data * v264_data));
          float v269_data = s0[24];
          float v271_data = ir1[8];
          ir1[8] = (v271_data + (v256_data * v269_data));
          float v274_data = s0[26];
          float v276_data = ir1[9];
          ir1[9] = (v276_data + (v256_data * v274_data));
          float v284_data = r0[9];
          float v293_data = s0[25];
          float v295_data = ir1[8];
          ir1[8] = (v295_data + (v284_data * v293_data));
          float v298_data = s0[27];
          float v300_data = ir1[9];
          ir1[9] = (v300_data + (v284_data * v298_data));
          float v303_data = s0[29];
          float v305_data = ir1[10];
          ir1[10] = (v305_data + (v284_data * v303_data));
          float v312_data = r0[10];
          float v322_data = s0[28];
          float v324_data = ir1[9];
          ir1[9] = (v324_data + (v312_data * v322_data));
          float v327_data = s0[30];
          float v329_data = ir1[10];
          ir1[10] = (v329_data + (v312_data * v327_data));
          float v332_data = s0[32];
          float v334_data = ir1[11];
          ir1[11] = (v334_data + (v312_data * v332_data));
          float v340_data = r0[11];
          float v351_data = s0[31];
          float v353_data = ir1[10];
          ir1[10] = (v353_data + (v340_data * v351_data));
          float v356_data = s0[33];
          float v358_data = ir1[11];
          ir1[11] = (v358_data + (v340_data * v356_data));
          float v361_data = s0[35];
          float v363_data = ir1[12];
          ir1[12] = (v363_data + (v340_data * v361_data));
          float v368_data = r0[12];
          float v380_data = s0[34];
          float v382_data = ir1[11];
          ir1[11] = (v382_data + (v368_data * v380_data));
          float v385_data = s0[36];
          float v387_data = ir1[12];
          ir1[12] = (v387_data + (v368_data * v385_data));
          float v390_data = s0[38];
          float v392_data = ir1[13];
          ir1[13] = (v392_data + (v368_data * v390_data));
          float v396_data = r0[13];
          float v409_data = s0[37];
          float v411_data = ir1[12];
          ir1[12] = (v411_data + (v396_data * v409_data));
          float v414_data = s0[39];
          float v416_data = ir1[13];
          ir1[13] = (v416_data + (v396_data * v414_data));
          float v419_data = s0[41];
          float v421_data = ir1[14];
          ir1[14] = (v421_data + (v396_data * v419_data));
          float v424_data = r0[14];
          float v438_data = s0[40];
          float v440_data = ir1[13];
          ir1[13] = (v440_data + (v424_data * v438_data));
          float v443_data = s0[42];
          float v445_data = ir1[14];
          ir1[14] = (v445_data + (v424_data * v443_data));
          float v448_data = s0[44];
          float v450_data = ir1[15];
          ir1[15] = (v450_data + (v424_data * v448_data));
          float v452_data = r0[15];
          float v467_data = s0[43];
          float v469_data = ir1[14];
          ir1[14] = (v469_data + (v452_data * v467_data));
          float v472_data = s0[45];
          float v474_data = ir1[15];
          ir1[15] = (v474_data + (v452_data * v472_data));
          // r1 = ir1
          #pragma unroll
          for (int32_t v476_n0 = 0; v476_n0 < 1; ++v476_n0) {
            #pragma unroll
            for (int32_t v477_n1 = 0; v477_n1 < 16; ++v477_n1) {
              int32_t v478_a = v476_n0 + v477_n1;
              float v479_data = ir1[v478_a];
              r1[v478_a] = v479_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v480_i0 = 0; v480_i0 < 1; ++v480_i0) {
            int32_t v485_lead = v22_lead + (v480_i0 * 16);
            #pragma unroll
            for (int32_t v481_i1 = 0; v481_i1 < 16; ++v481_i1) {
              float v483_data = r1[(v480_i0 + v481_i1)];
              glb_m0[(v485_lead + (v481_i1 * 16))] = v483_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

