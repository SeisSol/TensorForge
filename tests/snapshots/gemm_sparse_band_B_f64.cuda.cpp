// === base name ===
kernel_14d884aa021b2c7a

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_14d884aa021b2c7a = {{16, 8, 1}, 16, 16, 1, 8, 4096, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_14d884aa021b2c7a(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_14d884aa021b2c7a(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_14d884aa021b2c7a(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_14d884aa021b2c7a, block.x * block.y * block.z, 512 * sizeof(double));
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
  config.sharedMemBytes = 512 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_14d884aa021b2c7a(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_14d884aa021b2c7a(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_14d884aa021b2c7a, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_14d884aa021b2c7a<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_14d884aa021b2c7a(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 8 per block = block 16x8x1, 4096 B shared, occupancy grid
    // operands:
    //   m0 16×16(16×16) {0..16}×{0..16} strided
    //   m1 16×16(16×16) {0..16}×{0..16} strided
    //   m2 16×16(16×16) {0..16}×{0..16} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"double","launch":{"active_threads":16,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":512}],"shared_bytes":4096,"shared_elements":512,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[64 * threadIdx.y + 0];
      double * __restrict__ s0 = &localShrMem0[0];
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[v8_batchId0 * 256 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m1 = &m1[v8_batchId0 * 256 + 0 + m1_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[v8_batchId0 * 46 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v22_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
            int32_t v26_lead = v22_lead + (v23_i0 * 16);
            #pragma unroll
            for (int32_t v24_i1 = 0; v24_i1 < 16; ++v24_i1) {
              double v29_data = __ldcg(&glb_m1[(v26_lead + (v24_i1 * 16))]);
              r0[(v23_i0 + v24_i1)] = v29_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 8);
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 16], &glb_m2[0 + 0 + 1 * threadIdx.x + 16], 8);
          if (threadIdx.x < 14) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m2[0 + 0 + 1 * threadIdx.x + 32], 8);
          }
          __pipeline_commit();
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          double r1[16]{};
          // ir1 = +(r0 * s0)
          // [(0, 16), (0, 16)] [(0, 16)]
          double ir1[16]{};
          double v36_data = r0[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          double v37_data = s0[0];
          double v39_data = ir1[0];
          ir1[0] = (v39_data + (v36_data * v37_data));
          double v42_data = s0[2];
          double v44_data = ir1[1];
          ir1[1] = (v44_data + (v36_data * v42_data));
          double v60_data = r0[1];
          double v61_data = s0[1];
          double v63_data = ir1[0];
          ir1[0] = (v63_data + (v60_data * v61_data));
          double v66_data = s0[3];
          double v68_data = ir1[1];
          ir1[1] = (v68_data + (v60_data * v66_data));
          double v71_data = s0[5];
          double v73_data = ir1[2];
          ir1[2] = (v73_data + (v60_data * v71_data));
          double v88_data = r0[2];
          double v90_data = s0[4];
          double v92_data = ir1[1];
          ir1[1] = (v92_data + (v88_data * v90_data));
          double v95_data = s0[6];
          double v97_data = ir1[2];
          ir1[2] = (v97_data + (v88_data * v95_data));
          double v100_data = s0[8];
          double v102_data = ir1[3];
          ir1[3] = (v102_data + (v88_data * v100_data));
          double v116_data = r0[3];
          double v119_data = s0[7];
          double v121_data = ir1[2];
          ir1[2] = (v121_data + (v116_data * v119_data));
          double v124_data = s0[9];
          double v126_data = ir1[3];
          ir1[3] = (v126_data + (v116_data * v124_data));
          double v129_data = s0[11];
          double v131_data = ir1[4];
          ir1[4] = (v131_data + (v116_data * v129_data));
          double v144_data = r0[4];
          double v148_data = s0[10];
          double v150_data = ir1[3];
          ir1[3] = (v150_data + (v144_data * v148_data));
          double v153_data = s0[12];
          double v155_data = ir1[4];
          ir1[4] = (v155_data + (v144_data * v153_data));
          double v158_data = s0[14];
          double v160_data = ir1[5];
          ir1[5] = (v160_data + (v144_data * v158_data));
          double v172_data = r0[5];
          double v177_data = s0[13];
          double v179_data = ir1[4];
          ir1[4] = (v179_data + (v172_data * v177_data));
          double v182_data = s0[15];
          double v184_data = ir1[5];
          ir1[5] = (v184_data + (v172_data * v182_data));
          double v187_data = s0[17];
          double v189_data = ir1[6];
          ir1[6] = (v189_data + (v172_data * v187_data));
          double v200_data = r0[6];
          double v206_data = s0[16];
          double v208_data = ir1[5];
          ir1[5] = (v208_data + (v200_data * v206_data));
          double v211_data = s0[18];
          double v213_data = ir1[6];
          ir1[6] = (v213_data + (v200_data * v211_data));
          double v216_data = s0[20];
          double v218_data = ir1[7];
          ir1[7] = (v218_data + (v200_data * v216_data));
          double v228_data = r0[7];
          double v235_data = s0[19];
          double v237_data = ir1[6];
          ir1[6] = (v237_data + (v228_data * v235_data));
          double v240_data = s0[21];
          double v242_data = ir1[7];
          ir1[7] = (v242_data + (v228_data * v240_data));
          double v245_data = s0[23];
          double v247_data = ir1[8];
          ir1[8] = (v247_data + (v228_data * v245_data));
          double v256_data = r0[8];
          double v264_data = s0[22];
          double v266_data = ir1[7];
          ir1[7] = (v266_data + (v256_data * v264_data));
          double v269_data = s0[24];
          double v271_data = ir1[8];
          ir1[8] = (v271_data + (v256_data * v269_data));
          double v274_data = s0[26];
          double v276_data = ir1[9];
          ir1[9] = (v276_data + (v256_data * v274_data));
          double v284_data = r0[9];
          double v293_data = s0[25];
          double v295_data = ir1[8];
          ir1[8] = (v295_data + (v284_data * v293_data));
          double v298_data = s0[27];
          double v300_data = ir1[9];
          ir1[9] = (v300_data + (v284_data * v298_data));
          double v303_data = s0[29];
          double v305_data = ir1[10];
          ir1[10] = (v305_data + (v284_data * v303_data));
          double v312_data = r0[10];
          double v322_data = s0[28];
          double v324_data = ir1[9];
          ir1[9] = (v324_data + (v312_data * v322_data));
          double v327_data = s0[30];
          double v329_data = ir1[10];
          ir1[10] = (v329_data + (v312_data * v327_data));
          double v332_data = s0[32];
          double v334_data = ir1[11];
          ir1[11] = (v334_data + (v312_data * v332_data));
          double v340_data = r0[11];
          double v351_data = s0[31];
          double v353_data = ir1[10];
          ir1[10] = (v353_data + (v340_data * v351_data));
          double v356_data = s0[33];
          double v358_data = ir1[11];
          ir1[11] = (v358_data + (v340_data * v356_data));
          double v361_data = s0[35];
          double v363_data = ir1[12];
          ir1[12] = (v363_data + (v340_data * v361_data));
          double v368_data = r0[12];
          double v380_data = s0[34];
          double v382_data = ir1[11];
          ir1[11] = (v382_data + (v368_data * v380_data));
          double v385_data = s0[36];
          double v387_data = ir1[12];
          ir1[12] = (v387_data + (v368_data * v385_data));
          double v390_data = s0[38];
          double v392_data = ir1[13];
          ir1[13] = (v392_data + (v368_data * v390_data));
          double v396_data = r0[13];
          double v409_data = s0[37];
          double v411_data = ir1[12];
          ir1[12] = (v411_data + (v396_data * v409_data));
          double v414_data = s0[39];
          double v416_data = ir1[13];
          ir1[13] = (v416_data + (v396_data * v414_data));
          double v419_data = s0[41];
          double v421_data = ir1[14];
          ir1[14] = (v421_data + (v396_data * v419_data));
          double v424_data = r0[14];
          double v438_data = s0[40];
          double v440_data = ir1[13];
          ir1[13] = (v440_data + (v424_data * v438_data));
          double v443_data = s0[42];
          double v445_data = ir1[14];
          ir1[14] = (v445_data + (v424_data * v443_data));
          double v448_data = s0[44];
          double v450_data = ir1[15];
          ir1[15] = (v450_data + (v424_data * v448_data));
          double v452_data = r0[15];
          double v467_data = s0[43];
          double v469_data = ir1[14];
          ir1[14] = (v469_data + (v452_data * v467_data));
          double v472_data = s0[45];
          double v474_data = ir1[15];
          ir1[15] = (v474_data + (v452_data * v472_data));
          // r1 = ir1
          #pragma unroll
          for (int32_t v476_n0 = 0; v476_n0 < 1; ++v476_n0) {
            #pragma unroll
            for (int32_t v477_n1 = 0; v477_n1 < 16; ++v477_n1) {
              int32_t v478_a = v476_n0 + v477_n1;
              double v479_data = ir1[v478_a];
              r1[v478_a] = v479_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v480_i0 = 0; v480_i0 < 1; ++v480_i0) {
            int32_t v485_lead = v22_lead + (v480_i0 * 16);
            #pragma unroll
            for (int32_t v481_i1 = 0; v481_i1 < 16; ++v481_i1) {
              double v483_data = r1[(v480_i0 + v481_i1)];
              glb_m0[(v485_lead + (v481_i1 * 16))] = v483_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

