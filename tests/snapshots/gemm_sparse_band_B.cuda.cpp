// === base name ===
kernel_688956e9c76baa8f

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_688956e9c76baa8f = {{16, 8, 1}, 16, 16, 1, 8, 2560, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_688956e9c76baa8f(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_688956e9c76baa8f(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_688956e9c76baa8f(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_688956e9c76baa8f, block.x * block.y * block.z, 640 * sizeof(float));
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
void launcher_kernel_688956e9c76baa8f(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_688956e9c76baa8f(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_688956e9c76baa8f, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_688956e9c76baa8f<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_688956e9c76baa8f(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
      float* tempShrMem = &localShrMem0[64];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v11_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v11_batchId0 < numElements0; v11_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v12_ahead1 = v11_batchId0 + (gridDim.x * blockDim.y);
        size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v11_batchId0 * 256 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v11_batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v11_batchId0 * 46 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v25_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v26_i0 = 0; v26_i0 < 1; ++v26_i0) {
            int32_t v29_lead = v25_lead + (v26_i0 * 16);
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 16; ++v27_i1) {
              float v32_data = __ldcg(&glb_m1[(v29_lead + (v27_i1 * 16))]);
              r0[(v26_i0 + v27_i1)] = v32_data;
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
          // ir1 = +(r0 * s0)
          // [(0, 16), (0, 16)] [(0, 16)]
          float ir1[16]{};
          float v39_data = r0[0];
          float v40_data = s0[0];
          float v42_data = ir1[0];
          ir1[0] = (v42_data + (v39_data * v40_data));
          float v45_data = s0[2];
          float v47_data = ir1[1];
          ir1[1] = (v47_data + (v39_data * v45_data));
          float v63_data = r0[1];
          float v64_data = s0[1];
          float v66_data = ir1[0];
          ir1[0] = (v66_data + (v63_data * v64_data));
          float v69_data = s0[3];
          float v71_data = ir1[1];
          ir1[1] = (v71_data + (v63_data * v69_data));
          float v74_data = s0[5];
          float v76_data = ir1[2];
          ir1[2] = (v76_data + (v63_data * v74_data));
          float v91_data = r0[2];
          float v93_data = s0[4];
          float v95_data = ir1[1];
          ir1[1] = (v95_data + (v91_data * v93_data));
          float v98_data = s0[6];
          float v100_data = ir1[2];
          ir1[2] = (v100_data + (v91_data * v98_data));
          float v103_data = s0[8];
          float v105_data = ir1[3];
          ir1[3] = (v105_data + (v91_data * v103_data));
          float v119_data = r0[3];
          float v122_data = s0[7];
          float v124_data = ir1[2];
          ir1[2] = (v124_data + (v119_data * v122_data));
          float v127_data = s0[9];
          float v129_data = ir1[3];
          ir1[3] = (v129_data + (v119_data * v127_data));
          float v132_data = s0[11];
          float v134_data = ir1[4];
          ir1[4] = (v134_data + (v119_data * v132_data));
          float v147_data = r0[4];
          float v151_data = s0[10];
          float v153_data = ir1[3];
          ir1[3] = (v153_data + (v147_data * v151_data));
          float v156_data = s0[12];
          float v158_data = ir1[4];
          ir1[4] = (v158_data + (v147_data * v156_data));
          float v161_data = s0[14];
          float v163_data = ir1[5];
          ir1[5] = (v163_data + (v147_data * v161_data));
          float v175_data = r0[5];
          float v180_data = s0[13];
          float v182_data = ir1[4];
          ir1[4] = (v182_data + (v175_data * v180_data));
          float v185_data = s0[15];
          float v187_data = ir1[5];
          ir1[5] = (v187_data + (v175_data * v185_data));
          float v190_data = s0[17];
          float v192_data = ir1[6];
          ir1[6] = (v192_data + (v175_data * v190_data));
          float v203_data = r0[6];
          float v209_data = s0[16];
          float v211_data = ir1[5];
          ir1[5] = (v211_data + (v203_data * v209_data));
          float v214_data = s0[18];
          float v216_data = ir1[6];
          ir1[6] = (v216_data + (v203_data * v214_data));
          float v219_data = s0[20];
          float v221_data = ir1[7];
          ir1[7] = (v221_data + (v203_data * v219_data));
          float v231_data = r0[7];
          float v238_data = s0[19];
          float v240_data = ir1[6];
          ir1[6] = (v240_data + (v231_data * v238_data));
          float v243_data = s0[21];
          float v245_data = ir1[7];
          ir1[7] = (v245_data + (v231_data * v243_data));
          float v248_data = s0[23];
          float v250_data = ir1[8];
          ir1[8] = (v250_data + (v231_data * v248_data));
          float v259_data = r0[8];
          float v267_data = s0[22];
          float v269_data = ir1[7];
          ir1[7] = (v269_data + (v259_data * v267_data));
          float v272_data = s0[24];
          float v274_data = ir1[8];
          ir1[8] = (v274_data + (v259_data * v272_data));
          float v277_data = s0[26];
          float v279_data = ir1[9];
          ir1[9] = (v279_data + (v259_data * v277_data));
          float v287_data = r0[9];
          float v296_data = s0[25];
          float v298_data = ir1[8];
          ir1[8] = (v298_data + (v287_data * v296_data));
          float v301_data = s0[27];
          float v303_data = ir1[9];
          ir1[9] = (v303_data + (v287_data * v301_data));
          float v306_data = s0[29];
          float v308_data = ir1[10];
          ir1[10] = (v308_data + (v287_data * v306_data));
          float v315_data = r0[10];
          float v325_data = s0[28];
          float v327_data = ir1[9];
          ir1[9] = (v327_data + (v315_data * v325_data));
          float v330_data = s0[30];
          float v332_data = ir1[10];
          ir1[10] = (v332_data + (v315_data * v330_data));
          float v335_data = s0[32];
          float v337_data = ir1[11];
          ir1[11] = (v337_data + (v315_data * v335_data));
          float v343_data = r0[11];
          float v354_data = s0[31];
          float v356_data = ir1[10];
          ir1[10] = (v356_data + (v343_data * v354_data));
          float v359_data = s0[33];
          float v361_data = ir1[11];
          ir1[11] = (v361_data + (v343_data * v359_data));
          float v364_data = s0[35];
          float v366_data = ir1[12];
          ir1[12] = (v366_data + (v343_data * v364_data));
          float v371_data = r0[12];
          float v383_data = s0[34];
          float v385_data = ir1[11];
          ir1[11] = (v385_data + (v371_data * v383_data));
          float v388_data = s0[36];
          float v390_data = ir1[12];
          ir1[12] = (v390_data + (v371_data * v388_data));
          float v393_data = s0[38];
          float v395_data = ir1[13];
          ir1[13] = (v395_data + (v371_data * v393_data));
          float v399_data = r0[13];
          float v412_data = s0[37];
          float v414_data = ir1[12];
          ir1[12] = (v414_data + (v399_data * v412_data));
          float v417_data = s0[39];
          float v419_data = ir1[13];
          ir1[13] = (v419_data + (v399_data * v417_data));
          float v422_data = s0[41];
          float v424_data = ir1[14];
          ir1[14] = (v424_data + (v399_data * v422_data));
          float v427_data = r0[14];
          float v441_data = s0[40];
          float v443_data = ir1[13];
          ir1[13] = (v443_data + (v427_data * v441_data));
          float v446_data = s0[42];
          float v448_data = ir1[14];
          ir1[14] = (v448_data + (v427_data * v446_data));
          float v451_data = s0[44];
          float v453_data = ir1[15];
          ir1[15] = (v453_data + (v427_data * v451_data));
          float v455_data = r0[15];
          float v470_data = s0[43];
          float v472_data = ir1[14];
          ir1[14] = (v472_data + (v455_data * v470_data));
          float v475_data = s0[45];
          float v477_data = ir1[15];
          ir1[15] = (v477_data + (v455_data * v475_data));
          // r1 = ir1
          #pragma unroll
          for (int32_t v479_n0 = 0; v479_n0 < 1; ++v479_n0) {
            #pragma unroll
            for (int32_t v480_n1 = 0; v480_n1 < 16; ++v480_n1) {
              int32_t v481_a = v479_n0 + v480_n1;
              float v482_data = ir1[v481_a];
              r1[v481_a] = v482_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v483_i0 = 0; v483_i0 < 1; ++v483_i0) {
            int32_t v488_lead = v25_lead + (v483_i0 * 16);
            #pragma unroll
            for (int32_t v484_i1 = 0; v484_i1 < 16; ++v484_i1) {
              float v486_data = r1[(v483_i0 + v484_i1)];
              glb_m0[(v488_lead + (v484_i1 * 16))] = v486_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

