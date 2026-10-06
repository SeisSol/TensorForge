// === base name ===
kernel_de802b83e5d37a44

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_de802b83e5d37a44 = {{16, 8, 1}, 16, 16, 1, 8, 4096, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_de802b83e5d37a44(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_de802b83e5d37a44(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_de802b83e5d37a44(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_de802b83e5d37a44, block.x * block.y * block.z, 512 * sizeof(double));
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
void launcher_kernel_de802b83e5d37a44(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_de802b83e5d37a44(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_de802b83e5d37a44, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_de802b83e5d37a44<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_de802b83e5d37a44(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
      double* tempShrMem = &localShrMem0[48];
      double * __restrict__ s0 = &localShrMem0[0];
      for (size_t v11_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v11_batchId0 < numElements0; v11_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v12_ahead1 = v11_batchId0 + (gridDim.x * blockDim.y);
        size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[v11_batchId0 * 256 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m1 = &m1[v11_batchId0 * 256 + 0 + m1_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[v11_batchId0 * 46 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v25_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v26_i0 = 0; v26_i0 < 1; ++v26_i0) {
            int32_t v29_lead = v25_lead + (v26_i0 * 16);
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 16; ++v27_i1) {
              double v32_data = __ldcg(&glb_m1[(v29_lead + (v27_i1 * 16))]);
              r0[(v26_i0 + v27_i1)] = v32_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 8);
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 16], &glb_m2[0 + 0 + 1 * threadIdx.x + 16], 8);
          if (threadIdx.x < 14) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m2[0 + 0 + 1 * threadIdx.x + 32], 8);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          double r1[16]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // ir1 = +(r0 * s0)
          // [(0, 16), (0, 16)] [(0, 16)]
          double ir1[16]{};
          double v39_data = r0[0];
          double v40_data = s0[0];
          double v42_data = ir1[0];
          ir1[0] = (v42_data + (v39_data * v40_data));
          double v45_data = s0[2];
          double v47_data = ir1[1];
          ir1[1] = (v47_data + (v39_data * v45_data));
          double v63_data = r0[1];
          double v64_data = s0[1];
          double v66_data = ir1[0];
          ir1[0] = (v66_data + (v63_data * v64_data));
          double v69_data = s0[3];
          double v71_data = ir1[1];
          ir1[1] = (v71_data + (v63_data * v69_data));
          double v74_data = s0[5];
          double v76_data = ir1[2];
          ir1[2] = (v76_data + (v63_data * v74_data));
          double v91_data = r0[2];
          double v93_data = s0[4];
          double v95_data = ir1[1];
          ir1[1] = (v95_data + (v91_data * v93_data));
          double v98_data = s0[6];
          double v100_data = ir1[2];
          ir1[2] = (v100_data + (v91_data * v98_data));
          double v103_data = s0[8];
          double v105_data = ir1[3];
          ir1[3] = (v105_data + (v91_data * v103_data));
          double v119_data = r0[3];
          double v122_data = s0[7];
          double v124_data = ir1[2];
          ir1[2] = (v124_data + (v119_data * v122_data));
          double v127_data = s0[9];
          double v129_data = ir1[3];
          ir1[3] = (v129_data + (v119_data * v127_data));
          double v132_data = s0[11];
          double v134_data = ir1[4];
          ir1[4] = (v134_data + (v119_data * v132_data));
          double v147_data = r0[4];
          double v151_data = s0[10];
          double v153_data = ir1[3];
          ir1[3] = (v153_data + (v147_data * v151_data));
          double v156_data = s0[12];
          double v158_data = ir1[4];
          ir1[4] = (v158_data + (v147_data * v156_data));
          double v161_data = s0[14];
          double v163_data = ir1[5];
          ir1[5] = (v163_data + (v147_data * v161_data));
          double v175_data = r0[5];
          double v180_data = s0[13];
          double v182_data = ir1[4];
          ir1[4] = (v182_data + (v175_data * v180_data));
          double v185_data = s0[15];
          double v187_data = ir1[5];
          ir1[5] = (v187_data + (v175_data * v185_data));
          double v190_data = s0[17];
          double v192_data = ir1[6];
          ir1[6] = (v192_data + (v175_data * v190_data));
          double v203_data = r0[6];
          double v209_data = s0[16];
          double v211_data = ir1[5];
          ir1[5] = (v211_data + (v203_data * v209_data));
          double v214_data = s0[18];
          double v216_data = ir1[6];
          ir1[6] = (v216_data + (v203_data * v214_data));
          double v219_data = s0[20];
          double v221_data = ir1[7];
          ir1[7] = (v221_data + (v203_data * v219_data));
          double v231_data = r0[7];
          double v238_data = s0[19];
          double v240_data = ir1[6];
          ir1[6] = (v240_data + (v231_data * v238_data));
          double v243_data = s0[21];
          double v245_data = ir1[7];
          ir1[7] = (v245_data + (v231_data * v243_data));
          double v248_data = s0[23];
          double v250_data = ir1[8];
          ir1[8] = (v250_data + (v231_data * v248_data));
          double v259_data = r0[8];
          double v267_data = s0[22];
          double v269_data = ir1[7];
          ir1[7] = (v269_data + (v259_data * v267_data));
          double v272_data = s0[24];
          double v274_data = ir1[8];
          ir1[8] = (v274_data + (v259_data * v272_data));
          double v277_data = s0[26];
          double v279_data = ir1[9];
          ir1[9] = (v279_data + (v259_data * v277_data));
          double v287_data = r0[9];
          double v296_data = s0[25];
          double v298_data = ir1[8];
          ir1[8] = (v298_data + (v287_data * v296_data));
          double v301_data = s0[27];
          double v303_data = ir1[9];
          ir1[9] = (v303_data + (v287_data * v301_data));
          double v306_data = s0[29];
          double v308_data = ir1[10];
          ir1[10] = (v308_data + (v287_data * v306_data));
          double v315_data = r0[10];
          double v325_data = s0[28];
          double v327_data = ir1[9];
          ir1[9] = (v327_data + (v315_data * v325_data));
          double v330_data = s0[30];
          double v332_data = ir1[10];
          ir1[10] = (v332_data + (v315_data * v330_data));
          double v335_data = s0[32];
          double v337_data = ir1[11];
          ir1[11] = (v337_data + (v315_data * v335_data));
          double v343_data = r0[11];
          double v354_data = s0[31];
          double v356_data = ir1[10];
          ir1[10] = (v356_data + (v343_data * v354_data));
          double v359_data = s0[33];
          double v361_data = ir1[11];
          ir1[11] = (v361_data + (v343_data * v359_data));
          double v364_data = s0[35];
          double v366_data = ir1[12];
          ir1[12] = (v366_data + (v343_data * v364_data));
          double v371_data = r0[12];
          double v383_data = s0[34];
          double v385_data = ir1[11];
          ir1[11] = (v385_data + (v371_data * v383_data));
          double v388_data = s0[36];
          double v390_data = ir1[12];
          ir1[12] = (v390_data + (v371_data * v388_data));
          double v393_data = s0[38];
          double v395_data = ir1[13];
          ir1[13] = (v395_data + (v371_data * v393_data));
          double v399_data = r0[13];
          double v412_data = s0[37];
          double v414_data = ir1[12];
          ir1[12] = (v414_data + (v399_data * v412_data));
          double v417_data = s0[39];
          double v419_data = ir1[13];
          ir1[13] = (v419_data + (v399_data * v417_data));
          double v422_data = s0[41];
          double v424_data = ir1[14];
          ir1[14] = (v424_data + (v399_data * v422_data));
          double v427_data = r0[14];
          double v441_data = s0[40];
          double v443_data = ir1[13];
          ir1[13] = (v443_data + (v427_data * v441_data));
          double v446_data = s0[42];
          double v448_data = ir1[14];
          ir1[14] = (v448_data + (v427_data * v446_data));
          double v451_data = s0[44];
          double v453_data = ir1[15];
          ir1[15] = (v453_data + (v427_data * v451_data));
          double v455_data = r0[15];
          double v470_data = s0[43];
          double v472_data = ir1[14];
          ir1[14] = (v472_data + (v455_data * v470_data));
          double v475_data = s0[45];
          double v477_data = ir1[15];
          ir1[15] = (v477_data + (v455_data * v475_data));
          // r1 = ir1
          #pragma unroll
          for (int32_t v479_n0 = 0; v479_n0 < 1; ++v479_n0) {
            #pragma unroll
            for (int32_t v480_n1 = 0; v480_n1 < 16; ++v480_n1) {
              int32_t v481_a = v479_n0 + v480_n1;
              double v482_data = ir1[v481_a];
              r1[v481_a] = v482_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v483_i0 = 0; v483_i0 < 1; ++v483_i0) {
            int32_t v488_lead = v25_lead + (v483_i0 * 16);
            #pragma unroll
            for (int32_t v484_i1 = 0; v484_i1 < 16; ++v484_i1) {
              double v486_data = r1[(v483_i0 + v484_i1)];
              glb_m0[(v488_lead + (v484_i1 * 16))] = v486_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

