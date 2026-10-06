// === base name ===
kernel_ca8098ef5bf8cc1a

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_ca8098ef5bf8cc1a = {{32, 4, 1}, 32, 32, 1, 4, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_ca8098ef5bf8cc1a(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_ca8098ef5bf8cc1a(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_ca8098ef5bf8cc1a(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_ca8098ef5bf8cc1a, block.x * block.y * block.z, 256 * sizeof(float));
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
void launcher_kernel_ca8098ef5bf8cc1a(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_ca8098ef5bf8cc1a(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_ca8098ef5bf8cc1a, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_ca8098ef5bf8cc1a<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_ca8098ef5bf8cc1a(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 4 per block = block 32x4x1, 1024 B shared, occupancy grid
    // operands:
    //   m0 8×8(8×8) {0..8}×{0..8} strided
    //   m1 8×8(8×8) {0..8}×{0..8} strided
    //   m2 8(8) {0..8} strided
    // operations:
    //   t0[i,j] = m0[i,k] × m1[k,j]
    //   OUT = +(TMP, dims=[1])
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"OUT","bbox":[[0],[8]],"name":"m2","ordered":false,"parts":1,"shape":[8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0],[8]],"is_tmp":false,"name":"m2","offset":[0],"shape":[8]},"kind":"reduction","op":"+","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"target":[[0,-1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[64 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v11_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v11_batchId0 < numElements0; v11_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v12_ahead1 = v11_batchId0 + (gridDim.x * blockDim.y);
        size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v11_batchId0 * 64 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v11_batchId0 * 64 + 0 + m1_extraOffset];
          float *const __restrict__ glb_m2 = &m2[v11_batchId0 * 8 + 0 + m2_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v25_lead = threadIdx.x % 32;
          bool v26_g = v25_lead < 8;
          if (v26_g) {
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 8; ++v27_i1) {
              float v32_data = __ldcg(&glb_m0[(v25_lead + (v27_i1 * 8))]);
              r0[v27_i1] = v32_data;
            }
          }
          // s0 = load{g>s}(glb_m1[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m1[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m1[0 + 0 + 1 * threadIdx.x + 32], 4);
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m0););
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 8), (0, 8)] [(0, 8)]
          float v37_data = r0[0];
          float v38_data = s0[0];
          float v40_data = r1[0];
          r1[0] = (v40_data + (v37_data * v38_data));
          float v43_data = s0[8];
          float v45_data = r1[1];
          r1[1] = (v45_data + (v37_data * v43_data));
          float v48_data = s0[16];
          float v50_data = r1[2];
          r1[2] = (v50_data + (v37_data * v48_data));
          float v53_data = s0[24];
          float v55_data = r1[3];
          r1[3] = (v55_data + (v37_data * v53_data));
          float v58_data = s0[32];
          float v60_data = r1[4];
          r1[4] = (v60_data + (v37_data * v58_data));
          float v63_data = s0[40];
          float v65_data = r1[5];
          r1[5] = (v65_data + (v37_data * v63_data));
          float v68_data = s0[48];
          float v70_data = r1[6];
          r1[6] = (v70_data + (v37_data * v68_data));
          float v73_data = s0[56];
          float v75_data = r1[7];
          r1[7] = (v75_data + (v37_data * v73_data));
          float v77_data = r0[1];
          float v78_data = s0[1];
          float v80_data = r1[0];
          r1[0] = (v80_data + (v77_data * v78_data));
          float v83_data = s0[9];
          float v85_data = r1[1];
          r1[1] = (v85_data + (v77_data * v83_data));
          float v88_data = s0[17];
          float v90_data = r1[2];
          r1[2] = (v90_data + (v77_data * v88_data));
          float v93_data = s0[25];
          float v95_data = r1[3];
          r1[3] = (v95_data + (v77_data * v93_data));
          float v98_data = s0[33];
          float v100_data = r1[4];
          r1[4] = (v100_data + (v77_data * v98_data));
          float v103_data = s0[41];
          float v105_data = r1[5];
          r1[5] = (v105_data + (v77_data * v103_data));
          float v108_data = s0[49];
          float v110_data = r1[6];
          r1[6] = (v110_data + (v77_data * v108_data));
          float v113_data = s0[57];
          float v115_data = r1[7];
          r1[7] = (v115_data + (v77_data * v113_data));
          float v117_data = r0[2];
          float v118_data = s0[2];
          float v120_data = r1[0];
          r1[0] = (v120_data + (v117_data * v118_data));
          float v123_data = s0[10];
          float v125_data = r1[1];
          r1[1] = (v125_data + (v117_data * v123_data));
          float v128_data = s0[18];
          float v130_data = r1[2];
          r1[2] = (v130_data + (v117_data * v128_data));
          float v133_data = s0[26];
          float v135_data = r1[3];
          r1[3] = (v135_data + (v117_data * v133_data));
          float v138_data = s0[34];
          float v140_data = r1[4];
          r1[4] = (v140_data + (v117_data * v138_data));
          float v143_data = s0[42];
          float v145_data = r1[5];
          r1[5] = (v145_data + (v117_data * v143_data));
          float v148_data = s0[50];
          float v150_data = r1[6];
          r1[6] = (v150_data + (v117_data * v148_data));
          float v153_data = s0[58];
          float v155_data = r1[7];
          r1[7] = (v155_data + (v117_data * v153_data));
          float v157_data = r0[3];
          float v158_data = s0[3];
          float v160_data = r1[0];
          r1[0] = (v160_data + (v157_data * v158_data));
          float v163_data = s0[11];
          float v165_data = r1[1];
          r1[1] = (v165_data + (v157_data * v163_data));
          float v168_data = s0[19];
          float v170_data = r1[2];
          r1[2] = (v170_data + (v157_data * v168_data));
          float v173_data = s0[27];
          float v175_data = r1[3];
          r1[3] = (v175_data + (v157_data * v173_data));
          float v178_data = s0[35];
          float v180_data = r1[4];
          r1[4] = (v180_data + (v157_data * v178_data));
          float v183_data = s0[43];
          float v185_data = r1[5];
          r1[5] = (v185_data + (v157_data * v183_data));
          float v188_data = s0[51];
          float v190_data = r1[6];
          r1[6] = (v190_data + (v157_data * v188_data));
          float v193_data = s0[59];
          float v195_data = r1[7];
          r1[7] = (v195_data + (v157_data * v193_data));
          float v197_data = r0[4];
          float v198_data = s0[4];
          float v200_data = r1[0];
          r1[0] = (v200_data + (v197_data * v198_data));
          float v203_data = s0[12];
          float v205_data = r1[1];
          r1[1] = (v205_data + (v197_data * v203_data));
          float v208_data = s0[20];
          float v210_data = r1[2];
          r1[2] = (v210_data + (v197_data * v208_data));
          float v213_data = s0[28];
          float v215_data = r1[3];
          r1[3] = (v215_data + (v197_data * v213_data));
          float v218_data = s0[36];
          float v220_data = r1[4];
          r1[4] = (v220_data + (v197_data * v218_data));
          float v223_data = s0[44];
          float v225_data = r1[5];
          r1[5] = (v225_data + (v197_data * v223_data));
          float v228_data = s0[52];
          float v230_data = r1[6];
          r1[6] = (v230_data + (v197_data * v228_data));
          float v233_data = s0[60];
          float v235_data = r1[7];
          r1[7] = (v235_data + (v197_data * v233_data));
          float v237_data = r0[5];
          float v238_data = s0[5];
          float v240_data = r1[0];
          r1[0] = (v240_data + (v237_data * v238_data));
          float v243_data = s0[13];
          float v245_data = r1[1];
          r1[1] = (v245_data + (v237_data * v243_data));
          float v248_data = s0[21];
          float v250_data = r1[2];
          r1[2] = (v250_data + (v237_data * v248_data));
          float v253_data = s0[29];
          float v255_data = r1[3];
          r1[3] = (v255_data + (v237_data * v253_data));
          float v258_data = s0[37];
          float v260_data = r1[4];
          r1[4] = (v260_data + (v237_data * v258_data));
          float v263_data = s0[45];
          float v265_data = r1[5];
          r1[5] = (v265_data + (v237_data * v263_data));
          float v268_data = s0[53];
          float v270_data = r1[6];
          r1[6] = (v270_data + (v237_data * v268_data));
          float v273_data = s0[61];
          float v275_data = r1[7];
          r1[7] = (v275_data + (v237_data * v273_data));
          float v277_data = r0[6];
          float v278_data = s0[6];
          float v280_data = r1[0];
          r1[0] = (v280_data + (v277_data * v278_data));
          float v283_data = s0[14];
          float v285_data = r1[1];
          r1[1] = (v285_data + (v277_data * v283_data));
          float v288_data = s0[22];
          float v290_data = r1[2];
          r1[2] = (v290_data + (v277_data * v288_data));
          float v293_data = s0[30];
          float v295_data = r1[3];
          r1[3] = (v295_data + (v277_data * v293_data));
          float v298_data = s0[38];
          float v300_data = r1[4];
          r1[4] = (v300_data + (v277_data * v298_data));
          float v303_data = s0[46];
          float v305_data = r1[5];
          r1[5] = (v305_data + (v277_data * v303_data));
          float v308_data = s0[54];
          float v310_data = r1[6];
          r1[6] = (v310_data + (v277_data * v308_data));
          float v313_data = s0[62];
          float v315_data = r1[7];
          r1[7] = (v315_data + (v277_data * v313_data));
          float v317_data = r0[7];
          float v318_data = s0[7];
          float v320_data = r1[0];
          r1[0] = (v320_data + (v317_data * v318_data));
          float v323_data = s0[15];
          float v325_data = r1[1];
          r1[1] = (v325_data + (v317_data * v323_data));
          float v328_data = s0[23];
          float v330_data = r1[2];
          r1[2] = (v330_data + (v317_data * v328_data));
          float v333_data = s0[31];
          float v335_data = r1[3];
          r1[3] = (v335_data + (v317_data * v333_data));
          float v338_data = s0[39];
          float v340_data = r1[4];
          r1[4] = (v340_data + (v317_data * v338_data));
          float v343_data = s0[47];
          float v345_data = r1[5];
          r1[5] = (v345_data + (v317_data * v343_data));
          float v348_data = s0[55];
          float v350_data = r1[6];
          r1[6] = (v350_data + (v317_data * v348_data));
          float v353_data = s0[63];
          float v355_data = r1[7];
          r1[7] = (v355_data + (v317_data * v353_data));
          // glb_m2 = +(r1, dims=[1])
          if (v26_g) {
            float v358_acc0 = 0.0f;
            #pragma unroll
            for (int32_t v357_r1 = 0; v357_r1 < 8; ++v357_r1) {
              float v360_data = r1[v357_r1];
              v358_acc0 = (v358_acc0 + v360_data);
            }
            glb_m2[v25_lead] = v358_acc0;
          }
          __syncwarp();
        }
      }
    }
  }
}

