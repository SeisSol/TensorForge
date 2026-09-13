// === base name ===
kernel_5764b1ff9331c950

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_5764b1ff9331c950 = {{32, 4, 1}, 32, 32, 1, 4, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_5764b1ff9331c950(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_5764b1ff9331c950(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_5764b1ff9331c950(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_5764b1ff9331c950, block.x * block.y * block.z, 256 * sizeof(float));
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
void launcher_kernel_5764b1ff9331c950(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_5764b1ff9331c950(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_5764b1ff9331c950, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_5764b1ff9331c950<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_5764b1ff9331c950(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 4 per block = block 32x4x1, 1024 B shared, occupancy grid
    // operands:
    //   m0 8×8(8×8) {0..8}×{0..8} strided
    //   m1 8×8(8×8) {0..8}×{0..8} strided
    //   m2 8×8(8×8) {0..8}×{0..8} strided
    // operations:
    //   t0[i,j] = m0[i,k] × m1[k,j]
    //   C = abs(TMP)
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[64 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v6_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v6_batchId0 < numElements0; v6_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v7_ahead1 = v6_batchId0 + (gridDim.x * blockDim.y);
        size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 64 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v6_batchId0 * 64 + 0 + m1_extraOffset];
          float *const __restrict__ glb_m2 = &m2[v6_batchId0 * 64 + 0 + m2_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v20_lead = threadIdx.x % 32;
          bool v21_g = v20_lead < 8;
          if (v21_g) {
            #pragma unroll
            for (int32_t v22_i1 = 0; v22_i1 < 8; ++v22_i1) {
              float v27_data = __ldcg(&glb_m0[(v20_lead + (v22_i1 * 8))]);
              r0[v22_i1] = v27_data;
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
          float v32_data = r0[0];
          float v33_data = s0[0];
          float v35_data = r1[0];
          r1[0] = (v35_data + (v32_data * v33_data));
          float v38_data = s0[8];
          float v40_data = r1[1];
          r1[1] = (v40_data + (v32_data * v38_data));
          float v43_data = s0[16];
          float v45_data = r1[2];
          r1[2] = (v45_data + (v32_data * v43_data));
          float v48_data = s0[24];
          float v50_data = r1[3];
          r1[3] = (v50_data + (v32_data * v48_data));
          float v53_data = s0[32];
          float v55_data = r1[4];
          r1[4] = (v55_data + (v32_data * v53_data));
          float v58_data = s0[40];
          float v60_data = r1[5];
          r1[5] = (v60_data + (v32_data * v58_data));
          float v63_data = s0[48];
          float v65_data = r1[6];
          r1[6] = (v65_data + (v32_data * v63_data));
          float v68_data = s0[56];
          float v70_data = r1[7];
          r1[7] = (v70_data + (v32_data * v68_data));
          float v72_data = r0[1];
          float v73_data = s0[1];
          float v75_data = r1[0];
          r1[0] = (v75_data + (v72_data * v73_data));
          float v78_data = s0[9];
          float v80_data = r1[1];
          r1[1] = (v80_data + (v72_data * v78_data));
          float v83_data = s0[17];
          float v85_data = r1[2];
          r1[2] = (v85_data + (v72_data * v83_data));
          float v88_data = s0[25];
          float v90_data = r1[3];
          r1[3] = (v90_data + (v72_data * v88_data));
          float v93_data = s0[33];
          float v95_data = r1[4];
          r1[4] = (v95_data + (v72_data * v93_data));
          float v98_data = s0[41];
          float v100_data = r1[5];
          r1[5] = (v100_data + (v72_data * v98_data));
          float v103_data = s0[49];
          float v105_data = r1[6];
          r1[6] = (v105_data + (v72_data * v103_data));
          float v108_data = s0[57];
          float v110_data = r1[7];
          r1[7] = (v110_data + (v72_data * v108_data));
          float v112_data = r0[2];
          float v113_data = s0[2];
          float v115_data = r1[0];
          r1[0] = (v115_data + (v112_data * v113_data));
          float v118_data = s0[10];
          float v120_data = r1[1];
          r1[1] = (v120_data + (v112_data * v118_data));
          float v123_data = s0[18];
          float v125_data = r1[2];
          r1[2] = (v125_data + (v112_data * v123_data));
          float v128_data = s0[26];
          float v130_data = r1[3];
          r1[3] = (v130_data + (v112_data * v128_data));
          float v133_data = s0[34];
          float v135_data = r1[4];
          r1[4] = (v135_data + (v112_data * v133_data));
          float v138_data = s0[42];
          float v140_data = r1[5];
          r1[5] = (v140_data + (v112_data * v138_data));
          float v143_data = s0[50];
          float v145_data = r1[6];
          r1[6] = (v145_data + (v112_data * v143_data));
          float v148_data = s0[58];
          float v150_data = r1[7];
          r1[7] = (v150_data + (v112_data * v148_data));
          float v152_data = r0[3];
          float v153_data = s0[3];
          float v155_data = r1[0];
          r1[0] = (v155_data + (v152_data * v153_data));
          float v158_data = s0[11];
          float v160_data = r1[1];
          r1[1] = (v160_data + (v152_data * v158_data));
          float v163_data = s0[19];
          float v165_data = r1[2];
          r1[2] = (v165_data + (v152_data * v163_data));
          float v168_data = s0[27];
          float v170_data = r1[3];
          r1[3] = (v170_data + (v152_data * v168_data));
          float v173_data = s0[35];
          float v175_data = r1[4];
          r1[4] = (v175_data + (v152_data * v173_data));
          float v178_data = s0[43];
          float v180_data = r1[5];
          r1[5] = (v180_data + (v152_data * v178_data));
          float v183_data = s0[51];
          float v185_data = r1[6];
          r1[6] = (v185_data + (v152_data * v183_data));
          float v188_data = s0[59];
          float v190_data = r1[7];
          r1[7] = (v190_data + (v152_data * v188_data));
          float v192_data = r0[4];
          float v193_data = s0[4];
          float v195_data = r1[0];
          r1[0] = (v195_data + (v192_data * v193_data));
          float v198_data = s0[12];
          float v200_data = r1[1];
          r1[1] = (v200_data + (v192_data * v198_data));
          float v203_data = s0[20];
          float v205_data = r1[2];
          r1[2] = (v205_data + (v192_data * v203_data));
          float v208_data = s0[28];
          float v210_data = r1[3];
          r1[3] = (v210_data + (v192_data * v208_data));
          float v213_data = s0[36];
          float v215_data = r1[4];
          r1[4] = (v215_data + (v192_data * v213_data));
          float v218_data = s0[44];
          float v220_data = r1[5];
          r1[5] = (v220_data + (v192_data * v218_data));
          float v223_data = s0[52];
          float v225_data = r1[6];
          r1[6] = (v225_data + (v192_data * v223_data));
          float v228_data = s0[60];
          float v230_data = r1[7];
          r1[7] = (v230_data + (v192_data * v228_data));
          float v232_data = r0[5];
          float v233_data = s0[5];
          float v235_data = r1[0];
          r1[0] = (v235_data + (v232_data * v233_data));
          float v238_data = s0[13];
          float v240_data = r1[1];
          r1[1] = (v240_data + (v232_data * v238_data));
          float v243_data = s0[21];
          float v245_data = r1[2];
          r1[2] = (v245_data + (v232_data * v243_data));
          float v248_data = s0[29];
          float v250_data = r1[3];
          r1[3] = (v250_data + (v232_data * v248_data));
          float v253_data = s0[37];
          float v255_data = r1[4];
          r1[4] = (v255_data + (v232_data * v253_data));
          float v258_data = s0[45];
          float v260_data = r1[5];
          r1[5] = (v260_data + (v232_data * v258_data));
          float v263_data = s0[53];
          float v265_data = r1[6];
          r1[6] = (v265_data + (v232_data * v263_data));
          float v268_data = s0[61];
          float v270_data = r1[7];
          r1[7] = (v270_data + (v232_data * v268_data));
          float v272_data = r0[6];
          float v273_data = s0[6];
          float v275_data = r1[0];
          r1[0] = (v275_data + (v272_data * v273_data));
          float v278_data = s0[14];
          float v280_data = r1[1];
          r1[1] = (v280_data + (v272_data * v278_data));
          float v283_data = s0[22];
          float v285_data = r1[2];
          r1[2] = (v285_data + (v272_data * v283_data));
          float v288_data = s0[30];
          float v290_data = r1[3];
          r1[3] = (v290_data + (v272_data * v288_data));
          float v293_data = s0[38];
          float v295_data = r1[4];
          r1[4] = (v295_data + (v272_data * v293_data));
          float v298_data = s0[46];
          float v300_data = r1[5];
          r1[5] = (v300_data + (v272_data * v298_data));
          float v303_data = s0[54];
          float v305_data = r1[6];
          r1[6] = (v305_data + (v272_data * v303_data));
          float v308_data = s0[62];
          float v310_data = r1[7];
          r1[7] = (v310_data + (v272_data * v308_data));
          float v312_data = r0[7];
          float v313_data = s0[7];
          float v315_data = r1[0];
          r1[0] = (v315_data + (v312_data * v313_data));
          float v318_data = s0[15];
          float v320_data = r1[1];
          r1[1] = (v320_data + (v312_data * v318_data));
          float v323_data = s0[23];
          float v325_data = r1[2];
          r1[2] = (v325_data + (v312_data * v323_data));
          float v328_data = s0[31];
          float v330_data = r1[3];
          r1[3] = (v330_data + (v312_data * v328_data));
          float v333_data = s0[39];
          float v335_data = r1[4];
          r1[4] = (v335_data + (v312_data * v333_data));
          float v338_data = s0[47];
          float v340_data = r1[5];
          r1[5] = (v340_data + (v312_data * v338_data));
          float v343_data = s0[55];
          float v345_data = r1[6];
          r1[6] = (v345_data + (v312_data * v343_data));
          float v348_data = s0[63];
          float v350_data = r1[7];
          r1[7] = (v350_data + (v312_data * v348_data));
          __syncwarp();
          // s1 = store{r>s}(localShrMem0, r1);
          if (v21_g) {
            #pragma unroll
            for (int32_t v352_i1 = 0; v352_i1 < 8; ++v352_i1) {
              float v354_data = r1[v352_i1];
              int32_t v358_a = v20_lead + (v352_i1 * 8);
              s1[(v358_a ^ ((v358_a >> 5) & 31))] = v354_data;
            }
          }
          __syncwarp();
          // glb_m2 = abs(s1)
          if (v21_g) {
            #pragma unroll
            for (int32_t v362_k1 = 0; v362_k1 < 8; ++v362_k1) {
              int32_t v366_a = v20_lead + (v362_k1 * 8);
              float v370_data = s1[(v366_a ^ ((v366_a >> 5) & 31))];
              glb_m2[v366_a] = (fabsf(v370_data));
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

