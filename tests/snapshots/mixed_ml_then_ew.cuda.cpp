// === base name ===
kernel_ca970db30d10cb81

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_ca970db30d10cb81 = {{8, 16, 1}, 8, 8, 1, 16, 4608, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_ca970db30d10cb81(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_ca970db30d10cb81(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_ca970db30d10cb81(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (8, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_ca970db30d10cb81, block.x * block.y * block.z, 1152 * sizeof(float));
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
  config.block[0] = 8;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 1152 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_ca970db30d10cb81(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_ca970db30d10cb81(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_ca970db30d10cb81, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_ca970db30d10cb81<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_ca970db30d10cb81(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 8 lanes x 16 per block = block 8x16x1, 4608 B shared, occupancy grid
    // operands:
    //   m0 8×8(8×8) {0..8}×{0..8} strided
    //   m1 8×8(8×8) {0..8}×{0..8} strided
    //   m2 8×8(8×8) {0..8}×{0..8} strided
    // operations:
    //   t0[i,j] = m0[i,k] × m1[k,j]
    //   C = abs(TMP)
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1152}],"shared_bytes":4608,"shared_elements":1152,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[72 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v5_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v5_batchId0 < numElements0; v5_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v6_ahead1 = v5_batchId0 + (gridDim.x * blockDim.y);
        size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 64 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 64 + 0 + m1_extraOffset];
          float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 64 + 0 + m2_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v19_lead = threadIdx.x % 8;
          #pragma unroll
          for (int32_t v20_i0 = 0; v20_i0 < 1; ++v20_i0) {
            int32_t v23_lead = v19_lead + (v20_i0 * 8);
            #pragma unroll
            for (int32_t v21_i1 = 0; v21_i1 < 8; ++v21_i1) {
              float v26_data = __ldcg(&glb_m0[(v23_lead + (v21_i1 * 8))]);
              r0[(v20_i0 + v21_i1)] = v26_data;
            }
          }
          // s0 = load{g>s}(glb_m1[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 8; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 8], &glb_m1[0 + 0 + 1 * threadIdx.x + i * 8], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m0););
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          // r1 = +(r0 * s0) + None
          // [(0, 8), (0, 8)] [(0, 8)]
          float v30_data = r0[0];
          float v31_data = s0[0];
          float v33_data = r1[0];
          r1[0] = (v33_data + (v30_data * v31_data));
          float v36_data = s0[8];
          float v38_data = r1[1];
          r1[1] = (v38_data + (v30_data * v36_data));
          float v41_data = s0[16];
          float v43_data = r1[2];
          r1[2] = (v43_data + (v30_data * v41_data));
          float v46_data = s0[24];
          float v48_data = r1[3];
          r1[3] = (v48_data + (v30_data * v46_data));
          float v51_data = s0[32];
          float v53_data = r1[4];
          r1[4] = (v53_data + (v30_data * v51_data));
          float v56_data = s0[40];
          float v58_data = r1[5];
          r1[5] = (v58_data + (v30_data * v56_data));
          float v61_data = s0[48];
          float v63_data = r1[6];
          r1[6] = (v63_data + (v30_data * v61_data));
          float v66_data = s0[56];
          float v68_data = r1[7];
          r1[7] = (v68_data + (v30_data * v66_data));
          float v70_data = r0[1];
          float v71_data = s0[1];
          float v73_data = r1[0];
          r1[0] = (v73_data + (v70_data * v71_data));
          float v76_data = s0[9];
          float v78_data = r1[1];
          r1[1] = (v78_data + (v70_data * v76_data));
          float v81_data = s0[17];
          float v83_data = r1[2];
          r1[2] = (v83_data + (v70_data * v81_data));
          float v86_data = s0[25];
          float v88_data = r1[3];
          r1[3] = (v88_data + (v70_data * v86_data));
          float v91_data = s0[33];
          float v93_data = r1[4];
          r1[4] = (v93_data + (v70_data * v91_data));
          float v96_data = s0[41];
          float v98_data = r1[5];
          r1[5] = (v98_data + (v70_data * v96_data));
          float v101_data = s0[49];
          float v103_data = r1[6];
          r1[6] = (v103_data + (v70_data * v101_data));
          float v106_data = s0[57];
          float v108_data = r1[7];
          r1[7] = (v108_data + (v70_data * v106_data));
          float v110_data = r0[2];
          float v111_data = s0[2];
          float v113_data = r1[0];
          r1[0] = (v113_data + (v110_data * v111_data));
          float v116_data = s0[10];
          float v118_data = r1[1];
          r1[1] = (v118_data + (v110_data * v116_data));
          float v121_data = s0[18];
          float v123_data = r1[2];
          r1[2] = (v123_data + (v110_data * v121_data));
          float v126_data = s0[26];
          float v128_data = r1[3];
          r1[3] = (v128_data + (v110_data * v126_data));
          float v131_data = s0[34];
          float v133_data = r1[4];
          r1[4] = (v133_data + (v110_data * v131_data));
          float v136_data = s0[42];
          float v138_data = r1[5];
          r1[5] = (v138_data + (v110_data * v136_data));
          float v141_data = s0[50];
          float v143_data = r1[6];
          r1[6] = (v143_data + (v110_data * v141_data));
          float v146_data = s0[58];
          float v148_data = r1[7];
          r1[7] = (v148_data + (v110_data * v146_data));
          float v150_data = r0[3];
          float v151_data = s0[3];
          float v153_data = r1[0];
          r1[0] = (v153_data + (v150_data * v151_data));
          float v156_data = s0[11];
          float v158_data = r1[1];
          r1[1] = (v158_data + (v150_data * v156_data));
          float v161_data = s0[19];
          float v163_data = r1[2];
          r1[2] = (v163_data + (v150_data * v161_data));
          float v166_data = s0[27];
          float v168_data = r1[3];
          r1[3] = (v168_data + (v150_data * v166_data));
          float v171_data = s0[35];
          float v173_data = r1[4];
          r1[4] = (v173_data + (v150_data * v171_data));
          float v176_data = s0[43];
          float v178_data = r1[5];
          r1[5] = (v178_data + (v150_data * v176_data));
          float v181_data = s0[51];
          float v183_data = r1[6];
          r1[6] = (v183_data + (v150_data * v181_data));
          float v186_data = s0[59];
          float v188_data = r1[7];
          r1[7] = (v188_data + (v150_data * v186_data));
          float v190_data = r0[4];
          float v191_data = s0[4];
          float v193_data = r1[0];
          r1[0] = (v193_data + (v190_data * v191_data));
          float v196_data = s0[12];
          float v198_data = r1[1];
          r1[1] = (v198_data + (v190_data * v196_data));
          float v201_data = s0[20];
          float v203_data = r1[2];
          r1[2] = (v203_data + (v190_data * v201_data));
          float v206_data = s0[28];
          float v208_data = r1[3];
          r1[3] = (v208_data + (v190_data * v206_data));
          float v211_data = s0[36];
          float v213_data = r1[4];
          r1[4] = (v213_data + (v190_data * v211_data));
          float v216_data = s0[44];
          float v218_data = r1[5];
          r1[5] = (v218_data + (v190_data * v216_data));
          float v221_data = s0[52];
          float v223_data = r1[6];
          r1[6] = (v223_data + (v190_data * v221_data));
          float v226_data = s0[60];
          float v228_data = r1[7];
          r1[7] = (v228_data + (v190_data * v226_data));
          float v230_data = r0[5];
          float v231_data = s0[5];
          float v233_data = r1[0];
          r1[0] = (v233_data + (v230_data * v231_data));
          float v236_data = s0[13];
          float v238_data = r1[1];
          r1[1] = (v238_data + (v230_data * v236_data));
          float v241_data = s0[21];
          float v243_data = r1[2];
          r1[2] = (v243_data + (v230_data * v241_data));
          float v246_data = s0[29];
          float v248_data = r1[3];
          r1[3] = (v248_data + (v230_data * v246_data));
          float v251_data = s0[37];
          float v253_data = r1[4];
          r1[4] = (v253_data + (v230_data * v251_data));
          float v256_data = s0[45];
          float v258_data = r1[5];
          r1[5] = (v258_data + (v230_data * v256_data));
          float v261_data = s0[53];
          float v263_data = r1[6];
          r1[6] = (v263_data + (v230_data * v261_data));
          float v266_data = s0[61];
          float v268_data = r1[7];
          r1[7] = (v268_data + (v230_data * v266_data));
          float v270_data = r0[6];
          float v271_data = s0[6];
          float v273_data = r1[0];
          r1[0] = (v273_data + (v270_data * v271_data));
          float v276_data = s0[14];
          float v278_data = r1[1];
          r1[1] = (v278_data + (v270_data * v276_data));
          float v281_data = s0[22];
          float v283_data = r1[2];
          r1[2] = (v283_data + (v270_data * v281_data));
          float v286_data = s0[30];
          float v288_data = r1[3];
          r1[3] = (v288_data + (v270_data * v286_data));
          float v291_data = s0[38];
          float v293_data = r1[4];
          r1[4] = (v293_data + (v270_data * v291_data));
          float v296_data = s0[46];
          float v298_data = r1[5];
          r1[5] = (v298_data + (v270_data * v296_data));
          float v301_data = s0[54];
          float v303_data = r1[6];
          r1[6] = (v303_data + (v270_data * v301_data));
          float v306_data = s0[62];
          float v308_data = r1[7];
          r1[7] = (v308_data + (v270_data * v306_data));
          float v310_data = r0[7];
          float v311_data = s0[7];
          float v313_data = r1[0];
          r1[0] = (v313_data + (v310_data * v311_data));
          float v316_data = s0[15];
          float v318_data = r1[1];
          r1[1] = (v318_data + (v310_data * v316_data));
          float v321_data = s0[23];
          float v323_data = r1[2];
          r1[2] = (v323_data + (v310_data * v321_data));
          float v326_data = s0[31];
          float v328_data = r1[3];
          r1[3] = (v328_data + (v310_data * v326_data));
          float v331_data = s0[39];
          float v333_data = r1[4];
          r1[4] = (v333_data + (v310_data * v331_data));
          float v336_data = s0[47];
          float v338_data = r1[5];
          r1[5] = (v338_data + (v310_data * v336_data));
          float v341_data = s0[55];
          float v343_data = r1[6];
          r1[6] = (v343_data + (v310_data * v341_data));
          float v346_data = s0[63];
          float v348_data = r1[7];
          r1[7] = (v348_data + (v310_data * v346_data));
          // glb_m2 = abs(r1)
          #pragma unroll
          for (int32_t v350_k0 = 0; v350_k0 < 1; ++v350_k0) {
            int32_t v356_lead = v19_lead + (v350_k0 * 8);
            #pragma unroll
            for (int32_t v351_k1 = 0; v351_k1 < 8; ++v351_k1) {
              float v353_data = r1[(v350_k0 + v351_k1)];
              glb_m2[(v356_lead + (v351_k1 * 8))] = (fabsf(v353_data));
            }
          }
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
        }
      }
    }
  }
}

