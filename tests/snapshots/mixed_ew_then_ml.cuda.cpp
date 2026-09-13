// === base name ===
kernel_ee6983e687b6ba7a

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_ee6983e687b6ba7a = {{8, 16, 1}, 8, 8, 1, 16, 4608, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_ee6983e687b6ba7a(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_ee6983e687b6ba7a(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_ee6983e687b6ba7a(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_ee6983e687b6ba7a, block.x * block.y * block.z, 1152 * sizeof(float));
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
void launcher_kernel_ee6983e687b6ba7a(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_ee6983e687b6ba7a(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_ee6983e687b6ba7a, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_ee6983e687b6ba7a<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_ee6983e687b6ba7a(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
    //   TMP = abs(A)
    //   m1[i,j] = t0[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1152}],"shared_bytes":4608,"shared_elements":1152,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[72 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v5_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v5_batchId0 < numElements0; v5_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v6_ahead1 = v5_batchId0 + (gridDim.x * blockDim.y);
        size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 64 + 0 + m0_extraOffset];
          float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 64 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 64 + 0 + m2_extraOffset];
          // s1 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 8; i += 1) {
            __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + i * 8], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 8], 4);
          }
          __pipeline_commit();
          float r0[8]{};
          // r0 = abs(glb_m0)
          int32_t v20_lead = threadIdx.x % 8;
          #pragma unroll
          for (int32_t v21_k0 = 0; v21_k0 < 1; ++v21_k0) {
            int32_t v24_lead = v20_lead + (v21_k0 * 8);
            #pragma unroll
            for (int32_t v22_k1 = 0; v22_k1 < 8; ++v22_k1) {
              float v27_data = glb_m0[(v24_lead + (v22_k1 * 8))];
              r0[(v21_k0 + v22_k1)] = (fabsf(v27_data));
            }
          }
          // wait(s1 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          // ir1 = +(r0 * s1)
          // [(0, 8), (0, 8)] [(0, 8)]
          float ir1[8]{};
          float v35_data = r0[0];
          float v36_data = s1[0];
          float v38_data = ir1[0];
          ir1[0] = (v38_data + (v35_data * v36_data));
          float v41_data = s1[8];
          float v43_data = ir1[1];
          ir1[1] = (v43_data + (v35_data * v41_data));
          float v46_data = s1[16];
          float v48_data = ir1[2];
          ir1[2] = (v48_data + (v35_data * v46_data));
          float v51_data = s1[24];
          float v53_data = ir1[3];
          ir1[3] = (v53_data + (v35_data * v51_data));
          float v56_data = s1[32];
          float v58_data = ir1[4];
          ir1[4] = (v58_data + (v35_data * v56_data));
          float v61_data = s1[40];
          float v63_data = ir1[5];
          ir1[5] = (v63_data + (v35_data * v61_data));
          float v66_data = s1[48];
          float v68_data = ir1[6];
          ir1[6] = (v68_data + (v35_data * v66_data));
          float v71_data = s1[56];
          float v73_data = ir1[7];
          ir1[7] = (v73_data + (v35_data * v71_data));
          float v75_data = r0[1];
          float v76_data = s1[1];
          float v78_data = ir1[0];
          ir1[0] = (v78_data + (v75_data * v76_data));
          float v81_data = s1[9];
          float v83_data = ir1[1];
          ir1[1] = (v83_data + (v75_data * v81_data));
          float v86_data = s1[17];
          float v88_data = ir1[2];
          ir1[2] = (v88_data + (v75_data * v86_data));
          float v91_data = s1[25];
          float v93_data = ir1[3];
          ir1[3] = (v93_data + (v75_data * v91_data));
          float v96_data = s1[33];
          float v98_data = ir1[4];
          ir1[4] = (v98_data + (v75_data * v96_data));
          float v101_data = s1[41];
          float v103_data = ir1[5];
          ir1[5] = (v103_data + (v75_data * v101_data));
          float v106_data = s1[49];
          float v108_data = ir1[6];
          ir1[6] = (v108_data + (v75_data * v106_data));
          float v111_data = s1[57];
          float v113_data = ir1[7];
          ir1[7] = (v113_data + (v75_data * v111_data));
          float v115_data = r0[2];
          float v116_data = s1[2];
          float v118_data = ir1[0];
          ir1[0] = (v118_data + (v115_data * v116_data));
          float v121_data = s1[10];
          float v123_data = ir1[1];
          ir1[1] = (v123_data + (v115_data * v121_data));
          float v126_data = s1[18];
          float v128_data = ir1[2];
          ir1[2] = (v128_data + (v115_data * v126_data));
          float v131_data = s1[26];
          float v133_data = ir1[3];
          ir1[3] = (v133_data + (v115_data * v131_data));
          float v136_data = s1[34];
          float v138_data = ir1[4];
          ir1[4] = (v138_data + (v115_data * v136_data));
          float v141_data = s1[42];
          float v143_data = ir1[5];
          ir1[5] = (v143_data + (v115_data * v141_data));
          float v146_data = s1[50];
          float v148_data = ir1[6];
          ir1[6] = (v148_data + (v115_data * v146_data));
          float v151_data = s1[58];
          float v153_data = ir1[7];
          ir1[7] = (v153_data + (v115_data * v151_data));
          float v155_data = r0[3];
          float v156_data = s1[3];
          float v158_data = ir1[0];
          ir1[0] = (v158_data + (v155_data * v156_data));
          float v161_data = s1[11];
          float v163_data = ir1[1];
          ir1[1] = (v163_data + (v155_data * v161_data));
          float v166_data = s1[19];
          float v168_data = ir1[2];
          ir1[2] = (v168_data + (v155_data * v166_data));
          float v171_data = s1[27];
          float v173_data = ir1[3];
          ir1[3] = (v173_data + (v155_data * v171_data));
          float v176_data = s1[35];
          float v178_data = ir1[4];
          ir1[4] = (v178_data + (v155_data * v176_data));
          float v181_data = s1[43];
          float v183_data = ir1[5];
          ir1[5] = (v183_data + (v155_data * v181_data));
          float v186_data = s1[51];
          float v188_data = ir1[6];
          ir1[6] = (v188_data + (v155_data * v186_data));
          float v191_data = s1[59];
          float v193_data = ir1[7];
          ir1[7] = (v193_data + (v155_data * v191_data));
          float v195_data = r0[4];
          float v196_data = s1[4];
          float v198_data = ir1[0];
          ir1[0] = (v198_data + (v195_data * v196_data));
          float v201_data = s1[12];
          float v203_data = ir1[1];
          ir1[1] = (v203_data + (v195_data * v201_data));
          float v206_data = s1[20];
          float v208_data = ir1[2];
          ir1[2] = (v208_data + (v195_data * v206_data));
          float v211_data = s1[28];
          float v213_data = ir1[3];
          ir1[3] = (v213_data + (v195_data * v211_data));
          float v216_data = s1[36];
          float v218_data = ir1[4];
          ir1[4] = (v218_data + (v195_data * v216_data));
          float v221_data = s1[44];
          float v223_data = ir1[5];
          ir1[5] = (v223_data + (v195_data * v221_data));
          float v226_data = s1[52];
          float v228_data = ir1[6];
          ir1[6] = (v228_data + (v195_data * v226_data));
          float v231_data = s1[60];
          float v233_data = ir1[7];
          ir1[7] = (v233_data + (v195_data * v231_data));
          float v235_data = r0[5];
          float v236_data = s1[5];
          float v238_data = ir1[0];
          ir1[0] = (v238_data + (v235_data * v236_data));
          float v241_data = s1[13];
          float v243_data = ir1[1];
          ir1[1] = (v243_data + (v235_data * v241_data));
          float v246_data = s1[21];
          float v248_data = ir1[2];
          ir1[2] = (v248_data + (v235_data * v246_data));
          float v251_data = s1[29];
          float v253_data = ir1[3];
          ir1[3] = (v253_data + (v235_data * v251_data));
          float v256_data = s1[37];
          float v258_data = ir1[4];
          ir1[4] = (v258_data + (v235_data * v256_data));
          float v261_data = s1[45];
          float v263_data = ir1[5];
          ir1[5] = (v263_data + (v235_data * v261_data));
          float v266_data = s1[53];
          float v268_data = ir1[6];
          ir1[6] = (v268_data + (v235_data * v266_data));
          float v271_data = s1[61];
          float v273_data = ir1[7];
          ir1[7] = (v273_data + (v235_data * v271_data));
          float v275_data = r0[6];
          float v276_data = s1[6];
          float v278_data = ir1[0];
          ir1[0] = (v278_data + (v275_data * v276_data));
          float v281_data = s1[14];
          float v283_data = ir1[1];
          ir1[1] = (v283_data + (v275_data * v281_data));
          float v286_data = s1[22];
          float v288_data = ir1[2];
          ir1[2] = (v288_data + (v275_data * v286_data));
          float v291_data = s1[30];
          float v293_data = ir1[3];
          ir1[3] = (v293_data + (v275_data * v291_data));
          float v296_data = s1[38];
          float v298_data = ir1[4];
          ir1[4] = (v298_data + (v275_data * v296_data));
          float v301_data = s1[46];
          float v303_data = ir1[5];
          ir1[5] = (v303_data + (v275_data * v301_data));
          float v306_data = s1[54];
          float v308_data = ir1[6];
          ir1[6] = (v308_data + (v275_data * v306_data));
          float v311_data = s1[62];
          float v313_data = ir1[7];
          ir1[7] = (v313_data + (v275_data * v311_data));
          float v315_data = r0[7];
          float v316_data = s1[7];
          float v318_data = ir1[0];
          ir1[0] = (v318_data + (v315_data * v316_data));
          float v321_data = s1[15];
          float v323_data = ir1[1];
          ir1[1] = (v323_data + (v315_data * v321_data));
          float v326_data = s1[23];
          float v328_data = ir1[2];
          ir1[2] = (v328_data + (v315_data * v326_data));
          float v331_data = s1[31];
          float v333_data = ir1[3];
          ir1[3] = (v333_data + (v315_data * v331_data));
          float v336_data = s1[39];
          float v338_data = ir1[4];
          ir1[4] = (v338_data + (v315_data * v336_data));
          float v341_data = s1[47];
          float v343_data = ir1[5];
          ir1[5] = (v343_data + (v315_data * v341_data));
          float v346_data = s1[55];
          float v348_data = ir1[6];
          ir1[6] = (v348_data + (v315_data * v346_data));
          float v351_data = s1[63];
          float v353_data = ir1[7];
          ir1[7] = (v353_data + (v315_data * v351_data));
          // r1 = ir1
          #pragma unroll
          for (int32_t v358_n0 = 0; v358_n0 < 1; ++v358_n0) {
            #pragma unroll
            for (int32_t v359_n1 = 0; v359_n1 < 8; ++v359_n1) {
              int32_t v360_a = v358_n0 + v359_n1;
              float v361_data = ir1[v360_a];
              r1[v360_a] = v361_data;
            }
          }
          // glb_m1 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v365_i0 = 0; v365_i0 < 1; ++v365_i0) {
            int32_t v370_lead = v20_lead + (v365_i0 * 8);
            #pragma unroll
            for (int32_t v366_i1 = 0; v366_i1 < 8; ++v366_i1) {
              float v368_data = r1[(v365_i0 + v366_i1)];
              glb_m1[(v370_lead + (v366_i1 * 8))] = v368_data;
            }
          }
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
        }
      }
    }
  }
}

