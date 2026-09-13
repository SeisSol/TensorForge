// === base name ===
kernel_62aa1b52be732939

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_62aa1b52be732939 = {{32, 4, 1}, 32, 32, 1, 4, 1536, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_62aa1b52be732939(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_62aa1b52be732939(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_62aa1b52be732939(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_62aa1b52be732939, block.x * block.y * block.z, 384 * sizeof(float));
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
  config.sharedMemBytes = 384 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_62aa1b52be732939(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_62aa1b52be732939(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_62aa1b52be732939, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_62aa1b52be732939<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_62aa1b52be732939(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 4 per block = block 32x4x1, 1536 B shared, occupancy grid
    // operands:
    //   m0 8×8(8×8) {0..8}×{0..8} strided
    //   m1 8×4(8×4) {0..8}×{0..4} strided
    //   m2 8×4(8×4) {0..8}×{0..4} strided
    //   m3 8×8(8×8) {0..8}×{0..8} strided
    // operations:
    //   t0[i,j]@{0..8}×{0..4} = m0[i,k] × m1[k,j]
    //   t0[i,j]@{0..8}×{4..8} = m0[i,k] × m2[k,j]
    //   C = abs(TMP)
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":384}],"shared_bytes":1536,"shared_elements":384,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[8,4]],"name":"m1","ordered":false,"parts":1,"shape":[8,4],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[8,4]],"name":"m2","ordered":false,"parts":1,"shape":[8,4],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,4]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,4]],"is_tmp":true,"name":"t0","offset":[0,4],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[96 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[96];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s2 = &localShrMem0[64];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 64 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 32 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 32 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[v7_batchId0 * 64 + 0 + m3_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v22_lead = threadIdx.x % 32;
          bool v23_g = v22_lead < 8;
          if (v23_g) {
            #pragma unroll
            for (int32_t v24_i1 = 0; v24_i1 < 8; ++v24_i1) {
              float v29_data = __ldcg(&glb_m0[(v22_lead + (v24_i1 * 8))]);
              r0[v24_i1] = v29_data;
            }
          }
          // s0 = load{g>s}(glb_m1[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m1[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m0););
          // s2 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_commit();
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(1);
          float r1[4]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 8), (0, 4)] [(0, 8)]
          float v34_data = r0[0];
          float v35_data = s0[0];
          float v37_data = r1[0];
          r1[0] = (v37_data + (v34_data * v35_data));
          float v40_data = s0[8];
          float v42_data = r1[1];
          r1[1] = (v42_data + (v34_data * v40_data));
          float v45_data = s0[16];
          float v47_data = r1[2];
          r1[2] = (v47_data + (v34_data * v45_data));
          float v50_data = s0[24];
          float v52_data = r1[3];
          r1[3] = (v52_data + (v34_data * v50_data));
          float v54_data = r0[1];
          float v55_data = s0[1];
          float v57_data = r1[0];
          r1[0] = (v57_data + (v54_data * v55_data));
          float v60_data = s0[9];
          float v62_data = r1[1];
          r1[1] = (v62_data + (v54_data * v60_data));
          float v65_data = s0[17];
          float v67_data = r1[2];
          r1[2] = (v67_data + (v54_data * v65_data));
          float v70_data = s0[25];
          float v72_data = r1[3];
          r1[3] = (v72_data + (v54_data * v70_data));
          float v74_data = r0[2];
          float v75_data = s0[2];
          float v77_data = r1[0];
          r1[0] = (v77_data + (v74_data * v75_data));
          float v80_data = s0[10];
          float v82_data = r1[1];
          r1[1] = (v82_data + (v74_data * v80_data));
          float v85_data = s0[18];
          float v87_data = r1[2];
          r1[2] = (v87_data + (v74_data * v85_data));
          float v90_data = s0[26];
          float v92_data = r1[3];
          r1[3] = (v92_data + (v74_data * v90_data));
          float v94_data = r0[3];
          float v95_data = s0[3];
          float v97_data = r1[0];
          r1[0] = (v97_data + (v94_data * v95_data));
          float v100_data = s0[11];
          float v102_data = r1[1];
          r1[1] = (v102_data + (v94_data * v100_data));
          float v105_data = s0[19];
          float v107_data = r1[2];
          r1[2] = (v107_data + (v94_data * v105_data));
          float v110_data = s0[27];
          float v112_data = r1[3];
          r1[3] = (v112_data + (v94_data * v110_data));
          float v114_data = r0[4];
          float v115_data = s0[4];
          float v117_data = r1[0];
          r1[0] = (v117_data + (v114_data * v115_data));
          float v120_data = s0[12];
          float v122_data = r1[1];
          r1[1] = (v122_data + (v114_data * v120_data));
          float v125_data = s0[20];
          float v127_data = r1[2];
          r1[2] = (v127_data + (v114_data * v125_data));
          float v130_data = s0[28];
          float v132_data = r1[3];
          r1[3] = (v132_data + (v114_data * v130_data));
          float v134_data = r0[5];
          float v135_data = s0[5];
          float v137_data = r1[0];
          r1[0] = (v137_data + (v134_data * v135_data));
          float v140_data = s0[13];
          float v142_data = r1[1];
          r1[1] = (v142_data + (v134_data * v140_data));
          float v145_data = s0[21];
          float v147_data = r1[2];
          r1[2] = (v147_data + (v134_data * v145_data));
          float v150_data = s0[29];
          float v152_data = r1[3];
          r1[3] = (v152_data + (v134_data * v150_data));
          float v154_data = r0[6];
          float v155_data = s0[6];
          float v157_data = r1[0];
          r1[0] = (v157_data + (v154_data * v155_data));
          float v160_data = s0[14];
          float v162_data = r1[1];
          r1[1] = (v162_data + (v154_data * v160_data));
          float v165_data = s0[22];
          float v167_data = r1[2];
          r1[2] = (v167_data + (v154_data * v165_data));
          float v170_data = s0[30];
          float v172_data = r1[3];
          r1[3] = (v172_data + (v154_data * v170_data));
          float v174_data = r0[7];
          float v175_data = s0[7];
          float v177_data = r1[0];
          r1[0] = (v177_data + (v174_data * v175_data));
          float v180_data = s0[15];
          float v182_data = r1[1];
          r1[1] = (v182_data + (v174_data * v180_data));
          float v185_data = s0[23];
          float v187_data = r1[2];
          r1[2] = (v187_data + (v174_data * v185_data));
          float v190_data = s0[31];
          float v192_data = r1[3];
          r1[3] = (v192_data + (v174_data * v190_data));
          __syncwarp();
          // s1 = store{r>s}(localShrMem0, r1);
          if (v23_g) {
            #pragma unroll
            for (int32_t v194_i1 = 0; v194_i1 < 4; ++v194_i1) {
              float v196_data = r1[v194_i1];
              int32_t v200_a = v22_lead + (v194_i1 * 8);
              s1[(v200_a ^ ((v200_a >> 5) & 31))] = v196_data;
            }
          }
          // wait(s2 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r2[4]{};
          // r2 = +(r0 * s2) + None
          // [(0, 8), (0, 4)] [(0, 8)]
          float ir2[4]{};
          float v207_data = s2[0];
          float v209_data = ir2[0];
          ir2[0] = (v209_data + (v34_data * v207_data));
          float v212_data = s2[8];
          float v214_data = ir2[1];
          ir2[1] = (v214_data + (v34_data * v212_data));
          float v217_data = s2[16];
          float v219_data = ir2[2];
          ir2[2] = (v219_data + (v34_data * v217_data));
          float v222_data = s2[24];
          float v224_data = ir2[3];
          ir2[3] = (v224_data + (v34_data * v222_data));
          float v227_data = s2[1];
          float v229_data = ir2[0];
          ir2[0] = (v229_data + (v54_data * v227_data));
          float v232_data = s2[9];
          float v234_data = ir2[1];
          ir2[1] = (v234_data + (v54_data * v232_data));
          float v237_data = s2[17];
          float v239_data = ir2[2];
          ir2[2] = (v239_data + (v54_data * v237_data));
          float v242_data = s2[25];
          float v244_data = ir2[3];
          ir2[3] = (v244_data + (v54_data * v242_data));
          float v247_data = s2[2];
          float v249_data = ir2[0];
          ir2[0] = (v249_data + (v74_data * v247_data));
          float v252_data = s2[10];
          float v254_data = ir2[1];
          ir2[1] = (v254_data + (v74_data * v252_data));
          float v257_data = s2[18];
          float v259_data = ir2[2];
          ir2[2] = (v259_data + (v74_data * v257_data));
          float v262_data = s2[26];
          float v264_data = ir2[3];
          ir2[3] = (v264_data + (v74_data * v262_data));
          float v267_data = s2[3];
          float v269_data = ir2[0];
          ir2[0] = (v269_data + (v94_data * v267_data));
          float v272_data = s2[11];
          float v274_data = ir2[1];
          ir2[1] = (v274_data + (v94_data * v272_data));
          float v277_data = s2[19];
          float v279_data = ir2[2];
          ir2[2] = (v279_data + (v94_data * v277_data));
          float v282_data = s2[27];
          float v284_data = ir2[3];
          ir2[3] = (v284_data + (v94_data * v282_data));
          float v287_data = s2[4];
          float v289_data = ir2[0];
          ir2[0] = (v289_data + (v114_data * v287_data));
          float v292_data = s2[12];
          float v294_data = ir2[1];
          ir2[1] = (v294_data + (v114_data * v292_data));
          float v297_data = s2[20];
          float v299_data = ir2[2];
          ir2[2] = (v299_data + (v114_data * v297_data));
          float v302_data = s2[28];
          float v304_data = ir2[3];
          ir2[3] = (v304_data + (v114_data * v302_data));
          float v307_data = s2[5];
          float v309_data = ir2[0];
          ir2[0] = (v309_data + (v134_data * v307_data));
          float v312_data = s2[13];
          float v314_data = ir2[1];
          ir2[1] = (v314_data + (v134_data * v312_data));
          float v317_data = s2[21];
          float v319_data = ir2[2];
          ir2[2] = (v319_data + (v134_data * v317_data));
          float v322_data = s2[29];
          float v324_data = ir2[3];
          ir2[3] = (v324_data + (v134_data * v322_data));
          float v327_data = s2[6];
          float v329_data = ir2[0];
          ir2[0] = (v329_data + (v154_data * v327_data));
          float v332_data = s2[14];
          float v334_data = ir2[1];
          ir2[1] = (v334_data + (v154_data * v332_data));
          float v337_data = s2[22];
          float v339_data = ir2[2];
          ir2[2] = (v339_data + (v154_data * v337_data));
          float v342_data = s2[30];
          float v344_data = ir2[3];
          ir2[3] = (v344_data + (v154_data * v342_data));
          float v347_data = s2[7];
          float v349_data = ir2[0];
          ir2[0] = (v349_data + (v174_data * v347_data));
          float v352_data = s2[15];
          float v354_data = ir2[1];
          ir2[1] = (v354_data + (v174_data * v352_data));
          float v357_data = s2[23];
          float v359_data = ir2[2];
          ir2[2] = (v359_data + (v174_data * v357_data));
          float v362_data = s2[31];
          float v364_data = ir2[3];
          ir2[3] = (v364_data + (v174_data * v362_data));
          if (v23_g) {
            #pragma unroll
            for (int32_t v366_n1 = 0; v366_n1 < 4; ++v366_n1) {
              float v368_data = ir2[v366_n1];
              r2[v366_n1] = v368_data;
            }
          }
          // s1 = store{r>s}(localShrMem0, r2);
          if (v23_g) {
            #pragma unroll
            for (int32_t v369_i1 = 0; v369_i1 < 4; ++v369_i1) {
              float v371_data = r2[v369_i1];
              int32_t v376_a = v22_lead + ((v369_i1 + 4) * 8);
              s1[(v376_a ^ ((v376_a >> 5) & 31))] = v371_data;
            }
          }
          __syncwarp();
          // glb_m3 = abs(s1)
          if (v23_g) {
            #pragma unroll
            for (int32_t v380_k1 = 0; v380_k1 < 8; ++v380_k1) {
              int32_t v384_a = v22_lead + (v380_k1 * 8);
              float v388_data = s1[(v384_a ^ ((v384_a >> 5) & 31))];
              glb_m3[v384_a] = (fabsf(v388_data));
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

