// === base name ===
kernel_ca19e50784278db9

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_ca19e50784278db9 = {{32, 4, 1}, 32, 35, 1, 4, 512, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_ca19e50784278db9(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_ca19e50784278db9(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_ca19e50784278db9(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_ca19e50784278db9, block.x * block.y * block.z, 128 * sizeof(float));
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
  config.sharedMemBytes = 128 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_ca19e50784278db9(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_ca19e50784278db9(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_ca19e50784278db9, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_ca19e50784278db9<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_ca19e50784278db9(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (35 active) x 4 per block = block 32x4x1, 512 B shared, occupancy grid
    // operands:
    //   m0 35×4(35×4) {0..35}×{0..4} strided
    //   m1 35×8(35×8) {0..35}×{0..8} strided
    //   m2 8×4(8×4) {0..8}×{0..4} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":35,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":128}],"shared_bytes":512,"shared_elements":128,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[35,4]],"name":"m0","ordered":false,"parts":1,"shape":[35,4],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[35,8]],"name":"m1","ordered":false,"parts":1,"shape":[35,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,4]],"name":"m2","ordered":false,"parts":1,"shape":[8,4],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[35,4]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[35,4]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[35,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[35,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[32 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[32];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v11_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v11_batchId0 < numElements0; v11_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v12_ahead1 = v11_batchId0 + (gridDim.x * blockDim.y);
        size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v11_batchId0 * 140 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v11_batchId0 * 280 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v11_batchId0 * 32 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v25_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v26_i0 = 0; v26_i0 < 1; ++v26_i0) {
            int32_t v29_lead = v25_lead + (v26_i0 * 32);
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 8; ++v27_i1) {
              float v32_data = __ldcg(&glb_m1[(v29_lead + (v27_i1 * 35))]);
              r0[(v26_i0 + (v27_i1 * 2))] = v32_data;
            }
          }
          bool v35_g = v25_lead < 3;
          if (v35_g) {
            int32_t v38_lead = v25_lead + 32_i32;
            #pragma unroll
            for (int32_t v36_i1 = 0; v36_i1 < 8; ++v36_i1) {
              float v41_data = __ldcg(&glb_m1[(v38_lead + (v36_i1 * 35))]);
              r0[(1 + (v36_i1 * 2))] = v41_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp();
          // ir1 = +(r0 * s0)
          // [(0, 35), (0, 4)] [(0, 8)]
          float ir1[8]{};
          float v47_data = r0[0];
          float v48_data = s0[0];
          float v50_data = ir1[0];
          ir1[0] = (v50_data + (v47_data * v48_data));
          float v53_data = s0[8];
          float v55_data = ir1[2];
          ir1[2] = (v55_data + (v47_data * v53_data));
          float v58_data = s0[16];
          float v60_data = ir1[4];
          ir1[4] = (v60_data + (v47_data * v58_data));
          float v63_data = s0[24];
          float v65_data = ir1[6];
          ir1[6] = (v65_data + (v47_data * v63_data));
          float v67_data = r0[1];
          float v70_data = ir1[1];
          ir1[1] = (v70_data + (v67_data * v48_data));
          float v75_data = ir1[3];
          ir1[3] = (v75_data + (v67_data * v53_data));
          float v80_data = ir1[5];
          ir1[5] = (v80_data + (v67_data * v58_data));
          float v85_data = ir1[7];
          ir1[7] = (v85_data + (v67_data * v63_data));
          float v87_data = r0[2];
          float v88_data = s0[1];
          float v90_data = ir1[0];
          ir1[0] = (v90_data + (v87_data * v88_data));
          float v93_data = s0[9];
          float v95_data = ir1[2];
          ir1[2] = (v95_data + (v87_data * v93_data));
          float v98_data = s0[17];
          float v100_data = ir1[4];
          ir1[4] = (v100_data + (v87_data * v98_data));
          float v103_data = s0[25];
          float v105_data = ir1[6];
          ir1[6] = (v105_data + (v87_data * v103_data));
          float v107_data = r0[3];
          float v110_data = ir1[1];
          ir1[1] = (v110_data + (v107_data * v88_data));
          float v115_data = ir1[3];
          ir1[3] = (v115_data + (v107_data * v93_data));
          float v120_data = ir1[5];
          ir1[5] = (v120_data + (v107_data * v98_data));
          float v125_data = ir1[7];
          ir1[7] = (v125_data + (v107_data * v103_data));
          float v127_data = r0[4];
          float v128_data = s0[2];
          float v130_data = ir1[0];
          ir1[0] = (v130_data + (v127_data * v128_data));
          float v133_data = s0[10];
          float v135_data = ir1[2];
          ir1[2] = (v135_data + (v127_data * v133_data));
          float v138_data = s0[18];
          float v140_data = ir1[4];
          ir1[4] = (v140_data + (v127_data * v138_data));
          float v143_data = s0[26];
          float v145_data = ir1[6];
          ir1[6] = (v145_data + (v127_data * v143_data));
          float v147_data = r0[5];
          float v150_data = ir1[1];
          ir1[1] = (v150_data + (v147_data * v128_data));
          float v155_data = ir1[3];
          ir1[3] = (v155_data + (v147_data * v133_data));
          float v160_data = ir1[5];
          ir1[5] = (v160_data + (v147_data * v138_data));
          float v165_data = ir1[7];
          ir1[7] = (v165_data + (v147_data * v143_data));
          float v167_data = r0[6];
          float v168_data = s0[3];
          float v170_data = ir1[0];
          ir1[0] = (v170_data + (v167_data * v168_data));
          float v173_data = s0[11];
          float v175_data = ir1[2];
          ir1[2] = (v175_data + (v167_data * v173_data));
          float v178_data = s0[19];
          float v180_data = ir1[4];
          ir1[4] = (v180_data + (v167_data * v178_data));
          float v183_data = s0[27];
          float v185_data = ir1[6];
          ir1[6] = (v185_data + (v167_data * v183_data));
          float v187_data = r0[7];
          float v190_data = ir1[1];
          ir1[1] = (v190_data + (v187_data * v168_data));
          float v195_data = ir1[3];
          ir1[3] = (v195_data + (v187_data * v173_data));
          float v200_data = ir1[5];
          ir1[5] = (v200_data + (v187_data * v178_data));
          float v205_data = ir1[7];
          ir1[7] = (v205_data + (v187_data * v183_data));
          float v207_data = r0[8];
          float v208_data = s0[4];
          float v210_data = ir1[0];
          ir1[0] = (v210_data + (v207_data * v208_data));
          float v213_data = s0[12];
          float v215_data = ir1[2];
          ir1[2] = (v215_data + (v207_data * v213_data));
          float v218_data = s0[20];
          float v220_data = ir1[4];
          ir1[4] = (v220_data + (v207_data * v218_data));
          float v223_data = s0[28];
          float v225_data = ir1[6];
          ir1[6] = (v225_data + (v207_data * v223_data));
          float v227_data = r0[9];
          float v230_data = ir1[1];
          ir1[1] = (v230_data + (v227_data * v208_data));
          float v235_data = ir1[3];
          ir1[3] = (v235_data + (v227_data * v213_data));
          float v240_data = ir1[5];
          ir1[5] = (v240_data + (v227_data * v218_data));
          float v245_data = ir1[7];
          ir1[7] = (v245_data + (v227_data * v223_data));
          float v247_data = r0[10];
          float v248_data = s0[5];
          float v250_data = ir1[0];
          ir1[0] = (v250_data + (v247_data * v248_data));
          float v253_data = s0[13];
          float v255_data = ir1[2];
          ir1[2] = (v255_data + (v247_data * v253_data));
          float v258_data = s0[21];
          float v260_data = ir1[4];
          ir1[4] = (v260_data + (v247_data * v258_data));
          float v263_data = s0[29];
          float v265_data = ir1[6];
          ir1[6] = (v265_data + (v247_data * v263_data));
          float v267_data = r0[11];
          float v270_data = ir1[1];
          ir1[1] = (v270_data + (v267_data * v248_data));
          float v275_data = ir1[3];
          ir1[3] = (v275_data + (v267_data * v253_data));
          float v280_data = ir1[5];
          ir1[5] = (v280_data + (v267_data * v258_data));
          float v285_data = ir1[7];
          ir1[7] = (v285_data + (v267_data * v263_data));
          float v287_data = r0[12];
          float v288_data = s0[6];
          float v290_data = ir1[0];
          ir1[0] = (v290_data + (v287_data * v288_data));
          float v293_data = s0[14];
          float v295_data = ir1[2];
          ir1[2] = (v295_data + (v287_data * v293_data));
          float v298_data = s0[22];
          float v300_data = ir1[4];
          ir1[4] = (v300_data + (v287_data * v298_data));
          float v303_data = s0[30];
          float v305_data = ir1[6];
          ir1[6] = (v305_data + (v287_data * v303_data));
          float v307_data = r0[13];
          float v310_data = ir1[1];
          ir1[1] = (v310_data + (v307_data * v288_data));
          float v315_data = ir1[3];
          ir1[3] = (v315_data + (v307_data * v293_data));
          float v320_data = ir1[5];
          ir1[5] = (v320_data + (v307_data * v298_data));
          float v325_data = ir1[7];
          ir1[7] = (v325_data + (v307_data * v303_data));
          float v327_data = r0[14];
          float v328_data = s0[7];
          float v330_data = ir1[0];
          ir1[0] = (v330_data + (v327_data * v328_data));
          float v333_data = s0[15];
          float v335_data = ir1[2];
          ir1[2] = (v335_data + (v327_data * v333_data));
          float v338_data = s0[23];
          float v340_data = ir1[4];
          ir1[4] = (v340_data + (v327_data * v338_data));
          float v343_data = s0[31];
          float v345_data = ir1[6];
          ir1[6] = (v345_data + (v327_data * v343_data));
          float v347_data = r0[15];
          float v350_data = ir1[1];
          ir1[1] = (v350_data + (v347_data * v328_data));
          float v355_data = ir1[3];
          ir1[3] = (v355_data + (v347_data * v333_data));
          float v360_data = ir1[5];
          ir1[5] = (v360_data + (v347_data * v338_data));
          float v365_data = ir1[7];
          ir1[7] = (v365_data + (v347_data * v343_data));
          // r1 = ir1
          #pragma unroll
          for (int32_t v367_n0 = 0; v367_n0 < 1; ++v367_n0) {
            #pragma unroll
            for (int32_t v368_n1 = 0; v368_n1 < 4; ++v368_n1) {
              int32_t v370_a = v367_n0 + (v368_n1 * 2);
              float v371_data = ir1[v370_a];
              r1[v370_a] = v371_data;
            }
          }
          if (v35_g) {
            #pragma unroll
            for (int32_t v372_n1 = 0; v372_n1 < 4; ++v372_n1) {
              int32_t v374_a = 1 + (v372_n1 * 2);
              float v375_data = ir1[v374_a];
              r1[v374_a] = v375_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v376_i0 = 0; v376_i0 < 1; ++v376_i0) {
            int32_t v382_lead = v25_lead + (v376_i0 * 32);
            #pragma unroll
            for (int32_t v377_i1 = 0; v377_i1 < 4; ++v377_i1) {
              float v380_data = r1[(v376_i0 + (v377_i1 * 2))];
              glb_m0[(v382_lead + (v377_i1 * 35))] = v380_data;
            }
          }
          if (v35_g) {
            int32_t v390_lead = v25_lead + 32_i32;
            #pragma unroll
            for (int32_t v385_i1 = 0; v385_i1 < 4; ++v385_i1) {
              float v388_data = r1[(1 + (v385_i1 * 2))];
              glb_m0[(v390_lead + (v385_i1 * 35))] = v388_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

