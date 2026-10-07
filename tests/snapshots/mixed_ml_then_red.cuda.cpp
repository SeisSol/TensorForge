// === base name ===
kernel_7edac91d4ad5cff9

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_7edac91d4ad5cff9 = {{32, 4, 1}, 32, 32, 1, 4, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_7edac91d4ad5cff9(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_7edac91d4ad5cff9(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_7edac91d4ad5cff9(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_7edac91d4ad5cff9, block.x * block.y * block.z, 256 * sizeof(float));
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
void launcher_kernel_7edac91d4ad5cff9(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_7edac91d4ad5cff9(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_7edac91d4ad5cff9, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_7edac91d4ad5cff9<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_7edac91d4ad5cff9(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 64 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 64 + 0 + m1_extraOffset];
          float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 8 + 0 + m2_extraOffset];
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
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m1[0 + 0 + 1 * threadIdx.x + 32], 4);
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m0););
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          // r1 = +(r0 * s0) + None
          // [(0, 8), (0, 8)] [(0, 8)]
          float v34_data = r0[0];
          __syncwarp();
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
          float v55_data = s0[32];
          float v57_data = r1[4];
          r1[4] = (v57_data + (v34_data * v55_data));
          float v60_data = s0[40];
          float v62_data = r1[5];
          r1[5] = (v62_data + (v34_data * v60_data));
          float v65_data = s0[48];
          float v67_data = r1[6];
          r1[6] = (v67_data + (v34_data * v65_data));
          float v70_data = s0[56];
          float v72_data = r1[7];
          r1[7] = (v72_data + (v34_data * v70_data));
          float v74_data = r0[1];
          float v75_data = s0[1];
          float v77_data = r1[0];
          r1[0] = (v77_data + (v74_data * v75_data));
          float v80_data = s0[9];
          float v82_data = r1[1];
          r1[1] = (v82_data + (v74_data * v80_data));
          float v85_data = s0[17];
          float v87_data = r1[2];
          r1[2] = (v87_data + (v74_data * v85_data));
          float v90_data = s0[25];
          float v92_data = r1[3];
          r1[3] = (v92_data + (v74_data * v90_data));
          float v95_data = s0[33];
          float v97_data = r1[4];
          r1[4] = (v97_data + (v74_data * v95_data));
          float v100_data = s0[41];
          float v102_data = r1[5];
          r1[5] = (v102_data + (v74_data * v100_data));
          float v105_data = s0[49];
          float v107_data = r1[6];
          r1[6] = (v107_data + (v74_data * v105_data));
          float v110_data = s0[57];
          float v112_data = r1[7];
          r1[7] = (v112_data + (v74_data * v110_data));
          float v114_data = r0[2];
          float v115_data = s0[2];
          float v117_data = r1[0];
          r1[0] = (v117_data + (v114_data * v115_data));
          float v120_data = s0[10];
          float v122_data = r1[1];
          r1[1] = (v122_data + (v114_data * v120_data));
          float v125_data = s0[18];
          float v127_data = r1[2];
          r1[2] = (v127_data + (v114_data * v125_data));
          float v130_data = s0[26];
          float v132_data = r1[3];
          r1[3] = (v132_data + (v114_data * v130_data));
          float v135_data = s0[34];
          float v137_data = r1[4];
          r1[4] = (v137_data + (v114_data * v135_data));
          float v140_data = s0[42];
          float v142_data = r1[5];
          r1[5] = (v142_data + (v114_data * v140_data));
          float v145_data = s0[50];
          float v147_data = r1[6];
          r1[6] = (v147_data + (v114_data * v145_data));
          float v150_data = s0[58];
          float v152_data = r1[7];
          r1[7] = (v152_data + (v114_data * v150_data));
          float v154_data = r0[3];
          float v155_data = s0[3];
          float v157_data = r1[0];
          r1[0] = (v157_data + (v154_data * v155_data));
          float v160_data = s0[11];
          float v162_data = r1[1];
          r1[1] = (v162_data + (v154_data * v160_data));
          float v165_data = s0[19];
          float v167_data = r1[2];
          r1[2] = (v167_data + (v154_data * v165_data));
          float v170_data = s0[27];
          float v172_data = r1[3];
          r1[3] = (v172_data + (v154_data * v170_data));
          float v175_data = s0[35];
          float v177_data = r1[4];
          r1[4] = (v177_data + (v154_data * v175_data));
          float v180_data = s0[43];
          float v182_data = r1[5];
          r1[5] = (v182_data + (v154_data * v180_data));
          float v185_data = s0[51];
          float v187_data = r1[6];
          r1[6] = (v187_data + (v154_data * v185_data));
          float v190_data = s0[59];
          float v192_data = r1[7];
          r1[7] = (v192_data + (v154_data * v190_data));
          float v194_data = r0[4];
          float v195_data = s0[4];
          float v197_data = r1[0];
          r1[0] = (v197_data + (v194_data * v195_data));
          float v200_data = s0[12];
          float v202_data = r1[1];
          r1[1] = (v202_data + (v194_data * v200_data));
          float v205_data = s0[20];
          float v207_data = r1[2];
          r1[2] = (v207_data + (v194_data * v205_data));
          float v210_data = s0[28];
          float v212_data = r1[3];
          r1[3] = (v212_data + (v194_data * v210_data));
          float v215_data = s0[36];
          float v217_data = r1[4];
          r1[4] = (v217_data + (v194_data * v215_data));
          float v220_data = s0[44];
          float v222_data = r1[5];
          r1[5] = (v222_data + (v194_data * v220_data));
          float v225_data = s0[52];
          float v227_data = r1[6];
          r1[6] = (v227_data + (v194_data * v225_data));
          float v230_data = s0[60];
          float v232_data = r1[7];
          r1[7] = (v232_data + (v194_data * v230_data));
          float v234_data = r0[5];
          float v235_data = s0[5];
          float v237_data = r1[0];
          r1[0] = (v237_data + (v234_data * v235_data));
          float v240_data = s0[13];
          float v242_data = r1[1];
          r1[1] = (v242_data + (v234_data * v240_data));
          float v245_data = s0[21];
          float v247_data = r1[2];
          r1[2] = (v247_data + (v234_data * v245_data));
          float v250_data = s0[29];
          float v252_data = r1[3];
          r1[3] = (v252_data + (v234_data * v250_data));
          float v255_data = s0[37];
          float v257_data = r1[4];
          r1[4] = (v257_data + (v234_data * v255_data));
          float v260_data = s0[45];
          float v262_data = r1[5];
          r1[5] = (v262_data + (v234_data * v260_data));
          float v265_data = s0[53];
          float v267_data = r1[6];
          r1[6] = (v267_data + (v234_data * v265_data));
          float v270_data = s0[61];
          float v272_data = r1[7];
          r1[7] = (v272_data + (v234_data * v270_data));
          float v274_data = r0[6];
          float v275_data = s0[6];
          float v277_data = r1[0];
          r1[0] = (v277_data + (v274_data * v275_data));
          float v280_data = s0[14];
          float v282_data = r1[1];
          r1[1] = (v282_data + (v274_data * v280_data));
          float v285_data = s0[22];
          float v287_data = r1[2];
          r1[2] = (v287_data + (v274_data * v285_data));
          float v290_data = s0[30];
          float v292_data = r1[3];
          r1[3] = (v292_data + (v274_data * v290_data));
          float v295_data = s0[38];
          float v297_data = r1[4];
          r1[4] = (v297_data + (v274_data * v295_data));
          float v300_data = s0[46];
          float v302_data = r1[5];
          r1[5] = (v302_data + (v274_data * v300_data));
          float v305_data = s0[54];
          float v307_data = r1[6];
          r1[6] = (v307_data + (v274_data * v305_data));
          float v310_data = s0[62];
          float v312_data = r1[7];
          r1[7] = (v312_data + (v274_data * v310_data));
          float v314_data = r0[7];
          float v315_data = s0[7];
          float v317_data = r1[0];
          r1[0] = (v317_data + (v314_data * v315_data));
          float v320_data = s0[15];
          float v322_data = r1[1];
          r1[1] = (v322_data + (v314_data * v320_data));
          float v325_data = s0[23];
          float v327_data = r1[2];
          r1[2] = (v327_data + (v314_data * v325_data));
          float v330_data = s0[31];
          float v332_data = r1[3];
          r1[3] = (v332_data + (v314_data * v330_data));
          float v335_data = s0[39];
          float v337_data = r1[4];
          r1[4] = (v337_data + (v314_data * v335_data));
          float v340_data = s0[47];
          float v342_data = r1[5];
          r1[5] = (v342_data + (v314_data * v340_data));
          float v345_data = s0[55];
          float v347_data = r1[6];
          r1[6] = (v347_data + (v314_data * v345_data));
          float v350_data = s0[63];
          float v352_data = r1[7];
          r1[7] = (v352_data + (v314_data * v350_data));
          // glb_m2 = +(r1, dims=[1])
          if (v23_g) {
            float v355_acc0 = 0.0f;
            #pragma unroll
            for (int32_t v354_r1 = 0; v354_r1 < 8; ++v354_r1) {
              float v357_data = r1[v354_r1];
              v355_acc0 = (v355_acc0 + v357_data);
            }
            glb_m2[v22_lead] = v355_acc0;
          }
          __syncwarp();
        }
      }
    }
  }
}

