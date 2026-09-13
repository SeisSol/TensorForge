// === base name ===
kernel_96241f40069e5851

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_96241f40069e5851 = {{8, 16, 1}, 8, 8, 1, 16, 4608, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_96241f40069e5851(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_96241f40069e5851(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_96241f40069e5851(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_96241f40069e5851, block.x * block.y * block.z, 1152 * sizeof(float));
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
void launcher_kernel_96241f40069e5851(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_96241f40069e5851(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_96241f40069e5851, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_96241f40069e5851<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_96241f40069e5851(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
          int32_t v20_lead = threadIdx.x % 8;
          #pragma unroll
          for (int32_t v21_i0 = 0; v21_i0 < 1; ++v21_i0) {
            int32_t v24_lead = v20_lead + (v21_i0 * 8);
            #pragma unroll
            for (int32_t v22_i1 = 0; v22_i1 < 8; ++v22_i1) {
              float v27_data = __ldcg(&glb_m0[(v24_lead + (v22_i1 * 8))]);
              r0[(v21_i0 + v22_i1)] = v27_data;
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
          float v31_data = r0[0];
          float v32_data = s0[0];
          float v34_data = r1[0];
          r1[0] = (v34_data + (v31_data * v32_data));
          float v37_data = s0[8];
          float v39_data = r1[1];
          r1[1] = (v39_data + (v31_data * v37_data));
          float v42_data = s0[16];
          float v44_data = r1[2];
          r1[2] = (v44_data + (v31_data * v42_data));
          float v47_data = s0[24];
          float v49_data = r1[3];
          r1[3] = (v49_data + (v31_data * v47_data));
          float v52_data = s0[32];
          float v54_data = r1[4];
          r1[4] = (v54_data + (v31_data * v52_data));
          float v57_data = s0[40];
          float v59_data = r1[5];
          r1[5] = (v59_data + (v31_data * v57_data));
          float v62_data = s0[48];
          float v64_data = r1[6];
          r1[6] = (v64_data + (v31_data * v62_data));
          float v67_data = s0[56];
          float v69_data = r1[7];
          r1[7] = (v69_data + (v31_data * v67_data));
          float v71_data = r0[1];
          float v72_data = s0[1];
          float v74_data = r1[0];
          r1[0] = (v74_data + (v71_data * v72_data));
          float v77_data = s0[9];
          float v79_data = r1[1];
          r1[1] = (v79_data + (v71_data * v77_data));
          float v82_data = s0[17];
          float v84_data = r1[2];
          r1[2] = (v84_data + (v71_data * v82_data));
          float v87_data = s0[25];
          float v89_data = r1[3];
          r1[3] = (v89_data + (v71_data * v87_data));
          float v92_data = s0[33];
          float v94_data = r1[4];
          r1[4] = (v94_data + (v71_data * v92_data));
          float v97_data = s0[41];
          float v99_data = r1[5];
          r1[5] = (v99_data + (v71_data * v97_data));
          float v102_data = s0[49];
          float v104_data = r1[6];
          r1[6] = (v104_data + (v71_data * v102_data));
          float v107_data = s0[57];
          float v109_data = r1[7];
          r1[7] = (v109_data + (v71_data * v107_data));
          float v111_data = r0[2];
          float v112_data = s0[2];
          float v114_data = r1[0];
          r1[0] = (v114_data + (v111_data * v112_data));
          float v117_data = s0[10];
          float v119_data = r1[1];
          r1[1] = (v119_data + (v111_data * v117_data));
          float v122_data = s0[18];
          float v124_data = r1[2];
          r1[2] = (v124_data + (v111_data * v122_data));
          float v127_data = s0[26];
          float v129_data = r1[3];
          r1[3] = (v129_data + (v111_data * v127_data));
          float v132_data = s0[34];
          float v134_data = r1[4];
          r1[4] = (v134_data + (v111_data * v132_data));
          float v137_data = s0[42];
          float v139_data = r1[5];
          r1[5] = (v139_data + (v111_data * v137_data));
          float v142_data = s0[50];
          float v144_data = r1[6];
          r1[6] = (v144_data + (v111_data * v142_data));
          float v147_data = s0[58];
          float v149_data = r1[7];
          r1[7] = (v149_data + (v111_data * v147_data));
          float v151_data = r0[3];
          float v152_data = s0[3];
          float v154_data = r1[0];
          r1[0] = (v154_data + (v151_data * v152_data));
          float v157_data = s0[11];
          float v159_data = r1[1];
          r1[1] = (v159_data + (v151_data * v157_data));
          float v162_data = s0[19];
          float v164_data = r1[2];
          r1[2] = (v164_data + (v151_data * v162_data));
          float v167_data = s0[27];
          float v169_data = r1[3];
          r1[3] = (v169_data + (v151_data * v167_data));
          float v172_data = s0[35];
          float v174_data = r1[4];
          r1[4] = (v174_data + (v151_data * v172_data));
          float v177_data = s0[43];
          float v179_data = r1[5];
          r1[5] = (v179_data + (v151_data * v177_data));
          float v182_data = s0[51];
          float v184_data = r1[6];
          r1[6] = (v184_data + (v151_data * v182_data));
          float v187_data = s0[59];
          float v189_data = r1[7];
          r1[7] = (v189_data + (v151_data * v187_data));
          float v191_data = r0[4];
          float v192_data = s0[4];
          float v194_data = r1[0];
          r1[0] = (v194_data + (v191_data * v192_data));
          float v197_data = s0[12];
          float v199_data = r1[1];
          r1[1] = (v199_data + (v191_data * v197_data));
          float v202_data = s0[20];
          float v204_data = r1[2];
          r1[2] = (v204_data + (v191_data * v202_data));
          float v207_data = s0[28];
          float v209_data = r1[3];
          r1[3] = (v209_data + (v191_data * v207_data));
          float v212_data = s0[36];
          float v214_data = r1[4];
          r1[4] = (v214_data + (v191_data * v212_data));
          float v217_data = s0[44];
          float v219_data = r1[5];
          r1[5] = (v219_data + (v191_data * v217_data));
          float v222_data = s0[52];
          float v224_data = r1[6];
          r1[6] = (v224_data + (v191_data * v222_data));
          float v227_data = s0[60];
          float v229_data = r1[7];
          r1[7] = (v229_data + (v191_data * v227_data));
          float v231_data = r0[5];
          float v232_data = s0[5];
          float v234_data = r1[0];
          r1[0] = (v234_data + (v231_data * v232_data));
          float v237_data = s0[13];
          float v239_data = r1[1];
          r1[1] = (v239_data + (v231_data * v237_data));
          float v242_data = s0[21];
          float v244_data = r1[2];
          r1[2] = (v244_data + (v231_data * v242_data));
          float v247_data = s0[29];
          float v249_data = r1[3];
          r1[3] = (v249_data + (v231_data * v247_data));
          float v252_data = s0[37];
          float v254_data = r1[4];
          r1[4] = (v254_data + (v231_data * v252_data));
          float v257_data = s0[45];
          float v259_data = r1[5];
          r1[5] = (v259_data + (v231_data * v257_data));
          float v262_data = s0[53];
          float v264_data = r1[6];
          r1[6] = (v264_data + (v231_data * v262_data));
          float v267_data = s0[61];
          float v269_data = r1[7];
          r1[7] = (v269_data + (v231_data * v267_data));
          float v271_data = r0[6];
          float v272_data = s0[6];
          float v274_data = r1[0];
          r1[0] = (v274_data + (v271_data * v272_data));
          float v277_data = s0[14];
          float v279_data = r1[1];
          r1[1] = (v279_data + (v271_data * v277_data));
          float v282_data = s0[22];
          float v284_data = r1[2];
          r1[2] = (v284_data + (v271_data * v282_data));
          float v287_data = s0[30];
          float v289_data = r1[3];
          r1[3] = (v289_data + (v271_data * v287_data));
          float v292_data = s0[38];
          float v294_data = r1[4];
          r1[4] = (v294_data + (v271_data * v292_data));
          float v297_data = s0[46];
          float v299_data = r1[5];
          r1[5] = (v299_data + (v271_data * v297_data));
          float v302_data = s0[54];
          float v304_data = r1[6];
          r1[6] = (v304_data + (v271_data * v302_data));
          float v307_data = s0[62];
          float v309_data = r1[7];
          r1[7] = (v309_data + (v271_data * v307_data));
          float v311_data = r0[7];
          float v312_data = s0[7];
          float v314_data = r1[0];
          r1[0] = (v314_data + (v311_data * v312_data));
          float v317_data = s0[15];
          float v319_data = r1[1];
          r1[1] = (v319_data + (v311_data * v317_data));
          float v322_data = s0[23];
          float v324_data = r1[2];
          r1[2] = (v324_data + (v311_data * v322_data));
          float v327_data = s0[31];
          float v329_data = r1[3];
          r1[3] = (v329_data + (v311_data * v327_data));
          float v332_data = s0[39];
          float v334_data = r1[4];
          r1[4] = (v334_data + (v311_data * v332_data));
          float v337_data = s0[47];
          float v339_data = r1[5];
          r1[5] = (v339_data + (v311_data * v337_data));
          float v342_data = s0[55];
          float v344_data = r1[6];
          r1[6] = (v344_data + (v311_data * v342_data));
          float v347_data = s0[63];
          float v349_data = r1[7];
          r1[7] = (v349_data + (v311_data * v347_data));
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          // s1 = store{r>s}(localShrMem0, r1);
          #pragma unroll
          for (int32_t v351_i0 = 0; v351_i0 < 1; ++v351_i0) {
            int32_t v356_lead = v20_lead + (v351_i0 * 8);
            #pragma unroll
            for (int32_t v352_i1 = 0; v352_i1 < 8; ++v352_i1) {
              float v354_data = r1[(v351_i0 + v352_i1)];
              int32_t v358_a = v356_lead + (v352_i1 * 8);
              s1[(v358_a ^ ((v358_a >> 5) & 31))] = v354_data;
            }
          }
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          // glb_m2 = abs(s1)
          #pragma unroll
          for (int32_t v362_k0 = 0; v362_k0 < 1; ++v362_k0) {
            int32_t v365_lead = v20_lead + (v362_k0 * 8);
            #pragma unroll
            for (int32_t v363_k1 = 0; v363_k1 < 8; ++v363_k1) {
              int32_t v367_a = v365_lead + (v363_k1 * 8);
              float v371_data = s1[(v367_a ^ ((v367_a >> 5) & 31))];
              glb_m2[v367_a] = (fabsf(v371_data));
            }
          }
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
        }
      }
    }
  }
}

