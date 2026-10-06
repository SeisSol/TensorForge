// === base name ===
kernel_910f92dd269e3825

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_910f92dd269e3825 = {{8, 16, 1}, 8, 8, 1, 16, 4608, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_910f92dd269e3825(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_910f92dd269e3825(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_910f92dd269e3825(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_910f92dd269e3825, block.x * block.y * block.z, 1152 * sizeof(float));
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
void launcher_kernel_910f92dd269e3825(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_910f92dd269e3825(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_910f92dd269e3825, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_910f92dd269e3825<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_910f92dd269e3825(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
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
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1152}],"shared_bytes":4608,"shared_elements":1152,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[72 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v11_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v11_batchId0 < numElements0; v11_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v12_ahead1 = v11_batchId0 + (gridDim.x * blockDim.y);
        size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v11_batchId0 * 64 + 0 + m0_extraOffset];
          float *const __restrict__ glb_m1 = &m1[v11_batchId0 * 64 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v11_batchId0 * 64 + 0 + m2_extraOffset];
          // s1 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 8; i += 1) {
            __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + i * 8], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 8], 4);
          }
          __pipeline_commit();
          float r0[8]{};
          // r0 = abs(glb_m0)
          int32_t v26_lead = threadIdx.x % 8;
          #pragma unroll
          for (int32_t v27_k0 = 0; v27_k0 < 1; ++v27_k0) {
            int32_t v30_lead = v26_lead + (v27_k0 * 8);
            #pragma unroll
            for (int32_t v28_k1 = 0; v28_k1 < 8; ++v28_k1) {
              float v33_data = glb_m0[(v30_lead + (v28_k1 * 8))];
              r0[(v27_k0 + v28_k1)] = (fabsf(v33_data));
            }
          }
          // wait(s1 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          // ir1 = +(r0 * s1)
          // [(0, 8), (0, 8)] [(0, 8)]
          float ir1[8]{};
          float v41_data = r0[0];
          float v42_data = s1[0];
          float v44_data = ir1[0];
          ir1[0] = (v44_data + (v41_data * v42_data));
          float v47_data = s1[8];
          float v49_data = ir1[1];
          ir1[1] = (v49_data + (v41_data * v47_data));
          float v52_data = s1[16];
          float v54_data = ir1[2];
          ir1[2] = (v54_data + (v41_data * v52_data));
          float v57_data = s1[24];
          float v59_data = ir1[3];
          ir1[3] = (v59_data + (v41_data * v57_data));
          float v62_data = s1[32];
          float v64_data = ir1[4];
          ir1[4] = (v64_data + (v41_data * v62_data));
          float v67_data = s1[40];
          float v69_data = ir1[5];
          ir1[5] = (v69_data + (v41_data * v67_data));
          float v72_data = s1[48];
          float v74_data = ir1[6];
          ir1[6] = (v74_data + (v41_data * v72_data));
          float v77_data = s1[56];
          float v79_data = ir1[7];
          ir1[7] = (v79_data + (v41_data * v77_data));
          float v81_data = r0[1];
          float v82_data = s1[1];
          float v84_data = ir1[0];
          ir1[0] = (v84_data + (v81_data * v82_data));
          float v87_data = s1[9];
          float v89_data = ir1[1];
          ir1[1] = (v89_data + (v81_data * v87_data));
          float v92_data = s1[17];
          float v94_data = ir1[2];
          ir1[2] = (v94_data + (v81_data * v92_data));
          float v97_data = s1[25];
          float v99_data = ir1[3];
          ir1[3] = (v99_data + (v81_data * v97_data));
          float v102_data = s1[33];
          float v104_data = ir1[4];
          ir1[4] = (v104_data + (v81_data * v102_data));
          float v107_data = s1[41];
          float v109_data = ir1[5];
          ir1[5] = (v109_data + (v81_data * v107_data));
          float v112_data = s1[49];
          float v114_data = ir1[6];
          ir1[6] = (v114_data + (v81_data * v112_data));
          float v117_data = s1[57];
          float v119_data = ir1[7];
          ir1[7] = (v119_data + (v81_data * v117_data));
          float v121_data = r0[2];
          float v122_data = s1[2];
          float v124_data = ir1[0];
          ir1[0] = (v124_data + (v121_data * v122_data));
          float v127_data = s1[10];
          float v129_data = ir1[1];
          ir1[1] = (v129_data + (v121_data * v127_data));
          float v132_data = s1[18];
          float v134_data = ir1[2];
          ir1[2] = (v134_data + (v121_data * v132_data));
          float v137_data = s1[26];
          float v139_data = ir1[3];
          ir1[3] = (v139_data + (v121_data * v137_data));
          float v142_data = s1[34];
          float v144_data = ir1[4];
          ir1[4] = (v144_data + (v121_data * v142_data));
          float v147_data = s1[42];
          float v149_data = ir1[5];
          ir1[5] = (v149_data + (v121_data * v147_data));
          float v152_data = s1[50];
          float v154_data = ir1[6];
          ir1[6] = (v154_data + (v121_data * v152_data));
          float v157_data = s1[58];
          float v159_data = ir1[7];
          ir1[7] = (v159_data + (v121_data * v157_data));
          float v161_data = r0[3];
          float v162_data = s1[3];
          float v164_data = ir1[0];
          ir1[0] = (v164_data + (v161_data * v162_data));
          float v167_data = s1[11];
          float v169_data = ir1[1];
          ir1[1] = (v169_data + (v161_data * v167_data));
          float v172_data = s1[19];
          float v174_data = ir1[2];
          ir1[2] = (v174_data + (v161_data * v172_data));
          float v177_data = s1[27];
          float v179_data = ir1[3];
          ir1[3] = (v179_data + (v161_data * v177_data));
          float v182_data = s1[35];
          float v184_data = ir1[4];
          ir1[4] = (v184_data + (v161_data * v182_data));
          float v187_data = s1[43];
          float v189_data = ir1[5];
          ir1[5] = (v189_data + (v161_data * v187_data));
          float v192_data = s1[51];
          float v194_data = ir1[6];
          ir1[6] = (v194_data + (v161_data * v192_data));
          float v197_data = s1[59];
          float v199_data = ir1[7];
          ir1[7] = (v199_data + (v161_data * v197_data));
          float v201_data = r0[4];
          float v202_data = s1[4];
          float v204_data = ir1[0];
          ir1[0] = (v204_data + (v201_data * v202_data));
          float v207_data = s1[12];
          float v209_data = ir1[1];
          ir1[1] = (v209_data + (v201_data * v207_data));
          float v212_data = s1[20];
          float v214_data = ir1[2];
          ir1[2] = (v214_data + (v201_data * v212_data));
          float v217_data = s1[28];
          float v219_data = ir1[3];
          ir1[3] = (v219_data + (v201_data * v217_data));
          float v222_data = s1[36];
          float v224_data = ir1[4];
          ir1[4] = (v224_data + (v201_data * v222_data));
          float v227_data = s1[44];
          float v229_data = ir1[5];
          ir1[5] = (v229_data + (v201_data * v227_data));
          float v232_data = s1[52];
          float v234_data = ir1[6];
          ir1[6] = (v234_data + (v201_data * v232_data));
          float v237_data = s1[60];
          float v239_data = ir1[7];
          ir1[7] = (v239_data + (v201_data * v237_data));
          float v241_data = r0[5];
          float v242_data = s1[5];
          float v244_data = ir1[0];
          ir1[0] = (v244_data + (v241_data * v242_data));
          float v247_data = s1[13];
          float v249_data = ir1[1];
          ir1[1] = (v249_data + (v241_data * v247_data));
          float v252_data = s1[21];
          float v254_data = ir1[2];
          ir1[2] = (v254_data + (v241_data * v252_data));
          float v257_data = s1[29];
          float v259_data = ir1[3];
          ir1[3] = (v259_data + (v241_data * v257_data));
          float v262_data = s1[37];
          float v264_data = ir1[4];
          ir1[4] = (v264_data + (v241_data * v262_data));
          float v267_data = s1[45];
          float v269_data = ir1[5];
          ir1[5] = (v269_data + (v241_data * v267_data));
          float v272_data = s1[53];
          float v274_data = ir1[6];
          ir1[6] = (v274_data + (v241_data * v272_data));
          float v277_data = s1[61];
          float v279_data = ir1[7];
          ir1[7] = (v279_data + (v241_data * v277_data));
          float v281_data = r0[6];
          float v282_data = s1[6];
          float v284_data = ir1[0];
          ir1[0] = (v284_data + (v281_data * v282_data));
          float v287_data = s1[14];
          float v289_data = ir1[1];
          ir1[1] = (v289_data + (v281_data * v287_data));
          float v292_data = s1[22];
          float v294_data = ir1[2];
          ir1[2] = (v294_data + (v281_data * v292_data));
          float v297_data = s1[30];
          float v299_data = ir1[3];
          ir1[3] = (v299_data + (v281_data * v297_data));
          float v302_data = s1[38];
          float v304_data = ir1[4];
          ir1[4] = (v304_data + (v281_data * v302_data));
          float v307_data = s1[46];
          float v309_data = ir1[5];
          ir1[5] = (v309_data + (v281_data * v307_data));
          float v312_data = s1[54];
          float v314_data = ir1[6];
          ir1[6] = (v314_data + (v281_data * v312_data));
          float v317_data = s1[62];
          float v319_data = ir1[7];
          ir1[7] = (v319_data + (v281_data * v317_data));
          float v321_data = r0[7];
          float v322_data = s1[7];
          float v324_data = ir1[0];
          ir1[0] = (v324_data + (v321_data * v322_data));
          float v327_data = s1[15];
          float v329_data = ir1[1];
          ir1[1] = (v329_data + (v321_data * v327_data));
          float v332_data = s1[23];
          float v334_data = ir1[2];
          ir1[2] = (v334_data + (v321_data * v332_data));
          float v337_data = s1[31];
          float v339_data = ir1[3];
          ir1[3] = (v339_data + (v321_data * v337_data));
          float v342_data = s1[39];
          float v344_data = ir1[4];
          ir1[4] = (v344_data + (v321_data * v342_data));
          float v347_data = s1[47];
          float v349_data = ir1[5];
          ir1[5] = (v349_data + (v321_data * v347_data));
          float v352_data = s1[55];
          float v354_data = ir1[6];
          ir1[6] = (v354_data + (v321_data * v352_data));
          float v357_data = s1[63];
          float v359_data = ir1[7];
          ir1[7] = (v359_data + (v321_data * v357_data));
          // r1 = ir1
          #pragma unroll
          for (int32_t v364_n0 = 0; v364_n0 < 1; ++v364_n0) {
            #pragma unroll
            for (int32_t v365_n1 = 0; v365_n1 < 8; ++v365_n1) {
              int32_t v366_a = v364_n0 + v365_n1;
              float v367_data = ir1[v366_a];
              r1[v366_a] = v367_data;
            }
          }
          // glb_m1 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v371_i0 = 0; v371_i0 < 1; ++v371_i0) {
            int32_t v376_lead = v26_lead + (v371_i0 * 8);
            #pragma unroll
            for (int32_t v372_i1 = 0; v372_i1 < 8; ++v372_i1) {
              float v374_data = r1[(v371_i0 + v372_i1)];
              glb_m1[(v376_lead + (v372_i1 * 8))] = v374_data;
            }
          }
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
        }
      }
    }
  }
}

