// === base name ===
kernel_f85c7174df64e83a

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_f85c7174df64e83a = {{8, 16, 1}, 8, 8, 1, 16, 4608, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_f85c7174df64e83a(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_f85c7174df64e83a(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_f85c7174df64e83a(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_f85c7174df64e83a, block.x * block.y * block.z, 1152 * sizeof(float));
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
void launcher_kernel_f85c7174df64e83a(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_f85c7174df64e83a(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_f85c7174df64e83a, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_f85c7174df64e83a<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_f85c7174df64e83a(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
    //   t0[i,j] = m0[i,k] × m1[k,j]
    //   C = abs(TMP)
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1152}],"shared_bytes":4608,"shared_elements":1152,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[72 * threadIdx.y + 0];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 64 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 64 + 0 + m1_extraOffset];
          float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 64 + 0 + m2_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v22_lead = threadIdx.x % 8;
          #pragma unroll
          for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
            int32_t v26_lead = v22_lead + (v23_i0 * 8);
            #pragma unroll
            for (int32_t v24_i1 = 0; v24_i1 < 8; ++v24_i1) {
              float v29_data = __ldcg(&glb_m0[(v26_lead + (v24_i1 * 8))]);
              r0[(v23_i0 + v24_i1)] = v29_data;
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
          // r1 = +(r0 * s0) + None
          // [(0, 8), (0, 8)] [(0, 8)]
          float v33_data = r0[0];
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          float v34_data = s0[0];
          float v36_data = r1[0];
          r1[0] = (v36_data + (v33_data * v34_data));
          float v39_data = s0[8];
          float v41_data = r1[1];
          r1[1] = (v41_data + (v33_data * v39_data));
          float v44_data = s0[16];
          float v46_data = r1[2];
          r1[2] = (v46_data + (v33_data * v44_data));
          float v49_data = s0[24];
          float v51_data = r1[3];
          r1[3] = (v51_data + (v33_data * v49_data));
          float v54_data = s0[32];
          float v56_data = r1[4];
          r1[4] = (v56_data + (v33_data * v54_data));
          float v59_data = s0[40];
          float v61_data = r1[5];
          r1[5] = (v61_data + (v33_data * v59_data));
          float v64_data = s0[48];
          float v66_data = r1[6];
          r1[6] = (v66_data + (v33_data * v64_data));
          float v69_data = s0[56];
          float v71_data = r1[7];
          r1[7] = (v71_data + (v33_data * v69_data));
          float v73_data = r0[1];
          float v74_data = s0[1];
          float v76_data = r1[0];
          r1[0] = (v76_data + (v73_data * v74_data));
          float v79_data = s0[9];
          float v81_data = r1[1];
          r1[1] = (v81_data + (v73_data * v79_data));
          float v84_data = s0[17];
          float v86_data = r1[2];
          r1[2] = (v86_data + (v73_data * v84_data));
          float v89_data = s0[25];
          float v91_data = r1[3];
          r1[3] = (v91_data + (v73_data * v89_data));
          float v94_data = s0[33];
          float v96_data = r1[4];
          r1[4] = (v96_data + (v73_data * v94_data));
          float v99_data = s0[41];
          float v101_data = r1[5];
          r1[5] = (v101_data + (v73_data * v99_data));
          float v104_data = s0[49];
          float v106_data = r1[6];
          r1[6] = (v106_data + (v73_data * v104_data));
          float v109_data = s0[57];
          float v111_data = r1[7];
          r1[7] = (v111_data + (v73_data * v109_data));
          float v113_data = r0[2];
          float v114_data = s0[2];
          float v116_data = r1[0];
          r1[0] = (v116_data + (v113_data * v114_data));
          float v119_data = s0[10];
          float v121_data = r1[1];
          r1[1] = (v121_data + (v113_data * v119_data));
          float v124_data = s0[18];
          float v126_data = r1[2];
          r1[2] = (v126_data + (v113_data * v124_data));
          float v129_data = s0[26];
          float v131_data = r1[3];
          r1[3] = (v131_data + (v113_data * v129_data));
          float v134_data = s0[34];
          float v136_data = r1[4];
          r1[4] = (v136_data + (v113_data * v134_data));
          float v139_data = s0[42];
          float v141_data = r1[5];
          r1[5] = (v141_data + (v113_data * v139_data));
          float v144_data = s0[50];
          float v146_data = r1[6];
          r1[6] = (v146_data + (v113_data * v144_data));
          float v149_data = s0[58];
          float v151_data = r1[7];
          r1[7] = (v151_data + (v113_data * v149_data));
          float v153_data = r0[3];
          float v154_data = s0[3];
          float v156_data = r1[0];
          r1[0] = (v156_data + (v153_data * v154_data));
          float v159_data = s0[11];
          float v161_data = r1[1];
          r1[1] = (v161_data + (v153_data * v159_data));
          float v164_data = s0[19];
          float v166_data = r1[2];
          r1[2] = (v166_data + (v153_data * v164_data));
          float v169_data = s0[27];
          float v171_data = r1[3];
          r1[3] = (v171_data + (v153_data * v169_data));
          float v174_data = s0[35];
          float v176_data = r1[4];
          r1[4] = (v176_data + (v153_data * v174_data));
          float v179_data = s0[43];
          float v181_data = r1[5];
          r1[5] = (v181_data + (v153_data * v179_data));
          float v184_data = s0[51];
          float v186_data = r1[6];
          r1[6] = (v186_data + (v153_data * v184_data));
          float v189_data = s0[59];
          float v191_data = r1[7];
          r1[7] = (v191_data + (v153_data * v189_data));
          float v193_data = r0[4];
          float v194_data = s0[4];
          float v196_data = r1[0];
          r1[0] = (v196_data + (v193_data * v194_data));
          float v199_data = s0[12];
          float v201_data = r1[1];
          r1[1] = (v201_data + (v193_data * v199_data));
          float v204_data = s0[20];
          float v206_data = r1[2];
          r1[2] = (v206_data + (v193_data * v204_data));
          float v209_data = s0[28];
          float v211_data = r1[3];
          r1[3] = (v211_data + (v193_data * v209_data));
          float v214_data = s0[36];
          float v216_data = r1[4];
          r1[4] = (v216_data + (v193_data * v214_data));
          float v219_data = s0[44];
          float v221_data = r1[5];
          r1[5] = (v221_data + (v193_data * v219_data));
          float v224_data = s0[52];
          float v226_data = r1[6];
          r1[6] = (v226_data + (v193_data * v224_data));
          float v229_data = s0[60];
          float v231_data = r1[7];
          r1[7] = (v231_data + (v193_data * v229_data));
          float v233_data = r0[5];
          float v234_data = s0[5];
          float v236_data = r1[0];
          r1[0] = (v236_data + (v233_data * v234_data));
          float v239_data = s0[13];
          float v241_data = r1[1];
          r1[1] = (v241_data + (v233_data * v239_data));
          float v244_data = s0[21];
          float v246_data = r1[2];
          r1[2] = (v246_data + (v233_data * v244_data));
          float v249_data = s0[29];
          float v251_data = r1[3];
          r1[3] = (v251_data + (v233_data * v249_data));
          float v254_data = s0[37];
          float v256_data = r1[4];
          r1[4] = (v256_data + (v233_data * v254_data));
          float v259_data = s0[45];
          float v261_data = r1[5];
          r1[5] = (v261_data + (v233_data * v259_data));
          float v264_data = s0[53];
          float v266_data = r1[6];
          r1[6] = (v266_data + (v233_data * v264_data));
          float v269_data = s0[61];
          float v271_data = r1[7];
          r1[7] = (v271_data + (v233_data * v269_data));
          float v273_data = r0[6];
          float v274_data = s0[6];
          float v276_data = r1[0];
          r1[0] = (v276_data + (v273_data * v274_data));
          float v279_data = s0[14];
          float v281_data = r1[1];
          r1[1] = (v281_data + (v273_data * v279_data));
          float v284_data = s0[22];
          float v286_data = r1[2];
          r1[2] = (v286_data + (v273_data * v284_data));
          float v289_data = s0[30];
          float v291_data = r1[3];
          r1[3] = (v291_data + (v273_data * v289_data));
          float v294_data = s0[38];
          float v296_data = r1[4];
          r1[4] = (v296_data + (v273_data * v294_data));
          float v299_data = s0[46];
          float v301_data = r1[5];
          r1[5] = (v301_data + (v273_data * v299_data));
          float v304_data = s0[54];
          float v306_data = r1[6];
          r1[6] = (v306_data + (v273_data * v304_data));
          float v309_data = s0[62];
          float v311_data = r1[7];
          r1[7] = (v311_data + (v273_data * v309_data));
          float v313_data = r0[7];
          float v314_data = s0[7];
          float v316_data = r1[0];
          r1[0] = (v316_data + (v313_data * v314_data));
          float v319_data = s0[15];
          float v321_data = r1[1];
          r1[1] = (v321_data + (v313_data * v319_data));
          float v324_data = s0[23];
          float v326_data = r1[2];
          r1[2] = (v326_data + (v313_data * v324_data));
          float v329_data = s0[31];
          float v331_data = r1[3];
          r1[3] = (v331_data + (v313_data * v329_data));
          float v334_data = s0[39];
          float v336_data = r1[4];
          r1[4] = (v336_data + (v313_data * v334_data));
          float v339_data = s0[47];
          float v341_data = r1[5];
          r1[5] = (v341_data + (v313_data * v339_data));
          float v344_data = s0[55];
          float v346_data = r1[6];
          r1[6] = (v346_data + (v313_data * v344_data));
          float v349_data = s0[63];
          float v351_data = r1[7];
          r1[7] = (v351_data + (v313_data * v349_data));
          // glb_m2 = abs(r1)
          #pragma unroll
          for (int32_t v353_k0 = 0; v353_k0 < 1; ++v353_k0) {
            int32_t v359_lead = v22_lead + (v353_k0 * 8);
            #pragma unroll
            for (int32_t v354_k1 = 0; v354_k1 < 8; ++v354_k1) {
              float v356_data = r1[(v353_k0 + v354_k1)];
              glb_m2[(v359_lead + (v354_k1 * 8))] = (fabsf(v356_data));
            }
          }
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
        }
      }
    }
  }
}

