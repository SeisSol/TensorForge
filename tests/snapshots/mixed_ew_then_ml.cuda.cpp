// === base name ===
kernel_82c801327702da69

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_82c801327702da69 = {{8, 16, 1}, 8, 8, 1, 16, 4608, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_82c801327702da69(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_82c801327702da69(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_82c801327702da69(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_82c801327702da69, block.x * block.y * block.z, 1152 * sizeof(float));
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
void launcher_kernel_82c801327702da69(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_82c801327702da69(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_82c801327702da69, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_82c801327702da69<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_82c801327702da69(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 64 + 0 + m0_extraOffset];
          float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 64 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 64 + 0 + m2_extraOffset];
          // s1 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 8; i += 1) {
            __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + i * 8], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 8], 4);
          }
          __pipeline_commit();
          float r0[8]{};
          // r0 = abs(glb_m0)
          int32_t v22_lead = threadIdx.x % 8;
          #pragma unroll
          for (int32_t v23_k0 = 0; v23_k0 < 1; ++v23_k0) {
            int32_t v26_lead = v22_lead + (v23_k0 * 8);
            #pragma unroll
            for (int32_t v24_k1 = 0; v24_k1 < 8; ++v24_k1) {
              float v29_data = glb_m0[(v26_lead + (v24_k1 * 8))];
              r0[(v23_k0 + v24_k1)] = (fabsf(v29_data));
            }
          }
          // wait(s1 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          // ir1 = +(r0 * s1)
          // [(0, 8), (0, 8)] [(0, 8)]
          float ir1[8]{};
          float v38_data = r0[0];
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          float v39_data = s1[0];
          float v41_data = ir1[0];
          ir1[0] = (v41_data + (v38_data * v39_data));
          float v44_data = s1[8];
          float v46_data = ir1[1];
          ir1[1] = (v46_data + (v38_data * v44_data));
          float v49_data = s1[16];
          float v51_data = ir1[2];
          ir1[2] = (v51_data + (v38_data * v49_data));
          float v54_data = s1[24];
          float v56_data = ir1[3];
          ir1[3] = (v56_data + (v38_data * v54_data));
          float v59_data = s1[32];
          float v61_data = ir1[4];
          ir1[4] = (v61_data + (v38_data * v59_data));
          float v64_data = s1[40];
          float v66_data = ir1[5];
          ir1[5] = (v66_data + (v38_data * v64_data));
          float v69_data = s1[48];
          float v71_data = ir1[6];
          ir1[6] = (v71_data + (v38_data * v69_data));
          float v74_data = s1[56];
          float v76_data = ir1[7];
          ir1[7] = (v76_data + (v38_data * v74_data));
          float v78_data = r0[1];
          float v79_data = s1[1];
          float v81_data = ir1[0];
          ir1[0] = (v81_data + (v78_data * v79_data));
          float v84_data = s1[9];
          float v86_data = ir1[1];
          ir1[1] = (v86_data + (v78_data * v84_data));
          float v89_data = s1[17];
          float v91_data = ir1[2];
          ir1[2] = (v91_data + (v78_data * v89_data));
          float v94_data = s1[25];
          float v96_data = ir1[3];
          ir1[3] = (v96_data + (v78_data * v94_data));
          float v99_data = s1[33];
          float v101_data = ir1[4];
          ir1[4] = (v101_data + (v78_data * v99_data));
          float v104_data = s1[41];
          float v106_data = ir1[5];
          ir1[5] = (v106_data + (v78_data * v104_data));
          float v109_data = s1[49];
          float v111_data = ir1[6];
          ir1[6] = (v111_data + (v78_data * v109_data));
          float v114_data = s1[57];
          float v116_data = ir1[7];
          ir1[7] = (v116_data + (v78_data * v114_data));
          float v118_data = r0[2];
          float v119_data = s1[2];
          float v121_data = ir1[0];
          ir1[0] = (v121_data + (v118_data * v119_data));
          float v124_data = s1[10];
          float v126_data = ir1[1];
          ir1[1] = (v126_data + (v118_data * v124_data));
          float v129_data = s1[18];
          float v131_data = ir1[2];
          ir1[2] = (v131_data + (v118_data * v129_data));
          float v134_data = s1[26];
          float v136_data = ir1[3];
          ir1[3] = (v136_data + (v118_data * v134_data));
          float v139_data = s1[34];
          float v141_data = ir1[4];
          ir1[4] = (v141_data + (v118_data * v139_data));
          float v144_data = s1[42];
          float v146_data = ir1[5];
          ir1[5] = (v146_data + (v118_data * v144_data));
          float v149_data = s1[50];
          float v151_data = ir1[6];
          ir1[6] = (v151_data + (v118_data * v149_data));
          float v154_data = s1[58];
          float v156_data = ir1[7];
          ir1[7] = (v156_data + (v118_data * v154_data));
          float v158_data = r0[3];
          float v159_data = s1[3];
          float v161_data = ir1[0];
          ir1[0] = (v161_data + (v158_data * v159_data));
          float v164_data = s1[11];
          float v166_data = ir1[1];
          ir1[1] = (v166_data + (v158_data * v164_data));
          float v169_data = s1[19];
          float v171_data = ir1[2];
          ir1[2] = (v171_data + (v158_data * v169_data));
          float v174_data = s1[27];
          float v176_data = ir1[3];
          ir1[3] = (v176_data + (v158_data * v174_data));
          float v179_data = s1[35];
          float v181_data = ir1[4];
          ir1[4] = (v181_data + (v158_data * v179_data));
          float v184_data = s1[43];
          float v186_data = ir1[5];
          ir1[5] = (v186_data + (v158_data * v184_data));
          float v189_data = s1[51];
          float v191_data = ir1[6];
          ir1[6] = (v191_data + (v158_data * v189_data));
          float v194_data = s1[59];
          float v196_data = ir1[7];
          ir1[7] = (v196_data + (v158_data * v194_data));
          float v198_data = r0[4];
          float v199_data = s1[4];
          float v201_data = ir1[0];
          ir1[0] = (v201_data + (v198_data * v199_data));
          float v204_data = s1[12];
          float v206_data = ir1[1];
          ir1[1] = (v206_data + (v198_data * v204_data));
          float v209_data = s1[20];
          float v211_data = ir1[2];
          ir1[2] = (v211_data + (v198_data * v209_data));
          float v214_data = s1[28];
          float v216_data = ir1[3];
          ir1[3] = (v216_data + (v198_data * v214_data));
          float v219_data = s1[36];
          float v221_data = ir1[4];
          ir1[4] = (v221_data + (v198_data * v219_data));
          float v224_data = s1[44];
          float v226_data = ir1[5];
          ir1[5] = (v226_data + (v198_data * v224_data));
          float v229_data = s1[52];
          float v231_data = ir1[6];
          ir1[6] = (v231_data + (v198_data * v229_data));
          float v234_data = s1[60];
          float v236_data = ir1[7];
          ir1[7] = (v236_data + (v198_data * v234_data));
          float v238_data = r0[5];
          float v239_data = s1[5];
          float v241_data = ir1[0];
          ir1[0] = (v241_data + (v238_data * v239_data));
          float v244_data = s1[13];
          float v246_data = ir1[1];
          ir1[1] = (v246_data + (v238_data * v244_data));
          float v249_data = s1[21];
          float v251_data = ir1[2];
          ir1[2] = (v251_data + (v238_data * v249_data));
          float v254_data = s1[29];
          float v256_data = ir1[3];
          ir1[3] = (v256_data + (v238_data * v254_data));
          float v259_data = s1[37];
          float v261_data = ir1[4];
          ir1[4] = (v261_data + (v238_data * v259_data));
          float v264_data = s1[45];
          float v266_data = ir1[5];
          ir1[5] = (v266_data + (v238_data * v264_data));
          float v269_data = s1[53];
          float v271_data = ir1[6];
          ir1[6] = (v271_data + (v238_data * v269_data));
          float v274_data = s1[61];
          float v276_data = ir1[7];
          ir1[7] = (v276_data + (v238_data * v274_data));
          float v278_data = r0[6];
          float v279_data = s1[6];
          float v281_data = ir1[0];
          ir1[0] = (v281_data + (v278_data * v279_data));
          float v284_data = s1[14];
          float v286_data = ir1[1];
          ir1[1] = (v286_data + (v278_data * v284_data));
          float v289_data = s1[22];
          float v291_data = ir1[2];
          ir1[2] = (v291_data + (v278_data * v289_data));
          float v294_data = s1[30];
          float v296_data = ir1[3];
          ir1[3] = (v296_data + (v278_data * v294_data));
          float v299_data = s1[38];
          float v301_data = ir1[4];
          ir1[4] = (v301_data + (v278_data * v299_data));
          float v304_data = s1[46];
          float v306_data = ir1[5];
          ir1[5] = (v306_data + (v278_data * v304_data));
          float v309_data = s1[54];
          float v311_data = ir1[6];
          ir1[6] = (v311_data + (v278_data * v309_data));
          float v314_data = s1[62];
          float v316_data = ir1[7];
          ir1[7] = (v316_data + (v278_data * v314_data));
          float v318_data = r0[7];
          float v319_data = s1[7];
          float v321_data = ir1[0];
          ir1[0] = (v321_data + (v318_data * v319_data));
          float v324_data = s1[15];
          float v326_data = ir1[1];
          ir1[1] = (v326_data + (v318_data * v324_data));
          float v329_data = s1[23];
          float v331_data = ir1[2];
          ir1[2] = (v331_data + (v318_data * v329_data));
          float v334_data = s1[31];
          float v336_data = ir1[3];
          ir1[3] = (v336_data + (v318_data * v334_data));
          float v339_data = s1[39];
          float v341_data = ir1[4];
          ir1[4] = (v341_data + (v318_data * v339_data));
          float v344_data = s1[47];
          float v346_data = ir1[5];
          ir1[5] = (v346_data + (v318_data * v344_data));
          float v349_data = s1[55];
          float v351_data = ir1[6];
          ir1[6] = (v351_data + (v318_data * v349_data));
          float v354_data = s1[63];
          float v356_data = ir1[7];
          ir1[7] = (v356_data + (v318_data * v354_data));
          // r1 = ir1
          #pragma unroll
          for (int32_t v361_n0 = 0; v361_n0 < 1; ++v361_n0) {
            #pragma unroll
            for (int32_t v362_n1 = 0; v362_n1 < 8; ++v362_n1) {
              int32_t v363_a = v361_n0 + v362_n1;
              float v364_data = ir1[v363_a];
              r1[v363_a] = v364_data;
            }
          }
          // glb_m1 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v368_i0 = 0; v368_i0 < 1; ++v368_i0) {
            int32_t v373_lead = v22_lead + (v368_i0 * 8);
            #pragma unroll
            for (int32_t v369_i1 = 0; v369_i1 < 8; ++v369_i1) {
              float v371_data = r1[(v368_i0 + v369_i1)];
              glb_m1[(v373_lead + (v369_i1 * 8))] = v371_data;
            }
          }
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
        }
      }
    }
  }
}

