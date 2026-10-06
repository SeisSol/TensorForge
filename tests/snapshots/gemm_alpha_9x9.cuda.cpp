// === base name ===
kernel_c15c48ca6dcef532

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_c15c48ca6dcef532 = {{16, 8, 1}, 16, 9, 1, 8, 3584, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_c15c48ca6dcef532(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_c15c48ca6dcef532(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_c15c48ca6dcef532(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_c15c48ca6dcef532, block.x * block.y * block.z, 896 * sizeof(float));
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
  config.block[0] = 16;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 896 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_c15c48ca6dcef532(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_c15c48ca6dcef532(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_c15c48ca6dcef532, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_c15c48ca6dcef532<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_c15c48ca6dcef532(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (9 active) x 8 per block = block 16x8x1, 3584 B shared, occupancy grid
    // operands:
    //   m0 9×9(9×9) {0..9}×{0..9} strided
    //   m1 9×9(9×9) {0..9}×{0..9} strided
    //   m2 9×9(9×9) {0..9}×{0..9} strided
    //   m3 ()  scalar
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j] × m3[]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":9,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":896}],"shared_bytes":3584,"shared_elements":896,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[9,9]],"name":"m0","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[9,9]],"name":"m1","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[9,9]],"name":"m2","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"scalar","alias":null,"bbox":[[],[]],"name":"m3","ordered":false,"parts":1,"shape":[],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[9,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[9,9]},{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[9,9]},{"addressing":"scalar","bbox":[[],[]],"is_tmp":false,"name":"m3","offset":[],"shape":[]}],"permute":[[0,1],[0,1],[]],"target":[[0,-1],[-1,1],[]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[112 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[96];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v11_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v11_batchId0 < numElements0; v11_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v12_ahead1 = v11_batchId0 + (gridDim.x * blockDim.y);
        size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v11_batchId0 * 81 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v11_batchId0 * 81 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v11_batchId0 * 81 + 0 + m2_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v25_lead = threadIdx.x % 16;
          bool v26_g = v25_lead < 9;
          if (v26_g) {
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 9; ++v27_i1) {
              float v32_data = __ldcg(&glb_m1[(v25_lead + (v27_i1 * 9))]);
              r0[v27_i1] = v32_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 5; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
          if (threadIdx.x < 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 80], &glb_m2[0 + 0 + 1 * threadIdx.x + 80], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[9]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // ir1 = +(r0 * s0)
          // [(0, 9), (0, 9)] [(0, 9)]
          float ir1[9]{};
          float v38_data = r0[0];
          float v39_data = s0[0];
          float v41_data = ir1[0];
          ir1[0] = (v41_data + (v38_data * v39_data));
          float v44_data = s0[9];
          float v46_data = ir1[1];
          ir1[1] = (v46_data + (v38_data * v44_data));
          float v49_data = s0[18];
          float v51_data = ir1[2];
          ir1[2] = (v51_data + (v38_data * v49_data));
          float v54_data = s0[27];
          float v56_data = ir1[3];
          ir1[3] = (v56_data + (v38_data * v54_data));
          float v59_data = s0[36];
          float v61_data = ir1[4];
          ir1[4] = (v61_data + (v38_data * v59_data));
          float v64_data = s0[45];
          float v66_data = ir1[5];
          ir1[5] = (v66_data + (v38_data * v64_data));
          float v69_data = s0[54];
          float v71_data = ir1[6];
          ir1[6] = (v71_data + (v38_data * v69_data));
          float v74_data = s0[63];
          float v76_data = ir1[7];
          ir1[7] = (v76_data + (v38_data * v74_data));
          float v79_data = s0[72];
          float v81_data = ir1[8];
          ir1[8] = (v81_data + (v38_data * v79_data));
          float v83_data = r0[1];
          float v84_data = s0[1];
          float v86_data = ir1[0];
          ir1[0] = (v86_data + (v83_data * v84_data));
          float v89_data = s0[10];
          float v91_data = ir1[1];
          ir1[1] = (v91_data + (v83_data * v89_data));
          float v94_data = s0[19];
          float v96_data = ir1[2];
          ir1[2] = (v96_data + (v83_data * v94_data));
          float v99_data = s0[28];
          float v101_data = ir1[3];
          ir1[3] = (v101_data + (v83_data * v99_data));
          float v104_data = s0[37];
          float v106_data = ir1[4];
          ir1[4] = (v106_data + (v83_data * v104_data));
          float v109_data = s0[46];
          float v111_data = ir1[5];
          ir1[5] = (v111_data + (v83_data * v109_data));
          float v114_data = s0[55];
          float v116_data = ir1[6];
          ir1[6] = (v116_data + (v83_data * v114_data));
          float v119_data = s0[64];
          float v121_data = ir1[7];
          ir1[7] = (v121_data + (v83_data * v119_data));
          float v124_data = s0[73];
          float v126_data = ir1[8];
          ir1[8] = (v126_data + (v83_data * v124_data));
          float v128_data = r0[2];
          float v129_data = s0[2];
          float v131_data = ir1[0];
          ir1[0] = (v131_data + (v128_data * v129_data));
          float v134_data = s0[11];
          float v136_data = ir1[1];
          ir1[1] = (v136_data + (v128_data * v134_data));
          float v139_data = s0[20];
          float v141_data = ir1[2];
          ir1[2] = (v141_data + (v128_data * v139_data));
          float v144_data = s0[29];
          float v146_data = ir1[3];
          ir1[3] = (v146_data + (v128_data * v144_data));
          float v149_data = s0[38];
          float v151_data = ir1[4];
          ir1[4] = (v151_data + (v128_data * v149_data));
          float v154_data = s0[47];
          float v156_data = ir1[5];
          ir1[5] = (v156_data + (v128_data * v154_data));
          float v159_data = s0[56];
          float v161_data = ir1[6];
          ir1[6] = (v161_data + (v128_data * v159_data));
          float v164_data = s0[65];
          float v166_data = ir1[7];
          ir1[7] = (v166_data + (v128_data * v164_data));
          float v169_data = s0[74];
          float v171_data = ir1[8];
          ir1[8] = (v171_data + (v128_data * v169_data));
          float v173_data = r0[3];
          float v174_data = s0[3];
          float v176_data = ir1[0];
          ir1[0] = (v176_data + (v173_data * v174_data));
          float v179_data = s0[12];
          float v181_data = ir1[1];
          ir1[1] = (v181_data + (v173_data * v179_data));
          float v184_data = s0[21];
          float v186_data = ir1[2];
          ir1[2] = (v186_data + (v173_data * v184_data));
          float v189_data = s0[30];
          float v191_data = ir1[3];
          ir1[3] = (v191_data + (v173_data * v189_data));
          float v194_data = s0[39];
          float v196_data = ir1[4];
          ir1[4] = (v196_data + (v173_data * v194_data));
          float v199_data = s0[48];
          float v201_data = ir1[5];
          ir1[5] = (v201_data + (v173_data * v199_data));
          float v204_data = s0[57];
          float v206_data = ir1[6];
          ir1[6] = (v206_data + (v173_data * v204_data));
          float v209_data = s0[66];
          float v211_data = ir1[7];
          ir1[7] = (v211_data + (v173_data * v209_data));
          float v214_data = s0[75];
          float v216_data = ir1[8];
          ir1[8] = (v216_data + (v173_data * v214_data));
          float v218_data = r0[4];
          float v219_data = s0[4];
          float v221_data = ir1[0];
          ir1[0] = (v221_data + (v218_data * v219_data));
          float v224_data = s0[13];
          float v226_data = ir1[1];
          ir1[1] = (v226_data + (v218_data * v224_data));
          float v229_data = s0[22];
          float v231_data = ir1[2];
          ir1[2] = (v231_data + (v218_data * v229_data));
          float v234_data = s0[31];
          float v236_data = ir1[3];
          ir1[3] = (v236_data + (v218_data * v234_data));
          float v239_data = s0[40];
          float v241_data = ir1[4];
          ir1[4] = (v241_data + (v218_data * v239_data));
          float v244_data = s0[49];
          float v246_data = ir1[5];
          ir1[5] = (v246_data + (v218_data * v244_data));
          float v249_data = s0[58];
          float v251_data = ir1[6];
          ir1[6] = (v251_data + (v218_data * v249_data));
          float v254_data = s0[67];
          float v256_data = ir1[7];
          ir1[7] = (v256_data + (v218_data * v254_data));
          float v259_data = s0[76];
          float v261_data = ir1[8];
          ir1[8] = (v261_data + (v218_data * v259_data));
          float v263_data = r0[5];
          float v264_data = s0[5];
          float v266_data = ir1[0];
          ir1[0] = (v266_data + (v263_data * v264_data));
          float v269_data = s0[14];
          float v271_data = ir1[1];
          ir1[1] = (v271_data + (v263_data * v269_data));
          float v274_data = s0[23];
          float v276_data = ir1[2];
          ir1[2] = (v276_data + (v263_data * v274_data));
          float v279_data = s0[32];
          float v281_data = ir1[3];
          ir1[3] = (v281_data + (v263_data * v279_data));
          float v284_data = s0[41];
          float v286_data = ir1[4];
          ir1[4] = (v286_data + (v263_data * v284_data));
          float v289_data = s0[50];
          float v291_data = ir1[5];
          ir1[5] = (v291_data + (v263_data * v289_data));
          float v294_data = s0[59];
          float v296_data = ir1[6];
          ir1[6] = (v296_data + (v263_data * v294_data));
          float v299_data = s0[68];
          float v301_data = ir1[7];
          ir1[7] = (v301_data + (v263_data * v299_data));
          float v304_data = s0[77];
          float v306_data = ir1[8];
          ir1[8] = (v306_data + (v263_data * v304_data));
          float v308_data = r0[6];
          float v309_data = s0[6];
          float v311_data = ir1[0];
          ir1[0] = (v311_data + (v308_data * v309_data));
          float v314_data = s0[15];
          float v316_data = ir1[1];
          ir1[1] = (v316_data + (v308_data * v314_data));
          float v319_data = s0[24];
          float v321_data = ir1[2];
          ir1[2] = (v321_data + (v308_data * v319_data));
          float v324_data = s0[33];
          float v326_data = ir1[3];
          ir1[3] = (v326_data + (v308_data * v324_data));
          float v329_data = s0[42];
          float v331_data = ir1[4];
          ir1[4] = (v331_data + (v308_data * v329_data));
          float v334_data = s0[51];
          float v336_data = ir1[5];
          ir1[5] = (v336_data + (v308_data * v334_data));
          float v339_data = s0[60];
          float v341_data = ir1[6];
          ir1[6] = (v341_data + (v308_data * v339_data));
          float v344_data = s0[69];
          float v346_data = ir1[7];
          ir1[7] = (v346_data + (v308_data * v344_data));
          float v349_data = s0[78];
          float v351_data = ir1[8];
          ir1[8] = (v351_data + (v308_data * v349_data));
          float v353_data = r0[7];
          float v354_data = s0[7];
          float v356_data = ir1[0];
          ir1[0] = (v356_data + (v353_data * v354_data));
          float v359_data = s0[16];
          float v361_data = ir1[1];
          ir1[1] = (v361_data + (v353_data * v359_data));
          float v364_data = s0[25];
          float v366_data = ir1[2];
          ir1[2] = (v366_data + (v353_data * v364_data));
          float v369_data = s0[34];
          float v371_data = ir1[3];
          ir1[3] = (v371_data + (v353_data * v369_data));
          float v374_data = s0[43];
          float v376_data = ir1[4];
          ir1[4] = (v376_data + (v353_data * v374_data));
          float v379_data = s0[52];
          float v381_data = ir1[5];
          ir1[5] = (v381_data + (v353_data * v379_data));
          float v384_data = s0[61];
          float v386_data = ir1[6];
          ir1[6] = (v386_data + (v353_data * v384_data));
          float v389_data = s0[70];
          float v391_data = ir1[7];
          ir1[7] = (v391_data + (v353_data * v389_data));
          float v394_data = s0[79];
          float v396_data = ir1[8];
          ir1[8] = (v396_data + (v353_data * v394_data));
          float v398_data = r0[8];
          float v399_data = s0[8];
          float v401_data = ir1[0];
          ir1[0] = (v401_data + (v398_data * v399_data));
          float v404_data = s0[17];
          float v406_data = ir1[1];
          ir1[1] = (v406_data + (v398_data * v404_data));
          float v409_data = s0[26];
          float v411_data = ir1[2];
          ir1[2] = (v411_data + (v398_data * v409_data));
          float v414_data = s0[35];
          float v416_data = ir1[3];
          ir1[3] = (v416_data + (v398_data * v414_data));
          float v419_data = s0[44];
          float v421_data = ir1[4];
          ir1[4] = (v421_data + (v398_data * v419_data));
          float v424_data = s0[53];
          float v426_data = ir1[5];
          ir1[5] = (v426_data + (v398_data * v424_data));
          float v429_data = s0[62];
          float v431_data = ir1[6];
          ir1[6] = (v431_data + (v398_data * v429_data));
          float v434_data = s0[71];
          float v436_data = ir1[7];
          ir1[7] = (v436_data + (v398_data * v434_data));
          float v439_data = s0[80];
          float v441_data = ir1[8];
          ir1[8] = (v441_data + (v398_data * v439_data));
          // r1 = ir1 * glb_m3
          if (v26_g) {
            #pragma unroll
            for (int32_t v444_n1 = 0; v444_n1 < 9; ++v444_n1) {
              float v446_data = ir1[v444_n1];
              r1[v444_n1] = (v446_data * 13.0f);
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v26_g) {
            #pragma unroll
            for (int32_t v448_i1 = 0; v448_i1 < 9; ++v448_i1) {
              float v450_data = r1[v448_i1];
              glb_m0[(v25_lead + (v448_i1 * 9))] = v450_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

