// === base name ===
kernel_f09405ee8a6b1d34

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_f09405ee8a6b1d34 = {{32, 4, 1}, 32, 32, 1, 4, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_f09405ee8a6b1d34(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_f09405ee8a6b1d34(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_f09405ee8a6b1d34(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_f09405ee8a6b1d34, block.x * block.y * block.z, 256 * sizeof(float));
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
void launcher_kernel_f09405ee8a6b1d34(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_f09405ee8a6b1d34(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_f09405ee8a6b1d34, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_f09405ee8a6b1d34<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_f09405ee8a6b1d34(const float * m0, size_t m0_extraOffset, float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 4 per block = block 32x4x1, 1024 B shared, occupancy grid
    // operands:
    //   m0 8×8(8×8) {0..8}×{0..8} strided
    //   m1 8×8(8×8) {0..8}×{0..8} strided
    //   m2 8×8(8×8) {0..8}×{0..8} strided
    // operations:
    //   TMP = abs(A)
    //   m1[i,j] = t0[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[64 * threadIdx.y + 0];
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
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 32], &glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
          __pipeline_commit();
          float r0[8]{};
          // r0 = abs(glb_m0)
          int32_t v21_lead = threadIdx.x % 32;
          if (v21_lead < 8) {
            #pragma unroll
            for (int32_t v23_k1 = 0; v23_k1 < 8; ++v23_k1) {
              float v28_data = glb_m0[(v21_lead + (v23_k1 * 8))];
              r0[v23_k1] = (fabsf(v28_data));
            }
          }
          // wait(s1 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s1) + None
          // [(0, 8), (0, 8)] [(0, 8)]
          float ir1[8]{};
          float v36_data = r0[0];
          float v37_data = s1[0];
          float v39_data = ir1[0];
          ir1[0] = (v39_data + (v36_data * v37_data));
          float v42_data = s1[8];
          float v44_data = ir1[1];
          ir1[1] = (v44_data + (v36_data * v42_data));
          float v47_data = s1[16];
          float v49_data = ir1[2];
          ir1[2] = (v49_data + (v36_data * v47_data));
          float v52_data = s1[24];
          float v54_data = ir1[3];
          ir1[3] = (v54_data + (v36_data * v52_data));
          float v57_data = s1[32];
          float v59_data = ir1[4];
          ir1[4] = (v59_data + (v36_data * v57_data));
          float v62_data = s1[40];
          float v64_data = ir1[5];
          ir1[5] = (v64_data + (v36_data * v62_data));
          float v67_data = s1[48];
          float v69_data = ir1[6];
          ir1[6] = (v69_data + (v36_data * v67_data));
          float v72_data = s1[56];
          float v74_data = ir1[7];
          ir1[7] = (v74_data + (v36_data * v72_data));
          float v76_data = r0[1];
          float v77_data = s1[1];
          float v79_data = ir1[0];
          ir1[0] = (v79_data + (v76_data * v77_data));
          float v82_data = s1[9];
          float v84_data = ir1[1];
          ir1[1] = (v84_data + (v76_data * v82_data));
          float v87_data = s1[17];
          float v89_data = ir1[2];
          ir1[2] = (v89_data + (v76_data * v87_data));
          float v92_data = s1[25];
          float v94_data = ir1[3];
          ir1[3] = (v94_data + (v76_data * v92_data));
          float v97_data = s1[33];
          float v99_data = ir1[4];
          ir1[4] = (v99_data + (v76_data * v97_data));
          float v102_data = s1[41];
          float v104_data = ir1[5];
          ir1[5] = (v104_data + (v76_data * v102_data));
          float v107_data = s1[49];
          float v109_data = ir1[6];
          ir1[6] = (v109_data + (v76_data * v107_data));
          float v112_data = s1[57];
          float v114_data = ir1[7];
          ir1[7] = (v114_data + (v76_data * v112_data));
          float v116_data = r0[2];
          float v117_data = s1[2];
          float v119_data = ir1[0];
          ir1[0] = (v119_data + (v116_data * v117_data));
          float v122_data = s1[10];
          float v124_data = ir1[1];
          ir1[1] = (v124_data + (v116_data * v122_data));
          float v127_data = s1[18];
          float v129_data = ir1[2];
          ir1[2] = (v129_data + (v116_data * v127_data));
          float v132_data = s1[26];
          float v134_data = ir1[3];
          ir1[3] = (v134_data + (v116_data * v132_data));
          float v137_data = s1[34];
          float v139_data = ir1[4];
          ir1[4] = (v139_data + (v116_data * v137_data));
          float v142_data = s1[42];
          float v144_data = ir1[5];
          ir1[5] = (v144_data + (v116_data * v142_data));
          float v147_data = s1[50];
          float v149_data = ir1[6];
          ir1[6] = (v149_data + (v116_data * v147_data));
          float v152_data = s1[58];
          float v154_data = ir1[7];
          ir1[7] = (v154_data + (v116_data * v152_data));
          float v156_data = r0[3];
          float v157_data = s1[3];
          float v159_data = ir1[0];
          ir1[0] = (v159_data + (v156_data * v157_data));
          float v162_data = s1[11];
          float v164_data = ir1[1];
          ir1[1] = (v164_data + (v156_data * v162_data));
          float v167_data = s1[19];
          float v169_data = ir1[2];
          ir1[2] = (v169_data + (v156_data * v167_data));
          float v172_data = s1[27];
          float v174_data = ir1[3];
          ir1[3] = (v174_data + (v156_data * v172_data));
          float v177_data = s1[35];
          float v179_data = ir1[4];
          ir1[4] = (v179_data + (v156_data * v177_data));
          float v182_data = s1[43];
          float v184_data = ir1[5];
          ir1[5] = (v184_data + (v156_data * v182_data));
          float v187_data = s1[51];
          float v189_data = ir1[6];
          ir1[6] = (v189_data + (v156_data * v187_data));
          float v192_data = s1[59];
          float v194_data = ir1[7];
          ir1[7] = (v194_data + (v156_data * v192_data));
          float v196_data = r0[4];
          float v197_data = s1[4];
          float v199_data = ir1[0];
          ir1[0] = (v199_data + (v196_data * v197_data));
          float v202_data = s1[12];
          float v204_data = ir1[1];
          ir1[1] = (v204_data + (v196_data * v202_data));
          float v207_data = s1[20];
          float v209_data = ir1[2];
          ir1[2] = (v209_data + (v196_data * v207_data));
          float v212_data = s1[28];
          float v214_data = ir1[3];
          ir1[3] = (v214_data + (v196_data * v212_data));
          float v217_data = s1[36];
          float v219_data = ir1[4];
          ir1[4] = (v219_data + (v196_data * v217_data));
          float v222_data = s1[44];
          float v224_data = ir1[5];
          ir1[5] = (v224_data + (v196_data * v222_data));
          float v227_data = s1[52];
          float v229_data = ir1[6];
          ir1[6] = (v229_data + (v196_data * v227_data));
          float v232_data = s1[60];
          float v234_data = ir1[7];
          ir1[7] = (v234_data + (v196_data * v232_data));
          float v236_data = r0[5];
          float v237_data = s1[5];
          float v239_data = ir1[0];
          ir1[0] = (v239_data + (v236_data * v237_data));
          float v242_data = s1[13];
          float v244_data = ir1[1];
          ir1[1] = (v244_data + (v236_data * v242_data));
          float v247_data = s1[21];
          float v249_data = ir1[2];
          ir1[2] = (v249_data + (v236_data * v247_data));
          float v252_data = s1[29];
          float v254_data = ir1[3];
          ir1[3] = (v254_data + (v236_data * v252_data));
          float v257_data = s1[37];
          float v259_data = ir1[4];
          ir1[4] = (v259_data + (v236_data * v257_data));
          float v262_data = s1[45];
          float v264_data = ir1[5];
          ir1[5] = (v264_data + (v236_data * v262_data));
          float v267_data = s1[53];
          float v269_data = ir1[6];
          ir1[6] = (v269_data + (v236_data * v267_data));
          float v272_data = s1[61];
          float v274_data = ir1[7];
          ir1[7] = (v274_data + (v236_data * v272_data));
          float v276_data = r0[6];
          float v277_data = s1[6];
          float v279_data = ir1[0];
          ir1[0] = (v279_data + (v276_data * v277_data));
          float v282_data = s1[14];
          float v284_data = ir1[1];
          ir1[1] = (v284_data + (v276_data * v282_data));
          float v287_data = s1[22];
          float v289_data = ir1[2];
          ir1[2] = (v289_data + (v276_data * v287_data));
          float v292_data = s1[30];
          float v294_data = ir1[3];
          ir1[3] = (v294_data + (v276_data * v292_data));
          float v297_data = s1[38];
          float v299_data = ir1[4];
          ir1[4] = (v299_data + (v276_data * v297_data));
          float v302_data = s1[46];
          float v304_data = ir1[5];
          ir1[5] = (v304_data + (v276_data * v302_data));
          float v307_data = s1[54];
          float v309_data = ir1[6];
          ir1[6] = (v309_data + (v276_data * v307_data));
          float v312_data = s1[62];
          float v314_data = ir1[7];
          ir1[7] = (v314_data + (v276_data * v312_data));
          float v316_data = r0[7];
          float v317_data = s1[7];
          float v319_data = ir1[0];
          ir1[0] = (v319_data + (v316_data * v317_data));
          float v322_data = s1[15];
          float v324_data = ir1[1];
          ir1[1] = (v324_data + (v316_data * v322_data));
          float v327_data = s1[23];
          float v329_data = ir1[2];
          ir1[2] = (v329_data + (v316_data * v327_data));
          float v332_data = s1[31];
          float v334_data = ir1[3];
          ir1[3] = (v334_data + (v316_data * v332_data));
          float v337_data = s1[39];
          float v339_data = ir1[4];
          ir1[4] = (v339_data + (v316_data * v337_data));
          float v342_data = s1[47];
          float v344_data = ir1[5];
          ir1[5] = (v344_data + (v316_data * v342_data));
          float v347_data = s1[55];
          float v349_data = ir1[6];
          ir1[6] = (v349_data + (v316_data * v347_data));
          float v352_data = s1[63];
          float v354_data = ir1[7];
          ir1[7] = (v354_data + (v316_data * v352_data));
          if (v21_lead < 8) {
            #pragma unroll
            for (int32_t v357_n1 = 0; v357_n1 < 8; ++v357_n1) {
              float v359_data = ir1[v357_n1];
              r1[v357_n1] = v359_data;
            }
          }
          // glb_m1 = store{r>g}(r1);
          if (v21_lead < 8) {
            #pragma unroll
            for (int32_t v364_i1 = 0; v364_i1 < 8; ++v364_i1) {
              float v366_data = r1[v364_i1];
              glb_m1[(v21_lead + (v364_i1 * 8))] = v366_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

