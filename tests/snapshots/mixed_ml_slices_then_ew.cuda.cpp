// === base name ===
kernel_af3b213bad583f08

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_af3b213bad583f08 = {{8, 16, 1}, 8, 8, 1, 16, 6656, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_af3b213bad583f08(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_af3b213bad583f08(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_af3b213bad583f08(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_af3b213bad583f08, block.x * block.y * block.z, 1664 * sizeof(float));
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
  config.sharedMemBytes = 1664 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_af3b213bad583f08(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_af3b213bad583f08(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_af3b213bad583f08, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_af3b213bad583f08<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_af3b213bad583f08(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 8 lanes x 16 per block = block 8x16x1, 6656 B shared, occupancy grid
    // operands:
    //   m0 8×8(8×8) {0..8}×{0..8} strided
    //   m1 8×4(8×4) {0..8}×{0..4} strided
    //   m2 8×4(8×4) {0..8}×{0..4} strided
    //   m3 8×8(8×8) {0..8}×{0..8} strided
    // operations:
    //   t0[i,j]@{0..8}×{0..4} = m0[i,k] × m1[k,j]
    //   t0[i,j]@{0..8}×{4..8} = m0[i,k] × m2[k,j]
    //   C = abs(TMP)
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1664}],"shared_bytes":6656,"shared_elements":1664,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[8,4]],"name":"m1","ordered":false,"parts":1,"shape":[8,4],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[8,4]],"name":"m2","ordered":false,"parts":1,"shape":[8,4],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,4]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,4]],"is_tmp":true,"name":"t0","offset":[0,4],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[104 * threadIdx.y + 0];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      float * __restrict__ s2 = &localShrMem0[64];
      for (size_t v10_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v10_batchId0 < numElements0; v10_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v11_ahead1 = v10_batchId0 + (gridDim.x * blockDim.y);
        size_t v13_batchId1 = (v11_ahead1 < numElements0) ? v11_ahead1 : v10_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v10_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v10_batchId0 * 64 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v10_batchId0 * 32 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v10_batchId0 * 32 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[v10_batchId0 * 64 + 0 + m3_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v25_lead = threadIdx.x % 8;
          #pragma unroll
          for (int32_t v26_i0 = 0; v26_i0 < 1; ++v26_i0) {
            int32_t v29_lead = v25_lead + (v26_i0 * 8);
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 8; ++v27_i1) {
              float v32_data = __ldcg(&glb_m0[(v29_lead + (v27_i1 * 8))]);
              r0[(v26_i0 + v27_i1)] = v32_data;
            }
          }
          // s0 = load{g>s}(glb_m1[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m1[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 8], &glb_m1[0 + 0 + 1 * threadIdx.x + 8], 4);
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 16], &glb_m1[0 + 0 + 1 * threadIdx.x + 16], 4);
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 24], &glb_m1[0 + 0 + 1 * threadIdx.x + 24], 4);
          __pipeline_commit();
          // s2 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 8], &glb_m2[0 + 0 + 1 * threadIdx.x + 8], 4);
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 16], &glb_m2[0 + 0 + 1 * threadIdx.x + 16], 4);
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 24], &glb_m2[0 + 0 + 1 * threadIdx.x + 24], 4);
          __pipeline_commit();
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(1);
          float r1[4]{};
          // r1 = +(r0 * s0) + None
          // [(0, 8), (0, 4)] [(0, 8)]
          float v39_data = r0[0];
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          float v40_data = s0[0];
          float v42_data = r1[0];
          r1[0] = (v42_data + (v39_data * v40_data));
          float v45_data = s0[8];
          float v47_data = r1[1];
          r1[1] = (v47_data + (v39_data * v45_data));
          float v50_data = s0[16];
          float v52_data = r1[2];
          r1[2] = (v52_data + (v39_data * v50_data));
          float v55_data = s0[24];
          float v57_data = r1[3];
          r1[3] = (v57_data + (v39_data * v55_data));
          float v59_data = r0[1];
          float v60_data = s0[1];
          float v62_data = r1[0];
          r1[0] = (v62_data + (v59_data * v60_data));
          float v65_data = s0[9];
          float v67_data = r1[1];
          r1[1] = (v67_data + (v59_data * v65_data));
          float v70_data = s0[17];
          float v72_data = r1[2];
          r1[2] = (v72_data + (v59_data * v70_data));
          float v75_data = s0[25];
          float v77_data = r1[3];
          r1[3] = (v77_data + (v59_data * v75_data));
          float v79_data = r0[2];
          float v80_data = s0[2];
          float v82_data = r1[0];
          r1[0] = (v82_data + (v79_data * v80_data));
          float v85_data = s0[10];
          float v87_data = r1[1];
          r1[1] = (v87_data + (v79_data * v85_data));
          float v90_data = s0[18];
          float v92_data = r1[2];
          r1[2] = (v92_data + (v79_data * v90_data));
          float v95_data = s0[26];
          float v97_data = r1[3];
          r1[3] = (v97_data + (v79_data * v95_data));
          float v99_data = r0[3];
          float v100_data = s0[3];
          float v102_data = r1[0];
          r1[0] = (v102_data + (v99_data * v100_data));
          float v105_data = s0[11];
          float v107_data = r1[1];
          r1[1] = (v107_data + (v99_data * v105_data));
          float v110_data = s0[19];
          float v112_data = r1[2];
          r1[2] = (v112_data + (v99_data * v110_data));
          float v115_data = s0[27];
          float v117_data = r1[3];
          r1[3] = (v117_data + (v99_data * v115_data));
          float v119_data = r0[4];
          float v120_data = s0[4];
          float v122_data = r1[0];
          r1[0] = (v122_data + (v119_data * v120_data));
          float v125_data = s0[12];
          float v127_data = r1[1];
          r1[1] = (v127_data + (v119_data * v125_data));
          float v130_data = s0[20];
          float v132_data = r1[2];
          r1[2] = (v132_data + (v119_data * v130_data));
          float v135_data = s0[28];
          float v137_data = r1[3];
          r1[3] = (v137_data + (v119_data * v135_data));
          float v139_data = r0[5];
          float v140_data = s0[5];
          float v142_data = r1[0];
          r1[0] = (v142_data + (v139_data * v140_data));
          float v145_data = s0[13];
          float v147_data = r1[1];
          r1[1] = (v147_data + (v139_data * v145_data));
          float v150_data = s0[21];
          float v152_data = r1[2];
          r1[2] = (v152_data + (v139_data * v150_data));
          float v155_data = s0[29];
          float v157_data = r1[3];
          r1[3] = (v157_data + (v139_data * v155_data));
          float v159_data = r0[6];
          float v160_data = s0[6];
          float v162_data = r1[0];
          r1[0] = (v162_data + (v159_data * v160_data));
          float v165_data = s0[14];
          float v167_data = r1[1];
          r1[1] = (v167_data + (v159_data * v165_data));
          float v170_data = s0[22];
          float v172_data = r1[2];
          r1[2] = (v172_data + (v159_data * v170_data));
          float v175_data = s0[30];
          float v177_data = r1[3];
          r1[3] = (v177_data + (v159_data * v175_data));
          float v179_data = r0[7];
          float v180_data = s0[7];
          float v182_data = r1[0];
          r1[0] = (v182_data + (v179_data * v180_data));
          float v185_data = s0[15];
          float v187_data = r1[1];
          r1[1] = (v187_data + (v179_data * v185_data));
          float v190_data = s0[23];
          float v192_data = r1[2];
          r1[2] = (v192_data + (v179_data * v190_data));
          float v195_data = s0[31];
          float v197_data = r1[3];
          r1[3] = (v197_data + (v179_data * v195_data));
          // s1 = store{r>s}(localShrMem0, r1);
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          #pragma unroll
          for (int32_t v199_i0 = 0; v199_i0 < 1; ++v199_i0) {
            int32_t v204_lead = v25_lead + (v199_i0 * 8);
            #pragma unroll
            for (int32_t v200_i1 = 0; v200_i1 < 4; ++v200_i1) {
              float v202_data = r1[(v199_i0 + v200_i1)];
              int32_t v206_a = v204_lead + (v200_i1 * 8);
              s1[(v206_a ^ ((v206_a >> 5) & 31))] = v202_data;
            }
          }
          // wait(s2 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r2[4]{};
          // ir2 = +(r0 * s2)
          // [(0, 8), (0, 4)] [(0, 8)]
          float ir2[4]{};
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          float v217_data = s2[0];
          float v219_data = ir2[0];
          ir2[0] = (v219_data + (v39_data * v217_data));
          float v222_data = s2[8];
          float v224_data = ir2[1];
          ir2[1] = (v224_data + (v39_data * v222_data));
          float v227_data = s2[16];
          float v229_data = ir2[2];
          ir2[2] = (v229_data + (v39_data * v227_data));
          float v232_data = s2[24];
          float v234_data = ir2[3];
          ir2[3] = (v234_data + (v39_data * v232_data));
          float v237_data = s2[1];
          float v239_data = ir2[0];
          ir2[0] = (v239_data + (v59_data * v237_data));
          float v242_data = s2[9];
          float v244_data = ir2[1];
          ir2[1] = (v244_data + (v59_data * v242_data));
          float v247_data = s2[17];
          float v249_data = ir2[2];
          ir2[2] = (v249_data + (v59_data * v247_data));
          float v252_data = s2[25];
          float v254_data = ir2[3];
          ir2[3] = (v254_data + (v59_data * v252_data));
          float v257_data = s2[2];
          float v259_data = ir2[0];
          ir2[0] = (v259_data + (v79_data * v257_data));
          float v262_data = s2[10];
          float v264_data = ir2[1];
          ir2[1] = (v264_data + (v79_data * v262_data));
          float v267_data = s2[18];
          float v269_data = ir2[2];
          ir2[2] = (v269_data + (v79_data * v267_data));
          float v272_data = s2[26];
          float v274_data = ir2[3];
          ir2[3] = (v274_data + (v79_data * v272_data));
          float v277_data = s2[3];
          float v279_data = ir2[0];
          ir2[0] = (v279_data + (v99_data * v277_data));
          float v282_data = s2[11];
          float v284_data = ir2[1];
          ir2[1] = (v284_data + (v99_data * v282_data));
          float v287_data = s2[19];
          float v289_data = ir2[2];
          ir2[2] = (v289_data + (v99_data * v287_data));
          float v292_data = s2[27];
          float v294_data = ir2[3];
          ir2[3] = (v294_data + (v99_data * v292_data));
          float v297_data = s2[4];
          float v299_data = ir2[0];
          ir2[0] = (v299_data + (v119_data * v297_data));
          float v302_data = s2[12];
          float v304_data = ir2[1];
          ir2[1] = (v304_data + (v119_data * v302_data));
          float v307_data = s2[20];
          float v309_data = ir2[2];
          ir2[2] = (v309_data + (v119_data * v307_data));
          float v312_data = s2[28];
          float v314_data = ir2[3];
          ir2[3] = (v314_data + (v119_data * v312_data));
          float v317_data = s2[5];
          float v319_data = ir2[0];
          ir2[0] = (v319_data + (v139_data * v317_data));
          float v322_data = s2[13];
          float v324_data = ir2[1];
          ir2[1] = (v324_data + (v139_data * v322_data));
          float v327_data = s2[21];
          float v329_data = ir2[2];
          ir2[2] = (v329_data + (v139_data * v327_data));
          float v332_data = s2[29];
          float v334_data = ir2[3];
          ir2[3] = (v334_data + (v139_data * v332_data));
          float v337_data = s2[6];
          float v339_data = ir2[0];
          ir2[0] = (v339_data + (v159_data * v337_data));
          float v342_data = s2[14];
          float v344_data = ir2[1];
          ir2[1] = (v344_data + (v159_data * v342_data));
          float v347_data = s2[22];
          float v349_data = ir2[2];
          ir2[2] = (v349_data + (v159_data * v347_data));
          float v352_data = s2[30];
          float v354_data = ir2[3];
          ir2[3] = (v354_data + (v159_data * v352_data));
          float v357_data = s2[7];
          float v359_data = ir2[0];
          ir2[0] = (v359_data + (v179_data * v357_data));
          float v362_data = s2[15];
          float v364_data = ir2[1];
          ir2[1] = (v364_data + (v179_data * v362_data));
          float v367_data = s2[23];
          float v369_data = ir2[2];
          ir2[2] = (v369_data + (v179_data * v367_data));
          float v372_data = s2[31];
          float v374_data = ir2[3];
          ir2[3] = (v374_data + (v179_data * v372_data));
          // r2 = ir2
          #pragma unroll
          for (int32_t v376_n0 = 0; v376_n0 < 1; ++v376_n0) {
            #pragma unroll
            for (int32_t v377_n1 = 0; v377_n1 < 4; ++v377_n1) {
              int32_t v378_a = v376_n0 + v377_n1;
              float v379_data = ir2[v378_a];
              r2[v378_a] = v379_data;
            }
          }
          // s1 = store{r>s}(localShrMem0, r2);
          #pragma unroll
          for (int32_t v380_i0 = 0; v380_i0 < 1; ++v380_i0) {
            int32_t v385_lead = v25_lead + (v380_i0 * 8);
            #pragma unroll
            for (int32_t v381_i1 = 0; v381_i1 < 4; ++v381_i1) {
              float v383_data = r2[(v380_i0 + v381_i1)];
              int32_t v388_a = v385_lead + ((v381_i1 + 4) * 8);
              s1[(v388_a ^ ((v388_a >> 5) & 31))] = v383_data;
            }
          }
          // glb_m3 = abs(s1)
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          #pragma unroll
          for (int32_t v392_k0 = 0; v392_k0 < 1; ++v392_k0) {
            int32_t v395_lead = v25_lead + (v392_k0 * 8);
            #pragma unroll
            for (int32_t v393_k1 = 0; v393_k1 < 8; ++v393_k1) {
              int32_t v397_a = v395_lead + (v393_k1 * 8);
              float v401_data = s1[(v397_a ^ ((v397_a >> 5) & 31))];
              glb_m3[v397_a] = (std::fabs(v401_data));
            }
          }
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
        }
      }
    }
  }
}

