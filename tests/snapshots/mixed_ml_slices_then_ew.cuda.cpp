// === base name ===
kernel_b408497f1c1ddea8

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b408497f1c1ddea8 = {{8, 16, 1}, 8, 8, 1, 16, 6656, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b408497f1c1ddea8(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b408497f1c1ddea8(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b408497f1c1ddea8(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_b408497f1c1ddea8, block.x * block.y * block.z, 1664 * sizeof(float));
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
void launcher_kernel_b408497f1c1ddea8(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b408497f1c1ddea8(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_b408497f1c1ddea8, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_b408497f1c1ddea8<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_b408497f1c1ddea8(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
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
      float* tempShrMem = &localShrMem0[96];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s2 = &localShrMem0[64];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v13_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v13_batchId0 < numElements0; v13_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v14_ahead1 = v13_batchId0 + (gridDim.x * blockDim.y);
        size_t v16_batchId1 = (v14_ahead1 < numElements0) ? v14_ahead1 : v13_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v13_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v13_batchId0 * 64 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v13_batchId0 * 32 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v13_batchId0 * 32 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[v13_batchId0 * 64 + 0 + m3_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v28_lead = threadIdx.x % 8;
          #pragma unroll
          for (int32_t v29_i0 = 0; v29_i0 < 1; ++v29_i0) {
            int32_t v32_lead = v28_lead + (v29_i0 * 8);
            #pragma unroll
            for (int32_t v30_i1 = 0; v30_i1 < 8; ++v30_i1) {
              float v35_data = __ldcg(&glb_m0[(v32_lead + (v30_i1 * 8))]);
              r0[(v29_i0 + v30_i1)] = v35_data;
            }
          }
          // s0 = load{g>s}(glb_m1[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m1[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 8], &glb_m1[0 + 0 + 1 * threadIdx.x + 8], 4);
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 16], &glb_m1[0 + 0 + 1 * threadIdx.x + 16], 4);
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 24], &glb_m1[0 + 0 + 1 * threadIdx.x + 24], 4);
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m0););
          // s2 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 8], &glb_m2[0 + 0 + 1 * threadIdx.x + 8], 4);
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 16], &glb_m2[0 + 0 + 1 * threadIdx.x + 16], 4);
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 24], &glb_m2[0 + 0 + 1 * threadIdx.x + 24], 4);
          __pipeline_commit();
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(1);
          float r1[4]{};
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          // r1 = +(r0 * s0) + None
          // [(0, 8), (0, 4)] [(0, 8)]
          float v46_data = r0[0];
          float v47_data = s0[0];
          float v49_data = r1[0];
          r1[0] = (v49_data + (v46_data * v47_data));
          float v52_data = s0[8];
          float v54_data = r1[1];
          r1[1] = (v54_data + (v46_data * v52_data));
          float v57_data = s0[16];
          float v59_data = r1[2];
          r1[2] = (v59_data + (v46_data * v57_data));
          float v62_data = s0[24];
          float v64_data = r1[3];
          r1[3] = (v64_data + (v46_data * v62_data));
          float v66_data = r0[1];
          float v67_data = s0[1];
          float v69_data = r1[0];
          r1[0] = (v69_data + (v66_data * v67_data));
          float v72_data = s0[9];
          float v74_data = r1[1];
          r1[1] = (v74_data + (v66_data * v72_data));
          float v77_data = s0[17];
          float v79_data = r1[2];
          r1[2] = (v79_data + (v66_data * v77_data));
          float v82_data = s0[25];
          float v84_data = r1[3];
          r1[3] = (v84_data + (v66_data * v82_data));
          float v86_data = r0[2];
          float v87_data = s0[2];
          float v89_data = r1[0];
          r1[0] = (v89_data + (v86_data * v87_data));
          float v92_data = s0[10];
          float v94_data = r1[1];
          r1[1] = (v94_data + (v86_data * v92_data));
          float v97_data = s0[18];
          float v99_data = r1[2];
          r1[2] = (v99_data + (v86_data * v97_data));
          float v102_data = s0[26];
          float v104_data = r1[3];
          r1[3] = (v104_data + (v86_data * v102_data));
          float v106_data = r0[3];
          float v107_data = s0[3];
          float v109_data = r1[0];
          r1[0] = (v109_data + (v106_data * v107_data));
          float v112_data = s0[11];
          float v114_data = r1[1];
          r1[1] = (v114_data + (v106_data * v112_data));
          float v117_data = s0[19];
          float v119_data = r1[2];
          r1[2] = (v119_data + (v106_data * v117_data));
          float v122_data = s0[27];
          float v124_data = r1[3];
          r1[3] = (v124_data + (v106_data * v122_data));
          float v126_data = r0[4];
          float v127_data = s0[4];
          float v129_data = r1[0];
          r1[0] = (v129_data + (v126_data * v127_data));
          float v132_data = s0[12];
          float v134_data = r1[1];
          r1[1] = (v134_data + (v126_data * v132_data));
          float v137_data = s0[20];
          float v139_data = r1[2];
          r1[2] = (v139_data + (v126_data * v137_data));
          float v142_data = s0[28];
          float v144_data = r1[3];
          r1[3] = (v144_data + (v126_data * v142_data));
          float v146_data = r0[5];
          float v147_data = s0[5];
          float v149_data = r1[0];
          r1[0] = (v149_data + (v146_data * v147_data));
          float v152_data = s0[13];
          float v154_data = r1[1];
          r1[1] = (v154_data + (v146_data * v152_data));
          float v157_data = s0[21];
          float v159_data = r1[2];
          r1[2] = (v159_data + (v146_data * v157_data));
          float v162_data = s0[29];
          float v164_data = r1[3];
          r1[3] = (v164_data + (v146_data * v162_data));
          float v166_data = r0[6];
          float v167_data = s0[6];
          float v169_data = r1[0];
          r1[0] = (v169_data + (v166_data * v167_data));
          float v172_data = s0[14];
          float v174_data = r1[1];
          r1[1] = (v174_data + (v166_data * v172_data));
          float v177_data = s0[22];
          float v179_data = r1[2];
          r1[2] = (v179_data + (v166_data * v177_data));
          float v182_data = s0[30];
          float v184_data = r1[3];
          r1[3] = (v184_data + (v166_data * v182_data));
          float v186_data = r0[7];
          float v187_data = s0[7];
          float v189_data = r1[0];
          r1[0] = (v189_data + (v186_data * v187_data));
          float v192_data = s0[15];
          float v194_data = r1[1];
          r1[1] = (v194_data + (v186_data * v192_data));
          float v197_data = s0[23];
          float v199_data = r1[2];
          r1[2] = (v199_data + (v186_data * v197_data));
          float v202_data = s0[31];
          float v204_data = r1[3];
          r1[3] = (v204_data + (v186_data * v202_data));
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          // s1 = store{r>s}(localShrMem0, r1);
          #pragma unroll
          for (int32_t v206_i0 = 0; v206_i0 < 1; ++v206_i0) {
            int32_t v211_lead = v28_lead + (v206_i0 * 8);
            #pragma unroll
            for (int32_t v207_i1 = 0; v207_i1 < 4; ++v207_i1) {
              float v209_data = r1[(v206_i0 + v207_i1)];
              int32_t v213_a = v211_lead + (v207_i1 * 8);
              s1[(v213_a ^ ((v213_a >> 5) & 31))] = v209_data;
            }
          }
          // wait(s2 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r2[4]{};
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          // ir2 = +(r0 * s2)
          // [(0, 8), (0, 4)] [(0, 8)]
          float ir2[4]{};
          float v220_data = s2[0];
          float v222_data = ir2[0];
          ir2[0] = (v222_data + (v46_data * v220_data));
          float v225_data = s2[8];
          float v227_data = ir2[1];
          ir2[1] = (v227_data + (v46_data * v225_data));
          float v230_data = s2[16];
          float v232_data = ir2[2];
          ir2[2] = (v232_data + (v46_data * v230_data));
          float v235_data = s2[24];
          float v237_data = ir2[3];
          ir2[3] = (v237_data + (v46_data * v235_data));
          float v240_data = s2[1];
          float v242_data = ir2[0];
          ir2[0] = (v242_data + (v66_data * v240_data));
          float v245_data = s2[9];
          float v247_data = ir2[1];
          ir2[1] = (v247_data + (v66_data * v245_data));
          float v250_data = s2[17];
          float v252_data = ir2[2];
          ir2[2] = (v252_data + (v66_data * v250_data));
          float v255_data = s2[25];
          float v257_data = ir2[3];
          ir2[3] = (v257_data + (v66_data * v255_data));
          float v260_data = s2[2];
          float v262_data = ir2[0];
          ir2[0] = (v262_data + (v86_data * v260_data));
          float v265_data = s2[10];
          float v267_data = ir2[1];
          ir2[1] = (v267_data + (v86_data * v265_data));
          float v270_data = s2[18];
          float v272_data = ir2[2];
          ir2[2] = (v272_data + (v86_data * v270_data));
          float v275_data = s2[26];
          float v277_data = ir2[3];
          ir2[3] = (v277_data + (v86_data * v275_data));
          float v280_data = s2[3];
          float v282_data = ir2[0];
          ir2[0] = (v282_data + (v106_data * v280_data));
          float v285_data = s2[11];
          float v287_data = ir2[1];
          ir2[1] = (v287_data + (v106_data * v285_data));
          float v290_data = s2[19];
          float v292_data = ir2[2];
          ir2[2] = (v292_data + (v106_data * v290_data));
          float v295_data = s2[27];
          float v297_data = ir2[3];
          ir2[3] = (v297_data + (v106_data * v295_data));
          float v300_data = s2[4];
          float v302_data = ir2[0];
          ir2[0] = (v302_data + (v126_data * v300_data));
          float v305_data = s2[12];
          float v307_data = ir2[1];
          ir2[1] = (v307_data + (v126_data * v305_data));
          float v310_data = s2[20];
          float v312_data = ir2[2];
          ir2[2] = (v312_data + (v126_data * v310_data));
          float v315_data = s2[28];
          float v317_data = ir2[3];
          ir2[3] = (v317_data + (v126_data * v315_data));
          float v320_data = s2[5];
          float v322_data = ir2[0];
          ir2[0] = (v322_data + (v146_data * v320_data));
          float v325_data = s2[13];
          float v327_data = ir2[1];
          ir2[1] = (v327_data + (v146_data * v325_data));
          float v330_data = s2[21];
          float v332_data = ir2[2];
          ir2[2] = (v332_data + (v146_data * v330_data));
          float v335_data = s2[29];
          float v337_data = ir2[3];
          ir2[3] = (v337_data + (v146_data * v335_data));
          float v340_data = s2[6];
          float v342_data = ir2[0];
          ir2[0] = (v342_data + (v166_data * v340_data));
          float v345_data = s2[14];
          float v347_data = ir2[1];
          ir2[1] = (v347_data + (v166_data * v345_data));
          float v350_data = s2[22];
          float v352_data = ir2[2];
          ir2[2] = (v352_data + (v166_data * v350_data));
          float v355_data = s2[30];
          float v357_data = ir2[3];
          ir2[3] = (v357_data + (v166_data * v355_data));
          float v360_data = s2[7];
          float v362_data = ir2[0];
          ir2[0] = (v362_data + (v186_data * v360_data));
          float v365_data = s2[15];
          float v367_data = ir2[1];
          ir2[1] = (v367_data + (v186_data * v365_data));
          float v370_data = s2[23];
          float v372_data = ir2[2];
          ir2[2] = (v372_data + (v186_data * v370_data));
          float v375_data = s2[31];
          float v377_data = ir2[3];
          ir2[3] = (v377_data + (v186_data * v375_data));
          // r2 = ir2
          #pragma unroll
          for (int32_t v379_n0 = 0; v379_n0 < 1; ++v379_n0) {
            #pragma unroll
            for (int32_t v380_n1 = 0; v380_n1 < 4; ++v380_n1) {
              int32_t v381_a = v379_n0 + v380_n1;
              float v382_data = ir2[v381_a];
              r2[v381_a] = v382_data;
            }
          }
          // s1 = store{r>s}(localShrMem0, r2);
          #pragma unroll
          for (int32_t v383_i0 = 0; v383_i0 < 1; ++v383_i0) {
            int32_t v388_lead = v28_lead + (v383_i0 * 8);
            #pragma unroll
            for (int32_t v384_i1 = 0; v384_i1 < 4; ++v384_i1) {
              float v386_data = r2[(v383_i0 + v384_i1)];
              int32_t v391_a = v388_lead + ((v384_i1 + 4) * 8);
              s1[(v391_a ^ ((v391_a >> 5) & 31))] = v386_data;
            }
          }
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          // glb_m3 = abs(s1)
          #pragma unroll
          for (int32_t v395_k0 = 0; v395_k0 < 1; ++v395_k0) {
            int32_t v398_lead = v28_lead + (v395_k0 * 8);
            #pragma unroll
            for (int32_t v396_k1 = 0; v396_k1 < 8; ++v396_k1) {
              int32_t v400_a = v398_lead + (v396_k1 * 8);
              float v404_data = s1[(v400_a ^ ((v400_a >> 5) & 31))];
              glb_m3[v400_a] = (fabsf(v404_data));
            }
          }
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
        }
      }
    }
  }
}

