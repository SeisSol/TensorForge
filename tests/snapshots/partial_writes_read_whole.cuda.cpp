// === base name ===
kernel_59a48a62ee994a4d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_59a48a62ee994a4d = {{32, 4, 1}, 32, 32, 1, 4, 6144, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_59a48a62ee994a4d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_59a48a62ee994a4d(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_59a48a62ee994a4d(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_59a48a62ee994a4d, block.x * block.y * block.z, 1536 * sizeof(float));
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
  config.sharedMemBytes = 1536 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_59a48a62ee994a4d(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_59a48a62ee994a4d(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_59a48a62ee994a4d, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_59a48a62ee994a4d<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_59a48a62ee994a4d(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 4 per block = block 32x4x1, 6144 B shared, occupancy grid
    // operands:
    //   m0 32×9(32×9) {0..32}×{0..9} pointer_based
    //   m1 16×9(16×9) {0..16}×{0..9} pointer_based
    //   m2 16×9(16×9) {0..16}×{0..9} pointer_based
    //   m3 32×9(32×9) {0..32}×{0..9} pointer_based
    //   m4 9×9(9×9) {0..9}×{0..9} pointer_based
    // operations:
    //   t0[i,j] = m0[i,j]
    //   t0[i,j] += m1[i,j]
    //   t0[i,j] += m2[i,j]
    //   m3[i,j] = t0[i,k] × m4[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":1536}],"shared_bytes":6144,"shared_elements":1536,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"pointer_based","alias":"Q","bbox":[[0,0],[32,9]],"name":"m0","ordered":false,"parts":1,"shape":[32,9],"variant":false},{"addressing":"pointer_based","alias":"F0","bbox":[[0,0],[16,9]],"name":"m1","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"pointer_based","alias":"F1","bbox":[[0,0],[16,9]],"name":"m2","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"pointer_based","alias":"O","bbox":[[0,0],[32,9]],"name":"m3","ordered":false,"parts":1,"shape":[32,9],"variant":false},{"addressing":"pointer_based","alias":"M","bbox":[[0,0],[9,9]],"name":"m4","ordered":false,"parts":1,"shape":[9,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[32,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[32,9]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[384 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[384];
      float * __restrict__ s0 = &localShrMem0[96];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v6_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v6_batchId0 < numElements0; v6_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v7_ahead1 = v6_batchId0 + (gridDim.x * blockDim.y);
        size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v6_batchId0][0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v6_batchId0][0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v6_batchId0][0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[v6_batchId0][0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[v6_batchId0][0 + m4_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v22_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
            int32_t v26_lead = v22_lead + (v23_i0 * 32);
            #pragma unroll
            for (int32_t v24_i1 = 0; v24_i1 < 9; ++v24_i1) {
              float v29_data = __ldcg(&glb_m0[(v26_lead + (v24_i1 * 32))]);
              r0[(v23_i0 + v24_i1)] = v29_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m1);
          bool v32_g = v22_lead < 16;
          if (v32_g) {
            #pragma unroll
            for (int32_t v33_i1 = 0; v33_i1 < 9; ++v33_i1) {
              float v38_data = __ldcg(&glb_m1[(v22_lead + (v33_i1 * 16))]);
              r2[v33_i1] = v38_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 9)] []
          float v41_data = r0[0];
          float v42_data = r1[0];
          r1[0] = (v42_data + v41_data);
          float v44_data = r0[1];
          float v45_data = r1[1];
          r1[1] = (v45_data + v44_data);
          float v47_data = r0[2];
          float v48_data = r1[2];
          r1[2] = (v48_data + v47_data);
          float v50_data = r0[3];
          float v51_data = r1[3];
          r1[3] = (v51_data + v50_data);
          float v53_data = r0[4];
          float v54_data = r1[4];
          r1[4] = (v54_data + v53_data);
          float v56_data = r0[5];
          float v57_data = r1[5];
          r1[5] = (v57_data + v56_data);
          float v59_data = r0[6];
          float v60_data = r1[6];
          r1[6] = (v60_data + v59_data);
          float v62_data = r0[7];
          float v63_data = r1[7];
          r1[7] = (v63_data + v62_data);
          float v65_data = r0[8];
          float v66_data = r1[8];
          r1[8] = (v66_data + v65_data);
          // s0 = store{r>s}(localShrMem0, r1);
          #pragma unroll
          for (int32_t v68_i0 = 0; v68_i0 < 1; ++v68_i0) {
            int32_t v73_lead = v22_lead + (v68_i0 * 32);
            #pragma unroll
            for (int32_t v69_i1 = 0; v69_i1 < 9; ++v69_i1) {
              float v71_data = r1[(v68_i0 + v69_i1)];
              int32_t v75_a = v73_lead + (v69_i1 * 32);
              s0[(v75_a ^ ((v75_a >> 5) & 31))] = v71_data;
            }
          }
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          if (v32_g) {
            #pragma unroll
            for (int32_t v80_i1 = 0; v80_i1 < 9; ++v80_i1) {
              float v85_data = __ldcg(&glb_m2[(v22_lead + (v80_i1 * 16))]);
              r4[v80_i1] = v85_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          __syncwarp();
          // r3 = +(r2) + name: s0, type: SymbolType.SharedMem, lead: [0]
          // [(0, 16), (0, 9)] []
          float ir3[9]{};
          float v89_data = r2[0];
          float v90_data = ir3[0];
          ir3[0] = (v90_data + v89_data);
          float v92_data = r2[1];
          float v93_data = ir3[1];
          ir3[1] = (v93_data + v92_data);
          float v95_data = r2[2];
          float v96_data = ir3[2];
          ir3[2] = (v96_data + v95_data);
          float v98_data = r2[3];
          float v99_data = ir3[3];
          ir3[3] = (v99_data + v98_data);
          float v101_data = r2[4];
          float v102_data = ir3[4];
          ir3[4] = (v102_data + v101_data);
          float v104_data = r2[5];
          float v105_data = ir3[5];
          ir3[5] = (v105_data + v104_data);
          float v107_data = r2[6];
          float v108_data = ir3[6];
          ir3[6] = (v108_data + v107_data);
          float v110_data = r2[7];
          float v111_data = ir3[7];
          ir3[7] = (v111_data + v110_data);
          float v113_data = r2[8];
          float v114_data = ir3[8];
          ir3[8] = (v114_data + v113_data);
          if (v32_g) {
            #pragma unroll
            for (int32_t v116_n1 = 0; v116_n1 < 9; ++v116_n1) {
              float v118_data = ir3[v116_n1];
              int32_t v122_a = v22_lead + (v116_n1 * 32);
              float v126_data = s0[(v122_a ^ ((v122_a >> 5) & 31))];
              r3[v116_n1] = (v126_data + v118_data);
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r3);
          if (v32_g) {
            #pragma unroll
            for (int32_t v128_i1 = 0; v128_i1 < 9; ++v128_i1) {
              float v130_data = r3[v128_i1];
              int32_t v134_a = v22_lead + (v128_i1 * 32);
              s0[(v134_a ^ ((v134_a >> 5) & 31))] = v130_data;
            }
          }
          // s1 = load{g>s}(glb_m4[0, 1])
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 0], &glb_m4[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 32], &glb_m4[0 + 0 + 1 * threadIdx.x + 32], 4);
          if (threadIdx.x < 17) {
            __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 64], &glb_m4[0 + 0 + 1 * threadIdx.x + 64], 4);
          }
          __pipeline_commit();
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          __syncwarp();
          // r5 = +(r4) + name: s0, type: SymbolType.SharedMem, lead: [0]
          // [(0, 16), (0, 9)] []
          float ir5[9]{};
          float v143_data = r4[0];
          float v144_data = ir5[0];
          ir5[0] = (v144_data + v143_data);
          float v146_data = r4[1];
          float v147_data = ir5[1];
          ir5[1] = (v147_data + v146_data);
          float v149_data = r4[2];
          float v150_data = ir5[2];
          ir5[2] = (v150_data + v149_data);
          float v152_data = r4[3];
          float v153_data = ir5[3];
          ir5[3] = (v153_data + v152_data);
          float v155_data = r4[4];
          float v156_data = ir5[4];
          ir5[4] = (v156_data + v155_data);
          float v158_data = r4[5];
          float v159_data = ir5[5];
          ir5[5] = (v159_data + v158_data);
          float v161_data = r4[6];
          float v162_data = ir5[6];
          ir5[6] = (v162_data + v161_data);
          float v164_data = r4[7];
          float v165_data = ir5[7];
          ir5[7] = (v165_data + v164_data);
          float v167_data = r4[8];
          float v168_data = ir5[8];
          ir5[8] = (v168_data + v167_data);
          if (v32_g) {
            #pragma unroll
            for (int32_t v170_n1 = 0; v170_n1 < 9; ++v170_n1) {
              float v172_data = ir5[v170_n1];
              int32_t v176_a = v22_lead + (v170_n1 * 32);
              float v180_data = s0[(v176_a ^ ((v176_a >> 5) & 31))];
              r5[v170_n1] = (v180_data + v172_data);
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r5);
          if (v32_g) {
            #pragma unroll
            for (int32_t v182_i1 = 0; v182_i1 < 9; ++v182_i1) {
              float v184_data = r5[v182_i1];
              int32_t v188_a = v22_lead + (v182_i1 * 32);
              s0[(v188_a ^ ((v188_a >> 5) & 31))] = v184_data;
            }
          }
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          float r6[9]{};
          __syncwarp();
          // r6 = +(s0 * s1) + None
          // [(0, 32), (0, 9)] [(0, 9)]
          float ir6[9]{};
          float v200_data = s0[(v22_lead ^ ((v22_lead >> 5) & 31))];
          float v201_data = s1[0];
          float v203_data = ir6[0];
          ir6[0] = (v203_data + (v200_data * v201_data));
          float v206_data = s1[9];
          float v208_data = ir6[1];
          ir6[1] = (v208_data + (v200_data * v206_data));
          float v211_data = s1[18];
          float v213_data = ir6[2];
          ir6[2] = (v213_data + (v200_data * v211_data));
          float v216_data = s1[27];
          float v218_data = ir6[3];
          ir6[3] = (v218_data + (v200_data * v216_data));
          float v221_data = s1[36];
          float v223_data = ir6[4];
          ir6[4] = (v223_data + (v200_data * v221_data));
          float v226_data = s1[45];
          float v228_data = ir6[5];
          ir6[5] = (v228_data + (v200_data * v226_data));
          float v231_data = s1[54];
          float v233_data = ir6[6];
          ir6[6] = (v233_data + (v200_data * v231_data));
          float v236_data = s1[63];
          float v238_data = ir6[7];
          ir6[7] = (v238_data + (v200_data * v236_data));
          float v241_data = s1[72];
          float v243_data = ir6[8];
          ir6[8] = (v243_data + (v200_data * v241_data));
          int32_t v245_a = v22_lead + 32;
          float v249_data = s0[(v245_a ^ ((v245_a >> 5) & 31))];
          float v250_data = s1[1];
          float v252_data = ir6[0];
          ir6[0] = (v252_data + (v249_data * v250_data));
          float v255_data = s1[10];
          float v257_data = ir6[1];
          ir6[1] = (v257_data + (v249_data * v255_data));
          float v260_data = s1[19];
          float v262_data = ir6[2];
          ir6[2] = (v262_data + (v249_data * v260_data));
          float v265_data = s1[28];
          float v267_data = ir6[3];
          ir6[3] = (v267_data + (v249_data * v265_data));
          float v270_data = s1[37];
          float v272_data = ir6[4];
          ir6[4] = (v272_data + (v249_data * v270_data));
          float v275_data = s1[46];
          float v277_data = ir6[5];
          ir6[5] = (v277_data + (v249_data * v275_data));
          float v280_data = s1[55];
          float v282_data = ir6[6];
          ir6[6] = (v282_data + (v249_data * v280_data));
          float v285_data = s1[64];
          float v287_data = ir6[7];
          ir6[7] = (v287_data + (v249_data * v285_data));
          float v290_data = s1[73];
          float v292_data = ir6[8];
          ir6[8] = (v292_data + (v249_data * v290_data));
          int32_t v294_a = v22_lead + 64;
          float v298_data = s0[(v294_a ^ ((v294_a >> 5) & 31))];
          float v299_data = s1[2];
          float v301_data = ir6[0];
          ir6[0] = (v301_data + (v298_data * v299_data));
          float v304_data = s1[11];
          float v306_data = ir6[1];
          ir6[1] = (v306_data + (v298_data * v304_data));
          float v309_data = s1[20];
          float v311_data = ir6[2];
          ir6[2] = (v311_data + (v298_data * v309_data));
          float v314_data = s1[29];
          float v316_data = ir6[3];
          ir6[3] = (v316_data + (v298_data * v314_data));
          float v319_data = s1[38];
          float v321_data = ir6[4];
          ir6[4] = (v321_data + (v298_data * v319_data));
          float v324_data = s1[47];
          float v326_data = ir6[5];
          ir6[5] = (v326_data + (v298_data * v324_data));
          float v329_data = s1[56];
          float v331_data = ir6[6];
          ir6[6] = (v331_data + (v298_data * v329_data));
          float v334_data = s1[65];
          float v336_data = ir6[7];
          ir6[7] = (v336_data + (v298_data * v334_data));
          float v339_data = s1[74];
          float v341_data = ir6[8];
          ir6[8] = (v341_data + (v298_data * v339_data));
          int32_t v343_a = v22_lead + 96;
          float v347_data = s0[(v343_a ^ ((v343_a >> 5) & 31))];
          float v348_data = s1[3];
          float v350_data = ir6[0];
          ir6[0] = (v350_data + (v347_data * v348_data));
          float v353_data = s1[12];
          float v355_data = ir6[1];
          ir6[1] = (v355_data + (v347_data * v353_data));
          float v358_data = s1[21];
          float v360_data = ir6[2];
          ir6[2] = (v360_data + (v347_data * v358_data));
          float v363_data = s1[30];
          float v365_data = ir6[3];
          ir6[3] = (v365_data + (v347_data * v363_data));
          float v368_data = s1[39];
          float v370_data = ir6[4];
          ir6[4] = (v370_data + (v347_data * v368_data));
          float v373_data = s1[48];
          float v375_data = ir6[5];
          ir6[5] = (v375_data + (v347_data * v373_data));
          float v378_data = s1[57];
          float v380_data = ir6[6];
          ir6[6] = (v380_data + (v347_data * v378_data));
          float v383_data = s1[66];
          float v385_data = ir6[7];
          ir6[7] = (v385_data + (v347_data * v383_data));
          float v388_data = s1[75];
          float v390_data = ir6[8];
          ir6[8] = (v390_data + (v347_data * v388_data));
          int32_t v392_a = v22_lead + 128;
          float v396_data = s0[(v392_a ^ ((v392_a >> 5) & 31))];
          float v397_data = s1[4];
          float v399_data = ir6[0];
          ir6[0] = (v399_data + (v396_data * v397_data));
          float v402_data = s1[13];
          float v404_data = ir6[1];
          ir6[1] = (v404_data + (v396_data * v402_data));
          float v407_data = s1[22];
          float v409_data = ir6[2];
          ir6[2] = (v409_data + (v396_data * v407_data));
          float v412_data = s1[31];
          float v414_data = ir6[3];
          ir6[3] = (v414_data + (v396_data * v412_data));
          float v417_data = s1[40];
          float v419_data = ir6[4];
          ir6[4] = (v419_data + (v396_data * v417_data));
          float v422_data = s1[49];
          float v424_data = ir6[5];
          ir6[5] = (v424_data + (v396_data * v422_data));
          float v427_data = s1[58];
          float v429_data = ir6[6];
          ir6[6] = (v429_data + (v396_data * v427_data));
          float v432_data = s1[67];
          float v434_data = ir6[7];
          ir6[7] = (v434_data + (v396_data * v432_data));
          float v437_data = s1[76];
          float v439_data = ir6[8];
          ir6[8] = (v439_data + (v396_data * v437_data));
          int32_t v441_a = v22_lead + 160;
          float v445_data = s0[(v441_a ^ ((v441_a >> 5) & 31))];
          float v446_data = s1[5];
          float v448_data = ir6[0];
          ir6[0] = (v448_data + (v445_data * v446_data));
          float v451_data = s1[14];
          float v453_data = ir6[1];
          ir6[1] = (v453_data + (v445_data * v451_data));
          float v456_data = s1[23];
          float v458_data = ir6[2];
          ir6[2] = (v458_data + (v445_data * v456_data));
          float v461_data = s1[32];
          float v463_data = ir6[3];
          ir6[3] = (v463_data + (v445_data * v461_data));
          float v466_data = s1[41];
          float v468_data = ir6[4];
          ir6[4] = (v468_data + (v445_data * v466_data));
          float v471_data = s1[50];
          float v473_data = ir6[5];
          ir6[5] = (v473_data + (v445_data * v471_data));
          float v476_data = s1[59];
          float v478_data = ir6[6];
          ir6[6] = (v478_data + (v445_data * v476_data));
          float v481_data = s1[68];
          float v483_data = ir6[7];
          ir6[7] = (v483_data + (v445_data * v481_data));
          float v486_data = s1[77];
          float v488_data = ir6[8];
          ir6[8] = (v488_data + (v445_data * v486_data));
          int32_t v490_a = v22_lead + 192;
          float v494_data = s0[(v490_a ^ ((v490_a >> 5) & 31))];
          float v495_data = s1[6];
          float v497_data = ir6[0];
          ir6[0] = (v497_data + (v494_data * v495_data));
          float v500_data = s1[15];
          float v502_data = ir6[1];
          ir6[1] = (v502_data + (v494_data * v500_data));
          float v505_data = s1[24];
          float v507_data = ir6[2];
          ir6[2] = (v507_data + (v494_data * v505_data));
          float v510_data = s1[33];
          float v512_data = ir6[3];
          ir6[3] = (v512_data + (v494_data * v510_data));
          float v515_data = s1[42];
          float v517_data = ir6[4];
          ir6[4] = (v517_data + (v494_data * v515_data));
          float v520_data = s1[51];
          float v522_data = ir6[5];
          ir6[5] = (v522_data + (v494_data * v520_data));
          float v525_data = s1[60];
          float v527_data = ir6[6];
          ir6[6] = (v527_data + (v494_data * v525_data));
          float v530_data = s1[69];
          float v532_data = ir6[7];
          ir6[7] = (v532_data + (v494_data * v530_data));
          float v535_data = s1[78];
          float v537_data = ir6[8];
          ir6[8] = (v537_data + (v494_data * v535_data));
          int32_t v539_a = v22_lead + 224;
          float v543_data = s0[(v539_a ^ ((v539_a >> 5) & 31))];
          float v544_data = s1[7];
          float v546_data = ir6[0];
          ir6[0] = (v546_data + (v543_data * v544_data));
          float v549_data = s1[16];
          float v551_data = ir6[1];
          ir6[1] = (v551_data + (v543_data * v549_data));
          float v554_data = s1[25];
          float v556_data = ir6[2];
          ir6[2] = (v556_data + (v543_data * v554_data));
          float v559_data = s1[34];
          float v561_data = ir6[3];
          ir6[3] = (v561_data + (v543_data * v559_data));
          float v564_data = s1[43];
          float v566_data = ir6[4];
          ir6[4] = (v566_data + (v543_data * v564_data));
          float v569_data = s1[52];
          float v571_data = ir6[5];
          ir6[5] = (v571_data + (v543_data * v569_data));
          float v574_data = s1[61];
          float v576_data = ir6[6];
          ir6[6] = (v576_data + (v543_data * v574_data));
          float v579_data = s1[70];
          float v581_data = ir6[7];
          ir6[7] = (v581_data + (v543_data * v579_data));
          float v584_data = s1[79];
          float v586_data = ir6[8];
          ir6[8] = (v586_data + (v543_data * v584_data));
          int32_t v588_a = v22_lead + 256;
          float v592_data = s0[(v588_a ^ ((v588_a >> 5) & 31))];
          float v593_data = s1[8];
          float v595_data = ir6[0];
          ir6[0] = (v595_data + (v592_data * v593_data));
          float v598_data = s1[17];
          float v600_data = ir6[1];
          ir6[1] = (v600_data + (v592_data * v598_data));
          float v603_data = s1[26];
          float v605_data = ir6[2];
          ir6[2] = (v605_data + (v592_data * v603_data));
          float v608_data = s1[35];
          float v610_data = ir6[3];
          ir6[3] = (v610_data + (v592_data * v608_data));
          float v613_data = s1[44];
          float v615_data = ir6[4];
          ir6[4] = (v615_data + (v592_data * v613_data));
          float v618_data = s1[53];
          float v620_data = ir6[5];
          ir6[5] = (v620_data + (v592_data * v618_data));
          float v623_data = s1[62];
          float v625_data = ir6[6];
          ir6[6] = (v625_data + (v592_data * v623_data));
          float v628_data = s1[71];
          float v630_data = ir6[7];
          ir6[7] = (v630_data + (v592_data * v628_data));
          float v633_data = s1[80];
          float v635_data = ir6[8];
          ir6[8] = (v635_data + (v592_data * v633_data));
          #pragma unroll
          for (int32_t v637_n0 = 0; v637_n0 < 1; ++v637_n0) {
            #pragma unroll
            for (int32_t v638_n1 = 0; v638_n1 < 9; ++v638_n1) {
              int32_t v639_a = v637_n0 + v638_n1;
              float v640_data = ir6[v639_a];
              r6[v639_a] = v640_data;
            }
          }
          // glb_m3 = store{r>g}(r6);
          #pragma unroll
          for (int32_t v641_i0 = 0; v641_i0 < 1; ++v641_i0) {
            int32_t v646_lead = v22_lead + (v641_i0 * 32);
            #pragma unroll
            for (int32_t v642_i1 = 0; v642_i1 < 9; ++v642_i1) {
              float v644_data = r6[(v641_i0 + v642_i1)];
              glb_m3[(v646_lead + (v642_i1 * 32))] = v644_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

