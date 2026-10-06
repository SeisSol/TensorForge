// === base name ===
kernel_073de4e4e1b249b2

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_073de4e4e1b249b2 = {{32, 4, 1}, 32, 32, 1, 4, 1536, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_073de4e4e1b249b2(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_073de4e4e1b249b2(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_073de4e4e1b249b2(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_073de4e4e1b249b2, block.x * block.y * block.z, 384 * sizeof(float));
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
  config.sharedMemBytes = 384 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_073de4e4e1b249b2(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_073de4e4e1b249b2(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_073de4e4e1b249b2, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_073de4e4e1b249b2<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_073de4e4e1b249b2(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 4 per block = block 32x4x1, 1536 B shared, occupancy grid
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
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":384}],"shared_bytes":1536,"shared_elements":384,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"pointer_based","alias":"Q","bbox":[[0,0],[32,9]],"name":"m0","ordered":false,"parts":1,"shape":[32,9],"variant":false},{"addressing":"pointer_based","alias":"F0","bbox":[[0,0],[16,9]],"name":"m1","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"pointer_based","alias":"F1","bbox":[[0,0],[16,9]],"name":"m2","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"pointer_based","alias":"O","bbox":[[0,0],[32,9]],"name":"m3","ordered":false,"parts":1,"shape":[32,9],"variant":false},{"addressing":"pointer_based","alias":"M","bbox":[[0,0],[9,9]],"name":"m4","ordered":false,"parts":1,"shape":[9,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[32,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[32,9]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[96 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[96];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v11_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v11_batchId0 < numElements0; v11_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v12_ahead1 = v11_batchId0 + (gridDim.x * blockDim.y);
        size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v11_batchId0][0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v11_batchId0][0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v11_batchId0][0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[v11_batchId0][0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[v11_batchId0][0 + m4_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v27_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v28_i0 = 0; v28_i0 < 1; ++v28_i0) {
            int32_t v31_lead = v27_lead + (v28_i0 * 32);
            #pragma unroll
            for (int32_t v29_i1 = 0; v29_i1 < 9; ++v29_i1) {
              float v34_data = __ldcg(&glb_m0[(v31_lead + (v29_i1 * 32))]);
              r0[(v28_i0 + v29_i1)] = v34_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m1);
          bool v37_g = v27_lead < 16;
          if (v37_g) {
            #pragma unroll
            for (int32_t v38_i1 = 0; v38_i1 < 9; ++v38_i1) {
              float v43_data = __ldcg(&glb_m1[(v27_lead + (v38_i1 * 16))]);
              r2[v38_i1] = v43_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 9)] []
          float v46_data = r0[0];
          float v47_data = r1[0];
          r1[0] = (v47_data + v46_data);
          float v49_data = r0[1];
          float v50_data = r1[1];
          r1[1] = (v50_data + v49_data);
          float v52_data = r0[2];
          float v53_data = r1[2];
          r1[2] = (v53_data + v52_data);
          float v55_data = r0[3];
          float v56_data = r1[3];
          r1[3] = (v56_data + v55_data);
          float v58_data = r0[4];
          float v59_data = r1[4];
          r1[4] = (v59_data + v58_data);
          float v61_data = r0[5];
          float v62_data = r1[5];
          r1[5] = (v62_data + v61_data);
          float v64_data = r0[6];
          float v65_data = r1[6];
          r1[6] = (v65_data + v64_data);
          float v67_data = r0[7];
          float v68_data = r1[7];
          r1[7] = (v68_data + v67_data);
          float v70_data = r0[8];
          float v71_data = r1[8];
          r1[8] = (v71_data + v70_data);
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          if (v37_g) {
            #pragma unroll
            for (int32_t v74_i1 = 0; v74_i1 < 9; ++v74_i1) {
              float v79_data = __ldcg(&glb_m2[(v27_lead + (v74_i1 * 16))]);
              r4[v74_i1] = v79_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          // ir3 = +(r2)
          // [(0, 16), (0, 9)] []
          float ir3[9]{};
          float v83_data = r2[0];
          float v84_data = ir3[0];
          ir3[0] = (v84_data + v83_data);
          float v86_data = r2[1];
          float v87_data = ir3[1];
          ir3[1] = (v87_data + v86_data);
          float v89_data = r2[2];
          float v90_data = ir3[2];
          ir3[2] = (v90_data + v89_data);
          float v92_data = r2[3];
          float v93_data = ir3[3];
          ir3[3] = (v93_data + v92_data);
          float v95_data = r2[4];
          float v96_data = ir3[4];
          ir3[4] = (v96_data + v95_data);
          float v98_data = r2[5];
          float v99_data = ir3[5];
          ir3[5] = (v99_data + v98_data);
          float v101_data = r2[6];
          float v102_data = ir3[6];
          ir3[6] = (v102_data + v101_data);
          float v104_data = r2[7];
          float v105_data = ir3[7];
          ir3[7] = (v105_data + v104_data);
          float v107_data = r2[8];
          float v108_data = ir3[8];
          ir3[8] = (v108_data + v107_data);
          // r3 = ir3 + r1
          #pragma unroll
          for (int32_t v110_n1 = 0; v110_n1 < 9; ++v110_n1) {
            float v112_data = ir3[v110_n1];
            float v113_data = r1[v110_n1];
            r3[v110_n1] = (v113_data + v112_data);
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
          // ir5 = +(r4)
          // [(0, 16), (0, 9)] []
          float ir5[9]{};
          float v120_data = r4[0];
          float v121_data = ir5[0];
          ir5[0] = (v121_data + v120_data);
          float v123_data = r4[1];
          float v124_data = ir5[1];
          ir5[1] = (v124_data + v123_data);
          float v126_data = r4[2];
          float v127_data = ir5[2];
          ir5[2] = (v127_data + v126_data);
          float v129_data = r4[3];
          float v130_data = ir5[3];
          ir5[3] = (v130_data + v129_data);
          float v132_data = r4[4];
          float v133_data = ir5[4];
          ir5[4] = (v133_data + v132_data);
          float v135_data = r4[5];
          float v136_data = ir5[5];
          ir5[5] = (v136_data + v135_data);
          float v138_data = r4[6];
          float v139_data = ir5[6];
          ir5[6] = (v139_data + v138_data);
          float v141_data = r4[7];
          float v142_data = ir5[7];
          ir5[7] = (v142_data + v141_data);
          float v144_data = r4[8];
          float v145_data = ir5[8];
          ir5[8] = (v145_data + v144_data);
          // r5 = ir5 + r3
          #pragma unroll
          for (int32_t v147_n1 = 0; v147_n1 < 9; ++v147_n1) {
            float v149_data = ir5[v147_n1];
            float v150_data = r3[v147_n1];
            r5[v147_n1] = (v150_data + v149_data);
          }
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          float r6[9]{};
          __syncwarp();
          // ir6 = +(r5 * s1)
          // [(0, 32), (0, 9)] [(0, 9)]
          float ir6[9]{};
          float v154_data = r5[0];
          float v155_data = s1[0];
          float v157_data = ir6[0];
          ir6[0] = (v157_data + (v154_data * v155_data));
          float v160_data = s1[9];
          float v162_data = ir6[1];
          ir6[1] = (v162_data + (v154_data * v160_data));
          float v165_data = s1[18];
          float v167_data = ir6[2];
          ir6[2] = (v167_data + (v154_data * v165_data));
          float v170_data = s1[27];
          float v172_data = ir6[3];
          ir6[3] = (v172_data + (v154_data * v170_data));
          float v175_data = s1[36];
          float v177_data = ir6[4];
          ir6[4] = (v177_data + (v154_data * v175_data));
          float v180_data = s1[45];
          float v182_data = ir6[5];
          ir6[5] = (v182_data + (v154_data * v180_data));
          float v185_data = s1[54];
          float v187_data = ir6[6];
          ir6[6] = (v187_data + (v154_data * v185_data));
          float v190_data = s1[63];
          float v192_data = ir6[7];
          ir6[7] = (v192_data + (v154_data * v190_data));
          float v195_data = s1[72];
          float v197_data = ir6[8];
          ir6[8] = (v197_data + (v154_data * v195_data));
          float v199_data = r5[1];
          float v200_data = s1[1];
          float v202_data = ir6[0];
          ir6[0] = (v202_data + (v199_data * v200_data));
          float v205_data = s1[10];
          float v207_data = ir6[1];
          ir6[1] = (v207_data + (v199_data * v205_data));
          float v210_data = s1[19];
          float v212_data = ir6[2];
          ir6[2] = (v212_data + (v199_data * v210_data));
          float v215_data = s1[28];
          float v217_data = ir6[3];
          ir6[3] = (v217_data + (v199_data * v215_data));
          float v220_data = s1[37];
          float v222_data = ir6[4];
          ir6[4] = (v222_data + (v199_data * v220_data));
          float v225_data = s1[46];
          float v227_data = ir6[5];
          ir6[5] = (v227_data + (v199_data * v225_data));
          float v230_data = s1[55];
          float v232_data = ir6[6];
          ir6[6] = (v232_data + (v199_data * v230_data));
          float v235_data = s1[64];
          float v237_data = ir6[7];
          ir6[7] = (v237_data + (v199_data * v235_data));
          float v240_data = s1[73];
          float v242_data = ir6[8];
          ir6[8] = (v242_data + (v199_data * v240_data));
          float v244_data = r5[2];
          float v245_data = s1[2];
          float v247_data = ir6[0];
          ir6[0] = (v247_data + (v244_data * v245_data));
          float v250_data = s1[11];
          float v252_data = ir6[1];
          ir6[1] = (v252_data + (v244_data * v250_data));
          float v255_data = s1[20];
          float v257_data = ir6[2];
          ir6[2] = (v257_data + (v244_data * v255_data));
          float v260_data = s1[29];
          float v262_data = ir6[3];
          ir6[3] = (v262_data + (v244_data * v260_data));
          float v265_data = s1[38];
          float v267_data = ir6[4];
          ir6[4] = (v267_data + (v244_data * v265_data));
          float v270_data = s1[47];
          float v272_data = ir6[5];
          ir6[5] = (v272_data + (v244_data * v270_data));
          float v275_data = s1[56];
          float v277_data = ir6[6];
          ir6[6] = (v277_data + (v244_data * v275_data));
          float v280_data = s1[65];
          float v282_data = ir6[7];
          ir6[7] = (v282_data + (v244_data * v280_data));
          float v285_data = s1[74];
          float v287_data = ir6[8];
          ir6[8] = (v287_data + (v244_data * v285_data));
          float v289_data = r5[3];
          float v290_data = s1[3];
          float v292_data = ir6[0];
          ir6[0] = (v292_data + (v289_data * v290_data));
          float v295_data = s1[12];
          float v297_data = ir6[1];
          ir6[1] = (v297_data + (v289_data * v295_data));
          float v300_data = s1[21];
          float v302_data = ir6[2];
          ir6[2] = (v302_data + (v289_data * v300_data));
          float v305_data = s1[30];
          float v307_data = ir6[3];
          ir6[3] = (v307_data + (v289_data * v305_data));
          float v310_data = s1[39];
          float v312_data = ir6[4];
          ir6[4] = (v312_data + (v289_data * v310_data));
          float v315_data = s1[48];
          float v317_data = ir6[5];
          ir6[5] = (v317_data + (v289_data * v315_data));
          float v320_data = s1[57];
          float v322_data = ir6[6];
          ir6[6] = (v322_data + (v289_data * v320_data));
          float v325_data = s1[66];
          float v327_data = ir6[7];
          ir6[7] = (v327_data + (v289_data * v325_data));
          float v330_data = s1[75];
          float v332_data = ir6[8];
          ir6[8] = (v332_data + (v289_data * v330_data));
          float v334_data = r5[4];
          float v335_data = s1[4];
          float v337_data = ir6[0];
          ir6[0] = (v337_data + (v334_data * v335_data));
          float v340_data = s1[13];
          float v342_data = ir6[1];
          ir6[1] = (v342_data + (v334_data * v340_data));
          float v345_data = s1[22];
          float v347_data = ir6[2];
          ir6[2] = (v347_data + (v334_data * v345_data));
          float v350_data = s1[31];
          float v352_data = ir6[3];
          ir6[3] = (v352_data + (v334_data * v350_data));
          float v355_data = s1[40];
          float v357_data = ir6[4];
          ir6[4] = (v357_data + (v334_data * v355_data));
          float v360_data = s1[49];
          float v362_data = ir6[5];
          ir6[5] = (v362_data + (v334_data * v360_data));
          float v365_data = s1[58];
          float v367_data = ir6[6];
          ir6[6] = (v367_data + (v334_data * v365_data));
          float v370_data = s1[67];
          float v372_data = ir6[7];
          ir6[7] = (v372_data + (v334_data * v370_data));
          float v375_data = s1[76];
          float v377_data = ir6[8];
          ir6[8] = (v377_data + (v334_data * v375_data));
          float v379_data = r5[5];
          float v380_data = s1[5];
          float v382_data = ir6[0];
          ir6[0] = (v382_data + (v379_data * v380_data));
          float v385_data = s1[14];
          float v387_data = ir6[1];
          ir6[1] = (v387_data + (v379_data * v385_data));
          float v390_data = s1[23];
          float v392_data = ir6[2];
          ir6[2] = (v392_data + (v379_data * v390_data));
          float v395_data = s1[32];
          float v397_data = ir6[3];
          ir6[3] = (v397_data + (v379_data * v395_data));
          float v400_data = s1[41];
          float v402_data = ir6[4];
          ir6[4] = (v402_data + (v379_data * v400_data));
          float v405_data = s1[50];
          float v407_data = ir6[5];
          ir6[5] = (v407_data + (v379_data * v405_data));
          float v410_data = s1[59];
          float v412_data = ir6[6];
          ir6[6] = (v412_data + (v379_data * v410_data));
          float v415_data = s1[68];
          float v417_data = ir6[7];
          ir6[7] = (v417_data + (v379_data * v415_data));
          float v420_data = s1[77];
          float v422_data = ir6[8];
          ir6[8] = (v422_data + (v379_data * v420_data));
          float v424_data = r5[6];
          float v425_data = s1[6];
          float v427_data = ir6[0];
          ir6[0] = (v427_data + (v424_data * v425_data));
          float v430_data = s1[15];
          float v432_data = ir6[1];
          ir6[1] = (v432_data + (v424_data * v430_data));
          float v435_data = s1[24];
          float v437_data = ir6[2];
          ir6[2] = (v437_data + (v424_data * v435_data));
          float v440_data = s1[33];
          float v442_data = ir6[3];
          ir6[3] = (v442_data + (v424_data * v440_data));
          float v445_data = s1[42];
          float v447_data = ir6[4];
          ir6[4] = (v447_data + (v424_data * v445_data));
          float v450_data = s1[51];
          float v452_data = ir6[5];
          ir6[5] = (v452_data + (v424_data * v450_data));
          float v455_data = s1[60];
          float v457_data = ir6[6];
          ir6[6] = (v457_data + (v424_data * v455_data));
          float v460_data = s1[69];
          float v462_data = ir6[7];
          ir6[7] = (v462_data + (v424_data * v460_data));
          float v465_data = s1[78];
          float v467_data = ir6[8];
          ir6[8] = (v467_data + (v424_data * v465_data));
          float v469_data = r5[7];
          float v470_data = s1[7];
          float v472_data = ir6[0];
          ir6[0] = (v472_data + (v469_data * v470_data));
          float v475_data = s1[16];
          float v477_data = ir6[1];
          ir6[1] = (v477_data + (v469_data * v475_data));
          float v480_data = s1[25];
          float v482_data = ir6[2];
          ir6[2] = (v482_data + (v469_data * v480_data));
          float v485_data = s1[34];
          float v487_data = ir6[3];
          ir6[3] = (v487_data + (v469_data * v485_data));
          float v490_data = s1[43];
          float v492_data = ir6[4];
          ir6[4] = (v492_data + (v469_data * v490_data));
          float v495_data = s1[52];
          float v497_data = ir6[5];
          ir6[5] = (v497_data + (v469_data * v495_data));
          float v500_data = s1[61];
          float v502_data = ir6[6];
          ir6[6] = (v502_data + (v469_data * v500_data));
          float v505_data = s1[70];
          float v507_data = ir6[7];
          ir6[7] = (v507_data + (v469_data * v505_data));
          float v510_data = s1[79];
          float v512_data = ir6[8];
          ir6[8] = (v512_data + (v469_data * v510_data));
          float v514_data = r5[8];
          float v515_data = s1[8];
          float v517_data = ir6[0];
          ir6[0] = (v517_data + (v514_data * v515_data));
          float v520_data = s1[17];
          float v522_data = ir6[1];
          ir6[1] = (v522_data + (v514_data * v520_data));
          float v525_data = s1[26];
          float v527_data = ir6[2];
          ir6[2] = (v527_data + (v514_data * v525_data));
          float v530_data = s1[35];
          float v532_data = ir6[3];
          ir6[3] = (v532_data + (v514_data * v530_data));
          float v535_data = s1[44];
          float v537_data = ir6[4];
          ir6[4] = (v537_data + (v514_data * v535_data));
          float v540_data = s1[53];
          float v542_data = ir6[5];
          ir6[5] = (v542_data + (v514_data * v540_data));
          float v545_data = s1[62];
          float v547_data = ir6[6];
          ir6[6] = (v547_data + (v514_data * v545_data));
          float v550_data = s1[71];
          float v552_data = ir6[7];
          ir6[7] = (v552_data + (v514_data * v550_data));
          float v555_data = s1[80];
          float v557_data = ir6[8];
          ir6[8] = (v557_data + (v514_data * v555_data));
          // r6 = ir6
          #pragma unroll
          for (int32_t v559_n0 = 0; v559_n0 < 1; ++v559_n0) {
            #pragma unroll
            for (int32_t v560_n1 = 0; v560_n1 < 9; ++v560_n1) {
              int32_t v561_a = v559_n0 + v560_n1;
              float v562_data = ir6[v561_a];
              r6[v561_a] = v562_data;
            }
          }
          // glb_m3 = store{r>g}(r6);
          #pragma unroll
          for (int32_t v563_i0 = 0; v563_i0 < 1; ++v563_i0) {
            int32_t v568_lead = v27_lead + (v563_i0 * 32);
            #pragma unroll
            for (int32_t v564_i1 = 0; v564_i1 < 9; ++v564_i1) {
              float v566_data = r6[(v563_i0 + v564_i1)];
              glb_m3[(v568_lead + (v564_i1 * 32))] = v566_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

