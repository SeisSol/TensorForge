// === base name ===
kernel_359ca056b799bd9a

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_359ca056b799bd9a = {{32, 4, 1}, 32, 32, 1, 4, 1536, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_359ca056b799bd9a(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_359ca056b799bd9a(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_359ca056b799bd9a(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_359ca056b799bd9a, block.x * block.y * block.z, 384 * sizeof(float));
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
void launcher_kernel_359ca056b799bd9a(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_359ca056b799bd9a(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_359ca056b799bd9a, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_359ca056b799bd9a<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_359ca056b799bd9a(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
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
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v8_batchId0][0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v8_batchId0][0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v8_batchId0][0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[v8_batchId0][0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[v8_batchId0][0 + m4_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v24_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v25_i0 = 0; v25_i0 < 1; ++v25_i0) {
            int32_t v28_lead = v24_lead + (v25_i0 * 32);
            #pragma unroll
            for (int32_t v26_i1 = 0; v26_i1 < 9; ++v26_i1) {
              float v31_data = __ldcg(&glb_m0[(v28_lead + (v26_i1 * 32))]);
              r0[(v25_i0 + v26_i1)] = v31_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m1);
          bool v34_g = v24_lead < 16;
          if (v34_g) {
            #pragma unroll
            for (int32_t v35_i1 = 0; v35_i1 < 9; ++v35_i1) {
              float v40_data = __ldcg(&glb_m1[(v24_lead + (v35_i1 * 16))]);
              r2[v35_i1] = v40_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 9)] []
          float v43_data = r0[0];
          float v44_data = r1[0];
          r1[0] = (v44_data + v43_data);
          float v46_data = r0[1];
          float v47_data = r1[1];
          r1[1] = (v47_data + v46_data);
          float v49_data = r0[2];
          float v50_data = r1[2];
          r1[2] = (v50_data + v49_data);
          float v52_data = r0[3];
          float v53_data = r1[3];
          r1[3] = (v53_data + v52_data);
          float v55_data = r0[4];
          float v56_data = r1[4];
          r1[4] = (v56_data + v55_data);
          float v58_data = r0[5];
          float v59_data = r1[5];
          r1[5] = (v59_data + v58_data);
          float v61_data = r0[6];
          float v62_data = r1[6];
          r1[6] = (v62_data + v61_data);
          float v64_data = r0[7];
          float v65_data = r1[7];
          r1[7] = (v65_data + v64_data);
          float v67_data = r0[8];
          float v68_data = r1[8];
          r1[8] = (v68_data + v67_data);
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          if (v34_g) {
            #pragma unroll
            for (int32_t v71_i1 = 0; v71_i1 < 9; ++v71_i1) {
              float v76_data = __ldcg(&glb_m2[(v24_lead + (v71_i1 * 16))]);
              r4[v71_i1] = v76_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          // ir3 = +(r2)
          // [(0, 16), (0, 9)] []
          float ir3[9]{};
          float v80_data = r2[0];
          float v81_data = ir3[0];
          ir3[0] = (v81_data + v80_data);
          float v83_data = r2[1];
          float v84_data = ir3[1];
          ir3[1] = (v84_data + v83_data);
          float v86_data = r2[2];
          float v87_data = ir3[2];
          ir3[2] = (v87_data + v86_data);
          float v89_data = r2[3];
          float v90_data = ir3[3];
          ir3[3] = (v90_data + v89_data);
          float v92_data = r2[4];
          float v93_data = ir3[4];
          ir3[4] = (v93_data + v92_data);
          float v95_data = r2[5];
          float v96_data = ir3[5];
          ir3[5] = (v96_data + v95_data);
          float v98_data = r2[6];
          float v99_data = ir3[6];
          ir3[6] = (v99_data + v98_data);
          float v101_data = r2[7];
          float v102_data = ir3[7];
          ir3[7] = (v102_data + v101_data);
          float v104_data = r2[8];
          float v105_data = ir3[8];
          ir3[8] = (v105_data + v104_data);
          // r3 = ir3 + r1
          #pragma unroll
          for (int32_t v107_n1 = 0; v107_n1 < 9; ++v107_n1) {
            float v109_data = ir3[v107_n1];
            float v110_data = r1[v107_n1];
            r3[v107_n1] = (v110_data + v109_data);
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
          float v117_data = r4[0];
          float v118_data = ir5[0];
          ir5[0] = (v118_data + v117_data);
          float v120_data = r4[1];
          float v121_data = ir5[1];
          ir5[1] = (v121_data + v120_data);
          float v123_data = r4[2];
          float v124_data = ir5[2];
          ir5[2] = (v124_data + v123_data);
          float v126_data = r4[3];
          float v127_data = ir5[3];
          ir5[3] = (v127_data + v126_data);
          float v129_data = r4[4];
          float v130_data = ir5[4];
          ir5[4] = (v130_data + v129_data);
          float v132_data = r4[5];
          float v133_data = ir5[5];
          ir5[5] = (v133_data + v132_data);
          float v135_data = r4[6];
          float v136_data = ir5[6];
          ir5[6] = (v136_data + v135_data);
          float v138_data = r4[7];
          float v139_data = ir5[7];
          ir5[7] = (v139_data + v138_data);
          float v141_data = r4[8];
          float v142_data = ir5[8];
          ir5[8] = (v142_data + v141_data);
          // r5 = ir5 + r3
          #pragma unroll
          for (int32_t v144_n1 = 0; v144_n1 < 9; ++v144_n1) {
            float v146_data = ir5[v144_n1];
            float v147_data = r3[v144_n1];
            r5[v144_n1] = (v147_data + v146_data);
          }
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          float r6[9]{};
          // ir6 = +(r5 * s1)
          // [(0, 32), (0, 9)] [(0, 9)]
          float ir6[9]{};
          float v151_data = r5[0];
          __syncwarp();
          float v152_data = s1[0];
          float v154_data = ir6[0];
          ir6[0] = (v154_data + (v151_data * v152_data));
          float v157_data = s1[9];
          float v159_data = ir6[1];
          ir6[1] = (v159_data + (v151_data * v157_data));
          float v162_data = s1[18];
          float v164_data = ir6[2];
          ir6[2] = (v164_data + (v151_data * v162_data));
          float v167_data = s1[27];
          float v169_data = ir6[3];
          ir6[3] = (v169_data + (v151_data * v167_data));
          float v172_data = s1[36];
          float v174_data = ir6[4];
          ir6[4] = (v174_data + (v151_data * v172_data));
          float v177_data = s1[45];
          float v179_data = ir6[5];
          ir6[5] = (v179_data + (v151_data * v177_data));
          float v182_data = s1[54];
          float v184_data = ir6[6];
          ir6[6] = (v184_data + (v151_data * v182_data));
          float v187_data = s1[63];
          float v189_data = ir6[7];
          ir6[7] = (v189_data + (v151_data * v187_data));
          float v192_data = s1[72];
          float v194_data = ir6[8];
          ir6[8] = (v194_data + (v151_data * v192_data));
          float v196_data = r5[1];
          float v197_data = s1[1];
          float v199_data = ir6[0];
          ir6[0] = (v199_data + (v196_data * v197_data));
          float v202_data = s1[10];
          float v204_data = ir6[1];
          ir6[1] = (v204_data + (v196_data * v202_data));
          float v207_data = s1[19];
          float v209_data = ir6[2];
          ir6[2] = (v209_data + (v196_data * v207_data));
          float v212_data = s1[28];
          float v214_data = ir6[3];
          ir6[3] = (v214_data + (v196_data * v212_data));
          float v217_data = s1[37];
          float v219_data = ir6[4];
          ir6[4] = (v219_data + (v196_data * v217_data));
          float v222_data = s1[46];
          float v224_data = ir6[5];
          ir6[5] = (v224_data + (v196_data * v222_data));
          float v227_data = s1[55];
          float v229_data = ir6[6];
          ir6[6] = (v229_data + (v196_data * v227_data));
          float v232_data = s1[64];
          float v234_data = ir6[7];
          ir6[7] = (v234_data + (v196_data * v232_data));
          float v237_data = s1[73];
          float v239_data = ir6[8];
          ir6[8] = (v239_data + (v196_data * v237_data));
          float v241_data = r5[2];
          float v242_data = s1[2];
          float v244_data = ir6[0];
          ir6[0] = (v244_data + (v241_data * v242_data));
          float v247_data = s1[11];
          float v249_data = ir6[1];
          ir6[1] = (v249_data + (v241_data * v247_data));
          float v252_data = s1[20];
          float v254_data = ir6[2];
          ir6[2] = (v254_data + (v241_data * v252_data));
          float v257_data = s1[29];
          float v259_data = ir6[3];
          ir6[3] = (v259_data + (v241_data * v257_data));
          float v262_data = s1[38];
          float v264_data = ir6[4];
          ir6[4] = (v264_data + (v241_data * v262_data));
          float v267_data = s1[47];
          float v269_data = ir6[5];
          ir6[5] = (v269_data + (v241_data * v267_data));
          float v272_data = s1[56];
          float v274_data = ir6[6];
          ir6[6] = (v274_data + (v241_data * v272_data));
          float v277_data = s1[65];
          float v279_data = ir6[7];
          ir6[7] = (v279_data + (v241_data * v277_data));
          float v282_data = s1[74];
          float v284_data = ir6[8];
          ir6[8] = (v284_data + (v241_data * v282_data));
          float v286_data = r5[3];
          float v287_data = s1[3];
          float v289_data = ir6[0];
          ir6[0] = (v289_data + (v286_data * v287_data));
          float v292_data = s1[12];
          float v294_data = ir6[1];
          ir6[1] = (v294_data + (v286_data * v292_data));
          float v297_data = s1[21];
          float v299_data = ir6[2];
          ir6[2] = (v299_data + (v286_data * v297_data));
          float v302_data = s1[30];
          float v304_data = ir6[3];
          ir6[3] = (v304_data + (v286_data * v302_data));
          float v307_data = s1[39];
          float v309_data = ir6[4];
          ir6[4] = (v309_data + (v286_data * v307_data));
          float v312_data = s1[48];
          float v314_data = ir6[5];
          ir6[5] = (v314_data + (v286_data * v312_data));
          float v317_data = s1[57];
          float v319_data = ir6[6];
          ir6[6] = (v319_data + (v286_data * v317_data));
          float v322_data = s1[66];
          float v324_data = ir6[7];
          ir6[7] = (v324_data + (v286_data * v322_data));
          float v327_data = s1[75];
          float v329_data = ir6[8];
          ir6[8] = (v329_data + (v286_data * v327_data));
          float v331_data = r5[4];
          float v332_data = s1[4];
          float v334_data = ir6[0];
          ir6[0] = (v334_data + (v331_data * v332_data));
          float v337_data = s1[13];
          float v339_data = ir6[1];
          ir6[1] = (v339_data + (v331_data * v337_data));
          float v342_data = s1[22];
          float v344_data = ir6[2];
          ir6[2] = (v344_data + (v331_data * v342_data));
          float v347_data = s1[31];
          float v349_data = ir6[3];
          ir6[3] = (v349_data + (v331_data * v347_data));
          float v352_data = s1[40];
          float v354_data = ir6[4];
          ir6[4] = (v354_data + (v331_data * v352_data));
          float v357_data = s1[49];
          float v359_data = ir6[5];
          ir6[5] = (v359_data + (v331_data * v357_data));
          float v362_data = s1[58];
          float v364_data = ir6[6];
          ir6[6] = (v364_data + (v331_data * v362_data));
          float v367_data = s1[67];
          float v369_data = ir6[7];
          ir6[7] = (v369_data + (v331_data * v367_data));
          float v372_data = s1[76];
          float v374_data = ir6[8];
          ir6[8] = (v374_data + (v331_data * v372_data));
          float v376_data = r5[5];
          float v377_data = s1[5];
          float v379_data = ir6[0];
          ir6[0] = (v379_data + (v376_data * v377_data));
          float v382_data = s1[14];
          float v384_data = ir6[1];
          ir6[1] = (v384_data + (v376_data * v382_data));
          float v387_data = s1[23];
          float v389_data = ir6[2];
          ir6[2] = (v389_data + (v376_data * v387_data));
          float v392_data = s1[32];
          float v394_data = ir6[3];
          ir6[3] = (v394_data + (v376_data * v392_data));
          float v397_data = s1[41];
          float v399_data = ir6[4];
          ir6[4] = (v399_data + (v376_data * v397_data));
          float v402_data = s1[50];
          float v404_data = ir6[5];
          ir6[5] = (v404_data + (v376_data * v402_data));
          float v407_data = s1[59];
          float v409_data = ir6[6];
          ir6[6] = (v409_data + (v376_data * v407_data));
          float v412_data = s1[68];
          float v414_data = ir6[7];
          ir6[7] = (v414_data + (v376_data * v412_data));
          float v417_data = s1[77];
          float v419_data = ir6[8];
          ir6[8] = (v419_data + (v376_data * v417_data));
          float v421_data = r5[6];
          float v422_data = s1[6];
          float v424_data = ir6[0];
          ir6[0] = (v424_data + (v421_data * v422_data));
          float v427_data = s1[15];
          float v429_data = ir6[1];
          ir6[1] = (v429_data + (v421_data * v427_data));
          float v432_data = s1[24];
          float v434_data = ir6[2];
          ir6[2] = (v434_data + (v421_data * v432_data));
          float v437_data = s1[33];
          float v439_data = ir6[3];
          ir6[3] = (v439_data + (v421_data * v437_data));
          float v442_data = s1[42];
          float v444_data = ir6[4];
          ir6[4] = (v444_data + (v421_data * v442_data));
          float v447_data = s1[51];
          float v449_data = ir6[5];
          ir6[5] = (v449_data + (v421_data * v447_data));
          float v452_data = s1[60];
          float v454_data = ir6[6];
          ir6[6] = (v454_data + (v421_data * v452_data));
          float v457_data = s1[69];
          float v459_data = ir6[7];
          ir6[7] = (v459_data + (v421_data * v457_data));
          float v462_data = s1[78];
          float v464_data = ir6[8];
          ir6[8] = (v464_data + (v421_data * v462_data));
          float v466_data = r5[7];
          float v467_data = s1[7];
          float v469_data = ir6[0];
          ir6[0] = (v469_data + (v466_data * v467_data));
          float v472_data = s1[16];
          float v474_data = ir6[1];
          ir6[1] = (v474_data + (v466_data * v472_data));
          float v477_data = s1[25];
          float v479_data = ir6[2];
          ir6[2] = (v479_data + (v466_data * v477_data));
          float v482_data = s1[34];
          float v484_data = ir6[3];
          ir6[3] = (v484_data + (v466_data * v482_data));
          float v487_data = s1[43];
          float v489_data = ir6[4];
          ir6[4] = (v489_data + (v466_data * v487_data));
          float v492_data = s1[52];
          float v494_data = ir6[5];
          ir6[5] = (v494_data + (v466_data * v492_data));
          float v497_data = s1[61];
          float v499_data = ir6[6];
          ir6[6] = (v499_data + (v466_data * v497_data));
          float v502_data = s1[70];
          float v504_data = ir6[7];
          ir6[7] = (v504_data + (v466_data * v502_data));
          float v507_data = s1[79];
          float v509_data = ir6[8];
          ir6[8] = (v509_data + (v466_data * v507_data));
          float v511_data = r5[8];
          float v512_data = s1[8];
          float v514_data = ir6[0];
          ir6[0] = (v514_data + (v511_data * v512_data));
          float v517_data = s1[17];
          float v519_data = ir6[1];
          ir6[1] = (v519_data + (v511_data * v517_data));
          float v522_data = s1[26];
          float v524_data = ir6[2];
          ir6[2] = (v524_data + (v511_data * v522_data));
          float v527_data = s1[35];
          float v529_data = ir6[3];
          ir6[3] = (v529_data + (v511_data * v527_data));
          float v532_data = s1[44];
          float v534_data = ir6[4];
          ir6[4] = (v534_data + (v511_data * v532_data));
          float v537_data = s1[53];
          float v539_data = ir6[5];
          ir6[5] = (v539_data + (v511_data * v537_data));
          float v542_data = s1[62];
          float v544_data = ir6[6];
          ir6[6] = (v544_data + (v511_data * v542_data));
          float v547_data = s1[71];
          float v549_data = ir6[7];
          ir6[7] = (v549_data + (v511_data * v547_data));
          float v552_data = s1[80];
          float v554_data = ir6[8];
          ir6[8] = (v554_data + (v511_data * v552_data));
          // r6 = ir6
          #pragma unroll
          for (int32_t v556_n0 = 0; v556_n0 < 1; ++v556_n0) {
            #pragma unroll
            for (int32_t v557_n1 = 0; v557_n1 < 9; ++v557_n1) {
              int32_t v558_a = v556_n0 + v557_n1;
              float v559_data = ir6[v558_a];
              r6[v558_a] = v559_data;
            }
          }
          // glb_m3 = store{r>g}(r6);
          #pragma unroll
          for (int32_t v560_i0 = 0; v560_i0 < 1; ++v560_i0) {
            int32_t v565_lead = v24_lead + (v560_i0 * 32);
            #pragma unroll
            for (int32_t v561_i1 = 0; v561_i1 < 9; ++v561_i1) {
              float v563_data = r6[(v560_i0 + v561_i1)];
              glb_m3[(v565_lead + (v561_i1 * 32))] = v563_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

