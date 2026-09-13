// === base name ===
kernel_d2fac10f9bd6ae7f

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_d2fac10f9bd6ae7f = {{32, 4, 1}, 32, 32, 1, 4, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_d2fac10f9bd6ae7f(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_d2fac10f9bd6ae7f(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_d2fac10f9bd6ae7f(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_d2fac10f9bd6ae7f, block.x * block.y * block.z, 256 * sizeof(float));
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
void launcher_kernel_d2fac10f9bd6ae7f(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_d2fac10f9bd6ae7f(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_d2fac10f9bd6ae7f, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_d2fac10f9bd6ae7f<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_d2fac10f9bd6ae7f(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 4 per block = block 32x4x1, 1024 B shared, occupancy grid
    // operands:
    //   m0 8×8(8×8) {0..8}×{0..8} strided
    //   m1 8×8(8×8) {0..8}×{0..8} strided
    //   m2 8×8(8×8) {0..8}×{0..8} strided
    //   m3 8×8(8×8) {0..8}×{0..8} strided
    //   m4 8×8(8×8) {0..8}×{0..8} strided
    // operations:
    //   t0[i,j] = m0[i,k] × m1[k,j]
    //   t0[i,j] += m2[i,k] × m3[k,j]
    //   C = abs(TMP)
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"A1","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"A2","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m4","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[64 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s2 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 64 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 64 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 64 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[v7_batchId0 * 64 + 0 + m3_extraOffset];
          float *const __restrict__ glb_m4 = &m4[v7_batchId0 * 64 + 0 + m4_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v23_lead = threadIdx.x % 32;
          bool v24_g = v23_lead < 8;
          if (v24_g) {
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 8; ++v25_i1) {
              float v30_data = __ldcg(&glb_m0[(v23_lead + (v25_i1 * 8))]);
              r0[v25_i1] = v30_data;
            }
          }
          // s0 = load{g>s}(glb_m1[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m1[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m1[0 + 0 + 1 * threadIdx.x + 32], 4);
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m0););
          float r2[8]{};
          // r2 = load{g>r}(glb_m2);
          if (v24_g) {
            #pragma unroll
            for (int32_t v35_i1 = 0; v35_i1 < 8; ++v35_i1) {
              float v40_data = __ldcg(&glb_m2[(v23_lead + (v35_i1 * 8))]);
              r2[v35_i1] = v40_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 8), (0, 8)] [(0, 8)]
          float v43_data = r0[0];
          float v44_data = s0[0];
          float v46_data = r1[0];
          r1[0] = (v46_data + (v43_data * v44_data));
          float v49_data = s0[8];
          float v51_data = r1[1];
          r1[1] = (v51_data + (v43_data * v49_data));
          float v54_data = s0[16];
          float v56_data = r1[2];
          r1[2] = (v56_data + (v43_data * v54_data));
          float v59_data = s0[24];
          float v61_data = r1[3];
          r1[3] = (v61_data + (v43_data * v59_data));
          float v64_data = s0[32];
          float v66_data = r1[4];
          r1[4] = (v66_data + (v43_data * v64_data));
          float v69_data = s0[40];
          float v71_data = r1[5];
          r1[5] = (v71_data + (v43_data * v69_data));
          float v74_data = s0[48];
          float v76_data = r1[6];
          r1[6] = (v76_data + (v43_data * v74_data));
          float v79_data = s0[56];
          float v81_data = r1[7];
          r1[7] = (v81_data + (v43_data * v79_data));
          float v83_data = r0[1];
          float v84_data = s0[1];
          float v86_data = r1[0];
          r1[0] = (v86_data + (v83_data * v84_data));
          float v89_data = s0[9];
          float v91_data = r1[1];
          r1[1] = (v91_data + (v83_data * v89_data));
          float v94_data = s0[17];
          float v96_data = r1[2];
          r1[2] = (v96_data + (v83_data * v94_data));
          float v99_data = s0[25];
          float v101_data = r1[3];
          r1[3] = (v101_data + (v83_data * v99_data));
          float v104_data = s0[33];
          float v106_data = r1[4];
          r1[4] = (v106_data + (v83_data * v104_data));
          float v109_data = s0[41];
          float v111_data = r1[5];
          r1[5] = (v111_data + (v83_data * v109_data));
          float v114_data = s0[49];
          float v116_data = r1[6];
          r1[6] = (v116_data + (v83_data * v114_data));
          float v119_data = s0[57];
          float v121_data = r1[7];
          r1[7] = (v121_data + (v83_data * v119_data));
          float v123_data = r0[2];
          float v124_data = s0[2];
          float v126_data = r1[0];
          r1[0] = (v126_data + (v123_data * v124_data));
          float v129_data = s0[10];
          float v131_data = r1[1];
          r1[1] = (v131_data + (v123_data * v129_data));
          float v134_data = s0[18];
          float v136_data = r1[2];
          r1[2] = (v136_data + (v123_data * v134_data));
          float v139_data = s0[26];
          float v141_data = r1[3];
          r1[3] = (v141_data + (v123_data * v139_data));
          float v144_data = s0[34];
          float v146_data = r1[4];
          r1[4] = (v146_data + (v123_data * v144_data));
          float v149_data = s0[42];
          float v151_data = r1[5];
          r1[5] = (v151_data + (v123_data * v149_data));
          float v154_data = s0[50];
          float v156_data = r1[6];
          r1[6] = (v156_data + (v123_data * v154_data));
          float v159_data = s0[58];
          float v161_data = r1[7];
          r1[7] = (v161_data + (v123_data * v159_data));
          float v163_data = r0[3];
          float v164_data = s0[3];
          float v166_data = r1[0];
          r1[0] = (v166_data + (v163_data * v164_data));
          float v169_data = s0[11];
          float v171_data = r1[1];
          r1[1] = (v171_data + (v163_data * v169_data));
          float v174_data = s0[19];
          float v176_data = r1[2];
          r1[2] = (v176_data + (v163_data * v174_data));
          float v179_data = s0[27];
          float v181_data = r1[3];
          r1[3] = (v181_data + (v163_data * v179_data));
          float v184_data = s0[35];
          float v186_data = r1[4];
          r1[4] = (v186_data + (v163_data * v184_data));
          float v189_data = s0[43];
          float v191_data = r1[5];
          r1[5] = (v191_data + (v163_data * v189_data));
          float v194_data = s0[51];
          float v196_data = r1[6];
          r1[6] = (v196_data + (v163_data * v194_data));
          float v199_data = s0[59];
          float v201_data = r1[7];
          r1[7] = (v201_data + (v163_data * v199_data));
          float v203_data = r0[4];
          float v204_data = s0[4];
          float v206_data = r1[0];
          r1[0] = (v206_data + (v203_data * v204_data));
          float v209_data = s0[12];
          float v211_data = r1[1];
          r1[1] = (v211_data + (v203_data * v209_data));
          float v214_data = s0[20];
          float v216_data = r1[2];
          r1[2] = (v216_data + (v203_data * v214_data));
          float v219_data = s0[28];
          float v221_data = r1[3];
          r1[3] = (v221_data + (v203_data * v219_data));
          float v224_data = s0[36];
          float v226_data = r1[4];
          r1[4] = (v226_data + (v203_data * v224_data));
          float v229_data = s0[44];
          float v231_data = r1[5];
          r1[5] = (v231_data + (v203_data * v229_data));
          float v234_data = s0[52];
          float v236_data = r1[6];
          r1[6] = (v236_data + (v203_data * v234_data));
          float v239_data = s0[60];
          float v241_data = r1[7];
          r1[7] = (v241_data + (v203_data * v239_data));
          float v243_data = r0[5];
          float v244_data = s0[5];
          float v246_data = r1[0];
          r1[0] = (v246_data + (v243_data * v244_data));
          float v249_data = s0[13];
          float v251_data = r1[1];
          r1[1] = (v251_data + (v243_data * v249_data));
          float v254_data = s0[21];
          float v256_data = r1[2];
          r1[2] = (v256_data + (v243_data * v254_data));
          float v259_data = s0[29];
          float v261_data = r1[3];
          r1[3] = (v261_data + (v243_data * v259_data));
          float v264_data = s0[37];
          float v266_data = r1[4];
          r1[4] = (v266_data + (v243_data * v264_data));
          float v269_data = s0[45];
          float v271_data = r1[5];
          r1[5] = (v271_data + (v243_data * v269_data));
          float v274_data = s0[53];
          float v276_data = r1[6];
          r1[6] = (v276_data + (v243_data * v274_data));
          float v279_data = s0[61];
          float v281_data = r1[7];
          r1[7] = (v281_data + (v243_data * v279_data));
          float v283_data = r0[6];
          float v284_data = s0[6];
          float v286_data = r1[0];
          r1[0] = (v286_data + (v283_data * v284_data));
          float v289_data = s0[14];
          float v291_data = r1[1];
          r1[1] = (v291_data + (v283_data * v289_data));
          float v294_data = s0[22];
          float v296_data = r1[2];
          r1[2] = (v296_data + (v283_data * v294_data));
          float v299_data = s0[30];
          float v301_data = r1[3];
          r1[3] = (v301_data + (v283_data * v299_data));
          float v304_data = s0[38];
          float v306_data = r1[4];
          r1[4] = (v306_data + (v283_data * v304_data));
          float v309_data = s0[46];
          float v311_data = r1[5];
          r1[5] = (v311_data + (v283_data * v309_data));
          float v314_data = s0[54];
          float v316_data = r1[6];
          r1[6] = (v316_data + (v283_data * v314_data));
          float v319_data = s0[62];
          float v321_data = r1[7];
          r1[7] = (v321_data + (v283_data * v319_data));
          float v323_data = r0[7];
          float v324_data = s0[7];
          float v326_data = r1[0];
          r1[0] = (v326_data + (v323_data * v324_data));
          float v329_data = s0[15];
          float v331_data = r1[1];
          r1[1] = (v331_data + (v323_data * v329_data));
          float v334_data = s0[23];
          float v336_data = r1[2];
          r1[2] = (v336_data + (v323_data * v334_data));
          float v339_data = s0[31];
          float v341_data = r1[3];
          r1[3] = (v341_data + (v323_data * v339_data));
          float v344_data = s0[39];
          float v346_data = r1[4];
          r1[4] = (v346_data + (v323_data * v344_data));
          float v349_data = s0[47];
          float v351_data = r1[5];
          r1[5] = (v351_data + (v323_data * v349_data));
          float v354_data = s0[55];
          float v356_data = r1[6];
          r1[6] = (v356_data + (v323_data * v354_data));
          float v359_data = s0[63];
          float v361_data = r1[7];
          r1[7] = (v361_data + (v323_data * v359_data));
          __syncwarp();
          // s2 = load{g>s}(glb_m3[0, 1])
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 0], &glb_m3[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 32], &glb_m3[0 + 0 + 1 * threadIdx.x + 32], 4);
          __pipeline_commit();
          // wait(r2 = load{g>r}(glb_m2););
          // wait(s2 = load{g>s}(glb_m3[0, 1]));
          __pipeline_wait_prior(0);
          float r3[8]{};
          __syncwarp();
          // r3 = +(r2 * s2) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 8), (0, 8)] [(0, 8)]
          float ir3[8]{};
          float v367_data = r2[0];
          float v368_data = s2[0];
          float v370_data = ir3[0];
          ir3[0] = (v370_data + (v367_data * v368_data));
          float v373_data = s2[8];
          float v375_data = ir3[1];
          ir3[1] = (v375_data + (v367_data * v373_data));
          float v378_data = s2[16];
          float v380_data = ir3[2];
          ir3[2] = (v380_data + (v367_data * v378_data));
          float v383_data = s2[24];
          float v385_data = ir3[3];
          ir3[3] = (v385_data + (v367_data * v383_data));
          float v388_data = s2[32];
          float v390_data = ir3[4];
          ir3[4] = (v390_data + (v367_data * v388_data));
          float v393_data = s2[40];
          float v395_data = ir3[5];
          ir3[5] = (v395_data + (v367_data * v393_data));
          float v398_data = s2[48];
          float v400_data = ir3[6];
          ir3[6] = (v400_data + (v367_data * v398_data));
          float v403_data = s2[56];
          float v405_data = ir3[7];
          ir3[7] = (v405_data + (v367_data * v403_data));
          float v407_data = r2[1];
          float v408_data = s2[1];
          float v410_data = ir3[0];
          ir3[0] = (v410_data + (v407_data * v408_data));
          float v413_data = s2[9];
          float v415_data = ir3[1];
          ir3[1] = (v415_data + (v407_data * v413_data));
          float v418_data = s2[17];
          float v420_data = ir3[2];
          ir3[2] = (v420_data + (v407_data * v418_data));
          float v423_data = s2[25];
          float v425_data = ir3[3];
          ir3[3] = (v425_data + (v407_data * v423_data));
          float v428_data = s2[33];
          float v430_data = ir3[4];
          ir3[4] = (v430_data + (v407_data * v428_data));
          float v433_data = s2[41];
          float v435_data = ir3[5];
          ir3[5] = (v435_data + (v407_data * v433_data));
          float v438_data = s2[49];
          float v440_data = ir3[6];
          ir3[6] = (v440_data + (v407_data * v438_data));
          float v443_data = s2[57];
          float v445_data = ir3[7];
          ir3[7] = (v445_data + (v407_data * v443_data));
          float v447_data = r2[2];
          float v448_data = s2[2];
          float v450_data = ir3[0];
          ir3[0] = (v450_data + (v447_data * v448_data));
          float v453_data = s2[10];
          float v455_data = ir3[1];
          ir3[1] = (v455_data + (v447_data * v453_data));
          float v458_data = s2[18];
          float v460_data = ir3[2];
          ir3[2] = (v460_data + (v447_data * v458_data));
          float v463_data = s2[26];
          float v465_data = ir3[3];
          ir3[3] = (v465_data + (v447_data * v463_data));
          float v468_data = s2[34];
          float v470_data = ir3[4];
          ir3[4] = (v470_data + (v447_data * v468_data));
          float v473_data = s2[42];
          float v475_data = ir3[5];
          ir3[5] = (v475_data + (v447_data * v473_data));
          float v478_data = s2[50];
          float v480_data = ir3[6];
          ir3[6] = (v480_data + (v447_data * v478_data));
          float v483_data = s2[58];
          float v485_data = ir3[7];
          ir3[7] = (v485_data + (v447_data * v483_data));
          float v487_data = r2[3];
          float v488_data = s2[3];
          float v490_data = ir3[0];
          ir3[0] = (v490_data + (v487_data * v488_data));
          float v493_data = s2[11];
          float v495_data = ir3[1];
          ir3[1] = (v495_data + (v487_data * v493_data));
          float v498_data = s2[19];
          float v500_data = ir3[2];
          ir3[2] = (v500_data + (v487_data * v498_data));
          float v503_data = s2[27];
          float v505_data = ir3[3];
          ir3[3] = (v505_data + (v487_data * v503_data));
          float v508_data = s2[35];
          float v510_data = ir3[4];
          ir3[4] = (v510_data + (v487_data * v508_data));
          float v513_data = s2[43];
          float v515_data = ir3[5];
          ir3[5] = (v515_data + (v487_data * v513_data));
          float v518_data = s2[51];
          float v520_data = ir3[6];
          ir3[6] = (v520_data + (v487_data * v518_data));
          float v523_data = s2[59];
          float v525_data = ir3[7];
          ir3[7] = (v525_data + (v487_data * v523_data));
          float v527_data = r2[4];
          float v528_data = s2[4];
          float v530_data = ir3[0];
          ir3[0] = (v530_data + (v527_data * v528_data));
          float v533_data = s2[12];
          float v535_data = ir3[1];
          ir3[1] = (v535_data + (v527_data * v533_data));
          float v538_data = s2[20];
          float v540_data = ir3[2];
          ir3[2] = (v540_data + (v527_data * v538_data));
          float v543_data = s2[28];
          float v545_data = ir3[3];
          ir3[3] = (v545_data + (v527_data * v543_data));
          float v548_data = s2[36];
          float v550_data = ir3[4];
          ir3[4] = (v550_data + (v527_data * v548_data));
          float v553_data = s2[44];
          float v555_data = ir3[5];
          ir3[5] = (v555_data + (v527_data * v553_data));
          float v558_data = s2[52];
          float v560_data = ir3[6];
          ir3[6] = (v560_data + (v527_data * v558_data));
          float v563_data = s2[60];
          float v565_data = ir3[7];
          ir3[7] = (v565_data + (v527_data * v563_data));
          float v567_data = r2[5];
          float v568_data = s2[5];
          float v570_data = ir3[0];
          ir3[0] = (v570_data + (v567_data * v568_data));
          float v573_data = s2[13];
          float v575_data = ir3[1];
          ir3[1] = (v575_data + (v567_data * v573_data));
          float v578_data = s2[21];
          float v580_data = ir3[2];
          ir3[2] = (v580_data + (v567_data * v578_data));
          float v583_data = s2[29];
          float v585_data = ir3[3];
          ir3[3] = (v585_data + (v567_data * v583_data));
          float v588_data = s2[37];
          float v590_data = ir3[4];
          ir3[4] = (v590_data + (v567_data * v588_data));
          float v593_data = s2[45];
          float v595_data = ir3[5];
          ir3[5] = (v595_data + (v567_data * v593_data));
          float v598_data = s2[53];
          float v600_data = ir3[6];
          ir3[6] = (v600_data + (v567_data * v598_data));
          float v603_data = s2[61];
          float v605_data = ir3[7];
          ir3[7] = (v605_data + (v567_data * v603_data));
          float v607_data = r2[6];
          float v608_data = s2[6];
          float v610_data = ir3[0];
          ir3[0] = (v610_data + (v607_data * v608_data));
          float v613_data = s2[14];
          float v615_data = ir3[1];
          ir3[1] = (v615_data + (v607_data * v613_data));
          float v618_data = s2[22];
          float v620_data = ir3[2];
          ir3[2] = (v620_data + (v607_data * v618_data));
          float v623_data = s2[30];
          float v625_data = ir3[3];
          ir3[3] = (v625_data + (v607_data * v623_data));
          float v628_data = s2[38];
          float v630_data = ir3[4];
          ir3[4] = (v630_data + (v607_data * v628_data));
          float v633_data = s2[46];
          float v635_data = ir3[5];
          ir3[5] = (v635_data + (v607_data * v633_data));
          float v638_data = s2[54];
          float v640_data = ir3[6];
          ir3[6] = (v640_data + (v607_data * v638_data));
          float v643_data = s2[62];
          float v645_data = ir3[7];
          ir3[7] = (v645_data + (v607_data * v643_data));
          float v647_data = r2[7];
          float v648_data = s2[7];
          float v650_data = ir3[0];
          ir3[0] = (v650_data + (v647_data * v648_data));
          float v653_data = s2[15];
          float v655_data = ir3[1];
          ir3[1] = (v655_data + (v647_data * v653_data));
          float v658_data = s2[23];
          float v660_data = ir3[2];
          ir3[2] = (v660_data + (v647_data * v658_data));
          float v663_data = s2[31];
          float v665_data = ir3[3];
          ir3[3] = (v665_data + (v647_data * v663_data));
          float v668_data = s2[39];
          float v670_data = ir3[4];
          ir3[4] = (v670_data + (v647_data * v668_data));
          float v673_data = s2[47];
          float v675_data = ir3[5];
          ir3[5] = (v675_data + (v647_data * v673_data));
          float v678_data = s2[55];
          float v680_data = ir3[6];
          ir3[6] = (v680_data + (v647_data * v678_data));
          float v683_data = s2[63];
          float v685_data = ir3[7];
          ir3[7] = (v685_data + (v647_data * v683_data));
          if (v24_g) {
            #pragma unroll
            for (int32_t v687_n1 = 0; v687_n1 < 8; ++v687_n1) {
              float v689_data = ir3[v687_n1];
              float v690_data = r1[v687_n1];
              r3[v687_n1] = (v690_data + v689_data);
            }
          }
          __syncwarp();
          // s1 = store{r>s}(localShrMem0, r3);
          if (v24_g) {
            #pragma unroll
            for (int32_t v692_i1 = 0; v692_i1 < 8; ++v692_i1) {
              float v694_data = r3[v692_i1];
              int32_t v698_a = v23_lead + (v692_i1 * 8);
              s1[(v698_a ^ ((v698_a >> 5) & 31))] = v694_data;
            }
          }
          __syncwarp();
          // glb_m4 = abs(s1)
          if (v24_g) {
            #pragma unroll
            for (int32_t v702_k1 = 0; v702_k1 < 8; ++v702_k1) {
              int32_t v706_a = v23_lead + (v702_k1 * 8);
              float v710_data = s1[(v706_a ^ ((v706_a >> 5) & 31))];
              glb_m4[v706_a] = (fabsf(v710_data));
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

