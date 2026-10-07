// === base name ===
kernel_cca8dfed9fb34d1c

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_cca8dfed9fb34d1c = {{16, 8, 1}, 16, 12, 1, 8, 3584, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_cca8dfed9fb34d1c(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_cca8dfed9fb34d1c(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_cca8dfed9fb34d1c(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_cca8dfed9fb34d1c, block.x * block.y * block.z, 896 * sizeof(float));
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
void launcher_kernel_cca8dfed9fb34d1c(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_cca8dfed9fb34d1c(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_cca8dfed9fb34d1c, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_cca8dfed9fb34d1c<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, m6, m6_extraOffset, m7, m7_extraOffset, m8, m8_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_cca8dfed9fb34d1c(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 8 per block = block 16x8x1, 3584 B shared, occupancy grid
    // operands:
    //   m0 12×8(12×8) {0..12}×{0..8} strided
    //   m1 12×12(12×12) {0..12}×{0..12} strided
    //   m2 12×8(12×8) {0..12}×{0..8} strided
    //   m3 12×12(12×12) {0..12}×{0..12} strided
    //   m4 12×8(12×8) {0..12}×{0..8} strided
    //   m5 12×12(12×12) {0..12}×{0..12} strided
    //   m6 12×8(12×8) {0..12}×{0..8} strided
    //   m7 12×12(12×12) {0..12}×{0..12} strided
    //   m8 12×8(12×8) {0..12}×{0..8} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    //   m0[i,j] += m3[i,k] × m4[k,j]
    //   m0[i,j] += m5[i,k] × m6[k,j]
    //   m0[i,j] += m7[i,k] × m8[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":896}],"shared_bytes":3584,"shared_elements":896,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[12,8]],"name":"m0","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A0","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B0","bbox":[[0,0],[12,8]],"name":"m2","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A1","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[12,8]],"name":"m4","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A2","bbox":[[0,0],[12,12]],"name":"m5","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[12,8]],"name":"m6","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A3","bbox":[[0,0],[12,12]],"name":"m7","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B3","bbox":[[0,0],[12,8]],"name":"m8","ordered":false,"parts":1,"shape":[12,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m7","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m8","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[112 * threadIdx.y + 0];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      float * __restrict__ s2 = &localShrMem0[0];
      float * __restrict__ s3 = &localShrMem0[0];
      for (size_t v11_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v11_batchId0 < numElements0; v11_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v12_ahead1 = v11_batchId0 + (gridDim.x * blockDim.y);
        size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v11_batchId0 * 96 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v11_batchId0 * 144 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v11_batchId0 * 96 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[v11_batchId0 * 144 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[v11_batchId0 * 96 + 0 + m4_extraOffset];
          const float *const __restrict__ glb_m5 = &m5[v11_batchId0 * 144 + 0 + m5_extraOffset];
          const float *const __restrict__ glb_m6 = &m6[v11_batchId0 * 96 + 0 + m6_extraOffset];
          const float *const __restrict__ glb_m7 = &m7[v11_batchId0 * 144 + 0 + m7_extraOffset];
          const float *const __restrict__ glb_m8 = &m8[v11_batchId0 * 96 + 0 + m8_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v31_lead = threadIdx.x % 16;
          bool v32_g = v31_lead < 12;
          if (v32_g) {
            #pragma unroll
            for (int32_t v33_i1 = 0; v33_i1 < 12; ++v33_i1) {
              float v38_data = __ldcg(&glb_m1[(v31_lead + (v33_i1 * 12))]);
              r0[v33_i1] = v38_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 6; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
          __pipeline_commit();
          float r2[12]{};
          // r2 = load{g>r}(glb_m3);
          if (v32_g) {
            #pragma unroll
            for (int32_t v527_i1 = 0; v527_i1 < 12; ++v527_i1) {
              float v532_data = __ldcg(&glb_m3[(v31_lead + (v527_i1 * 12))]);
              r2[v527_i1] = v532_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          // ir1 = +(r0 * s0)
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir1[8]{};
          float v43_data = r0[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v44_data = s0[0];
          float v46_data = ir1[0];
          ir1[0] = (v46_data + (v43_data * v44_data));
          float v49_data = s0[12];
          float v51_data = ir1[1];
          ir1[1] = (v51_data + (v43_data * v49_data));
          float v54_data = s0[24];
          float v56_data = ir1[2];
          ir1[2] = (v56_data + (v43_data * v54_data));
          float v59_data = s0[36];
          float v61_data = ir1[3];
          ir1[3] = (v61_data + (v43_data * v59_data));
          float v64_data = s0[48];
          float v66_data = ir1[4];
          ir1[4] = (v66_data + (v43_data * v64_data));
          float v69_data = s0[60];
          float v71_data = ir1[5];
          ir1[5] = (v71_data + (v43_data * v69_data));
          float v74_data = s0[72];
          float v76_data = ir1[6];
          ir1[6] = (v76_data + (v43_data * v74_data));
          float v79_data = s0[84];
          float v81_data = ir1[7];
          ir1[7] = (v81_data + (v43_data * v79_data));
          float v83_data = r0[1];
          float v84_data = s0[1];
          float v86_data = ir1[0];
          ir1[0] = (v86_data + (v83_data * v84_data));
          float v89_data = s0[13];
          float v91_data = ir1[1];
          ir1[1] = (v91_data + (v83_data * v89_data));
          float v94_data = s0[25];
          float v96_data = ir1[2];
          ir1[2] = (v96_data + (v83_data * v94_data));
          float v99_data = s0[37];
          float v101_data = ir1[3];
          ir1[3] = (v101_data + (v83_data * v99_data));
          float v104_data = s0[49];
          float v106_data = ir1[4];
          ir1[4] = (v106_data + (v83_data * v104_data));
          float v109_data = s0[61];
          float v111_data = ir1[5];
          ir1[5] = (v111_data + (v83_data * v109_data));
          float v114_data = s0[73];
          float v116_data = ir1[6];
          ir1[6] = (v116_data + (v83_data * v114_data));
          float v119_data = s0[85];
          float v121_data = ir1[7];
          ir1[7] = (v121_data + (v83_data * v119_data));
          float v123_data = r0[2];
          float v124_data = s0[2];
          float v126_data = ir1[0];
          ir1[0] = (v126_data + (v123_data * v124_data));
          float v129_data = s0[14];
          float v131_data = ir1[1];
          ir1[1] = (v131_data + (v123_data * v129_data));
          float v134_data = s0[26];
          float v136_data = ir1[2];
          ir1[2] = (v136_data + (v123_data * v134_data));
          float v139_data = s0[38];
          float v141_data = ir1[3];
          ir1[3] = (v141_data + (v123_data * v139_data));
          float v144_data = s0[50];
          float v146_data = ir1[4];
          ir1[4] = (v146_data + (v123_data * v144_data));
          float v149_data = s0[62];
          float v151_data = ir1[5];
          ir1[5] = (v151_data + (v123_data * v149_data));
          float v154_data = s0[74];
          float v156_data = ir1[6];
          ir1[6] = (v156_data + (v123_data * v154_data));
          float v159_data = s0[86];
          float v161_data = ir1[7];
          ir1[7] = (v161_data + (v123_data * v159_data));
          float v163_data = r0[3];
          float v164_data = s0[3];
          float v166_data = ir1[0];
          ir1[0] = (v166_data + (v163_data * v164_data));
          float v169_data = s0[15];
          float v171_data = ir1[1];
          ir1[1] = (v171_data + (v163_data * v169_data));
          float v174_data = s0[27];
          float v176_data = ir1[2];
          ir1[2] = (v176_data + (v163_data * v174_data));
          float v179_data = s0[39];
          float v181_data = ir1[3];
          ir1[3] = (v181_data + (v163_data * v179_data));
          float v184_data = s0[51];
          float v186_data = ir1[4];
          ir1[4] = (v186_data + (v163_data * v184_data));
          float v189_data = s0[63];
          float v191_data = ir1[5];
          ir1[5] = (v191_data + (v163_data * v189_data));
          float v194_data = s0[75];
          float v196_data = ir1[6];
          ir1[6] = (v196_data + (v163_data * v194_data));
          float v199_data = s0[87];
          float v201_data = ir1[7];
          ir1[7] = (v201_data + (v163_data * v199_data));
          float v203_data = r0[4];
          float v204_data = s0[4];
          float v206_data = ir1[0];
          ir1[0] = (v206_data + (v203_data * v204_data));
          float v209_data = s0[16];
          float v211_data = ir1[1];
          ir1[1] = (v211_data + (v203_data * v209_data));
          float v214_data = s0[28];
          float v216_data = ir1[2];
          ir1[2] = (v216_data + (v203_data * v214_data));
          float v219_data = s0[40];
          float v221_data = ir1[3];
          ir1[3] = (v221_data + (v203_data * v219_data));
          float v224_data = s0[52];
          float v226_data = ir1[4];
          ir1[4] = (v226_data + (v203_data * v224_data));
          float v229_data = s0[64];
          float v231_data = ir1[5];
          ir1[5] = (v231_data + (v203_data * v229_data));
          float v234_data = s0[76];
          float v236_data = ir1[6];
          ir1[6] = (v236_data + (v203_data * v234_data));
          float v239_data = s0[88];
          float v241_data = ir1[7];
          ir1[7] = (v241_data + (v203_data * v239_data));
          float v243_data = r0[5];
          float v244_data = s0[5];
          float v246_data = ir1[0];
          ir1[0] = (v246_data + (v243_data * v244_data));
          float v249_data = s0[17];
          float v251_data = ir1[1];
          ir1[1] = (v251_data + (v243_data * v249_data));
          float v254_data = s0[29];
          float v256_data = ir1[2];
          ir1[2] = (v256_data + (v243_data * v254_data));
          float v259_data = s0[41];
          float v261_data = ir1[3];
          ir1[3] = (v261_data + (v243_data * v259_data));
          float v264_data = s0[53];
          float v266_data = ir1[4];
          ir1[4] = (v266_data + (v243_data * v264_data));
          float v269_data = s0[65];
          float v271_data = ir1[5];
          ir1[5] = (v271_data + (v243_data * v269_data));
          float v274_data = s0[77];
          float v276_data = ir1[6];
          ir1[6] = (v276_data + (v243_data * v274_data));
          float v279_data = s0[89];
          float v281_data = ir1[7];
          ir1[7] = (v281_data + (v243_data * v279_data));
          float v283_data = r0[6];
          float v284_data = s0[6];
          float v286_data = ir1[0];
          ir1[0] = (v286_data + (v283_data * v284_data));
          float v289_data = s0[18];
          float v291_data = ir1[1];
          ir1[1] = (v291_data + (v283_data * v289_data));
          float v294_data = s0[30];
          float v296_data = ir1[2];
          ir1[2] = (v296_data + (v283_data * v294_data));
          float v299_data = s0[42];
          float v301_data = ir1[3];
          ir1[3] = (v301_data + (v283_data * v299_data));
          float v304_data = s0[54];
          float v306_data = ir1[4];
          ir1[4] = (v306_data + (v283_data * v304_data));
          float v309_data = s0[66];
          float v311_data = ir1[5];
          ir1[5] = (v311_data + (v283_data * v309_data));
          float v314_data = s0[78];
          float v316_data = ir1[6];
          ir1[6] = (v316_data + (v283_data * v314_data));
          float v319_data = s0[90];
          float v321_data = ir1[7];
          ir1[7] = (v321_data + (v283_data * v319_data));
          float v323_data = r0[7];
          float v324_data = s0[7];
          float v326_data = ir1[0];
          ir1[0] = (v326_data + (v323_data * v324_data));
          float v329_data = s0[19];
          float v331_data = ir1[1];
          ir1[1] = (v331_data + (v323_data * v329_data));
          float v334_data = s0[31];
          float v336_data = ir1[2];
          ir1[2] = (v336_data + (v323_data * v334_data));
          float v339_data = s0[43];
          float v341_data = ir1[3];
          ir1[3] = (v341_data + (v323_data * v339_data));
          float v344_data = s0[55];
          float v346_data = ir1[4];
          ir1[4] = (v346_data + (v323_data * v344_data));
          float v349_data = s0[67];
          float v351_data = ir1[5];
          ir1[5] = (v351_data + (v323_data * v349_data));
          float v354_data = s0[79];
          float v356_data = ir1[6];
          ir1[6] = (v356_data + (v323_data * v354_data));
          float v359_data = s0[91];
          float v361_data = ir1[7];
          ir1[7] = (v361_data + (v323_data * v359_data));
          float v363_data = r0[8];
          float v364_data = s0[8];
          float v366_data = ir1[0];
          ir1[0] = (v366_data + (v363_data * v364_data));
          float v369_data = s0[20];
          float v371_data = ir1[1];
          ir1[1] = (v371_data + (v363_data * v369_data));
          float v374_data = s0[32];
          float v376_data = ir1[2];
          ir1[2] = (v376_data + (v363_data * v374_data));
          float v379_data = s0[44];
          float v381_data = ir1[3];
          ir1[3] = (v381_data + (v363_data * v379_data));
          float v384_data = s0[56];
          float v386_data = ir1[4];
          ir1[4] = (v386_data + (v363_data * v384_data));
          float v389_data = s0[68];
          float v391_data = ir1[5];
          ir1[5] = (v391_data + (v363_data * v389_data));
          float v394_data = s0[80];
          float v396_data = ir1[6];
          ir1[6] = (v396_data + (v363_data * v394_data));
          float v399_data = s0[92];
          float v401_data = ir1[7];
          ir1[7] = (v401_data + (v363_data * v399_data));
          float v403_data = r0[9];
          float v404_data = s0[9];
          float v406_data = ir1[0];
          ir1[0] = (v406_data + (v403_data * v404_data));
          float v409_data = s0[21];
          float v411_data = ir1[1];
          ir1[1] = (v411_data + (v403_data * v409_data));
          float v414_data = s0[33];
          float v416_data = ir1[2];
          ir1[2] = (v416_data + (v403_data * v414_data));
          float v419_data = s0[45];
          float v421_data = ir1[3];
          ir1[3] = (v421_data + (v403_data * v419_data));
          float v424_data = s0[57];
          float v426_data = ir1[4];
          ir1[4] = (v426_data + (v403_data * v424_data));
          float v429_data = s0[69];
          float v431_data = ir1[5];
          ir1[5] = (v431_data + (v403_data * v429_data));
          float v434_data = s0[81];
          float v436_data = ir1[6];
          ir1[6] = (v436_data + (v403_data * v434_data));
          float v439_data = s0[93];
          float v441_data = ir1[7];
          ir1[7] = (v441_data + (v403_data * v439_data));
          float v443_data = r0[10];
          float v444_data = s0[10];
          float v446_data = ir1[0];
          ir1[0] = (v446_data + (v443_data * v444_data));
          float v449_data = s0[22];
          float v451_data = ir1[1];
          ir1[1] = (v451_data + (v443_data * v449_data));
          float v454_data = s0[34];
          float v456_data = ir1[2];
          ir1[2] = (v456_data + (v443_data * v454_data));
          float v459_data = s0[46];
          float v461_data = ir1[3];
          ir1[3] = (v461_data + (v443_data * v459_data));
          float v464_data = s0[58];
          float v466_data = ir1[4];
          ir1[4] = (v466_data + (v443_data * v464_data));
          float v469_data = s0[70];
          float v471_data = ir1[5];
          ir1[5] = (v471_data + (v443_data * v469_data));
          float v474_data = s0[82];
          float v476_data = ir1[6];
          ir1[6] = (v476_data + (v443_data * v474_data));
          float v479_data = s0[94];
          float v481_data = ir1[7];
          ir1[7] = (v481_data + (v443_data * v479_data));
          float v483_data = r0[11];
          float v484_data = s0[11];
          float v486_data = ir1[0];
          ir1[0] = (v486_data + (v483_data * v484_data));
          float v489_data = s0[23];
          float v491_data = ir1[1];
          ir1[1] = (v491_data + (v483_data * v489_data));
          float v494_data = s0[35];
          float v496_data = ir1[2];
          ir1[2] = (v496_data + (v483_data * v494_data));
          float v499_data = s0[47];
          float v501_data = ir1[3];
          ir1[3] = (v501_data + (v483_data * v499_data));
          float v504_data = s0[59];
          float v506_data = ir1[4];
          ir1[4] = (v506_data + (v483_data * v504_data));
          float v509_data = s0[71];
          float v511_data = ir1[5];
          ir1[5] = (v511_data + (v483_data * v509_data));
          float v514_data = s0[83];
          float v516_data = ir1[6];
          ir1[6] = (v516_data + (v483_data * v514_data));
          float v519_data = s0[95];
          float v521_data = ir1[7];
          ir1[7] = (v521_data + (v483_data * v519_data));
          // r1 = ir1
          if (v32_g) {
            #pragma unroll
            for (int32_t v523_n1 = 0; v523_n1 < 8; ++v523_n1) {
              float v525_data = ir1[v523_n1];
              r1[v523_n1] = v525_data;
            }
          }
          // s1 = load{g>s}(glb_m4[0, 1])
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          #pragma unroll
          for (int32_t i = 0; i < 6; i += 1) {
            __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m4[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
          __pipeline_commit();
          float r4[12]{};
          // r4 = load{g>r}(glb_m5);
          if (v32_g) {
            #pragma unroll
            for (int32_t v1023_i1 = 0; v1023_i1 < 12; ++v1023_i1) {
              float v1028_data = __ldcg(&glb_m5[(v31_lead + (v1023_i1 * 12))]);
              r4[v1023_i1] = v1028_data;
            }
          }
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          float r3[8]{};
          // ir3 = +(r2 * s1)
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir3[8]{};
          float v537_data = r2[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v538_data = s1[0];
          float v540_data = ir3[0];
          ir3[0] = (v540_data + (v537_data * v538_data));
          float v543_data = s1[12];
          float v545_data = ir3[1];
          ir3[1] = (v545_data + (v537_data * v543_data));
          float v548_data = s1[24];
          float v550_data = ir3[2];
          ir3[2] = (v550_data + (v537_data * v548_data));
          float v553_data = s1[36];
          float v555_data = ir3[3];
          ir3[3] = (v555_data + (v537_data * v553_data));
          float v558_data = s1[48];
          float v560_data = ir3[4];
          ir3[4] = (v560_data + (v537_data * v558_data));
          float v563_data = s1[60];
          float v565_data = ir3[5];
          ir3[5] = (v565_data + (v537_data * v563_data));
          float v568_data = s1[72];
          float v570_data = ir3[6];
          ir3[6] = (v570_data + (v537_data * v568_data));
          float v573_data = s1[84];
          float v575_data = ir3[7];
          ir3[7] = (v575_data + (v537_data * v573_data));
          float v577_data = r2[1];
          float v578_data = s1[1];
          float v580_data = ir3[0];
          ir3[0] = (v580_data + (v577_data * v578_data));
          float v583_data = s1[13];
          float v585_data = ir3[1];
          ir3[1] = (v585_data + (v577_data * v583_data));
          float v588_data = s1[25];
          float v590_data = ir3[2];
          ir3[2] = (v590_data + (v577_data * v588_data));
          float v593_data = s1[37];
          float v595_data = ir3[3];
          ir3[3] = (v595_data + (v577_data * v593_data));
          float v598_data = s1[49];
          float v600_data = ir3[4];
          ir3[4] = (v600_data + (v577_data * v598_data));
          float v603_data = s1[61];
          float v605_data = ir3[5];
          ir3[5] = (v605_data + (v577_data * v603_data));
          float v608_data = s1[73];
          float v610_data = ir3[6];
          ir3[6] = (v610_data + (v577_data * v608_data));
          float v613_data = s1[85];
          float v615_data = ir3[7];
          ir3[7] = (v615_data + (v577_data * v613_data));
          float v617_data = r2[2];
          float v618_data = s1[2];
          float v620_data = ir3[0];
          ir3[0] = (v620_data + (v617_data * v618_data));
          float v623_data = s1[14];
          float v625_data = ir3[1];
          ir3[1] = (v625_data + (v617_data * v623_data));
          float v628_data = s1[26];
          float v630_data = ir3[2];
          ir3[2] = (v630_data + (v617_data * v628_data));
          float v633_data = s1[38];
          float v635_data = ir3[3];
          ir3[3] = (v635_data + (v617_data * v633_data));
          float v638_data = s1[50];
          float v640_data = ir3[4];
          ir3[4] = (v640_data + (v617_data * v638_data));
          float v643_data = s1[62];
          float v645_data = ir3[5];
          ir3[5] = (v645_data + (v617_data * v643_data));
          float v648_data = s1[74];
          float v650_data = ir3[6];
          ir3[6] = (v650_data + (v617_data * v648_data));
          float v653_data = s1[86];
          float v655_data = ir3[7];
          ir3[7] = (v655_data + (v617_data * v653_data));
          float v657_data = r2[3];
          float v658_data = s1[3];
          float v660_data = ir3[0];
          ir3[0] = (v660_data + (v657_data * v658_data));
          float v663_data = s1[15];
          float v665_data = ir3[1];
          ir3[1] = (v665_data + (v657_data * v663_data));
          float v668_data = s1[27];
          float v670_data = ir3[2];
          ir3[2] = (v670_data + (v657_data * v668_data));
          float v673_data = s1[39];
          float v675_data = ir3[3];
          ir3[3] = (v675_data + (v657_data * v673_data));
          float v678_data = s1[51];
          float v680_data = ir3[4];
          ir3[4] = (v680_data + (v657_data * v678_data));
          float v683_data = s1[63];
          float v685_data = ir3[5];
          ir3[5] = (v685_data + (v657_data * v683_data));
          float v688_data = s1[75];
          float v690_data = ir3[6];
          ir3[6] = (v690_data + (v657_data * v688_data));
          float v693_data = s1[87];
          float v695_data = ir3[7];
          ir3[7] = (v695_data + (v657_data * v693_data));
          float v697_data = r2[4];
          float v698_data = s1[4];
          float v700_data = ir3[0];
          ir3[0] = (v700_data + (v697_data * v698_data));
          float v703_data = s1[16];
          float v705_data = ir3[1];
          ir3[1] = (v705_data + (v697_data * v703_data));
          float v708_data = s1[28];
          float v710_data = ir3[2];
          ir3[2] = (v710_data + (v697_data * v708_data));
          float v713_data = s1[40];
          float v715_data = ir3[3];
          ir3[3] = (v715_data + (v697_data * v713_data));
          float v718_data = s1[52];
          float v720_data = ir3[4];
          ir3[4] = (v720_data + (v697_data * v718_data));
          float v723_data = s1[64];
          float v725_data = ir3[5];
          ir3[5] = (v725_data + (v697_data * v723_data));
          float v728_data = s1[76];
          float v730_data = ir3[6];
          ir3[6] = (v730_data + (v697_data * v728_data));
          float v733_data = s1[88];
          float v735_data = ir3[7];
          ir3[7] = (v735_data + (v697_data * v733_data));
          float v737_data = r2[5];
          float v738_data = s1[5];
          float v740_data = ir3[0];
          ir3[0] = (v740_data + (v737_data * v738_data));
          float v743_data = s1[17];
          float v745_data = ir3[1];
          ir3[1] = (v745_data + (v737_data * v743_data));
          float v748_data = s1[29];
          float v750_data = ir3[2];
          ir3[2] = (v750_data + (v737_data * v748_data));
          float v753_data = s1[41];
          float v755_data = ir3[3];
          ir3[3] = (v755_data + (v737_data * v753_data));
          float v758_data = s1[53];
          float v760_data = ir3[4];
          ir3[4] = (v760_data + (v737_data * v758_data));
          float v763_data = s1[65];
          float v765_data = ir3[5];
          ir3[5] = (v765_data + (v737_data * v763_data));
          float v768_data = s1[77];
          float v770_data = ir3[6];
          ir3[6] = (v770_data + (v737_data * v768_data));
          float v773_data = s1[89];
          float v775_data = ir3[7];
          ir3[7] = (v775_data + (v737_data * v773_data));
          float v777_data = r2[6];
          float v778_data = s1[6];
          float v780_data = ir3[0];
          ir3[0] = (v780_data + (v777_data * v778_data));
          float v783_data = s1[18];
          float v785_data = ir3[1];
          ir3[1] = (v785_data + (v777_data * v783_data));
          float v788_data = s1[30];
          float v790_data = ir3[2];
          ir3[2] = (v790_data + (v777_data * v788_data));
          float v793_data = s1[42];
          float v795_data = ir3[3];
          ir3[3] = (v795_data + (v777_data * v793_data));
          float v798_data = s1[54];
          float v800_data = ir3[4];
          ir3[4] = (v800_data + (v777_data * v798_data));
          float v803_data = s1[66];
          float v805_data = ir3[5];
          ir3[5] = (v805_data + (v777_data * v803_data));
          float v808_data = s1[78];
          float v810_data = ir3[6];
          ir3[6] = (v810_data + (v777_data * v808_data));
          float v813_data = s1[90];
          float v815_data = ir3[7];
          ir3[7] = (v815_data + (v777_data * v813_data));
          float v817_data = r2[7];
          float v818_data = s1[7];
          float v820_data = ir3[0];
          ir3[0] = (v820_data + (v817_data * v818_data));
          float v823_data = s1[19];
          float v825_data = ir3[1];
          ir3[1] = (v825_data + (v817_data * v823_data));
          float v828_data = s1[31];
          float v830_data = ir3[2];
          ir3[2] = (v830_data + (v817_data * v828_data));
          float v833_data = s1[43];
          float v835_data = ir3[3];
          ir3[3] = (v835_data + (v817_data * v833_data));
          float v838_data = s1[55];
          float v840_data = ir3[4];
          ir3[4] = (v840_data + (v817_data * v838_data));
          float v843_data = s1[67];
          float v845_data = ir3[5];
          ir3[5] = (v845_data + (v817_data * v843_data));
          float v848_data = s1[79];
          float v850_data = ir3[6];
          ir3[6] = (v850_data + (v817_data * v848_data));
          float v853_data = s1[91];
          float v855_data = ir3[7];
          ir3[7] = (v855_data + (v817_data * v853_data));
          float v857_data = r2[8];
          float v858_data = s1[8];
          float v860_data = ir3[0];
          ir3[0] = (v860_data + (v857_data * v858_data));
          float v863_data = s1[20];
          float v865_data = ir3[1];
          ir3[1] = (v865_data + (v857_data * v863_data));
          float v868_data = s1[32];
          float v870_data = ir3[2];
          ir3[2] = (v870_data + (v857_data * v868_data));
          float v873_data = s1[44];
          float v875_data = ir3[3];
          ir3[3] = (v875_data + (v857_data * v873_data));
          float v878_data = s1[56];
          float v880_data = ir3[4];
          ir3[4] = (v880_data + (v857_data * v878_data));
          float v883_data = s1[68];
          float v885_data = ir3[5];
          ir3[5] = (v885_data + (v857_data * v883_data));
          float v888_data = s1[80];
          float v890_data = ir3[6];
          ir3[6] = (v890_data + (v857_data * v888_data));
          float v893_data = s1[92];
          float v895_data = ir3[7];
          ir3[7] = (v895_data + (v857_data * v893_data));
          float v897_data = r2[9];
          float v898_data = s1[9];
          float v900_data = ir3[0];
          ir3[0] = (v900_data + (v897_data * v898_data));
          float v903_data = s1[21];
          float v905_data = ir3[1];
          ir3[1] = (v905_data + (v897_data * v903_data));
          float v908_data = s1[33];
          float v910_data = ir3[2];
          ir3[2] = (v910_data + (v897_data * v908_data));
          float v913_data = s1[45];
          float v915_data = ir3[3];
          ir3[3] = (v915_data + (v897_data * v913_data));
          float v918_data = s1[57];
          float v920_data = ir3[4];
          ir3[4] = (v920_data + (v897_data * v918_data));
          float v923_data = s1[69];
          float v925_data = ir3[5];
          ir3[5] = (v925_data + (v897_data * v923_data));
          float v928_data = s1[81];
          float v930_data = ir3[6];
          ir3[6] = (v930_data + (v897_data * v928_data));
          float v933_data = s1[93];
          float v935_data = ir3[7];
          ir3[7] = (v935_data + (v897_data * v933_data));
          float v937_data = r2[10];
          float v938_data = s1[10];
          float v940_data = ir3[0];
          ir3[0] = (v940_data + (v937_data * v938_data));
          float v943_data = s1[22];
          float v945_data = ir3[1];
          ir3[1] = (v945_data + (v937_data * v943_data));
          float v948_data = s1[34];
          float v950_data = ir3[2];
          ir3[2] = (v950_data + (v937_data * v948_data));
          float v953_data = s1[46];
          float v955_data = ir3[3];
          ir3[3] = (v955_data + (v937_data * v953_data));
          float v958_data = s1[58];
          float v960_data = ir3[4];
          ir3[4] = (v960_data + (v937_data * v958_data));
          float v963_data = s1[70];
          float v965_data = ir3[5];
          ir3[5] = (v965_data + (v937_data * v963_data));
          float v968_data = s1[82];
          float v970_data = ir3[6];
          ir3[6] = (v970_data + (v937_data * v968_data));
          float v973_data = s1[94];
          float v975_data = ir3[7];
          ir3[7] = (v975_data + (v937_data * v973_data));
          float v977_data = r2[11];
          float v978_data = s1[11];
          float v980_data = ir3[0];
          ir3[0] = (v980_data + (v977_data * v978_data));
          float v983_data = s1[23];
          float v985_data = ir3[1];
          ir3[1] = (v985_data + (v977_data * v983_data));
          float v988_data = s1[35];
          float v990_data = ir3[2];
          ir3[2] = (v990_data + (v977_data * v988_data));
          float v993_data = s1[47];
          float v995_data = ir3[3];
          ir3[3] = (v995_data + (v977_data * v993_data));
          float v998_data = s1[59];
          float v1000_data = ir3[4];
          ir3[4] = (v1000_data + (v977_data * v998_data));
          float v1003_data = s1[71];
          float v1005_data = ir3[5];
          ir3[5] = (v1005_data + (v977_data * v1003_data));
          float v1008_data = s1[83];
          float v1010_data = ir3[6];
          ir3[6] = (v1010_data + (v977_data * v1008_data));
          float v1013_data = s1[95];
          float v1015_data = ir3[7];
          ir3[7] = (v1015_data + (v977_data * v1013_data));
          // r3 = ir3 + r1
          if (v32_g) {
            #pragma unroll
            for (int32_t v1017_n1 = 0; v1017_n1 < 8; ++v1017_n1) {
              float v1019_data = ir3[v1017_n1];
              float v1020_data = r1[v1017_n1];
              r3[v1017_n1] = (v1020_data + v1019_data);
            }
          }
          // s2 = load{g>s}(glb_m6[0, 1])
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          #pragma unroll
          for (int32_t i = 0; i < 6; i += 1) {
            __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m6[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
          __pipeline_commit();
          float r6[12]{};
          // r6 = load{g>r}(glb_m7);
          if (v32_g) {
            #pragma unroll
            for (int32_t v1519_i1 = 0; v1519_i1 < 12; ++v1519_i1) {
              float v1524_data = __ldcg(&glb_m7[(v31_lead + (v1519_i1 * 12))]);
              r6[v1519_i1] = v1524_data;
            }
          }
          // wait(s2 = load{g>s}(glb_m6[0, 1]));
          __pipeline_wait_prior(0);
          float r5[8]{};
          // ir5 = +(r4 * s2)
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir5[8]{};
          float v1033_data = r4[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v1034_data = s2[0];
          float v1036_data = ir5[0];
          ir5[0] = (v1036_data + (v1033_data * v1034_data));
          float v1039_data = s2[12];
          float v1041_data = ir5[1];
          ir5[1] = (v1041_data + (v1033_data * v1039_data));
          float v1044_data = s2[24];
          float v1046_data = ir5[2];
          ir5[2] = (v1046_data + (v1033_data * v1044_data));
          float v1049_data = s2[36];
          float v1051_data = ir5[3];
          ir5[3] = (v1051_data + (v1033_data * v1049_data));
          float v1054_data = s2[48];
          float v1056_data = ir5[4];
          ir5[4] = (v1056_data + (v1033_data * v1054_data));
          float v1059_data = s2[60];
          float v1061_data = ir5[5];
          ir5[5] = (v1061_data + (v1033_data * v1059_data));
          float v1064_data = s2[72];
          float v1066_data = ir5[6];
          ir5[6] = (v1066_data + (v1033_data * v1064_data));
          float v1069_data = s2[84];
          float v1071_data = ir5[7];
          ir5[7] = (v1071_data + (v1033_data * v1069_data));
          float v1073_data = r4[1];
          float v1074_data = s2[1];
          float v1076_data = ir5[0];
          ir5[0] = (v1076_data + (v1073_data * v1074_data));
          float v1079_data = s2[13];
          float v1081_data = ir5[1];
          ir5[1] = (v1081_data + (v1073_data * v1079_data));
          float v1084_data = s2[25];
          float v1086_data = ir5[2];
          ir5[2] = (v1086_data + (v1073_data * v1084_data));
          float v1089_data = s2[37];
          float v1091_data = ir5[3];
          ir5[3] = (v1091_data + (v1073_data * v1089_data));
          float v1094_data = s2[49];
          float v1096_data = ir5[4];
          ir5[4] = (v1096_data + (v1073_data * v1094_data));
          float v1099_data = s2[61];
          float v1101_data = ir5[5];
          ir5[5] = (v1101_data + (v1073_data * v1099_data));
          float v1104_data = s2[73];
          float v1106_data = ir5[6];
          ir5[6] = (v1106_data + (v1073_data * v1104_data));
          float v1109_data = s2[85];
          float v1111_data = ir5[7];
          ir5[7] = (v1111_data + (v1073_data * v1109_data));
          float v1113_data = r4[2];
          float v1114_data = s2[2];
          float v1116_data = ir5[0];
          ir5[0] = (v1116_data + (v1113_data * v1114_data));
          float v1119_data = s2[14];
          float v1121_data = ir5[1];
          ir5[1] = (v1121_data + (v1113_data * v1119_data));
          float v1124_data = s2[26];
          float v1126_data = ir5[2];
          ir5[2] = (v1126_data + (v1113_data * v1124_data));
          float v1129_data = s2[38];
          float v1131_data = ir5[3];
          ir5[3] = (v1131_data + (v1113_data * v1129_data));
          float v1134_data = s2[50];
          float v1136_data = ir5[4];
          ir5[4] = (v1136_data + (v1113_data * v1134_data));
          float v1139_data = s2[62];
          float v1141_data = ir5[5];
          ir5[5] = (v1141_data + (v1113_data * v1139_data));
          float v1144_data = s2[74];
          float v1146_data = ir5[6];
          ir5[6] = (v1146_data + (v1113_data * v1144_data));
          float v1149_data = s2[86];
          float v1151_data = ir5[7];
          ir5[7] = (v1151_data + (v1113_data * v1149_data));
          float v1153_data = r4[3];
          float v1154_data = s2[3];
          float v1156_data = ir5[0];
          ir5[0] = (v1156_data + (v1153_data * v1154_data));
          float v1159_data = s2[15];
          float v1161_data = ir5[1];
          ir5[1] = (v1161_data + (v1153_data * v1159_data));
          float v1164_data = s2[27];
          float v1166_data = ir5[2];
          ir5[2] = (v1166_data + (v1153_data * v1164_data));
          float v1169_data = s2[39];
          float v1171_data = ir5[3];
          ir5[3] = (v1171_data + (v1153_data * v1169_data));
          float v1174_data = s2[51];
          float v1176_data = ir5[4];
          ir5[4] = (v1176_data + (v1153_data * v1174_data));
          float v1179_data = s2[63];
          float v1181_data = ir5[5];
          ir5[5] = (v1181_data + (v1153_data * v1179_data));
          float v1184_data = s2[75];
          float v1186_data = ir5[6];
          ir5[6] = (v1186_data + (v1153_data * v1184_data));
          float v1189_data = s2[87];
          float v1191_data = ir5[7];
          ir5[7] = (v1191_data + (v1153_data * v1189_data));
          float v1193_data = r4[4];
          float v1194_data = s2[4];
          float v1196_data = ir5[0];
          ir5[0] = (v1196_data + (v1193_data * v1194_data));
          float v1199_data = s2[16];
          float v1201_data = ir5[1];
          ir5[1] = (v1201_data + (v1193_data * v1199_data));
          float v1204_data = s2[28];
          float v1206_data = ir5[2];
          ir5[2] = (v1206_data + (v1193_data * v1204_data));
          float v1209_data = s2[40];
          float v1211_data = ir5[3];
          ir5[3] = (v1211_data + (v1193_data * v1209_data));
          float v1214_data = s2[52];
          float v1216_data = ir5[4];
          ir5[4] = (v1216_data + (v1193_data * v1214_data));
          float v1219_data = s2[64];
          float v1221_data = ir5[5];
          ir5[5] = (v1221_data + (v1193_data * v1219_data));
          float v1224_data = s2[76];
          float v1226_data = ir5[6];
          ir5[6] = (v1226_data + (v1193_data * v1224_data));
          float v1229_data = s2[88];
          float v1231_data = ir5[7];
          ir5[7] = (v1231_data + (v1193_data * v1229_data));
          float v1233_data = r4[5];
          float v1234_data = s2[5];
          float v1236_data = ir5[0];
          ir5[0] = (v1236_data + (v1233_data * v1234_data));
          float v1239_data = s2[17];
          float v1241_data = ir5[1];
          ir5[1] = (v1241_data + (v1233_data * v1239_data));
          float v1244_data = s2[29];
          float v1246_data = ir5[2];
          ir5[2] = (v1246_data + (v1233_data * v1244_data));
          float v1249_data = s2[41];
          float v1251_data = ir5[3];
          ir5[3] = (v1251_data + (v1233_data * v1249_data));
          float v1254_data = s2[53];
          float v1256_data = ir5[4];
          ir5[4] = (v1256_data + (v1233_data * v1254_data));
          float v1259_data = s2[65];
          float v1261_data = ir5[5];
          ir5[5] = (v1261_data + (v1233_data * v1259_data));
          float v1264_data = s2[77];
          float v1266_data = ir5[6];
          ir5[6] = (v1266_data + (v1233_data * v1264_data));
          float v1269_data = s2[89];
          float v1271_data = ir5[7];
          ir5[7] = (v1271_data + (v1233_data * v1269_data));
          float v1273_data = r4[6];
          float v1274_data = s2[6];
          float v1276_data = ir5[0];
          ir5[0] = (v1276_data + (v1273_data * v1274_data));
          float v1279_data = s2[18];
          float v1281_data = ir5[1];
          ir5[1] = (v1281_data + (v1273_data * v1279_data));
          float v1284_data = s2[30];
          float v1286_data = ir5[2];
          ir5[2] = (v1286_data + (v1273_data * v1284_data));
          float v1289_data = s2[42];
          float v1291_data = ir5[3];
          ir5[3] = (v1291_data + (v1273_data * v1289_data));
          float v1294_data = s2[54];
          float v1296_data = ir5[4];
          ir5[4] = (v1296_data + (v1273_data * v1294_data));
          float v1299_data = s2[66];
          float v1301_data = ir5[5];
          ir5[5] = (v1301_data + (v1273_data * v1299_data));
          float v1304_data = s2[78];
          float v1306_data = ir5[6];
          ir5[6] = (v1306_data + (v1273_data * v1304_data));
          float v1309_data = s2[90];
          float v1311_data = ir5[7];
          ir5[7] = (v1311_data + (v1273_data * v1309_data));
          float v1313_data = r4[7];
          float v1314_data = s2[7];
          float v1316_data = ir5[0];
          ir5[0] = (v1316_data + (v1313_data * v1314_data));
          float v1319_data = s2[19];
          float v1321_data = ir5[1];
          ir5[1] = (v1321_data + (v1313_data * v1319_data));
          float v1324_data = s2[31];
          float v1326_data = ir5[2];
          ir5[2] = (v1326_data + (v1313_data * v1324_data));
          float v1329_data = s2[43];
          float v1331_data = ir5[3];
          ir5[3] = (v1331_data + (v1313_data * v1329_data));
          float v1334_data = s2[55];
          float v1336_data = ir5[4];
          ir5[4] = (v1336_data + (v1313_data * v1334_data));
          float v1339_data = s2[67];
          float v1341_data = ir5[5];
          ir5[5] = (v1341_data + (v1313_data * v1339_data));
          float v1344_data = s2[79];
          float v1346_data = ir5[6];
          ir5[6] = (v1346_data + (v1313_data * v1344_data));
          float v1349_data = s2[91];
          float v1351_data = ir5[7];
          ir5[7] = (v1351_data + (v1313_data * v1349_data));
          float v1353_data = r4[8];
          float v1354_data = s2[8];
          float v1356_data = ir5[0];
          ir5[0] = (v1356_data + (v1353_data * v1354_data));
          float v1359_data = s2[20];
          float v1361_data = ir5[1];
          ir5[1] = (v1361_data + (v1353_data * v1359_data));
          float v1364_data = s2[32];
          float v1366_data = ir5[2];
          ir5[2] = (v1366_data + (v1353_data * v1364_data));
          float v1369_data = s2[44];
          float v1371_data = ir5[3];
          ir5[3] = (v1371_data + (v1353_data * v1369_data));
          float v1374_data = s2[56];
          float v1376_data = ir5[4];
          ir5[4] = (v1376_data + (v1353_data * v1374_data));
          float v1379_data = s2[68];
          float v1381_data = ir5[5];
          ir5[5] = (v1381_data + (v1353_data * v1379_data));
          float v1384_data = s2[80];
          float v1386_data = ir5[6];
          ir5[6] = (v1386_data + (v1353_data * v1384_data));
          float v1389_data = s2[92];
          float v1391_data = ir5[7];
          ir5[7] = (v1391_data + (v1353_data * v1389_data));
          float v1393_data = r4[9];
          float v1394_data = s2[9];
          float v1396_data = ir5[0];
          ir5[0] = (v1396_data + (v1393_data * v1394_data));
          float v1399_data = s2[21];
          float v1401_data = ir5[1];
          ir5[1] = (v1401_data + (v1393_data * v1399_data));
          float v1404_data = s2[33];
          float v1406_data = ir5[2];
          ir5[2] = (v1406_data + (v1393_data * v1404_data));
          float v1409_data = s2[45];
          float v1411_data = ir5[3];
          ir5[3] = (v1411_data + (v1393_data * v1409_data));
          float v1414_data = s2[57];
          float v1416_data = ir5[4];
          ir5[4] = (v1416_data + (v1393_data * v1414_data));
          float v1419_data = s2[69];
          float v1421_data = ir5[5];
          ir5[5] = (v1421_data + (v1393_data * v1419_data));
          float v1424_data = s2[81];
          float v1426_data = ir5[6];
          ir5[6] = (v1426_data + (v1393_data * v1424_data));
          float v1429_data = s2[93];
          float v1431_data = ir5[7];
          ir5[7] = (v1431_data + (v1393_data * v1429_data));
          float v1433_data = r4[10];
          float v1434_data = s2[10];
          float v1436_data = ir5[0];
          ir5[0] = (v1436_data + (v1433_data * v1434_data));
          float v1439_data = s2[22];
          float v1441_data = ir5[1];
          ir5[1] = (v1441_data + (v1433_data * v1439_data));
          float v1444_data = s2[34];
          float v1446_data = ir5[2];
          ir5[2] = (v1446_data + (v1433_data * v1444_data));
          float v1449_data = s2[46];
          float v1451_data = ir5[3];
          ir5[3] = (v1451_data + (v1433_data * v1449_data));
          float v1454_data = s2[58];
          float v1456_data = ir5[4];
          ir5[4] = (v1456_data + (v1433_data * v1454_data));
          float v1459_data = s2[70];
          float v1461_data = ir5[5];
          ir5[5] = (v1461_data + (v1433_data * v1459_data));
          float v1464_data = s2[82];
          float v1466_data = ir5[6];
          ir5[6] = (v1466_data + (v1433_data * v1464_data));
          float v1469_data = s2[94];
          float v1471_data = ir5[7];
          ir5[7] = (v1471_data + (v1433_data * v1469_data));
          float v1473_data = r4[11];
          float v1474_data = s2[11];
          float v1476_data = ir5[0];
          ir5[0] = (v1476_data + (v1473_data * v1474_data));
          float v1479_data = s2[23];
          float v1481_data = ir5[1];
          ir5[1] = (v1481_data + (v1473_data * v1479_data));
          float v1484_data = s2[35];
          float v1486_data = ir5[2];
          ir5[2] = (v1486_data + (v1473_data * v1484_data));
          float v1489_data = s2[47];
          float v1491_data = ir5[3];
          ir5[3] = (v1491_data + (v1473_data * v1489_data));
          float v1494_data = s2[59];
          float v1496_data = ir5[4];
          ir5[4] = (v1496_data + (v1473_data * v1494_data));
          float v1499_data = s2[71];
          float v1501_data = ir5[5];
          ir5[5] = (v1501_data + (v1473_data * v1499_data));
          float v1504_data = s2[83];
          float v1506_data = ir5[6];
          ir5[6] = (v1506_data + (v1473_data * v1504_data));
          float v1509_data = s2[95];
          float v1511_data = ir5[7];
          ir5[7] = (v1511_data + (v1473_data * v1509_data));
          // r5 = ir5 + r3
          if (v32_g) {
            #pragma unroll
            for (int32_t v1513_n1 = 0; v1513_n1 < 8; ++v1513_n1) {
              float v1515_data = ir5[v1513_n1];
              float v1516_data = r3[v1513_n1];
              r5[v1513_n1] = (v1516_data + v1515_data);
            }
          }
          // s3 = load{g>s}(glb_m8[0, 1])
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          #pragma unroll
          for (int32_t i = 0; i < 6; i += 1) {
            __pipeline_memcpy_async(&s3[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m8[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
          __pipeline_commit();
          // wait(s3 = load{g>s}(glb_m8[0, 1]));
          __pipeline_wait_prior(0);
          float r7[8]{};
          // ir7 = +(r6 * s3)
          // [(0, 12), (0, 8)] [(0, 12)]
          float ir7[8]{};
          float v1529_data = r6[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v1530_data = s3[0];
          float v1532_data = ir7[0];
          ir7[0] = (v1532_data + (v1529_data * v1530_data));
          float v1535_data = s3[12];
          float v1537_data = ir7[1];
          ir7[1] = (v1537_data + (v1529_data * v1535_data));
          float v1540_data = s3[24];
          float v1542_data = ir7[2];
          ir7[2] = (v1542_data + (v1529_data * v1540_data));
          float v1545_data = s3[36];
          float v1547_data = ir7[3];
          ir7[3] = (v1547_data + (v1529_data * v1545_data));
          float v1550_data = s3[48];
          float v1552_data = ir7[4];
          ir7[4] = (v1552_data + (v1529_data * v1550_data));
          float v1555_data = s3[60];
          float v1557_data = ir7[5];
          ir7[5] = (v1557_data + (v1529_data * v1555_data));
          float v1560_data = s3[72];
          float v1562_data = ir7[6];
          ir7[6] = (v1562_data + (v1529_data * v1560_data));
          float v1565_data = s3[84];
          float v1567_data = ir7[7];
          ir7[7] = (v1567_data + (v1529_data * v1565_data));
          float v1569_data = r6[1];
          float v1570_data = s3[1];
          float v1572_data = ir7[0];
          ir7[0] = (v1572_data + (v1569_data * v1570_data));
          float v1575_data = s3[13];
          float v1577_data = ir7[1];
          ir7[1] = (v1577_data + (v1569_data * v1575_data));
          float v1580_data = s3[25];
          float v1582_data = ir7[2];
          ir7[2] = (v1582_data + (v1569_data * v1580_data));
          float v1585_data = s3[37];
          float v1587_data = ir7[3];
          ir7[3] = (v1587_data + (v1569_data * v1585_data));
          float v1590_data = s3[49];
          float v1592_data = ir7[4];
          ir7[4] = (v1592_data + (v1569_data * v1590_data));
          float v1595_data = s3[61];
          float v1597_data = ir7[5];
          ir7[5] = (v1597_data + (v1569_data * v1595_data));
          float v1600_data = s3[73];
          float v1602_data = ir7[6];
          ir7[6] = (v1602_data + (v1569_data * v1600_data));
          float v1605_data = s3[85];
          float v1607_data = ir7[7];
          ir7[7] = (v1607_data + (v1569_data * v1605_data));
          float v1609_data = r6[2];
          float v1610_data = s3[2];
          float v1612_data = ir7[0];
          ir7[0] = (v1612_data + (v1609_data * v1610_data));
          float v1615_data = s3[14];
          float v1617_data = ir7[1];
          ir7[1] = (v1617_data + (v1609_data * v1615_data));
          float v1620_data = s3[26];
          float v1622_data = ir7[2];
          ir7[2] = (v1622_data + (v1609_data * v1620_data));
          float v1625_data = s3[38];
          float v1627_data = ir7[3];
          ir7[3] = (v1627_data + (v1609_data * v1625_data));
          float v1630_data = s3[50];
          float v1632_data = ir7[4];
          ir7[4] = (v1632_data + (v1609_data * v1630_data));
          float v1635_data = s3[62];
          float v1637_data = ir7[5];
          ir7[5] = (v1637_data + (v1609_data * v1635_data));
          float v1640_data = s3[74];
          float v1642_data = ir7[6];
          ir7[6] = (v1642_data + (v1609_data * v1640_data));
          float v1645_data = s3[86];
          float v1647_data = ir7[7];
          ir7[7] = (v1647_data + (v1609_data * v1645_data));
          float v1649_data = r6[3];
          float v1650_data = s3[3];
          float v1652_data = ir7[0];
          ir7[0] = (v1652_data + (v1649_data * v1650_data));
          float v1655_data = s3[15];
          float v1657_data = ir7[1];
          ir7[1] = (v1657_data + (v1649_data * v1655_data));
          float v1660_data = s3[27];
          float v1662_data = ir7[2];
          ir7[2] = (v1662_data + (v1649_data * v1660_data));
          float v1665_data = s3[39];
          float v1667_data = ir7[3];
          ir7[3] = (v1667_data + (v1649_data * v1665_data));
          float v1670_data = s3[51];
          float v1672_data = ir7[4];
          ir7[4] = (v1672_data + (v1649_data * v1670_data));
          float v1675_data = s3[63];
          float v1677_data = ir7[5];
          ir7[5] = (v1677_data + (v1649_data * v1675_data));
          float v1680_data = s3[75];
          float v1682_data = ir7[6];
          ir7[6] = (v1682_data + (v1649_data * v1680_data));
          float v1685_data = s3[87];
          float v1687_data = ir7[7];
          ir7[7] = (v1687_data + (v1649_data * v1685_data));
          float v1689_data = r6[4];
          float v1690_data = s3[4];
          float v1692_data = ir7[0];
          ir7[0] = (v1692_data + (v1689_data * v1690_data));
          float v1695_data = s3[16];
          float v1697_data = ir7[1];
          ir7[1] = (v1697_data + (v1689_data * v1695_data));
          float v1700_data = s3[28];
          float v1702_data = ir7[2];
          ir7[2] = (v1702_data + (v1689_data * v1700_data));
          float v1705_data = s3[40];
          float v1707_data = ir7[3];
          ir7[3] = (v1707_data + (v1689_data * v1705_data));
          float v1710_data = s3[52];
          float v1712_data = ir7[4];
          ir7[4] = (v1712_data + (v1689_data * v1710_data));
          float v1715_data = s3[64];
          float v1717_data = ir7[5];
          ir7[5] = (v1717_data + (v1689_data * v1715_data));
          float v1720_data = s3[76];
          float v1722_data = ir7[6];
          ir7[6] = (v1722_data + (v1689_data * v1720_data));
          float v1725_data = s3[88];
          float v1727_data = ir7[7];
          ir7[7] = (v1727_data + (v1689_data * v1725_data));
          float v1729_data = r6[5];
          float v1730_data = s3[5];
          float v1732_data = ir7[0];
          ir7[0] = (v1732_data + (v1729_data * v1730_data));
          float v1735_data = s3[17];
          float v1737_data = ir7[1];
          ir7[1] = (v1737_data + (v1729_data * v1735_data));
          float v1740_data = s3[29];
          float v1742_data = ir7[2];
          ir7[2] = (v1742_data + (v1729_data * v1740_data));
          float v1745_data = s3[41];
          float v1747_data = ir7[3];
          ir7[3] = (v1747_data + (v1729_data * v1745_data));
          float v1750_data = s3[53];
          float v1752_data = ir7[4];
          ir7[4] = (v1752_data + (v1729_data * v1750_data));
          float v1755_data = s3[65];
          float v1757_data = ir7[5];
          ir7[5] = (v1757_data + (v1729_data * v1755_data));
          float v1760_data = s3[77];
          float v1762_data = ir7[6];
          ir7[6] = (v1762_data + (v1729_data * v1760_data));
          float v1765_data = s3[89];
          float v1767_data = ir7[7];
          ir7[7] = (v1767_data + (v1729_data * v1765_data));
          float v1769_data = r6[6];
          float v1770_data = s3[6];
          float v1772_data = ir7[0];
          ir7[0] = (v1772_data + (v1769_data * v1770_data));
          float v1775_data = s3[18];
          float v1777_data = ir7[1];
          ir7[1] = (v1777_data + (v1769_data * v1775_data));
          float v1780_data = s3[30];
          float v1782_data = ir7[2];
          ir7[2] = (v1782_data + (v1769_data * v1780_data));
          float v1785_data = s3[42];
          float v1787_data = ir7[3];
          ir7[3] = (v1787_data + (v1769_data * v1785_data));
          float v1790_data = s3[54];
          float v1792_data = ir7[4];
          ir7[4] = (v1792_data + (v1769_data * v1790_data));
          float v1795_data = s3[66];
          float v1797_data = ir7[5];
          ir7[5] = (v1797_data + (v1769_data * v1795_data));
          float v1800_data = s3[78];
          float v1802_data = ir7[6];
          ir7[6] = (v1802_data + (v1769_data * v1800_data));
          float v1805_data = s3[90];
          float v1807_data = ir7[7];
          ir7[7] = (v1807_data + (v1769_data * v1805_data));
          float v1809_data = r6[7];
          float v1810_data = s3[7];
          float v1812_data = ir7[0];
          ir7[0] = (v1812_data + (v1809_data * v1810_data));
          float v1815_data = s3[19];
          float v1817_data = ir7[1];
          ir7[1] = (v1817_data + (v1809_data * v1815_data));
          float v1820_data = s3[31];
          float v1822_data = ir7[2];
          ir7[2] = (v1822_data + (v1809_data * v1820_data));
          float v1825_data = s3[43];
          float v1827_data = ir7[3];
          ir7[3] = (v1827_data + (v1809_data * v1825_data));
          float v1830_data = s3[55];
          float v1832_data = ir7[4];
          ir7[4] = (v1832_data + (v1809_data * v1830_data));
          float v1835_data = s3[67];
          float v1837_data = ir7[5];
          ir7[5] = (v1837_data + (v1809_data * v1835_data));
          float v1840_data = s3[79];
          float v1842_data = ir7[6];
          ir7[6] = (v1842_data + (v1809_data * v1840_data));
          float v1845_data = s3[91];
          float v1847_data = ir7[7];
          ir7[7] = (v1847_data + (v1809_data * v1845_data));
          float v1849_data = r6[8];
          float v1850_data = s3[8];
          float v1852_data = ir7[0];
          ir7[0] = (v1852_data + (v1849_data * v1850_data));
          float v1855_data = s3[20];
          float v1857_data = ir7[1];
          ir7[1] = (v1857_data + (v1849_data * v1855_data));
          float v1860_data = s3[32];
          float v1862_data = ir7[2];
          ir7[2] = (v1862_data + (v1849_data * v1860_data));
          float v1865_data = s3[44];
          float v1867_data = ir7[3];
          ir7[3] = (v1867_data + (v1849_data * v1865_data));
          float v1870_data = s3[56];
          float v1872_data = ir7[4];
          ir7[4] = (v1872_data + (v1849_data * v1870_data));
          float v1875_data = s3[68];
          float v1877_data = ir7[5];
          ir7[5] = (v1877_data + (v1849_data * v1875_data));
          float v1880_data = s3[80];
          float v1882_data = ir7[6];
          ir7[6] = (v1882_data + (v1849_data * v1880_data));
          float v1885_data = s3[92];
          float v1887_data = ir7[7];
          ir7[7] = (v1887_data + (v1849_data * v1885_data));
          float v1889_data = r6[9];
          float v1890_data = s3[9];
          float v1892_data = ir7[0];
          ir7[0] = (v1892_data + (v1889_data * v1890_data));
          float v1895_data = s3[21];
          float v1897_data = ir7[1];
          ir7[1] = (v1897_data + (v1889_data * v1895_data));
          float v1900_data = s3[33];
          float v1902_data = ir7[2];
          ir7[2] = (v1902_data + (v1889_data * v1900_data));
          float v1905_data = s3[45];
          float v1907_data = ir7[3];
          ir7[3] = (v1907_data + (v1889_data * v1905_data));
          float v1910_data = s3[57];
          float v1912_data = ir7[4];
          ir7[4] = (v1912_data + (v1889_data * v1910_data));
          float v1915_data = s3[69];
          float v1917_data = ir7[5];
          ir7[5] = (v1917_data + (v1889_data * v1915_data));
          float v1920_data = s3[81];
          float v1922_data = ir7[6];
          ir7[6] = (v1922_data + (v1889_data * v1920_data));
          float v1925_data = s3[93];
          float v1927_data = ir7[7];
          ir7[7] = (v1927_data + (v1889_data * v1925_data));
          float v1929_data = r6[10];
          float v1930_data = s3[10];
          float v1932_data = ir7[0];
          ir7[0] = (v1932_data + (v1929_data * v1930_data));
          float v1935_data = s3[22];
          float v1937_data = ir7[1];
          ir7[1] = (v1937_data + (v1929_data * v1935_data));
          float v1940_data = s3[34];
          float v1942_data = ir7[2];
          ir7[2] = (v1942_data + (v1929_data * v1940_data));
          float v1945_data = s3[46];
          float v1947_data = ir7[3];
          ir7[3] = (v1947_data + (v1929_data * v1945_data));
          float v1950_data = s3[58];
          float v1952_data = ir7[4];
          ir7[4] = (v1952_data + (v1929_data * v1950_data));
          float v1955_data = s3[70];
          float v1957_data = ir7[5];
          ir7[5] = (v1957_data + (v1929_data * v1955_data));
          float v1960_data = s3[82];
          float v1962_data = ir7[6];
          ir7[6] = (v1962_data + (v1929_data * v1960_data));
          float v1965_data = s3[94];
          float v1967_data = ir7[7];
          ir7[7] = (v1967_data + (v1929_data * v1965_data));
          float v1969_data = r6[11];
          float v1970_data = s3[11];
          float v1972_data = ir7[0];
          ir7[0] = (v1972_data + (v1969_data * v1970_data));
          float v1975_data = s3[23];
          float v1977_data = ir7[1];
          ir7[1] = (v1977_data + (v1969_data * v1975_data));
          float v1980_data = s3[35];
          float v1982_data = ir7[2];
          ir7[2] = (v1982_data + (v1969_data * v1980_data));
          float v1985_data = s3[47];
          float v1987_data = ir7[3];
          ir7[3] = (v1987_data + (v1969_data * v1985_data));
          float v1990_data = s3[59];
          float v1992_data = ir7[4];
          ir7[4] = (v1992_data + (v1969_data * v1990_data));
          float v1995_data = s3[71];
          float v1997_data = ir7[5];
          ir7[5] = (v1997_data + (v1969_data * v1995_data));
          float v2000_data = s3[83];
          float v2002_data = ir7[6];
          ir7[6] = (v2002_data + (v1969_data * v2000_data));
          float v2005_data = s3[95];
          float v2007_data = ir7[7];
          ir7[7] = (v2007_data + (v1969_data * v2005_data));
          // r7 = ir7 + r5
          if (v32_g) {
            #pragma unroll
            for (int32_t v2009_n1 = 0; v2009_n1 < 8; ++v2009_n1) {
              float v2011_data = ir7[v2009_n1];
              float v2012_data = r5[v2009_n1];
              r7[v2009_n1] = (v2012_data + v2011_data);
            }
          }
          // glb_m0 = store{r>g}(r7);
          if (v32_g) {
            #pragma unroll
            for (int32_t v2014_i1 = 0; v2014_i1 < 8; ++v2014_i1) {
              float v2016_data = r7[v2014_i1];
              glb_m0[(v31_lead + (v2014_i1 * 12))] = v2016_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

