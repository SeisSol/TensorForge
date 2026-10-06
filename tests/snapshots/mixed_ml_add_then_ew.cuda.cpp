// === base name ===
kernel_1755a047e3a917ef

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_1755a047e3a917ef = {{8, 16, 1}, 8, 8, 1, 16, 4608, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_1755a047e3a917ef(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_1755a047e3a917ef(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_1755a047e3a917ef(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_1755a047e3a917ef, block.x * block.y * block.z, 1152 * sizeof(float));
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
void launcher_kernel_1755a047e3a917ef(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_1755a047e3a917ef(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_1755a047e3a917ef, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_1755a047e3a917ef<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_1755a047e3a917ef(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
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
    //   m3 8×8(8×8) {0..8}×{0..8} strided
    //   m4 8×8(8×8) {0..8}×{0..8} strided
    // operations:
    //   t0[i,j] = m0[i,k] × m1[k,j]
    //   t0[i,j] += m2[i,k] × m3[k,j]
    //   C = abs(TMP)
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1152}],"shared_bytes":4608,"shared_elements":1152,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A1","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"A2","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m4","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[72 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s2 = &localShrMem0[0];
      for (size_t v12_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v12_batchId0 < numElements0; v12_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v13_ahead1 = v12_batchId0 + (gridDim.x * blockDim.y);
        size_t v15_batchId1 = (v13_ahead1 < numElements0) ? v13_ahead1 : v12_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v12_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v12_batchId0 * 64 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v12_batchId0 * 64 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v12_batchId0 * 64 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[v12_batchId0 * 64 + 0 + m3_extraOffset];
          float *const __restrict__ glb_m4 = &m4[v12_batchId0 * 64 + 0 + m4_extraOffset];
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
          #pragma unroll
          for (int32_t i = 0; i < 8; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 8], &glb_m1[0 + 0 + 1 * threadIdx.x + i * 8], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m0););
          float r2[8]{};
          // r2 = load{g>r}(glb_m2);
          #pragma unroll
          for (int32_t v39_i0 = 0; v39_i0 < 1; ++v39_i0) {
            int32_t v42_lead = v28_lead + (v39_i0 * 8);
            #pragma unroll
            for (int32_t v40_i1 = 0; v40_i1 < 8; ++v40_i1) {
              float v45_data = __ldcg(&glb_m2[(v42_lead + (v40_i1 * 8))]);
              r2[(v39_i0 + v40_i1)] = v45_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          // r1 = +(r0 * s0) + None
          // [(0, 8), (0, 8)] [(0, 8)]
          float v48_data = r0[0];
          float v49_data = s0[0];
          float v51_data = r1[0];
          r1[0] = (v51_data + (v48_data * v49_data));
          float v54_data = s0[8];
          float v56_data = r1[1];
          r1[1] = (v56_data + (v48_data * v54_data));
          float v59_data = s0[16];
          float v61_data = r1[2];
          r1[2] = (v61_data + (v48_data * v59_data));
          float v64_data = s0[24];
          float v66_data = r1[3];
          r1[3] = (v66_data + (v48_data * v64_data));
          float v69_data = s0[32];
          float v71_data = r1[4];
          r1[4] = (v71_data + (v48_data * v69_data));
          float v74_data = s0[40];
          float v76_data = r1[5];
          r1[5] = (v76_data + (v48_data * v74_data));
          float v79_data = s0[48];
          float v81_data = r1[6];
          r1[6] = (v81_data + (v48_data * v79_data));
          float v84_data = s0[56];
          float v86_data = r1[7];
          r1[7] = (v86_data + (v48_data * v84_data));
          float v88_data = r0[1];
          float v89_data = s0[1];
          float v91_data = r1[0];
          r1[0] = (v91_data + (v88_data * v89_data));
          float v94_data = s0[9];
          float v96_data = r1[1];
          r1[1] = (v96_data + (v88_data * v94_data));
          float v99_data = s0[17];
          float v101_data = r1[2];
          r1[2] = (v101_data + (v88_data * v99_data));
          float v104_data = s0[25];
          float v106_data = r1[3];
          r1[3] = (v106_data + (v88_data * v104_data));
          float v109_data = s0[33];
          float v111_data = r1[4];
          r1[4] = (v111_data + (v88_data * v109_data));
          float v114_data = s0[41];
          float v116_data = r1[5];
          r1[5] = (v116_data + (v88_data * v114_data));
          float v119_data = s0[49];
          float v121_data = r1[6];
          r1[6] = (v121_data + (v88_data * v119_data));
          float v124_data = s0[57];
          float v126_data = r1[7];
          r1[7] = (v126_data + (v88_data * v124_data));
          float v128_data = r0[2];
          float v129_data = s0[2];
          float v131_data = r1[0];
          r1[0] = (v131_data + (v128_data * v129_data));
          float v134_data = s0[10];
          float v136_data = r1[1];
          r1[1] = (v136_data + (v128_data * v134_data));
          float v139_data = s0[18];
          float v141_data = r1[2];
          r1[2] = (v141_data + (v128_data * v139_data));
          float v144_data = s0[26];
          float v146_data = r1[3];
          r1[3] = (v146_data + (v128_data * v144_data));
          float v149_data = s0[34];
          float v151_data = r1[4];
          r1[4] = (v151_data + (v128_data * v149_data));
          float v154_data = s0[42];
          float v156_data = r1[5];
          r1[5] = (v156_data + (v128_data * v154_data));
          float v159_data = s0[50];
          float v161_data = r1[6];
          r1[6] = (v161_data + (v128_data * v159_data));
          float v164_data = s0[58];
          float v166_data = r1[7];
          r1[7] = (v166_data + (v128_data * v164_data));
          float v168_data = r0[3];
          float v169_data = s0[3];
          float v171_data = r1[0];
          r1[0] = (v171_data + (v168_data * v169_data));
          float v174_data = s0[11];
          float v176_data = r1[1];
          r1[1] = (v176_data + (v168_data * v174_data));
          float v179_data = s0[19];
          float v181_data = r1[2];
          r1[2] = (v181_data + (v168_data * v179_data));
          float v184_data = s0[27];
          float v186_data = r1[3];
          r1[3] = (v186_data + (v168_data * v184_data));
          float v189_data = s0[35];
          float v191_data = r1[4];
          r1[4] = (v191_data + (v168_data * v189_data));
          float v194_data = s0[43];
          float v196_data = r1[5];
          r1[5] = (v196_data + (v168_data * v194_data));
          float v199_data = s0[51];
          float v201_data = r1[6];
          r1[6] = (v201_data + (v168_data * v199_data));
          float v204_data = s0[59];
          float v206_data = r1[7];
          r1[7] = (v206_data + (v168_data * v204_data));
          float v208_data = r0[4];
          float v209_data = s0[4];
          float v211_data = r1[0];
          r1[0] = (v211_data + (v208_data * v209_data));
          float v214_data = s0[12];
          float v216_data = r1[1];
          r1[1] = (v216_data + (v208_data * v214_data));
          float v219_data = s0[20];
          float v221_data = r1[2];
          r1[2] = (v221_data + (v208_data * v219_data));
          float v224_data = s0[28];
          float v226_data = r1[3];
          r1[3] = (v226_data + (v208_data * v224_data));
          float v229_data = s0[36];
          float v231_data = r1[4];
          r1[4] = (v231_data + (v208_data * v229_data));
          float v234_data = s0[44];
          float v236_data = r1[5];
          r1[5] = (v236_data + (v208_data * v234_data));
          float v239_data = s0[52];
          float v241_data = r1[6];
          r1[6] = (v241_data + (v208_data * v239_data));
          float v244_data = s0[60];
          float v246_data = r1[7];
          r1[7] = (v246_data + (v208_data * v244_data));
          float v248_data = r0[5];
          float v249_data = s0[5];
          float v251_data = r1[0];
          r1[0] = (v251_data + (v248_data * v249_data));
          float v254_data = s0[13];
          float v256_data = r1[1];
          r1[1] = (v256_data + (v248_data * v254_data));
          float v259_data = s0[21];
          float v261_data = r1[2];
          r1[2] = (v261_data + (v248_data * v259_data));
          float v264_data = s0[29];
          float v266_data = r1[3];
          r1[3] = (v266_data + (v248_data * v264_data));
          float v269_data = s0[37];
          float v271_data = r1[4];
          r1[4] = (v271_data + (v248_data * v269_data));
          float v274_data = s0[45];
          float v276_data = r1[5];
          r1[5] = (v276_data + (v248_data * v274_data));
          float v279_data = s0[53];
          float v281_data = r1[6];
          r1[6] = (v281_data + (v248_data * v279_data));
          float v284_data = s0[61];
          float v286_data = r1[7];
          r1[7] = (v286_data + (v248_data * v284_data));
          float v288_data = r0[6];
          float v289_data = s0[6];
          float v291_data = r1[0];
          r1[0] = (v291_data + (v288_data * v289_data));
          float v294_data = s0[14];
          float v296_data = r1[1];
          r1[1] = (v296_data + (v288_data * v294_data));
          float v299_data = s0[22];
          float v301_data = r1[2];
          r1[2] = (v301_data + (v288_data * v299_data));
          float v304_data = s0[30];
          float v306_data = r1[3];
          r1[3] = (v306_data + (v288_data * v304_data));
          float v309_data = s0[38];
          float v311_data = r1[4];
          r1[4] = (v311_data + (v288_data * v309_data));
          float v314_data = s0[46];
          float v316_data = r1[5];
          r1[5] = (v316_data + (v288_data * v314_data));
          float v319_data = s0[54];
          float v321_data = r1[6];
          r1[6] = (v321_data + (v288_data * v319_data));
          float v324_data = s0[62];
          float v326_data = r1[7];
          r1[7] = (v326_data + (v288_data * v324_data));
          float v328_data = r0[7];
          float v329_data = s0[7];
          float v331_data = r1[0];
          r1[0] = (v331_data + (v328_data * v329_data));
          float v334_data = s0[15];
          float v336_data = r1[1];
          r1[1] = (v336_data + (v328_data * v334_data));
          float v339_data = s0[23];
          float v341_data = r1[2];
          r1[2] = (v341_data + (v328_data * v339_data));
          float v344_data = s0[31];
          float v346_data = r1[3];
          r1[3] = (v346_data + (v328_data * v344_data));
          float v349_data = s0[39];
          float v351_data = r1[4];
          r1[4] = (v351_data + (v328_data * v349_data));
          float v354_data = s0[47];
          float v356_data = r1[5];
          r1[5] = (v356_data + (v328_data * v354_data));
          float v359_data = s0[55];
          float v361_data = r1[6];
          r1[6] = (v361_data + (v328_data * v359_data));
          float v364_data = s0[63];
          float v366_data = r1[7];
          r1[7] = (v366_data + (v328_data * v364_data));
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          // s2 = load{g>s}(glb_m3[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 8; i += 1) {
            __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + i * 8], &glb_m3[0 + 0 + 1 * threadIdx.x + i * 8], 4);
          }
          __pipeline_commit();
          // wait(r2 = load{g>r}(glb_m2););
          // wait(s2 = load{g>s}(glb_m3[0, 1]));
          __pipeline_wait_prior(0);
          float r3[8]{};
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          // ir3 = +(r2 * s2)
          // [(0, 8), (0, 8)] [(0, 8)]
          float ir3[8]{};
          float v371_data = r2[0];
          float v372_data = s2[0];
          float v374_data = ir3[0];
          ir3[0] = (v374_data + (v371_data * v372_data));
          float v377_data = s2[8];
          float v379_data = ir3[1];
          ir3[1] = (v379_data + (v371_data * v377_data));
          float v382_data = s2[16];
          float v384_data = ir3[2];
          ir3[2] = (v384_data + (v371_data * v382_data));
          float v387_data = s2[24];
          float v389_data = ir3[3];
          ir3[3] = (v389_data + (v371_data * v387_data));
          float v392_data = s2[32];
          float v394_data = ir3[4];
          ir3[4] = (v394_data + (v371_data * v392_data));
          float v397_data = s2[40];
          float v399_data = ir3[5];
          ir3[5] = (v399_data + (v371_data * v397_data));
          float v402_data = s2[48];
          float v404_data = ir3[6];
          ir3[6] = (v404_data + (v371_data * v402_data));
          float v407_data = s2[56];
          float v409_data = ir3[7];
          ir3[7] = (v409_data + (v371_data * v407_data));
          float v411_data = r2[1];
          float v412_data = s2[1];
          float v414_data = ir3[0];
          ir3[0] = (v414_data + (v411_data * v412_data));
          float v417_data = s2[9];
          float v419_data = ir3[1];
          ir3[1] = (v419_data + (v411_data * v417_data));
          float v422_data = s2[17];
          float v424_data = ir3[2];
          ir3[2] = (v424_data + (v411_data * v422_data));
          float v427_data = s2[25];
          float v429_data = ir3[3];
          ir3[3] = (v429_data + (v411_data * v427_data));
          float v432_data = s2[33];
          float v434_data = ir3[4];
          ir3[4] = (v434_data + (v411_data * v432_data));
          float v437_data = s2[41];
          float v439_data = ir3[5];
          ir3[5] = (v439_data + (v411_data * v437_data));
          float v442_data = s2[49];
          float v444_data = ir3[6];
          ir3[6] = (v444_data + (v411_data * v442_data));
          float v447_data = s2[57];
          float v449_data = ir3[7];
          ir3[7] = (v449_data + (v411_data * v447_data));
          float v451_data = r2[2];
          float v452_data = s2[2];
          float v454_data = ir3[0];
          ir3[0] = (v454_data + (v451_data * v452_data));
          float v457_data = s2[10];
          float v459_data = ir3[1];
          ir3[1] = (v459_data + (v451_data * v457_data));
          float v462_data = s2[18];
          float v464_data = ir3[2];
          ir3[2] = (v464_data + (v451_data * v462_data));
          float v467_data = s2[26];
          float v469_data = ir3[3];
          ir3[3] = (v469_data + (v451_data * v467_data));
          float v472_data = s2[34];
          float v474_data = ir3[4];
          ir3[4] = (v474_data + (v451_data * v472_data));
          float v477_data = s2[42];
          float v479_data = ir3[5];
          ir3[5] = (v479_data + (v451_data * v477_data));
          float v482_data = s2[50];
          float v484_data = ir3[6];
          ir3[6] = (v484_data + (v451_data * v482_data));
          float v487_data = s2[58];
          float v489_data = ir3[7];
          ir3[7] = (v489_data + (v451_data * v487_data));
          float v491_data = r2[3];
          float v492_data = s2[3];
          float v494_data = ir3[0];
          ir3[0] = (v494_data + (v491_data * v492_data));
          float v497_data = s2[11];
          float v499_data = ir3[1];
          ir3[1] = (v499_data + (v491_data * v497_data));
          float v502_data = s2[19];
          float v504_data = ir3[2];
          ir3[2] = (v504_data + (v491_data * v502_data));
          float v507_data = s2[27];
          float v509_data = ir3[3];
          ir3[3] = (v509_data + (v491_data * v507_data));
          float v512_data = s2[35];
          float v514_data = ir3[4];
          ir3[4] = (v514_data + (v491_data * v512_data));
          float v517_data = s2[43];
          float v519_data = ir3[5];
          ir3[5] = (v519_data + (v491_data * v517_data));
          float v522_data = s2[51];
          float v524_data = ir3[6];
          ir3[6] = (v524_data + (v491_data * v522_data));
          float v527_data = s2[59];
          float v529_data = ir3[7];
          ir3[7] = (v529_data + (v491_data * v527_data));
          float v531_data = r2[4];
          float v532_data = s2[4];
          float v534_data = ir3[0];
          ir3[0] = (v534_data + (v531_data * v532_data));
          float v537_data = s2[12];
          float v539_data = ir3[1];
          ir3[1] = (v539_data + (v531_data * v537_data));
          float v542_data = s2[20];
          float v544_data = ir3[2];
          ir3[2] = (v544_data + (v531_data * v542_data));
          float v547_data = s2[28];
          float v549_data = ir3[3];
          ir3[3] = (v549_data + (v531_data * v547_data));
          float v552_data = s2[36];
          float v554_data = ir3[4];
          ir3[4] = (v554_data + (v531_data * v552_data));
          float v557_data = s2[44];
          float v559_data = ir3[5];
          ir3[5] = (v559_data + (v531_data * v557_data));
          float v562_data = s2[52];
          float v564_data = ir3[6];
          ir3[6] = (v564_data + (v531_data * v562_data));
          float v567_data = s2[60];
          float v569_data = ir3[7];
          ir3[7] = (v569_data + (v531_data * v567_data));
          float v571_data = r2[5];
          float v572_data = s2[5];
          float v574_data = ir3[0];
          ir3[0] = (v574_data + (v571_data * v572_data));
          float v577_data = s2[13];
          float v579_data = ir3[1];
          ir3[1] = (v579_data + (v571_data * v577_data));
          float v582_data = s2[21];
          float v584_data = ir3[2];
          ir3[2] = (v584_data + (v571_data * v582_data));
          float v587_data = s2[29];
          float v589_data = ir3[3];
          ir3[3] = (v589_data + (v571_data * v587_data));
          float v592_data = s2[37];
          float v594_data = ir3[4];
          ir3[4] = (v594_data + (v571_data * v592_data));
          float v597_data = s2[45];
          float v599_data = ir3[5];
          ir3[5] = (v599_data + (v571_data * v597_data));
          float v602_data = s2[53];
          float v604_data = ir3[6];
          ir3[6] = (v604_data + (v571_data * v602_data));
          float v607_data = s2[61];
          float v609_data = ir3[7];
          ir3[7] = (v609_data + (v571_data * v607_data));
          float v611_data = r2[6];
          float v612_data = s2[6];
          float v614_data = ir3[0];
          ir3[0] = (v614_data + (v611_data * v612_data));
          float v617_data = s2[14];
          float v619_data = ir3[1];
          ir3[1] = (v619_data + (v611_data * v617_data));
          float v622_data = s2[22];
          float v624_data = ir3[2];
          ir3[2] = (v624_data + (v611_data * v622_data));
          float v627_data = s2[30];
          float v629_data = ir3[3];
          ir3[3] = (v629_data + (v611_data * v627_data));
          float v632_data = s2[38];
          float v634_data = ir3[4];
          ir3[4] = (v634_data + (v611_data * v632_data));
          float v637_data = s2[46];
          float v639_data = ir3[5];
          ir3[5] = (v639_data + (v611_data * v637_data));
          float v642_data = s2[54];
          float v644_data = ir3[6];
          ir3[6] = (v644_data + (v611_data * v642_data));
          float v647_data = s2[62];
          float v649_data = ir3[7];
          ir3[7] = (v649_data + (v611_data * v647_data));
          float v651_data = r2[7];
          float v652_data = s2[7];
          float v654_data = ir3[0];
          ir3[0] = (v654_data + (v651_data * v652_data));
          float v657_data = s2[15];
          float v659_data = ir3[1];
          ir3[1] = (v659_data + (v651_data * v657_data));
          float v662_data = s2[23];
          float v664_data = ir3[2];
          ir3[2] = (v664_data + (v651_data * v662_data));
          float v667_data = s2[31];
          float v669_data = ir3[3];
          ir3[3] = (v669_data + (v651_data * v667_data));
          float v672_data = s2[39];
          float v674_data = ir3[4];
          ir3[4] = (v674_data + (v651_data * v672_data));
          float v677_data = s2[47];
          float v679_data = ir3[5];
          ir3[5] = (v679_data + (v651_data * v677_data));
          float v682_data = s2[55];
          float v684_data = ir3[6];
          ir3[6] = (v684_data + (v651_data * v682_data));
          float v687_data = s2[63];
          float v689_data = ir3[7];
          ir3[7] = (v689_data + (v651_data * v687_data));
          // r3 = ir3 + r1
          #pragma unroll
          for (int32_t v691_n0 = 0; v691_n0 < 1; ++v691_n0) {
            #pragma unroll
            for (int32_t v692_n1 = 0; v692_n1 < 8; ++v692_n1) {
              int32_t v693_a = v691_n0 + v692_n1;
              float v694_data = ir3[v693_a];
              float v695_data = r1[v693_a];
              r3[v693_a] = (v695_data + v694_data);
            }
          }
          // glb_m4 = abs(r3)
          #pragma unroll
          for (int32_t v697_k0 = 0; v697_k0 < 1; ++v697_k0) {
            int32_t v703_lead = v28_lead + (v697_k0 * 8);
            #pragma unroll
            for (int32_t v698_k1 = 0; v698_k1 < 8; ++v698_k1) {
              float v700_data = r3[(v697_k0 + v698_k1)];
              glb_m4[(v703_lead + (v698_k1 * 8))] = (fabsf(v700_data));
            }
          }
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
        }
      }
    }
  }
}

