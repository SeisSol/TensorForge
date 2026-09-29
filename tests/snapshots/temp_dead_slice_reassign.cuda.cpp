// === base name ===
kernel_018a04b5ef6094b9

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_018a04b5ef6094b9 = {{16, 8, 1}, 16, 12, 1, 8, 10752, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_018a04b5ef6094b9(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_018a04b5ef6094b9(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_018a04b5ef6094b9(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_018a04b5ef6094b9, block.x * block.y * block.z, 2688 * sizeof(float));
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
  config.sharedMemBytes = 2688 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_018a04b5ef6094b9(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_018a04b5ef6094b9(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_018a04b5ef6094b9, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_018a04b5ef6094b9<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_018a04b5ef6094b9(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 8 per block = block 16x8x1, 10752 B shared, occupancy grid
    // operands:
    //   m0 6×12(6×12) {0..6}×{0..12} strided
    //   m1 12×12(12×12) {0..12}×{0..12} strided
    //   m2 6×12(6×12) {0..6}×{0..12} strided
    //   m3 6×12(6×12) {0..6}×{0..12} strided
    //   m4 12×12(12×12) {0..12}×{0..12} strided
    // operations:
    //   t0[i,j]@{6..12}×{0..12} = m0[i,k] × m1[k,j]
    //   t0[i,j] = m2[i,k] × m1[k,j]
    //   t0[i,j]@{6..12}×{0..12} = m3[i,j]
    //   m4[i,j] = t0[i,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":2688}],"shared_bytes":10752,"shared_elements":2688,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B2","bbox":[[0,0],[6,12]],"name":"m0","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"N","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"Y","bbox":[[0,0],[6,12]],"name":"m3","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m4","ordered":false,"parts":1,"shape":[12,12],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[6,12]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[336 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[320];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[160];
      for (size_t v6_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v6_batchId0 < numElements0; v6_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v7_ahead1 = v6_batchId0 + (gridDim.x * blockDim.y);
        size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 72 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v6_batchId0 * 144 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v6_batchId0 * 72 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[v6_batchId0 * 72 + 0 + m3_extraOffset];
          float *const __restrict__ glb_m4 = &m4[v6_batchId0 * 144 + 0 + m4_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v22_lead = threadIdx.x % 16;
          bool v23_g = v22_lead < 6;
          if (v23_g) {
            #pragma unroll
            for (int32_t v24_i1 = 0; v24_i1 < 12; ++v24_i1) {
              float v29_data = __ldcg(&glb_m0[(v22_lead + (v24_i1 * 6))]);
              r0[v24_i1] = v29_data;
            }
          }
          // s0 = load{g>s}(glb_m1[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 9; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m1[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m0););
          float r2[12]{};
          // r2 = load{g>r}(glb_m2);
          if (v23_g) {
            #pragma unroll
            for (int32_t v33_i1 = 0; v33_i1 < 12; ++v33_i1) {
              float v38_data = __ldcg(&glb_m2[(v22_lead + (v33_i1 * 6))]);
              r2[v33_i1] = v38_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[12]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // r1 = +(r0 * s0) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v41_data = r0[0];
          float v42_data = s0[0];
          float v44_data = r1[0];
          r1[0] = (v44_data + (v41_data * v42_data));
          float v47_data = s0[12];
          float v49_data = r1[1];
          r1[1] = (v49_data + (v41_data * v47_data));
          float v52_data = s0[24];
          float v54_data = r1[2];
          r1[2] = (v54_data + (v41_data * v52_data));
          float v57_data = s0[36];
          float v59_data = r1[3];
          r1[3] = (v59_data + (v41_data * v57_data));
          float v62_data = s0[48];
          float v64_data = r1[4];
          r1[4] = (v64_data + (v41_data * v62_data));
          float v67_data = s0[60];
          float v69_data = r1[5];
          r1[5] = (v69_data + (v41_data * v67_data));
          float v72_data = s0[72];
          float v74_data = r1[6];
          r1[6] = (v74_data + (v41_data * v72_data));
          float v77_data = s0[84];
          float v79_data = r1[7];
          r1[7] = (v79_data + (v41_data * v77_data));
          float v82_data = s0[96];
          float v84_data = r1[8];
          r1[8] = (v84_data + (v41_data * v82_data));
          float v87_data = s0[108];
          float v89_data = r1[9];
          r1[9] = (v89_data + (v41_data * v87_data));
          float v92_data = s0[120];
          float v94_data = r1[10];
          r1[10] = (v94_data + (v41_data * v92_data));
          float v97_data = s0[132];
          float v99_data = r1[11];
          r1[11] = (v99_data + (v41_data * v97_data));
          float v101_data = r0[1];
          float v102_data = s0[1];
          float v104_data = r1[0];
          r1[0] = (v104_data + (v101_data * v102_data));
          float v107_data = s0[13];
          float v109_data = r1[1];
          r1[1] = (v109_data + (v101_data * v107_data));
          float v112_data = s0[25];
          float v114_data = r1[2];
          r1[2] = (v114_data + (v101_data * v112_data));
          float v117_data = s0[37];
          float v119_data = r1[3];
          r1[3] = (v119_data + (v101_data * v117_data));
          float v122_data = s0[49];
          float v124_data = r1[4];
          r1[4] = (v124_data + (v101_data * v122_data));
          float v127_data = s0[61];
          float v129_data = r1[5];
          r1[5] = (v129_data + (v101_data * v127_data));
          float v132_data = s0[73];
          float v134_data = r1[6];
          r1[6] = (v134_data + (v101_data * v132_data));
          float v137_data = s0[85];
          float v139_data = r1[7];
          r1[7] = (v139_data + (v101_data * v137_data));
          float v142_data = s0[97];
          float v144_data = r1[8];
          r1[8] = (v144_data + (v101_data * v142_data));
          float v147_data = s0[109];
          float v149_data = r1[9];
          r1[9] = (v149_data + (v101_data * v147_data));
          float v152_data = s0[121];
          float v154_data = r1[10];
          r1[10] = (v154_data + (v101_data * v152_data));
          float v157_data = s0[133];
          float v159_data = r1[11];
          r1[11] = (v159_data + (v101_data * v157_data));
          float v161_data = r0[2];
          float v162_data = s0[2];
          float v164_data = r1[0];
          r1[0] = (v164_data + (v161_data * v162_data));
          float v167_data = s0[14];
          float v169_data = r1[1];
          r1[1] = (v169_data + (v161_data * v167_data));
          float v172_data = s0[26];
          float v174_data = r1[2];
          r1[2] = (v174_data + (v161_data * v172_data));
          float v177_data = s0[38];
          float v179_data = r1[3];
          r1[3] = (v179_data + (v161_data * v177_data));
          float v182_data = s0[50];
          float v184_data = r1[4];
          r1[4] = (v184_data + (v161_data * v182_data));
          float v187_data = s0[62];
          float v189_data = r1[5];
          r1[5] = (v189_data + (v161_data * v187_data));
          float v192_data = s0[74];
          float v194_data = r1[6];
          r1[6] = (v194_data + (v161_data * v192_data));
          float v197_data = s0[86];
          float v199_data = r1[7];
          r1[7] = (v199_data + (v161_data * v197_data));
          float v202_data = s0[98];
          float v204_data = r1[8];
          r1[8] = (v204_data + (v161_data * v202_data));
          float v207_data = s0[110];
          float v209_data = r1[9];
          r1[9] = (v209_data + (v161_data * v207_data));
          float v212_data = s0[122];
          float v214_data = r1[10];
          r1[10] = (v214_data + (v161_data * v212_data));
          float v217_data = s0[134];
          float v219_data = r1[11];
          r1[11] = (v219_data + (v161_data * v217_data));
          float v221_data = r0[3];
          float v222_data = s0[3];
          float v224_data = r1[0];
          r1[0] = (v224_data + (v221_data * v222_data));
          float v227_data = s0[15];
          float v229_data = r1[1];
          r1[1] = (v229_data + (v221_data * v227_data));
          float v232_data = s0[27];
          float v234_data = r1[2];
          r1[2] = (v234_data + (v221_data * v232_data));
          float v237_data = s0[39];
          float v239_data = r1[3];
          r1[3] = (v239_data + (v221_data * v237_data));
          float v242_data = s0[51];
          float v244_data = r1[4];
          r1[4] = (v244_data + (v221_data * v242_data));
          float v247_data = s0[63];
          float v249_data = r1[5];
          r1[5] = (v249_data + (v221_data * v247_data));
          float v252_data = s0[75];
          float v254_data = r1[6];
          r1[6] = (v254_data + (v221_data * v252_data));
          float v257_data = s0[87];
          float v259_data = r1[7];
          r1[7] = (v259_data + (v221_data * v257_data));
          float v262_data = s0[99];
          float v264_data = r1[8];
          r1[8] = (v264_data + (v221_data * v262_data));
          float v267_data = s0[111];
          float v269_data = r1[9];
          r1[9] = (v269_data + (v221_data * v267_data));
          float v272_data = s0[123];
          float v274_data = r1[10];
          r1[10] = (v274_data + (v221_data * v272_data));
          float v277_data = s0[135];
          float v279_data = r1[11];
          r1[11] = (v279_data + (v221_data * v277_data));
          float v281_data = r0[4];
          float v282_data = s0[4];
          float v284_data = r1[0];
          r1[0] = (v284_data + (v281_data * v282_data));
          float v287_data = s0[16];
          float v289_data = r1[1];
          r1[1] = (v289_data + (v281_data * v287_data));
          float v292_data = s0[28];
          float v294_data = r1[2];
          r1[2] = (v294_data + (v281_data * v292_data));
          float v297_data = s0[40];
          float v299_data = r1[3];
          r1[3] = (v299_data + (v281_data * v297_data));
          float v302_data = s0[52];
          float v304_data = r1[4];
          r1[4] = (v304_data + (v281_data * v302_data));
          float v307_data = s0[64];
          float v309_data = r1[5];
          r1[5] = (v309_data + (v281_data * v307_data));
          float v312_data = s0[76];
          float v314_data = r1[6];
          r1[6] = (v314_data + (v281_data * v312_data));
          float v317_data = s0[88];
          float v319_data = r1[7];
          r1[7] = (v319_data + (v281_data * v317_data));
          float v322_data = s0[100];
          float v324_data = r1[8];
          r1[8] = (v324_data + (v281_data * v322_data));
          float v327_data = s0[112];
          float v329_data = r1[9];
          r1[9] = (v329_data + (v281_data * v327_data));
          float v332_data = s0[124];
          float v334_data = r1[10];
          r1[10] = (v334_data + (v281_data * v332_data));
          float v337_data = s0[136];
          float v339_data = r1[11];
          r1[11] = (v339_data + (v281_data * v337_data));
          float v341_data = r0[5];
          float v342_data = s0[5];
          float v344_data = r1[0];
          r1[0] = (v344_data + (v341_data * v342_data));
          float v347_data = s0[17];
          float v349_data = r1[1];
          r1[1] = (v349_data + (v341_data * v347_data));
          float v352_data = s0[29];
          float v354_data = r1[2];
          r1[2] = (v354_data + (v341_data * v352_data));
          float v357_data = s0[41];
          float v359_data = r1[3];
          r1[3] = (v359_data + (v341_data * v357_data));
          float v362_data = s0[53];
          float v364_data = r1[4];
          r1[4] = (v364_data + (v341_data * v362_data));
          float v367_data = s0[65];
          float v369_data = r1[5];
          r1[5] = (v369_data + (v341_data * v367_data));
          float v372_data = s0[77];
          float v374_data = r1[6];
          r1[6] = (v374_data + (v341_data * v372_data));
          float v377_data = s0[89];
          float v379_data = r1[7];
          r1[7] = (v379_data + (v341_data * v377_data));
          float v382_data = s0[101];
          float v384_data = r1[8];
          r1[8] = (v384_data + (v341_data * v382_data));
          float v387_data = s0[113];
          float v389_data = r1[9];
          r1[9] = (v389_data + (v341_data * v387_data));
          float v392_data = s0[125];
          float v394_data = r1[10];
          r1[10] = (v394_data + (v341_data * v392_data));
          float v397_data = s0[137];
          float v399_data = r1[11];
          r1[11] = (v399_data + (v341_data * v397_data));
          float v401_data = r0[6];
          float v402_data = s0[6];
          float v404_data = r1[0];
          r1[0] = (v404_data + (v401_data * v402_data));
          float v407_data = s0[18];
          float v409_data = r1[1];
          r1[1] = (v409_data + (v401_data * v407_data));
          float v412_data = s0[30];
          float v414_data = r1[2];
          r1[2] = (v414_data + (v401_data * v412_data));
          float v417_data = s0[42];
          float v419_data = r1[3];
          r1[3] = (v419_data + (v401_data * v417_data));
          float v422_data = s0[54];
          float v424_data = r1[4];
          r1[4] = (v424_data + (v401_data * v422_data));
          float v427_data = s0[66];
          float v429_data = r1[5];
          r1[5] = (v429_data + (v401_data * v427_data));
          float v432_data = s0[78];
          float v434_data = r1[6];
          r1[6] = (v434_data + (v401_data * v432_data));
          float v437_data = s0[90];
          float v439_data = r1[7];
          r1[7] = (v439_data + (v401_data * v437_data));
          float v442_data = s0[102];
          float v444_data = r1[8];
          r1[8] = (v444_data + (v401_data * v442_data));
          float v447_data = s0[114];
          float v449_data = r1[9];
          r1[9] = (v449_data + (v401_data * v447_data));
          float v452_data = s0[126];
          float v454_data = r1[10];
          r1[10] = (v454_data + (v401_data * v452_data));
          float v457_data = s0[138];
          float v459_data = r1[11];
          r1[11] = (v459_data + (v401_data * v457_data));
          float v461_data = r0[7];
          float v462_data = s0[7];
          float v464_data = r1[0];
          r1[0] = (v464_data + (v461_data * v462_data));
          float v467_data = s0[19];
          float v469_data = r1[1];
          r1[1] = (v469_data + (v461_data * v467_data));
          float v472_data = s0[31];
          float v474_data = r1[2];
          r1[2] = (v474_data + (v461_data * v472_data));
          float v477_data = s0[43];
          float v479_data = r1[3];
          r1[3] = (v479_data + (v461_data * v477_data));
          float v482_data = s0[55];
          float v484_data = r1[4];
          r1[4] = (v484_data + (v461_data * v482_data));
          float v487_data = s0[67];
          float v489_data = r1[5];
          r1[5] = (v489_data + (v461_data * v487_data));
          float v492_data = s0[79];
          float v494_data = r1[6];
          r1[6] = (v494_data + (v461_data * v492_data));
          float v497_data = s0[91];
          float v499_data = r1[7];
          r1[7] = (v499_data + (v461_data * v497_data));
          float v502_data = s0[103];
          float v504_data = r1[8];
          r1[8] = (v504_data + (v461_data * v502_data));
          float v507_data = s0[115];
          float v509_data = r1[9];
          r1[9] = (v509_data + (v461_data * v507_data));
          float v512_data = s0[127];
          float v514_data = r1[10];
          r1[10] = (v514_data + (v461_data * v512_data));
          float v517_data = s0[139];
          float v519_data = r1[11];
          r1[11] = (v519_data + (v461_data * v517_data));
          float v521_data = r0[8];
          float v522_data = s0[8];
          float v524_data = r1[0];
          r1[0] = (v524_data + (v521_data * v522_data));
          float v527_data = s0[20];
          float v529_data = r1[1];
          r1[1] = (v529_data + (v521_data * v527_data));
          float v532_data = s0[32];
          float v534_data = r1[2];
          r1[2] = (v534_data + (v521_data * v532_data));
          float v537_data = s0[44];
          float v539_data = r1[3];
          r1[3] = (v539_data + (v521_data * v537_data));
          float v542_data = s0[56];
          float v544_data = r1[4];
          r1[4] = (v544_data + (v521_data * v542_data));
          float v547_data = s0[68];
          float v549_data = r1[5];
          r1[5] = (v549_data + (v521_data * v547_data));
          float v552_data = s0[80];
          float v554_data = r1[6];
          r1[6] = (v554_data + (v521_data * v552_data));
          float v557_data = s0[92];
          float v559_data = r1[7];
          r1[7] = (v559_data + (v521_data * v557_data));
          float v562_data = s0[104];
          float v564_data = r1[8];
          r1[8] = (v564_data + (v521_data * v562_data));
          float v567_data = s0[116];
          float v569_data = r1[9];
          r1[9] = (v569_data + (v521_data * v567_data));
          float v572_data = s0[128];
          float v574_data = r1[10];
          r1[10] = (v574_data + (v521_data * v572_data));
          float v577_data = s0[140];
          float v579_data = r1[11];
          r1[11] = (v579_data + (v521_data * v577_data));
          float v581_data = r0[9];
          float v582_data = s0[9];
          float v584_data = r1[0];
          r1[0] = (v584_data + (v581_data * v582_data));
          float v587_data = s0[21];
          float v589_data = r1[1];
          r1[1] = (v589_data + (v581_data * v587_data));
          float v592_data = s0[33];
          float v594_data = r1[2];
          r1[2] = (v594_data + (v581_data * v592_data));
          float v597_data = s0[45];
          float v599_data = r1[3];
          r1[3] = (v599_data + (v581_data * v597_data));
          float v602_data = s0[57];
          float v604_data = r1[4];
          r1[4] = (v604_data + (v581_data * v602_data));
          float v607_data = s0[69];
          float v609_data = r1[5];
          r1[5] = (v609_data + (v581_data * v607_data));
          float v612_data = s0[81];
          float v614_data = r1[6];
          r1[6] = (v614_data + (v581_data * v612_data));
          float v617_data = s0[93];
          float v619_data = r1[7];
          r1[7] = (v619_data + (v581_data * v617_data));
          float v622_data = s0[105];
          float v624_data = r1[8];
          r1[8] = (v624_data + (v581_data * v622_data));
          float v627_data = s0[117];
          float v629_data = r1[9];
          r1[9] = (v629_data + (v581_data * v627_data));
          float v632_data = s0[129];
          float v634_data = r1[10];
          r1[10] = (v634_data + (v581_data * v632_data));
          float v637_data = s0[141];
          float v639_data = r1[11];
          r1[11] = (v639_data + (v581_data * v637_data));
          float v641_data = r0[10];
          float v642_data = s0[10];
          float v644_data = r1[0];
          r1[0] = (v644_data + (v641_data * v642_data));
          float v647_data = s0[22];
          float v649_data = r1[1];
          r1[1] = (v649_data + (v641_data * v647_data));
          float v652_data = s0[34];
          float v654_data = r1[2];
          r1[2] = (v654_data + (v641_data * v652_data));
          float v657_data = s0[46];
          float v659_data = r1[3];
          r1[3] = (v659_data + (v641_data * v657_data));
          float v662_data = s0[58];
          float v664_data = r1[4];
          r1[4] = (v664_data + (v641_data * v662_data));
          float v667_data = s0[70];
          float v669_data = r1[5];
          r1[5] = (v669_data + (v641_data * v667_data));
          float v672_data = s0[82];
          float v674_data = r1[6];
          r1[6] = (v674_data + (v641_data * v672_data));
          float v677_data = s0[94];
          float v679_data = r1[7];
          r1[7] = (v679_data + (v641_data * v677_data));
          float v682_data = s0[106];
          float v684_data = r1[8];
          r1[8] = (v684_data + (v641_data * v682_data));
          float v687_data = s0[118];
          float v689_data = r1[9];
          r1[9] = (v689_data + (v641_data * v687_data));
          float v692_data = s0[130];
          float v694_data = r1[10];
          r1[10] = (v694_data + (v641_data * v692_data));
          float v697_data = s0[142];
          float v699_data = r1[11];
          r1[11] = (v699_data + (v641_data * v697_data));
          float v701_data = r0[11];
          float v702_data = s0[11];
          float v704_data = r1[0];
          r1[0] = (v704_data + (v701_data * v702_data));
          float v707_data = s0[23];
          float v709_data = r1[1];
          r1[1] = (v709_data + (v701_data * v707_data));
          float v712_data = s0[35];
          float v714_data = r1[2];
          r1[2] = (v714_data + (v701_data * v712_data));
          float v717_data = s0[47];
          float v719_data = r1[3];
          r1[3] = (v719_data + (v701_data * v717_data));
          float v722_data = s0[59];
          float v724_data = r1[4];
          r1[4] = (v724_data + (v701_data * v722_data));
          float v727_data = s0[71];
          float v729_data = r1[5];
          r1[5] = (v729_data + (v701_data * v727_data));
          float v732_data = s0[83];
          float v734_data = r1[6];
          r1[6] = (v734_data + (v701_data * v732_data));
          float v737_data = s0[95];
          float v739_data = r1[7];
          r1[7] = (v739_data + (v701_data * v737_data));
          float v742_data = s0[107];
          float v744_data = r1[8];
          r1[8] = (v744_data + (v701_data * v742_data));
          float v747_data = s0[119];
          float v749_data = r1[9];
          r1[9] = (v749_data + (v701_data * v747_data));
          float v752_data = s0[131];
          float v754_data = r1[10];
          r1[10] = (v754_data + (v701_data * v752_data));
          float v757_data = s0[143];
          float v759_data = r1[11];
          r1[11] = (v759_data + (v701_data * v757_data));
          // s1 = store{r>s}(localShrMem0, r1);
          if (v23_g) {
            int32_t v766_off = v22_lead + 6;
            #pragma unroll
            for (int32_t v761_i1 = 0; v761_i1 < 12; ++v761_i1) {
              float v763_data = r1[v761_i1];
              int32_t v768_a = v766_off + (v761_i1 * 12);
              s1[(v768_a ^ ((v768_a >> 4) & 15))] = v763_data;
            }
          }
          float r4[12]{};
          // r4 = load{g>r}(glb_m3);
          if (v23_g) {
            #pragma unroll
            for (int32_t v773_i1 = 0; v773_i1 < 12; ++v773_i1) {
              float v778_data = __ldcg(&glb_m3[(v22_lead + (v773_i1 * 6))]);
              r4[v773_i1] = v778_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m2););
          float r3[12]{};
          // ir3 = +(r2 * s0)
          // [(0, 6), (0, 12)] [(0, 12)]
          float ir3[12]{};
          float v782_data = r2[0];
          float v785_data = ir3[0];
          ir3[0] = (v785_data + (v782_data * v42_data));
          float v790_data = ir3[1];
          ir3[1] = (v790_data + (v782_data * v47_data));
          float v795_data = ir3[2];
          ir3[2] = (v795_data + (v782_data * v52_data));
          float v800_data = ir3[3];
          ir3[3] = (v800_data + (v782_data * v57_data));
          float v805_data = ir3[4];
          ir3[4] = (v805_data + (v782_data * v62_data));
          float v810_data = ir3[5];
          ir3[5] = (v810_data + (v782_data * v67_data));
          float v815_data = ir3[6];
          ir3[6] = (v815_data + (v782_data * v72_data));
          float v820_data = ir3[7];
          ir3[7] = (v820_data + (v782_data * v77_data));
          float v825_data = ir3[8];
          ir3[8] = (v825_data + (v782_data * v82_data));
          float v830_data = ir3[9];
          ir3[9] = (v830_data + (v782_data * v87_data));
          float v835_data = ir3[10];
          ir3[10] = (v835_data + (v782_data * v92_data));
          float v840_data = ir3[11];
          ir3[11] = (v840_data + (v782_data * v97_data));
          float v842_data = r2[1];
          float v845_data = ir3[0];
          ir3[0] = (v845_data + (v842_data * v102_data));
          float v850_data = ir3[1];
          ir3[1] = (v850_data + (v842_data * v107_data));
          float v855_data = ir3[2];
          ir3[2] = (v855_data + (v842_data * v112_data));
          float v860_data = ir3[3];
          ir3[3] = (v860_data + (v842_data * v117_data));
          float v865_data = ir3[4];
          ir3[4] = (v865_data + (v842_data * v122_data));
          float v870_data = ir3[5];
          ir3[5] = (v870_data + (v842_data * v127_data));
          float v875_data = ir3[6];
          ir3[6] = (v875_data + (v842_data * v132_data));
          float v880_data = ir3[7];
          ir3[7] = (v880_data + (v842_data * v137_data));
          float v885_data = ir3[8];
          ir3[8] = (v885_data + (v842_data * v142_data));
          float v890_data = ir3[9];
          ir3[9] = (v890_data + (v842_data * v147_data));
          float v895_data = ir3[10];
          ir3[10] = (v895_data + (v842_data * v152_data));
          float v900_data = ir3[11];
          ir3[11] = (v900_data + (v842_data * v157_data));
          float v902_data = r2[2];
          float v905_data = ir3[0];
          ir3[0] = (v905_data + (v902_data * v162_data));
          float v910_data = ir3[1];
          ir3[1] = (v910_data + (v902_data * v167_data));
          float v915_data = ir3[2];
          ir3[2] = (v915_data + (v902_data * v172_data));
          float v920_data = ir3[3];
          ir3[3] = (v920_data + (v902_data * v177_data));
          float v925_data = ir3[4];
          ir3[4] = (v925_data + (v902_data * v182_data));
          float v930_data = ir3[5];
          ir3[5] = (v930_data + (v902_data * v187_data));
          float v935_data = ir3[6];
          ir3[6] = (v935_data + (v902_data * v192_data));
          float v940_data = ir3[7];
          ir3[7] = (v940_data + (v902_data * v197_data));
          float v945_data = ir3[8];
          ir3[8] = (v945_data + (v902_data * v202_data));
          float v950_data = ir3[9];
          ir3[9] = (v950_data + (v902_data * v207_data));
          float v955_data = ir3[10];
          ir3[10] = (v955_data + (v902_data * v212_data));
          float v960_data = ir3[11];
          ir3[11] = (v960_data + (v902_data * v217_data));
          float v962_data = r2[3];
          float v965_data = ir3[0];
          ir3[0] = (v965_data + (v962_data * v222_data));
          float v970_data = ir3[1];
          ir3[1] = (v970_data + (v962_data * v227_data));
          float v975_data = ir3[2];
          ir3[2] = (v975_data + (v962_data * v232_data));
          float v980_data = ir3[3];
          ir3[3] = (v980_data + (v962_data * v237_data));
          float v985_data = ir3[4];
          ir3[4] = (v985_data + (v962_data * v242_data));
          float v990_data = ir3[5];
          ir3[5] = (v990_data + (v962_data * v247_data));
          float v995_data = ir3[6];
          ir3[6] = (v995_data + (v962_data * v252_data));
          float v1000_data = ir3[7];
          ir3[7] = (v1000_data + (v962_data * v257_data));
          float v1005_data = ir3[8];
          ir3[8] = (v1005_data + (v962_data * v262_data));
          float v1010_data = ir3[9];
          ir3[9] = (v1010_data + (v962_data * v267_data));
          float v1015_data = ir3[10];
          ir3[10] = (v1015_data + (v962_data * v272_data));
          float v1020_data = ir3[11];
          ir3[11] = (v1020_data + (v962_data * v277_data));
          float v1022_data = r2[4];
          float v1025_data = ir3[0];
          ir3[0] = (v1025_data + (v1022_data * v282_data));
          float v1030_data = ir3[1];
          ir3[1] = (v1030_data + (v1022_data * v287_data));
          float v1035_data = ir3[2];
          ir3[2] = (v1035_data + (v1022_data * v292_data));
          float v1040_data = ir3[3];
          ir3[3] = (v1040_data + (v1022_data * v297_data));
          float v1045_data = ir3[4];
          ir3[4] = (v1045_data + (v1022_data * v302_data));
          float v1050_data = ir3[5];
          ir3[5] = (v1050_data + (v1022_data * v307_data));
          float v1055_data = ir3[6];
          ir3[6] = (v1055_data + (v1022_data * v312_data));
          float v1060_data = ir3[7];
          ir3[7] = (v1060_data + (v1022_data * v317_data));
          float v1065_data = ir3[8];
          ir3[8] = (v1065_data + (v1022_data * v322_data));
          float v1070_data = ir3[9];
          ir3[9] = (v1070_data + (v1022_data * v327_data));
          float v1075_data = ir3[10];
          ir3[10] = (v1075_data + (v1022_data * v332_data));
          float v1080_data = ir3[11];
          ir3[11] = (v1080_data + (v1022_data * v337_data));
          float v1082_data = r2[5];
          float v1085_data = ir3[0];
          ir3[0] = (v1085_data + (v1082_data * v342_data));
          float v1090_data = ir3[1];
          ir3[1] = (v1090_data + (v1082_data * v347_data));
          float v1095_data = ir3[2];
          ir3[2] = (v1095_data + (v1082_data * v352_data));
          float v1100_data = ir3[3];
          ir3[3] = (v1100_data + (v1082_data * v357_data));
          float v1105_data = ir3[4];
          ir3[4] = (v1105_data + (v1082_data * v362_data));
          float v1110_data = ir3[5];
          ir3[5] = (v1110_data + (v1082_data * v367_data));
          float v1115_data = ir3[6];
          ir3[6] = (v1115_data + (v1082_data * v372_data));
          float v1120_data = ir3[7];
          ir3[7] = (v1120_data + (v1082_data * v377_data));
          float v1125_data = ir3[8];
          ir3[8] = (v1125_data + (v1082_data * v382_data));
          float v1130_data = ir3[9];
          ir3[9] = (v1130_data + (v1082_data * v387_data));
          float v1135_data = ir3[10];
          ir3[10] = (v1135_data + (v1082_data * v392_data));
          float v1140_data = ir3[11];
          ir3[11] = (v1140_data + (v1082_data * v397_data));
          float v1142_data = r2[6];
          float v1145_data = ir3[0];
          ir3[0] = (v1145_data + (v1142_data * v402_data));
          float v1150_data = ir3[1];
          ir3[1] = (v1150_data + (v1142_data * v407_data));
          float v1155_data = ir3[2];
          ir3[2] = (v1155_data + (v1142_data * v412_data));
          float v1160_data = ir3[3];
          ir3[3] = (v1160_data + (v1142_data * v417_data));
          float v1165_data = ir3[4];
          ir3[4] = (v1165_data + (v1142_data * v422_data));
          float v1170_data = ir3[5];
          ir3[5] = (v1170_data + (v1142_data * v427_data));
          float v1175_data = ir3[6];
          ir3[6] = (v1175_data + (v1142_data * v432_data));
          float v1180_data = ir3[7];
          ir3[7] = (v1180_data + (v1142_data * v437_data));
          float v1185_data = ir3[8];
          ir3[8] = (v1185_data + (v1142_data * v442_data));
          float v1190_data = ir3[9];
          ir3[9] = (v1190_data + (v1142_data * v447_data));
          float v1195_data = ir3[10];
          ir3[10] = (v1195_data + (v1142_data * v452_data));
          float v1200_data = ir3[11];
          ir3[11] = (v1200_data + (v1142_data * v457_data));
          float v1202_data = r2[7];
          float v1205_data = ir3[0];
          ir3[0] = (v1205_data + (v1202_data * v462_data));
          float v1210_data = ir3[1];
          ir3[1] = (v1210_data + (v1202_data * v467_data));
          float v1215_data = ir3[2];
          ir3[2] = (v1215_data + (v1202_data * v472_data));
          float v1220_data = ir3[3];
          ir3[3] = (v1220_data + (v1202_data * v477_data));
          float v1225_data = ir3[4];
          ir3[4] = (v1225_data + (v1202_data * v482_data));
          float v1230_data = ir3[5];
          ir3[5] = (v1230_data + (v1202_data * v487_data));
          float v1235_data = ir3[6];
          ir3[6] = (v1235_data + (v1202_data * v492_data));
          float v1240_data = ir3[7];
          ir3[7] = (v1240_data + (v1202_data * v497_data));
          float v1245_data = ir3[8];
          ir3[8] = (v1245_data + (v1202_data * v502_data));
          float v1250_data = ir3[9];
          ir3[9] = (v1250_data + (v1202_data * v507_data));
          float v1255_data = ir3[10];
          ir3[10] = (v1255_data + (v1202_data * v512_data));
          float v1260_data = ir3[11];
          ir3[11] = (v1260_data + (v1202_data * v517_data));
          float v1262_data = r2[8];
          float v1265_data = ir3[0];
          ir3[0] = (v1265_data + (v1262_data * v522_data));
          float v1270_data = ir3[1];
          ir3[1] = (v1270_data + (v1262_data * v527_data));
          float v1275_data = ir3[2];
          ir3[2] = (v1275_data + (v1262_data * v532_data));
          float v1280_data = ir3[3];
          ir3[3] = (v1280_data + (v1262_data * v537_data));
          float v1285_data = ir3[4];
          ir3[4] = (v1285_data + (v1262_data * v542_data));
          float v1290_data = ir3[5];
          ir3[5] = (v1290_data + (v1262_data * v547_data));
          float v1295_data = ir3[6];
          ir3[6] = (v1295_data + (v1262_data * v552_data));
          float v1300_data = ir3[7];
          ir3[7] = (v1300_data + (v1262_data * v557_data));
          float v1305_data = ir3[8];
          ir3[8] = (v1305_data + (v1262_data * v562_data));
          float v1310_data = ir3[9];
          ir3[9] = (v1310_data + (v1262_data * v567_data));
          float v1315_data = ir3[10];
          ir3[10] = (v1315_data + (v1262_data * v572_data));
          float v1320_data = ir3[11];
          ir3[11] = (v1320_data + (v1262_data * v577_data));
          float v1322_data = r2[9];
          float v1325_data = ir3[0];
          ir3[0] = (v1325_data + (v1322_data * v582_data));
          float v1330_data = ir3[1];
          ir3[1] = (v1330_data + (v1322_data * v587_data));
          float v1335_data = ir3[2];
          ir3[2] = (v1335_data + (v1322_data * v592_data));
          float v1340_data = ir3[3];
          ir3[3] = (v1340_data + (v1322_data * v597_data));
          float v1345_data = ir3[4];
          ir3[4] = (v1345_data + (v1322_data * v602_data));
          float v1350_data = ir3[5];
          ir3[5] = (v1350_data + (v1322_data * v607_data));
          float v1355_data = ir3[6];
          ir3[6] = (v1355_data + (v1322_data * v612_data));
          float v1360_data = ir3[7];
          ir3[7] = (v1360_data + (v1322_data * v617_data));
          float v1365_data = ir3[8];
          ir3[8] = (v1365_data + (v1322_data * v622_data));
          float v1370_data = ir3[9];
          ir3[9] = (v1370_data + (v1322_data * v627_data));
          float v1375_data = ir3[10];
          ir3[10] = (v1375_data + (v1322_data * v632_data));
          float v1380_data = ir3[11];
          ir3[11] = (v1380_data + (v1322_data * v637_data));
          float v1382_data = r2[10];
          float v1385_data = ir3[0];
          ir3[0] = (v1385_data + (v1382_data * v642_data));
          float v1390_data = ir3[1];
          ir3[1] = (v1390_data + (v1382_data * v647_data));
          float v1395_data = ir3[2];
          ir3[2] = (v1395_data + (v1382_data * v652_data));
          float v1400_data = ir3[3];
          ir3[3] = (v1400_data + (v1382_data * v657_data));
          float v1405_data = ir3[4];
          ir3[4] = (v1405_data + (v1382_data * v662_data));
          float v1410_data = ir3[5];
          ir3[5] = (v1410_data + (v1382_data * v667_data));
          float v1415_data = ir3[6];
          ir3[6] = (v1415_data + (v1382_data * v672_data));
          float v1420_data = ir3[7];
          ir3[7] = (v1420_data + (v1382_data * v677_data));
          float v1425_data = ir3[8];
          ir3[8] = (v1425_data + (v1382_data * v682_data));
          float v1430_data = ir3[9];
          ir3[9] = (v1430_data + (v1382_data * v687_data));
          float v1435_data = ir3[10];
          ir3[10] = (v1435_data + (v1382_data * v692_data));
          float v1440_data = ir3[11];
          ir3[11] = (v1440_data + (v1382_data * v697_data));
          float v1442_data = r2[11];
          float v1445_data = ir3[0];
          ir3[0] = (v1445_data + (v1442_data * v702_data));
          float v1450_data = ir3[1];
          ir3[1] = (v1450_data + (v1442_data * v707_data));
          float v1455_data = ir3[2];
          ir3[2] = (v1455_data + (v1442_data * v712_data));
          float v1460_data = ir3[3];
          ir3[3] = (v1460_data + (v1442_data * v717_data));
          float v1465_data = ir3[4];
          ir3[4] = (v1465_data + (v1442_data * v722_data));
          float v1470_data = ir3[5];
          ir3[5] = (v1470_data + (v1442_data * v727_data));
          float v1475_data = ir3[6];
          ir3[6] = (v1475_data + (v1442_data * v732_data));
          float v1480_data = ir3[7];
          ir3[7] = (v1480_data + (v1442_data * v737_data));
          float v1485_data = ir3[8];
          ir3[8] = (v1485_data + (v1442_data * v742_data));
          float v1490_data = ir3[9];
          ir3[9] = (v1490_data + (v1442_data * v747_data));
          float v1495_data = ir3[10];
          ir3[10] = (v1495_data + (v1442_data * v752_data));
          float v1500_data = ir3[11];
          ir3[11] = (v1500_data + (v1442_data * v757_data));
          // r3 = ir3
          if (v23_g) {
            #pragma unroll
            for (int32_t v1502_n1 = 0; v1502_n1 < 12; ++v1502_n1) {
              float v1504_data = ir3[v1502_n1];
              r3[v1502_n1] = v1504_data;
            }
          }
          // s1 = store{r>s, clear}(localShrMem0, r3);
          bool v1506_g = v22_lead < 12;
          if ((v22_lead >= 6) && v1506_g) {
            #pragma unroll
            for (int32_t v1508_z1 = 0; v1508_z1 < 12; ++v1508_z1) {
              int32_t v1513_a = v22_lead + (v1508_z1 * 12);
              s1[(v1513_a ^ ((v1513_a >> 4) & 15))] = 0.0f;
            }
          }
          if (v23_g) {
            #pragma unroll
            for (int32_t v1517_i1 = 0; v1517_i1 < 12; ++v1517_i1) {
              float v1519_data = r3[v1517_i1];
              int32_t v1523_a = v22_lead + (v1517_i1 * 12);
              s1[(v1523_a ^ ((v1523_a >> 4) & 15))] = v1519_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m3););
          float r5[12]{};
          // ir5 = +(r4)
          // [(0, 6), (0, 12)] []
          float ir5[12]{};
          float v1529_data = r4[0];
          float v1530_data = ir5[0];
          ir5[0] = (v1530_data + v1529_data);
          float v1532_data = r4[1];
          float v1533_data = ir5[1];
          ir5[1] = (v1533_data + v1532_data);
          float v1535_data = r4[2];
          float v1536_data = ir5[2];
          ir5[2] = (v1536_data + v1535_data);
          float v1538_data = r4[3];
          float v1539_data = ir5[3];
          ir5[3] = (v1539_data + v1538_data);
          float v1541_data = r4[4];
          float v1542_data = ir5[4];
          ir5[4] = (v1542_data + v1541_data);
          float v1544_data = r4[5];
          float v1545_data = ir5[5];
          ir5[5] = (v1545_data + v1544_data);
          float v1547_data = r4[6];
          float v1548_data = ir5[6];
          ir5[6] = (v1548_data + v1547_data);
          float v1550_data = r4[7];
          float v1551_data = ir5[7];
          ir5[7] = (v1551_data + v1550_data);
          float v1553_data = r4[8];
          float v1554_data = ir5[8];
          ir5[8] = (v1554_data + v1553_data);
          float v1556_data = r4[9];
          float v1557_data = ir5[9];
          ir5[9] = (v1557_data + v1556_data);
          float v1559_data = r4[10];
          float v1560_data = ir5[10];
          ir5[10] = (v1560_data + v1559_data);
          float v1562_data = r4[11];
          float v1563_data = ir5[11];
          ir5[11] = (v1563_data + v1562_data);
          // r5 = ir5
          if (v23_g) {
            #pragma unroll
            for (int32_t v1565_n1 = 0; v1565_n1 < 12; ++v1565_n1) {
              float v1567_data = ir5[v1565_n1];
              r5[v1565_n1] = v1567_data;
            }
          }
          // s1 = store{r>s}(localShrMem0, r5);
          if (v23_g) {
            int32_t v1573_off = v22_lead + 6;
            #pragma unroll
            for (int32_t v1568_i1 = 0; v1568_i1 < 12; ++v1568_i1) {
              float v1570_data = r5[v1568_i1];
              int32_t v1575_a = v1573_off + (v1568_i1 * 12);
              s1[(v1575_a ^ ((v1575_a >> 4) & 15))] = v1570_data;
            }
          }
          float r6[12]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // ir6 = +(s1)
          // [(0, 12), (0, 12)] []
          float ir6[12]{};
          float v1587_data = v1506_g ? (s1[(v22_lead ^ ((v22_lead >> 4) & 15))]) : (0.0f);
          float v1588_data = ir6[0];
          ir6[0] = (v1588_data + v1587_data);
          int32_t v1590_a = v22_lead + 12;
          float v1594_data = v1506_g ? (s1[(v1590_a ^ ((v1590_a >> 4) & 15))]) : (0.0f);
          float v1595_data = ir6[1];
          ir6[1] = (v1595_data + v1594_data);
          int32_t v1597_a = v22_lead + 24;
          float v1601_data = v1506_g ? (s1[(v1597_a ^ ((v1597_a >> 4) & 15))]) : (0.0f);
          float v1602_data = ir6[2];
          ir6[2] = (v1602_data + v1601_data);
          int32_t v1604_a = v22_lead + 36;
          float v1608_data = v1506_g ? (s1[(v1604_a ^ ((v1604_a >> 4) & 15))]) : (0.0f);
          float v1609_data = ir6[3];
          ir6[3] = (v1609_data + v1608_data);
          int32_t v1611_a = v22_lead + 48;
          float v1615_data = v1506_g ? (s1[(v1611_a ^ ((v1611_a >> 4) & 15))]) : (0.0f);
          float v1616_data = ir6[4];
          ir6[4] = (v1616_data + v1615_data);
          int32_t v1618_a = v22_lead + 60;
          float v1622_data = v1506_g ? (s1[(v1618_a ^ ((v1618_a >> 4) & 15))]) : (0.0f);
          float v1623_data = ir6[5];
          ir6[5] = (v1623_data + v1622_data);
          int32_t v1625_a = v22_lead + 72;
          float v1629_data = v1506_g ? (s1[(v1625_a ^ ((v1625_a >> 4) & 15))]) : (0.0f);
          float v1630_data = ir6[6];
          ir6[6] = (v1630_data + v1629_data);
          int32_t v1632_a = v22_lead + 84;
          float v1636_data = v1506_g ? (s1[(v1632_a ^ ((v1632_a >> 4) & 15))]) : (0.0f);
          float v1637_data = ir6[7];
          ir6[7] = (v1637_data + v1636_data);
          int32_t v1639_a = v22_lead + 96;
          float v1643_data = v1506_g ? (s1[(v1639_a ^ ((v1639_a >> 4) & 15))]) : (0.0f);
          float v1644_data = ir6[8];
          ir6[8] = (v1644_data + v1643_data);
          int32_t v1646_a = v22_lead + 108;
          float v1650_data = v1506_g ? (s1[(v1646_a ^ ((v1646_a >> 4) & 15))]) : (0.0f);
          float v1651_data = ir6[9];
          ir6[9] = (v1651_data + v1650_data);
          int32_t v1653_a = v22_lead + 120;
          float v1657_data = v1506_g ? (s1[(v1653_a ^ ((v1653_a >> 4) & 15))]) : (0.0f);
          float v1658_data = ir6[10];
          ir6[10] = (v1658_data + v1657_data);
          int32_t v1660_a = v22_lead + 132;
          float v1664_data = v1506_g ? (s1[(v1660_a ^ ((v1660_a >> 4) & 15))]) : (0.0f);
          float v1665_data = ir6[11];
          ir6[11] = (v1665_data + v1664_data);
          // r6 = ir6
          if (v1506_g) {
            #pragma unroll
            for (int32_t v1667_n1 = 0; v1667_n1 < 12; ++v1667_n1) {
              float v1669_data = ir6[v1667_n1];
              r6[v1667_n1] = v1669_data;
            }
          }
          // glb_m4 = store{r>g}(r6);
          if (v1506_g) {
            #pragma unroll
            for (int32_t v1670_i1 = 0; v1670_i1 < 12; ++v1670_i1) {
              float v1672_data = r6[v1670_i1];
              glb_m4[(v22_lead + (v1670_i1 * 12))] = v1672_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

