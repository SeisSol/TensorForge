// === base name ===
kernel_fdac3ecc037a4854

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_fdac3ecc037a4854 = {{32, 4, 1}, 32, 32, 1, 4, 3072, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_fdac3ecc037a4854(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_fdac3ecc037a4854(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_fdac3ecc037a4854(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_fdac3ecc037a4854, block.x * block.y * block.z, 768 * sizeof(float));
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
  config.sharedMemBytes = 768 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_fdac3ecc037a4854(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_fdac3ecc037a4854(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_fdac3ecc037a4854, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_fdac3ecc037a4854<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, m6, m6_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_fdac3ecc037a4854(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 4 per block = block 32x4x1, 3072 B shared, occupancy grid
    // operands:
    //   m0 32×16(32×16) {0..32}×{0..16} strided
    //   m1 32×12(32×12) {0..32}×{0..12} strided
    //   m2 12×16(12×16) {0..12}×{0..16} strided
    //   m3 32×12(32×12) {0..32}×{0..12} strided
    //   m4 12×8(12×8) {0..12}×{0..8} strided
    //   m5 32×12(32×12) {0..32}×{0..12} strided
    //   m6 12×8(12×8) {0..12}×{0..8} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    //   m0[i,j]@{0..32}×{0..8} += m3[i,k] × m4[k,j]
    //   m0[i,j]@{0..32}×{8..16} += m5[i,k] × m6[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":768}],"shared_bytes":3072,"shared_elements":768,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[32,16]],"name":"m0","ordered":false,"parts":1,"shape":[32,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[32,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,12],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[12,16]],"name":"m2","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"A0","bbox":[[0,0],[32,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,12],"variant":false},{"addressing":"strided","alias":"B0","bbox":[[0,0],[12,8]],"name":"m4","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A1","bbox":[[0,0],[32,12]],"name":"m5","ordered":false,"parts":1,"shape":[32,12],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[12,8]],"name":"m6","ordered":false,"parts":1,"shape":[12,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,12]},{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[12,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,8]],"is_tmp":false,"name":"m0","offset":[0,8],"shape":[32,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[32,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[192 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[192];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      float * __restrict__ s2 = &localShrMem0[0];
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 512 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 384 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 192 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[v7_batchId0 * 384 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[v7_batchId0 * 96 + 0 + m4_extraOffset];
          const float *const __restrict__ glb_m5 = &m5[v7_batchId0 * 384 + 0 + m5_extraOffset];
          const float *const __restrict__ glb_m6 = &m6[v7_batchId0 * 96 + 0 + m6_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v25_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v26_i0 = 0; v26_i0 < 1; ++v26_i0) {
            int32_t v29_lead = v25_lead + (v26_i0 * 32);
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 12; ++v27_i1) {
              float v32_data = __ldcg(&glb_m1[(v29_lead + (v27_i1 * 32))]);
              r0[(v26_i0 + v27_i1)] = v32_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 6; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 32], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 32], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = load{g>r}(glb_m3);
          #pragma unroll
          for (int32_t v36_i0 = 0; v36_i0 < 1; ++v36_i0) {
            int32_t v39_lead = v25_lead + (v36_i0 * 32);
            #pragma unroll
            for (int32_t v37_i1 = 0; v37_i1 < 12; ++v37_i1) {
              float v42_data = __ldcg(&glb_m3[(v39_lead + (v37_i1 * 32))]);
              r2[(v36_i0 + v37_i1)] = v42_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[16]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 32), (0, 16)] [(0, 12)]
          float ir1[16]{};
          float v46_data = r0[0];
          float v47_data = s0[0];
          float v49_data = ir1[0];
          ir1[0] = (v49_data + (v46_data * v47_data));
          float v52_data = s0[12];
          float v54_data = ir1[1];
          ir1[1] = (v54_data + (v46_data * v52_data));
          float v57_data = s0[24];
          float v59_data = ir1[2];
          ir1[2] = (v59_data + (v46_data * v57_data));
          float v62_data = s0[36];
          float v64_data = ir1[3];
          ir1[3] = (v64_data + (v46_data * v62_data));
          float v67_data = s0[48];
          float v69_data = ir1[4];
          ir1[4] = (v69_data + (v46_data * v67_data));
          float v72_data = s0[60];
          float v74_data = ir1[5];
          ir1[5] = (v74_data + (v46_data * v72_data));
          float v77_data = s0[72];
          float v79_data = ir1[6];
          ir1[6] = (v79_data + (v46_data * v77_data));
          float v82_data = s0[84];
          float v84_data = ir1[7];
          ir1[7] = (v84_data + (v46_data * v82_data));
          float v87_data = s0[96];
          float v89_data = ir1[8];
          ir1[8] = (v89_data + (v46_data * v87_data));
          float v92_data = s0[108];
          float v94_data = ir1[9];
          ir1[9] = (v94_data + (v46_data * v92_data));
          float v97_data = s0[120];
          float v99_data = ir1[10];
          ir1[10] = (v99_data + (v46_data * v97_data));
          float v102_data = s0[132];
          float v104_data = ir1[11];
          ir1[11] = (v104_data + (v46_data * v102_data));
          float v107_data = s0[144];
          float v109_data = ir1[12];
          ir1[12] = (v109_data + (v46_data * v107_data));
          float v112_data = s0[156];
          float v114_data = ir1[13];
          ir1[13] = (v114_data + (v46_data * v112_data));
          float v117_data = s0[168];
          float v119_data = ir1[14];
          ir1[14] = (v119_data + (v46_data * v117_data));
          float v122_data = s0[180];
          float v124_data = ir1[15];
          ir1[15] = (v124_data + (v46_data * v122_data));
          float v126_data = r0[1];
          float v127_data = s0[1];
          float v129_data = ir1[0];
          ir1[0] = (v129_data + (v126_data * v127_data));
          float v132_data = s0[13];
          float v134_data = ir1[1];
          ir1[1] = (v134_data + (v126_data * v132_data));
          float v137_data = s0[25];
          float v139_data = ir1[2];
          ir1[2] = (v139_data + (v126_data * v137_data));
          float v142_data = s0[37];
          float v144_data = ir1[3];
          ir1[3] = (v144_data + (v126_data * v142_data));
          float v147_data = s0[49];
          float v149_data = ir1[4];
          ir1[4] = (v149_data + (v126_data * v147_data));
          float v152_data = s0[61];
          float v154_data = ir1[5];
          ir1[5] = (v154_data + (v126_data * v152_data));
          float v157_data = s0[73];
          float v159_data = ir1[6];
          ir1[6] = (v159_data + (v126_data * v157_data));
          float v162_data = s0[85];
          float v164_data = ir1[7];
          ir1[7] = (v164_data + (v126_data * v162_data));
          float v167_data = s0[97];
          float v169_data = ir1[8];
          ir1[8] = (v169_data + (v126_data * v167_data));
          float v172_data = s0[109];
          float v174_data = ir1[9];
          ir1[9] = (v174_data + (v126_data * v172_data));
          float v177_data = s0[121];
          float v179_data = ir1[10];
          ir1[10] = (v179_data + (v126_data * v177_data));
          float v182_data = s0[133];
          float v184_data = ir1[11];
          ir1[11] = (v184_data + (v126_data * v182_data));
          float v187_data = s0[145];
          float v189_data = ir1[12];
          ir1[12] = (v189_data + (v126_data * v187_data));
          float v192_data = s0[157];
          float v194_data = ir1[13];
          ir1[13] = (v194_data + (v126_data * v192_data));
          float v197_data = s0[169];
          float v199_data = ir1[14];
          ir1[14] = (v199_data + (v126_data * v197_data));
          float v202_data = s0[181];
          float v204_data = ir1[15];
          ir1[15] = (v204_data + (v126_data * v202_data));
          float v206_data = r0[2];
          float v207_data = s0[2];
          float v209_data = ir1[0];
          ir1[0] = (v209_data + (v206_data * v207_data));
          float v212_data = s0[14];
          float v214_data = ir1[1];
          ir1[1] = (v214_data + (v206_data * v212_data));
          float v217_data = s0[26];
          float v219_data = ir1[2];
          ir1[2] = (v219_data + (v206_data * v217_data));
          float v222_data = s0[38];
          float v224_data = ir1[3];
          ir1[3] = (v224_data + (v206_data * v222_data));
          float v227_data = s0[50];
          float v229_data = ir1[4];
          ir1[4] = (v229_data + (v206_data * v227_data));
          float v232_data = s0[62];
          float v234_data = ir1[5];
          ir1[5] = (v234_data + (v206_data * v232_data));
          float v237_data = s0[74];
          float v239_data = ir1[6];
          ir1[6] = (v239_data + (v206_data * v237_data));
          float v242_data = s0[86];
          float v244_data = ir1[7];
          ir1[7] = (v244_data + (v206_data * v242_data));
          float v247_data = s0[98];
          float v249_data = ir1[8];
          ir1[8] = (v249_data + (v206_data * v247_data));
          float v252_data = s0[110];
          float v254_data = ir1[9];
          ir1[9] = (v254_data + (v206_data * v252_data));
          float v257_data = s0[122];
          float v259_data = ir1[10];
          ir1[10] = (v259_data + (v206_data * v257_data));
          float v262_data = s0[134];
          float v264_data = ir1[11];
          ir1[11] = (v264_data + (v206_data * v262_data));
          float v267_data = s0[146];
          float v269_data = ir1[12];
          ir1[12] = (v269_data + (v206_data * v267_data));
          float v272_data = s0[158];
          float v274_data = ir1[13];
          ir1[13] = (v274_data + (v206_data * v272_data));
          float v277_data = s0[170];
          float v279_data = ir1[14];
          ir1[14] = (v279_data + (v206_data * v277_data));
          float v282_data = s0[182];
          float v284_data = ir1[15];
          ir1[15] = (v284_data + (v206_data * v282_data));
          float v286_data = r0[3];
          float v287_data = s0[3];
          float v289_data = ir1[0];
          ir1[0] = (v289_data + (v286_data * v287_data));
          float v292_data = s0[15];
          float v294_data = ir1[1];
          ir1[1] = (v294_data + (v286_data * v292_data));
          float v297_data = s0[27];
          float v299_data = ir1[2];
          ir1[2] = (v299_data + (v286_data * v297_data));
          float v302_data = s0[39];
          float v304_data = ir1[3];
          ir1[3] = (v304_data + (v286_data * v302_data));
          float v307_data = s0[51];
          float v309_data = ir1[4];
          ir1[4] = (v309_data + (v286_data * v307_data));
          float v312_data = s0[63];
          float v314_data = ir1[5];
          ir1[5] = (v314_data + (v286_data * v312_data));
          float v317_data = s0[75];
          float v319_data = ir1[6];
          ir1[6] = (v319_data + (v286_data * v317_data));
          float v322_data = s0[87];
          float v324_data = ir1[7];
          ir1[7] = (v324_data + (v286_data * v322_data));
          float v327_data = s0[99];
          float v329_data = ir1[8];
          ir1[8] = (v329_data + (v286_data * v327_data));
          float v332_data = s0[111];
          float v334_data = ir1[9];
          ir1[9] = (v334_data + (v286_data * v332_data));
          float v337_data = s0[123];
          float v339_data = ir1[10];
          ir1[10] = (v339_data + (v286_data * v337_data));
          float v342_data = s0[135];
          float v344_data = ir1[11];
          ir1[11] = (v344_data + (v286_data * v342_data));
          float v347_data = s0[147];
          float v349_data = ir1[12];
          ir1[12] = (v349_data + (v286_data * v347_data));
          float v352_data = s0[159];
          float v354_data = ir1[13];
          ir1[13] = (v354_data + (v286_data * v352_data));
          float v357_data = s0[171];
          float v359_data = ir1[14];
          ir1[14] = (v359_data + (v286_data * v357_data));
          float v362_data = s0[183];
          float v364_data = ir1[15];
          ir1[15] = (v364_data + (v286_data * v362_data));
          float v366_data = r0[4];
          float v367_data = s0[4];
          float v369_data = ir1[0];
          ir1[0] = (v369_data + (v366_data * v367_data));
          float v372_data = s0[16];
          float v374_data = ir1[1];
          ir1[1] = (v374_data + (v366_data * v372_data));
          float v377_data = s0[28];
          float v379_data = ir1[2];
          ir1[2] = (v379_data + (v366_data * v377_data));
          float v382_data = s0[40];
          float v384_data = ir1[3];
          ir1[3] = (v384_data + (v366_data * v382_data));
          float v387_data = s0[52];
          float v389_data = ir1[4];
          ir1[4] = (v389_data + (v366_data * v387_data));
          float v392_data = s0[64];
          float v394_data = ir1[5];
          ir1[5] = (v394_data + (v366_data * v392_data));
          float v397_data = s0[76];
          float v399_data = ir1[6];
          ir1[6] = (v399_data + (v366_data * v397_data));
          float v402_data = s0[88];
          float v404_data = ir1[7];
          ir1[7] = (v404_data + (v366_data * v402_data));
          float v407_data = s0[100];
          float v409_data = ir1[8];
          ir1[8] = (v409_data + (v366_data * v407_data));
          float v412_data = s0[112];
          float v414_data = ir1[9];
          ir1[9] = (v414_data + (v366_data * v412_data));
          float v417_data = s0[124];
          float v419_data = ir1[10];
          ir1[10] = (v419_data + (v366_data * v417_data));
          float v422_data = s0[136];
          float v424_data = ir1[11];
          ir1[11] = (v424_data + (v366_data * v422_data));
          float v427_data = s0[148];
          float v429_data = ir1[12];
          ir1[12] = (v429_data + (v366_data * v427_data));
          float v432_data = s0[160];
          float v434_data = ir1[13];
          ir1[13] = (v434_data + (v366_data * v432_data));
          float v437_data = s0[172];
          float v439_data = ir1[14];
          ir1[14] = (v439_data + (v366_data * v437_data));
          float v442_data = s0[184];
          float v444_data = ir1[15];
          ir1[15] = (v444_data + (v366_data * v442_data));
          float v446_data = r0[5];
          float v447_data = s0[5];
          float v449_data = ir1[0];
          ir1[0] = (v449_data + (v446_data * v447_data));
          float v452_data = s0[17];
          float v454_data = ir1[1];
          ir1[1] = (v454_data + (v446_data * v452_data));
          float v457_data = s0[29];
          float v459_data = ir1[2];
          ir1[2] = (v459_data + (v446_data * v457_data));
          float v462_data = s0[41];
          float v464_data = ir1[3];
          ir1[3] = (v464_data + (v446_data * v462_data));
          float v467_data = s0[53];
          float v469_data = ir1[4];
          ir1[4] = (v469_data + (v446_data * v467_data));
          float v472_data = s0[65];
          float v474_data = ir1[5];
          ir1[5] = (v474_data + (v446_data * v472_data));
          float v477_data = s0[77];
          float v479_data = ir1[6];
          ir1[6] = (v479_data + (v446_data * v477_data));
          float v482_data = s0[89];
          float v484_data = ir1[7];
          ir1[7] = (v484_data + (v446_data * v482_data));
          float v487_data = s0[101];
          float v489_data = ir1[8];
          ir1[8] = (v489_data + (v446_data * v487_data));
          float v492_data = s0[113];
          float v494_data = ir1[9];
          ir1[9] = (v494_data + (v446_data * v492_data));
          float v497_data = s0[125];
          float v499_data = ir1[10];
          ir1[10] = (v499_data + (v446_data * v497_data));
          float v502_data = s0[137];
          float v504_data = ir1[11];
          ir1[11] = (v504_data + (v446_data * v502_data));
          float v507_data = s0[149];
          float v509_data = ir1[12];
          ir1[12] = (v509_data + (v446_data * v507_data));
          float v512_data = s0[161];
          float v514_data = ir1[13];
          ir1[13] = (v514_data + (v446_data * v512_data));
          float v517_data = s0[173];
          float v519_data = ir1[14];
          ir1[14] = (v519_data + (v446_data * v517_data));
          float v522_data = s0[185];
          float v524_data = ir1[15];
          ir1[15] = (v524_data + (v446_data * v522_data));
          float v526_data = r0[6];
          float v527_data = s0[6];
          float v529_data = ir1[0];
          ir1[0] = (v529_data + (v526_data * v527_data));
          float v532_data = s0[18];
          float v534_data = ir1[1];
          ir1[1] = (v534_data + (v526_data * v532_data));
          float v537_data = s0[30];
          float v539_data = ir1[2];
          ir1[2] = (v539_data + (v526_data * v537_data));
          float v542_data = s0[42];
          float v544_data = ir1[3];
          ir1[3] = (v544_data + (v526_data * v542_data));
          float v547_data = s0[54];
          float v549_data = ir1[4];
          ir1[4] = (v549_data + (v526_data * v547_data));
          float v552_data = s0[66];
          float v554_data = ir1[5];
          ir1[5] = (v554_data + (v526_data * v552_data));
          float v557_data = s0[78];
          float v559_data = ir1[6];
          ir1[6] = (v559_data + (v526_data * v557_data));
          float v562_data = s0[90];
          float v564_data = ir1[7];
          ir1[7] = (v564_data + (v526_data * v562_data));
          float v567_data = s0[102];
          float v569_data = ir1[8];
          ir1[8] = (v569_data + (v526_data * v567_data));
          float v572_data = s0[114];
          float v574_data = ir1[9];
          ir1[9] = (v574_data + (v526_data * v572_data));
          float v577_data = s0[126];
          float v579_data = ir1[10];
          ir1[10] = (v579_data + (v526_data * v577_data));
          float v582_data = s0[138];
          float v584_data = ir1[11];
          ir1[11] = (v584_data + (v526_data * v582_data));
          float v587_data = s0[150];
          float v589_data = ir1[12];
          ir1[12] = (v589_data + (v526_data * v587_data));
          float v592_data = s0[162];
          float v594_data = ir1[13];
          ir1[13] = (v594_data + (v526_data * v592_data));
          float v597_data = s0[174];
          float v599_data = ir1[14];
          ir1[14] = (v599_data + (v526_data * v597_data));
          float v602_data = s0[186];
          float v604_data = ir1[15];
          ir1[15] = (v604_data + (v526_data * v602_data));
          float v606_data = r0[7];
          float v607_data = s0[7];
          float v609_data = ir1[0];
          ir1[0] = (v609_data + (v606_data * v607_data));
          float v612_data = s0[19];
          float v614_data = ir1[1];
          ir1[1] = (v614_data + (v606_data * v612_data));
          float v617_data = s0[31];
          float v619_data = ir1[2];
          ir1[2] = (v619_data + (v606_data * v617_data));
          float v622_data = s0[43];
          float v624_data = ir1[3];
          ir1[3] = (v624_data + (v606_data * v622_data));
          float v627_data = s0[55];
          float v629_data = ir1[4];
          ir1[4] = (v629_data + (v606_data * v627_data));
          float v632_data = s0[67];
          float v634_data = ir1[5];
          ir1[5] = (v634_data + (v606_data * v632_data));
          float v637_data = s0[79];
          float v639_data = ir1[6];
          ir1[6] = (v639_data + (v606_data * v637_data));
          float v642_data = s0[91];
          float v644_data = ir1[7];
          ir1[7] = (v644_data + (v606_data * v642_data));
          float v647_data = s0[103];
          float v649_data = ir1[8];
          ir1[8] = (v649_data + (v606_data * v647_data));
          float v652_data = s0[115];
          float v654_data = ir1[9];
          ir1[9] = (v654_data + (v606_data * v652_data));
          float v657_data = s0[127];
          float v659_data = ir1[10];
          ir1[10] = (v659_data + (v606_data * v657_data));
          float v662_data = s0[139];
          float v664_data = ir1[11];
          ir1[11] = (v664_data + (v606_data * v662_data));
          float v667_data = s0[151];
          float v669_data = ir1[12];
          ir1[12] = (v669_data + (v606_data * v667_data));
          float v672_data = s0[163];
          float v674_data = ir1[13];
          ir1[13] = (v674_data + (v606_data * v672_data));
          float v677_data = s0[175];
          float v679_data = ir1[14];
          ir1[14] = (v679_data + (v606_data * v677_data));
          float v682_data = s0[187];
          float v684_data = ir1[15];
          ir1[15] = (v684_data + (v606_data * v682_data));
          float v686_data = r0[8];
          float v687_data = s0[8];
          float v689_data = ir1[0];
          ir1[0] = (v689_data + (v686_data * v687_data));
          float v692_data = s0[20];
          float v694_data = ir1[1];
          ir1[1] = (v694_data + (v686_data * v692_data));
          float v697_data = s0[32];
          float v699_data = ir1[2];
          ir1[2] = (v699_data + (v686_data * v697_data));
          float v702_data = s0[44];
          float v704_data = ir1[3];
          ir1[3] = (v704_data + (v686_data * v702_data));
          float v707_data = s0[56];
          float v709_data = ir1[4];
          ir1[4] = (v709_data + (v686_data * v707_data));
          float v712_data = s0[68];
          float v714_data = ir1[5];
          ir1[5] = (v714_data + (v686_data * v712_data));
          float v717_data = s0[80];
          float v719_data = ir1[6];
          ir1[6] = (v719_data + (v686_data * v717_data));
          float v722_data = s0[92];
          float v724_data = ir1[7];
          ir1[7] = (v724_data + (v686_data * v722_data));
          float v727_data = s0[104];
          float v729_data = ir1[8];
          ir1[8] = (v729_data + (v686_data * v727_data));
          float v732_data = s0[116];
          float v734_data = ir1[9];
          ir1[9] = (v734_data + (v686_data * v732_data));
          float v737_data = s0[128];
          float v739_data = ir1[10];
          ir1[10] = (v739_data + (v686_data * v737_data));
          float v742_data = s0[140];
          float v744_data = ir1[11];
          ir1[11] = (v744_data + (v686_data * v742_data));
          float v747_data = s0[152];
          float v749_data = ir1[12];
          ir1[12] = (v749_data + (v686_data * v747_data));
          float v752_data = s0[164];
          float v754_data = ir1[13];
          ir1[13] = (v754_data + (v686_data * v752_data));
          float v757_data = s0[176];
          float v759_data = ir1[14];
          ir1[14] = (v759_data + (v686_data * v757_data));
          float v762_data = s0[188];
          float v764_data = ir1[15];
          ir1[15] = (v764_data + (v686_data * v762_data));
          float v766_data = r0[9];
          float v767_data = s0[9];
          float v769_data = ir1[0];
          ir1[0] = (v769_data + (v766_data * v767_data));
          float v772_data = s0[21];
          float v774_data = ir1[1];
          ir1[1] = (v774_data + (v766_data * v772_data));
          float v777_data = s0[33];
          float v779_data = ir1[2];
          ir1[2] = (v779_data + (v766_data * v777_data));
          float v782_data = s0[45];
          float v784_data = ir1[3];
          ir1[3] = (v784_data + (v766_data * v782_data));
          float v787_data = s0[57];
          float v789_data = ir1[4];
          ir1[4] = (v789_data + (v766_data * v787_data));
          float v792_data = s0[69];
          float v794_data = ir1[5];
          ir1[5] = (v794_data + (v766_data * v792_data));
          float v797_data = s0[81];
          float v799_data = ir1[6];
          ir1[6] = (v799_data + (v766_data * v797_data));
          float v802_data = s0[93];
          float v804_data = ir1[7];
          ir1[7] = (v804_data + (v766_data * v802_data));
          float v807_data = s0[105];
          float v809_data = ir1[8];
          ir1[8] = (v809_data + (v766_data * v807_data));
          float v812_data = s0[117];
          float v814_data = ir1[9];
          ir1[9] = (v814_data + (v766_data * v812_data));
          float v817_data = s0[129];
          float v819_data = ir1[10];
          ir1[10] = (v819_data + (v766_data * v817_data));
          float v822_data = s0[141];
          float v824_data = ir1[11];
          ir1[11] = (v824_data + (v766_data * v822_data));
          float v827_data = s0[153];
          float v829_data = ir1[12];
          ir1[12] = (v829_data + (v766_data * v827_data));
          float v832_data = s0[165];
          float v834_data = ir1[13];
          ir1[13] = (v834_data + (v766_data * v832_data));
          float v837_data = s0[177];
          float v839_data = ir1[14];
          ir1[14] = (v839_data + (v766_data * v837_data));
          float v842_data = s0[189];
          float v844_data = ir1[15];
          ir1[15] = (v844_data + (v766_data * v842_data));
          float v846_data = r0[10];
          float v847_data = s0[10];
          float v849_data = ir1[0];
          ir1[0] = (v849_data + (v846_data * v847_data));
          float v852_data = s0[22];
          float v854_data = ir1[1];
          ir1[1] = (v854_data + (v846_data * v852_data));
          float v857_data = s0[34];
          float v859_data = ir1[2];
          ir1[2] = (v859_data + (v846_data * v857_data));
          float v862_data = s0[46];
          float v864_data = ir1[3];
          ir1[3] = (v864_data + (v846_data * v862_data));
          float v867_data = s0[58];
          float v869_data = ir1[4];
          ir1[4] = (v869_data + (v846_data * v867_data));
          float v872_data = s0[70];
          float v874_data = ir1[5];
          ir1[5] = (v874_data + (v846_data * v872_data));
          float v877_data = s0[82];
          float v879_data = ir1[6];
          ir1[6] = (v879_data + (v846_data * v877_data));
          float v882_data = s0[94];
          float v884_data = ir1[7];
          ir1[7] = (v884_data + (v846_data * v882_data));
          float v887_data = s0[106];
          float v889_data = ir1[8];
          ir1[8] = (v889_data + (v846_data * v887_data));
          float v892_data = s0[118];
          float v894_data = ir1[9];
          ir1[9] = (v894_data + (v846_data * v892_data));
          float v897_data = s0[130];
          float v899_data = ir1[10];
          ir1[10] = (v899_data + (v846_data * v897_data));
          float v902_data = s0[142];
          float v904_data = ir1[11];
          ir1[11] = (v904_data + (v846_data * v902_data));
          float v907_data = s0[154];
          float v909_data = ir1[12];
          ir1[12] = (v909_data + (v846_data * v907_data));
          float v912_data = s0[166];
          float v914_data = ir1[13];
          ir1[13] = (v914_data + (v846_data * v912_data));
          float v917_data = s0[178];
          float v919_data = ir1[14];
          ir1[14] = (v919_data + (v846_data * v917_data));
          float v922_data = s0[190];
          float v924_data = ir1[15];
          ir1[15] = (v924_data + (v846_data * v922_data));
          float v926_data = r0[11];
          float v927_data = s0[11];
          float v929_data = ir1[0];
          ir1[0] = (v929_data + (v926_data * v927_data));
          float v932_data = s0[23];
          float v934_data = ir1[1];
          ir1[1] = (v934_data + (v926_data * v932_data));
          float v937_data = s0[35];
          float v939_data = ir1[2];
          ir1[2] = (v939_data + (v926_data * v937_data));
          float v942_data = s0[47];
          float v944_data = ir1[3];
          ir1[3] = (v944_data + (v926_data * v942_data));
          float v947_data = s0[59];
          float v949_data = ir1[4];
          ir1[4] = (v949_data + (v926_data * v947_data));
          float v952_data = s0[71];
          float v954_data = ir1[5];
          ir1[5] = (v954_data + (v926_data * v952_data));
          float v957_data = s0[83];
          float v959_data = ir1[6];
          ir1[6] = (v959_data + (v926_data * v957_data));
          float v962_data = s0[95];
          float v964_data = ir1[7];
          ir1[7] = (v964_data + (v926_data * v962_data));
          float v967_data = s0[107];
          float v969_data = ir1[8];
          ir1[8] = (v969_data + (v926_data * v967_data));
          float v972_data = s0[119];
          float v974_data = ir1[9];
          ir1[9] = (v974_data + (v926_data * v972_data));
          float v977_data = s0[131];
          float v979_data = ir1[10];
          ir1[10] = (v979_data + (v926_data * v977_data));
          float v982_data = s0[143];
          float v984_data = ir1[11];
          ir1[11] = (v984_data + (v926_data * v982_data));
          float v987_data = s0[155];
          float v989_data = ir1[12];
          ir1[12] = (v989_data + (v926_data * v987_data));
          float v992_data = s0[167];
          float v994_data = ir1[13];
          ir1[13] = (v994_data + (v926_data * v992_data));
          float v997_data = s0[179];
          float v999_data = ir1[14];
          ir1[14] = (v999_data + (v926_data * v997_data));
          float v1002_data = s0[191];
          float v1004_data = ir1[15];
          ir1[15] = (v1004_data + (v926_data * v1002_data));
          #pragma unroll
          for (int32_t v1006_n0 = 0; v1006_n0 < 1; ++v1006_n0) {
            #pragma unroll
            for (int32_t v1007_n1 = 0; v1007_n1 < 16; ++v1007_n1) {
              int32_t v1008_a = v1006_n0 + v1007_n1;
              float v1009_data = ir1[v1008_a];
              r1[v1008_a] = v1009_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v1010_i0 = 0; v1010_i0 < 1; ++v1010_i0) {
            int32_t v1015_lead = v25_lead + (v1010_i0 * 32);
            #pragma unroll
            for (int32_t v1011_i1 = 0; v1011_i1 < 16; ++v1011_i1) {
              float v1013_data = r1[(v1010_i0 + v1011_i1)];
              glb_m0[(v1015_lead + (v1011_i1 * 32))] = v1013_data;
            }
          }
          __syncwarp();
          // s1 = load{g>s}(glb_m4[0, 1])
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 0], &glb_m4[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 32], &glb_m4[0 + 0 + 1 * threadIdx.x + 32], 4);
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 64], &glb_m4[0 + 0 + 1 * threadIdx.x + 64], 4);
          __pipeline_commit();
          // wait(r2 = load{g>r}(glb_m3););
          float r3[8]{};
          // r3 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v1022_i0 = 0; v1022_i0 < 1; ++v1022_i0) {
            int32_t v1025_lead = v25_lead + (v1022_i0 * 32);
            #pragma unroll
            for (int32_t v1023_i1 = 0; v1023_i1 < 8; ++v1023_i1) {
              float v1028_data = glb_m0[(v1025_lead + (v1023_i1 * 32))];
              r3[(v1022_i0 + v1023_i1)] = v1028_data;
            }
          }
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          float r5[12]{};
          // r5 = load{g>r}(glb_m5);
          #pragma unroll
          for (int32_t v1031_i0 = 0; v1031_i0 < 1; ++v1031_i0) {
            int32_t v1034_lead = v25_lead + (v1031_i0 * 32);
            #pragma unroll
            for (int32_t v1032_i1 = 0; v1032_i1 < 12; ++v1032_i1) {
              float v1037_data = __ldcg(&glb_m5[(v1034_lead + (v1032_i1 * 32))]);
              r5[(v1031_i0 + v1032_i1)] = v1037_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m0););
          float r4[8]{};
          __syncwarp();
          // r4 = +(r2 * s1) + name: r3, type: SymbolType.Register, lead: [0]
          // [(0, 32), (0, 8)] [(0, 12)]
          float ir4[8]{};
          float v1041_data = r2[0];
          float v1042_data = s1[0];
          float v1044_data = ir4[0];
          ir4[0] = (v1044_data + (v1041_data * v1042_data));
          float v1047_data = s1[12];
          float v1049_data = ir4[1];
          ir4[1] = (v1049_data + (v1041_data * v1047_data));
          float v1052_data = s1[24];
          float v1054_data = ir4[2];
          ir4[2] = (v1054_data + (v1041_data * v1052_data));
          float v1057_data = s1[36];
          float v1059_data = ir4[3];
          ir4[3] = (v1059_data + (v1041_data * v1057_data));
          float v1062_data = s1[48];
          float v1064_data = ir4[4];
          ir4[4] = (v1064_data + (v1041_data * v1062_data));
          float v1067_data = s1[60];
          float v1069_data = ir4[5];
          ir4[5] = (v1069_data + (v1041_data * v1067_data));
          float v1072_data = s1[72];
          float v1074_data = ir4[6];
          ir4[6] = (v1074_data + (v1041_data * v1072_data));
          float v1077_data = s1[84];
          float v1079_data = ir4[7];
          ir4[7] = (v1079_data + (v1041_data * v1077_data));
          float v1081_data = r2[1];
          float v1082_data = s1[1];
          float v1084_data = ir4[0];
          ir4[0] = (v1084_data + (v1081_data * v1082_data));
          float v1087_data = s1[13];
          float v1089_data = ir4[1];
          ir4[1] = (v1089_data + (v1081_data * v1087_data));
          float v1092_data = s1[25];
          float v1094_data = ir4[2];
          ir4[2] = (v1094_data + (v1081_data * v1092_data));
          float v1097_data = s1[37];
          float v1099_data = ir4[3];
          ir4[3] = (v1099_data + (v1081_data * v1097_data));
          float v1102_data = s1[49];
          float v1104_data = ir4[4];
          ir4[4] = (v1104_data + (v1081_data * v1102_data));
          float v1107_data = s1[61];
          float v1109_data = ir4[5];
          ir4[5] = (v1109_data + (v1081_data * v1107_data));
          float v1112_data = s1[73];
          float v1114_data = ir4[6];
          ir4[6] = (v1114_data + (v1081_data * v1112_data));
          float v1117_data = s1[85];
          float v1119_data = ir4[7];
          ir4[7] = (v1119_data + (v1081_data * v1117_data));
          float v1121_data = r2[2];
          float v1122_data = s1[2];
          float v1124_data = ir4[0];
          ir4[0] = (v1124_data + (v1121_data * v1122_data));
          float v1127_data = s1[14];
          float v1129_data = ir4[1];
          ir4[1] = (v1129_data + (v1121_data * v1127_data));
          float v1132_data = s1[26];
          float v1134_data = ir4[2];
          ir4[2] = (v1134_data + (v1121_data * v1132_data));
          float v1137_data = s1[38];
          float v1139_data = ir4[3];
          ir4[3] = (v1139_data + (v1121_data * v1137_data));
          float v1142_data = s1[50];
          float v1144_data = ir4[4];
          ir4[4] = (v1144_data + (v1121_data * v1142_data));
          float v1147_data = s1[62];
          float v1149_data = ir4[5];
          ir4[5] = (v1149_data + (v1121_data * v1147_data));
          float v1152_data = s1[74];
          float v1154_data = ir4[6];
          ir4[6] = (v1154_data + (v1121_data * v1152_data));
          float v1157_data = s1[86];
          float v1159_data = ir4[7];
          ir4[7] = (v1159_data + (v1121_data * v1157_data));
          float v1161_data = r2[3];
          float v1162_data = s1[3];
          float v1164_data = ir4[0];
          ir4[0] = (v1164_data + (v1161_data * v1162_data));
          float v1167_data = s1[15];
          float v1169_data = ir4[1];
          ir4[1] = (v1169_data + (v1161_data * v1167_data));
          float v1172_data = s1[27];
          float v1174_data = ir4[2];
          ir4[2] = (v1174_data + (v1161_data * v1172_data));
          float v1177_data = s1[39];
          float v1179_data = ir4[3];
          ir4[3] = (v1179_data + (v1161_data * v1177_data));
          float v1182_data = s1[51];
          float v1184_data = ir4[4];
          ir4[4] = (v1184_data + (v1161_data * v1182_data));
          float v1187_data = s1[63];
          float v1189_data = ir4[5];
          ir4[5] = (v1189_data + (v1161_data * v1187_data));
          float v1192_data = s1[75];
          float v1194_data = ir4[6];
          ir4[6] = (v1194_data + (v1161_data * v1192_data));
          float v1197_data = s1[87];
          float v1199_data = ir4[7];
          ir4[7] = (v1199_data + (v1161_data * v1197_data));
          float v1201_data = r2[4];
          float v1202_data = s1[4];
          float v1204_data = ir4[0];
          ir4[0] = (v1204_data + (v1201_data * v1202_data));
          float v1207_data = s1[16];
          float v1209_data = ir4[1];
          ir4[1] = (v1209_data + (v1201_data * v1207_data));
          float v1212_data = s1[28];
          float v1214_data = ir4[2];
          ir4[2] = (v1214_data + (v1201_data * v1212_data));
          float v1217_data = s1[40];
          float v1219_data = ir4[3];
          ir4[3] = (v1219_data + (v1201_data * v1217_data));
          float v1222_data = s1[52];
          float v1224_data = ir4[4];
          ir4[4] = (v1224_data + (v1201_data * v1222_data));
          float v1227_data = s1[64];
          float v1229_data = ir4[5];
          ir4[5] = (v1229_data + (v1201_data * v1227_data));
          float v1232_data = s1[76];
          float v1234_data = ir4[6];
          ir4[6] = (v1234_data + (v1201_data * v1232_data));
          float v1237_data = s1[88];
          float v1239_data = ir4[7];
          ir4[7] = (v1239_data + (v1201_data * v1237_data));
          float v1241_data = r2[5];
          float v1242_data = s1[5];
          float v1244_data = ir4[0];
          ir4[0] = (v1244_data + (v1241_data * v1242_data));
          float v1247_data = s1[17];
          float v1249_data = ir4[1];
          ir4[1] = (v1249_data + (v1241_data * v1247_data));
          float v1252_data = s1[29];
          float v1254_data = ir4[2];
          ir4[2] = (v1254_data + (v1241_data * v1252_data));
          float v1257_data = s1[41];
          float v1259_data = ir4[3];
          ir4[3] = (v1259_data + (v1241_data * v1257_data));
          float v1262_data = s1[53];
          float v1264_data = ir4[4];
          ir4[4] = (v1264_data + (v1241_data * v1262_data));
          float v1267_data = s1[65];
          float v1269_data = ir4[5];
          ir4[5] = (v1269_data + (v1241_data * v1267_data));
          float v1272_data = s1[77];
          float v1274_data = ir4[6];
          ir4[6] = (v1274_data + (v1241_data * v1272_data));
          float v1277_data = s1[89];
          float v1279_data = ir4[7];
          ir4[7] = (v1279_data + (v1241_data * v1277_data));
          float v1281_data = r2[6];
          float v1282_data = s1[6];
          float v1284_data = ir4[0];
          ir4[0] = (v1284_data + (v1281_data * v1282_data));
          float v1287_data = s1[18];
          float v1289_data = ir4[1];
          ir4[1] = (v1289_data + (v1281_data * v1287_data));
          float v1292_data = s1[30];
          float v1294_data = ir4[2];
          ir4[2] = (v1294_data + (v1281_data * v1292_data));
          float v1297_data = s1[42];
          float v1299_data = ir4[3];
          ir4[3] = (v1299_data + (v1281_data * v1297_data));
          float v1302_data = s1[54];
          float v1304_data = ir4[4];
          ir4[4] = (v1304_data + (v1281_data * v1302_data));
          float v1307_data = s1[66];
          float v1309_data = ir4[5];
          ir4[5] = (v1309_data + (v1281_data * v1307_data));
          float v1312_data = s1[78];
          float v1314_data = ir4[6];
          ir4[6] = (v1314_data + (v1281_data * v1312_data));
          float v1317_data = s1[90];
          float v1319_data = ir4[7];
          ir4[7] = (v1319_data + (v1281_data * v1317_data));
          float v1321_data = r2[7];
          float v1322_data = s1[7];
          float v1324_data = ir4[0];
          ir4[0] = (v1324_data + (v1321_data * v1322_data));
          float v1327_data = s1[19];
          float v1329_data = ir4[1];
          ir4[1] = (v1329_data + (v1321_data * v1327_data));
          float v1332_data = s1[31];
          float v1334_data = ir4[2];
          ir4[2] = (v1334_data + (v1321_data * v1332_data));
          float v1337_data = s1[43];
          float v1339_data = ir4[3];
          ir4[3] = (v1339_data + (v1321_data * v1337_data));
          float v1342_data = s1[55];
          float v1344_data = ir4[4];
          ir4[4] = (v1344_data + (v1321_data * v1342_data));
          float v1347_data = s1[67];
          float v1349_data = ir4[5];
          ir4[5] = (v1349_data + (v1321_data * v1347_data));
          float v1352_data = s1[79];
          float v1354_data = ir4[6];
          ir4[6] = (v1354_data + (v1321_data * v1352_data));
          float v1357_data = s1[91];
          float v1359_data = ir4[7];
          ir4[7] = (v1359_data + (v1321_data * v1357_data));
          float v1361_data = r2[8];
          float v1362_data = s1[8];
          float v1364_data = ir4[0];
          ir4[0] = (v1364_data + (v1361_data * v1362_data));
          float v1367_data = s1[20];
          float v1369_data = ir4[1];
          ir4[1] = (v1369_data + (v1361_data * v1367_data));
          float v1372_data = s1[32];
          float v1374_data = ir4[2];
          ir4[2] = (v1374_data + (v1361_data * v1372_data));
          float v1377_data = s1[44];
          float v1379_data = ir4[3];
          ir4[3] = (v1379_data + (v1361_data * v1377_data));
          float v1382_data = s1[56];
          float v1384_data = ir4[4];
          ir4[4] = (v1384_data + (v1361_data * v1382_data));
          float v1387_data = s1[68];
          float v1389_data = ir4[5];
          ir4[5] = (v1389_data + (v1361_data * v1387_data));
          float v1392_data = s1[80];
          float v1394_data = ir4[6];
          ir4[6] = (v1394_data + (v1361_data * v1392_data));
          float v1397_data = s1[92];
          float v1399_data = ir4[7];
          ir4[7] = (v1399_data + (v1361_data * v1397_data));
          float v1401_data = r2[9];
          float v1402_data = s1[9];
          float v1404_data = ir4[0];
          ir4[0] = (v1404_data + (v1401_data * v1402_data));
          float v1407_data = s1[21];
          float v1409_data = ir4[1];
          ir4[1] = (v1409_data + (v1401_data * v1407_data));
          float v1412_data = s1[33];
          float v1414_data = ir4[2];
          ir4[2] = (v1414_data + (v1401_data * v1412_data));
          float v1417_data = s1[45];
          float v1419_data = ir4[3];
          ir4[3] = (v1419_data + (v1401_data * v1417_data));
          float v1422_data = s1[57];
          float v1424_data = ir4[4];
          ir4[4] = (v1424_data + (v1401_data * v1422_data));
          float v1427_data = s1[69];
          float v1429_data = ir4[5];
          ir4[5] = (v1429_data + (v1401_data * v1427_data));
          float v1432_data = s1[81];
          float v1434_data = ir4[6];
          ir4[6] = (v1434_data + (v1401_data * v1432_data));
          float v1437_data = s1[93];
          float v1439_data = ir4[7];
          ir4[7] = (v1439_data + (v1401_data * v1437_data));
          float v1441_data = r2[10];
          float v1442_data = s1[10];
          float v1444_data = ir4[0];
          ir4[0] = (v1444_data + (v1441_data * v1442_data));
          float v1447_data = s1[22];
          float v1449_data = ir4[1];
          ir4[1] = (v1449_data + (v1441_data * v1447_data));
          float v1452_data = s1[34];
          float v1454_data = ir4[2];
          ir4[2] = (v1454_data + (v1441_data * v1452_data));
          float v1457_data = s1[46];
          float v1459_data = ir4[3];
          ir4[3] = (v1459_data + (v1441_data * v1457_data));
          float v1462_data = s1[58];
          float v1464_data = ir4[4];
          ir4[4] = (v1464_data + (v1441_data * v1462_data));
          float v1467_data = s1[70];
          float v1469_data = ir4[5];
          ir4[5] = (v1469_data + (v1441_data * v1467_data));
          float v1472_data = s1[82];
          float v1474_data = ir4[6];
          ir4[6] = (v1474_data + (v1441_data * v1472_data));
          float v1477_data = s1[94];
          float v1479_data = ir4[7];
          ir4[7] = (v1479_data + (v1441_data * v1477_data));
          float v1481_data = r2[11];
          float v1482_data = s1[11];
          float v1484_data = ir4[0];
          ir4[0] = (v1484_data + (v1481_data * v1482_data));
          float v1487_data = s1[23];
          float v1489_data = ir4[1];
          ir4[1] = (v1489_data + (v1481_data * v1487_data));
          float v1492_data = s1[35];
          float v1494_data = ir4[2];
          ir4[2] = (v1494_data + (v1481_data * v1492_data));
          float v1497_data = s1[47];
          float v1499_data = ir4[3];
          ir4[3] = (v1499_data + (v1481_data * v1497_data));
          float v1502_data = s1[59];
          float v1504_data = ir4[4];
          ir4[4] = (v1504_data + (v1481_data * v1502_data));
          float v1507_data = s1[71];
          float v1509_data = ir4[5];
          ir4[5] = (v1509_data + (v1481_data * v1507_data));
          float v1512_data = s1[83];
          float v1514_data = ir4[6];
          ir4[6] = (v1514_data + (v1481_data * v1512_data));
          float v1517_data = s1[95];
          float v1519_data = ir4[7];
          ir4[7] = (v1519_data + (v1481_data * v1517_data));
          #pragma unroll
          for (int32_t v1521_n0 = 0; v1521_n0 < 1; ++v1521_n0) {
            #pragma unroll
            for (int32_t v1522_n1 = 0; v1522_n1 < 8; ++v1522_n1) {
              int32_t v1523_a = v1521_n0 + v1522_n1;
              float v1524_data = ir4[v1523_a];
              float v1525_data = r3[v1523_a];
              r4[v1523_a] = (v1525_data + v1524_data);
            }
          }
          // glb_m0 = store{r>g}(r4);
          #pragma unroll
          for (int32_t v1527_i0 = 0; v1527_i0 < 1; ++v1527_i0) {
            int32_t v1532_lead = v25_lead + (v1527_i0 * 32);
            #pragma unroll
            for (int32_t v1528_i1 = 0; v1528_i1 < 8; ++v1528_i1) {
              float v1530_data = r4[(v1527_i0 + v1528_i1)];
              glb_m0[(v1532_lead + (v1528_i1 * 32))] = v1530_data;
            }
          }
          __syncwarp();
          // s2 = load{g>s}(glb_m6[0, 1])
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 0], &glb_m6[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 32], &glb_m6[0 + 0 + 1 * threadIdx.x + 32], 4);
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 64], &glb_m6[0 + 0 + 1 * threadIdx.x + 64], 4);
          __pipeline_commit();
          // wait(r5 = load{g>r}(glb_m5););
          float r6[8]{};
          // r6 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v1539_i0 = 0; v1539_i0 < 1; ++v1539_i0) {
            int32_t v1542_lead = v25_lead + (v1539_i0 * 32);
            #pragma unroll
            for (int32_t v1540_i1 = 0; v1540_i1 < 8; ++v1540_i1) {
              float v1546_data = glb_m0[(v1542_lead + ((v1540_i1 + 8) * 32))];
              r6[(v1539_i0 + v1540_i1)] = v1546_data;
            }
          }
          // wait(s2 = load{g>s}(glb_m6[0, 1]));
          __pipeline_wait_prior(0);
          // wait(r6 = load{g>r}(glb_m0););
          float r7[8]{};
          __syncwarp();
          // r7 = +(r5 * s2) + name: r6, type: SymbolType.Register, lead: [0]
          // [(0, 32), (0, 8)] [(0, 12)]
          float ir7[8]{};
          float v1550_data = r5[0];
          float v1551_data = s2[0];
          float v1553_data = ir7[0];
          ir7[0] = (v1553_data + (v1550_data * v1551_data));
          float v1556_data = s2[12];
          float v1558_data = ir7[1];
          ir7[1] = (v1558_data + (v1550_data * v1556_data));
          float v1561_data = s2[24];
          float v1563_data = ir7[2];
          ir7[2] = (v1563_data + (v1550_data * v1561_data));
          float v1566_data = s2[36];
          float v1568_data = ir7[3];
          ir7[3] = (v1568_data + (v1550_data * v1566_data));
          float v1571_data = s2[48];
          float v1573_data = ir7[4];
          ir7[4] = (v1573_data + (v1550_data * v1571_data));
          float v1576_data = s2[60];
          float v1578_data = ir7[5];
          ir7[5] = (v1578_data + (v1550_data * v1576_data));
          float v1581_data = s2[72];
          float v1583_data = ir7[6];
          ir7[6] = (v1583_data + (v1550_data * v1581_data));
          float v1586_data = s2[84];
          float v1588_data = ir7[7];
          ir7[7] = (v1588_data + (v1550_data * v1586_data));
          float v1590_data = r5[1];
          float v1591_data = s2[1];
          float v1593_data = ir7[0];
          ir7[0] = (v1593_data + (v1590_data * v1591_data));
          float v1596_data = s2[13];
          float v1598_data = ir7[1];
          ir7[1] = (v1598_data + (v1590_data * v1596_data));
          float v1601_data = s2[25];
          float v1603_data = ir7[2];
          ir7[2] = (v1603_data + (v1590_data * v1601_data));
          float v1606_data = s2[37];
          float v1608_data = ir7[3];
          ir7[3] = (v1608_data + (v1590_data * v1606_data));
          float v1611_data = s2[49];
          float v1613_data = ir7[4];
          ir7[4] = (v1613_data + (v1590_data * v1611_data));
          float v1616_data = s2[61];
          float v1618_data = ir7[5];
          ir7[5] = (v1618_data + (v1590_data * v1616_data));
          float v1621_data = s2[73];
          float v1623_data = ir7[6];
          ir7[6] = (v1623_data + (v1590_data * v1621_data));
          float v1626_data = s2[85];
          float v1628_data = ir7[7];
          ir7[7] = (v1628_data + (v1590_data * v1626_data));
          float v1630_data = r5[2];
          float v1631_data = s2[2];
          float v1633_data = ir7[0];
          ir7[0] = (v1633_data + (v1630_data * v1631_data));
          float v1636_data = s2[14];
          float v1638_data = ir7[1];
          ir7[1] = (v1638_data + (v1630_data * v1636_data));
          float v1641_data = s2[26];
          float v1643_data = ir7[2];
          ir7[2] = (v1643_data + (v1630_data * v1641_data));
          float v1646_data = s2[38];
          float v1648_data = ir7[3];
          ir7[3] = (v1648_data + (v1630_data * v1646_data));
          float v1651_data = s2[50];
          float v1653_data = ir7[4];
          ir7[4] = (v1653_data + (v1630_data * v1651_data));
          float v1656_data = s2[62];
          float v1658_data = ir7[5];
          ir7[5] = (v1658_data + (v1630_data * v1656_data));
          float v1661_data = s2[74];
          float v1663_data = ir7[6];
          ir7[6] = (v1663_data + (v1630_data * v1661_data));
          float v1666_data = s2[86];
          float v1668_data = ir7[7];
          ir7[7] = (v1668_data + (v1630_data * v1666_data));
          float v1670_data = r5[3];
          float v1671_data = s2[3];
          float v1673_data = ir7[0];
          ir7[0] = (v1673_data + (v1670_data * v1671_data));
          float v1676_data = s2[15];
          float v1678_data = ir7[1];
          ir7[1] = (v1678_data + (v1670_data * v1676_data));
          float v1681_data = s2[27];
          float v1683_data = ir7[2];
          ir7[2] = (v1683_data + (v1670_data * v1681_data));
          float v1686_data = s2[39];
          float v1688_data = ir7[3];
          ir7[3] = (v1688_data + (v1670_data * v1686_data));
          float v1691_data = s2[51];
          float v1693_data = ir7[4];
          ir7[4] = (v1693_data + (v1670_data * v1691_data));
          float v1696_data = s2[63];
          float v1698_data = ir7[5];
          ir7[5] = (v1698_data + (v1670_data * v1696_data));
          float v1701_data = s2[75];
          float v1703_data = ir7[6];
          ir7[6] = (v1703_data + (v1670_data * v1701_data));
          float v1706_data = s2[87];
          float v1708_data = ir7[7];
          ir7[7] = (v1708_data + (v1670_data * v1706_data));
          float v1710_data = r5[4];
          float v1711_data = s2[4];
          float v1713_data = ir7[0];
          ir7[0] = (v1713_data + (v1710_data * v1711_data));
          float v1716_data = s2[16];
          float v1718_data = ir7[1];
          ir7[1] = (v1718_data + (v1710_data * v1716_data));
          float v1721_data = s2[28];
          float v1723_data = ir7[2];
          ir7[2] = (v1723_data + (v1710_data * v1721_data));
          float v1726_data = s2[40];
          float v1728_data = ir7[3];
          ir7[3] = (v1728_data + (v1710_data * v1726_data));
          float v1731_data = s2[52];
          float v1733_data = ir7[4];
          ir7[4] = (v1733_data + (v1710_data * v1731_data));
          float v1736_data = s2[64];
          float v1738_data = ir7[5];
          ir7[5] = (v1738_data + (v1710_data * v1736_data));
          float v1741_data = s2[76];
          float v1743_data = ir7[6];
          ir7[6] = (v1743_data + (v1710_data * v1741_data));
          float v1746_data = s2[88];
          float v1748_data = ir7[7];
          ir7[7] = (v1748_data + (v1710_data * v1746_data));
          float v1750_data = r5[5];
          float v1751_data = s2[5];
          float v1753_data = ir7[0];
          ir7[0] = (v1753_data + (v1750_data * v1751_data));
          float v1756_data = s2[17];
          float v1758_data = ir7[1];
          ir7[1] = (v1758_data + (v1750_data * v1756_data));
          float v1761_data = s2[29];
          float v1763_data = ir7[2];
          ir7[2] = (v1763_data + (v1750_data * v1761_data));
          float v1766_data = s2[41];
          float v1768_data = ir7[3];
          ir7[3] = (v1768_data + (v1750_data * v1766_data));
          float v1771_data = s2[53];
          float v1773_data = ir7[4];
          ir7[4] = (v1773_data + (v1750_data * v1771_data));
          float v1776_data = s2[65];
          float v1778_data = ir7[5];
          ir7[5] = (v1778_data + (v1750_data * v1776_data));
          float v1781_data = s2[77];
          float v1783_data = ir7[6];
          ir7[6] = (v1783_data + (v1750_data * v1781_data));
          float v1786_data = s2[89];
          float v1788_data = ir7[7];
          ir7[7] = (v1788_data + (v1750_data * v1786_data));
          float v1790_data = r5[6];
          float v1791_data = s2[6];
          float v1793_data = ir7[0];
          ir7[0] = (v1793_data + (v1790_data * v1791_data));
          float v1796_data = s2[18];
          float v1798_data = ir7[1];
          ir7[1] = (v1798_data + (v1790_data * v1796_data));
          float v1801_data = s2[30];
          float v1803_data = ir7[2];
          ir7[2] = (v1803_data + (v1790_data * v1801_data));
          float v1806_data = s2[42];
          float v1808_data = ir7[3];
          ir7[3] = (v1808_data + (v1790_data * v1806_data));
          float v1811_data = s2[54];
          float v1813_data = ir7[4];
          ir7[4] = (v1813_data + (v1790_data * v1811_data));
          float v1816_data = s2[66];
          float v1818_data = ir7[5];
          ir7[5] = (v1818_data + (v1790_data * v1816_data));
          float v1821_data = s2[78];
          float v1823_data = ir7[6];
          ir7[6] = (v1823_data + (v1790_data * v1821_data));
          float v1826_data = s2[90];
          float v1828_data = ir7[7];
          ir7[7] = (v1828_data + (v1790_data * v1826_data));
          float v1830_data = r5[7];
          float v1831_data = s2[7];
          float v1833_data = ir7[0];
          ir7[0] = (v1833_data + (v1830_data * v1831_data));
          float v1836_data = s2[19];
          float v1838_data = ir7[1];
          ir7[1] = (v1838_data + (v1830_data * v1836_data));
          float v1841_data = s2[31];
          float v1843_data = ir7[2];
          ir7[2] = (v1843_data + (v1830_data * v1841_data));
          float v1846_data = s2[43];
          float v1848_data = ir7[3];
          ir7[3] = (v1848_data + (v1830_data * v1846_data));
          float v1851_data = s2[55];
          float v1853_data = ir7[4];
          ir7[4] = (v1853_data + (v1830_data * v1851_data));
          float v1856_data = s2[67];
          float v1858_data = ir7[5];
          ir7[5] = (v1858_data + (v1830_data * v1856_data));
          float v1861_data = s2[79];
          float v1863_data = ir7[6];
          ir7[6] = (v1863_data + (v1830_data * v1861_data));
          float v1866_data = s2[91];
          float v1868_data = ir7[7];
          ir7[7] = (v1868_data + (v1830_data * v1866_data));
          float v1870_data = r5[8];
          float v1871_data = s2[8];
          float v1873_data = ir7[0];
          ir7[0] = (v1873_data + (v1870_data * v1871_data));
          float v1876_data = s2[20];
          float v1878_data = ir7[1];
          ir7[1] = (v1878_data + (v1870_data * v1876_data));
          float v1881_data = s2[32];
          float v1883_data = ir7[2];
          ir7[2] = (v1883_data + (v1870_data * v1881_data));
          float v1886_data = s2[44];
          float v1888_data = ir7[3];
          ir7[3] = (v1888_data + (v1870_data * v1886_data));
          float v1891_data = s2[56];
          float v1893_data = ir7[4];
          ir7[4] = (v1893_data + (v1870_data * v1891_data));
          float v1896_data = s2[68];
          float v1898_data = ir7[5];
          ir7[5] = (v1898_data + (v1870_data * v1896_data));
          float v1901_data = s2[80];
          float v1903_data = ir7[6];
          ir7[6] = (v1903_data + (v1870_data * v1901_data));
          float v1906_data = s2[92];
          float v1908_data = ir7[7];
          ir7[7] = (v1908_data + (v1870_data * v1906_data));
          float v1910_data = r5[9];
          float v1911_data = s2[9];
          float v1913_data = ir7[0];
          ir7[0] = (v1913_data + (v1910_data * v1911_data));
          float v1916_data = s2[21];
          float v1918_data = ir7[1];
          ir7[1] = (v1918_data + (v1910_data * v1916_data));
          float v1921_data = s2[33];
          float v1923_data = ir7[2];
          ir7[2] = (v1923_data + (v1910_data * v1921_data));
          float v1926_data = s2[45];
          float v1928_data = ir7[3];
          ir7[3] = (v1928_data + (v1910_data * v1926_data));
          float v1931_data = s2[57];
          float v1933_data = ir7[4];
          ir7[4] = (v1933_data + (v1910_data * v1931_data));
          float v1936_data = s2[69];
          float v1938_data = ir7[5];
          ir7[5] = (v1938_data + (v1910_data * v1936_data));
          float v1941_data = s2[81];
          float v1943_data = ir7[6];
          ir7[6] = (v1943_data + (v1910_data * v1941_data));
          float v1946_data = s2[93];
          float v1948_data = ir7[7];
          ir7[7] = (v1948_data + (v1910_data * v1946_data));
          float v1950_data = r5[10];
          float v1951_data = s2[10];
          float v1953_data = ir7[0];
          ir7[0] = (v1953_data + (v1950_data * v1951_data));
          float v1956_data = s2[22];
          float v1958_data = ir7[1];
          ir7[1] = (v1958_data + (v1950_data * v1956_data));
          float v1961_data = s2[34];
          float v1963_data = ir7[2];
          ir7[2] = (v1963_data + (v1950_data * v1961_data));
          float v1966_data = s2[46];
          float v1968_data = ir7[3];
          ir7[3] = (v1968_data + (v1950_data * v1966_data));
          float v1971_data = s2[58];
          float v1973_data = ir7[4];
          ir7[4] = (v1973_data + (v1950_data * v1971_data));
          float v1976_data = s2[70];
          float v1978_data = ir7[5];
          ir7[5] = (v1978_data + (v1950_data * v1976_data));
          float v1981_data = s2[82];
          float v1983_data = ir7[6];
          ir7[6] = (v1983_data + (v1950_data * v1981_data));
          float v1986_data = s2[94];
          float v1988_data = ir7[7];
          ir7[7] = (v1988_data + (v1950_data * v1986_data));
          float v1990_data = r5[11];
          float v1991_data = s2[11];
          float v1993_data = ir7[0];
          ir7[0] = (v1993_data + (v1990_data * v1991_data));
          float v1996_data = s2[23];
          float v1998_data = ir7[1];
          ir7[1] = (v1998_data + (v1990_data * v1996_data));
          float v2001_data = s2[35];
          float v2003_data = ir7[2];
          ir7[2] = (v2003_data + (v1990_data * v2001_data));
          float v2006_data = s2[47];
          float v2008_data = ir7[3];
          ir7[3] = (v2008_data + (v1990_data * v2006_data));
          float v2011_data = s2[59];
          float v2013_data = ir7[4];
          ir7[4] = (v2013_data + (v1990_data * v2011_data));
          float v2016_data = s2[71];
          float v2018_data = ir7[5];
          ir7[5] = (v2018_data + (v1990_data * v2016_data));
          float v2021_data = s2[83];
          float v2023_data = ir7[6];
          ir7[6] = (v2023_data + (v1990_data * v2021_data));
          float v2026_data = s2[95];
          float v2028_data = ir7[7];
          ir7[7] = (v2028_data + (v1990_data * v2026_data));
          #pragma unroll
          for (int32_t v2030_n0 = 0; v2030_n0 < 1; ++v2030_n0) {
            #pragma unroll
            for (int32_t v2031_n1 = 0; v2031_n1 < 8; ++v2031_n1) {
              int32_t v2032_a = v2030_n0 + v2031_n1;
              float v2033_data = ir7[v2032_a];
              float v2034_data = r6[v2032_a];
              r7[v2032_a] = (v2034_data + v2033_data);
            }
          }
          // glb_m0 = store{r>g}(r7);
          #pragma unroll
          for (int32_t v2036_i0 = 0; v2036_i0 < 1; ++v2036_i0) {
            int32_t v2041_lead = v25_lead + (v2036_i0 * 32);
            #pragma unroll
            for (int32_t v2037_i1 = 0; v2037_i1 < 8; ++v2037_i1) {
              float v2039_data = r7[(v2036_i0 + v2037_i1)];
              glb_m0[(v2041_lead + ((v2037_i1 + 8) * 32))] = v2039_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

