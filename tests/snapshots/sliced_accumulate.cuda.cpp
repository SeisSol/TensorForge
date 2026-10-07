// === base name ===
kernel_c2f8de70d1799c58

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_c2f8de70d1799c58 = {{32, 4, 1}, 32, 32, 1, 4, 3072, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_c2f8de70d1799c58(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_c2f8de70d1799c58(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_c2f8de70d1799c58(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_c2f8de70d1799c58, block.x * block.y * block.z, 768 * sizeof(float));
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
void launcher_kernel_c2f8de70d1799c58(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_c2f8de70d1799c58(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_c2f8de70d1799c58, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_c2f8de70d1799c58<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, m6, m6_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_c2f8de70d1799c58(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
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
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":768}],"shared_bytes":3072,"shared_elements":768,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[32,16]],"name":"m0","ordered":false,"parts":1,"shape":[32,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[32,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,12],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[12,16]],"name":"m2","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"A0","bbox":[[0,0],[32,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,12],"variant":false},{"addressing":"strided","alias":"B0","bbox":[[0,0],[12,8]],"name":"m4","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A1","bbox":[[0,0],[32,12]],"name":"m5","ordered":false,"parts":1,"shape":[32,12],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[12,8]],"name":"m6","ordered":false,"parts":1,"shape":[12,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,12]},{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[12,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,8]],"is_tmp":false,"name":"m0","offset":[0,8],"shape":[32,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[32,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[192 * threadIdx.y + 0];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      float * __restrict__ s2 = &localShrMem0[0];
      for (size_t v10_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v10_batchId0 < numElements0; v10_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v11_ahead1 = v10_batchId0 + (gridDim.x * blockDim.y);
        size_t v13_batchId1 = (v11_ahead1 < numElements0) ? v11_ahead1 : v10_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v10_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v10_batchId0 * 512 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v10_batchId0 * 384 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v10_batchId0 * 192 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[v10_batchId0 * 384 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[v10_batchId0 * 96 + 0 + m4_extraOffset];
          const float *const __restrict__ glb_m5 = &m5[v10_batchId0 * 384 + 0 + m5_extraOffset];
          const float *const __restrict__ glb_m6 = &m6[v10_batchId0 * 96 + 0 + m6_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v28_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v29_i0 = 0; v29_i0 < 1; ++v29_i0) {
            int32_t v32_lead = v28_lead + (v29_i0 * 32);
            #pragma unroll
            for (int32_t v30_i1 = 0; v30_i1 < 12; ++v30_i1) {
              float v35_data = __ldcg(&glb_m1[(v32_lead + (v30_i1 * 32))]);
              r0[(v29_i0 + v30_i1)] = v35_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 6; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 32], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 32], 4);
          }
          __pipeline_commit();
          float r2[12]{};
          // r2 = load{g>r}(glb_m3);
          #pragma unroll
          for (int32_t v1013_i0 = 0; v1013_i0 < 1; ++v1013_i0) {
            int32_t v1016_lead = v28_lead + (v1013_i0 * 32);
            #pragma unroll
            for (int32_t v1014_i1 = 0; v1014_i1 < 12; ++v1014_i1) {
              float v1019_data = __ldcg(&glb_m3[(v1016_lead + (v1014_i1 * 32))]);
              r2[(v1013_i0 + v1014_i1)] = v1019_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[16]{};
          // ir1 = +(r0 * s0)
          // [(0, 32), (0, 16)] [(0, 12)]
          float ir1[16]{};
          float v40_data = r0[0];
          __syncwarp();
          float v41_data = s0[0];
          float v43_data = ir1[0];
          ir1[0] = (v43_data + (v40_data * v41_data));
          float v46_data = s0[12];
          float v48_data = ir1[1];
          ir1[1] = (v48_data + (v40_data * v46_data));
          float v51_data = s0[24];
          float v53_data = ir1[2];
          ir1[2] = (v53_data + (v40_data * v51_data));
          float v56_data = s0[36];
          float v58_data = ir1[3];
          ir1[3] = (v58_data + (v40_data * v56_data));
          float v61_data = s0[48];
          float v63_data = ir1[4];
          ir1[4] = (v63_data + (v40_data * v61_data));
          float v66_data = s0[60];
          float v68_data = ir1[5];
          ir1[5] = (v68_data + (v40_data * v66_data));
          float v71_data = s0[72];
          float v73_data = ir1[6];
          ir1[6] = (v73_data + (v40_data * v71_data));
          float v76_data = s0[84];
          float v78_data = ir1[7];
          ir1[7] = (v78_data + (v40_data * v76_data));
          float v81_data = s0[96];
          float v83_data = ir1[8];
          ir1[8] = (v83_data + (v40_data * v81_data));
          float v86_data = s0[108];
          float v88_data = ir1[9];
          ir1[9] = (v88_data + (v40_data * v86_data));
          float v91_data = s0[120];
          float v93_data = ir1[10];
          ir1[10] = (v93_data + (v40_data * v91_data));
          float v96_data = s0[132];
          float v98_data = ir1[11];
          ir1[11] = (v98_data + (v40_data * v96_data));
          float v101_data = s0[144];
          float v103_data = ir1[12];
          ir1[12] = (v103_data + (v40_data * v101_data));
          float v106_data = s0[156];
          float v108_data = ir1[13];
          ir1[13] = (v108_data + (v40_data * v106_data));
          float v111_data = s0[168];
          float v113_data = ir1[14];
          ir1[14] = (v113_data + (v40_data * v111_data));
          float v116_data = s0[180];
          float v118_data = ir1[15];
          ir1[15] = (v118_data + (v40_data * v116_data));
          float v120_data = r0[1];
          float v121_data = s0[1];
          float v123_data = ir1[0];
          ir1[0] = (v123_data + (v120_data * v121_data));
          float v126_data = s0[13];
          float v128_data = ir1[1];
          ir1[1] = (v128_data + (v120_data * v126_data));
          float v131_data = s0[25];
          float v133_data = ir1[2];
          ir1[2] = (v133_data + (v120_data * v131_data));
          float v136_data = s0[37];
          float v138_data = ir1[3];
          ir1[3] = (v138_data + (v120_data * v136_data));
          float v141_data = s0[49];
          float v143_data = ir1[4];
          ir1[4] = (v143_data + (v120_data * v141_data));
          float v146_data = s0[61];
          float v148_data = ir1[5];
          ir1[5] = (v148_data + (v120_data * v146_data));
          float v151_data = s0[73];
          float v153_data = ir1[6];
          ir1[6] = (v153_data + (v120_data * v151_data));
          float v156_data = s0[85];
          float v158_data = ir1[7];
          ir1[7] = (v158_data + (v120_data * v156_data));
          float v161_data = s0[97];
          float v163_data = ir1[8];
          ir1[8] = (v163_data + (v120_data * v161_data));
          float v166_data = s0[109];
          float v168_data = ir1[9];
          ir1[9] = (v168_data + (v120_data * v166_data));
          float v171_data = s0[121];
          float v173_data = ir1[10];
          ir1[10] = (v173_data + (v120_data * v171_data));
          float v176_data = s0[133];
          float v178_data = ir1[11];
          ir1[11] = (v178_data + (v120_data * v176_data));
          float v181_data = s0[145];
          float v183_data = ir1[12];
          ir1[12] = (v183_data + (v120_data * v181_data));
          float v186_data = s0[157];
          float v188_data = ir1[13];
          ir1[13] = (v188_data + (v120_data * v186_data));
          float v191_data = s0[169];
          float v193_data = ir1[14];
          ir1[14] = (v193_data + (v120_data * v191_data));
          float v196_data = s0[181];
          float v198_data = ir1[15];
          ir1[15] = (v198_data + (v120_data * v196_data));
          float v200_data = r0[2];
          float v201_data = s0[2];
          float v203_data = ir1[0];
          ir1[0] = (v203_data + (v200_data * v201_data));
          float v206_data = s0[14];
          float v208_data = ir1[1];
          ir1[1] = (v208_data + (v200_data * v206_data));
          float v211_data = s0[26];
          float v213_data = ir1[2];
          ir1[2] = (v213_data + (v200_data * v211_data));
          float v216_data = s0[38];
          float v218_data = ir1[3];
          ir1[3] = (v218_data + (v200_data * v216_data));
          float v221_data = s0[50];
          float v223_data = ir1[4];
          ir1[4] = (v223_data + (v200_data * v221_data));
          float v226_data = s0[62];
          float v228_data = ir1[5];
          ir1[5] = (v228_data + (v200_data * v226_data));
          float v231_data = s0[74];
          float v233_data = ir1[6];
          ir1[6] = (v233_data + (v200_data * v231_data));
          float v236_data = s0[86];
          float v238_data = ir1[7];
          ir1[7] = (v238_data + (v200_data * v236_data));
          float v241_data = s0[98];
          float v243_data = ir1[8];
          ir1[8] = (v243_data + (v200_data * v241_data));
          float v246_data = s0[110];
          float v248_data = ir1[9];
          ir1[9] = (v248_data + (v200_data * v246_data));
          float v251_data = s0[122];
          float v253_data = ir1[10];
          ir1[10] = (v253_data + (v200_data * v251_data));
          float v256_data = s0[134];
          float v258_data = ir1[11];
          ir1[11] = (v258_data + (v200_data * v256_data));
          float v261_data = s0[146];
          float v263_data = ir1[12];
          ir1[12] = (v263_data + (v200_data * v261_data));
          float v266_data = s0[158];
          float v268_data = ir1[13];
          ir1[13] = (v268_data + (v200_data * v266_data));
          float v271_data = s0[170];
          float v273_data = ir1[14];
          ir1[14] = (v273_data + (v200_data * v271_data));
          float v276_data = s0[182];
          float v278_data = ir1[15];
          ir1[15] = (v278_data + (v200_data * v276_data));
          float v280_data = r0[3];
          float v281_data = s0[3];
          float v283_data = ir1[0];
          ir1[0] = (v283_data + (v280_data * v281_data));
          float v286_data = s0[15];
          float v288_data = ir1[1];
          ir1[1] = (v288_data + (v280_data * v286_data));
          float v291_data = s0[27];
          float v293_data = ir1[2];
          ir1[2] = (v293_data + (v280_data * v291_data));
          float v296_data = s0[39];
          float v298_data = ir1[3];
          ir1[3] = (v298_data + (v280_data * v296_data));
          float v301_data = s0[51];
          float v303_data = ir1[4];
          ir1[4] = (v303_data + (v280_data * v301_data));
          float v306_data = s0[63];
          float v308_data = ir1[5];
          ir1[5] = (v308_data + (v280_data * v306_data));
          float v311_data = s0[75];
          float v313_data = ir1[6];
          ir1[6] = (v313_data + (v280_data * v311_data));
          float v316_data = s0[87];
          float v318_data = ir1[7];
          ir1[7] = (v318_data + (v280_data * v316_data));
          float v321_data = s0[99];
          float v323_data = ir1[8];
          ir1[8] = (v323_data + (v280_data * v321_data));
          float v326_data = s0[111];
          float v328_data = ir1[9];
          ir1[9] = (v328_data + (v280_data * v326_data));
          float v331_data = s0[123];
          float v333_data = ir1[10];
          ir1[10] = (v333_data + (v280_data * v331_data));
          float v336_data = s0[135];
          float v338_data = ir1[11];
          ir1[11] = (v338_data + (v280_data * v336_data));
          float v341_data = s0[147];
          float v343_data = ir1[12];
          ir1[12] = (v343_data + (v280_data * v341_data));
          float v346_data = s0[159];
          float v348_data = ir1[13];
          ir1[13] = (v348_data + (v280_data * v346_data));
          float v351_data = s0[171];
          float v353_data = ir1[14];
          ir1[14] = (v353_data + (v280_data * v351_data));
          float v356_data = s0[183];
          float v358_data = ir1[15];
          ir1[15] = (v358_data + (v280_data * v356_data));
          float v360_data = r0[4];
          float v361_data = s0[4];
          float v363_data = ir1[0];
          ir1[0] = (v363_data + (v360_data * v361_data));
          float v366_data = s0[16];
          float v368_data = ir1[1];
          ir1[1] = (v368_data + (v360_data * v366_data));
          float v371_data = s0[28];
          float v373_data = ir1[2];
          ir1[2] = (v373_data + (v360_data * v371_data));
          float v376_data = s0[40];
          float v378_data = ir1[3];
          ir1[3] = (v378_data + (v360_data * v376_data));
          float v381_data = s0[52];
          float v383_data = ir1[4];
          ir1[4] = (v383_data + (v360_data * v381_data));
          float v386_data = s0[64];
          float v388_data = ir1[5];
          ir1[5] = (v388_data + (v360_data * v386_data));
          float v391_data = s0[76];
          float v393_data = ir1[6];
          ir1[6] = (v393_data + (v360_data * v391_data));
          float v396_data = s0[88];
          float v398_data = ir1[7];
          ir1[7] = (v398_data + (v360_data * v396_data));
          float v401_data = s0[100];
          float v403_data = ir1[8];
          ir1[8] = (v403_data + (v360_data * v401_data));
          float v406_data = s0[112];
          float v408_data = ir1[9];
          ir1[9] = (v408_data + (v360_data * v406_data));
          float v411_data = s0[124];
          float v413_data = ir1[10];
          ir1[10] = (v413_data + (v360_data * v411_data));
          float v416_data = s0[136];
          float v418_data = ir1[11];
          ir1[11] = (v418_data + (v360_data * v416_data));
          float v421_data = s0[148];
          float v423_data = ir1[12];
          ir1[12] = (v423_data + (v360_data * v421_data));
          float v426_data = s0[160];
          float v428_data = ir1[13];
          ir1[13] = (v428_data + (v360_data * v426_data));
          float v431_data = s0[172];
          float v433_data = ir1[14];
          ir1[14] = (v433_data + (v360_data * v431_data));
          float v436_data = s0[184];
          float v438_data = ir1[15];
          ir1[15] = (v438_data + (v360_data * v436_data));
          float v440_data = r0[5];
          float v441_data = s0[5];
          float v443_data = ir1[0];
          ir1[0] = (v443_data + (v440_data * v441_data));
          float v446_data = s0[17];
          float v448_data = ir1[1];
          ir1[1] = (v448_data + (v440_data * v446_data));
          float v451_data = s0[29];
          float v453_data = ir1[2];
          ir1[2] = (v453_data + (v440_data * v451_data));
          float v456_data = s0[41];
          float v458_data = ir1[3];
          ir1[3] = (v458_data + (v440_data * v456_data));
          float v461_data = s0[53];
          float v463_data = ir1[4];
          ir1[4] = (v463_data + (v440_data * v461_data));
          float v466_data = s0[65];
          float v468_data = ir1[5];
          ir1[5] = (v468_data + (v440_data * v466_data));
          float v471_data = s0[77];
          float v473_data = ir1[6];
          ir1[6] = (v473_data + (v440_data * v471_data));
          float v476_data = s0[89];
          float v478_data = ir1[7];
          ir1[7] = (v478_data + (v440_data * v476_data));
          float v481_data = s0[101];
          float v483_data = ir1[8];
          ir1[8] = (v483_data + (v440_data * v481_data));
          float v486_data = s0[113];
          float v488_data = ir1[9];
          ir1[9] = (v488_data + (v440_data * v486_data));
          float v491_data = s0[125];
          float v493_data = ir1[10];
          ir1[10] = (v493_data + (v440_data * v491_data));
          float v496_data = s0[137];
          float v498_data = ir1[11];
          ir1[11] = (v498_data + (v440_data * v496_data));
          float v501_data = s0[149];
          float v503_data = ir1[12];
          ir1[12] = (v503_data + (v440_data * v501_data));
          float v506_data = s0[161];
          float v508_data = ir1[13];
          ir1[13] = (v508_data + (v440_data * v506_data));
          float v511_data = s0[173];
          float v513_data = ir1[14];
          ir1[14] = (v513_data + (v440_data * v511_data));
          float v516_data = s0[185];
          float v518_data = ir1[15];
          ir1[15] = (v518_data + (v440_data * v516_data));
          float v520_data = r0[6];
          float v521_data = s0[6];
          float v523_data = ir1[0];
          ir1[0] = (v523_data + (v520_data * v521_data));
          float v526_data = s0[18];
          float v528_data = ir1[1];
          ir1[1] = (v528_data + (v520_data * v526_data));
          float v531_data = s0[30];
          float v533_data = ir1[2];
          ir1[2] = (v533_data + (v520_data * v531_data));
          float v536_data = s0[42];
          float v538_data = ir1[3];
          ir1[3] = (v538_data + (v520_data * v536_data));
          float v541_data = s0[54];
          float v543_data = ir1[4];
          ir1[4] = (v543_data + (v520_data * v541_data));
          float v546_data = s0[66];
          float v548_data = ir1[5];
          ir1[5] = (v548_data + (v520_data * v546_data));
          float v551_data = s0[78];
          float v553_data = ir1[6];
          ir1[6] = (v553_data + (v520_data * v551_data));
          float v556_data = s0[90];
          float v558_data = ir1[7];
          ir1[7] = (v558_data + (v520_data * v556_data));
          float v561_data = s0[102];
          float v563_data = ir1[8];
          ir1[8] = (v563_data + (v520_data * v561_data));
          float v566_data = s0[114];
          float v568_data = ir1[9];
          ir1[9] = (v568_data + (v520_data * v566_data));
          float v571_data = s0[126];
          float v573_data = ir1[10];
          ir1[10] = (v573_data + (v520_data * v571_data));
          float v576_data = s0[138];
          float v578_data = ir1[11];
          ir1[11] = (v578_data + (v520_data * v576_data));
          float v581_data = s0[150];
          float v583_data = ir1[12];
          ir1[12] = (v583_data + (v520_data * v581_data));
          float v586_data = s0[162];
          float v588_data = ir1[13];
          ir1[13] = (v588_data + (v520_data * v586_data));
          float v591_data = s0[174];
          float v593_data = ir1[14];
          ir1[14] = (v593_data + (v520_data * v591_data));
          float v596_data = s0[186];
          float v598_data = ir1[15];
          ir1[15] = (v598_data + (v520_data * v596_data));
          float v600_data = r0[7];
          float v601_data = s0[7];
          float v603_data = ir1[0];
          ir1[0] = (v603_data + (v600_data * v601_data));
          float v606_data = s0[19];
          float v608_data = ir1[1];
          ir1[1] = (v608_data + (v600_data * v606_data));
          float v611_data = s0[31];
          float v613_data = ir1[2];
          ir1[2] = (v613_data + (v600_data * v611_data));
          float v616_data = s0[43];
          float v618_data = ir1[3];
          ir1[3] = (v618_data + (v600_data * v616_data));
          float v621_data = s0[55];
          float v623_data = ir1[4];
          ir1[4] = (v623_data + (v600_data * v621_data));
          float v626_data = s0[67];
          float v628_data = ir1[5];
          ir1[5] = (v628_data + (v600_data * v626_data));
          float v631_data = s0[79];
          float v633_data = ir1[6];
          ir1[6] = (v633_data + (v600_data * v631_data));
          float v636_data = s0[91];
          float v638_data = ir1[7];
          ir1[7] = (v638_data + (v600_data * v636_data));
          float v641_data = s0[103];
          float v643_data = ir1[8];
          ir1[8] = (v643_data + (v600_data * v641_data));
          float v646_data = s0[115];
          float v648_data = ir1[9];
          ir1[9] = (v648_data + (v600_data * v646_data));
          float v651_data = s0[127];
          float v653_data = ir1[10];
          ir1[10] = (v653_data + (v600_data * v651_data));
          float v656_data = s0[139];
          float v658_data = ir1[11];
          ir1[11] = (v658_data + (v600_data * v656_data));
          float v661_data = s0[151];
          float v663_data = ir1[12];
          ir1[12] = (v663_data + (v600_data * v661_data));
          float v666_data = s0[163];
          float v668_data = ir1[13];
          ir1[13] = (v668_data + (v600_data * v666_data));
          float v671_data = s0[175];
          float v673_data = ir1[14];
          ir1[14] = (v673_data + (v600_data * v671_data));
          float v676_data = s0[187];
          float v678_data = ir1[15];
          ir1[15] = (v678_data + (v600_data * v676_data));
          float v680_data = r0[8];
          float v681_data = s0[8];
          float v683_data = ir1[0];
          ir1[0] = (v683_data + (v680_data * v681_data));
          float v686_data = s0[20];
          float v688_data = ir1[1];
          ir1[1] = (v688_data + (v680_data * v686_data));
          float v691_data = s0[32];
          float v693_data = ir1[2];
          ir1[2] = (v693_data + (v680_data * v691_data));
          float v696_data = s0[44];
          float v698_data = ir1[3];
          ir1[3] = (v698_data + (v680_data * v696_data));
          float v701_data = s0[56];
          float v703_data = ir1[4];
          ir1[4] = (v703_data + (v680_data * v701_data));
          float v706_data = s0[68];
          float v708_data = ir1[5];
          ir1[5] = (v708_data + (v680_data * v706_data));
          float v711_data = s0[80];
          float v713_data = ir1[6];
          ir1[6] = (v713_data + (v680_data * v711_data));
          float v716_data = s0[92];
          float v718_data = ir1[7];
          ir1[7] = (v718_data + (v680_data * v716_data));
          float v721_data = s0[104];
          float v723_data = ir1[8];
          ir1[8] = (v723_data + (v680_data * v721_data));
          float v726_data = s0[116];
          float v728_data = ir1[9];
          ir1[9] = (v728_data + (v680_data * v726_data));
          float v731_data = s0[128];
          float v733_data = ir1[10];
          ir1[10] = (v733_data + (v680_data * v731_data));
          float v736_data = s0[140];
          float v738_data = ir1[11];
          ir1[11] = (v738_data + (v680_data * v736_data));
          float v741_data = s0[152];
          float v743_data = ir1[12];
          ir1[12] = (v743_data + (v680_data * v741_data));
          float v746_data = s0[164];
          float v748_data = ir1[13];
          ir1[13] = (v748_data + (v680_data * v746_data));
          float v751_data = s0[176];
          float v753_data = ir1[14];
          ir1[14] = (v753_data + (v680_data * v751_data));
          float v756_data = s0[188];
          float v758_data = ir1[15];
          ir1[15] = (v758_data + (v680_data * v756_data));
          float v760_data = r0[9];
          float v761_data = s0[9];
          float v763_data = ir1[0];
          ir1[0] = (v763_data + (v760_data * v761_data));
          float v766_data = s0[21];
          float v768_data = ir1[1];
          ir1[1] = (v768_data + (v760_data * v766_data));
          float v771_data = s0[33];
          float v773_data = ir1[2];
          ir1[2] = (v773_data + (v760_data * v771_data));
          float v776_data = s0[45];
          float v778_data = ir1[3];
          ir1[3] = (v778_data + (v760_data * v776_data));
          float v781_data = s0[57];
          float v783_data = ir1[4];
          ir1[4] = (v783_data + (v760_data * v781_data));
          float v786_data = s0[69];
          float v788_data = ir1[5];
          ir1[5] = (v788_data + (v760_data * v786_data));
          float v791_data = s0[81];
          float v793_data = ir1[6];
          ir1[6] = (v793_data + (v760_data * v791_data));
          float v796_data = s0[93];
          float v798_data = ir1[7];
          ir1[7] = (v798_data + (v760_data * v796_data));
          float v801_data = s0[105];
          float v803_data = ir1[8];
          ir1[8] = (v803_data + (v760_data * v801_data));
          float v806_data = s0[117];
          float v808_data = ir1[9];
          ir1[9] = (v808_data + (v760_data * v806_data));
          float v811_data = s0[129];
          float v813_data = ir1[10];
          ir1[10] = (v813_data + (v760_data * v811_data));
          float v816_data = s0[141];
          float v818_data = ir1[11];
          ir1[11] = (v818_data + (v760_data * v816_data));
          float v821_data = s0[153];
          float v823_data = ir1[12];
          ir1[12] = (v823_data + (v760_data * v821_data));
          float v826_data = s0[165];
          float v828_data = ir1[13];
          ir1[13] = (v828_data + (v760_data * v826_data));
          float v831_data = s0[177];
          float v833_data = ir1[14];
          ir1[14] = (v833_data + (v760_data * v831_data));
          float v836_data = s0[189];
          float v838_data = ir1[15];
          ir1[15] = (v838_data + (v760_data * v836_data));
          float v840_data = r0[10];
          float v841_data = s0[10];
          float v843_data = ir1[0];
          ir1[0] = (v843_data + (v840_data * v841_data));
          float v846_data = s0[22];
          float v848_data = ir1[1];
          ir1[1] = (v848_data + (v840_data * v846_data));
          float v851_data = s0[34];
          float v853_data = ir1[2];
          ir1[2] = (v853_data + (v840_data * v851_data));
          float v856_data = s0[46];
          float v858_data = ir1[3];
          ir1[3] = (v858_data + (v840_data * v856_data));
          float v861_data = s0[58];
          float v863_data = ir1[4];
          ir1[4] = (v863_data + (v840_data * v861_data));
          float v866_data = s0[70];
          float v868_data = ir1[5];
          ir1[5] = (v868_data + (v840_data * v866_data));
          float v871_data = s0[82];
          float v873_data = ir1[6];
          ir1[6] = (v873_data + (v840_data * v871_data));
          float v876_data = s0[94];
          float v878_data = ir1[7];
          ir1[7] = (v878_data + (v840_data * v876_data));
          float v881_data = s0[106];
          float v883_data = ir1[8];
          ir1[8] = (v883_data + (v840_data * v881_data));
          float v886_data = s0[118];
          float v888_data = ir1[9];
          ir1[9] = (v888_data + (v840_data * v886_data));
          float v891_data = s0[130];
          float v893_data = ir1[10];
          ir1[10] = (v893_data + (v840_data * v891_data));
          float v896_data = s0[142];
          float v898_data = ir1[11];
          ir1[11] = (v898_data + (v840_data * v896_data));
          float v901_data = s0[154];
          float v903_data = ir1[12];
          ir1[12] = (v903_data + (v840_data * v901_data));
          float v906_data = s0[166];
          float v908_data = ir1[13];
          ir1[13] = (v908_data + (v840_data * v906_data));
          float v911_data = s0[178];
          float v913_data = ir1[14];
          ir1[14] = (v913_data + (v840_data * v911_data));
          float v916_data = s0[190];
          float v918_data = ir1[15];
          ir1[15] = (v918_data + (v840_data * v916_data));
          float v920_data = r0[11];
          float v921_data = s0[11];
          float v923_data = ir1[0];
          ir1[0] = (v923_data + (v920_data * v921_data));
          float v926_data = s0[23];
          float v928_data = ir1[1];
          ir1[1] = (v928_data + (v920_data * v926_data));
          float v931_data = s0[35];
          float v933_data = ir1[2];
          ir1[2] = (v933_data + (v920_data * v931_data));
          float v936_data = s0[47];
          float v938_data = ir1[3];
          ir1[3] = (v938_data + (v920_data * v936_data));
          float v941_data = s0[59];
          float v943_data = ir1[4];
          ir1[4] = (v943_data + (v920_data * v941_data));
          float v946_data = s0[71];
          float v948_data = ir1[5];
          ir1[5] = (v948_data + (v920_data * v946_data));
          float v951_data = s0[83];
          float v953_data = ir1[6];
          ir1[6] = (v953_data + (v920_data * v951_data));
          float v956_data = s0[95];
          float v958_data = ir1[7];
          ir1[7] = (v958_data + (v920_data * v956_data));
          float v961_data = s0[107];
          float v963_data = ir1[8];
          ir1[8] = (v963_data + (v920_data * v961_data));
          float v966_data = s0[119];
          float v968_data = ir1[9];
          ir1[9] = (v968_data + (v920_data * v966_data));
          float v971_data = s0[131];
          float v973_data = ir1[10];
          ir1[10] = (v973_data + (v920_data * v971_data));
          float v976_data = s0[143];
          float v978_data = ir1[11];
          ir1[11] = (v978_data + (v920_data * v976_data));
          float v981_data = s0[155];
          float v983_data = ir1[12];
          ir1[12] = (v983_data + (v920_data * v981_data));
          float v986_data = s0[167];
          float v988_data = ir1[13];
          ir1[13] = (v988_data + (v920_data * v986_data));
          float v991_data = s0[179];
          float v993_data = ir1[14];
          ir1[14] = (v993_data + (v920_data * v991_data));
          float v996_data = s0[191];
          float v998_data = ir1[15];
          ir1[15] = (v998_data + (v920_data * v996_data));
          // r1 = ir1
          #pragma unroll
          for (int32_t v1000_n0 = 0; v1000_n0 < 1; ++v1000_n0) {
            #pragma unroll
            for (int32_t v1001_n1 = 0; v1001_n1 < 16; ++v1001_n1) {
              int32_t v1002_a = v1000_n0 + v1001_n1;
              float v1003_data = ir1[v1002_a];
              r1[v1002_a] = v1003_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v1004_i0 = 0; v1004_i0 < 1; ++v1004_i0) {
            int32_t v1009_lead = v28_lead + (v1004_i0 * 32);
            #pragma unroll
            for (int32_t v1005_i1 = 0; v1005_i1 < 16; ++v1005_i1) {
              float v1007_data = r1[(v1004_i0 + v1005_i1)];
              glb_m0[(v1009_lead + (v1005_i1 * 32))] = v1007_data;
            }
          }
          // s1 = load{g>s}(glb_m4[0, 1])
          __syncwarp();
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 0], &glb_m4[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 32], &glb_m4[0 + 0 + 1 * threadIdx.x + 32], 4);
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 64], &glb_m4[0 + 0 + 1 * threadIdx.x + 64], 4);
          __pipeline_commit();
          float r3[8]{};
          // r3 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v1025_i0 = 0; v1025_i0 < 1; ++v1025_i0) {
            int32_t v1028_lead = v28_lead + (v1025_i0 * 32);
            #pragma unroll
            for (int32_t v1026_i1 = 0; v1026_i1 < 8; ++v1026_i1) {
              float v1031_data = glb_m0[(v1028_lead + (v1026_i1 * 32))];
              r3[(v1025_i0 + v1026_i1)] = v1031_data;
            }
          }
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          float r5[12]{};
          // r5 = load{g>r}(glb_m5);
          #pragma unroll
          for (int32_t v1530_i0 = 0; v1530_i0 < 1; ++v1530_i0) {
            int32_t v1533_lead = v28_lead + (v1530_i0 * 32);
            #pragma unroll
            for (int32_t v1531_i1 = 0; v1531_i1 < 12; ++v1531_i1) {
              float v1536_data = __ldcg(&glb_m5[(v1533_lead + (v1531_i1 * 32))]);
              r5[(v1530_i0 + v1531_i1)] = v1536_data;
            }
          }
          float r4[8]{};
          // ir4 = +(r2 * s1)
          // [(0, 32), (0, 8)] [(0, 12)]
          float ir4[8]{};
          float v1035_data = r2[0];
          __syncwarp();
          float v1036_data = s1[0];
          float v1038_data = ir4[0];
          ir4[0] = (v1038_data + (v1035_data * v1036_data));
          float v1041_data = s1[12];
          float v1043_data = ir4[1];
          ir4[1] = (v1043_data + (v1035_data * v1041_data));
          float v1046_data = s1[24];
          float v1048_data = ir4[2];
          ir4[2] = (v1048_data + (v1035_data * v1046_data));
          float v1051_data = s1[36];
          float v1053_data = ir4[3];
          ir4[3] = (v1053_data + (v1035_data * v1051_data));
          float v1056_data = s1[48];
          float v1058_data = ir4[4];
          ir4[4] = (v1058_data + (v1035_data * v1056_data));
          float v1061_data = s1[60];
          float v1063_data = ir4[5];
          ir4[5] = (v1063_data + (v1035_data * v1061_data));
          float v1066_data = s1[72];
          float v1068_data = ir4[6];
          ir4[6] = (v1068_data + (v1035_data * v1066_data));
          float v1071_data = s1[84];
          float v1073_data = ir4[7];
          ir4[7] = (v1073_data + (v1035_data * v1071_data));
          float v1075_data = r2[1];
          float v1076_data = s1[1];
          float v1078_data = ir4[0];
          ir4[0] = (v1078_data + (v1075_data * v1076_data));
          float v1081_data = s1[13];
          float v1083_data = ir4[1];
          ir4[1] = (v1083_data + (v1075_data * v1081_data));
          float v1086_data = s1[25];
          float v1088_data = ir4[2];
          ir4[2] = (v1088_data + (v1075_data * v1086_data));
          float v1091_data = s1[37];
          float v1093_data = ir4[3];
          ir4[3] = (v1093_data + (v1075_data * v1091_data));
          float v1096_data = s1[49];
          float v1098_data = ir4[4];
          ir4[4] = (v1098_data + (v1075_data * v1096_data));
          float v1101_data = s1[61];
          float v1103_data = ir4[5];
          ir4[5] = (v1103_data + (v1075_data * v1101_data));
          float v1106_data = s1[73];
          float v1108_data = ir4[6];
          ir4[6] = (v1108_data + (v1075_data * v1106_data));
          float v1111_data = s1[85];
          float v1113_data = ir4[7];
          ir4[7] = (v1113_data + (v1075_data * v1111_data));
          float v1115_data = r2[2];
          float v1116_data = s1[2];
          float v1118_data = ir4[0];
          ir4[0] = (v1118_data + (v1115_data * v1116_data));
          float v1121_data = s1[14];
          float v1123_data = ir4[1];
          ir4[1] = (v1123_data + (v1115_data * v1121_data));
          float v1126_data = s1[26];
          float v1128_data = ir4[2];
          ir4[2] = (v1128_data + (v1115_data * v1126_data));
          float v1131_data = s1[38];
          float v1133_data = ir4[3];
          ir4[3] = (v1133_data + (v1115_data * v1131_data));
          float v1136_data = s1[50];
          float v1138_data = ir4[4];
          ir4[4] = (v1138_data + (v1115_data * v1136_data));
          float v1141_data = s1[62];
          float v1143_data = ir4[5];
          ir4[5] = (v1143_data + (v1115_data * v1141_data));
          float v1146_data = s1[74];
          float v1148_data = ir4[6];
          ir4[6] = (v1148_data + (v1115_data * v1146_data));
          float v1151_data = s1[86];
          float v1153_data = ir4[7];
          ir4[7] = (v1153_data + (v1115_data * v1151_data));
          float v1155_data = r2[3];
          float v1156_data = s1[3];
          float v1158_data = ir4[0];
          ir4[0] = (v1158_data + (v1155_data * v1156_data));
          float v1161_data = s1[15];
          float v1163_data = ir4[1];
          ir4[1] = (v1163_data + (v1155_data * v1161_data));
          float v1166_data = s1[27];
          float v1168_data = ir4[2];
          ir4[2] = (v1168_data + (v1155_data * v1166_data));
          float v1171_data = s1[39];
          float v1173_data = ir4[3];
          ir4[3] = (v1173_data + (v1155_data * v1171_data));
          float v1176_data = s1[51];
          float v1178_data = ir4[4];
          ir4[4] = (v1178_data + (v1155_data * v1176_data));
          float v1181_data = s1[63];
          float v1183_data = ir4[5];
          ir4[5] = (v1183_data + (v1155_data * v1181_data));
          float v1186_data = s1[75];
          float v1188_data = ir4[6];
          ir4[6] = (v1188_data + (v1155_data * v1186_data));
          float v1191_data = s1[87];
          float v1193_data = ir4[7];
          ir4[7] = (v1193_data + (v1155_data * v1191_data));
          float v1195_data = r2[4];
          float v1196_data = s1[4];
          float v1198_data = ir4[0];
          ir4[0] = (v1198_data + (v1195_data * v1196_data));
          float v1201_data = s1[16];
          float v1203_data = ir4[1];
          ir4[1] = (v1203_data + (v1195_data * v1201_data));
          float v1206_data = s1[28];
          float v1208_data = ir4[2];
          ir4[2] = (v1208_data + (v1195_data * v1206_data));
          float v1211_data = s1[40];
          float v1213_data = ir4[3];
          ir4[3] = (v1213_data + (v1195_data * v1211_data));
          float v1216_data = s1[52];
          float v1218_data = ir4[4];
          ir4[4] = (v1218_data + (v1195_data * v1216_data));
          float v1221_data = s1[64];
          float v1223_data = ir4[5];
          ir4[5] = (v1223_data + (v1195_data * v1221_data));
          float v1226_data = s1[76];
          float v1228_data = ir4[6];
          ir4[6] = (v1228_data + (v1195_data * v1226_data));
          float v1231_data = s1[88];
          float v1233_data = ir4[7];
          ir4[7] = (v1233_data + (v1195_data * v1231_data));
          float v1235_data = r2[5];
          float v1236_data = s1[5];
          float v1238_data = ir4[0];
          ir4[0] = (v1238_data + (v1235_data * v1236_data));
          float v1241_data = s1[17];
          float v1243_data = ir4[1];
          ir4[1] = (v1243_data + (v1235_data * v1241_data));
          float v1246_data = s1[29];
          float v1248_data = ir4[2];
          ir4[2] = (v1248_data + (v1235_data * v1246_data));
          float v1251_data = s1[41];
          float v1253_data = ir4[3];
          ir4[3] = (v1253_data + (v1235_data * v1251_data));
          float v1256_data = s1[53];
          float v1258_data = ir4[4];
          ir4[4] = (v1258_data + (v1235_data * v1256_data));
          float v1261_data = s1[65];
          float v1263_data = ir4[5];
          ir4[5] = (v1263_data + (v1235_data * v1261_data));
          float v1266_data = s1[77];
          float v1268_data = ir4[6];
          ir4[6] = (v1268_data + (v1235_data * v1266_data));
          float v1271_data = s1[89];
          float v1273_data = ir4[7];
          ir4[7] = (v1273_data + (v1235_data * v1271_data));
          float v1275_data = r2[6];
          float v1276_data = s1[6];
          float v1278_data = ir4[0];
          ir4[0] = (v1278_data + (v1275_data * v1276_data));
          float v1281_data = s1[18];
          float v1283_data = ir4[1];
          ir4[1] = (v1283_data + (v1275_data * v1281_data));
          float v1286_data = s1[30];
          float v1288_data = ir4[2];
          ir4[2] = (v1288_data + (v1275_data * v1286_data));
          float v1291_data = s1[42];
          float v1293_data = ir4[3];
          ir4[3] = (v1293_data + (v1275_data * v1291_data));
          float v1296_data = s1[54];
          float v1298_data = ir4[4];
          ir4[4] = (v1298_data + (v1275_data * v1296_data));
          float v1301_data = s1[66];
          float v1303_data = ir4[5];
          ir4[5] = (v1303_data + (v1275_data * v1301_data));
          float v1306_data = s1[78];
          float v1308_data = ir4[6];
          ir4[6] = (v1308_data + (v1275_data * v1306_data));
          float v1311_data = s1[90];
          float v1313_data = ir4[7];
          ir4[7] = (v1313_data + (v1275_data * v1311_data));
          float v1315_data = r2[7];
          float v1316_data = s1[7];
          float v1318_data = ir4[0];
          ir4[0] = (v1318_data + (v1315_data * v1316_data));
          float v1321_data = s1[19];
          float v1323_data = ir4[1];
          ir4[1] = (v1323_data + (v1315_data * v1321_data));
          float v1326_data = s1[31];
          float v1328_data = ir4[2];
          ir4[2] = (v1328_data + (v1315_data * v1326_data));
          float v1331_data = s1[43];
          float v1333_data = ir4[3];
          ir4[3] = (v1333_data + (v1315_data * v1331_data));
          float v1336_data = s1[55];
          float v1338_data = ir4[4];
          ir4[4] = (v1338_data + (v1315_data * v1336_data));
          float v1341_data = s1[67];
          float v1343_data = ir4[5];
          ir4[5] = (v1343_data + (v1315_data * v1341_data));
          float v1346_data = s1[79];
          float v1348_data = ir4[6];
          ir4[6] = (v1348_data + (v1315_data * v1346_data));
          float v1351_data = s1[91];
          float v1353_data = ir4[7];
          ir4[7] = (v1353_data + (v1315_data * v1351_data));
          float v1355_data = r2[8];
          float v1356_data = s1[8];
          float v1358_data = ir4[0];
          ir4[0] = (v1358_data + (v1355_data * v1356_data));
          float v1361_data = s1[20];
          float v1363_data = ir4[1];
          ir4[1] = (v1363_data + (v1355_data * v1361_data));
          float v1366_data = s1[32];
          float v1368_data = ir4[2];
          ir4[2] = (v1368_data + (v1355_data * v1366_data));
          float v1371_data = s1[44];
          float v1373_data = ir4[3];
          ir4[3] = (v1373_data + (v1355_data * v1371_data));
          float v1376_data = s1[56];
          float v1378_data = ir4[4];
          ir4[4] = (v1378_data + (v1355_data * v1376_data));
          float v1381_data = s1[68];
          float v1383_data = ir4[5];
          ir4[5] = (v1383_data + (v1355_data * v1381_data));
          float v1386_data = s1[80];
          float v1388_data = ir4[6];
          ir4[6] = (v1388_data + (v1355_data * v1386_data));
          float v1391_data = s1[92];
          float v1393_data = ir4[7];
          ir4[7] = (v1393_data + (v1355_data * v1391_data));
          float v1395_data = r2[9];
          float v1396_data = s1[9];
          float v1398_data = ir4[0];
          ir4[0] = (v1398_data + (v1395_data * v1396_data));
          float v1401_data = s1[21];
          float v1403_data = ir4[1];
          ir4[1] = (v1403_data + (v1395_data * v1401_data));
          float v1406_data = s1[33];
          float v1408_data = ir4[2];
          ir4[2] = (v1408_data + (v1395_data * v1406_data));
          float v1411_data = s1[45];
          float v1413_data = ir4[3];
          ir4[3] = (v1413_data + (v1395_data * v1411_data));
          float v1416_data = s1[57];
          float v1418_data = ir4[4];
          ir4[4] = (v1418_data + (v1395_data * v1416_data));
          float v1421_data = s1[69];
          float v1423_data = ir4[5];
          ir4[5] = (v1423_data + (v1395_data * v1421_data));
          float v1426_data = s1[81];
          float v1428_data = ir4[6];
          ir4[6] = (v1428_data + (v1395_data * v1426_data));
          float v1431_data = s1[93];
          float v1433_data = ir4[7];
          ir4[7] = (v1433_data + (v1395_data * v1431_data));
          float v1435_data = r2[10];
          float v1436_data = s1[10];
          float v1438_data = ir4[0];
          ir4[0] = (v1438_data + (v1435_data * v1436_data));
          float v1441_data = s1[22];
          float v1443_data = ir4[1];
          ir4[1] = (v1443_data + (v1435_data * v1441_data));
          float v1446_data = s1[34];
          float v1448_data = ir4[2];
          ir4[2] = (v1448_data + (v1435_data * v1446_data));
          float v1451_data = s1[46];
          float v1453_data = ir4[3];
          ir4[3] = (v1453_data + (v1435_data * v1451_data));
          float v1456_data = s1[58];
          float v1458_data = ir4[4];
          ir4[4] = (v1458_data + (v1435_data * v1456_data));
          float v1461_data = s1[70];
          float v1463_data = ir4[5];
          ir4[5] = (v1463_data + (v1435_data * v1461_data));
          float v1466_data = s1[82];
          float v1468_data = ir4[6];
          ir4[6] = (v1468_data + (v1435_data * v1466_data));
          float v1471_data = s1[94];
          float v1473_data = ir4[7];
          ir4[7] = (v1473_data + (v1435_data * v1471_data));
          float v1475_data = r2[11];
          float v1476_data = s1[11];
          float v1478_data = ir4[0];
          ir4[0] = (v1478_data + (v1475_data * v1476_data));
          float v1481_data = s1[23];
          float v1483_data = ir4[1];
          ir4[1] = (v1483_data + (v1475_data * v1481_data));
          float v1486_data = s1[35];
          float v1488_data = ir4[2];
          ir4[2] = (v1488_data + (v1475_data * v1486_data));
          float v1491_data = s1[47];
          float v1493_data = ir4[3];
          ir4[3] = (v1493_data + (v1475_data * v1491_data));
          float v1496_data = s1[59];
          float v1498_data = ir4[4];
          ir4[4] = (v1498_data + (v1475_data * v1496_data));
          float v1501_data = s1[71];
          float v1503_data = ir4[5];
          ir4[5] = (v1503_data + (v1475_data * v1501_data));
          float v1506_data = s1[83];
          float v1508_data = ir4[6];
          ir4[6] = (v1508_data + (v1475_data * v1506_data));
          float v1511_data = s1[95];
          float v1513_data = ir4[7];
          ir4[7] = (v1513_data + (v1475_data * v1511_data));
          // r4 = ir4 + r3
          #pragma unroll
          for (int32_t v1515_n0 = 0; v1515_n0 < 1; ++v1515_n0) {
            #pragma unroll
            for (int32_t v1516_n1 = 0; v1516_n1 < 8; ++v1516_n1) {
              int32_t v1517_a = v1515_n0 + v1516_n1;
              float v1518_data = ir4[v1517_a];
              float v1519_data = r3[v1517_a];
              r4[v1517_a] = (v1519_data + v1518_data);
            }
          }
          // glb_m0 = store{r>g}(r4);
          #pragma unroll
          for (int32_t v1521_i0 = 0; v1521_i0 < 1; ++v1521_i0) {
            int32_t v1526_lead = v28_lead + (v1521_i0 * 32);
            #pragma unroll
            for (int32_t v1522_i1 = 0; v1522_i1 < 8; ++v1522_i1) {
              float v1524_data = r4[(v1521_i0 + v1522_i1)];
              glb_m0[(v1526_lead + (v1522_i1 * 32))] = v1524_data;
            }
          }
          // s2 = load{g>s}(glb_m6[0, 1])
          __syncwarp();
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 0], &glb_m6[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 32], &glb_m6[0 + 0 + 1 * threadIdx.x + 32], 4);
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 64], &glb_m6[0 + 0 + 1 * threadIdx.x + 64], 4);
          __pipeline_commit();
          float r6[8]{};
          // r6 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v1542_i0 = 0; v1542_i0 < 1; ++v1542_i0) {
            int32_t v1545_lead = v28_lead + (v1542_i0 * 32);
            #pragma unroll
            for (int32_t v1543_i1 = 0; v1543_i1 < 8; ++v1543_i1) {
              float v1549_data = glb_m0[(v1545_lead + ((v1543_i1 + 8) * 32))];
              r6[(v1542_i0 + v1543_i1)] = v1549_data;
            }
          }
          // wait(s2 = load{g>s}(glb_m6[0, 1]));
          __pipeline_wait_prior(0);
          float r7[8]{};
          // ir7 = +(r5 * s2)
          // [(0, 32), (0, 8)] [(0, 12)]
          float ir7[8]{};
          float v1553_data = r5[0];
          __syncwarp();
          float v1554_data = s2[0];
          float v1556_data = ir7[0];
          ir7[0] = (v1556_data + (v1553_data * v1554_data));
          float v1559_data = s2[12];
          float v1561_data = ir7[1];
          ir7[1] = (v1561_data + (v1553_data * v1559_data));
          float v1564_data = s2[24];
          float v1566_data = ir7[2];
          ir7[2] = (v1566_data + (v1553_data * v1564_data));
          float v1569_data = s2[36];
          float v1571_data = ir7[3];
          ir7[3] = (v1571_data + (v1553_data * v1569_data));
          float v1574_data = s2[48];
          float v1576_data = ir7[4];
          ir7[4] = (v1576_data + (v1553_data * v1574_data));
          float v1579_data = s2[60];
          float v1581_data = ir7[5];
          ir7[5] = (v1581_data + (v1553_data * v1579_data));
          float v1584_data = s2[72];
          float v1586_data = ir7[6];
          ir7[6] = (v1586_data + (v1553_data * v1584_data));
          float v1589_data = s2[84];
          float v1591_data = ir7[7];
          ir7[7] = (v1591_data + (v1553_data * v1589_data));
          float v1593_data = r5[1];
          float v1594_data = s2[1];
          float v1596_data = ir7[0];
          ir7[0] = (v1596_data + (v1593_data * v1594_data));
          float v1599_data = s2[13];
          float v1601_data = ir7[1];
          ir7[1] = (v1601_data + (v1593_data * v1599_data));
          float v1604_data = s2[25];
          float v1606_data = ir7[2];
          ir7[2] = (v1606_data + (v1593_data * v1604_data));
          float v1609_data = s2[37];
          float v1611_data = ir7[3];
          ir7[3] = (v1611_data + (v1593_data * v1609_data));
          float v1614_data = s2[49];
          float v1616_data = ir7[4];
          ir7[4] = (v1616_data + (v1593_data * v1614_data));
          float v1619_data = s2[61];
          float v1621_data = ir7[5];
          ir7[5] = (v1621_data + (v1593_data * v1619_data));
          float v1624_data = s2[73];
          float v1626_data = ir7[6];
          ir7[6] = (v1626_data + (v1593_data * v1624_data));
          float v1629_data = s2[85];
          float v1631_data = ir7[7];
          ir7[7] = (v1631_data + (v1593_data * v1629_data));
          float v1633_data = r5[2];
          float v1634_data = s2[2];
          float v1636_data = ir7[0];
          ir7[0] = (v1636_data + (v1633_data * v1634_data));
          float v1639_data = s2[14];
          float v1641_data = ir7[1];
          ir7[1] = (v1641_data + (v1633_data * v1639_data));
          float v1644_data = s2[26];
          float v1646_data = ir7[2];
          ir7[2] = (v1646_data + (v1633_data * v1644_data));
          float v1649_data = s2[38];
          float v1651_data = ir7[3];
          ir7[3] = (v1651_data + (v1633_data * v1649_data));
          float v1654_data = s2[50];
          float v1656_data = ir7[4];
          ir7[4] = (v1656_data + (v1633_data * v1654_data));
          float v1659_data = s2[62];
          float v1661_data = ir7[5];
          ir7[5] = (v1661_data + (v1633_data * v1659_data));
          float v1664_data = s2[74];
          float v1666_data = ir7[6];
          ir7[6] = (v1666_data + (v1633_data * v1664_data));
          float v1669_data = s2[86];
          float v1671_data = ir7[7];
          ir7[7] = (v1671_data + (v1633_data * v1669_data));
          float v1673_data = r5[3];
          float v1674_data = s2[3];
          float v1676_data = ir7[0];
          ir7[0] = (v1676_data + (v1673_data * v1674_data));
          float v1679_data = s2[15];
          float v1681_data = ir7[1];
          ir7[1] = (v1681_data + (v1673_data * v1679_data));
          float v1684_data = s2[27];
          float v1686_data = ir7[2];
          ir7[2] = (v1686_data + (v1673_data * v1684_data));
          float v1689_data = s2[39];
          float v1691_data = ir7[3];
          ir7[3] = (v1691_data + (v1673_data * v1689_data));
          float v1694_data = s2[51];
          float v1696_data = ir7[4];
          ir7[4] = (v1696_data + (v1673_data * v1694_data));
          float v1699_data = s2[63];
          float v1701_data = ir7[5];
          ir7[5] = (v1701_data + (v1673_data * v1699_data));
          float v1704_data = s2[75];
          float v1706_data = ir7[6];
          ir7[6] = (v1706_data + (v1673_data * v1704_data));
          float v1709_data = s2[87];
          float v1711_data = ir7[7];
          ir7[7] = (v1711_data + (v1673_data * v1709_data));
          float v1713_data = r5[4];
          float v1714_data = s2[4];
          float v1716_data = ir7[0];
          ir7[0] = (v1716_data + (v1713_data * v1714_data));
          float v1719_data = s2[16];
          float v1721_data = ir7[1];
          ir7[1] = (v1721_data + (v1713_data * v1719_data));
          float v1724_data = s2[28];
          float v1726_data = ir7[2];
          ir7[2] = (v1726_data + (v1713_data * v1724_data));
          float v1729_data = s2[40];
          float v1731_data = ir7[3];
          ir7[3] = (v1731_data + (v1713_data * v1729_data));
          float v1734_data = s2[52];
          float v1736_data = ir7[4];
          ir7[4] = (v1736_data + (v1713_data * v1734_data));
          float v1739_data = s2[64];
          float v1741_data = ir7[5];
          ir7[5] = (v1741_data + (v1713_data * v1739_data));
          float v1744_data = s2[76];
          float v1746_data = ir7[6];
          ir7[6] = (v1746_data + (v1713_data * v1744_data));
          float v1749_data = s2[88];
          float v1751_data = ir7[7];
          ir7[7] = (v1751_data + (v1713_data * v1749_data));
          float v1753_data = r5[5];
          float v1754_data = s2[5];
          float v1756_data = ir7[0];
          ir7[0] = (v1756_data + (v1753_data * v1754_data));
          float v1759_data = s2[17];
          float v1761_data = ir7[1];
          ir7[1] = (v1761_data + (v1753_data * v1759_data));
          float v1764_data = s2[29];
          float v1766_data = ir7[2];
          ir7[2] = (v1766_data + (v1753_data * v1764_data));
          float v1769_data = s2[41];
          float v1771_data = ir7[3];
          ir7[3] = (v1771_data + (v1753_data * v1769_data));
          float v1774_data = s2[53];
          float v1776_data = ir7[4];
          ir7[4] = (v1776_data + (v1753_data * v1774_data));
          float v1779_data = s2[65];
          float v1781_data = ir7[5];
          ir7[5] = (v1781_data + (v1753_data * v1779_data));
          float v1784_data = s2[77];
          float v1786_data = ir7[6];
          ir7[6] = (v1786_data + (v1753_data * v1784_data));
          float v1789_data = s2[89];
          float v1791_data = ir7[7];
          ir7[7] = (v1791_data + (v1753_data * v1789_data));
          float v1793_data = r5[6];
          float v1794_data = s2[6];
          float v1796_data = ir7[0];
          ir7[0] = (v1796_data + (v1793_data * v1794_data));
          float v1799_data = s2[18];
          float v1801_data = ir7[1];
          ir7[1] = (v1801_data + (v1793_data * v1799_data));
          float v1804_data = s2[30];
          float v1806_data = ir7[2];
          ir7[2] = (v1806_data + (v1793_data * v1804_data));
          float v1809_data = s2[42];
          float v1811_data = ir7[3];
          ir7[3] = (v1811_data + (v1793_data * v1809_data));
          float v1814_data = s2[54];
          float v1816_data = ir7[4];
          ir7[4] = (v1816_data + (v1793_data * v1814_data));
          float v1819_data = s2[66];
          float v1821_data = ir7[5];
          ir7[5] = (v1821_data + (v1793_data * v1819_data));
          float v1824_data = s2[78];
          float v1826_data = ir7[6];
          ir7[6] = (v1826_data + (v1793_data * v1824_data));
          float v1829_data = s2[90];
          float v1831_data = ir7[7];
          ir7[7] = (v1831_data + (v1793_data * v1829_data));
          float v1833_data = r5[7];
          float v1834_data = s2[7];
          float v1836_data = ir7[0];
          ir7[0] = (v1836_data + (v1833_data * v1834_data));
          float v1839_data = s2[19];
          float v1841_data = ir7[1];
          ir7[1] = (v1841_data + (v1833_data * v1839_data));
          float v1844_data = s2[31];
          float v1846_data = ir7[2];
          ir7[2] = (v1846_data + (v1833_data * v1844_data));
          float v1849_data = s2[43];
          float v1851_data = ir7[3];
          ir7[3] = (v1851_data + (v1833_data * v1849_data));
          float v1854_data = s2[55];
          float v1856_data = ir7[4];
          ir7[4] = (v1856_data + (v1833_data * v1854_data));
          float v1859_data = s2[67];
          float v1861_data = ir7[5];
          ir7[5] = (v1861_data + (v1833_data * v1859_data));
          float v1864_data = s2[79];
          float v1866_data = ir7[6];
          ir7[6] = (v1866_data + (v1833_data * v1864_data));
          float v1869_data = s2[91];
          float v1871_data = ir7[7];
          ir7[7] = (v1871_data + (v1833_data * v1869_data));
          float v1873_data = r5[8];
          float v1874_data = s2[8];
          float v1876_data = ir7[0];
          ir7[0] = (v1876_data + (v1873_data * v1874_data));
          float v1879_data = s2[20];
          float v1881_data = ir7[1];
          ir7[1] = (v1881_data + (v1873_data * v1879_data));
          float v1884_data = s2[32];
          float v1886_data = ir7[2];
          ir7[2] = (v1886_data + (v1873_data * v1884_data));
          float v1889_data = s2[44];
          float v1891_data = ir7[3];
          ir7[3] = (v1891_data + (v1873_data * v1889_data));
          float v1894_data = s2[56];
          float v1896_data = ir7[4];
          ir7[4] = (v1896_data + (v1873_data * v1894_data));
          float v1899_data = s2[68];
          float v1901_data = ir7[5];
          ir7[5] = (v1901_data + (v1873_data * v1899_data));
          float v1904_data = s2[80];
          float v1906_data = ir7[6];
          ir7[6] = (v1906_data + (v1873_data * v1904_data));
          float v1909_data = s2[92];
          float v1911_data = ir7[7];
          ir7[7] = (v1911_data + (v1873_data * v1909_data));
          float v1913_data = r5[9];
          float v1914_data = s2[9];
          float v1916_data = ir7[0];
          ir7[0] = (v1916_data + (v1913_data * v1914_data));
          float v1919_data = s2[21];
          float v1921_data = ir7[1];
          ir7[1] = (v1921_data + (v1913_data * v1919_data));
          float v1924_data = s2[33];
          float v1926_data = ir7[2];
          ir7[2] = (v1926_data + (v1913_data * v1924_data));
          float v1929_data = s2[45];
          float v1931_data = ir7[3];
          ir7[3] = (v1931_data + (v1913_data * v1929_data));
          float v1934_data = s2[57];
          float v1936_data = ir7[4];
          ir7[4] = (v1936_data + (v1913_data * v1934_data));
          float v1939_data = s2[69];
          float v1941_data = ir7[5];
          ir7[5] = (v1941_data + (v1913_data * v1939_data));
          float v1944_data = s2[81];
          float v1946_data = ir7[6];
          ir7[6] = (v1946_data + (v1913_data * v1944_data));
          float v1949_data = s2[93];
          float v1951_data = ir7[7];
          ir7[7] = (v1951_data + (v1913_data * v1949_data));
          float v1953_data = r5[10];
          float v1954_data = s2[10];
          float v1956_data = ir7[0];
          ir7[0] = (v1956_data + (v1953_data * v1954_data));
          float v1959_data = s2[22];
          float v1961_data = ir7[1];
          ir7[1] = (v1961_data + (v1953_data * v1959_data));
          float v1964_data = s2[34];
          float v1966_data = ir7[2];
          ir7[2] = (v1966_data + (v1953_data * v1964_data));
          float v1969_data = s2[46];
          float v1971_data = ir7[3];
          ir7[3] = (v1971_data + (v1953_data * v1969_data));
          float v1974_data = s2[58];
          float v1976_data = ir7[4];
          ir7[4] = (v1976_data + (v1953_data * v1974_data));
          float v1979_data = s2[70];
          float v1981_data = ir7[5];
          ir7[5] = (v1981_data + (v1953_data * v1979_data));
          float v1984_data = s2[82];
          float v1986_data = ir7[6];
          ir7[6] = (v1986_data + (v1953_data * v1984_data));
          float v1989_data = s2[94];
          float v1991_data = ir7[7];
          ir7[7] = (v1991_data + (v1953_data * v1989_data));
          float v1993_data = r5[11];
          float v1994_data = s2[11];
          float v1996_data = ir7[0];
          ir7[0] = (v1996_data + (v1993_data * v1994_data));
          float v1999_data = s2[23];
          float v2001_data = ir7[1];
          ir7[1] = (v2001_data + (v1993_data * v1999_data));
          float v2004_data = s2[35];
          float v2006_data = ir7[2];
          ir7[2] = (v2006_data + (v1993_data * v2004_data));
          float v2009_data = s2[47];
          float v2011_data = ir7[3];
          ir7[3] = (v2011_data + (v1993_data * v2009_data));
          float v2014_data = s2[59];
          float v2016_data = ir7[4];
          ir7[4] = (v2016_data + (v1993_data * v2014_data));
          float v2019_data = s2[71];
          float v2021_data = ir7[5];
          ir7[5] = (v2021_data + (v1993_data * v2019_data));
          float v2024_data = s2[83];
          float v2026_data = ir7[6];
          ir7[6] = (v2026_data + (v1993_data * v2024_data));
          float v2029_data = s2[95];
          float v2031_data = ir7[7];
          ir7[7] = (v2031_data + (v1993_data * v2029_data));
          // r7 = ir7 + r6
          #pragma unroll
          for (int32_t v2033_n0 = 0; v2033_n0 < 1; ++v2033_n0) {
            #pragma unroll
            for (int32_t v2034_n1 = 0; v2034_n1 < 8; ++v2034_n1) {
              int32_t v2035_a = v2033_n0 + v2034_n1;
              float v2036_data = ir7[v2035_a];
              float v2037_data = r6[v2035_a];
              r7[v2035_a] = (v2037_data + v2036_data);
            }
          }
          // glb_m0 = store{r>g}(r7);
          #pragma unroll
          for (int32_t v2039_i0 = 0; v2039_i0 < 1; ++v2039_i0) {
            int32_t v2044_lead = v28_lead + (v2039_i0 * 32);
            #pragma unroll
            for (int32_t v2040_i1 = 0; v2040_i1 < 8; ++v2040_i1) {
              float v2042_data = r7[(v2039_i0 + v2040_i1)];
              glb_m0[(v2044_lead + ((v2040_i1 + 8) * 32))] = v2042_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

