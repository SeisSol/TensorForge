// === base name ===
kernel_76878adbb45718e3

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_76878adbb45718e3 = {{16, 8, 1}, 16, 12, 1, 8, 5632, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_76878adbb45718e3(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_76878adbb45718e3(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_76878adbb45718e3(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_76878adbb45718e3, block.x * block.y * block.z, 1408 * sizeof(float));
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
  config.sharedMemBytes = 1408 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_76878adbb45718e3(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_76878adbb45718e3(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_76878adbb45718e3, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_76878adbb45718e3<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_76878adbb45718e3(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 8 per block = block 16x8x1, 5632 B shared, occupancy grid
    // operands:
    //   m0 6×12(6×12) {0..6}×{0..12} strided
    //   m1 12×12(12×12) {0..12}×{0..12} strided
    //   m2 6×12(6×12) {0..6}×{0..12} strided
    //   m3 12×12(12×12) {0..12}×{0..12} strided
    //   m4 2×12(2×12) {0..2}×{0..12} strided
    //   m5 12×12(12×12) {0..12}×{0..12} strided
    // operations:
    //   t0[i,j]@{0..6}×{0..12} = m0[i,k] × m1[k,j]
    //   t0[i,j]@{6..12}×{0..12} = m2[i,k] × m1[k,j]
    //   m3[i,j] = t0[i,j]
    //   t0[i,j]@{6..12}×{0..12} = m4[i,k] × m1[k,j]
    //   m5[i,j] = t0[i,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1408}],"shared_bytes":5632,"shared_elements":1408,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B1","bbox":[[0,0],[6,12]],"name":"m0","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"X","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"N2","bbox":[[0,0],[2,12]],"name":"m4","ordered":false,"parts":1,"shape":[2,12],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m5","ordered":false,"parts":1,"shape":[12,12],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[2,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[2,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[176 * threadIdx.y + 0];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < numElements0; v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 72 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v9_batchId0 * 144 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v9_batchId0 * 72 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[v9_batchId0 * 144 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[v9_batchId0 * 24 + 0 + m4_extraOffset];
          float *const __restrict__ glb_m5 = &m5[v9_batchId0 * 144 + 0 + m5_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v26_lead = threadIdx.x % 16;
          bool v27_g = v26_lead < 6;
          if (v27_g) {
            #pragma unroll
            for (int32_t v28_i1 = 0; v28_i1 < 12; ++v28_i1) {
              float v33_data = __ldcg(&glb_m0[(v26_lead + (v28_i1 * 6))]);
              r0[v28_i1] = v33_data;
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
          if (v27_g) {
            #pragma unroll
            for (int32_t v37_i1 = 0; v37_i1 < 12; ++v37_i1) {
              float v42_data = __ldcg(&glb_m2[(v26_lead + (v37_i1 * 6))]);
              r2[v37_i1] = v42_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[12]{};
          // r1 = +(r0 * s0) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v45_data = r0[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v46_data = s0[0];
          float v48_data = r1[0];
          r1[0] = (v48_data + (v45_data * v46_data));
          float v51_data = s0[12];
          float v53_data = r1[1];
          r1[1] = (v53_data + (v45_data * v51_data));
          float v56_data = s0[24];
          float v58_data = r1[2];
          r1[2] = (v58_data + (v45_data * v56_data));
          float v61_data = s0[36];
          float v63_data = r1[3];
          r1[3] = (v63_data + (v45_data * v61_data));
          float v66_data = s0[48];
          float v68_data = r1[4];
          r1[4] = (v68_data + (v45_data * v66_data));
          float v71_data = s0[60];
          float v73_data = r1[5];
          r1[5] = (v73_data + (v45_data * v71_data));
          float v76_data = s0[72];
          float v78_data = r1[6];
          r1[6] = (v78_data + (v45_data * v76_data));
          float v81_data = s0[84];
          float v83_data = r1[7];
          r1[7] = (v83_data + (v45_data * v81_data));
          float v86_data = s0[96];
          float v88_data = r1[8];
          r1[8] = (v88_data + (v45_data * v86_data));
          float v91_data = s0[108];
          float v93_data = r1[9];
          r1[9] = (v93_data + (v45_data * v91_data));
          float v96_data = s0[120];
          float v98_data = r1[10];
          r1[10] = (v98_data + (v45_data * v96_data));
          float v101_data = s0[132];
          float v103_data = r1[11];
          r1[11] = (v103_data + (v45_data * v101_data));
          float v105_data = r0[1];
          float v106_data = s0[1];
          float v108_data = r1[0];
          r1[0] = (v108_data + (v105_data * v106_data));
          float v111_data = s0[13];
          float v113_data = r1[1];
          r1[1] = (v113_data + (v105_data * v111_data));
          float v116_data = s0[25];
          float v118_data = r1[2];
          r1[2] = (v118_data + (v105_data * v116_data));
          float v121_data = s0[37];
          float v123_data = r1[3];
          r1[3] = (v123_data + (v105_data * v121_data));
          float v126_data = s0[49];
          float v128_data = r1[4];
          r1[4] = (v128_data + (v105_data * v126_data));
          float v131_data = s0[61];
          float v133_data = r1[5];
          r1[5] = (v133_data + (v105_data * v131_data));
          float v136_data = s0[73];
          float v138_data = r1[6];
          r1[6] = (v138_data + (v105_data * v136_data));
          float v141_data = s0[85];
          float v143_data = r1[7];
          r1[7] = (v143_data + (v105_data * v141_data));
          float v146_data = s0[97];
          float v148_data = r1[8];
          r1[8] = (v148_data + (v105_data * v146_data));
          float v151_data = s0[109];
          float v153_data = r1[9];
          r1[9] = (v153_data + (v105_data * v151_data));
          float v156_data = s0[121];
          float v158_data = r1[10];
          r1[10] = (v158_data + (v105_data * v156_data));
          float v161_data = s0[133];
          float v163_data = r1[11];
          r1[11] = (v163_data + (v105_data * v161_data));
          float v165_data = r0[2];
          float v166_data = s0[2];
          float v168_data = r1[0];
          r1[0] = (v168_data + (v165_data * v166_data));
          float v171_data = s0[14];
          float v173_data = r1[1];
          r1[1] = (v173_data + (v165_data * v171_data));
          float v176_data = s0[26];
          float v178_data = r1[2];
          r1[2] = (v178_data + (v165_data * v176_data));
          float v181_data = s0[38];
          float v183_data = r1[3];
          r1[3] = (v183_data + (v165_data * v181_data));
          float v186_data = s0[50];
          float v188_data = r1[4];
          r1[4] = (v188_data + (v165_data * v186_data));
          float v191_data = s0[62];
          float v193_data = r1[5];
          r1[5] = (v193_data + (v165_data * v191_data));
          float v196_data = s0[74];
          float v198_data = r1[6];
          r1[6] = (v198_data + (v165_data * v196_data));
          float v201_data = s0[86];
          float v203_data = r1[7];
          r1[7] = (v203_data + (v165_data * v201_data));
          float v206_data = s0[98];
          float v208_data = r1[8];
          r1[8] = (v208_data + (v165_data * v206_data));
          float v211_data = s0[110];
          float v213_data = r1[9];
          r1[9] = (v213_data + (v165_data * v211_data));
          float v216_data = s0[122];
          float v218_data = r1[10];
          r1[10] = (v218_data + (v165_data * v216_data));
          float v221_data = s0[134];
          float v223_data = r1[11];
          r1[11] = (v223_data + (v165_data * v221_data));
          float v225_data = r0[3];
          float v226_data = s0[3];
          float v228_data = r1[0];
          r1[0] = (v228_data + (v225_data * v226_data));
          float v231_data = s0[15];
          float v233_data = r1[1];
          r1[1] = (v233_data + (v225_data * v231_data));
          float v236_data = s0[27];
          float v238_data = r1[2];
          r1[2] = (v238_data + (v225_data * v236_data));
          float v241_data = s0[39];
          float v243_data = r1[3];
          r1[3] = (v243_data + (v225_data * v241_data));
          float v246_data = s0[51];
          float v248_data = r1[4];
          r1[4] = (v248_data + (v225_data * v246_data));
          float v251_data = s0[63];
          float v253_data = r1[5];
          r1[5] = (v253_data + (v225_data * v251_data));
          float v256_data = s0[75];
          float v258_data = r1[6];
          r1[6] = (v258_data + (v225_data * v256_data));
          float v261_data = s0[87];
          float v263_data = r1[7];
          r1[7] = (v263_data + (v225_data * v261_data));
          float v266_data = s0[99];
          float v268_data = r1[8];
          r1[8] = (v268_data + (v225_data * v266_data));
          float v271_data = s0[111];
          float v273_data = r1[9];
          r1[9] = (v273_data + (v225_data * v271_data));
          float v276_data = s0[123];
          float v278_data = r1[10];
          r1[10] = (v278_data + (v225_data * v276_data));
          float v281_data = s0[135];
          float v283_data = r1[11];
          r1[11] = (v283_data + (v225_data * v281_data));
          float v285_data = r0[4];
          float v286_data = s0[4];
          float v288_data = r1[0];
          r1[0] = (v288_data + (v285_data * v286_data));
          float v291_data = s0[16];
          float v293_data = r1[1];
          r1[1] = (v293_data + (v285_data * v291_data));
          float v296_data = s0[28];
          float v298_data = r1[2];
          r1[2] = (v298_data + (v285_data * v296_data));
          float v301_data = s0[40];
          float v303_data = r1[3];
          r1[3] = (v303_data + (v285_data * v301_data));
          float v306_data = s0[52];
          float v308_data = r1[4];
          r1[4] = (v308_data + (v285_data * v306_data));
          float v311_data = s0[64];
          float v313_data = r1[5];
          r1[5] = (v313_data + (v285_data * v311_data));
          float v316_data = s0[76];
          float v318_data = r1[6];
          r1[6] = (v318_data + (v285_data * v316_data));
          float v321_data = s0[88];
          float v323_data = r1[7];
          r1[7] = (v323_data + (v285_data * v321_data));
          float v326_data = s0[100];
          float v328_data = r1[8];
          r1[8] = (v328_data + (v285_data * v326_data));
          float v331_data = s0[112];
          float v333_data = r1[9];
          r1[9] = (v333_data + (v285_data * v331_data));
          float v336_data = s0[124];
          float v338_data = r1[10];
          r1[10] = (v338_data + (v285_data * v336_data));
          float v341_data = s0[136];
          float v343_data = r1[11];
          r1[11] = (v343_data + (v285_data * v341_data));
          float v345_data = r0[5];
          float v346_data = s0[5];
          float v348_data = r1[0];
          r1[0] = (v348_data + (v345_data * v346_data));
          float v351_data = s0[17];
          float v353_data = r1[1];
          r1[1] = (v353_data + (v345_data * v351_data));
          float v356_data = s0[29];
          float v358_data = r1[2];
          r1[2] = (v358_data + (v345_data * v356_data));
          float v361_data = s0[41];
          float v363_data = r1[3];
          r1[3] = (v363_data + (v345_data * v361_data));
          float v366_data = s0[53];
          float v368_data = r1[4];
          r1[4] = (v368_data + (v345_data * v366_data));
          float v371_data = s0[65];
          float v373_data = r1[5];
          r1[5] = (v373_data + (v345_data * v371_data));
          float v376_data = s0[77];
          float v378_data = r1[6];
          r1[6] = (v378_data + (v345_data * v376_data));
          float v381_data = s0[89];
          float v383_data = r1[7];
          r1[7] = (v383_data + (v345_data * v381_data));
          float v386_data = s0[101];
          float v388_data = r1[8];
          r1[8] = (v388_data + (v345_data * v386_data));
          float v391_data = s0[113];
          float v393_data = r1[9];
          r1[9] = (v393_data + (v345_data * v391_data));
          float v396_data = s0[125];
          float v398_data = r1[10];
          r1[10] = (v398_data + (v345_data * v396_data));
          float v401_data = s0[137];
          float v403_data = r1[11];
          r1[11] = (v403_data + (v345_data * v401_data));
          float v405_data = r0[6];
          float v406_data = s0[6];
          float v408_data = r1[0];
          r1[0] = (v408_data + (v405_data * v406_data));
          float v411_data = s0[18];
          float v413_data = r1[1];
          r1[1] = (v413_data + (v405_data * v411_data));
          float v416_data = s0[30];
          float v418_data = r1[2];
          r1[2] = (v418_data + (v405_data * v416_data));
          float v421_data = s0[42];
          float v423_data = r1[3];
          r1[3] = (v423_data + (v405_data * v421_data));
          float v426_data = s0[54];
          float v428_data = r1[4];
          r1[4] = (v428_data + (v405_data * v426_data));
          float v431_data = s0[66];
          float v433_data = r1[5];
          r1[5] = (v433_data + (v405_data * v431_data));
          float v436_data = s0[78];
          float v438_data = r1[6];
          r1[6] = (v438_data + (v405_data * v436_data));
          float v441_data = s0[90];
          float v443_data = r1[7];
          r1[7] = (v443_data + (v405_data * v441_data));
          float v446_data = s0[102];
          float v448_data = r1[8];
          r1[8] = (v448_data + (v405_data * v446_data));
          float v451_data = s0[114];
          float v453_data = r1[9];
          r1[9] = (v453_data + (v405_data * v451_data));
          float v456_data = s0[126];
          float v458_data = r1[10];
          r1[10] = (v458_data + (v405_data * v456_data));
          float v461_data = s0[138];
          float v463_data = r1[11];
          r1[11] = (v463_data + (v405_data * v461_data));
          float v465_data = r0[7];
          float v466_data = s0[7];
          float v468_data = r1[0];
          r1[0] = (v468_data + (v465_data * v466_data));
          float v471_data = s0[19];
          float v473_data = r1[1];
          r1[1] = (v473_data + (v465_data * v471_data));
          float v476_data = s0[31];
          float v478_data = r1[2];
          r1[2] = (v478_data + (v465_data * v476_data));
          float v481_data = s0[43];
          float v483_data = r1[3];
          r1[3] = (v483_data + (v465_data * v481_data));
          float v486_data = s0[55];
          float v488_data = r1[4];
          r1[4] = (v488_data + (v465_data * v486_data));
          float v491_data = s0[67];
          float v493_data = r1[5];
          r1[5] = (v493_data + (v465_data * v491_data));
          float v496_data = s0[79];
          float v498_data = r1[6];
          r1[6] = (v498_data + (v465_data * v496_data));
          float v501_data = s0[91];
          float v503_data = r1[7];
          r1[7] = (v503_data + (v465_data * v501_data));
          float v506_data = s0[103];
          float v508_data = r1[8];
          r1[8] = (v508_data + (v465_data * v506_data));
          float v511_data = s0[115];
          float v513_data = r1[9];
          r1[9] = (v513_data + (v465_data * v511_data));
          float v516_data = s0[127];
          float v518_data = r1[10];
          r1[10] = (v518_data + (v465_data * v516_data));
          float v521_data = s0[139];
          float v523_data = r1[11];
          r1[11] = (v523_data + (v465_data * v521_data));
          float v525_data = r0[8];
          float v526_data = s0[8];
          float v528_data = r1[0];
          r1[0] = (v528_data + (v525_data * v526_data));
          float v531_data = s0[20];
          float v533_data = r1[1];
          r1[1] = (v533_data + (v525_data * v531_data));
          float v536_data = s0[32];
          float v538_data = r1[2];
          r1[2] = (v538_data + (v525_data * v536_data));
          float v541_data = s0[44];
          float v543_data = r1[3];
          r1[3] = (v543_data + (v525_data * v541_data));
          float v546_data = s0[56];
          float v548_data = r1[4];
          r1[4] = (v548_data + (v525_data * v546_data));
          float v551_data = s0[68];
          float v553_data = r1[5];
          r1[5] = (v553_data + (v525_data * v551_data));
          float v556_data = s0[80];
          float v558_data = r1[6];
          r1[6] = (v558_data + (v525_data * v556_data));
          float v561_data = s0[92];
          float v563_data = r1[7];
          r1[7] = (v563_data + (v525_data * v561_data));
          float v566_data = s0[104];
          float v568_data = r1[8];
          r1[8] = (v568_data + (v525_data * v566_data));
          float v571_data = s0[116];
          float v573_data = r1[9];
          r1[9] = (v573_data + (v525_data * v571_data));
          float v576_data = s0[128];
          float v578_data = r1[10];
          r1[10] = (v578_data + (v525_data * v576_data));
          float v581_data = s0[140];
          float v583_data = r1[11];
          r1[11] = (v583_data + (v525_data * v581_data));
          float v585_data = r0[9];
          float v586_data = s0[9];
          float v588_data = r1[0];
          r1[0] = (v588_data + (v585_data * v586_data));
          float v591_data = s0[21];
          float v593_data = r1[1];
          r1[1] = (v593_data + (v585_data * v591_data));
          float v596_data = s0[33];
          float v598_data = r1[2];
          r1[2] = (v598_data + (v585_data * v596_data));
          float v601_data = s0[45];
          float v603_data = r1[3];
          r1[3] = (v603_data + (v585_data * v601_data));
          float v606_data = s0[57];
          float v608_data = r1[4];
          r1[4] = (v608_data + (v585_data * v606_data));
          float v611_data = s0[69];
          float v613_data = r1[5];
          r1[5] = (v613_data + (v585_data * v611_data));
          float v616_data = s0[81];
          float v618_data = r1[6];
          r1[6] = (v618_data + (v585_data * v616_data));
          float v621_data = s0[93];
          float v623_data = r1[7];
          r1[7] = (v623_data + (v585_data * v621_data));
          float v626_data = s0[105];
          float v628_data = r1[8];
          r1[8] = (v628_data + (v585_data * v626_data));
          float v631_data = s0[117];
          float v633_data = r1[9];
          r1[9] = (v633_data + (v585_data * v631_data));
          float v636_data = s0[129];
          float v638_data = r1[10];
          r1[10] = (v638_data + (v585_data * v636_data));
          float v641_data = s0[141];
          float v643_data = r1[11];
          r1[11] = (v643_data + (v585_data * v641_data));
          float v645_data = r0[10];
          float v646_data = s0[10];
          float v648_data = r1[0];
          r1[0] = (v648_data + (v645_data * v646_data));
          float v651_data = s0[22];
          float v653_data = r1[1];
          r1[1] = (v653_data + (v645_data * v651_data));
          float v656_data = s0[34];
          float v658_data = r1[2];
          r1[2] = (v658_data + (v645_data * v656_data));
          float v661_data = s0[46];
          float v663_data = r1[3];
          r1[3] = (v663_data + (v645_data * v661_data));
          float v666_data = s0[58];
          float v668_data = r1[4];
          r1[4] = (v668_data + (v645_data * v666_data));
          float v671_data = s0[70];
          float v673_data = r1[5];
          r1[5] = (v673_data + (v645_data * v671_data));
          float v676_data = s0[82];
          float v678_data = r1[6];
          r1[6] = (v678_data + (v645_data * v676_data));
          float v681_data = s0[94];
          float v683_data = r1[7];
          r1[7] = (v683_data + (v645_data * v681_data));
          float v686_data = s0[106];
          float v688_data = r1[8];
          r1[8] = (v688_data + (v645_data * v686_data));
          float v691_data = s0[118];
          float v693_data = r1[9];
          r1[9] = (v693_data + (v645_data * v691_data));
          float v696_data = s0[130];
          float v698_data = r1[10];
          r1[10] = (v698_data + (v645_data * v696_data));
          float v701_data = s0[142];
          float v703_data = r1[11];
          r1[11] = (v703_data + (v645_data * v701_data));
          float v705_data = r0[11];
          float v706_data = s0[11];
          float v708_data = r1[0];
          r1[0] = (v708_data + (v705_data * v706_data));
          float v711_data = s0[23];
          float v713_data = r1[1];
          r1[1] = (v713_data + (v705_data * v711_data));
          float v716_data = s0[35];
          float v718_data = r1[2];
          r1[2] = (v718_data + (v705_data * v716_data));
          float v721_data = s0[47];
          float v723_data = r1[3];
          r1[3] = (v723_data + (v705_data * v721_data));
          float v726_data = s0[59];
          float v728_data = r1[4];
          r1[4] = (v728_data + (v705_data * v726_data));
          float v731_data = s0[71];
          float v733_data = r1[5];
          r1[5] = (v733_data + (v705_data * v731_data));
          float v736_data = s0[83];
          float v738_data = r1[6];
          r1[6] = (v738_data + (v705_data * v736_data));
          float v741_data = s0[95];
          float v743_data = r1[7];
          r1[7] = (v743_data + (v705_data * v741_data));
          float v746_data = s0[107];
          float v748_data = r1[8];
          r1[8] = (v748_data + (v705_data * v746_data));
          float v751_data = s0[119];
          float v753_data = r1[9];
          r1[9] = (v753_data + (v705_data * v751_data));
          float v756_data = s0[131];
          float v758_data = r1[10];
          r1[10] = (v758_data + (v705_data * v756_data));
          float v761_data = s0[143];
          float v763_data = r1[11];
          r1[11] = (v763_data + (v705_data * v761_data));
          // s1 = store{r>s}(localShrMem0, r1);
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          if (v27_g) {
            #pragma unroll
            for (int32_t v765_i1 = 0; v765_i1 < 12; ++v765_i1) {
              float v767_data = r1[v765_i1];
              int32_t v771_a = v26_lead + (v765_i1 * 12);
              s1[(v771_a ^ ((v771_a >> 4) & 15))] = v767_data;
            }
          }
          float r5[12]{};
          // r5 = load{g>r}(glb_m4);
          bool v776_g = v26_lead < 2;
          if (v776_g) {
            #pragma unroll
            for (int32_t v777_i1 = 0; v777_i1 < 12; ++v777_i1) {
              float v782_data = __ldcg(&glb_m4[(v26_lead + (v777_i1 * 2))]);
              r5[v777_i1] = v782_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m2););
          float r3[12]{};
          // ir3 = +(r2 * s0)
          // [(0, 6), (0, 12)] [(0, 12)]
          float ir3[12]{};
          float v786_data = r2[0];
          float v789_data = ir3[0];
          ir3[0] = (v789_data + (v786_data * v46_data));
          float v794_data = ir3[1];
          ir3[1] = (v794_data + (v786_data * v51_data));
          float v799_data = ir3[2];
          ir3[2] = (v799_data + (v786_data * v56_data));
          float v804_data = ir3[3];
          ir3[3] = (v804_data + (v786_data * v61_data));
          float v809_data = ir3[4];
          ir3[4] = (v809_data + (v786_data * v66_data));
          float v814_data = ir3[5];
          ir3[5] = (v814_data + (v786_data * v71_data));
          float v819_data = ir3[6];
          ir3[6] = (v819_data + (v786_data * v76_data));
          float v824_data = ir3[7];
          ir3[7] = (v824_data + (v786_data * v81_data));
          float v829_data = ir3[8];
          ir3[8] = (v829_data + (v786_data * v86_data));
          float v834_data = ir3[9];
          ir3[9] = (v834_data + (v786_data * v91_data));
          float v839_data = ir3[10];
          ir3[10] = (v839_data + (v786_data * v96_data));
          float v844_data = ir3[11];
          ir3[11] = (v844_data + (v786_data * v101_data));
          float v846_data = r2[1];
          float v849_data = ir3[0];
          ir3[0] = (v849_data + (v846_data * v106_data));
          float v854_data = ir3[1];
          ir3[1] = (v854_data + (v846_data * v111_data));
          float v859_data = ir3[2];
          ir3[2] = (v859_data + (v846_data * v116_data));
          float v864_data = ir3[3];
          ir3[3] = (v864_data + (v846_data * v121_data));
          float v869_data = ir3[4];
          ir3[4] = (v869_data + (v846_data * v126_data));
          float v874_data = ir3[5];
          ir3[5] = (v874_data + (v846_data * v131_data));
          float v879_data = ir3[6];
          ir3[6] = (v879_data + (v846_data * v136_data));
          float v884_data = ir3[7];
          ir3[7] = (v884_data + (v846_data * v141_data));
          float v889_data = ir3[8];
          ir3[8] = (v889_data + (v846_data * v146_data));
          float v894_data = ir3[9];
          ir3[9] = (v894_data + (v846_data * v151_data));
          float v899_data = ir3[10];
          ir3[10] = (v899_data + (v846_data * v156_data));
          float v904_data = ir3[11];
          ir3[11] = (v904_data + (v846_data * v161_data));
          float v906_data = r2[2];
          float v909_data = ir3[0];
          ir3[0] = (v909_data + (v906_data * v166_data));
          float v914_data = ir3[1];
          ir3[1] = (v914_data + (v906_data * v171_data));
          float v919_data = ir3[2];
          ir3[2] = (v919_data + (v906_data * v176_data));
          float v924_data = ir3[3];
          ir3[3] = (v924_data + (v906_data * v181_data));
          float v929_data = ir3[4];
          ir3[4] = (v929_data + (v906_data * v186_data));
          float v934_data = ir3[5];
          ir3[5] = (v934_data + (v906_data * v191_data));
          float v939_data = ir3[6];
          ir3[6] = (v939_data + (v906_data * v196_data));
          float v944_data = ir3[7];
          ir3[7] = (v944_data + (v906_data * v201_data));
          float v949_data = ir3[8];
          ir3[8] = (v949_data + (v906_data * v206_data));
          float v954_data = ir3[9];
          ir3[9] = (v954_data + (v906_data * v211_data));
          float v959_data = ir3[10];
          ir3[10] = (v959_data + (v906_data * v216_data));
          float v964_data = ir3[11];
          ir3[11] = (v964_data + (v906_data * v221_data));
          float v966_data = r2[3];
          float v969_data = ir3[0];
          ir3[0] = (v969_data + (v966_data * v226_data));
          float v974_data = ir3[1];
          ir3[1] = (v974_data + (v966_data * v231_data));
          float v979_data = ir3[2];
          ir3[2] = (v979_data + (v966_data * v236_data));
          float v984_data = ir3[3];
          ir3[3] = (v984_data + (v966_data * v241_data));
          float v989_data = ir3[4];
          ir3[4] = (v989_data + (v966_data * v246_data));
          float v994_data = ir3[5];
          ir3[5] = (v994_data + (v966_data * v251_data));
          float v999_data = ir3[6];
          ir3[6] = (v999_data + (v966_data * v256_data));
          float v1004_data = ir3[7];
          ir3[7] = (v1004_data + (v966_data * v261_data));
          float v1009_data = ir3[8];
          ir3[8] = (v1009_data + (v966_data * v266_data));
          float v1014_data = ir3[9];
          ir3[9] = (v1014_data + (v966_data * v271_data));
          float v1019_data = ir3[10];
          ir3[10] = (v1019_data + (v966_data * v276_data));
          float v1024_data = ir3[11];
          ir3[11] = (v1024_data + (v966_data * v281_data));
          float v1026_data = r2[4];
          float v1029_data = ir3[0];
          ir3[0] = (v1029_data + (v1026_data * v286_data));
          float v1034_data = ir3[1];
          ir3[1] = (v1034_data + (v1026_data * v291_data));
          float v1039_data = ir3[2];
          ir3[2] = (v1039_data + (v1026_data * v296_data));
          float v1044_data = ir3[3];
          ir3[3] = (v1044_data + (v1026_data * v301_data));
          float v1049_data = ir3[4];
          ir3[4] = (v1049_data + (v1026_data * v306_data));
          float v1054_data = ir3[5];
          ir3[5] = (v1054_data + (v1026_data * v311_data));
          float v1059_data = ir3[6];
          ir3[6] = (v1059_data + (v1026_data * v316_data));
          float v1064_data = ir3[7];
          ir3[7] = (v1064_data + (v1026_data * v321_data));
          float v1069_data = ir3[8];
          ir3[8] = (v1069_data + (v1026_data * v326_data));
          float v1074_data = ir3[9];
          ir3[9] = (v1074_data + (v1026_data * v331_data));
          float v1079_data = ir3[10];
          ir3[10] = (v1079_data + (v1026_data * v336_data));
          float v1084_data = ir3[11];
          ir3[11] = (v1084_data + (v1026_data * v341_data));
          float v1086_data = r2[5];
          float v1089_data = ir3[0];
          ir3[0] = (v1089_data + (v1086_data * v346_data));
          float v1094_data = ir3[1];
          ir3[1] = (v1094_data + (v1086_data * v351_data));
          float v1099_data = ir3[2];
          ir3[2] = (v1099_data + (v1086_data * v356_data));
          float v1104_data = ir3[3];
          ir3[3] = (v1104_data + (v1086_data * v361_data));
          float v1109_data = ir3[4];
          ir3[4] = (v1109_data + (v1086_data * v366_data));
          float v1114_data = ir3[5];
          ir3[5] = (v1114_data + (v1086_data * v371_data));
          float v1119_data = ir3[6];
          ir3[6] = (v1119_data + (v1086_data * v376_data));
          float v1124_data = ir3[7];
          ir3[7] = (v1124_data + (v1086_data * v381_data));
          float v1129_data = ir3[8];
          ir3[8] = (v1129_data + (v1086_data * v386_data));
          float v1134_data = ir3[9];
          ir3[9] = (v1134_data + (v1086_data * v391_data));
          float v1139_data = ir3[10];
          ir3[10] = (v1139_data + (v1086_data * v396_data));
          float v1144_data = ir3[11];
          ir3[11] = (v1144_data + (v1086_data * v401_data));
          float v1146_data = r2[6];
          float v1149_data = ir3[0];
          ir3[0] = (v1149_data + (v1146_data * v406_data));
          float v1154_data = ir3[1];
          ir3[1] = (v1154_data + (v1146_data * v411_data));
          float v1159_data = ir3[2];
          ir3[2] = (v1159_data + (v1146_data * v416_data));
          float v1164_data = ir3[3];
          ir3[3] = (v1164_data + (v1146_data * v421_data));
          float v1169_data = ir3[4];
          ir3[4] = (v1169_data + (v1146_data * v426_data));
          float v1174_data = ir3[5];
          ir3[5] = (v1174_data + (v1146_data * v431_data));
          float v1179_data = ir3[6];
          ir3[6] = (v1179_data + (v1146_data * v436_data));
          float v1184_data = ir3[7];
          ir3[7] = (v1184_data + (v1146_data * v441_data));
          float v1189_data = ir3[8];
          ir3[8] = (v1189_data + (v1146_data * v446_data));
          float v1194_data = ir3[9];
          ir3[9] = (v1194_data + (v1146_data * v451_data));
          float v1199_data = ir3[10];
          ir3[10] = (v1199_data + (v1146_data * v456_data));
          float v1204_data = ir3[11];
          ir3[11] = (v1204_data + (v1146_data * v461_data));
          float v1206_data = r2[7];
          float v1209_data = ir3[0];
          ir3[0] = (v1209_data + (v1206_data * v466_data));
          float v1214_data = ir3[1];
          ir3[1] = (v1214_data + (v1206_data * v471_data));
          float v1219_data = ir3[2];
          ir3[2] = (v1219_data + (v1206_data * v476_data));
          float v1224_data = ir3[3];
          ir3[3] = (v1224_data + (v1206_data * v481_data));
          float v1229_data = ir3[4];
          ir3[4] = (v1229_data + (v1206_data * v486_data));
          float v1234_data = ir3[5];
          ir3[5] = (v1234_data + (v1206_data * v491_data));
          float v1239_data = ir3[6];
          ir3[6] = (v1239_data + (v1206_data * v496_data));
          float v1244_data = ir3[7];
          ir3[7] = (v1244_data + (v1206_data * v501_data));
          float v1249_data = ir3[8];
          ir3[8] = (v1249_data + (v1206_data * v506_data));
          float v1254_data = ir3[9];
          ir3[9] = (v1254_data + (v1206_data * v511_data));
          float v1259_data = ir3[10];
          ir3[10] = (v1259_data + (v1206_data * v516_data));
          float v1264_data = ir3[11];
          ir3[11] = (v1264_data + (v1206_data * v521_data));
          float v1266_data = r2[8];
          float v1269_data = ir3[0];
          ir3[0] = (v1269_data + (v1266_data * v526_data));
          float v1274_data = ir3[1];
          ir3[1] = (v1274_data + (v1266_data * v531_data));
          float v1279_data = ir3[2];
          ir3[2] = (v1279_data + (v1266_data * v536_data));
          float v1284_data = ir3[3];
          ir3[3] = (v1284_data + (v1266_data * v541_data));
          float v1289_data = ir3[4];
          ir3[4] = (v1289_data + (v1266_data * v546_data));
          float v1294_data = ir3[5];
          ir3[5] = (v1294_data + (v1266_data * v551_data));
          float v1299_data = ir3[6];
          ir3[6] = (v1299_data + (v1266_data * v556_data));
          float v1304_data = ir3[7];
          ir3[7] = (v1304_data + (v1266_data * v561_data));
          float v1309_data = ir3[8];
          ir3[8] = (v1309_data + (v1266_data * v566_data));
          float v1314_data = ir3[9];
          ir3[9] = (v1314_data + (v1266_data * v571_data));
          float v1319_data = ir3[10];
          ir3[10] = (v1319_data + (v1266_data * v576_data));
          float v1324_data = ir3[11];
          ir3[11] = (v1324_data + (v1266_data * v581_data));
          float v1326_data = r2[9];
          float v1329_data = ir3[0];
          ir3[0] = (v1329_data + (v1326_data * v586_data));
          float v1334_data = ir3[1];
          ir3[1] = (v1334_data + (v1326_data * v591_data));
          float v1339_data = ir3[2];
          ir3[2] = (v1339_data + (v1326_data * v596_data));
          float v1344_data = ir3[3];
          ir3[3] = (v1344_data + (v1326_data * v601_data));
          float v1349_data = ir3[4];
          ir3[4] = (v1349_data + (v1326_data * v606_data));
          float v1354_data = ir3[5];
          ir3[5] = (v1354_data + (v1326_data * v611_data));
          float v1359_data = ir3[6];
          ir3[6] = (v1359_data + (v1326_data * v616_data));
          float v1364_data = ir3[7];
          ir3[7] = (v1364_data + (v1326_data * v621_data));
          float v1369_data = ir3[8];
          ir3[8] = (v1369_data + (v1326_data * v626_data));
          float v1374_data = ir3[9];
          ir3[9] = (v1374_data + (v1326_data * v631_data));
          float v1379_data = ir3[10];
          ir3[10] = (v1379_data + (v1326_data * v636_data));
          float v1384_data = ir3[11];
          ir3[11] = (v1384_data + (v1326_data * v641_data));
          float v1386_data = r2[10];
          float v1389_data = ir3[0];
          ir3[0] = (v1389_data + (v1386_data * v646_data));
          float v1394_data = ir3[1];
          ir3[1] = (v1394_data + (v1386_data * v651_data));
          float v1399_data = ir3[2];
          ir3[2] = (v1399_data + (v1386_data * v656_data));
          float v1404_data = ir3[3];
          ir3[3] = (v1404_data + (v1386_data * v661_data));
          float v1409_data = ir3[4];
          ir3[4] = (v1409_data + (v1386_data * v666_data));
          float v1414_data = ir3[5];
          ir3[5] = (v1414_data + (v1386_data * v671_data));
          float v1419_data = ir3[6];
          ir3[6] = (v1419_data + (v1386_data * v676_data));
          float v1424_data = ir3[7];
          ir3[7] = (v1424_data + (v1386_data * v681_data));
          float v1429_data = ir3[8];
          ir3[8] = (v1429_data + (v1386_data * v686_data));
          float v1434_data = ir3[9];
          ir3[9] = (v1434_data + (v1386_data * v691_data));
          float v1439_data = ir3[10];
          ir3[10] = (v1439_data + (v1386_data * v696_data));
          float v1444_data = ir3[11];
          ir3[11] = (v1444_data + (v1386_data * v701_data));
          float v1446_data = r2[11];
          float v1449_data = ir3[0];
          ir3[0] = (v1449_data + (v1446_data * v706_data));
          float v1454_data = ir3[1];
          ir3[1] = (v1454_data + (v1446_data * v711_data));
          float v1459_data = ir3[2];
          ir3[2] = (v1459_data + (v1446_data * v716_data));
          float v1464_data = ir3[3];
          ir3[3] = (v1464_data + (v1446_data * v721_data));
          float v1469_data = ir3[4];
          ir3[4] = (v1469_data + (v1446_data * v726_data));
          float v1474_data = ir3[5];
          ir3[5] = (v1474_data + (v1446_data * v731_data));
          float v1479_data = ir3[6];
          ir3[6] = (v1479_data + (v1446_data * v736_data));
          float v1484_data = ir3[7];
          ir3[7] = (v1484_data + (v1446_data * v741_data));
          float v1489_data = ir3[8];
          ir3[8] = (v1489_data + (v1446_data * v746_data));
          float v1494_data = ir3[9];
          ir3[9] = (v1494_data + (v1446_data * v751_data));
          float v1499_data = ir3[10];
          ir3[10] = (v1499_data + (v1446_data * v756_data));
          float v1504_data = ir3[11];
          ir3[11] = (v1504_data + (v1446_data * v761_data));
          // r3 = ir3
          if (v27_g) {
            #pragma unroll
            for (int32_t v1506_n1 = 0; v1506_n1 < 12; ++v1506_n1) {
              float v1508_data = ir3[v1506_n1];
              r3[v1506_n1] = v1508_data;
            }
          }
          // s1 = store{r>s}(localShrMem0, r3);
          if (v27_g) {
            int32_t v1514_off = v26_lead + 6;
            #pragma unroll
            for (int32_t v1509_i1 = 0; v1509_i1 < 12; ++v1509_i1) {
              float v1511_data = r3[v1509_i1];
              int32_t v1516_a = v1514_off + (v1509_i1 * 12);
              s1[(v1516_a ^ ((v1516_a >> 4) & 15))] = v1511_data;
            }
          }
          float r4[12]{};
          // ir4 = +(s1)
          // [(0, 12), (0, 12)] []
          float ir4[12]{};
          bool v1525_g = v26_lead < 12;
          int32_t v1527_sw = (v26_lead >> 4) & 15;
          int32_t v1528_sw = v26_lead ^ v1527_sw;
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v1529_data = v1525_g ? (s1[v1528_sw]) : (0.0f);
          float v1530_data = ir4[0];
          ir4[0] = (v1530_data + v1529_data);
          int32_t v1532_a = v26_lead + 12;
          int32_t v1533_sw = v1532_a >> 4;
          float v1536_data = v1525_g ? (s1[(v1532_a ^ (v1533_sw & 15))]) : (0.0f);
          float v1537_data = ir4[1];
          ir4[1] = (v1537_data + v1536_data);
          int32_t v1539_a = v26_lead + 24;
          int32_t v1540_sw = v1539_a >> 4;
          float v1543_data = v1525_g ? (s1[(v1539_a ^ (v1540_sw & 15))]) : (0.0f);
          float v1544_data = ir4[2];
          ir4[2] = (v1544_data + v1543_data);
          int32_t v1546_a = v26_lead + 36;
          int32_t v1547_sw = v1546_a >> 4;
          float v1550_data = v1525_g ? (s1[(v1546_a ^ (v1547_sw & 15))]) : (0.0f);
          float v1551_data = ir4[3];
          ir4[3] = (v1551_data + v1550_data);
          int32_t v1553_a = v26_lead + 48;
          int32_t v1554_sw = v1553_a >> 4;
          float v1557_data = v1525_g ? (s1[(v1553_a ^ (v1554_sw & 15))]) : (0.0f);
          float v1558_data = ir4[4];
          ir4[4] = (v1558_data + v1557_data);
          int32_t v1560_a = v26_lead + 60;
          int32_t v1561_sw = v1560_a >> 4;
          float v1564_data = v1525_g ? (s1[(v1560_a ^ (v1561_sw & 15))]) : (0.0f);
          float v1565_data = ir4[5];
          ir4[5] = (v1565_data + v1564_data);
          int32_t v1567_a = v26_lead + 72;
          int32_t v1568_sw = v1567_a >> 4;
          float v1571_data = v1525_g ? (s1[(v1567_a ^ (v1568_sw & 15))]) : (0.0f);
          float v1572_data = ir4[6];
          ir4[6] = (v1572_data + v1571_data);
          int32_t v1574_a = v26_lead + 84;
          int32_t v1575_sw = v1574_a >> 4;
          float v1578_data = v1525_g ? (s1[(v1574_a ^ (v1575_sw & 15))]) : (0.0f);
          float v1579_data = ir4[7];
          ir4[7] = (v1579_data + v1578_data);
          int32_t v1581_a = v26_lead + 96;
          int32_t v1582_sw = v1581_a >> 4;
          float v1585_data = v1525_g ? (s1[(v1581_a ^ (v1582_sw & 15))]) : (0.0f);
          float v1586_data = ir4[8];
          ir4[8] = (v1586_data + v1585_data);
          int32_t v1588_a = v26_lead + 108;
          int32_t v1589_sw = v1588_a >> 4;
          float v1592_data = v1525_g ? (s1[(v1588_a ^ (v1589_sw & 15))]) : (0.0f);
          float v1593_data = ir4[9];
          ir4[9] = (v1593_data + v1592_data);
          int32_t v1595_a = v26_lead + 120;
          int32_t v1596_sw = v1595_a >> 4;
          float v1599_data = v1525_g ? (s1[(v1595_a ^ (v1596_sw & 15))]) : (0.0f);
          float v1600_data = ir4[10];
          ir4[10] = (v1600_data + v1599_data);
          int32_t v1602_a = v26_lead + 132;
          int32_t v1603_sw = v1602_a >> 4;
          float v1606_data = v1525_g ? (s1[(v1602_a ^ (v1603_sw & 15))]) : (0.0f);
          float v1607_data = ir4[11];
          ir4[11] = (v1607_data + v1606_data);
          // r4 = ir4
          if (v1525_g) {
            #pragma unroll
            for (int32_t v1610_n1 = 0; v1610_n1 < 12; ++v1610_n1) {
              float v1612_data = ir4[v1610_n1];
              r4[v1610_n1] = v1612_data;
            }
          }
          // glb_m3 = store{r>g}(r4);
          if (v1525_g) {
            #pragma unroll
            for (int32_t v1614_i1 = 0; v1614_i1 < 12; ++v1614_i1) {
              float v1616_data = r4[v1614_i1];
              glb_m3[(v26_lead + (v1614_i1 * 12))] = v1616_data;
            }
          }
          // wait(r5 = load{g>r}(glb_m4););
          float r6[12]{};
          // ir6 = +(r5 * s0)
          // [(0, 2), (0, 12)] [(0, 12)]
          float ir6[12]{};
          float v1623_data = r5[0];
          float v1626_data = ir6[0];
          ir6[0] = (v1626_data + (v1623_data * v46_data));
          float v1631_data = ir6[1];
          ir6[1] = (v1631_data + (v1623_data * v51_data));
          float v1636_data = ir6[2];
          ir6[2] = (v1636_data + (v1623_data * v56_data));
          float v1641_data = ir6[3];
          ir6[3] = (v1641_data + (v1623_data * v61_data));
          float v1646_data = ir6[4];
          ir6[4] = (v1646_data + (v1623_data * v66_data));
          float v1651_data = ir6[5];
          ir6[5] = (v1651_data + (v1623_data * v71_data));
          float v1656_data = ir6[6];
          ir6[6] = (v1656_data + (v1623_data * v76_data));
          float v1661_data = ir6[7];
          ir6[7] = (v1661_data + (v1623_data * v81_data));
          float v1666_data = ir6[8];
          ir6[8] = (v1666_data + (v1623_data * v86_data));
          float v1671_data = ir6[9];
          ir6[9] = (v1671_data + (v1623_data * v91_data));
          float v1676_data = ir6[10];
          ir6[10] = (v1676_data + (v1623_data * v96_data));
          float v1681_data = ir6[11];
          ir6[11] = (v1681_data + (v1623_data * v101_data));
          float v1683_data = r5[1];
          float v1686_data = ir6[0];
          ir6[0] = (v1686_data + (v1683_data * v106_data));
          float v1691_data = ir6[1];
          ir6[1] = (v1691_data + (v1683_data * v111_data));
          float v1696_data = ir6[2];
          ir6[2] = (v1696_data + (v1683_data * v116_data));
          float v1701_data = ir6[3];
          ir6[3] = (v1701_data + (v1683_data * v121_data));
          float v1706_data = ir6[4];
          ir6[4] = (v1706_data + (v1683_data * v126_data));
          float v1711_data = ir6[5];
          ir6[5] = (v1711_data + (v1683_data * v131_data));
          float v1716_data = ir6[6];
          ir6[6] = (v1716_data + (v1683_data * v136_data));
          float v1721_data = ir6[7];
          ir6[7] = (v1721_data + (v1683_data * v141_data));
          float v1726_data = ir6[8];
          ir6[8] = (v1726_data + (v1683_data * v146_data));
          float v1731_data = ir6[9];
          ir6[9] = (v1731_data + (v1683_data * v151_data));
          float v1736_data = ir6[10];
          ir6[10] = (v1736_data + (v1683_data * v156_data));
          float v1741_data = ir6[11];
          ir6[11] = (v1741_data + (v1683_data * v161_data));
          float v1743_data = r5[2];
          float v1746_data = ir6[0];
          ir6[0] = (v1746_data + (v1743_data * v166_data));
          float v1751_data = ir6[1];
          ir6[1] = (v1751_data + (v1743_data * v171_data));
          float v1756_data = ir6[2];
          ir6[2] = (v1756_data + (v1743_data * v176_data));
          float v1761_data = ir6[3];
          ir6[3] = (v1761_data + (v1743_data * v181_data));
          float v1766_data = ir6[4];
          ir6[4] = (v1766_data + (v1743_data * v186_data));
          float v1771_data = ir6[5];
          ir6[5] = (v1771_data + (v1743_data * v191_data));
          float v1776_data = ir6[6];
          ir6[6] = (v1776_data + (v1743_data * v196_data));
          float v1781_data = ir6[7];
          ir6[7] = (v1781_data + (v1743_data * v201_data));
          float v1786_data = ir6[8];
          ir6[8] = (v1786_data + (v1743_data * v206_data));
          float v1791_data = ir6[9];
          ir6[9] = (v1791_data + (v1743_data * v211_data));
          float v1796_data = ir6[10];
          ir6[10] = (v1796_data + (v1743_data * v216_data));
          float v1801_data = ir6[11];
          ir6[11] = (v1801_data + (v1743_data * v221_data));
          float v1803_data = r5[3];
          float v1806_data = ir6[0];
          ir6[0] = (v1806_data + (v1803_data * v226_data));
          float v1811_data = ir6[1];
          ir6[1] = (v1811_data + (v1803_data * v231_data));
          float v1816_data = ir6[2];
          ir6[2] = (v1816_data + (v1803_data * v236_data));
          float v1821_data = ir6[3];
          ir6[3] = (v1821_data + (v1803_data * v241_data));
          float v1826_data = ir6[4];
          ir6[4] = (v1826_data + (v1803_data * v246_data));
          float v1831_data = ir6[5];
          ir6[5] = (v1831_data + (v1803_data * v251_data));
          float v1836_data = ir6[6];
          ir6[6] = (v1836_data + (v1803_data * v256_data));
          float v1841_data = ir6[7];
          ir6[7] = (v1841_data + (v1803_data * v261_data));
          float v1846_data = ir6[8];
          ir6[8] = (v1846_data + (v1803_data * v266_data));
          float v1851_data = ir6[9];
          ir6[9] = (v1851_data + (v1803_data * v271_data));
          float v1856_data = ir6[10];
          ir6[10] = (v1856_data + (v1803_data * v276_data));
          float v1861_data = ir6[11];
          ir6[11] = (v1861_data + (v1803_data * v281_data));
          float v1863_data = r5[4];
          float v1866_data = ir6[0];
          ir6[0] = (v1866_data + (v1863_data * v286_data));
          float v1871_data = ir6[1];
          ir6[1] = (v1871_data + (v1863_data * v291_data));
          float v1876_data = ir6[2];
          ir6[2] = (v1876_data + (v1863_data * v296_data));
          float v1881_data = ir6[3];
          ir6[3] = (v1881_data + (v1863_data * v301_data));
          float v1886_data = ir6[4];
          ir6[4] = (v1886_data + (v1863_data * v306_data));
          float v1891_data = ir6[5];
          ir6[5] = (v1891_data + (v1863_data * v311_data));
          float v1896_data = ir6[6];
          ir6[6] = (v1896_data + (v1863_data * v316_data));
          float v1901_data = ir6[7];
          ir6[7] = (v1901_data + (v1863_data * v321_data));
          float v1906_data = ir6[8];
          ir6[8] = (v1906_data + (v1863_data * v326_data));
          float v1911_data = ir6[9];
          ir6[9] = (v1911_data + (v1863_data * v331_data));
          float v1916_data = ir6[10];
          ir6[10] = (v1916_data + (v1863_data * v336_data));
          float v1921_data = ir6[11];
          ir6[11] = (v1921_data + (v1863_data * v341_data));
          float v1923_data = r5[5];
          float v1926_data = ir6[0];
          ir6[0] = (v1926_data + (v1923_data * v346_data));
          float v1931_data = ir6[1];
          ir6[1] = (v1931_data + (v1923_data * v351_data));
          float v1936_data = ir6[2];
          ir6[2] = (v1936_data + (v1923_data * v356_data));
          float v1941_data = ir6[3];
          ir6[3] = (v1941_data + (v1923_data * v361_data));
          float v1946_data = ir6[4];
          ir6[4] = (v1946_data + (v1923_data * v366_data));
          float v1951_data = ir6[5];
          ir6[5] = (v1951_data + (v1923_data * v371_data));
          float v1956_data = ir6[6];
          ir6[6] = (v1956_data + (v1923_data * v376_data));
          float v1961_data = ir6[7];
          ir6[7] = (v1961_data + (v1923_data * v381_data));
          float v1966_data = ir6[8];
          ir6[8] = (v1966_data + (v1923_data * v386_data));
          float v1971_data = ir6[9];
          ir6[9] = (v1971_data + (v1923_data * v391_data));
          float v1976_data = ir6[10];
          ir6[10] = (v1976_data + (v1923_data * v396_data));
          float v1981_data = ir6[11];
          ir6[11] = (v1981_data + (v1923_data * v401_data));
          float v1983_data = r5[6];
          float v1986_data = ir6[0];
          ir6[0] = (v1986_data + (v1983_data * v406_data));
          float v1991_data = ir6[1];
          ir6[1] = (v1991_data + (v1983_data * v411_data));
          float v1996_data = ir6[2];
          ir6[2] = (v1996_data + (v1983_data * v416_data));
          float v2001_data = ir6[3];
          ir6[3] = (v2001_data + (v1983_data * v421_data));
          float v2006_data = ir6[4];
          ir6[4] = (v2006_data + (v1983_data * v426_data));
          float v2011_data = ir6[5];
          ir6[5] = (v2011_data + (v1983_data * v431_data));
          float v2016_data = ir6[6];
          ir6[6] = (v2016_data + (v1983_data * v436_data));
          float v2021_data = ir6[7];
          ir6[7] = (v2021_data + (v1983_data * v441_data));
          float v2026_data = ir6[8];
          ir6[8] = (v2026_data + (v1983_data * v446_data));
          float v2031_data = ir6[9];
          ir6[9] = (v2031_data + (v1983_data * v451_data));
          float v2036_data = ir6[10];
          ir6[10] = (v2036_data + (v1983_data * v456_data));
          float v2041_data = ir6[11];
          ir6[11] = (v2041_data + (v1983_data * v461_data));
          float v2043_data = r5[7];
          float v2046_data = ir6[0];
          ir6[0] = (v2046_data + (v2043_data * v466_data));
          float v2051_data = ir6[1];
          ir6[1] = (v2051_data + (v2043_data * v471_data));
          float v2056_data = ir6[2];
          ir6[2] = (v2056_data + (v2043_data * v476_data));
          float v2061_data = ir6[3];
          ir6[3] = (v2061_data + (v2043_data * v481_data));
          float v2066_data = ir6[4];
          ir6[4] = (v2066_data + (v2043_data * v486_data));
          float v2071_data = ir6[5];
          ir6[5] = (v2071_data + (v2043_data * v491_data));
          float v2076_data = ir6[6];
          ir6[6] = (v2076_data + (v2043_data * v496_data));
          float v2081_data = ir6[7];
          ir6[7] = (v2081_data + (v2043_data * v501_data));
          float v2086_data = ir6[8];
          ir6[8] = (v2086_data + (v2043_data * v506_data));
          float v2091_data = ir6[9];
          ir6[9] = (v2091_data + (v2043_data * v511_data));
          float v2096_data = ir6[10];
          ir6[10] = (v2096_data + (v2043_data * v516_data));
          float v2101_data = ir6[11];
          ir6[11] = (v2101_data + (v2043_data * v521_data));
          float v2103_data = r5[8];
          float v2106_data = ir6[0];
          ir6[0] = (v2106_data + (v2103_data * v526_data));
          float v2111_data = ir6[1];
          ir6[1] = (v2111_data + (v2103_data * v531_data));
          float v2116_data = ir6[2];
          ir6[2] = (v2116_data + (v2103_data * v536_data));
          float v2121_data = ir6[3];
          ir6[3] = (v2121_data + (v2103_data * v541_data));
          float v2126_data = ir6[4];
          ir6[4] = (v2126_data + (v2103_data * v546_data));
          float v2131_data = ir6[5];
          ir6[5] = (v2131_data + (v2103_data * v551_data));
          float v2136_data = ir6[6];
          ir6[6] = (v2136_data + (v2103_data * v556_data));
          float v2141_data = ir6[7];
          ir6[7] = (v2141_data + (v2103_data * v561_data));
          float v2146_data = ir6[8];
          ir6[8] = (v2146_data + (v2103_data * v566_data));
          float v2151_data = ir6[9];
          ir6[9] = (v2151_data + (v2103_data * v571_data));
          float v2156_data = ir6[10];
          ir6[10] = (v2156_data + (v2103_data * v576_data));
          float v2161_data = ir6[11];
          ir6[11] = (v2161_data + (v2103_data * v581_data));
          float v2163_data = r5[9];
          float v2166_data = ir6[0];
          ir6[0] = (v2166_data + (v2163_data * v586_data));
          float v2171_data = ir6[1];
          ir6[1] = (v2171_data + (v2163_data * v591_data));
          float v2176_data = ir6[2];
          ir6[2] = (v2176_data + (v2163_data * v596_data));
          float v2181_data = ir6[3];
          ir6[3] = (v2181_data + (v2163_data * v601_data));
          float v2186_data = ir6[4];
          ir6[4] = (v2186_data + (v2163_data * v606_data));
          float v2191_data = ir6[5];
          ir6[5] = (v2191_data + (v2163_data * v611_data));
          float v2196_data = ir6[6];
          ir6[6] = (v2196_data + (v2163_data * v616_data));
          float v2201_data = ir6[7];
          ir6[7] = (v2201_data + (v2163_data * v621_data));
          float v2206_data = ir6[8];
          ir6[8] = (v2206_data + (v2163_data * v626_data));
          float v2211_data = ir6[9];
          ir6[9] = (v2211_data + (v2163_data * v631_data));
          float v2216_data = ir6[10];
          ir6[10] = (v2216_data + (v2163_data * v636_data));
          float v2221_data = ir6[11];
          ir6[11] = (v2221_data + (v2163_data * v641_data));
          float v2223_data = r5[10];
          float v2226_data = ir6[0];
          ir6[0] = (v2226_data + (v2223_data * v646_data));
          float v2231_data = ir6[1];
          ir6[1] = (v2231_data + (v2223_data * v651_data));
          float v2236_data = ir6[2];
          ir6[2] = (v2236_data + (v2223_data * v656_data));
          float v2241_data = ir6[3];
          ir6[3] = (v2241_data + (v2223_data * v661_data));
          float v2246_data = ir6[4];
          ir6[4] = (v2246_data + (v2223_data * v666_data));
          float v2251_data = ir6[5];
          ir6[5] = (v2251_data + (v2223_data * v671_data));
          float v2256_data = ir6[6];
          ir6[6] = (v2256_data + (v2223_data * v676_data));
          float v2261_data = ir6[7];
          ir6[7] = (v2261_data + (v2223_data * v681_data));
          float v2266_data = ir6[8];
          ir6[8] = (v2266_data + (v2223_data * v686_data));
          float v2271_data = ir6[9];
          ir6[9] = (v2271_data + (v2223_data * v691_data));
          float v2276_data = ir6[10];
          ir6[10] = (v2276_data + (v2223_data * v696_data));
          float v2281_data = ir6[11];
          ir6[11] = (v2281_data + (v2223_data * v701_data));
          float v2283_data = r5[11];
          float v2286_data = ir6[0];
          ir6[0] = (v2286_data + (v2283_data * v706_data));
          float v2291_data = ir6[1];
          ir6[1] = (v2291_data + (v2283_data * v711_data));
          float v2296_data = ir6[2];
          ir6[2] = (v2296_data + (v2283_data * v716_data));
          float v2301_data = ir6[3];
          ir6[3] = (v2301_data + (v2283_data * v721_data));
          float v2306_data = ir6[4];
          ir6[4] = (v2306_data + (v2283_data * v726_data));
          float v2311_data = ir6[5];
          ir6[5] = (v2311_data + (v2283_data * v731_data));
          float v2316_data = ir6[6];
          ir6[6] = (v2316_data + (v2283_data * v736_data));
          float v2321_data = ir6[7];
          ir6[7] = (v2321_data + (v2283_data * v741_data));
          float v2326_data = ir6[8];
          ir6[8] = (v2326_data + (v2283_data * v746_data));
          float v2331_data = ir6[9];
          ir6[9] = (v2331_data + (v2283_data * v751_data));
          float v2336_data = ir6[10];
          ir6[10] = (v2336_data + (v2283_data * v756_data));
          float v2341_data = ir6[11];
          ir6[11] = (v2341_data + (v2283_data * v761_data));
          // r6 = ir6
          if (v776_g) {
            #pragma unroll
            for (int32_t v2343_n1 = 0; v2343_n1 < 12; ++v2343_n1) {
              float v2345_data = ir6[v2343_n1];
              r6[v2343_n1] = v2345_data;
            }
          }
          // s1 = store{r>s, clear}(localShrMem0, r6);
          bool v2348_g = (v26_lead >= 8) && v1525_g;
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          if (v2348_g) {
            #pragma unroll
            for (int32_t v2349_z1 = 0; v2349_z1 < 12; ++v2349_z1) {
              int32_t v2354_a = v26_lead + (v2349_z1 * 12);
              s1[(v2354_a ^ ((v2354_a >> 4) & 15))] = 0.0f;
            }
          }
          if (v776_g) {
            int32_t v2363_off = v26_lead + 6;
            #pragma unroll
            for (int32_t v2358_i1 = 0; v2358_i1 < 12; ++v2358_i1) {
              float v2360_data = r6[v2358_i1];
              int32_t v2365_a = v2363_off + (v2358_i1 * 12);
              s1[(v2365_a ^ ((v2365_a >> 4) & 15))] = v2360_data;
            }
          }
          float r7[12]{};
          // ir7 = +(s1)
          // [(0, 12), (0, 12)] []
          float ir7[12]{};
          int32_t v2376_sw = v26_lead ^ v1527_sw;
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v2377_data = v1525_g ? (s1[v2376_sw]) : (0.0f);
          float v2378_data = ir7[0];
          ir7[0] = (v2378_data + v2377_data);
          float v2384_data = v1525_g ? (s1[(v1532_a ^ (v1533_sw & 15))]) : (0.0f);
          float v2385_data = ir7[1];
          ir7[1] = (v2385_data + v2384_data);
          float v2391_data = v1525_g ? (s1[(v1539_a ^ (v1540_sw & 15))]) : (0.0f);
          float v2392_data = ir7[2];
          ir7[2] = (v2392_data + v2391_data);
          float v2398_data = v1525_g ? (s1[(v1546_a ^ (v1547_sw & 15))]) : (0.0f);
          float v2399_data = ir7[3];
          ir7[3] = (v2399_data + v2398_data);
          float v2405_data = v1525_g ? (s1[(v1553_a ^ (v1554_sw & 15))]) : (0.0f);
          float v2406_data = ir7[4];
          ir7[4] = (v2406_data + v2405_data);
          float v2412_data = v1525_g ? (s1[(v1560_a ^ (v1561_sw & 15))]) : (0.0f);
          float v2413_data = ir7[5];
          ir7[5] = (v2413_data + v2412_data);
          float v2419_data = v1525_g ? (s1[(v1567_a ^ (v1568_sw & 15))]) : (0.0f);
          float v2420_data = ir7[6];
          ir7[6] = (v2420_data + v2419_data);
          float v2426_data = v1525_g ? (s1[(v1574_a ^ (v1575_sw & 15))]) : (0.0f);
          float v2427_data = ir7[7];
          ir7[7] = (v2427_data + v2426_data);
          float v2433_data = v1525_g ? (s1[(v1581_a ^ (v1582_sw & 15))]) : (0.0f);
          float v2434_data = ir7[8];
          ir7[8] = (v2434_data + v2433_data);
          float v2440_data = v1525_g ? (s1[(v1588_a ^ (v1589_sw & 15))]) : (0.0f);
          float v2441_data = ir7[9];
          ir7[9] = (v2441_data + v2440_data);
          float v2447_data = v1525_g ? (s1[(v1595_a ^ (v1596_sw & 15))]) : (0.0f);
          float v2448_data = ir7[10];
          ir7[10] = (v2448_data + v2447_data);
          float v2454_data = v1525_g ? (s1[(v1602_a ^ (v1603_sw & 15))]) : (0.0f);
          float v2455_data = ir7[11];
          ir7[11] = (v2455_data + v2454_data);
          // r7 = ir7
          if (v1525_g) {
            #pragma unroll
            for (int32_t v2457_n1 = 0; v2457_n1 < 12; ++v2457_n1) {
              float v2459_data = ir7[v2457_n1];
              r7[v2457_n1] = v2459_data;
            }
          }
          // glb_m5 = store{r>g}(r7);
          if (v1525_g) {
            #pragma unroll
            for (int32_t v2460_i1 = 0; v2460_i1 < 12; ++v2460_i1) {
              float v2462_data = r7[v2460_i1];
              glb_m5[(v26_lead + (v2460_i1 * 12))] = v2462_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

