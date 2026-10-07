// === base name ===
kernel_771e96f3a3a63c2c

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_771e96f3a3a63c2c = {{8, 16, 1}, 8, 8, 1, 16, 4608, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_771e96f3a3a63c2c(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_771e96f3a3a63c2c(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_771e96f3a3a63c2c(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_771e96f3a3a63c2c, block.x * block.y * block.z, 1152 * sizeof(float));
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
void launcher_kernel_771e96f3a3a63c2c(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_771e96f3a3a63c2c(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_771e96f3a3a63c2c, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_771e96f3a3a63c2c<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_771e96f3a3a63c2c(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
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
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s2 = &localShrMem0[0];
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < numElements0; v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 64 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v9_batchId0 * 64 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v9_batchId0 * 64 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[v9_batchId0 * 64 + 0 + m3_extraOffset];
          float *const __restrict__ glb_m4 = &m4[v9_batchId0 * 64 + 0 + m4_extraOffset];
          float r0[8]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v25_lead = threadIdx.x % 8;
          #pragma unroll
          for (int32_t v26_i0 = 0; v26_i0 < 1; ++v26_i0) {
            int32_t v29_lead = v25_lead + (v26_i0 * 8);
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 8; ++v27_i1) {
              float v32_data = __ldcg(&glb_m0[(v29_lead + (v27_i1 * 8))]);
              r0[(v26_i0 + v27_i1)] = v32_data;
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
          for (int32_t v36_i0 = 0; v36_i0 < 1; ++v36_i0) {
            int32_t v39_lead = v25_lead + (v36_i0 * 8);
            #pragma unroll
            for (int32_t v37_i1 = 0; v37_i1 < 8; ++v37_i1) {
              float v42_data = __ldcg(&glb_m2[(v39_lead + (v37_i1 * 8))]);
              r2[(v36_i0 + v37_i1)] = v42_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          // r1 = +(r0 * s0) + None
          // [(0, 8), (0, 8)] [(0, 8)]
          float v45_data = r0[0];
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          float v46_data = s0[0];
          float v48_data = r1[0];
          r1[0] = (v48_data + (v45_data * v46_data));
          float v51_data = s0[8];
          float v53_data = r1[1];
          r1[1] = (v53_data + (v45_data * v51_data));
          float v56_data = s0[16];
          float v58_data = r1[2];
          r1[2] = (v58_data + (v45_data * v56_data));
          float v61_data = s0[24];
          float v63_data = r1[3];
          r1[3] = (v63_data + (v45_data * v61_data));
          float v66_data = s0[32];
          float v68_data = r1[4];
          r1[4] = (v68_data + (v45_data * v66_data));
          float v71_data = s0[40];
          float v73_data = r1[5];
          r1[5] = (v73_data + (v45_data * v71_data));
          float v76_data = s0[48];
          float v78_data = r1[6];
          r1[6] = (v78_data + (v45_data * v76_data));
          float v81_data = s0[56];
          float v83_data = r1[7];
          r1[7] = (v83_data + (v45_data * v81_data));
          float v85_data = r0[1];
          float v86_data = s0[1];
          float v88_data = r1[0];
          r1[0] = (v88_data + (v85_data * v86_data));
          float v91_data = s0[9];
          float v93_data = r1[1];
          r1[1] = (v93_data + (v85_data * v91_data));
          float v96_data = s0[17];
          float v98_data = r1[2];
          r1[2] = (v98_data + (v85_data * v96_data));
          float v101_data = s0[25];
          float v103_data = r1[3];
          r1[3] = (v103_data + (v85_data * v101_data));
          float v106_data = s0[33];
          float v108_data = r1[4];
          r1[4] = (v108_data + (v85_data * v106_data));
          float v111_data = s0[41];
          float v113_data = r1[5];
          r1[5] = (v113_data + (v85_data * v111_data));
          float v116_data = s0[49];
          float v118_data = r1[6];
          r1[6] = (v118_data + (v85_data * v116_data));
          float v121_data = s0[57];
          float v123_data = r1[7];
          r1[7] = (v123_data + (v85_data * v121_data));
          float v125_data = r0[2];
          float v126_data = s0[2];
          float v128_data = r1[0];
          r1[0] = (v128_data + (v125_data * v126_data));
          float v131_data = s0[10];
          float v133_data = r1[1];
          r1[1] = (v133_data + (v125_data * v131_data));
          float v136_data = s0[18];
          float v138_data = r1[2];
          r1[2] = (v138_data + (v125_data * v136_data));
          float v141_data = s0[26];
          float v143_data = r1[3];
          r1[3] = (v143_data + (v125_data * v141_data));
          float v146_data = s0[34];
          float v148_data = r1[4];
          r1[4] = (v148_data + (v125_data * v146_data));
          float v151_data = s0[42];
          float v153_data = r1[5];
          r1[5] = (v153_data + (v125_data * v151_data));
          float v156_data = s0[50];
          float v158_data = r1[6];
          r1[6] = (v158_data + (v125_data * v156_data));
          float v161_data = s0[58];
          float v163_data = r1[7];
          r1[7] = (v163_data + (v125_data * v161_data));
          float v165_data = r0[3];
          float v166_data = s0[3];
          float v168_data = r1[0];
          r1[0] = (v168_data + (v165_data * v166_data));
          float v171_data = s0[11];
          float v173_data = r1[1];
          r1[1] = (v173_data + (v165_data * v171_data));
          float v176_data = s0[19];
          float v178_data = r1[2];
          r1[2] = (v178_data + (v165_data * v176_data));
          float v181_data = s0[27];
          float v183_data = r1[3];
          r1[3] = (v183_data + (v165_data * v181_data));
          float v186_data = s0[35];
          float v188_data = r1[4];
          r1[4] = (v188_data + (v165_data * v186_data));
          float v191_data = s0[43];
          float v193_data = r1[5];
          r1[5] = (v193_data + (v165_data * v191_data));
          float v196_data = s0[51];
          float v198_data = r1[6];
          r1[6] = (v198_data + (v165_data * v196_data));
          float v201_data = s0[59];
          float v203_data = r1[7];
          r1[7] = (v203_data + (v165_data * v201_data));
          float v205_data = r0[4];
          float v206_data = s0[4];
          float v208_data = r1[0];
          r1[0] = (v208_data + (v205_data * v206_data));
          float v211_data = s0[12];
          float v213_data = r1[1];
          r1[1] = (v213_data + (v205_data * v211_data));
          float v216_data = s0[20];
          float v218_data = r1[2];
          r1[2] = (v218_data + (v205_data * v216_data));
          float v221_data = s0[28];
          float v223_data = r1[3];
          r1[3] = (v223_data + (v205_data * v221_data));
          float v226_data = s0[36];
          float v228_data = r1[4];
          r1[4] = (v228_data + (v205_data * v226_data));
          float v231_data = s0[44];
          float v233_data = r1[5];
          r1[5] = (v233_data + (v205_data * v231_data));
          float v236_data = s0[52];
          float v238_data = r1[6];
          r1[6] = (v238_data + (v205_data * v236_data));
          float v241_data = s0[60];
          float v243_data = r1[7];
          r1[7] = (v243_data + (v205_data * v241_data));
          float v245_data = r0[5];
          float v246_data = s0[5];
          float v248_data = r1[0];
          r1[0] = (v248_data + (v245_data * v246_data));
          float v251_data = s0[13];
          float v253_data = r1[1];
          r1[1] = (v253_data + (v245_data * v251_data));
          float v256_data = s0[21];
          float v258_data = r1[2];
          r1[2] = (v258_data + (v245_data * v256_data));
          float v261_data = s0[29];
          float v263_data = r1[3];
          r1[3] = (v263_data + (v245_data * v261_data));
          float v266_data = s0[37];
          float v268_data = r1[4];
          r1[4] = (v268_data + (v245_data * v266_data));
          float v271_data = s0[45];
          float v273_data = r1[5];
          r1[5] = (v273_data + (v245_data * v271_data));
          float v276_data = s0[53];
          float v278_data = r1[6];
          r1[6] = (v278_data + (v245_data * v276_data));
          float v281_data = s0[61];
          float v283_data = r1[7];
          r1[7] = (v283_data + (v245_data * v281_data));
          float v285_data = r0[6];
          float v286_data = s0[6];
          float v288_data = r1[0];
          r1[0] = (v288_data + (v285_data * v286_data));
          float v291_data = s0[14];
          float v293_data = r1[1];
          r1[1] = (v293_data + (v285_data * v291_data));
          float v296_data = s0[22];
          float v298_data = r1[2];
          r1[2] = (v298_data + (v285_data * v296_data));
          float v301_data = s0[30];
          float v303_data = r1[3];
          r1[3] = (v303_data + (v285_data * v301_data));
          float v306_data = s0[38];
          float v308_data = r1[4];
          r1[4] = (v308_data + (v285_data * v306_data));
          float v311_data = s0[46];
          float v313_data = r1[5];
          r1[5] = (v313_data + (v285_data * v311_data));
          float v316_data = s0[54];
          float v318_data = r1[6];
          r1[6] = (v318_data + (v285_data * v316_data));
          float v321_data = s0[62];
          float v323_data = r1[7];
          r1[7] = (v323_data + (v285_data * v321_data));
          float v325_data = r0[7];
          float v326_data = s0[7];
          float v328_data = r1[0];
          r1[0] = (v328_data + (v325_data * v326_data));
          float v331_data = s0[15];
          float v333_data = r1[1];
          r1[1] = (v333_data + (v325_data * v331_data));
          float v336_data = s0[23];
          float v338_data = r1[2];
          r1[2] = (v338_data + (v325_data * v336_data));
          float v341_data = s0[31];
          float v343_data = r1[3];
          r1[3] = (v343_data + (v325_data * v341_data));
          float v346_data = s0[39];
          float v348_data = r1[4];
          r1[4] = (v348_data + (v325_data * v346_data));
          float v351_data = s0[47];
          float v353_data = r1[5];
          r1[5] = (v353_data + (v325_data * v351_data));
          float v356_data = s0[55];
          float v358_data = r1[6];
          r1[6] = (v358_data + (v325_data * v356_data));
          float v361_data = s0[63];
          float v363_data = r1[7];
          r1[7] = (v363_data + (v325_data * v361_data));
          // s2 = load{g>s}(glb_m3[0, 1])
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          #pragma unroll
          for (int32_t i = 0; i < 8; i += 1) {
            __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + i * 8], &glb_m3[0 + 0 + 1 * threadIdx.x + i * 8], 4);
          }
          __pipeline_commit();
          // wait(r2 = load{g>r}(glb_m2););
          // wait(s2 = load{g>s}(glb_m3[0, 1]));
          __pipeline_wait_prior(0);
          float r3[8]{};
          // ir3 = +(r2 * s2)
          // [(0, 8), (0, 8)] [(0, 8)]
          float ir3[8]{};
          float v368_data = r2[0];
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
          float v369_data = s2[0];
          float v371_data = ir3[0];
          ir3[0] = (v371_data + (v368_data * v369_data));
          float v374_data = s2[8];
          float v376_data = ir3[1];
          ir3[1] = (v376_data + (v368_data * v374_data));
          float v379_data = s2[16];
          float v381_data = ir3[2];
          ir3[2] = (v381_data + (v368_data * v379_data));
          float v384_data = s2[24];
          float v386_data = ir3[3];
          ir3[3] = (v386_data + (v368_data * v384_data));
          float v389_data = s2[32];
          float v391_data = ir3[4];
          ir3[4] = (v391_data + (v368_data * v389_data));
          float v394_data = s2[40];
          float v396_data = ir3[5];
          ir3[5] = (v396_data + (v368_data * v394_data));
          float v399_data = s2[48];
          float v401_data = ir3[6];
          ir3[6] = (v401_data + (v368_data * v399_data));
          float v404_data = s2[56];
          float v406_data = ir3[7];
          ir3[7] = (v406_data + (v368_data * v404_data));
          float v408_data = r2[1];
          float v409_data = s2[1];
          float v411_data = ir3[0];
          ir3[0] = (v411_data + (v408_data * v409_data));
          float v414_data = s2[9];
          float v416_data = ir3[1];
          ir3[1] = (v416_data + (v408_data * v414_data));
          float v419_data = s2[17];
          float v421_data = ir3[2];
          ir3[2] = (v421_data + (v408_data * v419_data));
          float v424_data = s2[25];
          float v426_data = ir3[3];
          ir3[3] = (v426_data + (v408_data * v424_data));
          float v429_data = s2[33];
          float v431_data = ir3[4];
          ir3[4] = (v431_data + (v408_data * v429_data));
          float v434_data = s2[41];
          float v436_data = ir3[5];
          ir3[5] = (v436_data + (v408_data * v434_data));
          float v439_data = s2[49];
          float v441_data = ir3[6];
          ir3[6] = (v441_data + (v408_data * v439_data));
          float v444_data = s2[57];
          float v446_data = ir3[7];
          ir3[7] = (v446_data + (v408_data * v444_data));
          float v448_data = r2[2];
          float v449_data = s2[2];
          float v451_data = ir3[0];
          ir3[0] = (v451_data + (v448_data * v449_data));
          float v454_data = s2[10];
          float v456_data = ir3[1];
          ir3[1] = (v456_data + (v448_data * v454_data));
          float v459_data = s2[18];
          float v461_data = ir3[2];
          ir3[2] = (v461_data + (v448_data * v459_data));
          float v464_data = s2[26];
          float v466_data = ir3[3];
          ir3[3] = (v466_data + (v448_data * v464_data));
          float v469_data = s2[34];
          float v471_data = ir3[4];
          ir3[4] = (v471_data + (v448_data * v469_data));
          float v474_data = s2[42];
          float v476_data = ir3[5];
          ir3[5] = (v476_data + (v448_data * v474_data));
          float v479_data = s2[50];
          float v481_data = ir3[6];
          ir3[6] = (v481_data + (v448_data * v479_data));
          float v484_data = s2[58];
          float v486_data = ir3[7];
          ir3[7] = (v486_data + (v448_data * v484_data));
          float v488_data = r2[3];
          float v489_data = s2[3];
          float v491_data = ir3[0];
          ir3[0] = (v491_data + (v488_data * v489_data));
          float v494_data = s2[11];
          float v496_data = ir3[1];
          ir3[1] = (v496_data + (v488_data * v494_data));
          float v499_data = s2[19];
          float v501_data = ir3[2];
          ir3[2] = (v501_data + (v488_data * v499_data));
          float v504_data = s2[27];
          float v506_data = ir3[3];
          ir3[3] = (v506_data + (v488_data * v504_data));
          float v509_data = s2[35];
          float v511_data = ir3[4];
          ir3[4] = (v511_data + (v488_data * v509_data));
          float v514_data = s2[43];
          float v516_data = ir3[5];
          ir3[5] = (v516_data + (v488_data * v514_data));
          float v519_data = s2[51];
          float v521_data = ir3[6];
          ir3[6] = (v521_data + (v488_data * v519_data));
          float v524_data = s2[59];
          float v526_data = ir3[7];
          ir3[7] = (v526_data + (v488_data * v524_data));
          float v528_data = r2[4];
          float v529_data = s2[4];
          float v531_data = ir3[0];
          ir3[0] = (v531_data + (v528_data * v529_data));
          float v534_data = s2[12];
          float v536_data = ir3[1];
          ir3[1] = (v536_data + (v528_data * v534_data));
          float v539_data = s2[20];
          float v541_data = ir3[2];
          ir3[2] = (v541_data + (v528_data * v539_data));
          float v544_data = s2[28];
          float v546_data = ir3[3];
          ir3[3] = (v546_data + (v528_data * v544_data));
          float v549_data = s2[36];
          float v551_data = ir3[4];
          ir3[4] = (v551_data + (v528_data * v549_data));
          float v554_data = s2[44];
          float v556_data = ir3[5];
          ir3[5] = (v556_data + (v528_data * v554_data));
          float v559_data = s2[52];
          float v561_data = ir3[6];
          ir3[6] = (v561_data + (v528_data * v559_data));
          float v564_data = s2[60];
          float v566_data = ir3[7];
          ir3[7] = (v566_data + (v528_data * v564_data));
          float v568_data = r2[5];
          float v569_data = s2[5];
          float v571_data = ir3[0];
          ir3[0] = (v571_data + (v568_data * v569_data));
          float v574_data = s2[13];
          float v576_data = ir3[1];
          ir3[1] = (v576_data + (v568_data * v574_data));
          float v579_data = s2[21];
          float v581_data = ir3[2];
          ir3[2] = (v581_data + (v568_data * v579_data));
          float v584_data = s2[29];
          float v586_data = ir3[3];
          ir3[3] = (v586_data + (v568_data * v584_data));
          float v589_data = s2[37];
          float v591_data = ir3[4];
          ir3[4] = (v591_data + (v568_data * v589_data));
          float v594_data = s2[45];
          float v596_data = ir3[5];
          ir3[5] = (v596_data + (v568_data * v594_data));
          float v599_data = s2[53];
          float v601_data = ir3[6];
          ir3[6] = (v601_data + (v568_data * v599_data));
          float v604_data = s2[61];
          float v606_data = ir3[7];
          ir3[7] = (v606_data + (v568_data * v604_data));
          float v608_data = r2[6];
          float v609_data = s2[6];
          float v611_data = ir3[0];
          ir3[0] = (v611_data + (v608_data * v609_data));
          float v614_data = s2[14];
          float v616_data = ir3[1];
          ir3[1] = (v616_data + (v608_data * v614_data));
          float v619_data = s2[22];
          float v621_data = ir3[2];
          ir3[2] = (v621_data + (v608_data * v619_data));
          float v624_data = s2[30];
          float v626_data = ir3[3];
          ir3[3] = (v626_data + (v608_data * v624_data));
          float v629_data = s2[38];
          float v631_data = ir3[4];
          ir3[4] = (v631_data + (v608_data * v629_data));
          float v634_data = s2[46];
          float v636_data = ir3[5];
          ir3[5] = (v636_data + (v608_data * v634_data));
          float v639_data = s2[54];
          float v641_data = ir3[6];
          ir3[6] = (v641_data + (v608_data * v639_data));
          float v644_data = s2[62];
          float v646_data = ir3[7];
          ir3[7] = (v646_data + (v608_data * v644_data));
          float v648_data = r2[7];
          float v649_data = s2[7];
          float v651_data = ir3[0];
          ir3[0] = (v651_data + (v648_data * v649_data));
          float v654_data = s2[15];
          float v656_data = ir3[1];
          ir3[1] = (v656_data + (v648_data * v654_data));
          float v659_data = s2[23];
          float v661_data = ir3[2];
          ir3[2] = (v661_data + (v648_data * v659_data));
          float v664_data = s2[31];
          float v666_data = ir3[3];
          ir3[3] = (v666_data + (v648_data * v664_data));
          float v669_data = s2[39];
          float v671_data = ir3[4];
          ir3[4] = (v671_data + (v648_data * v669_data));
          float v674_data = s2[47];
          float v676_data = ir3[5];
          ir3[5] = (v676_data + (v648_data * v674_data));
          float v679_data = s2[55];
          float v681_data = ir3[6];
          ir3[6] = (v681_data + (v648_data * v679_data));
          float v684_data = s2[63];
          float v686_data = ir3[7];
          ir3[7] = (v686_data + (v648_data * v684_data));
          // r3 = ir3 + r1
          #pragma unroll
          for (int32_t v688_n0 = 0; v688_n0 < 1; ++v688_n0) {
            #pragma unroll
            for (int32_t v689_n1 = 0; v689_n1 < 8; ++v689_n1) {
              int32_t v690_a = v688_n0 + v689_n1;
              float v691_data = ir3[v690_a];
              float v692_data = r1[v690_a];
              r3[v690_a] = (v692_data + v691_data);
            }
          }
          // glb_m4 = abs(r3)
          #pragma unroll
          for (int32_t v694_k0 = 0; v694_k0 < 1; ++v694_k0) {
            int32_t v700_lead = v25_lead + (v694_k0 * 8);
            #pragma unroll
            for (int32_t v695_k1 = 0; v695_k1 < 8; ++v695_k1) {
              float v697_data = r3[(v694_k0 + v695_k1)];
              glb_m4[(v700_lead + (v695_k1 * 8))] = (fabsf(v697_data));
            }
          }
          __syncwarp(0x000000ffu << (threadIdx.y % 4 * 8));
        }
      }
    }
  }
}

