// === base name ===
kernel_fc8a1ec82d9aa48b

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_fc8a1ec82d9aa48b = {{32, 4, 1}, 32, 32, 1, 4, 3072, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_fc8a1ec82d9aa48b(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_fc8a1ec82d9aa48b(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_fc8a1ec82d9aa48b(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_fc8a1ec82d9aa48b, block.x * block.y * block.z, 768 * sizeof(float));
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
void launcher_kernel_fc8a1ec82d9aa48b(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_fc8a1ec82d9aa48b(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_fc8a1ec82d9aa48b, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_fc8a1ec82d9aa48b<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_fc8a1ec82d9aa48b(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 4 per block = block 32x4x1, 3072 B shared, occupancy grid
    // operands:
    //   m0 32×13(32×13) {0..32}×{0..13} strided
    //   m1 32×12(32×12) {0..32}×{0..12} strided
    //   m2 12×13(12×13) {0..12}×{0..13} strided
    //   m3 32×13(32×13) {0..32}×{0..13} strided
    //   m4 13×13(13×13) {0..13}×{0..13} strided
    // operations:
    //   t0[i,j] = m0[i,j]
    //   t0[i,j] += m1[i,k] × m2[k,j]
    //   m0[i,j]@{0..32}×{4..5} = t0[i,j]@{0..32}×{4..5}
    //   m3[i,j] = m0[i,k] × m4[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":768}],"shared_bytes":3072,"shared_elements":768,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[32,13]],"name":"m0","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[32,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,12],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[12,13]],"name":"m2","ordered":false,"parts":1,"shape":[12,13],"variant":false},{"addressing":"strided","alias":"O","bbox":[[0,0],[32,13]],"name":"m3","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[13,13]],"name":"m4","ordered":false,"parts":1,"shape":[13,13],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,12]},{"addressing":"strided","bbox":[[0,0],[12,13]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[12,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,1]],"is_tmp":false,"name":"m0","offset":[0,4],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,1]],"is_tmp":true,"name":"t0","offset":[0,4],"shape":[32,13]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[0,0],[13,13]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[192 * threadIdx.y + 0];
      float * __restrict__ s1 = &localShrMem0[0];
      float * __restrict__ s2 = &localShrMem0[0];
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < numElements0; v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 416 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v9_batchId0 * 384 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v9_batchId0 * 156 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[v9_batchId0 * 416 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[v9_batchId0 * 169 + 0 + m4_extraOffset];
          float r0[13]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v25_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v26_i0 = 0; v26_i0 < 1; ++v26_i0) {
            int32_t v29_lead = v25_lead + (v26_i0 * 32);
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 13; ++v27_i1) {
              float v32_data = glb_m0[(v29_lead + (v27_i1 * 32))];
              r0[(v26_i0 + v27_i1)] = v32_data;
            }
          }
          float r2[12]{};
          // r2 = load{g>r}(glb_m1);
          #pragma unroll
          for (int32_t v35_i0 = 0; v35_i0 < 1; ++v35_i0) {
            int32_t v38_lead = v25_lead + (v35_i0 * 32);
            #pragma unroll
            for (int32_t v36_i1 = 0; v36_i1 < 12; ++v36_i1) {
              float v41_data = __ldcg(&glb_m1[(v38_lead + (v36_i1 * 32))]);
              r2[(v35_i0 + v36_i1)] = v41_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[13]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 13)] []
          float v44_data = r0[0];
          float v45_data = r1[0];
          r1[0] = (v45_data + v44_data);
          float v47_data = r0[1];
          float v48_data = r1[1];
          r1[1] = (v48_data + v47_data);
          float v50_data = r0[2];
          float v51_data = r1[2];
          r1[2] = (v51_data + v50_data);
          float v53_data = r0[3];
          float v54_data = r1[3];
          r1[3] = (v54_data + v53_data);
          float v56_data = r0[4];
          float v57_data = r1[4];
          r1[4] = (v57_data + v56_data);
          float v59_data = r0[5];
          float v60_data = r1[5];
          r1[5] = (v60_data + v59_data);
          float v62_data = r0[6];
          float v63_data = r1[6];
          r1[6] = (v63_data + v62_data);
          float v65_data = r0[7];
          float v66_data = r1[7];
          r1[7] = (v66_data + v65_data);
          float v68_data = r0[8];
          float v69_data = r1[8];
          r1[8] = (v69_data + v68_data);
          float v71_data = r0[9];
          float v72_data = r1[9];
          r1[9] = (v72_data + v71_data);
          float v74_data = r0[10];
          float v75_data = r1[10];
          r1[10] = (v75_data + v74_data);
          float v77_data = r0[11];
          float v78_data = r1[11];
          r1[11] = (v78_data + v77_data);
          float v80_data = r0[12];
          float v81_data = r1[12];
          r1[12] = (v81_data + v80_data);
          // s1 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 32], &glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 64], &glb_m2[0 + 0 + 1 * threadIdx.x + 64], 4);
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 96], &glb_m2[0 + 0 + 1 * threadIdx.x + 96], 4);
          if (threadIdx.x < 28) {
            __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 128], &glb_m2[0 + 0 + 1 * threadIdx.x + 128], 4);
          }
          __pipeline_commit();
          // wait(r2 = load{g>r}(glb_m1););
          // wait(s1 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r3[13]{};
          // ir3 = +(r2 * s1)
          // [(0, 32), (0, 13)] [(0, 12)]
          float ir3[13]{};
          float v90_data = r2[0];
          __syncwarp();
          float v91_data = s1[0];
          float v93_data = ir3[0];
          ir3[0] = (v93_data + (v90_data * v91_data));
          float v96_data = s1[12];
          float v98_data = ir3[1];
          ir3[1] = (v98_data + (v90_data * v96_data));
          float v101_data = s1[24];
          float v103_data = ir3[2];
          ir3[2] = (v103_data + (v90_data * v101_data));
          float v106_data = s1[36];
          float v108_data = ir3[3];
          ir3[3] = (v108_data + (v90_data * v106_data));
          float v111_data = s1[48];
          float v113_data = ir3[4];
          ir3[4] = (v113_data + (v90_data * v111_data));
          float v116_data = s1[60];
          float v118_data = ir3[5];
          ir3[5] = (v118_data + (v90_data * v116_data));
          float v121_data = s1[72];
          float v123_data = ir3[6];
          ir3[6] = (v123_data + (v90_data * v121_data));
          float v126_data = s1[84];
          float v128_data = ir3[7];
          ir3[7] = (v128_data + (v90_data * v126_data));
          float v131_data = s1[96];
          float v133_data = ir3[8];
          ir3[8] = (v133_data + (v90_data * v131_data));
          float v136_data = s1[108];
          float v138_data = ir3[9];
          ir3[9] = (v138_data + (v90_data * v136_data));
          float v141_data = s1[120];
          float v143_data = ir3[10];
          ir3[10] = (v143_data + (v90_data * v141_data));
          float v146_data = s1[132];
          float v148_data = ir3[11];
          ir3[11] = (v148_data + (v90_data * v146_data));
          float v151_data = s1[144];
          float v153_data = ir3[12];
          ir3[12] = (v153_data + (v90_data * v151_data));
          float v155_data = r2[1];
          float v156_data = s1[1];
          float v158_data = ir3[0];
          ir3[0] = (v158_data + (v155_data * v156_data));
          float v161_data = s1[13];
          float v163_data = ir3[1];
          ir3[1] = (v163_data + (v155_data * v161_data));
          float v166_data = s1[25];
          float v168_data = ir3[2];
          ir3[2] = (v168_data + (v155_data * v166_data));
          float v171_data = s1[37];
          float v173_data = ir3[3];
          ir3[3] = (v173_data + (v155_data * v171_data));
          float v176_data = s1[49];
          float v178_data = ir3[4];
          ir3[4] = (v178_data + (v155_data * v176_data));
          float v181_data = s1[61];
          float v183_data = ir3[5];
          ir3[5] = (v183_data + (v155_data * v181_data));
          float v186_data = s1[73];
          float v188_data = ir3[6];
          ir3[6] = (v188_data + (v155_data * v186_data));
          float v191_data = s1[85];
          float v193_data = ir3[7];
          ir3[7] = (v193_data + (v155_data * v191_data));
          float v196_data = s1[97];
          float v198_data = ir3[8];
          ir3[8] = (v198_data + (v155_data * v196_data));
          float v201_data = s1[109];
          float v203_data = ir3[9];
          ir3[9] = (v203_data + (v155_data * v201_data));
          float v206_data = s1[121];
          float v208_data = ir3[10];
          ir3[10] = (v208_data + (v155_data * v206_data));
          float v211_data = s1[133];
          float v213_data = ir3[11];
          ir3[11] = (v213_data + (v155_data * v211_data));
          float v216_data = s1[145];
          float v218_data = ir3[12];
          ir3[12] = (v218_data + (v155_data * v216_data));
          float v220_data = r2[2];
          float v221_data = s1[2];
          float v223_data = ir3[0];
          ir3[0] = (v223_data + (v220_data * v221_data));
          float v226_data = s1[14];
          float v228_data = ir3[1];
          ir3[1] = (v228_data + (v220_data * v226_data));
          float v231_data = s1[26];
          float v233_data = ir3[2];
          ir3[2] = (v233_data + (v220_data * v231_data));
          float v236_data = s1[38];
          float v238_data = ir3[3];
          ir3[3] = (v238_data + (v220_data * v236_data));
          float v241_data = s1[50];
          float v243_data = ir3[4];
          ir3[4] = (v243_data + (v220_data * v241_data));
          float v246_data = s1[62];
          float v248_data = ir3[5];
          ir3[5] = (v248_data + (v220_data * v246_data));
          float v251_data = s1[74];
          float v253_data = ir3[6];
          ir3[6] = (v253_data + (v220_data * v251_data));
          float v256_data = s1[86];
          float v258_data = ir3[7];
          ir3[7] = (v258_data + (v220_data * v256_data));
          float v261_data = s1[98];
          float v263_data = ir3[8];
          ir3[8] = (v263_data + (v220_data * v261_data));
          float v266_data = s1[110];
          float v268_data = ir3[9];
          ir3[9] = (v268_data + (v220_data * v266_data));
          float v271_data = s1[122];
          float v273_data = ir3[10];
          ir3[10] = (v273_data + (v220_data * v271_data));
          float v276_data = s1[134];
          float v278_data = ir3[11];
          ir3[11] = (v278_data + (v220_data * v276_data));
          float v281_data = s1[146];
          float v283_data = ir3[12];
          ir3[12] = (v283_data + (v220_data * v281_data));
          float v285_data = r2[3];
          float v286_data = s1[3];
          float v288_data = ir3[0];
          ir3[0] = (v288_data + (v285_data * v286_data));
          float v291_data = s1[15];
          float v293_data = ir3[1];
          ir3[1] = (v293_data + (v285_data * v291_data));
          float v296_data = s1[27];
          float v298_data = ir3[2];
          ir3[2] = (v298_data + (v285_data * v296_data));
          float v301_data = s1[39];
          float v303_data = ir3[3];
          ir3[3] = (v303_data + (v285_data * v301_data));
          float v306_data = s1[51];
          float v308_data = ir3[4];
          ir3[4] = (v308_data + (v285_data * v306_data));
          float v311_data = s1[63];
          float v313_data = ir3[5];
          ir3[5] = (v313_data + (v285_data * v311_data));
          float v316_data = s1[75];
          float v318_data = ir3[6];
          ir3[6] = (v318_data + (v285_data * v316_data));
          float v321_data = s1[87];
          float v323_data = ir3[7];
          ir3[7] = (v323_data + (v285_data * v321_data));
          float v326_data = s1[99];
          float v328_data = ir3[8];
          ir3[8] = (v328_data + (v285_data * v326_data));
          float v331_data = s1[111];
          float v333_data = ir3[9];
          ir3[9] = (v333_data + (v285_data * v331_data));
          float v336_data = s1[123];
          float v338_data = ir3[10];
          ir3[10] = (v338_data + (v285_data * v336_data));
          float v341_data = s1[135];
          float v343_data = ir3[11];
          ir3[11] = (v343_data + (v285_data * v341_data));
          float v346_data = s1[147];
          float v348_data = ir3[12];
          ir3[12] = (v348_data + (v285_data * v346_data));
          float v350_data = r2[4];
          float v351_data = s1[4];
          float v353_data = ir3[0];
          ir3[0] = (v353_data + (v350_data * v351_data));
          float v356_data = s1[16];
          float v358_data = ir3[1];
          ir3[1] = (v358_data + (v350_data * v356_data));
          float v361_data = s1[28];
          float v363_data = ir3[2];
          ir3[2] = (v363_data + (v350_data * v361_data));
          float v366_data = s1[40];
          float v368_data = ir3[3];
          ir3[3] = (v368_data + (v350_data * v366_data));
          float v371_data = s1[52];
          float v373_data = ir3[4];
          ir3[4] = (v373_data + (v350_data * v371_data));
          float v376_data = s1[64];
          float v378_data = ir3[5];
          ir3[5] = (v378_data + (v350_data * v376_data));
          float v381_data = s1[76];
          float v383_data = ir3[6];
          ir3[6] = (v383_data + (v350_data * v381_data));
          float v386_data = s1[88];
          float v388_data = ir3[7];
          ir3[7] = (v388_data + (v350_data * v386_data));
          float v391_data = s1[100];
          float v393_data = ir3[8];
          ir3[8] = (v393_data + (v350_data * v391_data));
          float v396_data = s1[112];
          float v398_data = ir3[9];
          ir3[9] = (v398_data + (v350_data * v396_data));
          float v401_data = s1[124];
          float v403_data = ir3[10];
          ir3[10] = (v403_data + (v350_data * v401_data));
          float v406_data = s1[136];
          float v408_data = ir3[11];
          ir3[11] = (v408_data + (v350_data * v406_data));
          float v411_data = s1[148];
          float v413_data = ir3[12];
          ir3[12] = (v413_data + (v350_data * v411_data));
          float v415_data = r2[5];
          float v416_data = s1[5];
          float v418_data = ir3[0];
          ir3[0] = (v418_data + (v415_data * v416_data));
          float v421_data = s1[17];
          float v423_data = ir3[1];
          ir3[1] = (v423_data + (v415_data * v421_data));
          float v426_data = s1[29];
          float v428_data = ir3[2];
          ir3[2] = (v428_data + (v415_data * v426_data));
          float v431_data = s1[41];
          float v433_data = ir3[3];
          ir3[3] = (v433_data + (v415_data * v431_data));
          float v436_data = s1[53];
          float v438_data = ir3[4];
          ir3[4] = (v438_data + (v415_data * v436_data));
          float v441_data = s1[65];
          float v443_data = ir3[5];
          ir3[5] = (v443_data + (v415_data * v441_data));
          float v446_data = s1[77];
          float v448_data = ir3[6];
          ir3[6] = (v448_data + (v415_data * v446_data));
          float v451_data = s1[89];
          float v453_data = ir3[7];
          ir3[7] = (v453_data + (v415_data * v451_data));
          float v456_data = s1[101];
          float v458_data = ir3[8];
          ir3[8] = (v458_data + (v415_data * v456_data));
          float v461_data = s1[113];
          float v463_data = ir3[9];
          ir3[9] = (v463_data + (v415_data * v461_data));
          float v466_data = s1[125];
          float v468_data = ir3[10];
          ir3[10] = (v468_data + (v415_data * v466_data));
          float v471_data = s1[137];
          float v473_data = ir3[11];
          ir3[11] = (v473_data + (v415_data * v471_data));
          float v476_data = s1[149];
          float v478_data = ir3[12];
          ir3[12] = (v478_data + (v415_data * v476_data));
          float v480_data = r2[6];
          float v481_data = s1[6];
          float v483_data = ir3[0];
          ir3[0] = (v483_data + (v480_data * v481_data));
          float v486_data = s1[18];
          float v488_data = ir3[1];
          ir3[1] = (v488_data + (v480_data * v486_data));
          float v491_data = s1[30];
          float v493_data = ir3[2];
          ir3[2] = (v493_data + (v480_data * v491_data));
          float v496_data = s1[42];
          float v498_data = ir3[3];
          ir3[3] = (v498_data + (v480_data * v496_data));
          float v501_data = s1[54];
          float v503_data = ir3[4];
          ir3[4] = (v503_data + (v480_data * v501_data));
          float v506_data = s1[66];
          float v508_data = ir3[5];
          ir3[5] = (v508_data + (v480_data * v506_data));
          float v511_data = s1[78];
          float v513_data = ir3[6];
          ir3[6] = (v513_data + (v480_data * v511_data));
          float v516_data = s1[90];
          float v518_data = ir3[7];
          ir3[7] = (v518_data + (v480_data * v516_data));
          float v521_data = s1[102];
          float v523_data = ir3[8];
          ir3[8] = (v523_data + (v480_data * v521_data));
          float v526_data = s1[114];
          float v528_data = ir3[9];
          ir3[9] = (v528_data + (v480_data * v526_data));
          float v531_data = s1[126];
          float v533_data = ir3[10];
          ir3[10] = (v533_data + (v480_data * v531_data));
          float v536_data = s1[138];
          float v538_data = ir3[11];
          ir3[11] = (v538_data + (v480_data * v536_data));
          float v541_data = s1[150];
          float v543_data = ir3[12];
          ir3[12] = (v543_data + (v480_data * v541_data));
          float v545_data = r2[7];
          float v546_data = s1[7];
          float v548_data = ir3[0];
          ir3[0] = (v548_data + (v545_data * v546_data));
          float v551_data = s1[19];
          float v553_data = ir3[1];
          ir3[1] = (v553_data + (v545_data * v551_data));
          float v556_data = s1[31];
          float v558_data = ir3[2];
          ir3[2] = (v558_data + (v545_data * v556_data));
          float v561_data = s1[43];
          float v563_data = ir3[3];
          ir3[3] = (v563_data + (v545_data * v561_data));
          float v566_data = s1[55];
          float v568_data = ir3[4];
          ir3[4] = (v568_data + (v545_data * v566_data));
          float v571_data = s1[67];
          float v573_data = ir3[5];
          ir3[5] = (v573_data + (v545_data * v571_data));
          float v576_data = s1[79];
          float v578_data = ir3[6];
          ir3[6] = (v578_data + (v545_data * v576_data));
          float v581_data = s1[91];
          float v583_data = ir3[7];
          ir3[7] = (v583_data + (v545_data * v581_data));
          float v586_data = s1[103];
          float v588_data = ir3[8];
          ir3[8] = (v588_data + (v545_data * v586_data));
          float v591_data = s1[115];
          float v593_data = ir3[9];
          ir3[9] = (v593_data + (v545_data * v591_data));
          float v596_data = s1[127];
          float v598_data = ir3[10];
          ir3[10] = (v598_data + (v545_data * v596_data));
          float v601_data = s1[139];
          float v603_data = ir3[11];
          ir3[11] = (v603_data + (v545_data * v601_data));
          float v606_data = s1[151];
          float v608_data = ir3[12];
          ir3[12] = (v608_data + (v545_data * v606_data));
          float v610_data = r2[8];
          float v611_data = s1[8];
          float v613_data = ir3[0];
          ir3[0] = (v613_data + (v610_data * v611_data));
          float v616_data = s1[20];
          float v618_data = ir3[1];
          ir3[1] = (v618_data + (v610_data * v616_data));
          float v621_data = s1[32];
          float v623_data = ir3[2];
          ir3[2] = (v623_data + (v610_data * v621_data));
          float v626_data = s1[44];
          float v628_data = ir3[3];
          ir3[3] = (v628_data + (v610_data * v626_data));
          float v631_data = s1[56];
          float v633_data = ir3[4];
          ir3[4] = (v633_data + (v610_data * v631_data));
          float v636_data = s1[68];
          float v638_data = ir3[5];
          ir3[5] = (v638_data + (v610_data * v636_data));
          float v641_data = s1[80];
          float v643_data = ir3[6];
          ir3[6] = (v643_data + (v610_data * v641_data));
          float v646_data = s1[92];
          float v648_data = ir3[7];
          ir3[7] = (v648_data + (v610_data * v646_data));
          float v651_data = s1[104];
          float v653_data = ir3[8];
          ir3[8] = (v653_data + (v610_data * v651_data));
          float v656_data = s1[116];
          float v658_data = ir3[9];
          ir3[9] = (v658_data + (v610_data * v656_data));
          float v661_data = s1[128];
          float v663_data = ir3[10];
          ir3[10] = (v663_data + (v610_data * v661_data));
          float v666_data = s1[140];
          float v668_data = ir3[11];
          ir3[11] = (v668_data + (v610_data * v666_data));
          float v671_data = s1[152];
          float v673_data = ir3[12];
          ir3[12] = (v673_data + (v610_data * v671_data));
          float v675_data = r2[9];
          float v676_data = s1[9];
          float v678_data = ir3[0];
          ir3[0] = (v678_data + (v675_data * v676_data));
          float v681_data = s1[21];
          float v683_data = ir3[1];
          ir3[1] = (v683_data + (v675_data * v681_data));
          float v686_data = s1[33];
          float v688_data = ir3[2];
          ir3[2] = (v688_data + (v675_data * v686_data));
          float v691_data = s1[45];
          float v693_data = ir3[3];
          ir3[3] = (v693_data + (v675_data * v691_data));
          float v696_data = s1[57];
          float v698_data = ir3[4];
          ir3[4] = (v698_data + (v675_data * v696_data));
          float v701_data = s1[69];
          float v703_data = ir3[5];
          ir3[5] = (v703_data + (v675_data * v701_data));
          float v706_data = s1[81];
          float v708_data = ir3[6];
          ir3[6] = (v708_data + (v675_data * v706_data));
          float v711_data = s1[93];
          float v713_data = ir3[7];
          ir3[7] = (v713_data + (v675_data * v711_data));
          float v716_data = s1[105];
          float v718_data = ir3[8];
          ir3[8] = (v718_data + (v675_data * v716_data));
          float v721_data = s1[117];
          float v723_data = ir3[9];
          ir3[9] = (v723_data + (v675_data * v721_data));
          float v726_data = s1[129];
          float v728_data = ir3[10];
          ir3[10] = (v728_data + (v675_data * v726_data));
          float v731_data = s1[141];
          float v733_data = ir3[11];
          ir3[11] = (v733_data + (v675_data * v731_data));
          float v736_data = s1[153];
          float v738_data = ir3[12];
          ir3[12] = (v738_data + (v675_data * v736_data));
          float v740_data = r2[10];
          float v741_data = s1[10];
          float v743_data = ir3[0];
          ir3[0] = (v743_data + (v740_data * v741_data));
          float v746_data = s1[22];
          float v748_data = ir3[1];
          ir3[1] = (v748_data + (v740_data * v746_data));
          float v751_data = s1[34];
          float v753_data = ir3[2];
          ir3[2] = (v753_data + (v740_data * v751_data));
          float v756_data = s1[46];
          float v758_data = ir3[3];
          ir3[3] = (v758_data + (v740_data * v756_data));
          float v761_data = s1[58];
          float v763_data = ir3[4];
          ir3[4] = (v763_data + (v740_data * v761_data));
          float v766_data = s1[70];
          float v768_data = ir3[5];
          ir3[5] = (v768_data + (v740_data * v766_data));
          float v771_data = s1[82];
          float v773_data = ir3[6];
          ir3[6] = (v773_data + (v740_data * v771_data));
          float v776_data = s1[94];
          float v778_data = ir3[7];
          ir3[7] = (v778_data + (v740_data * v776_data));
          float v781_data = s1[106];
          float v783_data = ir3[8];
          ir3[8] = (v783_data + (v740_data * v781_data));
          float v786_data = s1[118];
          float v788_data = ir3[9];
          ir3[9] = (v788_data + (v740_data * v786_data));
          float v791_data = s1[130];
          float v793_data = ir3[10];
          ir3[10] = (v793_data + (v740_data * v791_data));
          float v796_data = s1[142];
          float v798_data = ir3[11];
          ir3[11] = (v798_data + (v740_data * v796_data));
          float v801_data = s1[154];
          float v803_data = ir3[12];
          ir3[12] = (v803_data + (v740_data * v801_data));
          float v805_data = r2[11];
          float v806_data = s1[11];
          float v808_data = ir3[0];
          ir3[0] = (v808_data + (v805_data * v806_data));
          float v811_data = s1[23];
          float v813_data = ir3[1];
          ir3[1] = (v813_data + (v805_data * v811_data));
          float v816_data = s1[35];
          float v818_data = ir3[2];
          ir3[2] = (v818_data + (v805_data * v816_data));
          float v821_data = s1[47];
          float v823_data = ir3[3];
          ir3[3] = (v823_data + (v805_data * v821_data));
          float v826_data = s1[59];
          float v828_data = ir3[4];
          ir3[4] = (v828_data + (v805_data * v826_data));
          float v831_data = s1[71];
          float v833_data = ir3[5];
          ir3[5] = (v833_data + (v805_data * v831_data));
          float v836_data = s1[83];
          float v838_data = ir3[6];
          ir3[6] = (v838_data + (v805_data * v836_data));
          float v841_data = s1[95];
          float v843_data = ir3[7];
          ir3[7] = (v843_data + (v805_data * v841_data));
          float v846_data = s1[107];
          float v848_data = ir3[8];
          ir3[8] = (v848_data + (v805_data * v846_data));
          float v851_data = s1[119];
          float v853_data = ir3[9];
          ir3[9] = (v853_data + (v805_data * v851_data));
          float v856_data = s1[131];
          float v858_data = ir3[10];
          ir3[10] = (v858_data + (v805_data * v856_data));
          float v861_data = s1[143];
          float v863_data = ir3[11];
          ir3[11] = (v863_data + (v805_data * v861_data));
          float v866_data = s1[155];
          float v868_data = ir3[12];
          ir3[12] = (v868_data + (v805_data * v866_data));
          // r3 = ir3 + r1
          #pragma unroll
          for (int32_t v870_n0 = 0; v870_n0 < 1; ++v870_n0) {
            #pragma unroll
            for (int32_t v871_n1 = 0; v871_n1 < 13; ++v871_n1) {
              int32_t v872_a = v870_n0 + v871_n1;
              float v873_data = ir3[v872_a];
              float v874_data = r1[v872_a];
              r3[v872_a] = (v874_data + v873_data);
            }
          }
          float r4[1]{};
          // ir4 = +(r3)
          // [(0, 32), (0, 1)] []
          float ir4[1]{};
          float v878_data = r3[4];
          float v879_data = ir4[0];
          ir4[0] = (v879_data + v878_data);
          // r4 = ir4
          #pragma unroll
          for (int32_t v881_n0 = 0; v881_n0 < 1; ++v881_n0) {
            #pragma unroll
            for (int32_t v882_n1 = 0; v882_n1 < 1; ++v882_n1) {
              int32_t v883_a = v881_n0 + v882_n1;
              float v884_data = ir4[v883_a];
              r4[v883_a] = v884_data;
            }
          }
          // glb_m0 = store{r>g}(r4);
          #pragma unroll
          for (int32_t v885_i0 = 0; v885_i0 < 1; ++v885_i0) {
            int32_t v890_lead = v25_lead + (v885_i0 * 32);
            #pragma unroll
            for (int32_t v886_i1 = 0; v886_i1 < 1; ++v886_i1) {
              float v888_data = r4[(v885_i0 + v886_i1)];
              glb_m0[(v890_lead + ((v886_i1 + 4) * 32))] = v888_data;
            }
          }
          float r5[13]{};
          // r5 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v895_i0 = 0; v895_i0 < 1; ++v895_i0) {
            int32_t v898_lead = v25_lead + (v895_i0 * 32);
            #pragma unroll
            for (int32_t v896_i1 = 0; v896_i1 < 13; ++v896_i1) {
              float v901_data = glb_m0[(v898_lead + (v896_i1 * 32))];
              r5[(v895_i0 + v896_i1)] = v901_data;
            }
          }
          // s2 = load{g>s}(glb_m4[0, 1])
          __syncwarp();
          #pragma unroll
          for (int32_t i = 0; i < 5; i += 1) {
            __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + i * 32], &glb_m4[0 + 0 + 1 * threadIdx.x + i * 32], 4);
          }
          if (threadIdx.x < 9) {
            __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 160], &glb_m4[0 + 0 + 1 * threadIdx.x + 160], 4);
          }
          __pipeline_commit();
          // wait(r5 = load{g>r}(glb_m0););
          // wait(s2 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          float r6[13]{};
          // ir6 = +(r5 * s2)
          // [(0, 32), (0, 13)] [(0, 13)]
          float ir6[13]{};
          float v907_data = r5[0];
          __syncwarp();
          float v908_data = s2[0];
          float v910_data = ir6[0];
          ir6[0] = (v910_data + (v907_data * v908_data));
          float v913_data = s2[13];
          float v915_data = ir6[1];
          ir6[1] = (v915_data + (v907_data * v913_data));
          float v918_data = s2[26];
          float v920_data = ir6[2];
          ir6[2] = (v920_data + (v907_data * v918_data));
          float v923_data = s2[39];
          float v925_data = ir6[3];
          ir6[3] = (v925_data + (v907_data * v923_data));
          float v928_data = s2[52];
          float v930_data = ir6[4];
          ir6[4] = (v930_data + (v907_data * v928_data));
          float v933_data = s2[65];
          float v935_data = ir6[5];
          ir6[5] = (v935_data + (v907_data * v933_data));
          float v938_data = s2[78];
          float v940_data = ir6[6];
          ir6[6] = (v940_data + (v907_data * v938_data));
          float v943_data = s2[91];
          float v945_data = ir6[7];
          ir6[7] = (v945_data + (v907_data * v943_data));
          float v948_data = s2[104];
          float v950_data = ir6[8];
          ir6[8] = (v950_data + (v907_data * v948_data));
          float v953_data = s2[117];
          float v955_data = ir6[9];
          ir6[9] = (v955_data + (v907_data * v953_data));
          float v958_data = s2[130];
          float v960_data = ir6[10];
          ir6[10] = (v960_data + (v907_data * v958_data));
          float v963_data = s2[143];
          float v965_data = ir6[11];
          ir6[11] = (v965_data + (v907_data * v963_data));
          float v968_data = s2[156];
          float v970_data = ir6[12];
          ir6[12] = (v970_data + (v907_data * v968_data));
          float v972_data = r5[1];
          float v973_data = s2[1];
          float v975_data = ir6[0];
          ir6[0] = (v975_data + (v972_data * v973_data));
          float v978_data = s2[14];
          float v980_data = ir6[1];
          ir6[1] = (v980_data + (v972_data * v978_data));
          float v983_data = s2[27];
          float v985_data = ir6[2];
          ir6[2] = (v985_data + (v972_data * v983_data));
          float v988_data = s2[40];
          float v990_data = ir6[3];
          ir6[3] = (v990_data + (v972_data * v988_data));
          float v993_data = s2[53];
          float v995_data = ir6[4];
          ir6[4] = (v995_data + (v972_data * v993_data));
          float v998_data = s2[66];
          float v1000_data = ir6[5];
          ir6[5] = (v1000_data + (v972_data * v998_data));
          float v1003_data = s2[79];
          float v1005_data = ir6[6];
          ir6[6] = (v1005_data + (v972_data * v1003_data));
          float v1008_data = s2[92];
          float v1010_data = ir6[7];
          ir6[7] = (v1010_data + (v972_data * v1008_data));
          float v1013_data = s2[105];
          float v1015_data = ir6[8];
          ir6[8] = (v1015_data + (v972_data * v1013_data));
          float v1018_data = s2[118];
          float v1020_data = ir6[9];
          ir6[9] = (v1020_data + (v972_data * v1018_data));
          float v1023_data = s2[131];
          float v1025_data = ir6[10];
          ir6[10] = (v1025_data + (v972_data * v1023_data));
          float v1028_data = s2[144];
          float v1030_data = ir6[11];
          ir6[11] = (v1030_data + (v972_data * v1028_data));
          float v1033_data = s2[157];
          float v1035_data = ir6[12];
          ir6[12] = (v1035_data + (v972_data * v1033_data));
          float v1037_data = r5[2];
          float v1038_data = s2[2];
          float v1040_data = ir6[0];
          ir6[0] = (v1040_data + (v1037_data * v1038_data));
          float v1043_data = s2[15];
          float v1045_data = ir6[1];
          ir6[1] = (v1045_data + (v1037_data * v1043_data));
          float v1048_data = s2[28];
          float v1050_data = ir6[2];
          ir6[2] = (v1050_data + (v1037_data * v1048_data));
          float v1053_data = s2[41];
          float v1055_data = ir6[3];
          ir6[3] = (v1055_data + (v1037_data * v1053_data));
          float v1058_data = s2[54];
          float v1060_data = ir6[4];
          ir6[4] = (v1060_data + (v1037_data * v1058_data));
          float v1063_data = s2[67];
          float v1065_data = ir6[5];
          ir6[5] = (v1065_data + (v1037_data * v1063_data));
          float v1068_data = s2[80];
          float v1070_data = ir6[6];
          ir6[6] = (v1070_data + (v1037_data * v1068_data));
          float v1073_data = s2[93];
          float v1075_data = ir6[7];
          ir6[7] = (v1075_data + (v1037_data * v1073_data));
          float v1078_data = s2[106];
          float v1080_data = ir6[8];
          ir6[8] = (v1080_data + (v1037_data * v1078_data));
          float v1083_data = s2[119];
          float v1085_data = ir6[9];
          ir6[9] = (v1085_data + (v1037_data * v1083_data));
          float v1088_data = s2[132];
          float v1090_data = ir6[10];
          ir6[10] = (v1090_data + (v1037_data * v1088_data));
          float v1093_data = s2[145];
          float v1095_data = ir6[11];
          ir6[11] = (v1095_data + (v1037_data * v1093_data));
          float v1098_data = s2[158];
          float v1100_data = ir6[12];
          ir6[12] = (v1100_data + (v1037_data * v1098_data));
          float v1102_data = r5[3];
          float v1103_data = s2[3];
          float v1105_data = ir6[0];
          ir6[0] = (v1105_data + (v1102_data * v1103_data));
          float v1108_data = s2[16];
          float v1110_data = ir6[1];
          ir6[1] = (v1110_data + (v1102_data * v1108_data));
          float v1113_data = s2[29];
          float v1115_data = ir6[2];
          ir6[2] = (v1115_data + (v1102_data * v1113_data));
          float v1118_data = s2[42];
          float v1120_data = ir6[3];
          ir6[3] = (v1120_data + (v1102_data * v1118_data));
          float v1123_data = s2[55];
          float v1125_data = ir6[4];
          ir6[4] = (v1125_data + (v1102_data * v1123_data));
          float v1128_data = s2[68];
          float v1130_data = ir6[5];
          ir6[5] = (v1130_data + (v1102_data * v1128_data));
          float v1133_data = s2[81];
          float v1135_data = ir6[6];
          ir6[6] = (v1135_data + (v1102_data * v1133_data));
          float v1138_data = s2[94];
          float v1140_data = ir6[7];
          ir6[7] = (v1140_data + (v1102_data * v1138_data));
          float v1143_data = s2[107];
          float v1145_data = ir6[8];
          ir6[8] = (v1145_data + (v1102_data * v1143_data));
          float v1148_data = s2[120];
          float v1150_data = ir6[9];
          ir6[9] = (v1150_data + (v1102_data * v1148_data));
          float v1153_data = s2[133];
          float v1155_data = ir6[10];
          ir6[10] = (v1155_data + (v1102_data * v1153_data));
          float v1158_data = s2[146];
          float v1160_data = ir6[11];
          ir6[11] = (v1160_data + (v1102_data * v1158_data));
          float v1163_data = s2[159];
          float v1165_data = ir6[12];
          ir6[12] = (v1165_data + (v1102_data * v1163_data));
          float v1167_data = r5[4];
          float v1168_data = s2[4];
          float v1170_data = ir6[0];
          ir6[0] = (v1170_data + (v1167_data * v1168_data));
          float v1173_data = s2[17];
          float v1175_data = ir6[1];
          ir6[1] = (v1175_data + (v1167_data * v1173_data));
          float v1178_data = s2[30];
          float v1180_data = ir6[2];
          ir6[2] = (v1180_data + (v1167_data * v1178_data));
          float v1183_data = s2[43];
          float v1185_data = ir6[3];
          ir6[3] = (v1185_data + (v1167_data * v1183_data));
          float v1188_data = s2[56];
          float v1190_data = ir6[4];
          ir6[4] = (v1190_data + (v1167_data * v1188_data));
          float v1193_data = s2[69];
          float v1195_data = ir6[5];
          ir6[5] = (v1195_data + (v1167_data * v1193_data));
          float v1198_data = s2[82];
          float v1200_data = ir6[6];
          ir6[6] = (v1200_data + (v1167_data * v1198_data));
          float v1203_data = s2[95];
          float v1205_data = ir6[7];
          ir6[7] = (v1205_data + (v1167_data * v1203_data));
          float v1208_data = s2[108];
          float v1210_data = ir6[8];
          ir6[8] = (v1210_data + (v1167_data * v1208_data));
          float v1213_data = s2[121];
          float v1215_data = ir6[9];
          ir6[9] = (v1215_data + (v1167_data * v1213_data));
          float v1218_data = s2[134];
          float v1220_data = ir6[10];
          ir6[10] = (v1220_data + (v1167_data * v1218_data));
          float v1223_data = s2[147];
          float v1225_data = ir6[11];
          ir6[11] = (v1225_data + (v1167_data * v1223_data));
          float v1228_data = s2[160];
          float v1230_data = ir6[12];
          ir6[12] = (v1230_data + (v1167_data * v1228_data));
          float v1232_data = r5[5];
          float v1233_data = s2[5];
          float v1235_data = ir6[0];
          ir6[0] = (v1235_data + (v1232_data * v1233_data));
          float v1238_data = s2[18];
          float v1240_data = ir6[1];
          ir6[1] = (v1240_data + (v1232_data * v1238_data));
          float v1243_data = s2[31];
          float v1245_data = ir6[2];
          ir6[2] = (v1245_data + (v1232_data * v1243_data));
          float v1248_data = s2[44];
          float v1250_data = ir6[3];
          ir6[3] = (v1250_data + (v1232_data * v1248_data));
          float v1253_data = s2[57];
          float v1255_data = ir6[4];
          ir6[4] = (v1255_data + (v1232_data * v1253_data));
          float v1258_data = s2[70];
          float v1260_data = ir6[5];
          ir6[5] = (v1260_data + (v1232_data * v1258_data));
          float v1263_data = s2[83];
          float v1265_data = ir6[6];
          ir6[6] = (v1265_data + (v1232_data * v1263_data));
          float v1268_data = s2[96];
          float v1270_data = ir6[7];
          ir6[7] = (v1270_data + (v1232_data * v1268_data));
          float v1273_data = s2[109];
          float v1275_data = ir6[8];
          ir6[8] = (v1275_data + (v1232_data * v1273_data));
          float v1278_data = s2[122];
          float v1280_data = ir6[9];
          ir6[9] = (v1280_data + (v1232_data * v1278_data));
          float v1283_data = s2[135];
          float v1285_data = ir6[10];
          ir6[10] = (v1285_data + (v1232_data * v1283_data));
          float v1288_data = s2[148];
          float v1290_data = ir6[11];
          ir6[11] = (v1290_data + (v1232_data * v1288_data));
          float v1293_data = s2[161];
          float v1295_data = ir6[12];
          ir6[12] = (v1295_data + (v1232_data * v1293_data));
          float v1297_data = r5[6];
          float v1298_data = s2[6];
          float v1300_data = ir6[0];
          ir6[0] = (v1300_data + (v1297_data * v1298_data));
          float v1303_data = s2[19];
          float v1305_data = ir6[1];
          ir6[1] = (v1305_data + (v1297_data * v1303_data));
          float v1308_data = s2[32];
          float v1310_data = ir6[2];
          ir6[2] = (v1310_data + (v1297_data * v1308_data));
          float v1313_data = s2[45];
          float v1315_data = ir6[3];
          ir6[3] = (v1315_data + (v1297_data * v1313_data));
          float v1318_data = s2[58];
          float v1320_data = ir6[4];
          ir6[4] = (v1320_data + (v1297_data * v1318_data));
          float v1323_data = s2[71];
          float v1325_data = ir6[5];
          ir6[5] = (v1325_data + (v1297_data * v1323_data));
          float v1328_data = s2[84];
          float v1330_data = ir6[6];
          ir6[6] = (v1330_data + (v1297_data * v1328_data));
          float v1333_data = s2[97];
          float v1335_data = ir6[7];
          ir6[7] = (v1335_data + (v1297_data * v1333_data));
          float v1338_data = s2[110];
          float v1340_data = ir6[8];
          ir6[8] = (v1340_data + (v1297_data * v1338_data));
          float v1343_data = s2[123];
          float v1345_data = ir6[9];
          ir6[9] = (v1345_data + (v1297_data * v1343_data));
          float v1348_data = s2[136];
          float v1350_data = ir6[10];
          ir6[10] = (v1350_data + (v1297_data * v1348_data));
          float v1353_data = s2[149];
          float v1355_data = ir6[11];
          ir6[11] = (v1355_data + (v1297_data * v1353_data));
          float v1358_data = s2[162];
          float v1360_data = ir6[12];
          ir6[12] = (v1360_data + (v1297_data * v1358_data));
          float v1362_data = r5[7];
          float v1363_data = s2[7];
          float v1365_data = ir6[0];
          ir6[0] = (v1365_data + (v1362_data * v1363_data));
          float v1368_data = s2[20];
          float v1370_data = ir6[1];
          ir6[1] = (v1370_data + (v1362_data * v1368_data));
          float v1373_data = s2[33];
          float v1375_data = ir6[2];
          ir6[2] = (v1375_data + (v1362_data * v1373_data));
          float v1378_data = s2[46];
          float v1380_data = ir6[3];
          ir6[3] = (v1380_data + (v1362_data * v1378_data));
          float v1383_data = s2[59];
          float v1385_data = ir6[4];
          ir6[4] = (v1385_data + (v1362_data * v1383_data));
          float v1388_data = s2[72];
          float v1390_data = ir6[5];
          ir6[5] = (v1390_data + (v1362_data * v1388_data));
          float v1393_data = s2[85];
          float v1395_data = ir6[6];
          ir6[6] = (v1395_data + (v1362_data * v1393_data));
          float v1398_data = s2[98];
          float v1400_data = ir6[7];
          ir6[7] = (v1400_data + (v1362_data * v1398_data));
          float v1403_data = s2[111];
          float v1405_data = ir6[8];
          ir6[8] = (v1405_data + (v1362_data * v1403_data));
          float v1408_data = s2[124];
          float v1410_data = ir6[9];
          ir6[9] = (v1410_data + (v1362_data * v1408_data));
          float v1413_data = s2[137];
          float v1415_data = ir6[10];
          ir6[10] = (v1415_data + (v1362_data * v1413_data));
          float v1418_data = s2[150];
          float v1420_data = ir6[11];
          ir6[11] = (v1420_data + (v1362_data * v1418_data));
          float v1423_data = s2[163];
          float v1425_data = ir6[12];
          ir6[12] = (v1425_data + (v1362_data * v1423_data));
          float v1427_data = r5[8];
          float v1428_data = s2[8];
          float v1430_data = ir6[0];
          ir6[0] = (v1430_data + (v1427_data * v1428_data));
          float v1433_data = s2[21];
          float v1435_data = ir6[1];
          ir6[1] = (v1435_data + (v1427_data * v1433_data));
          float v1438_data = s2[34];
          float v1440_data = ir6[2];
          ir6[2] = (v1440_data + (v1427_data * v1438_data));
          float v1443_data = s2[47];
          float v1445_data = ir6[3];
          ir6[3] = (v1445_data + (v1427_data * v1443_data));
          float v1448_data = s2[60];
          float v1450_data = ir6[4];
          ir6[4] = (v1450_data + (v1427_data * v1448_data));
          float v1453_data = s2[73];
          float v1455_data = ir6[5];
          ir6[5] = (v1455_data + (v1427_data * v1453_data));
          float v1458_data = s2[86];
          float v1460_data = ir6[6];
          ir6[6] = (v1460_data + (v1427_data * v1458_data));
          float v1463_data = s2[99];
          float v1465_data = ir6[7];
          ir6[7] = (v1465_data + (v1427_data * v1463_data));
          float v1468_data = s2[112];
          float v1470_data = ir6[8];
          ir6[8] = (v1470_data + (v1427_data * v1468_data));
          float v1473_data = s2[125];
          float v1475_data = ir6[9];
          ir6[9] = (v1475_data + (v1427_data * v1473_data));
          float v1478_data = s2[138];
          float v1480_data = ir6[10];
          ir6[10] = (v1480_data + (v1427_data * v1478_data));
          float v1483_data = s2[151];
          float v1485_data = ir6[11];
          ir6[11] = (v1485_data + (v1427_data * v1483_data));
          float v1488_data = s2[164];
          float v1490_data = ir6[12];
          ir6[12] = (v1490_data + (v1427_data * v1488_data));
          float v1492_data = r5[9];
          float v1493_data = s2[9];
          float v1495_data = ir6[0];
          ir6[0] = (v1495_data + (v1492_data * v1493_data));
          float v1498_data = s2[22];
          float v1500_data = ir6[1];
          ir6[1] = (v1500_data + (v1492_data * v1498_data));
          float v1503_data = s2[35];
          float v1505_data = ir6[2];
          ir6[2] = (v1505_data + (v1492_data * v1503_data));
          float v1508_data = s2[48];
          float v1510_data = ir6[3];
          ir6[3] = (v1510_data + (v1492_data * v1508_data));
          float v1513_data = s2[61];
          float v1515_data = ir6[4];
          ir6[4] = (v1515_data + (v1492_data * v1513_data));
          float v1518_data = s2[74];
          float v1520_data = ir6[5];
          ir6[5] = (v1520_data + (v1492_data * v1518_data));
          float v1523_data = s2[87];
          float v1525_data = ir6[6];
          ir6[6] = (v1525_data + (v1492_data * v1523_data));
          float v1528_data = s2[100];
          float v1530_data = ir6[7];
          ir6[7] = (v1530_data + (v1492_data * v1528_data));
          float v1533_data = s2[113];
          float v1535_data = ir6[8];
          ir6[8] = (v1535_data + (v1492_data * v1533_data));
          float v1538_data = s2[126];
          float v1540_data = ir6[9];
          ir6[9] = (v1540_data + (v1492_data * v1538_data));
          float v1543_data = s2[139];
          float v1545_data = ir6[10];
          ir6[10] = (v1545_data + (v1492_data * v1543_data));
          float v1548_data = s2[152];
          float v1550_data = ir6[11];
          ir6[11] = (v1550_data + (v1492_data * v1548_data));
          float v1553_data = s2[165];
          float v1555_data = ir6[12];
          ir6[12] = (v1555_data + (v1492_data * v1553_data));
          float v1557_data = r5[10];
          float v1558_data = s2[10];
          float v1560_data = ir6[0];
          ir6[0] = (v1560_data + (v1557_data * v1558_data));
          float v1563_data = s2[23];
          float v1565_data = ir6[1];
          ir6[1] = (v1565_data + (v1557_data * v1563_data));
          float v1568_data = s2[36];
          float v1570_data = ir6[2];
          ir6[2] = (v1570_data + (v1557_data * v1568_data));
          float v1573_data = s2[49];
          float v1575_data = ir6[3];
          ir6[3] = (v1575_data + (v1557_data * v1573_data));
          float v1578_data = s2[62];
          float v1580_data = ir6[4];
          ir6[4] = (v1580_data + (v1557_data * v1578_data));
          float v1583_data = s2[75];
          float v1585_data = ir6[5];
          ir6[5] = (v1585_data + (v1557_data * v1583_data));
          float v1588_data = s2[88];
          float v1590_data = ir6[6];
          ir6[6] = (v1590_data + (v1557_data * v1588_data));
          float v1593_data = s2[101];
          float v1595_data = ir6[7];
          ir6[7] = (v1595_data + (v1557_data * v1593_data));
          float v1598_data = s2[114];
          float v1600_data = ir6[8];
          ir6[8] = (v1600_data + (v1557_data * v1598_data));
          float v1603_data = s2[127];
          float v1605_data = ir6[9];
          ir6[9] = (v1605_data + (v1557_data * v1603_data));
          float v1608_data = s2[140];
          float v1610_data = ir6[10];
          ir6[10] = (v1610_data + (v1557_data * v1608_data));
          float v1613_data = s2[153];
          float v1615_data = ir6[11];
          ir6[11] = (v1615_data + (v1557_data * v1613_data));
          float v1618_data = s2[166];
          float v1620_data = ir6[12];
          ir6[12] = (v1620_data + (v1557_data * v1618_data));
          float v1622_data = r5[11];
          float v1623_data = s2[11];
          float v1625_data = ir6[0];
          ir6[0] = (v1625_data + (v1622_data * v1623_data));
          float v1628_data = s2[24];
          float v1630_data = ir6[1];
          ir6[1] = (v1630_data + (v1622_data * v1628_data));
          float v1633_data = s2[37];
          float v1635_data = ir6[2];
          ir6[2] = (v1635_data + (v1622_data * v1633_data));
          float v1638_data = s2[50];
          float v1640_data = ir6[3];
          ir6[3] = (v1640_data + (v1622_data * v1638_data));
          float v1643_data = s2[63];
          float v1645_data = ir6[4];
          ir6[4] = (v1645_data + (v1622_data * v1643_data));
          float v1648_data = s2[76];
          float v1650_data = ir6[5];
          ir6[5] = (v1650_data + (v1622_data * v1648_data));
          float v1653_data = s2[89];
          float v1655_data = ir6[6];
          ir6[6] = (v1655_data + (v1622_data * v1653_data));
          float v1658_data = s2[102];
          float v1660_data = ir6[7];
          ir6[7] = (v1660_data + (v1622_data * v1658_data));
          float v1663_data = s2[115];
          float v1665_data = ir6[8];
          ir6[8] = (v1665_data + (v1622_data * v1663_data));
          float v1668_data = s2[128];
          float v1670_data = ir6[9];
          ir6[9] = (v1670_data + (v1622_data * v1668_data));
          float v1673_data = s2[141];
          float v1675_data = ir6[10];
          ir6[10] = (v1675_data + (v1622_data * v1673_data));
          float v1678_data = s2[154];
          float v1680_data = ir6[11];
          ir6[11] = (v1680_data + (v1622_data * v1678_data));
          float v1683_data = s2[167];
          float v1685_data = ir6[12];
          ir6[12] = (v1685_data + (v1622_data * v1683_data));
          float v1687_data = r5[12];
          float v1688_data = s2[12];
          float v1690_data = ir6[0];
          ir6[0] = (v1690_data + (v1687_data * v1688_data));
          float v1693_data = s2[25];
          float v1695_data = ir6[1];
          ir6[1] = (v1695_data + (v1687_data * v1693_data));
          float v1698_data = s2[38];
          float v1700_data = ir6[2];
          ir6[2] = (v1700_data + (v1687_data * v1698_data));
          float v1703_data = s2[51];
          float v1705_data = ir6[3];
          ir6[3] = (v1705_data + (v1687_data * v1703_data));
          float v1708_data = s2[64];
          float v1710_data = ir6[4];
          ir6[4] = (v1710_data + (v1687_data * v1708_data));
          float v1713_data = s2[77];
          float v1715_data = ir6[5];
          ir6[5] = (v1715_data + (v1687_data * v1713_data));
          float v1718_data = s2[90];
          float v1720_data = ir6[6];
          ir6[6] = (v1720_data + (v1687_data * v1718_data));
          float v1723_data = s2[103];
          float v1725_data = ir6[7];
          ir6[7] = (v1725_data + (v1687_data * v1723_data));
          float v1728_data = s2[116];
          float v1730_data = ir6[8];
          ir6[8] = (v1730_data + (v1687_data * v1728_data));
          float v1733_data = s2[129];
          float v1735_data = ir6[9];
          ir6[9] = (v1735_data + (v1687_data * v1733_data));
          float v1738_data = s2[142];
          float v1740_data = ir6[10];
          ir6[10] = (v1740_data + (v1687_data * v1738_data));
          float v1743_data = s2[155];
          float v1745_data = ir6[11];
          ir6[11] = (v1745_data + (v1687_data * v1743_data));
          float v1748_data = s2[168];
          float v1750_data = ir6[12];
          ir6[12] = (v1750_data + (v1687_data * v1748_data));
          // r6 = ir6
          #pragma unroll
          for (int32_t v1752_n0 = 0; v1752_n0 < 1; ++v1752_n0) {
            #pragma unroll
            for (int32_t v1753_n1 = 0; v1753_n1 < 13; ++v1753_n1) {
              int32_t v1754_a = v1752_n0 + v1753_n1;
              float v1755_data = ir6[v1754_a];
              r6[v1754_a] = v1755_data;
            }
          }
          // glb_m3 = store{r>g}(r6);
          #pragma unroll
          for (int32_t v1756_i0 = 0; v1756_i0 < 1; ++v1756_i0) {
            int32_t v1761_lead = v25_lead + (v1756_i0 * 32);
            #pragma unroll
            for (int32_t v1757_i1 = 0; v1757_i1 < 13; ++v1757_i1) {
              float v1759_data = r6[(v1756_i0 + v1757_i1)];
              glb_m3[(v1761_lead + (v1757_i1 * 32))] = v1759_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

