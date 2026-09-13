// === base name ===
kernel_3ac6cc1d380bd0e1

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_3ac6cc1d380bd0e1 = {{32, 4, 1}, 32, 32, 1, 4, 3072, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_3ac6cc1d380bd0e1(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_3ac6cc1d380bd0e1(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_3ac6cc1d380bd0e1(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_3ac6cc1d380bd0e1, block.x * block.y * block.z, 768 * sizeof(float));
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
void launcher_kernel_3ac6cc1d380bd0e1(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_3ac6cc1d380bd0e1(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_3ac6cc1d380bd0e1, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_3ac6cc1d380bd0e1<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_3ac6cc1d380bd0e1(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
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
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":768}],"shared_bytes":3072,"shared_elements":768,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[32,13]],"name":"m0","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[32,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,12],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[12,13]],"name":"m2","ordered":false,"parts":1,"shape":[12,13],"variant":false},{"addressing":"strided","alias":"O","bbox":[[0,0],[32,13]],"name":"m3","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[13,13]],"name":"m4","ordered":false,"parts":1,"shape":[13,13],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,12]},{"addressing":"strided","bbox":[[0,0],[12,13]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[12,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,1]],"is_tmp":false,"name":"m0","offset":[0,4],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,1]],"is_tmp":true,"name":"t0","offset":[0,4],"shape":[32,13]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[0,0],[13,13]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[192 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[192];
      float * __restrict__ s1 = &localShrMem0[0];
      float * __restrict__ s2 = &localShrMem0[0];
      for (size_t v6_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v6_batchId0 < numElements0; v6_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v7_ahead1 = v6_batchId0 + (gridDim.x * blockDim.y);
        size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 416 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v6_batchId0 * 384 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v6_batchId0 * 156 + 0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[v6_batchId0 * 416 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[v6_batchId0 * 169 + 0 + m4_extraOffset];
          float r0[13]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v22_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
            int32_t v26_lead = v22_lead + (v23_i0 * 32);
            #pragma unroll
            for (int32_t v24_i1 = 0; v24_i1 < 13; ++v24_i1) {
              float v29_data = glb_m0[(v26_lead + (v24_i1 * 32))];
              r0[(v23_i0 + v24_i1)] = v29_data;
            }
          }
          float r2[12]{};
          // r2 = load{g>r}(glb_m1);
          #pragma unroll
          for (int32_t v32_i0 = 0; v32_i0 < 1; ++v32_i0) {
            int32_t v35_lead = v22_lead + (v32_i0 * 32);
            #pragma unroll
            for (int32_t v33_i1 = 0; v33_i1 < 12; ++v33_i1) {
              float v38_data = __ldcg(&glb_m1[(v35_lead + (v33_i1 * 32))]);
              r2[(v32_i0 + v33_i1)] = v38_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[13]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 13)] []
          float v41_data = r0[0];
          float v42_data = r1[0];
          r1[0] = (v42_data + v41_data);
          float v44_data = r0[1];
          float v45_data = r1[1];
          r1[1] = (v45_data + v44_data);
          float v47_data = r0[2];
          float v48_data = r1[2];
          r1[2] = (v48_data + v47_data);
          float v50_data = r0[3];
          float v51_data = r1[3];
          r1[3] = (v51_data + v50_data);
          float v53_data = r0[4];
          float v54_data = r1[4];
          r1[4] = (v54_data + v53_data);
          float v56_data = r0[5];
          float v57_data = r1[5];
          r1[5] = (v57_data + v56_data);
          float v59_data = r0[6];
          float v60_data = r1[6];
          r1[6] = (v60_data + v59_data);
          float v62_data = r0[7];
          float v63_data = r1[7];
          r1[7] = (v63_data + v62_data);
          float v65_data = r0[8];
          float v66_data = r1[8];
          r1[8] = (v66_data + v65_data);
          float v68_data = r0[9];
          float v69_data = r1[9];
          r1[9] = (v69_data + v68_data);
          float v71_data = r0[10];
          float v72_data = r1[10];
          r1[10] = (v72_data + v71_data);
          float v74_data = r0[11];
          float v75_data = r1[11];
          r1[11] = (v75_data + v74_data);
          float v77_data = r0[12];
          float v78_data = r1[12];
          r1[12] = (v78_data + v77_data);
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
          __syncwarp();
          // r3 = +(r2 * s1) + name: r1, type: SymbolType.Register, lead: [0]
          // [(0, 32), (0, 13)] [(0, 12)]
          float ir3[13]{};
          float v87_data = r2[0];
          float v88_data = s1[0];
          float v90_data = ir3[0];
          ir3[0] = (v90_data + (v87_data * v88_data));
          float v93_data = s1[12];
          float v95_data = ir3[1];
          ir3[1] = (v95_data + (v87_data * v93_data));
          float v98_data = s1[24];
          float v100_data = ir3[2];
          ir3[2] = (v100_data + (v87_data * v98_data));
          float v103_data = s1[36];
          float v105_data = ir3[3];
          ir3[3] = (v105_data + (v87_data * v103_data));
          float v108_data = s1[48];
          float v110_data = ir3[4];
          ir3[4] = (v110_data + (v87_data * v108_data));
          float v113_data = s1[60];
          float v115_data = ir3[5];
          ir3[5] = (v115_data + (v87_data * v113_data));
          float v118_data = s1[72];
          float v120_data = ir3[6];
          ir3[6] = (v120_data + (v87_data * v118_data));
          float v123_data = s1[84];
          float v125_data = ir3[7];
          ir3[7] = (v125_data + (v87_data * v123_data));
          float v128_data = s1[96];
          float v130_data = ir3[8];
          ir3[8] = (v130_data + (v87_data * v128_data));
          float v133_data = s1[108];
          float v135_data = ir3[9];
          ir3[9] = (v135_data + (v87_data * v133_data));
          float v138_data = s1[120];
          float v140_data = ir3[10];
          ir3[10] = (v140_data + (v87_data * v138_data));
          float v143_data = s1[132];
          float v145_data = ir3[11];
          ir3[11] = (v145_data + (v87_data * v143_data));
          float v148_data = s1[144];
          float v150_data = ir3[12];
          ir3[12] = (v150_data + (v87_data * v148_data));
          float v152_data = r2[1];
          float v153_data = s1[1];
          float v155_data = ir3[0];
          ir3[0] = (v155_data + (v152_data * v153_data));
          float v158_data = s1[13];
          float v160_data = ir3[1];
          ir3[1] = (v160_data + (v152_data * v158_data));
          float v163_data = s1[25];
          float v165_data = ir3[2];
          ir3[2] = (v165_data + (v152_data * v163_data));
          float v168_data = s1[37];
          float v170_data = ir3[3];
          ir3[3] = (v170_data + (v152_data * v168_data));
          float v173_data = s1[49];
          float v175_data = ir3[4];
          ir3[4] = (v175_data + (v152_data * v173_data));
          float v178_data = s1[61];
          float v180_data = ir3[5];
          ir3[5] = (v180_data + (v152_data * v178_data));
          float v183_data = s1[73];
          float v185_data = ir3[6];
          ir3[6] = (v185_data + (v152_data * v183_data));
          float v188_data = s1[85];
          float v190_data = ir3[7];
          ir3[7] = (v190_data + (v152_data * v188_data));
          float v193_data = s1[97];
          float v195_data = ir3[8];
          ir3[8] = (v195_data + (v152_data * v193_data));
          float v198_data = s1[109];
          float v200_data = ir3[9];
          ir3[9] = (v200_data + (v152_data * v198_data));
          float v203_data = s1[121];
          float v205_data = ir3[10];
          ir3[10] = (v205_data + (v152_data * v203_data));
          float v208_data = s1[133];
          float v210_data = ir3[11];
          ir3[11] = (v210_data + (v152_data * v208_data));
          float v213_data = s1[145];
          float v215_data = ir3[12];
          ir3[12] = (v215_data + (v152_data * v213_data));
          float v217_data = r2[2];
          float v218_data = s1[2];
          float v220_data = ir3[0];
          ir3[0] = (v220_data + (v217_data * v218_data));
          float v223_data = s1[14];
          float v225_data = ir3[1];
          ir3[1] = (v225_data + (v217_data * v223_data));
          float v228_data = s1[26];
          float v230_data = ir3[2];
          ir3[2] = (v230_data + (v217_data * v228_data));
          float v233_data = s1[38];
          float v235_data = ir3[3];
          ir3[3] = (v235_data + (v217_data * v233_data));
          float v238_data = s1[50];
          float v240_data = ir3[4];
          ir3[4] = (v240_data + (v217_data * v238_data));
          float v243_data = s1[62];
          float v245_data = ir3[5];
          ir3[5] = (v245_data + (v217_data * v243_data));
          float v248_data = s1[74];
          float v250_data = ir3[6];
          ir3[6] = (v250_data + (v217_data * v248_data));
          float v253_data = s1[86];
          float v255_data = ir3[7];
          ir3[7] = (v255_data + (v217_data * v253_data));
          float v258_data = s1[98];
          float v260_data = ir3[8];
          ir3[8] = (v260_data + (v217_data * v258_data));
          float v263_data = s1[110];
          float v265_data = ir3[9];
          ir3[9] = (v265_data + (v217_data * v263_data));
          float v268_data = s1[122];
          float v270_data = ir3[10];
          ir3[10] = (v270_data + (v217_data * v268_data));
          float v273_data = s1[134];
          float v275_data = ir3[11];
          ir3[11] = (v275_data + (v217_data * v273_data));
          float v278_data = s1[146];
          float v280_data = ir3[12];
          ir3[12] = (v280_data + (v217_data * v278_data));
          float v282_data = r2[3];
          float v283_data = s1[3];
          float v285_data = ir3[0];
          ir3[0] = (v285_data + (v282_data * v283_data));
          float v288_data = s1[15];
          float v290_data = ir3[1];
          ir3[1] = (v290_data + (v282_data * v288_data));
          float v293_data = s1[27];
          float v295_data = ir3[2];
          ir3[2] = (v295_data + (v282_data * v293_data));
          float v298_data = s1[39];
          float v300_data = ir3[3];
          ir3[3] = (v300_data + (v282_data * v298_data));
          float v303_data = s1[51];
          float v305_data = ir3[4];
          ir3[4] = (v305_data + (v282_data * v303_data));
          float v308_data = s1[63];
          float v310_data = ir3[5];
          ir3[5] = (v310_data + (v282_data * v308_data));
          float v313_data = s1[75];
          float v315_data = ir3[6];
          ir3[6] = (v315_data + (v282_data * v313_data));
          float v318_data = s1[87];
          float v320_data = ir3[7];
          ir3[7] = (v320_data + (v282_data * v318_data));
          float v323_data = s1[99];
          float v325_data = ir3[8];
          ir3[8] = (v325_data + (v282_data * v323_data));
          float v328_data = s1[111];
          float v330_data = ir3[9];
          ir3[9] = (v330_data + (v282_data * v328_data));
          float v333_data = s1[123];
          float v335_data = ir3[10];
          ir3[10] = (v335_data + (v282_data * v333_data));
          float v338_data = s1[135];
          float v340_data = ir3[11];
          ir3[11] = (v340_data + (v282_data * v338_data));
          float v343_data = s1[147];
          float v345_data = ir3[12];
          ir3[12] = (v345_data + (v282_data * v343_data));
          float v347_data = r2[4];
          float v348_data = s1[4];
          float v350_data = ir3[0];
          ir3[0] = (v350_data + (v347_data * v348_data));
          float v353_data = s1[16];
          float v355_data = ir3[1];
          ir3[1] = (v355_data + (v347_data * v353_data));
          float v358_data = s1[28];
          float v360_data = ir3[2];
          ir3[2] = (v360_data + (v347_data * v358_data));
          float v363_data = s1[40];
          float v365_data = ir3[3];
          ir3[3] = (v365_data + (v347_data * v363_data));
          float v368_data = s1[52];
          float v370_data = ir3[4];
          ir3[4] = (v370_data + (v347_data * v368_data));
          float v373_data = s1[64];
          float v375_data = ir3[5];
          ir3[5] = (v375_data + (v347_data * v373_data));
          float v378_data = s1[76];
          float v380_data = ir3[6];
          ir3[6] = (v380_data + (v347_data * v378_data));
          float v383_data = s1[88];
          float v385_data = ir3[7];
          ir3[7] = (v385_data + (v347_data * v383_data));
          float v388_data = s1[100];
          float v390_data = ir3[8];
          ir3[8] = (v390_data + (v347_data * v388_data));
          float v393_data = s1[112];
          float v395_data = ir3[9];
          ir3[9] = (v395_data + (v347_data * v393_data));
          float v398_data = s1[124];
          float v400_data = ir3[10];
          ir3[10] = (v400_data + (v347_data * v398_data));
          float v403_data = s1[136];
          float v405_data = ir3[11];
          ir3[11] = (v405_data + (v347_data * v403_data));
          float v408_data = s1[148];
          float v410_data = ir3[12];
          ir3[12] = (v410_data + (v347_data * v408_data));
          float v412_data = r2[5];
          float v413_data = s1[5];
          float v415_data = ir3[0];
          ir3[0] = (v415_data + (v412_data * v413_data));
          float v418_data = s1[17];
          float v420_data = ir3[1];
          ir3[1] = (v420_data + (v412_data * v418_data));
          float v423_data = s1[29];
          float v425_data = ir3[2];
          ir3[2] = (v425_data + (v412_data * v423_data));
          float v428_data = s1[41];
          float v430_data = ir3[3];
          ir3[3] = (v430_data + (v412_data * v428_data));
          float v433_data = s1[53];
          float v435_data = ir3[4];
          ir3[4] = (v435_data + (v412_data * v433_data));
          float v438_data = s1[65];
          float v440_data = ir3[5];
          ir3[5] = (v440_data + (v412_data * v438_data));
          float v443_data = s1[77];
          float v445_data = ir3[6];
          ir3[6] = (v445_data + (v412_data * v443_data));
          float v448_data = s1[89];
          float v450_data = ir3[7];
          ir3[7] = (v450_data + (v412_data * v448_data));
          float v453_data = s1[101];
          float v455_data = ir3[8];
          ir3[8] = (v455_data + (v412_data * v453_data));
          float v458_data = s1[113];
          float v460_data = ir3[9];
          ir3[9] = (v460_data + (v412_data * v458_data));
          float v463_data = s1[125];
          float v465_data = ir3[10];
          ir3[10] = (v465_data + (v412_data * v463_data));
          float v468_data = s1[137];
          float v470_data = ir3[11];
          ir3[11] = (v470_data + (v412_data * v468_data));
          float v473_data = s1[149];
          float v475_data = ir3[12];
          ir3[12] = (v475_data + (v412_data * v473_data));
          float v477_data = r2[6];
          float v478_data = s1[6];
          float v480_data = ir3[0];
          ir3[0] = (v480_data + (v477_data * v478_data));
          float v483_data = s1[18];
          float v485_data = ir3[1];
          ir3[1] = (v485_data + (v477_data * v483_data));
          float v488_data = s1[30];
          float v490_data = ir3[2];
          ir3[2] = (v490_data + (v477_data * v488_data));
          float v493_data = s1[42];
          float v495_data = ir3[3];
          ir3[3] = (v495_data + (v477_data * v493_data));
          float v498_data = s1[54];
          float v500_data = ir3[4];
          ir3[4] = (v500_data + (v477_data * v498_data));
          float v503_data = s1[66];
          float v505_data = ir3[5];
          ir3[5] = (v505_data + (v477_data * v503_data));
          float v508_data = s1[78];
          float v510_data = ir3[6];
          ir3[6] = (v510_data + (v477_data * v508_data));
          float v513_data = s1[90];
          float v515_data = ir3[7];
          ir3[7] = (v515_data + (v477_data * v513_data));
          float v518_data = s1[102];
          float v520_data = ir3[8];
          ir3[8] = (v520_data + (v477_data * v518_data));
          float v523_data = s1[114];
          float v525_data = ir3[9];
          ir3[9] = (v525_data + (v477_data * v523_data));
          float v528_data = s1[126];
          float v530_data = ir3[10];
          ir3[10] = (v530_data + (v477_data * v528_data));
          float v533_data = s1[138];
          float v535_data = ir3[11];
          ir3[11] = (v535_data + (v477_data * v533_data));
          float v538_data = s1[150];
          float v540_data = ir3[12];
          ir3[12] = (v540_data + (v477_data * v538_data));
          float v542_data = r2[7];
          float v543_data = s1[7];
          float v545_data = ir3[0];
          ir3[0] = (v545_data + (v542_data * v543_data));
          float v548_data = s1[19];
          float v550_data = ir3[1];
          ir3[1] = (v550_data + (v542_data * v548_data));
          float v553_data = s1[31];
          float v555_data = ir3[2];
          ir3[2] = (v555_data + (v542_data * v553_data));
          float v558_data = s1[43];
          float v560_data = ir3[3];
          ir3[3] = (v560_data + (v542_data * v558_data));
          float v563_data = s1[55];
          float v565_data = ir3[4];
          ir3[4] = (v565_data + (v542_data * v563_data));
          float v568_data = s1[67];
          float v570_data = ir3[5];
          ir3[5] = (v570_data + (v542_data * v568_data));
          float v573_data = s1[79];
          float v575_data = ir3[6];
          ir3[6] = (v575_data + (v542_data * v573_data));
          float v578_data = s1[91];
          float v580_data = ir3[7];
          ir3[7] = (v580_data + (v542_data * v578_data));
          float v583_data = s1[103];
          float v585_data = ir3[8];
          ir3[8] = (v585_data + (v542_data * v583_data));
          float v588_data = s1[115];
          float v590_data = ir3[9];
          ir3[9] = (v590_data + (v542_data * v588_data));
          float v593_data = s1[127];
          float v595_data = ir3[10];
          ir3[10] = (v595_data + (v542_data * v593_data));
          float v598_data = s1[139];
          float v600_data = ir3[11];
          ir3[11] = (v600_data + (v542_data * v598_data));
          float v603_data = s1[151];
          float v605_data = ir3[12];
          ir3[12] = (v605_data + (v542_data * v603_data));
          float v607_data = r2[8];
          float v608_data = s1[8];
          float v610_data = ir3[0];
          ir3[0] = (v610_data + (v607_data * v608_data));
          float v613_data = s1[20];
          float v615_data = ir3[1];
          ir3[1] = (v615_data + (v607_data * v613_data));
          float v618_data = s1[32];
          float v620_data = ir3[2];
          ir3[2] = (v620_data + (v607_data * v618_data));
          float v623_data = s1[44];
          float v625_data = ir3[3];
          ir3[3] = (v625_data + (v607_data * v623_data));
          float v628_data = s1[56];
          float v630_data = ir3[4];
          ir3[4] = (v630_data + (v607_data * v628_data));
          float v633_data = s1[68];
          float v635_data = ir3[5];
          ir3[5] = (v635_data + (v607_data * v633_data));
          float v638_data = s1[80];
          float v640_data = ir3[6];
          ir3[6] = (v640_data + (v607_data * v638_data));
          float v643_data = s1[92];
          float v645_data = ir3[7];
          ir3[7] = (v645_data + (v607_data * v643_data));
          float v648_data = s1[104];
          float v650_data = ir3[8];
          ir3[8] = (v650_data + (v607_data * v648_data));
          float v653_data = s1[116];
          float v655_data = ir3[9];
          ir3[9] = (v655_data + (v607_data * v653_data));
          float v658_data = s1[128];
          float v660_data = ir3[10];
          ir3[10] = (v660_data + (v607_data * v658_data));
          float v663_data = s1[140];
          float v665_data = ir3[11];
          ir3[11] = (v665_data + (v607_data * v663_data));
          float v668_data = s1[152];
          float v670_data = ir3[12];
          ir3[12] = (v670_data + (v607_data * v668_data));
          float v672_data = r2[9];
          float v673_data = s1[9];
          float v675_data = ir3[0];
          ir3[0] = (v675_data + (v672_data * v673_data));
          float v678_data = s1[21];
          float v680_data = ir3[1];
          ir3[1] = (v680_data + (v672_data * v678_data));
          float v683_data = s1[33];
          float v685_data = ir3[2];
          ir3[2] = (v685_data + (v672_data * v683_data));
          float v688_data = s1[45];
          float v690_data = ir3[3];
          ir3[3] = (v690_data + (v672_data * v688_data));
          float v693_data = s1[57];
          float v695_data = ir3[4];
          ir3[4] = (v695_data + (v672_data * v693_data));
          float v698_data = s1[69];
          float v700_data = ir3[5];
          ir3[5] = (v700_data + (v672_data * v698_data));
          float v703_data = s1[81];
          float v705_data = ir3[6];
          ir3[6] = (v705_data + (v672_data * v703_data));
          float v708_data = s1[93];
          float v710_data = ir3[7];
          ir3[7] = (v710_data + (v672_data * v708_data));
          float v713_data = s1[105];
          float v715_data = ir3[8];
          ir3[8] = (v715_data + (v672_data * v713_data));
          float v718_data = s1[117];
          float v720_data = ir3[9];
          ir3[9] = (v720_data + (v672_data * v718_data));
          float v723_data = s1[129];
          float v725_data = ir3[10];
          ir3[10] = (v725_data + (v672_data * v723_data));
          float v728_data = s1[141];
          float v730_data = ir3[11];
          ir3[11] = (v730_data + (v672_data * v728_data));
          float v733_data = s1[153];
          float v735_data = ir3[12];
          ir3[12] = (v735_data + (v672_data * v733_data));
          float v737_data = r2[10];
          float v738_data = s1[10];
          float v740_data = ir3[0];
          ir3[0] = (v740_data + (v737_data * v738_data));
          float v743_data = s1[22];
          float v745_data = ir3[1];
          ir3[1] = (v745_data + (v737_data * v743_data));
          float v748_data = s1[34];
          float v750_data = ir3[2];
          ir3[2] = (v750_data + (v737_data * v748_data));
          float v753_data = s1[46];
          float v755_data = ir3[3];
          ir3[3] = (v755_data + (v737_data * v753_data));
          float v758_data = s1[58];
          float v760_data = ir3[4];
          ir3[4] = (v760_data + (v737_data * v758_data));
          float v763_data = s1[70];
          float v765_data = ir3[5];
          ir3[5] = (v765_data + (v737_data * v763_data));
          float v768_data = s1[82];
          float v770_data = ir3[6];
          ir3[6] = (v770_data + (v737_data * v768_data));
          float v773_data = s1[94];
          float v775_data = ir3[7];
          ir3[7] = (v775_data + (v737_data * v773_data));
          float v778_data = s1[106];
          float v780_data = ir3[8];
          ir3[8] = (v780_data + (v737_data * v778_data));
          float v783_data = s1[118];
          float v785_data = ir3[9];
          ir3[9] = (v785_data + (v737_data * v783_data));
          float v788_data = s1[130];
          float v790_data = ir3[10];
          ir3[10] = (v790_data + (v737_data * v788_data));
          float v793_data = s1[142];
          float v795_data = ir3[11];
          ir3[11] = (v795_data + (v737_data * v793_data));
          float v798_data = s1[154];
          float v800_data = ir3[12];
          ir3[12] = (v800_data + (v737_data * v798_data));
          float v802_data = r2[11];
          float v803_data = s1[11];
          float v805_data = ir3[0];
          ir3[0] = (v805_data + (v802_data * v803_data));
          float v808_data = s1[23];
          float v810_data = ir3[1];
          ir3[1] = (v810_data + (v802_data * v808_data));
          float v813_data = s1[35];
          float v815_data = ir3[2];
          ir3[2] = (v815_data + (v802_data * v813_data));
          float v818_data = s1[47];
          float v820_data = ir3[3];
          ir3[3] = (v820_data + (v802_data * v818_data));
          float v823_data = s1[59];
          float v825_data = ir3[4];
          ir3[4] = (v825_data + (v802_data * v823_data));
          float v828_data = s1[71];
          float v830_data = ir3[5];
          ir3[5] = (v830_data + (v802_data * v828_data));
          float v833_data = s1[83];
          float v835_data = ir3[6];
          ir3[6] = (v835_data + (v802_data * v833_data));
          float v838_data = s1[95];
          float v840_data = ir3[7];
          ir3[7] = (v840_data + (v802_data * v838_data));
          float v843_data = s1[107];
          float v845_data = ir3[8];
          ir3[8] = (v845_data + (v802_data * v843_data));
          float v848_data = s1[119];
          float v850_data = ir3[9];
          ir3[9] = (v850_data + (v802_data * v848_data));
          float v853_data = s1[131];
          float v855_data = ir3[10];
          ir3[10] = (v855_data + (v802_data * v853_data));
          float v858_data = s1[143];
          float v860_data = ir3[11];
          ir3[11] = (v860_data + (v802_data * v858_data));
          float v863_data = s1[155];
          float v865_data = ir3[12];
          ir3[12] = (v865_data + (v802_data * v863_data));
          #pragma unroll
          for (int32_t v867_n0 = 0; v867_n0 < 1; ++v867_n0) {
            #pragma unroll
            for (int32_t v868_n1 = 0; v868_n1 < 13; ++v868_n1) {
              int32_t v869_a = v867_n0 + v868_n1;
              float v870_data = ir3[v869_a];
              float v871_data = r1[v869_a];
              r3[v869_a] = (v871_data + v870_data);
            }
          }
          float r4[1]{};
          // r4 = +(r3) + None
          // [(0, 32), (0, 1)] []
          float ir4[1]{};
          float v875_data = r3[4];
          float v876_data = ir4[0];
          ir4[0] = (v876_data + v875_data);
          #pragma unroll
          for (int32_t v878_n0 = 0; v878_n0 < 1; ++v878_n0) {
            #pragma unroll
            for (int32_t v879_n1 = 0; v879_n1 < 1; ++v879_n1) {
              int32_t v880_a = v878_n0 + v879_n1;
              float v881_data = ir4[v880_a];
              r4[v880_a] = v881_data;
            }
          }
          // glb_m0 = store{r>g}(r4);
          #pragma unroll
          for (int32_t v882_i0 = 0; v882_i0 < 1; ++v882_i0) {
            int32_t v887_lead = v22_lead + (v882_i0 * 32);
            #pragma unroll
            for (int32_t v883_i1 = 0; v883_i1 < 1; ++v883_i1) {
              float v885_data = r4[(v882_i0 + v883_i1)];
              glb_m0[(v887_lead + ((v883_i1 + 4) * 32))] = v885_data;
            }
          }
          float r5[13]{};
          // r5 = load{g>r}(glb_m0);
          #pragma unroll
          for (int32_t v892_i0 = 0; v892_i0 < 1; ++v892_i0) {
            int32_t v895_lead = v22_lead + (v892_i0 * 32);
            #pragma unroll
            for (int32_t v893_i1 = 0; v893_i1 < 13; ++v893_i1) {
              float v898_data = glb_m0[(v895_lead + (v893_i1 * 32))];
              r5[(v892_i0 + v893_i1)] = v898_data;
            }
          }
          __syncwarp();
          // s2 = load{g>s}(glb_m4[0, 1])
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
          __syncwarp();
          // r6 = +(r5 * s2) + None
          // [(0, 32), (0, 13)] [(0, 13)]
          float ir6[13]{};
          float v904_data = r5[0];
          float v905_data = s2[0];
          float v907_data = ir6[0];
          ir6[0] = (v907_data + (v904_data * v905_data));
          float v910_data = s2[13];
          float v912_data = ir6[1];
          ir6[1] = (v912_data + (v904_data * v910_data));
          float v915_data = s2[26];
          float v917_data = ir6[2];
          ir6[2] = (v917_data + (v904_data * v915_data));
          float v920_data = s2[39];
          float v922_data = ir6[3];
          ir6[3] = (v922_data + (v904_data * v920_data));
          float v925_data = s2[52];
          float v927_data = ir6[4];
          ir6[4] = (v927_data + (v904_data * v925_data));
          float v930_data = s2[65];
          float v932_data = ir6[5];
          ir6[5] = (v932_data + (v904_data * v930_data));
          float v935_data = s2[78];
          float v937_data = ir6[6];
          ir6[6] = (v937_data + (v904_data * v935_data));
          float v940_data = s2[91];
          float v942_data = ir6[7];
          ir6[7] = (v942_data + (v904_data * v940_data));
          float v945_data = s2[104];
          float v947_data = ir6[8];
          ir6[8] = (v947_data + (v904_data * v945_data));
          float v950_data = s2[117];
          float v952_data = ir6[9];
          ir6[9] = (v952_data + (v904_data * v950_data));
          float v955_data = s2[130];
          float v957_data = ir6[10];
          ir6[10] = (v957_data + (v904_data * v955_data));
          float v960_data = s2[143];
          float v962_data = ir6[11];
          ir6[11] = (v962_data + (v904_data * v960_data));
          float v965_data = s2[156];
          float v967_data = ir6[12];
          ir6[12] = (v967_data + (v904_data * v965_data));
          float v969_data = r5[1];
          float v970_data = s2[1];
          float v972_data = ir6[0];
          ir6[0] = (v972_data + (v969_data * v970_data));
          float v975_data = s2[14];
          float v977_data = ir6[1];
          ir6[1] = (v977_data + (v969_data * v975_data));
          float v980_data = s2[27];
          float v982_data = ir6[2];
          ir6[2] = (v982_data + (v969_data * v980_data));
          float v985_data = s2[40];
          float v987_data = ir6[3];
          ir6[3] = (v987_data + (v969_data * v985_data));
          float v990_data = s2[53];
          float v992_data = ir6[4];
          ir6[4] = (v992_data + (v969_data * v990_data));
          float v995_data = s2[66];
          float v997_data = ir6[5];
          ir6[5] = (v997_data + (v969_data * v995_data));
          float v1000_data = s2[79];
          float v1002_data = ir6[6];
          ir6[6] = (v1002_data + (v969_data * v1000_data));
          float v1005_data = s2[92];
          float v1007_data = ir6[7];
          ir6[7] = (v1007_data + (v969_data * v1005_data));
          float v1010_data = s2[105];
          float v1012_data = ir6[8];
          ir6[8] = (v1012_data + (v969_data * v1010_data));
          float v1015_data = s2[118];
          float v1017_data = ir6[9];
          ir6[9] = (v1017_data + (v969_data * v1015_data));
          float v1020_data = s2[131];
          float v1022_data = ir6[10];
          ir6[10] = (v1022_data + (v969_data * v1020_data));
          float v1025_data = s2[144];
          float v1027_data = ir6[11];
          ir6[11] = (v1027_data + (v969_data * v1025_data));
          float v1030_data = s2[157];
          float v1032_data = ir6[12];
          ir6[12] = (v1032_data + (v969_data * v1030_data));
          float v1034_data = r5[2];
          float v1035_data = s2[2];
          float v1037_data = ir6[0];
          ir6[0] = (v1037_data + (v1034_data * v1035_data));
          float v1040_data = s2[15];
          float v1042_data = ir6[1];
          ir6[1] = (v1042_data + (v1034_data * v1040_data));
          float v1045_data = s2[28];
          float v1047_data = ir6[2];
          ir6[2] = (v1047_data + (v1034_data * v1045_data));
          float v1050_data = s2[41];
          float v1052_data = ir6[3];
          ir6[3] = (v1052_data + (v1034_data * v1050_data));
          float v1055_data = s2[54];
          float v1057_data = ir6[4];
          ir6[4] = (v1057_data + (v1034_data * v1055_data));
          float v1060_data = s2[67];
          float v1062_data = ir6[5];
          ir6[5] = (v1062_data + (v1034_data * v1060_data));
          float v1065_data = s2[80];
          float v1067_data = ir6[6];
          ir6[6] = (v1067_data + (v1034_data * v1065_data));
          float v1070_data = s2[93];
          float v1072_data = ir6[7];
          ir6[7] = (v1072_data + (v1034_data * v1070_data));
          float v1075_data = s2[106];
          float v1077_data = ir6[8];
          ir6[8] = (v1077_data + (v1034_data * v1075_data));
          float v1080_data = s2[119];
          float v1082_data = ir6[9];
          ir6[9] = (v1082_data + (v1034_data * v1080_data));
          float v1085_data = s2[132];
          float v1087_data = ir6[10];
          ir6[10] = (v1087_data + (v1034_data * v1085_data));
          float v1090_data = s2[145];
          float v1092_data = ir6[11];
          ir6[11] = (v1092_data + (v1034_data * v1090_data));
          float v1095_data = s2[158];
          float v1097_data = ir6[12];
          ir6[12] = (v1097_data + (v1034_data * v1095_data));
          float v1099_data = r5[3];
          float v1100_data = s2[3];
          float v1102_data = ir6[0];
          ir6[0] = (v1102_data + (v1099_data * v1100_data));
          float v1105_data = s2[16];
          float v1107_data = ir6[1];
          ir6[1] = (v1107_data + (v1099_data * v1105_data));
          float v1110_data = s2[29];
          float v1112_data = ir6[2];
          ir6[2] = (v1112_data + (v1099_data * v1110_data));
          float v1115_data = s2[42];
          float v1117_data = ir6[3];
          ir6[3] = (v1117_data + (v1099_data * v1115_data));
          float v1120_data = s2[55];
          float v1122_data = ir6[4];
          ir6[4] = (v1122_data + (v1099_data * v1120_data));
          float v1125_data = s2[68];
          float v1127_data = ir6[5];
          ir6[5] = (v1127_data + (v1099_data * v1125_data));
          float v1130_data = s2[81];
          float v1132_data = ir6[6];
          ir6[6] = (v1132_data + (v1099_data * v1130_data));
          float v1135_data = s2[94];
          float v1137_data = ir6[7];
          ir6[7] = (v1137_data + (v1099_data * v1135_data));
          float v1140_data = s2[107];
          float v1142_data = ir6[8];
          ir6[8] = (v1142_data + (v1099_data * v1140_data));
          float v1145_data = s2[120];
          float v1147_data = ir6[9];
          ir6[9] = (v1147_data + (v1099_data * v1145_data));
          float v1150_data = s2[133];
          float v1152_data = ir6[10];
          ir6[10] = (v1152_data + (v1099_data * v1150_data));
          float v1155_data = s2[146];
          float v1157_data = ir6[11];
          ir6[11] = (v1157_data + (v1099_data * v1155_data));
          float v1160_data = s2[159];
          float v1162_data = ir6[12];
          ir6[12] = (v1162_data + (v1099_data * v1160_data));
          float v1164_data = r5[4];
          float v1165_data = s2[4];
          float v1167_data = ir6[0];
          ir6[0] = (v1167_data + (v1164_data * v1165_data));
          float v1170_data = s2[17];
          float v1172_data = ir6[1];
          ir6[1] = (v1172_data + (v1164_data * v1170_data));
          float v1175_data = s2[30];
          float v1177_data = ir6[2];
          ir6[2] = (v1177_data + (v1164_data * v1175_data));
          float v1180_data = s2[43];
          float v1182_data = ir6[3];
          ir6[3] = (v1182_data + (v1164_data * v1180_data));
          float v1185_data = s2[56];
          float v1187_data = ir6[4];
          ir6[4] = (v1187_data + (v1164_data * v1185_data));
          float v1190_data = s2[69];
          float v1192_data = ir6[5];
          ir6[5] = (v1192_data + (v1164_data * v1190_data));
          float v1195_data = s2[82];
          float v1197_data = ir6[6];
          ir6[6] = (v1197_data + (v1164_data * v1195_data));
          float v1200_data = s2[95];
          float v1202_data = ir6[7];
          ir6[7] = (v1202_data + (v1164_data * v1200_data));
          float v1205_data = s2[108];
          float v1207_data = ir6[8];
          ir6[8] = (v1207_data + (v1164_data * v1205_data));
          float v1210_data = s2[121];
          float v1212_data = ir6[9];
          ir6[9] = (v1212_data + (v1164_data * v1210_data));
          float v1215_data = s2[134];
          float v1217_data = ir6[10];
          ir6[10] = (v1217_data + (v1164_data * v1215_data));
          float v1220_data = s2[147];
          float v1222_data = ir6[11];
          ir6[11] = (v1222_data + (v1164_data * v1220_data));
          float v1225_data = s2[160];
          float v1227_data = ir6[12];
          ir6[12] = (v1227_data + (v1164_data * v1225_data));
          float v1229_data = r5[5];
          float v1230_data = s2[5];
          float v1232_data = ir6[0];
          ir6[0] = (v1232_data + (v1229_data * v1230_data));
          float v1235_data = s2[18];
          float v1237_data = ir6[1];
          ir6[1] = (v1237_data + (v1229_data * v1235_data));
          float v1240_data = s2[31];
          float v1242_data = ir6[2];
          ir6[2] = (v1242_data + (v1229_data * v1240_data));
          float v1245_data = s2[44];
          float v1247_data = ir6[3];
          ir6[3] = (v1247_data + (v1229_data * v1245_data));
          float v1250_data = s2[57];
          float v1252_data = ir6[4];
          ir6[4] = (v1252_data + (v1229_data * v1250_data));
          float v1255_data = s2[70];
          float v1257_data = ir6[5];
          ir6[5] = (v1257_data + (v1229_data * v1255_data));
          float v1260_data = s2[83];
          float v1262_data = ir6[6];
          ir6[6] = (v1262_data + (v1229_data * v1260_data));
          float v1265_data = s2[96];
          float v1267_data = ir6[7];
          ir6[7] = (v1267_data + (v1229_data * v1265_data));
          float v1270_data = s2[109];
          float v1272_data = ir6[8];
          ir6[8] = (v1272_data + (v1229_data * v1270_data));
          float v1275_data = s2[122];
          float v1277_data = ir6[9];
          ir6[9] = (v1277_data + (v1229_data * v1275_data));
          float v1280_data = s2[135];
          float v1282_data = ir6[10];
          ir6[10] = (v1282_data + (v1229_data * v1280_data));
          float v1285_data = s2[148];
          float v1287_data = ir6[11];
          ir6[11] = (v1287_data + (v1229_data * v1285_data));
          float v1290_data = s2[161];
          float v1292_data = ir6[12];
          ir6[12] = (v1292_data + (v1229_data * v1290_data));
          float v1294_data = r5[6];
          float v1295_data = s2[6];
          float v1297_data = ir6[0];
          ir6[0] = (v1297_data + (v1294_data * v1295_data));
          float v1300_data = s2[19];
          float v1302_data = ir6[1];
          ir6[1] = (v1302_data + (v1294_data * v1300_data));
          float v1305_data = s2[32];
          float v1307_data = ir6[2];
          ir6[2] = (v1307_data + (v1294_data * v1305_data));
          float v1310_data = s2[45];
          float v1312_data = ir6[3];
          ir6[3] = (v1312_data + (v1294_data * v1310_data));
          float v1315_data = s2[58];
          float v1317_data = ir6[4];
          ir6[4] = (v1317_data + (v1294_data * v1315_data));
          float v1320_data = s2[71];
          float v1322_data = ir6[5];
          ir6[5] = (v1322_data + (v1294_data * v1320_data));
          float v1325_data = s2[84];
          float v1327_data = ir6[6];
          ir6[6] = (v1327_data + (v1294_data * v1325_data));
          float v1330_data = s2[97];
          float v1332_data = ir6[7];
          ir6[7] = (v1332_data + (v1294_data * v1330_data));
          float v1335_data = s2[110];
          float v1337_data = ir6[8];
          ir6[8] = (v1337_data + (v1294_data * v1335_data));
          float v1340_data = s2[123];
          float v1342_data = ir6[9];
          ir6[9] = (v1342_data + (v1294_data * v1340_data));
          float v1345_data = s2[136];
          float v1347_data = ir6[10];
          ir6[10] = (v1347_data + (v1294_data * v1345_data));
          float v1350_data = s2[149];
          float v1352_data = ir6[11];
          ir6[11] = (v1352_data + (v1294_data * v1350_data));
          float v1355_data = s2[162];
          float v1357_data = ir6[12];
          ir6[12] = (v1357_data + (v1294_data * v1355_data));
          float v1359_data = r5[7];
          float v1360_data = s2[7];
          float v1362_data = ir6[0];
          ir6[0] = (v1362_data + (v1359_data * v1360_data));
          float v1365_data = s2[20];
          float v1367_data = ir6[1];
          ir6[1] = (v1367_data + (v1359_data * v1365_data));
          float v1370_data = s2[33];
          float v1372_data = ir6[2];
          ir6[2] = (v1372_data + (v1359_data * v1370_data));
          float v1375_data = s2[46];
          float v1377_data = ir6[3];
          ir6[3] = (v1377_data + (v1359_data * v1375_data));
          float v1380_data = s2[59];
          float v1382_data = ir6[4];
          ir6[4] = (v1382_data + (v1359_data * v1380_data));
          float v1385_data = s2[72];
          float v1387_data = ir6[5];
          ir6[5] = (v1387_data + (v1359_data * v1385_data));
          float v1390_data = s2[85];
          float v1392_data = ir6[6];
          ir6[6] = (v1392_data + (v1359_data * v1390_data));
          float v1395_data = s2[98];
          float v1397_data = ir6[7];
          ir6[7] = (v1397_data + (v1359_data * v1395_data));
          float v1400_data = s2[111];
          float v1402_data = ir6[8];
          ir6[8] = (v1402_data + (v1359_data * v1400_data));
          float v1405_data = s2[124];
          float v1407_data = ir6[9];
          ir6[9] = (v1407_data + (v1359_data * v1405_data));
          float v1410_data = s2[137];
          float v1412_data = ir6[10];
          ir6[10] = (v1412_data + (v1359_data * v1410_data));
          float v1415_data = s2[150];
          float v1417_data = ir6[11];
          ir6[11] = (v1417_data + (v1359_data * v1415_data));
          float v1420_data = s2[163];
          float v1422_data = ir6[12];
          ir6[12] = (v1422_data + (v1359_data * v1420_data));
          float v1424_data = r5[8];
          float v1425_data = s2[8];
          float v1427_data = ir6[0];
          ir6[0] = (v1427_data + (v1424_data * v1425_data));
          float v1430_data = s2[21];
          float v1432_data = ir6[1];
          ir6[1] = (v1432_data + (v1424_data * v1430_data));
          float v1435_data = s2[34];
          float v1437_data = ir6[2];
          ir6[2] = (v1437_data + (v1424_data * v1435_data));
          float v1440_data = s2[47];
          float v1442_data = ir6[3];
          ir6[3] = (v1442_data + (v1424_data * v1440_data));
          float v1445_data = s2[60];
          float v1447_data = ir6[4];
          ir6[4] = (v1447_data + (v1424_data * v1445_data));
          float v1450_data = s2[73];
          float v1452_data = ir6[5];
          ir6[5] = (v1452_data + (v1424_data * v1450_data));
          float v1455_data = s2[86];
          float v1457_data = ir6[6];
          ir6[6] = (v1457_data + (v1424_data * v1455_data));
          float v1460_data = s2[99];
          float v1462_data = ir6[7];
          ir6[7] = (v1462_data + (v1424_data * v1460_data));
          float v1465_data = s2[112];
          float v1467_data = ir6[8];
          ir6[8] = (v1467_data + (v1424_data * v1465_data));
          float v1470_data = s2[125];
          float v1472_data = ir6[9];
          ir6[9] = (v1472_data + (v1424_data * v1470_data));
          float v1475_data = s2[138];
          float v1477_data = ir6[10];
          ir6[10] = (v1477_data + (v1424_data * v1475_data));
          float v1480_data = s2[151];
          float v1482_data = ir6[11];
          ir6[11] = (v1482_data + (v1424_data * v1480_data));
          float v1485_data = s2[164];
          float v1487_data = ir6[12];
          ir6[12] = (v1487_data + (v1424_data * v1485_data));
          float v1489_data = r5[9];
          float v1490_data = s2[9];
          float v1492_data = ir6[0];
          ir6[0] = (v1492_data + (v1489_data * v1490_data));
          float v1495_data = s2[22];
          float v1497_data = ir6[1];
          ir6[1] = (v1497_data + (v1489_data * v1495_data));
          float v1500_data = s2[35];
          float v1502_data = ir6[2];
          ir6[2] = (v1502_data + (v1489_data * v1500_data));
          float v1505_data = s2[48];
          float v1507_data = ir6[3];
          ir6[3] = (v1507_data + (v1489_data * v1505_data));
          float v1510_data = s2[61];
          float v1512_data = ir6[4];
          ir6[4] = (v1512_data + (v1489_data * v1510_data));
          float v1515_data = s2[74];
          float v1517_data = ir6[5];
          ir6[5] = (v1517_data + (v1489_data * v1515_data));
          float v1520_data = s2[87];
          float v1522_data = ir6[6];
          ir6[6] = (v1522_data + (v1489_data * v1520_data));
          float v1525_data = s2[100];
          float v1527_data = ir6[7];
          ir6[7] = (v1527_data + (v1489_data * v1525_data));
          float v1530_data = s2[113];
          float v1532_data = ir6[8];
          ir6[8] = (v1532_data + (v1489_data * v1530_data));
          float v1535_data = s2[126];
          float v1537_data = ir6[9];
          ir6[9] = (v1537_data + (v1489_data * v1535_data));
          float v1540_data = s2[139];
          float v1542_data = ir6[10];
          ir6[10] = (v1542_data + (v1489_data * v1540_data));
          float v1545_data = s2[152];
          float v1547_data = ir6[11];
          ir6[11] = (v1547_data + (v1489_data * v1545_data));
          float v1550_data = s2[165];
          float v1552_data = ir6[12];
          ir6[12] = (v1552_data + (v1489_data * v1550_data));
          float v1554_data = r5[10];
          float v1555_data = s2[10];
          float v1557_data = ir6[0];
          ir6[0] = (v1557_data + (v1554_data * v1555_data));
          float v1560_data = s2[23];
          float v1562_data = ir6[1];
          ir6[1] = (v1562_data + (v1554_data * v1560_data));
          float v1565_data = s2[36];
          float v1567_data = ir6[2];
          ir6[2] = (v1567_data + (v1554_data * v1565_data));
          float v1570_data = s2[49];
          float v1572_data = ir6[3];
          ir6[3] = (v1572_data + (v1554_data * v1570_data));
          float v1575_data = s2[62];
          float v1577_data = ir6[4];
          ir6[4] = (v1577_data + (v1554_data * v1575_data));
          float v1580_data = s2[75];
          float v1582_data = ir6[5];
          ir6[5] = (v1582_data + (v1554_data * v1580_data));
          float v1585_data = s2[88];
          float v1587_data = ir6[6];
          ir6[6] = (v1587_data + (v1554_data * v1585_data));
          float v1590_data = s2[101];
          float v1592_data = ir6[7];
          ir6[7] = (v1592_data + (v1554_data * v1590_data));
          float v1595_data = s2[114];
          float v1597_data = ir6[8];
          ir6[8] = (v1597_data + (v1554_data * v1595_data));
          float v1600_data = s2[127];
          float v1602_data = ir6[9];
          ir6[9] = (v1602_data + (v1554_data * v1600_data));
          float v1605_data = s2[140];
          float v1607_data = ir6[10];
          ir6[10] = (v1607_data + (v1554_data * v1605_data));
          float v1610_data = s2[153];
          float v1612_data = ir6[11];
          ir6[11] = (v1612_data + (v1554_data * v1610_data));
          float v1615_data = s2[166];
          float v1617_data = ir6[12];
          ir6[12] = (v1617_data + (v1554_data * v1615_data));
          float v1619_data = r5[11];
          float v1620_data = s2[11];
          float v1622_data = ir6[0];
          ir6[0] = (v1622_data + (v1619_data * v1620_data));
          float v1625_data = s2[24];
          float v1627_data = ir6[1];
          ir6[1] = (v1627_data + (v1619_data * v1625_data));
          float v1630_data = s2[37];
          float v1632_data = ir6[2];
          ir6[2] = (v1632_data + (v1619_data * v1630_data));
          float v1635_data = s2[50];
          float v1637_data = ir6[3];
          ir6[3] = (v1637_data + (v1619_data * v1635_data));
          float v1640_data = s2[63];
          float v1642_data = ir6[4];
          ir6[4] = (v1642_data + (v1619_data * v1640_data));
          float v1645_data = s2[76];
          float v1647_data = ir6[5];
          ir6[5] = (v1647_data + (v1619_data * v1645_data));
          float v1650_data = s2[89];
          float v1652_data = ir6[6];
          ir6[6] = (v1652_data + (v1619_data * v1650_data));
          float v1655_data = s2[102];
          float v1657_data = ir6[7];
          ir6[7] = (v1657_data + (v1619_data * v1655_data));
          float v1660_data = s2[115];
          float v1662_data = ir6[8];
          ir6[8] = (v1662_data + (v1619_data * v1660_data));
          float v1665_data = s2[128];
          float v1667_data = ir6[9];
          ir6[9] = (v1667_data + (v1619_data * v1665_data));
          float v1670_data = s2[141];
          float v1672_data = ir6[10];
          ir6[10] = (v1672_data + (v1619_data * v1670_data));
          float v1675_data = s2[154];
          float v1677_data = ir6[11];
          ir6[11] = (v1677_data + (v1619_data * v1675_data));
          float v1680_data = s2[167];
          float v1682_data = ir6[12];
          ir6[12] = (v1682_data + (v1619_data * v1680_data));
          float v1684_data = r5[12];
          float v1685_data = s2[12];
          float v1687_data = ir6[0];
          ir6[0] = (v1687_data + (v1684_data * v1685_data));
          float v1690_data = s2[25];
          float v1692_data = ir6[1];
          ir6[1] = (v1692_data + (v1684_data * v1690_data));
          float v1695_data = s2[38];
          float v1697_data = ir6[2];
          ir6[2] = (v1697_data + (v1684_data * v1695_data));
          float v1700_data = s2[51];
          float v1702_data = ir6[3];
          ir6[3] = (v1702_data + (v1684_data * v1700_data));
          float v1705_data = s2[64];
          float v1707_data = ir6[4];
          ir6[4] = (v1707_data + (v1684_data * v1705_data));
          float v1710_data = s2[77];
          float v1712_data = ir6[5];
          ir6[5] = (v1712_data + (v1684_data * v1710_data));
          float v1715_data = s2[90];
          float v1717_data = ir6[6];
          ir6[6] = (v1717_data + (v1684_data * v1715_data));
          float v1720_data = s2[103];
          float v1722_data = ir6[7];
          ir6[7] = (v1722_data + (v1684_data * v1720_data));
          float v1725_data = s2[116];
          float v1727_data = ir6[8];
          ir6[8] = (v1727_data + (v1684_data * v1725_data));
          float v1730_data = s2[129];
          float v1732_data = ir6[9];
          ir6[9] = (v1732_data + (v1684_data * v1730_data));
          float v1735_data = s2[142];
          float v1737_data = ir6[10];
          ir6[10] = (v1737_data + (v1684_data * v1735_data));
          float v1740_data = s2[155];
          float v1742_data = ir6[11];
          ir6[11] = (v1742_data + (v1684_data * v1740_data));
          float v1745_data = s2[168];
          float v1747_data = ir6[12];
          ir6[12] = (v1747_data + (v1684_data * v1745_data));
          #pragma unroll
          for (int32_t v1749_n0 = 0; v1749_n0 < 1; ++v1749_n0) {
            #pragma unroll
            for (int32_t v1750_n1 = 0; v1750_n1 < 13; ++v1750_n1) {
              int32_t v1751_a = v1749_n0 + v1750_n1;
              float v1752_data = ir6[v1751_a];
              r6[v1751_a] = v1752_data;
            }
          }
          // glb_m3 = store{r>g}(r6);
          #pragma unroll
          for (int32_t v1753_i0 = 0; v1753_i0 < 1; ++v1753_i0) {
            int32_t v1758_lead = v22_lead + (v1753_i0 * 32);
            #pragma unroll
            for (int32_t v1754_i1 = 0; v1754_i1 < 13; ++v1754_i1) {
              float v1756_data = r6[(v1753_i0 + v1754_i1)];
              glb_m3[(v1758_lead + (v1754_i1 * 32))] = v1756_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

