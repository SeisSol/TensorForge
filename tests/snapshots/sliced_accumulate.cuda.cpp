// === base name ===
kernel_f3ebfa251de9d57d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_f3ebfa251de9d57d = {{32, 4, 1}, 32, 32, 1, 4, 3072, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_f3ebfa251de9d57d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_f3ebfa251de9d57d(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_f3ebfa251de9d57d(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_f3ebfa251de9d57d, block.x * block.y * block.z, 768 * sizeof(float));
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
void launcher_kernel_f3ebfa251de9d57d(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_f3ebfa251de9d57d(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_f3ebfa251de9d57d, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_f3ebfa251de9d57d<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, m6, m6_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_f3ebfa251de9d57d(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0) {
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
          // wait(r0 = load{g>r}(glb_m1););
          float r2[12]{};
          // r2 = load{g>r}(glb_m3);
          #pragma unroll
          for (int32_t v39_i0 = 0; v39_i0 < 1; ++v39_i0) {
            int32_t v42_lead = v28_lead + (v39_i0 * 32);
            #pragma unroll
            for (int32_t v40_i1 = 0; v40_i1 < 12; ++v40_i1) {
              float v45_data = __ldcg(&glb_m3[(v42_lead + (v40_i1 * 32))]);
              r2[(v39_i0 + v40_i1)] = v45_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[16]{};
          // ir1 = +(r0 * s0)
          // [(0, 32), (0, 16)] [(0, 12)]
          float ir1[16]{};
          float v49_data = r0[0];
          __syncwarp();
          float v50_data = s0[0];
          float v52_data = ir1[0];
          ir1[0] = (v52_data + (v49_data * v50_data));
          float v55_data = s0[12];
          float v57_data = ir1[1];
          ir1[1] = (v57_data + (v49_data * v55_data));
          float v60_data = s0[24];
          float v62_data = ir1[2];
          ir1[2] = (v62_data + (v49_data * v60_data));
          float v65_data = s0[36];
          float v67_data = ir1[3];
          ir1[3] = (v67_data + (v49_data * v65_data));
          float v70_data = s0[48];
          float v72_data = ir1[4];
          ir1[4] = (v72_data + (v49_data * v70_data));
          float v75_data = s0[60];
          float v77_data = ir1[5];
          ir1[5] = (v77_data + (v49_data * v75_data));
          float v80_data = s0[72];
          float v82_data = ir1[6];
          ir1[6] = (v82_data + (v49_data * v80_data));
          float v85_data = s0[84];
          float v87_data = ir1[7];
          ir1[7] = (v87_data + (v49_data * v85_data));
          float v90_data = s0[96];
          float v92_data = ir1[8];
          ir1[8] = (v92_data + (v49_data * v90_data));
          float v95_data = s0[108];
          float v97_data = ir1[9];
          ir1[9] = (v97_data + (v49_data * v95_data));
          float v100_data = s0[120];
          float v102_data = ir1[10];
          ir1[10] = (v102_data + (v49_data * v100_data));
          float v105_data = s0[132];
          float v107_data = ir1[11];
          ir1[11] = (v107_data + (v49_data * v105_data));
          float v110_data = s0[144];
          float v112_data = ir1[12];
          ir1[12] = (v112_data + (v49_data * v110_data));
          float v115_data = s0[156];
          float v117_data = ir1[13];
          ir1[13] = (v117_data + (v49_data * v115_data));
          float v120_data = s0[168];
          float v122_data = ir1[14];
          ir1[14] = (v122_data + (v49_data * v120_data));
          float v125_data = s0[180];
          float v127_data = ir1[15];
          ir1[15] = (v127_data + (v49_data * v125_data));
          float v129_data = r0[1];
          float v130_data = s0[1];
          float v132_data = ir1[0];
          ir1[0] = (v132_data + (v129_data * v130_data));
          float v135_data = s0[13];
          float v137_data = ir1[1];
          ir1[1] = (v137_data + (v129_data * v135_data));
          float v140_data = s0[25];
          float v142_data = ir1[2];
          ir1[2] = (v142_data + (v129_data * v140_data));
          float v145_data = s0[37];
          float v147_data = ir1[3];
          ir1[3] = (v147_data + (v129_data * v145_data));
          float v150_data = s0[49];
          float v152_data = ir1[4];
          ir1[4] = (v152_data + (v129_data * v150_data));
          float v155_data = s0[61];
          float v157_data = ir1[5];
          ir1[5] = (v157_data + (v129_data * v155_data));
          float v160_data = s0[73];
          float v162_data = ir1[6];
          ir1[6] = (v162_data + (v129_data * v160_data));
          float v165_data = s0[85];
          float v167_data = ir1[7];
          ir1[7] = (v167_data + (v129_data * v165_data));
          float v170_data = s0[97];
          float v172_data = ir1[8];
          ir1[8] = (v172_data + (v129_data * v170_data));
          float v175_data = s0[109];
          float v177_data = ir1[9];
          ir1[9] = (v177_data + (v129_data * v175_data));
          float v180_data = s0[121];
          float v182_data = ir1[10];
          ir1[10] = (v182_data + (v129_data * v180_data));
          float v185_data = s0[133];
          float v187_data = ir1[11];
          ir1[11] = (v187_data + (v129_data * v185_data));
          float v190_data = s0[145];
          float v192_data = ir1[12];
          ir1[12] = (v192_data + (v129_data * v190_data));
          float v195_data = s0[157];
          float v197_data = ir1[13];
          ir1[13] = (v197_data + (v129_data * v195_data));
          float v200_data = s0[169];
          float v202_data = ir1[14];
          ir1[14] = (v202_data + (v129_data * v200_data));
          float v205_data = s0[181];
          float v207_data = ir1[15];
          ir1[15] = (v207_data + (v129_data * v205_data));
          float v209_data = r0[2];
          float v210_data = s0[2];
          float v212_data = ir1[0];
          ir1[0] = (v212_data + (v209_data * v210_data));
          float v215_data = s0[14];
          float v217_data = ir1[1];
          ir1[1] = (v217_data + (v209_data * v215_data));
          float v220_data = s0[26];
          float v222_data = ir1[2];
          ir1[2] = (v222_data + (v209_data * v220_data));
          float v225_data = s0[38];
          float v227_data = ir1[3];
          ir1[3] = (v227_data + (v209_data * v225_data));
          float v230_data = s0[50];
          float v232_data = ir1[4];
          ir1[4] = (v232_data + (v209_data * v230_data));
          float v235_data = s0[62];
          float v237_data = ir1[5];
          ir1[5] = (v237_data + (v209_data * v235_data));
          float v240_data = s0[74];
          float v242_data = ir1[6];
          ir1[6] = (v242_data + (v209_data * v240_data));
          float v245_data = s0[86];
          float v247_data = ir1[7];
          ir1[7] = (v247_data + (v209_data * v245_data));
          float v250_data = s0[98];
          float v252_data = ir1[8];
          ir1[8] = (v252_data + (v209_data * v250_data));
          float v255_data = s0[110];
          float v257_data = ir1[9];
          ir1[9] = (v257_data + (v209_data * v255_data));
          float v260_data = s0[122];
          float v262_data = ir1[10];
          ir1[10] = (v262_data + (v209_data * v260_data));
          float v265_data = s0[134];
          float v267_data = ir1[11];
          ir1[11] = (v267_data + (v209_data * v265_data));
          float v270_data = s0[146];
          float v272_data = ir1[12];
          ir1[12] = (v272_data + (v209_data * v270_data));
          float v275_data = s0[158];
          float v277_data = ir1[13];
          ir1[13] = (v277_data + (v209_data * v275_data));
          float v280_data = s0[170];
          float v282_data = ir1[14];
          ir1[14] = (v282_data + (v209_data * v280_data));
          float v285_data = s0[182];
          float v287_data = ir1[15];
          ir1[15] = (v287_data + (v209_data * v285_data));
          float v289_data = r0[3];
          float v290_data = s0[3];
          float v292_data = ir1[0];
          ir1[0] = (v292_data + (v289_data * v290_data));
          float v295_data = s0[15];
          float v297_data = ir1[1];
          ir1[1] = (v297_data + (v289_data * v295_data));
          float v300_data = s0[27];
          float v302_data = ir1[2];
          ir1[2] = (v302_data + (v289_data * v300_data));
          float v305_data = s0[39];
          float v307_data = ir1[3];
          ir1[3] = (v307_data + (v289_data * v305_data));
          float v310_data = s0[51];
          float v312_data = ir1[4];
          ir1[4] = (v312_data + (v289_data * v310_data));
          float v315_data = s0[63];
          float v317_data = ir1[5];
          ir1[5] = (v317_data + (v289_data * v315_data));
          float v320_data = s0[75];
          float v322_data = ir1[6];
          ir1[6] = (v322_data + (v289_data * v320_data));
          float v325_data = s0[87];
          float v327_data = ir1[7];
          ir1[7] = (v327_data + (v289_data * v325_data));
          float v330_data = s0[99];
          float v332_data = ir1[8];
          ir1[8] = (v332_data + (v289_data * v330_data));
          float v335_data = s0[111];
          float v337_data = ir1[9];
          ir1[9] = (v337_data + (v289_data * v335_data));
          float v340_data = s0[123];
          float v342_data = ir1[10];
          ir1[10] = (v342_data + (v289_data * v340_data));
          float v345_data = s0[135];
          float v347_data = ir1[11];
          ir1[11] = (v347_data + (v289_data * v345_data));
          float v350_data = s0[147];
          float v352_data = ir1[12];
          ir1[12] = (v352_data + (v289_data * v350_data));
          float v355_data = s0[159];
          float v357_data = ir1[13];
          ir1[13] = (v357_data + (v289_data * v355_data));
          float v360_data = s0[171];
          float v362_data = ir1[14];
          ir1[14] = (v362_data + (v289_data * v360_data));
          float v365_data = s0[183];
          float v367_data = ir1[15];
          ir1[15] = (v367_data + (v289_data * v365_data));
          float v369_data = r0[4];
          float v370_data = s0[4];
          float v372_data = ir1[0];
          ir1[0] = (v372_data + (v369_data * v370_data));
          float v375_data = s0[16];
          float v377_data = ir1[1];
          ir1[1] = (v377_data + (v369_data * v375_data));
          float v380_data = s0[28];
          float v382_data = ir1[2];
          ir1[2] = (v382_data + (v369_data * v380_data));
          float v385_data = s0[40];
          float v387_data = ir1[3];
          ir1[3] = (v387_data + (v369_data * v385_data));
          float v390_data = s0[52];
          float v392_data = ir1[4];
          ir1[4] = (v392_data + (v369_data * v390_data));
          float v395_data = s0[64];
          float v397_data = ir1[5];
          ir1[5] = (v397_data + (v369_data * v395_data));
          float v400_data = s0[76];
          float v402_data = ir1[6];
          ir1[6] = (v402_data + (v369_data * v400_data));
          float v405_data = s0[88];
          float v407_data = ir1[7];
          ir1[7] = (v407_data + (v369_data * v405_data));
          float v410_data = s0[100];
          float v412_data = ir1[8];
          ir1[8] = (v412_data + (v369_data * v410_data));
          float v415_data = s0[112];
          float v417_data = ir1[9];
          ir1[9] = (v417_data + (v369_data * v415_data));
          float v420_data = s0[124];
          float v422_data = ir1[10];
          ir1[10] = (v422_data + (v369_data * v420_data));
          float v425_data = s0[136];
          float v427_data = ir1[11];
          ir1[11] = (v427_data + (v369_data * v425_data));
          float v430_data = s0[148];
          float v432_data = ir1[12];
          ir1[12] = (v432_data + (v369_data * v430_data));
          float v435_data = s0[160];
          float v437_data = ir1[13];
          ir1[13] = (v437_data + (v369_data * v435_data));
          float v440_data = s0[172];
          float v442_data = ir1[14];
          ir1[14] = (v442_data + (v369_data * v440_data));
          float v445_data = s0[184];
          float v447_data = ir1[15];
          ir1[15] = (v447_data + (v369_data * v445_data));
          float v449_data = r0[5];
          float v450_data = s0[5];
          float v452_data = ir1[0];
          ir1[0] = (v452_data + (v449_data * v450_data));
          float v455_data = s0[17];
          float v457_data = ir1[1];
          ir1[1] = (v457_data + (v449_data * v455_data));
          float v460_data = s0[29];
          float v462_data = ir1[2];
          ir1[2] = (v462_data + (v449_data * v460_data));
          float v465_data = s0[41];
          float v467_data = ir1[3];
          ir1[3] = (v467_data + (v449_data * v465_data));
          float v470_data = s0[53];
          float v472_data = ir1[4];
          ir1[4] = (v472_data + (v449_data * v470_data));
          float v475_data = s0[65];
          float v477_data = ir1[5];
          ir1[5] = (v477_data + (v449_data * v475_data));
          float v480_data = s0[77];
          float v482_data = ir1[6];
          ir1[6] = (v482_data + (v449_data * v480_data));
          float v485_data = s0[89];
          float v487_data = ir1[7];
          ir1[7] = (v487_data + (v449_data * v485_data));
          float v490_data = s0[101];
          float v492_data = ir1[8];
          ir1[8] = (v492_data + (v449_data * v490_data));
          float v495_data = s0[113];
          float v497_data = ir1[9];
          ir1[9] = (v497_data + (v449_data * v495_data));
          float v500_data = s0[125];
          float v502_data = ir1[10];
          ir1[10] = (v502_data + (v449_data * v500_data));
          float v505_data = s0[137];
          float v507_data = ir1[11];
          ir1[11] = (v507_data + (v449_data * v505_data));
          float v510_data = s0[149];
          float v512_data = ir1[12];
          ir1[12] = (v512_data + (v449_data * v510_data));
          float v515_data = s0[161];
          float v517_data = ir1[13];
          ir1[13] = (v517_data + (v449_data * v515_data));
          float v520_data = s0[173];
          float v522_data = ir1[14];
          ir1[14] = (v522_data + (v449_data * v520_data));
          float v525_data = s0[185];
          float v527_data = ir1[15];
          ir1[15] = (v527_data + (v449_data * v525_data));
          float v529_data = r0[6];
          float v530_data = s0[6];
          float v532_data = ir1[0];
          ir1[0] = (v532_data + (v529_data * v530_data));
          float v535_data = s0[18];
          float v537_data = ir1[1];
          ir1[1] = (v537_data + (v529_data * v535_data));
          float v540_data = s0[30];
          float v542_data = ir1[2];
          ir1[2] = (v542_data + (v529_data * v540_data));
          float v545_data = s0[42];
          float v547_data = ir1[3];
          ir1[3] = (v547_data + (v529_data * v545_data));
          float v550_data = s0[54];
          float v552_data = ir1[4];
          ir1[4] = (v552_data + (v529_data * v550_data));
          float v555_data = s0[66];
          float v557_data = ir1[5];
          ir1[5] = (v557_data + (v529_data * v555_data));
          float v560_data = s0[78];
          float v562_data = ir1[6];
          ir1[6] = (v562_data + (v529_data * v560_data));
          float v565_data = s0[90];
          float v567_data = ir1[7];
          ir1[7] = (v567_data + (v529_data * v565_data));
          float v570_data = s0[102];
          float v572_data = ir1[8];
          ir1[8] = (v572_data + (v529_data * v570_data));
          float v575_data = s0[114];
          float v577_data = ir1[9];
          ir1[9] = (v577_data + (v529_data * v575_data));
          float v580_data = s0[126];
          float v582_data = ir1[10];
          ir1[10] = (v582_data + (v529_data * v580_data));
          float v585_data = s0[138];
          float v587_data = ir1[11];
          ir1[11] = (v587_data + (v529_data * v585_data));
          float v590_data = s0[150];
          float v592_data = ir1[12];
          ir1[12] = (v592_data + (v529_data * v590_data));
          float v595_data = s0[162];
          float v597_data = ir1[13];
          ir1[13] = (v597_data + (v529_data * v595_data));
          float v600_data = s0[174];
          float v602_data = ir1[14];
          ir1[14] = (v602_data + (v529_data * v600_data));
          float v605_data = s0[186];
          float v607_data = ir1[15];
          ir1[15] = (v607_data + (v529_data * v605_data));
          float v609_data = r0[7];
          float v610_data = s0[7];
          float v612_data = ir1[0];
          ir1[0] = (v612_data + (v609_data * v610_data));
          float v615_data = s0[19];
          float v617_data = ir1[1];
          ir1[1] = (v617_data + (v609_data * v615_data));
          float v620_data = s0[31];
          float v622_data = ir1[2];
          ir1[2] = (v622_data + (v609_data * v620_data));
          float v625_data = s0[43];
          float v627_data = ir1[3];
          ir1[3] = (v627_data + (v609_data * v625_data));
          float v630_data = s0[55];
          float v632_data = ir1[4];
          ir1[4] = (v632_data + (v609_data * v630_data));
          float v635_data = s0[67];
          float v637_data = ir1[5];
          ir1[5] = (v637_data + (v609_data * v635_data));
          float v640_data = s0[79];
          float v642_data = ir1[6];
          ir1[6] = (v642_data + (v609_data * v640_data));
          float v645_data = s0[91];
          float v647_data = ir1[7];
          ir1[7] = (v647_data + (v609_data * v645_data));
          float v650_data = s0[103];
          float v652_data = ir1[8];
          ir1[8] = (v652_data + (v609_data * v650_data));
          float v655_data = s0[115];
          float v657_data = ir1[9];
          ir1[9] = (v657_data + (v609_data * v655_data));
          float v660_data = s0[127];
          float v662_data = ir1[10];
          ir1[10] = (v662_data + (v609_data * v660_data));
          float v665_data = s0[139];
          float v667_data = ir1[11];
          ir1[11] = (v667_data + (v609_data * v665_data));
          float v670_data = s0[151];
          float v672_data = ir1[12];
          ir1[12] = (v672_data + (v609_data * v670_data));
          float v675_data = s0[163];
          float v677_data = ir1[13];
          ir1[13] = (v677_data + (v609_data * v675_data));
          float v680_data = s0[175];
          float v682_data = ir1[14];
          ir1[14] = (v682_data + (v609_data * v680_data));
          float v685_data = s0[187];
          float v687_data = ir1[15];
          ir1[15] = (v687_data + (v609_data * v685_data));
          float v689_data = r0[8];
          float v690_data = s0[8];
          float v692_data = ir1[0];
          ir1[0] = (v692_data + (v689_data * v690_data));
          float v695_data = s0[20];
          float v697_data = ir1[1];
          ir1[1] = (v697_data + (v689_data * v695_data));
          float v700_data = s0[32];
          float v702_data = ir1[2];
          ir1[2] = (v702_data + (v689_data * v700_data));
          float v705_data = s0[44];
          float v707_data = ir1[3];
          ir1[3] = (v707_data + (v689_data * v705_data));
          float v710_data = s0[56];
          float v712_data = ir1[4];
          ir1[4] = (v712_data + (v689_data * v710_data));
          float v715_data = s0[68];
          float v717_data = ir1[5];
          ir1[5] = (v717_data + (v689_data * v715_data));
          float v720_data = s0[80];
          float v722_data = ir1[6];
          ir1[6] = (v722_data + (v689_data * v720_data));
          float v725_data = s0[92];
          float v727_data = ir1[7];
          ir1[7] = (v727_data + (v689_data * v725_data));
          float v730_data = s0[104];
          float v732_data = ir1[8];
          ir1[8] = (v732_data + (v689_data * v730_data));
          float v735_data = s0[116];
          float v737_data = ir1[9];
          ir1[9] = (v737_data + (v689_data * v735_data));
          float v740_data = s0[128];
          float v742_data = ir1[10];
          ir1[10] = (v742_data + (v689_data * v740_data));
          float v745_data = s0[140];
          float v747_data = ir1[11];
          ir1[11] = (v747_data + (v689_data * v745_data));
          float v750_data = s0[152];
          float v752_data = ir1[12];
          ir1[12] = (v752_data + (v689_data * v750_data));
          float v755_data = s0[164];
          float v757_data = ir1[13];
          ir1[13] = (v757_data + (v689_data * v755_data));
          float v760_data = s0[176];
          float v762_data = ir1[14];
          ir1[14] = (v762_data + (v689_data * v760_data));
          float v765_data = s0[188];
          float v767_data = ir1[15];
          ir1[15] = (v767_data + (v689_data * v765_data));
          float v769_data = r0[9];
          float v770_data = s0[9];
          float v772_data = ir1[0];
          ir1[0] = (v772_data + (v769_data * v770_data));
          float v775_data = s0[21];
          float v777_data = ir1[1];
          ir1[1] = (v777_data + (v769_data * v775_data));
          float v780_data = s0[33];
          float v782_data = ir1[2];
          ir1[2] = (v782_data + (v769_data * v780_data));
          float v785_data = s0[45];
          float v787_data = ir1[3];
          ir1[3] = (v787_data + (v769_data * v785_data));
          float v790_data = s0[57];
          float v792_data = ir1[4];
          ir1[4] = (v792_data + (v769_data * v790_data));
          float v795_data = s0[69];
          float v797_data = ir1[5];
          ir1[5] = (v797_data + (v769_data * v795_data));
          float v800_data = s0[81];
          float v802_data = ir1[6];
          ir1[6] = (v802_data + (v769_data * v800_data));
          float v805_data = s0[93];
          float v807_data = ir1[7];
          ir1[7] = (v807_data + (v769_data * v805_data));
          float v810_data = s0[105];
          float v812_data = ir1[8];
          ir1[8] = (v812_data + (v769_data * v810_data));
          float v815_data = s0[117];
          float v817_data = ir1[9];
          ir1[9] = (v817_data + (v769_data * v815_data));
          float v820_data = s0[129];
          float v822_data = ir1[10];
          ir1[10] = (v822_data + (v769_data * v820_data));
          float v825_data = s0[141];
          float v827_data = ir1[11];
          ir1[11] = (v827_data + (v769_data * v825_data));
          float v830_data = s0[153];
          float v832_data = ir1[12];
          ir1[12] = (v832_data + (v769_data * v830_data));
          float v835_data = s0[165];
          float v837_data = ir1[13];
          ir1[13] = (v837_data + (v769_data * v835_data));
          float v840_data = s0[177];
          float v842_data = ir1[14];
          ir1[14] = (v842_data + (v769_data * v840_data));
          float v845_data = s0[189];
          float v847_data = ir1[15];
          ir1[15] = (v847_data + (v769_data * v845_data));
          float v849_data = r0[10];
          float v850_data = s0[10];
          float v852_data = ir1[0];
          ir1[0] = (v852_data + (v849_data * v850_data));
          float v855_data = s0[22];
          float v857_data = ir1[1];
          ir1[1] = (v857_data + (v849_data * v855_data));
          float v860_data = s0[34];
          float v862_data = ir1[2];
          ir1[2] = (v862_data + (v849_data * v860_data));
          float v865_data = s0[46];
          float v867_data = ir1[3];
          ir1[3] = (v867_data + (v849_data * v865_data));
          float v870_data = s0[58];
          float v872_data = ir1[4];
          ir1[4] = (v872_data + (v849_data * v870_data));
          float v875_data = s0[70];
          float v877_data = ir1[5];
          ir1[5] = (v877_data + (v849_data * v875_data));
          float v880_data = s0[82];
          float v882_data = ir1[6];
          ir1[6] = (v882_data + (v849_data * v880_data));
          float v885_data = s0[94];
          float v887_data = ir1[7];
          ir1[7] = (v887_data + (v849_data * v885_data));
          float v890_data = s0[106];
          float v892_data = ir1[8];
          ir1[8] = (v892_data + (v849_data * v890_data));
          float v895_data = s0[118];
          float v897_data = ir1[9];
          ir1[9] = (v897_data + (v849_data * v895_data));
          float v900_data = s0[130];
          float v902_data = ir1[10];
          ir1[10] = (v902_data + (v849_data * v900_data));
          float v905_data = s0[142];
          float v907_data = ir1[11];
          ir1[11] = (v907_data + (v849_data * v905_data));
          float v910_data = s0[154];
          float v912_data = ir1[12];
          ir1[12] = (v912_data + (v849_data * v910_data));
          float v915_data = s0[166];
          float v917_data = ir1[13];
          ir1[13] = (v917_data + (v849_data * v915_data));
          float v920_data = s0[178];
          float v922_data = ir1[14];
          ir1[14] = (v922_data + (v849_data * v920_data));
          float v925_data = s0[190];
          float v927_data = ir1[15];
          ir1[15] = (v927_data + (v849_data * v925_data));
          float v929_data = r0[11];
          float v930_data = s0[11];
          float v932_data = ir1[0];
          ir1[0] = (v932_data + (v929_data * v930_data));
          float v935_data = s0[23];
          float v937_data = ir1[1];
          ir1[1] = (v937_data + (v929_data * v935_data));
          float v940_data = s0[35];
          float v942_data = ir1[2];
          ir1[2] = (v942_data + (v929_data * v940_data));
          float v945_data = s0[47];
          float v947_data = ir1[3];
          ir1[3] = (v947_data + (v929_data * v945_data));
          float v950_data = s0[59];
          float v952_data = ir1[4];
          ir1[4] = (v952_data + (v929_data * v950_data));
          float v955_data = s0[71];
          float v957_data = ir1[5];
          ir1[5] = (v957_data + (v929_data * v955_data));
          float v960_data = s0[83];
          float v962_data = ir1[6];
          ir1[6] = (v962_data + (v929_data * v960_data));
          float v965_data = s0[95];
          float v967_data = ir1[7];
          ir1[7] = (v967_data + (v929_data * v965_data));
          float v970_data = s0[107];
          float v972_data = ir1[8];
          ir1[8] = (v972_data + (v929_data * v970_data));
          float v975_data = s0[119];
          float v977_data = ir1[9];
          ir1[9] = (v977_data + (v929_data * v975_data));
          float v980_data = s0[131];
          float v982_data = ir1[10];
          ir1[10] = (v982_data + (v929_data * v980_data));
          float v985_data = s0[143];
          float v987_data = ir1[11];
          ir1[11] = (v987_data + (v929_data * v985_data));
          float v990_data = s0[155];
          float v992_data = ir1[12];
          ir1[12] = (v992_data + (v929_data * v990_data));
          float v995_data = s0[167];
          float v997_data = ir1[13];
          ir1[13] = (v997_data + (v929_data * v995_data));
          float v1000_data = s0[179];
          float v1002_data = ir1[14];
          ir1[14] = (v1002_data + (v929_data * v1000_data));
          float v1005_data = s0[191];
          float v1007_data = ir1[15];
          ir1[15] = (v1007_data + (v929_data * v1005_data));
          // r1 = ir1
          #pragma unroll
          for (int32_t v1009_n0 = 0; v1009_n0 < 1; ++v1009_n0) {
            #pragma unroll
            for (int32_t v1010_n1 = 0; v1010_n1 < 16; ++v1010_n1) {
              int32_t v1011_a = v1009_n0 + v1010_n1;
              float v1012_data = ir1[v1011_a];
              r1[v1011_a] = v1012_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v1013_i0 = 0; v1013_i0 < 1; ++v1013_i0) {
            int32_t v1018_lead = v28_lead + (v1013_i0 * 32);
            #pragma unroll
            for (int32_t v1014_i1 = 0; v1014_i1 < 16; ++v1014_i1) {
              float v1016_data = r1[(v1013_i0 + v1014_i1)];
              glb_m0[(v1018_lead + (v1014_i1 * 32))] = v1016_data;
            }
          }
          // s1 = load{g>s}(glb_m4[0, 1])
          __syncwarp();
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 0], &glb_m4[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 32], &glb_m4[0 + 0 + 1 * threadIdx.x + 32], 4);
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 64], &glb_m4[0 + 0 + 1 * threadIdx.x + 64], 4);
          __pipeline_commit();
          // wait(r2 = load{g>r}(glb_m3););
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
          for (int32_t v1034_i0 = 0; v1034_i0 < 1; ++v1034_i0) {
            int32_t v1037_lead = v28_lead + (v1034_i0 * 32);
            #pragma unroll
            for (int32_t v1035_i1 = 0; v1035_i1 < 12; ++v1035_i1) {
              float v1040_data = __ldcg(&glb_m5[(v1037_lead + (v1035_i1 * 32))]);
              r5[(v1034_i0 + v1035_i1)] = v1040_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m0););
          float r4[8]{};
          // ir4 = +(r2 * s1)
          // [(0, 32), (0, 8)] [(0, 12)]
          float ir4[8]{};
          float v1044_data = r2[0];
          __syncwarp();
          float v1045_data = s1[0];
          float v1047_data = ir4[0];
          ir4[0] = (v1047_data + (v1044_data * v1045_data));
          float v1050_data = s1[12];
          float v1052_data = ir4[1];
          ir4[1] = (v1052_data + (v1044_data * v1050_data));
          float v1055_data = s1[24];
          float v1057_data = ir4[2];
          ir4[2] = (v1057_data + (v1044_data * v1055_data));
          float v1060_data = s1[36];
          float v1062_data = ir4[3];
          ir4[3] = (v1062_data + (v1044_data * v1060_data));
          float v1065_data = s1[48];
          float v1067_data = ir4[4];
          ir4[4] = (v1067_data + (v1044_data * v1065_data));
          float v1070_data = s1[60];
          float v1072_data = ir4[5];
          ir4[5] = (v1072_data + (v1044_data * v1070_data));
          float v1075_data = s1[72];
          float v1077_data = ir4[6];
          ir4[6] = (v1077_data + (v1044_data * v1075_data));
          float v1080_data = s1[84];
          float v1082_data = ir4[7];
          ir4[7] = (v1082_data + (v1044_data * v1080_data));
          float v1084_data = r2[1];
          float v1085_data = s1[1];
          float v1087_data = ir4[0];
          ir4[0] = (v1087_data + (v1084_data * v1085_data));
          float v1090_data = s1[13];
          float v1092_data = ir4[1];
          ir4[1] = (v1092_data + (v1084_data * v1090_data));
          float v1095_data = s1[25];
          float v1097_data = ir4[2];
          ir4[2] = (v1097_data + (v1084_data * v1095_data));
          float v1100_data = s1[37];
          float v1102_data = ir4[3];
          ir4[3] = (v1102_data + (v1084_data * v1100_data));
          float v1105_data = s1[49];
          float v1107_data = ir4[4];
          ir4[4] = (v1107_data + (v1084_data * v1105_data));
          float v1110_data = s1[61];
          float v1112_data = ir4[5];
          ir4[5] = (v1112_data + (v1084_data * v1110_data));
          float v1115_data = s1[73];
          float v1117_data = ir4[6];
          ir4[6] = (v1117_data + (v1084_data * v1115_data));
          float v1120_data = s1[85];
          float v1122_data = ir4[7];
          ir4[7] = (v1122_data + (v1084_data * v1120_data));
          float v1124_data = r2[2];
          float v1125_data = s1[2];
          float v1127_data = ir4[0];
          ir4[0] = (v1127_data + (v1124_data * v1125_data));
          float v1130_data = s1[14];
          float v1132_data = ir4[1];
          ir4[1] = (v1132_data + (v1124_data * v1130_data));
          float v1135_data = s1[26];
          float v1137_data = ir4[2];
          ir4[2] = (v1137_data + (v1124_data * v1135_data));
          float v1140_data = s1[38];
          float v1142_data = ir4[3];
          ir4[3] = (v1142_data + (v1124_data * v1140_data));
          float v1145_data = s1[50];
          float v1147_data = ir4[4];
          ir4[4] = (v1147_data + (v1124_data * v1145_data));
          float v1150_data = s1[62];
          float v1152_data = ir4[5];
          ir4[5] = (v1152_data + (v1124_data * v1150_data));
          float v1155_data = s1[74];
          float v1157_data = ir4[6];
          ir4[6] = (v1157_data + (v1124_data * v1155_data));
          float v1160_data = s1[86];
          float v1162_data = ir4[7];
          ir4[7] = (v1162_data + (v1124_data * v1160_data));
          float v1164_data = r2[3];
          float v1165_data = s1[3];
          float v1167_data = ir4[0];
          ir4[0] = (v1167_data + (v1164_data * v1165_data));
          float v1170_data = s1[15];
          float v1172_data = ir4[1];
          ir4[1] = (v1172_data + (v1164_data * v1170_data));
          float v1175_data = s1[27];
          float v1177_data = ir4[2];
          ir4[2] = (v1177_data + (v1164_data * v1175_data));
          float v1180_data = s1[39];
          float v1182_data = ir4[3];
          ir4[3] = (v1182_data + (v1164_data * v1180_data));
          float v1185_data = s1[51];
          float v1187_data = ir4[4];
          ir4[4] = (v1187_data + (v1164_data * v1185_data));
          float v1190_data = s1[63];
          float v1192_data = ir4[5];
          ir4[5] = (v1192_data + (v1164_data * v1190_data));
          float v1195_data = s1[75];
          float v1197_data = ir4[6];
          ir4[6] = (v1197_data + (v1164_data * v1195_data));
          float v1200_data = s1[87];
          float v1202_data = ir4[7];
          ir4[7] = (v1202_data + (v1164_data * v1200_data));
          float v1204_data = r2[4];
          float v1205_data = s1[4];
          float v1207_data = ir4[0];
          ir4[0] = (v1207_data + (v1204_data * v1205_data));
          float v1210_data = s1[16];
          float v1212_data = ir4[1];
          ir4[1] = (v1212_data + (v1204_data * v1210_data));
          float v1215_data = s1[28];
          float v1217_data = ir4[2];
          ir4[2] = (v1217_data + (v1204_data * v1215_data));
          float v1220_data = s1[40];
          float v1222_data = ir4[3];
          ir4[3] = (v1222_data + (v1204_data * v1220_data));
          float v1225_data = s1[52];
          float v1227_data = ir4[4];
          ir4[4] = (v1227_data + (v1204_data * v1225_data));
          float v1230_data = s1[64];
          float v1232_data = ir4[5];
          ir4[5] = (v1232_data + (v1204_data * v1230_data));
          float v1235_data = s1[76];
          float v1237_data = ir4[6];
          ir4[6] = (v1237_data + (v1204_data * v1235_data));
          float v1240_data = s1[88];
          float v1242_data = ir4[7];
          ir4[7] = (v1242_data + (v1204_data * v1240_data));
          float v1244_data = r2[5];
          float v1245_data = s1[5];
          float v1247_data = ir4[0];
          ir4[0] = (v1247_data + (v1244_data * v1245_data));
          float v1250_data = s1[17];
          float v1252_data = ir4[1];
          ir4[1] = (v1252_data + (v1244_data * v1250_data));
          float v1255_data = s1[29];
          float v1257_data = ir4[2];
          ir4[2] = (v1257_data + (v1244_data * v1255_data));
          float v1260_data = s1[41];
          float v1262_data = ir4[3];
          ir4[3] = (v1262_data + (v1244_data * v1260_data));
          float v1265_data = s1[53];
          float v1267_data = ir4[4];
          ir4[4] = (v1267_data + (v1244_data * v1265_data));
          float v1270_data = s1[65];
          float v1272_data = ir4[5];
          ir4[5] = (v1272_data + (v1244_data * v1270_data));
          float v1275_data = s1[77];
          float v1277_data = ir4[6];
          ir4[6] = (v1277_data + (v1244_data * v1275_data));
          float v1280_data = s1[89];
          float v1282_data = ir4[7];
          ir4[7] = (v1282_data + (v1244_data * v1280_data));
          float v1284_data = r2[6];
          float v1285_data = s1[6];
          float v1287_data = ir4[0];
          ir4[0] = (v1287_data + (v1284_data * v1285_data));
          float v1290_data = s1[18];
          float v1292_data = ir4[1];
          ir4[1] = (v1292_data + (v1284_data * v1290_data));
          float v1295_data = s1[30];
          float v1297_data = ir4[2];
          ir4[2] = (v1297_data + (v1284_data * v1295_data));
          float v1300_data = s1[42];
          float v1302_data = ir4[3];
          ir4[3] = (v1302_data + (v1284_data * v1300_data));
          float v1305_data = s1[54];
          float v1307_data = ir4[4];
          ir4[4] = (v1307_data + (v1284_data * v1305_data));
          float v1310_data = s1[66];
          float v1312_data = ir4[5];
          ir4[5] = (v1312_data + (v1284_data * v1310_data));
          float v1315_data = s1[78];
          float v1317_data = ir4[6];
          ir4[6] = (v1317_data + (v1284_data * v1315_data));
          float v1320_data = s1[90];
          float v1322_data = ir4[7];
          ir4[7] = (v1322_data + (v1284_data * v1320_data));
          float v1324_data = r2[7];
          float v1325_data = s1[7];
          float v1327_data = ir4[0];
          ir4[0] = (v1327_data + (v1324_data * v1325_data));
          float v1330_data = s1[19];
          float v1332_data = ir4[1];
          ir4[1] = (v1332_data + (v1324_data * v1330_data));
          float v1335_data = s1[31];
          float v1337_data = ir4[2];
          ir4[2] = (v1337_data + (v1324_data * v1335_data));
          float v1340_data = s1[43];
          float v1342_data = ir4[3];
          ir4[3] = (v1342_data + (v1324_data * v1340_data));
          float v1345_data = s1[55];
          float v1347_data = ir4[4];
          ir4[4] = (v1347_data + (v1324_data * v1345_data));
          float v1350_data = s1[67];
          float v1352_data = ir4[5];
          ir4[5] = (v1352_data + (v1324_data * v1350_data));
          float v1355_data = s1[79];
          float v1357_data = ir4[6];
          ir4[6] = (v1357_data + (v1324_data * v1355_data));
          float v1360_data = s1[91];
          float v1362_data = ir4[7];
          ir4[7] = (v1362_data + (v1324_data * v1360_data));
          float v1364_data = r2[8];
          float v1365_data = s1[8];
          float v1367_data = ir4[0];
          ir4[0] = (v1367_data + (v1364_data * v1365_data));
          float v1370_data = s1[20];
          float v1372_data = ir4[1];
          ir4[1] = (v1372_data + (v1364_data * v1370_data));
          float v1375_data = s1[32];
          float v1377_data = ir4[2];
          ir4[2] = (v1377_data + (v1364_data * v1375_data));
          float v1380_data = s1[44];
          float v1382_data = ir4[3];
          ir4[3] = (v1382_data + (v1364_data * v1380_data));
          float v1385_data = s1[56];
          float v1387_data = ir4[4];
          ir4[4] = (v1387_data + (v1364_data * v1385_data));
          float v1390_data = s1[68];
          float v1392_data = ir4[5];
          ir4[5] = (v1392_data + (v1364_data * v1390_data));
          float v1395_data = s1[80];
          float v1397_data = ir4[6];
          ir4[6] = (v1397_data + (v1364_data * v1395_data));
          float v1400_data = s1[92];
          float v1402_data = ir4[7];
          ir4[7] = (v1402_data + (v1364_data * v1400_data));
          float v1404_data = r2[9];
          float v1405_data = s1[9];
          float v1407_data = ir4[0];
          ir4[0] = (v1407_data + (v1404_data * v1405_data));
          float v1410_data = s1[21];
          float v1412_data = ir4[1];
          ir4[1] = (v1412_data + (v1404_data * v1410_data));
          float v1415_data = s1[33];
          float v1417_data = ir4[2];
          ir4[2] = (v1417_data + (v1404_data * v1415_data));
          float v1420_data = s1[45];
          float v1422_data = ir4[3];
          ir4[3] = (v1422_data + (v1404_data * v1420_data));
          float v1425_data = s1[57];
          float v1427_data = ir4[4];
          ir4[4] = (v1427_data + (v1404_data * v1425_data));
          float v1430_data = s1[69];
          float v1432_data = ir4[5];
          ir4[5] = (v1432_data + (v1404_data * v1430_data));
          float v1435_data = s1[81];
          float v1437_data = ir4[6];
          ir4[6] = (v1437_data + (v1404_data * v1435_data));
          float v1440_data = s1[93];
          float v1442_data = ir4[7];
          ir4[7] = (v1442_data + (v1404_data * v1440_data));
          float v1444_data = r2[10];
          float v1445_data = s1[10];
          float v1447_data = ir4[0];
          ir4[0] = (v1447_data + (v1444_data * v1445_data));
          float v1450_data = s1[22];
          float v1452_data = ir4[1];
          ir4[1] = (v1452_data + (v1444_data * v1450_data));
          float v1455_data = s1[34];
          float v1457_data = ir4[2];
          ir4[2] = (v1457_data + (v1444_data * v1455_data));
          float v1460_data = s1[46];
          float v1462_data = ir4[3];
          ir4[3] = (v1462_data + (v1444_data * v1460_data));
          float v1465_data = s1[58];
          float v1467_data = ir4[4];
          ir4[4] = (v1467_data + (v1444_data * v1465_data));
          float v1470_data = s1[70];
          float v1472_data = ir4[5];
          ir4[5] = (v1472_data + (v1444_data * v1470_data));
          float v1475_data = s1[82];
          float v1477_data = ir4[6];
          ir4[6] = (v1477_data + (v1444_data * v1475_data));
          float v1480_data = s1[94];
          float v1482_data = ir4[7];
          ir4[7] = (v1482_data + (v1444_data * v1480_data));
          float v1484_data = r2[11];
          float v1485_data = s1[11];
          float v1487_data = ir4[0];
          ir4[0] = (v1487_data + (v1484_data * v1485_data));
          float v1490_data = s1[23];
          float v1492_data = ir4[1];
          ir4[1] = (v1492_data + (v1484_data * v1490_data));
          float v1495_data = s1[35];
          float v1497_data = ir4[2];
          ir4[2] = (v1497_data + (v1484_data * v1495_data));
          float v1500_data = s1[47];
          float v1502_data = ir4[3];
          ir4[3] = (v1502_data + (v1484_data * v1500_data));
          float v1505_data = s1[59];
          float v1507_data = ir4[4];
          ir4[4] = (v1507_data + (v1484_data * v1505_data));
          float v1510_data = s1[71];
          float v1512_data = ir4[5];
          ir4[5] = (v1512_data + (v1484_data * v1510_data));
          float v1515_data = s1[83];
          float v1517_data = ir4[6];
          ir4[6] = (v1517_data + (v1484_data * v1515_data));
          float v1520_data = s1[95];
          float v1522_data = ir4[7];
          ir4[7] = (v1522_data + (v1484_data * v1520_data));
          // r4 = ir4 + r3
          #pragma unroll
          for (int32_t v1524_n0 = 0; v1524_n0 < 1; ++v1524_n0) {
            #pragma unroll
            for (int32_t v1525_n1 = 0; v1525_n1 < 8; ++v1525_n1) {
              int32_t v1526_a = v1524_n0 + v1525_n1;
              float v1527_data = ir4[v1526_a];
              float v1528_data = r3[v1526_a];
              r4[v1526_a] = (v1528_data + v1527_data);
            }
          }
          // glb_m0 = store{r>g}(r4);
          #pragma unroll
          for (int32_t v1530_i0 = 0; v1530_i0 < 1; ++v1530_i0) {
            int32_t v1535_lead = v28_lead + (v1530_i0 * 32);
            #pragma unroll
            for (int32_t v1531_i1 = 0; v1531_i1 < 8; ++v1531_i1) {
              float v1533_data = r4[(v1530_i0 + v1531_i1)];
              glb_m0[(v1535_lead + (v1531_i1 * 32))] = v1533_data;
            }
          }
          // s2 = load{g>s}(glb_m6[0, 1])
          __syncwarp();
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 0], &glb_m6[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 32], &glb_m6[0 + 0 + 1 * threadIdx.x + 32], 4);
          __pipeline_memcpy_async(&s2[0 + 0 + 1 * threadIdx.x + 64], &glb_m6[0 + 0 + 1 * threadIdx.x + 64], 4);
          __pipeline_commit();
          // wait(r5 = load{g>r}(glb_m5););
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
          // wait(r6 = load{g>r}(glb_m0););
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

