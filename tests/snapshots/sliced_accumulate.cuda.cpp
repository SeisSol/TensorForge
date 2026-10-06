// === base name ===
kernel_d8bcdcc1f8145a28

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_d8bcdcc1f8145a28 = {{32, 4, 1}, 32, 32, 1, 4, 3072, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_d8bcdcc1f8145a28(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_d8bcdcc1f8145a28(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_d8bcdcc1f8145a28(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_d8bcdcc1f8145a28, block.x * block.y * block.z, 768 * sizeof(float));
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
void launcher_kernel_d8bcdcc1f8145a28(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_d8bcdcc1f8145a28(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_d8bcdcc1f8145a28, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_d8bcdcc1f8145a28<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, m6, m6_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_d8bcdcc1f8145a28(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0) {
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
      float* tempShrMem = &localShrMem0[192];
      float * __restrict__ s0 = &localShrMem0[0];
      float * __restrict__ s1 = &localShrMem0[0];
      float * __restrict__ s2 = &localShrMem0[0];
      for (size_t v13_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v13_batchId0 < numElements0; v13_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v14_ahead1 = v13_batchId0 + (gridDim.x * blockDim.y);
        size_t v16_batchId1 = (v14_ahead1 < numElements0) ? v14_ahead1 : v13_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v13_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v13_batchId0 * 512 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v13_batchId0 * 384 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v13_batchId0 * 192 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[v13_batchId0 * 384 + 0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[v13_batchId0 * 96 + 0 + m4_extraOffset];
          const float *const __restrict__ glb_m5 = &m5[v13_batchId0 * 384 + 0 + m5_extraOffset];
          const float *const __restrict__ glb_m6 = &m6[v13_batchId0 * 96 + 0 + m6_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v31_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v32_i0 = 0; v32_i0 < 1; ++v32_i0) {
            int32_t v35_lead = v31_lead + (v32_i0 * 32);
            #pragma unroll
            for (int32_t v33_i1 = 0; v33_i1 < 12; ++v33_i1) {
              float v38_data = __ldcg(&glb_m1[(v35_lead + (v33_i1 * 32))]);
              r0[(v32_i0 + v33_i1)] = v38_data;
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
          for (int32_t v42_i0 = 0; v42_i0 < 1; ++v42_i0) {
            int32_t v45_lead = v31_lead + (v42_i0 * 32);
            #pragma unroll
            for (int32_t v43_i1 = 0; v43_i1 < 12; ++v43_i1) {
              float v48_data = __ldcg(&glb_m3[(v45_lead + (v43_i1 * 32))]);
              r2[(v42_i0 + v43_i1)] = v48_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[16]{};
          __syncwarp();
          // ir1 = +(r0 * s0)
          // [(0, 32), (0, 16)] [(0, 12)]
          float ir1[16]{};
          float v52_data = r0[0];
          float v53_data = s0[0];
          float v55_data = ir1[0];
          ir1[0] = (v55_data + (v52_data * v53_data));
          float v58_data = s0[12];
          float v60_data = ir1[1];
          ir1[1] = (v60_data + (v52_data * v58_data));
          float v63_data = s0[24];
          float v65_data = ir1[2];
          ir1[2] = (v65_data + (v52_data * v63_data));
          float v68_data = s0[36];
          float v70_data = ir1[3];
          ir1[3] = (v70_data + (v52_data * v68_data));
          float v73_data = s0[48];
          float v75_data = ir1[4];
          ir1[4] = (v75_data + (v52_data * v73_data));
          float v78_data = s0[60];
          float v80_data = ir1[5];
          ir1[5] = (v80_data + (v52_data * v78_data));
          float v83_data = s0[72];
          float v85_data = ir1[6];
          ir1[6] = (v85_data + (v52_data * v83_data));
          float v88_data = s0[84];
          float v90_data = ir1[7];
          ir1[7] = (v90_data + (v52_data * v88_data));
          float v93_data = s0[96];
          float v95_data = ir1[8];
          ir1[8] = (v95_data + (v52_data * v93_data));
          float v98_data = s0[108];
          float v100_data = ir1[9];
          ir1[9] = (v100_data + (v52_data * v98_data));
          float v103_data = s0[120];
          float v105_data = ir1[10];
          ir1[10] = (v105_data + (v52_data * v103_data));
          float v108_data = s0[132];
          float v110_data = ir1[11];
          ir1[11] = (v110_data + (v52_data * v108_data));
          float v113_data = s0[144];
          float v115_data = ir1[12];
          ir1[12] = (v115_data + (v52_data * v113_data));
          float v118_data = s0[156];
          float v120_data = ir1[13];
          ir1[13] = (v120_data + (v52_data * v118_data));
          float v123_data = s0[168];
          float v125_data = ir1[14];
          ir1[14] = (v125_data + (v52_data * v123_data));
          float v128_data = s0[180];
          float v130_data = ir1[15];
          ir1[15] = (v130_data + (v52_data * v128_data));
          float v132_data = r0[1];
          float v133_data = s0[1];
          float v135_data = ir1[0];
          ir1[0] = (v135_data + (v132_data * v133_data));
          float v138_data = s0[13];
          float v140_data = ir1[1];
          ir1[1] = (v140_data + (v132_data * v138_data));
          float v143_data = s0[25];
          float v145_data = ir1[2];
          ir1[2] = (v145_data + (v132_data * v143_data));
          float v148_data = s0[37];
          float v150_data = ir1[3];
          ir1[3] = (v150_data + (v132_data * v148_data));
          float v153_data = s0[49];
          float v155_data = ir1[4];
          ir1[4] = (v155_data + (v132_data * v153_data));
          float v158_data = s0[61];
          float v160_data = ir1[5];
          ir1[5] = (v160_data + (v132_data * v158_data));
          float v163_data = s0[73];
          float v165_data = ir1[6];
          ir1[6] = (v165_data + (v132_data * v163_data));
          float v168_data = s0[85];
          float v170_data = ir1[7];
          ir1[7] = (v170_data + (v132_data * v168_data));
          float v173_data = s0[97];
          float v175_data = ir1[8];
          ir1[8] = (v175_data + (v132_data * v173_data));
          float v178_data = s0[109];
          float v180_data = ir1[9];
          ir1[9] = (v180_data + (v132_data * v178_data));
          float v183_data = s0[121];
          float v185_data = ir1[10];
          ir1[10] = (v185_data + (v132_data * v183_data));
          float v188_data = s0[133];
          float v190_data = ir1[11];
          ir1[11] = (v190_data + (v132_data * v188_data));
          float v193_data = s0[145];
          float v195_data = ir1[12];
          ir1[12] = (v195_data + (v132_data * v193_data));
          float v198_data = s0[157];
          float v200_data = ir1[13];
          ir1[13] = (v200_data + (v132_data * v198_data));
          float v203_data = s0[169];
          float v205_data = ir1[14];
          ir1[14] = (v205_data + (v132_data * v203_data));
          float v208_data = s0[181];
          float v210_data = ir1[15];
          ir1[15] = (v210_data + (v132_data * v208_data));
          float v212_data = r0[2];
          float v213_data = s0[2];
          float v215_data = ir1[0];
          ir1[0] = (v215_data + (v212_data * v213_data));
          float v218_data = s0[14];
          float v220_data = ir1[1];
          ir1[1] = (v220_data + (v212_data * v218_data));
          float v223_data = s0[26];
          float v225_data = ir1[2];
          ir1[2] = (v225_data + (v212_data * v223_data));
          float v228_data = s0[38];
          float v230_data = ir1[3];
          ir1[3] = (v230_data + (v212_data * v228_data));
          float v233_data = s0[50];
          float v235_data = ir1[4];
          ir1[4] = (v235_data + (v212_data * v233_data));
          float v238_data = s0[62];
          float v240_data = ir1[5];
          ir1[5] = (v240_data + (v212_data * v238_data));
          float v243_data = s0[74];
          float v245_data = ir1[6];
          ir1[6] = (v245_data + (v212_data * v243_data));
          float v248_data = s0[86];
          float v250_data = ir1[7];
          ir1[7] = (v250_data + (v212_data * v248_data));
          float v253_data = s0[98];
          float v255_data = ir1[8];
          ir1[8] = (v255_data + (v212_data * v253_data));
          float v258_data = s0[110];
          float v260_data = ir1[9];
          ir1[9] = (v260_data + (v212_data * v258_data));
          float v263_data = s0[122];
          float v265_data = ir1[10];
          ir1[10] = (v265_data + (v212_data * v263_data));
          float v268_data = s0[134];
          float v270_data = ir1[11];
          ir1[11] = (v270_data + (v212_data * v268_data));
          float v273_data = s0[146];
          float v275_data = ir1[12];
          ir1[12] = (v275_data + (v212_data * v273_data));
          float v278_data = s0[158];
          float v280_data = ir1[13];
          ir1[13] = (v280_data + (v212_data * v278_data));
          float v283_data = s0[170];
          float v285_data = ir1[14];
          ir1[14] = (v285_data + (v212_data * v283_data));
          float v288_data = s0[182];
          float v290_data = ir1[15];
          ir1[15] = (v290_data + (v212_data * v288_data));
          float v292_data = r0[3];
          float v293_data = s0[3];
          float v295_data = ir1[0];
          ir1[0] = (v295_data + (v292_data * v293_data));
          float v298_data = s0[15];
          float v300_data = ir1[1];
          ir1[1] = (v300_data + (v292_data * v298_data));
          float v303_data = s0[27];
          float v305_data = ir1[2];
          ir1[2] = (v305_data + (v292_data * v303_data));
          float v308_data = s0[39];
          float v310_data = ir1[3];
          ir1[3] = (v310_data + (v292_data * v308_data));
          float v313_data = s0[51];
          float v315_data = ir1[4];
          ir1[4] = (v315_data + (v292_data * v313_data));
          float v318_data = s0[63];
          float v320_data = ir1[5];
          ir1[5] = (v320_data + (v292_data * v318_data));
          float v323_data = s0[75];
          float v325_data = ir1[6];
          ir1[6] = (v325_data + (v292_data * v323_data));
          float v328_data = s0[87];
          float v330_data = ir1[7];
          ir1[7] = (v330_data + (v292_data * v328_data));
          float v333_data = s0[99];
          float v335_data = ir1[8];
          ir1[8] = (v335_data + (v292_data * v333_data));
          float v338_data = s0[111];
          float v340_data = ir1[9];
          ir1[9] = (v340_data + (v292_data * v338_data));
          float v343_data = s0[123];
          float v345_data = ir1[10];
          ir1[10] = (v345_data + (v292_data * v343_data));
          float v348_data = s0[135];
          float v350_data = ir1[11];
          ir1[11] = (v350_data + (v292_data * v348_data));
          float v353_data = s0[147];
          float v355_data = ir1[12];
          ir1[12] = (v355_data + (v292_data * v353_data));
          float v358_data = s0[159];
          float v360_data = ir1[13];
          ir1[13] = (v360_data + (v292_data * v358_data));
          float v363_data = s0[171];
          float v365_data = ir1[14];
          ir1[14] = (v365_data + (v292_data * v363_data));
          float v368_data = s0[183];
          float v370_data = ir1[15];
          ir1[15] = (v370_data + (v292_data * v368_data));
          float v372_data = r0[4];
          float v373_data = s0[4];
          float v375_data = ir1[0];
          ir1[0] = (v375_data + (v372_data * v373_data));
          float v378_data = s0[16];
          float v380_data = ir1[1];
          ir1[1] = (v380_data + (v372_data * v378_data));
          float v383_data = s0[28];
          float v385_data = ir1[2];
          ir1[2] = (v385_data + (v372_data * v383_data));
          float v388_data = s0[40];
          float v390_data = ir1[3];
          ir1[3] = (v390_data + (v372_data * v388_data));
          float v393_data = s0[52];
          float v395_data = ir1[4];
          ir1[4] = (v395_data + (v372_data * v393_data));
          float v398_data = s0[64];
          float v400_data = ir1[5];
          ir1[5] = (v400_data + (v372_data * v398_data));
          float v403_data = s0[76];
          float v405_data = ir1[6];
          ir1[6] = (v405_data + (v372_data * v403_data));
          float v408_data = s0[88];
          float v410_data = ir1[7];
          ir1[7] = (v410_data + (v372_data * v408_data));
          float v413_data = s0[100];
          float v415_data = ir1[8];
          ir1[8] = (v415_data + (v372_data * v413_data));
          float v418_data = s0[112];
          float v420_data = ir1[9];
          ir1[9] = (v420_data + (v372_data * v418_data));
          float v423_data = s0[124];
          float v425_data = ir1[10];
          ir1[10] = (v425_data + (v372_data * v423_data));
          float v428_data = s0[136];
          float v430_data = ir1[11];
          ir1[11] = (v430_data + (v372_data * v428_data));
          float v433_data = s0[148];
          float v435_data = ir1[12];
          ir1[12] = (v435_data + (v372_data * v433_data));
          float v438_data = s0[160];
          float v440_data = ir1[13];
          ir1[13] = (v440_data + (v372_data * v438_data));
          float v443_data = s0[172];
          float v445_data = ir1[14];
          ir1[14] = (v445_data + (v372_data * v443_data));
          float v448_data = s0[184];
          float v450_data = ir1[15];
          ir1[15] = (v450_data + (v372_data * v448_data));
          float v452_data = r0[5];
          float v453_data = s0[5];
          float v455_data = ir1[0];
          ir1[0] = (v455_data + (v452_data * v453_data));
          float v458_data = s0[17];
          float v460_data = ir1[1];
          ir1[1] = (v460_data + (v452_data * v458_data));
          float v463_data = s0[29];
          float v465_data = ir1[2];
          ir1[2] = (v465_data + (v452_data * v463_data));
          float v468_data = s0[41];
          float v470_data = ir1[3];
          ir1[3] = (v470_data + (v452_data * v468_data));
          float v473_data = s0[53];
          float v475_data = ir1[4];
          ir1[4] = (v475_data + (v452_data * v473_data));
          float v478_data = s0[65];
          float v480_data = ir1[5];
          ir1[5] = (v480_data + (v452_data * v478_data));
          float v483_data = s0[77];
          float v485_data = ir1[6];
          ir1[6] = (v485_data + (v452_data * v483_data));
          float v488_data = s0[89];
          float v490_data = ir1[7];
          ir1[7] = (v490_data + (v452_data * v488_data));
          float v493_data = s0[101];
          float v495_data = ir1[8];
          ir1[8] = (v495_data + (v452_data * v493_data));
          float v498_data = s0[113];
          float v500_data = ir1[9];
          ir1[9] = (v500_data + (v452_data * v498_data));
          float v503_data = s0[125];
          float v505_data = ir1[10];
          ir1[10] = (v505_data + (v452_data * v503_data));
          float v508_data = s0[137];
          float v510_data = ir1[11];
          ir1[11] = (v510_data + (v452_data * v508_data));
          float v513_data = s0[149];
          float v515_data = ir1[12];
          ir1[12] = (v515_data + (v452_data * v513_data));
          float v518_data = s0[161];
          float v520_data = ir1[13];
          ir1[13] = (v520_data + (v452_data * v518_data));
          float v523_data = s0[173];
          float v525_data = ir1[14];
          ir1[14] = (v525_data + (v452_data * v523_data));
          float v528_data = s0[185];
          float v530_data = ir1[15];
          ir1[15] = (v530_data + (v452_data * v528_data));
          float v532_data = r0[6];
          float v533_data = s0[6];
          float v535_data = ir1[0];
          ir1[0] = (v535_data + (v532_data * v533_data));
          float v538_data = s0[18];
          float v540_data = ir1[1];
          ir1[1] = (v540_data + (v532_data * v538_data));
          float v543_data = s0[30];
          float v545_data = ir1[2];
          ir1[2] = (v545_data + (v532_data * v543_data));
          float v548_data = s0[42];
          float v550_data = ir1[3];
          ir1[3] = (v550_data + (v532_data * v548_data));
          float v553_data = s0[54];
          float v555_data = ir1[4];
          ir1[4] = (v555_data + (v532_data * v553_data));
          float v558_data = s0[66];
          float v560_data = ir1[5];
          ir1[5] = (v560_data + (v532_data * v558_data));
          float v563_data = s0[78];
          float v565_data = ir1[6];
          ir1[6] = (v565_data + (v532_data * v563_data));
          float v568_data = s0[90];
          float v570_data = ir1[7];
          ir1[7] = (v570_data + (v532_data * v568_data));
          float v573_data = s0[102];
          float v575_data = ir1[8];
          ir1[8] = (v575_data + (v532_data * v573_data));
          float v578_data = s0[114];
          float v580_data = ir1[9];
          ir1[9] = (v580_data + (v532_data * v578_data));
          float v583_data = s0[126];
          float v585_data = ir1[10];
          ir1[10] = (v585_data + (v532_data * v583_data));
          float v588_data = s0[138];
          float v590_data = ir1[11];
          ir1[11] = (v590_data + (v532_data * v588_data));
          float v593_data = s0[150];
          float v595_data = ir1[12];
          ir1[12] = (v595_data + (v532_data * v593_data));
          float v598_data = s0[162];
          float v600_data = ir1[13];
          ir1[13] = (v600_data + (v532_data * v598_data));
          float v603_data = s0[174];
          float v605_data = ir1[14];
          ir1[14] = (v605_data + (v532_data * v603_data));
          float v608_data = s0[186];
          float v610_data = ir1[15];
          ir1[15] = (v610_data + (v532_data * v608_data));
          float v612_data = r0[7];
          float v613_data = s0[7];
          float v615_data = ir1[0];
          ir1[0] = (v615_data + (v612_data * v613_data));
          float v618_data = s0[19];
          float v620_data = ir1[1];
          ir1[1] = (v620_data + (v612_data * v618_data));
          float v623_data = s0[31];
          float v625_data = ir1[2];
          ir1[2] = (v625_data + (v612_data * v623_data));
          float v628_data = s0[43];
          float v630_data = ir1[3];
          ir1[3] = (v630_data + (v612_data * v628_data));
          float v633_data = s0[55];
          float v635_data = ir1[4];
          ir1[4] = (v635_data + (v612_data * v633_data));
          float v638_data = s0[67];
          float v640_data = ir1[5];
          ir1[5] = (v640_data + (v612_data * v638_data));
          float v643_data = s0[79];
          float v645_data = ir1[6];
          ir1[6] = (v645_data + (v612_data * v643_data));
          float v648_data = s0[91];
          float v650_data = ir1[7];
          ir1[7] = (v650_data + (v612_data * v648_data));
          float v653_data = s0[103];
          float v655_data = ir1[8];
          ir1[8] = (v655_data + (v612_data * v653_data));
          float v658_data = s0[115];
          float v660_data = ir1[9];
          ir1[9] = (v660_data + (v612_data * v658_data));
          float v663_data = s0[127];
          float v665_data = ir1[10];
          ir1[10] = (v665_data + (v612_data * v663_data));
          float v668_data = s0[139];
          float v670_data = ir1[11];
          ir1[11] = (v670_data + (v612_data * v668_data));
          float v673_data = s0[151];
          float v675_data = ir1[12];
          ir1[12] = (v675_data + (v612_data * v673_data));
          float v678_data = s0[163];
          float v680_data = ir1[13];
          ir1[13] = (v680_data + (v612_data * v678_data));
          float v683_data = s0[175];
          float v685_data = ir1[14];
          ir1[14] = (v685_data + (v612_data * v683_data));
          float v688_data = s0[187];
          float v690_data = ir1[15];
          ir1[15] = (v690_data + (v612_data * v688_data));
          float v692_data = r0[8];
          float v693_data = s0[8];
          float v695_data = ir1[0];
          ir1[0] = (v695_data + (v692_data * v693_data));
          float v698_data = s0[20];
          float v700_data = ir1[1];
          ir1[1] = (v700_data + (v692_data * v698_data));
          float v703_data = s0[32];
          float v705_data = ir1[2];
          ir1[2] = (v705_data + (v692_data * v703_data));
          float v708_data = s0[44];
          float v710_data = ir1[3];
          ir1[3] = (v710_data + (v692_data * v708_data));
          float v713_data = s0[56];
          float v715_data = ir1[4];
          ir1[4] = (v715_data + (v692_data * v713_data));
          float v718_data = s0[68];
          float v720_data = ir1[5];
          ir1[5] = (v720_data + (v692_data * v718_data));
          float v723_data = s0[80];
          float v725_data = ir1[6];
          ir1[6] = (v725_data + (v692_data * v723_data));
          float v728_data = s0[92];
          float v730_data = ir1[7];
          ir1[7] = (v730_data + (v692_data * v728_data));
          float v733_data = s0[104];
          float v735_data = ir1[8];
          ir1[8] = (v735_data + (v692_data * v733_data));
          float v738_data = s0[116];
          float v740_data = ir1[9];
          ir1[9] = (v740_data + (v692_data * v738_data));
          float v743_data = s0[128];
          float v745_data = ir1[10];
          ir1[10] = (v745_data + (v692_data * v743_data));
          float v748_data = s0[140];
          float v750_data = ir1[11];
          ir1[11] = (v750_data + (v692_data * v748_data));
          float v753_data = s0[152];
          float v755_data = ir1[12];
          ir1[12] = (v755_data + (v692_data * v753_data));
          float v758_data = s0[164];
          float v760_data = ir1[13];
          ir1[13] = (v760_data + (v692_data * v758_data));
          float v763_data = s0[176];
          float v765_data = ir1[14];
          ir1[14] = (v765_data + (v692_data * v763_data));
          float v768_data = s0[188];
          float v770_data = ir1[15];
          ir1[15] = (v770_data + (v692_data * v768_data));
          float v772_data = r0[9];
          float v773_data = s0[9];
          float v775_data = ir1[0];
          ir1[0] = (v775_data + (v772_data * v773_data));
          float v778_data = s0[21];
          float v780_data = ir1[1];
          ir1[1] = (v780_data + (v772_data * v778_data));
          float v783_data = s0[33];
          float v785_data = ir1[2];
          ir1[2] = (v785_data + (v772_data * v783_data));
          float v788_data = s0[45];
          float v790_data = ir1[3];
          ir1[3] = (v790_data + (v772_data * v788_data));
          float v793_data = s0[57];
          float v795_data = ir1[4];
          ir1[4] = (v795_data + (v772_data * v793_data));
          float v798_data = s0[69];
          float v800_data = ir1[5];
          ir1[5] = (v800_data + (v772_data * v798_data));
          float v803_data = s0[81];
          float v805_data = ir1[6];
          ir1[6] = (v805_data + (v772_data * v803_data));
          float v808_data = s0[93];
          float v810_data = ir1[7];
          ir1[7] = (v810_data + (v772_data * v808_data));
          float v813_data = s0[105];
          float v815_data = ir1[8];
          ir1[8] = (v815_data + (v772_data * v813_data));
          float v818_data = s0[117];
          float v820_data = ir1[9];
          ir1[9] = (v820_data + (v772_data * v818_data));
          float v823_data = s0[129];
          float v825_data = ir1[10];
          ir1[10] = (v825_data + (v772_data * v823_data));
          float v828_data = s0[141];
          float v830_data = ir1[11];
          ir1[11] = (v830_data + (v772_data * v828_data));
          float v833_data = s0[153];
          float v835_data = ir1[12];
          ir1[12] = (v835_data + (v772_data * v833_data));
          float v838_data = s0[165];
          float v840_data = ir1[13];
          ir1[13] = (v840_data + (v772_data * v838_data));
          float v843_data = s0[177];
          float v845_data = ir1[14];
          ir1[14] = (v845_data + (v772_data * v843_data));
          float v848_data = s0[189];
          float v850_data = ir1[15];
          ir1[15] = (v850_data + (v772_data * v848_data));
          float v852_data = r0[10];
          float v853_data = s0[10];
          float v855_data = ir1[0];
          ir1[0] = (v855_data + (v852_data * v853_data));
          float v858_data = s0[22];
          float v860_data = ir1[1];
          ir1[1] = (v860_data + (v852_data * v858_data));
          float v863_data = s0[34];
          float v865_data = ir1[2];
          ir1[2] = (v865_data + (v852_data * v863_data));
          float v868_data = s0[46];
          float v870_data = ir1[3];
          ir1[3] = (v870_data + (v852_data * v868_data));
          float v873_data = s0[58];
          float v875_data = ir1[4];
          ir1[4] = (v875_data + (v852_data * v873_data));
          float v878_data = s0[70];
          float v880_data = ir1[5];
          ir1[5] = (v880_data + (v852_data * v878_data));
          float v883_data = s0[82];
          float v885_data = ir1[6];
          ir1[6] = (v885_data + (v852_data * v883_data));
          float v888_data = s0[94];
          float v890_data = ir1[7];
          ir1[7] = (v890_data + (v852_data * v888_data));
          float v893_data = s0[106];
          float v895_data = ir1[8];
          ir1[8] = (v895_data + (v852_data * v893_data));
          float v898_data = s0[118];
          float v900_data = ir1[9];
          ir1[9] = (v900_data + (v852_data * v898_data));
          float v903_data = s0[130];
          float v905_data = ir1[10];
          ir1[10] = (v905_data + (v852_data * v903_data));
          float v908_data = s0[142];
          float v910_data = ir1[11];
          ir1[11] = (v910_data + (v852_data * v908_data));
          float v913_data = s0[154];
          float v915_data = ir1[12];
          ir1[12] = (v915_data + (v852_data * v913_data));
          float v918_data = s0[166];
          float v920_data = ir1[13];
          ir1[13] = (v920_data + (v852_data * v918_data));
          float v923_data = s0[178];
          float v925_data = ir1[14];
          ir1[14] = (v925_data + (v852_data * v923_data));
          float v928_data = s0[190];
          float v930_data = ir1[15];
          ir1[15] = (v930_data + (v852_data * v928_data));
          float v932_data = r0[11];
          float v933_data = s0[11];
          float v935_data = ir1[0];
          ir1[0] = (v935_data + (v932_data * v933_data));
          float v938_data = s0[23];
          float v940_data = ir1[1];
          ir1[1] = (v940_data + (v932_data * v938_data));
          float v943_data = s0[35];
          float v945_data = ir1[2];
          ir1[2] = (v945_data + (v932_data * v943_data));
          float v948_data = s0[47];
          float v950_data = ir1[3];
          ir1[3] = (v950_data + (v932_data * v948_data));
          float v953_data = s0[59];
          float v955_data = ir1[4];
          ir1[4] = (v955_data + (v932_data * v953_data));
          float v958_data = s0[71];
          float v960_data = ir1[5];
          ir1[5] = (v960_data + (v932_data * v958_data));
          float v963_data = s0[83];
          float v965_data = ir1[6];
          ir1[6] = (v965_data + (v932_data * v963_data));
          float v968_data = s0[95];
          float v970_data = ir1[7];
          ir1[7] = (v970_data + (v932_data * v968_data));
          float v973_data = s0[107];
          float v975_data = ir1[8];
          ir1[8] = (v975_data + (v932_data * v973_data));
          float v978_data = s0[119];
          float v980_data = ir1[9];
          ir1[9] = (v980_data + (v932_data * v978_data));
          float v983_data = s0[131];
          float v985_data = ir1[10];
          ir1[10] = (v985_data + (v932_data * v983_data));
          float v988_data = s0[143];
          float v990_data = ir1[11];
          ir1[11] = (v990_data + (v932_data * v988_data));
          float v993_data = s0[155];
          float v995_data = ir1[12];
          ir1[12] = (v995_data + (v932_data * v993_data));
          float v998_data = s0[167];
          float v1000_data = ir1[13];
          ir1[13] = (v1000_data + (v932_data * v998_data));
          float v1003_data = s0[179];
          float v1005_data = ir1[14];
          ir1[14] = (v1005_data + (v932_data * v1003_data));
          float v1008_data = s0[191];
          float v1010_data = ir1[15];
          ir1[15] = (v1010_data + (v932_data * v1008_data));
          // r1 = ir1
          #pragma unroll
          for (int32_t v1012_n0 = 0; v1012_n0 < 1; ++v1012_n0) {
            #pragma unroll
            for (int32_t v1013_n1 = 0; v1013_n1 < 16; ++v1013_n1) {
              int32_t v1014_a = v1012_n0 + v1013_n1;
              float v1015_data = ir1[v1014_a];
              r1[v1014_a] = v1015_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v1016_i0 = 0; v1016_i0 < 1; ++v1016_i0) {
            int32_t v1021_lead = v31_lead + (v1016_i0 * 32);
            #pragma unroll
            for (int32_t v1017_i1 = 0; v1017_i1 < 16; ++v1017_i1) {
              float v1019_data = r1[(v1016_i0 + v1017_i1)];
              glb_m0[(v1021_lead + (v1017_i1 * 32))] = v1019_data;
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
          for (int32_t v1028_i0 = 0; v1028_i0 < 1; ++v1028_i0) {
            int32_t v1031_lead = v31_lead + (v1028_i0 * 32);
            #pragma unroll
            for (int32_t v1029_i1 = 0; v1029_i1 < 8; ++v1029_i1) {
              float v1034_data = glb_m0[(v1031_lead + (v1029_i1 * 32))];
              r3[(v1028_i0 + v1029_i1)] = v1034_data;
            }
          }
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          float r5[12]{};
          // r5 = load{g>r}(glb_m5);
          #pragma unroll
          for (int32_t v1037_i0 = 0; v1037_i0 < 1; ++v1037_i0) {
            int32_t v1040_lead = v31_lead + (v1037_i0 * 32);
            #pragma unroll
            for (int32_t v1038_i1 = 0; v1038_i1 < 12; ++v1038_i1) {
              float v1043_data = __ldcg(&glb_m5[(v1040_lead + (v1038_i1 * 32))]);
              r5[(v1037_i0 + v1038_i1)] = v1043_data;
            }
          }
          // wait(r3 = load{g>r}(glb_m0););
          float r4[8]{};
          __syncwarp();
          // ir4 = +(r2 * s1)
          // [(0, 32), (0, 8)] [(0, 12)]
          float ir4[8]{};
          float v1047_data = r2[0];
          float v1048_data = s1[0];
          float v1050_data = ir4[0];
          ir4[0] = (v1050_data + (v1047_data * v1048_data));
          float v1053_data = s1[12];
          float v1055_data = ir4[1];
          ir4[1] = (v1055_data + (v1047_data * v1053_data));
          float v1058_data = s1[24];
          float v1060_data = ir4[2];
          ir4[2] = (v1060_data + (v1047_data * v1058_data));
          float v1063_data = s1[36];
          float v1065_data = ir4[3];
          ir4[3] = (v1065_data + (v1047_data * v1063_data));
          float v1068_data = s1[48];
          float v1070_data = ir4[4];
          ir4[4] = (v1070_data + (v1047_data * v1068_data));
          float v1073_data = s1[60];
          float v1075_data = ir4[5];
          ir4[5] = (v1075_data + (v1047_data * v1073_data));
          float v1078_data = s1[72];
          float v1080_data = ir4[6];
          ir4[6] = (v1080_data + (v1047_data * v1078_data));
          float v1083_data = s1[84];
          float v1085_data = ir4[7];
          ir4[7] = (v1085_data + (v1047_data * v1083_data));
          float v1087_data = r2[1];
          float v1088_data = s1[1];
          float v1090_data = ir4[0];
          ir4[0] = (v1090_data + (v1087_data * v1088_data));
          float v1093_data = s1[13];
          float v1095_data = ir4[1];
          ir4[1] = (v1095_data + (v1087_data * v1093_data));
          float v1098_data = s1[25];
          float v1100_data = ir4[2];
          ir4[2] = (v1100_data + (v1087_data * v1098_data));
          float v1103_data = s1[37];
          float v1105_data = ir4[3];
          ir4[3] = (v1105_data + (v1087_data * v1103_data));
          float v1108_data = s1[49];
          float v1110_data = ir4[4];
          ir4[4] = (v1110_data + (v1087_data * v1108_data));
          float v1113_data = s1[61];
          float v1115_data = ir4[5];
          ir4[5] = (v1115_data + (v1087_data * v1113_data));
          float v1118_data = s1[73];
          float v1120_data = ir4[6];
          ir4[6] = (v1120_data + (v1087_data * v1118_data));
          float v1123_data = s1[85];
          float v1125_data = ir4[7];
          ir4[7] = (v1125_data + (v1087_data * v1123_data));
          float v1127_data = r2[2];
          float v1128_data = s1[2];
          float v1130_data = ir4[0];
          ir4[0] = (v1130_data + (v1127_data * v1128_data));
          float v1133_data = s1[14];
          float v1135_data = ir4[1];
          ir4[1] = (v1135_data + (v1127_data * v1133_data));
          float v1138_data = s1[26];
          float v1140_data = ir4[2];
          ir4[2] = (v1140_data + (v1127_data * v1138_data));
          float v1143_data = s1[38];
          float v1145_data = ir4[3];
          ir4[3] = (v1145_data + (v1127_data * v1143_data));
          float v1148_data = s1[50];
          float v1150_data = ir4[4];
          ir4[4] = (v1150_data + (v1127_data * v1148_data));
          float v1153_data = s1[62];
          float v1155_data = ir4[5];
          ir4[5] = (v1155_data + (v1127_data * v1153_data));
          float v1158_data = s1[74];
          float v1160_data = ir4[6];
          ir4[6] = (v1160_data + (v1127_data * v1158_data));
          float v1163_data = s1[86];
          float v1165_data = ir4[7];
          ir4[7] = (v1165_data + (v1127_data * v1163_data));
          float v1167_data = r2[3];
          float v1168_data = s1[3];
          float v1170_data = ir4[0];
          ir4[0] = (v1170_data + (v1167_data * v1168_data));
          float v1173_data = s1[15];
          float v1175_data = ir4[1];
          ir4[1] = (v1175_data + (v1167_data * v1173_data));
          float v1178_data = s1[27];
          float v1180_data = ir4[2];
          ir4[2] = (v1180_data + (v1167_data * v1178_data));
          float v1183_data = s1[39];
          float v1185_data = ir4[3];
          ir4[3] = (v1185_data + (v1167_data * v1183_data));
          float v1188_data = s1[51];
          float v1190_data = ir4[4];
          ir4[4] = (v1190_data + (v1167_data * v1188_data));
          float v1193_data = s1[63];
          float v1195_data = ir4[5];
          ir4[5] = (v1195_data + (v1167_data * v1193_data));
          float v1198_data = s1[75];
          float v1200_data = ir4[6];
          ir4[6] = (v1200_data + (v1167_data * v1198_data));
          float v1203_data = s1[87];
          float v1205_data = ir4[7];
          ir4[7] = (v1205_data + (v1167_data * v1203_data));
          float v1207_data = r2[4];
          float v1208_data = s1[4];
          float v1210_data = ir4[0];
          ir4[0] = (v1210_data + (v1207_data * v1208_data));
          float v1213_data = s1[16];
          float v1215_data = ir4[1];
          ir4[1] = (v1215_data + (v1207_data * v1213_data));
          float v1218_data = s1[28];
          float v1220_data = ir4[2];
          ir4[2] = (v1220_data + (v1207_data * v1218_data));
          float v1223_data = s1[40];
          float v1225_data = ir4[3];
          ir4[3] = (v1225_data + (v1207_data * v1223_data));
          float v1228_data = s1[52];
          float v1230_data = ir4[4];
          ir4[4] = (v1230_data + (v1207_data * v1228_data));
          float v1233_data = s1[64];
          float v1235_data = ir4[5];
          ir4[5] = (v1235_data + (v1207_data * v1233_data));
          float v1238_data = s1[76];
          float v1240_data = ir4[6];
          ir4[6] = (v1240_data + (v1207_data * v1238_data));
          float v1243_data = s1[88];
          float v1245_data = ir4[7];
          ir4[7] = (v1245_data + (v1207_data * v1243_data));
          float v1247_data = r2[5];
          float v1248_data = s1[5];
          float v1250_data = ir4[0];
          ir4[0] = (v1250_data + (v1247_data * v1248_data));
          float v1253_data = s1[17];
          float v1255_data = ir4[1];
          ir4[1] = (v1255_data + (v1247_data * v1253_data));
          float v1258_data = s1[29];
          float v1260_data = ir4[2];
          ir4[2] = (v1260_data + (v1247_data * v1258_data));
          float v1263_data = s1[41];
          float v1265_data = ir4[3];
          ir4[3] = (v1265_data + (v1247_data * v1263_data));
          float v1268_data = s1[53];
          float v1270_data = ir4[4];
          ir4[4] = (v1270_data + (v1247_data * v1268_data));
          float v1273_data = s1[65];
          float v1275_data = ir4[5];
          ir4[5] = (v1275_data + (v1247_data * v1273_data));
          float v1278_data = s1[77];
          float v1280_data = ir4[6];
          ir4[6] = (v1280_data + (v1247_data * v1278_data));
          float v1283_data = s1[89];
          float v1285_data = ir4[7];
          ir4[7] = (v1285_data + (v1247_data * v1283_data));
          float v1287_data = r2[6];
          float v1288_data = s1[6];
          float v1290_data = ir4[0];
          ir4[0] = (v1290_data + (v1287_data * v1288_data));
          float v1293_data = s1[18];
          float v1295_data = ir4[1];
          ir4[1] = (v1295_data + (v1287_data * v1293_data));
          float v1298_data = s1[30];
          float v1300_data = ir4[2];
          ir4[2] = (v1300_data + (v1287_data * v1298_data));
          float v1303_data = s1[42];
          float v1305_data = ir4[3];
          ir4[3] = (v1305_data + (v1287_data * v1303_data));
          float v1308_data = s1[54];
          float v1310_data = ir4[4];
          ir4[4] = (v1310_data + (v1287_data * v1308_data));
          float v1313_data = s1[66];
          float v1315_data = ir4[5];
          ir4[5] = (v1315_data + (v1287_data * v1313_data));
          float v1318_data = s1[78];
          float v1320_data = ir4[6];
          ir4[6] = (v1320_data + (v1287_data * v1318_data));
          float v1323_data = s1[90];
          float v1325_data = ir4[7];
          ir4[7] = (v1325_data + (v1287_data * v1323_data));
          float v1327_data = r2[7];
          float v1328_data = s1[7];
          float v1330_data = ir4[0];
          ir4[0] = (v1330_data + (v1327_data * v1328_data));
          float v1333_data = s1[19];
          float v1335_data = ir4[1];
          ir4[1] = (v1335_data + (v1327_data * v1333_data));
          float v1338_data = s1[31];
          float v1340_data = ir4[2];
          ir4[2] = (v1340_data + (v1327_data * v1338_data));
          float v1343_data = s1[43];
          float v1345_data = ir4[3];
          ir4[3] = (v1345_data + (v1327_data * v1343_data));
          float v1348_data = s1[55];
          float v1350_data = ir4[4];
          ir4[4] = (v1350_data + (v1327_data * v1348_data));
          float v1353_data = s1[67];
          float v1355_data = ir4[5];
          ir4[5] = (v1355_data + (v1327_data * v1353_data));
          float v1358_data = s1[79];
          float v1360_data = ir4[6];
          ir4[6] = (v1360_data + (v1327_data * v1358_data));
          float v1363_data = s1[91];
          float v1365_data = ir4[7];
          ir4[7] = (v1365_data + (v1327_data * v1363_data));
          float v1367_data = r2[8];
          float v1368_data = s1[8];
          float v1370_data = ir4[0];
          ir4[0] = (v1370_data + (v1367_data * v1368_data));
          float v1373_data = s1[20];
          float v1375_data = ir4[1];
          ir4[1] = (v1375_data + (v1367_data * v1373_data));
          float v1378_data = s1[32];
          float v1380_data = ir4[2];
          ir4[2] = (v1380_data + (v1367_data * v1378_data));
          float v1383_data = s1[44];
          float v1385_data = ir4[3];
          ir4[3] = (v1385_data + (v1367_data * v1383_data));
          float v1388_data = s1[56];
          float v1390_data = ir4[4];
          ir4[4] = (v1390_data + (v1367_data * v1388_data));
          float v1393_data = s1[68];
          float v1395_data = ir4[5];
          ir4[5] = (v1395_data + (v1367_data * v1393_data));
          float v1398_data = s1[80];
          float v1400_data = ir4[6];
          ir4[6] = (v1400_data + (v1367_data * v1398_data));
          float v1403_data = s1[92];
          float v1405_data = ir4[7];
          ir4[7] = (v1405_data + (v1367_data * v1403_data));
          float v1407_data = r2[9];
          float v1408_data = s1[9];
          float v1410_data = ir4[0];
          ir4[0] = (v1410_data + (v1407_data * v1408_data));
          float v1413_data = s1[21];
          float v1415_data = ir4[1];
          ir4[1] = (v1415_data + (v1407_data * v1413_data));
          float v1418_data = s1[33];
          float v1420_data = ir4[2];
          ir4[2] = (v1420_data + (v1407_data * v1418_data));
          float v1423_data = s1[45];
          float v1425_data = ir4[3];
          ir4[3] = (v1425_data + (v1407_data * v1423_data));
          float v1428_data = s1[57];
          float v1430_data = ir4[4];
          ir4[4] = (v1430_data + (v1407_data * v1428_data));
          float v1433_data = s1[69];
          float v1435_data = ir4[5];
          ir4[5] = (v1435_data + (v1407_data * v1433_data));
          float v1438_data = s1[81];
          float v1440_data = ir4[6];
          ir4[6] = (v1440_data + (v1407_data * v1438_data));
          float v1443_data = s1[93];
          float v1445_data = ir4[7];
          ir4[7] = (v1445_data + (v1407_data * v1443_data));
          float v1447_data = r2[10];
          float v1448_data = s1[10];
          float v1450_data = ir4[0];
          ir4[0] = (v1450_data + (v1447_data * v1448_data));
          float v1453_data = s1[22];
          float v1455_data = ir4[1];
          ir4[1] = (v1455_data + (v1447_data * v1453_data));
          float v1458_data = s1[34];
          float v1460_data = ir4[2];
          ir4[2] = (v1460_data + (v1447_data * v1458_data));
          float v1463_data = s1[46];
          float v1465_data = ir4[3];
          ir4[3] = (v1465_data + (v1447_data * v1463_data));
          float v1468_data = s1[58];
          float v1470_data = ir4[4];
          ir4[4] = (v1470_data + (v1447_data * v1468_data));
          float v1473_data = s1[70];
          float v1475_data = ir4[5];
          ir4[5] = (v1475_data + (v1447_data * v1473_data));
          float v1478_data = s1[82];
          float v1480_data = ir4[6];
          ir4[6] = (v1480_data + (v1447_data * v1478_data));
          float v1483_data = s1[94];
          float v1485_data = ir4[7];
          ir4[7] = (v1485_data + (v1447_data * v1483_data));
          float v1487_data = r2[11];
          float v1488_data = s1[11];
          float v1490_data = ir4[0];
          ir4[0] = (v1490_data + (v1487_data * v1488_data));
          float v1493_data = s1[23];
          float v1495_data = ir4[1];
          ir4[1] = (v1495_data + (v1487_data * v1493_data));
          float v1498_data = s1[35];
          float v1500_data = ir4[2];
          ir4[2] = (v1500_data + (v1487_data * v1498_data));
          float v1503_data = s1[47];
          float v1505_data = ir4[3];
          ir4[3] = (v1505_data + (v1487_data * v1503_data));
          float v1508_data = s1[59];
          float v1510_data = ir4[4];
          ir4[4] = (v1510_data + (v1487_data * v1508_data));
          float v1513_data = s1[71];
          float v1515_data = ir4[5];
          ir4[5] = (v1515_data + (v1487_data * v1513_data));
          float v1518_data = s1[83];
          float v1520_data = ir4[6];
          ir4[6] = (v1520_data + (v1487_data * v1518_data));
          float v1523_data = s1[95];
          float v1525_data = ir4[7];
          ir4[7] = (v1525_data + (v1487_data * v1523_data));
          // r4 = ir4 + r3
          #pragma unroll
          for (int32_t v1527_n0 = 0; v1527_n0 < 1; ++v1527_n0) {
            #pragma unroll
            for (int32_t v1528_n1 = 0; v1528_n1 < 8; ++v1528_n1) {
              int32_t v1529_a = v1527_n0 + v1528_n1;
              float v1530_data = ir4[v1529_a];
              float v1531_data = r3[v1529_a];
              r4[v1529_a] = (v1531_data + v1530_data);
            }
          }
          // glb_m0 = store{r>g}(r4);
          #pragma unroll
          for (int32_t v1533_i0 = 0; v1533_i0 < 1; ++v1533_i0) {
            int32_t v1538_lead = v31_lead + (v1533_i0 * 32);
            #pragma unroll
            for (int32_t v1534_i1 = 0; v1534_i1 < 8; ++v1534_i1) {
              float v1536_data = r4[(v1533_i0 + v1534_i1)];
              glb_m0[(v1538_lead + (v1534_i1 * 32))] = v1536_data;
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
          for (int32_t v1545_i0 = 0; v1545_i0 < 1; ++v1545_i0) {
            int32_t v1548_lead = v31_lead + (v1545_i0 * 32);
            #pragma unroll
            for (int32_t v1546_i1 = 0; v1546_i1 < 8; ++v1546_i1) {
              float v1552_data = glb_m0[(v1548_lead + ((v1546_i1 + 8) * 32))];
              r6[(v1545_i0 + v1546_i1)] = v1552_data;
            }
          }
          // wait(s2 = load{g>s}(glb_m6[0, 1]));
          __pipeline_wait_prior(0);
          // wait(r6 = load{g>r}(glb_m0););
          float r7[8]{};
          __syncwarp();
          // ir7 = +(r5 * s2)
          // [(0, 32), (0, 8)] [(0, 12)]
          float ir7[8]{};
          float v1556_data = r5[0];
          float v1557_data = s2[0];
          float v1559_data = ir7[0];
          ir7[0] = (v1559_data + (v1556_data * v1557_data));
          float v1562_data = s2[12];
          float v1564_data = ir7[1];
          ir7[1] = (v1564_data + (v1556_data * v1562_data));
          float v1567_data = s2[24];
          float v1569_data = ir7[2];
          ir7[2] = (v1569_data + (v1556_data * v1567_data));
          float v1572_data = s2[36];
          float v1574_data = ir7[3];
          ir7[3] = (v1574_data + (v1556_data * v1572_data));
          float v1577_data = s2[48];
          float v1579_data = ir7[4];
          ir7[4] = (v1579_data + (v1556_data * v1577_data));
          float v1582_data = s2[60];
          float v1584_data = ir7[5];
          ir7[5] = (v1584_data + (v1556_data * v1582_data));
          float v1587_data = s2[72];
          float v1589_data = ir7[6];
          ir7[6] = (v1589_data + (v1556_data * v1587_data));
          float v1592_data = s2[84];
          float v1594_data = ir7[7];
          ir7[7] = (v1594_data + (v1556_data * v1592_data));
          float v1596_data = r5[1];
          float v1597_data = s2[1];
          float v1599_data = ir7[0];
          ir7[0] = (v1599_data + (v1596_data * v1597_data));
          float v1602_data = s2[13];
          float v1604_data = ir7[1];
          ir7[1] = (v1604_data + (v1596_data * v1602_data));
          float v1607_data = s2[25];
          float v1609_data = ir7[2];
          ir7[2] = (v1609_data + (v1596_data * v1607_data));
          float v1612_data = s2[37];
          float v1614_data = ir7[3];
          ir7[3] = (v1614_data + (v1596_data * v1612_data));
          float v1617_data = s2[49];
          float v1619_data = ir7[4];
          ir7[4] = (v1619_data + (v1596_data * v1617_data));
          float v1622_data = s2[61];
          float v1624_data = ir7[5];
          ir7[5] = (v1624_data + (v1596_data * v1622_data));
          float v1627_data = s2[73];
          float v1629_data = ir7[6];
          ir7[6] = (v1629_data + (v1596_data * v1627_data));
          float v1632_data = s2[85];
          float v1634_data = ir7[7];
          ir7[7] = (v1634_data + (v1596_data * v1632_data));
          float v1636_data = r5[2];
          float v1637_data = s2[2];
          float v1639_data = ir7[0];
          ir7[0] = (v1639_data + (v1636_data * v1637_data));
          float v1642_data = s2[14];
          float v1644_data = ir7[1];
          ir7[1] = (v1644_data + (v1636_data * v1642_data));
          float v1647_data = s2[26];
          float v1649_data = ir7[2];
          ir7[2] = (v1649_data + (v1636_data * v1647_data));
          float v1652_data = s2[38];
          float v1654_data = ir7[3];
          ir7[3] = (v1654_data + (v1636_data * v1652_data));
          float v1657_data = s2[50];
          float v1659_data = ir7[4];
          ir7[4] = (v1659_data + (v1636_data * v1657_data));
          float v1662_data = s2[62];
          float v1664_data = ir7[5];
          ir7[5] = (v1664_data + (v1636_data * v1662_data));
          float v1667_data = s2[74];
          float v1669_data = ir7[6];
          ir7[6] = (v1669_data + (v1636_data * v1667_data));
          float v1672_data = s2[86];
          float v1674_data = ir7[7];
          ir7[7] = (v1674_data + (v1636_data * v1672_data));
          float v1676_data = r5[3];
          float v1677_data = s2[3];
          float v1679_data = ir7[0];
          ir7[0] = (v1679_data + (v1676_data * v1677_data));
          float v1682_data = s2[15];
          float v1684_data = ir7[1];
          ir7[1] = (v1684_data + (v1676_data * v1682_data));
          float v1687_data = s2[27];
          float v1689_data = ir7[2];
          ir7[2] = (v1689_data + (v1676_data * v1687_data));
          float v1692_data = s2[39];
          float v1694_data = ir7[3];
          ir7[3] = (v1694_data + (v1676_data * v1692_data));
          float v1697_data = s2[51];
          float v1699_data = ir7[4];
          ir7[4] = (v1699_data + (v1676_data * v1697_data));
          float v1702_data = s2[63];
          float v1704_data = ir7[5];
          ir7[5] = (v1704_data + (v1676_data * v1702_data));
          float v1707_data = s2[75];
          float v1709_data = ir7[6];
          ir7[6] = (v1709_data + (v1676_data * v1707_data));
          float v1712_data = s2[87];
          float v1714_data = ir7[7];
          ir7[7] = (v1714_data + (v1676_data * v1712_data));
          float v1716_data = r5[4];
          float v1717_data = s2[4];
          float v1719_data = ir7[0];
          ir7[0] = (v1719_data + (v1716_data * v1717_data));
          float v1722_data = s2[16];
          float v1724_data = ir7[1];
          ir7[1] = (v1724_data + (v1716_data * v1722_data));
          float v1727_data = s2[28];
          float v1729_data = ir7[2];
          ir7[2] = (v1729_data + (v1716_data * v1727_data));
          float v1732_data = s2[40];
          float v1734_data = ir7[3];
          ir7[3] = (v1734_data + (v1716_data * v1732_data));
          float v1737_data = s2[52];
          float v1739_data = ir7[4];
          ir7[4] = (v1739_data + (v1716_data * v1737_data));
          float v1742_data = s2[64];
          float v1744_data = ir7[5];
          ir7[5] = (v1744_data + (v1716_data * v1742_data));
          float v1747_data = s2[76];
          float v1749_data = ir7[6];
          ir7[6] = (v1749_data + (v1716_data * v1747_data));
          float v1752_data = s2[88];
          float v1754_data = ir7[7];
          ir7[7] = (v1754_data + (v1716_data * v1752_data));
          float v1756_data = r5[5];
          float v1757_data = s2[5];
          float v1759_data = ir7[0];
          ir7[0] = (v1759_data + (v1756_data * v1757_data));
          float v1762_data = s2[17];
          float v1764_data = ir7[1];
          ir7[1] = (v1764_data + (v1756_data * v1762_data));
          float v1767_data = s2[29];
          float v1769_data = ir7[2];
          ir7[2] = (v1769_data + (v1756_data * v1767_data));
          float v1772_data = s2[41];
          float v1774_data = ir7[3];
          ir7[3] = (v1774_data + (v1756_data * v1772_data));
          float v1777_data = s2[53];
          float v1779_data = ir7[4];
          ir7[4] = (v1779_data + (v1756_data * v1777_data));
          float v1782_data = s2[65];
          float v1784_data = ir7[5];
          ir7[5] = (v1784_data + (v1756_data * v1782_data));
          float v1787_data = s2[77];
          float v1789_data = ir7[6];
          ir7[6] = (v1789_data + (v1756_data * v1787_data));
          float v1792_data = s2[89];
          float v1794_data = ir7[7];
          ir7[7] = (v1794_data + (v1756_data * v1792_data));
          float v1796_data = r5[6];
          float v1797_data = s2[6];
          float v1799_data = ir7[0];
          ir7[0] = (v1799_data + (v1796_data * v1797_data));
          float v1802_data = s2[18];
          float v1804_data = ir7[1];
          ir7[1] = (v1804_data + (v1796_data * v1802_data));
          float v1807_data = s2[30];
          float v1809_data = ir7[2];
          ir7[2] = (v1809_data + (v1796_data * v1807_data));
          float v1812_data = s2[42];
          float v1814_data = ir7[3];
          ir7[3] = (v1814_data + (v1796_data * v1812_data));
          float v1817_data = s2[54];
          float v1819_data = ir7[4];
          ir7[4] = (v1819_data + (v1796_data * v1817_data));
          float v1822_data = s2[66];
          float v1824_data = ir7[5];
          ir7[5] = (v1824_data + (v1796_data * v1822_data));
          float v1827_data = s2[78];
          float v1829_data = ir7[6];
          ir7[6] = (v1829_data + (v1796_data * v1827_data));
          float v1832_data = s2[90];
          float v1834_data = ir7[7];
          ir7[7] = (v1834_data + (v1796_data * v1832_data));
          float v1836_data = r5[7];
          float v1837_data = s2[7];
          float v1839_data = ir7[0];
          ir7[0] = (v1839_data + (v1836_data * v1837_data));
          float v1842_data = s2[19];
          float v1844_data = ir7[1];
          ir7[1] = (v1844_data + (v1836_data * v1842_data));
          float v1847_data = s2[31];
          float v1849_data = ir7[2];
          ir7[2] = (v1849_data + (v1836_data * v1847_data));
          float v1852_data = s2[43];
          float v1854_data = ir7[3];
          ir7[3] = (v1854_data + (v1836_data * v1852_data));
          float v1857_data = s2[55];
          float v1859_data = ir7[4];
          ir7[4] = (v1859_data + (v1836_data * v1857_data));
          float v1862_data = s2[67];
          float v1864_data = ir7[5];
          ir7[5] = (v1864_data + (v1836_data * v1862_data));
          float v1867_data = s2[79];
          float v1869_data = ir7[6];
          ir7[6] = (v1869_data + (v1836_data * v1867_data));
          float v1872_data = s2[91];
          float v1874_data = ir7[7];
          ir7[7] = (v1874_data + (v1836_data * v1872_data));
          float v1876_data = r5[8];
          float v1877_data = s2[8];
          float v1879_data = ir7[0];
          ir7[0] = (v1879_data + (v1876_data * v1877_data));
          float v1882_data = s2[20];
          float v1884_data = ir7[1];
          ir7[1] = (v1884_data + (v1876_data * v1882_data));
          float v1887_data = s2[32];
          float v1889_data = ir7[2];
          ir7[2] = (v1889_data + (v1876_data * v1887_data));
          float v1892_data = s2[44];
          float v1894_data = ir7[3];
          ir7[3] = (v1894_data + (v1876_data * v1892_data));
          float v1897_data = s2[56];
          float v1899_data = ir7[4];
          ir7[4] = (v1899_data + (v1876_data * v1897_data));
          float v1902_data = s2[68];
          float v1904_data = ir7[5];
          ir7[5] = (v1904_data + (v1876_data * v1902_data));
          float v1907_data = s2[80];
          float v1909_data = ir7[6];
          ir7[6] = (v1909_data + (v1876_data * v1907_data));
          float v1912_data = s2[92];
          float v1914_data = ir7[7];
          ir7[7] = (v1914_data + (v1876_data * v1912_data));
          float v1916_data = r5[9];
          float v1917_data = s2[9];
          float v1919_data = ir7[0];
          ir7[0] = (v1919_data + (v1916_data * v1917_data));
          float v1922_data = s2[21];
          float v1924_data = ir7[1];
          ir7[1] = (v1924_data + (v1916_data * v1922_data));
          float v1927_data = s2[33];
          float v1929_data = ir7[2];
          ir7[2] = (v1929_data + (v1916_data * v1927_data));
          float v1932_data = s2[45];
          float v1934_data = ir7[3];
          ir7[3] = (v1934_data + (v1916_data * v1932_data));
          float v1937_data = s2[57];
          float v1939_data = ir7[4];
          ir7[4] = (v1939_data + (v1916_data * v1937_data));
          float v1942_data = s2[69];
          float v1944_data = ir7[5];
          ir7[5] = (v1944_data + (v1916_data * v1942_data));
          float v1947_data = s2[81];
          float v1949_data = ir7[6];
          ir7[6] = (v1949_data + (v1916_data * v1947_data));
          float v1952_data = s2[93];
          float v1954_data = ir7[7];
          ir7[7] = (v1954_data + (v1916_data * v1952_data));
          float v1956_data = r5[10];
          float v1957_data = s2[10];
          float v1959_data = ir7[0];
          ir7[0] = (v1959_data + (v1956_data * v1957_data));
          float v1962_data = s2[22];
          float v1964_data = ir7[1];
          ir7[1] = (v1964_data + (v1956_data * v1962_data));
          float v1967_data = s2[34];
          float v1969_data = ir7[2];
          ir7[2] = (v1969_data + (v1956_data * v1967_data));
          float v1972_data = s2[46];
          float v1974_data = ir7[3];
          ir7[3] = (v1974_data + (v1956_data * v1972_data));
          float v1977_data = s2[58];
          float v1979_data = ir7[4];
          ir7[4] = (v1979_data + (v1956_data * v1977_data));
          float v1982_data = s2[70];
          float v1984_data = ir7[5];
          ir7[5] = (v1984_data + (v1956_data * v1982_data));
          float v1987_data = s2[82];
          float v1989_data = ir7[6];
          ir7[6] = (v1989_data + (v1956_data * v1987_data));
          float v1992_data = s2[94];
          float v1994_data = ir7[7];
          ir7[7] = (v1994_data + (v1956_data * v1992_data));
          float v1996_data = r5[11];
          float v1997_data = s2[11];
          float v1999_data = ir7[0];
          ir7[0] = (v1999_data + (v1996_data * v1997_data));
          float v2002_data = s2[23];
          float v2004_data = ir7[1];
          ir7[1] = (v2004_data + (v1996_data * v2002_data));
          float v2007_data = s2[35];
          float v2009_data = ir7[2];
          ir7[2] = (v2009_data + (v1996_data * v2007_data));
          float v2012_data = s2[47];
          float v2014_data = ir7[3];
          ir7[3] = (v2014_data + (v1996_data * v2012_data));
          float v2017_data = s2[59];
          float v2019_data = ir7[4];
          ir7[4] = (v2019_data + (v1996_data * v2017_data));
          float v2022_data = s2[71];
          float v2024_data = ir7[5];
          ir7[5] = (v2024_data + (v1996_data * v2022_data));
          float v2027_data = s2[83];
          float v2029_data = ir7[6];
          ir7[6] = (v2029_data + (v1996_data * v2027_data));
          float v2032_data = s2[95];
          float v2034_data = ir7[7];
          ir7[7] = (v2034_data + (v1996_data * v2032_data));
          // r7 = ir7 + r6
          #pragma unroll
          for (int32_t v2036_n0 = 0; v2036_n0 < 1; ++v2036_n0) {
            #pragma unroll
            for (int32_t v2037_n1 = 0; v2037_n1 < 8; ++v2037_n1) {
              int32_t v2038_a = v2036_n0 + v2037_n1;
              float v2039_data = ir7[v2038_a];
              float v2040_data = r6[v2038_a];
              r7[v2038_a] = (v2040_data + v2039_data);
            }
          }
          // glb_m0 = store{r>g}(r7);
          #pragma unroll
          for (int32_t v2042_i0 = 0; v2042_i0 < 1; ++v2042_i0) {
            int32_t v2047_lead = v31_lead + (v2042_i0 * 32);
            #pragma unroll
            for (int32_t v2043_i1 = 0; v2043_i1 < 8; ++v2043_i1) {
              float v2045_data = r7[(v2042_i0 + v2043_i1)];
              glb_m0[(v2047_lead + ((v2043_i1 + 8) * 32))] = v2045_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

