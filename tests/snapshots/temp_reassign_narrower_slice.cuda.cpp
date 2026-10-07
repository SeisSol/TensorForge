// === base name ===
kernel_539a5bedd97fb832

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_539a5bedd97fb832 = {{16, 8, 1}, 16, 12, 1, 8, 5632, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_539a5bedd97fb832(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_539a5bedd97fb832(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_539a5bedd97fb832(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_539a5bedd97fb832, block.x * block.y * block.z, 1408 * sizeof(float));
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
void launcher_kernel_539a5bedd97fb832(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_539a5bedd97fb832(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_539a5bedd97fb832, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_539a5bedd97fb832<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_539a5bedd97fb832(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0) {
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
          float r2[12]{};
          // r2 = load{g>r}(glb_m2);
          if (v27_g) {
            #pragma unroll
            for (int32_t v768_i1 = 0; v768_i1 < 12; ++v768_i1) {
              float v773_data = __ldcg(&glb_m2[(v26_lead + (v768_i1 * 6))]);
              r2[v768_i1] = v773_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m1[0, 1]));
          __pipeline_wait_prior(0);
          float r1[12]{};
          // r1 = +(r0 * s0) + None
          // [(0, 6), (0, 12)] [(0, 12)]
          float v37_data = r0[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v38_data = s0[0];
          float v40_data = r1[0];
          r1[0] = (v40_data + (v37_data * v38_data));
          float v43_data = s0[12];
          float v45_data = r1[1];
          r1[1] = (v45_data + (v37_data * v43_data));
          float v48_data = s0[24];
          float v50_data = r1[2];
          r1[2] = (v50_data + (v37_data * v48_data));
          float v53_data = s0[36];
          float v55_data = r1[3];
          r1[3] = (v55_data + (v37_data * v53_data));
          float v58_data = s0[48];
          float v60_data = r1[4];
          r1[4] = (v60_data + (v37_data * v58_data));
          float v63_data = s0[60];
          float v65_data = r1[5];
          r1[5] = (v65_data + (v37_data * v63_data));
          float v68_data = s0[72];
          float v70_data = r1[6];
          r1[6] = (v70_data + (v37_data * v68_data));
          float v73_data = s0[84];
          float v75_data = r1[7];
          r1[7] = (v75_data + (v37_data * v73_data));
          float v78_data = s0[96];
          float v80_data = r1[8];
          r1[8] = (v80_data + (v37_data * v78_data));
          float v83_data = s0[108];
          float v85_data = r1[9];
          r1[9] = (v85_data + (v37_data * v83_data));
          float v88_data = s0[120];
          float v90_data = r1[10];
          r1[10] = (v90_data + (v37_data * v88_data));
          float v93_data = s0[132];
          float v95_data = r1[11];
          r1[11] = (v95_data + (v37_data * v93_data));
          float v97_data = r0[1];
          float v98_data = s0[1];
          float v100_data = r1[0];
          r1[0] = (v100_data + (v97_data * v98_data));
          float v103_data = s0[13];
          float v105_data = r1[1];
          r1[1] = (v105_data + (v97_data * v103_data));
          float v108_data = s0[25];
          float v110_data = r1[2];
          r1[2] = (v110_data + (v97_data * v108_data));
          float v113_data = s0[37];
          float v115_data = r1[3];
          r1[3] = (v115_data + (v97_data * v113_data));
          float v118_data = s0[49];
          float v120_data = r1[4];
          r1[4] = (v120_data + (v97_data * v118_data));
          float v123_data = s0[61];
          float v125_data = r1[5];
          r1[5] = (v125_data + (v97_data * v123_data));
          float v128_data = s0[73];
          float v130_data = r1[6];
          r1[6] = (v130_data + (v97_data * v128_data));
          float v133_data = s0[85];
          float v135_data = r1[7];
          r1[7] = (v135_data + (v97_data * v133_data));
          float v138_data = s0[97];
          float v140_data = r1[8];
          r1[8] = (v140_data + (v97_data * v138_data));
          float v143_data = s0[109];
          float v145_data = r1[9];
          r1[9] = (v145_data + (v97_data * v143_data));
          float v148_data = s0[121];
          float v150_data = r1[10];
          r1[10] = (v150_data + (v97_data * v148_data));
          float v153_data = s0[133];
          float v155_data = r1[11];
          r1[11] = (v155_data + (v97_data * v153_data));
          float v157_data = r0[2];
          float v158_data = s0[2];
          float v160_data = r1[0];
          r1[0] = (v160_data + (v157_data * v158_data));
          float v163_data = s0[14];
          float v165_data = r1[1];
          r1[1] = (v165_data + (v157_data * v163_data));
          float v168_data = s0[26];
          float v170_data = r1[2];
          r1[2] = (v170_data + (v157_data * v168_data));
          float v173_data = s0[38];
          float v175_data = r1[3];
          r1[3] = (v175_data + (v157_data * v173_data));
          float v178_data = s0[50];
          float v180_data = r1[4];
          r1[4] = (v180_data + (v157_data * v178_data));
          float v183_data = s0[62];
          float v185_data = r1[5];
          r1[5] = (v185_data + (v157_data * v183_data));
          float v188_data = s0[74];
          float v190_data = r1[6];
          r1[6] = (v190_data + (v157_data * v188_data));
          float v193_data = s0[86];
          float v195_data = r1[7];
          r1[7] = (v195_data + (v157_data * v193_data));
          float v198_data = s0[98];
          float v200_data = r1[8];
          r1[8] = (v200_data + (v157_data * v198_data));
          float v203_data = s0[110];
          float v205_data = r1[9];
          r1[9] = (v205_data + (v157_data * v203_data));
          float v208_data = s0[122];
          float v210_data = r1[10];
          r1[10] = (v210_data + (v157_data * v208_data));
          float v213_data = s0[134];
          float v215_data = r1[11];
          r1[11] = (v215_data + (v157_data * v213_data));
          float v217_data = r0[3];
          float v218_data = s0[3];
          float v220_data = r1[0];
          r1[0] = (v220_data + (v217_data * v218_data));
          float v223_data = s0[15];
          float v225_data = r1[1];
          r1[1] = (v225_data + (v217_data * v223_data));
          float v228_data = s0[27];
          float v230_data = r1[2];
          r1[2] = (v230_data + (v217_data * v228_data));
          float v233_data = s0[39];
          float v235_data = r1[3];
          r1[3] = (v235_data + (v217_data * v233_data));
          float v238_data = s0[51];
          float v240_data = r1[4];
          r1[4] = (v240_data + (v217_data * v238_data));
          float v243_data = s0[63];
          float v245_data = r1[5];
          r1[5] = (v245_data + (v217_data * v243_data));
          float v248_data = s0[75];
          float v250_data = r1[6];
          r1[6] = (v250_data + (v217_data * v248_data));
          float v253_data = s0[87];
          float v255_data = r1[7];
          r1[7] = (v255_data + (v217_data * v253_data));
          float v258_data = s0[99];
          float v260_data = r1[8];
          r1[8] = (v260_data + (v217_data * v258_data));
          float v263_data = s0[111];
          float v265_data = r1[9];
          r1[9] = (v265_data + (v217_data * v263_data));
          float v268_data = s0[123];
          float v270_data = r1[10];
          r1[10] = (v270_data + (v217_data * v268_data));
          float v273_data = s0[135];
          float v275_data = r1[11];
          r1[11] = (v275_data + (v217_data * v273_data));
          float v277_data = r0[4];
          float v278_data = s0[4];
          float v280_data = r1[0];
          r1[0] = (v280_data + (v277_data * v278_data));
          float v283_data = s0[16];
          float v285_data = r1[1];
          r1[1] = (v285_data + (v277_data * v283_data));
          float v288_data = s0[28];
          float v290_data = r1[2];
          r1[2] = (v290_data + (v277_data * v288_data));
          float v293_data = s0[40];
          float v295_data = r1[3];
          r1[3] = (v295_data + (v277_data * v293_data));
          float v298_data = s0[52];
          float v300_data = r1[4];
          r1[4] = (v300_data + (v277_data * v298_data));
          float v303_data = s0[64];
          float v305_data = r1[5];
          r1[5] = (v305_data + (v277_data * v303_data));
          float v308_data = s0[76];
          float v310_data = r1[6];
          r1[6] = (v310_data + (v277_data * v308_data));
          float v313_data = s0[88];
          float v315_data = r1[7];
          r1[7] = (v315_data + (v277_data * v313_data));
          float v318_data = s0[100];
          float v320_data = r1[8];
          r1[8] = (v320_data + (v277_data * v318_data));
          float v323_data = s0[112];
          float v325_data = r1[9];
          r1[9] = (v325_data + (v277_data * v323_data));
          float v328_data = s0[124];
          float v330_data = r1[10];
          r1[10] = (v330_data + (v277_data * v328_data));
          float v333_data = s0[136];
          float v335_data = r1[11];
          r1[11] = (v335_data + (v277_data * v333_data));
          float v337_data = r0[5];
          float v338_data = s0[5];
          float v340_data = r1[0];
          r1[0] = (v340_data + (v337_data * v338_data));
          float v343_data = s0[17];
          float v345_data = r1[1];
          r1[1] = (v345_data + (v337_data * v343_data));
          float v348_data = s0[29];
          float v350_data = r1[2];
          r1[2] = (v350_data + (v337_data * v348_data));
          float v353_data = s0[41];
          float v355_data = r1[3];
          r1[3] = (v355_data + (v337_data * v353_data));
          float v358_data = s0[53];
          float v360_data = r1[4];
          r1[4] = (v360_data + (v337_data * v358_data));
          float v363_data = s0[65];
          float v365_data = r1[5];
          r1[5] = (v365_data + (v337_data * v363_data));
          float v368_data = s0[77];
          float v370_data = r1[6];
          r1[6] = (v370_data + (v337_data * v368_data));
          float v373_data = s0[89];
          float v375_data = r1[7];
          r1[7] = (v375_data + (v337_data * v373_data));
          float v378_data = s0[101];
          float v380_data = r1[8];
          r1[8] = (v380_data + (v337_data * v378_data));
          float v383_data = s0[113];
          float v385_data = r1[9];
          r1[9] = (v385_data + (v337_data * v383_data));
          float v388_data = s0[125];
          float v390_data = r1[10];
          r1[10] = (v390_data + (v337_data * v388_data));
          float v393_data = s0[137];
          float v395_data = r1[11];
          r1[11] = (v395_data + (v337_data * v393_data));
          float v397_data = r0[6];
          float v398_data = s0[6];
          float v400_data = r1[0];
          r1[0] = (v400_data + (v397_data * v398_data));
          float v403_data = s0[18];
          float v405_data = r1[1];
          r1[1] = (v405_data + (v397_data * v403_data));
          float v408_data = s0[30];
          float v410_data = r1[2];
          r1[2] = (v410_data + (v397_data * v408_data));
          float v413_data = s0[42];
          float v415_data = r1[3];
          r1[3] = (v415_data + (v397_data * v413_data));
          float v418_data = s0[54];
          float v420_data = r1[4];
          r1[4] = (v420_data + (v397_data * v418_data));
          float v423_data = s0[66];
          float v425_data = r1[5];
          r1[5] = (v425_data + (v397_data * v423_data));
          float v428_data = s0[78];
          float v430_data = r1[6];
          r1[6] = (v430_data + (v397_data * v428_data));
          float v433_data = s0[90];
          float v435_data = r1[7];
          r1[7] = (v435_data + (v397_data * v433_data));
          float v438_data = s0[102];
          float v440_data = r1[8];
          r1[8] = (v440_data + (v397_data * v438_data));
          float v443_data = s0[114];
          float v445_data = r1[9];
          r1[9] = (v445_data + (v397_data * v443_data));
          float v448_data = s0[126];
          float v450_data = r1[10];
          r1[10] = (v450_data + (v397_data * v448_data));
          float v453_data = s0[138];
          float v455_data = r1[11];
          r1[11] = (v455_data + (v397_data * v453_data));
          float v457_data = r0[7];
          float v458_data = s0[7];
          float v460_data = r1[0];
          r1[0] = (v460_data + (v457_data * v458_data));
          float v463_data = s0[19];
          float v465_data = r1[1];
          r1[1] = (v465_data + (v457_data * v463_data));
          float v468_data = s0[31];
          float v470_data = r1[2];
          r1[2] = (v470_data + (v457_data * v468_data));
          float v473_data = s0[43];
          float v475_data = r1[3];
          r1[3] = (v475_data + (v457_data * v473_data));
          float v478_data = s0[55];
          float v480_data = r1[4];
          r1[4] = (v480_data + (v457_data * v478_data));
          float v483_data = s0[67];
          float v485_data = r1[5];
          r1[5] = (v485_data + (v457_data * v483_data));
          float v488_data = s0[79];
          float v490_data = r1[6];
          r1[6] = (v490_data + (v457_data * v488_data));
          float v493_data = s0[91];
          float v495_data = r1[7];
          r1[7] = (v495_data + (v457_data * v493_data));
          float v498_data = s0[103];
          float v500_data = r1[8];
          r1[8] = (v500_data + (v457_data * v498_data));
          float v503_data = s0[115];
          float v505_data = r1[9];
          r1[9] = (v505_data + (v457_data * v503_data));
          float v508_data = s0[127];
          float v510_data = r1[10];
          r1[10] = (v510_data + (v457_data * v508_data));
          float v513_data = s0[139];
          float v515_data = r1[11];
          r1[11] = (v515_data + (v457_data * v513_data));
          float v517_data = r0[8];
          float v518_data = s0[8];
          float v520_data = r1[0];
          r1[0] = (v520_data + (v517_data * v518_data));
          float v523_data = s0[20];
          float v525_data = r1[1];
          r1[1] = (v525_data + (v517_data * v523_data));
          float v528_data = s0[32];
          float v530_data = r1[2];
          r1[2] = (v530_data + (v517_data * v528_data));
          float v533_data = s0[44];
          float v535_data = r1[3];
          r1[3] = (v535_data + (v517_data * v533_data));
          float v538_data = s0[56];
          float v540_data = r1[4];
          r1[4] = (v540_data + (v517_data * v538_data));
          float v543_data = s0[68];
          float v545_data = r1[5];
          r1[5] = (v545_data + (v517_data * v543_data));
          float v548_data = s0[80];
          float v550_data = r1[6];
          r1[6] = (v550_data + (v517_data * v548_data));
          float v553_data = s0[92];
          float v555_data = r1[7];
          r1[7] = (v555_data + (v517_data * v553_data));
          float v558_data = s0[104];
          float v560_data = r1[8];
          r1[8] = (v560_data + (v517_data * v558_data));
          float v563_data = s0[116];
          float v565_data = r1[9];
          r1[9] = (v565_data + (v517_data * v563_data));
          float v568_data = s0[128];
          float v570_data = r1[10];
          r1[10] = (v570_data + (v517_data * v568_data));
          float v573_data = s0[140];
          float v575_data = r1[11];
          r1[11] = (v575_data + (v517_data * v573_data));
          float v577_data = r0[9];
          float v578_data = s0[9];
          float v580_data = r1[0];
          r1[0] = (v580_data + (v577_data * v578_data));
          float v583_data = s0[21];
          float v585_data = r1[1];
          r1[1] = (v585_data + (v577_data * v583_data));
          float v588_data = s0[33];
          float v590_data = r1[2];
          r1[2] = (v590_data + (v577_data * v588_data));
          float v593_data = s0[45];
          float v595_data = r1[3];
          r1[3] = (v595_data + (v577_data * v593_data));
          float v598_data = s0[57];
          float v600_data = r1[4];
          r1[4] = (v600_data + (v577_data * v598_data));
          float v603_data = s0[69];
          float v605_data = r1[5];
          r1[5] = (v605_data + (v577_data * v603_data));
          float v608_data = s0[81];
          float v610_data = r1[6];
          r1[6] = (v610_data + (v577_data * v608_data));
          float v613_data = s0[93];
          float v615_data = r1[7];
          r1[7] = (v615_data + (v577_data * v613_data));
          float v618_data = s0[105];
          float v620_data = r1[8];
          r1[8] = (v620_data + (v577_data * v618_data));
          float v623_data = s0[117];
          float v625_data = r1[9];
          r1[9] = (v625_data + (v577_data * v623_data));
          float v628_data = s0[129];
          float v630_data = r1[10];
          r1[10] = (v630_data + (v577_data * v628_data));
          float v633_data = s0[141];
          float v635_data = r1[11];
          r1[11] = (v635_data + (v577_data * v633_data));
          float v637_data = r0[10];
          float v638_data = s0[10];
          float v640_data = r1[0];
          r1[0] = (v640_data + (v637_data * v638_data));
          float v643_data = s0[22];
          float v645_data = r1[1];
          r1[1] = (v645_data + (v637_data * v643_data));
          float v648_data = s0[34];
          float v650_data = r1[2];
          r1[2] = (v650_data + (v637_data * v648_data));
          float v653_data = s0[46];
          float v655_data = r1[3];
          r1[3] = (v655_data + (v637_data * v653_data));
          float v658_data = s0[58];
          float v660_data = r1[4];
          r1[4] = (v660_data + (v637_data * v658_data));
          float v663_data = s0[70];
          float v665_data = r1[5];
          r1[5] = (v665_data + (v637_data * v663_data));
          float v668_data = s0[82];
          float v670_data = r1[6];
          r1[6] = (v670_data + (v637_data * v668_data));
          float v673_data = s0[94];
          float v675_data = r1[7];
          r1[7] = (v675_data + (v637_data * v673_data));
          float v678_data = s0[106];
          float v680_data = r1[8];
          r1[8] = (v680_data + (v637_data * v678_data));
          float v683_data = s0[118];
          float v685_data = r1[9];
          r1[9] = (v685_data + (v637_data * v683_data));
          float v688_data = s0[130];
          float v690_data = r1[10];
          r1[10] = (v690_data + (v637_data * v688_data));
          float v693_data = s0[142];
          float v695_data = r1[11];
          r1[11] = (v695_data + (v637_data * v693_data));
          float v697_data = r0[11];
          float v698_data = s0[11];
          float v700_data = r1[0];
          r1[0] = (v700_data + (v697_data * v698_data));
          float v703_data = s0[23];
          float v705_data = r1[1];
          r1[1] = (v705_data + (v697_data * v703_data));
          float v708_data = s0[35];
          float v710_data = r1[2];
          r1[2] = (v710_data + (v697_data * v708_data));
          float v713_data = s0[47];
          float v715_data = r1[3];
          r1[3] = (v715_data + (v697_data * v713_data));
          float v718_data = s0[59];
          float v720_data = r1[4];
          r1[4] = (v720_data + (v697_data * v718_data));
          float v723_data = s0[71];
          float v725_data = r1[5];
          r1[5] = (v725_data + (v697_data * v723_data));
          float v728_data = s0[83];
          float v730_data = r1[6];
          r1[6] = (v730_data + (v697_data * v728_data));
          float v733_data = s0[95];
          float v735_data = r1[7];
          r1[7] = (v735_data + (v697_data * v733_data));
          float v738_data = s0[107];
          float v740_data = r1[8];
          r1[8] = (v740_data + (v697_data * v738_data));
          float v743_data = s0[119];
          float v745_data = r1[9];
          r1[9] = (v745_data + (v697_data * v743_data));
          float v748_data = s0[131];
          float v750_data = r1[10];
          r1[10] = (v750_data + (v697_data * v748_data));
          float v753_data = s0[143];
          float v755_data = r1[11];
          r1[11] = (v755_data + (v697_data * v753_data));
          // s1 = store{r>s}(localShrMem0, r1);
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          if (v27_g) {
            #pragma unroll
            for (int32_t v757_i1 = 0; v757_i1 < 12; ++v757_i1) {
              float v759_data = r1[v757_i1];
              int32_t v763_a = v26_lead + (v757_i1 * 12);
              s1[(v763_a ^ ((v763_a >> 4) & 15))] = v759_data;
            }
          }
          float r5[12]{};
          // r5 = load{g>r}(glb_m4);
          bool v1613_g = v26_lead < 2;
          if (v1613_g) {
            #pragma unroll
            for (int32_t v1614_i1 = 0; v1614_i1 < 12; ++v1614_i1) {
              float v1619_data = __ldcg(&glb_m4[(v26_lead + (v1614_i1 * 2))]);
              r5[v1614_i1] = v1619_data;
            }
          }
          float r3[12]{};
          // ir3 = +(r2 * s0)
          // [(0, 6), (0, 12)] [(0, 12)]
          float ir3[12]{};
          float v777_data = r2[0];
          float v780_data = ir3[0];
          ir3[0] = (v780_data + (v777_data * v38_data));
          float v785_data = ir3[1];
          ir3[1] = (v785_data + (v777_data * v43_data));
          float v790_data = ir3[2];
          ir3[2] = (v790_data + (v777_data * v48_data));
          float v795_data = ir3[3];
          ir3[3] = (v795_data + (v777_data * v53_data));
          float v800_data = ir3[4];
          ir3[4] = (v800_data + (v777_data * v58_data));
          float v805_data = ir3[5];
          ir3[5] = (v805_data + (v777_data * v63_data));
          float v810_data = ir3[6];
          ir3[6] = (v810_data + (v777_data * v68_data));
          float v815_data = ir3[7];
          ir3[7] = (v815_data + (v777_data * v73_data));
          float v820_data = ir3[8];
          ir3[8] = (v820_data + (v777_data * v78_data));
          float v825_data = ir3[9];
          ir3[9] = (v825_data + (v777_data * v83_data));
          float v830_data = ir3[10];
          ir3[10] = (v830_data + (v777_data * v88_data));
          float v835_data = ir3[11];
          ir3[11] = (v835_data + (v777_data * v93_data));
          float v837_data = r2[1];
          float v840_data = ir3[0];
          ir3[0] = (v840_data + (v837_data * v98_data));
          float v845_data = ir3[1];
          ir3[1] = (v845_data + (v837_data * v103_data));
          float v850_data = ir3[2];
          ir3[2] = (v850_data + (v837_data * v108_data));
          float v855_data = ir3[3];
          ir3[3] = (v855_data + (v837_data * v113_data));
          float v860_data = ir3[4];
          ir3[4] = (v860_data + (v837_data * v118_data));
          float v865_data = ir3[5];
          ir3[5] = (v865_data + (v837_data * v123_data));
          float v870_data = ir3[6];
          ir3[6] = (v870_data + (v837_data * v128_data));
          float v875_data = ir3[7];
          ir3[7] = (v875_data + (v837_data * v133_data));
          float v880_data = ir3[8];
          ir3[8] = (v880_data + (v837_data * v138_data));
          float v885_data = ir3[9];
          ir3[9] = (v885_data + (v837_data * v143_data));
          float v890_data = ir3[10];
          ir3[10] = (v890_data + (v837_data * v148_data));
          float v895_data = ir3[11];
          ir3[11] = (v895_data + (v837_data * v153_data));
          float v897_data = r2[2];
          float v900_data = ir3[0];
          ir3[0] = (v900_data + (v897_data * v158_data));
          float v905_data = ir3[1];
          ir3[1] = (v905_data + (v897_data * v163_data));
          float v910_data = ir3[2];
          ir3[2] = (v910_data + (v897_data * v168_data));
          float v915_data = ir3[3];
          ir3[3] = (v915_data + (v897_data * v173_data));
          float v920_data = ir3[4];
          ir3[4] = (v920_data + (v897_data * v178_data));
          float v925_data = ir3[5];
          ir3[5] = (v925_data + (v897_data * v183_data));
          float v930_data = ir3[6];
          ir3[6] = (v930_data + (v897_data * v188_data));
          float v935_data = ir3[7];
          ir3[7] = (v935_data + (v897_data * v193_data));
          float v940_data = ir3[8];
          ir3[8] = (v940_data + (v897_data * v198_data));
          float v945_data = ir3[9];
          ir3[9] = (v945_data + (v897_data * v203_data));
          float v950_data = ir3[10];
          ir3[10] = (v950_data + (v897_data * v208_data));
          float v955_data = ir3[11];
          ir3[11] = (v955_data + (v897_data * v213_data));
          float v957_data = r2[3];
          float v960_data = ir3[0];
          ir3[0] = (v960_data + (v957_data * v218_data));
          float v965_data = ir3[1];
          ir3[1] = (v965_data + (v957_data * v223_data));
          float v970_data = ir3[2];
          ir3[2] = (v970_data + (v957_data * v228_data));
          float v975_data = ir3[3];
          ir3[3] = (v975_data + (v957_data * v233_data));
          float v980_data = ir3[4];
          ir3[4] = (v980_data + (v957_data * v238_data));
          float v985_data = ir3[5];
          ir3[5] = (v985_data + (v957_data * v243_data));
          float v990_data = ir3[6];
          ir3[6] = (v990_data + (v957_data * v248_data));
          float v995_data = ir3[7];
          ir3[7] = (v995_data + (v957_data * v253_data));
          float v1000_data = ir3[8];
          ir3[8] = (v1000_data + (v957_data * v258_data));
          float v1005_data = ir3[9];
          ir3[9] = (v1005_data + (v957_data * v263_data));
          float v1010_data = ir3[10];
          ir3[10] = (v1010_data + (v957_data * v268_data));
          float v1015_data = ir3[11];
          ir3[11] = (v1015_data + (v957_data * v273_data));
          float v1017_data = r2[4];
          float v1020_data = ir3[0];
          ir3[0] = (v1020_data + (v1017_data * v278_data));
          float v1025_data = ir3[1];
          ir3[1] = (v1025_data + (v1017_data * v283_data));
          float v1030_data = ir3[2];
          ir3[2] = (v1030_data + (v1017_data * v288_data));
          float v1035_data = ir3[3];
          ir3[3] = (v1035_data + (v1017_data * v293_data));
          float v1040_data = ir3[4];
          ir3[4] = (v1040_data + (v1017_data * v298_data));
          float v1045_data = ir3[5];
          ir3[5] = (v1045_data + (v1017_data * v303_data));
          float v1050_data = ir3[6];
          ir3[6] = (v1050_data + (v1017_data * v308_data));
          float v1055_data = ir3[7];
          ir3[7] = (v1055_data + (v1017_data * v313_data));
          float v1060_data = ir3[8];
          ir3[8] = (v1060_data + (v1017_data * v318_data));
          float v1065_data = ir3[9];
          ir3[9] = (v1065_data + (v1017_data * v323_data));
          float v1070_data = ir3[10];
          ir3[10] = (v1070_data + (v1017_data * v328_data));
          float v1075_data = ir3[11];
          ir3[11] = (v1075_data + (v1017_data * v333_data));
          float v1077_data = r2[5];
          float v1080_data = ir3[0];
          ir3[0] = (v1080_data + (v1077_data * v338_data));
          float v1085_data = ir3[1];
          ir3[1] = (v1085_data + (v1077_data * v343_data));
          float v1090_data = ir3[2];
          ir3[2] = (v1090_data + (v1077_data * v348_data));
          float v1095_data = ir3[3];
          ir3[3] = (v1095_data + (v1077_data * v353_data));
          float v1100_data = ir3[4];
          ir3[4] = (v1100_data + (v1077_data * v358_data));
          float v1105_data = ir3[5];
          ir3[5] = (v1105_data + (v1077_data * v363_data));
          float v1110_data = ir3[6];
          ir3[6] = (v1110_data + (v1077_data * v368_data));
          float v1115_data = ir3[7];
          ir3[7] = (v1115_data + (v1077_data * v373_data));
          float v1120_data = ir3[8];
          ir3[8] = (v1120_data + (v1077_data * v378_data));
          float v1125_data = ir3[9];
          ir3[9] = (v1125_data + (v1077_data * v383_data));
          float v1130_data = ir3[10];
          ir3[10] = (v1130_data + (v1077_data * v388_data));
          float v1135_data = ir3[11];
          ir3[11] = (v1135_data + (v1077_data * v393_data));
          float v1137_data = r2[6];
          float v1140_data = ir3[0];
          ir3[0] = (v1140_data + (v1137_data * v398_data));
          float v1145_data = ir3[1];
          ir3[1] = (v1145_data + (v1137_data * v403_data));
          float v1150_data = ir3[2];
          ir3[2] = (v1150_data + (v1137_data * v408_data));
          float v1155_data = ir3[3];
          ir3[3] = (v1155_data + (v1137_data * v413_data));
          float v1160_data = ir3[4];
          ir3[4] = (v1160_data + (v1137_data * v418_data));
          float v1165_data = ir3[5];
          ir3[5] = (v1165_data + (v1137_data * v423_data));
          float v1170_data = ir3[6];
          ir3[6] = (v1170_data + (v1137_data * v428_data));
          float v1175_data = ir3[7];
          ir3[7] = (v1175_data + (v1137_data * v433_data));
          float v1180_data = ir3[8];
          ir3[8] = (v1180_data + (v1137_data * v438_data));
          float v1185_data = ir3[9];
          ir3[9] = (v1185_data + (v1137_data * v443_data));
          float v1190_data = ir3[10];
          ir3[10] = (v1190_data + (v1137_data * v448_data));
          float v1195_data = ir3[11];
          ir3[11] = (v1195_data + (v1137_data * v453_data));
          float v1197_data = r2[7];
          float v1200_data = ir3[0];
          ir3[0] = (v1200_data + (v1197_data * v458_data));
          float v1205_data = ir3[1];
          ir3[1] = (v1205_data + (v1197_data * v463_data));
          float v1210_data = ir3[2];
          ir3[2] = (v1210_data + (v1197_data * v468_data));
          float v1215_data = ir3[3];
          ir3[3] = (v1215_data + (v1197_data * v473_data));
          float v1220_data = ir3[4];
          ir3[4] = (v1220_data + (v1197_data * v478_data));
          float v1225_data = ir3[5];
          ir3[5] = (v1225_data + (v1197_data * v483_data));
          float v1230_data = ir3[6];
          ir3[6] = (v1230_data + (v1197_data * v488_data));
          float v1235_data = ir3[7];
          ir3[7] = (v1235_data + (v1197_data * v493_data));
          float v1240_data = ir3[8];
          ir3[8] = (v1240_data + (v1197_data * v498_data));
          float v1245_data = ir3[9];
          ir3[9] = (v1245_data + (v1197_data * v503_data));
          float v1250_data = ir3[10];
          ir3[10] = (v1250_data + (v1197_data * v508_data));
          float v1255_data = ir3[11];
          ir3[11] = (v1255_data + (v1197_data * v513_data));
          float v1257_data = r2[8];
          float v1260_data = ir3[0];
          ir3[0] = (v1260_data + (v1257_data * v518_data));
          float v1265_data = ir3[1];
          ir3[1] = (v1265_data + (v1257_data * v523_data));
          float v1270_data = ir3[2];
          ir3[2] = (v1270_data + (v1257_data * v528_data));
          float v1275_data = ir3[3];
          ir3[3] = (v1275_data + (v1257_data * v533_data));
          float v1280_data = ir3[4];
          ir3[4] = (v1280_data + (v1257_data * v538_data));
          float v1285_data = ir3[5];
          ir3[5] = (v1285_data + (v1257_data * v543_data));
          float v1290_data = ir3[6];
          ir3[6] = (v1290_data + (v1257_data * v548_data));
          float v1295_data = ir3[7];
          ir3[7] = (v1295_data + (v1257_data * v553_data));
          float v1300_data = ir3[8];
          ir3[8] = (v1300_data + (v1257_data * v558_data));
          float v1305_data = ir3[9];
          ir3[9] = (v1305_data + (v1257_data * v563_data));
          float v1310_data = ir3[10];
          ir3[10] = (v1310_data + (v1257_data * v568_data));
          float v1315_data = ir3[11];
          ir3[11] = (v1315_data + (v1257_data * v573_data));
          float v1317_data = r2[9];
          float v1320_data = ir3[0];
          ir3[0] = (v1320_data + (v1317_data * v578_data));
          float v1325_data = ir3[1];
          ir3[1] = (v1325_data + (v1317_data * v583_data));
          float v1330_data = ir3[2];
          ir3[2] = (v1330_data + (v1317_data * v588_data));
          float v1335_data = ir3[3];
          ir3[3] = (v1335_data + (v1317_data * v593_data));
          float v1340_data = ir3[4];
          ir3[4] = (v1340_data + (v1317_data * v598_data));
          float v1345_data = ir3[5];
          ir3[5] = (v1345_data + (v1317_data * v603_data));
          float v1350_data = ir3[6];
          ir3[6] = (v1350_data + (v1317_data * v608_data));
          float v1355_data = ir3[7];
          ir3[7] = (v1355_data + (v1317_data * v613_data));
          float v1360_data = ir3[8];
          ir3[8] = (v1360_data + (v1317_data * v618_data));
          float v1365_data = ir3[9];
          ir3[9] = (v1365_data + (v1317_data * v623_data));
          float v1370_data = ir3[10];
          ir3[10] = (v1370_data + (v1317_data * v628_data));
          float v1375_data = ir3[11];
          ir3[11] = (v1375_data + (v1317_data * v633_data));
          float v1377_data = r2[10];
          float v1380_data = ir3[0];
          ir3[0] = (v1380_data + (v1377_data * v638_data));
          float v1385_data = ir3[1];
          ir3[1] = (v1385_data + (v1377_data * v643_data));
          float v1390_data = ir3[2];
          ir3[2] = (v1390_data + (v1377_data * v648_data));
          float v1395_data = ir3[3];
          ir3[3] = (v1395_data + (v1377_data * v653_data));
          float v1400_data = ir3[4];
          ir3[4] = (v1400_data + (v1377_data * v658_data));
          float v1405_data = ir3[5];
          ir3[5] = (v1405_data + (v1377_data * v663_data));
          float v1410_data = ir3[6];
          ir3[6] = (v1410_data + (v1377_data * v668_data));
          float v1415_data = ir3[7];
          ir3[7] = (v1415_data + (v1377_data * v673_data));
          float v1420_data = ir3[8];
          ir3[8] = (v1420_data + (v1377_data * v678_data));
          float v1425_data = ir3[9];
          ir3[9] = (v1425_data + (v1377_data * v683_data));
          float v1430_data = ir3[10];
          ir3[10] = (v1430_data + (v1377_data * v688_data));
          float v1435_data = ir3[11];
          ir3[11] = (v1435_data + (v1377_data * v693_data));
          float v1437_data = r2[11];
          float v1440_data = ir3[0];
          ir3[0] = (v1440_data + (v1437_data * v698_data));
          float v1445_data = ir3[1];
          ir3[1] = (v1445_data + (v1437_data * v703_data));
          float v1450_data = ir3[2];
          ir3[2] = (v1450_data + (v1437_data * v708_data));
          float v1455_data = ir3[3];
          ir3[3] = (v1455_data + (v1437_data * v713_data));
          float v1460_data = ir3[4];
          ir3[4] = (v1460_data + (v1437_data * v718_data));
          float v1465_data = ir3[5];
          ir3[5] = (v1465_data + (v1437_data * v723_data));
          float v1470_data = ir3[6];
          ir3[6] = (v1470_data + (v1437_data * v728_data));
          float v1475_data = ir3[7];
          ir3[7] = (v1475_data + (v1437_data * v733_data));
          float v1480_data = ir3[8];
          ir3[8] = (v1480_data + (v1437_data * v738_data));
          float v1485_data = ir3[9];
          ir3[9] = (v1485_data + (v1437_data * v743_data));
          float v1490_data = ir3[10];
          ir3[10] = (v1490_data + (v1437_data * v748_data));
          float v1495_data = ir3[11];
          ir3[11] = (v1495_data + (v1437_data * v753_data));
          // r3 = ir3
          if (v27_g) {
            #pragma unroll
            for (int32_t v1497_n1 = 0; v1497_n1 < 12; ++v1497_n1) {
              float v1499_data = ir3[v1497_n1];
              r3[v1497_n1] = v1499_data;
            }
          }
          // s1 = store{r>s}(localShrMem0, r3);
          if (v27_g) {
            int32_t v1505_off = v26_lead + 6;
            #pragma unroll
            for (int32_t v1500_i1 = 0; v1500_i1 < 12; ++v1500_i1) {
              float v1502_data = r3[v1500_i1];
              int32_t v1507_a = v1505_off + (v1500_i1 * 12);
              s1[(v1507_a ^ ((v1507_a >> 4) & 15))] = v1502_data;
            }
          }
          float r4[12]{};
          // ir4 = +(s1)
          // [(0, 12), (0, 12)] []
          float ir4[12]{};
          bool v1516_g = v26_lead < 12;
          int32_t v1518_sw = (v26_lead >> 4) & 15;
          int32_t v1519_sw = v26_lead ^ v1518_sw;
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v1520_data = v1516_g ? (s1[v1519_sw]) : (0.0f);
          float v1521_data = ir4[0];
          ir4[0] = (v1521_data + v1520_data);
          int32_t v1523_a = v26_lead + 12;
          int32_t v1524_sw = v1523_a >> 4;
          float v1527_data = v1516_g ? (s1[(v1523_a ^ (v1524_sw & 15))]) : (0.0f);
          float v1528_data = ir4[1];
          ir4[1] = (v1528_data + v1527_data);
          int32_t v1530_a = v26_lead + 24;
          int32_t v1531_sw = v1530_a >> 4;
          float v1534_data = v1516_g ? (s1[(v1530_a ^ (v1531_sw & 15))]) : (0.0f);
          float v1535_data = ir4[2];
          ir4[2] = (v1535_data + v1534_data);
          int32_t v1537_a = v26_lead + 36;
          int32_t v1538_sw = v1537_a >> 4;
          float v1541_data = v1516_g ? (s1[(v1537_a ^ (v1538_sw & 15))]) : (0.0f);
          float v1542_data = ir4[3];
          ir4[3] = (v1542_data + v1541_data);
          int32_t v1544_a = v26_lead + 48;
          int32_t v1545_sw = v1544_a >> 4;
          float v1548_data = v1516_g ? (s1[(v1544_a ^ (v1545_sw & 15))]) : (0.0f);
          float v1549_data = ir4[4];
          ir4[4] = (v1549_data + v1548_data);
          int32_t v1551_a = v26_lead + 60;
          int32_t v1552_sw = v1551_a >> 4;
          float v1555_data = v1516_g ? (s1[(v1551_a ^ (v1552_sw & 15))]) : (0.0f);
          float v1556_data = ir4[5];
          ir4[5] = (v1556_data + v1555_data);
          int32_t v1558_a = v26_lead + 72;
          int32_t v1559_sw = v1558_a >> 4;
          float v1562_data = v1516_g ? (s1[(v1558_a ^ (v1559_sw & 15))]) : (0.0f);
          float v1563_data = ir4[6];
          ir4[6] = (v1563_data + v1562_data);
          int32_t v1565_a = v26_lead + 84;
          int32_t v1566_sw = v1565_a >> 4;
          float v1569_data = v1516_g ? (s1[(v1565_a ^ (v1566_sw & 15))]) : (0.0f);
          float v1570_data = ir4[7];
          ir4[7] = (v1570_data + v1569_data);
          int32_t v1572_a = v26_lead + 96;
          int32_t v1573_sw = v1572_a >> 4;
          float v1576_data = v1516_g ? (s1[(v1572_a ^ (v1573_sw & 15))]) : (0.0f);
          float v1577_data = ir4[8];
          ir4[8] = (v1577_data + v1576_data);
          int32_t v1579_a = v26_lead + 108;
          int32_t v1580_sw = v1579_a >> 4;
          float v1583_data = v1516_g ? (s1[(v1579_a ^ (v1580_sw & 15))]) : (0.0f);
          float v1584_data = ir4[9];
          ir4[9] = (v1584_data + v1583_data);
          int32_t v1586_a = v26_lead + 120;
          int32_t v1587_sw = v1586_a >> 4;
          float v1590_data = v1516_g ? (s1[(v1586_a ^ (v1587_sw & 15))]) : (0.0f);
          float v1591_data = ir4[10];
          ir4[10] = (v1591_data + v1590_data);
          int32_t v1593_a = v26_lead + 132;
          int32_t v1594_sw = v1593_a >> 4;
          float v1597_data = v1516_g ? (s1[(v1593_a ^ (v1594_sw & 15))]) : (0.0f);
          float v1598_data = ir4[11];
          ir4[11] = (v1598_data + v1597_data);
          // r4 = ir4
          if (v1516_g) {
            #pragma unroll
            for (int32_t v1601_n1 = 0; v1601_n1 < 12; ++v1601_n1) {
              float v1603_data = ir4[v1601_n1];
              r4[v1601_n1] = v1603_data;
            }
          }
          // glb_m3 = store{r>g}(r4);
          if (v1516_g) {
            #pragma unroll
            for (int32_t v1605_i1 = 0; v1605_i1 < 12; ++v1605_i1) {
              float v1607_data = r4[v1605_i1];
              glb_m3[(v26_lead + (v1605_i1 * 12))] = v1607_data;
            }
          }
          float r6[12]{};
          // ir6 = +(r5 * s0)
          // [(0, 2), (0, 12)] [(0, 12)]
          float ir6[12]{};
          float v1623_data = r5[0];
          float v1626_data = ir6[0];
          ir6[0] = (v1626_data + (v1623_data * v38_data));
          float v1631_data = ir6[1];
          ir6[1] = (v1631_data + (v1623_data * v43_data));
          float v1636_data = ir6[2];
          ir6[2] = (v1636_data + (v1623_data * v48_data));
          float v1641_data = ir6[3];
          ir6[3] = (v1641_data + (v1623_data * v53_data));
          float v1646_data = ir6[4];
          ir6[4] = (v1646_data + (v1623_data * v58_data));
          float v1651_data = ir6[5];
          ir6[5] = (v1651_data + (v1623_data * v63_data));
          float v1656_data = ir6[6];
          ir6[6] = (v1656_data + (v1623_data * v68_data));
          float v1661_data = ir6[7];
          ir6[7] = (v1661_data + (v1623_data * v73_data));
          float v1666_data = ir6[8];
          ir6[8] = (v1666_data + (v1623_data * v78_data));
          float v1671_data = ir6[9];
          ir6[9] = (v1671_data + (v1623_data * v83_data));
          float v1676_data = ir6[10];
          ir6[10] = (v1676_data + (v1623_data * v88_data));
          float v1681_data = ir6[11];
          ir6[11] = (v1681_data + (v1623_data * v93_data));
          float v1683_data = r5[1];
          float v1686_data = ir6[0];
          ir6[0] = (v1686_data + (v1683_data * v98_data));
          float v1691_data = ir6[1];
          ir6[1] = (v1691_data + (v1683_data * v103_data));
          float v1696_data = ir6[2];
          ir6[2] = (v1696_data + (v1683_data * v108_data));
          float v1701_data = ir6[3];
          ir6[3] = (v1701_data + (v1683_data * v113_data));
          float v1706_data = ir6[4];
          ir6[4] = (v1706_data + (v1683_data * v118_data));
          float v1711_data = ir6[5];
          ir6[5] = (v1711_data + (v1683_data * v123_data));
          float v1716_data = ir6[6];
          ir6[6] = (v1716_data + (v1683_data * v128_data));
          float v1721_data = ir6[7];
          ir6[7] = (v1721_data + (v1683_data * v133_data));
          float v1726_data = ir6[8];
          ir6[8] = (v1726_data + (v1683_data * v138_data));
          float v1731_data = ir6[9];
          ir6[9] = (v1731_data + (v1683_data * v143_data));
          float v1736_data = ir6[10];
          ir6[10] = (v1736_data + (v1683_data * v148_data));
          float v1741_data = ir6[11];
          ir6[11] = (v1741_data + (v1683_data * v153_data));
          float v1743_data = r5[2];
          float v1746_data = ir6[0];
          ir6[0] = (v1746_data + (v1743_data * v158_data));
          float v1751_data = ir6[1];
          ir6[1] = (v1751_data + (v1743_data * v163_data));
          float v1756_data = ir6[2];
          ir6[2] = (v1756_data + (v1743_data * v168_data));
          float v1761_data = ir6[3];
          ir6[3] = (v1761_data + (v1743_data * v173_data));
          float v1766_data = ir6[4];
          ir6[4] = (v1766_data + (v1743_data * v178_data));
          float v1771_data = ir6[5];
          ir6[5] = (v1771_data + (v1743_data * v183_data));
          float v1776_data = ir6[6];
          ir6[6] = (v1776_data + (v1743_data * v188_data));
          float v1781_data = ir6[7];
          ir6[7] = (v1781_data + (v1743_data * v193_data));
          float v1786_data = ir6[8];
          ir6[8] = (v1786_data + (v1743_data * v198_data));
          float v1791_data = ir6[9];
          ir6[9] = (v1791_data + (v1743_data * v203_data));
          float v1796_data = ir6[10];
          ir6[10] = (v1796_data + (v1743_data * v208_data));
          float v1801_data = ir6[11];
          ir6[11] = (v1801_data + (v1743_data * v213_data));
          float v1803_data = r5[3];
          float v1806_data = ir6[0];
          ir6[0] = (v1806_data + (v1803_data * v218_data));
          float v1811_data = ir6[1];
          ir6[1] = (v1811_data + (v1803_data * v223_data));
          float v1816_data = ir6[2];
          ir6[2] = (v1816_data + (v1803_data * v228_data));
          float v1821_data = ir6[3];
          ir6[3] = (v1821_data + (v1803_data * v233_data));
          float v1826_data = ir6[4];
          ir6[4] = (v1826_data + (v1803_data * v238_data));
          float v1831_data = ir6[5];
          ir6[5] = (v1831_data + (v1803_data * v243_data));
          float v1836_data = ir6[6];
          ir6[6] = (v1836_data + (v1803_data * v248_data));
          float v1841_data = ir6[7];
          ir6[7] = (v1841_data + (v1803_data * v253_data));
          float v1846_data = ir6[8];
          ir6[8] = (v1846_data + (v1803_data * v258_data));
          float v1851_data = ir6[9];
          ir6[9] = (v1851_data + (v1803_data * v263_data));
          float v1856_data = ir6[10];
          ir6[10] = (v1856_data + (v1803_data * v268_data));
          float v1861_data = ir6[11];
          ir6[11] = (v1861_data + (v1803_data * v273_data));
          float v1863_data = r5[4];
          float v1866_data = ir6[0];
          ir6[0] = (v1866_data + (v1863_data * v278_data));
          float v1871_data = ir6[1];
          ir6[1] = (v1871_data + (v1863_data * v283_data));
          float v1876_data = ir6[2];
          ir6[2] = (v1876_data + (v1863_data * v288_data));
          float v1881_data = ir6[3];
          ir6[3] = (v1881_data + (v1863_data * v293_data));
          float v1886_data = ir6[4];
          ir6[4] = (v1886_data + (v1863_data * v298_data));
          float v1891_data = ir6[5];
          ir6[5] = (v1891_data + (v1863_data * v303_data));
          float v1896_data = ir6[6];
          ir6[6] = (v1896_data + (v1863_data * v308_data));
          float v1901_data = ir6[7];
          ir6[7] = (v1901_data + (v1863_data * v313_data));
          float v1906_data = ir6[8];
          ir6[8] = (v1906_data + (v1863_data * v318_data));
          float v1911_data = ir6[9];
          ir6[9] = (v1911_data + (v1863_data * v323_data));
          float v1916_data = ir6[10];
          ir6[10] = (v1916_data + (v1863_data * v328_data));
          float v1921_data = ir6[11];
          ir6[11] = (v1921_data + (v1863_data * v333_data));
          float v1923_data = r5[5];
          float v1926_data = ir6[0];
          ir6[0] = (v1926_data + (v1923_data * v338_data));
          float v1931_data = ir6[1];
          ir6[1] = (v1931_data + (v1923_data * v343_data));
          float v1936_data = ir6[2];
          ir6[2] = (v1936_data + (v1923_data * v348_data));
          float v1941_data = ir6[3];
          ir6[3] = (v1941_data + (v1923_data * v353_data));
          float v1946_data = ir6[4];
          ir6[4] = (v1946_data + (v1923_data * v358_data));
          float v1951_data = ir6[5];
          ir6[5] = (v1951_data + (v1923_data * v363_data));
          float v1956_data = ir6[6];
          ir6[6] = (v1956_data + (v1923_data * v368_data));
          float v1961_data = ir6[7];
          ir6[7] = (v1961_data + (v1923_data * v373_data));
          float v1966_data = ir6[8];
          ir6[8] = (v1966_data + (v1923_data * v378_data));
          float v1971_data = ir6[9];
          ir6[9] = (v1971_data + (v1923_data * v383_data));
          float v1976_data = ir6[10];
          ir6[10] = (v1976_data + (v1923_data * v388_data));
          float v1981_data = ir6[11];
          ir6[11] = (v1981_data + (v1923_data * v393_data));
          float v1983_data = r5[6];
          float v1986_data = ir6[0];
          ir6[0] = (v1986_data + (v1983_data * v398_data));
          float v1991_data = ir6[1];
          ir6[1] = (v1991_data + (v1983_data * v403_data));
          float v1996_data = ir6[2];
          ir6[2] = (v1996_data + (v1983_data * v408_data));
          float v2001_data = ir6[3];
          ir6[3] = (v2001_data + (v1983_data * v413_data));
          float v2006_data = ir6[4];
          ir6[4] = (v2006_data + (v1983_data * v418_data));
          float v2011_data = ir6[5];
          ir6[5] = (v2011_data + (v1983_data * v423_data));
          float v2016_data = ir6[6];
          ir6[6] = (v2016_data + (v1983_data * v428_data));
          float v2021_data = ir6[7];
          ir6[7] = (v2021_data + (v1983_data * v433_data));
          float v2026_data = ir6[8];
          ir6[8] = (v2026_data + (v1983_data * v438_data));
          float v2031_data = ir6[9];
          ir6[9] = (v2031_data + (v1983_data * v443_data));
          float v2036_data = ir6[10];
          ir6[10] = (v2036_data + (v1983_data * v448_data));
          float v2041_data = ir6[11];
          ir6[11] = (v2041_data + (v1983_data * v453_data));
          float v2043_data = r5[7];
          float v2046_data = ir6[0];
          ir6[0] = (v2046_data + (v2043_data * v458_data));
          float v2051_data = ir6[1];
          ir6[1] = (v2051_data + (v2043_data * v463_data));
          float v2056_data = ir6[2];
          ir6[2] = (v2056_data + (v2043_data * v468_data));
          float v2061_data = ir6[3];
          ir6[3] = (v2061_data + (v2043_data * v473_data));
          float v2066_data = ir6[4];
          ir6[4] = (v2066_data + (v2043_data * v478_data));
          float v2071_data = ir6[5];
          ir6[5] = (v2071_data + (v2043_data * v483_data));
          float v2076_data = ir6[6];
          ir6[6] = (v2076_data + (v2043_data * v488_data));
          float v2081_data = ir6[7];
          ir6[7] = (v2081_data + (v2043_data * v493_data));
          float v2086_data = ir6[8];
          ir6[8] = (v2086_data + (v2043_data * v498_data));
          float v2091_data = ir6[9];
          ir6[9] = (v2091_data + (v2043_data * v503_data));
          float v2096_data = ir6[10];
          ir6[10] = (v2096_data + (v2043_data * v508_data));
          float v2101_data = ir6[11];
          ir6[11] = (v2101_data + (v2043_data * v513_data));
          float v2103_data = r5[8];
          float v2106_data = ir6[0];
          ir6[0] = (v2106_data + (v2103_data * v518_data));
          float v2111_data = ir6[1];
          ir6[1] = (v2111_data + (v2103_data * v523_data));
          float v2116_data = ir6[2];
          ir6[2] = (v2116_data + (v2103_data * v528_data));
          float v2121_data = ir6[3];
          ir6[3] = (v2121_data + (v2103_data * v533_data));
          float v2126_data = ir6[4];
          ir6[4] = (v2126_data + (v2103_data * v538_data));
          float v2131_data = ir6[5];
          ir6[5] = (v2131_data + (v2103_data * v543_data));
          float v2136_data = ir6[6];
          ir6[6] = (v2136_data + (v2103_data * v548_data));
          float v2141_data = ir6[7];
          ir6[7] = (v2141_data + (v2103_data * v553_data));
          float v2146_data = ir6[8];
          ir6[8] = (v2146_data + (v2103_data * v558_data));
          float v2151_data = ir6[9];
          ir6[9] = (v2151_data + (v2103_data * v563_data));
          float v2156_data = ir6[10];
          ir6[10] = (v2156_data + (v2103_data * v568_data));
          float v2161_data = ir6[11];
          ir6[11] = (v2161_data + (v2103_data * v573_data));
          float v2163_data = r5[9];
          float v2166_data = ir6[0];
          ir6[0] = (v2166_data + (v2163_data * v578_data));
          float v2171_data = ir6[1];
          ir6[1] = (v2171_data + (v2163_data * v583_data));
          float v2176_data = ir6[2];
          ir6[2] = (v2176_data + (v2163_data * v588_data));
          float v2181_data = ir6[3];
          ir6[3] = (v2181_data + (v2163_data * v593_data));
          float v2186_data = ir6[4];
          ir6[4] = (v2186_data + (v2163_data * v598_data));
          float v2191_data = ir6[5];
          ir6[5] = (v2191_data + (v2163_data * v603_data));
          float v2196_data = ir6[6];
          ir6[6] = (v2196_data + (v2163_data * v608_data));
          float v2201_data = ir6[7];
          ir6[7] = (v2201_data + (v2163_data * v613_data));
          float v2206_data = ir6[8];
          ir6[8] = (v2206_data + (v2163_data * v618_data));
          float v2211_data = ir6[9];
          ir6[9] = (v2211_data + (v2163_data * v623_data));
          float v2216_data = ir6[10];
          ir6[10] = (v2216_data + (v2163_data * v628_data));
          float v2221_data = ir6[11];
          ir6[11] = (v2221_data + (v2163_data * v633_data));
          float v2223_data = r5[10];
          float v2226_data = ir6[0];
          ir6[0] = (v2226_data + (v2223_data * v638_data));
          float v2231_data = ir6[1];
          ir6[1] = (v2231_data + (v2223_data * v643_data));
          float v2236_data = ir6[2];
          ir6[2] = (v2236_data + (v2223_data * v648_data));
          float v2241_data = ir6[3];
          ir6[3] = (v2241_data + (v2223_data * v653_data));
          float v2246_data = ir6[4];
          ir6[4] = (v2246_data + (v2223_data * v658_data));
          float v2251_data = ir6[5];
          ir6[5] = (v2251_data + (v2223_data * v663_data));
          float v2256_data = ir6[6];
          ir6[6] = (v2256_data + (v2223_data * v668_data));
          float v2261_data = ir6[7];
          ir6[7] = (v2261_data + (v2223_data * v673_data));
          float v2266_data = ir6[8];
          ir6[8] = (v2266_data + (v2223_data * v678_data));
          float v2271_data = ir6[9];
          ir6[9] = (v2271_data + (v2223_data * v683_data));
          float v2276_data = ir6[10];
          ir6[10] = (v2276_data + (v2223_data * v688_data));
          float v2281_data = ir6[11];
          ir6[11] = (v2281_data + (v2223_data * v693_data));
          float v2283_data = r5[11];
          float v2286_data = ir6[0];
          ir6[0] = (v2286_data + (v2283_data * v698_data));
          float v2291_data = ir6[1];
          ir6[1] = (v2291_data + (v2283_data * v703_data));
          float v2296_data = ir6[2];
          ir6[2] = (v2296_data + (v2283_data * v708_data));
          float v2301_data = ir6[3];
          ir6[3] = (v2301_data + (v2283_data * v713_data));
          float v2306_data = ir6[4];
          ir6[4] = (v2306_data + (v2283_data * v718_data));
          float v2311_data = ir6[5];
          ir6[5] = (v2311_data + (v2283_data * v723_data));
          float v2316_data = ir6[6];
          ir6[6] = (v2316_data + (v2283_data * v728_data));
          float v2321_data = ir6[7];
          ir6[7] = (v2321_data + (v2283_data * v733_data));
          float v2326_data = ir6[8];
          ir6[8] = (v2326_data + (v2283_data * v738_data));
          float v2331_data = ir6[9];
          ir6[9] = (v2331_data + (v2283_data * v743_data));
          float v2336_data = ir6[10];
          ir6[10] = (v2336_data + (v2283_data * v748_data));
          float v2341_data = ir6[11];
          ir6[11] = (v2341_data + (v2283_data * v753_data));
          // r6 = ir6
          if (v1613_g) {
            #pragma unroll
            for (int32_t v2343_n1 = 0; v2343_n1 < 12; ++v2343_n1) {
              float v2345_data = ir6[v2343_n1];
              r6[v2343_n1] = v2345_data;
            }
          }
          // s1 = store{r>s, clear}(localShrMem0, r6);
          bool v2348_g = (v26_lead >= 8) && v1516_g;
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          if (v2348_g) {
            #pragma unroll
            for (int32_t v2349_z1 = 0; v2349_z1 < 12; ++v2349_z1) {
              int32_t v2354_a = v26_lead + (v2349_z1 * 12);
              s1[(v2354_a ^ ((v2354_a >> 4) & 15))] = 0.0f;
            }
          }
          if (v1613_g) {
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
          int32_t v2376_sw = v26_lead ^ v1518_sw;
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v2377_data = v1516_g ? (s1[v2376_sw]) : (0.0f);
          float v2378_data = ir7[0];
          ir7[0] = (v2378_data + v2377_data);
          float v2384_data = v1516_g ? (s1[(v1523_a ^ (v1524_sw & 15))]) : (0.0f);
          float v2385_data = ir7[1];
          ir7[1] = (v2385_data + v2384_data);
          float v2391_data = v1516_g ? (s1[(v1530_a ^ (v1531_sw & 15))]) : (0.0f);
          float v2392_data = ir7[2];
          ir7[2] = (v2392_data + v2391_data);
          float v2398_data = v1516_g ? (s1[(v1537_a ^ (v1538_sw & 15))]) : (0.0f);
          float v2399_data = ir7[3];
          ir7[3] = (v2399_data + v2398_data);
          float v2405_data = v1516_g ? (s1[(v1544_a ^ (v1545_sw & 15))]) : (0.0f);
          float v2406_data = ir7[4];
          ir7[4] = (v2406_data + v2405_data);
          float v2412_data = v1516_g ? (s1[(v1551_a ^ (v1552_sw & 15))]) : (0.0f);
          float v2413_data = ir7[5];
          ir7[5] = (v2413_data + v2412_data);
          float v2419_data = v1516_g ? (s1[(v1558_a ^ (v1559_sw & 15))]) : (0.0f);
          float v2420_data = ir7[6];
          ir7[6] = (v2420_data + v2419_data);
          float v2426_data = v1516_g ? (s1[(v1565_a ^ (v1566_sw & 15))]) : (0.0f);
          float v2427_data = ir7[7];
          ir7[7] = (v2427_data + v2426_data);
          float v2433_data = v1516_g ? (s1[(v1572_a ^ (v1573_sw & 15))]) : (0.0f);
          float v2434_data = ir7[8];
          ir7[8] = (v2434_data + v2433_data);
          float v2440_data = v1516_g ? (s1[(v1579_a ^ (v1580_sw & 15))]) : (0.0f);
          float v2441_data = ir7[9];
          ir7[9] = (v2441_data + v2440_data);
          float v2447_data = v1516_g ? (s1[(v1586_a ^ (v1587_sw & 15))]) : (0.0f);
          float v2448_data = ir7[10];
          ir7[10] = (v2448_data + v2447_data);
          float v2454_data = v1516_g ? (s1[(v1593_a ^ (v1594_sw & 15))]) : (0.0f);
          float v2455_data = ir7[11];
          ir7[11] = (v2455_data + v2454_data);
          // r7 = ir7
          if (v1516_g) {
            #pragma unroll
            for (int32_t v2457_n1 = 0; v2457_n1 < 12; ++v2457_n1) {
              float v2459_data = ir7[v2457_n1];
              r7[v2457_n1] = v2459_data;
            }
          }
          // glb_m5 = store{r>g}(r7);
          if (v1516_g) {
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

