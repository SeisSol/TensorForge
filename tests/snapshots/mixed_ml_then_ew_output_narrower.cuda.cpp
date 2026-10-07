// === base name ===
kernel_0d24cde7aee54ea5

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_0d24cde7aee54ea5 = {{16, 8, 1}, 16, 12, 1, 8, 5632, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_0d24cde7aee54ea5(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_0d24cde7aee54ea5(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_0d24cde7aee54ea5(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_0d24cde7aee54ea5, block.x * block.y * block.z, 1408 * sizeof(float));
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
void launcher_kernel_0d24cde7aee54ea5(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_0d24cde7aee54ea5(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_0d24cde7aee54ea5, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_0d24cde7aee54ea5<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_0d24cde7aee54ea5(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 8 per block = block 16x8x1, 5632 B shared, occupancy grid
    // operands:
    //   m0 32×32(12×12) {0..12}×{0..12} strided
    //   m1 32×32(12×12) {0..12}×{0..12} strided
    //   m2 32×32(12×12) {0..12}×{0..12} strided
    //   m3 32×32(4×12) {4..8}×{0..12} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    //   D = abs(N)
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1408}],"shared_bytes":5632,"shared_elements":1408,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[12,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"N","bbox":[[4,0],[8,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[4,0],[8,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[4,0],[8,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[176 * threadIdx.y + 0];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v8_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v8_batchId0 < numElements0; v8_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v9_ahead1 = v8_batchId0 + (gridDim.x * blockDim.y);
        size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 144 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 144 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 144 + 0 + m2_extraOffset];
          const float *const __restrict__ glb_m3 = &m3[v8_batchId0 * 48 + 0 + m3_extraOffset];
          float r0[12]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v23_lead = threadIdx.x % 16;
          bool v24_g = v23_lead < 12;
          if (v24_g) {
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 12; ++v25_i1) {
              float v30_data = __ldcg(&glb_m1[(v23_lead + (v25_i1 * 12))]);
              r0[v25_i1] = v30_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 9; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[12]{};
          // ir1 = +(r0 * s0)
          // [(0, 12), (0, 12)] [(0, 12)]
          float ir1[12]{};
          float v35_data = r0[0];
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v36_data = s0[0];
          float v38_data = ir1[0];
          ir1[0] = (v38_data + (v35_data * v36_data));
          float v41_data = s0[12];
          float v43_data = ir1[1];
          ir1[1] = (v43_data + (v35_data * v41_data));
          float v46_data = s0[24];
          float v48_data = ir1[2];
          ir1[2] = (v48_data + (v35_data * v46_data));
          float v51_data = s0[36];
          float v53_data = ir1[3];
          ir1[3] = (v53_data + (v35_data * v51_data));
          float v56_data = s0[48];
          float v58_data = ir1[4];
          ir1[4] = (v58_data + (v35_data * v56_data));
          float v61_data = s0[60];
          float v63_data = ir1[5];
          ir1[5] = (v63_data + (v35_data * v61_data));
          float v66_data = s0[72];
          float v68_data = ir1[6];
          ir1[6] = (v68_data + (v35_data * v66_data));
          float v71_data = s0[84];
          float v73_data = ir1[7];
          ir1[7] = (v73_data + (v35_data * v71_data));
          float v76_data = s0[96];
          float v78_data = ir1[8];
          ir1[8] = (v78_data + (v35_data * v76_data));
          float v81_data = s0[108];
          float v83_data = ir1[9];
          ir1[9] = (v83_data + (v35_data * v81_data));
          float v86_data = s0[120];
          float v88_data = ir1[10];
          ir1[10] = (v88_data + (v35_data * v86_data));
          float v91_data = s0[132];
          float v93_data = ir1[11];
          ir1[11] = (v93_data + (v35_data * v91_data));
          float v95_data = r0[1];
          float v96_data = s0[1];
          float v98_data = ir1[0];
          ir1[0] = (v98_data + (v95_data * v96_data));
          float v101_data = s0[13];
          float v103_data = ir1[1];
          ir1[1] = (v103_data + (v95_data * v101_data));
          float v106_data = s0[25];
          float v108_data = ir1[2];
          ir1[2] = (v108_data + (v95_data * v106_data));
          float v111_data = s0[37];
          float v113_data = ir1[3];
          ir1[3] = (v113_data + (v95_data * v111_data));
          float v116_data = s0[49];
          float v118_data = ir1[4];
          ir1[4] = (v118_data + (v95_data * v116_data));
          float v121_data = s0[61];
          float v123_data = ir1[5];
          ir1[5] = (v123_data + (v95_data * v121_data));
          float v126_data = s0[73];
          float v128_data = ir1[6];
          ir1[6] = (v128_data + (v95_data * v126_data));
          float v131_data = s0[85];
          float v133_data = ir1[7];
          ir1[7] = (v133_data + (v95_data * v131_data));
          float v136_data = s0[97];
          float v138_data = ir1[8];
          ir1[8] = (v138_data + (v95_data * v136_data));
          float v141_data = s0[109];
          float v143_data = ir1[9];
          ir1[9] = (v143_data + (v95_data * v141_data));
          float v146_data = s0[121];
          float v148_data = ir1[10];
          ir1[10] = (v148_data + (v95_data * v146_data));
          float v151_data = s0[133];
          float v153_data = ir1[11];
          ir1[11] = (v153_data + (v95_data * v151_data));
          float v155_data = r0[2];
          float v156_data = s0[2];
          float v158_data = ir1[0];
          ir1[0] = (v158_data + (v155_data * v156_data));
          float v161_data = s0[14];
          float v163_data = ir1[1];
          ir1[1] = (v163_data + (v155_data * v161_data));
          float v166_data = s0[26];
          float v168_data = ir1[2];
          ir1[2] = (v168_data + (v155_data * v166_data));
          float v171_data = s0[38];
          float v173_data = ir1[3];
          ir1[3] = (v173_data + (v155_data * v171_data));
          float v176_data = s0[50];
          float v178_data = ir1[4];
          ir1[4] = (v178_data + (v155_data * v176_data));
          float v181_data = s0[62];
          float v183_data = ir1[5];
          ir1[5] = (v183_data + (v155_data * v181_data));
          float v186_data = s0[74];
          float v188_data = ir1[6];
          ir1[6] = (v188_data + (v155_data * v186_data));
          float v191_data = s0[86];
          float v193_data = ir1[7];
          ir1[7] = (v193_data + (v155_data * v191_data));
          float v196_data = s0[98];
          float v198_data = ir1[8];
          ir1[8] = (v198_data + (v155_data * v196_data));
          float v201_data = s0[110];
          float v203_data = ir1[9];
          ir1[9] = (v203_data + (v155_data * v201_data));
          float v206_data = s0[122];
          float v208_data = ir1[10];
          ir1[10] = (v208_data + (v155_data * v206_data));
          float v211_data = s0[134];
          float v213_data = ir1[11];
          ir1[11] = (v213_data + (v155_data * v211_data));
          float v215_data = r0[3];
          float v216_data = s0[3];
          float v218_data = ir1[0];
          ir1[0] = (v218_data + (v215_data * v216_data));
          float v221_data = s0[15];
          float v223_data = ir1[1];
          ir1[1] = (v223_data + (v215_data * v221_data));
          float v226_data = s0[27];
          float v228_data = ir1[2];
          ir1[2] = (v228_data + (v215_data * v226_data));
          float v231_data = s0[39];
          float v233_data = ir1[3];
          ir1[3] = (v233_data + (v215_data * v231_data));
          float v236_data = s0[51];
          float v238_data = ir1[4];
          ir1[4] = (v238_data + (v215_data * v236_data));
          float v241_data = s0[63];
          float v243_data = ir1[5];
          ir1[5] = (v243_data + (v215_data * v241_data));
          float v246_data = s0[75];
          float v248_data = ir1[6];
          ir1[6] = (v248_data + (v215_data * v246_data));
          float v251_data = s0[87];
          float v253_data = ir1[7];
          ir1[7] = (v253_data + (v215_data * v251_data));
          float v256_data = s0[99];
          float v258_data = ir1[8];
          ir1[8] = (v258_data + (v215_data * v256_data));
          float v261_data = s0[111];
          float v263_data = ir1[9];
          ir1[9] = (v263_data + (v215_data * v261_data));
          float v266_data = s0[123];
          float v268_data = ir1[10];
          ir1[10] = (v268_data + (v215_data * v266_data));
          float v271_data = s0[135];
          float v273_data = ir1[11];
          ir1[11] = (v273_data + (v215_data * v271_data));
          float v275_data = r0[4];
          float v276_data = s0[4];
          float v278_data = ir1[0];
          ir1[0] = (v278_data + (v275_data * v276_data));
          float v281_data = s0[16];
          float v283_data = ir1[1];
          ir1[1] = (v283_data + (v275_data * v281_data));
          float v286_data = s0[28];
          float v288_data = ir1[2];
          ir1[2] = (v288_data + (v275_data * v286_data));
          float v291_data = s0[40];
          float v293_data = ir1[3];
          ir1[3] = (v293_data + (v275_data * v291_data));
          float v296_data = s0[52];
          float v298_data = ir1[4];
          ir1[4] = (v298_data + (v275_data * v296_data));
          float v301_data = s0[64];
          float v303_data = ir1[5];
          ir1[5] = (v303_data + (v275_data * v301_data));
          float v306_data = s0[76];
          float v308_data = ir1[6];
          ir1[6] = (v308_data + (v275_data * v306_data));
          float v311_data = s0[88];
          float v313_data = ir1[7];
          ir1[7] = (v313_data + (v275_data * v311_data));
          float v316_data = s0[100];
          float v318_data = ir1[8];
          ir1[8] = (v318_data + (v275_data * v316_data));
          float v321_data = s0[112];
          float v323_data = ir1[9];
          ir1[9] = (v323_data + (v275_data * v321_data));
          float v326_data = s0[124];
          float v328_data = ir1[10];
          ir1[10] = (v328_data + (v275_data * v326_data));
          float v331_data = s0[136];
          float v333_data = ir1[11];
          ir1[11] = (v333_data + (v275_data * v331_data));
          float v335_data = r0[5];
          float v336_data = s0[5];
          float v338_data = ir1[0];
          ir1[0] = (v338_data + (v335_data * v336_data));
          float v341_data = s0[17];
          float v343_data = ir1[1];
          ir1[1] = (v343_data + (v335_data * v341_data));
          float v346_data = s0[29];
          float v348_data = ir1[2];
          ir1[2] = (v348_data + (v335_data * v346_data));
          float v351_data = s0[41];
          float v353_data = ir1[3];
          ir1[3] = (v353_data + (v335_data * v351_data));
          float v356_data = s0[53];
          float v358_data = ir1[4];
          ir1[4] = (v358_data + (v335_data * v356_data));
          float v361_data = s0[65];
          float v363_data = ir1[5];
          ir1[5] = (v363_data + (v335_data * v361_data));
          float v366_data = s0[77];
          float v368_data = ir1[6];
          ir1[6] = (v368_data + (v335_data * v366_data));
          float v371_data = s0[89];
          float v373_data = ir1[7];
          ir1[7] = (v373_data + (v335_data * v371_data));
          float v376_data = s0[101];
          float v378_data = ir1[8];
          ir1[8] = (v378_data + (v335_data * v376_data));
          float v381_data = s0[113];
          float v383_data = ir1[9];
          ir1[9] = (v383_data + (v335_data * v381_data));
          float v386_data = s0[125];
          float v388_data = ir1[10];
          ir1[10] = (v388_data + (v335_data * v386_data));
          float v391_data = s0[137];
          float v393_data = ir1[11];
          ir1[11] = (v393_data + (v335_data * v391_data));
          float v395_data = r0[6];
          float v396_data = s0[6];
          float v398_data = ir1[0];
          ir1[0] = (v398_data + (v395_data * v396_data));
          float v401_data = s0[18];
          float v403_data = ir1[1];
          ir1[1] = (v403_data + (v395_data * v401_data));
          float v406_data = s0[30];
          float v408_data = ir1[2];
          ir1[2] = (v408_data + (v395_data * v406_data));
          float v411_data = s0[42];
          float v413_data = ir1[3];
          ir1[3] = (v413_data + (v395_data * v411_data));
          float v416_data = s0[54];
          float v418_data = ir1[4];
          ir1[4] = (v418_data + (v395_data * v416_data));
          float v421_data = s0[66];
          float v423_data = ir1[5];
          ir1[5] = (v423_data + (v395_data * v421_data));
          float v426_data = s0[78];
          float v428_data = ir1[6];
          ir1[6] = (v428_data + (v395_data * v426_data));
          float v431_data = s0[90];
          float v433_data = ir1[7];
          ir1[7] = (v433_data + (v395_data * v431_data));
          float v436_data = s0[102];
          float v438_data = ir1[8];
          ir1[8] = (v438_data + (v395_data * v436_data));
          float v441_data = s0[114];
          float v443_data = ir1[9];
          ir1[9] = (v443_data + (v395_data * v441_data));
          float v446_data = s0[126];
          float v448_data = ir1[10];
          ir1[10] = (v448_data + (v395_data * v446_data));
          float v451_data = s0[138];
          float v453_data = ir1[11];
          ir1[11] = (v453_data + (v395_data * v451_data));
          float v455_data = r0[7];
          float v456_data = s0[7];
          float v458_data = ir1[0];
          ir1[0] = (v458_data + (v455_data * v456_data));
          float v461_data = s0[19];
          float v463_data = ir1[1];
          ir1[1] = (v463_data + (v455_data * v461_data));
          float v466_data = s0[31];
          float v468_data = ir1[2];
          ir1[2] = (v468_data + (v455_data * v466_data));
          float v471_data = s0[43];
          float v473_data = ir1[3];
          ir1[3] = (v473_data + (v455_data * v471_data));
          float v476_data = s0[55];
          float v478_data = ir1[4];
          ir1[4] = (v478_data + (v455_data * v476_data));
          float v481_data = s0[67];
          float v483_data = ir1[5];
          ir1[5] = (v483_data + (v455_data * v481_data));
          float v486_data = s0[79];
          float v488_data = ir1[6];
          ir1[6] = (v488_data + (v455_data * v486_data));
          float v491_data = s0[91];
          float v493_data = ir1[7];
          ir1[7] = (v493_data + (v455_data * v491_data));
          float v496_data = s0[103];
          float v498_data = ir1[8];
          ir1[8] = (v498_data + (v455_data * v496_data));
          float v501_data = s0[115];
          float v503_data = ir1[9];
          ir1[9] = (v503_data + (v455_data * v501_data));
          float v506_data = s0[127];
          float v508_data = ir1[10];
          ir1[10] = (v508_data + (v455_data * v506_data));
          float v511_data = s0[139];
          float v513_data = ir1[11];
          ir1[11] = (v513_data + (v455_data * v511_data));
          float v515_data = r0[8];
          float v516_data = s0[8];
          float v518_data = ir1[0];
          ir1[0] = (v518_data + (v515_data * v516_data));
          float v521_data = s0[20];
          float v523_data = ir1[1];
          ir1[1] = (v523_data + (v515_data * v521_data));
          float v526_data = s0[32];
          float v528_data = ir1[2];
          ir1[2] = (v528_data + (v515_data * v526_data));
          float v531_data = s0[44];
          float v533_data = ir1[3];
          ir1[3] = (v533_data + (v515_data * v531_data));
          float v536_data = s0[56];
          float v538_data = ir1[4];
          ir1[4] = (v538_data + (v515_data * v536_data));
          float v541_data = s0[68];
          float v543_data = ir1[5];
          ir1[5] = (v543_data + (v515_data * v541_data));
          float v546_data = s0[80];
          float v548_data = ir1[6];
          ir1[6] = (v548_data + (v515_data * v546_data));
          float v551_data = s0[92];
          float v553_data = ir1[7];
          ir1[7] = (v553_data + (v515_data * v551_data));
          float v556_data = s0[104];
          float v558_data = ir1[8];
          ir1[8] = (v558_data + (v515_data * v556_data));
          float v561_data = s0[116];
          float v563_data = ir1[9];
          ir1[9] = (v563_data + (v515_data * v561_data));
          float v566_data = s0[128];
          float v568_data = ir1[10];
          ir1[10] = (v568_data + (v515_data * v566_data));
          float v571_data = s0[140];
          float v573_data = ir1[11];
          ir1[11] = (v573_data + (v515_data * v571_data));
          float v575_data = r0[9];
          float v576_data = s0[9];
          float v578_data = ir1[0];
          ir1[0] = (v578_data + (v575_data * v576_data));
          float v581_data = s0[21];
          float v583_data = ir1[1];
          ir1[1] = (v583_data + (v575_data * v581_data));
          float v586_data = s0[33];
          float v588_data = ir1[2];
          ir1[2] = (v588_data + (v575_data * v586_data));
          float v591_data = s0[45];
          float v593_data = ir1[3];
          ir1[3] = (v593_data + (v575_data * v591_data));
          float v596_data = s0[57];
          float v598_data = ir1[4];
          ir1[4] = (v598_data + (v575_data * v596_data));
          float v601_data = s0[69];
          float v603_data = ir1[5];
          ir1[5] = (v603_data + (v575_data * v601_data));
          float v606_data = s0[81];
          float v608_data = ir1[6];
          ir1[6] = (v608_data + (v575_data * v606_data));
          float v611_data = s0[93];
          float v613_data = ir1[7];
          ir1[7] = (v613_data + (v575_data * v611_data));
          float v616_data = s0[105];
          float v618_data = ir1[8];
          ir1[8] = (v618_data + (v575_data * v616_data));
          float v621_data = s0[117];
          float v623_data = ir1[9];
          ir1[9] = (v623_data + (v575_data * v621_data));
          float v626_data = s0[129];
          float v628_data = ir1[10];
          ir1[10] = (v628_data + (v575_data * v626_data));
          float v631_data = s0[141];
          float v633_data = ir1[11];
          ir1[11] = (v633_data + (v575_data * v631_data));
          float v635_data = r0[10];
          float v636_data = s0[10];
          float v638_data = ir1[0];
          ir1[0] = (v638_data + (v635_data * v636_data));
          float v641_data = s0[22];
          float v643_data = ir1[1];
          ir1[1] = (v643_data + (v635_data * v641_data));
          float v646_data = s0[34];
          float v648_data = ir1[2];
          ir1[2] = (v648_data + (v635_data * v646_data));
          float v651_data = s0[46];
          float v653_data = ir1[3];
          ir1[3] = (v653_data + (v635_data * v651_data));
          float v656_data = s0[58];
          float v658_data = ir1[4];
          ir1[4] = (v658_data + (v635_data * v656_data));
          float v661_data = s0[70];
          float v663_data = ir1[5];
          ir1[5] = (v663_data + (v635_data * v661_data));
          float v666_data = s0[82];
          float v668_data = ir1[6];
          ir1[6] = (v668_data + (v635_data * v666_data));
          float v671_data = s0[94];
          float v673_data = ir1[7];
          ir1[7] = (v673_data + (v635_data * v671_data));
          float v676_data = s0[106];
          float v678_data = ir1[8];
          ir1[8] = (v678_data + (v635_data * v676_data));
          float v681_data = s0[118];
          float v683_data = ir1[9];
          ir1[9] = (v683_data + (v635_data * v681_data));
          float v686_data = s0[130];
          float v688_data = ir1[10];
          ir1[10] = (v688_data + (v635_data * v686_data));
          float v691_data = s0[142];
          float v693_data = ir1[11];
          ir1[11] = (v693_data + (v635_data * v691_data));
          float v695_data = r0[11];
          float v696_data = s0[11];
          float v698_data = ir1[0];
          ir1[0] = (v698_data + (v695_data * v696_data));
          float v701_data = s0[23];
          float v703_data = ir1[1];
          ir1[1] = (v703_data + (v695_data * v701_data));
          float v706_data = s0[35];
          float v708_data = ir1[2];
          ir1[2] = (v708_data + (v695_data * v706_data));
          float v711_data = s0[47];
          float v713_data = ir1[3];
          ir1[3] = (v713_data + (v695_data * v711_data));
          float v716_data = s0[59];
          float v718_data = ir1[4];
          ir1[4] = (v718_data + (v695_data * v716_data));
          float v721_data = s0[71];
          float v723_data = ir1[5];
          ir1[5] = (v723_data + (v695_data * v721_data));
          float v726_data = s0[83];
          float v728_data = ir1[6];
          ir1[6] = (v728_data + (v695_data * v726_data));
          float v731_data = s0[95];
          float v733_data = ir1[7];
          ir1[7] = (v733_data + (v695_data * v731_data));
          float v736_data = s0[107];
          float v738_data = ir1[8];
          ir1[8] = (v738_data + (v695_data * v736_data));
          float v741_data = s0[119];
          float v743_data = ir1[9];
          ir1[9] = (v743_data + (v695_data * v741_data));
          float v746_data = s0[131];
          float v748_data = ir1[10];
          ir1[10] = (v748_data + (v695_data * v746_data));
          float v751_data = s0[143];
          float v753_data = ir1[11];
          ir1[11] = (v753_data + (v695_data * v751_data));
          // r1 = ir1
          if (v24_g) {
            #pragma unroll
            for (int32_t v755_n1 = 0; v755_n1 < 12; ++v755_n1) {
              float v757_data = ir1[v755_n1];
              r1[v755_n1] = v757_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          if (v24_g) {
            #pragma unroll
            for (int32_t v758_i1 = 0; v758_i1 < 12; ++v758_i1) {
              float v760_data = r1[v758_i1];
              glb_m0[(v23_lead + (v758_i1 * 12))] = v760_data;
            }
          }
          float r2[12]{};
          // r2 = abs(glb_m3)
          bool v766_g = v23_lead < 4;
          if (v766_g) {
            int32_t v771_a = (v23_lead + 4) - 4;
            #pragma unroll
            for (int32_t v767_k1 = 0; v767_k1 < 12; ++v767_k1) {
              float v774_data = glb_m3[(v771_a + (v767_k1 * 4))];
              r2[v767_k1] = (fabsf(v774_data));
            }
          }
          // glb_m0 = store{r>g}(r2);
          if (v766_g) {
            int32_t v783_off = v23_lead + 4;
            #pragma unroll
            for (int32_t v778_i1 = 0; v778_i1 < 12; ++v778_i1) {
              float v780_data = r2[v778_i1];
              glb_m0[(v783_off + (v778_i1 * 12))] = v780_data;
            }
          }
          if (v23_lead >= 12) {
            int32_t v791_off = (v23_lead + -16_i32) + 4;
            #pragma unroll
            for (int32_t v787_z1 = 0; v787_z1 < 12; ++v787_z1) {
              glb_m0[(v791_off + (v787_z1 * 12))] = 0.0f;
            }
          }
          if ((v23_lead >= 4) && (v23_lead < 8)) {
            int32_t v801_off = v23_lead + 4;
            #pragma unroll
            for (int32_t v797_z1 = 0; v797_z1 < 12; ++v797_z1) {
              glb_m0[(v801_off + (v797_z1 * 12))] = 0.0f;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

