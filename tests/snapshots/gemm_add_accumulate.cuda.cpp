// === base name ===
kernel_8bf9187bc85b5201

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_8bf9187bc85b5201 = {{16, 8, 1}, 16, 12, 1, 8, 4608, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_8bf9187bc85b5201(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_8bf9187bc85b5201(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_8bf9187bc85b5201(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_8bf9187bc85b5201, block.x * block.y * block.z, 1152 * sizeof(float));
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
  config.sharedMemBytes = 1152 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_8bf9187bc85b5201(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_8bf9187bc85b5201(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_8bf9187bc85b5201, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_8bf9187bc85b5201<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_8bf9187bc85b5201(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 8 per block = block 16x8x1, 4608 B shared, occupancy grid
    // operands:
    //   m0 12×8(12×8) {0..12}×{0..8} strided
    //   m1 12×16(12×16) {0..12}×{0..16} strided
    //   m2 16×8(16×8) {0..16}×{0..8} strided
    // operations:
    //   m0[i,j] += m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1152}],"shared_bytes":4608,"shared_elements":1152,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,8]],"name":"m0","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,16]],"name":"m1","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,16]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[144 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[128];
      float * __restrict__ s0 = &localShrMem0[0];
      for (size_t v11_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v11_batchId0 < numElements0; v11_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v12_ahead1 = v11_batchId0 + (gridDim.x * blockDim.y);
        size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v11_batchId0 * 96 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v11_batchId0 * 192 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v11_batchId0 * 128 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v25_lead = threadIdx.x % 16;
          bool v26_g = v25_lead < 12;
          if (v26_g) {
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 16; ++v27_i1) {
              float v32_data = __ldcg(&glb_m1[(v25_lead + (v27_i1 * 12))]);
              r0[v27_i1] = v32_data;
            }
          }
          // s0 = load{g>s}(glb_m2[0, 1])
          #pragma unroll
          for (int32_t i = 0; i < 8; i += 1) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + i * 16], &glb_m2[0 + 0 + 1 * threadIdx.x + i * 16], 4);
          }
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          float r1[8]{};
          // r1 = load{g>r}(glb_m0);
          if (v26_g) {
            #pragma unroll
            for (int32_t v36_i1 = 0; v36_i1 < 8; ++v36_i1) {
              float v41_data = glb_m0[(v25_lead + (v36_i1 * 12))];
              r1[v36_i1] = v41_data;
            }
          }
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          // wait(r1 = load{g>r}(glb_m0););
          float r2[8]{};
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          // ir2 = +(r0 * s0)
          // [(0, 12), (0, 8)] [(0, 16)]
          float ir2[8]{};
          float v45_data = r0[0];
          float v46_data = s0[0];
          float v48_data = ir2[0];
          ir2[0] = (v48_data + (v45_data * v46_data));
          float v51_data = s0[16];
          float v53_data = ir2[1];
          ir2[1] = (v53_data + (v45_data * v51_data));
          float v56_data = s0[32];
          float v58_data = ir2[2];
          ir2[2] = (v58_data + (v45_data * v56_data));
          float v61_data = s0[48];
          float v63_data = ir2[3];
          ir2[3] = (v63_data + (v45_data * v61_data));
          float v66_data = s0[64];
          float v68_data = ir2[4];
          ir2[4] = (v68_data + (v45_data * v66_data));
          float v71_data = s0[80];
          float v73_data = ir2[5];
          ir2[5] = (v73_data + (v45_data * v71_data));
          float v76_data = s0[96];
          float v78_data = ir2[6];
          ir2[6] = (v78_data + (v45_data * v76_data));
          float v81_data = s0[112];
          float v83_data = ir2[7];
          ir2[7] = (v83_data + (v45_data * v81_data));
          float v85_data = r0[1];
          float v86_data = s0[1];
          float v88_data = ir2[0];
          ir2[0] = (v88_data + (v85_data * v86_data));
          float v91_data = s0[17];
          float v93_data = ir2[1];
          ir2[1] = (v93_data + (v85_data * v91_data));
          float v96_data = s0[33];
          float v98_data = ir2[2];
          ir2[2] = (v98_data + (v85_data * v96_data));
          float v101_data = s0[49];
          float v103_data = ir2[3];
          ir2[3] = (v103_data + (v85_data * v101_data));
          float v106_data = s0[65];
          float v108_data = ir2[4];
          ir2[4] = (v108_data + (v85_data * v106_data));
          float v111_data = s0[81];
          float v113_data = ir2[5];
          ir2[5] = (v113_data + (v85_data * v111_data));
          float v116_data = s0[97];
          float v118_data = ir2[6];
          ir2[6] = (v118_data + (v85_data * v116_data));
          float v121_data = s0[113];
          float v123_data = ir2[7];
          ir2[7] = (v123_data + (v85_data * v121_data));
          float v125_data = r0[2];
          float v126_data = s0[2];
          float v128_data = ir2[0];
          ir2[0] = (v128_data + (v125_data * v126_data));
          float v131_data = s0[18];
          float v133_data = ir2[1];
          ir2[1] = (v133_data + (v125_data * v131_data));
          float v136_data = s0[34];
          float v138_data = ir2[2];
          ir2[2] = (v138_data + (v125_data * v136_data));
          float v141_data = s0[50];
          float v143_data = ir2[3];
          ir2[3] = (v143_data + (v125_data * v141_data));
          float v146_data = s0[66];
          float v148_data = ir2[4];
          ir2[4] = (v148_data + (v125_data * v146_data));
          float v151_data = s0[82];
          float v153_data = ir2[5];
          ir2[5] = (v153_data + (v125_data * v151_data));
          float v156_data = s0[98];
          float v158_data = ir2[6];
          ir2[6] = (v158_data + (v125_data * v156_data));
          float v161_data = s0[114];
          float v163_data = ir2[7];
          ir2[7] = (v163_data + (v125_data * v161_data));
          float v165_data = r0[3];
          float v166_data = s0[3];
          float v168_data = ir2[0];
          ir2[0] = (v168_data + (v165_data * v166_data));
          float v171_data = s0[19];
          float v173_data = ir2[1];
          ir2[1] = (v173_data + (v165_data * v171_data));
          float v176_data = s0[35];
          float v178_data = ir2[2];
          ir2[2] = (v178_data + (v165_data * v176_data));
          float v181_data = s0[51];
          float v183_data = ir2[3];
          ir2[3] = (v183_data + (v165_data * v181_data));
          float v186_data = s0[67];
          float v188_data = ir2[4];
          ir2[4] = (v188_data + (v165_data * v186_data));
          float v191_data = s0[83];
          float v193_data = ir2[5];
          ir2[5] = (v193_data + (v165_data * v191_data));
          float v196_data = s0[99];
          float v198_data = ir2[6];
          ir2[6] = (v198_data + (v165_data * v196_data));
          float v201_data = s0[115];
          float v203_data = ir2[7];
          ir2[7] = (v203_data + (v165_data * v201_data));
          float v205_data = r0[4];
          float v206_data = s0[4];
          float v208_data = ir2[0];
          ir2[0] = (v208_data + (v205_data * v206_data));
          float v211_data = s0[20];
          float v213_data = ir2[1];
          ir2[1] = (v213_data + (v205_data * v211_data));
          float v216_data = s0[36];
          float v218_data = ir2[2];
          ir2[2] = (v218_data + (v205_data * v216_data));
          float v221_data = s0[52];
          float v223_data = ir2[3];
          ir2[3] = (v223_data + (v205_data * v221_data));
          float v226_data = s0[68];
          float v228_data = ir2[4];
          ir2[4] = (v228_data + (v205_data * v226_data));
          float v231_data = s0[84];
          float v233_data = ir2[5];
          ir2[5] = (v233_data + (v205_data * v231_data));
          float v236_data = s0[100];
          float v238_data = ir2[6];
          ir2[6] = (v238_data + (v205_data * v236_data));
          float v241_data = s0[116];
          float v243_data = ir2[7];
          ir2[7] = (v243_data + (v205_data * v241_data));
          float v245_data = r0[5];
          float v246_data = s0[5];
          float v248_data = ir2[0];
          ir2[0] = (v248_data + (v245_data * v246_data));
          float v251_data = s0[21];
          float v253_data = ir2[1];
          ir2[1] = (v253_data + (v245_data * v251_data));
          float v256_data = s0[37];
          float v258_data = ir2[2];
          ir2[2] = (v258_data + (v245_data * v256_data));
          float v261_data = s0[53];
          float v263_data = ir2[3];
          ir2[3] = (v263_data + (v245_data * v261_data));
          float v266_data = s0[69];
          float v268_data = ir2[4];
          ir2[4] = (v268_data + (v245_data * v266_data));
          float v271_data = s0[85];
          float v273_data = ir2[5];
          ir2[5] = (v273_data + (v245_data * v271_data));
          float v276_data = s0[101];
          float v278_data = ir2[6];
          ir2[6] = (v278_data + (v245_data * v276_data));
          float v281_data = s0[117];
          float v283_data = ir2[7];
          ir2[7] = (v283_data + (v245_data * v281_data));
          float v285_data = r0[6];
          float v286_data = s0[6];
          float v288_data = ir2[0];
          ir2[0] = (v288_data + (v285_data * v286_data));
          float v291_data = s0[22];
          float v293_data = ir2[1];
          ir2[1] = (v293_data + (v285_data * v291_data));
          float v296_data = s0[38];
          float v298_data = ir2[2];
          ir2[2] = (v298_data + (v285_data * v296_data));
          float v301_data = s0[54];
          float v303_data = ir2[3];
          ir2[3] = (v303_data + (v285_data * v301_data));
          float v306_data = s0[70];
          float v308_data = ir2[4];
          ir2[4] = (v308_data + (v285_data * v306_data));
          float v311_data = s0[86];
          float v313_data = ir2[5];
          ir2[5] = (v313_data + (v285_data * v311_data));
          float v316_data = s0[102];
          float v318_data = ir2[6];
          ir2[6] = (v318_data + (v285_data * v316_data));
          float v321_data = s0[118];
          float v323_data = ir2[7];
          ir2[7] = (v323_data + (v285_data * v321_data));
          float v325_data = r0[7];
          float v326_data = s0[7];
          float v328_data = ir2[0];
          ir2[0] = (v328_data + (v325_data * v326_data));
          float v331_data = s0[23];
          float v333_data = ir2[1];
          ir2[1] = (v333_data + (v325_data * v331_data));
          float v336_data = s0[39];
          float v338_data = ir2[2];
          ir2[2] = (v338_data + (v325_data * v336_data));
          float v341_data = s0[55];
          float v343_data = ir2[3];
          ir2[3] = (v343_data + (v325_data * v341_data));
          float v346_data = s0[71];
          float v348_data = ir2[4];
          ir2[4] = (v348_data + (v325_data * v346_data));
          float v351_data = s0[87];
          float v353_data = ir2[5];
          ir2[5] = (v353_data + (v325_data * v351_data));
          float v356_data = s0[103];
          float v358_data = ir2[6];
          ir2[6] = (v358_data + (v325_data * v356_data));
          float v361_data = s0[119];
          float v363_data = ir2[7];
          ir2[7] = (v363_data + (v325_data * v361_data));
          float v365_data = r0[8];
          float v366_data = s0[8];
          float v368_data = ir2[0];
          ir2[0] = (v368_data + (v365_data * v366_data));
          float v371_data = s0[24];
          float v373_data = ir2[1];
          ir2[1] = (v373_data + (v365_data * v371_data));
          float v376_data = s0[40];
          float v378_data = ir2[2];
          ir2[2] = (v378_data + (v365_data * v376_data));
          float v381_data = s0[56];
          float v383_data = ir2[3];
          ir2[3] = (v383_data + (v365_data * v381_data));
          float v386_data = s0[72];
          float v388_data = ir2[4];
          ir2[4] = (v388_data + (v365_data * v386_data));
          float v391_data = s0[88];
          float v393_data = ir2[5];
          ir2[5] = (v393_data + (v365_data * v391_data));
          float v396_data = s0[104];
          float v398_data = ir2[6];
          ir2[6] = (v398_data + (v365_data * v396_data));
          float v401_data = s0[120];
          float v403_data = ir2[7];
          ir2[7] = (v403_data + (v365_data * v401_data));
          float v405_data = r0[9];
          float v406_data = s0[9];
          float v408_data = ir2[0];
          ir2[0] = (v408_data + (v405_data * v406_data));
          float v411_data = s0[25];
          float v413_data = ir2[1];
          ir2[1] = (v413_data + (v405_data * v411_data));
          float v416_data = s0[41];
          float v418_data = ir2[2];
          ir2[2] = (v418_data + (v405_data * v416_data));
          float v421_data = s0[57];
          float v423_data = ir2[3];
          ir2[3] = (v423_data + (v405_data * v421_data));
          float v426_data = s0[73];
          float v428_data = ir2[4];
          ir2[4] = (v428_data + (v405_data * v426_data));
          float v431_data = s0[89];
          float v433_data = ir2[5];
          ir2[5] = (v433_data + (v405_data * v431_data));
          float v436_data = s0[105];
          float v438_data = ir2[6];
          ir2[6] = (v438_data + (v405_data * v436_data));
          float v441_data = s0[121];
          float v443_data = ir2[7];
          ir2[7] = (v443_data + (v405_data * v441_data));
          float v445_data = r0[10];
          float v446_data = s0[10];
          float v448_data = ir2[0];
          ir2[0] = (v448_data + (v445_data * v446_data));
          float v451_data = s0[26];
          float v453_data = ir2[1];
          ir2[1] = (v453_data + (v445_data * v451_data));
          float v456_data = s0[42];
          float v458_data = ir2[2];
          ir2[2] = (v458_data + (v445_data * v456_data));
          float v461_data = s0[58];
          float v463_data = ir2[3];
          ir2[3] = (v463_data + (v445_data * v461_data));
          float v466_data = s0[74];
          float v468_data = ir2[4];
          ir2[4] = (v468_data + (v445_data * v466_data));
          float v471_data = s0[90];
          float v473_data = ir2[5];
          ir2[5] = (v473_data + (v445_data * v471_data));
          float v476_data = s0[106];
          float v478_data = ir2[6];
          ir2[6] = (v478_data + (v445_data * v476_data));
          float v481_data = s0[122];
          float v483_data = ir2[7];
          ir2[7] = (v483_data + (v445_data * v481_data));
          float v485_data = r0[11];
          float v486_data = s0[11];
          float v488_data = ir2[0];
          ir2[0] = (v488_data + (v485_data * v486_data));
          float v491_data = s0[27];
          float v493_data = ir2[1];
          ir2[1] = (v493_data + (v485_data * v491_data));
          float v496_data = s0[43];
          float v498_data = ir2[2];
          ir2[2] = (v498_data + (v485_data * v496_data));
          float v501_data = s0[59];
          float v503_data = ir2[3];
          ir2[3] = (v503_data + (v485_data * v501_data));
          float v506_data = s0[75];
          float v508_data = ir2[4];
          ir2[4] = (v508_data + (v485_data * v506_data));
          float v511_data = s0[91];
          float v513_data = ir2[5];
          ir2[5] = (v513_data + (v485_data * v511_data));
          float v516_data = s0[107];
          float v518_data = ir2[6];
          ir2[6] = (v518_data + (v485_data * v516_data));
          float v521_data = s0[123];
          float v523_data = ir2[7];
          ir2[7] = (v523_data + (v485_data * v521_data));
          float v525_data = r0[12];
          float v526_data = s0[12];
          float v528_data = ir2[0];
          ir2[0] = (v528_data + (v525_data * v526_data));
          float v531_data = s0[28];
          float v533_data = ir2[1];
          ir2[1] = (v533_data + (v525_data * v531_data));
          float v536_data = s0[44];
          float v538_data = ir2[2];
          ir2[2] = (v538_data + (v525_data * v536_data));
          float v541_data = s0[60];
          float v543_data = ir2[3];
          ir2[3] = (v543_data + (v525_data * v541_data));
          float v546_data = s0[76];
          float v548_data = ir2[4];
          ir2[4] = (v548_data + (v525_data * v546_data));
          float v551_data = s0[92];
          float v553_data = ir2[5];
          ir2[5] = (v553_data + (v525_data * v551_data));
          float v556_data = s0[108];
          float v558_data = ir2[6];
          ir2[6] = (v558_data + (v525_data * v556_data));
          float v561_data = s0[124];
          float v563_data = ir2[7];
          ir2[7] = (v563_data + (v525_data * v561_data));
          float v565_data = r0[13];
          float v566_data = s0[13];
          float v568_data = ir2[0];
          ir2[0] = (v568_data + (v565_data * v566_data));
          float v571_data = s0[29];
          float v573_data = ir2[1];
          ir2[1] = (v573_data + (v565_data * v571_data));
          float v576_data = s0[45];
          float v578_data = ir2[2];
          ir2[2] = (v578_data + (v565_data * v576_data));
          float v581_data = s0[61];
          float v583_data = ir2[3];
          ir2[3] = (v583_data + (v565_data * v581_data));
          float v586_data = s0[77];
          float v588_data = ir2[4];
          ir2[4] = (v588_data + (v565_data * v586_data));
          float v591_data = s0[93];
          float v593_data = ir2[5];
          ir2[5] = (v593_data + (v565_data * v591_data));
          float v596_data = s0[109];
          float v598_data = ir2[6];
          ir2[6] = (v598_data + (v565_data * v596_data));
          float v601_data = s0[125];
          float v603_data = ir2[7];
          ir2[7] = (v603_data + (v565_data * v601_data));
          float v605_data = r0[14];
          float v606_data = s0[14];
          float v608_data = ir2[0];
          ir2[0] = (v608_data + (v605_data * v606_data));
          float v611_data = s0[30];
          float v613_data = ir2[1];
          ir2[1] = (v613_data + (v605_data * v611_data));
          float v616_data = s0[46];
          float v618_data = ir2[2];
          ir2[2] = (v618_data + (v605_data * v616_data));
          float v621_data = s0[62];
          float v623_data = ir2[3];
          ir2[3] = (v623_data + (v605_data * v621_data));
          float v626_data = s0[78];
          float v628_data = ir2[4];
          ir2[4] = (v628_data + (v605_data * v626_data));
          float v631_data = s0[94];
          float v633_data = ir2[5];
          ir2[5] = (v633_data + (v605_data * v631_data));
          float v636_data = s0[110];
          float v638_data = ir2[6];
          ir2[6] = (v638_data + (v605_data * v636_data));
          float v641_data = s0[126];
          float v643_data = ir2[7];
          ir2[7] = (v643_data + (v605_data * v641_data));
          float v645_data = r0[15];
          float v646_data = s0[15];
          float v648_data = ir2[0];
          ir2[0] = (v648_data + (v645_data * v646_data));
          float v651_data = s0[31];
          float v653_data = ir2[1];
          ir2[1] = (v653_data + (v645_data * v651_data));
          float v656_data = s0[47];
          float v658_data = ir2[2];
          ir2[2] = (v658_data + (v645_data * v656_data));
          float v661_data = s0[63];
          float v663_data = ir2[3];
          ir2[3] = (v663_data + (v645_data * v661_data));
          float v666_data = s0[79];
          float v668_data = ir2[4];
          ir2[4] = (v668_data + (v645_data * v666_data));
          float v671_data = s0[95];
          float v673_data = ir2[5];
          ir2[5] = (v673_data + (v645_data * v671_data));
          float v676_data = s0[111];
          float v678_data = ir2[6];
          ir2[6] = (v678_data + (v645_data * v676_data));
          float v681_data = s0[127];
          float v683_data = ir2[7];
          ir2[7] = (v683_data + (v645_data * v681_data));
          // r2 = ir2 + r1
          if (v26_g) {
            #pragma unroll
            for (int32_t v685_n1 = 0; v685_n1 < 8; ++v685_n1) {
              float v687_data = ir2[v685_n1];
              float v688_data = r1[v685_n1];
              r2[v685_n1] = (v688_data + v687_data);
            }
          }
          // glb_m0 = store{r>g}(r2);
          if (v26_g) {
            #pragma unroll
            for (int32_t v690_i1 = 0; v690_i1 < 8; ++v690_i1) {
              float v692_data = r2[v690_i1];
              glb_m0[(v25_lead + (v690_i1 * 12))] = v692_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
      }
    }
  }
}

