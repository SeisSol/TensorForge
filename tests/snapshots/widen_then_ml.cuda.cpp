// === base name ===
kernel_e60846de29b9932f

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_e60846de29b9932f = {{32, 4, 1}, 32, 20, 1, 4, 3072, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_e60846de29b9932f(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_e60846de29b9932f(const float * m0, size_t m0_extraOffset, double * m1, size_t m1_extraOffset, const double * m2, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_e60846de29b9932f(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_e60846de29b9932f, block.x * block.y * block.z, 768 * sizeof(float));
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
void launcher_kernel_e60846de29b9932f(const float * m0, size_t m0_extraOffset, double * m1, size_t m1_extraOffset, const double * m2, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_e60846de29b9932f(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_e60846de29b9932f, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  const unsigned * flags0Arg = (const unsigned *)flags0;
  kernel_kernel_e60846de29b9932f<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m1_extraOffset, m2, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_e60846de29b9932f(const float * m0, size_t m0_extraOffset, double * m1, size_t m1_extraOffset, const double * m2, size_t numElements0, const unsigned * __restrict__ flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (20 active) x 4 per block = block 32x4x1, 3072 B shared, occupancy grid
    // operands:
    //   m0 10×9(10×9) {0..10}×{0..9} strided
    //   m1 20×9(20×9) {0..20}×{0..9} strided
    //   m2 20×10(20×10) {0..20}×{0..10} none
    // operations:
    //   wide = copy(I)
    //   m1[i,j] = m2[i,k] × t0[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":20,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":768}],"shared_bytes":3072,"shared_elements":768,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"I","bbox":[[0,0],[10,9]],"name":"m0","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"strided","alias":"canonical","bbox":[[0,0],[20,9]],"name":"m1","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"none","alias":"embed","bbox":[[0,0],[20,10]],"name":"m2","ordered":false,"parts":1,"shape":[20,10],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]},"kind":"elementwise","op":"COPY","ops":[{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,10]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,10]},{"addressing":"pointer_based","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[192 * threadIdx.y + 0];
      size_t v3_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < numElements0) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      const double *const __restrict__ glb_m2 = &m2[0];
      double * __restrict__ s0 = reinterpret_cast<double*>(&localShrMem0[0]);
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < numElements0; v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 90 + 0 + m0_extraOffset];
          double *const __restrict__ glb_m1 = &m1[v9_batchId0 * 180 + 0 + m1_extraOffset];
          double r0[9]{};
          // r0 = copy(glb_m0)
          int32_t v22_lead = threadIdx.x % 32;
          bool v23_g = v22_lead < 10;
          if (v23_g) {
            #pragma unroll
            for (int32_t v24_k1 = 0; v24_k1 < 9; ++v24_k1) {
              float v29_data = glb_m0[(v22_lead + (v24_k1 * 10))];
              r0[v24_k1] = v29_data;
            }
          }
          // s0 = store{r>s}(localShrMem0, r0);
          if (v23_g) {
            #pragma unroll
            for (int32_t v36_i1 = 0; v36_i1 < 9; ++v36_i1) {
              double v38_data = r0[v36_i1];
              s0[(v22_lead + (v36_i1 * 10))] = v38_data;
            }
          }
          double r1[9]{};
          // ir1 = +(glb_m2 * s0)
          // [(0, 20), (0, 9)] [(0, 10)]
          double ir1[9]{};
          bool v48_g = v22_lead < 20;
          double v49_data = v48_g ? (glb_m2[v22_lead]) : (0.0);
          __syncwarp();
          double v50_data = s0[0];
          double v55_data = s0[10];
          double v60_data = s0[20];
          double v65_data = s0[30];
          double v70_data = s0[40];
          double v75_data = s0[50];
          double v80_data = s0[60];
          double v85_data = s0[70];
          double v90_data = s0[80];
          double v95_data = v48_g ? (glb_m2[(v22_lead + 20)]) : (0.0);
          double v96_data = s0[1];
          double v101_data = s0[11];
          double v106_data = s0[21];
          double v111_data = s0[31];
          double v116_data = s0[41];
          double v121_data = s0[51];
          double v126_data = s0[61];
          double v131_data = s0[71];
          double v136_data = s0[81];
          double v141_data = v48_g ? (glb_m2[(v22_lead + 40)]) : (0.0);
          double v142_data = s0[2];
          double v147_data = s0[12];
          double v152_data = s0[22];
          double v157_data = s0[32];
          double v162_data = s0[42];
          double v167_data = s0[52];
          double v172_data = s0[62];
          double v177_data = s0[72];
          double v182_data = s0[82];
          double v187_data = v48_g ? (glb_m2[(v22_lead + 60)]) : (0.0);
          double v188_data = s0[3];
          double v193_data = s0[13];
          double v198_data = s0[23];
          double v203_data = s0[33];
          double v208_data = s0[43];
          double v213_data = s0[53];
          double v218_data = s0[63];
          double v223_data = s0[73];
          double v228_data = s0[83];
          double v233_data = v48_g ? (glb_m2[(v22_lead + 80)]) : (0.0);
          double v234_data = s0[4];
          double v239_data = s0[14];
          double v244_data = s0[24];
          double v249_data = s0[34];
          double v254_data = s0[44];
          double v259_data = s0[54];
          double v264_data = s0[64];
          double v269_data = s0[74];
          double v274_data = s0[84];
          double v279_data = v48_g ? (glb_m2[(v22_lead + 100)]) : (0.0);
          double v280_data = s0[5];
          double v285_data = s0[15];
          double v290_data = s0[25];
          double v295_data = s0[35];
          double v300_data = s0[45];
          double v305_data = s0[55];
          double v310_data = s0[65];
          double v315_data = s0[75];
          double v320_data = s0[85];
          double v325_data = v48_g ? (glb_m2[(v22_lead + 120)]) : (0.0);
          double v326_data = s0[6];
          double v331_data = s0[16];
          double v336_data = s0[26];
          double v341_data = s0[36];
          double v346_data = s0[46];
          double v351_data = s0[56];
          double v356_data = s0[66];
          double v361_data = s0[76];
          double v366_data = s0[86];
          double v371_data = v48_g ? (glb_m2[(v22_lead + 140)]) : (0.0);
          double v372_data = s0[7];
          double v377_data = s0[17];
          double v382_data = s0[27];
          double v387_data = s0[37];
          double v392_data = s0[47];
          double v397_data = s0[57];
          double v402_data = s0[67];
          double v407_data = s0[77];
          double v412_data = s0[87];
          double v417_data = v48_g ? (glb_m2[(v22_lead + 160)]) : (0.0);
          double v418_data = s0[8];
          double v423_data = s0[18];
          double v426_p = ((((((((v49_data * v55_data) + (v95_data * v101_data)) + (v141_data * v147_data)) + (v187_data * v193_data)) + (v233_data * v239_data)) + (v279_data * v285_data)) + (v325_data * v331_data)) + (v371_data * v377_data)) + (v417_data * v423_data);
          double v428_data = s0[28];
          double v431_p = ((((((((v49_data * v60_data) + (v95_data * v106_data)) + (v141_data * v152_data)) + (v187_data * v198_data)) + (v233_data * v244_data)) + (v279_data * v290_data)) + (v325_data * v336_data)) + (v371_data * v382_data)) + (v417_data * v428_data);
          double v433_data = s0[38];
          double v436_p = ((((((((v49_data * v65_data) + (v95_data * v111_data)) + (v141_data * v157_data)) + (v187_data * v203_data)) + (v233_data * v249_data)) + (v279_data * v295_data)) + (v325_data * v341_data)) + (v371_data * v387_data)) + (v417_data * v433_data);
          double v438_data = s0[48];
          double v441_p = ((((((((v49_data * v70_data) + (v95_data * v116_data)) + (v141_data * v162_data)) + (v187_data * v208_data)) + (v233_data * v254_data)) + (v279_data * v300_data)) + (v325_data * v346_data)) + (v371_data * v392_data)) + (v417_data * v438_data);
          double v443_data = s0[58];
          double v446_p = ((((((((v49_data * v75_data) + (v95_data * v121_data)) + (v141_data * v167_data)) + (v187_data * v213_data)) + (v233_data * v259_data)) + (v279_data * v305_data)) + (v325_data * v351_data)) + (v371_data * v397_data)) + (v417_data * v443_data);
          double v448_data = s0[68];
          double v451_p = ((((((((v49_data * v80_data) + (v95_data * v126_data)) + (v141_data * v172_data)) + (v187_data * v218_data)) + (v233_data * v264_data)) + (v279_data * v310_data)) + (v325_data * v356_data)) + (v371_data * v402_data)) + (v417_data * v448_data);
          double v453_data = s0[78];
          double v456_p = ((((((((v49_data * v85_data) + (v95_data * v131_data)) + (v141_data * v177_data)) + (v187_data * v223_data)) + (v233_data * v269_data)) + (v279_data * v315_data)) + (v325_data * v361_data)) + (v371_data * v407_data)) + (v417_data * v453_data);
          double v458_data = s0[88];
          double v461_p = ((((((((v49_data * v90_data) + (v95_data * v136_data)) + (v141_data * v182_data)) + (v187_data * v228_data)) + (v233_data * v274_data)) + (v279_data * v320_data)) + (v325_data * v366_data)) + (v371_data * v412_data)) + (v417_data * v458_data);
          double v463_data = v48_g ? (glb_m2[(v22_lead + 180)]) : (0.0);
          double v464_data = s0[9];
          ir1[0] = ((((((((((v49_data * v50_data) + (v95_data * v96_data)) + (v141_data * v142_data)) + (v187_data * v188_data)) + (v233_data * v234_data)) + (v279_data * v280_data)) + (v325_data * v326_data)) + (v371_data * v372_data)) + (v417_data * v418_data)) + (v463_data * v464_data));
          double v469_data = s0[19];
          ir1[1] = (v426_p + (v463_data * v469_data));
          double v474_data = s0[29];
          ir1[2] = (v431_p + (v463_data * v474_data));
          double v479_data = s0[39];
          ir1[3] = (v436_p + (v463_data * v479_data));
          double v484_data = s0[49];
          ir1[4] = (v441_p + (v463_data * v484_data));
          double v489_data = s0[59];
          ir1[5] = (v446_p + (v463_data * v489_data));
          double v494_data = s0[69];
          ir1[6] = (v451_p + (v463_data * v494_data));
          double v499_data = s0[79];
          ir1[7] = (v456_p + (v463_data * v499_data));
          double v504_data = s0[89];
          ir1[8] = (v461_p + (v463_data * v504_data));
          // r1 = ir1
          // glb_m1 = store{r>g}(r1);
          if (v48_g) {
            #pragma unroll
            for (int32_t v513_i1 = 0; v513_i1 < 9; ++v513_i1) {
              double v515_data = ir1[v513_i1];
              glb_m1[(v22_lead + (v513_i1 * 20))] = v515_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

