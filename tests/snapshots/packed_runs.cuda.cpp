// === base name ===
kernel_9613c07a105c1abc

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_9613c07a105c1abc = {{32, 4, 1}, 32, 40, 1, 4, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_9613c07a105c1abc(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_9613c07a105c1abc(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_9613c07a105c1abc(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_9613c07a105c1abc, block.x * block.y * block.z, 256 * sizeof(float));
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
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_9613c07a105c1abc(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_9613c07a105c1abc(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_9613c07a105c1abc, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_9613c07a105c1abc<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_9613c07a105c1abc(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: pack_operands=runs
    // launch: 32 lanes (40 active) x 4 per block = block 32x4x1, 1024 B shared, occupancy grid
    // operands:
    //   m0 40×6(40×6) {0..40}×{0..6} strided
    //   m1 40×8(40×8) {0..40}×{0..8} none ordered runs-packed
    //   m2 8×6(8×6) {0..8}×{0..6} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":40,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[40,6]],"name":"m0","ordered":false,"parts":1,"shape":[40,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[40,8]],"name":"m1","ordered":true,"parts":1,"shape":[40,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,6]],"name":"m2","ordered":false,"parts":1,"shape":[8,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[40,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[40,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[40,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[40,8]},{"addressing":"strided","bbox":[[0,0],[8,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[64 * threadIdx.y + 0];
      size_t v3_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < numElements0) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      const float *const __restrict__ glb_m1 = &m1[0];
      float * __restrict__ s0 = &localShrMem0[0];
      const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
      size_t v400_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v400_flagSecondAhead < numElements0) ? v400_flagSecondAhead : v3_batchId1)];
      const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 48 + 0 + m2_extraOffset];
      bool v408_g = threadIdx.x < 16;
      // s0 = load{g>s}(peel_glb_m2[0, 1])
      __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
      if (v408_g) {
        __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
      }
      __pipeline_commit();
      uint32_t v404_flagWord = flagWordFirst;
      uint32_t v405_flagWordNext = flagWordSecond;
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < numElements0; v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
        const float *const __restrict__ wrap_glb_m2 = &m2[v12_batchId1 * 48 + 0 + m2_extraOffset];
        size_t v422_flagAheadAhead = v12_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v422_flagAheadAhead < numElements0) ? v422_flagAheadAhead : v12_batchId1)];
        const bool allowed = static_cast<bool>(v404_flagWord);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 240 + 0 + m0_extraOffset];
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r0[12]{};
          // ir0 = +(glb_m1 * s0)
          // [(0, 40), (0, 6)] [(0, 8)]
          float ir0[12]{};
          int32_t v26_lead = threadIdx.x % 32;
          bool v30_g = v26_lead < 8;
          float v31_data = v30_g ? (glb_m1[v26_lead]) : (0.0f);
          __syncwarp();
          float v32_data = s0[0];
          ir0[0] = (v31_data * v32_data);
          float v37_data = s0[8];
          ir0[2] = (v31_data * v37_data);
          float v42_data = s0[16];
          ir0[4] = (v31_data * v42_data);
          float v47_data = s0[24];
          ir0[6] = (v31_data * v47_data);
          float v52_data = s0[32];
          ir0[8] = (v31_data * v52_data);
          float v57_data = s0[40];
          ir0[10] = (v31_data * v57_data);
          float v66_data = ((v26_lead >= 3) && (v26_lead < 21)) ? (glb_m1[(v26_lead + 5)]) : (0.0f);
          float v67_data = s0[1];
          float v69_data = ir0[0];
          ir0[0] = (v69_data + (v66_data * v67_data));
          float v72_data = s0[9];
          float v74_data = ir0[2];
          ir0[2] = (v74_data + (v66_data * v72_data));
          float v77_data = s0[17];
          float v79_data = ir0[4];
          ir0[4] = (v79_data + (v66_data * v77_data));
          float v82_data = s0[25];
          float v84_data = ir0[6];
          ir0[6] = (v84_data + (v66_data * v82_data));
          float v87_data = s0[33];
          float v89_data = ir0[8];
          ir0[8] = (v89_data + (v66_data * v87_data));
          float v92_data = s0[41];
          float v94_data = ir0[10];
          ir0[10] = (v94_data + (v66_data * v92_data));
          float v99_data = (v26_lead >= 28) ? (glb_m1[(v26_lead + -2)]) : (0.0f);
          float v100_data = s0[2];
          float v102_data = ir0[0];
          ir0[0] = (v102_data + (v99_data * v100_data));
          float v105_data = s0[10];
          float v107_data = ir0[2];
          ir0[2] = (v107_data + (v99_data * v105_data));
          float v110_data = s0[18];
          float v112_data = ir0[4];
          ir0[4] = (v112_data + (v99_data * v110_data));
          float v115_data = s0[26];
          float v117_data = ir0[6];
          ir0[6] = (v117_data + (v99_data * v115_data));
          float v120_data = s0[34];
          float v122_data = ir0[8];
          ir0[8] = (v122_data + (v99_data * v120_data));
          float v125_data = s0[42];
          float v127_data = ir0[10];
          ir0[10] = (v127_data + (v99_data * v125_data));
          int32_t v130_lead = v26_lead + 32_i32;
          float v134_data = (v26_lead < 4) ? (glb_m1[(v130_lead + -2)]) : (0.0f);
          ir0[1] = (v134_data * v100_data);
          ir0[3] = (v134_data * v105_data);
          ir0[5] = (v134_data * v110_data);
          ir0[7] = (v134_data * v115_data);
          ir0[9] = (v134_data * v120_data);
          ir0[11] = (v134_data * v125_data);
          float v168_data = ((v26_lead >= 1) && v30_g) ? (glb_m1[(v130_lead + 1)]) : (0.0f);
          float v169_data = s0[3];
          float v171_data = ir0[1];
          ir0[1] = (v171_data + (v168_data * v169_data));
          float v174_data = s0[11];
          float v176_data = ir0[3];
          ir0[3] = (v176_data + (v168_data * v174_data));
          float v179_data = s0[19];
          float v181_data = ir0[5];
          ir0[5] = (v181_data + (v168_data * v179_data));
          float v184_data = s0[27];
          float v186_data = ir0[7];
          ir0[7] = (v186_data + (v168_data * v184_data));
          float v189_data = s0[35];
          float v191_data = ir0[9];
          ir0[9] = (v191_data + (v168_data * v189_data));
          float v194_data = s0[43];
          float v196_data = ir0[11];
          ir0[11] = (v196_data + (v168_data * v194_data));
          float v203_data = ((v26_lead >= 17) && (v26_lead < 18)) ? (glb_m1[(v26_lead + 24)]) : (0.0f);
          float v204_data = s0[4];
          float v206_data = ir0[0];
          ir0[0] = (v206_data + (v203_data * v204_data));
          float v209_data = s0[12];
          float v211_data = ir0[2];
          ir0[2] = (v211_data + (v203_data * v209_data));
          float v214_data = s0[20];
          float v216_data = ir0[4];
          ir0[4] = (v216_data + (v203_data * v214_data));
          float v219_data = s0[28];
          float v221_data = ir0[6];
          ir0[6] = (v221_data + (v203_data * v219_data));
          float v224_data = s0[36];
          float v226_data = ir0[8];
          ir0[8] = (v226_data + (v203_data * v224_data));
          float v229_data = s0[44];
          float v231_data = ir0[10];
          ir0[10] = (v231_data + (v203_data * v229_data));
          float v236_data = (v26_lead >= 9) ? (glb_m1[(v26_lead + 33)]) : (0.0f);
          float v237_data = s0[5];
          float v239_data = ir0[0];
          ir0[0] = (v239_data + (v236_data * v237_data));
          float v242_data = s0[13];
          float v244_data = ir0[2];
          ir0[2] = (v244_data + (v236_data * v242_data));
          float v247_data = s0[21];
          float v249_data = ir0[4];
          ir0[4] = (v249_data + (v236_data * v247_data));
          float v252_data = s0[29];
          float v254_data = ir0[6];
          ir0[6] = (v254_data + (v236_data * v252_data));
          float v257_data = s0[37];
          float v259_data = ir0[8];
          ir0[8] = (v259_data + (v236_data * v257_data));
          float v262_data = s0[45];
          float v264_data = ir0[10];
          ir0[10] = (v264_data + (v236_data * v262_data));
          float v269_data = (v26_lead >= 31) ? (glb_m1[(v26_lead + 34)]) : (0.0f);
          float v270_data = s0[6];
          float v272_data = ir0[0];
          ir0[0] = (v272_data + (v269_data * v270_data));
          float v275_data = s0[14];
          float v277_data = ir0[2];
          ir0[2] = (v277_data + (v269_data * v275_data));
          float v280_data = s0[22];
          float v282_data = ir0[4];
          ir0[4] = (v282_data + (v269_data * v280_data));
          float v285_data = s0[30];
          float v287_data = ir0[6];
          ir0[6] = (v287_data + (v269_data * v285_data));
          float v290_data = s0[38];
          float v292_data = ir0[8];
          ir0[8] = (v292_data + (v269_data * v290_data));
          float v295_data = s0[46];
          float v297_data = ir0[10];
          ir0[10] = (v297_data + (v269_data * v295_data));
          float v302_data = (v26_lead < 1) ? (glb_m1[(v130_lead + 34)]) : (0.0f);
          float v305_data = ir0[1];
          ir0[1] = (v305_data + (v302_data * v270_data));
          float v310_data = ir0[3];
          ir0[3] = (v310_data + (v302_data * v275_data));
          float v315_data = ir0[5];
          ir0[5] = (v315_data + (v302_data * v280_data));
          float v320_data = ir0[7];
          ir0[7] = (v320_data + (v302_data * v285_data));
          float v325_data = ir0[9];
          ir0[9] = (v325_data + (v302_data * v290_data));
          float v330_data = ir0[11];
          ir0[11] = (v330_data + (v302_data * v295_data));
          float v337_data = ((v26_lead >= 12) && (v26_lead < 30)) ? (glb_m1[(v26_lead + 55)]) : (0.0f);
          float v338_data = s0[7];
          float v340_data = ir0[0];
          ir0[0] = (v340_data + (v337_data * v338_data));
          float v343_data = s0[15];
          float v345_data = ir0[2];
          ir0[2] = (v345_data + (v337_data * v343_data));
          float v348_data = s0[23];
          float v350_data = ir0[4];
          ir0[4] = (v350_data + (v337_data * v348_data));
          float v353_data = s0[31];
          float v355_data = ir0[6];
          ir0[6] = (v355_data + (v337_data * v353_data));
          float v358_data = s0[39];
          float v360_data = ir0[8];
          ir0[8] = (v360_data + (v337_data * v358_data));
          float v363_data = s0[47];
          float v365_data = ir0[10];
          ir0[10] = (v365_data + (v337_data * v363_data));
          // r0 = ir0
          #pragma unroll
          for (int32_t v369_n0 = 0; v369_n0 < 1; ++v369_n0) {
            #pragma unroll
            for (int32_t v370_n1 = 0; v370_n1 < 6; ++v370_n1) {
              int32_t v372_a = v369_n0 + (v370_n1 * 2);
              float v373_data = ir0[v372_a];
              r0[v372_a] = v373_data;
            }
          }
          if (v30_g) {
            #pragma unroll
            for (int32_t v375_n1 = 0; v375_n1 < 6; ++v375_n1) {
              int32_t v377_a = 1 + (v375_n1 * 2);
              float v378_data = ir0[v377_a];
              r0[v377_a] = v378_data;
            }
          }
          // glb_m0 = store{r>g}(r0);
          #pragma unroll
          for (int32_t v381_i0 = 0; v381_i0 < 1; ++v381_i0) {
            int32_t v387_lead = v26_lead + (v381_i0 * 32);
            #pragma unroll
            for (int32_t v382_i1 = 0; v382_i1 < 6; ++v382_i1) {
              float v385_data = r0[(v381_i0 + (v382_i1 * 2))];
              glb_m0[(v387_lead + (v382_i1 * 40))] = v385_data;
            }
          }
          if (v30_g) {
            #pragma unroll
            for (int32_t v391_i1 = 0; v391_i1 < 6; ++v391_i1) {
              float v394_data = r0[(1 + (v391_i1 * 2))];
              glb_m0[(v130_lead + (v391_i1 * 40))] = v394_data;
            }
          }
          __syncwarp();
        }
        else {
          __pipeline_wait_prior(0);
        }
        // s0 = load{g>s}(wrap_glb_m2[0, 1])
        __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
        if (v408_g) {
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
        }
        __pipeline_commit();
        v404_flagWord = v405_flagWordNext;
        v405_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
  }
}

