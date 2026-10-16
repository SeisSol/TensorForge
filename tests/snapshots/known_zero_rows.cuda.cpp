// === base name ===
kernel_3ad9bee91694bab9

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_3ad9bee91694bab9 = {{32, 4, 1}, 32, 40, 1, 4, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_3ad9bee91694bab9(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_3ad9bee91694bab9(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_3ad9bee91694bab9(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_3ad9bee91694bab9, block.x * block.y * block.z, 256 * sizeof(float));
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
void launcher_kernel_3ad9bee91694bab9(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_3ad9bee91694bab9(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_3ad9bee91694bab9, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_3ad9bee91694bab9<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_3ad9bee91694bab9(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
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
      size_t v482_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v482_flagSecondAhead < numElements0) ? v482_flagSecondAhead : v3_batchId1)];
      const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 48 + 0 + m2_extraOffset];
      bool v490_g = threadIdx.x < 16;
      // s0 = load{g>s}(peel_glb_m2[0, 1])
      __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
      if (v490_g) {
        __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
      }
      __pipeline_commit();
      uint32_t v486_flagWord = flagWordFirst;
      uint32_t v487_flagWordNext = flagWordSecond;
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < numElements0; v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
        const float *const __restrict__ wrap_glb_m2 = &m2[v12_batchId1 * 48 + 0 + m2_extraOffset];
        size_t v504_flagAheadAhead = v12_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v504_flagAheadAhead < numElements0) ? v504_flagAheadAhead : v12_batchId1)];
        const bool allowed = static_cast<bool>(v486_flagWord);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 240 + 0 + m0_extraOffset];
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r0[12]{};
          // ir0 = +(glb_m1 * s0)
          // [(0, 40), (0, 6)] [(0, 8)]
          float ir0[12]{};
          int32_t v26_lead = threadIdx.x % 32;
          int32_t v28_lead = v26_lead + 32_i32;
          bool v31_g = v26_lead < 8;
          float v32_data = v31_g ? (glb_m1[(v28_lead + -32)]) : (0.0f);
          __syncwarp();
          float v33_data = s0[0];
          ir0[1] = (v32_data * v33_data);
          float v38_data = s0[8];
          ir0[3] = (v32_data * v38_data);
          float v43_data = s0[16];
          ir0[5] = (v32_data * v43_data);
          float v48_data = s0[24];
          ir0[7] = (v32_data * v48_data);
          float v53_data = s0[32];
          ir0[9] = (v32_data * v53_data);
          float v58_data = s0[40];
          ir0[11] = (v32_data * v58_data);
          float v66_data = glb_m1[(v26_lead + 8)];
          float v67_data = s0[1];
          ir0[0] = (v66_data * v67_data);
          float v72_data = s0[9];
          ir0[2] = (v66_data * v72_data);
          float v77_data = s0[17];
          ir0[4] = (v66_data * v77_data);
          float v82_data = s0[25];
          ir0[6] = (v66_data * v82_data);
          float v87_data = s0[33];
          ir0[8] = (v66_data * v87_data);
          float v92_data = s0[41];
          ir0[10] = (v66_data * v92_data);
          float v98_data = v31_g ? (glb_m1[(v28_lead + 8)]) : (0.0f);
          float v101_data = ir0[1];
          ir0[1] = (v101_data + (v98_data * v67_data));
          float v106_data = ir0[3];
          ir0[3] = (v106_data + (v98_data * v72_data));
          float v111_data = ir0[5];
          ir0[5] = (v111_data + (v98_data * v77_data));
          float v116_data = ir0[7];
          ir0[7] = (v116_data + (v98_data * v82_data));
          float v121_data = ir0[9];
          ir0[9] = (v121_data + (v98_data * v87_data));
          float v126_data = ir0[11];
          ir0[11] = (v126_data + (v98_data * v92_data));
          float v130_data = v31_g ? (glb_m1[(v28_lead + 16)]) : (0.0f);
          float v131_data = s0[2];
          float v133_data = ir0[1];
          ir0[1] = (v133_data + (v130_data * v131_data));
          float v136_data = s0[10];
          float v138_data = ir0[3];
          ir0[3] = (v138_data + (v130_data * v136_data));
          float v141_data = s0[18];
          float v143_data = ir0[5];
          ir0[5] = (v143_data + (v130_data * v141_data));
          float v146_data = s0[26];
          float v148_data = ir0[7];
          ir0[7] = (v148_data + (v130_data * v146_data));
          float v151_data = s0[34];
          float v153_data = ir0[9];
          ir0[9] = (v153_data + (v130_data * v151_data));
          float v156_data = s0[42];
          float v158_data = ir0[11];
          ir0[11] = (v158_data + (v130_data * v156_data));
          float v163_data = (v26_lead >= 17) ? (glb_m1[(v26_lead + 39)]) : (0.0f);
          float v164_data = s0[3];
          float v166_data = ir0[0];
          ir0[0] = (v166_data + (v163_data * v164_data));
          float v169_data = s0[11];
          float v171_data = ir0[2];
          ir0[2] = (v171_data + (v163_data * v169_data));
          float v174_data = s0[19];
          float v176_data = ir0[4];
          ir0[4] = (v176_data + (v163_data * v174_data));
          float v179_data = s0[27];
          float v181_data = ir0[6];
          ir0[6] = (v181_data + (v163_data * v179_data));
          float v184_data = s0[35];
          float v186_data = ir0[8];
          ir0[8] = (v186_data + (v163_data * v184_data));
          float v189_data = s0[43];
          float v191_data = ir0[10];
          ir0[10] = (v191_data + (v163_data * v189_data));
          float v195_data = v31_g ? (glb_m1[(v28_lead + 39)]) : (0.0f);
          float v198_data = ir0[1];
          ir0[1] = (v198_data + (v195_data * v164_data));
          float v203_data = ir0[3];
          ir0[3] = (v203_data + (v195_data * v169_data));
          float v208_data = ir0[5];
          ir0[5] = (v208_data + (v195_data * v174_data));
          float v213_data = ir0[7];
          ir0[7] = (v213_data + (v195_data * v179_data));
          float v218_data = ir0[9];
          ir0[9] = (v218_data + (v195_data * v184_data));
          float v223_data = ir0[11];
          ir0[11] = (v223_data + (v195_data * v189_data));
          float v227_data = glb_m1[(v26_lead + 79)];
          float v228_data = s0[4];
          float v230_data = ir0[0];
          ir0[0] = (v230_data + (v227_data * v228_data));
          float v233_data = s0[12];
          float v235_data = ir0[2];
          ir0[2] = (v235_data + (v227_data * v233_data));
          float v238_data = s0[20];
          float v240_data = ir0[4];
          ir0[4] = (v240_data + (v227_data * v238_data));
          float v243_data = s0[28];
          float v245_data = ir0[6];
          ir0[6] = (v245_data + (v227_data * v243_data));
          float v248_data = s0[36];
          float v250_data = ir0[8];
          ir0[8] = (v250_data + (v227_data * v248_data));
          float v253_data = s0[44];
          float v255_data = ir0[10];
          ir0[10] = (v255_data + (v227_data * v253_data));
          float v259_data = v31_g ? (glb_m1[(v28_lead + 79)]) : (0.0f);
          float v262_data = ir0[1];
          ir0[1] = (v262_data + (v259_data * v228_data));
          float v267_data = ir0[3];
          ir0[3] = (v267_data + (v259_data * v233_data));
          float v272_data = ir0[5];
          ir0[5] = (v272_data + (v259_data * v238_data));
          float v277_data = ir0[7];
          ir0[7] = (v277_data + (v259_data * v243_data));
          float v282_data = ir0[9];
          ir0[9] = (v282_data + (v259_data * v248_data));
          float v287_data = ir0[11];
          ir0[11] = (v287_data + (v259_data * v253_data));
          float v291_data = v31_g ? (glb_m1[(v28_lead + 87)]) : (0.0f);
          float v292_data = s0[5];
          float v294_data = ir0[1];
          ir0[1] = (v294_data + (v291_data * v292_data));
          float v297_data = s0[13];
          float v299_data = ir0[3];
          ir0[3] = (v299_data + (v291_data * v297_data));
          float v302_data = s0[21];
          float v304_data = ir0[5];
          ir0[5] = (v304_data + (v291_data * v302_data));
          float v307_data = s0[29];
          float v309_data = ir0[7];
          ir0[7] = (v309_data + (v291_data * v307_data));
          float v312_data = s0[37];
          float v314_data = ir0[9];
          ir0[9] = (v314_data + (v291_data * v312_data));
          float v317_data = s0[45];
          float v319_data = ir0[11];
          ir0[11] = (v319_data + (v291_data * v317_data));
          float v323_data = glb_m1[(v26_lead + 127)];
          float v324_data = s0[6];
          float v326_data = ir0[0];
          ir0[0] = (v326_data + (v323_data * v324_data));
          float v329_data = s0[14];
          float v331_data = ir0[2];
          ir0[2] = (v331_data + (v323_data * v329_data));
          float v334_data = s0[22];
          float v336_data = ir0[4];
          ir0[4] = (v336_data + (v323_data * v334_data));
          float v339_data = s0[30];
          float v341_data = ir0[6];
          ir0[6] = (v341_data + (v323_data * v339_data));
          float v344_data = s0[38];
          float v346_data = ir0[8];
          ir0[8] = (v346_data + (v323_data * v344_data));
          float v349_data = s0[46];
          float v351_data = ir0[10];
          ir0[10] = (v351_data + (v323_data * v349_data));
          float v355_data = v31_g ? (glb_m1[(v28_lead + 127)]) : (0.0f);
          float v358_data = ir0[1];
          ir0[1] = (v358_data + (v355_data * v324_data));
          float v363_data = ir0[3];
          ir0[3] = (v363_data + (v355_data * v329_data));
          float v368_data = ir0[5];
          ir0[5] = (v368_data + (v355_data * v334_data));
          float v373_data = ir0[7];
          ir0[7] = (v373_data + (v355_data * v339_data));
          float v378_data = ir0[9];
          ir0[9] = (v378_data + (v355_data * v344_data));
          float v383_data = ir0[11];
          ir0[11] = (v383_data + (v355_data * v349_data));
          float v387_data = glb_m1[(v26_lead + 167)];
          float v388_data = s0[7];
          float v390_data = ir0[0];
          ir0[0] = (v390_data + (v387_data * v388_data));
          float v393_data = s0[15];
          float v395_data = ir0[2];
          ir0[2] = (v395_data + (v387_data * v393_data));
          float v398_data = s0[23];
          float v400_data = ir0[4];
          ir0[4] = (v400_data + (v387_data * v398_data));
          float v403_data = s0[31];
          float v405_data = ir0[6];
          ir0[6] = (v405_data + (v387_data * v403_data));
          float v408_data = s0[39];
          float v410_data = ir0[8];
          ir0[8] = (v410_data + (v387_data * v408_data));
          float v413_data = s0[47];
          float v415_data = ir0[10];
          ir0[10] = (v415_data + (v387_data * v413_data));
          float v419_data = v31_g ? (glb_m1[(v28_lead + 167)]) : (0.0f);
          float v422_data = ir0[1];
          ir0[1] = (v422_data + (v419_data * v388_data));
          float v427_data = ir0[3];
          ir0[3] = (v427_data + (v419_data * v393_data));
          float v432_data = ir0[5];
          ir0[5] = (v432_data + (v419_data * v398_data));
          float v437_data = ir0[7];
          ir0[7] = (v437_data + (v419_data * v403_data));
          float v442_data = ir0[9];
          ir0[9] = (v442_data + (v419_data * v408_data));
          float v447_data = ir0[11];
          ir0[11] = (v447_data + (v419_data * v413_data));
          // r0 = ir0
          #pragma unroll
          for (int32_t v451_n0 = 0; v451_n0 < 1; ++v451_n0) {
            #pragma unroll
            for (int32_t v452_n1 = 0; v452_n1 < 6; ++v452_n1) {
              int32_t v454_a = v451_n0 + (v452_n1 * 2);
              float v455_data = ir0[v454_a];
              r0[v454_a] = v455_data;
            }
          }
          if (v31_g) {
            #pragma unroll
            for (int32_t v457_n1 = 0; v457_n1 < 6; ++v457_n1) {
              int32_t v459_a = 1 + (v457_n1 * 2);
              float v460_data = ir0[v459_a];
              r0[v459_a] = v460_data;
            }
          }
          // glb_m0 = store{r>g}(r0);
          #pragma unroll
          for (int32_t v463_i0 = 0; v463_i0 < 1; ++v463_i0) {
            int32_t v469_lead = v26_lead + (v463_i0 * 32);
            #pragma unroll
            for (int32_t v464_i1 = 0; v464_i1 < 6; ++v464_i1) {
              float v467_data = r0[(v463_i0 + (v464_i1 * 2))];
              glb_m0[(v469_lead + (v464_i1 * 40))] = v467_data;
            }
          }
          if (v31_g) {
            #pragma unroll
            for (int32_t v473_i1 = 0; v473_i1 < 6; ++v473_i1) {
              float v476_data = r0[(1 + (v473_i1 * 2))];
              glb_m0[(v28_lead + (v473_i1 * 40))] = v476_data;
            }
          }
          __syncwarp();
        }
        else {
          __pipeline_wait_prior(0);
        }
        // s0 = load{g>s}(wrap_glb_m2[0, 1])
        __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
        if (v490_g) {
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
        }
        __pipeline_commit();
        v486_flagWord = v487_flagWordNext;
        v487_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
  }
}

