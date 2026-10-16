// === base name ===
kernel_521f9540157136ac

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_521f9540157136ac = {{32, 4, 1}, 32, 32, 1, 4, 4608, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_521f9540157136ac(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_521f9540157136ac(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_521f9540157136ac(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_521f9540157136ac, block.x * block.y * block.z, 1152 * sizeof(float));
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
  config.sharedMemBytes = 1152 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_521f9540157136ac(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_521f9540157136ac(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_521f9540157136ac, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  static const tensorforge::ValueArray<int32_t, 96> glb_m1_packArg{{0, -1227133514, 21, 460175040, 37, 28760940, 53, 920350080, 69, 57521880, 85, 1840700160, 101, 115043760, 117, -613566976, 133, 230087520, 149, 47934902, 166, 460175040, 182, 95869804, 199, 920350080, 215, 191739608, 232, 1840700160, 248, 383479216, 265, 57521664, 277, 766958432, 294, 115043766, 312, 1533916864, 329, 230087532, 347, -1227133568, 364, 460175064, 382, 28760832, 394, 920350128, 412, 57521664, 424, 1840700256, 442, 115043766, 460, -613566784, 478, 230087532, 496, 47934848, 509, 460175064, 527, 95869696, 540, 920350128, 558, 191739392, 571, 1840700256, 589, 383479222, 608, 57521856, 622, 766958444, 641, 115043712, 655, 1533916888, 674, 230087424, 688, -1227133520, 707, 460174848, 721, 28760928, 735, 920350134, 755, 57521856, 769, 1840700268}};
  kernel_kernel_521f9540157136ac<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m2, m2_extraOffset, glb_m1_packArg, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_521f9540157136ac(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, __grid_constant__ const tensorforge::ValueArray<int32_t, 96> glb_m1_pack, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: k_roll=4,pack_operands=bits
    // launch: 32 lanes x 4 per block = block 32x4x1, 4608 B shared, occupancy grid
    // operands:
    //   m0 32×6(32×6) {0..32}×{0..6} strided
    //   m1 32×48(32×48) {0..32}×{0..48} none ordered bits-packed
    //   m2 48×6(48×6) {0..48}×{0..6} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":1152}],"shared_bytes":4608,"shared_elements":1152,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[32,6]],"name":"m0","ordered":false,"parts":1,"shape":[32,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[32,48]],"name":"m1","ordered":true,"parts":1,"shape":[32,48],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[48,6]],"name":"m2","ordered":false,"parts":1,"shape":[48,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[32,48]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,48]},{"addressing":"strided","bbox":[[0,0],[48,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[48,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[288 * threadIdx.y + 0];
      size_t v3_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < numElements0) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      const float *const __restrict__ glb_m1 = &m1[0];
      float * __restrict__ s0 = &localShrMem0[0];
      const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
      size_t v140_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v140_flagSecondAhead < numElements0) ? v140_flagSecondAhead : v3_batchId1)];
      const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 288 + 0 + m2_extraOffset];
      // s0 = load{g>s}(peel_glb_m2[0, 1])
      #pragma unroll
      for (int32_t v150_i = 0; v150_i < 9; ++v150_i) {
        int32_t v152_a = threadIdx.x + (v150_i * 32);
        __pipeline_memcpy_async(&s0[v152_a], &peel_glb_m2[v152_a], 4);
      }
      __pipeline_commit();
      uint32_t v144_flagWord = flagWordFirst;
      uint32_t v145_flagWordNext = flagWordSecond;
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < numElements0; v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
        const float *const __restrict__ wrap_glb_m2 = &m2[v12_batchId1 * 288 + 0 + m2_extraOffset];
        size_t v162_flagAheadAhead = v12_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v162_flagAheadAhead < numElements0) ? v162_flagAheadAhead : v12_batchId1)];
        const bool allowed = static_cast<bool>(v144_flagWord);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 192 + 0 + m0_extraOffset];
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r0[6]{};
          // ir0 = +(glb_m1 * s0)
          // [(0, 32), (0, 6)] [(0, 48)]
          float ir0[6]{};
          int32_t v28_lead = threadIdx.x % 32;
          int32_t v34_bit = (1 << v28_lead) - 1;
          __syncwarp();
          #pragma unroll 4
          for (int32_t v26_k0 = 0; v26_k0 < 48; ++v26_k0) {
            int32_t v29_c = v26_k0 * 2;
            int32_t v32_word = glb_m1_pack[(v29_c + 1)];
            bool v39_g = ((v32_word >> v28_lead) & 1) != 0;
            int32_t v40_slot = (__popc((v32_word & v34_bit))) + (glb_m1_pack[v29_c]);
            float v42_data = v39_g ? (glb_m1[v40_slot]) : (0.0f);
            float v44_data = s0[v26_k0];
            float v46_data = ir0[0];
            ir0[0] = (v46_data + (v42_data * v44_data));
            float v57_data = v39_g ? (glb_m1[v40_slot]) : (0.0f);
            float v59_data = s0[(v26_k0 + 48)];
            float v61_data = ir0[1];
            ir0[1] = (v61_data + (v57_data * v59_data));
            float v72_data = v39_g ? (glb_m1[v40_slot]) : (0.0f);
            float v74_data = s0[(v26_k0 + 96)];
            float v76_data = ir0[2];
            ir0[2] = (v76_data + (v72_data * v74_data));
            float v87_data = v39_g ? (glb_m1[v40_slot]) : (0.0f);
            float v89_data = s0[(v26_k0 + 144)];
            float v91_data = ir0[3];
            ir0[3] = (v91_data + (v87_data * v89_data));
            float v102_data = v39_g ? (glb_m1[v40_slot]) : (0.0f);
            float v104_data = s0[(v26_k0 + 192)];
            float v106_data = ir0[4];
            ir0[4] = (v106_data + (v102_data * v104_data));
            float v117_data = v39_g ? (glb_m1[v40_slot]) : (0.0f);
            float v119_data = s0[(v26_k0 + 240)];
            float v121_data = ir0[5];
            ir0[5] = (v121_data + (v117_data * v119_data));
          }
          // r0 = ir0
          #pragma unroll
          for (int32_t v125_n0 = 0; v125_n0 < 1; ++v125_n0) {
            #pragma unroll
            for (int32_t v126_n1 = 0; v126_n1 < 6; ++v126_n1) {
              int32_t v127_a = v125_n0 + v126_n1;
              float v128_data = ir0[v127_a];
              r0[v127_a] = v128_data;
            }
          }
          // glb_m0 = store{r>g}(r0);
          #pragma unroll
          for (int32_t v131_i0 = 0; v131_i0 < 1; ++v131_i0) {
            int32_t v136_lead = v28_lead + (v131_i0 * 32);
            #pragma unroll
            for (int32_t v132_i1 = 0; v132_i1 < 6; ++v132_i1) {
              float v134_data = r0[(v131_i0 + v132_i1)];
              glb_m0[(v136_lead + (v132_i1 * 32))] = v134_data;
            }
          }
          __syncwarp();
        }
        else {
          __pipeline_wait_prior(0);
        }
        // s0 = load{g>s}(wrap_glb_m2[0, 1])
        #pragma unroll
        for (int32_t v154_i = 0; v154_i < 9; ++v154_i) {
          int32_t v156_a = threadIdx.x + (v154_i * 32);
          __pipeline_memcpy_async(&s0[v156_a], &wrap_glb_m2[v156_a], 4);
        }
        __pipeline_commit();
        v144_flagWord = v145_flagWordNext;
        v145_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
  }
}

