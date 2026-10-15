// === base name ===
kernel_2de7f5ff168171cf

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_2de7f5ff168171cf = {{32, 4, 1}, 32, 56, 1, 4, 1536, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_2de7f5ff168171cf(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_2de7f5ff168171cf(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_2de7f5ff168171cf(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_2de7f5ff168171cf, block.x * block.y * block.z, 384 * sizeof(float));
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
  config.sharedMemBytes = 384 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_2de7f5ff168171cf(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_2de7f5ff168171cf(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_2de7f5ff168171cf, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_2de7f5ff168171cf<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_2de7f5ff168171cf(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: k_roll=2,trip_reads=1
    // launch: 32 lanes (56 active) x 4 per block = block 32x4x1, 1536 B shared, occupancy grid
    // operands:
    //   m0 56×6(56×6) {0..56}×{0..6} strided
    //   m1 56×16(56×16) {0..56}×{0..16} none ordered
    //   m2 16×6(16×6) {0..16}×{0..6} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":56,"block":[32,4,1],"cooperative":false,"lead_width":1,"mults_per_block":4,"persistent":true,"sections":[{"barrier":false,"mults_per_block":4,"shared_elements":384}],"shared_bytes":1536,"shared_elements":384,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[56,6]],"name":"m0","ordered":false,"parts":1,"shape":[56,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[56,16]],"name":"m1","ordered":true,"parts":1,"shape":[56,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,6]],"name":"m2","ordered":false,"parts":1,"shape":[16,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[56,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[56,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[56,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[56,16]},{"addressing":"strided","bbox":[[0,0],[16,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[96 * threadIdx.y + 0];
      size_t v3_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < numElements0) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      const float *const __restrict__ glb_m1 = &m1[0];
      float * __restrict__ s0 = &localShrMem0[0];
      const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
      size_t v160_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v160_flagSecondAhead < numElements0) ? v160_flagSecondAhead : v3_batchId1)];
      const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 96 + 0 + m2_extraOffset];
      // s0 = load{g>s}(peel_glb_m2[0, 1])
      __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
      __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
      __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 64], &peel_glb_m2[0 + 0 + 1 * threadIdx.x + 64], 4);
      __pipeline_commit();
      uint32_t v164_flagWord = flagWordFirst;
      uint32_t v165_flagWordNext = flagWordSecond;
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < numElements0; v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
        const float *const __restrict__ wrap_glb_m2 = &m2[v12_batchId1 * 96 + 0 + m2_extraOffset];
        size_t v182_flagAheadAhead = v12_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v182_flagAheadAhead < numElements0) ? v182_flagAheadAhead : v12_batchId1)];
        const bool allowed = static_cast<bool>(v164_flagWord);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 336 + 0 + m0_extraOffset];
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r0[12]{};
          // ir0 = +(glb_m1 * s0)
          // [(0, 56), (0, 6)] [(0, 16)]
          float ir0[12]{};
          int32_t v27_lead = threadIdx.x % 32;
          int32_t v29_a = v27_lead * 2;
          float v143_trip[2]{};
          int32_t v152_a = (0 * 64) + v29_a;
          tensorforge::VectorT<float, 2> v157_tripRead = *(tensorforge::VectorT<float, 2>*)&glb_m1[v152_a];
          *(tensorforge::VectorT<float, 2>*)&v143_trip[0_i32] = v157_tripRead;
          __syncwarp();
          #pragma unroll 2
          for (int32_t v24_k0 = 0; v24_k0 < 16; ++v24_k0) {
            int32_t v148_tripAhead = v24_k0 + 1;
            int32_t v155_a = (((v148_tripAhead < 16) ? v148_tripAhead : v24_k0) * 64) + v29_a;
            tensorforge::VectorT<float, 2> v31_data = *(tensorforge::VectorT<float, 2>*)&v143_trip[0_i32];
            tensorforge::VectorT<float, 2> v158_tripRead = *(tensorforge::VectorT<float, 2>*)&glb_m1[v155_a];
            *(tensorforge::VectorT<float, 2>*)&v143_trip[0_i32] = v158_tripRead;
            float v32_p = v31_data[0];
            float v34_data = s0[v24_k0];
            float v36_data = ir0[0];
            ir0[0] = (v36_data + (v32_p * v34_data));
            float v41_data = s0[(v24_k0 + 16)];
            float v43_data = ir0[2];
            ir0[2] = (v43_data + (v32_p * v41_data));
            float v48_data = s0[(v24_k0 + 32)];
            float v50_data = ir0[4];
            ir0[4] = (v50_data + (v32_p * v48_data));
            float v55_data = s0[(v24_k0 + 48)];
            float v57_data = ir0[6];
            ir0[6] = (v57_data + (v32_p * v55_data));
            float v62_data = s0[(v24_k0 + 64)];
            float v64_data = ir0[8];
            ir0[8] = (v64_data + (v32_p * v62_data));
            float v69_data = s0[(v24_k0 + 80)];
            float v71_data = ir0[10];
            ir0[10] = (v71_data + (v32_p * v69_data));
            float v74_p = v31_data[1];
            float v77_data = ir0[1];
            ir0[1] = (v77_data + (v74_p * v34_data));
            float v83_data = ir0[3];
            ir0[3] = (v83_data + (v74_p * v41_data));
            float v89_data = ir0[5];
            ir0[5] = (v89_data + (v74_p * v48_data));
            float v95_data = ir0[7];
            ir0[7] = (v95_data + (v74_p * v55_data));
            float v101_data = ir0[9];
            ir0[9] = (v101_data + (v74_p * v62_data));
            float v107_data = ir0[11];
            ir0[11] = (v107_data + (v74_p * v69_data));
          }
          // r0 = ir0
          #pragma unroll
          for (int32_t v112_n0 = 0; v112_n0 < 1; ++v112_n0) {
            #pragma unroll
            for (int32_t v113_n1 = 0; v113_n1 < 6; ++v113_n1) {
              int32_t v115_a = v112_n0 + (v113_n1 * 2);
              float v116_data = ir0[v115_a];
              r0[v115_a] = v116_data;
            }
          }
          bool v117_g = v27_lead < 24;
          if (v117_g) {
            #pragma unroll
            for (int32_t v118_n1 = 0; v118_n1 < 6; ++v118_n1) {
              int32_t v120_a = 1 + (v118_n1 * 2);
              float v121_data = ir0[v120_a];
              r0[v120_a] = v121_data;
            }
          }
          // glb_m0 = store{r>g}(r0);
          #pragma unroll
          for (int32_t v125_i0 = 0; v125_i0 < 1; ++v125_i0) {
            int32_t v131_lead = v27_lead + (v125_i0 * 32);
            #pragma unroll
            for (int32_t v126_i1 = 0; v126_i1 < 6; ++v126_i1) {
              float v129_data = r0[(v125_i0 + (v126_i1 * 2))];
              glb_m0[(v131_lead + (v126_i1 * 56))] = v129_data;
            }
          }
          if (v117_g) {
            int32_t v140_lead = v27_lead + 32_i32;
            #pragma unroll
            for (int32_t v135_i1 = 0; v135_i1 < 6; ++v135_i1) {
              float v138_data = r0[(1 + (v135_i1 * 2))];
              glb_m0[(v140_lead + (v135_i1 * 56))] = v138_data;
            }
          }
          __syncwarp();
        }
        else {
          __pipeline_wait_prior(0);
        }
        // s0 = load{g>s}(wrap_glb_m2[0, 1])
        __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
        __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
        __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 64], &wrap_glb_m2[0 + 0 + 1 * threadIdx.x + 64], 4);
        __pipeline_commit();
        v164_flagWord = v165_flagWordNext;
        v165_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
  }
}

