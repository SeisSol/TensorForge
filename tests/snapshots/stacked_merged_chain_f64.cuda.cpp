// === base name ===
kernel_b6242dc25ee06880

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b6242dc25ee06880 = {{16, 8, 1}, 16, 20, 1, 8, 13312, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b6242dc25ee06880(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b6242dc25ee06880(const double ** m0, size_t m0_extraOffset, const double ** m1, size_t m1_extraOffset, double ** m2, size_t m2_extraOffset, const double * m3, const double ** m4, size_t m4_extraOffset, const double * m5, const double ** m6, size_t m6_extraOffset, const double * m7, const double ** m8, size_t m8_extraOffset, const double * m9, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b6242dc25ee06880(size_t numElements0, void* streamPtr) {
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
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_b6242dc25ee06880, block.x * block.y * block.z, 1664 * sizeof(double));
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
  config.sharedMemBytes = 1664 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_b6242dc25ee06880(const double ** m0, size_t m0_extraOffset, const double ** m1, size_t m1_extraOffset, double ** m2, size_t m2_extraOffset, const double * m3, const double ** m4, size_t m4_extraOffset, const double * m5, const double ** m6, size_t m6_extraOffset, const double * m7, const double ** m8, size_t m8_extraOffset, const double * m9, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b6242dc25ee06880(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_b6242dc25ee06880, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  const double *const * m0Arg = (const double *const *)m0;
  const double *const * m1Arg = (const double *const *)m1;
  double *const * m2Arg = (double *const *)m2;
  const double *const * m4Arg = (const double *const *)m4;
  const double *const * m6Arg = (const double *const *)m6;
  const double *const * m8Arg = (const double *const *)m8;
  const unsigned * flags0Arg = (const unsigned *)flags0;
  kernel_kernel_b6242dc25ee06880<<<grid,block,config.sharedMemBytes,stream>>>(m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3, m4Arg, m4_extraOffset, m5, m6Arg, m6_extraOffset, m7, m8Arg, m8_extraOffset, m9, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128, 1)
 kernel_kernel_b6242dc25ee06880(const double *const * __restrict__ m0, size_t m0_extraOffset, const double *const * __restrict__ m1, size_t m1_extraOffset, double *const * __restrict__ m2, size_t m2_extraOffset, const double * m3, const double *const * __restrict__ m4, size_t m4_extraOffset, const double * m5, const double *const * __restrict__ m6, size_t m6_extraOffset, const double * m7, const double *const * __restrict__ m8, size_t m8_extraOffset, const double * m9, size_t numElements0, const unsigned * __restrict__ flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: amd_matrix_k4=1,lanes_per_mult=16,merge_variants=1,mma_d_layout=1,mma_d_layout_loops=1
    // launch: 16 lanes (20 active) x 8 per block = block 16x8x1, 13312 B shared, occupancy grid
    // operands:
    //   v0 9×9(9×9) {0..9}×{0..9} pointer_based
    //   v1 20×10(20×10) {0..20}×{0..10} none
    //   m0 20×9(20×9) {0..20}×{0..9} pointer_based
    //   m1 9×9(9×9) {0..9}×{0..9} pointer_based
    //   m2 20×9(20×9) {0..20}×{0..9} pointer_based
    //   m3 20×10(20×10) {0..20}×{0..10} none ordered
    //   m4 9×9(9×9) {0..9}×{0..9} pointer_based
    //   m5 20×10(20×10) {0..20}×{0..10} none ordered
    //   m6 9×9(9×9) {0..9}×{0..9} pointer_based
    //   m7 20×10(20×10) {0..20}×{0..10} none ordered
    //   m8 9×9(9×9) {0..9}×{0..9} pointer_based
    //   m9 20×10(20×10) {0..20}×{0..10} none ordered
    // operations:
    //   for 4 iterations (any order):
    //     t0[i,j] = m0[i,k]@{0..10}×{0..9} × v0[k,j]
    //     m2[i,j] += v1[i,k] × t0[k,j]
    //     v0 ∈ {m1, m4, m6, m8}
    //     v1 ∈ {m3, m5, m7, m9}
    // tensorforge-meta: {"fp":"double","launch":{"active_threads":20,"block":[16,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":1664}],"shared_bytes":13312,"shared_elements":1664,"threads_per_mult":16},"loops":[{"body":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"v0","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,10]],"is_tmp":false,"name":"v1","offset":[0,0],"shape":[20,10]},{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"holes":[{"members":["m1","m4","m6","m8"],"stand_in":"v0"},{"members":["m3","m5","m7","m9"],"stand_in":"v1"}],"iterations":4,"kind":"for","sequential":false}],"operands":[{"addressing":"pointer_based","alias":"variant0","bbox":[[0,0],[9,9]],"name":"v0","ordered":false,"parts":1,"shape":[9,9],"variant":true},{"addressing":"none","alias":"variant1","bbox":[[0,0],[20,10]],"name":"v1","ordered":false,"parts":1,"shape":[20,10],"variant":true},{"addressing":"pointer_based","alias":"I","bbox":[[0,0],[20,9]],"name":"m0","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"pointer_based","alias":"S0","bbox":[[0,0],[9,9]],"name":"m1","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"pointer_based","alias":"Q","bbox":[[0,0],[20,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"none","alias":"K0","bbox":[[0,0],[20,10]],"name":"m3","ordered":true,"parts":1,"shape":[20,10],"variant":false},{"addressing":"pointer_based","alias":"S1","bbox":[[0,0],[9,9]],"name":"m4","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"none","alias":"K1","bbox":[[0,0],[20,10]],"name":"m5","ordered":true,"parts":1,"shape":[20,10],"variant":false},{"addressing":"pointer_based","alias":"S2","bbox":[[0,0],[9,9]],"name":"m6","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"none","alias":"K2","bbox":[[0,0],[20,10]],"name":"m7","ordered":true,"parts":1,"shape":[20,10],"variant":false},{"addressing":"pointer_based","alias":"S3","bbox":[[0,0],[9,9]],"name":"m8","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"none","alias":"K3","bbox":[[0,0],[20,10]],"name":"m9","ordered":true,"parts":1,"shape":[20,10],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,10]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[20,10]},{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,10]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[20,10]},{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,10]],"is_tmp":false,"name":"m7","offset":[0,0],"shape":[20,10]},{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m8","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,10]],"is_tmp":false,"name":"m9","offset":[0,0],"shape":[20,10]},{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[208 * threadIdx.y + 0];
      size_t v3_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < numElements0) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      const double *const __restrict__ glb_m3 = &m3[0];
      const double *const __restrict__ glb_m5 = &m5[0];
      const double *const __restrict__ glb_m7 = &m7[0];
      const double *const __restrict__ glb_m9 = &m9[0];
      double r0[9]{};
      const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
      size_t v1646_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v1646_flagSecondAhead < numElements0) ? v1646_flagSecondAhead : v3_batchId1)];
      const bool allowed_peel = static_cast<bool>(flagWordFirst);
      const double *const __restrict__ peel_glb_m0 = &m0[v3_batchId1][0 + m0_extraOffset];
      int32_t v1656_lead = threadIdx.x % 16;
      bool v1657_g = v1656_lead < 10;
      // r0 = load{g>r}(peel_glb_m0);
      if (v1657_g) {
        #pragma unroll
        for (int32_t v1666_i1 = 0; v1666_i1 < 9; ++v1666_i1) {
          double v1669_data = allowed_peel ? (__ldcg(&peel_glb_m0[(v1656_lead + (v1666_i1 * 20))])) : (0.0);
          r0[v1666_i1] = v1669_data;
        }
      }
      uint32_t v1650_flagWord = flagWordFirst;
      uint32_t v1651_flagWordNext = flagWordSecond;
      for (size_t v11_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v11_batchId0 < numElements0; v11_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v12_ahead1 = v11_batchId0 + (gridDim.x * blockDim.y);
        size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
        const double *const __restrict__ wrap_glb_m0 = &m0[v14_batchId1][0 + m0_extraOffset];
        size_t v1676_flagAheadAhead = v14_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v1676_flagAheadAhead < numElements0) ? v1676_flagAheadAhead : v14_batchId1)];
        const bool allowed = static_cast<bool>(v1650_flagWord);
        const bool allowed_next = static_cast<bool>(v1651_flagWordNext);
        if (allowed) {
          const double *const __restrict__ glb_m1 = &m1[v11_batchId0][0 + m1_extraOffset];
          double *const __restrict__ glb_m2 = &m2[v11_batchId0][0 + m2_extraOffset];
          const double *const __restrict__ glb_m4 = &m4[v11_batchId0][0 + m4_extraOffset];
          const double *const __restrict__ glb_m6 = &m6[v11_batchId0][0 + m6_extraOffset];
          const double *const __restrict__ glb_m8 = &m8[v11_batchId0][0 + m8_extraOffset];
          double * __restrict__ s0 = &localShrMem0[0];
          double * __restrict__ s1 = &localShrMem0[96];
          bool v48_g = threadIdx.x < 1;
          int32_t v1569_lead = v1656_lead + 16_i32;
          bool v1572_g = v1656_lead < 4;
          int32_t v466_a = v1656_lead * 2;
          int32_t v576_a = 32 + v466_a;
          int32_t v685_a = 64 + v466_a;
          int32_t v794_a = 96 + v466_a;
          int32_t v903_a = 128 + v466_a;
          int32_t v1012_a = 160 + v466_a;
          int32_t v1121_a = 192 + v466_a;
          int32_t v1230_a = 224 + v466_a;
          int32_t v1339_a = 256 + v466_a;
          int32_t v1448_a = 288 + v466_a;
          double r2[18]{};
          // r2 = load{g>r}(glb_m2);
          #pragma unroll
          for (int32_t v1558_i0 = 0; v1558_i0 < 1; ++v1558_i0) {
            int32_t v1561_lead = v1656_lead + (v1558_i0 * 16);
            #pragma unroll
            for (int32_t v1559_i1 = 0; v1559_i1 < 9; ++v1559_i1) {
              double v1564_data = glb_m2[(v1561_lead + (v1559_i1 * 20))];
              r2[(v1558_i0 + (v1559_i1 * 2))] = v1564_data;
            }
          }
          #pragma unroll
          for (int32_t v1567_i1 = 0; v1567_i1 < 9; ++v1567_i1) {
            double v1573_data = v1572_g ? (glb_m2[(v1569_lead + (v1567_i1 * 20))]) : (0.0);
            r2[(1 + (v1567_i1 * 2))] = v1573_data;
          }
          const double *const peel_v0Table = (0 == 0) ? glb_m1 : (0 == 1) ? glb_m4 : (0 == 2) ? glb_m6 : glb_m8;
          const double *const __restrict__ peel_glb_v0 = peel_v0Table;
          // s0 = load{g>s}(peel_glb_v0[0, 1])
          #pragma unroll
          for (int32_t v1631_i = 0; v1631_i < 5; ++v1631_i) {
            int32_t v1633_a = threadIdx.x + (v1631_i * 16);
            __pipeline_memcpy_async(&s0[v1633_a], &peel_glb_v0[v1633_a], 8);
          }
          if (v48_g) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 80], &peel_glb_v0[0 + 0 + 1 * threadIdx.x + 80], 8);
          }
          __pipeline_commit();
          for (int32_t batchIdv6 = 0; batchIdv6 < 4; ++batchIdv6) {
            int32_t v1624_tripAhead = batchIdv6 + 1;
            int32_t v1626_trip = (v1624_tripAhead < 4) ? v1624_tripAhead : batchIdv6;
            const double *const wrap_v0Table = (v1626_trip == 0) ? glb_m1 : (v1626_trip == 1) ? glb_m4 : (v1626_trip == 2) ? glb_m6 : glb_m8;
            const double *const __restrict__ wrap_glb_v0 = wrap_v0Table;
            const double *const v1Table = (batchIdv6 == 0) ? glb_m3 : (batchIdv6 == 1) ? glb_m5 : (batchIdv6 == 2) ? glb_m7 : glb_m9;
            const double *const __restrict__ glb_v1 = v1Table;
            // wait(s0 = load{g>s}(glb_v0[0, 1]));
            __pipeline_wait_prior(0);
            double r1[9]{};
            // r1 = +(r0 * s0) + None
            // [(0, 10), (0, 9)] [(0, 9)]
            double v51_data = r0[0];
            __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
            double v52_data = s0[0];
            double v57_data = s0[9];
            double v62_data = s0[18];
            double v67_data = s0[27];
            double v72_data = s0[36];
            double v77_data = s0[45];
            double v82_data = s0[54];
            double v87_data = s0[63];
            double v92_data = s0[72];
            double v96_data = r0[1];
            double v97_data = s0[1];
            double v102_data = s0[10];
            double v107_data = s0[19];
            double v112_data = s0[28];
            double v117_data = s0[37];
            double v122_data = s0[46];
            double v127_data = s0[55];
            double v132_data = s0[64];
            double v137_data = s0[73];
            double v141_data = r0[2];
            double v142_data = s0[2];
            double v147_data = s0[11];
            double v152_data = s0[20];
            double v157_data = s0[29];
            double v162_data = s0[38];
            double v167_data = s0[47];
            double v172_data = s0[56];
            double v177_data = s0[65];
            double v182_data = s0[74];
            double v186_data = r0[3];
            double v187_data = s0[3];
            double v192_data = s0[12];
            double v197_data = s0[21];
            double v202_data = s0[30];
            double v207_data = s0[39];
            double v212_data = s0[48];
            double v217_data = s0[57];
            double v222_data = s0[66];
            double v227_data = s0[75];
            double v231_data = r0[4];
            double v232_data = s0[4];
            double v237_data = s0[13];
            double v242_data = s0[22];
            double v247_data = s0[31];
            double v252_data = s0[40];
            double v257_data = s0[49];
            double v262_data = s0[58];
            double v267_data = s0[67];
            double v272_data = s0[76];
            double v276_data = r0[5];
            double v277_data = s0[5];
            double v282_data = s0[14];
            double v287_data = s0[23];
            double v292_data = s0[32];
            double v297_data = s0[41];
            double v302_data = s0[50];
            double v307_data = s0[59];
            double v312_data = s0[68];
            double v317_data = s0[77];
            double v321_data = r0[6];
            double v322_data = s0[6];
            double v327_data = s0[15];
            double v332_data = s0[24];
            double v337_data = s0[33];
            double v342_data = s0[42];
            double v347_data = s0[51];
            double v352_data = s0[60];
            double v357_data = s0[69];
            double v362_data = s0[78];
            double v366_data = r0[7];
            double v367_data = s0[7];
            double v372_data = s0[16];
            double v375_p = (((((((v51_data * v57_data) + (v96_data * v102_data)) + (v141_data * v147_data)) + (v186_data * v192_data)) + (v231_data * v237_data)) + (v276_data * v282_data)) + (v321_data * v327_data)) + (v366_data * v372_data);
            double v377_data = s0[25];
            double v380_p = (((((((v51_data * v62_data) + (v96_data * v107_data)) + (v141_data * v152_data)) + (v186_data * v197_data)) + (v231_data * v242_data)) + (v276_data * v287_data)) + (v321_data * v332_data)) + (v366_data * v377_data);
            double v382_data = s0[34];
            double v385_p = (((((((v51_data * v67_data) + (v96_data * v112_data)) + (v141_data * v157_data)) + (v186_data * v202_data)) + (v231_data * v247_data)) + (v276_data * v292_data)) + (v321_data * v337_data)) + (v366_data * v382_data);
            double v387_data = s0[43];
            double v390_p = (((((((v51_data * v72_data) + (v96_data * v117_data)) + (v141_data * v162_data)) + (v186_data * v207_data)) + (v231_data * v252_data)) + (v276_data * v297_data)) + (v321_data * v342_data)) + (v366_data * v387_data);
            double v392_data = s0[52];
            double v395_p = (((((((v51_data * v77_data) + (v96_data * v122_data)) + (v141_data * v167_data)) + (v186_data * v212_data)) + (v231_data * v257_data)) + (v276_data * v302_data)) + (v321_data * v347_data)) + (v366_data * v392_data);
            double v397_data = s0[61];
            double v400_p = (((((((v51_data * v82_data) + (v96_data * v127_data)) + (v141_data * v172_data)) + (v186_data * v217_data)) + (v231_data * v262_data)) + (v276_data * v307_data)) + (v321_data * v352_data)) + (v366_data * v397_data);
            double v402_data = s0[70];
            double v405_p = (((((((v51_data * v87_data) + (v96_data * v132_data)) + (v141_data * v177_data)) + (v186_data * v222_data)) + (v231_data * v267_data)) + (v276_data * v312_data)) + (v321_data * v357_data)) + (v366_data * v402_data);
            double v407_data = s0[79];
            double v410_p = (((((((v51_data * v92_data) + (v96_data * v137_data)) + (v141_data * v182_data)) + (v186_data * v227_data)) + (v231_data * v272_data)) + (v276_data * v317_data)) + (v321_data * v362_data)) + (v366_data * v407_data);
            double v411_data = r0[8];
            double v412_data = s0[8];
            r1[0] = (((((((((v51_data * v52_data) + (v96_data * v97_data)) + (v141_data * v142_data)) + (v186_data * v187_data)) + (v231_data * v232_data)) + (v276_data * v277_data)) + (v321_data * v322_data)) + (v366_data * v367_data)) + (v411_data * v412_data));
            double v417_data = s0[17];
            r1[1] = (v375_p + (v411_data * v417_data));
            double v422_data = s0[26];
            r1[2] = (v380_p + (v411_data * v422_data));
            double v427_data = s0[35];
            r1[3] = (v385_p + (v411_data * v427_data));
            double v432_data = s0[44];
            r1[4] = (v390_p + (v411_data * v432_data));
            double v437_data = s0[53];
            r1[5] = (v395_p + (v411_data * v437_data));
            double v442_data = s0[62];
            r1[6] = (v400_p + (v411_data * v442_data));
            double v447_data = s0[71];
            r1[7] = (v405_p + (v411_data * v447_data));
            double v452_data = s0[80];
            r1[8] = (v410_p + (v411_data * v452_data));
            // r1 = 0 past (0, 10)
            // s1 = store{r>s}(localShrMem0, r1);
            if (v1657_g) {
              #pragma unroll
              for (int32_t v457_i1 = 0; v457_i1 < 9; ++v457_i1) {
                double v459_data = r1[v457_i1];
                s1[(v1656_lead + (v457_i1 * 10))] = v459_data;
              }
            }
            // ir3 = +(glb_v1 * s1)
            // [(0, 20), (0, 9)] [(0, 10)]
            double ir3[18]{};
            tensorforge::VectorT<double, 2> v468_data = *(tensorforge::VectorT<double, 2>*)&glb_v1[v466_a];
            double v469_p = v468_data[0];
            __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
            double v470_data = s1[0];
            double v476_data = s1[10];
            double v482_data = s1[20];
            double v488_data = s1[30];
            double v494_data = s1[40];
            double v500_data = s1[50];
            double v506_data = s1[60];
            double v512_data = s1[70];
            double v518_data = s1[80];
            double v523_p = v468_data[1];
            tensorforge::VectorT<double, 2> v577_data = *(tensorforge::VectorT<double, 2>*)&glb_v1[v576_a];
            double v578_p = v577_data[0];
            double v579_data = s1[1];
            double v585_data = s1[11];
            double v591_data = s1[21];
            double v597_data = s1[31];
            double v603_data = s1[41];
            double v609_data = s1[51];
            double v615_data = s1[61];
            double v621_data = s1[71];
            double v627_data = s1[81];
            double v632_p = v577_data[1];
            tensorforge::VectorT<double, 2> v686_data = *(tensorforge::VectorT<double, 2>*)&glb_v1[v685_a];
            double v687_p = v686_data[0];
            double v688_data = s1[2];
            double v694_data = s1[12];
            double v700_data = s1[22];
            double v706_data = s1[32];
            double v712_data = s1[42];
            double v718_data = s1[52];
            double v724_data = s1[62];
            double v730_data = s1[72];
            double v736_data = s1[82];
            double v741_p = v686_data[1];
            tensorforge::VectorT<double, 2> v795_data = *(tensorforge::VectorT<double, 2>*)&glb_v1[v794_a];
            double v796_p = v795_data[0];
            double v797_data = s1[3];
            double v803_data = s1[13];
            double v809_data = s1[23];
            double v815_data = s1[33];
            double v821_data = s1[43];
            double v827_data = s1[53];
            double v833_data = s1[63];
            double v839_data = s1[73];
            double v845_data = s1[83];
            double v850_p = v795_data[1];
            tensorforge::VectorT<double, 2> v904_data = *(tensorforge::VectorT<double, 2>*)&glb_v1[v903_a];
            double v905_p = v904_data[0];
            double v906_data = s1[4];
            double v912_data = s1[14];
            double v918_data = s1[24];
            double v924_data = s1[34];
            double v930_data = s1[44];
            double v933_p = ((((v469_p * v494_data) + (v578_p * v603_data)) + (v687_p * v712_data)) + (v796_p * v821_data)) + (v905_p * v930_data);
            double v936_data = s1[54];
            double v939_p = ((((v469_p * v500_data) + (v578_p * v609_data)) + (v687_p * v718_data)) + (v796_p * v827_data)) + (v905_p * v936_data);
            double v942_data = s1[64];
            double v945_p = ((((v469_p * v506_data) + (v578_p * v615_data)) + (v687_p * v724_data)) + (v796_p * v833_data)) + (v905_p * v942_data);
            double v948_data = s1[74];
            double v951_p = ((((v469_p * v512_data) + (v578_p * v621_data)) + (v687_p * v730_data)) + (v796_p * v839_data)) + (v905_p * v948_data);
            double v954_data = s1[84];
            double v957_p = ((((v469_p * v518_data) + (v578_p * v627_data)) + (v687_p * v736_data)) + (v796_p * v845_data)) + (v905_p * v954_data);
            double v959_p = v904_data[1];
            double v963_p = ((((v523_p * v470_data) + (v632_p * v579_data)) + (v741_p * v688_data)) + (v850_p * v797_data)) + (v959_p * v906_data);
            double v969_p = ((((v523_p * v476_data) + (v632_p * v585_data)) + (v741_p * v694_data)) + (v850_p * v803_data)) + (v959_p * v912_data);
            double v975_p = ((((v523_p * v482_data) + (v632_p * v591_data)) + (v741_p * v700_data)) + (v850_p * v809_data)) + (v959_p * v918_data);
            double v981_p = ((((v523_p * v488_data) + (v632_p * v597_data)) + (v741_p * v706_data)) + (v850_p * v815_data)) + (v959_p * v924_data);
            double v987_p = ((((v523_p * v494_data) + (v632_p * v603_data)) + (v741_p * v712_data)) + (v850_p * v821_data)) + (v959_p * v930_data);
            double v993_p = ((((v523_p * v500_data) + (v632_p * v609_data)) + (v741_p * v718_data)) + (v850_p * v827_data)) + (v959_p * v936_data);
            double v999_p = ((((v523_p * v506_data) + (v632_p * v615_data)) + (v741_p * v724_data)) + (v850_p * v833_data)) + (v959_p * v942_data);
            double v1005_p = ((((v523_p * v512_data) + (v632_p * v621_data)) + (v741_p * v730_data)) + (v850_p * v839_data)) + (v959_p * v948_data);
            double v1011_p = ((((v523_p * v518_data) + (v632_p * v627_data)) + (v741_p * v736_data)) + (v850_p * v845_data)) + (v959_p * v954_data);
            tensorforge::VectorT<double, 2> v1013_data = *(tensorforge::VectorT<double, 2>*)&glb_v1[v1012_a];
            double v1014_p = v1013_data[0];
            double v1015_data = s1[5];
            double v1018_p = (((((v469_p * v470_data) + (v578_p * v579_data)) + (v687_p * v688_data)) + (v796_p * v797_data)) + (v905_p * v906_data)) + (v1014_p * v1015_data);
            double v1021_data = s1[15];
            double v1024_p = (((((v469_p * v476_data) + (v578_p * v585_data)) + (v687_p * v694_data)) + (v796_p * v803_data)) + (v905_p * v912_data)) + (v1014_p * v1021_data);
            double v1027_data = s1[25];
            double v1030_p = (((((v469_p * v482_data) + (v578_p * v591_data)) + (v687_p * v700_data)) + (v796_p * v809_data)) + (v905_p * v918_data)) + (v1014_p * v1027_data);
            double v1033_data = s1[35];
            double v1036_p = (((((v469_p * v488_data) + (v578_p * v597_data)) + (v687_p * v706_data)) + (v796_p * v815_data)) + (v905_p * v924_data)) + (v1014_p * v1033_data);
            double v1039_data = s1[45];
            double v1040_p = v1014_p * v1039_data;
            // s0 = load{g>s}(wrap_glb_v0[0, 1])
            #pragma unroll
            for (int32_t v1636_i = 0; v1636_i < 5; ++v1636_i) {
              int32_t v1638_a = threadIdx.x + (v1636_i * 16);
              __pipeline_memcpy_async(&s0[v1638_a], &wrap_glb_v0[v1638_a], 8);
            }
            if (v48_g) {
              __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 80], &wrap_glb_v0[0 + 0 + 1 * threadIdx.x + 80], 8);
            }
            __pipeline_commit();
            double v1045_data = s1[55];
            double v1051_data = s1[65];
            double v1057_data = s1[75];
            double v1063_data = s1[85];
            double v1068_p = v1013_data[1];
            tensorforge::VectorT<double, 2> v1122_data = *(tensorforge::VectorT<double, 2>*)&glb_v1[v1121_a];
            double v1123_p = v1122_data[0];
            double v1124_data = s1[6];
            double v1130_data = s1[16];
            double v1136_data = s1[26];
            double v1142_data = s1[36];
            double v1148_data = s1[46];
            double v1154_data = s1[56];
            double v1160_data = s1[66];
            double v1166_data = s1[76];
            double v1172_data = s1[86];
            double v1177_p = v1122_data[1];
            tensorforge::VectorT<double, 2> v1231_data = *(tensorforge::VectorT<double, 2>*)&glb_v1[v1230_a];
            double v1232_p = v1231_data[0];
            double v1233_data = s1[7];
            double v1239_data = s1[17];
            double v1245_data = s1[27];
            double v1251_data = s1[37];
            double v1257_data = s1[47];
            double v1263_data = s1[57];
            double v1269_data = s1[67];
            double v1275_data = s1[77];
            double v1281_data = s1[87];
            double v1286_p = v1231_data[1];
            tensorforge::VectorT<double, 2> v1340_data = *(tensorforge::VectorT<double, 2>*)&glb_v1[v1339_a];
            double v1341_p = v1340_data[0];
            double v1342_data = s1[8];
            double v1348_data = s1[18];
            double v1351_p = ((v1024_p + (v1123_p * v1130_data)) + (v1232_p * v1239_data)) + (v1341_p * v1348_data);
            double v1354_data = s1[28];
            double v1357_p = ((v1030_p + (v1123_p * v1136_data)) + (v1232_p * v1245_data)) + (v1341_p * v1354_data);
            double v1360_data = s1[38];
            double v1363_p = ((v1036_p + (v1123_p * v1142_data)) + (v1232_p * v1251_data)) + (v1341_p * v1360_data);
            double v1366_data = s1[48];
            double v1369_p = (((v933_p + v1040_p) + (v1123_p * v1148_data)) + (v1232_p * v1257_data)) + (v1341_p * v1366_data);
            double v1372_data = s1[58];
            double v1375_p = (((v939_p + (v1014_p * v1045_data)) + (v1123_p * v1154_data)) + (v1232_p * v1263_data)) + (v1341_p * v1372_data);
            double v1378_data = s1[68];
            double v1381_p = (((v945_p + (v1014_p * v1051_data)) + (v1123_p * v1160_data)) + (v1232_p * v1269_data)) + (v1341_p * v1378_data);
            double v1384_data = s1[78];
            double v1387_p = (((v951_p + (v1014_p * v1057_data)) + (v1123_p * v1166_data)) + (v1232_p * v1275_data)) + (v1341_p * v1384_data);
            double v1390_data = s1[88];
            double v1393_p = (((v957_p + (v1014_p * v1063_data)) + (v1123_p * v1172_data)) + (v1232_p * v1281_data)) + (v1341_p * v1390_data);
            double v1395_p = v1340_data[1];
            double v1399_p = (((v963_p + (v1068_p * v1015_data)) + (v1177_p * v1124_data)) + (v1286_p * v1233_data)) + (v1395_p * v1342_data);
            double v1405_p = (((v969_p + (v1068_p * v1021_data)) + (v1177_p * v1130_data)) + (v1286_p * v1239_data)) + (v1395_p * v1348_data);
            double v1411_p = (((v975_p + (v1068_p * v1027_data)) + (v1177_p * v1136_data)) + (v1286_p * v1245_data)) + (v1395_p * v1354_data);
            double v1417_p = (((v981_p + (v1068_p * v1033_data)) + (v1177_p * v1142_data)) + (v1286_p * v1251_data)) + (v1395_p * v1360_data);
            double v1423_p = (((v987_p + (v1068_p * v1039_data)) + (v1177_p * v1148_data)) + (v1286_p * v1257_data)) + (v1395_p * v1366_data);
            double v1429_p = (((v993_p + (v1068_p * v1045_data)) + (v1177_p * v1154_data)) + (v1286_p * v1263_data)) + (v1395_p * v1372_data);
            double v1435_p = (((v999_p + (v1068_p * v1051_data)) + (v1177_p * v1160_data)) + (v1286_p * v1269_data)) + (v1395_p * v1378_data);
            double v1441_p = (((v1005_p + (v1068_p * v1057_data)) + (v1177_p * v1166_data)) + (v1286_p * v1275_data)) + (v1395_p * v1384_data);
            double v1447_p = (((v1011_p + (v1068_p * v1063_data)) + (v1177_p * v1172_data)) + (v1286_p * v1281_data)) + (v1395_p * v1390_data);
            tensorforge::VectorT<double, 2> v1449_data = *(tensorforge::VectorT<double, 2>*)&glb_v1[v1448_a];
            double v1450_p = v1449_data[0];
            double v1451_data = s1[9];
            ir3[0] = ((((v1018_p + (v1123_p * v1124_data)) + (v1232_p * v1233_data)) + (v1341_p * v1342_data)) + (v1450_p * v1451_data));
            double v1457_data = s1[19];
            ir3[2] = (v1351_p + (v1450_p * v1457_data));
            double v1463_data = s1[29];
            ir3[4] = (v1357_p + (v1450_p * v1463_data));
            double v1469_data = s1[39];
            ir3[6] = (v1363_p + (v1450_p * v1469_data));
            double v1475_data = s1[49];
            ir3[8] = (v1369_p + (v1450_p * v1475_data));
            double v1481_data = s1[59];
            ir3[10] = (v1375_p + (v1450_p * v1481_data));
            double v1487_data = s1[69];
            ir3[12] = (v1381_p + (v1450_p * v1487_data));
            double v1493_data = s1[79];
            ir3[14] = (v1387_p + (v1450_p * v1493_data));
            double v1499_data = s1[89];
            ir3[16] = (v1393_p + (v1450_p * v1499_data));
            double v1504_p = v1449_data[1];
            ir3[1] = (v1399_p + (v1504_p * v1451_data));
            ir3[3] = (v1405_p + (v1504_p * v1457_data));
            ir3[5] = (v1411_p + (v1504_p * v1463_data));
            ir3[7] = (v1417_p + (v1504_p * v1469_data));
            ir3[9] = (v1423_p + (v1504_p * v1475_data));
            ir3[11] = (v1429_p + (v1504_p * v1481_data));
            ir3[13] = (v1435_p + (v1504_p * v1487_data));
            ir3[15] = (v1441_p + (v1504_p * v1493_data));
            ir3[17] = (v1447_p + (v1504_p * v1499_data));
            // r3 = ir3 + r2
            #pragma unroll
            for (int32_t v1576_n0 = 0; v1576_n0 < 1; ++v1576_n0) {
              #pragma unroll
              for (int32_t v1577_n1 = 0; v1577_n1 < 9; ++v1577_n1) {
                int32_t v1579_a = v1576_n0 + (v1577_n1 * 2);
                double v1580_data = ir3[v1579_a];
                double v1581_data = r2[v1579_a];
                r2[v1579_a] = (v1581_data + v1580_data);
              }
            }
            if (v1572_g) {
              #pragma unroll
              for (int32_t v1584_n1 = 0; v1584_n1 < 9; ++v1584_n1) {
                int32_t v1586_a = 1 + (v1584_n1 * 2);
                double v1587_data = ir3[v1586_a];
                double v1588_data = r2[v1586_a];
                r2[v1586_a] = (v1588_data + v1587_data);
              }
            }
            // glb_m2 = store{r>g}(r3);
          }
          // r0 = load{g>r}(wrap_glb_m0);
          if (v1657_g) {
            #pragma unroll
            for (int32_t v1670_i1 = 0; v1670_i1 < 9; ++v1670_i1) {
              double v1673_data = allowed_next ? (__ldcg(&wrap_glb_m0[(v1656_lead + (v1670_i1 * 20))])) : (0.0);
              r0[v1670_i1] = v1673_data;
            }
          }
          __pipeline_wait_prior(0);
          #pragma unroll
          for (int32_t v1608_i0 = 0; v1608_i0 < 1; ++v1608_i0) {
            int32_t v1610_lead = v1656_lead + (v1608_i0 * 16);
            #pragma unroll
            for (int32_t v1611_i1 = 0; v1611_i1 < 9; ++v1611_i1) {
              double v1614_data = r2[(v1608_i0 + (v1611_i1 * 2))];
              glb_m2[(v1610_lead + (v1611_i1 * 20))] = v1614_data;
            }
          }
          if (v1572_g) {
            #pragma unroll
            for (int32_t v1618_i1 = 0; v1618_i1 < 9; ++v1618_i1) {
              double v1621_data = r2[(1 + (v1618_i1 * 2))];
              glb_m2[(v1569_lead + (v1618_i1 * 20))] = v1621_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
        else {
          // r0 = load{g>r}(wrap_glb_m0);
          if (v1657_g) {
            #pragma unroll
            for (int32_t v1662_i1 = 0; v1662_i1 < 9; ++v1662_i1) {
              double v1665_data = allowed_next ? (__ldcg(&wrap_glb_m0[(v1656_lead + (v1662_i1 * 20))])) : (0.0);
              r0[v1662_i1] = v1665_data;
            }
          }
        }
        v1650_flagWord = v1651_flagWordNext;
        v1651_flagWordNext = flagWordAhead;
      }
    }
  }
}

