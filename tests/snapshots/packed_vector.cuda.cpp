// === base name ===
kernel_befd5657520b75db

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_befd5657520b75db = {{16, 2, 1}, 16, 64, 4, 2, 640, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_befd5657520b75db(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_befd5657520b75db(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_befd5657520b75db(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 2, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_befd5657520b75db, block.x * block.y * block.z, 160 * sizeof(float));
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
  config.block[1] = 2;
  config.block[2] = 1;
  config.sharedMemBytes = 160 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_befd5657520b75db(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_befd5657520b75db(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        cudaFuncSetAttribute(kernel_kernel_befd5657520b75db, cudaFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes);
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_befd5657520b75db<<<grid,block,config.sharedMemBytes,stream>>>(m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(32, 1)
 kernel_kernel_befd5657520b75db(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: lead_vectorize=1,pack_operands=runs
    // launch: 16 lanes (64 active), lead width 4 x 2 per block = block 16x2x1, 640 B shared, occupancy grid
    // operands:
    //   m0 64×6(64×6) {0..64}×{0..6} strided
    //   m1 64×8(64×8) {0..64}×{0..8} none ordered runs-packed
    //   m2 8×6(8×6) {0..8}×{0..6} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":64,"block":[16,2,1],"cooperative":false,"lead_width":4,"mults_per_block":2,"persistent":true,"sections":[{"barrier":false,"mults_per_block":2,"shared_elements":160}],"shared_bytes":640,"shared_elements":160,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[64,6]],"name":"m0","ordered":false,"parts":1,"shape":[64,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[64,8]],"name":"m1","ordered":true,"parts":1,"shape":[64,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,6]],"name":"m2","ordered":false,"parts":1,"shape":[8,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[64,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[64,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[64,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[64,8]},{"addressing":"strided","bbox":[[0,0],[8,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[80 * threadIdx.y + 0];
      size_t v3_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < numElements0) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      const float *const __restrict__ glb_m1 = &m1[0];
      float * __restrict__ s0 = &localShrMem0[0];
      const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
      size_t v321_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v321_flagSecondAhead < numElements0) ? v321_flagSecondAhead : v3_batchId1)];
      const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 48 + 0 + m2_extraOffset];
      bool v329_g = threadIdx.x < 12;
      // s0 = load{g>s}(peel_glb_m2[0, 1])
      if (v329_g) {
        __pipeline_memcpy_async(&s0[0 + 0 + 4 * threadIdx.x + 0], &peel_glb_m2[0 + 0 + 4 * threadIdx.x + 0], 16);
      }
      __pipeline_commit();
      uint32_t v325_flagWord = flagWordFirst;
      uint32_t v326_flagWordNext = flagWordSecond;
      for (size_t v9_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v9_batchId0 < numElements0; v9_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v10_ahead1 = v9_batchId0 + (gridDim.x * blockDim.y);
        size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
        const float *const __restrict__ wrap_glb_m2 = &m2[v12_batchId1 * 48 + 0 + m2_extraOffset];
        size_t v339_flagAheadAhead = v12_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v339_flagAheadAhead < numElements0) ? v339_flagAheadAhead : v12_batchId1)];
        const bool allowed = static_cast<bool>(v325_flagWord);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 384 + 0 + m0_extraOffset];
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r0[24]{};
          // ir0 = +(glb_m1 * s0)
          // [(0, 64), (0, 6)] [(0, 8)]
          float ir0[24]{};
          int32_t v25_lead = threadIdx.x % 16;
          int32_t v28_vlead = v25_lead * 4;
          tensorforge::VectorT<float, 4> v31_data = (v25_lead < 2) ? (*(tensorforge::VectorT<float, 4>*)&glb_m1[v28_vlead]) : (tensorforge::VectorT<float, 4>{});
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
          float v32_data = s0[0];
          tensorforge::VectorT<float, 4> v33_splat = {v32_data, v32_data, v32_data, v32_data};
          tensorforge::VectorT<float, 4> v34_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[0];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[0] = (tensorforge::fma(v31_data, v33_splat, v34_data));
          float v37_data = s0[8];
          tensorforge::VectorT<float, 4> v38_splat = {v37_data, v37_data, v37_data, v37_data};
          tensorforge::VectorT<float, 4> v39_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[4];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[4] = (tensorforge::fma(v31_data, v38_splat, v39_data));
          float v42_data = s0[16];
          tensorforge::VectorT<float, 4> v43_splat = {v42_data, v42_data, v42_data, v42_data};
          tensorforge::VectorT<float, 4> v44_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[8];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[8] = (tensorforge::fma(v31_data, v43_splat, v44_data));
          float v47_data = s0[24];
          tensorforge::VectorT<float, 4> v48_splat = {v47_data, v47_data, v47_data, v47_data};
          tensorforge::VectorT<float, 4> v49_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[12];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[12] = (tensorforge::fma(v31_data, v48_splat, v49_data));
          float v52_data = s0[32];
          tensorforge::VectorT<float, 4> v53_splat = {v52_data, v52_data, v52_data, v52_data};
          tensorforge::VectorT<float, 4> v54_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[16];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[16] = (tensorforge::fma(v31_data, v53_splat, v54_data));
          float v57_data = s0[40];
          tensorforge::VectorT<float, 4> v58_splat = {v57_data, v57_data, v57_data, v57_data};
          tensorforge::VectorT<float, 4> v59_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[20];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[20] = (tensorforge::fma(v31_data, v58_splat, v59_data));
          tensorforge::VectorT<float, 4> v64_data = ((v25_lead >= 2) && (v25_lead < 7)) ? (*(tensorforge::VectorT<float, 4>*)&glb_m1[v28_vlead]) : (tensorforge::VectorT<float, 4>{});
          float v65_data = s0[1];
          tensorforge::VectorT<float, 4> v66_splat = {v65_data, v65_data, v65_data, v65_data};
          tensorforge::VectorT<float, 4> v67_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[0];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[0] = (tensorforge::fma(v64_data, v66_splat, v67_data));
          float v70_data = s0[9];
          tensorforge::VectorT<float, 4> v71_splat = {v70_data, v70_data, v70_data, v70_data};
          tensorforge::VectorT<float, 4> v72_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[4];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[4] = (tensorforge::fma(v64_data, v71_splat, v72_data));
          float v75_data = s0[17];
          tensorforge::VectorT<float, 4> v76_splat = {v75_data, v75_data, v75_data, v75_data};
          tensorforge::VectorT<float, 4> v77_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[8];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[8] = (tensorforge::fma(v64_data, v76_splat, v77_data));
          float v80_data = s0[25];
          tensorforge::VectorT<float, 4> v81_splat = {v80_data, v80_data, v80_data, v80_data};
          tensorforge::VectorT<float, 4> v82_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[12];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[12] = (tensorforge::fma(v64_data, v81_splat, v82_data));
          float v85_data = s0[33];
          tensorforge::VectorT<float, 4> v86_splat = {v85_data, v85_data, v85_data, v85_data};
          tensorforge::VectorT<float, 4> v87_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[16];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[16] = (tensorforge::fma(v64_data, v86_splat, v87_data));
          float v90_data = s0[41];
          tensorforge::VectorT<float, 4> v91_splat = {v90_data, v90_data, v90_data, v90_data};
          tensorforge::VectorT<float, 4> v92_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[20];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[20] = (tensorforge::fma(v64_data, v91_splat, v92_data));
          tensorforge::VectorT<float, 4> v97_data = ((v25_lead >= 7) && (v25_lead < 9)) ? (*(tensorforge::VectorT<float, 4>*)&glb_m1[v28_vlead]) : (tensorforge::VectorT<float, 4>{});
          float v98_data = s0[2];
          tensorforge::VectorT<float, 4> v99_splat = {v98_data, v98_data, v98_data, v98_data};
          tensorforge::VectorT<float, 4> v100_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[0];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[0] = (tensorforge::fma(v97_data, v99_splat, v100_data));
          float v103_data = s0[10];
          tensorforge::VectorT<float, 4> v104_splat = {v103_data, v103_data, v103_data, v103_data};
          tensorforge::VectorT<float, 4> v105_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[4];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[4] = (tensorforge::fma(v97_data, v104_splat, v105_data));
          float v108_data = s0[18];
          tensorforge::VectorT<float, 4> v109_splat = {v108_data, v108_data, v108_data, v108_data};
          tensorforge::VectorT<float, 4> v110_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[8];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[8] = (tensorforge::fma(v97_data, v109_splat, v110_data));
          float v113_data = s0[26];
          tensorforge::VectorT<float, 4> v114_splat = {v113_data, v113_data, v113_data, v113_data};
          tensorforge::VectorT<float, 4> v115_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[12];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[12] = (tensorforge::fma(v97_data, v114_splat, v115_data));
          float v118_data = s0[34];
          tensorforge::VectorT<float, 4> v119_splat = {v118_data, v118_data, v118_data, v118_data};
          tensorforge::VectorT<float, 4> v120_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[16];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[16] = (tensorforge::fma(v97_data, v119_splat, v120_data));
          float v123_data = s0[42];
          tensorforge::VectorT<float, 4> v124_splat = {v123_data, v123_data, v123_data, v123_data};
          tensorforge::VectorT<float, 4> v125_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[20];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[20] = (tensorforge::fma(v97_data, v124_splat, v125_data));
          tensorforge::VectorT<float, 4> v129_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[(v28_vlead + 36)];
          float v130_data = s0[3];
          tensorforge::VectorT<float, 4> v131_splat = {v130_data, v130_data, v130_data, v130_data};
          tensorforge::VectorT<float, 4> v132_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[0];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[0] = (tensorforge::fma(v129_data, v131_splat, v132_data));
          float v135_data = s0[11];
          tensorforge::VectorT<float, 4> v136_splat = {v135_data, v135_data, v135_data, v135_data};
          tensorforge::VectorT<float, 4> v137_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[4];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[4] = (tensorforge::fma(v129_data, v136_splat, v137_data));
          float v140_data = s0[19];
          tensorforge::VectorT<float, 4> v141_splat = {v140_data, v140_data, v140_data, v140_data};
          tensorforge::VectorT<float, 4> v142_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[8];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[8] = (tensorforge::fma(v129_data, v141_splat, v142_data));
          float v145_data = s0[27];
          tensorforge::VectorT<float, 4> v146_splat = {v145_data, v145_data, v145_data, v145_data};
          tensorforge::VectorT<float, 4> v147_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[12];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[12] = (tensorforge::fma(v129_data, v146_splat, v147_data));
          float v150_data = s0[35];
          tensorforge::VectorT<float, 4> v151_splat = {v150_data, v150_data, v150_data, v150_data};
          tensorforge::VectorT<float, 4> v152_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[16];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[16] = (tensorforge::fma(v129_data, v151_splat, v152_data));
          float v155_data = s0[43];
          tensorforge::VectorT<float, 4> v156_splat = {v155_data, v155_data, v155_data, v155_data};
          tensorforge::VectorT<float, 4> v157_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[20];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[20] = (tensorforge::fma(v129_data, v156_splat, v157_data));
          tensorforge::VectorT<float, 4> v164_data = ((v25_lead >= 11) && (v25_lead < 13)) ? (*(tensorforge::VectorT<float, 4>*)&glb_m1[(v28_vlead + 56)]) : (tensorforge::VectorT<float, 4>{});
          float v165_data = s0[4];
          tensorforge::VectorT<float, 4> v166_splat = {v165_data, v165_data, v165_data, v165_data};
          tensorforge::VectorT<float, 4> v167_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[0];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[0] = (tensorforge::fma(v164_data, v166_splat, v167_data));
          float v170_data = s0[12];
          tensorforge::VectorT<float, 4> v171_splat = {v170_data, v170_data, v170_data, v170_data};
          tensorforge::VectorT<float, 4> v172_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[4];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[4] = (tensorforge::fma(v164_data, v171_splat, v172_data));
          float v175_data = s0[20];
          tensorforge::VectorT<float, 4> v176_splat = {v175_data, v175_data, v175_data, v175_data};
          tensorforge::VectorT<float, 4> v177_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[8];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[8] = (tensorforge::fma(v164_data, v176_splat, v177_data));
          float v180_data = s0[28];
          tensorforge::VectorT<float, 4> v181_splat = {v180_data, v180_data, v180_data, v180_data};
          tensorforge::VectorT<float, 4> v182_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[12];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[12] = (tensorforge::fma(v164_data, v181_splat, v182_data));
          float v185_data = s0[36];
          tensorforge::VectorT<float, 4> v186_splat = {v185_data, v185_data, v185_data, v185_data};
          tensorforge::VectorT<float, 4> v187_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[16];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[16] = (tensorforge::fma(v164_data, v186_splat, v187_data));
          float v190_data = s0[44];
          tensorforge::VectorT<float, 4> v191_splat = {v190_data, v190_data, v190_data, v190_data};
          tensorforge::VectorT<float, 4> v192_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[20];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[20] = (tensorforge::fma(v164_data, v191_splat, v192_data));
          tensorforge::VectorT<float, 4> v199_data = ((v25_lead >= 1) && (v25_lead < 6)) ? (*(tensorforge::VectorT<float, 4>*)&glb_m1[(v28_vlead + 104)]) : (tensorforge::VectorT<float, 4>{});
          float v200_data = s0[5];
          tensorforge::VectorT<float, 4> v201_splat = {v200_data, v200_data, v200_data, v200_data};
          tensorforge::VectorT<float, 4> v202_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[0];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[0] = (tensorforge::fma(v199_data, v201_splat, v202_data));
          float v205_data = s0[13];
          tensorforge::VectorT<float, 4> v206_splat = {v205_data, v205_data, v205_data, v205_data};
          tensorforge::VectorT<float, 4> v207_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[4];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[4] = (tensorforge::fma(v199_data, v206_splat, v207_data));
          float v210_data = s0[21];
          tensorforge::VectorT<float, 4> v211_splat = {v210_data, v210_data, v210_data, v210_data};
          tensorforge::VectorT<float, 4> v212_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[8];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[8] = (tensorforge::fma(v199_data, v211_splat, v212_data));
          float v215_data = s0[29];
          tensorforge::VectorT<float, 4> v216_splat = {v215_data, v215_data, v215_data, v215_data};
          tensorforge::VectorT<float, 4> v217_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[12];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[12] = (tensorforge::fma(v199_data, v216_splat, v217_data));
          float v220_data = s0[37];
          tensorforge::VectorT<float, 4> v221_splat = {v220_data, v220_data, v220_data, v220_data};
          tensorforge::VectorT<float, 4> v222_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[16];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[16] = (tensorforge::fma(v199_data, v221_splat, v222_data));
          float v225_data = s0[45];
          tensorforge::VectorT<float, 4> v226_splat = {v225_data, v225_data, v225_data, v225_data};
          tensorforge::VectorT<float, 4> v227_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[20];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[20] = (tensorforge::fma(v199_data, v226_splat, v227_data));
          tensorforge::VectorT<float, 4> v234_data = ((v25_lead >= 9) && (v25_lead < 10)) ? (*(tensorforge::VectorT<float, 4>*)&glb_m1[(v28_vlead + 92)]) : (tensorforge::VectorT<float, 4>{});
          float v235_data = s0[6];
          tensorforge::VectorT<float, 4> v236_splat = {v235_data, v235_data, v235_data, v235_data};
          tensorforge::VectorT<float, 4> v237_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[0];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[0] = (tensorforge::fma(v234_data, v236_splat, v237_data));
          float v240_data = s0[14];
          tensorforge::VectorT<float, 4> v241_splat = {v240_data, v240_data, v240_data, v240_data};
          tensorforge::VectorT<float, 4> v242_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[4];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[4] = (tensorforge::fma(v234_data, v241_splat, v242_data));
          float v245_data = s0[22];
          tensorforge::VectorT<float, 4> v246_splat = {v245_data, v245_data, v245_data, v245_data};
          tensorforge::VectorT<float, 4> v247_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[8];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[8] = (tensorforge::fma(v234_data, v246_splat, v247_data));
          float v250_data = s0[30];
          tensorforge::VectorT<float, 4> v251_splat = {v250_data, v250_data, v250_data, v250_data};
          tensorforge::VectorT<float, 4> v252_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[12];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[12] = (tensorforge::fma(v234_data, v251_splat, v252_data));
          float v255_data = s0[38];
          tensorforge::VectorT<float, 4> v256_splat = {v255_data, v255_data, v255_data, v255_data};
          tensorforge::VectorT<float, 4> v257_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[16];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[16] = (tensorforge::fma(v234_data, v256_splat, v257_data));
          float v260_data = s0[46];
          tensorforge::VectorT<float, 4> v261_splat = {v260_data, v260_data, v260_data, v260_data};
          tensorforge::VectorT<float, 4> v262_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[20];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[20] = (tensorforge::fma(v234_data, v261_splat, v262_data));
          tensorforge::VectorT<float, 4> v269_data = ((v25_lead >= 3) && (v25_lead < 14)) ? (*(tensorforge::VectorT<float, 4>*)&glb_m1[(v28_vlead + 120)]) : (tensorforge::VectorT<float, 4>{});
          float v270_data = s0[7];
          tensorforge::VectorT<float, 4> v271_splat = {v270_data, v270_data, v270_data, v270_data};
          tensorforge::VectorT<float, 4> v272_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[0];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[0] = (tensorforge::fma(v269_data, v271_splat, v272_data));
          float v275_data = s0[15];
          tensorforge::VectorT<float, 4> v276_splat = {v275_data, v275_data, v275_data, v275_data};
          tensorforge::VectorT<float, 4> v277_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[4];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[4] = (tensorforge::fma(v269_data, v276_splat, v277_data));
          float v280_data = s0[23];
          tensorforge::VectorT<float, 4> v281_splat = {v280_data, v280_data, v280_data, v280_data};
          tensorforge::VectorT<float, 4> v282_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[8];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[8] = (tensorforge::fma(v269_data, v281_splat, v282_data));
          float v285_data = s0[31];
          tensorforge::VectorT<float, 4> v286_splat = {v285_data, v285_data, v285_data, v285_data};
          tensorforge::VectorT<float, 4> v287_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[12];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[12] = (tensorforge::fma(v269_data, v286_splat, v287_data));
          float v290_data = s0[39];
          tensorforge::VectorT<float, 4> v291_splat = {v290_data, v290_data, v290_data, v290_data};
          tensorforge::VectorT<float, 4> v292_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[16];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[16] = (tensorforge::fma(v269_data, v291_splat, v292_data));
          float v295_data = s0[47];
          tensorforge::VectorT<float, 4> v296_splat = {v295_data, v295_data, v295_data, v295_data};
          tensorforge::VectorT<float, 4> v297_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[20];
          *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[20] = (tensorforge::fma(v269_data, v296_splat, v297_data));
          // r0 = ir0
          #pragma unroll
          for (int32_t v301_n0 = 0; v301_n0 < 1; ++v301_n0) {
            int32_t v303_vslot = v301_n0 * 4;
            #pragma unroll
            for (int32_t v302_n1 = 0; v302_n1 < 6; ++v302_n1) {
              int32_t v305_a = v303_vslot + (v302_n1 * 4);
              tensorforge::VectorT<float, 4> v306_data = *(tensorforge::VectorRelaxedT<float, 4>*)&ir0[v305_a];
              *(tensorforge::VectorRelaxedT<float, 4>*)&r0[v305_a] = v306_data;
            }
          }
          // glb_m0 = store{r>g}(r0);
          #pragma unroll
          for (int32_t v309_i0 = 0; v309_i0 < 1; ++v309_i0) {
            int32_t v311_vslot = v309_i0 * 4;
            int32_t v317_vlead = (v25_lead + (v309_i0 * 16)) * 4;
            #pragma unroll
            for (int32_t v310_i1 = 0; v310_i1 < 6; ++v310_i1) {
              tensorforge::VectorT<float, 4> v314_data = *(tensorforge::VectorRelaxedT<float, 4>*)&r0[(v311_vslot + (v310_i1 * 4))];
              *(tensorforge::VectorRelaxedT<float, 4>*)&glb_m0[(v317_vlead + (v310_i1 * 64))] = v314_data;
            }
          }
          __syncwarp(0x0000ffffu << (threadIdx.y % 2 * 16));
        }
        else {
          __pipeline_wait_prior(0);
        }
        // s0 = load{g>s}(wrap_glb_m2[0, 1])
        if (v329_g) {
          __pipeline_memcpy_async(&s0[0 + 0 + 4 * threadIdx.x + 0], &wrap_glb_m2[0 + 0 + 4 * threadIdx.x + 0], 16);
        }
        __pipeline_commit();
        v325_flagWord = v326_flagWordNext;
        v326_flagWordNext = flagWordAhead;
      }
      __pipeline_wait_prior(0);
    }
  }
}

