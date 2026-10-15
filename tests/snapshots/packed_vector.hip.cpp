// === base name ===
kernel_43e297339862ffb2

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_43e297339862ffb2 = {{16, 16, 1}, 16, 64, 4, 16, 3072, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_43e297339862ffb2(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_43e297339862ffb2(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_43e297339862ffb2(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_43e297339862ffb2, block.x * block.y * block.z, 768 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (768 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_43e297339862ffb2, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (768 * sizeof(float)));
          blocksPerSM = std::max(blocksPerSM, std::min(blocksNoLds, blocksByLds));
        }
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
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 768 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_43e297339862ffb2(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_43e297339862ffb2(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_43e297339862ffb2), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_43e297339862ffb2, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_43e297339862ffb2(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: lead_vectorize=1,pack_operands=runs
    // launch: 16 lanes (64 active), lead width 4 x 16 per block = block 16x16x1, 3072 B shared, occupancy grid
    // operands:
    //   m0 64×6(64×6) {0..64}×{0..6} strided
    //   m1 64×8(64×8) {0..64}×{0..8} none
    //   m2 8×6(8×6) {0..8}×{0..6} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":64,"block":[16,16,1],"cooperative":false,"lead_width":4,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":768}],"shared_bytes":3072,"shared_elements":768,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[64,6]],"name":"m0","ordered":false,"parts":1,"shape":[64,6],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[64,8]],"name":"m1","ordered":false,"parts":1,"shape":[64,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,6]],"name":"m2","ordered":false,"parts":1,"shape":[8,6],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[64,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[64,6]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[64,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[64,8]},{"addressing":"strided","bbox":[[0,0],[8,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 512];
      size_t v3_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < numElements0) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      tensorforge::VectorT<float, 2> v9_ld = __builtin_nontemporal_load(&*(tensorforge::VectorT<float, 2>*)&ptr_glb_m1[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      *(tensorforge::VectorT<float, 2>*)&glb_m1[0 + 0 + 2 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v9_ld;
      __syncthreads();
      float r0[24]{};
      const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
      size_t v314_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v314_flagSecondAhead < numElements0) ? v314_flagSecondAhead : v3_batchId1)];
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v3_batchId1 * 48 + 0 + m2_extraOffset];
      int32_t v322_lead = threadIdx.x % 16;
      bool v323_g = v322_lead < 2;
      // r0 = load{g>r}(peel_glb_m2);
      if (v323_g) {
        int32_t v334_vlead = v322_lead * 4;
        #pragma unroll
        for (int32_t v335_i1 = 0; v335_i1 < 6; ++v335_i1) {
          tensorforge::VectorT<float, 4> v338_data = __builtin_nontemporal_load(&*(tensorforge::VectorT<float, 4>*)&peel_glb_m2[(v334_vlead + (v335_i1 * 8))]);
          *(tensorforge::VectorRelaxedT<float, 4>*)&r0[(v335_i1 * 4)] = v338_data;
        }
      }
      uint32_t v318_flagWord = flagWordFirst;
      uint32_t v319_flagWordNext = flagWordSecond;
      for (size_t v10_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v10_batchId0 < numElements0; v10_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v11_ahead1 = v10_batchId0 + (gridDim.x * blockDim.y);
        size_t v13_batchId1 = (v11_ahead1 < numElements0) ? v11_ahead1 : v10_batchId0;
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v13_batchId1 * 48 + 0 + m2_extraOffset];
        size_t v348_flagAheadAhead = v13_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v348_flagAheadAhead < numElements0) ? v348_flagAheadAhead : v13_batchId1)];
        const bool allowed = static_cast<bool>(v318_flagWord);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v10_batchId0 * 384 + 0 + m0_extraOffset];
          float r1[24]{};
          // r1 = +(glb_m1 * r0) + None
          // [(0, 64), (0, 6)] [(0, 8)]
          tensorforge::VectorT<float, 4> v35_data = *(tensorforge::VectorRelaxedT<float, 4>*)&r0[0];
          tensorforge::VectorT<float, 4> v36_data = *(tensorforge::VectorRelaxedT<float, 4>*)&r0[4];
          tensorforge::VectorT<float, 4> v37_data = *(tensorforge::VectorRelaxedT<float, 4>*)&r0[8];
          tensorforge::VectorT<float, 4> v38_data = *(tensorforge::VectorRelaxedT<float, 4>*)&r0[12];
          float v43_tp{};
          float v44_tp{};
          float v45_tp{};
          float v46_tp{};
          tensorforge::transpose4x4b32(v43_tp, v44_tp, v45_tp, v46_tp, (v35_data[0]), (v36_data[0]), (v37_data[0]), (v38_data[0]));
          float v51_tp{};
          float v52_tp{};
          float v53_tp{};
          float v54_tp{};
          tensorforge::transpose4x4b32(v51_tp, v52_tp, v53_tp, v54_tp, (v35_data[1]), (v36_data[1]), (v37_data[1]), (v38_data[1]));
          float v59_tp{};
          float v60_tp{};
          float v61_tp{};
          float v62_tp{};
          tensorforge::transpose4x4b32(v59_tp, v60_tp, v61_tp, v62_tp, (v35_data[2]), (v36_data[2]), (v37_data[2]), (v38_data[2]));
          float v67_tp{};
          float v68_tp{};
          float v69_tp{};
          float v70_tp{};
          tensorforge::transpose4x4b32(v67_tp, v68_tp, v69_tp, v70_tp, (v35_data[3]), (v36_data[3]), (v37_data[3]), (v38_data[3]));
          tensorforge::VectorT<float, 4> v71_acc{};
          tensorforge::VectorT<float, 4> v72_acc{};
          tensorforge::VectorT<float, 4> v73_acc{};
          tensorforge::VectorT<float, 4> v74_acc{};
          int32_t v77_vlead = v322_lead * 4;
          tensorforge::VectorT<float, 4> v79_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v77_vlead];
          float v80_el = v79_data[0];
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v80_el, v71_acc, 2, 0, 0);
          float v82_el = v79_data[1];
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v82_el, v72_acc, 2, 0, 0);
          float v84_el = v79_data[2];
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v84_el, v73_acc, 2, 0, 0);
          float v86_el = v79_data[3];
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v43_tp, v86_el, v74_acc, 2, 0, 0);
          int32_t v88_a = v77_vlead + 64;
          tensorforge::VectorT<float, 4> v89_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v88_a];
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, (v89_data[0]), v81_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, (v89_data[1]), v83_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, (v89_data[2]), v85_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, (v89_data[3]), v87_acc, 2, 0, 0);
          int32_t v98_a = v77_vlead + 128;
          tensorforge::VectorT<float, 4> v99_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v98_a];
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, (v99_data[0]), v91_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, (v99_data[1]), v93_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, (v99_data[2]), v95_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v59_tp, (v99_data[3]), v97_acc, 2, 0, 0);
          int32_t v108_a = v77_vlead + 192;
          tensorforge::VectorT<float, 4> v109_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v108_a];
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, (v109_data[0]), v101_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v113_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, (v109_data[1]), v103_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v115_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, (v109_data[2]), v105_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v117_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, (v109_data[3]), v107_acc, 2, 0, 0);
          int32_t v118_a = v77_vlead + 256;
          tensorforge::VectorT<float, 4> v119_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v118_a];
          tensorforge::VectorT<float, 4> v121_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, (v119_data[0]), v111_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v123_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, (v119_data[1]), v113_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v125_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, (v119_data[2]), v115_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v127_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v44_tp, (v119_data[3]), v117_acc, 2, 0, 0);
          int32_t v128_a = v77_vlead + 320;
          tensorforge::VectorT<float, 4> v129_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v128_a];
          tensorforge::VectorT<float, 4> v131_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, (v129_data[0]), v121_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, (v129_data[1]), v123_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, (v129_data[2]), v125_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v137_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, (v129_data[3]), v127_acc, 2, 0, 0);
          int32_t v138_a = v77_vlead + 384;
          tensorforge::VectorT<float, 4> v139_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v138_a];
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, (v139_data[0]), v131_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, (v139_data[1]), v133_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, (v139_data[2]), v135_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v60_tp, (v139_data[3]), v137_acc, 2, 0, 0);
          int32_t v148_a = v77_vlead + 448;
          tensorforge::VectorT<float, 4> v149_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v148_a];
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, (v149_data[0]), v141_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, (v149_data[1]), v143_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, (v149_data[2]), v145_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, (v149_data[3]), v147_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v162_pk = {(v151_acc[0]), (v153_acc[0]), (v155_acc[0]), (v157_acc[0])};
          *(tensorforge::VectorRelaxedT<float, 4>*)&r1[0] = v162_pk;
          tensorforge::VectorT<float, 4> v167_pk = {(v151_acc[1]), (v153_acc[1]), (v155_acc[1]), (v157_acc[1])};
          *(tensorforge::VectorRelaxedT<float, 4>*)&r1[4] = v167_pk;
          tensorforge::VectorT<float, 4> v172_pk = {(v151_acc[2]), (v153_acc[2]), (v155_acc[2]), (v157_acc[2])};
          *(tensorforge::VectorRelaxedT<float, 4>*)&r1[8] = v172_pk;
          tensorforge::VectorT<float, 4> v177_pk = {(v151_acc[3]), (v153_acc[3]), (v155_acc[3]), (v157_acc[3])};
          *(tensorforge::VectorRelaxedT<float, 4>*)&r1[12] = v177_pk;
          tensorforge::VectorT<float, 4> v178_data = *(tensorforge::VectorRelaxedT<float, 4>*)&r0[16];
          tensorforge::VectorT<float, 4> v179_data = *(tensorforge::VectorRelaxedT<float, 4>*)&r0[20];
          // r0 = load{g>r}(wrap_glb_m2);
          if (v323_g) {
            #pragma unroll
            for (int32_t v341_i1 = 0; v341_i1 < 6; ++v341_i1) {
              tensorforge::VectorT<float, 4> v344_data = __builtin_nontemporal_load(&*(tensorforge::VectorT<float, 4>*)&wrap_glb_m2[(v77_vlead + (v341_i1 * 8))]);
              *(tensorforge::VectorRelaxedT<float, 4>*)&r0[(v341_i1 * 4)] = v344_data;
            }
          }
          float v183_tp{};
          float v184_tp{};
          float v185_tp{};
          float v186_tp{};
          tensorforge::transpose4x4b32(v183_tp, v184_tp, v185_tp, v186_tp, (v178_data[0]), (v179_data[0]), 0.0f, 0.0f);
          float v189_tp{};
          float v190_tp{};
          float v191_tp{};
          float v192_tp{};
          tensorforge::transpose4x4b32(v189_tp, v190_tp, v191_tp, v192_tp, (v178_data[1]), (v179_data[1]), 0.0f, 0.0f);
          float v195_tp{};
          float v196_tp{};
          float v197_tp{};
          float v198_tp{};
          tensorforge::transpose4x4b32(v195_tp, v196_tp, v197_tp, v198_tp, (v178_data[2]), (v179_data[2]), 0.0f, 0.0f);
          float v201_tp{};
          float v202_tp{};
          float v203_tp{};
          float v204_tp{};
          tensorforge::transpose4x4b32(v201_tp, v202_tp, v203_tp, v204_tp, (v178_data[3]), (v179_data[3]), 0.0f, 0.0f);
          tensorforge::VectorT<float, 4> v205_acc{};
          tensorforge::VectorT<float, 4> v206_acc{};
          tensorforge::VectorT<float, 4> v207_acc{};
          tensorforge::VectorT<float, 4> v208_acc{};
          tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v80_el, v205_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v82_el, v206_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v84_el, v207_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v221_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v183_tp, v86_el, v208_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v223_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v88_a];
          tensorforge::VectorT<float, 4> v225_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, (v223_data[0]), v215_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v227_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, (v223_data[1]), v217_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v229_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, (v223_data[2]), v219_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v231_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, (v223_data[3]), v221_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v233_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v98_a];
          tensorforge::VectorT<float, 4> v235_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v195_tp, (v233_data[0]), v225_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v237_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v195_tp, (v233_data[1]), v227_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v195_tp, (v233_data[2]), v229_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v241_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v195_tp, (v233_data[3]), v231_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v243_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v108_a];
          tensorforge::VectorT<float, 4> v245_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, (v243_data[0]), v235_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, (v243_data[1]), v237_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, (v243_data[2]), v239_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v251_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, (v243_data[3]), v241_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v253_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v118_a];
          tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, (v253_data[0]), v245_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v257_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, (v253_data[1]), v247_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v259_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, (v253_data[2]), v249_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v261_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v184_tp, (v253_data[3]), v251_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v263_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v128_a];
          tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, (v263_data[0]), v255_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v267_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, (v263_data[1]), v257_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v269_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, (v263_data[2]), v259_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, (v263_data[3]), v261_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v273_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v138_a];
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, (v273_data[0]), v265_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v277_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, (v273_data[1]), v267_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, (v273_data[2]), v269_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, (v273_data[3]), v271_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v283_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[v148_a];
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, (v283_data[0]), v275_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, (v283_data[1]), v277_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, (v283_data[2]), v279_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, (v283_data[3]), v281_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v296_pk = {(v285_acc[0]), (v287_acc[0]), (v289_acc[0]), (v291_acc[0])};
          *(tensorforge::VectorRelaxedT<float, 4>*)&r1[16] = v296_pk;
          tensorforge::VectorT<float, 4> v301_pk = {(v285_acc[1]), (v287_acc[1]), (v289_acc[1]), (v291_acc[1])};
          *(tensorforge::VectorRelaxedT<float, 4>*)&r1[20] = v301_pk;
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v302_i0 = 0; v302_i0 < 1; ++v302_i0) {
            int32_t v304_vslot = v302_i0 * 4;
            int32_t v310_vlead = (v322_lead + (v302_i0 * 16)) * 4;
            #pragma unroll
            for (int32_t v303_i1 = 0; v303_i1 < 6; ++v303_i1) {
              tensorforge::VectorT<float, 4> v307_data = *(tensorforge::VectorRelaxedT<float, 4>*)&r1[(v304_vslot + (v303_i1 * 4))];
              *(tensorforge::VectorRelaxedT<float, 4>*)&glb_m0[(v310_vlead + (v303_i1 * 64))] = v307_data;
            }
          }
        }
        else {
          // r0 = load{g>r}(wrap_glb_m2);
          if (v323_g) {
            int32_t v328_vlead = v322_lead * 4;
            #pragma unroll
            for (int32_t v329_i1 = 0; v329_i1 < 6; ++v329_i1) {
              tensorforge::VectorT<float, 4> v332_data = __builtin_nontemporal_load(&*(tensorforge::VectorT<float, 4>*)&wrap_glb_m2[(v328_vlead + (v329_i1 * 8))]);
              *(tensorforge::VectorRelaxedT<float, 4>*)&r0[(v329_i1 * 4)] = v332_data;
            }
          }
        }
        v318_flagWord = v319_flagWordNext;
        v319_flagWordNext = flagWordAhead;
      }
    }
  }
}

