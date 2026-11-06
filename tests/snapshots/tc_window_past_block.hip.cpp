// === base name ===
kernel_46ce2dec4a79dadf

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_46ce2dec4a79dadf = {{8, 32, 1}, 8, 8, 1, 32, 1280, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_46ce2dec4a79dadf(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_46ce2dec4a79dadf(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_46ce2dec4a79dadf(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (8, 32, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_46ce2dec4a79dadf, block.x * block.y * block.z, 320 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (320 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_46ce2dec4a79dadf, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (320 * sizeof(float)));
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
  config.block[0] = 8;
  config.block[1] = 32;
  config.block[2] = 1;
  config.sharedMemBytes = 320 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_46ce2dec4a79dadf(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_46ce2dec4a79dadf(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_46ce2dec4a79dadf), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_46ce2dec4a79dadf, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_46ce2dec4a79dadf(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 8 lanes x 32 per block = block 8x32x1, 1280 B shared, occupancy grid
    // operands:
    //   m0 8×9(8×9) {0..8}×{0..9} strided
    //   m1 8×16(8×4) {0..8}×{9..13} none ordered
    //   m2 16×9(4×9) {9..13}×{0..9} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,32,1],"cooperative":false,"lead_width":1,"mults_per_block":32,"persistent":true,"sections":[{"barrier":false,"mults_per_block":32,"shared_elements":320}],"shared_bytes":1280,"shared_elements":320,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[8,9]],"name":"m0","ordered":false,"parts":1,"shape":[8,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,9],[8,13]],"name":"m1","ordered":true,"parts":1,"shape":[8,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[9,0],[13,9]],"name":"m2","ordered":false,"parts":1,"shape":[16,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,9],[8,13]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,16]},{"addressing":"strided","bbox":[[9,0],[13,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[8 * threadIdx.y + 64];
      size_t v3_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < numElements0) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      if ((threadIdx.x + (threadIdx.y * 8)) < 32) {
        float v14_ld = ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0];
        glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v14_ld;
      }
      __syncthreads();
      float r0[9]{};
      const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
      size_t v123_flagSecondAhead = v3_batchId1 + (gridDim.x * blockDim.y);
      const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v123_flagSecondAhead < numElements0) ? v123_flagSecondAhead : v3_batchId1)];
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v3_batchId1 * 36 + 0 + m2_extraOffset];
      int32_t v131_lead = threadIdx.x % 8;
      bool v134_g = (v131_lead >= 1) && (v131_lead < 5);
      // r0 = load{g>r}(peel_glb_m2);
      if (v134_g) {
        int32_t v150_a = (v131_lead + 8_i32) - 9;
        #pragma unroll
        for (int32_t v151_i1 = 0; v151_i1 < 9; ++v151_i1) {
          float v154_data = __builtin_nontemporal_load(&peel_glb_m2[(v150_a + (v151_i1 * 4))]);
          r0[v151_i1] = v154_data;
        }
      }
      uint32_t v127_flagWord = flagWordFirst;
      uint32_t v128_flagWordNext = flagWordSecond;
      for (size_t v15_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v15_batchId0 < numElements0; v15_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v16_ahead1 = v15_batchId0 + (gridDim.x * blockDim.y);
        size_t v18_batchId1 = (v16_ahead1 < numElements0) ? v16_ahead1 : v15_batchId0;
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v18_batchId1 * 36 + 0 + m2_extraOffset];
        size_t v164_flagAheadAhead = v18_batchId1 + (gridDim.x * blockDim.y);
        const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v164_flagAheadAhead < numElements0) ? v164_flagAheadAhead : v18_batchId1)];
        const bool allowed = static_cast<bool>(v127_flagWord);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v15_batchId0 * 72 + 0 + m0_extraOffset];
          float r1[9]{};
          // ir1 = +(glb_m1 * r0)
          // [(0, 8), (0, 9)] [(9, 13)]
          float ir1[9]{};
          float v42_data = r0[0];
          float v43_data = r0[1];
          float v44_data = r0[2];
          float v45_data = r0[3];
          float v46_tp{};
          float v47_tp{};
          float v48_tp{};
          float v49_tp{};
          tensorforge::transpose4x4b32(v46_tp, v47_tp, v48_tp, v49_tp, v42_data, v43_data, v44_data, v45_data);
          tensorforge::VectorT<float, 4> v50_acc{};
          tensorforge::VectorT<float, 4> v55_data = *(tensorforge::VectorT<float, 4>*)&glb_m1[(v131_lead * 4)];
          float v56_el = v55_data[0];
          tensorforge::VectorT<float, 4> v57_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v47_tp, v56_el, v50_acc, 1, 0, 0);
          float v58_el = v55_data[1];
          tensorforge::VectorT<float, 4> v59_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v48_tp, v58_el, v57_acc, 1, 0, 0);
          float v60_el = v55_data[2];
          tensorforge::VectorT<float, 4> v61_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v60_el, v59_acc, 1, 0, 0);
          float v62_el = v55_data[3];
          tensorforge::VectorT<float, 4> v63_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v46_tp, v62_el, v61_acc, 1, 1, 0);
          ir1[0] = (v63_acc[0]);
          ir1[1] = (v63_acc[1]);
          ir1[2] = (v63_acc[2]);
          ir1[3] = (v63_acc[3]);
          float v68_data = r0[4];
          float v69_data = r0[5];
          float v70_data = r0[6];
          float v71_data = r0[7];
          float v72_tp{};
          float v73_tp{};
          float v74_tp{};
          float v75_tp{};
          tensorforge::transpose4x4b32(v72_tp, v73_tp, v74_tp, v75_tp, v68_data, v69_data, v70_data, v71_data);
          tensorforge::VectorT<float, 4> v76_acc{};
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v73_tp, v56_el, v76_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v74_tp, v58_el, v83_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v75_tp, v60_el, v85_acc, 1, 0, 0);
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v72_tp, v62_el, v87_acc, 1, 1, 0);
          ir1[4] = (v89_acc[0]);
          ir1[5] = (v89_acc[1]);
          ir1[6] = (v89_acc[2]);
          ir1[7] = (v89_acc[3]);
          float v106_acc{};
          float v107_data = r0[8];
          // r0 = load{g>r}(wrap_glb_m2);
          if (v134_g) {
            int32_t v157_a = (v131_lead + 8_i32) - 9;
            #pragma unroll
            for (int32_t v158_i1 = 0; v158_i1 < 9; ++v158_i1) {
              float v161_data = __builtin_nontemporal_load(&wrap_glb_m2[(v157_a + (v158_i1 * 4))]);
              r0[v158_i1] = v161_data;
            }
          }
          float v108_bc = tensorforge::broadcast<8, 4, 0>(v107_data);
          asm("s_nop 1\n\t"
          "v_fmac_f32_dpp %0, %1, %2 quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f32_dpp %0, %1, %3 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f32_dpp %0, %1, %4 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "s_nop 1\n\t"
          : "+&v"(v106_acc)
          : "v"(v108_bc), "v"(v56_el), "v"(v58_el), "v"(v60_el)
          );
          asm("s_nop 1\n\t"
          "v_fmac_f32_dpp %0, %1, %2 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "s_nop 1\n\t"
          : "+&v"(v106_acc)
          : "v"((tensorforge::broadcast<8, 4, 1>(v107_data))), "v"(v62_el)
          );
          ir1[8] = v106_acc;
          // r1 = ir1
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v114_i0 = 0; v114_i0 < 1; ++v114_i0) {
            int32_t v119_lead = v131_lead + (v114_i0 * 8);
            #pragma unroll
            for (int32_t v115_i1 = 0; v115_i1 < 9; ++v115_i1) {
              float v117_data = ir1[v115_i1];
              glb_m0[(v119_lead + (v115_i1 * 8))] = v117_data;
            }
          }
        }
        else {
          // r0 = load{g>r}(wrap_glb_m2);
          if (v134_g) {
            int32_t v143_a = (v131_lead + 8_i32) - 9;
            #pragma unroll
            for (int32_t v144_i1 = 0; v144_i1 < 9; ++v144_i1) {
              float v147_data = __builtin_nontemporal_load(&wrap_glb_m2[(v143_a + (v144_i1 * 4))]);
              r0[v144_i1] = v147_data;
            }
          }
        }
        v127_flagWord = v128_flagWordNext;
        v128_flagWordNext = flagWordAhead;
      }
    }
  }
}

