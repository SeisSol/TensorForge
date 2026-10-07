// === base name ===
kernel_b1d4324160e9e0f5

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b1d4324160e9e0f5 = {{16, 16, 1}, 16, 16, 1, 16, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b1d4324160e9e0f5(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b1d4324160e9e0f5(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b1d4324160e9e0f5(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_b1d4324160e9e0f5, block.x * block.y * block.z, 512 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (512 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_b1d4324160e9e0f5, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (512 * sizeof(float)));
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
  config.sharedMemBytes = 512 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_b1d4324160e9e0f5(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b1d4324160e9e0f5(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_b1d4324160e9e0f5), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_b1d4324160e9e0f5, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_b1d4324160e9e0f5(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 16 per block = block 16x16x1, 2048 B shared, occupancy grid
    // operands:
    //   m0 16×16(16×16) {0..16}×{0..16} strided
    //   m1 16×16(16×16) {0..16}×{0..16} none
    //   m2 16×16(16×16) {0..16}×{0..16} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":512}],"shared_bytes":2048,"shared_elements":512,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 256];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      float v9_ld = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v9_ld;
      // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
      __syncthreads();
      size_t v11_batchIdLane0 = threadIdx.y % 4;
      int32_t v27_lead = threadIdx.x % 16;
      int32_t v58_a = v27_lead + ((threadIdx.y % 4) * 16);
      int32_t v65_a = v58_a + 64;
      int32_t v71_a = v58_a + 128;
      int32_t v77_a = v58_a + 192;
      for (size_t v12_batchIdGroup0 = (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x); v12_batchIdGroup0 < numElements0; v12_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v13_row = v12_batchIdGroup0 + v11_batchIdLane0;
        const bool batchIdActive0 = v13_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v13_row]));
        size_t v15_batchId0 = batchIdActive0 ? v13_row : v12_batchIdGroup0;
        size_t v16_ahead1 = v15_batchId0 + (gridDim.x * blockDim.y);
        size_t v18_batchId1 = (v16_ahead1 < numElements0) ? v16_ahead1 : v15_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v15_batchId0 * 256 + 0 + m0_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v15_batchId0 * 256 + 0 + m2_extraOffset];
        float r0[16]{};
        // r0 = load{g>r}(glb_m2);
        #pragma unroll
        for (int32_t v28_i0 = 0; v28_i0 < 1; ++v28_i0) {
          int32_t v31_lead = v27_lead + (v28_i0 * 16);
          #pragma unroll
          for (int32_t v29_i1 = 0; v29_i1 < 16; ++v29_i1) {
            float v34_data = __builtin_nontemporal_load(&glb_m2[(v31_lead + (v29_i1 * 16))]);
            r0[(v28_i0 + v29_i1)] = v34_data;
          }
        }
        // wait(r0 = load{g>r}(glb_m2););
        float r1[16]{};
        // r1 = +(glb_m1 * r0) + None
        // [(0, 16), (0, 16)] [(0, 16)]
        float v37_data = r0[0];
        float v38_data = r0[1];
        float v39_data = r0[2];
        float v40_data = r0[3];
        float v41_data = r0[4];
        float v42_data = r0[5];
        float v43_data = r0[6];
        float v44_data = r0[7];
        float v45_data = r0[8];
        float v46_data = r0[9];
        float v47_data = r0[10];
        float v48_data = r0[11];
        float v49_data = r0[12];
        float v50_data = r0[13];
        float v51_data = r0[14];
        float v52_data = r0[15];
        tensorforge::transpose16x16b32(v37_data, v38_data, v39_data, v40_data, v41_data, v42_data, v43_data, v44_data, v45_data, v46_data, v47_data, v48_data, v49_data, v50_data, v51_data, v52_data);
        tensorforge::VectorT<float, 16> v53_acc{};
        float v60_data = glb_m1[v58_a];
        tensorforge::VectorT<float, 16> v61_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v37_data, v60_data, v53_acc, 0, 0, 4);
        tensorforge::VectorT<float, 16> v62_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v38_data, v60_data, v61_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v63_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v39_data, v60_data, v62_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v64_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v40_data, v60_data, v63_acc, 0, 0, 7);
        float v66_data = glb_m1[v65_a];
        tensorforge::VectorT<float, 16> v67_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v41_data, v66_data, v64_acc, 0, 0, 4);
        tensorforge::VectorT<float, 16> v68_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v42_data, v66_data, v67_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v69_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v43_data, v66_data, v68_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v70_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v44_data, v66_data, v69_acc, 0, 0, 7);
        float v72_data = glb_m1[v71_a];
        tensorforge::VectorT<float, 16> v73_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v45_data, v72_data, v70_acc, 0, 0, 4);
        tensorforge::VectorT<float, 16> v74_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v46_data, v72_data, v73_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v75_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v47_data, v72_data, v74_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v76_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v48_data, v72_data, v75_acc, 0, 0, 7);
        float v78_data = glb_m1[v77_a];
        tensorforge::VectorT<float, 16> v79_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v49_data, v78_data, v76_acc, 0, 0, 4);
        tensorforge::VectorT<float, 16> v80_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v50_data, v78_data, v79_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v81_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v51_data, v78_data, v80_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v82_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v52_data, v78_data, v81_acc, 0, 0, 7);
        float v83_el = v82_acc[0];
        float v85_el = v82_acc[4];
        float v86_sw = tensorforge::swap<32>(v85_el);
        float v88_el = v82_acc[8];
        float v91_el = v82_acc[12];
        float v92_sw = tensorforge::swap<32>(v91_el);
        r1[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v92_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v88_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v86_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v83_el, v83_el))))))));
        float v95_el = v82_acc[1];
        float v97_el = v82_acc[5];
        float v98_sw = tensorforge::swap<32>(v97_el);
        float v100_el = v82_acc[9];
        float v103_el = v82_acc[13];
        float v104_sw = tensorforge::swap<32>(v103_el);
        r1[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v104_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v100_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v98_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v95_el, v95_el))))))));
        float v107_el = v82_acc[2];
        float v109_el = v82_acc[6];
        float v110_sw = tensorforge::swap<32>(v109_el);
        float v112_el = v82_acc[10];
        float v115_el = v82_acc[14];
        float v116_sw = tensorforge::swap<32>(v115_el);
        r1[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v116_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v112_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v110_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v107_el, v107_el))))))));
        float v119_el = v82_acc[3];
        float v121_el = v82_acc[7];
        float v122_sw = tensorforge::swap<32>(v121_el);
        float v124_el = v82_acc[11];
        float v127_el = v82_acc[15];
        float v128_sw = tensorforge::swap<32>(v127_el);
        r1[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v128_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v124_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v122_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v119_el, v119_el))))))));
        float v132_sw = tensorforge::swap<32>(v83_el);
        float v137_sw = tensorforge::swap<32>(v88_el);
        r1[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v91_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v137_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v85_el, (tensorforge::dppUpdate<228, 1, 15, false>(v132_sw, v132_sw))))))));
        float v144_sw = tensorforge::swap<32>(v95_el);
        float v149_sw = tensorforge::swap<32>(v100_el);
        r1[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v103_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v149_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v97_el, (tensorforge::dppUpdate<228, 1, 15, false>(v144_sw, v144_sw))))))));
        float v156_sw = tensorforge::swap<32>(v107_el);
        float v161_sw = tensorforge::swap<32>(v112_el);
        r1[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v115_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v161_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v109_el, (tensorforge::dppUpdate<228, 1, 15, false>(v156_sw, v156_sw))))))));
        float v168_sw = tensorforge::swap<32>(v119_el);
        float v173_sw = tensorforge::swap<32>(v124_el);
        r1[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v127_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v173_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v121_el, (tensorforge::dppUpdate<228, 1, 15, false>(v168_sw, v168_sw))))))));
        float v180_sw = tensorforge::swap<64>(v83_el);
        r1[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v92_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v88_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v86_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v180_sw, v180_sw))))))));
        float v192_sw = tensorforge::swap<64>(v95_el);
        r1[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v104_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v100_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v98_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v192_sw, v192_sw))))))));
        float v204_sw = tensorforge::swap<64>(v107_el);
        r1[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v116_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v112_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v110_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v204_sw, v204_sw))))))));
        float v216_sw = tensorforge::swap<64>(v119_el);
        r1[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v128_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v124_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v122_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v216_sw, v216_sw))))))));
        float v229_sw = tensorforge::swap<64>(v132_sw);
        r1[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v91_el, (tensorforge::dppUpdate<228, 4, 15, false>(v137_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v85_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v229_sw, v229_sw))))))));
        float v241_sw = tensorforge::swap<64>(v144_sw);
        r1[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v103_el, (tensorforge::dppUpdate<228, 4, 15, false>(v149_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v97_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v241_sw, v241_sw))))))));
        float v253_sw = tensorforge::swap<64>(v156_sw);
        r1[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v115_el, (tensorforge::dppUpdate<228, 4, 15, false>(v161_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v109_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v253_sw, v253_sw))))))));
        float v265_sw = tensorforge::swap<64>(v168_sw);
        r1[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v127_el, (tensorforge::dppUpdate<228, 4, 15, false>(v173_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v121_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v265_sw, v265_sw))))))));
        // glb_m0 = store{r>g}(r1);
        #pragma unroll
        for (int32_t v275_i0 = 0; v275_i0 < 1; ++v275_i0) {
          #pragma unroll
          for (int32_t v276_i1 = 0; v276_i1 < 16; ++v276_i1) {
            float v278_data = r1[(v275_i0 + v276_i1)];
            if (batchIdActive0) {
              glb_m0[((v27_lead + (v275_i0 * 16)) + (v276_i1 * 16))] = v278_data;
            }
          }
        }
      }
    }
  }
}

