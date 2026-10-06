// === base name ===
kernel_ae9bb252c8e882d9

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_ae9bb252c8e882d9 = {{16, 16, 1}, 16, 16, 1, 16, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_ae9bb252c8e882d9(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_ae9bb252c8e882d9(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_ae9bb252c8e882d9(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_ae9bb252c8e882d9, block.x * block.y * block.z, 512 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (512 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_ae9bb252c8e882d9, block.x * block.y * block.z, 0));
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
void launcher_kernel_ae9bb252c8e882d9(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_ae9bb252c8e882d9(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_ae9bb252c8e882d9), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_ae9bb252c8e882d9, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_ae9bb252c8e882d9(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> m1, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
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
      float* tempShrMem = &localShrMem0[0];
      tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::ConstantMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      float v12_ld = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v12_ld;
      // wait(glb_m1 = load{g>s}(ptr_glb_m1[0, 1]));
      __syncthreads();
      size_t v14_batchIdLane0 = threadIdx.y % 4;
      int32_t v30_lead = threadIdx.x % 16;
      int32_t v61_a = v30_lead + ((threadIdx.y % 4) * 16);
      int32_t v68_a = v61_a + 64;
      int32_t v74_a = v61_a + 128;
      int32_t v80_a = v61_a + 192;
      for (size_t v15_batchIdGroup0 = (threadIdx.y - threadIdx.y % 4) + blockDim.y * (blockIdx.x); v15_batchIdGroup0 < numElements0; v15_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v16_row = v15_batchIdGroup0 + v14_batchIdLane0;
        const bool batchIdActive0 = v16_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v16_row]));
        size_t v18_batchId0 = batchIdActive0 ? v16_row : v15_batchIdGroup0;
        size_t v19_ahead1 = v18_batchId0 + (gridDim.x * blockDim.y);
        size_t v21_batchId1 = (v19_ahead1 < numElements0) ? v19_ahead1 : v18_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v18_batchId0 * 256 + 0 + m0_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v18_batchId0 * 256 + 0 + m2_extraOffset];
        float r0[16]{};
        // r0 = load{g>r}(glb_m2);
        #pragma unroll
        for (int32_t v31_i0 = 0; v31_i0 < 1; ++v31_i0) {
          int32_t v34_lead = v30_lead + (v31_i0 * 16);
          #pragma unroll
          for (int32_t v32_i1 = 0; v32_i1 < 16; ++v32_i1) {
            float v37_data = __builtin_nontemporal_load(&glb_m2[(v34_lead + (v32_i1 * 16))]);
            r0[(v31_i0 + v32_i1)] = v37_data;
          }
        }
        // wait(r0 = load{g>r}(glb_m2););
        float r1[16]{};
        // r1 = +(glb_m1 * r0) + None
        // [(0, 16), (0, 16)] [(0, 16)]
        float v40_data = r0[0];
        float v41_data = r0[1];
        float v42_data = r0[2];
        float v43_data = r0[3];
        float v44_data = r0[4];
        float v45_data = r0[5];
        float v46_data = r0[6];
        float v47_data = r0[7];
        float v48_data = r0[8];
        float v49_data = r0[9];
        float v50_data = r0[10];
        float v51_data = r0[11];
        float v52_data = r0[12];
        float v53_data = r0[13];
        float v54_data = r0[14];
        float v55_data = r0[15];
        tensorforge::transpose16x16b32(v40_data, v41_data, v42_data, v43_data, v44_data, v45_data, v46_data, v47_data, v48_data, v49_data, v50_data, v51_data, v52_data, v53_data, v54_data, v55_data);
        tensorforge::VectorT<float, 16> v56_acc{};
        float v63_data = glb_m1[v61_a];
        tensorforge::VectorT<float, 16> v64_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v40_data, v63_data, v56_acc, 0, 0, 4);
        tensorforge::VectorT<float, 16> v65_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v41_data, v63_data, v64_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v66_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v42_data, v63_data, v65_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v67_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v43_data, v63_data, v66_acc, 0, 0, 7);
        float v69_data = glb_m1[v68_a];
        tensorforge::VectorT<float, 16> v70_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v44_data, v69_data, v67_acc, 0, 0, 4);
        tensorforge::VectorT<float, 16> v71_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v45_data, v69_data, v70_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v72_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v46_data, v69_data, v71_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v73_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v47_data, v69_data, v72_acc, 0, 0, 7);
        float v75_data = glb_m1[v74_a];
        tensorforge::VectorT<float, 16> v76_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v48_data, v75_data, v73_acc, 0, 0, 4);
        tensorforge::VectorT<float, 16> v77_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v49_data, v75_data, v76_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v78_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v50_data, v75_data, v77_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v79_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v51_data, v75_data, v78_acc, 0, 0, 7);
        float v81_data = glb_m1[v80_a];
        tensorforge::VectorT<float, 16> v82_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v52_data, v81_data, v79_acc, 0, 0, 4);
        tensorforge::VectorT<float, 16> v83_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v53_data, v81_data, v82_acc, 0, 0, 5);
        tensorforge::VectorT<float, 16> v84_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v54_data, v81_data, v83_acc, 0, 0, 6);
        tensorforge::VectorT<float, 16> v85_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v55_data, v81_data, v84_acc, 0, 0, 7);
        float v86_el = v85_acc[0];
        float v88_el = v85_acc[4];
        float v89_sw = tensorforge::swap<32>(v88_el);
        float v91_el = v85_acc[8];
        float v94_el = v85_acc[12];
        float v95_sw = tensorforge::swap<32>(v94_el);
        r1[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v95_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v91_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v89_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v86_el, v86_el))))))));
        float v98_el = v85_acc[1];
        float v100_el = v85_acc[5];
        float v101_sw = tensorforge::swap<32>(v100_el);
        float v103_el = v85_acc[9];
        float v106_el = v85_acc[13];
        float v107_sw = tensorforge::swap<32>(v106_el);
        r1[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v107_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v103_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v101_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v98_el, v98_el))))))));
        float v110_el = v85_acc[2];
        float v112_el = v85_acc[6];
        float v113_sw = tensorforge::swap<32>(v112_el);
        float v115_el = v85_acc[10];
        float v118_el = v85_acc[14];
        float v119_sw = tensorforge::swap<32>(v118_el);
        r1[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v119_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v115_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v113_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v110_el, v110_el))))))));
        float v122_el = v85_acc[3];
        float v124_el = v85_acc[7];
        float v125_sw = tensorforge::swap<32>(v124_el);
        float v127_el = v85_acc[11];
        float v130_el = v85_acc[15];
        float v131_sw = tensorforge::swap<32>(v130_el);
        r1[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v131_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v127_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v125_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v122_el, v122_el))))))));
        float v135_sw = tensorforge::swap<32>(v86_el);
        float v140_sw = tensorforge::swap<32>(v91_el);
        r1[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v94_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v140_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v88_el, (tensorforge::dppUpdate<228, 1, 15, false>(v135_sw, v135_sw))))))));
        float v147_sw = tensorforge::swap<32>(v98_el);
        float v152_sw = tensorforge::swap<32>(v103_el);
        r1[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v106_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v152_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v100_el, (tensorforge::dppUpdate<228, 1, 15, false>(v147_sw, v147_sw))))))));
        float v159_sw = tensorforge::swap<32>(v110_el);
        float v164_sw = tensorforge::swap<32>(v115_el);
        r1[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v118_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v164_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v112_el, (tensorforge::dppUpdate<228, 1, 15, false>(v159_sw, v159_sw))))))));
        float v171_sw = tensorforge::swap<32>(v122_el);
        float v176_sw = tensorforge::swap<32>(v127_el);
        r1[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v130_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v176_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v124_el, (tensorforge::dppUpdate<228, 1, 15, false>(v171_sw, v171_sw))))))));
        float v183_sw = tensorforge::swap<64>(v86_el);
        r1[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v95_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v91_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v89_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v183_sw, v183_sw))))))));
        float v195_sw = tensorforge::swap<64>(v98_el);
        r1[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v107_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v103_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v101_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v195_sw, v195_sw))))))));
        float v207_sw = tensorforge::swap<64>(v110_el);
        r1[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v119_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v115_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v113_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v207_sw, v207_sw))))))));
        float v219_sw = tensorforge::swap<64>(v122_el);
        r1[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v131_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v127_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v125_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v219_sw, v219_sw))))))));
        float v232_sw = tensorforge::swap<64>(v135_sw);
        r1[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v94_el, (tensorforge::dppUpdate<228, 4, 15, false>(v140_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v88_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v232_sw, v232_sw))))))));
        float v244_sw = tensorforge::swap<64>(v147_sw);
        r1[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v106_el, (tensorforge::dppUpdate<228, 4, 15, false>(v152_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v100_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v244_sw, v244_sw))))))));
        float v256_sw = tensorforge::swap<64>(v159_sw);
        r1[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v118_el, (tensorforge::dppUpdate<228, 4, 15, false>(v164_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v112_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v256_sw, v256_sw))))))));
        float v268_sw = tensorforge::swap<64>(v171_sw);
        r1[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v130_el, (tensorforge::dppUpdate<228, 4, 15, false>(v176_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v124_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v268_sw, v268_sw))))))));
        // glb_m0 = store{r>g}(r1);
        #pragma unroll
        for (int32_t v278_i0 = 0; v278_i0 < 1; ++v278_i0) {
          #pragma unroll
          for (int32_t v279_i1 = 0; v279_i1 < 16; ++v279_i1) {
            float v281_data = r1[(v278_i0 + v279_i1)];
            if (batchIdActive0) {
              glb_m0[((v30_lead + (v278_i0 * 16)) + (v279_i1 * 16))] = v281_data;
            }
          }
        }
      }
    }
  }
}

