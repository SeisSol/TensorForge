// === base name ===
kernel_6928d77f5218cf60

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_6928d77f5218cf60 = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_6928d77f5218cf60(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_6928d77f5218cf60(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_6928d77f5218cf60(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_6928d77f5218cf60, block.x * block.y * block.z, 256 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_6928d77f5218cf60, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (256 * sizeof(float)));
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
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_6928d77f5218cf60(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_6928d77f5218cf60(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_6928d77f5218cf60), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  hipLaunchKernelGGL(kernel_kernel_6928d77f5218cf60, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, numElements0);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_6928d77f5218cf60(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
    // operands:
    //   m0 16×16(16×16) {0..16}×{0..16} strided
    //   m1 16×16(16×16) {0..16}×{0..16} strided
    //   m2 16×16(16×16) {0..16}×{0..16} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[0];
      int32_t v23_lead = threadIdx.x % 16;
      for (size_t v10_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v10_batchId0 < numElements0; v10_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v11_ahead1 = v10_batchId0 + (gridDim.x * blockDim.y);
        size_t v13_batchId1 = (v11_ahead1 < numElements0) ? v11_ahead1 : v10_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v10_batchId0 * 256 + 0 + m0_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v10_batchId0 * 256 + 0 + m1_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v10_batchId0 * 256 + 0 + m2_extraOffset];
        float r0[16]{};
        // r0 = load{g>r}(glb_m1);
        #pragma unroll
        for (int32_t v24_i0 = 0; v24_i0 < 1; ++v24_i0) {
          int32_t v27_lead = v23_lead + (v24_i0 * 16);
          #pragma unroll
          for (int32_t v25_i1 = 0; v25_i1 < 16; ++v25_i1) {
            float v30_data = __builtin_nontemporal_load(&glb_m1[(v27_lead + (v25_i1 * 16))]);
            r0[(v24_i0 + v25_i1)] = v30_data;
          }
        }
        float r1[16]{};
        // r1 = load{g>r}(glb_m2);
        #pragma unroll
        for (int32_t v33_i0 = 0; v33_i0 < 1; ++v33_i0) {
          int32_t v36_lead = v23_lead + (v33_i0 * 16);
          #pragma unroll
          for (int32_t v34_i1 = 0; v34_i1 < 16; ++v34_i1) {
            float v39_data = __builtin_nontemporal_load(&glb_m2[(v36_lead + (v34_i1 * 16))]);
            r1[(v33_i0 + v34_i1)] = v39_data;
          }
        }
        // wait(r0 = load{g>r}(glb_m1););
        // wait(r1 = load{g>r}(glb_m2););
        float r2[16]{};
        // r2 = +(r0 * r1) + None
        // [(0, 16), (0, 16)] [(0, 16)]
        float v42_data = r1[0];
        float v43_data = r1[1];
        float v44_data = r1[2];
        float v45_data = r1[3];
        float v46_data = r1[4];
        float v47_data = r1[5];
        float v48_data = r1[6];
        float v49_data = r1[7];
        float v50_data = r1[8];
        float v51_data = r1[9];
        float v52_data = r1[10];
        float v53_data = r1[11];
        float v54_data = r1[12];
        float v55_data = r1[13];
        float v56_data = r1[14];
        float v57_data = r1[15];
        tensorforge::transpose16x16b32(v42_data, v43_data, v44_data, v45_data, v46_data, v47_data, v48_data, v49_data, v50_data, v51_data, v52_data, v53_data, v54_data, v55_data, v56_data, v57_data);
        tensorforge::VectorT<float, 16> v58_acc{};
        float v59_data = r0[0];
        float v60_data = r0[1];
        float v61_data = r0[2];
        float v62_data = r0[3];
        float v63_data = r0[4];
        float v64_data = r0[5];
        float v65_data = r0[6];
        float v66_data = r0[7];
        float v67_data = r0[8];
        float v68_data = r0[9];
        float v69_data = r0[10];
        float v70_data = r0[11];
        float v71_data = r0[12];
        float v72_data = r0[13];
        float v73_data = r0[14];
        float v74_data = r0[15];
        tensorforge::VectorT<float, 16> v75_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v42_data, v59_data, v58_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v76_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v43_data, v60_data, v75_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v77_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v44_data, v61_data, v76_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v78_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v45_data, v62_data, v77_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v79_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v46_data, v63_data, v78_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v80_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v47_data, v64_data, v79_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v81_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v48_data, v65_data, v80_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v82_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v49_data, v66_data, v81_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v83_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v50_data, v67_data, v82_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v84_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v51_data, v68_data, v83_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v85_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v52_data, v69_data, v84_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v86_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v53_data, v70_data, v85_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v87_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v54_data, v71_data, v86_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v88_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v55_data, v72_data, v87_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v89_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v56_data, v73_data, v88_acc, 0, 0, 0);
        tensorforge::VectorT<float, 16> v90_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v57_data, v74_data, v89_acc, 0, 0, 0);
        float v91_el = v90_acc[0];
        float v93_el = v90_acc[4];
        float v94_sw = tensorforge::swap<32>(v93_el);
        float v96_el = v90_acc[8];
        float v99_el = v90_acc[12];
        float v100_sw = tensorforge::swap<32>(v99_el);
        r2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v100_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v96_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v94_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v91_el, v91_el))))))));
        float v103_el = v90_acc[1];
        float v105_el = v90_acc[5];
        float v106_sw = tensorforge::swap<32>(v105_el);
        float v108_el = v90_acc[9];
        float v111_el = v90_acc[13];
        float v112_sw = tensorforge::swap<32>(v111_el);
        r2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v112_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v108_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v106_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v103_el, v103_el))))))));
        float v115_el = v90_acc[2];
        float v117_el = v90_acc[6];
        float v118_sw = tensorforge::swap<32>(v117_el);
        float v120_el = v90_acc[10];
        float v123_el = v90_acc[14];
        float v124_sw = tensorforge::swap<32>(v123_el);
        r2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v124_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v120_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v118_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v115_el, v115_el))))))));
        float v127_el = v90_acc[3];
        float v129_el = v90_acc[7];
        float v130_sw = tensorforge::swap<32>(v129_el);
        float v132_el = v90_acc[11];
        float v135_el = v90_acc[15];
        float v136_sw = tensorforge::swap<32>(v135_el);
        r2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v136_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v132_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v130_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v127_el, v127_el))))))));
        float v140_sw = tensorforge::swap<32>(v91_el);
        float v145_sw = tensorforge::swap<32>(v96_el);
        r2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v99_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v145_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v93_el, (tensorforge::dppUpdate<228, 1, 15, false>(v140_sw, v140_sw))))))));
        float v152_sw = tensorforge::swap<32>(v103_el);
        float v157_sw = tensorforge::swap<32>(v108_el);
        r2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v111_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v157_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v105_el, (tensorforge::dppUpdate<228, 1, 15, false>(v152_sw, v152_sw))))))));
        float v164_sw = tensorforge::swap<32>(v115_el);
        float v169_sw = tensorforge::swap<32>(v120_el);
        r2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v123_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v169_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v117_el, (tensorforge::dppUpdate<228, 1, 15, false>(v164_sw, v164_sw))))))));
        float v176_sw = tensorforge::swap<32>(v127_el);
        float v181_sw = tensorforge::swap<32>(v132_el);
        r2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v135_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v181_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v129_el, (tensorforge::dppUpdate<228, 1, 15, false>(v176_sw, v176_sw))))))));
        float v188_sw = tensorforge::swap<64>(v91_el);
        r2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v100_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v96_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v94_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v188_sw, v188_sw))))))));
        float v200_sw = tensorforge::swap<64>(v103_el);
        r2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v112_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v108_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v106_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v200_sw, v200_sw))))))));
        float v212_sw = tensorforge::swap<64>(v115_el);
        r2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v124_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v120_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v118_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v212_sw, v212_sw))))))));
        float v224_sw = tensorforge::swap<64>(v127_el);
        r2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v136_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v132_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v130_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v224_sw, v224_sw))))))));
        float v237_sw = tensorforge::swap<64>(v140_sw);
        r2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v99_el, (tensorforge::dppUpdate<228, 4, 15, false>(v145_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v93_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v237_sw, v237_sw))))))));
        float v249_sw = tensorforge::swap<64>(v152_sw);
        r2[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v111_el, (tensorforge::dppUpdate<228, 4, 15, false>(v157_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v105_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v249_sw, v249_sw))))))));
        float v261_sw = tensorforge::swap<64>(v164_sw);
        r2[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v123_el, (tensorforge::dppUpdate<228, 4, 15, false>(v169_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v117_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v261_sw, v261_sw))))))));
        float v273_sw = tensorforge::swap<64>(v176_sw);
        r2[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v135_el, (tensorforge::dppUpdate<228, 4, 15, false>(v181_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v129_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v273_sw, v273_sw))))))));
        // glb_m0 = store{r>g}(r2);
        #pragma unroll
        for (int32_t v283_i0 = 0; v283_i0 < 1; ++v283_i0) {
          int32_t v288_lead = v23_lead + (v283_i0 * 16);
          #pragma unroll
          for (int32_t v284_i1 = 0; v284_i1 < 16; ++v284_i1) {
            float v286_data = r2[(v283_i0 + v284_i1)];
            glb_m0[(v288_lead + (v284_i1 * 16))] = v286_data;
          }
        }
      }
    }
  }
}

