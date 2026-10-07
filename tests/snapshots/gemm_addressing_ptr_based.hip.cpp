// === base name ===
kernel_af46474a310a5d86

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_af46474a310a5d86 = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_af46474a310a5d86(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_af46474a310a5d86(float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_af46474a310a5d86(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_af46474a310a5d86, block.x * block.y * block.z, 256 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_af46474a310a5d86, block.x * block.y * block.z, 0));
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
void launcher_kernel_af46474a310a5d86(float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_af46474a310a5d86(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_af46474a310a5d86), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_af46474a310a5d86, grid, block, config.sharedMemBytes, stream, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_af46474a310a5d86(float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
    // operands:
    //   m0 16×16(16×16) {0..16}×{0..16} pointer_based
    //   m1 16×16(16×16) {0..16}×{0..16} pointer_based
    //   m2 16×16(16×16) {0..16}×{0..16} pointer_based
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"pointer_based","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"pointer_based","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"pointer_based","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"pointer_based","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v7_batchId0][0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v7_batchId0][0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v7_batchId0][0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v21_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v22_i0 = 0; v22_i0 < 1; ++v22_i0) {
            int32_t v25_lead = v21_lead + (v22_i0 * 16);
            #pragma unroll
            for (int32_t v23_i1 = 0; v23_i1 < 16; ++v23_i1) {
              float v28_data = __builtin_nontemporal_load(&glb_m1[(v25_lead + (v23_i1 * 16))]);
              r0[(v22_i0 + v23_i1)] = v28_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m2);
          #pragma unroll
          for (int32_t v31_i0 = 0; v31_i0 < 1; ++v31_i0) {
            int32_t v34_lead = v21_lead + (v31_i0 * 16);
            #pragma unroll
            for (int32_t v32_i1 = 0; v32_i1 < 16; ++v32_i1) {
              float v37_data = __builtin_nontemporal_load(&glb_m2[(v34_lead + (v32_i1 * 16))]);
              r1[(v31_i0 + v32_i1)] = v37_data;
            }
          }
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v40_data = r1[0];
          float v41_data = r1[1];
          float v42_data = r1[2];
          float v43_data = r1[3];
          float v44_data = r1[4];
          float v45_data = r1[5];
          float v46_data = r1[6];
          float v47_data = r1[7];
          float v48_data = r1[8];
          float v49_data = r1[9];
          float v50_data = r1[10];
          float v51_data = r1[11];
          float v52_data = r1[12];
          float v53_data = r1[13];
          float v54_data = r1[14];
          float v55_data = r1[15];
          tensorforge::transpose16x16b32(v40_data, v41_data, v42_data, v43_data, v44_data, v45_data, v46_data, v47_data, v48_data, v49_data, v50_data, v51_data, v52_data, v53_data, v54_data, v55_data);
          tensorforge::VectorT<float, 16> v56_acc{};
          float v57_data = r0[0];
          float v58_data = r0[1];
          float v59_data = r0[2];
          float v60_data = r0[3];
          float v61_data = r0[4];
          float v62_data = r0[5];
          float v63_data = r0[6];
          float v64_data = r0[7];
          float v65_data = r0[8];
          float v66_data = r0[9];
          float v67_data = r0[10];
          float v68_data = r0[11];
          float v69_data = r0[12];
          float v70_data = r0[13];
          float v71_data = r0[14];
          float v72_data = r0[15];
          tensorforge::VectorT<float, 16> v73_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v40_data, v57_data, v56_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v74_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v41_data, v58_data, v73_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v75_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v42_data, v59_data, v74_acc, 0, 0, 0);
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
          float v89_el = v88_acc[0];
          float v91_el = v88_acc[4];
          float v92_sw = tensorforge::swap<32>(v91_el);
          float v94_el = v88_acc[8];
          float v97_el = v88_acc[12];
          float v98_sw = tensorforge::swap<32>(v97_el);
          r2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v98_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v94_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v92_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v89_el, v89_el))))))));
          float v101_el = v88_acc[1];
          float v103_el = v88_acc[5];
          float v104_sw = tensorforge::swap<32>(v103_el);
          float v106_el = v88_acc[9];
          float v109_el = v88_acc[13];
          float v110_sw = tensorforge::swap<32>(v109_el);
          r2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v110_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v106_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v104_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v101_el, v101_el))))))));
          float v113_el = v88_acc[2];
          float v115_el = v88_acc[6];
          float v116_sw = tensorforge::swap<32>(v115_el);
          float v118_el = v88_acc[10];
          float v121_el = v88_acc[14];
          float v122_sw = tensorforge::swap<32>(v121_el);
          r2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v122_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v118_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v116_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v113_el, v113_el))))))));
          float v125_el = v88_acc[3];
          float v127_el = v88_acc[7];
          float v128_sw = tensorforge::swap<32>(v127_el);
          float v130_el = v88_acc[11];
          float v133_el = v88_acc[15];
          float v134_sw = tensorforge::swap<32>(v133_el);
          r2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v134_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v130_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v128_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v125_el, v125_el))))))));
          float v138_sw = tensorforge::swap<32>(v89_el);
          float v143_sw = tensorforge::swap<32>(v94_el);
          r2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v97_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v143_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v91_el, (tensorforge::dppUpdate<228, 1, 15, false>(v138_sw, v138_sw))))))));
          float v150_sw = tensorforge::swap<32>(v101_el);
          float v155_sw = tensorforge::swap<32>(v106_el);
          r2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v109_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v155_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v103_el, (tensorforge::dppUpdate<228, 1, 15, false>(v150_sw, v150_sw))))))));
          float v162_sw = tensorforge::swap<32>(v113_el);
          float v167_sw = tensorforge::swap<32>(v118_el);
          r2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v121_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v167_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v115_el, (tensorforge::dppUpdate<228, 1, 15, false>(v162_sw, v162_sw))))))));
          float v174_sw = tensorforge::swap<32>(v125_el);
          float v179_sw = tensorforge::swap<32>(v130_el);
          r2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v133_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v179_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v127_el, (tensorforge::dppUpdate<228, 1, 15, false>(v174_sw, v174_sw))))))));
          float v186_sw = tensorforge::swap<64>(v89_el);
          r2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v98_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v94_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v92_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v186_sw, v186_sw))))))));
          float v198_sw = tensorforge::swap<64>(v101_el);
          r2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v110_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v106_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v104_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v198_sw, v198_sw))))))));
          float v210_sw = tensorforge::swap<64>(v113_el);
          r2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v122_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v118_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v116_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v210_sw, v210_sw))))))));
          float v222_sw = tensorforge::swap<64>(v125_el);
          r2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v134_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v130_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v128_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v222_sw, v222_sw))))))));
          float v235_sw = tensorforge::swap<64>(v138_sw);
          r2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v97_el, (tensorforge::dppUpdate<228, 4, 15, false>(v143_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v91_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v235_sw, v235_sw))))))));
          float v247_sw = tensorforge::swap<64>(v150_sw);
          r2[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v109_el, (tensorforge::dppUpdate<228, 4, 15, false>(v155_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v103_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v247_sw, v247_sw))))))));
          float v259_sw = tensorforge::swap<64>(v162_sw);
          r2[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v121_el, (tensorforge::dppUpdate<228, 4, 15, false>(v167_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v115_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v259_sw, v259_sw))))))));
          float v271_sw = tensorforge::swap<64>(v174_sw);
          r2[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v133_el, (tensorforge::dppUpdate<228, 4, 15, false>(v179_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v127_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v271_sw, v271_sw))))))));
          // glb_m0 = store{r>g}(r2);
          #pragma unroll
          for (int32_t v281_i0 = 0; v281_i0 < 1; ++v281_i0) {
            int32_t v286_lead = v21_lead + (v281_i0 * 16);
            #pragma unroll
            for (int32_t v282_i1 = 0; v282_i1 < 16; ++v282_i1) {
              float v284_data = r2[(v281_i0 + v282_i1)];
              glb_m0[(v286_lead + (v282_i1 * 16))] = v284_data;
            }
          }
        }
      }
    }
  }
}

