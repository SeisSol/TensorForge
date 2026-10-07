// === base name ===
kernel_4377403721eb7f2a

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_4377403721eb7f2a = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_4377403721eb7f2a(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_4377403721eb7f2a(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_4377403721eb7f2a(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_4377403721eb7f2a, block.x * block.y * block.z, 256 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (256 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_4377403721eb7f2a, block.x * block.y * block.z, 0));
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
void launcher_kernel_4377403721eb7f2a(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_4377403721eb7f2a(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_4377403721eb7f2a), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_4377403721eb7f2a, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_4377403721eb7f2a(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, size_t numElements0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
    // operands:
    //   m0 12×16(12×16) {0..12}×{0..16} strided
    //   m1 20×12(20×12) {0..20}×{0..12} strided
    //   m2 20×16(20×16) {0..20}×{0..16} strided
    // operations:
    //   m0[i,j] = m1[k,i] × m2[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,16]],"name":"m0","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[20,12]],"name":"m1","ordered":false,"parts":1,"shape":[20,12],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[20,16]],"name":"m2","ordered":false,"parts":1,"shape":[20,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[20,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[20,12]},{"addressing":"strided","bbox":[[0,0],[20,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,16]}],"permute":[[1,0],[0,1]],"target":[[-1,0],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v7_batchId0 * 192 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v7_batchId0 * 240 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v7_batchId0 * 320 + 0 + m2_extraOffset];
          float r0[20]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v22_lead = threadIdx.x % 16;
          bool v23_g = v22_lead < 12;
          #pragma unroll
          for (int32_t v19_i0 = 0; v19_i0 < 20; ++v19_i0) {
            if (v23_g) {
              float v28_data = __builtin_nontemporal_load(&glb_m1[(v19_i0 + (v22_lead * 20))]);
              r0[v19_i0] = v28_data;
            }
          }
          float r1[32]{};
          // r1 = load{g>r}(glb_m2);
          int32_t v33_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v34_i0 = 0; v34_i0 < 1; ++v34_i0) {
            int32_t v37_lead = v33_lead + (v34_i0 * 16);
            #pragma unroll
            for (int32_t v35_i1 = 0; v35_i1 < 16; ++v35_i1) {
              float v40_data = __builtin_nontemporal_load(&glb_m2[(v37_lead + (v35_i1 * 20))]);
              r1[(v34_i0 + (v35_i1 * 2))] = v40_data;
            }
          }
          if (v33_lead < 4) {
            int32_t v46_lead = v33_lead + 16_i32;
            #pragma unroll
            for (int32_t v44_i1 = 0; v44_i1 < 16; ++v44_i1) {
              float v49_data = __builtin_nontemporal_load(&glb_m2[(v46_lead + (v44_i1 * 20))]);
              r1[(1 + (v44_i1 * 2))] = v49_data;
            }
          }
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 16)] [(0, 20)]
          float v53_data = r1[0];
          float v54_data = r1[2];
          float v55_data = r1[4];
          float v56_data = r1[6];
          float v57_data = r1[8];
          float v58_data = r1[10];
          float v59_data = r1[12];
          float v60_data = r1[14];
          float v61_data = r1[16];
          float v62_data = r1[18];
          float v63_data = r1[20];
          float v64_data = r1[22];
          float v65_data = r1[24];
          float v66_data = r1[26];
          float v67_data = r1[28];
          float v68_data = r1[30];
          tensorforge::transpose16x16b32(v53_data, v54_data, v55_data, v56_data, v57_data, v58_data, v59_data, v60_data, v61_data, v62_data, v63_data, v64_data, v65_data, v66_data, v67_data, v68_data);
          float v69_data = r1[1];
          float v70_data = r1[3];
          float v71_data = r1[5];
          float v72_data = r1[7];
          float v73_data = r1[9];
          float v74_data = r1[11];
          float v75_data = r1[13];
          float v76_data = r1[15];
          float v77_data = r1[17];
          float v78_data = r1[19];
          float v79_data = r1[21];
          float v80_data = r1[23];
          float v81_data = r1[25];
          float v82_data = r1[27];
          float v83_data = r1[29];
          float v84_data = r1[31];
          tensorforge::transpose16x16b32(v69_data, v70_data, v71_data, v72_data, v73_data, v74_data, v75_data, v76_data, v77_data, v78_data, v79_data, v80_data, v81_data, v82_data, v83_data, v84_data);
          tensorforge::VectorT<float, 16> v85_acc{};
          float v86_data = r0[0];
          float v87_data = r0[1];
          float v88_data = r0[2];
          float v89_data = r0[3];
          float v90_data = r0[4];
          float v91_data = r0[5];
          float v92_data = r0[6];
          float v93_data = r0[7];
          float v94_data = r0[8];
          float v95_data = r0[9];
          float v96_data = r0[10];
          float v97_data = r0[11];
          float v98_data = r0[12];
          float v99_data = r0[13];
          float v100_data = r0[14];
          float v101_data = r0[15];
          tensorforge::VectorT<float, 16> v102_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v53_data, v86_data, v85_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v103_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v54_data, v87_data, v102_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v104_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v55_data, v88_data, v103_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v105_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v56_data, v89_data, v104_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v106_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v57_data, v90_data, v105_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v107_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v58_data, v91_data, v106_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v108_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v59_data, v92_data, v107_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v109_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v60_data, v93_data, v108_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v110_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v61_data, v94_data, v109_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v111_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v62_data, v95_data, v110_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v112_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v63_data, v96_data, v111_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v113_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v64_data, v97_data, v112_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v114_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v65_data, v98_data, v113_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v115_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v66_data, v99_data, v114_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v116_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v67_data, v100_data, v115_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v117_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v68_data, v101_data, v116_acc, 0, 0, 0);
          float v118_data = r0[16];
          float v119_data = r0[17];
          float v120_data = r0[18];
          float v121_data = r0[19];
          tensorforge::VectorT<float, 16> v123_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v69_data, v118_data, v117_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v124_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v70_data, v119_data, v123_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v125_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v71_data, v120_data, v124_acc, 0, 0, 0);
          tensorforge::VectorT<float, 16> v126_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v72_data, v121_data, v125_acc, 0, 0, 0);
          float v127_el = v126_acc[0];
          float v129_el = v126_acc[4];
          float v130_sw = tensorforge::swap<32>(v129_el);
          float v132_el = v126_acc[8];
          float v135_el = v126_acc[12];
          float v136_sw = tensorforge::swap<32>(v135_el);
          r2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v136_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v132_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v130_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v127_el, v127_el))))))));
          float v139_el = v126_acc[1];
          float v141_el = v126_acc[5];
          float v142_sw = tensorforge::swap<32>(v141_el);
          float v144_el = v126_acc[9];
          float v147_el = v126_acc[13];
          float v148_sw = tensorforge::swap<32>(v147_el);
          r2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v148_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v144_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v142_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v139_el, v139_el))))))));
          float v151_el = v126_acc[2];
          float v153_el = v126_acc[6];
          float v154_sw = tensorforge::swap<32>(v153_el);
          float v156_el = v126_acc[10];
          float v159_el = v126_acc[14];
          float v160_sw = tensorforge::swap<32>(v159_el);
          r2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v160_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v156_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v154_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v151_el, v151_el))))))));
          float v163_el = v126_acc[3];
          float v165_el = v126_acc[7];
          float v166_sw = tensorforge::swap<32>(v165_el);
          float v168_el = v126_acc[11];
          float v171_el = v126_acc[15];
          float v172_sw = tensorforge::swap<32>(v171_el);
          r2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v172_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v168_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v166_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v163_el, v163_el))))))));
          float v176_sw = tensorforge::swap<32>(v127_el);
          float v181_sw = tensorforge::swap<32>(v132_el);
          r2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v135_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v181_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v129_el, (tensorforge::dppUpdate<228, 1, 15, false>(v176_sw, v176_sw))))))));
          float v188_sw = tensorforge::swap<32>(v139_el);
          float v193_sw = tensorforge::swap<32>(v144_el);
          r2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v147_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v193_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v141_el, (tensorforge::dppUpdate<228, 1, 15, false>(v188_sw, v188_sw))))))));
          float v200_sw = tensorforge::swap<32>(v151_el);
          float v205_sw = tensorforge::swap<32>(v156_el);
          r2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v159_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v205_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v153_el, (tensorforge::dppUpdate<228, 1, 15, false>(v200_sw, v200_sw))))))));
          float v212_sw = tensorforge::swap<32>(v163_el);
          float v217_sw = tensorforge::swap<32>(v168_el);
          r2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v171_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v217_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v165_el, (tensorforge::dppUpdate<228, 1, 15, false>(v212_sw, v212_sw))))))));
          float v224_sw = tensorforge::swap<64>(v127_el);
          r2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v136_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v132_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v130_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v224_sw, v224_sw))))))));
          float v236_sw = tensorforge::swap<64>(v139_el);
          r2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v148_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v144_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v142_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v236_sw, v236_sw))))))));
          float v248_sw = tensorforge::swap<64>(v151_el);
          r2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v160_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v156_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v154_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v248_sw, v248_sw))))))));
          float v260_sw = tensorforge::swap<64>(v163_el);
          r2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v172_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v168_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v166_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v260_sw, v260_sw))))))));
          float v273_sw = tensorforge::swap<64>(v176_sw);
          r2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v135_el, (tensorforge::dppUpdate<228, 4, 15, false>(v181_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v129_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v273_sw, v273_sw))))))));
          float v285_sw = tensorforge::swap<64>(v188_sw);
          r2[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v147_el, (tensorforge::dppUpdate<228, 4, 15, false>(v193_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v141_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v285_sw, v285_sw))))))));
          float v297_sw = tensorforge::swap<64>(v200_sw);
          r2[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v159_el, (tensorforge::dppUpdate<228, 4, 15, false>(v205_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v153_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v297_sw, v297_sw))))))));
          float v309_sw = tensorforge::swap<64>(v212_sw);
          r2[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v171_el, (tensorforge::dppUpdate<228, 4, 15, false>(v217_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v165_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v309_sw, v309_sw))))))));
          // glb_m0 = store{r>g}(r2);
          if (v33_lead < 12) {
            #pragma unroll
            for (int32_t v320_i1 = 0; v320_i1 < 16; ++v320_i1) {
              float v322_data = r2[v320_i1];
              glb_m0[(v33_lead + (v320_i1 * 12))] = v322_data;
            }
          }
        }
      }
    }
  }
}

