// === base name ===
kernel_4759234f344be15b

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_4759234f344be15b = {{32, 8, 1}, 32, 32, 1, 8, 0, false, true, 2};
tensorforge::LaunchConfig launch_config_kernel_4759234f344be15b(size_t numElements0, size_t numElements1, void* streamPtr = nullptr);
void launcher_kernel_4759234f344be15b(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, size_t numElements0, size_t numElements1, unsigned * flags0 = nullptr, unsigned * flags1 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_4759234f344be15b(size_t numElements0, size_t numElements1, void* streamPtr) {
  (void)numElements0;
  (void)numElements1;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_4759234f344be15b, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (0 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_4759234f344be15b, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (0 * sizeof(float)));
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
  config.block[0] = 32;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 0 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_4759234f344be15b(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, size_t numElements0, size_t numElements1, unsigned * flags0, unsigned * flags1, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_4759234f344be15b(numElements0, numElements1, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_4759234f344be15b), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m4;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m5Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m5;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags1Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags1;
  hipLaunchKernelGGL(kernel_kernel_4759234f344be15b, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, m5Arg, m5_extraOffset, numElements0, numElements1, flags0Arg, flags1Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_4759234f344be15b(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m5, size_t m5_extraOffset, size_t numElements0, size_t numElements1, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags1) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 8 per block = block 32x8x1, 0 B shared, occupancy grid
    // operands:
    //   m0 16×16(16×16) {0..16}×{0..16} strided
    //   m1 16×16(16×16) {0..16}×{0..16} strided
    //   m2 16×16(16×16) {0..16}×{0..16} strided
    //   m3 16×16(16×16) {0..16}×{0..16} strided
    //   m4 16×16(16×16) {0..16}×{0..16} strided
    //   m5 16×16(16×16) {0..16}×{0..16} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    //   fence
    //   m3[i,j] = m4[i,k] × m5[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":0},{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"E","bbox":[[0,0],[16,16]],"name":"m3","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m4","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"F","bbox":[[0,0],[16,16]],"name":"m5","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"kind":"fence"},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      for (size_t v7_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v7_batchId0 < numElements0; v7_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v8_ahead1 = v7_batchId0 + (gridDim.x * blockDim.y);
        size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v7_batchId0 * 256 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v7_batchId0 * 256 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v7_batchId0 * 256 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v7_batchId0 * 256 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v7_batchId0 * 256 + 0 + m4_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m5[v7_batchId0 * 256 + 0 + m5_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v24_lead = threadIdx.x % 32;
          bool v25_g = v24_lead < 16;
          if (v25_g) {
            #pragma unroll
            for (int32_t v26_i1 = 0; v26_i1 < 16; ++v26_i1) {
              float v31_data = __builtin_nontemporal_load(&glb_m1[(v24_lead + (v26_i1 * 16))]);
              r0[v26_i1] = v31_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m2);
          if (v25_g) {
            #pragma unroll
            for (int32_t v34_i1 = 0; v34_i1 < 16; ++v34_i1) {
              float v39_data = __builtin_nontemporal_load(&glb_m2[(v24_lead + (v34_i1 * 16))]);
              r1[v34_i1] = v39_data;
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
          tensorforge::VectorT<float, 16> v75_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v42_data, v59_data, v58_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v76_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v43_data, v60_data, v75_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v77_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v44_data, v61_data, v76_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v78_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v45_data, v62_data, v77_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v79_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v46_data, v63_data, v78_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v80_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v47_data, v64_data, v79_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v81_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v48_data, v65_data, v80_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v82_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v49_data, v66_data, v81_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v83_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v50_data, v67_data, v82_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v84_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v51_data, v68_data, v83_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v85_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v52_data, v69_data, v84_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v86_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v53_data, v70_data, v85_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v87_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v54_data, v71_data, v86_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v88_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v55_data, v72_data, v87_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v89_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v56_data, v73_data, v88_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v90_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v57_data, v74_data, v89_acc, 1, 0, 0);
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
          if (v25_g) {
            #pragma unroll
            for (int32_t v283_i1 = 0; v283_i1 < 16; ++v283_i1) {
              float v285_data = r2[v283_i1];
              glb_m0[(v24_lead + (v283_i1 * 16))] = v285_data;
            }
          }
        }
      }
    }
    {
      __syncthreads();
      for (size_t v297_batchId0 = ((threadIdx.y + blockDim.y * (blockIdx.x)) + numElements0) % (gridDim.x * blockDim.y); v297_batchId0 < numElements1; v297_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v298_ahead1 = v297_batchId0 + (gridDim.x * blockDim.y);
        size_t v300_batchId1 = (v298_ahead1 < numElements1) ? v298_ahead1 : v297_batchId0;
        const bool allowed = flags1 == nullptr ? true : static_cast<bool>(flags1[v297_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v297_batchId0 * 256 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v297_batchId0 * 256 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v297_batchId0 * 256 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v297_batchId0 * 256 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v297_batchId0 * 256 + 0 + m4_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m5 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m5[v297_batchId0 * 256 + 0 + m5_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m4);
          int32_t v314_lead = threadIdx.x % 32;
          bool v315_g = v314_lead < 16;
          if (v315_g) {
            #pragma unroll
            for (int32_t v316_i1 = 0; v316_i1 < 16; ++v316_i1) {
              float v321_data = __builtin_nontemporal_load(&glb_m4[(v314_lead + (v316_i1 * 16))]);
              r0[v316_i1] = v321_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m5);
          if (v315_g) {
            #pragma unroll
            for (int32_t v324_i1 = 0; v324_i1 < 16; ++v324_i1) {
              float v329_data = __builtin_nontemporal_load(&glb_m5[(v314_lead + (v324_i1 * 16))]);
              r1[v324_i1] = v329_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m4););
          // wait(r1 = load{g>r}(glb_m5););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v332_data = r1[0];
          float v333_data = r1[1];
          float v334_data = r1[2];
          float v335_data = r1[3];
          float v336_data = r1[4];
          float v337_data = r1[5];
          float v338_data = r1[6];
          float v339_data = r1[7];
          float v340_data = r1[8];
          float v341_data = r1[9];
          float v342_data = r1[10];
          float v343_data = r1[11];
          float v344_data = r1[12];
          float v345_data = r1[13];
          float v346_data = r1[14];
          float v347_data = r1[15];
          tensorforge::transpose16x16b32(v332_data, v333_data, v334_data, v335_data, v336_data, v337_data, v338_data, v339_data, v340_data, v341_data, v342_data, v343_data, v344_data, v345_data, v346_data, v347_data);
          tensorforge::VectorT<float, 16> v348_acc{};
          float v349_data = r0[0];
          float v350_data = r0[1];
          float v351_data = r0[2];
          float v352_data = r0[3];
          float v353_data = r0[4];
          float v354_data = r0[5];
          float v355_data = r0[6];
          float v356_data = r0[7];
          float v357_data = r0[8];
          float v358_data = r0[9];
          float v359_data = r0[10];
          float v360_data = r0[11];
          float v361_data = r0[12];
          float v362_data = r0[13];
          float v363_data = r0[14];
          float v364_data = r0[15];
          tensorforge::VectorT<float, 16> v365_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v332_data, v349_data, v348_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v366_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v333_data, v350_data, v365_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v367_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v334_data, v351_data, v366_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v368_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v335_data, v352_data, v367_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v369_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v336_data, v353_data, v368_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v370_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v337_data, v354_data, v369_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v371_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v338_data, v355_data, v370_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v372_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v339_data, v356_data, v371_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v373_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v340_data, v357_data, v372_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v374_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v341_data, v358_data, v373_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v375_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v342_data, v359_data, v374_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v376_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v343_data, v360_data, v375_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v377_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v344_data, v361_data, v376_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v378_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v345_data, v362_data, v377_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v379_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v346_data, v363_data, v378_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v380_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v347_data, v364_data, v379_acc, 1, 0, 0);
          float v381_el = v380_acc[0];
          float v383_el = v380_acc[4];
          float v384_sw = tensorforge::swap<32>(v383_el);
          float v386_el = v380_acc[8];
          float v389_el = v380_acc[12];
          float v390_sw = tensorforge::swap<32>(v389_el);
          r2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v390_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v386_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v384_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v381_el, v381_el))))))));
          float v393_el = v380_acc[1];
          float v395_el = v380_acc[5];
          float v396_sw = tensorforge::swap<32>(v395_el);
          float v398_el = v380_acc[9];
          float v401_el = v380_acc[13];
          float v402_sw = tensorforge::swap<32>(v401_el);
          r2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v402_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v398_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v396_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v393_el, v393_el))))))));
          float v405_el = v380_acc[2];
          float v407_el = v380_acc[6];
          float v408_sw = tensorforge::swap<32>(v407_el);
          float v410_el = v380_acc[10];
          float v413_el = v380_acc[14];
          float v414_sw = tensorforge::swap<32>(v413_el);
          r2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v414_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v410_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v408_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v405_el, v405_el))))))));
          float v417_el = v380_acc[3];
          float v419_el = v380_acc[7];
          float v420_sw = tensorforge::swap<32>(v419_el);
          float v422_el = v380_acc[11];
          float v425_el = v380_acc[15];
          float v426_sw = tensorforge::swap<32>(v425_el);
          r2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v426_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v422_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v420_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v417_el, v417_el))))))));
          float v430_sw = tensorforge::swap<32>(v381_el);
          float v435_sw = tensorforge::swap<32>(v386_el);
          r2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v389_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v435_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v383_el, (tensorforge::dppUpdate<228, 1, 15, false>(v430_sw, v430_sw))))))));
          float v442_sw = tensorforge::swap<32>(v393_el);
          float v447_sw = tensorforge::swap<32>(v398_el);
          r2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v401_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v447_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v395_el, (tensorforge::dppUpdate<228, 1, 15, false>(v442_sw, v442_sw))))))));
          float v454_sw = tensorforge::swap<32>(v405_el);
          float v459_sw = tensorforge::swap<32>(v410_el);
          r2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v413_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v459_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v407_el, (tensorforge::dppUpdate<228, 1, 15, false>(v454_sw, v454_sw))))))));
          float v466_sw = tensorforge::swap<32>(v417_el);
          float v471_sw = tensorforge::swap<32>(v422_el);
          r2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v425_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v471_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v419_el, (tensorforge::dppUpdate<228, 1, 15, false>(v466_sw, v466_sw))))))));
          float v478_sw = tensorforge::swap<64>(v381_el);
          r2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v390_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v386_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v384_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v478_sw, v478_sw))))))));
          float v490_sw = tensorforge::swap<64>(v393_el);
          r2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v402_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v398_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v396_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v490_sw, v490_sw))))))));
          float v502_sw = tensorforge::swap<64>(v405_el);
          r2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v414_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v410_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v408_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v502_sw, v502_sw))))))));
          float v514_sw = tensorforge::swap<64>(v417_el);
          r2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v426_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v422_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v420_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v514_sw, v514_sw))))))));
          float v527_sw = tensorforge::swap<64>(v430_sw);
          r2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v389_el, (tensorforge::dppUpdate<228, 4, 15, false>(v435_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v383_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v527_sw, v527_sw))))))));
          float v539_sw = tensorforge::swap<64>(v442_sw);
          r2[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v401_el, (tensorforge::dppUpdate<228, 4, 15, false>(v447_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v395_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v539_sw, v539_sw))))))));
          float v551_sw = tensorforge::swap<64>(v454_sw);
          r2[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v413_el, (tensorforge::dppUpdate<228, 4, 15, false>(v459_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v407_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v551_sw, v551_sw))))))));
          float v563_sw = tensorforge::swap<64>(v466_sw);
          r2[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v425_el, (tensorforge::dppUpdate<228, 4, 15, false>(v471_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v419_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v563_sw, v563_sw))))))));
          // glb_m3 = store{r>g}(r2);
          if (v315_g) {
            #pragma unroll
            for (int32_t v573_i1 = 0; v573_i1 < 16; ++v573_i1) {
              float v575_data = r2[v573_i1];
              glb_m3[(v314_lead + (v573_i1 * 16))] = v575_data;
            }
          }
        }
      }
    }
  }
}

