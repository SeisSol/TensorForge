// === base name ===
kernel_526fc48f85c0f654

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_526fc48f85c0f654 = {{32, 8, 1}, 32, 32, 1, 8, 0, true, true, 2};
tensorforge::LaunchConfig launch_config_kernel_526fc48f85c0f654(size_t numElements0, size_t numElements1, void* streamPtr = nullptr);
void launcher_kernel_526fc48f85c0f654(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, size_t numElements1, unsigned * flags0 = nullptr, unsigned * flags1 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_526fc48f85c0f654(size_t numElements0, size_t numElements1, void* streamPtr) {
  (void)numElements0;
  (void)numElements1;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_526fc48f85c0f654, block.x * block.y * block.z, 0 * sizeof(float)));
        CHECK_ERR;
        if (blocksPerSM > 0) {
          gridsize = smCount * blocksPerSM;
        }
        else {
          gridsize = smCount;
        }
      }
      
  tensorforge::LaunchConfig config{};
  config.grid[0] = gridsize;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 32;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 0 * sizeof(float);
  config.cooperative = true;
  return config;
}
void launcher_kernel_526fc48f85c0f654(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, size_t numElements1, unsigned * flags0, unsigned * flags1, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_526fc48f85c0f654(numElements0, numElements1, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_526fc48f85c0f654), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m4;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags0;
  tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags1Arg = (tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace>)flags1;
  
    auto args = tensorforge::argsPtrs(m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, numElements0, numElements1, flags0Arg, flags1Arg);
    hipLaunchCooperativeKernel(kernel_kernel_526fc48f85c0f654, grid, block, args.data(), config.sharedMemBytes, stream);
  ;
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_526fc48f85c0f654(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, size_t numElements1, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags1) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes x 8 per block = block 32x8x1, 0 B shared, occupancy grid, cooperative
    // operands:
    //   m0 16×16(16×16) {0..16}×{0..16} strided
    //   m1 16×16(16×16) {0..16}×{0..16} strided
    //   m2 16×16(16×16) {0..16}×{0..16} strided
    //   m3 16×16(16×16) {0..16}×{0..16} strided
    //   m4 16×16(16×16) {0..16}×{0..16} strided
    // operations:
    //   m0[i,j] = m1[i,k] × m2[k,j]
    //   barrier
    //   m3[i,j] = m0[i,k] × m4[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,8,1],"cooperative":true,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":true,"mults_per_block":8,"shared_elements":0},{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"E","bbox":[[0,0],[16,16]],"name":"m3","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m4","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"kind":"barrier"},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      for (size_t v1_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v1_batchId0 < numElements0; v1_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v2_ahead1 = v1_batchId0 + (gridDim.x * blockDim.y);
        size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v1_batchId0 * 256 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v1_batchId0 * 256 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v1_batchId0 * 256 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v1_batchId0 * 256 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v1_batchId0 * 256 + 0 + m4_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v17_lead = threadIdx.x % 32;
          bool v18_g = v17_lead < 16;
          if (v18_g) {
            #pragma unroll
            for (int32_t v19_i1 = 0; v19_i1 < 16; ++v19_i1) {
              float v24_data = __builtin_nontemporal_load(&glb_m1[(v17_lead + (v19_i1 * 16))]);
              r0[v19_i1] = v24_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m2);
          if (v18_g) {
            #pragma unroll
            for (int32_t v27_i1 = 0; v27_i1 < 16; ++v27_i1) {
              float v32_data = __builtin_nontemporal_load(&glb_m2[(v17_lead + (v27_i1 * 16))]);
              r1[v27_i1] = v32_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v35_data = r1[0];
          float v36_data = r1[1];
          float v37_data = r1[2];
          float v38_data = r1[3];
          float v39_data = r1[4];
          float v40_data = r1[5];
          float v41_data = r1[6];
          float v42_data = r1[7];
          float v43_data = r1[8];
          float v44_data = r1[9];
          float v45_data = r1[10];
          float v46_data = r1[11];
          float v47_data = r1[12];
          float v48_data = r1[13];
          float v49_data = r1[14];
          float v50_data = r1[15];
          tensorforge::transpose16x16b32(v35_data, v36_data, v37_data, v38_data, v39_data, v40_data, v41_data, v42_data, v43_data, v44_data, v45_data, v46_data, v47_data, v48_data, v49_data, v50_data);
          tensorforge::VectorT<float, 16> v51_acc{};
          float v52_data = r0[0];
          float v53_data = r0[1];
          float v54_data = r0[2];
          float v55_data = r0[3];
          float v56_data = r0[4];
          float v57_data = r0[5];
          float v58_data = r0[6];
          float v59_data = r0[7];
          float v60_data = r0[8];
          float v61_data = r0[9];
          float v62_data = r0[10];
          float v63_data = r0[11];
          float v64_data = r0[12];
          float v65_data = r0[13];
          float v66_data = r0[14];
          float v67_data = r0[15];
          tensorforge::VectorT<float, 16> v68_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v35_data, v52_data, v51_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v69_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v36_data, v53_data, v68_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v70_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v37_data, v54_data, v69_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v71_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v38_data, v55_data, v70_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v72_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v39_data, v56_data, v71_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v73_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v40_data, v57_data, v72_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v74_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v41_data, v58_data, v73_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v75_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v42_data, v59_data, v74_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v76_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v43_data, v60_data, v75_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v77_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v44_data, v61_data, v76_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v78_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v45_data, v62_data, v77_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v79_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v46_data, v63_data, v78_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v80_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v47_data, v64_data, v79_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v81_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v48_data, v65_data, v80_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v82_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v49_data, v66_data, v81_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v83_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v50_data, v67_data, v82_acc, 1, 0, 0);
          float v84_el = v83_acc[0];
          float v86_el = v83_acc[4];
          float v87_sw = tensorforge::swap<32>(v86_el);
          float v89_el = v83_acc[8];
          float v92_el = v83_acc[12];
          float v93_sw = tensorforge::swap<32>(v92_el);
          r2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v93_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v89_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v87_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v84_el, v84_el))))))));
          float v96_el = v83_acc[1];
          float v98_el = v83_acc[5];
          float v99_sw = tensorforge::swap<32>(v98_el);
          float v101_el = v83_acc[9];
          float v104_el = v83_acc[13];
          float v105_sw = tensorforge::swap<32>(v104_el);
          r2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v105_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v101_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v99_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v96_el, v96_el))))))));
          float v108_el = v83_acc[2];
          float v110_el = v83_acc[6];
          float v111_sw = tensorforge::swap<32>(v110_el);
          float v113_el = v83_acc[10];
          float v116_el = v83_acc[14];
          float v117_sw = tensorforge::swap<32>(v116_el);
          r2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v117_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v113_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v111_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v108_el, v108_el))))))));
          float v120_el = v83_acc[3];
          float v122_el = v83_acc[7];
          float v123_sw = tensorforge::swap<32>(v122_el);
          float v125_el = v83_acc[11];
          float v128_el = v83_acc[15];
          float v129_sw = tensorforge::swap<32>(v128_el);
          r2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v129_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v125_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v123_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v120_el, v120_el))))))));
          float v133_sw = tensorforge::swap<32>(v84_el);
          float v138_sw = tensorforge::swap<32>(v89_el);
          r2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v92_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v138_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v86_el, (tensorforge::dppUpdate<228, 1, 15, false>(v133_sw, v133_sw))))))));
          float v145_sw = tensorforge::swap<32>(v96_el);
          float v150_sw = tensorforge::swap<32>(v101_el);
          r2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v104_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v150_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v98_el, (tensorforge::dppUpdate<228, 1, 15, false>(v145_sw, v145_sw))))))));
          float v157_sw = tensorforge::swap<32>(v108_el);
          float v162_sw = tensorforge::swap<32>(v113_el);
          r2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v116_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v162_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v110_el, (tensorforge::dppUpdate<228, 1, 15, false>(v157_sw, v157_sw))))))));
          float v169_sw = tensorforge::swap<32>(v120_el);
          float v174_sw = tensorforge::swap<32>(v125_el);
          r2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v128_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v174_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v122_el, (tensorforge::dppUpdate<228, 1, 15, false>(v169_sw, v169_sw))))))));
          float v181_sw = tensorforge::swap<64>(v84_el);
          r2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v93_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v89_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v87_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v181_sw, v181_sw))))))));
          float v193_sw = tensorforge::swap<64>(v96_el);
          r2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v105_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v101_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v99_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v193_sw, v193_sw))))))));
          float v205_sw = tensorforge::swap<64>(v108_el);
          r2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v117_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v113_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v111_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v205_sw, v205_sw))))))));
          float v217_sw = tensorforge::swap<64>(v120_el);
          r2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v129_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v125_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v123_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v217_sw, v217_sw))))))));
          float v230_sw = tensorforge::swap<64>(v133_sw);
          r2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v92_el, (tensorforge::dppUpdate<228, 4, 15, false>(v138_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v86_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v230_sw, v230_sw))))))));
          float v242_sw = tensorforge::swap<64>(v145_sw);
          r2[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v104_el, (tensorforge::dppUpdate<228, 4, 15, false>(v150_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v98_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v242_sw, v242_sw))))))));
          float v254_sw = tensorforge::swap<64>(v157_sw);
          r2[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v116_el, (tensorforge::dppUpdate<228, 4, 15, false>(v162_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v110_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v254_sw, v254_sw))))))));
          float v266_sw = tensorforge::swap<64>(v169_sw);
          r2[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v128_el, (tensorforge::dppUpdate<228, 4, 15, false>(v174_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v122_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v266_sw, v266_sw))))))));
          // glb_m0 = store{r>g}(r2);
          if (v18_g) {
            #pragma unroll
            for (int32_t v276_i1 = 0; v276_i1 < 16; ++v276_i1) {
              float v278_data = r2[v276_i1];
              glb_m0[(v17_lead + (v276_i1 * 16))] = v278_data;
            }
          }
        }
      }
    }
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements1 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements1 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      cooperative_groups::this_grid().sync();
      for (size_t v284_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v284_batchId0 < numElements1; v284_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v285_ahead1 = v284_batchId0 + (gridDim.x * blockDim.y);
        size_t v287_batchId1 = (v285_ahead1 < numElements1) ? v285_ahead1 : v284_batchId0;
        const bool allowed = flags1 == nullptr ? true : static_cast<bool>(flags1[v284_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v284_batchId0 * 256 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v284_batchId0 * 256 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v284_batchId0 * 256 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v284_batchId0 * 256 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v284_batchId0 * 256 + 0 + m4_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v300_lead = threadIdx.x % 32;
          bool v301_g = v300_lead < 16;
          if (v301_g) {
            #pragma unroll
            for (int32_t v302_i1 = 0; v302_i1 < 16; ++v302_i1) {
              float v307_data = __builtin_nontemporal_load(&glb_m0[(v300_lead + (v302_i1 * 16))]);
              r0[v302_i1] = v307_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m4);
          if (v301_g) {
            #pragma unroll
            for (int32_t v310_i1 = 0; v310_i1 < 16; ++v310_i1) {
              float v315_data = __builtin_nontemporal_load(&glb_m4[(v300_lead + (v310_i1 * 16))]);
              r1[v310_i1] = v315_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          // wait(r1 = load{g>r}(glb_m4););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v318_data = r1[0];
          float v319_data = r1[1];
          float v320_data = r1[2];
          float v321_data = r1[3];
          float v322_data = r1[4];
          float v323_data = r1[5];
          float v324_data = r1[6];
          float v325_data = r1[7];
          float v326_data = r1[8];
          float v327_data = r1[9];
          float v328_data = r1[10];
          float v329_data = r1[11];
          float v330_data = r1[12];
          float v331_data = r1[13];
          float v332_data = r1[14];
          float v333_data = r1[15];
          tensorforge::transpose16x16b32(v318_data, v319_data, v320_data, v321_data, v322_data, v323_data, v324_data, v325_data, v326_data, v327_data, v328_data, v329_data, v330_data, v331_data, v332_data, v333_data);
          tensorforge::VectorT<float, 16> v334_acc{};
          float v335_data = r0[0];
          float v336_data = r0[1];
          float v337_data = r0[2];
          float v338_data = r0[3];
          float v339_data = r0[4];
          float v340_data = r0[5];
          float v341_data = r0[6];
          float v342_data = r0[7];
          float v343_data = r0[8];
          float v344_data = r0[9];
          float v345_data = r0[10];
          float v346_data = r0[11];
          float v347_data = r0[12];
          float v348_data = r0[13];
          float v349_data = r0[14];
          float v350_data = r0[15];
          tensorforge::VectorT<float, 16> v351_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v318_data, v335_data, v334_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v352_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v319_data, v336_data, v351_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v353_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v320_data, v337_data, v352_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v354_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v321_data, v338_data, v353_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v355_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v322_data, v339_data, v354_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v356_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v323_data, v340_data, v355_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v357_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v324_data, v341_data, v356_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v358_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v325_data, v342_data, v357_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v359_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v326_data, v343_data, v358_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v360_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v327_data, v344_data, v359_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v361_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v328_data, v345_data, v360_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v362_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v329_data, v346_data, v361_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v363_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v330_data, v347_data, v362_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v364_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v331_data, v348_data, v363_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v365_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v332_data, v349_data, v364_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v366_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v333_data, v350_data, v365_acc, 1, 0, 0);
          float v367_el = v366_acc[0];
          float v369_el = v366_acc[4];
          float v370_sw = tensorforge::swap<32>(v369_el);
          float v372_el = v366_acc[8];
          float v375_el = v366_acc[12];
          float v376_sw = tensorforge::swap<32>(v375_el);
          r2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v376_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v372_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v370_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v367_el, v367_el))))))));
          float v379_el = v366_acc[1];
          float v381_el = v366_acc[5];
          float v382_sw = tensorforge::swap<32>(v381_el);
          float v384_el = v366_acc[9];
          float v387_el = v366_acc[13];
          float v388_sw = tensorforge::swap<32>(v387_el);
          r2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v388_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v384_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v382_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v379_el, v379_el))))))));
          float v391_el = v366_acc[2];
          float v393_el = v366_acc[6];
          float v394_sw = tensorforge::swap<32>(v393_el);
          float v396_el = v366_acc[10];
          float v399_el = v366_acc[14];
          float v400_sw = tensorforge::swap<32>(v399_el);
          r2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v400_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v396_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v394_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v391_el, v391_el))))))));
          float v403_el = v366_acc[3];
          float v405_el = v366_acc[7];
          float v406_sw = tensorforge::swap<32>(v405_el);
          float v408_el = v366_acc[11];
          float v411_el = v366_acc[15];
          float v412_sw = tensorforge::swap<32>(v411_el);
          r2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v412_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v408_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v406_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v403_el, v403_el))))))));
          float v416_sw = tensorforge::swap<32>(v367_el);
          float v421_sw = tensorforge::swap<32>(v372_el);
          r2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v375_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v421_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v369_el, (tensorforge::dppUpdate<228, 1, 15, false>(v416_sw, v416_sw))))))));
          float v428_sw = tensorforge::swap<32>(v379_el);
          float v433_sw = tensorforge::swap<32>(v384_el);
          r2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v387_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v433_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v381_el, (tensorforge::dppUpdate<228, 1, 15, false>(v428_sw, v428_sw))))))));
          float v440_sw = tensorforge::swap<32>(v391_el);
          float v445_sw = tensorforge::swap<32>(v396_el);
          r2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v399_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v445_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v393_el, (tensorforge::dppUpdate<228, 1, 15, false>(v440_sw, v440_sw))))))));
          float v452_sw = tensorforge::swap<32>(v403_el);
          float v457_sw = tensorforge::swap<32>(v408_el);
          r2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v411_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v457_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v405_el, (tensorforge::dppUpdate<228, 1, 15, false>(v452_sw, v452_sw))))))));
          float v464_sw = tensorforge::swap<64>(v367_el);
          r2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v376_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v372_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v370_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v464_sw, v464_sw))))))));
          float v476_sw = tensorforge::swap<64>(v379_el);
          r2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v388_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v384_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v382_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v476_sw, v476_sw))))))));
          float v488_sw = tensorforge::swap<64>(v391_el);
          r2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v400_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v396_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v394_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v488_sw, v488_sw))))))));
          float v500_sw = tensorforge::swap<64>(v403_el);
          r2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v412_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v408_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v406_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v500_sw, v500_sw))))))));
          float v513_sw = tensorforge::swap<64>(v416_sw);
          r2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v375_el, (tensorforge::dppUpdate<228, 4, 15, false>(v421_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v369_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v513_sw, v513_sw))))))));
          float v525_sw = tensorforge::swap<64>(v428_sw);
          r2[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v387_el, (tensorforge::dppUpdate<228, 4, 15, false>(v433_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v381_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v525_sw, v525_sw))))))));
          float v537_sw = tensorforge::swap<64>(v440_sw);
          r2[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v399_el, (tensorforge::dppUpdate<228, 4, 15, false>(v445_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v393_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v537_sw, v537_sw))))))));
          float v549_sw = tensorforge::swap<64>(v452_sw);
          r2[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v411_el, (tensorforge::dppUpdate<228, 4, 15, false>(v457_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v405_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v549_sw, v549_sw))))))));
          // glb_m3 = store{r>g}(r2);
          if (v301_g) {
            #pragma unroll
            for (int32_t v559_i1 = 0; v559_i1 < 16; ++v559_i1) {
              float v561_data = r2[v559_i1];
              glb_m3[(v300_lead + (v559_i1 * 16))] = v561_data;
            }
          }
        }
      }
    }
  }
}

