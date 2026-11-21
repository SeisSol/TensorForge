// === base name ===
kernel_12180cdf5c0996e6

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_12180cdf5c0996e6 = {{32, 8, 1}, 32, 32, 1, 8, 0, true, true, 2};
tensorforge::LaunchConfig launch_config_kernel_12180cdf5c0996e6(size_t numElements0, size_t numElements1, void* streamPtr = nullptr);
void launcher_kernel_12180cdf5c0996e6(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, size_t numElements1, unsigned * flags0 = nullptr, unsigned * flags1 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_12180cdf5c0996e6(size_t numElements0, size_t numElements1, void* streamPtr) {
  (void)numElements0;
  (void)numElements1;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_12180cdf5c0996e6, block.x * block.y * block.z, 0 * sizeof(float)));
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
void launcher_kernel_12180cdf5c0996e6(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, size_t numElements1, unsigned * flags0, unsigned * flags1, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_12180cdf5c0996e6(numElements0, numElements1, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_12180cdf5c0996e6), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m2;
  tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3Arg = (tensorforge::SpacePtr<float, tensorforge::GlobalMemspace>)m3;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m4;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)flags0;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> flags1Arg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)flags1;
  
    auto args = tensorforge::argsPtrs(m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, m2_extraOffset, m3Arg, m3_extraOffset, m4Arg, m4_extraOffset, numElements0, numElements1, flags0Arg, flags1Arg);
    hipLaunchCooperativeKernel(kernel_kernel_12180cdf5c0996e6, grid, block, args.data(), config.sharedMemBytes, stream);
  ;
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_12180cdf5c0996e6(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, size_t numElements1, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> flags0, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> flags1) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: cooperative_grid=1
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
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,8,1],"cooperative":true,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":true,"mults_per_block":8,"shared_elements":0},{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"E","bbox":[[0,0],[16,16]],"name":"m3","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m4","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"kind":"barrier"},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      size_t v8_batchIdLane0 = threadIdx.y % 2;
      size_t v11_batchIdGroup0First = ((threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x) < numElements0) ? (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x) : 0u;
      int32_t v30_lead = threadIdx.x % 32;
      bool v36_g = v30_lead < 16;
      float r0[16]{};
      size_t v306_row = v11_batchIdGroup0First + v8_batchIdLane0;
      const bool peel_batchIdActive0 = v306_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v306_row]));
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const peel_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[(peel_batchIdActive0 ? v306_row : v11_batchIdGroup0First) * 256 + 0 + m1_extraOffset];
      // r0 = load{g>r}(peel_glb_m1);
      #pragma unroll
      for (int32_t v314_i1 = 0; v314_i1 < 16; ++v314_i1) {
        float v317_data = v36_g ? (__builtin_nontemporal_load(&peel_glb_m1[(v30_lead + (v314_i1 * 16))])) : (0.0f);
        r0[v314_i1] = v317_data;
      }
      for (size_t v12_batchIdGroup0 = (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x); v12_batchIdGroup0 < numElements0; v12_batchIdGroup0 += (gridDim.x * blockDim.y)) {
        size_t v303_groupNextAhead = v12_batchIdGroup0 + (gridDim.x * blockDim.y);
        size_t v305_groupNext = (v303_groupNextAhead < numElements0) ? v303_groupNextAhead : v12_batchIdGroup0;
        size_t v310_row = v305_groupNext + v8_batchIdLane0;
        const bool wrap_batchIdActive0 = v310_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v310_row]));
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const wrap_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[(wrap_batchIdActive0 ? v310_row : v305_groupNext) * 256 + 0 + m1_extraOffset];
        size_t v13_row = v12_batchIdGroup0 + v8_batchIdLane0;
        const bool batchIdActive0 = v13_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v13_row]));
        size_t v15_batchId0 = batchIdActive0 ? v13_row : v12_batchIdGroup0;
        size_t v16_ahead1 = v15_batchId0 + (gridDim.x * blockDim.y);
        size_t v18_batchId1 = (v16_ahead1 < numElements0) ? v16_ahead1 : v15_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v15_batchId0 * 256 + 0 + m0_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v15_batchId0 * 256 + 0 + m2_extraOffset];
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v15_batchId0 * 256 + 0 + m3_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v15_batchId0 * 256 + 0 + m4_extraOffset];
        float r1[16]{};
        // r1 = load{g>r}(glb_m2);
        #pragma unroll
        for (int32_t v40_i1 = 0; v40_i1 < 16; ++v40_i1) {
          float v46_data = v36_g ? (__builtin_nontemporal_load(&glb_m2[(v30_lead + (v40_i1 * 16))])) : (0.0f);
          r1[v40_i1] = v46_data;
        }
        float r2[16]{};
        // ir2 = +(r0 * r1)
        // [(0, 16), (0, 16)] [(0, 16)]
        float ir2[16]{};
        float v50_data = r1[0];
        float v51_data = r1[1];
        float v52_data = r1[2];
        float v53_data = r1[3];
        float v54_data = r1[4];
        float v55_data = r1[5];
        float v56_data = r1[6];
        float v57_data = r1[7];
        float v58_data = r1[8];
        float v59_data = r1[9];
        float v60_data = r1[10];
        float v61_data = r1[11];
        float v62_data = r1[12];
        float v63_data = r1[13];
        float v64_data = r1[14];
        float v65_data = r1[15];
        tensorforge::transpose16x16b32(v50_data, v51_data, v52_data, v53_data, v54_data, v55_data, v56_data, v57_data, v58_data, v59_data, v60_data, v61_data, v62_data, v63_data, v64_data, v65_data);
        tensorforge::VectorT<float, 16> v66_acc{};
        float v67_data = r0[0];
        float v68_data = r0[1];
        float v69_data = r0[2];
        float v70_data = r0[3];
        float v71_data = r0[4];
        float v72_data = r0[5];
        float v73_data = r0[6];
        float v74_data = r0[7];
        float v75_data = r0[8];
        float v76_data = r0[9];
        float v77_data = r0[10];
        float v78_data = r0[11];
        float v79_data = r0[12];
        float v80_data = r0[13];
        float v81_data = r0[14];
        float v82_data = r0[15];
        // r0 = load{g>r}(wrap_glb_m1);
        #pragma unroll
        for (int32_t v318_i1 = 0; v318_i1 < 16; ++v318_i1) {
          float v321_data = v36_g ? (__builtin_nontemporal_load(&wrap_glb_m1[(v30_lead + (v318_i1 * 16))])) : (0.0f);
          r0[v318_i1] = v321_data;
        }
        tensorforge::VectorT<float, 16> v83_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v50_data, v67_data, v66_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v84_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v51_data, v68_data, v83_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v85_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v52_data, v69_data, v84_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v86_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v53_data, v70_data, v85_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v87_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v54_data, v71_data, v86_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v88_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v55_data, v72_data, v87_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v89_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v56_data, v73_data, v88_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v90_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v57_data, v74_data, v89_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v91_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v58_data, v75_data, v90_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v92_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v59_data, v76_data, v91_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v93_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v60_data, v77_data, v92_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v94_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v61_data, v78_data, v93_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v95_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v62_data, v79_data, v94_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v96_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v63_data, v80_data, v95_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v97_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v64_data, v81_data, v96_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v98_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v65_data, v82_data, v97_acc, 1, 0, 0);
        float v99_el = v98_acc[0];
        float v101_el = v98_acc[4];
        float v102_sw = tensorforge::swap<32>(v101_el);
        float v104_el = v98_acc[8];
        float v107_el = v98_acc[12];
        float v108_sw = tensorforge::swap<32>(v107_el);
        ir2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v108_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v104_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v102_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v99_el, v99_el))))))));
        float v111_el = v98_acc[1];
        float v113_el = v98_acc[5];
        float v114_sw = tensorforge::swap<32>(v113_el);
        float v116_el = v98_acc[9];
        float v119_el = v98_acc[13];
        float v120_sw = tensorforge::swap<32>(v119_el);
        ir2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v120_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v116_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v114_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v111_el, v111_el))))))));
        float v123_el = v98_acc[2];
        float v125_el = v98_acc[6];
        float v126_sw = tensorforge::swap<32>(v125_el);
        float v128_el = v98_acc[10];
        float v131_el = v98_acc[14];
        float v132_sw = tensorforge::swap<32>(v131_el);
        ir2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v132_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v128_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v126_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v123_el, v123_el))))))));
        float v135_el = v98_acc[3];
        float v137_el = v98_acc[7];
        float v138_sw = tensorforge::swap<32>(v137_el);
        float v140_el = v98_acc[11];
        float v143_el = v98_acc[15];
        float v144_sw = tensorforge::swap<32>(v143_el);
        ir2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v144_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v140_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v138_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v135_el, v135_el))))))));
        float v148_sw = tensorforge::swap<32>(v99_el);
        float v153_sw = tensorforge::swap<32>(v104_el);
        ir2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v107_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v153_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v101_el, (tensorforge::dppUpdate<228, 1, 15, false>(v148_sw, v148_sw))))))));
        float v160_sw = tensorforge::swap<32>(v111_el);
        float v165_sw = tensorforge::swap<32>(v116_el);
        ir2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v119_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v165_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v113_el, (tensorforge::dppUpdate<228, 1, 15, false>(v160_sw, v160_sw))))))));
        float v172_sw = tensorforge::swap<32>(v123_el);
        float v177_sw = tensorforge::swap<32>(v128_el);
        ir2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v131_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v177_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v125_el, (tensorforge::dppUpdate<228, 1, 15, false>(v172_sw, v172_sw))))))));
        float v184_sw = tensorforge::swap<32>(v135_el);
        float v189_sw = tensorforge::swap<32>(v140_el);
        ir2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v143_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v189_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v137_el, (tensorforge::dppUpdate<228, 1, 15, false>(v184_sw, v184_sw))))))));
        float v196_sw = tensorforge::swap<64>(v99_el);
        ir2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v108_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v104_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v102_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v196_sw, v196_sw))))))));
        float v208_sw = tensorforge::swap<64>(v111_el);
        ir2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v120_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v116_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v114_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v208_sw, v208_sw))))))));
        float v220_sw = tensorforge::swap<64>(v123_el);
        ir2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v132_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v128_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v126_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v220_sw, v220_sw))))))));
        float v232_sw = tensorforge::swap<64>(v135_el);
        ir2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v144_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v140_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v138_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v232_sw, v232_sw))))))));
        float v245_sw = tensorforge::swap<64>(v148_sw);
        ir2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v107_el, (tensorforge::dppUpdate<228, 4, 15, false>(v153_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v101_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v245_sw, v245_sw))))))));
        float v257_sw = tensorforge::swap<64>(v160_sw);
        ir2[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v119_el, (tensorforge::dppUpdate<228, 4, 15, false>(v165_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v113_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v257_sw, v257_sw))))))));
        float v269_sw = tensorforge::swap<64>(v172_sw);
        ir2[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v131_el, (tensorforge::dppUpdate<228, 4, 15, false>(v177_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v125_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v269_sw, v269_sw))))))));
        float v281_sw = tensorforge::swap<64>(v184_sw);
        ir2[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v143_el, (tensorforge::dppUpdate<228, 4, 15, false>(v189_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v137_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v281_sw, v281_sw))))))));
        // r2 = ir2
        // glb_m0 = store{r>g}(r2);
        if (v36_g) {
          #pragma unroll
          for (int32_t v296_i1 = 0; v296_i1 < 16; ++v296_i1) {
            float v298_data = ir2[v296_i1];
            if (batchIdActive0) {
              glb_m0[(v30_lead + (v296_i1 * 16))] = v298_data;
            }
          }
        }
      }
    }
    {
      cooperative_groups::this_grid().sync();
      size_t v652_batchIdLane1 = threadIdx.y % 2;
      size_t v655_batchIdGroup1First = ((threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x) < numElements1) ? (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x) : 0u;
      int32_t v674_lead = threadIdx.x % 32;
      bool v680_g = v674_lead < 16;
      float r0[16]{};
      size_t v950_row = v655_batchIdGroup1First + v652_batchIdLane1;
      const bool peel_batchIdActive1 = v950_row < numElements1 && (flags1 == nullptr || static_cast<bool>(flags1[v950_row]));
      tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const peel_glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[(peel_batchIdActive1 ? v950_row : v655_batchIdGroup1First) * 256 + 0 + m0_extraOffset];
      // r0 = load{g>r}(peel_glb_m0);
      #pragma unroll
      for (int32_t v958_i1 = 0; v958_i1 < 16; ++v958_i1) {
        float v961_data = v680_g ? (__builtin_nontemporal_load(&peel_glb_m0[(v674_lead + (v958_i1 * 16))])) : (0.0f);
        r0[v958_i1] = v961_data;
      }
      for (size_t v656_batchIdGroup1 = (threadIdx.y - threadIdx.y % 2) + blockDim.y * (blockIdx.x); v656_batchIdGroup1 < numElements1; v656_batchIdGroup1 += (gridDim.x * blockDim.y)) {
        size_t v947_groupNextAhead = v656_batchIdGroup1 + (gridDim.x * blockDim.y);
        size_t v949_groupNext = (v947_groupNextAhead < numElements1) ? v947_groupNextAhead : v656_batchIdGroup1;
        size_t v954_row = v949_groupNext + v652_batchIdLane1;
        const bool wrap_batchIdActive1 = v954_row < numElements1 && (flags1 == nullptr || static_cast<bool>(flags1[v954_row]));
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const wrap_glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[(wrap_batchIdActive1 ? v954_row : v949_groupNext) * 256 + 0 + m0_extraOffset];
        size_t v657_row = v656_batchIdGroup1 + v652_batchIdLane1;
        const bool batchIdActive1 = v657_row < numElements1 && (flags1 == nullptr || static_cast<bool>(flags1[v657_row]));
        size_t v659_batchId0 = batchIdActive1 ? v657_row : v656_batchIdGroup1;
        size_t v660_ahead1 = v659_batchId0 + (gridDim.x * blockDim.y);
        size_t v662_batchId1 = (v660_ahead1 < numElements1) ? v660_ahead1 : v659_batchId0;
        __builtin_amdgcn_sched_barrier(0);
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v659_batchId0 * 256 + 0 + m1_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v659_batchId0 * 256 + 0 + m2_extraOffset];
        tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v659_batchId0 * 256 + 0 + m3_extraOffset];
        tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v659_batchId0 * 256 + 0 + m4_extraOffset];
        float r1[16]{};
        // r1 = load{g>r}(glb_m4);
        #pragma unroll
        for (int32_t v684_i1 = 0; v684_i1 < 16; ++v684_i1) {
          float v690_data = v680_g ? (__builtin_nontemporal_load(&glb_m4[(v674_lead + (v684_i1 * 16))])) : (0.0f);
          r1[v684_i1] = v690_data;
        }
        float r2[16]{};
        // ir2 = +(r0 * r1)
        // [(0, 16), (0, 16)] [(0, 16)]
        float ir2[16]{};
        float v694_data = r1[0];
        float v695_data = r1[1];
        float v696_data = r1[2];
        float v697_data = r1[3];
        float v698_data = r1[4];
        float v699_data = r1[5];
        float v700_data = r1[6];
        float v701_data = r1[7];
        float v702_data = r1[8];
        float v703_data = r1[9];
        float v704_data = r1[10];
        float v705_data = r1[11];
        float v706_data = r1[12];
        float v707_data = r1[13];
        float v708_data = r1[14];
        float v709_data = r1[15];
        tensorforge::transpose16x16b32(v694_data, v695_data, v696_data, v697_data, v698_data, v699_data, v700_data, v701_data, v702_data, v703_data, v704_data, v705_data, v706_data, v707_data, v708_data, v709_data);
        tensorforge::VectorT<float, 16> v710_acc{};
        float v711_data = r0[0];
        float v712_data = r0[1];
        float v713_data = r0[2];
        float v714_data = r0[3];
        float v715_data = r0[4];
        float v716_data = r0[5];
        float v717_data = r0[6];
        float v718_data = r0[7];
        float v719_data = r0[8];
        float v720_data = r0[9];
        float v721_data = r0[10];
        float v722_data = r0[11];
        float v723_data = r0[12];
        float v724_data = r0[13];
        float v725_data = r0[14];
        float v726_data = r0[15];
        // r0 = load{g>r}(wrap_glb_m0);
        #pragma unroll
        for (int32_t v962_i1 = 0; v962_i1 < 16; ++v962_i1) {
          float v965_data = v680_g ? (__builtin_nontemporal_load(&wrap_glb_m0[(v674_lead + (v962_i1 * 16))])) : (0.0f);
          r0[v962_i1] = v965_data;
        }
        tensorforge::VectorT<float, 16> v727_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v694_data, v711_data, v710_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v728_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v695_data, v712_data, v727_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v729_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v696_data, v713_data, v728_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v730_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v697_data, v714_data, v729_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v731_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v698_data, v715_data, v730_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v732_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v699_data, v716_data, v731_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v733_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v700_data, v717_data, v732_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v734_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v701_data, v718_data, v733_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v735_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v702_data, v719_data, v734_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v736_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v703_data, v720_data, v735_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v737_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v704_data, v721_data, v736_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v738_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v705_data, v722_data, v737_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v739_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v706_data, v723_data, v738_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v740_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v707_data, v724_data, v739_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v741_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v708_data, v725_data, v740_acc, 1, 0, 0);
        tensorforge::VectorT<float, 16> v742_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v709_data, v726_data, v741_acc, 1, 0, 0);
        float v743_el = v742_acc[0];
        float v745_el = v742_acc[4];
        float v746_sw = tensorforge::swap<32>(v745_el);
        float v748_el = v742_acc[8];
        float v751_el = v742_acc[12];
        float v752_sw = tensorforge::swap<32>(v751_el);
        ir2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v752_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v748_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v746_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v743_el, v743_el))))))));
        float v755_el = v742_acc[1];
        float v757_el = v742_acc[5];
        float v758_sw = tensorforge::swap<32>(v757_el);
        float v760_el = v742_acc[9];
        float v763_el = v742_acc[13];
        float v764_sw = tensorforge::swap<32>(v763_el);
        ir2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v764_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v760_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v758_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v755_el, v755_el))))))));
        float v767_el = v742_acc[2];
        float v769_el = v742_acc[6];
        float v770_sw = tensorforge::swap<32>(v769_el);
        float v772_el = v742_acc[10];
        float v775_el = v742_acc[14];
        float v776_sw = tensorforge::swap<32>(v775_el);
        ir2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v776_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v772_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v770_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v767_el, v767_el))))))));
        float v779_el = v742_acc[3];
        float v781_el = v742_acc[7];
        float v782_sw = tensorforge::swap<32>(v781_el);
        float v784_el = v742_acc[11];
        float v787_el = v742_acc[15];
        float v788_sw = tensorforge::swap<32>(v787_el);
        ir2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v788_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v784_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v782_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v779_el, v779_el))))))));
        float v792_sw = tensorforge::swap<32>(v743_el);
        float v797_sw = tensorforge::swap<32>(v748_el);
        ir2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v751_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v797_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v745_el, (tensorforge::dppUpdate<228, 1, 15, false>(v792_sw, v792_sw))))))));
        float v804_sw = tensorforge::swap<32>(v755_el);
        float v809_sw = tensorforge::swap<32>(v760_el);
        ir2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v763_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v809_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v757_el, (tensorforge::dppUpdate<228, 1, 15, false>(v804_sw, v804_sw))))))));
        float v816_sw = tensorforge::swap<32>(v767_el);
        float v821_sw = tensorforge::swap<32>(v772_el);
        ir2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v775_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v821_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v769_el, (tensorforge::dppUpdate<228, 1, 15, false>(v816_sw, v816_sw))))))));
        float v828_sw = tensorforge::swap<32>(v779_el);
        float v833_sw = tensorforge::swap<32>(v784_el);
        ir2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v787_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v833_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v781_el, (tensorforge::dppUpdate<228, 1, 15, false>(v828_sw, v828_sw))))))));
        float v840_sw = tensorforge::swap<64>(v743_el);
        ir2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v752_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v748_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v746_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v840_sw, v840_sw))))))));
        float v852_sw = tensorforge::swap<64>(v755_el);
        ir2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v764_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v760_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v758_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v852_sw, v852_sw))))))));
        float v864_sw = tensorforge::swap<64>(v767_el);
        ir2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v776_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v772_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v770_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v864_sw, v864_sw))))))));
        float v876_sw = tensorforge::swap<64>(v779_el);
        ir2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v788_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v784_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v782_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v876_sw, v876_sw))))))));
        float v889_sw = tensorforge::swap<64>(v792_sw);
        ir2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v751_el, (tensorforge::dppUpdate<228, 4, 15, false>(v797_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v745_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v889_sw, v889_sw))))))));
        float v901_sw = tensorforge::swap<64>(v804_sw);
        ir2[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v763_el, (tensorforge::dppUpdate<228, 4, 15, false>(v809_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v757_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v901_sw, v901_sw))))))));
        float v913_sw = tensorforge::swap<64>(v816_sw);
        ir2[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v775_el, (tensorforge::dppUpdate<228, 4, 15, false>(v821_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v769_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v913_sw, v913_sw))))))));
        float v925_sw = tensorforge::swap<64>(v828_sw);
        ir2[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v787_el, (tensorforge::dppUpdate<228, 4, 15, false>(v833_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v781_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v925_sw, v925_sw))))))));
        // r2 = ir2
        // glb_m3 = store{r>g}(r2);
        if (v680_g) {
          #pragma unroll
          for (int32_t v940_i1 = 0; v940_i1 < 16; ++v940_i1) {
            float v942_data = ir2[v940_i1];
            if (batchIdActive1) {
              glb_m3[(v674_lead + (v940_i1 * 16))] = v942_data;
            }
          }
        }
      }
    }
  }
}

