// === base name ===
kernel_eddcd6cf598b9814

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_eddcd6cf598b9814 = {{32, 8, 1}, 32, 32, 1, 8, 0, true, true, 2};
tensorforge::LaunchConfig launch_config_kernel_eddcd6cf598b9814(size_t numElements0, size_t numElements1, void* streamPtr = nullptr);
void launcher_kernel_eddcd6cf598b9814(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, size_t numElements1, unsigned * flags0 = nullptr, unsigned * flags1 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_eddcd6cf598b9814(size_t numElements0, size_t numElements1, void* streamPtr) {
  (void)numElements0;
  (void)numElements1;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_eddcd6cf598b9814, block.x * block.y * block.z, 0 * sizeof(float)));
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
void launcher_kernel_eddcd6cf598b9814(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, size_t numElements1, unsigned * flags0, unsigned * flags1, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_eddcd6cf598b9814(numElements0, numElements1, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_eddcd6cf598b9814), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
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
    hipLaunchCooperativeKernel(kernel_kernel_eddcd6cf598b9814, grid, block, args.data(), config.sharedMemBytes, stream);
  ;
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_eddcd6cf598b9814(tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m2, size_t m2_extraOffset, tensorforge::SpacePtr<float, tensorforge::GlobalMemspace> m3, size_t m3_extraOffset, tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m4, size_t m4_extraOffset, size_t numElements0, size_t numElements1, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags0, tensorforge::SpacePtr<unsigned, tensorforge::GlobalMemspace> flags1) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
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
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,8,1],"cooperative":true,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":true,"mults_per_block":8,"shared_elements":0},{"barrier":false,"mults_per_block":8,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"E","bbox":[[0,0],[16,16]],"name":"m3","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m4","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"kind":"barrier"},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
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
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v23_lead = threadIdx.x % 32;
          bool v24_g = v23_lead < 16;
          if (v24_g) {
            #pragma unroll
            for (int32_t v25_i1 = 0; v25_i1 < 16; ++v25_i1) {
              float v30_data = __builtin_nontemporal_load(&glb_m1[(v23_lead + (v25_i1 * 16))]);
              r0[v25_i1] = v30_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m2);
          if (v24_g) {
            #pragma unroll
            for (int32_t v33_i1 = 0; v33_i1 < 16; ++v33_i1) {
              float v38_data = __builtin_nontemporal_load(&glb_m2[(v23_lead + (v33_i1 * 16))]);
              r1[v33_i1] = v38_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v41_data = r1[0];
          float v42_data = r1[1];
          float v43_data = r1[2];
          float v44_data = r1[3];
          float v45_data = r1[4];
          float v46_data = r1[5];
          float v47_data = r1[6];
          float v48_data = r1[7];
          float v49_data = r1[8];
          float v50_data = r1[9];
          float v51_data = r1[10];
          float v52_data = r1[11];
          float v53_data = r1[12];
          float v54_data = r1[13];
          float v55_data = r1[14];
          float v56_data = r1[15];
          tensorforge::transpose16x16b32(v41_data, v42_data, v43_data, v44_data, v45_data, v46_data, v47_data, v48_data, v49_data, v50_data, v51_data, v52_data, v53_data, v54_data, v55_data, v56_data);
          tensorforge::VectorT<float, 16> v57_acc{};
          float v58_data = r0[0];
          float v59_data = r0[1];
          float v60_data = r0[2];
          float v61_data = r0[3];
          float v62_data = r0[4];
          float v63_data = r0[5];
          float v64_data = r0[6];
          float v65_data = r0[7];
          float v66_data = r0[8];
          float v67_data = r0[9];
          float v68_data = r0[10];
          float v69_data = r0[11];
          float v70_data = r0[12];
          float v71_data = r0[13];
          float v72_data = r0[14];
          float v73_data = r0[15];
          tensorforge::VectorT<float, 16> v74_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v41_data, v58_data, v57_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v75_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v42_data, v59_data, v74_acc, 1, 0, 0);
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
          float v90_el = v89_acc[0];
          float v92_el = v89_acc[4];
          float v93_sw = tensorforge::swap<32>(v92_el);
          float v95_el = v89_acc[8];
          float v98_el = v89_acc[12];
          float v99_sw = tensorforge::swap<32>(v98_el);
          r2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v99_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v95_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v93_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v90_el, v90_el))))))));
          float v102_el = v89_acc[1];
          float v104_el = v89_acc[5];
          float v105_sw = tensorforge::swap<32>(v104_el);
          float v107_el = v89_acc[9];
          float v110_el = v89_acc[13];
          float v111_sw = tensorforge::swap<32>(v110_el);
          r2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v111_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v107_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v105_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v102_el, v102_el))))))));
          float v114_el = v89_acc[2];
          float v116_el = v89_acc[6];
          float v117_sw = tensorforge::swap<32>(v116_el);
          float v119_el = v89_acc[10];
          float v122_el = v89_acc[14];
          float v123_sw = tensorforge::swap<32>(v122_el);
          r2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v123_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v119_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v117_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v114_el, v114_el))))))));
          float v126_el = v89_acc[3];
          float v128_el = v89_acc[7];
          float v129_sw = tensorforge::swap<32>(v128_el);
          float v131_el = v89_acc[11];
          float v134_el = v89_acc[15];
          float v135_sw = tensorforge::swap<32>(v134_el);
          r2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v135_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v131_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v129_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v126_el, v126_el))))))));
          float v139_sw = tensorforge::swap<32>(v90_el);
          float v144_sw = tensorforge::swap<32>(v95_el);
          r2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v98_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v144_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v92_el, (tensorforge::dppUpdate<228, 1, 15, false>(v139_sw, v139_sw))))))));
          float v151_sw = tensorforge::swap<32>(v102_el);
          float v156_sw = tensorforge::swap<32>(v107_el);
          r2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v110_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v156_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v104_el, (tensorforge::dppUpdate<228, 1, 15, false>(v151_sw, v151_sw))))))));
          float v163_sw = tensorforge::swap<32>(v114_el);
          float v168_sw = tensorforge::swap<32>(v119_el);
          r2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v122_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v168_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v116_el, (tensorforge::dppUpdate<228, 1, 15, false>(v163_sw, v163_sw))))))));
          float v175_sw = tensorforge::swap<32>(v126_el);
          float v180_sw = tensorforge::swap<32>(v131_el);
          r2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v134_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v180_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v128_el, (tensorforge::dppUpdate<228, 1, 15, false>(v175_sw, v175_sw))))))));
          float v187_sw = tensorforge::swap<64>(v90_el);
          r2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v99_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v95_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v93_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v187_sw, v187_sw))))))));
          float v199_sw = tensorforge::swap<64>(v102_el);
          r2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v111_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v107_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v105_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v199_sw, v199_sw))))))));
          float v211_sw = tensorforge::swap<64>(v114_el);
          r2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v123_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v119_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v117_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v211_sw, v211_sw))))))));
          float v223_sw = tensorforge::swap<64>(v126_el);
          r2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v135_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v131_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v129_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v223_sw, v223_sw))))))));
          float v236_sw = tensorforge::swap<64>(v139_sw);
          r2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v98_el, (tensorforge::dppUpdate<228, 4, 15, false>(v144_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v92_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v236_sw, v236_sw))))))));
          float v248_sw = tensorforge::swap<64>(v151_sw);
          r2[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v110_el, (tensorforge::dppUpdate<228, 4, 15, false>(v156_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v104_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v248_sw, v248_sw))))))));
          float v260_sw = tensorforge::swap<64>(v163_sw);
          r2[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v122_el, (tensorforge::dppUpdate<228, 4, 15, false>(v168_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v116_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v260_sw, v260_sw))))))));
          float v272_sw = tensorforge::swap<64>(v175_sw);
          r2[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v134_el, (tensorforge::dppUpdate<228, 4, 15, false>(v180_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v128_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v272_sw, v272_sw))))))));
          // glb_m0 = store{r>g}(r2);
          if (v24_g) {
            #pragma unroll
            for (int32_t v282_i1 = 0; v282_i1 < 16; ++v282_i1) {
              float v284_data = r2[v282_i1];
              glb_m0[(v23_lead + (v282_i1 * 16))] = v284_data;
            }
          }
        }
      }
    }
    {
      cooperative_groups::this_grid().sync();
      for (size_t v296_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v296_batchId0 < numElements1; v296_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v297_ahead1 = v296_batchId0 + (gridDim.x * blockDim.y);
        size_t v299_batchId1 = (v297_ahead1 < numElements1) ? v297_ahead1 : v296_batchId0;
        const bool allowed = flags1 == nullptr ? true : static_cast<bool>(flags1[v296_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v296_batchId0 * 256 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[v296_batchId0 * 256 + 0 + m1_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v296_batchId0 * 256 + 0 + m2_extraOffset];
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[v296_batchId0 * 256 + 0 + m3_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[v296_batchId0 * 256 + 0 + m4_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v312_lead = threadIdx.x % 32;
          bool v313_g = v312_lead < 16;
          if (v313_g) {
            #pragma unroll
            for (int32_t v314_i1 = 0; v314_i1 < 16; ++v314_i1) {
              float v319_data = __builtin_nontemporal_load(&glb_m0[(v312_lead + (v314_i1 * 16))]);
              r0[v314_i1] = v319_data;
            }
          }
          float r1[16]{};
          // r1 = load{g>r}(glb_m4);
          if (v313_g) {
            #pragma unroll
            for (int32_t v322_i1 = 0; v322_i1 < 16; ++v322_i1) {
              float v327_data = __builtin_nontemporal_load(&glb_m4[(v312_lead + (v322_i1 * 16))]);
              r1[v322_i1] = v327_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          // wait(r1 = load{g>r}(glb_m4););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v330_data = r1[0];
          float v331_data = r1[1];
          float v332_data = r1[2];
          float v333_data = r1[3];
          float v334_data = r1[4];
          float v335_data = r1[5];
          float v336_data = r1[6];
          float v337_data = r1[7];
          float v338_data = r1[8];
          float v339_data = r1[9];
          float v340_data = r1[10];
          float v341_data = r1[11];
          float v342_data = r1[12];
          float v343_data = r1[13];
          float v344_data = r1[14];
          float v345_data = r1[15];
          tensorforge::transpose16x16b32(v330_data, v331_data, v332_data, v333_data, v334_data, v335_data, v336_data, v337_data, v338_data, v339_data, v340_data, v341_data, v342_data, v343_data, v344_data, v345_data);
          tensorforge::VectorT<float, 16> v346_acc{};
          float v347_data = r0[0];
          float v348_data = r0[1];
          float v349_data = r0[2];
          float v350_data = r0[3];
          float v351_data = r0[4];
          float v352_data = r0[5];
          float v353_data = r0[6];
          float v354_data = r0[7];
          float v355_data = r0[8];
          float v356_data = r0[9];
          float v357_data = r0[10];
          float v358_data = r0[11];
          float v359_data = r0[12];
          float v360_data = r0[13];
          float v361_data = r0[14];
          float v362_data = r0[15];
          tensorforge::VectorT<float, 16> v363_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v330_data, v347_data, v346_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v364_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v331_data, v348_data, v363_acc, 1, 0, 0);
          tensorforge::VectorT<float, 16> v365_acc = __builtin_amdgcn_mfma_f32_16x16x1f32(v332_data, v349_data, v364_acc, 1, 0, 0);
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
          float v379_el = v378_acc[0];
          float v381_el = v378_acc[4];
          float v382_sw = tensorforge::swap<32>(v381_el);
          float v384_el = v378_acc[8];
          float v387_el = v378_acc[12];
          float v388_sw = tensorforge::swap<32>(v387_el);
          r2[0] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v388_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v384_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v382_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v379_el, v379_el))))))));
          float v391_el = v378_acc[1];
          float v393_el = v378_acc[5];
          float v394_sw = tensorforge::swap<32>(v393_el);
          float v396_el = v378_acc[9];
          float v399_el = v378_acc[13];
          float v400_sw = tensorforge::swap<32>(v399_el);
          r2[1] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v400_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v396_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v394_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v391_el, v391_el))))))));
          float v403_el = v378_acc[2];
          float v405_el = v378_acc[6];
          float v406_sw = tensorforge::swap<32>(v405_el);
          float v408_el = v378_acc[10];
          float v411_el = v378_acc[14];
          float v412_sw = tensorforge::swap<32>(v411_el);
          r2[2] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v412_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v408_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v406_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v403_el, v403_el))))))));
          float v415_el = v378_acc[3];
          float v417_el = v378_acc[7];
          float v418_sw = tensorforge::swap<32>(v417_el);
          float v420_el = v378_acc[11];
          float v423_el = v378_acc[15];
          float v424_sw = tensorforge::swap<32>(v423_el);
          r2[3] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v424_sw)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v420_el)), (tensorforge::dppUpdate<228, 2, 15, false>(v418_sw, (tensorforge::dppUpdate<228, 1, 15, false>(v415_el, v415_el))))))));
          float v428_sw = tensorforge::swap<32>(v379_el);
          float v433_sw = tensorforge::swap<32>(v384_el);
          r2[4] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v387_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v433_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v381_el, (tensorforge::dppUpdate<228, 1, 15, false>(v428_sw, v428_sw))))))));
          float v440_sw = tensorforge::swap<32>(v391_el);
          float v445_sw = tensorforge::swap<32>(v396_el);
          r2[5] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v399_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v445_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v393_el, (tensorforge::dppUpdate<228, 1, 15, false>(v440_sw, v440_sw))))))));
          float v452_sw = tensorforge::swap<32>(v403_el);
          float v457_sw = tensorforge::swap<32>(v408_el);
          r2[6] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v411_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v457_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v405_el, (tensorforge::dppUpdate<228, 1, 15, false>(v452_sw, v452_sw))))))));
          float v464_sw = tensorforge::swap<32>(v415_el);
          float v469_sw = tensorforge::swap<32>(v420_el);
          r2[7] = (tensorforge::dppUpdate<228, 8, 15, false>((tensorforge::swap<64>(v423_el)), (tensorforge::dppUpdate<228, 4, 15, false>((tensorforge::swap<64>(v469_sw)), (tensorforge::dppUpdate<228, 2, 15, false>(v417_el, (tensorforge::dppUpdate<228, 1, 15, false>(v464_sw, v464_sw))))))));
          float v476_sw = tensorforge::swap<64>(v379_el);
          r2[8] = (tensorforge::dppUpdate<228, 8, 15, false>(v388_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v384_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v382_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v476_sw, v476_sw))))))));
          float v488_sw = tensorforge::swap<64>(v391_el);
          r2[9] = (tensorforge::dppUpdate<228, 8, 15, false>(v400_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v396_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v394_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v488_sw, v488_sw))))))));
          float v500_sw = tensorforge::swap<64>(v403_el);
          r2[10] = (tensorforge::dppUpdate<228, 8, 15, false>(v412_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v408_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v406_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v500_sw, v500_sw))))))));
          float v512_sw = tensorforge::swap<64>(v415_el);
          r2[11] = (tensorforge::dppUpdate<228, 8, 15, false>(v424_sw, (tensorforge::dppUpdate<228, 4, 15, false>(v420_el, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v418_sw)), (tensorforge::dppUpdate<228, 1, 15, false>(v512_sw, v512_sw))))))));
          float v525_sw = tensorforge::swap<64>(v428_sw);
          r2[12] = (tensorforge::dppUpdate<228, 8, 15, false>(v387_el, (tensorforge::dppUpdate<228, 4, 15, false>(v433_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v381_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v525_sw, v525_sw))))))));
          float v537_sw = tensorforge::swap<64>(v440_sw);
          r2[13] = (tensorforge::dppUpdate<228, 8, 15, false>(v399_el, (tensorforge::dppUpdate<228, 4, 15, false>(v445_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v393_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v537_sw, v537_sw))))))));
          float v549_sw = tensorforge::swap<64>(v452_sw);
          r2[14] = (tensorforge::dppUpdate<228, 8, 15, false>(v411_el, (tensorforge::dppUpdate<228, 4, 15, false>(v457_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v405_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v549_sw, v549_sw))))))));
          float v561_sw = tensorforge::swap<64>(v464_sw);
          r2[15] = (tensorforge::dppUpdate<228, 8, 15, false>(v423_el, (tensorforge::dppUpdate<228, 4, 15, false>(v469_sw, (tensorforge::dppUpdate<228, 2, 15, false>((tensorforge::swap<64>(v417_el)), (tensorforge::dppUpdate<228, 1, 15, false>(v561_sw, v561_sw))))))));
          // glb_m3 = store{r>g}(r2);
          if (v313_g) {
            #pragma unroll
            for (int32_t v571_i1 = 0; v571_i1 < 16; ++v571_i1) {
              float v573_data = r2[v571_i1];
              glb_m3[(v312_lead + (v571_i1 * 16))] = v573_data;
            }
          }
        }
      }
    }
  }
}

