// === base name ===
kernel_cb42b70e47781d14

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_cb42b70e47781d14 = {{32, 8, 1}, 32, 20, 1, 8, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_cb42b70e47781d14(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_cb42b70e47781d14(const float * m0, size_t m0_extraOffset, double * m1, size_t m1_extraOffset, const double * m2, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_cb42b70e47781d14(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_cb42b70e47781d14, block.x * block.y * block.z, 512 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (512 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_cb42b70e47781d14, block.x * block.y * block.z, 0));
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
  config.block[0] = 32;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 512 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_cb42b70e47781d14(const float * m0, size_t m0_extraOffset, double * m1, size_t m1_extraOffset, const double * m2, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_cb42b70e47781d14(numElements0, streamPtr);
  dim3 block (config.block[0], config.block[1], config.block[2]);
  dim3 grid (config.grid[0], config.grid[1], config.grid[2]);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_cb42b70e47781d14), hipFuncAttributeMaxDynamicSharedMemorySize, config.sharedMemBytes));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0Arg = (tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace>)m0;
  tensorforge::SpacePtr<double, tensorforge::GlobalMemspace> m1Arg = (tensorforge::SpacePtr<double, tensorforge::GlobalMemspace>)m1;
  tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace> m2Arg = (tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace>)m2;
  tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace> flags0Arg = (tensorforge::SpacePtr<const unsigned, tensorforge::GlobalMemspace>)flags0;
  hipLaunchKernelGGL(kernel_kernel_cb42b70e47781d14, grid, block, config.sharedMemBytes, stream, m0Arg, m0_extraOffset, m1Arg, m1_extraOffset, m2Arg, numElements0, flags0Arg);
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_cb42b70e47781d14(tensorforge::SpacePtr<const float, tensorforge::GlobalMemspace> m0, size_t m0_extraOffset, tensorforge::SpacePtr<double, tensorforge::GlobalMemspace> m1, size_t m1_extraOffset, tensorforge::SpacePtr<const double, tensorforge::ConstantMemspace> m2, size_t numElements0, tensorforge::SpacePtrRestrict<const unsigned, tensorforge::GlobalMemspace> flags0) {
  extern __shared__ char totalShrMemPtr[];
   {
    using namespace tensorforge::literals;
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // launch: 32 lanes (20 active) x 8 per block = block 32x8x1, 2048 B shared, occupancy grid
    // operands:
    //   m0 10×9(10×9) {0..10}×{0..9} strided
    //   m1 20×9(20×9) {0..20}×{0..9} strided
    //   m2 20×10(20×10) {0..20}×{0..10} none
    // operations:
    //   wide = copy(I)
    //   m1[i,j] = m2[i,k] × t0[k,j]
    // tensorforge-meta: {"fp":"float","launch":{"active_threads":20,"block":[32,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":512}],"shared_bytes":2048,"shared_elements":512,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"I","bbox":[[0,0],[10,9]],"name":"m0","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"strided","alias":"canonical","bbox":[[0,0],[20,9]],"name":"m1","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"none","alias":"embed","bbox":[[0,0],[20,10]],"name":"m2","ordered":false,"parts":1,"shape":[20,10],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]},"kind":"elementwise","op":"COPY","ops":[{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,10]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,10]},{"addressing":"pointer_based","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
    {
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[0 * threadIdx.y + 512];
      size_t v3_batchId1 = ((threadIdx.y + blockDim.y * (blockIdx.x)) < numElements0) ? (threadIdx.y + blockDim.y * (blockIdx.x)) : 0u;
      tensorforge::SpacePtrRestrict<const double, tensorforge::ConstantMemspace> const ptr_glb_m2 = (tensorforge::SpacePtrRestrict<const double, tensorforge::ConstantMemspace>)&m2[0];
      double * __restrict__ glb_m2 = reinterpret_cast<double*>(&totalShrMem[0]);
      // glb_m2 = load{g>s}(ptr_glb_m2[0, 1])
      if ((threadIdx.x + (threadIdx.y * 32)) < 200) {
        double v14_ld = ptr_glb_m2[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0];
        glb_m2[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = v14_ld;
      }
      __syncthreads();
      for (size_t v16_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v16_batchId0 < numElements0; v16_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v17_ahead1 = v16_batchId0 + (gridDim.x * blockDim.y);
        size_t v19_batchId1 = (v17_ahead1 < numElements0) ? v17_ahead1 : v16_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v16_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[v16_batchId0 * 90 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace> const glb_m1 = (tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace>)&m1[v16_batchId0 * 180 + 0 + m1_extraOffset];
          double r0[9]{};
          // r0 = copy(glb_m0)
          int32_t v29_lead = threadIdx.x % 32;
          bool v30_g = v29_lead < 10;
          if (v30_g) {
            #pragma unroll
            for (int32_t v31_k1 = 0; v31_k1 < 9; ++v31_k1) {
              float v36_data = glb_m0[(v29_lead + (v31_k1 * 10))];
              r0[v31_k1] = v36_data;
            }
          }
          // s0 = store{r>s}(localShrMem0, r0);
          double r1[9]{};
          // r1 = load{g>r}(s0);
          double r2[9]{};
          // ir2 = +(glb_m2 * r1)
          // [(0, 20), (0, 9)] [(0, 10)]
          double ir2[9]{};
          double v63_data = glb_m2[v29_lead];
          double v65_data = glb_m2[(v29_lead + 20)];
          double v67_data = glb_m2[(v29_lead + 40)];
          double v69_data = glb_m2[(v29_lead + 60)];
          double v71_data = glb_m2[(v29_lead + 80)];
          double v73_data = glb_m2[(v29_lead + 100)];
          double v75_data = glb_m2[(v29_lead + 120)];
          double v77_data = glb_m2[(v29_lead + 140)];
          double v79_data = glb_m2[(v29_lead + 160)];
          double v81_data = glb_m2[(v29_lead + 180)];
          double v82_acc{};
          double v83_acc{};
          double v84_acc{};
          double v85_acc{};
          double v86_acc{};
          double v87_acc{};
          double v88_acc{};
          double v89_acc{};
          double v90_acc{};
          double v91_data = r0[0];
          double v92_data = r0[1];
          double v93_data = r0[2];
          double v94_data = r0[3];
          double v95_data = r0[4];
          double v96_data = r0[5];
          double v97_data = r0[6];
          double v98_data = r0[7];
          double v99_data = r0[8];
          double v100_bc = tensorforge::broadcast<32, 16, 0>(v91_data);
          asm("s_nop 1\n\t"
          "v_fmac_f64_dpp %0, %1, %2 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %3 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %4 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %5 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %6 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %7 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %8 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %9 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %10 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %11 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "s_nop 1\n\t"
          : "+&v"(v82_acc)
          : "v"(v100_bc), "v"(v63_data), "v"(v65_data), "v"(v67_data), "v"(v69_data), "v"(v71_data), "v"(v73_data), "v"(v75_data), "v"(v77_data), "v"(v79_data), "v"(v81_data)
          );
          double v101_bc = tensorforge::broadcast<32, 16, 0>(v92_data);
          asm("s_nop 1\n\t"
          "v_fmac_f64_dpp %0, %1, %2 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %3 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %4 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %5 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %6 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %7 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %8 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %9 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %10 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %11 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "s_nop 1\n\t"
          : "+&v"(v83_acc)
          : "v"(v101_bc), "v"(v63_data), "v"(v65_data), "v"(v67_data), "v"(v69_data), "v"(v71_data), "v"(v73_data), "v"(v75_data), "v"(v77_data), "v"(v79_data), "v"(v81_data)
          );
          double v102_bc = tensorforge::broadcast<32, 16, 0>(v93_data);
          asm("s_nop 1\n\t"
          "v_fmac_f64_dpp %0, %1, %2 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %3 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %4 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %5 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %6 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %7 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %8 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %9 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %10 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %11 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "s_nop 1\n\t"
          : "+&v"(v84_acc)
          : "v"(v102_bc), "v"(v63_data), "v"(v65_data), "v"(v67_data), "v"(v69_data), "v"(v71_data), "v"(v73_data), "v"(v75_data), "v"(v77_data), "v"(v79_data), "v"(v81_data)
          );
          double v103_bc = tensorforge::broadcast<32, 16, 0>(v94_data);
          asm("s_nop 1\n\t"
          "v_fmac_f64_dpp %0, %1, %2 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %3 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %4 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %5 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %6 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %7 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %8 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %9 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %10 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %11 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "s_nop 1\n\t"
          : "+&v"(v85_acc)
          : "v"(v103_bc), "v"(v63_data), "v"(v65_data), "v"(v67_data), "v"(v69_data), "v"(v71_data), "v"(v73_data), "v"(v75_data), "v"(v77_data), "v"(v79_data), "v"(v81_data)
          );
          double v104_bc = tensorforge::broadcast<32, 16, 0>(v95_data);
          asm("s_nop 1\n\t"
          "v_fmac_f64_dpp %0, %1, %2 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %3 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %4 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %5 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %6 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %7 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %8 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %9 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %10 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %11 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "s_nop 1\n\t"
          : "+&v"(v86_acc)
          : "v"(v104_bc), "v"(v63_data), "v"(v65_data), "v"(v67_data), "v"(v69_data), "v"(v71_data), "v"(v73_data), "v"(v75_data), "v"(v77_data), "v"(v79_data), "v"(v81_data)
          );
          double v105_bc = tensorforge::broadcast<32, 16, 0>(v96_data);
          asm("s_nop 1\n\t"
          "v_fmac_f64_dpp %0, %1, %2 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %3 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %4 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %5 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %6 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %7 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %8 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %9 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %10 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %11 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "s_nop 1\n\t"
          : "+&v"(v87_acc)
          : "v"(v105_bc), "v"(v63_data), "v"(v65_data), "v"(v67_data), "v"(v69_data), "v"(v71_data), "v"(v73_data), "v"(v75_data), "v"(v77_data), "v"(v79_data), "v"(v81_data)
          );
          double v106_bc = tensorforge::broadcast<32, 16, 0>(v97_data);
          asm("s_nop 1\n\t"
          "v_fmac_f64_dpp %0, %1, %2 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %3 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %4 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %5 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %6 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %7 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %8 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %9 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %10 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %11 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "s_nop 1\n\t"
          : "+&v"(v88_acc)
          : "v"(v106_bc), "v"(v63_data), "v"(v65_data), "v"(v67_data), "v"(v69_data), "v"(v71_data), "v"(v73_data), "v"(v75_data), "v"(v77_data), "v"(v79_data), "v"(v81_data)
          );
          double v107_bc = tensorforge::broadcast<32, 16, 0>(v98_data);
          asm("s_nop 1\n\t"
          "v_fmac_f64_dpp %0, %1, %2 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %3 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %4 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %5 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %6 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %7 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %8 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %9 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %10 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %11 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "s_nop 1\n\t"
          : "+&v"(v89_acc)
          : "v"(v107_bc), "v"(v63_data), "v"(v65_data), "v"(v67_data), "v"(v69_data), "v"(v71_data), "v"(v73_data), "v"(v75_data), "v"(v77_data), "v"(v79_data), "v"(v81_data)
          );
          double v108_bc = tensorforge::broadcast<32, 16, 0>(v99_data);
          asm("s_nop 1\n\t"
          "v_fmac_f64_dpp %0, %1, %2 row_newbcast:0 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %3 row_newbcast:1 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %4 row_newbcast:2 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %5 row_newbcast:3 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %6 row_newbcast:4 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %7 row_newbcast:5 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %8 row_newbcast:6 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %9 row_newbcast:7 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %10 row_newbcast:8 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "v_fmac_f64_dpp %0, %1, %11 row_newbcast:9 row_mask:0xf bank_mask:0xf bound_ctrl:1\n\t"
          "s_nop 1\n\t"
          : "+&v"(v90_acc)
          : "v"(v108_bc), "v"(v63_data), "v"(v65_data), "v"(v67_data), "v"(v69_data), "v"(v71_data), "v"(v73_data), "v"(v75_data), "v"(v77_data), "v"(v79_data), "v"(v81_data)
          );
          ir2[0] = v82_acc;
          ir2[1] = v83_acc;
          ir2[2] = v84_acc;
          ir2[3] = v85_acc;
          ir2[4] = v86_acc;
          ir2[5] = v87_acc;
          ir2[6] = v88_acc;
          ir2[7] = v89_acc;
          ir2[8] = v90_acc;
          // r2 = ir2
          bool v109_g = v29_lead < 20;
          // glb_m1 = store{r>g}(r2);
          if (v109_g) {
            #pragma unroll
            for (int32_t v114_i1 = 0; v114_i1 < 9; ++v114_i1) {
              double v116_data = ir2[v114_i1];
              glb_m1[(v29_lead + (v114_i1 * 20))] = v116_data;
            }
          }
        }
      }
    }
  }
}

