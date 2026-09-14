// === base name ===
kernel_b64a4e513d704c92

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b64a4e513d704c92 = {{1, 16, 1}, 16, 12, 1, 16, 23552, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b64a4e513d704c92(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b64a4e513d704c92(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b64a4e513d704c92(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 16, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 16 - 1) / 16;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 5888 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_b64a4e513d704c92(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b64a4e513d704c92(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_b64a4e513d704c92(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_b64a4e513d704c92(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<5888 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 23552 B shared, occupancy grid
        // operands:
        //   m0 12×16(12×16) {0..12}×{0..16} strided
        //   m1 12×20(12×20) {0..12}×{0..20} strided
        //   m2 16×20(16×20) {0..16}×{0..20} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[j,k]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":5888}],"shared_bytes":23552,"shared_elements":5888,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,16]],"name":"m0","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,20]],"name":"m1","ordered":false,"parts":1,"shape":[12,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,20]],"name":"m2","ordered":false,"parts":1,"shape":[16,20],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,20]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,20]},{"addressing":"strided","bbox":[[0,0],[16,20]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,20]}],"permute":[[0,1],[1,0]],"target":[[0,-1],[1,-1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (368 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (352);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v5_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v5_batchId0 < numElements0; v5_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v6_ahead1 = v5_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 192 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 240 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 320 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 320> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v17_i1 = 0; v17_i1 < 20; ++v17_i1) {
                tensorforge::intel_esimd::simd<float, 12> v22_data;
                v22_data.copy_from(glb_m1 + ((v17_i1 * 12)));
                r0.template select<12, 1>((v17_i1 * 16)) = v22_data;
              }
              // s0 = load{g>s}(glb_m2[1, 0])
              #pragma unroll
              for (int32_t v25_i0 = 0; v25_i0 < 1; ++v25_i0) {
                int32_t v27_lead = v25_i0 * 16;
                #pragma unroll
                for (int32_t v26_i1 = 0; v26_i1 < 20; ++v26_i1) {
                  tensorforge::intel_esimd::simd<float, 16> v31_data;
                  v31_data.copy_from(glb_m2 + ((v27_lead + (v26_i1 * 16))));
                  tensorforge::slmStore<float, 16>(s0 + ((v27_lead + (v26_i1 * 17))), v31_data);
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[1, 0]));
              tensorforge::intel_esimd::simd<float, 256> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 12), (0, 16)] [(0, 20)]
              tensorforge::intel_esimd::simd<float, 256> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v36_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v37_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v38_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v39_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v40_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v48_data(r0.template select<16, 1>(192));
              tensorforge::intel_esimd::simd<float, 16> v49_data(r0.template select<16, 1>(208));
              tensorforge::intel_esimd::simd<float, 16> v50_data(r0.template select<16, 1>(224));
              tensorforge::intel_esimd::simd<float, 16> v51_data(r0.template select<16, 1>(240));
              tensorforge::intel_esimd::simd<float, 16> v52_data(r0.template select<16, 1>(256));
              tensorforge::intel_esimd::simd<float, 16> v53_data(r0.template select<16, 1>(272));
              tensorforge::intel_esimd::simd<float, 16> v54_data(r0.template select<16, 1>(288));
              tensorforge::intel_esimd::simd<float, 16> v55_data(r0.template select<16, 1>(304));
              tensorforge::intel_esimd::simd<float, 16> v56_acc{};
              tensorforge::intel_esimd::simd<float, 16> v61_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v56_acc += ((static_cast<float>(v61_data[0])) * v36_data);
              v56_acc += ((static_cast<float>(v61_data[1])) * v37_data);
              v56_acc += ((static_cast<float>(v61_data[2])) * v38_data);
              v56_acc += ((static_cast<float>(v61_data[3])) * v39_data);
              v56_acc += ((static_cast<float>(v61_data[4])) * v40_data);
              v56_acc += ((static_cast<float>(v61_data[5])) * v41_data);
              v56_acc += ((static_cast<float>(v61_data[6])) * v42_data);
              v56_acc += ((static_cast<float>(v61_data[7])) * v43_data);
              v56_acc += ((static_cast<float>(v61_data[8])) * v44_data);
              v56_acc += ((static_cast<float>(v61_data[9])) * v45_data);
              v56_acc += ((static_cast<float>(v61_data[10])) * v46_data);
              v56_acc += ((static_cast<float>(v61_data[11])) * v47_data);
              v56_acc += ((static_cast<float>(v61_data[12])) * v48_data);
              v56_acc += ((static_cast<float>(v61_data[13])) * v49_data);
              v56_acc += ((static_cast<float>(v61_data[14])) * v50_data);
              v56_acc += ((static_cast<float>(v61_data[15])) * v51_data);
              tensorforge::intel_esimd::simd<float, 16> v98_data = tensorforge::slmLoad<float, 16>(s0 + (272_i32));
              v56_acc += ((static_cast<float>(v98_data[0])) * v52_data);
              v56_acc += ((static_cast<float>(v98_data[1])) * v53_data);
              v56_acc += ((static_cast<float>(v98_data[2])) * v54_data);
              v56_acc += ((static_cast<float>(v98_data[3])) * v55_data);
              ir1.template select<16, 1>(0) = v56_acc;
              tensorforge::intel_esimd::simd<float, 16> v107_acc{};
              tensorforge::intel_esimd::simd<float, 16> v109_data = tensorforge::slmLoad<float, 16>(s0 + (1_i32));
              v107_acc += ((static_cast<float>(v109_data[0])) * v36_data);
              v107_acc += ((static_cast<float>(v109_data[1])) * v37_data);
              v107_acc += ((static_cast<float>(v109_data[2])) * v38_data);
              v107_acc += ((static_cast<float>(v109_data[3])) * v39_data);
              v107_acc += ((static_cast<float>(v109_data[4])) * v40_data);
              v107_acc += ((static_cast<float>(v109_data[5])) * v41_data);
              v107_acc += ((static_cast<float>(v109_data[6])) * v42_data);
              v107_acc += ((static_cast<float>(v109_data[7])) * v43_data);
              v107_acc += ((static_cast<float>(v109_data[8])) * v44_data);
              v107_acc += ((static_cast<float>(v109_data[9])) * v45_data);
              v107_acc += ((static_cast<float>(v109_data[10])) * v46_data);
              v107_acc += ((static_cast<float>(v109_data[11])) * v47_data);
              v107_acc += ((static_cast<float>(v109_data[12])) * v48_data);
              v107_acc += ((static_cast<float>(v109_data[13])) * v49_data);
              v107_acc += ((static_cast<float>(v109_data[14])) * v50_data);
              v107_acc += ((static_cast<float>(v109_data[15])) * v51_data);
              tensorforge::intel_esimd::simd<float, 16> v143_data = tensorforge::slmLoad<float, 16>(s0 + (273_i32));
              v107_acc += ((static_cast<float>(v143_data[0])) * v52_data);
              v107_acc += ((static_cast<float>(v143_data[1])) * v53_data);
              v107_acc += ((static_cast<float>(v143_data[2])) * v54_data);
              v107_acc += ((static_cast<float>(v143_data[3])) * v55_data);
              ir1.template select<16, 1>(16) = v107_acc;
              tensorforge::intel_esimd::simd<float, 16> v152_acc{};
              tensorforge::intel_esimd::simd<float, 16> v154_data = tensorforge::slmLoad<float, 16>(s0 + (2_i32));
              v152_acc += ((static_cast<float>(v154_data[0])) * v36_data);
              v152_acc += ((static_cast<float>(v154_data[1])) * v37_data);
              v152_acc += ((static_cast<float>(v154_data[2])) * v38_data);
              v152_acc += ((static_cast<float>(v154_data[3])) * v39_data);
              v152_acc += ((static_cast<float>(v154_data[4])) * v40_data);
              v152_acc += ((static_cast<float>(v154_data[5])) * v41_data);
              v152_acc += ((static_cast<float>(v154_data[6])) * v42_data);
              v152_acc += ((static_cast<float>(v154_data[7])) * v43_data);
              v152_acc += ((static_cast<float>(v154_data[8])) * v44_data);
              v152_acc += ((static_cast<float>(v154_data[9])) * v45_data);
              v152_acc += ((static_cast<float>(v154_data[10])) * v46_data);
              v152_acc += ((static_cast<float>(v154_data[11])) * v47_data);
              v152_acc += ((static_cast<float>(v154_data[12])) * v48_data);
              v152_acc += ((static_cast<float>(v154_data[13])) * v49_data);
              v152_acc += ((static_cast<float>(v154_data[14])) * v50_data);
              v152_acc += ((static_cast<float>(v154_data[15])) * v51_data);
              tensorforge::intel_esimd::simd<float, 16> v188_data = tensorforge::slmLoad<float, 16>(s0 + (274_i32));
              v152_acc += ((static_cast<float>(v188_data[0])) * v52_data);
              v152_acc += ((static_cast<float>(v188_data[1])) * v53_data);
              v152_acc += ((static_cast<float>(v188_data[2])) * v54_data);
              v152_acc += ((static_cast<float>(v188_data[3])) * v55_data);
              ir1.template select<16, 1>(32) = v152_acc;
              tensorforge::intel_esimd::simd<float, 16> v197_acc{};
              tensorforge::intel_esimd::simd<float, 16> v199_data = tensorforge::slmLoad<float, 16>(s0 + (3_i32));
              v197_acc += ((static_cast<float>(v199_data[0])) * v36_data);
              v197_acc += ((static_cast<float>(v199_data[1])) * v37_data);
              v197_acc += ((static_cast<float>(v199_data[2])) * v38_data);
              v197_acc += ((static_cast<float>(v199_data[3])) * v39_data);
              v197_acc += ((static_cast<float>(v199_data[4])) * v40_data);
              v197_acc += ((static_cast<float>(v199_data[5])) * v41_data);
              v197_acc += ((static_cast<float>(v199_data[6])) * v42_data);
              v197_acc += ((static_cast<float>(v199_data[7])) * v43_data);
              v197_acc += ((static_cast<float>(v199_data[8])) * v44_data);
              v197_acc += ((static_cast<float>(v199_data[9])) * v45_data);
              v197_acc += ((static_cast<float>(v199_data[10])) * v46_data);
              v197_acc += ((static_cast<float>(v199_data[11])) * v47_data);
              v197_acc += ((static_cast<float>(v199_data[12])) * v48_data);
              v197_acc += ((static_cast<float>(v199_data[13])) * v49_data);
              v197_acc += ((static_cast<float>(v199_data[14])) * v50_data);
              v197_acc += ((static_cast<float>(v199_data[15])) * v51_data);
              tensorforge::intel_esimd::simd<float, 16> v233_data = tensorforge::slmLoad<float, 16>(s0 + (275_i32));
              v197_acc += ((static_cast<float>(v233_data[0])) * v52_data);
              v197_acc += ((static_cast<float>(v233_data[1])) * v53_data);
              v197_acc += ((static_cast<float>(v233_data[2])) * v54_data);
              v197_acc += ((static_cast<float>(v233_data[3])) * v55_data);
              ir1.template select<16, 1>(48) = v197_acc;
              tensorforge::intel_esimd::simd<float, 16> v242_acc{};
              tensorforge::intel_esimd::simd<float, 16> v244_data = tensorforge::slmLoad<float, 16>(s0 + (4_i32));
              v242_acc += ((static_cast<float>(v244_data[0])) * v36_data);
              v242_acc += ((static_cast<float>(v244_data[1])) * v37_data);
              v242_acc += ((static_cast<float>(v244_data[2])) * v38_data);
              v242_acc += ((static_cast<float>(v244_data[3])) * v39_data);
              v242_acc += ((static_cast<float>(v244_data[4])) * v40_data);
              v242_acc += ((static_cast<float>(v244_data[5])) * v41_data);
              v242_acc += ((static_cast<float>(v244_data[6])) * v42_data);
              v242_acc += ((static_cast<float>(v244_data[7])) * v43_data);
              v242_acc += ((static_cast<float>(v244_data[8])) * v44_data);
              v242_acc += ((static_cast<float>(v244_data[9])) * v45_data);
              v242_acc += ((static_cast<float>(v244_data[10])) * v46_data);
              v242_acc += ((static_cast<float>(v244_data[11])) * v47_data);
              v242_acc += ((static_cast<float>(v244_data[12])) * v48_data);
              v242_acc += ((static_cast<float>(v244_data[13])) * v49_data);
              v242_acc += ((static_cast<float>(v244_data[14])) * v50_data);
              v242_acc += ((static_cast<float>(v244_data[15])) * v51_data);
              tensorforge::intel_esimd::simd<float, 16> v278_data = tensorforge::slmLoad<float, 16>(s0 + (276_i32));
              v242_acc += ((static_cast<float>(v278_data[0])) * v52_data);
              v242_acc += ((static_cast<float>(v278_data[1])) * v53_data);
              v242_acc += ((static_cast<float>(v278_data[2])) * v54_data);
              v242_acc += ((static_cast<float>(v278_data[3])) * v55_data);
              ir1.template select<16, 1>(64) = v242_acc;
              tensorforge::intel_esimd::simd<float, 16> v287_acc{};
              tensorforge::intel_esimd::simd<float, 16> v289_data = tensorforge::slmLoad<float, 16>(s0 + (5_i32));
              v287_acc += ((static_cast<float>(v289_data[0])) * v36_data);
              v287_acc += ((static_cast<float>(v289_data[1])) * v37_data);
              v287_acc += ((static_cast<float>(v289_data[2])) * v38_data);
              v287_acc += ((static_cast<float>(v289_data[3])) * v39_data);
              v287_acc += ((static_cast<float>(v289_data[4])) * v40_data);
              v287_acc += ((static_cast<float>(v289_data[5])) * v41_data);
              v287_acc += ((static_cast<float>(v289_data[6])) * v42_data);
              v287_acc += ((static_cast<float>(v289_data[7])) * v43_data);
              v287_acc += ((static_cast<float>(v289_data[8])) * v44_data);
              v287_acc += ((static_cast<float>(v289_data[9])) * v45_data);
              v287_acc += ((static_cast<float>(v289_data[10])) * v46_data);
              v287_acc += ((static_cast<float>(v289_data[11])) * v47_data);
              v287_acc += ((static_cast<float>(v289_data[12])) * v48_data);
              v287_acc += ((static_cast<float>(v289_data[13])) * v49_data);
              v287_acc += ((static_cast<float>(v289_data[14])) * v50_data);
              v287_acc += ((static_cast<float>(v289_data[15])) * v51_data);
              tensorforge::intel_esimd::simd<float, 16> v323_data = tensorforge::slmLoad<float, 16>(s0 + (277_i32));
              v287_acc += ((static_cast<float>(v323_data[0])) * v52_data);
              v287_acc += ((static_cast<float>(v323_data[1])) * v53_data);
              v287_acc += ((static_cast<float>(v323_data[2])) * v54_data);
              v287_acc += ((static_cast<float>(v323_data[3])) * v55_data);
              ir1.template select<16, 1>(80) = v287_acc;
              tensorforge::intel_esimd::simd<float, 16> v332_acc{};
              tensorforge::intel_esimd::simd<float, 16> v334_data = tensorforge::slmLoad<float, 16>(s0 + (6_i32));
              v332_acc += ((static_cast<float>(v334_data[0])) * v36_data);
              v332_acc += ((static_cast<float>(v334_data[1])) * v37_data);
              v332_acc += ((static_cast<float>(v334_data[2])) * v38_data);
              v332_acc += ((static_cast<float>(v334_data[3])) * v39_data);
              v332_acc += ((static_cast<float>(v334_data[4])) * v40_data);
              v332_acc += ((static_cast<float>(v334_data[5])) * v41_data);
              v332_acc += ((static_cast<float>(v334_data[6])) * v42_data);
              v332_acc += ((static_cast<float>(v334_data[7])) * v43_data);
              v332_acc += ((static_cast<float>(v334_data[8])) * v44_data);
              v332_acc += ((static_cast<float>(v334_data[9])) * v45_data);
              v332_acc += ((static_cast<float>(v334_data[10])) * v46_data);
              v332_acc += ((static_cast<float>(v334_data[11])) * v47_data);
              v332_acc += ((static_cast<float>(v334_data[12])) * v48_data);
              v332_acc += ((static_cast<float>(v334_data[13])) * v49_data);
              v332_acc += ((static_cast<float>(v334_data[14])) * v50_data);
              v332_acc += ((static_cast<float>(v334_data[15])) * v51_data);
              tensorforge::intel_esimd::simd<float, 16> v368_data = tensorforge::slmLoad<float, 16>(s0 + (278_i32));
              v332_acc += ((static_cast<float>(v368_data[0])) * v52_data);
              v332_acc += ((static_cast<float>(v368_data[1])) * v53_data);
              v332_acc += ((static_cast<float>(v368_data[2])) * v54_data);
              v332_acc += ((static_cast<float>(v368_data[3])) * v55_data);
              ir1.template select<16, 1>(96) = v332_acc;
              tensorforge::intel_esimd::simd<float, 16> v377_acc{};
              tensorforge::intel_esimd::simd<float, 16> v379_data = tensorforge::slmLoad<float, 16>(s0 + (7_i32));
              v377_acc += ((static_cast<float>(v379_data[0])) * v36_data);
              v377_acc += ((static_cast<float>(v379_data[1])) * v37_data);
              v377_acc += ((static_cast<float>(v379_data[2])) * v38_data);
              v377_acc += ((static_cast<float>(v379_data[3])) * v39_data);
              v377_acc += ((static_cast<float>(v379_data[4])) * v40_data);
              v377_acc += ((static_cast<float>(v379_data[5])) * v41_data);
              v377_acc += ((static_cast<float>(v379_data[6])) * v42_data);
              v377_acc += ((static_cast<float>(v379_data[7])) * v43_data);
              v377_acc += ((static_cast<float>(v379_data[8])) * v44_data);
              v377_acc += ((static_cast<float>(v379_data[9])) * v45_data);
              v377_acc += ((static_cast<float>(v379_data[10])) * v46_data);
              v377_acc += ((static_cast<float>(v379_data[11])) * v47_data);
              v377_acc += ((static_cast<float>(v379_data[12])) * v48_data);
              v377_acc += ((static_cast<float>(v379_data[13])) * v49_data);
              v377_acc += ((static_cast<float>(v379_data[14])) * v50_data);
              v377_acc += ((static_cast<float>(v379_data[15])) * v51_data);
              tensorforge::intel_esimd::simd<float, 16> v413_data = tensorforge::slmLoad<float, 16>(s0 + (279_i32));
              v377_acc += ((static_cast<float>(v413_data[0])) * v52_data);
              v377_acc += ((static_cast<float>(v413_data[1])) * v53_data);
              v377_acc += ((static_cast<float>(v413_data[2])) * v54_data);
              v377_acc += ((static_cast<float>(v413_data[3])) * v55_data);
              ir1.template select<16, 1>(112) = v377_acc;
              tensorforge::intel_esimd::simd<float, 16> v422_acc{};
              tensorforge::intel_esimd::simd<float, 16> v424_data = tensorforge::slmLoad<float, 16>(s0 + (8_i32));
              v422_acc += ((static_cast<float>(v424_data[0])) * v36_data);
              v422_acc += ((static_cast<float>(v424_data[1])) * v37_data);
              v422_acc += ((static_cast<float>(v424_data[2])) * v38_data);
              v422_acc += ((static_cast<float>(v424_data[3])) * v39_data);
              v422_acc += ((static_cast<float>(v424_data[4])) * v40_data);
              v422_acc += ((static_cast<float>(v424_data[5])) * v41_data);
              v422_acc += ((static_cast<float>(v424_data[6])) * v42_data);
              v422_acc += ((static_cast<float>(v424_data[7])) * v43_data);
              v422_acc += ((static_cast<float>(v424_data[8])) * v44_data);
              v422_acc += ((static_cast<float>(v424_data[9])) * v45_data);
              v422_acc += ((static_cast<float>(v424_data[10])) * v46_data);
              v422_acc += ((static_cast<float>(v424_data[11])) * v47_data);
              v422_acc += ((static_cast<float>(v424_data[12])) * v48_data);
              v422_acc += ((static_cast<float>(v424_data[13])) * v49_data);
              v422_acc += ((static_cast<float>(v424_data[14])) * v50_data);
              v422_acc += ((static_cast<float>(v424_data[15])) * v51_data);
              tensorforge::intel_esimd::simd<float, 16> v458_data = tensorforge::slmLoad<float, 16>(s0 + (280_i32));
              v422_acc += ((static_cast<float>(v458_data[0])) * v52_data);
              v422_acc += ((static_cast<float>(v458_data[1])) * v53_data);
              v422_acc += ((static_cast<float>(v458_data[2])) * v54_data);
              v422_acc += ((static_cast<float>(v458_data[3])) * v55_data);
              ir1.template select<16, 1>(128) = v422_acc;
              tensorforge::intel_esimd::simd<float, 16> v467_acc{};
              tensorforge::intel_esimd::simd<float, 16> v469_data = tensorforge::slmLoad<float, 16>(s0 + (9_i32));
              v467_acc += ((static_cast<float>(v469_data[0])) * v36_data);
              v467_acc += ((static_cast<float>(v469_data[1])) * v37_data);
              v467_acc += ((static_cast<float>(v469_data[2])) * v38_data);
              v467_acc += ((static_cast<float>(v469_data[3])) * v39_data);
              v467_acc += ((static_cast<float>(v469_data[4])) * v40_data);
              v467_acc += ((static_cast<float>(v469_data[5])) * v41_data);
              v467_acc += ((static_cast<float>(v469_data[6])) * v42_data);
              v467_acc += ((static_cast<float>(v469_data[7])) * v43_data);
              v467_acc += ((static_cast<float>(v469_data[8])) * v44_data);
              v467_acc += ((static_cast<float>(v469_data[9])) * v45_data);
              v467_acc += ((static_cast<float>(v469_data[10])) * v46_data);
              v467_acc += ((static_cast<float>(v469_data[11])) * v47_data);
              v467_acc += ((static_cast<float>(v469_data[12])) * v48_data);
              v467_acc += ((static_cast<float>(v469_data[13])) * v49_data);
              v467_acc += ((static_cast<float>(v469_data[14])) * v50_data);
              v467_acc += ((static_cast<float>(v469_data[15])) * v51_data);
              tensorforge::intel_esimd::simd<float, 16> v503_data = tensorforge::slmLoad<float, 16>(s0 + (281_i32));
              v467_acc += ((static_cast<float>(v503_data[0])) * v52_data);
              v467_acc += ((static_cast<float>(v503_data[1])) * v53_data);
              v467_acc += ((static_cast<float>(v503_data[2])) * v54_data);
              v467_acc += ((static_cast<float>(v503_data[3])) * v55_data);
              ir1.template select<16, 1>(144) = v467_acc;
              tensorforge::intel_esimd::simd<float, 16> v512_acc{};
              tensorforge::intel_esimd::simd<float, 16> v514_data = tensorforge::slmLoad<float, 16>(s0 + (10_i32));
              v512_acc += ((static_cast<float>(v514_data[0])) * v36_data);
              v512_acc += ((static_cast<float>(v514_data[1])) * v37_data);
              v512_acc += ((static_cast<float>(v514_data[2])) * v38_data);
              v512_acc += ((static_cast<float>(v514_data[3])) * v39_data);
              v512_acc += ((static_cast<float>(v514_data[4])) * v40_data);
              v512_acc += ((static_cast<float>(v514_data[5])) * v41_data);
              v512_acc += ((static_cast<float>(v514_data[6])) * v42_data);
              v512_acc += ((static_cast<float>(v514_data[7])) * v43_data);
              v512_acc += ((static_cast<float>(v514_data[8])) * v44_data);
              v512_acc += ((static_cast<float>(v514_data[9])) * v45_data);
              v512_acc += ((static_cast<float>(v514_data[10])) * v46_data);
              v512_acc += ((static_cast<float>(v514_data[11])) * v47_data);
              v512_acc += ((static_cast<float>(v514_data[12])) * v48_data);
              v512_acc += ((static_cast<float>(v514_data[13])) * v49_data);
              v512_acc += ((static_cast<float>(v514_data[14])) * v50_data);
              v512_acc += ((static_cast<float>(v514_data[15])) * v51_data);
              tensorforge::intel_esimd::simd<float, 16> v548_data = tensorforge::slmLoad<float, 16>(s0 + (282_i32));
              v512_acc += ((static_cast<float>(v548_data[0])) * v52_data);
              v512_acc += ((static_cast<float>(v548_data[1])) * v53_data);
              v512_acc += ((static_cast<float>(v548_data[2])) * v54_data);
              v512_acc += ((static_cast<float>(v548_data[3])) * v55_data);
              ir1.template select<16, 1>(160) = v512_acc;
              tensorforge::intel_esimd::simd<float, 16> v557_acc{};
              tensorforge::intel_esimd::simd<float, 16> v559_data = tensorforge::slmLoad<float, 16>(s0 + (11_i32));
              v557_acc += ((static_cast<float>(v559_data[0])) * v36_data);
              v557_acc += ((static_cast<float>(v559_data[1])) * v37_data);
              v557_acc += ((static_cast<float>(v559_data[2])) * v38_data);
              v557_acc += ((static_cast<float>(v559_data[3])) * v39_data);
              v557_acc += ((static_cast<float>(v559_data[4])) * v40_data);
              v557_acc += ((static_cast<float>(v559_data[5])) * v41_data);
              v557_acc += ((static_cast<float>(v559_data[6])) * v42_data);
              v557_acc += ((static_cast<float>(v559_data[7])) * v43_data);
              v557_acc += ((static_cast<float>(v559_data[8])) * v44_data);
              v557_acc += ((static_cast<float>(v559_data[9])) * v45_data);
              v557_acc += ((static_cast<float>(v559_data[10])) * v46_data);
              v557_acc += ((static_cast<float>(v559_data[11])) * v47_data);
              v557_acc += ((static_cast<float>(v559_data[12])) * v48_data);
              v557_acc += ((static_cast<float>(v559_data[13])) * v49_data);
              v557_acc += ((static_cast<float>(v559_data[14])) * v50_data);
              v557_acc += ((static_cast<float>(v559_data[15])) * v51_data);
              tensorforge::intel_esimd::simd<float, 16> v593_data = tensorforge::slmLoad<float, 16>(s0 + (283_i32));
              v557_acc += ((static_cast<float>(v593_data[0])) * v52_data);
              v557_acc += ((static_cast<float>(v593_data[1])) * v53_data);
              v557_acc += ((static_cast<float>(v593_data[2])) * v54_data);
              v557_acc += ((static_cast<float>(v593_data[3])) * v55_data);
              ir1.template select<16, 1>(176) = v557_acc;
              tensorforge::intel_esimd::simd<float, 16> v602_acc{};
              tensorforge::intel_esimd::simd<float, 16> v604_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              v602_acc += ((static_cast<float>(v604_data[0])) * v36_data);
              v602_acc += ((static_cast<float>(v604_data[1])) * v37_data);
              v602_acc += ((static_cast<float>(v604_data[2])) * v38_data);
              v602_acc += ((static_cast<float>(v604_data[3])) * v39_data);
              v602_acc += ((static_cast<float>(v604_data[4])) * v40_data);
              v602_acc += ((static_cast<float>(v604_data[5])) * v41_data);
              v602_acc += ((static_cast<float>(v604_data[6])) * v42_data);
              v602_acc += ((static_cast<float>(v604_data[7])) * v43_data);
              v602_acc += ((static_cast<float>(v604_data[8])) * v44_data);
              v602_acc += ((static_cast<float>(v604_data[9])) * v45_data);
              v602_acc += ((static_cast<float>(v604_data[10])) * v46_data);
              v602_acc += ((static_cast<float>(v604_data[11])) * v47_data);
              v602_acc += ((static_cast<float>(v604_data[12])) * v48_data);
              v602_acc += ((static_cast<float>(v604_data[13])) * v49_data);
              v602_acc += ((static_cast<float>(v604_data[14])) * v50_data);
              v602_acc += ((static_cast<float>(v604_data[15])) * v51_data);
              tensorforge::intel_esimd::simd<float, 16> v638_data = tensorforge::slmLoad<float, 16>(s0 + (284_i32));
              v602_acc += ((static_cast<float>(v638_data[0])) * v52_data);
              v602_acc += ((static_cast<float>(v638_data[1])) * v53_data);
              v602_acc += ((static_cast<float>(v638_data[2])) * v54_data);
              v602_acc += ((static_cast<float>(v638_data[3])) * v55_data);
              ir1.template select<16, 1>(192) = v602_acc;
              tensorforge::intel_esimd::simd<float, 16> v647_acc{};
              tensorforge::intel_esimd::simd<float, 16> v649_data = tensorforge::slmLoad<float, 16>(s0 + (13_i32));
              v647_acc += ((static_cast<float>(v649_data[0])) * v36_data);
              v647_acc += ((static_cast<float>(v649_data[1])) * v37_data);
              v647_acc += ((static_cast<float>(v649_data[2])) * v38_data);
              v647_acc += ((static_cast<float>(v649_data[3])) * v39_data);
              v647_acc += ((static_cast<float>(v649_data[4])) * v40_data);
              v647_acc += ((static_cast<float>(v649_data[5])) * v41_data);
              v647_acc += ((static_cast<float>(v649_data[6])) * v42_data);
              v647_acc += ((static_cast<float>(v649_data[7])) * v43_data);
              v647_acc += ((static_cast<float>(v649_data[8])) * v44_data);
              v647_acc += ((static_cast<float>(v649_data[9])) * v45_data);
              v647_acc += ((static_cast<float>(v649_data[10])) * v46_data);
              v647_acc += ((static_cast<float>(v649_data[11])) * v47_data);
              v647_acc += ((static_cast<float>(v649_data[12])) * v48_data);
              v647_acc += ((static_cast<float>(v649_data[13])) * v49_data);
              v647_acc += ((static_cast<float>(v649_data[14])) * v50_data);
              v647_acc += ((static_cast<float>(v649_data[15])) * v51_data);
              tensorforge::intel_esimd::simd<float, 16> v683_data = tensorforge::slmLoad<float, 16>(s0 + (285_i32));
              v647_acc += ((static_cast<float>(v683_data[0])) * v52_data);
              v647_acc += ((static_cast<float>(v683_data[1])) * v53_data);
              v647_acc += ((static_cast<float>(v683_data[2])) * v54_data);
              v647_acc += ((static_cast<float>(v683_data[3])) * v55_data);
              ir1.template select<16, 1>(208) = v647_acc;
              tensorforge::intel_esimd::simd<float, 16> v692_acc{};
              tensorforge::intel_esimd::simd<float, 16> v694_data = tensorforge::slmLoad<float, 16>(s0 + (14_i32));
              v692_acc += ((static_cast<float>(v694_data[0])) * v36_data);
              v692_acc += ((static_cast<float>(v694_data[1])) * v37_data);
              v692_acc += ((static_cast<float>(v694_data[2])) * v38_data);
              v692_acc += ((static_cast<float>(v694_data[3])) * v39_data);
              v692_acc += ((static_cast<float>(v694_data[4])) * v40_data);
              v692_acc += ((static_cast<float>(v694_data[5])) * v41_data);
              v692_acc += ((static_cast<float>(v694_data[6])) * v42_data);
              v692_acc += ((static_cast<float>(v694_data[7])) * v43_data);
              v692_acc += ((static_cast<float>(v694_data[8])) * v44_data);
              v692_acc += ((static_cast<float>(v694_data[9])) * v45_data);
              v692_acc += ((static_cast<float>(v694_data[10])) * v46_data);
              v692_acc += ((static_cast<float>(v694_data[11])) * v47_data);
              v692_acc += ((static_cast<float>(v694_data[12])) * v48_data);
              v692_acc += ((static_cast<float>(v694_data[13])) * v49_data);
              v692_acc += ((static_cast<float>(v694_data[14])) * v50_data);
              v692_acc += ((static_cast<float>(v694_data[15])) * v51_data);
              tensorforge::intel_esimd::simd<float, 16> v728_data = tensorforge::slmLoad<float, 16>(s0 + (286_i32));
              v692_acc += ((static_cast<float>(v728_data[0])) * v52_data);
              v692_acc += ((static_cast<float>(v728_data[1])) * v53_data);
              v692_acc += ((static_cast<float>(v728_data[2])) * v54_data);
              v692_acc += ((static_cast<float>(v728_data[3])) * v55_data);
              ir1.template select<16, 1>(224) = v692_acc;
              tensorforge::intel_esimd::simd<float, 16> v737_acc{};
              tensorforge::intel_esimd::simd<float, 16> v739_data = tensorforge::slmLoad<float, 16>(s0 + (15_i32));
              v737_acc += ((static_cast<float>(v739_data[0])) * v36_data);
              v737_acc += ((static_cast<float>(v739_data[1])) * v37_data);
              v737_acc += ((static_cast<float>(v739_data[2])) * v38_data);
              v737_acc += ((static_cast<float>(v739_data[3])) * v39_data);
              v737_acc += ((static_cast<float>(v739_data[4])) * v40_data);
              v737_acc += ((static_cast<float>(v739_data[5])) * v41_data);
              v737_acc += ((static_cast<float>(v739_data[6])) * v42_data);
              v737_acc += ((static_cast<float>(v739_data[7])) * v43_data);
              v737_acc += ((static_cast<float>(v739_data[8])) * v44_data);
              v737_acc += ((static_cast<float>(v739_data[9])) * v45_data);
              v737_acc += ((static_cast<float>(v739_data[10])) * v46_data);
              v737_acc += ((static_cast<float>(v739_data[11])) * v47_data);
              v737_acc += ((static_cast<float>(v739_data[12])) * v48_data);
              v737_acc += ((static_cast<float>(v739_data[13])) * v49_data);
              v737_acc += ((static_cast<float>(v739_data[14])) * v50_data);
              v737_acc += ((static_cast<float>(v739_data[15])) * v51_data);
              tensorforge::intel_esimd::simd<float, 16> v773_data = tensorforge::slmLoad<float, 16>(s0 + (287_i32));
              v737_acc += ((static_cast<float>(v773_data[0])) * v52_data);
              v737_acc += ((static_cast<float>(v773_data[1])) * v53_data);
              v737_acc += ((static_cast<float>(v773_data[2])) * v54_data);
              v737_acc += ((static_cast<float>(v773_data[3])) * v55_data);
              ir1.template select<16, 1>(240) = v737_acc;
              // r1 = ir1
              #pragma unroll
              for (int32_t v782_n1 = 0; v782_n1 < 16; ++v782_n1) {
                int32_t v783_a = v782_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v785_data(ir1.template select<12, 1>(v783_a));
                r1.template select<12, 1>(v783_a) = v785_data;
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v786_i1 = 0; v786_i1 < 16; ++v786_i1) {
                tensorforge::intel_esimd::simd<float, 12> v789_data(r1.template select<12, 1>((v786_i1 * 16)));
                v789_data.copy_to(glb_m0 + ((v786_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}

