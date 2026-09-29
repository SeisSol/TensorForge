// === base name ===
kernel_ebd5acecbf4d4dc5

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_ebd5acecbf4d4dc5 = {{1, 8, 1}, 32, 32, 1, 8, 3072, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_ebd5acecbf4d4dc5(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_ebd5acecbf4d4dc5(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_ebd5acecbf4d4dc5(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 8, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 8 - 1) / 8;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 768 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_ebd5acecbf4d4dc5(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_ebd5acecbf4d4dc5(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_ebd5acecbf4d4dc5(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_ebd5acecbf4d4dc5(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<768 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes x 8 per block = block 1x8x1, 3072 B shared, occupancy grid
        // operands:
        //   m0 32×9(32×9) {0..32}×{0..9} pointer_based
        //   m1 16×9(16×9) {0..16}×{0..9} pointer_based
        //   m2 16×9(16×9) {0..16}×{0..9} pointer_based
        //   m3 32×9(32×9) {0..32}×{0..9} pointer_based
        //   m4 9×9(9×9) {0..9}×{0..9} pointer_based
        // operations:
        //   t0[i,j] = m0[i,j]
        //   t0[i,j] += m1[i,j]
        //   t0[i,j] += m2[i,j]
        //   m3[i,j] = t0[i,k] × m4[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[1,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":768}],"shared_bytes":3072,"shared_elements":768,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"pointer_based","alias":"Q","bbox":[[0,0],[32,9]],"name":"m0","ordered":false,"parts":1,"shape":[32,9],"variant":false},{"addressing":"pointer_based","alias":"F0","bbox":[[0,0],[16,9]],"name":"m1","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"pointer_based","alias":"F1","bbox":[[0,0],[16,9]],"name":"m2","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"pointer_based","alias":"O","bbox":[[0,0],[32,9]],"name":"m3","ordered":false,"parts":1,"shape":[32,9],"variant":false},{"addressing":"pointer_based","alias":"M","bbox":[[0,0],[9,9]],"name":"m4","ordered":false,"parts":1,"shape":[9,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[32,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[32,9]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (96 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (96);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v5_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v5_batchId0 < numElements0; v5_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v6_ahead1 = v5_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v5_batchId0][0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v5_batchId0][0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v5_batchId0][0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v5_batchId0][0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v5_batchId0][0 + m4_extraOffset];
              tensorforge::intel_esimd::simd<float, 288> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v19_i0 = 0; v19_i0 < 1; ++v19_i0) {
                int32_t v21_lead = v19_i0 * 32;
                #pragma unroll
                for (int32_t v20_i1 = 0; v20_i1 < 9; ++v20_i1) {
                  int32_t v24_a = v21_lead + (v20_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v25_data;
                  v25_data.copy_from(glb_m0 + (v24_a));
                  r0.template select<32, 1>(v24_a) = v25_data;
                }
              }
              tensorforge::intel_esimd::simd<float, 288> r2(0.0f);
              // r2 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v28_i1 = 0; v28_i1 < 9; ++v28_i1) {
                tensorforge::intel_esimd::simd<float, 16> v33_data;
                v33_data.copy_from(glb_m1 + ((v28_i1 * 16)));
                r2.template select<16, 1>((v28_i1 * 32)) = v33_data;
              }
              // wait(r0 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<float, 288> r1(0.0f);
              // r1 = +(r0) + None
              // [(0, 32), (0, 9)] []
              tensorforge::intel_esimd::simd<float, 32> v37_data(r0.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 32> v38_data(r1.template select<32, 1>(0));
              r1.template select<32, 1>(0) = (v38_data + v37_data);
              tensorforge::intel_esimd::simd<float, 32> v40_data(r0.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v41_data(r1.template select<32, 1>(32));
              r1.template select<32, 1>(32) = (v41_data + v40_data);
              tensorforge::intel_esimd::simd<float, 32> v43_data(r0.template select<32, 1>(64));
              tensorforge::intel_esimd::simd<float, 32> v44_data(r1.template select<32, 1>(64));
              r1.template select<32, 1>(64) = (v44_data + v43_data);
              tensorforge::intel_esimd::simd<float, 32> v46_data(r0.template select<32, 1>(96));
              tensorforge::intel_esimd::simd<float, 32> v47_data(r1.template select<32, 1>(96));
              r1.template select<32, 1>(96) = (v47_data + v46_data);
              tensorforge::intel_esimd::simd<float, 32> v49_data(r0.template select<32, 1>(128));
              tensorforge::intel_esimd::simd<float, 32> v50_data(r1.template select<32, 1>(128));
              r1.template select<32, 1>(128) = (v50_data + v49_data);
              tensorforge::intel_esimd::simd<float, 32> v52_data(r0.template select<32, 1>(160));
              tensorforge::intel_esimd::simd<float, 32> v53_data(r1.template select<32, 1>(160));
              r1.template select<32, 1>(160) = (v53_data + v52_data);
              tensorforge::intel_esimd::simd<float, 32> v55_data(r0.template select<32, 1>(192));
              tensorforge::intel_esimd::simd<float, 32> v56_data(r1.template select<32, 1>(192));
              r1.template select<32, 1>(192) = (v56_data + v55_data);
              tensorforge::intel_esimd::simd<float, 32> v58_data(r0.template select<32, 1>(224));
              tensorforge::intel_esimd::simd<float, 32> v59_data(r1.template select<32, 1>(224));
              r1.template select<32, 1>(224) = (v59_data + v58_data);
              tensorforge::intel_esimd::simd<float, 32> v61_data(r0.template select<32, 1>(256));
              tensorforge::intel_esimd::simd<float, 32> v62_data(r1.template select<32, 1>(256));
              r1.template select<32, 1>(256) = (v62_data + v61_data);
              tensorforge::intel_esimd::simd<float, 288> r4(0.0f);
              // r4 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v65_i1 = 0; v65_i1 < 9; ++v65_i1) {
                tensorforge::intel_esimd::simd<float, 16> v70_data;
                v70_data.copy_from(glb_m2 + ((v65_i1 * 16)));
                r4.template select<16, 1>((v65_i1 * 32)) = v70_data;
              }
              // wait(r2 = load{g>r}(glb_m1););
              tensorforge::intel_esimd::simd<float, 288> r3(0.0f);
              // ir3 = +(r2)
              // [(0, 16), (0, 9)] []
              tensorforge::intel_esimd::simd<float, 288> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v75_data(r2.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 32> v76_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v76_data + v75_data);
              tensorforge::intel_esimd::simd<float, 32> v78_data(r2.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v79_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v79_data + v78_data);
              tensorforge::intel_esimd::simd<float, 32> v81_data(r2.template select<32, 1>(64));
              tensorforge::intel_esimd::simd<float, 32> v82_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v82_data + v81_data);
              tensorforge::intel_esimd::simd<float, 32> v84_data(r2.template select<32, 1>(96));
              tensorforge::intel_esimd::simd<float, 32> v85_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v85_data + v84_data);
              tensorforge::intel_esimd::simd<float, 32> v87_data(r2.template select<32, 1>(128));
              tensorforge::intel_esimd::simd<float, 32> v88_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v88_data + v87_data);
              tensorforge::intel_esimd::simd<float, 32> v90_data(r2.template select<32, 1>(160));
              tensorforge::intel_esimd::simd<float, 32> v91_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v91_data + v90_data);
              tensorforge::intel_esimd::simd<float, 32> v93_data(r2.template select<32, 1>(192));
              tensorforge::intel_esimd::simd<float, 32> v94_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v94_data + v93_data);
              tensorforge::intel_esimd::simd<float, 32> v96_data(r2.template select<32, 1>(224));
              tensorforge::intel_esimd::simd<float, 32> v97_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v97_data + v96_data);
              tensorforge::intel_esimd::simd<float, 32> v99_data(r2.template select<32, 1>(256));
              tensorforge::intel_esimd::simd<float, 32> v100_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v100_data + v99_data);
              // r3 = ir3 + r1
              #pragma unroll
              for (int32_t v102_n1 = 0; v102_n1 < 9; ++v102_n1) {
                int32_t v103_a = v102_n1 * 32;
                tensorforge::intel_esimd::simd<float, 32> v105_data(ir3.template select<32, 1>(v103_a));
                tensorforge::intel_esimd::simd<float, 32> v106_data(r1.template select<32, 1>(v103_a));
                r3.template select<32, 1>(v103_a) = (v106_data + v105_data);
              }
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v108_ld;
              v108_ld.copy_from(glb_m4 + (0 + 0 + 2 * 0 + 0));
              tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 2 * 0 + 0), v108_ld);
              tensorforge::intel_esimd::simd<float, 17> v109_ld;
              v109_ld.copy_from(glb_m4 + (0 + 0 + 1 * 0 + 64));
              tensorforge::slmStore<float, 17>(s1 + (0 + 0 + 1 * 0 + 64), v109_ld);
              // wait(r4 = load{g>r}(glb_m2););
              tensorforge::intel_esimd::simd<float, 288> r5(0.0f);
              // ir5 = +(r4)
              // [(0, 16), (0, 9)] []
              tensorforge::intel_esimd::simd<float, 288> ir5(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v112_data(r4.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 32> v113_data(ir5.template select<32, 1>(0));
              ir5.template select<32, 1>(0) = (v113_data + v112_data);
              tensorforge::intel_esimd::simd<float, 32> v115_data(r4.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v116_data(ir5.template select<32, 1>(32));
              ir5.template select<32, 1>(32) = (v116_data + v115_data);
              tensorforge::intel_esimd::simd<float, 32> v118_data(r4.template select<32, 1>(64));
              tensorforge::intel_esimd::simd<float, 32> v119_data(ir5.template select<32, 1>(64));
              ir5.template select<32, 1>(64) = (v119_data + v118_data);
              tensorforge::intel_esimd::simd<float, 32> v121_data(r4.template select<32, 1>(96));
              tensorforge::intel_esimd::simd<float, 32> v122_data(ir5.template select<32, 1>(96));
              ir5.template select<32, 1>(96) = (v122_data + v121_data);
              tensorforge::intel_esimd::simd<float, 32> v124_data(r4.template select<32, 1>(128));
              tensorforge::intel_esimd::simd<float, 32> v125_data(ir5.template select<32, 1>(128));
              ir5.template select<32, 1>(128) = (v125_data + v124_data);
              tensorforge::intel_esimd::simd<float, 32> v127_data(r4.template select<32, 1>(160));
              tensorforge::intel_esimd::simd<float, 32> v128_data(ir5.template select<32, 1>(160));
              ir5.template select<32, 1>(160) = (v128_data + v127_data);
              tensorforge::intel_esimd::simd<float, 32> v130_data(r4.template select<32, 1>(192));
              tensorforge::intel_esimd::simd<float, 32> v131_data(ir5.template select<32, 1>(192));
              ir5.template select<32, 1>(192) = (v131_data + v130_data);
              tensorforge::intel_esimd::simd<float, 32> v133_data(r4.template select<32, 1>(224));
              tensorforge::intel_esimd::simd<float, 32> v134_data(ir5.template select<32, 1>(224));
              ir5.template select<32, 1>(224) = (v134_data + v133_data);
              tensorforge::intel_esimd::simd<float, 32> v136_data(r4.template select<32, 1>(256));
              tensorforge::intel_esimd::simd<float, 32> v137_data(ir5.template select<32, 1>(256));
              ir5.template select<32, 1>(256) = (v137_data + v136_data);
              // r5 = ir5 + r3
              #pragma unroll
              for (int32_t v139_n1 = 0; v139_n1 < 9; ++v139_n1) {
                int32_t v140_a = v139_n1 * 32;
                tensorforge::intel_esimd::simd<float, 32> v142_data(ir5.template select<32, 1>(v140_a));
                tensorforge::intel_esimd::simd<float, 32> v143_data(r3.template select<32, 1>(v140_a));
                r5.template select<32, 1>(v140_a) = (v143_data + v142_data);
              }
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              tensorforge::intel_esimd::simd<float, 288> r6(0.0f);
              // ir6 = +(r5 * s1)
              // [(0, 32), (0, 9)] [(0, 9)]
              tensorforge::intel_esimd::simd<float, 288> ir6(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v147_data(r5.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 96> s1_w0 = tensorforge::slmLoad<float, 96>(s1 + 0);
              float v148_data = s1_w0[0];
              tensorforge::intel_esimd::simd<float, 32> v150_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v150_data + (v147_data * v148_data));
              float v153_data = s1_w0[9];
              tensorforge::intel_esimd::simd<float, 32> v155_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v155_data + (v147_data * v153_data));
              float v158_data = s1_w0[18];
              tensorforge::intel_esimd::simd<float, 32> v160_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v160_data + (v147_data * v158_data));
              float v163_data = s1_w0[27];
              tensorforge::intel_esimd::simd<float, 32> v165_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v165_data + (v147_data * v163_data));
              float v168_data = s1_w0[36];
              tensorforge::intel_esimd::simd<float, 32> v170_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v170_data + (v147_data * v168_data));
              float v173_data = s1_w0[45];
              tensorforge::intel_esimd::simd<float, 32> v175_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v175_data + (v147_data * v173_data));
              float v178_data = s1_w0[54];
              tensorforge::intel_esimd::simd<float, 32> v180_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v180_data + (v147_data * v178_data));
              float v183_data = s1_w0[63];
              tensorforge::intel_esimd::simd<float, 32> v185_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v185_data + (v147_data * v183_data));
              float v188_data = s1_w0[72];
              tensorforge::intel_esimd::simd<float, 32> v190_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v190_data + (v147_data * v188_data));
              tensorforge::intel_esimd::simd<float, 32> v192_data(r5.template select<32, 1>(32));
              float v193_data = s1_w0[1];
              tensorforge::intel_esimd::simd<float, 32> v195_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v195_data + (v192_data * v193_data));
              float v198_data = s1_w0[10];
              tensorforge::intel_esimd::simd<float, 32> v200_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v200_data + (v192_data * v198_data));
              float v203_data = s1_w0[19];
              tensorforge::intel_esimd::simd<float, 32> v205_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v205_data + (v192_data * v203_data));
              float v208_data = s1_w0[28];
              tensorforge::intel_esimd::simd<float, 32> v210_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v210_data + (v192_data * v208_data));
              float v213_data = s1_w0[37];
              tensorforge::intel_esimd::simd<float, 32> v215_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v215_data + (v192_data * v213_data));
              float v218_data = s1_w0[46];
              tensorforge::intel_esimd::simd<float, 32> v220_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v220_data + (v192_data * v218_data));
              float v223_data = s1_w0[55];
              tensorforge::intel_esimd::simd<float, 32> v225_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v225_data + (v192_data * v223_data));
              float v228_data = s1_w0[64];
              tensorforge::intel_esimd::simd<float, 32> v230_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v230_data + (v192_data * v228_data));
              float v233_data = s1_w0[73];
              tensorforge::intel_esimd::simd<float, 32> v235_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v235_data + (v192_data * v233_data));
              tensorforge::intel_esimd::simd<float, 32> v237_data(r5.template select<32, 1>(64));
              float v238_data = s1_w0[2];
              tensorforge::intel_esimd::simd<float, 32> v240_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v240_data + (v237_data * v238_data));
              float v243_data = s1_w0[11];
              tensorforge::intel_esimd::simd<float, 32> v245_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v245_data + (v237_data * v243_data));
              float v248_data = s1_w0[20];
              tensorforge::intel_esimd::simd<float, 32> v250_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v250_data + (v237_data * v248_data));
              float v253_data = s1_w0[29];
              tensorforge::intel_esimd::simd<float, 32> v255_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v255_data + (v237_data * v253_data));
              float v258_data = s1_w0[38];
              tensorforge::intel_esimd::simd<float, 32> v260_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v260_data + (v237_data * v258_data));
              float v263_data = s1_w0[47];
              tensorforge::intel_esimd::simd<float, 32> v265_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v265_data + (v237_data * v263_data));
              float v268_data = s1_w0[56];
              tensorforge::intel_esimd::simd<float, 32> v270_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v270_data + (v237_data * v268_data));
              float v273_data = s1_w0[65];
              tensorforge::intel_esimd::simd<float, 32> v275_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v275_data + (v237_data * v273_data));
              float v278_data = s1_w0[74];
              tensorforge::intel_esimd::simd<float, 32> v280_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v280_data + (v237_data * v278_data));
              tensorforge::intel_esimd::simd<float, 32> v282_data(r5.template select<32, 1>(96));
              float v283_data = s1_w0[3];
              tensorforge::intel_esimd::simd<float, 32> v285_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v285_data + (v282_data * v283_data));
              float v288_data = s1_w0[12];
              tensorforge::intel_esimd::simd<float, 32> v290_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v290_data + (v282_data * v288_data));
              float v293_data = s1_w0[21];
              tensorforge::intel_esimd::simd<float, 32> v295_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v295_data + (v282_data * v293_data));
              float v298_data = s1_w0[30];
              tensorforge::intel_esimd::simd<float, 32> v300_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v300_data + (v282_data * v298_data));
              float v303_data = s1_w0[39];
              tensorforge::intel_esimd::simd<float, 32> v305_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v305_data + (v282_data * v303_data));
              float v308_data = s1_w0[48];
              tensorforge::intel_esimd::simd<float, 32> v310_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v310_data + (v282_data * v308_data));
              float v313_data = s1_w0[57];
              tensorforge::intel_esimd::simd<float, 32> v315_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v315_data + (v282_data * v313_data));
              float v318_data = s1_w0[66];
              tensorforge::intel_esimd::simd<float, 32> v320_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v320_data + (v282_data * v318_data));
              float v323_data = s1_w0[75];
              tensorforge::intel_esimd::simd<float, 32> v325_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v325_data + (v282_data * v323_data));
              tensorforge::intel_esimd::simd<float, 32> v327_data(r5.template select<32, 1>(128));
              float v328_data = s1_w0[4];
              tensorforge::intel_esimd::simd<float, 32> v330_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v330_data + (v327_data * v328_data));
              float v333_data = s1_w0[13];
              tensorforge::intel_esimd::simd<float, 32> v335_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v335_data + (v327_data * v333_data));
              float v338_data = s1_w0[22];
              tensorforge::intel_esimd::simd<float, 32> v340_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v340_data + (v327_data * v338_data));
              float v343_data = s1_w0[31];
              tensorforge::intel_esimd::simd<float, 32> v345_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v345_data + (v327_data * v343_data));
              float v348_data = s1_w0[40];
              tensorforge::intel_esimd::simd<float, 32> v350_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v350_data + (v327_data * v348_data));
              float v353_data = s1_w0[49];
              tensorforge::intel_esimd::simd<float, 32> v355_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v355_data + (v327_data * v353_data));
              float v358_data = s1_w0[58];
              tensorforge::intel_esimd::simd<float, 32> v360_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v360_data + (v327_data * v358_data));
              float v363_data = s1_w0[67];
              tensorforge::intel_esimd::simd<float, 32> v365_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v365_data + (v327_data * v363_data));
              float v368_data = s1_w0[76];
              tensorforge::intel_esimd::simd<float, 32> v370_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v370_data + (v327_data * v368_data));
              tensorforge::intel_esimd::simd<float, 32> v372_data(r5.template select<32, 1>(160));
              float v373_data = s1_w0[5];
              tensorforge::intel_esimd::simd<float, 32> v375_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v375_data + (v372_data * v373_data));
              float v378_data = s1_w0[14];
              tensorforge::intel_esimd::simd<float, 32> v380_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v380_data + (v372_data * v378_data));
              float v383_data = s1_w0[23];
              tensorforge::intel_esimd::simd<float, 32> v385_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v385_data + (v372_data * v383_data));
              float v388_data = s1_w0[32];
              tensorforge::intel_esimd::simd<float, 32> v390_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v390_data + (v372_data * v388_data));
              float v393_data = s1_w0[41];
              tensorforge::intel_esimd::simd<float, 32> v395_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v395_data + (v372_data * v393_data));
              float v398_data = s1_w0[50];
              tensorforge::intel_esimd::simd<float, 32> v400_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v400_data + (v372_data * v398_data));
              float v403_data = s1_w0[59];
              tensorforge::intel_esimd::simd<float, 32> v405_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v405_data + (v372_data * v403_data));
              float v408_data = s1_w0[68];
              tensorforge::intel_esimd::simd<float, 32> v410_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v410_data + (v372_data * v408_data));
              float v413_data = s1_w0[77];
              tensorforge::intel_esimd::simd<float, 32> v415_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v415_data + (v372_data * v413_data));
              tensorforge::intel_esimd::simd<float, 32> v417_data(r5.template select<32, 1>(192));
              float v418_data = s1_w0[6];
              tensorforge::intel_esimd::simd<float, 32> v420_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v420_data + (v417_data * v418_data));
              float v423_data = s1_w0[15];
              tensorforge::intel_esimd::simd<float, 32> v425_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v425_data + (v417_data * v423_data));
              float v428_data = s1_w0[24];
              tensorforge::intel_esimd::simd<float, 32> v430_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v430_data + (v417_data * v428_data));
              float v433_data = s1_w0[33];
              tensorforge::intel_esimd::simd<float, 32> v435_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v435_data + (v417_data * v433_data));
              float v438_data = s1_w0[42];
              tensorforge::intel_esimd::simd<float, 32> v440_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v440_data + (v417_data * v438_data));
              float v443_data = s1_w0[51];
              tensorforge::intel_esimd::simd<float, 32> v445_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v445_data + (v417_data * v443_data));
              float v448_data = s1_w0[60];
              tensorforge::intel_esimd::simd<float, 32> v450_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v450_data + (v417_data * v448_data));
              float v453_data = s1_w0[69];
              tensorforge::intel_esimd::simd<float, 32> v455_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v455_data + (v417_data * v453_data));
              float v458_data = s1_w0[78];
              tensorforge::intel_esimd::simd<float, 32> v460_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v460_data + (v417_data * v458_data));
              tensorforge::intel_esimd::simd<float, 32> v462_data(r5.template select<32, 1>(224));
              float v463_data = s1_w0[7];
              tensorforge::intel_esimd::simd<float, 32> v465_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v465_data + (v462_data * v463_data));
              float v468_data = s1_w0[16];
              tensorforge::intel_esimd::simd<float, 32> v470_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v470_data + (v462_data * v468_data));
              float v473_data = s1_w0[25];
              tensorforge::intel_esimd::simd<float, 32> v475_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v475_data + (v462_data * v473_data));
              float v478_data = s1_w0[34];
              tensorforge::intel_esimd::simd<float, 32> v480_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v480_data + (v462_data * v478_data));
              float v483_data = s1_w0[43];
              tensorforge::intel_esimd::simd<float, 32> v485_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v485_data + (v462_data * v483_data));
              float v488_data = s1_w0[52];
              tensorforge::intel_esimd::simd<float, 32> v490_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v490_data + (v462_data * v488_data));
              float v493_data = s1_w0[61];
              tensorforge::intel_esimd::simd<float, 32> v495_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v495_data + (v462_data * v493_data));
              float v498_data = s1_w0[70];
              tensorforge::intel_esimd::simd<float, 32> v500_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v500_data + (v462_data * v498_data));
              float v503_data = s1_w0[79];
              tensorforge::intel_esimd::simd<float, 32> v505_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v505_data + (v462_data * v503_data));
              tensorforge::intel_esimd::simd<float, 32> v507_data(r5.template select<32, 1>(256));
              float v508_data = s1_w0[8];
              tensorforge::intel_esimd::simd<float, 32> v510_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v510_data + (v507_data * v508_data));
              float v513_data = s1_w0[17];
              tensorforge::intel_esimd::simd<float, 32> v515_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v515_data + (v507_data * v513_data));
              float v518_data = s1_w0[26];
              tensorforge::intel_esimd::simd<float, 32> v520_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v520_data + (v507_data * v518_data));
              float v523_data = s1_w0[35];
              tensorforge::intel_esimd::simd<float, 32> v525_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v525_data + (v507_data * v523_data));
              float v528_data = s1_w0[44];
              tensorforge::intel_esimd::simd<float, 32> v530_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v530_data + (v507_data * v528_data));
              float v533_data = s1_w0[53];
              tensorforge::intel_esimd::simd<float, 32> v535_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v535_data + (v507_data * v533_data));
              float v538_data = s1_w0[62];
              tensorforge::intel_esimd::simd<float, 32> v540_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v540_data + (v507_data * v538_data));
              float v543_data = s1_w0[71];
              tensorforge::intel_esimd::simd<float, 32> v545_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v545_data + (v507_data * v543_data));
              float v548_data = s1_w0[80];
              tensorforge::intel_esimd::simd<float, 32> v550_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v550_data + (v507_data * v548_data));
              // r6 = ir6
              #pragma unroll
              for (int32_t v552_n0 = 0; v552_n0 < 1; ++v552_n0) {
                int32_t v554_a = v552_n0 * 32;
                #pragma unroll
                for (int32_t v553_n1 = 0; v553_n1 < 9; ++v553_n1) {
                  int32_t v556_a = v554_a + (v553_n1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v557_data(ir6.template select<32, 1>(v556_a));
                  r6.template select<32, 1>(v556_a) = v557_data;
                }
              }
              // glb_m3 = store{r>g}(r6);
              #pragma unroll
              for (int32_t v558_i0 = 0; v558_i0 < 1; ++v558_i0) {
                int32_t v560_a = v558_i0 * 32;
                #pragma unroll
                for (int32_t v559_i1 = 0; v559_i1 < 9; ++v559_i1) {
                  int32_t v562_a = v560_a + (v559_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v563_data(r6.template select<32, 1>(v562_a));
                  v563_data.copy_to(glb_m3 + (v562_a));
                }
              }
            }
          }
        }
      }
    });
  });
}

