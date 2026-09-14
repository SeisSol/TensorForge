// === base name ===
kernel_7634d6f45b441208

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_7634d6f45b441208 = {{1, 8, 1}, 32, 32, 1, 8, 12288, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_7634d6f45b441208(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_7634d6f45b441208(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_7634d6f45b441208(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 3072 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_7634d6f45b441208(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_7634d6f45b441208(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_7634d6f45b441208(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_7634d6f45b441208(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<3072 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes x 8 per block = block 1x8x1, 12288 B shared, occupancy grid
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
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[1,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":3072}],"shared_bytes":12288,"shared_elements":3072,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"pointer_based","alias":"Q","bbox":[[0,0],[32,9]],"name":"m0","ordered":false,"parts":1,"shape":[32,9],"variant":false},{"addressing":"pointer_based","alias":"F0","bbox":[[0,0],[16,9]],"name":"m1","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"pointer_based","alias":"F1","bbox":[[0,0],[16,9]],"name":"m2","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"pointer_based","alias":"O","bbox":[[0,0],[32,9]],"name":"m3","ordered":false,"parts":1,"shape":[32,9],"variant":false},{"addressing":"pointer_based","alias":"M","bbox":[[0,0],[9,9]],"name":"m4","ordered":false,"parts":1,"shape":[9,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[32,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[32,9]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (384 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (384);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (96);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v6_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v6_batchId0 < numElements0; v6_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v7_ahead1 = v6_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v6_batchId0][0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v6_batchId0][0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v6_batchId0][0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v6_batchId0][0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v6_batchId0][0 + m4_extraOffset];
              tensorforge::intel_esimd::simd<float, 288> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v20_i0 = 0; v20_i0 < 1; ++v20_i0) {
                int32_t v22_lead = v20_i0 * 32;
                #pragma unroll
                for (int32_t v21_i1 = 0; v21_i1 < 9; ++v21_i1) {
                  int32_t v25_a = v22_lead + (v21_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v26_data;
                  v26_data.copy_from(glb_m0 + (v25_a));
                  r0.template select<32, 1>(v25_a) = v26_data;
                }
              }
              tensorforge::intel_esimd::simd<float, 288> r2(0.0f);
              // r2 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v29_i1 = 0; v29_i1 < 9; ++v29_i1) {
                tensorforge::intel_esimd::simd<float, 16> v34_data;
                v34_data.copy_from(glb_m1 + ((v29_i1 * 16)));
                r2.template select<16, 1>((v29_i1 * 32)) = v34_data;
              }
              // wait(r0 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<float, 288> r1(0.0f);
              // r1 = +(r0) + None
              // [(0, 32), (0, 9)] []
              tensorforge::intel_esimd::simd<float, 32> v38_data(r0.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 32> v39_data(r1.template select<32, 1>(0));
              r1.template select<32, 1>(0) = (v39_data + v38_data);
              tensorforge::intel_esimd::simd<float, 32> v41_data(r0.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v42_data(r1.template select<32, 1>(32));
              r1.template select<32, 1>(32) = (v42_data + v41_data);
              tensorforge::intel_esimd::simd<float, 32> v44_data(r0.template select<32, 1>(64));
              tensorforge::intel_esimd::simd<float, 32> v45_data(r1.template select<32, 1>(64));
              r1.template select<32, 1>(64) = (v45_data + v44_data);
              tensorforge::intel_esimd::simd<float, 32> v47_data(r0.template select<32, 1>(96));
              tensorforge::intel_esimd::simd<float, 32> v48_data(r1.template select<32, 1>(96));
              r1.template select<32, 1>(96) = (v48_data + v47_data);
              tensorforge::intel_esimd::simd<float, 32> v50_data(r0.template select<32, 1>(128));
              tensorforge::intel_esimd::simd<float, 32> v51_data(r1.template select<32, 1>(128));
              r1.template select<32, 1>(128) = (v51_data + v50_data);
              tensorforge::intel_esimd::simd<float, 32> v53_data(r0.template select<32, 1>(160));
              tensorforge::intel_esimd::simd<float, 32> v54_data(r1.template select<32, 1>(160));
              r1.template select<32, 1>(160) = (v54_data + v53_data);
              tensorforge::intel_esimd::simd<float, 32> v56_data(r0.template select<32, 1>(192));
              tensorforge::intel_esimd::simd<float, 32> v57_data(r1.template select<32, 1>(192));
              r1.template select<32, 1>(192) = (v57_data + v56_data);
              tensorforge::intel_esimd::simd<float, 32> v59_data(r0.template select<32, 1>(224));
              tensorforge::intel_esimd::simd<float, 32> v60_data(r1.template select<32, 1>(224));
              r1.template select<32, 1>(224) = (v60_data + v59_data);
              tensorforge::intel_esimd::simd<float, 32> v62_data(r0.template select<32, 1>(256));
              tensorforge::intel_esimd::simd<float, 32> v63_data(r1.template select<32, 1>(256));
              r1.template select<32, 1>(256) = (v63_data + v62_data);
              // s0 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v65_i0 = 0; v65_i0 < 1; ++v65_i0) {
                int32_t v67_a = v65_i0 * 32;
                #pragma unroll
                for (int32_t v66_i1 = 0; v66_i1 < 9; ++v66_i1) {
                  int32_t v69_a = v67_a + (v66_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v70_data(r1.template select<32, 1>(v69_a));
                  tensorforge::slmStore<float, 32>(s0 + (v69_a), v70_data);
                }
              }
              tensorforge::intel_esimd::simd<float, 288> r4(0.0f);
              // r4 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v74_i1 = 0; v74_i1 < 9; ++v74_i1) {
                tensorforge::intel_esimd::simd<float, 16> v79_data;
                v79_data.copy_from(glb_m2 + ((v74_i1 * 16)));
                r4.template select<16, 1>((v74_i1 * 32)) = v79_data;
              }
              // wait(r2 = load{g>r}(glb_m1););
              tensorforge::intel_esimd::simd<float, 288> r3(0.0f);
              // ir3 = +(r2)
              // [(0, 16), (0, 9)] []
              tensorforge::intel_esimd::simd<float, 288> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v84_data(r2.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 32> v85_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v85_data + v84_data);
              tensorforge::intel_esimd::simd<float, 32> v87_data(r2.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v88_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v88_data + v87_data);
              tensorforge::intel_esimd::simd<float, 32> v90_data(r2.template select<32, 1>(64));
              tensorforge::intel_esimd::simd<float, 32> v91_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v91_data + v90_data);
              tensorforge::intel_esimd::simd<float, 32> v93_data(r2.template select<32, 1>(96));
              tensorforge::intel_esimd::simd<float, 32> v94_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v94_data + v93_data);
              tensorforge::intel_esimd::simd<float, 32> v96_data(r2.template select<32, 1>(128));
              tensorforge::intel_esimd::simd<float, 32> v97_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v97_data + v96_data);
              tensorforge::intel_esimd::simd<float, 32> v99_data(r2.template select<32, 1>(160));
              tensorforge::intel_esimd::simd<float, 32> v100_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v100_data + v99_data);
              tensorforge::intel_esimd::simd<float, 32> v102_data(r2.template select<32, 1>(192));
              tensorforge::intel_esimd::simd<float, 32> v103_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v103_data + v102_data);
              tensorforge::intel_esimd::simd<float, 32> v105_data(r2.template select<32, 1>(224));
              tensorforge::intel_esimd::simd<float, 32> v106_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v106_data + v105_data);
              tensorforge::intel_esimd::simd<float, 32> v108_data(r2.template select<32, 1>(256));
              tensorforge::intel_esimd::simd<float, 32> v109_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v109_data + v108_data);
              // r3 = ir3 + s0
              #pragma unroll
              for (int32_t v111_n1 = 0; v111_n1 < 9; ++v111_n1) {
                int32_t v112_a = v111_n1 * 32;
                tensorforge::intel_esimd::simd<float, 16> v114_data(ir3.template select<16, 1>(v112_a));
                tensorforge::intel_esimd::simd<float, 16> v118_data = tensorforge::slmLoad<float, 16>(s0 + (v112_a));
                r3.template select<16, 1>(v112_a) = (v118_data + v114_data);
              }
              // s0 = store{r>s}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v120_i1 = 0; v120_i1 < 9; ++v120_i1) {
                int32_t v121_a = v120_i1 * 32;
                tensorforge::intel_esimd::simd<float, 16> v123_data(r3.template select<16, 1>(v121_a));
                tensorforge::slmStore<float, 16>(s0 + (v121_a), v123_data);
              }
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v127_ld;
              v127_ld.copy_from(glb_m4 + (0 + 0 + 2 * 0 + 0));
              tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 2 * 0 + 0), v127_ld);
              tensorforge::intel_esimd::simd<float, 17> v128_ld;
              v128_ld.copy_from(glb_m4 + (0 + 0 + 1 * 0 + 64));
              tensorforge::slmStore<float, 17>(s1 + (0 + 0 + 1 * 0 + 64), v128_ld);
              // wait(r4 = load{g>r}(glb_m2););
              tensorforge::intel_esimd::simd<float, 288> r5(0.0f);
              // ir5 = +(r4)
              // [(0, 16), (0, 9)] []
              tensorforge::intel_esimd::simd<float, 288> ir5(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v131_data(r4.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 32> v132_data(ir5.template select<32, 1>(0));
              ir5.template select<32, 1>(0) = (v132_data + v131_data);
              tensorforge::intel_esimd::simd<float, 32> v134_data(r4.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v135_data(ir5.template select<32, 1>(32));
              ir5.template select<32, 1>(32) = (v135_data + v134_data);
              tensorforge::intel_esimd::simd<float, 32> v137_data(r4.template select<32, 1>(64));
              tensorforge::intel_esimd::simd<float, 32> v138_data(ir5.template select<32, 1>(64));
              ir5.template select<32, 1>(64) = (v138_data + v137_data);
              tensorforge::intel_esimd::simd<float, 32> v140_data(r4.template select<32, 1>(96));
              tensorforge::intel_esimd::simd<float, 32> v141_data(ir5.template select<32, 1>(96));
              ir5.template select<32, 1>(96) = (v141_data + v140_data);
              tensorforge::intel_esimd::simd<float, 32> v143_data(r4.template select<32, 1>(128));
              tensorforge::intel_esimd::simd<float, 32> v144_data(ir5.template select<32, 1>(128));
              ir5.template select<32, 1>(128) = (v144_data + v143_data);
              tensorforge::intel_esimd::simd<float, 32> v146_data(r4.template select<32, 1>(160));
              tensorforge::intel_esimd::simd<float, 32> v147_data(ir5.template select<32, 1>(160));
              ir5.template select<32, 1>(160) = (v147_data + v146_data);
              tensorforge::intel_esimd::simd<float, 32> v149_data(r4.template select<32, 1>(192));
              tensorforge::intel_esimd::simd<float, 32> v150_data(ir5.template select<32, 1>(192));
              ir5.template select<32, 1>(192) = (v150_data + v149_data);
              tensorforge::intel_esimd::simd<float, 32> v152_data(r4.template select<32, 1>(224));
              tensorforge::intel_esimd::simd<float, 32> v153_data(ir5.template select<32, 1>(224));
              ir5.template select<32, 1>(224) = (v153_data + v152_data);
              tensorforge::intel_esimd::simd<float, 32> v155_data(r4.template select<32, 1>(256));
              tensorforge::intel_esimd::simd<float, 32> v156_data(ir5.template select<32, 1>(256));
              ir5.template select<32, 1>(256) = (v156_data + v155_data);
              // r5 = ir5 + s0
              #pragma unroll
              for (int32_t v158_n1 = 0; v158_n1 < 9; ++v158_n1) {
                int32_t v159_a = v158_n1 * 32;
                tensorforge::intel_esimd::simd<float, 16> v161_data(ir5.template select<16, 1>(v159_a));
                tensorforge::intel_esimd::simd<float, 16> v165_data = tensorforge::slmLoad<float, 16>(s0 + (v159_a));
                r5.template select<16, 1>(v159_a) = (v165_data + v161_data);
              }
              // s0 = store{r>s}(localShrMem0, r5);
              #pragma unroll
              for (int32_t v167_i1 = 0; v167_i1 < 9; ++v167_i1) {
                int32_t v168_a = v167_i1 * 32;
                tensorforge::intel_esimd::simd<float, 16> v170_data(r5.template select<16, 1>(v168_a));
                tensorforge::slmStore<float, 16>(s0 + (v168_a), v170_data);
              }
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              tensorforge::intel_esimd::simd<float, 288> r6(0.0f);
              // ir6 = +(s0 * s1)
              // [(0, 32), (0, 9)] [(0, 9)]
              tensorforge::intel_esimd::simd<float, 288> ir6(0.0f);
              tensorforge::intel_esimd::simd<float, 64> s0_run0 = tensorforge::slmLoad<float, 64>(s0 + (0_i32));
              tensorforge::intel_esimd::simd<float, 32> v179_data(s0_run0.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 96> s1_w0 = tensorforge::slmLoad<float, 96>(s1 + 0);
              float v180_data = s1_w0[0];
              tensorforge::intel_esimd::simd<float, 32> v182_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v182_data + (v179_data * v180_data));
              float v185_data = s1_w0[9];
              tensorforge::intel_esimd::simd<float, 32> v187_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v187_data + (v179_data * v185_data));
              float v190_data = s1_w0[18];
              tensorforge::intel_esimd::simd<float, 32> v192_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v192_data + (v179_data * v190_data));
              float v195_data = s1_w0[27];
              tensorforge::intel_esimd::simd<float, 32> v197_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v197_data + (v179_data * v195_data));
              float v200_data = s1_w0[36];
              tensorforge::intel_esimd::simd<float, 32> v202_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v202_data + (v179_data * v200_data));
              float v205_data = s1_w0[45];
              tensorforge::intel_esimd::simd<float, 32> v207_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v207_data + (v179_data * v205_data));
              float v210_data = s1_w0[54];
              tensorforge::intel_esimd::simd<float, 32> v212_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v212_data + (v179_data * v210_data));
              float v215_data = s1_w0[63];
              tensorforge::intel_esimd::simd<float, 32> v217_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v217_data + (v179_data * v215_data));
              float v220_data = s1_w0[72];
              tensorforge::intel_esimd::simd<float, 32> v222_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v222_data + (v179_data * v220_data));
              tensorforge::intel_esimd::simd<float, 32> v225_data(s0_run0.template select<32, 1>(32));
              float v226_data = s1_w0[1];
              tensorforge::intel_esimd::simd<float, 32> v228_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v228_data + (v225_data * v226_data));
              float v231_data = s1_w0[10];
              tensorforge::intel_esimd::simd<float, 32> v233_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v233_data + (v225_data * v231_data));
              float v236_data = s1_w0[19];
              tensorforge::intel_esimd::simd<float, 32> v238_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v238_data + (v225_data * v236_data));
              float v241_data = s1_w0[28];
              tensorforge::intel_esimd::simd<float, 32> v243_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v243_data + (v225_data * v241_data));
              float v246_data = s1_w0[37];
              tensorforge::intel_esimd::simd<float, 32> v248_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v248_data + (v225_data * v246_data));
              float v251_data = s1_w0[46];
              tensorforge::intel_esimd::simd<float, 32> v253_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v253_data + (v225_data * v251_data));
              float v256_data = s1_w0[55];
              tensorforge::intel_esimd::simd<float, 32> v258_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v258_data + (v225_data * v256_data));
              float v261_data = s1_w0[64];
              tensorforge::intel_esimd::simd<float, 32> v263_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v263_data + (v225_data * v261_data));
              float v266_data = s1_w0[73];
              tensorforge::intel_esimd::simd<float, 32> v268_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v268_data + (v225_data * v266_data));
              tensorforge::intel_esimd::simd<float, 64> s0_run1 = tensorforge::slmLoad<float, 64>(s0 + (64_i32));
              tensorforge::intel_esimd::simd<float, 32> v271_data(s0_run1.template select<32, 1>(0));
              float v272_data = s1_w0[2];
              tensorforge::intel_esimd::simd<float, 32> v274_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v274_data + (v271_data * v272_data));
              float v277_data = s1_w0[11];
              tensorforge::intel_esimd::simd<float, 32> v279_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v279_data + (v271_data * v277_data));
              float v282_data = s1_w0[20];
              tensorforge::intel_esimd::simd<float, 32> v284_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v284_data + (v271_data * v282_data));
              float v287_data = s1_w0[29];
              tensorforge::intel_esimd::simd<float, 32> v289_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v289_data + (v271_data * v287_data));
              float v292_data = s1_w0[38];
              tensorforge::intel_esimd::simd<float, 32> v294_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v294_data + (v271_data * v292_data));
              float v297_data = s1_w0[47];
              tensorforge::intel_esimd::simd<float, 32> v299_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v299_data + (v271_data * v297_data));
              float v302_data = s1_w0[56];
              tensorforge::intel_esimd::simd<float, 32> v304_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v304_data + (v271_data * v302_data));
              float v307_data = s1_w0[65];
              tensorforge::intel_esimd::simd<float, 32> v309_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v309_data + (v271_data * v307_data));
              float v312_data = s1_w0[74];
              tensorforge::intel_esimd::simd<float, 32> v314_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v314_data + (v271_data * v312_data));
              tensorforge::intel_esimd::simd<float, 32> v317_data(s0_run1.template select<32, 1>(32));
              float v318_data = s1_w0[3];
              tensorforge::intel_esimd::simd<float, 32> v320_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v320_data + (v317_data * v318_data));
              float v323_data = s1_w0[12];
              tensorforge::intel_esimd::simd<float, 32> v325_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v325_data + (v317_data * v323_data));
              float v328_data = s1_w0[21];
              tensorforge::intel_esimd::simd<float, 32> v330_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v330_data + (v317_data * v328_data));
              float v333_data = s1_w0[30];
              tensorforge::intel_esimd::simd<float, 32> v335_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v335_data + (v317_data * v333_data));
              float v338_data = s1_w0[39];
              tensorforge::intel_esimd::simd<float, 32> v340_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v340_data + (v317_data * v338_data));
              float v343_data = s1_w0[48];
              tensorforge::intel_esimd::simd<float, 32> v345_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v345_data + (v317_data * v343_data));
              float v348_data = s1_w0[57];
              tensorforge::intel_esimd::simd<float, 32> v350_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v350_data + (v317_data * v348_data));
              float v353_data = s1_w0[66];
              tensorforge::intel_esimd::simd<float, 32> v355_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v355_data + (v317_data * v353_data));
              float v358_data = s1_w0[75];
              tensorforge::intel_esimd::simd<float, 32> v360_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v360_data + (v317_data * v358_data));
              tensorforge::intel_esimd::simd<float, 64> s0_run2 = tensorforge::slmLoad<float, 64>(s0 + (128_i32));
              tensorforge::intel_esimd::simd<float, 32> v363_data(s0_run2.template select<32, 1>(0));
              float v364_data = s1_w0[4];
              tensorforge::intel_esimd::simd<float, 32> v366_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v366_data + (v363_data * v364_data));
              float v369_data = s1_w0[13];
              tensorforge::intel_esimd::simd<float, 32> v371_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v371_data + (v363_data * v369_data));
              float v374_data = s1_w0[22];
              tensorforge::intel_esimd::simd<float, 32> v376_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v376_data + (v363_data * v374_data));
              float v379_data = s1_w0[31];
              tensorforge::intel_esimd::simd<float, 32> v381_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v381_data + (v363_data * v379_data));
              float v384_data = s1_w0[40];
              tensorforge::intel_esimd::simd<float, 32> v386_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v386_data + (v363_data * v384_data));
              float v389_data = s1_w0[49];
              tensorforge::intel_esimd::simd<float, 32> v391_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v391_data + (v363_data * v389_data));
              float v394_data = s1_w0[58];
              tensorforge::intel_esimd::simd<float, 32> v396_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v396_data + (v363_data * v394_data));
              float v399_data = s1_w0[67];
              tensorforge::intel_esimd::simd<float, 32> v401_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v401_data + (v363_data * v399_data));
              float v404_data = s1_w0[76];
              tensorforge::intel_esimd::simd<float, 32> v406_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v406_data + (v363_data * v404_data));
              tensorforge::intel_esimd::simd<float, 32> v409_data(s0_run2.template select<32, 1>(32));
              float v410_data = s1_w0[5];
              tensorforge::intel_esimd::simd<float, 32> v412_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v412_data + (v409_data * v410_data));
              float v415_data = s1_w0[14];
              tensorforge::intel_esimd::simd<float, 32> v417_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v417_data + (v409_data * v415_data));
              float v420_data = s1_w0[23];
              tensorforge::intel_esimd::simd<float, 32> v422_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v422_data + (v409_data * v420_data));
              float v425_data = s1_w0[32];
              tensorforge::intel_esimd::simd<float, 32> v427_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v427_data + (v409_data * v425_data));
              float v430_data = s1_w0[41];
              tensorforge::intel_esimd::simd<float, 32> v432_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v432_data + (v409_data * v430_data));
              float v435_data = s1_w0[50];
              tensorforge::intel_esimd::simd<float, 32> v437_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v437_data + (v409_data * v435_data));
              float v440_data = s1_w0[59];
              tensorforge::intel_esimd::simd<float, 32> v442_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v442_data + (v409_data * v440_data));
              float v445_data = s1_w0[68];
              tensorforge::intel_esimd::simd<float, 32> v447_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v447_data + (v409_data * v445_data));
              float v450_data = s1_w0[77];
              tensorforge::intel_esimd::simd<float, 32> v452_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v452_data + (v409_data * v450_data));
              tensorforge::intel_esimd::simd<float, 64> s0_run3 = tensorforge::slmLoad<float, 64>(s0 + (192_i32));
              tensorforge::intel_esimd::simd<float, 32> v455_data(s0_run3.template select<32, 1>(0));
              float v456_data = s1_w0[6];
              tensorforge::intel_esimd::simd<float, 32> v458_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v458_data + (v455_data * v456_data));
              float v461_data = s1_w0[15];
              tensorforge::intel_esimd::simd<float, 32> v463_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v463_data + (v455_data * v461_data));
              float v466_data = s1_w0[24];
              tensorforge::intel_esimd::simd<float, 32> v468_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v468_data + (v455_data * v466_data));
              float v471_data = s1_w0[33];
              tensorforge::intel_esimd::simd<float, 32> v473_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v473_data + (v455_data * v471_data));
              float v476_data = s1_w0[42];
              tensorforge::intel_esimd::simd<float, 32> v478_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v478_data + (v455_data * v476_data));
              float v481_data = s1_w0[51];
              tensorforge::intel_esimd::simd<float, 32> v483_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v483_data + (v455_data * v481_data));
              float v486_data = s1_w0[60];
              tensorforge::intel_esimd::simd<float, 32> v488_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v488_data + (v455_data * v486_data));
              float v491_data = s1_w0[69];
              tensorforge::intel_esimd::simd<float, 32> v493_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v493_data + (v455_data * v491_data));
              float v496_data = s1_w0[78];
              tensorforge::intel_esimd::simd<float, 32> v498_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v498_data + (v455_data * v496_data));
              tensorforge::intel_esimd::simd<float, 32> v501_data(s0_run3.template select<32, 1>(32));
              float v502_data = s1_w0[7];
              tensorforge::intel_esimd::simd<float, 32> v504_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v504_data + (v501_data * v502_data));
              float v507_data = s1_w0[16];
              tensorforge::intel_esimd::simd<float, 32> v509_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v509_data + (v501_data * v507_data));
              float v512_data = s1_w0[25];
              tensorforge::intel_esimd::simd<float, 32> v514_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v514_data + (v501_data * v512_data));
              float v517_data = s1_w0[34];
              tensorforge::intel_esimd::simd<float, 32> v519_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v519_data + (v501_data * v517_data));
              float v522_data = s1_w0[43];
              tensorforge::intel_esimd::simd<float, 32> v524_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v524_data + (v501_data * v522_data));
              float v527_data = s1_w0[52];
              tensorforge::intel_esimd::simd<float, 32> v529_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v529_data + (v501_data * v527_data));
              float v532_data = s1_w0[61];
              tensorforge::intel_esimd::simd<float, 32> v534_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v534_data + (v501_data * v532_data));
              float v537_data = s1_w0[70];
              tensorforge::intel_esimd::simd<float, 32> v539_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v539_data + (v501_data * v537_data));
              float v542_data = s1_w0[79];
              tensorforge::intel_esimd::simd<float, 32> v544_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v544_data + (v501_data * v542_data));
              tensorforge::intel_esimd::simd<float, 32> v547_data = tensorforge::slmLoad<float, 32>(s0 + (256_i32));
              float v548_data = s1_w0[8];
              tensorforge::intel_esimd::simd<float, 32> v550_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v550_data + (v547_data * v548_data));
              float v553_data = s1_w0[17];
              tensorforge::intel_esimd::simd<float, 32> v555_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v555_data + (v547_data * v553_data));
              float v558_data = s1_w0[26];
              tensorforge::intel_esimd::simd<float, 32> v560_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v560_data + (v547_data * v558_data));
              float v563_data = s1_w0[35];
              tensorforge::intel_esimd::simd<float, 32> v565_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v565_data + (v547_data * v563_data));
              float v568_data = s1_w0[44];
              tensorforge::intel_esimd::simd<float, 32> v570_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v570_data + (v547_data * v568_data));
              float v573_data = s1_w0[53];
              tensorforge::intel_esimd::simd<float, 32> v575_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v575_data + (v547_data * v573_data));
              float v578_data = s1_w0[62];
              tensorforge::intel_esimd::simd<float, 32> v580_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v580_data + (v547_data * v578_data));
              float v583_data = s1_w0[71];
              tensorforge::intel_esimd::simd<float, 32> v585_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v585_data + (v547_data * v583_data));
              float v588_data = s1_w0[80];
              tensorforge::intel_esimd::simd<float, 32> v590_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v590_data + (v547_data * v588_data));
              // r6 = ir6
              #pragma unroll
              for (int32_t v592_n0 = 0; v592_n0 < 1; ++v592_n0) {
                int32_t v594_a = v592_n0 * 32;
                #pragma unroll
                for (int32_t v593_n1 = 0; v593_n1 < 9; ++v593_n1) {
                  int32_t v596_a = v594_a + (v593_n1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v597_data(ir6.template select<32, 1>(v596_a));
                  r6.template select<32, 1>(v596_a) = v597_data;
                }
              }
              // glb_m3 = store{r>g}(r6);
              #pragma unroll
              for (int32_t v598_i0 = 0; v598_i0 < 1; ++v598_i0) {
                int32_t v600_a = v598_i0 * 32;
                #pragma unroll
                for (int32_t v599_i1 = 0; v599_i1 < 9; ++v599_i1) {
                  int32_t v602_a = v600_a + (v599_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v603_data(r6.template select<32, 1>(v602_a));
                  v603_data.copy_to(glb_m3 + (v602_a));
                }
              }
            }
          }
        }
      }
    });
  });
}

