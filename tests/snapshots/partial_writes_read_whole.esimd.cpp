// === base name ===
kernel_b9f58d4d311cccd0

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b9f58d4d311cccd0 = {{1, 8, 1}, 32, 32, 1, 8, 3072, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b9f58d4d311cccd0(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b9f58d4d311cccd0(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b9f58d4d311cccd0(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_b9f58d4d311cccd0(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b9f58d4d311cccd0(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_b9f58d4d311cccd0(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_b9f58d4d311cccd0(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<768 * sizeof(float)>(); {
        using namespace tensorforge::literals;
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
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[1,8,1],"cooperative":false,"lead_width":1,"mults_per_block":8,"persistent":true,"sections":[{"barrier":false,"mults_per_block":8,"shared_elements":768}],"shared_bytes":3072,"shared_elements":768,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"pointer_based","alias":"Q","bbox":[[0,0],[32,9]],"name":"m0","ordered":false,"parts":1,"shape":[32,9],"variant":false},{"addressing":"pointer_based","alias":"F0","bbox":[[0,0],[16,9]],"name":"m1","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"pointer_based","alias":"F1","bbox":[[0,0],[16,9]],"name":"m2","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"pointer_based","alias":"O","bbox":[[0,0],[32,9]],"name":"m3","ordered":false,"parts":1,"shape":[32,9],"variant":false},{"addressing":"pointer_based","alias":"M","bbox":[[0,0],[9,9]],"name":"m4","ordered":false,"parts":1,"shape":[9,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[32,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[32,9]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (96 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (96);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v11_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v11_batchId0 < numElements0; v11_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v12_ahead1 = v11_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v11_batchId0][0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v11_batchId0][0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v11_batchId0][0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v11_batchId0][0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v11_batchId0][0 + m4_extraOffset];
              tensorforge::intel_esimd::simd<float, 288> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v25_i0 = 0; v25_i0 < 1; ++v25_i0) {
                int32_t v27_lead = v25_i0 * 32;
                #pragma unroll
                for (int32_t v26_i1 = 0; v26_i1 < 9; ++v26_i1) {
                  int32_t v30_a = v27_lead + (v26_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v31_data;
                  v31_data.copy_from(glb_m0 + (v30_a));
                  r0.template select<32, 1>(v30_a) = v31_data;
                }
              }
              tensorforge::intel_esimd::simd<float, 288> r2(0.0f);
              // r2 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v34_i1 = 0; v34_i1 < 9; ++v34_i1) {
                tensorforge::intel_esimd::simd<float, 16> v39_data;
                v39_data.copy_from(glb_m1 + ((v34_i1 * 16)));
                r2.template select<16, 1>((v34_i1 * 32)) = v39_data;
              }
              // wait(r0 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<float, 288> r1(0.0f);
              // r1 = +(r0) + None
              // [(0, 32), (0, 9)] []
              tensorforge::intel_esimd::simd<float, 32> v43_data(r0.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 32> v44_data(r1.template select<32, 1>(0));
              r1.template select<32, 1>(0) = (v44_data + v43_data);
              tensorforge::intel_esimd::simd<float, 32> v46_data(r0.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v47_data(r1.template select<32, 1>(32));
              r1.template select<32, 1>(32) = (v47_data + v46_data);
              tensorforge::intel_esimd::simd<float, 32> v49_data(r0.template select<32, 1>(64));
              tensorforge::intel_esimd::simd<float, 32> v50_data(r1.template select<32, 1>(64));
              r1.template select<32, 1>(64) = (v50_data + v49_data);
              tensorforge::intel_esimd::simd<float, 32> v52_data(r0.template select<32, 1>(96));
              tensorforge::intel_esimd::simd<float, 32> v53_data(r1.template select<32, 1>(96));
              r1.template select<32, 1>(96) = (v53_data + v52_data);
              tensorforge::intel_esimd::simd<float, 32> v55_data(r0.template select<32, 1>(128));
              tensorforge::intel_esimd::simd<float, 32> v56_data(r1.template select<32, 1>(128));
              r1.template select<32, 1>(128) = (v56_data + v55_data);
              tensorforge::intel_esimd::simd<float, 32> v58_data(r0.template select<32, 1>(160));
              tensorforge::intel_esimd::simd<float, 32> v59_data(r1.template select<32, 1>(160));
              r1.template select<32, 1>(160) = (v59_data + v58_data);
              tensorforge::intel_esimd::simd<float, 32> v61_data(r0.template select<32, 1>(192));
              tensorforge::intel_esimd::simd<float, 32> v62_data(r1.template select<32, 1>(192));
              r1.template select<32, 1>(192) = (v62_data + v61_data);
              tensorforge::intel_esimd::simd<float, 32> v64_data(r0.template select<32, 1>(224));
              tensorforge::intel_esimd::simd<float, 32> v65_data(r1.template select<32, 1>(224));
              r1.template select<32, 1>(224) = (v65_data + v64_data);
              tensorforge::intel_esimd::simd<float, 32> v67_data(r0.template select<32, 1>(256));
              tensorforge::intel_esimd::simd<float, 32> v68_data(r1.template select<32, 1>(256));
              r1.template select<32, 1>(256) = (v68_data + v67_data);
              tensorforge::intel_esimd::simd<float, 288> r4(0.0f);
              // r4 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v71_i1 = 0; v71_i1 < 9; ++v71_i1) {
                tensorforge::intel_esimd::simd<float, 16> v76_data;
                v76_data.copy_from(glb_m2 + ((v71_i1 * 16)));
                r4.template select<16, 1>((v71_i1 * 32)) = v76_data;
              }
              // wait(r2 = load{g>r}(glb_m1););
              tensorforge::intel_esimd::simd<float, 288> r3(0.0f);
              // ir3 = +(r2)
              // [(0, 16), (0, 9)] []
              tensorforge::intel_esimd::simd<float, 288> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v81_data(r2.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 32> v82_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v82_data + v81_data);
              tensorforge::intel_esimd::simd<float, 32> v84_data(r2.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v85_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v85_data + v84_data);
              tensorforge::intel_esimd::simd<float, 32> v87_data(r2.template select<32, 1>(64));
              tensorforge::intel_esimd::simd<float, 32> v88_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v88_data + v87_data);
              tensorforge::intel_esimd::simd<float, 32> v90_data(r2.template select<32, 1>(96));
              tensorforge::intel_esimd::simd<float, 32> v91_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v91_data + v90_data);
              tensorforge::intel_esimd::simd<float, 32> v93_data(r2.template select<32, 1>(128));
              tensorforge::intel_esimd::simd<float, 32> v94_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v94_data + v93_data);
              tensorforge::intel_esimd::simd<float, 32> v96_data(r2.template select<32, 1>(160));
              tensorforge::intel_esimd::simd<float, 32> v97_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v97_data + v96_data);
              tensorforge::intel_esimd::simd<float, 32> v99_data(r2.template select<32, 1>(192));
              tensorforge::intel_esimd::simd<float, 32> v100_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v100_data + v99_data);
              tensorforge::intel_esimd::simd<float, 32> v102_data(r2.template select<32, 1>(224));
              tensorforge::intel_esimd::simd<float, 32> v103_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v103_data + v102_data);
              tensorforge::intel_esimd::simd<float, 32> v105_data(r2.template select<32, 1>(256));
              tensorforge::intel_esimd::simd<float, 32> v106_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v106_data + v105_data);
              // r3 = ir3 + r1
              #pragma unroll
              for (int32_t v108_n1 = 0; v108_n1 < 9; ++v108_n1) {
                int32_t v109_a = v108_n1 * 32;
                tensorforge::intel_esimd::simd<float, 32> v111_data(ir3.template select<32, 1>(v109_a));
                tensorforge::intel_esimd::simd<float, 32> v112_data(r1.template select<32, 1>(v109_a));
                r3.template select<32, 1>(v109_a) = (v112_data + v111_data);
              }
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v114_ld;
              v114_ld.copy_from(glb_m4 + (0 + 0 + 2 * 0 + 0));
              tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 2 * 0 + 0), v114_ld);
              tensorforge::intel_esimd::simd<float, 17> v115_ld;
              v115_ld.copy_from(glb_m4 + (0 + 0 + 1 * 0 + 64));
              tensorforge::slmStore<float, 17>(s1 + (0 + 0 + 1 * 0 + 64), v115_ld);
              // wait(r4 = load{g>r}(glb_m2););
              tensorforge::intel_esimd::simd<float, 288> r5(0.0f);
              // ir5 = +(r4)
              // [(0, 16), (0, 9)] []
              tensorforge::intel_esimd::simd<float, 288> ir5(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v118_data(r4.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 32> v119_data(ir5.template select<32, 1>(0));
              ir5.template select<32, 1>(0) = (v119_data + v118_data);
              tensorforge::intel_esimd::simd<float, 32> v121_data(r4.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v122_data(ir5.template select<32, 1>(32));
              ir5.template select<32, 1>(32) = (v122_data + v121_data);
              tensorforge::intel_esimd::simd<float, 32> v124_data(r4.template select<32, 1>(64));
              tensorforge::intel_esimd::simd<float, 32> v125_data(ir5.template select<32, 1>(64));
              ir5.template select<32, 1>(64) = (v125_data + v124_data);
              tensorforge::intel_esimd::simd<float, 32> v127_data(r4.template select<32, 1>(96));
              tensorforge::intel_esimd::simd<float, 32> v128_data(ir5.template select<32, 1>(96));
              ir5.template select<32, 1>(96) = (v128_data + v127_data);
              tensorforge::intel_esimd::simd<float, 32> v130_data(r4.template select<32, 1>(128));
              tensorforge::intel_esimd::simd<float, 32> v131_data(ir5.template select<32, 1>(128));
              ir5.template select<32, 1>(128) = (v131_data + v130_data);
              tensorforge::intel_esimd::simd<float, 32> v133_data(r4.template select<32, 1>(160));
              tensorforge::intel_esimd::simd<float, 32> v134_data(ir5.template select<32, 1>(160));
              ir5.template select<32, 1>(160) = (v134_data + v133_data);
              tensorforge::intel_esimd::simd<float, 32> v136_data(r4.template select<32, 1>(192));
              tensorforge::intel_esimd::simd<float, 32> v137_data(ir5.template select<32, 1>(192));
              ir5.template select<32, 1>(192) = (v137_data + v136_data);
              tensorforge::intel_esimd::simd<float, 32> v139_data(r4.template select<32, 1>(224));
              tensorforge::intel_esimd::simd<float, 32> v140_data(ir5.template select<32, 1>(224));
              ir5.template select<32, 1>(224) = (v140_data + v139_data);
              tensorforge::intel_esimd::simd<float, 32> v142_data(r4.template select<32, 1>(256));
              tensorforge::intel_esimd::simd<float, 32> v143_data(ir5.template select<32, 1>(256));
              ir5.template select<32, 1>(256) = (v143_data + v142_data);
              // r5 = ir5 + r3
              #pragma unroll
              for (int32_t v145_n1 = 0; v145_n1 < 9; ++v145_n1) {
                int32_t v146_a = v145_n1 * 32;
                tensorforge::intel_esimd::simd<float, 32> v148_data(ir5.template select<32, 1>(v146_a));
                tensorforge::intel_esimd::simd<float, 32> v149_data(r3.template select<32, 1>(v146_a));
                r5.template select<32, 1>(v146_a) = (v149_data + v148_data);
              }
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              tensorforge::intel_esimd::simd<float, 288> r6(0.0f);
              // ir6 = +(r5 * s1)
              // [(0, 32), (0, 9)] [(0, 9)]
              tensorforge::intel_esimd::simd<float, 288> ir6(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v153_data(r5.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 96> s1_w0 = tensorforge::slmLoad<float, 96>(s1 + 0);
              float v154_data = s1_w0[0];
              tensorforge::intel_esimd::simd<float, 32> v156_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v156_data + (v153_data * v154_data));
              float v159_data = s1_w0[9];
              tensorforge::intel_esimd::simd<float, 32> v161_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v161_data + (v153_data * v159_data));
              float v164_data = s1_w0[18];
              tensorforge::intel_esimd::simd<float, 32> v166_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v166_data + (v153_data * v164_data));
              float v169_data = s1_w0[27];
              tensorforge::intel_esimd::simd<float, 32> v171_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v171_data + (v153_data * v169_data));
              float v174_data = s1_w0[36];
              tensorforge::intel_esimd::simd<float, 32> v176_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v176_data + (v153_data * v174_data));
              float v179_data = s1_w0[45];
              tensorforge::intel_esimd::simd<float, 32> v181_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v181_data + (v153_data * v179_data));
              float v184_data = s1_w0[54];
              tensorforge::intel_esimd::simd<float, 32> v186_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v186_data + (v153_data * v184_data));
              float v189_data = s1_w0[63];
              tensorforge::intel_esimd::simd<float, 32> v191_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v191_data + (v153_data * v189_data));
              float v194_data = s1_w0[72];
              tensorforge::intel_esimd::simd<float, 32> v196_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v196_data + (v153_data * v194_data));
              tensorforge::intel_esimd::simd<float, 32> v198_data(r5.template select<32, 1>(32));
              float v199_data = s1_w0[1];
              tensorforge::intel_esimd::simd<float, 32> v201_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v201_data + (v198_data * v199_data));
              float v204_data = s1_w0[10];
              tensorforge::intel_esimd::simd<float, 32> v206_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v206_data + (v198_data * v204_data));
              float v209_data = s1_w0[19];
              tensorforge::intel_esimd::simd<float, 32> v211_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v211_data + (v198_data * v209_data));
              float v214_data = s1_w0[28];
              tensorforge::intel_esimd::simd<float, 32> v216_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v216_data + (v198_data * v214_data));
              float v219_data = s1_w0[37];
              tensorforge::intel_esimd::simd<float, 32> v221_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v221_data + (v198_data * v219_data));
              float v224_data = s1_w0[46];
              tensorforge::intel_esimd::simd<float, 32> v226_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v226_data + (v198_data * v224_data));
              float v229_data = s1_w0[55];
              tensorforge::intel_esimd::simd<float, 32> v231_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v231_data + (v198_data * v229_data));
              float v234_data = s1_w0[64];
              tensorforge::intel_esimd::simd<float, 32> v236_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v236_data + (v198_data * v234_data));
              float v239_data = s1_w0[73];
              tensorforge::intel_esimd::simd<float, 32> v241_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v241_data + (v198_data * v239_data));
              tensorforge::intel_esimd::simd<float, 32> v243_data(r5.template select<32, 1>(64));
              float v244_data = s1_w0[2];
              tensorforge::intel_esimd::simd<float, 32> v246_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v246_data + (v243_data * v244_data));
              float v249_data = s1_w0[11];
              tensorforge::intel_esimd::simd<float, 32> v251_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v251_data + (v243_data * v249_data));
              float v254_data = s1_w0[20];
              tensorforge::intel_esimd::simd<float, 32> v256_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v256_data + (v243_data * v254_data));
              float v259_data = s1_w0[29];
              tensorforge::intel_esimd::simd<float, 32> v261_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v261_data + (v243_data * v259_data));
              float v264_data = s1_w0[38];
              tensorforge::intel_esimd::simd<float, 32> v266_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v266_data + (v243_data * v264_data));
              float v269_data = s1_w0[47];
              tensorforge::intel_esimd::simd<float, 32> v271_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v271_data + (v243_data * v269_data));
              float v274_data = s1_w0[56];
              tensorforge::intel_esimd::simd<float, 32> v276_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v276_data + (v243_data * v274_data));
              float v279_data = s1_w0[65];
              tensorforge::intel_esimd::simd<float, 32> v281_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v281_data + (v243_data * v279_data));
              float v284_data = s1_w0[74];
              tensorforge::intel_esimd::simd<float, 32> v286_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v286_data + (v243_data * v284_data));
              tensorforge::intel_esimd::simd<float, 32> v288_data(r5.template select<32, 1>(96));
              float v289_data = s1_w0[3];
              tensorforge::intel_esimd::simd<float, 32> v291_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v291_data + (v288_data * v289_data));
              float v294_data = s1_w0[12];
              tensorforge::intel_esimd::simd<float, 32> v296_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v296_data + (v288_data * v294_data));
              float v299_data = s1_w0[21];
              tensorforge::intel_esimd::simd<float, 32> v301_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v301_data + (v288_data * v299_data));
              float v304_data = s1_w0[30];
              tensorforge::intel_esimd::simd<float, 32> v306_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v306_data + (v288_data * v304_data));
              float v309_data = s1_w0[39];
              tensorforge::intel_esimd::simd<float, 32> v311_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v311_data + (v288_data * v309_data));
              float v314_data = s1_w0[48];
              tensorforge::intel_esimd::simd<float, 32> v316_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v316_data + (v288_data * v314_data));
              float v319_data = s1_w0[57];
              tensorforge::intel_esimd::simd<float, 32> v321_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v321_data + (v288_data * v319_data));
              float v324_data = s1_w0[66];
              tensorforge::intel_esimd::simd<float, 32> v326_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v326_data + (v288_data * v324_data));
              float v329_data = s1_w0[75];
              tensorforge::intel_esimd::simd<float, 32> v331_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v331_data + (v288_data * v329_data));
              tensorforge::intel_esimd::simd<float, 32> v333_data(r5.template select<32, 1>(128));
              float v334_data = s1_w0[4];
              tensorforge::intel_esimd::simd<float, 32> v336_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v336_data + (v333_data * v334_data));
              float v339_data = s1_w0[13];
              tensorforge::intel_esimd::simd<float, 32> v341_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v341_data + (v333_data * v339_data));
              float v344_data = s1_w0[22];
              tensorforge::intel_esimd::simd<float, 32> v346_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v346_data + (v333_data * v344_data));
              float v349_data = s1_w0[31];
              tensorforge::intel_esimd::simd<float, 32> v351_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v351_data + (v333_data * v349_data));
              float v354_data = s1_w0[40];
              tensorforge::intel_esimd::simd<float, 32> v356_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v356_data + (v333_data * v354_data));
              float v359_data = s1_w0[49];
              tensorforge::intel_esimd::simd<float, 32> v361_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v361_data + (v333_data * v359_data));
              float v364_data = s1_w0[58];
              tensorforge::intel_esimd::simd<float, 32> v366_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v366_data + (v333_data * v364_data));
              float v369_data = s1_w0[67];
              tensorforge::intel_esimd::simd<float, 32> v371_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v371_data + (v333_data * v369_data));
              float v374_data = s1_w0[76];
              tensorforge::intel_esimd::simd<float, 32> v376_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v376_data + (v333_data * v374_data));
              tensorforge::intel_esimd::simd<float, 32> v378_data(r5.template select<32, 1>(160));
              float v379_data = s1_w0[5];
              tensorforge::intel_esimd::simd<float, 32> v381_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v381_data + (v378_data * v379_data));
              float v384_data = s1_w0[14];
              tensorforge::intel_esimd::simd<float, 32> v386_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v386_data + (v378_data * v384_data));
              float v389_data = s1_w0[23];
              tensorforge::intel_esimd::simd<float, 32> v391_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v391_data + (v378_data * v389_data));
              float v394_data = s1_w0[32];
              tensorforge::intel_esimd::simd<float, 32> v396_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v396_data + (v378_data * v394_data));
              float v399_data = s1_w0[41];
              tensorforge::intel_esimd::simd<float, 32> v401_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v401_data + (v378_data * v399_data));
              float v404_data = s1_w0[50];
              tensorforge::intel_esimd::simd<float, 32> v406_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v406_data + (v378_data * v404_data));
              float v409_data = s1_w0[59];
              tensorforge::intel_esimd::simd<float, 32> v411_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v411_data + (v378_data * v409_data));
              float v414_data = s1_w0[68];
              tensorforge::intel_esimd::simd<float, 32> v416_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v416_data + (v378_data * v414_data));
              float v419_data = s1_w0[77];
              tensorforge::intel_esimd::simd<float, 32> v421_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v421_data + (v378_data * v419_data));
              tensorforge::intel_esimd::simd<float, 32> v423_data(r5.template select<32, 1>(192));
              float v424_data = s1_w0[6];
              tensorforge::intel_esimd::simd<float, 32> v426_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v426_data + (v423_data * v424_data));
              float v429_data = s1_w0[15];
              tensorforge::intel_esimd::simd<float, 32> v431_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v431_data + (v423_data * v429_data));
              float v434_data = s1_w0[24];
              tensorforge::intel_esimd::simd<float, 32> v436_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v436_data + (v423_data * v434_data));
              float v439_data = s1_w0[33];
              tensorforge::intel_esimd::simd<float, 32> v441_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v441_data + (v423_data * v439_data));
              float v444_data = s1_w0[42];
              tensorforge::intel_esimd::simd<float, 32> v446_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v446_data + (v423_data * v444_data));
              float v449_data = s1_w0[51];
              tensorforge::intel_esimd::simd<float, 32> v451_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v451_data + (v423_data * v449_data));
              float v454_data = s1_w0[60];
              tensorforge::intel_esimd::simd<float, 32> v456_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v456_data + (v423_data * v454_data));
              float v459_data = s1_w0[69];
              tensorforge::intel_esimd::simd<float, 32> v461_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v461_data + (v423_data * v459_data));
              float v464_data = s1_w0[78];
              tensorforge::intel_esimd::simd<float, 32> v466_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v466_data + (v423_data * v464_data));
              tensorforge::intel_esimd::simd<float, 32> v468_data(r5.template select<32, 1>(224));
              float v469_data = s1_w0[7];
              tensorforge::intel_esimd::simd<float, 32> v471_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v471_data + (v468_data * v469_data));
              float v474_data = s1_w0[16];
              tensorforge::intel_esimd::simd<float, 32> v476_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v476_data + (v468_data * v474_data));
              float v479_data = s1_w0[25];
              tensorforge::intel_esimd::simd<float, 32> v481_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v481_data + (v468_data * v479_data));
              float v484_data = s1_w0[34];
              tensorforge::intel_esimd::simd<float, 32> v486_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v486_data + (v468_data * v484_data));
              float v489_data = s1_w0[43];
              tensorforge::intel_esimd::simd<float, 32> v491_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v491_data + (v468_data * v489_data));
              float v494_data = s1_w0[52];
              tensorforge::intel_esimd::simd<float, 32> v496_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v496_data + (v468_data * v494_data));
              float v499_data = s1_w0[61];
              tensorforge::intel_esimd::simd<float, 32> v501_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v501_data + (v468_data * v499_data));
              float v504_data = s1_w0[70];
              tensorforge::intel_esimd::simd<float, 32> v506_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v506_data + (v468_data * v504_data));
              float v509_data = s1_w0[79];
              tensorforge::intel_esimd::simd<float, 32> v511_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v511_data + (v468_data * v509_data));
              tensorforge::intel_esimd::simd<float, 32> v513_data(r5.template select<32, 1>(256));
              float v514_data = s1_w0[8];
              tensorforge::intel_esimd::simd<float, 32> v516_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v516_data + (v513_data * v514_data));
              float v519_data = s1_w0[17];
              tensorforge::intel_esimd::simd<float, 32> v521_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v521_data + (v513_data * v519_data));
              float v524_data = s1_w0[26];
              tensorforge::intel_esimd::simd<float, 32> v526_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v526_data + (v513_data * v524_data));
              float v529_data = s1_w0[35];
              tensorforge::intel_esimd::simd<float, 32> v531_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v531_data + (v513_data * v529_data));
              float v534_data = s1_w0[44];
              tensorforge::intel_esimd::simd<float, 32> v536_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v536_data + (v513_data * v534_data));
              float v539_data = s1_w0[53];
              tensorforge::intel_esimd::simd<float, 32> v541_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v541_data + (v513_data * v539_data));
              float v544_data = s1_w0[62];
              tensorforge::intel_esimd::simd<float, 32> v546_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v546_data + (v513_data * v544_data));
              float v549_data = s1_w0[71];
              tensorforge::intel_esimd::simd<float, 32> v551_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v551_data + (v513_data * v549_data));
              float v554_data = s1_w0[80];
              tensorforge::intel_esimd::simd<float, 32> v556_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v556_data + (v513_data * v554_data));
              // r6 = ir6
              #pragma unroll
              for (int32_t v558_n0 = 0; v558_n0 < 1; ++v558_n0) {
                int32_t v560_a = v558_n0 * 32;
                #pragma unroll
                for (int32_t v559_n1 = 0; v559_n1 < 9; ++v559_n1) {
                  int32_t v562_a = v560_a + (v559_n1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v563_data(ir6.template select<32, 1>(v562_a));
                  r6.template select<32, 1>(v562_a) = v563_data;
                }
              }
              // glb_m3 = store{r>g}(r6);
              #pragma unroll
              for (int32_t v564_i0 = 0; v564_i0 < 1; ++v564_i0) {
                int32_t v566_a = v564_i0 * 32;
                #pragma unroll
                for (int32_t v565_i1 = 0; v565_i1 < 9; ++v565_i1) {
                  int32_t v568_a = v566_a + (v565_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v569_data(r6.template select<32, 1>(v568_a));
                  v569_data.copy_to(glb_m3 + (v568_a));
                }
              }
            }
          }
        }
      }
    });
  });
}

