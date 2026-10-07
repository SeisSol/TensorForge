// === base name ===
kernel_6964fba7cdd6abda

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_6964fba7cdd6abda = {{1, 8, 1}, 32, 32, 1, 8, 3072, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_6964fba7cdd6abda(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_6964fba7cdd6abda(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_6964fba7cdd6abda(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_6964fba7cdd6abda(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_6964fba7cdd6abda(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_6964fba7cdd6abda(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_6964fba7cdd6abda(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
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
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v8_batchId0][0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v8_batchId0][0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v8_batchId0][0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v8_batchId0][0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v8_batchId0][0 + m4_extraOffset];
              tensorforge::intel_esimd::simd<float, 288> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v22_i0 = 0; v22_i0 < 1; ++v22_i0) {
                int32_t v24_lead = v22_i0 * 32;
                #pragma unroll
                for (int32_t v23_i1 = 0; v23_i1 < 9; ++v23_i1) {
                  int32_t v27_a = v24_lead + (v23_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v28_data;
                  v28_data.copy_from(glb_m0 + (v27_a));
                  r0.template select<32, 1>(v27_a) = v28_data;
                }
              }
              tensorforge::intel_esimd::simd<float, 288> r2(0.0f);
              // r2 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v31_i1 = 0; v31_i1 < 9; ++v31_i1) {
                tensorforge::intel_esimd::simd<float, 16> v36_data;
                v36_data.copy_from(glb_m1 + ((v31_i1 * 16)));
                r2.template select<16, 1>((v31_i1 * 32)) = v36_data;
              }
              // wait(r0 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<float, 288> r1(0.0f);
              // r1 = +(r0) + None
              // [(0, 32), (0, 9)] []
              tensorforge::intel_esimd::simd<float, 32> v40_data(r0.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 32> v41_data(r1.template select<32, 1>(0));
              r1.template select<32, 1>(0) = (v41_data + v40_data);
              tensorforge::intel_esimd::simd<float, 32> v43_data(r0.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v44_data(r1.template select<32, 1>(32));
              r1.template select<32, 1>(32) = (v44_data + v43_data);
              tensorforge::intel_esimd::simd<float, 32> v46_data(r0.template select<32, 1>(64));
              tensorforge::intel_esimd::simd<float, 32> v47_data(r1.template select<32, 1>(64));
              r1.template select<32, 1>(64) = (v47_data + v46_data);
              tensorforge::intel_esimd::simd<float, 32> v49_data(r0.template select<32, 1>(96));
              tensorforge::intel_esimd::simd<float, 32> v50_data(r1.template select<32, 1>(96));
              r1.template select<32, 1>(96) = (v50_data + v49_data);
              tensorforge::intel_esimd::simd<float, 32> v52_data(r0.template select<32, 1>(128));
              tensorforge::intel_esimd::simd<float, 32> v53_data(r1.template select<32, 1>(128));
              r1.template select<32, 1>(128) = (v53_data + v52_data);
              tensorforge::intel_esimd::simd<float, 32> v55_data(r0.template select<32, 1>(160));
              tensorforge::intel_esimd::simd<float, 32> v56_data(r1.template select<32, 1>(160));
              r1.template select<32, 1>(160) = (v56_data + v55_data);
              tensorforge::intel_esimd::simd<float, 32> v58_data(r0.template select<32, 1>(192));
              tensorforge::intel_esimd::simd<float, 32> v59_data(r1.template select<32, 1>(192));
              r1.template select<32, 1>(192) = (v59_data + v58_data);
              tensorforge::intel_esimd::simd<float, 32> v61_data(r0.template select<32, 1>(224));
              tensorforge::intel_esimd::simd<float, 32> v62_data(r1.template select<32, 1>(224));
              r1.template select<32, 1>(224) = (v62_data + v61_data);
              tensorforge::intel_esimd::simd<float, 32> v64_data(r0.template select<32, 1>(256));
              tensorforge::intel_esimd::simd<float, 32> v65_data(r1.template select<32, 1>(256));
              r1.template select<32, 1>(256) = (v65_data + v64_data);
              tensorforge::intel_esimd::simd<float, 288> r4(0.0f);
              // r4 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v68_i1 = 0; v68_i1 < 9; ++v68_i1) {
                tensorforge::intel_esimd::simd<float, 16> v73_data;
                v73_data.copy_from(glb_m2 + ((v68_i1 * 16)));
                r4.template select<16, 1>((v68_i1 * 32)) = v73_data;
              }
              // wait(r2 = load{g>r}(glb_m1););
              tensorforge::intel_esimd::simd<float, 288> r3(0.0f);
              // ir3 = +(r2)
              // [(0, 16), (0, 9)] []
              tensorforge::intel_esimd::simd<float, 288> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v78_data(r2.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 32> v79_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v79_data + v78_data);
              tensorforge::intel_esimd::simd<float, 32> v81_data(r2.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v82_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v82_data + v81_data);
              tensorforge::intel_esimd::simd<float, 32> v84_data(r2.template select<32, 1>(64));
              tensorforge::intel_esimd::simd<float, 32> v85_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v85_data + v84_data);
              tensorforge::intel_esimd::simd<float, 32> v87_data(r2.template select<32, 1>(96));
              tensorforge::intel_esimd::simd<float, 32> v88_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v88_data + v87_data);
              tensorforge::intel_esimd::simd<float, 32> v90_data(r2.template select<32, 1>(128));
              tensorforge::intel_esimd::simd<float, 32> v91_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v91_data + v90_data);
              tensorforge::intel_esimd::simd<float, 32> v93_data(r2.template select<32, 1>(160));
              tensorforge::intel_esimd::simd<float, 32> v94_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v94_data + v93_data);
              tensorforge::intel_esimd::simd<float, 32> v96_data(r2.template select<32, 1>(192));
              tensorforge::intel_esimd::simd<float, 32> v97_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v97_data + v96_data);
              tensorforge::intel_esimd::simd<float, 32> v99_data(r2.template select<32, 1>(224));
              tensorforge::intel_esimd::simd<float, 32> v100_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v100_data + v99_data);
              tensorforge::intel_esimd::simd<float, 32> v102_data(r2.template select<32, 1>(256));
              tensorforge::intel_esimd::simd<float, 32> v103_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v103_data + v102_data);
              // r3 = ir3 + r1
              #pragma unroll
              for (int32_t v105_n1 = 0; v105_n1 < 9; ++v105_n1) {
                int32_t v106_a = v105_n1 * 32;
                tensorforge::intel_esimd::simd<float, 32> v108_data(ir3.template select<32, 1>(v106_a));
                tensorforge::intel_esimd::simd<float, 32> v109_data(r1.template select<32, 1>(v106_a));
                r3.template select<32, 1>(v106_a) = (v109_data + v108_data);
              }
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v111_ld;
              v111_ld.copy_from(glb_m4 + (0 + 0 + 2 * 0 + 0));
              tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 2 * 0 + 0), v111_ld);
              tensorforge::intel_esimd::simd<float, 17> v112_ld;
              v112_ld.copy_from(glb_m4 + (0 + 0 + 1 * 0 + 64));
              tensorforge::slmStore<float, 17>(s1 + (0 + 0 + 1 * 0 + 64), v112_ld);
              // wait(r4 = load{g>r}(glb_m2););
              tensorforge::intel_esimd::simd<float, 288> r5(0.0f);
              // ir5 = +(r4)
              // [(0, 16), (0, 9)] []
              tensorforge::intel_esimd::simd<float, 288> ir5(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v115_data(r4.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 32> v116_data(ir5.template select<32, 1>(0));
              ir5.template select<32, 1>(0) = (v116_data + v115_data);
              tensorforge::intel_esimd::simd<float, 32> v118_data(r4.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v119_data(ir5.template select<32, 1>(32));
              ir5.template select<32, 1>(32) = (v119_data + v118_data);
              tensorforge::intel_esimd::simd<float, 32> v121_data(r4.template select<32, 1>(64));
              tensorforge::intel_esimd::simd<float, 32> v122_data(ir5.template select<32, 1>(64));
              ir5.template select<32, 1>(64) = (v122_data + v121_data);
              tensorforge::intel_esimd::simd<float, 32> v124_data(r4.template select<32, 1>(96));
              tensorforge::intel_esimd::simd<float, 32> v125_data(ir5.template select<32, 1>(96));
              ir5.template select<32, 1>(96) = (v125_data + v124_data);
              tensorforge::intel_esimd::simd<float, 32> v127_data(r4.template select<32, 1>(128));
              tensorforge::intel_esimd::simd<float, 32> v128_data(ir5.template select<32, 1>(128));
              ir5.template select<32, 1>(128) = (v128_data + v127_data);
              tensorforge::intel_esimd::simd<float, 32> v130_data(r4.template select<32, 1>(160));
              tensorforge::intel_esimd::simd<float, 32> v131_data(ir5.template select<32, 1>(160));
              ir5.template select<32, 1>(160) = (v131_data + v130_data);
              tensorforge::intel_esimd::simd<float, 32> v133_data(r4.template select<32, 1>(192));
              tensorforge::intel_esimd::simd<float, 32> v134_data(ir5.template select<32, 1>(192));
              ir5.template select<32, 1>(192) = (v134_data + v133_data);
              tensorforge::intel_esimd::simd<float, 32> v136_data(r4.template select<32, 1>(224));
              tensorforge::intel_esimd::simd<float, 32> v137_data(ir5.template select<32, 1>(224));
              ir5.template select<32, 1>(224) = (v137_data + v136_data);
              tensorforge::intel_esimd::simd<float, 32> v139_data(r4.template select<32, 1>(256));
              tensorforge::intel_esimd::simd<float, 32> v140_data(ir5.template select<32, 1>(256));
              ir5.template select<32, 1>(256) = (v140_data + v139_data);
              // r5 = ir5 + r3
              #pragma unroll
              for (int32_t v142_n1 = 0; v142_n1 < 9; ++v142_n1) {
                int32_t v143_a = v142_n1 * 32;
                tensorforge::intel_esimd::simd<float, 32> v145_data(ir5.template select<32, 1>(v143_a));
                tensorforge::intel_esimd::simd<float, 32> v146_data(r3.template select<32, 1>(v143_a));
                r5.template select<32, 1>(v143_a) = (v146_data + v145_data);
              }
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              tensorforge::intel_esimd::simd<float, 288> r6(0.0f);
              // ir6 = +(r5 * s1)
              // [(0, 32), (0, 9)] [(0, 9)]
              tensorforge::intel_esimd::simd<float, 288> ir6(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v150_data(r5.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 96> s1_w0 = tensorforge::slmLoad<float, 96>(s1 + 0);
              float v151_data = s1_w0[0];
              tensorforge::intel_esimd::simd<float, 32> v153_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v153_data + (v150_data * v151_data));
              float v156_data = s1_w0[9];
              tensorforge::intel_esimd::simd<float, 32> v158_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v158_data + (v150_data * v156_data));
              float v161_data = s1_w0[18];
              tensorforge::intel_esimd::simd<float, 32> v163_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v163_data + (v150_data * v161_data));
              float v166_data = s1_w0[27];
              tensorforge::intel_esimd::simd<float, 32> v168_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v168_data + (v150_data * v166_data));
              float v171_data = s1_w0[36];
              tensorforge::intel_esimd::simd<float, 32> v173_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v173_data + (v150_data * v171_data));
              float v176_data = s1_w0[45];
              tensorforge::intel_esimd::simd<float, 32> v178_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v178_data + (v150_data * v176_data));
              float v181_data = s1_w0[54];
              tensorforge::intel_esimd::simd<float, 32> v183_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v183_data + (v150_data * v181_data));
              float v186_data = s1_w0[63];
              tensorforge::intel_esimd::simd<float, 32> v188_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v188_data + (v150_data * v186_data));
              float v191_data = s1_w0[72];
              tensorforge::intel_esimd::simd<float, 32> v193_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v193_data + (v150_data * v191_data));
              tensorforge::intel_esimd::simd<float, 32> v195_data(r5.template select<32, 1>(32));
              float v196_data = s1_w0[1];
              tensorforge::intel_esimd::simd<float, 32> v198_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v198_data + (v195_data * v196_data));
              float v201_data = s1_w0[10];
              tensorforge::intel_esimd::simd<float, 32> v203_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v203_data + (v195_data * v201_data));
              float v206_data = s1_w0[19];
              tensorforge::intel_esimd::simd<float, 32> v208_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v208_data + (v195_data * v206_data));
              float v211_data = s1_w0[28];
              tensorforge::intel_esimd::simd<float, 32> v213_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v213_data + (v195_data * v211_data));
              float v216_data = s1_w0[37];
              tensorforge::intel_esimd::simd<float, 32> v218_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v218_data + (v195_data * v216_data));
              float v221_data = s1_w0[46];
              tensorforge::intel_esimd::simd<float, 32> v223_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v223_data + (v195_data * v221_data));
              float v226_data = s1_w0[55];
              tensorforge::intel_esimd::simd<float, 32> v228_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v228_data + (v195_data * v226_data));
              float v231_data = s1_w0[64];
              tensorforge::intel_esimd::simd<float, 32> v233_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v233_data + (v195_data * v231_data));
              float v236_data = s1_w0[73];
              tensorforge::intel_esimd::simd<float, 32> v238_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v238_data + (v195_data * v236_data));
              tensorforge::intel_esimd::simd<float, 32> v240_data(r5.template select<32, 1>(64));
              float v241_data = s1_w0[2];
              tensorforge::intel_esimd::simd<float, 32> v243_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v243_data + (v240_data * v241_data));
              float v246_data = s1_w0[11];
              tensorforge::intel_esimd::simd<float, 32> v248_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v248_data + (v240_data * v246_data));
              float v251_data = s1_w0[20];
              tensorforge::intel_esimd::simd<float, 32> v253_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v253_data + (v240_data * v251_data));
              float v256_data = s1_w0[29];
              tensorforge::intel_esimd::simd<float, 32> v258_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v258_data + (v240_data * v256_data));
              float v261_data = s1_w0[38];
              tensorforge::intel_esimd::simd<float, 32> v263_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v263_data + (v240_data * v261_data));
              float v266_data = s1_w0[47];
              tensorforge::intel_esimd::simd<float, 32> v268_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v268_data + (v240_data * v266_data));
              float v271_data = s1_w0[56];
              tensorforge::intel_esimd::simd<float, 32> v273_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v273_data + (v240_data * v271_data));
              float v276_data = s1_w0[65];
              tensorforge::intel_esimd::simd<float, 32> v278_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v278_data + (v240_data * v276_data));
              float v281_data = s1_w0[74];
              tensorforge::intel_esimd::simd<float, 32> v283_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v283_data + (v240_data * v281_data));
              tensorforge::intel_esimd::simd<float, 32> v285_data(r5.template select<32, 1>(96));
              float v286_data = s1_w0[3];
              tensorforge::intel_esimd::simd<float, 32> v288_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v288_data + (v285_data * v286_data));
              float v291_data = s1_w0[12];
              tensorforge::intel_esimd::simd<float, 32> v293_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v293_data + (v285_data * v291_data));
              float v296_data = s1_w0[21];
              tensorforge::intel_esimd::simd<float, 32> v298_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v298_data + (v285_data * v296_data));
              float v301_data = s1_w0[30];
              tensorforge::intel_esimd::simd<float, 32> v303_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v303_data + (v285_data * v301_data));
              float v306_data = s1_w0[39];
              tensorforge::intel_esimd::simd<float, 32> v308_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v308_data + (v285_data * v306_data));
              float v311_data = s1_w0[48];
              tensorforge::intel_esimd::simd<float, 32> v313_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v313_data + (v285_data * v311_data));
              float v316_data = s1_w0[57];
              tensorforge::intel_esimd::simd<float, 32> v318_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v318_data + (v285_data * v316_data));
              float v321_data = s1_w0[66];
              tensorforge::intel_esimd::simd<float, 32> v323_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v323_data + (v285_data * v321_data));
              float v326_data = s1_w0[75];
              tensorforge::intel_esimd::simd<float, 32> v328_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v328_data + (v285_data * v326_data));
              tensorforge::intel_esimd::simd<float, 32> v330_data(r5.template select<32, 1>(128));
              float v331_data = s1_w0[4];
              tensorforge::intel_esimd::simd<float, 32> v333_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v333_data + (v330_data * v331_data));
              float v336_data = s1_w0[13];
              tensorforge::intel_esimd::simd<float, 32> v338_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v338_data + (v330_data * v336_data));
              float v341_data = s1_w0[22];
              tensorforge::intel_esimd::simd<float, 32> v343_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v343_data + (v330_data * v341_data));
              float v346_data = s1_w0[31];
              tensorforge::intel_esimd::simd<float, 32> v348_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v348_data + (v330_data * v346_data));
              float v351_data = s1_w0[40];
              tensorforge::intel_esimd::simd<float, 32> v353_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v353_data + (v330_data * v351_data));
              float v356_data = s1_w0[49];
              tensorforge::intel_esimd::simd<float, 32> v358_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v358_data + (v330_data * v356_data));
              float v361_data = s1_w0[58];
              tensorforge::intel_esimd::simd<float, 32> v363_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v363_data + (v330_data * v361_data));
              float v366_data = s1_w0[67];
              tensorforge::intel_esimd::simd<float, 32> v368_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v368_data + (v330_data * v366_data));
              float v371_data = s1_w0[76];
              tensorforge::intel_esimd::simd<float, 32> v373_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v373_data + (v330_data * v371_data));
              tensorforge::intel_esimd::simd<float, 32> v375_data(r5.template select<32, 1>(160));
              float v376_data = s1_w0[5];
              tensorforge::intel_esimd::simd<float, 32> v378_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v378_data + (v375_data * v376_data));
              float v381_data = s1_w0[14];
              tensorforge::intel_esimd::simd<float, 32> v383_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v383_data + (v375_data * v381_data));
              float v386_data = s1_w0[23];
              tensorforge::intel_esimd::simd<float, 32> v388_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v388_data + (v375_data * v386_data));
              float v391_data = s1_w0[32];
              tensorforge::intel_esimd::simd<float, 32> v393_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v393_data + (v375_data * v391_data));
              float v396_data = s1_w0[41];
              tensorforge::intel_esimd::simd<float, 32> v398_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v398_data + (v375_data * v396_data));
              float v401_data = s1_w0[50];
              tensorforge::intel_esimd::simd<float, 32> v403_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v403_data + (v375_data * v401_data));
              float v406_data = s1_w0[59];
              tensorforge::intel_esimd::simd<float, 32> v408_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v408_data + (v375_data * v406_data));
              float v411_data = s1_w0[68];
              tensorforge::intel_esimd::simd<float, 32> v413_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v413_data + (v375_data * v411_data));
              float v416_data = s1_w0[77];
              tensorforge::intel_esimd::simd<float, 32> v418_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v418_data + (v375_data * v416_data));
              tensorforge::intel_esimd::simd<float, 32> v420_data(r5.template select<32, 1>(192));
              float v421_data = s1_w0[6];
              tensorforge::intel_esimd::simd<float, 32> v423_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v423_data + (v420_data * v421_data));
              float v426_data = s1_w0[15];
              tensorforge::intel_esimd::simd<float, 32> v428_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v428_data + (v420_data * v426_data));
              float v431_data = s1_w0[24];
              tensorforge::intel_esimd::simd<float, 32> v433_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v433_data + (v420_data * v431_data));
              float v436_data = s1_w0[33];
              tensorforge::intel_esimd::simd<float, 32> v438_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v438_data + (v420_data * v436_data));
              float v441_data = s1_w0[42];
              tensorforge::intel_esimd::simd<float, 32> v443_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v443_data + (v420_data * v441_data));
              float v446_data = s1_w0[51];
              tensorforge::intel_esimd::simd<float, 32> v448_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v448_data + (v420_data * v446_data));
              float v451_data = s1_w0[60];
              tensorforge::intel_esimd::simd<float, 32> v453_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v453_data + (v420_data * v451_data));
              float v456_data = s1_w0[69];
              tensorforge::intel_esimd::simd<float, 32> v458_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v458_data + (v420_data * v456_data));
              float v461_data = s1_w0[78];
              tensorforge::intel_esimd::simd<float, 32> v463_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v463_data + (v420_data * v461_data));
              tensorforge::intel_esimd::simd<float, 32> v465_data(r5.template select<32, 1>(224));
              float v466_data = s1_w0[7];
              tensorforge::intel_esimd::simd<float, 32> v468_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v468_data + (v465_data * v466_data));
              float v471_data = s1_w0[16];
              tensorforge::intel_esimd::simd<float, 32> v473_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v473_data + (v465_data * v471_data));
              float v476_data = s1_w0[25];
              tensorforge::intel_esimd::simd<float, 32> v478_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v478_data + (v465_data * v476_data));
              float v481_data = s1_w0[34];
              tensorforge::intel_esimd::simd<float, 32> v483_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v483_data + (v465_data * v481_data));
              float v486_data = s1_w0[43];
              tensorforge::intel_esimd::simd<float, 32> v488_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v488_data + (v465_data * v486_data));
              float v491_data = s1_w0[52];
              tensorforge::intel_esimd::simd<float, 32> v493_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v493_data + (v465_data * v491_data));
              float v496_data = s1_w0[61];
              tensorforge::intel_esimd::simd<float, 32> v498_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v498_data + (v465_data * v496_data));
              float v501_data = s1_w0[70];
              tensorforge::intel_esimd::simd<float, 32> v503_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v503_data + (v465_data * v501_data));
              float v506_data = s1_w0[79];
              tensorforge::intel_esimd::simd<float, 32> v508_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v508_data + (v465_data * v506_data));
              tensorforge::intel_esimd::simd<float, 32> v510_data(r5.template select<32, 1>(256));
              float v511_data = s1_w0[8];
              tensorforge::intel_esimd::simd<float, 32> v513_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v513_data + (v510_data * v511_data));
              float v516_data = s1_w0[17];
              tensorforge::intel_esimd::simd<float, 32> v518_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v518_data + (v510_data * v516_data));
              float v521_data = s1_w0[26];
              tensorforge::intel_esimd::simd<float, 32> v523_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v523_data + (v510_data * v521_data));
              float v526_data = s1_w0[35];
              tensorforge::intel_esimd::simd<float, 32> v528_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v528_data + (v510_data * v526_data));
              float v531_data = s1_w0[44];
              tensorforge::intel_esimd::simd<float, 32> v533_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v533_data + (v510_data * v531_data));
              float v536_data = s1_w0[53];
              tensorforge::intel_esimd::simd<float, 32> v538_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v538_data + (v510_data * v536_data));
              float v541_data = s1_w0[62];
              tensorforge::intel_esimd::simd<float, 32> v543_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v543_data + (v510_data * v541_data));
              float v546_data = s1_w0[71];
              tensorforge::intel_esimd::simd<float, 32> v548_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v548_data + (v510_data * v546_data));
              float v551_data = s1_w0[80];
              tensorforge::intel_esimd::simd<float, 32> v553_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v553_data + (v510_data * v551_data));
              // r6 = ir6
              #pragma unroll
              for (int32_t v555_n0 = 0; v555_n0 < 1; ++v555_n0) {
                int32_t v557_a = v555_n0 * 32;
                #pragma unroll
                for (int32_t v556_n1 = 0; v556_n1 < 9; ++v556_n1) {
                  int32_t v559_a = v557_a + (v556_n1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v560_data(ir6.template select<32, 1>(v559_a));
                  r6.template select<32, 1>(v559_a) = v560_data;
                }
              }
              // glb_m3 = store{r>g}(r6);
              #pragma unroll
              for (int32_t v561_i0 = 0; v561_i0 < 1; ++v561_i0) {
                int32_t v563_a = v561_i0 * 32;
                #pragma unroll
                for (int32_t v562_i1 = 0; v562_i1 < 9; ++v562_i1) {
                  int32_t v565_a = v563_a + (v562_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v566_data(r6.template select<32, 1>(v565_a));
                  v566_data.copy_to(glb_m3 + (v565_a));
                }
              }
            }
          }
        }
      }
    });
  });
}

