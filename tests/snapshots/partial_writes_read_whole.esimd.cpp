// === base name ===
kernel_69028a1ce1e6e561

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_69028a1ce1e6e561 = {{1, 8, 1}, 32, 32, 1, 8, 12288, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_69028a1ce1e6e561(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_69028a1ce1e6e561(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_69028a1ce1e6e561(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 8, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 8;
  config.block[2] = 1;
  config.sharedMemBytes = 3072 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_69028a1ce1e6e561(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_69028a1ce1e6e561(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_69028a1ce1e6e561(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_69028a1ce1e6e561(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
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
            const float *const __restrict__ pf_glb_m0 = &m0[v9_batchId1][0 + m0_extraOffset];
            const float *const __restrict__ pf_glb_m1 = &m1[v9_batchId1][0 + m1_extraOffset];
            const float *const __restrict__ pf_glb_m2 = &m2[v9_batchId1][0 + m2_extraOffset];
            const float *const __restrict__ pf_glb_m4 = &m4[v9_batchId1][0 + m4_extraOffset];
            const bool allowed_next = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId1]);
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
              // s0 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v70_i0 = 0; v70_i0 < 1; ++v70_i0) {
                int32_t v72_a = v70_i0 * 32;
                #pragma unroll
                for (int32_t v71_i1 = 0; v71_i1 < 9; ++v71_i1) {
                  int32_t v74_a = v72_a + (v71_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v75_data(r1.template select<32, 1>(v74_a));
                  tensorforge::slmStore<float, 32>(s0 + (v74_a), v75_data);
                }
              }
              tensorforge::intel_esimd::simd<float, 288> r4(0.0f);
              // r4 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v79_i1 = 0; v79_i1 < 9; ++v79_i1) {
                tensorforge::intel_esimd::simd<float, 16> v84_data;
                v84_data.copy_from(glb_m2 + ((v79_i1 * 16)));
                r4.template select<16, 1>((v79_i1 * 32)) = v84_data;
              }
              // wait(r2 = load{g>r}(glb_m1););
              tensorforge::intel_esimd::simd<float, 288> r3(0.0f);
              // r3 = +(r2) + name: s0, type: SymbolType.SharedMem, lead: [0]
              // [(0, 16), (0, 9)] []
              tensorforge::intel_esimd::simd<float, 288> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v89_data(r2.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 32> v90_data(ir3.template select<32, 1>(0));
              ir3.template select<32, 1>(0) = (v90_data + v89_data);
              tensorforge::intel_esimd::simd<float, 32> v92_data(r2.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v93_data(ir3.template select<32, 1>(32));
              ir3.template select<32, 1>(32) = (v93_data + v92_data);
              tensorforge::intel_esimd::simd<float, 32> v95_data(r2.template select<32, 1>(64));
              tensorforge::intel_esimd::simd<float, 32> v96_data(ir3.template select<32, 1>(64));
              ir3.template select<32, 1>(64) = (v96_data + v95_data);
              tensorforge::intel_esimd::simd<float, 32> v98_data(r2.template select<32, 1>(96));
              tensorforge::intel_esimd::simd<float, 32> v99_data(ir3.template select<32, 1>(96));
              ir3.template select<32, 1>(96) = (v99_data + v98_data);
              tensorforge::intel_esimd::simd<float, 32> v101_data(r2.template select<32, 1>(128));
              tensorforge::intel_esimd::simd<float, 32> v102_data(ir3.template select<32, 1>(128));
              ir3.template select<32, 1>(128) = (v102_data + v101_data);
              tensorforge::intel_esimd::simd<float, 32> v104_data(r2.template select<32, 1>(160));
              tensorforge::intel_esimd::simd<float, 32> v105_data(ir3.template select<32, 1>(160));
              ir3.template select<32, 1>(160) = (v105_data + v104_data);
              tensorforge::intel_esimd::simd<float, 32> v107_data(r2.template select<32, 1>(192));
              tensorforge::intel_esimd::simd<float, 32> v108_data(ir3.template select<32, 1>(192));
              ir3.template select<32, 1>(192) = (v108_data + v107_data);
              tensorforge::intel_esimd::simd<float, 32> v110_data(r2.template select<32, 1>(224));
              tensorforge::intel_esimd::simd<float, 32> v111_data(ir3.template select<32, 1>(224));
              ir3.template select<32, 1>(224) = (v111_data + v110_data);
              tensorforge::intel_esimd::simd<float, 32> v113_data(r2.template select<32, 1>(256));
              tensorforge::intel_esimd::simd<float, 32> v114_data(ir3.template select<32, 1>(256));
              ir3.template select<32, 1>(256) = (v114_data + v113_data);
              #pragma unroll
              for (int32_t v116_n1 = 0; v116_n1 < 9; ++v116_n1) {
                int32_t v117_a = v116_n1 * 32;
                tensorforge::intel_esimd::simd<float, 16> v119_data(ir3.template select<16, 1>(v117_a));
                tensorforge::intel_esimd::simd<float, 16> v123_data = tensorforge::slmLoad<float, 16>(s0 + (v117_a));
                r3.template select<16, 1>(v117_a) = (v123_data + v119_data);
              }
              // s0 = store{r>s}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v125_i1 = 0; v125_i1 < 9; ++v125_i1) {
                int32_t v126_a = v125_i1 * 32;
                tensorforge::intel_esimd::simd<float, 16> v128_data(r3.template select<16, 1>(v126_a));
                tensorforge::slmStore<float, 16>(s0 + (v126_a), v128_data);
              }
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v132_ld;
              v132_ld.copy_from(glb_m4 + (0 + 0 + 2 * 0 + 0));
              tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 2 * 0 + 0), v132_ld);
              tensorforge::intel_esimd::simd<float, 17> v133_ld;
              v133_ld.copy_from(glb_m4 + (0 + 0 + 1 * 0 + 64));
              tensorforge::slmStore<float, 17>(s1 + (0 + 0 + 1 * 0 + 64), v133_ld);
              // wait(r4 = load{g>r}(glb_m2););
              tensorforge::intel_esimd::simd<float, 288> r5(0.0f);
              // r5 = +(r4) + name: s0, type: SymbolType.SharedMem, lead: [0]
              // [(0, 16), (0, 9)] []
              tensorforge::intel_esimd::simd<float, 288> ir5(0.0f);
              tensorforge::intel_esimd::simd<float, 32> v136_data(r4.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 32> v137_data(ir5.template select<32, 1>(0));
              ir5.template select<32, 1>(0) = (v137_data + v136_data);
              tensorforge::intel_esimd::simd<float, 32> v139_data(r4.template select<32, 1>(32));
              tensorforge::intel_esimd::simd<float, 32> v140_data(ir5.template select<32, 1>(32));
              ir5.template select<32, 1>(32) = (v140_data + v139_data);
              tensorforge::intel_esimd::simd<float, 32> v142_data(r4.template select<32, 1>(64));
              tensorforge::intel_esimd::simd<float, 32> v143_data(ir5.template select<32, 1>(64));
              ir5.template select<32, 1>(64) = (v143_data + v142_data);
              tensorforge::intel_esimd::simd<float, 32> v145_data(r4.template select<32, 1>(96));
              tensorforge::intel_esimd::simd<float, 32> v146_data(ir5.template select<32, 1>(96));
              ir5.template select<32, 1>(96) = (v146_data + v145_data);
              tensorforge::intel_esimd::simd<float, 32> v148_data(r4.template select<32, 1>(128));
              tensorforge::intel_esimd::simd<float, 32> v149_data(ir5.template select<32, 1>(128));
              ir5.template select<32, 1>(128) = (v149_data + v148_data);
              tensorforge::intel_esimd::simd<float, 32> v151_data(r4.template select<32, 1>(160));
              tensorforge::intel_esimd::simd<float, 32> v152_data(ir5.template select<32, 1>(160));
              ir5.template select<32, 1>(160) = (v152_data + v151_data);
              tensorforge::intel_esimd::simd<float, 32> v154_data(r4.template select<32, 1>(192));
              tensorforge::intel_esimd::simd<float, 32> v155_data(ir5.template select<32, 1>(192));
              ir5.template select<32, 1>(192) = (v155_data + v154_data);
              tensorforge::intel_esimd::simd<float, 32> v157_data(r4.template select<32, 1>(224));
              tensorforge::intel_esimd::simd<float, 32> v158_data(ir5.template select<32, 1>(224));
              ir5.template select<32, 1>(224) = (v158_data + v157_data);
              tensorforge::intel_esimd::simd<float, 32> v160_data(r4.template select<32, 1>(256));
              tensorforge::intel_esimd::simd<float, 32> v161_data(ir5.template select<32, 1>(256));
              ir5.template select<32, 1>(256) = (v161_data + v160_data);
              #pragma unroll
              for (int32_t v163_n1 = 0; v163_n1 < 9; ++v163_n1) {
                int32_t v164_a = v163_n1 * 32;
                tensorforge::intel_esimd::simd<float, 16> v166_data(ir5.template select<16, 1>(v164_a));
                tensorforge::intel_esimd::simd<float, 16> v170_data = tensorforge::slmLoad<float, 16>(s0 + (v164_a));
                r5.template select<16, 1>(v164_a) = (v170_data + v166_data);
              }
              // s0 = store{r>s}(localShrMem0, r5);
              #pragma unroll
              for (int32_t v172_i1 = 0; v172_i1 < 9; ++v172_i1) {
                int32_t v173_a = v172_i1 * 32;
                tensorforge::intel_esimd::simd<float, 16> v175_data(r5.template select<16, 1>(v173_a));
                tensorforge::slmStore<float, 16>(s0 + (v173_a), v175_data);
              }
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              tensorforge::intel_esimd::simd<float, 288> r6(0.0f);
              // r6 = +(s0 * s1) + None
              // [(0, 32), (0, 9)] [(0, 9)]
              tensorforge::intel_esimd::simd<float, 288> ir6(0.0f);
              tensorforge::intel_esimd::simd<float, 64> s0_run0 = tensorforge::slmLoad<float, 64>(s0 + (0_i32));
              tensorforge::intel_esimd::simd<float, 32> v184_data(s0_run0.template select<32, 1>(0));
              tensorforge::intel_esimd::simd<float, 96> s1_w0 = tensorforge::slmLoad<float, 96>(s1 + 0);
              float v185_data = s1_w0[0];
              tensorforge::intel_esimd::simd<float, 32> v187_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v187_data + (v184_data * v185_data));
              float v190_data = s1_w0[9];
              tensorforge::intel_esimd::simd<float, 32> v192_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v192_data + (v184_data * v190_data));
              float v195_data = s1_w0[18];
              tensorforge::intel_esimd::simd<float, 32> v197_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v197_data + (v184_data * v195_data));
              float v200_data = s1_w0[27];
              tensorforge::intel_esimd::simd<float, 32> v202_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v202_data + (v184_data * v200_data));
              float v205_data = s1_w0[36];
              tensorforge::intel_esimd::simd<float, 32> v207_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v207_data + (v184_data * v205_data));
              float v210_data = s1_w0[45];
              tensorforge::intel_esimd::simd<float, 32> v212_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v212_data + (v184_data * v210_data));
              float v215_data = s1_w0[54];
              tensorforge::intel_esimd::simd<float, 32> v217_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v217_data + (v184_data * v215_data));
              float v220_data = s1_w0[63];
              tensorforge::intel_esimd::simd<float, 32> v222_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v222_data + (v184_data * v220_data));
              float v225_data = s1_w0[72];
              tensorforge::intel_esimd::simd<float, 32> v227_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v227_data + (v184_data * v225_data));
              tensorforge::intel_esimd::simd<float, 32> v230_data(s0_run0.template select<32, 1>(32));
              float v231_data = s1_w0[1];
              tensorforge::intel_esimd::simd<float, 32> v233_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v233_data + (v230_data * v231_data));
              float v236_data = s1_w0[10];
              tensorforge::intel_esimd::simd<float, 32> v238_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v238_data + (v230_data * v236_data));
              float v241_data = s1_w0[19];
              tensorforge::intel_esimd::simd<float, 32> v243_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v243_data + (v230_data * v241_data));
              float v246_data = s1_w0[28];
              tensorforge::intel_esimd::simd<float, 32> v248_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v248_data + (v230_data * v246_data));
              float v251_data = s1_w0[37];
              tensorforge::intel_esimd::simd<float, 32> v253_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v253_data + (v230_data * v251_data));
              float v256_data = s1_w0[46];
              tensorforge::intel_esimd::simd<float, 32> v258_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v258_data + (v230_data * v256_data));
              float v261_data = s1_w0[55];
              tensorforge::intel_esimd::simd<float, 32> v263_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v263_data + (v230_data * v261_data));
              float v266_data = s1_w0[64];
              tensorforge::intel_esimd::simd<float, 32> v268_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v268_data + (v230_data * v266_data));
              float v271_data = s1_w0[73];
              tensorforge::intel_esimd::simd<float, 32> v273_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v273_data + (v230_data * v271_data));
              tensorforge::intel_esimd::simd<float, 64> s0_run1 = tensorforge::slmLoad<float, 64>(s0 + (64_i32));
              tensorforge::intel_esimd::simd<float, 32> v276_data(s0_run1.template select<32, 1>(0));
              float v277_data = s1_w0[2];
              tensorforge::intel_esimd::simd<float, 32> v279_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v279_data + (v276_data * v277_data));
              float v282_data = s1_w0[11];
              tensorforge::intel_esimd::simd<float, 32> v284_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v284_data + (v276_data * v282_data));
              float v287_data = s1_w0[20];
              tensorforge::intel_esimd::simd<float, 32> v289_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v289_data + (v276_data * v287_data));
              float v292_data = s1_w0[29];
              tensorforge::intel_esimd::simd<float, 32> v294_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v294_data + (v276_data * v292_data));
              float v297_data = s1_w0[38];
              tensorforge::intel_esimd::simd<float, 32> v299_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v299_data + (v276_data * v297_data));
              float v302_data = s1_w0[47];
              tensorforge::intel_esimd::simd<float, 32> v304_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v304_data + (v276_data * v302_data));
              float v307_data = s1_w0[56];
              tensorforge::intel_esimd::simd<float, 32> v309_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v309_data + (v276_data * v307_data));
              float v312_data = s1_w0[65];
              tensorforge::intel_esimd::simd<float, 32> v314_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v314_data + (v276_data * v312_data));
              float v317_data = s1_w0[74];
              tensorforge::intel_esimd::simd<float, 32> v319_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v319_data + (v276_data * v317_data));
              tensorforge::intel_esimd::simd<float, 32> v322_data(s0_run1.template select<32, 1>(32));
              float v323_data = s1_w0[3];
              tensorforge::intel_esimd::simd<float, 32> v325_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v325_data + (v322_data * v323_data));
              float v328_data = s1_w0[12];
              tensorforge::intel_esimd::simd<float, 32> v330_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v330_data + (v322_data * v328_data));
              float v333_data = s1_w0[21];
              tensorforge::intel_esimd::simd<float, 32> v335_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v335_data + (v322_data * v333_data));
              float v338_data = s1_w0[30];
              tensorforge::intel_esimd::simd<float, 32> v340_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v340_data + (v322_data * v338_data));
              float v343_data = s1_w0[39];
              tensorforge::intel_esimd::simd<float, 32> v345_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v345_data + (v322_data * v343_data));
              float v348_data = s1_w0[48];
              tensorforge::intel_esimd::simd<float, 32> v350_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v350_data + (v322_data * v348_data));
              float v353_data = s1_w0[57];
              tensorforge::intel_esimd::simd<float, 32> v355_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v355_data + (v322_data * v353_data));
              float v358_data = s1_w0[66];
              tensorforge::intel_esimd::simd<float, 32> v360_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v360_data + (v322_data * v358_data));
              float v363_data = s1_w0[75];
              tensorforge::intel_esimd::simd<float, 32> v365_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v365_data + (v322_data * v363_data));
              tensorforge::intel_esimd::simd<float, 64> s0_run2 = tensorforge::slmLoad<float, 64>(s0 + (128_i32));
              tensorforge::intel_esimd::simd<float, 32> v368_data(s0_run2.template select<32, 1>(0));
              float v369_data = s1_w0[4];
              tensorforge::intel_esimd::simd<float, 32> v371_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v371_data + (v368_data * v369_data));
              float v374_data = s1_w0[13];
              tensorforge::intel_esimd::simd<float, 32> v376_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v376_data + (v368_data * v374_data));
              float v379_data = s1_w0[22];
              tensorforge::intel_esimd::simd<float, 32> v381_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v381_data + (v368_data * v379_data));
              float v384_data = s1_w0[31];
              tensorforge::intel_esimd::simd<float, 32> v386_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v386_data + (v368_data * v384_data));
              float v389_data = s1_w0[40];
              tensorforge::intel_esimd::simd<float, 32> v391_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v391_data + (v368_data * v389_data));
              float v394_data = s1_w0[49];
              tensorforge::intel_esimd::simd<float, 32> v396_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v396_data + (v368_data * v394_data));
              float v399_data = s1_w0[58];
              tensorforge::intel_esimd::simd<float, 32> v401_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v401_data + (v368_data * v399_data));
              float v404_data = s1_w0[67];
              tensorforge::intel_esimd::simd<float, 32> v406_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v406_data + (v368_data * v404_data));
              float v409_data = s1_w0[76];
              tensorforge::intel_esimd::simd<float, 32> v411_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v411_data + (v368_data * v409_data));
              tensorforge::intel_esimd::simd<float, 32> v414_data(s0_run2.template select<32, 1>(32));
              float v415_data = s1_w0[5];
              tensorforge::intel_esimd::simd<float, 32> v417_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v417_data + (v414_data * v415_data));
              float v420_data = s1_w0[14];
              tensorforge::intel_esimd::simd<float, 32> v422_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v422_data + (v414_data * v420_data));
              float v425_data = s1_w0[23];
              tensorforge::intel_esimd::simd<float, 32> v427_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v427_data + (v414_data * v425_data));
              float v430_data = s1_w0[32];
              tensorforge::intel_esimd::simd<float, 32> v432_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v432_data + (v414_data * v430_data));
              float v435_data = s1_w0[41];
              tensorforge::intel_esimd::simd<float, 32> v437_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v437_data + (v414_data * v435_data));
              float v440_data = s1_w0[50];
              tensorforge::intel_esimd::simd<float, 32> v442_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v442_data + (v414_data * v440_data));
              float v445_data = s1_w0[59];
              tensorforge::intel_esimd::simd<float, 32> v447_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v447_data + (v414_data * v445_data));
              float v450_data = s1_w0[68];
              tensorforge::intel_esimd::simd<float, 32> v452_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v452_data + (v414_data * v450_data));
              float v455_data = s1_w0[77];
              tensorforge::intel_esimd::simd<float, 32> v457_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v457_data + (v414_data * v455_data));
              tensorforge::intel_esimd::simd<float, 64> s0_run3 = tensorforge::slmLoad<float, 64>(s0 + (192_i32));
              tensorforge::intel_esimd::simd<float, 32> v460_data(s0_run3.template select<32, 1>(0));
              float v461_data = s1_w0[6];
              tensorforge::intel_esimd::simd<float, 32> v463_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v463_data + (v460_data * v461_data));
              float v466_data = s1_w0[15];
              tensorforge::intel_esimd::simd<float, 32> v468_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v468_data + (v460_data * v466_data));
              float v471_data = s1_w0[24];
              tensorforge::intel_esimd::simd<float, 32> v473_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v473_data + (v460_data * v471_data));
              float v476_data = s1_w0[33];
              tensorforge::intel_esimd::simd<float, 32> v478_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v478_data + (v460_data * v476_data));
              float v481_data = s1_w0[42];
              tensorforge::intel_esimd::simd<float, 32> v483_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v483_data + (v460_data * v481_data));
              float v486_data = s1_w0[51];
              tensorforge::intel_esimd::simd<float, 32> v488_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v488_data + (v460_data * v486_data));
              float v491_data = s1_w0[60];
              tensorforge::intel_esimd::simd<float, 32> v493_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v493_data + (v460_data * v491_data));
              float v496_data = s1_w0[69];
              tensorforge::intel_esimd::simd<float, 32> v498_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v498_data + (v460_data * v496_data));
              float v501_data = s1_w0[78];
              tensorforge::intel_esimd::simd<float, 32> v503_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v503_data + (v460_data * v501_data));
              tensorforge::intel_esimd::simd<float, 32> v506_data(s0_run3.template select<32, 1>(32));
              float v507_data = s1_w0[7];
              tensorforge::intel_esimd::simd<float, 32> v509_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v509_data + (v506_data * v507_data));
              float v512_data = s1_w0[16];
              tensorforge::intel_esimd::simd<float, 32> v514_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v514_data + (v506_data * v512_data));
              float v517_data = s1_w0[25];
              tensorforge::intel_esimd::simd<float, 32> v519_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v519_data + (v506_data * v517_data));
              float v522_data = s1_w0[34];
              tensorforge::intel_esimd::simd<float, 32> v524_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v524_data + (v506_data * v522_data));
              float v527_data = s1_w0[43];
              tensorforge::intel_esimd::simd<float, 32> v529_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v529_data + (v506_data * v527_data));
              float v532_data = s1_w0[52];
              tensorforge::intel_esimd::simd<float, 32> v534_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v534_data + (v506_data * v532_data));
              float v537_data = s1_w0[61];
              tensorforge::intel_esimd::simd<float, 32> v539_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v539_data + (v506_data * v537_data));
              float v542_data = s1_w0[70];
              tensorforge::intel_esimd::simd<float, 32> v544_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v544_data + (v506_data * v542_data));
              float v547_data = s1_w0[79];
              tensorforge::intel_esimd::simd<float, 32> v549_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v549_data + (v506_data * v547_data));
              tensorforge::intel_esimd::simd<float, 32> v552_data = tensorforge::slmLoad<float, 32>(s0 + (256_i32));
              float v553_data = s1_w0[8];
              tensorforge::intel_esimd::simd<float, 32> v555_data(ir6.template select<32, 1>(0));
              ir6.template select<32, 1>(0) = (v555_data + (v552_data * v553_data));
              float v558_data = s1_w0[17];
              tensorforge::intel_esimd::simd<float, 32> v560_data(ir6.template select<32, 1>(32));
              ir6.template select<32, 1>(32) = (v560_data + (v552_data * v558_data));
              float v563_data = s1_w0[26];
              tensorforge::intel_esimd::simd<float, 32> v565_data(ir6.template select<32, 1>(64));
              ir6.template select<32, 1>(64) = (v565_data + (v552_data * v563_data));
              float v568_data = s1_w0[35];
              tensorforge::intel_esimd::simd<float, 32> v570_data(ir6.template select<32, 1>(96));
              ir6.template select<32, 1>(96) = (v570_data + (v552_data * v568_data));
              float v573_data = s1_w0[44];
              tensorforge::intel_esimd::simd<float, 32> v575_data(ir6.template select<32, 1>(128));
              ir6.template select<32, 1>(128) = (v575_data + (v552_data * v573_data));
              float v578_data = s1_w0[53];
              tensorforge::intel_esimd::simd<float, 32> v580_data(ir6.template select<32, 1>(160));
              ir6.template select<32, 1>(160) = (v580_data + (v552_data * v578_data));
              float v583_data = s1_w0[62];
              tensorforge::intel_esimd::simd<float, 32> v585_data(ir6.template select<32, 1>(192));
              ir6.template select<32, 1>(192) = (v585_data + (v552_data * v583_data));
              float v588_data = s1_w0[71];
              tensorforge::intel_esimd::simd<float, 32> v590_data(ir6.template select<32, 1>(224));
              ir6.template select<32, 1>(224) = (v590_data + (v552_data * v588_data));
              float v593_data = s1_w0[80];
              tensorforge::intel_esimd::simd<float, 32> v595_data(ir6.template select<32, 1>(256));
              ir6.template select<32, 1>(256) = (v595_data + (v552_data * v593_data));
              #pragma unroll
              for (int32_t v597_n0 = 0; v597_n0 < 1; ++v597_n0) {
                int32_t v599_a = v597_n0 * 32;
                #pragma unroll
                for (int32_t v598_n1 = 0; v598_n1 < 9; ++v598_n1) {
                  int32_t v601_a = v599_a + (v598_n1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v602_data(ir6.template select<32, 1>(v601_a));
                  r6.template select<32, 1>(v601_a) = v602_data;
                }
              }
              // glb_m3 = store{r>g}(r6);
              #pragma unroll
              for (int32_t v603_i0 = 0; v603_i0 < 1; ++v603_i0) {
                int32_t v605_a = v603_i0 * 32;
                #pragma unroll
                for (int32_t v604_i1 = 0; v604_i1 < 9; ++v604_i1) {
                  int32_t v607_a = v605_a + (v604_i1 * 32);
                  tensorforge::intel_esimd::simd<float, 32> v608_data(r6.template select<32, 1>(v607_a));
                  v608_data.copy_to(glb_m3 + (v607_a));
                }
              }
            }
            if (allowed_next) {
              tensorforge::prefetchRunsL2<1152, 576>(&pf_glb_m0[0], &pf_glb_m1[0]);
            }
            if (allowed_next) {
              tensorforge::prefetchRunsL2<576, 324>(&pf_glb_m2[0], &pf_glb_m4[0]);
            }
          }
        }
      }
    });
  });
}

