// === base name ===
kernel_15d850e132c3845b

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_15d850e132c3845b = {{1, 16, 1}, 16, 12, 1, 16, 10240, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_15d850e132c3845b(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_15d850e132c3845b(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_15d850e132c3845b(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 2560 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_15d850e132c3845b(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_15d850e132c3845b(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_15d850e132c3845b(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_15d850e132c3845b(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2560 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 10240 B shared, occupancy grid
        // operands:
        //   m0 32×32(12×12) {0..12}×{0..12} strided
        //   m1 32×32(12×12) {0..12}×{0..12} strided
        //   m2 32×32(12×12) {0..12}×{0..12} strided
        //   m3 32×32(12×12) {0..12}×{0..12} strided
        // operations:
        //   t0[i,j]@{0..12}×{0..6} = m0[i,k] × m1[k,j]
        //   m2[i,j] = m3[i,k] × t0[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2560}],"shared_bytes":10240,"shared_elements":2560,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B","bbox":[[0,0],[12,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,6]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]},{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (160 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (144);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v12_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v12_batchId0 < numElements0; v12_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v13_ahead1 = v12_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v15_batchId1 = (v13_ahead1 < numElements0) ? v13_ahead1 : v12_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v12_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v12_batchId0 * 144 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v12_batchId0 * 144 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v12_batchId0 * 144 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v12_batchId0 * 144 + 0 + m3_extraOffset];
              tensorforge::intel_esimd::simd<float, 192> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v25_i1 = 0; v25_i1 < 12; ++v25_i1) {
                tensorforge::intel_esimd::simd<float, 12> v30_data;
                v30_data.copy_from(glb_m0 + ((v25_i1 * 12)));
                r0.template select<12, 1>((v25_i1 * 16)) = v30_data;
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v33_ld;
              v33_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v33_ld);
              tensorforge::intel_esimd::simd<float, 64> v34_ld;
              v34_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v34_ld);
              tensorforge::intel_esimd::simd<float, 16> v35_ld;
              v35_ld.copy_from(glb_m1 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v35_ld);
              // wait(r0 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<float, 192> r2(0.0f);
              // r2 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v37_i1 = 0; v37_i1 < 12; ++v37_i1) {
                tensorforge::intel_esimd::simd<float, 12> v42_data;
                v42_data.copy_from(glb_m3 + ((v37_i1 * 12)));
                r2.template select<12, 1>((v37_i1 * 16)) = v42_data;
              }
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 96> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 12), (0, 6)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v48_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v49_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v50_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v51_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v52_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v53_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v54_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v55_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v56_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v57_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v58_acc{};
              tensorforge::intel_esimd::simd<float, 16> v62_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v58_acc += ((static_cast<float>(v62_data[0])) * v46_data);
              v58_acc += ((static_cast<float>(v62_data[1])) * v47_data);
              v58_acc += ((static_cast<float>(v62_data[2])) * v48_data);
              v58_acc += ((static_cast<float>(v62_data[3])) * v49_data);
              v58_acc += ((static_cast<float>(v62_data[4])) * v50_data);
              v58_acc += ((static_cast<float>(v62_data[5])) * v51_data);
              v58_acc += ((static_cast<float>(v62_data[6])) * v52_data);
              v58_acc += ((static_cast<float>(v62_data[7])) * v53_data);
              v58_acc += ((static_cast<float>(v62_data[8])) * v54_data);
              v58_acc += ((static_cast<float>(v62_data[9])) * v55_data);
              v58_acc += ((static_cast<float>(v62_data[10])) * v56_data);
              v58_acc += ((static_cast<float>(v62_data[11])) * v57_data);
              r1.template select<16, 1>(0) = v58_acc;
              tensorforge::intel_esimd::simd<float, 16> v87_acc{};
              tensorforge::intel_esimd::simd<float, 16> v89_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              v87_acc += ((static_cast<float>(v89_data[0])) * v46_data);
              v87_acc += ((static_cast<float>(v89_data[1])) * v47_data);
              v87_acc += ((static_cast<float>(v89_data[2])) * v48_data);
              v87_acc += ((static_cast<float>(v89_data[3])) * v49_data);
              v87_acc += ((static_cast<float>(v89_data[4])) * v50_data);
              v87_acc += ((static_cast<float>(v89_data[5])) * v51_data);
              v87_acc += ((static_cast<float>(v89_data[6])) * v52_data);
              v87_acc += ((static_cast<float>(v89_data[7])) * v53_data);
              v87_acc += ((static_cast<float>(v89_data[8])) * v54_data);
              v87_acc += ((static_cast<float>(v89_data[9])) * v55_data);
              v87_acc += ((static_cast<float>(v89_data[10])) * v56_data);
              v87_acc += ((static_cast<float>(v89_data[11])) * v57_data);
              r1.template select<16, 1>(16) = v87_acc;
              tensorforge::intel_esimd::simd<float, 16> v114_acc{};
              tensorforge::intel_esimd::simd<float, 16> v116_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              v114_acc += ((static_cast<float>(v116_data[0])) * v46_data);
              v114_acc += ((static_cast<float>(v116_data[1])) * v47_data);
              v114_acc += ((static_cast<float>(v116_data[2])) * v48_data);
              v114_acc += ((static_cast<float>(v116_data[3])) * v49_data);
              v114_acc += ((static_cast<float>(v116_data[4])) * v50_data);
              v114_acc += ((static_cast<float>(v116_data[5])) * v51_data);
              v114_acc += ((static_cast<float>(v116_data[6])) * v52_data);
              v114_acc += ((static_cast<float>(v116_data[7])) * v53_data);
              v114_acc += ((static_cast<float>(v116_data[8])) * v54_data);
              v114_acc += ((static_cast<float>(v116_data[9])) * v55_data);
              v114_acc += ((static_cast<float>(v116_data[10])) * v56_data);
              v114_acc += ((static_cast<float>(v116_data[11])) * v57_data);
              r1.template select<16, 1>(32) = v114_acc;
              tensorforge::intel_esimd::simd<float, 16> v141_acc{};
              tensorforge::intel_esimd::simd<float, 16> v143_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              v141_acc += ((static_cast<float>(v143_data[0])) * v46_data);
              v141_acc += ((static_cast<float>(v143_data[1])) * v47_data);
              v141_acc += ((static_cast<float>(v143_data[2])) * v48_data);
              v141_acc += ((static_cast<float>(v143_data[3])) * v49_data);
              v141_acc += ((static_cast<float>(v143_data[4])) * v50_data);
              v141_acc += ((static_cast<float>(v143_data[5])) * v51_data);
              v141_acc += ((static_cast<float>(v143_data[6])) * v52_data);
              v141_acc += ((static_cast<float>(v143_data[7])) * v53_data);
              v141_acc += ((static_cast<float>(v143_data[8])) * v54_data);
              v141_acc += ((static_cast<float>(v143_data[9])) * v55_data);
              v141_acc += ((static_cast<float>(v143_data[10])) * v56_data);
              v141_acc += ((static_cast<float>(v143_data[11])) * v57_data);
              r1.template select<16, 1>(48) = v141_acc;
              tensorforge::intel_esimd::simd<float, 16> v168_acc{};
              tensorforge::intel_esimd::simd<float, 16> v170_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v168_acc += ((static_cast<float>(v170_data[0])) * v46_data);
              v168_acc += ((static_cast<float>(v170_data[1])) * v47_data);
              v168_acc += ((static_cast<float>(v170_data[2])) * v48_data);
              v168_acc += ((static_cast<float>(v170_data[3])) * v49_data);
              v168_acc += ((static_cast<float>(v170_data[4])) * v50_data);
              v168_acc += ((static_cast<float>(v170_data[5])) * v51_data);
              v168_acc += ((static_cast<float>(v170_data[6])) * v52_data);
              v168_acc += ((static_cast<float>(v170_data[7])) * v53_data);
              v168_acc += ((static_cast<float>(v170_data[8])) * v54_data);
              v168_acc += ((static_cast<float>(v170_data[9])) * v55_data);
              v168_acc += ((static_cast<float>(v170_data[10])) * v56_data);
              v168_acc += ((static_cast<float>(v170_data[11])) * v57_data);
              r1.template select<16, 1>(64) = v168_acc;
              tensorforge::intel_esimd::simd<float, 16> v195_acc{};
              tensorforge::intel_esimd::simd<float, 16> v197_data = tensorforge::slmLoad<float, 16>(s0 + (60_i32));
              v195_acc += ((static_cast<float>(v197_data[0])) * v46_data);
              v195_acc += ((static_cast<float>(v197_data[1])) * v47_data);
              v195_acc += ((static_cast<float>(v197_data[2])) * v48_data);
              v195_acc += ((static_cast<float>(v197_data[3])) * v49_data);
              v195_acc += ((static_cast<float>(v197_data[4])) * v50_data);
              v195_acc += ((static_cast<float>(v197_data[5])) * v51_data);
              v195_acc += ((static_cast<float>(v197_data[6])) * v52_data);
              v195_acc += ((static_cast<float>(v197_data[7])) * v53_data);
              v195_acc += ((static_cast<float>(v197_data[8])) * v54_data);
              v195_acc += ((static_cast<float>(v197_data[9])) * v55_data);
              v195_acc += ((static_cast<float>(v197_data[10])) * v56_data);
              v195_acc += ((static_cast<float>(v197_data[11])) * v57_data);
              r1.template select<16, 1>(80) = v195_acc;
              // s1 = store{r>s, clear}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v222_z1 = 6; v222_z1 < 12; ++v222_z1) {
                s1[(v222_z1 * 12)] = 0.0f;
              }
              #pragma unroll
              for (int32_t v228_i1 = 0; v228_i1 < 6; ++v228_i1) {
                tensorforge::intel_esimd::simd<float, 12> v231_data(r1.template select<12, 1>((v228_i1 * 16)));
                tensorforge::slmStore<float, 12>(s1 + ((v228_i1 * 12)), v231_data);
              }
              // wait(r2 = load{g>r}(glb_m3););
              tensorforge::intel_esimd::simd<float, 192> r3(0.0f);
              // ir3 = +(r2 * s1)
              // [(0, 12), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 192> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v238_data(r2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v239_data(r2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v240_data(r2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v241_data(r2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v242_data(r2.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v243_data(r2.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v244_data(r2.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v245_data(r2.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v246_data(r2.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v247_data(r2.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v248_data(r2.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v249_data(r2.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v250_acc{};
              tensorforge::intel_esimd::simd<float, 16> v254_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v250_acc += ((static_cast<float>(v254_data[0])) * v238_data);
              v250_acc += ((static_cast<float>(v254_data[1])) * v239_data);
              v250_acc += ((static_cast<float>(v254_data[2])) * v240_data);
              v250_acc += ((static_cast<float>(v254_data[3])) * v241_data);
              v250_acc += ((static_cast<float>(v254_data[4])) * v242_data);
              v250_acc += ((static_cast<float>(v254_data[5])) * v243_data);
              v250_acc += ((static_cast<float>(v254_data[6])) * v244_data);
              v250_acc += ((static_cast<float>(v254_data[7])) * v245_data);
              v250_acc += ((static_cast<float>(v254_data[8])) * v246_data);
              v250_acc += ((static_cast<float>(v254_data[9])) * v247_data);
              v250_acc += ((static_cast<float>(v254_data[10])) * v248_data);
              v250_acc += ((static_cast<float>(v254_data[11])) * v249_data);
              ir3.template select<16, 1>(0) = v250_acc;
              tensorforge::intel_esimd::simd<float, 16> v279_acc{};
              tensorforge::intel_esimd::simd<float, 16> v281_data = tensorforge::slmLoad<float, 16>(s1 + (12_i32));
              v279_acc += ((static_cast<float>(v281_data[0])) * v238_data);
              v279_acc += ((static_cast<float>(v281_data[1])) * v239_data);
              v279_acc += ((static_cast<float>(v281_data[2])) * v240_data);
              v279_acc += ((static_cast<float>(v281_data[3])) * v241_data);
              v279_acc += ((static_cast<float>(v281_data[4])) * v242_data);
              v279_acc += ((static_cast<float>(v281_data[5])) * v243_data);
              v279_acc += ((static_cast<float>(v281_data[6])) * v244_data);
              v279_acc += ((static_cast<float>(v281_data[7])) * v245_data);
              v279_acc += ((static_cast<float>(v281_data[8])) * v246_data);
              v279_acc += ((static_cast<float>(v281_data[9])) * v247_data);
              v279_acc += ((static_cast<float>(v281_data[10])) * v248_data);
              v279_acc += ((static_cast<float>(v281_data[11])) * v249_data);
              ir3.template select<16, 1>(16) = v279_acc;
              tensorforge::intel_esimd::simd<float, 16> v306_acc{};
              tensorforge::intel_esimd::simd<float, 16> v308_data = tensorforge::slmLoad<float, 16>(s1 + (24_i32));
              v306_acc += ((static_cast<float>(v308_data[0])) * v238_data);
              v306_acc += ((static_cast<float>(v308_data[1])) * v239_data);
              v306_acc += ((static_cast<float>(v308_data[2])) * v240_data);
              v306_acc += ((static_cast<float>(v308_data[3])) * v241_data);
              v306_acc += ((static_cast<float>(v308_data[4])) * v242_data);
              v306_acc += ((static_cast<float>(v308_data[5])) * v243_data);
              v306_acc += ((static_cast<float>(v308_data[6])) * v244_data);
              v306_acc += ((static_cast<float>(v308_data[7])) * v245_data);
              v306_acc += ((static_cast<float>(v308_data[8])) * v246_data);
              v306_acc += ((static_cast<float>(v308_data[9])) * v247_data);
              v306_acc += ((static_cast<float>(v308_data[10])) * v248_data);
              v306_acc += ((static_cast<float>(v308_data[11])) * v249_data);
              ir3.template select<16, 1>(32) = v306_acc;
              tensorforge::intel_esimd::simd<float, 16> v333_acc{};
              tensorforge::intel_esimd::simd<float, 16> v335_data = tensorforge::slmLoad<float, 16>(s1 + (36_i32));
              v333_acc += ((static_cast<float>(v335_data[0])) * v238_data);
              v333_acc += ((static_cast<float>(v335_data[1])) * v239_data);
              v333_acc += ((static_cast<float>(v335_data[2])) * v240_data);
              v333_acc += ((static_cast<float>(v335_data[3])) * v241_data);
              v333_acc += ((static_cast<float>(v335_data[4])) * v242_data);
              v333_acc += ((static_cast<float>(v335_data[5])) * v243_data);
              v333_acc += ((static_cast<float>(v335_data[6])) * v244_data);
              v333_acc += ((static_cast<float>(v335_data[7])) * v245_data);
              v333_acc += ((static_cast<float>(v335_data[8])) * v246_data);
              v333_acc += ((static_cast<float>(v335_data[9])) * v247_data);
              v333_acc += ((static_cast<float>(v335_data[10])) * v248_data);
              v333_acc += ((static_cast<float>(v335_data[11])) * v249_data);
              ir3.template select<16, 1>(48) = v333_acc;
              tensorforge::intel_esimd::simd<float, 16> v360_acc{};
              tensorforge::intel_esimd::simd<float, 16> v362_data = tensorforge::slmLoad<float, 16>(s1 + (48_i32));
              v360_acc += ((static_cast<float>(v362_data[0])) * v238_data);
              v360_acc += ((static_cast<float>(v362_data[1])) * v239_data);
              v360_acc += ((static_cast<float>(v362_data[2])) * v240_data);
              v360_acc += ((static_cast<float>(v362_data[3])) * v241_data);
              v360_acc += ((static_cast<float>(v362_data[4])) * v242_data);
              v360_acc += ((static_cast<float>(v362_data[5])) * v243_data);
              v360_acc += ((static_cast<float>(v362_data[6])) * v244_data);
              v360_acc += ((static_cast<float>(v362_data[7])) * v245_data);
              v360_acc += ((static_cast<float>(v362_data[8])) * v246_data);
              v360_acc += ((static_cast<float>(v362_data[9])) * v247_data);
              v360_acc += ((static_cast<float>(v362_data[10])) * v248_data);
              v360_acc += ((static_cast<float>(v362_data[11])) * v249_data);
              ir3.template select<16, 1>(64) = v360_acc;
              tensorforge::intel_esimd::simd<float, 16> v387_acc{};
              tensorforge::intel_esimd::simd<float, 16> v389_data = tensorforge::slmLoad<float, 16>(s1 + (60_i32));
              v387_acc += ((static_cast<float>(v389_data[0])) * v238_data);
              v387_acc += ((static_cast<float>(v389_data[1])) * v239_data);
              v387_acc += ((static_cast<float>(v389_data[2])) * v240_data);
              v387_acc += ((static_cast<float>(v389_data[3])) * v241_data);
              v387_acc += ((static_cast<float>(v389_data[4])) * v242_data);
              v387_acc += ((static_cast<float>(v389_data[5])) * v243_data);
              v387_acc += ((static_cast<float>(v389_data[6])) * v244_data);
              v387_acc += ((static_cast<float>(v389_data[7])) * v245_data);
              v387_acc += ((static_cast<float>(v389_data[8])) * v246_data);
              v387_acc += ((static_cast<float>(v389_data[9])) * v247_data);
              v387_acc += ((static_cast<float>(v389_data[10])) * v248_data);
              v387_acc += ((static_cast<float>(v389_data[11])) * v249_data);
              ir3.template select<16, 1>(80) = v387_acc;
              tensorforge::intel_esimd::simd<float, 16> v414_acc{};
              tensorforge::intel_esimd::simd<float, 16> v416_data = tensorforge::slmLoad<float, 16>(s1 + (72_i32));
              v414_acc += ((static_cast<float>(v416_data[0])) * v238_data);
              v414_acc += ((static_cast<float>(v416_data[1])) * v239_data);
              v414_acc += ((static_cast<float>(v416_data[2])) * v240_data);
              v414_acc += ((static_cast<float>(v416_data[3])) * v241_data);
              v414_acc += ((static_cast<float>(v416_data[4])) * v242_data);
              v414_acc += ((static_cast<float>(v416_data[5])) * v243_data);
              v414_acc += ((static_cast<float>(v416_data[6])) * v244_data);
              v414_acc += ((static_cast<float>(v416_data[7])) * v245_data);
              v414_acc += ((static_cast<float>(v416_data[8])) * v246_data);
              v414_acc += ((static_cast<float>(v416_data[9])) * v247_data);
              v414_acc += ((static_cast<float>(v416_data[10])) * v248_data);
              v414_acc += ((static_cast<float>(v416_data[11])) * v249_data);
              ir3.template select<16, 1>(96) = v414_acc;
              tensorforge::intel_esimd::simd<float, 16> v441_acc{};
              tensorforge::intel_esimd::simd<float, 16> v443_data = tensorforge::slmLoad<float, 16>(s1 + (84_i32));
              v441_acc += ((static_cast<float>(v443_data[0])) * v238_data);
              v441_acc += ((static_cast<float>(v443_data[1])) * v239_data);
              v441_acc += ((static_cast<float>(v443_data[2])) * v240_data);
              v441_acc += ((static_cast<float>(v443_data[3])) * v241_data);
              v441_acc += ((static_cast<float>(v443_data[4])) * v242_data);
              v441_acc += ((static_cast<float>(v443_data[5])) * v243_data);
              v441_acc += ((static_cast<float>(v443_data[6])) * v244_data);
              v441_acc += ((static_cast<float>(v443_data[7])) * v245_data);
              v441_acc += ((static_cast<float>(v443_data[8])) * v246_data);
              v441_acc += ((static_cast<float>(v443_data[9])) * v247_data);
              v441_acc += ((static_cast<float>(v443_data[10])) * v248_data);
              v441_acc += ((static_cast<float>(v443_data[11])) * v249_data);
              ir3.template select<16, 1>(112) = v441_acc;
              tensorforge::intel_esimd::simd<float, 16> v468_acc{};
              tensorforge::intel_esimd::simd<float, 16> v470_data = tensorforge::slmLoad<float, 16>(s1 + (96_i32));
              v468_acc += ((static_cast<float>(v470_data[0])) * v238_data);
              v468_acc += ((static_cast<float>(v470_data[1])) * v239_data);
              v468_acc += ((static_cast<float>(v470_data[2])) * v240_data);
              v468_acc += ((static_cast<float>(v470_data[3])) * v241_data);
              v468_acc += ((static_cast<float>(v470_data[4])) * v242_data);
              v468_acc += ((static_cast<float>(v470_data[5])) * v243_data);
              v468_acc += ((static_cast<float>(v470_data[6])) * v244_data);
              v468_acc += ((static_cast<float>(v470_data[7])) * v245_data);
              v468_acc += ((static_cast<float>(v470_data[8])) * v246_data);
              v468_acc += ((static_cast<float>(v470_data[9])) * v247_data);
              v468_acc += ((static_cast<float>(v470_data[10])) * v248_data);
              v468_acc += ((static_cast<float>(v470_data[11])) * v249_data);
              ir3.template select<16, 1>(128) = v468_acc;
              tensorforge::intel_esimd::simd<float, 16> v495_acc{};
              tensorforge::intel_esimd::simd<float, 16> v497_data = tensorforge::slmLoad<float, 16>(s1 + (108_i32));
              v495_acc += ((static_cast<float>(v497_data[0])) * v238_data);
              v495_acc += ((static_cast<float>(v497_data[1])) * v239_data);
              v495_acc += ((static_cast<float>(v497_data[2])) * v240_data);
              v495_acc += ((static_cast<float>(v497_data[3])) * v241_data);
              v495_acc += ((static_cast<float>(v497_data[4])) * v242_data);
              v495_acc += ((static_cast<float>(v497_data[5])) * v243_data);
              v495_acc += ((static_cast<float>(v497_data[6])) * v244_data);
              v495_acc += ((static_cast<float>(v497_data[7])) * v245_data);
              v495_acc += ((static_cast<float>(v497_data[8])) * v246_data);
              v495_acc += ((static_cast<float>(v497_data[9])) * v247_data);
              v495_acc += ((static_cast<float>(v497_data[10])) * v248_data);
              v495_acc += ((static_cast<float>(v497_data[11])) * v249_data);
              ir3.template select<16, 1>(144) = v495_acc;
              tensorforge::intel_esimd::simd<float, 16> v522_acc{};
              tensorforge::intel_esimd::simd<float, 16> v524_data = tensorforge::slmLoad<float, 16>(s1 + (120_i32));
              v522_acc += ((static_cast<float>(v524_data[0])) * v238_data);
              v522_acc += ((static_cast<float>(v524_data[1])) * v239_data);
              v522_acc += ((static_cast<float>(v524_data[2])) * v240_data);
              v522_acc += ((static_cast<float>(v524_data[3])) * v241_data);
              v522_acc += ((static_cast<float>(v524_data[4])) * v242_data);
              v522_acc += ((static_cast<float>(v524_data[5])) * v243_data);
              v522_acc += ((static_cast<float>(v524_data[6])) * v244_data);
              v522_acc += ((static_cast<float>(v524_data[7])) * v245_data);
              v522_acc += ((static_cast<float>(v524_data[8])) * v246_data);
              v522_acc += ((static_cast<float>(v524_data[9])) * v247_data);
              v522_acc += ((static_cast<float>(v524_data[10])) * v248_data);
              v522_acc += ((static_cast<float>(v524_data[11])) * v249_data);
              ir3.template select<16, 1>(160) = v522_acc;
              tensorforge::intel_esimd::simd<float, 16> v549_acc{};
              tensorforge::intel_esimd::simd<float, 16> v551_data = tensorforge::slmLoad<float, 16>(s1 + (132_i32));
              v549_acc += ((static_cast<float>(v551_data[0])) * v238_data);
              v549_acc += ((static_cast<float>(v551_data[1])) * v239_data);
              v549_acc += ((static_cast<float>(v551_data[2])) * v240_data);
              v549_acc += ((static_cast<float>(v551_data[3])) * v241_data);
              v549_acc += ((static_cast<float>(v551_data[4])) * v242_data);
              v549_acc += ((static_cast<float>(v551_data[5])) * v243_data);
              v549_acc += ((static_cast<float>(v551_data[6])) * v244_data);
              v549_acc += ((static_cast<float>(v551_data[7])) * v245_data);
              v549_acc += ((static_cast<float>(v551_data[8])) * v246_data);
              v549_acc += ((static_cast<float>(v551_data[9])) * v247_data);
              v549_acc += ((static_cast<float>(v551_data[10])) * v248_data);
              v549_acc += ((static_cast<float>(v551_data[11])) * v249_data);
              ir3.template select<16, 1>(176) = v549_acc;
              // r3 = ir3
              #pragma unroll
              for (int32_t v576_n1 = 0; v576_n1 < 12; ++v576_n1) {
                int32_t v577_a = v576_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v579_data(ir3.template select<12, 1>(v577_a));
                r3.template select<12, 1>(v577_a) = v579_data;
              }
              // glb_m2 = store{r>g}(r3);
              #pragma unroll
              for (int32_t v580_i1 = 0; v580_i1 < 12; ++v580_i1) {
                tensorforge::intel_esimd::simd<float, 12> v583_data(r3.template select<12, 1>((v580_i1 * 16)));
                v583_data.copy_to(glb_m2 + ((v580_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}

