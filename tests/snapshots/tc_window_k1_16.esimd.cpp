// === base name ===
kernel_4487b5cbd85bdf2e

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_4487b5cbd85bdf2e = {{1, 16, 1}, 16, 16, 1, 16, 10240, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_4487b5cbd85bdf2e(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_4487b5cbd85bdf2e(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_4487b5cbd85bdf2e(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_4487b5cbd85bdf2e(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_4487b5cbd85bdf2e(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_4487b5cbd85bdf2e(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_4487b5cbd85bdf2e(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2560 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 10240 B shared, occupancy grid
        // operands:
        //   m0 16×9(16×9) {0..16}×{0..9} strided
        //   m1 16×20(16×16) {0..16}×{1..17} none ordered
        //   m2 20×9(16×9) {1..17}×{0..9} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2560}],"shared_bytes":10240,"shared_elements":2560,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,9]],"name":"m0","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,1],[16,17]],"name":"m1","ordered":true,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[1,0],[17,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[16,17]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[17,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (160 * item.get_local_id(1) + 0);
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const float *const __restrict__ glb_m1 = &m1[0];
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v447_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v447_flagSecondAhead < numElements0) ? v447_flagSecondAhead : v3_batchId1)];
          const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 144 + 0 + m2_extraOffset];
          // s0 = load{g>s}(peel_glb_m2[0, 1])
          tensorforge::intel_esimd::simd<float, 64> v455_ld;
          v455_ld.copy_from(peel_glb_m2 + (0 + 0 + 4 * 0 + 0));
          tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v455_ld);
          tensorforge::intel_esimd::simd<float, 64> v456_ld;
          v456_ld.copy_from(peel_glb_m2 + (0 + 0 + 4 * 0 + 64));
          tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v456_ld);
          tensorforge::intel_esimd::simd<float, 16> v457_ld;
          v457_ld.copy_from(peel_glb_m2 + (0 + 0 + 1 * 0 + 128));
          tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v457_ld);
          uint32_t v451_flagWord(flagWordFirst);
          uint32_t v452_flagWordNext(flagWordSecond);
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const float *const __restrict__ wrap_glb_m2 = &m2[v12_batchId1 * 144 + 0 + m2_extraOffset];
            size_t v463_flagAheadAhead = v12_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v463_flagAheadAhead < numElements0) ? v463_flagAheadAhead : v12_batchId1)];
            const bool allowed = static_cast<bool>(v451_flagWord);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 144 + 0 + m0_extraOffset];
              tensorforge::intel_esimd::simd<float, 144> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 16), (0, 9)] [(1, 17)]
              tensorforge::intel_esimd::simd<float, 144> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run0;
              glb_m1_run0.copy_from(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v27_data(glb_m1_run0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v31_data(glb_m1_run0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v35_data(glb_m1_run0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v39_data(glb_m1_run0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run1;
              glb_m1_run1.copy_from(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v43_data(glb_m1_run1.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v47_data(glb_m1_run1.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v51_data(glb_m1_run1.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v55_data(glb_m1_run1.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run2;
              glb_m1_run2.copy_from(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v59_data(glb_m1_run2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v63_data(glb_m1_run2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v67_data(glb_m1_run2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v71_data(glb_m1_run2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run3;
              glb_m1_run3.copy_from(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v75_data(glb_m1_run3.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v79_data(glb_m1_run3.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v83_data(glb_m1_run3.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v87_data(glb_m1_run3.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v88_acc{};
              tensorforge::intel_esimd::simd<float, 16> v91_data(0.0f);
              v91_data.template select<15, 1>(1) = tensorforge::slmLoad<float, 15>((s0 + (-1_i32)) + 1);
              v88_acc += ((static_cast<float>(v91_data[1])) * v27_data);
              v88_acc += ((static_cast<float>(v91_data[2])) * v31_data);
              v88_acc += ((static_cast<float>(v91_data[3])) * v35_data);
              v88_acc += ((static_cast<float>(v91_data[4])) * v39_data);
              v88_acc += ((static_cast<float>(v91_data[5])) * v43_data);
              v88_acc += ((static_cast<float>(v91_data[6])) * v47_data);
              v88_acc += ((static_cast<float>(v91_data[7])) * v51_data);
              v88_acc += ((static_cast<float>(v91_data[8])) * v55_data);
              v88_acc += ((static_cast<float>(v91_data[9])) * v59_data);
              v88_acc += ((static_cast<float>(v91_data[10])) * v63_data);
              v88_acc += ((static_cast<float>(v91_data[11])) * v67_data);
              v88_acc += ((static_cast<float>(v91_data[12])) * v71_data);
              v88_acc += ((static_cast<float>(v91_data[13])) * v75_data);
              v88_acc += ((static_cast<float>(v91_data[14])) * v79_data);
              v88_acc += ((static_cast<float>(v91_data[15])) * v83_data);
              tensorforge::intel_esimd::simd<float, 16> v125_data = tensorforge::slmLoad<float, 16>(s0 + (15_i32));
              v88_acc += ((static_cast<float>(v125_data[0])) * v87_data);
              ir0.template select<16, 1>(0) = v88_acc;
              tensorforge::intel_esimd::simd<float, 16> v128_acc{};
              v128_acc += ((static_cast<float>(v125_data[1])) * v27_data);
              v128_acc += ((static_cast<float>(v125_data[2])) * v31_data);
              v128_acc += ((static_cast<float>(v125_data[3])) * v35_data);
              v128_acc += ((static_cast<float>(v125_data[4])) * v39_data);
              v128_acc += ((static_cast<float>(v125_data[5])) * v43_data);
              v128_acc += ((static_cast<float>(v125_data[6])) * v47_data);
              v128_acc += ((static_cast<float>(v125_data[7])) * v51_data);
              v128_acc += ((static_cast<float>(v125_data[8])) * v55_data);
              v128_acc += ((static_cast<float>(v125_data[9])) * v59_data);
              v128_acc += ((static_cast<float>(v125_data[10])) * v63_data);
              v128_acc += ((static_cast<float>(v125_data[11])) * v67_data);
              v128_acc += ((static_cast<float>(v125_data[12])) * v71_data);
              v128_acc += ((static_cast<float>(v125_data[13])) * v75_data);
              v128_acc += ((static_cast<float>(v125_data[14])) * v79_data);
              v128_acc += ((static_cast<float>(v125_data[15])) * v83_data);
              tensorforge::intel_esimd::simd<float, 16> v163_data = tensorforge::slmLoad<float, 16>(s0 + (31_i32));
              v128_acc += ((static_cast<float>(v163_data[0])) * v87_data);
              ir0.template select<16, 1>(16) = v128_acc;
              tensorforge::intel_esimd::simd<float, 16> v166_acc{};
              v166_acc += ((static_cast<float>(v163_data[1])) * v27_data);
              v166_acc += ((static_cast<float>(v163_data[2])) * v31_data);
              v166_acc += ((static_cast<float>(v163_data[3])) * v35_data);
              v166_acc += ((static_cast<float>(v163_data[4])) * v39_data);
              v166_acc += ((static_cast<float>(v163_data[5])) * v43_data);
              v166_acc += ((static_cast<float>(v163_data[6])) * v47_data);
              v166_acc += ((static_cast<float>(v163_data[7])) * v51_data);
              v166_acc += ((static_cast<float>(v163_data[8])) * v55_data);
              v166_acc += ((static_cast<float>(v163_data[9])) * v59_data);
              v166_acc += ((static_cast<float>(v163_data[10])) * v63_data);
              v166_acc += ((static_cast<float>(v163_data[11])) * v67_data);
              v166_acc += ((static_cast<float>(v163_data[12])) * v71_data);
              v166_acc += ((static_cast<float>(v163_data[13])) * v75_data);
              v166_acc += ((static_cast<float>(v163_data[14])) * v79_data);
              v166_acc += ((static_cast<float>(v163_data[15])) * v83_data);
              tensorforge::intel_esimd::simd<float, 16> v201_data = tensorforge::slmLoad<float, 16>(s0 + (47_i32));
              v166_acc += ((static_cast<float>(v201_data[0])) * v87_data);
              ir0.template select<16, 1>(32) = v166_acc;
              tensorforge::intel_esimd::simd<float, 16> v204_acc{};
              v204_acc += ((static_cast<float>(v201_data[1])) * v27_data);
              v204_acc += ((static_cast<float>(v201_data[2])) * v31_data);
              v204_acc += ((static_cast<float>(v201_data[3])) * v35_data);
              v204_acc += ((static_cast<float>(v201_data[4])) * v39_data);
              v204_acc += ((static_cast<float>(v201_data[5])) * v43_data);
              v204_acc += ((static_cast<float>(v201_data[6])) * v47_data);
              v204_acc += ((static_cast<float>(v201_data[7])) * v51_data);
              v204_acc += ((static_cast<float>(v201_data[8])) * v55_data);
              v204_acc += ((static_cast<float>(v201_data[9])) * v59_data);
              v204_acc += ((static_cast<float>(v201_data[10])) * v63_data);
              v204_acc += ((static_cast<float>(v201_data[11])) * v67_data);
              v204_acc += ((static_cast<float>(v201_data[12])) * v71_data);
              v204_acc += ((static_cast<float>(v201_data[13])) * v75_data);
              v204_acc += ((static_cast<float>(v201_data[14])) * v79_data);
              v204_acc += ((static_cast<float>(v201_data[15])) * v83_data);
              tensorforge::intel_esimd::simd<float, 16> v239_data = tensorforge::slmLoad<float, 16>(s0 + (63_i32));
              v204_acc += ((static_cast<float>(v239_data[0])) * v87_data);
              ir0.template select<16, 1>(48) = v204_acc;
              tensorforge::intel_esimd::simd<float, 16> v242_acc{};
              v242_acc += ((static_cast<float>(v239_data[1])) * v27_data);
              v242_acc += ((static_cast<float>(v239_data[2])) * v31_data);
              v242_acc += ((static_cast<float>(v239_data[3])) * v35_data);
              v242_acc += ((static_cast<float>(v239_data[4])) * v39_data);
              v242_acc += ((static_cast<float>(v239_data[5])) * v43_data);
              v242_acc += ((static_cast<float>(v239_data[6])) * v47_data);
              v242_acc += ((static_cast<float>(v239_data[7])) * v51_data);
              v242_acc += ((static_cast<float>(v239_data[8])) * v55_data);
              v242_acc += ((static_cast<float>(v239_data[9])) * v59_data);
              v242_acc += ((static_cast<float>(v239_data[10])) * v63_data);
              v242_acc += ((static_cast<float>(v239_data[11])) * v67_data);
              v242_acc += ((static_cast<float>(v239_data[12])) * v71_data);
              v242_acc += ((static_cast<float>(v239_data[13])) * v75_data);
              v242_acc += ((static_cast<float>(v239_data[14])) * v79_data);
              v242_acc += ((static_cast<float>(v239_data[15])) * v83_data);
              tensorforge::intel_esimd::simd<float, 16> v277_data = tensorforge::slmLoad<float, 16>(s0 + (79_i32));
              v242_acc += ((static_cast<float>(v277_data[0])) * v87_data);
              ir0.template select<16, 1>(64) = v242_acc;
              tensorforge::intel_esimd::simd<float, 16> v280_acc{};
              v280_acc += ((static_cast<float>(v277_data[1])) * v27_data);
              v280_acc += ((static_cast<float>(v277_data[2])) * v31_data);
              v280_acc += ((static_cast<float>(v277_data[3])) * v35_data);
              v280_acc += ((static_cast<float>(v277_data[4])) * v39_data);
              v280_acc += ((static_cast<float>(v277_data[5])) * v43_data);
              v280_acc += ((static_cast<float>(v277_data[6])) * v47_data);
              v280_acc += ((static_cast<float>(v277_data[7])) * v51_data);
              v280_acc += ((static_cast<float>(v277_data[8])) * v55_data);
              v280_acc += ((static_cast<float>(v277_data[9])) * v59_data);
              v280_acc += ((static_cast<float>(v277_data[10])) * v63_data);
              v280_acc += ((static_cast<float>(v277_data[11])) * v67_data);
              v280_acc += ((static_cast<float>(v277_data[12])) * v71_data);
              v280_acc += ((static_cast<float>(v277_data[13])) * v75_data);
              v280_acc += ((static_cast<float>(v277_data[14])) * v79_data);
              v280_acc += ((static_cast<float>(v277_data[15])) * v83_data);
              tensorforge::intel_esimd::simd<float, 16> v315_data = tensorforge::slmLoad<float, 16>(s0 + (95_i32));
              v280_acc += ((static_cast<float>(v315_data[0])) * v87_data);
              ir0.template select<16, 1>(80) = v280_acc;
              tensorforge::intel_esimd::simd<float, 16> v318_acc{};
              v318_acc += ((static_cast<float>(v315_data[1])) * v27_data);
              v318_acc += ((static_cast<float>(v315_data[2])) * v31_data);
              v318_acc += ((static_cast<float>(v315_data[3])) * v35_data);
              v318_acc += ((static_cast<float>(v315_data[4])) * v39_data);
              v318_acc += ((static_cast<float>(v315_data[5])) * v43_data);
              v318_acc += ((static_cast<float>(v315_data[6])) * v47_data);
              v318_acc += ((static_cast<float>(v315_data[7])) * v51_data);
              v318_acc += ((static_cast<float>(v315_data[8])) * v55_data);
              v318_acc += ((static_cast<float>(v315_data[9])) * v59_data);
              v318_acc += ((static_cast<float>(v315_data[10])) * v63_data);
              v318_acc += ((static_cast<float>(v315_data[11])) * v67_data);
              v318_acc += ((static_cast<float>(v315_data[12])) * v71_data);
              v318_acc += ((static_cast<float>(v315_data[13])) * v75_data);
              v318_acc += ((static_cast<float>(v315_data[14])) * v79_data);
              v318_acc += ((static_cast<float>(v315_data[15])) * v83_data);
              tensorforge::intel_esimd::simd<float, 16> v353_data = tensorforge::slmLoad<float, 16>(s0 + (111_i32));
              v318_acc += ((static_cast<float>(v353_data[0])) * v87_data);
              ir0.template select<16, 1>(96) = v318_acc;
              tensorforge::intel_esimd::simd<float, 16> v356_acc{};
              v356_acc += ((static_cast<float>(v353_data[1])) * v27_data);
              v356_acc += ((static_cast<float>(v353_data[2])) * v31_data);
              v356_acc += ((static_cast<float>(v353_data[3])) * v35_data);
              v356_acc += ((static_cast<float>(v353_data[4])) * v39_data);
              v356_acc += ((static_cast<float>(v353_data[5])) * v43_data);
              v356_acc += ((static_cast<float>(v353_data[6])) * v47_data);
              v356_acc += ((static_cast<float>(v353_data[7])) * v51_data);
              v356_acc += ((static_cast<float>(v353_data[8])) * v55_data);
              v356_acc += ((static_cast<float>(v353_data[9])) * v59_data);
              v356_acc += ((static_cast<float>(v353_data[10])) * v63_data);
              v356_acc += ((static_cast<float>(v353_data[11])) * v67_data);
              v356_acc += ((static_cast<float>(v353_data[12])) * v71_data);
              v356_acc += ((static_cast<float>(v353_data[13])) * v75_data);
              v356_acc += ((static_cast<float>(v353_data[14])) * v79_data);
              v356_acc += ((static_cast<float>(v353_data[15])) * v83_data);
              tensorforge::intel_esimd::simd<float, 16> v391_data = tensorforge::slmLoad<float, 16>(s0 + (127_i32));
              v356_acc += ((static_cast<float>(v391_data[0])) * v87_data);
              ir0.template select<16, 1>(112) = v356_acc;
              tensorforge::intel_esimd::simd<float, 16> v394_acc{};
              v394_acc += ((static_cast<float>(v391_data[1])) * v27_data);
              v394_acc += ((static_cast<float>(v391_data[2])) * v31_data);
              v394_acc += ((static_cast<float>(v391_data[3])) * v35_data);
              v394_acc += ((static_cast<float>(v391_data[4])) * v39_data);
              v394_acc += ((static_cast<float>(v391_data[5])) * v43_data);
              v394_acc += ((static_cast<float>(v391_data[6])) * v47_data);
              v394_acc += ((static_cast<float>(v391_data[7])) * v51_data);
              v394_acc += ((static_cast<float>(v391_data[8])) * v55_data);
              v394_acc += ((static_cast<float>(v391_data[9])) * v59_data);
              v394_acc += ((static_cast<float>(v391_data[10])) * v63_data);
              v394_acc += ((static_cast<float>(v391_data[11])) * v67_data);
              v394_acc += ((static_cast<float>(v391_data[12])) * v71_data);
              v394_acc += ((static_cast<float>(v391_data[13])) * v75_data);
              v394_acc += ((static_cast<float>(v391_data[14])) * v79_data);
              v394_acc += ((static_cast<float>(v391_data[15])) * v83_data);
              tensorforge::intel_esimd::simd<float, 16> v429_data = tensorforge::slmLoad<float, 16>(s0 + (143_i32));
              v394_acc += ((static_cast<float>(v429_data[0])) * v87_data);
              ir0.template select<16, 1>(128) = v394_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v432_n0 = 0; v432_n0 < 1; ++v432_n0) {
                int32_t v434_a = v432_n0 * 16;
                #pragma unroll
                for (int32_t v433_n1 = 0; v433_n1 < 9; ++v433_n1) {
                  int32_t v436_a = v434_a + (v433_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v437_data(ir0.template select<16, 1>(v436_a));
                  r0.template select<16, 1>(v436_a) = v437_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v438_i0 = 0; v438_i0 < 1; ++v438_i0) {
                int32_t v440_a = v438_i0 * 16;
                #pragma unroll
                for (int32_t v439_i1 = 0; v439_i1 < 9; ++v439_i1) {
                  int32_t v442_a = v440_a + (v439_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v443_data(r0.template select<16, 1>(v442_a));
                  v443_data.copy_to(glb_m0 + (v442_a));
                }
              }
            }
            // s0 = load{g>s}(wrap_glb_m2[0, 1])
            tensorforge::intel_esimd::simd<float, 64> v458_ld;
            v458_ld.copy_from(wrap_glb_m2 + (0 + 0 + 4 * 0 + 0));
            tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v458_ld);
            tensorforge::intel_esimd::simd<float, 64> v459_ld;
            v459_ld.copy_from(wrap_glb_m2 + (0 + 0 + 4 * 0 + 64));
            tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v459_ld);
            tensorforge::intel_esimd::simd<float, 16> v460_ld;
            v460_ld.copy_from(wrap_glb_m2 + (0 + 0 + 1 * 0 + 128));
            tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v460_ld);
            v451_flagWord = v452_flagWordNext;
            v452_flagWordNext = flagWordAhead;
          }
        }
      }
    });
  });
}

