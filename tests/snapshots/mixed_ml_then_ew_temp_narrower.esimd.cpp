// === base name ===
kernel_57f5b49127855fa0

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_57f5b49127855fa0 = {{1, 16, 1}, 16, 12, 1, 16, 19456, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_57f5b49127855fa0(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_57f5b49127855fa0(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_57f5b49127855fa0(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 4864 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_57f5b49127855fa0(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_57f5b49127855fa0(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_57f5b49127855fa0(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_57f5b49127855fa0(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<4864 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 19456 B shared, occupancy grid
        // operands:
        //   m0 32×32(12×12) {0..12}×{0..12} strided
        //   m1 32×32(12×12) {0..12}×{0..12} strided
        //   m2 32×32(12×12) {0..12}×{0..12} strided
        //   m3 32×32(4×12) {4..8}×{0..12} strided
        //   m4 32×32(12×12) {0..12}×{0..12} strided
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   t1[i,j] = t0[i,k] × m2[k,j]
        //   t0 12×12(12×12) {0..12}×{0..12} pointer_based({4..8}×{0..12}) = abs(N)
        //   m4[i,j] = t1[i,k] × t0[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":4864}],"shared_bytes":19456,"shared_elements":4864,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B","bbox":[[0,0],[12,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"E","bbox":[[0,0],[12,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"N","bbox":[[4,0],[8,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m4","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[4,0],[8,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[4,0],[8,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t1","offset":[0,0],"shape":[12,12]},{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (304 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (288);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s2 = localShrMem0 + (144);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v13_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v13_batchId0 < numElements0; v13_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v14_ahead1 = v13_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v16_batchId1 = (v14_ahead1 < numElements0) ? v14_ahead1 : v13_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v13_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v13_batchId0 * 144 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v13_batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v13_batchId0 * 144 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v13_batchId0 * 48 + 0 + m3_extraOffset];
              float *const __restrict__ glb_m4 = &m4[v13_batchId0 * 144 + 0 + m4_extraOffset];
              tensorforge::intel_esimd::simd<float, 192> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v27_i1 = 0; v27_i1 < 12; ++v27_i1) {
                tensorforge::intel_esimd::simd<float, 12> v32_data;
                v32_data.copy_from(glb_m0 + ((v27_i1 * 12)));
                r0.template select<12, 1>((v27_i1 * 16)) = v32_data;
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v35_ld;
              v35_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v35_ld);
              tensorforge::intel_esimd::simd<float, 64> v36_ld;
              v36_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v36_ld);
              tensorforge::intel_esimd::simd<float, 16> v37_ld;
              v37_ld.copy_from(glb_m1 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v37_ld);
              // wait(r0 = load{g>r}(glb_m0););
              // s2 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v38_ld;
              v38_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s2 + (0 + 0 + 4 * 0 + 0), v38_ld);
              tensorforge::intel_esimd::simd<float, 64> v39_ld;
              v39_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s2 + (0 + 0 + 4 * 0 + 64), v39_ld);
              tensorforge::intel_esimd::simd<float, 16> v40_ld;
              v40_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s2 + (0 + 0 + 1 * 0 + 128), v40_ld);
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 192> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 12), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v48_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v49_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v50_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v51_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v52_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v53_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v54_acc{};
              tensorforge::intel_esimd::simd<float, 16> v58_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v54_acc += ((static_cast<float>(v58_data[0])) * v42_data);
              v54_acc += ((static_cast<float>(v58_data[1])) * v43_data);
              v54_acc += ((static_cast<float>(v58_data[2])) * v44_data);
              v54_acc += ((static_cast<float>(v58_data[3])) * v45_data);
              v54_acc += ((static_cast<float>(v58_data[4])) * v46_data);
              v54_acc += ((static_cast<float>(v58_data[5])) * v47_data);
              v54_acc += ((static_cast<float>(v58_data[6])) * v48_data);
              v54_acc += ((static_cast<float>(v58_data[7])) * v49_data);
              v54_acc += ((static_cast<float>(v58_data[8])) * v50_data);
              v54_acc += ((static_cast<float>(v58_data[9])) * v51_data);
              v54_acc += ((static_cast<float>(v58_data[10])) * v52_data);
              v54_acc += ((static_cast<float>(v58_data[11])) * v53_data);
              r1.template select<16, 1>(0) = v54_acc;
              tensorforge::intel_esimd::simd<float, 16> v83_acc{};
              tensorforge::intel_esimd::simd<float, 16> v85_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              v83_acc += ((static_cast<float>(v85_data[0])) * v42_data);
              v83_acc += ((static_cast<float>(v85_data[1])) * v43_data);
              v83_acc += ((static_cast<float>(v85_data[2])) * v44_data);
              v83_acc += ((static_cast<float>(v85_data[3])) * v45_data);
              v83_acc += ((static_cast<float>(v85_data[4])) * v46_data);
              v83_acc += ((static_cast<float>(v85_data[5])) * v47_data);
              v83_acc += ((static_cast<float>(v85_data[6])) * v48_data);
              v83_acc += ((static_cast<float>(v85_data[7])) * v49_data);
              v83_acc += ((static_cast<float>(v85_data[8])) * v50_data);
              v83_acc += ((static_cast<float>(v85_data[9])) * v51_data);
              v83_acc += ((static_cast<float>(v85_data[10])) * v52_data);
              v83_acc += ((static_cast<float>(v85_data[11])) * v53_data);
              r1.template select<16, 1>(16) = v83_acc;
              tensorforge::intel_esimd::simd<float, 16> v110_acc{};
              tensorforge::intel_esimd::simd<float, 16> v112_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              v110_acc += ((static_cast<float>(v112_data[0])) * v42_data);
              v110_acc += ((static_cast<float>(v112_data[1])) * v43_data);
              v110_acc += ((static_cast<float>(v112_data[2])) * v44_data);
              v110_acc += ((static_cast<float>(v112_data[3])) * v45_data);
              v110_acc += ((static_cast<float>(v112_data[4])) * v46_data);
              v110_acc += ((static_cast<float>(v112_data[5])) * v47_data);
              v110_acc += ((static_cast<float>(v112_data[6])) * v48_data);
              v110_acc += ((static_cast<float>(v112_data[7])) * v49_data);
              v110_acc += ((static_cast<float>(v112_data[8])) * v50_data);
              v110_acc += ((static_cast<float>(v112_data[9])) * v51_data);
              v110_acc += ((static_cast<float>(v112_data[10])) * v52_data);
              v110_acc += ((static_cast<float>(v112_data[11])) * v53_data);
              r1.template select<16, 1>(32) = v110_acc;
              tensorforge::intel_esimd::simd<float, 16> v137_acc{};
              tensorforge::intel_esimd::simd<float, 16> v139_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              v137_acc += ((static_cast<float>(v139_data[0])) * v42_data);
              v137_acc += ((static_cast<float>(v139_data[1])) * v43_data);
              v137_acc += ((static_cast<float>(v139_data[2])) * v44_data);
              v137_acc += ((static_cast<float>(v139_data[3])) * v45_data);
              v137_acc += ((static_cast<float>(v139_data[4])) * v46_data);
              v137_acc += ((static_cast<float>(v139_data[5])) * v47_data);
              v137_acc += ((static_cast<float>(v139_data[6])) * v48_data);
              v137_acc += ((static_cast<float>(v139_data[7])) * v49_data);
              v137_acc += ((static_cast<float>(v139_data[8])) * v50_data);
              v137_acc += ((static_cast<float>(v139_data[9])) * v51_data);
              v137_acc += ((static_cast<float>(v139_data[10])) * v52_data);
              v137_acc += ((static_cast<float>(v139_data[11])) * v53_data);
              r1.template select<16, 1>(48) = v137_acc;
              tensorforge::intel_esimd::simd<float, 16> v164_acc{};
              tensorforge::intel_esimd::simd<float, 16> v166_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v164_acc += ((static_cast<float>(v166_data[0])) * v42_data);
              v164_acc += ((static_cast<float>(v166_data[1])) * v43_data);
              v164_acc += ((static_cast<float>(v166_data[2])) * v44_data);
              v164_acc += ((static_cast<float>(v166_data[3])) * v45_data);
              v164_acc += ((static_cast<float>(v166_data[4])) * v46_data);
              v164_acc += ((static_cast<float>(v166_data[5])) * v47_data);
              v164_acc += ((static_cast<float>(v166_data[6])) * v48_data);
              v164_acc += ((static_cast<float>(v166_data[7])) * v49_data);
              v164_acc += ((static_cast<float>(v166_data[8])) * v50_data);
              v164_acc += ((static_cast<float>(v166_data[9])) * v51_data);
              v164_acc += ((static_cast<float>(v166_data[10])) * v52_data);
              v164_acc += ((static_cast<float>(v166_data[11])) * v53_data);
              r1.template select<16, 1>(64) = v164_acc;
              tensorforge::intel_esimd::simd<float, 16> v191_acc{};
              tensorforge::intel_esimd::simd<float, 16> v193_data = tensorforge::slmLoad<float, 16>(s0 + (60_i32));
              v191_acc += ((static_cast<float>(v193_data[0])) * v42_data);
              v191_acc += ((static_cast<float>(v193_data[1])) * v43_data);
              v191_acc += ((static_cast<float>(v193_data[2])) * v44_data);
              v191_acc += ((static_cast<float>(v193_data[3])) * v45_data);
              v191_acc += ((static_cast<float>(v193_data[4])) * v46_data);
              v191_acc += ((static_cast<float>(v193_data[5])) * v47_data);
              v191_acc += ((static_cast<float>(v193_data[6])) * v48_data);
              v191_acc += ((static_cast<float>(v193_data[7])) * v49_data);
              v191_acc += ((static_cast<float>(v193_data[8])) * v50_data);
              v191_acc += ((static_cast<float>(v193_data[9])) * v51_data);
              v191_acc += ((static_cast<float>(v193_data[10])) * v52_data);
              v191_acc += ((static_cast<float>(v193_data[11])) * v53_data);
              r1.template select<16, 1>(80) = v191_acc;
              tensorforge::intel_esimd::simd<float, 16> v218_acc{};
              tensorforge::intel_esimd::simd<float, 16> v220_data = tensorforge::slmLoad<float, 16>(s0 + (72_i32));
              v218_acc += ((static_cast<float>(v220_data[0])) * v42_data);
              v218_acc += ((static_cast<float>(v220_data[1])) * v43_data);
              v218_acc += ((static_cast<float>(v220_data[2])) * v44_data);
              v218_acc += ((static_cast<float>(v220_data[3])) * v45_data);
              v218_acc += ((static_cast<float>(v220_data[4])) * v46_data);
              v218_acc += ((static_cast<float>(v220_data[5])) * v47_data);
              v218_acc += ((static_cast<float>(v220_data[6])) * v48_data);
              v218_acc += ((static_cast<float>(v220_data[7])) * v49_data);
              v218_acc += ((static_cast<float>(v220_data[8])) * v50_data);
              v218_acc += ((static_cast<float>(v220_data[9])) * v51_data);
              v218_acc += ((static_cast<float>(v220_data[10])) * v52_data);
              v218_acc += ((static_cast<float>(v220_data[11])) * v53_data);
              r1.template select<16, 1>(96) = v218_acc;
              tensorforge::intel_esimd::simd<float, 16> v245_acc{};
              tensorforge::intel_esimd::simd<float, 16> v247_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v245_acc += ((static_cast<float>(v247_data[0])) * v42_data);
              v245_acc += ((static_cast<float>(v247_data[1])) * v43_data);
              v245_acc += ((static_cast<float>(v247_data[2])) * v44_data);
              v245_acc += ((static_cast<float>(v247_data[3])) * v45_data);
              v245_acc += ((static_cast<float>(v247_data[4])) * v46_data);
              v245_acc += ((static_cast<float>(v247_data[5])) * v47_data);
              v245_acc += ((static_cast<float>(v247_data[6])) * v48_data);
              v245_acc += ((static_cast<float>(v247_data[7])) * v49_data);
              v245_acc += ((static_cast<float>(v247_data[8])) * v50_data);
              v245_acc += ((static_cast<float>(v247_data[9])) * v51_data);
              v245_acc += ((static_cast<float>(v247_data[10])) * v52_data);
              v245_acc += ((static_cast<float>(v247_data[11])) * v53_data);
              r1.template select<16, 1>(112) = v245_acc;
              tensorforge::intel_esimd::simd<float, 16> v272_acc{};
              tensorforge::intel_esimd::simd<float, 16> v274_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              v272_acc += ((static_cast<float>(v274_data[0])) * v42_data);
              v272_acc += ((static_cast<float>(v274_data[1])) * v43_data);
              v272_acc += ((static_cast<float>(v274_data[2])) * v44_data);
              v272_acc += ((static_cast<float>(v274_data[3])) * v45_data);
              v272_acc += ((static_cast<float>(v274_data[4])) * v46_data);
              v272_acc += ((static_cast<float>(v274_data[5])) * v47_data);
              v272_acc += ((static_cast<float>(v274_data[6])) * v48_data);
              v272_acc += ((static_cast<float>(v274_data[7])) * v49_data);
              v272_acc += ((static_cast<float>(v274_data[8])) * v50_data);
              v272_acc += ((static_cast<float>(v274_data[9])) * v51_data);
              v272_acc += ((static_cast<float>(v274_data[10])) * v52_data);
              v272_acc += ((static_cast<float>(v274_data[11])) * v53_data);
              r1.template select<16, 1>(128) = v272_acc;
              tensorforge::intel_esimd::simd<float, 16> v299_acc{};
              tensorforge::intel_esimd::simd<float, 16> v301_data = tensorforge::slmLoad<float, 16>(s0 + (108_i32));
              v299_acc += ((static_cast<float>(v301_data[0])) * v42_data);
              v299_acc += ((static_cast<float>(v301_data[1])) * v43_data);
              v299_acc += ((static_cast<float>(v301_data[2])) * v44_data);
              v299_acc += ((static_cast<float>(v301_data[3])) * v45_data);
              v299_acc += ((static_cast<float>(v301_data[4])) * v46_data);
              v299_acc += ((static_cast<float>(v301_data[5])) * v47_data);
              v299_acc += ((static_cast<float>(v301_data[6])) * v48_data);
              v299_acc += ((static_cast<float>(v301_data[7])) * v49_data);
              v299_acc += ((static_cast<float>(v301_data[8])) * v50_data);
              v299_acc += ((static_cast<float>(v301_data[9])) * v51_data);
              v299_acc += ((static_cast<float>(v301_data[10])) * v52_data);
              v299_acc += ((static_cast<float>(v301_data[11])) * v53_data);
              r1.template select<16, 1>(144) = v299_acc;
              tensorforge::intel_esimd::simd<float, 16> v326_acc{};
              tensorforge::intel_esimd::simd<float, 16> v328_data = tensorforge::slmLoad<float, 16>(s0 + (120_i32));
              v326_acc += ((static_cast<float>(v328_data[0])) * v42_data);
              v326_acc += ((static_cast<float>(v328_data[1])) * v43_data);
              v326_acc += ((static_cast<float>(v328_data[2])) * v44_data);
              v326_acc += ((static_cast<float>(v328_data[3])) * v45_data);
              v326_acc += ((static_cast<float>(v328_data[4])) * v46_data);
              v326_acc += ((static_cast<float>(v328_data[5])) * v47_data);
              v326_acc += ((static_cast<float>(v328_data[6])) * v48_data);
              v326_acc += ((static_cast<float>(v328_data[7])) * v49_data);
              v326_acc += ((static_cast<float>(v328_data[8])) * v50_data);
              v326_acc += ((static_cast<float>(v328_data[9])) * v51_data);
              v326_acc += ((static_cast<float>(v328_data[10])) * v52_data);
              v326_acc += ((static_cast<float>(v328_data[11])) * v53_data);
              r1.template select<16, 1>(160) = v326_acc;
              tensorforge::intel_esimd::simd<float, 16> v353_acc{};
              tensorforge::intel_esimd::simd<float, 16> v355_data = tensorforge::slmLoad<float, 16>(s0 + (132_i32));
              v353_acc += ((static_cast<float>(v355_data[0])) * v42_data);
              v353_acc += ((static_cast<float>(v355_data[1])) * v43_data);
              v353_acc += ((static_cast<float>(v355_data[2])) * v44_data);
              v353_acc += ((static_cast<float>(v355_data[3])) * v45_data);
              v353_acc += ((static_cast<float>(v355_data[4])) * v46_data);
              v353_acc += ((static_cast<float>(v355_data[5])) * v47_data);
              v353_acc += ((static_cast<float>(v355_data[6])) * v48_data);
              v353_acc += ((static_cast<float>(v355_data[7])) * v49_data);
              v353_acc += ((static_cast<float>(v355_data[8])) * v50_data);
              v353_acc += ((static_cast<float>(v355_data[9])) * v51_data);
              v353_acc += ((static_cast<float>(v355_data[10])) * v52_data);
              v353_acc += ((static_cast<float>(v355_data[11])) * v53_data);
              r1.template select<16, 1>(176) = v353_acc;
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v380_i1 = 0; v380_i1 < 12; ++v380_i1) {
                tensorforge::intel_esimd::simd<float, 12> v383_data(r1.template select<12, 1>((v380_i1 * 16)));
                tensorforge::slmStore<float, 12>(s1 + ((v380_i1 * 12)), v383_data);
              }
              // wait(s2 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 192> r2(0.0f);
              // r2 = +(s1 * s2) + None
              // [(0, 12), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 16> v392_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v394_data = tensorforge::slmLoad<float, 16>(s1 + (12_i32));
              tensorforge::intel_esimd::simd<float, 16> v396_data = tensorforge::slmLoad<float, 16>(s1 + (24_i32));
              tensorforge::intel_esimd::simd<float, 16> v398_data = tensorforge::slmLoad<float, 16>(s1 + (36_i32));
              tensorforge::intel_esimd::simd<float, 16> v400_data = tensorforge::slmLoad<float, 16>(s1 + (48_i32));
              tensorforge::intel_esimd::simd<float, 16> v402_data = tensorforge::slmLoad<float, 16>(s1 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v404_data = tensorforge::slmLoad<float, 16>(s1 + (72_i32));
              tensorforge::intel_esimd::simd<float, 16> v406_data = tensorforge::slmLoad<float, 16>(s1 + (84_i32));
              tensorforge::intel_esimd::simd<float, 16> v408_data = tensorforge::slmLoad<float, 16>(s1 + (96_i32));
              tensorforge::intel_esimd::simd<float, 16> v410_data = tensorforge::slmLoad<float, 16>(s1 + (108_i32));
              tensorforge::intel_esimd::simd<float, 16> v412_data = tensorforge::slmLoad<float, 16>(s1 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v414_data = tensorforge::slmLoad<float, 16>(s1 + (132_i32));
              tensorforge::intel_esimd::simd<float, 16> v415_acc{};
              tensorforge::intel_esimd::simd<float, 16> v416_data = tensorforge::slmLoad<float, 16>(s2 + (0_i32));
              v415_acc += ((static_cast<float>(v416_data[0])) * v392_data);
              v415_acc += ((static_cast<float>(v416_data[1])) * v394_data);
              v415_acc += ((static_cast<float>(v416_data[2])) * v396_data);
              v415_acc += ((static_cast<float>(v416_data[3])) * v398_data);
              v415_acc += ((static_cast<float>(v416_data[4])) * v400_data);
              v415_acc += ((static_cast<float>(v416_data[5])) * v402_data);
              v415_acc += ((static_cast<float>(v416_data[6])) * v404_data);
              v415_acc += ((static_cast<float>(v416_data[7])) * v406_data);
              v415_acc += ((static_cast<float>(v416_data[8])) * v408_data);
              v415_acc += ((static_cast<float>(v416_data[9])) * v410_data);
              v415_acc += ((static_cast<float>(v416_data[10])) * v412_data);
              v415_acc += ((static_cast<float>(v416_data[11])) * v414_data);
              r2.template select<16, 1>(0) = v415_acc;
              tensorforge::intel_esimd::simd<float, 16> v441_acc{};
              tensorforge::intel_esimd::simd<float, 16> v442_data = tensorforge::slmLoad<float, 16>(s2 + (12_i32));
              v441_acc += ((static_cast<float>(v442_data[0])) * v392_data);
              v441_acc += ((static_cast<float>(v442_data[1])) * v394_data);
              v441_acc += ((static_cast<float>(v442_data[2])) * v396_data);
              v441_acc += ((static_cast<float>(v442_data[3])) * v398_data);
              v441_acc += ((static_cast<float>(v442_data[4])) * v400_data);
              v441_acc += ((static_cast<float>(v442_data[5])) * v402_data);
              v441_acc += ((static_cast<float>(v442_data[6])) * v404_data);
              v441_acc += ((static_cast<float>(v442_data[7])) * v406_data);
              v441_acc += ((static_cast<float>(v442_data[8])) * v408_data);
              v441_acc += ((static_cast<float>(v442_data[9])) * v410_data);
              v441_acc += ((static_cast<float>(v442_data[10])) * v412_data);
              v441_acc += ((static_cast<float>(v442_data[11])) * v414_data);
              r2.template select<16, 1>(16) = v441_acc;
              tensorforge::intel_esimd::simd<float, 16> v467_acc{};
              tensorforge::intel_esimd::simd<float, 16> v468_data = tensorforge::slmLoad<float, 16>(s2 + (24_i32));
              v467_acc += ((static_cast<float>(v468_data[0])) * v392_data);
              v467_acc += ((static_cast<float>(v468_data[1])) * v394_data);
              v467_acc += ((static_cast<float>(v468_data[2])) * v396_data);
              v467_acc += ((static_cast<float>(v468_data[3])) * v398_data);
              v467_acc += ((static_cast<float>(v468_data[4])) * v400_data);
              v467_acc += ((static_cast<float>(v468_data[5])) * v402_data);
              v467_acc += ((static_cast<float>(v468_data[6])) * v404_data);
              v467_acc += ((static_cast<float>(v468_data[7])) * v406_data);
              v467_acc += ((static_cast<float>(v468_data[8])) * v408_data);
              v467_acc += ((static_cast<float>(v468_data[9])) * v410_data);
              v467_acc += ((static_cast<float>(v468_data[10])) * v412_data);
              v467_acc += ((static_cast<float>(v468_data[11])) * v414_data);
              r2.template select<16, 1>(32) = v467_acc;
              tensorforge::intel_esimd::simd<float, 16> v493_acc{};
              tensorforge::intel_esimd::simd<float, 16> v494_data = tensorforge::slmLoad<float, 16>(s2 + (36_i32));
              v493_acc += ((static_cast<float>(v494_data[0])) * v392_data);
              v493_acc += ((static_cast<float>(v494_data[1])) * v394_data);
              v493_acc += ((static_cast<float>(v494_data[2])) * v396_data);
              v493_acc += ((static_cast<float>(v494_data[3])) * v398_data);
              v493_acc += ((static_cast<float>(v494_data[4])) * v400_data);
              v493_acc += ((static_cast<float>(v494_data[5])) * v402_data);
              v493_acc += ((static_cast<float>(v494_data[6])) * v404_data);
              v493_acc += ((static_cast<float>(v494_data[7])) * v406_data);
              v493_acc += ((static_cast<float>(v494_data[8])) * v408_data);
              v493_acc += ((static_cast<float>(v494_data[9])) * v410_data);
              v493_acc += ((static_cast<float>(v494_data[10])) * v412_data);
              v493_acc += ((static_cast<float>(v494_data[11])) * v414_data);
              r2.template select<16, 1>(48) = v493_acc;
              tensorforge::intel_esimd::simd<float, 16> v519_acc{};
              tensorforge::intel_esimd::simd<float, 16> v520_data = tensorforge::slmLoad<float, 16>(s2 + (48_i32));
              v519_acc += ((static_cast<float>(v520_data[0])) * v392_data);
              v519_acc += ((static_cast<float>(v520_data[1])) * v394_data);
              v519_acc += ((static_cast<float>(v520_data[2])) * v396_data);
              v519_acc += ((static_cast<float>(v520_data[3])) * v398_data);
              v519_acc += ((static_cast<float>(v520_data[4])) * v400_data);
              v519_acc += ((static_cast<float>(v520_data[5])) * v402_data);
              v519_acc += ((static_cast<float>(v520_data[6])) * v404_data);
              v519_acc += ((static_cast<float>(v520_data[7])) * v406_data);
              v519_acc += ((static_cast<float>(v520_data[8])) * v408_data);
              v519_acc += ((static_cast<float>(v520_data[9])) * v410_data);
              v519_acc += ((static_cast<float>(v520_data[10])) * v412_data);
              v519_acc += ((static_cast<float>(v520_data[11])) * v414_data);
              r2.template select<16, 1>(64) = v519_acc;
              tensorforge::intel_esimd::simd<float, 16> v545_acc{};
              tensorforge::intel_esimd::simd<float, 16> v546_data = tensorforge::slmLoad<float, 16>(s2 + (60_i32));
              v545_acc += ((static_cast<float>(v546_data[0])) * v392_data);
              v545_acc += ((static_cast<float>(v546_data[1])) * v394_data);
              v545_acc += ((static_cast<float>(v546_data[2])) * v396_data);
              v545_acc += ((static_cast<float>(v546_data[3])) * v398_data);
              v545_acc += ((static_cast<float>(v546_data[4])) * v400_data);
              v545_acc += ((static_cast<float>(v546_data[5])) * v402_data);
              v545_acc += ((static_cast<float>(v546_data[6])) * v404_data);
              v545_acc += ((static_cast<float>(v546_data[7])) * v406_data);
              v545_acc += ((static_cast<float>(v546_data[8])) * v408_data);
              v545_acc += ((static_cast<float>(v546_data[9])) * v410_data);
              v545_acc += ((static_cast<float>(v546_data[10])) * v412_data);
              v545_acc += ((static_cast<float>(v546_data[11])) * v414_data);
              r2.template select<16, 1>(80) = v545_acc;
              tensorforge::intel_esimd::simd<float, 16> v571_acc{};
              tensorforge::intel_esimd::simd<float, 16> v572_data = tensorforge::slmLoad<float, 16>(s2 + (72_i32));
              v571_acc += ((static_cast<float>(v572_data[0])) * v392_data);
              v571_acc += ((static_cast<float>(v572_data[1])) * v394_data);
              v571_acc += ((static_cast<float>(v572_data[2])) * v396_data);
              v571_acc += ((static_cast<float>(v572_data[3])) * v398_data);
              v571_acc += ((static_cast<float>(v572_data[4])) * v400_data);
              v571_acc += ((static_cast<float>(v572_data[5])) * v402_data);
              v571_acc += ((static_cast<float>(v572_data[6])) * v404_data);
              v571_acc += ((static_cast<float>(v572_data[7])) * v406_data);
              v571_acc += ((static_cast<float>(v572_data[8])) * v408_data);
              v571_acc += ((static_cast<float>(v572_data[9])) * v410_data);
              v571_acc += ((static_cast<float>(v572_data[10])) * v412_data);
              v571_acc += ((static_cast<float>(v572_data[11])) * v414_data);
              r2.template select<16, 1>(96) = v571_acc;
              tensorforge::intel_esimd::simd<float, 16> v597_acc{};
              tensorforge::intel_esimd::simd<float, 16> v598_data = tensorforge::slmLoad<float, 16>(s2 + (84_i32));
              v597_acc += ((static_cast<float>(v598_data[0])) * v392_data);
              v597_acc += ((static_cast<float>(v598_data[1])) * v394_data);
              v597_acc += ((static_cast<float>(v598_data[2])) * v396_data);
              v597_acc += ((static_cast<float>(v598_data[3])) * v398_data);
              v597_acc += ((static_cast<float>(v598_data[4])) * v400_data);
              v597_acc += ((static_cast<float>(v598_data[5])) * v402_data);
              v597_acc += ((static_cast<float>(v598_data[6])) * v404_data);
              v597_acc += ((static_cast<float>(v598_data[7])) * v406_data);
              v597_acc += ((static_cast<float>(v598_data[8])) * v408_data);
              v597_acc += ((static_cast<float>(v598_data[9])) * v410_data);
              v597_acc += ((static_cast<float>(v598_data[10])) * v412_data);
              v597_acc += ((static_cast<float>(v598_data[11])) * v414_data);
              r2.template select<16, 1>(112) = v597_acc;
              tensorforge::intel_esimd::simd<float, 16> v623_acc{};
              tensorforge::intel_esimd::simd<float, 16> v624_data = tensorforge::slmLoad<float, 16>(s2 + (96_i32));
              v623_acc += ((static_cast<float>(v624_data[0])) * v392_data);
              v623_acc += ((static_cast<float>(v624_data[1])) * v394_data);
              v623_acc += ((static_cast<float>(v624_data[2])) * v396_data);
              v623_acc += ((static_cast<float>(v624_data[3])) * v398_data);
              v623_acc += ((static_cast<float>(v624_data[4])) * v400_data);
              v623_acc += ((static_cast<float>(v624_data[5])) * v402_data);
              v623_acc += ((static_cast<float>(v624_data[6])) * v404_data);
              v623_acc += ((static_cast<float>(v624_data[7])) * v406_data);
              v623_acc += ((static_cast<float>(v624_data[8])) * v408_data);
              v623_acc += ((static_cast<float>(v624_data[9])) * v410_data);
              v623_acc += ((static_cast<float>(v624_data[10])) * v412_data);
              v623_acc += ((static_cast<float>(v624_data[11])) * v414_data);
              r2.template select<16, 1>(128) = v623_acc;
              tensorforge::intel_esimd::simd<float, 16> v649_acc{};
              tensorforge::intel_esimd::simd<float, 16> v650_data = tensorforge::slmLoad<float, 16>(s2 + (108_i32));
              v649_acc += ((static_cast<float>(v650_data[0])) * v392_data);
              v649_acc += ((static_cast<float>(v650_data[1])) * v394_data);
              v649_acc += ((static_cast<float>(v650_data[2])) * v396_data);
              v649_acc += ((static_cast<float>(v650_data[3])) * v398_data);
              v649_acc += ((static_cast<float>(v650_data[4])) * v400_data);
              v649_acc += ((static_cast<float>(v650_data[5])) * v402_data);
              v649_acc += ((static_cast<float>(v650_data[6])) * v404_data);
              v649_acc += ((static_cast<float>(v650_data[7])) * v406_data);
              v649_acc += ((static_cast<float>(v650_data[8])) * v408_data);
              v649_acc += ((static_cast<float>(v650_data[9])) * v410_data);
              v649_acc += ((static_cast<float>(v650_data[10])) * v412_data);
              v649_acc += ((static_cast<float>(v650_data[11])) * v414_data);
              r2.template select<16, 1>(144) = v649_acc;
              tensorforge::intel_esimd::simd<float, 16> v675_acc{};
              tensorforge::intel_esimd::simd<float, 16> v676_data = tensorforge::slmLoad<float, 16>(s2 + (120_i32));
              v675_acc += ((static_cast<float>(v676_data[0])) * v392_data);
              v675_acc += ((static_cast<float>(v676_data[1])) * v394_data);
              v675_acc += ((static_cast<float>(v676_data[2])) * v396_data);
              v675_acc += ((static_cast<float>(v676_data[3])) * v398_data);
              v675_acc += ((static_cast<float>(v676_data[4])) * v400_data);
              v675_acc += ((static_cast<float>(v676_data[5])) * v402_data);
              v675_acc += ((static_cast<float>(v676_data[6])) * v404_data);
              v675_acc += ((static_cast<float>(v676_data[7])) * v406_data);
              v675_acc += ((static_cast<float>(v676_data[8])) * v408_data);
              v675_acc += ((static_cast<float>(v676_data[9])) * v410_data);
              v675_acc += ((static_cast<float>(v676_data[10])) * v412_data);
              v675_acc += ((static_cast<float>(v676_data[11])) * v414_data);
              r2.template select<16, 1>(160) = v675_acc;
              tensorforge::intel_esimd::simd<float, 16> v701_acc{};
              tensorforge::intel_esimd::simd<float, 16> v702_data = tensorforge::slmLoad<float, 16>(s2 + (132_i32));
              v701_acc += ((static_cast<float>(v702_data[0])) * v392_data);
              v701_acc += ((static_cast<float>(v702_data[1])) * v394_data);
              v701_acc += ((static_cast<float>(v702_data[2])) * v396_data);
              v701_acc += ((static_cast<float>(v702_data[3])) * v398_data);
              v701_acc += ((static_cast<float>(v702_data[4])) * v400_data);
              v701_acc += ((static_cast<float>(v702_data[5])) * v402_data);
              v701_acc += ((static_cast<float>(v702_data[6])) * v404_data);
              v701_acc += ((static_cast<float>(v702_data[7])) * v406_data);
              v701_acc += ((static_cast<float>(v702_data[8])) * v408_data);
              v701_acc += ((static_cast<float>(v702_data[9])) * v410_data);
              v701_acc += ((static_cast<float>(v702_data[10])) * v412_data);
              v701_acc += ((static_cast<float>(v702_data[11])) * v414_data);
              r2.template select<16, 1>(176) = v701_acc;
              tensorforge::intel_esimd::simd<float, 192> r3(0.0f);
              // r3 = abs(glb_m3)
              #pragma unroll
              for (int32_t v728_k1 = 0; v728_k1 < 12; ++v728_k1) {
                tensorforge::intel_esimd::simd<float, 4> v735_data;
                v735_data.copy_from(glb_m3 + ((v728_k1 * 4)));
                r3.template select<4, 1>((v728_k1 * 16)) = (tensorforge::intel_esimd::abs(v735_data));
              }
              // s1 = store{r>s, clear}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v739_z1 = 0; v739_z1 < 12; ++v739_z1) {
                s1[(v739_z1 * 12)] = 0.0f;
              }
              #pragma unroll
              for (int32_t v745_z1 = 0; v745_z1 < 12; ++v745_z1) {
                s1[(8_i32 + (v745_z1 * 12))] = 0.0f;
              }
              #pragma unroll
              for (int32_t v752_i1 = 0; v752_i1 < 12; ++v752_i1) {
                tensorforge::intel_esimd::simd<float, 4> v755_data(r3.template select<4, 1>((v752_i1 * 16)));
                tensorforge::slmStore<float, 4>(s1 + ((4_i32 + (v752_i1 * 12))), v755_data);
              }
              tensorforge::intel_esimd::simd<float, 192> r4(0.0f);
              // ir4 = +(r2 * s1)
              // [(0, 12), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 192> ir4(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v763_data(r2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v764_data(r2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v765_data(r2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v766_data(r2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v767_data(r2.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v768_data(r2.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v769_data(r2.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v770_data(r2.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v771_data(r2.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v772_data(r2.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v773_data(r2.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v774_data(r2.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v775_acc{};
              tensorforge::intel_esimd::simd<float, 16> v779_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v775_acc += ((static_cast<float>(v779_data[0])) * v763_data);
              v775_acc += ((static_cast<float>(v779_data[1])) * v764_data);
              v775_acc += ((static_cast<float>(v779_data[2])) * v765_data);
              v775_acc += ((static_cast<float>(v779_data[3])) * v766_data);
              v775_acc += ((static_cast<float>(v779_data[4])) * v767_data);
              v775_acc += ((static_cast<float>(v779_data[5])) * v768_data);
              v775_acc += ((static_cast<float>(v779_data[6])) * v769_data);
              v775_acc += ((static_cast<float>(v779_data[7])) * v770_data);
              v775_acc += ((static_cast<float>(v779_data[8])) * v771_data);
              v775_acc += ((static_cast<float>(v779_data[9])) * v772_data);
              v775_acc += ((static_cast<float>(v779_data[10])) * v773_data);
              v775_acc += ((static_cast<float>(v779_data[11])) * v774_data);
              ir4.template select<16, 1>(0) = v775_acc;
              tensorforge::intel_esimd::simd<float, 16> v804_acc{};
              tensorforge::intel_esimd::simd<float, 16> v806_data = tensorforge::slmLoad<float, 16>(s1 + (12_i32));
              v804_acc += ((static_cast<float>(v806_data[0])) * v763_data);
              v804_acc += ((static_cast<float>(v806_data[1])) * v764_data);
              v804_acc += ((static_cast<float>(v806_data[2])) * v765_data);
              v804_acc += ((static_cast<float>(v806_data[3])) * v766_data);
              v804_acc += ((static_cast<float>(v806_data[4])) * v767_data);
              v804_acc += ((static_cast<float>(v806_data[5])) * v768_data);
              v804_acc += ((static_cast<float>(v806_data[6])) * v769_data);
              v804_acc += ((static_cast<float>(v806_data[7])) * v770_data);
              v804_acc += ((static_cast<float>(v806_data[8])) * v771_data);
              v804_acc += ((static_cast<float>(v806_data[9])) * v772_data);
              v804_acc += ((static_cast<float>(v806_data[10])) * v773_data);
              v804_acc += ((static_cast<float>(v806_data[11])) * v774_data);
              ir4.template select<16, 1>(16) = v804_acc;
              tensorforge::intel_esimd::simd<float, 16> v831_acc{};
              tensorforge::intel_esimd::simd<float, 16> v833_data = tensorforge::slmLoad<float, 16>(s1 + (24_i32));
              v831_acc += ((static_cast<float>(v833_data[0])) * v763_data);
              v831_acc += ((static_cast<float>(v833_data[1])) * v764_data);
              v831_acc += ((static_cast<float>(v833_data[2])) * v765_data);
              v831_acc += ((static_cast<float>(v833_data[3])) * v766_data);
              v831_acc += ((static_cast<float>(v833_data[4])) * v767_data);
              v831_acc += ((static_cast<float>(v833_data[5])) * v768_data);
              v831_acc += ((static_cast<float>(v833_data[6])) * v769_data);
              v831_acc += ((static_cast<float>(v833_data[7])) * v770_data);
              v831_acc += ((static_cast<float>(v833_data[8])) * v771_data);
              v831_acc += ((static_cast<float>(v833_data[9])) * v772_data);
              v831_acc += ((static_cast<float>(v833_data[10])) * v773_data);
              v831_acc += ((static_cast<float>(v833_data[11])) * v774_data);
              ir4.template select<16, 1>(32) = v831_acc;
              tensorforge::intel_esimd::simd<float, 16> v858_acc{};
              tensorforge::intel_esimd::simd<float, 16> v860_data = tensorforge::slmLoad<float, 16>(s1 + (36_i32));
              v858_acc += ((static_cast<float>(v860_data[0])) * v763_data);
              v858_acc += ((static_cast<float>(v860_data[1])) * v764_data);
              v858_acc += ((static_cast<float>(v860_data[2])) * v765_data);
              v858_acc += ((static_cast<float>(v860_data[3])) * v766_data);
              v858_acc += ((static_cast<float>(v860_data[4])) * v767_data);
              v858_acc += ((static_cast<float>(v860_data[5])) * v768_data);
              v858_acc += ((static_cast<float>(v860_data[6])) * v769_data);
              v858_acc += ((static_cast<float>(v860_data[7])) * v770_data);
              v858_acc += ((static_cast<float>(v860_data[8])) * v771_data);
              v858_acc += ((static_cast<float>(v860_data[9])) * v772_data);
              v858_acc += ((static_cast<float>(v860_data[10])) * v773_data);
              v858_acc += ((static_cast<float>(v860_data[11])) * v774_data);
              ir4.template select<16, 1>(48) = v858_acc;
              tensorforge::intel_esimd::simd<float, 16> v885_acc{};
              tensorforge::intel_esimd::simd<float, 16> v887_data = tensorforge::slmLoad<float, 16>(s1 + (48_i32));
              v885_acc += ((static_cast<float>(v887_data[0])) * v763_data);
              v885_acc += ((static_cast<float>(v887_data[1])) * v764_data);
              v885_acc += ((static_cast<float>(v887_data[2])) * v765_data);
              v885_acc += ((static_cast<float>(v887_data[3])) * v766_data);
              v885_acc += ((static_cast<float>(v887_data[4])) * v767_data);
              v885_acc += ((static_cast<float>(v887_data[5])) * v768_data);
              v885_acc += ((static_cast<float>(v887_data[6])) * v769_data);
              v885_acc += ((static_cast<float>(v887_data[7])) * v770_data);
              v885_acc += ((static_cast<float>(v887_data[8])) * v771_data);
              v885_acc += ((static_cast<float>(v887_data[9])) * v772_data);
              v885_acc += ((static_cast<float>(v887_data[10])) * v773_data);
              v885_acc += ((static_cast<float>(v887_data[11])) * v774_data);
              ir4.template select<16, 1>(64) = v885_acc;
              tensorforge::intel_esimd::simd<float, 16> v912_acc{};
              tensorforge::intel_esimd::simd<float, 16> v914_data = tensorforge::slmLoad<float, 16>(s1 + (60_i32));
              v912_acc += ((static_cast<float>(v914_data[0])) * v763_data);
              v912_acc += ((static_cast<float>(v914_data[1])) * v764_data);
              v912_acc += ((static_cast<float>(v914_data[2])) * v765_data);
              v912_acc += ((static_cast<float>(v914_data[3])) * v766_data);
              v912_acc += ((static_cast<float>(v914_data[4])) * v767_data);
              v912_acc += ((static_cast<float>(v914_data[5])) * v768_data);
              v912_acc += ((static_cast<float>(v914_data[6])) * v769_data);
              v912_acc += ((static_cast<float>(v914_data[7])) * v770_data);
              v912_acc += ((static_cast<float>(v914_data[8])) * v771_data);
              v912_acc += ((static_cast<float>(v914_data[9])) * v772_data);
              v912_acc += ((static_cast<float>(v914_data[10])) * v773_data);
              v912_acc += ((static_cast<float>(v914_data[11])) * v774_data);
              ir4.template select<16, 1>(80) = v912_acc;
              tensorforge::intel_esimd::simd<float, 16> v939_acc{};
              tensorforge::intel_esimd::simd<float, 16> v941_data = tensorforge::slmLoad<float, 16>(s1 + (72_i32));
              v939_acc += ((static_cast<float>(v941_data[0])) * v763_data);
              v939_acc += ((static_cast<float>(v941_data[1])) * v764_data);
              v939_acc += ((static_cast<float>(v941_data[2])) * v765_data);
              v939_acc += ((static_cast<float>(v941_data[3])) * v766_data);
              v939_acc += ((static_cast<float>(v941_data[4])) * v767_data);
              v939_acc += ((static_cast<float>(v941_data[5])) * v768_data);
              v939_acc += ((static_cast<float>(v941_data[6])) * v769_data);
              v939_acc += ((static_cast<float>(v941_data[7])) * v770_data);
              v939_acc += ((static_cast<float>(v941_data[8])) * v771_data);
              v939_acc += ((static_cast<float>(v941_data[9])) * v772_data);
              v939_acc += ((static_cast<float>(v941_data[10])) * v773_data);
              v939_acc += ((static_cast<float>(v941_data[11])) * v774_data);
              ir4.template select<16, 1>(96) = v939_acc;
              tensorforge::intel_esimd::simd<float, 16> v966_acc{};
              tensorforge::intel_esimd::simd<float, 16> v968_data = tensorforge::slmLoad<float, 16>(s1 + (84_i32));
              v966_acc += ((static_cast<float>(v968_data[0])) * v763_data);
              v966_acc += ((static_cast<float>(v968_data[1])) * v764_data);
              v966_acc += ((static_cast<float>(v968_data[2])) * v765_data);
              v966_acc += ((static_cast<float>(v968_data[3])) * v766_data);
              v966_acc += ((static_cast<float>(v968_data[4])) * v767_data);
              v966_acc += ((static_cast<float>(v968_data[5])) * v768_data);
              v966_acc += ((static_cast<float>(v968_data[6])) * v769_data);
              v966_acc += ((static_cast<float>(v968_data[7])) * v770_data);
              v966_acc += ((static_cast<float>(v968_data[8])) * v771_data);
              v966_acc += ((static_cast<float>(v968_data[9])) * v772_data);
              v966_acc += ((static_cast<float>(v968_data[10])) * v773_data);
              v966_acc += ((static_cast<float>(v968_data[11])) * v774_data);
              ir4.template select<16, 1>(112) = v966_acc;
              tensorforge::intel_esimd::simd<float, 16> v993_acc{};
              tensorforge::intel_esimd::simd<float, 16> v995_data = tensorforge::slmLoad<float, 16>(s1 + (96_i32));
              v993_acc += ((static_cast<float>(v995_data[0])) * v763_data);
              v993_acc += ((static_cast<float>(v995_data[1])) * v764_data);
              v993_acc += ((static_cast<float>(v995_data[2])) * v765_data);
              v993_acc += ((static_cast<float>(v995_data[3])) * v766_data);
              v993_acc += ((static_cast<float>(v995_data[4])) * v767_data);
              v993_acc += ((static_cast<float>(v995_data[5])) * v768_data);
              v993_acc += ((static_cast<float>(v995_data[6])) * v769_data);
              v993_acc += ((static_cast<float>(v995_data[7])) * v770_data);
              v993_acc += ((static_cast<float>(v995_data[8])) * v771_data);
              v993_acc += ((static_cast<float>(v995_data[9])) * v772_data);
              v993_acc += ((static_cast<float>(v995_data[10])) * v773_data);
              v993_acc += ((static_cast<float>(v995_data[11])) * v774_data);
              ir4.template select<16, 1>(128) = v993_acc;
              tensorforge::intel_esimd::simd<float, 16> v1020_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1022_data = tensorforge::slmLoad<float, 16>(s1 + (108_i32));
              v1020_acc += ((static_cast<float>(v1022_data[0])) * v763_data);
              v1020_acc += ((static_cast<float>(v1022_data[1])) * v764_data);
              v1020_acc += ((static_cast<float>(v1022_data[2])) * v765_data);
              v1020_acc += ((static_cast<float>(v1022_data[3])) * v766_data);
              v1020_acc += ((static_cast<float>(v1022_data[4])) * v767_data);
              v1020_acc += ((static_cast<float>(v1022_data[5])) * v768_data);
              v1020_acc += ((static_cast<float>(v1022_data[6])) * v769_data);
              v1020_acc += ((static_cast<float>(v1022_data[7])) * v770_data);
              v1020_acc += ((static_cast<float>(v1022_data[8])) * v771_data);
              v1020_acc += ((static_cast<float>(v1022_data[9])) * v772_data);
              v1020_acc += ((static_cast<float>(v1022_data[10])) * v773_data);
              v1020_acc += ((static_cast<float>(v1022_data[11])) * v774_data);
              ir4.template select<16, 1>(144) = v1020_acc;
              tensorforge::intel_esimd::simd<float, 16> v1047_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1049_data = tensorforge::slmLoad<float, 16>(s1 + (120_i32));
              v1047_acc += ((static_cast<float>(v1049_data[0])) * v763_data);
              v1047_acc += ((static_cast<float>(v1049_data[1])) * v764_data);
              v1047_acc += ((static_cast<float>(v1049_data[2])) * v765_data);
              v1047_acc += ((static_cast<float>(v1049_data[3])) * v766_data);
              v1047_acc += ((static_cast<float>(v1049_data[4])) * v767_data);
              v1047_acc += ((static_cast<float>(v1049_data[5])) * v768_data);
              v1047_acc += ((static_cast<float>(v1049_data[6])) * v769_data);
              v1047_acc += ((static_cast<float>(v1049_data[7])) * v770_data);
              v1047_acc += ((static_cast<float>(v1049_data[8])) * v771_data);
              v1047_acc += ((static_cast<float>(v1049_data[9])) * v772_data);
              v1047_acc += ((static_cast<float>(v1049_data[10])) * v773_data);
              v1047_acc += ((static_cast<float>(v1049_data[11])) * v774_data);
              ir4.template select<16, 1>(160) = v1047_acc;
              tensorforge::intel_esimd::simd<float, 16> v1074_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1076_data = tensorforge::slmLoad<float, 16>(s1 + (132_i32));
              v1074_acc += ((static_cast<float>(v1076_data[0])) * v763_data);
              v1074_acc += ((static_cast<float>(v1076_data[1])) * v764_data);
              v1074_acc += ((static_cast<float>(v1076_data[2])) * v765_data);
              v1074_acc += ((static_cast<float>(v1076_data[3])) * v766_data);
              v1074_acc += ((static_cast<float>(v1076_data[4])) * v767_data);
              v1074_acc += ((static_cast<float>(v1076_data[5])) * v768_data);
              v1074_acc += ((static_cast<float>(v1076_data[6])) * v769_data);
              v1074_acc += ((static_cast<float>(v1076_data[7])) * v770_data);
              v1074_acc += ((static_cast<float>(v1076_data[8])) * v771_data);
              v1074_acc += ((static_cast<float>(v1076_data[9])) * v772_data);
              v1074_acc += ((static_cast<float>(v1076_data[10])) * v773_data);
              v1074_acc += ((static_cast<float>(v1076_data[11])) * v774_data);
              ir4.template select<16, 1>(176) = v1074_acc;
              // r4 = ir4
              #pragma unroll
              for (int32_t v1101_n1 = 0; v1101_n1 < 12; ++v1101_n1) {
                int32_t v1102_a = v1101_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v1104_data(ir4.template select<12, 1>(v1102_a));
                r4.template select<12, 1>(v1102_a) = v1104_data;
              }
              // glb_m4 = store{r>g}(r4);
              #pragma unroll
              for (int32_t v1105_i1 = 0; v1105_i1 < 12; ++v1105_i1) {
                tensorforge::intel_esimd::simd<float, 12> v1108_data(r4.template select<12, 1>((v1105_i1 * 16)));
                v1108_data.copy_to(glb_m4 + ((v1105_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}

