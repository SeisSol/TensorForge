// === base name ===
kernel_9a5fe95fa233a908

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_9a5fe95fa233a908 = {{1, 16, 1}, 16, 12, 1, 16, 10240, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_9a5fe95fa233a908(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_9a5fe95fa233a908(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_9a5fe95fa233a908(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_9a5fe95fa233a908(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_9a5fe95fa233a908(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_9a5fe95fa233a908(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_9a5fe95fa233a908(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
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
        //   m3 32×32(4×12) {4..8}×{0..12} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        //   D = abs(N)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2560}],"shared_bytes":10240,"shared_elements":2560,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[12,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"N","bbox":[[4,0],[8,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[4,0],[8,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[4,0],[8,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (160 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 144 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 144 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v8_batchId0 * 48 + 0 + m3_extraOffset];
              tensorforge::intel_esimd::simd<float, 192> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v21_i1 = 0; v21_i1 < 12; ++v21_i1) {
                tensorforge::intel_esimd::simd<float, 12> v26_data;
                v26_data.copy_from(glb_m1 + ((v21_i1 * 12)));
                r0.template select<12, 1>((v21_i1 * 16)) = v26_data;
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v29_ld;
              v29_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v29_ld);
              tensorforge::intel_esimd::simd<float, 64> v30_ld;
              v30_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v30_ld);
              tensorforge::intel_esimd::simd<float, 16> v31_ld;
              v31_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v31_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 192> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 12), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 192> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v34_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v35_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v36_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v37_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v38_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v39_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v40_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v46_acc{};
              tensorforge::intel_esimd::simd<float, 16> v50_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v46_acc += ((static_cast<float>(v50_data[0])) * v34_data);
              v46_acc += ((static_cast<float>(v50_data[1])) * v35_data);
              v46_acc += ((static_cast<float>(v50_data[2])) * v36_data);
              v46_acc += ((static_cast<float>(v50_data[3])) * v37_data);
              v46_acc += ((static_cast<float>(v50_data[4])) * v38_data);
              v46_acc += ((static_cast<float>(v50_data[5])) * v39_data);
              v46_acc += ((static_cast<float>(v50_data[6])) * v40_data);
              v46_acc += ((static_cast<float>(v50_data[7])) * v41_data);
              v46_acc += ((static_cast<float>(v50_data[8])) * v42_data);
              v46_acc += ((static_cast<float>(v50_data[9])) * v43_data);
              v46_acc += ((static_cast<float>(v50_data[10])) * v44_data);
              v46_acc += ((static_cast<float>(v50_data[11])) * v45_data);
              ir1.template select<16, 1>(0) = v46_acc;
              tensorforge::intel_esimd::simd<float, 16> v75_acc{};
              tensorforge::intel_esimd::simd<float, 16> v77_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              v75_acc += ((static_cast<float>(v77_data[0])) * v34_data);
              v75_acc += ((static_cast<float>(v77_data[1])) * v35_data);
              v75_acc += ((static_cast<float>(v77_data[2])) * v36_data);
              v75_acc += ((static_cast<float>(v77_data[3])) * v37_data);
              v75_acc += ((static_cast<float>(v77_data[4])) * v38_data);
              v75_acc += ((static_cast<float>(v77_data[5])) * v39_data);
              v75_acc += ((static_cast<float>(v77_data[6])) * v40_data);
              v75_acc += ((static_cast<float>(v77_data[7])) * v41_data);
              v75_acc += ((static_cast<float>(v77_data[8])) * v42_data);
              v75_acc += ((static_cast<float>(v77_data[9])) * v43_data);
              v75_acc += ((static_cast<float>(v77_data[10])) * v44_data);
              v75_acc += ((static_cast<float>(v77_data[11])) * v45_data);
              ir1.template select<16, 1>(16) = v75_acc;
              tensorforge::intel_esimd::simd<float, 16> v102_acc{};
              tensorforge::intel_esimd::simd<float, 16> v104_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              v102_acc += ((static_cast<float>(v104_data[0])) * v34_data);
              v102_acc += ((static_cast<float>(v104_data[1])) * v35_data);
              v102_acc += ((static_cast<float>(v104_data[2])) * v36_data);
              v102_acc += ((static_cast<float>(v104_data[3])) * v37_data);
              v102_acc += ((static_cast<float>(v104_data[4])) * v38_data);
              v102_acc += ((static_cast<float>(v104_data[5])) * v39_data);
              v102_acc += ((static_cast<float>(v104_data[6])) * v40_data);
              v102_acc += ((static_cast<float>(v104_data[7])) * v41_data);
              v102_acc += ((static_cast<float>(v104_data[8])) * v42_data);
              v102_acc += ((static_cast<float>(v104_data[9])) * v43_data);
              v102_acc += ((static_cast<float>(v104_data[10])) * v44_data);
              v102_acc += ((static_cast<float>(v104_data[11])) * v45_data);
              ir1.template select<16, 1>(32) = v102_acc;
              tensorforge::intel_esimd::simd<float, 16> v129_acc{};
              tensorforge::intel_esimd::simd<float, 16> v131_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              v129_acc += ((static_cast<float>(v131_data[0])) * v34_data);
              v129_acc += ((static_cast<float>(v131_data[1])) * v35_data);
              v129_acc += ((static_cast<float>(v131_data[2])) * v36_data);
              v129_acc += ((static_cast<float>(v131_data[3])) * v37_data);
              v129_acc += ((static_cast<float>(v131_data[4])) * v38_data);
              v129_acc += ((static_cast<float>(v131_data[5])) * v39_data);
              v129_acc += ((static_cast<float>(v131_data[6])) * v40_data);
              v129_acc += ((static_cast<float>(v131_data[7])) * v41_data);
              v129_acc += ((static_cast<float>(v131_data[8])) * v42_data);
              v129_acc += ((static_cast<float>(v131_data[9])) * v43_data);
              v129_acc += ((static_cast<float>(v131_data[10])) * v44_data);
              v129_acc += ((static_cast<float>(v131_data[11])) * v45_data);
              ir1.template select<16, 1>(48) = v129_acc;
              tensorforge::intel_esimd::simd<float, 16> v156_acc{};
              tensorforge::intel_esimd::simd<float, 16> v158_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v156_acc += ((static_cast<float>(v158_data[0])) * v34_data);
              v156_acc += ((static_cast<float>(v158_data[1])) * v35_data);
              v156_acc += ((static_cast<float>(v158_data[2])) * v36_data);
              v156_acc += ((static_cast<float>(v158_data[3])) * v37_data);
              v156_acc += ((static_cast<float>(v158_data[4])) * v38_data);
              v156_acc += ((static_cast<float>(v158_data[5])) * v39_data);
              v156_acc += ((static_cast<float>(v158_data[6])) * v40_data);
              v156_acc += ((static_cast<float>(v158_data[7])) * v41_data);
              v156_acc += ((static_cast<float>(v158_data[8])) * v42_data);
              v156_acc += ((static_cast<float>(v158_data[9])) * v43_data);
              v156_acc += ((static_cast<float>(v158_data[10])) * v44_data);
              v156_acc += ((static_cast<float>(v158_data[11])) * v45_data);
              ir1.template select<16, 1>(64) = v156_acc;
              tensorforge::intel_esimd::simd<float, 16> v183_acc{};
              tensorforge::intel_esimd::simd<float, 16> v185_data = tensorforge::slmLoad<float, 16>(s0 + (60_i32));
              v183_acc += ((static_cast<float>(v185_data[0])) * v34_data);
              v183_acc += ((static_cast<float>(v185_data[1])) * v35_data);
              v183_acc += ((static_cast<float>(v185_data[2])) * v36_data);
              v183_acc += ((static_cast<float>(v185_data[3])) * v37_data);
              v183_acc += ((static_cast<float>(v185_data[4])) * v38_data);
              v183_acc += ((static_cast<float>(v185_data[5])) * v39_data);
              v183_acc += ((static_cast<float>(v185_data[6])) * v40_data);
              v183_acc += ((static_cast<float>(v185_data[7])) * v41_data);
              v183_acc += ((static_cast<float>(v185_data[8])) * v42_data);
              v183_acc += ((static_cast<float>(v185_data[9])) * v43_data);
              v183_acc += ((static_cast<float>(v185_data[10])) * v44_data);
              v183_acc += ((static_cast<float>(v185_data[11])) * v45_data);
              ir1.template select<16, 1>(80) = v183_acc;
              tensorforge::intel_esimd::simd<float, 16> v210_acc{};
              tensorforge::intel_esimd::simd<float, 16> v212_data = tensorforge::slmLoad<float, 16>(s0 + (72_i32));
              v210_acc += ((static_cast<float>(v212_data[0])) * v34_data);
              v210_acc += ((static_cast<float>(v212_data[1])) * v35_data);
              v210_acc += ((static_cast<float>(v212_data[2])) * v36_data);
              v210_acc += ((static_cast<float>(v212_data[3])) * v37_data);
              v210_acc += ((static_cast<float>(v212_data[4])) * v38_data);
              v210_acc += ((static_cast<float>(v212_data[5])) * v39_data);
              v210_acc += ((static_cast<float>(v212_data[6])) * v40_data);
              v210_acc += ((static_cast<float>(v212_data[7])) * v41_data);
              v210_acc += ((static_cast<float>(v212_data[8])) * v42_data);
              v210_acc += ((static_cast<float>(v212_data[9])) * v43_data);
              v210_acc += ((static_cast<float>(v212_data[10])) * v44_data);
              v210_acc += ((static_cast<float>(v212_data[11])) * v45_data);
              ir1.template select<16, 1>(96) = v210_acc;
              tensorforge::intel_esimd::simd<float, 16> v237_acc{};
              tensorforge::intel_esimd::simd<float, 16> v239_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v237_acc += ((static_cast<float>(v239_data[0])) * v34_data);
              v237_acc += ((static_cast<float>(v239_data[1])) * v35_data);
              v237_acc += ((static_cast<float>(v239_data[2])) * v36_data);
              v237_acc += ((static_cast<float>(v239_data[3])) * v37_data);
              v237_acc += ((static_cast<float>(v239_data[4])) * v38_data);
              v237_acc += ((static_cast<float>(v239_data[5])) * v39_data);
              v237_acc += ((static_cast<float>(v239_data[6])) * v40_data);
              v237_acc += ((static_cast<float>(v239_data[7])) * v41_data);
              v237_acc += ((static_cast<float>(v239_data[8])) * v42_data);
              v237_acc += ((static_cast<float>(v239_data[9])) * v43_data);
              v237_acc += ((static_cast<float>(v239_data[10])) * v44_data);
              v237_acc += ((static_cast<float>(v239_data[11])) * v45_data);
              ir1.template select<16, 1>(112) = v237_acc;
              tensorforge::intel_esimd::simd<float, 16> v264_acc{};
              tensorforge::intel_esimd::simd<float, 16> v266_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              v264_acc += ((static_cast<float>(v266_data[0])) * v34_data);
              v264_acc += ((static_cast<float>(v266_data[1])) * v35_data);
              v264_acc += ((static_cast<float>(v266_data[2])) * v36_data);
              v264_acc += ((static_cast<float>(v266_data[3])) * v37_data);
              v264_acc += ((static_cast<float>(v266_data[4])) * v38_data);
              v264_acc += ((static_cast<float>(v266_data[5])) * v39_data);
              v264_acc += ((static_cast<float>(v266_data[6])) * v40_data);
              v264_acc += ((static_cast<float>(v266_data[7])) * v41_data);
              v264_acc += ((static_cast<float>(v266_data[8])) * v42_data);
              v264_acc += ((static_cast<float>(v266_data[9])) * v43_data);
              v264_acc += ((static_cast<float>(v266_data[10])) * v44_data);
              v264_acc += ((static_cast<float>(v266_data[11])) * v45_data);
              ir1.template select<16, 1>(128) = v264_acc;
              tensorforge::intel_esimd::simd<float, 16> v291_acc{};
              tensorforge::intel_esimd::simd<float, 16> v293_data = tensorforge::slmLoad<float, 16>(s0 + (108_i32));
              v291_acc += ((static_cast<float>(v293_data[0])) * v34_data);
              v291_acc += ((static_cast<float>(v293_data[1])) * v35_data);
              v291_acc += ((static_cast<float>(v293_data[2])) * v36_data);
              v291_acc += ((static_cast<float>(v293_data[3])) * v37_data);
              v291_acc += ((static_cast<float>(v293_data[4])) * v38_data);
              v291_acc += ((static_cast<float>(v293_data[5])) * v39_data);
              v291_acc += ((static_cast<float>(v293_data[6])) * v40_data);
              v291_acc += ((static_cast<float>(v293_data[7])) * v41_data);
              v291_acc += ((static_cast<float>(v293_data[8])) * v42_data);
              v291_acc += ((static_cast<float>(v293_data[9])) * v43_data);
              v291_acc += ((static_cast<float>(v293_data[10])) * v44_data);
              v291_acc += ((static_cast<float>(v293_data[11])) * v45_data);
              ir1.template select<16, 1>(144) = v291_acc;
              tensorforge::intel_esimd::simd<float, 16> v318_acc{};
              tensorforge::intel_esimd::simd<float, 16> v320_data = tensorforge::slmLoad<float, 16>(s0 + (120_i32));
              v318_acc += ((static_cast<float>(v320_data[0])) * v34_data);
              v318_acc += ((static_cast<float>(v320_data[1])) * v35_data);
              v318_acc += ((static_cast<float>(v320_data[2])) * v36_data);
              v318_acc += ((static_cast<float>(v320_data[3])) * v37_data);
              v318_acc += ((static_cast<float>(v320_data[4])) * v38_data);
              v318_acc += ((static_cast<float>(v320_data[5])) * v39_data);
              v318_acc += ((static_cast<float>(v320_data[6])) * v40_data);
              v318_acc += ((static_cast<float>(v320_data[7])) * v41_data);
              v318_acc += ((static_cast<float>(v320_data[8])) * v42_data);
              v318_acc += ((static_cast<float>(v320_data[9])) * v43_data);
              v318_acc += ((static_cast<float>(v320_data[10])) * v44_data);
              v318_acc += ((static_cast<float>(v320_data[11])) * v45_data);
              ir1.template select<16, 1>(160) = v318_acc;
              tensorforge::intel_esimd::simd<float, 16> v345_acc{};
              tensorforge::intel_esimd::simd<float, 16> v347_data = tensorforge::slmLoad<float, 16>(s0 + (132_i32));
              v345_acc += ((static_cast<float>(v347_data[0])) * v34_data);
              v345_acc += ((static_cast<float>(v347_data[1])) * v35_data);
              v345_acc += ((static_cast<float>(v347_data[2])) * v36_data);
              v345_acc += ((static_cast<float>(v347_data[3])) * v37_data);
              v345_acc += ((static_cast<float>(v347_data[4])) * v38_data);
              v345_acc += ((static_cast<float>(v347_data[5])) * v39_data);
              v345_acc += ((static_cast<float>(v347_data[6])) * v40_data);
              v345_acc += ((static_cast<float>(v347_data[7])) * v41_data);
              v345_acc += ((static_cast<float>(v347_data[8])) * v42_data);
              v345_acc += ((static_cast<float>(v347_data[9])) * v43_data);
              v345_acc += ((static_cast<float>(v347_data[10])) * v44_data);
              v345_acc += ((static_cast<float>(v347_data[11])) * v45_data);
              ir1.template select<16, 1>(176) = v345_acc;
              // r1 = ir1
              #pragma unroll
              for (int32_t v372_n1 = 0; v372_n1 < 12; ++v372_n1) {
                int32_t v373_a = v372_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v375_data(ir1.template select<12, 1>(v373_a));
                r1.template select<12, 1>(v373_a) = v375_data;
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v376_i1 = 0; v376_i1 < 12; ++v376_i1) {
                tensorforge::intel_esimd::simd<float, 12> v379_data(r1.template select<12, 1>((v376_i1 * 16)));
                v379_data.copy_to(glb_m0 + ((v376_i1 * 12)));
              }
              tensorforge::intel_esimd::simd<float, 192> r2(0.0f);
              // r2 = abs(glb_m3)
              #pragma unroll
              for (int32_t v385_k1 = 0; v385_k1 < 12; ++v385_k1) {
                tensorforge::intel_esimd::simd<float, 4> v392_data;
                v392_data.copy_from(glb_m3 + ((v385_k1 * 4)));
                r2.template select<4, 1>((v385_k1 * 16)) = (tensorforge::intel_esimd::abs(v392_data));
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v396_i1 = 0; v396_i1 < 12; ++v396_i1) {
                tensorforge::intel_esimd::simd<float, 4> v399_data(r2.template select<4, 1>((v396_i1 * 16)));
                v399_data.copy_to(glb_m0 + ((4_i32 + (v396_i1 * 12))));
              }
              #pragma unroll
              for (int32_t v405_z1 = 0; v405_z1 < 12; ++v405_z1) {
                glb_m0[(v405_z1 * 12)] = 0.0f;
              }
              #pragma unroll
              for (int32_t v411_z1 = 0; v411_z1 < 12; ++v411_z1) {
                glb_m0[(8_i32 + (v411_z1 * 12))] = 0.0f;
              }
            }
          }
        }
      }
    });
  });
}

