// === base name ===
kernel_b92eb5b8b20742a2

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b92eb5b8b20742a2 = {{1, 16, 1}, 16, 12, 1, 16, 10240, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b92eb5b8b20742a2(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b92eb5b8b20742a2(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b92eb5b8b20742a2(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_b92eb5b8b20742a2(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b92eb5b8b20742a2(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_b92eb5b8b20742a2(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_b92eb5b8b20742a2(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2560 * sizeof(float)>(); {
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
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2560}],"shared_bytes":10240,"shared_elements":2560,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[12,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"N","bbox":[[4,0],[8,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[4,0],[8,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"strided","bbox":[[4,0],[8,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (160 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (144);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v5_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v5_batchId0 < numElements0; v5_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v6_ahead1 = v5_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 144 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 144 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v5_batchId0 * 48 + 0 + m3_extraOffset];
              tensorforge::intel_esimd::simd<float, 192> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v18_i1 = 0; v18_i1 < 12; ++v18_i1) {
                tensorforge::intel_esimd::simd<float, 12> v23_data;
                v23_data.copy_from(glb_m1 + ((v18_i1 * 12)));
                r0.template select<12, 1>((v18_i1 * 16)) = v23_data;
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v26_ld;
              v26_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v26_ld);
              tensorforge::intel_esimd::simd<float, 64> v27_ld;
              v27_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v27_ld);
              tensorforge::intel_esimd::simd<float, 16> v28_ld;
              v28_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v28_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 192> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 12), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 192> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v31_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v32_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v33_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v34_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v35_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v36_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v37_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v38_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v39_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v40_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v43_acc{};
              tensorforge::intel_esimd::simd<float, 16> v47_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v43_acc += ((static_cast<float>(v47_data[0])) * v31_data);
              v43_acc += ((static_cast<float>(v47_data[1])) * v32_data);
              v43_acc += ((static_cast<float>(v47_data[2])) * v33_data);
              v43_acc += ((static_cast<float>(v47_data[3])) * v34_data);
              v43_acc += ((static_cast<float>(v47_data[4])) * v35_data);
              v43_acc += ((static_cast<float>(v47_data[5])) * v36_data);
              v43_acc += ((static_cast<float>(v47_data[6])) * v37_data);
              v43_acc += ((static_cast<float>(v47_data[7])) * v38_data);
              v43_acc += ((static_cast<float>(v47_data[8])) * v39_data);
              v43_acc += ((static_cast<float>(v47_data[9])) * v40_data);
              v43_acc += ((static_cast<float>(v47_data[10])) * v41_data);
              v43_acc += ((static_cast<float>(v47_data[11])) * v42_data);
              ir1.template select<16, 1>(0) = v43_acc;
              tensorforge::intel_esimd::simd<float, 16> v72_acc{};
              tensorforge::intel_esimd::simd<float, 16> v74_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              v72_acc += ((static_cast<float>(v74_data[0])) * v31_data);
              v72_acc += ((static_cast<float>(v74_data[1])) * v32_data);
              v72_acc += ((static_cast<float>(v74_data[2])) * v33_data);
              v72_acc += ((static_cast<float>(v74_data[3])) * v34_data);
              v72_acc += ((static_cast<float>(v74_data[4])) * v35_data);
              v72_acc += ((static_cast<float>(v74_data[5])) * v36_data);
              v72_acc += ((static_cast<float>(v74_data[6])) * v37_data);
              v72_acc += ((static_cast<float>(v74_data[7])) * v38_data);
              v72_acc += ((static_cast<float>(v74_data[8])) * v39_data);
              v72_acc += ((static_cast<float>(v74_data[9])) * v40_data);
              v72_acc += ((static_cast<float>(v74_data[10])) * v41_data);
              v72_acc += ((static_cast<float>(v74_data[11])) * v42_data);
              ir1.template select<16, 1>(16) = v72_acc;
              tensorforge::intel_esimd::simd<float, 16> v99_acc{};
              tensorforge::intel_esimd::simd<float, 16> v101_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              v99_acc += ((static_cast<float>(v101_data[0])) * v31_data);
              v99_acc += ((static_cast<float>(v101_data[1])) * v32_data);
              v99_acc += ((static_cast<float>(v101_data[2])) * v33_data);
              v99_acc += ((static_cast<float>(v101_data[3])) * v34_data);
              v99_acc += ((static_cast<float>(v101_data[4])) * v35_data);
              v99_acc += ((static_cast<float>(v101_data[5])) * v36_data);
              v99_acc += ((static_cast<float>(v101_data[6])) * v37_data);
              v99_acc += ((static_cast<float>(v101_data[7])) * v38_data);
              v99_acc += ((static_cast<float>(v101_data[8])) * v39_data);
              v99_acc += ((static_cast<float>(v101_data[9])) * v40_data);
              v99_acc += ((static_cast<float>(v101_data[10])) * v41_data);
              v99_acc += ((static_cast<float>(v101_data[11])) * v42_data);
              ir1.template select<16, 1>(32) = v99_acc;
              tensorforge::intel_esimd::simd<float, 16> v126_acc{};
              tensorforge::intel_esimd::simd<float, 16> v128_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              v126_acc += ((static_cast<float>(v128_data[0])) * v31_data);
              v126_acc += ((static_cast<float>(v128_data[1])) * v32_data);
              v126_acc += ((static_cast<float>(v128_data[2])) * v33_data);
              v126_acc += ((static_cast<float>(v128_data[3])) * v34_data);
              v126_acc += ((static_cast<float>(v128_data[4])) * v35_data);
              v126_acc += ((static_cast<float>(v128_data[5])) * v36_data);
              v126_acc += ((static_cast<float>(v128_data[6])) * v37_data);
              v126_acc += ((static_cast<float>(v128_data[7])) * v38_data);
              v126_acc += ((static_cast<float>(v128_data[8])) * v39_data);
              v126_acc += ((static_cast<float>(v128_data[9])) * v40_data);
              v126_acc += ((static_cast<float>(v128_data[10])) * v41_data);
              v126_acc += ((static_cast<float>(v128_data[11])) * v42_data);
              ir1.template select<16, 1>(48) = v126_acc;
              tensorforge::intel_esimd::simd<float, 16> v153_acc{};
              tensorforge::intel_esimd::simd<float, 16> v155_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v153_acc += ((static_cast<float>(v155_data[0])) * v31_data);
              v153_acc += ((static_cast<float>(v155_data[1])) * v32_data);
              v153_acc += ((static_cast<float>(v155_data[2])) * v33_data);
              v153_acc += ((static_cast<float>(v155_data[3])) * v34_data);
              v153_acc += ((static_cast<float>(v155_data[4])) * v35_data);
              v153_acc += ((static_cast<float>(v155_data[5])) * v36_data);
              v153_acc += ((static_cast<float>(v155_data[6])) * v37_data);
              v153_acc += ((static_cast<float>(v155_data[7])) * v38_data);
              v153_acc += ((static_cast<float>(v155_data[8])) * v39_data);
              v153_acc += ((static_cast<float>(v155_data[9])) * v40_data);
              v153_acc += ((static_cast<float>(v155_data[10])) * v41_data);
              v153_acc += ((static_cast<float>(v155_data[11])) * v42_data);
              ir1.template select<16, 1>(64) = v153_acc;
              tensorforge::intel_esimd::simd<float, 16> v180_acc{};
              tensorforge::intel_esimd::simd<float, 16> v182_data = tensorforge::slmLoad<float, 16>(s0 + (60_i32));
              v180_acc += ((static_cast<float>(v182_data[0])) * v31_data);
              v180_acc += ((static_cast<float>(v182_data[1])) * v32_data);
              v180_acc += ((static_cast<float>(v182_data[2])) * v33_data);
              v180_acc += ((static_cast<float>(v182_data[3])) * v34_data);
              v180_acc += ((static_cast<float>(v182_data[4])) * v35_data);
              v180_acc += ((static_cast<float>(v182_data[5])) * v36_data);
              v180_acc += ((static_cast<float>(v182_data[6])) * v37_data);
              v180_acc += ((static_cast<float>(v182_data[7])) * v38_data);
              v180_acc += ((static_cast<float>(v182_data[8])) * v39_data);
              v180_acc += ((static_cast<float>(v182_data[9])) * v40_data);
              v180_acc += ((static_cast<float>(v182_data[10])) * v41_data);
              v180_acc += ((static_cast<float>(v182_data[11])) * v42_data);
              ir1.template select<16, 1>(80) = v180_acc;
              tensorforge::intel_esimd::simd<float, 16> v207_acc{};
              tensorforge::intel_esimd::simd<float, 16> v209_data = tensorforge::slmLoad<float, 16>(s0 + (72_i32));
              v207_acc += ((static_cast<float>(v209_data[0])) * v31_data);
              v207_acc += ((static_cast<float>(v209_data[1])) * v32_data);
              v207_acc += ((static_cast<float>(v209_data[2])) * v33_data);
              v207_acc += ((static_cast<float>(v209_data[3])) * v34_data);
              v207_acc += ((static_cast<float>(v209_data[4])) * v35_data);
              v207_acc += ((static_cast<float>(v209_data[5])) * v36_data);
              v207_acc += ((static_cast<float>(v209_data[6])) * v37_data);
              v207_acc += ((static_cast<float>(v209_data[7])) * v38_data);
              v207_acc += ((static_cast<float>(v209_data[8])) * v39_data);
              v207_acc += ((static_cast<float>(v209_data[9])) * v40_data);
              v207_acc += ((static_cast<float>(v209_data[10])) * v41_data);
              v207_acc += ((static_cast<float>(v209_data[11])) * v42_data);
              ir1.template select<16, 1>(96) = v207_acc;
              tensorforge::intel_esimd::simd<float, 16> v234_acc{};
              tensorforge::intel_esimd::simd<float, 16> v236_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v234_acc += ((static_cast<float>(v236_data[0])) * v31_data);
              v234_acc += ((static_cast<float>(v236_data[1])) * v32_data);
              v234_acc += ((static_cast<float>(v236_data[2])) * v33_data);
              v234_acc += ((static_cast<float>(v236_data[3])) * v34_data);
              v234_acc += ((static_cast<float>(v236_data[4])) * v35_data);
              v234_acc += ((static_cast<float>(v236_data[5])) * v36_data);
              v234_acc += ((static_cast<float>(v236_data[6])) * v37_data);
              v234_acc += ((static_cast<float>(v236_data[7])) * v38_data);
              v234_acc += ((static_cast<float>(v236_data[8])) * v39_data);
              v234_acc += ((static_cast<float>(v236_data[9])) * v40_data);
              v234_acc += ((static_cast<float>(v236_data[10])) * v41_data);
              v234_acc += ((static_cast<float>(v236_data[11])) * v42_data);
              ir1.template select<16, 1>(112) = v234_acc;
              tensorforge::intel_esimd::simd<float, 16> v261_acc{};
              tensorforge::intel_esimd::simd<float, 16> v263_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              v261_acc += ((static_cast<float>(v263_data[0])) * v31_data);
              v261_acc += ((static_cast<float>(v263_data[1])) * v32_data);
              v261_acc += ((static_cast<float>(v263_data[2])) * v33_data);
              v261_acc += ((static_cast<float>(v263_data[3])) * v34_data);
              v261_acc += ((static_cast<float>(v263_data[4])) * v35_data);
              v261_acc += ((static_cast<float>(v263_data[5])) * v36_data);
              v261_acc += ((static_cast<float>(v263_data[6])) * v37_data);
              v261_acc += ((static_cast<float>(v263_data[7])) * v38_data);
              v261_acc += ((static_cast<float>(v263_data[8])) * v39_data);
              v261_acc += ((static_cast<float>(v263_data[9])) * v40_data);
              v261_acc += ((static_cast<float>(v263_data[10])) * v41_data);
              v261_acc += ((static_cast<float>(v263_data[11])) * v42_data);
              ir1.template select<16, 1>(128) = v261_acc;
              tensorforge::intel_esimd::simd<float, 16> v288_acc{};
              tensorforge::intel_esimd::simd<float, 16> v290_data = tensorforge::slmLoad<float, 16>(s0 + (108_i32));
              v288_acc += ((static_cast<float>(v290_data[0])) * v31_data);
              v288_acc += ((static_cast<float>(v290_data[1])) * v32_data);
              v288_acc += ((static_cast<float>(v290_data[2])) * v33_data);
              v288_acc += ((static_cast<float>(v290_data[3])) * v34_data);
              v288_acc += ((static_cast<float>(v290_data[4])) * v35_data);
              v288_acc += ((static_cast<float>(v290_data[5])) * v36_data);
              v288_acc += ((static_cast<float>(v290_data[6])) * v37_data);
              v288_acc += ((static_cast<float>(v290_data[7])) * v38_data);
              v288_acc += ((static_cast<float>(v290_data[8])) * v39_data);
              v288_acc += ((static_cast<float>(v290_data[9])) * v40_data);
              v288_acc += ((static_cast<float>(v290_data[10])) * v41_data);
              v288_acc += ((static_cast<float>(v290_data[11])) * v42_data);
              ir1.template select<16, 1>(144) = v288_acc;
              tensorforge::intel_esimd::simd<float, 16> v315_acc{};
              tensorforge::intel_esimd::simd<float, 16> v317_data = tensorforge::slmLoad<float, 16>(s0 + (120_i32));
              v315_acc += ((static_cast<float>(v317_data[0])) * v31_data);
              v315_acc += ((static_cast<float>(v317_data[1])) * v32_data);
              v315_acc += ((static_cast<float>(v317_data[2])) * v33_data);
              v315_acc += ((static_cast<float>(v317_data[3])) * v34_data);
              v315_acc += ((static_cast<float>(v317_data[4])) * v35_data);
              v315_acc += ((static_cast<float>(v317_data[5])) * v36_data);
              v315_acc += ((static_cast<float>(v317_data[6])) * v37_data);
              v315_acc += ((static_cast<float>(v317_data[7])) * v38_data);
              v315_acc += ((static_cast<float>(v317_data[8])) * v39_data);
              v315_acc += ((static_cast<float>(v317_data[9])) * v40_data);
              v315_acc += ((static_cast<float>(v317_data[10])) * v41_data);
              v315_acc += ((static_cast<float>(v317_data[11])) * v42_data);
              ir1.template select<16, 1>(160) = v315_acc;
              tensorforge::intel_esimd::simd<float, 16> v342_acc{};
              tensorforge::intel_esimd::simd<float, 16> v344_data = tensorforge::slmLoad<float, 16>(s0 + (132_i32));
              v342_acc += ((static_cast<float>(v344_data[0])) * v31_data);
              v342_acc += ((static_cast<float>(v344_data[1])) * v32_data);
              v342_acc += ((static_cast<float>(v344_data[2])) * v33_data);
              v342_acc += ((static_cast<float>(v344_data[3])) * v34_data);
              v342_acc += ((static_cast<float>(v344_data[4])) * v35_data);
              v342_acc += ((static_cast<float>(v344_data[5])) * v36_data);
              v342_acc += ((static_cast<float>(v344_data[6])) * v37_data);
              v342_acc += ((static_cast<float>(v344_data[7])) * v38_data);
              v342_acc += ((static_cast<float>(v344_data[8])) * v39_data);
              v342_acc += ((static_cast<float>(v344_data[9])) * v40_data);
              v342_acc += ((static_cast<float>(v344_data[10])) * v41_data);
              v342_acc += ((static_cast<float>(v344_data[11])) * v42_data);
              ir1.template select<16, 1>(176) = v342_acc;
              // r1 = ir1
              #pragma unroll
              for (int32_t v369_n1 = 0; v369_n1 < 12; ++v369_n1) {
                int32_t v370_a = v369_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v372_data(ir1.template select<12, 1>(v370_a));
                r1.template select<12, 1>(v370_a) = v372_data;
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v373_i1 = 0; v373_i1 < 12; ++v373_i1) {
                tensorforge::intel_esimd::simd<float, 12> v376_data(r1.template select<12, 1>((v373_i1 * 16)));
                v376_data.copy_to(glb_m0 + ((v373_i1 * 12)));
              }
              tensorforge::intel_esimd::simd<float, 192> r2(0.0f);
              // r2 = abs(glb_m3)
              #pragma unroll
              for (int32_t v382_k1 = 0; v382_k1 < 12; ++v382_k1) {
                tensorforge::intel_esimd::simd<float, 4> v389_data;
                v389_data.copy_from(glb_m3 + ((v382_k1 * 4)));
                r2.template select<4, 1>((v382_k1 * 16)) = (tensorforge::intel_esimd::abs(v389_data));
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v393_i1 = 0; v393_i1 < 12; ++v393_i1) {
                tensorforge::intel_esimd::simd<float, 4> v396_data(r2.template select<4, 1>((v393_i1 * 16)));
                v396_data.copy_to(glb_m0 + ((4_i32 + (v393_i1 * 12))));
              }
              #pragma unroll
              for (int32_t v402_z1 = 0; v402_z1 < 12; ++v402_z1) {
                glb_m0[(v402_z1 * 12)] = 0.0f;
              }
              #pragma unroll
              for (int32_t v408_z1 = 0; v408_z1 < 12; ++v408_z1) {
                glb_m0[(8_i32 + (v408_z1 * 12))] = 0.0f;
              }
            }
          }
        }
      }
    });
  });
}

