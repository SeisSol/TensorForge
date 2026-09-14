// === base name ===
kernel_a1dd1c450f5c9afd

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_a1dd1c450f5c9afd = {{1, 16, 1}, 16, 12, 1, 16, 6144, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_a1dd1c450f5c9afd(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_a1dd1c450f5c9afd(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_a1dd1c450f5c9afd(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 1536 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_a1dd1c450f5c9afd(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_a1dd1c450f5c9afd(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_a1dd1c450f5c9afd(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_a1dd1c450f5c9afd(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<1536 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 6144 B shared, occupancy grid
        // operands:
        //   m0 32×32(12×6) {0..12}×{0..6} strided
        //   m1 32×32(6×6) {0..6}×{0..6} strided
        //   m2 32×32(12×6) {0..12}×{0..6} strided
        //   m3 32×32(12×12) {0..12}×{0..12} strided
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   m2[i,j] = m3[i,k] × t0[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1536}],"shared_bytes":6144,"shared_elements":1536,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B","bbox":[[0,0],[12,6]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[6,6]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,6]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,6]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,6]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[6,6]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]},{"addressing":"pointer_based","bbox":[[0,0],[12,6]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (96 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (80);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v6_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v6_batchId0 < numElements0; v6_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v7_ahead1 = v6_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v6_batchId0 * 36 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v6_batchId0 * 72 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v6_batchId0 * 144 + 0 + m3_extraOffset];
              tensorforge::intel_esimd::simd<float, 96> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v19_i1 = 0; v19_i1 < 6; ++v19_i1) {
                tensorforge::intel_esimd::simd<float, 12> v24_data;
                v24_data.copy_from(glb_m0 + ((v19_i1 * 12)));
                r0.template select<12, 1>((v19_i1 * 16)) = v24_data;
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v27_ld;
              v27_ld.copy_from(glb_m1 + (0 + 0 + 2 * 0 + 0));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 2 * 0 + 0), v27_ld);
              tensorforge::intel_esimd::simd<float, 4> v28_ld;
              v28_ld.copy_from(glb_m1 + (0 + 0 + 1 * 0 + 32));
              tensorforge::slmStore<float, 4>(s0 + (0 + 0 + 1 * 0 + 32), v28_ld);
              // wait(r0 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<float, 192> r2(0.0f);
              // r2 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v30_i1 = 0; v30_i1 < 12; ++v30_i1) {
                tensorforge::intel_esimd::simd<float, 12> v35_data;
                v35_data.copy_from(glb_m3 + ((v30_i1 * 12)));
                r2.template select<12, 1>((v30_i1 * 16)) = v35_data;
              }
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 96> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 12), (0, 6)] [(0, 6)]
              tensorforge::intel_esimd::simd<float, 16> v39_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v40_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v45_acc{};
              tensorforge::intel_esimd::simd<float, 16> v49_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v45_acc += ((static_cast<float>(v49_data[0])) * v39_data);
              v45_acc += ((static_cast<float>(v49_data[1])) * v40_data);
              v45_acc += ((static_cast<float>(v49_data[2])) * v41_data);
              v45_acc += ((static_cast<float>(v49_data[3])) * v42_data);
              v45_acc += ((static_cast<float>(v49_data[4])) * v43_data);
              v45_acc += ((static_cast<float>(v49_data[5])) * v44_data);
              r1.template select<16, 1>(0) = v45_acc;
              tensorforge::intel_esimd::simd<float, 16> v62_acc{};
              tensorforge::intel_esimd::simd<float, 16> v64_data = tensorforge::slmLoad<float, 16>(s0 + (6_i32));
              v62_acc += ((static_cast<float>(v64_data[0])) * v39_data);
              v62_acc += ((static_cast<float>(v64_data[1])) * v40_data);
              v62_acc += ((static_cast<float>(v64_data[2])) * v41_data);
              v62_acc += ((static_cast<float>(v64_data[3])) * v42_data);
              v62_acc += ((static_cast<float>(v64_data[4])) * v43_data);
              v62_acc += ((static_cast<float>(v64_data[5])) * v44_data);
              r1.template select<16, 1>(16) = v62_acc;
              tensorforge::intel_esimd::simd<float, 16> v77_acc{};
              tensorforge::intel_esimd::simd<float, 16> v79_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              v77_acc += ((static_cast<float>(v79_data[0])) * v39_data);
              v77_acc += ((static_cast<float>(v79_data[1])) * v40_data);
              v77_acc += ((static_cast<float>(v79_data[2])) * v41_data);
              v77_acc += ((static_cast<float>(v79_data[3])) * v42_data);
              v77_acc += ((static_cast<float>(v79_data[4])) * v43_data);
              v77_acc += ((static_cast<float>(v79_data[5])) * v44_data);
              r1.template select<16, 1>(32) = v77_acc;
              tensorforge::intel_esimd::simd<float, 16> v92_acc{};
              tensorforge::intel_esimd::simd<float, 16> v94_data = tensorforge::slmLoad<float, 16>(s0 + (18_i32));
              v92_acc += ((static_cast<float>(v94_data[0])) * v39_data);
              v92_acc += ((static_cast<float>(v94_data[1])) * v40_data);
              v92_acc += ((static_cast<float>(v94_data[2])) * v41_data);
              v92_acc += ((static_cast<float>(v94_data[3])) * v42_data);
              v92_acc += ((static_cast<float>(v94_data[4])) * v43_data);
              v92_acc += ((static_cast<float>(v94_data[5])) * v44_data);
              r1.template select<16, 1>(48) = v92_acc;
              tensorforge::intel_esimd::simd<float, 16> v107_acc{};
              tensorforge::intel_esimd::simd<float, 16> v109_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              v107_acc += ((static_cast<float>(v109_data[0])) * v39_data);
              v107_acc += ((static_cast<float>(v109_data[1])) * v40_data);
              v107_acc += ((static_cast<float>(v109_data[2])) * v41_data);
              v107_acc += ((static_cast<float>(v109_data[3])) * v42_data);
              v107_acc += ((static_cast<float>(v109_data[4])) * v43_data);
              v107_acc += ((static_cast<float>(v109_data[5])) * v44_data);
              r1.template select<16, 1>(64) = v107_acc;
              tensorforge::intel_esimd::simd<float, 16> v122_acc{};
              tensorforge::intel_esimd::simd<float, 16> v124_data = tensorforge::slmLoad<float, 16>(s0 + (30_i32));
              v122_acc += ((static_cast<float>(v124_data[0])) * v39_data);
              v122_acc += ((static_cast<float>(v124_data[1])) * v40_data);
              v122_acc += ((static_cast<float>(v124_data[2])) * v41_data);
              v122_acc += ((static_cast<float>(v124_data[3])) * v42_data);
              v122_acc += ((static_cast<float>(v124_data[4])) * v43_data);
              v122_acc += ((static_cast<float>(v124_data[5])) * v44_data);
              r1.template select<16, 1>(80) = v122_acc;
              // wait(r2 = load{g>r}(glb_m3););
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v137_i1 = 0; v137_i1 < 6; ++v137_i1) {
                tensorforge::intel_esimd::simd<float, 12> v140_data(r1.template select<12, 1>((v137_i1 * 16)));
                tensorforge::slmStore<float, 12>(s1 + ((v137_i1 * 12)), v140_data);
              }
              tensorforge::intel_esimd::simd<float, 96> r3(0.0f);
              // ir3 = +(r2 * s1)
              // [(0, 12), (0, 6)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 96> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v147_data(r2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v148_data(r2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v149_data(r2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v150_data(r2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v151_data(r2.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v152_data(r2.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v153_data(r2.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v154_data(r2.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v155_data(r2.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v156_data(r2.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v157_data(r2.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v158_data(r2.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v159_acc{};
              tensorforge::intel_esimd::simd<float, 16> v163_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v159_acc += ((static_cast<float>(v163_data[0])) * v147_data);
              v159_acc += ((static_cast<float>(v163_data[1])) * v148_data);
              v159_acc += ((static_cast<float>(v163_data[2])) * v149_data);
              v159_acc += ((static_cast<float>(v163_data[3])) * v150_data);
              v159_acc += ((static_cast<float>(v163_data[4])) * v151_data);
              v159_acc += ((static_cast<float>(v163_data[5])) * v152_data);
              v159_acc += ((static_cast<float>(v163_data[6])) * v153_data);
              v159_acc += ((static_cast<float>(v163_data[7])) * v154_data);
              v159_acc += ((static_cast<float>(v163_data[8])) * v155_data);
              v159_acc += ((static_cast<float>(v163_data[9])) * v156_data);
              v159_acc += ((static_cast<float>(v163_data[10])) * v157_data);
              v159_acc += ((static_cast<float>(v163_data[11])) * v158_data);
              ir3.template select<16, 1>(0) = v159_acc;
              tensorforge::intel_esimd::simd<float, 16> v188_acc{};
              tensorforge::intel_esimd::simd<float, 16> v190_data = tensorforge::slmLoad<float, 16>(s1 + (12_i32));
              v188_acc += ((static_cast<float>(v190_data[0])) * v147_data);
              v188_acc += ((static_cast<float>(v190_data[1])) * v148_data);
              v188_acc += ((static_cast<float>(v190_data[2])) * v149_data);
              v188_acc += ((static_cast<float>(v190_data[3])) * v150_data);
              v188_acc += ((static_cast<float>(v190_data[4])) * v151_data);
              v188_acc += ((static_cast<float>(v190_data[5])) * v152_data);
              v188_acc += ((static_cast<float>(v190_data[6])) * v153_data);
              v188_acc += ((static_cast<float>(v190_data[7])) * v154_data);
              v188_acc += ((static_cast<float>(v190_data[8])) * v155_data);
              v188_acc += ((static_cast<float>(v190_data[9])) * v156_data);
              v188_acc += ((static_cast<float>(v190_data[10])) * v157_data);
              v188_acc += ((static_cast<float>(v190_data[11])) * v158_data);
              ir3.template select<16, 1>(16) = v188_acc;
              tensorforge::intel_esimd::simd<float, 16> v215_acc{};
              tensorforge::intel_esimd::simd<float, 16> v217_data = tensorforge::slmLoad<float, 16>(s1 + (24_i32));
              v215_acc += ((static_cast<float>(v217_data[0])) * v147_data);
              v215_acc += ((static_cast<float>(v217_data[1])) * v148_data);
              v215_acc += ((static_cast<float>(v217_data[2])) * v149_data);
              v215_acc += ((static_cast<float>(v217_data[3])) * v150_data);
              v215_acc += ((static_cast<float>(v217_data[4])) * v151_data);
              v215_acc += ((static_cast<float>(v217_data[5])) * v152_data);
              v215_acc += ((static_cast<float>(v217_data[6])) * v153_data);
              v215_acc += ((static_cast<float>(v217_data[7])) * v154_data);
              v215_acc += ((static_cast<float>(v217_data[8])) * v155_data);
              v215_acc += ((static_cast<float>(v217_data[9])) * v156_data);
              v215_acc += ((static_cast<float>(v217_data[10])) * v157_data);
              v215_acc += ((static_cast<float>(v217_data[11])) * v158_data);
              ir3.template select<16, 1>(32) = v215_acc;
              tensorforge::intel_esimd::simd<float, 16> v242_acc{};
              tensorforge::intel_esimd::simd<float, 16> v244_data = tensorforge::slmLoad<float, 16>(s1 + (36_i32));
              v242_acc += ((static_cast<float>(v244_data[0])) * v147_data);
              v242_acc += ((static_cast<float>(v244_data[1])) * v148_data);
              v242_acc += ((static_cast<float>(v244_data[2])) * v149_data);
              v242_acc += ((static_cast<float>(v244_data[3])) * v150_data);
              v242_acc += ((static_cast<float>(v244_data[4])) * v151_data);
              v242_acc += ((static_cast<float>(v244_data[5])) * v152_data);
              v242_acc += ((static_cast<float>(v244_data[6])) * v153_data);
              v242_acc += ((static_cast<float>(v244_data[7])) * v154_data);
              v242_acc += ((static_cast<float>(v244_data[8])) * v155_data);
              v242_acc += ((static_cast<float>(v244_data[9])) * v156_data);
              v242_acc += ((static_cast<float>(v244_data[10])) * v157_data);
              v242_acc += ((static_cast<float>(v244_data[11])) * v158_data);
              ir3.template select<16, 1>(48) = v242_acc;
              tensorforge::intel_esimd::simd<float, 16> v269_acc{};
              tensorforge::intel_esimd::simd<float, 16> v271_data = tensorforge::slmLoad<float, 16>(s1 + (48_i32));
              v269_acc += ((static_cast<float>(v271_data[0])) * v147_data);
              v269_acc += ((static_cast<float>(v271_data[1])) * v148_data);
              v269_acc += ((static_cast<float>(v271_data[2])) * v149_data);
              v269_acc += ((static_cast<float>(v271_data[3])) * v150_data);
              v269_acc += ((static_cast<float>(v271_data[4])) * v151_data);
              v269_acc += ((static_cast<float>(v271_data[5])) * v152_data);
              v269_acc += ((static_cast<float>(v271_data[6])) * v153_data);
              v269_acc += ((static_cast<float>(v271_data[7])) * v154_data);
              v269_acc += ((static_cast<float>(v271_data[8])) * v155_data);
              v269_acc += ((static_cast<float>(v271_data[9])) * v156_data);
              v269_acc += ((static_cast<float>(v271_data[10])) * v157_data);
              v269_acc += ((static_cast<float>(v271_data[11])) * v158_data);
              ir3.template select<16, 1>(64) = v269_acc;
              tensorforge::intel_esimd::simd<float, 16> v296_acc{};
              tensorforge::intel_esimd::simd<float, 16> v298_data = tensorforge::slmLoad<float, 16>(s1 + (60_i32));
              v296_acc += ((static_cast<float>(v298_data[0])) * v147_data);
              v296_acc += ((static_cast<float>(v298_data[1])) * v148_data);
              v296_acc += ((static_cast<float>(v298_data[2])) * v149_data);
              v296_acc += ((static_cast<float>(v298_data[3])) * v150_data);
              v296_acc += ((static_cast<float>(v298_data[4])) * v151_data);
              v296_acc += ((static_cast<float>(v298_data[5])) * v152_data);
              v296_acc += ((static_cast<float>(v298_data[6])) * v153_data);
              v296_acc += ((static_cast<float>(v298_data[7])) * v154_data);
              v296_acc += ((static_cast<float>(v298_data[8])) * v155_data);
              v296_acc += ((static_cast<float>(v298_data[9])) * v156_data);
              v296_acc += ((static_cast<float>(v298_data[10])) * v157_data);
              v296_acc += ((static_cast<float>(v298_data[11])) * v158_data);
              ir3.template select<16, 1>(80) = v296_acc;
              // r3 = ir3
              #pragma unroll
              for (int32_t v323_n1 = 0; v323_n1 < 6; ++v323_n1) {
                int32_t v324_a = v323_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v326_data(ir3.template select<12, 1>(v324_a));
                r3.template select<12, 1>(v324_a) = v326_data;
              }
              // glb_m2 = store{r>g}(r3);
              #pragma unroll
              for (int32_t v327_i1 = 0; v327_i1 < 6; ++v327_i1) {
                tensorforge::intel_esimd::simd<float, 12> v330_data(r3.template select<12, 1>((v327_i1 * 16)));
                v330_data.copy_to(glb_m2 + ((v327_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}

