// === base name ===
kernel_b744a3fbb4320d97

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b744a3fbb4320d97 = {{1, 16, 1}, 16, 12, 1, 16, 19456, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b744a3fbb4320d97(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b744a3fbb4320d97(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b744a3fbb4320d97(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_b744a3fbb4320d97(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b744a3fbb4320d97(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_b744a3fbb4320d97(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_b744a3fbb4320d97(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
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
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s2 = localShrMem0 + (144);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v10_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v10_batchId0 < numElements0; v10_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v11_ahead1 = v10_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v13_batchId1 = (v11_ahead1 < numElements0) ? v11_ahead1 : v10_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v10_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v10_batchId0 * 144 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v10_batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v10_batchId0 * 144 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v10_batchId0 * 48 + 0 + m3_extraOffset];
              float *const __restrict__ glb_m4 = &m4[v10_batchId0 * 144 + 0 + m4_extraOffset];
              tensorforge::intel_esimd::simd<float, 192> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v24_i1 = 0; v24_i1 < 12; ++v24_i1) {
                tensorforge::intel_esimd::simd<float, 12> v29_data;
                v29_data.copy_from(glb_m0 + ((v24_i1 * 12)));
                r0.template select<12, 1>((v24_i1 * 16)) = v29_data;
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v32_ld;
              v32_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v32_ld);
              tensorforge::intel_esimd::simd<float, 64> v33_ld;
              v33_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v33_ld);
              tensorforge::intel_esimd::simd<float, 16> v34_ld;
              v34_ld.copy_from(glb_m1 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v34_ld);
              // wait(r0 = load{g>r}(glb_m0););
              // s2 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v35_ld;
              v35_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s2 + (0 + 0 + 4 * 0 + 0), v35_ld);
              tensorforge::intel_esimd::simd<float, 64> v36_ld;
              v36_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s2 + (0 + 0 + 4 * 0 + 64), v36_ld);
              tensorforge::intel_esimd::simd<float, 16> v37_ld;
              v37_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s2 + (0 + 0 + 1 * 0 + 128), v37_ld);
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 192> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 12), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 16> v39_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v40_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v48_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v49_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v50_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v51_acc{};
              tensorforge::intel_esimd::simd<float, 16> v55_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v51_acc += ((static_cast<float>(v55_data[0])) * v39_data);
              v51_acc += ((static_cast<float>(v55_data[1])) * v40_data);
              v51_acc += ((static_cast<float>(v55_data[2])) * v41_data);
              v51_acc += ((static_cast<float>(v55_data[3])) * v42_data);
              v51_acc += ((static_cast<float>(v55_data[4])) * v43_data);
              v51_acc += ((static_cast<float>(v55_data[5])) * v44_data);
              v51_acc += ((static_cast<float>(v55_data[6])) * v45_data);
              v51_acc += ((static_cast<float>(v55_data[7])) * v46_data);
              v51_acc += ((static_cast<float>(v55_data[8])) * v47_data);
              v51_acc += ((static_cast<float>(v55_data[9])) * v48_data);
              v51_acc += ((static_cast<float>(v55_data[10])) * v49_data);
              v51_acc += ((static_cast<float>(v55_data[11])) * v50_data);
              r1.template select<16, 1>(0) = v51_acc;
              tensorforge::intel_esimd::simd<float, 16> v80_acc{};
              tensorforge::intel_esimd::simd<float, 16> v82_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              v80_acc += ((static_cast<float>(v82_data[0])) * v39_data);
              v80_acc += ((static_cast<float>(v82_data[1])) * v40_data);
              v80_acc += ((static_cast<float>(v82_data[2])) * v41_data);
              v80_acc += ((static_cast<float>(v82_data[3])) * v42_data);
              v80_acc += ((static_cast<float>(v82_data[4])) * v43_data);
              v80_acc += ((static_cast<float>(v82_data[5])) * v44_data);
              v80_acc += ((static_cast<float>(v82_data[6])) * v45_data);
              v80_acc += ((static_cast<float>(v82_data[7])) * v46_data);
              v80_acc += ((static_cast<float>(v82_data[8])) * v47_data);
              v80_acc += ((static_cast<float>(v82_data[9])) * v48_data);
              v80_acc += ((static_cast<float>(v82_data[10])) * v49_data);
              v80_acc += ((static_cast<float>(v82_data[11])) * v50_data);
              r1.template select<16, 1>(16) = v80_acc;
              tensorforge::intel_esimd::simd<float, 16> v107_acc{};
              tensorforge::intel_esimd::simd<float, 16> v109_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              v107_acc += ((static_cast<float>(v109_data[0])) * v39_data);
              v107_acc += ((static_cast<float>(v109_data[1])) * v40_data);
              v107_acc += ((static_cast<float>(v109_data[2])) * v41_data);
              v107_acc += ((static_cast<float>(v109_data[3])) * v42_data);
              v107_acc += ((static_cast<float>(v109_data[4])) * v43_data);
              v107_acc += ((static_cast<float>(v109_data[5])) * v44_data);
              v107_acc += ((static_cast<float>(v109_data[6])) * v45_data);
              v107_acc += ((static_cast<float>(v109_data[7])) * v46_data);
              v107_acc += ((static_cast<float>(v109_data[8])) * v47_data);
              v107_acc += ((static_cast<float>(v109_data[9])) * v48_data);
              v107_acc += ((static_cast<float>(v109_data[10])) * v49_data);
              v107_acc += ((static_cast<float>(v109_data[11])) * v50_data);
              r1.template select<16, 1>(32) = v107_acc;
              tensorforge::intel_esimd::simd<float, 16> v134_acc{};
              tensorforge::intel_esimd::simd<float, 16> v136_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              v134_acc += ((static_cast<float>(v136_data[0])) * v39_data);
              v134_acc += ((static_cast<float>(v136_data[1])) * v40_data);
              v134_acc += ((static_cast<float>(v136_data[2])) * v41_data);
              v134_acc += ((static_cast<float>(v136_data[3])) * v42_data);
              v134_acc += ((static_cast<float>(v136_data[4])) * v43_data);
              v134_acc += ((static_cast<float>(v136_data[5])) * v44_data);
              v134_acc += ((static_cast<float>(v136_data[6])) * v45_data);
              v134_acc += ((static_cast<float>(v136_data[7])) * v46_data);
              v134_acc += ((static_cast<float>(v136_data[8])) * v47_data);
              v134_acc += ((static_cast<float>(v136_data[9])) * v48_data);
              v134_acc += ((static_cast<float>(v136_data[10])) * v49_data);
              v134_acc += ((static_cast<float>(v136_data[11])) * v50_data);
              r1.template select<16, 1>(48) = v134_acc;
              tensorforge::intel_esimd::simd<float, 16> v161_acc{};
              tensorforge::intel_esimd::simd<float, 16> v163_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v161_acc += ((static_cast<float>(v163_data[0])) * v39_data);
              v161_acc += ((static_cast<float>(v163_data[1])) * v40_data);
              v161_acc += ((static_cast<float>(v163_data[2])) * v41_data);
              v161_acc += ((static_cast<float>(v163_data[3])) * v42_data);
              v161_acc += ((static_cast<float>(v163_data[4])) * v43_data);
              v161_acc += ((static_cast<float>(v163_data[5])) * v44_data);
              v161_acc += ((static_cast<float>(v163_data[6])) * v45_data);
              v161_acc += ((static_cast<float>(v163_data[7])) * v46_data);
              v161_acc += ((static_cast<float>(v163_data[8])) * v47_data);
              v161_acc += ((static_cast<float>(v163_data[9])) * v48_data);
              v161_acc += ((static_cast<float>(v163_data[10])) * v49_data);
              v161_acc += ((static_cast<float>(v163_data[11])) * v50_data);
              r1.template select<16, 1>(64) = v161_acc;
              tensorforge::intel_esimd::simd<float, 16> v188_acc{};
              tensorforge::intel_esimd::simd<float, 16> v190_data = tensorforge::slmLoad<float, 16>(s0 + (60_i32));
              v188_acc += ((static_cast<float>(v190_data[0])) * v39_data);
              v188_acc += ((static_cast<float>(v190_data[1])) * v40_data);
              v188_acc += ((static_cast<float>(v190_data[2])) * v41_data);
              v188_acc += ((static_cast<float>(v190_data[3])) * v42_data);
              v188_acc += ((static_cast<float>(v190_data[4])) * v43_data);
              v188_acc += ((static_cast<float>(v190_data[5])) * v44_data);
              v188_acc += ((static_cast<float>(v190_data[6])) * v45_data);
              v188_acc += ((static_cast<float>(v190_data[7])) * v46_data);
              v188_acc += ((static_cast<float>(v190_data[8])) * v47_data);
              v188_acc += ((static_cast<float>(v190_data[9])) * v48_data);
              v188_acc += ((static_cast<float>(v190_data[10])) * v49_data);
              v188_acc += ((static_cast<float>(v190_data[11])) * v50_data);
              r1.template select<16, 1>(80) = v188_acc;
              tensorforge::intel_esimd::simd<float, 16> v215_acc{};
              tensorforge::intel_esimd::simd<float, 16> v217_data = tensorforge::slmLoad<float, 16>(s0 + (72_i32));
              v215_acc += ((static_cast<float>(v217_data[0])) * v39_data);
              v215_acc += ((static_cast<float>(v217_data[1])) * v40_data);
              v215_acc += ((static_cast<float>(v217_data[2])) * v41_data);
              v215_acc += ((static_cast<float>(v217_data[3])) * v42_data);
              v215_acc += ((static_cast<float>(v217_data[4])) * v43_data);
              v215_acc += ((static_cast<float>(v217_data[5])) * v44_data);
              v215_acc += ((static_cast<float>(v217_data[6])) * v45_data);
              v215_acc += ((static_cast<float>(v217_data[7])) * v46_data);
              v215_acc += ((static_cast<float>(v217_data[8])) * v47_data);
              v215_acc += ((static_cast<float>(v217_data[9])) * v48_data);
              v215_acc += ((static_cast<float>(v217_data[10])) * v49_data);
              v215_acc += ((static_cast<float>(v217_data[11])) * v50_data);
              r1.template select<16, 1>(96) = v215_acc;
              tensorforge::intel_esimd::simd<float, 16> v242_acc{};
              tensorforge::intel_esimd::simd<float, 16> v244_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v242_acc += ((static_cast<float>(v244_data[0])) * v39_data);
              v242_acc += ((static_cast<float>(v244_data[1])) * v40_data);
              v242_acc += ((static_cast<float>(v244_data[2])) * v41_data);
              v242_acc += ((static_cast<float>(v244_data[3])) * v42_data);
              v242_acc += ((static_cast<float>(v244_data[4])) * v43_data);
              v242_acc += ((static_cast<float>(v244_data[5])) * v44_data);
              v242_acc += ((static_cast<float>(v244_data[6])) * v45_data);
              v242_acc += ((static_cast<float>(v244_data[7])) * v46_data);
              v242_acc += ((static_cast<float>(v244_data[8])) * v47_data);
              v242_acc += ((static_cast<float>(v244_data[9])) * v48_data);
              v242_acc += ((static_cast<float>(v244_data[10])) * v49_data);
              v242_acc += ((static_cast<float>(v244_data[11])) * v50_data);
              r1.template select<16, 1>(112) = v242_acc;
              tensorforge::intel_esimd::simd<float, 16> v269_acc{};
              tensorforge::intel_esimd::simd<float, 16> v271_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              v269_acc += ((static_cast<float>(v271_data[0])) * v39_data);
              v269_acc += ((static_cast<float>(v271_data[1])) * v40_data);
              v269_acc += ((static_cast<float>(v271_data[2])) * v41_data);
              v269_acc += ((static_cast<float>(v271_data[3])) * v42_data);
              v269_acc += ((static_cast<float>(v271_data[4])) * v43_data);
              v269_acc += ((static_cast<float>(v271_data[5])) * v44_data);
              v269_acc += ((static_cast<float>(v271_data[6])) * v45_data);
              v269_acc += ((static_cast<float>(v271_data[7])) * v46_data);
              v269_acc += ((static_cast<float>(v271_data[8])) * v47_data);
              v269_acc += ((static_cast<float>(v271_data[9])) * v48_data);
              v269_acc += ((static_cast<float>(v271_data[10])) * v49_data);
              v269_acc += ((static_cast<float>(v271_data[11])) * v50_data);
              r1.template select<16, 1>(128) = v269_acc;
              tensorforge::intel_esimd::simd<float, 16> v296_acc{};
              tensorforge::intel_esimd::simd<float, 16> v298_data = tensorforge::slmLoad<float, 16>(s0 + (108_i32));
              v296_acc += ((static_cast<float>(v298_data[0])) * v39_data);
              v296_acc += ((static_cast<float>(v298_data[1])) * v40_data);
              v296_acc += ((static_cast<float>(v298_data[2])) * v41_data);
              v296_acc += ((static_cast<float>(v298_data[3])) * v42_data);
              v296_acc += ((static_cast<float>(v298_data[4])) * v43_data);
              v296_acc += ((static_cast<float>(v298_data[5])) * v44_data);
              v296_acc += ((static_cast<float>(v298_data[6])) * v45_data);
              v296_acc += ((static_cast<float>(v298_data[7])) * v46_data);
              v296_acc += ((static_cast<float>(v298_data[8])) * v47_data);
              v296_acc += ((static_cast<float>(v298_data[9])) * v48_data);
              v296_acc += ((static_cast<float>(v298_data[10])) * v49_data);
              v296_acc += ((static_cast<float>(v298_data[11])) * v50_data);
              r1.template select<16, 1>(144) = v296_acc;
              tensorforge::intel_esimd::simd<float, 16> v323_acc{};
              tensorforge::intel_esimd::simd<float, 16> v325_data = tensorforge::slmLoad<float, 16>(s0 + (120_i32));
              v323_acc += ((static_cast<float>(v325_data[0])) * v39_data);
              v323_acc += ((static_cast<float>(v325_data[1])) * v40_data);
              v323_acc += ((static_cast<float>(v325_data[2])) * v41_data);
              v323_acc += ((static_cast<float>(v325_data[3])) * v42_data);
              v323_acc += ((static_cast<float>(v325_data[4])) * v43_data);
              v323_acc += ((static_cast<float>(v325_data[5])) * v44_data);
              v323_acc += ((static_cast<float>(v325_data[6])) * v45_data);
              v323_acc += ((static_cast<float>(v325_data[7])) * v46_data);
              v323_acc += ((static_cast<float>(v325_data[8])) * v47_data);
              v323_acc += ((static_cast<float>(v325_data[9])) * v48_data);
              v323_acc += ((static_cast<float>(v325_data[10])) * v49_data);
              v323_acc += ((static_cast<float>(v325_data[11])) * v50_data);
              r1.template select<16, 1>(160) = v323_acc;
              tensorforge::intel_esimd::simd<float, 16> v350_acc{};
              tensorforge::intel_esimd::simd<float, 16> v352_data = tensorforge::slmLoad<float, 16>(s0 + (132_i32));
              v350_acc += ((static_cast<float>(v352_data[0])) * v39_data);
              v350_acc += ((static_cast<float>(v352_data[1])) * v40_data);
              v350_acc += ((static_cast<float>(v352_data[2])) * v41_data);
              v350_acc += ((static_cast<float>(v352_data[3])) * v42_data);
              v350_acc += ((static_cast<float>(v352_data[4])) * v43_data);
              v350_acc += ((static_cast<float>(v352_data[5])) * v44_data);
              v350_acc += ((static_cast<float>(v352_data[6])) * v45_data);
              v350_acc += ((static_cast<float>(v352_data[7])) * v46_data);
              v350_acc += ((static_cast<float>(v352_data[8])) * v47_data);
              v350_acc += ((static_cast<float>(v352_data[9])) * v48_data);
              v350_acc += ((static_cast<float>(v352_data[10])) * v49_data);
              v350_acc += ((static_cast<float>(v352_data[11])) * v50_data);
              r1.template select<16, 1>(176) = v350_acc;
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v377_i1 = 0; v377_i1 < 12; ++v377_i1) {
                tensorforge::intel_esimd::simd<float, 12> v380_data(r1.template select<12, 1>((v377_i1 * 16)));
                tensorforge::slmStore<float, 12>(s1 + ((v377_i1 * 12)), v380_data);
              }
              // wait(s2 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 192> r2(0.0f);
              // r2 = +(s1 * s2) + None
              // [(0, 12), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 16> v389_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v391_data = tensorforge::slmLoad<float, 16>(s1 + (12_i32));
              tensorforge::intel_esimd::simd<float, 16> v393_data = tensorforge::slmLoad<float, 16>(s1 + (24_i32));
              tensorforge::intel_esimd::simd<float, 16> v395_data = tensorforge::slmLoad<float, 16>(s1 + (36_i32));
              tensorforge::intel_esimd::simd<float, 16> v397_data = tensorforge::slmLoad<float, 16>(s1 + (48_i32));
              tensorforge::intel_esimd::simd<float, 16> v399_data = tensorforge::slmLoad<float, 16>(s1 + (60_i32));
              tensorforge::intel_esimd::simd<float, 16> v401_data = tensorforge::slmLoad<float, 16>(s1 + (72_i32));
              tensorforge::intel_esimd::simd<float, 16> v403_data = tensorforge::slmLoad<float, 16>(s1 + (84_i32));
              tensorforge::intel_esimd::simd<float, 16> v405_data = tensorforge::slmLoad<float, 16>(s1 + (96_i32));
              tensorforge::intel_esimd::simd<float, 16> v407_data = tensorforge::slmLoad<float, 16>(s1 + (108_i32));
              tensorforge::intel_esimd::simd<float, 16> v409_data = tensorforge::slmLoad<float, 16>(s1 + (120_i32));
              tensorforge::intel_esimd::simd<float, 16> v411_data = tensorforge::slmLoad<float, 16>(s1 + (132_i32));
              tensorforge::intel_esimd::simd<float, 16> v412_acc{};
              tensorforge::intel_esimd::simd<float, 16> v413_data = tensorforge::slmLoad<float, 16>(s2 + (0_i32));
              v412_acc += ((static_cast<float>(v413_data[0])) * v389_data);
              v412_acc += ((static_cast<float>(v413_data[1])) * v391_data);
              v412_acc += ((static_cast<float>(v413_data[2])) * v393_data);
              v412_acc += ((static_cast<float>(v413_data[3])) * v395_data);
              v412_acc += ((static_cast<float>(v413_data[4])) * v397_data);
              v412_acc += ((static_cast<float>(v413_data[5])) * v399_data);
              v412_acc += ((static_cast<float>(v413_data[6])) * v401_data);
              v412_acc += ((static_cast<float>(v413_data[7])) * v403_data);
              v412_acc += ((static_cast<float>(v413_data[8])) * v405_data);
              v412_acc += ((static_cast<float>(v413_data[9])) * v407_data);
              v412_acc += ((static_cast<float>(v413_data[10])) * v409_data);
              v412_acc += ((static_cast<float>(v413_data[11])) * v411_data);
              r2.template select<16, 1>(0) = v412_acc;
              tensorforge::intel_esimd::simd<float, 16> v438_acc{};
              tensorforge::intel_esimd::simd<float, 16> v439_data = tensorforge::slmLoad<float, 16>(s2 + (12_i32));
              v438_acc += ((static_cast<float>(v439_data[0])) * v389_data);
              v438_acc += ((static_cast<float>(v439_data[1])) * v391_data);
              v438_acc += ((static_cast<float>(v439_data[2])) * v393_data);
              v438_acc += ((static_cast<float>(v439_data[3])) * v395_data);
              v438_acc += ((static_cast<float>(v439_data[4])) * v397_data);
              v438_acc += ((static_cast<float>(v439_data[5])) * v399_data);
              v438_acc += ((static_cast<float>(v439_data[6])) * v401_data);
              v438_acc += ((static_cast<float>(v439_data[7])) * v403_data);
              v438_acc += ((static_cast<float>(v439_data[8])) * v405_data);
              v438_acc += ((static_cast<float>(v439_data[9])) * v407_data);
              v438_acc += ((static_cast<float>(v439_data[10])) * v409_data);
              v438_acc += ((static_cast<float>(v439_data[11])) * v411_data);
              r2.template select<16, 1>(16) = v438_acc;
              tensorforge::intel_esimd::simd<float, 16> v464_acc{};
              tensorforge::intel_esimd::simd<float, 16> v465_data = tensorforge::slmLoad<float, 16>(s2 + (24_i32));
              v464_acc += ((static_cast<float>(v465_data[0])) * v389_data);
              v464_acc += ((static_cast<float>(v465_data[1])) * v391_data);
              v464_acc += ((static_cast<float>(v465_data[2])) * v393_data);
              v464_acc += ((static_cast<float>(v465_data[3])) * v395_data);
              v464_acc += ((static_cast<float>(v465_data[4])) * v397_data);
              v464_acc += ((static_cast<float>(v465_data[5])) * v399_data);
              v464_acc += ((static_cast<float>(v465_data[6])) * v401_data);
              v464_acc += ((static_cast<float>(v465_data[7])) * v403_data);
              v464_acc += ((static_cast<float>(v465_data[8])) * v405_data);
              v464_acc += ((static_cast<float>(v465_data[9])) * v407_data);
              v464_acc += ((static_cast<float>(v465_data[10])) * v409_data);
              v464_acc += ((static_cast<float>(v465_data[11])) * v411_data);
              r2.template select<16, 1>(32) = v464_acc;
              tensorforge::intel_esimd::simd<float, 16> v490_acc{};
              tensorforge::intel_esimd::simd<float, 16> v491_data = tensorforge::slmLoad<float, 16>(s2 + (36_i32));
              v490_acc += ((static_cast<float>(v491_data[0])) * v389_data);
              v490_acc += ((static_cast<float>(v491_data[1])) * v391_data);
              v490_acc += ((static_cast<float>(v491_data[2])) * v393_data);
              v490_acc += ((static_cast<float>(v491_data[3])) * v395_data);
              v490_acc += ((static_cast<float>(v491_data[4])) * v397_data);
              v490_acc += ((static_cast<float>(v491_data[5])) * v399_data);
              v490_acc += ((static_cast<float>(v491_data[6])) * v401_data);
              v490_acc += ((static_cast<float>(v491_data[7])) * v403_data);
              v490_acc += ((static_cast<float>(v491_data[8])) * v405_data);
              v490_acc += ((static_cast<float>(v491_data[9])) * v407_data);
              v490_acc += ((static_cast<float>(v491_data[10])) * v409_data);
              v490_acc += ((static_cast<float>(v491_data[11])) * v411_data);
              r2.template select<16, 1>(48) = v490_acc;
              tensorforge::intel_esimd::simd<float, 16> v516_acc{};
              tensorforge::intel_esimd::simd<float, 16> v517_data = tensorforge::slmLoad<float, 16>(s2 + (48_i32));
              v516_acc += ((static_cast<float>(v517_data[0])) * v389_data);
              v516_acc += ((static_cast<float>(v517_data[1])) * v391_data);
              v516_acc += ((static_cast<float>(v517_data[2])) * v393_data);
              v516_acc += ((static_cast<float>(v517_data[3])) * v395_data);
              v516_acc += ((static_cast<float>(v517_data[4])) * v397_data);
              v516_acc += ((static_cast<float>(v517_data[5])) * v399_data);
              v516_acc += ((static_cast<float>(v517_data[6])) * v401_data);
              v516_acc += ((static_cast<float>(v517_data[7])) * v403_data);
              v516_acc += ((static_cast<float>(v517_data[8])) * v405_data);
              v516_acc += ((static_cast<float>(v517_data[9])) * v407_data);
              v516_acc += ((static_cast<float>(v517_data[10])) * v409_data);
              v516_acc += ((static_cast<float>(v517_data[11])) * v411_data);
              r2.template select<16, 1>(64) = v516_acc;
              tensorforge::intel_esimd::simd<float, 16> v542_acc{};
              tensorforge::intel_esimd::simd<float, 16> v543_data = tensorforge::slmLoad<float, 16>(s2 + (60_i32));
              v542_acc += ((static_cast<float>(v543_data[0])) * v389_data);
              v542_acc += ((static_cast<float>(v543_data[1])) * v391_data);
              v542_acc += ((static_cast<float>(v543_data[2])) * v393_data);
              v542_acc += ((static_cast<float>(v543_data[3])) * v395_data);
              v542_acc += ((static_cast<float>(v543_data[4])) * v397_data);
              v542_acc += ((static_cast<float>(v543_data[5])) * v399_data);
              v542_acc += ((static_cast<float>(v543_data[6])) * v401_data);
              v542_acc += ((static_cast<float>(v543_data[7])) * v403_data);
              v542_acc += ((static_cast<float>(v543_data[8])) * v405_data);
              v542_acc += ((static_cast<float>(v543_data[9])) * v407_data);
              v542_acc += ((static_cast<float>(v543_data[10])) * v409_data);
              v542_acc += ((static_cast<float>(v543_data[11])) * v411_data);
              r2.template select<16, 1>(80) = v542_acc;
              tensorforge::intel_esimd::simd<float, 16> v568_acc{};
              tensorforge::intel_esimd::simd<float, 16> v569_data = tensorforge::slmLoad<float, 16>(s2 + (72_i32));
              v568_acc += ((static_cast<float>(v569_data[0])) * v389_data);
              v568_acc += ((static_cast<float>(v569_data[1])) * v391_data);
              v568_acc += ((static_cast<float>(v569_data[2])) * v393_data);
              v568_acc += ((static_cast<float>(v569_data[3])) * v395_data);
              v568_acc += ((static_cast<float>(v569_data[4])) * v397_data);
              v568_acc += ((static_cast<float>(v569_data[5])) * v399_data);
              v568_acc += ((static_cast<float>(v569_data[6])) * v401_data);
              v568_acc += ((static_cast<float>(v569_data[7])) * v403_data);
              v568_acc += ((static_cast<float>(v569_data[8])) * v405_data);
              v568_acc += ((static_cast<float>(v569_data[9])) * v407_data);
              v568_acc += ((static_cast<float>(v569_data[10])) * v409_data);
              v568_acc += ((static_cast<float>(v569_data[11])) * v411_data);
              r2.template select<16, 1>(96) = v568_acc;
              tensorforge::intel_esimd::simd<float, 16> v594_acc{};
              tensorforge::intel_esimd::simd<float, 16> v595_data = tensorforge::slmLoad<float, 16>(s2 + (84_i32));
              v594_acc += ((static_cast<float>(v595_data[0])) * v389_data);
              v594_acc += ((static_cast<float>(v595_data[1])) * v391_data);
              v594_acc += ((static_cast<float>(v595_data[2])) * v393_data);
              v594_acc += ((static_cast<float>(v595_data[3])) * v395_data);
              v594_acc += ((static_cast<float>(v595_data[4])) * v397_data);
              v594_acc += ((static_cast<float>(v595_data[5])) * v399_data);
              v594_acc += ((static_cast<float>(v595_data[6])) * v401_data);
              v594_acc += ((static_cast<float>(v595_data[7])) * v403_data);
              v594_acc += ((static_cast<float>(v595_data[8])) * v405_data);
              v594_acc += ((static_cast<float>(v595_data[9])) * v407_data);
              v594_acc += ((static_cast<float>(v595_data[10])) * v409_data);
              v594_acc += ((static_cast<float>(v595_data[11])) * v411_data);
              r2.template select<16, 1>(112) = v594_acc;
              tensorforge::intel_esimd::simd<float, 16> v620_acc{};
              tensorforge::intel_esimd::simd<float, 16> v621_data = tensorforge::slmLoad<float, 16>(s2 + (96_i32));
              v620_acc += ((static_cast<float>(v621_data[0])) * v389_data);
              v620_acc += ((static_cast<float>(v621_data[1])) * v391_data);
              v620_acc += ((static_cast<float>(v621_data[2])) * v393_data);
              v620_acc += ((static_cast<float>(v621_data[3])) * v395_data);
              v620_acc += ((static_cast<float>(v621_data[4])) * v397_data);
              v620_acc += ((static_cast<float>(v621_data[5])) * v399_data);
              v620_acc += ((static_cast<float>(v621_data[6])) * v401_data);
              v620_acc += ((static_cast<float>(v621_data[7])) * v403_data);
              v620_acc += ((static_cast<float>(v621_data[8])) * v405_data);
              v620_acc += ((static_cast<float>(v621_data[9])) * v407_data);
              v620_acc += ((static_cast<float>(v621_data[10])) * v409_data);
              v620_acc += ((static_cast<float>(v621_data[11])) * v411_data);
              r2.template select<16, 1>(128) = v620_acc;
              tensorforge::intel_esimd::simd<float, 16> v646_acc{};
              tensorforge::intel_esimd::simd<float, 16> v647_data = tensorforge::slmLoad<float, 16>(s2 + (108_i32));
              v646_acc += ((static_cast<float>(v647_data[0])) * v389_data);
              v646_acc += ((static_cast<float>(v647_data[1])) * v391_data);
              v646_acc += ((static_cast<float>(v647_data[2])) * v393_data);
              v646_acc += ((static_cast<float>(v647_data[3])) * v395_data);
              v646_acc += ((static_cast<float>(v647_data[4])) * v397_data);
              v646_acc += ((static_cast<float>(v647_data[5])) * v399_data);
              v646_acc += ((static_cast<float>(v647_data[6])) * v401_data);
              v646_acc += ((static_cast<float>(v647_data[7])) * v403_data);
              v646_acc += ((static_cast<float>(v647_data[8])) * v405_data);
              v646_acc += ((static_cast<float>(v647_data[9])) * v407_data);
              v646_acc += ((static_cast<float>(v647_data[10])) * v409_data);
              v646_acc += ((static_cast<float>(v647_data[11])) * v411_data);
              r2.template select<16, 1>(144) = v646_acc;
              tensorforge::intel_esimd::simd<float, 16> v672_acc{};
              tensorforge::intel_esimd::simd<float, 16> v673_data = tensorforge::slmLoad<float, 16>(s2 + (120_i32));
              v672_acc += ((static_cast<float>(v673_data[0])) * v389_data);
              v672_acc += ((static_cast<float>(v673_data[1])) * v391_data);
              v672_acc += ((static_cast<float>(v673_data[2])) * v393_data);
              v672_acc += ((static_cast<float>(v673_data[3])) * v395_data);
              v672_acc += ((static_cast<float>(v673_data[4])) * v397_data);
              v672_acc += ((static_cast<float>(v673_data[5])) * v399_data);
              v672_acc += ((static_cast<float>(v673_data[6])) * v401_data);
              v672_acc += ((static_cast<float>(v673_data[7])) * v403_data);
              v672_acc += ((static_cast<float>(v673_data[8])) * v405_data);
              v672_acc += ((static_cast<float>(v673_data[9])) * v407_data);
              v672_acc += ((static_cast<float>(v673_data[10])) * v409_data);
              v672_acc += ((static_cast<float>(v673_data[11])) * v411_data);
              r2.template select<16, 1>(160) = v672_acc;
              tensorforge::intel_esimd::simd<float, 16> v698_acc{};
              tensorforge::intel_esimd::simd<float, 16> v699_data = tensorforge::slmLoad<float, 16>(s2 + (132_i32));
              v698_acc += ((static_cast<float>(v699_data[0])) * v389_data);
              v698_acc += ((static_cast<float>(v699_data[1])) * v391_data);
              v698_acc += ((static_cast<float>(v699_data[2])) * v393_data);
              v698_acc += ((static_cast<float>(v699_data[3])) * v395_data);
              v698_acc += ((static_cast<float>(v699_data[4])) * v397_data);
              v698_acc += ((static_cast<float>(v699_data[5])) * v399_data);
              v698_acc += ((static_cast<float>(v699_data[6])) * v401_data);
              v698_acc += ((static_cast<float>(v699_data[7])) * v403_data);
              v698_acc += ((static_cast<float>(v699_data[8])) * v405_data);
              v698_acc += ((static_cast<float>(v699_data[9])) * v407_data);
              v698_acc += ((static_cast<float>(v699_data[10])) * v409_data);
              v698_acc += ((static_cast<float>(v699_data[11])) * v411_data);
              r2.template select<16, 1>(176) = v698_acc;
              tensorforge::intel_esimd::simd<float, 192> r3(0.0f);
              // r3 = abs(glb_m3)
              #pragma unroll
              for (int32_t v725_k1 = 0; v725_k1 < 12; ++v725_k1) {
                tensorforge::intel_esimd::simd<float, 4> v732_data;
                v732_data.copy_from(glb_m3 + ((v725_k1 * 4)));
                r3.template select<4, 1>((v725_k1 * 16)) = (tensorforge::intel_esimd::abs(v732_data));
              }
              // s1 = store{r>s, clear}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v736_z1 = 0; v736_z1 < 12; ++v736_z1) {
                s1[(v736_z1 * 12)] = 0.0f;
              }
              #pragma unroll
              for (int32_t v742_z1 = 0; v742_z1 < 12; ++v742_z1) {
                s1[(8_i32 + (v742_z1 * 12))] = 0.0f;
              }
              #pragma unroll
              for (int32_t v749_i1 = 0; v749_i1 < 12; ++v749_i1) {
                tensorforge::intel_esimd::simd<float, 4> v752_data(r3.template select<4, 1>((v749_i1 * 16)));
                tensorforge::slmStore<float, 4>(s1 + ((4_i32 + (v749_i1 * 12))), v752_data);
              }
              tensorforge::intel_esimd::simd<float, 192> r4(0.0f);
              // ir4 = +(r2 * s1)
              // [(0, 12), (0, 12)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 192> ir4(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v760_data(r2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v761_data(r2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v762_data(r2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v763_data(r2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v764_data(r2.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v765_data(r2.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v766_data(r2.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v767_data(r2.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v768_data(r2.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v769_data(r2.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v770_data(r2.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v771_data(r2.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v772_acc{};
              tensorforge::intel_esimd::simd<float, 16> v776_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v772_acc += ((static_cast<float>(v776_data[0])) * v760_data);
              v772_acc += ((static_cast<float>(v776_data[1])) * v761_data);
              v772_acc += ((static_cast<float>(v776_data[2])) * v762_data);
              v772_acc += ((static_cast<float>(v776_data[3])) * v763_data);
              v772_acc += ((static_cast<float>(v776_data[4])) * v764_data);
              v772_acc += ((static_cast<float>(v776_data[5])) * v765_data);
              v772_acc += ((static_cast<float>(v776_data[6])) * v766_data);
              v772_acc += ((static_cast<float>(v776_data[7])) * v767_data);
              v772_acc += ((static_cast<float>(v776_data[8])) * v768_data);
              v772_acc += ((static_cast<float>(v776_data[9])) * v769_data);
              v772_acc += ((static_cast<float>(v776_data[10])) * v770_data);
              v772_acc += ((static_cast<float>(v776_data[11])) * v771_data);
              ir4.template select<16, 1>(0) = v772_acc;
              tensorforge::intel_esimd::simd<float, 16> v801_acc{};
              tensorforge::intel_esimd::simd<float, 16> v803_data = tensorforge::slmLoad<float, 16>(s1 + (12_i32));
              v801_acc += ((static_cast<float>(v803_data[0])) * v760_data);
              v801_acc += ((static_cast<float>(v803_data[1])) * v761_data);
              v801_acc += ((static_cast<float>(v803_data[2])) * v762_data);
              v801_acc += ((static_cast<float>(v803_data[3])) * v763_data);
              v801_acc += ((static_cast<float>(v803_data[4])) * v764_data);
              v801_acc += ((static_cast<float>(v803_data[5])) * v765_data);
              v801_acc += ((static_cast<float>(v803_data[6])) * v766_data);
              v801_acc += ((static_cast<float>(v803_data[7])) * v767_data);
              v801_acc += ((static_cast<float>(v803_data[8])) * v768_data);
              v801_acc += ((static_cast<float>(v803_data[9])) * v769_data);
              v801_acc += ((static_cast<float>(v803_data[10])) * v770_data);
              v801_acc += ((static_cast<float>(v803_data[11])) * v771_data);
              ir4.template select<16, 1>(16) = v801_acc;
              tensorforge::intel_esimd::simd<float, 16> v828_acc{};
              tensorforge::intel_esimd::simd<float, 16> v830_data = tensorforge::slmLoad<float, 16>(s1 + (24_i32));
              v828_acc += ((static_cast<float>(v830_data[0])) * v760_data);
              v828_acc += ((static_cast<float>(v830_data[1])) * v761_data);
              v828_acc += ((static_cast<float>(v830_data[2])) * v762_data);
              v828_acc += ((static_cast<float>(v830_data[3])) * v763_data);
              v828_acc += ((static_cast<float>(v830_data[4])) * v764_data);
              v828_acc += ((static_cast<float>(v830_data[5])) * v765_data);
              v828_acc += ((static_cast<float>(v830_data[6])) * v766_data);
              v828_acc += ((static_cast<float>(v830_data[7])) * v767_data);
              v828_acc += ((static_cast<float>(v830_data[8])) * v768_data);
              v828_acc += ((static_cast<float>(v830_data[9])) * v769_data);
              v828_acc += ((static_cast<float>(v830_data[10])) * v770_data);
              v828_acc += ((static_cast<float>(v830_data[11])) * v771_data);
              ir4.template select<16, 1>(32) = v828_acc;
              tensorforge::intel_esimd::simd<float, 16> v855_acc{};
              tensorforge::intel_esimd::simd<float, 16> v857_data = tensorforge::slmLoad<float, 16>(s1 + (36_i32));
              v855_acc += ((static_cast<float>(v857_data[0])) * v760_data);
              v855_acc += ((static_cast<float>(v857_data[1])) * v761_data);
              v855_acc += ((static_cast<float>(v857_data[2])) * v762_data);
              v855_acc += ((static_cast<float>(v857_data[3])) * v763_data);
              v855_acc += ((static_cast<float>(v857_data[4])) * v764_data);
              v855_acc += ((static_cast<float>(v857_data[5])) * v765_data);
              v855_acc += ((static_cast<float>(v857_data[6])) * v766_data);
              v855_acc += ((static_cast<float>(v857_data[7])) * v767_data);
              v855_acc += ((static_cast<float>(v857_data[8])) * v768_data);
              v855_acc += ((static_cast<float>(v857_data[9])) * v769_data);
              v855_acc += ((static_cast<float>(v857_data[10])) * v770_data);
              v855_acc += ((static_cast<float>(v857_data[11])) * v771_data);
              ir4.template select<16, 1>(48) = v855_acc;
              tensorforge::intel_esimd::simd<float, 16> v882_acc{};
              tensorforge::intel_esimd::simd<float, 16> v884_data = tensorforge::slmLoad<float, 16>(s1 + (48_i32));
              v882_acc += ((static_cast<float>(v884_data[0])) * v760_data);
              v882_acc += ((static_cast<float>(v884_data[1])) * v761_data);
              v882_acc += ((static_cast<float>(v884_data[2])) * v762_data);
              v882_acc += ((static_cast<float>(v884_data[3])) * v763_data);
              v882_acc += ((static_cast<float>(v884_data[4])) * v764_data);
              v882_acc += ((static_cast<float>(v884_data[5])) * v765_data);
              v882_acc += ((static_cast<float>(v884_data[6])) * v766_data);
              v882_acc += ((static_cast<float>(v884_data[7])) * v767_data);
              v882_acc += ((static_cast<float>(v884_data[8])) * v768_data);
              v882_acc += ((static_cast<float>(v884_data[9])) * v769_data);
              v882_acc += ((static_cast<float>(v884_data[10])) * v770_data);
              v882_acc += ((static_cast<float>(v884_data[11])) * v771_data);
              ir4.template select<16, 1>(64) = v882_acc;
              tensorforge::intel_esimd::simd<float, 16> v909_acc{};
              tensorforge::intel_esimd::simd<float, 16> v911_data = tensorforge::slmLoad<float, 16>(s1 + (60_i32));
              v909_acc += ((static_cast<float>(v911_data[0])) * v760_data);
              v909_acc += ((static_cast<float>(v911_data[1])) * v761_data);
              v909_acc += ((static_cast<float>(v911_data[2])) * v762_data);
              v909_acc += ((static_cast<float>(v911_data[3])) * v763_data);
              v909_acc += ((static_cast<float>(v911_data[4])) * v764_data);
              v909_acc += ((static_cast<float>(v911_data[5])) * v765_data);
              v909_acc += ((static_cast<float>(v911_data[6])) * v766_data);
              v909_acc += ((static_cast<float>(v911_data[7])) * v767_data);
              v909_acc += ((static_cast<float>(v911_data[8])) * v768_data);
              v909_acc += ((static_cast<float>(v911_data[9])) * v769_data);
              v909_acc += ((static_cast<float>(v911_data[10])) * v770_data);
              v909_acc += ((static_cast<float>(v911_data[11])) * v771_data);
              ir4.template select<16, 1>(80) = v909_acc;
              tensorforge::intel_esimd::simd<float, 16> v936_acc{};
              tensorforge::intel_esimd::simd<float, 16> v938_data = tensorforge::slmLoad<float, 16>(s1 + (72_i32));
              v936_acc += ((static_cast<float>(v938_data[0])) * v760_data);
              v936_acc += ((static_cast<float>(v938_data[1])) * v761_data);
              v936_acc += ((static_cast<float>(v938_data[2])) * v762_data);
              v936_acc += ((static_cast<float>(v938_data[3])) * v763_data);
              v936_acc += ((static_cast<float>(v938_data[4])) * v764_data);
              v936_acc += ((static_cast<float>(v938_data[5])) * v765_data);
              v936_acc += ((static_cast<float>(v938_data[6])) * v766_data);
              v936_acc += ((static_cast<float>(v938_data[7])) * v767_data);
              v936_acc += ((static_cast<float>(v938_data[8])) * v768_data);
              v936_acc += ((static_cast<float>(v938_data[9])) * v769_data);
              v936_acc += ((static_cast<float>(v938_data[10])) * v770_data);
              v936_acc += ((static_cast<float>(v938_data[11])) * v771_data);
              ir4.template select<16, 1>(96) = v936_acc;
              tensorforge::intel_esimd::simd<float, 16> v963_acc{};
              tensorforge::intel_esimd::simd<float, 16> v965_data = tensorforge::slmLoad<float, 16>(s1 + (84_i32));
              v963_acc += ((static_cast<float>(v965_data[0])) * v760_data);
              v963_acc += ((static_cast<float>(v965_data[1])) * v761_data);
              v963_acc += ((static_cast<float>(v965_data[2])) * v762_data);
              v963_acc += ((static_cast<float>(v965_data[3])) * v763_data);
              v963_acc += ((static_cast<float>(v965_data[4])) * v764_data);
              v963_acc += ((static_cast<float>(v965_data[5])) * v765_data);
              v963_acc += ((static_cast<float>(v965_data[6])) * v766_data);
              v963_acc += ((static_cast<float>(v965_data[7])) * v767_data);
              v963_acc += ((static_cast<float>(v965_data[8])) * v768_data);
              v963_acc += ((static_cast<float>(v965_data[9])) * v769_data);
              v963_acc += ((static_cast<float>(v965_data[10])) * v770_data);
              v963_acc += ((static_cast<float>(v965_data[11])) * v771_data);
              ir4.template select<16, 1>(112) = v963_acc;
              tensorforge::intel_esimd::simd<float, 16> v990_acc{};
              tensorforge::intel_esimd::simd<float, 16> v992_data = tensorforge::slmLoad<float, 16>(s1 + (96_i32));
              v990_acc += ((static_cast<float>(v992_data[0])) * v760_data);
              v990_acc += ((static_cast<float>(v992_data[1])) * v761_data);
              v990_acc += ((static_cast<float>(v992_data[2])) * v762_data);
              v990_acc += ((static_cast<float>(v992_data[3])) * v763_data);
              v990_acc += ((static_cast<float>(v992_data[4])) * v764_data);
              v990_acc += ((static_cast<float>(v992_data[5])) * v765_data);
              v990_acc += ((static_cast<float>(v992_data[6])) * v766_data);
              v990_acc += ((static_cast<float>(v992_data[7])) * v767_data);
              v990_acc += ((static_cast<float>(v992_data[8])) * v768_data);
              v990_acc += ((static_cast<float>(v992_data[9])) * v769_data);
              v990_acc += ((static_cast<float>(v992_data[10])) * v770_data);
              v990_acc += ((static_cast<float>(v992_data[11])) * v771_data);
              ir4.template select<16, 1>(128) = v990_acc;
              tensorforge::intel_esimd::simd<float, 16> v1017_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1019_data = tensorforge::slmLoad<float, 16>(s1 + (108_i32));
              v1017_acc += ((static_cast<float>(v1019_data[0])) * v760_data);
              v1017_acc += ((static_cast<float>(v1019_data[1])) * v761_data);
              v1017_acc += ((static_cast<float>(v1019_data[2])) * v762_data);
              v1017_acc += ((static_cast<float>(v1019_data[3])) * v763_data);
              v1017_acc += ((static_cast<float>(v1019_data[4])) * v764_data);
              v1017_acc += ((static_cast<float>(v1019_data[5])) * v765_data);
              v1017_acc += ((static_cast<float>(v1019_data[6])) * v766_data);
              v1017_acc += ((static_cast<float>(v1019_data[7])) * v767_data);
              v1017_acc += ((static_cast<float>(v1019_data[8])) * v768_data);
              v1017_acc += ((static_cast<float>(v1019_data[9])) * v769_data);
              v1017_acc += ((static_cast<float>(v1019_data[10])) * v770_data);
              v1017_acc += ((static_cast<float>(v1019_data[11])) * v771_data);
              ir4.template select<16, 1>(144) = v1017_acc;
              tensorforge::intel_esimd::simd<float, 16> v1044_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1046_data = tensorforge::slmLoad<float, 16>(s1 + (120_i32));
              v1044_acc += ((static_cast<float>(v1046_data[0])) * v760_data);
              v1044_acc += ((static_cast<float>(v1046_data[1])) * v761_data);
              v1044_acc += ((static_cast<float>(v1046_data[2])) * v762_data);
              v1044_acc += ((static_cast<float>(v1046_data[3])) * v763_data);
              v1044_acc += ((static_cast<float>(v1046_data[4])) * v764_data);
              v1044_acc += ((static_cast<float>(v1046_data[5])) * v765_data);
              v1044_acc += ((static_cast<float>(v1046_data[6])) * v766_data);
              v1044_acc += ((static_cast<float>(v1046_data[7])) * v767_data);
              v1044_acc += ((static_cast<float>(v1046_data[8])) * v768_data);
              v1044_acc += ((static_cast<float>(v1046_data[9])) * v769_data);
              v1044_acc += ((static_cast<float>(v1046_data[10])) * v770_data);
              v1044_acc += ((static_cast<float>(v1046_data[11])) * v771_data);
              ir4.template select<16, 1>(160) = v1044_acc;
              tensorforge::intel_esimd::simd<float, 16> v1071_acc{};
              tensorforge::intel_esimd::simd<float, 16> v1073_data = tensorforge::slmLoad<float, 16>(s1 + (132_i32));
              v1071_acc += ((static_cast<float>(v1073_data[0])) * v760_data);
              v1071_acc += ((static_cast<float>(v1073_data[1])) * v761_data);
              v1071_acc += ((static_cast<float>(v1073_data[2])) * v762_data);
              v1071_acc += ((static_cast<float>(v1073_data[3])) * v763_data);
              v1071_acc += ((static_cast<float>(v1073_data[4])) * v764_data);
              v1071_acc += ((static_cast<float>(v1073_data[5])) * v765_data);
              v1071_acc += ((static_cast<float>(v1073_data[6])) * v766_data);
              v1071_acc += ((static_cast<float>(v1073_data[7])) * v767_data);
              v1071_acc += ((static_cast<float>(v1073_data[8])) * v768_data);
              v1071_acc += ((static_cast<float>(v1073_data[9])) * v769_data);
              v1071_acc += ((static_cast<float>(v1073_data[10])) * v770_data);
              v1071_acc += ((static_cast<float>(v1073_data[11])) * v771_data);
              ir4.template select<16, 1>(176) = v1071_acc;
              // r4 = ir4
              #pragma unroll
              for (int32_t v1098_n1 = 0; v1098_n1 < 12; ++v1098_n1) {
                int32_t v1099_a = v1098_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v1101_data(ir4.template select<12, 1>(v1099_a));
                r4.template select<12, 1>(v1099_a) = v1101_data;
              }
              // glb_m4 = store{r>g}(r4);
              #pragma unroll
              for (int32_t v1102_i1 = 0; v1102_i1 < 12; ++v1102_i1) {
                tensorforge::intel_esimd::simd<float, 12> v1105_data(r4.template select<12, 1>((v1102_i1 * 16)));
                v1105_data.copy_to(glb_m4 + ((v1102_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}

