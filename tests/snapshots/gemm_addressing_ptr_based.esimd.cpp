// === base name ===
kernel_9fbe01fbbabdf83a

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_9fbe01fbbabdf83a = {{1, 16, 1}, 16, 16, 1, 16, 17408, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_9fbe01fbbabdf83a(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_9fbe01fbbabdf83a(float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_9fbe01fbbabdf83a(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 4352 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_9fbe01fbbabdf83a(float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_9fbe01fbbabdf83a(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_9fbe01fbbabdf83a(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_9fbe01fbbabdf83a(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<4352 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 17408 B shared, occupancy grid
        // operands:
        //   m0 16×16(16×16) {0..16}×{0..16} pointer_based
        //   m1 16×16(16×16) {0..16}×{0..16} pointer_based
        //   m2 16×16(16×16) {0..16}×{0..16} pointer_based
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":4352}],"shared_bytes":17408,"shared_elements":4352,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"pointer_based","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"pointer_based","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"pointer_based","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"pointer_based","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (272 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (256);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v11_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v11_batchId0 < numElements0; v11_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v12_ahead1 = v11_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v11_batchId0][0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v11_batchId0][0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v11_batchId0][0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
                int32_t v25_lead = v23_i0 * 16;
                #pragma unroll
                for (int32_t v24_i1 = 0; v24_i1 < 16; ++v24_i1) {
                  int32_t v28_a = v25_lead + (v24_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v29_data;
                  v29_data.copy_from(glb_m1 + (v28_a));
                  r0.template select<16, 1>(v28_a) = v29_data;
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v31_ld;
              v31_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v31_ld);
              tensorforge::intel_esimd::simd<float, 64> v32_ld;
              v32_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v32_ld);
              tensorforge::intel_esimd::simd<float, 64> v33_ld;
              v33_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 128));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 128), v33_ld);
              tensorforge::intel_esimd::simd<float, 64> v34_ld;
              v34_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 192));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 192), v34_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 256> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 16), (0, 16)] [(0, 16)]
              tensorforge::intel_esimd::simd<float, 256> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v37_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v38_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v39_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v40_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v48_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v49_data(r0.template select<16, 1>(192));
              tensorforge::intel_esimd::simd<float, 16> v50_data(r0.template select<16, 1>(208));
              tensorforge::intel_esimd::simd<float, 16> v51_data(r0.template select<16, 1>(224));
              tensorforge::intel_esimd::simd<float, 16> v52_data(r0.template select<16, 1>(240));
              tensorforge::intel_esimd::simd<float, 16> v53_acc{};
              tensorforge::intel_esimd::simd<float, 16> v57_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v53_acc += ((static_cast<float>(v57_data[0])) * v37_data);
              v53_acc += ((static_cast<float>(v57_data[1])) * v38_data);
              v53_acc += ((static_cast<float>(v57_data[2])) * v39_data);
              v53_acc += ((static_cast<float>(v57_data[3])) * v40_data);
              v53_acc += ((static_cast<float>(v57_data[4])) * v41_data);
              v53_acc += ((static_cast<float>(v57_data[5])) * v42_data);
              v53_acc += ((static_cast<float>(v57_data[6])) * v43_data);
              v53_acc += ((static_cast<float>(v57_data[7])) * v44_data);
              v53_acc += ((static_cast<float>(v57_data[8])) * v45_data);
              v53_acc += ((static_cast<float>(v57_data[9])) * v46_data);
              v53_acc += ((static_cast<float>(v57_data[10])) * v47_data);
              v53_acc += ((static_cast<float>(v57_data[11])) * v48_data);
              v53_acc += ((static_cast<float>(v57_data[12])) * v49_data);
              v53_acc += ((static_cast<float>(v57_data[13])) * v50_data);
              v53_acc += ((static_cast<float>(v57_data[14])) * v51_data);
              v53_acc += ((static_cast<float>(v57_data[15])) * v52_data);
              ir1.template select<16, 1>(0) = v53_acc;
              tensorforge::intel_esimd::simd<float, 16> v90_acc{};
              tensorforge::intel_esimd::simd<float, 16> v92_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v90_acc += ((static_cast<float>(v92_data[0])) * v37_data);
              v90_acc += ((static_cast<float>(v92_data[1])) * v38_data);
              v90_acc += ((static_cast<float>(v92_data[2])) * v39_data);
              v90_acc += ((static_cast<float>(v92_data[3])) * v40_data);
              v90_acc += ((static_cast<float>(v92_data[4])) * v41_data);
              v90_acc += ((static_cast<float>(v92_data[5])) * v42_data);
              v90_acc += ((static_cast<float>(v92_data[6])) * v43_data);
              v90_acc += ((static_cast<float>(v92_data[7])) * v44_data);
              v90_acc += ((static_cast<float>(v92_data[8])) * v45_data);
              v90_acc += ((static_cast<float>(v92_data[9])) * v46_data);
              v90_acc += ((static_cast<float>(v92_data[10])) * v47_data);
              v90_acc += ((static_cast<float>(v92_data[11])) * v48_data);
              v90_acc += ((static_cast<float>(v92_data[12])) * v49_data);
              v90_acc += ((static_cast<float>(v92_data[13])) * v50_data);
              v90_acc += ((static_cast<float>(v92_data[14])) * v51_data);
              v90_acc += ((static_cast<float>(v92_data[15])) * v52_data);
              ir1.template select<16, 1>(16) = v90_acc;
              tensorforge::intel_esimd::simd<float, 16> v125_acc{};
              tensorforge::intel_esimd::simd<float, 16> v127_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v125_acc += ((static_cast<float>(v127_data[0])) * v37_data);
              v125_acc += ((static_cast<float>(v127_data[1])) * v38_data);
              v125_acc += ((static_cast<float>(v127_data[2])) * v39_data);
              v125_acc += ((static_cast<float>(v127_data[3])) * v40_data);
              v125_acc += ((static_cast<float>(v127_data[4])) * v41_data);
              v125_acc += ((static_cast<float>(v127_data[5])) * v42_data);
              v125_acc += ((static_cast<float>(v127_data[6])) * v43_data);
              v125_acc += ((static_cast<float>(v127_data[7])) * v44_data);
              v125_acc += ((static_cast<float>(v127_data[8])) * v45_data);
              v125_acc += ((static_cast<float>(v127_data[9])) * v46_data);
              v125_acc += ((static_cast<float>(v127_data[10])) * v47_data);
              v125_acc += ((static_cast<float>(v127_data[11])) * v48_data);
              v125_acc += ((static_cast<float>(v127_data[12])) * v49_data);
              v125_acc += ((static_cast<float>(v127_data[13])) * v50_data);
              v125_acc += ((static_cast<float>(v127_data[14])) * v51_data);
              v125_acc += ((static_cast<float>(v127_data[15])) * v52_data);
              ir1.template select<16, 1>(32) = v125_acc;
              tensorforge::intel_esimd::simd<float, 16> v160_acc{};
              tensorforge::intel_esimd::simd<float, 16> v162_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v160_acc += ((static_cast<float>(v162_data[0])) * v37_data);
              v160_acc += ((static_cast<float>(v162_data[1])) * v38_data);
              v160_acc += ((static_cast<float>(v162_data[2])) * v39_data);
              v160_acc += ((static_cast<float>(v162_data[3])) * v40_data);
              v160_acc += ((static_cast<float>(v162_data[4])) * v41_data);
              v160_acc += ((static_cast<float>(v162_data[5])) * v42_data);
              v160_acc += ((static_cast<float>(v162_data[6])) * v43_data);
              v160_acc += ((static_cast<float>(v162_data[7])) * v44_data);
              v160_acc += ((static_cast<float>(v162_data[8])) * v45_data);
              v160_acc += ((static_cast<float>(v162_data[9])) * v46_data);
              v160_acc += ((static_cast<float>(v162_data[10])) * v47_data);
              v160_acc += ((static_cast<float>(v162_data[11])) * v48_data);
              v160_acc += ((static_cast<float>(v162_data[12])) * v49_data);
              v160_acc += ((static_cast<float>(v162_data[13])) * v50_data);
              v160_acc += ((static_cast<float>(v162_data[14])) * v51_data);
              v160_acc += ((static_cast<float>(v162_data[15])) * v52_data);
              ir1.template select<16, 1>(48) = v160_acc;
              tensorforge::intel_esimd::simd<float, 16> v195_acc{};
              tensorforge::intel_esimd::simd<float, 16> v197_data = tensorforge::slmLoad<float, 16>(s0 + (64_i32));
              v195_acc += ((static_cast<float>(v197_data[0])) * v37_data);
              v195_acc += ((static_cast<float>(v197_data[1])) * v38_data);
              v195_acc += ((static_cast<float>(v197_data[2])) * v39_data);
              v195_acc += ((static_cast<float>(v197_data[3])) * v40_data);
              v195_acc += ((static_cast<float>(v197_data[4])) * v41_data);
              v195_acc += ((static_cast<float>(v197_data[5])) * v42_data);
              v195_acc += ((static_cast<float>(v197_data[6])) * v43_data);
              v195_acc += ((static_cast<float>(v197_data[7])) * v44_data);
              v195_acc += ((static_cast<float>(v197_data[8])) * v45_data);
              v195_acc += ((static_cast<float>(v197_data[9])) * v46_data);
              v195_acc += ((static_cast<float>(v197_data[10])) * v47_data);
              v195_acc += ((static_cast<float>(v197_data[11])) * v48_data);
              v195_acc += ((static_cast<float>(v197_data[12])) * v49_data);
              v195_acc += ((static_cast<float>(v197_data[13])) * v50_data);
              v195_acc += ((static_cast<float>(v197_data[14])) * v51_data);
              v195_acc += ((static_cast<float>(v197_data[15])) * v52_data);
              ir1.template select<16, 1>(64) = v195_acc;
              tensorforge::intel_esimd::simd<float, 16> v230_acc{};
              tensorforge::intel_esimd::simd<float, 16> v232_data = tensorforge::slmLoad<float, 16>(s0 + (80_i32));
              v230_acc += ((static_cast<float>(v232_data[0])) * v37_data);
              v230_acc += ((static_cast<float>(v232_data[1])) * v38_data);
              v230_acc += ((static_cast<float>(v232_data[2])) * v39_data);
              v230_acc += ((static_cast<float>(v232_data[3])) * v40_data);
              v230_acc += ((static_cast<float>(v232_data[4])) * v41_data);
              v230_acc += ((static_cast<float>(v232_data[5])) * v42_data);
              v230_acc += ((static_cast<float>(v232_data[6])) * v43_data);
              v230_acc += ((static_cast<float>(v232_data[7])) * v44_data);
              v230_acc += ((static_cast<float>(v232_data[8])) * v45_data);
              v230_acc += ((static_cast<float>(v232_data[9])) * v46_data);
              v230_acc += ((static_cast<float>(v232_data[10])) * v47_data);
              v230_acc += ((static_cast<float>(v232_data[11])) * v48_data);
              v230_acc += ((static_cast<float>(v232_data[12])) * v49_data);
              v230_acc += ((static_cast<float>(v232_data[13])) * v50_data);
              v230_acc += ((static_cast<float>(v232_data[14])) * v51_data);
              v230_acc += ((static_cast<float>(v232_data[15])) * v52_data);
              ir1.template select<16, 1>(80) = v230_acc;
              tensorforge::intel_esimd::simd<float, 16> v265_acc{};
              tensorforge::intel_esimd::simd<float, 16> v267_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              v265_acc += ((static_cast<float>(v267_data[0])) * v37_data);
              v265_acc += ((static_cast<float>(v267_data[1])) * v38_data);
              v265_acc += ((static_cast<float>(v267_data[2])) * v39_data);
              v265_acc += ((static_cast<float>(v267_data[3])) * v40_data);
              v265_acc += ((static_cast<float>(v267_data[4])) * v41_data);
              v265_acc += ((static_cast<float>(v267_data[5])) * v42_data);
              v265_acc += ((static_cast<float>(v267_data[6])) * v43_data);
              v265_acc += ((static_cast<float>(v267_data[7])) * v44_data);
              v265_acc += ((static_cast<float>(v267_data[8])) * v45_data);
              v265_acc += ((static_cast<float>(v267_data[9])) * v46_data);
              v265_acc += ((static_cast<float>(v267_data[10])) * v47_data);
              v265_acc += ((static_cast<float>(v267_data[11])) * v48_data);
              v265_acc += ((static_cast<float>(v267_data[12])) * v49_data);
              v265_acc += ((static_cast<float>(v267_data[13])) * v50_data);
              v265_acc += ((static_cast<float>(v267_data[14])) * v51_data);
              v265_acc += ((static_cast<float>(v267_data[15])) * v52_data);
              ir1.template select<16, 1>(96) = v265_acc;
              tensorforge::intel_esimd::simd<float, 16> v300_acc{};
              tensorforge::intel_esimd::simd<float, 16> v302_data = tensorforge::slmLoad<float, 16>(s0 + (112_i32));
              v300_acc += ((static_cast<float>(v302_data[0])) * v37_data);
              v300_acc += ((static_cast<float>(v302_data[1])) * v38_data);
              v300_acc += ((static_cast<float>(v302_data[2])) * v39_data);
              v300_acc += ((static_cast<float>(v302_data[3])) * v40_data);
              v300_acc += ((static_cast<float>(v302_data[4])) * v41_data);
              v300_acc += ((static_cast<float>(v302_data[5])) * v42_data);
              v300_acc += ((static_cast<float>(v302_data[6])) * v43_data);
              v300_acc += ((static_cast<float>(v302_data[7])) * v44_data);
              v300_acc += ((static_cast<float>(v302_data[8])) * v45_data);
              v300_acc += ((static_cast<float>(v302_data[9])) * v46_data);
              v300_acc += ((static_cast<float>(v302_data[10])) * v47_data);
              v300_acc += ((static_cast<float>(v302_data[11])) * v48_data);
              v300_acc += ((static_cast<float>(v302_data[12])) * v49_data);
              v300_acc += ((static_cast<float>(v302_data[13])) * v50_data);
              v300_acc += ((static_cast<float>(v302_data[14])) * v51_data);
              v300_acc += ((static_cast<float>(v302_data[15])) * v52_data);
              ir1.template select<16, 1>(112) = v300_acc;
              tensorforge::intel_esimd::simd<float, 16> v335_acc{};
              tensorforge::intel_esimd::simd<float, 16> v337_data = tensorforge::slmLoad<float, 16>(s0 + (128_i32));
              v335_acc += ((static_cast<float>(v337_data[0])) * v37_data);
              v335_acc += ((static_cast<float>(v337_data[1])) * v38_data);
              v335_acc += ((static_cast<float>(v337_data[2])) * v39_data);
              v335_acc += ((static_cast<float>(v337_data[3])) * v40_data);
              v335_acc += ((static_cast<float>(v337_data[4])) * v41_data);
              v335_acc += ((static_cast<float>(v337_data[5])) * v42_data);
              v335_acc += ((static_cast<float>(v337_data[6])) * v43_data);
              v335_acc += ((static_cast<float>(v337_data[7])) * v44_data);
              v335_acc += ((static_cast<float>(v337_data[8])) * v45_data);
              v335_acc += ((static_cast<float>(v337_data[9])) * v46_data);
              v335_acc += ((static_cast<float>(v337_data[10])) * v47_data);
              v335_acc += ((static_cast<float>(v337_data[11])) * v48_data);
              v335_acc += ((static_cast<float>(v337_data[12])) * v49_data);
              v335_acc += ((static_cast<float>(v337_data[13])) * v50_data);
              v335_acc += ((static_cast<float>(v337_data[14])) * v51_data);
              v335_acc += ((static_cast<float>(v337_data[15])) * v52_data);
              ir1.template select<16, 1>(128) = v335_acc;
              tensorforge::intel_esimd::simd<float, 16> v370_acc{};
              tensorforge::intel_esimd::simd<float, 16> v372_data = tensorforge::slmLoad<float, 16>(s0 + (144_i32));
              v370_acc += ((static_cast<float>(v372_data[0])) * v37_data);
              v370_acc += ((static_cast<float>(v372_data[1])) * v38_data);
              v370_acc += ((static_cast<float>(v372_data[2])) * v39_data);
              v370_acc += ((static_cast<float>(v372_data[3])) * v40_data);
              v370_acc += ((static_cast<float>(v372_data[4])) * v41_data);
              v370_acc += ((static_cast<float>(v372_data[5])) * v42_data);
              v370_acc += ((static_cast<float>(v372_data[6])) * v43_data);
              v370_acc += ((static_cast<float>(v372_data[7])) * v44_data);
              v370_acc += ((static_cast<float>(v372_data[8])) * v45_data);
              v370_acc += ((static_cast<float>(v372_data[9])) * v46_data);
              v370_acc += ((static_cast<float>(v372_data[10])) * v47_data);
              v370_acc += ((static_cast<float>(v372_data[11])) * v48_data);
              v370_acc += ((static_cast<float>(v372_data[12])) * v49_data);
              v370_acc += ((static_cast<float>(v372_data[13])) * v50_data);
              v370_acc += ((static_cast<float>(v372_data[14])) * v51_data);
              v370_acc += ((static_cast<float>(v372_data[15])) * v52_data);
              ir1.template select<16, 1>(144) = v370_acc;
              tensorforge::intel_esimd::simd<float, 16> v405_acc{};
              tensorforge::intel_esimd::simd<float, 16> v407_data = tensorforge::slmLoad<float, 16>(s0 + (160_i32));
              v405_acc += ((static_cast<float>(v407_data[0])) * v37_data);
              v405_acc += ((static_cast<float>(v407_data[1])) * v38_data);
              v405_acc += ((static_cast<float>(v407_data[2])) * v39_data);
              v405_acc += ((static_cast<float>(v407_data[3])) * v40_data);
              v405_acc += ((static_cast<float>(v407_data[4])) * v41_data);
              v405_acc += ((static_cast<float>(v407_data[5])) * v42_data);
              v405_acc += ((static_cast<float>(v407_data[6])) * v43_data);
              v405_acc += ((static_cast<float>(v407_data[7])) * v44_data);
              v405_acc += ((static_cast<float>(v407_data[8])) * v45_data);
              v405_acc += ((static_cast<float>(v407_data[9])) * v46_data);
              v405_acc += ((static_cast<float>(v407_data[10])) * v47_data);
              v405_acc += ((static_cast<float>(v407_data[11])) * v48_data);
              v405_acc += ((static_cast<float>(v407_data[12])) * v49_data);
              v405_acc += ((static_cast<float>(v407_data[13])) * v50_data);
              v405_acc += ((static_cast<float>(v407_data[14])) * v51_data);
              v405_acc += ((static_cast<float>(v407_data[15])) * v52_data);
              ir1.template select<16, 1>(160) = v405_acc;
              tensorforge::intel_esimd::simd<float, 16> v440_acc{};
              tensorforge::intel_esimd::simd<float, 16> v442_data = tensorforge::slmLoad<float, 16>(s0 + (176_i32));
              v440_acc += ((static_cast<float>(v442_data[0])) * v37_data);
              v440_acc += ((static_cast<float>(v442_data[1])) * v38_data);
              v440_acc += ((static_cast<float>(v442_data[2])) * v39_data);
              v440_acc += ((static_cast<float>(v442_data[3])) * v40_data);
              v440_acc += ((static_cast<float>(v442_data[4])) * v41_data);
              v440_acc += ((static_cast<float>(v442_data[5])) * v42_data);
              v440_acc += ((static_cast<float>(v442_data[6])) * v43_data);
              v440_acc += ((static_cast<float>(v442_data[7])) * v44_data);
              v440_acc += ((static_cast<float>(v442_data[8])) * v45_data);
              v440_acc += ((static_cast<float>(v442_data[9])) * v46_data);
              v440_acc += ((static_cast<float>(v442_data[10])) * v47_data);
              v440_acc += ((static_cast<float>(v442_data[11])) * v48_data);
              v440_acc += ((static_cast<float>(v442_data[12])) * v49_data);
              v440_acc += ((static_cast<float>(v442_data[13])) * v50_data);
              v440_acc += ((static_cast<float>(v442_data[14])) * v51_data);
              v440_acc += ((static_cast<float>(v442_data[15])) * v52_data);
              ir1.template select<16, 1>(176) = v440_acc;
              tensorforge::intel_esimd::simd<float, 16> v475_acc{};
              tensorforge::intel_esimd::simd<float, 16> v477_data = tensorforge::slmLoad<float, 16>(s0 + (192_i32));
              v475_acc += ((static_cast<float>(v477_data[0])) * v37_data);
              v475_acc += ((static_cast<float>(v477_data[1])) * v38_data);
              v475_acc += ((static_cast<float>(v477_data[2])) * v39_data);
              v475_acc += ((static_cast<float>(v477_data[3])) * v40_data);
              v475_acc += ((static_cast<float>(v477_data[4])) * v41_data);
              v475_acc += ((static_cast<float>(v477_data[5])) * v42_data);
              v475_acc += ((static_cast<float>(v477_data[6])) * v43_data);
              v475_acc += ((static_cast<float>(v477_data[7])) * v44_data);
              v475_acc += ((static_cast<float>(v477_data[8])) * v45_data);
              v475_acc += ((static_cast<float>(v477_data[9])) * v46_data);
              v475_acc += ((static_cast<float>(v477_data[10])) * v47_data);
              v475_acc += ((static_cast<float>(v477_data[11])) * v48_data);
              v475_acc += ((static_cast<float>(v477_data[12])) * v49_data);
              v475_acc += ((static_cast<float>(v477_data[13])) * v50_data);
              v475_acc += ((static_cast<float>(v477_data[14])) * v51_data);
              v475_acc += ((static_cast<float>(v477_data[15])) * v52_data);
              ir1.template select<16, 1>(192) = v475_acc;
              tensorforge::intel_esimd::simd<float, 16> v510_acc{};
              tensorforge::intel_esimd::simd<float, 16> v512_data = tensorforge::slmLoad<float, 16>(s0 + (208_i32));
              v510_acc += ((static_cast<float>(v512_data[0])) * v37_data);
              v510_acc += ((static_cast<float>(v512_data[1])) * v38_data);
              v510_acc += ((static_cast<float>(v512_data[2])) * v39_data);
              v510_acc += ((static_cast<float>(v512_data[3])) * v40_data);
              v510_acc += ((static_cast<float>(v512_data[4])) * v41_data);
              v510_acc += ((static_cast<float>(v512_data[5])) * v42_data);
              v510_acc += ((static_cast<float>(v512_data[6])) * v43_data);
              v510_acc += ((static_cast<float>(v512_data[7])) * v44_data);
              v510_acc += ((static_cast<float>(v512_data[8])) * v45_data);
              v510_acc += ((static_cast<float>(v512_data[9])) * v46_data);
              v510_acc += ((static_cast<float>(v512_data[10])) * v47_data);
              v510_acc += ((static_cast<float>(v512_data[11])) * v48_data);
              v510_acc += ((static_cast<float>(v512_data[12])) * v49_data);
              v510_acc += ((static_cast<float>(v512_data[13])) * v50_data);
              v510_acc += ((static_cast<float>(v512_data[14])) * v51_data);
              v510_acc += ((static_cast<float>(v512_data[15])) * v52_data);
              ir1.template select<16, 1>(208) = v510_acc;
              tensorforge::intel_esimd::simd<float, 16> v545_acc{};
              tensorforge::intel_esimd::simd<float, 16> v547_data = tensorforge::slmLoad<float, 16>(s0 + (224_i32));
              v545_acc += ((static_cast<float>(v547_data[0])) * v37_data);
              v545_acc += ((static_cast<float>(v547_data[1])) * v38_data);
              v545_acc += ((static_cast<float>(v547_data[2])) * v39_data);
              v545_acc += ((static_cast<float>(v547_data[3])) * v40_data);
              v545_acc += ((static_cast<float>(v547_data[4])) * v41_data);
              v545_acc += ((static_cast<float>(v547_data[5])) * v42_data);
              v545_acc += ((static_cast<float>(v547_data[6])) * v43_data);
              v545_acc += ((static_cast<float>(v547_data[7])) * v44_data);
              v545_acc += ((static_cast<float>(v547_data[8])) * v45_data);
              v545_acc += ((static_cast<float>(v547_data[9])) * v46_data);
              v545_acc += ((static_cast<float>(v547_data[10])) * v47_data);
              v545_acc += ((static_cast<float>(v547_data[11])) * v48_data);
              v545_acc += ((static_cast<float>(v547_data[12])) * v49_data);
              v545_acc += ((static_cast<float>(v547_data[13])) * v50_data);
              v545_acc += ((static_cast<float>(v547_data[14])) * v51_data);
              v545_acc += ((static_cast<float>(v547_data[15])) * v52_data);
              ir1.template select<16, 1>(224) = v545_acc;
              tensorforge::intel_esimd::simd<float, 16> v580_acc{};
              tensorforge::intel_esimd::simd<float, 16> v582_data = tensorforge::slmLoad<float, 16>(s0 + (240_i32));
              v580_acc += ((static_cast<float>(v582_data[0])) * v37_data);
              v580_acc += ((static_cast<float>(v582_data[1])) * v38_data);
              v580_acc += ((static_cast<float>(v582_data[2])) * v39_data);
              v580_acc += ((static_cast<float>(v582_data[3])) * v40_data);
              v580_acc += ((static_cast<float>(v582_data[4])) * v41_data);
              v580_acc += ((static_cast<float>(v582_data[5])) * v42_data);
              v580_acc += ((static_cast<float>(v582_data[6])) * v43_data);
              v580_acc += ((static_cast<float>(v582_data[7])) * v44_data);
              v580_acc += ((static_cast<float>(v582_data[8])) * v45_data);
              v580_acc += ((static_cast<float>(v582_data[9])) * v46_data);
              v580_acc += ((static_cast<float>(v582_data[10])) * v47_data);
              v580_acc += ((static_cast<float>(v582_data[11])) * v48_data);
              v580_acc += ((static_cast<float>(v582_data[12])) * v49_data);
              v580_acc += ((static_cast<float>(v582_data[13])) * v50_data);
              v580_acc += ((static_cast<float>(v582_data[14])) * v51_data);
              v580_acc += ((static_cast<float>(v582_data[15])) * v52_data);
              ir1.template select<16, 1>(240) = v580_acc;
              // r1 = ir1
              #pragma unroll
              for (int32_t v615_n0 = 0; v615_n0 < 1; ++v615_n0) {
                int32_t v617_a = v615_n0 * 16;
                #pragma unroll
                for (int32_t v616_n1 = 0; v616_n1 < 16; ++v616_n1) {
                  int32_t v619_a = v617_a + (v616_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v620_data(ir1.template select<16, 1>(v619_a));
                  r1.template select<16, 1>(v619_a) = v620_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v621_i0 = 0; v621_i0 < 1; ++v621_i0) {
                int32_t v623_a = v621_i0 * 16;
                #pragma unroll
                for (int32_t v622_i1 = 0; v622_i1 < 16; ++v622_i1) {
                  int32_t v625_a = v623_a + (v622_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v626_data(r1.template select<16, 1>(v625_a));
                  v626_data.copy_to(glb_m0 + (v625_a));
                }
              }
            }
          }
        }
      }
    });
  });
}

