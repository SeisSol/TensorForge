// === base name ===
kernel_4af2ba182548314d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_4af2ba182548314d = {{1, 16, 1}, 16, 12, 1, 16, 10240, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_4af2ba182548314d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_4af2ba182548314d(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_4af2ba182548314d(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_4af2ba182548314d(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_4af2ba182548314d(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_4af2ba182548314d(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m1_stride, m2, m2_extraOffset, numElements0, elementIndices, flags0);
  CHECK_ERR;
}


// === kernel ===
class kernel_kernel_4af2ba182548314d_k;
inline void kernel_kernel_4af2ba182548314d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, size_t m1_stride, const float * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for<kernel_kernel_4af2ba182548314d_k>(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2560 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 10240 B shared, occupancy grid
        // operands:
        //   m0 12×7(12×7) {0..12}×{0..7} strided
        //   m1 12×20(12×20) {0..12}×{0..20} indexed(stride=runtime)
        //   m2 20×7(20×7) {0..20}×{0..7} indexed
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2560}],"shared_bytes":10240,"shared_elements":2560,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,7]],"name":"m0","ordered":false,"parts":1,"shape":[12,7],"variant":false},{"addressing":"indexed(stride=runtime)","alias":"A","bbox":[[0,0],[12,20]],"name":"m1","ordered":false,"parts":1,"shape":[12,20],"variant":false},{"addressing":"indexed","alias":"B","bbox":[[0,0],[20,7]],"name":"m2","ordered":false,"parts":1,"shape":[20,7],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,7]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,7]},"kind":"multilinear","ops":[{"addressing":"indexed(stride=runtime)","bbox":[[0,0],[12,20]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,20]},{"addressing":"indexed","bbox":[[0,0],[20,7]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,7]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (160 * item.get_local_id(1) + 0);
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::intel_esimd::simd<float, 320> r0(0.0f);
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v386_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v386_flagSecondAhead < numElements0) ? v386_flagSecondAhead : v3_batchId1)];
          const bool allowed_peel = static_cast<bool>(flagWordFirst);
          const size_t peel_elementIndex = static_cast<size_t>(elementIndices[v3_batchId1]);
          const float *const __restrict__ peel_glb_m1 = &m1[peel_elementIndex * m1_stride + 0 + m1_extraOffset];
          if (allowed_peel) {
            // r0 = load{g>r}(peel_glb_m1);
            #pragma unroll
            for (int32_t v402_i1 = 0; v402_i1 < 20; ++v402_i1) {
              tensorforge::intel_esimd::simd<float, 16> v404_data(0.0f);
              tensorforge::intel_esimd::simd<float, 12> v404_data_part;
              v404_data_part.copy_from(peel_glb_m1 + ((v402_i1 * 12)));
              v404_data.template select<12, 1>(0) = v404_data_part;
              r0.template select<16, 1>((v402_i1 * 16)) = v404_data;
            }
          }
          const float *const __restrict__ peel_glb_m2 = &m2[peel_elementIndex * 140 + 0 + m2_extraOffset];
          if (allowed_peel) {
            // s0 = load{g>s}(peel_glb_m2[0, 1])
            tensorforge::intel_esimd::simd<float, 64> v412_ld;
            v412_ld.copy_from(peel_glb_m2 + (0 + 0 + 4 * 0 + 0));
            tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v412_ld);
            tensorforge::intel_esimd::simd<float, 64> v413_ld;
            v413_ld.copy_from(peel_glb_m2 + (0 + 0 + 4 * 0 + 64));
            tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v413_ld);
            tensorforge::intel_esimd::simd<float, 12> v414_ld;
            v414_ld.copy_from(peel_glb_m2 + (0 + 0 + 1 * 0 + 128));
            tensorforge::slmStore<float, 12>(s0 + (0 + 0 + 1 * 0 + 128), v414_ld);
          }
          uint32_t v390_flagWord(flagWordFirst);
          uint32_t v391_flagWordNext(flagWordSecond);
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const size_t wrap_elementIndex = static_cast<size_t>(elementIndices[v11_batchId1]);
            const float *const __restrict__ wrap_glb_m1 = &m1[wrap_elementIndex * m1_stride + 0 + m1_extraOffset];
            const float *const __restrict__ wrap_glb_m2 = &m2[wrap_elementIndex * 140 + 0 + m2_extraOffset];
            size_t v420_flagAheadAhead = v11_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v420_flagAheadAhead < numElements0) ? v420_flagAheadAhead : v11_batchId1)];
            const bool allowed = static_cast<bool>(v390_flagWord);
            const bool allowed_next = static_cast<bool>(v391_flagWordNext);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 84 + 0 + m0_extraOffset];
              tensorforge::intel_esimd::simd<float, 112> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 12), (0, 7)] [(0, 20)]
              tensorforge::intel_esimd::simd<float, 112> ir1(0.0f);
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
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(192));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(208));
              tensorforge::intel_esimd::simd<float, 16> v48_data(r0.template select<16, 1>(224));
              tensorforge::intel_esimd::simd<float, 16> v49_data(r0.template select<16, 1>(240));
              tensorforge::intel_esimd::simd<float, 16> v50_data(r0.template select<16, 1>(256));
              tensorforge::intel_esimd::simd<float, 16> v51_data(r0.template select<16, 1>(272));
              tensorforge::intel_esimd::simd<float, 16> v52_data(r0.template select<16, 1>(288));
              tensorforge::intel_esimd::simd<float, 16> v53_data(r0.template select<16, 1>(304));
              if (allowed_next) {
                // r0 = load{g>r}(wrap_glb_m1);
                #pragma unroll
                for (int32_t v406_i1 = 0; v406_i1 < 20; ++v406_i1) {
                  tensorforge::intel_esimd::simd<float, 16> v408_data(0.0f);
                  tensorforge::intel_esimd::simd<float, 12> v408_data_part;
                  v408_data_part.copy_from(wrap_glb_m1 + ((v406_i1 * 12)));
                  v408_data.template select<12, 1>(0) = v408_data_part;
                  r0.template select<16, 1>((v406_i1 * 16)) = v408_data;
                }
              }
              tensorforge::intel_esimd::simd<float, 16> v54_acc{};
              tensorforge::intel_esimd::simd<float, 16> v58_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v54_acc += ((static_cast<float>(v58_data[0])) * v34_data);
              v54_acc += ((static_cast<float>(v58_data[1])) * v35_data);
              v54_acc += ((static_cast<float>(v58_data[2])) * v36_data);
              v54_acc += ((static_cast<float>(v58_data[3])) * v37_data);
              v54_acc += ((static_cast<float>(v58_data[4])) * v38_data);
              v54_acc += ((static_cast<float>(v58_data[5])) * v39_data);
              v54_acc += ((static_cast<float>(v58_data[6])) * v40_data);
              v54_acc += ((static_cast<float>(v58_data[7])) * v41_data);
              v54_acc += ((static_cast<float>(v58_data[8])) * v42_data);
              v54_acc += ((static_cast<float>(v58_data[9])) * v43_data);
              v54_acc += ((static_cast<float>(v58_data[10])) * v44_data);
              v54_acc += ((static_cast<float>(v58_data[11])) * v45_data);
              v54_acc += ((static_cast<float>(v58_data[12])) * v46_data);
              v54_acc += ((static_cast<float>(v58_data[13])) * v47_data);
              v54_acc += ((static_cast<float>(v58_data[14])) * v48_data);
              v54_acc += ((static_cast<float>(v58_data[15])) * v49_data);
              tensorforge::intel_esimd::simd<float, 16> v94_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v54_acc += ((static_cast<float>(v94_data[0])) * v50_data);
              v54_acc += ((static_cast<float>(v94_data[1])) * v51_data);
              v54_acc += ((static_cast<float>(v94_data[2])) * v52_data);
              v54_acc += ((static_cast<float>(v94_data[3])) * v53_data);
              ir1.template select<16, 1>(0) = v54_acc;
              tensorforge::intel_esimd::simd<float, 16> v103_acc{};
              tensorforge::intel_esimd::simd<float, 16> v105_data = tensorforge::slmLoad<float, 16>(s0 + (20_i32));
              v103_acc += ((static_cast<float>(v105_data[0])) * v34_data);
              v103_acc += ((static_cast<float>(v105_data[1])) * v35_data);
              v103_acc += ((static_cast<float>(v105_data[2])) * v36_data);
              v103_acc += ((static_cast<float>(v105_data[3])) * v37_data);
              v103_acc += ((static_cast<float>(v105_data[4])) * v38_data);
              v103_acc += ((static_cast<float>(v105_data[5])) * v39_data);
              v103_acc += ((static_cast<float>(v105_data[6])) * v40_data);
              v103_acc += ((static_cast<float>(v105_data[7])) * v41_data);
              v103_acc += ((static_cast<float>(v105_data[8])) * v42_data);
              v103_acc += ((static_cast<float>(v105_data[9])) * v43_data);
              v103_acc += ((static_cast<float>(v105_data[10])) * v44_data);
              v103_acc += ((static_cast<float>(v105_data[11])) * v45_data);
              v103_acc += ((static_cast<float>(v105_data[12])) * v46_data);
              v103_acc += ((static_cast<float>(v105_data[13])) * v47_data);
              v103_acc += ((static_cast<float>(v105_data[14])) * v48_data);
              v103_acc += ((static_cast<float>(v105_data[15])) * v49_data);
              tensorforge::intel_esimd::simd<float, 16> v139_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              v103_acc += ((static_cast<float>(v139_data[0])) * v50_data);
              v103_acc += ((static_cast<float>(v139_data[1])) * v51_data);
              v103_acc += ((static_cast<float>(v139_data[2])) * v52_data);
              v103_acc += ((static_cast<float>(v139_data[3])) * v53_data);
              ir1.template select<16, 1>(16) = v103_acc;
              tensorforge::intel_esimd::simd<float, 16> v148_acc{};
              tensorforge::intel_esimd::simd<float, 16> v150_data = tensorforge::slmLoad<float, 16>(s0 + (40_i32));
              v148_acc += ((static_cast<float>(v150_data[0])) * v34_data);
              v148_acc += ((static_cast<float>(v150_data[1])) * v35_data);
              v148_acc += ((static_cast<float>(v150_data[2])) * v36_data);
              v148_acc += ((static_cast<float>(v150_data[3])) * v37_data);
              v148_acc += ((static_cast<float>(v150_data[4])) * v38_data);
              v148_acc += ((static_cast<float>(v150_data[5])) * v39_data);
              v148_acc += ((static_cast<float>(v150_data[6])) * v40_data);
              v148_acc += ((static_cast<float>(v150_data[7])) * v41_data);
              v148_acc += ((static_cast<float>(v150_data[8])) * v42_data);
              v148_acc += ((static_cast<float>(v150_data[9])) * v43_data);
              v148_acc += ((static_cast<float>(v150_data[10])) * v44_data);
              v148_acc += ((static_cast<float>(v150_data[11])) * v45_data);
              v148_acc += ((static_cast<float>(v150_data[12])) * v46_data);
              v148_acc += ((static_cast<float>(v150_data[13])) * v47_data);
              v148_acc += ((static_cast<float>(v150_data[14])) * v48_data);
              v148_acc += ((static_cast<float>(v150_data[15])) * v49_data);
              tensorforge::intel_esimd::simd<float, 16> v184_data = tensorforge::slmLoad<float, 16>(s0 + (56_i32));
              v148_acc += ((static_cast<float>(v184_data[0])) * v50_data);
              v148_acc += ((static_cast<float>(v184_data[1])) * v51_data);
              v148_acc += ((static_cast<float>(v184_data[2])) * v52_data);
              v148_acc += ((static_cast<float>(v184_data[3])) * v53_data);
              ir1.template select<16, 1>(32) = v148_acc;
              tensorforge::intel_esimd::simd<float, 16> v193_acc{};
              tensorforge::intel_esimd::simd<float, 16> v195_data = tensorforge::slmLoad<float, 16>(s0 + (60_i32));
              v193_acc += ((static_cast<float>(v195_data[0])) * v34_data);
              v193_acc += ((static_cast<float>(v195_data[1])) * v35_data);
              v193_acc += ((static_cast<float>(v195_data[2])) * v36_data);
              v193_acc += ((static_cast<float>(v195_data[3])) * v37_data);
              v193_acc += ((static_cast<float>(v195_data[4])) * v38_data);
              v193_acc += ((static_cast<float>(v195_data[5])) * v39_data);
              v193_acc += ((static_cast<float>(v195_data[6])) * v40_data);
              v193_acc += ((static_cast<float>(v195_data[7])) * v41_data);
              v193_acc += ((static_cast<float>(v195_data[8])) * v42_data);
              v193_acc += ((static_cast<float>(v195_data[9])) * v43_data);
              v193_acc += ((static_cast<float>(v195_data[10])) * v44_data);
              v193_acc += ((static_cast<float>(v195_data[11])) * v45_data);
              v193_acc += ((static_cast<float>(v195_data[12])) * v46_data);
              v193_acc += ((static_cast<float>(v195_data[13])) * v47_data);
              v193_acc += ((static_cast<float>(v195_data[14])) * v48_data);
              v193_acc += ((static_cast<float>(v195_data[15])) * v49_data);
              tensorforge::intel_esimd::simd<float, 16> v229_data = tensorforge::slmLoad<float, 16>(s0 + (76_i32));
              v193_acc += ((static_cast<float>(v229_data[0])) * v50_data);
              v193_acc += ((static_cast<float>(v229_data[1])) * v51_data);
              v193_acc += ((static_cast<float>(v229_data[2])) * v52_data);
              v193_acc += ((static_cast<float>(v229_data[3])) * v53_data);
              ir1.template select<16, 1>(48) = v193_acc;
              tensorforge::intel_esimd::simd<float, 16> v238_acc{};
              tensorforge::intel_esimd::simd<float, 16> v240_data = tensorforge::slmLoad<float, 16>(s0 + (80_i32));
              v238_acc += ((static_cast<float>(v240_data[0])) * v34_data);
              v238_acc += ((static_cast<float>(v240_data[1])) * v35_data);
              v238_acc += ((static_cast<float>(v240_data[2])) * v36_data);
              v238_acc += ((static_cast<float>(v240_data[3])) * v37_data);
              v238_acc += ((static_cast<float>(v240_data[4])) * v38_data);
              v238_acc += ((static_cast<float>(v240_data[5])) * v39_data);
              v238_acc += ((static_cast<float>(v240_data[6])) * v40_data);
              v238_acc += ((static_cast<float>(v240_data[7])) * v41_data);
              v238_acc += ((static_cast<float>(v240_data[8])) * v42_data);
              v238_acc += ((static_cast<float>(v240_data[9])) * v43_data);
              v238_acc += ((static_cast<float>(v240_data[10])) * v44_data);
              v238_acc += ((static_cast<float>(v240_data[11])) * v45_data);
              v238_acc += ((static_cast<float>(v240_data[12])) * v46_data);
              v238_acc += ((static_cast<float>(v240_data[13])) * v47_data);
              v238_acc += ((static_cast<float>(v240_data[14])) * v48_data);
              v238_acc += ((static_cast<float>(v240_data[15])) * v49_data);
              tensorforge::intel_esimd::simd<float, 16> v274_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              v238_acc += ((static_cast<float>(v274_data[0])) * v50_data);
              v238_acc += ((static_cast<float>(v274_data[1])) * v51_data);
              v238_acc += ((static_cast<float>(v274_data[2])) * v52_data);
              v238_acc += ((static_cast<float>(v274_data[3])) * v53_data);
              ir1.template select<16, 1>(64) = v238_acc;
              tensorforge::intel_esimd::simd<float, 16> v283_acc{};
              tensorforge::intel_esimd::simd<float, 16> v285_data = tensorforge::slmLoad<float, 16>(s0 + (100_i32));
              v283_acc += ((static_cast<float>(v285_data[0])) * v34_data);
              v283_acc += ((static_cast<float>(v285_data[1])) * v35_data);
              v283_acc += ((static_cast<float>(v285_data[2])) * v36_data);
              v283_acc += ((static_cast<float>(v285_data[3])) * v37_data);
              v283_acc += ((static_cast<float>(v285_data[4])) * v38_data);
              v283_acc += ((static_cast<float>(v285_data[5])) * v39_data);
              v283_acc += ((static_cast<float>(v285_data[6])) * v40_data);
              v283_acc += ((static_cast<float>(v285_data[7])) * v41_data);
              v283_acc += ((static_cast<float>(v285_data[8])) * v42_data);
              v283_acc += ((static_cast<float>(v285_data[9])) * v43_data);
              v283_acc += ((static_cast<float>(v285_data[10])) * v44_data);
              v283_acc += ((static_cast<float>(v285_data[11])) * v45_data);
              v283_acc += ((static_cast<float>(v285_data[12])) * v46_data);
              v283_acc += ((static_cast<float>(v285_data[13])) * v47_data);
              v283_acc += ((static_cast<float>(v285_data[14])) * v48_data);
              v283_acc += ((static_cast<float>(v285_data[15])) * v49_data);
              tensorforge::intel_esimd::simd<float, 16> v319_data = tensorforge::slmLoad<float, 16>(s0 + (116_i32));
              v283_acc += ((static_cast<float>(v319_data[0])) * v50_data);
              v283_acc += ((static_cast<float>(v319_data[1])) * v51_data);
              v283_acc += ((static_cast<float>(v319_data[2])) * v52_data);
              v283_acc += ((static_cast<float>(v319_data[3])) * v53_data);
              ir1.template select<16, 1>(80) = v283_acc;
              tensorforge::intel_esimd::simd<float, 16> v328_acc{};
              tensorforge::intel_esimd::simd<float, 16> v330_data = tensorforge::slmLoad<float, 16>(s0 + (120_i32));
              v328_acc += ((static_cast<float>(v330_data[0])) * v34_data);
              v328_acc += ((static_cast<float>(v330_data[1])) * v35_data);
              v328_acc += ((static_cast<float>(v330_data[2])) * v36_data);
              v328_acc += ((static_cast<float>(v330_data[3])) * v37_data);
              v328_acc += ((static_cast<float>(v330_data[4])) * v38_data);
              v328_acc += ((static_cast<float>(v330_data[5])) * v39_data);
              v328_acc += ((static_cast<float>(v330_data[6])) * v40_data);
              v328_acc += ((static_cast<float>(v330_data[7])) * v41_data);
              v328_acc += ((static_cast<float>(v330_data[8])) * v42_data);
              v328_acc += ((static_cast<float>(v330_data[9])) * v43_data);
              v328_acc += ((static_cast<float>(v330_data[10])) * v44_data);
              v328_acc += ((static_cast<float>(v330_data[11])) * v45_data);
              v328_acc += ((static_cast<float>(v330_data[12])) * v46_data);
              v328_acc += ((static_cast<float>(v330_data[13])) * v47_data);
              v328_acc += ((static_cast<float>(v330_data[14])) * v48_data);
              v328_acc += ((static_cast<float>(v330_data[15])) * v49_data);
              tensorforge::intel_esimd::simd<float, 16> v364_data = tensorforge::slmLoad<float, 16>(s0 + (136_i32));
              v328_acc += ((static_cast<float>(v364_data[0])) * v50_data);
              v328_acc += ((static_cast<float>(v364_data[1])) * v51_data);
              v328_acc += ((static_cast<float>(v364_data[2])) * v52_data);
              v328_acc += ((static_cast<float>(v364_data[3])) * v53_data);
              ir1.template select<16, 1>(96) = v328_acc;
              // r1 = ir1
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v377_i1 = 0; v377_i1 < 7; ++v377_i1) {
                tensorforge::intel_esimd::simd<float, 12> v380_data(ir1.template select<12, 1>((v377_i1 * 16)));
                v380_data.copy_to(glb_m0 + ((v377_i1 * 12)));
              }
            }
            else {
              if (allowed_next) {
                // r0 = load{g>r}(wrap_glb_m1);
                #pragma unroll
                for (int32_t v398_i1 = 0; v398_i1 < 20; ++v398_i1) {
                  tensorforge::intel_esimd::simd<float, 16> v400_data(0.0f);
                  tensorforge::intel_esimd::simd<float, 12> v400_data_part;
                  v400_data_part.copy_from(wrap_glb_m1 + ((v398_i1 * 12)));
                  v400_data.template select<12, 1>(0) = v400_data_part;
                  r0.template select<16, 1>((v398_i1 * 16)) = v400_data;
                }
              }
            }
            if (allowed_next) {
              // s0 = load{g>s}(wrap_glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v415_ld;
              v415_ld.copy_from(wrap_glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v415_ld);
              tensorforge::intel_esimd::simd<float, 64> v416_ld;
              v416_ld.copy_from(wrap_glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v416_ld);
              tensorforge::intel_esimd::simd<float, 12> v417_ld;
              v417_ld.copy_from(wrap_glb_m2 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 12>(s0 + (0 + 0 + 1 * 0 + 128), v417_ld);
            }
            v390_flagWord = v391_flagWordNext;
            v391_flagWordNext = flagWordAhead;
          }
        }
      }
    });
  });
}

