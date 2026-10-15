// === base name ===
kernel_8c88dc41868734e1

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_8c88dc41868734e1 = {{1, 16, 1}, 16, 16, 1, 16, 17408, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_8c88dc41868734e1(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_8c88dc41868734e1(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_8c88dc41868734e1(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_8c88dc41868734e1(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_8c88dc41868734e1(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_8c88dc41868734e1(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_8c88dc41868734e1(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<4352 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 17408 B shared, occupancy grid
        // operands:
        //   m0 16×16(16×16) {0..16}×{0..16} strided
        //   m1 16×16(16×16) {0..16}×{0..16} none ordered
        //   m2 16×16(16×16) {0..16}×{0..16} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":4352}],"shared_bytes":17408,"shared_elements":4352,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":true,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (272 * item.get_local_id(1) + 0);
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const float *const __restrict__ glb_m1 = &m1[0];
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v663_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v663_flagSecondAhead < numElements0) ? v663_flagSecondAhead : v3_batchId1)];
          const float *const __restrict__ peel_glb_m2 = &m2[v3_batchId1 * 256 + 0 + m2_extraOffset];
          // s0 = load{g>s}(peel_glb_m2[0, 1])
          tensorforge::intel_esimd::simd<float, 64> v671_ld;
          v671_ld.copy_from(peel_glb_m2 + (0 + 0 + 4 * 0 + 0));
          tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v671_ld);
          tensorforge::intel_esimd::simd<float, 64> v672_ld;
          v672_ld.copy_from(peel_glb_m2 + (0 + 0 + 4 * 0 + 64));
          tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v672_ld);
          tensorforge::intel_esimd::simd<float, 64> v673_ld;
          v673_ld.copy_from(peel_glb_m2 + (0 + 0 + 4 * 0 + 128));
          tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 128), v673_ld);
          tensorforge::intel_esimd::simd<float, 64> v674_ld;
          v674_ld.copy_from(peel_glb_m2 + (0 + 0 + 4 * 0 + 192));
          tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 192), v674_ld);
          uint32_t v667_flagWord(flagWordFirst);
          uint32_t v668_flagWordNext(flagWordSecond);
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const float *const __restrict__ wrap_glb_m2 = &m2[v12_batchId1 * 256 + 0 + m2_extraOffset];
            size_t v681_flagAheadAhead = v12_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v681_flagAheadAhead < numElements0) ? v681_flagAheadAhead : v12_batchId1)];
            const bool allowed = static_cast<bool>(v667_flagWord);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 256 + 0 + m0_extraOffset];
              tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
              // ir0 = +(glb_m1 * s0)
              // [(0, 16), (0, 16)] [(0, 16)]
              tensorforge::intel_esimd::simd<float, 256> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run0;
              glb_m1_run0.copy_from(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v28_data(glb_m1_run0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v32_data(glb_m1_run0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v36_data(glb_m1_run0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v40_data(glb_m1_run0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run1;
              glb_m1_run1.copy_from(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v44_data(glb_m1_run1.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v48_data(glb_m1_run1.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v52_data(glb_m1_run1.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v56_data(glb_m1_run1.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run2;
              glb_m1_run2.copy_from(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v60_data(glb_m1_run2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v64_data(glb_m1_run2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v68_data(glb_m1_run2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v72_data(glb_m1_run2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run3;
              glb_m1_run3.copy_from(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v76_data(glb_m1_run3.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v80_data(glb_m1_run3.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v84_data(glb_m1_run3.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v88_data(glb_m1_run3.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v89_acc{};
              tensorforge::intel_esimd::simd<float, 16> v90_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v89_acc += ((static_cast<float>(v90_data[0])) * v28_data);
              v89_acc += ((static_cast<float>(v90_data[1])) * v32_data);
              v89_acc += ((static_cast<float>(v90_data[2])) * v36_data);
              v89_acc += ((static_cast<float>(v90_data[3])) * v40_data);
              v89_acc += ((static_cast<float>(v90_data[4])) * v44_data);
              v89_acc += ((static_cast<float>(v90_data[5])) * v48_data);
              v89_acc += ((static_cast<float>(v90_data[6])) * v52_data);
              v89_acc += ((static_cast<float>(v90_data[7])) * v56_data);
              v89_acc += ((static_cast<float>(v90_data[8])) * v60_data);
              v89_acc += ((static_cast<float>(v90_data[9])) * v64_data);
              v89_acc += ((static_cast<float>(v90_data[10])) * v68_data);
              v89_acc += ((static_cast<float>(v90_data[11])) * v72_data);
              v89_acc += ((static_cast<float>(v90_data[12])) * v76_data);
              v89_acc += ((static_cast<float>(v90_data[13])) * v80_data);
              v89_acc += ((static_cast<float>(v90_data[14])) * v84_data);
              v89_acc += ((static_cast<float>(v90_data[15])) * v88_data);
              ir0.template select<16, 1>(0) = v89_acc;
              tensorforge::intel_esimd::simd<float, 16> v123_acc{};
              tensorforge::intel_esimd::simd<float, 16> v125_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v123_acc += ((static_cast<float>(v125_data[0])) * v28_data);
              v123_acc += ((static_cast<float>(v125_data[1])) * v32_data);
              v123_acc += ((static_cast<float>(v125_data[2])) * v36_data);
              v123_acc += ((static_cast<float>(v125_data[3])) * v40_data);
              v123_acc += ((static_cast<float>(v125_data[4])) * v44_data);
              v123_acc += ((static_cast<float>(v125_data[5])) * v48_data);
              v123_acc += ((static_cast<float>(v125_data[6])) * v52_data);
              v123_acc += ((static_cast<float>(v125_data[7])) * v56_data);
              v123_acc += ((static_cast<float>(v125_data[8])) * v60_data);
              v123_acc += ((static_cast<float>(v125_data[9])) * v64_data);
              v123_acc += ((static_cast<float>(v125_data[10])) * v68_data);
              v123_acc += ((static_cast<float>(v125_data[11])) * v72_data);
              v123_acc += ((static_cast<float>(v125_data[12])) * v76_data);
              v123_acc += ((static_cast<float>(v125_data[13])) * v80_data);
              v123_acc += ((static_cast<float>(v125_data[14])) * v84_data);
              v123_acc += ((static_cast<float>(v125_data[15])) * v88_data);
              ir0.template select<16, 1>(16) = v123_acc;
              tensorforge::intel_esimd::simd<float, 16> v158_acc{};
              tensorforge::intel_esimd::simd<float, 16> v160_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v158_acc += ((static_cast<float>(v160_data[0])) * v28_data);
              v158_acc += ((static_cast<float>(v160_data[1])) * v32_data);
              v158_acc += ((static_cast<float>(v160_data[2])) * v36_data);
              v158_acc += ((static_cast<float>(v160_data[3])) * v40_data);
              v158_acc += ((static_cast<float>(v160_data[4])) * v44_data);
              v158_acc += ((static_cast<float>(v160_data[5])) * v48_data);
              v158_acc += ((static_cast<float>(v160_data[6])) * v52_data);
              v158_acc += ((static_cast<float>(v160_data[7])) * v56_data);
              v158_acc += ((static_cast<float>(v160_data[8])) * v60_data);
              v158_acc += ((static_cast<float>(v160_data[9])) * v64_data);
              v158_acc += ((static_cast<float>(v160_data[10])) * v68_data);
              v158_acc += ((static_cast<float>(v160_data[11])) * v72_data);
              v158_acc += ((static_cast<float>(v160_data[12])) * v76_data);
              v158_acc += ((static_cast<float>(v160_data[13])) * v80_data);
              v158_acc += ((static_cast<float>(v160_data[14])) * v84_data);
              v158_acc += ((static_cast<float>(v160_data[15])) * v88_data);
              ir0.template select<16, 1>(32) = v158_acc;
              tensorforge::intel_esimd::simd<float, 16> v193_acc{};
              tensorforge::intel_esimd::simd<float, 16> v195_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v193_acc += ((static_cast<float>(v195_data[0])) * v28_data);
              v193_acc += ((static_cast<float>(v195_data[1])) * v32_data);
              v193_acc += ((static_cast<float>(v195_data[2])) * v36_data);
              v193_acc += ((static_cast<float>(v195_data[3])) * v40_data);
              v193_acc += ((static_cast<float>(v195_data[4])) * v44_data);
              v193_acc += ((static_cast<float>(v195_data[5])) * v48_data);
              v193_acc += ((static_cast<float>(v195_data[6])) * v52_data);
              v193_acc += ((static_cast<float>(v195_data[7])) * v56_data);
              v193_acc += ((static_cast<float>(v195_data[8])) * v60_data);
              v193_acc += ((static_cast<float>(v195_data[9])) * v64_data);
              v193_acc += ((static_cast<float>(v195_data[10])) * v68_data);
              v193_acc += ((static_cast<float>(v195_data[11])) * v72_data);
              v193_acc += ((static_cast<float>(v195_data[12])) * v76_data);
              v193_acc += ((static_cast<float>(v195_data[13])) * v80_data);
              v193_acc += ((static_cast<float>(v195_data[14])) * v84_data);
              v193_acc += ((static_cast<float>(v195_data[15])) * v88_data);
              ir0.template select<16, 1>(48) = v193_acc;
              tensorforge::intel_esimd::simd<float, 16> v228_acc{};
              tensorforge::intel_esimd::simd<float, 16> v230_data = tensorforge::slmLoad<float, 16>(s0 + (64_i32));
              v228_acc += ((static_cast<float>(v230_data[0])) * v28_data);
              v228_acc += ((static_cast<float>(v230_data[1])) * v32_data);
              v228_acc += ((static_cast<float>(v230_data[2])) * v36_data);
              v228_acc += ((static_cast<float>(v230_data[3])) * v40_data);
              v228_acc += ((static_cast<float>(v230_data[4])) * v44_data);
              v228_acc += ((static_cast<float>(v230_data[5])) * v48_data);
              v228_acc += ((static_cast<float>(v230_data[6])) * v52_data);
              v228_acc += ((static_cast<float>(v230_data[7])) * v56_data);
              v228_acc += ((static_cast<float>(v230_data[8])) * v60_data);
              v228_acc += ((static_cast<float>(v230_data[9])) * v64_data);
              v228_acc += ((static_cast<float>(v230_data[10])) * v68_data);
              v228_acc += ((static_cast<float>(v230_data[11])) * v72_data);
              v228_acc += ((static_cast<float>(v230_data[12])) * v76_data);
              v228_acc += ((static_cast<float>(v230_data[13])) * v80_data);
              v228_acc += ((static_cast<float>(v230_data[14])) * v84_data);
              v228_acc += ((static_cast<float>(v230_data[15])) * v88_data);
              ir0.template select<16, 1>(64) = v228_acc;
              tensorforge::intel_esimd::simd<float, 16> v263_acc{};
              tensorforge::intel_esimd::simd<float, 16> v265_data = tensorforge::slmLoad<float, 16>(s0 + (80_i32));
              v263_acc += ((static_cast<float>(v265_data[0])) * v28_data);
              v263_acc += ((static_cast<float>(v265_data[1])) * v32_data);
              v263_acc += ((static_cast<float>(v265_data[2])) * v36_data);
              v263_acc += ((static_cast<float>(v265_data[3])) * v40_data);
              v263_acc += ((static_cast<float>(v265_data[4])) * v44_data);
              v263_acc += ((static_cast<float>(v265_data[5])) * v48_data);
              v263_acc += ((static_cast<float>(v265_data[6])) * v52_data);
              v263_acc += ((static_cast<float>(v265_data[7])) * v56_data);
              v263_acc += ((static_cast<float>(v265_data[8])) * v60_data);
              v263_acc += ((static_cast<float>(v265_data[9])) * v64_data);
              v263_acc += ((static_cast<float>(v265_data[10])) * v68_data);
              v263_acc += ((static_cast<float>(v265_data[11])) * v72_data);
              v263_acc += ((static_cast<float>(v265_data[12])) * v76_data);
              v263_acc += ((static_cast<float>(v265_data[13])) * v80_data);
              v263_acc += ((static_cast<float>(v265_data[14])) * v84_data);
              v263_acc += ((static_cast<float>(v265_data[15])) * v88_data);
              ir0.template select<16, 1>(80) = v263_acc;
              tensorforge::intel_esimd::simd<float, 16> v298_acc{};
              tensorforge::intel_esimd::simd<float, 16> v300_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              v298_acc += ((static_cast<float>(v300_data[0])) * v28_data);
              v298_acc += ((static_cast<float>(v300_data[1])) * v32_data);
              v298_acc += ((static_cast<float>(v300_data[2])) * v36_data);
              v298_acc += ((static_cast<float>(v300_data[3])) * v40_data);
              v298_acc += ((static_cast<float>(v300_data[4])) * v44_data);
              v298_acc += ((static_cast<float>(v300_data[5])) * v48_data);
              v298_acc += ((static_cast<float>(v300_data[6])) * v52_data);
              v298_acc += ((static_cast<float>(v300_data[7])) * v56_data);
              v298_acc += ((static_cast<float>(v300_data[8])) * v60_data);
              v298_acc += ((static_cast<float>(v300_data[9])) * v64_data);
              v298_acc += ((static_cast<float>(v300_data[10])) * v68_data);
              v298_acc += ((static_cast<float>(v300_data[11])) * v72_data);
              v298_acc += ((static_cast<float>(v300_data[12])) * v76_data);
              v298_acc += ((static_cast<float>(v300_data[13])) * v80_data);
              v298_acc += ((static_cast<float>(v300_data[14])) * v84_data);
              v298_acc += ((static_cast<float>(v300_data[15])) * v88_data);
              ir0.template select<16, 1>(96) = v298_acc;
              tensorforge::intel_esimd::simd<float, 16> v333_acc{};
              tensorforge::intel_esimd::simd<float, 16> v335_data = tensorforge::slmLoad<float, 16>(s0 + (112_i32));
              v333_acc += ((static_cast<float>(v335_data[0])) * v28_data);
              v333_acc += ((static_cast<float>(v335_data[1])) * v32_data);
              v333_acc += ((static_cast<float>(v335_data[2])) * v36_data);
              v333_acc += ((static_cast<float>(v335_data[3])) * v40_data);
              v333_acc += ((static_cast<float>(v335_data[4])) * v44_data);
              v333_acc += ((static_cast<float>(v335_data[5])) * v48_data);
              v333_acc += ((static_cast<float>(v335_data[6])) * v52_data);
              v333_acc += ((static_cast<float>(v335_data[7])) * v56_data);
              v333_acc += ((static_cast<float>(v335_data[8])) * v60_data);
              v333_acc += ((static_cast<float>(v335_data[9])) * v64_data);
              v333_acc += ((static_cast<float>(v335_data[10])) * v68_data);
              v333_acc += ((static_cast<float>(v335_data[11])) * v72_data);
              v333_acc += ((static_cast<float>(v335_data[12])) * v76_data);
              v333_acc += ((static_cast<float>(v335_data[13])) * v80_data);
              v333_acc += ((static_cast<float>(v335_data[14])) * v84_data);
              v333_acc += ((static_cast<float>(v335_data[15])) * v88_data);
              ir0.template select<16, 1>(112) = v333_acc;
              tensorforge::intel_esimd::simd<float, 16> v368_acc{};
              tensorforge::intel_esimd::simd<float, 16> v370_data = tensorforge::slmLoad<float, 16>(s0 + (128_i32));
              v368_acc += ((static_cast<float>(v370_data[0])) * v28_data);
              v368_acc += ((static_cast<float>(v370_data[1])) * v32_data);
              v368_acc += ((static_cast<float>(v370_data[2])) * v36_data);
              v368_acc += ((static_cast<float>(v370_data[3])) * v40_data);
              v368_acc += ((static_cast<float>(v370_data[4])) * v44_data);
              v368_acc += ((static_cast<float>(v370_data[5])) * v48_data);
              v368_acc += ((static_cast<float>(v370_data[6])) * v52_data);
              v368_acc += ((static_cast<float>(v370_data[7])) * v56_data);
              v368_acc += ((static_cast<float>(v370_data[8])) * v60_data);
              v368_acc += ((static_cast<float>(v370_data[9])) * v64_data);
              v368_acc += ((static_cast<float>(v370_data[10])) * v68_data);
              v368_acc += ((static_cast<float>(v370_data[11])) * v72_data);
              v368_acc += ((static_cast<float>(v370_data[12])) * v76_data);
              v368_acc += ((static_cast<float>(v370_data[13])) * v80_data);
              v368_acc += ((static_cast<float>(v370_data[14])) * v84_data);
              v368_acc += ((static_cast<float>(v370_data[15])) * v88_data);
              ir0.template select<16, 1>(128) = v368_acc;
              tensorforge::intel_esimd::simd<float, 16> v403_acc{};
              tensorforge::intel_esimd::simd<float, 16> v405_data = tensorforge::slmLoad<float, 16>(s0 + (144_i32));
              v403_acc += ((static_cast<float>(v405_data[0])) * v28_data);
              v403_acc += ((static_cast<float>(v405_data[1])) * v32_data);
              v403_acc += ((static_cast<float>(v405_data[2])) * v36_data);
              v403_acc += ((static_cast<float>(v405_data[3])) * v40_data);
              v403_acc += ((static_cast<float>(v405_data[4])) * v44_data);
              v403_acc += ((static_cast<float>(v405_data[5])) * v48_data);
              v403_acc += ((static_cast<float>(v405_data[6])) * v52_data);
              v403_acc += ((static_cast<float>(v405_data[7])) * v56_data);
              v403_acc += ((static_cast<float>(v405_data[8])) * v60_data);
              v403_acc += ((static_cast<float>(v405_data[9])) * v64_data);
              v403_acc += ((static_cast<float>(v405_data[10])) * v68_data);
              v403_acc += ((static_cast<float>(v405_data[11])) * v72_data);
              v403_acc += ((static_cast<float>(v405_data[12])) * v76_data);
              v403_acc += ((static_cast<float>(v405_data[13])) * v80_data);
              v403_acc += ((static_cast<float>(v405_data[14])) * v84_data);
              v403_acc += ((static_cast<float>(v405_data[15])) * v88_data);
              ir0.template select<16, 1>(144) = v403_acc;
              tensorforge::intel_esimd::simd<float, 16> v438_acc{};
              tensorforge::intel_esimd::simd<float, 16> v440_data = tensorforge::slmLoad<float, 16>(s0 + (160_i32));
              v438_acc += ((static_cast<float>(v440_data[0])) * v28_data);
              v438_acc += ((static_cast<float>(v440_data[1])) * v32_data);
              v438_acc += ((static_cast<float>(v440_data[2])) * v36_data);
              v438_acc += ((static_cast<float>(v440_data[3])) * v40_data);
              v438_acc += ((static_cast<float>(v440_data[4])) * v44_data);
              v438_acc += ((static_cast<float>(v440_data[5])) * v48_data);
              v438_acc += ((static_cast<float>(v440_data[6])) * v52_data);
              v438_acc += ((static_cast<float>(v440_data[7])) * v56_data);
              v438_acc += ((static_cast<float>(v440_data[8])) * v60_data);
              v438_acc += ((static_cast<float>(v440_data[9])) * v64_data);
              v438_acc += ((static_cast<float>(v440_data[10])) * v68_data);
              v438_acc += ((static_cast<float>(v440_data[11])) * v72_data);
              v438_acc += ((static_cast<float>(v440_data[12])) * v76_data);
              v438_acc += ((static_cast<float>(v440_data[13])) * v80_data);
              v438_acc += ((static_cast<float>(v440_data[14])) * v84_data);
              v438_acc += ((static_cast<float>(v440_data[15])) * v88_data);
              ir0.template select<16, 1>(160) = v438_acc;
              tensorforge::intel_esimd::simd<float, 16> v473_acc{};
              tensorforge::intel_esimd::simd<float, 16> v475_data = tensorforge::slmLoad<float, 16>(s0 + (176_i32));
              v473_acc += ((static_cast<float>(v475_data[0])) * v28_data);
              v473_acc += ((static_cast<float>(v475_data[1])) * v32_data);
              v473_acc += ((static_cast<float>(v475_data[2])) * v36_data);
              v473_acc += ((static_cast<float>(v475_data[3])) * v40_data);
              v473_acc += ((static_cast<float>(v475_data[4])) * v44_data);
              v473_acc += ((static_cast<float>(v475_data[5])) * v48_data);
              v473_acc += ((static_cast<float>(v475_data[6])) * v52_data);
              v473_acc += ((static_cast<float>(v475_data[7])) * v56_data);
              v473_acc += ((static_cast<float>(v475_data[8])) * v60_data);
              v473_acc += ((static_cast<float>(v475_data[9])) * v64_data);
              v473_acc += ((static_cast<float>(v475_data[10])) * v68_data);
              v473_acc += ((static_cast<float>(v475_data[11])) * v72_data);
              v473_acc += ((static_cast<float>(v475_data[12])) * v76_data);
              v473_acc += ((static_cast<float>(v475_data[13])) * v80_data);
              v473_acc += ((static_cast<float>(v475_data[14])) * v84_data);
              v473_acc += ((static_cast<float>(v475_data[15])) * v88_data);
              ir0.template select<16, 1>(176) = v473_acc;
              tensorforge::intel_esimd::simd<float, 16> v508_acc{};
              tensorforge::intel_esimd::simd<float, 16> v510_data = tensorforge::slmLoad<float, 16>(s0 + (192_i32));
              v508_acc += ((static_cast<float>(v510_data[0])) * v28_data);
              v508_acc += ((static_cast<float>(v510_data[1])) * v32_data);
              v508_acc += ((static_cast<float>(v510_data[2])) * v36_data);
              v508_acc += ((static_cast<float>(v510_data[3])) * v40_data);
              v508_acc += ((static_cast<float>(v510_data[4])) * v44_data);
              v508_acc += ((static_cast<float>(v510_data[5])) * v48_data);
              v508_acc += ((static_cast<float>(v510_data[6])) * v52_data);
              v508_acc += ((static_cast<float>(v510_data[7])) * v56_data);
              v508_acc += ((static_cast<float>(v510_data[8])) * v60_data);
              v508_acc += ((static_cast<float>(v510_data[9])) * v64_data);
              v508_acc += ((static_cast<float>(v510_data[10])) * v68_data);
              v508_acc += ((static_cast<float>(v510_data[11])) * v72_data);
              v508_acc += ((static_cast<float>(v510_data[12])) * v76_data);
              v508_acc += ((static_cast<float>(v510_data[13])) * v80_data);
              v508_acc += ((static_cast<float>(v510_data[14])) * v84_data);
              v508_acc += ((static_cast<float>(v510_data[15])) * v88_data);
              ir0.template select<16, 1>(192) = v508_acc;
              tensorforge::intel_esimd::simd<float, 16> v543_acc{};
              tensorforge::intel_esimd::simd<float, 16> v545_data = tensorforge::slmLoad<float, 16>(s0 + (208_i32));
              v543_acc += ((static_cast<float>(v545_data[0])) * v28_data);
              v543_acc += ((static_cast<float>(v545_data[1])) * v32_data);
              v543_acc += ((static_cast<float>(v545_data[2])) * v36_data);
              v543_acc += ((static_cast<float>(v545_data[3])) * v40_data);
              v543_acc += ((static_cast<float>(v545_data[4])) * v44_data);
              v543_acc += ((static_cast<float>(v545_data[5])) * v48_data);
              v543_acc += ((static_cast<float>(v545_data[6])) * v52_data);
              v543_acc += ((static_cast<float>(v545_data[7])) * v56_data);
              v543_acc += ((static_cast<float>(v545_data[8])) * v60_data);
              v543_acc += ((static_cast<float>(v545_data[9])) * v64_data);
              v543_acc += ((static_cast<float>(v545_data[10])) * v68_data);
              v543_acc += ((static_cast<float>(v545_data[11])) * v72_data);
              v543_acc += ((static_cast<float>(v545_data[12])) * v76_data);
              v543_acc += ((static_cast<float>(v545_data[13])) * v80_data);
              v543_acc += ((static_cast<float>(v545_data[14])) * v84_data);
              v543_acc += ((static_cast<float>(v545_data[15])) * v88_data);
              ir0.template select<16, 1>(208) = v543_acc;
              tensorforge::intel_esimd::simd<float, 16> v578_acc{};
              tensorforge::intel_esimd::simd<float, 16> v580_data = tensorforge::slmLoad<float, 16>(s0 + (224_i32));
              v578_acc += ((static_cast<float>(v580_data[0])) * v28_data);
              v578_acc += ((static_cast<float>(v580_data[1])) * v32_data);
              v578_acc += ((static_cast<float>(v580_data[2])) * v36_data);
              v578_acc += ((static_cast<float>(v580_data[3])) * v40_data);
              v578_acc += ((static_cast<float>(v580_data[4])) * v44_data);
              v578_acc += ((static_cast<float>(v580_data[5])) * v48_data);
              v578_acc += ((static_cast<float>(v580_data[6])) * v52_data);
              v578_acc += ((static_cast<float>(v580_data[7])) * v56_data);
              v578_acc += ((static_cast<float>(v580_data[8])) * v60_data);
              v578_acc += ((static_cast<float>(v580_data[9])) * v64_data);
              v578_acc += ((static_cast<float>(v580_data[10])) * v68_data);
              v578_acc += ((static_cast<float>(v580_data[11])) * v72_data);
              v578_acc += ((static_cast<float>(v580_data[12])) * v76_data);
              v578_acc += ((static_cast<float>(v580_data[13])) * v80_data);
              v578_acc += ((static_cast<float>(v580_data[14])) * v84_data);
              v578_acc += ((static_cast<float>(v580_data[15])) * v88_data);
              ir0.template select<16, 1>(224) = v578_acc;
              tensorforge::intel_esimd::simd<float, 16> v613_acc{};
              tensorforge::intel_esimd::simd<float, 16> v615_data = tensorforge::slmLoad<float, 16>(s0 + (240_i32));
              v613_acc += ((static_cast<float>(v615_data[0])) * v28_data);
              v613_acc += ((static_cast<float>(v615_data[1])) * v32_data);
              v613_acc += ((static_cast<float>(v615_data[2])) * v36_data);
              v613_acc += ((static_cast<float>(v615_data[3])) * v40_data);
              v613_acc += ((static_cast<float>(v615_data[4])) * v44_data);
              v613_acc += ((static_cast<float>(v615_data[5])) * v48_data);
              v613_acc += ((static_cast<float>(v615_data[6])) * v52_data);
              v613_acc += ((static_cast<float>(v615_data[7])) * v56_data);
              v613_acc += ((static_cast<float>(v615_data[8])) * v60_data);
              v613_acc += ((static_cast<float>(v615_data[9])) * v64_data);
              v613_acc += ((static_cast<float>(v615_data[10])) * v68_data);
              v613_acc += ((static_cast<float>(v615_data[11])) * v72_data);
              v613_acc += ((static_cast<float>(v615_data[12])) * v76_data);
              v613_acc += ((static_cast<float>(v615_data[13])) * v80_data);
              v613_acc += ((static_cast<float>(v615_data[14])) * v84_data);
              v613_acc += ((static_cast<float>(v615_data[15])) * v88_data);
              ir0.template select<16, 1>(240) = v613_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v648_n0 = 0; v648_n0 < 1; ++v648_n0) {
                int32_t v650_a = v648_n0 * 16;
                #pragma unroll
                for (int32_t v649_n1 = 0; v649_n1 < 16; ++v649_n1) {
                  int32_t v652_a = v650_a + (v649_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v653_data(ir0.template select<16, 1>(v652_a));
                  r0.template select<16, 1>(v652_a) = v653_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v654_i0 = 0; v654_i0 < 1; ++v654_i0) {
                int32_t v656_a = v654_i0 * 16;
                #pragma unroll
                for (int32_t v655_i1 = 0; v655_i1 < 16; ++v655_i1) {
                  int32_t v658_a = v656_a + (v655_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v659_data(r0.template select<16, 1>(v658_a));
                  v659_data.copy_to(glb_m0 + (v658_a));
                }
              }
            }
            // s0 = load{g>s}(wrap_glb_m2[0, 1])
            tensorforge::intel_esimd::simd<float, 64> v675_ld;
            v675_ld.copy_from(wrap_glb_m2 + (0 + 0 + 4 * 0 + 0));
            tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v675_ld);
            tensorforge::intel_esimd::simd<float, 64> v676_ld;
            v676_ld.copy_from(wrap_glb_m2 + (0 + 0 + 4 * 0 + 64));
            tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v676_ld);
            tensorforge::intel_esimd::simd<float, 64> v677_ld;
            v677_ld.copy_from(wrap_glb_m2 + (0 + 0 + 4 * 0 + 128));
            tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 128), v677_ld);
            tensorforge::intel_esimd::simd<float, 64> v678_ld;
            v678_ld.copy_from(wrap_glb_m2 + (0 + 0 + 4 * 0 + 192));
            tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 192), v678_ld);
            v667_flagWord = v668_flagWordNext;
            v668_flagWordNext = flagWordAhead;
          }
        }
      }
    });
  });
}

