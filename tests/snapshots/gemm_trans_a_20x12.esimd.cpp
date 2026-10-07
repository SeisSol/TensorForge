// === base name ===
kernel_7bbc13d96af8afbc

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_7bbc13d96af8afbc = {{1, 16, 1}, 16, 12, 1, 16, 37888, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_7bbc13d96af8afbc(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_7bbc13d96af8afbc(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_7bbc13d96af8afbc(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 9472 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_7bbc13d96af8afbc(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_7bbc13d96af8afbc(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_7bbc13d96af8afbc(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_7bbc13d96af8afbc(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<9472 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 37888 B shared, occupancy grid
        // operands:
        //   m0 12×16(12×16) {0..12}×{0..16} strided
        //   m1 20×12(20×12) {0..20}×{0..12} strided
        //   m2 20×16(20×16) {0..20}×{0..16} strided
        // operations:
        //   m0[i,j] = m1[k,i] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":9472}],"shared_bytes":37888,"shared_elements":9472,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,16]],"name":"m0","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[20,12]],"name":"m1","ordered":false,"parts":1,"shape":[20,12],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[20,16]],"name":"m2","ordered":false,"parts":1,"shape":[20,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[20,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[20,12]},{"addressing":"strided","bbox":[[0,0],[20,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,16]}],"permute":[[1,0],[0,1]],"target":[[-1,0],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (592 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (320);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 192 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v9_batchId0 * 240 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v9_batchId0 * 320 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m1[1, 0])
              #pragma unroll
              for (int32_t v20_i0 = 0; v20_i0 < 1; ++v20_i0) {
                int32_t v22_lead = v20_i0 * 16;
                #pragma unroll
                for (int32_t v21_i1 = 0; v21_i1 < 12; ++v21_i1) {
                  tensorforge::intel_esimd::simd<float, 16> v26_data;
                  v26_data.copy_from(glb_m1 + ((v22_lead + (v21_i1 * 20))));
                  tensorforge::slmStore<float, 16>(s0 + ((v22_lead + (v21_i1 * 21))), v26_data);
                }
              }
              #pragma unroll
              for (int32_t v29_i1 = 0; v29_i1 < 12; ++v29_i1) {
                tensorforge::intel_esimd::simd<float, 4> v35_data;
                v35_data.copy_from(glb_m1 + ((16_i32 + (v29_i1 * 20))));
                tensorforge::slmStore<float, 4>(s0 + ((16_i32 + (v29_i1 * 21))), v35_data);
              }
              // s1 = load{g>s}(glb_m2[0, 1])
              #pragma unroll
              for (int32_t i = 0; i < 20; i += 4) {
                tensorforge::intel_esimd::simd<float, 64> v38_ld;
                v38_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + i * 16));
                tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 4 * 0 + i * 16), v38_ld);
              }
              // wait(s0 = load{g>s}(glb_m1[1, 0]));
              // wait(s1 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
              // ir0 = +(s0 * s1)
              // [(0, 12), (0, 16)] [(0, 20)]
              tensorforge::intel_esimd::simd<float, 256> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v45_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v47_data = tensorforge::slmLoad<float, 16>(s0 + (1_i32));
              tensorforge::intel_esimd::simd<float, 16> v49_data = tensorforge::slmLoad<float, 16>(s0 + (2_i32));
              tensorforge::intel_esimd::simd<float, 16> v51_data = tensorforge::slmLoad<float, 16>(s0 + (3_i32));
              tensorforge::intel_esimd::simd<float, 16> v53_data = tensorforge::slmLoad<float, 16>(s0 + (4_i32));
              tensorforge::intel_esimd::simd<float, 16> v55_data = tensorforge::slmLoad<float, 16>(s0 + (5_i32));
              tensorforge::intel_esimd::simd<float, 16> v57_data = tensorforge::slmLoad<float, 16>(s0 + (6_i32));
              tensorforge::intel_esimd::simd<float, 16> v59_data = tensorforge::slmLoad<float, 16>(s0 + (7_i32));
              tensorforge::intel_esimd::simd<float, 16> v61_data = tensorforge::slmLoad<float, 16>(s0 + (8_i32));
              tensorforge::intel_esimd::simd<float, 16> v63_data = tensorforge::slmLoad<float, 16>(s0 + (9_i32));
              tensorforge::intel_esimd::simd<float, 16> v65_data = tensorforge::slmLoad<float, 16>(s0 + (10_i32));
              tensorforge::intel_esimd::simd<float, 16> v67_data = tensorforge::slmLoad<float, 16>(s0 + (11_i32));
              tensorforge::intel_esimd::simd<float, 16> v69_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              tensorforge::intel_esimd::simd<float, 16> v71_data = tensorforge::slmLoad<float, 16>(s0 + (13_i32));
              tensorforge::intel_esimd::simd<float, 16> v73_data = tensorforge::slmLoad<float, 16>(s0 + (14_i32));
              tensorforge::intel_esimd::simd<float, 16> v75_data = tensorforge::slmLoad<float, 16>(s0 + (15_i32));
              tensorforge::intel_esimd::simd<float, 16> v77_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              tensorforge::intel_esimd::simd<float, 16> v79_data = tensorforge::slmLoad<float, 16>(s0 + (17_i32));
              tensorforge::intel_esimd::simd<float, 16> v81_data = tensorforge::slmLoad<float, 16>(s0 + (18_i32));
              tensorforge::intel_esimd::simd<float, 16> v83_data = tensorforge::slmLoad<float, 16>(s0 + (19_i32));
              tensorforge::intel_esimd::simd<float, 16> v84_acc{};
              tensorforge::intel_esimd::simd<float, 16> v86_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v84_acc += ((static_cast<float>(v86_data[0])) * v45_data);
              v84_acc += ((static_cast<float>(v86_data[1])) * v47_data);
              v84_acc += ((static_cast<float>(v86_data[2])) * v49_data);
              v84_acc += ((static_cast<float>(v86_data[3])) * v51_data);
              v84_acc += ((static_cast<float>(v86_data[4])) * v53_data);
              v84_acc += ((static_cast<float>(v86_data[5])) * v55_data);
              v84_acc += ((static_cast<float>(v86_data[6])) * v57_data);
              v84_acc += ((static_cast<float>(v86_data[7])) * v59_data);
              v84_acc += ((static_cast<float>(v86_data[8])) * v61_data);
              v84_acc += ((static_cast<float>(v86_data[9])) * v63_data);
              v84_acc += ((static_cast<float>(v86_data[10])) * v65_data);
              v84_acc += ((static_cast<float>(v86_data[11])) * v67_data);
              v84_acc += ((static_cast<float>(v86_data[12])) * v69_data);
              v84_acc += ((static_cast<float>(v86_data[13])) * v71_data);
              v84_acc += ((static_cast<float>(v86_data[14])) * v73_data);
              v84_acc += ((static_cast<float>(v86_data[15])) * v75_data);
              tensorforge::intel_esimd::simd<float, 16> v122_data = tensorforge::slmLoad<float, 16>(s1 + (16_i32));
              v84_acc += ((static_cast<float>(v122_data[0])) * v77_data);
              v84_acc += ((static_cast<float>(v122_data[1])) * v79_data);
              v84_acc += ((static_cast<float>(v122_data[2])) * v81_data);
              v84_acc += ((static_cast<float>(v122_data[3])) * v83_data);
              ir0.template select<16, 1>(0) = v84_acc;
              tensorforge::intel_esimd::simd<float, 16> v131_acc{};
              tensorforge::intel_esimd::simd<float, 16> v133_data = tensorforge::slmLoad<float, 16>(s1 + (20_i32));
              v131_acc += ((static_cast<float>(v133_data[0])) * v45_data);
              v131_acc += ((static_cast<float>(v133_data[1])) * v47_data);
              v131_acc += ((static_cast<float>(v133_data[2])) * v49_data);
              v131_acc += ((static_cast<float>(v133_data[3])) * v51_data);
              v131_acc += ((static_cast<float>(v133_data[4])) * v53_data);
              v131_acc += ((static_cast<float>(v133_data[5])) * v55_data);
              v131_acc += ((static_cast<float>(v133_data[6])) * v57_data);
              v131_acc += ((static_cast<float>(v133_data[7])) * v59_data);
              v131_acc += ((static_cast<float>(v133_data[8])) * v61_data);
              v131_acc += ((static_cast<float>(v133_data[9])) * v63_data);
              v131_acc += ((static_cast<float>(v133_data[10])) * v65_data);
              v131_acc += ((static_cast<float>(v133_data[11])) * v67_data);
              v131_acc += ((static_cast<float>(v133_data[12])) * v69_data);
              v131_acc += ((static_cast<float>(v133_data[13])) * v71_data);
              v131_acc += ((static_cast<float>(v133_data[14])) * v73_data);
              v131_acc += ((static_cast<float>(v133_data[15])) * v75_data);
              tensorforge::intel_esimd::simd<float, 16> v167_data = tensorforge::slmLoad<float, 16>(s1 + (36_i32));
              v131_acc += ((static_cast<float>(v167_data[0])) * v77_data);
              v131_acc += ((static_cast<float>(v167_data[1])) * v79_data);
              v131_acc += ((static_cast<float>(v167_data[2])) * v81_data);
              v131_acc += ((static_cast<float>(v167_data[3])) * v83_data);
              ir0.template select<16, 1>(16) = v131_acc;
              tensorforge::intel_esimd::simd<float, 16> v176_acc{};
              tensorforge::intel_esimd::simd<float, 16> v178_data = tensorforge::slmLoad<float, 16>(s1 + (40_i32));
              v176_acc += ((static_cast<float>(v178_data[0])) * v45_data);
              v176_acc += ((static_cast<float>(v178_data[1])) * v47_data);
              v176_acc += ((static_cast<float>(v178_data[2])) * v49_data);
              v176_acc += ((static_cast<float>(v178_data[3])) * v51_data);
              v176_acc += ((static_cast<float>(v178_data[4])) * v53_data);
              v176_acc += ((static_cast<float>(v178_data[5])) * v55_data);
              v176_acc += ((static_cast<float>(v178_data[6])) * v57_data);
              v176_acc += ((static_cast<float>(v178_data[7])) * v59_data);
              v176_acc += ((static_cast<float>(v178_data[8])) * v61_data);
              v176_acc += ((static_cast<float>(v178_data[9])) * v63_data);
              v176_acc += ((static_cast<float>(v178_data[10])) * v65_data);
              v176_acc += ((static_cast<float>(v178_data[11])) * v67_data);
              v176_acc += ((static_cast<float>(v178_data[12])) * v69_data);
              v176_acc += ((static_cast<float>(v178_data[13])) * v71_data);
              v176_acc += ((static_cast<float>(v178_data[14])) * v73_data);
              v176_acc += ((static_cast<float>(v178_data[15])) * v75_data);
              tensorforge::intel_esimd::simd<float, 16> v212_data = tensorforge::slmLoad<float, 16>(s1 + (56_i32));
              v176_acc += ((static_cast<float>(v212_data[0])) * v77_data);
              v176_acc += ((static_cast<float>(v212_data[1])) * v79_data);
              v176_acc += ((static_cast<float>(v212_data[2])) * v81_data);
              v176_acc += ((static_cast<float>(v212_data[3])) * v83_data);
              ir0.template select<16, 1>(32) = v176_acc;
              tensorforge::intel_esimd::simd<float, 16> v221_acc{};
              tensorforge::intel_esimd::simd<float, 16> v223_data = tensorforge::slmLoad<float, 16>(s1 + (60_i32));
              v221_acc += ((static_cast<float>(v223_data[0])) * v45_data);
              v221_acc += ((static_cast<float>(v223_data[1])) * v47_data);
              v221_acc += ((static_cast<float>(v223_data[2])) * v49_data);
              v221_acc += ((static_cast<float>(v223_data[3])) * v51_data);
              v221_acc += ((static_cast<float>(v223_data[4])) * v53_data);
              v221_acc += ((static_cast<float>(v223_data[5])) * v55_data);
              v221_acc += ((static_cast<float>(v223_data[6])) * v57_data);
              v221_acc += ((static_cast<float>(v223_data[7])) * v59_data);
              v221_acc += ((static_cast<float>(v223_data[8])) * v61_data);
              v221_acc += ((static_cast<float>(v223_data[9])) * v63_data);
              v221_acc += ((static_cast<float>(v223_data[10])) * v65_data);
              v221_acc += ((static_cast<float>(v223_data[11])) * v67_data);
              v221_acc += ((static_cast<float>(v223_data[12])) * v69_data);
              v221_acc += ((static_cast<float>(v223_data[13])) * v71_data);
              v221_acc += ((static_cast<float>(v223_data[14])) * v73_data);
              v221_acc += ((static_cast<float>(v223_data[15])) * v75_data);
              tensorforge::intel_esimd::simd<float, 16> v257_data = tensorforge::slmLoad<float, 16>(s1 + (76_i32));
              v221_acc += ((static_cast<float>(v257_data[0])) * v77_data);
              v221_acc += ((static_cast<float>(v257_data[1])) * v79_data);
              v221_acc += ((static_cast<float>(v257_data[2])) * v81_data);
              v221_acc += ((static_cast<float>(v257_data[3])) * v83_data);
              ir0.template select<16, 1>(48) = v221_acc;
              tensorforge::intel_esimd::simd<float, 16> v266_acc{};
              tensorforge::intel_esimd::simd<float, 16> v268_data = tensorforge::slmLoad<float, 16>(s1 + (80_i32));
              v266_acc += ((static_cast<float>(v268_data[0])) * v45_data);
              v266_acc += ((static_cast<float>(v268_data[1])) * v47_data);
              v266_acc += ((static_cast<float>(v268_data[2])) * v49_data);
              v266_acc += ((static_cast<float>(v268_data[3])) * v51_data);
              v266_acc += ((static_cast<float>(v268_data[4])) * v53_data);
              v266_acc += ((static_cast<float>(v268_data[5])) * v55_data);
              v266_acc += ((static_cast<float>(v268_data[6])) * v57_data);
              v266_acc += ((static_cast<float>(v268_data[7])) * v59_data);
              v266_acc += ((static_cast<float>(v268_data[8])) * v61_data);
              v266_acc += ((static_cast<float>(v268_data[9])) * v63_data);
              v266_acc += ((static_cast<float>(v268_data[10])) * v65_data);
              v266_acc += ((static_cast<float>(v268_data[11])) * v67_data);
              v266_acc += ((static_cast<float>(v268_data[12])) * v69_data);
              v266_acc += ((static_cast<float>(v268_data[13])) * v71_data);
              v266_acc += ((static_cast<float>(v268_data[14])) * v73_data);
              v266_acc += ((static_cast<float>(v268_data[15])) * v75_data);
              tensorforge::intel_esimd::simd<float, 16> v302_data = tensorforge::slmLoad<float, 16>(s1 + (96_i32));
              v266_acc += ((static_cast<float>(v302_data[0])) * v77_data);
              v266_acc += ((static_cast<float>(v302_data[1])) * v79_data);
              v266_acc += ((static_cast<float>(v302_data[2])) * v81_data);
              v266_acc += ((static_cast<float>(v302_data[3])) * v83_data);
              ir0.template select<16, 1>(64) = v266_acc;
              tensorforge::intel_esimd::simd<float, 16> v311_acc{};
              tensorforge::intel_esimd::simd<float, 16> v313_data = tensorforge::slmLoad<float, 16>(s1 + (100_i32));
              v311_acc += ((static_cast<float>(v313_data[0])) * v45_data);
              v311_acc += ((static_cast<float>(v313_data[1])) * v47_data);
              v311_acc += ((static_cast<float>(v313_data[2])) * v49_data);
              v311_acc += ((static_cast<float>(v313_data[3])) * v51_data);
              v311_acc += ((static_cast<float>(v313_data[4])) * v53_data);
              v311_acc += ((static_cast<float>(v313_data[5])) * v55_data);
              v311_acc += ((static_cast<float>(v313_data[6])) * v57_data);
              v311_acc += ((static_cast<float>(v313_data[7])) * v59_data);
              v311_acc += ((static_cast<float>(v313_data[8])) * v61_data);
              v311_acc += ((static_cast<float>(v313_data[9])) * v63_data);
              v311_acc += ((static_cast<float>(v313_data[10])) * v65_data);
              v311_acc += ((static_cast<float>(v313_data[11])) * v67_data);
              v311_acc += ((static_cast<float>(v313_data[12])) * v69_data);
              v311_acc += ((static_cast<float>(v313_data[13])) * v71_data);
              v311_acc += ((static_cast<float>(v313_data[14])) * v73_data);
              v311_acc += ((static_cast<float>(v313_data[15])) * v75_data);
              tensorforge::intel_esimd::simd<float, 16> v347_data = tensorforge::slmLoad<float, 16>(s1 + (116_i32));
              v311_acc += ((static_cast<float>(v347_data[0])) * v77_data);
              v311_acc += ((static_cast<float>(v347_data[1])) * v79_data);
              v311_acc += ((static_cast<float>(v347_data[2])) * v81_data);
              v311_acc += ((static_cast<float>(v347_data[3])) * v83_data);
              ir0.template select<16, 1>(80) = v311_acc;
              tensorforge::intel_esimd::simd<float, 16> v356_acc{};
              tensorforge::intel_esimd::simd<float, 16> v358_data = tensorforge::slmLoad<float, 16>(s1 + (120_i32));
              v356_acc += ((static_cast<float>(v358_data[0])) * v45_data);
              v356_acc += ((static_cast<float>(v358_data[1])) * v47_data);
              v356_acc += ((static_cast<float>(v358_data[2])) * v49_data);
              v356_acc += ((static_cast<float>(v358_data[3])) * v51_data);
              v356_acc += ((static_cast<float>(v358_data[4])) * v53_data);
              v356_acc += ((static_cast<float>(v358_data[5])) * v55_data);
              v356_acc += ((static_cast<float>(v358_data[6])) * v57_data);
              v356_acc += ((static_cast<float>(v358_data[7])) * v59_data);
              v356_acc += ((static_cast<float>(v358_data[8])) * v61_data);
              v356_acc += ((static_cast<float>(v358_data[9])) * v63_data);
              v356_acc += ((static_cast<float>(v358_data[10])) * v65_data);
              v356_acc += ((static_cast<float>(v358_data[11])) * v67_data);
              v356_acc += ((static_cast<float>(v358_data[12])) * v69_data);
              v356_acc += ((static_cast<float>(v358_data[13])) * v71_data);
              v356_acc += ((static_cast<float>(v358_data[14])) * v73_data);
              v356_acc += ((static_cast<float>(v358_data[15])) * v75_data);
              tensorforge::intel_esimd::simd<float, 16> v392_data = tensorforge::slmLoad<float, 16>(s1 + (136_i32));
              v356_acc += ((static_cast<float>(v392_data[0])) * v77_data);
              v356_acc += ((static_cast<float>(v392_data[1])) * v79_data);
              v356_acc += ((static_cast<float>(v392_data[2])) * v81_data);
              v356_acc += ((static_cast<float>(v392_data[3])) * v83_data);
              ir0.template select<16, 1>(96) = v356_acc;
              tensorforge::intel_esimd::simd<float, 16> v401_acc{};
              tensorforge::intel_esimd::simd<float, 16> v403_data = tensorforge::slmLoad<float, 16>(s1 + (140_i32));
              v401_acc += ((static_cast<float>(v403_data[0])) * v45_data);
              v401_acc += ((static_cast<float>(v403_data[1])) * v47_data);
              v401_acc += ((static_cast<float>(v403_data[2])) * v49_data);
              v401_acc += ((static_cast<float>(v403_data[3])) * v51_data);
              v401_acc += ((static_cast<float>(v403_data[4])) * v53_data);
              v401_acc += ((static_cast<float>(v403_data[5])) * v55_data);
              v401_acc += ((static_cast<float>(v403_data[6])) * v57_data);
              v401_acc += ((static_cast<float>(v403_data[7])) * v59_data);
              v401_acc += ((static_cast<float>(v403_data[8])) * v61_data);
              v401_acc += ((static_cast<float>(v403_data[9])) * v63_data);
              v401_acc += ((static_cast<float>(v403_data[10])) * v65_data);
              v401_acc += ((static_cast<float>(v403_data[11])) * v67_data);
              v401_acc += ((static_cast<float>(v403_data[12])) * v69_data);
              v401_acc += ((static_cast<float>(v403_data[13])) * v71_data);
              v401_acc += ((static_cast<float>(v403_data[14])) * v73_data);
              v401_acc += ((static_cast<float>(v403_data[15])) * v75_data);
              tensorforge::intel_esimd::simd<float, 16> v437_data = tensorforge::slmLoad<float, 16>(s1 + (156_i32));
              v401_acc += ((static_cast<float>(v437_data[0])) * v77_data);
              v401_acc += ((static_cast<float>(v437_data[1])) * v79_data);
              v401_acc += ((static_cast<float>(v437_data[2])) * v81_data);
              v401_acc += ((static_cast<float>(v437_data[3])) * v83_data);
              ir0.template select<16, 1>(112) = v401_acc;
              tensorforge::intel_esimd::simd<float, 16> v446_acc{};
              tensorforge::intel_esimd::simd<float, 16> v448_data = tensorforge::slmLoad<float, 16>(s1 + (160_i32));
              v446_acc += ((static_cast<float>(v448_data[0])) * v45_data);
              v446_acc += ((static_cast<float>(v448_data[1])) * v47_data);
              v446_acc += ((static_cast<float>(v448_data[2])) * v49_data);
              v446_acc += ((static_cast<float>(v448_data[3])) * v51_data);
              v446_acc += ((static_cast<float>(v448_data[4])) * v53_data);
              v446_acc += ((static_cast<float>(v448_data[5])) * v55_data);
              v446_acc += ((static_cast<float>(v448_data[6])) * v57_data);
              v446_acc += ((static_cast<float>(v448_data[7])) * v59_data);
              v446_acc += ((static_cast<float>(v448_data[8])) * v61_data);
              v446_acc += ((static_cast<float>(v448_data[9])) * v63_data);
              v446_acc += ((static_cast<float>(v448_data[10])) * v65_data);
              v446_acc += ((static_cast<float>(v448_data[11])) * v67_data);
              v446_acc += ((static_cast<float>(v448_data[12])) * v69_data);
              v446_acc += ((static_cast<float>(v448_data[13])) * v71_data);
              v446_acc += ((static_cast<float>(v448_data[14])) * v73_data);
              v446_acc += ((static_cast<float>(v448_data[15])) * v75_data);
              tensorforge::intel_esimd::simd<float, 16> v482_data = tensorforge::slmLoad<float, 16>(s1 + (176_i32));
              v446_acc += ((static_cast<float>(v482_data[0])) * v77_data);
              v446_acc += ((static_cast<float>(v482_data[1])) * v79_data);
              v446_acc += ((static_cast<float>(v482_data[2])) * v81_data);
              v446_acc += ((static_cast<float>(v482_data[3])) * v83_data);
              ir0.template select<16, 1>(128) = v446_acc;
              tensorforge::intel_esimd::simd<float, 16> v491_acc{};
              tensorforge::intel_esimd::simd<float, 16> v493_data = tensorforge::slmLoad<float, 16>(s1 + (180_i32));
              v491_acc += ((static_cast<float>(v493_data[0])) * v45_data);
              v491_acc += ((static_cast<float>(v493_data[1])) * v47_data);
              v491_acc += ((static_cast<float>(v493_data[2])) * v49_data);
              v491_acc += ((static_cast<float>(v493_data[3])) * v51_data);
              v491_acc += ((static_cast<float>(v493_data[4])) * v53_data);
              v491_acc += ((static_cast<float>(v493_data[5])) * v55_data);
              v491_acc += ((static_cast<float>(v493_data[6])) * v57_data);
              v491_acc += ((static_cast<float>(v493_data[7])) * v59_data);
              v491_acc += ((static_cast<float>(v493_data[8])) * v61_data);
              v491_acc += ((static_cast<float>(v493_data[9])) * v63_data);
              v491_acc += ((static_cast<float>(v493_data[10])) * v65_data);
              v491_acc += ((static_cast<float>(v493_data[11])) * v67_data);
              v491_acc += ((static_cast<float>(v493_data[12])) * v69_data);
              v491_acc += ((static_cast<float>(v493_data[13])) * v71_data);
              v491_acc += ((static_cast<float>(v493_data[14])) * v73_data);
              v491_acc += ((static_cast<float>(v493_data[15])) * v75_data);
              tensorforge::intel_esimd::simd<float, 16> v527_data = tensorforge::slmLoad<float, 16>(s1 + (196_i32));
              v491_acc += ((static_cast<float>(v527_data[0])) * v77_data);
              v491_acc += ((static_cast<float>(v527_data[1])) * v79_data);
              v491_acc += ((static_cast<float>(v527_data[2])) * v81_data);
              v491_acc += ((static_cast<float>(v527_data[3])) * v83_data);
              ir0.template select<16, 1>(144) = v491_acc;
              tensorforge::intel_esimd::simd<float, 16> v536_acc{};
              tensorforge::intel_esimd::simd<float, 16> v538_data = tensorforge::slmLoad<float, 16>(s1 + (200_i32));
              v536_acc += ((static_cast<float>(v538_data[0])) * v45_data);
              v536_acc += ((static_cast<float>(v538_data[1])) * v47_data);
              v536_acc += ((static_cast<float>(v538_data[2])) * v49_data);
              v536_acc += ((static_cast<float>(v538_data[3])) * v51_data);
              v536_acc += ((static_cast<float>(v538_data[4])) * v53_data);
              v536_acc += ((static_cast<float>(v538_data[5])) * v55_data);
              v536_acc += ((static_cast<float>(v538_data[6])) * v57_data);
              v536_acc += ((static_cast<float>(v538_data[7])) * v59_data);
              v536_acc += ((static_cast<float>(v538_data[8])) * v61_data);
              v536_acc += ((static_cast<float>(v538_data[9])) * v63_data);
              v536_acc += ((static_cast<float>(v538_data[10])) * v65_data);
              v536_acc += ((static_cast<float>(v538_data[11])) * v67_data);
              v536_acc += ((static_cast<float>(v538_data[12])) * v69_data);
              v536_acc += ((static_cast<float>(v538_data[13])) * v71_data);
              v536_acc += ((static_cast<float>(v538_data[14])) * v73_data);
              v536_acc += ((static_cast<float>(v538_data[15])) * v75_data);
              tensorforge::intel_esimd::simd<float, 16> v572_data = tensorforge::slmLoad<float, 16>(s1 + (216_i32));
              v536_acc += ((static_cast<float>(v572_data[0])) * v77_data);
              v536_acc += ((static_cast<float>(v572_data[1])) * v79_data);
              v536_acc += ((static_cast<float>(v572_data[2])) * v81_data);
              v536_acc += ((static_cast<float>(v572_data[3])) * v83_data);
              ir0.template select<16, 1>(160) = v536_acc;
              tensorforge::intel_esimd::simd<float, 16> v581_acc{};
              tensorforge::intel_esimd::simd<float, 16> v583_data = tensorforge::slmLoad<float, 16>(s1 + (220_i32));
              v581_acc += ((static_cast<float>(v583_data[0])) * v45_data);
              v581_acc += ((static_cast<float>(v583_data[1])) * v47_data);
              v581_acc += ((static_cast<float>(v583_data[2])) * v49_data);
              v581_acc += ((static_cast<float>(v583_data[3])) * v51_data);
              v581_acc += ((static_cast<float>(v583_data[4])) * v53_data);
              v581_acc += ((static_cast<float>(v583_data[5])) * v55_data);
              v581_acc += ((static_cast<float>(v583_data[6])) * v57_data);
              v581_acc += ((static_cast<float>(v583_data[7])) * v59_data);
              v581_acc += ((static_cast<float>(v583_data[8])) * v61_data);
              v581_acc += ((static_cast<float>(v583_data[9])) * v63_data);
              v581_acc += ((static_cast<float>(v583_data[10])) * v65_data);
              v581_acc += ((static_cast<float>(v583_data[11])) * v67_data);
              v581_acc += ((static_cast<float>(v583_data[12])) * v69_data);
              v581_acc += ((static_cast<float>(v583_data[13])) * v71_data);
              v581_acc += ((static_cast<float>(v583_data[14])) * v73_data);
              v581_acc += ((static_cast<float>(v583_data[15])) * v75_data);
              tensorforge::intel_esimd::simd<float, 16> v617_data = tensorforge::slmLoad<float, 16>(s1 + (236_i32));
              v581_acc += ((static_cast<float>(v617_data[0])) * v77_data);
              v581_acc += ((static_cast<float>(v617_data[1])) * v79_data);
              v581_acc += ((static_cast<float>(v617_data[2])) * v81_data);
              v581_acc += ((static_cast<float>(v617_data[3])) * v83_data);
              ir0.template select<16, 1>(176) = v581_acc;
              tensorforge::intel_esimd::simd<float, 16> v626_acc{};
              tensorforge::intel_esimd::simd<float, 16> v628_data = tensorforge::slmLoad<float, 16>(s1 + (240_i32));
              v626_acc += ((static_cast<float>(v628_data[0])) * v45_data);
              v626_acc += ((static_cast<float>(v628_data[1])) * v47_data);
              v626_acc += ((static_cast<float>(v628_data[2])) * v49_data);
              v626_acc += ((static_cast<float>(v628_data[3])) * v51_data);
              v626_acc += ((static_cast<float>(v628_data[4])) * v53_data);
              v626_acc += ((static_cast<float>(v628_data[5])) * v55_data);
              v626_acc += ((static_cast<float>(v628_data[6])) * v57_data);
              v626_acc += ((static_cast<float>(v628_data[7])) * v59_data);
              v626_acc += ((static_cast<float>(v628_data[8])) * v61_data);
              v626_acc += ((static_cast<float>(v628_data[9])) * v63_data);
              v626_acc += ((static_cast<float>(v628_data[10])) * v65_data);
              v626_acc += ((static_cast<float>(v628_data[11])) * v67_data);
              v626_acc += ((static_cast<float>(v628_data[12])) * v69_data);
              v626_acc += ((static_cast<float>(v628_data[13])) * v71_data);
              v626_acc += ((static_cast<float>(v628_data[14])) * v73_data);
              v626_acc += ((static_cast<float>(v628_data[15])) * v75_data);
              tensorforge::intel_esimd::simd<float, 16> v662_data = tensorforge::slmLoad<float, 16>(s1 + (256_i32));
              v626_acc += ((static_cast<float>(v662_data[0])) * v77_data);
              v626_acc += ((static_cast<float>(v662_data[1])) * v79_data);
              v626_acc += ((static_cast<float>(v662_data[2])) * v81_data);
              v626_acc += ((static_cast<float>(v662_data[3])) * v83_data);
              ir0.template select<16, 1>(192) = v626_acc;
              tensorforge::intel_esimd::simd<float, 16> v671_acc{};
              tensorforge::intel_esimd::simd<float, 16> v673_data = tensorforge::slmLoad<float, 16>(s1 + (260_i32));
              v671_acc += ((static_cast<float>(v673_data[0])) * v45_data);
              v671_acc += ((static_cast<float>(v673_data[1])) * v47_data);
              v671_acc += ((static_cast<float>(v673_data[2])) * v49_data);
              v671_acc += ((static_cast<float>(v673_data[3])) * v51_data);
              v671_acc += ((static_cast<float>(v673_data[4])) * v53_data);
              v671_acc += ((static_cast<float>(v673_data[5])) * v55_data);
              v671_acc += ((static_cast<float>(v673_data[6])) * v57_data);
              v671_acc += ((static_cast<float>(v673_data[7])) * v59_data);
              v671_acc += ((static_cast<float>(v673_data[8])) * v61_data);
              v671_acc += ((static_cast<float>(v673_data[9])) * v63_data);
              v671_acc += ((static_cast<float>(v673_data[10])) * v65_data);
              v671_acc += ((static_cast<float>(v673_data[11])) * v67_data);
              v671_acc += ((static_cast<float>(v673_data[12])) * v69_data);
              v671_acc += ((static_cast<float>(v673_data[13])) * v71_data);
              v671_acc += ((static_cast<float>(v673_data[14])) * v73_data);
              v671_acc += ((static_cast<float>(v673_data[15])) * v75_data);
              tensorforge::intel_esimd::simd<float, 16> v707_data = tensorforge::slmLoad<float, 16>(s1 + (276_i32));
              v671_acc += ((static_cast<float>(v707_data[0])) * v77_data);
              v671_acc += ((static_cast<float>(v707_data[1])) * v79_data);
              v671_acc += ((static_cast<float>(v707_data[2])) * v81_data);
              v671_acc += ((static_cast<float>(v707_data[3])) * v83_data);
              ir0.template select<16, 1>(208) = v671_acc;
              tensorforge::intel_esimd::simd<float, 16> v716_acc{};
              tensorforge::intel_esimd::simd<float, 16> v718_data = tensorforge::slmLoad<float, 16>(s1 + (280_i32));
              v716_acc += ((static_cast<float>(v718_data[0])) * v45_data);
              v716_acc += ((static_cast<float>(v718_data[1])) * v47_data);
              v716_acc += ((static_cast<float>(v718_data[2])) * v49_data);
              v716_acc += ((static_cast<float>(v718_data[3])) * v51_data);
              v716_acc += ((static_cast<float>(v718_data[4])) * v53_data);
              v716_acc += ((static_cast<float>(v718_data[5])) * v55_data);
              v716_acc += ((static_cast<float>(v718_data[6])) * v57_data);
              v716_acc += ((static_cast<float>(v718_data[7])) * v59_data);
              v716_acc += ((static_cast<float>(v718_data[8])) * v61_data);
              v716_acc += ((static_cast<float>(v718_data[9])) * v63_data);
              v716_acc += ((static_cast<float>(v718_data[10])) * v65_data);
              v716_acc += ((static_cast<float>(v718_data[11])) * v67_data);
              v716_acc += ((static_cast<float>(v718_data[12])) * v69_data);
              v716_acc += ((static_cast<float>(v718_data[13])) * v71_data);
              v716_acc += ((static_cast<float>(v718_data[14])) * v73_data);
              v716_acc += ((static_cast<float>(v718_data[15])) * v75_data);
              tensorforge::intel_esimd::simd<float, 16> v752_data = tensorforge::slmLoad<float, 16>(s1 + (296_i32));
              v716_acc += ((static_cast<float>(v752_data[0])) * v77_data);
              v716_acc += ((static_cast<float>(v752_data[1])) * v79_data);
              v716_acc += ((static_cast<float>(v752_data[2])) * v81_data);
              v716_acc += ((static_cast<float>(v752_data[3])) * v83_data);
              ir0.template select<16, 1>(224) = v716_acc;
              tensorforge::intel_esimd::simd<float, 16> v761_acc{};
              tensorforge::intel_esimd::simd<float, 16> v763_data = tensorforge::slmLoad<float, 16>(s1 + (300_i32));
              v761_acc += ((static_cast<float>(v763_data[0])) * v45_data);
              v761_acc += ((static_cast<float>(v763_data[1])) * v47_data);
              v761_acc += ((static_cast<float>(v763_data[2])) * v49_data);
              v761_acc += ((static_cast<float>(v763_data[3])) * v51_data);
              v761_acc += ((static_cast<float>(v763_data[4])) * v53_data);
              v761_acc += ((static_cast<float>(v763_data[5])) * v55_data);
              v761_acc += ((static_cast<float>(v763_data[6])) * v57_data);
              v761_acc += ((static_cast<float>(v763_data[7])) * v59_data);
              v761_acc += ((static_cast<float>(v763_data[8])) * v61_data);
              v761_acc += ((static_cast<float>(v763_data[9])) * v63_data);
              v761_acc += ((static_cast<float>(v763_data[10])) * v65_data);
              v761_acc += ((static_cast<float>(v763_data[11])) * v67_data);
              v761_acc += ((static_cast<float>(v763_data[12])) * v69_data);
              v761_acc += ((static_cast<float>(v763_data[13])) * v71_data);
              v761_acc += ((static_cast<float>(v763_data[14])) * v73_data);
              v761_acc += ((static_cast<float>(v763_data[15])) * v75_data);
              tensorforge::intel_esimd::simd<float, 16> v797_data = tensorforge::slmLoad<float, 16>(s1 + (316_i32));
              v761_acc += ((static_cast<float>(v797_data[0])) * v77_data);
              v761_acc += ((static_cast<float>(v797_data[1])) * v79_data);
              v761_acc += ((static_cast<float>(v797_data[2])) * v81_data);
              v761_acc += ((static_cast<float>(v797_data[3])) * v83_data);
              ir0.template select<16, 1>(240) = v761_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v806_n1 = 0; v806_n1 < 16; ++v806_n1) {
                int32_t v807_a = v806_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v809_data(ir0.template select<12, 1>(v807_a));
                r0.template select<12, 1>(v807_a) = v809_data;
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v810_i1 = 0; v810_i1 < 16; ++v810_i1) {
                tensorforge::intel_esimd::simd<float, 12> v813_data(r0.template select<12, 1>((v810_i1 * 16)));
                v813_data.copy_to(glb_m0 + ((v810_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}

