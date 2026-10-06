// === base name ===
kernel_23d1a05e880b2b35

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_23d1a05e880b2b35 = {{1, 16, 1}, 16, 12, 1, 16, 37888, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_23d1a05e880b2b35(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_23d1a05e880b2b35(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_23d1a05e880b2b35(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_23d1a05e880b2b35(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_23d1a05e880b2b35(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_23d1a05e880b2b35(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_23d1a05e880b2b35(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (576);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (320);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v12_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v12_batchId0 < numElements0; v12_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v13_ahead1 = v12_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v15_batchId1 = (v13_ahead1 < numElements0) ? v13_ahead1 : v12_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v12_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v12_batchId0 * 192 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v12_batchId0 * 240 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v12_batchId0 * 320 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m1[1, 0])
              #pragma unroll
              for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
                int32_t v25_lead = v23_i0 * 16;
                #pragma unroll
                for (int32_t v24_i1 = 0; v24_i1 < 12; ++v24_i1) {
                  tensorforge::intel_esimd::simd<float, 16> v29_data;
                  v29_data.copy_from(glb_m1 + ((v25_lead + (v24_i1 * 20))));
                  tensorforge::slmStore<float, 16>(s0 + ((v25_lead + (v24_i1 * 21))), v29_data);
                }
              }
              #pragma unroll
              for (int32_t v32_i1 = 0; v32_i1 < 12; ++v32_i1) {
                tensorforge::intel_esimd::simd<float, 4> v38_data;
                v38_data.copy_from(glb_m1 + ((16_i32 + (v32_i1 * 20))));
                tensorforge::slmStore<float, 4>(s0 + ((16_i32 + (v32_i1 * 21))), v38_data);
              }
              // s1 = load{g>s}(glb_m2[0, 1])
              #pragma unroll
              for (int32_t i = 0; i < 20; i += 4) {
                tensorforge::intel_esimd::simd<float, 64> v41_ld;
                v41_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + i * 16));
                tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 4 * 0 + i * 16), v41_ld);
              }
              // wait(s0 = load{g>s}(glb_m1[1, 0]));
              // wait(s1 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
              // ir0 = +(s0 * s1)
              // [(0, 12), (0, 16)] [(0, 20)]
              tensorforge::intel_esimd::simd<float, 256> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v48_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v50_data = tensorforge::slmLoad<float, 16>(s0 + (1_i32));
              tensorforge::intel_esimd::simd<float, 16> v52_data = tensorforge::slmLoad<float, 16>(s0 + (2_i32));
              tensorforge::intel_esimd::simd<float, 16> v54_data = tensorforge::slmLoad<float, 16>(s0 + (3_i32));
              tensorforge::intel_esimd::simd<float, 16> v56_data = tensorforge::slmLoad<float, 16>(s0 + (4_i32));
              tensorforge::intel_esimd::simd<float, 16> v58_data = tensorforge::slmLoad<float, 16>(s0 + (5_i32));
              tensorforge::intel_esimd::simd<float, 16> v60_data = tensorforge::slmLoad<float, 16>(s0 + (6_i32));
              tensorforge::intel_esimd::simd<float, 16> v62_data = tensorforge::slmLoad<float, 16>(s0 + (7_i32));
              tensorforge::intel_esimd::simd<float, 16> v64_data = tensorforge::slmLoad<float, 16>(s0 + (8_i32));
              tensorforge::intel_esimd::simd<float, 16> v66_data = tensorforge::slmLoad<float, 16>(s0 + (9_i32));
              tensorforge::intel_esimd::simd<float, 16> v68_data = tensorforge::slmLoad<float, 16>(s0 + (10_i32));
              tensorforge::intel_esimd::simd<float, 16> v70_data = tensorforge::slmLoad<float, 16>(s0 + (11_i32));
              tensorforge::intel_esimd::simd<float, 16> v72_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              tensorforge::intel_esimd::simd<float, 16> v74_data = tensorforge::slmLoad<float, 16>(s0 + (13_i32));
              tensorforge::intel_esimd::simd<float, 16> v76_data = tensorforge::slmLoad<float, 16>(s0 + (14_i32));
              tensorforge::intel_esimd::simd<float, 16> v78_data = tensorforge::slmLoad<float, 16>(s0 + (15_i32));
              tensorforge::intel_esimd::simd<float, 16> v80_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              tensorforge::intel_esimd::simd<float, 16> v82_data = tensorforge::slmLoad<float, 16>(s0 + (17_i32));
              tensorforge::intel_esimd::simd<float, 16> v84_data = tensorforge::slmLoad<float, 16>(s0 + (18_i32));
              tensorforge::intel_esimd::simd<float, 16> v86_data = tensorforge::slmLoad<float, 16>(s0 + (19_i32));
              tensorforge::intel_esimd::simd<float, 16> v87_acc{};
              tensorforge::intel_esimd::simd<float, 16> v89_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v87_acc += ((static_cast<float>(v89_data[0])) * v48_data);
              v87_acc += ((static_cast<float>(v89_data[1])) * v50_data);
              v87_acc += ((static_cast<float>(v89_data[2])) * v52_data);
              v87_acc += ((static_cast<float>(v89_data[3])) * v54_data);
              v87_acc += ((static_cast<float>(v89_data[4])) * v56_data);
              v87_acc += ((static_cast<float>(v89_data[5])) * v58_data);
              v87_acc += ((static_cast<float>(v89_data[6])) * v60_data);
              v87_acc += ((static_cast<float>(v89_data[7])) * v62_data);
              v87_acc += ((static_cast<float>(v89_data[8])) * v64_data);
              v87_acc += ((static_cast<float>(v89_data[9])) * v66_data);
              v87_acc += ((static_cast<float>(v89_data[10])) * v68_data);
              v87_acc += ((static_cast<float>(v89_data[11])) * v70_data);
              v87_acc += ((static_cast<float>(v89_data[12])) * v72_data);
              v87_acc += ((static_cast<float>(v89_data[13])) * v74_data);
              v87_acc += ((static_cast<float>(v89_data[14])) * v76_data);
              v87_acc += ((static_cast<float>(v89_data[15])) * v78_data);
              tensorforge::intel_esimd::simd<float, 16> v125_data = tensorforge::slmLoad<float, 16>(s1 + (16_i32));
              v87_acc += ((static_cast<float>(v125_data[0])) * v80_data);
              v87_acc += ((static_cast<float>(v125_data[1])) * v82_data);
              v87_acc += ((static_cast<float>(v125_data[2])) * v84_data);
              v87_acc += ((static_cast<float>(v125_data[3])) * v86_data);
              ir0.template select<16, 1>(0) = v87_acc;
              tensorforge::intel_esimd::simd<float, 16> v134_acc{};
              tensorforge::intel_esimd::simd<float, 16> v136_data = tensorforge::slmLoad<float, 16>(s1 + (20_i32));
              v134_acc += ((static_cast<float>(v136_data[0])) * v48_data);
              v134_acc += ((static_cast<float>(v136_data[1])) * v50_data);
              v134_acc += ((static_cast<float>(v136_data[2])) * v52_data);
              v134_acc += ((static_cast<float>(v136_data[3])) * v54_data);
              v134_acc += ((static_cast<float>(v136_data[4])) * v56_data);
              v134_acc += ((static_cast<float>(v136_data[5])) * v58_data);
              v134_acc += ((static_cast<float>(v136_data[6])) * v60_data);
              v134_acc += ((static_cast<float>(v136_data[7])) * v62_data);
              v134_acc += ((static_cast<float>(v136_data[8])) * v64_data);
              v134_acc += ((static_cast<float>(v136_data[9])) * v66_data);
              v134_acc += ((static_cast<float>(v136_data[10])) * v68_data);
              v134_acc += ((static_cast<float>(v136_data[11])) * v70_data);
              v134_acc += ((static_cast<float>(v136_data[12])) * v72_data);
              v134_acc += ((static_cast<float>(v136_data[13])) * v74_data);
              v134_acc += ((static_cast<float>(v136_data[14])) * v76_data);
              v134_acc += ((static_cast<float>(v136_data[15])) * v78_data);
              tensorforge::intel_esimd::simd<float, 16> v170_data = tensorforge::slmLoad<float, 16>(s1 + (36_i32));
              v134_acc += ((static_cast<float>(v170_data[0])) * v80_data);
              v134_acc += ((static_cast<float>(v170_data[1])) * v82_data);
              v134_acc += ((static_cast<float>(v170_data[2])) * v84_data);
              v134_acc += ((static_cast<float>(v170_data[3])) * v86_data);
              ir0.template select<16, 1>(16) = v134_acc;
              tensorforge::intel_esimd::simd<float, 16> v179_acc{};
              tensorforge::intel_esimd::simd<float, 16> v181_data = tensorforge::slmLoad<float, 16>(s1 + (40_i32));
              v179_acc += ((static_cast<float>(v181_data[0])) * v48_data);
              v179_acc += ((static_cast<float>(v181_data[1])) * v50_data);
              v179_acc += ((static_cast<float>(v181_data[2])) * v52_data);
              v179_acc += ((static_cast<float>(v181_data[3])) * v54_data);
              v179_acc += ((static_cast<float>(v181_data[4])) * v56_data);
              v179_acc += ((static_cast<float>(v181_data[5])) * v58_data);
              v179_acc += ((static_cast<float>(v181_data[6])) * v60_data);
              v179_acc += ((static_cast<float>(v181_data[7])) * v62_data);
              v179_acc += ((static_cast<float>(v181_data[8])) * v64_data);
              v179_acc += ((static_cast<float>(v181_data[9])) * v66_data);
              v179_acc += ((static_cast<float>(v181_data[10])) * v68_data);
              v179_acc += ((static_cast<float>(v181_data[11])) * v70_data);
              v179_acc += ((static_cast<float>(v181_data[12])) * v72_data);
              v179_acc += ((static_cast<float>(v181_data[13])) * v74_data);
              v179_acc += ((static_cast<float>(v181_data[14])) * v76_data);
              v179_acc += ((static_cast<float>(v181_data[15])) * v78_data);
              tensorforge::intel_esimd::simd<float, 16> v215_data = tensorforge::slmLoad<float, 16>(s1 + (56_i32));
              v179_acc += ((static_cast<float>(v215_data[0])) * v80_data);
              v179_acc += ((static_cast<float>(v215_data[1])) * v82_data);
              v179_acc += ((static_cast<float>(v215_data[2])) * v84_data);
              v179_acc += ((static_cast<float>(v215_data[3])) * v86_data);
              ir0.template select<16, 1>(32) = v179_acc;
              tensorforge::intel_esimd::simd<float, 16> v224_acc{};
              tensorforge::intel_esimd::simd<float, 16> v226_data = tensorforge::slmLoad<float, 16>(s1 + (60_i32));
              v224_acc += ((static_cast<float>(v226_data[0])) * v48_data);
              v224_acc += ((static_cast<float>(v226_data[1])) * v50_data);
              v224_acc += ((static_cast<float>(v226_data[2])) * v52_data);
              v224_acc += ((static_cast<float>(v226_data[3])) * v54_data);
              v224_acc += ((static_cast<float>(v226_data[4])) * v56_data);
              v224_acc += ((static_cast<float>(v226_data[5])) * v58_data);
              v224_acc += ((static_cast<float>(v226_data[6])) * v60_data);
              v224_acc += ((static_cast<float>(v226_data[7])) * v62_data);
              v224_acc += ((static_cast<float>(v226_data[8])) * v64_data);
              v224_acc += ((static_cast<float>(v226_data[9])) * v66_data);
              v224_acc += ((static_cast<float>(v226_data[10])) * v68_data);
              v224_acc += ((static_cast<float>(v226_data[11])) * v70_data);
              v224_acc += ((static_cast<float>(v226_data[12])) * v72_data);
              v224_acc += ((static_cast<float>(v226_data[13])) * v74_data);
              v224_acc += ((static_cast<float>(v226_data[14])) * v76_data);
              v224_acc += ((static_cast<float>(v226_data[15])) * v78_data);
              tensorforge::intel_esimd::simd<float, 16> v260_data = tensorforge::slmLoad<float, 16>(s1 + (76_i32));
              v224_acc += ((static_cast<float>(v260_data[0])) * v80_data);
              v224_acc += ((static_cast<float>(v260_data[1])) * v82_data);
              v224_acc += ((static_cast<float>(v260_data[2])) * v84_data);
              v224_acc += ((static_cast<float>(v260_data[3])) * v86_data);
              ir0.template select<16, 1>(48) = v224_acc;
              tensorforge::intel_esimd::simd<float, 16> v269_acc{};
              tensorforge::intel_esimd::simd<float, 16> v271_data = tensorforge::slmLoad<float, 16>(s1 + (80_i32));
              v269_acc += ((static_cast<float>(v271_data[0])) * v48_data);
              v269_acc += ((static_cast<float>(v271_data[1])) * v50_data);
              v269_acc += ((static_cast<float>(v271_data[2])) * v52_data);
              v269_acc += ((static_cast<float>(v271_data[3])) * v54_data);
              v269_acc += ((static_cast<float>(v271_data[4])) * v56_data);
              v269_acc += ((static_cast<float>(v271_data[5])) * v58_data);
              v269_acc += ((static_cast<float>(v271_data[6])) * v60_data);
              v269_acc += ((static_cast<float>(v271_data[7])) * v62_data);
              v269_acc += ((static_cast<float>(v271_data[8])) * v64_data);
              v269_acc += ((static_cast<float>(v271_data[9])) * v66_data);
              v269_acc += ((static_cast<float>(v271_data[10])) * v68_data);
              v269_acc += ((static_cast<float>(v271_data[11])) * v70_data);
              v269_acc += ((static_cast<float>(v271_data[12])) * v72_data);
              v269_acc += ((static_cast<float>(v271_data[13])) * v74_data);
              v269_acc += ((static_cast<float>(v271_data[14])) * v76_data);
              v269_acc += ((static_cast<float>(v271_data[15])) * v78_data);
              tensorforge::intel_esimd::simd<float, 16> v305_data = tensorforge::slmLoad<float, 16>(s1 + (96_i32));
              v269_acc += ((static_cast<float>(v305_data[0])) * v80_data);
              v269_acc += ((static_cast<float>(v305_data[1])) * v82_data);
              v269_acc += ((static_cast<float>(v305_data[2])) * v84_data);
              v269_acc += ((static_cast<float>(v305_data[3])) * v86_data);
              ir0.template select<16, 1>(64) = v269_acc;
              tensorforge::intel_esimd::simd<float, 16> v314_acc{};
              tensorforge::intel_esimd::simd<float, 16> v316_data = tensorforge::slmLoad<float, 16>(s1 + (100_i32));
              v314_acc += ((static_cast<float>(v316_data[0])) * v48_data);
              v314_acc += ((static_cast<float>(v316_data[1])) * v50_data);
              v314_acc += ((static_cast<float>(v316_data[2])) * v52_data);
              v314_acc += ((static_cast<float>(v316_data[3])) * v54_data);
              v314_acc += ((static_cast<float>(v316_data[4])) * v56_data);
              v314_acc += ((static_cast<float>(v316_data[5])) * v58_data);
              v314_acc += ((static_cast<float>(v316_data[6])) * v60_data);
              v314_acc += ((static_cast<float>(v316_data[7])) * v62_data);
              v314_acc += ((static_cast<float>(v316_data[8])) * v64_data);
              v314_acc += ((static_cast<float>(v316_data[9])) * v66_data);
              v314_acc += ((static_cast<float>(v316_data[10])) * v68_data);
              v314_acc += ((static_cast<float>(v316_data[11])) * v70_data);
              v314_acc += ((static_cast<float>(v316_data[12])) * v72_data);
              v314_acc += ((static_cast<float>(v316_data[13])) * v74_data);
              v314_acc += ((static_cast<float>(v316_data[14])) * v76_data);
              v314_acc += ((static_cast<float>(v316_data[15])) * v78_data);
              tensorforge::intel_esimd::simd<float, 16> v350_data = tensorforge::slmLoad<float, 16>(s1 + (116_i32));
              v314_acc += ((static_cast<float>(v350_data[0])) * v80_data);
              v314_acc += ((static_cast<float>(v350_data[1])) * v82_data);
              v314_acc += ((static_cast<float>(v350_data[2])) * v84_data);
              v314_acc += ((static_cast<float>(v350_data[3])) * v86_data);
              ir0.template select<16, 1>(80) = v314_acc;
              tensorforge::intel_esimd::simd<float, 16> v359_acc{};
              tensorforge::intel_esimd::simd<float, 16> v361_data = tensorforge::slmLoad<float, 16>(s1 + (120_i32));
              v359_acc += ((static_cast<float>(v361_data[0])) * v48_data);
              v359_acc += ((static_cast<float>(v361_data[1])) * v50_data);
              v359_acc += ((static_cast<float>(v361_data[2])) * v52_data);
              v359_acc += ((static_cast<float>(v361_data[3])) * v54_data);
              v359_acc += ((static_cast<float>(v361_data[4])) * v56_data);
              v359_acc += ((static_cast<float>(v361_data[5])) * v58_data);
              v359_acc += ((static_cast<float>(v361_data[6])) * v60_data);
              v359_acc += ((static_cast<float>(v361_data[7])) * v62_data);
              v359_acc += ((static_cast<float>(v361_data[8])) * v64_data);
              v359_acc += ((static_cast<float>(v361_data[9])) * v66_data);
              v359_acc += ((static_cast<float>(v361_data[10])) * v68_data);
              v359_acc += ((static_cast<float>(v361_data[11])) * v70_data);
              v359_acc += ((static_cast<float>(v361_data[12])) * v72_data);
              v359_acc += ((static_cast<float>(v361_data[13])) * v74_data);
              v359_acc += ((static_cast<float>(v361_data[14])) * v76_data);
              v359_acc += ((static_cast<float>(v361_data[15])) * v78_data);
              tensorforge::intel_esimd::simd<float, 16> v395_data = tensorforge::slmLoad<float, 16>(s1 + (136_i32));
              v359_acc += ((static_cast<float>(v395_data[0])) * v80_data);
              v359_acc += ((static_cast<float>(v395_data[1])) * v82_data);
              v359_acc += ((static_cast<float>(v395_data[2])) * v84_data);
              v359_acc += ((static_cast<float>(v395_data[3])) * v86_data);
              ir0.template select<16, 1>(96) = v359_acc;
              tensorforge::intel_esimd::simd<float, 16> v404_acc{};
              tensorforge::intel_esimd::simd<float, 16> v406_data = tensorforge::slmLoad<float, 16>(s1 + (140_i32));
              v404_acc += ((static_cast<float>(v406_data[0])) * v48_data);
              v404_acc += ((static_cast<float>(v406_data[1])) * v50_data);
              v404_acc += ((static_cast<float>(v406_data[2])) * v52_data);
              v404_acc += ((static_cast<float>(v406_data[3])) * v54_data);
              v404_acc += ((static_cast<float>(v406_data[4])) * v56_data);
              v404_acc += ((static_cast<float>(v406_data[5])) * v58_data);
              v404_acc += ((static_cast<float>(v406_data[6])) * v60_data);
              v404_acc += ((static_cast<float>(v406_data[7])) * v62_data);
              v404_acc += ((static_cast<float>(v406_data[8])) * v64_data);
              v404_acc += ((static_cast<float>(v406_data[9])) * v66_data);
              v404_acc += ((static_cast<float>(v406_data[10])) * v68_data);
              v404_acc += ((static_cast<float>(v406_data[11])) * v70_data);
              v404_acc += ((static_cast<float>(v406_data[12])) * v72_data);
              v404_acc += ((static_cast<float>(v406_data[13])) * v74_data);
              v404_acc += ((static_cast<float>(v406_data[14])) * v76_data);
              v404_acc += ((static_cast<float>(v406_data[15])) * v78_data);
              tensorforge::intel_esimd::simd<float, 16> v440_data = tensorforge::slmLoad<float, 16>(s1 + (156_i32));
              v404_acc += ((static_cast<float>(v440_data[0])) * v80_data);
              v404_acc += ((static_cast<float>(v440_data[1])) * v82_data);
              v404_acc += ((static_cast<float>(v440_data[2])) * v84_data);
              v404_acc += ((static_cast<float>(v440_data[3])) * v86_data);
              ir0.template select<16, 1>(112) = v404_acc;
              tensorforge::intel_esimd::simd<float, 16> v449_acc{};
              tensorforge::intel_esimd::simd<float, 16> v451_data = tensorforge::slmLoad<float, 16>(s1 + (160_i32));
              v449_acc += ((static_cast<float>(v451_data[0])) * v48_data);
              v449_acc += ((static_cast<float>(v451_data[1])) * v50_data);
              v449_acc += ((static_cast<float>(v451_data[2])) * v52_data);
              v449_acc += ((static_cast<float>(v451_data[3])) * v54_data);
              v449_acc += ((static_cast<float>(v451_data[4])) * v56_data);
              v449_acc += ((static_cast<float>(v451_data[5])) * v58_data);
              v449_acc += ((static_cast<float>(v451_data[6])) * v60_data);
              v449_acc += ((static_cast<float>(v451_data[7])) * v62_data);
              v449_acc += ((static_cast<float>(v451_data[8])) * v64_data);
              v449_acc += ((static_cast<float>(v451_data[9])) * v66_data);
              v449_acc += ((static_cast<float>(v451_data[10])) * v68_data);
              v449_acc += ((static_cast<float>(v451_data[11])) * v70_data);
              v449_acc += ((static_cast<float>(v451_data[12])) * v72_data);
              v449_acc += ((static_cast<float>(v451_data[13])) * v74_data);
              v449_acc += ((static_cast<float>(v451_data[14])) * v76_data);
              v449_acc += ((static_cast<float>(v451_data[15])) * v78_data);
              tensorforge::intel_esimd::simd<float, 16> v485_data = tensorforge::slmLoad<float, 16>(s1 + (176_i32));
              v449_acc += ((static_cast<float>(v485_data[0])) * v80_data);
              v449_acc += ((static_cast<float>(v485_data[1])) * v82_data);
              v449_acc += ((static_cast<float>(v485_data[2])) * v84_data);
              v449_acc += ((static_cast<float>(v485_data[3])) * v86_data);
              ir0.template select<16, 1>(128) = v449_acc;
              tensorforge::intel_esimd::simd<float, 16> v494_acc{};
              tensorforge::intel_esimd::simd<float, 16> v496_data = tensorforge::slmLoad<float, 16>(s1 + (180_i32));
              v494_acc += ((static_cast<float>(v496_data[0])) * v48_data);
              v494_acc += ((static_cast<float>(v496_data[1])) * v50_data);
              v494_acc += ((static_cast<float>(v496_data[2])) * v52_data);
              v494_acc += ((static_cast<float>(v496_data[3])) * v54_data);
              v494_acc += ((static_cast<float>(v496_data[4])) * v56_data);
              v494_acc += ((static_cast<float>(v496_data[5])) * v58_data);
              v494_acc += ((static_cast<float>(v496_data[6])) * v60_data);
              v494_acc += ((static_cast<float>(v496_data[7])) * v62_data);
              v494_acc += ((static_cast<float>(v496_data[8])) * v64_data);
              v494_acc += ((static_cast<float>(v496_data[9])) * v66_data);
              v494_acc += ((static_cast<float>(v496_data[10])) * v68_data);
              v494_acc += ((static_cast<float>(v496_data[11])) * v70_data);
              v494_acc += ((static_cast<float>(v496_data[12])) * v72_data);
              v494_acc += ((static_cast<float>(v496_data[13])) * v74_data);
              v494_acc += ((static_cast<float>(v496_data[14])) * v76_data);
              v494_acc += ((static_cast<float>(v496_data[15])) * v78_data);
              tensorforge::intel_esimd::simd<float, 16> v530_data = tensorforge::slmLoad<float, 16>(s1 + (196_i32));
              v494_acc += ((static_cast<float>(v530_data[0])) * v80_data);
              v494_acc += ((static_cast<float>(v530_data[1])) * v82_data);
              v494_acc += ((static_cast<float>(v530_data[2])) * v84_data);
              v494_acc += ((static_cast<float>(v530_data[3])) * v86_data);
              ir0.template select<16, 1>(144) = v494_acc;
              tensorforge::intel_esimd::simd<float, 16> v539_acc{};
              tensorforge::intel_esimd::simd<float, 16> v541_data = tensorforge::slmLoad<float, 16>(s1 + (200_i32));
              v539_acc += ((static_cast<float>(v541_data[0])) * v48_data);
              v539_acc += ((static_cast<float>(v541_data[1])) * v50_data);
              v539_acc += ((static_cast<float>(v541_data[2])) * v52_data);
              v539_acc += ((static_cast<float>(v541_data[3])) * v54_data);
              v539_acc += ((static_cast<float>(v541_data[4])) * v56_data);
              v539_acc += ((static_cast<float>(v541_data[5])) * v58_data);
              v539_acc += ((static_cast<float>(v541_data[6])) * v60_data);
              v539_acc += ((static_cast<float>(v541_data[7])) * v62_data);
              v539_acc += ((static_cast<float>(v541_data[8])) * v64_data);
              v539_acc += ((static_cast<float>(v541_data[9])) * v66_data);
              v539_acc += ((static_cast<float>(v541_data[10])) * v68_data);
              v539_acc += ((static_cast<float>(v541_data[11])) * v70_data);
              v539_acc += ((static_cast<float>(v541_data[12])) * v72_data);
              v539_acc += ((static_cast<float>(v541_data[13])) * v74_data);
              v539_acc += ((static_cast<float>(v541_data[14])) * v76_data);
              v539_acc += ((static_cast<float>(v541_data[15])) * v78_data);
              tensorforge::intel_esimd::simd<float, 16> v575_data = tensorforge::slmLoad<float, 16>(s1 + (216_i32));
              v539_acc += ((static_cast<float>(v575_data[0])) * v80_data);
              v539_acc += ((static_cast<float>(v575_data[1])) * v82_data);
              v539_acc += ((static_cast<float>(v575_data[2])) * v84_data);
              v539_acc += ((static_cast<float>(v575_data[3])) * v86_data);
              ir0.template select<16, 1>(160) = v539_acc;
              tensorforge::intel_esimd::simd<float, 16> v584_acc{};
              tensorforge::intel_esimd::simd<float, 16> v586_data = tensorforge::slmLoad<float, 16>(s1 + (220_i32));
              v584_acc += ((static_cast<float>(v586_data[0])) * v48_data);
              v584_acc += ((static_cast<float>(v586_data[1])) * v50_data);
              v584_acc += ((static_cast<float>(v586_data[2])) * v52_data);
              v584_acc += ((static_cast<float>(v586_data[3])) * v54_data);
              v584_acc += ((static_cast<float>(v586_data[4])) * v56_data);
              v584_acc += ((static_cast<float>(v586_data[5])) * v58_data);
              v584_acc += ((static_cast<float>(v586_data[6])) * v60_data);
              v584_acc += ((static_cast<float>(v586_data[7])) * v62_data);
              v584_acc += ((static_cast<float>(v586_data[8])) * v64_data);
              v584_acc += ((static_cast<float>(v586_data[9])) * v66_data);
              v584_acc += ((static_cast<float>(v586_data[10])) * v68_data);
              v584_acc += ((static_cast<float>(v586_data[11])) * v70_data);
              v584_acc += ((static_cast<float>(v586_data[12])) * v72_data);
              v584_acc += ((static_cast<float>(v586_data[13])) * v74_data);
              v584_acc += ((static_cast<float>(v586_data[14])) * v76_data);
              v584_acc += ((static_cast<float>(v586_data[15])) * v78_data);
              tensorforge::intel_esimd::simd<float, 16> v620_data = tensorforge::slmLoad<float, 16>(s1 + (236_i32));
              v584_acc += ((static_cast<float>(v620_data[0])) * v80_data);
              v584_acc += ((static_cast<float>(v620_data[1])) * v82_data);
              v584_acc += ((static_cast<float>(v620_data[2])) * v84_data);
              v584_acc += ((static_cast<float>(v620_data[3])) * v86_data);
              ir0.template select<16, 1>(176) = v584_acc;
              tensorforge::intel_esimd::simd<float, 16> v629_acc{};
              tensorforge::intel_esimd::simd<float, 16> v631_data = tensorforge::slmLoad<float, 16>(s1 + (240_i32));
              v629_acc += ((static_cast<float>(v631_data[0])) * v48_data);
              v629_acc += ((static_cast<float>(v631_data[1])) * v50_data);
              v629_acc += ((static_cast<float>(v631_data[2])) * v52_data);
              v629_acc += ((static_cast<float>(v631_data[3])) * v54_data);
              v629_acc += ((static_cast<float>(v631_data[4])) * v56_data);
              v629_acc += ((static_cast<float>(v631_data[5])) * v58_data);
              v629_acc += ((static_cast<float>(v631_data[6])) * v60_data);
              v629_acc += ((static_cast<float>(v631_data[7])) * v62_data);
              v629_acc += ((static_cast<float>(v631_data[8])) * v64_data);
              v629_acc += ((static_cast<float>(v631_data[9])) * v66_data);
              v629_acc += ((static_cast<float>(v631_data[10])) * v68_data);
              v629_acc += ((static_cast<float>(v631_data[11])) * v70_data);
              v629_acc += ((static_cast<float>(v631_data[12])) * v72_data);
              v629_acc += ((static_cast<float>(v631_data[13])) * v74_data);
              v629_acc += ((static_cast<float>(v631_data[14])) * v76_data);
              v629_acc += ((static_cast<float>(v631_data[15])) * v78_data);
              tensorforge::intel_esimd::simd<float, 16> v665_data = tensorforge::slmLoad<float, 16>(s1 + (256_i32));
              v629_acc += ((static_cast<float>(v665_data[0])) * v80_data);
              v629_acc += ((static_cast<float>(v665_data[1])) * v82_data);
              v629_acc += ((static_cast<float>(v665_data[2])) * v84_data);
              v629_acc += ((static_cast<float>(v665_data[3])) * v86_data);
              ir0.template select<16, 1>(192) = v629_acc;
              tensorforge::intel_esimd::simd<float, 16> v674_acc{};
              tensorforge::intel_esimd::simd<float, 16> v676_data = tensorforge::slmLoad<float, 16>(s1 + (260_i32));
              v674_acc += ((static_cast<float>(v676_data[0])) * v48_data);
              v674_acc += ((static_cast<float>(v676_data[1])) * v50_data);
              v674_acc += ((static_cast<float>(v676_data[2])) * v52_data);
              v674_acc += ((static_cast<float>(v676_data[3])) * v54_data);
              v674_acc += ((static_cast<float>(v676_data[4])) * v56_data);
              v674_acc += ((static_cast<float>(v676_data[5])) * v58_data);
              v674_acc += ((static_cast<float>(v676_data[6])) * v60_data);
              v674_acc += ((static_cast<float>(v676_data[7])) * v62_data);
              v674_acc += ((static_cast<float>(v676_data[8])) * v64_data);
              v674_acc += ((static_cast<float>(v676_data[9])) * v66_data);
              v674_acc += ((static_cast<float>(v676_data[10])) * v68_data);
              v674_acc += ((static_cast<float>(v676_data[11])) * v70_data);
              v674_acc += ((static_cast<float>(v676_data[12])) * v72_data);
              v674_acc += ((static_cast<float>(v676_data[13])) * v74_data);
              v674_acc += ((static_cast<float>(v676_data[14])) * v76_data);
              v674_acc += ((static_cast<float>(v676_data[15])) * v78_data);
              tensorforge::intel_esimd::simd<float, 16> v710_data = tensorforge::slmLoad<float, 16>(s1 + (276_i32));
              v674_acc += ((static_cast<float>(v710_data[0])) * v80_data);
              v674_acc += ((static_cast<float>(v710_data[1])) * v82_data);
              v674_acc += ((static_cast<float>(v710_data[2])) * v84_data);
              v674_acc += ((static_cast<float>(v710_data[3])) * v86_data);
              ir0.template select<16, 1>(208) = v674_acc;
              tensorforge::intel_esimd::simd<float, 16> v719_acc{};
              tensorforge::intel_esimd::simd<float, 16> v721_data = tensorforge::slmLoad<float, 16>(s1 + (280_i32));
              v719_acc += ((static_cast<float>(v721_data[0])) * v48_data);
              v719_acc += ((static_cast<float>(v721_data[1])) * v50_data);
              v719_acc += ((static_cast<float>(v721_data[2])) * v52_data);
              v719_acc += ((static_cast<float>(v721_data[3])) * v54_data);
              v719_acc += ((static_cast<float>(v721_data[4])) * v56_data);
              v719_acc += ((static_cast<float>(v721_data[5])) * v58_data);
              v719_acc += ((static_cast<float>(v721_data[6])) * v60_data);
              v719_acc += ((static_cast<float>(v721_data[7])) * v62_data);
              v719_acc += ((static_cast<float>(v721_data[8])) * v64_data);
              v719_acc += ((static_cast<float>(v721_data[9])) * v66_data);
              v719_acc += ((static_cast<float>(v721_data[10])) * v68_data);
              v719_acc += ((static_cast<float>(v721_data[11])) * v70_data);
              v719_acc += ((static_cast<float>(v721_data[12])) * v72_data);
              v719_acc += ((static_cast<float>(v721_data[13])) * v74_data);
              v719_acc += ((static_cast<float>(v721_data[14])) * v76_data);
              v719_acc += ((static_cast<float>(v721_data[15])) * v78_data);
              tensorforge::intel_esimd::simd<float, 16> v755_data = tensorforge::slmLoad<float, 16>(s1 + (296_i32));
              v719_acc += ((static_cast<float>(v755_data[0])) * v80_data);
              v719_acc += ((static_cast<float>(v755_data[1])) * v82_data);
              v719_acc += ((static_cast<float>(v755_data[2])) * v84_data);
              v719_acc += ((static_cast<float>(v755_data[3])) * v86_data);
              ir0.template select<16, 1>(224) = v719_acc;
              tensorforge::intel_esimd::simd<float, 16> v764_acc{};
              tensorforge::intel_esimd::simd<float, 16> v766_data = tensorforge::slmLoad<float, 16>(s1 + (300_i32));
              v764_acc += ((static_cast<float>(v766_data[0])) * v48_data);
              v764_acc += ((static_cast<float>(v766_data[1])) * v50_data);
              v764_acc += ((static_cast<float>(v766_data[2])) * v52_data);
              v764_acc += ((static_cast<float>(v766_data[3])) * v54_data);
              v764_acc += ((static_cast<float>(v766_data[4])) * v56_data);
              v764_acc += ((static_cast<float>(v766_data[5])) * v58_data);
              v764_acc += ((static_cast<float>(v766_data[6])) * v60_data);
              v764_acc += ((static_cast<float>(v766_data[7])) * v62_data);
              v764_acc += ((static_cast<float>(v766_data[8])) * v64_data);
              v764_acc += ((static_cast<float>(v766_data[9])) * v66_data);
              v764_acc += ((static_cast<float>(v766_data[10])) * v68_data);
              v764_acc += ((static_cast<float>(v766_data[11])) * v70_data);
              v764_acc += ((static_cast<float>(v766_data[12])) * v72_data);
              v764_acc += ((static_cast<float>(v766_data[13])) * v74_data);
              v764_acc += ((static_cast<float>(v766_data[14])) * v76_data);
              v764_acc += ((static_cast<float>(v766_data[15])) * v78_data);
              tensorforge::intel_esimd::simd<float, 16> v800_data = tensorforge::slmLoad<float, 16>(s1 + (316_i32));
              v764_acc += ((static_cast<float>(v800_data[0])) * v80_data);
              v764_acc += ((static_cast<float>(v800_data[1])) * v82_data);
              v764_acc += ((static_cast<float>(v800_data[2])) * v84_data);
              v764_acc += ((static_cast<float>(v800_data[3])) * v86_data);
              ir0.template select<16, 1>(240) = v764_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v809_n1 = 0; v809_n1 < 16; ++v809_n1) {
                int32_t v810_a = v809_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v812_data(ir0.template select<12, 1>(v810_a));
                r0.template select<12, 1>(v810_a) = v812_data;
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v813_i1 = 0; v813_i1 < 16; ++v813_i1) {
                tensorforge::intel_esimd::simd<float, 12> v816_data(r0.template select<12, 1>((v813_i1 * 16)));
                v816_data.copy_to(glb_m0 + ((v813_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}

