// === base name ===
kernel_299f1fc1c7d92130

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_299f1fc1c7d92130 = {{1, 16, 1}, 16, 12, 1, 16, 37888, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_299f1fc1c7d92130(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_299f1fc1c7d92130(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_299f1fc1c7d92130(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_299f1fc1c7d92130(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_299f1fc1c7d92130(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_299f1fc1c7d92130(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_299f1fc1c7d92130(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<9472 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 37888 B shared, occupancy grid
        // operands:
        //   m0 12×16(12×16) {0..12}×{0..16} strided
        //   m1 20×12(20×12) {0..20}×{0..12} strided
        //   m2 20×16(20×16) {0..20}×{0..16} strided
        // operations:
        //   m0[i,j] = m1[k,i] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":9472}],"shared_bytes":37888,"shared_elements":9472,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,16]],"name":"m0","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[20,12]],"name":"m1","ordered":false,"parts":1,"shape":[20,12],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[20,16]],"name":"m2","ordered":false,"parts":1,"shape":[20,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[20,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[20,12]},{"addressing":"strided","bbox":[[0,0],[20,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,16]}],"permute":[[1,0],[0,1]],"target":[[-1,0],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (592 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (576);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (320);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v6_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v6_batchId0 < numElements0; v6_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v7_ahead1 = v6_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 192 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v6_batchId0 * 240 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v6_batchId0 * 320 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m1[1, 0])
              #pragma unroll
              for (int32_t v17_i0 = 0; v17_i0 < 1; ++v17_i0) {
                int32_t v19_lead = v17_i0 * 16;
                #pragma unroll
                for (int32_t v18_i1 = 0; v18_i1 < 12; ++v18_i1) {
                  tensorforge::intel_esimd::simd<float, 16> v23_data;
                  v23_data.copy_from(glb_m1 + ((v19_lead + (v18_i1 * 20))));
                  tensorforge::slmStore<float, 16>(s0 + ((v19_lead + (v18_i1 * 21))), v23_data);
                }
              }
              #pragma unroll
              for (int32_t v26_i1 = 0; v26_i1 < 12; ++v26_i1) {
                tensorforge::intel_esimd::simd<float, 4> v32_data;
                v32_data.copy_from(glb_m1 + ((16_i32 + (v26_i1 * 20))));
                tensorforge::slmStore<float, 4>(s0 + ((16_i32 + (v26_i1 * 21))), v32_data);
              }
              // s1 = load{g>s}(glb_m2[0, 1])
              #pragma unroll
              for (int32_t i = 0; i < 20; i += 4) {
                tensorforge::intel_esimd::simd<float, 64> v35_ld;
                v35_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + i * 16));
                tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 4 * 0 + i * 16), v35_ld);
              }
              // wait(s0 = load{g>s}(glb_m1[1, 0]));
              // wait(s1 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
              // ir0 = +(s0 * s1)
              // [(0, 12), (0, 16)] [(0, 20)]
              tensorforge::intel_esimd::simd<float, 256> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v42_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v44_data = tensorforge::slmLoad<float, 16>(s0 + (1_i32));
              tensorforge::intel_esimd::simd<float, 16> v46_data = tensorforge::slmLoad<float, 16>(s0 + (2_i32));
              tensorforge::intel_esimd::simd<float, 16> v48_data = tensorforge::slmLoad<float, 16>(s0 + (3_i32));
              tensorforge::intel_esimd::simd<float, 16> v50_data = tensorforge::slmLoad<float, 16>(s0 + (4_i32));
              tensorforge::intel_esimd::simd<float, 16> v52_data = tensorforge::slmLoad<float, 16>(s0 + (5_i32));
              tensorforge::intel_esimd::simd<float, 16> v54_data = tensorforge::slmLoad<float, 16>(s0 + (6_i32));
              tensorforge::intel_esimd::simd<float, 16> v56_data = tensorforge::slmLoad<float, 16>(s0 + (7_i32));
              tensorforge::intel_esimd::simd<float, 16> v58_data = tensorforge::slmLoad<float, 16>(s0 + (8_i32));
              tensorforge::intel_esimd::simd<float, 16> v60_data = tensorforge::slmLoad<float, 16>(s0 + (9_i32));
              tensorforge::intel_esimd::simd<float, 16> v62_data = tensorforge::slmLoad<float, 16>(s0 + (10_i32));
              tensorforge::intel_esimd::simd<float, 16> v64_data = tensorforge::slmLoad<float, 16>(s0 + (11_i32));
              tensorforge::intel_esimd::simd<float, 16> v66_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              tensorforge::intel_esimd::simd<float, 16> v68_data = tensorforge::slmLoad<float, 16>(s0 + (13_i32));
              tensorforge::intel_esimd::simd<float, 16> v70_data = tensorforge::slmLoad<float, 16>(s0 + (14_i32));
              tensorforge::intel_esimd::simd<float, 16> v72_data = tensorforge::slmLoad<float, 16>(s0 + (15_i32));
              tensorforge::intel_esimd::simd<float, 16> v74_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              tensorforge::intel_esimd::simd<float, 16> v76_data = tensorforge::slmLoad<float, 16>(s0 + (17_i32));
              tensorforge::intel_esimd::simd<float, 16> v78_data = tensorforge::slmLoad<float, 16>(s0 + (18_i32));
              tensorforge::intel_esimd::simd<float, 16> v80_data = tensorforge::slmLoad<float, 16>(s0 + (19_i32));
              tensorforge::intel_esimd::simd<float, 16> v81_acc{};
              tensorforge::intel_esimd::simd<float, 16> v83_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v81_acc += ((static_cast<float>(v83_data[0])) * v42_data);
              v81_acc += ((static_cast<float>(v83_data[1])) * v44_data);
              v81_acc += ((static_cast<float>(v83_data[2])) * v46_data);
              v81_acc += ((static_cast<float>(v83_data[3])) * v48_data);
              v81_acc += ((static_cast<float>(v83_data[4])) * v50_data);
              v81_acc += ((static_cast<float>(v83_data[5])) * v52_data);
              v81_acc += ((static_cast<float>(v83_data[6])) * v54_data);
              v81_acc += ((static_cast<float>(v83_data[7])) * v56_data);
              v81_acc += ((static_cast<float>(v83_data[8])) * v58_data);
              v81_acc += ((static_cast<float>(v83_data[9])) * v60_data);
              v81_acc += ((static_cast<float>(v83_data[10])) * v62_data);
              v81_acc += ((static_cast<float>(v83_data[11])) * v64_data);
              v81_acc += ((static_cast<float>(v83_data[12])) * v66_data);
              v81_acc += ((static_cast<float>(v83_data[13])) * v68_data);
              v81_acc += ((static_cast<float>(v83_data[14])) * v70_data);
              v81_acc += ((static_cast<float>(v83_data[15])) * v72_data);
              tensorforge::intel_esimd::simd<float, 16> v119_data = tensorforge::slmLoad<float, 16>(s1 + (16_i32));
              v81_acc += ((static_cast<float>(v119_data[0])) * v74_data);
              v81_acc += ((static_cast<float>(v119_data[1])) * v76_data);
              v81_acc += ((static_cast<float>(v119_data[2])) * v78_data);
              v81_acc += ((static_cast<float>(v119_data[3])) * v80_data);
              ir0.template select<16, 1>(0) = v81_acc;
              tensorforge::intel_esimd::simd<float, 16> v128_acc{};
              tensorforge::intel_esimd::simd<float, 16> v130_data = tensorforge::slmLoad<float, 16>(s1 + (20_i32));
              v128_acc += ((static_cast<float>(v130_data[0])) * v42_data);
              v128_acc += ((static_cast<float>(v130_data[1])) * v44_data);
              v128_acc += ((static_cast<float>(v130_data[2])) * v46_data);
              v128_acc += ((static_cast<float>(v130_data[3])) * v48_data);
              v128_acc += ((static_cast<float>(v130_data[4])) * v50_data);
              v128_acc += ((static_cast<float>(v130_data[5])) * v52_data);
              v128_acc += ((static_cast<float>(v130_data[6])) * v54_data);
              v128_acc += ((static_cast<float>(v130_data[7])) * v56_data);
              v128_acc += ((static_cast<float>(v130_data[8])) * v58_data);
              v128_acc += ((static_cast<float>(v130_data[9])) * v60_data);
              v128_acc += ((static_cast<float>(v130_data[10])) * v62_data);
              v128_acc += ((static_cast<float>(v130_data[11])) * v64_data);
              v128_acc += ((static_cast<float>(v130_data[12])) * v66_data);
              v128_acc += ((static_cast<float>(v130_data[13])) * v68_data);
              v128_acc += ((static_cast<float>(v130_data[14])) * v70_data);
              v128_acc += ((static_cast<float>(v130_data[15])) * v72_data);
              tensorforge::intel_esimd::simd<float, 16> v164_data = tensorforge::slmLoad<float, 16>(s1 + (36_i32));
              v128_acc += ((static_cast<float>(v164_data[0])) * v74_data);
              v128_acc += ((static_cast<float>(v164_data[1])) * v76_data);
              v128_acc += ((static_cast<float>(v164_data[2])) * v78_data);
              v128_acc += ((static_cast<float>(v164_data[3])) * v80_data);
              ir0.template select<16, 1>(16) = v128_acc;
              tensorforge::intel_esimd::simd<float, 16> v173_acc{};
              tensorforge::intel_esimd::simd<float, 16> v175_data = tensorforge::slmLoad<float, 16>(s1 + (40_i32));
              v173_acc += ((static_cast<float>(v175_data[0])) * v42_data);
              v173_acc += ((static_cast<float>(v175_data[1])) * v44_data);
              v173_acc += ((static_cast<float>(v175_data[2])) * v46_data);
              v173_acc += ((static_cast<float>(v175_data[3])) * v48_data);
              v173_acc += ((static_cast<float>(v175_data[4])) * v50_data);
              v173_acc += ((static_cast<float>(v175_data[5])) * v52_data);
              v173_acc += ((static_cast<float>(v175_data[6])) * v54_data);
              v173_acc += ((static_cast<float>(v175_data[7])) * v56_data);
              v173_acc += ((static_cast<float>(v175_data[8])) * v58_data);
              v173_acc += ((static_cast<float>(v175_data[9])) * v60_data);
              v173_acc += ((static_cast<float>(v175_data[10])) * v62_data);
              v173_acc += ((static_cast<float>(v175_data[11])) * v64_data);
              v173_acc += ((static_cast<float>(v175_data[12])) * v66_data);
              v173_acc += ((static_cast<float>(v175_data[13])) * v68_data);
              v173_acc += ((static_cast<float>(v175_data[14])) * v70_data);
              v173_acc += ((static_cast<float>(v175_data[15])) * v72_data);
              tensorforge::intel_esimd::simd<float, 16> v209_data = tensorforge::slmLoad<float, 16>(s1 + (56_i32));
              v173_acc += ((static_cast<float>(v209_data[0])) * v74_data);
              v173_acc += ((static_cast<float>(v209_data[1])) * v76_data);
              v173_acc += ((static_cast<float>(v209_data[2])) * v78_data);
              v173_acc += ((static_cast<float>(v209_data[3])) * v80_data);
              ir0.template select<16, 1>(32) = v173_acc;
              tensorforge::intel_esimd::simd<float, 16> v218_acc{};
              tensorforge::intel_esimd::simd<float, 16> v220_data = tensorforge::slmLoad<float, 16>(s1 + (60_i32));
              v218_acc += ((static_cast<float>(v220_data[0])) * v42_data);
              v218_acc += ((static_cast<float>(v220_data[1])) * v44_data);
              v218_acc += ((static_cast<float>(v220_data[2])) * v46_data);
              v218_acc += ((static_cast<float>(v220_data[3])) * v48_data);
              v218_acc += ((static_cast<float>(v220_data[4])) * v50_data);
              v218_acc += ((static_cast<float>(v220_data[5])) * v52_data);
              v218_acc += ((static_cast<float>(v220_data[6])) * v54_data);
              v218_acc += ((static_cast<float>(v220_data[7])) * v56_data);
              v218_acc += ((static_cast<float>(v220_data[8])) * v58_data);
              v218_acc += ((static_cast<float>(v220_data[9])) * v60_data);
              v218_acc += ((static_cast<float>(v220_data[10])) * v62_data);
              v218_acc += ((static_cast<float>(v220_data[11])) * v64_data);
              v218_acc += ((static_cast<float>(v220_data[12])) * v66_data);
              v218_acc += ((static_cast<float>(v220_data[13])) * v68_data);
              v218_acc += ((static_cast<float>(v220_data[14])) * v70_data);
              v218_acc += ((static_cast<float>(v220_data[15])) * v72_data);
              tensorforge::intel_esimd::simd<float, 16> v254_data = tensorforge::slmLoad<float, 16>(s1 + (76_i32));
              v218_acc += ((static_cast<float>(v254_data[0])) * v74_data);
              v218_acc += ((static_cast<float>(v254_data[1])) * v76_data);
              v218_acc += ((static_cast<float>(v254_data[2])) * v78_data);
              v218_acc += ((static_cast<float>(v254_data[3])) * v80_data);
              ir0.template select<16, 1>(48) = v218_acc;
              tensorforge::intel_esimd::simd<float, 16> v263_acc{};
              tensorforge::intel_esimd::simd<float, 16> v265_data = tensorforge::slmLoad<float, 16>(s1 + (80_i32));
              v263_acc += ((static_cast<float>(v265_data[0])) * v42_data);
              v263_acc += ((static_cast<float>(v265_data[1])) * v44_data);
              v263_acc += ((static_cast<float>(v265_data[2])) * v46_data);
              v263_acc += ((static_cast<float>(v265_data[3])) * v48_data);
              v263_acc += ((static_cast<float>(v265_data[4])) * v50_data);
              v263_acc += ((static_cast<float>(v265_data[5])) * v52_data);
              v263_acc += ((static_cast<float>(v265_data[6])) * v54_data);
              v263_acc += ((static_cast<float>(v265_data[7])) * v56_data);
              v263_acc += ((static_cast<float>(v265_data[8])) * v58_data);
              v263_acc += ((static_cast<float>(v265_data[9])) * v60_data);
              v263_acc += ((static_cast<float>(v265_data[10])) * v62_data);
              v263_acc += ((static_cast<float>(v265_data[11])) * v64_data);
              v263_acc += ((static_cast<float>(v265_data[12])) * v66_data);
              v263_acc += ((static_cast<float>(v265_data[13])) * v68_data);
              v263_acc += ((static_cast<float>(v265_data[14])) * v70_data);
              v263_acc += ((static_cast<float>(v265_data[15])) * v72_data);
              tensorforge::intel_esimd::simd<float, 16> v299_data = tensorforge::slmLoad<float, 16>(s1 + (96_i32));
              v263_acc += ((static_cast<float>(v299_data[0])) * v74_data);
              v263_acc += ((static_cast<float>(v299_data[1])) * v76_data);
              v263_acc += ((static_cast<float>(v299_data[2])) * v78_data);
              v263_acc += ((static_cast<float>(v299_data[3])) * v80_data);
              ir0.template select<16, 1>(64) = v263_acc;
              tensorforge::intel_esimd::simd<float, 16> v308_acc{};
              tensorforge::intel_esimd::simd<float, 16> v310_data = tensorforge::slmLoad<float, 16>(s1 + (100_i32));
              v308_acc += ((static_cast<float>(v310_data[0])) * v42_data);
              v308_acc += ((static_cast<float>(v310_data[1])) * v44_data);
              v308_acc += ((static_cast<float>(v310_data[2])) * v46_data);
              v308_acc += ((static_cast<float>(v310_data[3])) * v48_data);
              v308_acc += ((static_cast<float>(v310_data[4])) * v50_data);
              v308_acc += ((static_cast<float>(v310_data[5])) * v52_data);
              v308_acc += ((static_cast<float>(v310_data[6])) * v54_data);
              v308_acc += ((static_cast<float>(v310_data[7])) * v56_data);
              v308_acc += ((static_cast<float>(v310_data[8])) * v58_data);
              v308_acc += ((static_cast<float>(v310_data[9])) * v60_data);
              v308_acc += ((static_cast<float>(v310_data[10])) * v62_data);
              v308_acc += ((static_cast<float>(v310_data[11])) * v64_data);
              v308_acc += ((static_cast<float>(v310_data[12])) * v66_data);
              v308_acc += ((static_cast<float>(v310_data[13])) * v68_data);
              v308_acc += ((static_cast<float>(v310_data[14])) * v70_data);
              v308_acc += ((static_cast<float>(v310_data[15])) * v72_data);
              tensorforge::intel_esimd::simd<float, 16> v344_data = tensorforge::slmLoad<float, 16>(s1 + (116_i32));
              v308_acc += ((static_cast<float>(v344_data[0])) * v74_data);
              v308_acc += ((static_cast<float>(v344_data[1])) * v76_data);
              v308_acc += ((static_cast<float>(v344_data[2])) * v78_data);
              v308_acc += ((static_cast<float>(v344_data[3])) * v80_data);
              ir0.template select<16, 1>(80) = v308_acc;
              tensorforge::intel_esimd::simd<float, 16> v353_acc{};
              tensorforge::intel_esimd::simd<float, 16> v355_data = tensorforge::slmLoad<float, 16>(s1 + (120_i32));
              v353_acc += ((static_cast<float>(v355_data[0])) * v42_data);
              v353_acc += ((static_cast<float>(v355_data[1])) * v44_data);
              v353_acc += ((static_cast<float>(v355_data[2])) * v46_data);
              v353_acc += ((static_cast<float>(v355_data[3])) * v48_data);
              v353_acc += ((static_cast<float>(v355_data[4])) * v50_data);
              v353_acc += ((static_cast<float>(v355_data[5])) * v52_data);
              v353_acc += ((static_cast<float>(v355_data[6])) * v54_data);
              v353_acc += ((static_cast<float>(v355_data[7])) * v56_data);
              v353_acc += ((static_cast<float>(v355_data[8])) * v58_data);
              v353_acc += ((static_cast<float>(v355_data[9])) * v60_data);
              v353_acc += ((static_cast<float>(v355_data[10])) * v62_data);
              v353_acc += ((static_cast<float>(v355_data[11])) * v64_data);
              v353_acc += ((static_cast<float>(v355_data[12])) * v66_data);
              v353_acc += ((static_cast<float>(v355_data[13])) * v68_data);
              v353_acc += ((static_cast<float>(v355_data[14])) * v70_data);
              v353_acc += ((static_cast<float>(v355_data[15])) * v72_data);
              tensorforge::intel_esimd::simd<float, 16> v389_data = tensorforge::slmLoad<float, 16>(s1 + (136_i32));
              v353_acc += ((static_cast<float>(v389_data[0])) * v74_data);
              v353_acc += ((static_cast<float>(v389_data[1])) * v76_data);
              v353_acc += ((static_cast<float>(v389_data[2])) * v78_data);
              v353_acc += ((static_cast<float>(v389_data[3])) * v80_data);
              ir0.template select<16, 1>(96) = v353_acc;
              tensorforge::intel_esimd::simd<float, 16> v398_acc{};
              tensorforge::intel_esimd::simd<float, 16> v400_data = tensorforge::slmLoad<float, 16>(s1 + (140_i32));
              v398_acc += ((static_cast<float>(v400_data[0])) * v42_data);
              v398_acc += ((static_cast<float>(v400_data[1])) * v44_data);
              v398_acc += ((static_cast<float>(v400_data[2])) * v46_data);
              v398_acc += ((static_cast<float>(v400_data[3])) * v48_data);
              v398_acc += ((static_cast<float>(v400_data[4])) * v50_data);
              v398_acc += ((static_cast<float>(v400_data[5])) * v52_data);
              v398_acc += ((static_cast<float>(v400_data[6])) * v54_data);
              v398_acc += ((static_cast<float>(v400_data[7])) * v56_data);
              v398_acc += ((static_cast<float>(v400_data[8])) * v58_data);
              v398_acc += ((static_cast<float>(v400_data[9])) * v60_data);
              v398_acc += ((static_cast<float>(v400_data[10])) * v62_data);
              v398_acc += ((static_cast<float>(v400_data[11])) * v64_data);
              v398_acc += ((static_cast<float>(v400_data[12])) * v66_data);
              v398_acc += ((static_cast<float>(v400_data[13])) * v68_data);
              v398_acc += ((static_cast<float>(v400_data[14])) * v70_data);
              v398_acc += ((static_cast<float>(v400_data[15])) * v72_data);
              tensorforge::intel_esimd::simd<float, 16> v434_data = tensorforge::slmLoad<float, 16>(s1 + (156_i32));
              v398_acc += ((static_cast<float>(v434_data[0])) * v74_data);
              v398_acc += ((static_cast<float>(v434_data[1])) * v76_data);
              v398_acc += ((static_cast<float>(v434_data[2])) * v78_data);
              v398_acc += ((static_cast<float>(v434_data[3])) * v80_data);
              ir0.template select<16, 1>(112) = v398_acc;
              tensorforge::intel_esimd::simd<float, 16> v443_acc{};
              tensorforge::intel_esimd::simd<float, 16> v445_data = tensorforge::slmLoad<float, 16>(s1 + (160_i32));
              v443_acc += ((static_cast<float>(v445_data[0])) * v42_data);
              v443_acc += ((static_cast<float>(v445_data[1])) * v44_data);
              v443_acc += ((static_cast<float>(v445_data[2])) * v46_data);
              v443_acc += ((static_cast<float>(v445_data[3])) * v48_data);
              v443_acc += ((static_cast<float>(v445_data[4])) * v50_data);
              v443_acc += ((static_cast<float>(v445_data[5])) * v52_data);
              v443_acc += ((static_cast<float>(v445_data[6])) * v54_data);
              v443_acc += ((static_cast<float>(v445_data[7])) * v56_data);
              v443_acc += ((static_cast<float>(v445_data[8])) * v58_data);
              v443_acc += ((static_cast<float>(v445_data[9])) * v60_data);
              v443_acc += ((static_cast<float>(v445_data[10])) * v62_data);
              v443_acc += ((static_cast<float>(v445_data[11])) * v64_data);
              v443_acc += ((static_cast<float>(v445_data[12])) * v66_data);
              v443_acc += ((static_cast<float>(v445_data[13])) * v68_data);
              v443_acc += ((static_cast<float>(v445_data[14])) * v70_data);
              v443_acc += ((static_cast<float>(v445_data[15])) * v72_data);
              tensorforge::intel_esimd::simd<float, 16> v479_data = tensorforge::slmLoad<float, 16>(s1 + (176_i32));
              v443_acc += ((static_cast<float>(v479_data[0])) * v74_data);
              v443_acc += ((static_cast<float>(v479_data[1])) * v76_data);
              v443_acc += ((static_cast<float>(v479_data[2])) * v78_data);
              v443_acc += ((static_cast<float>(v479_data[3])) * v80_data);
              ir0.template select<16, 1>(128) = v443_acc;
              tensorforge::intel_esimd::simd<float, 16> v488_acc{};
              tensorforge::intel_esimd::simd<float, 16> v490_data = tensorforge::slmLoad<float, 16>(s1 + (180_i32));
              v488_acc += ((static_cast<float>(v490_data[0])) * v42_data);
              v488_acc += ((static_cast<float>(v490_data[1])) * v44_data);
              v488_acc += ((static_cast<float>(v490_data[2])) * v46_data);
              v488_acc += ((static_cast<float>(v490_data[3])) * v48_data);
              v488_acc += ((static_cast<float>(v490_data[4])) * v50_data);
              v488_acc += ((static_cast<float>(v490_data[5])) * v52_data);
              v488_acc += ((static_cast<float>(v490_data[6])) * v54_data);
              v488_acc += ((static_cast<float>(v490_data[7])) * v56_data);
              v488_acc += ((static_cast<float>(v490_data[8])) * v58_data);
              v488_acc += ((static_cast<float>(v490_data[9])) * v60_data);
              v488_acc += ((static_cast<float>(v490_data[10])) * v62_data);
              v488_acc += ((static_cast<float>(v490_data[11])) * v64_data);
              v488_acc += ((static_cast<float>(v490_data[12])) * v66_data);
              v488_acc += ((static_cast<float>(v490_data[13])) * v68_data);
              v488_acc += ((static_cast<float>(v490_data[14])) * v70_data);
              v488_acc += ((static_cast<float>(v490_data[15])) * v72_data);
              tensorforge::intel_esimd::simd<float, 16> v524_data = tensorforge::slmLoad<float, 16>(s1 + (196_i32));
              v488_acc += ((static_cast<float>(v524_data[0])) * v74_data);
              v488_acc += ((static_cast<float>(v524_data[1])) * v76_data);
              v488_acc += ((static_cast<float>(v524_data[2])) * v78_data);
              v488_acc += ((static_cast<float>(v524_data[3])) * v80_data);
              ir0.template select<16, 1>(144) = v488_acc;
              tensorforge::intel_esimd::simd<float, 16> v533_acc{};
              tensorforge::intel_esimd::simd<float, 16> v535_data = tensorforge::slmLoad<float, 16>(s1 + (200_i32));
              v533_acc += ((static_cast<float>(v535_data[0])) * v42_data);
              v533_acc += ((static_cast<float>(v535_data[1])) * v44_data);
              v533_acc += ((static_cast<float>(v535_data[2])) * v46_data);
              v533_acc += ((static_cast<float>(v535_data[3])) * v48_data);
              v533_acc += ((static_cast<float>(v535_data[4])) * v50_data);
              v533_acc += ((static_cast<float>(v535_data[5])) * v52_data);
              v533_acc += ((static_cast<float>(v535_data[6])) * v54_data);
              v533_acc += ((static_cast<float>(v535_data[7])) * v56_data);
              v533_acc += ((static_cast<float>(v535_data[8])) * v58_data);
              v533_acc += ((static_cast<float>(v535_data[9])) * v60_data);
              v533_acc += ((static_cast<float>(v535_data[10])) * v62_data);
              v533_acc += ((static_cast<float>(v535_data[11])) * v64_data);
              v533_acc += ((static_cast<float>(v535_data[12])) * v66_data);
              v533_acc += ((static_cast<float>(v535_data[13])) * v68_data);
              v533_acc += ((static_cast<float>(v535_data[14])) * v70_data);
              v533_acc += ((static_cast<float>(v535_data[15])) * v72_data);
              tensorforge::intel_esimd::simd<float, 16> v569_data = tensorforge::slmLoad<float, 16>(s1 + (216_i32));
              v533_acc += ((static_cast<float>(v569_data[0])) * v74_data);
              v533_acc += ((static_cast<float>(v569_data[1])) * v76_data);
              v533_acc += ((static_cast<float>(v569_data[2])) * v78_data);
              v533_acc += ((static_cast<float>(v569_data[3])) * v80_data);
              ir0.template select<16, 1>(160) = v533_acc;
              tensorforge::intel_esimd::simd<float, 16> v578_acc{};
              tensorforge::intel_esimd::simd<float, 16> v580_data = tensorforge::slmLoad<float, 16>(s1 + (220_i32));
              v578_acc += ((static_cast<float>(v580_data[0])) * v42_data);
              v578_acc += ((static_cast<float>(v580_data[1])) * v44_data);
              v578_acc += ((static_cast<float>(v580_data[2])) * v46_data);
              v578_acc += ((static_cast<float>(v580_data[3])) * v48_data);
              v578_acc += ((static_cast<float>(v580_data[4])) * v50_data);
              v578_acc += ((static_cast<float>(v580_data[5])) * v52_data);
              v578_acc += ((static_cast<float>(v580_data[6])) * v54_data);
              v578_acc += ((static_cast<float>(v580_data[7])) * v56_data);
              v578_acc += ((static_cast<float>(v580_data[8])) * v58_data);
              v578_acc += ((static_cast<float>(v580_data[9])) * v60_data);
              v578_acc += ((static_cast<float>(v580_data[10])) * v62_data);
              v578_acc += ((static_cast<float>(v580_data[11])) * v64_data);
              v578_acc += ((static_cast<float>(v580_data[12])) * v66_data);
              v578_acc += ((static_cast<float>(v580_data[13])) * v68_data);
              v578_acc += ((static_cast<float>(v580_data[14])) * v70_data);
              v578_acc += ((static_cast<float>(v580_data[15])) * v72_data);
              tensorforge::intel_esimd::simd<float, 16> v614_data = tensorforge::slmLoad<float, 16>(s1 + (236_i32));
              v578_acc += ((static_cast<float>(v614_data[0])) * v74_data);
              v578_acc += ((static_cast<float>(v614_data[1])) * v76_data);
              v578_acc += ((static_cast<float>(v614_data[2])) * v78_data);
              v578_acc += ((static_cast<float>(v614_data[3])) * v80_data);
              ir0.template select<16, 1>(176) = v578_acc;
              tensorforge::intel_esimd::simd<float, 16> v623_acc{};
              tensorforge::intel_esimd::simd<float, 16> v625_data = tensorforge::slmLoad<float, 16>(s1 + (240_i32));
              v623_acc += ((static_cast<float>(v625_data[0])) * v42_data);
              v623_acc += ((static_cast<float>(v625_data[1])) * v44_data);
              v623_acc += ((static_cast<float>(v625_data[2])) * v46_data);
              v623_acc += ((static_cast<float>(v625_data[3])) * v48_data);
              v623_acc += ((static_cast<float>(v625_data[4])) * v50_data);
              v623_acc += ((static_cast<float>(v625_data[5])) * v52_data);
              v623_acc += ((static_cast<float>(v625_data[6])) * v54_data);
              v623_acc += ((static_cast<float>(v625_data[7])) * v56_data);
              v623_acc += ((static_cast<float>(v625_data[8])) * v58_data);
              v623_acc += ((static_cast<float>(v625_data[9])) * v60_data);
              v623_acc += ((static_cast<float>(v625_data[10])) * v62_data);
              v623_acc += ((static_cast<float>(v625_data[11])) * v64_data);
              v623_acc += ((static_cast<float>(v625_data[12])) * v66_data);
              v623_acc += ((static_cast<float>(v625_data[13])) * v68_data);
              v623_acc += ((static_cast<float>(v625_data[14])) * v70_data);
              v623_acc += ((static_cast<float>(v625_data[15])) * v72_data);
              tensorforge::intel_esimd::simd<float, 16> v659_data = tensorforge::slmLoad<float, 16>(s1 + (256_i32));
              v623_acc += ((static_cast<float>(v659_data[0])) * v74_data);
              v623_acc += ((static_cast<float>(v659_data[1])) * v76_data);
              v623_acc += ((static_cast<float>(v659_data[2])) * v78_data);
              v623_acc += ((static_cast<float>(v659_data[3])) * v80_data);
              ir0.template select<16, 1>(192) = v623_acc;
              tensorforge::intel_esimd::simd<float, 16> v668_acc{};
              tensorforge::intel_esimd::simd<float, 16> v670_data = tensorforge::slmLoad<float, 16>(s1 + (260_i32));
              v668_acc += ((static_cast<float>(v670_data[0])) * v42_data);
              v668_acc += ((static_cast<float>(v670_data[1])) * v44_data);
              v668_acc += ((static_cast<float>(v670_data[2])) * v46_data);
              v668_acc += ((static_cast<float>(v670_data[3])) * v48_data);
              v668_acc += ((static_cast<float>(v670_data[4])) * v50_data);
              v668_acc += ((static_cast<float>(v670_data[5])) * v52_data);
              v668_acc += ((static_cast<float>(v670_data[6])) * v54_data);
              v668_acc += ((static_cast<float>(v670_data[7])) * v56_data);
              v668_acc += ((static_cast<float>(v670_data[8])) * v58_data);
              v668_acc += ((static_cast<float>(v670_data[9])) * v60_data);
              v668_acc += ((static_cast<float>(v670_data[10])) * v62_data);
              v668_acc += ((static_cast<float>(v670_data[11])) * v64_data);
              v668_acc += ((static_cast<float>(v670_data[12])) * v66_data);
              v668_acc += ((static_cast<float>(v670_data[13])) * v68_data);
              v668_acc += ((static_cast<float>(v670_data[14])) * v70_data);
              v668_acc += ((static_cast<float>(v670_data[15])) * v72_data);
              tensorforge::intel_esimd::simd<float, 16> v704_data = tensorforge::slmLoad<float, 16>(s1 + (276_i32));
              v668_acc += ((static_cast<float>(v704_data[0])) * v74_data);
              v668_acc += ((static_cast<float>(v704_data[1])) * v76_data);
              v668_acc += ((static_cast<float>(v704_data[2])) * v78_data);
              v668_acc += ((static_cast<float>(v704_data[3])) * v80_data);
              ir0.template select<16, 1>(208) = v668_acc;
              tensorforge::intel_esimd::simd<float, 16> v713_acc{};
              tensorforge::intel_esimd::simd<float, 16> v715_data = tensorforge::slmLoad<float, 16>(s1 + (280_i32));
              v713_acc += ((static_cast<float>(v715_data[0])) * v42_data);
              v713_acc += ((static_cast<float>(v715_data[1])) * v44_data);
              v713_acc += ((static_cast<float>(v715_data[2])) * v46_data);
              v713_acc += ((static_cast<float>(v715_data[3])) * v48_data);
              v713_acc += ((static_cast<float>(v715_data[4])) * v50_data);
              v713_acc += ((static_cast<float>(v715_data[5])) * v52_data);
              v713_acc += ((static_cast<float>(v715_data[6])) * v54_data);
              v713_acc += ((static_cast<float>(v715_data[7])) * v56_data);
              v713_acc += ((static_cast<float>(v715_data[8])) * v58_data);
              v713_acc += ((static_cast<float>(v715_data[9])) * v60_data);
              v713_acc += ((static_cast<float>(v715_data[10])) * v62_data);
              v713_acc += ((static_cast<float>(v715_data[11])) * v64_data);
              v713_acc += ((static_cast<float>(v715_data[12])) * v66_data);
              v713_acc += ((static_cast<float>(v715_data[13])) * v68_data);
              v713_acc += ((static_cast<float>(v715_data[14])) * v70_data);
              v713_acc += ((static_cast<float>(v715_data[15])) * v72_data);
              tensorforge::intel_esimd::simd<float, 16> v749_data = tensorforge::slmLoad<float, 16>(s1 + (296_i32));
              v713_acc += ((static_cast<float>(v749_data[0])) * v74_data);
              v713_acc += ((static_cast<float>(v749_data[1])) * v76_data);
              v713_acc += ((static_cast<float>(v749_data[2])) * v78_data);
              v713_acc += ((static_cast<float>(v749_data[3])) * v80_data);
              ir0.template select<16, 1>(224) = v713_acc;
              tensorforge::intel_esimd::simd<float, 16> v758_acc{};
              tensorforge::intel_esimd::simd<float, 16> v760_data = tensorforge::slmLoad<float, 16>(s1 + (300_i32));
              v758_acc += ((static_cast<float>(v760_data[0])) * v42_data);
              v758_acc += ((static_cast<float>(v760_data[1])) * v44_data);
              v758_acc += ((static_cast<float>(v760_data[2])) * v46_data);
              v758_acc += ((static_cast<float>(v760_data[3])) * v48_data);
              v758_acc += ((static_cast<float>(v760_data[4])) * v50_data);
              v758_acc += ((static_cast<float>(v760_data[5])) * v52_data);
              v758_acc += ((static_cast<float>(v760_data[6])) * v54_data);
              v758_acc += ((static_cast<float>(v760_data[7])) * v56_data);
              v758_acc += ((static_cast<float>(v760_data[8])) * v58_data);
              v758_acc += ((static_cast<float>(v760_data[9])) * v60_data);
              v758_acc += ((static_cast<float>(v760_data[10])) * v62_data);
              v758_acc += ((static_cast<float>(v760_data[11])) * v64_data);
              v758_acc += ((static_cast<float>(v760_data[12])) * v66_data);
              v758_acc += ((static_cast<float>(v760_data[13])) * v68_data);
              v758_acc += ((static_cast<float>(v760_data[14])) * v70_data);
              v758_acc += ((static_cast<float>(v760_data[15])) * v72_data);
              tensorforge::intel_esimd::simd<float, 16> v794_data = tensorforge::slmLoad<float, 16>(s1 + (316_i32));
              v758_acc += ((static_cast<float>(v794_data[0])) * v74_data);
              v758_acc += ((static_cast<float>(v794_data[1])) * v76_data);
              v758_acc += ((static_cast<float>(v794_data[2])) * v78_data);
              v758_acc += ((static_cast<float>(v794_data[3])) * v80_data);
              ir0.template select<16, 1>(240) = v758_acc;
              // r0 = ir0
              #pragma unroll
              for (int32_t v803_n1 = 0; v803_n1 < 16; ++v803_n1) {
                int32_t v804_a = v803_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v806_data(ir0.template select<12, 1>(v804_a));
                r0.template select<12, 1>(v804_a) = v806_data;
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v807_i1 = 0; v807_i1 < 16; ++v807_i1) {
                tensorforge::intel_esimd::simd<float, 12> v810_data(r0.template select<12, 1>((v807_i1 * 16)));
                v810_data.copy_to(glb_m0 + ((v807_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}

