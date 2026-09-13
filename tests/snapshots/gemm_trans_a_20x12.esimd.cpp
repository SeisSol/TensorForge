// === base name ===
kernel_4fa3a868da14af01

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_4fa3a868da14af01 = {{1, 16, 1}, 16, 12, 1, 16, 37888, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_4fa3a868da14af01(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_4fa3a868da14af01(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_4fa3a868da14af01(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (1, 16, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 1;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 9472 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_4fa3a868da14af01(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_4fa3a868da14af01(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_4fa3a868da14af01(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_4fa3a868da14af01(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
            const float *const __restrict__ pf_glb_m1 = &m1[v9_batchId1 * 240 + 0 + m1_extraOffset];
            const float *const __restrict__ pf_glb_m2 = &m2[v9_batchId1 * 320 + 0 + m2_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 192 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v6_batchId0 * 240 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v6_batchId0 * 320 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m1[1, 0])
              #pragma unroll
              for (int32_t v19_i0 = 0; v19_i0 < 1; ++v19_i0) {
                int32_t v21_lead = v19_i0 * 16;
                #pragma unroll
                for (int32_t v20_i1 = 0; v20_i1 < 12; ++v20_i1) {
                  tensorforge::intel_esimd::simd<float, 16> v25_data;
                  v25_data.copy_from(glb_m1 + ((v21_lead + (v20_i1 * 20))));
                  tensorforge::slmStore<float, 16>(s0 + ((v21_lead + (v20_i1 * 21))), v25_data);
                }
              }
              #pragma unroll
              for (int32_t v28_i1 = 0; v28_i1 < 12; ++v28_i1) {
                tensorforge::intel_esimd::simd<float, 4> v34_data;
                v34_data.copy_from(glb_m1 + ((16_i32 + (v28_i1 * 20))));
                tensorforge::slmStore<float, 4>(s0 + ((16_i32 + (v28_i1 * 21))), v34_data);
              }
              // s1 = load{g>s}(glb_m2[0, 1])
              #pragma unroll
              for (int32_t i = 0; i < 20; i += 4) {
                tensorforge::intel_esimd::simd<float, 64> v37_ld;
                v37_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + i * 16));
                tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 4 * 0 + i * 16), v37_ld);
              }
              // wait(s0 = load{g>s}(glb_m1[1, 0]));
              // wait(s1 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
              // r0 = +(s0 * s1) + None
              // [(0, 12), (0, 16)] [(0, 20)]
              tensorforge::intel_esimd::simd<float, 256> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v44_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v46_data = tensorforge::slmLoad<float, 16>(s0 + (1_i32));
              tensorforge::intel_esimd::simd<float, 16> v48_data = tensorforge::slmLoad<float, 16>(s0 + (2_i32));
              tensorforge::intel_esimd::simd<float, 16> v50_data = tensorforge::slmLoad<float, 16>(s0 + (3_i32));
              tensorforge::intel_esimd::simd<float, 16> v52_data = tensorforge::slmLoad<float, 16>(s0 + (4_i32));
              tensorforge::intel_esimd::simd<float, 16> v54_data = tensorforge::slmLoad<float, 16>(s0 + (5_i32));
              tensorforge::intel_esimd::simd<float, 16> v56_data = tensorforge::slmLoad<float, 16>(s0 + (6_i32));
              tensorforge::intel_esimd::simd<float, 16> v58_data = tensorforge::slmLoad<float, 16>(s0 + (7_i32));
              tensorforge::intel_esimd::simd<float, 16> v60_data = tensorforge::slmLoad<float, 16>(s0 + (8_i32));
              tensorforge::intel_esimd::simd<float, 16> v62_data = tensorforge::slmLoad<float, 16>(s0 + (9_i32));
              tensorforge::intel_esimd::simd<float, 16> v64_data = tensorforge::slmLoad<float, 16>(s0 + (10_i32));
              tensorforge::intel_esimd::simd<float, 16> v66_data = tensorforge::slmLoad<float, 16>(s0 + (11_i32));
              tensorforge::intel_esimd::simd<float, 16> v68_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              tensorforge::intel_esimd::simd<float, 16> v70_data = tensorforge::slmLoad<float, 16>(s0 + (13_i32));
              tensorforge::intel_esimd::simd<float, 16> v72_data = tensorforge::slmLoad<float, 16>(s0 + (14_i32));
              tensorforge::intel_esimd::simd<float, 16> v74_data = tensorforge::slmLoad<float, 16>(s0 + (15_i32));
              tensorforge::intel_esimd::simd<float, 16> v76_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              tensorforge::intel_esimd::simd<float, 16> v78_data = tensorforge::slmLoad<float, 16>(s0 + (17_i32));
              tensorforge::intel_esimd::simd<float, 16> v80_data = tensorforge::slmLoad<float, 16>(s0 + (18_i32));
              tensorforge::intel_esimd::simd<float, 16> v82_data = tensorforge::slmLoad<float, 16>(s0 + (19_i32));
              tensorforge::intel_esimd::simd<float, 16> v83_acc{};
              tensorforge::intel_esimd::simd<float, 16> v85_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v83_acc += ((static_cast<float>(v85_data[0])) * v44_data);
              v83_acc += ((static_cast<float>(v85_data[1])) * v46_data);
              v83_acc += ((static_cast<float>(v85_data[2])) * v48_data);
              v83_acc += ((static_cast<float>(v85_data[3])) * v50_data);
              v83_acc += ((static_cast<float>(v85_data[4])) * v52_data);
              v83_acc += ((static_cast<float>(v85_data[5])) * v54_data);
              v83_acc += ((static_cast<float>(v85_data[6])) * v56_data);
              v83_acc += ((static_cast<float>(v85_data[7])) * v58_data);
              v83_acc += ((static_cast<float>(v85_data[8])) * v60_data);
              v83_acc += ((static_cast<float>(v85_data[9])) * v62_data);
              v83_acc += ((static_cast<float>(v85_data[10])) * v64_data);
              v83_acc += ((static_cast<float>(v85_data[11])) * v66_data);
              v83_acc += ((static_cast<float>(v85_data[12])) * v68_data);
              v83_acc += ((static_cast<float>(v85_data[13])) * v70_data);
              v83_acc += ((static_cast<float>(v85_data[14])) * v72_data);
              v83_acc += ((static_cast<float>(v85_data[15])) * v74_data);
              tensorforge::intel_esimd::simd<float, 16> v121_data = tensorforge::slmLoad<float, 16>(s1 + (16_i32));
              v83_acc += ((static_cast<float>(v121_data[0])) * v76_data);
              v83_acc += ((static_cast<float>(v121_data[1])) * v78_data);
              v83_acc += ((static_cast<float>(v121_data[2])) * v80_data);
              v83_acc += ((static_cast<float>(v121_data[3])) * v82_data);
              ir0.template select<16, 1>(0) = v83_acc;
              tensorforge::intel_esimd::simd<float, 16> v130_acc{};
              tensorforge::intel_esimd::simd<float, 16> v132_data = tensorforge::slmLoad<float, 16>(s1 + (20_i32));
              v130_acc += ((static_cast<float>(v132_data[0])) * v44_data);
              v130_acc += ((static_cast<float>(v132_data[1])) * v46_data);
              v130_acc += ((static_cast<float>(v132_data[2])) * v48_data);
              v130_acc += ((static_cast<float>(v132_data[3])) * v50_data);
              v130_acc += ((static_cast<float>(v132_data[4])) * v52_data);
              v130_acc += ((static_cast<float>(v132_data[5])) * v54_data);
              v130_acc += ((static_cast<float>(v132_data[6])) * v56_data);
              v130_acc += ((static_cast<float>(v132_data[7])) * v58_data);
              v130_acc += ((static_cast<float>(v132_data[8])) * v60_data);
              v130_acc += ((static_cast<float>(v132_data[9])) * v62_data);
              v130_acc += ((static_cast<float>(v132_data[10])) * v64_data);
              v130_acc += ((static_cast<float>(v132_data[11])) * v66_data);
              v130_acc += ((static_cast<float>(v132_data[12])) * v68_data);
              v130_acc += ((static_cast<float>(v132_data[13])) * v70_data);
              v130_acc += ((static_cast<float>(v132_data[14])) * v72_data);
              v130_acc += ((static_cast<float>(v132_data[15])) * v74_data);
              tensorforge::intel_esimd::simd<float, 16> v166_data = tensorforge::slmLoad<float, 16>(s1 + (36_i32));
              v130_acc += ((static_cast<float>(v166_data[0])) * v76_data);
              v130_acc += ((static_cast<float>(v166_data[1])) * v78_data);
              v130_acc += ((static_cast<float>(v166_data[2])) * v80_data);
              v130_acc += ((static_cast<float>(v166_data[3])) * v82_data);
              ir0.template select<16, 1>(16) = v130_acc;
              tensorforge::intel_esimd::simd<float, 16> v175_acc{};
              tensorforge::intel_esimd::simd<float, 16> v177_data = tensorforge::slmLoad<float, 16>(s1 + (40_i32));
              v175_acc += ((static_cast<float>(v177_data[0])) * v44_data);
              v175_acc += ((static_cast<float>(v177_data[1])) * v46_data);
              v175_acc += ((static_cast<float>(v177_data[2])) * v48_data);
              v175_acc += ((static_cast<float>(v177_data[3])) * v50_data);
              v175_acc += ((static_cast<float>(v177_data[4])) * v52_data);
              v175_acc += ((static_cast<float>(v177_data[5])) * v54_data);
              v175_acc += ((static_cast<float>(v177_data[6])) * v56_data);
              v175_acc += ((static_cast<float>(v177_data[7])) * v58_data);
              v175_acc += ((static_cast<float>(v177_data[8])) * v60_data);
              v175_acc += ((static_cast<float>(v177_data[9])) * v62_data);
              v175_acc += ((static_cast<float>(v177_data[10])) * v64_data);
              v175_acc += ((static_cast<float>(v177_data[11])) * v66_data);
              v175_acc += ((static_cast<float>(v177_data[12])) * v68_data);
              v175_acc += ((static_cast<float>(v177_data[13])) * v70_data);
              v175_acc += ((static_cast<float>(v177_data[14])) * v72_data);
              v175_acc += ((static_cast<float>(v177_data[15])) * v74_data);
              tensorforge::intel_esimd::simd<float, 16> v211_data = tensorforge::slmLoad<float, 16>(s1 + (56_i32));
              v175_acc += ((static_cast<float>(v211_data[0])) * v76_data);
              v175_acc += ((static_cast<float>(v211_data[1])) * v78_data);
              v175_acc += ((static_cast<float>(v211_data[2])) * v80_data);
              v175_acc += ((static_cast<float>(v211_data[3])) * v82_data);
              ir0.template select<16, 1>(32) = v175_acc;
              tensorforge::intel_esimd::simd<float, 16> v220_acc{};
              tensorforge::intel_esimd::simd<float, 16> v222_data = tensorforge::slmLoad<float, 16>(s1 + (60_i32));
              v220_acc += ((static_cast<float>(v222_data[0])) * v44_data);
              v220_acc += ((static_cast<float>(v222_data[1])) * v46_data);
              v220_acc += ((static_cast<float>(v222_data[2])) * v48_data);
              v220_acc += ((static_cast<float>(v222_data[3])) * v50_data);
              v220_acc += ((static_cast<float>(v222_data[4])) * v52_data);
              v220_acc += ((static_cast<float>(v222_data[5])) * v54_data);
              v220_acc += ((static_cast<float>(v222_data[6])) * v56_data);
              v220_acc += ((static_cast<float>(v222_data[7])) * v58_data);
              v220_acc += ((static_cast<float>(v222_data[8])) * v60_data);
              v220_acc += ((static_cast<float>(v222_data[9])) * v62_data);
              v220_acc += ((static_cast<float>(v222_data[10])) * v64_data);
              v220_acc += ((static_cast<float>(v222_data[11])) * v66_data);
              v220_acc += ((static_cast<float>(v222_data[12])) * v68_data);
              v220_acc += ((static_cast<float>(v222_data[13])) * v70_data);
              v220_acc += ((static_cast<float>(v222_data[14])) * v72_data);
              v220_acc += ((static_cast<float>(v222_data[15])) * v74_data);
              tensorforge::intel_esimd::simd<float, 16> v256_data = tensorforge::slmLoad<float, 16>(s1 + (76_i32));
              v220_acc += ((static_cast<float>(v256_data[0])) * v76_data);
              v220_acc += ((static_cast<float>(v256_data[1])) * v78_data);
              v220_acc += ((static_cast<float>(v256_data[2])) * v80_data);
              v220_acc += ((static_cast<float>(v256_data[3])) * v82_data);
              ir0.template select<16, 1>(48) = v220_acc;
              tensorforge::intel_esimd::simd<float, 16> v265_acc{};
              tensorforge::intel_esimd::simd<float, 16> v267_data = tensorforge::slmLoad<float, 16>(s1 + (80_i32));
              v265_acc += ((static_cast<float>(v267_data[0])) * v44_data);
              v265_acc += ((static_cast<float>(v267_data[1])) * v46_data);
              v265_acc += ((static_cast<float>(v267_data[2])) * v48_data);
              v265_acc += ((static_cast<float>(v267_data[3])) * v50_data);
              v265_acc += ((static_cast<float>(v267_data[4])) * v52_data);
              v265_acc += ((static_cast<float>(v267_data[5])) * v54_data);
              v265_acc += ((static_cast<float>(v267_data[6])) * v56_data);
              v265_acc += ((static_cast<float>(v267_data[7])) * v58_data);
              v265_acc += ((static_cast<float>(v267_data[8])) * v60_data);
              v265_acc += ((static_cast<float>(v267_data[9])) * v62_data);
              v265_acc += ((static_cast<float>(v267_data[10])) * v64_data);
              v265_acc += ((static_cast<float>(v267_data[11])) * v66_data);
              v265_acc += ((static_cast<float>(v267_data[12])) * v68_data);
              v265_acc += ((static_cast<float>(v267_data[13])) * v70_data);
              v265_acc += ((static_cast<float>(v267_data[14])) * v72_data);
              v265_acc += ((static_cast<float>(v267_data[15])) * v74_data);
              tensorforge::intel_esimd::simd<float, 16> v301_data = tensorforge::slmLoad<float, 16>(s1 + (96_i32));
              v265_acc += ((static_cast<float>(v301_data[0])) * v76_data);
              v265_acc += ((static_cast<float>(v301_data[1])) * v78_data);
              v265_acc += ((static_cast<float>(v301_data[2])) * v80_data);
              v265_acc += ((static_cast<float>(v301_data[3])) * v82_data);
              ir0.template select<16, 1>(64) = v265_acc;
              tensorforge::intel_esimd::simd<float, 16> v310_acc{};
              tensorforge::intel_esimd::simd<float, 16> v312_data = tensorforge::slmLoad<float, 16>(s1 + (100_i32));
              v310_acc += ((static_cast<float>(v312_data[0])) * v44_data);
              v310_acc += ((static_cast<float>(v312_data[1])) * v46_data);
              v310_acc += ((static_cast<float>(v312_data[2])) * v48_data);
              v310_acc += ((static_cast<float>(v312_data[3])) * v50_data);
              v310_acc += ((static_cast<float>(v312_data[4])) * v52_data);
              v310_acc += ((static_cast<float>(v312_data[5])) * v54_data);
              v310_acc += ((static_cast<float>(v312_data[6])) * v56_data);
              v310_acc += ((static_cast<float>(v312_data[7])) * v58_data);
              v310_acc += ((static_cast<float>(v312_data[8])) * v60_data);
              v310_acc += ((static_cast<float>(v312_data[9])) * v62_data);
              v310_acc += ((static_cast<float>(v312_data[10])) * v64_data);
              v310_acc += ((static_cast<float>(v312_data[11])) * v66_data);
              v310_acc += ((static_cast<float>(v312_data[12])) * v68_data);
              v310_acc += ((static_cast<float>(v312_data[13])) * v70_data);
              v310_acc += ((static_cast<float>(v312_data[14])) * v72_data);
              v310_acc += ((static_cast<float>(v312_data[15])) * v74_data);
              tensorforge::intel_esimd::simd<float, 16> v346_data = tensorforge::slmLoad<float, 16>(s1 + (116_i32));
              v310_acc += ((static_cast<float>(v346_data[0])) * v76_data);
              v310_acc += ((static_cast<float>(v346_data[1])) * v78_data);
              v310_acc += ((static_cast<float>(v346_data[2])) * v80_data);
              v310_acc += ((static_cast<float>(v346_data[3])) * v82_data);
              ir0.template select<16, 1>(80) = v310_acc;
              tensorforge::intel_esimd::simd<float, 16> v355_acc{};
              tensorforge::intel_esimd::simd<float, 16> v357_data = tensorforge::slmLoad<float, 16>(s1 + (120_i32));
              v355_acc += ((static_cast<float>(v357_data[0])) * v44_data);
              v355_acc += ((static_cast<float>(v357_data[1])) * v46_data);
              v355_acc += ((static_cast<float>(v357_data[2])) * v48_data);
              v355_acc += ((static_cast<float>(v357_data[3])) * v50_data);
              v355_acc += ((static_cast<float>(v357_data[4])) * v52_data);
              v355_acc += ((static_cast<float>(v357_data[5])) * v54_data);
              v355_acc += ((static_cast<float>(v357_data[6])) * v56_data);
              v355_acc += ((static_cast<float>(v357_data[7])) * v58_data);
              v355_acc += ((static_cast<float>(v357_data[8])) * v60_data);
              v355_acc += ((static_cast<float>(v357_data[9])) * v62_data);
              v355_acc += ((static_cast<float>(v357_data[10])) * v64_data);
              v355_acc += ((static_cast<float>(v357_data[11])) * v66_data);
              v355_acc += ((static_cast<float>(v357_data[12])) * v68_data);
              v355_acc += ((static_cast<float>(v357_data[13])) * v70_data);
              v355_acc += ((static_cast<float>(v357_data[14])) * v72_data);
              v355_acc += ((static_cast<float>(v357_data[15])) * v74_data);
              tensorforge::intel_esimd::simd<float, 16> v391_data = tensorforge::slmLoad<float, 16>(s1 + (136_i32));
              v355_acc += ((static_cast<float>(v391_data[0])) * v76_data);
              v355_acc += ((static_cast<float>(v391_data[1])) * v78_data);
              v355_acc += ((static_cast<float>(v391_data[2])) * v80_data);
              v355_acc += ((static_cast<float>(v391_data[3])) * v82_data);
              ir0.template select<16, 1>(96) = v355_acc;
              tensorforge::intel_esimd::simd<float, 16> v400_acc{};
              tensorforge::intel_esimd::simd<float, 16> v402_data = tensorforge::slmLoad<float, 16>(s1 + (140_i32));
              v400_acc += ((static_cast<float>(v402_data[0])) * v44_data);
              v400_acc += ((static_cast<float>(v402_data[1])) * v46_data);
              v400_acc += ((static_cast<float>(v402_data[2])) * v48_data);
              v400_acc += ((static_cast<float>(v402_data[3])) * v50_data);
              v400_acc += ((static_cast<float>(v402_data[4])) * v52_data);
              v400_acc += ((static_cast<float>(v402_data[5])) * v54_data);
              v400_acc += ((static_cast<float>(v402_data[6])) * v56_data);
              v400_acc += ((static_cast<float>(v402_data[7])) * v58_data);
              v400_acc += ((static_cast<float>(v402_data[8])) * v60_data);
              v400_acc += ((static_cast<float>(v402_data[9])) * v62_data);
              v400_acc += ((static_cast<float>(v402_data[10])) * v64_data);
              v400_acc += ((static_cast<float>(v402_data[11])) * v66_data);
              v400_acc += ((static_cast<float>(v402_data[12])) * v68_data);
              v400_acc += ((static_cast<float>(v402_data[13])) * v70_data);
              v400_acc += ((static_cast<float>(v402_data[14])) * v72_data);
              v400_acc += ((static_cast<float>(v402_data[15])) * v74_data);
              tensorforge::intel_esimd::simd<float, 16> v436_data = tensorforge::slmLoad<float, 16>(s1 + (156_i32));
              v400_acc += ((static_cast<float>(v436_data[0])) * v76_data);
              v400_acc += ((static_cast<float>(v436_data[1])) * v78_data);
              v400_acc += ((static_cast<float>(v436_data[2])) * v80_data);
              v400_acc += ((static_cast<float>(v436_data[3])) * v82_data);
              ir0.template select<16, 1>(112) = v400_acc;
              tensorforge::intel_esimd::simd<float, 16> v445_acc{};
              tensorforge::intel_esimd::simd<float, 16> v447_data = tensorforge::slmLoad<float, 16>(s1 + (160_i32));
              v445_acc += ((static_cast<float>(v447_data[0])) * v44_data);
              v445_acc += ((static_cast<float>(v447_data[1])) * v46_data);
              v445_acc += ((static_cast<float>(v447_data[2])) * v48_data);
              v445_acc += ((static_cast<float>(v447_data[3])) * v50_data);
              v445_acc += ((static_cast<float>(v447_data[4])) * v52_data);
              v445_acc += ((static_cast<float>(v447_data[5])) * v54_data);
              v445_acc += ((static_cast<float>(v447_data[6])) * v56_data);
              v445_acc += ((static_cast<float>(v447_data[7])) * v58_data);
              v445_acc += ((static_cast<float>(v447_data[8])) * v60_data);
              v445_acc += ((static_cast<float>(v447_data[9])) * v62_data);
              v445_acc += ((static_cast<float>(v447_data[10])) * v64_data);
              v445_acc += ((static_cast<float>(v447_data[11])) * v66_data);
              v445_acc += ((static_cast<float>(v447_data[12])) * v68_data);
              v445_acc += ((static_cast<float>(v447_data[13])) * v70_data);
              v445_acc += ((static_cast<float>(v447_data[14])) * v72_data);
              v445_acc += ((static_cast<float>(v447_data[15])) * v74_data);
              tensorforge::intel_esimd::simd<float, 16> v481_data = tensorforge::slmLoad<float, 16>(s1 + (176_i32));
              v445_acc += ((static_cast<float>(v481_data[0])) * v76_data);
              v445_acc += ((static_cast<float>(v481_data[1])) * v78_data);
              v445_acc += ((static_cast<float>(v481_data[2])) * v80_data);
              v445_acc += ((static_cast<float>(v481_data[3])) * v82_data);
              ir0.template select<16, 1>(128) = v445_acc;
              tensorforge::intel_esimd::simd<float, 16> v490_acc{};
              tensorforge::intel_esimd::simd<float, 16> v492_data = tensorforge::slmLoad<float, 16>(s1 + (180_i32));
              v490_acc += ((static_cast<float>(v492_data[0])) * v44_data);
              v490_acc += ((static_cast<float>(v492_data[1])) * v46_data);
              v490_acc += ((static_cast<float>(v492_data[2])) * v48_data);
              v490_acc += ((static_cast<float>(v492_data[3])) * v50_data);
              v490_acc += ((static_cast<float>(v492_data[4])) * v52_data);
              v490_acc += ((static_cast<float>(v492_data[5])) * v54_data);
              v490_acc += ((static_cast<float>(v492_data[6])) * v56_data);
              v490_acc += ((static_cast<float>(v492_data[7])) * v58_data);
              v490_acc += ((static_cast<float>(v492_data[8])) * v60_data);
              v490_acc += ((static_cast<float>(v492_data[9])) * v62_data);
              v490_acc += ((static_cast<float>(v492_data[10])) * v64_data);
              v490_acc += ((static_cast<float>(v492_data[11])) * v66_data);
              v490_acc += ((static_cast<float>(v492_data[12])) * v68_data);
              v490_acc += ((static_cast<float>(v492_data[13])) * v70_data);
              v490_acc += ((static_cast<float>(v492_data[14])) * v72_data);
              v490_acc += ((static_cast<float>(v492_data[15])) * v74_data);
              tensorforge::intel_esimd::simd<float, 16> v526_data = tensorforge::slmLoad<float, 16>(s1 + (196_i32));
              v490_acc += ((static_cast<float>(v526_data[0])) * v76_data);
              v490_acc += ((static_cast<float>(v526_data[1])) * v78_data);
              v490_acc += ((static_cast<float>(v526_data[2])) * v80_data);
              v490_acc += ((static_cast<float>(v526_data[3])) * v82_data);
              ir0.template select<16, 1>(144) = v490_acc;
              tensorforge::intel_esimd::simd<float, 16> v535_acc{};
              tensorforge::intel_esimd::simd<float, 16> v537_data = tensorforge::slmLoad<float, 16>(s1 + (200_i32));
              v535_acc += ((static_cast<float>(v537_data[0])) * v44_data);
              v535_acc += ((static_cast<float>(v537_data[1])) * v46_data);
              v535_acc += ((static_cast<float>(v537_data[2])) * v48_data);
              v535_acc += ((static_cast<float>(v537_data[3])) * v50_data);
              v535_acc += ((static_cast<float>(v537_data[4])) * v52_data);
              v535_acc += ((static_cast<float>(v537_data[5])) * v54_data);
              v535_acc += ((static_cast<float>(v537_data[6])) * v56_data);
              v535_acc += ((static_cast<float>(v537_data[7])) * v58_data);
              v535_acc += ((static_cast<float>(v537_data[8])) * v60_data);
              v535_acc += ((static_cast<float>(v537_data[9])) * v62_data);
              v535_acc += ((static_cast<float>(v537_data[10])) * v64_data);
              v535_acc += ((static_cast<float>(v537_data[11])) * v66_data);
              v535_acc += ((static_cast<float>(v537_data[12])) * v68_data);
              v535_acc += ((static_cast<float>(v537_data[13])) * v70_data);
              v535_acc += ((static_cast<float>(v537_data[14])) * v72_data);
              v535_acc += ((static_cast<float>(v537_data[15])) * v74_data);
              tensorforge::intel_esimd::simd<float, 16> v571_data = tensorforge::slmLoad<float, 16>(s1 + (216_i32));
              v535_acc += ((static_cast<float>(v571_data[0])) * v76_data);
              v535_acc += ((static_cast<float>(v571_data[1])) * v78_data);
              v535_acc += ((static_cast<float>(v571_data[2])) * v80_data);
              v535_acc += ((static_cast<float>(v571_data[3])) * v82_data);
              ir0.template select<16, 1>(160) = v535_acc;
              tensorforge::intel_esimd::simd<float, 16> v580_acc{};
              tensorforge::intel_esimd::simd<float, 16> v582_data = tensorforge::slmLoad<float, 16>(s1 + (220_i32));
              v580_acc += ((static_cast<float>(v582_data[0])) * v44_data);
              v580_acc += ((static_cast<float>(v582_data[1])) * v46_data);
              v580_acc += ((static_cast<float>(v582_data[2])) * v48_data);
              v580_acc += ((static_cast<float>(v582_data[3])) * v50_data);
              v580_acc += ((static_cast<float>(v582_data[4])) * v52_data);
              v580_acc += ((static_cast<float>(v582_data[5])) * v54_data);
              v580_acc += ((static_cast<float>(v582_data[6])) * v56_data);
              v580_acc += ((static_cast<float>(v582_data[7])) * v58_data);
              v580_acc += ((static_cast<float>(v582_data[8])) * v60_data);
              v580_acc += ((static_cast<float>(v582_data[9])) * v62_data);
              v580_acc += ((static_cast<float>(v582_data[10])) * v64_data);
              v580_acc += ((static_cast<float>(v582_data[11])) * v66_data);
              v580_acc += ((static_cast<float>(v582_data[12])) * v68_data);
              v580_acc += ((static_cast<float>(v582_data[13])) * v70_data);
              v580_acc += ((static_cast<float>(v582_data[14])) * v72_data);
              v580_acc += ((static_cast<float>(v582_data[15])) * v74_data);
              tensorforge::intel_esimd::simd<float, 16> v616_data = tensorforge::slmLoad<float, 16>(s1 + (236_i32));
              v580_acc += ((static_cast<float>(v616_data[0])) * v76_data);
              v580_acc += ((static_cast<float>(v616_data[1])) * v78_data);
              v580_acc += ((static_cast<float>(v616_data[2])) * v80_data);
              v580_acc += ((static_cast<float>(v616_data[3])) * v82_data);
              ir0.template select<16, 1>(176) = v580_acc;
              tensorforge::intel_esimd::simd<float, 16> v625_acc{};
              tensorforge::intel_esimd::simd<float, 16> v627_data = tensorforge::slmLoad<float, 16>(s1 + (240_i32));
              v625_acc += ((static_cast<float>(v627_data[0])) * v44_data);
              v625_acc += ((static_cast<float>(v627_data[1])) * v46_data);
              v625_acc += ((static_cast<float>(v627_data[2])) * v48_data);
              v625_acc += ((static_cast<float>(v627_data[3])) * v50_data);
              v625_acc += ((static_cast<float>(v627_data[4])) * v52_data);
              v625_acc += ((static_cast<float>(v627_data[5])) * v54_data);
              v625_acc += ((static_cast<float>(v627_data[6])) * v56_data);
              v625_acc += ((static_cast<float>(v627_data[7])) * v58_data);
              v625_acc += ((static_cast<float>(v627_data[8])) * v60_data);
              v625_acc += ((static_cast<float>(v627_data[9])) * v62_data);
              v625_acc += ((static_cast<float>(v627_data[10])) * v64_data);
              v625_acc += ((static_cast<float>(v627_data[11])) * v66_data);
              v625_acc += ((static_cast<float>(v627_data[12])) * v68_data);
              v625_acc += ((static_cast<float>(v627_data[13])) * v70_data);
              v625_acc += ((static_cast<float>(v627_data[14])) * v72_data);
              v625_acc += ((static_cast<float>(v627_data[15])) * v74_data);
              tensorforge::intel_esimd::simd<float, 16> v661_data = tensorforge::slmLoad<float, 16>(s1 + (256_i32));
              v625_acc += ((static_cast<float>(v661_data[0])) * v76_data);
              v625_acc += ((static_cast<float>(v661_data[1])) * v78_data);
              v625_acc += ((static_cast<float>(v661_data[2])) * v80_data);
              v625_acc += ((static_cast<float>(v661_data[3])) * v82_data);
              ir0.template select<16, 1>(192) = v625_acc;
              tensorforge::intel_esimd::simd<float, 16> v670_acc{};
              tensorforge::intel_esimd::simd<float, 16> v672_data = tensorforge::slmLoad<float, 16>(s1 + (260_i32));
              v670_acc += ((static_cast<float>(v672_data[0])) * v44_data);
              v670_acc += ((static_cast<float>(v672_data[1])) * v46_data);
              v670_acc += ((static_cast<float>(v672_data[2])) * v48_data);
              v670_acc += ((static_cast<float>(v672_data[3])) * v50_data);
              v670_acc += ((static_cast<float>(v672_data[4])) * v52_data);
              v670_acc += ((static_cast<float>(v672_data[5])) * v54_data);
              v670_acc += ((static_cast<float>(v672_data[6])) * v56_data);
              v670_acc += ((static_cast<float>(v672_data[7])) * v58_data);
              v670_acc += ((static_cast<float>(v672_data[8])) * v60_data);
              v670_acc += ((static_cast<float>(v672_data[9])) * v62_data);
              v670_acc += ((static_cast<float>(v672_data[10])) * v64_data);
              v670_acc += ((static_cast<float>(v672_data[11])) * v66_data);
              v670_acc += ((static_cast<float>(v672_data[12])) * v68_data);
              v670_acc += ((static_cast<float>(v672_data[13])) * v70_data);
              v670_acc += ((static_cast<float>(v672_data[14])) * v72_data);
              v670_acc += ((static_cast<float>(v672_data[15])) * v74_data);
              tensorforge::intel_esimd::simd<float, 16> v706_data = tensorforge::slmLoad<float, 16>(s1 + (276_i32));
              v670_acc += ((static_cast<float>(v706_data[0])) * v76_data);
              v670_acc += ((static_cast<float>(v706_data[1])) * v78_data);
              v670_acc += ((static_cast<float>(v706_data[2])) * v80_data);
              v670_acc += ((static_cast<float>(v706_data[3])) * v82_data);
              ir0.template select<16, 1>(208) = v670_acc;
              tensorforge::intel_esimd::simd<float, 16> v715_acc{};
              tensorforge::intel_esimd::simd<float, 16> v717_data = tensorforge::slmLoad<float, 16>(s1 + (280_i32));
              v715_acc += ((static_cast<float>(v717_data[0])) * v44_data);
              v715_acc += ((static_cast<float>(v717_data[1])) * v46_data);
              v715_acc += ((static_cast<float>(v717_data[2])) * v48_data);
              v715_acc += ((static_cast<float>(v717_data[3])) * v50_data);
              v715_acc += ((static_cast<float>(v717_data[4])) * v52_data);
              v715_acc += ((static_cast<float>(v717_data[5])) * v54_data);
              v715_acc += ((static_cast<float>(v717_data[6])) * v56_data);
              v715_acc += ((static_cast<float>(v717_data[7])) * v58_data);
              v715_acc += ((static_cast<float>(v717_data[8])) * v60_data);
              v715_acc += ((static_cast<float>(v717_data[9])) * v62_data);
              v715_acc += ((static_cast<float>(v717_data[10])) * v64_data);
              v715_acc += ((static_cast<float>(v717_data[11])) * v66_data);
              v715_acc += ((static_cast<float>(v717_data[12])) * v68_data);
              v715_acc += ((static_cast<float>(v717_data[13])) * v70_data);
              v715_acc += ((static_cast<float>(v717_data[14])) * v72_data);
              v715_acc += ((static_cast<float>(v717_data[15])) * v74_data);
              tensorforge::intel_esimd::simd<float, 16> v751_data = tensorforge::slmLoad<float, 16>(s1 + (296_i32));
              v715_acc += ((static_cast<float>(v751_data[0])) * v76_data);
              v715_acc += ((static_cast<float>(v751_data[1])) * v78_data);
              v715_acc += ((static_cast<float>(v751_data[2])) * v80_data);
              v715_acc += ((static_cast<float>(v751_data[3])) * v82_data);
              ir0.template select<16, 1>(224) = v715_acc;
              tensorforge::intel_esimd::simd<float, 16> v760_acc{};
              tensorforge::intel_esimd::simd<float, 16> v762_data = tensorforge::slmLoad<float, 16>(s1 + (300_i32));
              v760_acc += ((static_cast<float>(v762_data[0])) * v44_data);
              v760_acc += ((static_cast<float>(v762_data[1])) * v46_data);
              v760_acc += ((static_cast<float>(v762_data[2])) * v48_data);
              v760_acc += ((static_cast<float>(v762_data[3])) * v50_data);
              v760_acc += ((static_cast<float>(v762_data[4])) * v52_data);
              v760_acc += ((static_cast<float>(v762_data[5])) * v54_data);
              v760_acc += ((static_cast<float>(v762_data[6])) * v56_data);
              v760_acc += ((static_cast<float>(v762_data[7])) * v58_data);
              v760_acc += ((static_cast<float>(v762_data[8])) * v60_data);
              v760_acc += ((static_cast<float>(v762_data[9])) * v62_data);
              v760_acc += ((static_cast<float>(v762_data[10])) * v64_data);
              v760_acc += ((static_cast<float>(v762_data[11])) * v66_data);
              v760_acc += ((static_cast<float>(v762_data[12])) * v68_data);
              v760_acc += ((static_cast<float>(v762_data[13])) * v70_data);
              v760_acc += ((static_cast<float>(v762_data[14])) * v72_data);
              v760_acc += ((static_cast<float>(v762_data[15])) * v74_data);
              tensorforge::intel_esimd::simd<float, 16> v796_data = tensorforge::slmLoad<float, 16>(s1 + (316_i32));
              v760_acc += ((static_cast<float>(v796_data[0])) * v76_data);
              v760_acc += ((static_cast<float>(v796_data[1])) * v78_data);
              v760_acc += ((static_cast<float>(v796_data[2])) * v80_data);
              v760_acc += ((static_cast<float>(v796_data[3])) * v82_data);
              ir0.template select<16, 1>(240) = v760_acc;
              #pragma unroll
              for (int32_t v805_n1 = 0; v805_n1 < 16; ++v805_n1) {
                int32_t v806_a = v805_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v808_data(ir0.template select<12, 1>(v806_a));
                r0.template select<12, 1>(v806_a) = v808_data;
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v809_i1 = 0; v809_i1 < 16; ++v809_i1) {
                tensorforge::intel_esimd::simd<float, 12> v812_data(r0.template select<12, 1>((v809_i1 * 16)));
                v812_data.copy_to(glb_m0 + ((v809_i1 * 12)));
              }
            }
            tensorforge::prefetchL2<240>(&pf_glb_m1[0]);
            tensorforge::prefetchL2<320>(&pf_glb_m2[0]);
          }
        }
      }
    });
  });
}

