// === base name ===
kernel_b892cde1c7c367aa

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b892cde1c7c367aa = {{1, 16, 1}, 16, 12, 1, 16, 9216, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b892cde1c7c367aa(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b892cde1c7c367aa(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b892cde1c7c367aa(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 2304 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_b892cde1c7c367aa(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b892cde1c7c367aa(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_b892cde1c7c367aa(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_b892cde1c7c367aa(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2304 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 9216 B shared, occupancy grid
        // operands:
        //   m0 12×8(12×8) {0..12}×{0..8} strided
        //   m1 12×16(12×16) {0..12}×{0..16} strided
        //   m2 16×8(16×8) {0..16}×{0..8} strided
        // operations:
        //   m0[i,j] += m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2304}],"shared_bytes":9216,"shared_elements":2304,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,8]],"name":"m0","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,16]],"name":"m1","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,16]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (144 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (128);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v5_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v5_batchId0 < numElements0; v5_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v6_ahead1 = v5_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
            const float *const __restrict__ pf_glb_m1 = &m1[v8_batchId1 * 192 + 0 + m1_extraOffset];
            const float *const __restrict__ pf_glb_m2 = &m2[v8_batchId1 * 128 + 0 + m2_extraOffset];
            float *const __restrict__ pf_glb_m0 = &m0[v8_batchId1 * 96 + 0 + m0_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 192 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 128 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v20_i1 = 0; v20_i1 < 16; ++v20_i1) {
                tensorforge::intel_esimd::simd<float, 12> v25_data;
                v25_data.copy_from(glb_m1 + ((v20_i1 * 12)));
                r0.template select<12, 1>((v20_i1 * 16)) = v25_data;
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v28_ld;
              v28_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v28_ld);
              tensorforge::intel_esimd::simd<float, 64> v29_ld;
              v29_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v29_ld);
              // wait(r0 = load{g>r}(glb_m1););
              tensorforge::intel_esimd::simd<float, 128> r1(0.0f);
              // r1 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v31_i1 = 0; v31_i1 < 8; ++v31_i1) {
                tensorforge::intel_esimd::simd<float, 12> v36_data;
                v36_data.copy_from(glb_m0 + ((v31_i1 * 12)));
                r1.template select<12, 1>((v31_i1 * 16)) = v36_data;
              }
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              // wait(r1 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<float, 128> r2(0.0f);
              // r2 = +(r0 * s0) + name: r1, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 16)]
              tensorforge::intel_esimd::simd<float, 128> ir2(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v48_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v49_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v50_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v51_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v52_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v53_data(r0.template select<16, 1>(192));
              tensorforge::intel_esimd::simd<float, 16> v54_data(r0.template select<16, 1>(208));
              tensorforge::intel_esimd::simd<float, 16> v55_data(r0.template select<16, 1>(224));
              tensorforge::intel_esimd::simd<float, 16> v56_data(r0.template select<16, 1>(240));
              tensorforge::intel_esimd::simd<float, 16> v57_acc{};
              tensorforge::intel_esimd::simd<float, 16> v61_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v57_acc += ((static_cast<float>(v61_data[0])) * v41_data);
              v57_acc += ((static_cast<float>(v61_data[1])) * v42_data);
              v57_acc += ((static_cast<float>(v61_data[2])) * v43_data);
              v57_acc += ((static_cast<float>(v61_data[3])) * v44_data);
              v57_acc += ((static_cast<float>(v61_data[4])) * v45_data);
              v57_acc += ((static_cast<float>(v61_data[5])) * v46_data);
              v57_acc += ((static_cast<float>(v61_data[6])) * v47_data);
              v57_acc += ((static_cast<float>(v61_data[7])) * v48_data);
              v57_acc += ((static_cast<float>(v61_data[8])) * v49_data);
              v57_acc += ((static_cast<float>(v61_data[9])) * v50_data);
              v57_acc += ((static_cast<float>(v61_data[10])) * v51_data);
              v57_acc += ((static_cast<float>(v61_data[11])) * v52_data);
              v57_acc += ((static_cast<float>(v61_data[12])) * v53_data);
              v57_acc += ((static_cast<float>(v61_data[13])) * v54_data);
              v57_acc += ((static_cast<float>(v61_data[14])) * v55_data);
              v57_acc += ((static_cast<float>(v61_data[15])) * v56_data);
              ir2.template select<16, 1>(0) = v57_acc;
              tensorforge::intel_esimd::simd<float, 16> v94_acc{};
              tensorforge::intel_esimd::simd<float, 16> v96_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v94_acc += ((static_cast<float>(v96_data[0])) * v41_data);
              v94_acc += ((static_cast<float>(v96_data[1])) * v42_data);
              v94_acc += ((static_cast<float>(v96_data[2])) * v43_data);
              v94_acc += ((static_cast<float>(v96_data[3])) * v44_data);
              v94_acc += ((static_cast<float>(v96_data[4])) * v45_data);
              v94_acc += ((static_cast<float>(v96_data[5])) * v46_data);
              v94_acc += ((static_cast<float>(v96_data[6])) * v47_data);
              v94_acc += ((static_cast<float>(v96_data[7])) * v48_data);
              v94_acc += ((static_cast<float>(v96_data[8])) * v49_data);
              v94_acc += ((static_cast<float>(v96_data[9])) * v50_data);
              v94_acc += ((static_cast<float>(v96_data[10])) * v51_data);
              v94_acc += ((static_cast<float>(v96_data[11])) * v52_data);
              v94_acc += ((static_cast<float>(v96_data[12])) * v53_data);
              v94_acc += ((static_cast<float>(v96_data[13])) * v54_data);
              v94_acc += ((static_cast<float>(v96_data[14])) * v55_data);
              v94_acc += ((static_cast<float>(v96_data[15])) * v56_data);
              ir2.template select<16, 1>(16) = v94_acc;
              tensorforge::intel_esimd::simd<float, 16> v129_acc{};
              tensorforge::intel_esimd::simd<float, 16> v131_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v129_acc += ((static_cast<float>(v131_data[0])) * v41_data);
              v129_acc += ((static_cast<float>(v131_data[1])) * v42_data);
              v129_acc += ((static_cast<float>(v131_data[2])) * v43_data);
              v129_acc += ((static_cast<float>(v131_data[3])) * v44_data);
              v129_acc += ((static_cast<float>(v131_data[4])) * v45_data);
              v129_acc += ((static_cast<float>(v131_data[5])) * v46_data);
              v129_acc += ((static_cast<float>(v131_data[6])) * v47_data);
              v129_acc += ((static_cast<float>(v131_data[7])) * v48_data);
              v129_acc += ((static_cast<float>(v131_data[8])) * v49_data);
              v129_acc += ((static_cast<float>(v131_data[9])) * v50_data);
              v129_acc += ((static_cast<float>(v131_data[10])) * v51_data);
              v129_acc += ((static_cast<float>(v131_data[11])) * v52_data);
              v129_acc += ((static_cast<float>(v131_data[12])) * v53_data);
              v129_acc += ((static_cast<float>(v131_data[13])) * v54_data);
              v129_acc += ((static_cast<float>(v131_data[14])) * v55_data);
              v129_acc += ((static_cast<float>(v131_data[15])) * v56_data);
              ir2.template select<16, 1>(32) = v129_acc;
              tensorforge::intel_esimd::simd<float, 16> v164_acc{};
              tensorforge::intel_esimd::simd<float, 16> v166_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v164_acc += ((static_cast<float>(v166_data[0])) * v41_data);
              v164_acc += ((static_cast<float>(v166_data[1])) * v42_data);
              v164_acc += ((static_cast<float>(v166_data[2])) * v43_data);
              v164_acc += ((static_cast<float>(v166_data[3])) * v44_data);
              v164_acc += ((static_cast<float>(v166_data[4])) * v45_data);
              v164_acc += ((static_cast<float>(v166_data[5])) * v46_data);
              v164_acc += ((static_cast<float>(v166_data[6])) * v47_data);
              v164_acc += ((static_cast<float>(v166_data[7])) * v48_data);
              v164_acc += ((static_cast<float>(v166_data[8])) * v49_data);
              v164_acc += ((static_cast<float>(v166_data[9])) * v50_data);
              v164_acc += ((static_cast<float>(v166_data[10])) * v51_data);
              v164_acc += ((static_cast<float>(v166_data[11])) * v52_data);
              v164_acc += ((static_cast<float>(v166_data[12])) * v53_data);
              v164_acc += ((static_cast<float>(v166_data[13])) * v54_data);
              v164_acc += ((static_cast<float>(v166_data[14])) * v55_data);
              v164_acc += ((static_cast<float>(v166_data[15])) * v56_data);
              ir2.template select<16, 1>(48) = v164_acc;
              tensorforge::intel_esimd::simd<float, 16> v199_acc{};
              tensorforge::intel_esimd::simd<float, 16> v201_data = tensorforge::slmLoad<float, 16>(s0 + (64_i32));
              v199_acc += ((static_cast<float>(v201_data[0])) * v41_data);
              v199_acc += ((static_cast<float>(v201_data[1])) * v42_data);
              v199_acc += ((static_cast<float>(v201_data[2])) * v43_data);
              v199_acc += ((static_cast<float>(v201_data[3])) * v44_data);
              v199_acc += ((static_cast<float>(v201_data[4])) * v45_data);
              v199_acc += ((static_cast<float>(v201_data[5])) * v46_data);
              v199_acc += ((static_cast<float>(v201_data[6])) * v47_data);
              v199_acc += ((static_cast<float>(v201_data[7])) * v48_data);
              v199_acc += ((static_cast<float>(v201_data[8])) * v49_data);
              v199_acc += ((static_cast<float>(v201_data[9])) * v50_data);
              v199_acc += ((static_cast<float>(v201_data[10])) * v51_data);
              v199_acc += ((static_cast<float>(v201_data[11])) * v52_data);
              v199_acc += ((static_cast<float>(v201_data[12])) * v53_data);
              v199_acc += ((static_cast<float>(v201_data[13])) * v54_data);
              v199_acc += ((static_cast<float>(v201_data[14])) * v55_data);
              v199_acc += ((static_cast<float>(v201_data[15])) * v56_data);
              ir2.template select<16, 1>(64) = v199_acc;
              tensorforge::intel_esimd::simd<float, 16> v234_acc{};
              tensorforge::intel_esimd::simd<float, 16> v236_data = tensorforge::slmLoad<float, 16>(s0 + (80_i32));
              v234_acc += ((static_cast<float>(v236_data[0])) * v41_data);
              v234_acc += ((static_cast<float>(v236_data[1])) * v42_data);
              v234_acc += ((static_cast<float>(v236_data[2])) * v43_data);
              v234_acc += ((static_cast<float>(v236_data[3])) * v44_data);
              v234_acc += ((static_cast<float>(v236_data[4])) * v45_data);
              v234_acc += ((static_cast<float>(v236_data[5])) * v46_data);
              v234_acc += ((static_cast<float>(v236_data[6])) * v47_data);
              v234_acc += ((static_cast<float>(v236_data[7])) * v48_data);
              v234_acc += ((static_cast<float>(v236_data[8])) * v49_data);
              v234_acc += ((static_cast<float>(v236_data[9])) * v50_data);
              v234_acc += ((static_cast<float>(v236_data[10])) * v51_data);
              v234_acc += ((static_cast<float>(v236_data[11])) * v52_data);
              v234_acc += ((static_cast<float>(v236_data[12])) * v53_data);
              v234_acc += ((static_cast<float>(v236_data[13])) * v54_data);
              v234_acc += ((static_cast<float>(v236_data[14])) * v55_data);
              v234_acc += ((static_cast<float>(v236_data[15])) * v56_data);
              ir2.template select<16, 1>(80) = v234_acc;
              tensorforge::intel_esimd::simd<float, 16> v269_acc{};
              tensorforge::intel_esimd::simd<float, 16> v271_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              v269_acc += ((static_cast<float>(v271_data[0])) * v41_data);
              v269_acc += ((static_cast<float>(v271_data[1])) * v42_data);
              v269_acc += ((static_cast<float>(v271_data[2])) * v43_data);
              v269_acc += ((static_cast<float>(v271_data[3])) * v44_data);
              v269_acc += ((static_cast<float>(v271_data[4])) * v45_data);
              v269_acc += ((static_cast<float>(v271_data[5])) * v46_data);
              v269_acc += ((static_cast<float>(v271_data[6])) * v47_data);
              v269_acc += ((static_cast<float>(v271_data[7])) * v48_data);
              v269_acc += ((static_cast<float>(v271_data[8])) * v49_data);
              v269_acc += ((static_cast<float>(v271_data[9])) * v50_data);
              v269_acc += ((static_cast<float>(v271_data[10])) * v51_data);
              v269_acc += ((static_cast<float>(v271_data[11])) * v52_data);
              v269_acc += ((static_cast<float>(v271_data[12])) * v53_data);
              v269_acc += ((static_cast<float>(v271_data[13])) * v54_data);
              v269_acc += ((static_cast<float>(v271_data[14])) * v55_data);
              v269_acc += ((static_cast<float>(v271_data[15])) * v56_data);
              ir2.template select<16, 1>(96) = v269_acc;
              tensorforge::intel_esimd::simd<float, 16> v304_acc{};
              tensorforge::intel_esimd::simd<float, 16> v306_data = tensorforge::slmLoad<float, 16>(s0 + (112_i32));
              v304_acc += ((static_cast<float>(v306_data[0])) * v41_data);
              v304_acc += ((static_cast<float>(v306_data[1])) * v42_data);
              v304_acc += ((static_cast<float>(v306_data[2])) * v43_data);
              v304_acc += ((static_cast<float>(v306_data[3])) * v44_data);
              v304_acc += ((static_cast<float>(v306_data[4])) * v45_data);
              v304_acc += ((static_cast<float>(v306_data[5])) * v46_data);
              v304_acc += ((static_cast<float>(v306_data[6])) * v47_data);
              v304_acc += ((static_cast<float>(v306_data[7])) * v48_data);
              v304_acc += ((static_cast<float>(v306_data[8])) * v49_data);
              v304_acc += ((static_cast<float>(v306_data[9])) * v50_data);
              v304_acc += ((static_cast<float>(v306_data[10])) * v51_data);
              v304_acc += ((static_cast<float>(v306_data[11])) * v52_data);
              v304_acc += ((static_cast<float>(v306_data[12])) * v53_data);
              v304_acc += ((static_cast<float>(v306_data[13])) * v54_data);
              v304_acc += ((static_cast<float>(v306_data[14])) * v55_data);
              v304_acc += ((static_cast<float>(v306_data[15])) * v56_data);
              ir2.template select<16, 1>(112) = v304_acc;
              #pragma unroll
              for (int32_t v339_n1 = 0; v339_n1 < 8; ++v339_n1) {
                int32_t v340_a = v339_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v342_data(ir2.template select<12, 1>(v340_a));
                tensorforge::intel_esimd::simd<float, 12> v343_data(r1.template select<12, 1>(v340_a));
                r2.template select<12, 1>(v340_a) = (v343_data + v342_data);
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v345_i1 = 0; v345_i1 < 8; ++v345_i1) {
                tensorforge::intel_esimd::simd<float, 12> v348_data(r2.template select<12, 1>((v345_i1 * 16)));
                v348_data.copy_to(glb_m0 + ((v345_i1 * 12)));
              }
            }
            tensorforge::prefetchRunsL2<768, 512, 384>(&pf_glb_m1[0], &pf_glb_m2[0], &pf_glb_m0[0]);
          }
        }
      }
    });
  });
}

