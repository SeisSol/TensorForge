// === base name ===
kernel_a00f4213f72b70f3

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_a00f4213f72b70f3 = {{1, 16, 1}, 16, 16, 1, 16, 12288, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_a00f4213f72b70f3(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_a00f4213f72b70f3(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_a00f4213f72b70f3(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 3072 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_a00f4213f72b70f3(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_a00f4213f72b70f3(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_a00f4213f72b70f3(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_a00f4213f72b70f3(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<3072 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 12288 B shared, occupancy grid
        // operands:
        //   m0 16×11(16×11) {0..16}×{0..11} strided
        //   m1 16×16(16×16) {0..16}×{0..16} strided
        //   m2 16×11(16×11) {0..16}×{0..11} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":3072}],"shared_bytes":12288,"shared_elements":3072,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[16,11]],"name":"m0","ordered":false,"parts":1,"shape":[16,11],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,11]],"name":"m2","ordered":false,"parts":1,"shape":[16,11],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,11]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,11]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,11]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,11]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (192 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 176 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 176 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v20_i0 = 0; v20_i0 < 1; ++v20_i0) {
                int32_t v22_lead = v20_i0 * 16;
                #pragma unroll
                for (int32_t v21_i1 = 0; v21_i1 < 16; ++v21_i1) {
                  int32_t v25_a = v22_lead + (v21_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v26_data;
                  v26_data.copy_from(glb_m1 + (v25_a));
                  r0.template select<16, 1>(v25_a) = v26_data;
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v28_ld;
              v28_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v28_ld);
              tensorforge::intel_esimd::simd<float, 64> v29_ld;
              v29_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v29_ld);
              tensorforge::intel_esimd::simd<float, 32> v30_ld;
              v30_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 128));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 2 * 0 + 128), v30_ld);
              tensorforge::intel_esimd::simd<float, 16> v31_ld;
              v31_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 160));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 160), v31_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 176> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 16), (0, 11)] [(0, 16)]
              tensorforge::intel_esimd::simd<float, 176> ir1(0.0f);
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
              tensorforge::intel_esimd::simd<float, 16> v50_acc{};
              tensorforge::intel_esimd::simd<float, 16> v54_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v50_acc += ((static_cast<float>(v54_data[0])) * v34_data);
              v50_acc += ((static_cast<float>(v54_data[1])) * v35_data);
              v50_acc += ((static_cast<float>(v54_data[2])) * v36_data);
              v50_acc += ((static_cast<float>(v54_data[3])) * v37_data);
              v50_acc += ((static_cast<float>(v54_data[4])) * v38_data);
              v50_acc += ((static_cast<float>(v54_data[5])) * v39_data);
              v50_acc += ((static_cast<float>(v54_data[6])) * v40_data);
              v50_acc += ((static_cast<float>(v54_data[7])) * v41_data);
              v50_acc += ((static_cast<float>(v54_data[8])) * v42_data);
              v50_acc += ((static_cast<float>(v54_data[9])) * v43_data);
              v50_acc += ((static_cast<float>(v54_data[10])) * v44_data);
              v50_acc += ((static_cast<float>(v54_data[11])) * v45_data);
              v50_acc += ((static_cast<float>(v54_data[12])) * v46_data);
              v50_acc += ((static_cast<float>(v54_data[13])) * v47_data);
              v50_acc += ((static_cast<float>(v54_data[14])) * v48_data);
              v50_acc += ((static_cast<float>(v54_data[15])) * v49_data);
              ir1.template select<16, 1>(0) = v50_acc;
              tensorforge::intel_esimd::simd<float, 16> v87_acc{};
              tensorforge::intel_esimd::simd<float, 16> v89_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v87_acc += ((static_cast<float>(v89_data[0])) * v34_data);
              v87_acc += ((static_cast<float>(v89_data[1])) * v35_data);
              v87_acc += ((static_cast<float>(v89_data[2])) * v36_data);
              v87_acc += ((static_cast<float>(v89_data[3])) * v37_data);
              v87_acc += ((static_cast<float>(v89_data[4])) * v38_data);
              v87_acc += ((static_cast<float>(v89_data[5])) * v39_data);
              v87_acc += ((static_cast<float>(v89_data[6])) * v40_data);
              v87_acc += ((static_cast<float>(v89_data[7])) * v41_data);
              v87_acc += ((static_cast<float>(v89_data[8])) * v42_data);
              v87_acc += ((static_cast<float>(v89_data[9])) * v43_data);
              v87_acc += ((static_cast<float>(v89_data[10])) * v44_data);
              v87_acc += ((static_cast<float>(v89_data[11])) * v45_data);
              v87_acc += ((static_cast<float>(v89_data[12])) * v46_data);
              v87_acc += ((static_cast<float>(v89_data[13])) * v47_data);
              v87_acc += ((static_cast<float>(v89_data[14])) * v48_data);
              v87_acc += ((static_cast<float>(v89_data[15])) * v49_data);
              ir1.template select<16, 1>(16) = v87_acc;
              tensorforge::intel_esimd::simd<float, 16> v122_acc{};
              tensorforge::intel_esimd::simd<float, 16> v124_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v122_acc += ((static_cast<float>(v124_data[0])) * v34_data);
              v122_acc += ((static_cast<float>(v124_data[1])) * v35_data);
              v122_acc += ((static_cast<float>(v124_data[2])) * v36_data);
              v122_acc += ((static_cast<float>(v124_data[3])) * v37_data);
              v122_acc += ((static_cast<float>(v124_data[4])) * v38_data);
              v122_acc += ((static_cast<float>(v124_data[5])) * v39_data);
              v122_acc += ((static_cast<float>(v124_data[6])) * v40_data);
              v122_acc += ((static_cast<float>(v124_data[7])) * v41_data);
              v122_acc += ((static_cast<float>(v124_data[8])) * v42_data);
              v122_acc += ((static_cast<float>(v124_data[9])) * v43_data);
              v122_acc += ((static_cast<float>(v124_data[10])) * v44_data);
              v122_acc += ((static_cast<float>(v124_data[11])) * v45_data);
              v122_acc += ((static_cast<float>(v124_data[12])) * v46_data);
              v122_acc += ((static_cast<float>(v124_data[13])) * v47_data);
              v122_acc += ((static_cast<float>(v124_data[14])) * v48_data);
              v122_acc += ((static_cast<float>(v124_data[15])) * v49_data);
              ir1.template select<16, 1>(32) = v122_acc;
              tensorforge::intel_esimd::simd<float, 16> v157_acc{};
              tensorforge::intel_esimd::simd<float, 16> v159_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v157_acc += ((static_cast<float>(v159_data[0])) * v34_data);
              v157_acc += ((static_cast<float>(v159_data[1])) * v35_data);
              v157_acc += ((static_cast<float>(v159_data[2])) * v36_data);
              v157_acc += ((static_cast<float>(v159_data[3])) * v37_data);
              v157_acc += ((static_cast<float>(v159_data[4])) * v38_data);
              v157_acc += ((static_cast<float>(v159_data[5])) * v39_data);
              v157_acc += ((static_cast<float>(v159_data[6])) * v40_data);
              v157_acc += ((static_cast<float>(v159_data[7])) * v41_data);
              v157_acc += ((static_cast<float>(v159_data[8])) * v42_data);
              v157_acc += ((static_cast<float>(v159_data[9])) * v43_data);
              v157_acc += ((static_cast<float>(v159_data[10])) * v44_data);
              v157_acc += ((static_cast<float>(v159_data[11])) * v45_data);
              v157_acc += ((static_cast<float>(v159_data[12])) * v46_data);
              v157_acc += ((static_cast<float>(v159_data[13])) * v47_data);
              v157_acc += ((static_cast<float>(v159_data[14])) * v48_data);
              v157_acc += ((static_cast<float>(v159_data[15])) * v49_data);
              ir1.template select<16, 1>(48) = v157_acc;
              tensorforge::intel_esimd::simd<float, 16> v192_acc{};
              tensorforge::intel_esimd::simd<float, 16> v194_data = tensorforge::slmLoad<float, 16>(s0 + (64_i32));
              v192_acc += ((static_cast<float>(v194_data[0])) * v34_data);
              v192_acc += ((static_cast<float>(v194_data[1])) * v35_data);
              v192_acc += ((static_cast<float>(v194_data[2])) * v36_data);
              v192_acc += ((static_cast<float>(v194_data[3])) * v37_data);
              v192_acc += ((static_cast<float>(v194_data[4])) * v38_data);
              v192_acc += ((static_cast<float>(v194_data[5])) * v39_data);
              v192_acc += ((static_cast<float>(v194_data[6])) * v40_data);
              v192_acc += ((static_cast<float>(v194_data[7])) * v41_data);
              v192_acc += ((static_cast<float>(v194_data[8])) * v42_data);
              v192_acc += ((static_cast<float>(v194_data[9])) * v43_data);
              v192_acc += ((static_cast<float>(v194_data[10])) * v44_data);
              v192_acc += ((static_cast<float>(v194_data[11])) * v45_data);
              v192_acc += ((static_cast<float>(v194_data[12])) * v46_data);
              v192_acc += ((static_cast<float>(v194_data[13])) * v47_data);
              v192_acc += ((static_cast<float>(v194_data[14])) * v48_data);
              v192_acc += ((static_cast<float>(v194_data[15])) * v49_data);
              ir1.template select<16, 1>(64) = v192_acc;
              tensorforge::intel_esimd::simd<float, 16> v227_acc{};
              tensorforge::intel_esimd::simd<float, 16> v229_data = tensorforge::slmLoad<float, 16>(s0 + (80_i32));
              v227_acc += ((static_cast<float>(v229_data[0])) * v34_data);
              v227_acc += ((static_cast<float>(v229_data[1])) * v35_data);
              v227_acc += ((static_cast<float>(v229_data[2])) * v36_data);
              v227_acc += ((static_cast<float>(v229_data[3])) * v37_data);
              v227_acc += ((static_cast<float>(v229_data[4])) * v38_data);
              v227_acc += ((static_cast<float>(v229_data[5])) * v39_data);
              v227_acc += ((static_cast<float>(v229_data[6])) * v40_data);
              v227_acc += ((static_cast<float>(v229_data[7])) * v41_data);
              v227_acc += ((static_cast<float>(v229_data[8])) * v42_data);
              v227_acc += ((static_cast<float>(v229_data[9])) * v43_data);
              v227_acc += ((static_cast<float>(v229_data[10])) * v44_data);
              v227_acc += ((static_cast<float>(v229_data[11])) * v45_data);
              v227_acc += ((static_cast<float>(v229_data[12])) * v46_data);
              v227_acc += ((static_cast<float>(v229_data[13])) * v47_data);
              v227_acc += ((static_cast<float>(v229_data[14])) * v48_data);
              v227_acc += ((static_cast<float>(v229_data[15])) * v49_data);
              ir1.template select<16, 1>(80) = v227_acc;
              tensorforge::intel_esimd::simd<float, 16> v262_acc{};
              tensorforge::intel_esimd::simd<float, 16> v264_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              v262_acc += ((static_cast<float>(v264_data[0])) * v34_data);
              v262_acc += ((static_cast<float>(v264_data[1])) * v35_data);
              v262_acc += ((static_cast<float>(v264_data[2])) * v36_data);
              v262_acc += ((static_cast<float>(v264_data[3])) * v37_data);
              v262_acc += ((static_cast<float>(v264_data[4])) * v38_data);
              v262_acc += ((static_cast<float>(v264_data[5])) * v39_data);
              v262_acc += ((static_cast<float>(v264_data[6])) * v40_data);
              v262_acc += ((static_cast<float>(v264_data[7])) * v41_data);
              v262_acc += ((static_cast<float>(v264_data[8])) * v42_data);
              v262_acc += ((static_cast<float>(v264_data[9])) * v43_data);
              v262_acc += ((static_cast<float>(v264_data[10])) * v44_data);
              v262_acc += ((static_cast<float>(v264_data[11])) * v45_data);
              v262_acc += ((static_cast<float>(v264_data[12])) * v46_data);
              v262_acc += ((static_cast<float>(v264_data[13])) * v47_data);
              v262_acc += ((static_cast<float>(v264_data[14])) * v48_data);
              v262_acc += ((static_cast<float>(v264_data[15])) * v49_data);
              ir1.template select<16, 1>(96) = v262_acc;
              tensorforge::intel_esimd::simd<float, 16> v297_acc{};
              tensorforge::intel_esimd::simd<float, 16> v299_data = tensorforge::slmLoad<float, 16>(s0 + (112_i32));
              v297_acc += ((static_cast<float>(v299_data[0])) * v34_data);
              v297_acc += ((static_cast<float>(v299_data[1])) * v35_data);
              v297_acc += ((static_cast<float>(v299_data[2])) * v36_data);
              v297_acc += ((static_cast<float>(v299_data[3])) * v37_data);
              v297_acc += ((static_cast<float>(v299_data[4])) * v38_data);
              v297_acc += ((static_cast<float>(v299_data[5])) * v39_data);
              v297_acc += ((static_cast<float>(v299_data[6])) * v40_data);
              v297_acc += ((static_cast<float>(v299_data[7])) * v41_data);
              v297_acc += ((static_cast<float>(v299_data[8])) * v42_data);
              v297_acc += ((static_cast<float>(v299_data[9])) * v43_data);
              v297_acc += ((static_cast<float>(v299_data[10])) * v44_data);
              v297_acc += ((static_cast<float>(v299_data[11])) * v45_data);
              v297_acc += ((static_cast<float>(v299_data[12])) * v46_data);
              v297_acc += ((static_cast<float>(v299_data[13])) * v47_data);
              v297_acc += ((static_cast<float>(v299_data[14])) * v48_data);
              v297_acc += ((static_cast<float>(v299_data[15])) * v49_data);
              ir1.template select<16, 1>(112) = v297_acc;
              tensorforge::intel_esimd::simd<float, 16> v332_acc{};
              tensorforge::intel_esimd::simd<float, 16> v334_data = tensorforge::slmLoad<float, 16>(s0 + (128_i32));
              v332_acc += ((static_cast<float>(v334_data[0])) * v34_data);
              v332_acc += ((static_cast<float>(v334_data[1])) * v35_data);
              v332_acc += ((static_cast<float>(v334_data[2])) * v36_data);
              v332_acc += ((static_cast<float>(v334_data[3])) * v37_data);
              v332_acc += ((static_cast<float>(v334_data[4])) * v38_data);
              v332_acc += ((static_cast<float>(v334_data[5])) * v39_data);
              v332_acc += ((static_cast<float>(v334_data[6])) * v40_data);
              v332_acc += ((static_cast<float>(v334_data[7])) * v41_data);
              v332_acc += ((static_cast<float>(v334_data[8])) * v42_data);
              v332_acc += ((static_cast<float>(v334_data[9])) * v43_data);
              v332_acc += ((static_cast<float>(v334_data[10])) * v44_data);
              v332_acc += ((static_cast<float>(v334_data[11])) * v45_data);
              v332_acc += ((static_cast<float>(v334_data[12])) * v46_data);
              v332_acc += ((static_cast<float>(v334_data[13])) * v47_data);
              v332_acc += ((static_cast<float>(v334_data[14])) * v48_data);
              v332_acc += ((static_cast<float>(v334_data[15])) * v49_data);
              ir1.template select<16, 1>(128) = v332_acc;
              tensorforge::intel_esimd::simd<float, 16> v367_acc{};
              tensorforge::intel_esimd::simd<float, 16> v369_data = tensorforge::slmLoad<float, 16>(s0 + (144_i32));
              v367_acc += ((static_cast<float>(v369_data[0])) * v34_data);
              v367_acc += ((static_cast<float>(v369_data[1])) * v35_data);
              v367_acc += ((static_cast<float>(v369_data[2])) * v36_data);
              v367_acc += ((static_cast<float>(v369_data[3])) * v37_data);
              v367_acc += ((static_cast<float>(v369_data[4])) * v38_data);
              v367_acc += ((static_cast<float>(v369_data[5])) * v39_data);
              v367_acc += ((static_cast<float>(v369_data[6])) * v40_data);
              v367_acc += ((static_cast<float>(v369_data[7])) * v41_data);
              v367_acc += ((static_cast<float>(v369_data[8])) * v42_data);
              v367_acc += ((static_cast<float>(v369_data[9])) * v43_data);
              v367_acc += ((static_cast<float>(v369_data[10])) * v44_data);
              v367_acc += ((static_cast<float>(v369_data[11])) * v45_data);
              v367_acc += ((static_cast<float>(v369_data[12])) * v46_data);
              v367_acc += ((static_cast<float>(v369_data[13])) * v47_data);
              v367_acc += ((static_cast<float>(v369_data[14])) * v48_data);
              v367_acc += ((static_cast<float>(v369_data[15])) * v49_data);
              ir1.template select<16, 1>(144) = v367_acc;
              tensorforge::intel_esimd::simd<float, 16> v402_acc{};
              tensorforge::intel_esimd::simd<float, 16> v404_data = tensorforge::slmLoad<float, 16>(s0 + (160_i32));
              v402_acc += ((static_cast<float>(v404_data[0])) * v34_data);
              v402_acc += ((static_cast<float>(v404_data[1])) * v35_data);
              v402_acc += ((static_cast<float>(v404_data[2])) * v36_data);
              v402_acc += ((static_cast<float>(v404_data[3])) * v37_data);
              v402_acc += ((static_cast<float>(v404_data[4])) * v38_data);
              v402_acc += ((static_cast<float>(v404_data[5])) * v39_data);
              v402_acc += ((static_cast<float>(v404_data[6])) * v40_data);
              v402_acc += ((static_cast<float>(v404_data[7])) * v41_data);
              v402_acc += ((static_cast<float>(v404_data[8])) * v42_data);
              v402_acc += ((static_cast<float>(v404_data[9])) * v43_data);
              v402_acc += ((static_cast<float>(v404_data[10])) * v44_data);
              v402_acc += ((static_cast<float>(v404_data[11])) * v45_data);
              v402_acc += ((static_cast<float>(v404_data[12])) * v46_data);
              v402_acc += ((static_cast<float>(v404_data[13])) * v47_data);
              v402_acc += ((static_cast<float>(v404_data[14])) * v48_data);
              v402_acc += ((static_cast<float>(v404_data[15])) * v49_data);
              ir1.template select<16, 1>(160) = v402_acc;
              // r1 = ir1
              #pragma unroll
              for (int32_t v437_n0 = 0; v437_n0 < 1; ++v437_n0) {
                int32_t v439_a = v437_n0 * 16;
                #pragma unroll
                for (int32_t v438_n1 = 0; v438_n1 < 11; ++v438_n1) {
                  int32_t v441_a = v439_a + (v438_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v442_data(ir1.template select<16, 1>(v441_a));
                  r1.template select<16, 1>(v441_a) = v442_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v443_i0 = 0; v443_i0 < 1; ++v443_i0) {
                int32_t v445_a = v443_i0 * 16;
                #pragma unroll
                for (int32_t v444_i1 = 0; v444_i1 < 11; ++v444_i1) {
                  int32_t v447_a = v445_a + (v444_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v448_data(r1.template select<16, 1>(v447_a));
                  v448_data.copy_to(glb_m0 + (v447_a));
                }
              }
            }
          }
        }
      }
    });
  });
}

