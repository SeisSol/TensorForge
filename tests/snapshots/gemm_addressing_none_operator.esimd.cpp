// === base name ===
kernel_e77cd79cee439305

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_e77cd79cee439305 = {{1, 16, 1}, 16, 16, 1, 16, 17408, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_e77cd79cee439305(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_e77cd79cee439305(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_e77cd79cee439305(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 4352 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_e77cd79cee439305(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_e77cd79cee439305(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_e77cd79cee439305(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_e77cd79cee439305(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<4352 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 17408 B shared, occupancy grid
        // operands:
        //   m0 16×16(16×16) {0..16}×{0..16} strided
        //   m1 16×16(16×16) {0..16}×{0..16} none
        //   m2 16×16(16×16) {0..16}×{0..16} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":4352}],"shared_bytes":17408,"shared_elements":4352,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (272 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (256);
          const float *const __restrict__ glb_m1 = &m1[0];
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v6_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v6_batchId0 < numElements0; v6_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v7_ahead1 = v6_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
            const float *const __restrict__ pf_glb_m2 = &m2[v9_batchId1 * 256 + 0 + m2_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 256 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v6_batchId0 * 256 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v17_ld;
              v17_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v17_ld);
              tensorforge::intel_esimd::simd<float, 64> v18_ld;
              v18_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v18_ld);
              tensorforge::intel_esimd::simd<float, 64> v19_ld;
              v19_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 128));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 128), v19_ld);
              tensorforge::intel_esimd::simd<float, 64> v20_ld;
              v20_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 192));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 192), v20_ld);
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
              // r0 = +(glb_m1 * s0) + None
              // [(0, 16), (0, 16)] [(0, 16)]
              tensorforge::intel_esimd::simd<float, 256> ir0(0.0f);
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run0;
              glb_m1_run0.copy_from(glb_m1 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v26_data(glb_m1_run0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v28_data(glb_m1_run0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v30_data(glb_m1_run0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v32_data(glb_m1_run0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run1;
              glb_m1_run1.copy_from(glb_m1 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v34_data(glb_m1_run1.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v36_data(glb_m1_run1.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v38_data(glb_m1_run1.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v40_data(glb_m1_run1.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run2;
              glb_m1_run2.copy_from(glb_m1 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v42_data(glb_m1_run2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v44_data(glb_m1_run2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v46_data(glb_m1_run2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v48_data(glb_m1_run2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m1_run3;
              glb_m1_run3.copy_from(glb_m1 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v50_data(glb_m1_run3.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v52_data(glb_m1_run3.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v54_data(glb_m1_run3.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v56_data(glb_m1_run3.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v57_acc{};
              tensorforge::intel_esimd::simd<float, 16> v58_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v57_acc += ((static_cast<float>(v58_data[0])) * v26_data);
              v57_acc += ((static_cast<float>(v58_data[1])) * v28_data);
              v57_acc += ((static_cast<float>(v58_data[2])) * v30_data);
              v57_acc += ((static_cast<float>(v58_data[3])) * v32_data);
              v57_acc += ((static_cast<float>(v58_data[4])) * v34_data);
              v57_acc += ((static_cast<float>(v58_data[5])) * v36_data);
              v57_acc += ((static_cast<float>(v58_data[6])) * v38_data);
              v57_acc += ((static_cast<float>(v58_data[7])) * v40_data);
              v57_acc += ((static_cast<float>(v58_data[8])) * v42_data);
              v57_acc += ((static_cast<float>(v58_data[9])) * v44_data);
              v57_acc += ((static_cast<float>(v58_data[10])) * v46_data);
              v57_acc += ((static_cast<float>(v58_data[11])) * v48_data);
              v57_acc += ((static_cast<float>(v58_data[12])) * v50_data);
              v57_acc += ((static_cast<float>(v58_data[13])) * v52_data);
              v57_acc += ((static_cast<float>(v58_data[14])) * v54_data);
              v57_acc += ((static_cast<float>(v58_data[15])) * v56_data);
              ir0.template select<16, 1>(0) = v57_acc;
              tensorforge::intel_esimd::simd<float, 16> v91_acc{};
              tensorforge::intel_esimd::simd<float, 16> v92_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v91_acc += ((static_cast<float>(v92_data[0])) * v26_data);
              v91_acc += ((static_cast<float>(v92_data[1])) * v28_data);
              v91_acc += ((static_cast<float>(v92_data[2])) * v30_data);
              v91_acc += ((static_cast<float>(v92_data[3])) * v32_data);
              v91_acc += ((static_cast<float>(v92_data[4])) * v34_data);
              v91_acc += ((static_cast<float>(v92_data[5])) * v36_data);
              v91_acc += ((static_cast<float>(v92_data[6])) * v38_data);
              v91_acc += ((static_cast<float>(v92_data[7])) * v40_data);
              v91_acc += ((static_cast<float>(v92_data[8])) * v42_data);
              v91_acc += ((static_cast<float>(v92_data[9])) * v44_data);
              v91_acc += ((static_cast<float>(v92_data[10])) * v46_data);
              v91_acc += ((static_cast<float>(v92_data[11])) * v48_data);
              v91_acc += ((static_cast<float>(v92_data[12])) * v50_data);
              v91_acc += ((static_cast<float>(v92_data[13])) * v52_data);
              v91_acc += ((static_cast<float>(v92_data[14])) * v54_data);
              v91_acc += ((static_cast<float>(v92_data[15])) * v56_data);
              ir0.template select<16, 1>(16) = v91_acc;
              tensorforge::intel_esimd::simd<float, 16> v125_acc{};
              tensorforge::intel_esimd::simd<float, 16> v126_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v125_acc += ((static_cast<float>(v126_data[0])) * v26_data);
              v125_acc += ((static_cast<float>(v126_data[1])) * v28_data);
              v125_acc += ((static_cast<float>(v126_data[2])) * v30_data);
              v125_acc += ((static_cast<float>(v126_data[3])) * v32_data);
              v125_acc += ((static_cast<float>(v126_data[4])) * v34_data);
              v125_acc += ((static_cast<float>(v126_data[5])) * v36_data);
              v125_acc += ((static_cast<float>(v126_data[6])) * v38_data);
              v125_acc += ((static_cast<float>(v126_data[7])) * v40_data);
              v125_acc += ((static_cast<float>(v126_data[8])) * v42_data);
              v125_acc += ((static_cast<float>(v126_data[9])) * v44_data);
              v125_acc += ((static_cast<float>(v126_data[10])) * v46_data);
              v125_acc += ((static_cast<float>(v126_data[11])) * v48_data);
              v125_acc += ((static_cast<float>(v126_data[12])) * v50_data);
              v125_acc += ((static_cast<float>(v126_data[13])) * v52_data);
              v125_acc += ((static_cast<float>(v126_data[14])) * v54_data);
              v125_acc += ((static_cast<float>(v126_data[15])) * v56_data);
              ir0.template select<16, 1>(32) = v125_acc;
              tensorforge::intel_esimd::simd<float, 16> v159_acc{};
              tensorforge::intel_esimd::simd<float, 16> v160_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v159_acc += ((static_cast<float>(v160_data[0])) * v26_data);
              v159_acc += ((static_cast<float>(v160_data[1])) * v28_data);
              v159_acc += ((static_cast<float>(v160_data[2])) * v30_data);
              v159_acc += ((static_cast<float>(v160_data[3])) * v32_data);
              v159_acc += ((static_cast<float>(v160_data[4])) * v34_data);
              v159_acc += ((static_cast<float>(v160_data[5])) * v36_data);
              v159_acc += ((static_cast<float>(v160_data[6])) * v38_data);
              v159_acc += ((static_cast<float>(v160_data[7])) * v40_data);
              v159_acc += ((static_cast<float>(v160_data[8])) * v42_data);
              v159_acc += ((static_cast<float>(v160_data[9])) * v44_data);
              v159_acc += ((static_cast<float>(v160_data[10])) * v46_data);
              v159_acc += ((static_cast<float>(v160_data[11])) * v48_data);
              v159_acc += ((static_cast<float>(v160_data[12])) * v50_data);
              v159_acc += ((static_cast<float>(v160_data[13])) * v52_data);
              v159_acc += ((static_cast<float>(v160_data[14])) * v54_data);
              v159_acc += ((static_cast<float>(v160_data[15])) * v56_data);
              ir0.template select<16, 1>(48) = v159_acc;
              tensorforge::intel_esimd::simd<float, 16> v193_acc{};
              tensorforge::intel_esimd::simd<float, 16> v194_data = tensorforge::slmLoad<float, 16>(s0 + (64_i32));
              v193_acc += ((static_cast<float>(v194_data[0])) * v26_data);
              v193_acc += ((static_cast<float>(v194_data[1])) * v28_data);
              v193_acc += ((static_cast<float>(v194_data[2])) * v30_data);
              v193_acc += ((static_cast<float>(v194_data[3])) * v32_data);
              v193_acc += ((static_cast<float>(v194_data[4])) * v34_data);
              v193_acc += ((static_cast<float>(v194_data[5])) * v36_data);
              v193_acc += ((static_cast<float>(v194_data[6])) * v38_data);
              v193_acc += ((static_cast<float>(v194_data[7])) * v40_data);
              v193_acc += ((static_cast<float>(v194_data[8])) * v42_data);
              v193_acc += ((static_cast<float>(v194_data[9])) * v44_data);
              v193_acc += ((static_cast<float>(v194_data[10])) * v46_data);
              v193_acc += ((static_cast<float>(v194_data[11])) * v48_data);
              v193_acc += ((static_cast<float>(v194_data[12])) * v50_data);
              v193_acc += ((static_cast<float>(v194_data[13])) * v52_data);
              v193_acc += ((static_cast<float>(v194_data[14])) * v54_data);
              v193_acc += ((static_cast<float>(v194_data[15])) * v56_data);
              ir0.template select<16, 1>(64) = v193_acc;
              tensorforge::intel_esimd::simd<float, 16> v227_acc{};
              tensorforge::intel_esimd::simd<float, 16> v228_data = tensorforge::slmLoad<float, 16>(s0 + (80_i32));
              v227_acc += ((static_cast<float>(v228_data[0])) * v26_data);
              v227_acc += ((static_cast<float>(v228_data[1])) * v28_data);
              v227_acc += ((static_cast<float>(v228_data[2])) * v30_data);
              v227_acc += ((static_cast<float>(v228_data[3])) * v32_data);
              v227_acc += ((static_cast<float>(v228_data[4])) * v34_data);
              v227_acc += ((static_cast<float>(v228_data[5])) * v36_data);
              v227_acc += ((static_cast<float>(v228_data[6])) * v38_data);
              v227_acc += ((static_cast<float>(v228_data[7])) * v40_data);
              v227_acc += ((static_cast<float>(v228_data[8])) * v42_data);
              v227_acc += ((static_cast<float>(v228_data[9])) * v44_data);
              v227_acc += ((static_cast<float>(v228_data[10])) * v46_data);
              v227_acc += ((static_cast<float>(v228_data[11])) * v48_data);
              v227_acc += ((static_cast<float>(v228_data[12])) * v50_data);
              v227_acc += ((static_cast<float>(v228_data[13])) * v52_data);
              v227_acc += ((static_cast<float>(v228_data[14])) * v54_data);
              v227_acc += ((static_cast<float>(v228_data[15])) * v56_data);
              ir0.template select<16, 1>(80) = v227_acc;
              tensorforge::intel_esimd::simd<float, 16> v261_acc{};
              tensorforge::intel_esimd::simd<float, 16> v262_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              v261_acc += ((static_cast<float>(v262_data[0])) * v26_data);
              v261_acc += ((static_cast<float>(v262_data[1])) * v28_data);
              v261_acc += ((static_cast<float>(v262_data[2])) * v30_data);
              v261_acc += ((static_cast<float>(v262_data[3])) * v32_data);
              v261_acc += ((static_cast<float>(v262_data[4])) * v34_data);
              v261_acc += ((static_cast<float>(v262_data[5])) * v36_data);
              v261_acc += ((static_cast<float>(v262_data[6])) * v38_data);
              v261_acc += ((static_cast<float>(v262_data[7])) * v40_data);
              v261_acc += ((static_cast<float>(v262_data[8])) * v42_data);
              v261_acc += ((static_cast<float>(v262_data[9])) * v44_data);
              v261_acc += ((static_cast<float>(v262_data[10])) * v46_data);
              v261_acc += ((static_cast<float>(v262_data[11])) * v48_data);
              v261_acc += ((static_cast<float>(v262_data[12])) * v50_data);
              v261_acc += ((static_cast<float>(v262_data[13])) * v52_data);
              v261_acc += ((static_cast<float>(v262_data[14])) * v54_data);
              v261_acc += ((static_cast<float>(v262_data[15])) * v56_data);
              ir0.template select<16, 1>(96) = v261_acc;
              tensorforge::intel_esimd::simd<float, 16> v295_acc{};
              tensorforge::intel_esimd::simd<float, 16> v296_data = tensorforge::slmLoad<float, 16>(s0 + (112_i32));
              v295_acc += ((static_cast<float>(v296_data[0])) * v26_data);
              v295_acc += ((static_cast<float>(v296_data[1])) * v28_data);
              v295_acc += ((static_cast<float>(v296_data[2])) * v30_data);
              v295_acc += ((static_cast<float>(v296_data[3])) * v32_data);
              v295_acc += ((static_cast<float>(v296_data[4])) * v34_data);
              v295_acc += ((static_cast<float>(v296_data[5])) * v36_data);
              v295_acc += ((static_cast<float>(v296_data[6])) * v38_data);
              v295_acc += ((static_cast<float>(v296_data[7])) * v40_data);
              v295_acc += ((static_cast<float>(v296_data[8])) * v42_data);
              v295_acc += ((static_cast<float>(v296_data[9])) * v44_data);
              v295_acc += ((static_cast<float>(v296_data[10])) * v46_data);
              v295_acc += ((static_cast<float>(v296_data[11])) * v48_data);
              v295_acc += ((static_cast<float>(v296_data[12])) * v50_data);
              v295_acc += ((static_cast<float>(v296_data[13])) * v52_data);
              v295_acc += ((static_cast<float>(v296_data[14])) * v54_data);
              v295_acc += ((static_cast<float>(v296_data[15])) * v56_data);
              ir0.template select<16, 1>(112) = v295_acc;
              tensorforge::intel_esimd::simd<float, 16> v329_acc{};
              tensorforge::intel_esimd::simd<float, 16> v330_data = tensorforge::slmLoad<float, 16>(s0 + (128_i32));
              v329_acc += ((static_cast<float>(v330_data[0])) * v26_data);
              v329_acc += ((static_cast<float>(v330_data[1])) * v28_data);
              v329_acc += ((static_cast<float>(v330_data[2])) * v30_data);
              v329_acc += ((static_cast<float>(v330_data[3])) * v32_data);
              v329_acc += ((static_cast<float>(v330_data[4])) * v34_data);
              v329_acc += ((static_cast<float>(v330_data[5])) * v36_data);
              v329_acc += ((static_cast<float>(v330_data[6])) * v38_data);
              v329_acc += ((static_cast<float>(v330_data[7])) * v40_data);
              v329_acc += ((static_cast<float>(v330_data[8])) * v42_data);
              v329_acc += ((static_cast<float>(v330_data[9])) * v44_data);
              v329_acc += ((static_cast<float>(v330_data[10])) * v46_data);
              v329_acc += ((static_cast<float>(v330_data[11])) * v48_data);
              v329_acc += ((static_cast<float>(v330_data[12])) * v50_data);
              v329_acc += ((static_cast<float>(v330_data[13])) * v52_data);
              v329_acc += ((static_cast<float>(v330_data[14])) * v54_data);
              v329_acc += ((static_cast<float>(v330_data[15])) * v56_data);
              ir0.template select<16, 1>(128) = v329_acc;
              tensorforge::intel_esimd::simd<float, 16> v363_acc{};
              tensorforge::intel_esimd::simd<float, 16> v364_data = tensorforge::slmLoad<float, 16>(s0 + (144_i32));
              v363_acc += ((static_cast<float>(v364_data[0])) * v26_data);
              v363_acc += ((static_cast<float>(v364_data[1])) * v28_data);
              v363_acc += ((static_cast<float>(v364_data[2])) * v30_data);
              v363_acc += ((static_cast<float>(v364_data[3])) * v32_data);
              v363_acc += ((static_cast<float>(v364_data[4])) * v34_data);
              v363_acc += ((static_cast<float>(v364_data[5])) * v36_data);
              v363_acc += ((static_cast<float>(v364_data[6])) * v38_data);
              v363_acc += ((static_cast<float>(v364_data[7])) * v40_data);
              v363_acc += ((static_cast<float>(v364_data[8])) * v42_data);
              v363_acc += ((static_cast<float>(v364_data[9])) * v44_data);
              v363_acc += ((static_cast<float>(v364_data[10])) * v46_data);
              v363_acc += ((static_cast<float>(v364_data[11])) * v48_data);
              v363_acc += ((static_cast<float>(v364_data[12])) * v50_data);
              v363_acc += ((static_cast<float>(v364_data[13])) * v52_data);
              v363_acc += ((static_cast<float>(v364_data[14])) * v54_data);
              v363_acc += ((static_cast<float>(v364_data[15])) * v56_data);
              ir0.template select<16, 1>(144) = v363_acc;
              tensorforge::intel_esimd::simd<float, 16> v397_acc{};
              tensorforge::intel_esimd::simd<float, 16> v398_data = tensorforge::slmLoad<float, 16>(s0 + (160_i32));
              v397_acc += ((static_cast<float>(v398_data[0])) * v26_data);
              v397_acc += ((static_cast<float>(v398_data[1])) * v28_data);
              v397_acc += ((static_cast<float>(v398_data[2])) * v30_data);
              v397_acc += ((static_cast<float>(v398_data[3])) * v32_data);
              v397_acc += ((static_cast<float>(v398_data[4])) * v34_data);
              v397_acc += ((static_cast<float>(v398_data[5])) * v36_data);
              v397_acc += ((static_cast<float>(v398_data[6])) * v38_data);
              v397_acc += ((static_cast<float>(v398_data[7])) * v40_data);
              v397_acc += ((static_cast<float>(v398_data[8])) * v42_data);
              v397_acc += ((static_cast<float>(v398_data[9])) * v44_data);
              v397_acc += ((static_cast<float>(v398_data[10])) * v46_data);
              v397_acc += ((static_cast<float>(v398_data[11])) * v48_data);
              v397_acc += ((static_cast<float>(v398_data[12])) * v50_data);
              v397_acc += ((static_cast<float>(v398_data[13])) * v52_data);
              v397_acc += ((static_cast<float>(v398_data[14])) * v54_data);
              v397_acc += ((static_cast<float>(v398_data[15])) * v56_data);
              ir0.template select<16, 1>(160) = v397_acc;
              tensorforge::intel_esimd::simd<float, 16> v431_acc{};
              tensorforge::intel_esimd::simd<float, 16> v432_data = tensorforge::slmLoad<float, 16>(s0 + (176_i32));
              v431_acc += ((static_cast<float>(v432_data[0])) * v26_data);
              v431_acc += ((static_cast<float>(v432_data[1])) * v28_data);
              v431_acc += ((static_cast<float>(v432_data[2])) * v30_data);
              v431_acc += ((static_cast<float>(v432_data[3])) * v32_data);
              v431_acc += ((static_cast<float>(v432_data[4])) * v34_data);
              v431_acc += ((static_cast<float>(v432_data[5])) * v36_data);
              v431_acc += ((static_cast<float>(v432_data[6])) * v38_data);
              v431_acc += ((static_cast<float>(v432_data[7])) * v40_data);
              v431_acc += ((static_cast<float>(v432_data[8])) * v42_data);
              v431_acc += ((static_cast<float>(v432_data[9])) * v44_data);
              v431_acc += ((static_cast<float>(v432_data[10])) * v46_data);
              v431_acc += ((static_cast<float>(v432_data[11])) * v48_data);
              v431_acc += ((static_cast<float>(v432_data[12])) * v50_data);
              v431_acc += ((static_cast<float>(v432_data[13])) * v52_data);
              v431_acc += ((static_cast<float>(v432_data[14])) * v54_data);
              v431_acc += ((static_cast<float>(v432_data[15])) * v56_data);
              ir0.template select<16, 1>(176) = v431_acc;
              tensorforge::intel_esimd::simd<float, 16> v465_acc{};
              tensorforge::intel_esimd::simd<float, 16> v466_data = tensorforge::slmLoad<float, 16>(s0 + (192_i32));
              v465_acc += ((static_cast<float>(v466_data[0])) * v26_data);
              v465_acc += ((static_cast<float>(v466_data[1])) * v28_data);
              v465_acc += ((static_cast<float>(v466_data[2])) * v30_data);
              v465_acc += ((static_cast<float>(v466_data[3])) * v32_data);
              v465_acc += ((static_cast<float>(v466_data[4])) * v34_data);
              v465_acc += ((static_cast<float>(v466_data[5])) * v36_data);
              v465_acc += ((static_cast<float>(v466_data[6])) * v38_data);
              v465_acc += ((static_cast<float>(v466_data[7])) * v40_data);
              v465_acc += ((static_cast<float>(v466_data[8])) * v42_data);
              v465_acc += ((static_cast<float>(v466_data[9])) * v44_data);
              v465_acc += ((static_cast<float>(v466_data[10])) * v46_data);
              v465_acc += ((static_cast<float>(v466_data[11])) * v48_data);
              v465_acc += ((static_cast<float>(v466_data[12])) * v50_data);
              v465_acc += ((static_cast<float>(v466_data[13])) * v52_data);
              v465_acc += ((static_cast<float>(v466_data[14])) * v54_data);
              v465_acc += ((static_cast<float>(v466_data[15])) * v56_data);
              ir0.template select<16, 1>(192) = v465_acc;
              tensorforge::intel_esimd::simd<float, 16> v499_acc{};
              tensorforge::intel_esimd::simd<float, 16> v500_data = tensorforge::slmLoad<float, 16>(s0 + (208_i32));
              v499_acc += ((static_cast<float>(v500_data[0])) * v26_data);
              v499_acc += ((static_cast<float>(v500_data[1])) * v28_data);
              v499_acc += ((static_cast<float>(v500_data[2])) * v30_data);
              v499_acc += ((static_cast<float>(v500_data[3])) * v32_data);
              v499_acc += ((static_cast<float>(v500_data[4])) * v34_data);
              v499_acc += ((static_cast<float>(v500_data[5])) * v36_data);
              v499_acc += ((static_cast<float>(v500_data[6])) * v38_data);
              v499_acc += ((static_cast<float>(v500_data[7])) * v40_data);
              v499_acc += ((static_cast<float>(v500_data[8])) * v42_data);
              v499_acc += ((static_cast<float>(v500_data[9])) * v44_data);
              v499_acc += ((static_cast<float>(v500_data[10])) * v46_data);
              v499_acc += ((static_cast<float>(v500_data[11])) * v48_data);
              v499_acc += ((static_cast<float>(v500_data[12])) * v50_data);
              v499_acc += ((static_cast<float>(v500_data[13])) * v52_data);
              v499_acc += ((static_cast<float>(v500_data[14])) * v54_data);
              v499_acc += ((static_cast<float>(v500_data[15])) * v56_data);
              ir0.template select<16, 1>(208) = v499_acc;
              tensorforge::intel_esimd::simd<float, 16> v533_acc{};
              tensorforge::intel_esimd::simd<float, 16> v534_data = tensorforge::slmLoad<float, 16>(s0 + (224_i32));
              v533_acc += ((static_cast<float>(v534_data[0])) * v26_data);
              v533_acc += ((static_cast<float>(v534_data[1])) * v28_data);
              v533_acc += ((static_cast<float>(v534_data[2])) * v30_data);
              v533_acc += ((static_cast<float>(v534_data[3])) * v32_data);
              v533_acc += ((static_cast<float>(v534_data[4])) * v34_data);
              v533_acc += ((static_cast<float>(v534_data[5])) * v36_data);
              v533_acc += ((static_cast<float>(v534_data[6])) * v38_data);
              v533_acc += ((static_cast<float>(v534_data[7])) * v40_data);
              v533_acc += ((static_cast<float>(v534_data[8])) * v42_data);
              v533_acc += ((static_cast<float>(v534_data[9])) * v44_data);
              v533_acc += ((static_cast<float>(v534_data[10])) * v46_data);
              v533_acc += ((static_cast<float>(v534_data[11])) * v48_data);
              v533_acc += ((static_cast<float>(v534_data[12])) * v50_data);
              v533_acc += ((static_cast<float>(v534_data[13])) * v52_data);
              v533_acc += ((static_cast<float>(v534_data[14])) * v54_data);
              v533_acc += ((static_cast<float>(v534_data[15])) * v56_data);
              ir0.template select<16, 1>(224) = v533_acc;
              tensorforge::intel_esimd::simd<float, 16> v567_acc{};
              tensorforge::intel_esimd::simd<float, 16> v568_data = tensorforge::slmLoad<float, 16>(s0 + (240_i32));
              v567_acc += ((static_cast<float>(v568_data[0])) * v26_data);
              v567_acc += ((static_cast<float>(v568_data[1])) * v28_data);
              v567_acc += ((static_cast<float>(v568_data[2])) * v30_data);
              v567_acc += ((static_cast<float>(v568_data[3])) * v32_data);
              v567_acc += ((static_cast<float>(v568_data[4])) * v34_data);
              v567_acc += ((static_cast<float>(v568_data[5])) * v36_data);
              v567_acc += ((static_cast<float>(v568_data[6])) * v38_data);
              v567_acc += ((static_cast<float>(v568_data[7])) * v40_data);
              v567_acc += ((static_cast<float>(v568_data[8])) * v42_data);
              v567_acc += ((static_cast<float>(v568_data[9])) * v44_data);
              v567_acc += ((static_cast<float>(v568_data[10])) * v46_data);
              v567_acc += ((static_cast<float>(v568_data[11])) * v48_data);
              v567_acc += ((static_cast<float>(v568_data[12])) * v50_data);
              v567_acc += ((static_cast<float>(v568_data[13])) * v52_data);
              v567_acc += ((static_cast<float>(v568_data[14])) * v54_data);
              v567_acc += ((static_cast<float>(v568_data[15])) * v56_data);
              ir0.template select<16, 1>(240) = v567_acc;
              #pragma unroll
              for (int32_t v601_n0 = 0; v601_n0 < 1; ++v601_n0) {
                int32_t v603_a = v601_n0 * 16;
                #pragma unroll
                for (int32_t v602_n1 = 0; v602_n1 < 16; ++v602_n1) {
                  int32_t v605_a = v603_a + (v602_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v606_data(ir0.template select<16, 1>(v605_a));
                  r0.template select<16, 1>(v605_a) = v606_data;
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v607_i0 = 0; v607_i0 < 1; ++v607_i0) {
                int32_t v609_a = v607_i0 * 16;
                #pragma unroll
                for (int32_t v608_i1 = 0; v608_i1 < 16; ++v608_i1) {
                  int32_t v611_a = v609_a + (v608_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v612_data(r0.template select<16, 1>(v611_a));
                  v612_data.copy_to(glb_m0 + (v611_a));
                }
              }
            }
            tensorforge::prefetchL2<256>(&pf_glb_m2[0]);
          }
        }
      }
    });
  });
}

