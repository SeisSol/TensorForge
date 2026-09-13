// === base name ===
kernel_8c4e66b142d97aea

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_8c4e66b142d97aea = {{1, 16, 1}, 16, 16, 1, 16, 17408, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_8c4e66b142d97aea(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_8c4e66b142d97aea(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_8c4e66b142d97aea(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_8c4e66b142d97aea(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_8c4e66b142d97aea(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_8c4e66b142d97aea(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_8c4e66b142d97aea(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<4352 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 17408 B shared, occupancy grid
        // operands:
        //   m0 16×16(16×16) {0..16}×{0..16} strided
        //   m1 16×16(16×16) {0..16}×{0..16} strided
        //   m2 16×16(16×16) {0..16}×{0..16} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":4352}],"shared_bytes":17408,"shared_elements":4352,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (272 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (256);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v5_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v5_batchId0 < numElements0; v5_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v6_ahead1 = v5_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
            const float *const __restrict__ pf_glb_m1 = &m1[v8_batchId1 * 256 + 0 + m1_extraOffset];
            const float *const __restrict__ pf_glb_m2 = &m2[v8_batchId1 * 256 + 0 + m2_extraOffset];
            float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 256 + 0 + m0_extraOffset];
            const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 256 + 0 + m1_extraOffset];
            const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 256 + 0 + m2_extraOffset];
            tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
            // r0 = load{g>r}(glb_m1);
            #pragma unroll
            for (int32_t v18_i0 = 0; v18_i0 < 1; ++v18_i0) {
              int32_t v20_lead = v18_i0 * 16;
              #pragma unroll
              for (int32_t v19_i1 = 0; v19_i1 < 16; ++v19_i1) {
                int32_t v23_a = v20_lead + (v19_i1 * 16);
                tensorforge::intel_esimd::simd<float, 16> v24_data;
                v24_data.copy_from(glb_m1 + (v23_a));
                r0.template select<16, 1>(v23_a) = v24_data;
              }
            }
            // s0 = load{g>s}(glb_m2[0, 1])
            tensorforge::intel_esimd::simd<float, 64> v26_ld;
            v26_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
            tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v26_ld);
            tensorforge::intel_esimd::simd<float, 64> v27_ld;
            v27_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
            tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v27_ld);
            tensorforge::intel_esimd::simd<float, 64> v28_ld;
            v28_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 128));
            tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 128), v28_ld);
            tensorforge::intel_esimd::simd<float, 64> v29_ld;
            v29_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 192));
            tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 192), v29_ld);
            // wait(r0 = load{g>r}(glb_m1););
            // wait(s0 = load{g>s}(glb_m2[0, 1]));
            tensorforge::intel_esimd::simd<float, 256> r1(0.0f);
            // r1 = +(r0 * s0) + None
            // [(0, 16), (0, 16)] [(0, 16)]
            tensorforge::intel_esimd::simd<float, 256> ir1(0.0f);
            tensorforge::intel_esimd::simd<float, 16> v32_data(r0.template select<16, 1>(0));
            tensorforge::intel_esimd::simd<float, 16> v33_data(r0.template select<16, 1>(16));
            tensorforge::intel_esimd::simd<float, 16> v34_data(r0.template select<16, 1>(32));
            tensorforge::intel_esimd::simd<float, 16> v35_data(r0.template select<16, 1>(48));
            tensorforge::intel_esimd::simd<float, 16> v36_data(r0.template select<16, 1>(64));
            tensorforge::intel_esimd::simd<float, 16> v37_data(r0.template select<16, 1>(80));
            tensorforge::intel_esimd::simd<float, 16> v38_data(r0.template select<16, 1>(96));
            tensorforge::intel_esimd::simd<float, 16> v39_data(r0.template select<16, 1>(112));
            tensorforge::intel_esimd::simd<float, 16> v40_data(r0.template select<16, 1>(128));
            tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(144));
            tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(160));
            tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(176));
            tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(192));
            tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(208));
            tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(224));
            tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(240));
            tensorforge::intel_esimd::simd<float, 16> v48_acc{};
            tensorforge::intel_esimd::simd<float, 16> v52_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
            v48_acc += ((static_cast<float>(v52_data[0])) * v32_data);
            v48_acc += ((static_cast<float>(v52_data[1])) * v33_data);
            v48_acc += ((static_cast<float>(v52_data[2])) * v34_data);
            v48_acc += ((static_cast<float>(v52_data[3])) * v35_data);
            v48_acc += ((static_cast<float>(v52_data[4])) * v36_data);
            v48_acc += ((static_cast<float>(v52_data[5])) * v37_data);
            v48_acc += ((static_cast<float>(v52_data[6])) * v38_data);
            v48_acc += ((static_cast<float>(v52_data[7])) * v39_data);
            v48_acc += ((static_cast<float>(v52_data[8])) * v40_data);
            v48_acc += ((static_cast<float>(v52_data[9])) * v41_data);
            v48_acc += ((static_cast<float>(v52_data[10])) * v42_data);
            v48_acc += ((static_cast<float>(v52_data[11])) * v43_data);
            v48_acc += ((static_cast<float>(v52_data[12])) * v44_data);
            v48_acc += ((static_cast<float>(v52_data[13])) * v45_data);
            v48_acc += ((static_cast<float>(v52_data[14])) * v46_data);
            v48_acc += ((static_cast<float>(v52_data[15])) * v47_data);
            ir1.template select<16, 1>(0) = v48_acc;
            tensorforge::intel_esimd::simd<float, 16> v85_acc{};
            tensorforge::intel_esimd::simd<float, 16> v87_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
            v85_acc += ((static_cast<float>(v87_data[0])) * v32_data);
            v85_acc += ((static_cast<float>(v87_data[1])) * v33_data);
            v85_acc += ((static_cast<float>(v87_data[2])) * v34_data);
            v85_acc += ((static_cast<float>(v87_data[3])) * v35_data);
            v85_acc += ((static_cast<float>(v87_data[4])) * v36_data);
            v85_acc += ((static_cast<float>(v87_data[5])) * v37_data);
            v85_acc += ((static_cast<float>(v87_data[6])) * v38_data);
            v85_acc += ((static_cast<float>(v87_data[7])) * v39_data);
            v85_acc += ((static_cast<float>(v87_data[8])) * v40_data);
            v85_acc += ((static_cast<float>(v87_data[9])) * v41_data);
            v85_acc += ((static_cast<float>(v87_data[10])) * v42_data);
            v85_acc += ((static_cast<float>(v87_data[11])) * v43_data);
            v85_acc += ((static_cast<float>(v87_data[12])) * v44_data);
            v85_acc += ((static_cast<float>(v87_data[13])) * v45_data);
            v85_acc += ((static_cast<float>(v87_data[14])) * v46_data);
            v85_acc += ((static_cast<float>(v87_data[15])) * v47_data);
            ir1.template select<16, 1>(16) = v85_acc;
            tensorforge::intel_esimd::simd<float, 16> v120_acc{};
            tensorforge::intel_esimd::simd<float, 16> v122_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
            v120_acc += ((static_cast<float>(v122_data[0])) * v32_data);
            v120_acc += ((static_cast<float>(v122_data[1])) * v33_data);
            v120_acc += ((static_cast<float>(v122_data[2])) * v34_data);
            v120_acc += ((static_cast<float>(v122_data[3])) * v35_data);
            v120_acc += ((static_cast<float>(v122_data[4])) * v36_data);
            v120_acc += ((static_cast<float>(v122_data[5])) * v37_data);
            v120_acc += ((static_cast<float>(v122_data[6])) * v38_data);
            v120_acc += ((static_cast<float>(v122_data[7])) * v39_data);
            v120_acc += ((static_cast<float>(v122_data[8])) * v40_data);
            v120_acc += ((static_cast<float>(v122_data[9])) * v41_data);
            v120_acc += ((static_cast<float>(v122_data[10])) * v42_data);
            v120_acc += ((static_cast<float>(v122_data[11])) * v43_data);
            v120_acc += ((static_cast<float>(v122_data[12])) * v44_data);
            v120_acc += ((static_cast<float>(v122_data[13])) * v45_data);
            v120_acc += ((static_cast<float>(v122_data[14])) * v46_data);
            v120_acc += ((static_cast<float>(v122_data[15])) * v47_data);
            ir1.template select<16, 1>(32) = v120_acc;
            tensorforge::intel_esimd::simd<float, 16> v155_acc{};
            tensorforge::intel_esimd::simd<float, 16> v157_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
            v155_acc += ((static_cast<float>(v157_data[0])) * v32_data);
            v155_acc += ((static_cast<float>(v157_data[1])) * v33_data);
            v155_acc += ((static_cast<float>(v157_data[2])) * v34_data);
            v155_acc += ((static_cast<float>(v157_data[3])) * v35_data);
            v155_acc += ((static_cast<float>(v157_data[4])) * v36_data);
            v155_acc += ((static_cast<float>(v157_data[5])) * v37_data);
            v155_acc += ((static_cast<float>(v157_data[6])) * v38_data);
            v155_acc += ((static_cast<float>(v157_data[7])) * v39_data);
            v155_acc += ((static_cast<float>(v157_data[8])) * v40_data);
            v155_acc += ((static_cast<float>(v157_data[9])) * v41_data);
            v155_acc += ((static_cast<float>(v157_data[10])) * v42_data);
            v155_acc += ((static_cast<float>(v157_data[11])) * v43_data);
            v155_acc += ((static_cast<float>(v157_data[12])) * v44_data);
            v155_acc += ((static_cast<float>(v157_data[13])) * v45_data);
            v155_acc += ((static_cast<float>(v157_data[14])) * v46_data);
            v155_acc += ((static_cast<float>(v157_data[15])) * v47_data);
            ir1.template select<16, 1>(48) = v155_acc;
            tensorforge::intel_esimd::simd<float, 16> v190_acc{};
            tensorforge::intel_esimd::simd<float, 16> v192_data = tensorforge::slmLoad<float, 16>(s0 + (64_i32));
            v190_acc += ((static_cast<float>(v192_data[0])) * v32_data);
            v190_acc += ((static_cast<float>(v192_data[1])) * v33_data);
            v190_acc += ((static_cast<float>(v192_data[2])) * v34_data);
            v190_acc += ((static_cast<float>(v192_data[3])) * v35_data);
            v190_acc += ((static_cast<float>(v192_data[4])) * v36_data);
            v190_acc += ((static_cast<float>(v192_data[5])) * v37_data);
            v190_acc += ((static_cast<float>(v192_data[6])) * v38_data);
            v190_acc += ((static_cast<float>(v192_data[7])) * v39_data);
            v190_acc += ((static_cast<float>(v192_data[8])) * v40_data);
            v190_acc += ((static_cast<float>(v192_data[9])) * v41_data);
            v190_acc += ((static_cast<float>(v192_data[10])) * v42_data);
            v190_acc += ((static_cast<float>(v192_data[11])) * v43_data);
            v190_acc += ((static_cast<float>(v192_data[12])) * v44_data);
            v190_acc += ((static_cast<float>(v192_data[13])) * v45_data);
            v190_acc += ((static_cast<float>(v192_data[14])) * v46_data);
            v190_acc += ((static_cast<float>(v192_data[15])) * v47_data);
            ir1.template select<16, 1>(64) = v190_acc;
            tensorforge::intel_esimd::simd<float, 16> v225_acc{};
            tensorforge::intel_esimd::simd<float, 16> v227_data = tensorforge::slmLoad<float, 16>(s0 + (80_i32));
            v225_acc += ((static_cast<float>(v227_data[0])) * v32_data);
            v225_acc += ((static_cast<float>(v227_data[1])) * v33_data);
            v225_acc += ((static_cast<float>(v227_data[2])) * v34_data);
            v225_acc += ((static_cast<float>(v227_data[3])) * v35_data);
            v225_acc += ((static_cast<float>(v227_data[4])) * v36_data);
            v225_acc += ((static_cast<float>(v227_data[5])) * v37_data);
            v225_acc += ((static_cast<float>(v227_data[6])) * v38_data);
            v225_acc += ((static_cast<float>(v227_data[7])) * v39_data);
            v225_acc += ((static_cast<float>(v227_data[8])) * v40_data);
            v225_acc += ((static_cast<float>(v227_data[9])) * v41_data);
            v225_acc += ((static_cast<float>(v227_data[10])) * v42_data);
            v225_acc += ((static_cast<float>(v227_data[11])) * v43_data);
            v225_acc += ((static_cast<float>(v227_data[12])) * v44_data);
            v225_acc += ((static_cast<float>(v227_data[13])) * v45_data);
            v225_acc += ((static_cast<float>(v227_data[14])) * v46_data);
            v225_acc += ((static_cast<float>(v227_data[15])) * v47_data);
            ir1.template select<16, 1>(80) = v225_acc;
            tensorforge::intel_esimd::simd<float, 16> v260_acc{};
            tensorforge::intel_esimd::simd<float, 16> v262_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
            v260_acc += ((static_cast<float>(v262_data[0])) * v32_data);
            v260_acc += ((static_cast<float>(v262_data[1])) * v33_data);
            v260_acc += ((static_cast<float>(v262_data[2])) * v34_data);
            v260_acc += ((static_cast<float>(v262_data[3])) * v35_data);
            v260_acc += ((static_cast<float>(v262_data[4])) * v36_data);
            v260_acc += ((static_cast<float>(v262_data[5])) * v37_data);
            v260_acc += ((static_cast<float>(v262_data[6])) * v38_data);
            v260_acc += ((static_cast<float>(v262_data[7])) * v39_data);
            v260_acc += ((static_cast<float>(v262_data[8])) * v40_data);
            v260_acc += ((static_cast<float>(v262_data[9])) * v41_data);
            v260_acc += ((static_cast<float>(v262_data[10])) * v42_data);
            v260_acc += ((static_cast<float>(v262_data[11])) * v43_data);
            v260_acc += ((static_cast<float>(v262_data[12])) * v44_data);
            v260_acc += ((static_cast<float>(v262_data[13])) * v45_data);
            v260_acc += ((static_cast<float>(v262_data[14])) * v46_data);
            v260_acc += ((static_cast<float>(v262_data[15])) * v47_data);
            ir1.template select<16, 1>(96) = v260_acc;
            tensorforge::intel_esimd::simd<float, 16> v295_acc{};
            tensorforge::intel_esimd::simd<float, 16> v297_data = tensorforge::slmLoad<float, 16>(s0 + (112_i32));
            v295_acc += ((static_cast<float>(v297_data[0])) * v32_data);
            v295_acc += ((static_cast<float>(v297_data[1])) * v33_data);
            v295_acc += ((static_cast<float>(v297_data[2])) * v34_data);
            v295_acc += ((static_cast<float>(v297_data[3])) * v35_data);
            v295_acc += ((static_cast<float>(v297_data[4])) * v36_data);
            v295_acc += ((static_cast<float>(v297_data[5])) * v37_data);
            v295_acc += ((static_cast<float>(v297_data[6])) * v38_data);
            v295_acc += ((static_cast<float>(v297_data[7])) * v39_data);
            v295_acc += ((static_cast<float>(v297_data[8])) * v40_data);
            v295_acc += ((static_cast<float>(v297_data[9])) * v41_data);
            v295_acc += ((static_cast<float>(v297_data[10])) * v42_data);
            v295_acc += ((static_cast<float>(v297_data[11])) * v43_data);
            v295_acc += ((static_cast<float>(v297_data[12])) * v44_data);
            v295_acc += ((static_cast<float>(v297_data[13])) * v45_data);
            v295_acc += ((static_cast<float>(v297_data[14])) * v46_data);
            v295_acc += ((static_cast<float>(v297_data[15])) * v47_data);
            ir1.template select<16, 1>(112) = v295_acc;
            tensorforge::intel_esimd::simd<float, 16> v330_acc{};
            tensorforge::intel_esimd::simd<float, 16> v332_data = tensorforge::slmLoad<float, 16>(s0 + (128_i32));
            v330_acc += ((static_cast<float>(v332_data[0])) * v32_data);
            v330_acc += ((static_cast<float>(v332_data[1])) * v33_data);
            v330_acc += ((static_cast<float>(v332_data[2])) * v34_data);
            v330_acc += ((static_cast<float>(v332_data[3])) * v35_data);
            v330_acc += ((static_cast<float>(v332_data[4])) * v36_data);
            v330_acc += ((static_cast<float>(v332_data[5])) * v37_data);
            v330_acc += ((static_cast<float>(v332_data[6])) * v38_data);
            v330_acc += ((static_cast<float>(v332_data[7])) * v39_data);
            v330_acc += ((static_cast<float>(v332_data[8])) * v40_data);
            v330_acc += ((static_cast<float>(v332_data[9])) * v41_data);
            v330_acc += ((static_cast<float>(v332_data[10])) * v42_data);
            v330_acc += ((static_cast<float>(v332_data[11])) * v43_data);
            v330_acc += ((static_cast<float>(v332_data[12])) * v44_data);
            v330_acc += ((static_cast<float>(v332_data[13])) * v45_data);
            v330_acc += ((static_cast<float>(v332_data[14])) * v46_data);
            v330_acc += ((static_cast<float>(v332_data[15])) * v47_data);
            ir1.template select<16, 1>(128) = v330_acc;
            tensorforge::intel_esimd::simd<float, 16> v365_acc{};
            tensorforge::intel_esimd::simd<float, 16> v367_data = tensorforge::slmLoad<float, 16>(s0 + (144_i32));
            v365_acc += ((static_cast<float>(v367_data[0])) * v32_data);
            v365_acc += ((static_cast<float>(v367_data[1])) * v33_data);
            v365_acc += ((static_cast<float>(v367_data[2])) * v34_data);
            v365_acc += ((static_cast<float>(v367_data[3])) * v35_data);
            v365_acc += ((static_cast<float>(v367_data[4])) * v36_data);
            v365_acc += ((static_cast<float>(v367_data[5])) * v37_data);
            v365_acc += ((static_cast<float>(v367_data[6])) * v38_data);
            v365_acc += ((static_cast<float>(v367_data[7])) * v39_data);
            v365_acc += ((static_cast<float>(v367_data[8])) * v40_data);
            v365_acc += ((static_cast<float>(v367_data[9])) * v41_data);
            v365_acc += ((static_cast<float>(v367_data[10])) * v42_data);
            v365_acc += ((static_cast<float>(v367_data[11])) * v43_data);
            v365_acc += ((static_cast<float>(v367_data[12])) * v44_data);
            v365_acc += ((static_cast<float>(v367_data[13])) * v45_data);
            v365_acc += ((static_cast<float>(v367_data[14])) * v46_data);
            v365_acc += ((static_cast<float>(v367_data[15])) * v47_data);
            ir1.template select<16, 1>(144) = v365_acc;
            tensorforge::intel_esimd::simd<float, 16> v400_acc{};
            tensorforge::intel_esimd::simd<float, 16> v402_data = tensorforge::slmLoad<float, 16>(s0 + (160_i32));
            v400_acc += ((static_cast<float>(v402_data[0])) * v32_data);
            v400_acc += ((static_cast<float>(v402_data[1])) * v33_data);
            v400_acc += ((static_cast<float>(v402_data[2])) * v34_data);
            v400_acc += ((static_cast<float>(v402_data[3])) * v35_data);
            v400_acc += ((static_cast<float>(v402_data[4])) * v36_data);
            v400_acc += ((static_cast<float>(v402_data[5])) * v37_data);
            v400_acc += ((static_cast<float>(v402_data[6])) * v38_data);
            v400_acc += ((static_cast<float>(v402_data[7])) * v39_data);
            v400_acc += ((static_cast<float>(v402_data[8])) * v40_data);
            v400_acc += ((static_cast<float>(v402_data[9])) * v41_data);
            v400_acc += ((static_cast<float>(v402_data[10])) * v42_data);
            v400_acc += ((static_cast<float>(v402_data[11])) * v43_data);
            v400_acc += ((static_cast<float>(v402_data[12])) * v44_data);
            v400_acc += ((static_cast<float>(v402_data[13])) * v45_data);
            v400_acc += ((static_cast<float>(v402_data[14])) * v46_data);
            v400_acc += ((static_cast<float>(v402_data[15])) * v47_data);
            ir1.template select<16, 1>(160) = v400_acc;
            tensorforge::intel_esimd::simd<float, 16> v435_acc{};
            tensorforge::intel_esimd::simd<float, 16> v437_data = tensorforge::slmLoad<float, 16>(s0 + (176_i32));
            v435_acc += ((static_cast<float>(v437_data[0])) * v32_data);
            v435_acc += ((static_cast<float>(v437_data[1])) * v33_data);
            v435_acc += ((static_cast<float>(v437_data[2])) * v34_data);
            v435_acc += ((static_cast<float>(v437_data[3])) * v35_data);
            v435_acc += ((static_cast<float>(v437_data[4])) * v36_data);
            v435_acc += ((static_cast<float>(v437_data[5])) * v37_data);
            v435_acc += ((static_cast<float>(v437_data[6])) * v38_data);
            v435_acc += ((static_cast<float>(v437_data[7])) * v39_data);
            v435_acc += ((static_cast<float>(v437_data[8])) * v40_data);
            v435_acc += ((static_cast<float>(v437_data[9])) * v41_data);
            v435_acc += ((static_cast<float>(v437_data[10])) * v42_data);
            v435_acc += ((static_cast<float>(v437_data[11])) * v43_data);
            v435_acc += ((static_cast<float>(v437_data[12])) * v44_data);
            v435_acc += ((static_cast<float>(v437_data[13])) * v45_data);
            v435_acc += ((static_cast<float>(v437_data[14])) * v46_data);
            v435_acc += ((static_cast<float>(v437_data[15])) * v47_data);
            ir1.template select<16, 1>(176) = v435_acc;
            tensorforge::intel_esimd::simd<float, 16> v470_acc{};
            tensorforge::intel_esimd::simd<float, 16> v472_data = tensorforge::slmLoad<float, 16>(s0 + (192_i32));
            v470_acc += ((static_cast<float>(v472_data[0])) * v32_data);
            v470_acc += ((static_cast<float>(v472_data[1])) * v33_data);
            v470_acc += ((static_cast<float>(v472_data[2])) * v34_data);
            v470_acc += ((static_cast<float>(v472_data[3])) * v35_data);
            v470_acc += ((static_cast<float>(v472_data[4])) * v36_data);
            v470_acc += ((static_cast<float>(v472_data[5])) * v37_data);
            v470_acc += ((static_cast<float>(v472_data[6])) * v38_data);
            v470_acc += ((static_cast<float>(v472_data[7])) * v39_data);
            v470_acc += ((static_cast<float>(v472_data[8])) * v40_data);
            v470_acc += ((static_cast<float>(v472_data[9])) * v41_data);
            v470_acc += ((static_cast<float>(v472_data[10])) * v42_data);
            v470_acc += ((static_cast<float>(v472_data[11])) * v43_data);
            v470_acc += ((static_cast<float>(v472_data[12])) * v44_data);
            v470_acc += ((static_cast<float>(v472_data[13])) * v45_data);
            v470_acc += ((static_cast<float>(v472_data[14])) * v46_data);
            v470_acc += ((static_cast<float>(v472_data[15])) * v47_data);
            ir1.template select<16, 1>(192) = v470_acc;
            tensorforge::intel_esimd::simd<float, 16> v505_acc{};
            tensorforge::intel_esimd::simd<float, 16> v507_data = tensorforge::slmLoad<float, 16>(s0 + (208_i32));
            v505_acc += ((static_cast<float>(v507_data[0])) * v32_data);
            v505_acc += ((static_cast<float>(v507_data[1])) * v33_data);
            v505_acc += ((static_cast<float>(v507_data[2])) * v34_data);
            v505_acc += ((static_cast<float>(v507_data[3])) * v35_data);
            v505_acc += ((static_cast<float>(v507_data[4])) * v36_data);
            v505_acc += ((static_cast<float>(v507_data[5])) * v37_data);
            v505_acc += ((static_cast<float>(v507_data[6])) * v38_data);
            v505_acc += ((static_cast<float>(v507_data[7])) * v39_data);
            v505_acc += ((static_cast<float>(v507_data[8])) * v40_data);
            v505_acc += ((static_cast<float>(v507_data[9])) * v41_data);
            v505_acc += ((static_cast<float>(v507_data[10])) * v42_data);
            v505_acc += ((static_cast<float>(v507_data[11])) * v43_data);
            v505_acc += ((static_cast<float>(v507_data[12])) * v44_data);
            v505_acc += ((static_cast<float>(v507_data[13])) * v45_data);
            v505_acc += ((static_cast<float>(v507_data[14])) * v46_data);
            v505_acc += ((static_cast<float>(v507_data[15])) * v47_data);
            ir1.template select<16, 1>(208) = v505_acc;
            tensorforge::intel_esimd::simd<float, 16> v540_acc{};
            tensorforge::intel_esimd::simd<float, 16> v542_data = tensorforge::slmLoad<float, 16>(s0 + (224_i32));
            v540_acc += ((static_cast<float>(v542_data[0])) * v32_data);
            v540_acc += ((static_cast<float>(v542_data[1])) * v33_data);
            v540_acc += ((static_cast<float>(v542_data[2])) * v34_data);
            v540_acc += ((static_cast<float>(v542_data[3])) * v35_data);
            v540_acc += ((static_cast<float>(v542_data[4])) * v36_data);
            v540_acc += ((static_cast<float>(v542_data[5])) * v37_data);
            v540_acc += ((static_cast<float>(v542_data[6])) * v38_data);
            v540_acc += ((static_cast<float>(v542_data[7])) * v39_data);
            v540_acc += ((static_cast<float>(v542_data[8])) * v40_data);
            v540_acc += ((static_cast<float>(v542_data[9])) * v41_data);
            v540_acc += ((static_cast<float>(v542_data[10])) * v42_data);
            v540_acc += ((static_cast<float>(v542_data[11])) * v43_data);
            v540_acc += ((static_cast<float>(v542_data[12])) * v44_data);
            v540_acc += ((static_cast<float>(v542_data[13])) * v45_data);
            v540_acc += ((static_cast<float>(v542_data[14])) * v46_data);
            v540_acc += ((static_cast<float>(v542_data[15])) * v47_data);
            ir1.template select<16, 1>(224) = v540_acc;
            tensorforge::intel_esimd::simd<float, 16> v575_acc{};
            tensorforge::intel_esimd::simd<float, 16> v577_data = tensorforge::slmLoad<float, 16>(s0 + (240_i32));
            v575_acc += ((static_cast<float>(v577_data[0])) * v32_data);
            v575_acc += ((static_cast<float>(v577_data[1])) * v33_data);
            v575_acc += ((static_cast<float>(v577_data[2])) * v34_data);
            v575_acc += ((static_cast<float>(v577_data[3])) * v35_data);
            v575_acc += ((static_cast<float>(v577_data[4])) * v36_data);
            v575_acc += ((static_cast<float>(v577_data[5])) * v37_data);
            v575_acc += ((static_cast<float>(v577_data[6])) * v38_data);
            v575_acc += ((static_cast<float>(v577_data[7])) * v39_data);
            v575_acc += ((static_cast<float>(v577_data[8])) * v40_data);
            v575_acc += ((static_cast<float>(v577_data[9])) * v41_data);
            v575_acc += ((static_cast<float>(v577_data[10])) * v42_data);
            v575_acc += ((static_cast<float>(v577_data[11])) * v43_data);
            v575_acc += ((static_cast<float>(v577_data[12])) * v44_data);
            v575_acc += ((static_cast<float>(v577_data[13])) * v45_data);
            v575_acc += ((static_cast<float>(v577_data[14])) * v46_data);
            v575_acc += ((static_cast<float>(v577_data[15])) * v47_data);
            ir1.template select<16, 1>(240) = v575_acc;
            #pragma unroll
            for (int32_t v610_n0 = 0; v610_n0 < 1; ++v610_n0) {
              int32_t v612_a = v610_n0 * 16;
              #pragma unroll
              for (int32_t v611_n1 = 0; v611_n1 < 16; ++v611_n1) {
                int32_t v614_a = v612_a + (v611_n1 * 16);
                tensorforge::intel_esimd::simd<float, 16> v615_data(ir1.template select<16, 1>(v614_a));
                r1.template select<16, 1>(v614_a) = v615_data;
              }
            }
            // glb_m0 = store{r>g}(r1);
            #pragma unroll
            for (int32_t v616_i0 = 0; v616_i0 < 1; ++v616_i0) {
              int32_t v618_a = v616_i0 * 16;
              #pragma unroll
              for (int32_t v617_i1 = 0; v617_i1 < 16; ++v617_i1) {
                int32_t v620_a = v618_a + (v617_i1 * 16);
                tensorforge::intel_esimd::simd<float, 16> v621_data(r1.template select<16, 1>(v620_a));
                v621_data.copy_to(glb_m0 + (v620_a));
              }
            }
            tensorforge::prefetchL2<256>(&pf_glb_m1[0]);
            tensorforge::prefetchL2<256>(&pf_glb_m2[0]);
          }
        }
      }
    });
  });
}

