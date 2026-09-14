// === base name ===
kernel_f1a214855a6425a4

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_f1a214855a6425a4 = {{1, 16, 1}, 16, 12, 1, 16, 9216, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_f1a214855a6425a4(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_f1a214855a6425a4(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_f1a214855a6425a4(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 2304 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_f1a214855a6425a4(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_f1a214855a6425a4(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_f1a214855a6425a4(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f1a214855a6425a4(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2304 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 9216 B shared, occupancy grid
        // operands:
        //   m0 12×8(12×8) {0..12}×{0..8} strided
        //   m1 32×16(32×16) {0..32}×{0..16} strided
        //   m2 16×8(16×8) {0..16}×{0..8} strided
        // operations:
        //   m0[i,j] = m1[i,k]@{4..16}×{0..16} × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2304}],"shared_bytes":9216,"shared_elements":2304,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,8]],"name":"m0","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[32,16]],"name":"m1","ordered":false,"parts":1,"shape":[32,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m1","offset":[4,0],"shape":[32,16]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
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
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 512 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 128 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v17_i1 = 0; v17_i1 < 16; ++v17_i1) {
                tensorforge::intel_esimd::simd<float, 12> v23_data;
                v23_data.copy_from(glb_m1 + ((4_i32 + (v17_i1 * 32))));
                r0.template select<12, 1>((v17_i1 * 16)) = v23_data;
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v26_ld;
              v26_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v26_ld);
              tensorforge::intel_esimd::simd<float, 64> v27_ld;
              v27_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v27_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 128> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 12), (0, 8)] [(0, 16)]
              tensorforge::intel_esimd::simd<float, 128> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v30_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v31_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v32_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v33_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v34_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v35_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v36_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v37_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v38_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v39_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v40_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(192));
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(208));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(224));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(240));
              tensorforge::intel_esimd::simd<float, 16> v46_acc{};
              tensorforge::intel_esimd::simd<float, 16> v50_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v46_acc += ((static_cast<float>(v50_data[0])) * v30_data);
              v46_acc += ((static_cast<float>(v50_data[1])) * v31_data);
              v46_acc += ((static_cast<float>(v50_data[2])) * v32_data);
              v46_acc += ((static_cast<float>(v50_data[3])) * v33_data);
              v46_acc += ((static_cast<float>(v50_data[4])) * v34_data);
              v46_acc += ((static_cast<float>(v50_data[5])) * v35_data);
              v46_acc += ((static_cast<float>(v50_data[6])) * v36_data);
              v46_acc += ((static_cast<float>(v50_data[7])) * v37_data);
              v46_acc += ((static_cast<float>(v50_data[8])) * v38_data);
              v46_acc += ((static_cast<float>(v50_data[9])) * v39_data);
              v46_acc += ((static_cast<float>(v50_data[10])) * v40_data);
              v46_acc += ((static_cast<float>(v50_data[11])) * v41_data);
              v46_acc += ((static_cast<float>(v50_data[12])) * v42_data);
              v46_acc += ((static_cast<float>(v50_data[13])) * v43_data);
              v46_acc += ((static_cast<float>(v50_data[14])) * v44_data);
              v46_acc += ((static_cast<float>(v50_data[15])) * v45_data);
              ir1.template select<16, 1>(0) = v46_acc;
              tensorforge::intel_esimd::simd<float, 16> v83_acc{};
              tensorforge::intel_esimd::simd<float, 16> v85_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v83_acc += ((static_cast<float>(v85_data[0])) * v30_data);
              v83_acc += ((static_cast<float>(v85_data[1])) * v31_data);
              v83_acc += ((static_cast<float>(v85_data[2])) * v32_data);
              v83_acc += ((static_cast<float>(v85_data[3])) * v33_data);
              v83_acc += ((static_cast<float>(v85_data[4])) * v34_data);
              v83_acc += ((static_cast<float>(v85_data[5])) * v35_data);
              v83_acc += ((static_cast<float>(v85_data[6])) * v36_data);
              v83_acc += ((static_cast<float>(v85_data[7])) * v37_data);
              v83_acc += ((static_cast<float>(v85_data[8])) * v38_data);
              v83_acc += ((static_cast<float>(v85_data[9])) * v39_data);
              v83_acc += ((static_cast<float>(v85_data[10])) * v40_data);
              v83_acc += ((static_cast<float>(v85_data[11])) * v41_data);
              v83_acc += ((static_cast<float>(v85_data[12])) * v42_data);
              v83_acc += ((static_cast<float>(v85_data[13])) * v43_data);
              v83_acc += ((static_cast<float>(v85_data[14])) * v44_data);
              v83_acc += ((static_cast<float>(v85_data[15])) * v45_data);
              ir1.template select<16, 1>(16) = v83_acc;
              tensorforge::intel_esimd::simd<float, 16> v118_acc{};
              tensorforge::intel_esimd::simd<float, 16> v120_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v118_acc += ((static_cast<float>(v120_data[0])) * v30_data);
              v118_acc += ((static_cast<float>(v120_data[1])) * v31_data);
              v118_acc += ((static_cast<float>(v120_data[2])) * v32_data);
              v118_acc += ((static_cast<float>(v120_data[3])) * v33_data);
              v118_acc += ((static_cast<float>(v120_data[4])) * v34_data);
              v118_acc += ((static_cast<float>(v120_data[5])) * v35_data);
              v118_acc += ((static_cast<float>(v120_data[6])) * v36_data);
              v118_acc += ((static_cast<float>(v120_data[7])) * v37_data);
              v118_acc += ((static_cast<float>(v120_data[8])) * v38_data);
              v118_acc += ((static_cast<float>(v120_data[9])) * v39_data);
              v118_acc += ((static_cast<float>(v120_data[10])) * v40_data);
              v118_acc += ((static_cast<float>(v120_data[11])) * v41_data);
              v118_acc += ((static_cast<float>(v120_data[12])) * v42_data);
              v118_acc += ((static_cast<float>(v120_data[13])) * v43_data);
              v118_acc += ((static_cast<float>(v120_data[14])) * v44_data);
              v118_acc += ((static_cast<float>(v120_data[15])) * v45_data);
              ir1.template select<16, 1>(32) = v118_acc;
              tensorforge::intel_esimd::simd<float, 16> v153_acc{};
              tensorforge::intel_esimd::simd<float, 16> v155_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v153_acc += ((static_cast<float>(v155_data[0])) * v30_data);
              v153_acc += ((static_cast<float>(v155_data[1])) * v31_data);
              v153_acc += ((static_cast<float>(v155_data[2])) * v32_data);
              v153_acc += ((static_cast<float>(v155_data[3])) * v33_data);
              v153_acc += ((static_cast<float>(v155_data[4])) * v34_data);
              v153_acc += ((static_cast<float>(v155_data[5])) * v35_data);
              v153_acc += ((static_cast<float>(v155_data[6])) * v36_data);
              v153_acc += ((static_cast<float>(v155_data[7])) * v37_data);
              v153_acc += ((static_cast<float>(v155_data[8])) * v38_data);
              v153_acc += ((static_cast<float>(v155_data[9])) * v39_data);
              v153_acc += ((static_cast<float>(v155_data[10])) * v40_data);
              v153_acc += ((static_cast<float>(v155_data[11])) * v41_data);
              v153_acc += ((static_cast<float>(v155_data[12])) * v42_data);
              v153_acc += ((static_cast<float>(v155_data[13])) * v43_data);
              v153_acc += ((static_cast<float>(v155_data[14])) * v44_data);
              v153_acc += ((static_cast<float>(v155_data[15])) * v45_data);
              ir1.template select<16, 1>(48) = v153_acc;
              tensorforge::intel_esimd::simd<float, 16> v188_acc{};
              tensorforge::intel_esimd::simd<float, 16> v190_data = tensorforge::slmLoad<float, 16>(s0 + (64_i32));
              v188_acc += ((static_cast<float>(v190_data[0])) * v30_data);
              v188_acc += ((static_cast<float>(v190_data[1])) * v31_data);
              v188_acc += ((static_cast<float>(v190_data[2])) * v32_data);
              v188_acc += ((static_cast<float>(v190_data[3])) * v33_data);
              v188_acc += ((static_cast<float>(v190_data[4])) * v34_data);
              v188_acc += ((static_cast<float>(v190_data[5])) * v35_data);
              v188_acc += ((static_cast<float>(v190_data[6])) * v36_data);
              v188_acc += ((static_cast<float>(v190_data[7])) * v37_data);
              v188_acc += ((static_cast<float>(v190_data[8])) * v38_data);
              v188_acc += ((static_cast<float>(v190_data[9])) * v39_data);
              v188_acc += ((static_cast<float>(v190_data[10])) * v40_data);
              v188_acc += ((static_cast<float>(v190_data[11])) * v41_data);
              v188_acc += ((static_cast<float>(v190_data[12])) * v42_data);
              v188_acc += ((static_cast<float>(v190_data[13])) * v43_data);
              v188_acc += ((static_cast<float>(v190_data[14])) * v44_data);
              v188_acc += ((static_cast<float>(v190_data[15])) * v45_data);
              ir1.template select<16, 1>(64) = v188_acc;
              tensorforge::intel_esimd::simd<float, 16> v223_acc{};
              tensorforge::intel_esimd::simd<float, 16> v225_data = tensorforge::slmLoad<float, 16>(s0 + (80_i32));
              v223_acc += ((static_cast<float>(v225_data[0])) * v30_data);
              v223_acc += ((static_cast<float>(v225_data[1])) * v31_data);
              v223_acc += ((static_cast<float>(v225_data[2])) * v32_data);
              v223_acc += ((static_cast<float>(v225_data[3])) * v33_data);
              v223_acc += ((static_cast<float>(v225_data[4])) * v34_data);
              v223_acc += ((static_cast<float>(v225_data[5])) * v35_data);
              v223_acc += ((static_cast<float>(v225_data[6])) * v36_data);
              v223_acc += ((static_cast<float>(v225_data[7])) * v37_data);
              v223_acc += ((static_cast<float>(v225_data[8])) * v38_data);
              v223_acc += ((static_cast<float>(v225_data[9])) * v39_data);
              v223_acc += ((static_cast<float>(v225_data[10])) * v40_data);
              v223_acc += ((static_cast<float>(v225_data[11])) * v41_data);
              v223_acc += ((static_cast<float>(v225_data[12])) * v42_data);
              v223_acc += ((static_cast<float>(v225_data[13])) * v43_data);
              v223_acc += ((static_cast<float>(v225_data[14])) * v44_data);
              v223_acc += ((static_cast<float>(v225_data[15])) * v45_data);
              ir1.template select<16, 1>(80) = v223_acc;
              tensorforge::intel_esimd::simd<float, 16> v258_acc{};
              tensorforge::intel_esimd::simd<float, 16> v260_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              v258_acc += ((static_cast<float>(v260_data[0])) * v30_data);
              v258_acc += ((static_cast<float>(v260_data[1])) * v31_data);
              v258_acc += ((static_cast<float>(v260_data[2])) * v32_data);
              v258_acc += ((static_cast<float>(v260_data[3])) * v33_data);
              v258_acc += ((static_cast<float>(v260_data[4])) * v34_data);
              v258_acc += ((static_cast<float>(v260_data[5])) * v35_data);
              v258_acc += ((static_cast<float>(v260_data[6])) * v36_data);
              v258_acc += ((static_cast<float>(v260_data[7])) * v37_data);
              v258_acc += ((static_cast<float>(v260_data[8])) * v38_data);
              v258_acc += ((static_cast<float>(v260_data[9])) * v39_data);
              v258_acc += ((static_cast<float>(v260_data[10])) * v40_data);
              v258_acc += ((static_cast<float>(v260_data[11])) * v41_data);
              v258_acc += ((static_cast<float>(v260_data[12])) * v42_data);
              v258_acc += ((static_cast<float>(v260_data[13])) * v43_data);
              v258_acc += ((static_cast<float>(v260_data[14])) * v44_data);
              v258_acc += ((static_cast<float>(v260_data[15])) * v45_data);
              ir1.template select<16, 1>(96) = v258_acc;
              tensorforge::intel_esimd::simd<float, 16> v293_acc{};
              tensorforge::intel_esimd::simd<float, 16> v295_data = tensorforge::slmLoad<float, 16>(s0 + (112_i32));
              v293_acc += ((static_cast<float>(v295_data[0])) * v30_data);
              v293_acc += ((static_cast<float>(v295_data[1])) * v31_data);
              v293_acc += ((static_cast<float>(v295_data[2])) * v32_data);
              v293_acc += ((static_cast<float>(v295_data[3])) * v33_data);
              v293_acc += ((static_cast<float>(v295_data[4])) * v34_data);
              v293_acc += ((static_cast<float>(v295_data[5])) * v35_data);
              v293_acc += ((static_cast<float>(v295_data[6])) * v36_data);
              v293_acc += ((static_cast<float>(v295_data[7])) * v37_data);
              v293_acc += ((static_cast<float>(v295_data[8])) * v38_data);
              v293_acc += ((static_cast<float>(v295_data[9])) * v39_data);
              v293_acc += ((static_cast<float>(v295_data[10])) * v40_data);
              v293_acc += ((static_cast<float>(v295_data[11])) * v41_data);
              v293_acc += ((static_cast<float>(v295_data[12])) * v42_data);
              v293_acc += ((static_cast<float>(v295_data[13])) * v43_data);
              v293_acc += ((static_cast<float>(v295_data[14])) * v44_data);
              v293_acc += ((static_cast<float>(v295_data[15])) * v45_data);
              ir1.template select<16, 1>(112) = v293_acc;
              // r1 = ir1
              #pragma unroll
              for (int32_t v328_n1 = 0; v328_n1 < 8; ++v328_n1) {
                int32_t v329_a = v328_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v331_data(ir1.template select<12, 1>(v329_a));
                r1.template select<12, 1>(v329_a) = v331_data;
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v332_i1 = 0; v332_i1 < 8; ++v332_i1) {
                tensorforge::intel_esimd::simd<float, 12> v335_data(r1.template select<12, 1>((v332_i1 * 16)));
                v335_data.copy_to(glb_m0 + ((v332_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}

