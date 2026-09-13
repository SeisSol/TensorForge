// === base name ===
kernel_f68a04ae225254c1

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_f68a04ae225254c1 = {{1, 16, 1}, 16, 12, 1, 16, 23552, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_f68a04ae225254c1(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_f68a04ae225254c1(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_f68a04ae225254c1(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 5888 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_f68a04ae225254c1(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_f68a04ae225254c1(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_f68a04ae225254c1(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f68a04ae225254c1(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<5888 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 23552 B shared, occupancy grid
        // operands:
        //   m0 12×16(12×16) {0..12}×{0..16} strided
        //   m1 12×20(12×20) {0..12}×{0..20} strided
        //   m2 16×20(16×20) {0..16}×{0..20} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[j,k]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":5888}],"shared_bytes":23552,"shared_elements":5888,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,16]],"name":"m0","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,20]],"name":"m1","ordered":false,"parts":1,"shape":[12,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,20]],"name":"m2","ordered":false,"parts":1,"shape":[16,20],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,20]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,20]},{"addressing":"strided","bbox":[[0,0],[16,20]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,20]}],"permute":[[0,1],[1,0]],"target":[[0,-1],[1,-1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (368 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (352);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v5_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v5_batchId0 < numElements0; v5_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v6_ahead1 = v5_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
            const float *const __restrict__ pf_glb_m1 = &m1[v8_batchId1 * 240 + 0 + m1_extraOffset];
            const float *const __restrict__ pf_glb_m2 = &m2[v8_batchId1 * 320 + 0 + m2_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 192 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 240 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 320 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 320> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v19_i1 = 0; v19_i1 < 20; ++v19_i1) {
                tensorforge::intel_esimd::simd<float, 12> v24_data;
                v24_data.copy_from(glb_m1 + ((v19_i1 * 12)));
                r0.template select<12, 1>((v19_i1 * 16)) = v24_data;
              }
              // s0 = load{g>s}(glb_m2[1, 0])
              #pragma unroll
              for (int32_t v27_i0 = 0; v27_i0 < 1; ++v27_i0) {
                int32_t v29_lead = v27_i0 * 16;
                #pragma unroll
                for (int32_t v28_i1 = 0; v28_i1 < 20; ++v28_i1) {
                  tensorforge::intel_esimd::simd<float, 16> v33_data;
                  v33_data.copy_from(glb_m2 + ((v29_lead + (v28_i1 * 16))));
                  tensorforge::slmStore<float, 16>(s0 + ((v29_lead + (v28_i1 * 17))), v33_data);
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[1, 0]));
              tensorforge::intel_esimd::simd<float, 256> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 12), (0, 16)] [(0, 20)]
              tensorforge::intel_esimd::simd<float, 256> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v38_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v39_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v40_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v48_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v49_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v50_data(r0.template select<16, 1>(192));
              tensorforge::intel_esimd::simd<float, 16> v51_data(r0.template select<16, 1>(208));
              tensorforge::intel_esimd::simd<float, 16> v52_data(r0.template select<16, 1>(224));
              tensorforge::intel_esimd::simd<float, 16> v53_data(r0.template select<16, 1>(240));
              tensorforge::intel_esimd::simd<float, 16> v54_data(r0.template select<16, 1>(256));
              tensorforge::intel_esimd::simd<float, 16> v55_data(r0.template select<16, 1>(272));
              tensorforge::intel_esimd::simd<float, 16> v56_data(r0.template select<16, 1>(288));
              tensorforge::intel_esimd::simd<float, 16> v57_data(r0.template select<16, 1>(304));
              tensorforge::intel_esimd::simd<float, 16> v58_acc{};
              tensorforge::intel_esimd::simd<float, 16> v63_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v58_acc += ((static_cast<float>(v63_data[0])) * v38_data);
              v58_acc += ((static_cast<float>(v63_data[1])) * v39_data);
              v58_acc += ((static_cast<float>(v63_data[2])) * v40_data);
              v58_acc += ((static_cast<float>(v63_data[3])) * v41_data);
              v58_acc += ((static_cast<float>(v63_data[4])) * v42_data);
              v58_acc += ((static_cast<float>(v63_data[5])) * v43_data);
              v58_acc += ((static_cast<float>(v63_data[6])) * v44_data);
              v58_acc += ((static_cast<float>(v63_data[7])) * v45_data);
              v58_acc += ((static_cast<float>(v63_data[8])) * v46_data);
              v58_acc += ((static_cast<float>(v63_data[9])) * v47_data);
              v58_acc += ((static_cast<float>(v63_data[10])) * v48_data);
              v58_acc += ((static_cast<float>(v63_data[11])) * v49_data);
              v58_acc += ((static_cast<float>(v63_data[12])) * v50_data);
              v58_acc += ((static_cast<float>(v63_data[13])) * v51_data);
              v58_acc += ((static_cast<float>(v63_data[14])) * v52_data);
              v58_acc += ((static_cast<float>(v63_data[15])) * v53_data);
              tensorforge::intel_esimd::simd<float, 16> v100_data = tensorforge::slmLoad<float, 16>(s0 + (272_i32));
              v58_acc += ((static_cast<float>(v100_data[0])) * v54_data);
              v58_acc += ((static_cast<float>(v100_data[1])) * v55_data);
              v58_acc += ((static_cast<float>(v100_data[2])) * v56_data);
              v58_acc += ((static_cast<float>(v100_data[3])) * v57_data);
              ir1.template select<16, 1>(0) = v58_acc;
              tensorforge::intel_esimd::simd<float, 16> v109_acc{};
              tensorforge::intel_esimd::simd<float, 16> v111_data = tensorforge::slmLoad<float, 16>(s0 + (1_i32));
              v109_acc += ((static_cast<float>(v111_data[0])) * v38_data);
              v109_acc += ((static_cast<float>(v111_data[1])) * v39_data);
              v109_acc += ((static_cast<float>(v111_data[2])) * v40_data);
              v109_acc += ((static_cast<float>(v111_data[3])) * v41_data);
              v109_acc += ((static_cast<float>(v111_data[4])) * v42_data);
              v109_acc += ((static_cast<float>(v111_data[5])) * v43_data);
              v109_acc += ((static_cast<float>(v111_data[6])) * v44_data);
              v109_acc += ((static_cast<float>(v111_data[7])) * v45_data);
              v109_acc += ((static_cast<float>(v111_data[8])) * v46_data);
              v109_acc += ((static_cast<float>(v111_data[9])) * v47_data);
              v109_acc += ((static_cast<float>(v111_data[10])) * v48_data);
              v109_acc += ((static_cast<float>(v111_data[11])) * v49_data);
              v109_acc += ((static_cast<float>(v111_data[12])) * v50_data);
              v109_acc += ((static_cast<float>(v111_data[13])) * v51_data);
              v109_acc += ((static_cast<float>(v111_data[14])) * v52_data);
              v109_acc += ((static_cast<float>(v111_data[15])) * v53_data);
              tensorforge::intel_esimd::simd<float, 16> v145_data = tensorforge::slmLoad<float, 16>(s0 + (273_i32));
              v109_acc += ((static_cast<float>(v145_data[0])) * v54_data);
              v109_acc += ((static_cast<float>(v145_data[1])) * v55_data);
              v109_acc += ((static_cast<float>(v145_data[2])) * v56_data);
              v109_acc += ((static_cast<float>(v145_data[3])) * v57_data);
              ir1.template select<16, 1>(16) = v109_acc;
              tensorforge::intel_esimd::simd<float, 16> v154_acc{};
              tensorforge::intel_esimd::simd<float, 16> v156_data = tensorforge::slmLoad<float, 16>(s0 + (2_i32));
              v154_acc += ((static_cast<float>(v156_data[0])) * v38_data);
              v154_acc += ((static_cast<float>(v156_data[1])) * v39_data);
              v154_acc += ((static_cast<float>(v156_data[2])) * v40_data);
              v154_acc += ((static_cast<float>(v156_data[3])) * v41_data);
              v154_acc += ((static_cast<float>(v156_data[4])) * v42_data);
              v154_acc += ((static_cast<float>(v156_data[5])) * v43_data);
              v154_acc += ((static_cast<float>(v156_data[6])) * v44_data);
              v154_acc += ((static_cast<float>(v156_data[7])) * v45_data);
              v154_acc += ((static_cast<float>(v156_data[8])) * v46_data);
              v154_acc += ((static_cast<float>(v156_data[9])) * v47_data);
              v154_acc += ((static_cast<float>(v156_data[10])) * v48_data);
              v154_acc += ((static_cast<float>(v156_data[11])) * v49_data);
              v154_acc += ((static_cast<float>(v156_data[12])) * v50_data);
              v154_acc += ((static_cast<float>(v156_data[13])) * v51_data);
              v154_acc += ((static_cast<float>(v156_data[14])) * v52_data);
              v154_acc += ((static_cast<float>(v156_data[15])) * v53_data);
              tensorforge::intel_esimd::simd<float, 16> v190_data = tensorforge::slmLoad<float, 16>(s0 + (274_i32));
              v154_acc += ((static_cast<float>(v190_data[0])) * v54_data);
              v154_acc += ((static_cast<float>(v190_data[1])) * v55_data);
              v154_acc += ((static_cast<float>(v190_data[2])) * v56_data);
              v154_acc += ((static_cast<float>(v190_data[3])) * v57_data);
              ir1.template select<16, 1>(32) = v154_acc;
              tensorforge::intel_esimd::simd<float, 16> v199_acc{};
              tensorforge::intel_esimd::simd<float, 16> v201_data = tensorforge::slmLoad<float, 16>(s0 + (3_i32));
              v199_acc += ((static_cast<float>(v201_data[0])) * v38_data);
              v199_acc += ((static_cast<float>(v201_data[1])) * v39_data);
              v199_acc += ((static_cast<float>(v201_data[2])) * v40_data);
              v199_acc += ((static_cast<float>(v201_data[3])) * v41_data);
              v199_acc += ((static_cast<float>(v201_data[4])) * v42_data);
              v199_acc += ((static_cast<float>(v201_data[5])) * v43_data);
              v199_acc += ((static_cast<float>(v201_data[6])) * v44_data);
              v199_acc += ((static_cast<float>(v201_data[7])) * v45_data);
              v199_acc += ((static_cast<float>(v201_data[8])) * v46_data);
              v199_acc += ((static_cast<float>(v201_data[9])) * v47_data);
              v199_acc += ((static_cast<float>(v201_data[10])) * v48_data);
              v199_acc += ((static_cast<float>(v201_data[11])) * v49_data);
              v199_acc += ((static_cast<float>(v201_data[12])) * v50_data);
              v199_acc += ((static_cast<float>(v201_data[13])) * v51_data);
              v199_acc += ((static_cast<float>(v201_data[14])) * v52_data);
              v199_acc += ((static_cast<float>(v201_data[15])) * v53_data);
              tensorforge::intel_esimd::simd<float, 16> v235_data = tensorforge::slmLoad<float, 16>(s0 + (275_i32));
              v199_acc += ((static_cast<float>(v235_data[0])) * v54_data);
              v199_acc += ((static_cast<float>(v235_data[1])) * v55_data);
              v199_acc += ((static_cast<float>(v235_data[2])) * v56_data);
              v199_acc += ((static_cast<float>(v235_data[3])) * v57_data);
              ir1.template select<16, 1>(48) = v199_acc;
              tensorforge::intel_esimd::simd<float, 16> v244_acc{};
              tensorforge::intel_esimd::simd<float, 16> v246_data = tensorforge::slmLoad<float, 16>(s0 + (4_i32));
              v244_acc += ((static_cast<float>(v246_data[0])) * v38_data);
              v244_acc += ((static_cast<float>(v246_data[1])) * v39_data);
              v244_acc += ((static_cast<float>(v246_data[2])) * v40_data);
              v244_acc += ((static_cast<float>(v246_data[3])) * v41_data);
              v244_acc += ((static_cast<float>(v246_data[4])) * v42_data);
              v244_acc += ((static_cast<float>(v246_data[5])) * v43_data);
              v244_acc += ((static_cast<float>(v246_data[6])) * v44_data);
              v244_acc += ((static_cast<float>(v246_data[7])) * v45_data);
              v244_acc += ((static_cast<float>(v246_data[8])) * v46_data);
              v244_acc += ((static_cast<float>(v246_data[9])) * v47_data);
              v244_acc += ((static_cast<float>(v246_data[10])) * v48_data);
              v244_acc += ((static_cast<float>(v246_data[11])) * v49_data);
              v244_acc += ((static_cast<float>(v246_data[12])) * v50_data);
              v244_acc += ((static_cast<float>(v246_data[13])) * v51_data);
              v244_acc += ((static_cast<float>(v246_data[14])) * v52_data);
              v244_acc += ((static_cast<float>(v246_data[15])) * v53_data);
              tensorforge::intel_esimd::simd<float, 16> v280_data = tensorforge::slmLoad<float, 16>(s0 + (276_i32));
              v244_acc += ((static_cast<float>(v280_data[0])) * v54_data);
              v244_acc += ((static_cast<float>(v280_data[1])) * v55_data);
              v244_acc += ((static_cast<float>(v280_data[2])) * v56_data);
              v244_acc += ((static_cast<float>(v280_data[3])) * v57_data);
              ir1.template select<16, 1>(64) = v244_acc;
              tensorforge::intel_esimd::simd<float, 16> v289_acc{};
              tensorforge::intel_esimd::simd<float, 16> v291_data = tensorforge::slmLoad<float, 16>(s0 + (5_i32));
              v289_acc += ((static_cast<float>(v291_data[0])) * v38_data);
              v289_acc += ((static_cast<float>(v291_data[1])) * v39_data);
              v289_acc += ((static_cast<float>(v291_data[2])) * v40_data);
              v289_acc += ((static_cast<float>(v291_data[3])) * v41_data);
              v289_acc += ((static_cast<float>(v291_data[4])) * v42_data);
              v289_acc += ((static_cast<float>(v291_data[5])) * v43_data);
              v289_acc += ((static_cast<float>(v291_data[6])) * v44_data);
              v289_acc += ((static_cast<float>(v291_data[7])) * v45_data);
              v289_acc += ((static_cast<float>(v291_data[8])) * v46_data);
              v289_acc += ((static_cast<float>(v291_data[9])) * v47_data);
              v289_acc += ((static_cast<float>(v291_data[10])) * v48_data);
              v289_acc += ((static_cast<float>(v291_data[11])) * v49_data);
              v289_acc += ((static_cast<float>(v291_data[12])) * v50_data);
              v289_acc += ((static_cast<float>(v291_data[13])) * v51_data);
              v289_acc += ((static_cast<float>(v291_data[14])) * v52_data);
              v289_acc += ((static_cast<float>(v291_data[15])) * v53_data);
              tensorforge::intel_esimd::simd<float, 16> v325_data = tensorforge::slmLoad<float, 16>(s0 + (277_i32));
              v289_acc += ((static_cast<float>(v325_data[0])) * v54_data);
              v289_acc += ((static_cast<float>(v325_data[1])) * v55_data);
              v289_acc += ((static_cast<float>(v325_data[2])) * v56_data);
              v289_acc += ((static_cast<float>(v325_data[3])) * v57_data);
              ir1.template select<16, 1>(80) = v289_acc;
              tensorforge::intel_esimd::simd<float, 16> v334_acc{};
              tensorforge::intel_esimd::simd<float, 16> v336_data = tensorforge::slmLoad<float, 16>(s0 + (6_i32));
              v334_acc += ((static_cast<float>(v336_data[0])) * v38_data);
              v334_acc += ((static_cast<float>(v336_data[1])) * v39_data);
              v334_acc += ((static_cast<float>(v336_data[2])) * v40_data);
              v334_acc += ((static_cast<float>(v336_data[3])) * v41_data);
              v334_acc += ((static_cast<float>(v336_data[4])) * v42_data);
              v334_acc += ((static_cast<float>(v336_data[5])) * v43_data);
              v334_acc += ((static_cast<float>(v336_data[6])) * v44_data);
              v334_acc += ((static_cast<float>(v336_data[7])) * v45_data);
              v334_acc += ((static_cast<float>(v336_data[8])) * v46_data);
              v334_acc += ((static_cast<float>(v336_data[9])) * v47_data);
              v334_acc += ((static_cast<float>(v336_data[10])) * v48_data);
              v334_acc += ((static_cast<float>(v336_data[11])) * v49_data);
              v334_acc += ((static_cast<float>(v336_data[12])) * v50_data);
              v334_acc += ((static_cast<float>(v336_data[13])) * v51_data);
              v334_acc += ((static_cast<float>(v336_data[14])) * v52_data);
              v334_acc += ((static_cast<float>(v336_data[15])) * v53_data);
              tensorforge::intel_esimd::simd<float, 16> v370_data = tensorforge::slmLoad<float, 16>(s0 + (278_i32));
              v334_acc += ((static_cast<float>(v370_data[0])) * v54_data);
              v334_acc += ((static_cast<float>(v370_data[1])) * v55_data);
              v334_acc += ((static_cast<float>(v370_data[2])) * v56_data);
              v334_acc += ((static_cast<float>(v370_data[3])) * v57_data);
              ir1.template select<16, 1>(96) = v334_acc;
              tensorforge::intel_esimd::simd<float, 16> v379_acc{};
              tensorforge::intel_esimd::simd<float, 16> v381_data = tensorforge::slmLoad<float, 16>(s0 + (7_i32));
              v379_acc += ((static_cast<float>(v381_data[0])) * v38_data);
              v379_acc += ((static_cast<float>(v381_data[1])) * v39_data);
              v379_acc += ((static_cast<float>(v381_data[2])) * v40_data);
              v379_acc += ((static_cast<float>(v381_data[3])) * v41_data);
              v379_acc += ((static_cast<float>(v381_data[4])) * v42_data);
              v379_acc += ((static_cast<float>(v381_data[5])) * v43_data);
              v379_acc += ((static_cast<float>(v381_data[6])) * v44_data);
              v379_acc += ((static_cast<float>(v381_data[7])) * v45_data);
              v379_acc += ((static_cast<float>(v381_data[8])) * v46_data);
              v379_acc += ((static_cast<float>(v381_data[9])) * v47_data);
              v379_acc += ((static_cast<float>(v381_data[10])) * v48_data);
              v379_acc += ((static_cast<float>(v381_data[11])) * v49_data);
              v379_acc += ((static_cast<float>(v381_data[12])) * v50_data);
              v379_acc += ((static_cast<float>(v381_data[13])) * v51_data);
              v379_acc += ((static_cast<float>(v381_data[14])) * v52_data);
              v379_acc += ((static_cast<float>(v381_data[15])) * v53_data);
              tensorforge::intel_esimd::simd<float, 16> v415_data = tensorforge::slmLoad<float, 16>(s0 + (279_i32));
              v379_acc += ((static_cast<float>(v415_data[0])) * v54_data);
              v379_acc += ((static_cast<float>(v415_data[1])) * v55_data);
              v379_acc += ((static_cast<float>(v415_data[2])) * v56_data);
              v379_acc += ((static_cast<float>(v415_data[3])) * v57_data);
              ir1.template select<16, 1>(112) = v379_acc;
              tensorforge::intel_esimd::simd<float, 16> v424_acc{};
              tensorforge::intel_esimd::simd<float, 16> v426_data = tensorforge::slmLoad<float, 16>(s0 + (8_i32));
              v424_acc += ((static_cast<float>(v426_data[0])) * v38_data);
              v424_acc += ((static_cast<float>(v426_data[1])) * v39_data);
              v424_acc += ((static_cast<float>(v426_data[2])) * v40_data);
              v424_acc += ((static_cast<float>(v426_data[3])) * v41_data);
              v424_acc += ((static_cast<float>(v426_data[4])) * v42_data);
              v424_acc += ((static_cast<float>(v426_data[5])) * v43_data);
              v424_acc += ((static_cast<float>(v426_data[6])) * v44_data);
              v424_acc += ((static_cast<float>(v426_data[7])) * v45_data);
              v424_acc += ((static_cast<float>(v426_data[8])) * v46_data);
              v424_acc += ((static_cast<float>(v426_data[9])) * v47_data);
              v424_acc += ((static_cast<float>(v426_data[10])) * v48_data);
              v424_acc += ((static_cast<float>(v426_data[11])) * v49_data);
              v424_acc += ((static_cast<float>(v426_data[12])) * v50_data);
              v424_acc += ((static_cast<float>(v426_data[13])) * v51_data);
              v424_acc += ((static_cast<float>(v426_data[14])) * v52_data);
              v424_acc += ((static_cast<float>(v426_data[15])) * v53_data);
              tensorforge::intel_esimd::simd<float, 16> v460_data = tensorforge::slmLoad<float, 16>(s0 + (280_i32));
              v424_acc += ((static_cast<float>(v460_data[0])) * v54_data);
              v424_acc += ((static_cast<float>(v460_data[1])) * v55_data);
              v424_acc += ((static_cast<float>(v460_data[2])) * v56_data);
              v424_acc += ((static_cast<float>(v460_data[3])) * v57_data);
              ir1.template select<16, 1>(128) = v424_acc;
              tensorforge::intel_esimd::simd<float, 16> v469_acc{};
              tensorforge::intel_esimd::simd<float, 16> v471_data = tensorforge::slmLoad<float, 16>(s0 + (9_i32));
              v469_acc += ((static_cast<float>(v471_data[0])) * v38_data);
              v469_acc += ((static_cast<float>(v471_data[1])) * v39_data);
              v469_acc += ((static_cast<float>(v471_data[2])) * v40_data);
              v469_acc += ((static_cast<float>(v471_data[3])) * v41_data);
              v469_acc += ((static_cast<float>(v471_data[4])) * v42_data);
              v469_acc += ((static_cast<float>(v471_data[5])) * v43_data);
              v469_acc += ((static_cast<float>(v471_data[6])) * v44_data);
              v469_acc += ((static_cast<float>(v471_data[7])) * v45_data);
              v469_acc += ((static_cast<float>(v471_data[8])) * v46_data);
              v469_acc += ((static_cast<float>(v471_data[9])) * v47_data);
              v469_acc += ((static_cast<float>(v471_data[10])) * v48_data);
              v469_acc += ((static_cast<float>(v471_data[11])) * v49_data);
              v469_acc += ((static_cast<float>(v471_data[12])) * v50_data);
              v469_acc += ((static_cast<float>(v471_data[13])) * v51_data);
              v469_acc += ((static_cast<float>(v471_data[14])) * v52_data);
              v469_acc += ((static_cast<float>(v471_data[15])) * v53_data);
              tensorforge::intel_esimd::simd<float, 16> v505_data = tensorforge::slmLoad<float, 16>(s0 + (281_i32));
              v469_acc += ((static_cast<float>(v505_data[0])) * v54_data);
              v469_acc += ((static_cast<float>(v505_data[1])) * v55_data);
              v469_acc += ((static_cast<float>(v505_data[2])) * v56_data);
              v469_acc += ((static_cast<float>(v505_data[3])) * v57_data);
              ir1.template select<16, 1>(144) = v469_acc;
              tensorforge::intel_esimd::simd<float, 16> v514_acc{};
              tensorforge::intel_esimd::simd<float, 16> v516_data = tensorforge::slmLoad<float, 16>(s0 + (10_i32));
              v514_acc += ((static_cast<float>(v516_data[0])) * v38_data);
              v514_acc += ((static_cast<float>(v516_data[1])) * v39_data);
              v514_acc += ((static_cast<float>(v516_data[2])) * v40_data);
              v514_acc += ((static_cast<float>(v516_data[3])) * v41_data);
              v514_acc += ((static_cast<float>(v516_data[4])) * v42_data);
              v514_acc += ((static_cast<float>(v516_data[5])) * v43_data);
              v514_acc += ((static_cast<float>(v516_data[6])) * v44_data);
              v514_acc += ((static_cast<float>(v516_data[7])) * v45_data);
              v514_acc += ((static_cast<float>(v516_data[8])) * v46_data);
              v514_acc += ((static_cast<float>(v516_data[9])) * v47_data);
              v514_acc += ((static_cast<float>(v516_data[10])) * v48_data);
              v514_acc += ((static_cast<float>(v516_data[11])) * v49_data);
              v514_acc += ((static_cast<float>(v516_data[12])) * v50_data);
              v514_acc += ((static_cast<float>(v516_data[13])) * v51_data);
              v514_acc += ((static_cast<float>(v516_data[14])) * v52_data);
              v514_acc += ((static_cast<float>(v516_data[15])) * v53_data);
              tensorforge::intel_esimd::simd<float, 16> v550_data = tensorforge::slmLoad<float, 16>(s0 + (282_i32));
              v514_acc += ((static_cast<float>(v550_data[0])) * v54_data);
              v514_acc += ((static_cast<float>(v550_data[1])) * v55_data);
              v514_acc += ((static_cast<float>(v550_data[2])) * v56_data);
              v514_acc += ((static_cast<float>(v550_data[3])) * v57_data);
              ir1.template select<16, 1>(160) = v514_acc;
              tensorforge::intel_esimd::simd<float, 16> v559_acc{};
              tensorforge::intel_esimd::simd<float, 16> v561_data = tensorforge::slmLoad<float, 16>(s0 + (11_i32));
              v559_acc += ((static_cast<float>(v561_data[0])) * v38_data);
              v559_acc += ((static_cast<float>(v561_data[1])) * v39_data);
              v559_acc += ((static_cast<float>(v561_data[2])) * v40_data);
              v559_acc += ((static_cast<float>(v561_data[3])) * v41_data);
              v559_acc += ((static_cast<float>(v561_data[4])) * v42_data);
              v559_acc += ((static_cast<float>(v561_data[5])) * v43_data);
              v559_acc += ((static_cast<float>(v561_data[6])) * v44_data);
              v559_acc += ((static_cast<float>(v561_data[7])) * v45_data);
              v559_acc += ((static_cast<float>(v561_data[8])) * v46_data);
              v559_acc += ((static_cast<float>(v561_data[9])) * v47_data);
              v559_acc += ((static_cast<float>(v561_data[10])) * v48_data);
              v559_acc += ((static_cast<float>(v561_data[11])) * v49_data);
              v559_acc += ((static_cast<float>(v561_data[12])) * v50_data);
              v559_acc += ((static_cast<float>(v561_data[13])) * v51_data);
              v559_acc += ((static_cast<float>(v561_data[14])) * v52_data);
              v559_acc += ((static_cast<float>(v561_data[15])) * v53_data);
              tensorforge::intel_esimd::simd<float, 16> v595_data = tensorforge::slmLoad<float, 16>(s0 + (283_i32));
              v559_acc += ((static_cast<float>(v595_data[0])) * v54_data);
              v559_acc += ((static_cast<float>(v595_data[1])) * v55_data);
              v559_acc += ((static_cast<float>(v595_data[2])) * v56_data);
              v559_acc += ((static_cast<float>(v595_data[3])) * v57_data);
              ir1.template select<16, 1>(176) = v559_acc;
              tensorforge::intel_esimd::simd<float, 16> v604_acc{};
              tensorforge::intel_esimd::simd<float, 16> v606_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              v604_acc += ((static_cast<float>(v606_data[0])) * v38_data);
              v604_acc += ((static_cast<float>(v606_data[1])) * v39_data);
              v604_acc += ((static_cast<float>(v606_data[2])) * v40_data);
              v604_acc += ((static_cast<float>(v606_data[3])) * v41_data);
              v604_acc += ((static_cast<float>(v606_data[4])) * v42_data);
              v604_acc += ((static_cast<float>(v606_data[5])) * v43_data);
              v604_acc += ((static_cast<float>(v606_data[6])) * v44_data);
              v604_acc += ((static_cast<float>(v606_data[7])) * v45_data);
              v604_acc += ((static_cast<float>(v606_data[8])) * v46_data);
              v604_acc += ((static_cast<float>(v606_data[9])) * v47_data);
              v604_acc += ((static_cast<float>(v606_data[10])) * v48_data);
              v604_acc += ((static_cast<float>(v606_data[11])) * v49_data);
              v604_acc += ((static_cast<float>(v606_data[12])) * v50_data);
              v604_acc += ((static_cast<float>(v606_data[13])) * v51_data);
              v604_acc += ((static_cast<float>(v606_data[14])) * v52_data);
              v604_acc += ((static_cast<float>(v606_data[15])) * v53_data);
              tensorforge::intel_esimd::simd<float, 16> v640_data = tensorforge::slmLoad<float, 16>(s0 + (284_i32));
              v604_acc += ((static_cast<float>(v640_data[0])) * v54_data);
              v604_acc += ((static_cast<float>(v640_data[1])) * v55_data);
              v604_acc += ((static_cast<float>(v640_data[2])) * v56_data);
              v604_acc += ((static_cast<float>(v640_data[3])) * v57_data);
              ir1.template select<16, 1>(192) = v604_acc;
              tensorforge::intel_esimd::simd<float, 16> v649_acc{};
              tensorforge::intel_esimd::simd<float, 16> v651_data = tensorforge::slmLoad<float, 16>(s0 + (13_i32));
              v649_acc += ((static_cast<float>(v651_data[0])) * v38_data);
              v649_acc += ((static_cast<float>(v651_data[1])) * v39_data);
              v649_acc += ((static_cast<float>(v651_data[2])) * v40_data);
              v649_acc += ((static_cast<float>(v651_data[3])) * v41_data);
              v649_acc += ((static_cast<float>(v651_data[4])) * v42_data);
              v649_acc += ((static_cast<float>(v651_data[5])) * v43_data);
              v649_acc += ((static_cast<float>(v651_data[6])) * v44_data);
              v649_acc += ((static_cast<float>(v651_data[7])) * v45_data);
              v649_acc += ((static_cast<float>(v651_data[8])) * v46_data);
              v649_acc += ((static_cast<float>(v651_data[9])) * v47_data);
              v649_acc += ((static_cast<float>(v651_data[10])) * v48_data);
              v649_acc += ((static_cast<float>(v651_data[11])) * v49_data);
              v649_acc += ((static_cast<float>(v651_data[12])) * v50_data);
              v649_acc += ((static_cast<float>(v651_data[13])) * v51_data);
              v649_acc += ((static_cast<float>(v651_data[14])) * v52_data);
              v649_acc += ((static_cast<float>(v651_data[15])) * v53_data);
              tensorforge::intel_esimd::simd<float, 16> v685_data = tensorforge::slmLoad<float, 16>(s0 + (285_i32));
              v649_acc += ((static_cast<float>(v685_data[0])) * v54_data);
              v649_acc += ((static_cast<float>(v685_data[1])) * v55_data);
              v649_acc += ((static_cast<float>(v685_data[2])) * v56_data);
              v649_acc += ((static_cast<float>(v685_data[3])) * v57_data);
              ir1.template select<16, 1>(208) = v649_acc;
              tensorforge::intel_esimd::simd<float, 16> v694_acc{};
              tensorforge::intel_esimd::simd<float, 16> v696_data = tensorforge::slmLoad<float, 16>(s0 + (14_i32));
              v694_acc += ((static_cast<float>(v696_data[0])) * v38_data);
              v694_acc += ((static_cast<float>(v696_data[1])) * v39_data);
              v694_acc += ((static_cast<float>(v696_data[2])) * v40_data);
              v694_acc += ((static_cast<float>(v696_data[3])) * v41_data);
              v694_acc += ((static_cast<float>(v696_data[4])) * v42_data);
              v694_acc += ((static_cast<float>(v696_data[5])) * v43_data);
              v694_acc += ((static_cast<float>(v696_data[6])) * v44_data);
              v694_acc += ((static_cast<float>(v696_data[7])) * v45_data);
              v694_acc += ((static_cast<float>(v696_data[8])) * v46_data);
              v694_acc += ((static_cast<float>(v696_data[9])) * v47_data);
              v694_acc += ((static_cast<float>(v696_data[10])) * v48_data);
              v694_acc += ((static_cast<float>(v696_data[11])) * v49_data);
              v694_acc += ((static_cast<float>(v696_data[12])) * v50_data);
              v694_acc += ((static_cast<float>(v696_data[13])) * v51_data);
              v694_acc += ((static_cast<float>(v696_data[14])) * v52_data);
              v694_acc += ((static_cast<float>(v696_data[15])) * v53_data);
              tensorforge::intel_esimd::simd<float, 16> v730_data = tensorforge::slmLoad<float, 16>(s0 + (286_i32));
              v694_acc += ((static_cast<float>(v730_data[0])) * v54_data);
              v694_acc += ((static_cast<float>(v730_data[1])) * v55_data);
              v694_acc += ((static_cast<float>(v730_data[2])) * v56_data);
              v694_acc += ((static_cast<float>(v730_data[3])) * v57_data);
              ir1.template select<16, 1>(224) = v694_acc;
              tensorforge::intel_esimd::simd<float, 16> v739_acc{};
              tensorforge::intel_esimd::simd<float, 16> v741_data = tensorforge::slmLoad<float, 16>(s0 + (15_i32));
              v739_acc += ((static_cast<float>(v741_data[0])) * v38_data);
              v739_acc += ((static_cast<float>(v741_data[1])) * v39_data);
              v739_acc += ((static_cast<float>(v741_data[2])) * v40_data);
              v739_acc += ((static_cast<float>(v741_data[3])) * v41_data);
              v739_acc += ((static_cast<float>(v741_data[4])) * v42_data);
              v739_acc += ((static_cast<float>(v741_data[5])) * v43_data);
              v739_acc += ((static_cast<float>(v741_data[6])) * v44_data);
              v739_acc += ((static_cast<float>(v741_data[7])) * v45_data);
              v739_acc += ((static_cast<float>(v741_data[8])) * v46_data);
              v739_acc += ((static_cast<float>(v741_data[9])) * v47_data);
              v739_acc += ((static_cast<float>(v741_data[10])) * v48_data);
              v739_acc += ((static_cast<float>(v741_data[11])) * v49_data);
              v739_acc += ((static_cast<float>(v741_data[12])) * v50_data);
              v739_acc += ((static_cast<float>(v741_data[13])) * v51_data);
              v739_acc += ((static_cast<float>(v741_data[14])) * v52_data);
              v739_acc += ((static_cast<float>(v741_data[15])) * v53_data);
              tensorforge::intel_esimd::simd<float, 16> v775_data = tensorforge::slmLoad<float, 16>(s0 + (287_i32));
              v739_acc += ((static_cast<float>(v775_data[0])) * v54_data);
              v739_acc += ((static_cast<float>(v775_data[1])) * v55_data);
              v739_acc += ((static_cast<float>(v775_data[2])) * v56_data);
              v739_acc += ((static_cast<float>(v775_data[3])) * v57_data);
              ir1.template select<16, 1>(240) = v739_acc;
              #pragma unroll
              for (int32_t v784_n1 = 0; v784_n1 < 16; ++v784_n1) {
                int32_t v785_a = v784_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v787_data(ir1.template select<12, 1>(v785_a));
                r1.template select<12, 1>(v785_a) = v787_data;
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v788_i1 = 0; v788_i1 < 16; ++v788_i1) {
                tensorforge::intel_esimd::simd<float, 12> v791_data(r1.template select<12, 1>((v788_i1 * 16)));
                v791_data.copy_to(glb_m0 + ((v788_i1 * 12)));
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

