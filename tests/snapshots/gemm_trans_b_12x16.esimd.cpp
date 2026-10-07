// === base name ===
kernel_65c483806288fde3

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_65c483806288fde3 = {{1, 16, 1}, 16, 12, 1, 16, 23552, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_65c483806288fde3(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_65c483806288fde3(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_65c483806288fde3(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 5888 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_65c483806288fde3(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_65c483806288fde3(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_65c483806288fde3(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_65c483806288fde3(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<5888 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 23552 B shared, occupancy grid
        // operands:
        //   m0 12×16(12×16) {0..12}×{0..16} strided
        //   m1 12×20(12×20) {0..12}×{0..20} strided
        //   m2 16×20(16×20) {0..16}×{0..20} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[j,k]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":5888}],"shared_bytes":23552,"shared_elements":5888,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,16]],"name":"m0","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,20]],"name":"m1","ordered":false,"parts":1,"shape":[12,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,20]],"name":"m2","ordered":false,"parts":1,"shape":[16,20],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,20]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,20]},{"addressing":"strided","bbox":[[0,0],[16,20]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,20]}],"permute":[[0,1],[1,0]],"target":[[0,-1],[1,-1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (368 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 192 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 240 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 320 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 320> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v20_i1 = 0; v20_i1 < 20; ++v20_i1) {
                tensorforge::intel_esimd::simd<float, 12> v25_data;
                v25_data.copy_from(glb_m1 + ((v20_i1 * 12)));
                r0.template select<12, 1>((v20_i1 * 16)) = v25_data;
              }
              // s0 = load{g>s}(glb_m2[1, 0])
              #pragma unroll
              for (int32_t v28_i0 = 0; v28_i0 < 1; ++v28_i0) {
                int32_t v30_lead = v28_i0 * 16;
                #pragma unroll
                for (int32_t v29_i1 = 0; v29_i1 < 20; ++v29_i1) {
                  tensorforge::intel_esimd::simd<float, 16> v34_data;
                  v34_data.copy_from(glb_m2 + ((v30_lead + (v29_i1 * 16))));
                  tensorforge::slmStore<float, 16>(s0 + ((v30_lead + (v29_i1 * 17))), v34_data);
                }
              }
              tensorforge::intel_esimd::simd<float, 256> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 12), (0, 16)] [(0, 20)]
              tensorforge::intel_esimd::simd<float, 256> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v39_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v40_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v48_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v49_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v50_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v51_data(r0.template select<16, 1>(192));
              tensorforge::intel_esimd::simd<float, 16> v52_data(r0.template select<16, 1>(208));
              tensorforge::intel_esimd::simd<float, 16> v53_data(r0.template select<16, 1>(224));
              tensorforge::intel_esimd::simd<float, 16> v54_data(r0.template select<16, 1>(240));
              tensorforge::intel_esimd::simd<float, 16> v55_data(r0.template select<16, 1>(256));
              tensorforge::intel_esimd::simd<float, 16> v56_data(r0.template select<16, 1>(272));
              tensorforge::intel_esimd::simd<float, 16> v57_data(r0.template select<16, 1>(288));
              tensorforge::intel_esimd::simd<float, 16> v58_data(r0.template select<16, 1>(304));
              tensorforge::intel_esimd::simd<float, 16> v59_acc{};
              tensorforge::intel_esimd::simd<float, 16> v64_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v59_acc += ((static_cast<float>(v64_data[0])) * v39_data);
              v59_acc += ((static_cast<float>(v64_data[1])) * v40_data);
              v59_acc += ((static_cast<float>(v64_data[2])) * v41_data);
              v59_acc += ((static_cast<float>(v64_data[3])) * v42_data);
              v59_acc += ((static_cast<float>(v64_data[4])) * v43_data);
              v59_acc += ((static_cast<float>(v64_data[5])) * v44_data);
              v59_acc += ((static_cast<float>(v64_data[6])) * v45_data);
              v59_acc += ((static_cast<float>(v64_data[7])) * v46_data);
              v59_acc += ((static_cast<float>(v64_data[8])) * v47_data);
              v59_acc += ((static_cast<float>(v64_data[9])) * v48_data);
              v59_acc += ((static_cast<float>(v64_data[10])) * v49_data);
              v59_acc += ((static_cast<float>(v64_data[11])) * v50_data);
              v59_acc += ((static_cast<float>(v64_data[12])) * v51_data);
              v59_acc += ((static_cast<float>(v64_data[13])) * v52_data);
              v59_acc += ((static_cast<float>(v64_data[14])) * v53_data);
              v59_acc += ((static_cast<float>(v64_data[15])) * v54_data);
              tensorforge::intel_esimd::simd<float, 16> v101_data = tensorforge::slmLoad<float, 16>(s0 + (272_i32));
              v59_acc += ((static_cast<float>(v101_data[0])) * v55_data);
              v59_acc += ((static_cast<float>(v101_data[1])) * v56_data);
              v59_acc += ((static_cast<float>(v101_data[2])) * v57_data);
              v59_acc += ((static_cast<float>(v101_data[3])) * v58_data);
              ir1.template select<16, 1>(0) = v59_acc;
              tensorforge::intel_esimd::simd<float, 16> v110_acc{};
              tensorforge::intel_esimd::simd<float, 16> v112_data = tensorforge::slmLoad<float, 16>(s0 + (1_i32));
              v110_acc += ((static_cast<float>(v112_data[0])) * v39_data);
              v110_acc += ((static_cast<float>(v112_data[1])) * v40_data);
              v110_acc += ((static_cast<float>(v112_data[2])) * v41_data);
              v110_acc += ((static_cast<float>(v112_data[3])) * v42_data);
              v110_acc += ((static_cast<float>(v112_data[4])) * v43_data);
              v110_acc += ((static_cast<float>(v112_data[5])) * v44_data);
              v110_acc += ((static_cast<float>(v112_data[6])) * v45_data);
              v110_acc += ((static_cast<float>(v112_data[7])) * v46_data);
              v110_acc += ((static_cast<float>(v112_data[8])) * v47_data);
              v110_acc += ((static_cast<float>(v112_data[9])) * v48_data);
              v110_acc += ((static_cast<float>(v112_data[10])) * v49_data);
              v110_acc += ((static_cast<float>(v112_data[11])) * v50_data);
              v110_acc += ((static_cast<float>(v112_data[12])) * v51_data);
              v110_acc += ((static_cast<float>(v112_data[13])) * v52_data);
              v110_acc += ((static_cast<float>(v112_data[14])) * v53_data);
              v110_acc += ((static_cast<float>(v112_data[15])) * v54_data);
              tensorforge::intel_esimd::simd<float, 16> v146_data = tensorforge::slmLoad<float, 16>(s0 + (273_i32));
              v110_acc += ((static_cast<float>(v146_data[0])) * v55_data);
              v110_acc += ((static_cast<float>(v146_data[1])) * v56_data);
              v110_acc += ((static_cast<float>(v146_data[2])) * v57_data);
              v110_acc += ((static_cast<float>(v146_data[3])) * v58_data);
              ir1.template select<16, 1>(16) = v110_acc;
              tensorforge::intel_esimd::simd<float, 16> v155_acc{};
              tensorforge::intel_esimd::simd<float, 16> v157_data = tensorforge::slmLoad<float, 16>(s0 + (2_i32));
              v155_acc += ((static_cast<float>(v157_data[0])) * v39_data);
              v155_acc += ((static_cast<float>(v157_data[1])) * v40_data);
              v155_acc += ((static_cast<float>(v157_data[2])) * v41_data);
              v155_acc += ((static_cast<float>(v157_data[3])) * v42_data);
              v155_acc += ((static_cast<float>(v157_data[4])) * v43_data);
              v155_acc += ((static_cast<float>(v157_data[5])) * v44_data);
              v155_acc += ((static_cast<float>(v157_data[6])) * v45_data);
              v155_acc += ((static_cast<float>(v157_data[7])) * v46_data);
              v155_acc += ((static_cast<float>(v157_data[8])) * v47_data);
              v155_acc += ((static_cast<float>(v157_data[9])) * v48_data);
              v155_acc += ((static_cast<float>(v157_data[10])) * v49_data);
              v155_acc += ((static_cast<float>(v157_data[11])) * v50_data);
              v155_acc += ((static_cast<float>(v157_data[12])) * v51_data);
              v155_acc += ((static_cast<float>(v157_data[13])) * v52_data);
              v155_acc += ((static_cast<float>(v157_data[14])) * v53_data);
              v155_acc += ((static_cast<float>(v157_data[15])) * v54_data);
              tensorforge::intel_esimd::simd<float, 16> v191_data = tensorforge::slmLoad<float, 16>(s0 + (274_i32));
              v155_acc += ((static_cast<float>(v191_data[0])) * v55_data);
              v155_acc += ((static_cast<float>(v191_data[1])) * v56_data);
              v155_acc += ((static_cast<float>(v191_data[2])) * v57_data);
              v155_acc += ((static_cast<float>(v191_data[3])) * v58_data);
              ir1.template select<16, 1>(32) = v155_acc;
              tensorforge::intel_esimd::simd<float, 16> v200_acc{};
              tensorforge::intel_esimd::simd<float, 16> v202_data = tensorforge::slmLoad<float, 16>(s0 + (3_i32));
              v200_acc += ((static_cast<float>(v202_data[0])) * v39_data);
              v200_acc += ((static_cast<float>(v202_data[1])) * v40_data);
              v200_acc += ((static_cast<float>(v202_data[2])) * v41_data);
              v200_acc += ((static_cast<float>(v202_data[3])) * v42_data);
              v200_acc += ((static_cast<float>(v202_data[4])) * v43_data);
              v200_acc += ((static_cast<float>(v202_data[5])) * v44_data);
              v200_acc += ((static_cast<float>(v202_data[6])) * v45_data);
              v200_acc += ((static_cast<float>(v202_data[7])) * v46_data);
              v200_acc += ((static_cast<float>(v202_data[8])) * v47_data);
              v200_acc += ((static_cast<float>(v202_data[9])) * v48_data);
              v200_acc += ((static_cast<float>(v202_data[10])) * v49_data);
              v200_acc += ((static_cast<float>(v202_data[11])) * v50_data);
              v200_acc += ((static_cast<float>(v202_data[12])) * v51_data);
              v200_acc += ((static_cast<float>(v202_data[13])) * v52_data);
              v200_acc += ((static_cast<float>(v202_data[14])) * v53_data);
              v200_acc += ((static_cast<float>(v202_data[15])) * v54_data);
              tensorforge::intel_esimd::simd<float, 16> v236_data = tensorforge::slmLoad<float, 16>(s0 + (275_i32));
              v200_acc += ((static_cast<float>(v236_data[0])) * v55_data);
              v200_acc += ((static_cast<float>(v236_data[1])) * v56_data);
              v200_acc += ((static_cast<float>(v236_data[2])) * v57_data);
              v200_acc += ((static_cast<float>(v236_data[3])) * v58_data);
              ir1.template select<16, 1>(48) = v200_acc;
              tensorforge::intel_esimd::simd<float, 16> v245_acc{};
              tensorforge::intel_esimd::simd<float, 16> v247_data = tensorforge::slmLoad<float, 16>(s0 + (4_i32));
              v245_acc += ((static_cast<float>(v247_data[0])) * v39_data);
              v245_acc += ((static_cast<float>(v247_data[1])) * v40_data);
              v245_acc += ((static_cast<float>(v247_data[2])) * v41_data);
              v245_acc += ((static_cast<float>(v247_data[3])) * v42_data);
              v245_acc += ((static_cast<float>(v247_data[4])) * v43_data);
              v245_acc += ((static_cast<float>(v247_data[5])) * v44_data);
              v245_acc += ((static_cast<float>(v247_data[6])) * v45_data);
              v245_acc += ((static_cast<float>(v247_data[7])) * v46_data);
              v245_acc += ((static_cast<float>(v247_data[8])) * v47_data);
              v245_acc += ((static_cast<float>(v247_data[9])) * v48_data);
              v245_acc += ((static_cast<float>(v247_data[10])) * v49_data);
              v245_acc += ((static_cast<float>(v247_data[11])) * v50_data);
              v245_acc += ((static_cast<float>(v247_data[12])) * v51_data);
              v245_acc += ((static_cast<float>(v247_data[13])) * v52_data);
              v245_acc += ((static_cast<float>(v247_data[14])) * v53_data);
              v245_acc += ((static_cast<float>(v247_data[15])) * v54_data);
              tensorforge::intel_esimd::simd<float, 16> v281_data = tensorforge::slmLoad<float, 16>(s0 + (276_i32));
              v245_acc += ((static_cast<float>(v281_data[0])) * v55_data);
              v245_acc += ((static_cast<float>(v281_data[1])) * v56_data);
              v245_acc += ((static_cast<float>(v281_data[2])) * v57_data);
              v245_acc += ((static_cast<float>(v281_data[3])) * v58_data);
              ir1.template select<16, 1>(64) = v245_acc;
              tensorforge::intel_esimd::simd<float, 16> v290_acc{};
              tensorforge::intel_esimd::simd<float, 16> v292_data = tensorforge::slmLoad<float, 16>(s0 + (5_i32));
              v290_acc += ((static_cast<float>(v292_data[0])) * v39_data);
              v290_acc += ((static_cast<float>(v292_data[1])) * v40_data);
              v290_acc += ((static_cast<float>(v292_data[2])) * v41_data);
              v290_acc += ((static_cast<float>(v292_data[3])) * v42_data);
              v290_acc += ((static_cast<float>(v292_data[4])) * v43_data);
              v290_acc += ((static_cast<float>(v292_data[5])) * v44_data);
              v290_acc += ((static_cast<float>(v292_data[6])) * v45_data);
              v290_acc += ((static_cast<float>(v292_data[7])) * v46_data);
              v290_acc += ((static_cast<float>(v292_data[8])) * v47_data);
              v290_acc += ((static_cast<float>(v292_data[9])) * v48_data);
              v290_acc += ((static_cast<float>(v292_data[10])) * v49_data);
              v290_acc += ((static_cast<float>(v292_data[11])) * v50_data);
              v290_acc += ((static_cast<float>(v292_data[12])) * v51_data);
              v290_acc += ((static_cast<float>(v292_data[13])) * v52_data);
              v290_acc += ((static_cast<float>(v292_data[14])) * v53_data);
              v290_acc += ((static_cast<float>(v292_data[15])) * v54_data);
              tensorforge::intel_esimd::simd<float, 16> v326_data = tensorforge::slmLoad<float, 16>(s0 + (277_i32));
              v290_acc += ((static_cast<float>(v326_data[0])) * v55_data);
              v290_acc += ((static_cast<float>(v326_data[1])) * v56_data);
              v290_acc += ((static_cast<float>(v326_data[2])) * v57_data);
              v290_acc += ((static_cast<float>(v326_data[3])) * v58_data);
              ir1.template select<16, 1>(80) = v290_acc;
              tensorforge::intel_esimd::simd<float, 16> v335_acc{};
              tensorforge::intel_esimd::simd<float, 16> v337_data = tensorforge::slmLoad<float, 16>(s0 + (6_i32));
              v335_acc += ((static_cast<float>(v337_data[0])) * v39_data);
              v335_acc += ((static_cast<float>(v337_data[1])) * v40_data);
              v335_acc += ((static_cast<float>(v337_data[2])) * v41_data);
              v335_acc += ((static_cast<float>(v337_data[3])) * v42_data);
              v335_acc += ((static_cast<float>(v337_data[4])) * v43_data);
              v335_acc += ((static_cast<float>(v337_data[5])) * v44_data);
              v335_acc += ((static_cast<float>(v337_data[6])) * v45_data);
              v335_acc += ((static_cast<float>(v337_data[7])) * v46_data);
              v335_acc += ((static_cast<float>(v337_data[8])) * v47_data);
              v335_acc += ((static_cast<float>(v337_data[9])) * v48_data);
              v335_acc += ((static_cast<float>(v337_data[10])) * v49_data);
              v335_acc += ((static_cast<float>(v337_data[11])) * v50_data);
              v335_acc += ((static_cast<float>(v337_data[12])) * v51_data);
              v335_acc += ((static_cast<float>(v337_data[13])) * v52_data);
              v335_acc += ((static_cast<float>(v337_data[14])) * v53_data);
              v335_acc += ((static_cast<float>(v337_data[15])) * v54_data);
              tensorforge::intel_esimd::simd<float, 16> v371_data = tensorforge::slmLoad<float, 16>(s0 + (278_i32));
              v335_acc += ((static_cast<float>(v371_data[0])) * v55_data);
              v335_acc += ((static_cast<float>(v371_data[1])) * v56_data);
              v335_acc += ((static_cast<float>(v371_data[2])) * v57_data);
              v335_acc += ((static_cast<float>(v371_data[3])) * v58_data);
              ir1.template select<16, 1>(96) = v335_acc;
              tensorforge::intel_esimd::simd<float, 16> v380_acc{};
              tensorforge::intel_esimd::simd<float, 16> v382_data = tensorforge::slmLoad<float, 16>(s0 + (7_i32));
              v380_acc += ((static_cast<float>(v382_data[0])) * v39_data);
              v380_acc += ((static_cast<float>(v382_data[1])) * v40_data);
              v380_acc += ((static_cast<float>(v382_data[2])) * v41_data);
              v380_acc += ((static_cast<float>(v382_data[3])) * v42_data);
              v380_acc += ((static_cast<float>(v382_data[4])) * v43_data);
              v380_acc += ((static_cast<float>(v382_data[5])) * v44_data);
              v380_acc += ((static_cast<float>(v382_data[6])) * v45_data);
              v380_acc += ((static_cast<float>(v382_data[7])) * v46_data);
              v380_acc += ((static_cast<float>(v382_data[8])) * v47_data);
              v380_acc += ((static_cast<float>(v382_data[9])) * v48_data);
              v380_acc += ((static_cast<float>(v382_data[10])) * v49_data);
              v380_acc += ((static_cast<float>(v382_data[11])) * v50_data);
              v380_acc += ((static_cast<float>(v382_data[12])) * v51_data);
              v380_acc += ((static_cast<float>(v382_data[13])) * v52_data);
              v380_acc += ((static_cast<float>(v382_data[14])) * v53_data);
              v380_acc += ((static_cast<float>(v382_data[15])) * v54_data);
              tensorforge::intel_esimd::simd<float, 16> v416_data = tensorforge::slmLoad<float, 16>(s0 + (279_i32));
              v380_acc += ((static_cast<float>(v416_data[0])) * v55_data);
              v380_acc += ((static_cast<float>(v416_data[1])) * v56_data);
              v380_acc += ((static_cast<float>(v416_data[2])) * v57_data);
              v380_acc += ((static_cast<float>(v416_data[3])) * v58_data);
              ir1.template select<16, 1>(112) = v380_acc;
              tensorforge::intel_esimd::simd<float, 16> v425_acc{};
              tensorforge::intel_esimd::simd<float, 16> v427_data = tensorforge::slmLoad<float, 16>(s0 + (8_i32));
              v425_acc += ((static_cast<float>(v427_data[0])) * v39_data);
              v425_acc += ((static_cast<float>(v427_data[1])) * v40_data);
              v425_acc += ((static_cast<float>(v427_data[2])) * v41_data);
              v425_acc += ((static_cast<float>(v427_data[3])) * v42_data);
              v425_acc += ((static_cast<float>(v427_data[4])) * v43_data);
              v425_acc += ((static_cast<float>(v427_data[5])) * v44_data);
              v425_acc += ((static_cast<float>(v427_data[6])) * v45_data);
              v425_acc += ((static_cast<float>(v427_data[7])) * v46_data);
              v425_acc += ((static_cast<float>(v427_data[8])) * v47_data);
              v425_acc += ((static_cast<float>(v427_data[9])) * v48_data);
              v425_acc += ((static_cast<float>(v427_data[10])) * v49_data);
              v425_acc += ((static_cast<float>(v427_data[11])) * v50_data);
              v425_acc += ((static_cast<float>(v427_data[12])) * v51_data);
              v425_acc += ((static_cast<float>(v427_data[13])) * v52_data);
              v425_acc += ((static_cast<float>(v427_data[14])) * v53_data);
              v425_acc += ((static_cast<float>(v427_data[15])) * v54_data);
              tensorforge::intel_esimd::simd<float, 16> v461_data = tensorforge::slmLoad<float, 16>(s0 + (280_i32));
              v425_acc += ((static_cast<float>(v461_data[0])) * v55_data);
              v425_acc += ((static_cast<float>(v461_data[1])) * v56_data);
              v425_acc += ((static_cast<float>(v461_data[2])) * v57_data);
              v425_acc += ((static_cast<float>(v461_data[3])) * v58_data);
              ir1.template select<16, 1>(128) = v425_acc;
              tensorforge::intel_esimd::simd<float, 16> v470_acc{};
              tensorforge::intel_esimd::simd<float, 16> v472_data = tensorforge::slmLoad<float, 16>(s0 + (9_i32));
              v470_acc += ((static_cast<float>(v472_data[0])) * v39_data);
              v470_acc += ((static_cast<float>(v472_data[1])) * v40_data);
              v470_acc += ((static_cast<float>(v472_data[2])) * v41_data);
              v470_acc += ((static_cast<float>(v472_data[3])) * v42_data);
              v470_acc += ((static_cast<float>(v472_data[4])) * v43_data);
              v470_acc += ((static_cast<float>(v472_data[5])) * v44_data);
              v470_acc += ((static_cast<float>(v472_data[6])) * v45_data);
              v470_acc += ((static_cast<float>(v472_data[7])) * v46_data);
              v470_acc += ((static_cast<float>(v472_data[8])) * v47_data);
              v470_acc += ((static_cast<float>(v472_data[9])) * v48_data);
              v470_acc += ((static_cast<float>(v472_data[10])) * v49_data);
              v470_acc += ((static_cast<float>(v472_data[11])) * v50_data);
              v470_acc += ((static_cast<float>(v472_data[12])) * v51_data);
              v470_acc += ((static_cast<float>(v472_data[13])) * v52_data);
              v470_acc += ((static_cast<float>(v472_data[14])) * v53_data);
              v470_acc += ((static_cast<float>(v472_data[15])) * v54_data);
              tensorforge::intel_esimd::simd<float, 16> v506_data = tensorforge::slmLoad<float, 16>(s0 + (281_i32));
              v470_acc += ((static_cast<float>(v506_data[0])) * v55_data);
              v470_acc += ((static_cast<float>(v506_data[1])) * v56_data);
              v470_acc += ((static_cast<float>(v506_data[2])) * v57_data);
              v470_acc += ((static_cast<float>(v506_data[3])) * v58_data);
              ir1.template select<16, 1>(144) = v470_acc;
              tensorforge::intel_esimd::simd<float, 16> v515_acc{};
              tensorforge::intel_esimd::simd<float, 16> v517_data = tensorforge::slmLoad<float, 16>(s0 + (10_i32));
              v515_acc += ((static_cast<float>(v517_data[0])) * v39_data);
              v515_acc += ((static_cast<float>(v517_data[1])) * v40_data);
              v515_acc += ((static_cast<float>(v517_data[2])) * v41_data);
              v515_acc += ((static_cast<float>(v517_data[3])) * v42_data);
              v515_acc += ((static_cast<float>(v517_data[4])) * v43_data);
              v515_acc += ((static_cast<float>(v517_data[5])) * v44_data);
              v515_acc += ((static_cast<float>(v517_data[6])) * v45_data);
              v515_acc += ((static_cast<float>(v517_data[7])) * v46_data);
              v515_acc += ((static_cast<float>(v517_data[8])) * v47_data);
              v515_acc += ((static_cast<float>(v517_data[9])) * v48_data);
              v515_acc += ((static_cast<float>(v517_data[10])) * v49_data);
              v515_acc += ((static_cast<float>(v517_data[11])) * v50_data);
              v515_acc += ((static_cast<float>(v517_data[12])) * v51_data);
              v515_acc += ((static_cast<float>(v517_data[13])) * v52_data);
              v515_acc += ((static_cast<float>(v517_data[14])) * v53_data);
              v515_acc += ((static_cast<float>(v517_data[15])) * v54_data);
              tensorforge::intel_esimd::simd<float, 16> v551_data = tensorforge::slmLoad<float, 16>(s0 + (282_i32));
              v515_acc += ((static_cast<float>(v551_data[0])) * v55_data);
              v515_acc += ((static_cast<float>(v551_data[1])) * v56_data);
              v515_acc += ((static_cast<float>(v551_data[2])) * v57_data);
              v515_acc += ((static_cast<float>(v551_data[3])) * v58_data);
              ir1.template select<16, 1>(160) = v515_acc;
              tensorforge::intel_esimd::simd<float, 16> v560_acc{};
              tensorforge::intel_esimd::simd<float, 16> v562_data = tensorforge::slmLoad<float, 16>(s0 + (11_i32));
              v560_acc += ((static_cast<float>(v562_data[0])) * v39_data);
              v560_acc += ((static_cast<float>(v562_data[1])) * v40_data);
              v560_acc += ((static_cast<float>(v562_data[2])) * v41_data);
              v560_acc += ((static_cast<float>(v562_data[3])) * v42_data);
              v560_acc += ((static_cast<float>(v562_data[4])) * v43_data);
              v560_acc += ((static_cast<float>(v562_data[5])) * v44_data);
              v560_acc += ((static_cast<float>(v562_data[6])) * v45_data);
              v560_acc += ((static_cast<float>(v562_data[7])) * v46_data);
              v560_acc += ((static_cast<float>(v562_data[8])) * v47_data);
              v560_acc += ((static_cast<float>(v562_data[9])) * v48_data);
              v560_acc += ((static_cast<float>(v562_data[10])) * v49_data);
              v560_acc += ((static_cast<float>(v562_data[11])) * v50_data);
              v560_acc += ((static_cast<float>(v562_data[12])) * v51_data);
              v560_acc += ((static_cast<float>(v562_data[13])) * v52_data);
              v560_acc += ((static_cast<float>(v562_data[14])) * v53_data);
              v560_acc += ((static_cast<float>(v562_data[15])) * v54_data);
              tensorforge::intel_esimd::simd<float, 16> v596_data = tensorforge::slmLoad<float, 16>(s0 + (283_i32));
              v560_acc += ((static_cast<float>(v596_data[0])) * v55_data);
              v560_acc += ((static_cast<float>(v596_data[1])) * v56_data);
              v560_acc += ((static_cast<float>(v596_data[2])) * v57_data);
              v560_acc += ((static_cast<float>(v596_data[3])) * v58_data);
              ir1.template select<16, 1>(176) = v560_acc;
              tensorforge::intel_esimd::simd<float, 16> v605_acc{};
              tensorforge::intel_esimd::simd<float, 16> v607_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              v605_acc += ((static_cast<float>(v607_data[0])) * v39_data);
              v605_acc += ((static_cast<float>(v607_data[1])) * v40_data);
              v605_acc += ((static_cast<float>(v607_data[2])) * v41_data);
              v605_acc += ((static_cast<float>(v607_data[3])) * v42_data);
              v605_acc += ((static_cast<float>(v607_data[4])) * v43_data);
              v605_acc += ((static_cast<float>(v607_data[5])) * v44_data);
              v605_acc += ((static_cast<float>(v607_data[6])) * v45_data);
              v605_acc += ((static_cast<float>(v607_data[7])) * v46_data);
              v605_acc += ((static_cast<float>(v607_data[8])) * v47_data);
              v605_acc += ((static_cast<float>(v607_data[9])) * v48_data);
              v605_acc += ((static_cast<float>(v607_data[10])) * v49_data);
              v605_acc += ((static_cast<float>(v607_data[11])) * v50_data);
              v605_acc += ((static_cast<float>(v607_data[12])) * v51_data);
              v605_acc += ((static_cast<float>(v607_data[13])) * v52_data);
              v605_acc += ((static_cast<float>(v607_data[14])) * v53_data);
              v605_acc += ((static_cast<float>(v607_data[15])) * v54_data);
              tensorforge::intel_esimd::simd<float, 16> v641_data = tensorforge::slmLoad<float, 16>(s0 + (284_i32));
              v605_acc += ((static_cast<float>(v641_data[0])) * v55_data);
              v605_acc += ((static_cast<float>(v641_data[1])) * v56_data);
              v605_acc += ((static_cast<float>(v641_data[2])) * v57_data);
              v605_acc += ((static_cast<float>(v641_data[3])) * v58_data);
              ir1.template select<16, 1>(192) = v605_acc;
              tensorforge::intel_esimd::simd<float, 16> v650_acc{};
              tensorforge::intel_esimd::simd<float, 16> v652_data = tensorforge::slmLoad<float, 16>(s0 + (13_i32));
              v650_acc += ((static_cast<float>(v652_data[0])) * v39_data);
              v650_acc += ((static_cast<float>(v652_data[1])) * v40_data);
              v650_acc += ((static_cast<float>(v652_data[2])) * v41_data);
              v650_acc += ((static_cast<float>(v652_data[3])) * v42_data);
              v650_acc += ((static_cast<float>(v652_data[4])) * v43_data);
              v650_acc += ((static_cast<float>(v652_data[5])) * v44_data);
              v650_acc += ((static_cast<float>(v652_data[6])) * v45_data);
              v650_acc += ((static_cast<float>(v652_data[7])) * v46_data);
              v650_acc += ((static_cast<float>(v652_data[8])) * v47_data);
              v650_acc += ((static_cast<float>(v652_data[9])) * v48_data);
              v650_acc += ((static_cast<float>(v652_data[10])) * v49_data);
              v650_acc += ((static_cast<float>(v652_data[11])) * v50_data);
              v650_acc += ((static_cast<float>(v652_data[12])) * v51_data);
              v650_acc += ((static_cast<float>(v652_data[13])) * v52_data);
              v650_acc += ((static_cast<float>(v652_data[14])) * v53_data);
              v650_acc += ((static_cast<float>(v652_data[15])) * v54_data);
              tensorforge::intel_esimd::simd<float, 16> v686_data = tensorforge::slmLoad<float, 16>(s0 + (285_i32));
              v650_acc += ((static_cast<float>(v686_data[0])) * v55_data);
              v650_acc += ((static_cast<float>(v686_data[1])) * v56_data);
              v650_acc += ((static_cast<float>(v686_data[2])) * v57_data);
              v650_acc += ((static_cast<float>(v686_data[3])) * v58_data);
              ir1.template select<16, 1>(208) = v650_acc;
              tensorforge::intel_esimd::simd<float, 16> v695_acc{};
              tensorforge::intel_esimd::simd<float, 16> v697_data = tensorforge::slmLoad<float, 16>(s0 + (14_i32));
              v695_acc += ((static_cast<float>(v697_data[0])) * v39_data);
              v695_acc += ((static_cast<float>(v697_data[1])) * v40_data);
              v695_acc += ((static_cast<float>(v697_data[2])) * v41_data);
              v695_acc += ((static_cast<float>(v697_data[3])) * v42_data);
              v695_acc += ((static_cast<float>(v697_data[4])) * v43_data);
              v695_acc += ((static_cast<float>(v697_data[5])) * v44_data);
              v695_acc += ((static_cast<float>(v697_data[6])) * v45_data);
              v695_acc += ((static_cast<float>(v697_data[7])) * v46_data);
              v695_acc += ((static_cast<float>(v697_data[8])) * v47_data);
              v695_acc += ((static_cast<float>(v697_data[9])) * v48_data);
              v695_acc += ((static_cast<float>(v697_data[10])) * v49_data);
              v695_acc += ((static_cast<float>(v697_data[11])) * v50_data);
              v695_acc += ((static_cast<float>(v697_data[12])) * v51_data);
              v695_acc += ((static_cast<float>(v697_data[13])) * v52_data);
              v695_acc += ((static_cast<float>(v697_data[14])) * v53_data);
              v695_acc += ((static_cast<float>(v697_data[15])) * v54_data);
              tensorforge::intel_esimd::simd<float, 16> v731_data = tensorforge::slmLoad<float, 16>(s0 + (286_i32));
              v695_acc += ((static_cast<float>(v731_data[0])) * v55_data);
              v695_acc += ((static_cast<float>(v731_data[1])) * v56_data);
              v695_acc += ((static_cast<float>(v731_data[2])) * v57_data);
              v695_acc += ((static_cast<float>(v731_data[3])) * v58_data);
              ir1.template select<16, 1>(224) = v695_acc;
              tensorforge::intel_esimd::simd<float, 16> v740_acc{};
              tensorforge::intel_esimd::simd<float, 16> v742_data = tensorforge::slmLoad<float, 16>(s0 + (15_i32));
              v740_acc += ((static_cast<float>(v742_data[0])) * v39_data);
              v740_acc += ((static_cast<float>(v742_data[1])) * v40_data);
              v740_acc += ((static_cast<float>(v742_data[2])) * v41_data);
              v740_acc += ((static_cast<float>(v742_data[3])) * v42_data);
              v740_acc += ((static_cast<float>(v742_data[4])) * v43_data);
              v740_acc += ((static_cast<float>(v742_data[5])) * v44_data);
              v740_acc += ((static_cast<float>(v742_data[6])) * v45_data);
              v740_acc += ((static_cast<float>(v742_data[7])) * v46_data);
              v740_acc += ((static_cast<float>(v742_data[8])) * v47_data);
              v740_acc += ((static_cast<float>(v742_data[9])) * v48_data);
              v740_acc += ((static_cast<float>(v742_data[10])) * v49_data);
              v740_acc += ((static_cast<float>(v742_data[11])) * v50_data);
              v740_acc += ((static_cast<float>(v742_data[12])) * v51_data);
              v740_acc += ((static_cast<float>(v742_data[13])) * v52_data);
              v740_acc += ((static_cast<float>(v742_data[14])) * v53_data);
              v740_acc += ((static_cast<float>(v742_data[15])) * v54_data);
              tensorforge::intel_esimd::simd<float, 16> v776_data = tensorforge::slmLoad<float, 16>(s0 + (287_i32));
              v740_acc += ((static_cast<float>(v776_data[0])) * v55_data);
              v740_acc += ((static_cast<float>(v776_data[1])) * v56_data);
              v740_acc += ((static_cast<float>(v776_data[2])) * v57_data);
              v740_acc += ((static_cast<float>(v776_data[3])) * v58_data);
              ir1.template select<16, 1>(240) = v740_acc;
              // r1 = ir1
              #pragma unroll
              for (int32_t v785_n1 = 0; v785_n1 < 16; ++v785_n1) {
                int32_t v786_a = v785_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v788_data(ir1.template select<12, 1>(v786_a));
                r1.template select<12, 1>(v786_a) = v788_data;
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v789_i1 = 0; v789_i1 < 16; ++v789_i1) {
                tensorforge::intel_esimd::simd<float, 12> v792_data(r1.template select<12, 1>((v789_i1 * 16)));
                v792_data.copy_to(glb_m0 + ((v789_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}

