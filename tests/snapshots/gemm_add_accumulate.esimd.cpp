// === base name ===
kernel_7b9a496497cd4837

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_7b9a496497cd4837 = {{1, 16, 1}, 16, 12, 1, 16, 9216, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_7b9a496497cd4837(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_7b9a496497cd4837(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_7b9a496497cd4837(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_7b9a496497cd4837(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_7b9a496497cd4837(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_7b9a496497cd4837(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_7b9a496497cd4837(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 192 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 128 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v17_i1 = 0; v17_i1 < 16; ++v17_i1) {
                tensorforge::intel_esimd::simd<float, 12> v22_data;
                v22_data.copy_from(glb_m1 + ((v17_i1 * 12)));
                r0.template select<12, 1>((v17_i1 * 16)) = v22_data;
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v25_ld;
              v25_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v25_ld);
              tensorforge::intel_esimd::simd<float, 64> v26_ld;
              v26_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v26_ld);
              // wait(r0 = load{g>r}(glb_m1););
              tensorforge::intel_esimd::simd<float, 128> r1(0.0f);
              // r1 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v28_i1 = 0; v28_i1 < 8; ++v28_i1) {
                tensorforge::intel_esimd::simd<float, 12> v33_data;
                v33_data.copy_from(glb_m0 + ((v28_i1 * 12)));
                r1.template select<12, 1>((v28_i1 * 16)) = v33_data;
              }
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              // wait(r1 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<float, 128> r2(0.0f);
              // ir2 = +(r0 * s0)
              // [(0, 12), (0, 8)] [(0, 16)]
              tensorforge::intel_esimd::simd<float, 128> ir2(0.0f);
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
              tensorforge::intel_esimd::simd<float, 16> v54_acc{};
              tensorforge::intel_esimd::simd<float, 16> v58_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v54_acc += ((static_cast<float>(v58_data[0])) * v38_data);
              v54_acc += ((static_cast<float>(v58_data[1])) * v39_data);
              v54_acc += ((static_cast<float>(v58_data[2])) * v40_data);
              v54_acc += ((static_cast<float>(v58_data[3])) * v41_data);
              v54_acc += ((static_cast<float>(v58_data[4])) * v42_data);
              v54_acc += ((static_cast<float>(v58_data[5])) * v43_data);
              v54_acc += ((static_cast<float>(v58_data[6])) * v44_data);
              v54_acc += ((static_cast<float>(v58_data[7])) * v45_data);
              v54_acc += ((static_cast<float>(v58_data[8])) * v46_data);
              v54_acc += ((static_cast<float>(v58_data[9])) * v47_data);
              v54_acc += ((static_cast<float>(v58_data[10])) * v48_data);
              v54_acc += ((static_cast<float>(v58_data[11])) * v49_data);
              v54_acc += ((static_cast<float>(v58_data[12])) * v50_data);
              v54_acc += ((static_cast<float>(v58_data[13])) * v51_data);
              v54_acc += ((static_cast<float>(v58_data[14])) * v52_data);
              v54_acc += ((static_cast<float>(v58_data[15])) * v53_data);
              ir2.template select<16, 1>(0) = v54_acc;
              tensorforge::intel_esimd::simd<float, 16> v91_acc{};
              tensorforge::intel_esimd::simd<float, 16> v93_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v91_acc += ((static_cast<float>(v93_data[0])) * v38_data);
              v91_acc += ((static_cast<float>(v93_data[1])) * v39_data);
              v91_acc += ((static_cast<float>(v93_data[2])) * v40_data);
              v91_acc += ((static_cast<float>(v93_data[3])) * v41_data);
              v91_acc += ((static_cast<float>(v93_data[4])) * v42_data);
              v91_acc += ((static_cast<float>(v93_data[5])) * v43_data);
              v91_acc += ((static_cast<float>(v93_data[6])) * v44_data);
              v91_acc += ((static_cast<float>(v93_data[7])) * v45_data);
              v91_acc += ((static_cast<float>(v93_data[8])) * v46_data);
              v91_acc += ((static_cast<float>(v93_data[9])) * v47_data);
              v91_acc += ((static_cast<float>(v93_data[10])) * v48_data);
              v91_acc += ((static_cast<float>(v93_data[11])) * v49_data);
              v91_acc += ((static_cast<float>(v93_data[12])) * v50_data);
              v91_acc += ((static_cast<float>(v93_data[13])) * v51_data);
              v91_acc += ((static_cast<float>(v93_data[14])) * v52_data);
              v91_acc += ((static_cast<float>(v93_data[15])) * v53_data);
              ir2.template select<16, 1>(16) = v91_acc;
              tensorforge::intel_esimd::simd<float, 16> v126_acc{};
              tensorforge::intel_esimd::simd<float, 16> v128_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v126_acc += ((static_cast<float>(v128_data[0])) * v38_data);
              v126_acc += ((static_cast<float>(v128_data[1])) * v39_data);
              v126_acc += ((static_cast<float>(v128_data[2])) * v40_data);
              v126_acc += ((static_cast<float>(v128_data[3])) * v41_data);
              v126_acc += ((static_cast<float>(v128_data[4])) * v42_data);
              v126_acc += ((static_cast<float>(v128_data[5])) * v43_data);
              v126_acc += ((static_cast<float>(v128_data[6])) * v44_data);
              v126_acc += ((static_cast<float>(v128_data[7])) * v45_data);
              v126_acc += ((static_cast<float>(v128_data[8])) * v46_data);
              v126_acc += ((static_cast<float>(v128_data[9])) * v47_data);
              v126_acc += ((static_cast<float>(v128_data[10])) * v48_data);
              v126_acc += ((static_cast<float>(v128_data[11])) * v49_data);
              v126_acc += ((static_cast<float>(v128_data[12])) * v50_data);
              v126_acc += ((static_cast<float>(v128_data[13])) * v51_data);
              v126_acc += ((static_cast<float>(v128_data[14])) * v52_data);
              v126_acc += ((static_cast<float>(v128_data[15])) * v53_data);
              ir2.template select<16, 1>(32) = v126_acc;
              tensorforge::intel_esimd::simd<float, 16> v161_acc{};
              tensorforge::intel_esimd::simd<float, 16> v163_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v161_acc += ((static_cast<float>(v163_data[0])) * v38_data);
              v161_acc += ((static_cast<float>(v163_data[1])) * v39_data);
              v161_acc += ((static_cast<float>(v163_data[2])) * v40_data);
              v161_acc += ((static_cast<float>(v163_data[3])) * v41_data);
              v161_acc += ((static_cast<float>(v163_data[4])) * v42_data);
              v161_acc += ((static_cast<float>(v163_data[5])) * v43_data);
              v161_acc += ((static_cast<float>(v163_data[6])) * v44_data);
              v161_acc += ((static_cast<float>(v163_data[7])) * v45_data);
              v161_acc += ((static_cast<float>(v163_data[8])) * v46_data);
              v161_acc += ((static_cast<float>(v163_data[9])) * v47_data);
              v161_acc += ((static_cast<float>(v163_data[10])) * v48_data);
              v161_acc += ((static_cast<float>(v163_data[11])) * v49_data);
              v161_acc += ((static_cast<float>(v163_data[12])) * v50_data);
              v161_acc += ((static_cast<float>(v163_data[13])) * v51_data);
              v161_acc += ((static_cast<float>(v163_data[14])) * v52_data);
              v161_acc += ((static_cast<float>(v163_data[15])) * v53_data);
              ir2.template select<16, 1>(48) = v161_acc;
              tensorforge::intel_esimd::simd<float, 16> v196_acc{};
              tensorforge::intel_esimd::simd<float, 16> v198_data = tensorforge::slmLoad<float, 16>(s0 + (64_i32));
              v196_acc += ((static_cast<float>(v198_data[0])) * v38_data);
              v196_acc += ((static_cast<float>(v198_data[1])) * v39_data);
              v196_acc += ((static_cast<float>(v198_data[2])) * v40_data);
              v196_acc += ((static_cast<float>(v198_data[3])) * v41_data);
              v196_acc += ((static_cast<float>(v198_data[4])) * v42_data);
              v196_acc += ((static_cast<float>(v198_data[5])) * v43_data);
              v196_acc += ((static_cast<float>(v198_data[6])) * v44_data);
              v196_acc += ((static_cast<float>(v198_data[7])) * v45_data);
              v196_acc += ((static_cast<float>(v198_data[8])) * v46_data);
              v196_acc += ((static_cast<float>(v198_data[9])) * v47_data);
              v196_acc += ((static_cast<float>(v198_data[10])) * v48_data);
              v196_acc += ((static_cast<float>(v198_data[11])) * v49_data);
              v196_acc += ((static_cast<float>(v198_data[12])) * v50_data);
              v196_acc += ((static_cast<float>(v198_data[13])) * v51_data);
              v196_acc += ((static_cast<float>(v198_data[14])) * v52_data);
              v196_acc += ((static_cast<float>(v198_data[15])) * v53_data);
              ir2.template select<16, 1>(64) = v196_acc;
              tensorforge::intel_esimd::simd<float, 16> v231_acc{};
              tensorforge::intel_esimd::simd<float, 16> v233_data = tensorforge::slmLoad<float, 16>(s0 + (80_i32));
              v231_acc += ((static_cast<float>(v233_data[0])) * v38_data);
              v231_acc += ((static_cast<float>(v233_data[1])) * v39_data);
              v231_acc += ((static_cast<float>(v233_data[2])) * v40_data);
              v231_acc += ((static_cast<float>(v233_data[3])) * v41_data);
              v231_acc += ((static_cast<float>(v233_data[4])) * v42_data);
              v231_acc += ((static_cast<float>(v233_data[5])) * v43_data);
              v231_acc += ((static_cast<float>(v233_data[6])) * v44_data);
              v231_acc += ((static_cast<float>(v233_data[7])) * v45_data);
              v231_acc += ((static_cast<float>(v233_data[8])) * v46_data);
              v231_acc += ((static_cast<float>(v233_data[9])) * v47_data);
              v231_acc += ((static_cast<float>(v233_data[10])) * v48_data);
              v231_acc += ((static_cast<float>(v233_data[11])) * v49_data);
              v231_acc += ((static_cast<float>(v233_data[12])) * v50_data);
              v231_acc += ((static_cast<float>(v233_data[13])) * v51_data);
              v231_acc += ((static_cast<float>(v233_data[14])) * v52_data);
              v231_acc += ((static_cast<float>(v233_data[15])) * v53_data);
              ir2.template select<16, 1>(80) = v231_acc;
              tensorforge::intel_esimd::simd<float, 16> v266_acc{};
              tensorforge::intel_esimd::simd<float, 16> v268_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              v266_acc += ((static_cast<float>(v268_data[0])) * v38_data);
              v266_acc += ((static_cast<float>(v268_data[1])) * v39_data);
              v266_acc += ((static_cast<float>(v268_data[2])) * v40_data);
              v266_acc += ((static_cast<float>(v268_data[3])) * v41_data);
              v266_acc += ((static_cast<float>(v268_data[4])) * v42_data);
              v266_acc += ((static_cast<float>(v268_data[5])) * v43_data);
              v266_acc += ((static_cast<float>(v268_data[6])) * v44_data);
              v266_acc += ((static_cast<float>(v268_data[7])) * v45_data);
              v266_acc += ((static_cast<float>(v268_data[8])) * v46_data);
              v266_acc += ((static_cast<float>(v268_data[9])) * v47_data);
              v266_acc += ((static_cast<float>(v268_data[10])) * v48_data);
              v266_acc += ((static_cast<float>(v268_data[11])) * v49_data);
              v266_acc += ((static_cast<float>(v268_data[12])) * v50_data);
              v266_acc += ((static_cast<float>(v268_data[13])) * v51_data);
              v266_acc += ((static_cast<float>(v268_data[14])) * v52_data);
              v266_acc += ((static_cast<float>(v268_data[15])) * v53_data);
              ir2.template select<16, 1>(96) = v266_acc;
              tensorforge::intel_esimd::simd<float, 16> v301_acc{};
              tensorforge::intel_esimd::simd<float, 16> v303_data = tensorforge::slmLoad<float, 16>(s0 + (112_i32));
              v301_acc += ((static_cast<float>(v303_data[0])) * v38_data);
              v301_acc += ((static_cast<float>(v303_data[1])) * v39_data);
              v301_acc += ((static_cast<float>(v303_data[2])) * v40_data);
              v301_acc += ((static_cast<float>(v303_data[3])) * v41_data);
              v301_acc += ((static_cast<float>(v303_data[4])) * v42_data);
              v301_acc += ((static_cast<float>(v303_data[5])) * v43_data);
              v301_acc += ((static_cast<float>(v303_data[6])) * v44_data);
              v301_acc += ((static_cast<float>(v303_data[7])) * v45_data);
              v301_acc += ((static_cast<float>(v303_data[8])) * v46_data);
              v301_acc += ((static_cast<float>(v303_data[9])) * v47_data);
              v301_acc += ((static_cast<float>(v303_data[10])) * v48_data);
              v301_acc += ((static_cast<float>(v303_data[11])) * v49_data);
              v301_acc += ((static_cast<float>(v303_data[12])) * v50_data);
              v301_acc += ((static_cast<float>(v303_data[13])) * v51_data);
              v301_acc += ((static_cast<float>(v303_data[14])) * v52_data);
              v301_acc += ((static_cast<float>(v303_data[15])) * v53_data);
              ir2.template select<16, 1>(112) = v301_acc;
              // r2 = ir2 + r1
              #pragma unroll
              for (int32_t v336_n1 = 0; v336_n1 < 8; ++v336_n1) {
                int32_t v337_a = v336_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v339_data(ir2.template select<12, 1>(v337_a));
                tensorforge::intel_esimd::simd<float, 12> v340_data(r1.template select<12, 1>(v337_a));
                r2.template select<12, 1>(v337_a) = (v340_data + v339_data);
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v342_i1 = 0; v342_i1 < 8; ++v342_i1) {
                tensorforge::intel_esimd::simd<float, 12> v345_data(r2.template select<12, 1>((v342_i1 * 16)));
                v345_data.copy_to(glb_m0 + ((v342_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}

