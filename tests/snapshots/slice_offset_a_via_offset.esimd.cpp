// === base name ===
kernel_da04320c636949a1

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_da04320c636949a1 = {{1, 16, 1}, 16, 12, 1, 16, 9216, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_da04320c636949a1(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_da04320c636949a1(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_da04320c636949a1(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_da04320c636949a1(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_da04320c636949a1(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_da04320c636949a1(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_da04320c636949a1(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2304 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 9216 B shared, occupancy grid
        // operands:
        //   m0 12×8(12×8) {0..12}×{0..8} strided
        //   m1 32×16(32×16) {0..32}×{0..16} strided
        //   m2 16×8(16×8) {0..16}×{0..8} strided
        // operations:
        //   m0[i,j] = m1[i,k]@{4..16}×{0..16} × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2304}],"shared_bytes":9216,"shared_elements":2304,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,8]],"name":"m0","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[32,16]],"name":"m1","ordered":false,"parts":1,"shape":[32,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m1","offset":[4,0],"shape":[32,16]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (144 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (128);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v11_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v11_batchId0 < numElements0; v11_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v12_ahead1 = v11_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v11_batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v11_batchId0 * 512 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v11_batchId0 * 128 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v23_i1 = 0; v23_i1 < 16; ++v23_i1) {
                tensorforge::intel_esimd::simd<float, 12> v29_data;
                v29_data.copy_from(glb_m1 + ((4_i32 + (v23_i1 * 32))));
                r0.template select<12, 1>((v23_i1 * 16)) = v29_data;
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v32_ld;
              v32_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v32_ld);
              tensorforge::intel_esimd::simd<float, 64> v33_ld;
              v33_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v33_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 128> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 12), (0, 8)] [(0, 16)]
              tensorforge::intel_esimd::simd<float, 128> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v36_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v37_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v38_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v39_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v40_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v48_data(r0.template select<16, 1>(192));
              tensorforge::intel_esimd::simd<float, 16> v49_data(r0.template select<16, 1>(208));
              tensorforge::intel_esimd::simd<float, 16> v50_data(r0.template select<16, 1>(224));
              tensorforge::intel_esimd::simd<float, 16> v51_data(r0.template select<16, 1>(240));
              tensorforge::intel_esimd::simd<float, 16> v52_acc{};
              tensorforge::intel_esimd::simd<float, 16> v56_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v52_acc += ((static_cast<float>(v56_data[0])) * v36_data);
              v52_acc += ((static_cast<float>(v56_data[1])) * v37_data);
              v52_acc += ((static_cast<float>(v56_data[2])) * v38_data);
              v52_acc += ((static_cast<float>(v56_data[3])) * v39_data);
              v52_acc += ((static_cast<float>(v56_data[4])) * v40_data);
              v52_acc += ((static_cast<float>(v56_data[5])) * v41_data);
              v52_acc += ((static_cast<float>(v56_data[6])) * v42_data);
              v52_acc += ((static_cast<float>(v56_data[7])) * v43_data);
              v52_acc += ((static_cast<float>(v56_data[8])) * v44_data);
              v52_acc += ((static_cast<float>(v56_data[9])) * v45_data);
              v52_acc += ((static_cast<float>(v56_data[10])) * v46_data);
              v52_acc += ((static_cast<float>(v56_data[11])) * v47_data);
              v52_acc += ((static_cast<float>(v56_data[12])) * v48_data);
              v52_acc += ((static_cast<float>(v56_data[13])) * v49_data);
              v52_acc += ((static_cast<float>(v56_data[14])) * v50_data);
              v52_acc += ((static_cast<float>(v56_data[15])) * v51_data);
              ir1.template select<16, 1>(0) = v52_acc;
              tensorforge::intel_esimd::simd<float, 16> v89_acc{};
              tensorforge::intel_esimd::simd<float, 16> v91_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v89_acc += ((static_cast<float>(v91_data[0])) * v36_data);
              v89_acc += ((static_cast<float>(v91_data[1])) * v37_data);
              v89_acc += ((static_cast<float>(v91_data[2])) * v38_data);
              v89_acc += ((static_cast<float>(v91_data[3])) * v39_data);
              v89_acc += ((static_cast<float>(v91_data[4])) * v40_data);
              v89_acc += ((static_cast<float>(v91_data[5])) * v41_data);
              v89_acc += ((static_cast<float>(v91_data[6])) * v42_data);
              v89_acc += ((static_cast<float>(v91_data[7])) * v43_data);
              v89_acc += ((static_cast<float>(v91_data[8])) * v44_data);
              v89_acc += ((static_cast<float>(v91_data[9])) * v45_data);
              v89_acc += ((static_cast<float>(v91_data[10])) * v46_data);
              v89_acc += ((static_cast<float>(v91_data[11])) * v47_data);
              v89_acc += ((static_cast<float>(v91_data[12])) * v48_data);
              v89_acc += ((static_cast<float>(v91_data[13])) * v49_data);
              v89_acc += ((static_cast<float>(v91_data[14])) * v50_data);
              v89_acc += ((static_cast<float>(v91_data[15])) * v51_data);
              ir1.template select<16, 1>(16) = v89_acc;
              tensorforge::intel_esimd::simd<float, 16> v124_acc{};
              tensorforge::intel_esimd::simd<float, 16> v126_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v124_acc += ((static_cast<float>(v126_data[0])) * v36_data);
              v124_acc += ((static_cast<float>(v126_data[1])) * v37_data);
              v124_acc += ((static_cast<float>(v126_data[2])) * v38_data);
              v124_acc += ((static_cast<float>(v126_data[3])) * v39_data);
              v124_acc += ((static_cast<float>(v126_data[4])) * v40_data);
              v124_acc += ((static_cast<float>(v126_data[5])) * v41_data);
              v124_acc += ((static_cast<float>(v126_data[6])) * v42_data);
              v124_acc += ((static_cast<float>(v126_data[7])) * v43_data);
              v124_acc += ((static_cast<float>(v126_data[8])) * v44_data);
              v124_acc += ((static_cast<float>(v126_data[9])) * v45_data);
              v124_acc += ((static_cast<float>(v126_data[10])) * v46_data);
              v124_acc += ((static_cast<float>(v126_data[11])) * v47_data);
              v124_acc += ((static_cast<float>(v126_data[12])) * v48_data);
              v124_acc += ((static_cast<float>(v126_data[13])) * v49_data);
              v124_acc += ((static_cast<float>(v126_data[14])) * v50_data);
              v124_acc += ((static_cast<float>(v126_data[15])) * v51_data);
              ir1.template select<16, 1>(32) = v124_acc;
              tensorforge::intel_esimd::simd<float, 16> v159_acc{};
              tensorforge::intel_esimd::simd<float, 16> v161_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v159_acc += ((static_cast<float>(v161_data[0])) * v36_data);
              v159_acc += ((static_cast<float>(v161_data[1])) * v37_data);
              v159_acc += ((static_cast<float>(v161_data[2])) * v38_data);
              v159_acc += ((static_cast<float>(v161_data[3])) * v39_data);
              v159_acc += ((static_cast<float>(v161_data[4])) * v40_data);
              v159_acc += ((static_cast<float>(v161_data[5])) * v41_data);
              v159_acc += ((static_cast<float>(v161_data[6])) * v42_data);
              v159_acc += ((static_cast<float>(v161_data[7])) * v43_data);
              v159_acc += ((static_cast<float>(v161_data[8])) * v44_data);
              v159_acc += ((static_cast<float>(v161_data[9])) * v45_data);
              v159_acc += ((static_cast<float>(v161_data[10])) * v46_data);
              v159_acc += ((static_cast<float>(v161_data[11])) * v47_data);
              v159_acc += ((static_cast<float>(v161_data[12])) * v48_data);
              v159_acc += ((static_cast<float>(v161_data[13])) * v49_data);
              v159_acc += ((static_cast<float>(v161_data[14])) * v50_data);
              v159_acc += ((static_cast<float>(v161_data[15])) * v51_data);
              ir1.template select<16, 1>(48) = v159_acc;
              tensorforge::intel_esimd::simd<float, 16> v194_acc{};
              tensorforge::intel_esimd::simd<float, 16> v196_data = tensorforge::slmLoad<float, 16>(s0 + (64_i32));
              v194_acc += ((static_cast<float>(v196_data[0])) * v36_data);
              v194_acc += ((static_cast<float>(v196_data[1])) * v37_data);
              v194_acc += ((static_cast<float>(v196_data[2])) * v38_data);
              v194_acc += ((static_cast<float>(v196_data[3])) * v39_data);
              v194_acc += ((static_cast<float>(v196_data[4])) * v40_data);
              v194_acc += ((static_cast<float>(v196_data[5])) * v41_data);
              v194_acc += ((static_cast<float>(v196_data[6])) * v42_data);
              v194_acc += ((static_cast<float>(v196_data[7])) * v43_data);
              v194_acc += ((static_cast<float>(v196_data[8])) * v44_data);
              v194_acc += ((static_cast<float>(v196_data[9])) * v45_data);
              v194_acc += ((static_cast<float>(v196_data[10])) * v46_data);
              v194_acc += ((static_cast<float>(v196_data[11])) * v47_data);
              v194_acc += ((static_cast<float>(v196_data[12])) * v48_data);
              v194_acc += ((static_cast<float>(v196_data[13])) * v49_data);
              v194_acc += ((static_cast<float>(v196_data[14])) * v50_data);
              v194_acc += ((static_cast<float>(v196_data[15])) * v51_data);
              ir1.template select<16, 1>(64) = v194_acc;
              tensorforge::intel_esimd::simd<float, 16> v229_acc{};
              tensorforge::intel_esimd::simd<float, 16> v231_data = tensorforge::slmLoad<float, 16>(s0 + (80_i32));
              v229_acc += ((static_cast<float>(v231_data[0])) * v36_data);
              v229_acc += ((static_cast<float>(v231_data[1])) * v37_data);
              v229_acc += ((static_cast<float>(v231_data[2])) * v38_data);
              v229_acc += ((static_cast<float>(v231_data[3])) * v39_data);
              v229_acc += ((static_cast<float>(v231_data[4])) * v40_data);
              v229_acc += ((static_cast<float>(v231_data[5])) * v41_data);
              v229_acc += ((static_cast<float>(v231_data[6])) * v42_data);
              v229_acc += ((static_cast<float>(v231_data[7])) * v43_data);
              v229_acc += ((static_cast<float>(v231_data[8])) * v44_data);
              v229_acc += ((static_cast<float>(v231_data[9])) * v45_data);
              v229_acc += ((static_cast<float>(v231_data[10])) * v46_data);
              v229_acc += ((static_cast<float>(v231_data[11])) * v47_data);
              v229_acc += ((static_cast<float>(v231_data[12])) * v48_data);
              v229_acc += ((static_cast<float>(v231_data[13])) * v49_data);
              v229_acc += ((static_cast<float>(v231_data[14])) * v50_data);
              v229_acc += ((static_cast<float>(v231_data[15])) * v51_data);
              ir1.template select<16, 1>(80) = v229_acc;
              tensorforge::intel_esimd::simd<float, 16> v264_acc{};
              tensorforge::intel_esimd::simd<float, 16> v266_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              v264_acc += ((static_cast<float>(v266_data[0])) * v36_data);
              v264_acc += ((static_cast<float>(v266_data[1])) * v37_data);
              v264_acc += ((static_cast<float>(v266_data[2])) * v38_data);
              v264_acc += ((static_cast<float>(v266_data[3])) * v39_data);
              v264_acc += ((static_cast<float>(v266_data[4])) * v40_data);
              v264_acc += ((static_cast<float>(v266_data[5])) * v41_data);
              v264_acc += ((static_cast<float>(v266_data[6])) * v42_data);
              v264_acc += ((static_cast<float>(v266_data[7])) * v43_data);
              v264_acc += ((static_cast<float>(v266_data[8])) * v44_data);
              v264_acc += ((static_cast<float>(v266_data[9])) * v45_data);
              v264_acc += ((static_cast<float>(v266_data[10])) * v46_data);
              v264_acc += ((static_cast<float>(v266_data[11])) * v47_data);
              v264_acc += ((static_cast<float>(v266_data[12])) * v48_data);
              v264_acc += ((static_cast<float>(v266_data[13])) * v49_data);
              v264_acc += ((static_cast<float>(v266_data[14])) * v50_data);
              v264_acc += ((static_cast<float>(v266_data[15])) * v51_data);
              ir1.template select<16, 1>(96) = v264_acc;
              tensorforge::intel_esimd::simd<float, 16> v299_acc{};
              tensorforge::intel_esimd::simd<float, 16> v301_data = tensorforge::slmLoad<float, 16>(s0 + (112_i32));
              v299_acc += ((static_cast<float>(v301_data[0])) * v36_data);
              v299_acc += ((static_cast<float>(v301_data[1])) * v37_data);
              v299_acc += ((static_cast<float>(v301_data[2])) * v38_data);
              v299_acc += ((static_cast<float>(v301_data[3])) * v39_data);
              v299_acc += ((static_cast<float>(v301_data[4])) * v40_data);
              v299_acc += ((static_cast<float>(v301_data[5])) * v41_data);
              v299_acc += ((static_cast<float>(v301_data[6])) * v42_data);
              v299_acc += ((static_cast<float>(v301_data[7])) * v43_data);
              v299_acc += ((static_cast<float>(v301_data[8])) * v44_data);
              v299_acc += ((static_cast<float>(v301_data[9])) * v45_data);
              v299_acc += ((static_cast<float>(v301_data[10])) * v46_data);
              v299_acc += ((static_cast<float>(v301_data[11])) * v47_data);
              v299_acc += ((static_cast<float>(v301_data[12])) * v48_data);
              v299_acc += ((static_cast<float>(v301_data[13])) * v49_data);
              v299_acc += ((static_cast<float>(v301_data[14])) * v50_data);
              v299_acc += ((static_cast<float>(v301_data[15])) * v51_data);
              ir1.template select<16, 1>(112) = v299_acc;
              // r1 = ir1
              #pragma unroll
              for (int32_t v334_n1 = 0; v334_n1 < 8; ++v334_n1) {
                int32_t v335_a = v334_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v337_data(ir1.template select<12, 1>(v335_a));
                r1.template select<12, 1>(v335_a) = v337_data;
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v338_i1 = 0; v338_i1 < 8; ++v338_i1) {
                tensorforge::intel_esimd::simd<float, 12> v341_data(r1.template select<12, 1>((v338_i1 * 16)));
                v341_data.copy_to(glb_m0 + ((v338_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}

