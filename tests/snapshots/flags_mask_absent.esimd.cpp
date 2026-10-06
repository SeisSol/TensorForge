// === base name ===
kernel_7a9ba9a4c016db58

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_7a9ba9a4c016db58 = {{1, 16, 1}, 16, 16, 1, 16, 17408, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_7a9ba9a4c016db58(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_7a9ba9a4c016db58(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_7a9ba9a4c016db58(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 4352 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_7a9ba9a4c016db58(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_7a9ba9a4c016db58(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_7a9ba9a4c016db58(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_7a9ba9a4c016db58(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<4352 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 17408 B shared, occupancy grid
        // operands:
        //   m0 16×16(16×16) {0..16}×{0..16} strided
        //   m1 16×16(16×16) {0..16}×{0..16} strided
        //   m2 16×16(16×16) {0..16}×{0..16} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":4352}],"shared_bytes":17408,"shared_elements":4352,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (272 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (256);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v11_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v11_batchId0 < numElements0; v11_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v12_ahead1 = v11_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
            float *const __restrict__ glb_m0 = &m0[v11_batchId0 * 256 + 0 + m0_extraOffset];
            const float *const __restrict__ glb_m1 = &m1[v11_batchId0 * 256 + 0 + m1_extraOffset];
            const float *const __restrict__ glb_m2 = &m2[v11_batchId0 * 256 + 0 + m2_extraOffset];
            tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
            // r0 = load{g>r}(glb_m1);
            #pragma unroll
            for (int32_t v22_i0 = 0; v22_i0 < 1; ++v22_i0) {
              int32_t v24_lead = v22_i0 * 16;
              #pragma unroll
              for (int32_t v23_i1 = 0; v23_i1 < 16; ++v23_i1) {
                int32_t v27_a = v24_lead + (v23_i1 * 16);
                tensorforge::intel_esimd::simd<float, 16> v28_data;
                v28_data.copy_from(glb_m1 + (v27_a));
                r0.template select<16, 1>(v27_a) = v28_data;
              }
            }
            // s0 = load{g>s}(glb_m2[0, 1])
            tensorforge::intel_esimd::simd<float, 64> v30_ld;
            v30_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
            tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v30_ld);
            tensorforge::intel_esimd::simd<float, 64> v31_ld;
            v31_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
            tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v31_ld);
            tensorforge::intel_esimd::simd<float, 64> v32_ld;
            v32_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 128));
            tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 128), v32_ld);
            tensorforge::intel_esimd::simd<float, 64> v33_ld;
            v33_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 192));
            tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 192), v33_ld);
            // wait(r0 = load{g>r}(glb_m1););
            // wait(s0 = load{g>s}(glb_m2[0, 1]));
            tensorforge::intel_esimd::simd<float, 256> r1(0.0f);
            // ir1 = +(r0 * s0)
            // [(0, 16), (0, 16)] [(0, 16)]
            tensorforge::intel_esimd::simd<float, 256> ir1(0.0f);
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
            tensorforge::intel_esimd::simd<float, 16> v334_acc{};
            tensorforge::intel_esimd::simd<float, 16> v336_data = tensorforge::slmLoad<float, 16>(s0 + (128_i32));
            v334_acc += ((static_cast<float>(v336_data[0])) * v36_data);
            v334_acc += ((static_cast<float>(v336_data[1])) * v37_data);
            v334_acc += ((static_cast<float>(v336_data[2])) * v38_data);
            v334_acc += ((static_cast<float>(v336_data[3])) * v39_data);
            v334_acc += ((static_cast<float>(v336_data[4])) * v40_data);
            v334_acc += ((static_cast<float>(v336_data[5])) * v41_data);
            v334_acc += ((static_cast<float>(v336_data[6])) * v42_data);
            v334_acc += ((static_cast<float>(v336_data[7])) * v43_data);
            v334_acc += ((static_cast<float>(v336_data[8])) * v44_data);
            v334_acc += ((static_cast<float>(v336_data[9])) * v45_data);
            v334_acc += ((static_cast<float>(v336_data[10])) * v46_data);
            v334_acc += ((static_cast<float>(v336_data[11])) * v47_data);
            v334_acc += ((static_cast<float>(v336_data[12])) * v48_data);
            v334_acc += ((static_cast<float>(v336_data[13])) * v49_data);
            v334_acc += ((static_cast<float>(v336_data[14])) * v50_data);
            v334_acc += ((static_cast<float>(v336_data[15])) * v51_data);
            ir1.template select<16, 1>(128) = v334_acc;
            tensorforge::intel_esimd::simd<float, 16> v369_acc{};
            tensorforge::intel_esimd::simd<float, 16> v371_data = tensorforge::slmLoad<float, 16>(s0 + (144_i32));
            v369_acc += ((static_cast<float>(v371_data[0])) * v36_data);
            v369_acc += ((static_cast<float>(v371_data[1])) * v37_data);
            v369_acc += ((static_cast<float>(v371_data[2])) * v38_data);
            v369_acc += ((static_cast<float>(v371_data[3])) * v39_data);
            v369_acc += ((static_cast<float>(v371_data[4])) * v40_data);
            v369_acc += ((static_cast<float>(v371_data[5])) * v41_data);
            v369_acc += ((static_cast<float>(v371_data[6])) * v42_data);
            v369_acc += ((static_cast<float>(v371_data[7])) * v43_data);
            v369_acc += ((static_cast<float>(v371_data[8])) * v44_data);
            v369_acc += ((static_cast<float>(v371_data[9])) * v45_data);
            v369_acc += ((static_cast<float>(v371_data[10])) * v46_data);
            v369_acc += ((static_cast<float>(v371_data[11])) * v47_data);
            v369_acc += ((static_cast<float>(v371_data[12])) * v48_data);
            v369_acc += ((static_cast<float>(v371_data[13])) * v49_data);
            v369_acc += ((static_cast<float>(v371_data[14])) * v50_data);
            v369_acc += ((static_cast<float>(v371_data[15])) * v51_data);
            ir1.template select<16, 1>(144) = v369_acc;
            tensorforge::intel_esimd::simd<float, 16> v404_acc{};
            tensorforge::intel_esimd::simd<float, 16> v406_data = tensorforge::slmLoad<float, 16>(s0 + (160_i32));
            v404_acc += ((static_cast<float>(v406_data[0])) * v36_data);
            v404_acc += ((static_cast<float>(v406_data[1])) * v37_data);
            v404_acc += ((static_cast<float>(v406_data[2])) * v38_data);
            v404_acc += ((static_cast<float>(v406_data[3])) * v39_data);
            v404_acc += ((static_cast<float>(v406_data[4])) * v40_data);
            v404_acc += ((static_cast<float>(v406_data[5])) * v41_data);
            v404_acc += ((static_cast<float>(v406_data[6])) * v42_data);
            v404_acc += ((static_cast<float>(v406_data[7])) * v43_data);
            v404_acc += ((static_cast<float>(v406_data[8])) * v44_data);
            v404_acc += ((static_cast<float>(v406_data[9])) * v45_data);
            v404_acc += ((static_cast<float>(v406_data[10])) * v46_data);
            v404_acc += ((static_cast<float>(v406_data[11])) * v47_data);
            v404_acc += ((static_cast<float>(v406_data[12])) * v48_data);
            v404_acc += ((static_cast<float>(v406_data[13])) * v49_data);
            v404_acc += ((static_cast<float>(v406_data[14])) * v50_data);
            v404_acc += ((static_cast<float>(v406_data[15])) * v51_data);
            ir1.template select<16, 1>(160) = v404_acc;
            tensorforge::intel_esimd::simd<float, 16> v439_acc{};
            tensorforge::intel_esimd::simd<float, 16> v441_data = tensorforge::slmLoad<float, 16>(s0 + (176_i32));
            v439_acc += ((static_cast<float>(v441_data[0])) * v36_data);
            v439_acc += ((static_cast<float>(v441_data[1])) * v37_data);
            v439_acc += ((static_cast<float>(v441_data[2])) * v38_data);
            v439_acc += ((static_cast<float>(v441_data[3])) * v39_data);
            v439_acc += ((static_cast<float>(v441_data[4])) * v40_data);
            v439_acc += ((static_cast<float>(v441_data[5])) * v41_data);
            v439_acc += ((static_cast<float>(v441_data[6])) * v42_data);
            v439_acc += ((static_cast<float>(v441_data[7])) * v43_data);
            v439_acc += ((static_cast<float>(v441_data[8])) * v44_data);
            v439_acc += ((static_cast<float>(v441_data[9])) * v45_data);
            v439_acc += ((static_cast<float>(v441_data[10])) * v46_data);
            v439_acc += ((static_cast<float>(v441_data[11])) * v47_data);
            v439_acc += ((static_cast<float>(v441_data[12])) * v48_data);
            v439_acc += ((static_cast<float>(v441_data[13])) * v49_data);
            v439_acc += ((static_cast<float>(v441_data[14])) * v50_data);
            v439_acc += ((static_cast<float>(v441_data[15])) * v51_data);
            ir1.template select<16, 1>(176) = v439_acc;
            tensorforge::intel_esimd::simd<float, 16> v474_acc{};
            tensorforge::intel_esimd::simd<float, 16> v476_data = tensorforge::slmLoad<float, 16>(s0 + (192_i32));
            v474_acc += ((static_cast<float>(v476_data[0])) * v36_data);
            v474_acc += ((static_cast<float>(v476_data[1])) * v37_data);
            v474_acc += ((static_cast<float>(v476_data[2])) * v38_data);
            v474_acc += ((static_cast<float>(v476_data[3])) * v39_data);
            v474_acc += ((static_cast<float>(v476_data[4])) * v40_data);
            v474_acc += ((static_cast<float>(v476_data[5])) * v41_data);
            v474_acc += ((static_cast<float>(v476_data[6])) * v42_data);
            v474_acc += ((static_cast<float>(v476_data[7])) * v43_data);
            v474_acc += ((static_cast<float>(v476_data[8])) * v44_data);
            v474_acc += ((static_cast<float>(v476_data[9])) * v45_data);
            v474_acc += ((static_cast<float>(v476_data[10])) * v46_data);
            v474_acc += ((static_cast<float>(v476_data[11])) * v47_data);
            v474_acc += ((static_cast<float>(v476_data[12])) * v48_data);
            v474_acc += ((static_cast<float>(v476_data[13])) * v49_data);
            v474_acc += ((static_cast<float>(v476_data[14])) * v50_data);
            v474_acc += ((static_cast<float>(v476_data[15])) * v51_data);
            ir1.template select<16, 1>(192) = v474_acc;
            tensorforge::intel_esimd::simd<float, 16> v509_acc{};
            tensorforge::intel_esimd::simd<float, 16> v511_data = tensorforge::slmLoad<float, 16>(s0 + (208_i32));
            v509_acc += ((static_cast<float>(v511_data[0])) * v36_data);
            v509_acc += ((static_cast<float>(v511_data[1])) * v37_data);
            v509_acc += ((static_cast<float>(v511_data[2])) * v38_data);
            v509_acc += ((static_cast<float>(v511_data[3])) * v39_data);
            v509_acc += ((static_cast<float>(v511_data[4])) * v40_data);
            v509_acc += ((static_cast<float>(v511_data[5])) * v41_data);
            v509_acc += ((static_cast<float>(v511_data[6])) * v42_data);
            v509_acc += ((static_cast<float>(v511_data[7])) * v43_data);
            v509_acc += ((static_cast<float>(v511_data[8])) * v44_data);
            v509_acc += ((static_cast<float>(v511_data[9])) * v45_data);
            v509_acc += ((static_cast<float>(v511_data[10])) * v46_data);
            v509_acc += ((static_cast<float>(v511_data[11])) * v47_data);
            v509_acc += ((static_cast<float>(v511_data[12])) * v48_data);
            v509_acc += ((static_cast<float>(v511_data[13])) * v49_data);
            v509_acc += ((static_cast<float>(v511_data[14])) * v50_data);
            v509_acc += ((static_cast<float>(v511_data[15])) * v51_data);
            ir1.template select<16, 1>(208) = v509_acc;
            tensorforge::intel_esimd::simd<float, 16> v544_acc{};
            tensorforge::intel_esimd::simd<float, 16> v546_data = tensorforge::slmLoad<float, 16>(s0 + (224_i32));
            v544_acc += ((static_cast<float>(v546_data[0])) * v36_data);
            v544_acc += ((static_cast<float>(v546_data[1])) * v37_data);
            v544_acc += ((static_cast<float>(v546_data[2])) * v38_data);
            v544_acc += ((static_cast<float>(v546_data[3])) * v39_data);
            v544_acc += ((static_cast<float>(v546_data[4])) * v40_data);
            v544_acc += ((static_cast<float>(v546_data[5])) * v41_data);
            v544_acc += ((static_cast<float>(v546_data[6])) * v42_data);
            v544_acc += ((static_cast<float>(v546_data[7])) * v43_data);
            v544_acc += ((static_cast<float>(v546_data[8])) * v44_data);
            v544_acc += ((static_cast<float>(v546_data[9])) * v45_data);
            v544_acc += ((static_cast<float>(v546_data[10])) * v46_data);
            v544_acc += ((static_cast<float>(v546_data[11])) * v47_data);
            v544_acc += ((static_cast<float>(v546_data[12])) * v48_data);
            v544_acc += ((static_cast<float>(v546_data[13])) * v49_data);
            v544_acc += ((static_cast<float>(v546_data[14])) * v50_data);
            v544_acc += ((static_cast<float>(v546_data[15])) * v51_data);
            ir1.template select<16, 1>(224) = v544_acc;
            tensorforge::intel_esimd::simd<float, 16> v579_acc{};
            tensorforge::intel_esimd::simd<float, 16> v581_data = tensorforge::slmLoad<float, 16>(s0 + (240_i32));
            v579_acc += ((static_cast<float>(v581_data[0])) * v36_data);
            v579_acc += ((static_cast<float>(v581_data[1])) * v37_data);
            v579_acc += ((static_cast<float>(v581_data[2])) * v38_data);
            v579_acc += ((static_cast<float>(v581_data[3])) * v39_data);
            v579_acc += ((static_cast<float>(v581_data[4])) * v40_data);
            v579_acc += ((static_cast<float>(v581_data[5])) * v41_data);
            v579_acc += ((static_cast<float>(v581_data[6])) * v42_data);
            v579_acc += ((static_cast<float>(v581_data[7])) * v43_data);
            v579_acc += ((static_cast<float>(v581_data[8])) * v44_data);
            v579_acc += ((static_cast<float>(v581_data[9])) * v45_data);
            v579_acc += ((static_cast<float>(v581_data[10])) * v46_data);
            v579_acc += ((static_cast<float>(v581_data[11])) * v47_data);
            v579_acc += ((static_cast<float>(v581_data[12])) * v48_data);
            v579_acc += ((static_cast<float>(v581_data[13])) * v49_data);
            v579_acc += ((static_cast<float>(v581_data[14])) * v50_data);
            v579_acc += ((static_cast<float>(v581_data[15])) * v51_data);
            ir1.template select<16, 1>(240) = v579_acc;
            // r1 = ir1
            #pragma unroll
            for (int32_t v614_n0 = 0; v614_n0 < 1; ++v614_n0) {
              int32_t v616_a = v614_n0 * 16;
              #pragma unroll
              for (int32_t v615_n1 = 0; v615_n1 < 16; ++v615_n1) {
                int32_t v618_a = v616_a + (v615_n1 * 16);
                tensorforge::intel_esimd::simd<float, 16> v619_data(ir1.template select<16, 1>(v618_a));
                r1.template select<16, 1>(v618_a) = v619_data;
              }
            }
            // glb_m0 = store{r>g}(r1);
            #pragma unroll
            for (int32_t v620_i0 = 0; v620_i0 < 1; ++v620_i0) {
              int32_t v622_a = v620_i0 * 16;
              #pragma unroll
              for (int32_t v621_i1 = 0; v621_i1 < 16; ++v621_i1) {
                int32_t v624_a = v622_a + (v621_i1 * 16);
                tensorforge::intel_esimd::simd<float, 16> v625_data(r1.template select<16, 1>(v624_a));
                v625_data.copy_to(glb_m0 + (v624_a));
              }
            }
          }
        }
      }
    });
  });
}

