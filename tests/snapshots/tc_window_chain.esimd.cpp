// === base name ===
kernel_580469a140789642

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_580469a140789642 = {{1, 16, 1}, 16, 16, 1, 16, 11264, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_580469a140789642(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_580469a140789642(const float * m0, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_580469a140789642(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 2816 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_580469a140789642(const float * m0, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_580469a140789642(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_580469a140789642(stream, grid, block, m0, m1, m1_extraOffset, m2, m2_extraOffset, m3, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_580469a140789642(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2816 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 11264 B shared, occupancy grid
        // operands:
        //   m0 16×20(16×17) {0..16}×{1..18} none
        //   m1 20×9(17×9) {1..18}×{0..9} strided
        //   m2 16×9(16×9) {0..16}×{0..9} strided
        //   m3 16×20(16×15) {0..16}×{1..16} none
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   m2[i,j] = m3[i,k] × t0[k,j]@{1..16}×{0..9}
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2816}],"shared_bytes":11264,"shared_elements":2816,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"none","alias":"A1","bbox":[[0,1],[16,18]],"name":"m0","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[1,0],[18,9]],"name":"m1","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[16,9]],"name":"m2","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"none","alias":"A2","bbox":[[0,1],[16,16]],"name":"m3","ordered":false,"parts":1,"shape":[16,20],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[16,18]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[18,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[16,16]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,20]},{"addressing":"pointer_based","bbox":[[1,0],[16,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[16,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (176 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (160);
          const float *const __restrict__ glb_m0 = &m0[0];
          const float *const __restrict__ glb_m3 = &m3[0];
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const float *const __restrict__ pf_glb_m1 = &m1[v11_batchId1 * 153 + 0 + m1_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 153 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 144 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v19_ld;
              v19_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v19_ld);
              tensorforge::intel_esimd::simd<float, 64> v20_ld;
              v20_ld.copy_from(glb_m1 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v20_ld);
              tensorforge::intel_esimd::simd<float, 16> v21_ld;
              v21_ld.copy_from(glb_m1 + (0 + 0 + 1 * 0 + 128));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 128), v21_ld);
              tensorforge::intel_esimd::simd<float, 9> v22_ld;
              v22_ld.copy_from(glb_m1 + (0 + 0 + 1 * 0 + 144));
              tensorforge::slmStore<float, 9>(s0 + (0 + 0 + 1 * 0 + 144), v22_ld);
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 144> r0(0.0f);
              // r0 = +(glb_m0 * s0) + None
              // [(0, 16), (0, 9)] [(1, 18)]
              tensorforge::intel_esimd::simd<float, 64> glb_m0_run0;
              glb_m0_run0.copy_from(glb_m0 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v27_data(glb_m0_run0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v29_data(glb_m0_run0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v31_data(glb_m0_run0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v33_data(glb_m0_run0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m0_run1;
              glb_m0_run1.copy_from(glb_m0 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v35_data(glb_m0_run1.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v37_data(glb_m0_run1.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v39_data(glb_m0_run1.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v41_data(glb_m0_run1.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m0_run2;
              glb_m0_run2.copy_from(glb_m0 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v43_data(glb_m0_run2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v45_data(glb_m0_run2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v47_data(glb_m0_run2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v49_data(glb_m0_run2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m0_run3;
              glb_m0_run3.copy_from(glb_m0 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v51_data(glb_m0_run3.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v53_data(glb_m0_run3.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v55_data(glb_m0_run3.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v57_data(glb_m0_run3.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v59_data;
              v59_data.copy_from(glb_m0 + (256_i32));
              tensorforge::intel_esimd::simd<float, 16> v60_acc{};
              tensorforge::intel_esimd::simd<float, 16> v63_data(0.0f);
              v63_data.template select<15, 1>(1) = tensorforge::slmLoad<float, 15>((s0 + (-1_i32)) + 1);
              v60_acc += ((static_cast<float>(v63_data[1])) * v27_data);
              v60_acc += ((static_cast<float>(v63_data[2])) * v29_data);
              v60_acc += ((static_cast<float>(v63_data[3])) * v31_data);
              v60_acc += ((static_cast<float>(v63_data[4])) * v33_data);
              v60_acc += ((static_cast<float>(v63_data[5])) * v35_data);
              v60_acc += ((static_cast<float>(v63_data[6])) * v37_data);
              v60_acc += ((static_cast<float>(v63_data[7])) * v39_data);
              v60_acc += ((static_cast<float>(v63_data[8])) * v41_data);
              v60_acc += ((static_cast<float>(v63_data[9])) * v43_data);
              v60_acc += ((static_cast<float>(v63_data[10])) * v45_data);
              v60_acc += ((static_cast<float>(v63_data[11])) * v47_data);
              v60_acc += ((static_cast<float>(v63_data[12])) * v49_data);
              v60_acc += ((static_cast<float>(v63_data[13])) * v51_data);
              v60_acc += ((static_cast<float>(v63_data[14])) * v53_data);
              v60_acc += ((static_cast<float>(v63_data[15])) * v55_data);
              tensorforge::intel_esimd::simd<float, 16> v99_data = tensorforge::slmLoad<float, 16>(s0 + (15_i32));
              v60_acc += ((static_cast<float>(v99_data[0])) * v57_data);
              v60_acc += ((static_cast<float>(v99_data[1])) * v59_data);
              r0.template select<16, 1>(0) = v60_acc;
              tensorforge::intel_esimd::simd<float, 16> v104_acc{};
              tensorforge::intel_esimd::simd<float, 16> v106_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v104_acc += ((static_cast<float>(v106_data[1])) * v27_data);
              v104_acc += ((static_cast<float>(v106_data[2])) * v29_data);
              v104_acc += ((static_cast<float>(v106_data[3])) * v31_data);
              v104_acc += ((static_cast<float>(v106_data[4])) * v33_data);
              v104_acc += ((static_cast<float>(v106_data[5])) * v35_data);
              v104_acc += ((static_cast<float>(v106_data[6])) * v37_data);
              v104_acc += ((static_cast<float>(v106_data[7])) * v39_data);
              v104_acc += ((static_cast<float>(v106_data[8])) * v41_data);
              v104_acc += ((static_cast<float>(v106_data[9])) * v43_data);
              v104_acc += ((static_cast<float>(v106_data[10])) * v45_data);
              v104_acc += ((static_cast<float>(v106_data[11])) * v47_data);
              v104_acc += ((static_cast<float>(v106_data[12])) * v49_data);
              v104_acc += ((static_cast<float>(v106_data[13])) * v51_data);
              v104_acc += ((static_cast<float>(v106_data[14])) * v53_data);
              v104_acc += ((static_cast<float>(v106_data[15])) * v55_data);
              tensorforge::intel_esimd::simd<float, 16> v139_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v104_acc += ((static_cast<float>(v139_data[0])) * v57_data);
              v104_acc += ((static_cast<float>(v139_data[1])) * v59_data);
              r0.template select<16, 1>(16) = v104_acc;
              tensorforge::intel_esimd::simd<float, 16> v144_acc{};
              tensorforge::intel_esimd::simd<float, 16> v146_data = tensorforge::slmLoad<float, 16>(s0 + (33_i32));
              v144_acc += ((static_cast<float>(v146_data[1])) * v27_data);
              v144_acc += ((static_cast<float>(v146_data[2])) * v29_data);
              v144_acc += ((static_cast<float>(v146_data[3])) * v31_data);
              v144_acc += ((static_cast<float>(v146_data[4])) * v33_data);
              v144_acc += ((static_cast<float>(v146_data[5])) * v35_data);
              v144_acc += ((static_cast<float>(v146_data[6])) * v37_data);
              v144_acc += ((static_cast<float>(v146_data[7])) * v39_data);
              v144_acc += ((static_cast<float>(v146_data[8])) * v41_data);
              v144_acc += ((static_cast<float>(v146_data[9])) * v43_data);
              v144_acc += ((static_cast<float>(v146_data[10])) * v45_data);
              v144_acc += ((static_cast<float>(v146_data[11])) * v47_data);
              v144_acc += ((static_cast<float>(v146_data[12])) * v49_data);
              v144_acc += ((static_cast<float>(v146_data[13])) * v51_data);
              v144_acc += ((static_cast<float>(v146_data[14])) * v53_data);
              v144_acc += ((static_cast<float>(v146_data[15])) * v55_data);
              tensorforge::intel_esimd::simd<float, 16> v179_data = tensorforge::slmLoad<float, 16>(s0 + (49_i32));
              v144_acc += ((static_cast<float>(v179_data[0])) * v57_data);
              v144_acc += ((static_cast<float>(v179_data[1])) * v59_data);
              r0.template select<16, 1>(32) = v144_acc;
              tensorforge::intel_esimd::simd<float, 16> v184_acc{};
              tensorforge::intel_esimd::simd<float, 16> v186_data = tensorforge::slmLoad<float, 16>(s0 + (50_i32));
              v184_acc += ((static_cast<float>(v186_data[1])) * v27_data);
              v184_acc += ((static_cast<float>(v186_data[2])) * v29_data);
              v184_acc += ((static_cast<float>(v186_data[3])) * v31_data);
              v184_acc += ((static_cast<float>(v186_data[4])) * v33_data);
              v184_acc += ((static_cast<float>(v186_data[5])) * v35_data);
              v184_acc += ((static_cast<float>(v186_data[6])) * v37_data);
              v184_acc += ((static_cast<float>(v186_data[7])) * v39_data);
              v184_acc += ((static_cast<float>(v186_data[8])) * v41_data);
              v184_acc += ((static_cast<float>(v186_data[9])) * v43_data);
              v184_acc += ((static_cast<float>(v186_data[10])) * v45_data);
              v184_acc += ((static_cast<float>(v186_data[11])) * v47_data);
              v184_acc += ((static_cast<float>(v186_data[12])) * v49_data);
              v184_acc += ((static_cast<float>(v186_data[13])) * v51_data);
              v184_acc += ((static_cast<float>(v186_data[14])) * v53_data);
              v184_acc += ((static_cast<float>(v186_data[15])) * v55_data);
              tensorforge::intel_esimd::simd<float, 16> v219_data = tensorforge::slmLoad<float, 16>(s0 + (66_i32));
              v184_acc += ((static_cast<float>(v219_data[0])) * v57_data);
              v184_acc += ((static_cast<float>(v219_data[1])) * v59_data);
              r0.template select<16, 1>(48) = v184_acc;
              tensorforge::intel_esimd::simd<float, 16> v224_acc{};
              tensorforge::intel_esimd::simd<float, 16> v226_data = tensorforge::slmLoad<float, 16>(s0 + (67_i32));
              v224_acc += ((static_cast<float>(v226_data[1])) * v27_data);
              v224_acc += ((static_cast<float>(v226_data[2])) * v29_data);
              v224_acc += ((static_cast<float>(v226_data[3])) * v31_data);
              v224_acc += ((static_cast<float>(v226_data[4])) * v33_data);
              v224_acc += ((static_cast<float>(v226_data[5])) * v35_data);
              v224_acc += ((static_cast<float>(v226_data[6])) * v37_data);
              v224_acc += ((static_cast<float>(v226_data[7])) * v39_data);
              v224_acc += ((static_cast<float>(v226_data[8])) * v41_data);
              v224_acc += ((static_cast<float>(v226_data[9])) * v43_data);
              v224_acc += ((static_cast<float>(v226_data[10])) * v45_data);
              v224_acc += ((static_cast<float>(v226_data[11])) * v47_data);
              v224_acc += ((static_cast<float>(v226_data[12])) * v49_data);
              v224_acc += ((static_cast<float>(v226_data[13])) * v51_data);
              v224_acc += ((static_cast<float>(v226_data[14])) * v53_data);
              v224_acc += ((static_cast<float>(v226_data[15])) * v55_data);
              tensorforge::intel_esimd::simd<float, 16> v259_data = tensorforge::slmLoad<float, 16>(s0 + (83_i32));
              v224_acc += ((static_cast<float>(v259_data[0])) * v57_data);
              v224_acc += ((static_cast<float>(v259_data[1])) * v59_data);
              r0.template select<16, 1>(64) = v224_acc;
              tensorforge::intel_esimd::simd<float, 16> v264_acc{};
              tensorforge::intel_esimd::simd<float, 16> v266_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v264_acc += ((static_cast<float>(v266_data[1])) * v27_data);
              v264_acc += ((static_cast<float>(v266_data[2])) * v29_data);
              v264_acc += ((static_cast<float>(v266_data[3])) * v31_data);
              v264_acc += ((static_cast<float>(v266_data[4])) * v33_data);
              v264_acc += ((static_cast<float>(v266_data[5])) * v35_data);
              v264_acc += ((static_cast<float>(v266_data[6])) * v37_data);
              v264_acc += ((static_cast<float>(v266_data[7])) * v39_data);
              v264_acc += ((static_cast<float>(v266_data[8])) * v41_data);
              v264_acc += ((static_cast<float>(v266_data[9])) * v43_data);
              v264_acc += ((static_cast<float>(v266_data[10])) * v45_data);
              v264_acc += ((static_cast<float>(v266_data[11])) * v47_data);
              v264_acc += ((static_cast<float>(v266_data[12])) * v49_data);
              v264_acc += ((static_cast<float>(v266_data[13])) * v51_data);
              v264_acc += ((static_cast<float>(v266_data[14])) * v53_data);
              v264_acc += ((static_cast<float>(v266_data[15])) * v55_data);
              tensorforge::intel_esimd::simd<float, 16> v299_data = tensorforge::slmLoad<float, 16>(s0 + (100_i32));
              v264_acc += ((static_cast<float>(v299_data[0])) * v57_data);
              v264_acc += ((static_cast<float>(v299_data[1])) * v59_data);
              r0.template select<16, 1>(80) = v264_acc;
              tensorforge::intel_esimd::simd<float, 16> v304_acc{};
              tensorforge::intel_esimd::simd<float, 16> v306_data = tensorforge::slmLoad<float, 16>(s0 + (101_i32));
              v304_acc += ((static_cast<float>(v306_data[1])) * v27_data);
              v304_acc += ((static_cast<float>(v306_data[2])) * v29_data);
              v304_acc += ((static_cast<float>(v306_data[3])) * v31_data);
              v304_acc += ((static_cast<float>(v306_data[4])) * v33_data);
              v304_acc += ((static_cast<float>(v306_data[5])) * v35_data);
              v304_acc += ((static_cast<float>(v306_data[6])) * v37_data);
              v304_acc += ((static_cast<float>(v306_data[7])) * v39_data);
              v304_acc += ((static_cast<float>(v306_data[8])) * v41_data);
              v304_acc += ((static_cast<float>(v306_data[9])) * v43_data);
              v304_acc += ((static_cast<float>(v306_data[10])) * v45_data);
              v304_acc += ((static_cast<float>(v306_data[11])) * v47_data);
              v304_acc += ((static_cast<float>(v306_data[12])) * v49_data);
              v304_acc += ((static_cast<float>(v306_data[13])) * v51_data);
              v304_acc += ((static_cast<float>(v306_data[14])) * v53_data);
              v304_acc += ((static_cast<float>(v306_data[15])) * v55_data);
              tensorforge::intel_esimd::simd<float, 16> v339_data = tensorforge::slmLoad<float, 16>(s0 + (117_i32));
              v304_acc += ((static_cast<float>(v339_data[0])) * v57_data);
              v304_acc += ((static_cast<float>(v339_data[1])) * v59_data);
              r0.template select<16, 1>(96) = v304_acc;
              tensorforge::intel_esimd::simd<float, 16> v344_acc{};
              tensorforge::intel_esimd::simd<float, 16> v346_data = tensorforge::slmLoad<float, 16>(s0 + (118_i32));
              v344_acc += ((static_cast<float>(v346_data[1])) * v27_data);
              v344_acc += ((static_cast<float>(v346_data[2])) * v29_data);
              v344_acc += ((static_cast<float>(v346_data[3])) * v31_data);
              v344_acc += ((static_cast<float>(v346_data[4])) * v33_data);
              v344_acc += ((static_cast<float>(v346_data[5])) * v35_data);
              v344_acc += ((static_cast<float>(v346_data[6])) * v37_data);
              v344_acc += ((static_cast<float>(v346_data[7])) * v39_data);
              v344_acc += ((static_cast<float>(v346_data[8])) * v41_data);
              v344_acc += ((static_cast<float>(v346_data[9])) * v43_data);
              v344_acc += ((static_cast<float>(v346_data[10])) * v45_data);
              v344_acc += ((static_cast<float>(v346_data[11])) * v47_data);
              v344_acc += ((static_cast<float>(v346_data[12])) * v49_data);
              v344_acc += ((static_cast<float>(v346_data[13])) * v51_data);
              v344_acc += ((static_cast<float>(v346_data[14])) * v53_data);
              v344_acc += ((static_cast<float>(v346_data[15])) * v55_data);
              tensorforge::intel_esimd::simd<float, 16> v379_data = tensorforge::slmLoad<float, 16>(s0 + (134_i32));
              v344_acc += ((static_cast<float>(v379_data[0])) * v57_data);
              v344_acc += ((static_cast<float>(v379_data[1])) * v59_data);
              r0.template select<16, 1>(112) = v344_acc;
              tensorforge::intel_esimd::simd<float, 16> v384_acc{};
              tensorforge::intel_esimd::simd<float, 16> v386_data = tensorforge::slmLoad<float, 16>(s0 + (135_i32));
              v384_acc += ((static_cast<float>(v386_data[1])) * v27_data);
              v384_acc += ((static_cast<float>(v386_data[2])) * v29_data);
              v384_acc += ((static_cast<float>(v386_data[3])) * v31_data);
              v384_acc += ((static_cast<float>(v386_data[4])) * v33_data);
              v384_acc += ((static_cast<float>(v386_data[5])) * v35_data);
              v384_acc += ((static_cast<float>(v386_data[6])) * v37_data);
              v384_acc += ((static_cast<float>(v386_data[7])) * v39_data);
              v384_acc += ((static_cast<float>(v386_data[8])) * v41_data);
              v384_acc += ((static_cast<float>(v386_data[9])) * v43_data);
              v384_acc += ((static_cast<float>(v386_data[10])) * v45_data);
              v384_acc += ((static_cast<float>(v386_data[11])) * v47_data);
              v384_acc += ((static_cast<float>(v386_data[12])) * v49_data);
              v384_acc += ((static_cast<float>(v386_data[13])) * v51_data);
              v384_acc += ((static_cast<float>(v386_data[14])) * v53_data);
              v384_acc += ((static_cast<float>(v386_data[15])) * v55_data);
              tensorforge::intel_esimd::simd<float, 16> v419_data = tensorforge::slmLoad<float, 16>(s0 + (151_i32));
              v384_acc += ((static_cast<float>(v419_data[0])) * v57_data);
              v384_acc += ((static_cast<float>(v419_data[1])) * v59_data);
              r0.template select<16, 1>(128) = v384_acc;
              // s1 = store{r>s}(localShrMem0, r0);
              #pragma unroll
              for (int32_t v424_i0 = 0; v424_i0 < 1; ++v424_i0) {
                int32_t v426_a = v424_i0 * 16;
                #pragma unroll
                for (int32_t v425_i1 = 0; v425_i1 < 9; ++v425_i1) {
                  int32_t v428_a = v426_a + (v425_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v429_data(r0.template select<16, 1>(v428_a));
                  tensorforge::slmStore<float, 16>(s1 + (v428_a), v429_data);
                }
              }
              tensorforge::intel_esimd::simd<float, 144> r1(0.0f);
              // ir1 = +(glb_m3 * s1)
              // [(0, 16), (0, 9)] [(1, 16)]
              tensorforge::intel_esimd::simd<float, 144> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 64> glb_m3_run4;
              glb_m3_run4.copy_from(glb_m3 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v437_data(glb_m3_run4.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v439_data(glb_m3_run4.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v441_data(glb_m3_run4.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v443_data(glb_m3_run4.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m3_run5;
              glb_m3_run5.copy_from(glb_m3 + (64_i32));
              tensorforge::intel_esimd::simd<float, 16> v445_data(glb_m3_run5.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v447_data(glb_m3_run5.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v449_data(glb_m3_run5.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v451_data(glb_m3_run5.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 64> glb_m3_run6;
              glb_m3_run6.copy_from(glb_m3 + (128_i32));
              tensorforge::intel_esimd::simd<float, 16> v453_data(glb_m3_run6.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v455_data(glb_m3_run6.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v457_data(glb_m3_run6.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v459_data(glb_m3_run6.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 48> glb_m3_run7;
              glb_m3_run7.copy_from(glb_m3 + (192_i32));
              tensorforge::intel_esimd::simd<float, 16> v461_data(glb_m3_run7.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v463_data(glb_m3_run7.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v465_data(glb_m3_run7.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v466_acc{};
              tensorforge::intel_esimd::simd<float, 16> v467_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v466_acc += ((static_cast<float>(v467_data[1])) * v437_data);
              v466_acc += ((static_cast<float>(v467_data[2])) * v439_data);
              v466_acc += ((static_cast<float>(v467_data[3])) * v441_data);
              v466_acc += ((static_cast<float>(v467_data[4])) * v443_data);
              v466_acc += ((static_cast<float>(v467_data[5])) * v445_data);
              v466_acc += ((static_cast<float>(v467_data[6])) * v447_data);
              v466_acc += ((static_cast<float>(v467_data[7])) * v449_data);
              v466_acc += ((static_cast<float>(v467_data[8])) * v451_data);
              v466_acc += ((static_cast<float>(v467_data[9])) * v453_data);
              v466_acc += ((static_cast<float>(v467_data[10])) * v455_data);
              v466_acc += ((static_cast<float>(v467_data[11])) * v457_data);
              v466_acc += ((static_cast<float>(v467_data[12])) * v459_data);
              v466_acc += ((static_cast<float>(v467_data[13])) * v461_data);
              v466_acc += ((static_cast<float>(v467_data[14])) * v463_data);
              v466_acc += ((static_cast<float>(v467_data[15])) * v465_data);
              ir1.template select<16, 1>(0) = v466_acc;
              tensorforge::intel_esimd::simd<float, 16> v499_acc{};
              tensorforge::intel_esimd::simd<float, 16> v500_data = tensorforge::slmLoad<float, 16>(s1 + (16_i32));
              v499_acc += ((static_cast<float>(v500_data[1])) * v437_data);
              v499_acc += ((static_cast<float>(v500_data[2])) * v439_data);
              v499_acc += ((static_cast<float>(v500_data[3])) * v441_data);
              v499_acc += ((static_cast<float>(v500_data[4])) * v443_data);
              v499_acc += ((static_cast<float>(v500_data[5])) * v445_data);
              v499_acc += ((static_cast<float>(v500_data[6])) * v447_data);
              v499_acc += ((static_cast<float>(v500_data[7])) * v449_data);
              v499_acc += ((static_cast<float>(v500_data[8])) * v451_data);
              v499_acc += ((static_cast<float>(v500_data[9])) * v453_data);
              v499_acc += ((static_cast<float>(v500_data[10])) * v455_data);
              v499_acc += ((static_cast<float>(v500_data[11])) * v457_data);
              v499_acc += ((static_cast<float>(v500_data[12])) * v459_data);
              v499_acc += ((static_cast<float>(v500_data[13])) * v461_data);
              v499_acc += ((static_cast<float>(v500_data[14])) * v463_data);
              v499_acc += ((static_cast<float>(v500_data[15])) * v465_data);
              ir1.template select<16, 1>(16) = v499_acc;
              tensorforge::intel_esimd::simd<float, 16> v532_acc{};
              tensorforge::intel_esimd::simd<float, 16> v533_data = tensorforge::slmLoad<float, 16>(s1 + (32_i32));
              v532_acc += ((static_cast<float>(v533_data[1])) * v437_data);
              v532_acc += ((static_cast<float>(v533_data[2])) * v439_data);
              v532_acc += ((static_cast<float>(v533_data[3])) * v441_data);
              v532_acc += ((static_cast<float>(v533_data[4])) * v443_data);
              v532_acc += ((static_cast<float>(v533_data[5])) * v445_data);
              v532_acc += ((static_cast<float>(v533_data[6])) * v447_data);
              v532_acc += ((static_cast<float>(v533_data[7])) * v449_data);
              v532_acc += ((static_cast<float>(v533_data[8])) * v451_data);
              v532_acc += ((static_cast<float>(v533_data[9])) * v453_data);
              v532_acc += ((static_cast<float>(v533_data[10])) * v455_data);
              v532_acc += ((static_cast<float>(v533_data[11])) * v457_data);
              v532_acc += ((static_cast<float>(v533_data[12])) * v459_data);
              v532_acc += ((static_cast<float>(v533_data[13])) * v461_data);
              v532_acc += ((static_cast<float>(v533_data[14])) * v463_data);
              v532_acc += ((static_cast<float>(v533_data[15])) * v465_data);
              ir1.template select<16, 1>(32) = v532_acc;
              tensorforge::intel_esimd::simd<float, 16> v565_acc{};
              tensorforge::intel_esimd::simd<float, 16> v566_data = tensorforge::slmLoad<float, 16>(s1 + (48_i32));
              v565_acc += ((static_cast<float>(v566_data[1])) * v437_data);
              v565_acc += ((static_cast<float>(v566_data[2])) * v439_data);
              v565_acc += ((static_cast<float>(v566_data[3])) * v441_data);
              v565_acc += ((static_cast<float>(v566_data[4])) * v443_data);
              v565_acc += ((static_cast<float>(v566_data[5])) * v445_data);
              v565_acc += ((static_cast<float>(v566_data[6])) * v447_data);
              v565_acc += ((static_cast<float>(v566_data[7])) * v449_data);
              v565_acc += ((static_cast<float>(v566_data[8])) * v451_data);
              v565_acc += ((static_cast<float>(v566_data[9])) * v453_data);
              v565_acc += ((static_cast<float>(v566_data[10])) * v455_data);
              v565_acc += ((static_cast<float>(v566_data[11])) * v457_data);
              v565_acc += ((static_cast<float>(v566_data[12])) * v459_data);
              v565_acc += ((static_cast<float>(v566_data[13])) * v461_data);
              v565_acc += ((static_cast<float>(v566_data[14])) * v463_data);
              v565_acc += ((static_cast<float>(v566_data[15])) * v465_data);
              ir1.template select<16, 1>(48) = v565_acc;
              tensorforge::intel_esimd::simd<float, 16> v598_acc{};
              tensorforge::intel_esimd::simd<float, 16> v599_data = tensorforge::slmLoad<float, 16>(s1 + (64_i32));
              v598_acc += ((static_cast<float>(v599_data[1])) * v437_data);
              v598_acc += ((static_cast<float>(v599_data[2])) * v439_data);
              v598_acc += ((static_cast<float>(v599_data[3])) * v441_data);
              v598_acc += ((static_cast<float>(v599_data[4])) * v443_data);
              v598_acc += ((static_cast<float>(v599_data[5])) * v445_data);
              v598_acc += ((static_cast<float>(v599_data[6])) * v447_data);
              v598_acc += ((static_cast<float>(v599_data[7])) * v449_data);
              v598_acc += ((static_cast<float>(v599_data[8])) * v451_data);
              v598_acc += ((static_cast<float>(v599_data[9])) * v453_data);
              v598_acc += ((static_cast<float>(v599_data[10])) * v455_data);
              v598_acc += ((static_cast<float>(v599_data[11])) * v457_data);
              v598_acc += ((static_cast<float>(v599_data[12])) * v459_data);
              v598_acc += ((static_cast<float>(v599_data[13])) * v461_data);
              v598_acc += ((static_cast<float>(v599_data[14])) * v463_data);
              v598_acc += ((static_cast<float>(v599_data[15])) * v465_data);
              ir1.template select<16, 1>(64) = v598_acc;
              tensorforge::intel_esimd::simd<float, 16> v631_acc{};
              tensorforge::intel_esimd::simd<float, 16> v632_data = tensorforge::slmLoad<float, 16>(s1 + (80_i32));
              v631_acc += ((static_cast<float>(v632_data[1])) * v437_data);
              v631_acc += ((static_cast<float>(v632_data[2])) * v439_data);
              v631_acc += ((static_cast<float>(v632_data[3])) * v441_data);
              v631_acc += ((static_cast<float>(v632_data[4])) * v443_data);
              v631_acc += ((static_cast<float>(v632_data[5])) * v445_data);
              v631_acc += ((static_cast<float>(v632_data[6])) * v447_data);
              v631_acc += ((static_cast<float>(v632_data[7])) * v449_data);
              v631_acc += ((static_cast<float>(v632_data[8])) * v451_data);
              v631_acc += ((static_cast<float>(v632_data[9])) * v453_data);
              v631_acc += ((static_cast<float>(v632_data[10])) * v455_data);
              v631_acc += ((static_cast<float>(v632_data[11])) * v457_data);
              v631_acc += ((static_cast<float>(v632_data[12])) * v459_data);
              v631_acc += ((static_cast<float>(v632_data[13])) * v461_data);
              v631_acc += ((static_cast<float>(v632_data[14])) * v463_data);
              v631_acc += ((static_cast<float>(v632_data[15])) * v465_data);
              ir1.template select<16, 1>(80) = v631_acc;
              tensorforge::intel_esimd::simd<float, 16> v664_acc{};
              tensorforge::intel_esimd::simd<float, 16> v665_data = tensorforge::slmLoad<float, 16>(s1 + (96_i32));
              v664_acc += ((static_cast<float>(v665_data[1])) * v437_data);
              v664_acc += ((static_cast<float>(v665_data[2])) * v439_data);
              v664_acc += ((static_cast<float>(v665_data[3])) * v441_data);
              v664_acc += ((static_cast<float>(v665_data[4])) * v443_data);
              v664_acc += ((static_cast<float>(v665_data[5])) * v445_data);
              v664_acc += ((static_cast<float>(v665_data[6])) * v447_data);
              v664_acc += ((static_cast<float>(v665_data[7])) * v449_data);
              v664_acc += ((static_cast<float>(v665_data[8])) * v451_data);
              v664_acc += ((static_cast<float>(v665_data[9])) * v453_data);
              v664_acc += ((static_cast<float>(v665_data[10])) * v455_data);
              v664_acc += ((static_cast<float>(v665_data[11])) * v457_data);
              v664_acc += ((static_cast<float>(v665_data[12])) * v459_data);
              v664_acc += ((static_cast<float>(v665_data[13])) * v461_data);
              v664_acc += ((static_cast<float>(v665_data[14])) * v463_data);
              v664_acc += ((static_cast<float>(v665_data[15])) * v465_data);
              ir1.template select<16, 1>(96) = v664_acc;
              tensorforge::intel_esimd::simd<float, 16> v697_acc{};
              tensorforge::intel_esimd::simd<float, 16> v698_data = tensorforge::slmLoad<float, 16>(s1 + (112_i32));
              v697_acc += ((static_cast<float>(v698_data[1])) * v437_data);
              v697_acc += ((static_cast<float>(v698_data[2])) * v439_data);
              v697_acc += ((static_cast<float>(v698_data[3])) * v441_data);
              v697_acc += ((static_cast<float>(v698_data[4])) * v443_data);
              v697_acc += ((static_cast<float>(v698_data[5])) * v445_data);
              v697_acc += ((static_cast<float>(v698_data[6])) * v447_data);
              v697_acc += ((static_cast<float>(v698_data[7])) * v449_data);
              v697_acc += ((static_cast<float>(v698_data[8])) * v451_data);
              v697_acc += ((static_cast<float>(v698_data[9])) * v453_data);
              v697_acc += ((static_cast<float>(v698_data[10])) * v455_data);
              v697_acc += ((static_cast<float>(v698_data[11])) * v457_data);
              v697_acc += ((static_cast<float>(v698_data[12])) * v459_data);
              v697_acc += ((static_cast<float>(v698_data[13])) * v461_data);
              v697_acc += ((static_cast<float>(v698_data[14])) * v463_data);
              v697_acc += ((static_cast<float>(v698_data[15])) * v465_data);
              ir1.template select<16, 1>(112) = v697_acc;
              tensorforge::intel_esimd::simd<float, 16> v730_acc{};
              tensorforge::intel_esimd::simd<float, 16> v731_data = tensorforge::slmLoad<float, 16>(s1 + (128_i32));
              v730_acc += ((static_cast<float>(v731_data[1])) * v437_data);
              v730_acc += ((static_cast<float>(v731_data[2])) * v439_data);
              v730_acc += ((static_cast<float>(v731_data[3])) * v441_data);
              v730_acc += ((static_cast<float>(v731_data[4])) * v443_data);
              v730_acc += ((static_cast<float>(v731_data[5])) * v445_data);
              v730_acc += ((static_cast<float>(v731_data[6])) * v447_data);
              v730_acc += ((static_cast<float>(v731_data[7])) * v449_data);
              v730_acc += ((static_cast<float>(v731_data[8])) * v451_data);
              v730_acc += ((static_cast<float>(v731_data[9])) * v453_data);
              v730_acc += ((static_cast<float>(v731_data[10])) * v455_data);
              v730_acc += ((static_cast<float>(v731_data[11])) * v457_data);
              v730_acc += ((static_cast<float>(v731_data[12])) * v459_data);
              v730_acc += ((static_cast<float>(v731_data[13])) * v461_data);
              v730_acc += ((static_cast<float>(v731_data[14])) * v463_data);
              v730_acc += ((static_cast<float>(v731_data[15])) * v465_data);
              ir1.template select<16, 1>(128) = v730_acc;
              // r1 = ir1
              #pragma unroll
              for (int32_t v763_n0 = 0; v763_n0 < 1; ++v763_n0) {
                int32_t v765_a = v763_n0 * 16;
                #pragma unroll
                for (int32_t v764_n1 = 0; v764_n1 < 9; ++v764_n1) {
                  int32_t v767_a = v765_a + (v764_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v768_data(ir1.template select<16, 1>(v767_a));
                  r1.template select<16, 1>(v767_a) = v768_data;
                }
              }
              // glb_m2 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v769_i0 = 0; v769_i0 < 1; ++v769_i0) {
                int32_t v771_a = v769_i0 * 16;
                #pragma unroll
                for (int32_t v770_i1 = 0; v770_i1 < 9; ++v770_i1) {
                  int32_t v773_a = v771_a + (v770_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v774_data(r1.template select<16, 1>(v773_a));
                  v774_data.copy_to(glb_m2 + (v773_a));
                }
              }
            }
            tensorforge::prefetchL2<153>(&pf_glb_m1[0]);
          }
        }
      }
    });
  });
}

