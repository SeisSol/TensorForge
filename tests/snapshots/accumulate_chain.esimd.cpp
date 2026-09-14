// === base name ===
kernel_0d74c915d8def092

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_0d74c915d8def092 = {{1, 16, 1}, 16, 12, 1, 16, 7168, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_0d74c915d8def092(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_0d74c915d8def092(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_0d74c915d8def092(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 1792 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_0d74c915d8def092(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_0d74c915d8def092(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_0d74c915d8def092(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, m6, m6_extraOffset, m7, m7_extraOffset, m8, m8_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_0d74c915d8def092(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<1792 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 7168 B shared, occupancy grid
        // operands:
        //   m0 12×8(12×8) {0..12}×{0..8} strided
        //   m1 12×12(12×12) {0..12}×{0..12} strided
        //   m2 12×8(12×8) {0..12}×{0..8} strided
        //   m3 12×12(12×12) {0..12}×{0..12} strided
        //   m4 12×8(12×8) {0..12}×{0..8} strided
        //   m5 12×12(12×12) {0..12}×{0..12} strided
        //   m6 12×8(12×8) {0..12}×{0..8} strided
        //   m7 12×12(12×12) {0..12}×{0..12} strided
        //   m8 12×8(12×8) {0..12}×{0..8} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        //   m0[i,j] += m3[i,k] × m4[k,j]
        //   m0[i,j] += m5[i,k] × m6[k,j]
        //   m0[i,j] += m7[i,k] × m8[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1792}],"shared_bytes":7168,"shared_elements":1792,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[12,8]],"name":"m0","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A0","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B0","bbox":[[0,0],[12,8]],"name":"m2","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A1","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[12,8]],"name":"m4","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A2","bbox":[[0,0],[12,12]],"name":"m5","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[12,8]],"name":"m6","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A3","bbox":[[0,0],[12,12]],"name":"m7","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B3","bbox":[[0,0],[12,8]],"name":"m8","ordered":false,"parts":1,"shape":[12,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m7","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m8","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (112 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (96);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s2 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s3 = localShrMem0 + (0);
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 96 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v8_batchId0 * 144 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v8_batchId0 * 96 + 0 + m4_extraOffset];
              const float *const __restrict__ glb_m5 = &m5[v8_batchId0 * 144 + 0 + m5_extraOffset];
              const float *const __restrict__ glb_m6 = &m6[v8_batchId0 * 96 + 0 + m6_extraOffset];
              const float *const __restrict__ glb_m7 = &m7[v8_batchId0 * 144 + 0 + m7_extraOffset];
              const float *const __restrict__ glb_m8 = &m8[v8_batchId0 * 96 + 0 + m8_extraOffset];
              tensorforge::intel_esimd::simd<float, 192> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v26_i1 = 0; v26_i1 < 12; ++v26_i1) {
                tensorforge::intel_esimd::simd<float, 12> v31_data;
                v31_data.copy_from(glb_m1 + ((v26_i1 * 12)));
                r0.template select<12, 1>((v26_i1 * 16)) = v31_data;
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v34_ld;
              v34_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v34_ld);
              tensorforge::intel_esimd::simd<float, 32> v35_ld;
              v35_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 64));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 2 * 0 + 64), v35_ld);
              // wait(r0 = load{g>r}(glb_m1););
              tensorforge::intel_esimd::simd<float, 192> r2(0.0f);
              // r2 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v37_i1 = 0; v37_i1 < 12; ++v37_i1) {
                tensorforge::intel_esimd::simd<float, 12> v42_data;
                v42_data.copy_from(glb_m3 + ((v37_i1 * 12)));
                r2.template select<12, 1>((v37_i1 * 16)) = v42_data;
              }
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 128> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 12), (0, 8)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 128> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v48_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v49_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v50_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v51_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v52_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v53_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v54_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v55_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v56_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v57_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v58_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v59_acc{};
              tensorforge::intel_esimd::simd<float, 16> v63_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v59_acc += ((static_cast<float>(v63_data[0])) * v47_data);
              v59_acc += ((static_cast<float>(v63_data[1])) * v48_data);
              v59_acc += ((static_cast<float>(v63_data[2])) * v49_data);
              v59_acc += ((static_cast<float>(v63_data[3])) * v50_data);
              v59_acc += ((static_cast<float>(v63_data[4])) * v51_data);
              v59_acc += ((static_cast<float>(v63_data[5])) * v52_data);
              v59_acc += ((static_cast<float>(v63_data[6])) * v53_data);
              v59_acc += ((static_cast<float>(v63_data[7])) * v54_data);
              v59_acc += ((static_cast<float>(v63_data[8])) * v55_data);
              v59_acc += ((static_cast<float>(v63_data[9])) * v56_data);
              v59_acc += ((static_cast<float>(v63_data[10])) * v57_data);
              v59_acc += ((static_cast<float>(v63_data[11])) * v58_data);
              ir1.template select<16, 1>(0) = v59_acc;
              tensorforge::intel_esimd::simd<float, 16> v88_acc{};
              tensorforge::intel_esimd::simd<float, 16> v90_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              v88_acc += ((static_cast<float>(v90_data[0])) * v47_data);
              v88_acc += ((static_cast<float>(v90_data[1])) * v48_data);
              v88_acc += ((static_cast<float>(v90_data[2])) * v49_data);
              v88_acc += ((static_cast<float>(v90_data[3])) * v50_data);
              v88_acc += ((static_cast<float>(v90_data[4])) * v51_data);
              v88_acc += ((static_cast<float>(v90_data[5])) * v52_data);
              v88_acc += ((static_cast<float>(v90_data[6])) * v53_data);
              v88_acc += ((static_cast<float>(v90_data[7])) * v54_data);
              v88_acc += ((static_cast<float>(v90_data[8])) * v55_data);
              v88_acc += ((static_cast<float>(v90_data[9])) * v56_data);
              v88_acc += ((static_cast<float>(v90_data[10])) * v57_data);
              v88_acc += ((static_cast<float>(v90_data[11])) * v58_data);
              ir1.template select<16, 1>(16) = v88_acc;
              tensorforge::intel_esimd::simd<float, 16> v115_acc{};
              tensorforge::intel_esimd::simd<float, 16> v117_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              v115_acc += ((static_cast<float>(v117_data[0])) * v47_data);
              v115_acc += ((static_cast<float>(v117_data[1])) * v48_data);
              v115_acc += ((static_cast<float>(v117_data[2])) * v49_data);
              v115_acc += ((static_cast<float>(v117_data[3])) * v50_data);
              v115_acc += ((static_cast<float>(v117_data[4])) * v51_data);
              v115_acc += ((static_cast<float>(v117_data[5])) * v52_data);
              v115_acc += ((static_cast<float>(v117_data[6])) * v53_data);
              v115_acc += ((static_cast<float>(v117_data[7])) * v54_data);
              v115_acc += ((static_cast<float>(v117_data[8])) * v55_data);
              v115_acc += ((static_cast<float>(v117_data[9])) * v56_data);
              v115_acc += ((static_cast<float>(v117_data[10])) * v57_data);
              v115_acc += ((static_cast<float>(v117_data[11])) * v58_data);
              ir1.template select<16, 1>(32) = v115_acc;
              tensorforge::intel_esimd::simd<float, 16> v142_acc{};
              tensorforge::intel_esimd::simd<float, 16> v144_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              v142_acc += ((static_cast<float>(v144_data[0])) * v47_data);
              v142_acc += ((static_cast<float>(v144_data[1])) * v48_data);
              v142_acc += ((static_cast<float>(v144_data[2])) * v49_data);
              v142_acc += ((static_cast<float>(v144_data[3])) * v50_data);
              v142_acc += ((static_cast<float>(v144_data[4])) * v51_data);
              v142_acc += ((static_cast<float>(v144_data[5])) * v52_data);
              v142_acc += ((static_cast<float>(v144_data[6])) * v53_data);
              v142_acc += ((static_cast<float>(v144_data[7])) * v54_data);
              v142_acc += ((static_cast<float>(v144_data[8])) * v55_data);
              v142_acc += ((static_cast<float>(v144_data[9])) * v56_data);
              v142_acc += ((static_cast<float>(v144_data[10])) * v57_data);
              v142_acc += ((static_cast<float>(v144_data[11])) * v58_data);
              ir1.template select<16, 1>(48) = v142_acc;
              tensorforge::intel_esimd::simd<float, 16> v169_acc{};
              tensorforge::intel_esimd::simd<float, 16> v171_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v169_acc += ((static_cast<float>(v171_data[0])) * v47_data);
              v169_acc += ((static_cast<float>(v171_data[1])) * v48_data);
              v169_acc += ((static_cast<float>(v171_data[2])) * v49_data);
              v169_acc += ((static_cast<float>(v171_data[3])) * v50_data);
              v169_acc += ((static_cast<float>(v171_data[4])) * v51_data);
              v169_acc += ((static_cast<float>(v171_data[5])) * v52_data);
              v169_acc += ((static_cast<float>(v171_data[6])) * v53_data);
              v169_acc += ((static_cast<float>(v171_data[7])) * v54_data);
              v169_acc += ((static_cast<float>(v171_data[8])) * v55_data);
              v169_acc += ((static_cast<float>(v171_data[9])) * v56_data);
              v169_acc += ((static_cast<float>(v171_data[10])) * v57_data);
              v169_acc += ((static_cast<float>(v171_data[11])) * v58_data);
              ir1.template select<16, 1>(64) = v169_acc;
              tensorforge::intel_esimd::simd<float, 16> v196_acc{};
              tensorforge::intel_esimd::simd<float, 16> v198_data = tensorforge::slmLoad<float, 16>(s0 + (60_i32));
              v196_acc += ((static_cast<float>(v198_data[0])) * v47_data);
              v196_acc += ((static_cast<float>(v198_data[1])) * v48_data);
              v196_acc += ((static_cast<float>(v198_data[2])) * v49_data);
              v196_acc += ((static_cast<float>(v198_data[3])) * v50_data);
              v196_acc += ((static_cast<float>(v198_data[4])) * v51_data);
              v196_acc += ((static_cast<float>(v198_data[5])) * v52_data);
              v196_acc += ((static_cast<float>(v198_data[6])) * v53_data);
              v196_acc += ((static_cast<float>(v198_data[7])) * v54_data);
              v196_acc += ((static_cast<float>(v198_data[8])) * v55_data);
              v196_acc += ((static_cast<float>(v198_data[9])) * v56_data);
              v196_acc += ((static_cast<float>(v198_data[10])) * v57_data);
              v196_acc += ((static_cast<float>(v198_data[11])) * v58_data);
              ir1.template select<16, 1>(80) = v196_acc;
              tensorforge::intel_esimd::simd<float, 16> v223_acc{};
              tensorforge::intel_esimd::simd<float, 16> v225_data = tensorforge::slmLoad<float, 16>(s0 + (72_i32));
              v223_acc += ((static_cast<float>(v225_data[0])) * v47_data);
              v223_acc += ((static_cast<float>(v225_data[1])) * v48_data);
              v223_acc += ((static_cast<float>(v225_data[2])) * v49_data);
              v223_acc += ((static_cast<float>(v225_data[3])) * v50_data);
              v223_acc += ((static_cast<float>(v225_data[4])) * v51_data);
              v223_acc += ((static_cast<float>(v225_data[5])) * v52_data);
              v223_acc += ((static_cast<float>(v225_data[6])) * v53_data);
              v223_acc += ((static_cast<float>(v225_data[7])) * v54_data);
              v223_acc += ((static_cast<float>(v225_data[8])) * v55_data);
              v223_acc += ((static_cast<float>(v225_data[9])) * v56_data);
              v223_acc += ((static_cast<float>(v225_data[10])) * v57_data);
              v223_acc += ((static_cast<float>(v225_data[11])) * v58_data);
              ir1.template select<16, 1>(96) = v223_acc;
              tensorforge::intel_esimd::simd<float, 16> v250_acc{};
              tensorforge::intel_esimd::simd<float, 16> v252_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v250_acc += ((static_cast<float>(v252_data[0])) * v47_data);
              v250_acc += ((static_cast<float>(v252_data[1])) * v48_data);
              v250_acc += ((static_cast<float>(v252_data[2])) * v49_data);
              v250_acc += ((static_cast<float>(v252_data[3])) * v50_data);
              v250_acc += ((static_cast<float>(v252_data[4])) * v51_data);
              v250_acc += ((static_cast<float>(v252_data[5])) * v52_data);
              v250_acc += ((static_cast<float>(v252_data[6])) * v53_data);
              v250_acc += ((static_cast<float>(v252_data[7])) * v54_data);
              v250_acc += ((static_cast<float>(v252_data[8])) * v55_data);
              v250_acc += ((static_cast<float>(v252_data[9])) * v56_data);
              v250_acc += ((static_cast<float>(v252_data[10])) * v57_data);
              v250_acc += ((static_cast<float>(v252_data[11])) * v58_data);
              ir1.template select<16, 1>(112) = v250_acc;
              // r1 = ir1
              #pragma unroll
              for (int32_t v277_n1 = 0; v277_n1 < 8; ++v277_n1) {
                int32_t v278_a = v277_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v280_data(ir1.template select<12, 1>(v278_a));
                r1.template select<12, 1>(v278_a) = v280_data;
              }
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v281_ld;
              v281_ld.copy_from(glb_m4 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 4 * 0 + 0), v281_ld);
              tensorforge::intel_esimd::simd<float, 32> v282_ld;
              v282_ld.copy_from(glb_m4 + (0 + 0 + 2 * 0 + 64));
              tensorforge::slmStore<float, 32>(s1 + (0 + 0 + 2 * 0 + 64), v282_ld);
              // wait(r2 = load{g>r}(glb_m3););
              tensorforge::intel_esimd::simd<float, 192> r4(0.0f);
              // r4 = load{g>r}(glb_m5);
              #pragma unroll
              for (int32_t v284_i1 = 0; v284_i1 < 12; ++v284_i1) {
                tensorforge::intel_esimd::simd<float, 12> v289_data;
                v289_data.copy_from(glb_m5 + ((v284_i1 * 12)));
                r4.template select<12, 1>((v284_i1 * 16)) = v289_data;
              }
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              tensorforge::intel_esimd::simd<float, 128> r3(0.0f);
              // ir3 = +(r2 * s1)
              // [(0, 12), (0, 8)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 128> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v294_data(r2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v295_data(r2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v296_data(r2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v297_data(r2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v298_data(r2.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v299_data(r2.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v300_data(r2.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v301_data(r2.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v302_data(r2.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v303_data(r2.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v304_data(r2.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v305_data(r2.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v306_acc{};
              tensorforge::intel_esimd::simd<float, 16> v310_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v306_acc += ((static_cast<float>(v310_data[0])) * v294_data);
              v306_acc += ((static_cast<float>(v310_data[1])) * v295_data);
              v306_acc += ((static_cast<float>(v310_data[2])) * v296_data);
              v306_acc += ((static_cast<float>(v310_data[3])) * v297_data);
              v306_acc += ((static_cast<float>(v310_data[4])) * v298_data);
              v306_acc += ((static_cast<float>(v310_data[5])) * v299_data);
              v306_acc += ((static_cast<float>(v310_data[6])) * v300_data);
              v306_acc += ((static_cast<float>(v310_data[7])) * v301_data);
              v306_acc += ((static_cast<float>(v310_data[8])) * v302_data);
              v306_acc += ((static_cast<float>(v310_data[9])) * v303_data);
              v306_acc += ((static_cast<float>(v310_data[10])) * v304_data);
              v306_acc += ((static_cast<float>(v310_data[11])) * v305_data);
              ir3.template select<16, 1>(0) = v306_acc;
              tensorforge::intel_esimd::simd<float, 16> v335_acc{};
              tensorforge::intel_esimd::simd<float, 16> v337_data = tensorforge::slmLoad<float, 16>(s1 + (12_i32));
              v335_acc += ((static_cast<float>(v337_data[0])) * v294_data);
              v335_acc += ((static_cast<float>(v337_data[1])) * v295_data);
              v335_acc += ((static_cast<float>(v337_data[2])) * v296_data);
              v335_acc += ((static_cast<float>(v337_data[3])) * v297_data);
              v335_acc += ((static_cast<float>(v337_data[4])) * v298_data);
              v335_acc += ((static_cast<float>(v337_data[5])) * v299_data);
              v335_acc += ((static_cast<float>(v337_data[6])) * v300_data);
              v335_acc += ((static_cast<float>(v337_data[7])) * v301_data);
              v335_acc += ((static_cast<float>(v337_data[8])) * v302_data);
              v335_acc += ((static_cast<float>(v337_data[9])) * v303_data);
              v335_acc += ((static_cast<float>(v337_data[10])) * v304_data);
              v335_acc += ((static_cast<float>(v337_data[11])) * v305_data);
              ir3.template select<16, 1>(16) = v335_acc;
              tensorforge::intel_esimd::simd<float, 16> v362_acc{};
              tensorforge::intel_esimd::simd<float, 16> v364_data = tensorforge::slmLoad<float, 16>(s1 + (24_i32));
              v362_acc += ((static_cast<float>(v364_data[0])) * v294_data);
              v362_acc += ((static_cast<float>(v364_data[1])) * v295_data);
              v362_acc += ((static_cast<float>(v364_data[2])) * v296_data);
              v362_acc += ((static_cast<float>(v364_data[3])) * v297_data);
              v362_acc += ((static_cast<float>(v364_data[4])) * v298_data);
              v362_acc += ((static_cast<float>(v364_data[5])) * v299_data);
              v362_acc += ((static_cast<float>(v364_data[6])) * v300_data);
              v362_acc += ((static_cast<float>(v364_data[7])) * v301_data);
              v362_acc += ((static_cast<float>(v364_data[8])) * v302_data);
              v362_acc += ((static_cast<float>(v364_data[9])) * v303_data);
              v362_acc += ((static_cast<float>(v364_data[10])) * v304_data);
              v362_acc += ((static_cast<float>(v364_data[11])) * v305_data);
              ir3.template select<16, 1>(32) = v362_acc;
              tensorforge::intel_esimd::simd<float, 16> v389_acc{};
              tensorforge::intel_esimd::simd<float, 16> v391_data = tensorforge::slmLoad<float, 16>(s1 + (36_i32));
              v389_acc += ((static_cast<float>(v391_data[0])) * v294_data);
              v389_acc += ((static_cast<float>(v391_data[1])) * v295_data);
              v389_acc += ((static_cast<float>(v391_data[2])) * v296_data);
              v389_acc += ((static_cast<float>(v391_data[3])) * v297_data);
              v389_acc += ((static_cast<float>(v391_data[4])) * v298_data);
              v389_acc += ((static_cast<float>(v391_data[5])) * v299_data);
              v389_acc += ((static_cast<float>(v391_data[6])) * v300_data);
              v389_acc += ((static_cast<float>(v391_data[7])) * v301_data);
              v389_acc += ((static_cast<float>(v391_data[8])) * v302_data);
              v389_acc += ((static_cast<float>(v391_data[9])) * v303_data);
              v389_acc += ((static_cast<float>(v391_data[10])) * v304_data);
              v389_acc += ((static_cast<float>(v391_data[11])) * v305_data);
              ir3.template select<16, 1>(48) = v389_acc;
              tensorforge::intel_esimd::simd<float, 16> v416_acc{};
              tensorforge::intel_esimd::simd<float, 16> v418_data = tensorforge::slmLoad<float, 16>(s1 + (48_i32));
              v416_acc += ((static_cast<float>(v418_data[0])) * v294_data);
              v416_acc += ((static_cast<float>(v418_data[1])) * v295_data);
              v416_acc += ((static_cast<float>(v418_data[2])) * v296_data);
              v416_acc += ((static_cast<float>(v418_data[3])) * v297_data);
              v416_acc += ((static_cast<float>(v418_data[4])) * v298_data);
              v416_acc += ((static_cast<float>(v418_data[5])) * v299_data);
              v416_acc += ((static_cast<float>(v418_data[6])) * v300_data);
              v416_acc += ((static_cast<float>(v418_data[7])) * v301_data);
              v416_acc += ((static_cast<float>(v418_data[8])) * v302_data);
              v416_acc += ((static_cast<float>(v418_data[9])) * v303_data);
              v416_acc += ((static_cast<float>(v418_data[10])) * v304_data);
              v416_acc += ((static_cast<float>(v418_data[11])) * v305_data);
              ir3.template select<16, 1>(64) = v416_acc;
              tensorforge::intel_esimd::simd<float, 16> v443_acc{};
              tensorforge::intel_esimd::simd<float, 16> v445_data = tensorforge::slmLoad<float, 16>(s1 + (60_i32));
              v443_acc += ((static_cast<float>(v445_data[0])) * v294_data);
              v443_acc += ((static_cast<float>(v445_data[1])) * v295_data);
              v443_acc += ((static_cast<float>(v445_data[2])) * v296_data);
              v443_acc += ((static_cast<float>(v445_data[3])) * v297_data);
              v443_acc += ((static_cast<float>(v445_data[4])) * v298_data);
              v443_acc += ((static_cast<float>(v445_data[5])) * v299_data);
              v443_acc += ((static_cast<float>(v445_data[6])) * v300_data);
              v443_acc += ((static_cast<float>(v445_data[7])) * v301_data);
              v443_acc += ((static_cast<float>(v445_data[8])) * v302_data);
              v443_acc += ((static_cast<float>(v445_data[9])) * v303_data);
              v443_acc += ((static_cast<float>(v445_data[10])) * v304_data);
              v443_acc += ((static_cast<float>(v445_data[11])) * v305_data);
              ir3.template select<16, 1>(80) = v443_acc;
              tensorforge::intel_esimd::simd<float, 16> v470_acc{};
              tensorforge::intel_esimd::simd<float, 16> v472_data = tensorforge::slmLoad<float, 16>(s1 + (72_i32));
              v470_acc += ((static_cast<float>(v472_data[0])) * v294_data);
              v470_acc += ((static_cast<float>(v472_data[1])) * v295_data);
              v470_acc += ((static_cast<float>(v472_data[2])) * v296_data);
              v470_acc += ((static_cast<float>(v472_data[3])) * v297_data);
              v470_acc += ((static_cast<float>(v472_data[4])) * v298_data);
              v470_acc += ((static_cast<float>(v472_data[5])) * v299_data);
              v470_acc += ((static_cast<float>(v472_data[6])) * v300_data);
              v470_acc += ((static_cast<float>(v472_data[7])) * v301_data);
              v470_acc += ((static_cast<float>(v472_data[8])) * v302_data);
              v470_acc += ((static_cast<float>(v472_data[9])) * v303_data);
              v470_acc += ((static_cast<float>(v472_data[10])) * v304_data);
              v470_acc += ((static_cast<float>(v472_data[11])) * v305_data);
              ir3.template select<16, 1>(96) = v470_acc;
              tensorforge::intel_esimd::simd<float, 16> v497_acc{};
              tensorforge::intel_esimd::simd<float, 16> v499_data = tensorforge::slmLoad<float, 16>(s1 + (84_i32));
              v497_acc += ((static_cast<float>(v499_data[0])) * v294_data);
              v497_acc += ((static_cast<float>(v499_data[1])) * v295_data);
              v497_acc += ((static_cast<float>(v499_data[2])) * v296_data);
              v497_acc += ((static_cast<float>(v499_data[3])) * v297_data);
              v497_acc += ((static_cast<float>(v499_data[4])) * v298_data);
              v497_acc += ((static_cast<float>(v499_data[5])) * v299_data);
              v497_acc += ((static_cast<float>(v499_data[6])) * v300_data);
              v497_acc += ((static_cast<float>(v499_data[7])) * v301_data);
              v497_acc += ((static_cast<float>(v499_data[8])) * v302_data);
              v497_acc += ((static_cast<float>(v499_data[9])) * v303_data);
              v497_acc += ((static_cast<float>(v499_data[10])) * v304_data);
              v497_acc += ((static_cast<float>(v499_data[11])) * v305_data);
              ir3.template select<16, 1>(112) = v497_acc;
              // r3 = ir3 + r1
              #pragma unroll
              for (int32_t v524_n1 = 0; v524_n1 < 8; ++v524_n1) {
                int32_t v525_a = v524_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v527_data(ir3.template select<12, 1>(v525_a));
                tensorforge::intel_esimd::simd<float, 12> v528_data(r1.template select<12, 1>(v525_a));
                r3.template select<12, 1>(v525_a) = (v528_data + v527_data);
              }
              // s2 = load{g>s}(glb_m6[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v530_ld;
              v530_ld.copy_from(glb_m6 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s2 + (0 + 0 + 4 * 0 + 0), v530_ld);
              tensorforge::intel_esimd::simd<float, 32> v531_ld;
              v531_ld.copy_from(glb_m6 + (0 + 0 + 2 * 0 + 64));
              tensorforge::slmStore<float, 32>(s2 + (0 + 0 + 2 * 0 + 64), v531_ld);
              // wait(r4 = load{g>r}(glb_m5););
              tensorforge::intel_esimd::simd<float, 192> r6(0.0f);
              // r6 = load{g>r}(glb_m7);
              #pragma unroll
              for (int32_t v533_i1 = 0; v533_i1 < 12; ++v533_i1) {
                tensorforge::intel_esimd::simd<float, 12> v538_data;
                v538_data.copy_from(glb_m7 + ((v533_i1 * 12)));
                r6.template select<12, 1>((v533_i1 * 16)) = v538_data;
              }
              // wait(s2 = load{g>s}(glb_m6[0, 1]));
              tensorforge::intel_esimd::simd<float, 128> r5(0.0f);
              // ir5 = +(r4 * s2)
              // [(0, 12), (0, 8)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 128> ir5(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v543_data(r4.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v544_data(r4.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v545_data(r4.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v546_data(r4.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v547_data(r4.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v548_data(r4.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v549_data(r4.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v550_data(r4.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v551_data(r4.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v552_data(r4.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v553_data(r4.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v554_data(r4.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v555_acc{};
              tensorforge::intel_esimd::simd<float, 16> v559_data = tensorforge::slmLoad<float, 16>(s2 + (0_i32));
              v555_acc += ((static_cast<float>(v559_data[0])) * v543_data);
              v555_acc += ((static_cast<float>(v559_data[1])) * v544_data);
              v555_acc += ((static_cast<float>(v559_data[2])) * v545_data);
              v555_acc += ((static_cast<float>(v559_data[3])) * v546_data);
              v555_acc += ((static_cast<float>(v559_data[4])) * v547_data);
              v555_acc += ((static_cast<float>(v559_data[5])) * v548_data);
              v555_acc += ((static_cast<float>(v559_data[6])) * v549_data);
              v555_acc += ((static_cast<float>(v559_data[7])) * v550_data);
              v555_acc += ((static_cast<float>(v559_data[8])) * v551_data);
              v555_acc += ((static_cast<float>(v559_data[9])) * v552_data);
              v555_acc += ((static_cast<float>(v559_data[10])) * v553_data);
              v555_acc += ((static_cast<float>(v559_data[11])) * v554_data);
              ir5.template select<16, 1>(0) = v555_acc;
              tensorforge::intel_esimd::simd<float, 16> v584_acc{};
              tensorforge::intel_esimd::simd<float, 16> v586_data = tensorforge::slmLoad<float, 16>(s2 + (12_i32));
              v584_acc += ((static_cast<float>(v586_data[0])) * v543_data);
              v584_acc += ((static_cast<float>(v586_data[1])) * v544_data);
              v584_acc += ((static_cast<float>(v586_data[2])) * v545_data);
              v584_acc += ((static_cast<float>(v586_data[3])) * v546_data);
              v584_acc += ((static_cast<float>(v586_data[4])) * v547_data);
              v584_acc += ((static_cast<float>(v586_data[5])) * v548_data);
              v584_acc += ((static_cast<float>(v586_data[6])) * v549_data);
              v584_acc += ((static_cast<float>(v586_data[7])) * v550_data);
              v584_acc += ((static_cast<float>(v586_data[8])) * v551_data);
              v584_acc += ((static_cast<float>(v586_data[9])) * v552_data);
              v584_acc += ((static_cast<float>(v586_data[10])) * v553_data);
              v584_acc += ((static_cast<float>(v586_data[11])) * v554_data);
              ir5.template select<16, 1>(16) = v584_acc;
              tensorforge::intel_esimd::simd<float, 16> v611_acc{};
              tensorforge::intel_esimd::simd<float, 16> v613_data = tensorforge::slmLoad<float, 16>(s2 + (24_i32));
              v611_acc += ((static_cast<float>(v613_data[0])) * v543_data);
              v611_acc += ((static_cast<float>(v613_data[1])) * v544_data);
              v611_acc += ((static_cast<float>(v613_data[2])) * v545_data);
              v611_acc += ((static_cast<float>(v613_data[3])) * v546_data);
              v611_acc += ((static_cast<float>(v613_data[4])) * v547_data);
              v611_acc += ((static_cast<float>(v613_data[5])) * v548_data);
              v611_acc += ((static_cast<float>(v613_data[6])) * v549_data);
              v611_acc += ((static_cast<float>(v613_data[7])) * v550_data);
              v611_acc += ((static_cast<float>(v613_data[8])) * v551_data);
              v611_acc += ((static_cast<float>(v613_data[9])) * v552_data);
              v611_acc += ((static_cast<float>(v613_data[10])) * v553_data);
              v611_acc += ((static_cast<float>(v613_data[11])) * v554_data);
              ir5.template select<16, 1>(32) = v611_acc;
              tensorforge::intel_esimd::simd<float, 16> v638_acc{};
              tensorforge::intel_esimd::simd<float, 16> v640_data = tensorforge::slmLoad<float, 16>(s2 + (36_i32));
              v638_acc += ((static_cast<float>(v640_data[0])) * v543_data);
              v638_acc += ((static_cast<float>(v640_data[1])) * v544_data);
              v638_acc += ((static_cast<float>(v640_data[2])) * v545_data);
              v638_acc += ((static_cast<float>(v640_data[3])) * v546_data);
              v638_acc += ((static_cast<float>(v640_data[4])) * v547_data);
              v638_acc += ((static_cast<float>(v640_data[5])) * v548_data);
              v638_acc += ((static_cast<float>(v640_data[6])) * v549_data);
              v638_acc += ((static_cast<float>(v640_data[7])) * v550_data);
              v638_acc += ((static_cast<float>(v640_data[8])) * v551_data);
              v638_acc += ((static_cast<float>(v640_data[9])) * v552_data);
              v638_acc += ((static_cast<float>(v640_data[10])) * v553_data);
              v638_acc += ((static_cast<float>(v640_data[11])) * v554_data);
              ir5.template select<16, 1>(48) = v638_acc;
              tensorforge::intel_esimd::simd<float, 16> v665_acc{};
              tensorforge::intel_esimd::simd<float, 16> v667_data = tensorforge::slmLoad<float, 16>(s2 + (48_i32));
              v665_acc += ((static_cast<float>(v667_data[0])) * v543_data);
              v665_acc += ((static_cast<float>(v667_data[1])) * v544_data);
              v665_acc += ((static_cast<float>(v667_data[2])) * v545_data);
              v665_acc += ((static_cast<float>(v667_data[3])) * v546_data);
              v665_acc += ((static_cast<float>(v667_data[4])) * v547_data);
              v665_acc += ((static_cast<float>(v667_data[5])) * v548_data);
              v665_acc += ((static_cast<float>(v667_data[6])) * v549_data);
              v665_acc += ((static_cast<float>(v667_data[7])) * v550_data);
              v665_acc += ((static_cast<float>(v667_data[8])) * v551_data);
              v665_acc += ((static_cast<float>(v667_data[9])) * v552_data);
              v665_acc += ((static_cast<float>(v667_data[10])) * v553_data);
              v665_acc += ((static_cast<float>(v667_data[11])) * v554_data);
              ir5.template select<16, 1>(64) = v665_acc;
              tensorforge::intel_esimd::simd<float, 16> v692_acc{};
              tensorforge::intel_esimd::simd<float, 16> v694_data = tensorforge::slmLoad<float, 16>(s2 + (60_i32));
              v692_acc += ((static_cast<float>(v694_data[0])) * v543_data);
              v692_acc += ((static_cast<float>(v694_data[1])) * v544_data);
              v692_acc += ((static_cast<float>(v694_data[2])) * v545_data);
              v692_acc += ((static_cast<float>(v694_data[3])) * v546_data);
              v692_acc += ((static_cast<float>(v694_data[4])) * v547_data);
              v692_acc += ((static_cast<float>(v694_data[5])) * v548_data);
              v692_acc += ((static_cast<float>(v694_data[6])) * v549_data);
              v692_acc += ((static_cast<float>(v694_data[7])) * v550_data);
              v692_acc += ((static_cast<float>(v694_data[8])) * v551_data);
              v692_acc += ((static_cast<float>(v694_data[9])) * v552_data);
              v692_acc += ((static_cast<float>(v694_data[10])) * v553_data);
              v692_acc += ((static_cast<float>(v694_data[11])) * v554_data);
              ir5.template select<16, 1>(80) = v692_acc;
              tensorforge::intel_esimd::simd<float, 16> v719_acc{};
              tensorforge::intel_esimd::simd<float, 16> v721_data = tensorforge::slmLoad<float, 16>(s2 + (72_i32));
              v719_acc += ((static_cast<float>(v721_data[0])) * v543_data);
              v719_acc += ((static_cast<float>(v721_data[1])) * v544_data);
              v719_acc += ((static_cast<float>(v721_data[2])) * v545_data);
              v719_acc += ((static_cast<float>(v721_data[3])) * v546_data);
              v719_acc += ((static_cast<float>(v721_data[4])) * v547_data);
              v719_acc += ((static_cast<float>(v721_data[5])) * v548_data);
              v719_acc += ((static_cast<float>(v721_data[6])) * v549_data);
              v719_acc += ((static_cast<float>(v721_data[7])) * v550_data);
              v719_acc += ((static_cast<float>(v721_data[8])) * v551_data);
              v719_acc += ((static_cast<float>(v721_data[9])) * v552_data);
              v719_acc += ((static_cast<float>(v721_data[10])) * v553_data);
              v719_acc += ((static_cast<float>(v721_data[11])) * v554_data);
              ir5.template select<16, 1>(96) = v719_acc;
              tensorforge::intel_esimd::simd<float, 16> v746_acc{};
              tensorforge::intel_esimd::simd<float, 16> v748_data = tensorforge::slmLoad<float, 16>(s2 + (84_i32));
              v746_acc += ((static_cast<float>(v748_data[0])) * v543_data);
              v746_acc += ((static_cast<float>(v748_data[1])) * v544_data);
              v746_acc += ((static_cast<float>(v748_data[2])) * v545_data);
              v746_acc += ((static_cast<float>(v748_data[3])) * v546_data);
              v746_acc += ((static_cast<float>(v748_data[4])) * v547_data);
              v746_acc += ((static_cast<float>(v748_data[5])) * v548_data);
              v746_acc += ((static_cast<float>(v748_data[6])) * v549_data);
              v746_acc += ((static_cast<float>(v748_data[7])) * v550_data);
              v746_acc += ((static_cast<float>(v748_data[8])) * v551_data);
              v746_acc += ((static_cast<float>(v748_data[9])) * v552_data);
              v746_acc += ((static_cast<float>(v748_data[10])) * v553_data);
              v746_acc += ((static_cast<float>(v748_data[11])) * v554_data);
              ir5.template select<16, 1>(112) = v746_acc;
              // r5 = ir5 + r3
              #pragma unroll
              for (int32_t v773_n1 = 0; v773_n1 < 8; ++v773_n1) {
                int32_t v774_a = v773_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v776_data(ir5.template select<12, 1>(v774_a));
                tensorforge::intel_esimd::simd<float, 12> v777_data(r3.template select<12, 1>(v774_a));
                r5.template select<12, 1>(v774_a) = (v777_data + v776_data);
              }
              // s3 = load{g>s}(glb_m8[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v779_ld;
              v779_ld.copy_from(glb_m8 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s3 + (0 + 0 + 4 * 0 + 0), v779_ld);
              tensorforge::intel_esimd::simd<float, 32> v780_ld;
              v780_ld.copy_from(glb_m8 + (0 + 0 + 2 * 0 + 64));
              tensorforge::slmStore<float, 32>(s3 + (0 + 0 + 2 * 0 + 64), v780_ld);
              // wait(r6 = load{g>r}(glb_m7););
              // wait(s3 = load{g>s}(glb_m8[0, 1]));
              tensorforge::intel_esimd::simd<float, 128> r7(0.0f);
              // ir7 = +(r6 * s3)
              // [(0, 12), (0, 8)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 128> ir7(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v783_data(r6.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v784_data(r6.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v785_data(r6.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v786_data(r6.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v787_data(r6.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v788_data(r6.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v789_data(r6.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v790_data(r6.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v791_data(r6.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v792_data(r6.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v793_data(r6.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v794_data(r6.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v795_acc{};
              tensorforge::intel_esimd::simd<float, 16> v799_data = tensorforge::slmLoad<float, 16>(s3 + (0_i32));
              v795_acc += ((static_cast<float>(v799_data[0])) * v783_data);
              v795_acc += ((static_cast<float>(v799_data[1])) * v784_data);
              v795_acc += ((static_cast<float>(v799_data[2])) * v785_data);
              v795_acc += ((static_cast<float>(v799_data[3])) * v786_data);
              v795_acc += ((static_cast<float>(v799_data[4])) * v787_data);
              v795_acc += ((static_cast<float>(v799_data[5])) * v788_data);
              v795_acc += ((static_cast<float>(v799_data[6])) * v789_data);
              v795_acc += ((static_cast<float>(v799_data[7])) * v790_data);
              v795_acc += ((static_cast<float>(v799_data[8])) * v791_data);
              v795_acc += ((static_cast<float>(v799_data[9])) * v792_data);
              v795_acc += ((static_cast<float>(v799_data[10])) * v793_data);
              v795_acc += ((static_cast<float>(v799_data[11])) * v794_data);
              ir7.template select<16, 1>(0) = v795_acc;
              tensorforge::intel_esimd::simd<float, 16> v824_acc{};
              tensorforge::intel_esimd::simd<float, 16> v826_data = tensorforge::slmLoad<float, 16>(s3 + (12_i32));
              v824_acc += ((static_cast<float>(v826_data[0])) * v783_data);
              v824_acc += ((static_cast<float>(v826_data[1])) * v784_data);
              v824_acc += ((static_cast<float>(v826_data[2])) * v785_data);
              v824_acc += ((static_cast<float>(v826_data[3])) * v786_data);
              v824_acc += ((static_cast<float>(v826_data[4])) * v787_data);
              v824_acc += ((static_cast<float>(v826_data[5])) * v788_data);
              v824_acc += ((static_cast<float>(v826_data[6])) * v789_data);
              v824_acc += ((static_cast<float>(v826_data[7])) * v790_data);
              v824_acc += ((static_cast<float>(v826_data[8])) * v791_data);
              v824_acc += ((static_cast<float>(v826_data[9])) * v792_data);
              v824_acc += ((static_cast<float>(v826_data[10])) * v793_data);
              v824_acc += ((static_cast<float>(v826_data[11])) * v794_data);
              ir7.template select<16, 1>(16) = v824_acc;
              tensorforge::intel_esimd::simd<float, 16> v851_acc{};
              tensorforge::intel_esimd::simd<float, 16> v853_data = tensorforge::slmLoad<float, 16>(s3 + (24_i32));
              v851_acc += ((static_cast<float>(v853_data[0])) * v783_data);
              v851_acc += ((static_cast<float>(v853_data[1])) * v784_data);
              v851_acc += ((static_cast<float>(v853_data[2])) * v785_data);
              v851_acc += ((static_cast<float>(v853_data[3])) * v786_data);
              v851_acc += ((static_cast<float>(v853_data[4])) * v787_data);
              v851_acc += ((static_cast<float>(v853_data[5])) * v788_data);
              v851_acc += ((static_cast<float>(v853_data[6])) * v789_data);
              v851_acc += ((static_cast<float>(v853_data[7])) * v790_data);
              v851_acc += ((static_cast<float>(v853_data[8])) * v791_data);
              v851_acc += ((static_cast<float>(v853_data[9])) * v792_data);
              v851_acc += ((static_cast<float>(v853_data[10])) * v793_data);
              v851_acc += ((static_cast<float>(v853_data[11])) * v794_data);
              ir7.template select<16, 1>(32) = v851_acc;
              tensorforge::intel_esimd::simd<float, 16> v878_acc{};
              tensorforge::intel_esimd::simd<float, 16> v880_data = tensorforge::slmLoad<float, 16>(s3 + (36_i32));
              v878_acc += ((static_cast<float>(v880_data[0])) * v783_data);
              v878_acc += ((static_cast<float>(v880_data[1])) * v784_data);
              v878_acc += ((static_cast<float>(v880_data[2])) * v785_data);
              v878_acc += ((static_cast<float>(v880_data[3])) * v786_data);
              v878_acc += ((static_cast<float>(v880_data[4])) * v787_data);
              v878_acc += ((static_cast<float>(v880_data[5])) * v788_data);
              v878_acc += ((static_cast<float>(v880_data[6])) * v789_data);
              v878_acc += ((static_cast<float>(v880_data[7])) * v790_data);
              v878_acc += ((static_cast<float>(v880_data[8])) * v791_data);
              v878_acc += ((static_cast<float>(v880_data[9])) * v792_data);
              v878_acc += ((static_cast<float>(v880_data[10])) * v793_data);
              v878_acc += ((static_cast<float>(v880_data[11])) * v794_data);
              ir7.template select<16, 1>(48) = v878_acc;
              tensorforge::intel_esimd::simd<float, 16> v905_acc{};
              tensorforge::intel_esimd::simd<float, 16> v907_data = tensorforge::slmLoad<float, 16>(s3 + (48_i32));
              v905_acc += ((static_cast<float>(v907_data[0])) * v783_data);
              v905_acc += ((static_cast<float>(v907_data[1])) * v784_data);
              v905_acc += ((static_cast<float>(v907_data[2])) * v785_data);
              v905_acc += ((static_cast<float>(v907_data[3])) * v786_data);
              v905_acc += ((static_cast<float>(v907_data[4])) * v787_data);
              v905_acc += ((static_cast<float>(v907_data[5])) * v788_data);
              v905_acc += ((static_cast<float>(v907_data[6])) * v789_data);
              v905_acc += ((static_cast<float>(v907_data[7])) * v790_data);
              v905_acc += ((static_cast<float>(v907_data[8])) * v791_data);
              v905_acc += ((static_cast<float>(v907_data[9])) * v792_data);
              v905_acc += ((static_cast<float>(v907_data[10])) * v793_data);
              v905_acc += ((static_cast<float>(v907_data[11])) * v794_data);
              ir7.template select<16, 1>(64) = v905_acc;
              tensorforge::intel_esimd::simd<float, 16> v932_acc{};
              tensorforge::intel_esimd::simd<float, 16> v934_data = tensorforge::slmLoad<float, 16>(s3 + (60_i32));
              v932_acc += ((static_cast<float>(v934_data[0])) * v783_data);
              v932_acc += ((static_cast<float>(v934_data[1])) * v784_data);
              v932_acc += ((static_cast<float>(v934_data[2])) * v785_data);
              v932_acc += ((static_cast<float>(v934_data[3])) * v786_data);
              v932_acc += ((static_cast<float>(v934_data[4])) * v787_data);
              v932_acc += ((static_cast<float>(v934_data[5])) * v788_data);
              v932_acc += ((static_cast<float>(v934_data[6])) * v789_data);
              v932_acc += ((static_cast<float>(v934_data[7])) * v790_data);
              v932_acc += ((static_cast<float>(v934_data[8])) * v791_data);
              v932_acc += ((static_cast<float>(v934_data[9])) * v792_data);
              v932_acc += ((static_cast<float>(v934_data[10])) * v793_data);
              v932_acc += ((static_cast<float>(v934_data[11])) * v794_data);
              ir7.template select<16, 1>(80) = v932_acc;
              tensorforge::intel_esimd::simd<float, 16> v959_acc{};
              tensorforge::intel_esimd::simd<float, 16> v961_data = tensorforge::slmLoad<float, 16>(s3 + (72_i32));
              v959_acc += ((static_cast<float>(v961_data[0])) * v783_data);
              v959_acc += ((static_cast<float>(v961_data[1])) * v784_data);
              v959_acc += ((static_cast<float>(v961_data[2])) * v785_data);
              v959_acc += ((static_cast<float>(v961_data[3])) * v786_data);
              v959_acc += ((static_cast<float>(v961_data[4])) * v787_data);
              v959_acc += ((static_cast<float>(v961_data[5])) * v788_data);
              v959_acc += ((static_cast<float>(v961_data[6])) * v789_data);
              v959_acc += ((static_cast<float>(v961_data[7])) * v790_data);
              v959_acc += ((static_cast<float>(v961_data[8])) * v791_data);
              v959_acc += ((static_cast<float>(v961_data[9])) * v792_data);
              v959_acc += ((static_cast<float>(v961_data[10])) * v793_data);
              v959_acc += ((static_cast<float>(v961_data[11])) * v794_data);
              ir7.template select<16, 1>(96) = v959_acc;
              tensorforge::intel_esimd::simd<float, 16> v986_acc{};
              tensorforge::intel_esimd::simd<float, 16> v988_data = tensorforge::slmLoad<float, 16>(s3 + (84_i32));
              v986_acc += ((static_cast<float>(v988_data[0])) * v783_data);
              v986_acc += ((static_cast<float>(v988_data[1])) * v784_data);
              v986_acc += ((static_cast<float>(v988_data[2])) * v785_data);
              v986_acc += ((static_cast<float>(v988_data[3])) * v786_data);
              v986_acc += ((static_cast<float>(v988_data[4])) * v787_data);
              v986_acc += ((static_cast<float>(v988_data[5])) * v788_data);
              v986_acc += ((static_cast<float>(v988_data[6])) * v789_data);
              v986_acc += ((static_cast<float>(v988_data[7])) * v790_data);
              v986_acc += ((static_cast<float>(v988_data[8])) * v791_data);
              v986_acc += ((static_cast<float>(v988_data[9])) * v792_data);
              v986_acc += ((static_cast<float>(v988_data[10])) * v793_data);
              v986_acc += ((static_cast<float>(v988_data[11])) * v794_data);
              ir7.template select<16, 1>(112) = v986_acc;
              // r7 = ir7 + r5
              #pragma unroll
              for (int32_t v1013_n1 = 0; v1013_n1 < 8; ++v1013_n1) {
                int32_t v1014_a = v1013_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v1016_data(ir7.template select<12, 1>(v1014_a));
                tensorforge::intel_esimd::simd<float, 12> v1017_data(r5.template select<12, 1>(v1014_a));
                r7.template select<12, 1>(v1014_a) = (v1017_data + v1016_data);
              }
              // glb_m0 = store{r>g}(r7);
              #pragma unroll
              for (int32_t v1019_i1 = 0; v1019_i1 < 8; ++v1019_i1) {
                tensorforge::intel_esimd::simd<float, 12> v1022_data(r7.template select<12, 1>((v1019_i1 * 16)));
                v1022_data.copy_to(glb_m0 + ((v1019_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}

