// === base name ===
kernel_a5f2fac68f9ec11a

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_a5f2fac68f9ec11a = {{1, 16, 1}, 16, 12, 1, 16, 7168, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_a5f2fac68f9ec11a(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_a5f2fac68f9ec11a(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_a5f2fac68f9ec11a(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 1792 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_a5f2fac68f9ec11a(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_a5f2fac68f9ec11a(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_a5f2fac68f9ec11a(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, m6, m6_extraOffset, m7, m7_extraOffset, m8, m8_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_a5f2fac68f9ec11a(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0) {
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
            const float *const __restrict__ pf_glb_m1 = &m1[v11_batchId1 * 144 + 0 + m1_extraOffset];
            const float *const __restrict__ pf_glb_m2 = &m2[v11_batchId1 * 96 + 0 + m2_extraOffset];
            const float *const __restrict__ pf_glb_m3 = &m3[v11_batchId1 * 144 + 0 + m3_extraOffset];
            const float *const __restrict__ pf_glb_m4 = &m4[v11_batchId1 * 96 + 0 + m4_extraOffset];
            const float *const __restrict__ pf_glb_m5 = &m5[v11_batchId1 * 144 + 0 + m5_extraOffset];
            const float *const __restrict__ pf_glb_m6 = &m6[v11_batchId1 * 96 + 0 + m6_extraOffset];
            const float *const __restrict__ pf_glb_m7 = &m7[v11_batchId1 * 144 + 0 + m7_extraOffset];
            const float *const __restrict__ pf_glb_m8 = &m8[v11_batchId1 * 96 + 0 + m8_extraOffset];
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
              for (int32_t v34_i1 = 0; v34_i1 < 12; ++v34_i1) {
                tensorforge::intel_esimd::simd<float, 12> v39_data;
                v39_data.copy_from(glb_m1 + ((v34_i1 * 12)));
                r0.template select<12, 1>((v34_i1 * 16)) = v39_data;
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v42_ld;
              v42_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v42_ld);
              tensorforge::intel_esimd::simd<float, 32> v43_ld;
              v43_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 64));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 2 * 0 + 64), v43_ld);
              // wait(r0 = load{g>r}(glb_m1););
              tensorforge::intel_esimd::simd<float, 192> r2(0.0f);
              // r2 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v45_i1 = 0; v45_i1 < 12; ++v45_i1) {
                tensorforge::intel_esimd::simd<float, 12> v50_data;
                v50_data.copy_from(glb_m3 + ((v45_i1 * 12)));
                r2.template select<12, 1>((v45_i1 * 16)) = v50_data;
              }
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 128> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 12), (0, 8)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 128> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v55_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v56_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v57_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v58_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v59_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v60_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v61_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v62_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v63_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v64_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v65_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v66_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v67_acc{};
              tensorforge::intel_esimd::simd<float, 16> v71_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v67_acc += ((static_cast<float>(v71_data[0])) * v55_data);
              v67_acc += ((static_cast<float>(v71_data[1])) * v56_data);
              v67_acc += ((static_cast<float>(v71_data[2])) * v57_data);
              v67_acc += ((static_cast<float>(v71_data[3])) * v58_data);
              v67_acc += ((static_cast<float>(v71_data[4])) * v59_data);
              v67_acc += ((static_cast<float>(v71_data[5])) * v60_data);
              v67_acc += ((static_cast<float>(v71_data[6])) * v61_data);
              v67_acc += ((static_cast<float>(v71_data[7])) * v62_data);
              v67_acc += ((static_cast<float>(v71_data[8])) * v63_data);
              v67_acc += ((static_cast<float>(v71_data[9])) * v64_data);
              v67_acc += ((static_cast<float>(v71_data[10])) * v65_data);
              v67_acc += ((static_cast<float>(v71_data[11])) * v66_data);
              ir1.template select<16, 1>(0) = v67_acc;
              tensorforge::intel_esimd::simd<float, 16> v96_acc{};
              tensorforge::intel_esimd::simd<float, 16> v98_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              v96_acc += ((static_cast<float>(v98_data[0])) * v55_data);
              v96_acc += ((static_cast<float>(v98_data[1])) * v56_data);
              v96_acc += ((static_cast<float>(v98_data[2])) * v57_data);
              v96_acc += ((static_cast<float>(v98_data[3])) * v58_data);
              v96_acc += ((static_cast<float>(v98_data[4])) * v59_data);
              v96_acc += ((static_cast<float>(v98_data[5])) * v60_data);
              v96_acc += ((static_cast<float>(v98_data[6])) * v61_data);
              v96_acc += ((static_cast<float>(v98_data[7])) * v62_data);
              v96_acc += ((static_cast<float>(v98_data[8])) * v63_data);
              v96_acc += ((static_cast<float>(v98_data[9])) * v64_data);
              v96_acc += ((static_cast<float>(v98_data[10])) * v65_data);
              v96_acc += ((static_cast<float>(v98_data[11])) * v66_data);
              ir1.template select<16, 1>(16) = v96_acc;
              tensorforge::intel_esimd::simd<float, 16> v123_acc{};
              tensorforge::intel_esimd::simd<float, 16> v125_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              v123_acc += ((static_cast<float>(v125_data[0])) * v55_data);
              v123_acc += ((static_cast<float>(v125_data[1])) * v56_data);
              v123_acc += ((static_cast<float>(v125_data[2])) * v57_data);
              v123_acc += ((static_cast<float>(v125_data[3])) * v58_data);
              v123_acc += ((static_cast<float>(v125_data[4])) * v59_data);
              v123_acc += ((static_cast<float>(v125_data[5])) * v60_data);
              v123_acc += ((static_cast<float>(v125_data[6])) * v61_data);
              v123_acc += ((static_cast<float>(v125_data[7])) * v62_data);
              v123_acc += ((static_cast<float>(v125_data[8])) * v63_data);
              v123_acc += ((static_cast<float>(v125_data[9])) * v64_data);
              v123_acc += ((static_cast<float>(v125_data[10])) * v65_data);
              v123_acc += ((static_cast<float>(v125_data[11])) * v66_data);
              ir1.template select<16, 1>(32) = v123_acc;
              tensorforge::intel_esimd::simd<float, 16> v150_acc{};
              tensorforge::intel_esimd::simd<float, 16> v152_data = tensorforge::slmLoad<float, 16>(s0 + (36_i32));
              v150_acc += ((static_cast<float>(v152_data[0])) * v55_data);
              v150_acc += ((static_cast<float>(v152_data[1])) * v56_data);
              v150_acc += ((static_cast<float>(v152_data[2])) * v57_data);
              v150_acc += ((static_cast<float>(v152_data[3])) * v58_data);
              v150_acc += ((static_cast<float>(v152_data[4])) * v59_data);
              v150_acc += ((static_cast<float>(v152_data[5])) * v60_data);
              v150_acc += ((static_cast<float>(v152_data[6])) * v61_data);
              v150_acc += ((static_cast<float>(v152_data[7])) * v62_data);
              v150_acc += ((static_cast<float>(v152_data[8])) * v63_data);
              v150_acc += ((static_cast<float>(v152_data[9])) * v64_data);
              v150_acc += ((static_cast<float>(v152_data[10])) * v65_data);
              v150_acc += ((static_cast<float>(v152_data[11])) * v66_data);
              ir1.template select<16, 1>(48) = v150_acc;
              tensorforge::intel_esimd::simd<float, 16> v177_acc{};
              tensorforge::intel_esimd::simd<float, 16> v179_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v177_acc += ((static_cast<float>(v179_data[0])) * v55_data);
              v177_acc += ((static_cast<float>(v179_data[1])) * v56_data);
              v177_acc += ((static_cast<float>(v179_data[2])) * v57_data);
              v177_acc += ((static_cast<float>(v179_data[3])) * v58_data);
              v177_acc += ((static_cast<float>(v179_data[4])) * v59_data);
              v177_acc += ((static_cast<float>(v179_data[5])) * v60_data);
              v177_acc += ((static_cast<float>(v179_data[6])) * v61_data);
              v177_acc += ((static_cast<float>(v179_data[7])) * v62_data);
              v177_acc += ((static_cast<float>(v179_data[8])) * v63_data);
              v177_acc += ((static_cast<float>(v179_data[9])) * v64_data);
              v177_acc += ((static_cast<float>(v179_data[10])) * v65_data);
              v177_acc += ((static_cast<float>(v179_data[11])) * v66_data);
              ir1.template select<16, 1>(64) = v177_acc;
              tensorforge::intel_esimd::simd<float, 16> v204_acc{};
              tensorforge::intel_esimd::simd<float, 16> v206_data = tensorforge::slmLoad<float, 16>(s0 + (60_i32));
              v204_acc += ((static_cast<float>(v206_data[0])) * v55_data);
              v204_acc += ((static_cast<float>(v206_data[1])) * v56_data);
              v204_acc += ((static_cast<float>(v206_data[2])) * v57_data);
              v204_acc += ((static_cast<float>(v206_data[3])) * v58_data);
              v204_acc += ((static_cast<float>(v206_data[4])) * v59_data);
              v204_acc += ((static_cast<float>(v206_data[5])) * v60_data);
              v204_acc += ((static_cast<float>(v206_data[6])) * v61_data);
              v204_acc += ((static_cast<float>(v206_data[7])) * v62_data);
              v204_acc += ((static_cast<float>(v206_data[8])) * v63_data);
              v204_acc += ((static_cast<float>(v206_data[9])) * v64_data);
              v204_acc += ((static_cast<float>(v206_data[10])) * v65_data);
              v204_acc += ((static_cast<float>(v206_data[11])) * v66_data);
              ir1.template select<16, 1>(80) = v204_acc;
              tensorforge::intel_esimd::simd<float, 16> v231_acc{};
              tensorforge::intel_esimd::simd<float, 16> v233_data = tensorforge::slmLoad<float, 16>(s0 + (72_i32));
              v231_acc += ((static_cast<float>(v233_data[0])) * v55_data);
              v231_acc += ((static_cast<float>(v233_data[1])) * v56_data);
              v231_acc += ((static_cast<float>(v233_data[2])) * v57_data);
              v231_acc += ((static_cast<float>(v233_data[3])) * v58_data);
              v231_acc += ((static_cast<float>(v233_data[4])) * v59_data);
              v231_acc += ((static_cast<float>(v233_data[5])) * v60_data);
              v231_acc += ((static_cast<float>(v233_data[6])) * v61_data);
              v231_acc += ((static_cast<float>(v233_data[7])) * v62_data);
              v231_acc += ((static_cast<float>(v233_data[8])) * v63_data);
              v231_acc += ((static_cast<float>(v233_data[9])) * v64_data);
              v231_acc += ((static_cast<float>(v233_data[10])) * v65_data);
              v231_acc += ((static_cast<float>(v233_data[11])) * v66_data);
              ir1.template select<16, 1>(96) = v231_acc;
              tensorforge::intel_esimd::simd<float, 16> v258_acc{};
              tensorforge::intel_esimd::simd<float, 16> v260_data = tensorforge::slmLoad<float, 16>(s0 + (84_i32));
              v258_acc += ((static_cast<float>(v260_data[0])) * v55_data);
              v258_acc += ((static_cast<float>(v260_data[1])) * v56_data);
              v258_acc += ((static_cast<float>(v260_data[2])) * v57_data);
              v258_acc += ((static_cast<float>(v260_data[3])) * v58_data);
              v258_acc += ((static_cast<float>(v260_data[4])) * v59_data);
              v258_acc += ((static_cast<float>(v260_data[5])) * v60_data);
              v258_acc += ((static_cast<float>(v260_data[6])) * v61_data);
              v258_acc += ((static_cast<float>(v260_data[7])) * v62_data);
              v258_acc += ((static_cast<float>(v260_data[8])) * v63_data);
              v258_acc += ((static_cast<float>(v260_data[9])) * v64_data);
              v258_acc += ((static_cast<float>(v260_data[10])) * v65_data);
              v258_acc += ((static_cast<float>(v260_data[11])) * v66_data);
              ir1.template select<16, 1>(112) = v258_acc;
              #pragma unroll
              for (int32_t v285_n1 = 0; v285_n1 < 8; ++v285_n1) {
                int32_t v286_a = v285_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v288_data(ir1.template select<12, 1>(v286_a));
                r1.template select<12, 1>(v286_a) = v288_data;
              }
              // s1 = load{g>s}(glb_m4[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v289_ld;
              v289_ld.copy_from(glb_m4 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s1 + (0 + 0 + 4 * 0 + 0), v289_ld);
              tensorforge::intel_esimd::simd<float, 32> v290_ld;
              v290_ld.copy_from(glb_m4 + (0 + 0 + 2 * 0 + 64));
              tensorforge::slmStore<float, 32>(s1 + (0 + 0 + 2 * 0 + 64), v290_ld);
              // wait(r2 = load{g>r}(glb_m3););
              tensorforge::intel_esimd::simd<float, 192> r4(0.0f);
              // r4 = load{g>r}(glb_m5);
              #pragma unroll
              for (int32_t v292_i1 = 0; v292_i1 < 12; ++v292_i1) {
                tensorforge::intel_esimd::simd<float, 12> v297_data;
                v297_data.copy_from(glb_m5 + ((v292_i1 * 12)));
                r4.template select<12, 1>((v292_i1 * 16)) = v297_data;
              }
              // wait(s1 = load{g>s}(glb_m4[0, 1]));
              tensorforge::intel_esimd::simd<float, 128> r3(0.0f);
              // r3 = +(r2 * s1) + name: r1, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 128> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v302_data(r2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v303_data(r2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v304_data(r2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v305_data(r2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v306_data(r2.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v307_data(r2.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v308_data(r2.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v309_data(r2.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v310_data(r2.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v311_data(r2.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v312_data(r2.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v313_data(r2.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v314_acc{};
              tensorforge::intel_esimd::simd<float, 16> v318_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v314_acc += ((static_cast<float>(v318_data[0])) * v302_data);
              v314_acc += ((static_cast<float>(v318_data[1])) * v303_data);
              v314_acc += ((static_cast<float>(v318_data[2])) * v304_data);
              v314_acc += ((static_cast<float>(v318_data[3])) * v305_data);
              v314_acc += ((static_cast<float>(v318_data[4])) * v306_data);
              v314_acc += ((static_cast<float>(v318_data[5])) * v307_data);
              v314_acc += ((static_cast<float>(v318_data[6])) * v308_data);
              v314_acc += ((static_cast<float>(v318_data[7])) * v309_data);
              v314_acc += ((static_cast<float>(v318_data[8])) * v310_data);
              v314_acc += ((static_cast<float>(v318_data[9])) * v311_data);
              v314_acc += ((static_cast<float>(v318_data[10])) * v312_data);
              v314_acc += ((static_cast<float>(v318_data[11])) * v313_data);
              ir3.template select<16, 1>(0) = v314_acc;
              tensorforge::intel_esimd::simd<float, 16> v343_acc{};
              tensorforge::intel_esimd::simd<float, 16> v345_data = tensorforge::slmLoad<float, 16>(s1 + (12_i32));
              v343_acc += ((static_cast<float>(v345_data[0])) * v302_data);
              v343_acc += ((static_cast<float>(v345_data[1])) * v303_data);
              v343_acc += ((static_cast<float>(v345_data[2])) * v304_data);
              v343_acc += ((static_cast<float>(v345_data[3])) * v305_data);
              v343_acc += ((static_cast<float>(v345_data[4])) * v306_data);
              v343_acc += ((static_cast<float>(v345_data[5])) * v307_data);
              v343_acc += ((static_cast<float>(v345_data[6])) * v308_data);
              v343_acc += ((static_cast<float>(v345_data[7])) * v309_data);
              v343_acc += ((static_cast<float>(v345_data[8])) * v310_data);
              v343_acc += ((static_cast<float>(v345_data[9])) * v311_data);
              v343_acc += ((static_cast<float>(v345_data[10])) * v312_data);
              v343_acc += ((static_cast<float>(v345_data[11])) * v313_data);
              ir3.template select<16, 1>(16) = v343_acc;
              tensorforge::intel_esimd::simd<float, 16> v370_acc{};
              tensorforge::intel_esimd::simd<float, 16> v372_data = tensorforge::slmLoad<float, 16>(s1 + (24_i32));
              v370_acc += ((static_cast<float>(v372_data[0])) * v302_data);
              v370_acc += ((static_cast<float>(v372_data[1])) * v303_data);
              v370_acc += ((static_cast<float>(v372_data[2])) * v304_data);
              v370_acc += ((static_cast<float>(v372_data[3])) * v305_data);
              v370_acc += ((static_cast<float>(v372_data[4])) * v306_data);
              v370_acc += ((static_cast<float>(v372_data[5])) * v307_data);
              v370_acc += ((static_cast<float>(v372_data[6])) * v308_data);
              v370_acc += ((static_cast<float>(v372_data[7])) * v309_data);
              v370_acc += ((static_cast<float>(v372_data[8])) * v310_data);
              v370_acc += ((static_cast<float>(v372_data[9])) * v311_data);
              v370_acc += ((static_cast<float>(v372_data[10])) * v312_data);
              v370_acc += ((static_cast<float>(v372_data[11])) * v313_data);
              ir3.template select<16, 1>(32) = v370_acc;
              tensorforge::intel_esimd::simd<float, 16> v397_acc{};
              tensorforge::intel_esimd::simd<float, 16> v399_data = tensorforge::slmLoad<float, 16>(s1 + (36_i32));
              v397_acc += ((static_cast<float>(v399_data[0])) * v302_data);
              v397_acc += ((static_cast<float>(v399_data[1])) * v303_data);
              v397_acc += ((static_cast<float>(v399_data[2])) * v304_data);
              v397_acc += ((static_cast<float>(v399_data[3])) * v305_data);
              v397_acc += ((static_cast<float>(v399_data[4])) * v306_data);
              v397_acc += ((static_cast<float>(v399_data[5])) * v307_data);
              v397_acc += ((static_cast<float>(v399_data[6])) * v308_data);
              v397_acc += ((static_cast<float>(v399_data[7])) * v309_data);
              v397_acc += ((static_cast<float>(v399_data[8])) * v310_data);
              v397_acc += ((static_cast<float>(v399_data[9])) * v311_data);
              v397_acc += ((static_cast<float>(v399_data[10])) * v312_data);
              v397_acc += ((static_cast<float>(v399_data[11])) * v313_data);
              ir3.template select<16, 1>(48) = v397_acc;
              tensorforge::intel_esimd::simd<float, 16> v424_acc{};
              tensorforge::intel_esimd::simd<float, 16> v426_data = tensorforge::slmLoad<float, 16>(s1 + (48_i32));
              v424_acc += ((static_cast<float>(v426_data[0])) * v302_data);
              v424_acc += ((static_cast<float>(v426_data[1])) * v303_data);
              v424_acc += ((static_cast<float>(v426_data[2])) * v304_data);
              v424_acc += ((static_cast<float>(v426_data[3])) * v305_data);
              v424_acc += ((static_cast<float>(v426_data[4])) * v306_data);
              v424_acc += ((static_cast<float>(v426_data[5])) * v307_data);
              v424_acc += ((static_cast<float>(v426_data[6])) * v308_data);
              v424_acc += ((static_cast<float>(v426_data[7])) * v309_data);
              v424_acc += ((static_cast<float>(v426_data[8])) * v310_data);
              v424_acc += ((static_cast<float>(v426_data[9])) * v311_data);
              v424_acc += ((static_cast<float>(v426_data[10])) * v312_data);
              v424_acc += ((static_cast<float>(v426_data[11])) * v313_data);
              ir3.template select<16, 1>(64) = v424_acc;
              tensorforge::intel_esimd::simd<float, 16> v451_acc{};
              tensorforge::intel_esimd::simd<float, 16> v453_data = tensorforge::slmLoad<float, 16>(s1 + (60_i32));
              v451_acc += ((static_cast<float>(v453_data[0])) * v302_data);
              v451_acc += ((static_cast<float>(v453_data[1])) * v303_data);
              v451_acc += ((static_cast<float>(v453_data[2])) * v304_data);
              v451_acc += ((static_cast<float>(v453_data[3])) * v305_data);
              v451_acc += ((static_cast<float>(v453_data[4])) * v306_data);
              v451_acc += ((static_cast<float>(v453_data[5])) * v307_data);
              v451_acc += ((static_cast<float>(v453_data[6])) * v308_data);
              v451_acc += ((static_cast<float>(v453_data[7])) * v309_data);
              v451_acc += ((static_cast<float>(v453_data[8])) * v310_data);
              v451_acc += ((static_cast<float>(v453_data[9])) * v311_data);
              v451_acc += ((static_cast<float>(v453_data[10])) * v312_data);
              v451_acc += ((static_cast<float>(v453_data[11])) * v313_data);
              ir3.template select<16, 1>(80) = v451_acc;
              tensorforge::intel_esimd::simd<float, 16> v478_acc{};
              tensorforge::intel_esimd::simd<float, 16> v480_data = tensorforge::slmLoad<float, 16>(s1 + (72_i32));
              v478_acc += ((static_cast<float>(v480_data[0])) * v302_data);
              v478_acc += ((static_cast<float>(v480_data[1])) * v303_data);
              v478_acc += ((static_cast<float>(v480_data[2])) * v304_data);
              v478_acc += ((static_cast<float>(v480_data[3])) * v305_data);
              v478_acc += ((static_cast<float>(v480_data[4])) * v306_data);
              v478_acc += ((static_cast<float>(v480_data[5])) * v307_data);
              v478_acc += ((static_cast<float>(v480_data[6])) * v308_data);
              v478_acc += ((static_cast<float>(v480_data[7])) * v309_data);
              v478_acc += ((static_cast<float>(v480_data[8])) * v310_data);
              v478_acc += ((static_cast<float>(v480_data[9])) * v311_data);
              v478_acc += ((static_cast<float>(v480_data[10])) * v312_data);
              v478_acc += ((static_cast<float>(v480_data[11])) * v313_data);
              ir3.template select<16, 1>(96) = v478_acc;
              tensorforge::intel_esimd::simd<float, 16> v505_acc{};
              tensorforge::intel_esimd::simd<float, 16> v507_data = tensorforge::slmLoad<float, 16>(s1 + (84_i32));
              v505_acc += ((static_cast<float>(v507_data[0])) * v302_data);
              v505_acc += ((static_cast<float>(v507_data[1])) * v303_data);
              v505_acc += ((static_cast<float>(v507_data[2])) * v304_data);
              v505_acc += ((static_cast<float>(v507_data[3])) * v305_data);
              v505_acc += ((static_cast<float>(v507_data[4])) * v306_data);
              v505_acc += ((static_cast<float>(v507_data[5])) * v307_data);
              v505_acc += ((static_cast<float>(v507_data[6])) * v308_data);
              v505_acc += ((static_cast<float>(v507_data[7])) * v309_data);
              v505_acc += ((static_cast<float>(v507_data[8])) * v310_data);
              v505_acc += ((static_cast<float>(v507_data[9])) * v311_data);
              v505_acc += ((static_cast<float>(v507_data[10])) * v312_data);
              v505_acc += ((static_cast<float>(v507_data[11])) * v313_data);
              ir3.template select<16, 1>(112) = v505_acc;
              #pragma unroll
              for (int32_t v532_n1 = 0; v532_n1 < 8; ++v532_n1) {
                int32_t v533_a = v532_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v535_data(ir3.template select<12, 1>(v533_a));
                tensorforge::intel_esimd::simd<float, 12> v536_data(r1.template select<12, 1>(v533_a));
                r3.template select<12, 1>(v533_a) = (v536_data + v535_data);
              }
              // s2 = load{g>s}(glb_m6[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v538_ld;
              v538_ld.copy_from(glb_m6 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s2 + (0 + 0 + 4 * 0 + 0), v538_ld);
              tensorforge::intel_esimd::simd<float, 32> v539_ld;
              v539_ld.copy_from(glb_m6 + (0 + 0 + 2 * 0 + 64));
              tensorforge::slmStore<float, 32>(s2 + (0 + 0 + 2 * 0 + 64), v539_ld);
              // wait(r4 = load{g>r}(glb_m5););
              tensorforge::intel_esimd::simd<float, 192> r6(0.0f);
              // r6 = load{g>r}(glb_m7);
              #pragma unroll
              for (int32_t v541_i1 = 0; v541_i1 < 12; ++v541_i1) {
                tensorforge::intel_esimd::simd<float, 12> v546_data;
                v546_data.copy_from(glb_m7 + ((v541_i1 * 12)));
                r6.template select<12, 1>((v541_i1 * 16)) = v546_data;
              }
              // wait(s2 = load{g>s}(glb_m6[0, 1]));
              tensorforge::intel_esimd::simd<float, 128> r5(0.0f);
              // r5 = +(r4 * s2) + name: r3, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 128> ir5(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v551_data(r4.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v552_data(r4.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v553_data(r4.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v554_data(r4.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v555_data(r4.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v556_data(r4.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v557_data(r4.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v558_data(r4.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v559_data(r4.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v560_data(r4.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v561_data(r4.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v562_data(r4.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v563_acc{};
              tensorforge::intel_esimd::simd<float, 16> v567_data = tensorforge::slmLoad<float, 16>(s2 + (0_i32));
              v563_acc += ((static_cast<float>(v567_data[0])) * v551_data);
              v563_acc += ((static_cast<float>(v567_data[1])) * v552_data);
              v563_acc += ((static_cast<float>(v567_data[2])) * v553_data);
              v563_acc += ((static_cast<float>(v567_data[3])) * v554_data);
              v563_acc += ((static_cast<float>(v567_data[4])) * v555_data);
              v563_acc += ((static_cast<float>(v567_data[5])) * v556_data);
              v563_acc += ((static_cast<float>(v567_data[6])) * v557_data);
              v563_acc += ((static_cast<float>(v567_data[7])) * v558_data);
              v563_acc += ((static_cast<float>(v567_data[8])) * v559_data);
              v563_acc += ((static_cast<float>(v567_data[9])) * v560_data);
              v563_acc += ((static_cast<float>(v567_data[10])) * v561_data);
              v563_acc += ((static_cast<float>(v567_data[11])) * v562_data);
              ir5.template select<16, 1>(0) = v563_acc;
              tensorforge::intel_esimd::simd<float, 16> v592_acc{};
              tensorforge::intel_esimd::simd<float, 16> v594_data = tensorforge::slmLoad<float, 16>(s2 + (12_i32));
              v592_acc += ((static_cast<float>(v594_data[0])) * v551_data);
              v592_acc += ((static_cast<float>(v594_data[1])) * v552_data);
              v592_acc += ((static_cast<float>(v594_data[2])) * v553_data);
              v592_acc += ((static_cast<float>(v594_data[3])) * v554_data);
              v592_acc += ((static_cast<float>(v594_data[4])) * v555_data);
              v592_acc += ((static_cast<float>(v594_data[5])) * v556_data);
              v592_acc += ((static_cast<float>(v594_data[6])) * v557_data);
              v592_acc += ((static_cast<float>(v594_data[7])) * v558_data);
              v592_acc += ((static_cast<float>(v594_data[8])) * v559_data);
              v592_acc += ((static_cast<float>(v594_data[9])) * v560_data);
              v592_acc += ((static_cast<float>(v594_data[10])) * v561_data);
              v592_acc += ((static_cast<float>(v594_data[11])) * v562_data);
              ir5.template select<16, 1>(16) = v592_acc;
              tensorforge::intel_esimd::simd<float, 16> v619_acc{};
              tensorforge::intel_esimd::simd<float, 16> v621_data = tensorforge::slmLoad<float, 16>(s2 + (24_i32));
              v619_acc += ((static_cast<float>(v621_data[0])) * v551_data);
              v619_acc += ((static_cast<float>(v621_data[1])) * v552_data);
              v619_acc += ((static_cast<float>(v621_data[2])) * v553_data);
              v619_acc += ((static_cast<float>(v621_data[3])) * v554_data);
              v619_acc += ((static_cast<float>(v621_data[4])) * v555_data);
              v619_acc += ((static_cast<float>(v621_data[5])) * v556_data);
              v619_acc += ((static_cast<float>(v621_data[6])) * v557_data);
              v619_acc += ((static_cast<float>(v621_data[7])) * v558_data);
              v619_acc += ((static_cast<float>(v621_data[8])) * v559_data);
              v619_acc += ((static_cast<float>(v621_data[9])) * v560_data);
              v619_acc += ((static_cast<float>(v621_data[10])) * v561_data);
              v619_acc += ((static_cast<float>(v621_data[11])) * v562_data);
              ir5.template select<16, 1>(32) = v619_acc;
              tensorforge::intel_esimd::simd<float, 16> v646_acc{};
              tensorforge::intel_esimd::simd<float, 16> v648_data = tensorforge::slmLoad<float, 16>(s2 + (36_i32));
              v646_acc += ((static_cast<float>(v648_data[0])) * v551_data);
              v646_acc += ((static_cast<float>(v648_data[1])) * v552_data);
              v646_acc += ((static_cast<float>(v648_data[2])) * v553_data);
              v646_acc += ((static_cast<float>(v648_data[3])) * v554_data);
              v646_acc += ((static_cast<float>(v648_data[4])) * v555_data);
              v646_acc += ((static_cast<float>(v648_data[5])) * v556_data);
              v646_acc += ((static_cast<float>(v648_data[6])) * v557_data);
              v646_acc += ((static_cast<float>(v648_data[7])) * v558_data);
              v646_acc += ((static_cast<float>(v648_data[8])) * v559_data);
              v646_acc += ((static_cast<float>(v648_data[9])) * v560_data);
              v646_acc += ((static_cast<float>(v648_data[10])) * v561_data);
              v646_acc += ((static_cast<float>(v648_data[11])) * v562_data);
              ir5.template select<16, 1>(48) = v646_acc;
              tensorforge::intel_esimd::simd<float, 16> v673_acc{};
              tensorforge::intel_esimd::simd<float, 16> v675_data = tensorforge::slmLoad<float, 16>(s2 + (48_i32));
              v673_acc += ((static_cast<float>(v675_data[0])) * v551_data);
              v673_acc += ((static_cast<float>(v675_data[1])) * v552_data);
              v673_acc += ((static_cast<float>(v675_data[2])) * v553_data);
              v673_acc += ((static_cast<float>(v675_data[3])) * v554_data);
              v673_acc += ((static_cast<float>(v675_data[4])) * v555_data);
              v673_acc += ((static_cast<float>(v675_data[5])) * v556_data);
              v673_acc += ((static_cast<float>(v675_data[6])) * v557_data);
              v673_acc += ((static_cast<float>(v675_data[7])) * v558_data);
              v673_acc += ((static_cast<float>(v675_data[8])) * v559_data);
              v673_acc += ((static_cast<float>(v675_data[9])) * v560_data);
              v673_acc += ((static_cast<float>(v675_data[10])) * v561_data);
              v673_acc += ((static_cast<float>(v675_data[11])) * v562_data);
              ir5.template select<16, 1>(64) = v673_acc;
              tensorforge::intel_esimd::simd<float, 16> v700_acc{};
              tensorforge::intel_esimd::simd<float, 16> v702_data = tensorforge::slmLoad<float, 16>(s2 + (60_i32));
              v700_acc += ((static_cast<float>(v702_data[0])) * v551_data);
              v700_acc += ((static_cast<float>(v702_data[1])) * v552_data);
              v700_acc += ((static_cast<float>(v702_data[2])) * v553_data);
              v700_acc += ((static_cast<float>(v702_data[3])) * v554_data);
              v700_acc += ((static_cast<float>(v702_data[4])) * v555_data);
              v700_acc += ((static_cast<float>(v702_data[5])) * v556_data);
              v700_acc += ((static_cast<float>(v702_data[6])) * v557_data);
              v700_acc += ((static_cast<float>(v702_data[7])) * v558_data);
              v700_acc += ((static_cast<float>(v702_data[8])) * v559_data);
              v700_acc += ((static_cast<float>(v702_data[9])) * v560_data);
              v700_acc += ((static_cast<float>(v702_data[10])) * v561_data);
              v700_acc += ((static_cast<float>(v702_data[11])) * v562_data);
              ir5.template select<16, 1>(80) = v700_acc;
              tensorforge::intel_esimd::simd<float, 16> v727_acc{};
              tensorforge::intel_esimd::simd<float, 16> v729_data = tensorforge::slmLoad<float, 16>(s2 + (72_i32));
              v727_acc += ((static_cast<float>(v729_data[0])) * v551_data);
              v727_acc += ((static_cast<float>(v729_data[1])) * v552_data);
              v727_acc += ((static_cast<float>(v729_data[2])) * v553_data);
              v727_acc += ((static_cast<float>(v729_data[3])) * v554_data);
              v727_acc += ((static_cast<float>(v729_data[4])) * v555_data);
              v727_acc += ((static_cast<float>(v729_data[5])) * v556_data);
              v727_acc += ((static_cast<float>(v729_data[6])) * v557_data);
              v727_acc += ((static_cast<float>(v729_data[7])) * v558_data);
              v727_acc += ((static_cast<float>(v729_data[8])) * v559_data);
              v727_acc += ((static_cast<float>(v729_data[9])) * v560_data);
              v727_acc += ((static_cast<float>(v729_data[10])) * v561_data);
              v727_acc += ((static_cast<float>(v729_data[11])) * v562_data);
              ir5.template select<16, 1>(96) = v727_acc;
              tensorforge::intel_esimd::simd<float, 16> v754_acc{};
              tensorforge::intel_esimd::simd<float, 16> v756_data = tensorforge::slmLoad<float, 16>(s2 + (84_i32));
              v754_acc += ((static_cast<float>(v756_data[0])) * v551_data);
              v754_acc += ((static_cast<float>(v756_data[1])) * v552_data);
              v754_acc += ((static_cast<float>(v756_data[2])) * v553_data);
              v754_acc += ((static_cast<float>(v756_data[3])) * v554_data);
              v754_acc += ((static_cast<float>(v756_data[4])) * v555_data);
              v754_acc += ((static_cast<float>(v756_data[5])) * v556_data);
              v754_acc += ((static_cast<float>(v756_data[6])) * v557_data);
              v754_acc += ((static_cast<float>(v756_data[7])) * v558_data);
              v754_acc += ((static_cast<float>(v756_data[8])) * v559_data);
              v754_acc += ((static_cast<float>(v756_data[9])) * v560_data);
              v754_acc += ((static_cast<float>(v756_data[10])) * v561_data);
              v754_acc += ((static_cast<float>(v756_data[11])) * v562_data);
              ir5.template select<16, 1>(112) = v754_acc;
              #pragma unroll
              for (int32_t v781_n1 = 0; v781_n1 < 8; ++v781_n1) {
                int32_t v782_a = v781_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v784_data(ir5.template select<12, 1>(v782_a));
                tensorforge::intel_esimd::simd<float, 12> v785_data(r3.template select<12, 1>(v782_a));
                r5.template select<12, 1>(v782_a) = (v785_data + v784_data);
              }
              // s3 = load{g>s}(glb_m8[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v787_ld;
              v787_ld.copy_from(glb_m8 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s3 + (0 + 0 + 4 * 0 + 0), v787_ld);
              tensorforge::intel_esimd::simd<float, 32> v788_ld;
              v788_ld.copy_from(glb_m8 + (0 + 0 + 2 * 0 + 64));
              tensorforge::slmStore<float, 32>(s3 + (0 + 0 + 2 * 0 + 64), v788_ld);
              // wait(r6 = load{g>r}(glb_m7););
              // wait(s3 = load{g>s}(glb_m8[0, 1]));
              tensorforge::intel_esimd::simd<float, 128> r7(0.0f);
              // r7 = +(r6 * s3) + name: r5, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 128> ir7(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v791_data(r6.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v792_data(r6.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v793_data(r6.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v794_data(r6.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v795_data(r6.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v796_data(r6.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v797_data(r6.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v798_data(r6.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v799_data(r6.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v800_data(r6.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v801_data(r6.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v802_data(r6.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v803_acc{};
              tensorforge::intel_esimd::simd<float, 16> v807_data = tensorforge::slmLoad<float, 16>(s3 + (0_i32));
              v803_acc += ((static_cast<float>(v807_data[0])) * v791_data);
              v803_acc += ((static_cast<float>(v807_data[1])) * v792_data);
              v803_acc += ((static_cast<float>(v807_data[2])) * v793_data);
              v803_acc += ((static_cast<float>(v807_data[3])) * v794_data);
              v803_acc += ((static_cast<float>(v807_data[4])) * v795_data);
              v803_acc += ((static_cast<float>(v807_data[5])) * v796_data);
              v803_acc += ((static_cast<float>(v807_data[6])) * v797_data);
              v803_acc += ((static_cast<float>(v807_data[7])) * v798_data);
              v803_acc += ((static_cast<float>(v807_data[8])) * v799_data);
              v803_acc += ((static_cast<float>(v807_data[9])) * v800_data);
              v803_acc += ((static_cast<float>(v807_data[10])) * v801_data);
              v803_acc += ((static_cast<float>(v807_data[11])) * v802_data);
              ir7.template select<16, 1>(0) = v803_acc;
              tensorforge::intel_esimd::simd<float, 16> v832_acc{};
              tensorforge::intel_esimd::simd<float, 16> v834_data = tensorforge::slmLoad<float, 16>(s3 + (12_i32));
              v832_acc += ((static_cast<float>(v834_data[0])) * v791_data);
              v832_acc += ((static_cast<float>(v834_data[1])) * v792_data);
              v832_acc += ((static_cast<float>(v834_data[2])) * v793_data);
              v832_acc += ((static_cast<float>(v834_data[3])) * v794_data);
              v832_acc += ((static_cast<float>(v834_data[4])) * v795_data);
              v832_acc += ((static_cast<float>(v834_data[5])) * v796_data);
              v832_acc += ((static_cast<float>(v834_data[6])) * v797_data);
              v832_acc += ((static_cast<float>(v834_data[7])) * v798_data);
              v832_acc += ((static_cast<float>(v834_data[8])) * v799_data);
              v832_acc += ((static_cast<float>(v834_data[9])) * v800_data);
              v832_acc += ((static_cast<float>(v834_data[10])) * v801_data);
              v832_acc += ((static_cast<float>(v834_data[11])) * v802_data);
              ir7.template select<16, 1>(16) = v832_acc;
              tensorforge::intel_esimd::simd<float, 16> v859_acc{};
              tensorforge::intel_esimd::simd<float, 16> v861_data = tensorforge::slmLoad<float, 16>(s3 + (24_i32));
              v859_acc += ((static_cast<float>(v861_data[0])) * v791_data);
              v859_acc += ((static_cast<float>(v861_data[1])) * v792_data);
              v859_acc += ((static_cast<float>(v861_data[2])) * v793_data);
              v859_acc += ((static_cast<float>(v861_data[3])) * v794_data);
              v859_acc += ((static_cast<float>(v861_data[4])) * v795_data);
              v859_acc += ((static_cast<float>(v861_data[5])) * v796_data);
              v859_acc += ((static_cast<float>(v861_data[6])) * v797_data);
              v859_acc += ((static_cast<float>(v861_data[7])) * v798_data);
              v859_acc += ((static_cast<float>(v861_data[8])) * v799_data);
              v859_acc += ((static_cast<float>(v861_data[9])) * v800_data);
              v859_acc += ((static_cast<float>(v861_data[10])) * v801_data);
              v859_acc += ((static_cast<float>(v861_data[11])) * v802_data);
              ir7.template select<16, 1>(32) = v859_acc;
              tensorforge::intel_esimd::simd<float, 16> v886_acc{};
              tensorforge::intel_esimd::simd<float, 16> v888_data = tensorforge::slmLoad<float, 16>(s3 + (36_i32));
              v886_acc += ((static_cast<float>(v888_data[0])) * v791_data);
              v886_acc += ((static_cast<float>(v888_data[1])) * v792_data);
              v886_acc += ((static_cast<float>(v888_data[2])) * v793_data);
              v886_acc += ((static_cast<float>(v888_data[3])) * v794_data);
              v886_acc += ((static_cast<float>(v888_data[4])) * v795_data);
              v886_acc += ((static_cast<float>(v888_data[5])) * v796_data);
              v886_acc += ((static_cast<float>(v888_data[6])) * v797_data);
              v886_acc += ((static_cast<float>(v888_data[7])) * v798_data);
              v886_acc += ((static_cast<float>(v888_data[8])) * v799_data);
              v886_acc += ((static_cast<float>(v888_data[9])) * v800_data);
              v886_acc += ((static_cast<float>(v888_data[10])) * v801_data);
              v886_acc += ((static_cast<float>(v888_data[11])) * v802_data);
              ir7.template select<16, 1>(48) = v886_acc;
              tensorforge::intel_esimd::simd<float, 16> v913_acc{};
              tensorforge::intel_esimd::simd<float, 16> v915_data = tensorforge::slmLoad<float, 16>(s3 + (48_i32));
              v913_acc += ((static_cast<float>(v915_data[0])) * v791_data);
              v913_acc += ((static_cast<float>(v915_data[1])) * v792_data);
              v913_acc += ((static_cast<float>(v915_data[2])) * v793_data);
              v913_acc += ((static_cast<float>(v915_data[3])) * v794_data);
              v913_acc += ((static_cast<float>(v915_data[4])) * v795_data);
              v913_acc += ((static_cast<float>(v915_data[5])) * v796_data);
              v913_acc += ((static_cast<float>(v915_data[6])) * v797_data);
              v913_acc += ((static_cast<float>(v915_data[7])) * v798_data);
              v913_acc += ((static_cast<float>(v915_data[8])) * v799_data);
              v913_acc += ((static_cast<float>(v915_data[9])) * v800_data);
              v913_acc += ((static_cast<float>(v915_data[10])) * v801_data);
              v913_acc += ((static_cast<float>(v915_data[11])) * v802_data);
              ir7.template select<16, 1>(64) = v913_acc;
              tensorforge::intel_esimd::simd<float, 16> v940_acc{};
              tensorforge::intel_esimd::simd<float, 16> v942_data = tensorforge::slmLoad<float, 16>(s3 + (60_i32));
              v940_acc += ((static_cast<float>(v942_data[0])) * v791_data);
              v940_acc += ((static_cast<float>(v942_data[1])) * v792_data);
              v940_acc += ((static_cast<float>(v942_data[2])) * v793_data);
              v940_acc += ((static_cast<float>(v942_data[3])) * v794_data);
              v940_acc += ((static_cast<float>(v942_data[4])) * v795_data);
              v940_acc += ((static_cast<float>(v942_data[5])) * v796_data);
              v940_acc += ((static_cast<float>(v942_data[6])) * v797_data);
              v940_acc += ((static_cast<float>(v942_data[7])) * v798_data);
              v940_acc += ((static_cast<float>(v942_data[8])) * v799_data);
              v940_acc += ((static_cast<float>(v942_data[9])) * v800_data);
              v940_acc += ((static_cast<float>(v942_data[10])) * v801_data);
              v940_acc += ((static_cast<float>(v942_data[11])) * v802_data);
              ir7.template select<16, 1>(80) = v940_acc;
              tensorforge::intel_esimd::simd<float, 16> v967_acc{};
              tensorforge::intel_esimd::simd<float, 16> v969_data = tensorforge::slmLoad<float, 16>(s3 + (72_i32));
              v967_acc += ((static_cast<float>(v969_data[0])) * v791_data);
              v967_acc += ((static_cast<float>(v969_data[1])) * v792_data);
              v967_acc += ((static_cast<float>(v969_data[2])) * v793_data);
              v967_acc += ((static_cast<float>(v969_data[3])) * v794_data);
              v967_acc += ((static_cast<float>(v969_data[4])) * v795_data);
              v967_acc += ((static_cast<float>(v969_data[5])) * v796_data);
              v967_acc += ((static_cast<float>(v969_data[6])) * v797_data);
              v967_acc += ((static_cast<float>(v969_data[7])) * v798_data);
              v967_acc += ((static_cast<float>(v969_data[8])) * v799_data);
              v967_acc += ((static_cast<float>(v969_data[9])) * v800_data);
              v967_acc += ((static_cast<float>(v969_data[10])) * v801_data);
              v967_acc += ((static_cast<float>(v969_data[11])) * v802_data);
              ir7.template select<16, 1>(96) = v967_acc;
              tensorforge::intel_esimd::simd<float, 16> v994_acc{};
              tensorforge::intel_esimd::simd<float, 16> v996_data = tensorforge::slmLoad<float, 16>(s3 + (84_i32));
              v994_acc += ((static_cast<float>(v996_data[0])) * v791_data);
              v994_acc += ((static_cast<float>(v996_data[1])) * v792_data);
              v994_acc += ((static_cast<float>(v996_data[2])) * v793_data);
              v994_acc += ((static_cast<float>(v996_data[3])) * v794_data);
              v994_acc += ((static_cast<float>(v996_data[4])) * v795_data);
              v994_acc += ((static_cast<float>(v996_data[5])) * v796_data);
              v994_acc += ((static_cast<float>(v996_data[6])) * v797_data);
              v994_acc += ((static_cast<float>(v996_data[7])) * v798_data);
              v994_acc += ((static_cast<float>(v996_data[8])) * v799_data);
              v994_acc += ((static_cast<float>(v996_data[9])) * v800_data);
              v994_acc += ((static_cast<float>(v996_data[10])) * v801_data);
              v994_acc += ((static_cast<float>(v996_data[11])) * v802_data);
              ir7.template select<16, 1>(112) = v994_acc;
              #pragma unroll
              for (int32_t v1021_n1 = 0; v1021_n1 < 8; ++v1021_n1) {
                int32_t v1022_a = v1021_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v1024_data(ir7.template select<12, 1>(v1022_a));
                tensorforge::intel_esimd::simd<float, 12> v1025_data(r5.template select<12, 1>(v1022_a));
                r7.template select<12, 1>(v1022_a) = (v1025_data + v1024_data);
              }
              // glb_m0 = store{r>g}(r7);
              #pragma unroll
              for (int32_t v1027_i1 = 0; v1027_i1 < 8; ++v1027_i1) {
                tensorforge::intel_esimd::simd<float, 12> v1030_data(r7.template select<12, 1>((v1027_i1 * 16)));
                v1030_data.copy_to(glb_m0 + ((v1027_i1 * 12)));
              }
            }
            tensorforge::prefetchRunsL2<576, 384, 576>(&pf_glb_m1[0], &pf_glb_m2[0], &pf_glb_m3[0]);
            tensorforge::prefetchRunsL2<384, 576, 384>(&pf_glb_m4[0], &pf_glb_m5[0], &pf_glb_m6[0]);
            tensorforge::prefetchRunsL2<576, 384>(&pf_glb_m7[0], &pf_glb_m8[0]);
          }
        }
      }
    });
  });
}

