// === base name ===
kernel_5f8c9aaa272d509b

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_5f8c9aaa272d509b = {{1, 16, 1}, 16, 12, 1, 16, 6144, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_5f8c9aaa272d509b(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_5f8c9aaa272d509b(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_5f8c9aaa272d509b(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 1536 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_5f8c9aaa272d509b(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_5f8c9aaa272d509b(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_5f8c9aaa272d509b(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_5f8c9aaa272d509b(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<1536 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 6144 B shared, occupancy grid
        // operands:
        //   m0 32×32(12×6) {0..12}×{0..6} strided
        //   m1 32×32(6×6) {0..6}×{0..6} strided
        //   m2 32×32(12×6) {0..12}×{0..6} strided
        //   m3 32×32(12×12) {0..12}×{0..12} strided
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   m2[i,j] = m3[i,k] × t0[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1536}],"shared_bytes":6144,"shared_elements":1536,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B","bbox":[[0,0],[12,6]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[6,6]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,6]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,6]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,6]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[6,6]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]},{"addressing":"pointer_based","bbox":[[0,0],[12,6]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (96 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (80);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v6_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v6_batchId0 < numElements0; v6_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v7_ahead1 = v6_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v9_batchId1 = (v7_ahead1 < numElements0) ? v7_ahead1 : v6_batchId0;
            const float *const __restrict__ pf_glb_m0 = &m0[v9_batchId1 * 72 + 0 + m0_extraOffset];
            const float *const __restrict__ pf_glb_m1 = &m1[v9_batchId1 * 36 + 0 + m1_extraOffset];
            const float *const __restrict__ pf_glb_m3 = &m3[v9_batchId1 * 144 + 0 + m3_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v6_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v6_batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v6_batchId0 * 36 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v6_batchId0 * 72 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v6_batchId0 * 144 + 0 + m3_extraOffset];
              tensorforge::intel_esimd::simd<float, 96> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v22_i1 = 0; v22_i1 < 6; ++v22_i1) {
                tensorforge::intel_esimd::simd<float, 12> v27_data;
                v27_data.copy_from(glb_m0 + ((v22_i1 * 12)));
                r0.template select<12, 1>((v22_i1 * 16)) = v27_data;
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v30_ld;
              v30_ld.copy_from(glb_m1 + (0 + 0 + 2 * 0 + 0));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 2 * 0 + 0), v30_ld);
              tensorforge::intel_esimd::simd<float, 4> v31_ld;
              v31_ld.copy_from(glb_m1 + (0 + 0 + 1 * 0 + 32));
              tensorforge::slmStore<float, 4>(s0 + (0 + 0 + 1 * 0 + 32), v31_ld);
              // wait(r0 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<float, 192> r2(0.0f);
              // r2 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v33_i1 = 0; v33_i1 < 12; ++v33_i1) {
                tensorforge::intel_esimd::simd<float, 12> v38_data;
                v38_data.copy_from(glb_m3 + ((v33_i1 * 12)));
                r2.template select<12, 1>((v33_i1 * 16)) = v38_data;
              }
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 96> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 12), (0, 6)] [(0, 6)]
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v48_acc{};
              tensorforge::intel_esimd::simd<float, 16> v52_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v48_acc += ((static_cast<float>(v52_data[0])) * v42_data);
              v48_acc += ((static_cast<float>(v52_data[1])) * v43_data);
              v48_acc += ((static_cast<float>(v52_data[2])) * v44_data);
              v48_acc += ((static_cast<float>(v52_data[3])) * v45_data);
              v48_acc += ((static_cast<float>(v52_data[4])) * v46_data);
              v48_acc += ((static_cast<float>(v52_data[5])) * v47_data);
              r1.template select<16, 1>(0) = v48_acc;
              tensorforge::intel_esimd::simd<float, 16> v65_acc{};
              tensorforge::intel_esimd::simd<float, 16> v67_data = tensorforge::slmLoad<float, 16>(s0 + (6_i32));
              v65_acc += ((static_cast<float>(v67_data[0])) * v42_data);
              v65_acc += ((static_cast<float>(v67_data[1])) * v43_data);
              v65_acc += ((static_cast<float>(v67_data[2])) * v44_data);
              v65_acc += ((static_cast<float>(v67_data[3])) * v45_data);
              v65_acc += ((static_cast<float>(v67_data[4])) * v46_data);
              v65_acc += ((static_cast<float>(v67_data[5])) * v47_data);
              r1.template select<16, 1>(16) = v65_acc;
              tensorforge::intel_esimd::simd<float, 16> v80_acc{};
              tensorforge::intel_esimd::simd<float, 16> v82_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              v80_acc += ((static_cast<float>(v82_data[0])) * v42_data);
              v80_acc += ((static_cast<float>(v82_data[1])) * v43_data);
              v80_acc += ((static_cast<float>(v82_data[2])) * v44_data);
              v80_acc += ((static_cast<float>(v82_data[3])) * v45_data);
              v80_acc += ((static_cast<float>(v82_data[4])) * v46_data);
              v80_acc += ((static_cast<float>(v82_data[5])) * v47_data);
              r1.template select<16, 1>(32) = v80_acc;
              tensorforge::intel_esimd::simd<float, 16> v95_acc{};
              tensorforge::intel_esimd::simd<float, 16> v97_data = tensorforge::slmLoad<float, 16>(s0 + (18_i32));
              v95_acc += ((static_cast<float>(v97_data[0])) * v42_data);
              v95_acc += ((static_cast<float>(v97_data[1])) * v43_data);
              v95_acc += ((static_cast<float>(v97_data[2])) * v44_data);
              v95_acc += ((static_cast<float>(v97_data[3])) * v45_data);
              v95_acc += ((static_cast<float>(v97_data[4])) * v46_data);
              v95_acc += ((static_cast<float>(v97_data[5])) * v47_data);
              r1.template select<16, 1>(48) = v95_acc;
              tensorforge::intel_esimd::simd<float, 16> v110_acc{};
              tensorforge::intel_esimd::simd<float, 16> v112_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              v110_acc += ((static_cast<float>(v112_data[0])) * v42_data);
              v110_acc += ((static_cast<float>(v112_data[1])) * v43_data);
              v110_acc += ((static_cast<float>(v112_data[2])) * v44_data);
              v110_acc += ((static_cast<float>(v112_data[3])) * v45_data);
              v110_acc += ((static_cast<float>(v112_data[4])) * v46_data);
              v110_acc += ((static_cast<float>(v112_data[5])) * v47_data);
              r1.template select<16, 1>(64) = v110_acc;
              tensorforge::intel_esimd::simd<float, 16> v125_acc{};
              tensorforge::intel_esimd::simd<float, 16> v127_data = tensorforge::slmLoad<float, 16>(s0 + (30_i32));
              v125_acc += ((static_cast<float>(v127_data[0])) * v42_data);
              v125_acc += ((static_cast<float>(v127_data[1])) * v43_data);
              v125_acc += ((static_cast<float>(v127_data[2])) * v44_data);
              v125_acc += ((static_cast<float>(v127_data[3])) * v45_data);
              v125_acc += ((static_cast<float>(v127_data[4])) * v46_data);
              v125_acc += ((static_cast<float>(v127_data[5])) * v47_data);
              r1.template select<16, 1>(80) = v125_acc;
              // wait(r2 = load{g>r}(glb_m3););
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v140_i1 = 0; v140_i1 < 6; ++v140_i1) {
                tensorforge::intel_esimd::simd<float, 12> v143_data(r1.template select<12, 1>((v140_i1 * 16)));
                tensorforge::slmStore<float, 12>(s1 + ((v140_i1 * 12)), v143_data);
              }
              tensorforge::intel_esimd::simd<float, 96> r3(0.0f);
              // r3 = +(r2 * s1) + None
              // [(0, 12), (0, 6)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 96> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v150_data(r2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v151_data(r2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v152_data(r2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v153_data(r2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v154_data(r2.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v155_data(r2.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v156_data(r2.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v157_data(r2.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v158_data(r2.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v159_data(r2.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v160_data(r2.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v161_data(r2.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v162_acc{};
              tensorforge::intel_esimd::simd<float, 16> v166_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v162_acc += ((static_cast<float>(v166_data[0])) * v150_data);
              v162_acc += ((static_cast<float>(v166_data[1])) * v151_data);
              v162_acc += ((static_cast<float>(v166_data[2])) * v152_data);
              v162_acc += ((static_cast<float>(v166_data[3])) * v153_data);
              v162_acc += ((static_cast<float>(v166_data[4])) * v154_data);
              v162_acc += ((static_cast<float>(v166_data[5])) * v155_data);
              v162_acc += ((static_cast<float>(v166_data[6])) * v156_data);
              v162_acc += ((static_cast<float>(v166_data[7])) * v157_data);
              v162_acc += ((static_cast<float>(v166_data[8])) * v158_data);
              v162_acc += ((static_cast<float>(v166_data[9])) * v159_data);
              v162_acc += ((static_cast<float>(v166_data[10])) * v160_data);
              v162_acc += ((static_cast<float>(v166_data[11])) * v161_data);
              ir3.template select<16, 1>(0) = v162_acc;
              tensorforge::intel_esimd::simd<float, 16> v191_acc{};
              tensorforge::intel_esimd::simd<float, 16> v193_data = tensorforge::slmLoad<float, 16>(s1 + (12_i32));
              v191_acc += ((static_cast<float>(v193_data[0])) * v150_data);
              v191_acc += ((static_cast<float>(v193_data[1])) * v151_data);
              v191_acc += ((static_cast<float>(v193_data[2])) * v152_data);
              v191_acc += ((static_cast<float>(v193_data[3])) * v153_data);
              v191_acc += ((static_cast<float>(v193_data[4])) * v154_data);
              v191_acc += ((static_cast<float>(v193_data[5])) * v155_data);
              v191_acc += ((static_cast<float>(v193_data[6])) * v156_data);
              v191_acc += ((static_cast<float>(v193_data[7])) * v157_data);
              v191_acc += ((static_cast<float>(v193_data[8])) * v158_data);
              v191_acc += ((static_cast<float>(v193_data[9])) * v159_data);
              v191_acc += ((static_cast<float>(v193_data[10])) * v160_data);
              v191_acc += ((static_cast<float>(v193_data[11])) * v161_data);
              ir3.template select<16, 1>(16) = v191_acc;
              tensorforge::intel_esimd::simd<float, 16> v218_acc{};
              tensorforge::intel_esimd::simd<float, 16> v220_data = tensorforge::slmLoad<float, 16>(s1 + (24_i32));
              v218_acc += ((static_cast<float>(v220_data[0])) * v150_data);
              v218_acc += ((static_cast<float>(v220_data[1])) * v151_data);
              v218_acc += ((static_cast<float>(v220_data[2])) * v152_data);
              v218_acc += ((static_cast<float>(v220_data[3])) * v153_data);
              v218_acc += ((static_cast<float>(v220_data[4])) * v154_data);
              v218_acc += ((static_cast<float>(v220_data[5])) * v155_data);
              v218_acc += ((static_cast<float>(v220_data[6])) * v156_data);
              v218_acc += ((static_cast<float>(v220_data[7])) * v157_data);
              v218_acc += ((static_cast<float>(v220_data[8])) * v158_data);
              v218_acc += ((static_cast<float>(v220_data[9])) * v159_data);
              v218_acc += ((static_cast<float>(v220_data[10])) * v160_data);
              v218_acc += ((static_cast<float>(v220_data[11])) * v161_data);
              ir3.template select<16, 1>(32) = v218_acc;
              tensorforge::intel_esimd::simd<float, 16> v245_acc{};
              tensorforge::intel_esimd::simd<float, 16> v247_data = tensorforge::slmLoad<float, 16>(s1 + (36_i32));
              v245_acc += ((static_cast<float>(v247_data[0])) * v150_data);
              v245_acc += ((static_cast<float>(v247_data[1])) * v151_data);
              v245_acc += ((static_cast<float>(v247_data[2])) * v152_data);
              v245_acc += ((static_cast<float>(v247_data[3])) * v153_data);
              v245_acc += ((static_cast<float>(v247_data[4])) * v154_data);
              v245_acc += ((static_cast<float>(v247_data[5])) * v155_data);
              v245_acc += ((static_cast<float>(v247_data[6])) * v156_data);
              v245_acc += ((static_cast<float>(v247_data[7])) * v157_data);
              v245_acc += ((static_cast<float>(v247_data[8])) * v158_data);
              v245_acc += ((static_cast<float>(v247_data[9])) * v159_data);
              v245_acc += ((static_cast<float>(v247_data[10])) * v160_data);
              v245_acc += ((static_cast<float>(v247_data[11])) * v161_data);
              ir3.template select<16, 1>(48) = v245_acc;
              tensorforge::intel_esimd::simd<float, 16> v272_acc{};
              tensorforge::intel_esimd::simd<float, 16> v274_data = tensorforge::slmLoad<float, 16>(s1 + (48_i32));
              v272_acc += ((static_cast<float>(v274_data[0])) * v150_data);
              v272_acc += ((static_cast<float>(v274_data[1])) * v151_data);
              v272_acc += ((static_cast<float>(v274_data[2])) * v152_data);
              v272_acc += ((static_cast<float>(v274_data[3])) * v153_data);
              v272_acc += ((static_cast<float>(v274_data[4])) * v154_data);
              v272_acc += ((static_cast<float>(v274_data[5])) * v155_data);
              v272_acc += ((static_cast<float>(v274_data[6])) * v156_data);
              v272_acc += ((static_cast<float>(v274_data[7])) * v157_data);
              v272_acc += ((static_cast<float>(v274_data[8])) * v158_data);
              v272_acc += ((static_cast<float>(v274_data[9])) * v159_data);
              v272_acc += ((static_cast<float>(v274_data[10])) * v160_data);
              v272_acc += ((static_cast<float>(v274_data[11])) * v161_data);
              ir3.template select<16, 1>(64) = v272_acc;
              tensorforge::intel_esimd::simd<float, 16> v299_acc{};
              tensorforge::intel_esimd::simd<float, 16> v301_data = tensorforge::slmLoad<float, 16>(s1 + (60_i32));
              v299_acc += ((static_cast<float>(v301_data[0])) * v150_data);
              v299_acc += ((static_cast<float>(v301_data[1])) * v151_data);
              v299_acc += ((static_cast<float>(v301_data[2])) * v152_data);
              v299_acc += ((static_cast<float>(v301_data[3])) * v153_data);
              v299_acc += ((static_cast<float>(v301_data[4])) * v154_data);
              v299_acc += ((static_cast<float>(v301_data[5])) * v155_data);
              v299_acc += ((static_cast<float>(v301_data[6])) * v156_data);
              v299_acc += ((static_cast<float>(v301_data[7])) * v157_data);
              v299_acc += ((static_cast<float>(v301_data[8])) * v158_data);
              v299_acc += ((static_cast<float>(v301_data[9])) * v159_data);
              v299_acc += ((static_cast<float>(v301_data[10])) * v160_data);
              v299_acc += ((static_cast<float>(v301_data[11])) * v161_data);
              ir3.template select<16, 1>(80) = v299_acc;
              #pragma unroll
              for (int32_t v326_n1 = 0; v326_n1 < 6; ++v326_n1) {
                int32_t v327_a = v326_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v329_data(ir3.template select<12, 1>(v327_a));
                r3.template select<12, 1>(v327_a) = v329_data;
              }
              // glb_m2 = store{r>g}(r3);
              #pragma unroll
              for (int32_t v330_i1 = 0; v330_i1 < 6; ++v330_i1) {
                tensorforge::intel_esimd::simd<float, 12> v333_data(r3.template select<12, 1>((v330_i1 * 16)));
                v333_data.copy_to(glb_m2 + ((v330_i1 * 12)));
              }
            }
            tensorforge::prefetchRunsL2<288, 144, 576>(&pf_glb_m0[0], &pf_glb_m1[0], &pf_glb_m3[0]);
          }
        }
      }
    });
  });
}

