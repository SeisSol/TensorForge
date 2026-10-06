// === base name ===
kernel_4224f7ddbd4e6b9f

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_4224f7ddbd4e6b9f = {{1, 16, 1}, 16, 12, 1, 16, 6144, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_4224f7ddbd4e6b9f(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_4224f7ddbd4e6b9f(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_4224f7ddbd4e6b9f(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 1536 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_4224f7ddbd4e6b9f(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_4224f7ddbd4e6b9f(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_4224f7ddbd4e6b9f(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_4224f7ddbd4e6b9f(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<1536 * sizeof(float)>(); {
        using namespace tensorforge::literals;
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
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1536}],"shared_bytes":6144,"shared_elements":1536,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B","bbox":[[0,0],[12,6]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[6,6]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,6]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,6]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,6]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[6,6]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]},{"addressing":"pointer_based","bbox":[[0,0],[12,6]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (96 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (80);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          tensorforge::SlmPtr<float> s1 = localShrMem0 + (0);
          for (size_t v12_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v12_batchId0 < numElements0; v12_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v13_ahead1 = v12_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v15_batchId1 = (v13_ahead1 < numElements0) ? v13_ahead1 : v12_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v12_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v12_batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v12_batchId0 * 36 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v12_batchId0 * 72 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v12_batchId0 * 144 + 0 + m3_extraOffset];
              tensorforge::intel_esimd::simd<float, 96> r0(0.0f);
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v25_i1 = 0; v25_i1 < 6; ++v25_i1) {
                tensorforge::intel_esimd::simd<float, 12> v30_data;
                v30_data.copy_from(glb_m0 + ((v25_i1 * 12)));
                r0.template select<12, 1>((v25_i1 * 16)) = v30_data;
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v33_ld;
              v33_ld.copy_from(glb_m1 + (0 + 0 + 2 * 0 + 0));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 2 * 0 + 0), v33_ld);
              tensorforge::intel_esimd::simd<float, 4> v34_ld;
              v34_ld.copy_from(glb_m1 + (0 + 0 + 1 * 0 + 32));
              tensorforge::slmStore<float, 4>(s0 + (0 + 0 + 1 * 0 + 32), v34_ld);
              // wait(r0 = load{g>r}(glb_m0););
              tensorforge::intel_esimd::simd<float, 192> r2(0.0f);
              // r2 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v36_i1 = 0; v36_i1 < 12; ++v36_i1) {
                tensorforge::intel_esimd::simd<float, 12> v41_data;
                v41_data.copy_from(glb_m3 + ((v36_i1 * 12)));
                r2.template select<12, 1>((v36_i1 * 16)) = v41_data;
              }
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              tensorforge::intel_esimd::simd<float, 96> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 12), (0, 6)] [(0, 6)]
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v48_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v49_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v50_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v51_acc{};
              tensorforge::intel_esimd::simd<float, 16> v55_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v51_acc += ((static_cast<float>(v55_data[0])) * v45_data);
              v51_acc += ((static_cast<float>(v55_data[1])) * v46_data);
              v51_acc += ((static_cast<float>(v55_data[2])) * v47_data);
              v51_acc += ((static_cast<float>(v55_data[3])) * v48_data);
              v51_acc += ((static_cast<float>(v55_data[4])) * v49_data);
              v51_acc += ((static_cast<float>(v55_data[5])) * v50_data);
              r1.template select<16, 1>(0) = v51_acc;
              tensorforge::intel_esimd::simd<float, 16> v68_acc{};
              tensorforge::intel_esimd::simd<float, 16> v70_data = tensorforge::slmLoad<float, 16>(s0 + (6_i32));
              v68_acc += ((static_cast<float>(v70_data[0])) * v45_data);
              v68_acc += ((static_cast<float>(v70_data[1])) * v46_data);
              v68_acc += ((static_cast<float>(v70_data[2])) * v47_data);
              v68_acc += ((static_cast<float>(v70_data[3])) * v48_data);
              v68_acc += ((static_cast<float>(v70_data[4])) * v49_data);
              v68_acc += ((static_cast<float>(v70_data[5])) * v50_data);
              r1.template select<16, 1>(16) = v68_acc;
              tensorforge::intel_esimd::simd<float, 16> v83_acc{};
              tensorforge::intel_esimd::simd<float, 16> v85_data = tensorforge::slmLoad<float, 16>(s0 + (12_i32));
              v83_acc += ((static_cast<float>(v85_data[0])) * v45_data);
              v83_acc += ((static_cast<float>(v85_data[1])) * v46_data);
              v83_acc += ((static_cast<float>(v85_data[2])) * v47_data);
              v83_acc += ((static_cast<float>(v85_data[3])) * v48_data);
              v83_acc += ((static_cast<float>(v85_data[4])) * v49_data);
              v83_acc += ((static_cast<float>(v85_data[5])) * v50_data);
              r1.template select<16, 1>(32) = v83_acc;
              tensorforge::intel_esimd::simd<float, 16> v98_acc{};
              tensorforge::intel_esimd::simd<float, 16> v100_data = tensorforge::slmLoad<float, 16>(s0 + (18_i32));
              v98_acc += ((static_cast<float>(v100_data[0])) * v45_data);
              v98_acc += ((static_cast<float>(v100_data[1])) * v46_data);
              v98_acc += ((static_cast<float>(v100_data[2])) * v47_data);
              v98_acc += ((static_cast<float>(v100_data[3])) * v48_data);
              v98_acc += ((static_cast<float>(v100_data[4])) * v49_data);
              v98_acc += ((static_cast<float>(v100_data[5])) * v50_data);
              r1.template select<16, 1>(48) = v98_acc;
              tensorforge::intel_esimd::simd<float, 16> v113_acc{};
              tensorforge::intel_esimd::simd<float, 16> v115_data = tensorforge::slmLoad<float, 16>(s0 + (24_i32));
              v113_acc += ((static_cast<float>(v115_data[0])) * v45_data);
              v113_acc += ((static_cast<float>(v115_data[1])) * v46_data);
              v113_acc += ((static_cast<float>(v115_data[2])) * v47_data);
              v113_acc += ((static_cast<float>(v115_data[3])) * v48_data);
              v113_acc += ((static_cast<float>(v115_data[4])) * v49_data);
              v113_acc += ((static_cast<float>(v115_data[5])) * v50_data);
              r1.template select<16, 1>(64) = v113_acc;
              tensorforge::intel_esimd::simd<float, 16> v128_acc{};
              tensorforge::intel_esimd::simd<float, 16> v130_data = tensorforge::slmLoad<float, 16>(s0 + (30_i32));
              v128_acc += ((static_cast<float>(v130_data[0])) * v45_data);
              v128_acc += ((static_cast<float>(v130_data[1])) * v46_data);
              v128_acc += ((static_cast<float>(v130_data[2])) * v47_data);
              v128_acc += ((static_cast<float>(v130_data[3])) * v48_data);
              v128_acc += ((static_cast<float>(v130_data[4])) * v49_data);
              v128_acc += ((static_cast<float>(v130_data[5])) * v50_data);
              r1.template select<16, 1>(80) = v128_acc;
              // wait(r2 = load{g>r}(glb_m3););
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v143_i1 = 0; v143_i1 < 6; ++v143_i1) {
                tensorforge::intel_esimd::simd<float, 12> v146_data(r1.template select<12, 1>((v143_i1 * 16)));
                tensorforge::slmStore<float, 12>(s1 + ((v143_i1 * 12)), v146_data);
              }
              tensorforge::intel_esimd::simd<float, 96> r3(0.0f);
              // ir3 = +(r2 * s1)
              // [(0, 12), (0, 6)] [(0, 12)]
              tensorforge::intel_esimd::simd<float, 96> ir3(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v153_data(r2.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v154_data(r2.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v155_data(r2.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v156_data(r2.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v157_data(r2.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v158_data(r2.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v159_data(r2.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v160_data(r2.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v161_data(r2.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v162_data(r2.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v163_data(r2.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v164_data(r2.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v165_acc{};
              tensorforge::intel_esimd::simd<float, 16> v169_data = tensorforge::slmLoad<float, 16>(s1 + (0_i32));
              v165_acc += ((static_cast<float>(v169_data[0])) * v153_data);
              v165_acc += ((static_cast<float>(v169_data[1])) * v154_data);
              v165_acc += ((static_cast<float>(v169_data[2])) * v155_data);
              v165_acc += ((static_cast<float>(v169_data[3])) * v156_data);
              v165_acc += ((static_cast<float>(v169_data[4])) * v157_data);
              v165_acc += ((static_cast<float>(v169_data[5])) * v158_data);
              v165_acc += ((static_cast<float>(v169_data[6])) * v159_data);
              v165_acc += ((static_cast<float>(v169_data[7])) * v160_data);
              v165_acc += ((static_cast<float>(v169_data[8])) * v161_data);
              v165_acc += ((static_cast<float>(v169_data[9])) * v162_data);
              v165_acc += ((static_cast<float>(v169_data[10])) * v163_data);
              v165_acc += ((static_cast<float>(v169_data[11])) * v164_data);
              ir3.template select<16, 1>(0) = v165_acc;
              tensorforge::intel_esimd::simd<float, 16> v194_acc{};
              tensorforge::intel_esimd::simd<float, 16> v196_data = tensorforge::slmLoad<float, 16>(s1 + (12_i32));
              v194_acc += ((static_cast<float>(v196_data[0])) * v153_data);
              v194_acc += ((static_cast<float>(v196_data[1])) * v154_data);
              v194_acc += ((static_cast<float>(v196_data[2])) * v155_data);
              v194_acc += ((static_cast<float>(v196_data[3])) * v156_data);
              v194_acc += ((static_cast<float>(v196_data[4])) * v157_data);
              v194_acc += ((static_cast<float>(v196_data[5])) * v158_data);
              v194_acc += ((static_cast<float>(v196_data[6])) * v159_data);
              v194_acc += ((static_cast<float>(v196_data[7])) * v160_data);
              v194_acc += ((static_cast<float>(v196_data[8])) * v161_data);
              v194_acc += ((static_cast<float>(v196_data[9])) * v162_data);
              v194_acc += ((static_cast<float>(v196_data[10])) * v163_data);
              v194_acc += ((static_cast<float>(v196_data[11])) * v164_data);
              ir3.template select<16, 1>(16) = v194_acc;
              tensorforge::intel_esimd::simd<float, 16> v221_acc{};
              tensorforge::intel_esimd::simd<float, 16> v223_data = tensorforge::slmLoad<float, 16>(s1 + (24_i32));
              v221_acc += ((static_cast<float>(v223_data[0])) * v153_data);
              v221_acc += ((static_cast<float>(v223_data[1])) * v154_data);
              v221_acc += ((static_cast<float>(v223_data[2])) * v155_data);
              v221_acc += ((static_cast<float>(v223_data[3])) * v156_data);
              v221_acc += ((static_cast<float>(v223_data[4])) * v157_data);
              v221_acc += ((static_cast<float>(v223_data[5])) * v158_data);
              v221_acc += ((static_cast<float>(v223_data[6])) * v159_data);
              v221_acc += ((static_cast<float>(v223_data[7])) * v160_data);
              v221_acc += ((static_cast<float>(v223_data[8])) * v161_data);
              v221_acc += ((static_cast<float>(v223_data[9])) * v162_data);
              v221_acc += ((static_cast<float>(v223_data[10])) * v163_data);
              v221_acc += ((static_cast<float>(v223_data[11])) * v164_data);
              ir3.template select<16, 1>(32) = v221_acc;
              tensorforge::intel_esimd::simd<float, 16> v248_acc{};
              tensorforge::intel_esimd::simd<float, 16> v250_data = tensorforge::slmLoad<float, 16>(s1 + (36_i32));
              v248_acc += ((static_cast<float>(v250_data[0])) * v153_data);
              v248_acc += ((static_cast<float>(v250_data[1])) * v154_data);
              v248_acc += ((static_cast<float>(v250_data[2])) * v155_data);
              v248_acc += ((static_cast<float>(v250_data[3])) * v156_data);
              v248_acc += ((static_cast<float>(v250_data[4])) * v157_data);
              v248_acc += ((static_cast<float>(v250_data[5])) * v158_data);
              v248_acc += ((static_cast<float>(v250_data[6])) * v159_data);
              v248_acc += ((static_cast<float>(v250_data[7])) * v160_data);
              v248_acc += ((static_cast<float>(v250_data[8])) * v161_data);
              v248_acc += ((static_cast<float>(v250_data[9])) * v162_data);
              v248_acc += ((static_cast<float>(v250_data[10])) * v163_data);
              v248_acc += ((static_cast<float>(v250_data[11])) * v164_data);
              ir3.template select<16, 1>(48) = v248_acc;
              tensorforge::intel_esimd::simd<float, 16> v275_acc{};
              tensorforge::intel_esimd::simd<float, 16> v277_data = tensorforge::slmLoad<float, 16>(s1 + (48_i32));
              v275_acc += ((static_cast<float>(v277_data[0])) * v153_data);
              v275_acc += ((static_cast<float>(v277_data[1])) * v154_data);
              v275_acc += ((static_cast<float>(v277_data[2])) * v155_data);
              v275_acc += ((static_cast<float>(v277_data[3])) * v156_data);
              v275_acc += ((static_cast<float>(v277_data[4])) * v157_data);
              v275_acc += ((static_cast<float>(v277_data[5])) * v158_data);
              v275_acc += ((static_cast<float>(v277_data[6])) * v159_data);
              v275_acc += ((static_cast<float>(v277_data[7])) * v160_data);
              v275_acc += ((static_cast<float>(v277_data[8])) * v161_data);
              v275_acc += ((static_cast<float>(v277_data[9])) * v162_data);
              v275_acc += ((static_cast<float>(v277_data[10])) * v163_data);
              v275_acc += ((static_cast<float>(v277_data[11])) * v164_data);
              ir3.template select<16, 1>(64) = v275_acc;
              tensorforge::intel_esimd::simd<float, 16> v302_acc{};
              tensorforge::intel_esimd::simd<float, 16> v304_data = tensorforge::slmLoad<float, 16>(s1 + (60_i32));
              v302_acc += ((static_cast<float>(v304_data[0])) * v153_data);
              v302_acc += ((static_cast<float>(v304_data[1])) * v154_data);
              v302_acc += ((static_cast<float>(v304_data[2])) * v155_data);
              v302_acc += ((static_cast<float>(v304_data[3])) * v156_data);
              v302_acc += ((static_cast<float>(v304_data[4])) * v157_data);
              v302_acc += ((static_cast<float>(v304_data[5])) * v158_data);
              v302_acc += ((static_cast<float>(v304_data[6])) * v159_data);
              v302_acc += ((static_cast<float>(v304_data[7])) * v160_data);
              v302_acc += ((static_cast<float>(v304_data[8])) * v161_data);
              v302_acc += ((static_cast<float>(v304_data[9])) * v162_data);
              v302_acc += ((static_cast<float>(v304_data[10])) * v163_data);
              v302_acc += ((static_cast<float>(v304_data[11])) * v164_data);
              ir3.template select<16, 1>(80) = v302_acc;
              // r3 = ir3
              #pragma unroll
              for (int32_t v329_n1 = 0; v329_n1 < 6; ++v329_n1) {
                int32_t v330_a = v329_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v332_data(ir3.template select<12, 1>(v330_a));
                r3.template select<12, 1>(v330_a) = v332_data;
              }
              // glb_m2 = store{r>g}(r3);
              #pragma unroll
              for (int32_t v333_i1 = 0; v333_i1 < 6; ++v333_i1) {
                tensorforge::intel_esimd::simd<float, 12> v336_data(r3.template select<12, 1>((v333_i1 * 16)));
                v336_data.copy_to(glb_m2 + ((v333_i1 * 12)));
              }
            }
          }
        }
      }
    });
  });
}

