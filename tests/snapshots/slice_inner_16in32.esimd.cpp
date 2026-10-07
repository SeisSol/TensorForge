// === base name ===
kernel_cea1d14c4310254c

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_cea1d14c4310254c = {{1, 16, 1}, 16, 16, 1, 16, 9216, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_cea1d14c4310254c(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_cea1d14c4310254c(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_cea1d14c4310254c(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_cea1d14c4310254c(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_cea1d14c4310254c(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_cea1d14c4310254c(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_cea1d14c4310254c(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2304 * sizeof(float)>(); {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 9216 B shared, occupancy grid
        // operands:
        //   m0 16×8(16×8) {0..16}×{0..8} strided
        //   m1 32×32(32×32) {0..32}×{0..32} strided
        //   m2 16×8(16×8) {0..16}×{0..8} strided
        // operations:
        //   m0[i,j] = m1[i,k]@{8..24}×{8..24} × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2304}],"shared_bytes":9216,"shared_elements":2304,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,8]],"name":"m0","ordered":false,"parts":1,"shape":[16,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[32,32]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[8,8],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (144 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 128 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 1024 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 128 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v20_i0 = 0; v20_i0 < 1; ++v20_i0) {
                int32_t v22_lead = v20_i0 * 16;
                int32_t v24_off = v22_lead + 8;
                #pragma unroll
                for (int32_t v21_i1 = 8; v21_i1 < 24; ++v21_i1) {
                  tensorforge::intel_esimd::simd<float, 16> v27_data;
                  v27_data.copy_from(glb_m1 + ((v24_off + (v21_i1 * 32))));
                  r0.template select<16, 1>((v22_lead + ((v21_i1 - 8) * 16))) = v27_data;
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v31_ld;
              v31_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v31_ld);
              tensorforge::intel_esimd::simd<float, 64> v32_ld;
              v32_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v32_ld);
              tensorforge::intel_esimd::simd<float, 128> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 16), (0, 8)] [(0, 16)]
              tensorforge::intel_esimd::simd<float, 128> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v35_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v36_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v37_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v38_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v39_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v40_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(192));
              tensorforge::intel_esimd::simd<float, 16> v48_data(r0.template select<16, 1>(208));
              tensorforge::intel_esimd::simd<float, 16> v49_data(r0.template select<16, 1>(224));
              tensorforge::intel_esimd::simd<float, 16> v50_data(r0.template select<16, 1>(240));
              tensorforge::intel_esimd::simd<float, 16> v51_acc{};
              tensorforge::intel_esimd::simd<float, 16> v55_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v51_acc += ((static_cast<float>(v55_data[0])) * v35_data);
              v51_acc += ((static_cast<float>(v55_data[1])) * v36_data);
              v51_acc += ((static_cast<float>(v55_data[2])) * v37_data);
              v51_acc += ((static_cast<float>(v55_data[3])) * v38_data);
              v51_acc += ((static_cast<float>(v55_data[4])) * v39_data);
              v51_acc += ((static_cast<float>(v55_data[5])) * v40_data);
              v51_acc += ((static_cast<float>(v55_data[6])) * v41_data);
              v51_acc += ((static_cast<float>(v55_data[7])) * v42_data);
              v51_acc += ((static_cast<float>(v55_data[8])) * v43_data);
              v51_acc += ((static_cast<float>(v55_data[9])) * v44_data);
              v51_acc += ((static_cast<float>(v55_data[10])) * v45_data);
              v51_acc += ((static_cast<float>(v55_data[11])) * v46_data);
              v51_acc += ((static_cast<float>(v55_data[12])) * v47_data);
              v51_acc += ((static_cast<float>(v55_data[13])) * v48_data);
              v51_acc += ((static_cast<float>(v55_data[14])) * v49_data);
              v51_acc += ((static_cast<float>(v55_data[15])) * v50_data);
              ir1.template select<16, 1>(0) = v51_acc;
              tensorforge::intel_esimd::simd<float, 16> v88_acc{};
              tensorforge::intel_esimd::simd<float, 16> v90_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v88_acc += ((static_cast<float>(v90_data[0])) * v35_data);
              v88_acc += ((static_cast<float>(v90_data[1])) * v36_data);
              v88_acc += ((static_cast<float>(v90_data[2])) * v37_data);
              v88_acc += ((static_cast<float>(v90_data[3])) * v38_data);
              v88_acc += ((static_cast<float>(v90_data[4])) * v39_data);
              v88_acc += ((static_cast<float>(v90_data[5])) * v40_data);
              v88_acc += ((static_cast<float>(v90_data[6])) * v41_data);
              v88_acc += ((static_cast<float>(v90_data[7])) * v42_data);
              v88_acc += ((static_cast<float>(v90_data[8])) * v43_data);
              v88_acc += ((static_cast<float>(v90_data[9])) * v44_data);
              v88_acc += ((static_cast<float>(v90_data[10])) * v45_data);
              v88_acc += ((static_cast<float>(v90_data[11])) * v46_data);
              v88_acc += ((static_cast<float>(v90_data[12])) * v47_data);
              v88_acc += ((static_cast<float>(v90_data[13])) * v48_data);
              v88_acc += ((static_cast<float>(v90_data[14])) * v49_data);
              v88_acc += ((static_cast<float>(v90_data[15])) * v50_data);
              ir1.template select<16, 1>(16) = v88_acc;
              tensorforge::intel_esimd::simd<float, 16> v123_acc{};
              tensorforge::intel_esimd::simd<float, 16> v125_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v123_acc += ((static_cast<float>(v125_data[0])) * v35_data);
              v123_acc += ((static_cast<float>(v125_data[1])) * v36_data);
              v123_acc += ((static_cast<float>(v125_data[2])) * v37_data);
              v123_acc += ((static_cast<float>(v125_data[3])) * v38_data);
              v123_acc += ((static_cast<float>(v125_data[4])) * v39_data);
              v123_acc += ((static_cast<float>(v125_data[5])) * v40_data);
              v123_acc += ((static_cast<float>(v125_data[6])) * v41_data);
              v123_acc += ((static_cast<float>(v125_data[7])) * v42_data);
              v123_acc += ((static_cast<float>(v125_data[8])) * v43_data);
              v123_acc += ((static_cast<float>(v125_data[9])) * v44_data);
              v123_acc += ((static_cast<float>(v125_data[10])) * v45_data);
              v123_acc += ((static_cast<float>(v125_data[11])) * v46_data);
              v123_acc += ((static_cast<float>(v125_data[12])) * v47_data);
              v123_acc += ((static_cast<float>(v125_data[13])) * v48_data);
              v123_acc += ((static_cast<float>(v125_data[14])) * v49_data);
              v123_acc += ((static_cast<float>(v125_data[15])) * v50_data);
              ir1.template select<16, 1>(32) = v123_acc;
              tensorforge::intel_esimd::simd<float, 16> v158_acc{};
              tensorforge::intel_esimd::simd<float, 16> v160_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v158_acc += ((static_cast<float>(v160_data[0])) * v35_data);
              v158_acc += ((static_cast<float>(v160_data[1])) * v36_data);
              v158_acc += ((static_cast<float>(v160_data[2])) * v37_data);
              v158_acc += ((static_cast<float>(v160_data[3])) * v38_data);
              v158_acc += ((static_cast<float>(v160_data[4])) * v39_data);
              v158_acc += ((static_cast<float>(v160_data[5])) * v40_data);
              v158_acc += ((static_cast<float>(v160_data[6])) * v41_data);
              v158_acc += ((static_cast<float>(v160_data[7])) * v42_data);
              v158_acc += ((static_cast<float>(v160_data[8])) * v43_data);
              v158_acc += ((static_cast<float>(v160_data[9])) * v44_data);
              v158_acc += ((static_cast<float>(v160_data[10])) * v45_data);
              v158_acc += ((static_cast<float>(v160_data[11])) * v46_data);
              v158_acc += ((static_cast<float>(v160_data[12])) * v47_data);
              v158_acc += ((static_cast<float>(v160_data[13])) * v48_data);
              v158_acc += ((static_cast<float>(v160_data[14])) * v49_data);
              v158_acc += ((static_cast<float>(v160_data[15])) * v50_data);
              ir1.template select<16, 1>(48) = v158_acc;
              tensorforge::intel_esimd::simd<float, 16> v193_acc{};
              tensorforge::intel_esimd::simd<float, 16> v195_data = tensorforge::slmLoad<float, 16>(s0 + (64_i32));
              v193_acc += ((static_cast<float>(v195_data[0])) * v35_data);
              v193_acc += ((static_cast<float>(v195_data[1])) * v36_data);
              v193_acc += ((static_cast<float>(v195_data[2])) * v37_data);
              v193_acc += ((static_cast<float>(v195_data[3])) * v38_data);
              v193_acc += ((static_cast<float>(v195_data[4])) * v39_data);
              v193_acc += ((static_cast<float>(v195_data[5])) * v40_data);
              v193_acc += ((static_cast<float>(v195_data[6])) * v41_data);
              v193_acc += ((static_cast<float>(v195_data[7])) * v42_data);
              v193_acc += ((static_cast<float>(v195_data[8])) * v43_data);
              v193_acc += ((static_cast<float>(v195_data[9])) * v44_data);
              v193_acc += ((static_cast<float>(v195_data[10])) * v45_data);
              v193_acc += ((static_cast<float>(v195_data[11])) * v46_data);
              v193_acc += ((static_cast<float>(v195_data[12])) * v47_data);
              v193_acc += ((static_cast<float>(v195_data[13])) * v48_data);
              v193_acc += ((static_cast<float>(v195_data[14])) * v49_data);
              v193_acc += ((static_cast<float>(v195_data[15])) * v50_data);
              ir1.template select<16, 1>(64) = v193_acc;
              tensorforge::intel_esimd::simd<float, 16> v228_acc{};
              tensorforge::intel_esimd::simd<float, 16> v230_data = tensorforge::slmLoad<float, 16>(s0 + (80_i32));
              v228_acc += ((static_cast<float>(v230_data[0])) * v35_data);
              v228_acc += ((static_cast<float>(v230_data[1])) * v36_data);
              v228_acc += ((static_cast<float>(v230_data[2])) * v37_data);
              v228_acc += ((static_cast<float>(v230_data[3])) * v38_data);
              v228_acc += ((static_cast<float>(v230_data[4])) * v39_data);
              v228_acc += ((static_cast<float>(v230_data[5])) * v40_data);
              v228_acc += ((static_cast<float>(v230_data[6])) * v41_data);
              v228_acc += ((static_cast<float>(v230_data[7])) * v42_data);
              v228_acc += ((static_cast<float>(v230_data[8])) * v43_data);
              v228_acc += ((static_cast<float>(v230_data[9])) * v44_data);
              v228_acc += ((static_cast<float>(v230_data[10])) * v45_data);
              v228_acc += ((static_cast<float>(v230_data[11])) * v46_data);
              v228_acc += ((static_cast<float>(v230_data[12])) * v47_data);
              v228_acc += ((static_cast<float>(v230_data[13])) * v48_data);
              v228_acc += ((static_cast<float>(v230_data[14])) * v49_data);
              v228_acc += ((static_cast<float>(v230_data[15])) * v50_data);
              ir1.template select<16, 1>(80) = v228_acc;
              tensorforge::intel_esimd::simd<float, 16> v263_acc{};
              tensorforge::intel_esimd::simd<float, 16> v265_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              v263_acc += ((static_cast<float>(v265_data[0])) * v35_data);
              v263_acc += ((static_cast<float>(v265_data[1])) * v36_data);
              v263_acc += ((static_cast<float>(v265_data[2])) * v37_data);
              v263_acc += ((static_cast<float>(v265_data[3])) * v38_data);
              v263_acc += ((static_cast<float>(v265_data[4])) * v39_data);
              v263_acc += ((static_cast<float>(v265_data[5])) * v40_data);
              v263_acc += ((static_cast<float>(v265_data[6])) * v41_data);
              v263_acc += ((static_cast<float>(v265_data[7])) * v42_data);
              v263_acc += ((static_cast<float>(v265_data[8])) * v43_data);
              v263_acc += ((static_cast<float>(v265_data[9])) * v44_data);
              v263_acc += ((static_cast<float>(v265_data[10])) * v45_data);
              v263_acc += ((static_cast<float>(v265_data[11])) * v46_data);
              v263_acc += ((static_cast<float>(v265_data[12])) * v47_data);
              v263_acc += ((static_cast<float>(v265_data[13])) * v48_data);
              v263_acc += ((static_cast<float>(v265_data[14])) * v49_data);
              v263_acc += ((static_cast<float>(v265_data[15])) * v50_data);
              ir1.template select<16, 1>(96) = v263_acc;
              tensorforge::intel_esimd::simd<float, 16> v298_acc{};
              tensorforge::intel_esimd::simd<float, 16> v300_data = tensorforge::slmLoad<float, 16>(s0 + (112_i32));
              v298_acc += ((static_cast<float>(v300_data[0])) * v35_data);
              v298_acc += ((static_cast<float>(v300_data[1])) * v36_data);
              v298_acc += ((static_cast<float>(v300_data[2])) * v37_data);
              v298_acc += ((static_cast<float>(v300_data[3])) * v38_data);
              v298_acc += ((static_cast<float>(v300_data[4])) * v39_data);
              v298_acc += ((static_cast<float>(v300_data[5])) * v40_data);
              v298_acc += ((static_cast<float>(v300_data[6])) * v41_data);
              v298_acc += ((static_cast<float>(v300_data[7])) * v42_data);
              v298_acc += ((static_cast<float>(v300_data[8])) * v43_data);
              v298_acc += ((static_cast<float>(v300_data[9])) * v44_data);
              v298_acc += ((static_cast<float>(v300_data[10])) * v45_data);
              v298_acc += ((static_cast<float>(v300_data[11])) * v46_data);
              v298_acc += ((static_cast<float>(v300_data[12])) * v47_data);
              v298_acc += ((static_cast<float>(v300_data[13])) * v48_data);
              v298_acc += ((static_cast<float>(v300_data[14])) * v49_data);
              v298_acc += ((static_cast<float>(v300_data[15])) * v50_data);
              ir1.template select<16, 1>(112) = v298_acc;
              // r1 = ir1
              #pragma unroll
              for (int32_t v333_n0 = 0; v333_n0 < 1; ++v333_n0) {
                int32_t v335_a = v333_n0 * 16;
                #pragma unroll
                for (int32_t v334_n1 = 0; v334_n1 < 8; ++v334_n1) {
                  int32_t v337_a = v335_a + (v334_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v338_data(ir1.template select<16, 1>(v337_a));
                  r1.template select<16, 1>(v337_a) = v338_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v339_i0 = 0; v339_i0 < 1; ++v339_i0) {
                int32_t v341_a = v339_i0 * 16;
                #pragma unroll
                for (int32_t v340_i1 = 0; v340_i1 < 8; ++v340_i1) {
                  int32_t v343_a = v341_a + (v340_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v344_data(r1.template select<16, 1>(v343_a));
                  v344_data.copy_to(glb_m0 + (v343_a));
                }
              }
            }
          }
        }
      }
    });
  });
}

