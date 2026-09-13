// === base name ===
kernel_4ceb22a590b93fdc

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_4ceb22a590b93fdc = {{1, 16, 1}, 16, 12, 1, 16, 9216, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_4ceb22a590b93fdc(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_4ceb22a590b93fdc(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_4ceb22a590b93fdc(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 2304 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_4ceb22a590b93fdc(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_4ceb22a590b93fdc(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_4ceb22a590b93fdc(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_4ceb22a590b93fdc(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<2304 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 1x16x1, 9216 B shared, occupancy grid
        // operands:
        //   m0 16×8(12×8) {4..16}×{0..8} strided
        //   m1 16×16(12×16) {4..16}×{0..16} strided
        //   m2 16×8(16×8) {0..16}×{0..8} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2304}],"shared_bytes":9216,"shared_elements":2304,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[4,0],[16,8]],"name":"m0","ordered":false,"parts":1,"shape":[16,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[4,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[4,0],[16,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[4,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
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
            const float *const __restrict__ pf_glb_m1 = &m1[v8_batchId1 * 192 + 0 + m1_extraOffset];
            const float *const __restrict__ pf_glb_m2 = &m2[v8_batchId1 * 128 + 0 + m2_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 192 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 128 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v19_i1 = 0; v19_i1 < 16; ++v19_i1) {
                tensorforge::intel_esimd::simd<float, 12> v26_data;
                v26_data.copy_from(glb_m1 + ((v19_i1 * 12)));
                r0.template select<12, 1>((v19_i1 * 16)) = v26_data;
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v29_ld;
              v29_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v29_ld);
              tensorforge::intel_esimd::simd<float, 64> v30_ld;
              v30_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v30_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 128> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(16, 28), (0, 8)] [(0, 16)]
              tensorforge::intel_esimd::simd<float, 128> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v33_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v34_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v35_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v36_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v37_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v38_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v39_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v40_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(192));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(208));
              tensorforge::intel_esimd::simd<float, 16> v47_data(r0.template select<16, 1>(224));
              tensorforge::intel_esimd::simd<float, 16> v48_data(r0.template select<16, 1>(240));
              tensorforge::intel_esimd::simd<float, 16> v49_acc{};
              tensorforge::intel_esimd::simd<float, 16> v53_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v49_acc += ((static_cast<float>(v53_data[0])) * v33_data);
              v49_acc += ((static_cast<float>(v53_data[1])) * v34_data);
              v49_acc += ((static_cast<float>(v53_data[2])) * v35_data);
              v49_acc += ((static_cast<float>(v53_data[3])) * v36_data);
              v49_acc += ((static_cast<float>(v53_data[4])) * v37_data);
              v49_acc += ((static_cast<float>(v53_data[5])) * v38_data);
              v49_acc += ((static_cast<float>(v53_data[6])) * v39_data);
              v49_acc += ((static_cast<float>(v53_data[7])) * v40_data);
              v49_acc += ((static_cast<float>(v53_data[8])) * v41_data);
              v49_acc += ((static_cast<float>(v53_data[9])) * v42_data);
              v49_acc += ((static_cast<float>(v53_data[10])) * v43_data);
              v49_acc += ((static_cast<float>(v53_data[11])) * v44_data);
              v49_acc += ((static_cast<float>(v53_data[12])) * v45_data);
              v49_acc += ((static_cast<float>(v53_data[13])) * v46_data);
              v49_acc += ((static_cast<float>(v53_data[14])) * v47_data);
              v49_acc += ((static_cast<float>(v53_data[15])) * v48_data);
              ir1.template select<16, 1>(0) = v49_acc;
              tensorforge::intel_esimd::simd<float, 16> v86_acc{};
              tensorforge::intel_esimd::simd<float, 16> v88_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v86_acc += ((static_cast<float>(v88_data[0])) * v33_data);
              v86_acc += ((static_cast<float>(v88_data[1])) * v34_data);
              v86_acc += ((static_cast<float>(v88_data[2])) * v35_data);
              v86_acc += ((static_cast<float>(v88_data[3])) * v36_data);
              v86_acc += ((static_cast<float>(v88_data[4])) * v37_data);
              v86_acc += ((static_cast<float>(v88_data[5])) * v38_data);
              v86_acc += ((static_cast<float>(v88_data[6])) * v39_data);
              v86_acc += ((static_cast<float>(v88_data[7])) * v40_data);
              v86_acc += ((static_cast<float>(v88_data[8])) * v41_data);
              v86_acc += ((static_cast<float>(v88_data[9])) * v42_data);
              v86_acc += ((static_cast<float>(v88_data[10])) * v43_data);
              v86_acc += ((static_cast<float>(v88_data[11])) * v44_data);
              v86_acc += ((static_cast<float>(v88_data[12])) * v45_data);
              v86_acc += ((static_cast<float>(v88_data[13])) * v46_data);
              v86_acc += ((static_cast<float>(v88_data[14])) * v47_data);
              v86_acc += ((static_cast<float>(v88_data[15])) * v48_data);
              ir1.template select<16, 1>(16) = v86_acc;
              tensorforge::intel_esimd::simd<float, 16> v121_acc{};
              tensorforge::intel_esimd::simd<float, 16> v123_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v121_acc += ((static_cast<float>(v123_data[0])) * v33_data);
              v121_acc += ((static_cast<float>(v123_data[1])) * v34_data);
              v121_acc += ((static_cast<float>(v123_data[2])) * v35_data);
              v121_acc += ((static_cast<float>(v123_data[3])) * v36_data);
              v121_acc += ((static_cast<float>(v123_data[4])) * v37_data);
              v121_acc += ((static_cast<float>(v123_data[5])) * v38_data);
              v121_acc += ((static_cast<float>(v123_data[6])) * v39_data);
              v121_acc += ((static_cast<float>(v123_data[7])) * v40_data);
              v121_acc += ((static_cast<float>(v123_data[8])) * v41_data);
              v121_acc += ((static_cast<float>(v123_data[9])) * v42_data);
              v121_acc += ((static_cast<float>(v123_data[10])) * v43_data);
              v121_acc += ((static_cast<float>(v123_data[11])) * v44_data);
              v121_acc += ((static_cast<float>(v123_data[12])) * v45_data);
              v121_acc += ((static_cast<float>(v123_data[13])) * v46_data);
              v121_acc += ((static_cast<float>(v123_data[14])) * v47_data);
              v121_acc += ((static_cast<float>(v123_data[15])) * v48_data);
              ir1.template select<16, 1>(32) = v121_acc;
              tensorforge::intel_esimd::simd<float, 16> v156_acc{};
              tensorforge::intel_esimd::simd<float, 16> v158_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v156_acc += ((static_cast<float>(v158_data[0])) * v33_data);
              v156_acc += ((static_cast<float>(v158_data[1])) * v34_data);
              v156_acc += ((static_cast<float>(v158_data[2])) * v35_data);
              v156_acc += ((static_cast<float>(v158_data[3])) * v36_data);
              v156_acc += ((static_cast<float>(v158_data[4])) * v37_data);
              v156_acc += ((static_cast<float>(v158_data[5])) * v38_data);
              v156_acc += ((static_cast<float>(v158_data[6])) * v39_data);
              v156_acc += ((static_cast<float>(v158_data[7])) * v40_data);
              v156_acc += ((static_cast<float>(v158_data[8])) * v41_data);
              v156_acc += ((static_cast<float>(v158_data[9])) * v42_data);
              v156_acc += ((static_cast<float>(v158_data[10])) * v43_data);
              v156_acc += ((static_cast<float>(v158_data[11])) * v44_data);
              v156_acc += ((static_cast<float>(v158_data[12])) * v45_data);
              v156_acc += ((static_cast<float>(v158_data[13])) * v46_data);
              v156_acc += ((static_cast<float>(v158_data[14])) * v47_data);
              v156_acc += ((static_cast<float>(v158_data[15])) * v48_data);
              ir1.template select<16, 1>(48) = v156_acc;
              tensorforge::intel_esimd::simd<float, 16> v191_acc{};
              tensorforge::intel_esimd::simd<float, 16> v193_data = tensorforge::slmLoad<float, 16>(s0 + (64_i32));
              v191_acc += ((static_cast<float>(v193_data[0])) * v33_data);
              v191_acc += ((static_cast<float>(v193_data[1])) * v34_data);
              v191_acc += ((static_cast<float>(v193_data[2])) * v35_data);
              v191_acc += ((static_cast<float>(v193_data[3])) * v36_data);
              v191_acc += ((static_cast<float>(v193_data[4])) * v37_data);
              v191_acc += ((static_cast<float>(v193_data[5])) * v38_data);
              v191_acc += ((static_cast<float>(v193_data[6])) * v39_data);
              v191_acc += ((static_cast<float>(v193_data[7])) * v40_data);
              v191_acc += ((static_cast<float>(v193_data[8])) * v41_data);
              v191_acc += ((static_cast<float>(v193_data[9])) * v42_data);
              v191_acc += ((static_cast<float>(v193_data[10])) * v43_data);
              v191_acc += ((static_cast<float>(v193_data[11])) * v44_data);
              v191_acc += ((static_cast<float>(v193_data[12])) * v45_data);
              v191_acc += ((static_cast<float>(v193_data[13])) * v46_data);
              v191_acc += ((static_cast<float>(v193_data[14])) * v47_data);
              v191_acc += ((static_cast<float>(v193_data[15])) * v48_data);
              ir1.template select<16, 1>(64) = v191_acc;
              tensorforge::intel_esimd::simd<float, 16> v226_acc{};
              tensorforge::intel_esimd::simd<float, 16> v228_data = tensorforge::slmLoad<float, 16>(s0 + (80_i32));
              v226_acc += ((static_cast<float>(v228_data[0])) * v33_data);
              v226_acc += ((static_cast<float>(v228_data[1])) * v34_data);
              v226_acc += ((static_cast<float>(v228_data[2])) * v35_data);
              v226_acc += ((static_cast<float>(v228_data[3])) * v36_data);
              v226_acc += ((static_cast<float>(v228_data[4])) * v37_data);
              v226_acc += ((static_cast<float>(v228_data[5])) * v38_data);
              v226_acc += ((static_cast<float>(v228_data[6])) * v39_data);
              v226_acc += ((static_cast<float>(v228_data[7])) * v40_data);
              v226_acc += ((static_cast<float>(v228_data[8])) * v41_data);
              v226_acc += ((static_cast<float>(v228_data[9])) * v42_data);
              v226_acc += ((static_cast<float>(v228_data[10])) * v43_data);
              v226_acc += ((static_cast<float>(v228_data[11])) * v44_data);
              v226_acc += ((static_cast<float>(v228_data[12])) * v45_data);
              v226_acc += ((static_cast<float>(v228_data[13])) * v46_data);
              v226_acc += ((static_cast<float>(v228_data[14])) * v47_data);
              v226_acc += ((static_cast<float>(v228_data[15])) * v48_data);
              ir1.template select<16, 1>(80) = v226_acc;
              tensorforge::intel_esimd::simd<float, 16> v261_acc{};
              tensorforge::intel_esimd::simd<float, 16> v263_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              v261_acc += ((static_cast<float>(v263_data[0])) * v33_data);
              v261_acc += ((static_cast<float>(v263_data[1])) * v34_data);
              v261_acc += ((static_cast<float>(v263_data[2])) * v35_data);
              v261_acc += ((static_cast<float>(v263_data[3])) * v36_data);
              v261_acc += ((static_cast<float>(v263_data[4])) * v37_data);
              v261_acc += ((static_cast<float>(v263_data[5])) * v38_data);
              v261_acc += ((static_cast<float>(v263_data[6])) * v39_data);
              v261_acc += ((static_cast<float>(v263_data[7])) * v40_data);
              v261_acc += ((static_cast<float>(v263_data[8])) * v41_data);
              v261_acc += ((static_cast<float>(v263_data[9])) * v42_data);
              v261_acc += ((static_cast<float>(v263_data[10])) * v43_data);
              v261_acc += ((static_cast<float>(v263_data[11])) * v44_data);
              v261_acc += ((static_cast<float>(v263_data[12])) * v45_data);
              v261_acc += ((static_cast<float>(v263_data[13])) * v46_data);
              v261_acc += ((static_cast<float>(v263_data[14])) * v47_data);
              v261_acc += ((static_cast<float>(v263_data[15])) * v48_data);
              ir1.template select<16, 1>(96) = v261_acc;
              tensorforge::intel_esimd::simd<float, 16> v296_acc{};
              tensorforge::intel_esimd::simd<float, 16> v298_data = tensorforge::slmLoad<float, 16>(s0 + (112_i32));
              v296_acc += ((static_cast<float>(v298_data[0])) * v33_data);
              v296_acc += ((static_cast<float>(v298_data[1])) * v34_data);
              v296_acc += ((static_cast<float>(v298_data[2])) * v35_data);
              v296_acc += ((static_cast<float>(v298_data[3])) * v36_data);
              v296_acc += ((static_cast<float>(v298_data[4])) * v37_data);
              v296_acc += ((static_cast<float>(v298_data[5])) * v38_data);
              v296_acc += ((static_cast<float>(v298_data[6])) * v39_data);
              v296_acc += ((static_cast<float>(v298_data[7])) * v40_data);
              v296_acc += ((static_cast<float>(v298_data[8])) * v41_data);
              v296_acc += ((static_cast<float>(v298_data[9])) * v42_data);
              v296_acc += ((static_cast<float>(v298_data[10])) * v43_data);
              v296_acc += ((static_cast<float>(v298_data[11])) * v44_data);
              v296_acc += ((static_cast<float>(v298_data[12])) * v45_data);
              v296_acc += ((static_cast<float>(v298_data[13])) * v46_data);
              v296_acc += ((static_cast<float>(v298_data[14])) * v47_data);
              v296_acc += ((static_cast<float>(v298_data[15])) * v48_data);
              ir1.template select<16, 1>(112) = v296_acc;
              #pragma unroll
              for (int32_t v331_n1 = 0; v331_n1 < 8; ++v331_n1) {
                int32_t v332_a = v331_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v334_data(ir1.template select<12, 1>(v332_a));
                r1.template select<12, 1>(v332_a) = v334_data;
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v335_i1 = 0; v335_i1 < 8; ++v335_i1) {
                tensorforge::intel_esimd::simd<float, 12> v338_data(r1.template select<12, 1>((v335_i1 * 16)));
                v338_data.copy_to(glb_m0 + ((v335_i1 * 12)));
              }
            }
            tensorforge::prefetchRunsL2<768, 512>(&pf_glb_m1[0], &pf_glb_m2[0]);
          }
        }
      }
    });
  });
}

