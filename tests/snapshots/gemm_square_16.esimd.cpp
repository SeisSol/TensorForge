// === base name ===
kernel_028856f32f862b54

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_028856f32f862b54 = {{1, 16, 1}, 16, 16, 1, 16, 17408, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_028856f32f862b54(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_028856f32f862b54(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_028856f32f862b54(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 4352 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_028856f32f862b54(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_028856f32f862b54(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_028856f32f862b54(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_028856f32f862b54(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<4352 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 17408 B shared, occupancy grid
        // operands:
        //   m0 16×16(16×16) {0..16}×{0..16} strided
        //   m1 16×16(16×16) {0..16}×{0..16} strided
        //   m2 16×16(16×16) {0..16}×{0..16} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":4352}],"shared_bytes":17408,"shared_elements":4352,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (272 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (256);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v5_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v5_batchId0 < numElements0; v5_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v6_ahead1 = v5_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
            const float *const __restrict__ pf_glb_m1 = &m1[v8_batchId1 * 256 + 0 + m1_extraOffset];
            const float *const __restrict__ pf_glb_m2 = &m2[v8_batchId1 * 256 + 0 + m2_extraOffset];
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 256 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 256 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v19_i0 = 0; v19_i0 < 1; ++v19_i0) {
                int32_t v21_lead = v19_i0 * 16;
                #pragma unroll
                for (int32_t v20_i1 = 0; v20_i1 < 16; ++v20_i1) {
                  int32_t v24_a = v21_lead + (v20_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v25_data;
                  v25_data.copy_from(glb_m1 + (v24_a));
                  r0.template select<16, 1>(v24_a) = v25_data;
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v27_ld;
              v27_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v27_ld);
              tensorforge::intel_esimd::simd<float, 64> v28_ld;
              v28_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v28_ld);
              tensorforge::intel_esimd::simd<float, 64> v29_ld;
              v29_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 128));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 128), v29_ld);
              tensorforge::intel_esimd::simd<float, 64> v30_ld;
              v30_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 192));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 192), v30_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 256> r1(0.0f);
              // r1 = +(r0 * s0) + None
              // [(0, 16), (0, 16)] [(0, 16)]
              tensorforge::intel_esimd::simd<float, 256> ir1(0.0f);
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
              tensorforge::intel_esimd::simd<float, 16> v331_acc{};
              tensorforge::intel_esimd::simd<float, 16> v333_data = tensorforge::slmLoad<float, 16>(s0 + (128_i32));
              v331_acc += ((static_cast<float>(v333_data[0])) * v33_data);
              v331_acc += ((static_cast<float>(v333_data[1])) * v34_data);
              v331_acc += ((static_cast<float>(v333_data[2])) * v35_data);
              v331_acc += ((static_cast<float>(v333_data[3])) * v36_data);
              v331_acc += ((static_cast<float>(v333_data[4])) * v37_data);
              v331_acc += ((static_cast<float>(v333_data[5])) * v38_data);
              v331_acc += ((static_cast<float>(v333_data[6])) * v39_data);
              v331_acc += ((static_cast<float>(v333_data[7])) * v40_data);
              v331_acc += ((static_cast<float>(v333_data[8])) * v41_data);
              v331_acc += ((static_cast<float>(v333_data[9])) * v42_data);
              v331_acc += ((static_cast<float>(v333_data[10])) * v43_data);
              v331_acc += ((static_cast<float>(v333_data[11])) * v44_data);
              v331_acc += ((static_cast<float>(v333_data[12])) * v45_data);
              v331_acc += ((static_cast<float>(v333_data[13])) * v46_data);
              v331_acc += ((static_cast<float>(v333_data[14])) * v47_data);
              v331_acc += ((static_cast<float>(v333_data[15])) * v48_data);
              ir1.template select<16, 1>(128) = v331_acc;
              tensorforge::intel_esimd::simd<float, 16> v366_acc{};
              tensorforge::intel_esimd::simd<float, 16> v368_data = tensorforge::slmLoad<float, 16>(s0 + (144_i32));
              v366_acc += ((static_cast<float>(v368_data[0])) * v33_data);
              v366_acc += ((static_cast<float>(v368_data[1])) * v34_data);
              v366_acc += ((static_cast<float>(v368_data[2])) * v35_data);
              v366_acc += ((static_cast<float>(v368_data[3])) * v36_data);
              v366_acc += ((static_cast<float>(v368_data[4])) * v37_data);
              v366_acc += ((static_cast<float>(v368_data[5])) * v38_data);
              v366_acc += ((static_cast<float>(v368_data[6])) * v39_data);
              v366_acc += ((static_cast<float>(v368_data[7])) * v40_data);
              v366_acc += ((static_cast<float>(v368_data[8])) * v41_data);
              v366_acc += ((static_cast<float>(v368_data[9])) * v42_data);
              v366_acc += ((static_cast<float>(v368_data[10])) * v43_data);
              v366_acc += ((static_cast<float>(v368_data[11])) * v44_data);
              v366_acc += ((static_cast<float>(v368_data[12])) * v45_data);
              v366_acc += ((static_cast<float>(v368_data[13])) * v46_data);
              v366_acc += ((static_cast<float>(v368_data[14])) * v47_data);
              v366_acc += ((static_cast<float>(v368_data[15])) * v48_data);
              ir1.template select<16, 1>(144) = v366_acc;
              tensorforge::intel_esimd::simd<float, 16> v401_acc{};
              tensorforge::intel_esimd::simd<float, 16> v403_data = tensorforge::slmLoad<float, 16>(s0 + (160_i32));
              v401_acc += ((static_cast<float>(v403_data[0])) * v33_data);
              v401_acc += ((static_cast<float>(v403_data[1])) * v34_data);
              v401_acc += ((static_cast<float>(v403_data[2])) * v35_data);
              v401_acc += ((static_cast<float>(v403_data[3])) * v36_data);
              v401_acc += ((static_cast<float>(v403_data[4])) * v37_data);
              v401_acc += ((static_cast<float>(v403_data[5])) * v38_data);
              v401_acc += ((static_cast<float>(v403_data[6])) * v39_data);
              v401_acc += ((static_cast<float>(v403_data[7])) * v40_data);
              v401_acc += ((static_cast<float>(v403_data[8])) * v41_data);
              v401_acc += ((static_cast<float>(v403_data[9])) * v42_data);
              v401_acc += ((static_cast<float>(v403_data[10])) * v43_data);
              v401_acc += ((static_cast<float>(v403_data[11])) * v44_data);
              v401_acc += ((static_cast<float>(v403_data[12])) * v45_data);
              v401_acc += ((static_cast<float>(v403_data[13])) * v46_data);
              v401_acc += ((static_cast<float>(v403_data[14])) * v47_data);
              v401_acc += ((static_cast<float>(v403_data[15])) * v48_data);
              ir1.template select<16, 1>(160) = v401_acc;
              tensorforge::intel_esimd::simd<float, 16> v436_acc{};
              tensorforge::intel_esimd::simd<float, 16> v438_data = tensorforge::slmLoad<float, 16>(s0 + (176_i32));
              v436_acc += ((static_cast<float>(v438_data[0])) * v33_data);
              v436_acc += ((static_cast<float>(v438_data[1])) * v34_data);
              v436_acc += ((static_cast<float>(v438_data[2])) * v35_data);
              v436_acc += ((static_cast<float>(v438_data[3])) * v36_data);
              v436_acc += ((static_cast<float>(v438_data[4])) * v37_data);
              v436_acc += ((static_cast<float>(v438_data[5])) * v38_data);
              v436_acc += ((static_cast<float>(v438_data[6])) * v39_data);
              v436_acc += ((static_cast<float>(v438_data[7])) * v40_data);
              v436_acc += ((static_cast<float>(v438_data[8])) * v41_data);
              v436_acc += ((static_cast<float>(v438_data[9])) * v42_data);
              v436_acc += ((static_cast<float>(v438_data[10])) * v43_data);
              v436_acc += ((static_cast<float>(v438_data[11])) * v44_data);
              v436_acc += ((static_cast<float>(v438_data[12])) * v45_data);
              v436_acc += ((static_cast<float>(v438_data[13])) * v46_data);
              v436_acc += ((static_cast<float>(v438_data[14])) * v47_data);
              v436_acc += ((static_cast<float>(v438_data[15])) * v48_data);
              ir1.template select<16, 1>(176) = v436_acc;
              tensorforge::intel_esimd::simd<float, 16> v471_acc{};
              tensorforge::intel_esimd::simd<float, 16> v473_data = tensorforge::slmLoad<float, 16>(s0 + (192_i32));
              v471_acc += ((static_cast<float>(v473_data[0])) * v33_data);
              v471_acc += ((static_cast<float>(v473_data[1])) * v34_data);
              v471_acc += ((static_cast<float>(v473_data[2])) * v35_data);
              v471_acc += ((static_cast<float>(v473_data[3])) * v36_data);
              v471_acc += ((static_cast<float>(v473_data[4])) * v37_data);
              v471_acc += ((static_cast<float>(v473_data[5])) * v38_data);
              v471_acc += ((static_cast<float>(v473_data[6])) * v39_data);
              v471_acc += ((static_cast<float>(v473_data[7])) * v40_data);
              v471_acc += ((static_cast<float>(v473_data[8])) * v41_data);
              v471_acc += ((static_cast<float>(v473_data[9])) * v42_data);
              v471_acc += ((static_cast<float>(v473_data[10])) * v43_data);
              v471_acc += ((static_cast<float>(v473_data[11])) * v44_data);
              v471_acc += ((static_cast<float>(v473_data[12])) * v45_data);
              v471_acc += ((static_cast<float>(v473_data[13])) * v46_data);
              v471_acc += ((static_cast<float>(v473_data[14])) * v47_data);
              v471_acc += ((static_cast<float>(v473_data[15])) * v48_data);
              ir1.template select<16, 1>(192) = v471_acc;
              tensorforge::intel_esimd::simd<float, 16> v506_acc{};
              tensorforge::intel_esimd::simd<float, 16> v508_data = tensorforge::slmLoad<float, 16>(s0 + (208_i32));
              v506_acc += ((static_cast<float>(v508_data[0])) * v33_data);
              v506_acc += ((static_cast<float>(v508_data[1])) * v34_data);
              v506_acc += ((static_cast<float>(v508_data[2])) * v35_data);
              v506_acc += ((static_cast<float>(v508_data[3])) * v36_data);
              v506_acc += ((static_cast<float>(v508_data[4])) * v37_data);
              v506_acc += ((static_cast<float>(v508_data[5])) * v38_data);
              v506_acc += ((static_cast<float>(v508_data[6])) * v39_data);
              v506_acc += ((static_cast<float>(v508_data[7])) * v40_data);
              v506_acc += ((static_cast<float>(v508_data[8])) * v41_data);
              v506_acc += ((static_cast<float>(v508_data[9])) * v42_data);
              v506_acc += ((static_cast<float>(v508_data[10])) * v43_data);
              v506_acc += ((static_cast<float>(v508_data[11])) * v44_data);
              v506_acc += ((static_cast<float>(v508_data[12])) * v45_data);
              v506_acc += ((static_cast<float>(v508_data[13])) * v46_data);
              v506_acc += ((static_cast<float>(v508_data[14])) * v47_data);
              v506_acc += ((static_cast<float>(v508_data[15])) * v48_data);
              ir1.template select<16, 1>(208) = v506_acc;
              tensorforge::intel_esimd::simd<float, 16> v541_acc{};
              tensorforge::intel_esimd::simd<float, 16> v543_data = tensorforge::slmLoad<float, 16>(s0 + (224_i32));
              v541_acc += ((static_cast<float>(v543_data[0])) * v33_data);
              v541_acc += ((static_cast<float>(v543_data[1])) * v34_data);
              v541_acc += ((static_cast<float>(v543_data[2])) * v35_data);
              v541_acc += ((static_cast<float>(v543_data[3])) * v36_data);
              v541_acc += ((static_cast<float>(v543_data[4])) * v37_data);
              v541_acc += ((static_cast<float>(v543_data[5])) * v38_data);
              v541_acc += ((static_cast<float>(v543_data[6])) * v39_data);
              v541_acc += ((static_cast<float>(v543_data[7])) * v40_data);
              v541_acc += ((static_cast<float>(v543_data[8])) * v41_data);
              v541_acc += ((static_cast<float>(v543_data[9])) * v42_data);
              v541_acc += ((static_cast<float>(v543_data[10])) * v43_data);
              v541_acc += ((static_cast<float>(v543_data[11])) * v44_data);
              v541_acc += ((static_cast<float>(v543_data[12])) * v45_data);
              v541_acc += ((static_cast<float>(v543_data[13])) * v46_data);
              v541_acc += ((static_cast<float>(v543_data[14])) * v47_data);
              v541_acc += ((static_cast<float>(v543_data[15])) * v48_data);
              ir1.template select<16, 1>(224) = v541_acc;
              tensorforge::intel_esimd::simd<float, 16> v576_acc{};
              tensorforge::intel_esimd::simd<float, 16> v578_data = tensorforge::slmLoad<float, 16>(s0 + (240_i32));
              v576_acc += ((static_cast<float>(v578_data[0])) * v33_data);
              v576_acc += ((static_cast<float>(v578_data[1])) * v34_data);
              v576_acc += ((static_cast<float>(v578_data[2])) * v35_data);
              v576_acc += ((static_cast<float>(v578_data[3])) * v36_data);
              v576_acc += ((static_cast<float>(v578_data[4])) * v37_data);
              v576_acc += ((static_cast<float>(v578_data[5])) * v38_data);
              v576_acc += ((static_cast<float>(v578_data[6])) * v39_data);
              v576_acc += ((static_cast<float>(v578_data[7])) * v40_data);
              v576_acc += ((static_cast<float>(v578_data[8])) * v41_data);
              v576_acc += ((static_cast<float>(v578_data[9])) * v42_data);
              v576_acc += ((static_cast<float>(v578_data[10])) * v43_data);
              v576_acc += ((static_cast<float>(v578_data[11])) * v44_data);
              v576_acc += ((static_cast<float>(v578_data[12])) * v45_data);
              v576_acc += ((static_cast<float>(v578_data[13])) * v46_data);
              v576_acc += ((static_cast<float>(v578_data[14])) * v47_data);
              v576_acc += ((static_cast<float>(v578_data[15])) * v48_data);
              ir1.template select<16, 1>(240) = v576_acc;
              #pragma unroll
              for (int32_t v611_n0 = 0; v611_n0 < 1; ++v611_n0) {
                int32_t v613_a = v611_n0 * 16;
                #pragma unroll
                for (int32_t v612_n1 = 0; v612_n1 < 16; ++v612_n1) {
                  int32_t v615_a = v613_a + (v612_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v616_data(ir1.template select<16, 1>(v615_a));
                  r1.template select<16, 1>(v615_a) = v616_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v617_i0 = 0; v617_i0 < 1; ++v617_i0) {
                int32_t v619_a = v617_i0 * 16;
                #pragma unroll
                for (int32_t v618_i1 = 0; v618_i1 < 16; ++v618_i1) {
                  int32_t v621_a = v619_a + (v618_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v622_data(r1.template select<16, 1>(v621_a));
                  v622_data.copy_to(glb_m0 + (v621_a));
                }
              }
            }
            tensorforge::prefetchL2<256>(&pf_glb_m1[0]);
            tensorforge::prefetchL2<256>(&pf_glb_m2[0]);
          }
        }
      }
    });
  });
}

