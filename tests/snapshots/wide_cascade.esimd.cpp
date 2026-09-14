// === base name ===
kernel_4f9b055f61da0110

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_4f9b055f61da0110 = {{1, 16, 1}, 16, 16, 1, 16, 12288, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_4f9b055f61da0110(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_4f9b055f61da0110(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_4f9b055f61da0110(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 3072 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_4f9b055f61da0110(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_4f9b055f61da0110(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_4f9b055f61da0110(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_4f9b055f61da0110(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<3072 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 12288 B shared, occupancy grid
        // operands:
        //   m0 16×11(16×11) {0..16}×{0..11} strided
        //   m1 16×16(16×16) {0..16}×{0..16} strided
        //   m2 16×11(16×11) {0..16}×{0..11} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":3072}],"shared_bytes":12288,"shared_elements":3072,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[16,11]],"name":"m0","ordered":false,"parts":1,"shape":[16,11],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,11]],"name":"m2","ordered":false,"parts":1,"shape":[16,11],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,11]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,11]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,11]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,11]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (192 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (176);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v5_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v5_batchId0 < numElements0; v5_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v6_ahead1 = v5_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 176 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 176 + 0 + m2_extraOffset];
              tensorforge::intel_esimd::simd<float, 256> r0(0.0f);
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v17_i0 = 0; v17_i0 < 1; ++v17_i0) {
                int32_t v19_lead = v17_i0 * 16;
                #pragma unroll
                for (int32_t v18_i1 = 0; v18_i1 < 16; ++v18_i1) {
                  int32_t v22_a = v19_lead + (v18_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v23_data;
                  v23_data.copy_from(glb_m1 + (v22_a));
                  r0.template select<16, 1>(v22_a) = v23_data;
                }
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v25_ld;
              v25_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 0));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 0), v25_ld);
              tensorforge::intel_esimd::simd<float, 64> v26_ld;
              v26_ld.copy_from(glb_m2 + (0 + 0 + 4 * 0 + 64));
              tensorforge::slmStore<float, 64>(s0 + (0 + 0 + 4 * 0 + 64), v26_ld);
              tensorforge::intel_esimd::simd<float, 32> v27_ld;
              v27_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 128));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 2 * 0 + 128), v27_ld);
              tensorforge::intel_esimd::simd<float, 16> v28_ld;
              v28_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 160));
              tensorforge::slmStore<float, 16>(s0 + (0 + 0 + 1 * 0 + 160), v28_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 176> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 16), (0, 11)] [(0, 16)]
              tensorforge::intel_esimd::simd<float, 176> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v31_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v32_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v33_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v34_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v35_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v36_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v37_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v38_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v39_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v40_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(192));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(208));
              tensorforge::intel_esimd::simd<float, 16> v45_data(r0.template select<16, 1>(224));
              tensorforge::intel_esimd::simd<float, 16> v46_data(r0.template select<16, 1>(240));
              tensorforge::intel_esimd::simd<float, 16> v47_acc{};
              tensorforge::intel_esimd::simd<float, 16> v51_data = tensorforge::slmLoad<float, 16>(s0 + (0_i32));
              v47_acc += ((static_cast<float>(v51_data[0])) * v31_data);
              v47_acc += ((static_cast<float>(v51_data[1])) * v32_data);
              v47_acc += ((static_cast<float>(v51_data[2])) * v33_data);
              v47_acc += ((static_cast<float>(v51_data[3])) * v34_data);
              v47_acc += ((static_cast<float>(v51_data[4])) * v35_data);
              v47_acc += ((static_cast<float>(v51_data[5])) * v36_data);
              v47_acc += ((static_cast<float>(v51_data[6])) * v37_data);
              v47_acc += ((static_cast<float>(v51_data[7])) * v38_data);
              v47_acc += ((static_cast<float>(v51_data[8])) * v39_data);
              v47_acc += ((static_cast<float>(v51_data[9])) * v40_data);
              v47_acc += ((static_cast<float>(v51_data[10])) * v41_data);
              v47_acc += ((static_cast<float>(v51_data[11])) * v42_data);
              v47_acc += ((static_cast<float>(v51_data[12])) * v43_data);
              v47_acc += ((static_cast<float>(v51_data[13])) * v44_data);
              v47_acc += ((static_cast<float>(v51_data[14])) * v45_data);
              v47_acc += ((static_cast<float>(v51_data[15])) * v46_data);
              ir1.template select<16, 1>(0) = v47_acc;
              tensorforge::intel_esimd::simd<float, 16> v84_acc{};
              tensorforge::intel_esimd::simd<float, 16> v86_data = tensorforge::slmLoad<float, 16>(s0 + (16_i32));
              v84_acc += ((static_cast<float>(v86_data[0])) * v31_data);
              v84_acc += ((static_cast<float>(v86_data[1])) * v32_data);
              v84_acc += ((static_cast<float>(v86_data[2])) * v33_data);
              v84_acc += ((static_cast<float>(v86_data[3])) * v34_data);
              v84_acc += ((static_cast<float>(v86_data[4])) * v35_data);
              v84_acc += ((static_cast<float>(v86_data[5])) * v36_data);
              v84_acc += ((static_cast<float>(v86_data[6])) * v37_data);
              v84_acc += ((static_cast<float>(v86_data[7])) * v38_data);
              v84_acc += ((static_cast<float>(v86_data[8])) * v39_data);
              v84_acc += ((static_cast<float>(v86_data[9])) * v40_data);
              v84_acc += ((static_cast<float>(v86_data[10])) * v41_data);
              v84_acc += ((static_cast<float>(v86_data[11])) * v42_data);
              v84_acc += ((static_cast<float>(v86_data[12])) * v43_data);
              v84_acc += ((static_cast<float>(v86_data[13])) * v44_data);
              v84_acc += ((static_cast<float>(v86_data[14])) * v45_data);
              v84_acc += ((static_cast<float>(v86_data[15])) * v46_data);
              ir1.template select<16, 1>(16) = v84_acc;
              tensorforge::intel_esimd::simd<float, 16> v119_acc{};
              tensorforge::intel_esimd::simd<float, 16> v121_data = tensorforge::slmLoad<float, 16>(s0 + (32_i32));
              v119_acc += ((static_cast<float>(v121_data[0])) * v31_data);
              v119_acc += ((static_cast<float>(v121_data[1])) * v32_data);
              v119_acc += ((static_cast<float>(v121_data[2])) * v33_data);
              v119_acc += ((static_cast<float>(v121_data[3])) * v34_data);
              v119_acc += ((static_cast<float>(v121_data[4])) * v35_data);
              v119_acc += ((static_cast<float>(v121_data[5])) * v36_data);
              v119_acc += ((static_cast<float>(v121_data[6])) * v37_data);
              v119_acc += ((static_cast<float>(v121_data[7])) * v38_data);
              v119_acc += ((static_cast<float>(v121_data[8])) * v39_data);
              v119_acc += ((static_cast<float>(v121_data[9])) * v40_data);
              v119_acc += ((static_cast<float>(v121_data[10])) * v41_data);
              v119_acc += ((static_cast<float>(v121_data[11])) * v42_data);
              v119_acc += ((static_cast<float>(v121_data[12])) * v43_data);
              v119_acc += ((static_cast<float>(v121_data[13])) * v44_data);
              v119_acc += ((static_cast<float>(v121_data[14])) * v45_data);
              v119_acc += ((static_cast<float>(v121_data[15])) * v46_data);
              ir1.template select<16, 1>(32) = v119_acc;
              tensorforge::intel_esimd::simd<float, 16> v154_acc{};
              tensorforge::intel_esimd::simd<float, 16> v156_data = tensorforge::slmLoad<float, 16>(s0 + (48_i32));
              v154_acc += ((static_cast<float>(v156_data[0])) * v31_data);
              v154_acc += ((static_cast<float>(v156_data[1])) * v32_data);
              v154_acc += ((static_cast<float>(v156_data[2])) * v33_data);
              v154_acc += ((static_cast<float>(v156_data[3])) * v34_data);
              v154_acc += ((static_cast<float>(v156_data[4])) * v35_data);
              v154_acc += ((static_cast<float>(v156_data[5])) * v36_data);
              v154_acc += ((static_cast<float>(v156_data[6])) * v37_data);
              v154_acc += ((static_cast<float>(v156_data[7])) * v38_data);
              v154_acc += ((static_cast<float>(v156_data[8])) * v39_data);
              v154_acc += ((static_cast<float>(v156_data[9])) * v40_data);
              v154_acc += ((static_cast<float>(v156_data[10])) * v41_data);
              v154_acc += ((static_cast<float>(v156_data[11])) * v42_data);
              v154_acc += ((static_cast<float>(v156_data[12])) * v43_data);
              v154_acc += ((static_cast<float>(v156_data[13])) * v44_data);
              v154_acc += ((static_cast<float>(v156_data[14])) * v45_data);
              v154_acc += ((static_cast<float>(v156_data[15])) * v46_data);
              ir1.template select<16, 1>(48) = v154_acc;
              tensorforge::intel_esimd::simd<float, 16> v189_acc{};
              tensorforge::intel_esimd::simd<float, 16> v191_data = tensorforge::slmLoad<float, 16>(s0 + (64_i32));
              v189_acc += ((static_cast<float>(v191_data[0])) * v31_data);
              v189_acc += ((static_cast<float>(v191_data[1])) * v32_data);
              v189_acc += ((static_cast<float>(v191_data[2])) * v33_data);
              v189_acc += ((static_cast<float>(v191_data[3])) * v34_data);
              v189_acc += ((static_cast<float>(v191_data[4])) * v35_data);
              v189_acc += ((static_cast<float>(v191_data[5])) * v36_data);
              v189_acc += ((static_cast<float>(v191_data[6])) * v37_data);
              v189_acc += ((static_cast<float>(v191_data[7])) * v38_data);
              v189_acc += ((static_cast<float>(v191_data[8])) * v39_data);
              v189_acc += ((static_cast<float>(v191_data[9])) * v40_data);
              v189_acc += ((static_cast<float>(v191_data[10])) * v41_data);
              v189_acc += ((static_cast<float>(v191_data[11])) * v42_data);
              v189_acc += ((static_cast<float>(v191_data[12])) * v43_data);
              v189_acc += ((static_cast<float>(v191_data[13])) * v44_data);
              v189_acc += ((static_cast<float>(v191_data[14])) * v45_data);
              v189_acc += ((static_cast<float>(v191_data[15])) * v46_data);
              ir1.template select<16, 1>(64) = v189_acc;
              tensorforge::intel_esimd::simd<float, 16> v224_acc{};
              tensorforge::intel_esimd::simd<float, 16> v226_data = tensorforge::slmLoad<float, 16>(s0 + (80_i32));
              v224_acc += ((static_cast<float>(v226_data[0])) * v31_data);
              v224_acc += ((static_cast<float>(v226_data[1])) * v32_data);
              v224_acc += ((static_cast<float>(v226_data[2])) * v33_data);
              v224_acc += ((static_cast<float>(v226_data[3])) * v34_data);
              v224_acc += ((static_cast<float>(v226_data[4])) * v35_data);
              v224_acc += ((static_cast<float>(v226_data[5])) * v36_data);
              v224_acc += ((static_cast<float>(v226_data[6])) * v37_data);
              v224_acc += ((static_cast<float>(v226_data[7])) * v38_data);
              v224_acc += ((static_cast<float>(v226_data[8])) * v39_data);
              v224_acc += ((static_cast<float>(v226_data[9])) * v40_data);
              v224_acc += ((static_cast<float>(v226_data[10])) * v41_data);
              v224_acc += ((static_cast<float>(v226_data[11])) * v42_data);
              v224_acc += ((static_cast<float>(v226_data[12])) * v43_data);
              v224_acc += ((static_cast<float>(v226_data[13])) * v44_data);
              v224_acc += ((static_cast<float>(v226_data[14])) * v45_data);
              v224_acc += ((static_cast<float>(v226_data[15])) * v46_data);
              ir1.template select<16, 1>(80) = v224_acc;
              tensorforge::intel_esimd::simd<float, 16> v259_acc{};
              tensorforge::intel_esimd::simd<float, 16> v261_data = tensorforge::slmLoad<float, 16>(s0 + (96_i32));
              v259_acc += ((static_cast<float>(v261_data[0])) * v31_data);
              v259_acc += ((static_cast<float>(v261_data[1])) * v32_data);
              v259_acc += ((static_cast<float>(v261_data[2])) * v33_data);
              v259_acc += ((static_cast<float>(v261_data[3])) * v34_data);
              v259_acc += ((static_cast<float>(v261_data[4])) * v35_data);
              v259_acc += ((static_cast<float>(v261_data[5])) * v36_data);
              v259_acc += ((static_cast<float>(v261_data[6])) * v37_data);
              v259_acc += ((static_cast<float>(v261_data[7])) * v38_data);
              v259_acc += ((static_cast<float>(v261_data[8])) * v39_data);
              v259_acc += ((static_cast<float>(v261_data[9])) * v40_data);
              v259_acc += ((static_cast<float>(v261_data[10])) * v41_data);
              v259_acc += ((static_cast<float>(v261_data[11])) * v42_data);
              v259_acc += ((static_cast<float>(v261_data[12])) * v43_data);
              v259_acc += ((static_cast<float>(v261_data[13])) * v44_data);
              v259_acc += ((static_cast<float>(v261_data[14])) * v45_data);
              v259_acc += ((static_cast<float>(v261_data[15])) * v46_data);
              ir1.template select<16, 1>(96) = v259_acc;
              tensorforge::intel_esimd::simd<float, 16> v294_acc{};
              tensorforge::intel_esimd::simd<float, 16> v296_data = tensorforge::slmLoad<float, 16>(s0 + (112_i32));
              v294_acc += ((static_cast<float>(v296_data[0])) * v31_data);
              v294_acc += ((static_cast<float>(v296_data[1])) * v32_data);
              v294_acc += ((static_cast<float>(v296_data[2])) * v33_data);
              v294_acc += ((static_cast<float>(v296_data[3])) * v34_data);
              v294_acc += ((static_cast<float>(v296_data[4])) * v35_data);
              v294_acc += ((static_cast<float>(v296_data[5])) * v36_data);
              v294_acc += ((static_cast<float>(v296_data[6])) * v37_data);
              v294_acc += ((static_cast<float>(v296_data[7])) * v38_data);
              v294_acc += ((static_cast<float>(v296_data[8])) * v39_data);
              v294_acc += ((static_cast<float>(v296_data[9])) * v40_data);
              v294_acc += ((static_cast<float>(v296_data[10])) * v41_data);
              v294_acc += ((static_cast<float>(v296_data[11])) * v42_data);
              v294_acc += ((static_cast<float>(v296_data[12])) * v43_data);
              v294_acc += ((static_cast<float>(v296_data[13])) * v44_data);
              v294_acc += ((static_cast<float>(v296_data[14])) * v45_data);
              v294_acc += ((static_cast<float>(v296_data[15])) * v46_data);
              ir1.template select<16, 1>(112) = v294_acc;
              tensorforge::intel_esimd::simd<float, 16> v329_acc{};
              tensorforge::intel_esimd::simd<float, 16> v331_data = tensorforge::slmLoad<float, 16>(s0 + (128_i32));
              v329_acc += ((static_cast<float>(v331_data[0])) * v31_data);
              v329_acc += ((static_cast<float>(v331_data[1])) * v32_data);
              v329_acc += ((static_cast<float>(v331_data[2])) * v33_data);
              v329_acc += ((static_cast<float>(v331_data[3])) * v34_data);
              v329_acc += ((static_cast<float>(v331_data[4])) * v35_data);
              v329_acc += ((static_cast<float>(v331_data[5])) * v36_data);
              v329_acc += ((static_cast<float>(v331_data[6])) * v37_data);
              v329_acc += ((static_cast<float>(v331_data[7])) * v38_data);
              v329_acc += ((static_cast<float>(v331_data[8])) * v39_data);
              v329_acc += ((static_cast<float>(v331_data[9])) * v40_data);
              v329_acc += ((static_cast<float>(v331_data[10])) * v41_data);
              v329_acc += ((static_cast<float>(v331_data[11])) * v42_data);
              v329_acc += ((static_cast<float>(v331_data[12])) * v43_data);
              v329_acc += ((static_cast<float>(v331_data[13])) * v44_data);
              v329_acc += ((static_cast<float>(v331_data[14])) * v45_data);
              v329_acc += ((static_cast<float>(v331_data[15])) * v46_data);
              ir1.template select<16, 1>(128) = v329_acc;
              tensorforge::intel_esimd::simd<float, 16> v364_acc{};
              tensorforge::intel_esimd::simd<float, 16> v366_data = tensorforge::slmLoad<float, 16>(s0 + (144_i32));
              v364_acc += ((static_cast<float>(v366_data[0])) * v31_data);
              v364_acc += ((static_cast<float>(v366_data[1])) * v32_data);
              v364_acc += ((static_cast<float>(v366_data[2])) * v33_data);
              v364_acc += ((static_cast<float>(v366_data[3])) * v34_data);
              v364_acc += ((static_cast<float>(v366_data[4])) * v35_data);
              v364_acc += ((static_cast<float>(v366_data[5])) * v36_data);
              v364_acc += ((static_cast<float>(v366_data[6])) * v37_data);
              v364_acc += ((static_cast<float>(v366_data[7])) * v38_data);
              v364_acc += ((static_cast<float>(v366_data[8])) * v39_data);
              v364_acc += ((static_cast<float>(v366_data[9])) * v40_data);
              v364_acc += ((static_cast<float>(v366_data[10])) * v41_data);
              v364_acc += ((static_cast<float>(v366_data[11])) * v42_data);
              v364_acc += ((static_cast<float>(v366_data[12])) * v43_data);
              v364_acc += ((static_cast<float>(v366_data[13])) * v44_data);
              v364_acc += ((static_cast<float>(v366_data[14])) * v45_data);
              v364_acc += ((static_cast<float>(v366_data[15])) * v46_data);
              ir1.template select<16, 1>(144) = v364_acc;
              tensorforge::intel_esimd::simd<float, 16> v399_acc{};
              tensorforge::intel_esimd::simd<float, 16> v401_data = tensorforge::slmLoad<float, 16>(s0 + (160_i32));
              v399_acc += ((static_cast<float>(v401_data[0])) * v31_data);
              v399_acc += ((static_cast<float>(v401_data[1])) * v32_data);
              v399_acc += ((static_cast<float>(v401_data[2])) * v33_data);
              v399_acc += ((static_cast<float>(v401_data[3])) * v34_data);
              v399_acc += ((static_cast<float>(v401_data[4])) * v35_data);
              v399_acc += ((static_cast<float>(v401_data[5])) * v36_data);
              v399_acc += ((static_cast<float>(v401_data[6])) * v37_data);
              v399_acc += ((static_cast<float>(v401_data[7])) * v38_data);
              v399_acc += ((static_cast<float>(v401_data[8])) * v39_data);
              v399_acc += ((static_cast<float>(v401_data[9])) * v40_data);
              v399_acc += ((static_cast<float>(v401_data[10])) * v41_data);
              v399_acc += ((static_cast<float>(v401_data[11])) * v42_data);
              v399_acc += ((static_cast<float>(v401_data[12])) * v43_data);
              v399_acc += ((static_cast<float>(v401_data[13])) * v44_data);
              v399_acc += ((static_cast<float>(v401_data[14])) * v45_data);
              v399_acc += ((static_cast<float>(v401_data[15])) * v46_data);
              ir1.template select<16, 1>(160) = v399_acc;
              // r1 = ir1
              #pragma unroll
              for (int32_t v434_n0 = 0; v434_n0 < 1; ++v434_n0) {
                int32_t v436_a = v434_n0 * 16;
                #pragma unroll
                for (int32_t v435_n1 = 0; v435_n1 < 11; ++v435_n1) {
                  int32_t v438_a = v436_a + (v435_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v439_data(ir1.template select<16, 1>(v438_a));
                  r1.template select<16, 1>(v438_a) = v439_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v440_i0 = 0; v440_i0 < 1; ++v440_i0) {
                int32_t v442_a = v440_i0 * 16;
                #pragma unroll
                for (int32_t v441_i1 = 0; v441_i1 < 11; ++v441_i1) {
                  int32_t v444_a = v442_a + (v441_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v445_data(r1.template select<16, 1>(v444_a));
                  v445_data.copy_to(glb_m0 + (v444_a));
                }
              }
            }
          }
        }
      }
    });
  });
}

