// === base name ===
kernel_febe62c31f424345

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_febe62c31f424345 = {{1, 16, 1}, 16, 16, 1, 16, 4096, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_febe62c31f424345(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_febe62c31f424345(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_febe62c31f424345(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 1024 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_febe62c31f424345(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_febe62c31f424345(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_febe62c31f424345(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_febe62c31f424345(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
      tensorforge::slmReserve<1024 * sizeof(float)>(); {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 1x16x1, 4096 B shared, occupancy grid
        // operands:
        //   m0 16×16(16×16) {0..16}×{0..16} strided
        //   m1 16×16(16×16) {0..16}×{0..16} strided
        //   m2 16×16(16×16) {0..16}×{0..16} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[1,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1024}],"shared_bytes":4096,"shared_elements":1024,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          tensorforge::SlmPtr<float> totalShrMem = tensorforge::SlmPtr<float>(0);
          tensorforge::SlmPtr<float> localShrMem0 = totalShrMem + (64 * item.get_local_id(1) + 0);
          tensorforge::SlmPtr<float> tempShrMem = localShrMem0 + (48);
          tensorforge::SlmPtr<float> s0 = localShrMem0 + (0);
          for (size_t v5_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v5_batchId0 < numElements0; v5_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v6_ahead1 = v5_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 256 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 46 + 0 + m2_extraOffset];
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
              tensorforge::intel_esimd::simd<float, 32> v25_ld;
              v25_ld.copy_from(glb_m2 + (0 + 0 + 2 * 0 + 0));
              tensorforge::slmStore<float, 32>(s0 + (0 + 0 + 2 * 0 + 0), v25_ld);
              tensorforge::intel_esimd::simd<float, 14> v26_ld;
              v26_ld.copy_from(glb_m2 + (0 + 0 + 1 * 0 + 32));
              tensorforge::slmStore<float, 14>(s0 + (0 + 0 + 1 * 0 + 32), v26_ld);
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              tensorforge::intel_esimd::simd<float, 256> r1(0.0f);
              // ir1 = +(r0 * s0)
              // [(0, 16), (0, 16)] [(0, 16)]
              tensorforge::intel_esimd::simd<float, 256> ir1(0.0f);
              tensorforge::intel_esimd::simd<float, 16> v29_data(r0.template select<16, 1>(0));
              tensorforge::intel_esimd::simd<float, 16> v30_data(r0.template select<16, 1>(16));
              tensorforge::intel_esimd::simd<float, 16> v31_data(r0.template select<16, 1>(32));
              tensorforge::intel_esimd::simd<float, 16> v32_data(r0.template select<16, 1>(48));
              tensorforge::intel_esimd::simd<float, 16> v33_data(r0.template select<16, 1>(64));
              tensorforge::intel_esimd::simd<float, 16> v34_data(r0.template select<16, 1>(80));
              tensorforge::intel_esimd::simd<float, 16> v35_data(r0.template select<16, 1>(96));
              tensorforge::intel_esimd::simd<float, 16> v36_data(r0.template select<16, 1>(112));
              tensorforge::intel_esimd::simd<float, 16> v37_data(r0.template select<16, 1>(128));
              tensorforge::intel_esimd::simd<float, 16> v38_data(r0.template select<16, 1>(144));
              tensorforge::intel_esimd::simd<float, 16> v39_data(r0.template select<16, 1>(160));
              tensorforge::intel_esimd::simd<float, 16> v40_data(r0.template select<16, 1>(176));
              tensorforge::intel_esimd::simd<float, 16> v41_data(r0.template select<16, 1>(192));
              tensorforge::intel_esimd::simd<float, 16> v42_data(r0.template select<16, 1>(208));
              tensorforge::intel_esimd::simd<float, 16> v43_data(r0.template select<16, 1>(224));
              tensorforge::intel_esimd::simd<float, 16> v44_data(r0.template select<16, 1>(240));
              tensorforge::intel_esimd::simd<float, 16> v45_acc{};
              tensorforge::intel_esimd::simd<float, 16> v46_lin = tensorforge::slmLoad<float, 16>(s0 + (0));
              float v47_bc = static_cast<float>(v46_lin[0]);
              v45_acc += (v47_bc * v29_data);
              float v49_bc = static_cast<float>(v46_lin[1]);
              v45_acc += (v49_bc * v30_data);
              float v51_bc = static_cast<float>(v46_lin[2]);
              v45_acc += (v51_bc * v31_data);
              float v53_bc = static_cast<float>(v46_lin[3]);
              v45_acc += (v53_bc * v32_data);
              float v55_bc = static_cast<float>(v46_lin[4]);
              v45_acc += (v55_bc * v33_data);
              float v57_bc = static_cast<float>(v46_lin[5]);
              v45_acc += (v57_bc * v34_data);
              float v59_bc = static_cast<float>(v46_lin[6]);
              v45_acc += (v59_bc * v35_data);
              float v61_bc = static_cast<float>(v46_lin[7]);
              v45_acc += (v61_bc * v36_data);
              float v63_bc = static_cast<float>(v46_lin[8]);
              v45_acc += (v63_bc * v37_data);
              float v65_bc = static_cast<float>(v46_lin[9]);
              v45_acc += (v65_bc * v38_data);
              float v67_bc = static_cast<float>(v46_lin[10]);
              v45_acc += (v67_bc * v39_data);
              float v69_bc = static_cast<float>(v46_lin[11]);
              v45_acc += (v69_bc * v40_data);
              float v71_bc = static_cast<float>(v46_lin[12]);
              v45_acc += (v71_bc * v41_data);
              float v73_bc = static_cast<float>(v46_lin[13]);
              v45_acc += (v73_bc * v42_data);
              float v75_bc = static_cast<float>(v46_lin[14]);
              v45_acc += (v75_bc * v43_data);
              float v77_bc = static_cast<float>(v46_lin[15]);
              v45_acc += (v77_bc * v44_data);
              ir1.template select<16, 1>(0) = v45_acc;
              tensorforge::intel_esimd::simd<float, 16> v79_acc{};
              v79_acc += (v47_bc * v29_data);
              v79_acc += (v49_bc * v30_data);
              v79_acc += (v51_bc * v31_data);
              v79_acc += (v53_bc * v32_data);
              v79_acc += (v55_bc * v33_data);
              v79_acc += (v57_bc * v34_data);
              v79_acc += (v59_bc * v35_data);
              v79_acc += (v61_bc * v36_data);
              v79_acc += (v63_bc * v37_data);
              v79_acc += (v65_bc * v38_data);
              v79_acc += (v67_bc * v39_data);
              v79_acc += (v69_bc * v40_data);
              v79_acc += (v71_bc * v41_data);
              v79_acc += (v73_bc * v42_data);
              v79_acc += (v75_bc * v43_data);
              v79_acc += (v77_bc * v44_data);
              ir1.template select<16, 1>(16) = v79_acc;
              tensorforge::intel_esimd::simd<float, 16> v113_acc{};
              v113_acc += (v47_bc * v29_data);
              v113_acc += (v49_bc * v30_data);
              v113_acc += (v51_bc * v31_data);
              v113_acc += (v53_bc * v32_data);
              v113_acc += (v55_bc * v33_data);
              v113_acc += (v57_bc * v34_data);
              v113_acc += (v59_bc * v35_data);
              v113_acc += (v61_bc * v36_data);
              v113_acc += (v63_bc * v37_data);
              v113_acc += (v65_bc * v38_data);
              v113_acc += (v67_bc * v39_data);
              v113_acc += (v69_bc * v40_data);
              v113_acc += (v71_bc * v41_data);
              v113_acc += (v73_bc * v42_data);
              v113_acc += (v75_bc * v43_data);
              v113_acc += (v77_bc * v44_data);
              ir1.template select<16, 1>(32) = v113_acc;
              tensorforge::intel_esimd::simd<float, 16> v147_acc{};
              v147_acc += (v47_bc * v29_data);
              v147_acc += (v49_bc * v30_data);
              v147_acc += (v51_bc * v31_data);
              v147_acc += (v53_bc * v32_data);
              v147_acc += (v55_bc * v33_data);
              v147_acc += (v57_bc * v34_data);
              v147_acc += (v59_bc * v35_data);
              v147_acc += (v61_bc * v36_data);
              v147_acc += (v63_bc * v37_data);
              v147_acc += (v65_bc * v38_data);
              v147_acc += (v67_bc * v39_data);
              v147_acc += (v69_bc * v40_data);
              v147_acc += (v71_bc * v41_data);
              v147_acc += (v73_bc * v42_data);
              v147_acc += (v75_bc * v43_data);
              v147_acc += (v77_bc * v44_data);
              ir1.template select<16, 1>(48) = v147_acc;
              tensorforge::intel_esimd::simd<float, 16> v181_acc{};
              v181_acc += (v47_bc * v29_data);
              v181_acc += (v49_bc * v30_data);
              v181_acc += (v51_bc * v31_data);
              v181_acc += (v53_bc * v32_data);
              v181_acc += (v55_bc * v33_data);
              v181_acc += (v57_bc * v34_data);
              v181_acc += (v59_bc * v35_data);
              v181_acc += (v61_bc * v36_data);
              v181_acc += (v63_bc * v37_data);
              v181_acc += (v65_bc * v38_data);
              v181_acc += (v67_bc * v39_data);
              v181_acc += (v69_bc * v40_data);
              v181_acc += (v71_bc * v41_data);
              v181_acc += (v73_bc * v42_data);
              v181_acc += (v75_bc * v43_data);
              v181_acc += (v77_bc * v44_data);
              ir1.template select<16, 1>(64) = v181_acc;
              tensorforge::intel_esimd::simd<float, 16> v215_acc{};
              v215_acc += (v47_bc * v29_data);
              v215_acc += (v49_bc * v30_data);
              v215_acc += (v51_bc * v31_data);
              v215_acc += (v53_bc * v32_data);
              v215_acc += (v55_bc * v33_data);
              v215_acc += (v57_bc * v34_data);
              v215_acc += (v59_bc * v35_data);
              v215_acc += (v61_bc * v36_data);
              v215_acc += (v63_bc * v37_data);
              v215_acc += (v65_bc * v38_data);
              v215_acc += (v67_bc * v39_data);
              v215_acc += (v69_bc * v40_data);
              v215_acc += (v71_bc * v41_data);
              v215_acc += (v73_bc * v42_data);
              v215_acc += (v75_bc * v43_data);
              v215_acc += (v77_bc * v44_data);
              ir1.template select<16, 1>(80) = v215_acc;
              tensorforge::intel_esimd::simd<float, 16> v249_acc{};
              v249_acc += (v47_bc * v29_data);
              v249_acc += (v49_bc * v30_data);
              v249_acc += (v51_bc * v31_data);
              v249_acc += (v53_bc * v32_data);
              v249_acc += (v55_bc * v33_data);
              v249_acc += (v57_bc * v34_data);
              v249_acc += (v59_bc * v35_data);
              v249_acc += (v61_bc * v36_data);
              v249_acc += (v63_bc * v37_data);
              v249_acc += (v65_bc * v38_data);
              v249_acc += (v67_bc * v39_data);
              v249_acc += (v69_bc * v40_data);
              v249_acc += (v71_bc * v41_data);
              v249_acc += (v73_bc * v42_data);
              v249_acc += (v75_bc * v43_data);
              v249_acc += (v77_bc * v44_data);
              ir1.template select<16, 1>(96) = v249_acc;
              tensorforge::intel_esimd::simd<float, 16> v283_acc{};
              v283_acc += (v47_bc * v29_data);
              v283_acc += (v49_bc * v30_data);
              v283_acc += (v51_bc * v31_data);
              v283_acc += (v53_bc * v32_data);
              v283_acc += (v55_bc * v33_data);
              v283_acc += (v57_bc * v34_data);
              v283_acc += (v59_bc * v35_data);
              v283_acc += (v61_bc * v36_data);
              v283_acc += (v63_bc * v37_data);
              v283_acc += (v65_bc * v38_data);
              v283_acc += (v67_bc * v39_data);
              v283_acc += (v69_bc * v40_data);
              v283_acc += (v71_bc * v41_data);
              v283_acc += (v73_bc * v42_data);
              v283_acc += (v75_bc * v43_data);
              v283_acc += (v77_bc * v44_data);
              ir1.template select<16, 1>(112) = v283_acc;
              tensorforge::intel_esimd::simd<float, 16> v317_acc{};
              v317_acc += (v47_bc * v29_data);
              v317_acc += (v49_bc * v30_data);
              v317_acc += (v51_bc * v31_data);
              v317_acc += (v53_bc * v32_data);
              v317_acc += (v55_bc * v33_data);
              v317_acc += (v57_bc * v34_data);
              v317_acc += (v59_bc * v35_data);
              v317_acc += (v61_bc * v36_data);
              v317_acc += (v63_bc * v37_data);
              v317_acc += (v65_bc * v38_data);
              v317_acc += (v67_bc * v39_data);
              v317_acc += (v69_bc * v40_data);
              v317_acc += (v71_bc * v41_data);
              v317_acc += (v73_bc * v42_data);
              v317_acc += (v75_bc * v43_data);
              v317_acc += (v77_bc * v44_data);
              ir1.template select<16, 1>(128) = v317_acc;
              tensorforge::intel_esimd::simd<float, 16> v351_acc{};
              v351_acc += (v47_bc * v29_data);
              v351_acc += (v49_bc * v30_data);
              v351_acc += (v51_bc * v31_data);
              v351_acc += (v53_bc * v32_data);
              v351_acc += (v55_bc * v33_data);
              v351_acc += (v57_bc * v34_data);
              v351_acc += (v59_bc * v35_data);
              v351_acc += (v61_bc * v36_data);
              v351_acc += (v63_bc * v37_data);
              v351_acc += (v65_bc * v38_data);
              v351_acc += (v67_bc * v39_data);
              v351_acc += (v69_bc * v40_data);
              v351_acc += (v71_bc * v41_data);
              v351_acc += (v73_bc * v42_data);
              v351_acc += (v75_bc * v43_data);
              v351_acc += (v77_bc * v44_data);
              ir1.template select<16, 1>(144) = v351_acc;
              tensorforge::intel_esimd::simd<float, 16> v385_acc{};
              v385_acc += (v47_bc * v29_data);
              v385_acc += (v49_bc * v30_data);
              v385_acc += (v51_bc * v31_data);
              v385_acc += (v53_bc * v32_data);
              v385_acc += (v55_bc * v33_data);
              v385_acc += (v57_bc * v34_data);
              v385_acc += (v59_bc * v35_data);
              v385_acc += (v61_bc * v36_data);
              v385_acc += (v63_bc * v37_data);
              v385_acc += (v65_bc * v38_data);
              v385_acc += (v67_bc * v39_data);
              v385_acc += (v69_bc * v40_data);
              v385_acc += (v71_bc * v41_data);
              v385_acc += (v73_bc * v42_data);
              v385_acc += (v75_bc * v43_data);
              v385_acc += (v77_bc * v44_data);
              ir1.template select<16, 1>(160) = v385_acc;
              tensorforge::intel_esimd::simd<float, 16> v419_acc{};
              v419_acc += (v47_bc * v29_data);
              v419_acc += (v49_bc * v30_data);
              v419_acc += (v51_bc * v31_data);
              v419_acc += (v53_bc * v32_data);
              v419_acc += (v55_bc * v33_data);
              v419_acc += (v57_bc * v34_data);
              v419_acc += (v59_bc * v35_data);
              v419_acc += (v61_bc * v36_data);
              v419_acc += (v63_bc * v37_data);
              v419_acc += (v65_bc * v38_data);
              v419_acc += (v67_bc * v39_data);
              v419_acc += (v69_bc * v40_data);
              v419_acc += (v71_bc * v41_data);
              v419_acc += (v73_bc * v42_data);
              v419_acc += (v75_bc * v43_data);
              v419_acc += (v77_bc * v44_data);
              ir1.template select<16, 1>(176) = v419_acc;
              tensorforge::intel_esimd::simd<float, 16> v453_acc{};
              v453_acc += (v47_bc * v29_data);
              v453_acc += (v49_bc * v30_data);
              v453_acc += (v51_bc * v31_data);
              v453_acc += (v53_bc * v32_data);
              v453_acc += (v55_bc * v33_data);
              v453_acc += (v57_bc * v34_data);
              v453_acc += (v59_bc * v35_data);
              v453_acc += (v61_bc * v36_data);
              v453_acc += (v63_bc * v37_data);
              v453_acc += (v65_bc * v38_data);
              v453_acc += (v67_bc * v39_data);
              v453_acc += (v69_bc * v40_data);
              v453_acc += (v71_bc * v41_data);
              v453_acc += (v73_bc * v42_data);
              v453_acc += (v75_bc * v43_data);
              v453_acc += (v77_bc * v44_data);
              ir1.template select<16, 1>(192) = v453_acc;
              tensorforge::intel_esimd::simd<float, 16> v487_acc{};
              v487_acc += (v47_bc * v29_data);
              v487_acc += (v49_bc * v30_data);
              v487_acc += (v51_bc * v31_data);
              v487_acc += (v53_bc * v32_data);
              v487_acc += (v55_bc * v33_data);
              v487_acc += (v57_bc * v34_data);
              v487_acc += (v59_bc * v35_data);
              v487_acc += (v61_bc * v36_data);
              v487_acc += (v63_bc * v37_data);
              v487_acc += (v65_bc * v38_data);
              v487_acc += (v67_bc * v39_data);
              v487_acc += (v69_bc * v40_data);
              v487_acc += (v71_bc * v41_data);
              v487_acc += (v73_bc * v42_data);
              v487_acc += (v75_bc * v43_data);
              v487_acc += (v77_bc * v44_data);
              ir1.template select<16, 1>(208) = v487_acc;
              tensorforge::intel_esimd::simd<float, 16> v521_acc{};
              v521_acc += (v47_bc * v29_data);
              v521_acc += (v49_bc * v30_data);
              v521_acc += (v51_bc * v31_data);
              v521_acc += (v53_bc * v32_data);
              v521_acc += (v55_bc * v33_data);
              v521_acc += (v57_bc * v34_data);
              v521_acc += (v59_bc * v35_data);
              v521_acc += (v61_bc * v36_data);
              v521_acc += (v63_bc * v37_data);
              v521_acc += (v65_bc * v38_data);
              v521_acc += (v67_bc * v39_data);
              v521_acc += (v69_bc * v40_data);
              v521_acc += (v71_bc * v41_data);
              v521_acc += (v73_bc * v42_data);
              v521_acc += (v75_bc * v43_data);
              v521_acc += (v77_bc * v44_data);
              ir1.template select<16, 1>(224) = v521_acc;
              tensorforge::intel_esimd::simd<float, 16> v555_acc{};
              v555_acc += (v47_bc * v29_data);
              v555_acc += (v49_bc * v30_data);
              v555_acc += (v51_bc * v31_data);
              v555_acc += (v53_bc * v32_data);
              v555_acc += (v55_bc * v33_data);
              v555_acc += (v57_bc * v34_data);
              v555_acc += (v59_bc * v35_data);
              v555_acc += (v61_bc * v36_data);
              v555_acc += (v63_bc * v37_data);
              v555_acc += (v65_bc * v38_data);
              v555_acc += (v67_bc * v39_data);
              v555_acc += (v69_bc * v40_data);
              v555_acc += (v71_bc * v41_data);
              v555_acc += (v73_bc * v42_data);
              v555_acc += (v75_bc * v43_data);
              v555_acc += (v77_bc * v44_data);
              ir1.template select<16, 1>(240) = v555_acc;
              // r1 = ir1
              #pragma unroll
              for (int32_t v589_n0 = 0; v589_n0 < 1; ++v589_n0) {
                int32_t v591_a = v589_n0 * 16;
                #pragma unroll
                for (int32_t v590_n1 = 0; v590_n1 < 16; ++v590_n1) {
                  int32_t v593_a = v591_a + (v590_n1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v594_data(ir1.template select<16, 1>(v593_a));
                  r1.template select<16, 1>(v593_a) = v594_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v595_i0 = 0; v595_i0 < 1; ++v595_i0) {
                int32_t v597_a = v595_i0 * 16;
                #pragma unroll
                for (int32_t v596_i1 = 0; v596_i1 < 16; ++v596_i1) {
                  int32_t v599_a = v597_a + (v596_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v600_data(r1.template select<16, 1>(v599_a));
                  v600_data.copy_to(glb_m0 + (v599_a));
                }
              }
            }
          }
        }
      }
    });
  });
}

