// === base name ===
kernel_961b3b08ae3213d2

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_961b3b08ae3213d2 = {{16, 16, 1}, 16, 20, 1, 16, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_961b3b08ae3213d2(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_961b3b08ae3213d2(const double ** m0, size_t m0_extraOffset, const double ** m1, size_t m1_extraOffset, double ** m2, size_t m2_extraOffset, const double * m3, const double ** m4, size_t m4_extraOffset, const double * m5, const double ** m6, size_t m6_extraOffset, const double * m7, const double ** m8, size_t m8_extraOffset, const double * m9, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_961b3b08ae3213d2(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (16, 16, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 16 - 1) / 16;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 16;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 256 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_961b3b08ae3213d2(const double ** m0, size_t m0_extraOffset, const double ** m1, size_t m1_extraOffset, double ** m2, size_t m2_extraOffset, const double * m3, const double ** m4, size_t m4_extraOffset, const double * m5, const double ** m6, size_t m6_extraOffset, const double * m7, const double ** m8, size_t m8_extraOffset, const double * m9, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_961b3b08ae3213d2(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_961b3b08ae3213d2(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m4, m4_extraOffset, m5, m6, m6_extraOffset, m7, m8, m8_extraOffset, m9, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_961b3b08ae3213d2(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const double ** m0, size_t m0_extraOffset, const double ** m1, size_t m1_extraOffset, double ** m2, size_t m2_extraOffset, const double * m3, const double ** m4, size_t m4_extraOffset, const double * m5, const double ** m6, size_t m6_extraOffset, const double * m7, const double ** m8, size_t m8_extraOffset, const double * m9, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: amd_matrix_k4=1,lanes_per_mult=16,merge_variants=1,mma_d_layout=1,mma_d_layout_loops=1
        // launch: 16 lanes (20 active) x 16 per block = block 16x16x1, 2048 B shared, occupancy grid
        // operands:
        //   v0 9×9(9×9) {0..9}×{0..9} pointer_based
        //   v1 20×10(20×10) {0..20}×{0..10} none
        //   m0 20×9(20×9) {0..20}×{0..9} pointer_based
        //   m1 9×9(9×9) {0..9}×{0..9} pointer_based
        //   m2 20×9(20×9) {0..20}×{0..9} pointer_based
        //   m3 20×10(20×10) {0..20}×{0..10} none ordered
        //   m4 9×9(9×9) {0..9}×{0..9} pointer_based
        //   m5 20×10(20×10) {0..20}×{0..10} none ordered
        //   m6 9×9(9×9) {0..9}×{0..9} pointer_based
        //   m7 20×10(20×10) {0..20}×{0..10} none ordered
        //   m8 9×9(9×9) {0..9}×{0..9} pointer_based
        //   m9 20×10(20×10) {0..20}×{0..10} none ordered
        // operations:
        //   for 4 iterations (any order):
        //     t0[i,j] = m0[i,k]@{0..10}×{0..9} × v0[k,j]
        //     m2[i,j] += v1[i,k] × t0[k,j]
        //     v0 ∈ {m1, m4, m6, m8}
        //     v1 ∈ {m3, m5, m7, m9}
        // tensorforge-meta: {"fp":"double","launch":{"active_threads":20,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":2048,"shared_elements":256,"threads_per_mult":16},"loops":[{"body":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"v0","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,10]],"is_tmp":false,"name":"v1","offset":[0,0],"shape":[20,10]},{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"holes":[{"members":["m1","m4","m6","m8"],"stand_in":"v0"},{"members":["m3","m5","m7","m9"],"stand_in":"v1"}],"iterations":4,"kind":"for","sequential":false}],"operands":[{"addressing":"pointer_based","alias":"variant0","bbox":[[0,0],[9,9]],"name":"v0","ordered":false,"parts":1,"shape":[9,9],"variant":true},{"addressing":"none","alias":"variant1","bbox":[[0,0],[20,10]],"name":"v1","ordered":false,"parts":1,"shape":[20,10],"variant":true},{"addressing":"pointer_based","alias":"I","bbox":[[0,0],[20,9]],"name":"m0","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"pointer_based","alias":"S0","bbox":[[0,0],[9,9]],"name":"m1","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"pointer_based","alias":"Q","bbox":[[0,0],[20,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"none","alias":"K0","bbox":[[0,0],[20,10]],"name":"m3","ordered":true,"parts":1,"shape":[20,10],"variant":false},{"addressing":"pointer_based","alias":"S1","bbox":[[0,0],[9,9]],"name":"m4","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"none","alias":"K1","bbox":[[0,0],[20,10]],"name":"m5","ordered":true,"parts":1,"shape":[20,10],"variant":false},{"addressing":"pointer_based","alias":"S2","bbox":[[0,0],[9,9]],"name":"m6","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"none","alias":"K2","bbox":[[0,0],[20,10]],"name":"m7","ordered":true,"parts":1,"shape":[20,10],"variant":false},{"addressing":"pointer_based","alias":"S3","bbox":[[0,0],[9,9]],"name":"m8","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"none","alias":"K3","bbox":[[0,0],[20,10]],"name":"m9","ordered":true,"parts":1,"shape":[20,10],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,10]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[20,10]},{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,10]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[20,10]},{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,10]],"is_tmp":false,"name":"m7","offset":[0,0],"shape":[20,10]},{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[20,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m8","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"pointer_based","bbox":[[0,0],[20,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[20,10]],"is_tmp":false,"name":"m9","offset":[0,0],"shape":[20,10]},{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[10,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          double* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const double *const __restrict__ glb_m3 = &m3[0];
          const double *const __restrict__ glb_m5 = &m5[0];
          const double *const __restrict__ glb_m7 = &m7[0];
          const double *const __restrict__ glb_m9 = &m9[0];
          double r0[9]{};
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v1895_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v1895_flagSecondAhead < numElements0) ? v1895_flagSecondAhead : v3_batchId1)];
          const bool allowed_peel = static_cast<bool>(flagWordFirst);
          const double *const __restrict__ peel_glb_m0 = &m0[v3_batchId1][0 + m0_extraOffset];
          int32_t v1905_lead = item.get_local_id(2) % 16;
          bool v1906_g = v1905_lead < 10;
          bool v1923_g = allowed_peel && v1906_g;
          // r0 = load{g>r}(peel_glb_m0);
          #pragma unroll
          for (int32_t v1915_i1 = 0; v1915_i1 < 9; ++v1915_i1) {
            double v1918_data_pre = peel_glb_m0[v1923_g ? ((v1905_lead + (v1915_i1 * 20))) : (0)];
            double v1918_data = v1923_g ? (v1918_data_pre) : (0.0);
            if (v1906_g) {
              r0[v1915_i1] = v1918_data;
            }
          }
          uint32_t v1899_flagWord = flagWordFirst;
          uint32_t v1900_flagWordNext = flagWordSecond;
          for (size_t v11_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v11_batchId0 < numElements0; v11_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v12_ahead1 = v11_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
            const double *const __restrict__ wrap_glb_m0 = &m0[v14_batchId1][0 + m0_extraOffset];
            size_t v1928_flagAheadAhead = v14_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v1928_flagAheadAhead < numElements0) ? v1928_flagAheadAhead : v14_batchId1)];
            const bool allowed = static_cast<bool>(v1899_flagWord);
            const bool allowed_next = static_cast<bool>(v1900_flagWordNext);
            if (allowed) {
              const double *const __restrict__ glb_m1 = &m1[v11_batchId0][0 + m1_extraOffset];
              double *const __restrict__ glb_m2 = &m2[v11_batchId0][0 + m2_extraOffset];
              const double *const __restrict__ glb_m4 = &m4[v11_batchId0][0 + m4_extraOffset];
              const double *const __restrict__ glb_m6 = &m6[v11_batchId0][0 + m6_extraOffset];
              const double *const __restrict__ glb_m8 = &m8[v11_batchId0][0 + m8_extraOffset];
              bool v49_g = v1905_lead < 9;
              int32_t v1840_lead = v1905_lead + 16_i32;
              bool v1843_g = v1905_lead < 4;
              int32_t v557_a = v1905_lead * 2;
              int32_t v685_a = 32 + v557_a;
              int32_t v812_a = 64 + v557_a;
              int32_t v939_a = 96 + v557_a;
              int32_t v1066_a = 128 + v557_a;
              int32_t v1193_a = 160 + v557_a;
              int32_t v1320_a = 192 + v557_a;
              int32_t v1447_a = 224 + v557_a;
              int32_t v1574_a = 256 + v557_a;
              int32_t v1701_a = 288 + v557_a;
              double r1[9]{};
              const double *const peel_v0Table = (0 == 0) ? glb_m1 : (0 == 1) ? glb_m4 : (0 == 2) ? glb_m6 : glb_m8;
              const double *const __restrict__ peel_glb_v0 = peel_v0Table;
              // r1 = load{g>r}(peel_glb_v0);
              #pragma unroll
              for (int32_t v1886_i1 = 0; v1886_i1 < 9; ++v1886_i1) {
                double v1889_data_pre = peel_glb_v0[v49_g ? ((v1905_lead + (v1886_i1 * 9))) : (0)];
                double v1889_data = v49_g ? (v1889_data_pre) : (0.0);
                r1[v1886_i1] = v1889_data;
              }
              for (int32_t batchIdv6 = 0; batchIdv6 < 4; ++batchIdv6) {
                int32_t v1879_tripAhead = batchIdv6 + 1;
                int32_t v1881_trip = (v1879_tripAhead < 4) ? v1879_tripAhead : batchIdv6;
                const double *const wrap_v0Table = (v1881_trip == 0) ? glb_m1 : (v1881_trip == 1) ? glb_m4 : (v1881_trip == 2) ? glb_m6 : glb_m8;
                const double *const __restrict__ wrap_glb_v0 = wrap_v0Table;
                const double *const v1Table = (batchIdv6 == 0) ? glb_m3 : (batchIdv6 == 1) ? glb_m5 : (batchIdv6 == 2) ? glb_m7 : glb_m9;
                const double *const __restrict__ glb_v1 = v1Table;
                double r4[18]{};
                double r2[9]{};
                // r2 = +(r0 * r1) + None
                // [(0, 10), (0, 9)] [(0, 9)]
                double v53_data = r0[0];
                double v54_data = r1[0];
                double v55_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
                double v60_data = r1[1];
                double v61_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
                double v66_data = r1[2];
                double v67_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
                double v72_data = r1[3];
                double v73_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
                double v78_data = r1[4];
                double v79_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
                double v84_data = r1[5];
                double v85_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
                double v90_data = r1[6];
                double v91_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
                double v96_data = r1[7];
                double v97_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
                double v102_data = r1[8];
                double v103_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
                double v107_data = r0[1];
                double v109_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
                double v115_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
                double v121_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
                double v127_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
                double v133_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
                double v139_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
                double v145_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
                double v151_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
                double v157_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
                double v161_data = r0[2];
                double v163_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
                double v169_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
                double v175_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
                double v181_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
                double v187_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
                double v193_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
                double v199_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
                double v205_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
                double v211_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
                double v215_data = r0[3];
                double v217_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
                double v223_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
                double v229_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
                double v235_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
                double v241_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
                double v247_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
                double v253_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
                double v259_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
                double v265_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
                double v269_data = r0[4];
                double v271_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
                double v277_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
                double v283_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
                double v289_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
                double v295_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
                double v301_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
                double v307_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
                double v313_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
                double v319_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
                double v323_data = r0[5];
                double v325_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
                double v331_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
                double v337_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
                double v343_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
                double v349_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
                double v355_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
                double v361_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
                double v367_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
                double v373_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
                double v377_data = r0[6];
                double v379_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
                double v385_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
                double v391_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
                double v397_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
                double v403_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
                double v409_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
                double v415_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
                double v421_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
                double v427_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
                double v431_data = r0[7];
                double v433_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
                double v439_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
                double v442_p = (((((((v53_data * v61_bc) + (v107_data * v115_bc)) + (v161_data * v169_bc)) + (v215_data * v223_bc)) + (v269_data * v277_bc)) + (v323_data * v331_bc)) + (v377_data * v385_bc)) + (v431_data * v439_bc);
                double v445_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
                double v448_p = (((((((v53_data * v67_bc) + (v107_data * v121_bc)) + (v161_data * v175_bc)) + (v215_data * v229_bc)) + (v269_data * v283_bc)) + (v323_data * v337_bc)) + (v377_data * v391_bc)) + (v431_data * v445_bc);
                double v451_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
                double v454_p = (((((((v53_data * v73_bc) + (v107_data * v127_bc)) + (v161_data * v181_bc)) + (v215_data * v235_bc)) + (v269_data * v289_bc)) + (v323_data * v343_bc)) + (v377_data * v397_bc)) + (v431_data * v451_bc);
                double v457_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
                double v460_p = (((((((v53_data * v79_bc) + (v107_data * v133_bc)) + (v161_data * v187_bc)) + (v215_data * v241_bc)) + (v269_data * v295_bc)) + (v323_data * v349_bc)) + (v377_data * v403_bc)) + (v431_data * v457_bc);
                double v463_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
                double v466_p = (((((((v53_data * v85_bc) + (v107_data * v139_bc)) + (v161_data * v193_bc)) + (v215_data * v247_bc)) + (v269_data * v301_bc)) + (v323_data * v355_bc)) + (v377_data * v409_bc)) + (v431_data * v463_bc);
                double v469_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
                double v472_p = (((((((v53_data * v91_bc) + (v107_data * v145_bc)) + (v161_data * v199_bc)) + (v215_data * v253_bc)) + (v269_data * v307_bc)) + (v323_data * v361_bc)) + (v377_data * v415_bc)) + (v431_data * v469_bc);
                double v475_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
                double v478_p = (((((((v53_data * v97_bc) + (v107_data * v151_bc)) + (v161_data * v205_bc)) + (v215_data * v259_bc)) + (v269_data * v313_bc)) + (v323_data * v367_bc)) + (v377_data * v421_bc)) + (v431_data * v475_bc);
                double v481_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
                double v484_p = (((((((v53_data * v103_bc) + (v107_data * v157_bc)) + (v161_data * v211_bc)) + (v215_data * v265_bc)) + (v269_data * v319_bc)) + (v323_data * v373_bc)) + (v377_data * v427_bc)) + (v431_data * v481_bc);
                double v485_data = r0[8];
                double v487_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
                r2[0] = (((((((((v53_data * v55_bc) + (v107_data * v109_bc)) + (v161_data * v163_bc)) + (v215_data * v217_bc)) + (v269_data * v271_bc)) + (v323_data * v325_bc)) + (v377_data * v379_bc)) + (v431_data * v433_bc)) + (v485_data * v487_bc));
                double v493_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
                r2[1] = (v442_p + (v485_data * v493_bc));
                double v499_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
                r2[2] = (v448_p + (v485_data * v499_bc));
                double v505_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
                r2[3] = (v454_p + (v485_data * v505_bc));
                double v511_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
                r2[4] = (v460_p + (v485_data * v511_bc));
                double v517_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
                r2[5] = (v466_p + (v485_data * v517_bc));
                double v523_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
                r2[6] = (v472_p + (v485_data * v523_bc));
                double v529_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
                r2[7] = (v478_p + (v485_data * v529_bc));
                double v535_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
                r2[8] = (v484_p + (v485_data * v535_bc));
                // r2 = 0 past (0, 10)
                // s0 = store{r>s}(localShrMem0, r2);
                double r3[9]{};
                // r3 = load{g>r}(s0);
                double r5[18]{};
                // ir5 = +(glb_v1 * r3)
                // [(0, 20), (0, 9)] [(0, 10)]
                double ir5[18]{};
                sycl::vec<double, 2> v559_data = *(sycl::vec<double, 2>*)&glb_v1[v557_a];
                double v560_p = v559_data[0];
                double v561_data = r2[0];
                double v562_bc = sycl::select_from_group(item.get_sub_group(), v561_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
                double v568_data = r2[1];
                double v569_bc = sycl::select_from_group(item.get_sub_group(), v568_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
                double v575_data = r2[2];
                double v576_bc = sycl::select_from_group(item.get_sub_group(), v575_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
                double v582_data = r2[3];
                double v583_bc = sycl::select_from_group(item.get_sub_group(), v582_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
                double v589_data = r2[4];
                double v590_bc = sycl::select_from_group(item.get_sub_group(), v589_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
                double v596_data = r2[5];
                double v597_bc = sycl::select_from_group(item.get_sub_group(), v596_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
                double v603_data = r2[6];
                double v604_bc = sycl::select_from_group(item.get_sub_group(), v603_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
                double v610_data = r2[7];
                double v611_bc = sycl::select_from_group(item.get_sub_group(), v610_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
                double v617_data = r2[8];
                double v618_bc = sycl::select_from_group(item.get_sub_group(), v617_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
                double v623_p = v559_data[1];
                sycl::vec<double, 2> v686_data = *(sycl::vec<double, 2>*)&glb_v1[v685_a];
                double v687_p = v686_data[0];
                double v689_bc = sycl::select_from_group(item.get_sub_group(), v561_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
                double v696_bc = sycl::select_from_group(item.get_sub_group(), v568_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
                double v703_bc = sycl::select_from_group(item.get_sub_group(), v575_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
                double v710_bc = sycl::select_from_group(item.get_sub_group(), v582_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
                double v717_bc = sycl::select_from_group(item.get_sub_group(), v589_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
                double v724_bc = sycl::select_from_group(item.get_sub_group(), v596_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
                double v731_bc = sycl::select_from_group(item.get_sub_group(), v603_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
                double v738_bc = sycl::select_from_group(item.get_sub_group(), v610_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
                double v745_bc = sycl::select_from_group(item.get_sub_group(), v617_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
                double v750_p = v686_data[1];
                sycl::vec<double, 2> v813_data = *(sycl::vec<double, 2>*)&glb_v1[v812_a];
                double v814_p = v813_data[0];
                double v816_bc = sycl::select_from_group(item.get_sub_group(), v561_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
                double v823_bc = sycl::select_from_group(item.get_sub_group(), v568_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
                double v830_bc = sycl::select_from_group(item.get_sub_group(), v575_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
                double v837_bc = sycl::select_from_group(item.get_sub_group(), v582_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
                double v844_bc = sycl::select_from_group(item.get_sub_group(), v589_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
                double v851_bc = sycl::select_from_group(item.get_sub_group(), v596_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
                double v858_bc = sycl::select_from_group(item.get_sub_group(), v603_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
                double v865_bc = sycl::select_from_group(item.get_sub_group(), v610_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
                double v872_bc = sycl::select_from_group(item.get_sub_group(), v617_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
                double v877_p = v813_data[1];
                double v910_p = ((v623_p * v590_bc) + (v750_p * v717_bc)) + (v877_p * v844_bc);
                double v917_p = ((v623_p * v597_bc) + (v750_p * v724_bc)) + (v877_p * v851_bc);
                double v924_p = ((v623_p * v604_bc) + (v750_p * v731_bc)) + (v877_p * v858_bc);
                double v931_p = ((v623_p * v611_bc) + (v750_p * v738_bc)) + (v877_p * v865_bc);
                double v938_p = ((v623_p * v618_bc) + (v750_p * v745_bc)) + (v877_p * v872_bc);
                sycl::vec<double, 2> v940_data = *(sycl::vec<double, 2>*)&glb_v1[v939_a];
                double v941_p = v940_data[0];
                double v943_bc = sycl::select_from_group(item.get_sub_group(), v561_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
                double v946_p = (((v560_p * v562_bc) + (v687_p * v689_bc)) + (v814_p * v816_bc)) + (v941_p * v943_bc);
                double v950_bc = sycl::select_from_group(item.get_sub_group(), v568_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
                double v953_p = (((v560_p * v569_bc) + (v687_p * v696_bc)) + (v814_p * v823_bc)) + (v941_p * v950_bc);
                double v957_bc = sycl::select_from_group(item.get_sub_group(), v575_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
                double v960_p = (((v560_p * v576_bc) + (v687_p * v703_bc)) + (v814_p * v830_bc)) + (v941_p * v957_bc);
                double v964_bc = sycl::select_from_group(item.get_sub_group(), v582_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
                double v967_p = (((v560_p * v583_bc) + (v687_p * v710_bc)) + (v814_p * v837_bc)) + (v941_p * v964_bc);
                double v971_bc = sycl::select_from_group(item.get_sub_group(), v589_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
                double v974_p = (((v560_p * v590_bc) + (v687_p * v717_bc)) + (v814_p * v844_bc)) + (v941_p * v971_bc);
                double v978_bc = sycl::select_from_group(item.get_sub_group(), v596_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
                double v981_p = (((v560_p * v597_bc) + (v687_p * v724_bc)) + (v814_p * v851_bc)) + (v941_p * v978_bc);
                double v985_bc = sycl::select_from_group(item.get_sub_group(), v603_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
                double v988_p = (((v560_p * v604_bc) + (v687_p * v731_bc)) + (v814_p * v858_bc)) + (v941_p * v985_bc);
                double v992_bc = sycl::select_from_group(item.get_sub_group(), v610_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
                double v995_p = (((v560_p * v611_bc) + (v687_p * v738_bc)) + (v814_p * v865_bc)) + (v941_p * v992_bc);
                double v999_bc = sycl::select_from_group(item.get_sub_group(), v617_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
                double v1002_p = (((v560_p * v618_bc) + (v687_p * v745_bc)) + (v814_p * v872_bc)) + (v941_p * v999_bc);
                double v1004_p = v940_data[1];
                double v1009_p = (((v623_p * v562_bc) + (v750_p * v689_bc)) + (v877_p * v816_bc)) + (v1004_p * v943_bc);
                double v1016_p = (((v623_p * v569_bc) + (v750_p * v696_bc)) + (v877_p * v823_bc)) + (v1004_p * v950_bc);
                double v1023_p = (((v623_p * v576_bc) + (v750_p * v703_bc)) + (v877_p * v830_bc)) + (v1004_p * v957_bc);
                double v1030_p = (((v623_p * v583_bc) + (v750_p * v710_bc)) + (v877_p * v837_bc)) + (v1004_p * v964_bc);
                double v1035_p = v1004_p * v971_bc;
                // r4 = load{g>r}(glb_m2);
                #pragma unroll
                for (int32_t v1829_i0 = 0; v1829_i0 < 1; ++v1829_i0) {
                  int32_t v1832_lead = v1905_lead + (v1829_i0 * 16);
                  #pragma unroll
                  for (int32_t v1830_i1 = 0; v1830_i1 < 9; ++v1830_i1) {
                    double v1835_data = glb_m2[(v1832_lead + (v1830_i1 * 20))];
                    r4[(v1829_i0 + (v1830_i1 * 2))] = v1835_data;
                  }
                }
                #pragma unroll
                for (int32_t v1838_i1 = 0; v1838_i1 < 9; ++v1838_i1) {
                  double v1844_data_pre = glb_m2[v1843_g ? ((v1840_lead + (v1838_i1 * 20))) : (0)];
                  double v1844_data = v1843_g ? (v1844_data_pre) : (0.0);
                  r4[(1 + (v1838_i1 * 2))] = v1844_data;
                }
                sycl::vec<double, 2> v1067_data = *(sycl::vec<double, 2>*)&glb_v1[v1066_a];
                double v1068_p = v1067_data[0];
                double v1070_bc = sycl::select_from_group(item.get_sub_group(), v561_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
                double v1077_bc = sycl::select_from_group(item.get_sub_group(), v568_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
                double v1084_bc = sycl::select_from_group(item.get_sub_group(), v575_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
                double v1091_bc = sycl::select_from_group(item.get_sub_group(), v582_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
                double v1098_bc = sycl::select_from_group(item.get_sub_group(), v589_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
                double v1105_bc = sycl::select_from_group(item.get_sub_group(), v596_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
                double v1112_bc = sycl::select_from_group(item.get_sub_group(), v603_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
                double v1119_bc = sycl::select_from_group(item.get_sub_group(), v610_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
                double v1126_bc = sycl::select_from_group(item.get_sub_group(), v617_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
                double v1131_p = v1067_data[1];
                sycl::vec<double, 2> v1194_data = *(sycl::vec<double, 2>*)&glb_v1[v1193_a];
                double v1195_p = v1194_data[0];
                double v1197_bc = sycl::select_from_group(item.get_sub_group(), v561_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
                double v1204_bc = sycl::select_from_group(item.get_sub_group(), v568_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
                double v1211_bc = sycl::select_from_group(item.get_sub_group(), v575_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
                double v1218_bc = sycl::select_from_group(item.get_sub_group(), v582_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
                double v1225_bc = sycl::select_from_group(item.get_sub_group(), v589_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
                double v1232_bc = sycl::select_from_group(item.get_sub_group(), v596_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
                double v1239_bc = sycl::select_from_group(item.get_sub_group(), v603_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
                double v1246_bc = sycl::select_from_group(item.get_sub_group(), v610_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
                double v1253_bc = sycl::select_from_group(item.get_sub_group(), v617_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
                double v1258_p = v1194_data[1];
                double v1270_p = (v1016_p + (v1131_p * v1077_bc)) + (v1258_p * v1204_bc);
                double v1277_p = (v1023_p + (v1131_p * v1084_bc)) + (v1258_p * v1211_bc);
                double v1284_p = (v1030_p + (v1131_p * v1091_bc)) + (v1258_p * v1218_bc);
                double v1291_p = ((v910_p + v1035_p) + (v1131_p * v1098_bc)) + (v1258_p * v1225_bc);
                double v1298_p = ((v917_p + (v1004_p * v978_bc)) + (v1131_p * v1105_bc)) + (v1258_p * v1232_bc);
                double v1305_p = ((v924_p + (v1004_p * v985_bc)) + (v1131_p * v1112_bc)) + (v1258_p * v1239_bc);
                double v1312_p = ((v931_p + (v1004_p * v992_bc)) + (v1131_p * v1119_bc)) + (v1258_p * v1246_bc);
                double v1319_p = ((v938_p + (v1004_p * v999_bc)) + (v1131_p * v1126_bc)) + (v1258_p * v1253_bc);
                sycl::vec<double, 2> v1321_data = *(sycl::vec<double, 2>*)&glb_v1[v1320_a];
                double v1322_p = v1321_data[0];
                double v1324_bc = sycl::select_from_group(item.get_sub_group(), v561_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
                double v1327_p = ((v946_p + (v1068_p * v1070_bc)) + (v1195_p * v1197_bc)) + (v1322_p * v1324_bc);
                double v1331_bc = sycl::select_from_group(item.get_sub_group(), v568_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
                double v1334_p = ((v953_p + (v1068_p * v1077_bc)) + (v1195_p * v1204_bc)) + (v1322_p * v1331_bc);
                double v1338_bc = sycl::select_from_group(item.get_sub_group(), v575_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
                double v1341_p = ((v960_p + (v1068_p * v1084_bc)) + (v1195_p * v1211_bc)) + (v1322_p * v1338_bc);
                double v1345_bc = sycl::select_from_group(item.get_sub_group(), v582_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
                double v1348_p = ((v967_p + (v1068_p * v1091_bc)) + (v1195_p * v1218_bc)) + (v1322_p * v1345_bc);
                double v1352_bc = sycl::select_from_group(item.get_sub_group(), v589_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
                double v1355_p = ((v974_p + (v1068_p * v1098_bc)) + (v1195_p * v1225_bc)) + (v1322_p * v1352_bc);
                double v1359_bc = sycl::select_from_group(item.get_sub_group(), v596_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
                double v1362_p = ((v981_p + (v1068_p * v1105_bc)) + (v1195_p * v1232_bc)) + (v1322_p * v1359_bc);
                double v1366_bc = sycl::select_from_group(item.get_sub_group(), v603_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
                double v1369_p = ((v988_p + (v1068_p * v1112_bc)) + (v1195_p * v1239_bc)) + (v1322_p * v1366_bc);
                double v1373_bc = sycl::select_from_group(item.get_sub_group(), v610_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
                double v1376_p = ((v995_p + (v1068_p * v1119_bc)) + (v1195_p * v1246_bc)) + (v1322_p * v1373_bc);
                double v1380_bc = sycl::select_from_group(item.get_sub_group(), v617_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
                double v1383_p = ((v1002_p + (v1068_p * v1126_bc)) + (v1195_p * v1253_bc)) + (v1322_p * v1380_bc);
                double v1385_p = v1321_data[1];
                double v1390_p = ((v1009_p + (v1131_p * v1070_bc)) + (v1258_p * v1197_bc)) + (v1385_p * v1324_bc);
                double v1395_p = v1385_p * v1331_bc;
                // r1 = load{g>r}(wrap_glb_v0);
                #pragma unroll
                for (int32_t v1890_i1 = 0; v1890_i1 < 9; ++v1890_i1) {
                  double v1893_data_pre = wrap_glb_v0[v49_g ? ((v1905_lead + (v1890_i1 * 9))) : (0)];
                  double v1893_data = v49_g ? (v1893_data_pre) : (0.0);
                  r1[v1890_i1] = v1893_data;
                }
                sycl::vec<double, 2> v1448_data = *(sycl::vec<double, 2>*)&glb_v1[v1447_a];
                double v1449_p = v1448_data[0];
                double v1451_bc = sycl::select_from_group(item.get_sub_group(), v561_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
                double v1458_bc = sycl::select_from_group(item.get_sub_group(), v568_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
                double v1465_bc = sycl::select_from_group(item.get_sub_group(), v575_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
                double v1472_bc = sycl::select_from_group(item.get_sub_group(), v582_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
                double v1479_bc = sycl::select_from_group(item.get_sub_group(), v589_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
                double v1486_bc = sycl::select_from_group(item.get_sub_group(), v596_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
                double v1493_bc = sycl::select_from_group(item.get_sub_group(), v603_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
                double v1500_bc = sycl::select_from_group(item.get_sub_group(), v610_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
                double v1507_bc = sycl::select_from_group(item.get_sub_group(), v617_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
                double v1512_p = v1448_data[1];
                sycl::vec<double, 2> v1575_data = *(sycl::vec<double, 2>*)&glb_v1[v1574_a];
                double v1576_p = v1575_data[0];
                double v1578_bc = sycl::select_from_group(item.get_sub_group(), v561_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
                double v1585_bc = sycl::select_from_group(item.get_sub_group(), v568_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
                double v1588_p = (v1334_p + (v1449_p * v1458_bc)) + (v1576_p * v1585_bc);
                double v1592_bc = sycl::select_from_group(item.get_sub_group(), v575_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
                double v1595_p = (v1341_p + (v1449_p * v1465_bc)) + (v1576_p * v1592_bc);
                double v1599_bc = sycl::select_from_group(item.get_sub_group(), v582_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
                double v1602_p = (v1348_p + (v1449_p * v1472_bc)) + (v1576_p * v1599_bc);
                double v1606_bc = sycl::select_from_group(item.get_sub_group(), v589_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
                double v1609_p = (v1355_p + (v1449_p * v1479_bc)) + (v1576_p * v1606_bc);
                double v1613_bc = sycl::select_from_group(item.get_sub_group(), v596_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
                double v1616_p = (v1362_p + (v1449_p * v1486_bc)) + (v1576_p * v1613_bc);
                double v1620_bc = sycl::select_from_group(item.get_sub_group(), v603_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
                double v1623_p = (v1369_p + (v1449_p * v1493_bc)) + (v1576_p * v1620_bc);
                double v1627_bc = sycl::select_from_group(item.get_sub_group(), v610_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
                double v1630_p = (v1376_p + (v1449_p * v1500_bc)) + (v1576_p * v1627_bc);
                double v1634_bc = sycl::select_from_group(item.get_sub_group(), v617_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
                double v1637_p = (v1383_p + (v1449_p * v1507_bc)) + (v1576_p * v1634_bc);
                double v1639_p = v1575_data[1];
                double v1644_p = (v1390_p + (v1512_p * v1451_bc)) + (v1639_p * v1578_bc);
                double v1651_p = ((v1270_p + v1395_p) + (v1512_p * v1458_bc)) + (v1639_p * v1585_bc);
                double v1658_p = ((v1277_p + (v1385_p * v1338_bc)) + (v1512_p * v1465_bc)) + (v1639_p * v1592_bc);
                double v1665_p = ((v1284_p + (v1385_p * v1345_bc)) + (v1512_p * v1472_bc)) + (v1639_p * v1599_bc);
                double v1672_p = ((v1291_p + (v1385_p * v1352_bc)) + (v1512_p * v1479_bc)) + (v1639_p * v1606_bc);
                double v1679_p = ((v1298_p + (v1385_p * v1359_bc)) + (v1512_p * v1486_bc)) + (v1639_p * v1613_bc);
                double v1686_p = ((v1305_p + (v1385_p * v1366_bc)) + (v1512_p * v1493_bc)) + (v1639_p * v1620_bc);
                double v1693_p = ((v1312_p + (v1385_p * v1373_bc)) + (v1512_p * v1500_bc)) + (v1639_p * v1627_bc);
                double v1700_p = ((v1319_p + (v1385_p * v1380_bc)) + (v1512_p * v1507_bc)) + (v1639_p * v1634_bc);
                sycl::vec<double, 2> v1702_data = *(sycl::vec<double, 2>*)&glb_v1[v1701_a];
                double v1703_p = v1702_data[0];
                double v1705_bc = sycl::select_from_group(item.get_sub_group(), v561_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
                ir5[0] = (((v1327_p + (v1449_p * v1451_bc)) + (v1576_p * v1578_bc)) + (v1703_p * v1705_bc));
                double v1712_bc = sycl::select_from_group(item.get_sub_group(), v568_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
                ir5[2] = (v1588_p + (v1703_p * v1712_bc));
                double v1719_bc = sycl::select_from_group(item.get_sub_group(), v575_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
                ir5[4] = (v1595_p + (v1703_p * v1719_bc));
                double v1726_bc = sycl::select_from_group(item.get_sub_group(), v582_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
                ir5[6] = (v1602_p + (v1703_p * v1726_bc));
                double v1733_bc = sycl::select_from_group(item.get_sub_group(), v589_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
                ir5[8] = (v1609_p + (v1703_p * v1733_bc));
                double v1740_bc = sycl::select_from_group(item.get_sub_group(), v596_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
                ir5[10] = (v1616_p + (v1703_p * v1740_bc));
                double v1747_bc = sycl::select_from_group(item.get_sub_group(), v603_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
                ir5[12] = (v1623_p + (v1703_p * v1747_bc));
                double v1754_bc = sycl::select_from_group(item.get_sub_group(), v610_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
                ir5[14] = (v1630_p + (v1703_p * v1754_bc));
                double v1761_bc = sycl::select_from_group(item.get_sub_group(), v617_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
                ir5[16] = (v1637_p + (v1703_p * v1761_bc));
                double v1766_p = v1702_data[1];
                ir5[1] = (v1644_p + (v1766_p * v1705_bc));
                ir5[3] = (v1651_p + (v1766_p * v1712_bc));
                ir5[5] = (v1658_p + (v1766_p * v1719_bc));
                ir5[7] = (v1665_p + (v1766_p * v1726_bc));
                ir5[9] = (v1672_p + (v1766_p * v1733_bc));
                ir5[11] = (v1679_p + (v1766_p * v1740_bc));
                ir5[13] = (v1686_p + (v1766_p * v1747_bc));
                ir5[15] = (v1693_p + (v1766_p * v1754_bc));
                ir5[17] = (v1700_p + (v1766_p * v1761_bc));
                // r5 = ir5 + r4
                #pragma unroll
                for (int32_t v1847_n0 = 0; v1847_n0 < 1; ++v1847_n0) {
                  #pragma unroll
                  for (int32_t v1848_n1 = 0; v1848_n1 < 9; ++v1848_n1) {
                    int32_t v1850_a = v1847_n0 + (v1848_n1 * 2);
                    double v1851_data = ir5[v1850_a];
                    double v1852_data = r4[v1850_a];
                    r5[v1850_a] = (v1852_data + v1851_data);
                  }
                }
                #pragma unroll
                for (int32_t v1855_n1 = 0; v1855_n1 < 9; ++v1855_n1) {
                  int32_t v1857_a = 1 + (v1855_n1 * 2);
                  double v1858_data_pre = ir5[v1857_a];
                  double v1858_data = v1843_g ? (v1858_data_pre) : (0.0);
                  double v1859_data_pre = r4[v1857_a];
                  double v1859_data = v1843_g ? (v1859_data_pre) : (0.0);
                  if (v1843_g) {
                    r5[v1857_a] = (v1859_data + v1858_data);
                  }
                }
                // glb_m2 = store{r>g}(r5);
                #pragma unroll
                for (int32_t v1861_i0 = 0; v1861_i0 < 1; ++v1861_i0) {
                  int32_t v1867_lead = v1905_lead + (v1861_i0 * 16);
                  #pragma unroll
                  for (int32_t v1862_i1 = 0; v1862_i1 < 9; ++v1862_i1) {
                    double v1865_data = r5[(v1861_i0 + (v1862_i1 * 2))];
                    glb_m2[(v1867_lead + (v1862_i1 * 20))] = v1865_data;
                  }
                }
                #pragma unroll
                for (int32_t v1871_i1 = 0; v1871_i1 < 9; ++v1871_i1) {
                  double v1874_data_pre = r5[(1 + (v1871_i1 * 2))];
                  double v1874_data = v1843_g ? (v1874_data_pre) : (0.0);
                  if (v1843_g) {
                    glb_m2[(v1840_lead + (v1871_i1 * 20))] = v1874_data;
                  }
                }
              }
              bool v1924_g = allowed_next && v1906_g;
              // r0 = load{g>r}(wrap_glb_m0);
              #pragma unroll
              for (int32_t v1919_i1 = 0; v1919_i1 < 9; ++v1919_i1) {
                double v1922_data_pre = wrap_glb_m0[v1924_g ? ((v1905_lead + (v1919_i1 * 20))) : (0)];
                double v1922_data = v1924_g ? (v1922_data_pre) : (0.0);
                if (v1906_g) {
                  r0[v1919_i1] = v1922_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
            }
            else {
              bool v1925_g = allowed_next && v1906_g;
              // r0 = load{g>r}(wrap_glb_m0);
              #pragma unroll
              for (int32_t v1911_i1 = 0; v1911_i1 < 9; ++v1911_i1) {
                double v1914_data_pre = wrap_glb_m0[v1925_g ? ((v1905_lead + (v1911_i1 * 20))) : (0)];
                double v1914_data = v1925_g ? (v1914_data_pre) : (0.0);
                if (v1906_g) {
                  r0[v1911_i1] = v1914_data;
                }
              }
            }
            v1899_flagWord = v1900_flagWordNext;
            v1900_flagWordNext = flagWordAhead;
          }
        }
      });
    }
  });
}

