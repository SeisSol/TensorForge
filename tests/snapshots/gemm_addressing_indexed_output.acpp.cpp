// === base name ===
kernel_bef6e2b2755e5f6e

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_bef6e2b2755e5f6e = {{16, 16, 1}, 16, 9, 1, 16, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_bef6e2b2755e5f6e(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_bef6e2b2755e5f6e(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_bef6e2b2755e5f6e(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_bef6e2b2755e5f6e(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_bef6e2b2755e5f6e(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_bef6e2b2755e5f6e(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, elementIndices, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_bef6e2b2755e5f6e(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (9 active) x 16 per block = block 16x16x1, 2048 B shared, occupancy grid
        // operands:
        //   m0 9×9(9×9) {0..9}×{0..9} indexed
        //   m1 9×9(9×9) {0..9}×{0..9} indexed(offset=8)
        //   m2 9×9(9×9) {0..9}×{0..9} indexed(stride=96)
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"double","launch":{"active_threads":9,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":2048,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"indexed","alias":"C","bbox":[[0,0],[9,9]],"name":"m0","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"indexed(offset=8)","alias":"A","bbox":[[0,0],[9,9]],"name":"m1","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"indexed(stride=96)","alias":"B","bbox":[[0,0],[9,9]],"name":"m2","ordered":false,"parts":1,"shape":[9,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"indexed","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[9,9]},"kind":"multilinear","ops":[{"addressing":"indexed(offset=8)","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[9,9]},{"addressing":"indexed(stride=96)","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          double* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          double r0[9]{};
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v540_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v540_flagSecondAhead < numElements0) ? v540_flagSecondAhead : v3_batchId1)];
          const bool allowed_peel = static_cast<bool>(flagWordFirst);
          const double *const __restrict__ peel_glb_m1 = &m1[static_cast<size_t>(elementIndices[v3_batchId1]) * 81 + 0 + 8 + m1_extraOffset];
          int32_t v550_lead = item.get_local_id(2) % 16;
          bool v551_g = v550_lead < 9;
          bool v568_g = allowed_peel && v551_g;
          // r0 = load{g>r}(peel_glb_m1);
          #pragma unroll
          for (int32_t v560_i1 = 0; v560_i1 < 9; ++v560_i1) {
            double v563_data_pre = peel_glb_m1[v568_g ? ((v550_lead + (v560_i1 * 9))) : (0)];
            double v563_data = v568_g ? (v563_data_pre) : (0.0);
            r0[v560_i1] = v563_data;
          }
          uint32_t v544_flagWord = flagWordFirst;
          uint32_t v545_flagWordNext = flagWordSecond;
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const double *const __restrict__ wrap_glb_m1 = &m1[static_cast<size_t>(elementIndices[v10_batchId1]) * 81 + 0 + 8 + m1_extraOffset];
            size_t v573_flagAheadAhead = v10_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v573_flagAheadAhead < numElements0) ? v573_flagAheadAhead : v10_batchId1)];
            const bool allowed = static_cast<bool>(v544_flagWord);
            const bool allowed_next = static_cast<bool>(v545_flagWordNext);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[static_cast<size_t>(elementIndices[v7_batchId0]) * 81 + 0 + m0_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[static_cast<size_t>(elementIndices[v7_batchId0]) * 96 + 0 + m2_extraOffset];
              double r1[9]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v31_i1 = 0; v31_i1 < 9; ++v31_i1) {
                double v37_data_pre = glb_m2[v551_g ? ((v550_lead + (v31_i1 * 9))) : (0)];
                double v37_data = v551_g ? (v37_data_pre) : (0.0);
                r1[v31_i1] = v37_data;
              }
              double r2[9]{};
              // ir2 = +(r0 * r1)
              // [(0, 9), (0, 9)] [(0, 9)]
              double ir2[9]{};
              double v41_data = r0[0];
              double v42_data = r1[0];
              double v43_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v48_data = r1[1];
              double v49_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v54_data = r1[2];
              double v55_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v60_data = r1[3];
              double v61_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v66_data = r1[4];
              double v67_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v72_data = r1[5];
              double v73_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v78_data = r1[6];
              double v79_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v84_data = r1[7];
              double v85_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v90_data = r1[8];
              double v91_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v95_data = r0[1];
              double v97_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v103_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v109_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v115_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v121_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v127_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v133_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v139_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v145_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v149_data = r0[2];
              double v151_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v157_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v163_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v169_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v175_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v181_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v187_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v193_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v199_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v203_data = r0[3];
              double v205_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v211_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v217_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v223_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v229_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v235_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v241_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v247_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v253_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v257_data = r0[4];
              double v259_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v265_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v271_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v277_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v283_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v289_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v295_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v301_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v307_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v311_data = r0[5];
              double v313_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v319_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v325_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v331_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v337_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v343_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v349_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v355_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v361_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v365_data = r0[6];
              double v367_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v373_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v379_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v385_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v391_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v397_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v403_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v409_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v415_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v419_data = r0[7];
              double v421_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v424_p = (((((((v41_data * v43_bc) + (v95_data * v97_bc)) + (v149_data * v151_bc)) + (v203_data * v205_bc)) + (v257_data * v259_bc)) + (v311_data * v313_bc)) + (v365_data * v367_bc)) + (v419_data * v421_bc);
              double v427_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v430_p = (((((((v41_data * v49_bc) + (v95_data * v103_bc)) + (v149_data * v157_bc)) + (v203_data * v211_bc)) + (v257_data * v265_bc)) + (v311_data * v319_bc)) + (v365_data * v373_bc)) + (v419_data * v427_bc);
              double v433_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v436_p = (((((((v41_data * v55_bc) + (v95_data * v109_bc)) + (v149_data * v163_bc)) + (v203_data * v217_bc)) + (v257_data * v271_bc)) + (v311_data * v325_bc)) + (v365_data * v379_bc)) + (v419_data * v433_bc);
              double v439_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v442_p = (((((((v41_data * v61_bc) + (v95_data * v115_bc)) + (v149_data * v169_bc)) + (v203_data * v223_bc)) + (v257_data * v277_bc)) + (v311_data * v331_bc)) + (v365_data * v385_bc)) + (v419_data * v439_bc);
              double v445_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v448_p = (((((((v41_data * v67_bc) + (v95_data * v121_bc)) + (v149_data * v175_bc)) + (v203_data * v229_bc)) + (v257_data * v283_bc)) + (v311_data * v337_bc)) + (v365_data * v391_bc)) + (v419_data * v445_bc);
              double v451_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v454_p = (((((((v41_data * v73_bc) + (v95_data * v127_bc)) + (v149_data * v181_bc)) + (v203_data * v235_bc)) + (v257_data * v289_bc)) + (v311_data * v343_bc)) + (v365_data * v397_bc)) + (v419_data * v451_bc);
              double v457_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v460_p = (((((((v41_data * v79_bc) + (v95_data * v133_bc)) + (v149_data * v187_bc)) + (v203_data * v241_bc)) + (v257_data * v295_bc)) + (v311_data * v349_bc)) + (v365_data * v403_bc)) + (v419_data * v457_bc);
              double v463_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v466_p = (((((((v41_data * v85_bc) + (v95_data * v139_bc)) + (v149_data * v193_bc)) + (v203_data * v247_bc)) + (v257_data * v301_bc)) + (v311_data * v355_bc)) + (v365_data * v409_bc)) + (v419_data * v463_bc);
              double v469_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v472_p = (((((((v41_data * v91_bc) + (v95_data * v145_bc)) + (v149_data * v199_bc)) + (v203_data * v253_bc)) + (v257_data * v307_bc)) + (v311_data * v361_bc)) + (v365_data * v415_bc)) + (v419_data * v469_bc);
              double v473_data = r0[8];
              bool v569_g = allowed_next && v551_g;
              // r0 = load{g>r}(wrap_glb_m1);
              #pragma unroll
              for (int32_t v564_i1 = 0; v564_i1 < 9; ++v564_i1) {
                double v567_data_pre = wrap_glb_m1[v569_g ? ((v550_lead + (v564_i1 * 9))) : (0)];
                double v567_data = v569_g ? (v567_data_pre) : (0.0);
                r0[v564_i1] = v567_data;
              }
              double v475_bc = sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              ir2[0] = (v424_p + (v473_data * v475_bc));
              double v481_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              ir2[1] = (v430_p + (v473_data * v481_bc));
              double v487_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              ir2[2] = (v436_p + (v473_data * v487_bc));
              double v493_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              ir2[3] = (v442_p + (v473_data * v493_bc));
              double v499_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              ir2[4] = (v448_p + (v473_data * v499_bc));
              double v505_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              ir2[5] = (v454_p + (v473_data * v505_bc));
              double v511_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              ir2[6] = (v460_p + (v473_data * v511_bc));
              double v517_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              ir2[7] = (v466_p + (v473_data * v517_bc));
              double v523_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              ir2[8] = (v472_p + (v473_data * v523_bc));
              // r2 = ir2
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v532_i1 = 0; v532_i1 < 9; ++v532_i1) {
                double v534_data = ir2[v532_i1];
                if (v551_g) {
                  glb_m0[(v550_lead + (v532_i1 * 9))] = v534_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
            }
            else {
              bool v570_g = allowed_next && v551_g;
              // r0 = load{g>r}(wrap_glb_m1);
              #pragma unroll
              for (int32_t v556_i1 = 0; v556_i1 < 9; ++v556_i1) {
                double v559_data_pre = wrap_glb_m1[v570_g ? ((v550_lead + (v556_i1 * 9))) : (0)];
                double v559_data = v570_g ? (v559_data_pre) : (0.0);
                r0[v556_i1] = v559_data;
              }
            }
            v544_flagWord = v545_flagWordNext;
            v545_flagWordNext = flagWordAhead;
          }
        }
      });
    }
  });
}

