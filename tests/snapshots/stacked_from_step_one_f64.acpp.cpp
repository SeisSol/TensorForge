// === base name ===
kernel_d75e6fbd6556f07b

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_d75e6fbd6556f07b = {{16, 16, 1}, 16, 40, 1, 16, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_d75e6fbd6556f07b(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_d75e6fbd6556f07b(const double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, double * m2, size_t m2_extraOffset, const double * m3, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_d75e6fbd6556f07b(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_d75e6fbd6556f07b(const double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, double * m2, size_t m2_extraOffset, const double * m3, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_d75e6fbd6556f07b(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_d75e6fbd6556f07b(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_d75e6fbd6556f07b(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, double * m2, size_t m2_extraOffset, const double * m3, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: amd_matrix_k4=1,lanes_per_mult=16
        // launch: 16 lanes (40 active) x 16 per block = block 16x16x1, 2048 B shared, occupancy grid
        // operands:
        //   m0 30×9(30×9) {0..30}×{0..9} strided
        //   m1 9×9(9×9) {0..9}×{0..9} strided
        //   m2 40×9(40×9) {0..40}×{0..9} strided
        //   m3 40×30(40×30) {0..40}×{0..30} none ordered runs-packed
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   m2[i,j] = m3[i,k]@{0..40}×{1..30} × t0[k,j]@{1..30}×{0..9}
        // tensorforge-meta: {"fp":"double","launch":{"active_threads":40,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":2048,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"X","bbox":[[0,0],[30,9]],"name":"m0","ordered":false,"parts":1,"shape":[30,9],"variant":false},{"addressing":"strided","alias":"S","bbox":[[0,0],[9,9]],"name":"m1","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[40,9]],"name":"m2","ordered":false,"parts":1,"shape":[40,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[40,30]],"name":"m3","ordered":true,"parts":1,"shape":[40,30],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[30,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[30,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[30,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[30,9]},{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[40,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[40,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[40,30]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[40,30]},{"addressing":"strided","bbox":[[1,0],[30,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[30,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          double* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          size_t v3_batchId1 = ((item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) < numElements0) ? (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))) : 0u;
          const double *const __restrict__ glb_m3 = &m3[0];
          double r0[18]{};
          const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[v3_batchId1];
          size_t v5411_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v5411_flagSecondAhead < numElements0) ? v5411_flagSecondAhead : v3_batchId1)];
          const double *const __restrict__ peel_glb_m0 = &m0[v3_batchId1 * 270 + 0 + m0_extraOffset];
          int32_t v5419_lead = item.get_local_id(2) % 16;
          int32_t v5421_lead = v5419_lead + 16_i32;
          bool v5422_g = v5419_lead < 14;
          // r0 = load{g>r}(peel_glb_m0);
          #pragma unroll
          for (int32_t v5444_i0 = 0; v5444_i0 < 1; ++v5444_i0) {
            int32_t v5446_lead = v5419_lead + (v5444_i0 * 16);
            #pragma unroll
            for (int32_t v5447_i1 = 0; v5447_i1 < 9; ++v5447_i1) {
              double v5450_data = peel_glb_m0[(v5446_lead + (v5447_i1 * 30))];
              r0[(v5444_i0 + (v5447_i1 * 2))] = v5450_data;
            }
          }
          #pragma unroll
          for (int32_t v5453_i1 = 0; v5453_i1 < 9; ++v5453_i1) {
            double v5456_data_pre = peel_glb_m0[v5422_g ? ((v5421_lead + (v5453_i1 * 30))) : (0)];
            double v5456_data = v5422_g ? (v5456_data_pre) : (0.0);
            r0[(1 + (v5453_i1 * 2))] = v5456_data;
          }
          uint32_t v5415_flagWord = flagWordFirst;
          uint32_t v5416_flagWordNext = flagWordSecond;
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const double *const __restrict__ wrap_glb_m0 = &m0[v12_batchId1 * 270 + 0 + m0_extraOffset];
            size_t v5476_flagAheadAhead = v12_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v5476_flagAheadAhead < numElements0) ? v5476_flagAheadAhead : v12_batchId1)];
            const bool allowed = static_cast<bool>(v5415_flagWord);
            if (allowed) {
              const double *const __restrict__ glb_m1 = &m1[v9_batchId0 * 81 + 0 + m1_extraOffset];
              double *const __restrict__ glb_m2 = &m2[v9_batchId0 * 360 + 0 + m2_extraOffset];
              double r1[9]{};
              // r1 = load{g>r}(glb_m1);
              bool v48_g = v5419_lead < 9;
              #pragma unroll
              for (int32_t v43_i1 = 0; v43_i1 < 9; ++v43_i1) {
                double v49_data_pre = glb_m1[v48_g ? ((v5419_lead + (v43_i1 * 9))) : (0)];
                double v49_data = v48_g ? (v49_data_pre) : (0.0);
                r1[v43_i1] = v49_data;
              }
              double r2[18]{};
              // r2 = +(r0 * r1) + None
              // [(0, 30), (0, 9)] [(0, 9)]
              double v52_data = r0[0];
              double v53_data = r1[0];
              double v54_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v59_data = r1[1];
              double v60_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v65_data = r1[2];
              double v66_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v71_data = r1[3];
              double v72_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v77_data = r1[4];
              double v78_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v83_data = r1[5];
              double v84_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v89_data = r1[6];
              double v90_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v95_data = r1[7];
              double v96_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v101_data = r1[8];
              double v102_bc = sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v106_data = r0[1];
              double v160_data = r0[2];
              double v162_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v168_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v174_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v180_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v186_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v192_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v198_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v204_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v210_bc = sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v214_data = r0[3];
              double v268_data = r0[4];
              double v270_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v276_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v282_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v288_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v294_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v300_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v306_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v312_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v318_bc = sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v322_data = r0[5];
              double v376_data = r0[6];
              double v378_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v384_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v390_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v396_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v402_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v408_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v414_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v420_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v426_bc = sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v430_data = r0[7];
              double v484_data = r0[8];
              double v486_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v492_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v498_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v504_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v510_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v516_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v522_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v528_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v534_bc = sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v538_data = r0[9];
              double v592_data = r0[10];
              double v594_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v600_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v606_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v612_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v618_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v624_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v630_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v636_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v642_bc = sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v646_data = r0[11];
              double v700_data = r0[12];
              double v702_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v708_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v714_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v720_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v726_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v732_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v738_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v744_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v750_bc = sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v754_data = r0[13];
              double v808_data = r0[14];
              double v810_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v816_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v819_p = (((((((v52_data * v60_bc) + (v160_data * v168_bc)) + (v268_data * v276_bc)) + (v376_data * v384_bc)) + (v484_data * v492_bc)) + (v592_data * v600_bc)) + (v700_data * v708_bc)) + (v808_data * v816_bc);
              double v822_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v825_p = (((((((v52_data * v66_bc) + (v160_data * v174_bc)) + (v268_data * v282_bc)) + (v376_data * v390_bc)) + (v484_data * v498_bc)) + (v592_data * v606_bc)) + (v700_data * v714_bc)) + (v808_data * v822_bc);
              double v828_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v831_p = (((((((v52_data * v72_bc) + (v160_data * v180_bc)) + (v268_data * v288_bc)) + (v376_data * v396_bc)) + (v484_data * v504_bc)) + (v592_data * v612_bc)) + (v700_data * v720_bc)) + (v808_data * v828_bc);
              double v834_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v837_p = (((((((v52_data * v78_bc) + (v160_data * v186_bc)) + (v268_data * v294_bc)) + (v376_data * v402_bc)) + (v484_data * v510_bc)) + (v592_data * v618_bc)) + (v700_data * v726_bc)) + (v808_data * v834_bc);
              double v840_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v843_p = (((((((v52_data * v84_bc) + (v160_data * v192_bc)) + (v268_data * v300_bc)) + (v376_data * v408_bc)) + (v484_data * v516_bc)) + (v592_data * v624_bc)) + (v700_data * v732_bc)) + (v808_data * v840_bc);
              double v846_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v849_p = (((((((v52_data * v90_bc) + (v160_data * v198_bc)) + (v268_data * v306_bc)) + (v376_data * v414_bc)) + (v484_data * v522_bc)) + (v592_data * v630_bc)) + (v700_data * v738_bc)) + (v808_data * v846_bc);
              double v852_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v855_p = (((((((v52_data * v96_bc) + (v160_data * v204_bc)) + (v268_data * v312_bc)) + (v376_data * v420_bc)) + (v484_data * v528_bc)) + (v592_data * v636_bc)) + (v700_data * v744_bc)) + (v808_data * v852_bc);
              double v858_bc = sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v861_p = (((((((v52_data * v102_bc) + (v160_data * v210_bc)) + (v268_data * v318_bc)) + (v376_data * v426_bc)) + (v484_data * v534_bc)) + (v592_data * v642_bc)) + (v700_data * v750_bc)) + (v808_data * v858_bc);
              double v862_data = r0[15];
              double v867_p = (((((((v106_data * v54_bc) + (v214_data * v162_bc)) + (v322_data * v270_bc)) + (v430_data * v378_bc)) + (v538_data * v486_bc)) + (v646_data * v594_bc)) + (v754_data * v702_bc)) + (v862_data * v810_bc);
              double v873_p = (((((((v106_data * v60_bc) + (v214_data * v168_bc)) + (v322_data * v276_bc)) + (v430_data * v384_bc)) + (v538_data * v492_bc)) + (v646_data * v600_bc)) + (v754_data * v708_bc)) + (v862_data * v816_bc);
              double v879_p = (((((((v106_data * v66_bc) + (v214_data * v174_bc)) + (v322_data * v282_bc)) + (v430_data * v390_bc)) + (v538_data * v498_bc)) + (v646_data * v606_bc)) + (v754_data * v714_bc)) + (v862_data * v822_bc);
              double v885_p = (((((((v106_data * v72_bc) + (v214_data * v180_bc)) + (v322_data * v288_bc)) + (v430_data * v396_bc)) + (v538_data * v504_bc)) + (v646_data * v612_bc)) + (v754_data * v720_bc)) + (v862_data * v828_bc);
              double v891_p = (((((((v106_data * v78_bc) + (v214_data * v186_bc)) + (v322_data * v294_bc)) + (v430_data * v402_bc)) + (v538_data * v510_bc)) + (v646_data * v618_bc)) + (v754_data * v726_bc)) + (v862_data * v834_bc);
              double v897_p = (((((((v106_data * v84_bc) + (v214_data * v192_bc)) + (v322_data * v300_bc)) + (v430_data * v408_bc)) + (v538_data * v516_bc)) + (v646_data * v624_bc)) + (v754_data * v732_bc)) + (v862_data * v840_bc);
              double v903_p = (((((((v106_data * v90_bc) + (v214_data * v198_bc)) + (v322_data * v306_bc)) + (v430_data * v414_bc)) + (v538_data * v522_bc)) + (v646_data * v630_bc)) + (v754_data * v738_bc)) + (v862_data * v846_bc);
              double v909_p = (((((((v106_data * v96_bc) + (v214_data * v204_bc)) + (v322_data * v312_bc)) + (v430_data * v420_bc)) + (v538_data * v528_bc)) + (v646_data * v636_bc)) + (v754_data * v744_bc)) + (v862_data * v852_bc);
              double v915_p = (((((((v106_data * v102_bc) + (v214_data * v210_bc)) + (v322_data * v318_bc)) + (v430_data * v426_bc)) + (v538_data * v534_bc)) + (v646_data * v642_bc)) + (v754_data * v750_bc)) + (v862_data * v858_bc);
              double v916_data = r0[16];
              double v918_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v921_p = ((((((((v52_data * v54_bc) + (v160_data * v162_bc)) + (v268_data * v270_bc)) + (v376_data * v378_bc)) + (v484_data * v486_bc)) + (v592_data * v594_bc)) + (v700_data * v702_bc)) + (v808_data * v810_bc)) + (v916_data * v918_bc);
              r2[0] = v921_p;
              double v924_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v927_p = v819_p + (v916_data * v924_bc);
              r2[2] = v927_p;
              double v930_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v933_p = v825_p + (v916_data * v930_bc);
              r2[4] = v933_p;
              double v936_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v939_p = v831_p + (v916_data * v936_bc);
              r2[6] = v939_p;
              double v942_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v945_p = v837_p + (v916_data * v942_bc);
              r2[8] = v945_p;
              double v948_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v951_p = v843_p + (v916_data * v948_bc);
              r2[10] = v951_p;
              double v954_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v957_p = v849_p + (v916_data * v954_bc);
              r2[12] = v957_p;
              double v960_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v963_p = v855_p + (v916_data * v960_bc);
              r2[14] = v963_p;
              double v966_bc = sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v969_p = v861_p + (v916_data * v966_bc);
              r2[16] = v969_p;
              double v970_data = r0[17];
              r2[1] = (v867_p + (v970_data * v918_bc));
              r2[3] = (v873_p + (v970_data * v924_bc));
              r2[5] = (v879_p + (v970_data * v930_bc));
              r2[7] = (v885_p + (v970_data * v936_bc));
              r2[9] = (v891_p + (v970_data * v942_bc));
              r2[11] = (v897_p + (v970_data * v948_bc));
              r2[13] = (v903_p + (v970_data * v954_bc));
              r2[15] = (v909_p + (v970_data * v960_bc));
              r2[17] = (v915_p + (v970_data * v966_bc));
              // r2 = 0 past (0, 30)
              // s0 = store{r>s}(localShrMem0, r2);
              double r3[18]{};
              // r3 = load{g>r}(s0);
              double r4[27]{};
              // ir4 = +(glb_m3 * r3)
              // [(0, 40), (0, 9)] [(1, 30)]
              double ir4[27]{};
              double v1066_data = glb_m3[v5419_lead];
              double v1068_bc = sycl::select_from_group(item.get_sub_group(), v921_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v1074_bc = sycl::select_from_group(item.get_sub_group(), v927_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v1080_bc = sycl::select_from_group(item.get_sub_group(), v933_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v1086_bc = sycl::select_from_group(item.get_sub_group(), v939_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v1092_bc = sycl::select_from_group(item.get_sub_group(), v945_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v1098_bc = sycl::select_from_group(item.get_sub_group(), v951_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v1104_bc = sycl::select_from_group(item.get_sub_group(), v957_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v1110_bc = sycl::select_from_group(item.get_sub_group(), v963_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v1116_bc = sycl::select_from_group(item.get_sub_group(), v969_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v1123_data = glb_m3[v5421_lead];
              int32_t v1178_lead = v5419_lead + 32_i32;
              bool v1180_g = v5419_lead < 8;
              double v1181_data_pre = glb_m3[v1180_g ? (v1178_lead) : (0)];
              double v1181_data = v1180_g ? (v1181_data_pre) : (0.0);
              double v1237_data = glb_m3[(v5419_lead + 40)];
              double v1239_bc = sycl::select_from_group(item.get_sub_group(), v921_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v1245_bc = sycl::select_from_group(item.get_sub_group(), v927_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v1251_bc = sycl::select_from_group(item.get_sub_group(), v933_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v1257_bc = sycl::select_from_group(item.get_sub_group(), v939_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v1263_bc = sycl::select_from_group(item.get_sub_group(), v945_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v1269_bc = sycl::select_from_group(item.get_sub_group(), v951_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v1275_bc = sycl::select_from_group(item.get_sub_group(), v957_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v1281_bc = sycl::select_from_group(item.get_sub_group(), v963_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v1287_bc = sycl::select_from_group(item.get_sub_group(), v969_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v1293_data = glb_m3[(v5421_lead + 40)];
              double v1349_data_pre = glb_m3[v1180_g ? ((v1178_lead + 40)) : (0)];
              double v1349_data = v1180_g ? (v1349_data_pre) : (0.0);
              double v1405_data = glb_m3[(v5419_lead + 80)];
              double v1407_bc = sycl::select_from_group(item.get_sub_group(), v921_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v1413_bc = sycl::select_from_group(item.get_sub_group(), v927_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v1419_bc = sycl::select_from_group(item.get_sub_group(), v933_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v1425_bc = sycl::select_from_group(item.get_sub_group(), v939_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v1431_bc = sycl::select_from_group(item.get_sub_group(), v945_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v1437_bc = sycl::select_from_group(item.get_sub_group(), v951_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v1443_bc = sycl::select_from_group(item.get_sub_group(), v957_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v1449_bc = sycl::select_from_group(item.get_sub_group(), v963_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v1455_bc = sycl::select_from_group(item.get_sub_group(), v969_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v1461_data = glb_m3[(v5421_lead + 80)];
              double v1517_data_pre = glb_m3[v1180_g ? ((v1178_lead + 80)) : (0)];
              double v1517_data = v1180_g ? (v1517_data_pre) : (0.0);
              double v1573_data = glb_m3[(v5419_lead + 120)];
              double v1575_bc = sycl::select_from_group(item.get_sub_group(), v921_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v1581_bc = sycl::select_from_group(item.get_sub_group(), v927_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v1587_bc = sycl::select_from_group(item.get_sub_group(), v933_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v1593_bc = sycl::select_from_group(item.get_sub_group(), v939_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v1599_bc = sycl::select_from_group(item.get_sub_group(), v945_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v1605_bc = sycl::select_from_group(item.get_sub_group(), v951_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v1611_bc = sycl::select_from_group(item.get_sub_group(), v957_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v1617_bc = sycl::select_from_group(item.get_sub_group(), v963_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v1623_bc = sycl::select_from_group(item.get_sub_group(), v969_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v1629_data = glb_m3[(v5421_lead + 120)];
              double v1685_data_pre = glb_m3[v1180_g ? ((v1178_lead + 120)) : (0)];
              double v1685_data = v1180_g ? (v1685_data_pre) : (0.0);
              double v1741_data = glb_m3[(v5419_lead + 160)];
              double v1743_bc = sycl::select_from_group(item.get_sub_group(), v921_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v1749_bc = sycl::select_from_group(item.get_sub_group(), v927_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v1755_bc = sycl::select_from_group(item.get_sub_group(), v933_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v1761_bc = sycl::select_from_group(item.get_sub_group(), v939_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v1767_bc = sycl::select_from_group(item.get_sub_group(), v945_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v1773_bc = sycl::select_from_group(item.get_sub_group(), v951_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v1779_bc = sycl::select_from_group(item.get_sub_group(), v957_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v1785_bc = sycl::select_from_group(item.get_sub_group(), v963_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v1791_bc = sycl::select_from_group(item.get_sub_group(), v969_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v1797_data = glb_m3[(v5421_lead + 160)];
              double v1853_data_pre = glb_m3[v1180_g ? ((v1178_lead + 160)) : (0)];
              double v1853_data = v1180_g ? (v1853_data_pre) : (0.0);
              double v1909_data = glb_m3[(v5419_lead + 200)];
              double v1911_bc = sycl::select_from_group(item.get_sub_group(), v921_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v1917_bc = sycl::select_from_group(item.get_sub_group(), v927_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v1923_bc = sycl::select_from_group(item.get_sub_group(), v933_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v1929_bc = sycl::select_from_group(item.get_sub_group(), v939_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v1935_bc = sycl::select_from_group(item.get_sub_group(), v945_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v1941_bc = sycl::select_from_group(item.get_sub_group(), v951_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v1947_bc = sycl::select_from_group(item.get_sub_group(), v957_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v1953_bc = sycl::select_from_group(item.get_sub_group(), v963_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v1959_bc = sycl::select_from_group(item.get_sub_group(), v969_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v1965_data = glb_m3[(v5421_lead + 200)];
              double v2021_data_pre = glb_m3[v1180_g ? ((v1178_lead + 200)) : (0)];
              double v2021_data = v1180_g ? (v2021_data_pre) : (0.0);
              double v2077_data = glb_m3[(v5419_lead + 240)];
              double v2079_bc = sycl::select_from_group(item.get_sub_group(), v921_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v2085_bc = sycl::select_from_group(item.get_sub_group(), v927_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v2091_bc = sycl::select_from_group(item.get_sub_group(), v933_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v2097_bc = sycl::select_from_group(item.get_sub_group(), v939_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v2103_bc = sycl::select_from_group(item.get_sub_group(), v945_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v2109_bc = sycl::select_from_group(item.get_sub_group(), v951_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v2115_bc = sycl::select_from_group(item.get_sub_group(), v957_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v2121_bc = sycl::select_from_group(item.get_sub_group(), v963_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v2127_bc = sycl::select_from_group(item.get_sub_group(), v969_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v2133_data = glb_m3[(v5421_lead + 240)];
              double v2189_data_pre = glb_m3[v1180_g ? ((v1178_lead + 240)) : (0)];
              double v2189_data = v1180_g ? (v2189_data_pre) : (0.0);
              double v2245_data = glb_m3[(v5419_lead + 280)];
              double v2247_bc = sycl::select_from_group(item.get_sub_group(), v921_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v2253_bc = sycl::select_from_group(item.get_sub_group(), v927_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v2259_bc = sycl::select_from_group(item.get_sub_group(), v933_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v2265_bc = sycl::select_from_group(item.get_sub_group(), v939_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v2271_bc = sycl::select_from_group(item.get_sub_group(), v945_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v2277_bc = sycl::select_from_group(item.get_sub_group(), v951_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v2283_bc = sycl::select_from_group(item.get_sub_group(), v957_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v2289_bc = sycl::select_from_group(item.get_sub_group(), v963_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v2295_bc = sycl::select_from_group(item.get_sub_group(), v969_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v2301_data = glb_m3[(v5421_lead + 280)];
              double v2357_data_pre = glb_m3[v1180_g ? ((v1178_lead + 280)) : (0)];
              double v2357_data = v1180_g ? (v2357_data_pre) : (0.0);
              double v2413_data = glb_m3[(v5419_lead + 320)];
              double v2415_bc = sycl::select_from_group(item.get_sub_group(), v921_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              double v2421_bc = sycl::select_from_group(item.get_sub_group(), v927_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              double v2427_bc = sycl::select_from_group(item.get_sub_group(), v933_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              double v2433_bc = sycl::select_from_group(item.get_sub_group(), v939_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              double v2439_bc = sycl::select_from_group(item.get_sub_group(), v945_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              double v2445_bc = sycl::select_from_group(item.get_sub_group(), v951_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              double v2451_bc = sycl::select_from_group(item.get_sub_group(), v957_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              double v2457_bc = sycl::select_from_group(item.get_sub_group(), v963_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              double v2463_bc = sycl::select_from_group(item.get_sub_group(), v969_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              double v2469_data = glb_m3[(v5421_lead + 320)];
              double v2525_data_pre = glb_m3[v1180_g ? ((v1178_lead + 320)) : (0)];
              double v2525_data = v1180_g ? (v2525_data_pre) : (0.0);
              double v2581_data = glb_m3[(v5419_lead + 360)];
              double v2583_bc = sycl::select_from_group(item.get_sub_group(), v921_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              double v2589_bc = sycl::select_from_group(item.get_sub_group(), v927_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              double v2595_bc = sycl::select_from_group(item.get_sub_group(), v933_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              double v2601_bc = sycl::select_from_group(item.get_sub_group(), v939_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              double v2607_bc = sycl::select_from_group(item.get_sub_group(), v945_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              double v2613_bc = sycl::select_from_group(item.get_sub_group(), v951_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              double v2619_bc = sycl::select_from_group(item.get_sub_group(), v957_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              double v2625_bc = sycl::select_from_group(item.get_sub_group(), v963_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              double v2631_bc = sycl::select_from_group(item.get_sub_group(), v969_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              double v2637_data = glb_m3[(v5421_lead + 360)];
              double v2693_data_pre = glb_m3[v1180_g ? ((v1178_lead + 360)) : (0)];
              double v2693_data = v1180_g ? (v2693_data_pre) : (0.0);
              double v2749_data = glb_m3[(v5419_lead + 400)];
              double v2751_bc = sycl::select_from_group(item.get_sub_group(), v921_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              double v2757_bc = sycl::select_from_group(item.get_sub_group(), v927_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              double v2763_bc = sycl::select_from_group(item.get_sub_group(), v933_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              double v2769_bc = sycl::select_from_group(item.get_sub_group(), v939_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              double v2775_bc = sycl::select_from_group(item.get_sub_group(), v945_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              double v2781_bc = sycl::select_from_group(item.get_sub_group(), v951_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              double v2787_bc = sycl::select_from_group(item.get_sub_group(), v957_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              double v2793_bc = sycl::select_from_group(item.get_sub_group(), v963_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              double v2799_bc = sycl::select_from_group(item.get_sub_group(), v969_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              double v2805_data = glb_m3[(v5421_lead + 400)];
              double v2861_data_pre = glb_m3[v1180_g ? ((v1178_lead + 400)) : (0)];
              double v2861_data = v1180_g ? (v2861_data_pre) : (0.0);
              double v2917_data = glb_m3[(v5419_lead + 440)];
              double v2919_bc = sycl::select_from_group(item.get_sub_group(), v921_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              double v2925_bc = sycl::select_from_group(item.get_sub_group(), v927_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              double v2931_bc = sycl::select_from_group(item.get_sub_group(), v933_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              double v2937_bc = sycl::select_from_group(item.get_sub_group(), v939_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              double v2943_bc = sycl::select_from_group(item.get_sub_group(), v945_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              double v2949_bc = sycl::select_from_group(item.get_sub_group(), v951_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              double v2955_bc = sycl::select_from_group(item.get_sub_group(), v957_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              double v2961_bc = sycl::select_from_group(item.get_sub_group(), v963_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              double v2967_bc = sycl::select_from_group(item.get_sub_group(), v969_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              double v2973_data = glb_m3[(v5421_lead + 440)];
              double v3029_data_pre = glb_m3[v1180_g ? ((v1178_lead + 440)) : (0)];
              double v3029_data = v1180_g ? (v3029_data_pre) : (0.0);
              double v3085_data = glb_m3[(v5419_lead + 480)];
              double v3087_bc = sycl::select_from_group(item.get_sub_group(), v921_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              double v3093_bc = sycl::select_from_group(item.get_sub_group(), v927_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              double v3099_bc = sycl::select_from_group(item.get_sub_group(), v933_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              double v3105_bc = sycl::select_from_group(item.get_sub_group(), v939_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              double v3111_bc = sycl::select_from_group(item.get_sub_group(), v945_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              double v3117_bc = sycl::select_from_group(item.get_sub_group(), v951_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              double v3123_bc = sycl::select_from_group(item.get_sub_group(), v957_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              double v3129_bc = sycl::select_from_group(item.get_sub_group(), v963_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              double v3135_bc = sycl::select_from_group(item.get_sub_group(), v969_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              double v3141_data = glb_m3[(v5421_lead + 480)];
              double v3197_data_pre = glb_m3[v1180_g ? ((v1178_lead + 480)) : (0)];
              double v3197_data = v1180_g ? (v3197_data_pre) : (0.0);
              double v3253_data = glb_m3[(v5419_lead + 520)];
              double v3255_bc = sycl::select_from_group(item.get_sub_group(), v921_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              double v3261_bc = sycl::select_from_group(item.get_sub_group(), v927_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              double v3267_bc = sycl::select_from_group(item.get_sub_group(), v933_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              double v3273_bc = sycl::select_from_group(item.get_sub_group(), v939_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              double v3279_bc = sycl::select_from_group(item.get_sub_group(), v945_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              double v3285_bc = sycl::select_from_group(item.get_sub_group(), v951_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              double v3291_bc = sycl::select_from_group(item.get_sub_group(), v957_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              double v3297_bc = sycl::select_from_group(item.get_sub_group(), v963_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              double v3303_bc = sycl::select_from_group(item.get_sub_group(), v969_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              double v3309_data = glb_m3[(v5421_lead + 520)];
              double v3365_data_pre = glb_m3[v1180_g ? ((v1178_lead + 520)) : (0)];
              double v3365_data = v1180_g ? (v3365_data_pre) : (0.0);
              double v3421_data = glb_m3[(v5419_lead + 560)];
              double v3423_bc = sycl::select_from_group(item.get_sub_group(), v921_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              double v3429_bc = sycl::select_from_group(item.get_sub_group(), v927_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              double v3435_bc = sycl::select_from_group(item.get_sub_group(), v933_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              double v3441_bc = sycl::select_from_group(item.get_sub_group(), v939_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              double v3447_bc = sycl::select_from_group(item.get_sub_group(), v945_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              double v3453_bc = sycl::select_from_group(item.get_sub_group(), v951_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              double v3459_bc = sycl::select_from_group(item.get_sub_group(), v957_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              double v3465_bc = sycl::select_from_group(item.get_sub_group(), v963_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              double v3471_bc = sycl::select_from_group(item.get_sub_group(), v969_p, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              double v3477_data = glb_m3[(v5421_lead + 560)];
              double v3533_data_pre = glb_m3[v1180_g ? ((v1178_lead + 560)) : (0)];
              double v3533_data = v1180_g ? (v3533_data_pre) : (0.0);
              double v3589_data = glb_m3[(v5419_lead + 600)];
              double v3590_data = r2[1];
              double v3591_bc = sycl::select_from_group(item.get_sub_group(), v3590_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v3596_data = r2[3];
              double v3597_bc = sycl::select_from_group(item.get_sub_group(), v3596_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v3602_data = r2[5];
              double v3603_bc = sycl::select_from_group(item.get_sub_group(), v3602_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v3608_data = r2[7];
              double v3609_bc = sycl::select_from_group(item.get_sub_group(), v3608_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v3614_data = r2[9];
              double v3615_bc = sycl::select_from_group(item.get_sub_group(), v3614_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v3620_data = r2[11];
              double v3621_bc = sycl::select_from_group(item.get_sub_group(), v3620_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v3626_data = r2[13];
              double v3627_bc = sycl::select_from_group(item.get_sub_group(), v3626_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v3632_data = r2[15];
              double v3633_bc = sycl::select_from_group(item.get_sub_group(), v3632_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v3638_data = r2[17];
              double v3639_bc = sycl::select_from_group(item.get_sub_group(), v3638_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v3645_data = glb_m3[(v5421_lead + 600)];
              double v3701_data_pre = glb_m3[v1180_g ? ((v1178_lead + 600)) : (0)];
              double v3701_data = v1180_g ? (v3701_data_pre) : (0.0);
              double v3757_data = glb_m3[(v5419_lead + 640)];
              double v3759_bc = sycl::select_from_group(item.get_sub_group(), v3590_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v3765_bc = sycl::select_from_group(item.get_sub_group(), v3596_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v3771_bc = sycl::select_from_group(item.get_sub_group(), v3602_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v3777_bc = sycl::select_from_group(item.get_sub_group(), v3608_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v3783_bc = sycl::select_from_group(item.get_sub_group(), v3614_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v3789_bc = sycl::select_from_group(item.get_sub_group(), v3620_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v3795_bc = sycl::select_from_group(item.get_sub_group(), v3626_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v3801_bc = sycl::select_from_group(item.get_sub_group(), v3632_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v3807_bc = sycl::select_from_group(item.get_sub_group(), v3638_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v3813_data = glb_m3[(v5421_lead + 640)];
              double v3869_data_pre = glb_m3[v1180_g ? ((v1178_lead + 640)) : (0)];
              double v3869_data = v1180_g ? (v3869_data_pre) : (0.0);
              double v3925_data = glb_m3[(v5419_lead + 680)];
              double v3927_bc = sycl::select_from_group(item.get_sub_group(), v3590_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v3933_bc = sycl::select_from_group(item.get_sub_group(), v3596_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v3936_p = (((((((((((((((((v1066_data * v1074_bc) + (v1237_data * v1245_bc)) + (v1405_data * v1413_bc)) + (v1573_data * v1581_bc)) + (v1741_data * v1749_bc)) + (v1909_data * v1917_bc)) + (v2077_data * v2085_bc)) + (v2245_data * v2253_bc)) + (v2413_data * v2421_bc)) + (v2581_data * v2589_bc)) + (v2749_data * v2757_bc)) + (v2917_data * v2925_bc)) + (v3085_data * v3093_bc)) + (v3253_data * v3261_bc)) + (v3421_data * v3429_bc)) + (v3589_data * v3597_bc)) + (v3757_data * v3765_bc)) + (v3925_data * v3933_bc);
              double v3939_bc = sycl::select_from_group(item.get_sub_group(), v3602_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v3942_p = (((((((((((((((((v1066_data * v1080_bc) + (v1237_data * v1251_bc)) + (v1405_data * v1419_bc)) + (v1573_data * v1587_bc)) + (v1741_data * v1755_bc)) + (v1909_data * v1923_bc)) + (v2077_data * v2091_bc)) + (v2245_data * v2259_bc)) + (v2413_data * v2427_bc)) + (v2581_data * v2595_bc)) + (v2749_data * v2763_bc)) + (v2917_data * v2931_bc)) + (v3085_data * v3099_bc)) + (v3253_data * v3267_bc)) + (v3421_data * v3435_bc)) + (v3589_data * v3603_bc)) + (v3757_data * v3771_bc)) + (v3925_data * v3939_bc);
              double v3945_bc = sycl::select_from_group(item.get_sub_group(), v3608_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v3948_p = (((((((((((((((((v1066_data * v1086_bc) + (v1237_data * v1257_bc)) + (v1405_data * v1425_bc)) + (v1573_data * v1593_bc)) + (v1741_data * v1761_bc)) + (v1909_data * v1929_bc)) + (v2077_data * v2097_bc)) + (v2245_data * v2265_bc)) + (v2413_data * v2433_bc)) + (v2581_data * v2601_bc)) + (v2749_data * v2769_bc)) + (v2917_data * v2937_bc)) + (v3085_data * v3105_bc)) + (v3253_data * v3273_bc)) + (v3421_data * v3441_bc)) + (v3589_data * v3609_bc)) + (v3757_data * v3777_bc)) + (v3925_data * v3945_bc);
              double v3951_bc = sycl::select_from_group(item.get_sub_group(), v3614_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v3954_p = (((((((((((((((((v1066_data * v1092_bc) + (v1237_data * v1263_bc)) + (v1405_data * v1431_bc)) + (v1573_data * v1599_bc)) + (v1741_data * v1767_bc)) + (v1909_data * v1935_bc)) + (v2077_data * v2103_bc)) + (v2245_data * v2271_bc)) + (v2413_data * v2439_bc)) + (v2581_data * v2607_bc)) + (v2749_data * v2775_bc)) + (v2917_data * v2943_bc)) + (v3085_data * v3111_bc)) + (v3253_data * v3279_bc)) + (v3421_data * v3447_bc)) + (v3589_data * v3615_bc)) + (v3757_data * v3783_bc)) + (v3925_data * v3951_bc);
              double v3957_bc = sycl::select_from_group(item.get_sub_group(), v3620_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v3960_p = (((((((((((((((((v1066_data * v1098_bc) + (v1237_data * v1269_bc)) + (v1405_data * v1437_bc)) + (v1573_data * v1605_bc)) + (v1741_data * v1773_bc)) + (v1909_data * v1941_bc)) + (v2077_data * v2109_bc)) + (v2245_data * v2277_bc)) + (v2413_data * v2445_bc)) + (v2581_data * v2613_bc)) + (v2749_data * v2781_bc)) + (v2917_data * v2949_bc)) + (v3085_data * v3117_bc)) + (v3253_data * v3285_bc)) + (v3421_data * v3453_bc)) + (v3589_data * v3621_bc)) + (v3757_data * v3789_bc)) + (v3925_data * v3957_bc);
              double v3963_bc = sycl::select_from_group(item.get_sub_group(), v3626_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v3966_p = (((((((((((((((((v1066_data * v1104_bc) + (v1237_data * v1275_bc)) + (v1405_data * v1443_bc)) + (v1573_data * v1611_bc)) + (v1741_data * v1779_bc)) + (v1909_data * v1947_bc)) + (v2077_data * v2115_bc)) + (v2245_data * v2283_bc)) + (v2413_data * v2451_bc)) + (v2581_data * v2619_bc)) + (v2749_data * v2787_bc)) + (v2917_data * v2955_bc)) + (v3085_data * v3123_bc)) + (v3253_data * v3291_bc)) + (v3421_data * v3459_bc)) + (v3589_data * v3627_bc)) + (v3757_data * v3795_bc)) + (v3925_data * v3963_bc);
              double v3969_bc = sycl::select_from_group(item.get_sub_group(), v3632_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v3972_p = (((((((((((((((((v1066_data * v1110_bc) + (v1237_data * v1281_bc)) + (v1405_data * v1449_bc)) + (v1573_data * v1617_bc)) + (v1741_data * v1785_bc)) + (v1909_data * v1953_bc)) + (v2077_data * v2121_bc)) + (v2245_data * v2289_bc)) + (v2413_data * v2457_bc)) + (v2581_data * v2625_bc)) + (v2749_data * v2793_bc)) + (v2917_data * v2961_bc)) + (v3085_data * v3129_bc)) + (v3253_data * v3297_bc)) + (v3421_data * v3465_bc)) + (v3589_data * v3633_bc)) + (v3757_data * v3801_bc)) + (v3925_data * v3969_bc);
              double v3975_bc = sycl::select_from_group(item.get_sub_group(), v3638_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v3978_p = (((((((((((((((((v1066_data * v1116_bc) + (v1237_data * v1287_bc)) + (v1405_data * v1455_bc)) + (v1573_data * v1623_bc)) + (v1741_data * v1791_bc)) + (v1909_data * v1959_bc)) + (v2077_data * v2127_bc)) + (v2245_data * v2295_bc)) + (v2413_data * v2463_bc)) + (v2581_data * v2631_bc)) + (v2749_data * v2799_bc)) + (v2917_data * v2967_bc)) + (v3085_data * v3135_bc)) + (v3253_data * v3303_bc)) + (v3421_data * v3471_bc)) + (v3589_data * v3639_bc)) + (v3757_data * v3807_bc)) + (v3925_data * v3975_bc);
              double v3981_data = glb_m3[(v5421_lead + 680)];
              double v3986_p = (((((((((((((((((v1123_data * v1068_bc) + (v1293_data * v1239_bc)) + (v1461_data * v1407_bc)) + (v1629_data * v1575_bc)) + (v1797_data * v1743_bc)) + (v1965_data * v1911_bc)) + (v2133_data * v2079_bc)) + (v2301_data * v2247_bc)) + (v2469_data * v2415_bc)) + (v2637_data * v2583_bc)) + (v2805_data * v2751_bc)) + (v2973_data * v2919_bc)) + (v3141_data * v3087_bc)) + (v3309_data * v3255_bc)) + (v3477_data * v3423_bc)) + (v3645_data * v3591_bc)) + (v3813_data * v3759_bc)) + (v3981_data * v3927_bc);
              double v3992_p = (((((((((((((((((v1123_data * v1074_bc) + (v1293_data * v1245_bc)) + (v1461_data * v1413_bc)) + (v1629_data * v1581_bc)) + (v1797_data * v1749_bc)) + (v1965_data * v1917_bc)) + (v2133_data * v2085_bc)) + (v2301_data * v2253_bc)) + (v2469_data * v2421_bc)) + (v2637_data * v2589_bc)) + (v2805_data * v2757_bc)) + (v2973_data * v2925_bc)) + (v3141_data * v3093_bc)) + (v3309_data * v3261_bc)) + (v3477_data * v3429_bc)) + (v3645_data * v3597_bc)) + (v3813_data * v3765_bc)) + (v3981_data * v3933_bc);
              double v3998_p = (((((((((((((((((v1123_data * v1080_bc) + (v1293_data * v1251_bc)) + (v1461_data * v1419_bc)) + (v1629_data * v1587_bc)) + (v1797_data * v1755_bc)) + (v1965_data * v1923_bc)) + (v2133_data * v2091_bc)) + (v2301_data * v2259_bc)) + (v2469_data * v2427_bc)) + (v2637_data * v2595_bc)) + (v2805_data * v2763_bc)) + (v2973_data * v2931_bc)) + (v3141_data * v3099_bc)) + (v3309_data * v3267_bc)) + (v3477_data * v3435_bc)) + (v3645_data * v3603_bc)) + (v3813_data * v3771_bc)) + (v3981_data * v3939_bc);
              double v4004_p = (((((((((((((((((v1123_data * v1086_bc) + (v1293_data * v1257_bc)) + (v1461_data * v1425_bc)) + (v1629_data * v1593_bc)) + (v1797_data * v1761_bc)) + (v1965_data * v1929_bc)) + (v2133_data * v2097_bc)) + (v2301_data * v2265_bc)) + (v2469_data * v2433_bc)) + (v2637_data * v2601_bc)) + (v2805_data * v2769_bc)) + (v2973_data * v2937_bc)) + (v3141_data * v3105_bc)) + (v3309_data * v3273_bc)) + (v3477_data * v3441_bc)) + (v3645_data * v3609_bc)) + (v3813_data * v3777_bc)) + (v3981_data * v3945_bc);
              double v4010_p = (((((((((((((((((v1123_data * v1092_bc) + (v1293_data * v1263_bc)) + (v1461_data * v1431_bc)) + (v1629_data * v1599_bc)) + (v1797_data * v1767_bc)) + (v1965_data * v1935_bc)) + (v2133_data * v2103_bc)) + (v2301_data * v2271_bc)) + (v2469_data * v2439_bc)) + (v2637_data * v2607_bc)) + (v2805_data * v2775_bc)) + (v2973_data * v2943_bc)) + (v3141_data * v3111_bc)) + (v3309_data * v3279_bc)) + (v3477_data * v3447_bc)) + (v3645_data * v3615_bc)) + (v3813_data * v3783_bc)) + (v3981_data * v3951_bc);
              double v4016_p = (((((((((((((((((v1123_data * v1098_bc) + (v1293_data * v1269_bc)) + (v1461_data * v1437_bc)) + (v1629_data * v1605_bc)) + (v1797_data * v1773_bc)) + (v1965_data * v1941_bc)) + (v2133_data * v2109_bc)) + (v2301_data * v2277_bc)) + (v2469_data * v2445_bc)) + (v2637_data * v2613_bc)) + (v2805_data * v2781_bc)) + (v2973_data * v2949_bc)) + (v3141_data * v3117_bc)) + (v3309_data * v3285_bc)) + (v3477_data * v3453_bc)) + (v3645_data * v3621_bc)) + (v3813_data * v3789_bc)) + (v3981_data * v3957_bc);
              double v4022_p = (((((((((((((((((v1123_data * v1104_bc) + (v1293_data * v1275_bc)) + (v1461_data * v1443_bc)) + (v1629_data * v1611_bc)) + (v1797_data * v1779_bc)) + (v1965_data * v1947_bc)) + (v2133_data * v2115_bc)) + (v2301_data * v2283_bc)) + (v2469_data * v2451_bc)) + (v2637_data * v2619_bc)) + (v2805_data * v2787_bc)) + (v2973_data * v2955_bc)) + (v3141_data * v3123_bc)) + (v3309_data * v3291_bc)) + (v3477_data * v3459_bc)) + (v3645_data * v3627_bc)) + (v3813_data * v3795_bc)) + (v3981_data * v3963_bc);
              double v4028_p = (((((((((((((((((v1123_data * v1110_bc) + (v1293_data * v1281_bc)) + (v1461_data * v1449_bc)) + (v1629_data * v1617_bc)) + (v1797_data * v1785_bc)) + (v1965_data * v1953_bc)) + (v2133_data * v2121_bc)) + (v2301_data * v2289_bc)) + (v2469_data * v2457_bc)) + (v2637_data * v2625_bc)) + (v2805_data * v2793_bc)) + (v2973_data * v2961_bc)) + (v3141_data * v3129_bc)) + (v3309_data * v3297_bc)) + (v3477_data * v3465_bc)) + (v3645_data * v3633_bc)) + (v3813_data * v3801_bc)) + (v3981_data * v3969_bc);
              double v4034_p = (((((((((((((((((v1123_data * v1116_bc) + (v1293_data * v1287_bc)) + (v1461_data * v1455_bc)) + (v1629_data * v1623_bc)) + (v1797_data * v1791_bc)) + (v1965_data * v1959_bc)) + (v2133_data * v2127_bc)) + (v2301_data * v2295_bc)) + (v2469_data * v2463_bc)) + (v2637_data * v2631_bc)) + (v2805_data * v2799_bc)) + (v2973_data * v2967_bc)) + (v3141_data * v3135_bc)) + (v3309_data * v3303_bc)) + (v3477_data * v3471_bc)) + (v3645_data * v3639_bc)) + (v3813_data * v3807_bc)) + (v3981_data * v3975_bc);
              double v4037_data_pre = glb_m3[v1180_g ? ((v1178_lead + 680)) : (0)];
              double v4037_data = v1180_g ? (v4037_data_pre) : (0.0);
              double v4042_p = (((((((((((((((((v1181_data * v1068_bc) + (v1349_data * v1239_bc)) + (v1517_data * v1407_bc)) + (v1685_data * v1575_bc)) + (v1853_data * v1743_bc)) + (v2021_data * v1911_bc)) + (v2189_data * v2079_bc)) + (v2357_data * v2247_bc)) + (v2525_data * v2415_bc)) + (v2693_data * v2583_bc)) + (v2861_data * v2751_bc)) + (v3029_data * v2919_bc)) + (v3197_data * v3087_bc)) + (v3365_data * v3255_bc)) + (v3533_data * v3423_bc)) + (v3701_data * v3591_bc)) + (v3869_data * v3759_bc)) + (v4037_data * v3927_bc);
              double v4048_p = (((((((((((((((((v1181_data * v1074_bc) + (v1349_data * v1245_bc)) + (v1517_data * v1413_bc)) + (v1685_data * v1581_bc)) + (v1853_data * v1749_bc)) + (v2021_data * v1917_bc)) + (v2189_data * v2085_bc)) + (v2357_data * v2253_bc)) + (v2525_data * v2421_bc)) + (v2693_data * v2589_bc)) + (v2861_data * v2757_bc)) + (v3029_data * v2925_bc)) + (v3197_data * v3093_bc)) + (v3365_data * v3261_bc)) + (v3533_data * v3429_bc)) + (v3701_data * v3597_bc)) + (v3869_data * v3765_bc)) + (v4037_data * v3933_bc);
              double v4054_p = (((((((((((((((((v1181_data * v1080_bc) + (v1349_data * v1251_bc)) + (v1517_data * v1419_bc)) + (v1685_data * v1587_bc)) + (v1853_data * v1755_bc)) + (v2021_data * v1923_bc)) + (v2189_data * v2091_bc)) + (v2357_data * v2259_bc)) + (v2525_data * v2427_bc)) + (v2693_data * v2595_bc)) + (v2861_data * v2763_bc)) + (v3029_data * v2931_bc)) + (v3197_data * v3099_bc)) + (v3365_data * v3267_bc)) + (v3533_data * v3435_bc)) + (v3701_data * v3603_bc)) + (v3869_data * v3771_bc)) + (v4037_data * v3939_bc);
              double v4060_p = (((((((((((((((((v1181_data * v1086_bc) + (v1349_data * v1257_bc)) + (v1517_data * v1425_bc)) + (v1685_data * v1593_bc)) + (v1853_data * v1761_bc)) + (v2021_data * v1929_bc)) + (v2189_data * v2097_bc)) + (v2357_data * v2265_bc)) + (v2525_data * v2433_bc)) + (v2693_data * v2601_bc)) + (v2861_data * v2769_bc)) + (v3029_data * v2937_bc)) + (v3197_data * v3105_bc)) + (v3365_data * v3273_bc)) + (v3533_data * v3441_bc)) + (v3701_data * v3609_bc)) + (v3869_data * v3777_bc)) + (v4037_data * v3945_bc);
              double v4066_p = (((((((((((((((((v1181_data * v1092_bc) + (v1349_data * v1263_bc)) + (v1517_data * v1431_bc)) + (v1685_data * v1599_bc)) + (v1853_data * v1767_bc)) + (v2021_data * v1935_bc)) + (v2189_data * v2103_bc)) + (v2357_data * v2271_bc)) + (v2525_data * v2439_bc)) + (v2693_data * v2607_bc)) + (v2861_data * v2775_bc)) + (v3029_data * v2943_bc)) + (v3197_data * v3111_bc)) + (v3365_data * v3279_bc)) + (v3533_data * v3447_bc)) + (v3701_data * v3615_bc)) + (v3869_data * v3783_bc)) + (v4037_data * v3951_bc);
              double v4072_p = (((((((((((((((((v1181_data * v1098_bc) + (v1349_data * v1269_bc)) + (v1517_data * v1437_bc)) + (v1685_data * v1605_bc)) + (v1853_data * v1773_bc)) + (v2021_data * v1941_bc)) + (v2189_data * v2109_bc)) + (v2357_data * v2277_bc)) + (v2525_data * v2445_bc)) + (v2693_data * v2613_bc)) + (v2861_data * v2781_bc)) + (v3029_data * v2949_bc)) + (v3197_data * v3117_bc)) + (v3365_data * v3285_bc)) + (v3533_data * v3453_bc)) + (v3701_data * v3621_bc)) + (v3869_data * v3789_bc)) + (v4037_data * v3957_bc);
              double v4078_p = (((((((((((((((((v1181_data * v1104_bc) + (v1349_data * v1275_bc)) + (v1517_data * v1443_bc)) + (v1685_data * v1611_bc)) + (v1853_data * v1779_bc)) + (v2021_data * v1947_bc)) + (v2189_data * v2115_bc)) + (v2357_data * v2283_bc)) + (v2525_data * v2451_bc)) + (v2693_data * v2619_bc)) + (v2861_data * v2787_bc)) + (v3029_data * v2955_bc)) + (v3197_data * v3123_bc)) + (v3365_data * v3291_bc)) + (v3533_data * v3459_bc)) + (v3701_data * v3627_bc)) + (v3869_data * v3795_bc)) + (v4037_data * v3963_bc);
              double v4084_p = (((((((((((((((((v1181_data * v1110_bc) + (v1349_data * v1281_bc)) + (v1517_data * v1449_bc)) + (v1685_data * v1617_bc)) + (v1853_data * v1785_bc)) + (v2021_data * v1953_bc)) + (v2189_data * v2121_bc)) + (v2357_data * v2289_bc)) + (v2525_data * v2457_bc)) + (v2693_data * v2625_bc)) + (v2861_data * v2793_bc)) + (v3029_data * v2961_bc)) + (v3197_data * v3129_bc)) + (v3365_data * v3297_bc)) + (v3533_data * v3465_bc)) + (v3701_data * v3633_bc)) + (v3869_data * v3801_bc)) + (v4037_data * v3969_bc);
              double v4090_p = (((((((((((((((((v1181_data * v1116_bc) + (v1349_data * v1287_bc)) + (v1517_data * v1455_bc)) + (v1685_data * v1623_bc)) + (v1853_data * v1791_bc)) + (v2021_data * v1959_bc)) + (v2189_data * v2127_bc)) + (v2357_data * v2295_bc)) + (v2525_data * v2463_bc)) + (v2693_data * v2631_bc)) + (v2861_data * v2799_bc)) + (v3029_data * v2967_bc)) + (v3197_data * v3135_bc)) + (v3365_data * v3303_bc)) + (v3533_data * v3471_bc)) + (v3701_data * v3639_bc)) + (v3869_data * v3807_bc)) + (v4037_data * v3975_bc);
              double v4093_data = glb_m3[(v5419_lead + 720)];
              double v4095_bc = sycl::select_from_group(item.get_sub_group(), v3590_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              ir4[0] = (((((((((((((((((((v1066_data * v1068_bc) + (v1237_data * v1239_bc)) + (v1405_data * v1407_bc)) + (v1573_data * v1575_bc)) + (v1741_data * v1743_bc)) + (v1909_data * v1911_bc)) + (v2077_data * v2079_bc)) + (v2245_data * v2247_bc)) + (v2413_data * v2415_bc)) + (v2581_data * v2583_bc)) + (v2749_data * v2751_bc)) + (v2917_data * v2919_bc)) + (v3085_data * v3087_bc)) + (v3253_data * v3255_bc)) + (v3421_data * v3423_bc)) + (v3589_data * v3591_bc)) + (v3757_data * v3759_bc)) + (v3925_data * v3927_bc)) + (v4093_data * v4095_bc));
              double v4101_bc = sycl::select_from_group(item.get_sub_group(), v3596_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              ir4[3] = (v3936_p + (v4093_data * v4101_bc));
              double v4107_bc = sycl::select_from_group(item.get_sub_group(), v3602_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              ir4[6] = (v3942_p + (v4093_data * v4107_bc));
              double v4113_bc = sycl::select_from_group(item.get_sub_group(), v3608_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              ir4[9] = (v3948_p + (v4093_data * v4113_bc));
              double v4119_bc = sycl::select_from_group(item.get_sub_group(), v3614_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              ir4[12] = (v3954_p + (v4093_data * v4119_bc));
              double v4125_bc = sycl::select_from_group(item.get_sub_group(), v3620_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              ir4[15] = (v3960_p + (v4093_data * v4125_bc));
              double v4131_bc = sycl::select_from_group(item.get_sub_group(), v3626_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              ir4[18] = (v3966_p + (v4093_data * v4131_bc));
              double v4137_bc = sycl::select_from_group(item.get_sub_group(), v3632_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              ir4[21] = (v3972_p + (v4093_data * v4137_bc));
              double v4143_bc = sycl::select_from_group(item.get_sub_group(), v3638_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              ir4[24] = (v3978_p + (v4093_data * v4143_bc));
              double v4149_data = glb_m3[(v5421_lead + 720)];
              double v4205_data_pre = glb_m3[v1180_g ? ((v1178_lead + 720)) : (0)];
              double v4205_data = v1180_g ? (v4205_data_pre) : (0.0);
              double v4261_data = glb_m3[(v5421_lead + 744)];
              double v4263_bc = sycl::select_from_group(item.get_sub_group(), v3590_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v4269_bc = sycl::select_from_group(item.get_sub_group(), v3596_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v4275_bc = sycl::select_from_group(item.get_sub_group(), v3602_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v4281_bc = sycl::select_from_group(item.get_sub_group(), v3608_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v4287_bc = sycl::select_from_group(item.get_sub_group(), v3614_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v4293_bc = sycl::select_from_group(item.get_sub_group(), v3620_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v4299_bc = sycl::select_from_group(item.get_sub_group(), v3626_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v4305_bc = sycl::select_from_group(item.get_sub_group(), v3632_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v4311_bc = sycl::select_from_group(item.get_sub_group(), v3638_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v4317_data_pre = glb_m3[v1180_g ? ((v1178_lead + 744)) : (0)];
              double v4317_data = v1180_g ? (v4317_data_pre) : (0.0);
              double v4373_data = glb_m3[(v5421_lead + 768)];
              double v4375_bc = sycl::select_from_group(item.get_sub_group(), v3590_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v4381_bc = sycl::select_from_group(item.get_sub_group(), v3596_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v4387_bc = sycl::select_from_group(item.get_sub_group(), v3602_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v4393_bc = sycl::select_from_group(item.get_sub_group(), v3608_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v4399_bc = sycl::select_from_group(item.get_sub_group(), v3614_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v4405_bc = sycl::select_from_group(item.get_sub_group(), v3620_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v4411_bc = sycl::select_from_group(item.get_sub_group(), v3626_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v4417_bc = sycl::select_from_group(item.get_sub_group(), v3632_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v4423_bc = sycl::select_from_group(item.get_sub_group(), v3638_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v4429_data_pre = glb_m3[v1180_g ? ((v1178_lead + 768)) : (0)];
              double v4429_data = v1180_g ? (v4429_data_pre) : (0.0);
              double v4485_data = glb_m3[(v5421_lead + 792)];
              double v4487_bc = sycl::select_from_group(item.get_sub_group(), v3590_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v4493_bc = sycl::select_from_group(item.get_sub_group(), v3596_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v4499_bc = sycl::select_from_group(item.get_sub_group(), v3602_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v4505_bc = sycl::select_from_group(item.get_sub_group(), v3608_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v4511_bc = sycl::select_from_group(item.get_sub_group(), v3614_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v4517_bc = sycl::select_from_group(item.get_sub_group(), v3620_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v4523_bc = sycl::select_from_group(item.get_sub_group(), v3626_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v4529_bc = sycl::select_from_group(item.get_sub_group(), v3632_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v4535_bc = sycl::select_from_group(item.get_sub_group(), v3638_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v4541_data_pre = glb_m3[v1180_g ? ((v1178_lead + 792)) : (0)];
              double v4541_data = v1180_g ? (v4541_data_pre) : (0.0);
              double v4597_data = glb_m3[(v5421_lead + 816)];
              double v4599_bc = sycl::select_from_group(item.get_sub_group(), v3590_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v4605_bc = sycl::select_from_group(item.get_sub_group(), v3596_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v4611_bc = sycl::select_from_group(item.get_sub_group(), v3602_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v4617_bc = sycl::select_from_group(item.get_sub_group(), v3608_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v4623_bc = sycl::select_from_group(item.get_sub_group(), v3614_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v4629_bc = sycl::select_from_group(item.get_sub_group(), v3620_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v4635_bc = sycl::select_from_group(item.get_sub_group(), v3626_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v4641_bc = sycl::select_from_group(item.get_sub_group(), v3632_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v4647_bc = sycl::select_from_group(item.get_sub_group(), v3638_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v4653_data_pre = glb_m3[v1180_g ? ((v1178_lead + 816)) : (0)];
              double v4653_data = v1180_g ? (v4653_data_pre) : (0.0);
              double v4709_data = glb_m3[(v5421_lead + 840)];
              double v4711_bc = sycl::select_from_group(item.get_sub_group(), v3590_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v4717_bc = sycl::select_from_group(item.get_sub_group(), v3596_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v4723_bc = sycl::select_from_group(item.get_sub_group(), v3602_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v4729_bc = sycl::select_from_group(item.get_sub_group(), v3608_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v4735_bc = sycl::select_from_group(item.get_sub_group(), v3614_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v4741_bc = sycl::select_from_group(item.get_sub_group(), v3620_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v4747_bc = sycl::select_from_group(item.get_sub_group(), v3626_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v4753_bc = sycl::select_from_group(item.get_sub_group(), v3632_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v4759_bc = sycl::select_from_group(item.get_sub_group(), v3638_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              double v4765_data_pre = glb_m3[v1180_g ? ((v1178_lead + 840)) : (0)];
              double v4765_data = v1180_g ? (v4765_data_pre) : (0.0);
              double v4821_data = glb_m3[(v5421_lead + 864)];
              double v4823_bc = sycl::select_from_group(item.get_sub_group(), v3590_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              double v4829_bc = sycl::select_from_group(item.get_sub_group(), v3596_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              double v4835_bc = sycl::select_from_group(item.get_sub_group(), v3602_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              double v4841_bc = sycl::select_from_group(item.get_sub_group(), v3608_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              double v4847_bc = sycl::select_from_group(item.get_sub_group(), v3614_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              double v4853_bc = sycl::select_from_group(item.get_sub_group(), v3620_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              double v4859_bc = sycl::select_from_group(item.get_sub_group(), v3626_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              double v4865_bc = sycl::select_from_group(item.get_sub_group(), v3632_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              double v4871_bc = sycl::select_from_group(item.get_sub_group(), v3638_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              double v4877_data_pre = glb_m3[v1180_g ? ((v1178_lead + 864)) : (0)];
              double v4877_data = v1180_g ? (v4877_data_pre) : (0.0);
              double v4933_data = glb_m3[(v5421_lead + 888)];
              double v4935_bc = sycl::select_from_group(item.get_sub_group(), v3590_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              double v4941_bc = sycl::select_from_group(item.get_sub_group(), v3596_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              double v4947_bc = sycl::select_from_group(item.get_sub_group(), v3602_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              double v4953_bc = sycl::select_from_group(item.get_sub_group(), v3608_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              double v4959_bc = sycl::select_from_group(item.get_sub_group(), v3614_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              double v4965_bc = sycl::select_from_group(item.get_sub_group(), v3620_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              double v4971_bc = sycl::select_from_group(item.get_sub_group(), v3626_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              double v4977_bc = sycl::select_from_group(item.get_sub_group(), v3632_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              double v4983_bc = sycl::select_from_group(item.get_sub_group(), v3638_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              double v4989_data_pre = glb_m3[v1180_g ? ((v1178_lead + 888)) : (0)];
              double v4989_data = v1180_g ? (v4989_data_pre) : (0.0);
              double v5042_p = (((((((v4090_p + (v4205_data * v4143_bc)) + (v4317_data * v4311_bc)) + (v4429_data * v4423_bc)) + (v4541_data * v4535_bc)) + (v4653_data * v4647_bc)) + (v4765_data * v4759_bc)) + (v4877_data * v4871_bc)) + (v4989_data * v4983_bc);
              double v5045_data = glb_m3[(v5421_lead + 912)];
              double v5047_bc = sycl::select_from_group(item.get_sub_group(), v3590_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              double v5050_p = ((((((((v3986_p + (v4149_data * v4095_bc)) + (v4261_data * v4263_bc)) + (v4373_data * v4375_bc)) + (v4485_data * v4487_bc)) + (v4597_data * v4599_bc)) + (v4709_data * v4711_bc)) + (v4821_data * v4823_bc)) + (v4933_data * v4935_bc)) + (v5045_data * v5047_bc);
              double v5053_bc = sycl::select_from_group(item.get_sub_group(), v3596_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              double v5056_p = ((((((((v3992_p + (v4149_data * v4101_bc)) + (v4261_data * v4269_bc)) + (v4373_data * v4381_bc)) + (v4485_data * v4493_bc)) + (v4597_data * v4605_bc)) + (v4709_data * v4717_bc)) + (v4821_data * v4829_bc)) + (v4933_data * v4941_bc)) + (v5045_data * v5053_bc);
              double v5059_bc = sycl::select_from_group(item.get_sub_group(), v3602_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              double v5062_p = ((((((((v3998_p + (v4149_data * v4107_bc)) + (v4261_data * v4275_bc)) + (v4373_data * v4387_bc)) + (v4485_data * v4499_bc)) + (v4597_data * v4611_bc)) + (v4709_data * v4723_bc)) + (v4821_data * v4835_bc)) + (v4933_data * v4947_bc)) + (v5045_data * v5059_bc);
              double v5065_bc = sycl::select_from_group(item.get_sub_group(), v3608_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              double v5068_p = ((((((((v4004_p + (v4149_data * v4113_bc)) + (v4261_data * v4281_bc)) + (v4373_data * v4393_bc)) + (v4485_data * v4505_bc)) + (v4597_data * v4617_bc)) + (v4709_data * v4729_bc)) + (v4821_data * v4841_bc)) + (v4933_data * v4953_bc)) + (v5045_data * v5065_bc);
              double v5071_bc = sycl::select_from_group(item.get_sub_group(), v3614_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              double v5074_p = ((((((((v4010_p + (v4149_data * v4119_bc)) + (v4261_data * v4287_bc)) + (v4373_data * v4399_bc)) + (v4485_data * v4511_bc)) + (v4597_data * v4623_bc)) + (v4709_data * v4735_bc)) + (v4821_data * v4847_bc)) + (v4933_data * v4959_bc)) + (v5045_data * v5071_bc);
              double v5077_bc = sycl::select_from_group(item.get_sub_group(), v3620_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              double v5080_p = ((((((((v4016_p + (v4149_data * v4125_bc)) + (v4261_data * v4293_bc)) + (v4373_data * v4405_bc)) + (v4485_data * v4517_bc)) + (v4597_data * v4629_bc)) + (v4709_data * v4741_bc)) + (v4821_data * v4853_bc)) + (v4933_data * v4965_bc)) + (v5045_data * v5077_bc);
              double v5083_bc = sycl::select_from_group(item.get_sub_group(), v3626_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              double v5086_p = ((((((((v4022_p + (v4149_data * v4131_bc)) + (v4261_data * v4299_bc)) + (v4373_data * v4411_bc)) + (v4485_data * v4523_bc)) + (v4597_data * v4635_bc)) + (v4709_data * v4747_bc)) + (v4821_data * v4859_bc)) + (v4933_data * v4971_bc)) + (v5045_data * v5083_bc);
              double v5089_bc = sycl::select_from_group(item.get_sub_group(), v3632_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              double v5092_p = ((((((((v4028_p + (v4149_data * v4137_bc)) + (v4261_data * v4305_bc)) + (v4373_data * v4417_bc)) + (v4485_data * v4529_bc)) + (v4597_data * v4641_bc)) + (v4709_data * v4753_bc)) + (v4821_data * v4865_bc)) + (v4933_data * v4977_bc)) + (v5045_data * v5089_bc);
              double v5095_bc = sycl::select_from_group(item.get_sub_group(), v3638_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              double v5098_p = ((((((((v4034_p + (v4149_data * v4143_bc)) + (v4261_data * v4311_bc)) + (v4373_data * v4423_bc)) + (v4485_data * v4535_bc)) + (v4597_data * v4647_bc)) + (v4709_data * v4759_bc)) + (v4821_data * v4871_bc)) + (v4933_data * v4983_bc)) + (v5045_data * v5095_bc);
              double v5101_data_pre = glb_m3[v1180_g ? ((v1178_lead + 912)) : (0)];
              double v5101_data = v1180_g ? (v5101_data_pre) : (0.0);
              double v5106_p = ((((((((v4042_p + (v4205_data * v4095_bc)) + (v4317_data * v4263_bc)) + (v4429_data * v4375_bc)) + (v4541_data * v4487_bc)) + (v4653_data * v4599_bc)) + (v4765_data * v4711_bc)) + (v4877_data * v4823_bc)) + (v4989_data * v4935_bc)) + (v5101_data * v5047_bc);
              double v5112_p = ((((((((v4048_p + (v4205_data * v4101_bc)) + (v4317_data * v4269_bc)) + (v4429_data * v4381_bc)) + (v4541_data * v4493_bc)) + (v4653_data * v4605_bc)) + (v4765_data * v4717_bc)) + (v4877_data * v4829_bc)) + (v4989_data * v4941_bc)) + (v5101_data * v5053_bc);
              double v5118_p = ((((((((v4054_p + (v4205_data * v4107_bc)) + (v4317_data * v4275_bc)) + (v4429_data * v4387_bc)) + (v4541_data * v4499_bc)) + (v4653_data * v4611_bc)) + (v4765_data * v4723_bc)) + (v4877_data * v4835_bc)) + (v4989_data * v4947_bc)) + (v5101_data * v5059_bc);
              double v5124_p = ((((((((v4060_p + (v4205_data * v4113_bc)) + (v4317_data * v4281_bc)) + (v4429_data * v4393_bc)) + (v4541_data * v4505_bc)) + (v4653_data * v4617_bc)) + (v4765_data * v4729_bc)) + (v4877_data * v4841_bc)) + (v4989_data * v4953_bc)) + (v5101_data * v5065_bc);
              double v5130_p = ((((((((v4066_p + (v4205_data * v4119_bc)) + (v4317_data * v4287_bc)) + (v4429_data * v4399_bc)) + (v4541_data * v4511_bc)) + (v4653_data * v4623_bc)) + (v4765_data * v4735_bc)) + (v4877_data * v4847_bc)) + (v4989_data * v4959_bc)) + (v5101_data * v5071_bc);
              double v5136_p = ((((((((v4072_p + (v4205_data * v4125_bc)) + (v4317_data * v4293_bc)) + (v4429_data * v4405_bc)) + (v4541_data * v4517_bc)) + (v4653_data * v4629_bc)) + (v4765_data * v4741_bc)) + (v4877_data * v4853_bc)) + (v4989_data * v4965_bc)) + (v5101_data * v5077_bc);
              double v5142_p = ((((((((v4078_p + (v4205_data * v4131_bc)) + (v4317_data * v4299_bc)) + (v4429_data * v4411_bc)) + (v4541_data * v4523_bc)) + (v4653_data * v4635_bc)) + (v4765_data * v4747_bc)) + (v4877_data * v4859_bc)) + (v4989_data * v4971_bc)) + (v5101_data * v5083_bc);
              double v5148_p = ((((((((v4084_p + (v4205_data * v4137_bc)) + (v4317_data * v4305_bc)) + (v4429_data * v4417_bc)) + (v4541_data * v4529_bc)) + (v4653_data * v4641_bc)) + (v4765_data * v4753_bc)) + (v4877_data * v4865_bc)) + (v4989_data * v4977_bc)) + (v5101_data * v5089_bc);
              double v5152_p = v5101_data * v5095_bc;
              // r0 = load{g>r}(wrap_glb_m0);
              #pragma unroll
              for (int32_t v5459_i0 = 0; v5459_i0 < 1; ++v5459_i0) {
                int32_t v5461_lead = v5419_lead + (v5459_i0 * 16);
                #pragma unroll
                for (int32_t v5462_i1 = 0; v5462_i1 < 9; ++v5462_i1) {
                  double v5465_data = wrap_glb_m0[(v5461_lead + (v5462_i1 * 30))];
                  r0[(v5459_i0 + (v5462_i1 * 2))] = v5465_data;
                }
              }
              #pragma unroll
              for (int32_t v5468_i1 = 0; v5468_i1 < 9; ++v5468_i1) {
                double v5471_data_pre = wrap_glb_m0[v5422_g ? ((v5421_lead + (v5468_i1 * 30))) : (0)];
                double v5471_data = v5422_g ? (v5471_data_pre) : (0.0);
                r0[(1 + (v5468_i1 * 2))] = v5471_data;
              }
              double v5157_data = glb_m3[(v5421_lead + 936)];
              double v5159_bc = sycl::select_from_group(item.get_sub_group(), v3590_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              double v5165_bc = sycl::select_from_group(item.get_sub_group(), v3596_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              double v5168_p = v5056_p + (v5157_data * v5165_bc);
              double v5171_bc = sycl::select_from_group(item.get_sub_group(), v3602_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              double v5174_p = v5062_p + (v5157_data * v5171_bc);
              double v5177_bc = sycl::select_from_group(item.get_sub_group(), v3608_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              double v5180_p = v5068_p + (v5157_data * v5177_bc);
              double v5183_bc = sycl::select_from_group(item.get_sub_group(), v3614_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              double v5186_p = v5074_p + (v5157_data * v5183_bc);
              double v5189_bc = sycl::select_from_group(item.get_sub_group(), v3620_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              double v5192_p = v5080_p + (v5157_data * v5189_bc);
              double v5195_bc = sycl::select_from_group(item.get_sub_group(), v3626_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              double v5198_p = v5086_p + (v5157_data * v5195_bc);
              double v5201_bc = sycl::select_from_group(item.get_sub_group(), v3632_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              double v5204_p = v5092_p + (v5157_data * v5201_bc);
              double v5207_bc = sycl::select_from_group(item.get_sub_group(), v3638_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              double v5210_p = v5098_p + (v5157_data * v5207_bc);
              double v5213_data_pre = glb_m3[v1180_g ? ((v1178_lead + 936)) : (0)];
              double v5213_data = v1180_g ? (v5213_data_pre) : (0.0);
              double v5218_p = v5106_p + (v5213_data * v5159_bc);
              double v5224_p = v5112_p + (v5213_data * v5165_bc);
              double v5230_p = v5118_p + (v5213_data * v5171_bc);
              double v5236_p = v5124_p + (v5213_data * v5177_bc);
              double v5242_p = v5130_p + (v5213_data * v5183_bc);
              double v5248_p = v5136_p + (v5213_data * v5189_bc);
              double v5254_p = v5142_p + (v5213_data * v5195_bc);
              double v5260_p = v5148_p + (v5213_data * v5201_bc);
              double v5266_p = (v5042_p + v5152_p) + (v5213_data * v5207_bc);
              double v5269_data = glb_m3[(v5421_lead + 960)];
              double v5271_bc = sycl::select_from_group(item.get_sub_group(), v3590_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              ir4[1] = ((v5050_p + (v5157_data * v5159_bc)) + (v5269_data * v5271_bc));
              double v5277_bc = sycl::select_from_group(item.get_sub_group(), v3596_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              ir4[4] = (v5168_p + (v5269_data * v5277_bc));
              double v5283_bc = sycl::select_from_group(item.get_sub_group(), v3602_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              ir4[7] = (v5174_p + (v5269_data * v5283_bc));
              double v5289_bc = sycl::select_from_group(item.get_sub_group(), v3608_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              ir4[10] = (v5180_p + (v5269_data * v5289_bc));
              double v5295_bc = sycl::select_from_group(item.get_sub_group(), v3614_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              ir4[13] = (v5186_p + (v5269_data * v5295_bc));
              double v5301_bc = sycl::select_from_group(item.get_sub_group(), v3620_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              ir4[16] = (v5192_p + (v5269_data * v5301_bc));
              double v5307_bc = sycl::select_from_group(item.get_sub_group(), v3626_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              ir4[19] = (v5198_p + (v5269_data * v5307_bc));
              double v5313_bc = sycl::select_from_group(item.get_sub_group(), v3632_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              ir4[22] = (v5204_p + (v5269_data * v5313_bc));
              double v5319_bc = sycl::select_from_group(item.get_sub_group(), v3638_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              ir4[25] = (v5210_p + (v5269_data * v5319_bc));
              double v5325_data_pre = glb_m3[v1180_g ? ((v1178_lead + 960)) : (0)];
              double v5325_data = v1180_g ? (v5325_data_pre) : (0.0);
              ir4[2] = (v5218_p + (v5325_data * v5271_bc));
              ir4[5] = (v5224_p + (v5325_data * v5277_bc));
              ir4[8] = (v5230_p + (v5325_data * v5283_bc));
              ir4[11] = (v5236_p + (v5325_data * v5289_bc));
              ir4[14] = (v5242_p + (v5325_data * v5295_bc));
              ir4[17] = (v5248_p + (v5325_data * v5301_bc));
              ir4[20] = (v5254_p + (v5325_data * v5307_bc));
              ir4[23] = (v5260_p + (v5325_data * v5313_bc));
              ir4[26] = (v5266_p + (v5325_data * v5319_bc));
              // r4 = ir4
              // glb_m2 = store{r>g}(r4);
              #pragma unroll
              for (int32_t v5389_i0 = 0; v5389_i0 < 2; ++v5389_i0) {
                int32_t v5395_lead = v5419_lead + (v5389_i0 * 16);
                #pragma unroll
                for (int32_t v5390_i1 = 0; v5390_i1 < 9; ++v5390_i1) {
                  double v5393_data = ir4[(v5389_i0 + (v5390_i1 * 3))];
                  glb_m2[(v5395_lead + (v5390_i1 * 40))] = v5393_data;
                }
              }
              #pragma unroll
              for (int32_t v5399_i1 = 0; v5399_i1 < 9; ++v5399_i1) {
                double v5402_data = ir4[((v5399_i1 * 3) + 2)];
                if (v1180_g) {
                  glb_m2[(v1178_lead + (v5399_i1 * 40))] = v5402_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
            }
            else {
              // r0 = load{g>r}(wrap_glb_m0);
              #pragma unroll
              for (int32_t v5429_i0 = 0; v5429_i0 < 1; ++v5429_i0) {
                int32_t v5431_lead = v5419_lead + (v5429_i0 * 16);
                #pragma unroll
                for (int32_t v5432_i1 = 0; v5432_i1 < 9; ++v5432_i1) {
                  double v5435_data = wrap_glb_m0[(v5431_lead + (v5432_i1 * 30))];
                  r0[(v5429_i0 + (v5432_i1 * 2))] = v5435_data;
                }
              }
              #pragma unroll
              for (int32_t v5438_i1 = 0; v5438_i1 < 9; ++v5438_i1) {
                double v5441_data_pre = wrap_glb_m0[v5422_g ? ((v5421_lead + (v5438_i1 * 30))) : (0)];
                double v5441_data = v5422_g ? (v5441_data_pre) : (0.0);
                r0[(1 + (v5438_i1 * 2))] = v5441_data;
              }
            }
            v5415_flagWord = v5416_flagWordNext;
            v5416_flagWordNext = flagWordAhead;
          }
        }
      });
    }
  });
}

