// === base name ===
kernel_03f8ae31f1bf34b7

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_03f8ae31f1bf34b7 = {{16, 16, 1}, 16, 9, 1, 16, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_03f8ae31f1bf34b7(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_03f8ae31f1bf34b7(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_03f8ae31f1bf34b7(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_03f8ae31f1bf34b7(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_03f8ae31f1bf34b7(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_03f8ae31f1bf34b7(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, elementIndices, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_03f8ae31f1bf34b7(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, const unsigned * elementIndices, unsigned * flags0) {
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
          size_t v541_flagSecondAhead = v3_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
          const uint32_t flagWordSecond = flags0 == nullptr ? 1 : flags0[((v541_flagSecondAhead < numElements0) ? v541_flagSecondAhead : v3_batchId1)];
          const bool allowed_peel = static_cast<bool>(flagWordFirst);
          const size_t peel_elementIndex = static_cast<size_t>(elementIndices[v3_batchId1]);
          const double *const __restrict__ peel_glb_m1 = &m1[peel_elementIndex * 81 + 0 + 8 + m1_extraOffset];
          int32_t v552_lead = item.get_local_id(2) % 16;
          bool v553_g = v552_lead < 9;
          bool v571_g = allowed_peel && v553_g;
          // r0 = load{g>r}(peel_glb_m1);
          #pragma unroll
          for (int32_t v563_i1 = 0; v563_i1 < 9; ++v563_i1) {
            double v566_data_pre = peel_glb_m1[v571_g ? ((v552_lead + (v563_i1 * 9))) : (0)];
            double v566_data = v571_g ? (v566_data_pre) : (0.0);
            r0[v563_i1] = v566_data;
          }
          uint32_t v545_flagWord = flagWordFirst;
          uint32_t v546_flagWordNext = flagWordSecond;
          size_t v581_elementIndex = peel_elementIndex;
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const size_t wrap_elementIndex = static_cast<size_t>(elementIndices[v10_batchId1]);
            const double *const __restrict__ wrap_glb_m1 = &m1[wrap_elementIndex * 81 + 0 + 8 + m1_extraOffset];
            size_t v576_flagAheadAhead = v10_batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            const uint32_t flagWordAhead = flags0 == nullptr ? 1 : flags0[((v576_flagAheadAhead < numElements0) ? v576_flagAheadAhead : v10_batchId1)];
            const bool allowed = static_cast<bool>(v545_flagWord);
            const bool allowed_next = static_cast<bool>(v546_flagWordNext);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[v581_elementIndex * 81 + 0 + m0_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[v581_elementIndex * 96 + 0 + m2_extraOffset];
              double r1[9]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v32_i1 = 0; v32_i1 < 9; ++v32_i1) {
                double v38_data_pre = glb_m2[v553_g ? ((v552_lead + (v32_i1 * 9))) : (0)];
                double v38_data = v553_g ? (v38_data_pre) : (0.0);
                r1[v32_i1] = v38_data;
              }
              double r2[9]{};
              // ir2 = +(r0 * r1)
              // [(0, 9), (0, 9)] [(0, 9)]
              double ir2[9]{};
              double v42_data = r0[0];
              double v43_data = r1[0];
              double v44_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v49_data = r1[1];
              double v50_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v55_data = r1[2];
              double v56_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v61_data = r1[3];
              double v62_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v67_data = r1[4];
              double v68_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v73_data = r1[5];
              double v74_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v79_data = r1[6];
              double v80_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v85_data = r1[7];
              double v86_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v91_data = r1[8];
              double v92_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              double v96_data = r0[1];
              double v98_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v104_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v110_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v116_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v122_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v128_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v134_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v140_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v146_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              double v150_data = r0[2];
              double v152_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v158_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v164_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v170_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v176_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v182_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v188_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v194_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v200_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              double v204_data = r0[3];
              double v206_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v212_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v218_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v224_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v230_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v236_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v242_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v248_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v254_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              double v258_data = r0[4];
              double v260_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v266_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v272_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v278_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v284_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v290_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v296_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v302_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v308_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              double v312_data = r0[5];
              double v314_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v320_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v326_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v332_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v338_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v344_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v350_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v356_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v362_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              double v366_data = r0[6];
              double v368_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v374_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v380_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v386_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v392_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v398_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v404_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v410_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v416_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              double v420_data = r0[7];
              double v422_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v425_p = (((((((v42_data * v44_bc) + (v96_data * v98_bc)) + (v150_data * v152_bc)) + (v204_data * v206_bc)) + (v258_data * v260_bc)) + (v312_data * v314_bc)) + (v366_data * v368_bc)) + (v420_data * v422_bc);
              double v428_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v431_p = (((((((v42_data * v50_bc) + (v96_data * v104_bc)) + (v150_data * v158_bc)) + (v204_data * v212_bc)) + (v258_data * v266_bc)) + (v312_data * v320_bc)) + (v366_data * v374_bc)) + (v420_data * v428_bc);
              double v434_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v437_p = (((((((v42_data * v56_bc) + (v96_data * v110_bc)) + (v150_data * v164_bc)) + (v204_data * v218_bc)) + (v258_data * v272_bc)) + (v312_data * v326_bc)) + (v366_data * v380_bc)) + (v420_data * v434_bc);
              double v440_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v443_p = (((((((v42_data * v62_bc) + (v96_data * v116_bc)) + (v150_data * v170_bc)) + (v204_data * v224_bc)) + (v258_data * v278_bc)) + (v312_data * v332_bc)) + (v366_data * v386_bc)) + (v420_data * v440_bc);
              double v446_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v449_p = (((((((v42_data * v68_bc) + (v96_data * v122_bc)) + (v150_data * v176_bc)) + (v204_data * v230_bc)) + (v258_data * v284_bc)) + (v312_data * v338_bc)) + (v366_data * v392_bc)) + (v420_data * v446_bc);
              double v452_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v455_p = (((((((v42_data * v74_bc) + (v96_data * v128_bc)) + (v150_data * v182_bc)) + (v204_data * v236_bc)) + (v258_data * v290_bc)) + (v312_data * v344_bc)) + (v366_data * v398_bc)) + (v420_data * v452_bc);
              double v458_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v461_p = (((((((v42_data * v80_bc) + (v96_data * v134_bc)) + (v150_data * v188_bc)) + (v204_data * v242_bc)) + (v258_data * v296_bc)) + (v312_data * v350_bc)) + (v366_data * v404_bc)) + (v420_data * v458_bc);
              double v464_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v467_p = (((((((v42_data * v86_bc) + (v96_data * v140_bc)) + (v150_data * v194_bc)) + (v204_data * v248_bc)) + (v258_data * v302_bc)) + (v312_data * v356_bc)) + (v366_data * v410_bc)) + (v420_data * v464_bc);
              double v470_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              double v473_p = (((((((v42_data * v92_bc) + (v96_data * v146_bc)) + (v150_data * v200_bc)) + (v204_data * v254_bc)) + (v258_data * v308_bc)) + (v312_data * v362_bc)) + (v366_data * v416_bc)) + (v420_data * v470_bc);
              double v474_data = r0[8];
              bool v572_g = allowed_next && v553_g;
              // r0 = load{g>r}(wrap_glb_m1);
              #pragma unroll
              for (int32_t v567_i1 = 0; v567_i1 < 9; ++v567_i1) {
                double v570_data_pre = wrap_glb_m1[v572_g ? ((v552_lead + (v567_i1 * 9))) : (0)];
                double v570_data = v572_g ? (v570_data_pre) : (0.0);
                r0[v567_i1] = v570_data;
              }
              double v476_bc = sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              ir2[0] = (v425_p + (v474_data * v476_bc));
              double v482_bc = sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              ir2[1] = (v431_p + (v474_data * v482_bc));
              double v488_bc = sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              ir2[2] = (v437_p + (v474_data * v488_bc));
              double v494_bc = sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              ir2[3] = (v443_p + (v474_data * v494_bc));
              double v500_bc = sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              ir2[4] = (v449_p + (v474_data * v500_bc));
              double v506_bc = sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              ir2[5] = (v455_p + (v474_data * v506_bc));
              double v512_bc = sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              ir2[6] = (v461_p + (v474_data * v512_bc));
              double v518_bc = sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              ir2[7] = (v467_p + (v474_data * v518_bc));
              double v524_bc = sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              ir2[8] = (v473_p + (v474_data * v524_bc));
              // r2 = ir2
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v533_i1 = 0; v533_i1 < 9; ++v533_i1) {
                double v535_data = ir2[v533_i1];
                if (v553_g) {
                  glb_m0[(v552_lead + (v533_i1 * 9))] = v535_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
            }
            else {
              bool v573_g = allowed_next && v553_g;
              // r0 = load{g>r}(wrap_glb_m1);
              #pragma unroll
              for (int32_t v559_i1 = 0; v559_i1 < 9; ++v559_i1) {
                double v562_data_pre = wrap_glb_m1[v573_g ? ((v552_lead + (v559_i1 * 9))) : (0)];
                double v562_data = v573_g ? (v562_data_pre) : (0.0);
                r0[v559_i1] = v562_data;
              }
            }
            v545_flagWord = v546_flagWordNext;
            v546_flagWordNext = flagWordAhead;
            v581_elementIndex = wrap_elementIndex;
          }
        }
      });
    }
  });
}

