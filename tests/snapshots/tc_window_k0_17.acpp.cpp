// === base name ===
kernel_827b9b96238606d8

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_827b9b96238606d8 = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_827b9b96238606d8(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_827b9b96238606d8(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_827b9b96238606d8(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (16, 16, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 16;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_827b9b96238606d8(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_827b9b96238606d8(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_827b9b96238606d8(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_827b9b96238606d8(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 16×9(16×9) {0..16}×{0..9} strided
        //   m1 16×20(16×17) {0..16}×{0..17} none
        //   m2 20×9(17×9) {0..17}×{0..9} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,9]],"name":"m0","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[16,17]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[17,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[16,17]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[0,0],[17,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          const float *const __restrict__ glb_m1 = &m1[0];
          for (size_t v4_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v4_batchId0 < numElements0; v4_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v5_ahead1 = v4_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v4_batchId0 * 144 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v4_batchId0 * 153 + 0 + m2_extraOffset];
              float r0[18]{};
              // r0 = load{g>r}(glb_m2);
              int32_t v17_lead = item.get_local_id(2) % 16;
              #pragma unroll
              for (int32_t v18_i0 = 0; v18_i0 < 1; ++v18_i0) {
                int32_t v21_lead = v17_lead + (v18_i0 * 16);
                #pragma unroll
                for (int32_t v19_i1 = 0; v19_i1 < 9; ++v19_i1) {
                  float v24_data = glb_m2[(v21_lead + (v19_i1 * 17))];
                  r0[(v18_i0 + (v19_i1 * 2))] = v24_data;
                }
              }
              if (v17_lead < 1) {
                int32_t v30_lead = v17_lead + 16_i32;
                #pragma unroll
                for (int32_t v28_i1 = 0; v28_i1 < 9; ++v28_i1) {
                  float v33_data = glb_m2[(v30_lead + (v28_i1 * 17))];
                  r0[(1 + (v28_i1 * 2))] = v33_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m2););
              float r1[9]{};
              // ir1 = +(glb_m1 * r0)
              // [(0, 16), (0, 9)] [(0, 17)]
              float ir1[9]{};
              float v41_data = glb_m1[v17_lead];
              float v42_data = r0[0];
              float v45_data = ir1[0];
              ir1[0] = (v45_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v48_data = r0[2];
              float v51_data = ir1[1];
              ir1[1] = (v51_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v54_data = r0[4];
              float v57_data = ir1[2];
              ir1[2] = (v57_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v60_data = r0[6];
              float v63_data = ir1[3];
              ir1[3] = (v63_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v66_data = r0[8];
              float v69_data = ir1[4];
              ir1[4] = (v69_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v72_data = r0[10];
              float v75_data = ir1[5];
              ir1[5] = (v75_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v78_data = r0[12];
              float v81_data = ir1[6];
              ir1[6] = (v81_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v84_data = r0[14];
              float v87_data = ir1[7];
              ir1[7] = (v87_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v90_data = r0[16];
              float v93_data = ir1[8];
              ir1[8] = (v93_data + (v41_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v96_data = glb_m1[(v17_lead + 16)];
              float v100_data = ir1[0];
              ir1[0] = (v100_data + (v96_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v106_data = ir1[1];
              ir1[1] = (v106_data + (v96_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v112_data = ir1[2];
              ir1[2] = (v112_data + (v96_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v118_data = ir1[3];
              ir1[3] = (v118_data + (v96_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v124_data = ir1[4];
              ir1[4] = (v124_data + (v96_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v130_data = ir1[5];
              ir1[5] = (v130_data + (v96_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v136_data = ir1[6];
              ir1[6] = (v136_data + (v96_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v142_data = ir1[7];
              ir1[7] = (v142_data + (v96_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v148_data = ir1[8];
              ir1[8] = (v148_data + (v96_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v151_data = glb_m1[(v17_lead + 32)];
              float v155_data = ir1[0];
              ir1[0] = (v155_data + (v151_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v161_data = ir1[1];
              ir1[1] = (v161_data + (v151_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v167_data = ir1[2];
              ir1[2] = (v167_data + (v151_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v173_data = ir1[3];
              ir1[3] = (v173_data + (v151_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v179_data = ir1[4];
              ir1[4] = (v179_data + (v151_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v185_data = ir1[5];
              ir1[5] = (v185_data + (v151_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v191_data = ir1[6];
              ir1[6] = (v191_data + (v151_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v197_data = ir1[7];
              ir1[7] = (v197_data + (v151_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v203_data = ir1[8];
              ir1[8] = (v203_data + (v151_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v206_data = glb_m1[(v17_lead + 48)];
              float v210_data = ir1[0];
              ir1[0] = (v210_data + (v206_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v216_data = ir1[1];
              ir1[1] = (v216_data + (v206_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v222_data = ir1[2];
              ir1[2] = (v222_data + (v206_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v228_data = ir1[3];
              ir1[3] = (v228_data + (v206_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v234_data = ir1[4];
              ir1[4] = (v234_data + (v206_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v240_data = ir1[5];
              ir1[5] = (v240_data + (v206_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v246_data = ir1[6];
              ir1[6] = (v246_data + (v206_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v252_data = ir1[7];
              ir1[7] = (v252_data + (v206_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v258_data = ir1[8];
              ir1[8] = (v258_data + (v206_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v261_data = glb_m1[(v17_lead + 64)];
              float v265_data = ir1[0];
              ir1[0] = (v265_data + (v261_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v271_data = ir1[1];
              ir1[1] = (v271_data + (v261_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v277_data = ir1[2];
              ir1[2] = (v277_data + (v261_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v283_data = ir1[3];
              ir1[3] = (v283_data + (v261_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v289_data = ir1[4];
              ir1[4] = (v289_data + (v261_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v295_data = ir1[5];
              ir1[5] = (v295_data + (v261_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v301_data = ir1[6];
              ir1[6] = (v301_data + (v261_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v307_data = ir1[7];
              ir1[7] = (v307_data + (v261_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v313_data = ir1[8];
              ir1[8] = (v313_data + (v261_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v316_data = glb_m1[(v17_lead + 80)];
              float v320_data = ir1[0];
              ir1[0] = (v320_data + (v316_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v326_data = ir1[1];
              ir1[1] = (v326_data + (v316_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v332_data = ir1[2];
              ir1[2] = (v332_data + (v316_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v338_data = ir1[3];
              ir1[3] = (v338_data + (v316_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v344_data = ir1[4];
              ir1[4] = (v344_data + (v316_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v350_data = ir1[5];
              ir1[5] = (v350_data + (v316_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v356_data = ir1[6];
              ir1[6] = (v356_data + (v316_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v362_data = ir1[7];
              ir1[7] = (v362_data + (v316_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v368_data = ir1[8];
              ir1[8] = (v368_data + (v316_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v371_data = glb_m1[(v17_lead + 96)];
              float v375_data = ir1[0];
              ir1[0] = (v375_data + (v371_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v381_data = ir1[1];
              ir1[1] = (v381_data + (v371_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v387_data = ir1[2];
              ir1[2] = (v387_data + (v371_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v393_data = ir1[3];
              ir1[3] = (v393_data + (v371_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v399_data = ir1[4];
              ir1[4] = (v399_data + (v371_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v405_data = ir1[5];
              ir1[5] = (v405_data + (v371_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v411_data = ir1[6];
              ir1[6] = (v411_data + (v371_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v417_data = ir1[7];
              ir1[7] = (v417_data + (v371_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v423_data = ir1[8];
              ir1[8] = (v423_data + (v371_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v426_data = glb_m1[(v17_lead + 112)];
              float v430_data = ir1[0];
              ir1[0] = (v430_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v436_data = ir1[1];
              ir1[1] = (v436_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v442_data = ir1[2];
              ir1[2] = (v442_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v448_data = ir1[3];
              ir1[3] = (v448_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v454_data = ir1[4];
              ir1[4] = (v454_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v460_data = ir1[5];
              ir1[5] = (v460_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v466_data = ir1[6];
              ir1[6] = (v466_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v472_data = ir1[7];
              ir1[7] = (v472_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v478_data = ir1[8];
              ir1[8] = (v478_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v481_data = glb_m1[(v17_lead + 128)];
              float v485_data = ir1[0];
              ir1[0] = (v485_data + (v481_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v491_data = ir1[1];
              ir1[1] = (v491_data + (v481_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v497_data = ir1[2];
              ir1[2] = (v497_data + (v481_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v503_data = ir1[3];
              ir1[3] = (v503_data + (v481_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v509_data = ir1[4];
              ir1[4] = (v509_data + (v481_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v515_data = ir1[5];
              ir1[5] = (v515_data + (v481_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v521_data = ir1[6];
              ir1[6] = (v521_data + (v481_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v527_data = ir1[7];
              ir1[7] = (v527_data + (v481_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v533_data = ir1[8];
              ir1[8] = (v533_data + (v481_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v536_data = glb_m1[(v17_lead + 144)];
              float v540_data = ir1[0];
              ir1[0] = (v540_data + (v536_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v546_data = ir1[1];
              ir1[1] = (v546_data + (v536_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v552_data = ir1[2];
              ir1[2] = (v552_data + (v536_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v558_data = ir1[3];
              ir1[3] = (v558_data + (v536_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v564_data = ir1[4];
              ir1[4] = (v564_data + (v536_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v570_data = ir1[5];
              ir1[5] = (v570_data + (v536_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v576_data = ir1[6];
              ir1[6] = (v576_data + (v536_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v582_data = ir1[7];
              ir1[7] = (v582_data + (v536_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v588_data = ir1[8];
              ir1[8] = (v588_data + (v536_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v591_data = glb_m1[(v17_lead + 160)];
              float v595_data = ir1[0];
              ir1[0] = (v595_data + (v591_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v601_data = ir1[1];
              ir1[1] = (v601_data + (v591_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v607_data = ir1[2];
              ir1[2] = (v607_data + (v591_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v613_data = ir1[3];
              ir1[3] = (v613_data + (v591_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v619_data = ir1[4];
              ir1[4] = (v619_data + (v591_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v625_data = ir1[5];
              ir1[5] = (v625_data + (v591_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v631_data = ir1[6];
              ir1[6] = (v631_data + (v591_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v637_data = ir1[7];
              ir1[7] = (v637_data + (v591_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v643_data = ir1[8];
              ir1[8] = (v643_data + (v591_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v646_data = glb_m1[(v17_lead + 176)];
              float v650_data = ir1[0];
              ir1[0] = (v650_data + (v646_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v656_data = ir1[1];
              ir1[1] = (v656_data + (v646_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v662_data = ir1[2];
              ir1[2] = (v662_data + (v646_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v668_data = ir1[3];
              ir1[3] = (v668_data + (v646_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v674_data = ir1[4];
              ir1[4] = (v674_data + (v646_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v680_data = ir1[5];
              ir1[5] = (v680_data + (v646_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v686_data = ir1[6];
              ir1[6] = (v686_data + (v646_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v692_data = ir1[7];
              ir1[7] = (v692_data + (v646_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v698_data = ir1[8];
              ir1[8] = (v698_data + (v646_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v701_data = glb_m1[(v17_lead + 192)];
              float v705_data = ir1[0];
              ir1[0] = (v705_data + (v701_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v711_data = ir1[1];
              ir1[1] = (v711_data + (v701_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v717_data = ir1[2];
              ir1[2] = (v717_data + (v701_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v723_data = ir1[3];
              ir1[3] = (v723_data + (v701_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v729_data = ir1[4];
              ir1[4] = (v729_data + (v701_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v735_data = ir1[5];
              ir1[5] = (v735_data + (v701_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v741_data = ir1[6];
              ir1[6] = (v741_data + (v701_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v747_data = ir1[7];
              ir1[7] = (v747_data + (v701_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v753_data = ir1[8];
              ir1[8] = (v753_data + (v701_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v756_data = glb_m1[(v17_lead + 208)];
              float v760_data = ir1[0];
              ir1[0] = (v760_data + (v756_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v766_data = ir1[1];
              ir1[1] = (v766_data + (v756_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v772_data = ir1[2];
              ir1[2] = (v772_data + (v756_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v778_data = ir1[3];
              ir1[3] = (v778_data + (v756_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v784_data = ir1[4];
              ir1[4] = (v784_data + (v756_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v790_data = ir1[5];
              ir1[5] = (v790_data + (v756_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v796_data = ir1[6];
              ir1[6] = (v796_data + (v756_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v802_data = ir1[7];
              ir1[7] = (v802_data + (v756_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v808_data = ir1[8];
              ir1[8] = (v808_data + (v756_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v811_data = glb_m1[(v17_lead + 224)];
              float v815_data = ir1[0];
              ir1[0] = (v815_data + (v811_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v821_data = ir1[1];
              ir1[1] = (v821_data + (v811_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v827_data = ir1[2];
              ir1[2] = (v827_data + (v811_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v833_data = ir1[3];
              ir1[3] = (v833_data + (v811_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v839_data = ir1[4];
              ir1[4] = (v839_data + (v811_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v845_data = ir1[5];
              ir1[5] = (v845_data + (v811_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v851_data = ir1[6];
              ir1[6] = (v851_data + (v811_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v857_data = ir1[7];
              ir1[7] = (v857_data + (v811_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v863_data = ir1[8];
              ir1[8] = (v863_data + (v811_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v866_data = glb_m1[(v17_lead + 240)];
              float v870_data = ir1[0];
              ir1[0] = (v870_data + (v866_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v876_data = ir1[1];
              ir1[1] = (v876_data + (v866_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v882_data = ir1[2];
              ir1[2] = (v882_data + (v866_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v888_data = ir1[3];
              ir1[3] = (v888_data + (v866_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v894_data = ir1[4];
              ir1[4] = (v894_data + (v866_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v900_data = ir1[5];
              ir1[5] = (v900_data + (v866_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v906_data = ir1[6];
              ir1[6] = (v906_data + (v866_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v912_data = ir1[7];
              ir1[7] = (v912_data + (v866_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v918_data = ir1[8];
              ir1[8] = (v918_data + (v866_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v921_data = glb_m1[(v17_lead + 256)];
              float v922_data = r0[1];
              float v925_data = ir1[0];
              ir1[0] = (v925_data + (v921_data * (sycl::select_from_group(item.get_sub_group(), v922_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v928_data = r0[3];
              float v931_data = ir1[1];
              ir1[1] = (v931_data + (v921_data * (sycl::select_from_group(item.get_sub_group(), v928_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v934_data = r0[5];
              float v937_data = ir1[2];
              ir1[2] = (v937_data + (v921_data * (sycl::select_from_group(item.get_sub_group(), v934_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v940_data = r0[7];
              float v943_data = ir1[3];
              ir1[3] = (v943_data + (v921_data * (sycl::select_from_group(item.get_sub_group(), v940_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v946_data = r0[9];
              float v949_data = ir1[4];
              ir1[4] = (v949_data + (v921_data * (sycl::select_from_group(item.get_sub_group(), v946_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v952_data = r0[11];
              float v955_data = ir1[5];
              ir1[5] = (v955_data + (v921_data * (sycl::select_from_group(item.get_sub_group(), v952_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v958_data = r0[13];
              float v961_data = ir1[6];
              ir1[6] = (v961_data + (v921_data * (sycl::select_from_group(item.get_sub_group(), v958_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v964_data = r0[15];
              float v967_data = ir1[7];
              ir1[7] = (v967_data + (v921_data * (sycl::select_from_group(item.get_sub_group(), v964_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v970_data = r0[17];
              float v973_data = ir1[8];
              ir1[8] = (v973_data + (v921_data * (sycl::select_from_group(item.get_sub_group(), v970_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              // r1 = ir1
              #pragma unroll
              for (int32_t v975_n0 = 0; v975_n0 < 1; ++v975_n0) {
                #pragma unroll
                for (int32_t v976_n1 = 0; v976_n1 < 9; ++v976_n1) {
                  int32_t v977_a = v975_n0 + v976_n1;
                  float v978_data = ir1[v977_a];
                  r1[v977_a] = v978_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v979_i0 = 0; v979_i0 < 1; ++v979_i0) {
                int32_t v984_lead = v17_lead + (v979_i0 * 16);
                #pragma unroll
                for (int32_t v980_i1 = 0; v980_i1 < 9; ++v980_i1) {
                  float v982_data = r1[(v979_i0 + v980_i1)];
                  glb_m0[(v984_lead + (v980_i1 * 16))] = v982_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
            }
          }
        }
      });
    }
  });
}

