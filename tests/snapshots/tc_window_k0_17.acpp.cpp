// === base name ===
kernel_bbf50da60270b5ac

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_bbf50da60270b5ac = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_bbf50da60270b5ac(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_bbf50da60270b5ac(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_bbf50da60270b5ac(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_bbf50da60270b5ac(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_bbf50da60270b5ac(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_bbf50da60270b5ac(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_bbf50da60270b5ac(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 16×9(16×9) {0..16}×{0..9} strided
        //   m1 16×20(16×17) {0..16}×{0..17} none
        //   m2 20×9(17×9) {0..17}×{0..9} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,9]],"name":"m0","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[16,17]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[17,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[16,17]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[0,0],[17,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          const float *const __restrict__ glb_m1 = &m1[0];
          for (size_t v10_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v10_batchId0 < numElements0; v10_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v11_ahead1 = v10_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v13_batchId1 = (v11_ahead1 < numElements0) ? v11_ahead1 : v10_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v10_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v10_batchId0 * 144 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v10_batchId0 * 153 + 0 + m2_extraOffset];
              float r0[18]{};
              // r0 = load{g>r}(glb_m2);
              int32_t v23_lead = item.get_local_id(2) % 16;
              #pragma unroll
              for (int32_t v24_i0 = 0; v24_i0 < 1; ++v24_i0) {
                int32_t v27_lead = v23_lead + (v24_i0 * 16);
                #pragma unroll
                for (int32_t v25_i1 = 0; v25_i1 < 9; ++v25_i1) {
                  float v30_data = glb_m2[(v27_lead + (v25_i1 * 17))];
                  r0[(v24_i0 + (v25_i1 * 2))] = v30_data;
                }
              }
              if (v23_lead < 1) {
                int32_t v36_lead = v23_lead + 16_i32;
                #pragma unroll
                for (int32_t v34_i1 = 0; v34_i1 < 9; ++v34_i1) {
                  float v39_data = glb_m2[(v36_lead + (v34_i1 * 17))];
                  r0[(1 + (v34_i1 * 2))] = v39_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m2););
              float r1[9]{};
              // ir1 = +(glb_m1 * r0)
              // [(0, 16), (0, 9)] [(0, 17)]
              float ir1[9]{};
              float v47_data = glb_m1[v23_lead];
              float v48_data = r0[0];
              float v51_data = ir1[0];
              ir1[0] = (v51_data + (v47_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v54_data = r0[2];
              float v57_data = ir1[1];
              ir1[1] = (v57_data + (v47_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v60_data = r0[4];
              float v63_data = ir1[2];
              ir1[2] = (v63_data + (v47_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v66_data = r0[6];
              float v69_data = ir1[3];
              ir1[3] = (v69_data + (v47_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v72_data = r0[8];
              float v75_data = ir1[4];
              ir1[4] = (v75_data + (v47_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v78_data = r0[10];
              float v81_data = ir1[5];
              ir1[5] = (v81_data + (v47_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v84_data = r0[12];
              float v87_data = ir1[6];
              ir1[6] = (v87_data + (v47_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v90_data = r0[14];
              float v93_data = ir1[7];
              ir1[7] = (v93_data + (v47_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v96_data = r0[16];
              float v99_data = ir1[8];
              ir1[8] = (v99_data + (v47_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v102_data = glb_m1[(v23_lead + 16)];
              float v106_data = ir1[0];
              ir1[0] = (v106_data + (v102_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v112_data = ir1[1];
              ir1[1] = (v112_data + (v102_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v118_data = ir1[2];
              ir1[2] = (v118_data + (v102_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v124_data = ir1[3];
              ir1[3] = (v124_data + (v102_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v130_data = ir1[4];
              ir1[4] = (v130_data + (v102_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v136_data = ir1[5];
              ir1[5] = (v136_data + (v102_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v142_data = ir1[6];
              ir1[6] = (v142_data + (v102_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v148_data = ir1[7];
              ir1[7] = (v148_data + (v102_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v154_data = ir1[8];
              ir1[8] = (v154_data + (v102_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v157_data = glb_m1[(v23_lead + 32)];
              float v161_data = ir1[0];
              ir1[0] = (v161_data + (v157_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v167_data = ir1[1];
              ir1[1] = (v167_data + (v157_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v173_data = ir1[2];
              ir1[2] = (v173_data + (v157_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v179_data = ir1[3];
              ir1[3] = (v179_data + (v157_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v185_data = ir1[4];
              ir1[4] = (v185_data + (v157_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v191_data = ir1[5];
              ir1[5] = (v191_data + (v157_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v197_data = ir1[6];
              ir1[6] = (v197_data + (v157_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v203_data = ir1[7];
              ir1[7] = (v203_data + (v157_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v209_data = ir1[8];
              ir1[8] = (v209_data + (v157_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v212_data = glb_m1[(v23_lead + 48)];
              float v216_data = ir1[0];
              ir1[0] = (v216_data + (v212_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v222_data = ir1[1];
              ir1[1] = (v222_data + (v212_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v228_data = ir1[2];
              ir1[2] = (v228_data + (v212_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v234_data = ir1[3];
              ir1[3] = (v234_data + (v212_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v240_data = ir1[4];
              ir1[4] = (v240_data + (v212_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v246_data = ir1[5];
              ir1[5] = (v246_data + (v212_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v252_data = ir1[6];
              ir1[6] = (v252_data + (v212_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v258_data = ir1[7];
              ir1[7] = (v258_data + (v212_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v264_data = ir1[8];
              ir1[8] = (v264_data + (v212_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v267_data = glb_m1[(v23_lead + 64)];
              float v271_data = ir1[0];
              ir1[0] = (v271_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v277_data = ir1[1];
              ir1[1] = (v277_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v283_data = ir1[2];
              ir1[2] = (v283_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v289_data = ir1[3];
              ir1[3] = (v289_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v295_data = ir1[4];
              ir1[4] = (v295_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v301_data = ir1[5];
              ir1[5] = (v301_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v307_data = ir1[6];
              ir1[6] = (v307_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v313_data = ir1[7];
              ir1[7] = (v313_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v319_data = ir1[8];
              ir1[8] = (v319_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v322_data = glb_m1[(v23_lead + 80)];
              float v326_data = ir1[0];
              ir1[0] = (v326_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v332_data = ir1[1];
              ir1[1] = (v332_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v338_data = ir1[2];
              ir1[2] = (v338_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v344_data = ir1[3];
              ir1[3] = (v344_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v350_data = ir1[4];
              ir1[4] = (v350_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v356_data = ir1[5];
              ir1[5] = (v356_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v362_data = ir1[6];
              ir1[6] = (v362_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v368_data = ir1[7];
              ir1[7] = (v368_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v374_data = ir1[8];
              ir1[8] = (v374_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v377_data = glb_m1[(v23_lead + 96)];
              float v381_data = ir1[0];
              ir1[0] = (v381_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v387_data = ir1[1];
              ir1[1] = (v387_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v393_data = ir1[2];
              ir1[2] = (v393_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v399_data = ir1[3];
              ir1[3] = (v399_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v405_data = ir1[4];
              ir1[4] = (v405_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v411_data = ir1[5];
              ir1[5] = (v411_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v417_data = ir1[6];
              ir1[6] = (v417_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v423_data = ir1[7];
              ir1[7] = (v423_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v429_data = ir1[8];
              ir1[8] = (v429_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v432_data = glb_m1[(v23_lead + 112)];
              float v436_data = ir1[0];
              ir1[0] = (v436_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v442_data = ir1[1];
              ir1[1] = (v442_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v448_data = ir1[2];
              ir1[2] = (v448_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v454_data = ir1[3];
              ir1[3] = (v454_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v460_data = ir1[4];
              ir1[4] = (v460_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v466_data = ir1[5];
              ir1[5] = (v466_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v472_data = ir1[6];
              ir1[6] = (v472_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v478_data = ir1[7];
              ir1[7] = (v478_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v484_data = ir1[8];
              ir1[8] = (v484_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v487_data = glb_m1[(v23_lead + 128)];
              float v491_data = ir1[0];
              ir1[0] = (v491_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v497_data = ir1[1];
              ir1[1] = (v497_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v503_data = ir1[2];
              ir1[2] = (v503_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v509_data = ir1[3];
              ir1[3] = (v509_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v515_data = ir1[4];
              ir1[4] = (v515_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v521_data = ir1[5];
              ir1[5] = (v521_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v527_data = ir1[6];
              ir1[6] = (v527_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v533_data = ir1[7];
              ir1[7] = (v533_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v539_data = ir1[8];
              ir1[8] = (v539_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v542_data = glb_m1[(v23_lead + 144)];
              float v546_data = ir1[0];
              ir1[0] = (v546_data + (v542_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v552_data = ir1[1];
              ir1[1] = (v552_data + (v542_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v558_data = ir1[2];
              ir1[2] = (v558_data + (v542_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v564_data = ir1[3];
              ir1[3] = (v564_data + (v542_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v570_data = ir1[4];
              ir1[4] = (v570_data + (v542_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v576_data = ir1[5];
              ir1[5] = (v576_data + (v542_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v582_data = ir1[6];
              ir1[6] = (v582_data + (v542_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v588_data = ir1[7];
              ir1[7] = (v588_data + (v542_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v594_data = ir1[8];
              ir1[8] = (v594_data + (v542_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v597_data = glb_m1[(v23_lead + 160)];
              float v601_data = ir1[0];
              ir1[0] = (v601_data + (v597_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v607_data = ir1[1];
              ir1[1] = (v607_data + (v597_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v613_data = ir1[2];
              ir1[2] = (v613_data + (v597_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v619_data = ir1[3];
              ir1[3] = (v619_data + (v597_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v625_data = ir1[4];
              ir1[4] = (v625_data + (v597_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v631_data = ir1[5];
              ir1[5] = (v631_data + (v597_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v637_data = ir1[6];
              ir1[6] = (v637_data + (v597_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v643_data = ir1[7];
              ir1[7] = (v643_data + (v597_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v649_data = ir1[8];
              ir1[8] = (v649_data + (v597_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v652_data = glb_m1[(v23_lead + 176)];
              float v656_data = ir1[0];
              ir1[0] = (v656_data + (v652_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v662_data = ir1[1];
              ir1[1] = (v662_data + (v652_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v668_data = ir1[2];
              ir1[2] = (v668_data + (v652_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v674_data = ir1[3];
              ir1[3] = (v674_data + (v652_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v680_data = ir1[4];
              ir1[4] = (v680_data + (v652_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v686_data = ir1[5];
              ir1[5] = (v686_data + (v652_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v692_data = ir1[6];
              ir1[6] = (v692_data + (v652_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v698_data = ir1[7];
              ir1[7] = (v698_data + (v652_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v704_data = ir1[8];
              ir1[8] = (v704_data + (v652_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v707_data = glb_m1[(v23_lead + 192)];
              float v711_data = ir1[0];
              ir1[0] = (v711_data + (v707_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v717_data = ir1[1];
              ir1[1] = (v717_data + (v707_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v723_data = ir1[2];
              ir1[2] = (v723_data + (v707_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v729_data = ir1[3];
              ir1[3] = (v729_data + (v707_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v735_data = ir1[4];
              ir1[4] = (v735_data + (v707_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v741_data = ir1[5];
              ir1[5] = (v741_data + (v707_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v747_data = ir1[6];
              ir1[6] = (v747_data + (v707_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v753_data = ir1[7];
              ir1[7] = (v753_data + (v707_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v759_data = ir1[8];
              ir1[8] = (v759_data + (v707_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v762_data = glb_m1[(v23_lead + 208)];
              float v766_data = ir1[0];
              ir1[0] = (v766_data + (v762_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v772_data = ir1[1];
              ir1[1] = (v772_data + (v762_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v778_data = ir1[2];
              ir1[2] = (v778_data + (v762_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v784_data = ir1[3];
              ir1[3] = (v784_data + (v762_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v790_data = ir1[4];
              ir1[4] = (v790_data + (v762_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v796_data = ir1[5];
              ir1[5] = (v796_data + (v762_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v802_data = ir1[6];
              ir1[6] = (v802_data + (v762_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v808_data = ir1[7];
              ir1[7] = (v808_data + (v762_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v814_data = ir1[8];
              ir1[8] = (v814_data + (v762_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v817_data = glb_m1[(v23_lead + 224)];
              float v821_data = ir1[0];
              ir1[0] = (v821_data + (v817_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v827_data = ir1[1];
              ir1[1] = (v827_data + (v817_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v833_data = ir1[2];
              ir1[2] = (v833_data + (v817_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v839_data = ir1[3];
              ir1[3] = (v839_data + (v817_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v845_data = ir1[4];
              ir1[4] = (v845_data + (v817_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v851_data = ir1[5];
              ir1[5] = (v851_data + (v817_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v857_data = ir1[6];
              ir1[6] = (v857_data + (v817_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v863_data = ir1[7];
              ir1[7] = (v863_data + (v817_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v869_data = ir1[8];
              ir1[8] = (v869_data + (v817_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v872_data = glb_m1[(v23_lead + 240)];
              float v876_data = ir1[0];
              ir1[0] = (v876_data + (v872_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v882_data = ir1[1];
              ir1[1] = (v882_data + (v872_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v888_data = ir1[2];
              ir1[2] = (v888_data + (v872_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v894_data = ir1[3];
              ir1[3] = (v894_data + (v872_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v900_data = ir1[4];
              ir1[4] = (v900_data + (v872_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v906_data = ir1[5];
              ir1[5] = (v906_data + (v872_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v912_data = ir1[6];
              ir1[6] = (v912_data + (v872_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v918_data = ir1[7];
              ir1[7] = (v918_data + (v872_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v924_data = ir1[8];
              ir1[8] = (v924_data + (v872_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v927_data = glb_m1[(v23_lead + 256)];
              float v928_data = r0[1];
              float v931_data = ir1[0];
              ir1[0] = (v931_data + (v927_data * (sycl::select_from_group(item.get_sub_group(), v928_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v934_data = r0[3];
              float v937_data = ir1[1];
              ir1[1] = (v937_data + (v927_data * (sycl::select_from_group(item.get_sub_group(), v934_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v940_data = r0[5];
              float v943_data = ir1[2];
              ir1[2] = (v943_data + (v927_data * (sycl::select_from_group(item.get_sub_group(), v940_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v946_data = r0[7];
              float v949_data = ir1[3];
              ir1[3] = (v949_data + (v927_data * (sycl::select_from_group(item.get_sub_group(), v946_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v952_data = r0[9];
              float v955_data = ir1[4];
              ir1[4] = (v955_data + (v927_data * (sycl::select_from_group(item.get_sub_group(), v952_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v958_data = r0[11];
              float v961_data = ir1[5];
              ir1[5] = (v961_data + (v927_data * (sycl::select_from_group(item.get_sub_group(), v958_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v964_data = r0[13];
              float v967_data = ir1[6];
              ir1[6] = (v967_data + (v927_data * (sycl::select_from_group(item.get_sub_group(), v964_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v970_data = r0[15];
              float v973_data = ir1[7];
              ir1[7] = (v973_data + (v927_data * (sycl::select_from_group(item.get_sub_group(), v970_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v976_data = r0[17];
              float v979_data = ir1[8];
              ir1[8] = (v979_data + (v927_data * (sycl::select_from_group(item.get_sub_group(), v976_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              // r1 = ir1
              #pragma unroll
              for (int32_t v981_n0 = 0; v981_n0 < 1; ++v981_n0) {
                #pragma unroll
                for (int32_t v982_n1 = 0; v982_n1 < 9; ++v982_n1) {
                  int32_t v983_a = v981_n0 + v982_n1;
                  float v984_data = ir1[v983_a];
                  r1[v983_a] = v984_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v985_i0 = 0; v985_i0 < 1; ++v985_i0) {
                int32_t v990_lead = v23_lead + (v985_i0 * 16);
                #pragma unroll
                for (int32_t v986_i1 = 0; v986_i1 < 9; ++v986_i1) {
                  float v988_data = r1[(v985_i0 + v986_i1)];
                  glb_m0[(v990_lead + (v986_i1 * 16))] = v988_data;
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

