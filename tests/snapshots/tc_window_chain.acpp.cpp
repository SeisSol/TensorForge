// === base name ===
kernel_97c7842b181b04e7

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_97c7842b181b04e7 = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_97c7842b181b04e7(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_97c7842b181b04e7(const float * m0, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_97c7842b181b04e7(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_97c7842b181b04e7(const float * m0, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_97c7842b181b04e7(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_97c7842b181b04e7(stream, grid, block, m0, m1, m1_extraOffset, m2, m2_extraOffset, m3, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_97c7842b181b04e7(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 16×20(16×17) {0..16}×{1..18} none
        //   m1 20×9(17×9) {1..18}×{0..9} strided
        //   m2 16×9(16×9) {0..16}×{0..9} strided
        //   m3 16×20(16×15) {0..16}×{1..16} none
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   m2[i,j] = m3[i,k] × t0[k,j]@{1..16}×{0..9}
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"none","alias":"A1","bbox":[[0,1],[16,18]],"name":"m0","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B","bbox":[[1,0],[18,9]],"name":"m1","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[16,9]],"name":"m2","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"none","alias":"A2","bbox":[[0,1],[16,16]],"name":"m3","ordered":false,"parts":1,"shape":[16,20],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[16,18]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[18,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[16,16]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,20]},{"addressing":"pointer_based","bbox":[[1,0],[16,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[16,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          const float *const __restrict__ glb_m0 = &m0[0];
          const float *const __restrict__ glb_m3 = &m3[0];
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m1 = &m1[v9_batchId0 * 153 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v9_batchId0 * 144 + 0 + m2_extraOffset];
              float r0[18]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v22_lead = item.get_local_id(2) % 16;
              if (v22_lead >= 1) {
                int32_t v27_a = v22_lead - 1;
                #pragma unroll
                for (int32_t v24_i1 = 0; v24_i1 < 9; ++v24_i1) {
                  float v30_data = glb_m1[(v27_a + (v24_i1 * 17))];
                  r0[(v24_i1 * 2)] = v30_data;
                }
              }
              if (v22_lead < 2) {
                int32_t v37_a = (v22_lead + 16_i32) - 1;
                #pragma unroll
                for (int32_t v34_i1 = 0; v34_i1 < 9; ++v34_i1) {
                  float v40_data = glb_m1[(v37_a + (v34_i1 * 17))];
                  r0[(1 + (v34_i1 * 2))] = v40_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              float r1[9]{};
              // r1 = +(glb_m0 * r0) + None
              // [(0, 16), (0, 9)] [(1, 18)]
              float v47_data = glb_m0[v22_lead];
              float v48_data = r0[0];
              float v51_data = r1[0];
              r1[0] = (v51_data + (v47_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v54_data = r0[2];
              float v57_data = r1[1];
              r1[1] = (v57_data + (v47_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v60_data = r0[4];
              float v63_data = r1[2];
              r1[2] = (v63_data + (v47_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v66_data = r0[6];
              float v69_data = r1[3];
              r1[3] = (v69_data + (v47_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v72_data = r0[8];
              float v75_data = r1[4];
              r1[4] = (v75_data + (v47_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v78_data = r0[10];
              float v81_data = r1[5];
              r1[5] = (v81_data + (v47_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v84_data = r0[12];
              float v87_data = r1[6];
              r1[6] = (v87_data + (v47_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v90_data = r0[14];
              float v93_data = r1[7];
              r1[7] = (v93_data + (v47_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v96_data = r0[16];
              float v99_data = r1[8];
              r1[8] = (v99_data + (v47_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              int32_t v101_a = v22_lead + 16;
              float v102_data = glb_m0[v101_a];
              float v106_data = r1[0];
              r1[0] = (v106_data + (v102_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v112_data = r1[1];
              r1[1] = (v112_data + (v102_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v118_data = r1[2];
              r1[2] = (v118_data + (v102_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v124_data = r1[3];
              r1[3] = (v124_data + (v102_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v130_data = r1[4];
              r1[4] = (v130_data + (v102_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v136_data = r1[5];
              r1[5] = (v136_data + (v102_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v142_data = r1[6];
              r1[6] = (v142_data + (v102_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v148_data = r1[7];
              r1[7] = (v148_data + (v102_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v154_data = r1[8];
              r1[8] = (v154_data + (v102_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              int32_t v156_a = v22_lead + 32;
              float v157_data = glb_m0[v156_a];
              float v161_data = r1[0];
              r1[0] = (v161_data + (v157_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v167_data = r1[1];
              r1[1] = (v167_data + (v157_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v173_data = r1[2];
              r1[2] = (v173_data + (v157_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v179_data = r1[3];
              r1[3] = (v179_data + (v157_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v185_data = r1[4];
              r1[4] = (v185_data + (v157_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v191_data = r1[5];
              r1[5] = (v191_data + (v157_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v197_data = r1[6];
              r1[6] = (v197_data + (v157_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v203_data = r1[7];
              r1[7] = (v203_data + (v157_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v209_data = r1[8];
              r1[8] = (v209_data + (v157_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              int32_t v211_a = v22_lead + 48;
              float v212_data = glb_m0[v211_a];
              float v216_data = r1[0];
              r1[0] = (v216_data + (v212_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v222_data = r1[1];
              r1[1] = (v222_data + (v212_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v228_data = r1[2];
              r1[2] = (v228_data + (v212_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v234_data = r1[3];
              r1[3] = (v234_data + (v212_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v240_data = r1[4];
              r1[4] = (v240_data + (v212_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v246_data = r1[5];
              r1[5] = (v246_data + (v212_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v252_data = r1[6];
              r1[6] = (v252_data + (v212_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v258_data = r1[7];
              r1[7] = (v258_data + (v212_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v264_data = r1[8];
              r1[8] = (v264_data + (v212_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              int32_t v266_a = v22_lead + 64;
              float v267_data = glb_m0[v266_a];
              float v271_data = r1[0];
              r1[0] = (v271_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v277_data = r1[1];
              r1[1] = (v277_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v283_data = r1[2];
              r1[2] = (v283_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v289_data = r1[3];
              r1[3] = (v289_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v295_data = r1[4];
              r1[4] = (v295_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v301_data = r1[5];
              r1[5] = (v301_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v307_data = r1[6];
              r1[6] = (v307_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v313_data = r1[7];
              r1[7] = (v313_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v319_data = r1[8];
              r1[8] = (v319_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              int32_t v321_a = v22_lead + 80;
              float v322_data = glb_m0[v321_a];
              float v326_data = r1[0];
              r1[0] = (v326_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v332_data = r1[1];
              r1[1] = (v332_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v338_data = r1[2];
              r1[2] = (v338_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v344_data = r1[3];
              r1[3] = (v344_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v350_data = r1[4];
              r1[4] = (v350_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v356_data = r1[5];
              r1[5] = (v356_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v362_data = r1[6];
              r1[6] = (v362_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v368_data = r1[7];
              r1[7] = (v368_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v374_data = r1[8];
              r1[8] = (v374_data + (v322_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              int32_t v376_a = v22_lead + 96;
              float v377_data = glb_m0[v376_a];
              float v381_data = r1[0];
              r1[0] = (v381_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v387_data = r1[1];
              r1[1] = (v387_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v393_data = r1[2];
              r1[2] = (v393_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v399_data = r1[3];
              r1[3] = (v399_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v405_data = r1[4];
              r1[4] = (v405_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v411_data = r1[5];
              r1[5] = (v411_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v417_data = r1[6];
              r1[6] = (v417_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v423_data = r1[7];
              r1[7] = (v423_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v429_data = r1[8];
              r1[8] = (v429_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              int32_t v431_a = v22_lead + 112;
              float v432_data = glb_m0[v431_a];
              float v436_data = r1[0];
              r1[0] = (v436_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v442_data = r1[1];
              r1[1] = (v442_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v448_data = r1[2];
              r1[2] = (v448_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v454_data = r1[3];
              r1[3] = (v454_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v460_data = r1[4];
              r1[4] = (v460_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v466_data = r1[5];
              r1[5] = (v466_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v472_data = r1[6];
              r1[6] = (v472_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v478_data = r1[7];
              r1[7] = (v478_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v484_data = r1[8];
              r1[8] = (v484_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              int32_t v486_a = v22_lead + 128;
              float v487_data = glb_m0[v486_a];
              float v491_data = r1[0];
              r1[0] = (v491_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v497_data = r1[1];
              r1[1] = (v497_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v503_data = r1[2];
              r1[2] = (v503_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v509_data = r1[3];
              r1[3] = (v509_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v515_data = r1[4];
              r1[4] = (v515_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v521_data = r1[5];
              r1[5] = (v521_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v527_data = r1[6];
              r1[6] = (v527_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v533_data = r1[7];
              r1[7] = (v533_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v539_data = r1[8];
              r1[8] = (v539_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              int32_t v541_a = v22_lead + 144;
              float v542_data = glb_m0[v541_a];
              float v546_data = r1[0];
              r1[0] = (v546_data + (v542_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v552_data = r1[1];
              r1[1] = (v552_data + (v542_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v558_data = r1[2];
              r1[2] = (v558_data + (v542_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v564_data = r1[3];
              r1[3] = (v564_data + (v542_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v570_data = r1[4];
              r1[4] = (v570_data + (v542_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v576_data = r1[5];
              r1[5] = (v576_data + (v542_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v582_data = r1[6];
              r1[6] = (v582_data + (v542_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v588_data = r1[7];
              r1[7] = (v588_data + (v542_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v594_data = r1[8];
              r1[8] = (v594_data + (v542_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              int32_t v596_a = v22_lead + 160;
              float v597_data = glb_m0[v596_a];
              float v601_data = r1[0];
              r1[0] = (v601_data + (v597_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v607_data = r1[1];
              r1[1] = (v607_data + (v597_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v613_data = r1[2];
              r1[2] = (v613_data + (v597_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v619_data = r1[3];
              r1[3] = (v619_data + (v597_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v625_data = r1[4];
              r1[4] = (v625_data + (v597_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v631_data = r1[5];
              r1[5] = (v631_data + (v597_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v637_data = r1[6];
              r1[6] = (v637_data + (v597_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v643_data = r1[7];
              r1[7] = (v643_data + (v597_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v649_data = r1[8];
              r1[8] = (v649_data + (v597_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              int32_t v651_a = v22_lead + 176;
              float v652_data = glb_m0[v651_a];
              float v656_data = r1[0];
              r1[0] = (v656_data + (v652_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v662_data = r1[1];
              r1[1] = (v662_data + (v652_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v668_data = r1[2];
              r1[2] = (v668_data + (v652_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v674_data = r1[3];
              r1[3] = (v674_data + (v652_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v680_data = r1[4];
              r1[4] = (v680_data + (v652_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v686_data = r1[5];
              r1[5] = (v686_data + (v652_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v692_data = r1[6];
              r1[6] = (v692_data + (v652_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v698_data = r1[7];
              r1[7] = (v698_data + (v652_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v704_data = r1[8];
              r1[8] = (v704_data + (v652_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              int32_t v706_a = v22_lead + 192;
              float v707_data = glb_m0[v706_a];
              float v711_data = r1[0];
              r1[0] = (v711_data + (v707_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v717_data = r1[1];
              r1[1] = (v717_data + (v707_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v723_data = r1[2];
              r1[2] = (v723_data + (v707_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v729_data = r1[3];
              r1[3] = (v729_data + (v707_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v735_data = r1[4];
              r1[4] = (v735_data + (v707_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v741_data = r1[5];
              r1[5] = (v741_data + (v707_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v747_data = r1[6];
              r1[6] = (v747_data + (v707_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v753_data = r1[7];
              r1[7] = (v753_data + (v707_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v759_data = r1[8];
              r1[8] = (v759_data + (v707_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              int32_t v761_a = v22_lead + 208;
              float v762_data = glb_m0[v761_a];
              float v766_data = r1[0];
              r1[0] = (v766_data + (v762_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v772_data = r1[1];
              r1[1] = (v772_data + (v762_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v778_data = r1[2];
              r1[2] = (v778_data + (v762_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v784_data = r1[3];
              r1[3] = (v784_data + (v762_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v790_data = r1[4];
              r1[4] = (v790_data + (v762_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v796_data = r1[5];
              r1[5] = (v796_data + (v762_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v802_data = r1[6];
              r1[6] = (v802_data + (v762_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v808_data = r1[7];
              r1[7] = (v808_data + (v762_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v814_data = r1[8];
              r1[8] = (v814_data + (v762_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              int32_t v816_a = v22_lead + 224;
              float v817_data = glb_m0[v816_a];
              float v821_data = r1[0];
              r1[0] = (v821_data + (v817_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v827_data = r1[1];
              r1[1] = (v827_data + (v817_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v833_data = r1[2];
              r1[2] = (v833_data + (v817_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v839_data = r1[3];
              r1[3] = (v839_data + (v817_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v845_data = r1[4];
              r1[4] = (v845_data + (v817_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v851_data = r1[5];
              r1[5] = (v851_data + (v817_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v857_data = r1[6];
              r1[6] = (v857_data + (v817_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v863_data = r1[7];
              r1[7] = (v863_data + (v817_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v869_data = r1[8];
              r1[8] = (v869_data + (v817_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v872_data = glb_m0[(v22_lead + 240)];
              float v873_data = r0[1];
              float v876_data = r1[0];
              r1[0] = (v876_data + (v872_data * (sycl::select_from_group(item.get_sub_group(), v873_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v879_data = r0[3];
              float v882_data = r1[1];
              r1[1] = (v882_data + (v872_data * (sycl::select_from_group(item.get_sub_group(), v879_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v885_data = r0[5];
              float v888_data = r1[2];
              r1[2] = (v888_data + (v872_data * (sycl::select_from_group(item.get_sub_group(), v885_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v891_data = r0[7];
              float v894_data = r1[3];
              r1[3] = (v894_data + (v872_data * (sycl::select_from_group(item.get_sub_group(), v891_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v897_data = r0[9];
              float v900_data = r1[4];
              r1[4] = (v900_data + (v872_data * (sycl::select_from_group(item.get_sub_group(), v897_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v903_data = r0[11];
              float v906_data = r1[5];
              r1[5] = (v906_data + (v872_data * (sycl::select_from_group(item.get_sub_group(), v903_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v909_data = r0[13];
              float v912_data = r1[6];
              r1[6] = (v912_data + (v872_data * (sycl::select_from_group(item.get_sub_group(), v909_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v915_data = r0[15];
              float v918_data = r1[7];
              r1[7] = (v918_data + (v872_data * (sycl::select_from_group(item.get_sub_group(), v915_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v921_data = r0[17];
              float v924_data = r1[8];
              r1[8] = (v924_data + (v872_data * (sycl::select_from_group(item.get_sub_group(), v921_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v927_data = glb_m0[(v22_lead + 256)];
              float v931_data = r1[0];
              r1[0] = (v931_data + (v927_data * (sycl::select_from_group(item.get_sub_group(), v873_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v937_data = r1[1];
              r1[1] = (v937_data + (v927_data * (sycl::select_from_group(item.get_sub_group(), v879_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v943_data = r1[2];
              r1[2] = (v943_data + (v927_data * (sycl::select_from_group(item.get_sub_group(), v885_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v949_data = r1[3];
              r1[3] = (v949_data + (v927_data * (sycl::select_from_group(item.get_sub_group(), v891_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v955_data = r1[4];
              r1[4] = (v955_data + (v927_data * (sycl::select_from_group(item.get_sub_group(), v897_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v961_data = r1[5];
              r1[5] = (v961_data + (v927_data * (sycl::select_from_group(item.get_sub_group(), v903_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v967_data = r1[6];
              r1[6] = (v967_data + (v927_data * (sycl::select_from_group(item.get_sub_group(), v909_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v973_data = r1[7];
              r1[7] = (v973_data + (v927_data * (sycl::select_from_group(item.get_sub_group(), v915_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v979_data = r1[8];
              r1[8] = (v979_data + (v927_data * (sycl::select_from_group(item.get_sub_group(), v921_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float r2[9]{};
              // ir2 = +(glb_m3 * r1)
              // [(0, 16), (0, 9)] [(1, 16)]
              float ir2[9]{};
              float v986_data = glb_m3[v22_lead];
              float v987_data = r1[0];
              float v990_data = ir2[0];
              ir2[0] = (v990_data + (v986_data * (sycl::select_from_group(item.get_sub_group(), v987_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v993_data = r1[1];
              float v996_data = ir2[1];
              ir2[1] = (v996_data + (v986_data * (sycl::select_from_group(item.get_sub_group(), v993_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v999_data = r1[2];
              float v1002_data = ir2[2];
              ir2[2] = (v1002_data + (v986_data * (sycl::select_from_group(item.get_sub_group(), v999_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1005_data = r1[3];
              float v1008_data = ir2[3];
              ir2[3] = (v1008_data + (v986_data * (sycl::select_from_group(item.get_sub_group(), v1005_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1011_data = r1[4];
              float v1014_data = ir2[4];
              ir2[4] = (v1014_data + (v986_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1017_data = r1[5];
              float v1020_data = ir2[5];
              ir2[5] = (v1020_data + (v986_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1023_data = r1[6];
              float v1026_data = ir2[6];
              ir2[6] = (v1026_data + (v986_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1029_data = r1[7];
              float v1032_data = ir2[7];
              ir2[7] = (v1032_data + (v986_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1035_data = r1[8];
              float v1038_data = ir2[8];
              ir2[8] = (v1038_data + (v986_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1041_data = glb_m3[v101_a];
              float v1045_data = ir2[0];
              ir2[0] = (v1045_data + (v1041_data * (sycl::select_from_group(item.get_sub_group(), v987_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1051_data = ir2[1];
              ir2[1] = (v1051_data + (v1041_data * (sycl::select_from_group(item.get_sub_group(), v993_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1057_data = ir2[2];
              ir2[2] = (v1057_data + (v1041_data * (sycl::select_from_group(item.get_sub_group(), v999_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1063_data = ir2[3];
              ir2[3] = (v1063_data + (v1041_data * (sycl::select_from_group(item.get_sub_group(), v1005_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1069_data = ir2[4];
              ir2[4] = (v1069_data + (v1041_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1075_data = ir2[5];
              ir2[5] = (v1075_data + (v1041_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1081_data = ir2[6];
              ir2[6] = (v1081_data + (v1041_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1087_data = ir2[7];
              ir2[7] = (v1087_data + (v1041_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1093_data = ir2[8];
              ir2[8] = (v1093_data + (v1041_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1096_data = glb_m3[v156_a];
              float v1100_data = ir2[0];
              ir2[0] = (v1100_data + (v1096_data * (sycl::select_from_group(item.get_sub_group(), v987_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1106_data = ir2[1];
              ir2[1] = (v1106_data + (v1096_data * (sycl::select_from_group(item.get_sub_group(), v993_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1112_data = ir2[2];
              ir2[2] = (v1112_data + (v1096_data * (sycl::select_from_group(item.get_sub_group(), v999_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1118_data = ir2[3];
              ir2[3] = (v1118_data + (v1096_data * (sycl::select_from_group(item.get_sub_group(), v1005_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1124_data = ir2[4];
              ir2[4] = (v1124_data + (v1096_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1130_data = ir2[5];
              ir2[5] = (v1130_data + (v1096_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1136_data = ir2[6];
              ir2[6] = (v1136_data + (v1096_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1142_data = ir2[7];
              ir2[7] = (v1142_data + (v1096_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1148_data = ir2[8];
              ir2[8] = (v1148_data + (v1096_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1151_data = glb_m3[v211_a];
              float v1155_data = ir2[0];
              ir2[0] = (v1155_data + (v1151_data * (sycl::select_from_group(item.get_sub_group(), v987_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1161_data = ir2[1];
              ir2[1] = (v1161_data + (v1151_data * (sycl::select_from_group(item.get_sub_group(), v993_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1167_data = ir2[2];
              ir2[2] = (v1167_data + (v1151_data * (sycl::select_from_group(item.get_sub_group(), v999_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1173_data = ir2[3];
              ir2[3] = (v1173_data + (v1151_data * (sycl::select_from_group(item.get_sub_group(), v1005_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1179_data = ir2[4];
              ir2[4] = (v1179_data + (v1151_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1185_data = ir2[5];
              ir2[5] = (v1185_data + (v1151_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1191_data = ir2[6];
              ir2[6] = (v1191_data + (v1151_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1197_data = ir2[7];
              ir2[7] = (v1197_data + (v1151_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1203_data = ir2[8];
              ir2[8] = (v1203_data + (v1151_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1206_data = glb_m3[v266_a];
              float v1210_data = ir2[0];
              ir2[0] = (v1210_data + (v1206_data * (sycl::select_from_group(item.get_sub_group(), v987_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1216_data = ir2[1];
              ir2[1] = (v1216_data + (v1206_data * (sycl::select_from_group(item.get_sub_group(), v993_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1222_data = ir2[2];
              ir2[2] = (v1222_data + (v1206_data * (sycl::select_from_group(item.get_sub_group(), v999_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1228_data = ir2[3];
              ir2[3] = (v1228_data + (v1206_data * (sycl::select_from_group(item.get_sub_group(), v1005_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1234_data = ir2[4];
              ir2[4] = (v1234_data + (v1206_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1240_data = ir2[5];
              ir2[5] = (v1240_data + (v1206_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1246_data = ir2[6];
              ir2[6] = (v1246_data + (v1206_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1252_data = ir2[7];
              ir2[7] = (v1252_data + (v1206_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1258_data = ir2[8];
              ir2[8] = (v1258_data + (v1206_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1261_data = glb_m3[v321_a];
              float v1265_data = ir2[0];
              ir2[0] = (v1265_data + (v1261_data * (sycl::select_from_group(item.get_sub_group(), v987_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1271_data = ir2[1];
              ir2[1] = (v1271_data + (v1261_data * (sycl::select_from_group(item.get_sub_group(), v993_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1277_data = ir2[2];
              ir2[2] = (v1277_data + (v1261_data * (sycl::select_from_group(item.get_sub_group(), v999_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1283_data = ir2[3];
              ir2[3] = (v1283_data + (v1261_data * (sycl::select_from_group(item.get_sub_group(), v1005_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1289_data = ir2[4];
              ir2[4] = (v1289_data + (v1261_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1295_data = ir2[5];
              ir2[5] = (v1295_data + (v1261_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1301_data = ir2[6];
              ir2[6] = (v1301_data + (v1261_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1307_data = ir2[7];
              ir2[7] = (v1307_data + (v1261_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1313_data = ir2[8];
              ir2[8] = (v1313_data + (v1261_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1316_data = glb_m3[v376_a];
              float v1320_data = ir2[0];
              ir2[0] = (v1320_data + (v1316_data * (sycl::select_from_group(item.get_sub_group(), v987_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1326_data = ir2[1];
              ir2[1] = (v1326_data + (v1316_data * (sycl::select_from_group(item.get_sub_group(), v993_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1332_data = ir2[2];
              ir2[2] = (v1332_data + (v1316_data * (sycl::select_from_group(item.get_sub_group(), v999_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1338_data = ir2[3];
              ir2[3] = (v1338_data + (v1316_data * (sycl::select_from_group(item.get_sub_group(), v1005_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1344_data = ir2[4];
              ir2[4] = (v1344_data + (v1316_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1350_data = ir2[5];
              ir2[5] = (v1350_data + (v1316_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1356_data = ir2[6];
              ir2[6] = (v1356_data + (v1316_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1362_data = ir2[7];
              ir2[7] = (v1362_data + (v1316_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1368_data = ir2[8];
              ir2[8] = (v1368_data + (v1316_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1371_data = glb_m3[v431_a];
              float v1375_data = ir2[0];
              ir2[0] = (v1375_data + (v1371_data * (sycl::select_from_group(item.get_sub_group(), v987_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1381_data = ir2[1];
              ir2[1] = (v1381_data + (v1371_data * (sycl::select_from_group(item.get_sub_group(), v993_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1387_data = ir2[2];
              ir2[2] = (v1387_data + (v1371_data * (sycl::select_from_group(item.get_sub_group(), v999_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1393_data = ir2[3];
              ir2[3] = (v1393_data + (v1371_data * (sycl::select_from_group(item.get_sub_group(), v1005_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1399_data = ir2[4];
              ir2[4] = (v1399_data + (v1371_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1405_data = ir2[5];
              ir2[5] = (v1405_data + (v1371_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1411_data = ir2[6];
              ir2[6] = (v1411_data + (v1371_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1417_data = ir2[7];
              ir2[7] = (v1417_data + (v1371_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1423_data = ir2[8];
              ir2[8] = (v1423_data + (v1371_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1426_data = glb_m3[v486_a];
              float v1430_data = ir2[0];
              ir2[0] = (v1430_data + (v1426_data * (sycl::select_from_group(item.get_sub_group(), v987_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1436_data = ir2[1];
              ir2[1] = (v1436_data + (v1426_data * (sycl::select_from_group(item.get_sub_group(), v993_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1442_data = ir2[2];
              ir2[2] = (v1442_data + (v1426_data * (sycl::select_from_group(item.get_sub_group(), v999_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1448_data = ir2[3];
              ir2[3] = (v1448_data + (v1426_data * (sycl::select_from_group(item.get_sub_group(), v1005_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1454_data = ir2[4];
              ir2[4] = (v1454_data + (v1426_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1460_data = ir2[5];
              ir2[5] = (v1460_data + (v1426_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1466_data = ir2[6];
              ir2[6] = (v1466_data + (v1426_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1472_data = ir2[7];
              ir2[7] = (v1472_data + (v1426_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1478_data = ir2[8];
              ir2[8] = (v1478_data + (v1426_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1481_data = glb_m3[v541_a];
              float v1485_data = ir2[0];
              ir2[0] = (v1485_data + (v1481_data * (sycl::select_from_group(item.get_sub_group(), v987_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1491_data = ir2[1];
              ir2[1] = (v1491_data + (v1481_data * (sycl::select_from_group(item.get_sub_group(), v993_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1497_data = ir2[2];
              ir2[2] = (v1497_data + (v1481_data * (sycl::select_from_group(item.get_sub_group(), v999_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1503_data = ir2[3];
              ir2[3] = (v1503_data + (v1481_data * (sycl::select_from_group(item.get_sub_group(), v1005_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1509_data = ir2[4];
              ir2[4] = (v1509_data + (v1481_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1515_data = ir2[5];
              ir2[5] = (v1515_data + (v1481_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1521_data = ir2[6];
              ir2[6] = (v1521_data + (v1481_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1527_data = ir2[7];
              ir2[7] = (v1527_data + (v1481_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1533_data = ir2[8];
              ir2[8] = (v1533_data + (v1481_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1536_data = glb_m3[v596_a];
              float v1540_data = ir2[0];
              ir2[0] = (v1540_data + (v1536_data * (sycl::select_from_group(item.get_sub_group(), v987_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1546_data = ir2[1];
              ir2[1] = (v1546_data + (v1536_data * (sycl::select_from_group(item.get_sub_group(), v993_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1552_data = ir2[2];
              ir2[2] = (v1552_data + (v1536_data * (sycl::select_from_group(item.get_sub_group(), v999_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1558_data = ir2[3];
              ir2[3] = (v1558_data + (v1536_data * (sycl::select_from_group(item.get_sub_group(), v1005_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1564_data = ir2[4];
              ir2[4] = (v1564_data + (v1536_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1570_data = ir2[5];
              ir2[5] = (v1570_data + (v1536_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1576_data = ir2[6];
              ir2[6] = (v1576_data + (v1536_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1582_data = ir2[7];
              ir2[7] = (v1582_data + (v1536_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1588_data = ir2[8];
              ir2[8] = (v1588_data + (v1536_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1591_data = glb_m3[v651_a];
              float v1595_data = ir2[0];
              ir2[0] = (v1595_data + (v1591_data * (sycl::select_from_group(item.get_sub_group(), v987_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1601_data = ir2[1];
              ir2[1] = (v1601_data + (v1591_data * (sycl::select_from_group(item.get_sub_group(), v993_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1607_data = ir2[2];
              ir2[2] = (v1607_data + (v1591_data * (sycl::select_from_group(item.get_sub_group(), v999_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1613_data = ir2[3];
              ir2[3] = (v1613_data + (v1591_data * (sycl::select_from_group(item.get_sub_group(), v1005_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1619_data = ir2[4];
              ir2[4] = (v1619_data + (v1591_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1625_data = ir2[5];
              ir2[5] = (v1625_data + (v1591_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1631_data = ir2[6];
              ir2[6] = (v1631_data + (v1591_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1637_data = ir2[7];
              ir2[7] = (v1637_data + (v1591_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1643_data = ir2[8];
              ir2[8] = (v1643_data + (v1591_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1646_data = glb_m3[v706_a];
              float v1650_data = ir2[0];
              ir2[0] = (v1650_data + (v1646_data * (sycl::select_from_group(item.get_sub_group(), v987_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1656_data = ir2[1];
              ir2[1] = (v1656_data + (v1646_data * (sycl::select_from_group(item.get_sub_group(), v993_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1662_data = ir2[2];
              ir2[2] = (v1662_data + (v1646_data * (sycl::select_from_group(item.get_sub_group(), v999_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1668_data = ir2[3];
              ir2[3] = (v1668_data + (v1646_data * (sycl::select_from_group(item.get_sub_group(), v1005_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1674_data = ir2[4];
              ir2[4] = (v1674_data + (v1646_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1680_data = ir2[5];
              ir2[5] = (v1680_data + (v1646_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1686_data = ir2[6];
              ir2[6] = (v1686_data + (v1646_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1692_data = ir2[7];
              ir2[7] = (v1692_data + (v1646_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1698_data = ir2[8];
              ir2[8] = (v1698_data + (v1646_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1701_data = glb_m3[v761_a];
              float v1705_data = ir2[0];
              ir2[0] = (v1705_data + (v1701_data * (sycl::select_from_group(item.get_sub_group(), v987_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1711_data = ir2[1];
              ir2[1] = (v1711_data + (v1701_data * (sycl::select_from_group(item.get_sub_group(), v993_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1717_data = ir2[2];
              ir2[2] = (v1717_data + (v1701_data * (sycl::select_from_group(item.get_sub_group(), v999_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1723_data = ir2[3];
              ir2[3] = (v1723_data + (v1701_data * (sycl::select_from_group(item.get_sub_group(), v1005_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1729_data = ir2[4];
              ir2[4] = (v1729_data + (v1701_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1735_data = ir2[5];
              ir2[5] = (v1735_data + (v1701_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1741_data = ir2[6];
              ir2[6] = (v1741_data + (v1701_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1747_data = ir2[7];
              ir2[7] = (v1747_data + (v1701_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1753_data = ir2[8];
              ir2[8] = (v1753_data + (v1701_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1756_data = glb_m3[v816_a];
              float v1760_data = ir2[0];
              ir2[0] = (v1760_data + (v1756_data * (sycl::select_from_group(item.get_sub_group(), v987_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1766_data = ir2[1];
              ir2[1] = (v1766_data + (v1756_data * (sycl::select_from_group(item.get_sub_group(), v993_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1772_data = ir2[2];
              ir2[2] = (v1772_data + (v1756_data * (sycl::select_from_group(item.get_sub_group(), v999_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1778_data = ir2[3];
              ir2[3] = (v1778_data + (v1756_data * (sycl::select_from_group(item.get_sub_group(), v1005_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1784_data = ir2[4];
              ir2[4] = (v1784_data + (v1756_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1790_data = ir2[5];
              ir2[5] = (v1790_data + (v1756_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1796_data = ir2[6];
              ir2[6] = (v1796_data + (v1756_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1802_data = ir2[7];
              ir2[7] = (v1802_data + (v1756_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1808_data = ir2[8];
              ir2[8] = (v1808_data + (v1756_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              // r2 = ir2
              #pragma unroll
              for (int32_t v1810_n0 = 0; v1810_n0 < 1; ++v1810_n0) {
                #pragma unroll
                for (int32_t v1811_n1 = 0; v1811_n1 < 9; ++v1811_n1) {
                  int32_t v1812_a = v1810_n0 + v1811_n1;
                  float v1813_data = ir2[v1812_a];
                  r2[v1812_a] = v1813_data;
                }
              }
              // glb_m2 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v1814_i0 = 0; v1814_i0 < 1; ++v1814_i0) {
                int32_t v1819_lead = v22_lead + (v1814_i0 * 16);
                #pragma unroll
                for (int32_t v1815_i1 = 0; v1815_i1 < 9; ++v1815_i1) {
                  float v1817_data = r2[(v1814_i0 + v1815_i1)];
                  glb_m2[(v1819_lead + (v1815_i1 * 16))] = v1817_data;
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

