// === base name ===
kernel_49e25cda84b7ee98

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_49e25cda84b7ee98 = {{16, 16, 1}, 16, 10, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_49e25cda84b7ee98(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_49e25cda84b7ee98(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_49e25cda84b7ee98(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_49e25cda84b7ee98(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_49e25cda84b7ee98(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_49e25cda84b7ee98(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, m3, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_49e25cda84b7ee98(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (10 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 10×9(10×9) {0..10}×{0..9} strided
        //   m1 16×20(10×17) {0..10}×{1..18} none
        //   m2 20×9(17×9) {1..18}×{0..9} strided
        //   m3 16×20(10×18) {0..10}×{1..19} none
        //   m4 20×9(18×9) {1..19}×{0..9} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        //   m0[i,j] += m3[i,k] × m4[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":10,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[10,9]],"name":"m0","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"none","alias":"A1","bbox":[[0,1],[10,18]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[1,0],[18,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"none","alias":"A2","bbox":[[0,1],[10,19]],"name":"m3","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[1,0],[19,9]],"name":"m4","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[10,18]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[18,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[10,19]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[19,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          const float *const __restrict__ glb_m1 = &m1[0];
          const float *const __restrict__ glb_m3 = &m3[0];
          for (size_t v11_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v11_batchId0 < numElements0; v11_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v12_ahead1 = v11_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v14_batchId1 = (v12_ahead1 < numElements0) ? v12_ahead1 : v11_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v11_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v11_batchId0 * 90 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v11_batchId0 * 153 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v11_batchId0 * 162 + 0 + m4_extraOffset];
              float r0[18]{};
              // r0 = load{g>r}(glb_m2);
              int32_t v25_lead = item.get_local_id(2) % 16;
              bool v26_g = v25_lead >= 1;
              if (v26_g) {
                int32_t v30_a = v25_lead - 1;
                #pragma unroll
                for (int32_t v27_i1 = 0; v27_i1 < 9; ++v27_i1) {
                  float v33_data = glb_m2[(v30_a + (v27_i1 * 17))];
                  r0[(v27_i1 * 2)] = v33_data;
                }
              }
              if (v25_lead < 2) {
                int32_t v40_a = (v25_lead + 16_i32) - 1;
                #pragma unroll
                for (int32_t v37_i1 = 0; v37_i1 < 9; ++v37_i1) {
                  float v43_data = glb_m2[(v40_a + (v37_i1 * 17))];
                  r0[(1 + (v37_i1 * 2))] = v43_data;
                }
              }
              float r2[18]{};
              // r2 = load{g>r}(glb_m4);
              if (v26_g) {
                int32_t v50_a = v25_lead - 1;
                #pragma unroll
                for (int32_t v47_i1 = 0; v47_i1 < 9; ++v47_i1) {
                  float v53_data = glb_m4[(v50_a + (v47_i1 * 18))];
                  r2[(v47_i1 * 2)] = v53_data;
                }
              }
              if (v25_lead < 3) {
                int32_t v60_a = (v25_lead + 16_i32) - 1;
                #pragma unroll
                for (int32_t v57_i1 = 0; v57_i1 < 9; ++v57_i1) {
                  float v63_data = glb_m4[(v60_a + (v57_i1 * 18))];
                  r2[(1 + (v57_i1 * 2))] = v63_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m2););
              float r1[9]{};
              // ir1 = +(glb_m1 * r0)
              // [(0, 10), (0, 9)] [(1, 18)]
              float ir1[9]{};
              bool v71_g = v25_lead < 10;
              float v72_data_pre = glb_m1[v71_g ? (v25_lead) : (0)];
              float v72_data = v71_g ? (v72_data_pre) : (0.0f);
              float v73_data = r0[0];
              float v76_data = ir1[0];
              ir1[0] = (v76_data + (v72_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v79_data = r0[2];
              float v82_data = ir1[1];
              ir1[1] = (v82_data + (v72_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v85_data = r0[4];
              float v88_data = ir1[2];
              ir1[2] = (v88_data + (v72_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v91_data = r0[6];
              float v94_data = ir1[3];
              ir1[3] = (v94_data + (v72_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v97_data = r0[8];
              float v100_data = ir1[4];
              ir1[4] = (v100_data + (v72_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v103_data = r0[10];
              float v106_data = ir1[5];
              ir1[5] = (v106_data + (v72_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v109_data = r0[12];
              float v112_data = ir1[6];
              ir1[6] = (v112_data + (v72_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v115_data = r0[14];
              float v118_data = ir1[7];
              ir1[7] = (v118_data + (v72_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v121_data = r0[16];
              float v124_data = ir1[8];
              ir1[8] = (v124_data + (v72_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              int32_t v126_a = v25_lead + 10;
              float v127_data_pre = glb_m1[v71_g ? (v126_a) : (0)];
              float v127_data = v71_g ? (v127_data_pre) : (0.0f);
              float v131_data = ir1[0];
              ir1[0] = (v131_data + (v127_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v137_data = ir1[1];
              ir1[1] = (v137_data + (v127_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v143_data = ir1[2];
              ir1[2] = (v143_data + (v127_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v149_data = ir1[3];
              ir1[3] = (v149_data + (v127_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v155_data = ir1[4];
              ir1[4] = (v155_data + (v127_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v161_data = ir1[5];
              ir1[5] = (v161_data + (v127_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v167_data = ir1[6];
              ir1[6] = (v167_data + (v127_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v173_data = ir1[7];
              ir1[7] = (v173_data + (v127_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v179_data = ir1[8];
              ir1[8] = (v179_data + (v127_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              int32_t v181_a = v25_lead + 20;
              float v182_data_pre = glb_m1[v71_g ? (v181_a) : (0)];
              float v182_data = v71_g ? (v182_data_pre) : (0.0f);
              float v186_data = ir1[0];
              ir1[0] = (v186_data + (v182_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v192_data = ir1[1];
              ir1[1] = (v192_data + (v182_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v198_data = ir1[2];
              ir1[2] = (v198_data + (v182_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v204_data = ir1[3];
              ir1[3] = (v204_data + (v182_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v210_data = ir1[4];
              ir1[4] = (v210_data + (v182_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v216_data = ir1[5];
              ir1[5] = (v216_data + (v182_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v222_data = ir1[6];
              ir1[6] = (v222_data + (v182_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v228_data = ir1[7];
              ir1[7] = (v228_data + (v182_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v234_data = ir1[8];
              ir1[8] = (v234_data + (v182_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              int32_t v236_a = v25_lead + 30;
              float v237_data_pre = glb_m1[v71_g ? (v236_a) : (0)];
              float v237_data = v71_g ? (v237_data_pre) : (0.0f);
              float v241_data = ir1[0];
              ir1[0] = (v241_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v247_data = ir1[1];
              ir1[1] = (v247_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v253_data = ir1[2];
              ir1[2] = (v253_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v259_data = ir1[3];
              ir1[3] = (v259_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v265_data = ir1[4];
              ir1[4] = (v265_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v271_data = ir1[5];
              ir1[5] = (v271_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v277_data = ir1[6];
              ir1[6] = (v277_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v283_data = ir1[7];
              ir1[7] = (v283_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v289_data = ir1[8];
              ir1[8] = (v289_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              int32_t v291_a = v25_lead + 40;
              float v292_data_pre = glb_m1[v71_g ? (v291_a) : (0)];
              float v292_data = v71_g ? (v292_data_pre) : (0.0f);
              float v296_data = ir1[0];
              ir1[0] = (v296_data + (v292_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v302_data = ir1[1];
              ir1[1] = (v302_data + (v292_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v308_data = ir1[2];
              ir1[2] = (v308_data + (v292_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v314_data = ir1[3];
              ir1[3] = (v314_data + (v292_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v320_data = ir1[4];
              ir1[4] = (v320_data + (v292_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v326_data = ir1[5];
              ir1[5] = (v326_data + (v292_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v332_data = ir1[6];
              ir1[6] = (v332_data + (v292_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v338_data = ir1[7];
              ir1[7] = (v338_data + (v292_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v344_data = ir1[8];
              ir1[8] = (v344_data + (v292_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              int32_t v346_a = v25_lead + 50;
              float v347_data_pre = glb_m1[v71_g ? (v346_a) : (0)];
              float v347_data = v71_g ? (v347_data_pre) : (0.0f);
              float v351_data = ir1[0];
              ir1[0] = (v351_data + (v347_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v357_data = ir1[1];
              ir1[1] = (v357_data + (v347_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v363_data = ir1[2];
              ir1[2] = (v363_data + (v347_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v369_data = ir1[3];
              ir1[3] = (v369_data + (v347_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v375_data = ir1[4];
              ir1[4] = (v375_data + (v347_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v381_data = ir1[5];
              ir1[5] = (v381_data + (v347_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v387_data = ir1[6];
              ir1[6] = (v387_data + (v347_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v393_data = ir1[7];
              ir1[7] = (v393_data + (v347_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v399_data = ir1[8];
              ir1[8] = (v399_data + (v347_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              int32_t v401_a = v25_lead + 60;
              float v402_data_pre = glb_m1[v71_g ? (v401_a) : (0)];
              float v402_data = v71_g ? (v402_data_pre) : (0.0f);
              float v406_data = ir1[0];
              ir1[0] = (v406_data + (v402_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v412_data = ir1[1];
              ir1[1] = (v412_data + (v402_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v418_data = ir1[2];
              ir1[2] = (v418_data + (v402_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v424_data = ir1[3];
              ir1[3] = (v424_data + (v402_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v430_data = ir1[4];
              ir1[4] = (v430_data + (v402_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v436_data = ir1[5];
              ir1[5] = (v436_data + (v402_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v442_data = ir1[6];
              ir1[6] = (v442_data + (v402_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v448_data = ir1[7];
              ir1[7] = (v448_data + (v402_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v454_data = ir1[8];
              ir1[8] = (v454_data + (v402_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              int32_t v456_a = v25_lead + 70;
              float v457_data_pre = glb_m1[v71_g ? (v456_a) : (0)];
              float v457_data = v71_g ? (v457_data_pre) : (0.0f);
              float v461_data = ir1[0];
              ir1[0] = (v461_data + (v457_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v467_data = ir1[1];
              ir1[1] = (v467_data + (v457_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v473_data = ir1[2];
              ir1[2] = (v473_data + (v457_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v479_data = ir1[3];
              ir1[3] = (v479_data + (v457_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v485_data = ir1[4];
              ir1[4] = (v485_data + (v457_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v491_data = ir1[5];
              ir1[5] = (v491_data + (v457_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v497_data = ir1[6];
              ir1[6] = (v497_data + (v457_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v503_data = ir1[7];
              ir1[7] = (v503_data + (v457_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v509_data = ir1[8];
              ir1[8] = (v509_data + (v457_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              int32_t v511_a = v25_lead + 80;
              float v512_data_pre = glb_m1[v71_g ? (v511_a) : (0)];
              float v512_data = v71_g ? (v512_data_pre) : (0.0f);
              float v516_data = ir1[0];
              ir1[0] = (v516_data + (v512_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v522_data = ir1[1];
              ir1[1] = (v522_data + (v512_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v528_data = ir1[2];
              ir1[2] = (v528_data + (v512_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v534_data = ir1[3];
              ir1[3] = (v534_data + (v512_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v540_data = ir1[4];
              ir1[4] = (v540_data + (v512_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v546_data = ir1[5];
              ir1[5] = (v546_data + (v512_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v552_data = ir1[6];
              ir1[6] = (v552_data + (v512_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v558_data = ir1[7];
              ir1[7] = (v558_data + (v512_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v564_data = ir1[8];
              ir1[8] = (v564_data + (v512_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              int32_t v566_a = v25_lead + 90;
              float v567_data_pre = glb_m1[v71_g ? (v566_a) : (0)];
              float v567_data = v71_g ? (v567_data_pre) : (0.0f);
              float v571_data = ir1[0];
              ir1[0] = (v571_data + (v567_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v577_data = ir1[1];
              ir1[1] = (v577_data + (v567_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v583_data = ir1[2];
              ir1[2] = (v583_data + (v567_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v589_data = ir1[3];
              ir1[3] = (v589_data + (v567_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v595_data = ir1[4];
              ir1[4] = (v595_data + (v567_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v601_data = ir1[5];
              ir1[5] = (v601_data + (v567_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v607_data = ir1[6];
              ir1[6] = (v607_data + (v567_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v613_data = ir1[7];
              ir1[7] = (v613_data + (v567_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v619_data = ir1[8];
              ir1[8] = (v619_data + (v567_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              int32_t v621_a = v25_lead + 100;
              float v622_data_pre = glb_m1[v71_g ? (v621_a) : (0)];
              float v622_data = v71_g ? (v622_data_pre) : (0.0f);
              float v626_data = ir1[0];
              ir1[0] = (v626_data + (v622_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v632_data = ir1[1];
              ir1[1] = (v632_data + (v622_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v638_data = ir1[2];
              ir1[2] = (v638_data + (v622_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v644_data = ir1[3];
              ir1[3] = (v644_data + (v622_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v650_data = ir1[4];
              ir1[4] = (v650_data + (v622_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v656_data = ir1[5];
              ir1[5] = (v656_data + (v622_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v662_data = ir1[6];
              ir1[6] = (v662_data + (v622_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v668_data = ir1[7];
              ir1[7] = (v668_data + (v622_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v674_data = ir1[8];
              ir1[8] = (v674_data + (v622_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              int32_t v676_a = v25_lead + 110;
              float v677_data_pre = glb_m1[v71_g ? (v676_a) : (0)];
              float v677_data = v71_g ? (v677_data_pre) : (0.0f);
              float v681_data = ir1[0];
              ir1[0] = (v681_data + (v677_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v687_data = ir1[1];
              ir1[1] = (v687_data + (v677_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v693_data = ir1[2];
              ir1[2] = (v693_data + (v677_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v699_data = ir1[3];
              ir1[3] = (v699_data + (v677_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v705_data = ir1[4];
              ir1[4] = (v705_data + (v677_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v711_data = ir1[5];
              ir1[5] = (v711_data + (v677_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v717_data = ir1[6];
              ir1[6] = (v717_data + (v677_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v723_data = ir1[7];
              ir1[7] = (v723_data + (v677_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v729_data = ir1[8];
              ir1[8] = (v729_data + (v677_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              int32_t v731_a = v25_lead + 120;
              float v732_data_pre = glb_m1[v71_g ? (v731_a) : (0)];
              float v732_data = v71_g ? (v732_data_pre) : (0.0f);
              float v736_data = ir1[0];
              ir1[0] = (v736_data + (v732_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v742_data = ir1[1];
              ir1[1] = (v742_data + (v732_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v748_data = ir1[2];
              ir1[2] = (v748_data + (v732_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v754_data = ir1[3];
              ir1[3] = (v754_data + (v732_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v760_data = ir1[4];
              ir1[4] = (v760_data + (v732_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v766_data = ir1[5];
              ir1[5] = (v766_data + (v732_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v772_data = ir1[6];
              ir1[6] = (v772_data + (v732_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v778_data = ir1[7];
              ir1[7] = (v778_data + (v732_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v784_data = ir1[8];
              ir1[8] = (v784_data + (v732_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              int32_t v786_a = v25_lead + 130;
              float v787_data_pre = glb_m1[v71_g ? (v786_a) : (0)];
              float v787_data = v71_g ? (v787_data_pre) : (0.0f);
              float v791_data = ir1[0];
              ir1[0] = (v791_data + (v787_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v797_data = ir1[1];
              ir1[1] = (v797_data + (v787_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v803_data = ir1[2];
              ir1[2] = (v803_data + (v787_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v809_data = ir1[3];
              ir1[3] = (v809_data + (v787_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v815_data = ir1[4];
              ir1[4] = (v815_data + (v787_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v821_data = ir1[5];
              ir1[5] = (v821_data + (v787_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v827_data = ir1[6];
              ir1[6] = (v827_data + (v787_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v833_data = ir1[7];
              ir1[7] = (v833_data + (v787_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v839_data = ir1[8];
              ir1[8] = (v839_data + (v787_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              int32_t v841_a = v25_lead + 140;
              float v842_data_pre = glb_m1[v71_g ? (v841_a) : (0)];
              float v842_data = v71_g ? (v842_data_pre) : (0.0f);
              float v846_data = ir1[0];
              ir1[0] = (v846_data + (v842_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v852_data = ir1[1];
              ir1[1] = (v852_data + (v842_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v858_data = ir1[2];
              ir1[2] = (v858_data + (v842_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v864_data = ir1[3];
              ir1[3] = (v864_data + (v842_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v870_data = ir1[4];
              ir1[4] = (v870_data + (v842_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v876_data = ir1[5];
              ir1[5] = (v876_data + (v842_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v882_data = ir1[6];
              ir1[6] = (v882_data + (v842_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v888_data = ir1[7];
              ir1[7] = (v888_data + (v842_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v894_data = ir1[8];
              ir1[8] = (v894_data + (v842_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              int32_t v896_a = v25_lead + 150;
              float v897_data_pre = glb_m1[v71_g ? (v896_a) : (0)];
              float v897_data = v71_g ? (v897_data_pre) : (0.0f);
              float v898_data = r0[1];
              float v901_data = ir1[0];
              ir1[0] = (v901_data + (v897_data * (sycl::select_from_group(item.get_sub_group(), v898_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v904_data = r0[3];
              float v907_data = ir1[1];
              ir1[1] = (v907_data + (v897_data * (sycl::select_from_group(item.get_sub_group(), v904_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v910_data = r0[5];
              float v913_data = ir1[2];
              ir1[2] = (v913_data + (v897_data * (sycl::select_from_group(item.get_sub_group(), v910_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v916_data = r0[7];
              float v919_data = ir1[3];
              ir1[3] = (v919_data + (v897_data * (sycl::select_from_group(item.get_sub_group(), v916_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v922_data = r0[9];
              float v925_data = ir1[4];
              ir1[4] = (v925_data + (v897_data * (sycl::select_from_group(item.get_sub_group(), v922_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v928_data = r0[11];
              float v931_data = ir1[5];
              ir1[5] = (v931_data + (v897_data * (sycl::select_from_group(item.get_sub_group(), v928_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v934_data = r0[13];
              float v937_data = ir1[6];
              ir1[6] = (v937_data + (v897_data * (sycl::select_from_group(item.get_sub_group(), v934_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v940_data = r0[15];
              float v943_data = ir1[7];
              ir1[7] = (v943_data + (v897_data * (sycl::select_from_group(item.get_sub_group(), v940_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v946_data = r0[17];
              float v949_data = ir1[8];
              ir1[8] = (v949_data + (v897_data * (sycl::select_from_group(item.get_sub_group(), v946_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              int32_t v951_a = v25_lead + 160;
              float v952_data_pre = glb_m1[v71_g ? (v951_a) : (0)];
              float v952_data = v71_g ? (v952_data_pre) : (0.0f);
              float v956_data = ir1[0];
              ir1[0] = (v956_data + (v952_data * (sycl::select_from_group(item.get_sub_group(), v898_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v962_data = ir1[1];
              ir1[1] = (v962_data + (v952_data * (sycl::select_from_group(item.get_sub_group(), v904_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v968_data = ir1[2];
              ir1[2] = (v968_data + (v952_data * (sycl::select_from_group(item.get_sub_group(), v910_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v974_data = ir1[3];
              ir1[3] = (v974_data + (v952_data * (sycl::select_from_group(item.get_sub_group(), v916_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v980_data = ir1[4];
              ir1[4] = (v980_data + (v952_data * (sycl::select_from_group(item.get_sub_group(), v922_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v986_data = ir1[5];
              ir1[5] = (v986_data + (v952_data * (sycl::select_from_group(item.get_sub_group(), v928_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v992_data = ir1[6];
              ir1[6] = (v992_data + (v952_data * (sycl::select_from_group(item.get_sub_group(), v934_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v998_data = ir1[7];
              ir1[7] = (v998_data + (v952_data * (sycl::select_from_group(item.get_sub_group(), v940_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1004_data = ir1[8];
              ir1[8] = (v1004_data + (v952_data * (sycl::select_from_group(item.get_sub_group(), v946_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              // r1 = ir1
              if (v71_g) {
                #pragma unroll
                for (int32_t v1007_n1 = 0; v1007_n1 < 9; ++v1007_n1) {
                  float v1009_data = ir1[v1007_n1];
                  r1[v1007_n1] = v1009_data;
                }
              }
              // wait(r2 = load{g>r}(glb_m4););
              float r3[9]{};
              // ir3 = +(glb_m3 * r2)
              // [(0, 10), (0, 9)] [(1, 19)]
              float ir3[9]{};
              float v1016_data_pre = glb_m3[v71_g ? (v25_lead) : (0)];
              float v1016_data = v71_g ? (v1016_data_pre) : (0.0f);
              float v1017_data = r2[0];
              float v1020_data = ir3[0];
              ir3[0] = (v1020_data + (v1016_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1023_data = r2[2];
              float v1026_data = ir3[1];
              ir3[1] = (v1026_data + (v1016_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1029_data = r2[4];
              float v1032_data = ir3[2];
              ir3[2] = (v1032_data + (v1016_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1035_data = r2[6];
              float v1038_data = ir3[3];
              ir3[3] = (v1038_data + (v1016_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1041_data = r2[8];
              float v1044_data = ir3[4];
              ir3[4] = (v1044_data + (v1016_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1047_data = r2[10];
              float v1050_data = ir3[5];
              ir3[5] = (v1050_data + (v1016_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1053_data = r2[12];
              float v1056_data = ir3[6];
              ir3[6] = (v1056_data + (v1016_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1059_data = r2[14];
              float v1062_data = ir3[7];
              ir3[7] = (v1062_data + (v1016_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1065_data = r2[16];
              float v1068_data = ir3[8];
              ir3[8] = (v1068_data + (v1016_data * (sycl::select_from_group(item.get_sub_group(), v1065_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1071_data_pre = glb_m3[v71_g ? (v126_a) : (0)];
              float v1071_data = v71_g ? (v1071_data_pre) : (0.0f);
              float v1075_data = ir3[0];
              ir3[0] = (v1075_data + (v1071_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1081_data = ir3[1];
              ir3[1] = (v1081_data + (v1071_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1087_data = ir3[2];
              ir3[2] = (v1087_data + (v1071_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1093_data = ir3[3];
              ir3[3] = (v1093_data + (v1071_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1099_data = ir3[4];
              ir3[4] = (v1099_data + (v1071_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1105_data = ir3[5];
              ir3[5] = (v1105_data + (v1071_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1111_data = ir3[6];
              ir3[6] = (v1111_data + (v1071_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1117_data = ir3[7];
              ir3[7] = (v1117_data + (v1071_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1123_data = ir3[8];
              ir3[8] = (v1123_data + (v1071_data * (sycl::select_from_group(item.get_sub_group(), v1065_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1126_data_pre = glb_m3[v71_g ? (v181_a) : (0)];
              float v1126_data = v71_g ? (v1126_data_pre) : (0.0f);
              float v1130_data = ir3[0];
              ir3[0] = (v1130_data + (v1126_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1136_data = ir3[1];
              ir3[1] = (v1136_data + (v1126_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1142_data = ir3[2];
              ir3[2] = (v1142_data + (v1126_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1148_data = ir3[3];
              ir3[3] = (v1148_data + (v1126_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1154_data = ir3[4];
              ir3[4] = (v1154_data + (v1126_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1160_data = ir3[5];
              ir3[5] = (v1160_data + (v1126_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1166_data = ir3[6];
              ir3[6] = (v1166_data + (v1126_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1172_data = ir3[7];
              ir3[7] = (v1172_data + (v1126_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1178_data = ir3[8];
              ir3[8] = (v1178_data + (v1126_data * (sycl::select_from_group(item.get_sub_group(), v1065_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1181_data_pre = glb_m3[v71_g ? (v236_a) : (0)];
              float v1181_data = v71_g ? (v1181_data_pre) : (0.0f);
              float v1185_data = ir3[0];
              ir3[0] = (v1185_data + (v1181_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1191_data = ir3[1];
              ir3[1] = (v1191_data + (v1181_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1197_data = ir3[2];
              ir3[2] = (v1197_data + (v1181_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1203_data = ir3[3];
              ir3[3] = (v1203_data + (v1181_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1209_data = ir3[4];
              ir3[4] = (v1209_data + (v1181_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1215_data = ir3[5];
              ir3[5] = (v1215_data + (v1181_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1221_data = ir3[6];
              ir3[6] = (v1221_data + (v1181_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1227_data = ir3[7];
              ir3[7] = (v1227_data + (v1181_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1233_data = ir3[8];
              ir3[8] = (v1233_data + (v1181_data * (sycl::select_from_group(item.get_sub_group(), v1065_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1236_data_pre = glb_m3[v71_g ? (v291_a) : (0)];
              float v1236_data = v71_g ? (v1236_data_pre) : (0.0f);
              float v1240_data = ir3[0];
              ir3[0] = (v1240_data + (v1236_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1246_data = ir3[1];
              ir3[1] = (v1246_data + (v1236_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1252_data = ir3[2];
              ir3[2] = (v1252_data + (v1236_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1258_data = ir3[3];
              ir3[3] = (v1258_data + (v1236_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1264_data = ir3[4];
              ir3[4] = (v1264_data + (v1236_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1270_data = ir3[5];
              ir3[5] = (v1270_data + (v1236_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1276_data = ir3[6];
              ir3[6] = (v1276_data + (v1236_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1282_data = ir3[7];
              ir3[7] = (v1282_data + (v1236_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1288_data = ir3[8];
              ir3[8] = (v1288_data + (v1236_data * (sycl::select_from_group(item.get_sub_group(), v1065_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1291_data_pre = glb_m3[v71_g ? (v346_a) : (0)];
              float v1291_data = v71_g ? (v1291_data_pre) : (0.0f);
              float v1295_data = ir3[0];
              ir3[0] = (v1295_data + (v1291_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1301_data = ir3[1];
              ir3[1] = (v1301_data + (v1291_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1307_data = ir3[2];
              ir3[2] = (v1307_data + (v1291_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1313_data = ir3[3];
              ir3[3] = (v1313_data + (v1291_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1319_data = ir3[4];
              ir3[4] = (v1319_data + (v1291_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1325_data = ir3[5];
              ir3[5] = (v1325_data + (v1291_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1331_data = ir3[6];
              ir3[6] = (v1331_data + (v1291_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1337_data = ir3[7];
              ir3[7] = (v1337_data + (v1291_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1343_data = ir3[8];
              ir3[8] = (v1343_data + (v1291_data * (sycl::select_from_group(item.get_sub_group(), v1065_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1346_data_pre = glb_m3[v71_g ? (v401_a) : (0)];
              float v1346_data = v71_g ? (v1346_data_pre) : (0.0f);
              float v1350_data = ir3[0];
              ir3[0] = (v1350_data + (v1346_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1356_data = ir3[1];
              ir3[1] = (v1356_data + (v1346_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1362_data = ir3[2];
              ir3[2] = (v1362_data + (v1346_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1368_data = ir3[3];
              ir3[3] = (v1368_data + (v1346_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1374_data = ir3[4];
              ir3[4] = (v1374_data + (v1346_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1380_data = ir3[5];
              ir3[5] = (v1380_data + (v1346_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1386_data = ir3[6];
              ir3[6] = (v1386_data + (v1346_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1392_data = ir3[7];
              ir3[7] = (v1392_data + (v1346_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1398_data = ir3[8];
              ir3[8] = (v1398_data + (v1346_data * (sycl::select_from_group(item.get_sub_group(), v1065_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1401_data_pre = glb_m3[v71_g ? (v456_a) : (0)];
              float v1401_data = v71_g ? (v1401_data_pre) : (0.0f);
              float v1405_data = ir3[0];
              ir3[0] = (v1405_data + (v1401_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1411_data = ir3[1];
              ir3[1] = (v1411_data + (v1401_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1417_data = ir3[2];
              ir3[2] = (v1417_data + (v1401_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1423_data = ir3[3];
              ir3[3] = (v1423_data + (v1401_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1429_data = ir3[4];
              ir3[4] = (v1429_data + (v1401_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1435_data = ir3[5];
              ir3[5] = (v1435_data + (v1401_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1441_data = ir3[6];
              ir3[6] = (v1441_data + (v1401_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1447_data = ir3[7];
              ir3[7] = (v1447_data + (v1401_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1453_data = ir3[8];
              ir3[8] = (v1453_data + (v1401_data * (sycl::select_from_group(item.get_sub_group(), v1065_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1456_data_pre = glb_m3[v71_g ? (v511_a) : (0)];
              float v1456_data = v71_g ? (v1456_data_pre) : (0.0f);
              float v1460_data = ir3[0];
              ir3[0] = (v1460_data + (v1456_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1466_data = ir3[1];
              ir3[1] = (v1466_data + (v1456_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1472_data = ir3[2];
              ir3[2] = (v1472_data + (v1456_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1478_data = ir3[3];
              ir3[3] = (v1478_data + (v1456_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1484_data = ir3[4];
              ir3[4] = (v1484_data + (v1456_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1490_data = ir3[5];
              ir3[5] = (v1490_data + (v1456_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1496_data = ir3[6];
              ir3[6] = (v1496_data + (v1456_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1502_data = ir3[7];
              ir3[7] = (v1502_data + (v1456_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1508_data = ir3[8];
              ir3[8] = (v1508_data + (v1456_data * (sycl::select_from_group(item.get_sub_group(), v1065_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1511_data_pre = glb_m3[v71_g ? (v566_a) : (0)];
              float v1511_data = v71_g ? (v1511_data_pre) : (0.0f);
              float v1515_data = ir3[0];
              ir3[0] = (v1515_data + (v1511_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1521_data = ir3[1];
              ir3[1] = (v1521_data + (v1511_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1527_data = ir3[2];
              ir3[2] = (v1527_data + (v1511_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1533_data = ir3[3];
              ir3[3] = (v1533_data + (v1511_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1539_data = ir3[4];
              ir3[4] = (v1539_data + (v1511_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1545_data = ir3[5];
              ir3[5] = (v1545_data + (v1511_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1551_data = ir3[6];
              ir3[6] = (v1551_data + (v1511_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1557_data = ir3[7];
              ir3[7] = (v1557_data + (v1511_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1563_data = ir3[8];
              ir3[8] = (v1563_data + (v1511_data * (sycl::select_from_group(item.get_sub_group(), v1065_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1566_data_pre = glb_m3[v71_g ? (v621_a) : (0)];
              float v1566_data = v71_g ? (v1566_data_pre) : (0.0f);
              float v1570_data = ir3[0];
              ir3[0] = (v1570_data + (v1566_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1576_data = ir3[1];
              ir3[1] = (v1576_data + (v1566_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1582_data = ir3[2];
              ir3[2] = (v1582_data + (v1566_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1588_data = ir3[3];
              ir3[3] = (v1588_data + (v1566_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1594_data = ir3[4];
              ir3[4] = (v1594_data + (v1566_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1600_data = ir3[5];
              ir3[5] = (v1600_data + (v1566_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1606_data = ir3[6];
              ir3[6] = (v1606_data + (v1566_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1612_data = ir3[7];
              ir3[7] = (v1612_data + (v1566_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1618_data = ir3[8];
              ir3[8] = (v1618_data + (v1566_data * (sycl::select_from_group(item.get_sub_group(), v1065_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1621_data_pre = glb_m3[v71_g ? (v676_a) : (0)];
              float v1621_data = v71_g ? (v1621_data_pre) : (0.0f);
              float v1625_data = ir3[0];
              ir3[0] = (v1625_data + (v1621_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1631_data = ir3[1];
              ir3[1] = (v1631_data + (v1621_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1637_data = ir3[2];
              ir3[2] = (v1637_data + (v1621_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1643_data = ir3[3];
              ir3[3] = (v1643_data + (v1621_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1649_data = ir3[4];
              ir3[4] = (v1649_data + (v1621_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1655_data = ir3[5];
              ir3[5] = (v1655_data + (v1621_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1661_data = ir3[6];
              ir3[6] = (v1661_data + (v1621_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1667_data = ir3[7];
              ir3[7] = (v1667_data + (v1621_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1673_data = ir3[8];
              ir3[8] = (v1673_data + (v1621_data * (sycl::select_from_group(item.get_sub_group(), v1065_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1676_data_pre = glb_m3[v71_g ? (v731_a) : (0)];
              float v1676_data = v71_g ? (v1676_data_pre) : (0.0f);
              float v1680_data = ir3[0];
              ir3[0] = (v1680_data + (v1676_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1686_data = ir3[1];
              ir3[1] = (v1686_data + (v1676_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1692_data = ir3[2];
              ir3[2] = (v1692_data + (v1676_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1698_data = ir3[3];
              ir3[3] = (v1698_data + (v1676_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1704_data = ir3[4];
              ir3[4] = (v1704_data + (v1676_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1710_data = ir3[5];
              ir3[5] = (v1710_data + (v1676_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1716_data = ir3[6];
              ir3[6] = (v1716_data + (v1676_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1722_data = ir3[7];
              ir3[7] = (v1722_data + (v1676_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1728_data = ir3[8];
              ir3[8] = (v1728_data + (v1676_data * (sycl::select_from_group(item.get_sub_group(), v1065_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1731_data_pre = glb_m3[v71_g ? (v786_a) : (0)];
              float v1731_data = v71_g ? (v1731_data_pre) : (0.0f);
              float v1735_data = ir3[0];
              ir3[0] = (v1735_data + (v1731_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1741_data = ir3[1];
              ir3[1] = (v1741_data + (v1731_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1747_data = ir3[2];
              ir3[2] = (v1747_data + (v1731_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1753_data = ir3[3];
              ir3[3] = (v1753_data + (v1731_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1759_data = ir3[4];
              ir3[4] = (v1759_data + (v1731_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1765_data = ir3[5];
              ir3[5] = (v1765_data + (v1731_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1771_data = ir3[6];
              ir3[6] = (v1771_data + (v1731_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1777_data = ir3[7];
              ir3[7] = (v1777_data + (v1731_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1783_data = ir3[8];
              ir3[8] = (v1783_data + (v1731_data * (sycl::select_from_group(item.get_sub_group(), v1065_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1786_data_pre = glb_m3[v71_g ? (v841_a) : (0)];
              float v1786_data = v71_g ? (v1786_data_pre) : (0.0f);
              float v1790_data = ir3[0];
              ir3[0] = (v1790_data + (v1786_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1796_data = ir3[1];
              ir3[1] = (v1796_data + (v1786_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1802_data = ir3[2];
              ir3[2] = (v1802_data + (v1786_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1808_data = ir3[3];
              ir3[3] = (v1808_data + (v1786_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1814_data = ir3[4];
              ir3[4] = (v1814_data + (v1786_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1820_data = ir3[5];
              ir3[5] = (v1820_data + (v1786_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1826_data = ir3[6];
              ir3[6] = (v1826_data + (v1786_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1832_data = ir3[7];
              ir3[7] = (v1832_data + (v1786_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1838_data = ir3[8];
              ir3[8] = (v1838_data + (v1786_data * (sycl::select_from_group(item.get_sub_group(), v1065_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1841_data_pre = glb_m3[v71_g ? (v896_a) : (0)];
              float v1841_data = v71_g ? (v1841_data_pre) : (0.0f);
              float v1842_data = r2[1];
              float v1845_data = ir3[0];
              ir3[0] = (v1845_data + (v1841_data * (sycl::select_from_group(item.get_sub_group(), v1842_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1848_data = r2[3];
              float v1851_data = ir3[1];
              ir3[1] = (v1851_data + (v1841_data * (sycl::select_from_group(item.get_sub_group(), v1848_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1854_data = r2[5];
              float v1857_data = ir3[2];
              ir3[2] = (v1857_data + (v1841_data * (sycl::select_from_group(item.get_sub_group(), v1854_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1860_data = r2[7];
              float v1863_data = ir3[3];
              ir3[3] = (v1863_data + (v1841_data * (sycl::select_from_group(item.get_sub_group(), v1860_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1866_data = r2[9];
              float v1869_data = ir3[4];
              ir3[4] = (v1869_data + (v1841_data * (sycl::select_from_group(item.get_sub_group(), v1866_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1872_data = r2[11];
              float v1875_data = ir3[5];
              ir3[5] = (v1875_data + (v1841_data * (sycl::select_from_group(item.get_sub_group(), v1872_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1878_data = r2[13];
              float v1881_data = ir3[6];
              ir3[6] = (v1881_data + (v1841_data * (sycl::select_from_group(item.get_sub_group(), v1878_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1884_data = r2[15];
              float v1887_data = ir3[7];
              ir3[7] = (v1887_data + (v1841_data * (sycl::select_from_group(item.get_sub_group(), v1884_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1890_data = r2[17];
              float v1893_data = ir3[8];
              ir3[8] = (v1893_data + (v1841_data * (sycl::select_from_group(item.get_sub_group(), v1890_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1896_data_pre = glb_m3[v71_g ? (v951_a) : (0)];
              float v1896_data = v71_g ? (v1896_data_pre) : (0.0f);
              float v1900_data = ir3[0];
              ir3[0] = (v1900_data + (v1896_data * (sycl::select_from_group(item.get_sub_group(), v1842_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1906_data = ir3[1];
              ir3[1] = (v1906_data + (v1896_data * (sycl::select_from_group(item.get_sub_group(), v1848_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1912_data = ir3[2];
              ir3[2] = (v1912_data + (v1896_data * (sycl::select_from_group(item.get_sub_group(), v1854_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1918_data = ir3[3];
              ir3[3] = (v1918_data + (v1896_data * (sycl::select_from_group(item.get_sub_group(), v1860_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1924_data = ir3[4];
              ir3[4] = (v1924_data + (v1896_data * (sycl::select_from_group(item.get_sub_group(), v1866_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1930_data = ir3[5];
              ir3[5] = (v1930_data + (v1896_data * (sycl::select_from_group(item.get_sub_group(), v1872_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1936_data = ir3[6];
              ir3[6] = (v1936_data + (v1896_data * (sycl::select_from_group(item.get_sub_group(), v1878_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1942_data = ir3[7];
              ir3[7] = (v1942_data + (v1896_data * (sycl::select_from_group(item.get_sub_group(), v1884_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1948_data = ir3[8];
              ir3[8] = (v1948_data + (v1896_data * (sycl::select_from_group(item.get_sub_group(), v1890_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1951_data_pre = glb_m3[v71_g ? ((v25_lead + 170)) : (0)];
              float v1951_data = v71_g ? (v1951_data_pre) : (0.0f);
              float v1955_data = ir3[0];
              ir3[0] = (v1955_data + (v1951_data * (sycl::select_from_group(item.get_sub_group(), v1842_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1961_data = ir3[1];
              ir3[1] = (v1961_data + (v1951_data * (sycl::select_from_group(item.get_sub_group(), v1848_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1967_data = ir3[2];
              ir3[2] = (v1967_data + (v1951_data * (sycl::select_from_group(item.get_sub_group(), v1854_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1973_data = ir3[3];
              ir3[3] = (v1973_data + (v1951_data * (sycl::select_from_group(item.get_sub_group(), v1860_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1979_data = ir3[4];
              ir3[4] = (v1979_data + (v1951_data * (sycl::select_from_group(item.get_sub_group(), v1866_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1985_data = ir3[5];
              ir3[5] = (v1985_data + (v1951_data * (sycl::select_from_group(item.get_sub_group(), v1872_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1991_data = ir3[6];
              ir3[6] = (v1991_data + (v1951_data * (sycl::select_from_group(item.get_sub_group(), v1878_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1997_data = ir3[7];
              ir3[7] = (v1997_data + (v1951_data * (sycl::select_from_group(item.get_sub_group(), v1884_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v2003_data = ir3[8];
              ir3[8] = (v2003_data + (v1951_data * (sycl::select_from_group(item.get_sub_group(), v1890_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              // r3 = ir3 + r1
              if (v71_g) {
                #pragma unroll
                for (int32_t v2006_n1 = 0; v2006_n1 < 9; ++v2006_n1) {
                  float v2008_data = ir3[v2006_n1];
                  float v2009_data = r1[v2006_n1];
                  r3[v2006_n1] = (v2009_data + v2008_data);
                }
              }
              // glb_m0 = store{r>g}(r3);
              if (v71_g) {
                #pragma unroll
                for (int32_t v2012_i1 = 0; v2012_i1 < 9; ++v2012_i1) {
                  float v2014_data = r3[v2012_i1];
                  glb_m0[(v25_lead + (v2012_i1 * 10))] = v2014_data;
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

