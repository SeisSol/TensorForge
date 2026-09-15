// === base name ===
kernel_5cf0fc212fdf1786

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_5cf0fc212fdf1786 = {{16, 16, 1}, 16, 10, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_5cf0fc212fdf1786(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_5cf0fc212fdf1786(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_5cf0fc212fdf1786(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_5cf0fc212fdf1786(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_5cf0fc212fdf1786(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_5cf0fc212fdf1786(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, m3, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_5cf0fc212fdf1786(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
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
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":10,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[10,9]],"name":"m0","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"none","alias":"A1","bbox":[[0,1],[10,18]],"name":"m1","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[1,0],[18,9]],"name":"m2","ordered":false,"parts":1,"shape":[20,9],"variant":false},{"addressing":"none","alias":"A2","bbox":[[0,1],[10,19]],"name":"m3","ordered":false,"parts":1,"shape":[16,20],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[1,0],[19,9]],"name":"m4","ordered":false,"parts":1,"shape":[20,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[10,18]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[18,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,1],[10,19]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[16,20]},{"addressing":"strided","bbox":[[1,0],[19,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[20,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          const float *const __restrict__ glb_m1 = &m1[0];
          const float *const __restrict__ glb_m3 = &m3[0];
          for (size_t v5_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v5_batchId0 < numElements0; v5_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v6_ahead1 = v5_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 90 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 153 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v5_batchId0 * 162 + 0 + m4_extraOffset];
              float r0[18]{};
              // r0 = load{g>r}(glb_m2);
              int32_t v19_lead = item.get_local_id(2) % 16;
              bool v20_g = v19_lead >= 1;
              if (v20_g) {
                int32_t v24_a = v19_lead - 1;
                #pragma unroll
                for (int32_t v21_i1 = 0; v21_i1 < 9; ++v21_i1) {
                  float v27_data = glb_m2[(v24_a + (v21_i1 * 17))];
                  r0[(v21_i1 * 2)] = v27_data;
                }
              }
              if (v19_lead < 2) {
                int32_t v34_a = (v19_lead + 16_i32) - 1;
                #pragma unroll
                for (int32_t v31_i1 = 0; v31_i1 < 9; ++v31_i1) {
                  float v37_data = glb_m2[(v34_a + (v31_i1 * 17))];
                  r0[(1 + (v31_i1 * 2))] = v37_data;
                }
              }
              float r2[18]{};
              // r2 = load{g>r}(glb_m4);
              if (v20_g) {
                int32_t v44_a = v19_lead - 1;
                #pragma unroll
                for (int32_t v41_i1 = 0; v41_i1 < 9; ++v41_i1) {
                  float v47_data = glb_m4[(v44_a + (v41_i1 * 18))];
                  r2[(v41_i1 * 2)] = v47_data;
                }
              }
              if (v19_lead < 3) {
                int32_t v54_a = (v19_lead + 16_i32) - 1;
                #pragma unroll
                for (int32_t v51_i1 = 0; v51_i1 < 9; ++v51_i1) {
                  float v57_data = glb_m4[(v54_a + (v51_i1 * 18))];
                  r2[(1 + (v51_i1 * 2))] = v57_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m2););
              float r1[9]{};
              // ir1 = +(glb_m1 * r0)
              // [(0, 10), (0, 9)] [(1, 18)]
              float ir1[9]{};
              bool v65_g = v19_lead < 10;
              float v66_data = v65_g ? (glb_m1[v19_lead]) : (0.0f);
              float v67_data = r0[0];
              float v70_data = ir1[0];
              ir1[0] = (v70_data + (v66_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v73_data = r0[2];
              float v76_data = ir1[1];
              ir1[1] = (v76_data + (v66_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v79_data = r0[4];
              float v82_data = ir1[2];
              ir1[2] = (v82_data + (v66_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v85_data = r0[6];
              float v88_data = ir1[3];
              ir1[3] = (v88_data + (v66_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v91_data = r0[8];
              float v94_data = ir1[4];
              ir1[4] = (v94_data + (v66_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v97_data = r0[10];
              float v100_data = ir1[5];
              ir1[5] = (v100_data + (v66_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v103_data = r0[12];
              float v106_data = ir1[6];
              ir1[6] = (v106_data + (v66_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v109_data = r0[14];
              float v112_data = ir1[7];
              ir1[7] = (v112_data + (v66_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v115_data = r0[16];
              float v118_data = ir1[8];
              ir1[8] = (v118_data + (v66_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              int32_t v120_a = v19_lead + 10;
              float v121_data = v65_g ? (glb_m1[v120_a]) : (0.0f);
              float v125_data = ir1[0];
              ir1[0] = (v125_data + (v121_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v131_data = ir1[1];
              ir1[1] = (v131_data + (v121_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v137_data = ir1[2];
              ir1[2] = (v137_data + (v121_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v143_data = ir1[3];
              ir1[3] = (v143_data + (v121_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v149_data = ir1[4];
              ir1[4] = (v149_data + (v121_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v155_data = ir1[5];
              ir1[5] = (v155_data + (v121_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v161_data = ir1[6];
              ir1[6] = (v161_data + (v121_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v167_data = ir1[7];
              ir1[7] = (v167_data + (v121_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v173_data = ir1[8];
              ir1[8] = (v173_data + (v121_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              int32_t v175_a = v19_lead + 20;
              float v176_data = v65_g ? (glb_m1[v175_a]) : (0.0f);
              float v180_data = ir1[0];
              ir1[0] = (v180_data + (v176_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v186_data = ir1[1];
              ir1[1] = (v186_data + (v176_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v192_data = ir1[2];
              ir1[2] = (v192_data + (v176_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v198_data = ir1[3];
              ir1[3] = (v198_data + (v176_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v204_data = ir1[4];
              ir1[4] = (v204_data + (v176_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v210_data = ir1[5];
              ir1[5] = (v210_data + (v176_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v216_data = ir1[6];
              ir1[6] = (v216_data + (v176_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v222_data = ir1[7];
              ir1[7] = (v222_data + (v176_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v228_data = ir1[8];
              ir1[8] = (v228_data + (v176_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              int32_t v230_a = v19_lead + 30;
              float v231_data = v65_g ? (glb_m1[v230_a]) : (0.0f);
              float v235_data = ir1[0];
              ir1[0] = (v235_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v241_data = ir1[1];
              ir1[1] = (v241_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v247_data = ir1[2];
              ir1[2] = (v247_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v253_data = ir1[3];
              ir1[3] = (v253_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v259_data = ir1[4];
              ir1[4] = (v259_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v265_data = ir1[5];
              ir1[5] = (v265_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v271_data = ir1[6];
              ir1[6] = (v271_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v277_data = ir1[7];
              ir1[7] = (v277_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v283_data = ir1[8];
              ir1[8] = (v283_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              int32_t v285_a = v19_lead + 40;
              float v286_data = v65_g ? (glb_m1[v285_a]) : (0.0f);
              float v290_data = ir1[0];
              ir1[0] = (v290_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v296_data = ir1[1];
              ir1[1] = (v296_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v302_data = ir1[2];
              ir1[2] = (v302_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v308_data = ir1[3];
              ir1[3] = (v308_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v314_data = ir1[4];
              ir1[4] = (v314_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v320_data = ir1[5];
              ir1[5] = (v320_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v326_data = ir1[6];
              ir1[6] = (v326_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v332_data = ir1[7];
              ir1[7] = (v332_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v338_data = ir1[8];
              ir1[8] = (v338_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              int32_t v340_a = v19_lead + 50;
              float v341_data = v65_g ? (glb_m1[v340_a]) : (0.0f);
              float v345_data = ir1[0];
              ir1[0] = (v345_data + (v341_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v351_data = ir1[1];
              ir1[1] = (v351_data + (v341_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v357_data = ir1[2];
              ir1[2] = (v357_data + (v341_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v363_data = ir1[3];
              ir1[3] = (v363_data + (v341_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v369_data = ir1[4];
              ir1[4] = (v369_data + (v341_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v375_data = ir1[5];
              ir1[5] = (v375_data + (v341_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v381_data = ir1[6];
              ir1[6] = (v381_data + (v341_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v387_data = ir1[7];
              ir1[7] = (v387_data + (v341_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v393_data = ir1[8];
              ir1[8] = (v393_data + (v341_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              int32_t v395_a = v19_lead + 60;
              float v396_data = v65_g ? (glb_m1[v395_a]) : (0.0f);
              float v400_data = ir1[0];
              ir1[0] = (v400_data + (v396_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v406_data = ir1[1];
              ir1[1] = (v406_data + (v396_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v412_data = ir1[2];
              ir1[2] = (v412_data + (v396_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v418_data = ir1[3];
              ir1[3] = (v418_data + (v396_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v424_data = ir1[4];
              ir1[4] = (v424_data + (v396_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v430_data = ir1[5];
              ir1[5] = (v430_data + (v396_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v436_data = ir1[6];
              ir1[6] = (v436_data + (v396_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v442_data = ir1[7];
              ir1[7] = (v442_data + (v396_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v448_data = ir1[8];
              ir1[8] = (v448_data + (v396_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              int32_t v450_a = v19_lead + 70;
              float v451_data = v65_g ? (glb_m1[v450_a]) : (0.0f);
              float v455_data = ir1[0];
              ir1[0] = (v455_data + (v451_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v461_data = ir1[1];
              ir1[1] = (v461_data + (v451_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v467_data = ir1[2];
              ir1[2] = (v467_data + (v451_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v473_data = ir1[3];
              ir1[3] = (v473_data + (v451_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v479_data = ir1[4];
              ir1[4] = (v479_data + (v451_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v485_data = ir1[5];
              ir1[5] = (v485_data + (v451_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v491_data = ir1[6];
              ir1[6] = (v491_data + (v451_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v497_data = ir1[7];
              ir1[7] = (v497_data + (v451_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v503_data = ir1[8];
              ir1[8] = (v503_data + (v451_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              int32_t v505_a = v19_lead + 80;
              float v506_data = v65_g ? (glb_m1[v505_a]) : (0.0f);
              float v510_data = ir1[0];
              ir1[0] = (v510_data + (v506_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v516_data = ir1[1];
              ir1[1] = (v516_data + (v506_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v522_data = ir1[2];
              ir1[2] = (v522_data + (v506_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v528_data = ir1[3];
              ir1[3] = (v528_data + (v506_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v534_data = ir1[4];
              ir1[4] = (v534_data + (v506_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v540_data = ir1[5];
              ir1[5] = (v540_data + (v506_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v546_data = ir1[6];
              ir1[6] = (v546_data + (v506_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v552_data = ir1[7];
              ir1[7] = (v552_data + (v506_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v558_data = ir1[8];
              ir1[8] = (v558_data + (v506_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              int32_t v560_a = v19_lead + 90;
              float v561_data = v65_g ? (glb_m1[v560_a]) : (0.0f);
              float v565_data = ir1[0];
              ir1[0] = (v565_data + (v561_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v571_data = ir1[1];
              ir1[1] = (v571_data + (v561_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v577_data = ir1[2];
              ir1[2] = (v577_data + (v561_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v583_data = ir1[3];
              ir1[3] = (v583_data + (v561_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v589_data = ir1[4];
              ir1[4] = (v589_data + (v561_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v595_data = ir1[5];
              ir1[5] = (v595_data + (v561_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v601_data = ir1[6];
              ir1[6] = (v601_data + (v561_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v607_data = ir1[7];
              ir1[7] = (v607_data + (v561_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v613_data = ir1[8];
              ir1[8] = (v613_data + (v561_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              int32_t v615_a = v19_lead + 100;
              float v616_data = v65_g ? (glb_m1[v615_a]) : (0.0f);
              float v620_data = ir1[0];
              ir1[0] = (v620_data + (v616_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v626_data = ir1[1];
              ir1[1] = (v626_data + (v616_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v632_data = ir1[2];
              ir1[2] = (v632_data + (v616_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v638_data = ir1[3];
              ir1[3] = (v638_data + (v616_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v644_data = ir1[4];
              ir1[4] = (v644_data + (v616_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v650_data = ir1[5];
              ir1[5] = (v650_data + (v616_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v656_data = ir1[6];
              ir1[6] = (v656_data + (v616_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v662_data = ir1[7];
              ir1[7] = (v662_data + (v616_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v668_data = ir1[8];
              ir1[8] = (v668_data + (v616_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              int32_t v670_a = v19_lead + 110;
              float v671_data = v65_g ? (glb_m1[v670_a]) : (0.0f);
              float v675_data = ir1[0];
              ir1[0] = (v675_data + (v671_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v681_data = ir1[1];
              ir1[1] = (v681_data + (v671_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v687_data = ir1[2];
              ir1[2] = (v687_data + (v671_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v693_data = ir1[3];
              ir1[3] = (v693_data + (v671_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v699_data = ir1[4];
              ir1[4] = (v699_data + (v671_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v705_data = ir1[5];
              ir1[5] = (v705_data + (v671_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v711_data = ir1[6];
              ir1[6] = (v711_data + (v671_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v717_data = ir1[7];
              ir1[7] = (v717_data + (v671_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v723_data = ir1[8];
              ir1[8] = (v723_data + (v671_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              int32_t v725_a = v19_lead + 120;
              float v726_data = v65_g ? (glb_m1[v725_a]) : (0.0f);
              float v730_data = ir1[0];
              ir1[0] = (v730_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v736_data = ir1[1];
              ir1[1] = (v736_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v742_data = ir1[2];
              ir1[2] = (v742_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v748_data = ir1[3];
              ir1[3] = (v748_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v754_data = ir1[4];
              ir1[4] = (v754_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v760_data = ir1[5];
              ir1[5] = (v760_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v766_data = ir1[6];
              ir1[6] = (v766_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v772_data = ir1[7];
              ir1[7] = (v772_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v778_data = ir1[8];
              ir1[8] = (v778_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              int32_t v780_a = v19_lead + 130;
              float v781_data = v65_g ? (glb_m1[v780_a]) : (0.0f);
              float v785_data = ir1[0];
              ir1[0] = (v785_data + (v781_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v791_data = ir1[1];
              ir1[1] = (v791_data + (v781_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v797_data = ir1[2];
              ir1[2] = (v797_data + (v781_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v803_data = ir1[3];
              ir1[3] = (v803_data + (v781_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v809_data = ir1[4];
              ir1[4] = (v809_data + (v781_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v815_data = ir1[5];
              ir1[5] = (v815_data + (v781_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v821_data = ir1[6];
              ir1[6] = (v821_data + (v781_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v827_data = ir1[7];
              ir1[7] = (v827_data + (v781_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v833_data = ir1[8];
              ir1[8] = (v833_data + (v781_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              int32_t v835_a = v19_lead + 140;
              float v836_data = v65_g ? (glb_m1[v835_a]) : (0.0f);
              float v840_data = ir1[0];
              ir1[0] = (v840_data + (v836_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v846_data = ir1[1];
              ir1[1] = (v846_data + (v836_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v852_data = ir1[2];
              ir1[2] = (v852_data + (v836_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v858_data = ir1[3];
              ir1[3] = (v858_data + (v836_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v864_data = ir1[4];
              ir1[4] = (v864_data + (v836_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v870_data = ir1[5];
              ir1[5] = (v870_data + (v836_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v876_data = ir1[6];
              ir1[6] = (v876_data + (v836_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v882_data = ir1[7];
              ir1[7] = (v882_data + (v836_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v888_data = ir1[8];
              ir1[8] = (v888_data + (v836_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              int32_t v890_a = v19_lead + 150;
              float v891_data = v65_g ? (glb_m1[v890_a]) : (0.0f);
              float v892_data = r0[1];
              float v895_data = ir1[0];
              ir1[0] = (v895_data + (v891_data * (sycl::select_from_group(item.get_sub_group(), v892_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v898_data = r0[3];
              float v901_data = ir1[1];
              ir1[1] = (v901_data + (v891_data * (sycl::select_from_group(item.get_sub_group(), v898_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v904_data = r0[5];
              float v907_data = ir1[2];
              ir1[2] = (v907_data + (v891_data * (sycl::select_from_group(item.get_sub_group(), v904_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v910_data = r0[7];
              float v913_data = ir1[3];
              ir1[3] = (v913_data + (v891_data * (sycl::select_from_group(item.get_sub_group(), v910_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v916_data = r0[9];
              float v919_data = ir1[4];
              ir1[4] = (v919_data + (v891_data * (sycl::select_from_group(item.get_sub_group(), v916_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v922_data = r0[11];
              float v925_data = ir1[5];
              ir1[5] = (v925_data + (v891_data * (sycl::select_from_group(item.get_sub_group(), v922_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v928_data = r0[13];
              float v931_data = ir1[6];
              ir1[6] = (v931_data + (v891_data * (sycl::select_from_group(item.get_sub_group(), v928_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v934_data = r0[15];
              float v937_data = ir1[7];
              ir1[7] = (v937_data + (v891_data * (sycl::select_from_group(item.get_sub_group(), v934_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v940_data = r0[17];
              float v943_data = ir1[8];
              ir1[8] = (v943_data + (v891_data * (sycl::select_from_group(item.get_sub_group(), v940_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              int32_t v945_a = v19_lead + 160;
              float v946_data = v65_g ? (glb_m1[v945_a]) : (0.0f);
              float v950_data = ir1[0];
              ir1[0] = (v950_data + (v946_data * (sycl::select_from_group(item.get_sub_group(), v892_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v956_data = ir1[1];
              ir1[1] = (v956_data + (v946_data * (sycl::select_from_group(item.get_sub_group(), v898_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v962_data = ir1[2];
              ir1[2] = (v962_data + (v946_data * (sycl::select_from_group(item.get_sub_group(), v904_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v968_data = ir1[3];
              ir1[3] = (v968_data + (v946_data * (sycl::select_from_group(item.get_sub_group(), v910_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v974_data = ir1[4];
              ir1[4] = (v974_data + (v946_data * (sycl::select_from_group(item.get_sub_group(), v916_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v980_data = ir1[5];
              ir1[5] = (v980_data + (v946_data * (sycl::select_from_group(item.get_sub_group(), v922_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v986_data = ir1[6];
              ir1[6] = (v986_data + (v946_data * (sycl::select_from_group(item.get_sub_group(), v928_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v992_data = ir1[7];
              ir1[7] = (v992_data + (v946_data * (sycl::select_from_group(item.get_sub_group(), v934_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v998_data = ir1[8];
              ir1[8] = (v998_data + (v946_data * (sycl::select_from_group(item.get_sub_group(), v940_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              // r1 = ir1
              if (v65_g) {
                #pragma unroll
                for (int32_t v1001_n1 = 0; v1001_n1 < 9; ++v1001_n1) {
                  float v1003_data = ir1[v1001_n1];
                  r1[v1001_n1] = v1003_data;
                }
              }
              // wait(r2 = load{g>r}(glb_m4););
              float r3[9]{};
              // ir3 = +(glb_m3 * r2)
              // [(0, 10), (0, 9)] [(1, 19)]
              float ir3[9]{};
              float v1010_data = v65_g ? (glb_m3[v19_lead]) : (0.0f);
              float v1011_data = r2[0];
              float v1014_data = ir3[0];
              ir3[0] = (v1014_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1017_data = r2[2];
              float v1020_data = ir3[1];
              ir3[1] = (v1020_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1023_data = r2[4];
              float v1026_data = ir3[2];
              ir3[2] = (v1026_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1029_data = r2[6];
              float v1032_data = ir3[3];
              ir3[3] = (v1032_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1035_data = r2[8];
              float v1038_data = ir3[4];
              ir3[4] = (v1038_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1041_data = r2[10];
              float v1044_data = ir3[5];
              ir3[5] = (v1044_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1047_data = r2[12];
              float v1050_data = ir3[6];
              ir3[6] = (v1050_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1053_data = r2[14];
              float v1056_data = ir3[7];
              ir3[7] = (v1056_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1059_data = r2[16];
              float v1062_data = ir3[8];
              ir3[8] = (v1062_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1065_data = v65_g ? (glb_m3[v120_a]) : (0.0f);
              float v1069_data = ir3[0];
              ir3[0] = (v1069_data + (v1065_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1075_data = ir3[1];
              ir3[1] = (v1075_data + (v1065_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1081_data = ir3[2];
              ir3[2] = (v1081_data + (v1065_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1087_data = ir3[3];
              ir3[3] = (v1087_data + (v1065_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1093_data = ir3[4];
              ir3[4] = (v1093_data + (v1065_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1099_data = ir3[5];
              ir3[5] = (v1099_data + (v1065_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1105_data = ir3[6];
              ir3[6] = (v1105_data + (v1065_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1111_data = ir3[7];
              ir3[7] = (v1111_data + (v1065_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1117_data = ir3[8];
              ir3[8] = (v1117_data + (v1065_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1120_data = v65_g ? (glb_m3[v175_a]) : (0.0f);
              float v1124_data = ir3[0];
              ir3[0] = (v1124_data + (v1120_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1130_data = ir3[1];
              ir3[1] = (v1130_data + (v1120_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1136_data = ir3[2];
              ir3[2] = (v1136_data + (v1120_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1142_data = ir3[3];
              ir3[3] = (v1142_data + (v1120_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1148_data = ir3[4];
              ir3[4] = (v1148_data + (v1120_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1154_data = ir3[5];
              ir3[5] = (v1154_data + (v1120_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1160_data = ir3[6];
              ir3[6] = (v1160_data + (v1120_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1166_data = ir3[7];
              ir3[7] = (v1166_data + (v1120_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1172_data = ir3[8];
              ir3[8] = (v1172_data + (v1120_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1175_data = v65_g ? (glb_m3[v230_a]) : (0.0f);
              float v1179_data = ir3[0];
              ir3[0] = (v1179_data + (v1175_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1185_data = ir3[1];
              ir3[1] = (v1185_data + (v1175_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1191_data = ir3[2];
              ir3[2] = (v1191_data + (v1175_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1197_data = ir3[3];
              ir3[3] = (v1197_data + (v1175_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1203_data = ir3[4];
              ir3[4] = (v1203_data + (v1175_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1209_data = ir3[5];
              ir3[5] = (v1209_data + (v1175_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1215_data = ir3[6];
              ir3[6] = (v1215_data + (v1175_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1221_data = ir3[7];
              ir3[7] = (v1221_data + (v1175_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1227_data = ir3[8];
              ir3[8] = (v1227_data + (v1175_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1230_data = v65_g ? (glb_m3[v285_a]) : (0.0f);
              float v1234_data = ir3[0];
              ir3[0] = (v1234_data + (v1230_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1240_data = ir3[1];
              ir3[1] = (v1240_data + (v1230_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1246_data = ir3[2];
              ir3[2] = (v1246_data + (v1230_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1252_data = ir3[3];
              ir3[3] = (v1252_data + (v1230_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1258_data = ir3[4];
              ir3[4] = (v1258_data + (v1230_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1264_data = ir3[5];
              ir3[5] = (v1264_data + (v1230_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1270_data = ir3[6];
              ir3[6] = (v1270_data + (v1230_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1276_data = ir3[7];
              ir3[7] = (v1276_data + (v1230_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1282_data = ir3[8];
              ir3[8] = (v1282_data + (v1230_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1285_data = v65_g ? (glb_m3[v340_a]) : (0.0f);
              float v1289_data = ir3[0];
              ir3[0] = (v1289_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1295_data = ir3[1];
              ir3[1] = (v1295_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1301_data = ir3[2];
              ir3[2] = (v1301_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1307_data = ir3[3];
              ir3[3] = (v1307_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1313_data = ir3[4];
              ir3[4] = (v1313_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1319_data = ir3[5];
              ir3[5] = (v1319_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1325_data = ir3[6];
              ir3[6] = (v1325_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1331_data = ir3[7];
              ir3[7] = (v1331_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1337_data = ir3[8];
              ir3[8] = (v1337_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1340_data = v65_g ? (glb_m3[v395_a]) : (0.0f);
              float v1344_data = ir3[0];
              ir3[0] = (v1344_data + (v1340_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1350_data = ir3[1];
              ir3[1] = (v1350_data + (v1340_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1356_data = ir3[2];
              ir3[2] = (v1356_data + (v1340_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1362_data = ir3[3];
              ir3[3] = (v1362_data + (v1340_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1368_data = ir3[4];
              ir3[4] = (v1368_data + (v1340_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1374_data = ir3[5];
              ir3[5] = (v1374_data + (v1340_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1380_data = ir3[6];
              ir3[6] = (v1380_data + (v1340_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1386_data = ir3[7];
              ir3[7] = (v1386_data + (v1340_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1392_data = ir3[8];
              ir3[8] = (v1392_data + (v1340_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1395_data = v65_g ? (glb_m3[v450_a]) : (0.0f);
              float v1399_data = ir3[0];
              ir3[0] = (v1399_data + (v1395_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1405_data = ir3[1];
              ir3[1] = (v1405_data + (v1395_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1411_data = ir3[2];
              ir3[2] = (v1411_data + (v1395_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1417_data = ir3[3];
              ir3[3] = (v1417_data + (v1395_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1423_data = ir3[4];
              ir3[4] = (v1423_data + (v1395_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1429_data = ir3[5];
              ir3[5] = (v1429_data + (v1395_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1435_data = ir3[6];
              ir3[6] = (v1435_data + (v1395_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1441_data = ir3[7];
              ir3[7] = (v1441_data + (v1395_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1447_data = ir3[8];
              ir3[8] = (v1447_data + (v1395_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1450_data = v65_g ? (glb_m3[v505_a]) : (0.0f);
              float v1454_data = ir3[0];
              ir3[0] = (v1454_data + (v1450_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1460_data = ir3[1];
              ir3[1] = (v1460_data + (v1450_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1466_data = ir3[2];
              ir3[2] = (v1466_data + (v1450_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1472_data = ir3[3];
              ir3[3] = (v1472_data + (v1450_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1478_data = ir3[4];
              ir3[4] = (v1478_data + (v1450_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1484_data = ir3[5];
              ir3[5] = (v1484_data + (v1450_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1490_data = ir3[6];
              ir3[6] = (v1490_data + (v1450_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1496_data = ir3[7];
              ir3[7] = (v1496_data + (v1450_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1502_data = ir3[8];
              ir3[8] = (v1502_data + (v1450_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1505_data = v65_g ? (glb_m3[v560_a]) : (0.0f);
              float v1509_data = ir3[0];
              ir3[0] = (v1509_data + (v1505_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1515_data = ir3[1];
              ir3[1] = (v1515_data + (v1505_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1521_data = ir3[2];
              ir3[2] = (v1521_data + (v1505_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1527_data = ir3[3];
              ir3[3] = (v1527_data + (v1505_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1533_data = ir3[4];
              ir3[4] = (v1533_data + (v1505_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1539_data = ir3[5];
              ir3[5] = (v1539_data + (v1505_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1545_data = ir3[6];
              ir3[6] = (v1545_data + (v1505_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1551_data = ir3[7];
              ir3[7] = (v1551_data + (v1505_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1557_data = ir3[8];
              ir3[8] = (v1557_data + (v1505_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1560_data = v65_g ? (glb_m3[v615_a]) : (0.0f);
              float v1564_data = ir3[0];
              ir3[0] = (v1564_data + (v1560_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1570_data = ir3[1];
              ir3[1] = (v1570_data + (v1560_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1576_data = ir3[2];
              ir3[2] = (v1576_data + (v1560_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1582_data = ir3[3];
              ir3[3] = (v1582_data + (v1560_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1588_data = ir3[4];
              ir3[4] = (v1588_data + (v1560_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1594_data = ir3[5];
              ir3[5] = (v1594_data + (v1560_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1600_data = ir3[6];
              ir3[6] = (v1600_data + (v1560_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1606_data = ir3[7];
              ir3[7] = (v1606_data + (v1560_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1612_data = ir3[8];
              ir3[8] = (v1612_data + (v1560_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1615_data = v65_g ? (glb_m3[v670_a]) : (0.0f);
              float v1619_data = ir3[0];
              ir3[0] = (v1619_data + (v1615_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1625_data = ir3[1];
              ir3[1] = (v1625_data + (v1615_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1631_data = ir3[2];
              ir3[2] = (v1631_data + (v1615_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1637_data = ir3[3];
              ir3[3] = (v1637_data + (v1615_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1643_data = ir3[4];
              ir3[4] = (v1643_data + (v1615_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1649_data = ir3[5];
              ir3[5] = (v1649_data + (v1615_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1655_data = ir3[6];
              ir3[6] = (v1655_data + (v1615_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1661_data = ir3[7];
              ir3[7] = (v1661_data + (v1615_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1667_data = ir3[8];
              ir3[8] = (v1667_data + (v1615_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1670_data = v65_g ? (glb_m3[v725_a]) : (0.0f);
              float v1674_data = ir3[0];
              ir3[0] = (v1674_data + (v1670_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1680_data = ir3[1];
              ir3[1] = (v1680_data + (v1670_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1686_data = ir3[2];
              ir3[2] = (v1686_data + (v1670_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1692_data = ir3[3];
              ir3[3] = (v1692_data + (v1670_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1698_data = ir3[4];
              ir3[4] = (v1698_data + (v1670_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1704_data = ir3[5];
              ir3[5] = (v1704_data + (v1670_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1710_data = ir3[6];
              ir3[6] = (v1710_data + (v1670_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1716_data = ir3[7];
              ir3[7] = (v1716_data + (v1670_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1722_data = ir3[8];
              ir3[8] = (v1722_data + (v1670_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1725_data = v65_g ? (glb_m3[v780_a]) : (0.0f);
              float v1729_data = ir3[0];
              ir3[0] = (v1729_data + (v1725_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1735_data = ir3[1];
              ir3[1] = (v1735_data + (v1725_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1741_data = ir3[2];
              ir3[2] = (v1741_data + (v1725_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1747_data = ir3[3];
              ir3[3] = (v1747_data + (v1725_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1753_data = ir3[4];
              ir3[4] = (v1753_data + (v1725_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1759_data = ir3[5];
              ir3[5] = (v1759_data + (v1725_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1765_data = ir3[6];
              ir3[6] = (v1765_data + (v1725_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1771_data = ir3[7];
              ir3[7] = (v1771_data + (v1725_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1777_data = ir3[8];
              ir3[8] = (v1777_data + (v1725_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1780_data = v65_g ? (glb_m3[v835_a]) : (0.0f);
              float v1784_data = ir3[0];
              ir3[0] = (v1784_data + (v1780_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1790_data = ir3[1];
              ir3[1] = (v1790_data + (v1780_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1796_data = ir3[2];
              ir3[2] = (v1796_data + (v1780_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1802_data = ir3[3];
              ir3[3] = (v1802_data + (v1780_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1808_data = ir3[4];
              ir3[4] = (v1808_data + (v1780_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1814_data = ir3[5];
              ir3[5] = (v1814_data + (v1780_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1820_data = ir3[6];
              ir3[6] = (v1820_data + (v1780_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1826_data = ir3[7];
              ir3[7] = (v1826_data + (v1780_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1832_data = ir3[8];
              ir3[8] = (v1832_data + (v1780_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1835_data = v65_g ? (glb_m3[v890_a]) : (0.0f);
              float v1836_data = r2[1];
              float v1839_data = ir3[0];
              ir3[0] = (v1839_data + (v1835_data * (sycl::select_from_group(item.get_sub_group(), v1836_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1842_data = r2[3];
              float v1845_data = ir3[1];
              ir3[1] = (v1845_data + (v1835_data * (sycl::select_from_group(item.get_sub_group(), v1842_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1848_data = r2[5];
              float v1851_data = ir3[2];
              ir3[2] = (v1851_data + (v1835_data * (sycl::select_from_group(item.get_sub_group(), v1848_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1854_data = r2[7];
              float v1857_data = ir3[3];
              ir3[3] = (v1857_data + (v1835_data * (sycl::select_from_group(item.get_sub_group(), v1854_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1860_data = r2[9];
              float v1863_data = ir3[4];
              ir3[4] = (v1863_data + (v1835_data * (sycl::select_from_group(item.get_sub_group(), v1860_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1866_data = r2[11];
              float v1869_data = ir3[5];
              ir3[5] = (v1869_data + (v1835_data * (sycl::select_from_group(item.get_sub_group(), v1866_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1872_data = r2[13];
              float v1875_data = ir3[6];
              ir3[6] = (v1875_data + (v1835_data * (sycl::select_from_group(item.get_sub_group(), v1872_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1878_data = r2[15];
              float v1881_data = ir3[7];
              ir3[7] = (v1881_data + (v1835_data * (sycl::select_from_group(item.get_sub_group(), v1878_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1884_data = r2[17];
              float v1887_data = ir3[8];
              ir3[8] = (v1887_data + (v1835_data * (sycl::select_from_group(item.get_sub_group(), v1884_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1890_data = v65_g ? (glb_m3[v945_a]) : (0.0f);
              float v1894_data = ir3[0];
              ir3[0] = (v1894_data + (v1890_data * (sycl::select_from_group(item.get_sub_group(), v1836_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1900_data = ir3[1];
              ir3[1] = (v1900_data + (v1890_data * (sycl::select_from_group(item.get_sub_group(), v1842_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1906_data = ir3[2];
              ir3[2] = (v1906_data + (v1890_data * (sycl::select_from_group(item.get_sub_group(), v1848_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1912_data = ir3[3];
              ir3[3] = (v1912_data + (v1890_data * (sycl::select_from_group(item.get_sub_group(), v1854_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1918_data = ir3[4];
              ir3[4] = (v1918_data + (v1890_data * (sycl::select_from_group(item.get_sub_group(), v1860_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1924_data = ir3[5];
              ir3[5] = (v1924_data + (v1890_data * (sycl::select_from_group(item.get_sub_group(), v1866_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1930_data = ir3[6];
              ir3[6] = (v1930_data + (v1890_data * (sycl::select_from_group(item.get_sub_group(), v1872_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1936_data = ir3[7];
              ir3[7] = (v1936_data + (v1890_data * (sycl::select_from_group(item.get_sub_group(), v1878_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1942_data = ir3[8];
              ir3[8] = (v1942_data + (v1890_data * (sycl::select_from_group(item.get_sub_group(), v1884_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1945_data = v65_g ? (glb_m3[(v19_lead + 170)]) : (0.0f);
              float v1949_data = ir3[0];
              ir3[0] = (v1949_data + (v1945_data * (sycl::select_from_group(item.get_sub_group(), v1836_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1955_data = ir3[1];
              ir3[1] = (v1955_data + (v1945_data * (sycl::select_from_group(item.get_sub_group(), v1842_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1961_data = ir3[2];
              ir3[2] = (v1961_data + (v1945_data * (sycl::select_from_group(item.get_sub_group(), v1848_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1967_data = ir3[3];
              ir3[3] = (v1967_data + (v1945_data * (sycl::select_from_group(item.get_sub_group(), v1854_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1973_data = ir3[4];
              ir3[4] = (v1973_data + (v1945_data * (sycl::select_from_group(item.get_sub_group(), v1860_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1979_data = ir3[5];
              ir3[5] = (v1979_data + (v1945_data * (sycl::select_from_group(item.get_sub_group(), v1866_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1985_data = ir3[6];
              ir3[6] = (v1985_data + (v1945_data * (sycl::select_from_group(item.get_sub_group(), v1872_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1991_data = ir3[7];
              ir3[7] = (v1991_data + (v1945_data * (sycl::select_from_group(item.get_sub_group(), v1878_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1997_data = ir3[8];
              ir3[8] = (v1997_data + (v1945_data * (sycl::select_from_group(item.get_sub_group(), v1884_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              // r3 = ir3 + r1
              if (v65_g) {
                #pragma unroll
                for (int32_t v2000_n1 = 0; v2000_n1 < 9; ++v2000_n1) {
                  float v2002_data = ir3[v2000_n1];
                  float v2003_data = r1[v2000_n1];
                  r3[v2000_n1] = (v2003_data + v2002_data);
                }
              }
              // glb_m0 = store{r>g}(r3);
              if (v65_g) {
                #pragma unroll
                for (int32_t v2006_i1 = 0; v2006_i1 < 9; ++v2006_i1) {
                  float v2008_data = r3[v2006_i1];
                  glb_m0[(v19_lead + (v2006_i1 * 10))] = v2008_data;
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

