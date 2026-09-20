// === base name ===
kernel_73a563650d77dedf

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_73a563650d77dedf = {{16, 16, 1}, 16, 10, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_73a563650d77dedf(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_73a563650d77dedf(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_73a563650d77dedf(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_73a563650d77dedf(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_73a563650d77dedf(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_73a563650d77dedf(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, m3, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_73a563650d77dedf(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (10 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 10×9(10×9) {0..10}×{0..9} strided
        //   m1 10×17(10×17) {0..10}×{0..17} none
        //   m2 17×9(17×9) {0..17}×{0..9} strided
        //   m3 10×17(10×17) {0..10}×{0..17} none
        //   m4 17×9(17×9) {0..17}×{0..9} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        //   m0[i,j] += m3[i,k] × m4[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":10,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[10,9]],"name":"m0","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"none","alias":"A1","bbox":[[0,0],[10,17]],"name":"m1","ordered":false,"parts":1,"shape":[10,17],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[17,9]],"name":"m2","ordered":false,"parts":1,"shape":[17,9],"variant":false},{"addressing":"none","alias":"A2","bbox":[[0,0],[10,17]],"name":"m3","ordered":false,"parts":1,"shape":[10,17],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[17,9]],"name":"m4","ordered":false,"parts":1,"shape":[17,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[10,17]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[10,17]},{"addressing":"strided","bbox":[[0,0],[17,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[17,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[10,17]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[10,17]},{"addressing":"strided","bbox":[[0,0],[17,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[17,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
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
              const float *const __restrict__ glb_m4 = &m4[v5_batchId0 * 153 + 0 + m4_extraOffset];
              float r0[18]{};
              // r0 = load{g>r}(glb_m2);
              int32_t v19_lead = item.get_local_id(2) % 16;
              #pragma unroll
              for (int32_t v20_i0 = 0; v20_i0 < 1; ++v20_i0) {
                int32_t v23_lead = v19_lead + (v20_i0 * 16);
                #pragma unroll
                for (int32_t v21_i1 = 0; v21_i1 < 9; ++v21_i1) {
                  float v26_data = glb_m2[(v23_lead + (v21_i1 * 17))];
                  r0[(v20_i0 + (v21_i1 * 2))] = v26_data;
                }
              }
              bool v29_g = v19_lead < 1;
              if (v29_g) {
                int32_t v32_lead = v19_lead + 16_i32;
                #pragma unroll
                for (int32_t v30_i1 = 0; v30_i1 < 9; ++v30_i1) {
                  float v35_data = glb_m2[(v32_lead + (v30_i1 * 17))];
                  r0[(1 + (v30_i1 * 2))] = v35_data;
                }
              }
              float r2[18]{};
              // r2 = load{g>r}(glb_m4);
              #pragma unroll
              for (int32_t v39_i0 = 0; v39_i0 < 1; ++v39_i0) {
                int32_t v42_lead = v19_lead + (v39_i0 * 16);
                #pragma unroll
                for (int32_t v40_i1 = 0; v40_i1 < 9; ++v40_i1) {
                  float v45_data = glb_m4[(v42_lead + (v40_i1 * 17))];
                  r2[(v39_i0 + (v40_i1 * 2))] = v45_data;
                }
              }
              if (v29_g) {
                int32_t v50_lead = v19_lead + 16_i32;
                #pragma unroll
                for (int32_t v48_i1 = 0; v48_i1 < 9; ++v48_i1) {
                  float v53_data = glb_m4[(v50_lead + (v48_i1 * 17))];
                  r2[(1 + (v48_i1 * 2))] = v53_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m2););
              float r1[9]{};
              // ir1 = +(glb_m1 * r0)
              // [(0, 10), (0, 9)] [(0, 17)]
              float ir1[9]{};
              bool v61_g = v19_lead < 10;
              float v62_data_pre = glb_m1[v61_g ? (v19_lead) : (0)];
              float v62_data = v61_g ? (v62_data_pre) : (0.0f);
              float v63_data = r0[0];
              float v66_data = ir1[0];
              ir1[0] = (v66_data + (v62_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v69_data = r0[2];
              float v72_data = ir1[1];
              ir1[1] = (v72_data + (v62_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v75_data = r0[4];
              float v78_data = ir1[2];
              ir1[2] = (v78_data + (v62_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v81_data = r0[6];
              float v84_data = ir1[3];
              ir1[3] = (v84_data + (v62_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v87_data = r0[8];
              float v90_data = ir1[4];
              ir1[4] = (v90_data + (v62_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v93_data = r0[10];
              float v96_data = ir1[5];
              ir1[5] = (v96_data + (v62_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v99_data = r0[12];
              float v102_data = ir1[6];
              ir1[6] = (v102_data + (v62_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v105_data = r0[14];
              float v108_data = ir1[7];
              ir1[7] = (v108_data + (v62_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v111_data = r0[16];
              float v114_data = ir1[8];
              ir1[8] = (v114_data + (v62_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              int32_t v116_a = v19_lead + 10;
              float v117_data_pre = glb_m1[v61_g ? (v116_a) : (0)];
              float v117_data = v61_g ? (v117_data_pre) : (0.0f);
              float v121_data = ir1[0];
              ir1[0] = (v121_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v127_data = ir1[1];
              ir1[1] = (v127_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v133_data = ir1[2];
              ir1[2] = (v133_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v139_data = ir1[3];
              ir1[3] = (v139_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v145_data = ir1[4];
              ir1[4] = (v145_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v151_data = ir1[5];
              ir1[5] = (v151_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v157_data = ir1[6];
              ir1[6] = (v157_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v163_data = ir1[7];
              ir1[7] = (v163_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v169_data = ir1[8];
              ir1[8] = (v169_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              int32_t v171_a = v19_lead + 20;
              float v172_data_pre = glb_m1[v61_g ? (v171_a) : (0)];
              float v172_data = v61_g ? (v172_data_pre) : (0.0f);
              float v176_data = ir1[0];
              ir1[0] = (v176_data + (v172_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v182_data = ir1[1];
              ir1[1] = (v182_data + (v172_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v188_data = ir1[2];
              ir1[2] = (v188_data + (v172_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v194_data = ir1[3];
              ir1[3] = (v194_data + (v172_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v200_data = ir1[4];
              ir1[4] = (v200_data + (v172_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v206_data = ir1[5];
              ir1[5] = (v206_data + (v172_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v212_data = ir1[6];
              ir1[6] = (v212_data + (v172_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v218_data = ir1[7];
              ir1[7] = (v218_data + (v172_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v224_data = ir1[8];
              ir1[8] = (v224_data + (v172_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              int32_t v226_a = v19_lead + 30;
              float v227_data_pre = glb_m1[v61_g ? (v226_a) : (0)];
              float v227_data = v61_g ? (v227_data_pre) : (0.0f);
              float v231_data = ir1[0];
              ir1[0] = (v231_data + (v227_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v237_data = ir1[1];
              ir1[1] = (v237_data + (v227_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v243_data = ir1[2];
              ir1[2] = (v243_data + (v227_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v249_data = ir1[3];
              ir1[3] = (v249_data + (v227_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v255_data = ir1[4];
              ir1[4] = (v255_data + (v227_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v261_data = ir1[5];
              ir1[5] = (v261_data + (v227_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v267_data = ir1[6];
              ir1[6] = (v267_data + (v227_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v273_data = ir1[7];
              ir1[7] = (v273_data + (v227_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v279_data = ir1[8];
              ir1[8] = (v279_data + (v227_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              int32_t v281_a = v19_lead + 40;
              float v282_data_pre = glb_m1[v61_g ? (v281_a) : (0)];
              float v282_data = v61_g ? (v282_data_pre) : (0.0f);
              float v286_data = ir1[0];
              ir1[0] = (v286_data + (v282_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v292_data = ir1[1];
              ir1[1] = (v292_data + (v282_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v298_data = ir1[2];
              ir1[2] = (v298_data + (v282_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v304_data = ir1[3];
              ir1[3] = (v304_data + (v282_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v310_data = ir1[4];
              ir1[4] = (v310_data + (v282_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v316_data = ir1[5];
              ir1[5] = (v316_data + (v282_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v322_data = ir1[6];
              ir1[6] = (v322_data + (v282_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v328_data = ir1[7];
              ir1[7] = (v328_data + (v282_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v334_data = ir1[8];
              ir1[8] = (v334_data + (v282_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              int32_t v336_a = v19_lead + 50;
              float v337_data_pre = glb_m1[v61_g ? (v336_a) : (0)];
              float v337_data = v61_g ? (v337_data_pre) : (0.0f);
              float v341_data = ir1[0];
              ir1[0] = (v341_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v347_data = ir1[1];
              ir1[1] = (v347_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v353_data = ir1[2];
              ir1[2] = (v353_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v359_data = ir1[3];
              ir1[3] = (v359_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v365_data = ir1[4];
              ir1[4] = (v365_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v371_data = ir1[5];
              ir1[5] = (v371_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v377_data = ir1[6];
              ir1[6] = (v377_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v383_data = ir1[7];
              ir1[7] = (v383_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v389_data = ir1[8];
              ir1[8] = (v389_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              int32_t v391_a = v19_lead + 60;
              float v392_data_pre = glb_m1[v61_g ? (v391_a) : (0)];
              float v392_data = v61_g ? (v392_data_pre) : (0.0f);
              float v396_data = ir1[0];
              ir1[0] = (v396_data + (v392_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v402_data = ir1[1];
              ir1[1] = (v402_data + (v392_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v408_data = ir1[2];
              ir1[2] = (v408_data + (v392_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v414_data = ir1[3];
              ir1[3] = (v414_data + (v392_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v420_data = ir1[4];
              ir1[4] = (v420_data + (v392_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v426_data = ir1[5];
              ir1[5] = (v426_data + (v392_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v432_data = ir1[6];
              ir1[6] = (v432_data + (v392_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v438_data = ir1[7];
              ir1[7] = (v438_data + (v392_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v444_data = ir1[8];
              ir1[8] = (v444_data + (v392_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              int32_t v446_a = v19_lead + 70;
              float v447_data_pre = glb_m1[v61_g ? (v446_a) : (0)];
              float v447_data = v61_g ? (v447_data_pre) : (0.0f);
              float v451_data = ir1[0];
              ir1[0] = (v451_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v457_data = ir1[1];
              ir1[1] = (v457_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v463_data = ir1[2];
              ir1[2] = (v463_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v469_data = ir1[3];
              ir1[3] = (v469_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v475_data = ir1[4];
              ir1[4] = (v475_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v481_data = ir1[5];
              ir1[5] = (v481_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v487_data = ir1[6];
              ir1[6] = (v487_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v493_data = ir1[7];
              ir1[7] = (v493_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v499_data = ir1[8];
              ir1[8] = (v499_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              int32_t v501_a = v19_lead + 80;
              float v502_data_pre = glb_m1[v61_g ? (v501_a) : (0)];
              float v502_data = v61_g ? (v502_data_pre) : (0.0f);
              float v506_data = ir1[0];
              ir1[0] = (v506_data + (v502_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v512_data = ir1[1];
              ir1[1] = (v512_data + (v502_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v518_data = ir1[2];
              ir1[2] = (v518_data + (v502_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v524_data = ir1[3];
              ir1[3] = (v524_data + (v502_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v530_data = ir1[4];
              ir1[4] = (v530_data + (v502_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v536_data = ir1[5];
              ir1[5] = (v536_data + (v502_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v542_data = ir1[6];
              ir1[6] = (v542_data + (v502_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v548_data = ir1[7];
              ir1[7] = (v548_data + (v502_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v554_data = ir1[8];
              ir1[8] = (v554_data + (v502_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              int32_t v556_a = v19_lead + 90;
              float v557_data_pre = glb_m1[v61_g ? (v556_a) : (0)];
              float v557_data = v61_g ? (v557_data_pre) : (0.0f);
              float v561_data = ir1[0];
              ir1[0] = (v561_data + (v557_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v567_data = ir1[1];
              ir1[1] = (v567_data + (v557_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v573_data = ir1[2];
              ir1[2] = (v573_data + (v557_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v579_data = ir1[3];
              ir1[3] = (v579_data + (v557_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v585_data = ir1[4];
              ir1[4] = (v585_data + (v557_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v591_data = ir1[5];
              ir1[5] = (v591_data + (v557_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v597_data = ir1[6];
              ir1[6] = (v597_data + (v557_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v603_data = ir1[7];
              ir1[7] = (v603_data + (v557_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v609_data = ir1[8];
              ir1[8] = (v609_data + (v557_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              int32_t v611_a = v19_lead + 100;
              float v612_data_pre = glb_m1[v61_g ? (v611_a) : (0)];
              float v612_data = v61_g ? (v612_data_pre) : (0.0f);
              float v616_data = ir1[0];
              ir1[0] = (v616_data + (v612_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v622_data = ir1[1];
              ir1[1] = (v622_data + (v612_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v628_data = ir1[2];
              ir1[2] = (v628_data + (v612_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v634_data = ir1[3];
              ir1[3] = (v634_data + (v612_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v640_data = ir1[4];
              ir1[4] = (v640_data + (v612_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v646_data = ir1[5];
              ir1[5] = (v646_data + (v612_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v652_data = ir1[6];
              ir1[6] = (v652_data + (v612_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v658_data = ir1[7];
              ir1[7] = (v658_data + (v612_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v664_data = ir1[8];
              ir1[8] = (v664_data + (v612_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              int32_t v666_a = v19_lead + 110;
              float v667_data_pre = glb_m1[v61_g ? (v666_a) : (0)];
              float v667_data = v61_g ? (v667_data_pre) : (0.0f);
              float v671_data = ir1[0];
              ir1[0] = (v671_data + (v667_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v677_data = ir1[1];
              ir1[1] = (v677_data + (v667_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v683_data = ir1[2];
              ir1[2] = (v683_data + (v667_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v689_data = ir1[3];
              ir1[3] = (v689_data + (v667_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v695_data = ir1[4];
              ir1[4] = (v695_data + (v667_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v701_data = ir1[5];
              ir1[5] = (v701_data + (v667_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v707_data = ir1[6];
              ir1[6] = (v707_data + (v667_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v713_data = ir1[7];
              ir1[7] = (v713_data + (v667_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v719_data = ir1[8];
              ir1[8] = (v719_data + (v667_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              int32_t v721_a = v19_lead + 120;
              float v722_data_pre = glb_m1[v61_g ? (v721_a) : (0)];
              float v722_data = v61_g ? (v722_data_pre) : (0.0f);
              float v726_data = ir1[0];
              ir1[0] = (v726_data + (v722_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v732_data = ir1[1];
              ir1[1] = (v732_data + (v722_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v738_data = ir1[2];
              ir1[2] = (v738_data + (v722_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v744_data = ir1[3];
              ir1[3] = (v744_data + (v722_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v750_data = ir1[4];
              ir1[4] = (v750_data + (v722_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v756_data = ir1[5];
              ir1[5] = (v756_data + (v722_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v762_data = ir1[6];
              ir1[6] = (v762_data + (v722_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v768_data = ir1[7];
              ir1[7] = (v768_data + (v722_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v774_data = ir1[8];
              ir1[8] = (v774_data + (v722_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              int32_t v776_a = v19_lead + 130;
              float v777_data_pre = glb_m1[v61_g ? (v776_a) : (0)];
              float v777_data = v61_g ? (v777_data_pre) : (0.0f);
              float v781_data = ir1[0];
              ir1[0] = (v781_data + (v777_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v787_data = ir1[1];
              ir1[1] = (v787_data + (v777_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v793_data = ir1[2];
              ir1[2] = (v793_data + (v777_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v799_data = ir1[3];
              ir1[3] = (v799_data + (v777_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v805_data = ir1[4];
              ir1[4] = (v805_data + (v777_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v811_data = ir1[5];
              ir1[5] = (v811_data + (v777_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v817_data = ir1[6];
              ir1[6] = (v817_data + (v777_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v823_data = ir1[7];
              ir1[7] = (v823_data + (v777_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v829_data = ir1[8];
              ir1[8] = (v829_data + (v777_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              int32_t v831_a = v19_lead + 140;
              float v832_data_pre = glb_m1[v61_g ? (v831_a) : (0)];
              float v832_data = v61_g ? (v832_data_pre) : (0.0f);
              float v836_data = ir1[0];
              ir1[0] = (v836_data + (v832_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v842_data = ir1[1];
              ir1[1] = (v842_data + (v832_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v848_data = ir1[2];
              ir1[2] = (v848_data + (v832_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v854_data = ir1[3];
              ir1[3] = (v854_data + (v832_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v860_data = ir1[4];
              ir1[4] = (v860_data + (v832_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v866_data = ir1[5];
              ir1[5] = (v866_data + (v832_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v872_data = ir1[6];
              ir1[6] = (v872_data + (v832_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v878_data = ir1[7];
              ir1[7] = (v878_data + (v832_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v884_data = ir1[8];
              ir1[8] = (v884_data + (v832_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              int32_t v886_a = v19_lead + 150;
              float v887_data_pre = glb_m1[v61_g ? (v886_a) : (0)];
              float v887_data = v61_g ? (v887_data_pre) : (0.0f);
              float v891_data = ir1[0];
              ir1[0] = (v891_data + (v887_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v897_data = ir1[1];
              ir1[1] = (v897_data + (v887_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v903_data = ir1[2];
              ir1[2] = (v903_data + (v887_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v909_data = ir1[3];
              ir1[3] = (v909_data + (v887_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v915_data = ir1[4];
              ir1[4] = (v915_data + (v887_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v921_data = ir1[5];
              ir1[5] = (v921_data + (v887_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v927_data = ir1[6];
              ir1[6] = (v927_data + (v887_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v933_data = ir1[7];
              ir1[7] = (v933_data + (v887_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v939_data = ir1[8];
              ir1[8] = (v939_data + (v887_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              int32_t v941_a = v19_lead + 160;
              float v942_data_pre = glb_m1[v61_g ? (v941_a) : (0)];
              float v942_data = v61_g ? (v942_data_pre) : (0.0f);
              float v943_data = r0[1];
              float v946_data = ir1[0];
              ir1[0] = (v946_data + (v942_data * (sycl::select_from_group(item.get_sub_group(), v943_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v949_data = r0[3];
              float v952_data = ir1[1];
              ir1[1] = (v952_data + (v942_data * (sycl::select_from_group(item.get_sub_group(), v949_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v955_data = r0[5];
              float v958_data = ir1[2];
              ir1[2] = (v958_data + (v942_data * (sycl::select_from_group(item.get_sub_group(), v955_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v961_data = r0[7];
              float v964_data = ir1[3];
              ir1[3] = (v964_data + (v942_data * (sycl::select_from_group(item.get_sub_group(), v961_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v967_data = r0[9];
              float v970_data = ir1[4];
              ir1[4] = (v970_data + (v942_data * (sycl::select_from_group(item.get_sub_group(), v967_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v973_data = r0[11];
              float v976_data = ir1[5];
              ir1[5] = (v976_data + (v942_data * (sycl::select_from_group(item.get_sub_group(), v973_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v979_data = r0[13];
              float v982_data = ir1[6];
              ir1[6] = (v982_data + (v942_data * (sycl::select_from_group(item.get_sub_group(), v979_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v985_data = r0[15];
              float v988_data = ir1[7];
              ir1[7] = (v988_data + (v942_data * (sycl::select_from_group(item.get_sub_group(), v985_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v991_data = r0[17];
              float v994_data = ir1[8];
              ir1[8] = (v994_data + (v942_data * (sycl::select_from_group(item.get_sub_group(), v991_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              // r1 = ir1
              if (v61_g) {
                #pragma unroll
                for (int32_t v997_n1 = 0; v997_n1 < 9; ++v997_n1) {
                  float v999_data = ir1[v997_n1];
                  r1[v997_n1] = v999_data;
                }
              }
              // wait(r2 = load{g>r}(glb_m4););
              float r3[9]{};
              // ir3 = +(glb_m3 * r2)
              // [(0, 10), (0, 9)] [(0, 17)]
              float ir3[9]{};
              float v1006_data_pre = glb_m3[v61_g ? (v19_lead) : (0)];
              float v1006_data = v61_g ? (v1006_data_pre) : (0.0f);
              float v1007_data = r2[0];
              float v1010_data = ir3[0];
              ir3[0] = (v1010_data + (v1006_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1013_data = r2[2];
              float v1016_data = ir3[1];
              ir3[1] = (v1016_data + (v1006_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1019_data = r2[4];
              float v1022_data = ir3[2];
              ir3[2] = (v1022_data + (v1006_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1025_data = r2[6];
              float v1028_data = ir3[3];
              ir3[3] = (v1028_data + (v1006_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1031_data = r2[8];
              float v1034_data = ir3[4];
              ir3[4] = (v1034_data + (v1006_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1037_data = r2[10];
              float v1040_data = ir3[5];
              ir3[5] = (v1040_data + (v1006_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1043_data = r2[12];
              float v1046_data = ir3[6];
              ir3[6] = (v1046_data + (v1006_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1049_data = r2[14];
              float v1052_data = ir3[7];
              ir3[7] = (v1052_data + (v1006_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1055_data = r2[16];
              float v1058_data = ir3[8];
              ir3[8] = (v1058_data + (v1006_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1061_data_pre = glb_m3[v61_g ? (v116_a) : (0)];
              float v1061_data = v61_g ? (v1061_data_pre) : (0.0f);
              float v1065_data = ir3[0];
              ir3[0] = (v1065_data + (v1061_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1071_data = ir3[1];
              ir3[1] = (v1071_data + (v1061_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1077_data = ir3[2];
              ir3[2] = (v1077_data + (v1061_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1083_data = ir3[3];
              ir3[3] = (v1083_data + (v1061_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1089_data = ir3[4];
              ir3[4] = (v1089_data + (v1061_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1095_data = ir3[5];
              ir3[5] = (v1095_data + (v1061_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1101_data = ir3[6];
              ir3[6] = (v1101_data + (v1061_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1107_data = ir3[7];
              ir3[7] = (v1107_data + (v1061_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1113_data = ir3[8];
              ir3[8] = (v1113_data + (v1061_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1116_data_pre = glb_m3[v61_g ? (v171_a) : (0)];
              float v1116_data = v61_g ? (v1116_data_pre) : (0.0f);
              float v1120_data = ir3[0];
              ir3[0] = (v1120_data + (v1116_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1126_data = ir3[1];
              ir3[1] = (v1126_data + (v1116_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1132_data = ir3[2];
              ir3[2] = (v1132_data + (v1116_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1138_data = ir3[3];
              ir3[3] = (v1138_data + (v1116_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1144_data = ir3[4];
              ir3[4] = (v1144_data + (v1116_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1150_data = ir3[5];
              ir3[5] = (v1150_data + (v1116_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1156_data = ir3[6];
              ir3[6] = (v1156_data + (v1116_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1162_data = ir3[7];
              ir3[7] = (v1162_data + (v1116_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1168_data = ir3[8];
              ir3[8] = (v1168_data + (v1116_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1171_data_pre = glb_m3[v61_g ? (v226_a) : (0)];
              float v1171_data = v61_g ? (v1171_data_pre) : (0.0f);
              float v1175_data = ir3[0];
              ir3[0] = (v1175_data + (v1171_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1181_data = ir3[1];
              ir3[1] = (v1181_data + (v1171_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1187_data = ir3[2];
              ir3[2] = (v1187_data + (v1171_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1193_data = ir3[3];
              ir3[3] = (v1193_data + (v1171_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1199_data = ir3[4];
              ir3[4] = (v1199_data + (v1171_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1205_data = ir3[5];
              ir3[5] = (v1205_data + (v1171_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1211_data = ir3[6];
              ir3[6] = (v1211_data + (v1171_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1217_data = ir3[7];
              ir3[7] = (v1217_data + (v1171_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1223_data = ir3[8];
              ir3[8] = (v1223_data + (v1171_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1226_data_pre = glb_m3[v61_g ? (v281_a) : (0)];
              float v1226_data = v61_g ? (v1226_data_pre) : (0.0f);
              float v1230_data = ir3[0];
              ir3[0] = (v1230_data + (v1226_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1236_data = ir3[1];
              ir3[1] = (v1236_data + (v1226_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1242_data = ir3[2];
              ir3[2] = (v1242_data + (v1226_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1248_data = ir3[3];
              ir3[3] = (v1248_data + (v1226_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1254_data = ir3[4];
              ir3[4] = (v1254_data + (v1226_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1260_data = ir3[5];
              ir3[5] = (v1260_data + (v1226_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1266_data = ir3[6];
              ir3[6] = (v1266_data + (v1226_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1272_data = ir3[7];
              ir3[7] = (v1272_data + (v1226_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1278_data = ir3[8];
              ir3[8] = (v1278_data + (v1226_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1281_data_pre = glb_m3[v61_g ? (v336_a) : (0)];
              float v1281_data = v61_g ? (v1281_data_pre) : (0.0f);
              float v1285_data = ir3[0];
              ir3[0] = (v1285_data + (v1281_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1291_data = ir3[1];
              ir3[1] = (v1291_data + (v1281_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1297_data = ir3[2];
              ir3[2] = (v1297_data + (v1281_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1303_data = ir3[3];
              ir3[3] = (v1303_data + (v1281_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1309_data = ir3[4];
              ir3[4] = (v1309_data + (v1281_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1315_data = ir3[5];
              ir3[5] = (v1315_data + (v1281_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1321_data = ir3[6];
              ir3[6] = (v1321_data + (v1281_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1327_data = ir3[7];
              ir3[7] = (v1327_data + (v1281_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1333_data = ir3[8];
              ir3[8] = (v1333_data + (v1281_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1336_data_pre = glb_m3[v61_g ? (v391_a) : (0)];
              float v1336_data = v61_g ? (v1336_data_pre) : (0.0f);
              float v1340_data = ir3[0];
              ir3[0] = (v1340_data + (v1336_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1346_data = ir3[1];
              ir3[1] = (v1346_data + (v1336_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1352_data = ir3[2];
              ir3[2] = (v1352_data + (v1336_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1358_data = ir3[3];
              ir3[3] = (v1358_data + (v1336_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1364_data = ir3[4];
              ir3[4] = (v1364_data + (v1336_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1370_data = ir3[5];
              ir3[5] = (v1370_data + (v1336_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1376_data = ir3[6];
              ir3[6] = (v1376_data + (v1336_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1382_data = ir3[7];
              ir3[7] = (v1382_data + (v1336_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1388_data = ir3[8];
              ir3[8] = (v1388_data + (v1336_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1391_data_pre = glb_m3[v61_g ? (v446_a) : (0)];
              float v1391_data = v61_g ? (v1391_data_pre) : (0.0f);
              float v1395_data = ir3[0];
              ir3[0] = (v1395_data + (v1391_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1401_data = ir3[1];
              ir3[1] = (v1401_data + (v1391_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1407_data = ir3[2];
              ir3[2] = (v1407_data + (v1391_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1413_data = ir3[3];
              ir3[3] = (v1413_data + (v1391_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1419_data = ir3[4];
              ir3[4] = (v1419_data + (v1391_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1425_data = ir3[5];
              ir3[5] = (v1425_data + (v1391_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1431_data = ir3[6];
              ir3[6] = (v1431_data + (v1391_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1437_data = ir3[7];
              ir3[7] = (v1437_data + (v1391_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1443_data = ir3[8];
              ir3[8] = (v1443_data + (v1391_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1446_data_pre = glb_m3[v61_g ? (v501_a) : (0)];
              float v1446_data = v61_g ? (v1446_data_pre) : (0.0f);
              float v1450_data = ir3[0];
              ir3[0] = (v1450_data + (v1446_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1456_data = ir3[1];
              ir3[1] = (v1456_data + (v1446_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1462_data = ir3[2];
              ir3[2] = (v1462_data + (v1446_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1468_data = ir3[3];
              ir3[3] = (v1468_data + (v1446_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1474_data = ir3[4];
              ir3[4] = (v1474_data + (v1446_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1480_data = ir3[5];
              ir3[5] = (v1480_data + (v1446_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1486_data = ir3[6];
              ir3[6] = (v1486_data + (v1446_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1492_data = ir3[7];
              ir3[7] = (v1492_data + (v1446_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1498_data = ir3[8];
              ir3[8] = (v1498_data + (v1446_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1501_data_pre = glb_m3[v61_g ? (v556_a) : (0)];
              float v1501_data = v61_g ? (v1501_data_pre) : (0.0f);
              float v1505_data = ir3[0];
              ir3[0] = (v1505_data + (v1501_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1511_data = ir3[1];
              ir3[1] = (v1511_data + (v1501_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1517_data = ir3[2];
              ir3[2] = (v1517_data + (v1501_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1523_data = ir3[3];
              ir3[3] = (v1523_data + (v1501_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1529_data = ir3[4];
              ir3[4] = (v1529_data + (v1501_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1535_data = ir3[5];
              ir3[5] = (v1535_data + (v1501_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1541_data = ir3[6];
              ir3[6] = (v1541_data + (v1501_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1547_data = ir3[7];
              ir3[7] = (v1547_data + (v1501_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1553_data = ir3[8];
              ir3[8] = (v1553_data + (v1501_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1556_data_pre = glb_m3[v61_g ? (v611_a) : (0)];
              float v1556_data = v61_g ? (v1556_data_pre) : (0.0f);
              float v1560_data = ir3[0];
              ir3[0] = (v1560_data + (v1556_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1566_data = ir3[1];
              ir3[1] = (v1566_data + (v1556_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1572_data = ir3[2];
              ir3[2] = (v1572_data + (v1556_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1578_data = ir3[3];
              ir3[3] = (v1578_data + (v1556_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1584_data = ir3[4];
              ir3[4] = (v1584_data + (v1556_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1590_data = ir3[5];
              ir3[5] = (v1590_data + (v1556_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1596_data = ir3[6];
              ir3[6] = (v1596_data + (v1556_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1602_data = ir3[7];
              ir3[7] = (v1602_data + (v1556_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1608_data = ir3[8];
              ir3[8] = (v1608_data + (v1556_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1611_data_pre = glb_m3[v61_g ? (v666_a) : (0)];
              float v1611_data = v61_g ? (v1611_data_pre) : (0.0f);
              float v1615_data = ir3[0];
              ir3[0] = (v1615_data + (v1611_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1621_data = ir3[1];
              ir3[1] = (v1621_data + (v1611_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1627_data = ir3[2];
              ir3[2] = (v1627_data + (v1611_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1633_data = ir3[3];
              ir3[3] = (v1633_data + (v1611_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1639_data = ir3[4];
              ir3[4] = (v1639_data + (v1611_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1645_data = ir3[5];
              ir3[5] = (v1645_data + (v1611_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1651_data = ir3[6];
              ir3[6] = (v1651_data + (v1611_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1657_data = ir3[7];
              ir3[7] = (v1657_data + (v1611_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1663_data = ir3[8];
              ir3[8] = (v1663_data + (v1611_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1666_data_pre = glb_m3[v61_g ? (v721_a) : (0)];
              float v1666_data = v61_g ? (v1666_data_pre) : (0.0f);
              float v1670_data = ir3[0];
              ir3[0] = (v1670_data + (v1666_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1676_data = ir3[1];
              ir3[1] = (v1676_data + (v1666_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1682_data = ir3[2];
              ir3[2] = (v1682_data + (v1666_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1688_data = ir3[3];
              ir3[3] = (v1688_data + (v1666_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1694_data = ir3[4];
              ir3[4] = (v1694_data + (v1666_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1700_data = ir3[5];
              ir3[5] = (v1700_data + (v1666_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1706_data = ir3[6];
              ir3[6] = (v1706_data + (v1666_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1712_data = ir3[7];
              ir3[7] = (v1712_data + (v1666_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1718_data = ir3[8];
              ir3[8] = (v1718_data + (v1666_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1721_data_pre = glb_m3[v61_g ? (v776_a) : (0)];
              float v1721_data = v61_g ? (v1721_data_pre) : (0.0f);
              float v1725_data = ir3[0];
              ir3[0] = (v1725_data + (v1721_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1731_data = ir3[1];
              ir3[1] = (v1731_data + (v1721_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1737_data = ir3[2];
              ir3[2] = (v1737_data + (v1721_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1743_data = ir3[3];
              ir3[3] = (v1743_data + (v1721_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1749_data = ir3[4];
              ir3[4] = (v1749_data + (v1721_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1755_data = ir3[5];
              ir3[5] = (v1755_data + (v1721_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1761_data = ir3[6];
              ir3[6] = (v1761_data + (v1721_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1767_data = ir3[7];
              ir3[7] = (v1767_data + (v1721_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1773_data = ir3[8];
              ir3[8] = (v1773_data + (v1721_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1776_data_pre = glb_m3[v61_g ? (v831_a) : (0)];
              float v1776_data = v61_g ? (v1776_data_pre) : (0.0f);
              float v1780_data = ir3[0];
              ir3[0] = (v1780_data + (v1776_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1786_data = ir3[1];
              ir3[1] = (v1786_data + (v1776_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1792_data = ir3[2];
              ir3[2] = (v1792_data + (v1776_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1798_data = ir3[3];
              ir3[3] = (v1798_data + (v1776_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1804_data = ir3[4];
              ir3[4] = (v1804_data + (v1776_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1810_data = ir3[5];
              ir3[5] = (v1810_data + (v1776_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1816_data = ir3[6];
              ir3[6] = (v1816_data + (v1776_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1822_data = ir3[7];
              ir3[7] = (v1822_data + (v1776_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1828_data = ir3[8];
              ir3[8] = (v1828_data + (v1776_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1831_data_pre = glb_m3[v61_g ? (v886_a) : (0)];
              float v1831_data = v61_g ? (v1831_data_pre) : (0.0f);
              float v1835_data = ir3[0];
              ir3[0] = (v1835_data + (v1831_data * (sycl::select_from_group(item.get_sub_group(), v1007_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1841_data = ir3[1];
              ir3[1] = (v1841_data + (v1831_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1847_data = ir3[2];
              ir3[2] = (v1847_data + (v1831_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1853_data = ir3[3];
              ir3[3] = (v1853_data + (v1831_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1859_data = ir3[4];
              ir3[4] = (v1859_data + (v1831_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1865_data = ir3[5];
              ir3[5] = (v1865_data + (v1831_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1871_data = ir3[6];
              ir3[6] = (v1871_data + (v1831_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1877_data = ir3[7];
              ir3[7] = (v1877_data + (v1831_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1883_data = ir3[8];
              ir3[8] = (v1883_data + (v1831_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1886_data_pre = glb_m3[v61_g ? (v941_a) : (0)];
              float v1886_data = v61_g ? (v1886_data_pre) : (0.0f);
              float v1887_data = r2[1];
              float v1890_data = ir3[0];
              ir3[0] = (v1890_data + (v1886_data * (sycl::select_from_group(item.get_sub_group(), v1887_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1893_data = r2[3];
              float v1896_data = ir3[1];
              ir3[1] = (v1896_data + (v1886_data * (sycl::select_from_group(item.get_sub_group(), v1893_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1899_data = r2[5];
              float v1902_data = ir3[2];
              ir3[2] = (v1902_data + (v1886_data * (sycl::select_from_group(item.get_sub_group(), v1899_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1905_data = r2[7];
              float v1908_data = ir3[3];
              ir3[3] = (v1908_data + (v1886_data * (sycl::select_from_group(item.get_sub_group(), v1905_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1911_data = r2[9];
              float v1914_data = ir3[4];
              ir3[4] = (v1914_data + (v1886_data * (sycl::select_from_group(item.get_sub_group(), v1911_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1917_data = r2[11];
              float v1920_data = ir3[5];
              ir3[5] = (v1920_data + (v1886_data * (sycl::select_from_group(item.get_sub_group(), v1917_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1923_data = r2[13];
              float v1926_data = ir3[6];
              ir3[6] = (v1926_data + (v1886_data * (sycl::select_from_group(item.get_sub_group(), v1923_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1929_data = r2[15];
              float v1932_data = ir3[7];
              ir3[7] = (v1932_data + (v1886_data * (sycl::select_from_group(item.get_sub_group(), v1929_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1935_data = r2[17];
              float v1938_data = ir3[8];
              ir3[8] = (v1938_data + (v1886_data * (sycl::select_from_group(item.get_sub_group(), v1935_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              // r3 = ir3 + r1
              if (v61_g) {
                #pragma unroll
                for (int32_t v1941_n1 = 0; v1941_n1 < 9; ++v1941_n1) {
                  float v1943_data = ir3[v1941_n1];
                  float v1944_data = r1[v1941_n1];
                  r3[v1941_n1] = (v1944_data + v1943_data);
                }
              }
              // glb_m0 = store{r>g}(r3);
              if (v61_g) {
                #pragma unroll
                for (int32_t v1947_i1 = 0; v1947_i1 < 9; ++v1947_i1) {
                  float v1949_data = r3[v1947_i1];
                  glb_m0[(v19_lead + (v1947_i1 * 10))] = v1949_data;
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

