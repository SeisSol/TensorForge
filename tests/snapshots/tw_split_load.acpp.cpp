// === base name ===
kernel_c2628bd16eacb480

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_c2628bd16eacb480 = {{16, 16, 1}, 16, 10, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_c2628bd16eacb480(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_c2628bd16eacb480(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_c2628bd16eacb480(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_c2628bd16eacb480(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_c2628bd16eacb480(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_c2628bd16eacb480(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, m3, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_c2628bd16eacb480(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
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
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":10,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[10,9]],"name":"m0","ordered":false,"parts":1,"shape":[10,9],"variant":false},{"addressing":"none","alias":"A1","bbox":[[0,0],[10,17]],"name":"m1","ordered":false,"parts":1,"shape":[10,17],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[17,9]],"name":"m2","ordered":false,"parts":1,"shape":[17,9],"variant":false},{"addressing":"none","alias":"A2","bbox":[[0,0],[10,17]],"name":"m3","ordered":false,"parts":1,"shape":[10,17],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[17,9]],"name":"m4","ordered":false,"parts":1,"shape":[17,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[10,17]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[10,17]},{"addressing":"strided","bbox":[[0,0],[17,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[17,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[10,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[10,9]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[10,17]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[10,17]},{"addressing":"strided","bbox":[[0,0],[17,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[17,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
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
              const float *const __restrict__ glb_m4 = &m4[v11_batchId0 * 153 + 0 + m4_extraOffset];
              float r0[18]{};
              // r0 = load{g>r}(glb_m2);
              int32_t v25_lead = item.get_local_id(2) % 16;
              #pragma unroll
              for (int32_t v26_i0 = 0; v26_i0 < 1; ++v26_i0) {
                int32_t v29_lead = v25_lead + (v26_i0 * 16);
                #pragma unroll
                for (int32_t v27_i1 = 0; v27_i1 < 9; ++v27_i1) {
                  float v32_data = glb_m2[(v29_lead + (v27_i1 * 17))];
                  r0[(v26_i0 + (v27_i1 * 2))] = v32_data;
                }
              }
              bool v35_g = v25_lead < 1;
              if (v35_g) {
                int32_t v38_lead = v25_lead + 16_i32;
                #pragma unroll
                for (int32_t v36_i1 = 0; v36_i1 < 9; ++v36_i1) {
                  float v41_data = glb_m2[(v38_lead + (v36_i1 * 17))];
                  r0[(1 + (v36_i1 * 2))] = v41_data;
                }
              }
              float r2[18]{};
              // r2 = load{g>r}(glb_m4);
              #pragma unroll
              for (int32_t v45_i0 = 0; v45_i0 < 1; ++v45_i0) {
                int32_t v48_lead = v25_lead + (v45_i0 * 16);
                #pragma unroll
                for (int32_t v46_i1 = 0; v46_i1 < 9; ++v46_i1) {
                  float v51_data = glb_m4[(v48_lead + (v46_i1 * 17))];
                  r2[(v45_i0 + (v46_i1 * 2))] = v51_data;
                }
              }
              if (v35_g) {
                int32_t v56_lead = v25_lead + 16_i32;
                #pragma unroll
                for (int32_t v54_i1 = 0; v54_i1 < 9; ++v54_i1) {
                  float v59_data = glb_m4[(v56_lead + (v54_i1 * 17))];
                  r2[(1 + (v54_i1 * 2))] = v59_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m2););
              float r1[9]{};
              // ir1 = +(glb_m1 * r0)
              // [(0, 10), (0, 9)] [(0, 17)]
              float ir1[9]{};
              bool v67_g = v25_lead < 10;
              float v68_data_pre = glb_m1[v67_g ? (v25_lead) : (0)];
              float v68_data = v67_g ? (v68_data_pre) : (0.0f);
              float v69_data = r0[0];
              float v72_data = ir1[0];
              ir1[0] = (v72_data + (v68_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v75_data = r0[2];
              float v78_data = ir1[1];
              ir1[1] = (v78_data + (v68_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v81_data = r0[4];
              float v84_data = ir1[2];
              ir1[2] = (v84_data + (v68_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v87_data = r0[6];
              float v90_data = ir1[3];
              ir1[3] = (v90_data + (v68_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v93_data = r0[8];
              float v96_data = ir1[4];
              ir1[4] = (v96_data + (v68_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v99_data = r0[10];
              float v102_data = ir1[5];
              ir1[5] = (v102_data + (v68_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v105_data = r0[12];
              float v108_data = ir1[6];
              ir1[6] = (v108_data + (v68_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v111_data = r0[14];
              float v114_data = ir1[7];
              ir1[7] = (v114_data + (v68_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v117_data = r0[16];
              float v120_data = ir1[8];
              ir1[8] = (v120_data + (v68_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              int32_t v122_a = v25_lead + 10;
              float v123_data_pre = glb_m1[v67_g ? (v122_a) : (0)];
              float v123_data = v67_g ? (v123_data_pre) : (0.0f);
              float v127_data = ir1[0];
              ir1[0] = (v127_data + (v123_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v133_data = ir1[1];
              ir1[1] = (v133_data + (v123_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v139_data = ir1[2];
              ir1[2] = (v139_data + (v123_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v145_data = ir1[3];
              ir1[3] = (v145_data + (v123_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v151_data = ir1[4];
              ir1[4] = (v151_data + (v123_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v157_data = ir1[5];
              ir1[5] = (v157_data + (v123_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v163_data = ir1[6];
              ir1[6] = (v163_data + (v123_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v169_data = ir1[7];
              ir1[7] = (v169_data + (v123_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v175_data = ir1[8];
              ir1[8] = (v175_data + (v123_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              int32_t v177_a = v25_lead + 20;
              float v178_data_pre = glb_m1[v67_g ? (v177_a) : (0)];
              float v178_data = v67_g ? (v178_data_pre) : (0.0f);
              float v182_data = ir1[0];
              ir1[0] = (v182_data + (v178_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v188_data = ir1[1];
              ir1[1] = (v188_data + (v178_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v194_data = ir1[2];
              ir1[2] = (v194_data + (v178_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v200_data = ir1[3];
              ir1[3] = (v200_data + (v178_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v206_data = ir1[4];
              ir1[4] = (v206_data + (v178_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v212_data = ir1[5];
              ir1[5] = (v212_data + (v178_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v218_data = ir1[6];
              ir1[6] = (v218_data + (v178_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v224_data = ir1[7];
              ir1[7] = (v224_data + (v178_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v230_data = ir1[8];
              ir1[8] = (v230_data + (v178_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              int32_t v232_a = v25_lead + 30;
              float v233_data_pre = glb_m1[v67_g ? (v232_a) : (0)];
              float v233_data = v67_g ? (v233_data_pre) : (0.0f);
              float v237_data = ir1[0];
              ir1[0] = (v237_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v243_data = ir1[1];
              ir1[1] = (v243_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v249_data = ir1[2];
              ir1[2] = (v249_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v255_data = ir1[3];
              ir1[3] = (v255_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v261_data = ir1[4];
              ir1[4] = (v261_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v267_data = ir1[5];
              ir1[5] = (v267_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v273_data = ir1[6];
              ir1[6] = (v273_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v279_data = ir1[7];
              ir1[7] = (v279_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v285_data = ir1[8];
              ir1[8] = (v285_data + (v233_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              int32_t v287_a = v25_lead + 40;
              float v288_data_pre = glb_m1[v67_g ? (v287_a) : (0)];
              float v288_data = v67_g ? (v288_data_pre) : (0.0f);
              float v292_data = ir1[0];
              ir1[0] = (v292_data + (v288_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v298_data = ir1[1];
              ir1[1] = (v298_data + (v288_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v304_data = ir1[2];
              ir1[2] = (v304_data + (v288_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v310_data = ir1[3];
              ir1[3] = (v310_data + (v288_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v316_data = ir1[4];
              ir1[4] = (v316_data + (v288_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v322_data = ir1[5];
              ir1[5] = (v322_data + (v288_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v328_data = ir1[6];
              ir1[6] = (v328_data + (v288_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v334_data = ir1[7];
              ir1[7] = (v334_data + (v288_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v340_data = ir1[8];
              ir1[8] = (v340_data + (v288_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              int32_t v342_a = v25_lead + 50;
              float v343_data_pre = glb_m1[v67_g ? (v342_a) : (0)];
              float v343_data = v67_g ? (v343_data_pre) : (0.0f);
              float v347_data = ir1[0];
              ir1[0] = (v347_data + (v343_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v353_data = ir1[1];
              ir1[1] = (v353_data + (v343_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v359_data = ir1[2];
              ir1[2] = (v359_data + (v343_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v365_data = ir1[3];
              ir1[3] = (v365_data + (v343_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v371_data = ir1[4];
              ir1[4] = (v371_data + (v343_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v377_data = ir1[5];
              ir1[5] = (v377_data + (v343_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v383_data = ir1[6];
              ir1[6] = (v383_data + (v343_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v389_data = ir1[7];
              ir1[7] = (v389_data + (v343_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v395_data = ir1[8];
              ir1[8] = (v395_data + (v343_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              int32_t v397_a = v25_lead + 60;
              float v398_data_pre = glb_m1[v67_g ? (v397_a) : (0)];
              float v398_data = v67_g ? (v398_data_pre) : (0.0f);
              float v402_data = ir1[0];
              ir1[0] = (v402_data + (v398_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v408_data = ir1[1];
              ir1[1] = (v408_data + (v398_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v414_data = ir1[2];
              ir1[2] = (v414_data + (v398_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v420_data = ir1[3];
              ir1[3] = (v420_data + (v398_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v426_data = ir1[4];
              ir1[4] = (v426_data + (v398_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v432_data = ir1[5];
              ir1[5] = (v432_data + (v398_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v438_data = ir1[6];
              ir1[6] = (v438_data + (v398_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v444_data = ir1[7];
              ir1[7] = (v444_data + (v398_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v450_data = ir1[8];
              ir1[8] = (v450_data + (v398_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              int32_t v452_a = v25_lead + 70;
              float v453_data_pre = glb_m1[v67_g ? (v452_a) : (0)];
              float v453_data = v67_g ? (v453_data_pre) : (0.0f);
              float v457_data = ir1[0];
              ir1[0] = (v457_data + (v453_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v463_data = ir1[1];
              ir1[1] = (v463_data + (v453_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v469_data = ir1[2];
              ir1[2] = (v469_data + (v453_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v475_data = ir1[3];
              ir1[3] = (v475_data + (v453_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v481_data = ir1[4];
              ir1[4] = (v481_data + (v453_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v487_data = ir1[5];
              ir1[5] = (v487_data + (v453_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v493_data = ir1[6];
              ir1[6] = (v493_data + (v453_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v499_data = ir1[7];
              ir1[7] = (v499_data + (v453_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v505_data = ir1[8];
              ir1[8] = (v505_data + (v453_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              int32_t v507_a = v25_lead + 80;
              float v508_data_pre = glb_m1[v67_g ? (v507_a) : (0)];
              float v508_data = v67_g ? (v508_data_pre) : (0.0f);
              float v512_data = ir1[0];
              ir1[0] = (v512_data + (v508_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v518_data = ir1[1];
              ir1[1] = (v518_data + (v508_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v524_data = ir1[2];
              ir1[2] = (v524_data + (v508_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v530_data = ir1[3];
              ir1[3] = (v530_data + (v508_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v536_data = ir1[4];
              ir1[4] = (v536_data + (v508_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v542_data = ir1[5];
              ir1[5] = (v542_data + (v508_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v548_data = ir1[6];
              ir1[6] = (v548_data + (v508_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v554_data = ir1[7];
              ir1[7] = (v554_data + (v508_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v560_data = ir1[8];
              ir1[8] = (v560_data + (v508_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              int32_t v562_a = v25_lead + 90;
              float v563_data_pre = glb_m1[v67_g ? (v562_a) : (0)];
              float v563_data = v67_g ? (v563_data_pre) : (0.0f);
              float v567_data = ir1[0];
              ir1[0] = (v567_data + (v563_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v573_data = ir1[1];
              ir1[1] = (v573_data + (v563_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v579_data = ir1[2];
              ir1[2] = (v579_data + (v563_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v585_data = ir1[3];
              ir1[3] = (v585_data + (v563_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v591_data = ir1[4];
              ir1[4] = (v591_data + (v563_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v597_data = ir1[5];
              ir1[5] = (v597_data + (v563_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v603_data = ir1[6];
              ir1[6] = (v603_data + (v563_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v609_data = ir1[7];
              ir1[7] = (v609_data + (v563_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v615_data = ir1[8];
              ir1[8] = (v615_data + (v563_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              int32_t v617_a = v25_lead + 100;
              float v618_data_pre = glb_m1[v67_g ? (v617_a) : (0)];
              float v618_data = v67_g ? (v618_data_pre) : (0.0f);
              float v622_data = ir1[0];
              ir1[0] = (v622_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v628_data = ir1[1];
              ir1[1] = (v628_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v634_data = ir1[2];
              ir1[2] = (v634_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v640_data = ir1[3];
              ir1[3] = (v640_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v646_data = ir1[4];
              ir1[4] = (v646_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v652_data = ir1[5];
              ir1[5] = (v652_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v658_data = ir1[6];
              ir1[6] = (v658_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v664_data = ir1[7];
              ir1[7] = (v664_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v670_data = ir1[8];
              ir1[8] = (v670_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              int32_t v672_a = v25_lead + 110;
              float v673_data_pre = glb_m1[v67_g ? (v672_a) : (0)];
              float v673_data = v67_g ? (v673_data_pre) : (0.0f);
              float v677_data = ir1[0];
              ir1[0] = (v677_data + (v673_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v683_data = ir1[1];
              ir1[1] = (v683_data + (v673_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v689_data = ir1[2];
              ir1[2] = (v689_data + (v673_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v695_data = ir1[3];
              ir1[3] = (v695_data + (v673_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v701_data = ir1[4];
              ir1[4] = (v701_data + (v673_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v707_data = ir1[5];
              ir1[5] = (v707_data + (v673_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v713_data = ir1[6];
              ir1[6] = (v713_data + (v673_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v719_data = ir1[7];
              ir1[7] = (v719_data + (v673_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v725_data = ir1[8];
              ir1[8] = (v725_data + (v673_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              int32_t v727_a = v25_lead + 120;
              float v728_data_pre = glb_m1[v67_g ? (v727_a) : (0)];
              float v728_data = v67_g ? (v728_data_pre) : (0.0f);
              float v732_data = ir1[0];
              ir1[0] = (v732_data + (v728_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v738_data = ir1[1];
              ir1[1] = (v738_data + (v728_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v744_data = ir1[2];
              ir1[2] = (v744_data + (v728_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v750_data = ir1[3];
              ir1[3] = (v750_data + (v728_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v756_data = ir1[4];
              ir1[4] = (v756_data + (v728_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v762_data = ir1[5];
              ir1[5] = (v762_data + (v728_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v768_data = ir1[6];
              ir1[6] = (v768_data + (v728_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v774_data = ir1[7];
              ir1[7] = (v774_data + (v728_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v780_data = ir1[8];
              ir1[8] = (v780_data + (v728_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              int32_t v782_a = v25_lead + 130;
              float v783_data_pre = glb_m1[v67_g ? (v782_a) : (0)];
              float v783_data = v67_g ? (v783_data_pre) : (0.0f);
              float v787_data = ir1[0];
              ir1[0] = (v787_data + (v783_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v793_data = ir1[1];
              ir1[1] = (v793_data + (v783_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v799_data = ir1[2];
              ir1[2] = (v799_data + (v783_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v805_data = ir1[3];
              ir1[3] = (v805_data + (v783_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v811_data = ir1[4];
              ir1[4] = (v811_data + (v783_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v817_data = ir1[5];
              ir1[5] = (v817_data + (v783_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v823_data = ir1[6];
              ir1[6] = (v823_data + (v783_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v829_data = ir1[7];
              ir1[7] = (v829_data + (v783_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v835_data = ir1[8];
              ir1[8] = (v835_data + (v783_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              int32_t v837_a = v25_lead + 140;
              float v838_data_pre = glb_m1[v67_g ? (v837_a) : (0)];
              float v838_data = v67_g ? (v838_data_pre) : (0.0f);
              float v842_data = ir1[0];
              ir1[0] = (v842_data + (v838_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v848_data = ir1[1];
              ir1[1] = (v848_data + (v838_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v854_data = ir1[2];
              ir1[2] = (v854_data + (v838_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v860_data = ir1[3];
              ir1[3] = (v860_data + (v838_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v866_data = ir1[4];
              ir1[4] = (v866_data + (v838_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v872_data = ir1[5];
              ir1[5] = (v872_data + (v838_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v878_data = ir1[6];
              ir1[6] = (v878_data + (v838_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v884_data = ir1[7];
              ir1[7] = (v884_data + (v838_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v890_data = ir1[8];
              ir1[8] = (v890_data + (v838_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              int32_t v892_a = v25_lead + 150;
              float v893_data_pre = glb_m1[v67_g ? (v892_a) : (0)];
              float v893_data = v67_g ? (v893_data_pre) : (0.0f);
              float v897_data = ir1[0];
              ir1[0] = (v897_data + (v893_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v903_data = ir1[1];
              ir1[1] = (v903_data + (v893_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v909_data = ir1[2];
              ir1[2] = (v909_data + (v893_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v915_data = ir1[3];
              ir1[3] = (v915_data + (v893_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v921_data = ir1[4];
              ir1[4] = (v921_data + (v893_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v927_data = ir1[5];
              ir1[5] = (v927_data + (v893_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v933_data = ir1[6];
              ir1[6] = (v933_data + (v893_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v939_data = ir1[7];
              ir1[7] = (v939_data + (v893_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v945_data = ir1[8];
              ir1[8] = (v945_data + (v893_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              int32_t v947_a = v25_lead + 160;
              float v948_data_pre = glb_m1[v67_g ? (v947_a) : (0)];
              float v948_data = v67_g ? (v948_data_pre) : (0.0f);
              float v949_data = r0[1];
              float v952_data = ir1[0];
              ir1[0] = (v952_data + (v948_data * (sycl::select_from_group(item.get_sub_group(), v949_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v955_data = r0[3];
              float v958_data = ir1[1];
              ir1[1] = (v958_data + (v948_data * (sycl::select_from_group(item.get_sub_group(), v955_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v961_data = r0[5];
              float v964_data = ir1[2];
              ir1[2] = (v964_data + (v948_data * (sycl::select_from_group(item.get_sub_group(), v961_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v967_data = r0[7];
              float v970_data = ir1[3];
              ir1[3] = (v970_data + (v948_data * (sycl::select_from_group(item.get_sub_group(), v967_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v973_data = r0[9];
              float v976_data = ir1[4];
              ir1[4] = (v976_data + (v948_data * (sycl::select_from_group(item.get_sub_group(), v973_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v979_data = r0[11];
              float v982_data = ir1[5];
              ir1[5] = (v982_data + (v948_data * (sycl::select_from_group(item.get_sub_group(), v979_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v985_data = r0[13];
              float v988_data = ir1[6];
              ir1[6] = (v988_data + (v948_data * (sycl::select_from_group(item.get_sub_group(), v985_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v991_data = r0[15];
              float v994_data = ir1[7];
              ir1[7] = (v994_data + (v948_data * (sycl::select_from_group(item.get_sub_group(), v991_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v997_data = r0[17];
              float v1000_data = ir1[8];
              ir1[8] = (v1000_data + (v948_data * (sycl::select_from_group(item.get_sub_group(), v997_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              // r1 = ir1
              if (v67_g) {
                #pragma unroll
                for (int32_t v1003_n1 = 0; v1003_n1 < 9; ++v1003_n1) {
                  float v1005_data = ir1[v1003_n1];
                  r1[v1003_n1] = v1005_data;
                }
              }
              // wait(r2 = load{g>r}(glb_m4););
              float r3[9]{};
              // ir3 = +(glb_m3 * r2)
              // [(0, 10), (0, 9)] [(0, 17)]
              float ir3[9]{};
              float v1012_data_pre = glb_m3[v67_g ? (v25_lead) : (0)];
              float v1012_data = v67_g ? (v1012_data_pre) : (0.0f);
              float v1013_data = r2[0];
              float v1016_data = ir3[0];
              ir3[0] = (v1016_data + (v1012_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1019_data = r2[2];
              float v1022_data = ir3[1];
              ir3[1] = (v1022_data + (v1012_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1025_data = r2[4];
              float v1028_data = ir3[2];
              ir3[2] = (v1028_data + (v1012_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1031_data = r2[6];
              float v1034_data = ir3[3];
              ir3[3] = (v1034_data + (v1012_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1037_data = r2[8];
              float v1040_data = ir3[4];
              ir3[4] = (v1040_data + (v1012_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1043_data = r2[10];
              float v1046_data = ir3[5];
              ir3[5] = (v1046_data + (v1012_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1049_data = r2[12];
              float v1052_data = ir3[6];
              ir3[6] = (v1052_data + (v1012_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1055_data = r2[14];
              float v1058_data = ir3[7];
              ir3[7] = (v1058_data + (v1012_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1061_data = r2[16];
              float v1064_data = ir3[8];
              ir3[8] = (v1064_data + (v1012_data * (sycl::select_from_group(item.get_sub_group(), v1061_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1067_data_pre = glb_m3[v67_g ? (v122_a) : (0)];
              float v1067_data = v67_g ? (v1067_data_pre) : (0.0f);
              float v1071_data = ir3[0];
              ir3[0] = (v1071_data + (v1067_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1077_data = ir3[1];
              ir3[1] = (v1077_data + (v1067_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1083_data = ir3[2];
              ir3[2] = (v1083_data + (v1067_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1089_data = ir3[3];
              ir3[3] = (v1089_data + (v1067_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1095_data = ir3[4];
              ir3[4] = (v1095_data + (v1067_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1101_data = ir3[5];
              ir3[5] = (v1101_data + (v1067_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1107_data = ir3[6];
              ir3[6] = (v1107_data + (v1067_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1113_data = ir3[7];
              ir3[7] = (v1113_data + (v1067_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1119_data = ir3[8];
              ir3[8] = (v1119_data + (v1067_data * (sycl::select_from_group(item.get_sub_group(), v1061_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1122_data_pre = glb_m3[v67_g ? (v177_a) : (0)];
              float v1122_data = v67_g ? (v1122_data_pre) : (0.0f);
              float v1126_data = ir3[0];
              ir3[0] = (v1126_data + (v1122_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1132_data = ir3[1];
              ir3[1] = (v1132_data + (v1122_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1138_data = ir3[2];
              ir3[2] = (v1138_data + (v1122_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1144_data = ir3[3];
              ir3[3] = (v1144_data + (v1122_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1150_data = ir3[4];
              ir3[4] = (v1150_data + (v1122_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1156_data = ir3[5];
              ir3[5] = (v1156_data + (v1122_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1162_data = ir3[6];
              ir3[6] = (v1162_data + (v1122_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1168_data = ir3[7];
              ir3[7] = (v1168_data + (v1122_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1174_data = ir3[8];
              ir3[8] = (v1174_data + (v1122_data * (sycl::select_from_group(item.get_sub_group(), v1061_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1177_data_pre = glb_m3[v67_g ? (v232_a) : (0)];
              float v1177_data = v67_g ? (v1177_data_pre) : (0.0f);
              float v1181_data = ir3[0];
              ir3[0] = (v1181_data + (v1177_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1187_data = ir3[1];
              ir3[1] = (v1187_data + (v1177_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1193_data = ir3[2];
              ir3[2] = (v1193_data + (v1177_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1199_data = ir3[3];
              ir3[3] = (v1199_data + (v1177_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1205_data = ir3[4];
              ir3[4] = (v1205_data + (v1177_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1211_data = ir3[5];
              ir3[5] = (v1211_data + (v1177_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1217_data = ir3[6];
              ir3[6] = (v1217_data + (v1177_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1223_data = ir3[7];
              ir3[7] = (v1223_data + (v1177_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1229_data = ir3[8];
              ir3[8] = (v1229_data + (v1177_data * (sycl::select_from_group(item.get_sub_group(), v1061_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1232_data_pre = glb_m3[v67_g ? (v287_a) : (0)];
              float v1232_data = v67_g ? (v1232_data_pre) : (0.0f);
              float v1236_data = ir3[0];
              ir3[0] = (v1236_data + (v1232_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1242_data = ir3[1];
              ir3[1] = (v1242_data + (v1232_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1248_data = ir3[2];
              ir3[2] = (v1248_data + (v1232_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1254_data = ir3[3];
              ir3[3] = (v1254_data + (v1232_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1260_data = ir3[4];
              ir3[4] = (v1260_data + (v1232_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1266_data = ir3[5];
              ir3[5] = (v1266_data + (v1232_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1272_data = ir3[6];
              ir3[6] = (v1272_data + (v1232_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1278_data = ir3[7];
              ir3[7] = (v1278_data + (v1232_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1284_data = ir3[8];
              ir3[8] = (v1284_data + (v1232_data * (sycl::select_from_group(item.get_sub_group(), v1061_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1287_data_pre = glb_m3[v67_g ? (v342_a) : (0)];
              float v1287_data = v67_g ? (v1287_data_pre) : (0.0f);
              float v1291_data = ir3[0];
              ir3[0] = (v1291_data + (v1287_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1297_data = ir3[1];
              ir3[1] = (v1297_data + (v1287_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1303_data = ir3[2];
              ir3[2] = (v1303_data + (v1287_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1309_data = ir3[3];
              ir3[3] = (v1309_data + (v1287_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1315_data = ir3[4];
              ir3[4] = (v1315_data + (v1287_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1321_data = ir3[5];
              ir3[5] = (v1321_data + (v1287_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1327_data = ir3[6];
              ir3[6] = (v1327_data + (v1287_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1333_data = ir3[7];
              ir3[7] = (v1333_data + (v1287_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1339_data = ir3[8];
              ir3[8] = (v1339_data + (v1287_data * (sycl::select_from_group(item.get_sub_group(), v1061_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1342_data_pre = glb_m3[v67_g ? (v397_a) : (0)];
              float v1342_data = v67_g ? (v1342_data_pre) : (0.0f);
              float v1346_data = ir3[0];
              ir3[0] = (v1346_data + (v1342_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1352_data = ir3[1];
              ir3[1] = (v1352_data + (v1342_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1358_data = ir3[2];
              ir3[2] = (v1358_data + (v1342_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1364_data = ir3[3];
              ir3[3] = (v1364_data + (v1342_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1370_data = ir3[4];
              ir3[4] = (v1370_data + (v1342_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1376_data = ir3[5];
              ir3[5] = (v1376_data + (v1342_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1382_data = ir3[6];
              ir3[6] = (v1382_data + (v1342_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1388_data = ir3[7];
              ir3[7] = (v1388_data + (v1342_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1394_data = ir3[8];
              ir3[8] = (v1394_data + (v1342_data * (sycl::select_from_group(item.get_sub_group(), v1061_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1397_data_pre = glb_m3[v67_g ? (v452_a) : (0)];
              float v1397_data = v67_g ? (v1397_data_pre) : (0.0f);
              float v1401_data = ir3[0];
              ir3[0] = (v1401_data + (v1397_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1407_data = ir3[1];
              ir3[1] = (v1407_data + (v1397_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1413_data = ir3[2];
              ir3[2] = (v1413_data + (v1397_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1419_data = ir3[3];
              ir3[3] = (v1419_data + (v1397_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1425_data = ir3[4];
              ir3[4] = (v1425_data + (v1397_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1431_data = ir3[5];
              ir3[5] = (v1431_data + (v1397_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1437_data = ir3[6];
              ir3[6] = (v1437_data + (v1397_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1443_data = ir3[7];
              ir3[7] = (v1443_data + (v1397_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1449_data = ir3[8];
              ir3[8] = (v1449_data + (v1397_data * (sycl::select_from_group(item.get_sub_group(), v1061_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1452_data_pre = glb_m3[v67_g ? (v507_a) : (0)];
              float v1452_data = v67_g ? (v1452_data_pre) : (0.0f);
              float v1456_data = ir3[0];
              ir3[0] = (v1456_data + (v1452_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1462_data = ir3[1];
              ir3[1] = (v1462_data + (v1452_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1468_data = ir3[2];
              ir3[2] = (v1468_data + (v1452_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1474_data = ir3[3];
              ir3[3] = (v1474_data + (v1452_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1480_data = ir3[4];
              ir3[4] = (v1480_data + (v1452_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1486_data = ir3[5];
              ir3[5] = (v1486_data + (v1452_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1492_data = ir3[6];
              ir3[6] = (v1492_data + (v1452_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1498_data = ir3[7];
              ir3[7] = (v1498_data + (v1452_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1504_data = ir3[8];
              ir3[8] = (v1504_data + (v1452_data * (sycl::select_from_group(item.get_sub_group(), v1061_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1507_data_pre = glb_m3[v67_g ? (v562_a) : (0)];
              float v1507_data = v67_g ? (v1507_data_pre) : (0.0f);
              float v1511_data = ir3[0];
              ir3[0] = (v1511_data + (v1507_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1517_data = ir3[1];
              ir3[1] = (v1517_data + (v1507_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1523_data = ir3[2];
              ir3[2] = (v1523_data + (v1507_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1529_data = ir3[3];
              ir3[3] = (v1529_data + (v1507_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1535_data = ir3[4];
              ir3[4] = (v1535_data + (v1507_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1541_data = ir3[5];
              ir3[5] = (v1541_data + (v1507_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1547_data = ir3[6];
              ir3[6] = (v1547_data + (v1507_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1553_data = ir3[7];
              ir3[7] = (v1553_data + (v1507_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1559_data = ir3[8];
              ir3[8] = (v1559_data + (v1507_data * (sycl::select_from_group(item.get_sub_group(), v1061_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1562_data_pre = glb_m3[v67_g ? (v617_a) : (0)];
              float v1562_data = v67_g ? (v1562_data_pre) : (0.0f);
              float v1566_data = ir3[0];
              ir3[0] = (v1566_data + (v1562_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1572_data = ir3[1];
              ir3[1] = (v1572_data + (v1562_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1578_data = ir3[2];
              ir3[2] = (v1578_data + (v1562_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1584_data = ir3[3];
              ir3[3] = (v1584_data + (v1562_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1590_data = ir3[4];
              ir3[4] = (v1590_data + (v1562_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1596_data = ir3[5];
              ir3[5] = (v1596_data + (v1562_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1602_data = ir3[6];
              ir3[6] = (v1602_data + (v1562_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1608_data = ir3[7];
              ir3[7] = (v1608_data + (v1562_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1614_data = ir3[8];
              ir3[8] = (v1614_data + (v1562_data * (sycl::select_from_group(item.get_sub_group(), v1061_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1617_data_pre = glb_m3[v67_g ? (v672_a) : (0)];
              float v1617_data = v67_g ? (v1617_data_pre) : (0.0f);
              float v1621_data = ir3[0];
              ir3[0] = (v1621_data + (v1617_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1627_data = ir3[1];
              ir3[1] = (v1627_data + (v1617_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1633_data = ir3[2];
              ir3[2] = (v1633_data + (v1617_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1639_data = ir3[3];
              ir3[3] = (v1639_data + (v1617_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1645_data = ir3[4];
              ir3[4] = (v1645_data + (v1617_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1651_data = ir3[5];
              ir3[5] = (v1651_data + (v1617_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1657_data = ir3[6];
              ir3[6] = (v1657_data + (v1617_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1663_data = ir3[7];
              ir3[7] = (v1663_data + (v1617_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1669_data = ir3[8];
              ir3[8] = (v1669_data + (v1617_data * (sycl::select_from_group(item.get_sub_group(), v1061_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1672_data_pre = glb_m3[v67_g ? (v727_a) : (0)];
              float v1672_data = v67_g ? (v1672_data_pre) : (0.0f);
              float v1676_data = ir3[0];
              ir3[0] = (v1676_data + (v1672_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1682_data = ir3[1];
              ir3[1] = (v1682_data + (v1672_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1688_data = ir3[2];
              ir3[2] = (v1688_data + (v1672_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1694_data = ir3[3];
              ir3[3] = (v1694_data + (v1672_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1700_data = ir3[4];
              ir3[4] = (v1700_data + (v1672_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1706_data = ir3[5];
              ir3[5] = (v1706_data + (v1672_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1712_data = ir3[6];
              ir3[6] = (v1712_data + (v1672_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1718_data = ir3[7];
              ir3[7] = (v1718_data + (v1672_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1724_data = ir3[8];
              ir3[8] = (v1724_data + (v1672_data * (sycl::select_from_group(item.get_sub_group(), v1061_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1727_data_pre = glb_m3[v67_g ? (v782_a) : (0)];
              float v1727_data = v67_g ? (v1727_data_pre) : (0.0f);
              float v1731_data = ir3[0];
              ir3[0] = (v1731_data + (v1727_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1737_data = ir3[1];
              ir3[1] = (v1737_data + (v1727_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1743_data = ir3[2];
              ir3[2] = (v1743_data + (v1727_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1749_data = ir3[3];
              ir3[3] = (v1749_data + (v1727_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1755_data = ir3[4];
              ir3[4] = (v1755_data + (v1727_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1761_data = ir3[5];
              ir3[5] = (v1761_data + (v1727_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1767_data = ir3[6];
              ir3[6] = (v1767_data + (v1727_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1773_data = ir3[7];
              ir3[7] = (v1773_data + (v1727_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1779_data = ir3[8];
              ir3[8] = (v1779_data + (v1727_data * (sycl::select_from_group(item.get_sub_group(), v1061_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1782_data_pre = glb_m3[v67_g ? (v837_a) : (0)];
              float v1782_data = v67_g ? (v1782_data_pre) : (0.0f);
              float v1786_data = ir3[0];
              ir3[0] = (v1786_data + (v1782_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1792_data = ir3[1];
              ir3[1] = (v1792_data + (v1782_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1798_data = ir3[2];
              ir3[2] = (v1798_data + (v1782_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1804_data = ir3[3];
              ir3[3] = (v1804_data + (v1782_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1810_data = ir3[4];
              ir3[4] = (v1810_data + (v1782_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1816_data = ir3[5];
              ir3[5] = (v1816_data + (v1782_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1822_data = ir3[6];
              ir3[6] = (v1822_data + (v1782_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1828_data = ir3[7];
              ir3[7] = (v1828_data + (v1782_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1834_data = ir3[8];
              ir3[8] = (v1834_data + (v1782_data * (sycl::select_from_group(item.get_sub_group(), v1061_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1837_data_pre = glb_m3[v67_g ? (v892_a) : (0)];
              float v1837_data = v67_g ? (v1837_data_pre) : (0.0f);
              float v1841_data = ir3[0];
              ir3[0] = (v1841_data + (v1837_data * (sycl::select_from_group(item.get_sub_group(), v1013_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1847_data = ir3[1];
              ir3[1] = (v1847_data + (v1837_data * (sycl::select_from_group(item.get_sub_group(), v1019_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1853_data = ir3[2];
              ir3[2] = (v1853_data + (v1837_data * (sycl::select_from_group(item.get_sub_group(), v1025_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1859_data = ir3[3];
              ir3[3] = (v1859_data + (v1837_data * (sycl::select_from_group(item.get_sub_group(), v1031_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1865_data = ir3[4];
              ir3[4] = (v1865_data + (v1837_data * (sycl::select_from_group(item.get_sub_group(), v1037_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1871_data = ir3[5];
              ir3[5] = (v1871_data + (v1837_data * (sycl::select_from_group(item.get_sub_group(), v1043_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1877_data = ir3[6];
              ir3[6] = (v1877_data + (v1837_data * (sycl::select_from_group(item.get_sub_group(), v1049_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1883_data = ir3[7];
              ir3[7] = (v1883_data + (v1837_data * (sycl::select_from_group(item.get_sub_group(), v1055_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1889_data = ir3[8];
              ir3[8] = (v1889_data + (v1837_data * (sycl::select_from_group(item.get_sub_group(), v1061_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1892_data_pre = glb_m3[v67_g ? (v947_a) : (0)];
              float v1892_data = v67_g ? (v1892_data_pre) : (0.0f);
              float v1893_data = r2[1];
              float v1896_data = ir3[0];
              ir3[0] = (v1896_data + (v1892_data * (sycl::select_from_group(item.get_sub_group(), v1893_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1899_data = r2[3];
              float v1902_data = ir3[1];
              ir3[1] = (v1902_data + (v1892_data * (sycl::select_from_group(item.get_sub_group(), v1899_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1905_data = r2[5];
              float v1908_data = ir3[2];
              ir3[2] = (v1908_data + (v1892_data * (sycl::select_from_group(item.get_sub_group(), v1905_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1911_data = r2[7];
              float v1914_data = ir3[3];
              ir3[3] = (v1914_data + (v1892_data * (sycl::select_from_group(item.get_sub_group(), v1911_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1917_data = r2[9];
              float v1920_data = ir3[4];
              ir3[4] = (v1920_data + (v1892_data * (sycl::select_from_group(item.get_sub_group(), v1917_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1923_data = r2[11];
              float v1926_data = ir3[5];
              ir3[5] = (v1926_data + (v1892_data * (sycl::select_from_group(item.get_sub_group(), v1923_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1929_data = r2[13];
              float v1932_data = ir3[6];
              ir3[6] = (v1932_data + (v1892_data * (sycl::select_from_group(item.get_sub_group(), v1929_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1935_data = r2[15];
              float v1938_data = ir3[7];
              ir3[7] = (v1938_data + (v1892_data * (sycl::select_from_group(item.get_sub_group(), v1935_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1941_data = r2[17];
              float v1944_data = ir3[8];
              ir3[8] = (v1944_data + (v1892_data * (sycl::select_from_group(item.get_sub_group(), v1941_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              // r3 = ir3 + r1
              if (v67_g) {
                #pragma unroll
                for (int32_t v1947_n1 = 0; v1947_n1 < 9; ++v1947_n1) {
                  float v1949_data = ir3[v1947_n1];
                  float v1950_data = r1[v1947_n1];
                  r3[v1947_n1] = (v1950_data + v1949_data);
                }
              }
              // glb_m0 = store{r>g}(r3);
              if (v67_g) {
                #pragma unroll
                for (int32_t v1953_i1 = 0; v1953_i1 < 9; ++v1953_i1) {
                  float v1955_data = r3[v1953_i1];
                  glb_m0[(v25_lead + (v1953_i1 * 10))] = v1955_data;
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

