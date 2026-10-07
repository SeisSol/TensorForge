// === base name ===
kernel_0d1299bb75d24e68

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_0d1299bb75d24e68 = {{16, 16, 1}, 16, 10, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_0d1299bb75d24e68(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_0d1299bb75d24e68(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_0d1299bb75d24e68(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_0d1299bb75d24e68(float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_0d1299bb75d24e68(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_0d1299bb75d24e68(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, m3, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_0d1299bb75d24e68(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, const float * m2, size_t m2_extraOffset, const float * m3, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
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
          const float *const __restrict__ glb_m1 = &m1[0];
          const float *const __restrict__ glb_m3 = &m3[0];
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 90 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v9_batchId0 * 153 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v9_batchId0 * 153 + 0 + m4_extraOffset];
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
              bool v33_g = v23_lead < 1;
              if (v33_g) {
                int32_t v36_lead = v23_lead + 16_i32;
                #pragma unroll
                for (int32_t v34_i1 = 0; v34_i1 < 9; ++v34_i1) {
                  float v39_data = glb_m2[(v36_lead + (v34_i1 * 17))];
                  r0[(1 + (v34_i1 * 2))] = v39_data;
                }
              }
              float r2[18]{};
              // r2 = load{g>r}(glb_m4);
              #pragma unroll
              for (int32_t v987_i0 = 0; v987_i0 < 1; ++v987_i0) {
                int32_t v990_lead = v23_lead + (v987_i0 * 16);
                #pragma unroll
                for (int32_t v988_i1 = 0; v988_i1 < 9; ++v988_i1) {
                  float v993_data = glb_m4[(v990_lead + (v988_i1 * 17))];
                  r2[(v987_i0 + (v988_i1 * 2))] = v993_data;
                }
              }
              if (v33_g) {
                int32_t v998_lead = v23_lead + 16_i32;
                #pragma unroll
                for (int32_t v996_i1 = 0; v996_i1 < 9; ++v996_i1) {
                  float v1001_data = glb_m4[(v998_lead + (v996_i1 * 17))];
                  r2[(1 + (v996_i1 * 2))] = v1001_data;
                }
              }
              float r1[9]{};
              // ir1 = +(glb_m1 * r0)
              // [(0, 10), (0, 9)] [(0, 17)]
              float ir1[9]{};
              bool v47_g = v23_lead < 10;
              float v48_data_pre = glb_m1[v47_g ? (v23_lead) : (0)];
              float v48_data = v47_g ? (v48_data_pre) : (0.0f);
              float v49_data = r0[0];
              float v52_data = ir1[0];
              ir1[0] = (v52_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v55_data = r0[2];
              float v58_data = ir1[1];
              ir1[1] = (v58_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v61_data = r0[4];
              float v64_data = ir1[2];
              ir1[2] = (v64_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v67_data = r0[6];
              float v70_data = ir1[3];
              ir1[3] = (v70_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v73_data = r0[8];
              float v76_data = ir1[4];
              ir1[4] = (v76_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v79_data = r0[10];
              float v82_data = ir1[5];
              ir1[5] = (v82_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v85_data = r0[12];
              float v88_data = ir1[6];
              ir1[6] = (v88_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v91_data = r0[14];
              float v94_data = ir1[7];
              ir1[7] = (v94_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v97_data = r0[16];
              float v100_data = ir1[8];
              ir1[8] = (v100_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              int32_t v102_a = v23_lead + 10;
              float v103_data_pre = glb_m1[v47_g ? (v102_a) : (0)];
              float v103_data = v47_g ? (v103_data_pre) : (0.0f);
              float v107_data = ir1[0];
              ir1[0] = (v107_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v113_data = ir1[1];
              ir1[1] = (v113_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v119_data = ir1[2];
              ir1[2] = (v119_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v125_data = ir1[3];
              ir1[3] = (v125_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v131_data = ir1[4];
              ir1[4] = (v131_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v137_data = ir1[5];
              ir1[5] = (v137_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v143_data = ir1[6];
              ir1[6] = (v143_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v149_data = ir1[7];
              ir1[7] = (v149_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v155_data = ir1[8];
              ir1[8] = (v155_data + (v103_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              int32_t v157_a = v23_lead + 20;
              float v158_data_pre = glb_m1[v47_g ? (v157_a) : (0)];
              float v158_data = v47_g ? (v158_data_pre) : (0.0f);
              float v162_data = ir1[0];
              ir1[0] = (v162_data + (v158_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v168_data = ir1[1];
              ir1[1] = (v168_data + (v158_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v174_data = ir1[2];
              ir1[2] = (v174_data + (v158_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v180_data = ir1[3];
              ir1[3] = (v180_data + (v158_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v186_data = ir1[4];
              ir1[4] = (v186_data + (v158_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v192_data = ir1[5];
              ir1[5] = (v192_data + (v158_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v198_data = ir1[6];
              ir1[6] = (v198_data + (v158_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v204_data = ir1[7];
              ir1[7] = (v204_data + (v158_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v210_data = ir1[8];
              ir1[8] = (v210_data + (v158_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              int32_t v212_a = v23_lead + 30;
              float v213_data_pre = glb_m1[v47_g ? (v212_a) : (0)];
              float v213_data = v47_g ? (v213_data_pre) : (0.0f);
              float v217_data = ir1[0];
              ir1[0] = (v217_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v223_data = ir1[1];
              ir1[1] = (v223_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v229_data = ir1[2];
              ir1[2] = (v229_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v235_data = ir1[3];
              ir1[3] = (v235_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v241_data = ir1[4];
              ir1[4] = (v241_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v247_data = ir1[5];
              ir1[5] = (v247_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v253_data = ir1[6];
              ir1[6] = (v253_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v259_data = ir1[7];
              ir1[7] = (v259_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v265_data = ir1[8];
              ir1[8] = (v265_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              int32_t v267_a = v23_lead + 40;
              float v268_data_pre = glb_m1[v47_g ? (v267_a) : (0)];
              float v268_data = v47_g ? (v268_data_pre) : (0.0f);
              float v272_data = ir1[0];
              ir1[0] = (v272_data + (v268_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v278_data = ir1[1];
              ir1[1] = (v278_data + (v268_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v284_data = ir1[2];
              ir1[2] = (v284_data + (v268_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v290_data = ir1[3];
              ir1[3] = (v290_data + (v268_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v296_data = ir1[4];
              ir1[4] = (v296_data + (v268_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v302_data = ir1[5];
              ir1[5] = (v302_data + (v268_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v308_data = ir1[6];
              ir1[6] = (v308_data + (v268_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v314_data = ir1[7];
              ir1[7] = (v314_data + (v268_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v320_data = ir1[8];
              ir1[8] = (v320_data + (v268_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              int32_t v322_a = v23_lead + 50;
              float v323_data_pre = glb_m1[v47_g ? (v322_a) : (0)];
              float v323_data = v47_g ? (v323_data_pre) : (0.0f);
              float v327_data = ir1[0];
              ir1[0] = (v327_data + (v323_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v333_data = ir1[1];
              ir1[1] = (v333_data + (v323_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v339_data = ir1[2];
              ir1[2] = (v339_data + (v323_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v345_data = ir1[3];
              ir1[3] = (v345_data + (v323_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v351_data = ir1[4];
              ir1[4] = (v351_data + (v323_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v357_data = ir1[5];
              ir1[5] = (v357_data + (v323_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v363_data = ir1[6];
              ir1[6] = (v363_data + (v323_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v369_data = ir1[7];
              ir1[7] = (v369_data + (v323_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v375_data = ir1[8];
              ir1[8] = (v375_data + (v323_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              int32_t v377_a = v23_lead + 60;
              float v378_data_pre = glb_m1[v47_g ? (v377_a) : (0)];
              float v378_data = v47_g ? (v378_data_pre) : (0.0f);
              float v382_data = ir1[0];
              ir1[0] = (v382_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v388_data = ir1[1];
              ir1[1] = (v388_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v394_data = ir1[2];
              ir1[2] = (v394_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v400_data = ir1[3];
              ir1[3] = (v400_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v406_data = ir1[4];
              ir1[4] = (v406_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v412_data = ir1[5];
              ir1[5] = (v412_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v418_data = ir1[6];
              ir1[6] = (v418_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v424_data = ir1[7];
              ir1[7] = (v424_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v430_data = ir1[8];
              ir1[8] = (v430_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              int32_t v432_a = v23_lead + 70;
              float v433_data_pre = glb_m1[v47_g ? (v432_a) : (0)];
              float v433_data = v47_g ? (v433_data_pre) : (0.0f);
              float v437_data = ir1[0];
              ir1[0] = (v437_data + (v433_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v443_data = ir1[1];
              ir1[1] = (v443_data + (v433_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v449_data = ir1[2];
              ir1[2] = (v449_data + (v433_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v455_data = ir1[3];
              ir1[3] = (v455_data + (v433_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v461_data = ir1[4];
              ir1[4] = (v461_data + (v433_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v467_data = ir1[5];
              ir1[5] = (v467_data + (v433_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v473_data = ir1[6];
              ir1[6] = (v473_data + (v433_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v479_data = ir1[7];
              ir1[7] = (v479_data + (v433_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v485_data = ir1[8];
              ir1[8] = (v485_data + (v433_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              int32_t v487_a = v23_lead + 80;
              float v488_data_pre = glb_m1[v47_g ? (v487_a) : (0)];
              float v488_data = v47_g ? (v488_data_pre) : (0.0f);
              float v492_data = ir1[0];
              ir1[0] = (v492_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v498_data = ir1[1];
              ir1[1] = (v498_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v504_data = ir1[2];
              ir1[2] = (v504_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v510_data = ir1[3];
              ir1[3] = (v510_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v516_data = ir1[4];
              ir1[4] = (v516_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v522_data = ir1[5];
              ir1[5] = (v522_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v528_data = ir1[6];
              ir1[6] = (v528_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v534_data = ir1[7];
              ir1[7] = (v534_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v540_data = ir1[8];
              ir1[8] = (v540_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              int32_t v542_a = v23_lead + 90;
              float v543_data_pre = glb_m1[v47_g ? (v542_a) : (0)];
              float v543_data = v47_g ? (v543_data_pre) : (0.0f);
              float v547_data = ir1[0];
              ir1[0] = (v547_data + (v543_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v553_data = ir1[1];
              ir1[1] = (v553_data + (v543_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v559_data = ir1[2];
              ir1[2] = (v559_data + (v543_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v565_data = ir1[3];
              ir1[3] = (v565_data + (v543_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v571_data = ir1[4];
              ir1[4] = (v571_data + (v543_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v577_data = ir1[5];
              ir1[5] = (v577_data + (v543_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v583_data = ir1[6];
              ir1[6] = (v583_data + (v543_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v589_data = ir1[7];
              ir1[7] = (v589_data + (v543_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v595_data = ir1[8];
              ir1[8] = (v595_data + (v543_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              int32_t v597_a = v23_lead + 100;
              float v598_data_pre = glb_m1[v47_g ? (v597_a) : (0)];
              float v598_data = v47_g ? (v598_data_pre) : (0.0f);
              float v602_data = ir1[0];
              ir1[0] = (v602_data + (v598_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v608_data = ir1[1];
              ir1[1] = (v608_data + (v598_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v614_data = ir1[2];
              ir1[2] = (v614_data + (v598_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v620_data = ir1[3];
              ir1[3] = (v620_data + (v598_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v626_data = ir1[4];
              ir1[4] = (v626_data + (v598_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v632_data = ir1[5];
              ir1[5] = (v632_data + (v598_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v638_data = ir1[6];
              ir1[6] = (v638_data + (v598_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v644_data = ir1[7];
              ir1[7] = (v644_data + (v598_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v650_data = ir1[8];
              ir1[8] = (v650_data + (v598_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              int32_t v652_a = v23_lead + 110;
              float v653_data_pre = glb_m1[v47_g ? (v652_a) : (0)];
              float v653_data = v47_g ? (v653_data_pre) : (0.0f);
              float v657_data = ir1[0];
              ir1[0] = (v657_data + (v653_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v663_data = ir1[1];
              ir1[1] = (v663_data + (v653_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v669_data = ir1[2];
              ir1[2] = (v669_data + (v653_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v675_data = ir1[3];
              ir1[3] = (v675_data + (v653_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v681_data = ir1[4];
              ir1[4] = (v681_data + (v653_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v687_data = ir1[5];
              ir1[5] = (v687_data + (v653_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v693_data = ir1[6];
              ir1[6] = (v693_data + (v653_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v699_data = ir1[7];
              ir1[7] = (v699_data + (v653_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v705_data = ir1[8];
              ir1[8] = (v705_data + (v653_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              int32_t v707_a = v23_lead + 120;
              float v708_data_pre = glb_m1[v47_g ? (v707_a) : (0)];
              float v708_data = v47_g ? (v708_data_pre) : (0.0f);
              float v712_data = ir1[0];
              ir1[0] = (v712_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v718_data = ir1[1];
              ir1[1] = (v718_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v724_data = ir1[2];
              ir1[2] = (v724_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v730_data = ir1[3];
              ir1[3] = (v730_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v736_data = ir1[4];
              ir1[4] = (v736_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v742_data = ir1[5];
              ir1[5] = (v742_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v748_data = ir1[6];
              ir1[6] = (v748_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v754_data = ir1[7];
              ir1[7] = (v754_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v760_data = ir1[8];
              ir1[8] = (v760_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              int32_t v762_a = v23_lead + 130;
              float v763_data_pre = glb_m1[v47_g ? (v762_a) : (0)];
              float v763_data = v47_g ? (v763_data_pre) : (0.0f);
              float v767_data = ir1[0];
              ir1[0] = (v767_data + (v763_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v773_data = ir1[1];
              ir1[1] = (v773_data + (v763_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v779_data = ir1[2];
              ir1[2] = (v779_data + (v763_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v785_data = ir1[3];
              ir1[3] = (v785_data + (v763_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v791_data = ir1[4];
              ir1[4] = (v791_data + (v763_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v797_data = ir1[5];
              ir1[5] = (v797_data + (v763_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v803_data = ir1[6];
              ir1[6] = (v803_data + (v763_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v809_data = ir1[7];
              ir1[7] = (v809_data + (v763_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v815_data = ir1[8];
              ir1[8] = (v815_data + (v763_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              int32_t v817_a = v23_lead + 140;
              float v818_data_pre = glb_m1[v47_g ? (v817_a) : (0)];
              float v818_data = v47_g ? (v818_data_pre) : (0.0f);
              float v822_data = ir1[0];
              ir1[0] = (v822_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v828_data = ir1[1];
              ir1[1] = (v828_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v834_data = ir1[2];
              ir1[2] = (v834_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v840_data = ir1[3];
              ir1[3] = (v840_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v846_data = ir1[4];
              ir1[4] = (v846_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v852_data = ir1[5];
              ir1[5] = (v852_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v858_data = ir1[6];
              ir1[6] = (v858_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v864_data = ir1[7];
              ir1[7] = (v864_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v870_data = ir1[8];
              ir1[8] = (v870_data + (v818_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              int32_t v872_a = v23_lead + 150;
              float v873_data_pre = glb_m1[v47_g ? (v872_a) : (0)];
              float v873_data = v47_g ? (v873_data_pre) : (0.0f);
              float v877_data = ir1[0];
              ir1[0] = (v877_data + (v873_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v883_data = ir1[1];
              ir1[1] = (v883_data + (v873_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v889_data = ir1[2];
              ir1[2] = (v889_data + (v873_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v895_data = ir1[3];
              ir1[3] = (v895_data + (v873_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v901_data = ir1[4];
              ir1[4] = (v901_data + (v873_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v907_data = ir1[5];
              ir1[5] = (v907_data + (v873_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v913_data = ir1[6];
              ir1[6] = (v913_data + (v873_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v919_data = ir1[7];
              ir1[7] = (v919_data + (v873_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v925_data = ir1[8];
              ir1[8] = (v925_data + (v873_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              int32_t v927_a = v23_lead + 160;
              float v928_data_pre = glb_m1[v47_g ? (v927_a) : (0)];
              float v928_data = v47_g ? (v928_data_pre) : (0.0f);
              float v929_data = r0[1];
              float v932_data = ir1[0];
              ir1[0] = (v932_data + (v928_data * (sycl::select_from_group(item.get_sub_group(), v929_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v935_data = r0[3];
              float v938_data = ir1[1];
              ir1[1] = (v938_data + (v928_data * (sycl::select_from_group(item.get_sub_group(), v935_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v941_data = r0[5];
              float v944_data = ir1[2];
              ir1[2] = (v944_data + (v928_data * (sycl::select_from_group(item.get_sub_group(), v941_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v947_data = r0[7];
              float v950_data = ir1[3];
              ir1[3] = (v950_data + (v928_data * (sycl::select_from_group(item.get_sub_group(), v947_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v953_data = r0[9];
              float v956_data = ir1[4];
              ir1[4] = (v956_data + (v928_data * (sycl::select_from_group(item.get_sub_group(), v953_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v959_data = r0[11];
              float v962_data = ir1[5];
              ir1[5] = (v962_data + (v928_data * (sycl::select_from_group(item.get_sub_group(), v959_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v965_data = r0[13];
              float v968_data = ir1[6];
              ir1[6] = (v968_data + (v928_data * (sycl::select_from_group(item.get_sub_group(), v965_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v971_data = r0[15];
              float v974_data = ir1[7];
              ir1[7] = (v974_data + (v928_data * (sycl::select_from_group(item.get_sub_group(), v971_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v977_data = r0[17];
              float v980_data = ir1[8];
              ir1[8] = (v980_data + (v928_data * (sycl::select_from_group(item.get_sub_group(), v977_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              // r1 = ir1
              if (v47_g) {
                #pragma unroll
                for (int32_t v983_n1 = 0; v983_n1 < 9; ++v983_n1) {
                  float v985_data = ir1[v983_n1];
                  r1[v983_n1] = v985_data;
                }
              }
              float r3[9]{};
              // ir3 = +(glb_m3 * r2)
              // [(0, 10), (0, 9)] [(0, 17)]
              float ir3[9]{};
              float v1010_data_pre = glb_m3[v47_g ? (v23_lead) : (0)];
              float v1010_data = v47_g ? (v1010_data_pre) : (0.0f);
              float v1011_data = r2[0];
              float v1014_data = ir3[0];
              ir3[0] = (v1014_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1017_data = r2[2];
              float v1020_data = ir3[1];
              ir3[1] = (v1020_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1023_data = r2[4];
              float v1026_data = ir3[2];
              ir3[2] = (v1026_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1029_data = r2[6];
              float v1032_data = ir3[3];
              ir3[3] = (v1032_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1035_data = r2[8];
              float v1038_data = ir3[4];
              ir3[4] = (v1038_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1041_data = r2[10];
              float v1044_data = ir3[5];
              ir3[5] = (v1044_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1047_data = r2[12];
              float v1050_data = ir3[6];
              ir3[6] = (v1050_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1053_data = r2[14];
              float v1056_data = ir3[7];
              ir3[7] = (v1056_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1059_data = r2[16];
              float v1062_data = ir3[8];
              ir3[8] = (v1062_data + (v1010_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1065_data_pre = glb_m3[v47_g ? (v102_a) : (0)];
              float v1065_data = v47_g ? (v1065_data_pre) : (0.0f);
              float v1069_data = ir3[0];
              ir3[0] = (v1069_data + (v1065_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1075_data = ir3[1];
              ir3[1] = (v1075_data + (v1065_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1081_data = ir3[2];
              ir3[2] = (v1081_data + (v1065_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1087_data = ir3[3];
              ir3[3] = (v1087_data + (v1065_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1093_data = ir3[4];
              ir3[4] = (v1093_data + (v1065_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1099_data = ir3[5];
              ir3[5] = (v1099_data + (v1065_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1105_data = ir3[6];
              ir3[6] = (v1105_data + (v1065_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1111_data = ir3[7];
              ir3[7] = (v1111_data + (v1065_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1117_data = ir3[8];
              ir3[8] = (v1117_data + (v1065_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1120_data_pre = glb_m3[v47_g ? (v157_a) : (0)];
              float v1120_data = v47_g ? (v1120_data_pre) : (0.0f);
              float v1124_data = ir3[0];
              ir3[0] = (v1124_data + (v1120_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1130_data = ir3[1];
              ir3[1] = (v1130_data + (v1120_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1136_data = ir3[2];
              ir3[2] = (v1136_data + (v1120_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1142_data = ir3[3];
              ir3[3] = (v1142_data + (v1120_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1148_data = ir3[4];
              ir3[4] = (v1148_data + (v1120_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1154_data = ir3[5];
              ir3[5] = (v1154_data + (v1120_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1160_data = ir3[6];
              ir3[6] = (v1160_data + (v1120_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1166_data = ir3[7];
              ir3[7] = (v1166_data + (v1120_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1172_data = ir3[8];
              ir3[8] = (v1172_data + (v1120_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1175_data_pre = glb_m3[v47_g ? (v212_a) : (0)];
              float v1175_data = v47_g ? (v1175_data_pre) : (0.0f);
              float v1179_data = ir3[0];
              ir3[0] = (v1179_data + (v1175_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1185_data = ir3[1];
              ir3[1] = (v1185_data + (v1175_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1191_data = ir3[2];
              ir3[2] = (v1191_data + (v1175_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1197_data = ir3[3];
              ir3[3] = (v1197_data + (v1175_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1203_data = ir3[4];
              ir3[4] = (v1203_data + (v1175_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1209_data = ir3[5];
              ir3[5] = (v1209_data + (v1175_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1215_data = ir3[6];
              ir3[6] = (v1215_data + (v1175_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1221_data = ir3[7];
              ir3[7] = (v1221_data + (v1175_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1227_data = ir3[8];
              ir3[8] = (v1227_data + (v1175_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1230_data_pre = glb_m3[v47_g ? (v267_a) : (0)];
              float v1230_data = v47_g ? (v1230_data_pre) : (0.0f);
              float v1234_data = ir3[0];
              ir3[0] = (v1234_data + (v1230_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1240_data = ir3[1];
              ir3[1] = (v1240_data + (v1230_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1246_data = ir3[2];
              ir3[2] = (v1246_data + (v1230_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1252_data = ir3[3];
              ir3[3] = (v1252_data + (v1230_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1258_data = ir3[4];
              ir3[4] = (v1258_data + (v1230_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1264_data = ir3[5];
              ir3[5] = (v1264_data + (v1230_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1270_data = ir3[6];
              ir3[6] = (v1270_data + (v1230_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1276_data = ir3[7];
              ir3[7] = (v1276_data + (v1230_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1282_data = ir3[8];
              ir3[8] = (v1282_data + (v1230_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1285_data_pre = glb_m3[v47_g ? (v322_a) : (0)];
              float v1285_data = v47_g ? (v1285_data_pre) : (0.0f);
              float v1289_data = ir3[0];
              ir3[0] = (v1289_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1295_data = ir3[1];
              ir3[1] = (v1295_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1301_data = ir3[2];
              ir3[2] = (v1301_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1307_data = ir3[3];
              ir3[3] = (v1307_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1313_data = ir3[4];
              ir3[4] = (v1313_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1319_data = ir3[5];
              ir3[5] = (v1319_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1325_data = ir3[6];
              ir3[6] = (v1325_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1331_data = ir3[7];
              ir3[7] = (v1331_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1337_data = ir3[8];
              ir3[8] = (v1337_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1340_data_pre = glb_m3[v47_g ? (v377_a) : (0)];
              float v1340_data = v47_g ? (v1340_data_pre) : (0.0f);
              float v1344_data = ir3[0];
              ir3[0] = (v1344_data + (v1340_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1350_data = ir3[1];
              ir3[1] = (v1350_data + (v1340_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1356_data = ir3[2];
              ir3[2] = (v1356_data + (v1340_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1362_data = ir3[3];
              ir3[3] = (v1362_data + (v1340_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1368_data = ir3[4];
              ir3[4] = (v1368_data + (v1340_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1374_data = ir3[5];
              ir3[5] = (v1374_data + (v1340_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1380_data = ir3[6];
              ir3[6] = (v1380_data + (v1340_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1386_data = ir3[7];
              ir3[7] = (v1386_data + (v1340_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1392_data = ir3[8];
              ir3[8] = (v1392_data + (v1340_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1395_data_pre = glb_m3[v47_g ? (v432_a) : (0)];
              float v1395_data = v47_g ? (v1395_data_pre) : (0.0f);
              float v1399_data = ir3[0];
              ir3[0] = (v1399_data + (v1395_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1405_data = ir3[1];
              ir3[1] = (v1405_data + (v1395_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1411_data = ir3[2];
              ir3[2] = (v1411_data + (v1395_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1417_data = ir3[3];
              ir3[3] = (v1417_data + (v1395_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1423_data = ir3[4];
              ir3[4] = (v1423_data + (v1395_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1429_data = ir3[5];
              ir3[5] = (v1429_data + (v1395_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1435_data = ir3[6];
              ir3[6] = (v1435_data + (v1395_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1441_data = ir3[7];
              ir3[7] = (v1441_data + (v1395_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1447_data = ir3[8];
              ir3[8] = (v1447_data + (v1395_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1450_data_pre = glb_m3[v47_g ? (v487_a) : (0)];
              float v1450_data = v47_g ? (v1450_data_pre) : (0.0f);
              float v1454_data = ir3[0];
              ir3[0] = (v1454_data + (v1450_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1460_data = ir3[1];
              ir3[1] = (v1460_data + (v1450_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1466_data = ir3[2];
              ir3[2] = (v1466_data + (v1450_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1472_data = ir3[3];
              ir3[3] = (v1472_data + (v1450_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1478_data = ir3[4];
              ir3[4] = (v1478_data + (v1450_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1484_data = ir3[5];
              ir3[5] = (v1484_data + (v1450_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1490_data = ir3[6];
              ir3[6] = (v1490_data + (v1450_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1496_data = ir3[7];
              ir3[7] = (v1496_data + (v1450_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1502_data = ir3[8];
              ir3[8] = (v1502_data + (v1450_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1505_data_pre = glb_m3[v47_g ? (v542_a) : (0)];
              float v1505_data = v47_g ? (v1505_data_pre) : (0.0f);
              float v1509_data = ir3[0];
              ir3[0] = (v1509_data + (v1505_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1515_data = ir3[1];
              ir3[1] = (v1515_data + (v1505_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1521_data = ir3[2];
              ir3[2] = (v1521_data + (v1505_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1527_data = ir3[3];
              ir3[3] = (v1527_data + (v1505_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1533_data = ir3[4];
              ir3[4] = (v1533_data + (v1505_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1539_data = ir3[5];
              ir3[5] = (v1539_data + (v1505_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1545_data = ir3[6];
              ir3[6] = (v1545_data + (v1505_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1551_data = ir3[7];
              ir3[7] = (v1551_data + (v1505_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1557_data = ir3[8];
              ir3[8] = (v1557_data + (v1505_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1560_data_pre = glb_m3[v47_g ? (v597_a) : (0)];
              float v1560_data = v47_g ? (v1560_data_pre) : (0.0f);
              float v1564_data = ir3[0];
              ir3[0] = (v1564_data + (v1560_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1570_data = ir3[1];
              ir3[1] = (v1570_data + (v1560_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1576_data = ir3[2];
              ir3[2] = (v1576_data + (v1560_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1582_data = ir3[3];
              ir3[3] = (v1582_data + (v1560_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1588_data = ir3[4];
              ir3[4] = (v1588_data + (v1560_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1594_data = ir3[5];
              ir3[5] = (v1594_data + (v1560_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1600_data = ir3[6];
              ir3[6] = (v1600_data + (v1560_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1606_data = ir3[7];
              ir3[7] = (v1606_data + (v1560_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1612_data = ir3[8];
              ir3[8] = (v1612_data + (v1560_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1615_data_pre = glb_m3[v47_g ? (v652_a) : (0)];
              float v1615_data = v47_g ? (v1615_data_pre) : (0.0f);
              float v1619_data = ir3[0];
              ir3[0] = (v1619_data + (v1615_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1625_data = ir3[1];
              ir3[1] = (v1625_data + (v1615_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1631_data = ir3[2];
              ir3[2] = (v1631_data + (v1615_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1637_data = ir3[3];
              ir3[3] = (v1637_data + (v1615_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1643_data = ir3[4];
              ir3[4] = (v1643_data + (v1615_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1649_data = ir3[5];
              ir3[5] = (v1649_data + (v1615_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1655_data = ir3[6];
              ir3[6] = (v1655_data + (v1615_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1661_data = ir3[7];
              ir3[7] = (v1661_data + (v1615_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1667_data = ir3[8];
              ir3[8] = (v1667_data + (v1615_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1670_data_pre = glb_m3[v47_g ? (v707_a) : (0)];
              float v1670_data = v47_g ? (v1670_data_pre) : (0.0f);
              float v1674_data = ir3[0];
              ir3[0] = (v1674_data + (v1670_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1680_data = ir3[1];
              ir3[1] = (v1680_data + (v1670_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1686_data = ir3[2];
              ir3[2] = (v1686_data + (v1670_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1692_data = ir3[3];
              ir3[3] = (v1692_data + (v1670_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1698_data = ir3[4];
              ir3[4] = (v1698_data + (v1670_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1704_data = ir3[5];
              ir3[5] = (v1704_data + (v1670_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1710_data = ir3[6];
              ir3[6] = (v1710_data + (v1670_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1716_data = ir3[7];
              ir3[7] = (v1716_data + (v1670_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1722_data = ir3[8];
              ir3[8] = (v1722_data + (v1670_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v1725_data_pre = glb_m3[v47_g ? (v762_a) : (0)];
              float v1725_data = v47_g ? (v1725_data_pre) : (0.0f);
              float v1729_data = ir3[0];
              ir3[0] = (v1729_data + (v1725_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1735_data = ir3[1];
              ir3[1] = (v1735_data + (v1725_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1741_data = ir3[2];
              ir3[2] = (v1741_data + (v1725_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1747_data = ir3[3];
              ir3[3] = (v1747_data + (v1725_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1753_data = ir3[4];
              ir3[4] = (v1753_data + (v1725_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1759_data = ir3[5];
              ir3[5] = (v1759_data + (v1725_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1765_data = ir3[6];
              ir3[6] = (v1765_data + (v1725_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1771_data = ir3[7];
              ir3[7] = (v1771_data + (v1725_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1777_data = ir3[8];
              ir3[8] = (v1777_data + (v1725_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v1780_data_pre = glb_m3[v47_g ? (v817_a) : (0)];
              float v1780_data = v47_g ? (v1780_data_pre) : (0.0f);
              float v1784_data = ir3[0];
              ir3[0] = (v1784_data + (v1780_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1790_data = ir3[1];
              ir3[1] = (v1790_data + (v1780_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1796_data = ir3[2];
              ir3[2] = (v1796_data + (v1780_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1802_data = ir3[3];
              ir3[3] = (v1802_data + (v1780_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1808_data = ir3[4];
              ir3[4] = (v1808_data + (v1780_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1814_data = ir3[5];
              ir3[5] = (v1814_data + (v1780_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1820_data = ir3[6];
              ir3[6] = (v1820_data + (v1780_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1826_data = ir3[7];
              ir3[7] = (v1826_data + (v1780_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1832_data = ir3[8];
              ir3[8] = (v1832_data + (v1780_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v1835_data_pre = glb_m3[v47_g ? (v872_a) : (0)];
              float v1835_data = v47_g ? (v1835_data_pre) : (0.0f);
              float v1839_data = ir3[0];
              ir3[0] = (v1839_data + (v1835_data * (sycl::select_from_group(item.get_sub_group(), v1011_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1845_data = ir3[1];
              ir3[1] = (v1845_data + (v1835_data * (sycl::select_from_group(item.get_sub_group(), v1017_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1851_data = ir3[2];
              ir3[2] = (v1851_data + (v1835_data * (sycl::select_from_group(item.get_sub_group(), v1023_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1857_data = ir3[3];
              ir3[3] = (v1857_data + (v1835_data * (sycl::select_from_group(item.get_sub_group(), v1029_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1863_data = ir3[4];
              ir3[4] = (v1863_data + (v1835_data * (sycl::select_from_group(item.get_sub_group(), v1035_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1869_data = ir3[5];
              ir3[5] = (v1869_data + (v1835_data * (sycl::select_from_group(item.get_sub_group(), v1041_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1875_data = ir3[6];
              ir3[6] = (v1875_data + (v1835_data * (sycl::select_from_group(item.get_sub_group(), v1047_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1881_data = ir3[7];
              ir3[7] = (v1881_data + (v1835_data * (sycl::select_from_group(item.get_sub_group(), v1053_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1887_data = ir3[8];
              ir3[8] = (v1887_data + (v1835_data * (sycl::select_from_group(item.get_sub_group(), v1059_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v1890_data_pre = glb_m3[v47_g ? (v927_a) : (0)];
              float v1890_data = v47_g ? (v1890_data_pre) : (0.0f);
              float v1891_data = r2[1];
              float v1894_data = ir3[0];
              ir3[0] = (v1894_data + (v1890_data * (sycl::select_from_group(item.get_sub_group(), v1891_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1897_data = r2[3];
              float v1900_data = ir3[1];
              ir3[1] = (v1900_data + (v1890_data * (sycl::select_from_group(item.get_sub_group(), v1897_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1903_data = r2[5];
              float v1906_data = ir3[2];
              ir3[2] = (v1906_data + (v1890_data * (sycl::select_from_group(item.get_sub_group(), v1903_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1909_data = r2[7];
              float v1912_data = ir3[3];
              ir3[3] = (v1912_data + (v1890_data * (sycl::select_from_group(item.get_sub_group(), v1909_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1915_data = r2[9];
              float v1918_data = ir3[4];
              ir3[4] = (v1918_data + (v1890_data * (sycl::select_from_group(item.get_sub_group(), v1915_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1921_data = r2[11];
              float v1924_data = ir3[5];
              ir3[5] = (v1924_data + (v1890_data * (sycl::select_from_group(item.get_sub_group(), v1921_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1927_data = r2[13];
              float v1930_data = ir3[6];
              ir3[6] = (v1930_data + (v1890_data * (sycl::select_from_group(item.get_sub_group(), v1927_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1933_data = r2[15];
              float v1936_data = ir3[7];
              ir3[7] = (v1936_data + (v1890_data * (sycl::select_from_group(item.get_sub_group(), v1933_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1939_data = r2[17];
              float v1942_data = ir3[8];
              ir3[8] = (v1942_data + (v1890_data * (sycl::select_from_group(item.get_sub_group(), v1939_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              // r3 = ir3 + r1
              if (v47_g) {
                #pragma unroll
                for (int32_t v1945_n1 = 0; v1945_n1 < 9; ++v1945_n1) {
                  float v1947_data = ir3[v1945_n1];
                  float v1948_data = r1[v1945_n1];
                  r3[v1945_n1] = (v1948_data + v1947_data);
                }
              }
              // glb_m0 = store{r>g}(r3);
              if (v47_g) {
                #pragma unroll
                for (int32_t v1951_i1 = 0; v1951_i1 < 9; ++v1951_i1) {
                  float v1953_data = r3[v1951_i1];
                  glb_m0[(v23_lead + (v1951_i1 * 10))] = v1953_data;
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

