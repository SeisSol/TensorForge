// === base name ===
kernel_7a78e7e4b27dfda3

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_7a78e7e4b27dfda3 = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_7a78e7e4b27dfda3(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_7a78e7e4b27dfda3(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_7a78e7e4b27dfda3(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_7a78e7e4b27dfda3(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_7a78e7e4b27dfda3(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_7a78e7e4b27dfda3(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_7a78e7e4b27dfda3(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 12×8(12×8) {0..12}×{0..8} strided
        //   m1 32×16(12×16) {4..16}×{0..16} strided
        //   m2 16×8(16×8) {0..16}×{0..8} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,8]],"name":"m0","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[4,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[32,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m1","offset":[4,0],"shape":[32,16]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v9_batchId0 * 192 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v9_batchId0 * 128 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v23_lead = item.get_local_id(2) % 16;
              bool v24_g = v23_lead < 12;
              if (v24_g) {
                int32_t v29_a = (v23_lead + 4) - 4;
                #pragma unroll
                for (int32_t v25_i1 = 0; v25_i1 < 16; ++v25_i1) {
                  float v32_data = glb_m1[(v29_a + (v25_i1 * 12))];
                  r0[v25_i1] = v32_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v35_i0 = 0; v35_i0 < 1; ++v35_i0) {
                int32_t v38_lead = v23_lead + (v35_i0 * 16);
                #pragma unroll
                for (int32_t v36_i1 = 0; v36_i1 < 8; ++v36_i1) {
                  float v41_data = glb_m2[(v38_lead + (v36_i1 * 16))];
                  r1[(v35_i0 + v36_i1)] = v41_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // ir2 = +(r0 * r1)
              // [(0, 12), (0, 8)] [(0, 16)]
              float ir2[8]{};
              float v45_data = r0[0];
              float v46_data = r1[0];
              float v49_data = ir2[0];
              ir2[0] = (v49_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v52_data = r1[1];
              float v55_data = ir2[1];
              ir2[1] = (v55_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v58_data = r1[2];
              float v61_data = ir2[2];
              ir2[2] = (v61_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v64_data = r1[3];
              float v67_data = ir2[3];
              ir2[3] = (v67_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v70_data = r1[4];
              float v73_data = ir2[4];
              ir2[4] = (v73_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v76_data = r1[5];
              float v79_data = ir2[5];
              ir2[5] = (v79_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v82_data = r1[6];
              float v85_data = ir2[6];
              ir2[6] = (v85_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v88_data = r1[7];
              float v91_data = ir2[7];
              ir2[7] = (v91_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v93_data = r0[1];
              float v97_data = ir2[0];
              ir2[0] = (v97_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v103_data = ir2[1];
              ir2[1] = (v103_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v109_data = ir2[2];
              ir2[2] = (v109_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v115_data = ir2[3];
              ir2[3] = (v115_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v121_data = ir2[4];
              ir2[4] = (v121_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v127_data = ir2[5];
              ir2[5] = (v127_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v133_data = ir2[6];
              ir2[6] = (v133_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v139_data = ir2[7];
              ir2[7] = (v139_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v141_data = r0[2];
              float v145_data = ir2[0];
              ir2[0] = (v145_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v151_data = ir2[1];
              ir2[1] = (v151_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v157_data = ir2[2];
              ir2[2] = (v157_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v163_data = ir2[3];
              ir2[3] = (v163_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v169_data = ir2[4];
              ir2[4] = (v169_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v175_data = ir2[5];
              ir2[5] = (v175_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v181_data = ir2[6];
              ir2[6] = (v181_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v187_data = ir2[7];
              ir2[7] = (v187_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v189_data = r0[3];
              float v193_data = ir2[0];
              ir2[0] = (v193_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v199_data = ir2[1];
              ir2[1] = (v199_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v205_data = ir2[2];
              ir2[2] = (v205_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v211_data = ir2[3];
              ir2[3] = (v211_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v217_data = ir2[4];
              ir2[4] = (v217_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v223_data = ir2[5];
              ir2[5] = (v223_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v229_data = ir2[6];
              ir2[6] = (v229_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v235_data = ir2[7];
              ir2[7] = (v235_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v237_data = r0[4];
              float v241_data = ir2[0];
              ir2[0] = (v241_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v247_data = ir2[1];
              ir2[1] = (v247_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v253_data = ir2[2];
              ir2[2] = (v253_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v259_data = ir2[3];
              ir2[3] = (v259_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v265_data = ir2[4];
              ir2[4] = (v265_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v271_data = ir2[5];
              ir2[5] = (v271_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v277_data = ir2[6];
              ir2[6] = (v277_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v283_data = ir2[7];
              ir2[7] = (v283_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v285_data = r0[5];
              float v289_data = ir2[0];
              ir2[0] = (v289_data + (v285_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v295_data = ir2[1];
              ir2[1] = (v295_data + (v285_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v301_data = ir2[2];
              ir2[2] = (v301_data + (v285_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v307_data = ir2[3];
              ir2[3] = (v307_data + (v285_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v313_data = ir2[4];
              ir2[4] = (v313_data + (v285_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v319_data = ir2[5];
              ir2[5] = (v319_data + (v285_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v325_data = ir2[6];
              ir2[6] = (v325_data + (v285_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v331_data = ir2[7];
              ir2[7] = (v331_data + (v285_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v333_data = r0[6];
              float v337_data = ir2[0];
              ir2[0] = (v337_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v343_data = ir2[1];
              ir2[1] = (v343_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v349_data = ir2[2];
              ir2[2] = (v349_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v355_data = ir2[3];
              ir2[3] = (v355_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v361_data = ir2[4];
              ir2[4] = (v361_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v367_data = ir2[5];
              ir2[5] = (v367_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v373_data = ir2[6];
              ir2[6] = (v373_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v379_data = ir2[7];
              ir2[7] = (v379_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v381_data = r0[7];
              float v385_data = ir2[0];
              ir2[0] = (v385_data + (v381_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v391_data = ir2[1];
              ir2[1] = (v391_data + (v381_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v397_data = ir2[2];
              ir2[2] = (v397_data + (v381_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v403_data = ir2[3];
              ir2[3] = (v403_data + (v381_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v409_data = ir2[4];
              ir2[4] = (v409_data + (v381_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v415_data = ir2[5];
              ir2[5] = (v415_data + (v381_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v421_data = ir2[6];
              ir2[6] = (v421_data + (v381_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v427_data = ir2[7];
              ir2[7] = (v427_data + (v381_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v429_data = r0[8];
              float v433_data = ir2[0];
              ir2[0] = (v433_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v439_data = ir2[1];
              ir2[1] = (v439_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v445_data = ir2[2];
              ir2[2] = (v445_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v451_data = ir2[3];
              ir2[3] = (v451_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v457_data = ir2[4];
              ir2[4] = (v457_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v463_data = ir2[5];
              ir2[5] = (v463_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v469_data = ir2[6];
              ir2[6] = (v469_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v475_data = ir2[7];
              ir2[7] = (v475_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v477_data = r0[9];
              float v481_data = ir2[0];
              ir2[0] = (v481_data + (v477_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v487_data = ir2[1];
              ir2[1] = (v487_data + (v477_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v493_data = ir2[2];
              ir2[2] = (v493_data + (v477_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v499_data = ir2[3];
              ir2[3] = (v499_data + (v477_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v505_data = ir2[4];
              ir2[4] = (v505_data + (v477_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v511_data = ir2[5];
              ir2[5] = (v511_data + (v477_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v517_data = ir2[6];
              ir2[6] = (v517_data + (v477_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v523_data = ir2[7];
              ir2[7] = (v523_data + (v477_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v525_data = r0[10];
              float v529_data = ir2[0];
              ir2[0] = (v529_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v535_data = ir2[1];
              ir2[1] = (v535_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v541_data = ir2[2];
              ir2[2] = (v541_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v547_data = ir2[3];
              ir2[3] = (v547_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v553_data = ir2[4];
              ir2[4] = (v553_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v559_data = ir2[5];
              ir2[5] = (v559_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v565_data = ir2[6];
              ir2[6] = (v565_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v571_data = ir2[7];
              ir2[7] = (v571_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v573_data = r0[11];
              float v577_data = ir2[0];
              ir2[0] = (v577_data + (v573_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v583_data = ir2[1];
              ir2[1] = (v583_data + (v573_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v589_data = ir2[2];
              ir2[2] = (v589_data + (v573_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v595_data = ir2[3];
              ir2[3] = (v595_data + (v573_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v601_data = ir2[4];
              ir2[4] = (v601_data + (v573_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v607_data = ir2[5];
              ir2[5] = (v607_data + (v573_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v613_data = ir2[6];
              ir2[6] = (v613_data + (v573_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v619_data = ir2[7];
              ir2[7] = (v619_data + (v573_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v621_data = r0[12];
              float v625_data = ir2[0];
              ir2[0] = (v625_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v631_data = ir2[1];
              ir2[1] = (v631_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v637_data = ir2[2];
              ir2[2] = (v637_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v643_data = ir2[3];
              ir2[3] = (v643_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v649_data = ir2[4];
              ir2[4] = (v649_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v655_data = ir2[5];
              ir2[5] = (v655_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v661_data = ir2[6];
              ir2[6] = (v661_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v667_data = ir2[7];
              ir2[7] = (v667_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v669_data = r0[13];
              float v673_data = ir2[0];
              ir2[0] = (v673_data + (v669_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v679_data = ir2[1];
              ir2[1] = (v679_data + (v669_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v685_data = ir2[2];
              ir2[2] = (v685_data + (v669_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v691_data = ir2[3];
              ir2[3] = (v691_data + (v669_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v697_data = ir2[4];
              ir2[4] = (v697_data + (v669_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v703_data = ir2[5];
              ir2[5] = (v703_data + (v669_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v709_data = ir2[6];
              ir2[6] = (v709_data + (v669_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v715_data = ir2[7];
              ir2[7] = (v715_data + (v669_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v717_data = r0[14];
              float v721_data = ir2[0];
              ir2[0] = (v721_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v727_data = ir2[1];
              ir2[1] = (v727_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v733_data = ir2[2];
              ir2[2] = (v733_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v739_data = ir2[3];
              ir2[3] = (v739_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v745_data = ir2[4];
              ir2[4] = (v745_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v751_data = ir2[5];
              ir2[5] = (v751_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v757_data = ir2[6];
              ir2[6] = (v757_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v763_data = ir2[7];
              ir2[7] = (v763_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v765_data = r0[15];
              float v769_data = ir2[0];
              ir2[0] = (v769_data + (v765_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v775_data = ir2[1];
              ir2[1] = (v775_data + (v765_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v781_data = ir2[2];
              ir2[2] = (v781_data + (v765_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v787_data = ir2[3];
              ir2[3] = (v787_data + (v765_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v793_data = ir2[4];
              ir2[4] = (v793_data + (v765_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v799_data = ir2[5];
              ir2[5] = (v799_data + (v765_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v805_data = ir2[6];
              ir2[6] = (v805_data + (v765_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v811_data = ir2[7];
              ir2[7] = (v811_data + (v765_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              // r2 = ir2
              if (v24_g) {
                #pragma unroll
                for (int32_t v813_n1 = 0; v813_n1 < 8; ++v813_n1) {
                  float v815_data = ir2[v813_n1];
                  r2[v813_n1] = v815_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v24_g) {
                #pragma unroll
                for (int32_t v816_i1 = 0; v816_i1 < 8; ++v816_i1) {
                  float v818_data = r2[v816_i1];
                  glb_m0[(v23_lead + (v816_i1 * 12))] = v818_data;
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

