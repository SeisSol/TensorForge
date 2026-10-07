// === base name ===
kernel_3dc70fddd25d151e

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_3dc70fddd25d151e = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_3dc70fddd25d151e(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_3dc70fddd25d151e(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_3dc70fddd25d151e(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_3dc70fddd25d151e(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_3dc70fddd25d151e(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_3dc70fddd25d151e(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_3dc70fddd25d151e(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 8×8(8×8) {0..8}×{0..8} strided
        //   m1 8×8(8×8) {0..8}×{0..8} strided
        //   m2 8(8) {0..8} strided
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   OUT = +(TMP, dims=[1])
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"OUT","bbox":[[0],[8]],"name":"m2","ordered":false,"parts":1,"shape":[8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0],[8]],"is_tmp":false,"name":"m2","offset":[0],"shape":[8]},"kind":"reduction","op":"+","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"target":[[0,-1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 64 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 8 + 0 + m2_extraOffset];
              float r0[8]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v21_lead = item.get_local_id(2) % 16;
              bool v22_g = v21_lead < 8;
              if (v22_g) {
                #pragma unroll
                for (int32_t v23_i1 = 0; v23_i1 < 8; ++v23_i1) {
                  float v28_data = glb_m0[(v21_lead + (v23_i1 * 8))];
                  r0[v23_i1] = v28_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m1);
              if (v22_g) {
                #pragma unroll
                for (int32_t v31_i1 = 0; v31_i1 < 8; ++v31_i1) {
                  float v36_data = glb_m1[(v21_lead + (v31_i1 * 8))];
                  r1[v31_i1] = v36_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              // wait(r1 = load{g>r}(glb_m1););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              float v39_data = r0[0];
              float v40_data = r1[0];
              float v43_data = r2[0];
              r2[0] = (v43_data + (v39_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v46_data = r1[1];
              float v49_data = r2[1];
              r2[1] = (v49_data + (v39_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v52_data = r1[2];
              float v55_data = r2[2];
              r2[2] = (v55_data + (v39_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v58_data = r1[3];
              float v61_data = r2[3];
              r2[3] = (v61_data + (v39_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v64_data = r1[4];
              float v67_data = r2[4];
              r2[4] = (v67_data + (v39_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v70_data = r1[5];
              float v73_data = r2[5];
              r2[5] = (v73_data + (v39_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v76_data = r1[6];
              float v79_data = r2[6];
              r2[6] = (v79_data + (v39_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v82_data = r1[7];
              float v85_data = r2[7];
              r2[7] = (v85_data + (v39_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v87_data = r0[1];
              float v91_data = r2[0];
              r2[0] = (v91_data + (v87_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v97_data = r2[1];
              r2[1] = (v97_data + (v87_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v103_data = r2[2];
              r2[2] = (v103_data + (v87_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v109_data = r2[3];
              r2[3] = (v109_data + (v87_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v115_data = r2[4];
              r2[4] = (v115_data + (v87_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v121_data = r2[5];
              r2[5] = (v121_data + (v87_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v127_data = r2[6];
              r2[6] = (v127_data + (v87_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v133_data = r2[7];
              r2[7] = (v133_data + (v87_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v135_data = r0[2];
              float v139_data = r2[0];
              r2[0] = (v139_data + (v135_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v145_data = r2[1];
              r2[1] = (v145_data + (v135_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v151_data = r2[2];
              r2[2] = (v151_data + (v135_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v157_data = r2[3];
              r2[3] = (v157_data + (v135_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v163_data = r2[4];
              r2[4] = (v163_data + (v135_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v169_data = r2[5];
              r2[5] = (v169_data + (v135_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v175_data = r2[6];
              r2[6] = (v175_data + (v135_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v181_data = r2[7];
              r2[7] = (v181_data + (v135_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v183_data = r0[3];
              float v187_data = r2[0];
              r2[0] = (v187_data + (v183_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v193_data = r2[1];
              r2[1] = (v193_data + (v183_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v199_data = r2[2];
              r2[2] = (v199_data + (v183_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v205_data = r2[3];
              r2[3] = (v205_data + (v183_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v211_data = r2[4];
              r2[4] = (v211_data + (v183_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v217_data = r2[5];
              r2[5] = (v217_data + (v183_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v223_data = r2[6];
              r2[6] = (v223_data + (v183_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v229_data = r2[7];
              r2[7] = (v229_data + (v183_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v231_data = r0[4];
              float v235_data = r2[0];
              r2[0] = (v235_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v241_data = r2[1];
              r2[1] = (v241_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v247_data = r2[2];
              r2[2] = (v247_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v253_data = r2[3];
              r2[3] = (v253_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v259_data = r2[4];
              r2[4] = (v259_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v265_data = r2[5];
              r2[5] = (v265_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v271_data = r2[6];
              r2[6] = (v271_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v277_data = r2[7];
              r2[7] = (v277_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v279_data = r0[5];
              float v283_data = r2[0];
              r2[0] = (v283_data + (v279_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v289_data = r2[1];
              r2[1] = (v289_data + (v279_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v295_data = r2[2];
              r2[2] = (v295_data + (v279_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v301_data = r2[3];
              r2[3] = (v301_data + (v279_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v307_data = r2[4];
              r2[4] = (v307_data + (v279_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v313_data = r2[5];
              r2[5] = (v313_data + (v279_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v319_data = r2[6];
              r2[6] = (v319_data + (v279_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v325_data = r2[7];
              r2[7] = (v325_data + (v279_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v327_data = r0[6];
              float v331_data = r2[0];
              r2[0] = (v331_data + (v327_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v337_data = r2[1];
              r2[1] = (v337_data + (v327_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v343_data = r2[2];
              r2[2] = (v343_data + (v327_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v349_data = r2[3];
              r2[3] = (v349_data + (v327_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v355_data = r2[4];
              r2[4] = (v355_data + (v327_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v361_data = r2[5];
              r2[5] = (v361_data + (v327_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v367_data = r2[6];
              r2[6] = (v367_data + (v327_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v373_data = r2[7];
              r2[7] = (v373_data + (v327_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v375_data = r0[7];
              float v379_data = r2[0];
              r2[0] = (v379_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v385_data = r2[1];
              r2[1] = (v385_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v391_data = r2[2];
              r2[2] = (v391_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v397_data = r2[3];
              r2[3] = (v397_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v403_data = r2[4];
              r2[4] = (v403_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v409_data = r2[5];
              r2[5] = (v409_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v415_data = r2[6];
              r2[6] = (v415_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v421_data = r2[7];
              r2[7] = (v421_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              // glb_m2 = +(r2, dims=[1])
              if (v22_g) {
                float v424_acc0 = 0.0f;
                #pragma unroll
                for (int32_t v423_r1 = 0; v423_r1 < 8; ++v423_r1) {
                  float v426_data = r2[v423_r1];
                  v424_acc0 = (v424_acc0 + v426_data);
                }
                glb_m2[v21_lead] = v424_acc0;
              }
              sycl::group_barrier(item.get_sub_group());
            }
          }
        }
      });
    }
  });
}

