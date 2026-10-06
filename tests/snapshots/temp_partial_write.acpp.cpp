// === base name ===
kernel_6b5ee4b736daaf75

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_6b5ee4b736daaf75 = {{16, 16, 1}, 16, 12, 1, 16, 10240, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_6b5ee4b736daaf75(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_6b5ee4b736daaf75(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_6b5ee4b736daaf75(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 2560 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_6b5ee4b736daaf75(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_6b5ee4b736daaf75(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_6b5ee4b736daaf75(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_6b5ee4b736daaf75(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (2560, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 10240 B shared, occupancy grid
        // operands:
        //   m0 32×32(12×12) {0..12}×{0..12} strided
        //   m1 32×32(12×12) {0..12}×{0..12} strided
        //   m2 32×32(12×12) {0..12}×{0..12} strided
        //   m3 32×32(12×12) {0..12}×{0..12} strided
        // operations:
        //   t0[i,j]@{0..12}×{0..6} = m0[i,k] × m1[k,j]
        //   m2[i,j] = m3[i,k] × t0[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2560}],"shared_bytes":10240,"shared_elements":2560,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B","bbox":[[0,0],[12,12]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,6]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]},{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[160 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[144];
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v10_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v10_batchId0 < numElements0; v10_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v11_ahead1 = v10_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v13_batchId1 = (v11_ahead1 < numElements0) ? v11_ahead1 : v10_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v10_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v10_batchId0 * 144 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v10_batchId0 * 144 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v10_batchId0 * 144 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v10_batchId0 * 144 + 0 + m3_extraOffset];
              float r0[12]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v25_lead = item.get_local_id(2) % 16;
              bool v26_g = v25_lead < 12;
              if (v26_g) {
                #pragma unroll
                for (int32_t v27_i1 = 0; v27_i1 < 12; ++v27_i1) {
                  float v32_data = glb_m0[(v25_lead + (v27_i1 * 12))];
                  r0[v27_i1] = v32_data;
                }
              }
              float r1[12]{};
              // r1 = load{g>r}(glb_m1);
              if (v26_g) {
                #pragma unroll
                for (int32_t v35_i1 = 0; v35_i1 < 12; ++v35_i1) {
                  float v40_data = glb_m1[(v25_lead + (v35_i1 * 12))];
                  r1[v35_i1] = v40_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r3[12]{};
              // r3 = load{g>r}(glb_m3);
              if (v26_g) {
                #pragma unroll
                for (int32_t v43_i1 = 0; v43_i1 < 12; ++v43_i1) {
                  float v48_data = glb_m3[(v25_lead + (v43_i1 * 12))];
                  r3[v43_i1] = v48_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m1););
              float r2[6]{};
              // r2 = +(r0 * r1) + None
              // [(0, 12), (0, 6)] [(0, 12)]
              float v51_data = r0[0];
              float v52_data = r1[0];
              float v55_data = r2[0];
              r2[0] = (v55_data + (v51_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v58_data = r1[1];
              float v61_data = r2[1];
              r2[1] = (v61_data + (v51_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v64_data = r1[2];
              float v67_data = r2[2];
              r2[2] = (v67_data + (v51_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v70_data = r1[3];
              float v73_data = r2[3];
              r2[3] = (v73_data + (v51_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v76_data = r1[4];
              float v79_data = r2[4];
              r2[4] = (v79_data + (v51_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v82_data = r1[5];
              float v85_data = r2[5];
              r2[5] = (v85_data + (v51_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v87_data = r0[1];
              float v91_data = r2[0];
              r2[0] = (v91_data + (v87_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v97_data = r2[1];
              r2[1] = (v97_data + (v87_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v103_data = r2[2];
              r2[2] = (v103_data + (v87_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v109_data = r2[3];
              r2[3] = (v109_data + (v87_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v115_data = r2[4];
              r2[4] = (v115_data + (v87_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v121_data = r2[5];
              r2[5] = (v121_data + (v87_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v123_data = r0[2];
              float v127_data = r2[0];
              r2[0] = (v127_data + (v123_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v133_data = r2[1];
              r2[1] = (v133_data + (v123_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v139_data = r2[2];
              r2[2] = (v139_data + (v123_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v145_data = r2[3];
              r2[3] = (v145_data + (v123_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v151_data = r2[4];
              r2[4] = (v151_data + (v123_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v157_data = r2[5];
              r2[5] = (v157_data + (v123_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v159_data = r0[3];
              float v163_data = r2[0];
              r2[0] = (v163_data + (v159_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v169_data = r2[1];
              r2[1] = (v169_data + (v159_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v175_data = r2[2];
              r2[2] = (v175_data + (v159_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v181_data = r2[3];
              r2[3] = (v181_data + (v159_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v187_data = r2[4];
              r2[4] = (v187_data + (v159_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v193_data = r2[5];
              r2[5] = (v193_data + (v159_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v195_data = r0[4];
              float v199_data = r2[0];
              r2[0] = (v199_data + (v195_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v205_data = r2[1];
              r2[1] = (v205_data + (v195_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v211_data = r2[2];
              r2[2] = (v211_data + (v195_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v217_data = r2[3];
              r2[3] = (v217_data + (v195_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v223_data = r2[4];
              r2[4] = (v223_data + (v195_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v229_data = r2[5];
              r2[5] = (v229_data + (v195_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v231_data = r0[5];
              float v235_data = r2[0];
              r2[0] = (v235_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v241_data = r2[1];
              r2[1] = (v241_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v247_data = r2[2];
              r2[2] = (v247_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v253_data = r2[3];
              r2[3] = (v253_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v259_data = r2[4];
              r2[4] = (v259_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v265_data = r2[5];
              r2[5] = (v265_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v267_data = r0[6];
              float v271_data = r2[0];
              r2[0] = (v271_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v277_data = r2[1];
              r2[1] = (v277_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v283_data = r2[2];
              r2[2] = (v283_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v289_data = r2[3];
              r2[3] = (v289_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v295_data = r2[4];
              r2[4] = (v295_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v301_data = r2[5];
              r2[5] = (v301_data + (v267_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v303_data = r0[7];
              float v307_data = r2[0];
              r2[0] = (v307_data + (v303_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v313_data = r2[1];
              r2[1] = (v313_data + (v303_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v319_data = r2[2];
              r2[2] = (v319_data + (v303_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v325_data = r2[3];
              r2[3] = (v325_data + (v303_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v331_data = r2[4];
              r2[4] = (v331_data + (v303_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v337_data = r2[5];
              r2[5] = (v337_data + (v303_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v339_data = r0[8];
              float v343_data = r2[0];
              r2[0] = (v343_data + (v339_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v349_data = r2[1];
              r2[1] = (v349_data + (v339_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v355_data = r2[2];
              r2[2] = (v355_data + (v339_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v361_data = r2[3];
              r2[3] = (v361_data + (v339_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v367_data = r2[4];
              r2[4] = (v367_data + (v339_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v373_data = r2[5];
              r2[5] = (v373_data + (v339_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v375_data = r0[9];
              float v379_data = r2[0];
              r2[0] = (v379_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v385_data = r2[1];
              r2[1] = (v385_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v391_data = r2[2];
              r2[2] = (v391_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v397_data = r2[3];
              r2[3] = (v397_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v403_data = r2[4];
              r2[4] = (v403_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v409_data = r2[5];
              r2[5] = (v409_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v411_data = r0[10];
              float v415_data = r2[0];
              r2[0] = (v415_data + (v411_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v421_data = r2[1];
              r2[1] = (v421_data + (v411_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v427_data = r2[2];
              r2[2] = (v427_data + (v411_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v433_data = r2[3];
              r2[3] = (v433_data + (v411_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v439_data = r2[4];
              r2[4] = (v439_data + (v411_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v445_data = r2[5];
              r2[5] = (v445_data + (v411_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v447_data = r0[11];
              float v451_data = r2[0];
              r2[0] = (v451_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v457_data = r2[1];
              r2[1] = (v457_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v463_data = r2[2];
              r2[2] = (v463_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v469_data = r2[3];
              r2[3] = (v469_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v475_data = r2[4];
              r2[4] = (v475_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v481_data = r2[5];
              r2[5] = (v481_data + (v447_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              // s0 = store{r>s, clear}(localShrMem0, r2);
              if (v26_g) {
                #pragma unroll
                for (int32_t v483_z1 = 6; v483_z1 < 12; ++v483_z1) {
                  int32_t v488_a = v25_lead + (v483_z1 * 12);
                  s0[(v488_a ^ ((v488_a >> 4) & 15))] = 0.0f;
                }
              }
              if (v26_g) {
                #pragma unroll
                for (int32_t v492_i1 = 0; v492_i1 < 6; ++v492_i1) {
                  float v494_data = r2[v492_i1];
                  int32_t v498_a = v25_lead + (v492_i1 * 12);
                  s0[(v498_a ^ ((v498_a >> 4) & 15))] = v494_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m3););
              float r4[12]{};
              sycl::group_barrier(item.get_sub_group());
              // ir4 = +(r3 * s0)
              // [(0, 12), (0, 12)] [(0, 12)]
              float ir4[12]{};
              float v504_data = r3[0];
              float v505_data = s0[0];
              float v507_data = ir4[0];
              ir4[0] = (v507_data + (v504_data * v505_data));
              float v510_data = s0[12];
              float v512_data = ir4[1];
              ir4[1] = (v512_data + (v504_data * v510_data));
              float v515_data = s0[25];
              float v517_data = ir4[2];
              ir4[2] = (v517_data + (v504_data * v515_data));
              float v520_data = s0[38];
              float v522_data = ir4[3];
              ir4[3] = (v522_data + (v504_data * v520_data));
              float v525_data = s0[51];
              float v527_data = ir4[4];
              ir4[4] = (v527_data + (v504_data * v525_data));
              float v530_data = s0[63];
              float v532_data = ir4[5];
              ir4[5] = (v532_data + (v504_data * v530_data));
              float v535_data = s0[76];
              float v537_data = ir4[6];
              ir4[6] = (v537_data + (v504_data * v535_data));
              float v540_data = s0[81];
              float v542_data = ir4[7];
              ir4[7] = (v542_data + (v504_data * v540_data));
              float v545_data = s0[102];
              float v547_data = ir4[8];
              ir4[8] = (v547_data + (v504_data * v545_data));
              float v550_data = s0[106];
              float v552_data = ir4[9];
              ir4[9] = (v552_data + (v504_data * v550_data));
              float v555_data = s0[127];
              float v557_data = ir4[10];
              ir4[10] = (v557_data + (v504_data * v555_data));
              float v560_data = s0[140];
              float v562_data = ir4[11];
              ir4[11] = (v562_data + (v504_data * v560_data));
              float v564_data = r3[1];
              float v565_data = s0[1];
              float v567_data = ir4[0];
              ir4[0] = (v567_data + (v564_data * v565_data));
              float v570_data = s0[13];
              float v572_data = ir4[1];
              ir4[1] = (v572_data + (v564_data * v570_data));
              float v575_data = s0[24];
              float v577_data = ir4[2];
              ir4[2] = (v577_data + (v564_data * v575_data));
              float v580_data = s0[39];
              float v582_data = ir4[3];
              ir4[3] = (v582_data + (v564_data * v580_data));
              float v585_data = s0[50];
              float v587_data = ir4[4];
              ir4[4] = (v587_data + (v564_data * v585_data));
              float v590_data = s0[62];
              float v592_data = ir4[5];
              ir4[5] = (v592_data + (v564_data * v590_data));
              float v595_data = s0[77];
              float v597_data = ir4[6];
              ir4[6] = (v597_data + (v564_data * v595_data));
              float v600_data = s0[80];
              float v602_data = ir4[7];
              ir4[7] = (v602_data + (v564_data * v600_data));
              float v605_data = s0[103];
              float v607_data = ir4[8];
              ir4[8] = (v607_data + (v564_data * v605_data));
              float v610_data = s0[107];
              float v612_data = ir4[9];
              ir4[9] = (v612_data + (v564_data * v610_data));
              float v615_data = s0[126];
              float v617_data = ir4[10];
              ir4[10] = (v617_data + (v564_data * v615_data));
              float v620_data = s0[141];
              float v622_data = ir4[11];
              ir4[11] = (v622_data + (v564_data * v620_data));
              float v624_data = r3[2];
              float v625_data = s0[2];
              float v627_data = ir4[0];
              ir4[0] = (v627_data + (v624_data * v625_data));
              float v630_data = s0[14];
              float v632_data = ir4[1];
              ir4[1] = (v632_data + (v624_data * v630_data));
              float v635_data = s0[27];
              float v637_data = ir4[2];
              ir4[2] = (v637_data + (v624_data * v635_data));
              float v640_data = s0[36];
              float v642_data = ir4[3];
              ir4[3] = (v642_data + (v624_data * v640_data));
              float v645_data = s0[49];
              float v647_data = ir4[4];
              ir4[4] = (v647_data + (v624_data * v645_data));
              float v650_data = s0[61];
              float v652_data = ir4[5];
              ir4[5] = (v652_data + (v624_data * v650_data));
              float v655_data = s0[78];
              float v657_data = ir4[6];
              ir4[6] = (v657_data + (v624_data * v655_data));
              float v660_data = s0[83];
              float v662_data = ir4[7];
              ir4[7] = (v662_data + (v624_data * v660_data));
              float v665_data = s0[100];
              float v667_data = ir4[8];
              ir4[8] = (v667_data + (v624_data * v665_data));
              float v670_data = s0[104];
              float v672_data = ir4[9];
              ir4[9] = (v672_data + (v624_data * v670_data));
              float v675_data = s0[125];
              float v677_data = ir4[10];
              ir4[10] = (v677_data + (v624_data * v675_data));
              float v680_data = s0[142];
              float v682_data = ir4[11];
              ir4[11] = (v682_data + (v624_data * v680_data));
              float v684_data = r3[3];
              float v685_data = s0[3];
              float v687_data = ir4[0];
              ir4[0] = (v687_data + (v684_data * v685_data));
              float v690_data = s0[15];
              float v692_data = ir4[1];
              ir4[1] = (v692_data + (v684_data * v690_data));
              float v695_data = s0[26];
              float v697_data = ir4[2];
              ir4[2] = (v697_data + (v684_data * v695_data));
              float v700_data = s0[37];
              float v702_data = ir4[3];
              ir4[3] = (v702_data + (v684_data * v700_data));
              float v705_data = s0[48];
              float v707_data = ir4[4];
              ir4[4] = (v707_data + (v684_data * v705_data));
              float v710_data = s0[60];
              float v712_data = ir4[5];
              ir4[5] = (v712_data + (v684_data * v710_data));
              float v715_data = s0[79];
              float v717_data = ir4[6];
              ir4[6] = (v717_data + (v684_data * v715_data));
              float v720_data = s0[82];
              float v722_data = ir4[7];
              ir4[7] = (v722_data + (v684_data * v720_data));
              float v725_data = s0[101];
              float v727_data = ir4[8];
              ir4[8] = (v727_data + (v684_data * v725_data));
              float v730_data = s0[105];
              float v732_data = ir4[9];
              ir4[9] = (v732_data + (v684_data * v730_data));
              float v735_data = s0[124];
              float v737_data = ir4[10];
              ir4[10] = (v737_data + (v684_data * v735_data));
              float v740_data = s0[143];
              float v742_data = ir4[11];
              ir4[11] = (v742_data + (v684_data * v740_data));
              float v744_data = r3[4];
              float v745_data = s0[4];
              float v747_data = ir4[0];
              ir4[0] = (v747_data + (v744_data * v745_data));
              float v750_data = s0[17];
              float v752_data = ir4[1];
              ir4[1] = (v752_data + (v744_data * v750_data));
              float v755_data = s0[29];
              float v757_data = ir4[2];
              ir4[2] = (v757_data + (v744_data * v755_data));
              float v760_data = s0[42];
              float v762_data = ir4[3];
              ir4[3] = (v762_data + (v744_data * v760_data));
              float v765_data = s0[55];
              float v767_data = ir4[4];
              ir4[4] = (v767_data + (v744_data * v765_data));
              float v770_data = s0[68];
              float v772_data = ir4[5];
              ir4[5] = (v772_data + (v744_data * v770_data));
              float v775_data = s0[72];
              float v777_data = ir4[6];
              ir4[6] = (v777_data + (v744_data * v775_data));
              float v780_data = s0[93];
              float v782_data = ir4[7];
              ir4[7] = (v782_data + (v744_data * v780_data));
              float v785_data = s0[98];
              float v787_data = ir4[8];
              ir4[8] = (v787_data + (v744_data * v785_data));
              float v790_data = s0[119];
              float v792_data = ir4[9];
              ir4[9] = (v792_data + (v744_data * v790_data));
              float v795_data = s0[123];
              float v797_data = ir4[10];
              ir4[10] = (v797_data + (v744_data * v795_data));
              float v800_data = s0[128];
              float v802_data = ir4[11];
              ir4[11] = (v802_data + (v744_data * v800_data));
              float v804_data = r3[5];
              float v805_data = s0[5];
              float v807_data = ir4[0];
              ir4[0] = (v807_data + (v804_data * v805_data));
              float v810_data = s0[16];
              float v812_data = ir4[1];
              ir4[1] = (v812_data + (v804_data * v810_data));
              float v815_data = s0[28];
              float v817_data = ir4[2];
              ir4[2] = (v817_data + (v804_data * v815_data));
              float v820_data = s0[43];
              float v822_data = ir4[3];
              ir4[3] = (v822_data + (v804_data * v820_data));
              float v825_data = s0[54];
              float v827_data = ir4[4];
              ir4[4] = (v827_data + (v804_data * v825_data));
              float v830_data = s0[69];
              float v832_data = ir4[5];
              ir4[5] = (v832_data + (v804_data * v830_data));
              float v835_data = s0[73];
              float v837_data = ir4[6];
              ir4[6] = (v837_data + (v804_data * v835_data));
              float v840_data = s0[92];
              float v842_data = ir4[7];
              ir4[7] = (v842_data + (v804_data * v840_data));
              float v845_data = s0[99];
              float v847_data = ir4[8];
              ir4[8] = (v847_data + (v804_data * v845_data));
              float v850_data = s0[118];
              float v852_data = ir4[9];
              ir4[9] = (v852_data + (v804_data * v850_data));
              float v855_data = s0[122];
              float v857_data = ir4[10];
              ir4[10] = (v857_data + (v804_data * v855_data));
              float v860_data = s0[129];
              float v862_data = ir4[11];
              ir4[11] = (v862_data + (v804_data * v860_data));
              float v864_data = r3[6];
              float v865_data = s0[6];
              float v867_data = ir4[0];
              ir4[0] = (v867_data + (v864_data * v865_data));
              float v870_data = s0[19];
              float v872_data = ir4[1];
              ir4[1] = (v872_data + (v864_data * v870_data));
              float v875_data = s0[31];
              float v877_data = ir4[2];
              ir4[2] = (v877_data + (v864_data * v875_data));
              float v880_data = s0[40];
              float v882_data = ir4[3];
              ir4[3] = (v882_data + (v864_data * v880_data));
              float v885_data = s0[53];
              float v887_data = ir4[4];
              ir4[4] = (v887_data + (v864_data * v885_data));
              float v890_data = s0[70];
              float v892_data = ir4[5];
              ir4[5] = (v892_data + (v864_data * v890_data));
              float v895_data = s0[74];
              float v897_data = ir4[6];
              ir4[6] = (v897_data + (v864_data * v895_data));
              float v900_data = s0[95];
              float v902_data = ir4[7];
              ir4[7] = (v902_data + (v864_data * v900_data));
              float v905_data = s0[96];
              float v907_data = ir4[8];
              ir4[8] = (v907_data + (v864_data * v905_data));
              float v910_data = s0[117];
              float v912_data = ir4[9];
              ir4[9] = (v912_data + (v864_data * v910_data));
              float v915_data = s0[121];
              float v917_data = ir4[10];
              ir4[10] = (v917_data + (v864_data * v915_data));
              float v920_data = s0[130];
              float v922_data = ir4[11];
              ir4[11] = (v922_data + (v864_data * v920_data));
              float v924_data = r3[7];
              float v925_data = s0[7];
              float v927_data = ir4[0];
              ir4[0] = (v927_data + (v924_data * v925_data));
              float v930_data = s0[18];
              float v932_data = ir4[1];
              ir4[1] = (v932_data + (v924_data * v930_data));
              float v935_data = s0[30];
              float v937_data = ir4[2];
              ir4[2] = (v937_data + (v924_data * v935_data));
              float v940_data = s0[41];
              float v942_data = ir4[3];
              ir4[3] = (v942_data + (v924_data * v940_data));
              float v945_data = s0[52];
              float v947_data = ir4[4];
              ir4[4] = (v947_data + (v924_data * v945_data));
              float v950_data = s0[71];
              float v952_data = ir4[5];
              ir4[5] = (v952_data + (v924_data * v950_data));
              float v955_data = s0[75];
              float v957_data = ir4[6];
              ir4[6] = (v957_data + (v924_data * v955_data));
              float v960_data = s0[94];
              float v962_data = ir4[7];
              ir4[7] = (v962_data + (v924_data * v960_data));
              float v965_data = s0[97];
              float v967_data = ir4[8];
              ir4[8] = (v967_data + (v924_data * v965_data));
              float v970_data = s0[116];
              float v972_data = ir4[9];
              ir4[9] = (v972_data + (v924_data * v970_data));
              float v975_data = s0[120];
              float v977_data = ir4[10];
              ir4[10] = (v977_data + (v924_data * v975_data));
              float v980_data = s0[131];
              float v982_data = ir4[11];
              ir4[11] = (v982_data + (v924_data * v980_data));
              float v984_data = r3[8];
              float v985_data = s0[8];
              float v987_data = ir4[0];
              ir4[0] = (v987_data + (v984_data * v985_data));
              float v990_data = s0[21];
              float v992_data = ir4[1];
              ir4[1] = (v992_data + (v984_data * v990_data));
              float v995_data = s0[34];
              float v997_data = ir4[2];
              ir4[2] = (v997_data + (v984_data * v995_data));
              float v1000_data = s0[46];
              float v1002_data = ir4[3];
              ir4[3] = (v1002_data + (v984_data * v1000_data));
              float v1005_data = s0[59];
              float v1007_data = ir4[4];
              ir4[4] = (v1007_data + (v984_data * v1005_data));
              float v1010_data = s0[64];
              float v1012_data = ir4[5];
              ir4[5] = (v1012_data + (v984_data * v1010_data));
              float v1015_data = s0[85];
              float v1017_data = ir4[6];
              ir4[6] = (v1017_data + (v984_data * v1015_data));
              float v1020_data = s0[89];
              float v1022_data = ir4[7];
              ir4[7] = (v1022_data + (v984_data * v1020_data));
              float v1025_data = s0[110];
              float v1027_data = ir4[8];
              ir4[8] = (v1027_data + (v984_data * v1025_data));
              float v1030_data = s0[115];
              float v1032_data = ir4[9];
              ir4[9] = (v1032_data + (v984_data * v1030_data));
              float v1035_data = s0[136];
              float v1037_data = ir4[10];
              ir4[10] = (v1037_data + (v984_data * v1035_data));
              float v1040_data = s0[132];
              float v1042_data = ir4[11];
              ir4[11] = (v1042_data + (v984_data * v1040_data));
              float v1044_data = r3[9];
              float v1045_data = s0[9];
              float v1047_data = ir4[0];
              ir4[0] = (v1047_data + (v1044_data * v1045_data));
              float v1050_data = s0[20];
              float v1052_data = ir4[1];
              ir4[1] = (v1052_data + (v1044_data * v1050_data));
              float v1055_data = s0[35];
              float v1057_data = ir4[2];
              ir4[2] = (v1057_data + (v1044_data * v1055_data));
              float v1060_data = s0[47];
              float v1062_data = ir4[3];
              ir4[3] = (v1062_data + (v1044_data * v1060_data));
              float v1065_data = s0[58];
              float v1067_data = ir4[4];
              ir4[4] = (v1067_data + (v1044_data * v1065_data));
              float v1070_data = s0[65];
              float v1072_data = ir4[5];
              ir4[5] = (v1072_data + (v1044_data * v1070_data));
              float v1075_data = s0[84];
              float v1077_data = ir4[6];
              ir4[6] = (v1077_data + (v1044_data * v1075_data));
              float v1080_data = s0[88];
              float v1082_data = ir4[7];
              ir4[7] = (v1082_data + (v1044_data * v1080_data));
              float v1085_data = s0[111];
              float v1087_data = ir4[8];
              ir4[8] = (v1087_data + (v1044_data * v1085_data));
              float v1090_data = s0[114];
              float v1092_data = ir4[9];
              ir4[9] = (v1092_data + (v1044_data * v1090_data));
              float v1095_data = s0[137];
              float v1097_data = ir4[10];
              ir4[10] = (v1097_data + (v1044_data * v1095_data));
              float v1100_data = s0[133];
              float v1102_data = ir4[11];
              ir4[11] = (v1102_data + (v1044_data * v1100_data));
              float v1104_data = r3[10];
              float v1105_data = s0[10];
              float v1107_data = ir4[0];
              ir4[0] = (v1107_data + (v1104_data * v1105_data));
              float v1110_data = s0[23];
              float v1112_data = ir4[1];
              ir4[1] = (v1112_data + (v1104_data * v1110_data));
              float v1115_data = s0[32];
              float v1117_data = ir4[2];
              ir4[2] = (v1117_data + (v1104_data * v1115_data));
              float v1120_data = s0[44];
              float v1122_data = ir4[3];
              ir4[3] = (v1122_data + (v1104_data * v1120_data));
              float v1125_data = s0[57];
              float v1127_data = ir4[4];
              ir4[4] = (v1127_data + (v1104_data * v1125_data));
              float v1130_data = s0[66];
              float v1132_data = ir4[5];
              ir4[5] = (v1132_data + (v1104_data * v1130_data));
              float v1135_data = s0[87];
              float v1137_data = ir4[6];
              ir4[6] = (v1137_data + (v1104_data * v1135_data));
              float v1140_data = s0[91];
              float v1142_data = ir4[7];
              ir4[7] = (v1142_data + (v1104_data * v1140_data));
              float v1145_data = s0[108];
              float v1147_data = ir4[8];
              ir4[8] = (v1147_data + (v1104_data * v1145_data));
              float v1150_data = s0[113];
              float v1152_data = ir4[9];
              ir4[9] = (v1152_data + (v1104_data * v1150_data));
              float v1155_data = s0[138];
              float v1157_data = ir4[10];
              ir4[10] = (v1157_data + (v1104_data * v1155_data));
              float v1160_data = s0[134];
              float v1162_data = ir4[11];
              ir4[11] = (v1162_data + (v1104_data * v1160_data));
              float v1164_data = r3[11];
              float v1165_data = s0[11];
              float v1167_data = ir4[0];
              ir4[0] = (v1167_data + (v1164_data * v1165_data));
              float v1170_data = s0[22];
              float v1172_data = ir4[1];
              ir4[1] = (v1172_data + (v1164_data * v1170_data));
              float v1175_data = s0[33];
              float v1177_data = ir4[2];
              ir4[2] = (v1177_data + (v1164_data * v1175_data));
              float v1180_data = s0[45];
              float v1182_data = ir4[3];
              ir4[3] = (v1182_data + (v1164_data * v1180_data));
              float v1185_data = s0[56];
              float v1187_data = ir4[4];
              ir4[4] = (v1187_data + (v1164_data * v1185_data));
              float v1190_data = s0[67];
              float v1192_data = ir4[5];
              ir4[5] = (v1192_data + (v1164_data * v1190_data));
              float v1195_data = s0[86];
              float v1197_data = ir4[6];
              ir4[6] = (v1197_data + (v1164_data * v1195_data));
              float v1200_data = s0[90];
              float v1202_data = ir4[7];
              ir4[7] = (v1202_data + (v1164_data * v1200_data));
              float v1205_data = s0[109];
              float v1207_data = ir4[8];
              ir4[8] = (v1207_data + (v1164_data * v1205_data));
              float v1210_data = s0[112];
              float v1212_data = ir4[9];
              ir4[9] = (v1212_data + (v1164_data * v1210_data));
              float v1215_data = s0[139];
              float v1217_data = ir4[10];
              ir4[10] = (v1217_data + (v1164_data * v1215_data));
              float v1220_data = s0[135];
              float v1222_data = ir4[11];
              ir4[11] = (v1222_data + (v1164_data * v1220_data));
              // r4 = ir4
              if (v26_g) {
                #pragma unroll
                for (int32_t v1224_n1 = 0; v1224_n1 < 12; ++v1224_n1) {
                  float v1226_data = ir4[v1224_n1];
                  r4[v1224_n1] = v1226_data;
                }
              }
              // glb_m2 = store{r>g}(r4);
              if (v26_g) {
                #pragma unroll
                for (int32_t v1227_i1 = 0; v1227_i1 < 12; ++v1227_i1) {
                  float v1229_data = r4[v1227_i1];
                  glb_m2[(v25_lead + (v1227_i1 * 12))] = v1229_data;
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

