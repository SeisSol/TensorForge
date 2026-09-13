// === base name ===
kernel_7446ff07e5e8a416

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_7446ff07e5e8a416 = {{16, 16, 1}, 16, 16, 1, 16, 5120, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_7446ff07e5e8a416(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_7446ff07e5e8a416(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_7446ff07e5e8a416(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 1280 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_7446ff07e5e8a416(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_7446ff07e5e8a416(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_7446ff07e5e8a416(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_7446ff07e5e8a416(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 5120 B shared, occupancy grid
        // operands:
        //   m0 8×8(8×8) {0..8}×{0..8} strided
        //   m1 8×4(8×4) {0..8}×{0..4} strided
        //   m2 8×4(8×4) {0..8}×{0..4} strided
        //   m3 8×8(8×8) {0..8}×{0..8} strided
        // operations:
        //   t0[i,j]@{0..8}×{0..4} = m0[i,k] × m1[k,j]
        //   t0[i,j]@{0..8}×{4..8} = m0[i,k] × m2[k,j]
        //   C = abs(TMP)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1280}],"shared_bytes":5120,"shared_elements":1280,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[8,4]],"name":"m1","ordered":false,"parts":1,"shape":[8,4],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[8,4]],"name":"m2","ordered":false,"parts":1,"shape":[8,4],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m3","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,4]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,4]],"is_tmp":true,"name":"t0","offset":[0,4],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[80 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[64];
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v4_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v4_batchId0 < numElements0; v4_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v5_ahead1 = v4_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v4_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v4_batchId0 * 32 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v4_batchId0 * 32 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v4_batchId0 * 64 + 0 + m3_extraOffset];
              float r0[8]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v19_lead = item.get_local_id(2) % 16;
              bool v20_g = v19_lead < 8;
              if (v20_g) {
                #pragma unroll
                for (int32_t v21_i1 = 0; v21_i1 < 8; ++v21_i1) {
                  float v26_data = glb_m0[(v19_lead + (v21_i1 * 8))];
                  r0[v21_i1] = v26_data;
                }
              }
              float r1[4]{};
              // r1 = load{g>r}(glb_m1);
              if (v20_g) {
                #pragma unroll
                for (int32_t v29_i1 = 0; v29_i1 < 4; ++v29_i1) {
                  float v34_data = glb_m1[(v19_lead + (v29_i1 * 8))];
                  r1[v29_i1] = v34_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r3[4]{};
              // r3 = load{g>r}(glb_m2);
              if (v20_g) {
                #pragma unroll
                for (int32_t v37_i1 = 0; v37_i1 < 4; ++v37_i1) {
                  float v42_data = glb_m2[(v19_lead + (v37_i1 * 8))];
                  r3[v37_i1] = v42_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m1););
              float r2[4]{};
              // r2 = +(r0 * r1) + None
              // [(0, 8), (0, 4)] [(0, 8)]
              float v45_data = r0[0];
              float v46_data = r1[0];
              float v49_data = r2[0];
              r2[0] = (v49_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v52_data = r1[1];
              float v55_data = r2[1];
              r2[1] = (v55_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v58_data = r1[2];
              float v61_data = r2[2];
              r2[2] = (v61_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v64_data = r1[3];
              float v67_data = r2[3];
              r2[3] = (v67_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v69_data = r0[1];
              float v73_data = r2[0];
              r2[0] = (v73_data + (v69_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v79_data = r2[1];
              r2[1] = (v79_data + (v69_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v85_data = r2[2];
              r2[2] = (v85_data + (v69_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v91_data = r2[3];
              r2[3] = (v91_data + (v69_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v93_data = r0[2];
              float v97_data = r2[0];
              r2[0] = (v97_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v103_data = r2[1];
              r2[1] = (v103_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v109_data = r2[2];
              r2[2] = (v109_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v115_data = r2[3];
              r2[3] = (v115_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v117_data = r0[3];
              float v121_data = r2[0];
              r2[0] = (v121_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v127_data = r2[1];
              r2[1] = (v127_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v133_data = r2[2];
              r2[2] = (v133_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v139_data = r2[3];
              r2[3] = (v139_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v141_data = r0[4];
              float v145_data = r2[0];
              r2[0] = (v145_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v151_data = r2[1];
              r2[1] = (v151_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v157_data = r2[2];
              r2[2] = (v157_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v163_data = r2[3];
              r2[3] = (v163_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v165_data = r0[5];
              float v169_data = r2[0];
              r2[0] = (v169_data + (v165_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v175_data = r2[1];
              r2[1] = (v175_data + (v165_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v181_data = r2[2];
              r2[2] = (v181_data + (v165_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v187_data = r2[3];
              r2[3] = (v187_data + (v165_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v189_data = r0[6];
              float v193_data = r2[0];
              r2[0] = (v193_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v199_data = r2[1];
              r2[1] = (v199_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v205_data = r2[2];
              r2[2] = (v205_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v211_data = r2[3];
              r2[3] = (v211_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v213_data = r0[7];
              float v217_data = r2[0];
              r2[0] = (v217_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v223_data = r2[1];
              r2[1] = (v223_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v229_data = r2[2];
              r2[2] = (v229_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v235_data = r2[3];
              r2[3] = (v235_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              // s0 = store{r>s}(localShrMem0, r2);
              if (v20_g) {
                #pragma unroll
                for (int32_t v237_i1 = 0; v237_i1 < 4; ++v237_i1) {
                  float v239_data = r2[v237_i1];
                  int32_t v243_a = v19_lead + (v237_i1 * 8);
                  s0[(v243_a ^ ((v243_a >> 5) & 31))] = v239_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m2););
              float r4[4]{};
              // r4 = +(r0 * r3) + None
              // [(0, 8), (0, 4)] [(0, 8)]
              float ir4[4]{};
              float v250_data = r3[0];
              float v253_data = ir4[0];
              ir4[0] = (v253_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v250_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v256_data = r3[1];
              float v259_data = ir4[1];
              ir4[1] = (v259_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v256_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v262_data = r3[2];
              float v265_data = ir4[2];
              ir4[2] = (v265_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v262_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v268_data = r3[3];
              float v271_data = ir4[3];
              ir4[3] = (v271_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v268_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v277_data = ir4[0];
              ir4[0] = (v277_data + (v69_data * (sycl::select_from_group(item.get_sub_group(), v250_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v283_data = ir4[1];
              ir4[1] = (v283_data + (v69_data * (sycl::select_from_group(item.get_sub_group(), v256_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v289_data = ir4[2];
              ir4[2] = (v289_data + (v69_data * (sycl::select_from_group(item.get_sub_group(), v262_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v295_data = ir4[3];
              ir4[3] = (v295_data + (v69_data * (sycl::select_from_group(item.get_sub_group(), v268_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v301_data = ir4[0];
              ir4[0] = (v301_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v250_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v307_data = ir4[1];
              ir4[1] = (v307_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v256_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v313_data = ir4[2];
              ir4[2] = (v313_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v262_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v319_data = ir4[3];
              ir4[3] = (v319_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v268_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v325_data = ir4[0];
              ir4[0] = (v325_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v250_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v331_data = ir4[1];
              ir4[1] = (v331_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v256_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v337_data = ir4[2];
              ir4[2] = (v337_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v262_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v343_data = ir4[3];
              ir4[3] = (v343_data + (v117_data * (sycl::select_from_group(item.get_sub_group(), v268_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v349_data = ir4[0];
              ir4[0] = (v349_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v250_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v355_data = ir4[1];
              ir4[1] = (v355_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v256_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v361_data = ir4[2];
              ir4[2] = (v361_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v262_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v367_data = ir4[3];
              ir4[3] = (v367_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v268_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v373_data = ir4[0];
              ir4[0] = (v373_data + (v165_data * (sycl::select_from_group(item.get_sub_group(), v250_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v379_data = ir4[1];
              ir4[1] = (v379_data + (v165_data * (sycl::select_from_group(item.get_sub_group(), v256_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v385_data = ir4[2];
              ir4[2] = (v385_data + (v165_data * (sycl::select_from_group(item.get_sub_group(), v262_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v391_data = ir4[3];
              ir4[3] = (v391_data + (v165_data * (sycl::select_from_group(item.get_sub_group(), v268_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v397_data = ir4[0];
              ir4[0] = (v397_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v250_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v403_data = ir4[1];
              ir4[1] = (v403_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v256_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v409_data = ir4[2];
              ir4[2] = (v409_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v262_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v415_data = ir4[3];
              ir4[3] = (v415_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v268_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v421_data = ir4[0];
              ir4[0] = (v421_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v250_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v427_data = ir4[1];
              ir4[1] = (v427_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v256_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v433_data = ir4[2];
              ir4[2] = (v433_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v262_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v439_data = ir4[3];
              ir4[3] = (v439_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v268_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              if (v20_g) {
                #pragma unroll
                for (int32_t v441_n1 = 0; v441_n1 < 4; ++v441_n1) {
                  float v443_data = ir4[v441_n1];
                  r4[v441_n1] = v443_data;
                }
              }
              // s0 = store{r>s}(localShrMem0, r4);
              if (v20_g) {
                #pragma unroll
                for (int32_t v444_i1 = 0; v444_i1 < 4; ++v444_i1) {
                  float v446_data = r4[v444_i1];
                  int32_t v451_a = v19_lead + ((v444_i1 + 4) * 8);
                  s0[(v451_a ^ ((v451_a >> 5) & 31))] = v446_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
              // glb_m3 = abs(s0)
              if (v20_g) {
                #pragma unroll
                for (int32_t v455_k1 = 0; v455_k1 < 8; ++v455_k1) {
                  int32_t v459_a = v19_lead + (v455_k1 * 8);
                  float v463_data = s0[(v459_a ^ ((v459_a >> 5) & 31))];
                  glb_m3[v459_a] = (sycl::fabs(v463_data));
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

