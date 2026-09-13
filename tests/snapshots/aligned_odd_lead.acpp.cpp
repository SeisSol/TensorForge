// === base name ===
kernel_b256534b2481fab6

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b256534b2481fab6 = {{32, 1, 1}, 32, 35, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b256534b2481fab6(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b256534b2481fab6(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b256534b2481fab6(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (32, 1, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 32;
  config.block[1] = 1;
  config.block[2] = 1;
  config.sharedMemBytes = 0 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_b256534b2481fab6(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b256534b2481fab6(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_b256534b2481fab6(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_b256534b2481fab6(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes (35 active) x 1 per block = block 32x1x1, 0 B shared, occupancy grid
        // operands:
        //   m0 35×4(35×4) {0..35}×{0..4} strided
        //   m1 35×8(35×8) {0..35}×{0..8} strided
        //   m2 8×4(8×4) {0..8}×{0..4} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":35,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[35,4]],"name":"m0","ordered":false,"parts":1,"shape":[35,4],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[35,8]],"name":"m1","ordered":false,"parts":1,"shape":[35,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,4]],"name":"m2","ordered":false,"parts":1,"shape":[8,4],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[35,4]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[35,4]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[35,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[35,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          for (size_t v1_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v1_batchId0 < numElements0; v1_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v2_ahead1 = v1_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v1_batchId0 * 140 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v1_batchId0 * 280 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v1_batchId0 * 32 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v15_lead = item.get_local_id(2) % 32;
              #pragma unroll
              for (int32_t v16_i0 = 0; v16_i0 < 1; ++v16_i0) {
                int32_t v19_lead = v15_lead + (v16_i0 * 32);
                #pragma unroll
                for (int32_t v17_i1 = 0; v17_i1 < 8; ++v17_i1) {
                  float v22_data = glb_m1[(v19_lead + (v17_i1 * 35))];
                  r0[(v16_i0 + (v17_i1 * 2))] = v22_data;
                }
              }
              bool v25_g = v15_lead < 3;
              if (v25_g) {
                int32_t v28_lead = v15_lead + 32_i32;
                #pragma unroll
                for (int32_t v26_i1 = 0; v26_i1 < 8; ++v26_i1) {
                  float v31_data = glb_m1[(v28_lead + (v26_i1 * 35))];
                  r0[(1 + (v26_i1 * 2))] = v31_data;
                }
              }
              float r1[4]{};
              // r1 = load{g>r}(glb_m2);
              if (v15_lead < 8) {
                #pragma unroll
                for (int32_t v36_i1 = 0; v36_i1 < 4; ++v36_i1) {
                  float v41_data = glb_m2[(v15_lead + (v36_i1 * 8))];
                  r1[v36_i1] = v41_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 35), (0, 4)] [(0, 8)]
              float ir2[8]{};
              float v45_data = r0[0];
              float v46_data = r1[0];
              float v47_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v49_data = ir2[0];
              ir2[0] = (v49_data + (v45_data * v47_bc));
              float v52_data = r1[1];
              float v53_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v55_data = ir2[2];
              ir2[2] = (v55_data + (v45_data * v53_bc));
              float v58_data = r1[2];
              float v59_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v61_data = ir2[4];
              ir2[4] = (v61_data + (v45_data * v59_bc));
              float v64_data = r1[3];
              float v65_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v67_data = ir2[6];
              ir2[6] = (v67_data + (v45_data * v65_bc));
              float v69_data = r0[1];
              float v73_data = ir2[1];
              ir2[1] = (v73_data + (v69_data * v47_bc));
              float v79_data = ir2[3];
              ir2[3] = (v79_data + (v69_data * v53_bc));
              float v85_data = ir2[5];
              ir2[5] = (v85_data + (v69_data * v59_bc));
              float v91_data = ir2[7];
              ir2[7] = (v91_data + (v69_data * v65_bc));
              float v93_data = r0[2];
              float v95_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v97_data = ir2[0];
              ir2[0] = (v97_data + (v93_data * v95_bc));
              float v101_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v103_data = ir2[2];
              ir2[2] = (v103_data + (v93_data * v101_bc));
              float v107_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v109_data = ir2[4];
              ir2[4] = (v109_data + (v93_data * v107_bc));
              float v113_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v115_data = ir2[6];
              ir2[6] = (v115_data + (v93_data * v113_bc));
              float v117_data = r0[3];
              float v121_data = ir2[1];
              ir2[1] = (v121_data + (v117_data * v95_bc));
              float v127_data = ir2[3];
              ir2[3] = (v127_data + (v117_data * v101_bc));
              float v133_data = ir2[5];
              ir2[5] = (v133_data + (v117_data * v107_bc));
              float v139_data = ir2[7];
              ir2[7] = (v139_data + (v117_data * v113_bc));
              float v141_data = r0[4];
              float v143_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v145_data = ir2[0];
              ir2[0] = (v145_data + (v141_data * v143_bc));
              float v149_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v151_data = ir2[2];
              ir2[2] = (v151_data + (v141_data * v149_bc));
              float v155_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v157_data = ir2[4];
              ir2[4] = (v157_data + (v141_data * v155_bc));
              float v161_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v163_data = ir2[6];
              ir2[6] = (v163_data + (v141_data * v161_bc));
              float v165_data = r0[5];
              float v169_data = ir2[1];
              ir2[1] = (v169_data + (v165_data * v143_bc));
              float v175_data = ir2[3];
              ir2[3] = (v175_data + (v165_data * v149_bc));
              float v181_data = ir2[5];
              ir2[5] = (v181_data + (v165_data * v155_bc));
              float v187_data = ir2[7];
              ir2[7] = (v187_data + (v165_data * v161_bc));
              float v189_data = r0[6];
              float v191_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v193_data = ir2[0];
              ir2[0] = (v193_data + (v189_data * v191_bc));
              float v197_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v199_data = ir2[2];
              ir2[2] = (v199_data + (v189_data * v197_bc));
              float v203_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v205_data = ir2[4];
              ir2[4] = (v205_data + (v189_data * v203_bc));
              float v209_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v211_data = ir2[6];
              ir2[6] = (v211_data + (v189_data * v209_bc));
              float v213_data = r0[7];
              float v217_data = ir2[1];
              ir2[1] = (v217_data + (v213_data * v191_bc));
              float v223_data = ir2[3];
              ir2[3] = (v223_data + (v213_data * v197_bc));
              float v229_data = ir2[5];
              ir2[5] = (v229_data + (v213_data * v203_bc));
              float v235_data = ir2[7];
              ir2[7] = (v235_data + (v213_data * v209_bc));
              float v237_data = r0[8];
              float v239_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v241_data = ir2[0];
              ir2[0] = (v241_data + (v237_data * v239_bc));
              float v245_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v247_data = ir2[2];
              ir2[2] = (v247_data + (v237_data * v245_bc));
              float v251_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v253_data = ir2[4];
              ir2[4] = (v253_data + (v237_data * v251_bc));
              float v257_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v259_data = ir2[6];
              ir2[6] = (v259_data + (v237_data * v257_bc));
              float v261_data = r0[9];
              float v265_data = ir2[1];
              ir2[1] = (v265_data + (v261_data * v239_bc));
              float v271_data = ir2[3];
              ir2[3] = (v271_data + (v261_data * v245_bc));
              float v277_data = ir2[5];
              ir2[5] = (v277_data + (v261_data * v251_bc));
              float v283_data = ir2[7];
              ir2[7] = (v283_data + (v261_data * v257_bc));
              float v285_data = r0[10];
              float v287_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v289_data = ir2[0];
              ir2[0] = (v289_data + (v285_data * v287_bc));
              float v293_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v295_data = ir2[2];
              ir2[2] = (v295_data + (v285_data * v293_bc));
              float v299_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v301_data = ir2[4];
              ir2[4] = (v301_data + (v285_data * v299_bc));
              float v305_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v307_data = ir2[6];
              ir2[6] = (v307_data + (v285_data * v305_bc));
              float v309_data = r0[11];
              float v313_data = ir2[1];
              ir2[1] = (v313_data + (v309_data * v287_bc));
              float v319_data = ir2[3];
              ir2[3] = (v319_data + (v309_data * v293_bc));
              float v325_data = ir2[5];
              ir2[5] = (v325_data + (v309_data * v299_bc));
              float v331_data = ir2[7];
              ir2[7] = (v331_data + (v309_data * v305_bc));
              float v333_data = r0[12];
              float v335_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v337_data = ir2[0];
              ir2[0] = (v337_data + (v333_data * v335_bc));
              float v341_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v343_data = ir2[2];
              ir2[2] = (v343_data + (v333_data * v341_bc));
              float v347_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v349_data = ir2[4];
              ir2[4] = (v349_data + (v333_data * v347_bc));
              float v353_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v355_data = ir2[6];
              ir2[6] = (v355_data + (v333_data * v353_bc));
              float v357_data = r0[13];
              float v361_data = ir2[1];
              ir2[1] = (v361_data + (v357_data * v335_bc));
              float v367_data = ir2[3];
              ir2[3] = (v367_data + (v357_data * v341_bc));
              float v373_data = ir2[5];
              ir2[5] = (v373_data + (v357_data * v347_bc));
              float v379_data = ir2[7];
              ir2[7] = (v379_data + (v357_data * v353_bc));
              float v381_data = r0[14];
              float v383_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v385_data = ir2[0];
              ir2[0] = (v385_data + (v381_data * v383_bc));
              float v389_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v391_data = ir2[2];
              ir2[2] = (v391_data + (v381_data * v389_bc));
              float v395_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v397_data = ir2[4];
              ir2[4] = (v397_data + (v381_data * v395_bc));
              float v401_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v403_data = ir2[6];
              ir2[6] = (v403_data + (v381_data * v401_bc));
              float v405_data = r0[15];
              float v409_data = ir2[1];
              ir2[1] = (v409_data + (v405_data * v383_bc));
              float v415_data = ir2[3];
              ir2[3] = (v415_data + (v405_data * v389_bc));
              float v421_data = ir2[5];
              ir2[5] = (v421_data + (v405_data * v395_bc));
              float v427_data = ir2[7];
              ir2[7] = (v427_data + (v405_data * v401_bc));
              #pragma unroll
              for (int32_t v429_n0 = 0; v429_n0 < 1; ++v429_n0) {
                #pragma unroll
                for (int32_t v430_n1 = 0; v430_n1 < 4; ++v430_n1) {
                  int32_t v432_a = v429_n0 + (v430_n1 * 2);
                  float v433_data = ir2[v432_a];
                  r2[v432_a] = v433_data;
                }
              }
              if (v25_g) {
                #pragma unroll
                for (int32_t v434_n1 = 0; v434_n1 < 4; ++v434_n1) {
                  int32_t v436_a = 1 + (v434_n1 * 2);
                  float v437_data = ir2[v436_a];
                  r2[v436_a] = v437_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v438_i0 = 0; v438_i0 < 1; ++v438_i0) {
                int32_t v444_lead = v15_lead + (v438_i0 * 32);
                #pragma unroll
                for (int32_t v439_i1 = 0; v439_i1 < 4; ++v439_i1) {
                  float v442_data = r2[(v438_i0 + (v439_i1 * 2))];
                  glb_m0[(v444_lead + (v439_i1 * 35))] = v442_data;
                }
              }
              if (v25_g) {
                int32_t v452_lead = v15_lead + 32_i32;
                #pragma unroll
                for (int32_t v447_i1 = 0; v447_i1 < 4; ++v447_i1) {
                  float v450_data = r2[(1 + (v447_i1 * 2))];
                  glb_m0[(v452_lead + (v447_i1 * 35))] = v450_data;
                }
              }
              item.barrier();
            }
          }
        }
      });
    }
  });
}

