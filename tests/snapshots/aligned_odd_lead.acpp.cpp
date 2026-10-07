// === base name ===
kernel_08735a1f980c6843

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_08735a1f980c6843 = {{32, 1, 1}, 32, 35, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_08735a1f980c6843(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_08735a1f980c6843(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_08735a1f980c6843(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (32, 1, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 1 - 1) / 1;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 32;
  config.block[1] = 1;
  config.block[2] = 1;
  config.sharedMemBytes = 0 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_08735a1f980c6843(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_08735a1f980c6843(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_08735a1f980c6843(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_08735a1f980c6843(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes (35 active) x 1 per block = block 32x1x1, 0 B shared, occupancy grid
        // operands:
        //   m0 35×4(35×4) {0..35}×{0..4} strided
        //   m1 35×8(35×8) {0..35}×{0..8} strided
        //   m2 8×4(8×4) {0..8}×{0..4} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":35,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[35,4]],"name":"m0","ordered":false,"parts":1,"shape":[35,4],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[35,8]],"name":"m1","ordered":false,"parts":1,"shape":[35,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,4]],"name":"m2","ordered":false,"parts":1,"shape":[8,4],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[35,4]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[35,4]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[35,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[35,8]},{"addressing":"strided","bbox":[[0,0],[8,4]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,4]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 140 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 280 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 32 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v21_lead = item.get_local_id(2) % 32;
              #pragma unroll
              for (int32_t v22_i0 = 0; v22_i0 < 1; ++v22_i0) {
                int32_t v25_lead = v21_lead + (v22_i0 * 32);
                #pragma unroll
                for (int32_t v23_i1 = 0; v23_i1 < 8; ++v23_i1) {
                  float v28_data = glb_m1[(v25_lead + (v23_i1 * 35))];
                  r0[(v22_i0 + (v23_i1 * 2))] = v28_data;
                }
              }
              bool v31_g = v21_lead < 3;
              if (v31_g) {
                int32_t v34_lead = v21_lead + 32_i32;
                #pragma unroll
                for (int32_t v32_i1 = 0; v32_i1 < 8; ++v32_i1) {
                  float v37_data = glb_m1[(v34_lead + (v32_i1 * 35))];
                  r0[(1 + (v32_i1 * 2))] = v37_data;
                }
              }
              float r1[4]{};
              // r1 = load{g>r}(glb_m2);
              if (v21_lead < 8) {
                #pragma unroll
                for (int32_t v42_i1 = 0; v42_i1 < 4; ++v42_i1) {
                  float v47_data = glb_m2[(v21_lead + (v42_i1 * 8))];
                  r1[v42_i1] = v47_data;
                }
              }
              float r2[8]{};
              // ir2 = +(r0 * r1)
              // [(0, 35), (0, 4)] [(0, 8)]
              float ir2[8]{};
              float v51_data = r0[0];
              float v52_data = r1[0];
              float v53_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v55_data = ir2[0];
              ir2[0] = (v55_data + (v51_data * v53_bc));
              float v58_data = r1[1];
              float v59_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v61_data = ir2[2];
              ir2[2] = (v61_data + (v51_data * v59_bc));
              float v64_data = r1[2];
              float v65_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v67_data = ir2[4];
              ir2[4] = (v67_data + (v51_data * v65_bc));
              float v70_data = r1[3];
              float v71_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0));
              float v73_data = ir2[6];
              ir2[6] = (v73_data + (v51_data * v71_bc));
              float v75_data = r0[1];
              float v79_data = ir2[1];
              ir2[1] = (v79_data + (v75_data * v53_bc));
              float v85_data = ir2[3];
              ir2[3] = (v85_data + (v75_data * v59_bc));
              float v91_data = ir2[5];
              ir2[5] = (v91_data + (v75_data * v65_bc));
              float v97_data = ir2[7];
              ir2[7] = (v97_data + (v75_data * v71_bc));
              float v99_data = r0[2];
              float v101_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v103_data = ir2[0];
              ir2[0] = (v103_data + (v99_data * v101_bc));
              float v107_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v109_data = ir2[2];
              ir2[2] = (v109_data + (v99_data * v107_bc));
              float v113_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v115_data = ir2[4];
              ir2[4] = (v115_data + (v99_data * v113_bc));
              float v119_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1));
              float v121_data = ir2[6];
              ir2[6] = (v121_data + (v99_data * v119_bc));
              float v123_data = r0[3];
              float v127_data = ir2[1];
              ir2[1] = (v127_data + (v123_data * v101_bc));
              float v133_data = ir2[3];
              ir2[3] = (v133_data + (v123_data * v107_bc));
              float v139_data = ir2[5];
              ir2[5] = (v139_data + (v123_data * v113_bc));
              float v145_data = ir2[7];
              ir2[7] = (v145_data + (v123_data * v119_bc));
              float v147_data = r0[4];
              float v149_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v151_data = ir2[0];
              ir2[0] = (v151_data + (v147_data * v149_bc));
              float v155_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v157_data = ir2[2];
              ir2[2] = (v157_data + (v147_data * v155_bc));
              float v161_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v163_data = ir2[4];
              ir2[4] = (v163_data + (v147_data * v161_bc));
              float v167_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2));
              float v169_data = ir2[6];
              ir2[6] = (v169_data + (v147_data * v167_bc));
              float v171_data = r0[5];
              float v175_data = ir2[1];
              ir2[1] = (v175_data + (v171_data * v149_bc));
              float v181_data = ir2[3];
              ir2[3] = (v181_data + (v171_data * v155_bc));
              float v187_data = ir2[5];
              ir2[5] = (v187_data + (v171_data * v161_bc));
              float v193_data = ir2[7];
              ir2[7] = (v193_data + (v171_data * v167_bc));
              float v195_data = r0[6];
              float v197_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v199_data = ir2[0];
              ir2[0] = (v199_data + (v195_data * v197_bc));
              float v203_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v205_data = ir2[2];
              ir2[2] = (v205_data + (v195_data * v203_bc));
              float v209_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v211_data = ir2[4];
              ir2[4] = (v211_data + (v195_data * v209_bc));
              float v215_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3));
              float v217_data = ir2[6];
              ir2[6] = (v217_data + (v195_data * v215_bc));
              float v219_data = r0[7];
              float v223_data = ir2[1];
              ir2[1] = (v223_data + (v219_data * v197_bc));
              float v229_data = ir2[3];
              ir2[3] = (v229_data + (v219_data * v203_bc));
              float v235_data = ir2[5];
              ir2[5] = (v235_data + (v219_data * v209_bc));
              float v241_data = ir2[7];
              ir2[7] = (v241_data + (v219_data * v215_bc));
              float v243_data = r0[8];
              float v245_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v247_data = ir2[0];
              ir2[0] = (v247_data + (v243_data * v245_bc));
              float v251_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v253_data = ir2[2];
              ir2[2] = (v253_data + (v243_data * v251_bc));
              float v257_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v259_data = ir2[4];
              ir2[4] = (v259_data + (v243_data * v257_bc));
              float v263_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4));
              float v265_data = ir2[6];
              ir2[6] = (v265_data + (v243_data * v263_bc));
              float v267_data = r0[9];
              float v271_data = ir2[1];
              ir2[1] = (v271_data + (v267_data * v245_bc));
              float v277_data = ir2[3];
              ir2[3] = (v277_data + (v267_data * v251_bc));
              float v283_data = ir2[5];
              ir2[5] = (v283_data + (v267_data * v257_bc));
              float v289_data = ir2[7];
              ir2[7] = (v289_data + (v267_data * v263_bc));
              float v291_data = r0[10];
              float v293_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v295_data = ir2[0];
              ir2[0] = (v295_data + (v291_data * v293_bc));
              float v299_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v301_data = ir2[2];
              ir2[2] = (v301_data + (v291_data * v299_bc));
              float v305_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v307_data = ir2[4];
              ir2[4] = (v307_data + (v291_data * v305_bc));
              float v311_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5));
              float v313_data = ir2[6];
              ir2[6] = (v313_data + (v291_data * v311_bc));
              float v315_data = r0[11];
              float v319_data = ir2[1];
              ir2[1] = (v319_data + (v315_data * v293_bc));
              float v325_data = ir2[3];
              ir2[3] = (v325_data + (v315_data * v299_bc));
              float v331_data = ir2[5];
              ir2[5] = (v331_data + (v315_data * v305_bc));
              float v337_data = ir2[7];
              ir2[7] = (v337_data + (v315_data * v311_bc));
              float v339_data = r0[12];
              float v341_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v343_data = ir2[0];
              ir2[0] = (v343_data + (v339_data * v341_bc));
              float v347_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v349_data = ir2[2];
              ir2[2] = (v349_data + (v339_data * v347_bc));
              float v353_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v355_data = ir2[4];
              ir2[4] = (v355_data + (v339_data * v353_bc));
              float v359_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6));
              float v361_data = ir2[6];
              ir2[6] = (v361_data + (v339_data * v359_bc));
              float v363_data = r0[13];
              float v367_data = ir2[1];
              ir2[1] = (v367_data + (v363_data * v341_bc));
              float v373_data = ir2[3];
              ir2[3] = (v373_data + (v363_data * v347_bc));
              float v379_data = ir2[5];
              ir2[5] = (v379_data + (v363_data * v353_bc));
              float v385_data = ir2[7];
              ir2[7] = (v385_data + (v363_data * v359_bc));
              float v387_data = r0[14];
              float v389_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v391_data = ir2[0];
              ir2[0] = (v391_data + (v387_data * v389_bc));
              float v395_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v397_data = ir2[2];
              ir2[2] = (v397_data + (v387_data * v395_bc));
              float v401_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v403_data = ir2[4];
              ir2[4] = (v403_data + (v387_data * v401_bc));
              float v407_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7));
              float v409_data = ir2[6];
              ir2[6] = (v409_data + (v387_data * v407_bc));
              float v411_data = r0[15];
              float v415_data = ir2[1];
              ir2[1] = (v415_data + (v411_data * v389_bc));
              float v421_data = ir2[3];
              ir2[3] = (v421_data + (v411_data * v395_bc));
              float v427_data = ir2[5];
              ir2[5] = (v427_data + (v411_data * v401_bc));
              float v433_data = ir2[7];
              ir2[7] = (v433_data + (v411_data * v407_bc));
              // r2 = ir2
              #pragma unroll
              for (int32_t v435_n0 = 0; v435_n0 < 1; ++v435_n0) {
                #pragma unroll
                for (int32_t v436_n1 = 0; v436_n1 < 4; ++v436_n1) {
                  int32_t v438_a = v435_n0 + (v436_n1 * 2);
                  float v439_data = ir2[v438_a];
                  r2[v438_a] = v439_data;
                }
              }
              if (v31_g) {
                #pragma unroll
                for (int32_t v440_n1 = 0; v440_n1 < 4; ++v440_n1) {
                  int32_t v442_a = 1 + (v440_n1 * 2);
                  float v443_data = ir2[v442_a];
                  r2[v442_a] = v443_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v444_i0 = 0; v444_i0 < 1; ++v444_i0) {
                int32_t v450_lead = v21_lead + (v444_i0 * 32);
                #pragma unroll
                for (int32_t v445_i1 = 0; v445_i1 < 4; ++v445_i1) {
                  float v448_data = r2[(v444_i0 + (v445_i1 * 2))];
                  glb_m0[(v450_lead + (v445_i1 * 35))] = v448_data;
                }
              }
              if (v31_g) {
                int32_t v458_lead = v21_lead + 32_i32;
                #pragma unroll
                for (int32_t v453_i1 = 0; v453_i1 < 4; ++v453_i1) {
                  float v456_data = r2[(1 + (v453_i1 * 2))];
                  glb_m0[(v458_lead + (v453_i1 * 35))] = v456_data;
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

