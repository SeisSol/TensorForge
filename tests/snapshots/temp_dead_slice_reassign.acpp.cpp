// === base name ===
kernel_4b92f52e18b8c879

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_4b92f52e18b8c879 = {{16, 16, 1}, 16, 12, 1, 16, 10240, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_4b92f52e18b8c879(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_4b92f52e18b8c879(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_4b92f52e18b8c879(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_4b92f52e18b8c879(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_4b92f52e18b8c879(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_4b92f52e18b8c879(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_4b92f52e18b8c879(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (2560, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 10240 B shared, occupancy grid
        // operands:
        //   m0 6×12(6×12) {0..6}×{0..12} strided
        //   m1 12×12(12×12) {0..12}×{0..12} strided
        //   m2 6×12(6×12) {0..6}×{0..12} strided
        //   m3 6×12(6×12) {0..6}×{0..12} strided
        //   m4 12×12(12×12) {0..12}×{0..12} strided
        // operations:
        //   t0[i,j]@{6..12}×{0..12} = m0[i,k] × m1[k,j]
        //   t0[i,j] = m2[i,k] × m1[k,j]
        //   t0[i,j]@{6..12}×{0..12} = m3[i,j]
        //   m4[i,j] = t0[i,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2560}],"shared_bytes":10240,"shared_elements":2560,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B2","bbox":[[0,0],[6,12]],"name":"m0","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"N","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"Y","bbox":[[0,0],[6,12]],"name":"m3","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m4","ordered":false,"parts":1,"shape":[12,12],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[6,12]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[160 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[144];
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v4_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v4_batchId0 < numElements0; v4_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v5_ahead1 = v4_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v4_batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v4_batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v4_batchId0 * 72 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v4_batchId0 * 72 + 0 + m3_extraOffset];
              float *const __restrict__ glb_m4 = &m4[v4_batchId0 * 144 + 0 + m4_extraOffset];
              float r0[12]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v20_lead = item.get_local_id(2) % 16;
              bool v21_g = v20_lead < 6;
              if (v21_g) {
                #pragma unroll
                for (int32_t v22_i1 = 0; v22_i1 < 12; ++v22_i1) {
                  float v27_data = glb_m0[(v20_lead + (v22_i1 * 6))];
                  r0[v22_i1] = v27_data;
                }
              }
              float r1[12]{};
              // r1 = load{g>r}(glb_m1);
              bool v30_g = v20_lead < 12;
              if (v30_g) {
                #pragma unroll
                for (int32_t v31_i1 = 0; v31_i1 < 12; ++v31_i1) {
                  float v36_data = glb_m1[(v20_lead + (v31_i1 * 12))];
                  r1[v31_i1] = v36_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r3[12]{};
              // r3 = load{g>r}(glb_m2);
              if (v21_g) {
                #pragma unroll
                for (int32_t v39_i1 = 0; v39_i1 < 12; ++v39_i1) {
                  float v44_data = glb_m2[(v20_lead + (v39_i1 * 6))];
                  r3[v39_i1] = v44_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m1););
              float r2[12]{};
              // r2 = +(r0 * r1) + None
              // [(0, 6), (0, 12)] [(0, 12)]
              float v47_data = r0[0];
              float v48_data = r1[0];
              float v49_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v51_data = r2[0];
              r2[0] = (v51_data + (v47_data * v49_bc));
              float v54_data = r1[1];
              float v55_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v57_data = r2[1];
              r2[1] = (v57_data + (v47_data * v55_bc));
              float v60_data = r1[2];
              float v61_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v63_data = r2[2];
              r2[2] = (v63_data + (v47_data * v61_bc));
              float v66_data = r1[3];
              float v67_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v69_data = r2[3];
              r2[3] = (v69_data + (v47_data * v67_bc));
              float v72_data = r1[4];
              float v73_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v75_data = r2[4];
              r2[4] = (v75_data + (v47_data * v73_bc));
              float v78_data = r1[5];
              float v79_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v81_data = r2[5];
              r2[5] = (v81_data + (v47_data * v79_bc));
              float v84_data = r1[6];
              float v85_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v87_data = r2[6];
              r2[6] = (v87_data + (v47_data * v85_bc));
              float v90_data = r1[7];
              float v91_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v93_data = r2[7];
              r2[7] = (v93_data + (v47_data * v91_bc));
              float v96_data = r1[8];
              float v97_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v99_data = r2[8];
              r2[8] = (v99_data + (v47_data * v97_bc));
              float v102_data = r1[9];
              float v103_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v105_data = r2[9];
              r2[9] = (v105_data + (v47_data * v103_bc));
              float v108_data = r1[10];
              float v109_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v111_data = r2[10];
              r2[10] = (v111_data + (v47_data * v109_bc));
              float v114_data = r1[11];
              float v115_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v117_data = r2[11];
              r2[11] = (v117_data + (v47_data * v115_bc));
              float v119_data = r0[1];
              float v121_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v123_data = r2[0];
              r2[0] = (v123_data + (v119_data * v121_bc));
              float v127_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v129_data = r2[1];
              r2[1] = (v129_data + (v119_data * v127_bc));
              float v133_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v135_data = r2[2];
              r2[2] = (v135_data + (v119_data * v133_bc));
              float v139_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v141_data = r2[3];
              r2[3] = (v141_data + (v119_data * v139_bc));
              float v145_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v147_data = r2[4];
              r2[4] = (v147_data + (v119_data * v145_bc));
              float v151_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v153_data = r2[5];
              r2[5] = (v153_data + (v119_data * v151_bc));
              float v157_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v159_data = r2[6];
              r2[6] = (v159_data + (v119_data * v157_bc));
              float v163_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v165_data = r2[7];
              r2[7] = (v165_data + (v119_data * v163_bc));
              float v169_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v171_data = r2[8];
              r2[8] = (v171_data + (v119_data * v169_bc));
              float v175_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v177_data = r2[9];
              r2[9] = (v177_data + (v119_data * v175_bc));
              float v181_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v183_data = r2[10];
              r2[10] = (v183_data + (v119_data * v181_bc));
              float v187_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v189_data = r2[11];
              r2[11] = (v189_data + (v119_data * v187_bc));
              float v191_data = r0[2];
              float v193_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v195_data = r2[0];
              r2[0] = (v195_data + (v191_data * v193_bc));
              float v199_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v201_data = r2[1];
              r2[1] = (v201_data + (v191_data * v199_bc));
              float v205_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v207_data = r2[2];
              r2[2] = (v207_data + (v191_data * v205_bc));
              float v211_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v213_data = r2[3];
              r2[3] = (v213_data + (v191_data * v211_bc));
              float v217_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v219_data = r2[4];
              r2[4] = (v219_data + (v191_data * v217_bc));
              float v223_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v225_data = r2[5];
              r2[5] = (v225_data + (v191_data * v223_bc));
              float v229_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v231_data = r2[6];
              r2[6] = (v231_data + (v191_data * v229_bc));
              float v235_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v237_data = r2[7];
              r2[7] = (v237_data + (v191_data * v235_bc));
              float v241_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v243_data = r2[8];
              r2[8] = (v243_data + (v191_data * v241_bc));
              float v247_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v249_data = r2[9];
              r2[9] = (v249_data + (v191_data * v247_bc));
              float v253_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v255_data = r2[10];
              r2[10] = (v255_data + (v191_data * v253_bc));
              float v259_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v261_data = r2[11];
              r2[11] = (v261_data + (v191_data * v259_bc));
              float v263_data = r0[3];
              float v265_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v267_data = r2[0];
              r2[0] = (v267_data + (v263_data * v265_bc));
              float v271_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v273_data = r2[1];
              r2[1] = (v273_data + (v263_data * v271_bc));
              float v277_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v279_data = r2[2];
              r2[2] = (v279_data + (v263_data * v277_bc));
              float v283_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v285_data = r2[3];
              r2[3] = (v285_data + (v263_data * v283_bc));
              float v289_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v291_data = r2[4];
              r2[4] = (v291_data + (v263_data * v289_bc));
              float v295_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v297_data = r2[5];
              r2[5] = (v297_data + (v263_data * v295_bc));
              float v301_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v303_data = r2[6];
              r2[6] = (v303_data + (v263_data * v301_bc));
              float v307_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v309_data = r2[7];
              r2[7] = (v309_data + (v263_data * v307_bc));
              float v313_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v315_data = r2[8];
              r2[8] = (v315_data + (v263_data * v313_bc));
              float v319_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v321_data = r2[9];
              r2[9] = (v321_data + (v263_data * v319_bc));
              float v325_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v327_data = r2[10];
              r2[10] = (v327_data + (v263_data * v325_bc));
              float v331_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v333_data = r2[11];
              r2[11] = (v333_data + (v263_data * v331_bc));
              float v335_data = r0[4];
              float v337_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v339_data = r2[0];
              r2[0] = (v339_data + (v335_data * v337_bc));
              float v343_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v345_data = r2[1];
              r2[1] = (v345_data + (v335_data * v343_bc));
              float v349_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v351_data = r2[2];
              r2[2] = (v351_data + (v335_data * v349_bc));
              float v355_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v357_data = r2[3];
              r2[3] = (v357_data + (v335_data * v355_bc));
              float v361_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v363_data = r2[4];
              r2[4] = (v363_data + (v335_data * v361_bc));
              float v367_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v369_data = r2[5];
              r2[5] = (v369_data + (v335_data * v367_bc));
              float v373_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v375_data = r2[6];
              r2[6] = (v375_data + (v335_data * v373_bc));
              float v379_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v381_data = r2[7];
              r2[7] = (v381_data + (v335_data * v379_bc));
              float v385_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v387_data = r2[8];
              r2[8] = (v387_data + (v335_data * v385_bc));
              float v391_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v393_data = r2[9];
              r2[9] = (v393_data + (v335_data * v391_bc));
              float v397_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v399_data = r2[10];
              r2[10] = (v399_data + (v335_data * v397_bc));
              float v403_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v405_data = r2[11];
              r2[11] = (v405_data + (v335_data * v403_bc));
              float v407_data = r0[5];
              float v409_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v411_data = r2[0];
              r2[0] = (v411_data + (v407_data * v409_bc));
              float v415_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v417_data = r2[1];
              r2[1] = (v417_data + (v407_data * v415_bc));
              float v421_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v423_data = r2[2];
              r2[2] = (v423_data + (v407_data * v421_bc));
              float v427_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v429_data = r2[3];
              r2[3] = (v429_data + (v407_data * v427_bc));
              float v433_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v435_data = r2[4];
              r2[4] = (v435_data + (v407_data * v433_bc));
              float v439_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v441_data = r2[5];
              r2[5] = (v441_data + (v407_data * v439_bc));
              float v445_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v447_data = r2[6];
              r2[6] = (v447_data + (v407_data * v445_bc));
              float v451_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v453_data = r2[7];
              r2[7] = (v453_data + (v407_data * v451_bc));
              float v457_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v459_data = r2[8];
              r2[8] = (v459_data + (v407_data * v457_bc));
              float v463_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v465_data = r2[9];
              r2[9] = (v465_data + (v407_data * v463_bc));
              float v469_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v471_data = r2[10];
              r2[10] = (v471_data + (v407_data * v469_bc));
              float v475_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v477_data = r2[11];
              r2[11] = (v477_data + (v407_data * v475_bc));
              float v479_data = r0[6];
              float v481_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v483_data = r2[0];
              r2[0] = (v483_data + (v479_data * v481_bc));
              float v487_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v489_data = r2[1];
              r2[1] = (v489_data + (v479_data * v487_bc));
              float v493_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v495_data = r2[2];
              r2[2] = (v495_data + (v479_data * v493_bc));
              float v499_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v501_data = r2[3];
              r2[3] = (v501_data + (v479_data * v499_bc));
              float v505_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v507_data = r2[4];
              r2[4] = (v507_data + (v479_data * v505_bc));
              float v511_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v513_data = r2[5];
              r2[5] = (v513_data + (v479_data * v511_bc));
              float v517_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v519_data = r2[6];
              r2[6] = (v519_data + (v479_data * v517_bc));
              float v523_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v525_data = r2[7];
              r2[7] = (v525_data + (v479_data * v523_bc));
              float v529_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v531_data = r2[8];
              r2[8] = (v531_data + (v479_data * v529_bc));
              float v535_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v537_data = r2[9];
              r2[9] = (v537_data + (v479_data * v535_bc));
              float v541_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v543_data = r2[10];
              r2[10] = (v543_data + (v479_data * v541_bc));
              float v547_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v549_data = r2[11];
              r2[11] = (v549_data + (v479_data * v547_bc));
              float v551_data = r0[7];
              float v553_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v555_data = r2[0];
              r2[0] = (v555_data + (v551_data * v553_bc));
              float v559_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v561_data = r2[1];
              r2[1] = (v561_data + (v551_data * v559_bc));
              float v565_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v567_data = r2[2];
              r2[2] = (v567_data + (v551_data * v565_bc));
              float v571_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v573_data = r2[3];
              r2[3] = (v573_data + (v551_data * v571_bc));
              float v577_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v579_data = r2[4];
              r2[4] = (v579_data + (v551_data * v577_bc));
              float v583_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v585_data = r2[5];
              r2[5] = (v585_data + (v551_data * v583_bc));
              float v589_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v591_data = r2[6];
              r2[6] = (v591_data + (v551_data * v589_bc));
              float v595_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v597_data = r2[7];
              r2[7] = (v597_data + (v551_data * v595_bc));
              float v601_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v603_data = r2[8];
              r2[8] = (v603_data + (v551_data * v601_bc));
              float v607_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v609_data = r2[9];
              r2[9] = (v609_data + (v551_data * v607_bc));
              float v613_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v615_data = r2[10];
              r2[10] = (v615_data + (v551_data * v613_bc));
              float v619_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v621_data = r2[11];
              r2[11] = (v621_data + (v551_data * v619_bc));
              float v623_data = r0[8];
              float v625_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v627_data = r2[0];
              r2[0] = (v627_data + (v623_data * v625_bc));
              float v631_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v633_data = r2[1];
              r2[1] = (v633_data + (v623_data * v631_bc));
              float v637_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v639_data = r2[2];
              r2[2] = (v639_data + (v623_data * v637_bc));
              float v643_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v645_data = r2[3];
              r2[3] = (v645_data + (v623_data * v643_bc));
              float v649_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v651_data = r2[4];
              r2[4] = (v651_data + (v623_data * v649_bc));
              float v655_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v657_data = r2[5];
              r2[5] = (v657_data + (v623_data * v655_bc));
              float v661_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v663_data = r2[6];
              r2[6] = (v663_data + (v623_data * v661_bc));
              float v667_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v669_data = r2[7];
              r2[7] = (v669_data + (v623_data * v667_bc));
              float v673_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v675_data = r2[8];
              r2[8] = (v675_data + (v623_data * v673_bc));
              float v679_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v681_data = r2[9];
              r2[9] = (v681_data + (v623_data * v679_bc));
              float v685_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v687_data = r2[10];
              r2[10] = (v687_data + (v623_data * v685_bc));
              float v691_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v693_data = r2[11];
              r2[11] = (v693_data + (v623_data * v691_bc));
              float v695_data = r0[9];
              float v697_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v699_data = r2[0];
              r2[0] = (v699_data + (v695_data * v697_bc));
              float v703_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v705_data = r2[1];
              r2[1] = (v705_data + (v695_data * v703_bc));
              float v709_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v711_data = r2[2];
              r2[2] = (v711_data + (v695_data * v709_bc));
              float v715_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v717_data = r2[3];
              r2[3] = (v717_data + (v695_data * v715_bc));
              float v721_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v723_data = r2[4];
              r2[4] = (v723_data + (v695_data * v721_bc));
              float v727_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v729_data = r2[5];
              r2[5] = (v729_data + (v695_data * v727_bc));
              float v733_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v735_data = r2[6];
              r2[6] = (v735_data + (v695_data * v733_bc));
              float v739_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v741_data = r2[7];
              r2[7] = (v741_data + (v695_data * v739_bc));
              float v745_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v747_data = r2[8];
              r2[8] = (v747_data + (v695_data * v745_bc));
              float v751_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v753_data = r2[9];
              r2[9] = (v753_data + (v695_data * v751_bc));
              float v757_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v759_data = r2[10];
              r2[10] = (v759_data + (v695_data * v757_bc));
              float v763_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v765_data = r2[11];
              r2[11] = (v765_data + (v695_data * v763_bc));
              float v767_data = r0[10];
              float v769_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v771_data = r2[0];
              r2[0] = (v771_data + (v767_data * v769_bc));
              float v775_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v777_data = r2[1];
              r2[1] = (v777_data + (v767_data * v775_bc));
              float v781_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v783_data = r2[2];
              r2[2] = (v783_data + (v767_data * v781_bc));
              float v787_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v789_data = r2[3];
              r2[3] = (v789_data + (v767_data * v787_bc));
              float v793_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v795_data = r2[4];
              r2[4] = (v795_data + (v767_data * v793_bc));
              float v799_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v801_data = r2[5];
              r2[5] = (v801_data + (v767_data * v799_bc));
              float v805_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v807_data = r2[6];
              r2[6] = (v807_data + (v767_data * v805_bc));
              float v811_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v813_data = r2[7];
              r2[7] = (v813_data + (v767_data * v811_bc));
              float v817_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v819_data = r2[8];
              r2[8] = (v819_data + (v767_data * v817_bc));
              float v823_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v825_data = r2[9];
              r2[9] = (v825_data + (v767_data * v823_bc));
              float v829_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v831_data = r2[10];
              r2[10] = (v831_data + (v767_data * v829_bc));
              float v835_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v837_data = r2[11];
              r2[11] = (v837_data + (v767_data * v835_bc));
              float v839_data = r0[11];
              float v841_bc = sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v843_data = r2[0];
              r2[0] = (v843_data + (v839_data * v841_bc));
              float v847_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v849_data = r2[1];
              r2[1] = (v849_data + (v839_data * v847_bc));
              float v853_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v855_data = r2[2];
              r2[2] = (v855_data + (v839_data * v853_bc));
              float v859_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v861_data = r2[3];
              r2[3] = (v861_data + (v839_data * v859_bc));
              float v865_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v867_data = r2[4];
              r2[4] = (v867_data + (v839_data * v865_bc));
              float v871_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v873_data = r2[5];
              r2[5] = (v873_data + (v839_data * v871_bc));
              float v877_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v879_data = r2[6];
              r2[6] = (v879_data + (v839_data * v877_bc));
              float v883_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v885_data = r2[7];
              r2[7] = (v885_data + (v839_data * v883_bc));
              float v889_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v891_data = r2[8];
              r2[8] = (v891_data + (v839_data * v889_bc));
              float v895_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v897_data = r2[9];
              r2[9] = (v897_data + (v839_data * v895_bc));
              float v901_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v903_data = r2[10];
              r2[10] = (v903_data + (v839_data * v901_bc));
              float v907_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v909_data = r2[11];
              r2[11] = (v909_data + (v839_data * v907_bc));
              // s0 = store{r>s}(localShrMem0, r2);
              if (v21_g) {
                int32_t v916_off = v20_lead + 6;
                #pragma unroll
                for (int32_t v911_i1 = 0; v911_i1 < 12; ++v911_i1) {
                  float v913_data = r2[v911_i1];
                  int32_t v918_a = v916_off + (v911_i1 * 12);
                  s0[(v918_a ^ ((v918_a >> 4) & 15))] = v913_data;
                }
              }
              float r5[12]{};
              // r5 = load{g>r}(glb_m3);
              if (v21_g) {
                #pragma unroll
                for (int32_t v923_i1 = 0; v923_i1 < 12; ++v923_i1) {
                  float v928_data = glb_m3[(v20_lead + (v923_i1 * 6))];
                  r5[v923_i1] = v928_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m2););
              float r4[12]{};
              // ir4 = +(r3 * r1)
              // [(0, 6), (0, 12)] [(0, 12)]
              float ir4[12]{};
              float v932_data = r3[0];
              float v936_data = ir4[0];
              ir4[0] = (v936_data + (v932_data * v49_bc));
              float v942_data = ir4[1];
              ir4[1] = (v942_data + (v932_data * v55_bc));
              float v948_data = ir4[2];
              ir4[2] = (v948_data + (v932_data * v61_bc));
              float v954_data = ir4[3];
              ir4[3] = (v954_data + (v932_data * v67_bc));
              float v960_data = ir4[4];
              ir4[4] = (v960_data + (v932_data * v73_bc));
              float v966_data = ir4[5];
              ir4[5] = (v966_data + (v932_data * v79_bc));
              float v972_data = ir4[6];
              ir4[6] = (v972_data + (v932_data * v85_bc));
              float v978_data = ir4[7];
              ir4[7] = (v978_data + (v932_data * v91_bc));
              float v984_data = ir4[8];
              ir4[8] = (v984_data + (v932_data * v97_bc));
              float v990_data = ir4[9];
              ir4[9] = (v990_data + (v932_data * v103_bc));
              float v996_data = ir4[10];
              ir4[10] = (v996_data + (v932_data * v109_bc));
              float v1002_data = ir4[11];
              ir4[11] = (v1002_data + (v932_data * v115_bc));
              float v1004_data = r3[1];
              float v1008_data = ir4[0];
              ir4[0] = (v1008_data + (v1004_data * v121_bc));
              float v1014_data = ir4[1];
              ir4[1] = (v1014_data + (v1004_data * v127_bc));
              float v1020_data = ir4[2];
              ir4[2] = (v1020_data + (v1004_data * v133_bc));
              float v1026_data = ir4[3];
              ir4[3] = (v1026_data + (v1004_data * v139_bc));
              float v1032_data = ir4[4];
              ir4[4] = (v1032_data + (v1004_data * v145_bc));
              float v1038_data = ir4[5];
              ir4[5] = (v1038_data + (v1004_data * v151_bc));
              float v1044_data = ir4[6];
              ir4[6] = (v1044_data + (v1004_data * v157_bc));
              float v1050_data = ir4[7];
              ir4[7] = (v1050_data + (v1004_data * v163_bc));
              float v1056_data = ir4[8];
              ir4[8] = (v1056_data + (v1004_data * v169_bc));
              float v1062_data = ir4[9];
              ir4[9] = (v1062_data + (v1004_data * v175_bc));
              float v1068_data = ir4[10];
              ir4[10] = (v1068_data + (v1004_data * v181_bc));
              float v1074_data = ir4[11];
              ir4[11] = (v1074_data + (v1004_data * v187_bc));
              float v1076_data = r3[2];
              float v1080_data = ir4[0];
              ir4[0] = (v1080_data + (v1076_data * v193_bc));
              float v1086_data = ir4[1];
              ir4[1] = (v1086_data + (v1076_data * v199_bc));
              float v1092_data = ir4[2];
              ir4[2] = (v1092_data + (v1076_data * v205_bc));
              float v1098_data = ir4[3];
              ir4[3] = (v1098_data + (v1076_data * v211_bc));
              float v1104_data = ir4[4];
              ir4[4] = (v1104_data + (v1076_data * v217_bc));
              float v1110_data = ir4[5];
              ir4[5] = (v1110_data + (v1076_data * v223_bc));
              float v1116_data = ir4[6];
              ir4[6] = (v1116_data + (v1076_data * v229_bc));
              float v1122_data = ir4[7];
              ir4[7] = (v1122_data + (v1076_data * v235_bc));
              float v1128_data = ir4[8];
              ir4[8] = (v1128_data + (v1076_data * v241_bc));
              float v1134_data = ir4[9];
              ir4[9] = (v1134_data + (v1076_data * v247_bc));
              float v1140_data = ir4[10];
              ir4[10] = (v1140_data + (v1076_data * v253_bc));
              float v1146_data = ir4[11];
              ir4[11] = (v1146_data + (v1076_data * v259_bc));
              float v1148_data = r3[3];
              float v1152_data = ir4[0];
              ir4[0] = (v1152_data + (v1148_data * v265_bc));
              float v1158_data = ir4[1];
              ir4[1] = (v1158_data + (v1148_data * v271_bc));
              float v1164_data = ir4[2];
              ir4[2] = (v1164_data + (v1148_data * v277_bc));
              float v1170_data = ir4[3];
              ir4[3] = (v1170_data + (v1148_data * v283_bc));
              float v1176_data = ir4[4];
              ir4[4] = (v1176_data + (v1148_data * v289_bc));
              float v1182_data = ir4[5];
              ir4[5] = (v1182_data + (v1148_data * v295_bc));
              float v1188_data = ir4[6];
              ir4[6] = (v1188_data + (v1148_data * v301_bc));
              float v1194_data = ir4[7];
              ir4[7] = (v1194_data + (v1148_data * v307_bc));
              float v1200_data = ir4[8];
              ir4[8] = (v1200_data + (v1148_data * v313_bc));
              float v1206_data = ir4[9];
              ir4[9] = (v1206_data + (v1148_data * v319_bc));
              float v1212_data = ir4[10];
              ir4[10] = (v1212_data + (v1148_data * v325_bc));
              float v1218_data = ir4[11];
              ir4[11] = (v1218_data + (v1148_data * v331_bc));
              float v1220_data = r3[4];
              float v1224_data = ir4[0];
              ir4[0] = (v1224_data + (v1220_data * v337_bc));
              float v1230_data = ir4[1];
              ir4[1] = (v1230_data + (v1220_data * v343_bc));
              float v1236_data = ir4[2];
              ir4[2] = (v1236_data + (v1220_data * v349_bc));
              float v1242_data = ir4[3];
              ir4[3] = (v1242_data + (v1220_data * v355_bc));
              float v1248_data = ir4[4];
              ir4[4] = (v1248_data + (v1220_data * v361_bc));
              float v1254_data = ir4[5];
              ir4[5] = (v1254_data + (v1220_data * v367_bc));
              float v1260_data = ir4[6];
              ir4[6] = (v1260_data + (v1220_data * v373_bc));
              float v1266_data = ir4[7];
              ir4[7] = (v1266_data + (v1220_data * v379_bc));
              float v1272_data = ir4[8];
              ir4[8] = (v1272_data + (v1220_data * v385_bc));
              float v1278_data = ir4[9];
              ir4[9] = (v1278_data + (v1220_data * v391_bc));
              float v1284_data = ir4[10];
              ir4[10] = (v1284_data + (v1220_data * v397_bc));
              float v1290_data = ir4[11];
              ir4[11] = (v1290_data + (v1220_data * v403_bc));
              float v1292_data = r3[5];
              float v1296_data = ir4[0];
              ir4[0] = (v1296_data + (v1292_data * v409_bc));
              float v1302_data = ir4[1];
              ir4[1] = (v1302_data + (v1292_data * v415_bc));
              float v1308_data = ir4[2];
              ir4[2] = (v1308_data + (v1292_data * v421_bc));
              float v1314_data = ir4[3];
              ir4[3] = (v1314_data + (v1292_data * v427_bc));
              float v1320_data = ir4[4];
              ir4[4] = (v1320_data + (v1292_data * v433_bc));
              float v1326_data = ir4[5];
              ir4[5] = (v1326_data + (v1292_data * v439_bc));
              float v1332_data = ir4[6];
              ir4[6] = (v1332_data + (v1292_data * v445_bc));
              float v1338_data = ir4[7];
              ir4[7] = (v1338_data + (v1292_data * v451_bc));
              float v1344_data = ir4[8];
              ir4[8] = (v1344_data + (v1292_data * v457_bc));
              float v1350_data = ir4[9];
              ir4[9] = (v1350_data + (v1292_data * v463_bc));
              float v1356_data = ir4[10];
              ir4[10] = (v1356_data + (v1292_data * v469_bc));
              float v1362_data = ir4[11];
              ir4[11] = (v1362_data + (v1292_data * v475_bc));
              float v1364_data = r3[6];
              float v1368_data = ir4[0];
              ir4[0] = (v1368_data + (v1364_data * v481_bc));
              float v1374_data = ir4[1];
              ir4[1] = (v1374_data + (v1364_data * v487_bc));
              float v1380_data = ir4[2];
              ir4[2] = (v1380_data + (v1364_data * v493_bc));
              float v1386_data = ir4[3];
              ir4[3] = (v1386_data + (v1364_data * v499_bc));
              float v1392_data = ir4[4];
              ir4[4] = (v1392_data + (v1364_data * v505_bc));
              float v1398_data = ir4[5];
              ir4[5] = (v1398_data + (v1364_data * v511_bc));
              float v1404_data = ir4[6];
              ir4[6] = (v1404_data + (v1364_data * v517_bc));
              float v1410_data = ir4[7];
              ir4[7] = (v1410_data + (v1364_data * v523_bc));
              float v1416_data = ir4[8];
              ir4[8] = (v1416_data + (v1364_data * v529_bc));
              float v1422_data = ir4[9];
              ir4[9] = (v1422_data + (v1364_data * v535_bc));
              float v1428_data = ir4[10];
              ir4[10] = (v1428_data + (v1364_data * v541_bc));
              float v1434_data = ir4[11];
              ir4[11] = (v1434_data + (v1364_data * v547_bc));
              float v1436_data = r3[7];
              float v1440_data = ir4[0];
              ir4[0] = (v1440_data + (v1436_data * v553_bc));
              float v1446_data = ir4[1];
              ir4[1] = (v1446_data + (v1436_data * v559_bc));
              float v1452_data = ir4[2];
              ir4[2] = (v1452_data + (v1436_data * v565_bc));
              float v1458_data = ir4[3];
              ir4[3] = (v1458_data + (v1436_data * v571_bc));
              float v1464_data = ir4[4];
              ir4[4] = (v1464_data + (v1436_data * v577_bc));
              float v1470_data = ir4[5];
              ir4[5] = (v1470_data + (v1436_data * v583_bc));
              float v1476_data = ir4[6];
              ir4[6] = (v1476_data + (v1436_data * v589_bc));
              float v1482_data = ir4[7];
              ir4[7] = (v1482_data + (v1436_data * v595_bc));
              float v1488_data = ir4[8];
              ir4[8] = (v1488_data + (v1436_data * v601_bc));
              float v1494_data = ir4[9];
              ir4[9] = (v1494_data + (v1436_data * v607_bc));
              float v1500_data = ir4[10];
              ir4[10] = (v1500_data + (v1436_data * v613_bc));
              float v1506_data = ir4[11];
              ir4[11] = (v1506_data + (v1436_data * v619_bc));
              float v1508_data = r3[8];
              float v1512_data = ir4[0];
              ir4[0] = (v1512_data + (v1508_data * v625_bc));
              float v1518_data = ir4[1];
              ir4[1] = (v1518_data + (v1508_data * v631_bc));
              float v1524_data = ir4[2];
              ir4[2] = (v1524_data + (v1508_data * v637_bc));
              float v1530_data = ir4[3];
              ir4[3] = (v1530_data + (v1508_data * v643_bc));
              float v1536_data = ir4[4];
              ir4[4] = (v1536_data + (v1508_data * v649_bc));
              float v1542_data = ir4[5];
              ir4[5] = (v1542_data + (v1508_data * v655_bc));
              float v1548_data = ir4[6];
              ir4[6] = (v1548_data + (v1508_data * v661_bc));
              float v1554_data = ir4[7];
              ir4[7] = (v1554_data + (v1508_data * v667_bc));
              float v1560_data = ir4[8];
              ir4[8] = (v1560_data + (v1508_data * v673_bc));
              float v1566_data = ir4[9];
              ir4[9] = (v1566_data + (v1508_data * v679_bc));
              float v1572_data = ir4[10];
              ir4[10] = (v1572_data + (v1508_data * v685_bc));
              float v1578_data = ir4[11];
              ir4[11] = (v1578_data + (v1508_data * v691_bc));
              float v1580_data = r3[9];
              float v1584_data = ir4[0];
              ir4[0] = (v1584_data + (v1580_data * v697_bc));
              float v1590_data = ir4[1];
              ir4[1] = (v1590_data + (v1580_data * v703_bc));
              float v1596_data = ir4[2];
              ir4[2] = (v1596_data + (v1580_data * v709_bc));
              float v1602_data = ir4[3];
              ir4[3] = (v1602_data + (v1580_data * v715_bc));
              float v1608_data = ir4[4];
              ir4[4] = (v1608_data + (v1580_data * v721_bc));
              float v1614_data = ir4[5];
              ir4[5] = (v1614_data + (v1580_data * v727_bc));
              float v1620_data = ir4[6];
              ir4[6] = (v1620_data + (v1580_data * v733_bc));
              float v1626_data = ir4[7];
              ir4[7] = (v1626_data + (v1580_data * v739_bc));
              float v1632_data = ir4[8];
              ir4[8] = (v1632_data + (v1580_data * v745_bc));
              float v1638_data = ir4[9];
              ir4[9] = (v1638_data + (v1580_data * v751_bc));
              float v1644_data = ir4[10];
              ir4[10] = (v1644_data + (v1580_data * v757_bc));
              float v1650_data = ir4[11];
              ir4[11] = (v1650_data + (v1580_data * v763_bc));
              float v1652_data = r3[10];
              float v1656_data = ir4[0];
              ir4[0] = (v1656_data + (v1652_data * v769_bc));
              float v1662_data = ir4[1];
              ir4[1] = (v1662_data + (v1652_data * v775_bc));
              float v1668_data = ir4[2];
              ir4[2] = (v1668_data + (v1652_data * v781_bc));
              float v1674_data = ir4[3];
              ir4[3] = (v1674_data + (v1652_data * v787_bc));
              float v1680_data = ir4[4];
              ir4[4] = (v1680_data + (v1652_data * v793_bc));
              float v1686_data = ir4[5];
              ir4[5] = (v1686_data + (v1652_data * v799_bc));
              float v1692_data = ir4[6];
              ir4[6] = (v1692_data + (v1652_data * v805_bc));
              float v1698_data = ir4[7];
              ir4[7] = (v1698_data + (v1652_data * v811_bc));
              float v1704_data = ir4[8];
              ir4[8] = (v1704_data + (v1652_data * v817_bc));
              float v1710_data = ir4[9];
              ir4[9] = (v1710_data + (v1652_data * v823_bc));
              float v1716_data = ir4[10];
              ir4[10] = (v1716_data + (v1652_data * v829_bc));
              float v1722_data = ir4[11];
              ir4[11] = (v1722_data + (v1652_data * v835_bc));
              float v1724_data = r3[11];
              float v1728_data = ir4[0];
              ir4[0] = (v1728_data + (v1724_data * v841_bc));
              float v1734_data = ir4[1];
              ir4[1] = (v1734_data + (v1724_data * v847_bc));
              float v1740_data = ir4[2];
              ir4[2] = (v1740_data + (v1724_data * v853_bc));
              float v1746_data = ir4[3];
              ir4[3] = (v1746_data + (v1724_data * v859_bc));
              float v1752_data = ir4[4];
              ir4[4] = (v1752_data + (v1724_data * v865_bc));
              float v1758_data = ir4[5];
              ir4[5] = (v1758_data + (v1724_data * v871_bc));
              float v1764_data = ir4[6];
              ir4[6] = (v1764_data + (v1724_data * v877_bc));
              float v1770_data = ir4[7];
              ir4[7] = (v1770_data + (v1724_data * v883_bc));
              float v1776_data = ir4[8];
              ir4[8] = (v1776_data + (v1724_data * v889_bc));
              float v1782_data = ir4[9];
              ir4[9] = (v1782_data + (v1724_data * v895_bc));
              float v1788_data = ir4[10];
              ir4[10] = (v1788_data + (v1724_data * v901_bc));
              float v1794_data = ir4[11];
              ir4[11] = (v1794_data + (v1724_data * v907_bc));
              // r4 = ir4
              if (v21_g) {
                #pragma unroll
                for (int32_t v1796_n1 = 0; v1796_n1 < 12; ++v1796_n1) {
                  float v1798_data = ir4[v1796_n1];
                  r4[v1796_n1] = v1798_data;
                }
              }
              // s0 = store{r>s, clear}(localShrMem0, r4);
              if ((v20_lead >= 6) && v30_g) {
                #pragma unroll
                for (int32_t v1801_z1 = 0; v1801_z1 < 12; ++v1801_z1) {
                  int32_t v1806_a = v20_lead + (v1801_z1 * 12);
                  s0[(v1806_a ^ ((v1806_a >> 4) & 15))] = 0.0f;
                }
              }
              if (v21_g) {
                #pragma unroll
                for (int32_t v1810_i1 = 0; v1810_i1 < 12; ++v1810_i1) {
                  float v1812_data = r4[v1810_i1];
                  int32_t v1816_a = v20_lead + (v1810_i1 * 12);
                  s0[(v1816_a ^ ((v1816_a >> 4) & 15))] = v1812_data;
                }
              }
              // wait(r5 = load{g>r}(glb_m3););
              float r6[12]{};
              // ir6 = +(r5)
              // [(0, 6), (0, 12)] []
              float ir6[12]{};
              float v1822_data = r5[0];
              float v1823_data = ir6[0];
              ir6[0] = (v1823_data + v1822_data);
              float v1825_data = r5[1];
              float v1826_data = ir6[1];
              ir6[1] = (v1826_data + v1825_data);
              float v1828_data = r5[2];
              float v1829_data = ir6[2];
              ir6[2] = (v1829_data + v1828_data);
              float v1831_data = r5[3];
              float v1832_data = ir6[3];
              ir6[3] = (v1832_data + v1831_data);
              float v1834_data = r5[4];
              float v1835_data = ir6[4];
              ir6[4] = (v1835_data + v1834_data);
              float v1837_data = r5[5];
              float v1838_data = ir6[5];
              ir6[5] = (v1838_data + v1837_data);
              float v1840_data = r5[6];
              float v1841_data = ir6[6];
              ir6[6] = (v1841_data + v1840_data);
              float v1843_data = r5[7];
              float v1844_data = ir6[7];
              ir6[7] = (v1844_data + v1843_data);
              float v1846_data = r5[8];
              float v1847_data = ir6[8];
              ir6[8] = (v1847_data + v1846_data);
              float v1849_data = r5[9];
              float v1850_data = ir6[9];
              ir6[9] = (v1850_data + v1849_data);
              float v1852_data = r5[10];
              float v1853_data = ir6[10];
              ir6[10] = (v1853_data + v1852_data);
              float v1855_data = r5[11];
              float v1856_data = ir6[11];
              ir6[11] = (v1856_data + v1855_data);
              // r6 = ir6
              if (v21_g) {
                #pragma unroll
                for (int32_t v1858_n1 = 0; v1858_n1 < 12; ++v1858_n1) {
                  float v1860_data = ir6[v1858_n1];
                  r6[v1858_n1] = v1860_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
              // s0 = store{r>s}(localShrMem0, r6);
              if (v21_g) {
                int32_t v1866_off = v20_lead + 6;
                #pragma unroll
                for (int32_t v1861_i1 = 0; v1861_i1 < 12; ++v1861_i1) {
                  float v1863_data = r6[v1861_i1];
                  int32_t v1868_a = v1866_off + (v1861_i1 * 12);
                  s0[(v1868_a ^ ((v1868_a >> 4) & 15))] = v1863_data;
                }
              }
              float r7[12]{};
              sycl::group_barrier(item.get_sub_group());
              // ir7 = +(s0)
              // [(0, 12), (0, 12)] []
              float ir7[12]{};
              float v1880_data = v30_g ? (s0[(v20_lead ^ ((v20_lead >> 4) & 15))]) : (0.0f);
              float v1881_data = ir7[0];
              ir7[0] = (v1881_data + v1880_data);
              int32_t v1883_a = v20_lead + 12;
              float v1887_data = v30_g ? (s0[(v1883_a ^ ((v1883_a >> 4) & 15))]) : (0.0f);
              float v1888_data = ir7[1];
              ir7[1] = (v1888_data + v1887_data);
              int32_t v1890_a = v20_lead + 24;
              float v1894_data = v30_g ? (s0[(v1890_a ^ ((v1890_a >> 4) & 15))]) : (0.0f);
              float v1895_data = ir7[2];
              ir7[2] = (v1895_data + v1894_data);
              int32_t v1897_a = v20_lead + 36;
              float v1901_data = v30_g ? (s0[(v1897_a ^ ((v1897_a >> 4) & 15))]) : (0.0f);
              float v1902_data = ir7[3];
              ir7[3] = (v1902_data + v1901_data);
              int32_t v1904_a = v20_lead + 48;
              float v1908_data = v30_g ? (s0[(v1904_a ^ ((v1904_a >> 4) & 15))]) : (0.0f);
              float v1909_data = ir7[4];
              ir7[4] = (v1909_data + v1908_data);
              int32_t v1911_a = v20_lead + 60;
              float v1915_data = v30_g ? (s0[(v1911_a ^ ((v1911_a >> 4) & 15))]) : (0.0f);
              float v1916_data = ir7[5];
              ir7[5] = (v1916_data + v1915_data);
              int32_t v1918_a = v20_lead + 72;
              float v1922_data = v30_g ? (s0[(v1918_a ^ ((v1918_a >> 4) & 15))]) : (0.0f);
              float v1923_data = ir7[6];
              ir7[6] = (v1923_data + v1922_data);
              int32_t v1925_a = v20_lead + 84;
              float v1929_data = v30_g ? (s0[(v1925_a ^ ((v1925_a >> 4) & 15))]) : (0.0f);
              float v1930_data = ir7[7];
              ir7[7] = (v1930_data + v1929_data);
              int32_t v1932_a = v20_lead + 96;
              float v1936_data = v30_g ? (s0[(v1932_a ^ ((v1932_a >> 4) & 15))]) : (0.0f);
              float v1937_data = ir7[8];
              ir7[8] = (v1937_data + v1936_data);
              int32_t v1939_a = v20_lead + 108;
              float v1943_data = v30_g ? (s0[(v1939_a ^ ((v1939_a >> 4) & 15))]) : (0.0f);
              float v1944_data = ir7[9];
              ir7[9] = (v1944_data + v1943_data);
              int32_t v1946_a = v20_lead + 120;
              float v1950_data = v30_g ? (s0[(v1946_a ^ ((v1946_a >> 4) & 15))]) : (0.0f);
              float v1951_data = ir7[10];
              ir7[10] = (v1951_data + v1950_data);
              int32_t v1953_a = v20_lead + 132;
              float v1957_data = v30_g ? (s0[(v1953_a ^ ((v1953_a >> 4) & 15))]) : (0.0f);
              float v1958_data = ir7[11];
              ir7[11] = (v1958_data + v1957_data);
              // r7 = ir7
              if (v30_g) {
                #pragma unroll
                for (int32_t v1960_n1 = 0; v1960_n1 < 12; ++v1960_n1) {
                  float v1962_data = ir7[v1960_n1];
                  r7[v1960_n1] = v1962_data;
                }
              }
              // glb_m4 = store{r>g}(r7);
              if (v30_g) {
                #pragma unroll
                for (int32_t v1963_i1 = 0; v1963_i1 < 12; ++v1963_i1) {
                  float v1965_data = r7[v1963_i1];
                  glb_m4[(v20_lead + (v1963_i1 * 12))] = v1965_data;
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

