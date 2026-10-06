// === base name ===
kernel_acfd3b4ce6e44e9c

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_acfd3b4ce6e44e9c = {{16, 16, 1}, 16, 12, 1, 16, 10240, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_acfd3b4ce6e44e9c(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_acfd3b4ce6e44e9c(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_acfd3b4ce6e44e9c(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_acfd3b4ce6e44e9c(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_acfd3b4ce6e44e9c(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_acfd3b4ce6e44e9c(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_acfd3b4ce6e44e9c(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (2560, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
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
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2560}],"shared_bytes":10240,"shared_elements":2560,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B2","bbox":[[0,0],[6,12]],"name":"m0","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"N","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"Y","bbox":[[0,0],[6,12]],"name":"m3","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m4","ordered":false,"parts":1,"shape":[12,12],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[6,12]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[160 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[144];
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v10_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v10_batchId0 < numElements0; v10_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v11_ahead1 = v10_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v13_batchId1 = (v11_ahead1 < numElements0) ? v11_ahead1 : v10_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v10_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v10_batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v10_batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v10_batchId0 * 72 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v10_batchId0 * 72 + 0 + m3_extraOffset];
              float *const __restrict__ glb_m4 = &m4[v10_batchId0 * 144 + 0 + m4_extraOffset];
              float r0[12]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v26_lead = item.get_local_id(2) % 16;
              bool v27_g = v26_lead < 6;
              if (v27_g) {
                #pragma unroll
                for (int32_t v28_i1 = 0; v28_i1 < 12; ++v28_i1) {
                  float v33_data = glb_m0[(v26_lead + (v28_i1 * 6))];
                  r0[v28_i1] = v33_data;
                }
              }
              float r1[12]{};
              // r1 = load{g>r}(glb_m1);
              bool v36_g = v26_lead < 12;
              if (v36_g) {
                #pragma unroll
                for (int32_t v37_i1 = 0; v37_i1 < 12; ++v37_i1) {
                  float v42_data = glb_m1[(v26_lead + (v37_i1 * 12))];
                  r1[v37_i1] = v42_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r3[12]{};
              // r3 = load{g>r}(glb_m2);
              if (v27_g) {
                #pragma unroll
                for (int32_t v45_i1 = 0; v45_i1 < 12; ++v45_i1) {
                  float v50_data = glb_m2[(v26_lead + (v45_i1 * 6))];
                  r3[v45_i1] = v50_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m1););
              float r2[12]{};
              // r2 = +(r0 * r1) + None
              // [(0, 6), (0, 12)] [(0, 12)]
              float v53_data = r0[0];
              float v54_data = r1[0];
              float v55_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v57_data = r2[0];
              r2[0] = (v57_data + (v53_data * v55_bc));
              float v60_data = r1[1];
              float v61_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v63_data = r2[1];
              r2[1] = (v63_data + (v53_data * v61_bc));
              float v66_data = r1[2];
              float v67_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v69_data = r2[2];
              r2[2] = (v69_data + (v53_data * v67_bc));
              float v72_data = r1[3];
              float v73_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v75_data = r2[3];
              r2[3] = (v75_data + (v53_data * v73_bc));
              float v78_data = r1[4];
              float v79_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v81_data = r2[4];
              r2[4] = (v81_data + (v53_data * v79_bc));
              float v84_data = r1[5];
              float v85_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v87_data = r2[5];
              r2[5] = (v87_data + (v53_data * v85_bc));
              float v90_data = r1[6];
              float v91_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v93_data = r2[6];
              r2[6] = (v93_data + (v53_data * v91_bc));
              float v96_data = r1[7];
              float v97_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v99_data = r2[7];
              r2[7] = (v99_data + (v53_data * v97_bc));
              float v102_data = r1[8];
              float v103_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v105_data = r2[8];
              r2[8] = (v105_data + (v53_data * v103_bc));
              float v108_data = r1[9];
              float v109_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v111_data = r2[9];
              r2[9] = (v111_data + (v53_data * v109_bc));
              float v114_data = r1[10];
              float v115_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v117_data = r2[10];
              r2[10] = (v117_data + (v53_data * v115_bc));
              float v120_data = r1[11];
              float v121_bc = sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v123_data = r2[11];
              r2[11] = (v123_data + (v53_data * v121_bc));
              float v125_data = r0[1];
              float v127_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v129_data = r2[0];
              r2[0] = (v129_data + (v125_data * v127_bc));
              float v133_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v135_data = r2[1];
              r2[1] = (v135_data + (v125_data * v133_bc));
              float v139_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v141_data = r2[2];
              r2[2] = (v141_data + (v125_data * v139_bc));
              float v145_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v147_data = r2[3];
              r2[3] = (v147_data + (v125_data * v145_bc));
              float v151_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v153_data = r2[4];
              r2[4] = (v153_data + (v125_data * v151_bc));
              float v157_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v159_data = r2[5];
              r2[5] = (v159_data + (v125_data * v157_bc));
              float v163_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v165_data = r2[6];
              r2[6] = (v165_data + (v125_data * v163_bc));
              float v169_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v171_data = r2[7];
              r2[7] = (v171_data + (v125_data * v169_bc));
              float v175_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v177_data = r2[8];
              r2[8] = (v177_data + (v125_data * v175_bc));
              float v181_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v183_data = r2[9];
              r2[9] = (v183_data + (v125_data * v181_bc));
              float v187_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v189_data = r2[10];
              r2[10] = (v189_data + (v125_data * v187_bc));
              float v193_bc = sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v195_data = r2[11];
              r2[11] = (v195_data + (v125_data * v193_bc));
              float v197_data = r0[2];
              float v199_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v201_data = r2[0];
              r2[0] = (v201_data + (v197_data * v199_bc));
              float v205_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v207_data = r2[1];
              r2[1] = (v207_data + (v197_data * v205_bc));
              float v211_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v213_data = r2[2];
              r2[2] = (v213_data + (v197_data * v211_bc));
              float v217_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v219_data = r2[3];
              r2[3] = (v219_data + (v197_data * v217_bc));
              float v223_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v225_data = r2[4];
              r2[4] = (v225_data + (v197_data * v223_bc));
              float v229_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v231_data = r2[5];
              r2[5] = (v231_data + (v197_data * v229_bc));
              float v235_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v237_data = r2[6];
              r2[6] = (v237_data + (v197_data * v235_bc));
              float v241_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v243_data = r2[7];
              r2[7] = (v243_data + (v197_data * v241_bc));
              float v247_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v249_data = r2[8];
              r2[8] = (v249_data + (v197_data * v247_bc));
              float v253_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v255_data = r2[9];
              r2[9] = (v255_data + (v197_data * v253_bc));
              float v259_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v261_data = r2[10];
              r2[10] = (v261_data + (v197_data * v259_bc));
              float v265_bc = sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v267_data = r2[11];
              r2[11] = (v267_data + (v197_data * v265_bc));
              float v269_data = r0[3];
              float v271_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v273_data = r2[0];
              r2[0] = (v273_data + (v269_data * v271_bc));
              float v277_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v279_data = r2[1];
              r2[1] = (v279_data + (v269_data * v277_bc));
              float v283_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v285_data = r2[2];
              r2[2] = (v285_data + (v269_data * v283_bc));
              float v289_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v291_data = r2[3];
              r2[3] = (v291_data + (v269_data * v289_bc));
              float v295_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v297_data = r2[4];
              r2[4] = (v297_data + (v269_data * v295_bc));
              float v301_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v303_data = r2[5];
              r2[5] = (v303_data + (v269_data * v301_bc));
              float v307_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v309_data = r2[6];
              r2[6] = (v309_data + (v269_data * v307_bc));
              float v313_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v315_data = r2[7];
              r2[7] = (v315_data + (v269_data * v313_bc));
              float v319_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v321_data = r2[8];
              r2[8] = (v321_data + (v269_data * v319_bc));
              float v325_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v327_data = r2[9];
              r2[9] = (v327_data + (v269_data * v325_bc));
              float v331_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v333_data = r2[10];
              r2[10] = (v333_data + (v269_data * v331_bc));
              float v337_bc = sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v339_data = r2[11];
              r2[11] = (v339_data + (v269_data * v337_bc));
              float v341_data = r0[4];
              float v343_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v345_data = r2[0];
              r2[0] = (v345_data + (v341_data * v343_bc));
              float v349_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v351_data = r2[1];
              r2[1] = (v351_data + (v341_data * v349_bc));
              float v355_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v357_data = r2[2];
              r2[2] = (v357_data + (v341_data * v355_bc));
              float v361_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v363_data = r2[3];
              r2[3] = (v363_data + (v341_data * v361_bc));
              float v367_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v369_data = r2[4];
              r2[4] = (v369_data + (v341_data * v367_bc));
              float v373_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v375_data = r2[5];
              r2[5] = (v375_data + (v341_data * v373_bc));
              float v379_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v381_data = r2[6];
              r2[6] = (v381_data + (v341_data * v379_bc));
              float v385_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v387_data = r2[7];
              r2[7] = (v387_data + (v341_data * v385_bc));
              float v391_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v393_data = r2[8];
              r2[8] = (v393_data + (v341_data * v391_bc));
              float v397_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v399_data = r2[9];
              r2[9] = (v399_data + (v341_data * v397_bc));
              float v403_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v405_data = r2[10];
              r2[10] = (v405_data + (v341_data * v403_bc));
              float v409_bc = sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v411_data = r2[11];
              r2[11] = (v411_data + (v341_data * v409_bc));
              float v413_data = r0[5];
              float v415_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v417_data = r2[0];
              r2[0] = (v417_data + (v413_data * v415_bc));
              float v421_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v423_data = r2[1];
              r2[1] = (v423_data + (v413_data * v421_bc));
              float v427_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v429_data = r2[2];
              r2[2] = (v429_data + (v413_data * v427_bc));
              float v433_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v435_data = r2[3];
              r2[3] = (v435_data + (v413_data * v433_bc));
              float v439_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v441_data = r2[4];
              r2[4] = (v441_data + (v413_data * v439_bc));
              float v445_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v447_data = r2[5];
              r2[5] = (v447_data + (v413_data * v445_bc));
              float v451_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v453_data = r2[6];
              r2[6] = (v453_data + (v413_data * v451_bc));
              float v457_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v459_data = r2[7];
              r2[7] = (v459_data + (v413_data * v457_bc));
              float v463_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v465_data = r2[8];
              r2[8] = (v465_data + (v413_data * v463_bc));
              float v469_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v471_data = r2[9];
              r2[9] = (v471_data + (v413_data * v469_bc));
              float v475_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v477_data = r2[10];
              r2[10] = (v477_data + (v413_data * v475_bc));
              float v481_bc = sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v483_data = r2[11];
              r2[11] = (v483_data + (v413_data * v481_bc));
              float v485_data = r0[6];
              float v487_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v489_data = r2[0];
              r2[0] = (v489_data + (v485_data * v487_bc));
              float v493_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v495_data = r2[1];
              r2[1] = (v495_data + (v485_data * v493_bc));
              float v499_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v501_data = r2[2];
              r2[2] = (v501_data + (v485_data * v499_bc));
              float v505_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v507_data = r2[3];
              r2[3] = (v507_data + (v485_data * v505_bc));
              float v511_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v513_data = r2[4];
              r2[4] = (v513_data + (v485_data * v511_bc));
              float v517_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v519_data = r2[5];
              r2[5] = (v519_data + (v485_data * v517_bc));
              float v523_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v525_data = r2[6];
              r2[6] = (v525_data + (v485_data * v523_bc));
              float v529_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v531_data = r2[7];
              r2[7] = (v531_data + (v485_data * v529_bc));
              float v535_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v537_data = r2[8];
              r2[8] = (v537_data + (v485_data * v535_bc));
              float v541_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v543_data = r2[9];
              r2[9] = (v543_data + (v485_data * v541_bc));
              float v547_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v549_data = r2[10];
              r2[10] = (v549_data + (v485_data * v547_bc));
              float v553_bc = sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v555_data = r2[11];
              r2[11] = (v555_data + (v485_data * v553_bc));
              float v557_data = r0[7];
              float v559_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v561_data = r2[0];
              r2[0] = (v561_data + (v557_data * v559_bc));
              float v565_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v567_data = r2[1];
              r2[1] = (v567_data + (v557_data * v565_bc));
              float v571_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v573_data = r2[2];
              r2[2] = (v573_data + (v557_data * v571_bc));
              float v577_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v579_data = r2[3];
              r2[3] = (v579_data + (v557_data * v577_bc));
              float v583_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v585_data = r2[4];
              r2[4] = (v585_data + (v557_data * v583_bc));
              float v589_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v591_data = r2[5];
              r2[5] = (v591_data + (v557_data * v589_bc));
              float v595_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v597_data = r2[6];
              r2[6] = (v597_data + (v557_data * v595_bc));
              float v601_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v603_data = r2[7];
              r2[7] = (v603_data + (v557_data * v601_bc));
              float v607_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v609_data = r2[8];
              r2[8] = (v609_data + (v557_data * v607_bc));
              float v613_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v615_data = r2[9];
              r2[9] = (v615_data + (v557_data * v613_bc));
              float v619_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v621_data = r2[10];
              r2[10] = (v621_data + (v557_data * v619_bc));
              float v625_bc = sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v627_data = r2[11];
              r2[11] = (v627_data + (v557_data * v625_bc));
              float v629_data = r0[8];
              float v631_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v633_data = r2[0];
              r2[0] = (v633_data + (v629_data * v631_bc));
              float v637_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v639_data = r2[1];
              r2[1] = (v639_data + (v629_data * v637_bc));
              float v643_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v645_data = r2[2];
              r2[2] = (v645_data + (v629_data * v643_bc));
              float v649_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v651_data = r2[3];
              r2[3] = (v651_data + (v629_data * v649_bc));
              float v655_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v657_data = r2[4];
              r2[4] = (v657_data + (v629_data * v655_bc));
              float v661_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v663_data = r2[5];
              r2[5] = (v663_data + (v629_data * v661_bc));
              float v667_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v669_data = r2[6];
              r2[6] = (v669_data + (v629_data * v667_bc));
              float v673_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v675_data = r2[7];
              r2[7] = (v675_data + (v629_data * v673_bc));
              float v679_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v681_data = r2[8];
              r2[8] = (v681_data + (v629_data * v679_bc));
              float v685_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v687_data = r2[9];
              r2[9] = (v687_data + (v629_data * v685_bc));
              float v691_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v693_data = r2[10];
              r2[10] = (v693_data + (v629_data * v691_bc));
              float v697_bc = sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v699_data = r2[11];
              r2[11] = (v699_data + (v629_data * v697_bc));
              float v701_data = r0[9];
              float v703_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v705_data = r2[0];
              r2[0] = (v705_data + (v701_data * v703_bc));
              float v709_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v711_data = r2[1];
              r2[1] = (v711_data + (v701_data * v709_bc));
              float v715_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v717_data = r2[2];
              r2[2] = (v717_data + (v701_data * v715_bc));
              float v721_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v723_data = r2[3];
              r2[3] = (v723_data + (v701_data * v721_bc));
              float v727_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v729_data = r2[4];
              r2[4] = (v729_data + (v701_data * v727_bc));
              float v733_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v735_data = r2[5];
              r2[5] = (v735_data + (v701_data * v733_bc));
              float v739_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v741_data = r2[6];
              r2[6] = (v741_data + (v701_data * v739_bc));
              float v745_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v747_data = r2[7];
              r2[7] = (v747_data + (v701_data * v745_bc));
              float v751_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v753_data = r2[8];
              r2[8] = (v753_data + (v701_data * v751_bc));
              float v757_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v759_data = r2[9];
              r2[9] = (v759_data + (v701_data * v757_bc));
              float v763_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v765_data = r2[10];
              r2[10] = (v765_data + (v701_data * v763_bc));
              float v769_bc = sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v771_data = r2[11];
              r2[11] = (v771_data + (v701_data * v769_bc));
              float v773_data = r0[10];
              float v775_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v777_data = r2[0];
              r2[0] = (v777_data + (v773_data * v775_bc));
              float v781_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v783_data = r2[1];
              r2[1] = (v783_data + (v773_data * v781_bc));
              float v787_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v789_data = r2[2];
              r2[2] = (v789_data + (v773_data * v787_bc));
              float v793_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v795_data = r2[3];
              r2[3] = (v795_data + (v773_data * v793_bc));
              float v799_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v801_data = r2[4];
              r2[4] = (v801_data + (v773_data * v799_bc));
              float v805_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v807_data = r2[5];
              r2[5] = (v807_data + (v773_data * v805_bc));
              float v811_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v813_data = r2[6];
              r2[6] = (v813_data + (v773_data * v811_bc));
              float v817_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v819_data = r2[7];
              r2[7] = (v819_data + (v773_data * v817_bc));
              float v823_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v825_data = r2[8];
              r2[8] = (v825_data + (v773_data * v823_bc));
              float v829_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v831_data = r2[9];
              r2[9] = (v831_data + (v773_data * v829_bc));
              float v835_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v837_data = r2[10];
              r2[10] = (v837_data + (v773_data * v835_bc));
              float v841_bc = sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v843_data = r2[11];
              r2[11] = (v843_data + (v773_data * v841_bc));
              float v845_data = r0[11];
              float v847_bc = sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v849_data = r2[0];
              r2[0] = (v849_data + (v845_data * v847_bc));
              float v853_bc = sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v855_data = r2[1];
              r2[1] = (v855_data + (v845_data * v853_bc));
              float v859_bc = sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v861_data = r2[2];
              r2[2] = (v861_data + (v845_data * v859_bc));
              float v865_bc = sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v867_data = r2[3];
              r2[3] = (v867_data + (v845_data * v865_bc));
              float v871_bc = sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v873_data = r2[4];
              r2[4] = (v873_data + (v845_data * v871_bc));
              float v877_bc = sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v879_data = r2[5];
              r2[5] = (v879_data + (v845_data * v877_bc));
              float v883_bc = sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v885_data = r2[6];
              r2[6] = (v885_data + (v845_data * v883_bc));
              float v889_bc = sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v891_data = r2[7];
              r2[7] = (v891_data + (v845_data * v889_bc));
              float v895_bc = sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v897_data = r2[8];
              r2[8] = (v897_data + (v845_data * v895_bc));
              float v901_bc = sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v903_data = r2[9];
              r2[9] = (v903_data + (v845_data * v901_bc));
              float v907_bc = sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v909_data = r2[10];
              r2[10] = (v909_data + (v845_data * v907_bc));
              float v913_bc = sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v915_data = r2[11];
              r2[11] = (v915_data + (v845_data * v913_bc));
              // s0 = store{r>s}(localShrMem0, r2);
              if (v27_g) {
                int32_t v922_off = v26_lead + 6;
                #pragma unroll
                for (int32_t v917_i1 = 0; v917_i1 < 12; ++v917_i1) {
                  float v919_data = r2[v917_i1];
                  int32_t v924_a = v922_off + (v917_i1 * 12);
                  s0[(v924_a ^ ((v924_a >> 4) & 15))] = v919_data;
                }
              }
              float r5[12]{};
              // r5 = load{g>r}(glb_m3);
              if (v27_g) {
                #pragma unroll
                for (int32_t v929_i1 = 0; v929_i1 < 12; ++v929_i1) {
                  float v934_data = glb_m3[(v26_lead + (v929_i1 * 6))];
                  r5[v929_i1] = v934_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m2););
              float r4[12]{};
              // ir4 = +(r3 * r1)
              // [(0, 6), (0, 12)] [(0, 12)]
              float ir4[12]{};
              float v938_data = r3[0];
              float v942_data = ir4[0];
              ir4[0] = (v942_data + (v938_data * v55_bc));
              float v948_data = ir4[1];
              ir4[1] = (v948_data + (v938_data * v61_bc));
              float v954_data = ir4[2];
              ir4[2] = (v954_data + (v938_data * v67_bc));
              float v960_data = ir4[3];
              ir4[3] = (v960_data + (v938_data * v73_bc));
              float v966_data = ir4[4];
              ir4[4] = (v966_data + (v938_data * v79_bc));
              float v972_data = ir4[5];
              ir4[5] = (v972_data + (v938_data * v85_bc));
              float v978_data = ir4[6];
              ir4[6] = (v978_data + (v938_data * v91_bc));
              float v984_data = ir4[7];
              ir4[7] = (v984_data + (v938_data * v97_bc));
              float v990_data = ir4[8];
              ir4[8] = (v990_data + (v938_data * v103_bc));
              float v996_data = ir4[9];
              ir4[9] = (v996_data + (v938_data * v109_bc));
              float v1002_data = ir4[10];
              ir4[10] = (v1002_data + (v938_data * v115_bc));
              float v1008_data = ir4[11];
              ir4[11] = (v1008_data + (v938_data * v121_bc));
              float v1010_data = r3[1];
              float v1014_data = ir4[0];
              ir4[0] = (v1014_data + (v1010_data * v127_bc));
              float v1020_data = ir4[1];
              ir4[1] = (v1020_data + (v1010_data * v133_bc));
              float v1026_data = ir4[2];
              ir4[2] = (v1026_data + (v1010_data * v139_bc));
              float v1032_data = ir4[3];
              ir4[3] = (v1032_data + (v1010_data * v145_bc));
              float v1038_data = ir4[4];
              ir4[4] = (v1038_data + (v1010_data * v151_bc));
              float v1044_data = ir4[5];
              ir4[5] = (v1044_data + (v1010_data * v157_bc));
              float v1050_data = ir4[6];
              ir4[6] = (v1050_data + (v1010_data * v163_bc));
              float v1056_data = ir4[7];
              ir4[7] = (v1056_data + (v1010_data * v169_bc));
              float v1062_data = ir4[8];
              ir4[8] = (v1062_data + (v1010_data * v175_bc));
              float v1068_data = ir4[9];
              ir4[9] = (v1068_data + (v1010_data * v181_bc));
              float v1074_data = ir4[10];
              ir4[10] = (v1074_data + (v1010_data * v187_bc));
              float v1080_data = ir4[11];
              ir4[11] = (v1080_data + (v1010_data * v193_bc));
              float v1082_data = r3[2];
              float v1086_data = ir4[0];
              ir4[0] = (v1086_data + (v1082_data * v199_bc));
              float v1092_data = ir4[1];
              ir4[1] = (v1092_data + (v1082_data * v205_bc));
              float v1098_data = ir4[2];
              ir4[2] = (v1098_data + (v1082_data * v211_bc));
              float v1104_data = ir4[3];
              ir4[3] = (v1104_data + (v1082_data * v217_bc));
              float v1110_data = ir4[4];
              ir4[4] = (v1110_data + (v1082_data * v223_bc));
              float v1116_data = ir4[5];
              ir4[5] = (v1116_data + (v1082_data * v229_bc));
              float v1122_data = ir4[6];
              ir4[6] = (v1122_data + (v1082_data * v235_bc));
              float v1128_data = ir4[7];
              ir4[7] = (v1128_data + (v1082_data * v241_bc));
              float v1134_data = ir4[8];
              ir4[8] = (v1134_data + (v1082_data * v247_bc));
              float v1140_data = ir4[9];
              ir4[9] = (v1140_data + (v1082_data * v253_bc));
              float v1146_data = ir4[10];
              ir4[10] = (v1146_data + (v1082_data * v259_bc));
              float v1152_data = ir4[11];
              ir4[11] = (v1152_data + (v1082_data * v265_bc));
              float v1154_data = r3[3];
              float v1158_data = ir4[0];
              ir4[0] = (v1158_data + (v1154_data * v271_bc));
              float v1164_data = ir4[1];
              ir4[1] = (v1164_data + (v1154_data * v277_bc));
              float v1170_data = ir4[2];
              ir4[2] = (v1170_data + (v1154_data * v283_bc));
              float v1176_data = ir4[3];
              ir4[3] = (v1176_data + (v1154_data * v289_bc));
              float v1182_data = ir4[4];
              ir4[4] = (v1182_data + (v1154_data * v295_bc));
              float v1188_data = ir4[5];
              ir4[5] = (v1188_data + (v1154_data * v301_bc));
              float v1194_data = ir4[6];
              ir4[6] = (v1194_data + (v1154_data * v307_bc));
              float v1200_data = ir4[7];
              ir4[7] = (v1200_data + (v1154_data * v313_bc));
              float v1206_data = ir4[8];
              ir4[8] = (v1206_data + (v1154_data * v319_bc));
              float v1212_data = ir4[9];
              ir4[9] = (v1212_data + (v1154_data * v325_bc));
              float v1218_data = ir4[10];
              ir4[10] = (v1218_data + (v1154_data * v331_bc));
              float v1224_data = ir4[11];
              ir4[11] = (v1224_data + (v1154_data * v337_bc));
              float v1226_data = r3[4];
              float v1230_data = ir4[0];
              ir4[0] = (v1230_data + (v1226_data * v343_bc));
              float v1236_data = ir4[1];
              ir4[1] = (v1236_data + (v1226_data * v349_bc));
              float v1242_data = ir4[2];
              ir4[2] = (v1242_data + (v1226_data * v355_bc));
              float v1248_data = ir4[3];
              ir4[3] = (v1248_data + (v1226_data * v361_bc));
              float v1254_data = ir4[4];
              ir4[4] = (v1254_data + (v1226_data * v367_bc));
              float v1260_data = ir4[5];
              ir4[5] = (v1260_data + (v1226_data * v373_bc));
              float v1266_data = ir4[6];
              ir4[6] = (v1266_data + (v1226_data * v379_bc));
              float v1272_data = ir4[7];
              ir4[7] = (v1272_data + (v1226_data * v385_bc));
              float v1278_data = ir4[8];
              ir4[8] = (v1278_data + (v1226_data * v391_bc));
              float v1284_data = ir4[9];
              ir4[9] = (v1284_data + (v1226_data * v397_bc));
              float v1290_data = ir4[10];
              ir4[10] = (v1290_data + (v1226_data * v403_bc));
              float v1296_data = ir4[11];
              ir4[11] = (v1296_data + (v1226_data * v409_bc));
              float v1298_data = r3[5];
              float v1302_data = ir4[0];
              ir4[0] = (v1302_data + (v1298_data * v415_bc));
              float v1308_data = ir4[1];
              ir4[1] = (v1308_data + (v1298_data * v421_bc));
              float v1314_data = ir4[2];
              ir4[2] = (v1314_data + (v1298_data * v427_bc));
              float v1320_data = ir4[3];
              ir4[3] = (v1320_data + (v1298_data * v433_bc));
              float v1326_data = ir4[4];
              ir4[4] = (v1326_data + (v1298_data * v439_bc));
              float v1332_data = ir4[5];
              ir4[5] = (v1332_data + (v1298_data * v445_bc));
              float v1338_data = ir4[6];
              ir4[6] = (v1338_data + (v1298_data * v451_bc));
              float v1344_data = ir4[7];
              ir4[7] = (v1344_data + (v1298_data * v457_bc));
              float v1350_data = ir4[8];
              ir4[8] = (v1350_data + (v1298_data * v463_bc));
              float v1356_data = ir4[9];
              ir4[9] = (v1356_data + (v1298_data * v469_bc));
              float v1362_data = ir4[10];
              ir4[10] = (v1362_data + (v1298_data * v475_bc));
              float v1368_data = ir4[11];
              ir4[11] = (v1368_data + (v1298_data * v481_bc));
              float v1370_data = r3[6];
              float v1374_data = ir4[0];
              ir4[0] = (v1374_data + (v1370_data * v487_bc));
              float v1380_data = ir4[1];
              ir4[1] = (v1380_data + (v1370_data * v493_bc));
              float v1386_data = ir4[2];
              ir4[2] = (v1386_data + (v1370_data * v499_bc));
              float v1392_data = ir4[3];
              ir4[3] = (v1392_data + (v1370_data * v505_bc));
              float v1398_data = ir4[4];
              ir4[4] = (v1398_data + (v1370_data * v511_bc));
              float v1404_data = ir4[5];
              ir4[5] = (v1404_data + (v1370_data * v517_bc));
              float v1410_data = ir4[6];
              ir4[6] = (v1410_data + (v1370_data * v523_bc));
              float v1416_data = ir4[7];
              ir4[7] = (v1416_data + (v1370_data * v529_bc));
              float v1422_data = ir4[8];
              ir4[8] = (v1422_data + (v1370_data * v535_bc));
              float v1428_data = ir4[9];
              ir4[9] = (v1428_data + (v1370_data * v541_bc));
              float v1434_data = ir4[10];
              ir4[10] = (v1434_data + (v1370_data * v547_bc));
              float v1440_data = ir4[11];
              ir4[11] = (v1440_data + (v1370_data * v553_bc));
              float v1442_data = r3[7];
              float v1446_data = ir4[0];
              ir4[0] = (v1446_data + (v1442_data * v559_bc));
              float v1452_data = ir4[1];
              ir4[1] = (v1452_data + (v1442_data * v565_bc));
              float v1458_data = ir4[2];
              ir4[2] = (v1458_data + (v1442_data * v571_bc));
              float v1464_data = ir4[3];
              ir4[3] = (v1464_data + (v1442_data * v577_bc));
              float v1470_data = ir4[4];
              ir4[4] = (v1470_data + (v1442_data * v583_bc));
              float v1476_data = ir4[5];
              ir4[5] = (v1476_data + (v1442_data * v589_bc));
              float v1482_data = ir4[6];
              ir4[6] = (v1482_data + (v1442_data * v595_bc));
              float v1488_data = ir4[7];
              ir4[7] = (v1488_data + (v1442_data * v601_bc));
              float v1494_data = ir4[8];
              ir4[8] = (v1494_data + (v1442_data * v607_bc));
              float v1500_data = ir4[9];
              ir4[9] = (v1500_data + (v1442_data * v613_bc));
              float v1506_data = ir4[10];
              ir4[10] = (v1506_data + (v1442_data * v619_bc));
              float v1512_data = ir4[11];
              ir4[11] = (v1512_data + (v1442_data * v625_bc));
              float v1514_data = r3[8];
              float v1518_data = ir4[0];
              ir4[0] = (v1518_data + (v1514_data * v631_bc));
              float v1524_data = ir4[1];
              ir4[1] = (v1524_data + (v1514_data * v637_bc));
              float v1530_data = ir4[2];
              ir4[2] = (v1530_data + (v1514_data * v643_bc));
              float v1536_data = ir4[3];
              ir4[3] = (v1536_data + (v1514_data * v649_bc));
              float v1542_data = ir4[4];
              ir4[4] = (v1542_data + (v1514_data * v655_bc));
              float v1548_data = ir4[5];
              ir4[5] = (v1548_data + (v1514_data * v661_bc));
              float v1554_data = ir4[6];
              ir4[6] = (v1554_data + (v1514_data * v667_bc));
              float v1560_data = ir4[7];
              ir4[7] = (v1560_data + (v1514_data * v673_bc));
              float v1566_data = ir4[8];
              ir4[8] = (v1566_data + (v1514_data * v679_bc));
              float v1572_data = ir4[9];
              ir4[9] = (v1572_data + (v1514_data * v685_bc));
              float v1578_data = ir4[10];
              ir4[10] = (v1578_data + (v1514_data * v691_bc));
              float v1584_data = ir4[11];
              ir4[11] = (v1584_data + (v1514_data * v697_bc));
              float v1586_data = r3[9];
              float v1590_data = ir4[0];
              ir4[0] = (v1590_data + (v1586_data * v703_bc));
              float v1596_data = ir4[1];
              ir4[1] = (v1596_data + (v1586_data * v709_bc));
              float v1602_data = ir4[2];
              ir4[2] = (v1602_data + (v1586_data * v715_bc));
              float v1608_data = ir4[3];
              ir4[3] = (v1608_data + (v1586_data * v721_bc));
              float v1614_data = ir4[4];
              ir4[4] = (v1614_data + (v1586_data * v727_bc));
              float v1620_data = ir4[5];
              ir4[5] = (v1620_data + (v1586_data * v733_bc));
              float v1626_data = ir4[6];
              ir4[6] = (v1626_data + (v1586_data * v739_bc));
              float v1632_data = ir4[7];
              ir4[7] = (v1632_data + (v1586_data * v745_bc));
              float v1638_data = ir4[8];
              ir4[8] = (v1638_data + (v1586_data * v751_bc));
              float v1644_data = ir4[9];
              ir4[9] = (v1644_data + (v1586_data * v757_bc));
              float v1650_data = ir4[10];
              ir4[10] = (v1650_data + (v1586_data * v763_bc));
              float v1656_data = ir4[11];
              ir4[11] = (v1656_data + (v1586_data * v769_bc));
              float v1658_data = r3[10];
              float v1662_data = ir4[0];
              ir4[0] = (v1662_data + (v1658_data * v775_bc));
              float v1668_data = ir4[1];
              ir4[1] = (v1668_data + (v1658_data * v781_bc));
              float v1674_data = ir4[2];
              ir4[2] = (v1674_data + (v1658_data * v787_bc));
              float v1680_data = ir4[3];
              ir4[3] = (v1680_data + (v1658_data * v793_bc));
              float v1686_data = ir4[4];
              ir4[4] = (v1686_data + (v1658_data * v799_bc));
              float v1692_data = ir4[5];
              ir4[5] = (v1692_data + (v1658_data * v805_bc));
              float v1698_data = ir4[6];
              ir4[6] = (v1698_data + (v1658_data * v811_bc));
              float v1704_data = ir4[7];
              ir4[7] = (v1704_data + (v1658_data * v817_bc));
              float v1710_data = ir4[8];
              ir4[8] = (v1710_data + (v1658_data * v823_bc));
              float v1716_data = ir4[9];
              ir4[9] = (v1716_data + (v1658_data * v829_bc));
              float v1722_data = ir4[10];
              ir4[10] = (v1722_data + (v1658_data * v835_bc));
              float v1728_data = ir4[11];
              ir4[11] = (v1728_data + (v1658_data * v841_bc));
              float v1730_data = r3[11];
              float v1734_data = ir4[0];
              ir4[0] = (v1734_data + (v1730_data * v847_bc));
              float v1740_data = ir4[1];
              ir4[1] = (v1740_data + (v1730_data * v853_bc));
              float v1746_data = ir4[2];
              ir4[2] = (v1746_data + (v1730_data * v859_bc));
              float v1752_data = ir4[3];
              ir4[3] = (v1752_data + (v1730_data * v865_bc));
              float v1758_data = ir4[4];
              ir4[4] = (v1758_data + (v1730_data * v871_bc));
              float v1764_data = ir4[5];
              ir4[5] = (v1764_data + (v1730_data * v877_bc));
              float v1770_data = ir4[6];
              ir4[6] = (v1770_data + (v1730_data * v883_bc));
              float v1776_data = ir4[7];
              ir4[7] = (v1776_data + (v1730_data * v889_bc));
              float v1782_data = ir4[8];
              ir4[8] = (v1782_data + (v1730_data * v895_bc));
              float v1788_data = ir4[9];
              ir4[9] = (v1788_data + (v1730_data * v901_bc));
              float v1794_data = ir4[10];
              ir4[10] = (v1794_data + (v1730_data * v907_bc));
              float v1800_data = ir4[11];
              ir4[11] = (v1800_data + (v1730_data * v913_bc));
              // r4 = ir4
              if (v27_g) {
                #pragma unroll
                for (int32_t v1802_n1 = 0; v1802_n1 < 12; ++v1802_n1) {
                  float v1804_data = ir4[v1802_n1];
                  r4[v1802_n1] = v1804_data;
                }
              }
              // s0 = store{r>s, clear}(localShrMem0, r4);
              if ((v26_lead >= 6) && v36_g) {
                #pragma unroll
                for (int32_t v1807_z1 = 0; v1807_z1 < 12; ++v1807_z1) {
                  int32_t v1812_a = v26_lead + (v1807_z1 * 12);
                  s0[(v1812_a ^ ((v1812_a >> 4) & 15))] = 0.0f;
                }
              }
              if (v27_g) {
                #pragma unroll
                for (int32_t v1816_i1 = 0; v1816_i1 < 12; ++v1816_i1) {
                  float v1818_data = r4[v1816_i1];
                  int32_t v1822_a = v26_lead + (v1816_i1 * 12);
                  s0[(v1822_a ^ ((v1822_a >> 4) & 15))] = v1818_data;
                }
              }
              // wait(r5 = load{g>r}(glb_m3););
              float r6[12]{};
              // ir6 = +(r5)
              // [(0, 6), (0, 12)] []
              float ir6[12]{};
              float v1828_data = r5[0];
              float v1829_data = ir6[0];
              ir6[0] = (v1829_data + v1828_data);
              float v1831_data = r5[1];
              float v1832_data = ir6[1];
              ir6[1] = (v1832_data + v1831_data);
              float v1834_data = r5[2];
              float v1835_data = ir6[2];
              ir6[2] = (v1835_data + v1834_data);
              float v1837_data = r5[3];
              float v1838_data = ir6[3];
              ir6[3] = (v1838_data + v1837_data);
              float v1840_data = r5[4];
              float v1841_data = ir6[4];
              ir6[4] = (v1841_data + v1840_data);
              float v1843_data = r5[5];
              float v1844_data = ir6[5];
              ir6[5] = (v1844_data + v1843_data);
              float v1846_data = r5[6];
              float v1847_data = ir6[6];
              ir6[6] = (v1847_data + v1846_data);
              float v1849_data = r5[7];
              float v1850_data = ir6[7];
              ir6[7] = (v1850_data + v1849_data);
              float v1852_data = r5[8];
              float v1853_data = ir6[8];
              ir6[8] = (v1853_data + v1852_data);
              float v1855_data = r5[9];
              float v1856_data = ir6[9];
              ir6[9] = (v1856_data + v1855_data);
              float v1858_data = r5[10];
              float v1859_data = ir6[10];
              ir6[10] = (v1859_data + v1858_data);
              float v1861_data = r5[11];
              float v1862_data = ir6[11];
              ir6[11] = (v1862_data + v1861_data);
              // r6 = ir6
              if (v27_g) {
                #pragma unroll
                for (int32_t v1864_n1 = 0; v1864_n1 < 12; ++v1864_n1) {
                  float v1866_data = ir6[v1864_n1];
                  r6[v1864_n1] = v1866_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
              // s0 = store{r>s}(localShrMem0, r6);
              if (v27_g) {
                int32_t v1872_off = v26_lead + 6;
                #pragma unroll
                for (int32_t v1867_i1 = 0; v1867_i1 < 12; ++v1867_i1) {
                  float v1869_data = r6[v1867_i1];
                  int32_t v1874_a = v1872_off + (v1867_i1 * 12);
                  s0[(v1874_a ^ ((v1874_a >> 4) & 15))] = v1869_data;
                }
              }
              float r7[12]{};
              sycl::group_barrier(item.get_sub_group());
              // ir7 = +(s0)
              // [(0, 12), (0, 12)] []
              float ir7[12]{};
              float v1886_data_pre = s0[v36_g ? ((v26_lead ^ ((v26_lead >> 4) & 15))) : (0)];
              float v1886_data = v36_g ? (v1886_data_pre) : (0.0f);
              float v1887_data = ir7[0];
              ir7[0] = (v1887_data + v1886_data);
              int32_t v1889_a = v26_lead + 12;
              float v1893_data_pre = s0[v36_g ? ((v1889_a ^ ((v1889_a >> 4) & 15))) : (0)];
              float v1893_data = v36_g ? (v1893_data_pre) : (0.0f);
              float v1894_data = ir7[1];
              ir7[1] = (v1894_data + v1893_data);
              int32_t v1896_a = v26_lead + 24;
              float v1900_data_pre = s0[v36_g ? ((v1896_a ^ ((v1896_a >> 4) & 15))) : (0)];
              float v1900_data = v36_g ? (v1900_data_pre) : (0.0f);
              float v1901_data = ir7[2];
              ir7[2] = (v1901_data + v1900_data);
              int32_t v1903_a = v26_lead + 36;
              float v1907_data_pre = s0[v36_g ? ((v1903_a ^ ((v1903_a >> 4) & 15))) : (0)];
              float v1907_data = v36_g ? (v1907_data_pre) : (0.0f);
              float v1908_data = ir7[3];
              ir7[3] = (v1908_data + v1907_data);
              int32_t v1910_a = v26_lead + 48;
              float v1914_data_pre = s0[v36_g ? ((v1910_a ^ ((v1910_a >> 4) & 15))) : (0)];
              float v1914_data = v36_g ? (v1914_data_pre) : (0.0f);
              float v1915_data = ir7[4];
              ir7[4] = (v1915_data + v1914_data);
              int32_t v1917_a = v26_lead + 60;
              float v1921_data_pre = s0[v36_g ? ((v1917_a ^ ((v1917_a >> 4) & 15))) : (0)];
              float v1921_data = v36_g ? (v1921_data_pre) : (0.0f);
              float v1922_data = ir7[5];
              ir7[5] = (v1922_data + v1921_data);
              int32_t v1924_a = v26_lead + 72;
              float v1928_data_pre = s0[v36_g ? ((v1924_a ^ ((v1924_a >> 4) & 15))) : (0)];
              float v1928_data = v36_g ? (v1928_data_pre) : (0.0f);
              float v1929_data = ir7[6];
              ir7[6] = (v1929_data + v1928_data);
              int32_t v1931_a = v26_lead + 84;
              float v1935_data_pre = s0[v36_g ? ((v1931_a ^ ((v1931_a >> 4) & 15))) : (0)];
              float v1935_data = v36_g ? (v1935_data_pre) : (0.0f);
              float v1936_data = ir7[7];
              ir7[7] = (v1936_data + v1935_data);
              int32_t v1938_a = v26_lead + 96;
              float v1942_data_pre = s0[v36_g ? ((v1938_a ^ ((v1938_a >> 4) & 15))) : (0)];
              float v1942_data = v36_g ? (v1942_data_pre) : (0.0f);
              float v1943_data = ir7[8];
              ir7[8] = (v1943_data + v1942_data);
              int32_t v1945_a = v26_lead + 108;
              float v1949_data_pre = s0[v36_g ? ((v1945_a ^ ((v1945_a >> 4) & 15))) : (0)];
              float v1949_data = v36_g ? (v1949_data_pre) : (0.0f);
              float v1950_data = ir7[9];
              ir7[9] = (v1950_data + v1949_data);
              int32_t v1952_a = v26_lead + 120;
              float v1956_data_pre = s0[v36_g ? ((v1952_a ^ ((v1952_a >> 4) & 15))) : (0)];
              float v1956_data = v36_g ? (v1956_data_pre) : (0.0f);
              float v1957_data = ir7[10];
              ir7[10] = (v1957_data + v1956_data);
              int32_t v1959_a = v26_lead + 132;
              float v1963_data_pre = s0[v36_g ? ((v1959_a ^ ((v1959_a >> 4) & 15))) : (0)];
              float v1963_data = v36_g ? (v1963_data_pre) : (0.0f);
              float v1964_data = ir7[11];
              ir7[11] = (v1964_data + v1963_data);
              // r7 = ir7
              if (v36_g) {
                #pragma unroll
                for (int32_t v1966_n1 = 0; v1966_n1 < 12; ++v1966_n1) {
                  float v1968_data = ir7[v1966_n1];
                  r7[v1966_n1] = v1968_data;
                }
              }
              // glb_m4 = store{r>g}(r7);
              if (v36_g) {
                #pragma unroll
                for (int32_t v1969_i1 = 0; v1969_i1 < 12; ++v1969_i1) {
                  float v1971_data = r7[v1969_i1];
                  glb_m4[(v26_lead + (v1969_i1 * 12))] = v1971_data;
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

