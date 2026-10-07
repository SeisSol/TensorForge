// === base name ===
kernel_0ac1e4e51b387eb8

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_0ac1e4e51b387eb8 = {{16, 16, 1}, 16, 12, 1, 16, 10240, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_0ac1e4e51b387eb8(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_0ac1e4e51b387eb8(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_0ac1e4e51b387eb8(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_0ac1e4e51b387eb8(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_0ac1e4e51b387eb8(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_0ac1e4e51b387eb8(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_0ac1e4e51b387eb8(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
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
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v8_batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v8_batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v8_batchId0 * 72 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v8_batchId0 * 72 + 0 + m3_extraOffset];
              float *const __restrict__ glb_m4 = &m4[v8_batchId0 * 144 + 0 + m4_extraOffset];
              float r0[12]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v24_lead = item.get_local_id(2) % 16;
              bool v25_g = v24_lead < 6;
              if (v25_g) {
                #pragma unroll
                for (int32_t v26_i1 = 0; v26_i1 < 12; ++v26_i1) {
                  float v31_data = glb_m0[(v24_lead + (v26_i1 * 6))];
                  r0[v26_i1] = v31_data;
                }
              }
              float r1[12]{};
              // r1 = load{g>r}(glb_m1);
              bool v34_g = v24_lead < 12;
              if (v34_g) {
                #pragma unroll
                for (int32_t v35_i1 = 0; v35_i1 < 12; ++v35_i1) {
                  float v40_data = glb_m1[(v24_lead + (v35_i1 * 12))];
                  r1[v35_i1] = v40_data;
                }
              }
              float r3[12]{};
              // r3 = load{g>r}(glb_m2);
              if (v25_g) {
                #pragma unroll
                for (int32_t v919_i1 = 0; v919_i1 < 12; ++v919_i1) {
                  float v924_data = glb_m2[(v24_lead + (v919_i1 * 6))];
                  r3[v919_i1] = v924_data;
                }
              }
              float r2[12]{};
              // r2 = +(r0 * r1) + None
              // [(0, 6), (0, 12)] [(0, 12)]
              float v43_data = r0[0];
              float v44_data = r1[0];
              float v45_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v47_data = r2[0];
              r2[0] = (v47_data + (v43_data * v45_bc));
              float v50_data = r1[1];
              float v51_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v53_data = r2[1];
              r2[1] = (v53_data + (v43_data * v51_bc));
              float v56_data = r1[2];
              float v57_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v59_data = r2[2];
              r2[2] = (v59_data + (v43_data * v57_bc));
              float v62_data = r1[3];
              float v63_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v65_data = r2[3];
              r2[3] = (v65_data + (v43_data * v63_bc));
              float v68_data = r1[4];
              float v69_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v71_data = r2[4];
              r2[4] = (v71_data + (v43_data * v69_bc));
              float v74_data = r1[5];
              float v75_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v77_data = r2[5];
              r2[5] = (v77_data + (v43_data * v75_bc));
              float v80_data = r1[6];
              float v81_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v83_data = r2[6];
              r2[6] = (v83_data + (v43_data * v81_bc));
              float v86_data = r1[7];
              float v87_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v89_data = r2[7];
              r2[7] = (v89_data + (v43_data * v87_bc));
              float v92_data = r1[8];
              float v93_bc = sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v95_data = r2[8];
              r2[8] = (v95_data + (v43_data * v93_bc));
              float v98_data = r1[9];
              float v99_bc = sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v101_data = r2[9];
              r2[9] = (v101_data + (v43_data * v99_bc));
              float v104_data = r1[10];
              float v105_bc = sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v107_data = r2[10];
              r2[10] = (v107_data + (v43_data * v105_bc));
              float v110_data = r1[11];
              float v111_bc = sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v113_data = r2[11];
              r2[11] = (v113_data + (v43_data * v111_bc));
              float v115_data = r0[1];
              float v117_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v119_data = r2[0];
              r2[0] = (v119_data + (v115_data * v117_bc));
              float v123_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v125_data = r2[1];
              r2[1] = (v125_data + (v115_data * v123_bc));
              float v129_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v131_data = r2[2];
              r2[2] = (v131_data + (v115_data * v129_bc));
              float v135_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v137_data = r2[3];
              r2[3] = (v137_data + (v115_data * v135_bc));
              float v141_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v143_data = r2[4];
              r2[4] = (v143_data + (v115_data * v141_bc));
              float v147_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v149_data = r2[5];
              r2[5] = (v149_data + (v115_data * v147_bc));
              float v153_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v155_data = r2[6];
              r2[6] = (v155_data + (v115_data * v153_bc));
              float v159_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v161_data = r2[7];
              r2[7] = (v161_data + (v115_data * v159_bc));
              float v165_bc = sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v167_data = r2[8];
              r2[8] = (v167_data + (v115_data * v165_bc));
              float v171_bc = sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v173_data = r2[9];
              r2[9] = (v173_data + (v115_data * v171_bc));
              float v177_bc = sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v179_data = r2[10];
              r2[10] = (v179_data + (v115_data * v177_bc));
              float v183_bc = sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v185_data = r2[11];
              r2[11] = (v185_data + (v115_data * v183_bc));
              float v187_data = r0[2];
              float v189_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v191_data = r2[0];
              r2[0] = (v191_data + (v187_data * v189_bc));
              float v195_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v197_data = r2[1];
              r2[1] = (v197_data + (v187_data * v195_bc));
              float v201_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v203_data = r2[2];
              r2[2] = (v203_data + (v187_data * v201_bc));
              float v207_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v209_data = r2[3];
              r2[3] = (v209_data + (v187_data * v207_bc));
              float v213_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v215_data = r2[4];
              r2[4] = (v215_data + (v187_data * v213_bc));
              float v219_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v221_data = r2[5];
              r2[5] = (v221_data + (v187_data * v219_bc));
              float v225_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v227_data = r2[6];
              r2[6] = (v227_data + (v187_data * v225_bc));
              float v231_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v233_data = r2[7];
              r2[7] = (v233_data + (v187_data * v231_bc));
              float v237_bc = sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v239_data = r2[8];
              r2[8] = (v239_data + (v187_data * v237_bc));
              float v243_bc = sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v245_data = r2[9];
              r2[9] = (v245_data + (v187_data * v243_bc));
              float v249_bc = sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v251_data = r2[10];
              r2[10] = (v251_data + (v187_data * v249_bc));
              float v255_bc = sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v257_data = r2[11];
              r2[11] = (v257_data + (v187_data * v255_bc));
              float v259_data = r0[3];
              float v261_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v263_data = r2[0];
              r2[0] = (v263_data + (v259_data * v261_bc));
              float v267_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v269_data = r2[1];
              r2[1] = (v269_data + (v259_data * v267_bc));
              float v273_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v275_data = r2[2];
              r2[2] = (v275_data + (v259_data * v273_bc));
              float v279_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v281_data = r2[3];
              r2[3] = (v281_data + (v259_data * v279_bc));
              float v285_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v287_data = r2[4];
              r2[4] = (v287_data + (v259_data * v285_bc));
              float v291_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v293_data = r2[5];
              r2[5] = (v293_data + (v259_data * v291_bc));
              float v297_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v299_data = r2[6];
              r2[6] = (v299_data + (v259_data * v297_bc));
              float v303_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v305_data = r2[7];
              r2[7] = (v305_data + (v259_data * v303_bc));
              float v309_bc = sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v311_data = r2[8];
              r2[8] = (v311_data + (v259_data * v309_bc));
              float v315_bc = sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v317_data = r2[9];
              r2[9] = (v317_data + (v259_data * v315_bc));
              float v321_bc = sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v323_data = r2[10];
              r2[10] = (v323_data + (v259_data * v321_bc));
              float v327_bc = sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v329_data = r2[11];
              r2[11] = (v329_data + (v259_data * v327_bc));
              float v331_data = r0[4];
              float v333_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v335_data = r2[0];
              r2[0] = (v335_data + (v331_data * v333_bc));
              float v339_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v341_data = r2[1];
              r2[1] = (v341_data + (v331_data * v339_bc));
              float v345_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v347_data = r2[2];
              r2[2] = (v347_data + (v331_data * v345_bc));
              float v351_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v353_data = r2[3];
              r2[3] = (v353_data + (v331_data * v351_bc));
              float v357_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v359_data = r2[4];
              r2[4] = (v359_data + (v331_data * v357_bc));
              float v363_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v365_data = r2[5];
              r2[5] = (v365_data + (v331_data * v363_bc));
              float v369_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v371_data = r2[6];
              r2[6] = (v371_data + (v331_data * v369_bc));
              float v375_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v377_data = r2[7];
              r2[7] = (v377_data + (v331_data * v375_bc));
              float v381_bc = sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v383_data = r2[8];
              r2[8] = (v383_data + (v331_data * v381_bc));
              float v387_bc = sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v389_data = r2[9];
              r2[9] = (v389_data + (v331_data * v387_bc));
              float v393_bc = sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v395_data = r2[10];
              r2[10] = (v395_data + (v331_data * v393_bc));
              float v399_bc = sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v401_data = r2[11];
              r2[11] = (v401_data + (v331_data * v399_bc));
              float v403_data = r0[5];
              float v405_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v407_data = r2[0];
              r2[0] = (v407_data + (v403_data * v405_bc));
              float v411_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v413_data = r2[1];
              r2[1] = (v413_data + (v403_data * v411_bc));
              float v417_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v419_data = r2[2];
              r2[2] = (v419_data + (v403_data * v417_bc));
              float v423_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v425_data = r2[3];
              r2[3] = (v425_data + (v403_data * v423_bc));
              float v429_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v431_data = r2[4];
              r2[4] = (v431_data + (v403_data * v429_bc));
              float v435_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v437_data = r2[5];
              r2[5] = (v437_data + (v403_data * v435_bc));
              float v441_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v443_data = r2[6];
              r2[6] = (v443_data + (v403_data * v441_bc));
              float v447_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v449_data = r2[7];
              r2[7] = (v449_data + (v403_data * v447_bc));
              float v453_bc = sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v455_data = r2[8];
              r2[8] = (v455_data + (v403_data * v453_bc));
              float v459_bc = sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v461_data = r2[9];
              r2[9] = (v461_data + (v403_data * v459_bc));
              float v465_bc = sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v467_data = r2[10];
              r2[10] = (v467_data + (v403_data * v465_bc));
              float v471_bc = sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v473_data = r2[11];
              r2[11] = (v473_data + (v403_data * v471_bc));
              float v475_data = r0[6];
              float v477_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v479_data = r2[0];
              r2[0] = (v479_data + (v475_data * v477_bc));
              float v483_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v485_data = r2[1];
              r2[1] = (v485_data + (v475_data * v483_bc));
              float v489_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v491_data = r2[2];
              r2[2] = (v491_data + (v475_data * v489_bc));
              float v495_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v497_data = r2[3];
              r2[3] = (v497_data + (v475_data * v495_bc));
              float v501_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v503_data = r2[4];
              r2[4] = (v503_data + (v475_data * v501_bc));
              float v507_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v509_data = r2[5];
              r2[5] = (v509_data + (v475_data * v507_bc));
              float v513_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v515_data = r2[6];
              r2[6] = (v515_data + (v475_data * v513_bc));
              float v519_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v521_data = r2[7];
              r2[7] = (v521_data + (v475_data * v519_bc));
              float v525_bc = sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v527_data = r2[8];
              r2[8] = (v527_data + (v475_data * v525_bc));
              float v531_bc = sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v533_data = r2[9];
              r2[9] = (v533_data + (v475_data * v531_bc));
              float v537_bc = sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v539_data = r2[10];
              r2[10] = (v539_data + (v475_data * v537_bc));
              float v543_bc = sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v545_data = r2[11];
              r2[11] = (v545_data + (v475_data * v543_bc));
              float v547_data = r0[7];
              float v549_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v551_data = r2[0];
              r2[0] = (v551_data + (v547_data * v549_bc));
              float v555_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v557_data = r2[1];
              r2[1] = (v557_data + (v547_data * v555_bc));
              float v561_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v563_data = r2[2];
              r2[2] = (v563_data + (v547_data * v561_bc));
              float v567_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v569_data = r2[3];
              r2[3] = (v569_data + (v547_data * v567_bc));
              float v573_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v575_data = r2[4];
              r2[4] = (v575_data + (v547_data * v573_bc));
              float v579_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v581_data = r2[5];
              r2[5] = (v581_data + (v547_data * v579_bc));
              float v585_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v587_data = r2[6];
              r2[6] = (v587_data + (v547_data * v585_bc));
              float v591_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v593_data = r2[7];
              r2[7] = (v593_data + (v547_data * v591_bc));
              float v597_bc = sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v599_data = r2[8];
              r2[8] = (v599_data + (v547_data * v597_bc));
              float v603_bc = sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v605_data = r2[9];
              r2[9] = (v605_data + (v547_data * v603_bc));
              float v609_bc = sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v611_data = r2[10];
              r2[10] = (v611_data + (v547_data * v609_bc));
              float v615_bc = sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v617_data = r2[11];
              r2[11] = (v617_data + (v547_data * v615_bc));
              float v619_data = r0[8];
              float v621_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v623_data = r2[0];
              r2[0] = (v623_data + (v619_data * v621_bc));
              float v627_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v629_data = r2[1];
              r2[1] = (v629_data + (v619_data * v627_bc));
              float v633_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v635_data = r2[2];
              r2[2] = (v635_data + (v619_data * v633_bc));
              float v639_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v641_data = r2[3];
              r2[3] = (v641_data + (v619_data * v639_bc));
              float v645_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v647_data = r2[4];
              r2[4] = (v647_data + (v619_data * v645_bc));
              float v651_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v653_data = r2[5];
              r2[5] = (v653_data + (v619_data * v651_bc));
              float v657_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v659_data = r2[6];
              r2[6] = (v659_data + (v619_data * v657_bc));
              float v663_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v665_data = r2[7];
              r2[7] = (v665_data + (v619_data * v663_bc));
              float v669_bc = sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v671_data = r2[8];
              r2[8] = (v671_data + (v619_data * v669_bc));
              float v675_bc = sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v677_data = r2[9];
              r2[9] = (v677_data + (v619_data * v675_bc));
              float v681_bc = sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v683_data = r2[10];
              r2[10] = (v683_data + (v619_data * v681_bc));
              float v687_bc = sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v689_data = r2[11];
              r2[11] = (v689_data + (v619_data * v687_bc));
              float v691_data = r0[9];
              float v693_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v695_data = r2[0];
              r2[0] = (v695_data + (v691_data * v693_bc));
              float v699_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v701_data = r2[1];
              r2[1] = (v701_data + (v691_data * v699_bc));
              float v705_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v707_data = r2[2];
              r2[2] = (v707_data + (v691_data * v705_bc));
              float v711_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v713_data = r2[3];
              r2[3] = (v713_data + (v691_data * v711_bc));
              float v717_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v719_data = r2[4];
              r2[4] = (v719_data + (v691_data * v717_bc));
              float v723_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v725_data = r2[5];
              r2[5] = (v725_data + (v691_data * v723_bc));
              float v729_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v731_data = r2[6];
              r2[6] = (v731_data + (v691_data * v729_bc));
              float v735_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v737_data = r2[7];
              r2[7] = (v737_data + (v691_data * v735_bc));
              float v741_bc = sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v743_data = r2[8];
              r2[8] = (v743_data + (v691_data * v741_bc));
              float v747_bc = sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v749_data = r2[9];
              r2[9] = (v749_data + (v691_data * v747_bc));
              float v753_bc = sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v755_data = r2[10];
              r2[10] = (v755_data + (v691_data * v753_bc));
              float v759_bc = sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v761_data = r2[11];
              r2[11] = (v761_data + (v691_data * v759_bc));
              float v763_data = r0[10];
              float v765_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v767_data = r2[0];
              r2[0] = (v767_data + (v763_data * v765_bc));
              float v771_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v773_data = r2[1];
              r2[1] = (v773_data + (v763_data * v771_bc));
              float v777_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v779_data = r2[2];
              r2[2] = (v779_data + (v763_data * v777_bc));
              float v783_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v785_data = r2[3];
              r2[3] = (v785_data + (v763_data * v783_bc));
              float v789_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v791_data = r2[4];
              r2[4] = (v791_data + (v763_data * v789_bc));
              float v795_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v797_data = r2[5];
              r2[5] = (v797_data + (v763_data * v795_bc));
              float v801_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v803_data = r2[6];
              r2[6] = (v803_data + (v763_data * v801_bc));
              float v807_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v809_data = r2[7];
              r2[7] = (v809_data + (v763_data * v807_bc));
              float v813_bc = sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v815_data = r2[8];
              r2[8] = (v815_data + (v763_data * v813_bc));
              float v819_bc = sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v821_data = r2[9];
              r2[9] = (v821_data + (v763_data * v819_bc));
              float v825_bc = sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v827_data = r2[10];
              r2[10] = (v827_data + (v763_data * v825_bc));
              float v831_bc = sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v833_data = r2[11];
              r2[11] = (v833_data + (v763_data * v831_bc));
              float v835_data = r0[11];
              float v837_bc = sycl::select_from_group(item.get_sub_group(), v44_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v839_data = r2[0];
              r2[0] = (v839_data + (v835_data * v837_bc));
              float v843_bc = sycl::select_from_group(item.get_sub_group(), v50_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v845_data = r2[1];
              r2[1] = (v845_data + (v835_data * v843_bc));
              float v849_bc = sycl::select_from_group(item.get_sub_group(), v56_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v851_data = r2[2];
              r2[2] = (v851_data + (v835_data * v849_bc));
              float v855_bc = sycl::select_from_group(item.get_sub_group(), v62_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v857_data = r2[3];
              r2[3] = (v857_data + (v835_data * v855_bc));
              float v861_bc = sycl::select_from_group(item.get_sub_group(), v68_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v863_data = r2[4];
              r2[4] = (v863_data + (v835_data * v861_bc));
              float v867_bc = sycl::select_from_group(item.get_sub_group(), v74_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v869_data = r2[5];
              r2[5] = (v869_data + (v835_data * v867_bc));
              float v873_bc = sycl::select_from_group(item.get_sub_group(), v80_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v875_data = r2[6];
              r2[6] = (v875_data + (v835_data * v873_bc));
              float v879_bc = sycl::select_from_group(item.get_sub_group(), v86_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v881_data = r2[7];
              r2[7] = (v881_data + (v835_data * v879_bc));
              float v885_bc = sycl::select_from_group(item.get_sub_group(), v92_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v887_data = r2[8];
              r2[8] = (v887_data + (v835_data * v885_bc));
              float v891_bc = sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v893_data = r2[9];
              r2[9] = (v893_data + (v835_data * v891_bc));
              float v897_bc = sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v899_data = r2[10];
              r2[10] = (v899_data + (v835_data * v897_bc));
              float v903_bc = sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v905_data = r2[11];
              r2[11] = (v905_data + (v835_data * v903_bc));
              // s0 = store{r>s}(localShrMem0, r2);
              if (v25_g) {
                int32_t v912_off = v24_lead + 6;
                #pragma unroll
                for (int32_t v907_i1 = 0; v907_i1 < 12; ++v907_i1) {
                  float v909_data = r2[v907_i1];
                  int32_t v914_a = v912_off + (v907_i1 * 12);
                  s0[(v914_a ^ ((v914_a >> 4) & 15))] = v909_data;
                }
              }
              float r5[12]{};
              // r5 = load{g>r}(glb_m3);
              if (v25_g) {
                #pragma unroll
                for (int32_t v1817_i1 = 0; v1817_i1 < 12; ++v1817_i1) {
                  float v1822_data = glb_m3[(v24_lead + (v1817_i1 * 6))];
                  r5[v1817_i1] = v1822_data;
                }
              }
              float r4[12]{};
              // ir4 = +(r3 * r1)
              // [(0, 6), (0, 12)] [(0, 12)]
              float ir4[12]{};
              float v928_data = r3[0];
              float v932_data = ir4[0];
              ir4[0] = (v932_data + (v928_data * v45_bc));
              float v938_data = ir4[1];
              ir4[1] = (v938_data + (v928_data * v51_bc));
              float v944_data = ir4[2];
              ir4[2] = (v944_data + (v928_data * v57_bc));
              float v950_data = ir4[3];
              ir4[3] = (v950_data + (v928_data * v63_bc));
              float v956_data = ir4[4];
              ir4[4] = (v956_data + (v928_data * v69_bc));
              float v962_data = ir4[5];
              ir4[5] = (v962_data + (v928_data * v75_bc));
              float v968_data = ir4[6];
              ir4[6] = (v968_data + (v928_data * v81_bc));
              float v974_data = ir4[7];
              ir4[7] = (v974_data + (v928_data * v87_bc));
              float v980_data = ir4[8];
              ir4[8] = (v980_data + (v928_data * v93_bc));
              float v986_data = ir4[9];
              ir4[9] = (v986_data + (v928_data * v99_bc));
              float v992_data = ir4[10];
              ir4[10] = (v992_data + (v928_data * v105_bc));
              float v998_data = ir4[11];
              ir4[11] = (v998_data + (v928_data * v111_bc));
              float v1000_data = r3[1];
              float v1004_data = ir4[0];
              ir4[0] = (v1004_data + (v1000_data * v117_bc));
              float v1010_data = ir4[1];
              ir4[1] = (v1010_data + (v1000_data * v123_bc));
              float v1016_data = ir4[2];
              ir4[2] = (v1016_data + (v1000_data * v129_bc));
              float v1022_data = ir4[3];
              ir4[3] = (v1022_data + (v1000_data * v135_bc));
              float v1028_data = ir4[4];
              ir4[4] = (v1028_data + (v1000_data * v141_bc));
              float v1034_data = ir4[5];
              ir4[5] = (v1034_data + (v1000_data * v147_bc));
              float v1040_data = ir4[6];
              ir4[6] = (v1040_data + (v1000_data * v153_bc));
              float v1046_data = ir4[7];
              ir4[7] = (v1046_data + (v1000_data * v159_bc));
              float v1052_data = ir4[8];
              ir4[8] = (v1052_data + (v1000_data * v165_bc));
              float v1058_data = ir4[9];
              ir4[9] = (v1058_data + (v1000_data * v171_bc));
              float v1064_data = ir4[10];
              ir4[10] = (v1064_data + (v1000_data * v177_bc));
              float v1070_data = ir4[11];
              ir4[11] = (v1070_data + (v1000_data * v183_bc));
              float v1072_data = r3[2];
              float v1076_data = ir4[0];
              ir4[0] = (v1076_data + (v1072_data * v189_bc));
              float v1082_data = ir4[1];
              ir4[1] = (v1082_data + (v1072_data * v195_bc));
              float v1088_data = ir4[2];
              ir4[2] = (v1088_data + (v1072_data * v201_bc));
              float v1094_data = ir4[3];
              ir4[3] = (v1094_data + (v1072_data * v207_bc));
              float v1100_data = ir4[4];
              ir4[4] = (v1100_data + (v1072_data * v213_bc));
              float v1106_data = ir4[5];
              ir4[5] = (v1106_data + (v1072_data * v219_bc));
              float v1112_data = ir4[6];
              ir4[6] = (v1112_data + (v1072_data * v225_bc));
              float v1118_data = ir4[7];
              ir4[7] = (v1118_data + (v1072_data * v231_bc));
              float v1124_data = ir4[8];
              ir4[8] = (v1124_data + (v1072_data * v237_bc));
              float v1130_data = ir4[9];
              ir4[9] = (v1130_data + (v1072_data * v243_bc));
              float v1136_data = ir4[10];
              ir4[10] = (v1136_data + (v1072_data * v249_bc));
              float v1142_data = ir4[11];
              ir4[11] = (v1142_data + (v1072_data * v255_bc));
              float v1144_data = r3[3];
              float v1148_data = ir4[0];
              ir4[0] = (v1148_data + (v1144_data * v261_bc));
              float v1154_data = ir4[1];
              ir4[1] = (v1154_data + (v1144_data * v267_bc));
              float v1160_data = ir4[2];
              ir4[2] = (v1160_data + (v1144_data * v273_bc));
              float v1166_data = ir4[3];
              ir4[3] = (v1166_data + (v1144_data * v279_bc));
              float v1172_data = ir4[4];
              ir4[4] = (v1172_data + (v1144_data * v285_bc));
              float v1178_data = ir4[5];
              ir4[5] = (v1178_data + (v1144_data * v291_bc));
              float v1184_data = ir4[6];
              ir4[6] = (v1184_data + (v1144_data * v297_bc));
              float v1190_data = ir4[7];
              ir4[7] = (v1190_data + (v1144_data * v303_bc));
              float v1196_data = ir4[8];
              ir4[8] = (v1196_data + (v1144_data * v309_bc));
              float v1202_data = ir4[9];
              ir4[9] = (v1202_data + (v1144_data * v315_bc));
              float v1208_data = ir4[10];
              ir4[10] = (v1208_data + (v1144_data * v321_bc));
              float v1214_data = ir4[11];
              ir4[11] = (v1214_data + (v1144_data * v327_bc));
              float v1216_data = r3[4];
              float v1220_data = ir4[0];
              ir4[0] = (v1220_data + (v1216_data * v333_bc));
              float v1226_data = ir4[1];
              ir4[1] = (v1226_data + (v1216_data * v339_bc));
              float v1232_data = ir4[2];
              ir4[2] = (v1232_data + (v1216_data * v345_bc));
              float v1238_data = ir4[3];
              ir4[3] = (v1238_data + (v1216_data * v351_bc));
              float v1244_data = ir4[4];
              ir4[4] = (v1244_data + (v1216_data * v357_bc));
              float v1250_data = ir4[5];
              ir4[5] = (v1250_data + (v1216_data * v363_bc));
              float v1256_data = ir4[6];
              ir4[6] = (v1256_data + (v1216_data * v369_bc));
              float v1262_data = ir4[7];
              ir4[7] = (v1262_data + (v1216_data * v375_bc));
              float v1268_data = ir4[8];
              ir4[8] = (v1268_data + (v1216_data * v381_bc));
              float v1274_data = ir4[9];
              ir4[9] = (v1274_data + (v1216_data * v387_bc));
              float v1280_data = ir4[10];
              ir4[10] = (v1280_data + (v1216_data * v393_bc));
              float v1286_data = ir4[11];
              ir4[11] = (v1286_data + (v1216_data * v399_bc));
              float v1288_data = r3[5];
              float v1292_data = ir4[0];
              ir4[0] = (v1292_data + (v1288_data * v405_bc));
              float v1298_data = ir4[1];
              ir4[1] = (v1298_data + (v1288_data * v411_bc));
              float v1304_data = ir4[2];
              ir4[2] = (v1304_data + (v1288_data * v417_bc));
              float v1310_data = ir4[3];
              ir4[3] = (v1310_data + (v1288_data * v423_bc));
              float v1316_data = ir4[4];
              ir4[4] = (v1316_data + (v1288_data * v429_bc));
              float v1322_data = ir4[5];
              ir4[5] = (v1322_data + (v1288_data * v435_bc));
              float v1328_data = ir4[6];
              ir4[6] = (v1328_data + (v1288_data * v441_bc));
              float v1334_data = ir4[7];
              ir4[7] = (v1334_data + (v1288_data * v447_bc));
              float v1340_data = ir4[8];
              ir4[8] = (v1340_data + (v1288_data * v453_bc));
              float v1346_data = ir4[9];
              ir4[9] = (v1346_data + (v1288_data * v459_bc));
              float v1352_data = ir4[10];
              ir4[10] = (v1352_data + (v1288_data * v465_bc));
              float v1358_data = ir4[11];
              ir4[11] = (v1358_data + (v1288_data * v471_bc));
              float v1360_data = r3[6];
              float v1364_data = ir4[0];
              ir4[0] = (v1364_data + (v1360_data * v477_bc));
              float v1370_data = ir4[1];
              ir4[1] = (v1370_data + (v1360_data * v483_bc));
              float v1376_data = ir4[2];
              ir4[2] = (v1376_data + (v1360_data * v489_bc));
              float v1382_data = ir4[3];
              ir4[3] = (v1382_data + (v1360_data * v495_bc));
              float v1388_data = ir4[4];
              ir4[4] = (v1388_data + (v1360_data * v501_bc));
              float v1394_data = ir4[5];
              ir4[5] = (v1394_data + (v1360_data * v507_bc));
              float v1400_data = ir4[6];
              ir4[6] = (v1400_data + (v1360_data * v513_bc));
              float v1406_data = ir4[7];
              ir4[7] = (v1406_data + (v1360_data * v519_bc));
              float v1412_data = ir4[8];
              ir4[8] = (v1412_data + (v1360_data * v525_bc));
              float v1418_data = ir4[9];
              ir4[9] = (v1418_data + (v1360_data * v531_bc));
              float v1424_data = ir4[10];
              ir4[10] = (v1424_data + (v1360_data * v537_bc));
              float v1430_data = ir4[11];
              ir4[11] = (v1430_data + (v1360_data * v543_bc));
              float v1432_data = r3[7];
              float v1436_data = ir4[0];
              ir4[0] = (v1436_data + (v1432_data * v549_bc));
              float v1442_data = ir4[1];
              ir4[1] = (v1442_data + (v1432_data * v555_bc));
              float v1448_data = ir4[2];
              ir4[2] = (v1448_data + (v1432_data * v561_bc));
              float v1454_data = ir4[3];
              ir4[3] = (v1454_data + (v1432_data * v567_bc));
              float v1460_data = ir4[4];
              ir4[4] = (v1460_data + (v1432_data * v573_bc));
              float v1466_data = ir4[5];
              ir4[5] = (v1466_data + (v1432_data * v579_bc));
              float v1472_data = ir4[6];
              ir4[6] = (v1472_data + (v1432_data * v585_bc));
              float v1478_data = ir4[7];
              ir4[7] = (v1478_data + (v1432_data * v591_bc));
              float v1484_data = ir4[8];
              ir4[8] = (v1484_data + (v1432_data * v597_bc));
              float v1490_data = ir4[9];
              ir4[9] = (v1490_data + (v1432_data * v603_bc));
              float v1496_data = ir4[10];
              ir4[10] = (v1496_data + (v1432_data * v609_bc));
              float v1502_data = ir4[11];
              ir4[11] = (v1502_data + (v1432_data * v615_bc));
              float v1504_data = r3[8];
              float v1508_data = ir4[0];
              ir4[0] = (v1508_data + (v1504_data * v621_bc));
              float v1514_data = ir4[1];
              ir4[1] = (v1514_data + (v1504_data * v627_bc));
              float v1520_data = ir4[2];
              ir4[2] = (v1520_data + (v1504_data * v633_bc));
              float v1526_data = ir4[3];
              ir4[3] = (v1526_data + (v1504_data * v639_bc));
              float v1532_data = ir4[4];
              ir4[4] = (v1532_data + (v1504_data * v645_bc));
              float v1538_data = ir4[5];
              ir4[5] = (v1538_data + (v1504_data * v651_bc));
              float v1544_data = ir4[6];
              ir4[6] = (v1544_data + (v1504_data * v657_bc));
              float v1550_data = ir4[7];
              ir4[7] = (v1550_data + (v1504_data * v663_bc));
              float v1556_data = ir4[8];
              ir4[8] = (v1556_data + (v1504_data * v669_bc));
              float v1562_data = ir4[9];
              ir4[9] = (v1562_data + (v1504_data * v675_bc));
              float v1568_data = ir4[10];
              ir4[10] = (v1568_data + (v1504_data * v681_bc));
              float v1574_data = ir4[11];
              ir4[11] = (v1574_data + (v1504_data * v687_bc));
              float v1576_data = r3[9];
              float v1580_data = ir4[0];
              ir4[0] = (v1580_data + (v1576_data * v693_bc));
              float v1586_data = ir4[1];
              ir4[1] = (v1586_data + (v1576_data * v699_bc));
              float v1592_data = ir4[2];
              ir4[2] = (v1592_data + (v1576_data * v705_bc));
              float v1598_data = ir4[3];
              ir4[3] = (v1598_data + (v1576_data * v711_bc));
              float v1604_data = ir4[4];
              ir4[4] = (v1604_data + (v1576_data * v717_bc));
              float v1610_data = ir4[5];
              ir4[5] = (v1610_data + (v1576_data * v723_bc));
              float v1616_data = ir4[6];
              ir4[6] = (v1616_data + (v1576_data * v729_bc));
              float v1622_data = ir4[7];
              ir4[7] = (v1622_data + (v1576_data * v735_bc));
              float v1628_data = ir4[8];
              ir4[8] = (v1628_data + (v1576_data * v741_bc));
              float v1634_data = ir4[9];
              ir4[9] = (v1634_data + (v1576_data * v747_bc));
              float v1640_data = ir4[10];
              ir4[10] = (v1640_data + (v1576_data * v753_bc));
              float v1646_data = ir4[11];
              ir4[11] = (v1646_data + (v1576_data * v759_bc));
              float v1648_data = r3[10];
              float v1652_data = ir4[0];
              ir4[0] = (v1652_data + (v1648_data * v765_bc));
              float v1658_data = ir4[1];
              ir4[1] = (v1658_data + (v1648_data * v771_bc));
              float v1664_data = ir4[2];
              ir4[2] = (v1664_data + (v1648_data * v777_bc));
              float v1670_data = ir4[3];
              ir4[3] = (v1670_data + (v1648_data * v783_bc));
              float v1676_data = ir4[4];
              ir4[4] = (v1676_data + (v1648_data * v789_bc));
              float v1682_data = ir4[5];
              ir4[5] = (v1682_data + (v1648_data * v795_bc));
              float v1688_data = ir4[6];
              ir4[6] = (v1688_data + (v1648_data * v801_bc));
              float v1694_data = ir4[7];
              ir4[7] = (v1694_data + (v1648_data * v807_bc));
              float v1700_data = ir4[8];
              ir4[8] = (v1700_data + (v1648_data * v813_bc));
              float v1706_data = ir4[9];
              ir4[9] = (v1706_data + (v1648_data * v819_bc));
              float v1712_data = ir4[10];
              ir4[10] = (v1712_data + (v1648_data * v825_bc));
              float v1718_data = ir4[11];
              ir4[11] = (v1718_data + (v1648_data * v831_bc));
              float v1720_data = r3[11];
              float v1724_data = ir4[0];
              ir4[0] = (v1724_data + (v1720_data * v837_bc));
              float v1730_data = ir4[1];
              ir4[1] = (v1730_data + (v1720_data * v843_bc));
              float v1736_data = ir4[2];
              ir4[2] = (v1736_data + (v1720_data * v849_bc));
              float v1742_data = ir4[3];
              ir4[3] = (v1742_data + (v1720_data * v855_bc));
              float v1748_data = ir4[4];
              ir4[4] = (v1748_data + (v1720_data * v861_bc));
              float v1754_data = ir4[5];
              ir4[5] = (v1754_data + (v1720_data * v867_bc));
              float v1760_data = ir4[6];
              ir4[6] = (v1760_data + (v1720_data * v873_bc));
              float v1766_data = ir4[7];
              ir4[7] = (v1766_data + (v1720_data * v879_bc));
              float v1772_data = ir4[8];
              ir4[8] = (v1772_data + (v1720_data * v885_bc));
              float v1778_data = ir4[9];
              ir4[9] = (v1778_data + (v1720_data * v891_bc));
              float v1784_data = ir4[10];
              ir4[10] = (v1784_data + (v1720_data * v897_bc));
              float v1790_data = ir4[11];
              ir4[11] = (v1790_data + (v1720_data * v903_bc));
              // r4 = ir4
              if (v25_g) {
                #pragma unroll
                for (int32_t v1792_n1 = 0; v1792_n1 < 12; ++v1792_n1) {
                  float v1794_data = ir4[v1792_n1];
                  r4[v1792_n1] = v1794_data;
                }
              }
              // s0 = store{r>s, clear}(localShrMem0, r4);
              sycl::group_barrier(item.get_sub_group());
              if ((v24_lead >= 6) && v34_g) {
                #pragma unroll
                for (int32_t v1797_z1 = 0; v1797_z1 < 12; ++v1797_z1) {
                  int32_t v1802_a = v24_lead + (v1797_z1 * 12);
                  s0[(v1802_a ^ ((v1802_a >> 4) & 15))] = 0.0f;
                }
              }
              if (v25_g) {
                #pragma unroll
                for (int32_t v1806_i1 = 0; v1806_i1 < 12; ++v1806_i1) {
                  float v1808_data = r4[v1806_i1];
                  int32_t v1812_a = v24_lead + (v1806_i1 * 12);
                  s0[(v1812_a ^ ((v1812_a >> 4) & 15))] = v1808_data;
                }
              }
              float r6[12]{};
              // ir6 = +(r5)
              // [(0, 6), (0, 12)] []
              float ir6[12]{};
              float v1826_data = r5[0];
              float v1827_data = ir6[0];
              ir6[0] = (v1827_data + v1826_data);
              float v1829_data = r5[1];
              float v1830_data = ir6[1];
              ir6[1] = (v1830_data + v1829_data);
              float v1832_data = r5[2];
              float v1833_data = ir6[2];
              ir6[2] = (v1833_data + v1832_data);
              float v1835_data = r5[3];
              float v1836_data = ir6[3];
              ir6[3] = (v1836_data + v1835_data);
              float v1838_data = r5[4];
              float v1839_data = ir6[4];
              ir6[4] = (v1839_data + v1838_data);
              float v1841_data = r5[5];
              float v1842_data = ir6[5];
              ir6[5] = (v1842_data + v1841_data);
              float v1844_data = r5[6];
              float v1845_data = ir6[6];
              ir6[6] = (v1845_data + v1844_data);
              float v1847_data = r5[7];
              float v1848_data = ir6[7];
              ir6[7] = (v1848_data + v1847_data);
              float v1850_data = r5[8];
              float v1851_data = ir6[8];
              ir6[8] = (v1851_data + v1850_data);
              float v1853_data = r5[9];
              float v1854_data = ir6[9];
              ir6[9] = (v1854_data + v1853_data);
              float v1856_data = r5[10];
              float v1857_data = ir6[10];
              ir6[10] = (v1857_data + v1856_data);
              float v1859_data = r5[11];
              float v1860_data = ir6[11];
              ir6[11] = (v1860_data + v1859_data);
              // r6 = ir6
              if (v25_g) {
                #pragma unroll
                for (int32_t v1862_n1 = 0; v1862_n1 < 12; ++v1862_n1) {
                  float v1864_data = ir6[v1862_n1];
                  r6[v1862_n1] = v1864_data;
                }
              }
              // s0 = store{r>s}(localShrMem0, r6);
              sycl::group_barrier(item.get_sub_group());
              if (v25_g) {
                int32_t v1870_off = v24_lead + 6;
                #pragma unroll
                for (int32_t v1865_i1 = 0; v1865_i1 < 12; ++v1865_i1) {
                  float v1867_data = r6[v1865_i1];
                  int32_t v1872_a = v1870_off + (v1865_i1 * 12);
                  s0[(v1872_a ^ ((v1872_a >> 4) & 15))] = v1867_data;
                }
              }
              float r7[12]{};
              // ir7 = +(s0)
              // [(0, 12), (0, 12)] []
              float ir7[12]{};
              int32_t v1883_sw = v24_lead ^ ((v24_lead >> 4) & 15);
              sycl::group_barrier(item.get_sub_group());
              float v1884_data_pre = s0[v34_g ? (v1883_sw) : (0)];
              float v1884_data = v34_g ? (v1884_data_pre) : (0.0f);
              float v1885_data = ir7[0];
              ir7[0] = (v1885_data + v1884_data);
              int32_t v1887_a = v24_lead + 12;
              float v1891_data_pre = s0[v34_g ? ((v1887_a ^ ((v1887_a >> 4) & 15))) : (0)];
              float v1891_data = v34_g ? (v1891_data_pre) : (0.0f);
              float v1892_data = ir7[1];
              ir7[1] = (v1892_data + v1891_data);
              int32_t v1894_a = v24_lead + 24;
              float v1898_data_pre = s0[v34_g ? ((v1894_a ^ ((v1894_a >> 4) & 15))) : (0)];
              float v1898_data = v34_g ? (v1898_data_pre) : (0.0f);
              float v1899_data = ir7[2];
              ir7[2] = (v1899_data + v1898_data);
              int32_t v1901_a = v24_lead + 36;
              float v1905_data_pre = s0[v34_g ? ((v1901_a ^ ((v1901_a >> 4) & 15))) : (0)];
              float v1905_data = v34_g ? (v1905_data_pre) : (0.0f);
              float v1906_data = ir7[3];
              ir7[3] = (v1906_data + v1905_data);
              int32_t v1908_a = v24_lead + 48;
              float v1912_data_pre = s0[v34_g ? ((v1908_a ^ ((v1908_a >> 4) & 15))) : (0)];
              float v1912_data = v34_g ? (v1912_data_pre) : (0.0f);
              float v1913_data = ir7[4];
              ir7[4] = (v1913_data + v1912_data);
              int32_t v1915_a = v24_lead + 60;
              float v1919_data_pre = s0[v34_g ? ((v1915_a ^ ((v1915_a >> 4) & 15))) : (0)];
              float v1919_data = v34_g ? (v1919_data_pre) : (0.0f);
              float v1920_data = ir7[5];
              ir7[5] = (v1920_data + v1919_data);
              int32_t v1922_a = v24_lead + 72;
              float v1926_data_pre = s0[v34_g ? ((v1922_a ^ ((v1922_a >> 4) & 15))) : (0)];
              float v1926_data = v34_g ? (v1926_data_pre) : (0.0f);
              float v1927_data = ir7[6];
              ir7[6] = (v1927_data + v1926_data);
              int32_t v1929_a = v24_lead + 84;
              float v1933_data_pre = s0[v34_g ? ((v1929_a ^ ((v1929_a >> 4) & 15))) : (0)];
              float v1933_data = v34_g ? (v1933_data_pre) : (0.0f);
              float v1934_data = ir7[7];
              ir7[7] = (v1934_data + v1933_data);
              int32_t v1936_a = v24_lead + 96;
              float v1940_data_pre = s0[v34_g ? ((v1936_a ^ ((v1936_a >> 4) & 15))) : (0)];
              float v1940_data = v34_g ? (v1940_data_pre) : (0.0f);
              float v1941_data = ir7[8];
              ir7[8] = (v1941_data + v1940_data);
              int32_t v1943_a = v24_lead + 108;
              float v1947_data_pre = s0[v34_g ? ((v1943_a ^ ((v1943_a >> 4) & 15))) : (0)];
              float v1947_data = v34_g ? (v1947_data_pre) : (0.0f);
              float v1948_data = ir7[9];
              ir7[9] = (v1948_data + v1947_data);
              int32_t v1950_a = v24_lead + 120;
              float v1954_data_pre = s0[v34_g ? ((v1950_a ^ ((v1950_a >> 4) & 15))) : (0)];
              float v1954_data = v34_g ? (v1954_data_pre) : (0.0f);
              float v1955_data = ir7[10];
              ir7[10] = (v1955_data + v1954_data);
              int32_t v1957_a = v24_lead + 132;
              float v1961_data_pre = s0[v34_g ? ((v1957_a ^ ((v1957_a >> 4) & 15))) : (0)];
              float v1961_data = v34_g ? (v1961_data_pre) : (0.0f);
              float v1962_data = ir7[11];
              ir7[11] = (v1962_data + v1961_data);
              // r7 = ir7
              if (v34_g) {
                #pragma unroll
                for (int32_t v1964_n1 = 0; v1964_n1 < 12; ++v1964_n1) {
                  float v1966_data = ir7[v1964_n1];
                  r7[v1964_n1] = v1966_data;
                }
              }
              // glb_m4 = store{r>g}(r7);
              if (v34_g) {
                #pragma unroll
                for (int32_t v1967_i1 = 0; v1967_i1 < 12; ++v1967_i1) {
                  float v1969_data = r7[v1967_i1];
                  glb_m4[(v24_lead + (v1967_i1 * 12))] = v1969_data;
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

