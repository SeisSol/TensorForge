// === base name ===
kernel_dbad5634ea458386

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_dbad5634ea458386 = {{16, 16, 1}, 16, 12, 1, 16, 10240, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_dbad5634ea458386(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_dbad5634ea458386(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_dbad5634ea458386(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_dbad5634ea458386(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_dbad5634ea458386(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_dbad5634ea458386(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_dbad5634ea458386(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, float * m5, size_t m5_extraOffset, size_t numElements0, unsigned * flags0) {
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
        //   m3 12×12(12×12) {0..12}×{0..12} strided
        //   m4 2×12(2×12) {0..2}×{0..12} strided
        //   m5 12×12(12×12) {0..12}×{0..12} strided
        // operations:
        //   t0[i,j]@{0..6}×{0..12} = m0[i,k] × m1[k,j]
        //   t0[i,j]@{6..12}×{0..12} = m2[i,k] × m1[k,j]
        //   m3[i,j] = t0[i,j]
        //   t0[i,j]@{6..12}×{0..12} = m4[i,k] × m1[k,j]
        //   m5[i,j] = t0[i,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":2560}],"shared_bytes":10240,"shared_elements":2560,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B1","bbox":[[0,0],[6,12]],"name":"m0","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[6,12]],"name":"m2","ordered":false,"parts":1,"shape":[6,12],"variant":false},{"addressing":"strided","alias":"X","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"N2","bbox":[[0,0],[2,12]],"name":"m4","ordered":false,"parts":1,"shape":[2,12],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,12]],"name":"m5","ordered":false,"parts":1,"shape":[12,12],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[6,12]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[6,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[6,12]],"is_tmp":true,"name":"t0","offset":[6,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[2,12]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[2,12]},{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[12,12]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[12,12]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,12]}],"permute":[[0,1]],"target":[[0,1]]}],"version":"0.0.1"}
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
              float *const __restrict__ glb_m3 = &m3[v8_batchId0 * 144 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v8_batchId0 * 24 + 0 + m4_extraOffset];
              float *const __restrict__ glb_m5 = &m5[v8_batchId0 * 144 + 0 + m5_extraOffset];
              float r0[12]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v25_lead = item.get_local_id(2) % 16;
              bool v26_g = v25_lead < 6;
              if (v26_g) {
                #pragma unroll
                for (int32_t v27_i1 = 0; v27_i1 < 12; ++v27_i1) {
                  float v32_data = glb_m0[(v25_lead + (v27_i1 * 6))];
                  r0[v27_i1] = v32_data;
                }
              }
              float r1[12]{};
              // r1 = load{g>r}(glb_m1);
              bool v35_g = v25_lead < 12;
              if (v35_g) {
                #pragma unroll
                for (int32_t v36_i1 = 0; v36_i1 < 12; ++v36_i1) {
                  float v41_data = glb_m1[(v25_lead + (v36_i1 * 12))];
                  r1[v36_i1] = v41_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r3[12]{};
              // r3 = load{g>r}(glb_m2);
              if (v26_g) {
                #pragma unroll
                for (int32_t v44_i1 = 0; v44_i1 < 12; ++v44_i1) {
                  float v49_data = glb_m2[(v25_lead + (v44_i1 * 6))];
                  r3[v44_i1] = v49_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m1););
              float r2[12]{};
              // r2 = +(r0 * r1) + None
              // [(0, 6), (0, 12)] [(0, 12)]
              float v52_data = r0[0];
              float v53_data = r1[0];
              float v54_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v56_data = r2[0];
              r2[0] = (v56_data + (v52_data * v54_bc));
              float v59_data = r1[1];
              float v60_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v62_data = r2[1];
              r2[1] = (v62_data + (v52_data * v60_bc));
              float v65_data = r1[2];
              float v66_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v68_data = r2[2];
              r2[2] = (v68_data + (v52_data * v66_bc));
              float v71_data = r1[3];
              float v72_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v74_data = r2[3];
              r2[3] = (v74_data + (v52_data * v72_bc));
              float v77_data = r1[4];
              float v78_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v80_data = r2[4];
              r2[4] = (v80_data + (v52_data * v78_bc));
              float v83_data = r1[5];
              float v84_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v86_data = r2[5];
              r2[5] = (v86_data + (v52_data * v84_bc));
              float v89_data = r1[6];
              float v90_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v92_data = r2[6];
              r2[6] = (v92_data + (v52_data * v90_bc));
              float v95_data = r1[7];
              float v96_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v98_data = r2[7];
              r2[7] = (v98_data + (v52_data * v96_bc));
              float v101_data = r1[8];
              float v102_bc = sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v104_data = r2[8];
              r2[8] = (v104_data + (v52_data * v102_bc));
              float v107_data = r1[9];
              float v108_bc = sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v110_data = r2[9];
              r2[9] = (v110_data + (v52_data * v108_bc));
              float v113_data = r1[10];
              float v114_bc = sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v116_data = r2[10];
              r2[10] = (v116_data + (v52_data * v114_bc));
              float v119_data = r1[11];
              float v120_bc = sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v122_data = r2[11];
              r2[11] = (v122_data + (v52_data * v120_bc));
              float v124_data = r0[1];
              float v126_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v128_data = r2[0];
              r2[0] = (v128_data + (v124_data * v126_bc));
              float v132_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v134_data = r2[1];
              r2[1] = (v134_data + (v124_data * v132_bc));
              float v138_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v140_data = r2[2];
              r2[2] = (v140_data + (v124_data * v138_bc));
              float v144_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v146_data = r2[3];
              r2[3] = (v146_data + (v124_data * v144_bc));
              float v150_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v152_data = r2[4];
              r2[4] = (v152_data + (v124_data * v150_bc));
              float v156_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v158_data = r2[5];
              r2[5] = (v158_data + (v124_data * v156_bc));
              float v162_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v164_data = r2[6];
              r2[6] = (v164_data + (v124_data * v162_bc));
              float v168_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v170_data = r2[7];
              r2[7] = (v170_data + (v124_data * v168_bc));
              float v174_bc = sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v176_data = r2[8];
              r2[8] = (v176_data + (v124_data * v174_bc));
              float v180_bc = sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v182_data = r2[9];
              r2[9] = (v182_data + (v124_data * v180_bc));
              float v186_bc = sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v188_data = r2[10];
              r2[10] = (v188_data + (v124_data * v186_bc));
              float v192_bc = sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v194_data = r2[11];
              r2[11] = (v194_data + (v124_data * v192_bc));
              float v196_data = r0[2];
              float v198_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v200_data = r2[0];
              r2[0] = (v200_data + (v196_data * v198_bc));
              float v204_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v206_data = r2[1];
              r2[1] = (v206_data + (v196_data * v204_bc));
              float v210_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v212_data = r2[2];
              r2[2] = (v212_data + (v196_data * v210_bc));
              float v216_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v218_data = r2[3];
              r2[3] = (v218_data + (v196_data * v216_bc));
              float v222_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v224_data = r2[4];
              r2[4] = (v224_data + (v196_data * v222_bc));
              float v228_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v230_data = r2[5];
              r2[5] = (v230_data + (v196_data * v228_bc));
              float v234_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v236_data = r2[6];
              r2[6] = (v236_data + (v196_data * v234_bc));
              float v240_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v242_data = r2[7];
              r2[7] = (v242_data + (v196_data * v240_bc));
              float v246_bc = sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v248_data = r2[8];
              r2[8] = (v248_data + (v196_data * v246_bc));
              float v252_bc = sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v254_data = r2[9];
              r2[9] = (v254_data + (v196_data * v252_bc));
              float v258_bc = sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v260_data = r2[10];
              r2[10] = (v260_data + (v196_data * v258_bc));
              float v264_bc = sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v266_data = r2[11];
              r2[11] = (v266_data + (v196_data * v264_bc));
              float v268_data = r0[3];
              float v270_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v272_data = r2[0];
              r2[0] = (v272_data + (v268_data * v270_bc));
              float v276_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v278_data = r2[1];
              r2[1] = (v278_data + (v268_data * v276_bc));
              float v282_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v284_data = r2[2];
              r2[2] = (v284_data + (v268_data * v282_bc));
              float v288_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v290_data = r2[3];
              r2[3] = (v290_data + (v268_data * v288_bc));
              float v294_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v296_data = r2[4];
              r2[4] = (v296_data + (v268_data * v294_bc));
              float v300_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v302_data = r2[5];
              r2[5] = (v302_data + (v268_data * v300_bc));
              float v306_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v308_data = r2[6];
              r2[6] = (v308_data + (v268_data * v306_bc));
              float v312_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v314_data = r2[7];
              r2[7] = (v314_data + (v268_data * v312_bc));
              float v318_bc = sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v320_data = r2[8];
              r2[8] = (v320_data + (v268_data * v318_bc));
              float v324_bc = sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v326_data = r2[9];
              r2[9] = (v326_data + (v268_data * v324_bc));
              float v330_bc = sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v332_data = r2[10];
              r2[10] = (v332_data + (v268_data * v330_bc));
              float v336_bc = sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v338_data = r2[11];
              r2[11] = (v338_data + (v268_data * v336_bc));
              float v340_data = r0[4];
              float v342_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v344_data = r2[0];
              r2[0] = (v344_data + (v340_data * v342_bc));
              float v348_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v350_data = r2[1];
              r2[1] = (v350_data + (v340_data * v348_bc));
              float v354_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v356_data = r2[2];
              r2[2] = (v356_data + (v340_data * v354_bc));
              float v360_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v362_data = r2[3];
              r2[3] = (v362_data + (v340_data * v360_bc));
              float v366_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v368_data = r2[4];
              r2[4] = (v368_data + (v340_data * v366_bc));
              float v372_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v374_data = r2[5];
              r2[5] = (v374_data + (v340_data * v372_bc));
              float v378_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v380_data = r2[6];
              r2[6] = (v380_data + (v340_data * v378_bc));
              float v384_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v386_data = r2[7];
              r2[7] = (v386_data + (v340_data * v384_bc));
              float v390_bc = sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v392_data = r2[8];
              r2[8] = (v392_data + (v340_data * v390_bc));
              float v396_bc = sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v398_data = r2[9];
              r2[9] = (v398_data + (v340_data * v396_bc));
              float v402_bc = sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v404_data = r2[10];
              r2[10] = (v404_data + (v340_data * v402_bc));
              float v408_bc = sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v410_data = r2[11];
              r2[11] = (v410_data + (v340_data * v408_bc));
              float v412_data = r0[5];
              float v414_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v416_data = r2[0];
              r2[0] = (v416_data + (v412_data * v414_bc));
              float v420_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v422_data = r2[1];
              r2[1] = (v422_data + (v412_data * v420_bc));
              float v426_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v428_data = r2[2];
              r2[2] = (v428_data + (v412_data * v426_bc));
              float v432_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v434_data = r2[3];
              r2[3] = (v434_data + (v412_data * v432_bc));
              float v438_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v440_data = r2[4];
              r2[4] = (v440_data + (v412_data * v438_bc));
              float v444_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v446_data = r2[5];
              r2[5] = (v446_data + (v412_data * v444_bc));
              float v450_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v452_data = r2[6];
              r2[6] = (v452_data + (v412_data * v450_bc));
              float v456_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v458_data = r2[7];
              r2[7] = (v458_data + (v412_data * v456_bc));
              float v462_bc = sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v464_data = r2[8];
              r2[8] = (v464_data + (v412_data * v462_bc));
              float v468_bc = sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v470_data = r2[9];
              r2[9] = (v470_data + (v412_data * v468_bc));
              float v474_bc = sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v476_data = r2[10];
              r2[10] = (v476_data + (v412_data * v474_bc));
              float v480_bc = sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v482_data = r2[11];
              r2[11] = (v482_data + (v412_data * v480_bc));
              float v484_data = r0[6];
              float v486_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v488_data = r2[0];
              r2[0] = (v488_data + (v484_data * v486_bc));
              float v492_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v494_data = r2[1];
              r2[1] = (v494_data + (v484_data * v492_bc));
              float v498_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v500_data = r2[2];
              r2[2] = (v500_data + (v484_data * v498_bc));
              float v504_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v506_data = r2[3];
              r2[3] = (v506_data + (v484_data * v504_bc));
              float v510_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v512_data = r2[4];
              r2[4] = (v512_data + (v484_data * v510_bc));
              float v516_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v518_data = r2[5];
              r2[5] = (v518_data + (v484_data * v516_bc));
              float v522_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v524_data = r2[6];
              r2[6] = (v524_data + (v484_data * v522_bc));
              float v528_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v530_data = r2[7];
              r2[7] = (v530_data + (v484_data * v528_bc));
              float v534_bc = sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v536_data = r2[8];
              r2[8] = (v536_data + (v484_data * v534_bc));
              float v540_bc = sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v542_data = r2[9];
              r2[9] = (v542_data + (v484_data * v540_bc));
              float v546_bc = sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v548_data = r2[10];
              r2[10] = (v548_data + (v484_data * v546_bc));
              float v552_bc = sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v554_data = r2[11];
              r2[11] = (v554_data + (v484_data * v552_bc));
              float v556_data = r0[7];
              float v558_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v560_data = r2[0];
              r2[0] = (v560_data + (v556_data * v558_bc));
              float v564_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v566_data = r2[1];
              r2[1] = (v566_data + (v556_data * v564_bc));
              float v570_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v572_data = r2[2];
              r2[2] = (v572_data + (v556_data * v570_bc));
              float v576_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v578_data = r2[3];
              r2[3] = (v578_data + (v556_data * v576_bc));
              float v582_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v584_data = r2[4];
              r2[4] = (v584_data + (v556_data * v582_bc));
              float v588_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v590_data = r2[5];
              r2[5] = (v590_data + (v556_data * v588_bc));
              float v594_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v596_data = r2[6];
              r2[6] = (v596_data + (v556_data * v594_bc));
              float v600_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v602_data = r2[7];
              r2[7] = (v602_data + (v556_data * v600_bc));
              float v606_bc = sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v608_data = r2[8];
              r2[8] = (v608_data + (v556_data * v606_bc));
              float v612_bc = sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v614_data = r2[9];
              r2[9] = (v614_data + (v556_data * v612_bc));
              float v618_bc = sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v620_data = r2[10];
              r2[10] = (v620_data + (v556_data * v618_bc));
              float v624_bc = sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v626_data = r2[11];
              r2[11] = (v626_data + (v556_data * v624_bc));
              float v628_data = r0[8];
              float v630_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v632_data = r2[0];
              r2[0] = (v632_data + (v628_data * v630_bc));
              float v636_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v638_data = r2[1];
              r2[1] = (v638_data + (v628_data * v636_bc));
              float v642_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v644_data = r2[2];
              r2[2] = (v644_data + (v628_data * v642_bc));
              float v648_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v650_data = r2[3];
              r2[3] = (v650_data + (v628_data * v648_bc));
              float v654_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v656_data = r2[4];
              r2[4] = (v656_data + (v628_data * v654_bc));
              float v660_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v662_data = r2[5];
              r2[5] = (v662_data + (v628_data * v660_bc));
              float v666_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v668_data = r2[6];
              r2[6] = (v668_data + (v628_data * v666_bc));
              float v672_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v674_data = r2[7];
              r2[7] = (v674_data + (v628_data * v672_bc));
              float v678_bc = sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v680_data = r2[8];
              r2[8] = (v680_data + (v628_data * v678_bc));
              float v684_bc = sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v686_data = r2[9];
              r2[9] = (v686_data + (v628_data * v684_bc));
              float v690_bc = sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v692_data = r2[10];
              r2[10] = (v692_data + (v628_data * v690_bc));
              float v696_bc = sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v698_data = r2[11];
              r2[11] = (v698_data + (v628_data * v696_bc));
              float v700_data = r0[9];
              float v702_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v704_data = r2[0];
              r2[0] = (v704_data + (v700_data * v702_bc));
              float v708_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v710_data = r2[1];
              r2[1] = (v710_data + (v700_data * v708_bc));
              float v714_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v716_data = r2[2];
              r2[2] = (v716_data + (v700_data * v714_bc));
              float v720_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v722_data = r2[3];
              r2[3] = (v722_data + (v700_data * v720_bc));
              float v726_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v728_data = r2[4];
              r2[4] = (v728_data + (v700_data * v726_bc));
              float v732_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v734_data = r2[5];
              r2[5] = (v734_data + (v700_data * v732_bc));
              float v738_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v740_data = r2[6];
              r2[6] = (v740_data + (v700_data * v738_bc));
              float v744_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v746_data = r2[7];
              r2[7] = (v746_data + (v700_data * v744_bc));
              float v750_bc = sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v752_data = r2[8];
              r2[8] = (v752_data + (v700_data * v750_bc));
              float v756_bc = sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v758_data = r2[9];
              r2[9] = (v758_data + (v700_data * v756_bc));
              float v762_bc = sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v764_data = r2[10];
              r2[10] = (v764_data + (v700_data * v762_bc));
              float v768_bc = sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v770_data = r2[11];
              r2[11] = (v770_data + (v700_data * v768_bc));
              float v772_data = r0[10];
              float v774_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v776_data = r2[0];
              r2[0] = (v776_data + (v772_data * v774_bc));
              float v780_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v782_data = r2[1];
              r2[1] = (v782_data + (v772_data * v780_bc));
              float v786_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v788_data = r2[2];
              r2[2] = (v788_data + (v772_data * v786_bc));
              float v792_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v794_data = r2[3];
              r2[3] = (v794_data + (v772_data * v792_bc));
              float v798_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v800_data = r2[4];
              r2[4] = (v800_data + (v772_data * v798_bc));
              float v804_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v806_data = r2[5];
              r2[5] = (v806_data + (v772_data * v804_bc));
              float v810_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v812_data = r2[6];
              r2[6] = (v812_data + (v772_data * v810_bc));
              float v816_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v818_data = r2[7];
              r2[7] = (v818_data + (v772_data * v816_bc));
              float v822_bc = sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v824_data = r2[8];
              r2[8] = (v824_data + (v772_data * v822_bc));
              float v828_bc = sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v830_data = r2[9];
              r2[9] = (v830_data + (v772_data * v828_bc));
              float v834_bc = sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v836_data = r2[10];
              r2[10] = (v836_data + (v772_data * v834_bc));
              float v840_bc = sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v842_data = r2[11];
              r2[11] = (v842_data + (v772_data * v840_bc));
              float v844_data = r0[11];
              float v846_bc = sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v848_data = r2[0];
              r2[0] = (v848_data + (v844_data * v846_bc));
              float v852_bc = sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v854_data = r2[1];
              r2[1] = (v854_data + (v844_data * v852_bc));
              float v858_bc = sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v860_data = r2[2];
              r2[2] = (v860_data + (v844_data * v858_bc));
              float v864_bc = sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v866_data = r2[3];
              r2[3] = (v866_data + (v844_data * v864_bc));
              float v870_bc = sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v872_data = r2[4];
              r2[4] = (v872_data + (v844_data * v870_bc));
              float v876_bc = sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v878_data = r2[5];
              r2[5] = (v878_data + (v844_data * v876_bc));
              float v882_bc = sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v884_data = r2[6];
              r2[6] = (v884_data + (v844_data * v882_bc));
              float v888_bc = sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v890_data = r2[7];
              r2[7] = (v890_data + (v844_data * v888_bc));
              float v894_bc = sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v896_data = r2[8];
              r2[8] = (v896_data + (v844_data * v894_bc));
              float v900_bc = sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v902_data = r2[9];
              r2[9] = (v902_data + (v844_data * v900_bc));
              float v906_bc = sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v908_data = r2[10];
              r2[10] = (v908_data + (v844_data * v906_bc));
              float v912_bc = sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v914_data = r2[11];
              r2[11] = (v914_data + (v844_data * v912_bc));
              // s0 = store{r>s}(localShrMem0, r2);
              if (v26_g) {
                #pragma unroll
                for (int32_t v916_i1 = 0; v916_i1 < 12; ++v916_i1) {
                  float v918_data = r2[v916_i1];
                  int32_t v922_a = v25_lead + (v916_i1 * 12);
                  s0[(v922_a ^ ((v922_a >> 4) & 15))] = v918_data;
                }
              }
              float r6[12]{};
              // r6 = load{g>r}(glb_m4);
              bool v927_g = v25_lead < 2;
              if (v927_g) {
                #pragma unroll
                for (int32_t v928_i1 = 0; v928_i1 < 12; ++v928_i1) {
                  float v933_data = glb_m4[(v25_lead + (v928_i1 * 2))];
                  r6[v928_i1] = v933_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m2););
              float r4[12]{};
              // ir4 = +(r3 * r1)
              // [(0, 6), (0, 12)] [(0, 12)]
              float ir4[12]{};
              float v937_data = r3[0];
              float v941_data = ir4[0];
              ir4[0] = (v941_data + (v937_data * v54_bc));
              float v947_data = ir4[1];
              ir4[1] = (v947_data + (v937_data * v60_bc));
              float v953_data = ir4[2];
              ir4[2] = (v953_data + (v937_data * v66_bc));
              float v959_data = ir4[3];
              ir4[3] = (v959_data + (v937_data * v72_bc));
              float v965_data = ir4[4];
              ir4[4] = (v965_data + (v937_data * v78_bc));
              float v971_data = ir4[5];
              ir4[5] = (v971_data + (v937_data * v84_bc));
              float v977_data = ir4[6];
              ir4[6] = (v977_data + (v937_data * v90_bc));
              float v983_data = ir4[7];
              ir4[7] = (v983_data + (v937_data * v96_bc));
              float v989_data = ir4[8];
              ir4[8] = (v989_data + (v937_data * v102_bc));
              float v995_data = ir4[9];
              ir4[9] = (v995_data + (v937_data * v108_bc));
              float v1001_data = ir4[10];
              ir4[10] = (v1001_data + (v937_data * v114_bc));
              float v1007_data = ir4[11];
              ir4[11] = (v1007_data + (v937_data * v120_bc));
              float v1009_data = r3[1];
              float v1013_data = ir4[0];
              ir4[0] = (v1013_data + (v1009_data * v126_bc));
              float v1019_data = ir4[1];
              ir4[1] = (v1019_data + (v1009_data * v132_bc));
              float v1025_data = ir4[2];
              ir4[2] = (v1025_data + (v1009_data * v138_bc));
              float v1031_data = ir4[3];
              ir4[3] = (v1031_data + (v1009_data * v144_bc));
              float v1037_data = ir4[4];
              ir4[4] = (v1037_data + (v1009_data * v150_bc));
              float v1043_data = ir4[5];
              ir4[5] = (v1043_data + (v1009_data * v156_bc));
              float v1049_data = ir4[6];
              ir4[6] = (v1049_data + (v1009_data * v162_bc));
              float v1055_data = ir4[7];
              ir4[7] = (v1055_data + (v1009_data * v168_bc));
              float v1061_data = ir4[8];
              ir4[8] = (v1061_data + (v1009_data * v174_bc));
              float v1067_data = ir4[9];
              ir4[9] = (v1067_data + (v1009_data * v180_bc));
              float v1073_data = ir4[10];
              ir4[10] = (v1073_data + (v1009_data * v186_bc));
              float v1079_data = ir4[11];
              ir4[11] = (v1079_data + (v1009_data * v192_bc));
              float v1081_data = r3[2];
              float v1085_data = ir4[0];
              ir4[0] = (v1085_data + (v1081_data * v198_bc));
              float v1091_data = ir4[1];
              ir4[1] = (v1091_data + (v1081_data * v204_bc));
              float v1097_data = ir4[2];
              ir4[2] = (v1097_data + (v1081_data * v210_bc));
              float v1103_data = ir4[3];
              ir4[3] = (v1103_data + (v1081_data * v216_bc));
              float v1109_data = ir4[4];
              ir4[4] = (v1109_data + (v1081_data * v222_bc));
              float v1115_data = ir4[5];
              ir4[5] = (v1115_data + (v1081_data * v228_bc));
              float v1121_data = ir4[6];
              ir4[6] = (v1121_data + (v1081_data * v234_bc));
              float v1127_data = ir4[7];
              ir4[7] = (v1127_data + (v1081_data * v240_bc));
              float v1133_data = ir4[8];
              ir4[8] = (v1133_data + (v1081_data * v246_bc));
              float v1139_data = ir4[9];
              ir4[9] = (v1139_data + (v1081_data * v252_bc));
              float v1145_data = ir4[10];
              ir4[10] = (v1145_data + (v1081_data * v258_bc));
              float v1151_data = ir4[11];
              ir4[11] = (v1151_data + (v1081_data * v264_bc));
              float v1153_data = r3[3];
              float v1157_data = ir4[0];
              ir4[0] = (v1157_data + (v1153_data * v270_bc));
              float v1163_data = ir4[1];
              ir4[1] = (v1163_data + (v1153_data * v276_bc));
              float v1169_data = ir4[2];
              ir4[2] = (v1169_data + (v1153_data * v282_bc));
              float v1175_data = ir4[3];
              ir4[3] = (v1175_data + (v1153_data * v288_bc));
              float v1181_data = ir4[4];
              ir4[4] = (v1181_data + (v1153_data * v294_bc));
              float v1187_data = ir4[5];
              ir4[5] = (v1187_data + (v1153_data * v300_bc));
              float v1193_data = ir4[6];
              ir4[6] = (v1193_data + (v1153_data * v306_bc));
              float v1199_data = ir4[7];
              ir4[7] = (v1199_data + (v1153_data * v312_bc));
              float v1205_data = ir4[8];
              ir4[8] = (v1205_data + (v1153_data * v318_bc));
              float v1211_data = ir4[9];
              ir4[9] = (v1211_data + (v1153_data * v324_bc));
              float v1217_data = ir4[10];
              ir4[10] = (v1217_data + (v1153_data * v330_bc));
              float v1223_data = ir4[11];
              ir4[11] = (v1223_data + (v1153_data * v336_bc));
              float v1225_data = r3[4];
              float v1229_data = ir4[0];
              ir4[0] = (v1229_data + (v1225_data * v342_bc));
              float v1235_data = ir4[1];
              ir4[1] = (v1235_data + (v1225_data * v348_bc));
              float v1241_data = ir4[2];
              ir4[2] = (v1241_data + (v1225_data * v354_bc));
              float v1247_data = ir4[3];
              ir4[3] = (v1247_data + (v1225_data * v360_bc));
              float v1253_data = ir4[4];
              ir4[4] = (v1253_data + (v1225_data * v366_bc));
              float v1259_data = ir4[5];
              ir4[5] = (v1259_data + (v1225_data * v372_bc));
              float v1265_data = ir4[6];
              ir4[6] = (v1265_data + (v1225_data * v378_bc));
              float v1271_data = ir4[7];
              ir4[7] = (v1271_data + (v1225_data * v384_bc));
              float v1277_data = ir4[8];
              ir4[8] = (v1277_data + (v1225_data * v390_bc));
              float v1283_data = ir4[9];
              ir4[9] = (v1283_data + (v1225_data * v396_bc));
              float v1289_data = ir4[10];
              ir4[10] = (v1289_data + (v1225_data * v402_bc));
              float v1295_data = ir4[11];
              ir4[11] = (v1295_data + (v1225_data * v408_bc));
              float v1297_data = r3[5];
              float v1301_data = ir4[0];
              ir4[0] = (v1301_data + (v1297_data * v414_bc));
              float v1307_data = ir4[1];
              ir4[1] = (v1307_data + (v1297_data * v420_bc));
              float v1313_data = ir4[2];
              ir4[2] = (v1313_data + (v1297_data * v426_bc));
              float v1319_data = ir4[3];
              ir4[3] = (v1319_data + (v1297_data * v432_bc));
              float v1325_data = ir4[4];
              ir4[4] = (v1325_data + (v1297_data * v438_bc));
              float v1331_data = ir4[5];
              ir4[5] = (v1331_data + (v1297_data * v444_bc));
              float v1337_data = ir4[6];
              ir4[6] = (v1337_data + (v1297_data * v450_bc));
              float v1343_data = ir4[7];
              ir4[7] = (v1343_data + (v1297_data * v456_bc));
              float v1349_data = ir4[8];
              ir4[8] = (v1349_data + (v1297_data * v462_bc));
              float v1355_data = ir4[9];
              ir4[9] = (v1355_data + (v1297_data * v468_bc));
              float v1361_data = ir4[10];
              ir4[10] = (v1361_data + (v1297_data * v474_bc));
              float v1367_data = ir4[11];
              ir4[11] = (v1367_data + (v1297_data * v480_bc));
              float v1369_data = r3[6];
              float v1373_data = ir4[0];
              ir4[0] = (v1373_data + (v1369_data * v486_bc));
              float v1379_data = ir4[1];
              ir4[1] = (v1379_data + (v1369_data * v492_bc));
              float v1385_data = ir4[2];
              ir4[2] = (v1385_data + (v1369_data * v498_bc));
              float v1391_data = ir4[3];
              ir4[3] = (v1391_data + (v1369_data * v504_bc));
              float v1397_data = ir4[4];
              ir4[4] = (v1397_data + (v1369_data * v510_bc));
              float v1403_data = ir4[5];
              ir4[5] = (v1403_data + (v1369_data * v516_bc));
              float v1409_data = ir4[6];
              ir4[6] = (v1409_data + (v1369_data * v522_bc));
              float v1415_data = ir4[7];
              ir4[7] = (v1415_data + (v1369_data * v528_bc));
              float v1421_data = ir4[8];
              ir4[8] = (v1421_data + (v1369_data * v534_bc));
              float v1427_data = ir4[9];
              ir4[9] = (v1427_data + (v1369_data * v540_bc));
              float v1433_data = ir4[10];
              ir4[10] = (v1433_data + (v1369_data * v546_bc));
              float v1439_data = ir4[11];
              ir4[11] = (v1439_data + (v1369_data * v552_bc));
              float v1441_data = r3[7];
              float v1445_data = ir4[0];
              ir4[0] = (v1445_data + (v1441_data * v558_bc));
              float v1451_data = ir4[1];
              ir4[1] = (v1451_data + (v1441_data * v564_bc));
              float v1457_data = ir4[2];
              ir4[2] = (v1457_data + (v1441_data * v570_bc));
              float v1463_data = ir4[3];
              ir4[3] = (v1463_data + (v1441_data * v576_bc));
              float v1469_data = ir4[4];
              ir4[4] = (v1469_data + (v1441_data * v582_bc));
              float v1475_data = ir4[5];
              ir4[5] = (v1475_data + (v1441_data * v588_bc));
              float v1481_data = ir4[6];
              ir4[6] = (v1481_data + (v1441_data * v594_bc));
              float v1487_data = ir4[7];
              ir4[7] = (v1487_data + (v1441_data * v600_bc));
              float v1493_data = ir4[8];
              ir4[8] = (v1493_data + (v1441_data * v606_bc));
              float v1499_data = ir4[9];
              ir4[9] = (v1499_data + (v1441_data * v612_bc));
              float v1505_data = ir4[10];
              ir4[10] = (v1505_data + (v1441_data * v618_bc));
              float v1511_data = ir4[11];
              ir4[11] = (v1511_data + (v1441_data * v624_bc));
              float v1513_data = r3[8];
              float v1517_data = ir4[0];
              ir4[0] = (v1517_data + (v1513_data * v630_bc));
              float v1523_data = ir4[1];
              ir4[1] = (v1523_data + (v1513_data * v636_bc));
              float v1529_data = ir4[2];
              ir4[2] = (v1529_data + (v1513_data * v642_bc));
              float v1535_data = ir4[3];
              ir4[3] = (v1535_data + (v1513_data * v648_bc));
              float v1541_data = ir4[4];
              ir4[4] = (v1541_data + (v1513_data * v654_bc));
              float v1547_data = ir4[5];
              ir4[5] = (v1547_data + (v1513_data * v660_bc));
              float v1553_data = ir4[6];
              ir4[6] = (v1553_data + (v1513_data * v666_bc));
              float v1559_data = ir4[7];
              ir4[7] = (v1559_data + (v1513_data * v672_bc));
              float v1565_data = ir4[8];
              ir4[8] = (v1565_data + (v1513_data * v678_bc));
              float v1571_data = ir4[9];
              ir4[9] = (v1571_data + (v1513_data * v684_bc));
              float v1577_data = ir4[10];
              ir4[10] = (v1577_data + (v1513_data * v690_bc));
              float v1583_data = ir4[11];
              ir4[11] = (v1583_data + (v1513_data * v696_bc));
              float v1585_data = r3[9];
              float v1589_data = ir4[0];
              ir4[0] = (v1589_data + (v1585_data * v702_bc));
              float v1595_data = ir4[1];
              ir4[1] = (v1595_data + (v1585_data * v708_bc));
              float v1601_data = ir4[2];
              ir4[2] = (v1601_data + (v1585_data * v714_bc));
              float v1607_data = ir4[3];
              ir4[3] = (v1607_data + (v1585_data * v720_bc));
              float v1613_data = ir4[4];
              ir4[4] = (v1613_data + (v1585_data * v726_bc));
              float v1619_data = ir4[5];
              ir4[5] = (v1619_data + (v1585_data * v732_bc));
              float v1625_data = ir4[6];
              ir4[6] = (v1625_data + (v1585_data * v738_bc));
              float v1631_data = ir4[7];
              ir4[7] = (v1631_data + (v1585_data * v744_bc));
              float v1637_data = ir4[8];
              ir4[8] = (v1637_data + (v1585_data * v750_bc));
              float v1643_data = ir4[9];
              ir4[9] = (v1643_data + (v1585_data * v756_bc));
              float v1649_data = ir4[10];
              ir4[10] = (v1649_data + (v1585_data * v762_bc));
              float v1655_data = ir4[11];
              ir4[11] = (v1655_data + (v1585_data * v768_bc));
              float v1657_data = r3[10];
              float v1661_data = ir4[0];
              ir4[0] = (v1661_data + (v1657_data * v774_bc));
              float v1667_data = ir4[1];
              ir4[1] = (v1667_data + (v1657_data * v780_bc));
              float v1673_data = ir4[2];
              ir4[2] = (v1673_data + (v1657_data * v786_bc));
              float v1679_data = ir4[3];
              ir4[3] = (v1679_data + (v1657_data * v792_bc));
              float v1685_data = ir4[4];
              ir4[4] = (v1685_data + (v1657_data * v798_bc));
              float v1691_data = ir4[5];
              ir4[5] = (v1691_data + (v1657_data * v804_bc));
              float v1697_data = ir4[6];
              ir4[6] = (v1697_data + (v1657_data * v810_bc));
              float v1703_data = ir4[7];
              ir4[7] = (v1703_data + (v1657_data * v816_bc));
              float v1709_data = ir4[8];
              ir4[8] = (v1709_data + (v1657_data * v822_bc));
              float v1715_data = ir4[9];
              ir4[9] = (v1715_data + (v1657_data * v828_bc));
              float v1721_data = ir4[10];
              ir4[10] = (v1721_data + (v1657_data * v834_bc));
              float v1727_data = ir4[11];
              ir4[11] = (v1727_data + (v1657_data * v840_bc));
              float v1729_data = r3[11];
              float v1733_data = ir4[0];
              ir4[0] = (v1733_data + (v1729_data * v846_bc));
              float v1739_data = ir4[1];
              ir4[1] = (v1739_data + (v1729_data * v852_bc));
              float v1745_data = ir4[2];
              ir4[2] = (v1745_data + (v1729_data * v858_bc));
              float v1751_data = ir4[3];
              ir4[3] = (v1751_data + (v1729_data * v864_bc));
              float v1757_data = ir4[4];
              ir4[4] = (v1757_data + (v1729_data * v870_bc));
              float v1763_data = ir4[5];
              ir4[5] = (v1763_data + (v1729_data * v876_bc));
              float v1769_data = ir4[6];
              ir4[6] = (v1769_data + (v1729_data * v882_bc));
              float v1775_data = ir4[7];
              ir4[7] = (v1775_data + (v1729_data * v888_bc));
              float v1781_data = ir4[8];
              ir4[8] = (v1781_data + (v1729_data * v894_bc));
              float v1787_data = ir4[9];
              ir4[9] = (v1787_data + (v1729_data * v900_bc));
              float v1793_data = ir4[10];
              ir4[10] = (v1793_data + (v1729_data * v906_bc));
              float v1799_data = ir4[11];
              ir4[11] = (v1799_data + (v1729_data * v912_bc));
              // r4 = ir4
              if (v26_g) {
                #pragma unroll
                for (int32_t v1801_n1 = 0; v1801_n1 < 12; ++v1801_n1) {
                  float v1803_data = ir4[v1801_n1];
                  r4[v1801_n1] = v1803_data;
                }
              }
              // s0 = store{r>s}(localShrMem0, r4);
              if (v26_g) {
                int32_t v1809_off = v25_lead + 6;
                #pragma unroll
                for (int32_t v1804_i1 = 0; v1804_i1 < 12; ++v1804_i1) {
                  float v1806_data = r4[v1804_i1];
                  int32_t v1811_a = v1809_off + (v1804_i1 * 12);
                  s0[(v1811_a ^ ((v1811_a >> 4) & 15))] = v1806_data;
                }
              }
              float r5[12]{};
              // ir5 = +(s0)
              // [(0, 12), (0, 12)] []
              float ir5[12]{};
              int32_t v1821_sw = (v25_lead >> 4) & 15;
              int32_t v1822_sw = v25_lead ^ v1821_sw;
              sycl::group_barrier(item.get_sub_group());
              float v1823_data_pre = s0[v35_g ? (v1822_sw) : (0)];
              float v1823_data = v35_g ? (v1823_data_pre) : (0.0f);
              float v1824_data = ir5[0];
              ir5[0] = (v1824_data + v1823_data);
              int32_t v1826_a = v25_lead + 12;
              int32_t v1827_sw = v1826_a >> 4;
              float v1830_data_pre = s0[v35_g ? ((v1826_a ^ (v1827_sw & 15))) : (0)];
              float v1830_data = v35_g ? (v1830_data_pre) : (0.0f);
              float v1831_data = ir5[1];
              ir5[1] = (v1831_data + v1830_data);
              int32_t v1833_a = v25_lead + 24;
              int32_t v1834_sw = v1833_a >> 4;
              float v1837_data_pre = s0[v35_g ? ((v1833_a ^ (v1834_sw & 15))) : (0)];
              float v1837_data = v35_g ? (v1837_data_pre) : (0.0f);
              float v1838_data = ir5[2];
              ir5[2] = (v1838_data + v1837_data);
              int32_t v1840_a = v25_lead + 36;
              int32_t v1841_sw = v1840_a >> 4;
              float v1844_data_pre = s0[v35_g ? ((v1840_a ^ (v1841_sw & 15))) : (0)];
              float v1844_data = v35_g ? (v1844_data_pre) : (0.0f);
              float v1845_data = ir5[3];
              ir5[3] = (v1845_data + v1844_data);
              int32_t v1847_a = v25_lead + 48;
              int32_t v1848_sw = v1847_a >> 4;
              float v1851_data_pre = s0[v35_g ? ((v1847_a ^ (v1848_sw & 15))) : (0)];
              float v1851_data = v35_g ? (v1851_data_pre) : (0.0f);
              float v1852_data = ir5[4];
              ir5[4] = (v1852_data + v1851_data);
              int32_t v1854_a = v25_lead + 60;
              int32_t v1855_sw = v1854_a >> 4;
              float v1858_data_pre = s0[v35_g ? ((v1854_a ^ (v1855_sw & 15))) : (0)];
              float v1858_data = v35_g ? (v1858_data_pre) : (0.0f);
              float v1859_data = ir5[5];
              ir5[5] = (v1859_data + v1858_data);
              int32_t v1861_a = v25_lead + 72;
              int32_t v1862_sw = v1861_a >> 4;
              float v1865_data_pre = s0[v35_g ? ((v1861_a ^ (v1862_sw & 15))) : (0)];
              float v1865_data = v35_g ? (v1865_data_pre) : (0.0f);
              float v1866_data = ir5[6];
              ir5[6] = (v1866_data + v1865_data);
              int32_t v1868_a = v25_lead + 84;
              int32_t v1869_sw = v1868_a >> 4;
              float v1872_data_pre = s0[v35_g ? ((v1868_a ^ (v1869_sw & 15))) : (0)];
              float v1872_data = v35_g ? (v1872_data_pre) : (0.0f);
              float v1873_data = ir5[7];
              ir5[7] = (v1873_data + v1872_data);
              int32_t v1875_a = v25_lead + 96;
              int32_t v1876_sw = v1875_a >> 4;
              float v1879_data_pre = s0[v35_g ? ((v1875_a ^ (v1876_sw & 15))) : (0)];
              float v1879_data = v35_g ? (v1879_data_pre) : (0.0f);
              float v1880_data = ir5[8];
              ir5[8] = (v1880_data + v1879_data);
              int32_t v1882_a = v25_lead + 108;
              int32_t v1883_sw = v1882_a >> 4;
              float v1886_data_pre = s0[v35_g ? ((v1882_a ^ (v1883_sw & 15))) : (0)];
              float v1886_data = v35_g ? (v1886_data_pre) : (0.0f);
              float v1887_data = ir5[9];
              ir5[9] = (v1887_data + v1886_data);
              int32_t v1889_a = v25_lead + 120;
              int32_t v1890_sw = v1889_a >> 4;
              float v1893_data_pre = s0[v35_g ? ((v1889_a ^ (v1890_sw & 15))) : (0)];
              float v1893_data = v35_g ? (v1893_data_pre) : (0.0f);
              float v1894_data = ir5[10];
              ir5[10] = (v1894_data + v1893_data);
              int32_t v1896_a = v25_lead + 132;
              int32_t v1897_sw = v1896_a >> 4;
              float v1900_data_pre = s0[v35_g ? ((v1896_a ^ (v1897_sw & 15))) : (0)];
              float v1900_data = v35_g ? (v1900_data_pre) : (0.0f);
              float v1901_data = ir5[11];
              ir5[11] = (v1901_data + v1900_data);
              // r5 = ir5
              if (v35_g) {
                #pragma unroll
                for (int32_t v1903_n1 = 0; v1903_n1 < 12; ++v1903_n1) {
                  float v1905_data = ir5[v1903_n1];
                  r5[v1903_n1] = v1905_data;
                }
              }
              // glb_m3 = store{r>g}(r5);
              if (v35_g) {
                #pragma unroll
                for (int32_t v1906_i1 = 0; v1906_i1 < 12; ++v1906_i1) {
                  float v1908_data = r5[v1906_i1];
                  glb_m3[(v25_lead + (v1906_i1 * 12))] = v1908_data;
                }
              }
              // wait(r6 = load{g>r}(glb_m4););
              float r7[12]{};
              // ir7 = +(r6 * r1)
              // [(0, 2), (0, 12)] [(0, 12)]
              float ir7[12]{};
              float v1915_data = r6[0];
              float v1919_data = ir7[0];
              ir7[0] = (v1919_data + (v1915_data * v54_bc));
              float v1925_data = ir7[1];
              ir7[1] = (v1925_data + (v1915_data * v60_bc));
              float v1931_data = ir7[2];
              ir7[2] = (v1931_data + (v1915_data * v66_bc));
              float v1937_data = ir7[3];
              ir7[3] = (v1937_data + (v1915_data * v72_bc));
              float v1943_data = ir7[4];
              ir7[4] = (v1943_data + (v1915_data * v78_bc));
              float v1949_data = ir7[5];
              ir7[5] = (v1949_data + (v1915_data * v84_bc));
              float v1955_data = ir7[6];
              ir7[6] = (v1955_data + (v1915_data * v90_bc));
              float v1961_data = ir7[7];
              ir7[7] = (v1961_data + (v1915_data * v96_bc));
              float v1967_data = ir7[8];
              ir7[8] = (v1967_data + (v1915_data * v102_bc));
              float v1973_data = ir7[9];
              ir7[9] = (v1973_data + (v1915_data * v108_bc));
              float v1979_data = ir7[10];
              ir7[10] = (v1979_data + (v1915_data * v114_bc));
              float v1985_data = ir7[11];
              ir7[11] = (v1985_data + (v1915_data * v120_bc));
              float v1987_data = r6[1];
              float v1991_data = ir7[0];
              ir7[0] = (v1991_data + (v1987_data * v126_bc));
              float v1997_data = ir7[1];
              ir7[1] = (v1997_data + (v1987_data * v132_bc));
              float v2003_data = ir7[2];
              ir7[2] = (v2003_data + (v1987_data * v138_bc));
              float v2009_data = ir7[3];
              ir7[3] = (v2009_data + (v1987_data * v144_bc));
              float v2015_data = ir7[4];
              ir7[4] = (v2015_data + (v1987_data * v150_bc));
              float v2021_data = ir7[5];
              ir7[5] = (v2021_data + (v1987_data * v156_bc));
              float v2027_data = ir7[6];
              ir7[6] = (v2027_data + (v1987_data * v162_bc));
              float v2033_data = ir7[7];
              ir7[7] = (v2033_data + (v1987_data * v168_bc));
              float v2039_data = ir7[8];
              ir7[8] = (v2039_data + (v1987_data * v174_bc));
              float v2045_data = ir7[9];
              ir7[9] = (v2045_data + (v1987_data * v180_bc));
              float v2051_data = ir7[10];
              ir7[10] = (v2051_data + (v1987_data * v186_bc));
              float v2057_data = ir7[11];
              ir7[11] = (v2057_data + (v1987_data * v192_bc));
              float v2059_data = r6[2];
              float v2063_data = ir7[0];
              ir7[0] = (v2063_data + (v2059_data * v198_bc));
              float v2069_data = ir7[1];
              ir7[1] = (v2069_data + (v2059_data * v204_bc));
              float v2075_data = ir7[2];
              ir7[2] = (v2075_data + (v2059_data * v210_bc));
              float v2081_data = ir7[3];
              ir7[3] = (v2081_data + (v2059_data * v216_bc));
              float v2087_data = ir7[4];
              ir7[4] = (v2087_data + (v2059_data * v222_bc));
              float v2093_data = ir7[5];
              ir7[5] = (v2093_data + (v2059_data * v228_bc));
              float v2099_data = ir7[6];
              ir7[6] = (v2099_data + (v2059_data * v234_bc));
              float v2105_data = ir7[7];
              ir7[7] = (v2105_data + (v2059_data * v240_bc));
              float v2111_data = ir7[8];
              ir7[8] = (v2111_data + (v2059_data * v246_bc));
              float v2117_data = ir7[9];
              ir7[9] = (v2117_data + (v2059_data * v252_bc));
              float v2123_data = ir7[10];
              ir7[10] = (v2123_data + (v2059_data * v258_bc));
              float v2129_data = ir7[11];
              ir7[11] = (v2129_data + (v2059_data * v264_bc));
              float v2131_data = r6[3];
              float v2135_data = ir7[0];
              ir7[0] = (v2135_data + (v2131_data * v270_bc));
              float v2141_data = ir7[1];
              ir7[1] = (v2141_data + (v2131_data * v276_bc));
              float v2147_data = ir7[2];
              ir7[2] = (v2147_data + (v2131_data * v282_bc));
              float v2153_data = ir7[3];
              ir7[3] = (v2153_data + (v2131_data * v288_bc));
              float v2159_data = ir7[4];
              ir7[4] = (v2159_data + (v2131_data * v294_bc));
              float v2165_data = ir7[5];
              ir7[5] = (v2165_data + (v2131_data * v300_bc));
              float v2171_data = ir7[6];
              ir7[6] = (v2171_data + (v2131_data * v306_bc));
              float v2177_data = ir7[7];
              ir7[7] = (v2177_data + (v2131_data * v312_bc));
              float v2183_data = ir7[8];
              ir7[8] = (v2183_data + (v2131_data * v318_bc));
              float v2189_data = ir7[9];
              ir7[9] = (v2189_data + (v2131_data * v324_bc));
              float v2195_data = ir7[10];
              ir7[10] = (v2195_data + (v2131_data * v330_bc));
              float v2201_data = ir7[11];
              ir7[11] = (v2201_data + (v2131_data * v336_bc));
              float v2203_data = r6[4];
              float v2207_data = ir7[0];
              ir7[0] = (v2207_data + (v2203_data * v342_bc));
              float v2213_data = ir7[1];
              ir7[1] = (v2213_data + (v2203_data * v348_bc));
              float v2219_data = ir7[2];
              ir7[2] = (v2219_data + (v2203_data * v354_bc));
              float v2225_data = ir7[3];
              ir7[3] = (v2225_data + (v2203_data * v360_bc));
              float v2231_data = ir7[4];
              ir7[4] = (v2231_data + (v2203_data * v366_bc));
              float v2237_data = ir7[5];
              ir7[5] = (v2237_data + (v2203_data * v372_bc));
              float v2243_data = ir7[6];
              ir7[6] = (v2243_data + (v2203_data * v378_bc));
              float v2249_data = ir7[7];
              ir7[7] = (v2249_data + (v2203_data * v384_bc));
              float v2255_data = ir7[8];
              ir7[8] = (v2255_data + (v2203_data * v390_bc));
              float v2261_data = ir7[9];
              ir7[9] = (v2261_data + (v2203_data * v396_bc));
              float v2267_data = ir7[10];
              ir7[10] = (v2267_data + (v2203_data * v402_bc));
              float v2273_data = ir7[11];
              ir7[11] = (v2273_data + (v2203_data * v408_bc));
              float v2275_data = r6[5];
              float v2279_data = ir7[0];
              ir7[0] = (v2279_data + (v2275_data * v414_bc));
              float v2285_data = ir7[1];
              ir7[1] = (v2285_data + (v2275_data * v420_bc));
              float v2291_data = ir7[2];
              ir7[2] = (v2291_data + (v2275_data * v426_bc));
              float v2297_data = ir7[3];
              ir7[3] = (v2297_data + (v2275_data * v432_bc));
              float v2303_data = ir7[4];
              ir7[4] = (v2303_data + (v2275_data * v438_bc));
              float v2309_data = ir7[5];
              ir7[5] = (v2309_data + (v2275_data * v444_bc));
              float v2315_data = ir7[6];
              ir7[6] = (v2315_data + (v2275_data * v450_bc));
              float v2321_data = ir7[7];
              ir7[7] = (v2321_data + (v2275_data * v456_bc));
              float v2327_data = ir7[8];
              ir7[8] = (v2327_data + (v2275_data * v462_bc));
              float v2333_data = ir7[9];
              ir7[9] = (v2333_data + (v2275_data * v468_bc));
              float v2339_data = ir7[10];
              ir7[10] = (v2339_data + (v2275_data * v474_bc));
              float v2345_data = ir7[11];
              ir7[11] = (v2345_data + (v2275_data * v480_bc));
              float v2347_data = r6[6];
              float v2351_data = ir7[0];
              ir7[0] = (v2351_data + (v2347_data * v486_bc));
              float v2357_data = ir7[1];
              ir7[1] = (v2357_data + (v2347_data * v492_bc));
              float v2363_data = ir7[2];
              ir7[2] = (v2363_data + (v2347_data * v498_bc));
              float v2369_data = ir7[3];
              ir7[3] = (v2369_data + (v2347_data * v504_bc));
              float v2375_data = ir7[4];
              ir7[4] = (v2375_data + (v2347_data * v510_bc));
              float v2381_data = ir7[5];
              ir7[5] = (v2381_data + (v2347_data * v516_bc));
              float v2387_data = ir7[6];
              ir7[6] = (v2387_data + (v2347_data * v522_bc));
              float v2393_data = ir7[7];
              ir7[7] = (v2393_data + (v2347_data * v528_bc));
              float v2399_data = ir7[8];
              ir7[8] = (v2399_data + (v2347_data * v534_bc));
              float v2405_data = ir7[9];
              ir7[9] = (v2405_data + (v2347_data * v540_bc));
              float v2411_data = ir7[10];
              ir7[10] = (v2411_data + (v2347_data * v546_bc));
              float v2417_data = ir7[11];
              ir7[11] = (v2417_data + (v2347_data * v552_bc));
              float v2419_data = r6[7];
              float v2423_data = ir7[0];
              ir7[0] = (v2423_data + (v2419_data * v558_bc));
              float v2429_data = ir7[1];
              ir7[1] = (v2429_data + (v2419_data * v564_bc));
              float v2435_data = ir7[2];
              ir7[2] = (v2435_data + (v2419_data * v570_bc));
              float v2441_data = ir7[3];
              ir7[3] = (v2441_data + (v2419_data * v576_bc));
              float v2447_data = ir7[4];
              ir7[4] = (v2447_data + (v2419_data * v582_bc));
              float v2453_data = ir7[5];
              ir7[5] = (v2453_data + (v2419_data * v588_bc));
              float v2459_data = ir7[6];
              ir7[6] = (v2459_data + (v2419_data * v594_bc));
              float v2465_data = ir7[7];
              ir7[7] = (v2465_data + (v2419_data * v600_bc));
              float v2471_data = ir7[8];
              ir7[8] = (v2471_data + (v2419_data * v606_bc));
              float v2477_data = ir7[9];
              ir7[9] = (v2477_data + (v2419_data * v612_bc));
              float v2483_data = ir7[10];
              ir7[10] = (v2483_data + (v2419_data * v618_bc));
              float v2489_data = ir7[11];
              ir7[11] = (v2489_data + (v2419_data * v624_bc));
              float v2491_data = r6[8];
              float v2495_data = ir7[0];
              ir7[0] = (v2495_data + (v2491_data * v630_bc));
              float v2501_data = ir7[1];
              ir7[1] = (v2501_data + (v2491_data * v636_bc));
              float v2507_data = ir7[2];
              ir7[2] = (v2507_data + (v2491_data * v642_bc));
              float v2513_data = ir7[3];
              ir7[3] = (v2513_data + (v2491_data * v648_bc));
              float v2519_data = ir7[4];
              ir7[4] = (v2519_data + (v2491_data * v654_bc));
              float v2525_data = ir7[5];
              ir7[5] = (v2525_data + (v2491_data * v660_bc));
              float v2531_data = ir7[6];
              ir7[6] = (v2531_data + (v2491_data * v666_bc));
              float v2537_data = ir7[7];
              ir7[7] = (v2537_data + (v2491_data * v672_bc));
              float v2543_data = ir7[8];
              ir7[8] = (v2543_data + (v2491_data * v678_bc));
              float v2549_data = ir7[9];
              ir7[9] = (v2549_data + (v2491_data * v684_bc));
              float v2555_data = ir7[10];
              ir7[10] = (v2555_data + (v2491_data * v690_bc));
              float v2561_data = ir7[11];
              ir7[11] = (v2561_data + (v2491_data * v696_bc));
              float v2563_data = r6[9];
              float v2567_data = ir7[0];
              ir7[0] = (v2567_data + (v2563_data * v702_bc));
              float v2573_data = ir7[1];
              ir7[1] = (v2573_data + (v2563_data * v708_bc));
              float v2579_data = ir7[2];
              ir7[2] = (v2579_data + (v2563_data * v714_bc));
              float v2585_data = ir7[3];
              ir7[3] = (v2585_data + (v2563_data * v720_bc));
              float v2591_data = ir7[4];
              ir7[4] = (v2591_data + (v2563_data * v726_bc));
              float v2597_data = ir7[5];
              ir7[5] = (v2597_data + (v2563_data * v732_bc));
              float v2603_data = ir7[6];
              ir7[6] = (v2603_data + (v2563_data * v738_bc));
              float v2609_data = ir7[7];
              ir7[7] = (v2609_data + (v2563_data * v744_bc));
              float v2615_data = ir7[8];
              ir7[8] = (v2615_data + (v2563_data * v750_bc));
              float v2621_data = ir7[9];
              ir7[9] = (v2621_data + (v2563_data * v756_bc));
              float v2627_data = ir7[10];
              ir7[10] = (v2627_data + (v2563_data * v762_bc));
              float v2633_data = ir7[11];
              ir7[11] = (v2633_data + (v2563_data * v768_bc));
              float v2635_data = r6[10];
              float v2639_data = ir7[0];
              ir7[0] = (v2639_data + (v2635_data * v774_bc));
              float v2645_data = ir7[1];
              ir7[1] = (v2645_data + (v2635_data * v780_bc));
              float v2651_data = ir7[2];
              ir7[2] = (v2651_data + (v2635_data * v786_bc));
              float v2657_data = ir7[3];
              ir7[3] = (v2657_data + (v2635_data * v792_bc));
              float v2663_data = ir7[4];
              ir7[4] = (v2663_data + (v2635_data * v798_bc));
              float v2669_data = ir7[5];
              ir7[5] = (v2669_data + (v2635_data * v804_bc));
              float v2675_data = ir7[6];
              ir7[6] = (v2675_data + (v2635_data * v810_bc));
              float v2681_data = ir7[7];
              ir7[7] = (v2681_data + (v2635_data * v816_bc));
              float v2687_data = ir7[8];
              ir7[8] = (v2687_data + (v2635_data * v822_bc));
              float v2693_data = ir7[9];
              ir7[9] = (v2693_data + (v2635_data * v828_bc));
              float v2699_data = ir7[10];
              ir7[10] = (v2699_data + (v2635_data * v834_bc));
              float v2705_data = ir7[11];
              ir7[11] = (v2705_data + (v2635_data * v840_bc));
              float v2707_data = r6[11];
              float v2711_data = ir7[0];
              ir7[0] = (v2711_data + (v2707_data * v846_bc));
              float v2717_data = ir7[1];
              ir7[1] = (v2717_data + (v2707_data * v852_bc));
              float v2723_data = ir7[2];
              ir7[2] = (v2723_data + (v2707_data * v858_bc));
              float v2729_data = ir7[3];
              ir7[3] = (v2729_data + (v2707_data * v864_bc));
              float v2735_data = ir7[4];
              ir7[4] = (v2735_data + (v2707_data * v870_bc));
              float v2741_data = ir7[5];
              ir7[5] = (v2741_data + (v2707_data * v876_bc));
              float v2747_data = ir7[6];
              ir7[6] = (v2747_data + (v2707_data * v882_bc));
              float v2753_data = ir7[7];
              ir7[7] = (v2753_data + (v2707_data * v888_bc));
              float v2759_data = ir7[8];
              ir7[8] = (v2759_data + (v2707_data * v894_bc));
              float v2765_data = ir7[9];
              ir7[9] = (v2765_data + (v2707_data * v900_bc));
              float v2771_data = ir7[10];
              ir7[10] = (v2771_data + (v2707_data * v906_bc));
              float v2777_data = ir7[11];
              ir7[11] = (v2777_data + (v2707_data * v912_bc));
              // r7 = ir7
              if (v927_g) {
                #pragma unroll
                for (int32_t v2779_n1 = 0; v2779_n1 < 12; ++v2779_n1) {
                  float v2781_data = ir7[v2779_n1];
                  r7[v2779_n1] = v2781_data;
                }
              }
              // s0 = store{r>s, clear}(localShrMem0, r7);
              bool v2783_g = (v25_lead >= 8) && v35_g;
              sycl::group_barrier(item.get_sub_group());
              if (v2783_g) {
                #pragma unroll
                for (int32_t v2784_z1 = 0; v2784_z1 < 12; ++v2784_z1) {
                  int32_t v2789_a = v25_lead + (v2784_z1 * 12);
                  s0[(v2789_a ^ ((v2789_a >> 4) & 15))] = 0.0f;
                }
              }
              if (v927_g) {
                int32_t v2798_off = v25_lead + 6;
                #pragma unroll
                for (int32_t v2793_i1 = 0; v2793_i1 < 12; ++v2793_i1) {
                  float v2795_data = r7[v2793_i1];
                  int32_t v2800_a = v2798_off + (v2793_i1 * 12);
                  s0[(v2800_a ^ ((v2800_a >> 4) & 15))] = v2795_data;
                }
              }
              float r8[12]{};
              // ir8 = +(s0)
              // [(0, 12), (0, 12)] []
              float ir8[12]{};
              int32_t v2811_sw = v25_lead ^ v1821_sw;
              sycl::group_barrier(item.get_sub_group());
              float v2812_data_pre = s0[v35_g ? (v2811_sw) : (0)];
              float v2812_data = v35_g ? (v2812_data_pre) : (0.0f);
              float v2813_data = ir8[0];
              ir8[0] = (v2813_data + v2812_data);
              float v2819_data_pre = s0[v35_g ? ((v1826_a ^ (v1827_sw & 15))) : (0)];
              float v2819_data = v35_g ? (v2819_data_pre) : (0.0f);
              float v2820_data = ir8[1];
              ir8[1] = (v2820_data + v2819_data);
              float v2826_data_pre = s0[v35_g ? ((v1833_a ^ (v1834_sw & 15))) : (0)];
              float v2826_data = v35_g ? (v2826_data_pre) : (0.0f);
              float v2827_data = ir8[2];
              ir8[2] = (v2827_data + v2826_data);
              float v2833_data_pre = s0[v35_g ? ((v1840_a ^ (v1841_sw & 15))) : (0)];
              float v2833_data = v35_g ? (v2833_data_pre) : (0.0f);
              float v2834_data = ir8[3];
              ir8[3] = (v2834_data + v2833_data);
              float v2840_data_pre = s0[v35_g ? ((v1847_a ^ (v1848_sw & 15))) : (0)];
              float v2840_data = v35_g ? (v2840_data_pre) : (0.0f);
              float v2841_data = ir8[4];
              ir8[4] = (v2841_data + v2840_data);
              float v2847_data_pre = s0[v35_g ? ((v1854_a ^ (v1855_sw & 15))) : (0)];
              float v2847_data = v35_g ? (v2847_data_pre) : (0.0f);
              float v2848_data = ir8[5];
              ir8[5] = (v2848_data + v2847_data);
              float v2854_data_pre = s0[v35_g ? ((v1861_a ^ (v1862_sw & 15))) : (0)];
              float v2854_data = v35_g ? (v2854_data_pre) : (0.0f);
              float v2855_data = ir8[6];
              ir8[6] = (v2855_data + v2854_data);
              float v2861_data_pre = s0[v35_g ? ((v1868_a ^ (v1869_sw & 15))) : (0)];
              float v2861_data = v35_g ? (v2861_data_pre) : (0.0f);
              float v2862_data = ir8[7];
              ir8[7] = (v2862_data + v2861_data);
              float v2868_data_pre = s0[v35_g ? ((v1875_a ^ (v1876_sw & 15))) : (0)];
              float v2868_data = v35_g ? (v2868_data_pre) : (0.0f);
              float v2869_data = ir8[8];
              ir8[8] = (v2869_data + v2868_data);
              float v2875_data_pre = s0[v35_g ? ((v1882_a ^ (v1883_sw & 15))) : (0)];
              float v2875_data = v35_g ? (v2875_data_pre) : (0.0f);
              float v2876_data = ir8[9];
              ir8[9] = (v2876_data + v2875_data);
              float v2882_data_pre = s0[v35_g ? ((v1889_a ^ (v1890_sw & 15))) : (0)];
              float v2882_data = v35_g ? (v2882_data_pre) : (0.0f);
              float v2883_data = ir8[10];
              ir8[10] = (v2883_data + v2882_data);
              float v2889_data_pre = s0[v35_g ? ((v1896_a ^ (v1897_sw & 15))) : (0)];
              float v2889_data = v35_g ? (v2889_data_pre) : (0.0f);
              float v2890_data = ir8[11];
              ir8[11] = (v2890_data + v2889_data);
              // r8 = ir8
              if (v35_g) {
                #pragma unroll
                for (int32_t v2892_n1 = 0; v2892_n1 < 12; ++v2892_n1) {
                  float v2894_data = ir8[v2892_n1];
                  r8[v2892_n1] = v2894_data;
                }
              }
              // glb_m5 = store{r>g}(r8);
              if (v35_g) {
                #pragma unroll
                for (int32_t v2895_i1 = 0; v2895_i1 < 12; ++v2895_i1) {
                  float v2897_data = r8[v2895_i1];
                  glb_m5[(v25_lead + (v2895_i1 * 12))] = v2897_data;
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

