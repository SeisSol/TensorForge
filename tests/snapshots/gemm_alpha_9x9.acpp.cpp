// === base name ===
kernel_6a376f613d663b57

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_6a376f613d663b57 = {{16, 16, 1}, 16, 9, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_6a376f613d663b57(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_6a376f613d663b57(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_6a376f613d663b57(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_6a376f613d663b57(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_6a376f613d663b57(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_6a376f613d663b57(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_6a376f613d663b57(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (9 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 9×9(9×9) {0..9}×{0..9} strided
        //   m1 9×9(9×9) {0..9}×{0..9} strided
        //   m2 9×9(9×9) {0..9}×{0..9} strided
        //   m3 ()  scalar
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j] × m3[]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":9,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[9,9]],"name":"m0","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[9,9]],"name":"m1","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[9,9]],"name":"m2","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"scalar","alias":null,"bbox":[[],[]],"name":"m3","ordered":false,"parts":1,"shape":[],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[9,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[9,9]},{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[9,9]},{"addressing":"scalar","bbox":[[],[]],"is_tmp":false,"name":"m3","offset":[],"shape":[]}],"permute":[[0,1],[0,1],[]],"target":[[0,-1],[-1,1],[]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 81 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 81 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 81 + 0 + m2_extraOffset];
              float r0[9]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v21_lead = item.get_local_id(2) % 16;
              bool v22_g = v21_lead < 9;
              if (v22_g) {
                #pragma unroll
                for (int32_t v23_i1 = 0; v23_i1 < 9; ++v23_i1) {
                  float v28_data = glb_m1[(v21_lead + (v23_i1 * 9))];
                  r0[v23_i1] = v28_data;
                }
              }
              float r1[9]{};
              // r1 = load{g>r}(glb_m2);
              if (v22_g) {
                #pragma unroll
                for (int32_t v31_i1 = 0; v31_i1 < 9; ++v31_i1) {
                  float v36_data = glb_m2[(v21_lead + (v31_i1 * 9))];
                  r1[v31_i1] = v36_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[9]{};
              // ir2 = +(r0 * r1)
              // [(0, 9), (0, 9)] [(0, 9)]
              float ir2[9]{};
              float v40_data = r0[0];
              float v41_data = r1[0];
              float v44_data = ir2[0];
              ir2[0] = (v44_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v47_data = r1[1];
              float v50_data = ir2[1];
              ir2[1] = (v50_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v53_data = r1[2];
              float v56_data = ir2[2];
              ir2[2] = (v56_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v59_data = r1[3];
              float v62_data = ir2[3];
              ir2[3] = (v62_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v65_data = r1[4];
              float v68_data = ir2[4];
              ir2[4] = (v68_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v71_data = r1[5];
              float v74_data = ir2[5];
              ir2[5] = (v74_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v77_data = r1[6];
              float v80_data = ir2[6];
              ir2[6] = (v80_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v83_data = r1[7];
              float v86_data = ir2[7];
              ir2[7] = (v86_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v89_data = r1[8];
              float v92_data = ir2[8];
              ir2[8] = (v92_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v94_data = r0[1];
              float v98_data = ir2[0];
              ir2[0] = (v98_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v104_data = ir2[1];
              ir2[1] = (v104_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v110_data = ir2[2];
              ir2[2] = (v110_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v116_data = ir2[3];
              ir2[3] = (v116_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v122_data = ir2[4];
              ir2[4] = (v122_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v128_data = ir2[5];
              ir2[5] = (v128_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v134_data = ir2[6];
              ir2[6] = (v134_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v140_data = ir2[7];
              ir2[7] = (v140_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v146_data = ir2[8];
              ir2[8] = (v146_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v148_data = r0[2];
              float v152_data = ir2[0];
              ir2[0] = (v152_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v158_data = ir2[1];
              ir2[1] = (v158_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v164_data = ir2[2];
              ir2[2] = (v164_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v170_data = ir2[3];
              ir2[3] = (v170_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v176_data = ir2[4];
              ir2[4] = (v176_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v182_data = ir2[5];
              ir2[5] = (v182_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v188_data = ir2[6];
              ir2[6] = (v188_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v194_data = ir2[7];
              ir2[7] = (v194_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v200_data = ir2[8];
              ir2[8] = (v200_data + (v148_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v202_data = r0[3];
              float v206_data = ir2[0];
              ir2[0] = (v206_data + (v202_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v212_data = ir2[1];
              ir2[1] = (v212_data + (v202_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v218_data = ir2[2];
              ir2[2] = (v218_data + (v202_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v224_data = ir2[3];
              ir2[3] = (v224_data + (v202_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v230_data = ir2[4];
              ir2[4] = (v230_data + (v202_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v236_data = ir2[5];
              ir2[5] = (v236_data + (v202_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v242_data = ir2[6];
              ir2[6] = (v242_data + (v202_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v248_data = ir2[7];
              ir2[7] = (v248_data + (v202_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v254_data = ir2[8];
              ir2[8] = (v254_data + (v202_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v256_data = r0[4];
              float v260_data = ir2[0];
              ir2[0] = (v260_data + (v256_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v266_data = ir2[1];
              ir2[1] = (v266_data + (v256_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v272_data = ir2[2];
              ir2[2] = (v272_data + (v256_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v278_data = ir2[3];
              ir2[3] = (v278_data + (v256_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v284_data = ir2[4];
              ir2[4] = (v284_data + (v256_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v290_data = ir2[5];
              ir2[5] = (v290_data + (v256_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v296_data = ir2[6];
              ir2[6] = (v296_data + (v256_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v302_data = ir2[7];
              ir2[7] = (v302_data + (v256_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v308_data = ir2[8];
              ir2[8] = (v308_data + (v256_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v310_data = r0[5];
              float v314_data = ir2[0];
              ir2[0] = (v314_data + (v310_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v320_data = ir2[1];
              ir2[1] = (v320_data + (v310_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v326_data = ir2[2];
              ir2[2] = (v326_data + (v310_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v332_data = ir2[3];
              ir2[3] = (v332_data + (v310_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v338_data = ir2[4];
              ir2[4] = (v338_data + (v310_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v344_data = ir2[5];
              ir2[5] = (v344_data + (v310_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v350_data = ir2[6];
              ir2[6] = (v350_data + (v310_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v356_data = ir2[7];
              ir2[7] = (v356_data + (v310_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v362_data = ir2[8];
              ir2[8] = (v362_data + (v310_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v364_data = r0[6];
              float v368_data = ir2[0];
              ir2[0] = (v368_data + (v364_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v374_data = ir2[1];
              ir2[1] = (v374_data + (v364_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v380_data = ir2[2];
              ir2[2] = (v380_data + (v364_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v386_data = ir2[3];
              ir2[3] = (v386_data + (v364_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v392_data = ir2[4];
              ir2[4] = (v392_data + (v364_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v398_data = ir2[5];
              ir2[5] = (v398_data + (v364_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v404_data = ir2[6];
              ir2[6] = (v404_data + (v364_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v410_data = ir2[7];
              ir2[7] = (v410_data + (v364_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v416_data = ir2[8];
              ir2[8] = (v416_data + (v364_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v418_data = r0[7];
              float v422_data = ir2[0];
              ir2[0] = (v422_data + (v418_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v428_data = ir2[1];
              ir2[1] = (v428_data + (v418_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v434_data = ir2[2];
              ir2[2] = (v434_data + (v418_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v440_data = ir2[3];
              ir2[3] = (v440_data + (v418_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v446_data = ir2[4];
              ir2[4] = (v446_data + (v418_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v452_data = ir2[5];
              ir2[5] = (v452_data + (v418_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v458_data = ir2[6];
              ir2[6] = (v458_data + (v418_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v464_data = ir2[7];
              ir2[7] = (v464_data + (v418_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v470_data = ir2[8];
              ir2[8] = (v470_data + (v418_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v472_data = r0[8];
              float v476_data = ir2[0];
              ir2[0] = (v476_data + (v472_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v482_data = ir2[1];
              ir2[1] = (v482_data + (v472_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v488_data = ir2[2];
              ir2[2] = (v488_data + (v472_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v494_data = ir2[3];
              ir2[3] = (v494_data + (v472_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v500_data = ir2[4];
              ir2[4] = (v500_data + (v472_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v506_data = ir2[5];
              ir2[5] = (v506_data + (v472_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v512_data = ir2[6];
              ir2[6] = (v512_data + (v472_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v518_data = ir2[7];
              ir2[7] = (v518_data + (v472_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v524_data = ir2[8];
              ir2[8] = (v524_data + (v472_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              // r2 = ir2 * glb_m3
              if (v22_g) {
                #pragma unroll
                for (int32_t v527_n1 = 0; v527_n1 < 9; ++v527_n1) {
                  float v529_data = ir2[v527_n1];
                  r2[v527_n1] = (v529_data * 13.0f);
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v22_g) {
                #pragma unroll
                for (int32_t v531_i1 = 0; v531_i1 < 9; ++v531_i1) {
                  float v533_data = r2[v531_i1];
                  glb_m0[(v21_lead + (v531_i1 * 9))] = v533_data;
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

