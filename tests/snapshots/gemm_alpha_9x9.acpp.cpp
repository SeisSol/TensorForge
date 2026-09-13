// === base name ===
kernel_8012ecb37a10bbc1

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_8012ecb37a10bbc1 = {{16, 16, 1}, 16, 9, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_8012ecb37a10bbc1(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_8012ecb37a10bbc1(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_8012ecb37a10bbc1(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_8012ecb37a10bbc1(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_8012ecb37a10bbc1(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_8012ecb37a10bbc1(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_8012ecb37a10bbc1(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
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
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":9,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[9,9]],"name":"m0","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[9,9]],"name":"m1","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[9,9]],"name":"m2","ordered":false,"parts":1,"shape":[9,9],"variant":false},{"addressing":"scalar","alias":null,"bbox":[[],[]],"name":"m3","ordered":false,"parts":1,"shape":[],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[9,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[9,9]},{"addressing":"strided","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[9,9]},{"addressing":"scalar","bbox":[[],[]],"is_tmp":false,"name":"m3","offset":[],"shape":[]}],"permute":[[0,1],[0,1],[]],"target":[[0,-1],[-1,1],[]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t v3_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v3_batchId0 < numElements0; v3_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v4_ahead1 = v3_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v3_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v3_batchId0 * 81 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 81 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v3_batchId0 * 81 + 0 + m2_extraOffset];
              float r0[9]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v17_lead = item.get_local_id(2) % 16;
              bool v18_g = v17_lead < 9;
              if (v18_g) {
                #pragma unroll
                for (int32_t v19_i1 = 0; v19_i1 < 9; ++v19_i1) {
                  float v24_data = glb_m1[(v17_lead + (v19_i1 * 9))];
                  r0[v19_i1] = v24_data;
                }
              }
              float r1[9]{};
              // r1 = load{g>r}(glb_m2);
              if (v18_g) {
                #pragma unroll
                for (int32_t v27_i1 = 0; v27_i1 < 9; ++v27_i1) {
                  float v32_data = glb_m2[(v17_lead + (v27_i1 * 9))];
                  r1[v27_i1] = v32_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[9]{};
              // r2 = +(r0 * r1) + None
              // [(0, 9), (0, 9)] [(0, 9)]
              float ir2[9]{};
              float v36_data = r0[0];
              float v37_data = r1[0];
              float v40_data = ir2[0];
              ir2[0] = (v40_data + (v36_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v43_data = r1[1];
              float v46_data = ir2[1];
              ir2[1] = (v46_data + (v36_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v49_data = r1[2];
              float v52_data = ir2[2];
              ir2[2] = (v52_data + (v36_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v55_data = r1[3];
              float v58_data = ir2[3];
              ir2[3] = (v58_data + (v36_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v61_data = r1[4];
              float v64_data = ir2[4];
              ir2[4] = (v64_data + (v36_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v67_data = r1[5];
              float v70_data = ir2[5];
              ir2[5] = (v70_data + (v36_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v73_data = r1[6];
              float v76_data = ir2[6];
              ir2[6] = (v76_data + (v36_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v79_data = r1[7];
              float v82_data = ir2[7];
              ir2[7] = (v82_data + (v36_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v85_data = r1[8];
              float v88_data = ir2[8];
              ir2[8] = (v88_data + (v36_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v90_data = r0[1];
              float v94_data = ir2[0];
              ir2[0] = (v94_data + (v90_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v100_data = ir2[1];
              ir2[1] = (v100_data + (v90_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v106_data = ir2[2];
              ir2[2] = (v106_data + (v90_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v112_data = ir2[3];
              ir2[3] = (v112_data + (v90_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v118_data = ir2[4];
              ir2[4] = (v118_data + (v90_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v124_data = ir2[5];
              ir2[5] = (v124_data + (v90_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v130_data = ir2[6];
              ir2[6] = (v130_data + (v90_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v136_data = ir2[7];
              ir2[7] = (v136_data + (v90_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v142_data = ir2[8];
              ir2[8] = (v142_data + (v90_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v144_data = r0[2];
              float v148_data = ir2[0];
              ir2[0] = (v148_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v154_data = ir2[1];
              ir2[1] = (v154_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v160_data = ir2[2];
              ir2[2] = (v160_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v166_data = ir2[3];
              ir2[3] = (v166_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v172_data = ir2[4];
              ir2[4] = (v172_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v178_data = ir2[5];
              ir2[5] = (v178_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v184_data = ir2[6];
              ir2[6] = (v184_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v190_data = ir2[7];
              ir2[7] = (v190_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v196_data = ir2[8];
              ir2[8] = (v196_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v198_data = r0[3];
              float v202_data = ir2[0];
              ir2[0] = (v202_data + (v198_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v208_data = ir2[1];
              ir2[1] = (v208_data + (v198_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v214_data = ir2[2];
              ir2[2] = (v214_data + (v198_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v220_data = ir2[3];
              ir2[3] = (v220_data + (v198_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v226_data = ir2[4];
              ir2[4] = (v226_data + (v198_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v232_data = ir2[5];
              ir2[5] = (v232_data + (v198_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v238_data = ir2[6];
              ir2[6] = (v238_data + (v198_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v244_data = ir2[7];
              ir2[7] = (v244_data + (v198_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v250_data = ir2[8];
              ir2[8] = (v250_data + (v198_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v252_data = r0[4];
              float v256_data = ir2[0];
              ir2[0] = (v256_data + (v252_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v262_data = ir2[1];
              ir2[1] = (v262_data + (v252_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v268_data = ir2[2];
              ir2[2] = (v268_data + (v252_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v274_data = ir2[3];
              ir2[3] = (v274_data + (v252_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v280_data = ir2[4];
              ir2[4] = (v280_data + (v252_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v286_data = ir2[5];
              ir2[5] = (v286_data + (v252_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v292_data = ir2[6];
              ir2[6] = (v292_data + (v252_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v298_data = ir2[7];
              ir2[7] = (v298_data + (v252_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v304_data = ir2[8];
              ir2[8] = (v304_data + (v252_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v306_data = r0[5];
              float v310_data = ir2[0];
              ir2[0] = (v310_data + (v306_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v316_data = ir2[1];
              ir2[1] = (v316_data + (v306_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v322_data = ir2[2];
              ir2[2] = (v322_data + (v306_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v328_data = ir2[3];
              ir2[3] = (v328_data + (v306_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v334_data = ir2[4];
              ir2[4] = (v334_data + (v306_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v340_data = ir2[5];
              ir2[5] = (v340_data + (v306_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v346_data = ir2[6];
              ir2[6] = (v346_data + (v306_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v352_data = ir2[7];
              ir2[7] = (v352_data + (v306_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v358_data = ir2[8];
              ir2[8] = (v358_data + (v306_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v360_data = r0[6];
              float v364_data = ir2[0];
              ir2[0] = (v364_data + (v360_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v370_data = ir2[1];
              ir2[1] = (v370_data + (v360_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v376_data = ir2[2];
              ir2[2] = (v376_data + (v360_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v382_data = ir2[3];
              ir2[3] = (v382_data + (v360_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v388_data = ir2[4];
              ir2[4] = (v388_data + (v360_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v394_data = ir2[5];
              ir2[5] = (v394_data + (v360_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v400_data = ir2[6];
              ir2[6] = (v400_data + (v360_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v406_data = ir2[7];
              ir2[7] = (v406_data + (v360_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v412_data = ir2[8];
              ir2[8] = (v412_data + (v360_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v414_data = r0[7];
              float v418_data = ir2[0];
              ir2[0] = (v418_data + (v414_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v424_data = ir2[1];
              ir2[1] = (v424_data + (v414_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v430_data = ir2[2];
              ir2[2] = (v430_data + (v414_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v436_data = ir2[3];
              ir2[3] = (v436_data + (v414_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v442_data = ir2[4];
              ir2[4] = (v442_data + (v414_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v448_data = ir2[5];
              ir2[5] = (v448_data + (v414_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v454_data = ir2[6];
              ir2[6] = (v454_data + (v414_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v460_data = ir2[7];
              ir2[7] = (v460_data + (v414_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v466_data = ir2[8];
              ir2[8] = (v466_data + (v414_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v468_data = r0[8];
              float v472_data = ir2[0];
              ir2[0] = (v472_data + (v468_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v478_data = ir2[1];
              ir2[1] = (v478_data + (v468_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v484_data = ir2[2];
              ir2[2] = (v484_data + (v468_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v490_data = ir2[3];
              ir2[3] = (v490_data + (v468_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v496_data = ir2[4];
              ir2[4] = (v496_data + (v468_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v502_data = ir2[5];
              ir2[5] = (v502_data + (v468_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v508_data = ir2[6];
              ir2[6] = (v508_data + (v468_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v514_data = ir2[7];
              ir2[7] = (v514_data + (v468_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v520_data = ir2[8];
              ir2[8] = (v520_data + (v468_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              if (v18_g) {
                #pragma unroll
                for (int32_t v523_n1 = 0; v523_n1 < 9; ++v523_n1) {
                  float v525_data = ir2[v523_n1];
                  r2[v523_n1] = (v525_data * 13.0f);
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v18_g) {
                #pragma unroll
                for (int32_t v527_i1 = 0; v527_i1 < 9; ++v527_i1) {
                  float v529_data = r2[v527_i1];
                  glb_m0[(v17_lead + (v527_i1 * 9))] = v529_data;
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

