// === base name ===
kernel_c6b1bc9dbb71c171

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_c6b1bc9dbb71c171 = {{16, 16, 1}, 16, 16, 1, 16, 5120, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_c6b1bc9dbb71c171(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_c6b1bc9dbb71c171(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_c6b1bc9dbb71c171(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_c6b1bc9dbb71c171(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_c6b1bc9dbb71c171(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_c6b1bc9dbb71c171(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_c6b1bc9dbb71c171(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 5120 B shared, occupancy grid
        // operands:
        //   m0 8×8(8×8) {0..8}×{0..8} strided
        //   m1 8×8(8×8) {0..8}×{0..8} strided
        //   m2 8×8(8×8) {0..8}×{0..8} strided
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   C = abs(TMP)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":1280}],"shared_bytes":5120,"shared_elements":1280,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1\n"}
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
              const float *const __restrict__ glb_m1 = &m1[v4_batchId0 * 64 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v4_batchId0 * 64 + 0 + m2_extraOffset];
              float r0[8]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v18_lead = item.get_local_id(2) % 16;
              bool v19_g = v18_lead < 8;
              if (v19_g) {
                #pragma unroll
                for (int32_t v20_i1 = 0; v20_i1 < 8; ++v20_i1) {
                  float v25_data = glb_m0[(v18_lead + (v20_i1 * 8))];
                  r0[v20_i1] = v25_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m1);
              if (v19_g) {
                #pragma unroll
                for (int32_t v28_i1 = 0; v28_i1 < 8; ++v28_i1) {
                  float v33_data = glb_m1[(v18_lead + (v28_i1 * 8))];
                  r1[v28_i1] = v33_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              // wait(r1 = load{g>r}(glb_m1););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              float v36_data = r0[0];
              float v37_data = r1[0];
              float v40_data = r2[0];
              r2[0] = (v40_data + (v36_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v43_data = r1[1];
              float v46_data = r2[1];
              r2[1] = (v46_data + (v36_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v49_data = r1[2];
              float v52_data = r2[2];
              r2[2] = (v52_data + (v36_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v55_data = r1[3];
              float v58_data = r2[3];
              r2[3] = (v58_data + (v36_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v61_data = r1[4];
              float v64_data = r2[4];
              r2[4] = (v64_data + (v36_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v67_data = r1[5];
              float v70_data = r2[5];
              r2[5] = (v70_data + (v36_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v73_data = r1[6];
              float v76_data = r2[6];
              r2[6] = (v76_data + (v36_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v79_data = r1[7];
              float v82_data = r2[7];
              r2[7] = (v82_data + (v36_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v84_data = r0[1];
              float v88_data = r2[0];
              r2[0] = (v88_data + (v84_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v94_data = r2[1];
              r2[1] = (v94_data + (v84_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v100_data = r2[2];
              r2[2] = (v100_data + (v84_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v106_data = r2[3];
              r2[3] = (v106_data + (v84_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v112_data = r2[4];
              r2[4] = (v112_data + (v84_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v118_data = r2[5];
              r2[5] = (v118_data + (v84_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v124_data = r2[6];
              r2[6] = (v124_data + (v84_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v130_data = r2[7];
              r2[7] = (v130_data + (v84_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v132_data = r0[2];
              float v136_data = r2[0];
              r2[0] = (v136_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v142_data = r2[1];
              r2[1] = (v142_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v148_data = r2[2];
              r2[2] = (v148_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v154_data = r2[3];
              r2[3] = (v154_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v160_data = r2[4];
              r2[4] = (v160_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v166_data = r2[5];
              r2[5] = (v166_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v172_data = r2[6];
              r2[6] = (v172_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v178_data = r2[7];
              r2[7] = (v178_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v180_data = r0[3];
              float v184_data = r2[0];
              r2[0] = (v184_data + (v180_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v190_data = r2[1];
              r2[1] = (v190_data + (v180_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v196_data = r2[2];
              r2[2] = (v196_data + (v180_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v202_data = r2[3];
              r2[3] = (v202_data + (v180_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v208_data = r2[4];
              r2[4] = (v208_data + (v180_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v214_data = r2[5];
              r2[5] = (v214_data + (v180_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v220_data = r2[6];
              r2[6] = (v220_data + (v180_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v226_data = r2[7];
              r2[7] = (v226_data + (v180_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v228_data = r0[4];
              float v232_data = r2[0];
              r2[0] = (v232_data + (v228_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v238_data = r2[1];
              r2[1] = (v238_data + (v228_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v244_data = r2[2];
              r2[2] = (v244_data + (v228_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v250_data = r2[3];
              r2[3] = (v250_data + (v228_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v256_data = r2[4];
              r2[4] = (v256_data + (v228_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v262_data = r2[5];
              r2[5] = (v262_data + (v228_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v268_data = r2[6];
              r2[6] = (v268_data + (v228_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v274_data = r2[7];
              r2[7] = (v274_data + (v228_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v276_data = r0[5];
              float v280_data = r2[0];
              r2[0] = (v280_data + (v276_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v286_data = r2[1];
              r2[1] = (v286_data + (v276_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v292_data = r2[2];
              r2[2] = (v292_data + (v276_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v298_data = r2[3];
              r2[3] = (v298_data + (v276_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v304_data = r2[4];
              r2[4] = (v304_data + (v276_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v310_data = r2[5];
              r2[5] = (v310_data + (v276_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v316_data = r2[6];
              r2[6] = (v316_data + (v276_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v322_data = r2[7];
              r2[7] = (v322_data + (v276_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v324_data = r0[6];
              float v328_data = r2[0];
              r2[0] = (v328_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v334_data = r2[1];
              r2[1] = (v334_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v340_data = r2[2];
              r2[2] = (v340_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v346_data = r2[3];
              r2[3] = (v346_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v352_data = r2[4];
              r2[4] = (v352_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v358_data = r2[5];
              r2[5] = (v358_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v364_data = r2[6];
              r2[6] = (v364_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v370_data = r2[7];
              r2[7] = (v370_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v372_data = r0[7];
              float v376_data = r2[0];
              r2[0] = (v376_data + (v372_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v382_data = r2[1];
              r2[1] = (v382_data + (v372_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v388_data = r2[2];
              r2[2] = (v388_data + (v372_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v394_data = r2[3];
              r2[3] = (v394_data + (v372_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v400_data = r2[4];
              r2[4] = (v400_data + (v372_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v406_data = r2[5];
              r2[5] = (v406_data + (v372_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v412_data = r2[6];
              r2[6] = (v412_data + (v372_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v418_data = r2[7];
              r2[7] = (v418_data + (v372_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              // s0 = store{r>s}(localShrMem0, r2);
              if (v19_g) {
                #pragma unroll
                for (int32_t v420_i1 = 0; v420_i1 < 8; ++v420_i1) {
                  float v422_data = r2[v420_i1];
                  int32_t v426_a = v18_lead + (v420_i1 * 8);
                  s0[(v426_a ^ ((v426_a >> 5) & 31))] = v422_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
              // glb_m2 = abs(s0)
              if (v19_g) {
                #pragma unroll
                for (int32_t v430_k1 = 0; v430_k1 < 8; ++v430_k1) {
                  int32_t v434_a = v18_lead + (v430_k1 * 8);
                  float v438_data = s0[(v434_a ^ ((v434_a >> 5) & 31))];
                  glb_m2[v434_a] = (sycl::fabs(v438_data));
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

