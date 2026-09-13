// === base name ===
kernel_4f384ef036093cbc

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_4f384ef036093cbc = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_4f384ef036093cbc(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_4f384ef036093cbc(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_4f384ef036093cbc(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_4f384ef036093cbc(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_4f384ef036093cbc(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_4f384ef036093cbc(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_4f384ef036093cbc(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 16×16(16×16) {0..16}×{0..16} strided
        //   m1 16×16(16×16) {0..16}×{0..16} strided
        //   m2 16×16(16×16) {0..16}×{0..16} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          int32_t v16_lead = item.get_local_id(2) % 16;
          for (size_t v3_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v3_batchId0 < numElements0; v3_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v4_ahead1 = v3_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
            float *const __restrict__ glb_m0 = &m0[v3_batchId0 * 256 + 0 + m0_extraOffset];
            const float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 256 + 0 + m1_extraOffset];
            const float *const __restrict__ glb_m2 = &m2[v3_batchId0 * 256 + 0 + m2_extraOffset];
            float r0[16]{};
            // r0 = load{g>r}(glb_m1);
            #pragma unroll
            for (int32_t v17_i0 = 0; v17_i0 < 1; ++v17_i0) {
              int32_t v20_lead = v16_lead + (v17_i0 * 16);
              #pragma unroll
              for (int32_t v18_i1 = 0; v18_i1 < 16; ++v18_i1) {
                float v23_data = glb_m1[(v20_lead + (v18_i1 * 16))];
                r0[(v17_i0 + v18_i1)] = v23_data;
              }
            }
            float r1[16]{};
            // r1 = load{g>r}(glb_m2);
            #pragma unroll
            for (int32_t v26_i0 = 0; v26_i0 < 1; ++v26_i0) {
              int32_t v29_lead = v16_lead + (v26_i0 * 16);
              #pragma unroll
              for (int32_t v27_i1 = 0; v27_i1 < 16; ++v27_i1) {
                float v32_data = glb_m2[(v29_lead + (v27_i1 * 16))];
                r1[(v26_i0 + v27_i1)] = v32_data;
              }
            }
            // wait(r0 = load{g>r}(glb_m1););
            // wait(r1 = load{g>r}(glb_m2););
            float r2[16]{};
            // r2 = +(r0 * r1) + None
            // [(0, 16), (0, 16)] [(0, 16)]
            float ir2[16]{};
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
            float v91_data = r1[9];
            float v94_data = ir2[9];
            ir2[9] = (v94_data + (v36_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v97_data = r1[10];
            float v100_data = ir2[10];
            ir2[10] = (v100_data + (v36_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v103_data = r1[11];
            float v106_data = ir2[11];
            ir2[11] = (v106_data + (v36_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v109_data = r1[12];
            float v112_data = ir2[12];
            ir2[12] = (v112_data + (v36_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v115_data = r1[13];
            float v118_data = ir2[13];
            ir2[13] = (v118_data + (v36_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v121_data = r1[14];
            float v124_data = ir2[14];
            ir2[14] = (v124_data + (v36_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v127_data = r1[15];
            float v130_data = ir2[15];
            ir2[15] = (v130_data + (v36_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v132_data = r0[1];
            float v136_data = ir2[0];
            ir2[0] = (v136_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v142_data = ir2[1];
            ir2[1] = (v142_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v148_data = ir2[2];
            ir2[2] = (v148_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v154_data = ir2[3];
            ir2[3] = (v154_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v160_data = ir2[4];
            ir2[4] = (v160_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v166_data = ir2[5];
            ir2[5] = (v166_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v172_data = ir2[6];
            ir2[6] = (v172_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v178_data = ir2[7];
            ir2[7] = (v178_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v184_data = ir2[8];
            ir2[8] = (v184_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v190_data = ir2[9];
            ir2[9] = (v190_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v196_data = ir2[10];
            ir2[10] = (v196_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v202_data = ir2[11];
            ir2[11] = (v202_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v208_data = ir2[12];
            ir2[12] = (v208_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v214_data = ir2[13];
            ir2[13] = (v214_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v220_data = ir2[14];
            ir2[14] = (v220_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v226_data = ir2[15];
            ir2[15] = (v226_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v228_data = r0[2];
            float v232_data = ir2[0];
            ir2[0] = (v232_data + (v228_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v238_data = ir2[1];
            ir2[1] = (v238_data + (v228_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v244_data = ir2[2];
            ir2[2] = (v244_data + (v228_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v250_data = ir2[3];
            ir2[3] = (v250_data + (v228_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v256_data = ir2[4];
            ir2[4] = (v256_data + (v228_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v262_data = ir2[5];
            ir2[5] = (v262_data + (v228_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v268_data = ir2[6];
            ir2[6] = (v268_data + (v228_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v274_data = ir2[7];
            ir2[7] = (v274_data + (v228_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v280_data = ir2[8];
            ir2[8] = (v280_data + (v228_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v286_data = ir2[9];
            ir2[9] = (v286_data + (v228_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v292_data = ir2[10];
            ir2[10] = (v292_data + (v228_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v298_data = ir2[11];
            ir2[11] = (v298_data + (v228_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v304_data = ir2[12];
            ir2[12] = (v304_data + (v228_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v310_data = ir2[13];
            ir2[13] = (v310_data + (v228_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v316_data = ir2[14];
            ir2[14] = (v316_data + (v228_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v322_data = ir2[15];
            ir2[15] = (v322_data + (v228_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v324_data = r0[3];
            float v328_data = ir2[0];
            ir2[0] = (v328_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v334_data = ir2[1];
            ir2[1] = (v334_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v340_data = ir2[2];
            ir2[2] = (v340_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v346_data = ir2[3];
            ir2[3] = (v346_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v352_data = ir2[4];
            ir2[4] = (v352_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v358_data = ir2[5];
            ir2[5] = (v358_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v364_data = ir2[6];
            ir2[6] = (v364_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v370_data = ir2[7];
            ir2[7] = (v370_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v376_data = ir2[8];
            ir2[8] = (v376_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v382_data = ir2[9];
            ir2[9] = (v382_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v388_data = ir2[10];
            ir2[10] = (v388_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v394_data = ir2[11];
            ir2[11] = (v394_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v400_data = ir2[12];
            ir2[12] = (v400_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v406_data = ir2[13];
            ir2[13] = (v406_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v412_data = ir2[14];
            ir2[14] = (v412_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v418_data = ir2[15];
            ir2[15] = (v418_data + (v324_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v420_data = r0[4];
            float v424_data = ir2[0];
            ir2[0] = (v424_data + (v420_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v430_data = ir2[1];
            ir2[1] = (v430_data + (v420_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v436_data = ir2[2];
            ir2[2] = (v436_data + (v420_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v442_data = ir2[3];
            ir2[3] = (v442_data + (v420_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v448_data = ir2[4];
            ir2[4] = (v448_data + (v420_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v454_data = ir2[5];
            ir2[5] = (v454_data + (v420_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v460_data = ir2[6];
            ir2[6] = (v460_data + (v420_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v466_data = ir2[7];
            ir2[7] = (v466_data + (v420_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v472_data = ir2[8];
            ir2[8] = (v472_data + (v420_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v478_data = ir2[9];
            ir2[9] = (v478_data + (v420_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v484_data = ir2[10];
            ir2[10] = (v484_data + (v420_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v490_data = ir2[11];
            ir2[11] = (v490_data + (v420_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v496_data = ir2[12];
            ir2[12] = (v496_data + (v420_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v502_data = ir2[13];
            ir2[13] = (v502_data + (v420_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v508_data = ir2[14];
            ir2[14] = (v508_data + (v420_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v514_data = ir2[15];
            ir2[15] = (v514_data + (v420_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v516_data = r0[5];
            float v520_data = ir2[0];
            ir2[0] = (v520_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v526_data = ir2[1];
            ir2[1] = (v526_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v532_data = ir2[2];
            ir2[2] = (v532_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v538_data = ir2[3];
            ir2[3] = (v538_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v544_data = ir2[4];
            ir2[4] = (v544_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v550_data = ir2[5];
            ir2[5] = (v550_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v556_data = ir2[6];
            ir2[6] = (v556_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v562_data = ir2[7];
            ir2[7] = (v562_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v568_data = ir2[8];
            ir2[8] = (v568_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v574_data = ir2[9];
            ir2[9] = (v574_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v580_data = ir2[10];
            ir2[10] = (v580_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v586_data = ir2[11];
            ir2[11] = (v586_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v592_data = ir2[12];
            ir2[12] = (v592_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v598_data = ir2[13];
            ir2[13] = (v598_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v604_data = ir2[14];
            ir2[14] = (v604_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v610_data = ir2[15];
            ir2[15] = (v610_data + (v516_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v612_data = r0[6];
            float v616_data = ir2[0];
            ir2[0] = (v616_data + (v612_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v622_data = ir2[1];
            ir2[1] = (v622_data + (v612_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v628_data = ir2[2];
            ir2[2] = (v628_data + (v612_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v634_data = ir2[3];
            ir2[3] = (v634_data + (v612_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v640_data = ir2[4];
            ir2[4] = (v640_data + (v612_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v646_data = ir2[5];
            ir2[5] = (v646_data + (v612_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v652_data = ir2[6];
            ir2[6] = (v652_data + (v612_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v658_data = ir2[7];
            ir2[7] = (v658_data + (v612_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v664_data = ir2[8];
            ir2[8] = (v664_data + (v612_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v670_data = ir2[9];
            ir2[9] = (v670_data + (v612_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v676_data = ir2[10];
            ir2[10] = (v676_data + (v612_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v682_data = ir2[11];
            ir2[11] = (v682_data + (v612_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v688_data = ir2[12];
            ir2[12] = (v688_data + (v612_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v694_data = ir2[13];
            ir2[13] = (v694_data + (v612_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v700_data = ir2[14];
            ir2[14] = (v700_data + (v612_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v706_data = ir2[15];
            ir2[15] = (v706_data + (v612_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v708_data = r0[7];
            float v712_data = ir2[0];
            ir2[0] = (v712_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v718_data = ir2[1];
            ir2[1] = (v718_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v724_data = ir2[2];
            ir2[2] = (v724_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v730_data = ir2[3];
            ir2[3] = (v730_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v736_data = ir2[4];
            ir2[4] = (v736_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v742_data = ir2[5];
            ir2[5] = (v742_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v748_data = ir2[6];
            ir2[6] = (v748_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v754_data = ir2[7];
            ir2[7] = (v754_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v760_data = ir2[8];
            ir2[8] = (v760_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v766_data = ir2[9];
            ir2[9] = (v766_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v772_data = ir2[10];
            ir2[10] = (v772_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v778_data = ir2[11];
            ir2[11] = (v778_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v784_data = ir2[12];
            ir2[12] = (v784_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v790_data = ir2[13];
            ir2[13] = (v790_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v796_data = ir2[14];
            ir2[14] = (v796_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v802_data = ir2[15];
            ir2[15] = (v802_data + (v708_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v804_data = r0[8];
            float v808_data = ir2[0];
            ir2[0] = (v808_data + (v804_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v814_data = ir2[1];
            ir2[1] = (v814_data + (v804_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v820_data = ir2[2];
            ir2[2] = (v820_data + (v804_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v826_data = ir2[3];
            ir2[3] = (v826_data + (v804_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v832_data = ir2[4];
            ir2[4] = (v832_data + (v804_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v838_data = ir2[5];
            ir2[5] = (v838_data + (v804_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v844_data = ir2[6];
            ir2[6] = (v844_data + (v804_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v850_data = ir2[7];
            ir2[7] = (v850_data + (v804_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v856_data = ir2[8];
            ir2[8] = (v856_data + (v804_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v862_data = ir2[9];
            ir2[9] = (v862_data + (v804_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v868_data = ir2[10];
            ir2[10] = (v868_data + (v804_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v874_data = ir2[11];
            ir2[11] = (v874_data + (v804_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v880_data = ir2[12];
            ir2[12] = (v880_data + (v804_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v886_data = ir2[13];
            ir2[13] = (v886_data + (v804_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v892_data = ir2[14];
            ir2[14] = (v892_data + (v804_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v898_data = ir2[15];
            ir2[15] = (v898_data + (v804_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v900_data = r0[9];
            float v904_data = ir2[0];
            ir2[0] = (v904_data + (v900_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v910_data = ir2[1];
            ir2[1] = (v910_data + (v900_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v916_data = ir2[2];
            ir2[2] = (v916_data + (v900_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v922_data = ir2[3];
            ir2[3] = (v922_data + (v900_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v928_data = ir2[4];
            ir2[4] = (v928_data + (v900_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v934_data = ir2[5];
            ir2[5] = (v934_data + (v900_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v940_data = ir2[6];
            ir2[6] = (v940_data + (v900_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v946_data = ir2[7];
            ir2[7] = (v946_data + (v900_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v952_data = ir2[8];
            ir2[8] = (v952_data + (v900_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v958_data = ir2[9];
            ir2[9] = (v958_data + (v900_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v964_data = ir2[10];
            ir2[10] = (v964_data + (v900_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v970_data = ir2[11];
            ir2[11] = (v970_data + (v900_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v976_data = ir2[12];
            ir2[12] = (v976_data + (v900_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v982_data = ir2[13];
            ir2[13] = (v982_data + (v900_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v988_data = ir2[14];
            ir2[14] = (v988_data + (v900_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v994_data = ir2[15];
            ir2[15] = (v994_data + (v900_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v996_data = r0[10];
            float v1000_data = ir2[0];
            ir2[0] = (v1000_data + (v996_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1006_data = ir2[1];
            ir2[1] = (v1006_data + (v996_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1012_data = ir2[2];
            ir2[2] = (v1012_data + (v996_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1018_data = ir2[3];
            ir2[3] = (v1018_data + (v996_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1024_data = ir2[4];
            ir2[4] = (v1024_data + (v996_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1030_data = ir2[5];
            ir2[5] = (v1030_data + (v996_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1036_data = ir2[6];
            ir2[6] = (v1036_data + (v996_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1042_data = ir2[7];
            ir2[7] = (v1042_data + (v996_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1048_data = ir2[8];
            ir2[8] = (v1048_data + (v996_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1054_data = ir2[9];
            ir2[9] = (v1054_data + (v996_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1060_data = ir2[10];
            ir2[10] = (v1060_data + (v996_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1066_data = ir2[11];
            ir2[11] = (v1066_data + (v996_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1072_data = ir2[12];
            ir2[12] = (v1072_data + (v996_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1078_data = ir2[13];
            ir2[13] = (v1078_data + (v996_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1084_data = ir2[14];
            ir2[14] = (v1084_data + (v996_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1090_data = ir2[15];
            ir2[15] = (v1090_data + (v996_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1092_data = r0[11];
            float v1096_data = ir2[0];
            ir2[0] = (v1096_data + (v1092_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1102_data = ir2[1];
            ir2[1] = (v1102_data + (v1092_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1108_data = ir2[2];
            ir2[2] = (v1108_data + (v1092_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1114_data = ir2[3];
            ir2[3] = (v1114_data + (v1092_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1120_data = ir2[4];
            ir2[4] = (v1120_data + (v1092_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1126_data = ir2[5];
            ir2[5] = (v1126_data + (v1092_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1132_data = ir2[6];
            ir2[6] = (v1132_data + (v1092_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1138_data = ir2[7];
            ir2[7] = (v1138_data + (v1092_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1144_data = ir2[8];
            ir2[8] = (v1144_data + (v1092_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1150_data = ir2[9];
            ir2[9] = (v1150_data + (v1092_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1156_data = ir2[10];
            ir2[10] = (v1156_data + (v1092_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1162_data = ir2[11];
            ir2[11] = (v1162_data + (v1092_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1168_data = ir2[12];
            ir2[12] = (v1168_data + (v1092_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1174_data = ir2[13];
            ir2[13] = (v1174_data + (v1092_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1180_data = ir2[14];
            ir2[14] = (v1180_data + (v1092_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1186_data = ir2[15];
            ir2[15] = (v1186_data + (v1092_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1188_data = r0[12];
            float v1192_data = ir2[0];
            ir2[0] = (v1192_data + (v1188_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1198_data = ir2[1];
            ir2[1] = (v1198_data + (v1188_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1204_data = ir2[2];
            ir2[2] = (v1204_data + (v1188_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1210_data = ir2[3];
            ir2[3] = (v1210_data + (v1188_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1216_data = ir2[4];
            ir2[4] = (v1216_data + (v1188_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1222_data = ir2[5];
            ir2[5] = (v1222_data + (v1188_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1228_data = ir2[6];
            ir2[6] = (v1228_data + (v1188_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1234_data = ir2[7];
            ir2[7] = (v1234_data + (v1188_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1240_data = ir2[8];
            ir2[8] = (v1240_data + (v1188_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1246_data = ir2[9];
            ir2[9] = (v1246_data + (v1188_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1252_data = ir2[10];
            ir2[10] = (v1252_data + (v1188_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1258_data = ir2[11];
            ir2[11] = (v1258_data + (v1188_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1264_data = ir2[12];
            ir2[12] = (v1264_data + (v1188_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1270_data = ir2[13];
            ir2[13] = (v1270_data + (v1188_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1276_data = ir2[14];
            ir2[14] = (v1276_data + (v1188_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1282_data = ir2[15];
            ir2[15] = (v1282_data + (v1188_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1284_data = r0[13];
            float v1288_data = ir2[0];
            ir2[0] = (v1288_data + (v1284_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1294_data = ir2[1];
            ir2[1] = (v1294_data + (v1284_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1300_data = ir2[2];
            ir2[2] = (v1300_data + (v1284_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1306_data = ir2[3];
            ir2[3] = (v1306_data + (v1284_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1312_data = ir2[4];
            ir2[4] = (v1312_data + (v1284_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1318_data = ir2[5];
            ir2[5] = (v1318_data + (v1284_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1324_data = ir2[6];
            ir2[6] = (v1324_data + (v1284_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1330_data = ir2[7];
            ir2[7] = (v1330_data + (v1284_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1336_data = ir2[8];
            ir2[8] = (v1336_data + (v1284_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1342_data = ir2[9];
            ir2[9] = (v1342_data + (v1284_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1348_data = ir2[10];
            ir2[10] = (v1348_data + (v1284_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1354_data = ir2[11];
            ir2[11] = (v1354_data + (v1284_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1360_data = ir2[12];
            ir2[12] = (v1360_data + (v1284_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1366_data = ir2[13];
            ir2[13] = (v1366_data + (v1284_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1372_data = ir2[14];
            ir2[14] = (v1372_data + (v1284_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1378_data = ir2[15];
            ir2[15] = (v1378_data + (v1284_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1380_data = r0[14];
            float v1384_data = ir2[0];
            ir2[0] = (v1384_data + (v1380_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1390_data = ir2[1];
            ir2[1] = (v1390_data + (v1380_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1396_data = ir2[2];
            ir2[2] = (v1396_data + (v1380_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1402_data = ir2[3];
            ir2[3] = (v1402_data + (v1380_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1408_data = ir2[4];
            ir2[4] = (v1408_data + (v1380_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1414_data = ir2[5];
            ir2[5] = (v1414_data + (v1380_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1420_data = ir2[6];
            ir2[6] = (v1420_data + (v1380_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1426_data = ir2[7];
            ir2[7] = (v1426_data + (v1380_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1432_data = ir2[8];
            ir2[8] = (v1432_data + (v1380_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1438_data = ir2[9];
            ir2[9] = (v1438_data + (v1380_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1444_data = ir2[10];
            ir2[10] = (v1444_data + (v1380_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1450_data = ir2[11];
            ir2[11] = (v1450_data + (v1380_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1456_data = ir2[12];
            ir2[12] = (v1456_data + (v1380_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1462_data = ir2[13];
            ir2[13] = (v1462_data + (v1380_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1468_data = ir2[14];
            ir2[14] = (v1468_data + (v1380_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1474_data = ir2[15];
            ir2[15] = (v1474_data + (v1380_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1476_data = r0[15];
            float v1480_data = ir2[0];
            ir2[0] = (v1480_data + (v1476_data * (sycl::select_from_group(item.get_sub_group(), v37_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1486_data = ir2[1];
            ir2[1] = (v1486_data + (v1476_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1492_data = ir2[2];
            ir2[2] = (v1492_data + (v1476_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1498_data = ir2[3];
            ir2[3] = (v1498_data + (v1476_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1504_data = ir2[4];
            ir2[4] = (v1504_data + (v1476_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1510_data = ir2[5];
            ir2[5] = (v1510_data + (v1476_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1516_data = ir2[6];
            ir2[6] = (v1516_data + (v1476_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1522_data = ir2[7];
            ir2[7] = (v1522_data + (v1476_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1528_data = ir2[8];
            ir2[8] = (v1528_data + (v1476_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1534_data = ir2[9];
            ir2[9] = (v1534_data + (v1476_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1540_data = ir2[10];
            ir2[10] = (v1540_data + (v1476_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1546_data = ir2[11];
            ir2[11] = (v1546_data + (v1476_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1552_data = ir2[12];
            ir2[12] = (v1552_data + (v1476_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1558_data = ir2[13];
            ir2[13] = (v1558_data + (v1476_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1564_data = ir2[14];
            ir2[14] = (v1564_data + (v1476_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1570_data = ir2[15];
            ir2[15] = (v1570_data + (v1476_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            #pragma unroll
            for (int32_t v1572_n0 = 0; v1572_n0 < 1; ++v1572_n0) {
              #pragma unroll
              for (int32_t v1573_n1 = 0; v1573_n1 < 16; ++v1573_n1) {
                int32_t v1574_a = v1572_n0 + v1573_n1;
                float v1575_data = ir2[v1574_a];
                r2[v1574_a] = v1575_data;
              }
            }
            // glb_m0 = store{r>g}(r2);
            #pragma unroll
            for (int32_t v1576_i0 = 0; v1576_i0 < 1; ++v1576_i0) {
              int32_t v1581_lead = v16_lead + (v1576_i0 * 16);
              #pragma unroll
              for (int32_t v1577_i1 = 0; v1577_i1 < 16; ++v1577_i1) {
                float v1579_data = r2[(v1576_i0 + v1577_i1)];
                glb_m0[(v1581_lead + (v1577_i1 * 16))] = v1579_data;
              }
            }
            sycl::group_barrier(item.get_sub_group());
          }
        }
      });
    }
  });
}

