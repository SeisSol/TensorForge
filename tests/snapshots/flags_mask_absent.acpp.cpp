// === base name ===
kernel_68c475d5907309e1

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_68c475d5907309e1 = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_68c475d5907309e1(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_68c475d5907309e1(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_68c475d5907309e1(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_68c475d5907309e1(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_68c475d5907309e1(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_68c475d5907309e1(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_68c475d5907309e1(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 16×16(16×16) {0..16}×{0..16} strided
        //   m1 16×16(16×16) {0..16}×{0..16} strided
        //   m2 16×16(16×16) {0..16}×{0..16} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          int32_t v22_lead = item.get_local_id(2) % 16;
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 256 + 0 + m0_extraOffset];
            const float *const __restrict__ glb_m1 = &m1[v9_batchId0 * 256 + 0 + m1_extraOffset];
            const float *const __restrict__ glb_m2 = &m2[v9_batchId0 * 256 + 0 + m2_extraOffset];
            float r0[16]{};
            // r0 = load{g>r}(glb_m1);
            #pragma unroll
            for (int32_t v23_i0 = 0; v23_i0 < 1; ++v23_i0) {
              int32_t v26_lead = v22_lead + (v23_i0 * 16);
              #pragma unroll
              for (int32_t v24_i1 = 0; v24_i1 < 16; ++v24_i1) {
                float v29_data = glb_m1[(v26_lead + (v24_i1 * 16))];
                r0[(v23_i0 + v24_i1)] = v29_data;
              }
            }
            float r1[16]{};
            // r1 = load{g>r}(glb_m2);
            #pragma unroll
            for (int32_t v32_i0 = 0; v32_i0 < 1; ++v32_i0) {
              int32_t v35_lead = v22_lead + (v32_i0 * 16);
              #pragma unroll
              for (int32_t v33_i1 = 0; v33_i1 < 16; ++v33_i1) {
                float v38_data = glb_m2[(v35_lead + (v33_i1 * 16))];
                r1[(v32_i0 + v33_i1)] = v38_data;
              }
            }
            // wait(r0 = load{g>r}(glb_m1););
            // wait(r1 = load{g>r}(glb_m2););
            float r2[16]{};
            // ir2 = +(r0 * r1)
            // [(0, 16), (0, 16)] [(0, 16)]
            float ir2[16]{};
            float v42_data = r0[0];
            float v43_data = r1[0];
            float v46_data = ir2[0];
            ir2[0] = (v46_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v49_data = r1[1];
            float v52_data = ir2[1];
            ir2[1] = (v52_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v55_data = r1[2];
            float v58_data = ir2[2];
            ir2[2] = (v58_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v61_data = r1[3];
            float v64_data = ir2[3];
            ir2[3] = (v64_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v67_data = r1[4];
            float v70_data = ir2[4];
            ir2[4] = (v70_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v73_data = r1[5];
            float v76_data = ir2[5];
            ir2[5] = (v76_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v79_data = r1[6];
            float v82_data = ir2[6];
            ir2[6] = (v82_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v85_data = r1[7];
            float v88_data = ir2[7];
            ir2[7] = (v88_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v91_data = r1[8];
            float v94_data = ir2[8];
            ir2[8] = (v94_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v97_data = r1[9];
            float v100_data = ir2[9];
            ir2[9] = (v100_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v103_data = r1[10];
            float v106_data = ir2[10];
            ir2[10] = (v106_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v109_data = r1[11];
            float v112_data = ir2[11];
            ir2[11] = (v112_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v115_data = r1[12];
            float v118_data = ir2[12];
            ir2[12] = (v118_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v121_data = r1[13];
            float v124_data = ir2[13];
            ir2[13] = (v124_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v127_data = r1[14];
            float v130_data = ir2[14];
            ir2[14] = (v130_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v133_data = r1[15];
            float v136_data = ir2[15];
            ir2[15] = (v136_data + (v42_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v138_data = r0[1];
            float v142_data = ir2[0];
            ir2[0] = (v142_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v148_data = ir2[1];
            ir2[1] = (v148_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v154_data = ir2[2];
            ir2[2] = (v154_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v160_data = ir2[3];
            ir2[3] = (v160_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v166_data = ir2[4];
            ir2[4] = (v166_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v172_data = ir2[5];
            ir2[5] = (v172_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v178_data = ir2[6];
            ir2[6] = (v178_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v184_data = ir2[7];
            ir2[7] = (v184_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v190_data = ir2[8];
            ir2[8] = (v190_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v196_data = ir2[9];
            ir2[9] = (v196_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v202_data = ir2[10];
            ir2[10] = (v202_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v208_data = ir2[11];
            ir2[11] = (v208_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v214_data = ir2[12];
            ir2[12] = (v214_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v220_data = ir2[13];
            ir2[13] = (v220_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v226_data = ir2[14];
            ir2[14] = (v226_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v232_data = ir2[15];
            ir2[15] = (v232_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v234_data = r0[2];
            float v238_data = ir2[0];
            ir2[0] = (v238_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v244_data = ir2[1];
            ir2[1] = (v244_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v250_data = ir2[2];
            ir2[2] = (v250_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v256_data = ir2[3];
            ir2[3] = (v256_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v262_data = ir2[4];
            ir2[4] = (v262_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v268_data = ir2[5];
            ir2[5] = (v268_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v274_data = ir2[6];
            ir2[6] = (v274_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v280_data = ir2[7];
            ir2[7] = (v280_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v286_data = ir2[8];
            ir2[8] = (v286_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v292_data = ir2[9];
            ir2[9] = (v292_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v298_data = ir2[10];
            ir2[10] = (v298_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v304_data = ir2[11];
            ir2[11] = (v304_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v310_data = ir2[12];
            ir2[12] = (v310_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v316_data = ir2[13];
            ir2[13] = (v316_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v322_data = ir2[14];
            ir2[14] = (v322_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v328_data = ir2[15];
            ir2[15] = (v328_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v330_data = r0[3];
            float v334_data = ir2[0];
            ir2[0] = (v334_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v340_data = ir2[1];
            ir2[1] = (v340_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v346_data = ir2[2];
            ir2[2] = (v346_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v352_data = ir2[3];
            ir2[3] = (v352_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v358_data = ir2[4];
            ir2[4] = (v358_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v364_data = ir2[5];
            ir2[5] = (v364_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v370_data = ir2[6];
            ir2[6] = (v370_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v376_data = ir2[7];
            ir2[7] = (v376_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v382_data = ir2[8];
            ir2[8] = (v382_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v388_data = ir2[9];
            ir2[9] = (v388_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v394_data = ir2[10];
            ir2[10] = (v394_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v400_data = ir2[11];
            ir2[11] = (v400_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v406_data = ir2[12];
            ir2[12] = (v406_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v412_data = ir2[13];
            ir2[13] = (v412_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v418_data = ir2[14];
            ir2[14] = (v418_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v424_data = ir2[15];
            ir2[15] = (v424_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v426_data = r0[4];
            float v430_data = ir2[0];
            ir2[0] = (v430_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v436_data = ir2[1];
            ir2[1] = (v436_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v442_data = ir2[2];
            ir2[2] = (v442_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v448_data = ir2[3];
            ir2[3] = (v448_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v454_data = ir2[4];
            ir2[4] = (v454_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v460_data = ir2[5];
            ir2[5] = (v460_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v466_data = ir2[6];
            ir2[6] = (v466_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v472_data = ir2[7];
            ir2[7] = (v472_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v478_data = ir2[8];
            ir2[8] = (v478_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v484_data = ir2[9];
            ir2[9] = (v484_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v490_data = ir2[10];
            ir2[10] = (v490_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v496_data = ir2[11];
            ir2[11] = (v496_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v502_data = ir2[12];
            ir2[12] = (v502_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v508_data = ir2[13];
            ir2[13] = (v508_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v514_data = ir2[14];
            ir2[14] = (v514_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v520_data = ir2[15];
            ir2[15] = (v520_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v522_data = r0[5];
            float v526_data = ir2[0];
            ir2[0] = (v526_data + (v522_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v532_data = ir2[1];
            ir2[1] = (v532_data + (v522_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v538_data = ir2[2];
            ir2[2] = (v538_data + (v522_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v544_data = ir2[3];
            ir2[3] = (v544_data + (v522_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v550_data = ir2[4];
            ir2[4] = (v550_data + (v522_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v556_data = ir2[5];
            ir2[5] = (v556_data + (v522_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v562_data = ir2[6];
            ir2[6] = (v562_data + (v522_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v568_data = ir2[7];
            ir2[7] = (v568_data + (v522_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v574_data = ir2[8];
            ir2[8] = (v574_data + (v522_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v580_data = ir2[9];
            ir2[9] = (v580_data + (v522_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v586_data = ir2[10];
            ir2[10] = (v586_data + (v522_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v592_data = ir2[11];
            ir2[11] = (v592_data + (v522_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v598_data = ir2[12];
            ir2[12] = (v598_data + (v522_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v604_data = ir2[13];
            ir2[13] = (v604_data + (v522_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v610_data = ir2[14];
            ir2[14] = (v610_data + (v522_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v616_data = ir2[15];
            ir2[15] = (v616_data + (v522_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v618_data = r0[6];
            float v622_data = ir2[0];
            ir2[0] = (v622_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v628_data = ir2[1];
            ir2[1] = (v628_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v634_data = ir2[2];
            ir2[2] = (v634_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v640_data = ir2[3];
            ir2[3] = (v640_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v646_data = ir2[4];
            ir2[4] = (v646_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v652_data = ir2[5];
            ir2[5] = (v652_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v658_data = ir2[6];
            ir2[6] = (v658_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v664_data = ir2[7];
            ir2[7] = (v664_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v670_data = ir2[8];
            ir2[8] = (v670_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v676_data = ir2[9];
            ir2[9] = (v676_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v682_data = ir2[10];
            ir2[10] = (v682_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v688_data = ir2[11];
            ir2[11] = (v688_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v694_data = ir2[12];
            ir2[12] = (v694_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v700_data = ir2[13];
            ir2[13] = (v700_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v706_data = ir2[14];
            ir2[14] = (v706_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v712_data = ir2[15];
            ir2[15] = (v712_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v714_data = r0[7];
            float v718_data = ir2[0];
            ir2[0] = (v718_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v724_data = ir2[1];
            ir2[1] = (v724_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v730_data = ir2[2];
            ir2[2] = (v730_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v736_data = ir2[3];
            ir2[3] = (v736_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v742_data = ir2[4];
            ir2[4] = (v742_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v748_data = ir2[5];
            ir2[5] = (v748_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v754_data = ir2[6];
            ir2[6] = (v754_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v760_data = ir2[7];
            ir2[7] = (v760_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v766_data = ir2[8];
            ir2[8] = (v766_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v772_data = ir2[9];
            ir2[9] = (v772_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v778_data = ir2[10];
            ir2[10] = (v778_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v784_data = ir2[11];
            ir2[11] = (v784_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v790_data = ir2[12];
            ir2[12] = (v790_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v796_data = ir2[13];
            ir2[13] = (v796_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v802_data = ir2[14];
            ir2[14] = (v802_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v808_data = ir2[15];
            ir2[15] = (v808_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v810_data = r0[8];
            float v814_data = ir2[0];
            ir2[0] = (v814_data + (v810_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v820_data = ir2[1];
            ir2[1] = (v820_data + (v810_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v826_data = ir2[2];
            ir2[2] = (v826_data + (v810_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v832_data = ir2[3];
            ir2[3] = (v832_data + (v810_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v838_data = ir2[4];
            ir2[4] = (v838_data + (v810_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v844_data = ir2[5];
            ir2[5] = (v844_data + (v810_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v850_data = ir2[6];
            ir2[6] = (v850_data + (v810_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v856_data = ir2[7];
            ir2[7] = (v856_data + (v810_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v862_data = ir2[8];
            ir2[8] = (v862_data + (v810_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v868_data = ir2[9];
            ir2[9] = (v868_data + (v810_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v874_data = ir2[10];
            ir2[10] = (v874_data + (v810_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v880_data = ir2[11];
            ir2[11] = (v880_data + (v810_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v886_data = ir2[12];
            ir2[12] = (v886_data + (v810_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v892_data = ir2[13];
            ir2[13] = (v892_data + (v810_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v898_data = ir2[14];
            ir2[14] = (v898_data + (v810_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v904_data = ir2[15];
            ir2[15] = (v904_data + (v810_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v906_data = r0[9];
            float v910_data = ir2[0];
            ir2[0] = (v910_data + (v906_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v916_data = ir2[1];
            ir2[1] = (v916_data + (v906_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v922_data = ir2[2];
            ir2[2] = (v922_data + (v906_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v928_data = ir2[3];
            ir2[3] = (v928_data + (v906_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v934_data = ir2[4];
            ir2[4] = (v934_data + (v906_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v940_data = ir2[5];
            ir2[5] = (v940_data + (v906_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v946_data = ir2[6];
            ir2[6] = (v946_data + (v906_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v952_data = ir2[7];
            ir2[7] = (v952_data + (v906_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v958_data = ir2[8];
            ir2[8] = (v958_data + (v906_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v964_data = ir2[9];
            ir2[9] = (v964_data + (v906_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v970_data = ir2[10];
            ir2[10] = (v970_data + (v906_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v976_data = ir2[11];
            ir2[11] = (v976_data + (v906_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v982_data = ir2[12];
            ir2[12] = (v982_data + (v906_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v988_data = ir2[13];
            ir2[13] = (v988_data + (v906_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v994_data = ir2[14];
            ir2[14] = (v994_data + (v906_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v1000_data = ir2[15];
            ir2[15] = (v1000_data + (v906_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v1002_data = r0[10];
            float v1006_data = ir2[0];
            ir2[0] = (v1006_data + (v1002_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1012_data = ir2[1];
            ir2[1] = (v1012_data + (v1002_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1018_data = ir2[2];
            ir2[2] = (v1018_data + (v1002_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1024_data = ir2[3];
            ir2[3] = (v1024_data + (v1002_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1030_data = ir2[4];
            ir2[4] = (v1030_data + (v1002_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1036_data = ir2[5];
            ir2[5] = (v1036_data + (v1002_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1042_data = ir2[6];
            ir2[6] = (v1042_data + (v1002_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1048_data = ir2[7];
            ir2[7] = (v1048_data + (v1002_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1054_data = ir2[8];
            ir2[8] = (v1054_data + (v1002_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1060_data = ir2[9];
            ir2[9] = (v1060_data + (v1002_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1066_data = ir2[10];
            ir2[10] = (v1066_data + (v1002_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1072_data = ir2[11];
            ir2[11] = (v1072_data + (v1002_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1078_data = ir2[12];
            ir2[12] = (v1078_data + (v1002_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1084_data = ir2[13];
            ir2[13] = (v1084_data + (v1002_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1090_data = ir2[14];
            ir2[14] = (v1090_data + (v1002_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1096_data = ir2[15];
            ir2[15] = (v1096_data + (v1002_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1098_data = r0[11];
            float v1102_data = ir2[0];
            ir2[0] = (v1102_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1108_data = ir2[1];
            ir2[1] = (v1108_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1114_data = ir2[2];
            ir2[2] = (v1114_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1120_data = ir2[3];
            ir2[3] = (v1120_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1126_data = ir2[4];
            ir2[4] = (v1126_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1132_data = ir2[5];
            ir2[5] = (v1132_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1138_data = ir2[6];
            ir2[6] = (v1138_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1144_data = ir2[7];
            ir2[7] = (v1144_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1150_data = ir2[8];
            ir2[8] = (v1150_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1156_data = ir2[9];
            ir2[9] = (v1156_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1162_data = ir2[10];
            ir2[10] = (v1162_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1168_data = ir2[11];
            ir2[11] = (v1168_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1174_data = ir2[12];
            ir2[12] = (v1174_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1180_data = ir2[13];
            ir2[13] = (v1180_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1186_data = ir2[14];
            ir2[14] = (v1186_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1192_data = ir2[15];
            ir2[15] = (v1192_data + (v1098_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1194_data = r0[12];
            float v1198_data = ir2[0];
            ir2[0] = (v1198_data + (v1194_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1204_data = ir2[1];
            ir2[1] = (v1204_data + (v1194_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1210_data = ir2[2];
            ir2[2] = (v1210_data + (v1194_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1216_data = ir2[3];
            ir2[3] = (v1216_data + (v1194_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1222_data = ir2[4];
            ir2[4] = (v1222_data + (v1194_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1228_data = ir2[5];
            ir2[5] = (v1228_data + (v1194_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1234_data = ir2[6];
            ir2[6] = (v1234_data + (v1194_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1240_data = ir2[7];
            ir2[7] = (v1240_data + (v1194_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1246_data = ir2[8];
            ir2[8] = (v1246_data + (v1194_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1252_data = ir2[9];
            ir2[9] = (v1252_data + (v1194_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1258_data = ir2[10];
            ir2[10] = (v1258_data + (v1194_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1264_data = ir2[11];
            ir2[11] = (v1264_data + (v1194_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1270_data = ir2[12];
            ir2[12] = (v1270_data + (v1194_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1276_data = ir2[13];
            ir2[13] = (v1276_data + (v1194_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1282_data = ir2[14];
            ir2[14] = (v1282_data + (v1194_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1288_data = ir2[15];
            ir2[15] = (v1288_data + (v1194_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1290_data = r0[13];
            float v1294_data = ir2[0];
            ir2[0] = (v1294_data + (v1290_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1300_data = ir2[1];
            ir2[1] = (v1300_data + (v1290_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1306_data = ir2[2];
            ir2[2] = (v1306_data + (v1290_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1312_data = ir2[3];
            ir2[3] = (v1312_data + (v1290_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1318_data = ir2[4];
            ir2[4] = (v1318_data + (v1290_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1324_data = ir2[5];
            ir2[5] = (v1324_data + (v1290_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1330_data = ir2[6];
            ir2[6] = (v1330_data + (v1290_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1336_data = ir2[7];
            ir2[7] = (v1336_data + (v1290_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1342_data = ir2[8];
            ir2[8] = (v1342_data + (v1290_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1348_data = ir2[9];
            ir2[9] = (v1348_data + (v1290_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1354_data = ir2[10];
            ir2[10] = (v1354_data + (v1290_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1360_data = ir2[11];
            ir2[11] = (v1360_data + (v1290_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1366_data = ir2[12];
            ir2[12] = (v1366_data + (v1290_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1372_data = ir2[13];
            ir2[13] = (v1372_data + (v1290_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1378_data = ir2[14];
            ir2[14] = (v1378_data + (v1290_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1384_data = ir2[15];
            ir2[15] = (v1384_data + (v1290_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1386_data = r0[14];
            float v1390_data = ir2[0];
            ir2[0] = (v1390_data + (v1386_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1396_data = ir2[1];
            ir2[1] = (v1396_data + (v1386_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1402_data = ir2[2];
            ir2[2] = (v1402_data + (v1386_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1408_data = ir2[3];
            ir2[3] = (v1408_data + (v1386_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1414_data = ir2[4];
            ir2[4] = (v1414_data + (v1386_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1420_data = ir2[5];
            ir2[5] = (v1420_data + (v1386_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1426_data = ir2[6];
            ir2[6] = (v1426_data + (v1386_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1432_data = ir2[7];
            ir2[7] = (v1432_data + (v1386_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1438_data = ir2[8];
            ir2[8] = (v1438_data + (v1386_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1444_data = ir2[9];
            ir2[9] = (v1444_data + (v1386_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1450_data = ir2[10];
            ir2[10] = (v1450_data + (v1386_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1456_data = ir2[11];
            ir2[11] = (v1456_data + (v1386_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1462_data = ir2[12];
            ir2[12] = (v1462_data + (v1386_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1468_data = ir2[13];
            ir2[13] = (v1468_data + (v1386_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1474_data = ir2[14];
            ir2[14] = (v1474_data + (v1386_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1480_data = ir2[15];
            ir2[15] = (v1480_data + (v1386_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1482_data = r0[15];
            float v1486_data = ir2[0];
            ir2[0] = (v1486_data + (v1482_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1492_data = ir2[1];
            ir2[1] = (v1492_data + (v1482_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1498_data = ir2[2];
            ir2[2] = (v1498_data + (v1482_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1504_data = ir2[3];
            ir2[3] = (v1504_data + (v1482_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1510_data = ir2[4];
            ir2[4] = (v1510_data + (v1482_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1516_data = ir2[5];
            ir2[5] = (v1516_data + (v1482_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1522_data = ir2[6];
            ir2[6] = (v1522_data + (v1482_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1528_data = ir2[7];
            ir2[7] = (v1528_data + (v1482_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1534_data = ir2[8];
            ir2[8] = (v1534_data + (v1482_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1540_data = ir2[9];
            ir2[9] = (v1540_data + (v1482_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1546_data = ir2[10];
            ir2[10] = (v1546_data + (v1482_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1552_data = ir2[11];
            ir2[11] = (v1552_data + (v1482_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1558_data = ir2[12];
            ir2[12] = (v1558_data + (v1482_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1564_data = ir2[13];
            ir2[13] = (v1564_data + (v1482_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1570_data = ir2[14];
            ir2[14] = (v1570_data + (v1482_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1576_data = ir2[15];
            ir2[15] = (v1576_data + (v1482_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            // r2 = ir2
            #pragma unroll
            for (int32_t v1578_n0 = 0; v1578_n0 < 1; ++v1578_n0) {
              #pragma unroll
              for (int32_t v1579_n1 = 0; v1579_n1 < 16; ++v1579_n1) {
                int32_t v1580_a = v1578_n0 + v1579_n1;
                float v1581_data = ir2[v1580_a];
                r2[v1580_a] = v1581_data;
              }
            }
            // glb_m0 = store{r>g}(r2);
            #pragma unroll
            for (int32_t v1582_i0 = 0; v1582_i0 < 1; ++v1582_i0) {
              int32_t v1587_lead = v22_lead + (v1582_i0 * 16);
              #pragma unroll
              for (int32_t v1583_i1 = 0; v1583_i1 < 16; ++v1583_i1) {
                float v1585_data = r2[(v1582_i0 + v1583_i1)];
                glb_m0[(v1587_lead + (v1583_i1 * 16))] = v1585_data;
              }
            }
            sycl::group_barrier(item.get_sub_group());
          }
        }
      });
    }
  });
}

