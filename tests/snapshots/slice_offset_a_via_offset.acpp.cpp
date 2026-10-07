// === base name ===
kernel_9cbf31750be932cc

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_9cbf31750be932cc = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_9cbf31750be932cc(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_9cbf31750be932cc(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_9cbf31750be932cc(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_9cbf31750be932cc(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_9cbf31750be932cc(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_9cbf31750be932cc(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_9cbf31750be932cc(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 12×8(12×8) {0..12}×{0..8} strided
        //   m1 32×16(32×16) {0..32}×{0..16} strided
        //   m2 16×8(16×8) {0..16}×{0..8} strided
        // operations:
        //   m0[i,j] = m1[i,k]@{4..16}×{0..16} × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,8]],"name":"m0","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[32,16]],"name":"m1","ordered":false,"parts":1,"shape":[32,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m1","offset":[4,0],"shape":[32,16]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 512 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 128 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v21_lead = item.get_local_id(2) % 16;
              bool v22_g = v21_lead < 12;
              if (v22_g) {
                int32_t v26_off = v21_lead + 4;
                #pragma unroll
                for (int32_t v23_i1 = 0; v23_i1 < 16; ++v23_i1) {
                  float v29_data = glb_m1[(v26_off + (v23_i1 * 32))];
                  r0[v23_i1] = v29_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v32_i0 = 0; v32_i0 < 1; ++v32_i0) {
                int32_t v35_lead = v21_lead + (v32_i0 * 16);
                #pragma unroll
                for (int32_t v33_i1 = 0; v33_i1 < 8; ++v33_i1) {
                  float v38_data = glb_m2[(v35_lead + (v33_i1 * 16))];
                  r1[(v32_i0 + v33_i1)] = v38_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // ir2 = +(r0 * r1)
              // [(0, 12), (0, 8)] [(0, 16)]
              float ir2[8]{};
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
              float v90_data = r0[1];
              float v94_data = ir2[0];
              ir2[0] = (v94_data + (v90_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v100_data = ir2[1];
              ir2[1] = (v100_data + (v90_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v106_data = ir2[2];
              ir2[2] = (v106_data + (v90_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v112_data = ir2[3];
              ir2[3] = (v112_data + (v90_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v118_data = ir2[4];
              ir2[4] = (v118_data + (v90_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v124_data = ir2[5];
              ir2[5] = (v124_data + (v90_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v130_data = ir2[6];
              ir2[6] = (v130_data + (v90_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v136_data = ir2[7];
              ir2[7] = (v136_data + (v90_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v138_data = r0[2];
              float v142_data = ir2[0];
              ir2[0] = (v142_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v148_data = ir2[1];
              ir2[1] = (v148_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v154_data = ir2[2];
              ir2[2] = (v154_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v160_data = ir2[3];
              ir2[3] = (v160_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v166_data = ir2[4];
              ir2[4] = (v166_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v172_data = ir2[5];
              ir2[5] = (v172_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v178_data = ir2[6];
              ir2[6] = (v178_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v184_data = ir2[7];
              ir2[7] = (v184_data + (v138_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v186_data = r0[3];
              float v190_data = ir2[0];
              ir2[0] = (v190_data + (v186_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v196_data = ir2[1];
              ir2[1] = (v196_data + (v186_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v202_data = ir2[2];
              ir2[2] = (v202_data + (v186_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v208_data = ir2[3];
              ir2[3] = (v208_data + (v186_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v214_data = ir2[4];
              ir2[4] = (v214_data + (v186_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v220_data = ir2[5];
              ir2[5] = (v220_data + (v186_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v226_data = ir2[6];
              ir2[6] = (v226_data + (v186_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v232_data = ir2[7];
              ir2[7] = (v232_data + (v186_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v234_data = r0[4];
              float v238_data = ir2[0];
              ir2[0] = (v238_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v244_data = ir2[1];
              ir2[1] = (v244_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v250_data = ir2[2];
              ir2[2] = (v250_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v256_data = ir2[3];
              ir2[3] = (v256_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v262_data = ir2[4];
              ir2[4] = (v262_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v268_data = ir2[5];
              ir2[5] = (v268_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v274_data = ir2[6];
              ir2[6] = (v274_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v280_data = ir2[7];
              ir2[7] = (v280_data + (v234_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v282_data = r0[5];
              float v286_data = ir2[0];
              ir2[0] = (v286_data + (v282_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v292_data = ir2[1];
              ir2[1] = (v292_data + (v282_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v298_data = ir2[2];
              ir2[2] = (v298_data + (v282_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v304_data = ir2[3];
              ir2[3] = (v304_data + (v282_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v310_data = ir2[4];
              ir2[4] = (v310_data + (v282_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v316_data = ir2[5];
              ir2[5] = (v316_data + (v282_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v322_data = ir2[6];
              ir2[6] = (v322_data + (v282_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v328_data = ir2[7];
              ir2[7] = (v328_data + (v282_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v330_data = r0[6];
              float v334_data = ir2[0];
              ir2[0] = (v334_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v340_data = ir2[1];
              ir2[1] = (v340_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v346_data = ir2[2];
              ir2[2] = (v346_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v352_data = ir2[3];
              ir2[3] = (v352_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v358_data = ir2[4];
              ir2[4] = (v358_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v364_data = ir2[5];
              ir2[5] = (v364_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v370_data = ir2[6];
              ir2[6] = (v370_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v376_data = ir2[7];
              ir2[7] = (v376_data + (v330_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v378_data = r0[7];
              float v382_data = ir2[0];
              ir2[0] = (v382_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v388_data = ir2[1];
              ir2[1] = (v388_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v394_data = ir2[2];
              ir2[2] = (v394_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v400_data = ir2[3];
              ir2[3] = (v400_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v406_data = ir2[4];
              ir2[4] = (v406_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v412_data = ir2[5];
              ir2[5] = (v412_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v418_data = ir2[6];
              ir2[6] = (v418_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v424_data = ir2[7];
              ir2[7] = (v424_data + (v378_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v426_data = r0[8];
              float v430_data = ir2[0];
              ir2[0] = (v430_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v436_data = ir2[1];
              ir2[1] = (v436_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v442_data = ir2[2];
              ir2[2] = (v442_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v448_data = ir2[3];
              ir2[3] = (v448_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v454_data = ir2[4];
              ir2[4] = (v454_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v460_data = ir2[5];
              ir2[5] = (v460_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v466_data = ir2[6];
              ir2[6] = (v466_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v472_data = ir2[7];
              ir2[7] = (v472_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v474_data = r0[9];
              float v478_data = ir2[0];
              ir2[0] = (v478_data + (v474_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v484_data = ir2[1];
              ir2[1] = (v484_data + (v474_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v490_data = ir2[2];
              ir2[2] = (v490_data + (v474_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v496_data = ir2[3];
              ir2[3] = (v496_data + (v474_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v502_data = ir2[4];
              ir2[4] = (v502_data + (v474_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v508_data = ir2[5];
              ir2[5] = (v508_data + (v474_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v514_data = ir2[6];
              ir2[6] = (v514_data + (v474_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v520_data = ir2[7];
              ir2[7] = (v520_data + (v474_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v522_data = r0[10];
              float v526_data = ir2[0];
              ir2[0] = (v526_data + (v522_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v532_data = ir2[1];
              ir2[1] = (v532_data + (v522_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v538_data = ir2[2];
              ir2[2] = (v538_data + (v522_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v544_data = ir2[3];
              ir2[3] = (v544_data + (v522_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v550_data = ir2[4];
              ir2[4] = (v550_data + (v522_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v556_data = ir2[5];
              ir2[5] = (v556_data + (v522_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v562_data = ir2[6];
              ir2[6] = (v562_data + (v522_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v568_data = ir2[7];
              ir2[7] = (v568_data + (v522_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v570_data = r0[11];
              float v574_data = ir2[0];
              ir2[0] = (v574_data + (v570_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v580_data = ir2[1];
              ir2[1] = (v580_data + (v570_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v586_data = ir2[2];
              ir2[2] = (v586_data + (v570_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v592_data = ir2[3];
              ir2[3] = (v592_data + (v570_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v598_data = ir2[4];
              ir2[4] = (v598_data + (v570_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v604_data = ir2[5];
              ir2[5] = (v604_data + (v570_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v610_data = ir2[6];
              ir2[6] = (v610_data + (v570_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v616_data = ir2[7];
              ir2[7] = (v616_data + (v570_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v618_data = r0[12];
              float v622_data = ir2[0];
              ir2[0] = (v622_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v628_data = ir2[1];
              ir2[1] = (v628_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v634_data = ir2[2];
              ir2[2] = (v634_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v640_data = ir2[3];
              ir2[3] = (v640_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v646_data = ir2[4];
              ir2[4] = (v646_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v652_data = ir2[5];
              ir2[5] = (v652_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v658_data = ir2[6];
              ir2[6] = (v658_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v664_data = ir2[7];
              ir2[7] = (v664_data + (v618_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v666_data = r0[13];
              float v670_data = ir2[0];
              ir2[0] = (v670_data + (v666_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v676_data = ir2[1];
              ir2[1] = (v676_data + (v666_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v682_data = ir2[2];
              ir2[2] = (v682_data + (v666_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v688_data = ir2[3];
              ir2[3] = (v688_data + (v666_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v694_data = ir2[4];
              ir2[4] = (v694_data + (v666_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v700_data = ir2[5];
              ir2[5] = (v700_data + (v666_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v706_data = ir2[6];
              ir2[6] = (v706_data + (v666_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v712_data = ir2[7];
              ir2[7] = (v712_data + (v666_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v714_data = r0[14];
              float v718_data = ir2[0];
              ir2[0] = (v718_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v724_data = ir2[1];
              ir2[1] = (v724_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v730_data = ir2[2];
              ir2[2] = (v730_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v736_data = ir2[3];
              ir2[3] = (v736_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v742_data = ir2[4];
              ir2[4] = (v742_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v748_data = ir2[5];
              ir2[5] = (v748_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v754_data = ir2[6];
              ir2[6] = (v754_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v760_data = ir2[7];
              ir2[7] = (v760_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v762_data = r0[15];
              float v766_data = ir2[0];
              ir2[0] = (v766_data + (v762_data * (sycl::select_from_group(item.get_sub_group(), v43_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v772_data = ir2[1];
              ir2[1] = (v772_data + (v762_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v778_data = ir2[2];
              ir2[2] = (v778_data + (v762_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v784_data = ir2[3];
              ir2[3] = (v784_data + (v762_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v790_data = ir2[4];
              ir2[4] = (v790_data + (v762_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v796_data = ir2[5];
              ir2[5] = (v796_data + (v762_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v802_data = ir2[6];
              ir2[6] = (v802_data + (v762_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v808_data = ir2[7];
              ir2[7] = (v808_data + (v762_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              // r2 = ir2
              if (v22_g) {
                #pragma unroll
                for (int32_t v810_n1 = 0; v810_n1 < 8; ++v810_n1) {
                  float v812_data = ir2[v810_n1];
                  r2[v810_n1] = v812_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v22_g) {
                #pragma unroll
                for (int32_t v813_i1 = 0; v813_i1 < 8; ++v813_i1) {
                  float v815_data = r2[v813_i1];
                  glb_m0[(v21_lead + (v813_i1 * 12))] = v815_data;
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

