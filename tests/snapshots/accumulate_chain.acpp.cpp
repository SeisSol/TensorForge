// === base name ===
kernel_bea245535eeca693

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_bea245535eeca693 = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_bea245535eeca693(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_bea245535eeca693(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_bea245535eeca693(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_bea245535eeca693(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_bea245535eeca693(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_bea245535eeca693(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, m6, m6_extraOffset, m7, m7_extraOffset, m8, m8_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_bea245535eeca693(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 12×8(12×8) {0..12}×{0..8} strided
        //   m1 12×12(12×12) {0..12}×{0..12} strided
        //   m2 12×8(12×8) {0..12}×{0..8} strided
        //   m3 12×12(12×12) {0..12}×{0..12} strided
        //   m4 12×8(12×8) {0..12}×{0..8} strided
        //   m5 12×12(12×12) {0..12}×{0..12} strided
        //   m6 12×8(12×8) {0..12}×{0..8} strided
        //   m7 12×12(12×12) {0..12}×{0..12} strided
        //   m8 12×8(12×8) {0..12}×{0..8} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        //   m0[i,j] += m3[i,k] × m4[k,j]
        //   m0[i,j] += m5[i,k] × m6[k,j]
        //   m0[i,j] += m7[i,k] × m8[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[12,8]],"name":"m0","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A0","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B0","bbox":[[0,0],[12,8]],"name":"m2","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A1","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[12,8]],"name":"m4","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A2","bbox":[[0,0],[12,12]],"name":"m5","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[12,8]],"name":"m6","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A3","bbox":[[0,0],[12,12]],"name":"m7","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B3","bbox":[[0,0],[12,8]],"name":"m8","ordered":false,"parts":1,"shape":[12,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m7","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m8","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 96 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v7_batchId0 * 144 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v7_batchId0 * 96 + 0 + m4_extraOffset];
              const float *const __restrict__ glb_m5 = &m5[v7_batchId0 * 144 + 0 + m5_extraOffset];
              const float *const __restrict__ glb_m6 = &m6[v7_batchId0 * 96 + 0 + m6_extraOffset];
              const float *const __restrict__ glb_m7 = &m7[v7_batchId0 * 144 + 0 + m7_extraOffset];
              const float *const __restrict__ glb_m8 = &m8[v7_batchId0 * 96 + 0 + m8_extraOffset];
              float r0[12]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v27_lead = item.get_local_id(2) % 16;
              bool v28_g = v27_lead < 12;
              if (v28_g) {
                #pragma unroll
                for (int32_t v29_i1 = 0; v29_i1 < 12; ++v29_i1) {
                  float v34_data = glb_m1[(v27_lead + (v29_i1 * 12))];
                  r0[v29_i1] = v34_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              if (v28_g) {
                #pragma unroll
                for (int32_t v37_i1 = 0; v37_i1 < 8; ++v37_i1) {
                  float v42_data = glb_m2[(v27_lead + (v37_i1 * 12))];
                  r1[v37_i1] = v42_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              float r3[12]{};
              // r3 = load{g>r}(glb_m3);
              if (v28_g) {
                #pragma unroll
                for (int32_t v45_i1 = 0; v45_i1 < 12; ++v45_i1) {
                  float v50_data = glb_m3[(v27_lead + (v45_i1 * 12))];
                  r3[v45_i1] = v50_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // ir2 = +(r0 * r1)
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir2[8]{};
              float v54_data = r0[0];
              float v55_data = r1[0];
              float v58_data = ir2[0];
              ir2[0] = (v58_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v61_data = r1[1];
              float v64_data = ir2[1];
              ir2[1] = (v64_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v67_data = r1[2];
              float v70_data = ir2[2];
              ir2[2] = (v70_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v73_data = r1[3];
              float v76_data = ir2[3];
              ir2[3] = (v76_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v79_data = r1[4];
              float v82_data = ir2[4];
              ir2[4] = (v82_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v85_data = r1[5];
              float v88_data = ir2[5];
              ir2[5] = (v88_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v91_data = r1[6];
              float v94_data = ir2[6];
              ir2[6] = (v94_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v97_data = r1[7];
              float v100_data = ir2[7];
              ir2[7] = (v100_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v102_data = r0[1];
              float v106_data = ir2[0];
              ir2[0] = (v106_data + (v102_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v112_data = ir2[1];
              ir2[1] = (v112_data + (v102_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v118_data = ir2[2];
              ir2[2] = (v118_data + (v102_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v124_data = ir2[3];
              ir2[3] = (v124_data + (v102_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v130_data = ir2[4];
              ir2[4] = (v130_data + (v102_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v136_data = ir2[5];
              ir2[5] = (v136_data + (v102_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v142_data = ir2[6];
              ir2[6] = (v142_data + (v102_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v148_data = ir2[7];
              ir2[7] = (v148_data + (v102_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v150_data = r0[2];
              float v154_data = ir2[0];
              ir2[0] = (v154_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v160_data = ir2[1];
              ir2[1] = (v160_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v166_data = ir2[2];
              ir2[2] = (v166_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v172_data = ir2[3];
              ir2[3] = (v172_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v178_data = ir2[4];
              ir2[4] = (v178_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v184_data = ir2[5];
              ir2[5] = (v184_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v190_data = ir2[6];
              ir2[6] = (v190_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v196_data = ir2[7];
              ir2[7] = (v196_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v198_data = r0[3];
              float v202_data = ir2[0];
              ir2[0] = (v202_data + (v198_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v208_data = ir2[1];
              ir2[1] = (v208_data + (v198_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v214_data = ir2[2];
              ir2[2] = (v214_data + (v198_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v220_data = ir2[3];
              ir2[3] = (v220_data + (v198_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v226_data = ir2[4];
              ir2[4] = (v226_data + (v198_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v232_data = ir2[5];
              ir2[5] = (v232_data + (v198_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v238_data = ir2[6];
              ir2[6] = (v238_data + (v198_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v244_data = ir2[7];
              ir2[7] = (v244_data + (v198_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v246_data = r0[4];
              float v250_data = ir2[0];
              ir2[0] = (v250_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v256_data = ir2[1];
              ir2[1] = (v256_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v262_data = ir2[2];
              ir2[2] = (v262_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v268_data = ir2[3];
              ir2[3] = (v268_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v274_data = ir2[4];
              ir2[4] = (v274_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v280_data = ir2[5];
              ir2[5] = (v280_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v286_data = ir2[6];
              ir2[6] = (v286_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v292_data = ir2[7];
              ir2[7] = (v292_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v294_data = r0[5];
              float v298_data = ir2[0];
              ir2[0] = (v298_data + (v294_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v304_data = ir2[1];
              ir2[1] = (v304_data + (v294_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v310_data = ir2[2];
              ir2[2] = (v310_data + (v294_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v316_data = ir2[3];
              ir2[3] = (v316_data + (v294_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v322_data = ir2[4];
              ir2[4] = (v322_data + (v294_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v328_data = ir2[5];
              ir2[5] = (v328_data + (v294_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v334_data = ir2[6];
              ir2[6] = (v334_data + (v294_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v340_data = ir2[7];
              ir2[7] = (v340_data + (v294_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v342_data = r0[6];
              float v346_data = ir2[0];
              ir2[0] = (v346_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v352_data = ir2[1];
              ir2[1] = (v352_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v358_data = ir2[2];
              ir2[2] = (v358_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v364_data = ir2[3];
              ir2[3] = (v364_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v370_data = ir2[4];
              ir2[4] = (v370_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v376_data = ir2[5];
              ir2[5] = (v376_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v382_data = ir2[6];
              ir2[6] = (v382_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v388_data = ir2[7];
              ir2[7] = (v388_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v390_data = r0[7];
              float v394_data = ir2[0];
              ir2[0] = (v394_data + (v390_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v400_data = ir2[1];
              ir2[1] = (v400_data + (v390_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v406_data = ir2[2];
              ir2[2] = (v406_data + (v390_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v412_data = ir2[3];
              ir2[3] = (v412_data + (v390_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v418_data = ir2[4];
              ir2[4] = (v418_data + (v390_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v424_data = ir2[5];
              ir2[5] = (v424_data + (v390_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v430_data = ir2[6];
              ir2[6] = (v430_data + (v390_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v436_data = ir2[7];
              ir2[7] = (v436_data + (v390_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v438_data = r0[8];
              float v442_data = ir2[0];
              ir2[0] = (v442_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v448_data = ir2[1];
              ir2[1] = (v448_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v454_data = ir2[2];
              ir2[2] = (v454_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v460_data = ir2[3];
              ir2[3] = (v460_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v466_data = ir2[4];
              ir2[4] = (v466_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v472_data = ir2[5];
              ir2[5] = (v472_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v478_data = ir2[6];
              ir2[6] = (v478_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v484_data = ir2[7];
              ir2[7] = (v484_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v486_data = r0[9];
              float v490_data = ir2[0];
              ir2[0] = (v490_data + (v486_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v496_data = ir2[1];
              ir2[1] = (v496_data + (v486_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v502_data = ir2[2];
              ir2[2] = (v502_data + (v486_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v508_data = ir2[3];
              ir2[3] = (v508_data + (v486_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v514_data = ir2[4];
              ir2[4] = (v514_data + (v486_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v520_data = ir2[5];
              ir2[5] = (v520_data + (v486_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v526_data = ir2[6];
              ir2[6] = (v526_data + (v486_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v532_data = ir2[7];
              ir2[7] = (v532_data + (v486_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v534_data = r0[10];
              float v538_data = ir2[0];
              ir2[0] = (v538_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v544_data = ir2[1];
              ir2[1] = (v544_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v550_data = ir2[2];
              ir2[2] = (v550_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v556_data = ir2[3];
              ir2[3] = (v556_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v562_data = ir2[4];
              ir2[4] = (v562_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v568_data = ir2[5];
              ir2[5] = (v568_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v574_data = ir2[6];
              ir2[6] = (v574_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v580_data = ir2[7];
              ir2[7] = (v580_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v582_data = r0[11];
              float v586_data = ir2[0];
              ir2[0] = (v586_data + (v582_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v592_data = ir2[1];
              ir2[1] = (v592_data + (v582_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v598_data = ir2[2];
              ir2[2] = (v598_data + (v582_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v604_data = ir2[3];
              ir2[3] = (v604_data + (v582_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v610_data = ir2[4];
              ir2[4] = (v610_data + (v582_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v616_data = ir2[5];
              ir2[5] = (v616_data + (v582_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v622_data = ir2[6];
              ir2[6] = (v622_data + (v582_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v628_data = ir2[7];
              ir2[7] = (v628_data + (v582_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              // r2 = ir2
              if (v28_g) {
                #pragma unroll
                for (int32_t v630_n1 = 0; v630_n1 < 8; ++v630_n1) {
                  float v632_data = ir2[v630_n1];
                  r2[v630_n1] = v632_data;
                }
              }
              float r4[8]{};
              // r4 = load{g>r}(glb_m4);
              if (v28_g) {
                #pragma unroll
                for (int32_t v634_i1 = 0; v634_i1 < 8; ++v634_i1) {
                  float v639_data = glb_m4[(v27_lead + (v634_i1 * 12))];
                  r4[v634_i1] = v639_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m3););
              float r6[12]{};
              // r6 = load{g>r}(glb_m5);
              if (v28_g) {
                #pragma unroll
                for (int32_t v642_i1 = 0; v642_i1 < 12; ++v642_i1) {
                  float v647_data = glb_m5[(v27_lead + (v642_i1 * 12))];
                  r6[v642_i1] = v647_data;
                }
              }
              // wait(r4 = load{g>r}(glb_m4););
              float r5[8]{};
              // ir5 = +(r3 * r4)
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir5[8]{};
              float v651_data = r3[0];
              float v652_data = r4[0];
              float v655_data = ir5[0];
              ir5[0] = (v655_data + (v651_data * (sycl::select_from_group(item.get_sub_group(), v652_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v658_data = r4[1];
              float v661_data = ir5[1];
              ir5[1] = (v661_data + (v651_data * (sycl::select_from_group(item.get_sub_group(), v658_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v664_data = r4[2];
              float v667_data = ir5[2];
              ir5[2] = (v667_data + (v651_data * (sycl::select_from_group(item.get_sub_group(), v664_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v670_data = r4[3];
              float v673_data = ir5[3];
              ir5[3] = (v673_data + (v651_data * (sycl::select_from_group(item.get_sub_group(), v670_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v676_data = r4[4];
              float v679_data = ir5[4];
              ir5[4] = (v679_data + (v651_data * (sycl::select_from_group(item.get_sub_group(), v676_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v682_data = r4[5];
              float v685_data = ir5[5];
              ir5[5] = (v685_data + (v651_data * (sycl::select_from_group(item.get_sub_group(), v682_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v688_data = r4[6];
              float v691_data = ir5[6];
              ir5[6] = (v691_data + (v651_data * (sycl::select_from_group(item.get_sub_group(), v688_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v694_data = r4[7];
              float v697_data = ir5[7];
              ir5[7] = (v697_data + (v651_data * (sycl::select_from_group(item.get_sub_group(), v694_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v699_data = r3[1];
              float v703_data = ir5[0];
              ir5[0] = (v703_data + (v699_data * (sycl::select_from_group(item.get_sub_group(), v652_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v709_data = ir5[1];
              ir5[1] = (v709_data + (v699_data * (sycl::select_from_group(item.get_sub_group(), v658_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v715_data = ir5[2];
              ir5[2] = (v715_data + (v699_data * (sycl::select_from_group(item.get_sub_group(), v664_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v721_data = ir5[3];
              ir5[3] = (v721_data + (v699_data * (sycl::select_from_group(item.get_sub_group(), v670_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v727_data = ir5[4];
              ir5[4] = (v727_data + (v699_data * (sycl::select_from_group(item.get_sub_group(), v676_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v733_data = ir5[5];
              ir5[5] = (v733_data + (v699_data * (sycl::select_from_group(item.get_sub_group(), v682_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v739_data = ir5[6];
              ir5[6] = (v739_data + (v699_data * (sycl::select_from_group(item.get_sub_group(), v688_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v745_data = ir5[7];
              ir5[7] = (v745_data + (v699_data * (sycl::select_from_group(item.get_sub_group(), v694_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v747_data = r3[2];
              float v751_data = ir5[0];
              ir5[0] = (v751_data + (v747_data * (sycl::select_from_group(item.get_sub_group(), v652_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v757_data = ir5[1];
              ir5[1] = (v757_data + (v747_data * (sycl::select_from_group(item.get_sub_group(), v658_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v763_data = ir5[2];
              ir5[2] = (v763_data + (v747_data * (sycl::select_from_group(item.get_sub_group(), v664_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v769_data = ir5[3];
              ir5[3] = (v769_data + (v747_data * (sycl::select_from_group(item.get_sub_group(), v670_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v775_data = ir5[4];
              ir5[4] = (v775_data + (v747_data * (sycl::select_from_group(item.get_sub_group(), v676_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v781_data = ir5[5];
              ir5[5] = (v781_data + (v747_data * (sycl::select_from_group(item.get_sub_group(), v682_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v787_data = ir5[6];
              ir5[6] = (v787_data + (v747_data * (sycl::select_from_group(item.get_sub_group(), v688_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v793_data = ir5[7];
              ir5[7] = (v793_data + (v747_data * (sycl::select_from_group(item.get_sub_group(), v694_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v795_data = r3[3];
              float v799_data = ir5[0];
              ir5[0] = (v799_data + (v795_data * (sycl::select_from_group(item.get_sub_group(), v652_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v805_data = ir5[1];
              ir5[1] = (v805_data + (v795_data * (sycl::select_from_group(item.get_sub_group(), v658_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v811_data = ir5[2];
              ir5[2] = (v811_data + (v795_data * (sycl::select_from_group(item.get_sub_group(), v664_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v817_data = ir5[3];
              ir5[3] = (v817_data + (v795_data * (sycl::select_from_group(item.get_sub_group(), v670_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v823_data = ir5[4];
              ir5[4] = (v823_data + (v795_data * (sycl::select_from_group(item.get_sub_group(), v676_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v829_data = ir5[5];
              ir5[5] = (v829_data + (v795_data * (sycl::select_from_group(item.get_sub_group(), v682_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v835_data = ir5[6];
              ir5[6] = (v835_data + (v795_data * (sycl::select_from_group(item.get_sub_group(), v688_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v841_data = ir5[7];
              ir5[7] = (v841_data + (v795_data * (sycl::select_from_group(item.get_sub_group(), v694_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v843_data = r3[4];
              float v847_data = ir5[0];
              ir5[0] = (v847_data + (v843_data * (sycl::select_from_group(item.get_sub_group(), v652_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v853_data = ir5[1];
              ir5[1] = (v853_data + (v843_data * (sycl::select_from_group(item.get_sub_group(), v658_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v859_data = ir5[2];
              ir5[2] = (v859_data + (v843_data * (sycl::select_from_group(item.get_sub_group(), v664_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v865_data = ir5[3];
              ir5[3] = (v865_data + (v843_data * (sycl::select_from_group(item.get_sub_group(), v670_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v871_data = ir5[4];
              ir5[4] = (v871_data + (v843_data * (sycl::select_from_group(item.get_sub_group(), v676_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v877_data = ir5[5];
              ir5[5] = (v877_data + (v843_data * (sycl::select_from_group(item.get_sub_group(), v682_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v883_data = ir5[6];
              ir5[6] = (v883_data + (v843_data * (sycl::select_from_group(item.get_sub_group(), v688_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v889_data = ir5[7];
              ir5[7] = (v889_data + (v843_data * (sycl::select_from_group(item.get_sub_group(), v694_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v891_data = r3[5];
              float v895_data = ir5[0];
              ir5[0] = (v895_data + (v891_data * (sycl::select_from_group(item.get_sub_group(), v652_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v901_data = ir5[1];
              ir5[1] = (v901_data + (v891_data * (sycl::select_from_group(item.get_sub_group(), v658_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v907_data = ir5[2];
              ir5[2] = (v907_data + (v891_data * (sycl::select_from_group(item.get_sub_group(), v664_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v913_data = ir5[3];
              ir5[3] = (v913_data + (v891_data * (sycl::select_from_group(item.get_sub_group(), v670_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v919_data = ir5[4];
              ir5[4] = (v919_data + (v891_data * (sycl::select_from_group(item.get_sub_group(), v676_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v925_data = ir5[5];
              ir5[5] = (v925_data + (v891_data * (sycl::select_from_group(item.get_sub_group(), v682_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v931_data = ir5[6];
              ir5[6] = (v931_data + (v891_data * (sycl::select_from_group(item.get_sub_group(), v688_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v937_data = ir5[7];
              ir5[7] = (v937_data + (v891_data * (sycl::select_from_group(item.get_sub_group(), v694_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v939_data = r3[6];
              float v943_data = ir5[0];
              ir5[0] = (v943_data + (v939_data * (sycl::select_from_group(item.get_sub_group(), v652_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v949_data = ir5[1];
              ir5[1] = (v949_data + (v939_data * (sycl::select_from_group(item.get_sub_group(), v658_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v955_data = ir5[2];
              ir5[2] = (v955_data + (v939_data * (sycl::select_from_group(item.get_sub_group(), v664_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v961_data = ir5[3];
              ir5[3] = (v961_data + (v939_data * (sycl::select_from_group(item.get_sub_group(), v670_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v967_data = ir5[4];
              ir5[4] = (v967_data + (v939_data * (sycl::select_from_group(item.get_sub_group(), v676_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v973_data = ir5[5];
              ir5[5] = (v973_data + (v939_data * (sycl::select_from_group(item.get_sub_group(), v682_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v979_data = ir5[6];
              ir5[6] = (v979_data + (v939_data * (sycl::select_from_group(item.get_sub_group(), v688_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v985_data = ir5[7];
              ir5[7] = (v985_data + (v939_data * (sycl::select_from_group(item.get_sub_group(), v694_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v987_data = r3[7];
              float v991_data = ir5[0];
              ir5[0] = (v991_data + (v987_data * (sycl::select_from_group(item.get_sub_group(), v652_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v997_data = ir5[1];
              ir5[1] = (v997_data + (v987_data * (sycl::select_from_group(item.get_sub_group(), v658_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1003_data = ir5[2];
              ir5[2] = (v1003_data + (v987_data * (sycl::select_from_group(item.get_sub_group(), v664_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1009_data = ir5[3];
              ir5[3] = (v1009_data + (v987_data * (sycl::select_from_group(item.get_sub_group(), v670_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1015_data = ir5[4];
              ir5[4] = (v1015_data + (v987_data * (sycl::select_from_group(item.get_sub_group(), v676_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1021_data = ir5[5];
              ir5[5] = (v1021_data + (v987_data * (sycl::select_from_group(item.get_sub_group(), v682_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1027_data = ir5[6];
              ir5[6] = (v1027_data + (v987_data * (sycl::select_from_group(item.get_sub_group(), v688_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1033_data = ir5[7];
              ir5[7] = (v1033_data + (v987_data * (sycl::select_from_group(item.get_sub_group(), v694_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1035_data = r3[8];
              float v1039_data = ir5[0];
              ir5[0] = (v1039_data + (v1035_data * (sycl::select_from_group(item.get_sub_group(), v652_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1045_data = ir5[1];
              ir5[1] = (v1045_data + (v1035_data * (sycl::select_from_group(item.get_sub_group(), v658_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1051_data = ir5[2];
              ir5[2] = (v1051_data + (v1035_data * (sycl::select_from_group(item.get_sub_group(), v664_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1057_data = ir5[3];
              ir5[3] = (v1057_data + (v1035_data * (sycl::select_from_group(item.get_sub_group(), v670_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1063_data = ir5[4];
              ir5[4] = (v1063_data + (v1035_data * (sycl::select_from_group(item.get_sub_group(), v676_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1069_data = ir5[5];
              ir5[5] = (v1069_data + (v1035_data * (sycl::select_from_group(item.get_sub_group(), v682_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1075_data = ir5[6];
              ir5[6] = (v1075_data + (v1035_data * (sycl::select_from_group(item.get_sub_group(), v688_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1081_data = ir5[7];
              ir5[7] = (v1081_data + (v1035_data * (sycl::select_from_group(item.get_sub_group(), v694_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1083_data = r3[9];
              float v1087_data = ir5[0];
              ir5[0] = (v1087_data + (v1083_data * (sycl::select_from_group(item.get_sub_group(), v652_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1093_data = ir5[1];
              ir5[1] = (v1093_data + (v1083_data * (sycl::select_from_group(item.get_sub_group(), v658_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1099_data = ir5[2];
              ir5[2] = (v1099_data + (v1083_data * (sycl::select_from_group(item.get_sub_group(), v664_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1105_data = ir5[3];
              ir5[3] = (v1105_data + (v1083_data * (sycl::select_from_group(item.get_sub_group(), v670_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1111_data = ir5[4];
              ir5[4] = (v1111_data + (v1083_data * (sycl::select_from_group(item.get_sub_group(), v676_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1117_data = ir5[5];
              ir5[5] = (v1117_data + (v1083_data * (sycl::select_from_group(item.get_sub_group(), v682_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1123_data = ir5[6];
              ir5[6] = (v1123_data + (v1083_data * (sycl::select_from_group(item.get_sub_group(), v688_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1129_data = ir5[7];
              ir5[7] = (v1129_data + (v1083_data * (sycl::select_from_group(item.get_sub_group(), v694_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1131_data = r3[10];
              float v1135_data = ir5[0];
              ir5[0] = (v1135_data + (v1131_data * (sycl::select_from_group(item.get_sub_group(), v652_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1141_data = ir5[1];
              ir5[1] = (v1141_data + (v1131_data * (sycl::select_from_group(item.get_sub_group(), v658_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1147_data = ir5[2];
              ir5[2] = (v1147_data + (v1131_data * (sycl::select_from_group(item.get_sub_group(), v664_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1153_data = ir5[3];
              ir5[3] = (v1153_data + (v1131_data * (sycl::select_from_group(item.get_sub_group(), v670_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1159_data = ir5[4];
              ir5[4] = (v1159_data + (v1131_data * (sycl::select_from_group(item.get_sub_group(), v676_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1165_data = ir5[5];
              ir5[5] = (v1165_data + (v1131_data * (sycl::select_from_group(item.get_sub_group(), v682_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1171_data = ir5[6];
              ir5[6] = (v1171_data + (v1131_data * (sycl::select_from_group(item.get_sub_group(), v688_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1177_data = ir5[7];
              ir5[7] = (v1177_data + (v1131_data * (sycl::select_from_group(item.get_sub_group(), v694_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1179_data = r3[11];
              float v1183_data = ir5[0];
              ir5[0] = (v1183_data + (v1179_data * (sycl::select_from_group(item.get_sub_group(), v652_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1189_data = ir5[1];
              ir5[1] = (v1189_data + (v1179_data * (sycl::select_from_group(item.get_sub_group(), v658_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1195_data = ir5[2];
              ir5[2] = (v1195_data + (v1179_data * (sycl::select_from_group(item.get_sub_group(), v664_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1201_data = ir5[3];
              ir5[3] = (v1201_data + (v1179_data * (sycl::select_from_group(item.get_sub_group(), v670_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1207_data = ir5[4];
              ir5[4] = (v1207_data + (v1179_data * (sycl::select_from_group(item.get_sub_group(), v676_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1213_data = ir5[5];
              ir5[5] = (v1213_data + (v1179_data * (sycl::select_from_group(item.get_sub_group(), v682_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1219_data = ir5[6];
              ir5[6] = (v1219_data + (v1179_data * (sycl::select_from_group(item.get_sub_group(), v688_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1225_data = ir5[7];
              ir5[7] = (v1225_data + (v1179_data * (sycl::select_from_group(item.get_sub_group(), v694_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              // r5 = ir5 + r2
              if (v28_g) {
                #pragma unroll
                for (int32_t v1227_n1 = 0; v1227_n1 < 8; ++v1227_n1) {
                  float v1229_data = ir5[v1227_n1];
                  float v1230_data = r2[v1227_n1];
                  r5[v1227_n1] = (v1230_data + v1229_data);
                }
              }
              float r7[8]{};
              // r7 = load{g>r}(glb_m6);
              if (v28_g) {
                #pragma unroll
                for (int32_t v1233_i1 = 0; v1233_i1 < 8; ++v1233_i1) {
                  float v1238_data = glb_m6[(v27_lead + (v1233_i1 * 12))];
                  r7[v1233_i1] = v1238_data;
                }
              }
              // wait(r6 = load{g>r}(glb_m5););
              float r9[12]{};
              // r9 = load{g>r}(glb_m7);
              if (v28_g) {
                #pragma unroll
                for (int32_t v1241_i1 = 0; v1241_i1 < 12; ++v1241_i1) {
                  float v1246_data = glb_m7[(v27_lead + (v1241_i1 * 12))];
                  r9[v1241_i1] = v1246_data;
                }
              }
              // wait(r7 = load{g>r}(glb_m6););
              float r8[8]{};
              // ir8 = +(r6 * r7)
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir8[8]{};
              float v1250_data = r6[0];
              float v1251_data = r7[0];
              float v1254_data = ir8[0];
              ir8[0] = (v1254_data + (v1250_data * (sycl::select_from_group(item.get_sub_group(), v1251_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1257_data = r7[1];
              float v1260_data = ir8[1];
              ir8[1] = (v1260_data + (v1250_data * (sycl::select_from_group(item.get_sub_group(), v1257_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1263_data = r7[2];
              float v1266_data = ir8[2];
              ir8[2] = (v1266_data + (v1250_data * (sycl::select_from_group(item.get_sub_group(), v1263_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1269_data = r7[3];
              float v1272_data = ir8[3];
              ir8[3] = (v1272_data + (v1250_data * (sycl::select_from_group(item.get_sub_group(), v1269_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1275_data = r7[4];
              float v1278_data = ir8[4];
              ir8[4] = (v1278_data + (v1250_data * (sycl::select_from_group(item.get_sub_group(), v1275_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1281_data = r7[5];
              float v1284_data = ir8[5];
              ir8[5] = (v1284_data + (v1250_data * (sycl::select_from_group(item.get_sub_group(), v1281_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1287_data = r7[6];
              float v1290_data = ir8[6];
              ir8[6] = (v1290_data + (v1250_data * (sycl::select_from_group(item.get_sub_group(), v1287_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1293_data = r7[7];
              float v1296_data = ir8[7];
              ir8[7] = (v1296_data + (v1250_data * (sycl::select_from_group(item.get_sub_group(), v1293_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1298_data = r6[1];
              float v1302_data = ir8[0];
              ir8[0] = (v1302_data + (v1298_data * (sycl::select_from_group(item.get_sub_group(), v1251_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1308_data = ir8[1];
              ir8[1] = (v1308_data + (v1298_data * (sycl::select_from_group(item.get_sub_group(), v1257_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1314_data = ir8[2];
              ir8[2] = (v1314_data + (v1298_data * (sycl::select_from_group(item.get_sub_group(), v1263_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1320_data = ir8[3];
              ir8[3] = (v1320_data + (v1298_data * (sycl::select_from_group(item.get_sub_group(), v1269_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1326_data = ir8[4];
              ir8[4] = (v1326_data + (v1298_data * (sycl::select_from_group(item.get_sub_group(), v1275_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1332_data = ir8[5];
              ir8[5] = (v1332_data + (v1298_data * (sycl::select_from_group(item.get_sub_group(), v1281_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1338_data = ir8[6];
              ir8[6] = (v1338_data + (v1298_data * (sycl::select_from_group(item.get_sub_group(), v1287_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1344_data = ir8[7];
              ir8[7] = (v1344_data + (v1298_data * (sycl::select_from_group(item.get_sub_group(), v1293_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1346_data = r6[2];
              float v1350_data = ir8[0];
              ir8[0] = (v1350_data + (v1346_data * (sycl::select_from_group(item.get_sub_group(), v1251_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1356_data = ir8[1];
              ir8[1] = (v1356_data + (v1346_data * (sycl::select_from_group(item.get_sub_group(), v1257_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1362_data = ir8[2];
              ir8[2] = (v1362_data + (v1346_data * (sycl::select_from_group(item.get_sub_group(), v1263_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1368_data = ir8[3];
              ir8[3] = (v1368_data + (v1346_data * (sycl::select_from_group(item.get_sub_group(), v1269_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1374_data = ir8[4];
              ir8[4] = (v1374_data + (v1346_data * (sycl::select_from_group(item.get_sub_group(), v1275_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1380_data = ir8[5];
              ir8[5] = (v1380_data + (v1346_data * (sycl::select_from_group(item.get_sub_group(), v1281_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1386_data = ir8[6];
              ir8[6] = (v1386_data + (v1346_data * (sycl::select_from_group(item.get_sub_group(), v1287_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1392_data = ir8[7];
              ir8[7] = (v1392_data + (v1346_data * (sycl::select_from_group(item.get_sub_group(), v1293_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1394_data = r6[3];
              float v1398_data = ir8[0];
              ir8[0] = (v1398_data + (v1394_data * (sycl::select_from_group(item.get_sub_group(), v1251_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1404_data = ir8[1];
              ir8[1] = (v1404_data + (v1394_data * (sycl::select_from_group(item.get_sub_group(), v1257_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1410_data = ir8[2];
              ir8[2] = (v1410_data + (v1394_data * (sycl::select_from_group(item.get_sub_group(), v1263_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1416_data = ir8[3];
              ir8[3] = (v1416_data + (v1394_data * (sycl::select_from_group(item.get_sub_group(), v1269_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1422_data = ir8[4];
              ir8[4] = (v1422_data + (v1394_data * (sycl::select_from_group(item.get_sub_group(), v1275_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1428_data = ir8[5];
              ir8[5] = (v1428_data + (v1394_data * (sycl::select_from_group(item.get_sub_group(), v1281_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1434_data = ir8[6];
              ir8[6] = (v1434_data + (v1394_data * (sycl::select_from_group(item.get_sub_group(), v1287_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1440_data = ir8[7];
              ir8[7] = (v1440_data + (v1394_data * (sycl::select_from_group(item.get_sub_group(), v1293_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1442_data = r6[4];
              float v1446_data = ir8[0];
              ir8[0] = (v1446_data + (v1442_data * (sycl::select_from_group(item.get_sub_group(), v1251_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1452_data = ir8[1];
              ir8[1] = (v1452_data + (v1442_data * (sycl::select_from_group(item.get_sub_group(), v1257_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1458_data = ir8[2];
              ir8[2] = (v1458_data + (v1442_data * (sycl::select_from_group(item.get_sub_group(), v1263_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1464_data = ir8[3];
              ir8[3] = (v1464_data + (v1442_data * (sycl::select_from_group(item.get_sub_group(), v1269_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1470_data = ir8[4];
              ir8[4] = (v1470_data + (v1442_data * (sycl::select_from_group(item.get_sub_group(), v1275_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1476_data = ir8[5];
              ir8[5] = (v1476_data + (v1442_data * (sycl::select_from_group(item.get_sub_group(), v1281_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1482_data = ir8[6];
              ir8[6] = (v1482_data + (v1442_data * (sycl::select_from_group(item.get_sub_group(), v1287_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1488_data = ir8[7];
              ir8[7] = (v1488_data + (v1442_data * (sycl::select_from_group(item.get_sub_group(), v1293_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1490_data = r6[5];
              float v1494_data = ir8[0];
              ir8[0] = (v1494_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v1251_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1500_data = ir8[1];
              ir8[1] = (v1500_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v1257_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1506_data = ir8[2];
              ir8[2] = (v1506_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v1263_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1512_data = ir8[3];
              ir8[3] = (v1512_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v1269_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1518_data = ir8[4];
              ir8[4] = (v1518_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v1275_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1524_data = ir8[5];
              ir8[5] = (v1524_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v1281_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1530_data = ir8[6];
              ir8[6] = (v1530_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v1287_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1536_data = ir8[7];
              ir8[7] = (v1536_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v1293_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1538_data = r6[6];
              float v1542_data = ir8[0];
              ir8[0] = (v1542_data + (v1538_data * (sycl::select_from_group(item.get_sub_group(), v1251_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1548_data = ir8[1];
              ir8[1] = (v1548_data + (v1538_data * (sycl::select_from_group(item.get_sub_group(), v1257_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1554_data = ir8[2];
              ir8[2] = (v1554_data + (v1538_data * (sycl::select_from_group(item.get_sub_group(), v1263_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1560_data = ir8[3];
              ir8[3] = (v1560_data + (v1538_data * (sycl::select_from_group(item.get_sub_group(), v1269_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1566_data = ir8[4];
              ir8[4] = (v1566_data + (v1538_data * (sycl::select_from_group(item.get_sub_group(), v1275_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1572_data = ir8[5];
              ir8[5] = (v1572_data + (v1538_data * (sycl::select_from_group(item.get_sub_group(), v1281_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1578_data = ir8[6];
              ir8[6] = (v1578_data + (v1538_data * (sycl::select_from_group(item.get_sub_group(), v1287_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1584_data = ir8[7];
              ir8[7] = (v1584_data + (v1538_data * (sycl::select_from_group(item.get_sub_group(), v1293_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1586_data = r6[7];
              float v1590_data = ir8[0];
              ir8[0] = (v1590_data + (v1586_data * (sycl::select_from_group(item.get_sub_group(), v1251_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1596_data = ir8[1];
              ir8[1] = (v1596_data + (v1586_data * (sycl::select_from_group(item.get_sub_group(), v1257_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1602_data = ir8[2];
              ir8[2] = (v1602_data + (v1586_data * (sycl::select_from_group(item.get_sub_group(), v1263_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1608_data = ir8[3];
              ir8[3] = (v1608_data + (v1586_data * (sycl::select_from_group(item.get_sub_group(), v1269_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1614_data = ir8[4];
              ir8[4] = (v1614_data + (v1586_data * (sycl::select_from_group(item.get_sub_group(), v1275_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1620_data = ir8[5];
              ir8[5] = (v1620_data + (v1586_data * (sycl::select_from_group(item.get_sub_group(), v1281_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1626_data = ir8[6];
              ir8[6] = (v1626_data + (v1586_data * (sycl::select_from_group(item.get_sub_group(), v1287_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1632_data = ir8[7];
              ir8[7] = (v1632_data + (v1586_data * (sycl::select_from_group(item.get_sub_group(), v1293_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1634_data = r6[8];
              float v1638_data = ir8[0];
              ir8[0] = (v1638_data + (v1634_data * (sycl::select_from_group(item.get_sub_group(), v1251_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1644_data = ir8[1];
              ir8[1] = (v1644_data + (v1634_data * (sycl::select_from_group(item.get_sub_group(), v1257_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1650_data = ir8[2];
              ir8[2] = (v1650_data + (v1634_data * (sycl::select_from_group(item.get_sub_group(), v1263_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1656_data = ir8[3];
              ir8[3] = (v1656_data + (v1634_data * (sycl::select_from_group(item.get_sub_group(), v1269_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1662_data = ir8[4];
              ir8[4] = (v1662_data + (v1634_data * (sycl::select_from_group(item.get_sub_group(), v1275_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1668_data = ir8[5];
              ir8[5] = (v1668_data + (v1634_data * (sycl::select_from_group(item.get_sub_group(), v1281_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1674_data = ir8[6];
              ir8[6] = (v1674_data + (v1634_data * (sycl::select_from_group(item.get_sub_group(), v1287_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1680_data = ir8[7];
              ir8[7] = (v1680_data + (v1634_data * (sycl::select_from_group(item.get_sub_group(), v1293_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1682_data = r6[9];
              float v1686_data = ir8[0];
              ir8[0] = (v1686_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1251_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1692_data = ir8[1];
              ir8[1] = (v1692_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1257_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1698_data = ir8[2];
              ir8[2] = (v1698_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1263_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1704_data = ir8[3];
              ir8[3] = (v1704_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1269_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1710_data = ir8[4];
              ir8[4] = (v1710_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1275_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1716_data = ir8[5];
              ir8[5] = (v1716_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1281_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1722_data = ir8[6];
              ir8[6] = (v1722_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1287_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1728_data = ir8[7];
              ir8[7] = (v1728_data + (v1682_data * (sycl::select_from_group(item.get_sub_group(), v1293_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1730_data = r6[10];
              float v1734_data = ir8[0];
              ir8[0] = (v1734_data + (v1730_data * (sycl::select_from_group(item.get_sub_group(), v1251_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1740_data = ir8[1];
              ir8[1] = (v1740_data + (v1730_data * (sycl::select_from_group(item.get_sub_group(), v1257_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1746_data = ir8[2];
              ir8[2] = (v1746_data + (v1730_data * (sycl::select_from_group(item.get_sub_group(), v1263_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1752_data = ir8[3];
              ir8[3] = (v1752_data + (v1730_data * (sycl::select_from_group(item.get_sub_group(), v1269_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1758_data = ir8[4];
              ir8[4] = (v1758_data + (v1730_data * (sycl::select_from_group(item.get_sub_group(), v1275_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1764_data = ir8[5];
              ir8[5] = (v1764_data + (v1730_data * (sycl::select_from_group(item.get_sub_group(), v1281_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1770_data = ir8[6];
              ir8[6] = (v1770_data + (v1730_data * (sycl::select_from_group(item.get_sub_group(), v1287_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1776_data = ir8[7];
              ir8[7] = (v1776_data + (v1730_data * (sycl::select_from_group(item.get_sub_group(), v1293_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1778_data = r6[11];
              float v1782_data = ir8[0];
              ir8[0] = (v1782_data + (v1778_data * (sycl::select_from_group(item.get_sub_group(), v1251_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1788_data = ir8[1];
              ir8[1] = (v1788_data + (v1778_data * (sycl::select_from_group(item.get_sub_group(), v1257_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1794_data = ir8[2];
              ir8[2] = (v1794_data + (v1778_data * (sycl::select_from_group(item.get_sub_group(), v1263_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1800_data = ir8[3];
              ir8[3] = (v1800_data + (v1778_data * (sycl::select_from_group(item.get_sub_group(), v1269_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1806_data = ir8[4];
              ir8[4] = (v1806_data + (v1778_data * (sycl::select_from_group(item.get_sub_group(), v1275_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1812_data = ir8[5];
              ir8[5] = (v1812_data + (v1778_data * (sycl::select_from_group(item.get_sub_group(), v1281_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1818_data = ir8[6];
              ir8[6] = (v1818_data + (v1778_data * (sycl::select_from_group(item.get_sub_group(), v1287_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1824_data = ir8[7];
              ir8[7] = (v1824_data + (v1778_data * (sycl::select_from_group(item.get_sub_group(), v1293_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              // r8 = ir8 + r5
              if (v28_g) {
                #pragma unroll
                for (int32_t v1826_n1 = 0; v1826_n1 < 8; ++v1826_n1) {
                  float v1828_data = ir8[v1826_n1];
                  float v1829_data = r5[v1826_n1];
                  r8[v1826_n1] = (v1829_data + v1828_data);
                }
              }
              float r10[8]{};
              // r10 = load{g>r}(glb_m8);
              if (v28_g) {
                #pragma unroll
                for (int32_t v1832_i1 = 0; v1832_i1 < 8; ++v1832_i1) {
                  float v1837_data = glb_m8[(v27_lead + (v1832_i1 * 12))];
                  r10[v1832_i1] = v1837_data;
                }
              }
              // wait(r9 = load{g>r}(glb_m7););
              // wait(r10 = load{g>r}(glb_m8););
              float r11[8]{};
              // ir11 = +(r9 * r10)
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir11[8]{};
              float v1841_data = r9[0];
              float v1842_data = r10[0];
              float v1845_data = ir11[0];
              ir11[0] = (v1845_data + (v1841_data * (sycl::select_from_group(item.get_sub_group(), v1842_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1848_data = r10[1];
              float v1851_data = ir11[1];
              ir11[1] = (v1851_data + (v1841_data * (sycl::select_from_group(item.get_sub_group(), v1848_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1854_data = r10[2];
              float v1857_data = ir11[2];
              ir11[2] = (v1857_data + (v1841_data * (sycl::select_from_group(item.get_sub_group(), v1854_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1860_data = r10[3];
              float v1863_data = ir11[3];
              ir11[3] = (v1863_data + (v1841_data * (sycl::select_from_group(item.get_sub_group(), v1860_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1866_data = r10[4];
              float v1869_data = ir11[4];
              ir11[4] = (v1869_data + (v1841_data * (sycl::select_from_group(item.get_sub_group(), v1866_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1872_data = r10[5];
              float v1875_data = ir11[5];
              ir11[5] = (v1875_data + (v1841_data * (sycl::select_from_group(item.get_sub_group(), v1872_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1878_data = r10[6];
              float v1881_data = ir11[6];
              ir11[6] = (v1881_data + (v1841_data * (sycl::select_from_group(item.get_sub_group(), v1878_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1884_data = r10[7];
              float v1887_data = ir11[7];
              ir11[7] = (v1887_data + (v1841_data * (sycl::select_from_group(item.get_sub_group(), v1884_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1889_data = r9[1];
              float v1893_data = ir11[0];
              ir11[0] = (v1893_data + (v1889_data * (sycl::select_from_group(item.get_sub_group(), v1842_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1899_data = ir11[1];
              ir11[1] = (v1899_data + (v1889_data * (sycl::select_from_group(item.get_sub_group(), v1848_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1905_data = ir11[2];
              ir11[2] = (v1905_data + (v1889_data * (sycl::select_from_group(item.get_sub_group(), v1854_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1911_data = ir11[3];
              ir11[3] = (v1911_data + (v1889_data * (sycl::select_from_group(item.get_sub_group(), v1860_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1917_data = ir11[4];
              ir11[4] = (v1917_data + (v1889_data * (sycl::select_from_group(item.get_sub_group(), v1866_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1923_data = ir11[5];
              ir11[5] = (v1923_data + (v1889_data * (sycl::select_from_group(item.get_sub_group(), v1872_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1929_data = ir11[6];
              ir11[6] = (v1929_data + (v1889_data * (sycl::select_from_group(item.get_sub_group(), v1878_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1935_data = ir11[7];
              ir11[7] = (v1935_data + (v1889_data * (sycl::select_from_group(item.get_sub_group(), v1884_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1937_data = r9[2];
              float v1941_data = ir11[0];
              ir11[0] = (v1941_data + (v1937_data * (sycl::select_from_group(item.get_sub_group(), v1842_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1947_data = ir11[1];
              ir11[1] = (v1947_data + (v1937_data * (sycl::select_from_group(item.get_sub_group(), v1848_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1953_data = ir11[2];
              ir11[2] = (v1953_data + (v1937_data * (sycl::select_from_group(item.get_sub_group(), v1854_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1959_data = ir11[3];
              ir11[3] = (v1959_data + (v1937_data * (sycl::select_from_group(item.get_sub_group(), v1860_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1965_data = ir11[4];
              ir11[4] = (v1965_data + (v1937_data * (sycl::select_from_group(item.get_sub_group(), v1866_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1971_data = ir11[5];
              ir11[5] = (v1971_data + (v1937_data * (sycl::select_from_group(item.get_sub_group(), v1872_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1977_data = ir11[6];
              ir11[6] = (v1977_data + (v1937_data * (sycl::select_from_group(item.get_sub_group(), v1878_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1983_data = ir11[7];
              ir11[7] = (v1983_data + (v1937_data * (sycl::select_from_group(item.get_sub_group(), v1884_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1985_data = r9[3];
              float v1989_data = ir11[0];
              ir11[0] = (v1989_data + (v1985_data * (sycl::select_from_group(item.get_sub_group(), v1842_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1995_data = ir11[1];
              ir11[1] = (v1995_data + (v1985_data * (sycl::select_from_group(item.get_sub_group(), v1848_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v2001_data = ir11[2];
              ir11[2] = (v2001_data + (v1985_data * (sycl::select_from_group(item.get_sub_group(), v1854_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v2007_data = ir11[3];
              ir11[3] = (v2007_data + (v1985_data * (sycl::select_from_group(item.get_sub_group(), v1860_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v2013_data = ir11[4];
              ir11[4] = (v2013_data + (v1985_data * (sycl::select_from_group(item.get_sub_group(), v1866_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v2019_data = ir11[5];
              ir11[5] = (v2019_data + (v1985_data * (sycl::select_from_group(item.get_sub_group(), v1872_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v2025_data = ir11[6];
              ir11[6] = (v2025_data + (v1985_data * (sycl::select_from_group(item.get_sub_group(), v1878_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v2031_data = ir11[7];
              ir11[7] = (v2031_data + (v1985_data * (sycl::select_from_group(item.get_sub_group(), v1884_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v2033_data = r9[4];
              float v2037_data = ir11[0];
              ir11[0] = (v2037_data + (v2033_data * (sycl::select_from_group(item.get_sub_group(), v1842_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v2043_data = ir11[1];
              ir11[1] = (v2043_data + (v2033_data * (sycl::select_from_group(item.get_sub_group(), v1848_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v2049_data = ir11[2];
              ir11[2] = (v2049_data + (v2033_data * (sycl::select_from_group(item.get_sub_group(), v1854_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v2055_data = ir11[3];
              ir11[3] = (v2055_data + (v2033_data * (sycl::select_from_group(item.get_sub_group(), v1860_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v2061_data = ir11[4];
              ir11[4] = (v2061_data + (v2033_data * (sycl::select_from_group(item.get_sub_group(), v1866_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v2067_data = ir11[5];
              ir11[5] = (v2067_data + (v2033_data * (sycl::select_from_group(item.get_sub_group(), v1872_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v2073_data = ir11[6];
              ir11[6] = (v2073_data + (v2033_data * (sycl::select_from_group(item.get_sub_group(), v1878_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v2079_data = ir11[7];
              ir11[7] = (v2079_data + (v2033_data * (sycl::select_from_group(item.get_sub_group(), v1884_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v2081_data = r9[5];
              float v2085_data = ir11[0];
              ir11[0] = (v2085_data + (v2081_data * (sycl::select_from_group(item.get_sub_group(), v1842_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v2091_data = ir11[1];
              ir11[1] = (v2091_data + (v2081_data * (sycl::select_from_group(item.get_sub_group(), v1848_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v2097_data = ir11[2];
              ir11[2] = (v2097_data + (v2081_data * (sycl::select_from_group(item.get_sub_group(), v1854_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v2103_data = ir11[3];
              ir11[3] = (v2103_data + (v2081_data * (sycl::select_from_group(item.get_sub_group(), v1860_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v2109_data = ir11[4];
              ir11[4] = (v2109_data + (v2081_data * (sycl::select_from_group(item.get_sub_group(), v1866_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v2115_data = ir11[5];
              ir11[5] = (v2115_data + (v2081_data * (sycl::select_from_group(item.get_sub_group(), v1872_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v2121_data = ir11[6];
              ir11[6] = (v2121_data + (v2081_data * (sycl::select_from_group(item.get_sub_group(), v1878_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v2127_data = ir11[7];
              ir11[7] = (v2127_data + (v2081_data * (sycl::select_from_group(item.get_sub_group(), v1884_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v2129_data = r9[6];
              float v2133_data = ir11[0];
              ir11[0] = (v2133_data + (v2129_data * (sycl::select_from_group(item.get_sub_group(), v1842_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v2139_data = ir11[1];
              ir11[1] = (v2139_data + (v2129_data * (sycl::select_from_group(item.get_sub_group(), v1848_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v2145_data = ir11[2];
              ir11[2] = (v2145_data + (v2129_data * (sycl::select_from_group(item.get_sub_group(), v1854_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v2151_data = ir11[3];
              ir11[3] = (v2151_data + (v2129_data * (sycl::select_from_group(item.get_sub_group(), v1860_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v2157_data = ir11[4];
              ir11[4] = (v2157_data + (v2129_data * (sycl::select_from_group(item.get_sub_group(), v1866_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v2163_data = ir11[5];
              ir11[5] = (v2163_data + (v2129_data * (sycl::select_from_group(item.get_sub_group(), v1872_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v2169_data = ir11[6];
              ir11[6] = (v2169_data + (v2129_data * (sycl::select_from_group(item.get_sub_group(), v1878_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v2175_data = ir11[7];
              ir11[7] = (v2175_data + (v2129_data * (sycl::select_from_group(item.get_sub_group(), v1884_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v2177_data = r9[7];
              float v2181_data = ir11[0];
              ir11[0] = (v2181_data + (v2177_data * (sycl::select_from_group(item.get_sub_group(), v1842_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v2187_data = ir11[1];
              ir11[1] = (v2187_data + (v2177_data * (sycl::select_from_group(item.get_sub_group(), v1848_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v2193_data = ir11[2];
              ir11[2] = (v2193_data + (v2177_data * (sycl::select_from_group(item.get_sub_group(), v1854_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v2199_data = ir11[3];
              ir11[3] = (v2199_data + (v2177_data * (sycl::select_from_group(item.get_sub_group(), v1860_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v2205_data = ir11[4];
              ir11[4] = (v2205_data + (v2177_data * (sycl::select_from_group(item.get_sub_group(), v1866_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v2211_data = ir11[5];
              ir11[5] = (v2211_data + (v2177_data * (sycl::select_from_group(item.get_sub_group(), v1872_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v2217_data = ir11[6];
              ir11[6] = (v2217_data + (v2177_data * (sycl::select_from_group(item.get_sub_group(), v1878_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v2223_data = ir11[7];
              ir11[7] = (v2223_data + (v2177_data * (sycl::select_from_group(item.get_sub_group(), v1884_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v2225_data = r9[8];
              float v2229_data = ir11[0];
              ir11[0] = (v2229_data + (v2225_data * (sycl::select_from_group(item.get_sub_group(), v1842_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v2235_data = ir11[1];
              ir11[1] = (v2235_data + (v2225_data * (sycl::select_from_group(item.get_sub_group(), v1848_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v2241_data = ir11[2];
              ir11[2] = (v2241_data + (v2225_data * (sycl::select_from_group(item.get_sub_group(), v1854_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v2247_data = ir11[3];
              ir11[3] = (v2247_data + (v2225_data * (sycl::select_from_group(item.get_sub_group(), v1860_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v2253_data = ir11[4];
              ir11[4] = (v2253_data + (v2225_data * (sycl::select_from_group(item.get_sub_group(), v1866_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v2259_data = ir11[5];
              ir11[5] = (v2259_data + (v2225_data * (sycl::select_from_group(item.get_sub_group(), v1872_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v2265_data = ir11[6];
              ir11[6] = (v2265_data + (v2225_data * (sycl::select_from_group(item.get_sub_group(), v1878_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v2271_data = ir11[7];
              ir11[7] = (v2271_data + (v2225_data * (sycl::select_from_group(item.get_sub_group(), v1884_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v2273_data = r9[9];
              float v2277_data = ir11[0];
              ir11[0] = (v2277_data + (v2273_data * (sycl::select_from_group(item.get_sub_group(), v1842_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v2283_data = ir11[1];
              ir11[1] = (v2283_data + (v2273_data * (sycl::select_from_group(item.get_sub_group(), v1848_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v2289_data = ir11[2];
              ir11[2] = (v2289_data + (v2273_data * (sycl::select_from_group(item.get_sub_group(), v1854_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v2295_data = ir11[3];
              ir11[3] = (v2295_data + (v2273_data * (sycl::select_from_group(item.get_sub_group(), v1860_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v2301_data = ir11[4];
              ir11[4] = (v2301_data + (v2273_data * (sycl::select_from_group(item.get_sub_group(), v1866_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v2307_data = ir11[5];
              ir11[5] = (v2307_data + (v2273_data * (sycl::select_from_group(item.get_sub_group(), v1872_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v2313_data = ir11[6];
              ir11[6] = (v2313_data + (v2273_data * (sycl::select_from_group(item.get_sub_group(), v1878_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v2319_data = ir11[7];
              ir11[7] = (v2319_data + (v2273_data * (sycl::select_from_group(item.get_sub_group(), v1884_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v2321_data = r9[10];
              float v2325_data = ir11[0];
              ir11[0] = (v2325_data + (v2321_data * (sycl::select_from_group(item.get_sub_group(), v1842_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v2331_data = ir11[1];
              ir11[1] = (v2331_data + (v2321_data * (sycl::select_from_group(item.get_sub_group(), v1848_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v2337_data = ir11[2];
              ir11[2] = (v2337_data + (v2321_data * (sycl::select_from_group(item.get_sub_group(), v1854_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v2343_data = ir11[3];
              ir11[3] = (v2343_data + (v2321_data * (sycl::select_from_group(item.get_sub_group(), v1860_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v2349_data = ir11[4];
              ir11[4] = (v2349_data + (v2321_data * (sycl::select_from_group(item.get_sub_group(), v1866_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v2355_data = ir11[5];
              ir11[5] = (v2355_data + (v2321_data * (sycl::select_from_group(item.get_sub_group(), v1872_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v2361_data = ir11[6];
              ir11[6] = (v2361_data + (v2321_data * (sycl::select_from_group(item.get_sub_group(), v1878_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v2367_data = ir11[7];
              ir11[7] = (v2367_data + (v2321_data * (sycl::select_from_group(item.get_sub_group(), v1884_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v2369_data = r9[11];
              float v2373_data = ir11[0];
              ir11[0] = (v2373_data + (v2369_data * (sycl::select_from_group(item.get_sub_group(), v1842_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v2379_data = ir11[1];
              ir11[1] = (v2379_data + (v2369_data * (sycl::select_from_group(item.get_sub_group(), v1848_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v2385_data = ir11[2];
              ir11[2] = (v2385_data + (v2369_data * (sycl::select_from_group(item.get_sub_group(), v1854_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v2391_data = ir11[3];
              ir11[3] = (v2391_data + (v2369_data * (sycl::select_from_group(item.get_sub_group(), v1860_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v2397_data = ir11[4];
              ir11[4] = (v2397_data + (v2369_data * (sycl::select_from_group(item.get_sub_group(), v1866_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v2403_data = ir11[5];
              ir11[5] = (v2403_data + (v2369_data * (sycl::select_from_group(item.get_sub_group(), v1872_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v2409_data = ir11[6];
              ir11[6] = (v2409_data + (v2369_data * (sycl::select_from_group(item.get_sub_group(), v1878_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v2415_data = ir11[7];
              ir11[7] = (v2415_data + (v2369_data * (sycl::select_from_group(item.get_sub_group(), v1884_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              // r11 = ir11 + r8
              if (v28_g) {
                #pragma unroll
                for (int32_t v2417_n1 = 0; v2417_n1 < 8; ++v2417_n1) {
                  float v2419_data = ir11[v2417_n1];
                  float v2420_data = r8[v2417_n1];
                  r11[v2417_n1] = (v2420_data + v2419_data);
                }
              }
              // glb_m0 = store{r>g}(r11);
              if (v28_g) {
                #pragma unroll
                for (int32_t v2422_i1 = 0; v2422_i1 < 8; ++v2422_i1) {
                  float v2424_data = r11[v2422_i1];
                  glb_m0[(v27_lead + (v2422_i1 * 12))] = v2424_data;
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

