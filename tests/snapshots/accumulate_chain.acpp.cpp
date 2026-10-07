// === base name ===
kernel_71be049c856cf22c

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_71be049c856cf22c = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_71be049c856cf22c(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_71be049c856cf22c(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_71be049c856cf22c(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_71be049c856cf22c(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_71be049c856cf22c(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_71be049c856cf22c(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, m6, m6_extraOffset, m7, m7_extraOffset, m8, m8_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_71be049c856cf22c(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0) {
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
              float r3[12]{};
              // r3 = load{g>r}(glb_m3);
              if (v28_g) {
                #pragma unroll
                for (int32_t v626_i1 = 0; v626_i1 < 12; ++v626_i1) {
                  float v631_data = glb_m3[(v27_lead + (v626_i1 * 12))];
                  r3[v626_i1] = v631_data;
                }
              }
              float r2[8]{};
              // ir2 = +(r0 * r1)
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir2[8]{};
              float v46_data = r0[0];
              float v47_data = r1[0];
              float v50_data = ir2[0];
              ir2[0] = (v50_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v53_data = r1[1];
              float v56_data = ir2[1];
              ir2[1] = (v56_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v59_data = r1[2];
              float v62_data = ir2[2];
              ir2[2] = (v62_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v65_data = r1[3];
              float v68_data = ir2[3];
              ir2[3] = (v68_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v71_data = r1[4];
              float v74_data = ir2[4];
              ir2[4] = (v74_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v77_data = r1[5];
              float v80_data = ir2[5];
              ir2[5] = (v80_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v83_data = r1[6];
              float v86_data = ir2[6];
              ir2[6] = (v86_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v89_data = r1[7];
              float v92_data = ir2[7];
              ir2[7] = (v92_data + (v46_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v94_data = r0[1];
              float v98_data = ir2[0];
              ir2[0] = (v98_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v104_data = ir2[1];
              ir2[1] = (v104_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v110_data = ir2[2];
              ir2[2] = (v110_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v116_data = ir2[3];
              ir2[3] = (v116_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v122_data = ir2[4];
              ir2[4] = (v122_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v128_data = ir2[5];
              ir2[5] = (v128_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v134_data = ir2[6];
              ir2[6] = (v134_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v140_data = ir2[7];
              ir2[7] = (v140_data + (v94_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v142_data = r0[2];
              float v146_data = ir2[0];
              ir2[0] = (v146_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v152_data = ir2[1];
              ir2[1] = (v152_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v158_data = ir2[2];
              ir2[2] = (v158_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v164_data = ir2[3];
              ir2[3] = (v164_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v170_data = ir2[4];
              ir2[4] = (v170_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v176_data = ir2[5];
              ir2[5] = (v176_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v182_data = ir2[6];
              ir2[6] = (v182_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v188_data = ir2[7];
              ir2[7] = (v188_data + (v142_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v190_data = r0[3];
              float v194_data = ir2[0];
              ir2[0] = (v194_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v200_data = ir2[1];
              ir2[1] = (v200_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v206_data = ir2[2];
              ir2[2] = (v206_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v212_data = ir2[3];
              ir2[3] = (v212_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v218_data = ir2[4];
              ir2[4] = (v218_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v224_data = ir2[5];
              ir2[5] = (v224_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v230_data = ir2[6];
              ir2[6] = (v230_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v236_data = ir2[7];
              ir2[7] = (v236_data + (v190_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v238_data = r0[4];
              float v242_data = ir2[0];
              ir2[0] = (v242_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v248_data = ir2[1];
              ir2[1] = (v248_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v254_data = ir2[2];
              ir2[2] = (v254_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v260_data = ir2[3];
              ir2[3] = (v260_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v266_data = ir2[4];
              ir2[4] = (v266_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v272_data = ir2[5];
              ir2[5] = (v272_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v278_data = ir2[6];
              ir2[6] = (v278_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v284_data = ir2[7];
              ir2[7] = (v284_data + (v238_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v286_data = r0[5];
              float v290_data = ir2[0];
              ir2[0] = (v290_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v296_data = ir2[1];
              ir2[1] = (v296_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v302_data = ir2[2];
              ir2[2] = (v302_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v308_data = ir2[3];
              ir2[3] = (v308_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v314_data = ir2[4];
              ir2[4] = (v314_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v320_data = ir2[5];
              ir2[5] = (v320_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v326_data = ir2[6];
              ir2[6] = (v326_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v332_data = ir2[7];
              ir2[7] = (v332_data + (v286_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v334_data = r0[6];
              float v338_data = ir2[0];
              ir2[0] = (v338_data + (v334_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v344_data = ir2[1];
              ir2[1] = (v344_data + (v334_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v350_data = ir2[2];
              ir2[2] = (v350_data + (v334_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v356_data = ir2[3];
              ir2[3] = (v356_data + (v334_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v362_data = ir2[4];
              ir2[4] = (v362_data + (v334_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v368_data = ir2[5];
              ir2[5] = (v368_data + (v334_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v374_data = ir2[6];
              ir2[6] = (v374_data + (v334_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v380_data = ir2[7];
              ir2[7] = (v380_data + (v334_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v382_data = r0[7];
              float v386_data = ir2[0];
              ir2[0] = (v386_data + (v382_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v392_data = ir2[1];
              ir2[1] = (v392_data + (v382_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v398_data = ir2[2];
              ir2[2] = (v398_data + (v382_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v404_data = ir2[3];
              ir2[3] = (v404_data + (v382_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v410_data = ir2[4];
              ir2[4] = (v410_data + (v382_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v416_data = ir2[5];
              ir2[5] = (v416_data + (v382_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v422_data = ir2[6];
              ir2[6] = (v422_data + (v382_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v428_data = ir2[7];
              ir2[7] = (v428_data + (v382_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v430_data = r0[8];
              float v434_data = ir2[0];
              ir2[0] = (v434_data + (v430_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v440_data = ir2[1];
              ir2[1] = (v440_data + (v430_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v446_data = ir2[2];
              ir2[2] = (v446_data + (v430_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v452_data = ir2[3];
              ir2[3] = (v452_data + (v430_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v458_data = ir2[4];
              ir2[4] = (v458_data + (v430_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v464_data = ir2[5];
              ir2[5] = (v464_data + (v430_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v470_data = ir2[6];
              ir2[6] = (v470_data + (v430_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v476_data = ir2[7];
              ir2[7] = (v476_data + (v430_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v478_data = r0[9];
              float v482_data = ir2[0];
              ir2[0] = (v482_data + (v478_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v488_data = ir2[1];
              ir2[1] = (v488_data + (v478_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v494_data = ir2[2];
              ir2[2] = (v494_data + (v478_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v500_data = ir2[3];
              ir2[3] = (v500_data + (v478_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v506_data = ir2[4];
              ir2[4] = (v506_data + (v478_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v512_data = ir2[5];
              ir2[5] = (v512_data + (v478_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v518_data = ir2[6];
              ir2[6] = (v518_data + (v478_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v524_data = ir2[7];
              ir2[7] = (v524_data + (v478_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v526_data = r0[10];
              float v530_data = ir2[0];
              ir2[0] = (v530_data + (v526_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v536_data = ir2[1];
              ir2[1] = (v536_data + (v526_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v542_data = ir2[2];
              ir2[2] = (v542_data + (v526_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v548_data = ir2[3];
              ir2[3] = (v548_data + (v526_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v554_data = ir2[4];
              ir2[4] = (v554_data + (v526_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v560_data = ir2[5];
              ir2[5] = (v560_data + (v526_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v566_data = ir2[6];
              ir2[6] = (v566_data + (v526_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v572_data = ir2[7];
              ir2[7] = (v572_data + (v526_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v574_data = r0[11];
              float v578_data = ir2[0];
              ir2[0] = (v578_data + (v574_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v584_data = ir2[1];
              ir2[1] = (v584_data + (v574_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v590_data = ir2[2];
              ir2[2] = (v590_data + (v574_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v596_data = ir2[3];
              ir2[3] = (v596_data + (v574_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v602_data = ir2[4];
              ir2[4] = (v602_data + (v574_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v608_data = ir2[5];
              ir2[5] = (v608_data + (v574_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v614_data = ir2[6];
              ir2[6] = (v614_data + (v574_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v620_data = ir2[7];
              ir2[7] = (v620_data + (v574_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              // r2 = ir2
              if (v28_g) {
                #pragma unroll
                for (int32_t v622_n1 = 0; v622_n1 < 8; ++v622_n1) {
                  float v624_data = ir2[v622_n1];
                  r2[v622_n1] = v624_data;
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
              float r6[12]{};
              // r6 = load{g>r}(glb_m5);
              if (v28_g) {
                #pragma unroll
                for (int32_t v1225_i1 = 0; v1225_i1 < 12; ++v1225_i1) {
                  float v1230_data = glb_m5[(v27_lead + (v1225_i1 * 12))];
                  r6[v1225_i1] = v1230_data;
                }
              }
              float r5[8]{};
              // ir5 = +(r3 * r4)
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir5[8]{};
              float v643_data = r3[0];
              float v644_data = r4[0];
              float v647_data = ir5[0];
              ir5[0] = (v647_data + (v643_data * (sycl::select_from_group(item.get_sub_group(), v644_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v650_data = r4[1];
              float v653_data = ir5[1];
              ir5[1] = (v653_data + (v643_data * (sycl::select_from_group(item.get_sub_group(), v650_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v656_data = r4[2];
              float v659_data = ir5[2];
              ir5[2] = (v659_data + (v643_data * (sycl::select_from_group(item.get_sub_group(), v656_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v662_data = r4[3];
              float v665_data = ir5[3];
              ir5[3] = (v665_data + (v643_data * (sycl::select_from_group(item.get_sub_group(), v662_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v668_data = r4[4];
              float v671_data = ir5[4];
              ir5[4] = (v671_data + (v643_data * (sycl::select_from_group(item.get_sub_group(), v668_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v674_data = r4[5];
              float v677_data = ir5[5];
              ir5[5] = (v677_data + (v643_data * (sycl::select_from_group(item.get_sub_group(), v674_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v680_data = r4[6];
              float v683_data = ir5[6];
              ir5[6] = (v683_data + (v643_data * (sycl::select_from_group(item.get_sub_group(), v680_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v686_data = r4[7];
              float v689_data = ir5[7];
              ir5[7] = (v689_data + (v643_data * (sycl::select_from_group(item.get_sub_group(), v686_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v691_data = r3[1];
              float v695_data = ir5[0];
              ir5[0] = (v695_data + (v691_data * (sycl::select_from_group(item.get_sub_group(), v644_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v701_data = ir5[1];
              ir5[1] = (v701_data + (v691_data * (sycl::select_from_group(item.get_sub_group(), v650_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v707_data = ir5[2];
              ir5[2] = (v707_data + (v691_data * (sycl::select_from_group(item.get_sub_group(), v656_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v713_data = ir5[3];
              ir5[3] = (v713_data + (v691_data * (sycl::select_from_group(item.get_sub_group(), v662_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v719_data = ir5[4];
              ir5[4] = (v719_data + (v691_data * (sycl::select_from_group(item.get_sub_group(), v668_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v725_data = ir5[5];
              ir5[5] = (v725_data + (v691_data * (sycl::select_from_group(item.get_sub_group(), v674_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v731_data = ir5[6];
              ir5[6] = (v731_data + (v691_data * (sycl::select_from_group(item.get_sub_group(), v680_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v737_data = ir5[7];
              ir5[7] = (v737_data + (v691_data * (sycl::select_from_group(item.get_sub_group(), v686_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v739_data = r3[2];
              float v743_data = ir5[0];
              ir5[0] = (v743_data + (v739_data * (sycl::select_from_group(item.get_sub_group(), v644_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v749_data = ir5[1];
              ir5[1] = (v749_data + (v739_data * (sycl::select_from_group(item.get_sub_group(), v650_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v755_data = ir5[2];
              ir5[2] = (v755_data + (v739_data * (sycl::select_from_group(item.get_sub_group(), v656_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v761_data = ir5[3];
              ir5[3] = (v761_data + (v739_data * (sycl::select_from_group(item.get_sub_group(), v662_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v767_data = ir5[4];
              ir5[4] = (v767_data + (v739_data * (sycl::select_from_group(item.get_sub_group(), v668_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v773_data = ir5[5];
              ir5[5] = (v773_data + (v739_data * (sycl::select_from_group(item.get_sub_group(), v674_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v779_data = ir5[6];
              ir5[6] = (v779_data + (v739_data * (sycl::select_from_group(item.get_sub_group(), v680_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v785_data = ir5[7];
              ir5[7] = (v785_data + (v739_data * (sycl::select_from_group(item.get_sub_group(), v686_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v787_data = r3[3];
              float v791_data = ir5[0];
              ir5[0] = (v791_data + (v787_data * (sycl::select_from_group(item.get_sub_group(), v644_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v797_data = ir5[1];
              ir5[1] = (v797_data + (v787_data * (sycl::select_from_group(item.get_sub_group(), v650_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v803_data = ir5[2];
              ir5[2] = (v803_data + (v787_data * (sycl::select_from_group(item.get_sub_group(), v656_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v809_data = ir5[3];
              ir5[3] = (v809_data + (v787_data * (sycl::select_from_group(item.get_sub_group(), v662_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v815_data = ir5[4];
              ir5[4] = (v815_data + (v787_data * (sycl::select_from_group(item.get_sub_group(), v668_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v821_data = ir5[5];
              ir5[5] = (v821_data + (v787_data * (sycl::select_from_group(item.get_sub_group(), v674_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v827_data = ir5[6];
              ir5[6] = (v827_data + (v787_data * (sycl::select_from_group(item.get_sub_group(), v680_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v833_data = ir5[7];
              ir5[7] = (v833_data + (v787_data * (sycl::select_from_group(item.get_sub_group(), v686_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v835_data = r3[4];
              float v839_data = ir5[0];
              ir5[0] = (v839_data + (v835_data * (sycl::select_from_group(item.get_sub_group(), v644_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v845_data = ir5[1];
              ir5[1] = (v845_data + (v835_data * (sycl::select_from_group(item.get_sub_group(), v650_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v851_data = ir5[2];
              ir5[2] = (v851_data + (v835_data * (sycl::select_from_group(item.get_sub_group(), v656_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v857_data = ir5[3];
              ir5[3] = (v857_data + (v835_data * (sycl::select_from_group(item.get_sub_group(), v662_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v863_data = ir5[4];
              ir5[4] = (v863_data + (v835_data * (sycl::select_from_group(item.get_sub_group(), v668_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v869_data = ir5[5];
              ir5[5] = (v869_data + (v835_data * (sycl::select_from_group(item.get_sub_group(), v674_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v875_data = ir5[6];
              ir5[6] = (v875_data + (v835_data * (sycl::select_from_group(item.get_sub_group(), v680_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v881_data = ir5[7];
              ir5[7] = (v881_data + (v835_data * (sycl::select_from_group(item.get_sub_group(), v686_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v883_data = r3[5];
              float v887_data = ir5[0];
              ir5[0] = (v887_data + (v883_data * (sycl::select_from_group(item.get_sub_group(), v644_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v893_data = ir5[1];
              ir5[1] = (v893_data + (v883_data * (sycl::select_from_group(item.get_sub_group(), v650_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v899_data = ir5[2];
              ir5[2] = (v899_data + (v883_data * (sycl::select_from_group(item.get_sub_group(), v656_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v905_data = ir5[3];
              ir5[3] = (v905_data + (v883_data * (sycl::select_from_group(item.get_sub_group(), v662_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v911_data = ir5[4];
              ir5[4] = (v911_data + (v883_data * (sycl::select_from_group(item.get_sub_group(), v668_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v917_data = ir5[5];
              ir5[5] = (v917_data + (v883_data * (sycl::select_from_group(item.get_sub_group(), v674_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v923_data = ir5[6];
              ir5[6] = (v923_data + (v883_data * (sycl::select_from_group(item.get_sub_group(), v680_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v929_data = ir5[7];
              ir5[7] = (v929_data + (v883_data * (sycl::select_from_group(item.get_sub_group(), v686_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v931_data = r3[6];
              float v935_data = ir5[0];
              ir5[0] = (v935_data + (v931_data * (sycl::select_from_group(item.get_sub_group(), v644_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v941_data = ir5[1];
              ir5[1] = (v941_data + (v931_data * (sycl::select_from_group(item.get_sub_group(), v650_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v947_data = ir5[2];
              ir5[2] = (v947_data + (v931_data * (sycl::select_from_group(item.get_sub_group(), v656_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v953_data = ir5[3];
              ir5[3] = (v953_data + (v931_data * (sycl::select_from_group(item.get_sub_group(), v662_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v959_data = ir5[4];
              ir5[4] = (v959_data + (v931_data * (sycl::select_from_group(item.get_sub_group(), v668_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v965_data = ir5[5];
              ir5[5] = (v965_data + (v931_data * (sycl::select_from_group(item.get_sub_group(), v674_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v971_data = ir5[6];
              ir5[6] = (v971_data + (v931_data * (sycl::select_from_group(item.get_sub_group(), v680_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v977_data = ir5[7];
              ir5[7] = (v977_data + (v931_data * (sycl::select_from_group(item.get_sub_group(), v686_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v979_data = r3[7];
              float v983_data = ir5[0];
              ir5[0] = (v983_data + (v979_data * (sycl::select_from_group(item.get_sub_group(), v644_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v989_data = ir5[1];
              ir5[1] = (v989_data + (v979_data * (sycl::select_from_group(item.get_sub_group(), v650_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v995_data = ir5[2];
              ir5[2] = (v995_data + (v979_data * (sycl::select_from_group(item.get_sub_group(), v656_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1001_data = ir5[3];
              ir5[3] = (v1001_data + (v979_data * (sycl::select_from_group(item.get_sub_group(), v662_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1007_data = ir5[4];
              ir5[4] = (v1007_data + (v979_data * (sycl::select_from_group(item.get_sub_group(), v668_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1013_data = ir5[5];
              ir5[5] = (v1013_data + (v979_data * (sycl::select_from_group(item.get_sub_group(), v674_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1019_data = ir5[6];
              ir5[6] = (v1019_data + (v979_data * (sycl::select_from_group(item.get_sub_group(), v680_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1025_data = ir5[7];
              ir5[7] = (v1025_data + (v979_data * (sycl::select_from_group(item.get_sub_group(), v686_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1027_data = r3[8];
              float v1031_data = ir5[0];
              ir5[0] = (v1031_data + (v1027_data * (sycl::select_from_group(item.get_sub_group(), v644_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1037_data = ir5[1];
              ir5[1] = (v1037_data + (v1027_data * (sycl::select_from_group(item.get_sub_group(), v650_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1043_data = ir5[2];
              ir5[2] = (v1043_data + (v1027_data * (sycl::select_from_group(item.get_sub_group(), v656_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1049_data = ir5[3];
              ir5[3] = (v1049_data + (v1027_data * (sycl::select_from_group(item.get_sub_group(), v662_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1055_data = ir5[4];
              ir5[4] = (v1055_data + (v1027_data * (sycl::select_from_group(item.get_sub_group(), v668_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1061_data = ir5[5];
              ir5[5] = (v1061_data + (v1027_data * (sycl::select_from_group(item.get_sub_group(), v674_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1067_data = ir5[6];
              ir5[6] = (v1067_data + (v1027_data * (sycl::select_from_group(item.get_sub_group(), v680_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1073_data = ir5[7];
              ir5[7] = (v1073_data + (v1027_data * (sycl::select_from_group(item.get_sub_group(), v686_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1075_data = r3[9];
              float v1079_data = ir5[0];
              ir5[0] = (v1079_data + (v1075_data * (sycl::select_from_group(item.get_sub_group(), v644_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1085_data = ir5[1];
              ir5[1] = (v1085_data + (v1075_data * (sycl::select_from_group(item.get_sub_group(), v650_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1091_data = ir5[2];
              ir5[2] = (v1091_data + (v1075_data * (sycl::select_from_group(item.get_sub_group(), v656_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1097_data = ir5[3];
              ir5[3] = (v1097_data + (v1075_data * (sycl::select_from_group(item.get_sub_group(), v662_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1103_data = ir5[4];
              ir5[4] = (v1103_data + (v1075_data * (sycl::select_from_group(item.get_sub_group(), v668_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1109_data = ir5[5];
              ir5[5] = (v1109_data + (v1075_data * (sycl::select_from_group(item.get_sub_group(), v674_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1115_data = ir5[6];
              ir5[6] = (v1115_data + (v1075_data * (sycl::select_from_group(item.get_sub_group(), v680_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1121_data = ir5[7];
              ir5[7] = (v1121_data + (v1075_data * (sycl::select_from_group(item.get_sub_group(), v686_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1123_data = r3[10];
              float v1127_data = ir5[0];
              ir5[0] = (v1127_data + (v1123_data * (sycl::select_from_group(item.get_sub_group(), v644_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1133_data = ir5[1];
              ir5[1] = (v1133_data + (v1123_data * (sycl::select_from_group(item.get_sub_group(), v650_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1139_data = ir5[2];
              ir5[2] = (v1139_data + (v1123_data * (sycl::select_from_group(item.get_sub_group(), v656_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1145_data = ir5[3];
              ir5[3] = (v1145_data + (v1123_data * (sycl::select_from_group(item.get_sub_group(), v662_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1151_data = ir5[4];
              ir5[4] = (v1151_data + (v1123_data * (sycl::select_from_group(item.get_sub_group(), v668_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1157_data = ir5[5];
              ir5[5] = (v1157_data + (v1123_data * (sycl::select_from_group(item.get_sub_group(), v674_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1163_data = ir5[6];
              ir5[6] = (v1163_data + (v1123_data * (sycl::select_from_group(item.get_sub_group(), v680_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1169_data = ir5[7];
              ir5[7] = (v1169_data + (v1123_data * (sycl::select_from_group(item.get_sub_group(), v686_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1171_data = r3[11];
              float v1175_data = ir5[0];
              ir5[0] = (v1175_data + (v1171_data * (sycl::select_from_group(item.get_sub_group(), v644_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1181_data = ir5[1];
              ir5[1] = (v1181_data + (v1171_data * (sycl::select_from_group(item.get_sub_group(), v650_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1187_data = ir5[2];
              ir5[2] = (v1187_data + (v1171_data * (sycl::select_from_group(item.get_sub_group(), v656_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1193_data = ir5[3];
              ir5[3] = (v1193_data + (v1171_data * (sycl::select_from_group(item.get_sub_group(), v662_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1199_data = ir5[4];
              ir5[4] = (v1199_data + (v1171_data * (sycl::select_from_group(item.get_sub_group(), v668_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1205_data = ir5[5];
              ir5[5] = (v1205_data + (v1171_data * (sycl::select_from_group(item.get_sub_group(), v674_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1211_data = ir5[6];
              ir5[6] = (v1211_data + (v1171_data * (sycl::select_from_group(item.get_sub_group(), v680_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1217_data = ir5[7];
              ir5[7] = (v1217_data + (v1171_data * (sycl::select_from_group(item.get_sub_group(), v686_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              // r5 = ir5 + r2
              if (v28_g) {
                #pragma unroll
                for (int32_t v1219_n1 = 0; v1219_n1 < 8; ++v1219_n1) {
                  float v1221_data = ir5[v1219_n1];
                  float v1222_data = r2[v1219_n1];
                  r5[v1219_n1] = (v1222_data + v1221_data);
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
              float r9[12]{};
              // r9 = load{g>r}(glb_m7);
              if (v28_g) {
                #pragma unroll
                for (int32_t v1824_i1 = 0; v1824_i1 < 12; ++v1824_i1) {
                  float v1829_data = glb_m7[(v27_lead + (v1824_i1 * 12))];
                  r9[v1824_i1] = v1829_data;
                }
              }
              float r8[8]{};
              // ir8 = +(r6 * r7)
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir8[8]{};
              float v1242_data = r6[0];
              float v1243_data = r7[0];
              float v1246_data = ir8[0];
              ir8[0] = (v1246_data + (v1242_data * (sycl::select_from_group(item.get_sub_group(), v1243_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1249_data = r7[1];
              float v1252_data = ir8[1];
              ir8[1] = (v1252_data + (v1242_data * (sycl::select_from_group(item.get_sub_group(), v1249_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1255_data = r7[2];
              float v1258_data = ir8[2];
              ir8[2] = (v1258_data + (v1242_data * (sycl::select_from_group(item.get_sub_group(), v1255_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1261_data = r7[3];
              float v1264_data = ir8[3];
              ir8[3] = (v1264_data + (v1242_data * (sycl::select_from_group(item.get_sub_group(), v1261_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1267_data = r7[4];
              float v1270_data = ir8[4];
              ir8[4] = (v1270_data + (v1242_data * (sycl::select_from_group(item.get_sub_group(), v1267_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1273_data = r7[5];
              float v1276_data = ir8[5];
              ir8[5] = (v1276_data + (v1242_data * (sycl::select_from_group(item.get_sub_group(), v1273_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1279_data = r7[6];
              float v1282_data = ir8[6];
              ir8[6] = (v1282_data + (v1242_data * (sycl::select_from_group(item.get_sub_group(), v1279_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1285_data = r7[7];
              float v1288_data = ir8[7];
              ir8[7] = (v1288_data + (v1242_data * (sycl::select_from_group(item.get_sub_group(), v1285_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1290_data = r6[1];
              float v1294_data = ir8[0];
              ir8[0] = (v1294_data + (v1290_data * (sycl::select_from_group(item.get_sub_group(), v1243_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1300_data = ir8[1];
              ir8[1] = (v1300_data + (v1290_data * (sycl::select_from_group(item.get_sub_group(), v1249_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1306_data = ir8[2];
              ir8[2] = (v1306_data + (v1290_data * (sycl::select_from_group(item.get_sub_group(), v1255_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1312_data = ir8[3];
              ir8[3] = (v1312_data + (v1290_data * (sycl::select_from_group(item.get_sub_group(), v1261_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1318_data = ir8[4];
              ir8[4] = (v1318_data + (v1290_data * (sycl::select_from_group(item.get_sub_group(), v1267_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1324_data = ir8[5];
              ir8[5] = (v1324_data + (v1290_data * (sycl::select_from_group(item.get_sub_group(), v1273_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1330_data = ir8[6];
              ir8[6] = (v1330_data + (v1290_data * (sycl::select_from_group(item.get_sub_group(), v1279_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1336_data = ir8[7];
              ir8[7] = (v1336_data + (v1290_data * (sycl::select_from_group(item.get_sub_group(), v1285_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1338_data = r6[2];
              float v1342_data = ir8[0];
              ir8[0] = (v1342_data + (v1338_data * (sycl::select_from_group(item.get_sub_group(), v1243_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1348_data = ir8[1];
              ir8[1] = (v1348_data + (v1338_data * (sycl::select_from_group(item.get_sub_group(), v1249_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1354_data = ir8[2];
              ir8[2] = (v1354_data + (v1338_data * (sycl::select_from_group(item.get_sub_group(), v1255_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1360_data = ir8[3];
              ir8[3] = (v1360_data + (v1338_data * (sycl::select_from_group(item.get_sub_group(), v1261_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1366_data = ir8[4];
              ir8[4] = (v1366_data + (v1338_data * (sycl::select_from_group(item.get_sub_group(), v1267_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1372_data = ir8[5];
              ir8[5] = (v1372_data + (v1338_data * (sycl::select_from_group(item.get_sub_group(), v1273_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1378_data = ir8[6];
              ir8[6] = (v1378_data + (v1338_data * (sycl::select_from_group(item.get_sub_group(), v1279_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1384_data = ir8[7];
              ir8[7] = (v1384_data + (v1338_data * (sycl::select_from_group(item.get_sub_group(), v1285_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1386_data = r6[3];
              float v1390_data = ir8[0];
              ir8[0] = (v1390_data + (v1386_data * (sycl::select_from_group(item.get_sub_group(), v1243_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1396_data = ir8[1];
              ir8[1] = (v1396_data + (v1386_data * (sycl::select_from_group(item.get_sub_group(), v1249_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1402_data = ir8[2];
              ir8[2] = (v1402_data + (v1386_data * (sycl::select_from_group(item.get_sub_group(), v1255_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1408_data = ir8[3];
              ir8[3] = (v1408_data + (v1386_data * (sycl::select_from_group(item.get_sub_group(), v1261_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1414_data = ir8[4];
              ir8[4] = (v1414_data + (v1386_data * (sycl::select_from_group(item.get_sub_group(), v1267_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1420_data = ir8[5];
              ir8[5] = (v1420_data + (v1386_data * (sycl::select_from_group(item.get_sub_group(), v1273_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1426_data = ir8[6];
              ir8[6] = (v1426_data + (v1386_data * (sycl::select_from_group(item.get_sub_group(), v1279_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1432_data = ir8[7];
              ir8[7] = (v1432_data + (v1386_data * (sycl::select_from_group(item.get_sub_group(), v1285_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1434_data = r6[4];
              float v1438_data = ir8[0];
              ir8[0] = (v1438_data + (v1434_data * (sycl::select_from_group(item.get_sub_group(), v1243_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1444_data = ir8[1];
              ir8[1] = (v1444_data + (v1434_data * (sycl::select_from_group(item.get_sub_group(), v1249_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1450_data = ir8[2];
              ir8[2] = (v1450_data + (v1434_data * (sycl::select_from_group(item.get_sub_group(), v1255_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1456_data = ir8[3];
              ir8[3] = (v1456_data + (v1434_data * (sycl::select_from_group(item.get_sub_group(), v1261_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1462_data = ir8[4];
              ir8[4] = (v1462_data + (v1434_data * (sycl::select_from_group(item.get_sub_group(), v1267_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1468_data = ir8[5];
              ir8[5] = (v1468_data + (v1434_data * (sycl::select_from_group(item.get_sub_group(), v1273_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1474_data = ir8[6];
              ir8[6] = (v1474_data + (v1434_data * (sycl::select_from_group(item.get_sub_group(), v1279_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1480_data = ir8[7];
              ir8[7] = (v1480_data + (v1434_data * (sycl::select_from_group(item.get_sub_group(), v1285_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1482_data = r6[5];
              float v1486_data = ir8[0];
              ir8[0] = (v1486_data + (v1482_data * (sycl::select_from_group(item.get_sub_group(), v1243_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1492_data = ir8[1];
              ir8[1] = (v1492_data + (v1482_data * (sycl::select_from_group(item.get_sub_group(), v1249_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1498_data = ir8[2];
              ir8[2] = (v1498_data + (v1482_data * (sycl::select_from_group(item.get_sub_group(), v1255_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1504_data = ir8[3];
              ir8[3] = (v1504_data + (v1482_data * (sycl::select_from_group(item.get_sub_group(), v1261_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1510_data = ir8[4];
              ir8[4] = (v1510_data + (v1482_data * (sycl::select_from_group(item.get_sub_group(), v1267_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1516_data = ir8[5];
              ir8[5] = (v1516_data + (v1482_data * (sycl::select_from_group(item.get_sub_group(), v1273_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1522_data = ir8[6];
              ir8[6] = (v1522_data + (v1482_data * (sycl::select_from_group(item.get_sub_group(), v1279_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1528_data = ir8[7];
              ir8[7] = (v1528_data + (v1482_data * (sycl::select_from_group(item.get_sub_group(), v1285_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1530_data = r6[6];
              float v1534_data = ir8[0];
              ir8[0] = (v1534_data + (v1530_data * (sycl::select_from_group(item.get_sub_group(), v1243_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1540_data = ir8[1];
              ir8[1] = (v1540_data + (v1530_data * (sycl::select_from_group(item.get_sub_group(), v1249_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1546_data = ir8[2];
              ir8[2] = (v1546_data + (v1530_data * (sycl::select_from_group(item.get_sub_group(), v1255_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1552_data = ir8[3];
              ir8[3] = (v1552_data + (v1530_data * (sycl::select_from_group(item.get_sub_group(), v1261_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1558_data = ir8[4];
              ir8[4] = (v1558_data + (v1530_data * (sycl::select_from_group(item.get_sub_group(), v1267_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1564_data = ir8[5];
              ir8[5] = (v1564_data + (v1530_data * (sycl::select_from_group(item.get_sub_group(), v1273_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1570_data = ir8[6];
              ir8[6] = (v1570_data + (v1530_data * (sycl::select_from_group(item.get_sub_group(), v1279_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1576_data = ir8[7];
              ir8[7] = (v1576_data + (v1530_data * (sycl::select_from_group(item.get_sub_group(), v1285_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1578_data = r6[7];
              float v1582_data = ir8[0];
              ir8[0] = (v1582_data + (v1578_data * (sycl::select_from_group(item.get_sub_group(), v1243_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1588_data = ir8[1];
              ir8[1] = (v1588_data + (v1578_data * (sycl::select_from_group(item.get_sub_group(), v1249_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1594_data = ir8[2];
              ir8[2] = (v1594_data + (v1578_data * (sycl::select_from_group(item.get_sub_group(), v1255_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1600_data = ir8[3];
              ir8[3] = (v1600_data + (v1578_data * (sycl::select_from_group(item.get_sub_group(), v1261_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1606_data = ir8[4];
              ir8[4] = (v1606_data + (v1578_data * (sycl::select_from_group(item.get_sub_group(), v1267_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1612_data = ir8[5];
              ir8[5] = (v1612_data + (v1578_data * (sycl::select_from_group(item.get_sub_group(), v1273_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1618_data = ir8[6];
              ir8[6] = (v1618_data + (v1578_data * (sycl::select_from_group(item.get_sub_group(), v1279_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1624_data = ir8[7];
              ir8[7] = (v1624_data + (v1578_data * (sycl::select_from_group(item.get_sub_group(), v1285_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1626_data = r6[8];
              float v1630_data = ir8[0];
              ir8[0] = (v1630_data + (v1626_data * (sycl::select_from_group(item.get_sub_group(), v1243_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1636_data = ir8[1];
              ir8[1] = (v1636_data + (v1626_data * (sycl::select_from_group(item.get_sub_group(), v1249_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1642_data = ir8[2];
              ir8[2] = (v1642_data + (v1626_data * (sycl::select_from_group(item.get_sub_group(), v1255_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1648_data = ir8[3];
              ir8[3] = (v1648_data + (v1626_data * (sycl::select_from_group(item.get_sub_group(), v1261_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1654_data = ir8[4];
              ir8[4] = (v1654_data + (v1626_data * (sycl::select_from_group(item.get_sub_group(), v1267_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1660_data = ir8[5];
              ir8[5] = (v1660_data + (v1626_data * (sycl::select_from_group(item.get_sub_group(), v1273_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1666_data = ir8[6];
              ir8[6] = (v1666_data + (v1626_data * (sycl::select_from_group(item.get_sub_group(), v1279_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1672_data = ir8[7];
              ir8[7] = (v1672_data + (v1626_data * (sycl::select_from_group(item.get_sub_group(), v1285_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1674_data = r6[9];
              float v1678_data = ir8[0];
              ir8[0] = (v1678_data + (v1674_data * (sycl::select_from_group(item.get_sub_group(), v1243_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1684_data = ir8[1];
              ir8[1] = (v1684_data + (v1674_data * (sycl::select_from_group(item.get_sub_group(), v1249_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1690_data = ir8[2];
              ir8[2] = (v1690_data + (v1674_data * (sycl::select_from_group(item.get_sub_group(), v1255_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1696_data = ir8[3];
              ir8[3] = (v1696_data + (v1674_data * (sycl::select_from_group(item.get_sub_group(), v1261_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1702_data = ir8[4];
              ir8[4] = (v1702_data + (v1674_data * (sycl::select_from_group(item.get_sub_group(), v1267_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1708_data = ir8[5];
              ir8[5] = (v1708_data + (v1674_data * (sycl::select_from_group(item.get_sub_group(), v1273_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1714_data = ir8[6];
              ir8[6] = (v1714_data + (v1674_data * (sycl::select_from_group(item.get_sub_group(), v1279_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1720_data = ir8[7];
              ir8[7] = (v1720_data + (v1674_data * (sycl::select_from_group(item.get_sub_group(), v1285_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1722_data = r6[10];
              float v1726_data = ir8[0];
              ir8[0] = (v1726_data + (v1722_data * (sycl::select_from_group(item.get_sub_group(), v1243_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1732_data = ir8[1];
              ir8[1] = (v1732_data + (v1722_data * (sycl::select_from_group(item.get_sub_group(), v1249_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1738_data = ir8[2];
              ir8[2] = (v1738_data + (v1722_data * (sycl::select_from_group(item.get_sub_group(), v1255_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1744_data = ir8[3];
              ir8[3] = (v1744_data + (v1722_data * (sycl::select_from_group(item.get_sub_group(), v1261_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1750_data = ir8[4];
              ir8[4] = (v1750_data + (v1722_data * (sycl::select_from_group(item.get_sub_group(), v1267_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1756_data = ir8[5];
              ir8[5] = (v1756_data + (v1722_data * (sycl::select_from_group(item.get_sub_group(), v1273_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1762_data = ir8[6];
              ir8[6] = (v1762_data + (v1722_data * (sycl::select_from_group(item.get_sub_group(), v1279_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1768_data = ir8[7];
              ir8[7] = (v1768_data + (v1722_data * (sycl::select_from_group(item.get_sub_group(), v1285_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1770_data = r6[11];
              float v1774_data = ir8[0];
              ir8[0] = (v1774_data + (v1770_data * (sycl::select_from_group(item.get_sub_group(), v1243_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1780_data = ir8[1];
              ir8[1] = (v1780_data + (v1770_data * (sycl::select_from_group(item.get_sub_group(), v1249_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1786_data = ir8[2];
              ir8[2] = (v1786_data + (v1770_data * (sycl::select_from_group(item.get_sub_group(), v1255_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1792_data = ir8[3];
              ir8[3] = (v1792_data + (v1770_data * (sycl::select_from_group(item.get_sub_group(), v1261_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1798_data = ir8[4];
              ir8[4] = (v1798_data + (v1770_data * (sycl::select_from_group(item.get_sub_group(), v1267_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1804_data = ir8[5];
              ir8[5] = (v1804_data + (v1770_data * (sycl::select_from_group(item.get_sub_group(), v1273_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1810_data = ir8[6];
              ir8[6] = (v1810_data + (v1770_data * (sycl::select_from_group(item.get_sub_group(), v1279_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1816_data = ir8[7];
              ir8[7] = (v1816_data + (v1770_data * (sycl::select_from_group(item.get_sub_group(), v1285_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              // r8 = ir8 + r5
              if (v28_g) {
                #pragma unroll
                for (int32_t v1818_n1 = 0; v1818_n1 < 8; ++v1818_n1) {
                  float v1820_data = ir8[v1818_n1];
                  float v1821_data = r5[v1818_n1];
                  r8[v1818_n1] = (v1821_data + v1820_data);
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

