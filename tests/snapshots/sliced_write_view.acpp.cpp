// === base name ===
kernel_146b80fe844b9586

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_146b80fe844b9586 = {{32, 1, 1}, 32, 32, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_146b80fe844b9586(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_146b80fe844b9586(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_146b80fe844b9586(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (32, 1, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 1 - 1) / 1;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 32;
  config.block[1] = 1;
  config.block[2] = 1;
  config.sharedMemBytes = 0 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_146b80fe844b9586(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_146b80fe844b9586(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_146b80fe844b9586(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_146b80fe844b9586(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes x 1 per block = block 32x1x1, 0 B shared, occupancy grid
        // operands:
        //   m0 32×13(32×13) {0..32}×{0..13} strided
        //   m1 32×13(32×13) {0..32}×{0..13} strided
        //   m2 13×13(13×13) {0..13}×{0..13} strided
        //   m3 32×13(32×13) {0..32}×{0..13} strided
        //   m4 13×13(13×13) {0..13}×{0..13} strided
        // operations:
        //   m0[i,j]@{0..32}×{8..9} = m1[i,k]@{0..32}×{10..13} × m2[k,j]@{10..13}×{8..9}
        //   m3[i,j] = m0[i,k] × m4[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[32,13]],"name":"m0","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"Q","bbox":[[0,0],[32,13]],"name":"m1","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"S","bbox":[[0,0],[13,13]],"name":"m2","ordered":false,"parts":1,"shape":[13,13],"variant":false},{"addressing":"strided","alias":"O","bbox":[[0,0],[32,13]],"name":"m3","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[13,13]],"name":"m4","ordered":false,"parts":1,"shape":[13,13],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,1]],"is_tmp":false,"name":"m0","offset":[0,8],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,10],[32,13]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[10,0],[13,1]],"is_tmp":false,"name":"m2","offset":[0,8],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[0,0],[13,13]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 416 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 416 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 169 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v7_batchId0 * 416 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v7_batchId0 * 169 + 0 + m4_extraOffset];
              float r0[3]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v23_lead = item.get_local_id(2) % 32;
              #pragma unroll
              for (int32_t v24_i0 = 0; v24_i0 < 1; ++v24_i0) {
                int32_t v27_lead = v23_lead + (v24_i0 * 32);
                #pragma unroll
                for (int32_t v25_i1 = 10; v25_i1 < 13; ++v25_i1) {
                  float v30_data = glb_m1[(v27_lead + (v25_i1 * 32))];
                  r0[(v24_i0 + (v25_i1 - 10))] = v30_data;
                }
              }
              float r1[1]{};
              // r1 = load{g>r}(glb_m2);
              bool v35_g = v23_lead < 13;
              if ((v23_lead >= 10) && v35_g) {
                #pragma unroll
                for (int32_t v37_i1 = 8; v37_i1 < 9; ++v37_i1) {
                  float v42_data = glb_m2[(v23_lead + (v37_i1 * 13))];
                  r1[(v37_i1 - 8)] = v42_data;
                }
              }
              float r2[1]{};
              // ir2 = +(r0 * r1)
              // [(0, 32), (0, 1)] [(10, 13)]
              float ir2[1]{};
              float v47_data = r0[0];
              float v48_data = r1[0];
              float v51_data = ir2[0];
              ir2[0] = (v51_data + (v47_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v53_data = r0[1];
              float v57_data = ir2[0];
              ir2[0] = (v57_data + (v53_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v59_data = r0[2];
              float v63_data = ir2[0];
              ir2[0] = (v63_data + (v59_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              // r2 = ir2
              #pragma unroll
              for (int32_t v65_n0 = 0; v65_n0 < 1; ++v65_n0) {
                #pragma unroll
                for (int32_t v66_n1 = 0; v66_n1 < 1; ++v66_n1) {
                  int32_t v67_a = v65_n0 + v66_n1;
                  float v68_data = ir2[v67_a];
                  r2[v67_a] = v68_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v69_i0 = 0; v69_i0 < 1; ++v69_i0) {
                int32_t v74_lead = v23_lead + (v69_i0 * 32);
                #pragma unroll
                for (int32_t v70_i1 = 0; v70_i1 < 1; ++v70_i1) {
                  float v72_data = r2[(v69_i0 + v70_i1)];
                  glb_m0[(v74_lead + ((v70_i1 + 8) * 32))] = v72_data;
                }
              }
              float r3[13]{};
              // r3 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v79_i0 = 0; v79_i0 < 1; ++v79_i0) {
                int32_t v82_lead = v23_lead + (v79_i0 * 32);
                #pragma unroll
                for (int32_t v80_i1 = 0; v80_i1 < 13; ++v80_i1) {
                  float v85_data = glb_m0[(v82_lead + (v80_i1 * 32))];
                  r3[(v79_i0 + v80_i1)] = v85_data;
                }
              }
              float r4[13]{};
              // r4 = load{g>r}(glb_m4);
              if (v35_g) {
                #pragma unroll
                for (int32_t v88_i1 = 0; v88_i1 < 13; ++v88_i1) {
                  float v93_data = glb_m4[(v23_lead + (v88_i1 * 13))];
                  r4[v88_i1] = v93_data;
                }
              }
              float r5[13]{};
              // ir5 = +(r3 * r4)
              // [(0, 32), (0, 13)] [(0, 13)]
              float ir5[13]{};
              float v97_data = r3[0];
              float v98_data = r4[0];
              float v101_data = ir5[0];
              ir5[0] = (v101_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v104_data = r4[1];
              float v107_data = ir5[1];
              ir5[1] = (v107_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v110_data = r4[2];
              float v113_data = ir5[2];
              ir5[2] = (v113_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v116_data = r4[3];
              float v119_data = ir5[3];
              ir5[3] = (v119_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v122_data = r4[4];
              float v125_data = ir5[4];
              ir5[4] = (v125_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v128_data = r4[5];
              float v131_data = ir5[5];
              ir5[5] = (v131_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v134_data = r4[6];
              float v137_data = ir5[6];
              ir5[6] = (v137_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v140_data = r4[7];
              float v143_data = ir5[7];
              ir5[7] = (v143_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v140_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v146_data = r4[8];
              float v149_data = ir5[8];
              ir5[8] = (v149_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v146_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v152_data = r4[9];
              float v155_data = ir5[9];
              ir5[9] = (v155_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v152_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v158_data = r4[10];
              float v161_data = ir5[10];
              ir5[10] = (v161_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v158_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v164_data = r4[11];
              float v167_data = ir5[11];
              ir5[11] = (v167_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v164_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v170_data = r4[12];
              float v173_data = ir5[12];
              ir5[12] = (v173_data + (v97_data * (sycl::select_from_group(item.get_sub_group(), v170_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v175_data = r3[1];
              float v179_data = ir5[0];
              ir5[0] = (v179_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v185_data = ir5[1];
              ir5[1] = (v185_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v191_data = ir5[2];
              ir5[2] = (v191_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v197_data = ir5[3];
              ir5[3] = (v197_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v203_data = ir5[4];
              ir5[4] = (v203_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v209_data = ir5[5];
              ir5[5] = (v209_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v215_data = ir5[6];
              ir5[6] = (v215_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v221_data = ir5[7];
              ir5[7] = (v221_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v140_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v227_data = ir5[8];
              ir5[8] = (v227_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v146_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v233_data = ir5[9];
              ir5[9] = (v233_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v152_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v239_data = ir5[10];
              ir5[10] = (v239_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v158_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v245_data = ir5[11];
              ir5[11] = (v245_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v164_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v251_data = ir5[12];
              ir5[12] = (v251_data + (v175_data * (sycl::select_from_group(item.get_sub_group(), v170_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v253_data = r3[2];
              float v257_data = ir5[0];
              ir5[0] = (v257_data + (v253_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v263_data = ir5[1];
              ir5[1] = (v263_data + (v253_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v269_data = ir5[2];
              ir5[2] = (v269_data + (v253_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v275_data = ir5[3];
              ir5[3] = (v275_data + (v253_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v281_data = ir5[4];
              ir5[4] = (v281_data + (v253_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v287_data = ir5[5];
              ir5[5] = (v287_data + (v253_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v293_data = ir5[6];
              ir5[6] = (v293_data + (v253_data * (sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v299_data = ir5[7];
              ir5[7] = (v299_data + (v253_data * (sycl::select_from_group(item.get_sub_group(), v140_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v305_data = ir5[8];
              ir5[8] = (v305_data + (v253_data * (sycl::select_from_group(item.get_sub_group(), v146_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v311_data = ir5[9];
              ir5[9] = (v311_data + (v253_data * (sycl::select_from_group(item.get_sub_group(), v152_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v317_data = ir5[10];
              ir5[10] = (v317_data + (v253_data * (sycl::select_from_group(item.get_sub_group(), v158_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v323_data = ir5[11];
              ir5[11] = (v323_data + (v253_data * (sycl::select_from_group(item.get_sub_group(), v164_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v329_data = ir5[12];
              ir5[12] = (v329_data + (v253_data * (sycl::select_from_group(item.get_sub_group(), v170_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v331_data = r3[3];
              float v335_data = ir5[0];
              ir5[0] = (v335_data + (v331_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v341_data = ir5[1];
              ir5[1] = (v341_data + (v331_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v347_data = ir5[2];
              ir5[2] = (v347_data + (v331_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v353_data = ir5[3];
              ir5[3] = (v353_data + (v331_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v359_data = ir5[4];
              ir5[4] = (v359_data + (v331_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v365_data = ir5[5];
              ir5[5] = (v365_data + (v331_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v371_data = ir5[6];
              ir5[6] = (v371_data + (v331_data * (sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v377_data = ir5[7];
              ir5[7] = (v377_data + (v331_data * (sycl::select_from_group(item.get_sub_group(), v140_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v383_data = ir5[8];
              ir5[8] = (v383_data + (v331_data * (sycl::select_from_group(item.get_sub_group(), v146_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v389_data = ir5[9];
              ir5[9] = (v389_data + (v331_data * (sycl::select_from_group(item.get_sub_group(), v152_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v395_data = ir5[10];
              ir5[10] = (v395_data + (v331_data * (sycl::select_from_group(item.get_sub_group(), v158_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v401_data = ir5[11];
              ir5[11] = (v401_data + (v331_data * (sycl::select_from_group(item.get_sub_group(), v164_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v407_data = ir5[12];
              ir5[12] = (v407_data + (v331_data * (sycl::select_from_group(item.get_sub_group(), v170_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v409_data = r3[4];
              float v413_data = ir5[0];
              ir5[0] = (v413_data + (v409_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v419_data = ir5[1];
              ir5[1] = (v419_data + (v409_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v425_data = ir5[2];
              ir5[2] = (v425_data + (v409_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v431_data = ir5[3];
              ir5[3] = (v431_data + (v409_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v437_data = ir5[4];
              ir5[4] = (v437_data + (v409_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v443_data = ir5[5];
              ir5[5] = (v443_data + (v409_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v449_data = ir5[6];
              ir5[6] = (v449_data + (v409_data * (sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v455_data = ir5[7];
              ir5[7] = (v455_data + (v409_data * (sycl::select_from_group(item.get_sub_group(), v140_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v461_data = ir5[8];
              ir5[8] = (v461_data + (v409_data * (sycl::select_from_group(item.get_sub_group(), v146_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v467_data = ir5[9];
              ir5[9] = (v467_data + (v409_data * (sycl::select_from_group(item.get_sub_group(), v152_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v473_data = ir5[10];
              ir5[10] = (v473_data + (v409_data * (sycl::select_from_group(item.get_sub_group(), v158_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v479_data = ir5[11];
              ir5[11] = (v479_data + (v409_data * (sycl::select_from_group(item.get_sub_group(), v164_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v485_data = ir5[12];
              ir5[12] = (v485_data + (v409_data * (sycl::select_from_group(item.get_sub_group(), v170_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v487_data = r3[5];
              float v491_data = ir5[0];
              ir5[0] = (v491_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v497_data = ir5[1];
              ir5[1] = (v497_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v503_data = ir5[2];
              ir5[2] = (v503_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v509_data = ir5[3];
              ir5[3] = (v509_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v515_data = ir5[4];
              ir5[4] = (v515_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v521_data = ir5[5];
              ir5[5] = (v521_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v527_data = ir5[6];
              ir5[6] = (v527_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v533_data = ir5[7];
              ir5[7] = (v533_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v140_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v539_data = ir5[8];
              ir5[8] = (v539_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v146_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v545_data = ir5[9];
              ir5[9] = (v545_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v152_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v551_data = ir5[10];
              ir5[10] = (v551_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v158_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v557_data = ir5[11];
              ir5[11] = (v557_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v164_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v563_data = ir5[12];
              ir5[12] = (v563_data + (v487_data * (sycl::select_from_group(item.get_sub_group(), v170_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v565_data = r3[6];
              float v569_data = ir5[0];
              ir5[0] = (v569_data + (v565_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v575_data = ir5[1];
              ir5[1] = (v575_data + (v565_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v581_data = ir5[2];
              ir5[2] = (v581_data + (v565_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v587_data = ir5[3];
              ir5[3] = (v587_data + (v565_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v593_data = ir5[4];
              ir5[4] = (v593_data + (v565_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v599_data = ir5[5];
              ir5[5] = (v599_data + (v565_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v605_data = ir5[6];
              ir5[6] = (v605_data + (v565_data * (sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v611_data = ir5[7];
              ir5[7] = (v611_data + (v565_data * (sycl::select_from_group(item.get_sub_group(), v140_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v617_data = ir5[8];
              ir5[8] = (v617_data + (v565_data * (sycl::select_from_group(item.get_sub_group(), v146_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v623_data = ir5[9];
              ir5[9] = (v623_data + (v565_data * (sycl::select_from_group(item.get_sub_group(), v152_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v629_data = ir5[10];
              ir5[10] = (v629_data + (v565_data * (sycl::select_from_group(item.get_sub_group(), v158_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v635_data = ir5[11];
              ir5[11] = (v635_data + (v565_data * (sycl::select_from_group(item.get_sub_group(), v164_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v641_data = ir5[12];
              ir5[12] = (v641_data + (v565_data * (sycl::select_from_group(item.get_sub_group(), v170_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v643_data = r3[7];
              float v647_data = ir5[0];
              ir5[0] = (v647_data + (v643_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v653_data = ir5[1];
              ir5[1] = (v653_data + (v643_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v659_data = ir5[2];
              ir5[2] = (v659_data + (v643_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v665_data = ir5[3];
              ir5[3] = (v665_data + (v643_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v671_data = ir5[4];
              ir5[4] = (v671_data + (v643_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v677_data = ir5[5];
              ir5[5] = (v677_data + (v643_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v683_data = ir5[6];
              ir5[6] = (v683_data + (v643_data * (sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v689_data = ir5[7];
              ir5[7] = (v689_data + (v643_data * (sycl::select_from_group(item.get_sub_group(), v140_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v695_data = ir5[8];
              ir5[8] = (v695_data + (v643_data * (sycl::select_from_group(item.get_sub_group(), v146_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v701_data = ir5[9];
              ir5[9] = (v701_data + (v643_data * (sycl::select_from_group(item.get_sub_group(), v152_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v707_data = ir5[10];
              ir5[10] = (v707_data + (v643_data * (sycl::select_from_group(item.get_sub_group(), v158_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v713_data = ir5[11];
              ir5[11] = (v713_data + (v643_data * (sycl::select_from_group(item.get_sub_group(), v164_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v719_data = ir5[12];
              ir5[12] = (v719_data + (v643_data * (sycl::select_from_group(item.get_sub_group(), v170_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v721_data = r3[8];
              float v725_data = ir5[0];
              ir5[0] = (v725_data + (v721_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v731_data = ir5[1];
              ir5[1] = (v731_data + (v721_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v737_data = ir5[2];
              ir5[2] = (v737_data + (v721_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v743_data = ir5[3];
              ir5[3] = (v743_data + (v721_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v749_data = ir5[4];
              ir5[4] = (v749_data + (v721_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v755_data = ir5[5];
              ir5[5] = (v755_data + (v721_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v761_data = ir5[6];
              ir5[6] = (v761_data + (v721_data * (sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v767_data = ir5[7];
              ir5[7] = (v767_data + (v721_data * (sycl::select_from_group(item.get_sub_group(), v140_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v773_data = ir5[8];
              ir5[8] = (v773_data + (v721_data * (sycl::select_from_group(item.get_sub_group(), v146_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v779_data = ir5[9];
              ir5[9] = (v779_data + (v721_data * (sycl::select_from_group(item.get_sub_group(), v152_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v785_data = ir5[10];
              ir5[10] = (v785_data + (v721_data * (sycl::select_from_group(item.get_sub_group(), v158_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v791_data = ir5[11];
              ir5[11] = (v791_data + (v721_data * (sycl::select_from_group(item.get_sub_group(), v164_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v797_data = ir5[12];
              ir5[12] = (v797_data + (v721_data * (sycl::select_from_group(item.get_sub_group(), v170_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v799_data = r3[9];
              float v803_data = ir5[0];
              ir5[0] = (v803_data + (v799_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v809_data = ir5[1];
              ir5[1] = (v809_data + (v799_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v815_data = ir5[2];
              ir5[2] = (v815_data + (v799_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v821_data = ir5[3];
              ir5[3] = (v821_data + (v799_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v827_data = ir5[4];
              ir5[4] = (v827_data + (v799_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v833_data = ir5[5];
              ir5[5] = (v833_data + (v799_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v839_data = ir5[6];
              ir5[6] = (v839_data + (v799_data * (sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v845_data = ir5[7];
              ir5[7] = (v845_data + (v799_data * (sycl::select_from_group(item.get_sub_group(), v140_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v851_data = ir5[8];
              ir5[8] = (v851_data + (v799_data * (sycl::select_from_group(item.get_sub_group(), v146_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v857_data = ir5[9];
              ir5[9] = (v857_data + (v799_data * (sycl::select_from_group(item.get_sub_group(), v152_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v863_data = ir5[10];
              ir5[10] = (v863_data + (v799_data * (sycl::select_from_group(item.get_sub_group(), v158_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v869_data = ir5[11];
              ir5[11] = (v869_data + (v799_data * (sycl::select_from_group(item.get_sub_group(), v164_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v875_data = ir5[12];
              ir5[12] = (v875_data + (v799_data * (sycl::select_from_group(item.get_sub_group(), v170_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v877_data = r3[10];
              float v881_data = ir5[0];
              ir5[0] = (v881_data + (v877_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v887_data = ir5[1];
              ir5[1] = (v887_data + (v877_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v893_data = ir5[2];
              ir5[2] = (v893_data + (v877_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v899_data = ir5[3];
              ir5[3] = (v899_data + (v877_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v905_data = ir5[4];
              ir5[4] = (v905_data + (v877_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v911_data = ir5[5];
              ir5[5] = (v911_data + (v877_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v917_data = ir5[6];
              ir5[6] = (v917_data + (v877_data * (sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v923_data = ir5[7];
              ir5[7] = (v923_data + (v877_data * (sycl::select_from_group(item.get_sub_group(), v140_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v929_data = ir5[8];
              ir5[8] = (v929_data + (v877_data * (sycl::select_from_group(item.get_sub_group(), v146_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v935_data = ir5[9];
              ir5[9] = (v935_data + (v877_data * (sycl::select_from_group(item.get_sub_group(), v152_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v941_data = ir5[10];
              ir5[10] = (v941_data + (v877_data * (sycl::select_from_group(item.get_sub_group(), v158_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v947_data = ir5[11];
              ir5[11] = (v947_data + (v877_data * (sycl::select_from_group(item.get_sub_group(), v164_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v953_data = ir5[12];
              ir5[12] = (v953_data + (v877_data * (sycl::select_from_group(item.get_sub_group(), v170_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v955_data = r3[11];
              float v959_data = ir5[0];
              ir5[0] = (v959_data + (v955_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v965_data = ir5[1];
              ir5[1] = (v965_data + (v955_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v971_data = ir5[2];
              ir5[2] = (v971_data + (v955_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v977_data = ir5[3];
              ir5[3] = (v977_data + (v955_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v983_data = ir5[4];
              ir5[4] = (v983_data + (v955_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v989_data = ir5[5];
              ir5[5] = (v989_data + (v955_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v995_data = ir5[6];
              ir5[6] = (v995_data + (v955_data * (sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1001_data = ir5[7];
              ir5[7] = (v1001_data + (v955_data * (sycl::select_from_group(item.get_sub_group(), v140_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1007_data = ir5[8];
              ir5[8] = (v1007_data + (v955_data * (sycl::select_from_group(item.get_sub_group(), v146_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1013_data = ir5[9];
              ir5[9] = (v1013_data + (v955_data * (sycl::select_from_group(item.get_sub_group(), v152_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1019_data = ir5[10];
              ir5[10] = (v1019_data + (v955_data * (sycl::select_from_group(item.get_sub_group(), v158_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1025_data = ir5[11];
              ir5[11] = (v1025_data + (v955_data * (sycl::select_from_group(item.get_sub_group(), v164_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1031_data = ir5[12];
              ir5[12] = (v1031_data + (v955_data * (sycl::select_from_group(item.get_sub_group(), v170_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1033_data = r3[12];
              float v1037_data = ir5[0];
              ir5[0] = (v1037_data + (v1033_data * (sycl::select_from_group(item.get_sub_group(), v98_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v1043_data = ir5[1];
              ir5[1] = (v1043_data + (v1033_data * (sycl::select_from_group(item.get_sub_group(), v104_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v1049_data = ir5[2];
              ir5[2] = (v1049_data + (v1033_data * (sycl::select_from_group(item.get_sub_group(), v110_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v1055_data = ir5[3];
              ir5[3] = (v1055_data + (v1033_data * (sycl::select_from_group(item.get_sub_group(), v116_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v1061_data = ir5[4];
              ir5[4] = (v1061_data + (v1033_data * (sycl::select_from_group(item.get_sub_group(), v122_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v1067_data = ir5[5];
              ir5[5] = (v1067_data + (v1033_data * (sycl::select_from_group(item.get_sub_group(), v128_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v1073_data = ir5[6];
              ir5[6] = (v1073_data + (v1033_data * (sycl::select_from_group(item.get_sub_group(), v134_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v1079_data = ir5[7];
              ir5[7] = (v1079_data + (v1033_data * (sycl::select_from_group(item.get_sub_group(), v140_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v1085_data = ir5[8];
              ir5[8] = (v1085_data + (v1033_data * (sycl::select_from_group(item.get_sub_group(), v146_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v1091_data = ir5[9];
              ir5[9] = (v1091_data + (v1033_data * (sycl::select_from_group(item.get_sub_group(), v152_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v1097_data = ir5[10];
              ir5[10] = (v1097_data + (v1033_data * (sycl::select_from_group(item.get_sub_group(), v158_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v1103_data = ir5[11];
              ir5[11] = (v1103_data + (v1033_data * (sycl::select_from_group(item.get_sub_group(), v164_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v1109_data = ir5[12];
              ir5[12] = (v1109_data + (v1033_data * (sycl::select_from_group(item.get_sub_group(), v170_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              // r5 = ir5
              #pragma unroll
              for (int32_t v1111_n0 = 0; v1111_n0 < 1; ++v1111_n0) {
                #pragma unroll
                for (int32_t v1112_n1 = 0; v1112_n1 < 13; ++v1112_n1) {
                  int32_t v1113_a = v1111_n0 + v1112_n1;
                  float v1114_data = ir5[v1113_a];
                  r5[v1113_a] = v1114_data;
                }
              }
              // glb_m3 = store{r>g}(r5);
              #pragma unroll
              for (int32_t v1115_i0 = 0; v1115_i0 < 1; ++v1115_i0) {
                int32_t v1120_lead = v23_lead + (v1115_i0 * 32);
                #pragma unroll
                for (int32_t v1116_i1 = 0; v1116_i1 < 13; ++v1116_i1) {
                  float v1118_data = r5[(v1115_i0 + v1116_i1)];
                  glb_m3[(v1120_lead + (v1116_i1 * 32))] = v1118_data;
                }
              }
              item.barrier();
            }
          }
        }
      });
    }
  });
}

