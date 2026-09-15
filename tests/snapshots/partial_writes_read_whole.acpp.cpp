// === base name ===
<<<<<<< HEAD
kernel_d065402a2f7cb620
=======
kernel_9e708de55c52b821
>>>>>>> fix: keep narrowing accumulation chains in registers

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
<<<<<<< HEAD
inline constexpr tensorforge::LaunchInfo launch_info_kernel_d065402a2f7cb620 = {{32, 1, 1}, 32, 32, 1, 1, 1152, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_d065402a2f7cb620(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_d065402a2f7cb620(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);
=======
inline constexpr tensorforge::LaunchInfo launch_info_kernel_9e708de55c52b821 = {{32, 1, 1}, 32, 32, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_9e708de55c52b821(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_9e708de55c52b821(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);
>>>>>>> fix: keep narrowing accumulation chains in registers


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
<<<<<<< HEAD
tensorforge::LaunchConfig launch_config_kernel_d065402a2f7cb620(size_t numElements0, void* streamPtr) {
=======
tensorforge::LaunchConfig launch_config_kernel_9e708de55c52b821(size_t numElements0, void* streamPtr) {
>>>>>>> fix: keep narrowing accumulation chains in registers
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
<<<<<<< HEAD
void launcher_kernel_d065402a2f7cb620(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_d065402a2f7cb620(numElements0, streamPtr);
=======
void launcher_kernel_9e708de55c52b821(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_9e708de55c52b821(numElements0, streamPtr);
>>>>>>> fix: keep narrowing accumulation chains in registers
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
<<<<<<< HEAD
  kernel_kernel_d065402a2f7cb620(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
=======
  kernel_kernel_9e708de55c52b821(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
>>>>>>> fix: keep narrowing accumulation chains in registers
  CHECK_ERR;
}


// === kernel ===
<<<<<<< HEAD
inline void kernel_kernel_d065402a2f7cb620(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
=======
inline void kernel_kernel_9e708de55c52b821(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
>>>>>>> fix: keep narrowing accumulation chains in registers
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes x 1 per block = block 32x1x1, 0 B shared, occupancy grid
        // operands:
        //   m0 32×9(32×9) {0..32}×{0..9} pointer_based
        //   m1 16×9(16×9) {0..16}×{0..9} pointer_based
        //   m2 16×9(16×9) {0..16}×{0..9} pointer_based
        //   m3 32×9(32×9) {0..32}×{0..9} pointer_based
        //   m4 9×9(9×9) {0..9}×{0..9} pointer_based
        // operations:
        //   t0[i,j] = m0[i,j]
        //   t0[i,j] += m1[i,j]
        //   t0[i,j] += m2[i,j]
        //   m3[i,j] = t0[i,k] × m4[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"pointer_based","alias":"Q","bbox":[[0,0],[32,9]],"name":"m0","ordered":false,"parts":1,"shape":[32,9],"variant":false},{"addressing":"pointer_based","alias":"F0","bbox":[[0,0],[16,9]],"name":"m1","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"pointer_based","alias":"F1","bbox":[[0,0],[16,9]],"name":"m2","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"pointer_based","alias":"O","bbox":[[0,0],[32,9]],"name":"m3","ordered":false,"parts":1,"shape":[32,9],"variant":false},{"addressing":"pointer_based","alias":"M","bbox":[[0,0],[9,9]],"name":"m4","ordered":false,"parts":1,"shape":[9,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[32,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[32,9]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          for (size_t v1_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v1_batchId0 < numElements0; v1_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v2_ahead1 = v1_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v1_batchId0][0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v1_batchId0][0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v1_batchId0][0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v1_batchId0][0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v1_batchId0][0 + m4_extraOffset];
              float r0[9]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v17_lead = item.get_local_id(2) % 32;
              #pragma unroll
              for (int32_t v18_i0 = 0; v18_i0 < 1; ++v18_i0) {
                int32_t v21_lead = v17_lead + (v18_i0 * 32);
                #pragma unroll
                for (int32_t v19_i1 = 0; v19_i1 < 9; ++v19_i1) {
                  float v24_data = glb_m0[(v21_lead + (v19_i1 * 32))];
                  r0[(v18_i0 + v19_i1)] = v24_data;
                }
              }
              float r2[9]{};
              // r2 = load{g>r}(glb_m1);
              bool v27_g = v17_lead < 16;
              if (v27_g) {
                #pragma unroll
                for (int32_t v28_i1 = 0; v28_i1 < 9; ++v28_i1) {
                  float v33_data = glb_m1[(v17_lead + (v28_i1 * 16))];
                  r2[v28_i1] = v33_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r1[9]{};
              // r1 = +(r0) + None
              // [(0, 32), (0, 9)] []
              float v36_data = r0[0];
              float v37_data = r1[0];
              r1[0] = (v37_data + v36_data);
              float v39_data = r0[1];
              float v40_data = r1[1];
              r1[1] = (v40_data + v39_data);
              float v42_data = r0[2];
              float v43_data = r1[2];
              r1[2] = (v43_data + v42_data);
              float v45_data = r0[3];
              float v46_data = r1[3];
              r1[3] = (v46_data + v45_data);
              float v48_data = r0[4];
              float v49_data = r1[4];
              r1[4] = (v49_data + v48_data);
              float v51_data = r0[5];
              float v52_data = r1[5];
              r1[5] = (v52_data + v51_data);
              float v54_data = r0[6];
              float v55_data = r1[6];
              r1[6] = (v55_data + v54_data);
              float v57_data = r0[7];
              float v58_data = r1[7];
              r1[7] = (v58_data + v57_data);
              float v60_data = r0[8];
              float v61_data = r1[8];
              r1[8] = (v61_data + v60_data);
              float r4[9]{};
              // r4 = load{g>r}(glb_m2);
              if (v27_g) {
                #pragma unroll
                for (int32_t v64_i1 = 0; v64_i1 < 9; ++v64_i1) {
                  float v69_data = glb_m2[(v17_lead + (v64_i1 * 16))];
                  r4[v64_i1] = v69_data;
                }
              }
              // wait(r2 = load{g>r}(glb_m1););
              float r3[9]{};
              // ir3 = +(r2)
              // [(0, 16), (0, 9)] []
              float ir3[9]{};
              float v73_data = r2[0];
              float v74_data = ir3[0];
              ir3[0] = (v74_data + v73_data);
              float v76_data = r2[1];
              float v77_data = ir3[1];
              ir3[1] = (v77_data + v76_data);
              float v79_data = r2[2];
              float v80_data = ir3[2];
              ir3[2] = (v80_data + v79_data);
              float v82_data = r2[3];
              float v83_data = ir3[3];
              ir3[3] = (v83_data + v82_data);
              float v85_data = r2[4];
              float v86_data = ir3[4];
              ir3[4] = (v86_data + v85_data);
              float v88_data = r2[5];
              float v89_data = ir3[5];
              ir3[5] = (v89_data + v88_data);
              float v91_data = r2[6];
              float v92_data = ir3[6];
              ir3[6] = (v92_data + v91_data);
              float v94_data = r2[7];
              float v95_data = ir3[7];
              ir3[7] = (v95_data + v94_data);
              float v97_data = r2[8];
              float v98_data = ir3[8];
              ir3[8] = (v98_data + v97_data);
              // r3 = ir3 + r1
              #pragma unroll
              for (int32_t v100_n1 = 0; v100_n1 < 9; ++v100_n1) {
                float v102_data = ir3[v100_n1];
                float v103_data = r1[v100_n1];
                r3[v100_n1] = (v103_data + v102_data);
              }
              float r6[9]{};
              // r6 = load{g>r}(glb_m4);
              if (v17_lead < 9) {
                #pragma unroll
                for (int32_t v107_i1 = 0; v107_i1 < 9; ++v107_i1) {
                  float v112_data = glb_m4[(v17_lead + (v107_i1 * 9))];
                  r6[v107_i1] = v112_data;
                }
              }
              // wait(r4 = load{g>r}(glb_m2););
              float r5[9]{};
              // ir5 = +(r4)
              // [(0, 16), (0, 9)] []
              float ir5[9]{};
              float v116_data = r4[0];
              float v117_data = ir5[0];
              ir5[0] = (v117_data + v116_data);
              float v119_data = r4[1];
              float v120_data = ir5[1];
              ir5[1] = (v120_data + v119_data);
              float v122_data = r4[2];
              float v123_data = ir5[2];
              ir5[2] = (v123_data + v122_data);
              float v125_data = r4[3];
              float v126_data = ir5[3];
              ir5[3] = (v126_data + v125_data);
              float v128_data = r4[4];
              float v129_data = ir5[4];
              ir5[4] = (v129_data + v128_data);
              float v131_data = r4[5];
              float v132_data = ir5[5];
              ir5[5] = (v132_data + v131_data);
              float v134_data = r4[6];
              float v135_data = ir5[6];
              ir5[6] = (v135_data + v134_data);
              float v137_data = r4[7];
              float v138_data = ir5[7];
              ir5[7] = (v138_data + v137_data);
              float v140_data = r4[8];
              float v141_data = ir5[8];
              ir5[8] = (v141_data + v140_data);
              // r5 = ir5 + r3
              #pragma unroll
              for (int32_t v143_n1 = 0; v143_n1 < 9; ++v143_n1) {
                float v145_data = ir5[v143_n1];
                float v146_data = r3[v143_n1];
                r5[v143_n1] = (v146_data + v145_data);
              }
              // wait(r6 = load{g>r}(glb_m4););
              float r7[9]{};
              // ir7 = +(r5 * r6)
              // [(0, 32), (0, 9)] [(0, 9)]
              float ir7[9]{};
              float v150_data = r5[0];
              float v151_data = r6[0];
              float v154_data = ir7[0];
              ir7[0] = (v154_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v151_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v157_data = r6[1];
              float v160_data = ir7[1];
              ir7[1] = (v160_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v157_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v163_data = r6[2];
              float v166_data = ir7[2];
              ir7[2] = (v166_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v163_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v169_data = r6[3];
              float v172_data = ir7[3];
              ir7[3] = (v172_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v169_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v175_data = r6[4];
              float v178_data = ir7[4];
              ir7[4] = (v178_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v175_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v181_data = r6[5];
              float v184_data = ir7[5];
              ir7[5] = (v184_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v181_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v187_data = r6[6];
              float v190_data = ir7[6];
              ir7[6] = (v190_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v187_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v193_data = r6[7];
              float v196_data = ir7[7];
              ir7[7] = (v196_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v193_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v199_data = r6[8];
              float v202_data = ir7[8];
              ir7[8] = (v202_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v204_data = r5[1];
              float v208_data = ir7[0];
              ir7[0] = (v208_data + (v204_data * (sycl::select_from_group(item.get_sub_group(), v151_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v214_data = ir7[1];
              ir7[1] = (v214_data + (v204_data * (sycl::select_from_group(item.get_sub_group(), v157_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v220_data = ir7[2];
              ir7[2] = (v220_data + (v204_data * (sycl::select_from_group(item.get_sub_group(), v163_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v226_data = ir7[3];
              ir7[3] = (v226_data + (v204_data * (sycl::select_from_group(item.get_sub_group(), v169_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v232_data = ir7[4];
              ir7[4] = (v232_data + (v204_data * (sycl::select_from_group(item.get_sub_group(), v175_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v238_data = ir7[5];
              ir7[5] = (v238_data + (v204_data * (sycl::select_from_group(item.get_sub_group(), v181_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v244_data = ir7[6];
              ir7[6] = (v244_data + (v204_data * (sycl::select_from_group(item.get_sub_group(), v187_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v250_data = ir7[7];
              ir7[7] = (v250_data + (v204_data * (sycl::select_from_group(item.get_sub_group(), v193_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v256_data = ir7[8];
              ir7[8] = (v256_data + (v204_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v258_data = r5[2];
              float v262_data = ir7[0];
              ir7[0] = (v262_data + (v258_data * (sycl::select_from_group(item.get_sub_group(), v151_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v268_data = ir7[1];
              ir7[1] = (v268_data + (v258_data * (sycl::select_from_group(item.get_sub_group(), v157_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v274_data = ir7[2];
              ir7[2] = (v274_data + (v258_data * (sycl::select_from_group(item.get_sub_group(), v163_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v280_data = ir7[3];
              ir7[3] = (v280_data + (v258_data * (sycl::select_from_group(item.get_sub_group(), v169_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v286_data = ir7[4];
              ir7[4] = (v286_data + (v258_data * (sycl::select_from_group(item.get_sub_group(), v175_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v292_data = ir7[5];
              ir7[5] = (v292_data + (v258_data * (sycl::select_from_group(item.get_sub_group(), v181_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v298_data = ir7[6];
              ir7[6] = (v298_data + (v258_data * (sycl::select_from_group(item.get_sub_group(), v187_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v304_data = ir7[7];
              ir7[7] = (v304_data + (v258_data * (sycl::select_from_group(item.get_sub_group(), v193_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v310_data = ir7[8];
              ir7[8] = (v310_data + (v258_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v312_data = r5[3];
              float v316_data = ir7[0];
              ir7[0] = (v316_data + (v312_data * (sycl::select_from_group(item.get_sub_group(), v151_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v322_data = ir7[1];
              ir7[1] = (v322_data + (v312_data * (sycl::select_from_group(item.get_sub_group(), v157_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v328_data = ir7[2];
              ir7[2] = (v328_data + (v312_data * (sycl::select_from_group(item.get_sub_group(), v163_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v334_data = ir7[3];
              ir7[3] = (v334_data + (v312_data * (sycl::select_from_group(item.get_sub_group(), v169_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v340_data = ir7[4];
              ir7[4] = (v340_data + (v312_data * (sycl::select_from_group(item.get_sub_group(), v175_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v346_data = ir7[5];
              ir7[5] = (v346_data + (v312_data * (sycl::select_from_group(item.get_sub_group(), v181_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v352_data = ir7[6];
              ir7[6] = (v352_data + (v312_data * (sycl::select_from_group(item.get_sub_group(), v187_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v358_data = ir7[7];
              ir7[7] = (v358_data + (v312_data * (sycl::select_from_group(item.get_sub_group(), v193_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v364_data = ir7[8];
              ir7[8] = (v364_data + (v312_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v366_data = r5[4];
              float v370_data = ir7[0];
              ir7[0] = (v370_data + (v366_data * (sycl::select_from_group(item.get_sub_group(), v151_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v376_data = ir7[1];
              ir7[1] = (v376_data + (v366_data * (sycl::select_from_group(item.get_sub_group(), v157_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v382_data = ir7[2];
              ir7[2] = (v382_data + (v366_data * (sycl::select_from_group(item.get_sub_group(), v163_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v388_data = ir7[3];
              ir7[3] = (v388_data + (v366_data * (sycl::select_from_group(item.get_sub_group(), v169_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v394_data = ir7[4];
              ir7[4] = (v394_data + (v366_data * (sycl::select_from_group(item.get_sub_group(), v175_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v400_data = ir7[5];
              ir7[5] = (v400_data + (v366_data * (sycl::select_from_group(item.get_sub_group(), v181_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v406_data = ir7[6];
              ir7[6] = (v406_data + (v366_data * (sycl::select_from_group(item.get_sub_group(), v187_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v412_data = ir7[7];
              ir7[7] = (v412_data + (v366_data * (sycl::select_from_group(item.get_sub_group(), v193_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v418_data = ir7[8];
              ir7[8] = (v418_data + (v366_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v420_data = r5[5];
              float v424_data = ir7[0];
              ir7[0] = (v424_data + (v420_data * (sycl::select_from_group(item.get_sub_group(), v151_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v430_data = ir7[1];
              ir7[1] = (v430_data + (v420_data * (sycl::select_from_group(item.get_sub_group(), v157_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v436_data = ir7[2];
              ir7[2] = (v436_data + (v420_data * (sycl::select_from_group(item.get_sub_group(), v163_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v442_data = ir7[3];
              ir7[3] = (v442_data + (v420_data * (sycl::select_from_group(item.get_sub_group(), v169_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v448_data = ir7[4];
              ir7[4] = (v448_data + (v420_data * (sycl::select_from_group(item.get_sub_group(), v175_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v454_data = ir7[5];
              ir7[5] = (v454_data + (v420_data * (sycl::select_from_group(item.get_sub_group(), v181_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v460_data = ir7[6];
              ir7[6] = (v460_data + (v420_data * (sycl::select_from_group(item.get_sub_group(), v187_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v466_data = ir7[7];
              ir7[7] = (v466_data + (v420_data * (sycl::select_from_group(item.get_sub_group(), v193_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v472_data = ir7[8];
              ir7[8] = (v472_data + (v420_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v474_data = r5[6];
              float v478_data = ir7[0];
              ir7[0] = (v478_data + (v474_data * (sycl::select_from_group(item.get_sub_group(), v151_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v484_data = ir7[1];
              ir7[1] = (v484_data + (v474_data * (sycl::select_from_group(item.get_sub_group(), v157_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v490_data = ir7[2];
              ir7[2] = (v490_data + (v474_data * (sycl::select_from_group(item.get_sub_group(), v163_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v496_data = ir7[3];
              ir7[3] = (v496_data + (v474_data * (sycl::select_from_group(item.get_sub_group(), v169_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v502_data = ir7[4];
              ir7[4] = (v502_data + (v474_data * (sycl::select_from_group(item.get_sub_group(), v175_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v508_data = ir7[5];
              ir7[5] = (v508_data + (v474_data * (sycl::select_from_group(item.get_sub_group(), v181_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v514_data = ir7[6];
              ir7[6] = (v514_data + (v474_data * (sycl::select_from_group(item.get_sub_group(), v187_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v520_data = ir7[7];
              ir7[7] = (v520_data + (v474_data * (sycl::select_from_group(item.get_sub_group(), v193_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v526_data = ir7[8];
              ir7[8] = (v526_data + (v474_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v528_data = r5[7];
              float v532_data = ir7[0];
              ir7[0] = (v532_data + (v528_data * (sycl::select_from_group(item.get_sub_group(), v151_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v538_data = ir7[1];
              ir7[1] = (v538_data + (v528_data * (sycl::select_from_group(item.get_sub_group(), v157_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v544_data = ir7[2];
              ir7[2] = (v544_data + (v528_data * (sycl::select_from_group(item.get_sub_group(), v163_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v550_data = ir7[3];
              ir7[3] = (v550_data + (v528_data * (sycl::select_from_group(item.get_sub_group(), v169_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v556_data = ir7[4];
              ir7[4] = (v556_data + (v528_data * (sycl::select_from_group(item.get_sub_group(), v175_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v562_data = ir7[5];
              ir7[5] = (v562_data + (v528_data * (sycl::select_from_group(item.get_sub_group(), v181_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v568_data = ir7[6];
              ir7[6] = (v568_data + (v528_data * (sycl::select_from_group(item.get_sub_group(), v187_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v574_data = ir7[7];
              ir7[7] = (v574_data + (v528_data * (sycl::select_from_group(item.get_sub_group(), v193_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v580_data = ir7[8];
              ir7[8] = (v580_data + (v528_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v582_data = r5[8];
              float v586_data = ir7[0];
              ir7[0] = (v586_data + (v582_data * (sycl::select_from_group(item.get_sub_group(), v151_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v592_data = ir7[1];
              ir7[1] = (v592_data + (v582_data * (sycl::select_from_group(item.get_sub_group(), v157_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v598_data = ir7[2];
              ir7[2] = (v598_data + (v582_data * (sycl::select_from_group(item.get_sub_group(), v163_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v604_data = ir7[3];
              ir7[3] = (v604_data + (v582_data * (sycl::select_from_group(item.get_sub_group(), v169_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v610_data = ir7[4];
              ir7[4] = (v610_data + (v582_data * (sycl::select_from_group(item.get_sub_group(), v175_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v616_data = ir7[5];
              ir7[5] = (v616_data + (v582_data * (sycl::select_from_group(item.get_sub_group(), v181_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v622_data = ir7[6];
              ir7[6] = (v622_data + (v582_data * (sycl::select_from_group(item.get_sub_group(), v187_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v628_data = ir7[7];
              ir7[7] = (v628_data + (v582_data * (sycl::select_from_group(item.get_sub_group(), v193_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v634_data = ir7[8];
              ir7[8] = (v634_data + (v582_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              // r7 = ir7
              #pragma unroll
              for (int32_t v636_n0 = 0; v636_n0 < 1; ++v636_n0) {
                #pragma unroll
                for (int32_t v637_n1 = 0; v637_n1 < 9; ++v637_n1) {
                  int32_t v638_a = v636_n0 + v637_n1;
                  float v639_data = ir7[v638_a];
                  r7[v638_a] = v639_data;
                }
              }
              // glb_m3 = store{r>g}(r7);
              #pragma unroll
              for (int32_t v640_i0 = 0; v640_i0 < 1; ++v640_i0) {
                int32_t v645_lead = v17_lead + (v640_i0 * 32);
                #pragma unroll
                for (int32_t v641_i1 = 0; v641_i1 < 9; ++v641_i1) {
                  float v643_data = r7[(v640_i0 + v641_i1)];
                  glb_m3[(v645_lead + (v641_i1 * 32))] = v643_data;
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

