// === base name ===
kernel_ad7bb3b1c5157cd8

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_ad7bb3b1c5157cd8 = {{32, 1, 1}, 32, 32, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_ad7bb3b1c5157cd8(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_ad7bb3b1c5157cd8(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_ad7bb3b1c5157cd8(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_ad7bb3b1c5157cd8(const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_ad7bb3b1c5157cd8(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_ad7bb3b1c5157cd8(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_ad7bb3b1c5157cd8(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float ** m0, size_t m0_extraOffset, const float ** m1, size_t m1_extraOffset, const float ** m2, size_t m2_extraOffset, float ** m3, size_t m3_extraOffset, const float ** m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
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
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"pointer_based","alias":"Q","bbox":[[0,0],[32,9]],"name":"m0","ordered":false,"parts":1,"shape":[32,9],"variant":false},{"addressing":"pointer_based","alias":"F0","bbox":[[0,0],[16,9]],"name":"m1","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"pointer_based","alias":"F1","bbox":[[0,0],[16,9]],"name":"m2","ordered":false,"parts":1,"shape":[16,9],"variant":false},{"addressing":"pointer_based","alias":"O","bbox":[[0,0],[32,9]],"name":"m3","ordered":false,"parts":1,"shape":[32,9],"variant":false},{"addressing":"pointer_based","alias":"M","bbox":[[0,0],[9,9]],"name":"m4","ordered":false,"parts":1,"shape":[9,9],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[32,9]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"pointer_based","bbox":[[0,0],[16,9]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,9]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[32,9]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,9]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,9]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,9]},{"addressing":"pointer_based","bbox":[[0,0],[9,9]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[9,9]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v7_batchId0][0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v7_batchId0][0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v7_batchId0][0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v7_batchId0][0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v7_batchId0][0 + m4_extraOffset];
              float r0[9]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v23_lead = item.get_local_id(2) % 32;
              #pragma unroll
              for (int32_t v24_i0 = 0; v24_i0 < 1; ++v24_i0) {
                int32_t v27_lead = v23_lead + (v24_i0 * 32);
                #pragma unroll
                for (int32_t v25_i1 = 0; v25_i1 < 9; ++v25_i1) {
                  float v30_data = glb_m0[(v27_lead + (v25_i1 * 32))];
                  r0[(v24_i0 + v25_i1)] = v30_data;
                }
              }
              float r2[9]{};
              // r2 = load{g>r}(glb_m1);
              bool v33_g = v23_lead < 16;
              if (v33_g) {
                #pragma unroll
                for (int32_t v34_i1 = 0; v34_i1 < 9; ++v34_i1) {
                  float v39_data = glb_m1[(v23_lead + (v34_i1 * 16))];
                  r2[v34_i1] = v39_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r1[9]{};
              // r1 = +(r0) + None
              // [(0, 32), (0, 9)] []
              float v42_data = r0[0];
              float v43_data = r1[0];
              r1[0] = (v43_data + v42_data);
              float v45_data = r0[1];
              float v46_data = r1[1];
              r1[1] = (v46_data + v45_data);
              float v48_data = r0[2];
              float v49_data = r1[2];
              r1[2] = (v49_data + v48_data);
              float v51_data = r0[3];
              float v52_data = r1[3];
              r1[3] = (v52_data + v51_data);
              float v54_data = r0[4];
              float v55_data = r1[4];
              r1[4] = (v55_data + v54_data);
              float v57_data = r0[5];
              float v58_data = r1[5];
              r1[5] = (v58_data + v57_data);
              float v60_data = r0[6];
              float v61_data = r1[6];
              r1[6] = (v61_data + v60_data);
              float v63_data = r0[7];
              float v64_data = r1[7];
              r1[7] = (v64_data + v63_data);
              float v66_data = r0[8];
              float v67_data = r1[8];
              r1[8] = (v67_data + v66_data);
              float r4[9]{};
              // r4 = load{g>r}(glb_m2);
              if (v33_g) {
                #pragma unroll
                for (int32_t v70_i1 = 0; v70_i1 < 9; ++v70_i1) {
                  float v75_data = glb_m2[(v23_lead + (v70_i1 * 16))];
                  r4[v70_i1] = v75_data;
                }
              }
              // wait(r2 = load{g>r}(glb_m1););
              float r3[9]{};
              // ir3 = +(r2)
              // [(0, 16), (0, 9)] []
              float ir3[9]{};
              float v79_data = r2[0];
              float v80_data = ir3[0];
              ir3[0] = (v80_data + v79_data);
              float v82_data = r2[1];
              float v83_data = ir3[1];
              ir3[1] = (v83_data + v82_data);
              float v85_data = r2[2];
              float v86_data = ir3[2];
              ir3[2] = (v86_data + v85_data);
              float v88_data = r2[3];
              float v89_data = ir3[3];
              ir3[3] = (v89_data + v88_data);
              float v91_data = r2[4];
              float v92_data = ir3[4];
              ir3[4] = (v92_data + v91_data);
              float v94_data = r2[5];
              float v95_data = ir3[5];
              ir3[5] = (v95_data + v94_data);
              float v97_data = r2[6];
              float v98_data = ir3[6];
              ir3[6] = (v98_data + v97_data);
              float v100_data = r2[7];
              float v101_data = ir3[7];
              ir3[7] = (v101_data + v100_data);
              float v103_data = r2[8];
              float v104_data = ir3[8];
              ir3[8] = (v104_data + v103_data);
              // r3 = ir3 + r1
              #pragma unroll
              for (int32_t v106_n1 = 0; v106_n1 < 9; ++v106_n1) {
                float v108_data = ir3[v106_n1];
                float v109_data = r1[v106_n1];
                r3[v106_n1] = (v109_data + v108_data);
              }
              float r6[9]{};
              // r6 = load{g>r}(glb_m4);
              if (v23_lead < 9) {
                #pragma unroll
                for (int32_t v113_i1 = 0; v113_i1 < 9; ++v113_i1) {
                  float v118_data = glb_m4[(v23_lead + (v113_i1 * 9))];
                  r6[v113_i1] = v118_data;
                }
              }
              // wait(r4 = load{g>r}(glb_m2););
              float r5[9]{};
              // ir5 = +(r4)
              // [(0, 16), (0, 9)] []
              float ir5[9]{};
              float v122_data = r4[0];
              float v123_data = ir5[0];
              ir5[0] = (v123_data + v122_data);
              float v125_data = r4[1];
              float v126_data = ir5[1];
              ir5[1] = (v126_data + v125_data);
              float v128_data = r4[2];
              float v129_data = ir5[2];
              ir5[2] = (v129_data + v128_data);
              float v131_data = r4[3];
              float v132_data = ir5[3];
              ir5[3] = (v132_data + v131_data);
              float v134_data = r4[4];
              float v135_data = ir5[4];
              ir5[4] = (v135_data + v134_data);
              float v137_data = r4[5];
              float v138_data = ir5[5];
              ir5[5] = (v138_data + v137_data);
              float v140_data = r4[6];
              float v141_data = ir5[6];
              ir5[6] = (v141_data + v140_data);
              float v143_data = r4[7];
              float v144_data = ir5[7];
              ir5[7] = (v144_data + v143_data);
              float v146_data = r4[8];
              float v147_data = ir5[8];
              ir5[8] = (v147_data + v146_data);
              // r5 = ir5 + r3
              #pragma unroll
              for (int32_t v149_n1 = 0; v149_n1 < 9; ++v149_n1) {
                float v151_data = ir5[v149_n1];
                float v152_data = r3[v149_n1];
                r5[v149_n1] = (v152_data + v151_data);
              }
              // wait(r6 = load{g>r}(glb_m4););
              float r7[9]{};
              // ir7 = +(r5 * r6)
              // [(0, 32), (0, 9)] [(0, 9)]
              float ir7[9]{};
              float v156_data = r5[0];
              float v157_data = r6[0];
              float v160_data = ir7[0];
              ir7[0] = (v160_data + (v156_data * (sycl::select_from_group(item.get_sub_group(), v157_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v163_data = r6[1];
              float v166_data = ir7[1];
              ir7[1] = (v166_data + (v156_data * (sycl::select_from_group(item.get_sub_group(), v163_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v169_data = r6[2];
              float v172_data = ir7[2];
              ir7[2] = (v172_data + (v156_data * (sycl::select_from_group(item.get_sub_group(), v169_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v175_data = r6[3];
              float v178_data = ir7[3];
              ir7[3] = (v178_data + (v156_data * (sycl::select_from_group(item.get_sub_group(), v175_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v181_data = r6[4];
              float v184_data = ir7[4];
              ir7[4] = (v184_data + (v156_data * (sycl::select_from_group(item.get_sub_group(), v181_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v187_data = r6[5];
              float v190_data = ir7[5];
              ir7[5] = (v190_data + (v156_data * (sycl::select_from_group(item.get_sub_group(), v187_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v193_data = r6[6];
              float v196_data = ir7[6];
              ir7[6] = (v196_data + (v156_data * (sycl::select_from_group(item.get_sub_group(), v193_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v199_data = r6[7];
              float v202_data = ir7[7];
              ir7[7] = (v202_data + (v156_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v205_data = r6[8];
              float v208_data = ir7[8];
              ir7[8] = (v208_data + (v156_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v210_data = r5[1];
              float v214_data = ir7[0];
              ir7[0] = (v214_data + (v210_data * (sycl::select_from_group(item.get_sub_group(), v157_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v220_data = ir7[1];
              ir7[1] = (v220_data + (v210_data * (sycl::select_from_group(item.get_sub_group(), v163_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v226_data = ir7[2];
              ir7[2] = (v226_data + (v210_data * (sycl::select_from_group(item.get_sub_group(), v169_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v232_data = ir7[3];
              ir7[3] = (v232_data + (v210_data * (sycl::select_from_group(item.get_sub_group(), v175_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v238_data = ir7[4];
              ir7[4] = (v238_data + (v210_data * (sycl::select_from_group(item.get_sub_group(), v181_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v244_data = ir7[5];
              ir7[5] = (v244_data + (v210_data * (sycl::select_from_group(item.get_sub_group(), v187_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v250_data = ir7[6];
              ir7[6] = (v250_data + (v210_data * (sycl::select_from_group(item.get_sub_group(), v193_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v256_data = ir7[7];
              ir7[7] = (v256_data + (v210_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v262_data = ir7[8];
              ir7[8] = (v262_data + (v210_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v264_data = r5[2];
              float v268_data = ir7[0];
              ir7[0] = (v268_data + (v264_data * (sycl::select_from_group(item.get_sub_group(), v157_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v274_data = ir7[1];
              ir7[1] = (v274_data + (v264_data * (sycl::select_from_group(item.get_sub_group(), v163_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v280_data = ir7[2];
              ir7[2] = (v280_data + (v264_data * (sycl::select_from_group(item.get_sub_group(), v169_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v286_data = ir7[3];
              ir7[3] = (v286_data + (v264_data * (sycl::select_from_group(item.get_sub_group(), v175_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v292_data = ir7[4];
              ir7[4] = (v292_data + (v264_data * (sycl::select_from_group(item.get_sub_group(), v181_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v298_data = ir7[5];
              ir7[5] = (v298_data + (v264_data * (sycl::select_from_group(item.get_sub_group(), v187_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v304_data = ir7[6];
              ir7[6] = (v304_data + (v264_data * (sycl::select_from_group(item.get_sub_group(), v193_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v310_data = ir7[7];
              ir7[7] = (v310_data + (v264_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v316_data = ir7[8];
              ir7[8] = (v316_data + (v264_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v318_data = r5[3];
              float v322_data = ir7[0];
              ir7[0] = (v322_data + (v318_data * (sycl::select_from_group(item.get_sub_group(), v157_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v328_data = ir7[1];
              ir7[1] = (v328_data + (v318_data * (sycl::select_from_group(item.get_sub_group(), v163_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v334_data = ir7[2];
              ir7[2] = (v334_data + (v318_data * (sycl::select_from_group(item.get_sub_group(), v169_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v340_data = ir7[3];
              ir7[3] = (v340_data + (v318_data * (sycl::select_from_group(item.get_sub_group(), v175_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v346_data = ir7[4];
              ir7[4] = (v346_data + (v318_data * (sycl::select_from_group(item.get_sub_group(), v181_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v352_data = ir7[5];
              ir7[5] = (v352_data + (v318_data * (sycl::select_from_group(item.get_sub_group(), v187_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v358_data = ir7[6];
              ir7[6] = (v358_data + (v318_data * (sycl::select_from_group(item.get_sub_group(), v193_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v364_data = ir7[7];
              ir7[7] = (v364_data + (v318_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v370_data = ir7[8];
              ir7[8] = (v370_data + (v318_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v372_data = r5[4];
              float v376_data = ir7[0];
              ir7[0] = (v376_data + (v372_data * (sycl::select_from_group(item.get_sub_group(), v157_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v382_data = ir7[1];
              ir7[1] = (v382_data + (v372_data * (sycl::select_from_group(item.get_sub_group(), v163_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v388_data = ir7[2];
              ir7[2] = (v388_data + (v372_data * (sycl::select_from_group(item.get_sub_group(), v169_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v394_data = ir7[3];
              ir7[3] = (v394_data + (v372_data * (sycl::select_from_group(item.get_sub_group(), v175_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v400_data = ir7[4];
              ir7[4] = (v400_data + (v372_data * (sycl::select_from_group(item.get_sub_group(), v181_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v406_data = ir7[5];
              ir7[5] = (v406_data + (v372_data * (sycl::select_from_group(item.get_sub_group(), v187_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v412_data = ir7[6];
              ir7[6] = (v412_data + (v372_data * (sycl::select_from_group(item.get_sub_group(), v193_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v418_data = ir7[7];
              ir7[7] = (v418_data + (v372_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v424_data = ir7[8];
              ir7[8] = (v424_data + (v372_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v426_data = r5[5];
              float v430_data = ir7[0];
              ir7[0] = (v430_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v157_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v436_data = ir7[1];
              ir7[1] = (v436_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v163_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v442_data = ir7[2];
              ir7[2] = (v442_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v169_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v448_data = ir7[3];
              ir7[3] = (v448_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v175_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v454_data = ir7[4];
              ir7[4] = (v454_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v181_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v460_data = ir7[5];
              ir7[5] = (v460_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v187_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v466_data = ir7[6];
              ir7[6] = (v466_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v193_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v472_data = ir7[7];
              ir7[7] = (v472_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v478_data = ir7[8];
              ir7[8] = (v478_data + (v426_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v480_data = r5[6];
              float v484_data = ir7[0];
              ir7[0] = (v484_data + (v480_data * (sycl::select_from_group(item.get_sub_group(), v157_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v490_data = ir7[1];
              ir7[1] = (v490_data + (v480_data * (sycl::select_from_group(item.get_sub_group(), v163_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v496_data = ir7[2];
              ir7[2] = (v496_data + (v480_data * (sycl::select_from_group(item.get_sub_group(), v169_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v502_data = ir7[3];
              ir7[3] = (v502_data + (v480_data * (sycl::select_from_group(item.get_sub_group(), v175_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v508_data = ir7[4];
              ir7[4] = (v508_data + (v480_data * (sycl::select_from_group(item.get_sub_group(), v181_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v514_data = ir7[5];
              ir7[5] = (v514_data + (v480_data * (sycl::select_from_group(item.get_sub_group(), v187_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v520_data = ir7[6];
              ir7[6] = (v520_data + (v480_data * (sycl::select_from_group(item.get_sub_group(), v193_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v526_data = ir7[7];
              ir7[7] = (v526_data + (v480_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v532_data = ir7[8];
              ir7[8] = (v532_data + (v480_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v534_data = r5[7];
              float v538_data = ir7[0];
              ir7[0] = (v538_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v157_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v544_data = ir7[1];
              ir7[1] = (v544_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v163_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v550_data = ir7[2];
              ir7[2] = (v550_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v169_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v556_data = ir7[3];
              ir7[3] = (v556_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v175_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v562_data = ir7[4];
              ir7[4] = (v562_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v181_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v568_data = ir7[5];
              ir7[5] = (v568_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v187_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v574_data = ir7[6];
              ir7[6] = (v574_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v193_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v580_data = ir7[7];
              ir7[7] = (v580_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v586_data = ir7[8];
              ir7[8] = (v586_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v588_data = r5[8];
              float v592_data = ir7[0];
              ir7[0] = (v592_data + (v588_data * (sycl::select_from_group(item.get_sub_group(), v157_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v598_data = ir7[1];
              ir7[1] = (v598_data + (v588_data * (sycl::select_from_group(item.get_sub_group(), v163_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v604_data = ir7[2];
              ir7[2] = (v604_data + (v588_data * (sycl::select_from_group(item.get_sub_group(), v169_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v610_data = ir7[3];
              ir7[3] = (v610_data + (v588_data * (sycl::select_from_group(item.get_sub_group(), v175_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v616_data = ir7[4];
              ir7[4] = (v616_data + (v588_data * (sycl::select_from_group(item.get_sub_group(), v181_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v622_data = ir7[5];
              ir7[5] = (v622_data + (v588_data * (sycl::select_from_group(item.get_sub_group(), v187_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v628_data = ir7[6];
              ir7[6] = (v628_data + (v588_data * (sycl::select_from_group(item.get_sub_group(), v193_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v634_data = ir7[7];
              ir7[7] = (v634_data + (v588_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v640_data = ir7[8];
              ir7[8] = (v640_data + (v588_data * (sycl::select_from_group(item.get_sub_group(), v205_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              // r7 = ir7
              #pragma unroll
              for (int32_t v642_n0 = 0; v642_n0 < 1; ++v642_n0) {
                #pragma unroll
                for (int32_t v643_n1 = 0; v643_n1 < 9; ++v643_n1) {
                  int32_t v644_a = v642_n0 + v643_n1;
                  float v645_data = ir7[v644_a];
                  r7[v644_a] = v645_data;
                }
              }
              // glb_m3 = store{r>g}(r7);
              #pragma unroll
              for (int32_t v646_i0 = 0; v646_i0 < 1; ++v646_i0) {
                int32_t v651_lead = v23_lead + (v646_i0 * 32);
                #pragma unroll
                for (int32_t v647_i1 = 0; v647_i1 < 9; ++v647_i1) {
                  float v649_data = r7[(v646_i0 + v647_i1)];
                  glb_m3[(v651_lead + (v647_i1 * 32))] = v649_data;
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

