// === base name ===
kernel_1062f44ca70d2951

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_1062f44ca70d2951 = {{32, 1, 1}, 32, 32, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_1062f44ca70d2951(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_1062f44ca70d2951(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_1062f44ca70d2951(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_1062f44ca70d2951(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_1062f44ca70d2951(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_1062f44ca70d2951(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, m6, m6_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_1062f44ca70d2951(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes x 1 per block = block 32x1x1, 0 B shared, occupancy grid
        // operands:
        //   m0 32×16(32×16) {0..32}×{0..16} strided
        //   m1 32×12(32×12) {0..32}×{0..12} strided
        //   m2 12×16(12×16) {0..12}×{0..16} strided
        //   m3 32×12(32×12) {0..32}×{0..12} strided
        //   m4 12×8(12×8) {0..12}×{0..8} strided
        //   m5 32×12(32×12) {0..32}×{0..12} strided
        //   m6 12×8(12×8) {0..12}×{0..8} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        //   m0[i,j]@{0..32}×{0..8} += m3[i,k] × m4[k,j]
        //   m0[i,j]@{0..32}×{8..16} += m5[i,k] × m6[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[32,16]],"name":"m0","ordered":false,"parts":1,"shape":[32,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[32,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,12],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[12,16]],"name":"m2","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"A0","bbox":[[0,0],[32,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,12],"variant":false},{"addressing":"strided","alias":"B0","bbox":[[0,0],[12,8]],"name":"m4","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A1","bbox":[[0,0],[32,12]],"name":"m5","ordered":false,"parts":1,"shape":[32,12],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[12,8]],"name":"m6","ordered":false,"parts":1,"shape":[12,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,12]},{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[12,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,8]],"is_tmp":false,"name":"m0","offset":[0,8],"shape":[32,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[32,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v7_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 512 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 384 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 192 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v7_batchId0 * 384 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v7_batchId0 * 96 + 0 + m4_extraOffset];
              const float *const __restrict__ glb_m5 = &m5[v7_batchId0 * 384 + 0 + m5_extraOffset];
              const float *const __restrict__ glb_m6 = &m6[v7_batchId0 * 96 + 0 + m6_extraOffset];
              float r0[12]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v25_lead = item.get_local_id(2) % 32;
              #pragma unroll
              for (int32_t v26_i0 = 0; v26_i0 < 1; ++v26_i0) {
                int32_t v29_lead = v25_lead + (v26_i0 * 32);
                #pragma unroll
                for (int32_t v27_i1 = 0; v27_i1 < 12; ++v27_i1) {
                  float v32_data = glb_m1[(v29_lead + (v27_i1 * 32))];
                  r0[(v26_i0 + v27_i1)] = v32_data;
                }
              }
              float r1[16]{};
              // r1 = load{g>r}(glb_m2);
              bool v35_g = v25_lead < 12;
              if (v35_g) {
                #pragma unroll
                for (int32_t v36_i1 = 0; v36_i1 < 16; ++v36_i1) {
                  float v41_data = glb_m2[(v25_lead + (v36_i1 * 12))];
                  r1[v36_i1] = v41_data;
                }
              }
              float r3[12]{};
              // r3 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v1210_i0 = 0; v1210_i0 < 1; ++v1210_i0) {
                int32_t v1213_lead = v25_lead + (v1210_i0 * 32);
                #pragma unroll
                for (int32_t v1211_i1 = 0; v1211_i1 < 12; ++v1211_i1) {
                  float v1216_data = glb_m3[(v1213_lead + (v1211_i1 * 32))];
                  r3[(v1210_i0 + v1211_i1)] = v1216_data;
                }
              }
              float r2[16]{};
              // ir2 = +(r0 * r1)
              // [(0, 32), (0, 16)] [(0, 12)]
              float ir2[16]{};
              float v45_data = r0[0];
              float v46_data = r1[0];
              float v49_data = ir2[0];
              ir2[0] = (v49_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v52_data = r1[1];
              float v55_data = ir2[1];
              ir2[1] = (v55_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v58_data = r1[2];
              float v61_data = ir2[2];
              ir2[2] = (v61_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v64_data = r1[3];
              float v67_data = ir2[3];
              ir2[3] = (v67_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v70_data = r1[4];
              float v73_data = ir2[4];
              ir2[4] = (v73_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v76_data = r1[5];
              float v79_data = ir2[5];
              ir2[5] = (v79_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v82_data = r1[6];
              float v85_data = ir2[6];
              ir2[6] = (v85_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v88_data = r1[7];
              float v91_data = ir2[7];
              ir2[7] = (v91_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v94_data = r1[8];
              float v97_data = ir2[8];
              ir2[8] = (v97_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v100_data = r1[9];
              float v103_data = ir2[9];
              ir2[9] = (v103_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v106_data = r1[10];
              float v109_data = ir2[10];
              ir2[10] = (v109_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v112_data = r1[11];
              float v115_data = ir2[11];
              ir2[11] = (v115_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v118_data = r1[12];
              float v121_data = ir2[12];
              ir2[12] = (v121_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v124_data = r1[13];
              float v127_data = ir2[13];
              ir2[13] = (v127_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v124_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v130_data = r1[14];
              float v133_data = ir2[14];
              ir2[14] = (v133_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v130_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v136_data = r1[15];
              float v139_data = ir2[15];
              ir2[15] = (v139_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v136_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v141_data = r0[1];
              float v145_data = ir2[0];
              ir2[0] = (v145_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v151_data = ir2[1];
              ir2[1] = (v151_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v157_data = ir2[2];
              ir2[2] = (v157_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v163_data = ir2[3];
              ir2[3] = (v163_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v169_data = ir2[4];
              ir2[4] = (v169_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v175_data = ir2[5];
              ir2[5] = (v175_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v181_data = ir2[6];
              ir2[6] = (v181_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v187_data = ir2[7];
              ir2[7] = (v187_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v193_data = ir2[8];
              ir2[8] = (v193_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v199_data = ir2[9];
              ir2[9] = (v199_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v205_data = ir2[10];
              ir2[10] = (v205_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v211_data = ir2[11];
              ir2[11] = (v211_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v217_data = ir2[12];
              ir2[12] = (v217_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v223_data = ir2[13];
              ir2[13] = (v223_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v124_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v229_data = ir2[14];
              ir2[14] = (v229_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v130_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v235_data = ir2[15];
              ir2[15] = (v235_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v136_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v237_data = r0[2];
              float v241_data = ir2[0];
              ir2[0] = (v241_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v247_data = ir2[1];
              ir2[1] = (v247_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v253_data = ir2[2];
              ir2[2] = (v253_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v259_data = ir2[3];
              ir2[3] = (v259_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v265_data = ir2[4];
              ir2[4] = (v265_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v271_data = ir2[5];
              ir2[5] = (v271_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v277_data = ir2[6];
              ir2[6] = (v277_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v283_data = ir2[7];
              ir2[7] = (v283_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v289_data = ir2[8];
              ir2[8] = (v289_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v295_data = ir2[9];
              ir2[9] = (v295_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v301_data = ir2[10];
              ir2[10] = (v301_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v307_data = ir2[11];
              ir2[11] = (v307_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v313_data = ir2[12];
              ir2[12] = (v313_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v319_data = ir2[13];
              ir2[13] = (v319_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v124_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v325_data = ir2[14];
              ir2[14] = (v325_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v130_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v331_data = ir2[15];
              ir2[15] = (v331_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v136_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v333_data = r0[3];
              float v337_data = ir2[0];
              ir2[0] = (v337_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v343_data = ir2[1];
              ir2[1] = (v343_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v349_data = ir2[2];
              ir2[2] = (v349_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v355_data = ir2[3];
              ir2[3] = (v355_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v361_data = ir2[4];
              ir2[4] = (v361_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v367_data = ir2[5];
              ir2[5] = (v367_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v373_data = ir2[6];
              ir2[6] = (v373_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v379_data = ir2[7];
              ir2[7] = (v379_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v385_data = ir2[8];
              ir2[8] = (v385_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v391_data = ir2[9];
              ir2[9] = (v391_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v397_data = ir2[10];
              ir2[10] = (v397_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v403_data = ir2[11];
              ir2[11] = (v403_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v409_data = ir2[12];
              ir2[12] = (v409_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v415_data = ir2[13];
              ir2[13] = (v415_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v124_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v421_data = ir2[14];
              ir2[14] = (v421_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v130_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v427_data = ir2[15];
              ir2[15] = (v427_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v136_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v429_data = r0[4];
              float v433_data = ir2[0];
              ir2[0] = (v433_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v439_data = ir2[1];
              ir2[1] = (v439_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v445_data = ir2[2];
              ir2[2] = (v445_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v451_data = ir2[3];
              ir2[3] = (v451_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v457_data = ir2[4];
              ir2[4] = (v457_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v463_data = ir2[5];
              ir2[5] = (v463_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v469_data = ir2[6];
              ir2[6] = (v469_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v475_data = ir2[7];
              ir2[7] = (v475_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v481_data = ir2[8];
              ir2[8] = (v481_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v487_data = ir2[9];
              ir2[9] = (v487_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v493_data = ir2[10];
              ir2[10] = (v493_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v499_data = ir2[11];
              ir2[11] = (v499_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v505_data = ir2[12];
              ir2[12] = (v505_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v511_data = ir2[13];
              ir2[13] = (v511_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v124_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v517_data = ir2[14];
              ir2[14] = (v517_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v130_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v523_data = ir2[15];
              ir2[15] = (v523_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v136_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v525_data = r0[5];
              float v529_data = ir2[0];
              ir2[0] = (v529_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v535_data = ir2[1];
              ir2[1] = (v535_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v541_data = ir2[2];
              ir2[2] = (v541_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v547_data = ir2[3];
              ir2[3] = (v547_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v553_data = ir2[4];
              ir2[4] = (v553_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v559_data = ir2[5];
              ir2[5] = (v559_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v565_data = ir2[6];
              ir2[6] = (v565_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v571_data = ir2[7];
              ir2[7] = (v571_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v577_data = ir2[8];
              ir2[8] = (v577_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v583_data = ir2[9];
              ir2[9] = (v583_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v589_data = ir2[10];
              ir2[10] = (v589_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v595_data = ir2[11];
              ir2[11] = (v595_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v601_data = ir2[12];
              ir2[12] = (v601_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v607_data = ir2[13];
              ir2[13] = (v607_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v124_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v613_data = ir2[14];
              ir2[14] = (v613_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v130_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v619_data = ir2[15];
              ir2[15] = (v619_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v136_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v621_data = r0[6];
              float v625_data = ir2[0];
              ir2[0] = (v625_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v631_data = ir2[1];
              ir2[1] = (v631_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v637_data = ir2[2];
              ir2[2] = (v637_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v643_data = ir2[3];
              ir2[3] = (v643_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v649_data = ir2[4];
              ir2[4] = (v649_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v655_data = ir2[5];
              ir2[5] = (v655_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v661_data = ir2[6];
              ir2[6] = (v661_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v667_data = ir2[7];
              ir2[7] = (v667_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v673_data = ir2[8];
              ir2[8] = (v673_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v679_data = ir2[9];
              ir2[9] = (v679_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v685_data = ir2[10];
              ir2[10] = (v685_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v691_data = ir2[11];
              ir2[11] = (v691_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v697_data = ir2[12];
              ir2[12] = (v697_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v703_data = ir2[13];
              ir2[13] = (v703_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v124_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v709_data = ir2[14];
              ir2[14] = (v709_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v130_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v715_data = ir2[15];
              ir2[15] = (v715_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v136_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v717_data = r0[7];
              float v721_data = ir2[0];
              ir2[0] = (v721_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v727_data = ir2[1];
              ir2[1] = (v727_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v733_data = ir2[2];
              ir2[2] = (v733_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v739_data = ir2[3];
              ir2[3] = (v739_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v745_data = ir2[4];
              ir2[4] = (v745_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v751_data = ir2[5];
              ir2[5] = (v751_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v757_data = ir2[6];
              ir2[6] = (v757_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v763_data = ir2[7];
              ir2[7] = (v763_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v769_data = ir2[8];
              ir2[8] = (v769_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v775_data = ir2[9];
              ir2[9] = (v775_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v781_data = ir2[10];
              ir2[10] = (v781_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v787_data = ir2[11];
              ir2[11] = (v787_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v793_data = ir2[12];
              ir2[12] = (v793_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v799_data = ir2[13];
              ir2[13] = (v799_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v124_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v805_data = ir2[14];
              ir2[14] = (v805_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v130_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v811_data = ir2[15];
              ir2[15] = (v811_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v136_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v813_data = r0[8];
              float v817_data = ir2[0];
              ir2[0] = (v817_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v823_data = ir2[1];
              ir2[1] = (v823_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v829_data = ir2[2];
              ir2[2] = (v829_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v835_data = ir2[3];
              ir2[3] = (v835_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v841_data = ir2[4];
              ir2[4] = (v841_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v847_data = ir2[5];
              ir2[5] = (v847_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v853_data = ir2[6];
              ir2[6] = (v853_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v859_data = ir2[7];
              ir2[7] = (v859_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v865_data = ir2[8];
              ir2[8] = (v865_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v871_data = ir2[9];
              ir2[9] = (v871_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v877_data = ir2[10];
              ir2[10] = (v877_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v883_data = ir2[11];
              ir2[11] = (v883_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v889_data = ir2[12];
              ir2[12] = (v889_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v895_data = ir2[13];
              ir2[13] = (v895_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v124_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v901_data = ir2[14];
              ir2[14] = (v901_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v130_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v907_data = ir2[15];
              ir2[15] = (v907_data + (v813_data * (sycl::select_from_group(item.get_sub_group(), v136_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v909_data = r0[9];
              float v913_data = ir2[0];
              ir2[0] = (v913_data + (v909_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v919_data = ir2[1];
              ir2[1] = (v919_data + (v909_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v925_data = ir2[2];
              ir2[2] = (v925_data + (v909_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v931_data = ir2[3];
              ir2[3] = (v931_data + (v909_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v937_data = ir2[4];
              ir2[4] = (v937_data + (v909_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v943_data = ir2[5];
              ir2[5] = (v943_data + (v909_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v949_data = ir2[6];
              ir2[6] = (v949_data + (v909_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v955_data = ir2[7];
              ir2[7] = (v955_data + (v909_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v961_data = ir2[8];
              ir2[8] = (v961_data + (v909_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v967_data = ir2[9];
              ir2[9] = (v967_data + (v909_data * (sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v973_data = ir2[10];
              ir2[10] = (v973_data + (v909_data * (sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v979_data = ir2[11];
              ir2[11] = (v979_data + (v909_data * (sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v985_data = ir2[12];
              ir2[12] = (v985_data + (v909_data * (sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v991_data = ir2[13];
              ir2[13] = (v991_data + (v909_data * (sycl::select_from_group(item.get_sub_group(), v124_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v997_data = ir2[14];
              ir2[14] = (v997_data + (v909_data * (sycl::select_from_group(item.get_sub_group(), v130_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1003_data = ir2[15];
              ir2[15] = (v1003_data + (v909_data * (sycl::select_from_group(item.get_sub_group(), v136_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1005_data = r0[10];
              float v1009_data = ir2[0];
              ir2[0] = (v1009_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1015_data = ir2[1];
              ir2[1] = (v1015_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1021_data = ir2[2];
              ir2[2] = (v1021_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1027_data = ir2[3];
              ir2[3] = (v1027_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1033_data = ir2[4];
              ir2[4] = (v1033_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1039_data = ir2[5];
              ir2[5] = (v1039_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1045_data = ir2[6];
              ir2[6] = (v1045_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1051_data = ir2[7];
              ir2[7] = (v1051_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1057_data = ir2[8];
              ir2[8] = (v1057_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1063_data = ir2[9];
              ir2[9] = (v1063_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1069_data = ir2[10];
              ir2[10] = (v1069_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1075_data = ir2[11];
              ir2[11] = (v1075_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1081_data = ir2[12];
              ir2[12] = (v1081_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1087_data = ir2[13];
              ir2[13] = (v1087_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v124_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1093_data = ir2[14];
              ir2[14] = (v1093_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v130_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1099_data = ir2[15];
              ir2[15] = (v1099_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v136_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1101_data = r0[11];
              float v1105_data = ir2[0];
              ir2[0] = (v1105_data + (v1101_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1111_data = ir2[1];
              ir2[1] = (v1111_data + (v1101_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1117_data = ir2[2];
              ir2[2] = (v1117_data + (v1101_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1123_data = ir2[3];
              ir2[3] = (v1123_data + (v1101_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1129_data = ir2[4];
              ir2[4] = (v1129_data + (v1101_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1135_data = ir2[5];
              ir2[5] = (v1135_data + (v1101_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1141_data = ir2[6];
              ir2[6] = (v1141_data + (v1101_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1147_data = ir2[7];
              ir2[7] = (v1147_data + (v1101_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1153_data = ir2[8];
              ir2[8] = (v1153_data + (v1101_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1159_data = ir2[9];
              ir2[9] = (v1159_data + (v1101_data * (sycl::select_from_group(item.get_sub_group(), v100_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1165_data = ir2[10];
              ir2[10] = (v1165_data + (v1101_data * (sycl::select_from_group(item.get_sub_group(), v106_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1171_data = ir2[11];
              ir2[11] = (v1171_data + (v1101_data * (sycl::select_from_group(item.get_sub_group(), v112_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1177_data = ir2[12];
              ir2[12] = (v1177_data + (v1101_data * (sycl::select_from_group(item.get_sub_group(), v118_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1183_data = ir2[13];
              ir2[13] = (v1183_data + (v1101_data * (sycl::select_from_group(item.get_sub_group(), v124_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1189_data = ir2[14];
              ir2[14] = (v1189_data + (v1101_data * (sycl::select_from_group(item.get_sub_group(), v130_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1195_data = ir2[15];
              ir2[15] = (v1195_data + (v1101_data * (sycl::select_from_group(item.get_sub_group(), v136_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              // r2 = ir2
              #pragma unroll
              for (int32_t v1197_n0 = 0; v1197_n0 < 1; ++v1197_n0) {
                #pragma unroll
                for (int32_t v1198_n1 = 0; v1198_n1 < 16; ++v1198_n1) {
                  int32_t v1199_a = v1197_n0 + v1198_n1;
                  float v1200_data = ir2[v1199_a];
                  r2[v1199_a] = v1200_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v1201_i0 = 0; v1201_i0 < 1; ++v1201_i0) {
                int32_t v1206_lead = v25_lead + (v1201_i0 * 32);
                #pragma unroll
                for (int32_t v1202_i1 = 0; v1202_i1 < 16; ++v1202_i1) {
                  float v1204_data = r2[(v1201_i0 + v1202_i1)];
                  glb_m0[(v1206_lead + (v1202_i1 * 32))] = v1204_data;
                }
              }
              float r4[8]{};
              // r4 = load{g>r}(glb_m4);
              if (v35_g) {
                #pragma unroll
                for (int32_t v1219_i1 = 0; v1219_i1 < 8; ++v1219_i1) {
                  float v1224_data = glb_m4[(v25_lead + (v1219_i1 * 12))];
                  r4[v1219_i1] = v1224_data;
                }
              }
              float r5[8]{};
              // r5 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v1227_i0 = 0; v1227_i0 < 1; ++v1227_i0) {
                int32_t v1230_lead = v25_lead + (v1227_i0 * 32);
                #pragma unroll
                for (int32_t v1228_i1 = 0; v1228_i1 < 8; ++v1228_i1) {
                  float v1233_data = glb_m0[(v1230_lead + (v1228_i1 * 32))];
                  r5[(v1227_i0 + v1228_i1)] = v1233_data;
                }
              }
              float r7[12]{};
              // r7 = load{g>r}(glb_m5);
              #pragma unroll
              for (int32_t v1828_i0 = 0; v1828_i0 < 1; ++v1828_i0) {
                int32_t v1831_lead = v25_lead + (v1828_i0 * 32);
                #pragma unroll
                for (int32_t v1829_i1 = 0; v1829_i1 < 12; ++v1829_i1) {
                  float v1834_data = glb_m5[(v1831_lead + (v1829_i1 * 32))];
                  r7[(v1828_i0 + v1829_i1)] = v1834_data;
                }
              }
              float r6[8]{};
              // ir6 = +(r3 * r4)
              // [(0, 32), (0, 8)] [(0, 12)]
              float ir6[8]{};
              float v1237_data = r3[0];
              float v1238_data = r4[0];
              float v1241_data = ir6[0];
              ir6[0] = (v1241_data + (v1237_data * (sycl::select_from_group(item.get_sub_group(), v1238_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1244_data = r4[1];
              float v1247_data = ir6[1];
              ir6[1] = (v1247_data + (v1237_data * (sycl::select_from_group(item.get_sub_group(), v1244_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1250_data = r4[2];
              float v1253_data = ir6[2];
              ir6[2] = (v1253_data + (v1237_data * (sycl::select_from_group(item.get_sub_group(), v1250_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1256_data = r4[3];
              float v1259_data = ir6[3];
              ir6[3] = (v1259_data + (v1237_data * (sycl::select_from_group(item.get_sub_group(), v1256_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1262_data = r4[4];
              float v1265_data = ir6[4];
              ir6[4] = (v1265_data + (v1237_data * (sycl::select_from_group(item.get_sub_group(), v1262_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1268_data = r4[5];
              float v1271_data = ir6[5];
              ir6[5] = (v1271_data + (v1237_data * (sycl::select_from_group(item.get_sub_group(), v1268_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1274_data = r4[6];
              float v1277_data = ir6[6];
              ir6[6] = (v1277_data + (v1237_data * (sycl::select_from_group(item.get_sub_group(), v1274_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1280_data = r4[7];
              float v1283_data = ir6[7];
              ir6[7] = (v1283_data + (v1237_data * (sycl::select_from_group(item.get_sub_group(), v1280_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1285_data = r3[1];
              float v1289_data = ir6[0];
              ir6[0] = (v1289_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v1238_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1295_data = ir6[1];
              ir6[1] = (v1295_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v1244_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1301_data = ir6[2];
              ir6[2] = (v1301_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v1250_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1307_data = ir6[3];
              ir6[3] = (v1307_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v1256_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1313_data = ir6[4];
              ir6[4] = (v1313_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v1262_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1319_data = ir6[5];
              ir6[5] = (v1319_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v1268_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1325_data = ir6[6];
              ir6[6] = (v1325_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v1274_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1331_data = ir6[7];
              ir6[7] = (v1331_data + (v1285_data * (sycl::select_from_group(item.get_sub_group(), v1280_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1333_data = r3[2];
              float v1337_data = ir6[0];
              ir6[0] = (v1337_data + (v1333_data * (sycl::select_from_group(item.get_sub_group(), v1238_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1343_data = ir6[1];
              ir6[1] = (v1343_data + (v1333_data * (sycl::select_from_group(item.get_sub_group(), v1244_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1349_data = ir6[2];
              ir6[2] = (v1349_data + (v1333_data * (sycl::select_from_group(item.get_sub_group(), v1250_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1355_data = ir6[3];
              ir6[3] = (v1355_data + (v1333_data * (sycl::select_from_group(item.get_sub_group(), v1256_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1361_data = ir6[4];
              ir6[4] = (v1361_data + (v1333_data * (sycl::select_from_group(item.get_sub_group(), v1262_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1367_data = ir6[5];
              ir6[5] = (v1367_data + (v1333_data * (sycl::select_from_group(item.get_sub_group(), v1268_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1373_data = ir6[6];
              ir6[6] = (v1373_data + (v1333_data * (sycl::select_from_group(item.get_sub_group(), v1274_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1379_data = ir6[7];
              ir6[7] = (v1379_data + (v1333_data * (sycl::select_from_group(item.get_sub_group(), v1280_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1381_data = r3[3];
              float v1385_data = ir6[0];
              ir6[0] = (v1385_data + (v1381_data * (sycl::select_from_group(item.get_sub_group(), v1238_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1391_data = ir6[1];
              ir6[1] = (v1391_data + (v1381_data * (sycl::select_from_group(item.get_sub_group(), v1244_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1397_data = ir6[2];
              ir6[2] = (v1397_data + (v1381_data * (sycl::select_from_group(item.get_sub_group(), v1250_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1403_data = ir6[3];
              ir6[3] = (v1403_data + (v1381_data * (sycl::select_from_group(item.get_sub_group(), v1256_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1409_data = ir6[4];
              ir6[4] = (v1409_data + (v1381_data * (sycl::select_from_group(item.get_sub_group(), v1262_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1415_data = ir6[5];
              ir6[5] = (v1415_data + (v1381_data * (sycl::select_from_group(item.get_sub_group(), v1268_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1421_data = ir6[6];
              ir6[6] = (v1421_data + (v1381_data * (sycl::select_from_group(item.get_sub_group(), v1274_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1427_data = ir6[7];
              ir6[7] = (v1427_data + (v1381_data * (sycl::select_from_group(item.get_sub_group(), v1280_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1429_data = r3[4];
              float v1433_data = ir6[0];
              ir6[0] = (v1433_data + (v1429_data * (sycl::select_from_group(item.get_sub_group(), v1238_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1439_data = ir6[1];
              ir6[1] = (v1439_data + (v1429_data * (sycl::select_from_group(item.get_sub_group(), v1244_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1445_data = ir6[2];
              ir6[2] = (v1445_data + (v1429_data * (sycl::select_from_group(item.get_sub_group(), v1250_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1451_data = ir6[3];
              ir6[3] = (v1451_data + (v1429_data * (sycl::select_from_group(item.get_sub_group(), v1256_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1457_data = ir6[4];
              ir6[4] = (v1457_data + (v1429_data * (sycl::select_from_group(item.get_sub_group(), v1262_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1463_data = ir6[5];
              ir6[5] = (v1463_data + (v1429_data * (sycl::select_from_group(item.get_sub_group(), v1268_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1469_data = ir6[6];
              ir6[6] = (v1469_data + (v1429_data * (sycl::select_from_group(item.get_sub_group(), v1274_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1475_data = ir6[7];
              ir6[7] = (v1475_data + (v1429_data * (sycl::select_from_group(item.get_sub_group(), v1280_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1477_data = r3[5];
              float v1481_data = ir6[0];
              ir6[0] = (v1481_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v1238_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1487_data = ir6[1];
              ir6[1] = (v1487_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v1244_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1493_data = ir6[2];
              ir6[2] = (v1493_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v1250_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1499_data = ir6[3];
              ir6[3] = (v1499_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v1256_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1505_data = ir6[4];
              ir6[4] = (v1505_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v1262_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1511_data = ir6[5];
              ir6[5] = (v1511_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v1268_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1517_data = ir6[6];
              ir6[6] = (v1517_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v1274_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1523_data = ir6[7];
              ir6[7] = (v1523_data + (v1477_data * (sycl::select_from_group(item.get_sub_group(), v1280_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1525_data = r3[6];
              float v1529_data = ir6[0];
              ir6[0] = (v1529_data + (v1525_data * (sycl::select_from_group(item.get_sub_group(), v1238_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1535_data = ir6[1];
              ir6[1] = (v1535_data + (v1525_data * (sycl::select_from_group(item.get_sub_group(), v1244_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1541_data = ir6[2];
              ir6[2] = (v1541_data + (v1525_data * (sycl::select_from_group(item.get_sub_group(), v1250_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1547_data = ir6[3];
              ir6[3] = (v1547_data + (v1525_data * (sycl::select_from_group(item.get_sub_group(), v1256_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1553_data = ir6[4];
              ir6[4] = (v1553_data + (v1525_data * (sycl::select_from_group(item.get_sub_group(), v1262_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1559_data = ir6[5];
              ir6[5] = (v1559_data + (v1525_data * (sycl::select_from_group(item.get_sub_group(), v1268_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1565_data = ir6[6];
              ir6[6] = (v1565_data + (v1525_data * (sycl::select_from_group(item.get_sub_group(), v1274_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1571_data = ir6[7];
              ir6[7] = (v1571_data + (v1525_data * (sycl::select_from_group(item.get_sub_group(), v1280_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1573_data = r3[7];
              float v1577_data = ir6[0];
              ir6[0] = (v1577_data + (v1573_data * (sycl::select_from_group(item.get_sub_group(), v1238_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1583_data = ir6[1];
              ir6[1] = (v1583_data + (v1573_data * (sycl::select_from_group(item.get_sub_group(), v1244_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1589_data = ir6[2];
              ir6[2] = (v1589_data + (v1573_data * (sycl::select_from_group(item.get_sub_group(), v1250_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1595_data = ir6[3];
              ir6[3] = (v1595_data + (v1573_data * (sycl::select_from_group(item.get_sub_group(), v1256_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1601_data = ir6[4];
              ir6[4] = (v1601_data + (v1573_data * (sycl::select_from_group(item.get_sub_group(), v1262_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1607_data = ir6[5];
              ir6[5] = (v1607_data + (v1573_data * (sycl::select_from_group(item.get_sub_group(), v1268_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1613_data = ir6[6];
              ir6[6] = (v1613_data + (v1573_data * (sycl::select_from_group(item.get_sub_group(), v1274_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1619_data = ir6[7];
              ir6[7] = (v1619_data + (v1573_data * (sycl::select_from_group(item.get_sub_group(), v1280_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1621_data = r3[8];
              float v1625_data = ir6[0];
              ir6[0] = (v1625_data + (v1621_data * (sycl::select_from_group(item.get_sub_group(), v1238_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1631_data = ir6[1];
              ir6[1] = (v1631_data + (v1621_data * (sycl::select_from_group(item.get_sub_group(), v1244_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1637_data = ir6[2];
              ir6[2] = (v1637_data + (v1621_data * (sycl::select_from_group(item.get_sub_group(), v1250_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1643_data = ir6[3];
              ir6[3] = (v1643_data + (v1621_data * (sycl::select_from_group(item.get_sub_group(), v1256_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1649_data = ir6[4];
              ir6[4] = (v1649_data + (v1621_data * (sycl::select_from_group(item.get_sub_group(), v1262_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1655_data = ir6[5];
              ir6[5] = (v1655_data + (v1621_data * (sycl::select_from_group(item.get_sub_group(), v1268_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1661_data = ir6[6];
              ir6[6] = (v1661_data + (v1621_data * (sycl::select_from_group(item.get_sub_group(), v1274_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1667_data = ir6[7];
              ir6[7] = (v1667_data + (v1621_data * (sycl::select_from_group(item.get_sub_group(), v1280_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1669_data = r3[9];
              float v1673_data = ir6[0];
              ir6[0] = (v1673_data + (v1669_data * (sycl::select_from_group(item.get_sub_group(), v1238_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1679_data = ir6[1];
              ir6[1] = (v1679_data + (v1669_data * (sycl::select_from_group(item.get_sub_group(), v1244_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1685_data = ir6[2];
              ir6[2] = (v1685_data + (v1669_data * (sycl::select_from_group(item.get_sub_group(), v1250_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1691_data = ir6[3];
              ir6[3] = (v1691_data + (v1669_data * (sycl::select_from_group(item.get_sub_group(), v1256_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1697_data = ir6[4];
              ir6[4] = (v1697_data + (v1669_data * (sycl::select_from_group(item.get_sub_group(), v1262_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1703_data = ir6[5];
              ir6[5] = (v1703_data + (v1669_data * (sycl::select_from_group(item.get_sub_group(), v1268_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1709_data = ir6[6];
              ir6[6] = (v1709_data + (v1669_data * (sycl::select_from_group(item.get_sub_group(), v1274_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1715_data = ir6[7];
              ir6[7] = (v1715_data + (v1669_data * (sycl::select_from_group(item.get_sub_group(), v1280_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1717_data = r3[10];
              float v1721_data = ir6[0];
              ir6[0] = (v1721_data + (v1717_data * (sycl::select_from_group(item.get_sub_group(), v1238_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1727_data = ir6[1];
              ir6[1] = (v1727_data + (v1717_data * (sycl::select_from_group(item.get_sub_group(), v1244_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1733_data = ir6[2];
              ir6[2] = (v1733_data + (v1717_data * (sycl::select_from_group(item.get_sub_group(), v1250_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1739_data = ir6[3];
              ir6[3] = (v1739_data + (v1717_data * (sycl::select_from_group(item.get_sub_group(), v1256_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1745_data = ir6[4];
              ir6[4] = (v1745_data + (v1717_data * (sycl::select_from_group(item.get_sub_group(), v1262_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1751_data = ir6[5];
              ir6[5] = (v1751_data + (v1717_data * (sycl::select_from_group(item.get_sub_group(), v1268_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1757_data = ir6[6];
              ir6[6] = (v1757_data + (v1717_data * (sycl::select_from_group(item.get_sub_group(), v1274_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1763_data = ir6[7];
              ir6[7] = (v1763_data + (v1717_data * (sycl::select_from_group(item.get_sub_group(), v1280_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1765_data = r3[11];
              float v1769_data = ir6[0];
              ir6[0] = (v1769_data + (v1765_data * (sycl::select_from_group(item.get_sub_group(), v1238_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1775_data = ir6[1];
              ir6[1] = (v1775_data + (v1765_data * (sycl::select_from_group(item.get_sub_group(), v1244_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1781_data = ir6[2];
              ir6[2] = (v1781_data + (v1765_data * (sycl::select_from_group(item.get_sub_group(), v1250_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1787_data = ir6[3];
              ir6[3] = (v1787_data + (v1765_data * (sycl::select_from_group(item.get_sub_group(), v1256_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1793_data = ir6[4];
              ir6[4] = (v1793_data + (v1765_data * (sycl::select_from_group(item.get_sub_group(), v1262_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1799_data = ir6[5];
              ir6[5] = (v1799_data + (v1765_data * (sycl::select_from_group(item.get_sub_group(), v1268_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1805_data = ir6[6];
              ir6[6] = (v1805_data + (v1765_data * (sycl::select_from_group(item.get_sub_group(), v1274_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1811_data = ir6[7];
              ir6[7] = (v1811_data + (v1765_data * (sycl::select_from_group(item.get_sub_group(), v1280_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              // r6 = ir6 + r5
              #pragma unroll
              for (int32_t v1813_n0 = 0; v1813_n0 < 1; ++v1813_n0) {
                #pragma unroll
                for (int32_t v1814_n1 = 0; v1814_n1 < 8; ++v1814_n1) {
                  int32_t v1815_a = v1813_n0 + v1814_n1;
                  float v1816_data = ir6[v1815_a];
                  float v1817_data = r5[v1815_a];
                  r6[v1815_a] = (v1817_data + v1816_data);
                }
              }
              // glb_m0 = store{r>g}(r6);
              #pragma unroll
              for (int32_t v1819_i0 = 0; v1819_i0 < 1; ++v1819_i0) {
                int32_t v1824_lead = v25_lead + (v1819_i0 * 32);
                #pragma unroll
                for (int32_t v1820_i1 = 0; v1820_i1 < 8; ++v1820_i1) {
                  float v1822_data = r6[(v1819_i0 + v1820_i1)];
                  glb_m0[(v1824_lead + (v1820_i1 * 32))] = v1822_data;
                }
              }
              float r8[8]{};
              // r8 = load{g>r}(glb_m6);
              if (v35_g) {
                #pragma unroll
                for (int32_t v1837_i1 = 0; v1837_i1 < 8; ++v1837_i1) {
                  float v1842_data = glb_m6[(v25_lead + (v1837_i1 * 12))];
                  r8[v1837_i1] = v1842_data;
                }
              }
              float r9[8]{};
              // r9 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v1845_i0 = 0; v1845_i0 < 1; ++v1845_i0) {
                int32_t v1848_lead = v25_lead + (v1845_i0 * 32);
                #pragma unroll
                for (int32_t v1846_i1 = 0; v1846_i1 < 8; ++v1846_i1) {
                  float v1852_data = glb_m0[(v1848_lead + ((v1846_i1 + 8) * 32))];
                  r9[(v1845_i0 + v1846_i1)] = v1852_data;
                }
              }
              float r10[8]{};
              // ir10 = +(r7 * r8)
              // [(0, 32), (0, 8)] [(0, 12)]
              float ir10[8]{};
              float v1856_data = r7[0];
              float v1857_data = r8[0];
              float v1860_data = ir10[0];
              ir10[0] = (v1860_data + (v1856_data * (sycl::select_from_group(item.get_sub_group(), v1857_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1863_data = r8[1];
              float v1866_data = ir10[1];
              ir10[1] = (v1866_data + (v1856_data * (sycl::select_from_group(item.get_sub_group(), v1863_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1869_data = r8[2];
              float v1872_data = ir10[2];
              ir10[2] = (v1872_data + (v1856_data * (sycl::select_from_group(item.get_sub_group(), v1869_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1875_data = r8[3];
              float v1878_data = ir10[3];
              ir10[3] = (v1878_data + (v1856_data * (sycl::select_from_group(item.get_sub_group(), v1875_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1881_data = r8[4];
              float v1884_data = ir10[4];
              ir10[4] = (v1884_data + (v1856_data * (sycl::select_from_group(item.get_sub_group(), v1881_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1887_data = r8[5];
              float v1890_data = ir10[5];
              ir10[5] = (v1890_data + (v1856_data * (sycl::select_from_group(item.get_sub_group(), v1887_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1893_data = r8[6];
              float v1896_data = ir10[6];
              ir10[6] = (v1896_data + (v1856_data * (sycl::select_from_group(item.get_sub_group(), v1893_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1899_data = r8[7];
              float v1902_data = ir10[7];
              ir10[7] = (v1902_data + (v1856_data * (sycl::select_from_group(item.get_sub_group(), v1899_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1904_data = r7[1];
              float v1908_data = ir10[0];
              ir10[0] = (v1908_data + (v1904_data * (sycl::select_from_group(item.get_sub_group(), v1857_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1914_data = ir10[1];
              ir10[1] = (v1914_data + (v1904_data * (sycl::select_from_group(item.get_sub_group(), v1863_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1920_data = ir10[2];
              ir10[2] = (v1920_data + (v1904_data * (sycl::select_from_group(item.get_sub_group(), v1869_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1926_data = ir10[3];
              ir10[3] = (v1926_data + (v1904_data * (sycl::select_from_group(item.get_sub_group(), v1875_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1932_data = ir10[4];
              ir10[4] = (v1932_data + (v1904_data * (sycl::select_from_group(item.get_sub_group(), v1881_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1938_data = ir10[5];
              ir10[5] = (v1938_data + (v1904_data * (sycl::select_from_group(item.get_sub_group(), v1887_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1944_data = ir10[6];
              ir10[6] = (v1944_data + (v1904_data * (sycl::select_from_group(item.get_sub_group(), v1893_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1950_data = ir10[7];
              ir10[7] = (v1950_data + (v1904_data * (sycl::select_from_group(item.get_sub_group(), v1899_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1952_data = r7[2];
              float v1956_data = ir10[0];
              ir10[0] = (v1956_data + (v1952_data * (sycl::select_from_group(item.get_sub_group(), v1857_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1962_data = ir10[1];
              ir10[1] = (v1962_data + (v1952_data * (sycl::select_from_group(item.get_sub_group(), v1863_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1968_data = ir10[2];
              ir10[2] = (v1968_data + (v1952_data * (sycl::select_from_group(item.get_sub_group(), v1869_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1974_data = ir10[3];
              ir10[3] = (v1974_data + (v1952_data * (sycl::select_from_group(item.get_sub_group(), v1875_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1980_data = ir10[4];
              ir10[4] = (v1980_data + (v1952_data * (sycl::select_from_group(item.get_sub_group(), v1881_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1986_data = ir10[5];
              ir10[5] = (v1986_data + (v1952_data * (sycl::select_from_group(item.get_sub_group(), v1887_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1992_data = ir10[6];
              ir10[6] = (v1992_data + (v1952_data * (sycl::select_from_group(item.get_sub_group(), v1893_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1998_data = ir10[7];
              ir10[7] = (v1998_data + (v1952_data * (sycl::select_from_group(item.get_sub_group(), v1899_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v2000_data = r7[3];
              float v2004_data = ir10[0];
              ir10[0] = (v2004_data + (v2000_data * (sycl::select_from_group(item.get_sub_group(), v1857_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v2010_data = ir10[1];
              ir10[1] = (v2010_data + (v2000_data * (sycl::select_from_group(item.get_sub_group(), v1863_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v2016_data = ir10[2];
              ir10[2] = (v2016_data + (v2000_data * (sycl::select_from_group(item.get_sub_group(), v1869_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v2022_data = ir10[3];
              ir10[3] = (v2022_data + (v2000_data * (sycl::select_from_group(item.get_sub_group(), v1875_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v2028_data = ir10[4];
              ir10[4] = (v2028_data + (v2000_data * (sycl::select_from_group(item.get_sub_group(), v1881_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v2034_data = ir10[5];
              ir10[5] = (v2034_data + (v2000_data * (sycl::select_from_group(item.get_sub_group(), v1887_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v2040_data = ir10[6];
              ir10[6] = (v2040_data + (v2000_data * (sycl::select_from_group(item.get_sub_group(), v1893_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v2046_data = ir10[7];
              ir10[7] = (v2046_data + (v2000_data * (sycl::select_from_group(item.get_sub_group(), v1899_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v2048_data = r7[4];
              float v2052_data = ir10[0];
              ir10[0] = (v2052_data + (v2048_data * (sycl::select_from_group(item.get_sub_group(), v1857_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v2058_data = ir10[1];
              ir10[1] = (v2058_data + (v2048_data * (sycl::select_from_group(item.get_sub_group(), v1863_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v2064_data = ir10[2];
              ir10[2] = (v2064_data + (v2048_data * (sycl::select_from_group(item.get_sub_group(), v1869_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v2070_data = ir10[3];
              ir10[3] = (v2070_data + (v2048_data * (sycl::select_from_group(item.get_sub_group(), v1875_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v2076_data = ir10[4];
              ir10[4] = (v2076_data + (v2048_data * (sycl::select_from_group(item.get_sub_group(), v1881_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v2082_data = ir10[5];
              ir10[5] = (v2082_data + (v2048_data * (sycl::select_from_group(item.get_sub_group(), v1887_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v2088_data = ir10[6];
              ir10[6] = (v2088_data + (v2048_data * (sycl::select_from_group(item.get_sub_group(), v1893_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v2094_data = ir10[7];
              ir10[7] = (v2094_data + (v2048_data * (sycl::select_from_group(item.get_sub_group(), v1899_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v2096_data = r7[5];
              float v2100_data = ir10[0];
              ir10[0] = (v2100_data + (v2096_data * (sycl::select_from_group(item.get_sub_group(), v1857_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v2106_data = ir10[1];
              ir10[1] = (v2106_data + (v2096_data * (sycl::select_from_group(item.get_sub_group(), v1863_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v2112_data = ir10[2];
              ir10[2] = (v2112_data + (v2096_data * (sycl::select_from_group(item.get_sub_group(), v1869_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v2118_data = ir10[3];
              ir10[3] = (v2118_data + (v2096_data * (sycl::select_from_group(item.get_sub_group(), v1875_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v2124_data = ir10[4];
              ir10[4] = (v2124_data + (v2096_data * (sycl::select_from_group(item.get_sub_group(), v1881_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v2130_data = ir10[5];
              ir10[5] = (v2130_data + (v2096_data * (sycl::select_from_group(item.get_sub_group(), v1887_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v2136_data = ir10[6];
              ir10[6] = (v2136_data + (v2096_data * (sycl::select_from_group(item.get_sub_group(), v1893_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v2142_data = ir10[7];
              ir10[7] = (v2142_data + (v2096_data * (sycl::select_from_group(item.get_sub_group(), v1899_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v2144_data = r7[6];
              float v2148_data = ir10[0];
              ir10[0] = (v2148_data + (v2144_data * (sycl::select_from_group(item.get_sub_group(), v1857_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v2154_data = ir10[1];
              ir10[1] = (v2154_data + (v2144_data * (sycl::select_from_group(item.get_sub_group(), v1863_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v2160_data = ir10[2];
              ir10[2] = (v2160_data + (v2144_data * (sycl::select_from_group(item.get_sub_group(), v1869_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v2166_data = ir10[3];
              ir10[3] = (v2166_data + (v2144_data * (sycl::select_from_group(item.get_sub_group(), v1875_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v2172_data = ir10[4];
              ir10[4] = (v2172_data + (v2144_data * (sycl::select_from_group(item.get_sub_group(), v1881_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v2178_data = ir10[5];
              ir10[5] = (v2178_data + (v2144_data * (sycl::select_from_group(item.get_sub_group(), v1887_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v2184_data = ir10[6];
              ir10[6] = (v2184_data + (v2144_data * (sycl::select_from_group(item.get_sub_group(), v1893_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v2190_data = ir10[7];
              ir10[7] = (v2190_data + (v2144_data * (sycl::select_from_group(item.get_sub_group(), v1899_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v2192_data = r7[7];
              float v2196_data = ir10[0];
              ir10[0] = (v2196_data + (v2192_data * (sycl::select_from_group(item.get_sub_group(), v1857_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v2202_data = ir10[1];
              ir10[1] = (v2202_data + (v2192_data * (sycl::select_from_group(item.get_sub_group(), v1863_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v2208_data = ir10[2];
              ir10[2] = (v2208_data + (v2192_data * (sycl::select_from_group(item.get_sub_group(), v1869_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v2214_data = ir10[3];
              ir10[3] = (v2214_data + (v2192_data * (sycl::select_from_group(item.get_sub_group(), v1875_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v2220_data = ir10[4];
              ir10[4] = (v2220_data + (v2192_data * (sycl::select_from_group(item.get_sub_group(), v1881_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v2226_data = ir10[5];
              ir10[5] = (v2226_data + (v2192_data * (sycl::select_from_group(item.get_sub_group(), v1887_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v2232_data = ir10[6];
              ir10[6] = (v2232_data + (v2192_data * (sycl::select_from_group(item.get_sub_group(), v1893_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v2238_data = ir10[7];
              ir10[7] = (v2238_data + (v2192_data * (sycl::select_from_group(item.get_sub_group(), v1899_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v2240_data = r7[8];
              float v2244_data = ir10[0];
              ir10[0] = (v2244_data + (v2240_data * (sycl::select_from_group(item.get_sub_group(), v1857_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v2250_data = ir10[1];
              ir10[1] = (v2250_data + (v2240_data * (sycl::select_from_group(item.get_sub_group(), v1863_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v2256_data = ir10[2];
              ir10[2] = (v2256_data + (v2240_data * (sycl::select_from_group(item.get_sub_group(), v1869_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v2262_data = ir10[3];
              ir10[3] = (v2262_data + (v2240_data * (sycl::select_from_group(item.get_sub_group(), v1875_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v2268_data = ir10[4];
              ir10[4] = (v2268_data + (v2240_data * (sycl::select_from_group(item.get_sub_group(), v1881_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v2274_data = ir10[5];
              ir10[5] = (v2274_data + (v2240_data * (sycl::select_from_group(item.get_sub_group(), v1887_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v2280_data = ir10[6];
              ir10[6] = (v2280_data + (v2240_data * (sycl::select_from_group(item.get_sub_group(), v1893_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v2286_data = ir10[7];
              ir10[7] = (v2286_data + (v2240_data * (sycl::select_from_group(item.get_sub_group(), v1899_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v2288_data = r7[9];
              float v2292_data = ir10[0];
              ir10[0] = (v2292_data + (v2288_data * (sycl::select_from_group(item.get_sub_group(), v1857_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v2298_data = ir10[1];
              ir10[1] = (v2298_data + (v2288_data * (sycl::select_from_group(item.get_sub_group(), v1863_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v2304_data = ir10[2];
              ir10[2] = (v2304_data + (v2288_data * (sycl::select_from_group(item.get_sub_group(), v1869_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v2310_data = ir10[3];
              ir10[3] = (v2310_data + (v2288_data * (sycl::select_from_group(item.get_sub_group(), v1875_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v2316_data = ir10[4];
              ir10[4] = (v2316_data + (v2288_data * (sycl::select_from_group(item.get_sub_group(), v1881_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v2322_data = ir10[5];
              ir10[5] = (v2322_data + (v2288_data * (sycl::select_from_group(item.get_sub_group(), v1887_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v2328_data = ir10[6];
              ir10[6] = (v2328_data + (v2288_data * (sycl::select_from_group(item.get_sub_group(), v1893_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v2334_data = ir10[7];
              ir10[7] = (v2334_data + (v2288_data * (sycl::select_from_group(item.get_sub_group(), v1899_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v2336_data = r7[10];
              float v2340_data = ir10[0];
              ir10[0] = (v2340_data + (v2336_data * (sycl::select_from_group(item.get_sub_group(), v1857_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v2346_data = ir10[1];
              ir10[1] = (v2346_data + (v2336_data * (sycl::select_from_group(item.get_sub_group(), v1863_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v2352_data = ir10[2];
              ir10[2] = (v2352_data + (v2336_data * (sycl::select_from_group(item.get_sub_group(), v1869_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v2358_data = ir10[3];
              ir10[3] = (v2358_data + (v2336_data * (sycl::select_from_group(item.get_sub_group(), v1875_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v2364_data = ir10[4];
              ir10[4] = (v2364_data + (v2336_data * (sycl::select_from_group(item.get_sub_group(), v1881_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v2370_data = ir10[5];
              ir10[5] = (v2370_data + (v2336_data * (sycl::select_from_group(item.get_sub_group(), v1887_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v2376_data = ir10[6];
              ir10[6] = (v2376_data + (v2336_data * (sycl::select_from_group(item.get_sub_group(), v1893_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v2382_data = ir10[7];
              ir10[7] = (v2382_data + (v2336_data * (sycl::select_from_group(item.get_sub_group(), v1899_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v2384_data = r7[11];
              float v2388_data = ir10[0];
              ir10[0] = (v2388_data + (v2384_data * (sycl::select_from_group(item.get_sub_group(), v1857_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v2394_data = ir10[1];
              ir10[1] = (v2394_data + (v2384_data * (sycl::select_from_group(item.get_sub_group(), v1863_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v2400_data = ir10[2];
              ir10[2] = (v2400_data + (v2384_data * (sycl::select_from_group(item.get_sub_group(), v1869_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v2406_data = ir10[3];
              ir10[3] = (v2406_data + (v2384_data * (sycl::select_from_group(item.get_sub_group(), v1875_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v2412_data = ir10[4];
              ir10[4] = (v2412_data + (v2384_data * (sycl::select_from_group(item.get_sub_group(), v1881_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v2418_data = ir10[5];
              ir10[5] = (v2418_data + (v2384_data * (sycl::select_from_group(item.get_sub_group(), v1887_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v2424_data = ir10[6];
              ir10[6] = (v2424_data + (v2384_data * (sycl::select_from_group(item.get_sub_group(), v1893_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v2430_data = ir10[7];
              ir10[7] = (v2430_data + (v2384_data * (sycl::select_from_group(item.get_sub_group(), v1899_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              // r10 = ir10 + r9
              #pragma unroll
              for (int32_t v2432_n0 = 0; v2432_n0 < 1; ++v2432_n0) {
                #pragma unroll
                for (int32_t v2433_n1 = 0; v2433_n1 < 8; ++v2433_n1) {
                  int32_t v2434_a = v2432_n0 + v2433_n1;
                  float v2435_data = ir10[v2434_a];
                  float v2436_data = r9[v2434_a];
                  r10[v2434_a] = (v2436_data + v2435_data);
                }
              }
              // glb_m0 = store{r>g}(r10);
              #pragma unroll
              for (int32_t v2438_i0 = 0; v2438_i0 < 1; ++v2438_i0) {
                int32_t v2443_lead = v25_lead + (v2438_i0 * 32);
                #pragma unroll
                for (int32_t v2439_i1 = 0; v2439_i1 < 8; ++v2439_i1) {
                  float v2441_data = r10[(v2438_i0 + v2439_i1)];
                  glb_m0[(v2443_lead + ((v2439_i1 + 8) * 32))] = v2441_data;
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

