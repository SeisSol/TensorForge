// === base name ===
kernel_2f143e10f10473c0

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_2f143e10f10473c0 = {{32, 1, 1}, 32, 32, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_2f143e10f10473c0(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_2f143e10f10473c0(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_2f143e10f10473c0(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_2f143e10f10473c0(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_2f143e10f10473c0(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_2f143e10f10473c0(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, m6, m6_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_2f143e10f10473c0(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0) {
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
              // wait(r0 = load{g>r}(glb_m1););
              float r3[12]{};
              // r3 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v44_i0 = 0; v44_i0 < 1; ++v44_i0) {
                int32_t v47_lead = v25_lead + (v44_i0 * 32);
                #pragma unroll
                for (int32_t v45_i1 = 0; v45_i1 < 12; ++v45_i1) {
                  float v50_data = glb_m3[(v47_lead + (v45_i1 * 32))];
                  r3[(v44_i0 + v45_i1)] = v50_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m2););
              float r2[16]{};
              // ir2 = +(r0 * r1)
              // [(0, 32), (0, 16)] [(0, 12)]
              float ir2[16]{};
              float v54_data = r0[0];
              float v55_data = r1[0];
              float v58_data = ir2[0];
              ir2[0] = (v58_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v61_data = r1[1];
              float v64_data = ir2[1];
              ir2[1] = (v64_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v67_data = r1[2];
              float v70_data = ir2[2];
              ir2[2] = (v70_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v73_data = r1[3];
              float v76_data = ir2[3];
              ir2[3] = (v76_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v79_data = r1[4];
              float v82_data = ir2[4];
              ir2[4] = (v82_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v85_data = r1[5];
              float v88_data = ir2[5];
              ir2[5] = (v88_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v91_data = r1[6];
              float v94_data = ir2[6];
              ir2[6] = (v94_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v97_data = r1[7];
              float v100_data = ir2[7];
              ir2[7] = (v100_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v103_data = r1[8];
              float v106_data = ir2[8];
              ir2[8] = (v106_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v109_data = r1[9];
              float v112_data = ir2[9];
              ir2[9] = (v112_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v115_data = r1[10];
              float v118_data = ir2[10];
              ir2[10] = (v118_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v121_data = r1[11];
              float v124_data = ir2[11];
              ir2[11] = (v124_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v127_data = r1[12];
              float v130_data = ir2[12];
              ir2[12] = (v130_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v133_data = r1[13];
              float v136_data = ir2[13];
              ir2[13] = (v136_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v139_data = r1[14];
              float v142_data = ir2[14];
              ir2[14] = (v142_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v145_data = r1[15];
              float v148_data = ir2[15];
              ir2[15] = (v148_data + (v54_data * (sycl::select_from_group(item.get_sub_group(), v145_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v150_data = r0[1];
              float v154_data = ir2[0];
              ir2[0] = (v154_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v160_data = ir2[1];
              ir2[1] = (v160_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v166_data = ir2[2];
              ir2[2] = (v166_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v172_data = ir2[3];
              ir2[3] = (v172_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v178_data = ir2[4];
              ir2[4] = (v178_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v184_data = ir2[5];
              ir2[5] = (v184_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v190_data = ir2[6];
              ir2[6] = (v190_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v196_data = ir2[7];
              ir2[7] = (v196_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v202_data = ir2[8];
              ir2[8] = (v202_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v208_data = ir2[9];
              ir2[9] = (v208_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v214_data = ir2[10];
              ir2[10] = (v214_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v220_data = ir2[11];
              ir2[11] = (v220_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v226_data = ir2[12];
              ir2[12] = (v226_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v232_data = ir2[13];
              ir2[13] = (v232_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v238_data = ir2[14];
              ir2[14] = (v238_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v244_data = ir2[15];
              ir2[15] = (v244_data + (v150_data * (sycl::select_from_group(item.get_sub_group(), v145_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v246_data = r0[2];
              float v250_data = ir2[0];
              ir2[0] = (v250_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v256_data = ir2[1];
              ir2[1] = (v256_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v262_data = ir2[2];
              ir2[2] = (v262_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v268_data = ir2[3];
              ir2[3] = (v268_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v274_data = ir2[4];
              ir2[4] = (v274_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v280_data = ir2[5];
              ir2[5] = (v280_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v286_data = ir2[6];
              ir2[6] = (v286_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v292_data = ir2[7];
              ir2[7] = (v292_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v298_data = ir2[8];
              ir2[8] = (v298_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v304_data = ir2[9];
              ir2[9] = (v304_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v310_data = ir2[10];
              ir2[10] = (v310_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v316_data = ir2[11];
              ir2[11] = (v316_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v322_data = ir2[12];
              ir2[12] = (v322_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v328_data = ir2[13];
              ir2[13] = (v328_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v334_data = ir2[14];
              ir2[14] = (v334_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v340_data = ir2[15];
              ir2[15] = (v340_data + (v246_data * (sycl::select_from_group(item.get_sub_group(), v145_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v342_data = r0[3];
              float v346_data = ir2[0];
              ir2[0] = (v346_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v352_data = ir2[1];
              ir2[1] = (v352_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v358_data = ir2[2];
              ir2[2] = (v358_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v364_data = ir2[3];
              ir2[3] = (v364_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v370_data = ir2[4];
              ir2[4] = (v370_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v376_data = ir2[5];
              ir2[5] = (v376_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v382_data = ir2[6];
              ir2[6] = (v382_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v388_data = ir2[7];
              ir2[7] = (v388_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v394_data = ir2[8];
              ir2[8] = (v394_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v400_data = ir2[9];
              ir2[9] = (v400_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v406_data = ir2[10];
              ir2[10] = (v406_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v412_data = ir2[11];
              ir2[11] = (v412_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v418_data = ir2[12];
              ir2[12] = (v418_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v424_data = ir2[13];
              ir2[13] = (v424_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v430_data = ir2[14];
              ir2[14] = (v430_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v436_data = ir2[15];
              ir2[15] = (v436_data + (v342_data * (sycl::select_from_group(item.get_sub_group(), v145_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v438_data = r0[4];
              float v442_data = ir2[0];
              ir2[0] = (v442_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v448_data = ir2[1];
              ir2[1] = (v448_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v454_data = ir2[2];
              ir2[2] = (v454_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v460_data = ir2[3];
              ir2[3] = (v460_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v466_data = ir2[4];
              ir2[4] = (v466_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v472_data = ir2[5];
              ir2[5] = (v472_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v478_data = ir2[6];
              ir2[6] = (v478_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v484_data = ir2[7];
              ir2[7] = (v484_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v490_data = ir2[8];
              ir2[8] = (v490_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v496_data = ir2[9];
              ir2[9] = (v496_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v502_data = ir2[10];
              ir2[10] = (v502_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v508_data = ir2[11];
              ir2[11] = (v508_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v514_data = ir2[12];
              ir2[12] = (v514_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v520_data = ir2[13];
              ir2[13] = (v520_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v526_data = ir2[14];
              ir2[14] = (v526_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v532_data = ir2[15];
              ir2[15] = (v532_data + (v438_data * (sycl::select_from_group(item.get_sub_group(), v145_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v534_data = r0[5];
              float v538_data = ir2[0];
              ir2[0] = (v538_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v544_data = ir2[1];
              ir2[1] = (v544_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v550_data = ir2[2];
              ir2[2] = (v550_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v556_data = ir2[3];
              ir2[3] = (v556_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v562_data = ir2[4];
              ir2[4] = (v562_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v568_data = ir2[5];
              ir2[5] = (v568_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v574_data = ir2[6];
              ir2[6] = (v574_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v580_data = ir2[7];
              ir2[7] = (v580_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v586_data = ir2[8];
              ir2[8] = (v586_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v592_data = ir2[9];
              ir2[9] = (v592_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v598_data = ir2[10];
              ir2[10] = (v598_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v604_data = ir2[11];
              ir2[11] = (v604_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v610_data = ir2[12];
              ir2[12] = (v610_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v616_data = ir2[13];
              ir2[13] = (v616_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v622_data = ir2[14];
              ir2[14] = (v622_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v628_data = ir2[15];
              ir2[15] = (v628_data + (v534_data * (sycl::select_from_group(item.get_sub_group(), v145_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v630_data = r0[6];
              float v634_data = ir2[0];
              ir2[0] = (v634_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v640_data = ir2[1];
              ir2[1] = (v640_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v646_data = ir2[2];
              ir2[2] = (v646_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v652_data = ir2[3];
              ir2[3] = (v652_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v658_data = ir2[4];
              ir2[4] = (v658_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v664_data = ir2[5];
              ir2[5] = (v664_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v670_data = ir2[6];
              ir2[6] = (v670_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v676_data = ir2[7];
              ir2[7] = (v676_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v682_data = ir2[8];
              ir2[8] = (v682_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v688_data = ir2[9];
              ir2[9] = (v688_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v694_data = ir2[10];
              ir2[10] = (v694_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v700_data = ir2[11];
              ir2[11] = (v700_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v706_data = ir2[12];
              ir2[12] = (v706_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v712_data = ir2[13];
              ir2[13] = (v712_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v718_data = ir2[14];
              ir2[14] = (v718_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v724_data = ir2[15];
              ir2[15] = (v724_data + (v630_data * (sycl::select_from_group(item.get_sub_group(), v145_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v726_data = r0[7];
              float v730_data = ir2[0];
              ir2[0] = (v730_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v736_data = ir2[1];
              ir2[1] = (v736_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v742_data = ir2[2];
              ir2[2] = (v742_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v748_data = ir2[3];
              ir2[3] = (v748_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v754_data = ir2[4];
              ir2[4] = (v754_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v760_data = ir2[5];
              ir2[5] = (v760_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v766_data = ir2[6];
              ir2[6] = (v766_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v772_data = ir2[7];
              ir2[7] = (v772_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v778_data = ir2[8];
              ir2[8] = (v778_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v784_data = ir2[9];
              ir2[9] = (v784_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v790_data = ir2[10];
              ir2[10] = (v790_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v796_data = ir2[11];
              ir2[11] = (v796_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v802_data = ir2[12];
              ir2[12] = (v802_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v808_data = ir2[13];
              ir2[13] = (v808_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v814_data = ir2[14];
              ir2[14] = (v814_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v820_data = ir2[15];
              ir2[15] = (v820_data + (v726_data * (sycl::select_from_group(item.get_sub_group(), v145_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v822_data = r0[8];
              float v826_data = ir2[0];
              ir2[0] = (v826_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v832_data = ir2[1];
              ir2[1] = (v832_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v838_data = ir2[2];
              ir2[2] = (v838_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v844_data = ir2[3];
              ir2[3] = (v844_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v850_data = ir2[4];
              ir2[4] = (v850_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v856_data = ir2[5];
              ir2[5] = (v856_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v862_data = ir2[6];
              ir2[6] = (v862_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v868_data = ir2[7];
              ir2[7] = (v868_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v874_data = ir2[8];
              ir2[8] = (v874_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v880_data = ir2[9];
              ir2[9] = (v880_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v886_data = ir2[10];
              ir2[10] = (v886_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v892_data = ir2[11];
              ir2[11] = (v892_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v898_data = ir2[12];
              ir2[12] = (v898_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v904_data = ir2[13];
              ir2[13] = (v904_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v910_data = ir2[14];
              ir2[14] = (v910_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v916_data = ir2[15];
              ir2[15] = (v916_data + (v822_data * (sycl::select_from_group(item.get_sub_group(), v145_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v918_data = r0[9];
              float v922_data = ir2[0];
              ir2[0] = (v922_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v928_data = ir2[1];
              ir2[1] = (v928_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v934_data = ir2[2];
              ir2[2] = (v934_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v940_data = ir2[3];
              ir2[3] = (v940_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v946_data = ir2[4];
              ir2[4] = (v946_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v952_data = ir2[5];
              ir2[5] = (v952_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v958_data = ir2[6];
              ir2[6] = (v958_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v964_data = ir2[7];
              ir2[7] = (v964_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v970_data = ir2[8];
              ir2[8] = (v970_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v976_data = ir2[9];
              ir2[9] = (v976_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v982_data = ir2[10];
              ir2[10] = (v982_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v988_data = ir2[11];
              ir2[11] = (v988_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v994_data = ir2[12];
              ir2[12] = (v994_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1000_data = ir2[13];
              ir2[13] = (v1000_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1006_data = ir2[14];
              ir2[14] = (v1006_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1012_data = ir2[15];
              ir2[15] = (v1012_data + (v918_data * (sycl::select_from_group(item.get_sub_group(), v145_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1014_data = r0[10];
              float v1018_data = ir2[0];
              ir2[0] = (v1018_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1024_data = ir2[1];
              ir2[1] = (v1024_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1030_data = ir2[2];
              ir2[2] = (v1030_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1036_data = ir2[3];
              ir2[3] = (v1036_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1042_data = ir2[4];
              ir2[4] = (v1042_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1048_data = ir2[5];
              ir2[5] = (v1048_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1054_data = ir2[6];
              ir2[6] = (v1054_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1060_data = ir2[7];
              ir2[7] = (v1060_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1066_data = ir2[8];
              ir2[8] = (v1066_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1072_data = ir2[9];
              ir2[9] = (v1072_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1078_data = ir2[10];
              ir2[10] = (v1078_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1084_data = ir2[11];
              ir2[11] = (v1084_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1090_data = ir2[12];
              ir2[12] = (v1090_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1096_data = ir2[13];
              ir2[13] = (v1096_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1102_data = ir2[14];
              ir2[14] = (v1102_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1108_data = ir2[15];
              ir2[15] = (v1108_data + (v1014_data * (sycl::select_from_group(item.get_sub_group(), v145_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1110_data = r0[11];
              float v1114_data = ir2[0];
              ir2[0] = (v1114_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1120_data = ir2[1];
              ir2[1] = (v1120_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1126_data = ir2[2];
              ir2[2] = (v1126_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1132_data = ir2[3];
              ir2[3] = (v1132_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1138_data = ir2[4];
              ir2[4] = (v1138_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1144_data = ir2[5];
              ir2[5] = (v1144_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1150_data = ir2[6];
              ir2[6] = (v1150_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1156_data = ir2[7];
              ir2[7] = (v1156_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1162_data = ir2[8];
              ir2[8] = (v1162_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1168_data = ir2[9];
              ir2[9] = (v1168_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1174_data = ir2[10];
              ir2[10] = (v1174_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1180_data = ir2[11];
              ir2[11] = (v1180_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1186_data = ir2[12];
              ir2[12] = (v1186_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1192_data = ir2[13];
              ir2[13] = (v1192_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1198_data = ir2[14];
              ir2[14] = (v1198_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1204_data = ir2[15];
              ir2[15] = (v1204_data + (v1110_data * (sycl::select_from_group(item.get_sub_group(), v145_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              // r2 = ir2
              #pragma unroll
              for (int32_t v1206_n0 = 0; v1206_n0 < 1; ++v1206_n0) {
                #pragma unroll
                for (int32_t v1207_n1 = 0; v1207_n1 < 16; ++v1207_n1) {
                  int32_t v1208_a = v1206_n0 + v1207_n1;
                  float v1209_data = ir2[v1208_a];
                  r2[v1208_a] = v1209_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v1210_i0 = 0; v1210_i0 < 1; ++v1210_i0) {
                int32_t v1215_lead = v25_lead + (v1210_i0 * 32);
                #pragma unroll
                for (int32_t v1211_i1 = 0; v1211_i1 < 16; ++v1211_i1) {
                  float v1213_data = r2[(v1210_i0 + v1211_i1)];
                  glb_m0[(v1215_lead + (v1211_i1 * 32))] = v1213_data;
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
              // wait(r3 = load{g>r}(glb_m3););
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
              // wait(r4 = load{g>r}(glb_m4););
              float r7[12]{};
              // r7 = load{g>r}(glb_m5);
              #pragma unroll
              for (int32_t v1236_i0 = 0; v1236_i0 < 1; ++v1236_i0) {
                int32_t v1239_lead = v25_lead + (v1236_i0 * 32);
                #pragma unroll
                for (int32_t v1237_i1 = 0; v1237_i1 < 12; ++v1237_i1) {
                  float v1242_data = glb_m5[(v1239_lead + (v1237_i1 * 32))];
                  r7[(v1236_i0 + v1237_i1)] = v1242_data;
                }
              }
              // wait(r5 = load{g>r}(glb_m0););
              float r6[8]{};
              // ir6 = +(r3 * r4)
              // [(0, 32), (0, 8)] [(0, 12)]
              float ir6[8]{};
              float v1246_data = r3[0];
              float v1247_data = r4[0];
              float v1250_data = ir6[0];
              ir6[0] = (v1250_data + (v1246_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1253_data = r4[1];
              float v1256_data = ir6[1];
              ir6[1] = (v1256_data + (v1246_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1259_data = r4[2];
              float v1262_data = ir6[2];
              ir6[2] = (v1262_data + (v1246_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1265_data = r4[3];
              float v1268_data = ir6[3];
              ir6[3] = (v1268_data + (v1246_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1271_data = r4[4];
              float v1274_data = ir6[4];
              ir6[4] = (v1274_data + (v1246_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1277_data = r4[5];
              float v1280_data = ir6[5];
              ir6[5] = (v1280_data + (v1246_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1283_data = r4[6];
              float v1286_data = ir6[6];
              ir6[6] = (v1286_data + (v1246_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1289_data = r4[7];
              float v1292_data = ir6[7];
              ir6[7] = (v1292_data + (v1246_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1294_data = r3[1];
              float v1298_data = ir6[0];
              ir6[0] = (v1298_data + (v1294_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1304_data = ir6[1];
              ir6[1] = (v1304_data + (v1294_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1310_data = ir6[2];
              ir6[2] = (v1310_data + (v1294_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1316_data = ir6[3];
              ir6[3] = (v1316_data + (v1294_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1322_data = ir6[4];
              ir6[4] = (v1322_data + (v1294_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1328_data = ir6[5];
              ir6[5] = (v1328_data + (v1294_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1334_data = ir6[6];
              ir6[6] = (v1334_data + (v1294_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1340_data = ir6[7];
              ir6[7] = (v1340_data + (v1294_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1342_data = r3[2];
              float v1346_data = ir6[0];
              ir6[0] = (v1346_data + (v1342_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1352_data = ir6[1];
              ir6[1] = (v1352_data + (v1342_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1358_data = ir6[2];
              ir6[2] = (v1358_data + (v1342_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1364_data = ir6[3];
              ir6[3] = (v1364_data + (v1342_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1370_data = ir6[4];
              ir6[4] = (v1370_data + (v1342_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1376_data = ir6[5];
              ir6[5] = (v1376_data + (v1342_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1382_data = ir6[6];
              ir6[6] = (v1382_data + (v1342_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1388_data = ir6[7];
              ir6[7] = (v1388_data + (v1342_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1390_data = r3[3];
              float v1394_data = ir6[0];
              ir6[0] = (v1394_data + (v1390_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1400_data = ir6[1];
              ir6[1] = (v1400_data + (v1390_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1406_data = ir6[2];
              ir6[2] = (v1406_data + (v1390_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1412_data = ir6[3];
              ir6[3] = (v1412_data + (v1390_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1418_data = ir6[4];
              ir6[4] = (v1418_data + (v1390_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1424_data = ir6[5];
              ir6[5] = (v1424_data + (v1390_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1430_data = ir6[6];
              ir6[6] = (v1430_data + (v1390_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1436_data = ir6[7];
              ir6[7] = (v1436_data + (v1390_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1438_data = r3[4];
              float v1442_data = ir6[0];
              ir6[0] = (v1442_data + (v1438_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1448_data = ir6[1];
              ir6[1] = (v1448_data + (v1438_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1454_data = ir6[2];
              ir6[2] = (v1454_data + (v1438_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1460_data = ir6[3];
              ir6[3] = (v1460_data + (v1438_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1466_data = ir6[4];
              ir6[4] = (v1466_data + (v1438_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1472_data = ir6[5];
              ir6[5] = (v1472_data + (v1438_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1478_data = ir6[6];
              ir6[6] = (v1478_data + (v1438_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1484_data = ir6[7];
              ir6[7] = (v1484_data + (v1438_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1486_data = r3[5];
              float v1490_data = ir6[0];
              ir6[0] = (v1490_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1496_data = ir6[1];
              ir6[1] = (v1496_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1502_data = ir6[2];
              ir6[2] = (v1502_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1508_data = ir6[3];
              ir6[3] = (v1508_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1514_data = ir6[4];
              ir6[4] = (v1514_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1520_data = ir6[5];
              ir6[5] = (v1520_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1526_data = ir6[6];
              ir6[6] = (v1526_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1532_data = ir6[7];
              ir6[7] = (v1532_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1534_data = r3[6];
              float v1538_data = ir6[0];
              ir6[0] = (v1538_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1544_data = ir6[1];
              ir6[1] = (v1544_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1550_data = ir6[2];
              ir6[2] = (v1550_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1556_data = ir6[3];
              ir6[3] = (v1556_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1562_data = ir6[4];
              ir6[4] = (v1562_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1568_data = ir6[5];
              ir6[5] = (v1568_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1574_data = ir6[6];
              ir6[6] = (v1574_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1580_data = ir6[7];
              ir6[7] = (v1580_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1582_data = r3[7];
              float v1586_data = ir6[0];
              ir6[0] = (v1586_data + (v1582_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1592_data = ir6[1];
              ir6[1] = (v1592_data + (v1582_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1598_data = ir6[2];
              ir6[2] = (v1598_data + (v1582_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1604_data = ir6[3];
              ir6[3] = (v1604_data + (v1582_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1610_data = ir6[4];
              ir6[4] = (v1610_data + (v1582_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1616_data = ir6[5];
              ir6[5] = (v1616_data + (v1582_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1622_data = ir6[6];
              ir6[6] = (v1622_data + (v1582_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1628_data = ir6[7];
              ir6[7] = (v1628_data + (v1582_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1630_data = r3[8];
              float v1634_data = ir6[0];
              ir6[0] = (v1634_data + (v1630_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1640_data = ir6[1];
              ir6[1] = (v1640_data + (v1630_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1646_data = ir6[2];
              ir6[2] = (v1646_data + (v1630_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1652_data = ir6[3];
              ir6[3] = (v1652_data + (v1630_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1658_data = ir6[4];
              ir6[4] = (v1658_data + (v1630_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1664_data = ir6[5];
              ir6[5] = (v1664_data + (v1630_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1670_data = ir6[6];
              ir6[6] = (v1670_data + (v1630_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1676_data = ir6[7];
              ir6[7] = (v1676_data + (v1630_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1678_data = r3[9];
              float v1682_data = ir6[0];
              ir6[0] = (v1682_data + (v1678_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1688_data = ir6[1];
              ir6[1] = (v1688_data + (v1678_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1694_data = ir6[2];
              ir6[2] = (v1694_data + (v1678_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1700_data = ir6[3];
              ir6[3] = (v1700_data + (v1678_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1706_data = ir6[4];
              ir6[4] = (v1706_data + (v1678_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1712_data = ir6[5];
              ir6[5] = (v1712_data + (v1678_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1718_data = ir6[6];
              ir6[6] = (v1718_data + (v1678_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1724_data = ir6[7];
              ir6[7] = (v1724_data + (v1678_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1726_data = r3[10];
              float v1730_data = ir6[0];
              ir6[0] = (v1730_data + (v1726_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1736_data = ir6[1];
              ir6[1] = (v1736_data + (v1726_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1742_data = ir6[2];
              ir6[2] = (v1742_data + (v1726_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1748_data = ir6[3];
              ir6[3] = (v1748_data + (v1726_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1754_data = ir6[4];
              ir6[4] = (v1754_data + (v1726_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1760_data = ir6[5];
              ir6[5] = (v1760_data + (v1726_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1766_data = ir6[6];
              ir6[6] = (v1766_data + (v1726_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1772_data = ir6[7];
              ir6[7] = (v1772_data + (v1726_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1774_data = r3[11];
              float v1778_data = ir6[0];
              ir6[0] = (v1778_data + (v1774_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1784_data = ir6[1];
              ir6[1] = (v1784_data + (v1774_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1790_data = ir6[2];
              ir6[2] = (v1790_data + (v1774_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1796_data = ir6[3];
              ir6[3] = (v1796_data + (v1774_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1802_data = ir6[4];
              ir6[4] = (v1802_data + (v1774_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1808_data = ir6[5];
              ir6[5] = (v1808_data + (v1774_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1814_data = ir6[6];
              ir6[6] = (v1814_data + (v1774_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1820_data = ir6[7];
              ir6[7] = (v1820_data + (v1774_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              // r6 = ir6 + r5
              #pragma unroll
              for (int32_t v1822_n0 = 0; v1822_n0 < 1; ++v1822_n0) {
                #pragma unroll
                for (int32_t v1823_n1 = 0; v1823_n1 < 8; ++v1823_n1) {
                  int32_t v1824_a = v1822_n0 + v1823_n1;
                  float v1825_data = ir6[v1824_a];
                  float v1826_data = r5[v1824_a];
                  r6[v1824_a] = (v1826_data + v1825_data);
                }
              }
              // glb_m0 = store{r>g}(r6);
              #pragma unroll
              for (int32_t v1828_i0 = 0; v1828_i0 < 1; ++v1828_i0) {
                int32_t v1833_lead = v25_lead + (v1828_i0 * 32);
                #pragma unroll
                for (int32_t v1829_i1 = 0; v1829_i1 < 8; ++v1829_i1) {
                  float v1831_data = r6[(v1828_i0 + v1829_i1)];
                  glb_m0[(v1833_lead + (v1829_i1 * 32))] = v1831_data;
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
              // wait(r7 = load{g>r}(glb_m5););
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
              // wait(r8 = load{g>r}(glb_m6););
              // wait(r9 = load{g>r}(glb_m0););
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

