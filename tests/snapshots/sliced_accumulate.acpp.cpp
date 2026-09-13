// === base name ===
kernel_844a8b65e9e596a8

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_844a8b65e9e596a8 = {{32, 1, 1}, 32, 32, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_844a8b65e9e596a8(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_844a8b65e9e596a8(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_844a8b65e9e596a8(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (32, 1, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 32;
  config.block[1] = 1;
  config.block[2] = 1;
  config.sharedMemBytes = 0 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_844a8b65e9e596a8(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_844a8b65e9e596a8(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_844a8b65e9e596a8(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, m6, m6_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_844a8b65e9e596a8(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
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
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[32,16]],"name":"m0","ordered":false,"parts":1,"shape":[32,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[32,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,12],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[12,16]],"name":"m2","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"A0","bbox":[[0,0],[32,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,12],"variant":false},{"addressing":"strided","alias":"B0","bbox":[[0,0],[12,8]],"name":"m4","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A1","bbox":[[0,0],[32,12]],"name":"m5","ordered":false,"parts":1,"shape":[32,12],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[12,8]],"name":"m6","ordered":false,"parts":1,"shape":[12,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,12]},{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[12,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,8]],"is_tmp":false,"name":"m0","offset":[0,8],"shape":[32,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[32,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          for (size_t v1_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v1_batchId0 < numElements0; v1_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v2_ahead1 = v1_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v1_batchId0 * 512 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v1_batchId0 * 384 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v1_batchId0 * 192 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v1_batchId0 * 384 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v1_batchId0 * 96 + 0 + m4_extraOffset];
              const float *const __restrict__ glb_m5 = &m5[v1_batchId0 * 384 + 0 + m5_extraOffset];
              const float *const __restrict__ glb_m6 = &m6[v1_batchId0 * 96 + 0 + m6_extraOffset];
              float r0[12]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v19_lead = item.get_local_id(2) % 32;
              #pragma unroll
              for (int32_t v20_i0 = 0; v20_i0 < 1; ++v20_i0) {
                int32_t v23_lead = v19_lead + (v20_i0 * 32);
                #pragma unroll
                for (int32_t v21_i1 = 0; v21_i1 < 12; ++v21_i1) {
                  float v26_data = glb_m1[(v23_lead + (v21_i1 * 32))];
                  r0[(v20_i0 + v21_i1)] = v26_data;
                }
              }
              float r1[16]{};
              // r1 = load{g>r}(glb_m2);
              bool v29_g = v19_lead < 12;
              if (v29_g) {
                #pragma unroll
                for (int32_t v30_i1 = 0; v30_i1 < 16; ++v30_i1) {
                  float v35_data = glb_m2[(v19_lead + (v30_i1 * 12))];
                  r1[v30_i1] = v35_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              float r3[12]{};
              // r3 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v38_i0 = 0; v38_i0 < 1; ++v38_i0) {
                int32_t v41_lead = v19_lead + (v38_i0 * 32);
                #pragma unroll
                for (int32_t v39_i1 = 0; v39_i1 < 12; ++v39_i1) {
                  float v44_data = glb_m3[(v41_lead + (v39_i1 * 32))];
                  r3[(v38_i0 + v39_i1)] = v44_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m2););
              float r2[16]{};
              // r2 = +(r0 * r1) + None
              // [(0, 32), (0, 16)] [(0, 12)]
              float ir2[16]{};
              float v48_data = r0[0];
              float v49_data = r1[0];
              float v52_data = ir2[0];
              ir2[0] = (v52_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v55_data = r1[1];
              float v58_data = ir2[1];
              ir2[1] = (v58_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v61_data = r1[2];
              float v64_data = ir2[2];
              ir2[2] = (v64_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v67_data = r1[3];
              float v70_data = ir2[3];
              ir2[3] = (v70_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v73_data = r1[4];
              float v76_data = ir2[4];
              ir2[4] = (v76_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v79_data = r1[5];
              float v82_data = ir2[5];
              ir2[5] = (v82_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v85_data = r1[6];
              float v88_data = ir2[6];
              ir2[6] = (v88_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v91_data = r1[7];
              float v94_data = ir2[7];
              ir2[7] = (v94_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v97_data = r1[8];
              float v100_data = ir2[8];
              ir2[8] = (v100_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v103_data = r1[9];
              float v106_data = ir2[9];
              ir2[9] = (v106_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v109_data = r1[10];
              float v112_data = ir2[10];
              ir2[10] = (v112_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v115_data = r1[11];
              float v118_data = ir2[11];
              ir2[11] = (v118_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v121_data = r1[12];
              float v124_data = ir2[12];
              ir2[12] = (v124_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v127_data = r1[13];
              float v130_data = ir2[13];
              ir2[13] = (v130_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v133_data = r1[14];
              float v136_data = ir2[14];
              ir2[14] = (v136_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v139_data = r1[15];
              float v142_data = ir2[15];
              ir2[15] = (v142_data + (v48_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v144_data = r0[1];
              float v148_data = ir2[0];
              ir2[0] = (v148_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v154_data = ir2[1];
              ir2[1] = (v154_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v160_data = ir2[2];
              ir2[2] = (v160_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v166_data = ir2[3];
              ir2[3] = (v166_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v172_data = ir2[4];
              ir2[4] = (v172_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v178_data = ir2[5];
              ir2[5] = (v178_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v184_data = ir2[6];
              ir2[6] = (v184_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v190_data = ir2[7];
              ir2[7] = (v190_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v196_data = ir2[8];
              ir2[8] = (v196_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v202_data = ir2[9];
              ir2[9] = (v202_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v208_data = ir2[10];
              ir2[10] = (v208_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v214_data = ir2[11];
              ir2[11] = (v214_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v220_data = ir2[12];
              ir2[12] = (v220_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v226_data = ir2[13];
              ir2[13] = (v226_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v232_data = ir2[14];
              ir2[14] = (v232_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v238_data = ir2[15];
              ir2[15] = (v238_data + (v144_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v240_data = r0[2];
              float v244_data = ir2[0];
              ir2[0] = (v244_data + (v240_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v250_data = ir2[1];
              ir2[1] = (v250_data + (v240_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v256_data = ir2[2];
              ir2[2] = (v256_data + (v240_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v262_data = ir2[3];
              ir2[3] = (v262_data + (v240_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v268_data = ir2[4];
              ir2[4] = (v268_data + (v240_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v274_data = ir2[5];
              ir2[5] = (v274_data + (v240_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v280_data = ir2[6];
              ir2[6] = (v280_data + (v240_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v286_data = ir2[7];
              ir2[7] = (v286_data + (v240_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v292_data = ir2[8];
              ir2[8] = (v292_data + (v240_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v298_data = ir2[9];
              ir2[9] = (v298_data + (v240_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v304_data = ir2[10];
              ir2[10] = (v304_data + (v240_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v310_data = ir2[11];
              ir2[11] = (v310_data + (v240_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v316_data = ir2[12];
              ir2[12] = (v316_data + (v240_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v322_data = ir2[13];
              ir2[13] = (v322_data + (v240_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v328_data = ir2[14];
              ir2[14] = (v328_data + (v240_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v334_data = ir2[15];
              ir2[15] = (v334_data + (v240_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v336_data = r0[3];
              float v340_data = ir2[0];
              ir2[0] = (v340_data + (v336_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v346_data = ir2[1];
              ir2[1] = (v346_data + (v336_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v352_data = ir2[2];
              ir2[2] = (v352_data + (v336_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v358_data = ir2[3];
              ir2[3] = (v358_data + (v336_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v364_data = ir2[4];
              ir2[4] = (v364_data + (v336_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v370_data = ir2[5];
              ir2[5] = (v370_data + (v336_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v376_data = ir2[6];
              ir2[6] = (v376_data + (v336_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v382_data = ir2[7];
              ir2[7] = (v382_data + (v336_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v388_data = ir2[8];
              ir2[8] = (v388_data + (v336_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v394_data = ir2[9];
              ir2[9] = (v394_data + (v336_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v400_data = ir2[10];
              ir2[10] = (v400_data + (v336_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v406_data = ir2[11];
              ir2[11] = (v406_data + (v336_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v412_data = ir2[12];
              ir2[12] = (v412_data + (v336_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v418_data = ir2[13];
              ir2[13] = (v418_data + (v336_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v424_data = ir2[14];
              ir2[14] = (v424_data + (v336_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v430_data = ir2[15];
              ir2[15] = (v430_data + (v336_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v432_data = r0[4];
              float v436_data = ir2[0];
              ir2[0] = (v436_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v442_data = ir2[1];
              ir2[1] = (v442_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v448_data = ir2[2];
              ir2[2] = (v448_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v454_data = ir2[3];
              ir2[3] = (v454_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v460_data = ir2[4];
              ir2[4] = (v460_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v466_data = ir2[5];
              ir2[5] = (v466_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v472_data = ir2[6];
              ir2[6] = (v472_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v478_data = ir2[7];
              ir2[7] = (v478_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v484_data = ir2[8];
              ir2[8] = (v484_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v490_data = ir2[9];
              ir2[9] = (v490_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v496_data = ir2[10];
              ir2[10] = (v496_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v502_data = ir2[11];
              ir2[11] = (v502_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v508_data = ir2[12];
              ir2[12] = (v508_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v514_data = ir2[13];
              ir2[13] = (v514_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v520_data = ir2[14];
              ir2[14] = (v520_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v526_data = ir2[15];
              ir2[15] = (v526_data + (v432_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v528_data = r0[5];
              float v532_data = ir2[0];
              ir2[0] = (v532_data + (v528_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v538_data = ir2[1];
              ir2[1] = (v538_data + (v528_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v544_data = ir2[2];
              ir2[2] = (v544_data + (v528_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v550_data = ir2[3];
              ir2[3] = (v550_data + (v528_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v556_data = ir2[4];
              ir2[4] = (v556_data + (v528_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v562_data = ir2[5];
              ir2[5] = (v562_data + (v528_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v568_data = ir2[6];
              ir2[6] = (v568_data + (v528_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v574_data = ir2[7];
              ir2[7] = (v574_data + (v528_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v580_data = ir2[8];
              ir2[8] = (v580_data + (v528_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v586_data = ir2[9];
              ir2[9] = (v586_data + (v528_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v592_data = ir2[10];
              ir2[10] = (v592_data + (v528_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v598_data = ir2[11];
              ir2[11] = (v598_data + (v528_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v604_data = ir2[12];
              ir2[12] = (v604_data + (v528_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v610_data = ir2[13];
              ir2[13] = (v610_data + (v528_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v616_data = ir2[14];
              ir2[14] = (v616_data + (v528_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v622_data = ir2[15];
              ir2[15] = (v622_data + (v528_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v624_data = r0[6];
              float v628_data = ir2[0];
              ir2[0] = (v628_data + (v624_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v634_data = ir2[1];
              ir2[1] = (v634_data + (v624_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v640_data = ir2[2];
              ir2[2] = (v640_data + (v624_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v646_data = ir2[3];
              ir2[3] = (v646_data + (v624_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v652_data = ir2[4];
              ir2[4] = (v652_data + (v624_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v658_data = ir2[5];
              ir2[5] = (v658_data + (v624_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v664_data = ir2[6];
              ir2[6] = (v664_data + (v624_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v670_data = ir2[7];
              ir2[7] = (v670_data + (v624_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v676_data = ir2[8];
              ir2[8] = (v676_data + (v624_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v682_data = ir2[9];
              ir2[9] = (v682_data + (v624_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v688_data = ir2[10];
              ir2[10] = (v688_data + (v624_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v694_data = ir2[11];
              ir2[11] = (v694_data + (v624_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v700_data = ir2[12];
              ir2[12] = (v700_data + (v624_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v706_data = ir2[13];
              ir2[13] = (v706_data + (v624_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v712_data = ir2[14];
              ir2[14] = (v712_data + (v624_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v718_data = ir2[15];
              ir2[15] = (v718_data + (v624_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v720_data = r0[7];
              float v724_data = ir2[0];
              ir2[0] = (v724_data + (v720_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v730_data = ir2[1];
              ir2[1] = (v730_data + (v720_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v736_data = ir2[2];
              ir2[2] = (v736_data + (v720_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v742_data = ir2[3];
              ir2[3] = (v742_data + (v720_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v748_data = ir2[4];
              ir2[4] = (v748_data + (v720_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v754_data = ir2[5];
              ir2[5] = (v754_data + (v720_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v760_data = ir2[6];
              ir2[6] = (v760_data + (v720_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v766_data = ir2[7];
              ir2[7] = (v766_data + (v720_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v772_data = ir2[8];
              ir2[8] = (v772_data + (v720_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v778_data = ir2[9];
              ir2[9] = (v778_data + (v720_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v784_data = ir2[10];
              ir2[10] = (v784_data + (v720_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v790_data = ir2[11];
              ir2[11] = (v790_data + (v720_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v796_data = ir2[12];
              ir2[12] = (v796_data + (v720_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v802_data = ir2[13];
              ir2[13] = (v802_data + (v720_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v808_data = ir2[14];
              ir2[14] = (v808_data + (v720_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v814_data = ir2[15];
              ir2[15] = (v814_data + (v720_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v816_data = r0[8];
              float v820_data = ir2[0];
              ir2[0] = (v820_data + (v816_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v826_data = ir2[1];
              ir2[1] = (v826_data + (v816_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v832_data = ir2[2];
              ir2[2] = (v832_data + (v816_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v838_data = ir2[3];
              ir2[3] = (v838_data + (v816_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v844_data = ir2[4];
              ir2[4] = (v844_data + (v816_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v850_data = ir2[5];
              ir2[5] = (v850_data + (v816_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v856_data = ir2[6];
              ir2[6] = (v856_data + (v816_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v862_data = ir2[7];
              ir2[7] = (v862_data + (v816_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v868_data = ir2[8];
              ir2[8] = (v868_data + (v816_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v874_data = ir2[9];
              ir2[9] = (v874_data + (v816_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v880_data = ir2[10];
              ir2[10] = (v880_data + (v816_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v886_data = ir2[11];
              ir2[11] = (v886_data + (v816_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v892_data = ir2[12];
              ir2[12] = (v892_data + (v816_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v898_data = ir2[13];
              ir2[13] = (v898_data + (v816_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v904_data = ir2[14];
              ir2[14] = (v904_data + (v816_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v910_data = ir2[15];
              ir2[15] = (v910_data + (v816_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v912_data = r0[9];
              float v916_data = ir2[0];
              ir2[0] = (v916_data + (v912_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v922_data = ir2[1];
              ir2[1] = (v922_data + (v912_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v928_data = ir2[2];
              ir2[2] = (v928_data + (v912_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v934_data = ir2[3];
              ir2[3] = (v934_data + (v912_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v940_data = ir2[4];
              ir2[4] = (v940_data + (v912_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v946_data = ir2[5];
              ir2[5] = (v946_data + (v912_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v952_data = ir2[6];
              ir2[6] = (v952_data + (v912_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v958_data = ir2[7];
              ir2[7] = (v958_data + (v912_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v964_data = ir2[8];
              ir2[8] = (v964_data + (v912_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v970_data = ir2[9];
              ir2[9] = (v970_data + (v912_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v976_data = ir2[10];
              ir2[10] = (v976_data + (v912_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v982_data = ir2[11];
              ir2[11] = (v982_data + (v912_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v988_data = ir2[12];
              ir2[12] = (v988_data + (v912_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v994_data = ir2[13];
              ir2[13] = (v994_data + (v912_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1000_data = ir2[14];
              ir2[14] = (v1000_data + (v912_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1006_data = ir2[15];
              ir2[15] = (v1006_data + (v912_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1008_data = r0[10];
              float v1012_data = ir2[0];
              ir2[0] = (v1012_data + (v1008_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1018_data = ir2[1];
              ir2[1] = (v1018_data + (v1008_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1024_data = ir2[2];
              ir2[2] = (v1024_data + (v1008_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1030_data = ir2[3];
              ir2[3] = (v1030_data + (v1008_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1036_data = ir2[4];
              ir2[4] = (v1036_data + (v1008_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1042_data = ir2[5];
              ir2[5] = (v1042_data + (v1008_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1048_data = ir2[6];
              ir2[6] = (v1048_data + (v1008_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1054_data = ir2[7];
              ir2[7] = (v1054_data + (v1008_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1060_data = ir2[8];
              ir2[8] = (v1060_data + (v1008_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1066_data = ir2[9];
              ir2[9] = (v1066_data + (v1008_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1072_data = ir2[10];
              ir2[10] = (v1072_data + (v1008_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1078_data = ir2[11];
              ir2[11] = (v1078_data + (v1008_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1084_data = ir2[12];
              ir2[12] = (v1084_data + (v1008_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1090_data = ir2[13];
              ir2[13] = (v1090_data + (v1008_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1096_data = ir2[14];
              ir2[14] = (v1096_data + (v1008_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1102_data = ir2[15];
              ir2[15] = (v1102_data + (v1008_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1104_data = r0[11];
              float v1108_data = ir2[0];
              ir2[0] = (v1108_data + (v1104_data * (sycl::select_from_group(item.get_sub_group(), v49_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1114_data = ir2[1];
              ir2[1] = (v1114_data + (v1104_data * (sycl::select_from_group(item.get_sub_group(), v55_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1120_data = ir2[2];
              ir2[2] = (v1120_data + (v1104_data * (sycl::select_from_group(item.get_sub_group(), v61_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1126_data = ir2[3];
              ir2[3] = (v1126_data + (v1104_data * (sycl::select_from_group(item.get_sub_group(), v67_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1132_data = ir2[4];
              ir2[4] = (v1132_data + (v1104_data * (sycl::select_from_group(item.get_sub_group(), v73_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1138_data = ir2[5];
              ir2[5] = (v1138_data + (v1104_data * (sycl::select_from_group(item.get_sub_group(), v79_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1144_data = ir2[6];
              ir2[6] = (v1144_data + (v1104_data * (sycl::select_from_group(item.get_sub_group(), v85_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1150_data = ir2[7];
              ir2[7] = (v1150_data + (v1104_data * (sycl::select_from_group(item.get_sub_group(), v91_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1156_data = ir2[8];
              ir2[8] = (v1156_data + (v1104_data * (sycl::select_from_group(item.get_sub_group(), v97_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1162_data = ir2[9];
              ir2[9] = (v1162_data + (v1104_data * (sycl::select_from_group(item.get_sub_group(), v103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1168_data = ir2[10];
              ir2[10] = (v1168_data + (v1104_data * (sycl::select_from_group(item.get_sub_group(), v109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1174_data = ir2[11];
              ir2[11] = (v1174_data + (v1104_data * (sycl::select_from_group(item.get_sub_group(), v115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1180_data = ir2[12];
              ir2[12] = (v1180_data + (v1104_data * (sycl::select_from_group(item.get_sub_group(), v121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1186_data = ir2[13];
              ir2[13] = (v1186_data + (v1104_data * (sycl::select_from_group(item.get_sub_group(), v127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1192_data = ir2[14];
              ir2[14] = (v1192_data + (v1104_data * (sycl::select_from_group(item.get_sub_group(), v133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1198_data = ir2[15];
              ir2[15] = (v1198_data + (v1104_data * (sycl::select_from_group(item.get_sub_group(), v139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              #pragma unroll
              for (int32_t v1200_n0 = 0; v1200_n0 < 1; ++v1200_n0) {
                #pragma unroll
                for (int32_t v1201_n1 = 0; v1201_n1 < 16; ++v1201_n1) {
                  int32_t v1202_a = v1200_n0 + v1201_n1;
                  float v1203_data = ir2[v1202_a];
                  r2[v1202_a] = v1203_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v1204_i0 = 0; v1204_i0 < 1; ++v1204_i0) {
                int32_t v1209_lead = v19_lead + (v1204_i0 * 32);
                #pragma unroll
                for (int32_t v1205_i1 = 0; v1205_i1 < 16; ++v1205_i1) {
                  float v1207_data = r2[(v1204_i0 + v1205_i1)];
                  glb_m0[(v1209_lead + (v1205_i1 * 32))] = v1207_data;
                }
              }
              float r4[8]{};
              // r4 = load{g>r}(glb_m4);
              if (v29_g) {
                #pragma unroll
                for (int32_t v1213_i1 = 0; v1213_i1 < 8; ++v1213_i1) {
                  float v1218_data = glb_m4[(v19_lead + (v1213_i1 * 12))];
                  r4[v1213_i1] = v1218_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m3););
              float r5[8]{};
              // r5 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v1221_i0 = 0; v1221_i0 < 1; ++v1221_i0) {
                int32_t v1224_lead = v19_lead + (v1221_i0 * 32);
                #pragma unroll
                for (int32_t v1222_i1 = 0; v1222_i1 < 8; ++v1222_i1) {
                  float v1227_data = glb_m0[(v1224_lead + (v1222_i1 * 32))];
                  r5[(v1221_i0 + v1222_i1)] = v1227_data;
                }
              }
              // wait(r4 = load{g>r}(glb_m4););
              float r7[12]{};
              // r7 = load{g>r}(glb_m5);
              #pragma unroll
              for (int32_t v1230_i0 = 0; v1230_i0 < 1; ++v1230_i0) {
                int32_t v1233_lead = v19_lead + (v1230_i0 * 32);
                #pragma unroll
                for (int32_t v1231_i1 = 0; v1231_i1 < 12; ++v1231_i1) {
                  float v1236_data = glb_m5[(v1233_lead + (v1231_i1 * 32))];
                  r7[(v1230_i0 + v1231_i1)] = v1236_data;
                }
              }
              // wait(r5 = load{g>r}(glb_m0););
              float r6[8]{};
              // r6 = +(r3 * r4) + name: r5, type: SymbolType.Register, lead: [0]
              // [(0, 32), (0, 8)] [(0, 12)]
              float ir6[8]{};
              float v1240_data = r3[0];
              float v1241_data = r4[0];
              float v1244_data = ir6[0];
              ir6[0] = (v1244_data + (v1240_data * (sycl::select_from_group(item.get_sub_group(), v1241_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1247_data = r4[1];
              float v1250_data = ir6[1];
              ir6[1] = (v1250_data + (v1240_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1253_data = r4[2];
              float v1256_data = ir6[2];
              ir6[2] = (v1256_data + (v1240_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1259_data = r4[3];
              float v1262_data = ir6[3];
              ir6[3] = (v1262_data + (v1240_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1265_data = r4[4];
              float v1268_data = ir6[4];
              ir6[4] = (v1268_data + (v1240_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1271_data = r4[5];
              float v1274_data = ir6[5];
              ir6[5] = (v1274_data + (v1240_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1277_data = r4[6];
              float v1280_data = ir6[6];
              ir6[6] = (v1280_data + (v1240_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1283_data = r4[7];
              float v1286_data = ir6[7];
              ir6[7] = (v1286_data + (v1240_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1288_data = r3[1];
              float v1292_data = ir6[0];
              ir6[0] = (v1292_data + (v1288_data * (sycl::select_from_group(item.get_sub_group(), v1241_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1298_data = ir6[1];
              ir6[1] = (v1298_data + (v1288_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1304_data = ir6[2];
              ir6[2] = (v1304_data + (v1288_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1310_data = ir6[3];
              ir6[3] = (v1310_data + (v1288_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1316_data = ir6[4];
              ir6[4] = (v1316_data + (v1288_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1322_data = ir6[5];
              ir6[5] = (v1322_data + (v1288_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1328_data = ir6[6];
              ir6[6] = (v1328_data + (v1288_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1334_data = ir6[7];
              ir6[7] = (v1334_data + (v1288_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1336_data = r3[2];
              float v1340_data = ir6[0];
              ir6[0] = (v1340_data + (v1336_data * (sycl::select_from_group(item.get_sub_group(), v1241_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1346_data = ir6[1];
              ir6[1] = (v1346_data + (v1336_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1352_data = ir6[2];
              ir6[2] = (v1352_data + (v1336_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1358_data = ir6[3];
              ir6[3] = (v1358_data + (v1336_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1364_data = ir6[4];
              ir6[4] = (v1364_data + (v1336_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1370_data = ir6[5];
              ir6[5] = (v1370_data + (v1336_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1376_data = ir6[6];
              ir6[6] = (v1376_data + (v1336_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1382_data = ir6[7];
              ir6[7] = (v1382_data + (v1336_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1384_data = r3[3];
              float v1388_data = ir6[0];
              ir6[0] = (v1388_data + (v1384_data * (sycl::select_from_group(item.get_sub_group(), v1241_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1394_data = ir6[1];
              ir6[1] = (v1394_data + (v1384_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1400_data = ir6[2];
              ir6[2] = (v1400_data + (v1384_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1406_data = ir6[3];
              ir6[3] = (v1406_data + (v1384_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1412_data = ir6[4];
              ir6[4] = (v1412_data + (v1384_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1418_data = ir6[5];
              ir6[5] = (v1418_data + (v1384_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1424_data = ir6[6];
              ir6[6] = (v1424_data + (v1384_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1430_data = ir6[7];
              ir6[7] = (v1430_data + (v1384_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1432_data = r3[4];
              float v1436_data = ir6[0];
              ir6[0] = (v1436_data + (v1432_data * (sycl::select_from_group(item.get_sub_group(), v1241_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1442_data = ir6[1];
              ir6[1] = (v1442_data + (v1432_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1448_data = ir6[2];
              ir6[2] = (v1448_data + (v1432_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1454_data = ir6[3];
              ir6[3] = (v1454_data + (v1432_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1460_data = ir6[4];
              ir6[4] = (v1460_data + (v1432_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1466_data = ir6[5];
              ir6[5] = (v1466_data + (v1432_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1472_data = ir6[6];
              ir6[6] = (v1472_data + (v1432_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1478_data = ir6[7];
              ir6[7] = (v1478_data + (v1432_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1480_data = r3[5];
              float v1484_data = ir6[0];
              ir6[0] = (v1484_data + (v1480_data * (sycl::select_from_group(item.get_sub_group(), v1241_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1490_data = ir6[1];
              ir6[1] = (v1490_data + (v1480_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1496_data = ir6[2];
              ir6[2] = (v1496_data + (v1480_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1502_data = ir6[3];
              ir6[3] = (v1502_data + (v1480_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1508_data = ir6[4];
              ir6[4] = (v1508_data + (v1480_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1514_data = ir6[5];
              ir6[5] = (v1514_data + (v1480_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1520_data = ir6[6];
              ir6[6] = (v1520_data + (v1480_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1526_data = ir6[7];
              ir6[7] = (v1526_data + (v1480_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1528_data = r3[6];
              float v1532_data = ir6[0];
              ir6[0] = (v1532_data + (v1528_data * (sycl::select_from_group(item.get_sub_group(), v1241_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1538_data = ir6[1];
              ir6[1] = (v1538_data + (v1528_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1544_data = ir6[2];
              ir6[2] = (v1544_data + (v1528_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1550_data = ir6[3];
              ir6[3] = (v1550_data + (v1528_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1556_data = ir6[4];
              ir6[4] = (v1556_data + (v1528_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1562_data = ir6[5];
              ir6[5] = (v1562_data + (v1528_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1568_data = ir6[6];
              ir6[6] = (v1568_data + (v1528_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1574_data = ir6[7];
              ir6[7] = (v1574_data + (v1528_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1576_data = r3[7];
              float v1580_data = ir6[0];
              ir6[0] = (v1580_data + (v1576_data * (sycl::select_from_group(item.get_sub_group(), v1241_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1586_data = ir6[1];
              ir6[1] = (v1586_data + (v1576_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1592_data = ir6[2];
              ir6[2] = (v1592_data + (v1576_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1598_data = ir6[3];
              ir6[3] = (v1598_data + (v1576_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1604_data = ir6[4];
              ir6[4] = (v1604_data + (v1576_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1610_data = ir6[5];
              ir6[5] = (v1610_data + (v1576_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1616_data = ir6[6];
              ir6[6] = (v1616_data + (v1576_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1622_data = ir6[7];
              ir6[7] = (v1622_data + (v1576_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1624_data = r3[8];
              float v1628_data = ir6[0];
              ir6[0] = (v1628_data + (v1624_data * (sycl::select_from_group(item.get_sub_group(), v1241_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1634_data = ir6[1];
              ir6[1] = (v1634_data + (v1624_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1640_data = ir6[2];
              ir6[2] = (v1640_data + (v1624_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1646_data = ir6[3];
              ir6[3] = (v1646_data + (v1624_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1652_data = ir6[4];
              ir6[4] = (v1652_data + (v1624_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1658_data = ir6[5];
              ir6[5] = (v1658_data + (v1624_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1664_data = ir6[6];
              ir6[6] = (v1664_data + (v1624_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1670_data = ir6[7];
              ir6[7] = (v1670_data + (v1624_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1672_data = r3[9];
              float v1676_data = ir6[0];
              ir6[0] = (v1676_data + (v1672_data * (sycl::select_from_group(item.get_sub_group(), v1241_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1682_data = ir6[1];
              ir6[1] = (v1682_data + (v1672_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1688_data = ir6[2];
              ir6[2] = (v1688_data + (v1672_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1694_data = ir6[3];
              ir6[3] = (v1694_data + (v1672_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1700_data = ir6[4];
              ir6[4] = (v1700_data + (v1672_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1706_data = ir6[5];
              ir6[5] = (v1706_data + (v1672_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1712_data = ir6[6];
              ir6[6] = (v1712_data + (v1672_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1718_data = ir6[7];
              ir6[7] = (v1718_data + (v1672_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1720_data = r3[10];
              float v1724_data = ir6[0];
              ir6[0] = (v1724_data + (v1720_data * (sycl::select_from_group(item.get_sub_group(), v1241_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1730_data = ir6[1];
              ir6[1] = (v1730_data + (v1720_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1736_data = ir6[2];
              ir6[2] = (v1736_data + (v1720_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1742_data = ir6[3];
              ir6[3] = (v1742_data + (v1720_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1748_data = ir6[4];
              ir6[4] = (v1748_data + (v1720_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1754_data = ir6[5];
              ir6[5] = (v1754_data + (v1720_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1760_data = ir6[6];
              ir6[6] = (v1760_data + (v1720_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1766_data = ir6[7];
              ir6[7] = (v1766_data + (v1720_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1768_data = r3[11];
              float v1772_data = ir6[0];
              ir6[0] = (v1772_data + (v1768_data * (sycl::select_from_group(item.get_sub_group(), v1241_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1778_data = ir6[1];
              ir6[1] = (v1778_data + (v1768_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1784_data = ir6[2];
              ir6[2] = (v1784_data + (v1768_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1790_data = ir6[3];
              ir6[3] = (v1790_data + (v1768_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1796_data = ir6[4];
              ir6[4] = (v1796_data + (v1768_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1802_data = ir6[5];
              ir6[5] = (v1802_data + (v1768_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1808_data = ir6[6];
              ir6[6] = (v1808_data + (v1768_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1814_data = ir6[7];
              ir6[7] = (v1814_data + (v1768_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              #pragma unroll
              for (int32_t v1816_n0 = 0; v1816_n0 < 1; ++v1816_n0) {
                #pragma unroll
                for (int32_t v1817_n1 = 0; v1817_n1 < 8; ++v1817_n1) {
                  int32_t v1818_a = v1816_n0 + v1817_n1;
                  float v1819_data = ir6[v1818_a];
                  float v1820_data = r5[v1818_a];
                  r6[v1818_a] = (v1820_data + v1819_data);
                }
              }
              // glb_m0 = store{r>g}(r6);
              #pragma unroll
              for (int32_t v1822_i0 = 0; v1822_i0 < 1; ++v1822_i0) {
                int32_t v1827_lead = v19_lead + (v1822_i0 * 32);
                #pragma unroll
                for (int32_t v1823_i1 = 0; v1823_i1 < 8; ++v1823_i1) {
                  float v1825_data = r6[(v1822_i0 + v1823_i1)];
                  glb_m0[(v1827_lead + (v1823_i1 * 32))] = v1825_data;
                }
              }
              float r8[8]{};
              // r8 = load{g>r}(glb_m6);
              if (v29_g) {
                #pragma unroll
                for (int32_t v1831_i1 = 0; v1831_i1 < 8; ++v1831_i1) {
                  float v1836_data = glb_m6[(v19_lead + (v1831_i1 * 12))];
                  r8[v1831_i1] = v1836_data;
                }
              }
              // wait(r7 = load{g>r}(glb_m5););
              float r9[8]{};
              // r9 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v1839_i0 = 0; v1839_i0 < 1; ++v1839_i0) {
                int32_t v1842_lead = v19_lead + (v1839_i0 * 32);
                #pragma unroll
                for (int32_t v1840_i1 = 0; v1840_i1 < 8; ++v1840_i1) {
                  float v1846_data = glb_m0[(v1842_lead + ((v1840_i1 + 8) * 32))];
                  r9[(v1839_i0 + v1840_i1)] = v1846_data;
                }
              }
              // wait(r8 = load{g>r}(glb_m6););
              // wait(r9 = load{g>r}(glb_m0););
              float r10[8]{};
              // r10 = +(r7 * r8) + name: r9, type: SymbolType.Register, lead: [0]
              // [(0, 32), (0, 8)] [(0, 12)]
              float ir10[8]{};
              float v1850_data = r7[0];
              float v1851_data = r8[0];
              float v1854_data = ir10[0];
              ir10[0] = (v1854_data + (v1850_data * (sycl::select_from_group(item.get_sub_group(), v1851_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1857_data = r8[1];
              float v1860_data = ir10[1];
              ir10[1] = (v1860_data + (v1850_data * (sycl::select_from_group(item.get_sub_group(), v1857_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1863_data = r8[2];
              float v1866_data = ir10[2];
              ir10[2] = (v1866_data + (v1850_data * (sycl::select_from_group(item.get_sub_group(), v1863_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1869_data = r8[3];
              float v1872_data = ir10[3];
              ir10[3] = (v1872_data + (v1850_data * (sycl::select_from_group(item.get_sub_group(), v1869_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1875_data = r8[4];
              float v1878_data = ir10[4];
              ir10[4] = (v1878_data + (v1850_data * (sycl::select_from_group(item.get_sub_group(), v1875_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1881_data = r8[5];
              float v1884_data = ir10[5];
              ir10[5] = (v1884_data + (v1850_data * (sycl::select_from_group(item.get_sub_group(), v1881_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1887_data = r8[6];
              float v1890_data = ir10[6];
              ir10[6] = (v1890_data + (v1850_data * (sycl::select_from_group(item.get_sub_group(), v1887_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1893_data = r8[7];
              float v1896_data = ir10[7];
              ir10[7] = (v1896_data + (v1850_data * (sycl::select_from_group(item.get_sub_group(), v1893_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1898_data = r7[1];
              float v1902_data = ir10[0];
              ir10[0] = (v1902_data + (v1898_data * (sycl::select_from_group(item.get_sub_group(), v1851_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1908_data = ir10[1];
              ir10[1] = (v1908_data + (v1898_data * (sycl::select_from_group(item.get_sub_group(), v1857_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1914_data = ir10[2];
              ir10[2] = (v1914_data + (v1898_data * (sycl::select_from_group(item.get_sub_group(), v1863_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1920_data = ir10[3];
              ir10[3] = (v1920_data + (v1898_data * (sycl::select_from_group(item.get_sub_group(), v1869_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1926_data = ir10[4];
              ir10[4] = (v1926_data + (v1898_data * (sycl::select_from_group(item.get_sub_group(), v1875_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1932_data = ir10[5];
              ir10[5] = (v1932_data + (v1898_data * (sycl::select_from_group(item.get_sub_group(), v1881_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1938_data = ir10[6];
              ir10[6] = (v1938_data + (v1898_data * (sycl::select_from_group(item.get_sub_group(), v1887_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1944_data = ir10[7];
              ir10[7] = (v1944_data + (v1898_data * (sycl::select_from_group(item.get_sub_group(), v1893_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1946_data = r7[2];
              float v1950_data = ir10[0];
              ir10[0] = (v1950_data + (v1946_data * (sycl::select_from_group(item.get_sub_group(), v1851_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1956_data = ir10[1];
              ir10[1] = (v1956_data + (v1946_data * (sycl::select_from_group(item.get_sub_group(), v1857_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1962_data = ir10[2];
              ir10[2] = (v1962_data + (v1946_data * (sycl::select_from_group(item.get_sub_group(), v1863_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1968_data = ir10[3];
              ir10[3] = (v1968_data + (v1946_data * (sycl::select_from_group(item.get_sub_group(), v1869_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1974_data = ir10[4];
              ir10[4] = (v1974_data + (v1946_data * (sycl::select_from_group(item.get_sub_group(), v1875_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1980_data = ir10[5];
              ir10[5] = (v1980_data + (v1946_data * (sycl::select_from_group(item.get_sub_group(), v1881_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1986_data = ir10[6];
              ir10[6] = (v1986_data + (v1946_data * (sycl::select_from_group(item.get_sub_group(), v1887_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1992_data = ir10[7];
              ir10[7] = (v1992_data + (v1946_data * (sycl::select_from_group(item.get_sub_group(), v1893_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1994_data = r7[3];
              float v1998_data = ir10[0];
              ir10[0] = (v1998_data + (v1994_data * (sycl::select_from_group(item.get_sub_group(), v1851_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v2004_data = ir10[1];
              ir10[1] = (v2004_data + (v1994_data * (sycl::select_from_group(item.get_sub_group(), v1857_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v2010_data = ir10[2];
              ir10[2] = (v2010_data + (v1994_data * (sycl::select_from_group(item.get_sub_group(), v1863_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v2016_data = ir10[3];
              ir10[3] = (v2016_data + (v1994_data * (sycl::select_from_group(item.get_sub_group(), v1869_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v2022_data = ir10[4];
              ir10[4] = (v2022_data + (v1994_data * (sycl::select_from_group(item.get_sub_group(), v1875_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v2028_data = ir10[5];
              ir10[5] = (v2028_data + (v1994_data * (sycl::select_from_group(item.get_sub_group(), v1881_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v2034_data = ir10[6];
              ir10[6] = (v2034_data + (v1994_data * (sycl::select_from_group(item.get_sub_group(), v1887_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v2040_data = ir10[7];
              ir10[7] = (v2040_data + (v1994_data * (sycl::select_from_group(item.get_sub_group(), v1893_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v2042_data = r7[4];
              float v2046_data = ir10[0];
              ir10[0] = (v2046_data + (v2042_data * (sycl::select_from_group(item.get_sub_group(), v1851_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v2052_data = ir10[1];
              ir10[1] = (v2052_data + (v2042_data * (sycl::select_from_group(item.get_sub_group(), v1857_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v2058_data = ir10[2];
              ir10[2] = (v2058_data + (v2042_data * (sycl::select_from_group(item.get_sub_group(), v1863_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v2064_data = ir10[3];
              ir10[3] = (v2064_data + (v2042_data * (sycl::select_from_group(item.get_sub_group(), v1869_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v2070_data = ir10[4];
              ir10[4] = (v2070_data + (v2042_data * (sycl::select_from_group(item.get_sub_group(), v1875_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v2076_data = ir10[5];
              ir10[5] = (v2076_data + (v2042_data * (sycl::select_from_group(item.get_sub_group(), v1881_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v2082_data = ir10[6];
              ir10[6] = (v2082_data + (v2042_data * (sycl::select_from_group(item.get_sub_group(), v1887_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v2088_data = ir10[7];
              ir10[7] = (v2088_data + (v2042_data * (sycl::select_from_group(item.get_sub_group(), v1893_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v2090_data = r7[5];
              float v2094_data = ir10[0];
              ir10[0] = (v2094_data + (v2090_data * (sycl::select_from_group(item.get_sub_group(), v1851_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v2100_data = ir10[1];
              ir10[1] = (v2100_data + (v2090_data * (sycl::select_from_group(item.get_sub_group(), v1857_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v2106_data = ir10[2];
              ir10[2] = (v2106_data + (v2090_data * (sycl::select_from_group(item.get_sub_group(), v1863_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v2112_data = ir10[3];
              ir10[3] = (v2112_data + (v2090_data * (sycl::select_from_group(item.get_sub_group(), v1869_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v2118_data = ir10[4];
              ir10[4] = (v2118_data + (v2090_data * (sycl::select_from_group(item.get_sub_group(), v1875_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v2124_data = ir10[5];
              ir10[5] = (v2124_data + (v2090_data * (sycl::select_from_group(item.get_sub_group(), v1881_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v2130_data = ir10[6];
              ir10[6] = (v2130_data + (v2090_data * (sycl::select_from_group(item.get_sub_group(), v1887_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v2136_data = ir10[7];
              ir10[7] = (v2136_data + (v2090_data * (sycl::select_from_group(item.get_sub_group(), v1893_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v2138_data = r7[6];
              float v2142_data = ir10[0];
              ir10[0] = (v2142_data + (v2138_data * (sycl::select_from_group(item.get_sub_group(), v1851_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v2148_data = ir10[1];
              ir10[1] = (v2148_data + (v2138_data * (sycl::select_from_group(item.get_sub_group(), v1857_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v2154_data = ir10[2];
              ir10[2] = (v2154_data + (v2138_data * (sycl::select_from_group(item.get_sub_group(), v1863_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v2160_data = ir10[3];
              ir10[3] = (v2160_data + (v2138_data * (sycl::select_from_group(item.get_sub_group(), v1869_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v2166_data = ir10[4];
              ir10[4] = (v2166_data + (v2138_data * (sycl::select_from_group(item.get_sub_group(), v1875_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v2172_data = ir10[5];
              ir10[5] = (v2172_data + (v2138_data * (sycl::select_from_group(item.get_sub_group(), v1881_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v2178_data = ir10[6];
              ir10[6] = (v2178_data + (v2138_data * (sycl::select_from_group(item.get_sub_group(), v1887_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v2184_data = ir10[7];
              ir10[7] = (v2184_data + (v2138_data * (sycl::select_from_group(item.get_sub_group(), v1893_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v2186_data = r7[7];
              float v2190_data = ir10[0];
              ir10[0] = (v2190_data + (v2186_data * (sycl::select_from_group(item.get_sub_group(), v1851_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v2196_data = ir10[1];
              ir10[1] = (v2196_data + (v2186_data * (sycl::select_from_group(item.get_sub_group(), v1857_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v2202_data = ir10[2];
              ir10[2] = (v2202_data + (v2186_data * (sycl::select_from_group(item.get_sub_group(), v1863_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v2208_data = ir10[3];
              ir10[3] = (v2208_data + (v2186_data * (sycl::select_from_group(item.get_sub_group(), v1869_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v2214_data = ir10[4];
              ir10[4] = (v2214_data + (v2186_data * (sycl::select_from_group(item.get_sub_group(), v1875_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v2220_data = ir10[5];
              ir10[5] = (v2220_data + (v2186_data * (sycl::select_from_group(item.get_sub_group(), v1881_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v2226_data = ir10[6];
              ir10[6] = (v2226_data + (v2186_data * (sycl::select_from_group(item.get_sub_group(), v1887_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v2232_data = ir10[7];
              ir10[7] = (v2232_data + (v2186_data * (sycl::select_from_group(item.get_sub_group(), v1893_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v2234_data = r7[8];
              float v2238_data = ir10[0];
              ir10[0] = (v2238_data + (v2234_data * (sycl::select_from_group(item.get_sub_group(), v1851_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v2244_data = ir10[1];
              ir10[1] = (v2244_data + (v2234_data * (sycl::select_from_group(item.get_sub_group(), v1857_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v2250_data = ir10[2];
              ir10[2] = (v2250_data + (v2234_data * (sycl::select_from_group(item.get_sub_group(), v1863_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v2256_data = ir10[3];
              ir10[3] = (v2256_data + (v2234_data * (sycl::select_from_group(item.get_sub_group(), v1869_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v2262_data = ir10[4];
              ir10[4] = (v2262_data + (v2234_data * (sycl::select_from_group(item.get_sub_group(), v1875_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v2268_data = ir10[5];
              ir10[5] = (v2268_data + (v2234_data * (sycl::select_from_group(item.get_sub_group(), v1881_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v2274_data = ir10[6];
              ir10[6] = (v2274_data + (v2234_data * (sycl::select_from_group(item.get_sub_group(), v1887_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v2280_data = ir10[7];
              ir10[7] = (v2280_data + (v2234_data * (sycl::select_from_group(item.get_sub_group(), v1893_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v2282_data = r7[9];
              float v2286_data = ir10[0];
              ir10[0] = (v2286_data + (v2282_data * (sycl::select_from_group(item.get_sub_group(), v1851_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v2292_data = ir10[1];
              ir10[1] = (v2292_data + (v2282_data * (sycl::select_from_group(item.get_sub_group(), v1857_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v2298_data = ir10[2];
              ir10[2] = (v2298_data + (v2282_data * (sycl::select_from_group(item.get_sub_group(), v1863_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v2304_data = ir10[3];
              ir10[3] = (v2304_data + (v2282_data * (sycl::select_from_group(item.get_sub_group(), v1869_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v2310_data = ir10[4];
              ir10[4] = (v2310_data + (v2282_data * (sycl::select_from_group(item.get_sub_group(), v1875_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v2316_data = ir10[5];
              ir10[5] = (v2316_data + (v2282_data * (sycl::select_from_group(item.get_sub_group(), v1881_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v2322_data = ir10[6];
              ir10[6] = (v2322_data + (v2282_data * (sycl::select_from_group(item.get_sub_group(), v1887_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v2328_data = ir10[7];
              ir10[7] = (v2328_data + (v2282_data * (sycl::select_from_group(item.get_sub_group(), v1893_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v2330_data = r7[10];
              float v2334_data = ir10[0];
              ir10[0] = (v2334_data + (v2330_data * (sycl::select_from_group(item.get_sub_group(), v1851_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v2340_data = ir10[1];
              ir10[1] = (v2340_data + (v2330_data * (sycl::select_from_group(item.get_sub_group(), v1857_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v2346_data = ir10[2];
              ir10[2] = (v2346_data + (v2330_data * (sycl::select_from_group(item.get_sub_group(), v1863_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v2352_data = ir10[3];
              ir10[3] = (v2352_data + (v2330_data * (sycl::select_from_group(item.get_sub_group(), v1869_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v2358_data = ir10[4];
              ir10[4] = (v2358_data + (v2330_data * (sycl::select_from_group(item.get_sub_group(), v1875_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v2364_data = ir10[5];
              ir10[5] = (v2364_data + (v2330_data * (sycl::select_from_group(item.get_sub_group(), v1881_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v2370_data = ir10[6];
              ir10[6] = (v2370_data + (v2330_data * (sycl::select_from_group(item.get_sub_group(), v1887_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v2376_data = ir10[7];
              ir10[7] = (v2376_data + (v2330_data * (sycl::select_from_group(item.get_sub_group(), v1893_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v2378_data = r7[11];
              float v2382_data = ir10[0];
              ir10[0] = (v2382_data + (v2378_data * (sycl::select_from_group(item.get_sub_group(), v1851_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v2388_data = ir10[1];
              ir10[1] = (v2388_data + (v2378_data * (sycl::select_from_group(item.get_sub_group(), v1857_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v2394_data = ir10[2];
              ir10[2] = (v2394_data + (v2378_data * (sycl::select_from_group(item.get_sub_group(), v1863_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v2400_data = ir10[3];
              ir10[3] = (v2400_data + (v2378_data * (sycl::select_from_group(item.get_sub_group(), v1869_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v2406_data = ir10[4];
              ir10[4] = (v2406_data + (v2378_data * (sycl::select_from_group(item.get_sub_group(), v1875_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v2412_data = ir10[5];
              ir10[5] = (v2412_data + (v2378_data * (sycl::select_from_group(item.get_sub_group(), v1881_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v2418_data = ir10[6];
              ir10[6] = (v2418_data + (v2378_data * (sycl::select_from_group(item.get_sub_group(), v1887_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v2424_data = ir10[7];
              ir10[7] = (v2424_data + (v2378_data * (sycl::select_from_group(item.get_sub_group(), v1893_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              #pragma unroll
              for (int32_t v2426_n0 = 0; v2426_n0 < 1; ++v2426_n0) {
                #pragma unroll
                for (int32_t v2427_n1 = 0; v2427_n1 < 8; ++v2427_n1) {
                  int32_t v2428_a = v2426_n0 + v2427_n1;
                  float v2429_data = ir10[v2428_a];
                  float v2430_data = r9[v2428_a];
                  r10[v2428_a] = (v2430_data + v2429_data);
                }
              }
              // glb_m0 = store{r>g}(r10);
              #pragma unroll
              for (int32_t v2432_i0 = 0; v2432_i0 < 1; ++v2432_i0) {
                int32_t v2437_lead = v19_lead + (v2432_i0 * 32);
                #pragma unroll
                for (int32_t v2433_i1 = 0; v2433_i1 < 8; ++v2433_i1) {
                  float v2435_data = r10[(v2432_i0 + v2433_i1)];
                  glb_m0[(v2437_lead + ((v2433_i1 + 8) * 32))] = v2435_data;
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

