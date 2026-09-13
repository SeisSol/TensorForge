// === base name ===
kernel_b05d0c8c8bcf60cc

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_b05d0c8c8bcf60cc = {{32, 1, 1}, 32, 32, 1, 1, 0, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_b05d0c8c8bcf60cc(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_b05d0c8c8bcf60cc(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_b05d0c8c8bcf60cc(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_b05d0c8c8bcf60cc(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_b05d0c8c8bcf60cc(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_b05d0c8c8bcf60cc(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_b05d0c8c8bcf60cc(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 32 lanes x 1 per block = block 32x1x1, 0 B shared, occupancy grid
        // operands:
        //   m0 32×13(32×13) {0..32}×{0..13} strided
        //   m1 32×12(32×12) {0..32}×{0..12} strided
        //   m2 12×13(12×13) {0..12}×{0..13} strided
        //   m3 32×13(32×13) {0..32}×{0..13} strided
        //   m4 13×13(13×13) {0..13}×{0..13} strided
        // operations:
        //   t0[i,j] = m0[i,j]
        //   t0[i,j] += m1[i,k] × m2[k,j]
        //   m0[i,j]@{0..32}×{4..5} = t0[i,j]@{0..32}×{4..5}
        //   m3[i,j] = m0[i,k] × m4[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":32,"block":[32,1,1],"cooperative":false,"lead_width":1,"mults_per_block":1,"persistent":true,"sections":[{"barrier":false,"mults_per_block":1,"shared_elements":0}],"shared_bytes":0,"shared_elements":0,"threads_per_mult":32},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[32,13]],"name":"m0","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[32,12]],"name":"m1","ordered":false,"parts":1,"shape":[32,12],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[12,13]],"name":"m2","ordered":false,"parts":1,"shape":[12,13],"variant":false},{"addressing":"strided","alias":"O","bbox":[[0,0],[32,13]],"name":"m3","ordered":false,"parts":1,"shape":[32,13],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[13,13]],"name":"m4","ordered":false,"parts":1,"shape":[13,13],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]}],"permute":[[0,1]],"target":[[0,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,12]},{"addressing":"strided","bbox":[[0,0],[12,13]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[12,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,1]],"is_tmp":false,"name":"m0","offset":[0,4],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,1]],"is_tmp":true,"name":"t0","offset":[0,4],"shape":[32,13]}],"permute":[[0,1]],"target":[[0,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,13]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[32,13]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,13]},{"addressing":"strided","bbox":[[0,0],[13,13]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[13,13]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          for (size_t v1_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v1_batchId0 < numElements0; v1_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v2_ahead1 = v1_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v4_batchId1 = (v2_ahead1 < numElements0) ? v2_ahead1 : v1_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v1_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v1_batchId0 * 416 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v1_batchId0 * 384 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v1_batchId0 * 156 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v1_batchId0 * 416 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v1_batchId0 * 169 + 0 + m4_extraOffset];
              float r0[13]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v17_lead = item.get_local_id(2) % 32;
              #pragma unroll
              for (int32_t v18_i0 = 0; v18_i0 < 1; ++v18_i0) {
                int32_t v21_lead = v17_lead + (v18_i0 * 32);
                #pragma unroll
                for (int32_t v19_i1 = 0; v19_i1 < 13; ++v19_i1) {
                  float v24_data = glb_m0[(v21_lead + (v19_i1 * 32))];
                  r0[(v18_i0 + v19_i1)] = v24_data;
                }
              }
              float r2[12]{};
              // r2 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v27_i0 = 0; v27_i0 < 1; ++v27_i0) {
                int32_t v30_lead = v17_lead + (v27_i0 * 32);
                #pragma unroll
                for (int32_t v28_i1 = 0; v28_i1 < 12; ++v28_i1) {
                  float v33_data = glb_m1[(v30_lead + (v28_i1 * 32))];
                  r2[(v27_i0 + v28_i1)] = v33_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r1[13]{};
              // r1 = +(r0) + None
              // [(0, 32), (0, 13)] []
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
              float v63_data = r0[9];
              float v64_data = r1[9];
              r1[9] = (v64_data + v63_data);
              float v66_data = r0[10];
              float v67_data = r1[10];
              r1[10] = (v67_data + v66_data);
              float v69_data = r0[11];
              float v70_data = r1[11];
              r1[11] = (v70_data + v69_data);
              float v72_data = r0[12];
              float v73_data = r1[12];
              r1[12] = (v73_data + v72_data);
              float r3[13]{};
              // r3 = load{g>r}(glb_m2);
              if (v17_lead < 12) {
                #pragma unroll
                for (int32_t v77_i1 = 0; v77_i1 < 13; ++v77_i1) {
                  float v82_data = glb_m2[(v17_lead + (v77_i1 * 12))];
                  r3[v77_i1] = v82_data;
                }
              }
              // wait(r2 = load{g>r}(glb_m1););
              // wait(r3 = load{g>r}(glb_m2););
              float r4[13]{};
              // r4 = +(r2 * r3) + name: r1, type: SymbolType.Register, lead: [0]
              // [(0, 32), (0, 13)] [(0, 12)]
              float ir4[13]{};
              float v86_data = r2[0];
              float v87_data = r3[0];
              float v90_data = ir4[0];
              ir4[0] = (v90_data + (v86_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v93_data = r3[1];
              float v96_data = ir4[1];
              ir4[1] = (v96_data + (v86_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v99_data = r3[2];
              float v102_data = ir4[2];
              ir4[2] = (v102_data + (v86_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v105_data = r3[3];
              float v108_data = ir4[3];
              ir4[3] = (v108_data + (v86_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v111_data = r3[4];
              float v114_data = ir4[4];
              ir4[4] = (v114_data + (v86_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v117_data = r3[5];
              float v120_data = ir4[5];
              ir4[5] = (v120_data + (v86_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v123_data = r3[6];
              float v126_data = ir4[6];
              ir4[6] = (v126_data + (v86_data * (sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v129_data = r3[7];
              float v132_data = ir4[7];
              ir4[7] = (v132_data + (v86_data * (sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v135_data = r3[8];
              float v138_data = ir4[8];
              ir4[8] = (v138_data + (v86_data * (sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v141_data = r3[9];
              float v144_data = ir4[9];
              ir4[9] = (v144_data + (v86_data * (sycl::select_from_group(item.get_sub_group(), v141_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v147_data = r3[10];
              float v150_data = ir4[10];
              ir4[10] = (v150_data + (v86_data * (sycl::select_from_group(item.get_sub_group(), v147_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v153_data = r3[11];
              float v156_data = ir4[11];
              ir4[11] = (v156_data + (v86_data * (sycl::select_from_group(item.get_sub_group(), v153_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v159_data = r3[12];
              float v162_data = ir4[12];
              ir4[12] = (v162_data + (v86_data * (sycl::select_from_group(item.get_sub_group(), v159_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v164_data = r2[1];
              float v168_data = ir4[0];
              ir4[0] = (v168_data + (v164_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v174_data = ir4[1];
              ir4[1] = (v174_data + (v164_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v180_data = ir4[2];
              ir4[2] = (v180_data + (v164_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v186_data = ir4[3];
              ir4[3] = (v186_data + (v164_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v192_data = ir4[4];
              ir4[4] = (v192_data + (v164_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v198_data = ir4[5];
              ir4[5] = (v198_data + (v164_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v204_data = ir4[6];
              ir4[6] = (v204_data + (v164_data * (sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v210_data = ir4[7];
              ir4[7] = (v210_data + (v164_data * (sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v216_data = ir4[8];
              ir4[8] = (v216_data + (v164_data * (sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v222_data = ir4[9];
              ir4[9] = (v222_data + (v164_data * (sycl::select_from_group(item.get_sub_group(), v141_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v228_data = ir4[10];
              ir4[10] = (v228_data + (v164_data * (sycl::select_from_group(item.get_sub_group(), v147_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v234_data = ir4[11];
              ir4[11] = (v234_data + (v164_data * (sycl::select_from_group(item.get_sub_group(), v153_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v240_data = ir4[12];
              ir4[12] = (v240_data + (v164_data * (sycl::select_from_group(item.get_sub_group(), v159_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v242_data = r2[2];
              float v246_data = ir4[0];
              ir4[0] = (v246_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v252_data = ir4[1];
              ir4[1] = (v252_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v258_data = ir4[2];
              ir4[2] = (v258_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v264_data = ir4[3];
              ir4[3] = (v264_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v270_data = ir4[4];
              ir4[4] = (v270_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v276_data = ir4[5];
              ir4[5] = (v276_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v282_data = ir4[6];
              ir4[6] = (v282_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v288_data = ir4[7];
              ir4[7] = (v288_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v294_data = ir4[8];
              ir4[8] = (v294_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v300_data = ir4[9];
              ir4[9] = (v300_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v141_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v306_data = ir4[10];
              ir4[10] = (v306_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v147_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v312_data = ir4[11];
              ir4[11] = (v312_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v153_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v318_data = ir4[12];
              ir4[12] = (v318_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v159_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v320_data = r2[3];
              float v324_data = ir4[0];
              ir4[0] = (v324_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v330_data = ir4[1];
              ir4[1] = (v330_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v336_data = ir4[2];
              ir4[2] = (v336_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v342_data = ir4[3];
              ir4[3] = (v342_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v348_data = ir4[4];
              ir4[4] = (v348_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v354_data = ir4[5];
              ir4[5] = (v354_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v360_data = ir4[6];
              ir4[6] = (v360_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v366_data = ir4[7];
              ir4[7] = (v366_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v372_data = ir4[8];
              ir4[8] = (v372_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v378_data = ir4[9];
              ir4[9] = (v378_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v141_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v384_data = ir4[10];
              ir4[10] = (v384_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v147_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v390_data = ir4[11];
              ir4[11] = (v390_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v153_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v396_data = ir4[12];
              ir4[12] = (v396_data + (v320_data * (sycl::select_from_group(item.get_sub_group(), v159_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v398_data = r2[4];
              float v402_data = ir4[0];
              ir4[0] = (v402_data + (v398_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v408_data = ir4[1];
              ir4[1] = (v408_data + (v398_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v414_data = ir4[2];
              ir4[2] = (v414_data + (v398_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v420_data = ir4[3];
              ir4[3] = (v420_data + (v398_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v426_data = ir4[4];
              ir4[4] = (v426_data + (v398_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v432_data = ir4[5];
              ir4[5] = (v432_data + (v398_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v438_data = ir4[6];
              ir4[6] = (v438_data + (v398_data * (sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v444_data = ir4[7];
              ir4[7] = (v444_data + (v398_data * (sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v450_data = ir4[8];
              ir4[8] = (v450_data + (v398_data * (sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v456_data = ir4[9];
              ir4[9] = (v456_data + (v398_data * (sycl::select_from_group(item.get_sub_group(), v141_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v462_data = ir4[10];
              ir4[10] = (v462_data + (v398_data * (sycl::select_from_group(item.get_sub_group(), v147_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v468_data = ir4[11];
              ir4[11] = (v468_data + (v398_data * (sycl::select_from_group(item.get_sub_group(), v153_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v474_data = ir4[12];
              ir4[12] = (v474_data + (v398_data * (sycl::select_from_group(item.get_sub_group(), v159_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v476_data = r2[5];
              float v480_data = ir4[0];
              ir4[0] = (v480_data + (v476_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v486_data = ir4[1];
              ir4[1] = (v486_data + (v476_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v492_data = ir4[2];
              ir4[2] = (v492_data + (v476_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v498_data = ir4[3];
              ir4[3] = (v498_data + (v476_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v504_data = ir4[4];
              ir4[4] = (v504_data + (v476_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v510_data = ir4[5];
              ir4[5] = (v510_data + (v476_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v516_data = ir4[6];
              ir4[6] = (v516_data + (v476_data * (sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v522_data = ir4[7];
              ir4[7] = (v522_data + (v476_data * (sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v528_data = ir4[8];
              ir4[8] = (v528_data + (v476_data * (sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v534_data = ir4[9];
              ir4[9] = (v534_data + (v476_data * (sycl::select_from_group(item.get_sub_group(), v141_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v540_data = ir4[10];
              ir4[10] = (v540_data + (v476_data * (sycl::select_from_group(item.get_sub_group(), v147_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v546_data = ir4[11];
              ir4[11] = (v546_data + (v476_data * (sycl::select_from_group(item.get_sub_group(), v153_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v552_data = ir4[12];
              ir4[12] = (v552_data + (v476_data * (sycl::select_from_group(item.get_sub_group(), v159_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v554_data = r2[6];
              float v558_data = ir4[0];
              ir4[0] = (v558_data + (v554_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v564_data = ir4[1];
              ir4[1] = (v564_data + (v554_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v570_data = ir4[2];
              ir4[2] = (v570_data + (v554_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v576_data = ir4[3];
              ir4[3] = (v576_data + (v554_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v582_data = ir4[4];
              ir4[4] = (v582_data + (v554_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v588_data = ir4[5];
              ir4[5] = (v588_data + (v554_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v594_data = ir4[6];
              ir4[6] = (v594_data + (v554_data * (sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v600_data = ir4[7];
              ir4[7] = (v600_data + (v554_data * (sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v606_data = ir4[8];
              ir4[8] = (v606_data + (v554_data * (sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v612_data = ir4[9];
              ir4[9] = (v612_data + (v554_data * (sycl::select_from_group(item.get_sub_group(), v141_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v618_data = ir4[10];
              ir4[10] = (v618_data + (v554_data * (sycl::select_from_group(item.get_sub_group(), v147_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v624_data = ir4[11];
              ir4[11] = (v624_data + (v554_data * (sycl::select_from_group(item.get_sub_group(), v153_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v630_data = ir4[12];
              ir4[12] = (v630_data + (v554_data * (sycl::select_from_group(item.get_sub_group(), v159_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v632_data = r2[7];
              float v636_data = ir4[0];
              ir4[0] = (v636_data + (v632_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v642_data = ir4[1];
              ir4[1] = (v642_data + (v632_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v648_data = ir4[2];
              ir4[2] = (v648_data + (v632_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v654_data = ir4[3];
              ir4[3] = (v654_data + (v632_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v660_data = ir4[4];
              ir4[4] = (v660_data + (v632_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v666_data = ir4[5];
              ir4[5] = (v666_data + (v632_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v672_data = ir4[6];
              ir4[6] = (v672_data + (v632_data * (sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v678_data = ir4[7];
              ir4[7] = (v678_data + (v632_data * (sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v684_data = ir4[8];
              ir4[8] = (v684_data + (v632_data * (sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v690_data = ir4[9];
              ir4[9] = (v690_data + (v632_data * (sycl::select_from_group(item.get_sub_group(), v141_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v696_data = ir4[10];
              ir4[10] = (v696_data + (v632_data * (sycl::select_from_group(item.get_sub_group(), v147_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v702_data = ir4[11];
              ir4[11] = (v702_data + (v632_data * (sycl::select_from_group(item.get_sub_group(), v153_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v708_data = ir4[12];
              ir4[12] = (v708_data + (v632_data * (sycl::select_from_group(item.get_sub_group(), v159_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v710_data = r2[8];
              float v714_data = ir4[0];
              ir4[0] = (v714_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v720_data = ir4[1];
              ir4[1] = (v720_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v726_data = ir4[2];
              ir4[2] = (v726_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v732_data = ir4[3];
              ir4[3] = (v732_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v738_data = ir4[4];
              ir4[4] = (v738_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v744_data = ir4[5];
              ir4[5] = (v744_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v750_data = ir4[6];
              ir4[6] = (v750_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v756_data = ir4[7];
              ir4[7] = (v756_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v762_data = ir4[8];
              ir4[8] = (v762_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v768_data = ir4[9];
              ir4[9] = (v768_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v141_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v774_data = ir4[10];
              ir4[10] = (v774_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v147_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v780_data = ir4[11];
              ir4[11] = (v780_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v153_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v786_data = ir4[12];
              ir4[12] = (v786_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v159_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v788_data = r2[9];
              float v792_data = ir4[0];
              ir4[0] = (v792_data + (v788_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v798_data = ir4[1];
              ir4[1] = (v798_data + (v788_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v804_data = ir4[2];
              ir4[2] = (v804_data + (v788_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v810_data = ir4[3];
              ir4[3] = (v810_data + (v788_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v816_data = ir4[4];
              ir4[4] = (v816_data + (v788_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v822_data = ir4[5];
              ir4[5] = (v822_data + (v788_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v828_data = ir4[6];
              ir4[6] = (v828_data + (v788_data * (sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v834_data = ir4[7];
              ir4[7] = (v834_data + (v788_data * (sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v840_data = ir4[8];
              ir4[8] = (v840_data + (v788_data * (sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v846_data = ir4[9];
              ir4[9] = (v846_data + (v788_data * (sycl::select_from_group(item.get_sub_group(), v141_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v852_data = ir4[10];
              ir4[10] = (v852_data + (v788_data * (sycl::select_from_group(item.get_sub_group(), v147_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v858_data = ir4[11];
              ir4[11] = (v858_data + (v788_data * (sycl::select_from_group(item.get_sub_group(), v153_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v864_data = ir4[12];
              ir4[12] = (v864_data + (v788_data * (sycl::select_from_group(item.get_sub_group(), v159_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v866_data = r2[10];
              float v870_data = ir4[0];
              ir4[0] = (v870_data + (v866_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v876_data = ir4[1];
              ir4[1] = (v876_data + (v866_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v882_data = ir4[2];
              ir4[2] = (v882_data + (v866_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v888_data = ir4[3];
              ir4[3] = (v888_data + (v866_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v894_data = ir4[4];
              ir4[4] = (v894_data + (v866_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v900_data = ir4[5];
              ir4[5] = (v900_data + (v866_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v906_data = ir4[6];
              ir4[6] = (v906_data + (v866_data * (sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v912_data = ir4[7];
              ir4[7] = (v912_data + (v866_data * (sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v918_data = ir4[8];
              ir4[8] = (v918_data + (v866_data * (sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v924_data = ir4[9];
              ir4[9] = (v924_data + (v866_data * (sycl::select_from_group(item.get_sub_group(), v141_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v930_data = ir4[10];
              ir4[10] = (v930_data + (v866_data * (sycl::select_from_group(item.get_sub_group(), v147_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v936_data = ir4[11];
              ir4[11] = (v936_data + (v866_data * (sycl::select_from_group(item.get_sub_group(), v153_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v942_data = ir4[12];
              ir4[12] = (v942_data + (v866_data * (sycl::select_from_group(item.get_sub_group(), v159_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v944_data = r2[11];
              float v948_data = ir4[0];
              ir4[0] = (v948_data + (v944_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v954_data = ir4[1];
              ir4[1] = (v954_data + (v944_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v960_data = ir4[2];
              ir4[2] = (v960_data + (v944_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v966_data = ir4[3];
              ir4[3] = (v966_data + (v944_data * (sycl::select_from_group(item.get_sub_group(), v105_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v972_data = ir4[4];
              ir4[4] = (v972_data + (v944_data * (sycl::select_from_group(item.get_sub_group(), v111_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v978_data = ir4[5];
              ir4[5] = (v978_data + (v944_data * (sycl::select_from_group(item.get_sub_group(), v117_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v984_data = ir4[6];
              ir4[6] = (v984_data + (v944_data * (sycl::select_from_group(item.get_sub_group(), v123_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v990_data = ir4[7];
              ir4[7] = (v990_data + (v944_data * (sycl::select_from_group(item.get_sub_group(), v129_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v996_data = ir4[8];
              ir4[8] = (v996_data + (v944_data * (sycl::select_from_group(item.get_sub_group(), v135_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1002_data = ir4[9];
              ir4[9] = (v1002_data + (v944_data * (sycl::select_from_group(item.get_sub_group(), v141_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1008_data = ir4[10];
              ir4[10] = (v1008_data + (v944_data * (sycl::select_from_group(item.get_sub_group(), v147_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1014_data = ir4[11];
              ir4[11] = (v1014_data + (v944_data * (sycl::select_from_group(item.get_sub_group(), v153_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1020_data = ir4[12];
              ir4[12] = (v1020_data + (v944_data * (sycl::select_from_group(item.get_sub_group(), v159_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              #pragma unroll
              for (int32_t v1022_n0 = 0; v1022_n0 < 1; ++v1022_n0) {
                #pragma unroll
                for (int32_t v1023_n1 = 0; v1023_n1 < 13; ++v1023_n1) {
                  int32_t v1024_a = v1022_n0 + v1023_n1;
                  float v1025_data = ir4[v1024_a];
                  float v1026_data = r1[v1024_a];
                  r4[v1024_a] = (v1026_data + v1025_data);
                }
              }
              float r5[1]{};
              // r5 = +(r4) + None
              // [(0, 32), (0, 1)] []
              float ir5[1]{};
              float v1030_data = r4[4];
              float v1031_data = ir5[0];
              ir5[0] = (v1031_data + v1030_data);
              #pragma unroll
              for (int32_t v1033_n0 = 0; v1033_n0 < 1; ++v1033_n0) {
                #pragma unroll
                for (int32_t v1034_n1 = 0; v1034_n1 < 1; ++v1034_n1) {
                  int32_t v1035_a = v1033_n0 + v1034_n1;
                  float v1036_data = ir5[v1035_a];
                  r5[v1035_a] = v1036_data;
                }
              }
              // glb_m0 = store{r>g}(r5);
              #pragma unroll
              for (int32_t v1037_i0 = 0; v1037_i0 < 1; ++v1037_i0) {
                int32_t v1042_lead = v17_lead + (v1037_i0 * 32);
                #pragma unroll
                for (int32_t v1038_i1 = 0; v1038_i1 < 1; ++v1038_i1) {
                  float v1040_data = r5[(v1037_i0 + v1038_i1)];
                  glb_m0[(v1042_lead + ((v1038_i1 + 4) * 32))] = v1040_data;
                }
              }
              float r6[13]{};
              // r6 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v1047_i0 = 0; v1047_i0 < 1; ++v1047_i0) {
                int32_t v1050_lead = v17_lead + (v1047_i0 * 32);
                #pragma unroll
                for (int32_t v1048_i1 = 0; v1048_i1 < 13; ++v1048_i1) {
                  float v1053_data = glb_m0[(v1050_lead + (v1048_i1 * 32))];
                  r6[(v1047_i0 + v1048_i1)] = v1053_data;
                }
              }
              float r7[13]{};
              // r7 = load{g>r}(glb_m4);
              if (v17_lead < 13) {
                #pragma unroll
                for (int32_t v1057_i1 = 0; v1057_i1 < 13; ++v1057_i1) {
                  float v1062_data = glb_m4[(v17_lead + (v1057_i1 * 13))];
                  r7[v1057_i1] = v1062_data;
                }
              }
              // wait(r6 = load{g>r}(glb_m0););
              // wait(r7 = load{g>r}(glb_m4););
              float r8[13]{};
              // r8 = +(r6 * r7) + None
              // [(0, 32), (0, 13)] [(0, 13)]
              float ir8[13]{};
              float v1066_data = r6[0];
              float v1067_data = r7[0];
              float v1070_data = ir8[0];
              ir8[0] = (v1070_data + (v1066_data * (sycl::select_from_group(item.get_sub_group(), v1067_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1073_data = r7[1];
              float v1076_data = ir8[1];
              ir8[1] = (v1076_data + (v1066_data * (sycl::select_from_group(item.get_sub_group(), v1073_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1079_data = r7[2];
              float v1082_data = ir8[2];
              ir8[2] = (v1082_data + (v1066_data * (sycl::select_from_group(item.get_sub_group(), v1079_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1085_data = r7[3];
              float v1088_data = ir8[3];
              ir8[3] = (v1088_data + (v1066_data * (sycl::select_from_group(item.get_sub_group(), v1085_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1091_data = r7[4];
              float v1094_data = ir8[4];
              ir8[4] = (v1094_data + (v1066_data * (sycl::select_from_group(item.get_sub_group(), v1091_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1097_data = r7[5];
              float v1100_data = ir8[5];
              ir8[5] = (v1100_data + (v1066_data * (sycl::select_from_group(item.get_sub_group(), v1097_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1103_data = r7[6];
              float v1106_data = ir8[6];
              ir8[6] = (v1106_data + (v1066_data * (sycl::select_from_group(item.get_sub_group(), v1103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1109_data = r7[7];
              float v1112_data = ir8[7];
              ir8[7] = (v1112_data + (v1066_data * (sycl::select_from_group(item.get_sub_group(), v1109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1115_data = r7[8];
              float v1118_data = ir8[8];
              ir8[8] = (v1118_data + (v1066_data * (sycl::select_from_group(item.get_sub_group(), v1115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1121_data = r7[9];
              float v1124_data = ir8[9];
              ir8[9] = (v1124_data + (v1066_data * (sycl::select_from_group(item.get_sub_group(), v1121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1127_data = r7[10];
              float v1130_data = ir8[10];
              ir8[10] = (v1130_data + (v1066_data * (sycl::select_from_group(item.get_sub_group(), v1127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1133_data = r7[11];
              float v1136_data = ir8[11];
              ir8[11] = (v1136_data + (v1066_data * (sycl::select_from_group(item.get_sub_group(), v1133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1139_data = r7[12];
              float v1142_data = ir8[12];
              ir8[12] = (v1142_data + (v1066_data * (sycl::select_from_group(item.get_sub_group(), v1139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (0)))));
              float v1144_data = r6[1];
              float v1148_data = ir8[0];
              ir8[0] = (v1148_data + (v1144_data * (sycl::select_from_group(item.get_sub_group(), v1067_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1154_data = ir8[1];
              ir8[1] = (v1154_data + (v1144_data * (sycl::select_from_group(item.get_sub_group(), v1073_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1160_data = ir8[2];
              ir8[2] = (v1160_data + (v1144_data * (sycl::select_from_group(item.get_sub_group(), v1079_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1166_data = ir8[3];
              ir8[3] = (v1166_data + (v1144_data * (sycl::select_from_group(item.get_sub_group(), v1085_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1172_data = ir8[4];
              ir8[4] = (v1172_data + (v1144_data * (sycl::select_from_group(item.get_sub_group(), v1091_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1178_data = ir8[5];
              ir8[5] = (v1178_data + (v1144_data * (sycl::select_from_group(item.get_sub_group(), v1097_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1184_data = ir8[6];
              ir8[6] = (v1184_data + (v1144_data * (sycl::select_from_group(item.get_sub_group(), v1103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1190_data = ir8[7];
              ir8[7] = (v1190_data + (v1144_data * (sycl::select_from_group(item.get_sub_group(), v1109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1196_data = ir8[8];
              ir8[8] = (v1196_data + (v1144_data * (sycl::select_from_group(item.get_sub_group(), v1115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1202_data = ir8[9];
              ir8[9] = (v1202_data + (v1144_data * (sycl::select_from_group(item.get_sub_group(), v1121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1208_data = ir8[10];
              ir8[10] = (v1208_data + (v1144_data * (sycl::select_from_group(item.get_sub_group(), v1127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1214_data = ir8[11];
              ir8[11] = (v1214_data + (v1144_data * (sycl::select_from_group(item.get_sub_group(), v1133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1220_data = ir8[12];
              ir8[12] = (v1220_data + (v1144_data * (sycl::select_from_group(item.get_sub_group(), v1139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (1)))));
              float v1222_data = r6[2];
              float v1226_data = ir8[0];
              ir8[0] = (v1226_data + (v1222_data * (sycl::select_from_group(item.get_sub_group(), v1067_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1232_data = ir8[1];
              ir8[1] = (v1232_data + (v1222_data * (sycl::select_from_group(item.get_sub_group(), v1073_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1238_data = ir8[2];
              ir8[2] = (v1238_data + (v1222_data * (sycl::select_from_group(item.get_sub_group(), v1079_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1244_data = ir8[3];
              ir8[3] = (v1244_data + (v1222_data * (sycl::select_from_group(item.get_sub_group(), v1085_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1250_data = ir8[4];
              ir8[4] = (v1250_data + (v1222_data * (sycl::select_from_group(item.get_sub_group(), v1091_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1256_data = ir8[5];
              ir8[5] = (v1256_data + (v1222_data * (sycl::select_from_group(item.get_sub_group(), v1097_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1262_data = ir8[6];
              ir8[6] = (v1262_data + (v1222_data * (sycl::select_from_group(item.get_sub_group(), v1103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1268_data = ir8[7];
              ir8[7] = (v1268_data + (v1222_data * (sycl::select_from_group(item.get_sub_group(), v1109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1274_data = ir8[8];
              ir8[8] = (v1274_data + (v1222_data * (sycl::select_from_group(item.get_sub_group(), v1115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1280_data = ir8[9];
              ir8[9] = (v1280_data + (v1222_data * (sycl::select_from_group(item.get_sub_group(), v1121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1286_data = ir8[10];
              ir8[10] = (v1286_data + (v1222_data * (sycl::select_from_group(item.get_sub_group(), v1127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1292_data = ir8[11];
              ir8[11] = (v1292_data + (v1222_data * (sycl::select_from_group(item.get_sub_group(), v1133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1298_data = ir8[12];
              ir8[12] = (v1298_data + (v1222_data * (sycl::select_from_group(item.get_sub_group(), v1139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (2)))));
              float v1300_data = r6[3];
              float v1304_data = ir8[0];
              ir8[0] = (v1304_data + (v1300_data * (sycl::select_from_group(item.get_sub_group(), v1067_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1310_data = ir8[1];
              ir8[1] = (v1310_data + (v1300_data * (sycl::select_from_group(item.get_sub_group(), v1073_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1316_data = ir8[2];
              ir8[2] = (v1316_data + (v1300_data * (sycl::select_from_group(item.get_sub_group(), v1079_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1322_data = ir8[3];
              ir8[3] = (v1322_data + (v1300_data * (sycl::select_from_group(item.get_sub_group(), v1085_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1328_data = ir8[4];
              ir8[4] = (v1328_data + (v1300_data * (sycl::select_from_group(item.get_sub_group(), v1091_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1334_data = ir8[5];
              ir8[5] = (v1334_data + (v1300_data * (sycl::select_from_group(item.get_sub_group(), v1097_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1340_data = ir8[6];
              ir8[6] = (v1340_data + (v1300_data * (sycl::select_from_group(item.get_sub_group(), v1103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1346_data = ir8[7];
              ir8[7] = (v1346_data + (v1300_data * (sycl::select_from_group(item.get_sub_group(), v1109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1352_data = ir8[8];
              ir8[8] = (v1352_data + (v1300_data * (sycl::select_from_group(item.get_sub_group(), v1115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1358_data = ir8[9];
              ir8[9] = (v1358_data + (v1300_data * (sycl::select_from_group(item.get_sub_group(), v1121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1364_data = ir8[10];
              ir8[10] = (v1364_data + (v1300_data * (sycl::select_from_group(item.get_sub_group(), v1127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1370_data = ir8[11];
              ir8[11] = (v1370_data + (v1300_data * (sycl::select_from_group(item.get_sub_group(), v1133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1376_data = ir8[12];
              ir8[12] = (v1376_data + (v1300_data * (sycl::select_from_group(item.get_sub_group(), v1139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (3)))));
              float v1378_data = r6[4];
              float v1382_data = ir8[0];
              ir8[0] = (v1382_data + (v1378_data * (sycl::select_from_group(item.get_sub_group(), v1067_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1388_data = ir8[1];
              ir8[1] = (v1388_data + (v1378_data * (sycl::select_from_group(item.get_sub_group(), v1073_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1394_data = ir8[2];
              ir8[2] = (v1394_data + (v1378_data * (sycl::select_from_group(item.get_sub_group(), v1079_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1400_data = ir8[3];
              ir8[3] = (v1400_data + (v1378_data * (sycl::select_from_group(item.get_sub_group(), v1085_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1406_data = ir8[4];
              ir8[4] = (v1406_data + (v1378_data * (sycl::select_from_group(item.get_sub_group(), v1091_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1412_data = ir8[5];
              ir8[5] = (v1412_data + (v1378_data * (sycl::select_from_group(item.get_sub_group(), v1097_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1418_data = ir8[6];
              ir8[6] = (v1418_data + (v1378_data * (sycl::select_from_group(item.get_sub_group(), v1103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1424_data = ir8[7];
              ir8[7] = (v1424_data + (v1378_data * (sycl::select_from_group(item.get_sub_group(), v1109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1430_data = ir8[8];
              ir8[8] = (v1430_data + (v1378_data * (sycl::select_from_group(item.get_sub_group(), v1115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1436_data = ir8[9];
              ir8[9] = (v1436_data + (v1378_data * (sycl::select_from_group(item.get_sub_group(), v1121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1442_data = ir8[10];
              ir8[10] = (v1442_data + (v1378_data * (sycl::select_from_group(item.get_sub_group(), v1127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1448_data = ir8[11];
              ir8[11] = (v1448_data + (v1378_data * (sycl::select_from_group(item.get_sub_group(), v1133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1454_data = ir8[12];
              ir8[12] = (v1454_data + (v1378_data * (sycl::select_from_group(item.get_sub_group(), v1139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (4)))));
              float v1456_data = r6[5];
              float v1460_data = ir8[0];
              ir8[0] = (v1460_data + (v1456_data * (sycl::select_from_group(item.get_sub_group(), v1067_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1466_data = ir8[1];
              ir8[1] = (v1466_data + (v1456_data * (sycl::select_from_group(item.get_sub_group(), v1073_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1472_data = ir8[2];
              ir8[2] = (v1472_data + (v1456_data * (sycl::select_from_group(item.get_sub_group(), v1079_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1478_data = ir8[3];
              ir8[3] = (v1478_data + (v1456_data * (sycl::select_from_group(item.get_sub_group(), v1085_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1484_data = ir8[4];
              ir8[4] = (v1484_data + (v1456_data * (sycl::select_from_group(item.get_sub_group(), v1091_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1490_data = ir8[5];
              ir8[5] = (v1490_data + (v1456_data * (sycl::select_from_group(item.get_sub_group(), v1097_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1496_data = ir8[6];
              ir8[6] = (v1496_data + (v1456_data * (sycl::select_from_group(item.get_sub_group(), v1103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1502_data = ir8[7];
              ir8[7] = (v1502_data + (v1456_data * (sycl::select_from_group(item.get_sub_group(), v1109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1508_data = ir8[8];
              ir8[8] = (v1508_data + (v1456_data * (sycl::select_from_group(item.get_sub_group(), v1115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1514_data = ir8[9];
              ir8[9] = (v1514_data + (v1456_data * (sycl::select_from_group(item.get_sub_group(), v1121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1520_data = ir8[10];
              ir8[10] = (v1520_data + (v1456_data * (sycl::select_from_group(item.get_sub_group(), v1127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1526_data = ir8[11];
              ir8[11] = (v1526_data + (v1456_data * (sycl::select_from_group(item.get_sub_group(), v1133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1532_data = ir8[12];
              ir8[12] = (v1532_data + (v1456_data * (sycl::select_from_group(item.get_sub_group(), v1139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (5)))));
              float v1534_data = r6[6];
              float v1538_data = ir8[0];
              ir8[0] = (v1538_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1067_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1544_data = ir8[1];
              ir8[1] = (v1544_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1073_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1550_data = ir8[2];
              ir8[2] = (v1550_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1079_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1556_data = ir8[3];
              ir8[3] = (v1556_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1085_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1562_data = ir8[4];
              ir8[4] = (v1562_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1091_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1568_data = ir8[5];
              ir8[5] = (v1568_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1097_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1574_data = ir8[6];
              ir8[6] = (v1574_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1580_data = ir8[7];
              ir8[7] = (v1580_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1586_data = ir8[8];
              ir8[8] = (v1586_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1592_data = ir8[9];
              ir8[9] = (v1592_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1598_data = ir8[10];
              ir8[10] = (v1598_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1604_data = ir8[11];
              ir8[11] = (v1604_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1610_data = ir8[12];
              ir8[12] = (v1610_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (6)))));
              float v1612_data = r6[7];
              float v1616_data = ir8[0];
              ir8[0] = (v1616_data + (v1612_data * (sycl::select_from_group(item.get_sub_group(), v1067_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1622_data = ir8[1];
              ir8[1] = (v1622_data + (v1612_data * (sycl::select_from_group(item.get_sub_group(), v1073_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1628_data = ir8[2];
              ir8[2] = (v1628_data + (v1612_data * (sycl::select_from_group(item.get_sub_group(), v1079_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1634_data = ir8[3];
              ir8[3] = (v1634_data + (v1612_data * (sycl::select_from_group(item.get_sub_group(), v1085_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1640_data = ir8[4];
              ir8[4] = (v1640_data + (v1612_data * (sycl::select_from_group(item.get_sub_group(), v1091_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1646_data = ir8[5];
              ir8[5] = (v1646_data + (v1612_data * (sycl::select_from_group(item.get_sub_group(), v1097_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1652_data = ir8[6];
              ir8[6] = (v1652_data + (v1612_data * (sycl::select_from_group(item.get_sub_group(), v1103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1658_data = ir8[7];
              ir8[7] = (v1658_data + (v1612_data * (sycl::select_from_group(item.get_sub_group(), v1109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1664_data = ir8[8];
              ir8[8] = (v1664_data + (v1612_data * (sycl::select_from_group(item.get_sub_group(), v1115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1670_data = ir8[9];
              ir8[9] = (v1670_data + (v1612_data * (sycl::select_from_group(item.get_sub_group(), v1121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1676_data = ir8[10];
              ir8[10] = (v1676_data + (v1612_data * (sycl::select_from_group(item.get_sub_group(), v1127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1682_data = ir8[11];
              ir8[11] = (v1682_data + (v1612_data * (sycl::select_from_group(item.get_sub_group(), v1133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1688_data = ir8[12];
              ir8[12] = (v1688_data + (v1612_data * (sycl::select_from_group(item.get_sub_group(), v1139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (7)))));
              float v1690_data = r6[8];
              float v1694_data = ir8[0];
              ir8[0] = (v1694_data + (v1690_data * (sycl::select_from_group(item.get_sub_group(), v1067_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1700_data = ir8[1];
              ir8[1] = (v1700_data + (v1690_data * (sycl::select_from_group(item.get_sub_group(), v1073_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1706_data = ir8[2];
              ir8[2] = (v1706_data + (v1690_data * (sycl::select_from_group(item.get_sub_group(), v1079_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1712_data = ir8[3];
              ir8[3] = (v1712_data + (v1690_data * (sycl::select_from_group(item.get_sub_group(), v1085_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1718_data = ir8[4];
              ir8[4] = (v1718_data + (v1690_data * (sycl::select_from_group(item.get_sub_group(), v1091_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1724_data = ir8[5];
              ir8[5] = (v1724_data + (v1690_data * (sycl::select_from_group(item.get_sub_group(), v1097_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1730_data = ir8[6];
              ir8[6] = (v1730_data + (v1690_data * (sycl::select_from_group(item.get_sub_group(), v1103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1736_data = ir8[7];
              ir8[7] = (v1736_data + (v1690_data * (sycl::select_from_group(item.get_sub_group(), v1109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1742_data = ir8[8];
              ir8[8] = (v1742_data + (v1690_data * (sycl::select_from_group(item.get_sub_group(), v1115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1748_data = ir8[9];
              ir8[9] = (v1748_data + (v1690_data * (sycl::select_from_group(item.get_sub_group(), v1121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1754_data = ir8[10];
              ir8[10] = (v1754_data + (v1690_data * (sycl::select_from_group(item.get_sub_group(), v1127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1760_data = ir8[11];
              ir8[11] = (v1760_data + (v1690_data * (sycl::select_from_group(item.get_sub_group(), v1133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1766_data = ir8[12];
              ir8[12] = (v1766_data + (v1690_data * (sycl::select_from_group(item.get_sub_group(), v1139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (8)))));
              float v1768_data = r6[9];
              float v1772_data = ir8[0];
              ir8[0] = (v1772_data + (v1768_data * (sycl::select_from_group(item.get_sub_group(), v1067_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1778_data = ir8[1];
              ir8[1] = (v1778_data + (v1768_data * (sycl::select_from_group(item.get_sub_group(), v1073_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1784_data = ir8[2];
              ir8[2] = (v1784_data + (v1768_data * (sycl::select_from_group(item.get_sub_group(), v1079_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1790_data = ir8[3];
              ir8[3] = (v1790_data + (v1768_data * (sycl::select_from_group(item.get_sub_group(), v1085_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1796_data = ir8[4];
              ir8[4] = (v1796_data + (v1768_data * (sycl::select_from_group(item.get_sub_group(), v1091_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1802_data = ir8[5];
              ir8[5] = (v1802_data + (v1768_data * (sycl::select_from_group(item.get_sub_group(), v1097_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1808_data = ir8[6];
              ir8[6] = (v1808_data + (v1768_data * (sycl::select_from_group(item.get_sub_group(), v1103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1814_data = ir8[7];
              ir8[7] = (v1814_data + (v1768_data * (sycl::select_from_group(item.get_sub_group(), v1109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1820_data = ir8[8];
              ir8[8] = (v1820_data + (v1768_data * (sycl::select_from_group(item.get_sub_group(), v1115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1826_data = ir8[9];
              ir8[9] = (v1826_data + (v1768_data * (sycl::select_from_group(item.get_sub_group(), v1121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1832_data = ir8[10];
              ir8[10] = (v1832_data + (v1768_data * (sycl::select_from_group(item.get_sub_group(), v1127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1838_data = ir8[11];
              ir8[11] = (v1838_data + (v1768_data * (sycl::select_from_group(item.get_sub_group(), v1133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1844_data = ir8[12];
              ir8[12] = (v1844_data + (v1768_data * (sycl::select_from_group(item.get_sub_group(), v1139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (9)))));
              float v1846_data = r6[10];
              float v1850_data = ir8[0];
              ir8[0] = (v1850_data + (v1846_data * (sycl::select_from_group(item.get_sub_group(), v1067_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1856_data = ir8[1];
              ir8[1] = (v1856_data + (v1846_data * (sycl::select_from_group(item.get_sub_group(), v1073_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1862_data = ir8[2];
              ir8[2] = (v1862_data + (v1846_data * (sycl::select_from_group(item.get_sub_group(), v1079_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1868_data = ir8[3];
              ir8[3] = (v1868_data + (v1846_data * (sycl::select_from_group(item.get_sub_group(), v1085_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1874_data = ir8[4];
              ir8[4] = (v1874_data + (v1846_data * (sycl::select_from_group(item.get_sub_group(), v1091_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1880_data = ir8[5];
              ir8[5] = (v1880_data + (v1846_data * (sycl::select_from_group(item.get_sub_group(), v1097_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1886_data = ir8[6];
              ir8[6] = (v1886_data + (v1846_data * (sycl::select_from_group(item.get_sub_group(), v1103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1892_data = ir8[7];
              ir8[7] = (v1892_data + (v1846_data * (sycl::select_from_group(item.get_sub_group(), v1109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1898_data = ir8[8];
              ir8[8] = (v1898_data + (v1846_data * (sycl::select_from_group(item.get_sub_group(), v1115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1904_data = ir8[9];
              ir8[9] = (v1904_data + (v1846_data * (sycl::select_from_group(item.get_sub_group(), v1121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1910_data = ir8[10];
              ir8[10] = (v1910_data + (v1846_data * (sycl::select_from_group(item.get_sub_group(), v1127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1916_data = ir8[11];
              ir8[11] = (v1916_data + (v1846_data * (sycl::select_from_group(item.get_sub_group(), v1133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1922_data = ir8[12];
              ir8[12] = (v1922_data + (v1846_data * (sycl::select_from_group(item.get_sub_group(), v1139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (10)))));
              float v1924_data = r6[11];
              float v1928_data = ir8[0];
              ir8[0] = (v1928_data + (v1924_data * (sycl::select_from_group(item.get_sub_group(), v1067_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1934_data = ir8[1];
              ir8[1] = (v1934_data + (v1924_data * (sycl::select_from_group(item.get_sub_group(), v1073_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1940_data = ir8[2];
              ir8[2] = (v1940_data + (v1924_data * (sycl::select_from_group(item.get_sub_group(), v1079_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1946_data = ir8[3];
              ir8[3] = (v1946_data + (v1924_data * (sycl::select_from_group(item.get_sub_group(), v1085_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1952_data = ir8[4];
              ir8[4] = (v1952_data + (v1924_data * (sycl::select_from_group(item.get_sub_group(), v1091_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1958_data = ir8[5];
              ir8[5] = (v1958_data + (v1924_data * (sycl::select_from_group(item.get_sub_group(), v1097_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1964_data = ir8[6];
              ir8[6] = (v1964_data + (v1924_data * (sycl::select_from_group(item.get_sub_group(), v1103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1970_data = ir8[7];
              ir8[7] = (v1970_data + (v1924_data * (sycl::select_from_group(item.get_sub_group(), v1109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1976_data = ir8[8];
              ir8[8] = (v1976_data + (v1924_data * (sycl::select_from_group(item.get_sub_group(), v1115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1982_data = ir8[9];
              ir8[9] = (v1982_data + (v1924_data * (sycl::select_from_group(item.get_sub_group(), v1121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1988_data = ir8[10];
              ir8[10] = (v1988_data + (v1924_data * (sycl::select_from_group(item.get_sub_group(), v1127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v1994_data = ir8[11];
              ir8[11] = (v1994_data + (v1924_data * (sycl::select_from_group(item.get_sub_group(), v1133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v2000_data = ir8[12];
              ir8[12] = (v2000_data + (v1924_data * (sycl::select_from_group(item.get_sub_group(), v1139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (11)))));
              float v2002_data = r6[12];
              float v2006_data = ir8[0];
              ir8[0] = (v2006_data + (v2002_data * (sycl::select_from_group(item.get_sub_group(), v1067_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v2012_data = ir8[1];
              ir8[1] = (v2012_data + (v2002_data * (sycl::select_from_group(item.get_sub_group(), v1073_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v2018_data = ir8[2];
              ir8[2] = (v2018_data + (v2002_data * (sycl::select_from_group(item.get_sub_group(), v1079_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v2024_data = ir8[3];
              ir8[3] = (v2024_data + (v2002_data * (sycl::select_from_group(item.get_sub_group(), v1085_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v2030_data = ir8[4];
              ir8[4] = (v2030_data + (v2002_data * (sycl::select_from_group(item.get_sub_group(), v1091_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v2036_data = ir8[5];
              ir8[5] = (v2036_data + (v2002_data * (sycl::select_from_group(item.get_sub_group(), v1097_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v2042_data = ir8[6];
              ir8[6] = (v2042_data + (v2002_data * (sycl::select_from_group(item.get_sub_group(), v1103_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v2048_data = ir8[7];
              ir8[7] = (v2048_data + (v2002_data * (sycl::select_from_group(item.get_sub_group(), v1109_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v2054_data = ir8[8];
              ir8[8] = (v2054_data + (v2002_data * (sycl::select_from_group(item.get_sub_group(), v1115_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v2060_data = ir8[9];
              ir8[9] = (v2060_data + (v2002_data * (sycl::select_from_group(item.get_sub_group(), v1121_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v2066_data = ir8[10];
              ir8[10] = (v2066_data + (v2002_data * (sycl::select_from_group(item.get_sub_group(), v1127_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v2072_data = ir8[11];
              ir8[11] = (v2072_data + (v2002_data * (sycl::select_from_group(item.get_sub_group(), v1133_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              float v2078_data = ir8[12];
              ir8[12] = (v2078_data + (v2002_data * (sycl::select_from_group(item.get_sub_group(), v1139_data, (item.get_sub_group().get_local_linear_id() / 32) * 32 + (12)))));
              #pragma unroll
              for (int32_t v2080_n0 = 0; v2080_n0 < 1; ++v2080_n0) {
                #pragma unroll
                for (int32_t v2081_n1 = 0; v2081_n1 < 13; ++v2081_n1) {
                  int32_t v2082_a = v2080_n0 + v2081_n1;
                  float v2083_data = ir8[v2082_a];
                  r8[v2082_a] = v2083_data;
                }
              }
              // glb_m3 = store{r>g}(r8);
              #pragma unroll
              for (int32_t v2084_i0 = 0; v2084_i0 < 1; ++v2084_i0) {
                int32_t v2089_lead = v17_lead + (v2084_i0 * 32);
                #pragma unroll
                for (int32_t v2085_i1 = 0; v2085_i1 < 13; ++v2085_i1) {
                  float v2087_data = r8[(v2084_i0 + v2085_i1)];
                  glb_m3[(v2089_lead + (v2085_i1 * 32))] = v2087_data;
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

