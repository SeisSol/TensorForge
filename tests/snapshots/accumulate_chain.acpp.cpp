// === base name ===
kernel_11c8a6fab3dc0736

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_11c8a6fab3dc0736 = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_11c8a6fab3dc0736(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_11c8a6fab3dc0736(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_11c8a6fab3dc0736(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (16, 16, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 16;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_11c8a6fab3dc0736(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_11c8a6fab3dc0736(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_11c8a6fab3dc0736(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, m6, m6_extraOffset, m7, m7_extraOffset, m8, m8_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_11c8a6fab3dc0736(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
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
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"D","bbox":[[0,0],[12,8]],"name":"m0","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A0","bbox":[[0,0],[12,12]],"name":"m1","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B0","bbox":[[0,0],[12,8]],"name":"m2","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A1","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B1","bbox":[[0,0],[12,8]],"name":"m4","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A2","bbox":[[0,0],[12,12]],"name":"m5","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B2","bbox":[[0,0],[12,8]],"name":"m6","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A3","bbox":[[0,0],[12,12]],"name":"m7","ordered":false,"parts":1,"shape":[12,12],"variant":false},{"addressing":"strided","alias":"B3","bbox":[[0,0],[12,8]],"name":"m8","ordered":false,"parts":1,"shape":[12,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m4","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m5","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m6","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m7","offset":[0,0],"shape":[12,12]},{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m8","offset":[0,0],"shape":[12,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t v3_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v3_batchId0 < numElements0; v3_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v4_ahead1 = v3_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v3_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v3_batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v3_batchId0 * 96 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v3_batchId0 * 144 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v3_batchId0 * 96 + 0 + m4_extraOffset];
              const float *const __restrict__ glb_m5 = &m5[v3_batchId0 * 144 + 0 + m5_extraOffset];
              const float *const __restrict__ glb_m6 = &m6[v3_batchId0 * 96 + 0 + m6_extraOffset];
              const float *const __restrict__ glb_m7 = &m7[v3_batchId0 * 144 + 0 + m7_extraOffset];
              const float *const __restrict__ glb_m8 = &m8[v3_batchId0 * 96 + 0 + m8_extraOffset];
              float r0[12]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v23_lead = item.get_local_id(2) % 16;
              bool v24_g = v23_lead < 12;
              if (v24_g) {
                #pragma unroll
                for (int32_t v25_i1 = 0; v25_i1 < 12; ++v25_i1) {
                  float v30_data = glb_m1[(v23_lead + (v25_i1 * 12))];
                  r0[v25_i1] = v30_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              if (v24_g) {
                #pragma unroll
                for (int32_t v33_i1 = 0; v33_i1 < 8; ++v33_i1) {
                  float v38_data = glb_m2[(v23_lead + (v33_i1 * 12))];
                  r1[v33_i1] = v38_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              float r3[12]{};
              // r3 = load{g>r}(glb_m3);
              if (v24_g) {
                #pragma unroll
                for (int32_t v41_i1 = 0; v41_i1 < 12; ++v41_i1) {
                  float v46_data = glb_m3[(v23_lead + (v41_i1 * 12))];
                  r3[v41_i1] = v46_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir2[8]{};
              float v50_data = r0[0];
              float v51_data = r1[0];
              float v54_data = ir2[0];
              ir2[0] = (v54_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v57_data = r1[1];
              float v60_data = ir2[1];
              ir2[1] = (v60_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v63_data = r1[2];
              float v66_data = ir2[2];
              ir2[2] = (v66_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v69_data = r1[3];
              float v72_data = ir2[3];
              ir2[3] = (v72_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v75_data = r1[4];
              float v78_data = ir2[4];
              ir2[4] = (v78_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v81_data = r1[5];
              float v84_data = ir2[5];
              ir2[5] = (v84_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v87_data = r1[6];
              float v90_data = ir2[6];
              ir2[6] = (v90_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v93_data = r1[7];
              float v96_data = ir2[7];
              ir2[7] = (v96_data + (v50_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v98_data = r0[1];
              float v102_data = ir2[0];
              ir2[0] = (v102_data + (v98_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v108_data = ir2[1];
              ir2[1] = (v108_data + (v98_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v114_data = ir2[2];
              ir2[2] = (v114_data + (v98_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v120_data = ir2[3];
              ir2[3] = (v120_data + (v98_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v126_data = ir2[4];
              ir2[4] = (v126_data + (v98_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v132_data = ir2[5];
              ir2[5] = (v132_data + (v98_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v138_data = ir2[6];
              ir2[6] = (v138_data + (v98_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v144_data = ir2[7];
              ir2[7] = (v144_data + (v98_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v146_data = r0[2];
              float v150_data = ir2[0];
              ir2[0] = (v150_data + (v146_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v156_data = ir2[1];
              ir2[1] = (v156_data + (v146_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v162_data = ir2[2];
              ir2[2] = (v162_data + (v146_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v168_data = ir2[3];
              ir2[3] = (v168_data + (v146_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v174_data = ir2[4];
              ir2[4] = (v174_data + (v146_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v180_data = ir2[5];
              ir2[5] = (v180_data + (v146_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v186_data = ir2[6];
              ir2[6] = (v186_data + (v146_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v192_data = ir2[7];
              ir2[7] = (v192_data + (v146_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v194_data = r0[3];
              float v198_data = ir2[0];
              ir2[0] = (v198_data + (v194_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v204_data = ir2[1];
              ir2[1] = (v204_data + (v194_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v210_data = ir2[2];
              ir2[2] = (v210_data + (v194_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v216_data = ir2[3];
              ir2[3] = (v216_data + (v194_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v222_data = ir2[4];
              ir2[4] = (v222_data + (v194_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v228_data = ir2[5];
              ir2[5] = (v228_data + (v194_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v234_data = ir2[6];
              ir2[6] = (v234_data + (v194_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v240_data = ir2[7];
              ir2[7] = (v240_data + (v194_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v242_data = r0[4];
              float v246_data = ir2[0];
              ir2[0] = (v246_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v252_data = ir2[1];
              ir2[1] = (v252_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v258_data = ir2[2];
              ir2[2] = (v258_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v264_data = ir2[3];
              ir2[3] = (v264_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v270_data = ir2[4];
              ir2[4] = (v270_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v276_data = ir2[5];
              ir2[5] = (v276_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v282_data = ir2[6];
              ir2[6] = (v282_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v288_data = ir2[7];
              ir2[7] = (v288_data + (v242_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v290_data = r0[5];
              float v294_data = ir2[0];
              ir2[0] = (v294_data + (v290_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v300_data = ir2[1];
              ir2[1] = (v300_data + (v290_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v306_data = ir2[2];
              ir2[2] = (v306_data + (v290_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v312_data = ir2[3];
              ir2[3] = (v312_data + (v290_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v318_data = ir2[4];
              ir2[4] = (v318_data + (v290_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v324_data = ir2[5];
              ir2[5] = (v324_data + (v290_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v330_data = ir2[6];
              ir2[6] = (v330_data + (v290_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v336_data = ir2[7];
              ir2[7] = (v336_data + (v290_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v338_data = r0[6];
              float v342_data = ir2[0];
              ir2[0] = (v342_data + (v338_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v348_data = ir2[1];
              ir2[1] = (v348_data + (v338_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v354_data = ir2[2];
              ir2[2] = (v354_data + (v338_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v360_data = ir2[3];
              ir2[3] = (v360_data + (v338_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v366_data = ir2[4];
              ir2[4] = (v366_data + (v338_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v372_data = ir2[5];
              ir2[5] = (v372_data + (v338_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v378_data = ir2[6];
              ir2[6] = (v378_data + (v338_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v384_data = ir2[7];
              ir2[7] = (v384_data + (v338_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v386_data = r0[7];
              float v390_data = ir2[0];
              ir2[0] = (v390_data + (v386_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v396_data = ir2[1];
              ir2[1] = (v396_data + (v386_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v402_data = ir2[2];
              ir2[2] = (v402_data + (v386_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v408_data = ir2[3];
              ir2[3] = (v408_data + (v386_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v414_data = ir2[4];
              ir2[4] = (v414_data + (v386_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v420_data = ir2[5];
              ir2[5] = (v420_data + (v386_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v426_data = ir2[6];
              ir2[6] = (v426_data + (v386_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v432_data = ir2[7];
              ir2[7] = (v432_data + (v386_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v434_data = r0[8];
              float v438_data = ir2[0];
              ir2[0] = (v438_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v444_data = ir2[1];
              ir2[1] = (v444_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v450_data = ir2[2];
              ir2[2] = (v450_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v456_data = ir2[3];
              ir2[3] = (v456_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v462_data = ir2[4];
              ir2[4] = (v462_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v468_data = ir2[5];
              ir2[5] = (v468_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v474_data = ir2[6];
              ir2[6] = (v474_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v480_data = ir2[7];
              ir2[7] = (v480_data + (v434_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v482_data = r0[9];
              float v486_data = ir2[0];
              ir2[0] = (v486_data + (v482_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v492_data = ir2[1];
              ir2[1] = (v492_data + (v482_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v498_data = ir2[2];
              ir2[2] = (v498_data + (v482_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v504_data = ir2[3];
              ir2[3] = (v504_data + (v482_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v510_data = ir2[4];
              ir2[4] = (v510_data + (v482_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v516_data = ir2[5];
              ir2[5] = (v516_data + (v482_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v522_data = ir2[6];
              ir2[6] = (v522_data + (v482_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v528_data = ir2[7];
              ir2[7] = (v528_data + (v482_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v530_data = r0[10];
              float v534_data = ir2[0];
              ir2[0] = (v534_data + (v530_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v540_data = ir2[1];
              ir2[1] = (v540_data + (v530_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v546_data = ir2[2];
              ir2[2] = (v546_data + (v530_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v552_data = ir2[3];
              ir2[3] = (v552_data + (v530_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v558_data = ir2[4];
              ir2[4] = (v558_data + (v530_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v564_data = ir2[5];
              ir2[5] = (v564_data + (v530_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v570_data = ir2[6];
              ir2[6] = (v570_data + (v530_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v576_data = ir2[7];
              ir2[7] = (v576_data + (v530_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v578_data = r0[11];
              float v582_data = ir2[0];
              ir2[0] = (v582_data + (v578_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v588_data = ir2[1];
              ir2[1] = (v588_data + (v578_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v594_data = ir2[2];
              ir2[2] = (v594_data + (v578_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v600_data = ir2[3];
              ir2[3] = (v600_data + (v578_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v606_data = ir2[4];
              ir2[4] = (v606_data + (v578_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v612_data = ir2[5];
              ir2[5] = (v612_data + (v578_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v618_data = ir2[6];
              ir2[6] = (v618_data + (v578_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v624_data = ir2[7];
              ir2[7] = (v624_data + (v578_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              if (v24_g) {
                #pragma unroll
                for (int32_t v626_n1 = 0; v626_n1 < 8; ++v626_n1) {
                  float v628_data = ir2[v626_n1];
                  r2[v626_n1] = v628_data;
                }
              }
              float r4[8]{};
              // r4 = load{g>r}(glb_m4);
              if (v24_g) {
                #pragma unroll
                for (int32_t v630_i1 = 0; v630_i1 < 8; ++v630_i1) {
                  float v635_data = glb_m4[(v23_lead + (v630_i1 * 12))];
                  r4[v630_i1] = v635_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m3););
              float r6[12]{};
              // r6 = load{g>r}(glb_m5);
              if (v24_g) {
                #pragma unroll
                for (int32_t v638_i1 = 0; v638_i1 < 12; ++v638_i1) {
                  float v643_data = glb_m5[(v23_lead + (v638_i1 * 12))];
                  r6[v638_i1] = v643_data;
                }
              }
              // wait(r4 = load{g>r}(glb_m4););
              float r5[8]{};
              // r5 = +(r3 * r4) + name: r2, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir5[8]{};
              float v647_data = r3[0];
              float v648_data = r4[0];
              float v651_data = ir5[0];
              ir5[0] = (v651_data + (v647_data * (sycl::select_from_group(item.get_sub_group(), v648_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v654_data = r4[1];
              float v657_data = ir5[1];
              ir5[1] = (v657_data + (v647_data * (sycl::select_from_group(item.get_sub_group(), v654_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v660_data = r4[2];
              float v663_data = ir5[2];
              ir5[2] = (v663_data + (v647_data * (sycl::select_from_group(item.get_sub_group(), v660_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v666_data = r4[3];
              float v669_data = ir5[3];
              ir5[3] = (v669_data + (v647_data * (sycl::select_from_group(item.get_sub_group(), v666_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v672_data = r4[4];
              float v675_data = ir5[4];
              ir5[4] = (v675_data + (v647_data * (sycl::select_from_group(item.get_sub_group(), v672_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v678_data = r4[5];
              float v681_data = ir5[5];
              ir5[5] = (v681_data + (v647_data * (sycl::select_from_group(item.get_sub_group(), v678_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v684_data = r4[6];
              float v687_data = ir5[6];
              ir5[6] = (v687_data + (v647_data * (sycl::select_from_group(item.get_sub_group(), v684_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v690_data = r4[7];
              float v693_data = ir5[7];
              ir5[7] = (v693_data + (v647_data * (sycl::select_from_group(item.get_sub_group(), v690_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v695_data = r3[1];
              float v699_data = ir5[0];
              ir5[0] = (v699_data + (v695_data * (sycl::select_from_group(item.get_sub_group(), v648_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v705_data = ir5[1];
              ir5[1] = (v705_data + (v695_data * (sycl::select_from_group(item.get_sub_group(), v654_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v711_data = ir5[2];
              ir5[2] = (v711_data + (v695_data * (sycl::select_from_group(item.get_sub_group(), v660_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v717_data = ir5[3];
              ir5[3] = (v717_data + (v695_data * (sycl::select_from_group(item.get_sub_group(), v666_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v723_data = ir5[4];
              ir5[4] = (v723_data + (v695_data * (sycl::select_from_group(item.get_sub_group(), v672_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v729_data = ir5[5];
              ir5[5] = (v729_data + (v695_data * (sycl::select_from_group(item.get_sub_group(), v678_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v735_data = ir5[6];
              ir5[6] = (v735_data + (v695_data * (sycl::select_from_group(item.get_sub_group(), v684_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v741_data = ir5[7];
              ir5[7] = (v741_data + (v695_data * (sycl::select_from_group(item.get_sub_group(), v690_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v743_data = r3[2];
              float v747_data = ir5[0];
              ir5[0] = (v747_data + (v743_data * (sycl::select_from_group(item.get_sub_group(), v648_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v753_data = ir5[1];
              ir5[1] = (v753_data + (v743_data * (sycl::select_from_group(item.get_sub_group(), v654_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v759_data = ir5[2];
              ir5[2] = (v759_data + (v743_data * (sycl::select_from_group(item.get_sub_group(), v660_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v765_data = ir5[3];
              ir5[3] = (v765_data + (v743_data * (sycl::select_from_group(item.get_sub_group(), v666_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v771_data = ir5[4];
              ir5[4] = (v771_data + (v743_data * (sycl::select_from_group(item.get_sub_group(), v672_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v777_data = ir5[5];
              ir5[5] = (v777_data + (v743_data * (sycl::select_from_group(item.get_sub_group(), v678_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v783_data = ir5[6];
              ir5[6] = (v783_data + (v743_data * (sycl::select_from_group(item.get_sub_group(), v684_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v789_data = ir5[7];
              ir5[7] = (v789_data + (v743_data * (sycl::select_from_group(item.get_sub_group(), v690_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v791_data = r3[3];
              float v795_data = ir5[0];
              ir5[0] = (v795_data + (v791_data * (sycl::select_from_group(item.get_sub_group(), v648_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v801_data = ir5[1];
              ir5[1] = (v801_data + (v791_data * (sycl::select_from_group(item.get_sub_group(), v654_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v807_data = ir5[2];
              ir5[2] = (v807_data + (v791_data * (sycl::select_from_group(item.get_sub_group(), v660_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v813_data = ir5[3];
              ir5[3] = (v813_data + (v791_data * (sycl::select_from_group(item.get_sub_group(), v666_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v819_data = ir5[4];
              ir5[4] = (v819_data + (v791_data * (sycl::select_from_group(item.get_sub_group(), v672_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v825_data = ir5[5];
              ir5[5] = (v825_data + (v791_data * (sycl::select_from_group(item.get_sub_group(), v678_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v831_data = ir5[6];
              ir5[6] = (v831_data + (v791_data * (sycl::select_from_group(item.get_sub_group(), v684_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v837_data = ir5[7];
              ir5[7] = (v837_data + (v791_data * (sycl::select_from_group(item.get_sub_group(), v690_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v839_data = r3[4];
              float v843_data = ir5[0];
              ir5[0] = (v843_data + (v839_data * (sycl::select_from_group(item.get_sub_group(), v648_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v849_data = ir5[1];
              ir5[1] = (v849_data + (v839_data * (sycl::select_from_group(item.get_sub_group(), v654_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v855_data = ir5[2];
              ir5[2] = (v855_data + (v839_data * (sycl::select_from_group(item.get_sub_group(), v660_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v861_data = ir5[3];
              ir5[3] = (v861_data + (v839_data * (sycl::select_from_group(item.get_sub_group(), v666_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v867_data = ir5[4];
              ir5[4] = (v867_data + (v839_data * (sycl::select_from_group(item.get_sub_group(), v672_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v873_data = ir5[5];
              ir5[5] = (v873_data + (v839_data * (sycl::select_from_group(item.get_sub_group(), v678_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v879_data = ir5[6];
              ir5[6] = (v879_data + (v839_data * (sycl::select_from_group(item.get_sub_group(), v684_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v885_data = ir5[7];
              ir5[7] = (v885_data + (v839_data * (sycl::select_from_group(item.get_sub_group(), v690_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v887_data = r3[5];
              float v891_data = ir5[0];
              ir5[0] = (v891_data + (v887_data * (sycl::select_from_group(item.get_sub_group(), v648_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v897_data = ir5[1];
              ir5[1] = (v897_data + (v887_data * (sycl::select_from_group(item.get_sub_group(), v654_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v903_data = ir5[2];
              ir5[2] = (v903_data + (v887_data * (sycl::select_from_group(item.get_sub_group(), v660_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v909_data = ir5[3];
              ir5[3] = (v909_data + (v887_data * (sycl::select_from_group(item.get_sub_group(), v666_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v915_data = ir5[4];
              ir5[4] = (v915_data + (v887_data * (sycl::select_from_group(item.get_sub_group(), v672_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v921_data = ir5[5];
              ir5[5] = (v921_data + (v887_data * (sycl::select_from_group(item.get_sub_group(), v678_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v927_data = ir5[6];
              ir5[6] = (v927_data + (v887_data * (sycl::select_from_group(item.get_sub_group(), v684_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v933_data = ir5[7];
              ir5[7] = (v933_data + (v887_data * (sycl::select_from_group(item.get_sub_group(), v690_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v935_data = r3[6];
              float v939_data = ir5[0];
              ir5[0] = (v939_data + (v935_data * (sycl::select_from_group(item.get_sub_group(), v648_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v945_data = ir5[1];
              ir5[1] = (v945_data + (v935_data * (sycl::select_from_group(item.get_sub_group(), v654_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v951_data = ir5[2];
              ir5[2] = (v951_data + (v935_data * (sycl::select_from_group(item.get_sub_group(), v660_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v957_data = ir5[3];
              ir5[3] = (v957_data + (v935_data * (sycl::select_from_group(item.get_sub_group(), v666_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v963_data = ir5[4];
              ir5[4] = (v963_data + (v935_data * (sycl::select_from_group(item.get_sub_group(), v672_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v969_data = ir5[5];
              ir5[5] = (v969_data + (v935_data * (sycl::select_from_group(item.get_sub_group(), v678_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v975_data = ir5[6];
              ir5[6] = (v975_data + (v935_data * (sycl::select_from_group(item.get_sub_group(), v684_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v981_data = ir5[7];
              ir5[7] = (v981_data + (v935_data * (sycl::select_from_group(item.get_sub_group(), v690_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v983_data = r3[7];
              float v987_data = ir5[0];
              ir5[0] = (v987_data + (v983_data * (sycl::select_from_group(item.get_sub_group(), v648_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v993_data = ir5[1];
              ir5[1] = (v993_data + (v983_data * (sycl::select_from_group(item.get_sub_group(), v654_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v999_data = ir5[2];
              ir5[2] = (v999_data + (v983_data * (sycl::select_from_group(item.get_sub_group(), v660_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1005_data = ir5[3];
              ir5[3] = (v1005_data + (v983_data * (sycl::select_from_group(item.get_sub_group(), v666_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1011_data = ir5[4];
              ir5[4] = (v1011_data + (v983_data * (sycl::select_from_group(item.get_sub_group(), v672_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1017_data = ir5[5];
              ir5[5] = (v1017_data + (v983_data * (sycl::select_from_group(item.get_sub_group(), v678_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1023_data = ir5[6];
              ir5[6] = (v1023_data + (v983_data * (sycl::select_from_group(item.get_sub_group(), v684_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1029_data = ir5[7];
              ir5[7] = (v1029_data + (v983_data * (sycl::select_from_group(item.get_sub_group(), v690_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1031_data = r3[8];
              float v1035_data = ir5[0];
              ir5[0] = (v1035_data + (v1031_data * (sycl::select_from_group(item.get_sub_group(), v648_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1041_data = ir5[1];
              ir5[1] = (v1041_data + (v1031_data * (sycl::select_from_group(item.get_sub_group(), v654_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1047_data = ir5[2];
              ir5[2] = (v1047_data + (v1031_data * (sycl::select_from_group(item.get_sub_group(), v660_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1053_data = ir5[3];
              ir5[3] = (v1053_data + (v1031_data * (sycl::select_from_group(item.get_sub_group(), v666_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1059_data = ir5[4];
              ir5[4] = (v1059_data + (v1031_data * (sycl::select_from_group(item.get_sub_group(), v672_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1065_data = ir5[5];
              ir5[5] = (v1065_data + (v1031_data * (sycl::select_from_group(item.get_sub_group(), v678_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1071_data = ir5[6];
              ir5[6] = (v1071_data + (v1031_data * (sycl::select_from_group(item.get_sub_group(), v684_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1077_data = ir5[7];
              ir5[7] = (v1077_data + (v1031_data * (sycl::select_from_group(item.get_sub_group(), v690_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1079_data = r3[9];
              float v1083_data = ir5[0];
              ir5[0] = (v1083_data + (v1079_data * (sycl::select_from_group(item.get_sub_group(), v648_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1089_data = ir5[1];
              ir5[1] = (v1089_data + (v1079_data * (sycl::select_from_group(item.get_sub_group(), v654_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1095_data = ir5[2];
              ir5[2] = (v1095_data + (v1079_data * (sycl::select_from_group(item.get_sub_group(), v660_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1101_data = ir5[3];
              ir5[3] = (v1101_data + (v1079_data * (sycl::select_from_group(item.get_sub_group(), v666_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1107_data = ir5[4];
              ir5[4] = (v1107_data + (v1079_data * (sycl::select_from_group(item.get_sub_group(), v672_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1113_data = ir5[5];
              ir5[5] = (v1113_data + (v1079_data * (sycl::select_from_group(item.get_sub_group(), v678_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1119_data = ir5[6];
              ir5[6] = (v1119_data + (v1079_data * (sycl::select_from_group(item.get_sub_group(), v684_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1125_data = ir5[7];
              ir5[7] = (v1125_data + (v1079_data * (sycl::select_from_group(item.get_sub_group(), v690_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1127_data = r3[10];
              float v1131_data = ir5[0];
              ir5[0] = (v1131_data + (v1127_data * (sycl::select_from_group(item.get_sub_group(), v648_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1137_data = ir5[1];
              ir5[1] = (v1137_data + (v1127_data * (sycl::select_from_group(item.get_sub_group(), v654_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1143_data = ir5[2];
              ir5[2] = (v1143_data + (v1127_data * (sycl::select_from_group(item.get_sub_group(), v660_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1149_data = ir5[3];
              ir5[3] = (v1149_data + (v1127_data * (sycl::select_from_group(item.get_sub_group(), v666_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1155_data = ir5[4];
              ir5[4] = (v1155_data + (v1127_data * (sycl::select_from_group(item.get_sub_group(), v672_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1161_data = ir5[5];
              ir5[5] = (v1161_data + (v1127_data * (sycl::select_from_group(item.get_sub_group(), v678_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1167_data = ir5[6];
              ir5[6] = (v1167_data + (v1127_data * (sycl::select_from_group(item.get_sub_group(), v684_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1173_data = ir5[7];
              ir5[7] = (v1173_data + (v1127_data * (sycl::select_from_group(item.get_sub_group(), v690_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1175_data = r3[11];
              float v1179_data = ir5[0];
              ir5[0] = (v1179_data + (v1175_data * (sycl::select_from_group(item.get_sub_group(), v648_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1185_data = ir5[1];
              ir5[1] = (v1185_data + (v1175_data * (sycl::select_from_group(item.get_sub_group(), v654_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1191_data = ir5[2];
              ir5[2] = (v1191_data + (v1175_data * (sycl::select_from_group(item.get_sub_group(), v660_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1197_data = ir5[3];
              ir5[3] = (v1197_data + (v1175_data * (sycl::select_from_group(item.get_sub_group(), v666_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1203_data = ir5[4];
              ir5[4] = (v1203_data + (v1175_data * (sycl::select_from_group(item.get_sub_group(), v672_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1209_data = ir5[5];
              ir5[5] = (v1209_data + (v1175_data * (sycl::select_from_group(item.get_sub_group(), v678_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1215_data = ir5[6];
              ir5[6] = (v1215_data + (v1175_data * (sycl::select_from_group(item.get_sub_group(), v684_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1221_data = ir5[7];
              ir5[7] = (v1221_data + (v1175_data * (sycl::select_from_group(item.get_sub_group(), v690_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              if (v24_g) {
                #pragma unroll
                for (int32_t v1223_n1 = 0; v1223_n1 < 8; ++v1223_n1) {
                  float v1225_data = ir5[v1223_n1];
                  float v1226_data = r2[v1223_n1];
                  r5[v1223_n1] = (v1226_data + v1225_data);
                }
              }
              float r7[8]{};
              // r7 = load{g>r}(glb_m6);
              if (v24_g) {
                #pragma unroll
                for (int32_t v1229_i1 = 0; v1229_i1 < 8; ++v1229_i1) {
                  float v1234_data = glb_m6[(v23_lead + (v1229_i1 * 12))];
                  r7[v1229_i1] = v1234_data;
                }
              }
              // wait(r6 = load{g>r}(glb_m5););
              float r9[12]{};
              // r9 = load{g>r}(glb_m7);
              if (v24_g) {
                #pragma unroll
                for (int32_t v1237_i1 = 0; v1237_i1 < 12; ++v1237_i1) {
                  float v1242_data = glb_m7[(v23_lead + (v1237_i1 * 12))];
                  r9[v1237_i1] = v1242_data;
                }
              }
              // wait(r7 = load{g>r}(glb_m6););
              float r8[8]{};
              // r8 = +(r6 * r7) + name: r5, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir8[8]{};
              float v1246_data = r6[0];
              float v1247_data = r7[0];
              float v1250_data = ir8[0];
              ir8[0] = (v1250_data + (v1246_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1253_data = r7[1];
              float v1256_data = ir8[1];
              ir8[1] = (v1256_data + (v1246_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1259_data = r7[2];
              float v1262_data = ir8[2];
              ir8[2] = (v1262_data + (v1246_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1265_data = r7[3];
              float v1268_data = ir8[3];
              ir8[3] = (v1268_data + (v1246_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1271_data = r7[4];
              float v1274_data = ir8[4];
              ir8[4] = (v1274_data + (v1246_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1277_data = r7[5];
              float v1280_data = ir8[5];
              ir8[5] = (v1280_data + (v1246_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1283_data = r7[6];
              float v1286_data = ir8[6];
              ir8[6] = (v1286_data + (v1246_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1289_data = r7[7];
              float v1292_data = ir8[7];
              ir8[7] = (v1292_data + (v1246_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1294_data = r6[1];
              float v1298_data = ir8[0];
              ir8[0] = (v1298_data + (v1294_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1304_data = ir8[1];
              ir8[1] = (v1304_data + (v1294_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1310_data = ir8[2];
              ir8[2] = (v1310_data + (v1294_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1316_data = ir8[3];
              ir8[3] = (v1316_data + (v1294_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1322_data = ir8[4];
              ir8[4] = (v1322_data + (v1294_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1328_data = ir8[5];
              ir8[5] = (v1328_data + (v1294_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1334_data = ir8[6];
              ir8[6] = (v1334_data + (v1294_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1340_data = ir8[7];
              ir8[7] = (v1340_data + (v1294_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1342_data = r6[2];
              float v1346_data = ir8[0];
              ir8[0] = (v1346_data + (v1342_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1352_data = ir8[1];
              ir8[1] = (v1352_data + (v1342_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1358_data = ir8[2];
              ir8[2] = (v1358_data + (v1342_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1364_data = ir8[3];
              ir8[3] = (v1364_data + (v1342_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1370_data = ir8[4];
              ir8[4] = (v1370_data + (v1342_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1376_data = ir8[5];
              ir8[5] = (v1376_data + (v1342_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1382_data = ir8[6];
              ir8[6] = (v1382_data + (v1342_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1388_data = ir8[7];
              ir8[7] = (v1388_data + (v1342_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1390_data = r6[3];
              float v1394_data = ir8[0];
              ir8[0] = (v1394_data + (v1390_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1400_data = ir8[1];
              ir8[1] = (v1400_data + (v1390_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1406_data = ir8[2];
              ir8[2] = (v1406_data + (v1390_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1412_data = ir8[3];
              ir8[3] = (v1412_data + (v1390_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1418_data = ir8[4];
              ir8[4] = (v1418_data + (v1390_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1424_data = ir8[5];
              ir8[5] = (v1424_data + (v1390_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1430_data = ir8[6];
              ir8[6] = (v1430_data + (v1390_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1436_data = ir8[7];
              ir8[7] = (v1436_data + (v1390_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1438_data = r6[4];
              float v1442_data = ir8[0];
              ir8[0] = (v1442_data + (v1438_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1448_data = ir8[1];
              ir8[1] = (v1448_data + (v1438_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1454_data = ir8[2];
              ir8[2] = (v1454_data + (v1438_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1460_data = ir8[3];
              ir8[3] = (v1460_data + (v1438_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1466_data = ir8[4];
              ir8[4] = (v1466_data + (v1438_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1472_data = ir8[5];
              ir8[5] = (v1472_data + (v1438_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1478_data = ir8[6];
              ir8[6] = (v1478_data + (v1438_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1484_data = ir8[7];
              ir8[7] = (v1484_data + (v1438_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1486_data = r6[5];
              float v1490_data = ir8[0];
              ir8[0] = (v1490_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1496_data = ir8[1];
              ir8[1] = (v1496_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1502_data = ir8[2];
              ir8[2] = (v1502_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1508_data = ir8[3];
              ir8[3] = (v1508_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1514_data = ir8[4];
              ir8[4] = (v1514_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1520_data = ir8[5];
              ir8[5] = (v1520_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1526_data = ir8[6];
              ir8[6] = (v1526_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1532_data = ir8[7];
              ir8[7] = (v1532_data + (v1486_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1534_data = r6[6];
              float v1538_data = ir8[0];
              ir8[0] = (v1538_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1544_data = ir8[1];
              ir8[1] = (v1544_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1550_data = ir8[2];
              ir8[2] = (v1550_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1556_data = ir8[3];
              ir8[3] = (v1556_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1562_data = ir8[4];
              ir8[4] = (v1562_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1568_data = ir8[5];
              ir8[5] = (v1568_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1574_data = ir8[6];
              ir8[6] = (v1574_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1580_data = ir8[7];
              ir8[7] = (v1580_data + (v1534_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1582_data = r6[7];
              float v1586_data = ir8[0];
              ir8[0] = (v1586_data + (v1582_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1592_data = ir8[1];
              ir8[1] = (v1592_data + (v1582_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1598_data = ir8[2];
              ir8[2] = (v1598_data + (v1582_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1604_data = ir8[3];
              ir8[3] = (v1604_data + (v1582_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1610_data = ir8[4];
              ir8[4] = (v1610_data + (v1582_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1616_data = ir8[5];
              ir8[5] = (v1616_data + (v1582_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1622_data = ir8[6];
              ir8[6] = (v1622_data + (v1582_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1628_data = ir8[7];
              ir8[7] = (v1628_data + (v1582_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1630_data = r6[8];
              float v1634_data = ir8[0];
              ir8[0] = (v1634_data + (v1630_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1640_data = ir8[1];
              ir8[1] = (v1640_data + (v1630_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1646_data = ir8[2];
              ir8[2] = (v1646_data + (v1630_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1652_data = ir8[3];
              ir8[3] = (v1652_data + (v1630_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1658_data = ir8[4];
              ir8[4] = (v1658_data + (v1630_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1664_data = ir8[5];
              ir8[5] = (v1664_data + (v1630_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1670_data = ir8[6];
              ir8[6] = (v1670_data + (v1630_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1676_data = ir8[7];
              ir8[7] = (v1676_data + (v1630_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1678_data = r6[9];
              float v1682_data = ir8[0];
              ir8[0] = (v1682_data + (v1678_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1688_data = ir8[1];
              ir8[1] = (v1688_data + (v1678_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1694_data = ir8[2];
              ir8[2] = (v1694_data + (v1678_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1700_data = ir8[3];
              ir8[3] = (v1700_data + (v1678_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1706_data = ir8[4];
              ir8[4] = (v1706_data + (v1678_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1712_data = ir8[5];
              ir8[5] = (v1712_data + (v1678_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1718_data = ir8[6];
              ir8[6] = (v1718_data + (v1678_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1724_data = ir8[7];
              ir8[7] = (v1724_data + (v1678_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1726_data = r6[10];
              float v1730_data = ir8[0];
              ir8[0] = (v1730_data + (v1726_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1736_data = ir8[1];
              ir8[1] = (v1736_data + (v1726_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1742_data = ir8[2];
              ir8[2] = (v1742_data + (v1726_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1748_data = ir8[3];
              ir8[3] = (v1748_data + (v1726_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1754_data = ir8[4];
              ir8[4] = (v1754_data + (v1726_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1760_data = ir8[5];
              ir8[5] = (v1760_data + (v1726_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1766_data = ir8[6];
              ir8[6] = (v1766_data + (v1726_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1772_data = ir8[7];
              ir8[7] = (v1772_data + (v1726_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1774_data = r6[11];
              float v1778_data = ir8[0];
              ir8[0] = (v1778_data + (v1774_data * (sycl::select_from_group(item.get_sub_group(), v1247_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1784_data = ir8[1];
              ir8[1] = (v1784_data + (v1774_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1790_data = ir8[2];
              ir8[2] = (v1790_data + (v1774_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1796_data = ir8[3];
              ir8[3] = (v1796_data + (v1774_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1802_data = ir8[4];
              ir8[4] = (v1802_data + (v1774_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1808_data = ir8[5];
              ir8[5] = (v1808_data + (v1774_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1814_data = ir8[6];
              ir8[6] = (v1814_data + (v1774_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1820_data = ir8[7];
              ir8[7] = (v1820_data + (v1774_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              if (v24_g) {
                #pragma unroll
                for (int32_t v1822_n1 = 0; v1822_n1 < 8; ++v1822_n1) {
                  float v1824_data = ir8[v1822_n1];
                  float v1825_data = r5[v1822_n1];
                  r8[v1822_n1] = (v1825_data + v1824_data);
                }
              }
              float r10[8]{};
              // r10 = load{g>r}(glb_m8);
              if (v24_g) {
                #pragma unroll
                for (int32_t v1828_i1 = 0; v1828_i1 < 8; ++v1828_i1) {
                  float v1833_data = glb_m8[(v23_lead + (v1828_i1 * 12))];
                  r10[v1828_i1] = v1833_data;
                }
              }
              // wait(r9 = load{g>r}(glb_m7););
              // wait(r10 = load{g>r}(glb_m8););
              float r11[8]{};
              // r11 = +(r9 * r10) + name: r8, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir11[8]{};
              float v1837_data = r9[0];
              float v1838_data = r10[0];
              float v1841_data = ir11[0];
              ir11[0] = (v1841_data + (v1837_data * (sycl::select_from_group(item.get_sub_group(), v1838_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1844_data = r10[1];
              float v1847_data = ir11[1];
              ir11[1] = (v1847_data + (v1837_data * (sycl::select_from_group(item.get_sub_group(), v1844_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1850_data = r10[2];
              float v1853_data = ir11[2];
              ir11[2] = (v1853_data + (v1837_data * (sycl::select_from_group(item.get_sub_group(), v1850_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1856_data = r10[3];
              float v1859_data = ir11[3];
              ir11[3] = (v1859_data + (v1837_data * (sycl::select_from_group(item.get_sub_group(), v1856_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1862_data = r10[4];
              float v1865_data = ir11[4];
              ir11[4] = (v1865_data + (v1837_data * (sycl::select_from_group(item.get_sub_group(), v1862_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1868_data = r10[5];
              float v1871_data = ir11[5];
              ir11[5] = (v1871_data + (v1837_data * (sycl::select_from_group(item.get_sub_group(), v1868_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1874_data = r10[6];
              float v1877_data = ir11[6];
              ir11[6] = (v1877_data + (v1837_data * (sycl::select_from_group(item.get_sub_group(), v1874_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1880_data = r10[7];
              float v1883_data = ir11[7];
              ir11[7] = (v1883_data + (v1837_data * (sycl::select_from_group(item.get_sub_group(), v1880_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1885_data = r9[1];
              float v1889_data = ir11[0];
              ir11[0] = (v1889_data + (v1885_data * (sycl::select_from_group(item.get_sub_group(), v1838_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1895_data = ir11[1];
              ir11[1] = (v1895_data + (v1885_data * (sycl::select_from_group(item.get_sub_group(), v1844_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1901_data = ir11[2];
              ir11[2] = (v1901_data + (v1885_data * (sycl::select_from_group(item.get_sub_group(), v1850_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1907_data = ir11[3];
              ir11[3] = (v1907_data + (v1885_data * (sycl::select_from_group(item.get_sub_group(), v1856_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1913_data = ir11[4];
              ir11[4] = (v1913_data + (v1885_data * (sycl::select_from_group(item.get_sub_group(), v1862_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1919_data = ir11[5];
              ir11[5] = (v1919_data + (v1885_data * (sycl::select_from_group(item.get_sub_group(), v1868_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1925_data = ir11[6];
              ir11[6] = (v1925_data + (v1885_data * (sycl::select_from_group(item.get_sub_group(), v1874_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1931_data = ir11[7];
              ir11[7] = (v1931_data + (v1885_data * (sycl::select_from_group(item.get_sub_group(), v1880_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1933_data = r9[2];
              float v1937_data = ir11[0];
              ir11[0] = (v1937_data + (v1933_data * (sycl::select_from_group(item.get_sub_group(), v1838_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1943_data = ir11[1];
              ir11[1] = (v1943_data + (v1933_data * (sycl::select_from_group(item.get_sub_group(), v1844_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1949_data = ir11[2];
              ir11[2] = (v1949_data + (v1933_data * (sycl::select_from_group(item.get_sub_group(), v1850_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1955_data = ir11[3];
              ir11[3] = (v1955_data + (v1933_data * (sycl::select_from_group(item.get_sub_group(), v1856_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1961_data = ir11[4];
              ir11[4] = (v1961_data + (v1933_data * (sycl::select_from_group(item.get_sub_group(), v1862_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1967_data = ir11[5];
              ir11[5] = (v1967_data + (v1933_data * (sycl::select_from_group(item.get_sub_group(), v1868_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1973_data = ir11[6];
              ir11[6] = (v1973_data + (v1933_data * (sycl::select_from_group(item.get_sub_group(), v1874_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1979_data = ir11[7];
              ir11[7] = (v1979_data + (v1933_data * (sycl::select_from_group(item.get_sub_group(), v1880_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1981_data = r9[3];
              float v1985_data = ir11[0];
              ir11[0] = (v1985_data + (v1981_data * (sycl::select_from_group(item.get_sub_group(), v1838_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1991_data = ir11[1];
              ir11[1] = (v1991_data + (v1981_data * (sycl::select_from_group(item.get_sub_group(), v1844_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1997_data = ir11[2];
              ir11[2] = (v1997_data + (v1981_data * (sycl::select_from_group(item.get_sub_group(), v1850_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v2003_data = ir11[3];
              ir11[3] = (v2003_data + (v1981_data * (sycl::select_from_group(item.get_sub_group(), v1856_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v2009_data = ir11[4];
              ir11[4] = (v2009_data + (v1981_data * (sycl::select_from_group(item.get_sub_group(), v1862_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v2015_data = ir11[5];
              ir11[5] = (v2015_data + (v1981_data * (sycl::select_from_group(item.get_sub_group(), v1868_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v2021_data = ir11[6];
              ir11[6] = (v2021_data + (v1981_data * (sycl::select_from_group(item.get_sub_group(), v1874_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v2027_data = ir11[7];
              ir11[7] = (v2027_data + (v1981_data * (sycl::select_from_group(item.get_sub_group(), v1880_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v2029_data = r9[4];
              float v2033_data = ir11[0];
              ir11[0] = (v2033_data + (v2029_data * (sycl::select_from_group(item.get_sub_group(), v1838_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v2039_data = ir11[1];
              ir11[1] = (v2039_data + (v2029_data * (sycl::select_from_group(item.get_sub_group(), v1844_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v2045_data = ir11[2];
              ir11[2] = (v2045_data + (v2029_data * (sycl::select_from_group(item.get_sub_group(), v1850_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v2051_data = ir11[3];
              ir11[3] = (v2051_data + (v2029_data * (sycl::select_from_group(item.get_sub_group(), v1856_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v2057_data = ir11[4];
              ir11[4] = (v2057_data + (v2029_data * (sycl::select_from_group(item.get_sub_group(), v1862_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v2063_data = ir11[5];
              ir11[5] = (v2063_data + (v2029_data * (sycl::select_from_group(item.get_sub_group(), v1868_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v2069_data = ir11[6];
              ir11[6] = (v2069_data + (v2029_data * (sycl::select_from_group(item.get_sub_group(), v1874_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v2075_data = ir11[7];
              ir11[7] = (v2075_data + (v2029_data * (sycl::select_from_group(item.get_sub_group(), v1880_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v2077_data = r9[5];
              float v2081_data = ir11[0];
              ir11[0] = (v2081_data + (v2077_data * (sycl::select_from_group(item.get_sub_group(), v1838_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v2087_data = ir11[1];
              ir11[1] = (v2087_data + (v2077_data * (sycl::select_from_group(item.get_sub_group(), v1844_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v2093_data = ir11[2];
              ir11[2] = (v2093_data + (v2077_data * (sycl::select_from_group(item.get_sub_group(), v1850_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v2099_data = ir11[3];
              ir11[3] = (v2099_data + (v2077_data * (sycl::select_from_group(item.get_sub_group(), v1856_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v2105_data = ir11[4];
              ir11[4] = (v2105_data + (v2077_data * (sycl::select_from_group(item.get_sub_group(), v1862_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v2111_data = ir11[5];
              ir11[5] = (v2111_data + (v2077_data * (sycl::select_from_group(item.get_sub_group(), v1868_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v2117_data = ir11[6];
              ir11[6] = (v2117_data + (v2077_data * (sycl::select_from_group(item.get_sub_group(), v1874_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v2123_data = ir11[7];
              ir11[7] = (v2123_data + (v2077_data * (sycl::select_from_group(item.get_sub_group(), v1880_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v2125_data = r9[6];
              float v2129_data = ir11[0];
              ir11[0] = (v2129_data + (v2125_data * (sycl::select_from_group(item.get_sub_group(), v1838_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v2135_data = ir11[1];
              ir11[1] = (v2135_data + (v2125_data * (sycl::select_from_group(item.get_sub_group(), v1844_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v2141_data = ir11[2];
              ir11[2] = (v2141_data + (v2125_data * (sycl::select_from_group(item.get_sub_group(), v1850_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v2147_data = ir11[3];
              ir11[3] = (v2147_data + (v2125_data * (sycl::select_from_group(item.get_sub_group(), v1856_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v2153_data = ir11[4];
              ir11[4] = (v2153_data + (v2125_data * (sycl::select_from_group(item.get_sub_group(), v1862_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v2159_data = ir11[5];
              ir11[5] = (v2159_data + (v2125_data * (sycl::select_from_group(item.get_sub_group(), v1868_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v2165_data = ir11[6];
              ir11[6] = (v2165_data + (v2125_data * (sycl::select_from_group(item.get_sub_group(), v1874_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v2171_data = ir11[7];
              ir11[7] = (v2171_data + (v2125_data * (sycl::select_from_group(item.get_sub_group(), v1880_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v2173_data = r9[7];
              float v2177_data = ir11[0];
              ir11[0] = (v2177_data + (v2173_data * (sycl::select_from_group(item.get_sub_group(), v1838_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v2183_data = ir11[1];
              ir11[1] = (v2183_data + (v2173_data * (sycl::select_from_group(item.get_sub_group(), v1844_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v2189_data = ir11[2];
              ir11[2] = (v2189_data + (v2173_data * (sycl::select_from_group(item.get_sub_group(), v1850_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v2195_data = ir11[3];
              ir11[3] = (v2195_data + (v2173_data * (sycl::select_from_group(item.get_sub_group(), v1856_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v2201_data = ir11[4];
              ir11[4] = (v2201_data + (v2173_data * (sycl::select_from_group(item.get_sub_group(), v1862_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v2207_data = ir11[5];
              ir11[5] = (v2207_data + (v2173_data * (sycl::select_from_group(item.get_sub_group(), v1868_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v2213_data = ir11[6];
              ir11[6] = (v2213_data + (v2173_data * (sycl::select_from_group(item.get_sub_group(), v1874_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v2219_data = ir11[7];
              ir11[7] = (v2219_data + (v2173_data * (sycl::select_from_group(item.get_sub_group(), v1880_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v2221_data = r9[8];
              float v2225_data = ir11[0];
              ir11[0] = (v2225_data + (v2221_data * (sycl::select_from_group(item.get_sub_group(), v1838_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v2231_data = ir11[1];
              ir11[1] = (v2231_data + (v2221_data * (sycl::select_from_group(item.get_sub_group(), v1844_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v2237_data = ir11[2];
              ir11[2] = (v2237_data + (v2221_data * (sycl::select_from_group(item.get_sub_group(), v1850_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v2243_data = ir11[3];
              ir11[3] = (v2243_data + (v2221_data * (sycl::select_from_group(item.get_sub_group(), v1856_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v2249_data = ir11[4];
              ir11[4] = (v2249_data + (v2221_data * (sycl::select_from_group(item.get_sub_group(), v1862_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v2255_data = ir11[5];
              ir11[5] = (v2255_data + (v2221_data * (sycl::select_from_group(item.get_sub_group(), v1868_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v2261_data = ir11[6];
              ir11[6] = (v2261_data + (v2221_data * (sycl::select_from_group(item.get_sub_group(), v1874_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v2267_data = ir11[7];
              ir11[7] = (v2267_data + (v2221_data * (sycl::select_from_group(item.get_sub_group(), v1880_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v2269_data = r9[9];
              float v2273_data = ir11[0];
              ir11[0] = (v2273_data + (v2269_data * (sycl::select_from_group(item.get_sub_group(), v1838_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v2279_data = ir11[1];
              ir11[1] = (v2279_data + (v2269_data * (sycl::select_from_group(item.get_sub_group(), v1844_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v2285_data = ir11[2];
              ir11[2] = (v2285_data + (v2269_data * (sycl::select_from_group(item.get_sub_group(), v1850_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v2291_data = ir11[3];
              ir11[3] = (v2291_data + (v2269_data * (sycl::select_from_group(item.get_sub_group(), v1856_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v2297_data = ir11[4];
              ir11[4] = (v2297_data + (v2269_data * (sycl::select_from_group(item.get_sub_group(), v1862_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v2303_data = ir11[5];
              ir11[5] = (v2303_data + (v2269_data * (sycl::select_from_group(item.get_sub_group(), v1868_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v2309_data = ir11[6];
              ir11[6] = (v2309_data + (v2269_data * (sycl::select_from_group(item.get_sub_group(), v1874_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v2315_data = ir11[7];
              ir11[7] = (v2315_data + (v2269_data * (sycl::select_from_group(item.get_sub_group(), v1880_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v2317_data = r9[10];
              float v2321_data = ir11[0];
              ir11[0] = (v2321_data + (v2317_data * (sycl::select_from_group(item.get_sub_group(), v1838_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v2327_data = ir11[1];
              ir11[1] = (v2327_data + (v2317_data * (sycl::select_from_group(item.get_sub_group(), v1844_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v2333_data = ir11[2];
              ir11[2] = (v2333_data + (v2317_data * (sycl::select_from_group(item.get_sub_group(), v1850_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v2339_data = ir11[3];
              ir11[3] = (v2339_data + (v2317_data * (sycl::select_from_group(item.get_sub_group(), v1856_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v2345_data = ir11[4];
              ir11[4] = (v2345_data + (v2317_data * (sycl::select_from_group(item.get_sub_group(), v1862_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v2351_data = ir11[5];
              ir11[5] = (v2351_data + (v2317_data * (sycl::select_from_group(item.get_sub_group(), v1868_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v2357_data = ir11[6];
              ir11[6] = (v2357_data + (v2317_data * (sycl::select_from_group(item.get_sub_group(), v1874_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v2363_data = ir11[7];
              ir11[7] = (v2363_data + (v2317_data * (sycl::select_from_group(item.get_sub_group(), v1880_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v2365_data = r9[11];
              float v2369_data = ir11[0];
              ir11[0] = (v2369_data + (v2365_data * (sycl::select_from_group(item.get_sub_group(), v1838_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v2375_data = ir11[1];
              ir11[1] = (v2375_data + (v2365_data * (sycl::select_from_group(item.get_sub_group(), v1844_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v2381_data = ir11[2];
              ir11[2] = (v2381_data + (v2365_data * (sycl::select_from_group(item.get_sub_group(), v1850_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v2387_data = ir11[3];
              ir11[3] = (v2387_data + (v2365_data * (sycl::select_from_group(item.get_sub_group(), v1856_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v2393_data = ir11[4];
              ir11[4] = (v2393_data + (v2365_data * (sycl::select_from_group(item.get_sub_group(), v1862_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v2399_data = ir11[5];
              ir11[5] = (v2399_data + (v2365_data * (sycl::select_from_group(item.get_sub_group(), v1868_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v2405_data = ir11[6];
              ir11[6] = (v2405_data + (v2365_data * (sycl::select_from_group(item.get_sub_group(), v1874_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v2411_data = ir11[7];
              ir11[7] = (v2411_data + (v2365_data * (sycl::select_from_group(item.get_sub_group(), v1880_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              if (v24_g) {
                #pragma unroll
                for (int32_t v2413_n1 = 0; v2413_n1 < 8; ++v2413_n1) {
                  float v2415_data = ir11[v2413_n1];
                  float v2416_data = r8[v2413_n1];
                  r11[v2413_n1] = (v2416_data + v2415_data);
                }
              }
              // glb_m0 = store{r>g}(r11);
              if (v24_g) {
                #pragma unroll
                for (int32_t v2418_i1 = 0; v2418_i1 < 8; ++v2418_i1) {
                  float v2420_data = r11[v2418_i1];
                  glb_m0[(v23_lead + (v2418_i1 * 12))] = v2420_data;
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

