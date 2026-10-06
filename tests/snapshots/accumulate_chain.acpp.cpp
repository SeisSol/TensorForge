// === base name ===
kernel_763d57f095b2dd3c

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_763d57f095b2dd3c = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_763d57f095b2dd3c(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_763d57f095b2dd3c(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_763d57f095b2dd3c(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_763d57f095b2dd3c(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_763d57f095b2dd3c(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_763d57f095b2dd3c(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, m4, m4_extraOffset, m5, m5_extraOffset, m6, m6_extraOffset, m7, m7_extraOffset, m8, m8_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_763d57f095b2dd3c(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, const float * m4, size_t m4_extraOffset, const float * m5, size_t m5_extraOffset, const float * m6, size_t m6_extraOffset, const float * m7, size_t m7_extraOffset, const float * m8, size_t m8_extraOffset, size_t numElements0, unsigned * flags0) {
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
          float* tempShrMem = &localShrMem0[0];
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v9_batchId0 * 144 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v9_batchId0 * 96 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v9_batchId0 * 144 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v9_batchId0 * 96 + 0 + m4_extraOffset];
              const float *const __restrict__ glb_m5 = &m5[v9_batchId0 * 144 + 0 + m5_extraOffset];
              const float *const __restrict__ glb_m6 = &m6[v9_batchId0 * 96 + 0 + m6_extraOffset];
              const float *const __restrict__ glb_m7 = &m7[v9_batchId0 * 144 + 0 + m7_extraOffset];
              const float *const __restrict__ glb_m8 = &m8[v9_batchId0 * 96 + 0 + m8_extraOffset];
              float r0[12]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v29_lead = item.get_local_id(2) % 16;
              bool v30_g = v29_lead < 12;
              if (v30_g) {
                #pragma unroll
                for (int32_t v31_i1 = 0; v31_i1 < 12; ++v31_i1) {
                  float v36_data = glb_m1[(v29_lead + (v31_i1 * 12))];
                  r0[v31_i1] = v36_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              if (v30_g) {
                #pragma unroll
                for (int32_t v39_i1 = 0; v39_i1 < 8; ++v39_i1) {
                  float v44_data = glb_m2[(v29_lead + (v39_i1 * 12))];
                  r1[v39_i1] = v44_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              float r3[12]{};
              // r3 = load{g>r}(glb_m3);
              if (v30_g) {
                #pragma unroll
                for (int32_t v47_i1 = 0; v47_i1 < 12; ++v47_i1) {
                  float v52_data = glb_m3[(v29_lead + (v47_i1 * 12))];
                  r3[v47_i1] = v52_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // ir2 = +(r0 * r1)
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir2[8]{};
              float v56_data = r0[0];
              float v57_data = r1[0];
              float v60_data = ir2[0];
              ir2[0] = (v60_data + (v56_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v63_data = r1[1];
              float v66_data = ir2[1];
              ir2[1] = (v66_data + (v56_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v69_data = r1[2];
              float v72_data = ir2[2];
              ir2[2] = (v72_data + (v56_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v75_data = r1[3];
              float v78_data = ir2[3];
              ir2[3] = (v78_data + (v56_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v81_data = r1[4];
              float v84_data = ir2[4];
              ir2[4] = (v84_data + (v56_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v87_data = r1[5];
              float v90_data = ir2[5];
              ir2[5] = (v90_data + (v56_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v93_data = r1[6];
              float v96_data = ir2[6];
              ir2[6] = (v96_data + (v56_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v99_data = r1[7];
              float v102_data = ir2[7];
              ir2[7] = (v102_data + (v56_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v104_data = r0[1];
              float v108_data = ir2[0];
              ir2[0] = (v108_data + (v104_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v114_data = ir2[1];
              ir2[1] = (v114_data + (v104_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v120_data = ir2[2];
              ir2[2] = (v120_data + (v104_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v126_data = ir2[3];
              ir2[3] = (v126_data + (v104_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v132_data = ir2[4];
              ir2[4] = (v132_data + (v104_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v138_data = ir2[5];
              ir2[5] = (v138_data + (v104_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v144_data = ir2[6];
              ir2[6] = (v144_data + (v104_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v150_data = ir2[7];
              ir2[7] = (v150_data + (v104_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v152_data = r0[2];
              float v156_data = ir2[0];
              ir2[0] = (v156_data + (v152_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v162_data = ir2[1];
              ir2[1] = (v162_data + (v152_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v168_data = ir2[2];
              ir2[2] = (v168_data + (v152_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v174_data = ir2[3];
              ir2[3] = (v174_data + (v152_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v180_data = ir2[4];
              ir2[4] = (v180_data + (v152_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v186_data = ir2[5];
              ir2[5] = (v186_data + (v152_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v192_data = ir2[6];
              ir2[6] = (v192_data + (v152_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v198_data = ir2[7];
              ir2[7] = (v198_data + (v152_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v200_data = r0[3];
              float v204_data = ir2[0];
              ir2[0] = (v204_data + (v200_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v210_data = ir2[1];
              ir2[1] = (v210_data + (v200_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v216_data = ir2[2];
              ir2[2] = (v216_data + (v200_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v222_data = ir2[3];
              ir2[3] = (v222_data + (v200_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v228_data = ir2[4];
              ir2[4] = (v228_data + (v200_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v234_data = ir2[5];
              ir2[5] = (v234_data + (v200_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v240_data = ir2[6];
              ir2[6] = (v240_data + (v200_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v246_data = ir2[7];
              ir2[7] = (v246_data + (v200_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v248_data = r0[4];
              float v252_data = ir2[0];
              ir2[0] = (v252_data + (v248_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v258_data = ir2[1];
              ir2[1] = (v258_data + (v248_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v264_data = ir2[2];
              ir2[2] = (v264_data + (v248_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v270_data = ir2[3];
              ir2[3] = (v270_data + (v248_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v276_data = ir2[4];
              ir2[4] = (v276_data + (v248_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v282_data = ir2[5];
              ir2[5] = (v282_data + (v248_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v288_data = ir2[6];
              ir2[6] = (v288_data + (v248_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v294_data = ir2[7];
              ir2[7] = (v294_data + (v248_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v296_data = r0[5];
              float v300_data = ir2[0];
              ir2[0] = (v300_data + (v296_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v306_data = ir2[1];
              ir2[1] = (v306_data + (v296_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v312_data = ir2[2];
              ir2[2] = (v312_data + (v296_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v318_data = ir2[3];
              ir2[3] = (v318_data + (v296_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v324_data = ir2[4];
              ir2[4] = (v324_data + (v296_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v330_data = ir2[5];
              ir2[5] = (v330_data + (v296_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v336_data = ir2[6];
              ir2[6] = (v336_data + (v296_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v342_data = ir2[7];
              ir2[7] = (v342_data + (v296_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v344_data = r0[6];
              float v348_data = ir2[0];
              ir2[0] = (v348_data + (v344_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v354_data = ir2[1];
              ir2[1] = (v354_data + (v344_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v360_data = ir2[2];
              ir2[2] = (v360_data + (v344_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v366_data = ir2[3];
              ir2[3] = (v366_data + (v344_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v372_data = ir2[4];
              ir2[4] = (v372_data + (v344_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v378_data = ir2[5];
              ir2[5] = (v378_data + (v344_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v384_data = ir2[6];
              ir2[6] = (v384_data + (v344_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v390_data = ir2[7];
              ir2[7] = (v390_data + (v344_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v392_data = r0[7];
              float v396_data = ir2[0];
              ir2[0] = (v396_data + (v392_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v402_data = ir2[1];
              ir2[1] = (v402_data + (v392_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v408_data = ir2[2];
              ir2[2] = (v408_data + (v392_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v414_data = ir2[3];
              ir2[3] = (v414_data + (v392_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v420_data = ir2[4];
              ir2[4] = (v420_data + (v392_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v426_data = ir2[5];
              ir2[5] = (v426_data + (v392_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v432_data = ir2[6];
              ir2[6] = (v432_data + (v392_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v438_data = ir2[7];
              ir2[7] = (v438_data + (v392_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v440_data = r0[8];
              float v444_data = ir2[0];
              ir2[0] = (v444_data + (v440_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v450_data = ir2[1];
              ir2[1] = (v450_data + (v440_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v456_data = ir2[2];
              ir2[2] = (v456_data + (v440_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v462_data = ir2[3];
              ir2[3] = (v462_data + (v440_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v468_data = ir2[4];
              ir2[4] = (v468_data + (v440_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v474_data = ir2[5];
              ir2[5] = (v474_data + (v440_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v480_data = ir2[6];
              ir2[6] = (v480_data + (v440_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v486_data = ir2[7];
              ir2[7] = (v486_data + (v440_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v488_data = r0[9];
              float v492_data = ir2[0];
              ir2[0] = (v492_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v498_data = ir2[1];
              ir2[1] = (v498_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v504_data = ir2[2];
              ir2[2] = (v504_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v510_data = ir2[3];
              ir2[3] = (v510_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v516_data = ir2[4];
              ir2[4] = (v516_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v522_data = ir2[5];
              ir2[5] = (v522_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v528_data = ir2[6];
              ir2[6] = (v528_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v534_data = ir2[7];
              ir2[7] = (v534_data + (v488_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v536_data = r0[10];
              float v540_data = ir2[0];
              ir2[0] = (v540_data + (v536_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v546_data = ir2[1];
              ir2[1] = (v546_data + (v536_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v552_data = ir2[2];
              ir2[2] = (v552_data + (v536_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v558_data = ir2[3];
              ir2[3] = (v558_data + (v536_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v564_data = ir2[4];
              ir2[4] = (v564_data + (v536_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v570_data = ir2[5];
              ir2[5] = (v570_data + (v536_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v576_data = ir2[6];
              ir2[6] = (v576_data + (v536_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v582_data = ir2[7];
              ir2[7] = (v582_data + (v536_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v584_data = r0[11];
              float v588_data = ir2[0];
              ir2[0] = (v588_data + (v584_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v594_data = ir2[1];
              ir2[1] = (v594_data + (v584_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v600_data = ir2[2];
              ir2[2] = (v600_data + (v584_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v606_data = ir2[3];
              ir2[3] = (v606_data + (v584_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v612_data = ir2[4];
              ir2[4] = (v612_data + (v584_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v618_data = ir2[5];
              ir2[5] = (v618_data + (v584_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v624_data = ir2[6];
              ir2[6] = (v624_data + (v584_data * (sycl::select_from_group(item.get_sub_group(), v93_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v630_data = ir2[7];
              ir2[7] = (v630_data + (v584_data * (sycl::select_from_group(item.get_sub_group(), v99_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              // r2 = ir2
              if (v30_g) {
                #pragma unroll
                for (int32_t v632_n1 = 0; v632_n1 < 8; ++v632_n1) {
                  float v634_data = ir2[v632_n1];
                  r2[v632_n1] = v634_data;
                }
              }
              float r4[8]{};
              // r4 = load{g>r}(glb_m4);
              if (v30_g) {
                #pragma unroll
                for (int32_t v636_i1 = 0; v636_i1 < 8; ++v636_i1) {
                  float v641_data = glb_m4[(v29_lead + (v636_i1 * 12))];
                  r4[v636_i1] = v641_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m3););
              float r6[12]{};
              // r6 = load{g>r}(glb_m5);
              if (v30_g) {
                #pragma unroll
                for (int32_t v644_i1 = 0; v644_i1 < 12; ++v644_i1) {
                  float v649_data = glb_m5[(v29_lead + (v644_i1 * 12))];
                  r6[v644_i1] = v649_data;
                }
              }
              // wait(r4 = load{g>r}(glb_m4););
              float r5[8]{};
              // ir5 = +(r3 * r4)
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir5[8]{};
              float v653_data = r3[0];
              float v654_data = r4[0];
              float v657_data = ir5[0];
              ir5[0] = (v657_data + (v653_data * (sycl::select_from_group(item.get_sub_group(), v654_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v660_data = r4[1];
              float v663_data = ir5[1];
              ir5[1] = (v663_data + (v653_data * (sycl::select_from_group(item.get_sub_group(), v660_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v666_data = r4[2];
              float v669_data = ir5[2];
              ir5[2] = (v669_data + (v653_data * (sycl::select_from_group(item.get_sub_group(), v666_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v672_data = r4[3];
              float v675_data = ir5[3];
              ir5[3] = (v675_data + (v653_data * (sycl::select_from_group(item.get_sub_group(), v672_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v678_data = r4[4];
              float v681_data = ir5[4];
              ir5[4] = (v681_data + (v653_data * (sycl::select_from_group(item.get_sub_group(), v678_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v684_data = r4[5];
              float v687_data = ir5[5];
              ir5[5] = (v687_data + (v653_data * (sycl::select_from_group(item.get_sub_group(), v684_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v690_data = r4[6];
              float v693_data = ir5[6];
              ir5[6] = (v693_data + (v653_data * (sycl::select_from_group(item.get_sub_group(), v690_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v696_data = r4[7];
              float v699_data = ir5[7];
              ir5[7] = (v699_data + (v653_data * (sycl::select_from_group(item.get_sub_group(), v696_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v701_data = r3[1];
              float v705_data = ir5[0];
              ir5[0] = (v705_data + (v701_data * (sycl::select_from_group(item.get_sub_group(), v654_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v711_data = ir5[1];
              ir5[1] = (v711_data + (v701_data * (sycl::select_from_group(item.get_sub_group(), v660_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v717_data = ir5[2];
              ir5[2] = (v717_data + (v701_data * (sycl::select_from_group(item.get_sub_group(), v666_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v723_data = ir5[3];
              ir5[3] = (v723_data + (v701_data * (sycl::select_from_group(item.get_sub_group(), v672_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v729_data = ir5[4];
              ir5[4] = (v729_data + (v701_data * (sycl::select_from_group(item.get_sub_group(), v678_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v735_data = ir5[5];
              ir5[5] = (v735_data + (v701_data * (sycl::select_from_group(item.get_sub_group(), v684_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v741_data = ir5[6];
              ir5[6] = (v741_data + (v701_data * (sycl::select_from_group(item.get_sub_group(), v690_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v747_data = ir5[7];
              ir5[7] = (v747_data + (v701_data * (sycl::select_from_group(item.get_sub_group(), v696_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v749_data = r3[2];
              float v753_data = ir5[0];
              ir5[0] = (v753_data + (v749_data * (sycl::select_from_group(item.get_sub_group(), v654_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v759_data = ir5[1];
              ir5[1] = (v759_data + (v749_data * (sycl::select_from_group(item.get_sub_group(), v660_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v765_data = ir5[2];
              ir5[2] = (v765_data + (v749_data * (sycl::select_from_group(item.get_sub_group(), v666_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v771_data = ir5[3];
              ir5[3] = (v771_data + (v749_data * (sycl::select_from_group(item.get_sub_group(), v672_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v777_data = ir5[4];
              ir5[4] = (v777_data + (v749_data * (sycl::select_from_group(item.get_sub_group(), v678_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v783_data = ir5[5];
              ir5[5] = (v783_data + (v749_data * (sycl::select_from_group(item.get_sub_group(), v684_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v789_data = ir5[6];
              ir5[6] = (v789_data + (v749_data * (sycl::select_from_group(item.get_sub_group(), v690_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v795_data = ir5[7];
              ir5[7] = (v795_data + (v749_data * (sycl::select_from_group(item.get_sub_group(), v696_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v797_data = r3[3];
              float v801_data = ir5[0];
              ir5[0] = (v801_data + (v797_data * (sycl::select_from_group(item.get_sub_group(), v654_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v807_data = ir5[1];
              ir5[1] = (v807_data + (v797_data * (sycl::select_from_group(item.get_sub_group(), v660_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v813_data = ir5[2];
              ir5[2] = (v813_data + (v797_data * (sycl::select_from_group(item.get_sub_group(), v666_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v819_data = ir5[3];
              ir5[3] = (v819_data + (v797_data * (sycl::select_from_group(item.get_sub_group(), v672_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v825_data = ir5[4];
              ir5[4] = (v825_data + (v797_data * (sycl::select_from_group(item.get_sub_group(), v678_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v831_data = ir5[5];
              ir5[5] = (v831_data + (v797_data * (sycl::select_from_group(item.get_sub_group(), v684_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v837_data = ir5[6];
              ir5[6] = (v837_data + (v797_data * (sycl::select_from_group(item.get_sub_group(), v690_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v843_data = ir5[7];
              ir5[7] = (v843_data + (v797_data * (sycl::select_from_group(item.get_sub_group(), v696_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v845_data = r3[4];
              float v849_data = ir5[0];
              ir5[0] = (v849_data + (v845_data * (sycl::select_from_group(item.get_sub_group(), v654_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v855_data = ir5[1];
              ir5[1] = (v855_data + (v845_data * (sycl::select_from_group(item.get_sub_group(), v660_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v861_data = ir5[2];
              ir5[2] = (v861_data + (v845_data * (sycl::select_from_group(item.get_sub_group(), v666_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v867_data = ir5[3];
              ir5[3] = (v867_data + (v845_data * (sycl::select_from_group(item.get_sub_group(), v672_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v873_data = ir5[4];
              ir5[4] = (v873_data + (v845_data * (sycl::select_from_group(item.get_sub_group(), v678_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v879_data = ir5[5];
              ir5[5] = (v879_data + (v845_data * (sycl::select_from_group(item.get_sub_group(), v684_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v885_data = ir5[6];
              ir5[6] = (v885_data + (v845_data * (sycl::select_from_group(item.get_sub_group(), v690_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v891_data = ir5[7];
              ir5[7] = (v891_data + (v845_data * (sycl::select_from_group(item.get_sub_group(), v696_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v893_data = r3[5];
              float v897_data = ir5[0];
              ir5[0] = (v897_data + (v893_data * (sycl::select_from_group(item.get_sub_group(), v654_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v903_data = ir5[1];
              ir5[1] = (v903_data + (v893_data * (sycl::select_from_group(item.get_sub_group(), v660_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v909_data = ir5[2];
              ir5[2] = (v909_data + (v893_data * (sycl::select_from_group(item.get_sub_group(), v666_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v915_data = ir5[3];
              ir5[3] = (v915_data + (v893_data * (sycl::select_from_group(item.get_sub_group(), v672_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v921_data = ir5[4];
              ir5[4] = (v921_data + (v893_data * (sycl::select_from_group(item.get_sub_group(), v678_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v927_data = ir5[5];
              ir5[5] = (v927_data + (v893_data * (sycl::select_from_group(item.get_sub_group(), v684_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v933_data = ir5[6];
              ir5[6] = (v933_data + (v893_data * (sycl::select_from_group(item.get_sub_group(), v690_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v939_data = ir5[7];
              ir5[7] = (v939_data + (v893_data * (sycl::select_from_group(item.get_sub_group(), v696_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v941_data = r3[6];
              float v945_data = ir5[0];
              ir5[0] = (v945_data + (v941_data * (sycl::select_from_group(item.get_sub_group(), v654_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v951_data = ir5[1];
              ir5[1] = (v951_data + (v941_data * (sycl::select_from_group(item.get_sub_group(), v660_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v957_data = ir5[2];
              ir5[2] = (v957_data + (v941_data * (sycl::select_from_group(item.get_sub_group(), v666_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v963_data = ir5[3];
              ir5[3] = (v963_data + (v941_data * (sycl::select_from_group(item.get_sub_group(), v672_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v969_data = ir5[4];
              ir5[4] = (v969_data + (v941_data * (sycl::select_from_group(item.get_sub_group(), v678_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v975_data = ir5[5];
              ir5[5] = (v975_data + (v941_data * (sycl::select_from_group(item.get_sub_group(), v684_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v981_data = ir5[6];
              ir5[6] = (v981_data + (v941_data * (sycl::select_from_group(item.get_sub_group(), v690_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v987_data = ir5[7];
              ir5[7] = (v987_data + (v941_data * (sycl::select_from_group(item.get_sub_group(), v696_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v989_data = r3[7];
              float v993_data = ir5[0];
              ir5[0] = (v993_data + (v989_data * (sycl::select_from_group(item.get_sub_group(), v654_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v999_data = ir5[1];
              ir5[1] = (v999_data + (v989_data * (sycl::select_from_group(item.get_sub_group(), v660_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1005_data = ir5[2];
              ir5[2] = (v1005_data + (v989_data * (sycl::select_from_group(item.get_sub_group(), v666_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1011_data = ir5[3];
              ir5[3] = (v1011_data + (v989_data * (sycl::select_from_group(item.get_sub_group(), v672_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1017_data = ir5[4];
              ir5[4] = (v1017_data + (v989_data * (sycl::select_from_group(item.get_sub_group(), v678_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1023_data = ir5[5];
              ir5[5] = (v1023_data + (v989_data * (sycl::select_from_group(item.get_sub_group(), v684_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1029_data = ir5[6];
              ir5[6] = (v1029_data + (v989_data * (sycl::select_from_group(item.get_sub_group(), v690_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1035_data = ir5[7];
              ir5[7] = (v1035_data + (v989_data * (sycl::select_from_group(item.get_sub_group(), v696_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1037_data = r3[8];
              float v1041_data = ir5[0];
              ir5[0] = (v1041_data + (v1037_data * (sycl::select_from_group(item.get_sub_group(), v654_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1047_data = ir5[1];
              ir5[1] = (v1047_data + (v1037_data * (sycl::select_from_group(item.get_sub_group(), v660_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1053_data = ir5[2];
              ir5[2] = (v1053_data + (v1037_data * (sycl::select_from_group(item.get_sub_group(), v666_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1059_data = ir5[3];
              ir5[3] = (v1059_data + (v1037_data * (sycl::select_from_group(item.get_sub_group(), v672_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1065_data = ir5[4];
              ir5[4] = (v1065_data + (v1037_data * (sycl::select_from_group(item.get_sub_group(), v678_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1071_data = ir5[5];
              ir5[5] = (v1071_data + (v1037_data * (sycl::select_from_group(item.get_sub_group(), v684_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1077_data = ir5[6];
              ir5[6] = (v1077_data + (v1037_data * (sycl::select_from_group(item.get_sub_group(), v690_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1083_data = ir5[7];
              ir5[7] = (v1083_data + (v1037_data * (sycl::select_from_group(item.get_sub_group(), v696_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1085_data = r3[9];
              float v1089_data = ir5[0];
              ir5[0] = (v1089_data + (v1085_data * (sycl::select_from_group(item.get_sub_group(), v654_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1095_data = ir5[1];
              ir5[1] = (v1095_data + (v1085_data * (sycl::select_from_group(item.get_sub_group(), v660_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1101_data = ir5[2];
              ir5[2] = (v1101_data + (v1085_data * (sycl::select_from_group(item.get_sub_group(), v666_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1107_data = ir5[3];
              ir5[3] = (v1107_data + (v1085_data * (sycl::select_from_group(item.get_sub_group(), v672_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1113_data = ir5[4];
              ir5[4] = (v1113_data + (v1085_data * (sycl::select_from_group(item.get_sub_group(), v678_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1119_data = ir5[5];
              ir5[5] = (v1119_data + (v1085_data * (sycl::select_from_group(item.get_sub_group(), v684_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1125_data = ir5[6];
              ir5[6] = (v1125_data + (v1085_data * (sycl::select_from_group(item.get_sub_group(), v690_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1131_data = ir5[7];
              ir5[7] = (v1131_data + (v1085_data * (sycl::select_from_group(item.get_sub_group(), v696_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1133_data = r3[10];
              float v1137_data = ir5[0];
              ir5[0] = (v1137_data + (v1133_data * (sycl::select_from_group(item.get_sub_group(), v654_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1143_data = ir5[1];
              ir5[1] = (v1143_data + (v1133_data * (sycl::select_from_group(item.get_sub_group(), v660_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1149_data = ir5[2];
              ir5[2] = (v1149_data + (v1133_data * (sycl::select_from_group(item.get_sub_group(), v666_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1155_data = ir5[3];
              ir5[3] = (v1155_data + (v1133_data * (sycl::select_from_group(item.get_sub_group(), v672_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1161_data = ir5[4];
              ir5[4] = (v1161_data + (v1133_data * (sycl::select_from_group(item.get_sub_group(), v678_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1167_data = ir5[5];
              ir5[5] = (v1167_data + (v1133_data * (sycl::select_from_group(item.get_sub_group(), v684_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1173_data = ir5[6];
              ir5[6] = (v1173_data + (v1133_data * (sycl::select_from_group(item.get_sub_group(), v690_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1179_data = ir5[7];
              ir5[7] = (v1179_data + (v1133_data * (sycl::select_from_group(item.get_sub_group(), v696_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1181_data = r3[11];
              float v1185_data = ir5[0];
              ir5[0] = (v1185_data + (v1181_data * (sycl::select_from_group(item.get_sub_group(), v654_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1191_data = ir5[1];
              ir5[1] = (v1191_data + (v1181_data * (sycl::select_from_group(item.get_sub_group(), v660_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1197_data = ir5[2];
              ir5[2] = (v1197_data + (v1181_data * (sycl::select_from_group(item.get_sub_group(), v666_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1203_data = ir5[3];
              ir5[3] = (v1203_data + (v1181_data * (sycl::select_from_group(item.get_sub_group(), v672_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1209_data = ir5[4];
              ir5[4] = (v1209_data + (v1181_data * (sycl::select_from_group(item.get_sub_group(), v678_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1215_data = ir5[5];
              ir5[5] = (v1215_data + (v1181_data * (sycl::select_from_group(item.get_sub_group(), v684_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1221_data = ir5[6];
              ir5[6] = (v1221_data + (v1181_data * (sycl::select_from_group(item.get_sub_group(), v690_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1227_data = ir5[7];
              ir5[7] = (v1227_data + (v1181_data * (sycl::select_from_group(item.get_sub_group(), v696_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              // r5 = ir5 + r2
              if (v30_g) {
                #pragma unroll
                for (int32_t v1229_n1 = 0; v1229_n1 < 8; ++v1229_n1) {
                  float v1231_data = ir5[v1229_n1];
                  float v1232_data = r2[v1229_n1];
                  r5[v1229_n1] = (v1232_data + v1231_data);
                }
              }
              float r7[8]{};
              // r7 = load{g>r}(glb_m6);
              if (v30_g) {
                #pragma unroll
                for (int32_t v1235_i1 = 0; v1235_i1 < 8; ++v1235_i1) {
                  float v1240_data = glb_m6[(v29_lead + (v1235_i1 * 12))];
                  r7[v1235_i1] = v1240_data;
                }
              }
              // wait(r6 = load{g>r}(glb_m5););
              float r9[12]{};
              // r9 = load{g>r}(glb_m7);
              if (v30_g) {
                #pragma unroll
                for (int32_t v1243_i1 = 0; v1243_i1 < 12; ++v1243_i1) {
                  float v1248_data = glb_m7[(v29_lead + (v1243_i1 * 12))];
                  r9[v1243_i1] = v1248_data;
                }
              }
              // wait(r7 = load{g>r}(glb_m6););
              float r8[8]{};
              // ir8 = +(r6 * r7)
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir8[8]{};
              float v1252_data = r6[0];
              float v1253_data = r7[0];
              float v1256_data = ir8[0];
              ir8[0] = (v1256_data + (v1252_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1259_data = r7[1];
              float v1262_data = ir8[1];
              ir8[1] = (v1262_data + (v1252_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1265_data = r7[2];
              float v1268_data = ir8[2];
              ir8[2] = (v1268_data + (v1252_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1271_data = r7[3];
              float v1274_data = ir8[3];
              ir8[3] = (v1274_data + (v1252_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1277_data = r7[4];
              float v1280_data = ir8[4];
              ir8[4] = (v1280_data + (v1252_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1283_data = r7[5];
              float v1286_data = ir8[5];
              ir8[5] = (v1286_data + (v1252_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1289_data = r7[6];
              float v1292_data = ir8[6];
              ir8[6] = (v1292_data + (v1252_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1295_data = r7[7];
              float v1298_data = ir8[7];
              ir8[7] = (v1298_data + (v1252_data * (sycl::select_from_group(item.get_sub_group(), v1295_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1300_data = r6[1];
              float v1304_data = ir8[0];
              ir8[0] = (v1304_data + (v1300_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1310_data = ir8[1];
              ir8[1] = (v1310_data + (v1300_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1316_data = ir8[2];
              ir8[2] = (v1316_data + (v1300_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1322_data = ir8[3];
              ir8[3] = (v1322_data + (v1300_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1328_data = ir8[4];
              ir8[4] = (v1328_data + (v1300_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1334_data = ir8[5];
              ir8[5] = (v1334_data + (v1300_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1340_data = ir8[6];
              ir8[6] = (v1340_data + (v1300_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1346_data = ir8[7];
              ir8[7] = (v1346_data + (v1300_data * (sycl::select_from_group(item.get_sub_group(), v1295_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1348_data = r6[2];
              float v1352_data = ir8[0];
              ir8[0] = (v1352_data + (v1348_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1358_data = ir8[1];
              ir8[1] = (v1358_data + (v1348_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1364_data = ir8[2];
              ir8[2] = (v1364_data + (v1348_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1370_data = ir8[3];
              ir8[3] = (v1370_data + (v1348_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1376_data = ir8[4];
              ir8[4] = (v1376_data + (v1348_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1382_data = ir8[5];
              ir8[5] = (v1382_data + (v1348_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1388_data = ir8[6];
              ir8[6] = (v1388_data + (v1348_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1394_data = ir8[7];
              ir8[7] = (v1394_data + (v1348_data * (sycl::select_from_group(item.get_sub_group(), v1295_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1396_data = r6[3];
              float v1400_data = ir8[0];
              ir8[0] = (v1400_data + (v1396_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1406_data = ir8[1];
              ir8[1] = (v1406_data + (v1396_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1412_data = ir8[2];
              ir8[2] = (v1412_data + (v1396_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1418_data = ir8[3];
              ir8[3] = (v1418_data + (v1396_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1424_data = ir8[4];
              ir8[4] = (v1424_data + (v1396_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1430_data = ir8[5];
              ir8[5] = (v1430_data + (v1396_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1436_data = ir8[6];
              ir8[6] = (v1436_data + (v1396_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1442_data = ir8[7];
              ir8[7] = (v1442_data + (v1396_data * (sycl::select_from_group(item.get_sub_group(), v1295_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1444_data = r6[4];
              float v1448_data = ir8[0];
              ir8[0] = (v1448_data + (v1444_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1454_data = ir8[1];
              ir8[1] = (v1454_data + (v1444_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1460_data = ir8[2];
              ir8[2] = (v1460_data + (v1444_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1466_data = ir8[3];
              ir8[3] = (v1466_data + (v1444_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1472_data = ir8[4];
              ir8[4] = (v1472_data + (v1444_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1478_data = ir8[5];
              ir8[5] = (v1478_data + (v1444_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1484_data = ir8[6];
              ir8[6] = (v1484_data + (v1444_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1490_data = ir8[7];
              ir8[7] = (v1490_data + (v1444_data * (sycl::select_from_group(item.get_sub_group(), v1295_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v1492_data = r6[5];
              float v1496_data = ir8[0];
              ir8[0] = (v1496_data + (v1492_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1502_data = ir8[1];
              ir8[1] = (v1502_data + (v1492_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1508_data = ir8[2];
              ir8[2] = (v1508_data + (v1492_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1514_data = ir8[3];
              ir8[3] = (v1514_data + (v1492_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1520_data = ir8[4];
              ir8[4] = (v1520_data + (v1492_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1526_data = ir8[5];
              ir8[5] = (v1526_data + (v1492_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1532_data = ir8[6];
              ir8[6] = (v1532_data + (v1492_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1538_data = ir8[7];
              ir8[7] = (v1538_data + (v1492_data * (sycl::select_from_group(item.get_sub_group(), v1295_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v1540_data = r6[6];
              float v1544_data = ir8[0];
              ir8[0] = (v1544_data + (v1540_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1550_data = ir8[1];
              ir8[1] = (v1550_data + (v1540_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1556_data = ir8[2];
              ir8[2] = (v1556_data + (v1540_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1562_data = ir8[3];
              ir8[3] = (v1562_data + (v1540_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1568_data = ir8[4];
              ir8[4] = (v1568_data + (v1540_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1574_data = ir8[5];
              ir8[5] = (v1574_data + (v1540_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1580_data = ir8[6];
              ir8[6] = (v1580_data + (v1540_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1586_data = ir8[7];
              ir8[7] = (v1586_data + (v1540_data * (sycl::select_from_group(item.get_sub_group(), v1295_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v1588_data = r6[7];
              float v1592_data = ir8[0];
              ir8[0] = (v1592_data + (v1588_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1598_data = ir8[1];
              ir8[1] = (v1598_data + (v1588_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1604_data = ir8[2];
              ir8[2] = (v1604_data + (v1588_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1610_data = ir8[3];
              ir8[3] = (v1610_data + (v1588_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1616_data = ir8[4];
              ir8[4] = (v1616_data + (v1588_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1622_data = ir8[5];
              ir8[5] = (v1622_data + (v1588_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1628_data = ir8[6];
              ir8[6] = (v1628_data + (v1588_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1634_data = ir8[7];
              ir8[7] = (v1634_data + (v1588_data * (sycl::select_from_group(item.get_sub_group(), v1295_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v1636_data = r6[8];
              float v1640_data = ir8[0];
              ir8[0] = (v1640_data + (v1636_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1646_data = ir8[1];
              ir8[1] = (v1646_data + (v1636_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1652_data = ir8[2];
              ir8[2] = (v1652_data + (v1636_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1658_data = ir8[3];
              ir8[3] = (v1658_data + (v1636_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1664_data = ir8[4];
              ir8[4] = (v1664_data + (v1636_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1670_data = ir8[5];
              ir8[5] = (v1670_data + (v1636_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1676_data = ir8[6];
              ir8[6] = (v1676_data + (v1636_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1682_data = ir8[7];
              ir8[7] = (v1682_data + (v1636_data * (sycl::select_from_group(item.get_sub_group(), v1295_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v1684_data = r6[9];
              float v1688_data = ir8[0];
              ir8[0] = (v1688_data + (v1684_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1694_data = ir8[1];
              ir8[1] = (v1694_data + (v1684_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1700_data = ir8[2];
              ir8[2] = (v1700_data + (v1684_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1706_data = ir8[3];
              ir8[3] = (v1706_data + (v1684_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1712_data = ir8[4];
              ir8[4] = (v1712_data + (v1684_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1718_data = ir8[5];
              ir8[5] = (v1718_data + (v1684_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1724_data = ir8[6];
              ir8[6] = (v1724_data + (v1684_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1730_data = ir8[7];
              ir8[7] = (v1730_data + (v1684_data * (sycl::select_from_group(item.get_sub_group(), v1295_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v1732_data = r6[10];
              float v1736_data = ir8[0];
              ir8[0] = (v1736_data + (v1732_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1742_data = ir8[1];
              ir8[1] = (v1742_data + (v1732_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1748_data = ir8[2];
              ir8[2] = (v1748_data + (v1732_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1754_data = ir8[3];
              ir8[3] = (v1754_data + (v1732_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1760_data = ir8[4];
              ir8[4] = (v1760_data + (v1732_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1766_data = ir8[5];
              ir8[5] = (v1766_data + (v1732_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1772_data = ir8[6];
              ir8[6] = (v1772_data + (v1732_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1778_data = ir8[7];
              ir8[7] = (v1778_data + (v1732_data * (sycl::select_from_group(item.get_sub_group(), v1295_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v1780_data = r6[11];
              float v1784_data = ir8[0];
              ir8[0] = (v1784_data + (v1780_data * (sycl::select_from_group(item.get_sub_group(), v1253_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1790_data = ir8[1];
              ir8[1] = (v1790_data + (v1780_data * (sycl::select_from_group(item.get_sub_group(), v1259_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1796_data = ir8[2];
              ir8[2] = (v1796_data + (v1780_data * (sycl::select_from_group(item.get_sub_group(), v1265_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1802_data = ir8[3];
              ir8[3] = (v1802_data + (v1780_data * (sycl::select_from_group(item.get_sub_group(), v1271_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1808_data = ir8[4];
              ir8[4] = (v1808_data + (v1780_data * (sycl::select_from_group(item.get_sub_group(), v1277_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1814_data = ir8[5];
              ir8[5] = (v1814_data + (v1780_data * (sycl::select_from_group(item.get_sub_group(), v1283_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1820_data = ir8[6];
              ir8[6] = (v1820_data + (v1780_data * (sycl::select_from_group(item.get_sub_group(), v1289_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v1826_data = ir8[7];
              ir8[7] = (v1826_data + (v1780_data * (sycl::select_from_group(item.get_sub_group(), v1295_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              // r8 = ir8 + r5
              if (v30_g) {
                #pragma unroll
                for (int32_t v1828_n1 = 0; v1828_n1 < 8; ++v1828_n1) {
                  float v1830_data = ir8[v1828_n1];
                  float v1831_data = r5[v1828_n1];
                  r8[v1828_n1] = (v1831_data + v1830_data);
                }
              }
              float r10[8]{};
              // r10 = load{g>r}(glb_m8);
              if (v30_g) {
                #pragma unroll
                for (int32_t v1834_i1 = 0; v1834_i1 < 8; ++v1834_i1) {
                  float v1839_data = glb_m8[(v29_lead + (v1834_i1 * 12))];
                  r10[v1834_i1] = v1839_data;
                }
              }
              // wait(r9 = load{g>r}(glb_m7););
              // wait(r10 = load{g>r}(glb_m8););
              float r11[8]{};
              // ir11 = +(r9 * r10)
              // [(0, 12), (0, 8)] [(0, 12)]
              float ir11[8]{};
              float v1843_data = r9[0];
              float v1844_data = r10[0];
              float v1847_data = ir11[0];
              ir11[0] = (v1847_data + (v1843_data * (sycl::select_from_group(item.get_sub_group(), v1844_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1850_data = r10[1];
              float v1853_data = ir11[1];
              ir11[1] = (v1853_data + (v1843_data * (sycl::select_from_group(item.get_sub_group(), v1850_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1856_data = r10[2];
              float v1859_data = ir11[2];
              ir11[2] = (v1859_data + (v1843_data * (sycl::select_from_group(item.get_sub_group(), v1856_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1862_data = r10[3];
              float v1865_data = ir11[3];
              ir11[3] = (v1865_data + (v1843_data * (sycl::select_from_group(item.get_sub_group(), v1862_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1868_data = r10[4];
              float v1871_data = ir11[4];
              ir11[4] = (v1871_data + (v1843_data * (sycl::select_from_group(item.get_sub_group(), v1868_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1874_data = r10[5];
              float v1877_data = ir11[5];
              ir11[5] = (v1877_data + (v1843_data * (sycl::select_from_group(item.get_sub_group(), v1874_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1880_data = r10[6];
              float v1883_data = ir11[6];
              ir11[6] = (v1883_data + (v1843_data * (sycl::select_from_group(item.get_sub_group(), v1880_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1886_data = r10[7];
              float v1889_data = ir11[7];
              ir11[7] = (v1889_data + (v1843_data * (sycl::select_from_group(item.get_sub_group(), v1886_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v1891_data = r9[1];
              float v1895_data = ir11[0];
              ir11[0] = (v1895_data + (v1891_data * (sycl::select_from_group(item.get_sub_group(), v1844_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1901_data = ir11[1];
              ir11[1] = (v1901_data + (v1891_data * (sycl::select_from_group(item.get_sub_group(), v1850_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1907_data = ir11[2];
              ir11[2] = (v1907_data + (v1891_data * (sycl::select_from_group(item.get_sub_group(), v1856_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1913_data = ir11[3];
              ir11[3] = (v1913_data + (v1891_data * (sycl::select_from_group(item.get_sub_group(), v1862_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1919_data = ir11[4];
              ir11[4] = (v1919_data + (v1891_data * (sycl::select_from_group(item.get_sub_group(), v1868_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1925_data = ir11[5];
              ir11[5] = (v1925_data + (v1891_data * (sycl::select_from_group(item.get_sub_group(), v1874_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1931_data = ir11[6];
              ir11[6] = (v1931_data + (v1891_data * (sycl::select_from_group(item.get_sub_group(), v1880_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1937_data = ir11[7];
              ir11[7] = (v1937_data + (v1891_data * (sycl::select_from_group(item.get_sub_group(), v1886_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v1939_data = r9[2];
              float v1943_data = ir11[0];
              ir11[0] = (v1943_data + (v1939_data * (sycl::select_from_group(item.get_sub_group(), v1844_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1949_data = ir11[1];
              ir11[1] = (v1949_data + (v1939_data * (sycl::select_from_group(item.get_sub_group(), v1850_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1955_data = ir11[2];
              ir11[2] = (v1955_data + (v1939_data * (sycl::select_from_group(item.get_sub_group(), v1856_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1961_data = ir11[3];
              ir11[3] = (v1961_data + (v1939_data * (sycl::select_from_group(item.get_sub_group(), v1862_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1967_data = ir11[4];
              ir11[4] = (v1967_data + (v1939_data * (sycl::select_from_group(item.get_sub_group(), v1868_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1973_data = ir11[5];
              ir11[5] = (v1973_data + (v1939_data * (sycl::select_from_group(item.get_sub_group(), v1874_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1979_data = ir11[6];
              ir11[6] = (v1979_data + (v1939_data * (sycl::select_from_group(item.get_sub_group(), v1880_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1985_data = ir11[7];
              ir11[7] = (v1985_data + (v1939_data * (sycl::select_from_group(item.get_sub_group(), v1886_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v1987_data = r9[3];
              float v1991_data = ir11[0];
              ir11[0] = (v1991_data + (v1987_data * (sycl::select_from_group(item.get_sub_group(), v1844_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v1997_data = ir11[1];
              ir11[1] = (v1997_data + (v1987_data * (sycl::select_from_group(item.get_sub_group(), v1850_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v2003_data = ir11[2];
              ir11[2] = (v2003_data + (v1987_data * (sycl::select_from_group(item.get_sub_group(), v1856_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v2009_data = ir11[3];
              ir11[3] = (v2009_data + (v1987_data * (sycl::select_from_group(item.get_sub_group(), v1862_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v2015_data = ir11[4];
              ir11[4] = (v2015_data + (v1987_data * (sycl::select_from_group(item.get_sub_group(), v1868_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v2021_data = ir11[5];
              ir11[5] = (v2021_data + (v1987_data * (sycl::select_from_group(item.get_sub_group(), v1874_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v2027_data = ir11[6];
              ir11[6] = (v2027_data + (v1987_data * (sycl::select_from_group(item.get_sub_group(), v1880_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v2033_data = ir11[7];
              ir11[7] = (v2033_data + (v1987_data * (sycl::select_from_group(item.get_sub_group(), v1886_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v2035_data = r9[4];
              float v2039_data = ir11[0];
              ir11[0] = (v2039_data + (v2035_data * (sycl::select_from_group(item.get_sub_group(), v1844_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v2045_data = ir11[1];
              ir11[1] = (v2045_data + (v2035_data * (sycl::select_from_group(item.get_sub_group(), v1850_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v2051_data = ir11[2];
              ir11[2] = (v2051_data + (v2035_data * (sycl::select_from_group(item.get_sub_group(), v1856_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v2057_data = ir11[3];
              ir11[3] = (v2057_data + (v2035_data * (sycl::select_from_group(item.get_sub_group(), v1862_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v2063_data = ir11[4];
              ir11[4] = (v2063_data + (v2035_data * (sycl::select_from_group(item.get_sub_group(), v1868_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v2069_data = ir11[5];
              ir11[5] = (v2069_data + (v2035_data * (sycl::select_from_group(item.get_sub_group(), v1874_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v2075_data = ir11[6];
              ir11[6] = (v2075_data + (v2035_data * (sycl::select_from_group(item.get_sub_group(), v1880_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v2081_data = ir11[7];
              ir11[7] = (v2081_data + (v2035_data * (sycl::select_from_group(item.get_sub_group(), v1886_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v2083_data = r9[5];
              float v2087_data = ir11[0];
              ir11[0] = (v2087_data + (v2083_data * (sycl::select_from_group(item.get_sub_group(), v1844_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v2093_data = ir11[1];
              ir11[1] = (v2093_data + (v2083_data * (sycl::select_from_group(item.get_sub_group(), v1850_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v2099_data = ir11[2];
              ir11[2] = (v2099_data + (v2083_data * (sycl::select_from_group(item.get_sub_group(), v1856_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v2105_data = ir11[3];
              ir11[3] = (v2105_data + (v2083_data * (sycl::select_from_group(item.get_sub_group(), v1862_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v2111_data = ir11[4];
              ir11[4] = (v2111_data + (v2083_data * (sycl::select_from_group(item.get_sub_group(), v1868_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v2117_data = ir11[5];
              ir11[5] = (v2117_data + (v2083_data * (sycl::select_from_group(item.get_sub_group(), v1874_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v2123_data = ir11[6];
              ir11[6] = (v2123_data + (v2083_data * (sycl::select_from_group(item.get_sub_group(), v1880_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v2129_data = ir11[7];
              ir11[7] = (v2129_data + (v2083_data * (sycl::select_from_group(item.get_sub_group(), v1886_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v2131_data = r9[6];
              float v2135_data = ir11[0];
              ir11[0] = (v2135_data + (v2131_data * (sycl::select_from_group(item.get_sub_group(), v1844_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v2141_data = ir11[1];
              ir11[1] = (v2141_data + (v2131_data * (sycl::select_from_group(item.get_sub_group(), v1850_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v2147_data = ir11[2];
              ir11[2] = (v2147_data + (v2131_data * (sycl::select_from_group(item.get_sub_group(), v1856_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v2153_data = ir11[3];
              ir11[3] = (v2153_data + (v2131_data * (sycl::select_from_group(item.get_sub_group(), v1862_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v2159_data = ir11[4];
              ir11[4] = (v2159_data + (v2131_data * (sycl::select_from_group(item.get_sub_group(), v1868_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v2165_data = ir11[5];
              ir11[5] = (v2165_data + (v2131_data * (sycl::select_from_group(item.get_sub_group(), v1874_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v2171_data = ir11[6];
              ir11[6] = (v2171_data + (v2131_data * (sycl::select_from_group(item.get_sub_group(), v1880_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v2177_data = ir11[7];
              ir11[7] = (v2177_data + (v2131_data * (sycl::select_from_group(item.get_sub_group(), v1886_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v2179_data = r9[7];
              float v2183_data = ir11[0];
              ir11[0] = (v2183_data + (v2179_data * (sycl::select_from_group(item.get_sub_group(), v1844_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v2189_data = ir11[1];
              ir11[1] = (v2189_data + (v2179_data * (sycl::select_from_group(item.get_sub_group(), v1850_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v2195_data = ir11[2];
              ir11[2] = (v2195_data + (v2179_data * (sycl::select_from_group(item.get_sub_group(), v1856_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v2201_data = ir11[3];
              ir11[3] = (v2201_data + (v2179_data * (sycl::select_from_group(item.get_sub_group(), v1862_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v2207_data = ir11[4];
              ir11[4] = (v2207_data + (v2179_data * (sycl::select_from_group(item.get_sub_group(), v1868_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v2213_data = ir11[5];
              ir11[5] = (v2213_data + (v2179_data * (sycl::select_from_group(item.get_sub_group(), v1874_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v2219_data = ir11[6];
              ir11[6] = (v2219_data + (v2179_data * (sycl::select_from_group(item.get_sub_group(), v1880_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v2225_data = ir11[7];
              ir11[7] = (v2225_data + (v2179_data * (sycl::select_from_group(item.get_sub_group(), v1886_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v2227_data = r9[8];
              float v2231_data = ir11[0];
              ir11[0] = (v2231_data + (v2227_data * (sycl::select_from_group(item.get_sub_group(), v1844_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v2237_data = ir11[1];
              ir11[1] = (v2237_data + (v2227_data * (sycl::select_from_group(item.get_sub_group(), v1850_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v2243_data = ir11[2];
              ir11[2] = (v2243_data + (v2227_data * (sycl::select_from_group(item.get_sub_group(), v1856_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v2249_data = ir11[3];
              ir11[3] = (v2249_data + (v2227_data * (sycl::select_from_group(item.get_sub_group(), v1862_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v2255_data = ir11[4];
              ir11[4] = (v2255_data + (v2227_data * (sycl::select_from_group(item.get_sub_group(), v1868_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v2261_data = ir11[5];
              ir11[5] = (v2261_data + (v2227_data * (sycl::select_from_group(item.get_sub_group(), v1874_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v2267_data = ir11[6];
              ir11[6] = (v2267_data + (v2227_data * (sycl::select_from_group(item.get_sub_group(), v1880_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v2273_data = ir11[7];
              ir11[7] = (v2273_data + (v2227_data * (sycl::select_from_group(item.get_sub_group(), v1886_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v2275_data = r9[9];
              float v2279_data = ir11[0];
              ir11[0] = (v2279_data + (v2275_data * (sycl::select_from_group(item.get_sub_group(), v1844_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v2285_data = ir11[1];
              ir11[1] = (v2285_data + (v2275_data * (sycl::select_from_group(item.get_sub_group(), v1850_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v2291_data = ir11[2];
              ir11[2] = (v2291_data + (v2275_data * (sycl::select_from_group(item.get_sub_group(), v1856_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v2297_data = ir11[3];
              ir11[3] = (v2297_data + (v2275_data * (sycl::select_from_group(item.get_sub_group(), v1862_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v2303_data = ir11[4];
              ir11[4] = (v2303_data + (v2275_data * (sycl::select_from_group(item.get_sub_group(), v1868_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v2309_data = ir11[5];
              ir11[5] = (v2309_data + (v2275_data * (sycl::select_from_group(item.get_sub_group(), v1874_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v2315_data = ir11[6];
              ir11[6] = (v2315_data + (v2275_data * (sycl::select_from_group(item.get_sub_group(), v1880_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v2321_data = ir11[7];
              ir11[7] = (v2321_data + (v2275_data * (sycl::select_from_group(item.get_sub_group(), v1886_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v2323_data = r9[10];
              float v2327_data = ir11[0];
              ir11[0] = (v2327_data + (v2323_data * (sycl::select_from_group(item.get_sub_group(), v1844_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v2333_data = ir11[1];
              ir11[1] = (v2333_data + (v2323_data * (sycl::select_from_group(item.get_sub_group(), v1850_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v2339_data = ir11[2];
              ir11[2] = (v2339_data + (v2323_data * (sycl::select_from_group(item.get_sub_group(), v1856_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v2345_data = ir11[3];
              ir11[3] = (v2345_data + (v2323_data * (sycl::select_from_group(item.get_sub_group(), v1862_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v2351_data = ir11[4];
              ir11[4] = (v2351_data + (v2323_data * (sycl::select_from_group(item.get_sub_group(), v1868_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v2357_data = ir11[5];
              ir11[5] = (v2357_data + (v2323_data * (sycl::select_from_group(item.get_sub_group(), v1874_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v2363_data = ir11[6];
              ir11[6] = (v2363_data + (v2323_data * (sycl::select_from_group(item.get_sub_group(), v1880_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v2369_data = ir11[7];
              ir11[7] = (v2369_data + (v2323_data * (sycl::select_from_group(item.get_sub_group(), v1886_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v2371_data = r9[11];
              float v2375_data = ir11[0];
              ir11[0] = (v2375_data + (v2371_data * (sycl::select_from_group(item.get_sub_group(), v1844_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v2381_data = ir11[1];
              ir11[1] = (v2381_data + (v2371_data * (sycl::select_from_group(item.get_sub_group(), v1850_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v2387_data = ir11[2];
              ir11[2] = (v2387_data + (v2371_data * (sycl::select_from_group(item.get_sub_group(), v1856_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v2393_data = ir11[3];
              ir11[3] = (v2393_data + (v2371_data * (sycl::select_from_group(item.get_sub_group(), v1862_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v2399_data = ir11[4];
              ir11[4] = (v2399_data + (v2371_data * (sycl::select_from_group(item.get_sub_group(), v1868_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v2405_data = ir11[5];
              ir11[5] = (v2405_data + (v2371_data * (sycl::select_from_group(item.get_sub_group(), v1874_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v2411_data = ir11[6];
              ir11[6] = (v2411_data + (v2371_data * (sycl::select_from_group(item.get_sub_group(), v1880_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v2417_data = ir11[7];
              ir11[7] = (v2417_data + (v2371_data * (sycl::select_from_group(item.get_sub_group(), v1886_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              // r11 = ir11 + r8
              if (v30_g) {
                #pragma unroll
                for (int32_t v2419_n1 = 0; v2419_n1 < 8; ++v2419_n1) {
                  float v2421_data = ir11[v2419_n1];
                  float v2422_data = r8[v2419_n1];
                  r11[v2419_n1] = (v2422_data + v2421_data);
                }
              }
              // glb_m0 = store{r>g}(r11);
              if (v30_g) {
                #pragma unroll
                for (int32_t v2424_i1 = 0; v2424_i1 < 8; ++v2424_i1) {
                  float v2426_data = r11[v2424_i1];
                  glb_m0[(v29_lead + (v2424_i1 * 12))] = v2426_data;
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

