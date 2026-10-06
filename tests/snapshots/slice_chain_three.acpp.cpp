// === base name ===
kernel_95cde8168b56786d

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_95cde8168b56786d = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_95cde8168b56786d(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_95cde8168b56786d(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_95cde8168b56786d(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_95cde8168b56786d(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_95cde8168b56786d(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_95cde8168b56786d(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, m3, m3_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_95cde8168b56786d(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, const float * m3, size_t m3_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 32×32(12×6) {0..12}×{0..6} strided
        //   m1 32×32(6×6) {0..6}×{0..6} strided
        //   m2 32×32(12×6) {0..12}×{0..6} strided
        //   m3 32×32(12×12) {0..12}×{0..12} strided
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   m2[i,j] = m3[i,k] × t0[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"B","bbox":[[0,0],[12,6]],"name":"m0","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[6,6]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"D","bbox":[[0,0],[12,6]],"name":"m2","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,12]],"name":"m3","ordered":false,"parts":1,"shape":[32,32],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[12,6]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,6]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,6]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[6,6]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[32,32]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,6]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[32,32]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,12]],"is_tmp":false,"name":"m3","offset":[0,0],"shape":[32,32]},{"addressing":"pointer_based","bbox":[[0,0],[12,6]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[12,6]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v9_batchId0 * 36 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v9_batchId0 * 72 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v9_batchId0 * 144 + 0 + m3_extraOffset];
              float r0[6]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v24_lead = item.get_local_id(2) % 16;
              bool v25_g = v24_lead < 12;
              if (v25_g) {
                #pragma unroll
                for (int32_t v26_i1 = 0; v26_i1 < 6; ++v26_i1) {
                  float v31_data = glb_m0[(v24_lead + (v26_i1 * 12))];
                  r0[v26_i1] = v31_data;
                }
              }
              float r1[6]{};
              // r1 = load{g>r}(glb_m1);
              if (v24_lead < 6) {
                #pragma unroll
                for (int32_t v35_i1 = 0; v35_i1 < 6; ++v35_i1) {
                  float v40_data = glb_m1[(v24_lead + (v35_i1 * 6))];
                  r1[v35_i1] = v40_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r3[12]{};
              // r3 = load{g>r}(glb_m3);
              if (v25_g) {
                #pragma unroll
                for (int32_t v43_i1 = 0; v43_i1 < 12; ++v43_i1) {
                  float v48_data = glb_m3[(v24_lead + (v43_i1 * 12))];
                  r3[v43_i1] = v48_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m1););
              float r2[6]{};
              // r2 = +(r0 * r1) + None
              // [(0, 12), (0, 6)] [(0, 6)]
              float v51_data = r0[0];
              float v52_data = r1[0];
              float v55_data = r2[0];
              r2[0] = (v55_data + (v51_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v58_data = r1[1];
              float v61_data = r2[1];
              r2[1] = (v61_data + (v51_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v64_data = r1[2];
              float v67_data = r2[2];
              r2[2] = (v67_data + (v51_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v70_data = r1[3];
              float v73_data = r2[3];
              r2[3] = (v73_data + (v51_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v76_data = r1[4];
              float v79_data = r2[4];
              r2[4] = (v79_data + (v51_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v82_data = r1[5];
              float v85_data = r2[5];
              r2[5] = (v85_data + (v51_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v87_data = r0[1];
              float v91_data = r2[0];
              r2[0] = (v91_data + (v87_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v97_data = r2[1];
              r2[1] = (v97_data + (v87_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v103_data = r2[2];
              r2[2] = (v103_data + (v87_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v109_data = r2[3];
              r2[3] = (v109_data + (v87_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v115_data = r2[4];
              r2[4] = (v115_data + (v87_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v121_data = r2[5];
              r2[5] = (v121_data + (v87_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v123_data = r0[2];
              float v127_data = r2[0];
              r2[0] = (v127_data + (v123_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v133_data = r2[1];
              r2[1] = (v133_data + (v123_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v139_data = r2[2];
              r2[2] = (v139_data + (v123_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v145_data = r2[3];
              r2[3] = (v145_data + (v123_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v151_data = r2[4];
              r2[4] = (v151_data + (v123_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v157_data = r2[5];
              r2[5] = (v157_data + (v123_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v159_data = r0[3];
              float v163_data = r2[0];
              r2[0] = (v163_data + (v159_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v169_data = r2[1];
              r2[1] = (v169_data + (v159_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v175_data = r2[2];
              r2[2] = (v175_data + (v159_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v181_data = r2[3];
              r2[3] = (v181_data + (v159_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v187_data = r2[4];
              r2[4] = (v187_data + (v159_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v193_data = r2[5];
              r2[5] = (v193_data + (v159_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v195_data = r0[4];
              float v199_data = r2[0];
              r2[0] = (v199_data + (v195_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v205_data = r2[1];
              r2[1] = (v205_data + (v195_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v211_data = r2[2];
              r2[2] = (v211_data + (v195_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v217_data = r2[3];
              r2[3] = (v217_data + (v195_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v223_data = r2[4];
              r2[4] = (v223_data + (v195_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v229_data = r2[5];
              r2[5] = (v229_data + (v195_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v231_data = r0[5];
              float v235_data = r2[0];
              r2[0] = (v235_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v241_data = r2[1];
              r2[1] = (v241_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v247_data = r2[2];
              r2[2] = (v247_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v253_data = r2[3];
              r2[3] = (v253_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v259_data = r2[4];
              r2[4] = (v259_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v265_data = r2[5];
              r2[5] = (v265_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              // wait(r3 = load{g>r}(glb_m3););
              float r4[6]{};
              // ir4 = +(r3 * r2)
              // [(0, 12), (0, 6)] [(0, 12)]
              float ir4[6]{};
              float v269_data = r3[0];
              float v270_data = r2[0];
              float v273_data = ir4[0];
              ir4[0] = (v273_data + (v269_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v276_data = r2[1];
              float v279_data = ir4[1];
              ir4[1] = (v279_data + (v269_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v282_data = r2[2];
              float v285_data = ir4[2];
              ir4[2] = (v285_data + (v269_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v288_data = r2[3];
              float v291_data = ir4[3];
              ir4[3] = (v291_data + (v269_data * (sycl::select_from_group(item.get_sub_group(), v288_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v294_data = r2[4];
              float v297_data = ir4[4];
              ir4[4] = (v297_data + (v269_data * (sycl::select_from_group(item.get_sub_group(), v294_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v300_data = r2[5];
              float v303_data = ir4[5];
              ir4[5] = (v303_data + (v269_data * (sycl::select_from_group(item.get_sub_group(), v300_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v305_data = r3[1];
              float v309_data = ir4[0];
              ir4[0] = (v309_data + (v305_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v315_data = ir4[1];
              ir4[1] = (v315_data + (v305_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v321_data = ir4[2];
              ir4[2] = (v321_data + (v305_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v327_data = ir4[3];
              ir4[3] = (v327_data + (v305_data * (sycl::select_from_group(item.get_sub_group(), v288_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v333_data = ir4[4];
              ir4[4] = (v333_data + (v305_data * (sycl::select_from_group(item.get_sub_group(), v294_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v339_data = ir4[5];
              ir4[5] = (v339_data + (v305_data * (sycl::select_from_group(item.get_sub_group(), v300_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v341_data = r3[2];
              float v345_data = ir4[0];
              ir4[0] = (v345_data + (v341_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v351_data = ir4[1];
              ir4[1] = (v351_data + (v341_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v357_data = ir4[2];
              ir4[2] = (v357_data + (v341_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v363_data = ir4[3];
              ir4[3] = (v363_data + (v341_data * (sycl::select_from_group(item.get_sub_group(), v288_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v369_data = ir4[4];
              ir4[4] = (v369_data + (v341_data * (sycl::select_from_group(item.get_sub_group(), v294_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v375_data = ir4[5];
              ir4[5] = (v375_data + (v341_data * (sycl::select_from_group(item.get_sub_group(), v300_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v377_data = r3[3];
              float v381_data = ir4[0];
              ir4[0] = (v381_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v387_data = ir4[1];
              ir4[1] = (v387_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v393_data = ir4[2];
              ir4[2] = (v393_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v399_data = ir4[3];
              ir4[3] = (v399_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v288_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v405_data = ir4[4];
              ir4[4] = (v405_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v294_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v411_data = ir4[5];
              ir4[5] = (v411_data + (v377_data * (sycl::select_from_group(item.get_sub_group(), v300_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v413_data = r3[4];
              float v417_data = ir4[0];
              ir4[0] = (v417_data + (v413_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v423_data = ir4[1];
              ir4[1] = (v423_data + (v413_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v429_data = ir4[2];
              ir4[2] = (v429_data + (v413_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v435_data = ir4[3];
              ir4[3] = (v435_data + (v413_data * (sycl::select_from_group(item.get_sub_group(), v288_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v441_data = ir4[4];
              ir4[4] = (v441_data + (v413_data * (sycl::select_from_group(item.get_sub_group(), v294_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v447_data = ir4[5];
              ir4[5] = (v447_data + (v413_data * (sycl::select_from_group(item.get_sub_group(), v300_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v449_data = r3[5];
              float v453_data = ir4[0];
              ir4[0] = (v453_data + (v449_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v459_data = ir4[1];
              ir4[1] = (v459_data + (v449_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v465_data = ir4[2];
              ir4[2] = (v465_data + (v449_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v471_data = ir4[3];
              ir4[3] = (v471_data + (v449_data * (sycl::select_from_group(item.get_sub_group(), v288_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v477_data = ir4[4];
              ir4[4] = (v477_data + (v449_data * (sycl::select_from_group(item.get_sub_group(), v294_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v483_data = ir4[5];
              ir4[5] = (v483_data + (v449_data * (sycl::select_from_group(item.get_sub_group(), v300_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v485_data = r3[6];
              float v489_data = ir4[0];
              ir4[0] = (v489_data + (v485_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v495_data = ir4[1];
              ir4[1] = (v495_data + (v485_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v501_data = ir4[2];
              ir4[2] = (v501_data + (v485_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v507_data = ir4[3];
              ir4[3] = (v507_data + (v485_data * (sycl::select_from_group(item.get_sub_group(), v288_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v513_data = ir4[4];
              ir4[4] = (v513_data + (v485_data * (sycl::select_from_group(item.get_sub_group(), v294_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v519_data = ir4[5];
              ir4[5] = (v519_data + (v485_data * (sycl::select_from_group(item.get_sub_group(), v300_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v521_data = r3[7];
              float v525_data = ir4[0];
              ir4[0] = (v525_data + (v521_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v531_data = ir4[1];
              ir4[1] = (v531_data + (v521_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v537_data = ir4[2];
              ir4[2] = (v537_data + (v521_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v543_data = ir4[3];
              ir4[3] = (v543_data + (v521_data * (sycl::select_from_group(item.get_sub_group(), v288_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v549_data = ir4[4];
              ir4[4] = (v549_data + (v521_data * (sycl::select_from_group(item.get_sub_group(), v294_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v555_data = ir4[5];
              ir4[5] = (v555_data + (v521_data * (sycl::select_from_group(item.get_sub_group(), v300_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v557_data = r3[8];
              float v561_data = ir4[0];
              ir4[0] = (v561_data + (v557_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v567_data = ir4[1];
              ir4[1] = (v567_data + (v557_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v573_data = ir4[2];
              ir4[2] = (v573_data + (v557_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v579_data = ir4[3];
              ir4[3] = (v579_data + (v557_data * (sycl::select_from_group(item.get_sub_group(), v288_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v585_data = ir4[4];
              ir4[4] = (v585_data + (v557_data * (sycl::select_from_group(item.get_sub_group(), v294_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v591_data = ir4[5];
              ir4[5] = (v591_data + (v557_data * (sycl::select_from_group(item.get_sub_group(), v300_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v593_data = r3[9];
              float v597_data = ir4[0];
              ir4[0] = (v597_data + (v593_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v603_data = ir4[1];
              ir4[1] = (v603_data + (v593_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v609_data = ir4[2];
              ir4[2] = (v609_data + (v593_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v615_data = ir4[3];
              ir4[3] = (v615_data + (v593_data * (sycl::select_from_group(item.get_sub_group(), v288_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v621_data = ir4[4];
              ir4[4] = (v621_data + (v593_data * (sycl::select_from_group(item.get_sub_group(), v294_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v627_data = ir4[5];
              ir4[5] = (v627_data + (v593_data * (sycl::select_from_group(item.get_sub_group(), v300_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v629_data = r3[10];
              float v633_data = ir4[0];
              ir4[0] = (v633_data + (v629_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v639_data = ir4[1];
              ir4[1] = (v639_data + (v629_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v645_data = ir4[2];
              ir4[2] = (v645_data + (v629_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v651_data = ir4[3];
              ir4[3] = (v651_data + (v629_data * (sycl::select_from_group(item.get_sub_group(), v288_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v657_data = ir4[4];
              ir4[4] = (v657_data + (v629_data * (sycl::select_from_group(item.get_sub_group(), v294_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v663_data = ir4[5];
              ir4[5] = (v663_data + (v629_data * (sycl::select_from_group(item.get_sub_group(), v300_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v665_data = r3[11];
              float v669_data = ir4[0];
              ir4[0] = (v669_data + (v665_data * (sycl::select_from_group(item.get_sub_group(), v270_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v675_data = ir4[1];
              ir4[1] = (v675_data + (v665_data * (sycl::select_from_group(item.get_sub_group(), v276_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v681_data = ir4[2];
              ir4[2] = (v681_data + (v665_data * (sycl::select_from_group(item.get_sub_group(), v282_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v687_data = ir4[3];
              ir4[3] = (v687_data + (v665_data * (sycl::select_from_group(item.get_sub_group(), v288_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v693_data = ir4[4];
              ir4[4] = (v693_data + (v665_data * (sycl::select_from_group(item.get_sub_group(), v294_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v699_data = ir4[5];
              ir4[5] = (v699_data + (v665_data * (sycl::select_from_group(item.get_sub_group(), v300_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              // r4 = ir4
              if (v25_g) {
                #pragma unroll
                for (int32_t v701_n1 = 0; v701_n1 < 6; ++v701_n1) {
                  float v703_data = ir4[v701_n1];
                  r4[v701_n1] = v703_data;
                }
              }
              // glb_m2 = store{r>g}(r4);
              if (v25_g) {
                #pragma unroll
                for (int32_t v704_i1 = 0; v704_i1 < 6; ++v704_i1) {
                  float v706_data = r4[v704_i1];
                  glb_m2[(v24_lead + (v704_i1 * 12))] = v706_data;
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

