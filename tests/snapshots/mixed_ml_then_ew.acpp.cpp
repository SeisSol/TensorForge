// === base name ===
kernel_926b3c8fb3eb29dd

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_926b3c8fb3eb29dd = {{8, 2, 1}, 8, 8, 1, 2, 64, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_926b3c8fb3eb29dd(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_926b3c8fb3eb29dd(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_926b3c8fb3eb29dd(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (8, 2, 1);
  tensorforge::LaunchConfig config{};
  config.grid[0] = (numElements0 + 2 - 1) / 2;
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 8;
  config.block[1] = 2;
  config.block[2] = 1;
  config.sharedMemBytes = 16 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_926b3c8fb3eb29dd(const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_926b3c8fb3eb29dd(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_926b3c8fb3eb29dd(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_926b3c8fb3eb29dd(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (16, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 8 lanes x 2 per block = block 8x2x1, 64 B shared, occupancy grid
        // operands:
        //   m0 8×8(8×8) {0..8}×{0..8} strided
        //   m1 8×8(8×8) {0..8}×{0..8} strided
        //   m2 8×8(8×8) {0..8}×{0..8} strided
        // operations:
        //   t0[i,j] = m0[i,k] × m1[k,j]
        //   C = abs(TMP)
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":8,"block":[8,2,1],"cooperative":false,"lead_width":1,"mults_per_block":2,"persistent":true,"sections":[{"barrier":false,"mults_per_block":2,"shared_elements":16}],"shared_bytes":64,"shared_elements":16,"threads_per_mult":8},"loops":[],"operands":[{"addressing":"strided","alias":"A","bbox":[[0,0],[8,8]],"name":"m0","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[8,8]],"name":"m1","ordered":false,"parts":1,"shape":[8,8],"variant":false},{"addressing":"strided","alias":"C","bbox":[[0,0],[8,8]],"name":"m2","ordered":false,"parts":1,"shape":[8,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[8,8]},{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[8,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]},{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[8,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[8,8]},"kind":"elementwise","op":"ABS","ops":[{"addressing":"pointer_based","bbox":[[0,0],[8,8]],"is_tmp":true,"name":"t0","offset":[0,0],"shape":[8,8]}],"permute":[[0,1]],"scalars":[],"target":[[0,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[8 * item.get_local_id(1) + 0];
          size_t v8_batchIdLane0 = item.get_local_id(1) % 2;
          int32_t v25_lead = item.get_local_id(2) % 8;
          for (size_t v9_batchIdGroup0 = (item.get_local_id(1) - item.get_local_id(1) % 2) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)); v9_batchIdGroup0 < numElements0; v9_batchIdGroup0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_row = v9_batchIdGroup0 + v8_batchIdLane0;
            const bool batchIdActive0 = v10_row < numElements0 && (flags0 == nullptr || static_cast<bool>(flags0[v10_row]));
            size_t v12_batchId0 = batchIdActive0 ? v10_row : v9_batchIdGroup0;
            size_t v13_ahead1 = v12_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v15_batchId1 = (v13_ahead1 < numElements0) ? v13_ahead1 : v12_batchId0;
            const float *const __restrict__ glb_m0 = &m0[v12_batchId0 * 64 + 0 + m0_extraOffset];
            const float *const __restrict__ glb_m1 = &m1[v12_batchId0 * 64 + 0 + m1_extraOffset];
            float *const __restrict__ glb_m2 = &m2[v12_batchId0 * 64 + 0 + m2_extraOffset];
            float r0[8]{};
            // r0 = load{g>r}(glb_m0);
            #pragma unroll
            for (int32_t v26_i0 = 0; v26_i0 < 1; ++v26_i0) {
              int32_t v29_lead = v25_lead + (v26_i0 * 8);
              #pragma unroll
              for (int32_t v27_i1 = 0; v27_i1 < 8; ++v27_i1) {
                float v32_data = glb_m0[(v29_lead + (v27_i1 * 8))];
                r0[(v26_i0 + v27_i1)] = v32_data;
              }
            }
            float r1[8]{};
            // r1 = load{g>r}(glb_m1);
            #pragma unroll
            for (int32_t v35_i0 = 0; v35_i0 < 1; ++v35_i0) {
              int32_t v38_lead = v25_lead + (v35_i0 * 8);
              #pragma unroll
              for (int32_t v36_i1 = 0; v36_i1 < 8; ++v36_i1) {
                float v41_data = glb_m1[(v38_lead + (v36_i1 * 8))];
                r1[(v35_i0 + v36_i1)] = v41_data;
              }
            }
            // wait(r0 = load{g>r}(glb_m0););
            // wait(r1 = load{g>r}(glb_m1););
            float r2[8]{};
            // r2 = +(r0 * r1) + None
            // [(0, 8), (0, 8)] [(0, 8)]
            float v44_data = r0[0];
            float v45_data = r1[0];
            float v48_data = r2[0];
            r2[0] = (v48_data + (v44_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v51_data = r1[1];
            float v54_data = r2[1];
            r2[1] = (v54_data + (v44_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v57_data = r1[2];
            float v60_data = r2[2];
            r2[2] = (v60_data + (v44_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v63_data = r1[3];
            float v66_data = r2[3];
            r2[3] = (v66_data + (v44_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v69_data = r1[4];
            float v72_data = r2[4];
            r2[4] = (v72_data + (v44_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v75_data = r1[5];
            float v78_data = r2[5];
            r2[5] = (v78_data + (v44_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v81_data = r1[6];
            float v84_data = r2[6];
            r2[6] = (v84_data + (v44_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v87_data = r1[7];
            float v90_data = r2[7];
            r2[7] = (v90_data + (v44_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (0)))));
            float v92_data = r0[1];
            float v96_data = r2[0];
            r2[0] = (v96_data + (v92_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v102_data = r2[1];
            r2[1] = (v102_data + (v92_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v108_data = r2[2];
            r2[2] = (v108_data + (v92_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v114_data = r2[3];
            r2[3] = (v114_data + (v92_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v120_data = r2[4];
            r2[4] = (v120_data + (v92_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v126_data = r2[5];
            r2[5] = (v126_data + (v92_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v132_data = r2[6];
            r2[6] = (v132_data + (v92_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v138_data = r2[7];
            r2[7] = (v138_data + (v92_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (1)))));
            float v140_data = r0[2];
            float v144_data = r2[0];
            r2[0] = (v144_data + (v140_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v150_data = r2[1];
            r2[1] = (v150_data + (v140_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v156_data = r2[2];
            r2[2] = (v156_data + (v140_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v162_data = r2[3];
            r2[3] = (v162_data + (v140_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v168_data = r2[4];
            r2[4] = (v168_data + (v140_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v174_data = r2[5];
            r2[5] = (v174_data + (v140_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v180_data = r2[6];
            r2[6] = (v180_data + (v140_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v186_data = r2[7];
            r2[7] = (v186_data + (v140_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (2)))));
            float v188_data = r0[3];
            float v192_data = r2[0];
            r2[0] = (v192_data + (v188_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v198_data = r2[1];
            r2[1] = (v198_data + (v188_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v204_data = r2[2];
            r2[2] = (v204_data + (v188_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v210_data = r2[3];
            r2[3] = (v210_data + (v188_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v216_data = r2[4];
            r2[4] = (v216_data + (v188_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v222_data = r2[5];
            r2[5] = (v222_data + (v188_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v228_data = r2[6];
            r2[6] = (v228_data + (v188_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v234_data = r2[7];
            r2[7] = (v234_data + (v188_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (3)))));
            float v236_data = r0[4];
            float v240_data = r2[0];
            r2[0] = (v240_data + (v236_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v246_data = r2[1];
            r2[1] = (v246_data + (v236_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v252_data = r2[2];
            r2[2] = (v252_data + (v236_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v258_data = r2[3];
            r2[3] = (v258_data + (v236_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v264_data = r2[4];
            r2[4] = (v264_data + (v236_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v270_data = r2[5];
            r2[5] = (v270_data + (v236_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v276_data = r2[6];
            r2[6] = (v276_data + (v236_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v282_data = r2[7];
            r2[7] = (v282_data + (v236_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (4)))));
            float v284_data = r0[5];
            float v288_data = r2[0];
            r2[0] = (v288_data + (v284_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v294_data = r2[1];
            r2[1] = (v294_data + (v284_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v300_data = r2[2];
            r2[2] = (v300_data + (v284_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v306_data = r2[3];
            r2[3] = (v306_data + (v284_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v312_data = r2[4];
            r2[4] = (v312_data + (v284_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v318_data = r2[5];
            r2[5] = (v318_data + (v284_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v324_data = r2[6];
            r2[6] = (v324_data + (v284_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v330_data = r2[7];
            r2[7] = (v330_data + (v284_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (5)))));
            float v332_data = r0[6];
            float v336_data = r2[0];
            r2[0] = (v336_data + (v332_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v342_data = r2[1];
            r2[1] = (v342_data + (v332_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v348_data = r2[2];
            r2[2] = (v348_data + (v332_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v354_data = r2[3];
            r2[3] = (v354_data + (v332_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v360_data = r2[4];
            r2[4] = (v360_data + (v332_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v366_data = r2[5];
            r2[5] = (v366_data + (v332_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v372_data = r2[6];
            r2[6] = (v372_data + (v332_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v378_data = r2[7];
            r2[7] = (v378_data + (v332_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (6)))));
            float v380_data = r0[7];
            float v384_data = r2[0];
            r2[0] = (v384_data + (v380_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v390_data = r2[1];
            r2[1] = (v390_data + (v380_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v396_data = r2[2];
            r2[2] = (v396_data + (v380_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v402_data = r2[3];
            r2[3] = (v402_data + (v380_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v408_data = r2[4];
            r2[4] = (v408_data + (v380_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v414_data = r2[5];
            r2[5] = (v414_data + (v380_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v420_data = r2[6];
            r2[6] = (v420_data + (v380_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            float v426_data = r2[7];
            r2[7] = (v426_data + (v380_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 8) * 8 + (7)))));
            // glb_m2 = abs(r2)
            #pragma unroll
            for (int32_t v428_k0 = 0; v428_k0 < 1; ++v428_k0) {
              #pragma unroll
              for (int32_t v429_k1 = 0; v429_k1 < 8; ++v429_k1) {
                float v431_data = r2[(v428_k0 + v429_k1)];
                float v432_e = sycl::fabs(v431_data);
                if (batchIdActive0) {
                  glb_m2[((v25_lead + (v428_k0 * 8)) + (v429_k1 * 8))] = v432_e;
                }
              }
            }
            item.barrier();
          }
        }
      });
    }
  });
}

