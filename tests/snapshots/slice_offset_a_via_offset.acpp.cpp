// === base name ===
kernel_c80231f2f33075b8

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_c80231f2f33075b8 = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_c80231f2f33075b8(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_c80231f2f33075b8(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_c80231f2f33075b8(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_c80231f2f33075b8(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_c80231f2f33075b8(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_c80231f2f33075b8(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_c80231f2f33075b8(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 12×8(12×8) {0..12}×{0..8} strided
        //   m1 32×16(32×16) {0..32}×{0..16} strided
        //   m2 16×8(16×8) {0..16}×{0..8} strided
        // operations:
        //   m0[i,j] = m1[i,k]@{4..16}×{0..16} × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,8]],"name":"m0","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[32,16]],"name":"m1","ordered":false,"parts":1,"shape":[32,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m1","offset":[4,0],"shape":[32,16]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v9_batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v9_batchId0 * 512 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v9_batchId0 * 128 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v23_lead = item.get_local_id(2) % 16;
              bool v24_g = v23_lead < 12;
              if (v24_g) {
                int32_t v28_off = v23_lead + 4;
                #pragma unroll
                for (int32_t v25_i1 = 0; v25_i1 < 16; ++v25_i1) {
                  float v31_data = glb_m1[(v28_off + (v25_i1 * 32))];
                  r0[v25_i1] = v31_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v34_i0 = 0; v34_i0 < 1; ++v34_i0) {
                int32_t v37_lead = v23_lead + (v34_i0 * 16);
                #pragma unroll
                for (int32_t v35_i1 = 0; v35_i1 < 8; ++v35_i1) {
                  float v40_data = glb_m2[(v37_lead + (v35_i1 * 16))];
                  r1[(v34_i0 + v35_i1)] = v40_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // ir2 = +(r0 * r1)
              // [(0, 12), (0, 8)] [(0, 16)]
              float ir2[8]{};
              float v44_data = r0[0];
              float v45_data = r1[0];
              float v48_data = ir2[0];
              ir2[0] = (v48_data + (v44_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v51_data = r1[1];
              float v54_data = ir2[1];
              ir2[1] = (v54_data + (v44_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v57_data = r1[2];
              float v60_data = ir2[2];
              ir2[2] = (v60_data + (v44_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v63_data = r1[3];
              float v66_data = ir2[3];
              ir2[3] = (v66_data + (v44_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v69_data = r1[4];
              float v72_data = ir2[4];
              ir2[4] = (v72_data + (v44_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v75_data = r1[5];
              float v78_data = ir2[5];
              ir2[5] = (v78_data + (v44_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v81_data = r1[6];
              float v84_data = ir2[6];
              ir2[6] = (v84_data + (v44_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v87_data = r1[7];
              float v90_data = ir2[7];
              ir2[7] = (v90_data + (v44_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v92_data = r0[1];
              float v96_data = ir2[0];
              ir2[0] = (v96_data + (v92_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v102_data = ir2[1];
              ir2[1] = (v102_data + (v92_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v108_data = ir2[2];
              ir2[2] = (v108_data + (v92_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v114_data = ir2[3];
              ir2[3] = (v114_data + (v92_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v120_data = ir2[4];
              ir2[4] = (v120_data + (v92_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v126_data = ir2[5];
              ir2[5] = (v126_data + (v92_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v132_data = ir2[6];
              ir2[6] = (v132_data + (v92_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v138_data = ir2[7];
              ir2[7] = (v138_data + (v92_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v140_data = r0[2];
              float v144_data = ir2[0];
              ir2[0] = (v144_data + (v140_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v150_data = ir2[1];
              ir2[1] = (v150_data + (v140_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v156_data = ir2[2];
              ir2[2] = (v156_data + (v140_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v162_data = ir2[3];
              ir2[3] = (v162_data + (v140_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v168_data = ir2[4];
              ir2[4] = (v168_data + (v140_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v174_data = ir2[5];
              ir2[5] = (v174_data + (v140_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v180_data = ir2[6];
              ir2[6] = (v180_data + (v140_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v186_data = ir2[7];
              ir2[7] = (v186_data + (v140_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v188_data = r0[3];
              float v192_data = ir2[0];
              ir2[0] = (v192_data + (v188_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v198_data = ir2[1];
              ir2[1] = (v198_data + (v188_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v204_data = ir2[2];
              ir2[2] = (v204_data + (v188_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v210_data = ir2[3];
              ir2[3] = (v210_data + (v188_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v216_data = ir2[4];
              ir2[4] = (v216_data + (v188_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v222_data = ir2[5];
              ir2[5] = (v222_data + (v188_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v228_data = ir2[6];
              ir2[6] = (v228_data + (v188_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v234_data = ir2[7];
              ir2[7] = (v234_data + (v188_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v236_data = r0[4];
              float v240_data = ir2[0];
              ir2[0] = (v240_data + (v236_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v246_data = ir2[1];
              ir2[1] = (v246_data + (v236_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v252_data = ir2[2];
              ir2[2] = (v252_data + (v236_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v258_data = ir2[3];
              ir2[3] = (v258_data + (v236_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v264_data = ir2[4];
              ir2[4] = (v264_data + (v236_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v270_data = ir2[5];
              ir2[5] = (v270_data + (v236_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v276_data = ir2[6];
              ir2[6] = (v276_data + (v236_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v282_data = ir2[7];
              ir2[7] = (v282_data + (v236_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v284_data = r0[5];
              float v288_data = ir2[0];
              ir2[0] = (v288_data + (v284_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v294_data = ir2[1];
              ir2[1] = (v294_data + (v284_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v300_data = ir2[2];
              ir2[2] = (v300_data + (v284_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v306_data = ir2[3];
              ir2[3] = (v306_data + (v284_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v312_data = ir2[4];
              ir2[4] = (v312_data + (v284_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v318_data = ir2[5];
              ir2[5] = (v318_data + (v284_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v324_data = ir2[6];
              ir2[6] = (v324_data + (v284_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v330_data = ir2[7];
              ir2[7] = (v330_data + (v284_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v332_data = r0[6];
              float v336_data = ir2[0];
              ir2[0] = (v336_data + (v332_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v342_data = ir2[1];
              ir2[1] = (v342_data + (v332_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v348_data = ir2[2];
              ir2[2] = (v348_data + (v332_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v354_data = ir2[3];
              ir2[3] = (v354_data + (v332_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v360_data = ir2[4];
              ir2[4] = (v360_data + (v332_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v366_data = ir2[5];
              ir2[5] = (v366_data + (v332_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v372_data = ir2[6];
              ir2[6] = (v372_data + (v332_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v378_data = ir2[7];
              ir2[7] = (v378_data + (v332_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v380_data = r0[7];
              float v384_data = ir2[0];
              ir2[0] = (v384_data + (v380_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v390_data = ir2[1];
              ir2[1] = (v390_data + (v380_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v396_data = ir2[2];
              ir2[2] = (v396_data + (v380_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v402_data = ir2[3];
              ir2[3] = (v402_data + (v380_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v408_data = ir2[4];
              ir2[4] = (v408_data + (v380_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v414_data = ir2[5];
              ir2[5] = (v414_data + (v380_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v420_data = ir2[6];
              ir2[6] = (v420_data + (v380_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v426_data = ir2[7];
              ir2[7] = (v426_data + (v380_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v428_data = r0[8];
              float v432_data = ir2[0];
              ir2[0] = (v432_data + (v428_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v438_data = ir2[1];
              ir2[1] = (v438_data + (v428_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v444_data = ir2[2];
              ir2[2] = (v444_data + (v428_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v450_data = ir2[3];
              ir2[3] = (v450_data + (v428_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v456_data = ir2[4];
              ir2[4] = (v456_data + (v428_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v462_data = ir2[5];
              ir2[5] = (v462_data + (v428_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v468_data = ir2[6];
              ir2[6] = (v468_data + (v428_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v474_data = ir2[7];
              ir2[7] = (v474_data + (v428_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v476_data = r0[9];
              float v480_data = ir2[0];
              ir2[0] = (v480_data + (v476_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v486_data = ir2[1];
              ir2[1] = (v486_data + (v476_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v492_data = ir2[2];
              ir2[2] = (v492_data + (v476_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v498_data = ir2[3];
              ir2[3] = (v498_data + (v476_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v504_data = ir2[4];
              ir2[4] = (v504_data + (v476_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v510_data = ir2[5];
              ir2[5] = (v510_data + (v476_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v516_data = ir2[6];
              ir2[6] = (v516_data + (v476_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v522_data = ir2[7];
              ir2[7] = (v522_data + (v476_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v524_data = r0[10];
              float v528_data = ir2[0];
              ir2[0] = (v528_data + (v524_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v534_data = ir2[1];
              ir2[1] = (v534_data + (v524_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v540_data = ir2[2];
              ir2[2] = (v540_data + (v524_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v546_data = ir2[3];
              ir2[3] = (v546_data + (v524_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v552_data = ir2[4];
              ir2[4] = (v552_data + (v524_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v558_data = ir2[5];
              ir2[5] = (v558_data + (v524_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v564_data = ir2[6];
              ir2[6] = (v564_data + (v524_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v570_data = ir2[7];
              ir2[7] = (v570_data + (v524_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v572_data = r0[11];
              float v576_data = ir2[0];
              ir2[0] = (v576_data + (v572_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v582_data = ir2[1];
              ir2[1] = (v582_data + (v572_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v588_data = ir2[2];
              ir2[2] = (v588_data + (v572_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v594_data = ir2[3];
              ir2[3] = (v594_data + (v572_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v600_data = ir2[4];
              ir2[4] = (v600_data + (v572_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v606_data = ir2[5];
              ir2[5] = (v606_data + (v572_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v612_data = ir2[6];
              ir2[6] = (v612_data + (v572_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v618_data = ir2[7];
              ir2[7] = (v618_data + (v572_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v620_data = r0[12];
              float v624_data = ir2[0];
              ir2[0] = (v624_data + (v620_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v630_data = ir2[1];
              ir2[1] = (v630_data + (v620_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v636_data = ir2[2];
              ir2[2] = (v636_data + (v620_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v642_data = ir2[3];
              ir2[3] = (v642_data + (v620_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v648_data = ir2[4];
              ir2[4] = (v648_data + (v620_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v654_data = ir2[5];
              ir2[5] = (v654_data + (v620_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v660_data = ir2[6];
              ir2[6] = (v660_data + (v620_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v666_data = ir2[7];
              ir2[7] = (v666_data + (v620_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v668_data = r0[13];
              float v672_data = ir2[0];
              ir2[0] = (v672_data + (v668_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v678_data = ir2[1];
              ir2[1] = (v678_data + (v668_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v684_data = ir2[2];
              ir2[2] = (v684_data + (v668_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v690_data = ir2[3];
              ir2[3] = (v690_data + (v668_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v696_data = ir2[4];
              ir2[4] = (v696_data + (v668_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v702_data = ir2[5];
              ir2[5] = (v702_data + (v668_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v708_data = ir2[6];
              ir2[6] = (v708_data + (v668_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v714_data = ir2[7];
              ir2[7] = (v714_data + (v668_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v716_data = r0[14];
              float v720_data = ir2[0];
              ir2[0] = (v720_data + (v716_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v726_data = ir2[1];
              ir2[1] = (v726_data + (v716_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v732_data = ir2[2];
              ir2[2] = (v732_data + (v716_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v738_data = ir2[3];
              ir2[3] = (v738_data + (v716_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v744_data = ir2[4];
              ir2[4] = (v744_data + (v716_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v750_data = ir2[5];
              ir2[5] = (v750_data + (v716_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v756_data = ir2[6];
              ir2[6] = (v756_data + (v716_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v762_data = ir2[7];
              ir2[7] = (v762_data + (v716_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v764_data = r0[15];
              float v768_data = ir2[0];
              ir2[0] = (v768_data + (v764_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v774_data = ir2[1];
              ir2[1] = (v774_data + (v764_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v780_data = ir2[2];
              ir2[2] = (v780_data + (v764_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v786_data = ir2[3];
              ir2[3] = (v786_data + (v764_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v792_data = ir2[4];
              ir2[4] = (v792_data + (v764_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v798_data = ir2[5];
              ir2[5] = (v798_data + (v764_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v804_data = ir2[6];
              ir2[6] = (v804_data + (v764_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v810_data = ir2[7];
              ir2[7] = (v810_data + (v764_data * (sycl::select_from_group(item.get_sub_group(), v87_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              // r2 = ir2
              if (v24_g) {
                #pragma unroll
                for (int32_t v812_n1 = 0; v812_n1 < 8; ++v812_n1) {
                  float v814_data = ir2[v812_n1];
                  r2[v812_n1] = v814_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v24_g) {
                #pragma unroll
                for (int32_t v815_i1 = 0; v815_i1 < 8; ++v815_i1) {
                  float v817_data = r2[v815_i1];
                  glb_m0[(v23_lead + (v815_i1 * 12))] = v817_data;
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

