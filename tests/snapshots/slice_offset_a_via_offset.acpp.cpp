// === base name ===
kernel_ea99c05838c28d07

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_ea99c05838c28d07 = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_ea99c05838c28d07(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_ea99c05838c28d07(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_ea99c05838c28d07(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_ea99c05838c28d07(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_ea99c05838c28d07(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_ea99c05838c28d07(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_ea99c05838c28d07(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 12×8(12×8) {0..12}×{0..8} strided
        //   m1 32×16(32×16) {0..32}×{0..16} strided
        //   m2 16×8(16×8) {0..16}×{0..8} strided
        // operations:
        //   m0[i,j] = m1[i,k]@{4..16}×{0..16} × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,8]],"name":"m0","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[32,16]],"name":"m1","ordered":false,"parts":1,"shape":[32,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m1","offset":[4,0],"shape":[32,16]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
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
              const float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 512 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v3_batchId0 * 128 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v17_lead = item.get_local_id(2) % 16;
              bool v18_g = v17_lead < 12;
              if (v18_g) {
                int32_t v22_off = v17_lead + 4;
                #pragma unroll
                for (int32_t v19_i1 = 0; v19_i1 < 16; ++v19_i1) {
                  float v25_data = glb_m1[(v22_off + (v19_i1 * 32))];
                  r0[v19_i1] = v25_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v28_i0 = 0; v28_i0 < 1; ++v28_i0) {
                int32_t v31_lead = v17_lead + (v28_i0 * 16);
                #pragma unroll
                for (int32_t v29_i1 = 0; v29_i1 < 8; ++v29_i1) {
                  float v34_data = glb_m2[(v31_lead + (v29_i1 * 16))];
                  r1[(v28_i0 + v29_i1)] = v34_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 12), (0, 8)] [(0, 16)]
              float ir2[8]{};
              float v38_data = r0[0];
              float v39_data = r1[0];
              float v42_data = ir2[0];
              ir2[0] = (v42_data + (v38_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v45_data = r1[1];
              float v48_data = ir2[1];
              ir2[1] = (v48_data + (v38_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v51_data = r1[2];
              float v54_data = ir2[2];
              ir2[2] = (v54_data + (v38_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v57_data = r1[3];
              float v60_data = ir2[3];
              ir2[3] = (v60_data + (v38_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v63_data = r1[4];
              float v66_data = ir2[4];
              ir2[4] = (v66_data + (v38_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v69_data = r1[5];
              float v72_data = ir2[5];
              ir2[5] = (v72_data + (v38_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v75_data = r1[6];
              float v78_data = ir2[6];
              ir2[6] = (v78_data + (v38_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v81_data = r1[7];
              float v84_data = ir2[7];
              ir2[7] = (v84_data + (v38_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v86_data = r0[1];
              float v90_data = ir2[0];
              ir2[0] = (v90_data + (v86_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v96_data = ir2[1];
              ir2[1] = (v96_data + (v86_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v102_data = ir2[2];
              ir2[2] = (v102_data + (v86_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v108_data = ir2[3];
              ir2[3] = (v108_data + (v86_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v114_data = ir2[4];
              ir2[4] = (v114_data + (v86_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v120_data = ir2[5];
              ir2[5] = (v120_data + (v86_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v126_data = ir2[6];
              ir2[6] = (v126_data + (v86_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v132_data = ir2[7];
              ir2[7] = (v132_data + (v86_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v134_data = r0[2];
              float v138_data = ir2[0];
              ir2[0] = (v138_data + (v134_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v144_data = ir2[1];
              ir2[1] = (v144_data + (v134_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v150_data = ir2[2];
              ir2[2] = (v150_data + (v134_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v156_data = ir2[3];
              ir2[3] = (v156_data + (v134_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v162_data = ir2[4];
              ir2[4] = (v162_data + (v134_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v168_data = ir2[5];
              ir2[5] = (v168_data + (v134_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v174_data = ir2[6];
              ir2[6] = (v174_data + (v134_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v180_data = ir2[7];
              ir2[7] = (v180_data + (v134_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v182_data = r0[3];
              float v186_data = ir2[0];
              ir2[0] = (v186_data + (v182_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v192_data = ir2[1];
              ir2[1] = (v192_data + (v182_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v198_data = ir2[2];
              ir2[2] = (v198_data + (v182_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v204_data = ir2[3];
              ir2[3] = (v204_data + (v182_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v210_data = ir2[4];
              ir2[4] = (v210_data + (v182_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v216_data = ir2[5];
              ir2[5] = (v216_data + (v182_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v222_data = ir2[6];
              ir2[6] = (v222_data + (v182_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v228_data = ir2[7];
              ir2[7] = (v228_data + (v182_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v230_data = r0[4];
              float v234_data = ir2[0];
              ir2[0] = (v234_data + (v230_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v240_data = ir2[1];
              ir2[1] = (v240_data + (v230_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v246_data = ir2[2];
              ir2[2] = (v246_data + (v230_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v252_data = ir2[3];
              ir2[3] = (v252_data + (v230_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v258_data = ir2[4];
              ir2[4] = (v258_data + (v230_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v264_data = ir2[5];
              ir2[5] = (v264_data + (v230_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v270_data = ir2[6];
              ir2[6] = (v270_data + (v230_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v276_data = ir2[7];
              ir2[7] = (v276_data + (v230_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v278_data = r0[5];
              float v282_data = ir2[0];
              ir2[0] = (v282_data + (v278_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v288_data = ir2[1];
              ir2[1] = (v288_data + (v278_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v294_data = ir2[2];
              ir2[2] = (v294_data + (v278_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v300_data = ir2[3];
              ir2[3] = (v300_data + (v278_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v306_data = ir2[4];
              ir2[4] = (v306_data + (v278_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v312_data = ir2[5];
              ir2[5] = (v312_data + (v278_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v318_data = ir2[6];
              ir2[6] = (v318_data + (v278_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v324_data = ir2[7];
              ir2[7] = (v324_data + (v278_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v326_data = r0[6];
              float v330_data = ir2[0];
              ir2[0] = (v330_data + (v326_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v336_data = ir2[1];
              ir2[1] = (v336_data + (v326_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v342_data = ir2[2];
              ir2[2] = (v342_data + (v326_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v348_data = ir2[3];
              ir2[3] = (v348_data + (v326_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v354_data = ir2[4];
              ir2[4] = (v354_data + (v326_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v360_data = ir2[5];
              ir2[5] = (v360_data + (v326_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v366_data = ir2[6];
              ir2[6] = (v366_data + (v326_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v372_data = ir2[7];
              ir2[7] = (v372_data + (v326_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v374_data = r0[7];
              float v378_data = ir2[0];
              ir2[0] = (v378_data + (v374_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v384_data = ir2[1];
              ir2[1] = (v384_data + (v374_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v390_data = ir2[2];
              ir2[2] = (v390_data + (v374_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v396_data = ir2[3];
              ir2[3] = (v396_data + (v374_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v402_data = ir2[4];
              ir2[4] = (v402_data + (v374_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v408_data = ir2[5];
              ir2[5] = (v408_data + (v374_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v414_data = ir2[6];
              ir2[6] = (v414_data + (v374_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v420_data = ir2[7];
              ir2[7] = (v420_data + (v374_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v422_data = r0[8];
              float v426_data = ir2[0];
              ir2[0] = (v426_data + (v422_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v432_data = ir2[1];
              ir2[1] = (v432_data + (v422_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v438_data = ir2[2];
              ir2[2] = (v438_data + (v422_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v444_data = ir2[3];
              ir2[3] = (v444_data + (v422_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v450_data = ir2[4];
              ir2[4] = (v450_data + (v422_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v456_data = ir2[5];
              ir2[5] = (v456_data + (v422_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v462_data = ir2[6];
              ir2[6] = (v462_data + (v422_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v468_data = ir2[7];
              ir2[7] = (v468_data + (v422_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v470_data = r0[9];
              float v474_data = ir2[0];
              ir2[0] = (v474_data + (v470_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v480_data = ir2[1];
              ir2[1] = (v480_data + (v470_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v486_data = ir2[2];
              ir2[2] = (v486_data + (v470_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v492_data = ir2[3];
              ir2[3] = (v492_data + (v470_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v498_data = ir2[4];
              ir2[4] = (v498_data + (v470_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v504_data = ir2[5];
              ir2[5] = (v504_data + (v470_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v510_data = ir2[6];
              ir2[6] = (v510_data + (v470_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v516_data = ir2[7];
              ir2[7] = (v516_data + (v470_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v518_data = r0[10];
              float v522_data = ir2[0];
              ir2[0] = (v522_data + (v518_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v528_data = ir2[1];
              ir2[1] = (v528_data + (v518_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v534_data = ir2[2];
              ir2[2] = (v534_data + (v518_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v540_data = ir2[3];
              ir2[3] = (v540_data + (v518_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v546_data = ir2[4];
              ir2[4] = (v546_data + (v518_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v552_data = ir2[5];
              ir2[5] = (v552_data + (v518_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v558_data = ir2[6];
              ir2[6] = (v558_data + (v518_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v564_data = ir2[7];
              ir2[7] = (v564_data + (v518_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v566_data = r0[11];
              float v570_data = ir2[0];
              ir2[0] = (v570_data + (v566_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v576_data = ir2[1];
              ir2[1] = (v576_data + (v566_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v582_data = ir2[2];
              ir2[2] = (v582_data + (v566_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v588_data = ir2[3];
              ir2[3] = (v588_data + (v566_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v594_data = ir2[4];
              ir2[4] = (v594_data + (v566_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v600_data = ir2[5];
              ir2[5] = (v600_data + (v566_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v606_data = ir2[6];
              ir2[6] = (v606_data + (v566_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v612_data = ir2[7];
              ir2[7] = (v612_data + (v566_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v614_data = r0[12];
              float v618_data = ir2[0];
              ir2[0] = (v618_data + (v614_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v624_data = ir2[1];
              ir2[1] = (v624_data + (v614_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v630_data = ir2[2];
              ir2[2] = (v630_data + (v614_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v636_data = ir2[3];
              ir2[3] = (v636_data + (v614_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v642_data = ir2[4];
              ir2[4] = (v642_data + (v614_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v648_data = ir2[5];
              ir2[5] = (v648_data + (v614_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v654_data = ir2[6];
              ir2[6] = (v654_data + (v614_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v660_data = ir2[7];
              ir2[7] = (v660_data + (v614_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v662_data = r0[13];
              float v666_data = ir2[0];
              ir2[0] = (v666_data + (v662_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v672_data = ir2[1];
              ir2[1] = (v672_data + (v662_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v678_data = ir2[2];
              ir2[2] = (v678_data + (v662_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v684_data = ir2[3];
              ir2[3] = (v684_data + (v662_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v690_data = ir2[4];
              ir2[4] = (v690_data + (v662_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v696_data = ir2[5];
              ir2[5] = (v696_data + (v662_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v702_data = ir2[6];
              ir2[6] = (v702_data + (v662_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v708_data = ir2[7];
              ir2[7] = (v708_data + (v662_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v710_data = r0[14];
              float v714_data = ir2[0];
              ir2[0] = (v714_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v720_data = ir2[1];
              ir2[1] = (v720_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v726_data = ir2[2];
              ir2[2] = (v726_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v732_data = ir2[3];
              ir2[3] = (v732_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v738_data = ir2[4];
              ir2[4] = (v738_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v744_data = ir2[5];
              ir2[5] = (v744_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v750_data = ir2[6];
              ir2[6] = (v750_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v756_data = ir2[7];
              ir2[7] = (v756_data + (v710_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v758_data = r0[15];
              float v762_data = ir2[0];
              ir2[0] = (v762_data + (v758_data * (sycl::select_from_group(item.get_sub_group(), v39_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v768_data = ir2[1];
              ir2[1] = (v768_data + (v758_data * (sycl::select_from_group(item.get_sub_group(), v45_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v774_data = ir2[2];
              ir2[2] = (v774_data + (v758_data * (sycl::select_from_group(item.get_sub_group(), v51_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v780_data = ir2[3];
              ir2[3] = (v780_data + (v758_data * (sycl::select_from_group(item.get_sub_group(), v57_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v786_data = ir2[4];
              ir2[4] = (v786_data + (v758_data * (sycl::select_from_group(item.get_sub_group(), v63_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v792_data = ir2[5];
              ir2[5] = (v792_data + (v758_data * (sycl::select_from_group(item.get_sub_group(), v69_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v798_data = ir2[6];
              ir2[6] = (v798_data + (v758_data * (sycl::select_from_group(item.get_sub_group(), v75_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v804_data = ir2[7];
              ir2[7] = (v804_data + (v758_data * (sycl::select_from_group(item.get_sub_group(), v81_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              if (v18_g) {
                #pragma unroll
                for (int32_t v806_n1 = 0; v806_n1 < 8; ++v806_n1) {
                  float v808_data = ir2[v806_n1];
                  r2[v806_n1] = v808_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v18_g) {
                #pragma unroll
                for (int32_t v809_i1 = 0; v809_i1 < 8; ++v809_i1) {
                  float v811_data = r2[v809_i1];
                  glb_m0[(v17_lead + (v809_i1 * 12))] = v811_data;
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

