// === base name ===
kernel_914cb21f8f0bc3ba

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_914cb21f8f0bc3ba = {{16, 16, 1}, 16, 12, 1, 16, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_914cb21f8f0bc3ba(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_914cb21f8f0bc3ba(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_914cb21f8f0bc3ba(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 256 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_914cb21f8f0bc3ba(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_914cb21f8f0bc3ba(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_914cb21f8f0bc3ba(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_914cb21f8f0bc3ba(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 2048 B shared, occupancy grid
        // operands:
        //   m0 12×8(12×8) {0..12}×{0..8} strided
        //   m1 12×16(12×16) {0..12}×{0..16} strided
        //   m2 16×8(16×8) {0..16}×{0..8} strided
        // operations:
        //   m0[i,j] += m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"double","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":2048,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,8]],"name":"m0","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,16]],"name":"m1","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,16]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          double* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          double* tempShrMem = &localShrMem0[0];
          for (size_t v3_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v3_batchId0 < numElements0; v3_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v4_ahead1 = v3_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v3_batchId0]);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[v3_batchId0 * 96 + 0 + m0_extraOffset];
              const double *const __restrict__ glb_m1 = &m1[v3_batchId0 * 192 + 0 + m1_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[v3_batchId0 * 128 + 0 + m2_extraOffset];
              double r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v17_lead = item.get_local_id(2) % 16;
              bool v18_g = v17_lead < 12;
              if (v18_g) {
                #pragma unroll
                for (int32_t v19_i1 = 0; v19_i1 < 16; ++v19_i1) {
                  double v24_data = glb_m1[(v17_lead + (v19_i1 * 12))];
                  r0[v19_i1] = v24_data;
                }
              }
              double r1[8]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v27_i0 = 0; v27_i0 < 1; ++v27_i0) {
                int32_t v30_lead = v17_lead + (v27_i0 * 16);
                #pragma unroll
                for (int32_t v28_i1 = 0; v28_i1 < 8; ++v28_i1) {
                  double v33_data = glb_m2[(v30_lead + (v28_i1 * 16))];
                  r1[(v27_i0 + v28_i1)] = v33_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              double r2[8]{};
              // r2 = load{g>r}(glb_m0);
              if (v18_g) {
                #pragma unroll
                for (int32_t v36_i1 = 0; v36_i1 < 8; ++v36_i1) {
                  double v41_data = glb_m0[(v17_lead + (v36_i1 * 12))];
                  r2[v36_i1] = v41_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m2););
              // wait(r2 = load{g>r}(glb_m0););
              double r3[8]{};
              // r3 = +(r0 * r1) + name: r2, type: SymbolType.Register, lead: [0]
              // [(0, 12), (0, 8)] [(0, 16)]
              double ir3[8]{};
              double v45_data = r0[0];
              double v46_data = r1[0];
              double v49_data = ir3[0];
              ir3[0] = (v49_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v52_data = r1[1];
              double v55_data = ir3[1];
              ir3[1] = (v55_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v58_data = r1[2];
              double v61_data = ir3[2];
              ir3[2] = (v61_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v64_data = r1[3];
              double v67_data = ir3[3];
              ir3[3] = (v67_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v70_data = r1[4];
              double v73_data = ir3[4];
              ir3[4] = (v73_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v76_data = r1[5];
              double v79_data = ir3[5];
              ir3[5] = (v79_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v82_data = r1[6];
              double v85_data = ir3[6];
              ir3[6] = (v85_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v88_data = r1[7];
              double v91_data = ir3[7];
              ir3[7] = (v91_data + (v45_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v93_data = r0[1];
              double v97_data = ir3[0];
              ir3[0] = (v97_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v103_data = ir3[1];
              ir3[1] = (v103_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v109_data = ir3[2];
              ir3[2] = (v109_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v115_data = ir3[3];
              ir3[3] = (v115_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v121_data = ir3[4];
              ir3[4] = (v121_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v127_data = ir3[5];
              ir3[5] = (v127_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v133_data = ir3[6];
              ir3[6] = (v133_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v139_data = ir3[7];
              ir3[7] = (v139_data + (v93_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v141_data = r0[2];
              double v145_data = ir3[0];
              ir3[0] = (v145_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v151_data = ir3[1];
              ir3[1] = (v151_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v157_data = ir3[2];
              ir3[2] = (v157_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v163_data = ir3[3];
              ir3[3] = (v163_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v169_data = ir3[4];
              ir3[4] = (v169_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v175_data = ir3[5];
              ir3[5] = (v175_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v181_data = ir3[6];
              ir3[6] = (v181_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v187_data = ir3[7];
              ir3[7] = (v187_data + (v141_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v189_data = r0[3];
              double v193_data = ir3[0];
              ir3[0] = (v193_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v199_data = ir3[1];
              ir3[1] = (v199_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v205_data = ir3[2];
              ir3[2] = (v205_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v211_data = ir3[3];
              ir3[3] = (v211_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v217_data = ir3[4];
              ir3[4] = (v217_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v223_data = ir3[5];
              ir3[5] = (v223_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v229_data = ir3[6];
              ir3[6] = (v229_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v235_data = ir3[7];
              ir3[7] = (v235_data + (v189_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v237_data = r0[4];
              double v241_data = ir3[0];
              ir3[0] = (v241_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v247_data = ir3[1];
              ir3[1] = (v247_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v253_data = ir3[2];
              ir3[2] = (v253_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v259_data = ir3[3];
              ir3[3] = (v259_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v265_data = ir3[4];
              ir3[4] = (v265_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v271_data = ir3[5];
              ir3[5] = (v271_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v277_data = ir3[6];
              ir3[6] = (v277_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v283_data = ir3[7];
              ir3[7] = (v283_data + (v237_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v285_data = r0[5];
              double v289_data = ir3[0];
              ir3[0] = (v289_data + (v285_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v295_data = ir3[1];
              ir3[1] = (v295_data + (v285_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v301_data = ir3[2];
              ir3[2] = (v301_data + (v285_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v307_data = ir3[3];
              ir3[3] = (v307_data + (v285_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v313_data = ir3[4];
              ir3[4] = (v313_data + (v285_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v319_data = ir3[5];
              ir3[5] = (v319_data + (v285_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v325_data = ir3[6];
              ir3[6] = (v325_data + (v285_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v331_data = ir3[7];
              ir3[7] = (v331_data + (v285_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v333_data = r0[6];
              double v337_data = ir3[0];
              ir3[0] = (v337_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v343_data = ir3[1];
              ir3[1] = (v343_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v349_data = ir3[2];
              ir3[2] = (v349_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v355_data = ir3[3];
              ir3[3] = (v355_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v361_data = ir3[4];
              ir3[4] = (v361_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v367_data = ir3[5];
              ir3[5] = (v367_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v373_data = ir3[6];
              ir3[6] = (v373_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v379_data = ir3[7];
              ir3[7] = (v379_data + (v333_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v381_data = r0[7];
              double v385_data = ir3[0];
              ir3[0] = (v385_data + (v381_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v391_data = ir3[1];
              ir3[1] = (v391_data + (v381_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v397_data = ir3[2];
              ir3[2] = (v397_data + (v381_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v403_data = ir3[3];
              ir3[3] = (v403_data + (v381_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v409_data = ir3[4];
              ir3[4] = (v409_data + (v381_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v415_data = ir3[5];
              ir3[5] = (v415_data + (v381_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v421_data = ir3[6];
              ir3[6] = (v421_data + (v381_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v427_data = ir3[7];
              ir3[7] = (v427_data + (v381_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v429_data = r0[8];
              double v433_data = ir3[0];
              ir3[0] = (v433_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v439_data = ir3[1];
              ir3[1] = (v439_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v445_data = ir3[2];
              ir3[2] = (v445_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v451_data = ir3[3];
              ir3[3] = (v451_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v457_data = ir3[4];
              ir3[4] = (v457_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v463_data = ir3[5];
              ir3[5] = (v463_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v469_data = ir3[6];
              ir3[6] = (v469_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v475_data = ir3[7];
              ir3[7] = (v475_data + (v429_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v477_data = r0[9];
              double v481_data = ir3[0];
              ir3[0] = (v481_data + (v477_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v487_data = ir3[1];
              ir3[1] = (v487_data + (v477_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v493_data = ir3[2];
              ir3[2] = (v493_data + (v477_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v499_data = ir3[3];
              ir3[3] = (v499_data + (v477_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v505_data = ir3[4];
              ir3[4] = (v505_data + (v477_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v511_data = ir3[5];
              ir3[5] = (v511_data + (v477_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v517_data = ir3[6];
              ir3[6] = (v517_data + (v477_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v523_data = ir3[7];
              ir3[7] = (v523_data + (v477_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v525_data = r0[10];
              double v529_data = ir3[0];
              ir3[0] = (v529_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v535_data = ir3[1];
              ir3[1] = (v535_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v541_data = ir3[2];
              ir3[2] = (v541_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v547_data = ir3[3];
              ir3[3] = (v547_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v553_data = ir3[4];
              ir3[4] = (v553_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v559_data = ir3[5];
              ir3[5] = (v559_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v565_data = ir3[6];
              ir3[6] = (v565_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v571_data = ir3[7];
              ir3[7] = (v571_data + (v525_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v573_data = r0[11];
              double v577_data = ir3[0];
              ir3[0] = (v577_data + (v573_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v583_data = ir3[1];
              ir3[1] = (v583_data + (v573_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v589_data = ir3[2];
              ir3[2] = (v589_data + (v573_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v595_data = ir3[3];
              ir3[3] = (v595_data + (v573_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v601_data = ir3[4];
              ir3[4] = (v601_data + (v573_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v607_data = ir3[5];
              ir3[5] = (v607_data + (v573_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v613_data = ir3[6];
              ir3[6] = (v613_data + (v573_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v619_data = ir3[7];
              ir3[7] = (v619_data + (v573_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v621_data = r0[12];
              double v625_data = ir3[0];
              ir3[0] = (v625_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v631_data = ir3[1];
              ir3[1] = (v631_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v637_data = ir3[2];
              ir3[2] = (v637_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v643_data = ir3[3];
              ir3[3] = (v643_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v649_data = ir3[4];
              ir3[4] = (v649_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v655_data = ir3[5];
              ir3[5] = (v655_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v661_data = ir3[6];
              ir3[6] = (v661_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v667_data = ir3[7];
              ir3[7] = (v667_data + (v621_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v669_data = r0[13];
              double v673_data = ir3[0];
              ir3[0] = (v673_data + (v669_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v679_data = ir3[1];
              ir3[1] = (v679_data + (v669_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v685_data = ir3[2];
              ir3[2] = (v685_data + (v669_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v691_data = ir3[3];
              ir3[3] = (v691_data + (v669_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v697_data = ir3[4];
              ir3[4] = (v697_data + (v669_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v703_data = ir3[5];
              ir3[5] = (v703_data + (v669_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v709_data = ir3[6];
              ir3[6] = (v709_data + (v669_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v715_data = ir3[7];
              ir3[7] = (v715_data + (v669_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v717_data = r0[14];
              double v721_data = ir3[0];
              ir3[0] = (v721_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v727_data = ir3[1];
              ir3[1] = (v727_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v733_data = ir3[2];
              ir3[2] = (v733_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v739_data = ir3[3];
              ir3[3] = (v739_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v745_data = ir3[4];
              ir3[4] = (v745_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v751_data = ir3[5];
              ir3[5] = (v751_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v757_data = ir3[6];
              ir3[6] = (v757_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v763_data = ir3[7];
              ir3[7] = (v763_data + (v717_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v765_data = r0[15];
              double v769_data = ir3[0];
              ir3[0] = (v769_data + (v765_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v775_data = ir3[1];
              ir3[1] = (v775_data + (v765_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v781_data = ir3[2];
              ir3[2] = (v781_data + (v765_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v787_data = ir3[3];
              ir3[3] = (v787_data + (v765_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v793_data = ir3[4];
              ir3[4] = (v793_data + (v765_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v799_data = ir3[5];
              ir3[5] = (v799_data + (v765_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v805_data = ir3[6];
              ir3[6] = (v805_data + (v765_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v811_data = ir3[7];
              ir3[7] = (v811_data + (v765_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              if (v18_g) {
                #pragma unroll
                for (int32_t v813_n1 = 0; v813_n1 < 8; ++v813_n1) {
                  double v815_data = ir3[v813_n1];
                  double v816_data = r2[v813_n1];
                  r3[v813_n1] = (v816_data + v815_data);
                }
              }
              // glb_m0 = store{r>g}(r3);
              if (v18_g) {
                #pragma unroll
                for (int32_t v818_i1 = 0; v818_i1 < 8; ++v818_i1) {
                  double v820_data = r3[v818_i1];
                  glb_m0[(v17_lead + (v818_i1 * 12))] = v820_data;
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

