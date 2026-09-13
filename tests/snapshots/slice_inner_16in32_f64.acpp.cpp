// === base name ===
kernel_bd118e2822c4a268

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_bd118e2822c4a268 = {{16, 16, 1}, 16, 16, 1, 16, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_bd118e2822c4a268(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_bd118e2822c4a268(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_bd118e2822c4a268(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_bd118e2822c4a268(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_bd118e2822c4a268(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_bd118e2822c4a268(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_bd118e2822c4a268(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 2048 B shared, occupancy grid
        // operands:
        //   m0 16×8(16×8) {0..16}×{0..8} strided
        //   m1 32×32(32×32) {0..32}×{0..32} strided
        //   m2 16×8(16×8) {0..16}×{0..8} strided
        // operations:
        //   m0[i,j] = m1[i,k]@{8..24}×{8..24} × m2[k,j]
        // tensorforge-meta: {"fp":"double","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":2048,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,8]],"name":"m0","ordered":false,"parts":1,"shape":[16,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[32,32]],"name":"m1","ordered":false,"parts":1,"shape":[32,32],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[8,8],"shape":[32,32]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
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
              double *const __restrict__ glb_m0 = &m0[v3_batchId0 * 128 + 0 + m0_extraOffset];
              const double *const __restrict__ glb_m1 = &m1[v3_batchId0 * 1024 + 0 + m1_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[v3_batchId0 * 128 + 0 + m2_extraOffset];
              double r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v17_lead = item.get_local_id(2) % 16;
              #pragma unroll
              for (int32_t v18_i0 = 0; v18_i0 < 1; ++v18_i0) {
                int32_t v22_off = (v17_lead + (v18_i0 * 16)) + 8;
                #pragma unroll
                for (int32_t v19_i1 = 8; v19_i1 < 24; ++v19_i1) {
                  double v25_data = glb_m1[(v22_off + (v19_i1 * 32))];
                  r0[(v18_i0 + (v19_i1 - 8))] = v25_data;
                }
              }
              double r1[8]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v29_i0 = 0; v29_i0 < 1; ++v29_i0) {
                int32_t v32_lead = v17_lead + (v29_i0 * 16);
                #pragma unroll
                for (int32_t v30_i1 = 0; v30_i1 < 8; ++v30_i1) {
                  double v35_data = glb_m2[(v32_lead + (v30_i1 * 16))];
                  r1[(v29_i0 + v30_i1)] = v35_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              double r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 16), (0, 8)] [(0, 16)]
              double ir2[8]{};
              double v39_data = r0[0];
              double v40_data = r1[0];
              double v43_data = ir2[0];
              ir2[0] = (v43_data + (v39_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v46_data = r1[1];
              double v49_data = ir2[1];
              ir2[1] = (v49_data + (v39_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v52_data = r1[2];
              double v55_data = ir2[2];
              ir2[2] = (v55_data + (v39_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v58_data = r1[3];
              double v61_data = ir2[3];
              ir2[3] = (v61_data + (v39_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v64_data = r1[4];
              double v67_data = ir2[4];
              ir2[4] = (v67_data + (v39_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v70_data = r1[5];
              double v73_data = ir2[5];
              ir2[5] = (v73_data + (v39_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v76_data = r1[6];
              double v79_data = ir2[6];
              ir2[6] = (v79_data + (v39_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v82_data = r1[7];
              double v85_data = ir2[7];
              ir2[7] = (v85_data + (v39_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v87_data = r0[1];
              double v91_data = ir2[0];
              ir2[0] = (v91_data + (v87_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v97_data = ir2[1];
              ir2[1] = (v97_data + (v87_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v103_data = ir2[2];
              ir2[2] = (v103_data + (v87_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v109_data = ir2[3];
              ir2[3] = (v109_data + (v87_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v115_data = ir2[4];
              ir2[4] = (v115_data + (v87_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v121_data = ir2[5];
              ir2[5] = (v121_data + (v87_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v127_data = ir2[6];
              ir2[6] = (v127_data + (v87_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v133_data = ir2[7];
              ir2[7] = (v133_data + (v87_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v135_data = r0[2];
              double v139_data = ir2[0];
              ir2[0] = (v139_data + (v135_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v145_data = ir2[1];
              ir2[1] = (v145_data + (v135_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v151_data = ir2[2];
              ir2[2] = (v151_data + (v135_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v157_data = ir2[3];
              ir2[3] = (v157_data + (v135_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v163_data = ir2[4];
              ir2[4] = (v163_data + (v135_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v169_data = ir2[5];
              ir2[5] = (v169_data + (v135_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v175_data = ir2[6];
              ir2[6] = (v175_data + (v135_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v181_data = ir2[7];
              ir2[7] = (v181_data + (v135_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v183_data = r0[3];
              double v187_data = ir2[0];
              ir2[0] = (v187_data + (v183_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v193_data = ir2[1];
              ir2[1] = (v193_data + (v183_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v199_data = ir2[2];
              ir2[2] = (v199_data + (v183_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v205_data = ir2[3];
              ir2[3] = (v205_data + (v183_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v211_data = ir2[4];
              ir2[4] = (v211_data + (v183_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v217_data = ir2[5];
              ir2[5] = (v217_data + (v183_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v223_data = ir2[6];
              ir2[6] = (v223_data + (v183_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v229_data = ir2[7];
              ir2[7] = (v229_data + (v183_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v231_data = r0[4];
              double v235_data = ir2[0];
              ir2[0] = (v235_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v241_data = ir2[1];
              ir2[1] = (v241_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v247_data = ir2[2];
              ir2[2] = (v247_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v253_data = ir2[3];
              ir2[3] = (v253_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v259_data = ir2[4];
              ir2[4] = (v259_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v265_data = ir2[5];
              ir2[5] = (v265_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v271_data = ir2[6];
              ir2[6] = (v271_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v277_data = ir2[7];
              ir2[7] = (v277_data + (v231_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v279_data = r0[5];
              double v283_data = ir2[0];
              ir2[0] = (v283_data + (v279_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v289_data = ir2[1];
              ir2[1] = (v289_data + (v279_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v295_data = ir2[2];
              ir2[2] = (v295_data + (v279_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v301_data = ir2[3];
              ir2[3] = (v301_data + (v279_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v307_data = ir2[4];
              ir2[4] = (v307_data + (v279_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v313_data = ir2[5];
              ir2[5] = (v313_data + (v279_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v319_data = ir2[6];
              ir2[6] = (v319_data + (v279_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v325_data = ir2[7];
              ir2[7] = (v325_data + (v279_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v327_data = r0[6];
              double v331_data = ir2[0];
              ir2[0] = (v331_data + (v327_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v337_data = ir2[1];
              ir2[1] = (v337_data + (v327_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v343_data = ir2[2];
              ir2[2] = (v343_data + (v327_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v349_data = ir2[3];
              ir2[3] = (v349_data + (v327_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v355_data = ir2[4];
              ir2[4] = (v355_data + (v327_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v361_data = ir2[5];
              ir2[5] = (v361_data + (v327_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v367_data = ir2[6];
              ir2[6] = (v367_data + (v327_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v373_data = ir2[7];
              ir2[7] = (v373_data + (v327_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v375_data = r0[7];
              double v379_data = ir2[0];
              ir2[0] = (v379_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v385_data = ir2[1];
              ir2[1] = (v385_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v391_data = ir2[2];
              ir2[2] = (v391_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v397_data = ir2[3];
              ir2[3] = (v397_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v403_data = ir2[4];
              ir2[4] = (v403_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v409_data = ir2[5];
              ir2[5] = (v409_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v415_data = ir2[6];
              ir2[6] = (v415_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v421_data = ir2[7];
              ir2[7] = (v421_data + (v375_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v423_data = r0[8];
              double v427_data = ir2[0];
              ir2[0] = (v427_data + (v423_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v433_data = ir2[1];
              ir2[1] = (v433_data + (v423_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v439_data = ir2[2];
              ir2[2] = (v439_data + (v423_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v445_data = ir2[3];
              ir2[3] = (v445_data + (v423_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v451_data = ir2[4];
              ir2[4] = (v451_data + (v423_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v457_data = ir2[5];
              ir2[5] = (v457_data + (v423_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v463_data = ir2[6];
              ir2[6] = (v463_data + (v423_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v469_data = ir2[7];
              ir2[7] = (v469_data + (v423_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v471_data = r0[9];
              double v475_data = ir2[0];
              ir2[0] = (v475_data + (v471_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v481_data = ir2[1];
              ir2[1] = (v481_data + (v471_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v487_data = ir2[2];
              ir2[2] = (v487_data + (v471_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v493_data = ir2[3];
              ir2[3] = (v493_data + (v471_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v499_data = ir2[4];
              ir2[4] = (v499_data + (v471_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v505_data = ir2[5];
              ir2[5] = (v505_data + (v471_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v511_data = ir2[6];
              ir2[6] = (v511_data + (v471_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v517_data = ir2[7];
              ir2[7] = (v517_data + (v471_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v519_data = r0[10];
              double v523_data = ir2[0];
              ir2[0] = (v523_data + (v519_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v529_data = ir2[1];
              ir2[1] = (v529_data + (v519_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v535_data = ir2[2];
              ir2[2] = (v535_data + (v519_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v541_data = ir2[3];
              ir2[3] = (v541_data + (v519_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v547_data = ir2[4];
              ir2[4] = (v547_data + (v519_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v553_data = ir2[5];
              ir2[5] = (v553_data + (v519_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v559_data = ir2[6];
              ir2[6] = (v559_data + (v519_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v565_data = ir2[7];
              ir2[7] = (v565_data + (v519_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v567_data = r0[11];
              double v571_data = ir2[0];
              ir2[0] = (v571_data + (v567_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v577_data = ir2[1];
              ir2[1] = (v577_data + (v567_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v583_data = ir2[2];
              ir2[2] = (v583_data + (v567_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v589_data = ir2[3];
              ir2[3] = (v589_data + (v567_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v595_data = ir2[4];
              ir2[4] = (v595_data + (v567_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v601_data = ir2[5];
              ir2[5] = (v601_data + (v567_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v607_data = ir2[6];
              ir2[6] = (v607_data + (v567_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v613_data = ir2[7];
              ir2[7] = (v613_data + (v567_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v615_data = r0[12];
              double v619_data = ir2[0];
              ir2[0] = (v619_data + (v615_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v625_data = ir2[1];
              ir2[1] = (v625_data + (v615_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v631_data = ir2[2];
              ir2[2] = (v631_data + (v615_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v637_data = ir2[3];
              ir2[3] = (v637_data + (v615_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v643_data = ir2[4];
              ir2[4] = (v643_data + (v615_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v649_data = ir2[5];
              ir2[5] = (v649_data + (v615_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v655_data = ir2[6];
              ir2[6] = (v655_data + (v615_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v661_data = ir2[7];
              ir2[7] = (v661_data + (v615_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v663_data = r0[13];
              double v667_data = ir2[0];
              ir2[0] = (v667_data + (v663_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v673_data = ir2[1];
              ir2[1] = (v673_data + (v663_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v679_data = ir2[2];
              ir2[2] = (v679_data + (v663_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v685_data = ir2[3];
              ir2[3] = (v685_data + (v663_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v691_data = ir2[4];
              ir2[4] = (v691_data + (v663_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v697_data = ir2[5];
              ir2[5] = (v697_data + (v663_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v703_data = ir2[6];
              ir2[6] = (v703_data + (v663_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v709_data = ir2[7];
              ir2[7] = (v709_data + (v663_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v711_data = r0[14];
              double v715_data = ir2[0];
              ir2[0] = (v715_data + (v711_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v721_data = ir2[1];
              ir2[1] = (v721_data + (v711_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v727_data = ir2[2];
              ir2[2] = (v727_data + (v711_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v733_data = ir2[3];
              ir2[3] = (v733_data + (v711_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v739_data = ir2[4];
              ir2[4] = (v739_data + (v711_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v745_data = ir2[5];
              ir2[5] = (v745_data + (v711_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v751_data = ir2[6];
              ir2[6] = (v751_data + (v711_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v757_data = ir2[7];
              ir2[7] = (v757_data + (v711_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v759_data = r0[15];
              double v763_data = ir2[0];
              ir2[0] = (v763_data + (v759_data * (sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v769_data = ir2[1];
              ir2[1] = (v769_data + (v759_data * (sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v775_data = ir2[2];
              ir2[2] = (v775_data + (v759_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v781_data = ir2[3];
              ir2[3] = (v781_data + (v759_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v787_data = ir2[4];
              ir2[4] = (v787_data + (v759_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v793_data = ir2[5];
              ir2[5] = (v793_data + (v759_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v799_data = ir2[6];
              ir2[6] = (v799_data + (v759_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v805_data = ir2[7];
              ir2[7] = (v805_data + (v759_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              #pragma unroll
              for (int32_t v807_n0 = 0; v807_n0 < 1; ++v807_n0) {
                #pragma unroll
                for (int32_t v808_n1 = 0; v808_n1 < 8; ++v808_n1) {
                  int32_t v809_a = v807_n0 + v808_n1;
                  double v810_data = ir2[v809_a];
                  r2[v809_a] = v810_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v811_i0 = 0; v811_i0 < 1; ++v811_i0) {
                int32_t v816_lead = v17_lead + (v811_i0 * 16);
                #pragma unroll
                for (int32_t v812_i1 = 0; v812_i1 < 8; ++v812_i1) {
                  double v814_data = r2[(v811_i0 + v812_i1)];
                  glb_m0[(v816_lead + (v812_i1 * 16))] = v814_data;
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

