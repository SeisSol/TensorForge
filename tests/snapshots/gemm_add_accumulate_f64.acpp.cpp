// === base name ===
kernel_cc63e2acd6225497

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_cc63e2acd6225497 = {{16, 16, 1}, 16, 12, 1, 16, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_cc63e2acd6225497(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_cc63e2acd6225497(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_cc63e2acd6225497(size_t numElements0, void* streamPtr) {
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
  config.sharedMemBytes = 256 * sizeof(double);
  config.cooperative = false;
  return config;
}
void launcher_kernel_cc63e2acd6225497(double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_cc63e2acd6225497(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_cc63e2acd6225497(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_cc63e2acd6225497(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double * m0, size_t m0_extraOffset, const double * m1, size_t m1_extraOffset, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 2048 B shared, occupancy grid
        // operands:
        //   m0 12×8(12×8) {0..12}×{0..8} strided
        //   m1 12×16(12×16) {0..12}×{0..16} strided
        //   m2 16×8(16×8) {0..16}×{0..8} strided
        // operations:
        //   m0[i,j] += m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"double","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":2048,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[12,8]],"name":"m0","ordered":false,"parts":1,"shape":[12,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[12,16]],"name":"m1","ordered":false,"parts":1,"shape":[12,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":true,"dest":{"addressing":"strided","bbox":[[0,0],[12,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[12,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[12,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[12,16]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          double* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          double* tempShrMem = &localShrMem0[0];
          for (size_t v9_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v9_batchId0 < numElements0; v9_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v10_ahead1 = v9_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v12_batchId1 = (v10_ahead1 < numElements0) ? v10_ahead1 : v9_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v9_batchId0]);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[v9_batchId0 * 96 + 0 + m0_extraOffset];
              const double *const __restrict__ glb_m1 = &m1[v9_batchId0 * 192 + 0 + m1_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[v9_batchId0 * 128 + 0 + m2_extraOffset];
              double r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v23_lead = item.get_local_id(2) % 16;
              bool v24_g = v23_lead < 12;
              if (v24_g) {
                #pragma unroll
                for (int32_t v25_i1 = 0; v25_i1 < 16; ++v25_i1) {
                  double v30_data = glb_m1[(v23_lead + (v25_i1 * 12))];
                  r0[v25_i1] = v30_data;
                }
              }
              double r1[8]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v33_i0 = 0; v33_i0 < 1; ++v33_i0) {
                int32_t v36_lead = v23_lead + (v33_i0 * 16);
                #pragma unroll
                for (int32_t v34_i1 = 0; v34_i1 < 8; ++v34_i1) {
                  double v39_data = glb_m2[(v36_lead + (v34_i1 * 16))];
                  r1[(v33_i0 + v34_i1)] = v39_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              double r2[8]{};
              // r2 = load{g>r}(glb_m0);
              if (v24_g) {
                #pragma unroll
                for (int32_t v42_i1 = 0; v42_i1 < 8; ++v42_i1) {
                  double v47_data = glb_m0[(v23_lead + (v42_i1 * 12))];
                  r2[v42_i1] = v47_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m2););
              // wait(r2 = load{g>r}(glb_m0););
              double r3[8]{};
              // ir3 = +(r0 * r1)
              // [(0, 12), (0, 8)] [(0, 16)]
              double ir3[8]{};
              double v51_data = r0[0];
              double v52_data = r1[0];
              double v55_data = ir3[0];
              ir3[0] = (v55_data + (v51_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v58_data = r1[1];
              double v61_data = ir3[1];
              ir3[1] = (v61_data + (v51_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v64_data = r1[2];
              double v67_data = ir3[2];
              ir3[2] = (v67_data + (v51_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v70_data = r1[3];
              double v73_data = ir3[3];
              ir3[3] = (v73_data + (v51_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v76_data = r1[4];
              double v79_data = ir3[4];
              ir3[4] = (v79_data + (v51_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v82_data = r1[5];
              double v85_data = ir3[5];
              ir3[5] = (v85_data + (v51_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v88_data = r1[6];
              double v91_data = ir3[6];
              ir3[6] = (v91_data + (v51_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v94_data = r1[7];
              double v97_data = ir3[7];
              ir3[7] = (v97_data + (v51_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v99_data = r0[1];
              double v103_data = ir3[0];
              ir3[0] = (v103_data + (v99_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v109_data = ir3[1];
              ir3[1] = (v109_data + (v99_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v115_data = ir3[2];
              ir3[2] = (v115_data + (v99_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v121_data = ir3[3];
              ir3[3] = (v121_data + (v99_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v127_data = ir3[4];
              ir3[4] = (v127_data + (v99_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v133_data = ir3[5];
              ir3[5] = (v133_data + (v99_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v139_data = ir3[6];
              ir3[6] = (v139_data + (v99_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v145_data = ir3[7];
              ir3[7] = (v145_data + (v99_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v147_data = r0[2];
              double v151_data = ir3[0];
              ir3[0] = (v151_data + (v147_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v157_data = ir3[1];
              ir3[1] = (v157_data + (v147_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v163_data = ir3[2];
              ir3[2] = (v163_data + (v147_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v169_data = ir3[3];
              ir3[3] = (v169_data + (v147_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v175_data = ir3[4];
              ir3[4] = (v175_data + (v147_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v181_data = ir3[5];
              ir3[5] = (v181_data + (v147_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v187_data = ir3[6];
              ir3[6] = (v187_data + (v147_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v193_data = ir3[7];
              ir3[7] = (v193_data + (v147_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v195_data = r0[3];
              double v199_data = ir3[0];
              ir3[0] = (v199_data + (v195_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v205_data = ir3[1];
              ir3[1] = (v205_data + (v195_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v211_data = ir3[2];
              ir3[2] = (v211_data + (v195_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v217_data = ir3[3];
              ir3[3] = (v217_data + (v195_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v223_data = ir3[4];
              ir3[4] = (v223_data + (v195_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v229_data = ir3[5];
              ir3[5] = (v229_data + (v195_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v235_data = ir3[6];
              ir3[6] = (v235_data + (v195_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v241_data = ir3[7];
              ir3[7] = (v241_data + (v195_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v243_data = r0[4];
              double v247_data = ir3[0];
              ir3[0] = (v247_data + (v243_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v253_data = ir3[1];
              ir3[1] = (v253_data + (v243_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v259_data = ir3[2];
              ir3[2] = (v259_data + (v243_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v265_data = ir3[3];
              ir3[3] = (v265_data + (v243_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v271_data = ir3[4];
              ir3[4] = (v271_data + (v243_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v277_data = ir3[5];
              ir3[5] = (v277_data + (v243_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v283_data = ir3[6];
              ir3[6] = (v283_data + (v243_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v289_data = ir3[7];
              ir3[7] = (v289_data + (v243_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v291_data = r0[5];
              double v295_data = ir3[0];
              ir3[0] = (v295_data + (v291_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v301_data = ir3[1];
              ir3[1] = (v301_data + (v291_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v307_data = ir3[2];
              ir3[2] = (v307_data + (v291_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v313_data = ir3[3];
              ir3[3] = (v313_data + (v291_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v319_data = ir3[4];
              ir3[4] = (v319_data + (v291_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v325_data = ir3[5];
              ir3[5] = (v325_data + (v291_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v331_data = ir3[6];
              ir3[6] = (v331_data + (v291_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v337_data = ir3[7];
              ir3[7] = (v337_data + (v291_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v339_data = r0[6];
              double v343_data = ir3[0];
              ir3[0] = (v343_data + (v339_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v349_data = ir3[1];
              ir3[1] = (v349_data + (v339_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v355_data = ir3[2];
              ir3[2] = (v355_data + (v339_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v361_data = ir3[3];
              ir3[3] = (v361_data + (v339_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v367_data = ir3[4];
              ir3[4] = (v367_data + (v339_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v373_data = ir3[5];
              ir3[5] = (v373_data + (v339_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v379_data = ir3[6];
              ir3[6] = (v379_data + (v339_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v385_data = ir3[7];
              ir3[7] = (v385_data + (v339_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v387_data = r0[7];
              double v391_data = ir3[0];
              ir3[0] = (v391_data + (v387_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v397_data = ir3[1];
              ir3[1] = (v397_data + (v387_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v403_data = ir3[2];
              ir3[2] = (v403_data + (v387_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v409_data = ir3[3];
              ir3[3] = (v409_data + (v387_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v415_data = ir3[4];
              ir3[4] = (v415_data + (v387_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v421_data = ir3[5];
              ir3[5] = (v421_data + (v387_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v427_data = ir3[6];
              ir3[6] = (v427_data + (v387_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v433_data = ir3[7];
              ir3[7] = (v433_data + (v387_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v435_data = r0[8];
              double v439_data = ir3[0];
              ir3[0] = (v439_data + (v435_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v445_data = ir3[1];
              ir3[1] = (v445_data + (v435_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v451_data = ir3[2];
              ir3[2] = (v451_data + (v435_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v457_data = ir3[3];
              ir3[3] = (v457_data + (v435_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v463_data = ir3[4];
              ir3[4] = (v463_data + (v435_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v469_data = ir3[5];
              ir3[5] = (v469_data + (v435_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v475_data = ir3[6];
              ir3[6] = (v475_data + (v435_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v481_data = ir3[7];
              ir3[7] = (v481_data + (v435_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v483_data = r0[9];
              double v487_data = ir3[0];
              ir3[0] = (v487_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v493_data = ir3[1];
              ir3[1] = (v493_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v499_data = ir3[2];
              ir3[2] = (v499_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v505_data = ir3[3];
              ir3[3] = (v505_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v511_data = ir3[4];
              ir3[4] = (v511_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v517_data = ir3[5];
              ir3[5] = (v517_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v523_data = ir3[6];
              ir3[6] = (v523_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v529_data = ir3[7];
              ir3[7] = (v529_data + (v483_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v531_data = r0[10];
              double v535_data = ir3[0];
              ir3[0] = (v535_data + (v531_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v541_data = ir3[1];
              ir3[1] = (v541_data + (v531_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v547_data = ir3[2];
              ir3[2] = (v547_data + (v531_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v553_data = ir3[3];
              ir3[3] = (v553_data + (v531_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v559_data = ir3[4];
              ir3[4] = (v559_data + (v531_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v565_data = ir3[5];
              ir3[5] = (v565_data + (v531_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v571_data = ir3[6];
              ir3[6] = (v571_data + (v531_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v577_data = ir3[7];
              ir3[7] = (v577_data + (v531_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v579_data = r0[11];
              double v583_data = ir3[0];
              ir3[0] = (v583_data + (v579_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v589_data = ir3[1];
              ir3[1] = (v589_data + (v579_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v595_data = ir3[2];
              ir3[2] = (v595_data + (v579_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v601_data = ir3[3];
              ir3[3] = (v601_data + (v579_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v607_data = ir3[4];
              ir3[4] = (v607_data + (v579_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v613_data = ir3[5];
              ir3[5] = (v613_data + (v579_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v619_data = ir3[6];
              ir3[6] = (v619_data + (v579_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v625_data = ir3[7];
              ir3[7] = (v625_data + (v579_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v627_data = r0[12];
              double v631_data = ir3[0];
              ir3[0] = (v631_data + (v627_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v637_data = ir3[1];
              ir3[1] = (v637_data + (v627_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v643_data = ir3[2];
              ir3[2] = (v643_data + (v627_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v649_data = ir3[3];
              ir3[3] = (v649_data + (v627_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v655_data = ir3[4];
              ir3[4] = (v655_data + (v627_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v661_data = ir3[5];
              ir3[5] = (v661_data + (v627_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v667_data = ir3[6];
              ir3[6] = (v667_data + (v627_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v673_data = ir3[7];
              ir3[7] = (v673_data + (v627_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v675_data = r0[13];
              double v679_data = ir3[0];
              ir3[0] = (v679_data + (v675_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v685_data = ir3[1];
              ir3[1] = (v685_data + (v675_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v691_data = ir3[2];
              ir3[2] = (v691_data + (v675_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v697_data = ir3[3];
              ir3[3] = (v697_data + (v675_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v703_data = ir3[4];
              ir3[4] = (v703_data + (v675_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v709_data = ir3[5];
              ir3[5] = (v709_data + (v675_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v715_data = ir3[6];
              ir3[6] = (v715_data + (v675_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v721_data = ir3[7];
              ir3[7] = (v721_data + (v675_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v723_data = r0[14];
              double v727_data = ir3[0];
              ir3[0] = (v727_data + (v723_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v733_data = ir3[1];
              ir3[1] = (v733_data + (v723_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v739_data = ir3[2];
              ir3[2] = (v739_data + (v723_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v745_data = ir3[3];
              ir3[3] = (v745_data + (v723_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v751_data = ir3[4];
              ir3[4] = (v751_data + (v723_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v757_data = ir3[5];
              ir3[5] = (v757_data + (v723_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v763_data = ir3[6];
              ir3[6] = (v763_data + (v723_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v769_data = ir3[7];
              ir3[7] = (v769_data + (v723_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v771_data = r0[15];
              double v775_data = ir3[0];
              ir3[0] = (v775_data + (v771_data * (sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v781_data = ir3[1];
              ir3[1] = (v781_data + (v771_data * (sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v787_data = ir3[2];
              ir3[2] = (v787_data + (v771_data * (sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v793_data = ir3[3];
              ir3[3] = (v793_data + (v771_data * (sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v799_data = ir3[4];
              ir3[4] = (v799_data + (v771_data * (sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v805_data = ir3[5];
              ir3[5] = (v805_data + (v771_data * (sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v811_data = ir3[6];
              ir3[6] = (v811_data + (v771_data * (sycl::select_from_group(item.get_sub_group(), v88_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v817_data = ir3[7];
              ir3[7] = (v817_data + (v771_data * (sycl::select_from_group(item.get_sub_group(), v94_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              // r3 = ir3 + r2
              if (v24_g) {
                #pragma unroll
                for (int32_t v819_n1 = 0; v819_n1 < 8; ++v819_n1) {
                  double v821_data = ir3[v819_n1];
                  double v822_data = r2[v819_n1];
                  r3[v819_n1] = (v822_data + v821_data);
                }
              }
              // glb_m0 = store{r>g}(r3);
              if (v24_g) {
                #pragma unroll
                for (int32_t v824_i1 = 0; v824_i1 < 8; ++v824_i1) {
                  double v826_data = r3[v824_i1];
                  glb_m0[(v23_lead + (v824_i1 * 12))] = v826_data;
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

