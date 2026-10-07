// === base name ===
kernel_5be3b2ec7020eb23

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_5be3b2ec7020eb23 = {{16, 16, 1}, 16, 16, 1, 16, 2048, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_5be3b2ec7020eb23(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_5be3b2ec7020eb23(double * m0, size_t m0_extraOffset, const double * m1, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_5be3b2ec7020eb23(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_5be3b2ec7020eb23(double * m0, size_t m0_extraOffset, const double * m1, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_5be3b2ec7020eb23(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_5be3b2ec7020eb23(stream, grid, block, m0, m0_extraOffset, m1, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_5be3b2ec7020eb23(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double * m0, size_t m0_extraOffset, const double * m1, const double * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 2048 B shared, occupancy grid
        // operands:
        //   m0 16×16(16×16) {0..16}×{0..16} strided
        //   m1 16×16(16×16) {0..16}×{0..16} none
        //   m2 16×16(16×16) {0..16}×{0..16} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"double","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":2048,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"none","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"none","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          double* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          const double *const __restrict__ glb_m1 = &m1[0];
          for (size_t v8_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v8_batchId0 < numElements0; v8_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v9_ahead1 = v8_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v11_batchId1 = (v9_ahead1 < numElements0) ? v9_ahead1 : v8_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v8_batchId0]);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[v8_batchId0 * 256 + 0 + m0_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[v8_batchId0 * 256 + 0 + m2_extraOffset];
              double r0[16]{};
              // r0 = load{g>r}(glb_m2);
              int32_t v21_lead = item.get_local_id(2) % 16;
              #pragma unroll
              for (int32_t v22_i0 = 0; v22_i0 < 1; ++v22_i0) {
                int32_t v25_lead = v21_lead + (v22_i0 * 16);
                #pragma unroll
                for (int32_t v23_i1 = 0; v23_i1 < 16; ++v23_i1) {
                  double v28_data = glb_m2[(v25_lead + (v23_i1 * 16))];
                  r0[(v22_i0 + v23_i1)] = v28_data;
                }
              }
              double r1[16]{};
              // ir1 = +(glb_m1 * r0)
              // [(0, 16), (0, 16)] [(0, 16)]
              double ir1[16]{};
              double v35_data = glb_m1[v21_lead];
              double v36_data = r0[0];
              double v39_data = ir1[0];
              ir1[0] = (v39_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v42_data = r0[1];
              double v45_data = ir1[1];
              ir1[1] = (v45_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v48_data = r0[2];
              double v51_data = ir1[2];
              ir1[2] = (v51_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v54_data = r0[3];
              double v57_data = ir1[3];
              ir1[3] = (v57_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v60_data = r0[4];
              double v63_data = ir1[4];
              ir1[4] = (v63_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v66_data = r0[5];
              double v69_data = ir1[5];
              ir1[5] = (v69_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v72_data = r0[6];
              double v75_data = ir1[6];
              ir1[6] = (v75_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v78_data = r0[7];
              double v81_data = ir1[7];
              ir1[7] = (v81_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v84_data = r0[8];
              double v87_data = ir1[8];
              ir1[8] = (v87_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v90_data = r0[9];
              double v93_data = ir1[9];
              ir1[9] = (v93_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v96_data = r0[10];
              double v99_data = ir1[10];
              ir1[10] = (v99_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v102_data = r0[11];
              double v105_data = ir1[11];
              ir1[11] = (v105_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v108_data = r0[12];
              double v111_data = ir1[12];
              ir1[12] = (v111_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v114_data = r0[13];
              double v117_data = ir1[13];
              ir1[13] = (v117_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v120_data = r0[14];
              double v123_data = ir1[14];
              ir1[14] = (v123_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v126_data = r0[15];
              double v129_data = ir1[15];
              ir1[15] = (v129_data + (v35_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              double v132_data = glb_m1[(v21_lead + 16)];
              double v136_data = ir1[0];
              ir1[0] = (v136_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v142_data = ir1[1];
              ir1[1] = (v142_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v148_data = ir1[2];
              ir1[2] = (v148_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v154_data = ir1[3];
              ir1[3] = (v154_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v160_data = ir1[4];
              ir1[4] = (v160_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v166_data = ir1[5];
              ir1[5] = (v166_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v172_data = ir1[6];
              ir1[6] = (v172_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v178_data = ir1[7];
              ir1[7] = (v178_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v184_data = ir1[8];
              ir1[8] = (v184_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v190_data = ir1[9];
              ir1[9] = (v190_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v196_data = ir1[10];
              ir1[10] = (v196_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v202_data = ir1[11];
              ir1[11] = (v202_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v208_data = ir1[12];
              ir1[12] = (v208_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v214_data = ir1[13];
              ir1[13] = (v214_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v220_data = ir1[14];
              ir1[14] = (v220_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v226_data = ir1[15];
              ir1[15] = (v226_data + (v132_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              double v229_data = glb_m1[(v21_lead + 32)];
              double v233_data = ir1[0];
              ir1[0] = (v233_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v239_data = ir1[1];
              ir1[1] = (v239_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v245_data = ir1[2];
              ir1[2] = (v245_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v251_data = ir1[3];
              ir1[3] = (v251_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v257_data = ir1[4];
              ir1[4] = (v257_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v263_data = ir1[5];
              ir1[5] = (v263_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v269_data = ir1[6];
              ir1[6] = (v269_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v275_data = ir1[7];
              ir1[7] = (v275_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v281_data = ir1[8];
              ir1[8] = (v281_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v287_data = ir1[9];
              ir1[9] = (v287_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v293_data = ir1[10];
              ir1[10] = (v293_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v299_data = ir1[11];
              ir1[11] = (v299_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v305_data = ir1[12];
              ir1[12] = (v305_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v311_data = ir1[13];
              ir1[13] = (v311_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v317_data = ir1[14];
              ir1[14] = (v317_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v323_data = ir1[15];
              ir1[15] = (v323_data + (v229_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              double v326_data = glb_m1[(v21_lead + 48)];
              double v330_data = ir1[0];
              ir1[0] = (v330_data + (v326_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v336_data = ir1[1];
              ir1[1] = (v336_data + (v326_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v342_data = ir1[2];
              ir1[2] = (v342_data + (v326_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v348_data = ir1[3];
              ir1[3] = (v348_data + (v326_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v354_data = ir1[4];
              ir1[4] = (v354_data + (v326_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v360_data = ir1[5];
              ir1[5] = (v360_data + (v326_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v366_data = ir1[6];
              ir1[6] = (v366_data + (v326_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v372_data = ir1[7];
              ir1[7] = (v372_data + (v326_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v378_data = ir1[8];
              ir1[8] = (v378_data + (v326_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v384_data = ir1[9];
              ir1[9] = (v384_data + (v326_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v390_data = ir1[10];
              ir1[10] = (v390_data + (v326_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v396_data = ir1[11];
              ir1[11] = (v396_data + (v326_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v402_data = ir1[12];
              ir1[12] = (v402_data + (v326_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v408_data = ir1[13];
              ir1[13] = (v408_data + (v326_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v414_data = ir1[14];
              ir1[14] = (v414_data + (v326_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v420_data = ir1[15];
              ir1[15] = (v420_data + (v326_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              double v423_data = glb_m1[(v21_lead + 64)];
              double v427_data = ir1[0];
              ir1[0] = (v427_data + (v423_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v433_data = ir1[1];
              ir1[1] = (v433_data + (v423_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v439_data = ir1[2];
              ir1[2] = (v439_data + (v423_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v445_data = ir1[3];
              ir1[3] = (v445_data + (v423_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v451_data = ir1[4];
              ir1[4] = (v451_data + (v423_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v457_data = ir1[5];
              ir1[5] = (v457_data + (v423_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v463_data = ir1[6];
              ir1[6] = (v463_data + (v423_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v469_data = ir1[7];
              ir1[7] = (v469_data + (v423_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v475_data = ir1[8];
              ir1[8] = (v475_data + (v423_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v481_data = ir1[9];
              ir1[9] = (v481_data + (v423_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v487_data = ir1[10];
              ir1[10] = (v487_data + (v423_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v493_data = ir1[11];
              ir1[11] = (v493_data + (v423_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v499_data = ir1[12];
              ir1[12] = (v499_data + (v423_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v505_data = ir1[13];
              ir1[13] = (v505_data + (v423_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v511_data = ir1[14];
              ir1[14] = (v511_data + (v423_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v517_data = ir1[15];
              ir1[15] = (v517_data + (v423_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              double v520_data = glb_m1[(v21_lead + 80)];
              double v524_data = ir1[0];
              ir1[0] = (v524_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v530_data = ir1[1];
              ir1[1] = (v530_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v536_data = ir1[2];
              ir1[2] = (v536_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v542_data = ir1[3];
              ir1[3] = (v542_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v548_data = ir1[4];
              ir1[4] = (v548_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v554_data = ir1[5];
              ir1[5] = (v554_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v560_data = ir1[6];
              ir1[6] = (v560_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v566_data = ir1[7];
              ir1[7] = (v566_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v572_data = ir1[8];
              ir1[8] = (v572_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v578_data = ir1[9];
              ir1[9] = (v578_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v584_data = ir1[10];
              ir1[10] = (v584_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v590_data = ir1[11];
              ir1[11] = (v590_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v596_data = ir1[12];
              ir1[12] = (v596_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v602_data = ir1[13];
              ir1[13] = (v602_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v608_data = ir1[14];
              ir1[14] = (v608_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v614_data = ir1[15];
              ir1[15] = (v614_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              double v617_data = glb_m1[(v21_lead + 96)];
              double v621_data = ir1[0];
              ir1[0] = (v621_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v627_data = ir1[1];
              ir1[1] = (v627_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v633_data = ir1[2];
              ir1[2] = (v633_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v639_data = ir1[3];
              ir1[3] = (v639_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v645_data = ir1[4];
              ir1[4] = (v645_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v651_data = ir1[5];
              ir1[5] = (v651_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v657_data = ir1[6];
              ir1[6] = (v657_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v663_data = ir1[7];
              ir1[7] = (v663_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v669_data = ir1[8];
              ir1[8] = (v669_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v675_data = ir1[9];
              ir1[9] = (v675_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v681_data = ir1[10];
              ir1[10] = (v681_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v687_data = ir1[11];
              ir1[11] = (v687_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v693_data = ir1[12];
              ir1[12] = (v693_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v699_data = ir1[13];
              ir1[13] = (v699_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v705_data = ir1[14];
              ir1[14] = (v705_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v711_data = ir1[15];
              ir1[15] = (v711_data + (v617_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              double v714_data = glb_m1[(v21_lead + 112)];
              double v718_data = ir1[0];
              ir1[0] = (v718_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v724_data = ir1[1];
              ir1[1] = (v724_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v730_data = ir1[2];
              ir1[2] = (v730_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v736_data = ir1[3];
              ir1[3] = (v736_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v742_data = ir1[4];
              ir1[4] = (v742_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v748_data = ir1[5];
              ir1[5] = (v748_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v754_data = ir1[6];
              ir1[6] = (v754_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v760_data = ir1[7];
              ir1[7] = (v760_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v766_data = ir1[8];
              ir1[8] = (v766_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v772_data = ir1[9];
              ir1[9] = (v772_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v778_data = ir1[10];
              ir1[10] = (v778_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v784_data = ir1[11];
              ir1[11] = (v784_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v790_data = ir1[12];
              ir1[12] = (v790_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v796_data = ir1[13];
              ir1[13] = (v796_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v802_data = ir1[14];
              ir1[14] = (v802_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v808_data = ir1[15];
              ir1[15] = (v808_data + (v714_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              double v811_data = glb_m1[(v21_lead + 128)];
              double v815_data = ir1[0];
              ir1[0] = (v815_data + (v811_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v821_data = ir1[1];
              ir1[1] = (v821_data + (v811_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v827_data = ir1[2];
              ir1[2] = (v827_data + (v811_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v833_data = ir1[3];
              ir1[3] = (v833_data + (v811_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v839_data = ir1[4];
              ir1[4] = (v839_data + (v811_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v845_data = ir1[5];
              ir1[5] = (v845_data + (v811_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v851_data = ir1[6];
              ir1[6] = (v851_data + (v811_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v857_data = ir1[7];
              ir1[7] = (v857_data + (v811_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v863_data = ir1[8];
              ir1[8] = (v863_data + (v811_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v869_data = ir1[9];
              ir1[9] = (v869_data + (v811_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v875_data = ir1[10];
              ir1[10] = (v875_data + (v811_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v881_data = ir1[11];
              ir1[11] = (v881_data + (v811_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v887_data = ir1[12];
              ir1[12] = (v887_data + (v811_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v893_data = ir1[13];
              ir1[13] = (v893_data + (v811_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v899_data = ir1[14];
              ir1[14] = (v899_data + (v811_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v905_data = ir1[15];
              ir1[15] = (v905_data + (v811_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              double v908_data = glb_m1[(v21_lead + 144)];
              double v912_data = ir1[0];
              ir1[0] = (v912_data + (v908_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v918_data = ir1[1];
              ir1[1] = (v918_data + (v908_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v924_data = ir1[2];
              ir1[2] = (v924_data + (v908_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v930_data = ir1[3];
              ir1[3] = (v930_data + (v908_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v936_data = ir1[4];
              ir1[4] = (v936_data + (v908_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v942_data = ir1[5];
              ir1[5] = (v942_data + (v908_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v948_data = ir1[6];
              ir1[6] = (v948_data + (v908_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v954_data = ir1[7];
              ir1[7] = (v954_data + (v908_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v960_data = ir1[8];
              ir1[8] = (v960_data + (v908_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v966_data = ir1[9];
              ir1[9] = (v966_data + (v908_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v972_data = ir1[10];
              ir1[10] = (v972_data + (v908_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v978_data = ir1[11];
              ir1[11] = (v978_data + (v908_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v984_data = ir1[12];
              ir1[12] = (v984_data + (v908_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v990_data = ir1[13];
              ir1[13] = (v990_data + (v908_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v996_data = ir1[14];
              ir1[14] = (v996_data + (v908_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v1002_data = ir1[15];
              ir1[15] = (v1002_data + (v908_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              double v1005_data = glb_m1[(v21_lead + 160)];
              double v1009_data = ir1[0];
              ir1[0] = (v1009_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v1015_data = ir1[1];
              ir1[1] = (v1015_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v1021_data = ir1[2];
              ir1[2] = (v1021_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v1027_data = ir1[3];
              ir1[3] = (v1027_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v1033_data = ir1[4];
              ir1[4] = (v1033_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v1039_data = ir1[5];
              ir1[5] = (v1039_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v1045_data = ir1[6];
              ir1[6] = (v1045_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v1051_data = ir1[7];
              ir1[7] = (v1051_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v1057_data = ir1[8];
              ir1[8] = (v1057_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v1063_data = ir1[9];
              ir1[9] = (v1063_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v1069_data = ir1[10];
              ir1[10] = (v1069_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v1075_data = ir1[11];
              ir1[11] = (v1075_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v1081_data = ir1[12];
              ir1[12] = (v1081_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v1087_data = ir1[13];
              ir1[13] = (v1087_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v1093_data = ir1[14];
              ir1[14] = (v1093_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v1099_data = ir1[15];
              ir1[15] = (v1099_data + (v1005_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              double v1102_data = glb_m1[(v21_lead + 176)];
              double v1106_data = ir1[0];
              ir1[0] = (v1106_data + (v1102_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v1112_data = ir1[1];
              ir1[1] = (v1112_data + (v1102_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v1118_data = ir1[2];
              ir1[2] = (v1118_data + (v1102_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v1124_data = ir1[3];
              ir1[3] = (v1124_data + (v1102_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v1130_data = ir1[4];
              ir1[4] = (v1130_data + (v1102_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v1136_data = ir1[5];
              ir1[5] = (v1136_data + (v1102_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v1142_data = ir1[6];
              ir1[6] = (v1142_data + (v1102_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v1148_data = ir1[7];
              ir1[7] = (v1148_data + (v1102_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v1154_data = ir1[8];
              ir1[8] = (v1154_data + (v1102_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v1160_data = ir1[9];
              ir1[9] = (v1160_data + (v1102_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v1166_data = ir1[10];
              ir1[10] = (v1166_data + (v1102_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v1172_data = ir1[11];
              ir1[11] = (v1172_data + (v1102_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v1178_data = ir1[12];
              ir1[12] = (v1178_data + (v1102_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v1184_data = ir1[13];
              ir1[13] = (v1184_data + (v1102_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v1190_data = ir1[14];
              ir1[14] = (v1190_data + (v1102_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v1196_data = ir1[15];
              ir1[15] = (v1196_data + (v1102_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              double v1199_data = glb_m1[(v21_lead + 192)];
              double v1203_data = ir1[0];
              ir1[0] = (v1203_data + (v1199_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v1209_data = ir1[1];
              ir1[1] = (v1209_data + (v1199_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v1215_data = ir1[2];
              ir1[2] = (v1215_data + (v1199_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v1221_data = ir1[3];
              ir1[3] = (v1221_data + (v1199_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v1227_data = ir1[4];
              ir1[4] = (v1227_data + (v1199_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v1233_data = ir1[5];
              ir1[5] = (v1233_data + (v1199_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v1239_data = ir1[6];
              ir1[6] = (v1239_data + (v1199_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v1245_data = ir1[7];
              ir1[7] = (v1245_data + (v1199_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v1251_data = ir1[8];
              ir1[8] = (v1251_data + (v1199_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v1257_data = ir1[9];
              ir1[9] = (v1257_data + (v1199_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v1263_data = ir1[10];
              ir1[10] = (v1263_data + (v1199_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v1269_data = ir1[11];
              ir1[11] = (v1269_data + (v1199_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v1275_data = ir1[12];
              ir1[12] = (v1275_data + (v1199_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v1281_data = ir1[13];
              ir1[13] = (v1281_data + (v1199_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v1287_data = ir1[14];
              ir1[14] = (v1287_data + (v1199_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v1293_data = ir1[15];
              ir1[15] = (v1293_data + (v1199_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              double v1296_data = glb_m1[(v21_lead + 208)];
              double v1300_data = ir1[0];
              ir1[0] = (v1300_data + (v1296_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v1306_data = ir1[1];
              ir1[1] = (v1306_data + (v1296_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v1312_data = ir1[2];
              ir1[2] = (v1312_data + (v1296_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v1318_data = ir1[3];
              ir1[3] = (v1318_data + (v1296_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v1324_data = ir1[4];
              ir1[4] = (v1324_data + (v1296_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v1330_data = ir1[5];
              ir1[5] = (v1330_data + (v1296_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v1336_data = ir1[6];
              ir1[6] = (v1336_data + (v1296_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v1342_data = ir1[7];
              ir1[7] = (v1342_data + (v1296_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v1348_data = ir1[8];
              ir1[8] = (v1348_data + (v1296_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v1354_data = ir1[9];
              ir1[9] = (v1354_data + (v1296_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v1360_data = ir1[10];
              ir1[10] = (v1360_data + (v1296_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v1366_data = ir1[11];
              ir1[11] = (v1366_data + (v1296_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v1372_data = ir1[12];
              ir1[12] = (v1372_data + (v1296_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v1378_data = ir1[13];
              ir1[13] = (v1378_data + (v1296_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v1384_data = ir1[14];
              ir1[14] = (v1384_data + (v1296_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v1390_data = ir1[15];
              ir1[15] = (v1390_data + (v1296_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              double v1393_data = glb_m1[(v21_lead + 224)];
              double v1397_data = ir1[0];
              ir1[0] = (v1397_data + (v1393_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v1403_data = ir1[1];
              ir1[1] = (v1403_data + (v1393_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v1409_data = ir1[2];
              ir1[2] = (v1409_data + (v1393_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v1415_data = ir1[3];
              ir1[3] = (v1415_data + (v1393_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v1421_data = ir1[4];
              ir1[4] = (v1421_data + (v1393_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v1427_data = ir1[5];
              ir1[5] = (v1427_data + (v1393_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v1433_data = ir1[6];
              ir1[6] = (v1433_data + (v1393_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v1439_data = ir1[7];
              ir1[7] = (v1439_data + (v1393_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v1445_data = ir1[8];
              ir1[8] = (v1445_data + (v1393_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v1451_data = ir1[9];
              ir1[9] = (v1451_data + (v1393_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v1457_data = ir1[10];
              ir1[10] = (v1457_data + (v1393_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v1463_data = ir1[11];
              ir1[11] = (v1463_data + (v1393_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v1469_data = ir1[12];
              ir1[12] = (v1469_data + (v1393_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v1475_data = ir1[13];
              ir1[13] = (v1475_data + (v1393_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v1481_data = ir1[14];
              ir1[14] = (v1481_data + (v1393_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v1487_data = ir1[15];
              ir1[15] = (v1487_data + (v1393_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              double v1490_data = glb_m1[(v21_lead + 240)];
              double v1494_data = ir1[0];
              ir1[0] = (v1494_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v36_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v1500_data = ir1[1];
              ir1[1] = (v1500_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v42_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v1506_data = ir1[2];
              ir1[2] = (v1506_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v48_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v1512_data = ir1[3];
              ir1[3] = (v1512_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v54_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v1518_data = ir1[4];
              ir1[4] = (v1518_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v60_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v1524_data = ir1[5];
              ir1[5] = (v1524_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v66_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v1530_data = ir1[6];
              ir1[6] = (v1530_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v72_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v1536_data = ir1[7];
              ir1[7] = (v1536_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v78_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v1542_data = ir1[8];
              ir1[8] = (v1542_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v84_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v1548_data = ir1[9];
              ir1[9] = (v1548_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v90_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v1554_data = ir1[10];
              ir1[10] = (v1554_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v96_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v1560_data = ir1[11];
              ir1[11] = (v1560_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v102_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v1566_data = ir1[12];
              ir1[12] = (v1566_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v108_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v1572_data = ir1[13];
              ir1[13] = (v1572_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v114_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v1578_data = ir1[14];
              ir1[14] = (v1578_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v120_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              double v1584_data = ir1[15];
              ir1[15] = (v1584_data + (v1490_data * (sycl::select_from_group(item.get_sub_group(), v126_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              // r1 = ir1
              #pragma unroll
              for (int32_t v1586_n0 = 0; v1586_n0 < 1; ++v1586_n0) {
                #pragma unroll
                for (int32_t v1587_n1 = 0; v1587_n1 < 16; ++v1587_n1) {
                  int32_t v1588_a = v1586_n0 + v1587_n1;
                  double v1589_data = ir1[v1588_a];
                  r1[v1588_a] = v1589_data;
                }
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v1590_i0 = 0; v1590_i0 < 1; ++v1590_i0) {
                int32_t v1595_lead = v21_lead + (v1590_i0 * 16);
                #pragma unroll
                for (int32_t v1591_i1 = 0; v1591_i1 < 16; ++v1591_i1) {
                  double v1593_data = r1[(v1590_i0 + v1591_i1)];
                  glb_m0[(v1595_lead + (v1591_i1 * 16))] = v1593_data;
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

