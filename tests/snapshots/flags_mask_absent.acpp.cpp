// === base name ===
kernel_5f9368264237bfc4

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_5f9368264237bfc4 = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_5f9368264237bfc4(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_5f9368264237bfc4(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_5f9368264237bfc4(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_5f9368264237bfc4(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_5f9368264237bfc4(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_5f9368264237bfc4(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_5f9368264237bfc4(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        using namespace tensorforge::literals;
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 16×16(16×16) {0..16}×{0..16} strided
        //   m1 16×16(16×16) {0..16}×{0..16} strided
        //   m2 16×16(16×16) {0..16}×{0..16} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1"}
        {
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          int32_t v20_lead = item.get_local_id(2) % 16;
          for (size_t v7_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v7_batchId0 < numElements0; v7_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v8_ahead1 = v7_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v10_batchId1 = (v8_ahead1 < numElements0) ? v8_ahead1 : v7_batchId0;
            float *const __restrict__ glb_m0 = &m0[v7_batchId0 * 256 + 0 + m0_extraOffset];
            const float *const __restrict__ glb_m1 = &m1[v7_batchId0 * 256 + 0 + m1_extraOffset];
            const float *const __restrict__ glb_m2 = &m2[v7_batchId0 * 256 + 0 + m2_extraOffset];
            float r0[16]{};
            // r0 = load{g>r}(glb_m1);
            #pragma unroll
            for (int32_t v21_i0 = 0; v21_i0 < 1; ++v21_i0) {
              int32_t v24_lead = v20_lead + (v21_i0 * 16);
              #pragma unroll
              for (int32_t v22_i1 = 0; v22_i1 < 16; ++v22_i1) {
                float v27_data = glb_m1[(v24_lead + (v22_i1 * 16))];
                r0[(v21_i0 + v22_i1)] = v27_data;
              }
            }
            float r1[16]{};
            // r1 = load{g>r}(glb_m2);
            #pragma unroll
            for (int32_t v30_i0 = 0; v30_i0 < 1; ++v30_i0) {
              int32_t v33_lead = v20_lead + (v30_i0 * 16);
              #pragma unroll
              for (int32_t v31_i1 = 0; v31_i1 < 16; ++v31_i1) {
                float v36_data = glb_m2[(v33_lead + (v31_i1 * 16))];
                r1[(v30_i0 + v31_i1)] = v36_data;
              }
            }
            // wait(r0 = load{g>r}(glb_m1););
            // wait(r1 = load{g>r}(glb_m2););
            float r2[16]{};
            // ir2 = +(r0 * r1)
            // [(0, 16), (0, 16)] [(0, 16)]
            float ir2[16]{};
            float v40_data = r0[0];
            float v41_data = r1[0];
            float v44_data = ir2[0];
            ir2[0] = (v44_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v47_data = r1[1];
            float v50_data = ir2[1];
            ir2[1] = (v50_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v53_data = r1[2];
            float v56_data = ir2[2];
            ir2[2] = (v56_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v59_data = r1[3];
            float v62_data = ir2[3];
            ir2[3] = (v62_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v65_data = r1[4];
            float v68_data = ir2[4];
            ir2[4] = (v68_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v71_data = r1[5];
            float v74_data = ir2[5];
            ir2[5] = (v74_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v77_data = r1[6];
            float v80_data = ir2[6];
            ir2[6] = (v80_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v83_data = r1[7];
            float v86_data = ir2[7];
            ir2[7] = (v86_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v89_data = r1[8];
            float v92_data = ir2[8];
            ir2[8] = (v92_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v95_data = r1[9];
            float v98_data = ir2[9];
            ir2[9] = (v98_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v101_data = r1[10];
            float v104_data = ir2[10];
            ir2[10] = (v104_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v107_data = r1[11];
            float v110_data = ir2[11];
            ir2[11] = (v110_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v113_data = r1[12];
            float v116_data = ir2[12];
            ir2[12] = (v116_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v119_data = r1[13];
            float v122_data = ir2[13];
            ir2[13] = (v122_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v125_data = r1[14];
            float v128_data = ir2[14];
            ir2[14] = (v128_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v125_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v131_data = r1[15];
            float v134_data = ir2[15];
            ir2[15] = (v134_data + (v40_data * (sycl::select_from_group(item.get_sub_group(), v131_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
            float v136_data = r0[1];
            float v140_data = ir2[0];
            ir2[0] = (v140_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v146_data = ir2[1];
            ir2[1] = (v146_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v152_data = ir2[2];
            ir2[2] = (v152_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v158_data = ir2[3];
            ir2[3] = (v158_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v164_data = ir2[4];
            ir2[4] = (v164_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v170_data = ir2[5];
            ir2[5] = (v170_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v176_data = ir2[6];
            ir2[6] = (v176_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v182_data = ir2[7];
            ir2[7] = (v182_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v188_data = ir2[8];
            ir2[8] = (v188_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v194_data = ir2[9];
            ir2[9] = (v194_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v200_data = ir2[10];
            ir2[10] = (v200_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v206_data = ir2[11];
            ir2[11] = (v206_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v212_data = ir2[12];
            ir2[12] = (v212_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v218_data = ir2[13];
            ir2[13] = (v218_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v224_data = ir2[14];
            ir2[14] = (v224_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v125_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v230_data = ir2[15];
            ir2[15] = (v230_data + (v136_data * (sycl::select_from_group(item.get_sub_group(), v131_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
            float v232_data = r0[2];
            float v236_data = ir2[0];
            ir2[0] = (v236_data + (v232_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v242_data = ir2[1];
            ir2[1] = (v242_data + (v232_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v248_data = ir2[2];
            ir2[2] = (v248_data + (v232_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v254_data = ir2[3];
            ir2[3] = (v254_data + (v232_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v260_data = ir2[4];
            ir2[4] = (v260_data + (v232_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v266_data = ir2[5];
            ir2[5] = (v266_data + (v232_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v272_data = ir2[6];
            ir2[6] = (v272_data + (v232_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v278_data = ir2[7];
            ir2[7] = (v278_data + (v232_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v284_data = ir2[8];
            ir2[8] = (v284_data + (v232_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v290_data = ir2[9];
            ir2[9] = (v290_data + (v232_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v296_data = ir2[10];
            ir2[10] = (v296_data + (v232_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v302_data = ir2[11];
            ir2[11] = (v302_data + (v232_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v308_data = ir2[12];
            ir2[12] = (v308_data + (v232_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v314_data = ir2[13];
            ir2[13] = (v314_data + (v232_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v320_data = ir2[14];
            ir2[14] = (v320_data + (v232_data * (sycl::select_from_group(item.get_sub_group(), v125_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v326_data = ir2[15];
            ir2[15] = (v326_data + (v232_data * (sycl::select_from_group(item.get_sub_group(), v131_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
            float v328_data = r0[3];
            float v332_data = ir2[0];
            ir2[0] = (v332_data + (v328_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v338_data = ir2[1];
            ir2[1] = (v338_data + (v328_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v344_data = ir2[2];
            ir2[2] = (v344_data + (v328_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v350_data = ir2[3];
            ir2[3] = (v350_data + (v328_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v356_data = ir2[4];
            ir2[4] = (v356_data + (v328_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v362_data = ir2[5];
            ir2[5] = (v362_data + (v328_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v368_data = ir2[6];
            ir2[6] = (v368_data + (v328_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v374_data = ir2[7];
            ir2[7] = (v374_data + (v328_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v380_data = ir2[8];
            ir2[8] = (v380_data + (v328_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v386_data = ir2[9];
            ir2[9] = (v386_data + (v328_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v392_data = ir2[10];
            ir2[10] = (v392_data + (v328_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v398_data = ir2[11];
            ir2[11] = (v398_data + (v328_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v404_data = ir2[12];
            ir2[12] = (v404_data + (v328_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v410_data = ir2[13];
            ir2[13] = (v410_data + (v328_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v416_data = ir2[14];
            ir2[14] = (v416_data + (v328_data * (sycl::select_from_group(item.get_sub_group(), v125_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v422_data = ir2[15];
            ir2[15] = (v422_data + (v328_data * (sycl::select_from_group(item.get_sub_group(), v131_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
            float v424_data = r0[4];
            float v428_data = ir2[0];
            ir2[0] = (v428_data + (v424_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v434_data = ir2[1];
            ir2[1] = (v434_data + (v424_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v440_data = ir2[2];
            ir2[2] = (v440_data + (v424_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v446_data = ir2[3];
            ir2[3] = (v446_data + (v424_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v452_data = ir2[4];
            ir2[4] = (v452_data + (v424_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v458_data = ir2[5];
            ir2[5] = (v458_data + (v424_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v464_data = ir2[6];
            ir2[6] = (v464_data + (v424_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v470_data = ir2[7];
            ir2[7] = (v470_data + (v424_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v476_data = ir2[8];
            ir2[8] = (v476_data + (v424_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v482_data = ir2[9];
            ir2[9] = (v482_data + (v424_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v488_data = ir2[10];
            ir2[10] = (v488_data + (v424_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v494_data = ir2[11];
            ir2[11] = (v494_data + (v424_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v500_data = ir2[12];
            ir2[12] = (v500_data + (v424_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v506_data = ir2[13];
            ir2[13] = (v506_data + (v424_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v512_data = ir2[14];
            ir2[14] = (v512_data + (v424_data * (sycl::select_from_group(item.get_sub_group(), v125_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v518_data = ir2[15];
            ir2[15] = (v518_data + (v424_data * (sycl::select_from_group(item.get_sub_group(), v131_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
            float v520_data = r0[5];
            float v524_data = ir2[0];
            ir2[0] = (v524_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v530_data = ir2[1];
            ir2[1] = (v530_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v536_data = ir2[2];
            ir2[2] = (v536_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v542_data = ir2[3];
            ir2[3] = (v542_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v548_data = ir2[4];
            ir2[4] = (v548_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v554_data = ir2[5];
            ir2[5] = (v554_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v560_data = ir2[6];
            ir2[6] = (v560_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v566_data = ir2[7];
            ir2[7] = (v566_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v572_data = ir2[8];
            ir2[8] = (v572_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v578_data = ir2[9];
            ir2[9] = (v578_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v584_data = ir2[10];
            ir2[10] = (v584_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v590_data = ir2[11];
            ir2[11] = (v590_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v596_data = ir2[12];
            ir2[12] = (v596_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v602_data = ir2[13];
            ir2[13] = (v602_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v608_data = ir2[14];
            ir2[14] = (v608_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v125_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v614_data = ir2[15];
            ir2[15] = (v614_data + (v520_data * (sycl::select_from_group(item.get_sub_group(), v131_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
            float v616_data = r0[6];
            float v620_data = ir2[0];
            ir2[0] = (v620_data + (v616_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v626_data = ir2[1];
            ir2[1] = (v626_data + (v616_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v632_data = ir2[2];
            ir2[2] = (v632_data + (v616_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v638_data = ir2[3];
            ir2[3] = (v638_data + (v616_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v644_data = ir2[4];
            ir2[4] = (v644_data + (v616_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v650_data = ir2[5];
            ir2[5] = (v650_data + (v616_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v656_data = ir2[6];
            ir2[6] = (v656_data + (v616_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v662_data = ir2[7];
            ir2[7] = (v662_data + (v616_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v668_data = ir2[8];
            ir2[8] = (v668_data + (v616_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v674_data = ir2[9];
            ir2[9] = (v674_data + (v616_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v680_data = ir2[10];
            ir2[10] = (v680_data + (v616_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v686_data = ir2[11];
            ir2[11] = (v686_data + (v616_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v692_data = ir2[12];
            ir2[12] = (v692_data + (v616_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v698_data = ir2[13];
            ir2[13] = (v698_data + (v616_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v704_data = ir2[14];
            ir2[14] = (v704_data + (v616_data * (sycl::select_from_group(item.get_sub_group(), v125_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v710_data = ir2[15];
            ir2[15] = (v710_data + (v616_data * (sycl::select_from_group(item.get_sub_group(), v131_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
            float v712_data = r0[7];
            float v716_data = ir2[0];
            ir2[0] = (v716_data + (v712_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v722_data = ir2[1];
            ir2[1] = (v722_data + (v712_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v728_data = ir2[2];
            ir2[2] = (v728_data + (v712_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v734_data = ir2[3];
            ir2[3] = (v734_data + (v712_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v740_data = ir2[4];
            ir2[4] = (v740_data + (v712_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v746_data = ir2[5];
            ir2[5] = (v746_data + (v712_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v752_data = ir2[6];
            ir2[6] = (v752_data + (v712_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v758_data = ir2[7];
            ir2[7] = (v758_data + (v712_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v764_data = ir2[8];
            ir2[8] = (v764_data + (v712_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v770_data = ir2[9];
            ir2[9] = (v770_data + (v712_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v776_data = ir2[10];
            ir2[10] = (v776_data + (v712_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v782_data = ir2[11];
            ir2[11] = (v782_data + (v712_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v788_data = ir2[12];
            ir2[12] = (v788_data + (v712_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v794_data = ir2[13];
            ir2[13] = (v794_data + (v712_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v800_data = ir2[14];
            ir2[14] = (v800_data + (v712_data * (sycl::select_from_group(item.get_sub_group(), v125_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v806_data = ir2[15];
            ir2[15] = (v806_data + (v712_data * (sycl::select_from_group(item.get_sub_group(), v131_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
            float v808_data = r0[8];
            float v812_data = ir2[0];
            ir2[0] = (v812_data + (v808_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v818_data = ir2[1];
            ir2[1] = (v818_data + (v808_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v824_data = ir2[2];
            ir2[2] = (v824_data + (v808_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v830_data = ir2[3];
            ir2[3] = (v830_data + (v808_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v836_data = ir2[4];
            ir2[4] = (v836_data + (v808_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v842_data = ir2[5];
            ir2[5] = (v842_data + (v808_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v848_data = ir2[6];
            ir2[6] = (v848_data + (v808_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v854_data = ir2[7];
            ir2[7] = (v854_data + (v808_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v860_data = ir2[8];
            ir2[8] = (v860_data + (v808_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v866_data = ir2[9];
            ir2[9] = (v866_data + (v808_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v872_data = ir2[10];
            ir2[10] = (v872_data + (v808_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v878_data = ir2[11];
            ir2[11] = (v878_data + (v808_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v884_data = ir2[12];
            ir2[12] = (v884_data + (v808_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v890_data = ir2[13];
            ir2[13] = (v890_data + (v808_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v896_data = ir2[14];
            ir2[14] = (v896_data + (v808_data * (sycl::select_from_group(item.get_sub_group(), v125_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v902_data = ir2[15];
            ir2[15] = (v902_data + (v808_data * (sycl::select_from_group(item.get_sub_group(), v131_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
            float v904_data = r0[9];
            float v908_data = ir2[0];
            ir2[0] = (v908_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v914_data = ir2[1];
            ir2[1] = (v914_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v920_data = ir2[2];
            ir2[2] = (v920_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v926_data = ir2[3];
            ir2[3] = (v926_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v932_data = ir2[4];
            ir2[4] = (v932_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v938_data = ir2[5];
            ir2[5] = (v938_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v944_data = ir2[6];
            ir2[6] = (v944_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v950_data = ir2[7];
            ir2[7] = (v950_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v956_data = ir2[8];
            ir2[8] = (v956_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v962_data = ir2[9];
            ir2[9] = (v962_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v968_data = ir2[10];
            ir2[10] = (v968_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v974_data = ir2[11];
            ir2[11] = (v974_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v980_data = ir2[12];
            ir2[12] = (v980_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v986_data = ir2[13];
            ir2[13] = (v986_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v992_data = ir2[14];
            ir2[14] = (v992_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v125_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v998_data = ir2[15];
            ir2[15] = (v998_data + (v904_data * (sycl::select_from_group(item.get_sub_group(), v131_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
            float v1000_data = r0[10];
            float v1004_data = ir2[0];
            ir2[0] = (v1004_data + (v1000_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1010_data = ir2[1];
            ir2[1] = (v1010_data + (v1000_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1016_data = ir2[2];
            ir2[2] = (v1016_data + (v1000_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1022_data = ir2[3];
            ir2[3] = (v1022_data + (v1000_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1028_data = ir2[4];
            ir2[4] = (v1028_data + (v1000_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1034_data = ir2[5];
            ir2[5] = (v1034_data + (v1000_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1040_data = ir2[6];
            ir2[6] = (v1040_data + (v1000_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1046_data = ir2[7];
            ir2[7] = (v1046_data + (v1000_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1052_data = ir2[8];
            ir2[8] = (v1052_data + (v1000_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1058_data = ir2[9];
            ir2[9] = (v1058_data + (v1000_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1064_data = ir2[10];
            ir2[10] = (v1064_data + (v1000_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1070_data = ir2[11];
            ir2[11] = (v1070_data + (v1000_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1076_data = ir2[12];
            ir2[12] = (v1076_data + (v1000_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1082_data = ir2[13];
            ir2[13] = (v1082_data + (v1000_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1088_data = ir2[14];
            ir2[14] = (v1088_data + (v1000_data * (sycl::select_from_group(item.get_sub_group(), v125_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1094_data = ir2[15];
            ir2[15] = (v1094_data + (v1000_data * (sycl::select_from_group(item.get_sub_group(), v131_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
            float v1096_data = r0[11];
            float v1100_data = ir2[0];
            ir2[0] = (v1100_data + (v1096_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1106_data = ir2[1];
            ir2[1] = (v1106_data + (v1096_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1112_data = ir2[2];
            ir2[2] = (v1112_data + (v1096_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1118_data = ir2[3];
            ir2[3] = (v1118_data + (v1096_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1124_data = ir2[4];
            ir2[4] = (v1124_data + (v1096_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1130_data = ir2[5];
            ir2[5] = (v1130_data + (v1096_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1136_data = ir2[6];
            ir2[6] = (v1136_data + (v1096_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1142_data = ir2[7];
            ir2[7] = (v1142_data + (v1096_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1148_data = ir2[8];
            ir2[8] = (v1148_data + (v1096_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1154_data = ir2[9];
            ir2[9] = (v1154_data + (v1096_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1160_data = ir2[10];
            ir2[10] = (v1160_data + (v1096_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1166_data = ir2[11];
            ir2[11] = (v1166_data + (v1096_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1172_data = ir2[12];
            ir2[12] = (v1172_data + (v1096_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1178_data = ir2[13];
            ir2[13] = (v1178_data + (v1096_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1184_data = ir2[14];
            ir2[14] = (v1184_data + (v1096_data * (sycl::select_from_group(item.get_sub_group(), v125_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1190_data = ir2[15];
            ir2[15] = (v1190_data + (v1096_data * (sycl::select_from_group(item.get_sub_group(), v131_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
            float v1192_data = r0[12];
            float v1196_data = ir2[0];
            ir2[0] = (v1196_data + (v1192_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1202_data = ir2[1];
            ir2[1] = (v1202_data + (v1192_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1208_data = ir2[2];
            ir2[2] = (v1208_data + (v1192_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1214_data = ir2[3];
            ir2[3] = (v1214_data + (v1192_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1220_data = ir2[4];
            ir2[4] = (v1220_data + (v1192_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1226_data = ir2[5];
            ir2[5] = (v1226_data + (v1192_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1232_data = ir2[6];
            ir2[6] = (v1232_data + (v1192_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1238_data = ir2[7];
            ir2[7] = (v1238_data + (v1192_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1244_data = ir2[8];
            ir2[8] = (v1244_data + (v1192_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1250_data = ir2[9];
            ir2[9] = (v1250_data + (v1192_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1256_data = ir2[10];
            ir2[10] = (v1256_data + (v1192_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1262_data = ir2[11];
            ir2[11] = (v1262_data + (v1192_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1268_data = ir2[12];
            ir2[12] = (v1268_data + (v1192_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1274_data = ir2[13];
            ir2[13] = (v1274_data + (v1192_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1280_data = ir2[14];
            ir2[14] = (v1280_data + (v1192_data * (sycl::select_from_group(item.get_sub_group(), v125_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1286_data = ir2[15];
            ir2[15] = (v1286_data + (v1192_data * (sycl::select_from_group(item.get_sub_group(), v131_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
            float v1288_data = r0[13];
            float v1292_data = ir2[0];
            ir2[0] = (v1292_data + (v1288_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1298_data = ir2[1];
            ir2[1] = (v1298_data + (v1288_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1304_data = ir2[2];
            ir2[2] = (v1304_data + (v1288_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1310_data = ir2[3];
            ir2[3] = (v1310_data + (v1288_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1316_data = ir2[4];
            ir2[4] = (v1316_data + (v1288_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1322_data = ir2[5];
            ir2[5] = (v1322_data + (v1288_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1328_data = ir2[6];
            ir2[6] = (v1328_data + (v1288_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1334_data = ir2[7];
            ir2[7] = (v1334_data + (v1288_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1340_data = ir2[8];
            ir2[8] = (v1340_data + (v1288_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1346_data = ir2[9];
            ir2[9] = (v1346_data + (v1288_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1352_data = ir2[10];
            ir2[10] = (v1352_data + (v1288_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1358_data = ir2[11];
            ir2[11] = (v1358_data + (v1288_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1364_data = ir2[12];
            ir2[12] = (v1364_data + (v1288_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1370_data = ir2[13];
            ir2[13] = (v1370_data + (v1288_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1376_data = ir2[14];
            ir2[14] = (v1376_data + (v1288_data * (sycl::select_from_group(item.get_sub_group(), v125_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1382_data = ir2[15];
            ir2[15] = (v1382_data + (v1288_data * (sycl::select_from_group(item.get_sub_group(), v131_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
            float v1384_data = r0[14];
            float v1388_data = ir2[0];
            ir2[0] = (v1388_data + (v1384_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1394_data = ir2[1];
            ir2[1] = (v1394_data + (v1384_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1400_data = ir2[2];
            ir2[2] = (v1400_data + (v1384_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1406_data = ir2[3];
            ir2[3] = (v1406_data + (v1384_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1412_data = ir2[4];
            ir2[4] = (v1412_data + (v1384_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1418_data = ir2[5];
            ir2[5] = (v1418_data + (v1384_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1424_data = ir2[6];
            ir2[6] = (v1424_data + (v1384_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1430_data = ir2[7];
            ir2[7] = (v1430_data + (v1384_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1436_data = ir2[8];
            ir2[8] = (v1436_data + (v1384_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1442_data = ir2[9];
            ir2[9] = (v1442_data + (v1384_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1448_data = ir2[10];
            ir2[10] = (v1448_data + (v1384_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1454_data = ir2[11];
            ir2[11] = (v1454_data + (v1384_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1460_data = ir2[12];
            ir2[12] = (v1460_data + (v1384_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1466_data = ir2[13];
            ir2[13] = (v1466_data + (v1384_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1472_data = ir2[14];
            ir2[14] = (v1472_data + (v1384_data * (sycl::select_from_group(item.get_sub_group(), v125_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1478_data = ir2[15];
            ir2[15] = (v1478_data + (v1384_data * (sycl::select_from_group(item.get_sub_group(), v131_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
            float v1480_data = r0[15];
            float v1484_data = ir2[0];
            ir2[0] = (v1484_data + (v1480_data * (sycl::select_from_group(item.get_sub_group(), v41_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1490_data = ir2[1];
            ir2[1] = (v1490_data + (v1480_data * (sycl::select_from_group(item.get_sub_group(), v47_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1496_data = ir2[2];
            ir2[2] = (v1496_data + (v1480_data * (sycl::select_from_group(item.get_sub_group(), v53_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1502_data = ir2[3];
            ir2[3] = (v1502_data + (v1480_data * (sycl::select_from_group(item.get_sub_group(), v59_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1508_data = ir2[4];
            ir2[4] = (v1508_data + (v1480_data * (sycl::select_from_group(item.get_sub_group(), v65_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1514_data = ir2[5];
            ir2[5] = (v1514_data + (v1480_data * (sycl::select_from_group(item.get_sub_group(), v71_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1520_data = ir2[6];
            ir2[6] = (v1520_data + (v1480_data * (sycl::select_from_group(item.get_sub_group(), v77_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1526_data = ir2[7];
            ir2[7] = (v1526_data + (v1480_data * (sycl::select_from_group(item.get_sub_group(), v83_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1532_data = ir2[8];
            ir2[8] = (v1532_data + (v1480_data * (sycl::select_from_group(item.get_sub_group(), v89_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1538_data = ir2[9];
            ir2[9] = (v1538_data + (v1480_data * (sycl::select_from_group(item.get_sub_group(), v95_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1544_data = ir2[10];
            ir2[10] = (v1544_data + (v1480_data * (sycl::select_from_group(item.get_sub_group(), v101_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1550_data = ir2[11];
            ir2[11] = (v1550_data + (v1480_data * (sycl::select_from_group(item.get_sub_group(), v107_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1556_data = ir2[12];
            ir2[12] = (v1556_data + (v1480_data * (sycl::select_from_group(item.get_sub_group(), v113_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1562_data = ir2[13];
            ir2[13] = (v1562_data + (v1480_data * (sycl::select_from_group(item.get_sub_group(), v119_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1568_data = ir2[14];
            ir2[14] = (v1568_data + (v1480_data * (sycl::select_from_group(item.get_sub_group(), v125_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            float v1574_data = ir2[15];
            ir2[15] = (v1574_data + (v1480_data * (sycl::select_from_group(item.get_sub_group(), v131_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
            // r2 = ir2
            #pragma unroll
            for (int32_t v1576_n0 = 0; v1576_n0 < 1; ++v1576_n0) {
              #pragma unroll
              for (int32_t v1577_n1 = 0; v1577_n1 < 16; ++v1577_n1) {
                int32_t v1578_a = v1576_n0 + v1577_n1;
                float v1579_data = ir2[v1578_a];
                r2[v1578_a] = v1579_data;
              }
            }
            // glb_m0 = store{r>g}(r2);
            #pragma unroll
            for (int32_t v1580_i0 = 0; v1580_i0 < 1; ++v1580_i0) {
              int32_t v1585_lead = v20_lead + (v1580_i0 * 16);
              #pragma unroll
              for (int32_t v1581_i1 = 0; v1581_i1 < 16; ++v1581_i1) {
                float v1583_data = r2[(v1580_i0 + v1581_i1)];
                glb_m0[(v1585_lead + (v1581_i1 * 16))] = v1583_data;
              }
            }
            sycl::group_barrier(item.get_sub_group());
          }
        }
      });
    }
  });
}

