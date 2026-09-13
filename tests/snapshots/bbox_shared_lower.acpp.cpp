// === base name ===
kernel_981c81a6404e8c8a

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_981c81a6404e8c8a = {{16, 16, 1}, 16, 12, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_981c81a6404e8c8a(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_981c81a6404e8c8a(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_981c81a6404e8c8a(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_981c81a6404e8c8a(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_981c81a6404e8c8a(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_981c81a6404e8c8a(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_981c81a6404e8c8a(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes (12 active) x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 16×8(12×8) {4..16}×{0..8} strided
        //   m1 16×16(12×16) {4..16}×{0..16} strided
        //   m2 16×8(16×8) {0..16}×{0..8} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":12,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[4,0],[16,8]],"name":"m0","ordered":false,"parts":1,"shape":[16,8],"variant":false},{"addressing":"strided","alias":"A","bbox":[[4,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,8]],"name":"m2","ordered":false,"parts":1,"shape":[16,8],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[4,0],[16,8]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,8]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[4,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,8]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,8]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
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
              const float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 192 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v3_batchId0 * 128 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v17_lead = item.get_local_id(2) % 16;
              bool v18_g = v17_lead < 12;
              if (v18_g) {
                int32_t v23_a = (v17_lead + 4) - 4;
                #pragma unroll
                for (int32_t v19_i1 = 0; v19_i1 < 16; ++v19_i1) {
                  float v26_data = glb_m1[(v23_a + (v19_i1 * 12))];
                  r0[v19_i1] = v26_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v29_i0 = 0; v29_i0 < 1; ++v29_i0) {
                int32_t v32_lead = v17_lead + (v29_i0 * 16);
                #pragma unroll
                for (int32_t v30_i1 = 0; v30_i1 < 8; ++v30_i1) {
                  float v35_data = glb_m2[(v32_lead + (v30_i1 * 16))];
                  r1[(v29_i0 + v30_i1)] = v35_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(16, 28), (0, 8)] [(0, 16)]
              float ir2[8]{};
              float v40_data = r1[0];
              float v41_bc = sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v46_data = r1[1];
              float v47_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v52_data = r1[2];
              float v53_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v58_data = r1[3];
              float v59_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v64_data = r1[4];
              float v65_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v70_data = r1[5];
              float v71_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v76_data = r1[6];
              float v77_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              float v82_data = r1[7];
              float v83_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0));
              if (v18_g) {
                float v39_data = r0[0];
                float v43_data = ir2[0];
                ir2[0] = (v43_data + (v39_data * v41_bc));
                float v49_data = ir2[1];
                ir2[1] = (v49_data + (v39_data * v47_bc));
                float v55_data = ir2[2];
                ir2[2] = (v55_data + (v39_data * v53_bc));
                float v61_data = ir2[3];
                ir2[3] = (v61_data + (v39_data * v59_bc));
                float v67_data = ir2[4];
                ir2[4] = (v67_data + (v39_data * v65_bc));
                float v73_data = ir2[5];
                ir2[5] = (v73_data + (v39_data * v71_bc));
                float v79_data = ir2[6];
                ir2[6] = (v79_data + (v39_data * v77_bc));
                float v85_data = ir2[7];
                ir2[7] = (v85_data + (v39_data * v83_bc));
              }
              float v89_bc = sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v95_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v101_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v107_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v113_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v119_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v125_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              float v131_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1));
              if (v18_g) {
                float v87_data = r0[1];
                float v91_data = ir2[0];
                ir2[0] = (v91_data + (v87_data * v89_bc));
                float v97_data = ir2[1];
                ir2[1] = (v97_data + (v87_data * v95_bc));
                float v103_data = ir2[2];
                ir2[2] = (v103_data + (v87_data * v101_bc));
                float v109_data = ir2[3];
                ir2[3] = (v109_data + (v87_data * v107_bc));
                float v115_data = ir2[4];
                ir2[4] = (v115_data + (v87_data * v113_bc));
                float v121_data = ir2[5];
                ir2[5] = (v121_data + (v87_data * v119_bc));
                float v127_data = ir2[6];
                ir2[6] = (v127_data + (v87_data * v125_bc));
                float v133_data = ir2[7];
                ir2[7] = (v133_data + (v87_data * v131_bc));
              }
              float v137_bc = sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v143_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v149_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v155_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v161_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v167_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v173_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              float v179_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2));
              if (v18_g) {
                float v135_data = r0[2];
                float v139_data = ir2[0];
                ir2[0] = (v139_data + (v135_data * v137_bc));
                float v145_data = ir2[1];
                ir2[1] = (v145_data + (v135_data * v143_bc));
                float v151_data = ir2[2];
                ir2[2] = (v151_data + (v135_data * v149_bc));
                float v157_data = ir2[3];
                ir2[3] = (v157_data + (v135_data * v155_bc));
                float v163_data = ir2[4];
                ir2[4] = (v163_data + (v135_data * v161_bc));
                float v169_data = ir2[5];
                ir2[5] = (v169_data + (v135_data * v167_bc));
                float v175_data = ir2[6];
                ir2[6] = (v175_data + (v135_data * v173_bc));
                float v181_data = ir2[7];
                ir2[7] = (v181_data + (v135_data * v179_bc));
              }
              float v185_bc = sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v191_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v197_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v203_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v209_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v215_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v221_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              float v227_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3));
              if (v18_g) {
                float v183_data = r0[3];
                float v187_data = ir2[0];
                ir2[0] = (v187_data + (v183_data * v185_bc));
                float v193_data = ir2[1];
                ir2[1] = (v193_data + (v183_data * v191_bc));
                float v199_data = ir2[2];
                ir2[2] = (v199_data + (v183_data * v197_bc));
                float v205_data = ir2[3];
                ir2[3] = (v205_data + (v183_data * v203_bc));
                float v211_data = ir2[4];
                ir2[4] = (v211_data + (v183_data * v209_bc));
                float v217_data = ir2[5];
                ir2[5] = (v217_data + (v183_data * v215_bc));
                float v223_data = ir2[6];
                ir2[6] = (v223_data + (v183_data * v221_bc));
                float v229_data = ir2[7];
                ir2[7] = (v229_data + (v183_data * v227_bc));
              }
              float v233_bc = sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v239_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v245_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v251_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v257_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v263_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v269_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              float v275_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4));
              if (v18_g) {
                float v231_data = r0[4];
                float v235_data = ir2[0];
                ir2[0] = (v235_data + (v231_data * v233_bc));
                float v241_data = ir2[1];
                ir2[1] = (v241_data + (v231_data * v239_bc));
                float v247_data = ir2[2];
                ir2[2] = (v247_data + (v231_data * v245_bc));
                float v253_data = ir2[3];
                ir2[3] = (v253_data + (v231_data * v251_bc));
                float v259_data = ir2[4];
                ir2[4] = (v259_data + (v231_data * v257_bc));
                float v265_data = ir2[5];
                ir2[5] = (v265_data + (v231_data * v263_bc));
                float v271_data = ir2[6];
                ir2[6] = (v271_data + (v231_data * v269_bc));
                float v277_data = ir2[7];
                ir2[7] = (v277_data + (v231_data * v275_bc));
              }
              float v281_bc = sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v287_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v293_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v299_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v305_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v311_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v317_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              float v323_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5));
              if (v18_g) {
                float v279_data = r0[5];
                float v283_data = ir2[0];
                ir2[0] = (v283_data + (v279_data * v281_bc));
                float v289_data = ir2[1];
                ir2[1] = (v289_data + (v279_data * v287_bc));
                float v295_data = ir2[2];
                ir2[2] = (v295_data + (v279_data * v293_bc));
                float v301_data = ir2[3];
                ir2[3] = (v301_data + (v279_data * v299_bc));
                float v307_data = ir2[4];
                ir2[4] = (v307_data + (v279_data * v305_bc));
                float v313_data = ir2[5];
                ir2[5] = (v313_data + (v279_data * v311_bc));
                float v319_data = ir2[6];
                ir2[6] = (v319_data + (v279_data * v317_bc));
                float v325_data = ir2[7];
                ir2[7] = (v325_data + (v279_data * v323_bc));
              }
              float v329_bc = sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v335_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v341_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v347_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v353_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v359_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v365_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              float v371_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6));
              if (v18_g) {
                float v327_data = r0[6];
                float v331_data = ir2[0];
                ir2[0] = (v331_data + (v327_data * v329_bc));
                float v337_data = ir2[1];
                ir2[1] = (v337_data + (v327_data * v335_bc));
                float v343_data = ir2[2];
                ir2[2] = (v343_data + (v327_data * v341_bc));
                float v349_data = ir2[3];
                ir2[3] = (v349_data + (v327_data * v347_bc));
                float v355_data = ir2[4];
                ir2[4] = (v355_data + (v327_data * v353_bc));
                float v361_data = ir2[5];
                ir2[5] = (v361_data + (v327_data * v359_bc));
                float v367_data = ir2[6];
                ir2[6] = (v367_data + (v327_data * v365_bc));
                float v373_data = ir2[7];
                ir2[7] = (v373_data + (v327_data * v371_bc));
              }
              float v377_bc = sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v383_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v389_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v395_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v401_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v407_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v413_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              float v419_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7));
              if (v18_g) {
                float v375_data = r0[7];
                float v379_data = ir2[0];
                ir2[0] = (v379_data + (v375_data * v377_bc));
                float v385_data = ir2[1];
                ir2[1] = (v385_data + (v375_data * v383_bc));
                float v391_data = ir2[2];
                ir2[2] = (v391_data + (v375_data * v389_bc));
                float v397_data = ir2[3];
                ir2[3] = (v397_data + (v375_data * v395_bc));
                float v403_data = ir2[4];
                ir2[4] = (v403_data + (v375_data * v401_bc));
                float v409_data = ir2[5];
                ir2[5] = (v409_data + (v375_data * v407_bc));
                float v415_data = ir2[6];
                ir2[6] = (v415_data + (v375_data * v413_bc));
                float v421_data = ir2[7];
                ir2[7] = (v421_data + (v375_data * v419_bc));
              }
              float v425_bc = sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v431_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v437_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v443_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v449_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v455_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v461_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              float v467_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8));
              if (v18_g) {
                float v423_data = r0[8];
                float v427_data = ir2[0];
                ir2[0] = (v427_data + (v423_data * v425_bc));
                float v433_data = ir2[1];
                ir2[1] = (v433_data + (v423_data * v431_bc));
                float v439_data = ir2[2];
                ir2[2] = (v439_data + (v423_data * v437_bc));
                float v445_data = ir2[3];
                ir2[3] = (v445_data + (v423_data * v443_bc));
                float v451_data = ir2[4];
                ir2[4] = (v451_data + (v423_data * v449_bc));
                float v457_data = ir2[5];
                ir2[5] = (v457_data + (v423_data * v455_bc));
                float v463_data = ir2[6];
                ir2[6] = (v463_data + (v423_data * v461_bc));
                float v469_data = ir2[7];
                ir2[7] = (v469_data + (v423_data * v467_bc));
              }
              float v473_bc = sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v479_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v485_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v491_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v497_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v503_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v509_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              float v515_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9));
              if (v18_g) {
                float v471_data = r0[9];
                float v475_data = ir2[0];
                ir2[0] = (v475_data + (v471_data * v473_bc));
                float v481_data = ir2[1];
                ir2[1] = (v481_data + (v471_data * v479_bc));
                float v487_data = ir2[2];
                ir2[2] = (v487_data + (v471_data * v485_bc));
                float v493_data = ir2[3];
                ir2[3] = (v493_data + (v471_data * v491_bc));
                float v499_data = ir2[4];
                ir2[4] = (v499_data + (v471_data * v497_bc));
                float v505_data = ir2[5];
                ir2[5] = (v505_data + (v471_data * v503_bc));
                float v511_data = ir2[6];
                ir2[6] = (v511_data + (v471_data * v509_bc));
                float v517_data = ir2[7];
                ir2[7] = (v517_data + (v471_data * v515_bc));
              }
              float v521_bc = sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v527_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v533_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v539_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v545_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v551_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v557_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              float v563_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10));
              if (v18_g) {
                float v519_data = r0[10];
                float v523_data = ir2[0];
                ir2[0] = (v523_data + (v519_data * v521_bc));
                float v529_data = ir2[1];
                ir2[1] = (v529_data + (v519_data * v527_bc));
                float v535_data = ir2[2];
                ir2[2] = (v535_data + (v519_data * v533_bc));
                float v541_data = ir2[3];
                ir2[3] = (v541_data + (v519_data * v539_bc));
                float v547_data = ir2[4];
                ir2[4] = (v547_data + (v519_data * v545_bc));
                float v553_data = ir2[5];
                ir2[5] = (v553_data + (v519_data * v551_bc));
                float v559_data = ir2[6];
                ir2[6] = (v559_data + (v519_data * v557_bc));
                float v565_data = ir2[7];
                ir2[7] = (v565_data + (v519_data * v563_bc));
              }
              float v569_bc = sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v575_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v581_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v587_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v593_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v599_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v605_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              float v611_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11));
              if (v18_g) {
                float v567_data = r0[11];
                float v571_data = ir2[0];
                ir2[0] = (v571_data + (v567_data * v569_bc));
                float v577_data = ir2[1];
                ir2[1] = (v577_data + (v567_data * v575_bc));
                float v583_data = ir2[2];
                ir2[2] = (v583_data + (v567_data * v581_bc));
                float v589_data = ir2[3];
                ir2[3] = (v589_data + (v567_data * v587_bc));
                float v595_data = ir2[4];
                ir2[4] = (v595_data + (v567_data * v593_bc));
                float v601_data = ir2[5];
                ir2[5] = (v601_data + (v567_data * v599_bc));
                float v607_data = ir2[6];
                ir2[6] = (v607_data + (v567_data * v605_bc));
                float v613_data = ir2[7];
                ir2[7] = (v613_data + (v567_data * v611_bc));
              }
              float v617_bc = sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v623_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v629_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v635_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v641_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v647_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v653_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              float v659_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12));
              if (v18_g) {
                float v615_data = r0[12];
                float v619_data = ir2[0];
                ir2[0] = (v619_data + (v615_data * v617_bc));
                float v625_data = ir2[1];
                ir2[1] = (v625_data + (v615_data * v623_bc));
                float v631_data = ir2[2];
                ir2[2] = (v631_data + (v615_data * v629_bc));
                float v637_data = ir2[3];
                ir2[3] = (v637_data + (v615_data * v635_bc));
                float v643_data = ir2[4];
                ir2[4] = (v643_data + (v615_data * v641_bc));
                float v649_data = ir2[5];
                ir2[5] = (v649_data + (v615_data * v647_bc));
                float v655_data = ir2[6];
                ir2[6] = (v655_data + (v615_data * v653_bc));
                float v661_data = ir2[7];
                ir2[7] = (v661_data + (v615_data * v659_bc));
              }
              float v665_bc = sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v671_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v677_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v683_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v689_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v695_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v701_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              float v707_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13));
              if (v18_g) {
                float v663_data = r0[13];
                float v667_data = ir2[0];
                ir2[0] = (v667_data + (v663_data * v665_bc));
                float v673_data = ir2[1];
                ir2[1] = (v673_data + (v663_data * v671_bc));
                float v679_data = ir2[2];
                ir2[2] = (v679_data + (v663_data * v677_bc));
                float v685_data = ir2[3];
                ir2[3] = (v685_data + (v663_data * v683_bc));
                float v691_data = ir2[4];
                ir2[4] = (v691_data + (v663_data * v689_bc));
                float v697_data = ir2[5];
                ir2[5] = (v697_data + (v663_data * v695_bc));
                float v703_data = ir2[6];
                ir2[6] = (v703_data + (v663_data * v701_bc));
                float v709_data = ir2[7];
                ir2[7] = (v709_data + (v663_data * v707_bc));
              }
              float v713_bc = sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v719_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v725_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v731_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v737_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v743_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v749_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              float v755_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14));
              if (v18_g) {
                float v711_data = r0[14];
                float v715_data = ir2[0];
                ir2[0] = (v715_data + (v711_data * v713_bc));
                float v721_data = ir2[1];
                ir2[1] = (v721_data + (v711_data * v719_bc));
                float v727_data = ir2[2];
                ir2[2] = (v727_data + (v711_data * v725_bc));
                float v733_data = ir2[3];
                ir2[3] = (v733_data + (v711_data * v731_bc));
                float v739_data = ir2[4];
                ir2[4] = (v739_data + (v711_data * v737_bc));
                float v745_data = ir2[5];
                ir2[5] = (v745_data + (v711_data * v743_bc));
                float v751_data = ir2[6];
                ir2[6] = (v751_data + (v711_data * v749_bc));
                float v757_data = ir2[7];
                ir2[7] = (v757_data + (v711_data * v755_bc));
              }
              float v761_bc = sycl::select_from_group(item.get_sub_group(), v40_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              float v767_bc = sycl::select_from_group(item.get_sub_group(), v46_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              float v773_bc = sycl::select_from_group(item.get_sub_group(), v52_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              float v779_bc = sycl::select_from_group(item.get_sub_group(), v58_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              float v785_bc = sycl::select_from_group(item.get_sub_group(), v64_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              float v791_bc = sycl::select_from_group(item.get_sub_group(), v70_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              float v797_bc = sycl::select_from_group(item.get_sub_group(), v76_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              float v803_bc = sycl::select_from_group(item.get_sub_group(), v82_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15));
              if (v18_g) {
                float v759_data = r0[15];
                float v763_data = ir2[0];
                ir2[0] = (v763_data + (v759_data * v761_bc));
                float v769_data = ir2[1];
                ir2[1] = (v769_data + (v759_data * v767_bc));
                float v775_data = ir2[2];
                ir2[2] = (v775_data + (v759_data * v773_bc));
                float v781_data = ir2[3];
                ir2[3] = (v781_data + (v759_data * v779_bc));
                float v787_data = ir2[4];
                ir2[4] = (v787_data + (v759_data * v785_bc));
                float v793_data = ir2[5];
                ir2[5] = (v793_data + (v759_data * v791_bc));
                float v799_data = ir2[6];
                ir2[6] = (v799_data + (v759_data * v797_bc));
                float v805_data = ir2[7];
                ir2[7] = (v805_data + (v759_data * v803_bc));
              }
              if (v18_g) {
                #pragma unroll
                for (int32_t v807_n1 = 0; v807_n1 < 8; ++v807_n1) {
                  float v809_data = ir2[v807_n1];
                  r2[v807_n1] = v809_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v18_g) {
                int32_t v816_a = ((v17_lead + 16_i32) + -12) - 4;
                #pragma unroll
                for (int32_t v810_i1 = 0; v810_i1 < 8; ++v810_i1) {
                  float v812_data = r2[v810_i1];
                  glb_m0[(v816_a + (v810_i1 * 12))] = v812_data;
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

